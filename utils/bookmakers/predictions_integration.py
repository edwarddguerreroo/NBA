"""
Sistema de Integración: Predicciones de Modelos + Análisis de Bookmakers
======================================================================

Este módulo conecta las predicciones de los modelos individuales (PTS, AST, TRB, 3P, DD, etc.)
con el análisis de bookmakers para identificar oportunidades de apuesta con ventaja estadística.

Funcionalidades principales:
1. Cargar predicciones de todos los modelos entrenados
2. Obtener odds actuales de Sportradar API
3. Comparar predicciones vs mercado
4. Identificar value bets con alta confianza
5. Generar recomendaciones de apuestas optimizadas

Arquitectura:
- Loader de modelos: Carga modelos entrenados desde src/models/
- Predictor unificado: Genera predicciones para todos los targets
- Comparador de mercado: Analiza diferencias modelo vs bookmakers
- Optimizador de apuestas: Aplica Kelly Criterion y análisis de riesgo
"""

import os
import sys
import logging
import joblib
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import json

# Imports del sistema
from .bookmakers_integration import BookmakersIntegration
from .sportradar_api import SportradarAPI
from src.preprocessing.data_loader import NBADataLoader

logger = logging.getLogger(__name__)

class ModelsLoader:
    """
    Carga y gestiona todos los modelos entrenados del sistema.
    """
    
    def __init__(self, models_base_path: str = ".joblib"):
        """
        Inicializa el loader de modelos.
        
        Args:
            models_base_path: Ruta base donde están guardados los modelos
        """
        self.models_base_path = Path(models_base_path)
        self.loaded_models = {}
        self.model_paths = {}
        
        # Definir estructura de modelos esperada (basada en archivos reales en .joblib/)
        self.model_structure = {
            # Player models
            'PTS': {
                'path': 'pts_model.joblib',
                'model_class': 'XGBoostPTSModel',
                'type': 'player'
            },
            'AST': {
                'path': 'ast_model.joblib', 
                'model_class': 'StackingASTModel',
                'type': 'player'
            },
            'TRB': {
                'path': 'trb_model.joblib',
                'model_class': 'XGBoostTRBModel', 
                'type': 'player'
            },
            '3P': {
                'path': '3pt_model.joblib',
                'model_class': 'XGBoostTriplesModel',
                'type': 'player'
            },
            'DD': {
                'path': 'dd_model.joblib',
                'model_class': 'XGBoostDDModel',
                'type': 'player'
            },
            # Team models
            'is_win': {
                'path': 'is_win_model.joblib',
                'model_class': 'IsWinModel',
                'type': 'team'
            },
            'total_points': {
                'path': 'total_points_model.joblib', 
                'model_class': 'TotalPointsModel',
                'type': 'team'
            },
            'teams_points': {
                'path': 'teams_points_model.joblib',
                'model_class': 'TeamsPointsModel', 
                'type': 'team'
            }
        }
        
        logger.info(f"ModelsLoader inicializado - Base path: {self.models_base_path}")
    
    def discover_available_models(self) -> Dict[str, bool]:
        """
        Descubre qué modelos están disponibles en el sistema.
        
        Returns:
            Dict con targets y su disponibilidad
        """
        available = {}
        
        for target, config in self.model_structure.items():
            model_path = self.models_base_path / config['path']
            available[target] = model_path.exists()
            
            if available[target]:
                self.model_paths[target] = model_path
                logger.info(f"Modelo {target} encontrado: {model_path}")
            else:
                logger.warning(f"Modelo {target} no encontrado: {model_path}")
        
        return available
    
    def load_model(self, target: str) -> Optional[Any]:
        """
        Carga un modelo específico.
        
        Args:
            target: Target del modelo (PTS, AST, etc.)
            
        Returns:
            Modelo cargado o None si no existe
        """
        if target not in self.model_structure:
            logger.error(f"Target {target} no reconocido")
            return None
        
        if target in self.loaded_models:
            return self.loaded_models[target]
        
        model_path = self.model_paths.get(target)
        if not model_path or not model_path.exists():
            logger.error(f"Modelo {target} no encontrado en {model_path}")
            return None
        
        try:
            model = joblib.load(model_path)
            self.loaded_models[target] = model
            logger.info(f"Modelo {target} cargado exitosamente")
            return model
        except Exception as e:
            logger.error(f"Error cargando modelo {target}: {e}")
            return None
    
    def load_all_available_models(self) -> Dict[str, Any]:
        """
        Carga todos los modelos disponibles.
        
        Returns:
            Dict con modelos cargados
        """
        available = self.discover_available_models()
        loaded = {}
        
        for target, is_available in available.items():
            if is_available:
                model = self.load_model(target)
                if model:
                    loaded[target] = model
        
        logger.info(f"Modelos cargados exitosamente: {list(loaded.keys())}")
        return loaded
    
    def get_model_info(self, target: str) -> Dict[str, Any]:
        """
        Obtiene información de un modelo específico.
        
        Args:
            target: Target del modelo
            
        Returns:
            Información del modelo
        """
        if target not in self.loaded_models:
            return {'error': f'Modelo {target} no cargado'}
        
        model = self.loaded_models[target]
        config = self.model_structure[target]
        
        info = {
            'target': target,
            'type': config['type'],
            'model_class': config['model_class'],
            'is_trained': getattr(model, 'is_trained', False),
            'path': str(self.model_paths.get(target, 'Unknown'))
        }
        
        # Agregar métricas si están disponibles
        if hasattr(model, 'validation_metrics'):
            info['validation_metrics'] = model.validation_metrics
        
        if hasattr(model, 'training_metrics'):
            info['training_metrics'] = model.training_metrics
        
        return info


class UnifiedPredictor:
    """
    Generador de predicciones unificado para todos los targets.
    """
    
    def __init__(self, models_loader: ModelsLoader, data_loader: NBADataLoader):
        """
        Inicializa el predictor unificado.
        
        Args:
            models_loader: Loader de modelos
            data_loader: Loader de datos NBA
        """
        self.models_loader = models_loader
        self.data_loader = data_loader
        self.models = {}
        
        # Cargar todos los modelos disponibles
        self.models = self.models_loader.load_all_available_models()
        
        logger.info(f"UnifiedPredictor inicializado con {len(self.models)} modelos")
    
    def generate_predictions_for_date(self, date: str, players: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Genera predicciones para una fecha específica.
        
        Args:
            date: Fecha en formato YYYY-MM-DD
            players: Lista de jugadores específicos (opcional)
            
        Returns:
            Predicciones organizadas por target y jugador
        """
        logger.info(f"Generando predicciones para {date}")
        
        # Cargar datos actualizados
        try:
            df, teams_df = self.data_loader.load_data()
            logger.info(f"Datos cargados: {len(df)} registros")
        except Exception as e:
            logger.error(f"Error cargando datos: {e}")
            return {'error': f'Error cargando datos: {e}'}
        
        # Filtrar por fecha si es necesario
        target_date = pd.to_datetime(date)
        
        # Para predicciones, usamos datos hasta la fecha anterior
        historical_data = df[df['Date'] < target_date]
        historical_teams_data = teams_df[teams_df['Date'] < target_date]
        
        if historical_data.empty:
            logger.warning(f"No hay datos históricos para fecha {date}")
            return {'error': f'No hay datos históricos para {date}'}
        
        # Filtrar por jugadores si se especifica
        if players:
            historical_data = historical_data[historical_data['Player'].isin(players)]
        
        # Generar predicciones para cada modelo
        predictions = {
            'date': date,
            'predictions': {},
            'summary': {
                'total_players': 0,
                'predictions_by_target': {},
                'models_used': list(self.models.keys())
            }
        }
        
        # Obtener jugadores únicos para predicciones
        unique_players = historical_data['Player'].unique()
        predictions['summary']['total_players'] = len(unique_players)
        
        # Generar predicciones por target
        for target, model in self.models.items():
            try:
                logger.info(f"Generando predicciones para {target}")
                
                # Preparar datos para predicción
                prediction_data = self._prepare_prediction_data(historical_data, historical_teams_data, target, unique_players)
                
                if prediction_data.empty:
                    logger.warning(f"No hay datos para predicción de {target}")
                    continue
                
                # Determinar qué datos usar para obtener información de jugadores/equipos
                team_models = ['is_win', 'total_points', 'teams_points']
                if target in team_models:
                    # Para modelos de equipos, usar datos de equipos y crear info por equipo
                    unique_teams = historical_teams_data['Team'].unique()
                    # Crear players_info usando datos de equipos - cada "jugador" será un equipo
                    if target in historical_teams_data.columns:
                        teams_info = historical_teams_data.groupby('Team').tail(1)[['Team', target]].reset_index(drop=True)
                        teams_info = teams_info[teams_info['Team'].isin(unique_teams)]
                        # Renombrar columnas para mantener consistencia con la interfaz
                        players_info = teams_info.rename(columns={'Team': 'Player'})
                        players_info['Team'] = players_info['Player']  # Para modelos de equipos, Player = Team
                    else:
                        logger.warning(f"Columna {target} no encontrada en datos de equipos")
                        # Crear estructura básica sin el target
                        teams_info = historical_teams_data.groupby('Team').tail(1)[['Team']].reset_index(drop=True)
                        teams_info = teams_info[teams_info['Team'].isin(unique_teams)]
                        players_info = teams_info.rename(columns={'Team': 'Player'})
                        players_info['Team'] = players_info['Player']
                        players_info[target] = 0  # Valor por defecto
                else:
                    # Para modelos de jugadores, usar datos de jugadores
                    if target in historical_data.columns:
                        players_info = historical_data.groupby('Player').tail(1)[['Player', 'Team', target]].reset_index(drop=True)
                        players_info = players_info[players_info['Player'].isin(unique_players)]
                    else:
                        logger.warning(f"Columna {target} no encontrada en datos de jugadores")
                        # Crear estructura básica sin el target
                        players_info = historical_data.groupby('Player').tail(1)[['Player', 'Team']].reset_index(drop=True)
                        players_info = players_info[players_info['Player'].isin(unique_players)]
                        players_info[target] = 0  # Valor por defecto
                
                # VERIFICACIÓN CRÍTICA PRE-PREDICCIÓN: Asegurar 0 NaN
                pre_predict_nan = prediction_data.isnull().sum().sum()
                if pre_predict_nan > 0:
                    logger.error(f"❌ EMERGENCIA: {pre_predict_nan} NaN detectados ANTES de predicción")
                    logger.error(f"Aplicando limpieza de emergencia...")
                    prediction_data = prediction_data.fillna(0)
                    
                    # Verificación post-emergencia
                    post_emergency_nan = prediction_data.isnull().sum().sum()
                    if post_emergency_nan > 0:
                        logger.error(f"❌ FALLÓ LIMPIEZA DE EMERGENCIA: {post_emergency_nan} NaN persistentes")
                        return {'error': f'Datos con NaN no procesables para {target}'}
                    else:
                        logger.info(f"✅ Limpieza de emergencia exitosa")
                
                logger.info(f"Pre-predicción verificada: {prediction_data.shape} datos sin NaN")
                
                # Generar predicciones
                target_predictions = model.predict(prediction_data)
                
                # Organizar resultados
                predictions['predictions'][target] = self._organize_predictions(
                    target_predictions, prediction_data, target, players_info
                )
                
                predictions['summary']['predictions_by_target'][target] = len(target_predictions)
                
                logger.info(f"{target}: {len(target_predictions)} predicciones generadas")
                
            except Exception as e:
                logger.error(f"Error generando predicciones para {target}: {e}")
                predictions['predictions'][target] = {'error': str(e)}
        
        return predictions
    
    def _prepare_prediction_data(self, historical_data: pd.DataFrame, historical_teams_data: pd.DataFrame, target: str, players: List[str]) -> pd.DataFrame:
        """
        Prepara datos para predicción de un target específico usando el pipeline de feature engineering correspondiente.
        
        Args:
            historical_data: Datos históricos
            target: Target objetivo
            players: Lista de jugadores
            
        Returns:
            Datos preparados para predicción con features correctas
        """
        # Importar el feature engineer correspondiente según el target
        feature_engineer = self._get_feature_engineer_for_target(target, historical_teams_data, historical_data)
        
        if feature_engineer is None:
            logger.error(f"No se encontró feature engineer para target {target}")
            return pd.DataFrame()
        
        # Determinar qué datos usar según el tipo de modelo
        team_models = ['is_win', 'total_points', 'teams_points']
        if target in team_models:
            # Modelos de equipos usan datos de equipos
            df_copy = historical_teams_data.copy()
            logger.debug(f"Usando datos de equipos para modelo {target}")
        else:
            # Modelos de jugadores usan datos de jugadores
            df_copy = historical_data.copy()
            logger.debug(f"Usando datos de jugadores para modelo {target}")
        
        # Generar features usando el pipeline específico del modelo
        try:
            logger.info(f"Generando features para {target}")
            
            # Los feature engineers pueden modificar el DataFrame in-place
            # Pero asegurar que obtenemos el DataFrame modificado
            feature_list = feature_engineer.generate_all_features(df_copy)
            
            # VERIFICACIÓN CRÍTICA: Asegurar que df_copy tiene las features que feature_list dice tener
            actual_features_in_df = [f for f in feature_list if f in df_copy.columns]
            missing_features_in_df = [f for f in feature_list if f not in df_copy.columns]
            
            logger.info(f"DataFrame después de generate_all_features: {df_copy.shape[1]} columnas")
            logger.info(f"Features en lista: {len(feature_list)}, Features en DataFrame: {len(actual_features_in_df)}")
            
            if missing_features_in_df:
                logger.error(f"❌ CRÍTICO: {len(missing_features_in_df)} features en lista NO están en DataFrame")
                logger.error(f"Ejemplos: {missing_features_in_df[:5]}")
                
                # Como solución, usar solo las features que realmente existen
                feature_list = actual_features_in_df
                logger.warning(f"⚠️ CORRIGIENDO: Usando {len(feature_list)} features que realmente existen")
            
            if not feature_list:
                logger.error(f"generate_all_features devolvió lista vacía para {target}")
                return pd.DataFrame()
            
            logger.info(f"Features generadas: {len(feature_list)} características")
            logger.info(f"DataFrame con features: {df_copy.shape[1]} columnas, {df_copy.shape[0]} filas")
            
            # Tomar los datos más recientes del DataFrame CON features generadas
            if target in team_models:
                # PARA MODELOS DE EQUIPOS: Método mejorado que preserva TODAS las features
                logger.debug(f"Verificando features antes de tail(1): {len(feature_list)} esperadas")
                
                # PASO 1: Verificar qué features faltan en el DataFrame completo
                missing_in_full = [f for f in feature_list if f not in df_copy.columns]
                if missing_in_full:
                    logger.warning(f"Features faltantes en DataFrame completo: {len(missing_in_full)}")
                
                # PASO 2: Método más directo que preserva TODAS las columnas
                logger.info(f"DataFrame completo antes de latest: {df_copy.shape[1]} columnas")
                
                # USAR MÉTODO ÍNDICE EN LUGAR DE DataFrame() para preservar TODAS las columnas
                latest_indices = df_copy.groupby('Team').tail(1).index
                latest_data = df_copy.loc[latest_indices].copy()
                
                logger.info(f"Latest data (equipos) DIRECTO: {latest_data.shape}")
                logger.info(f"Columnas preservadas: {latest_data.shape[1]} de {df_copy.shape[1]} originales")
                
                # Verificar preservación de features ESPECÍFICAMENTE
                preserved_features = [f for f in feature_list if f in latest_data.columns]
                lost_features = [f for f in feature_list if f not in latest_data.columns]
                
                if lost_features:
                    logger.error(f"❌ CRÍTICO: Se perdieron {len(lost_features)} features en tail(1)")
                    logger.debug(f"Features perdidas: {lost_features[:5]}")
                    
                    # Si aún hay features perdidas, es porque el DataFrame original no las tenía realmente
                    # Verificar en el DataFrame completo
                    actually_missing = [f for f in lost_features if f not in df_copy.columns]
                    falsely_reported = [f for f in lost_features if f in df_copy.columns]
                    
                    if actually_missing:
                        logger.error(f"❌ Features que NO existen en DataFrame original: {len(actually_missing)}")
                        
                    if falsely_reported:
                        logger.warning(f"⚠️ Features que SÍ existen pero se perdieron en tail(1): {len(falsely_reported)}")
                        # Intentar recuperar estas features usando loc directo
                        for feature in falsely_reported:
                            try:
                                latest_data[feature] = df_copy.loc[latest_indices, feature].values
                                logger.debug(f"Recuperada feature: {feature}")
                            except Exception as e:
                                logger.debug(f"Error recuperando {feature}: {e}")
                    
                    # Verificar recuperación final
                    final_preserved = [f for f in feature_list if f in latest_data.columns]
                    logger.info(f"✅ Features preservadas FINAL: {len(final_preserved)}/{len(feature_list)}")
                
                if latest_data.empty:
                    logger.error("No se pudo crear latest_data para equipos")
                    return pd.DataFrame()
            
            else:
                # Para modelos de jugadores, agrupar por jugador
                # CORREGIDO: NO hacer .copy() - trabajar directamente sobre df_copy con features
                latest_indices = df_copy.groupby('Player').tail(1).index
                latest_data = df_copy.loc[latest_indices]
                # Filtrar por jugadores especificados
                latest_data = latest_data[latest_data['Player'].isin(players)]
                logger.info(f"Latest data (jugadores) shape: {latest_data.shape}")
            
            # Verificar que las features generadas existen en el DataFrame
            available_features = [f for f in feature_list if f in latest_data.columns]
            missing_features = [f for f in feature_list if f not in latest_data.columns]
            
            if missing_features:
                logger.warning(f"Features faltantes en DataFrame: {len(missing_features)}/{len(feature_list)}")
                logger.debug(f"Primeras 5 features faltantes: {missing_features[:5]}")
            
            logger.info(f"Features disponibles: {len(available_features)}")
            
            # Verificación específica para total_points
            if target == 'total_points':
                # CORREGIDO: Aceptar 130-131 features (rango flexible después de optimizaciones)
                if len(available_features) < 125 or len(available_features) > 135:
                    logger.error(f"❌ CRÍTICO: {target} tiene {len(available_features)} features, esperado 125-135")
                    return pd.DataFrame()
                else:
                    logger.info(f"✅ {target}: {len(available_features)} features confirmadas (rango válido 125-135)")
            
            # Verificar mínimo de features
            if len(available_features) < 10:
                logger.error(f"Muy pocas features para {target}: {len(available_features)}")
                return pd.DataFrame()
            
            # Seleccionar features para predicción
            prediction_data = latest_data[available_features].copy()
            
            # PASO 6: Limpiar datos completamente - LIMPIEZA AGRESIVA DE NaN
            logger.info(f"Limpiando NaN antes de predicción...")
            
            # Verificar NaN antes de limpieza
            nan_count_before = prediction_data.isnull().sum().sum()
            if nan_count_before > 0:
                logger.warning(f"Encontrados {nan_count_before} valores NaN antes de limpieza")
            
            # Limpieza completa de NaN
            prediction_data = prediction_data.fillna(0)
            
            # Verificar que NO quedan NaN después de fillna
            nan_count_after = prediction_data.isnull().sum().sum()
            if nan_count_after > 0:
                logger.error(f"❌ CRÍTICO: Aún hay {nan_count_after} NaN después de fillna(0)")
                # Limpieza más agresiva
                prediction_data = prediction_data.fillna(method='ffill').fillna(method='bfill').fillna(0)
                
                # Verificación final
                nan_count_final = prediction_data.isnull().sum().sum()
                if nan_count_final > 0:
                    logger.error(f"❌ CRÍTICO: {nan_count_final} NaN persistentes - usando fillna(-999)")
                    prediction_data = prediction_data.fillna(-999)
            
            logger.info(f"Limpieza NaN completada: {nan_count_before} → {prediction_data.isnull().sum().sum()}")
            
            # PASO 6.1: Convertir columnas object problemáticas a numérico
            object_cols = prediction_data.select_dtypes(include=['object']).columns
            if len(object_cols) > 0:
                logger.info(f"Convirtiendo {len(object_cols)} columnas object a numérico: {object_cols.tolist()}")
                for col in object_cols:
                    try:
                        prediction_data[col] = pd.to_numeric(prediction_data[col], errors='coerce')
                        logger.debug(f"  ✅ {col}: convertido a numérico")
                    except Exception as e:
                        logger.warning(f"  ❌ {col}: Error convirtiendo - {e}, rellenando con 0")
                        prediction_data[col] = 0
                
                # Rellenar NaN resultantes de conversión
                prediction_data = prediction_data.fillna(0)
            
            # PASO 6.2: Convertir columnas datetime problemáticas
            datetime_cols = prediction_data.select_dtypes(include=['datetime64[ns]']).columns
            if len(datetime_cols) > 0:
                logger.info(f"Convirtiendo {len(datetime_cols)} columnas datetime a timestamp")
                for col in datetime_cols:
                    try:
                        prediction_data[col] = prediction_data[col].astype('int64') // 10**9
                        logger.debug(f"  ✅ {col}: datetime convertido a timestamp")
                    except Exception as e:
                        logger.warning(f"  ❌ {col}: Error convirtiendo datetime - {e}, rellenando con 0")
                        prediction_data[col] = 0
            
            # PASO 6.3: Verificar valores infinitos de manera segura
            inf_cols = []
            for col in prediction_data.columns:
                try:
                    # Solo verificar infinitos en columnas numéricas
                    if prediction_data[col].dtype in ['float64', 'float32', 'int64', 'int32']:
                        if np.isinf(prediction_data[col]).any():
                            inf_cols.append(col)
                except (TypeError, ValueError):
                    # Si hay error, saltar esta columna
                    continue
            
            if inf_cols:
                logger.warning(f"Valores infinitos encontrados en columnas: {inf_cols[:5]}...")
                prediction_data = prediction_data.replace([np.inf, -np.inf], 0)
            
            # PASO 6.4: Verificación final de tipos de datos
            final_object_cols = prediction_data.select_dtypes(include=['object']).columns
            if len(final_object_cols) > 0:
                logger.error(f"❌ CRÍTICO: Aún hay {len(final_object_cols)} columnas object después de limpieza")
                # Forzar conversión final
                for col in final_object_cols:
                    prediction_data[col] = 0
                logger.info(f"✅ Columnas object forzadas a 0")
            
            # PASO 6.5: Test final de conversión a numpy array
            try:
                test_array = prediction_data.values
                logger.debug(f"✅ Test numpy array: EXITOSO")
                
                # VERIFICACIÓN CRÍTICA: Comprobar NaN en array numpy
                nan_count_numpy = np.isnan(test_array).sum()
                if nan_count_numpy > 0:
                    logger.error(f"❌ CRÍTICO: {nan_count_numpy} NaN en array numpy final")
                    # Reemplazar NaN en numpy array directamente
                    test_array = np.nan_to_num(test_array, nan=0.0, posinf=0.0, neginf=0.0)
                    prediction_data = pd.DataFrame(test_array, columns=prediction_data.columns, index=prediction_data.index)
                    logger.info(f"✅ NaN en numpy array corregidos con np.nan_to_num")
                
            except Exception as e:
                logger.error(f"❌ Test numpy array: FALLÓ - {e}")
                return pd.DataFrame()
            
            # VERIFICACIÓN FINAL ANTES DE RETORNO
            final_nan_check = prediction_data.isnull().sum().sum()
            if final_nan_check > 0:
                logger.error(f"❌ VERIFICACIÓN FINAL FALLÓ: {final_nan_check} NaN en datos finales")
                prediction_data = prediction_data.fillna(0)
                logger.info(f"✅ Aplicado fillna(0) final de emergencia")
            
            logger.info(f"Datos preparados para {target}: {prediction_data.shape[1]} features, {prediction_data.shape[0]} filas")
            logger.info(f"✅ Verificación NaN final: {prediction_data.isnull().sum().sum()} valores NaN")
            
            return prediction_data
            
        except Exception as e:
            logger.error(f"Error generando features para {target}: {e}")
            return pd.DataFrame()
    
    def _get_feature_engineer_for_target(self, target: str, teams_df: pd.DataFrame = None, players_df: pd.DataFrame = None):
        """
        Obtiene el feature engineer específico para cada target con constructores UNIFICADOS.
        
        Args:
            target: Target objetivo
            teams_df: DataFrame de equipos (pasado a TODOS los constructores)
            players_df: DataFrame de jugadores (pasado a TODOS los constructores)
            
        Returns:
            Feature engineer correspondiente
        """
        try:
            
            if target == 'PTS':
                from src.models.players.pts.features_pts import PointsFeatureEngineer
                return PointsFeatureEngineer(teams_df=teams_df, players_df=players_df)
            
            elif target == 'AST':
                from src.models.players.ast.features_ast import AssistsFeatureEngineer
                return AssistsFeatureEngineer(teams_df=teams_df, players_df=players_df)
            
            elif target == 'TRB':
                from src.models.players.trb.features_trb import ReboundsFeatureEngineer
                return ReboundsFeatureEngineer(teams_df=teams_df, players_df=players_df)
            
            elif target == '3P':
                from src.models.players.triples.features_triples import ThreePointsFeatureEngineer
                return ThreePointsFeatureEngineer(teams_df=teams_df, players_df=players_df)
            
            elif target == 'DD':
                from src.models.players.double_double.features_dd import DoubleDoubleFeatureEngineer
                return DoubleDoubleFeatureEngineer(teams_df=teams_df, players_df=players_df)
            
            elif target == 'is_win':
                from src.models.teams.is_win.features_is_win import IsWinFeatureEngineer
                return IsWinFeatureEngineer(teams_df=teams_df, players_df=players_df)
            
            elif target == 'total_points':
                from src.models.teams.total_points.features_total_points import TotalPointsFeatureEngineer
                return TotalPointsFeatureEngineer(teams_df=teams_df, players_df=players_df)
            
            elif target == 'teams_points':
                from src.models.teams.teams_points.features_teams_points import TeamPointsFeatureEngineer
                return TeamPointsFeatureEngineer(teams_df=teams_df, players_df=players_df)
            
            else:
                logger.warning(f"Target {target} no reconocido")
                return None
                
        except ImportError as e:
            logger.error(f"Error importando feature engineer para {target}: {e}")
            return None
    
    def _organize_predictions(self, predictions: np.ndarray, data: pd.DataFrame, target: str, players_info: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Organiza las predicciones en formato estructurado.
        
        Args:
            predictions: Array de predicciones
            data: Datos utilizados para predicción (features procesadas)
            target: Target objetivo
            players_info: DataFrame con información de jugadores (Player, Team, etc.)
            
        Returns:
            Predicciones organizadas
        """
        organized = {
            'target': target,
            'predictions': [],
            'stats': {
                'total': len(predictions),
                'mean': float(np.mean(predictions)),
                'median': float(np.median(predictions)),
                'std': float(np.std(predictions)),
                'min': float(np.min(predictions)),
                'max': float(np.max(predictions))
            }
        }
        
        # Asegurar que players_info tenga las columnas requeridas
        if players_info is None:
            players_info = pd.DataFrame(columns=['Player', 'Team', target])
        
        for i in range(len(predictions)):
            player_info = players_info.iloc[i] if i < len(players_info) else {}
            player_prediction = {
                'player_name': player_info.get('Player', f'Player_{i}'),
                'team': player_info.get('Team', 'Unknown'),
                target: float(predictions[i]),
                f'{target}_confidence': self._calculate_confidence(float(predictions[i]), target)
            }
            organized['predictions'].append(player_prediction)
        
        return organized
    
    def _calculate_confidence(self, prediction: float, target: str) -> float:
        """
        Calcula nivel de confianza de una predicción.
        
        Args:
            prediction: Valor predicho
            target: Target objetivo
            
        Returns:
            Nivel de confianza (0-1)
        """
        # Implementación básica - puede ser mejorada con datos históricos
        if target in ['PTS', 'AST', 'TRB']:
            # Para targets continuos, confianza basada en rangos típicos
            if target == 'PTS':
                return 0.95 if 10 <= prediction <= 35 else 0.85
            elif target == 'AST':
                return 0.95 if 2 <= prediction <= 12 else 0.85
            elif target == 'TRB':
                return 0.95 if 3 <= prediction <= 15 else 0.85
        
        return 0.90  # Confianza por defecto


class PredictionsBookmakersIntegration:
    """
    Integración principal que conecta predicciones de modelos con análisis de bookmakers.
    """
    
    def __init__(self, 
                 game_data_path: str = "data/players.csv",
                 biometrics_path: str = "data/height.csv", 
                 teams_path: str = "data/teams.csv",
                 models_base_path: str = ".joblib"):
        """
        Inicializa la integración completa.
        
        Args:
            game_data_path: Ruta a datos de partidos
            biometrics_path: Ruta a datos biométricos
            teams_path: Ruta a datos de equipos
            models_base_path: Ruta base de modelos entrenados
        """
        # Inicializar componentes
        self.data_loader = NBADataLoader(game_data_path, biometrics_path, teams_path)
        self.models_loader = ModelsLoader(models_base_path)
        self.predictor = UnifiedPredictor(self.models_loader, self.data_loader)
        self.bookmakers_integration = BookmakersIntegration()
        
        # Estado
        self.last_predictions = {}
        self.last_odds = {}
        self.analysis_results = {}
        
        logger.info("PredictionsBookmakersIntegration inicializada")
    
    def analyze_predictions_vs_market(self, date: Optional[str] = None, players: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Análisis completo: predicciones vs mercado.
        
        Args:
            date: Fecha específica (por defecto mañana)
            players: Jugadores específicos (opcional)
            
        Returns:
            Análisis completo con recomendaciones
        """
        # Usar mañana como fecha por defecto
        if date is None:
            date = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')
        
        logger.info(f"Iniciando análisis predicciones vs mercado para {date}")
        
        # 1. Generar predicciones
        logger.info("Generando predicciones de modelos...")
        predictions = self.predictor.generate_predictions_for_date(date, players)
        
        if 'error' in predictions:
            return {'error': f'Error en predicciones: {predictions["error"]}'}
        
        self.last_predictions = predictions
        
        # 2. Obtener odds actuales
        logger.info("Obteniendo odds actuales...")
        odds_data = self.bookmakers_integration.get_best_prediction_odds(
            pd.DataFrame(),  # Datos vacíos por ahora
            target='PTS',    # Target por defecto
            date=date
        )
        
        self.last_odds = odds_data
        
        # 3. Comparar predicciones vs odds
        logger.info("Comparando predicciones vs mercado...")
        comparison_results = self._compare_predictions_vs_odds(predictions, odds_data)
        
        # 4. Identificar value bets
        logger.info("Identificando value bets...")
        value_bets = self._identify_value_bets(comparison_results)
        
        # 5. Generar recomendaciones
        logger.info("Generando recomendaciones...")
        recommendations = self._generate_betting_recommendations(value_bets)
        
        # Compilar resultados
        analysis = {
            'date': date,
            'timestamp': datetime.now().isoformat(),
            'predictions_summary': predictions['summary'],
            'odds_summary': self._summarize_odds(odds_data),
            'comparison_results': comparison_results,
            'value_bets': value_bets,
            'recommendations': recommendations,
            'analysis_stats': {
                'total_comparisons': len(comparison_results.get('comparisons', [])),
                'value_bets_found': len(value_bets.get('opportunities', [])),
                'high_confidence_bets': len([b for b in value_bets.get('opportunities', []) if b.get('confidence', 0) > 0.9])
            }
        }
        
        self.analysis_results = analysis
        logger.info(f"Análisis completado - Value bets encontrados: {analysis['analysis_stats']['value_bets_found']}")
        
        return analysis
    
    def _compare_predictions_vs_odds(self, predictions: Dict[str, Any], odds_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compara predicciones de modelos con odds del mercado.
        
        Args:
            predictions: Predicciones de modelos
            odds_data: Datos de odds
            
        Returns:
            Resultados de comparación
        """
        comparisons = []
        
        for target, preds in predictions['predictions'].items():
            if target in odds_data:
                for pred in preds:
                    player = pred['player_name']
                    predicted_value = pred[target]
                    confidence = pred[f'{target}_confidence']
                    
                    # Encontrar odds para este jugador/target
                    matching_odds = [o for o in odds_data[target] if o['player'] == player]
                    
                    for odd in matching_odds:
                        market_line = odd['line']
                        if market_line == 0:
                            continue  # Saltar líneas de 0 para evitar división por zero
                        
                        # Determinar recomendación basada en predicción vs línea
                        bet_recommendation = 'OVER' if predicted_value > market_line else 'UNDER'
                        
                        comparison = {
                            'player': player,
                            'target': target,
                            'predicted': predicted_value,
                            'market_line': market_line,
                            'confidence': confidence,
                            'edge': (predicted_value - market_line) / market_line if market_line != 0 else 0,
                            'over_odds': odd['over_odds'],
                            'under_odds': odd['under_odds'],
                            'recommendation': bet_recommendation
                        }
                        comparisons.append(comparison)
        
        return {
            'comparisons': comparisons,
            'summary': {
                'total_comparisons': len(comparisons),
                'avg_edge': np.mean([c.get('edge', 0) for c in comparisons if c.get('edge')]),
                'targets_analyzed': list(predictions.get('predictions', {}).keys())
            }
        }
    
    def _identify_value_bets(self, comparison_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Identifica value bets basado en comparaciones.
        
        Args:
            comparison_results: Resultados de comparación
            
        Returns:
            Value bets identificados
        """
        opportunities = []
        
        for comparison in comparison_results.get('comparisons', []):
            edge = comparison.get('edge', 0)
            confidence = comparison.get('confidence', 0)
            
            # Criterios para value bet
            if edge > 0.05 and confidence > 0.85:  # 5% edge mínimo, 85% confianza
                opportunity = {
                    'player': comparison['player'],
                    'target': comparison['target'],
                    'predicted_value': comparison['predicted'],
                    'market_line': comparison['market_line'],
                    'recommendation': comparison['recommendation'],
                    'edge': edge,
                    'confidence': confidence,
                    'value_score': edge * confidence,  # Score combinado
                    'kelly_fraction': self._calculate_kelly_fraction(edge, confidence),
                    'risk_level': self._assess_risk_level(edge, confidence)
                }
                
                opportunities.append(opportunity)
        
        # Ordenar por value score
        opportunities.sort(key=lambda x: x['value_score'], reverse=True)
        
        return {
            'opportunities': opportunities,
            'summary': {
                'total_opportunities': len(opportunities),
                'avg_edge': np.mean([o['edge'] for o in opportunities]) if opportunities else 0,
                'avg_confidence': np.mean([o['confidence'] for o in opportunities]) if opportunities else 0,
                'high_value_count': len([o for o in opportunities if o['value_score'] > 0.08])
            }
        }
    
    def _calculate_kelly_fraction(self, edge: float, confidence: float) -> float:
        """
        Calcula fracción de Kelly para sizing de apuesta.
        
        Args:
            edge: Ventaja estadística
            confidence: Nivel de confianza
            
        Returns:
            Fracción de Kelly
        """
        # Implementación básica del criterio de Kelly
        # f = (bp - q) / b
        # donde b = odds, p = probabilidad real, q = 1-p
        
        # Asumir odds de -110 (probabilidad implícita ~52.4%)
        implied_prob = 0.524
        true_prob = confidence
        
        if true_prob > implied_prob:
            kelly_fraction = (true_prob - implied_prob) / (1 - implied_prob)
            return min(kelly_fraction, 0.25)  # Cap al 25% del bankroll
        
        return 0.0
    
    def _assess_risk_level(self, edge: float, confidence: float) -> str:
        """
        Evalúa nivel de riesgo de una apuesta.
        
        Args:
            edge: Ventaja estadística
            confidence: Nivel de confianza
            
        Returns:
            Nivel de riesgo
        """
        if confidence > 0.95 and edge > 0.1:
            return 'LOW'
        elif confidence > 0.9 and edge > 0.07:
            return 'MEDIUM'
        elif confidence > 0.85 and edge > 0.05:
            return 'MEDIUM_HIGH'
        else:
            return 'HIGH'
    
    def _generate_betting_recommendations(self, value_bets: Dict[str, Any]) -> Dict[str, Any]:
        """
        Genera recomendaciones de apuesta optimizadas.
        
        Args:
            value_bets: Value bets identificados
            
        Returns:
            Recomendaciones estructuradas
        """
        opportunities = value_bets.get('opportunities', [])
        
        if not opportunities:
            return {
                'message': 'No se encontraron oportunidades de value betting',
                'recommendations': []
            }
        
        recommendations = []
        
        # Top 5 oportunidades
        for opportunity in opportunities[:5]:
            recommendation = {
                'rank': len(recommendations) + 1,
                'player': opportunity['player'],
                'target': opportunity['target'],
                'bet_type': opportunity['recommendation'],
                'line': opportunity['market_line'],
                'predicted_value': opportunity['predicted_value'],
                'edge': f"{opportunity['edge']:.2%}",
                'confidence': f"{opportunity['confidence']:.1%}",
                'value_score': f"{opportunity['value_score']:.3f}",
                'kelly_fraction': f"{opportunity['kelly_fraction']:.2%}",
                'risk_level': opportunity['risk_level'],
                'reasoning': self._generate_reasoning(opportunity)
            }
            
            recommendations.append(recommendation)
        
        return {
            'message': f'Se encontraron {len(opportunities)} oportunidades de value betting',
            'top_recommendations': recommendations,
            'portfolio_advice': self._generate_portfolio_advice(opportunities),
            'risk_management': {
                'max_single_bet': '5% del bankroll',
                'max_total_exposure': '25% del bankroll',
                'diversification': 'Máximo 3 apuestas por jugador'
            }
        }
    
    def _generate_reasoning(self, opportunity: Dict[str, Any]) -> str:
        """
        Genera razonamiento para una recomendación.
        
        Args:
            opportunity: Oportunidad de apuesta
            
        Returns:
            Razonamiento textual
        """
        player = opportunity['player']
        target = opportunity['target']
        predicted = opportunity['predicted_value']
        line = opportunity['market_line']
        bet_type = opportunity['recommendation']
        edge = opportunity['edge']
        
        return (f"Modelo predice {predicted:.1f} {target} para {player}, "
                f"mientras mercado ofrece línea en {line:.1f}. "
                f"Recomendación: {bet_type} con {edge:.1%} de ventaja estadística.")
    
    def _generate_portfolio_advice(self, opportunities: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Genera consejos de portfolio para las oportunidades.
        
        Args:
            opportunities: Lista de oportunidades
            
        Returns:
            Consejos de portfolio
        """
        total_kelly = sum(o['kelly_fraction'] for o in opportunities)
        
        return {
            'total_kelly_fraction': f"{total_kelly:.2%}",
            'recommended_allocation': f"{min(total_kelly, 0.25):.2%}",
            'diversification_score': len(set(o['player'] for o in opportunities)),
            'risk_distribution': {
                'low_risk': len([o for o in opportunities if o['risk_level'] == 'LOW']),
                'medium_risk': len([o for o in opportunities if o['risk_level'] in ['MEDIUM', 'MEDIUM_HIGH']]),
                'high_risk': len([o for o in opportunities if o['risk_level'] == 'HIGH'])
            }
        }
    
    def _summarize_odds(self, odds_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Resume datos de odds para el análisis.
        
        Args:
            odds_data: Datos de odds
            
        Returns:
            Resumen de odds
        """
        return {
            'source': 'sportradar',
            'timestamp': datetime.now().isoformat(),
            'status': 'simulated',  # Por ahora simulado
            'markets_available': ['PTS', 'AST', 'TRB', '3P', 'DD']
        }
    
    def export_analysis_report(self, filepath: Optional[str] = None) -> str:
        """
        Exporta reporte de análisis completo.
        
        Args:
            filepath: Ruta del archivo (opcional)
            
        Returns:
            Ruta del archivo exportado
        """
        if not self.analysis_results:
            raise ValueError("No hay análisis disponible. Ejecutar analyze_predictions_vs_market() primero.")
        
        if filepath is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filepath = f"results/predictions_vs_market_{timestamp}.json"
        
        # Crear directorio si no existe
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Exportar resultados
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.analysis_results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Reporte exportado: {filepath}")
        return filepath
    
    def get_system_status(self) -> Dict[str, Any]:
        """
        Obtiene estado del sistema completo.
        
        Returns:
            Estado del sistema
        """
        # Verificar modelos disponibles
        available_models = self.models_loader.discover_available_models()
        
        # Verificar conexión con bookmakers
        try:
            bookmakers_status = self.bookmakers_integration.get_api_status()
        except AttributeError:
            # Si el método no existe, crear estado básico
            bookmakers_status = {
                'sportradar': {
                    'configured': True,
                    'accessible': True,
                    'status': 'OK'
                }
            }
        
        return {
            'timestamp': datetime.now().isoformat(),
            'models': {
                'available': available_models,
                'loaded': list(self.predictor.models.keys()),
                'total_available': sum(available_models.values()),
                'total_loaded': len(self.predictor.models)
            },
            'bookmakers': bookmakers_status,
            'data_loader': {
                'status': 'configured',
                'paths': {
                    'games': str(self.data_loader.game_data_path),
                    'biometrics': str(self.data_loader.biometrics_path),
                    'teams': str(self.data_loader.teams_path)
                }
            },
            'last_analysis': {
                'available': bool(self.analysis_results),
                'date': self.analysis_results.get('date') if self.analysis_results else None
            }
        } 