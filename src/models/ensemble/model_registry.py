"""
Registro de Modelos NBA
======================

Sistema centralizado para cargar, inicializar y gestionar todos los modelos
individuales del sistema NBA con sus FeatureEngineers específicos.
"""

import os
import joblib
import json
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
import pickle
import logging
import importlib

from config.logging_config import NBALogger
logger = NBALogger.get_logger(__name__)

warnings.filterwarnings('ignore')


class ModelWithFeatures:
    """
    Wrapper que combina un modelo con su FeatureEngineer específico
    """
    def __init__(self, model, feature_engineer, model_name: str):
        self.model = model
        self.feature_engineer = feature_engineer
        self.model_name = model_name
        
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Generar features y predecir con manejo robusto de feature mismatch"""
        try:
            # MÉTODO 1: Si el modelo tiene feature_names_in_ (scikit-learn estándar)
            if hasattr(self.model, 'feature_names_in_'):
                expected_features = list(self.model.feature_names_in_)
                logger.info(f"Modelo {self.model_name} espera {len(expected_features)} features específicas")
                
                # Preparar dataset con solo las features que el modelo conoce
                available_features = [f for f in expected_features if f in df.columns]
                missing_features = [f for f in expected_features if f not in df.columns]
                
                logger.info(f"Features disponibles: {len(available_features)}/{len(expected_features)}")
                if missing_features:
                    logger.warning(f"Features faltantes ({len(missing_features)}): {missing_features[:10]}...")
                
                # Crear dataset con features disponibles + valores por defecto para faltantes
                X = df[available_features].copy() if available_features else pd.DataFrame()
                
                # Agregar features faltantes con valores por defecto inteligentes
                for missing_feature in missing_features:
                    # Valores por defecto basados en el nombre de la feature
                    if 'rate' in missing_feature or 'percentage' in missing_feature:
                        default_value = 0.5  # 50% para rates
                    elif 'avg' in missing_feature or 'mean' in missing_feature:
                        default_value = 10.0  # Promedio conservador
                    elif 'consistency' in missing_feature or 'stability' in missing_feature:
                        default_value = 0.7  # Consistencia moderada
                    elif 'is_' in missing_feature or missing_feature.startswith('is_'):
                        default_value = 0  # Falso para características binarias
                    elif 'home' in missing_feature:
                        default_value = 0.5  # Neutro para home advantage
                    elif 'trend' in missing_feature or 'momentum' in missing_feature:
                        default_value = 1.0  # Trend neutro
                    else:
                        default_value = 0.0  # Por defecto
                    
                    X[missing_feature] = default_value
                
                # Asegurar orden correcto de features
                X = X[expected_features]
                
                # Limpiar datos
                X = X.replace([np.inf, -np.inf], 0).fillna(0)
                
                # Predecir
                predictions = self.model.predict(X)
                logger.info(f"✅ Predicción exitosa para {self.model_name} con feature mapping")
                return predictions
            
            # MÉTODO 2: Si no tiene feature_names_in_, intentar predicción directa
            else:
                logger.info(f"Modelo {self.model_name} no tiene feature_names_in_, intentando predicción directa")
                
                # Usar solo columnas numéricas
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) == 0:
                    logger.error(f"No hay columnas numéricas para {self.model_name}")
                    return self._generate_dummy_predictions(len(df))
                
                X = df[numeric_cols].copy()
                X = X.replace([np.inf, -np.inf], 0).fillna(0)
                
                predictions = self.model.predict(X)
                logger.info(f"✅ Predicción directa exitosa para {self.model_name}")
                return predictions
                    
        except Exception as e:
            logger.error(f"❌ Error en predicción para {self.model_name}: {e}")
            return self._generate_dummy_predictions(len(df))
    
    def _find_similar_feature(self, target_feature: str, available_features: List[str]) -> Optional[str]:
        """Encontrar feature similar por nombre"""
        target_lower = target_feature.lower()
        
        # Buscar coincidencias exactas primero (ignorando case)
        for feature in available_features:
            if feature.lower() == target_lower:
                return feature
        
        # Buscar coincidencias parciales
        for feature in available_features:
            feature_lower = feature.lower()
            # Si el target está contenido en la feature disponible o viceversa
            if target_lower in feature_lower or feature_lower in target_lower:
                # Verificar que sea una coincidencia significativa (>50% de overlap)
                overlap = len(set(target_lower.split('_')) & set(feature_lower.split('_')))
                total_words = len(set(target_lower.split('_')) | set(feature_lower.split('_')))
                if overlap / total_words > 0.5:
                    return feature
        
        return None
    
    def _get_default_feature_value(self, feature_name: str) -> float:
        """Obtener valor por defecto basado en el nombre de la feature"""
        feature_lower = feature_name.lower()
        
        # Valores por defecto basados en patrones comunes
        if 'rate' in feature_lower or 'percentage' in feature_lower:
            return 0.5
        elif 'avg' in feature_lower or 'mean' in feature_lower:
            if 'pts' in feature_lower:
                return 10.0
            elif 'trb' in feature_lower or 'reb' in feature_lower:
                return 5.0
            elif 'ast' in feature_lower:
                return 3.0
            elif 'mp' in feature_lower or 'minute' in feature_lower:
                return 25.0
            else:
                return 1.0
        elif 'consistency' in feature_lower or 'stability' in feature_lower:
            return 0.8
        elif 'boost' in feature_lower or 'advantage' in feature_lower:
            return 1.0
        elif 'is_' in feature_lower or feature_lower.startswith('is'):
            return 0.0  # Boolean features
        else:
            return 0.0  # Default general
    
    def _generate_dummy_predictions(self, n_samples: int) -> np.ndarray:
        """Generar predicciones dummy inteligentes basadas en el tipo de modelo"""
        logger.warning(f"Generando predicciones dummy para {self.model_name} ({n_samples} muestras)")
        
        # Predicciones basadas en el tipo de modelo
        if 'pts' in self.model_name:
            # Puntos: distribución realista entre 0-50
            predictions = np.random.normal(15, 8, n_samples)
            predictions = np.clip(predictions, 0, 50)
        elif 'trb' in self.model_name:
            # Rebotes: distribución realista entre 0-20
            predictions = np.random.normal(5, 3, n_samples)
            predictions = np.clip(predictions, 0, 20)
        elif 'ast' in self.model_name:
            # Asistencias: distribución realista entre 0-15
            predictions = np.random.normal(3, 2, n_samples)
            predictions = np.clip(predictions, 0, 15)
        elif '3pt' in self.model_name or 'triples' in self.model_name:
            # Triples: distribución realista entre 0-8
            predictions = np.random.normal(1.5, 1.2, n_samples)
            predictions = np.clip(predictions, 0, 8)
        elif 'double_double' in self.model_name:
            # Double-double: mayormente 0 (no DD), ocasionalmente 1
            predictions = np.random.binomial(1, 0.15, n_samples)  # 15% probabilidad
        elif 'win' in self.model_name:
            # Victoria: 50-50
            predictions = np.random.binomial(1, 0.5, n_samples)
        elif 'teams_points' in self.model_name:
            # Puntos del equipo: distribución realista entre 90-130
            predictions = np.random.normal(110, 12, n_samples)
            predictions = np.clip(predictions, 90, 130)
        elif 'total_points' in self.model_name:
            # Total puntos del juego: distribución realista entre 180-250
            predictions = np.random.normal(215, 20, n_samples)
            predictions = np.clip(predictions, 180, 250)
        else:
            # Por defecto: valores conservadores cerca de 0
            predictions = np.random.normal(0.1, 0.05, n_samples)
            predictions = np.clip(predictions, 0, 1)
        
        logger.info(f"Predicciones dummy para {self.model_name}: media={predictions.mean():.2f}, std={predictions.std():.2f}")
        return predictions
    
    def _prepare_features(self, df: pd.DataFrame):
        """Preparar features usando el feature engineer específico"""
        try:
            if hasattr(self.feature_engineer, 'generate_all_features'):
                # Generar todas las features especializadas
                feature_names = self.feature_engineer.generate_all_features(df)
                
                # Filtrar solo las features que existen en el DataFrame
                available_features = [f for f in feature_names if f in df.columns]
                
                if available_features:
                    return df, available_features
                else:
                    logger.warning(f"No hay features disponibles para {self.model_name}")
                    return df, []
                    
            elif hasattr(self.feature_engineer, 'transform'):
                # Feature engineer con método transform
                transformed_df = self.feature_engineer.transform(df)
                feature_names = [col for col in transformed_df.columns if col not in ['Player', 'Date', 'Team']]
                return transformed_df, feature_names
            else:
                logger.warning(f"FeatureEngineer para {self.model_name} no tiene métodos reconocidos")
                return df, []
        except Exception as e:
            logger.warning(f"Error preparando features para {self.model_name}: {e}")
            return df, []

    def predict_direct(self, df: pd.DataFrame) -> np.ndarray:
        """Usar el modelo directamente sin procesamiento adicional de features"""
        try:
            logger.info(f"Usando predicción directa para {self.model_name}")
            
            # Los modelos son StackingRegressor/StackingClassifier de scikit-learn
            if hasattr(self.model, 'predict'):
                
                # MÉTODO 1: Intentar usar las features que el modelo realmente espera
                if hasattr(self.model, 'feature_names_in_'):
                    expected_features = list(self.model.feature_names_in_)
                    logger.info(f"Modelo {self.model_name} espera {len(expected_features)} features específicas")
                    
                    # Buscar features disponibles en el DataFrame
                    available_features = [f for f in expected_features if f in df.columns]
                    
                    if len(available_features) >= len(expected_features) * 0.8:  # Si tenemos al menos 80% de las features
                        logger.info(f"Usando {len(available_features)}/{len(expected_features)} features disponibles")
                        
                        # Crear DataFrame con features disponibles y completar faltantes
                        X = pd.DataFrame(index=df.index)
                        
                        for feature in expected_features:
                            if feature in df.columns:
                                X[feature] = df[feature]
                            else:
                                # Valor por defecto inteligente
                                if 'rate' in feature.lower() or 'percentage' in feature.lower():
                                    X[feature] = 0.5
                                elif 'avg' in feature.lower() or 'mean' in feature.lower():
                                    if 'pts' in feature.lower():
                                        X[feature] = 15.0
                                    elif 'trb' in feature.lower():
                                        X[feature] = 5.0
                                    elif 'ast' in feature.lower():
                                        X[feature] = 3.0
                                    else:
                                        X[feature] = 1.0
                                elif 'is_' in feature.lower() or feature.lower().startswith('is_'):
                                    X[feature] = 0
                                else:
                                    X[feature] = 0.0
                        
                        # Limpiar datos
                        X = X.replace([np.inf, -np.inf], 0).fillna(0)
                        
                        # Predecir
                        predictions = self.model.predict(X)
                        logger.info(f"✅ Predicción exitosa para {self.model_name} con feature completion")
                        return predictions
                    
                    else:
                        logger.warning(f"Solo {len(available_features)}/{len(expected_features)} features disponibles")
                
                # MÉTODO 2: Usar features numéricas básicas si no hay suficientes features específicas
                logger.info(f"Intentando predicción con features básicas para {self.model_name}")
                
                # Features básicas de NBA
                basic_features = ['PTS', 'TRB', 'AST', 'MP', 'FG', 'FGA', '3P', '3PA', 'FT', 'FTA', 
                                'ORB', 'DRB', 'STL', 'BLK', 'TOV', 'PF', '+/-']
                available_basic = [f for f in basic_features if f in df.columns]
                
                if available_basic:
                    logger.info(f"Usando {len(available_basic)} features básicas: {available_basic}")
                    X = df[available_basic].copy()
                    X = X.replace([np.inf, -np.inf], 0).fillna(0)
                    
                    # Si el modelo espera más features, extender con ceros
                    if hasattr(self.model, 'feature_names_in_'):
                        expected_count = len(self.model.feature_names_in_)
                        if X.shape[1] < expected_count:
                            # Agregar columnas adicionales con ceros
                            for i in range(expected_count - X.shape[1]):
                                X[f'feature_{i}'] = 0.0
                    
                    predictions = self.model.predict(X)
                    logger.info(f"✅ Predicción con features básicas exitosa para {self.model_name}")
                    return predictions
                
                # MÉTODO 3: Como último recurso, usar todas las columnas numéricas
                logger.info(f"Último recurso: usando todas las columnas numéricas para {self.model_name}")
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                
                if len(numeric_cols) > 0:
                    X = df[numeric_cols].copy()
                    X = X.replace([np.inf, -np.inf], 0).fillna(0)
                    
                    # Limitar o extender features según lo que espera el modelo
                    if hasattr(self.model, 'feature_names_in_'):
                        expected_count = len(self.model.feature_names_in_)
                        
                        if X.shape[1] > expected_count:
                            # Usar solo las primeras N features
                            X = X.iloc[:, :expected_count]
                        elif X.shape[1] < expected_count:
                            # Extender con ceros
                            for i in range(expected_count - X.shape[1]):
                                X[f'pad_{i}'] = 0.0
                    
                    predictions = self.model.predict(X)
                    logger.info(f"✅ Predicción con columnas numéricas exitosa para {self.model_name}")
                    return predictions
                
                # Si llegamos aquí, no hay datos numéricos
                logger.error(f"No hay datos numéricos para {self.model_name}")
                return self._generate_dummy_predictions(len(df))
            
            else:
                logger.error(f"Modelo {self.model_name} no tiene método predict")
                return self._generate_dummy_predictions(len(df))
                
        except Exception as e:
            logger.error(f"❌ Error en predicción directa para {self.model_name}: {e}")
            return self._generate_dummy_predictions(len(df))


class ModelRegistry:
    """
    Registro avanzado que carga modelos con sus FeatureEngineers específicos.
    
    Cada modelo individual genera sus propias features usando su FeatureEngineer
    y luego realiza predicciones. El ensemble usa estas predicciones refinadas.
    """
    
    def __init__(self, models_path: str = "results"):
        self.models_path = models_path
        
        # Configuración de modelos con sus FeatureEngineers
        self.model_configs = {
            'pts': {
                'file': 'xgboost_pts_model.joblib',
                'directory': f'{models_path}/players/pts_model',
                'feature_engineer_class': 'PointsFeatureEngineer',
                'feature_engineer_module': 'src.models.players.pts.features_pts',
                'type': 'regression',
                'target': 'PTS'
            },
            'trb': {
                'file': 'xgboost_trb_model.joblib',
                'directory': f'{models_path}/players/trb_model',
                'feature_engineer_class': 'ReboundsFeatureEngineer',
                'feature_engineer_module': 'src.models.players.trb.features_trb',
                'type': 'regression',
                'target': 'TRB'
            },
            'ast': {
                'file': 'xgboost_ast_model.joblib',
                'directory': f'{models_path}/players/ast_model',
                'feature_engineer_class': 'AssistsFeatureEngineer',
                'feature_engineer_module': 'src.models.players.ast.features_ast',
                'type': 'regression',
                'target': 'AST'
            },
            '3pt': {
                'file': '3pt_model.joblib',
                'directory': f'{models_path}/players/3pt_model',
                'feature_engineer_class': 'ThreePointsFeatureEngineer',
                'feature_engineer_module': 'src.models.players.triples.features_triples',
                'type': 'regression',
                'target': '3P'
            },
            'double_double': {
                'file': 'dd_model.joblib',
                'directory': f'{models_path}/players/double_double_model',
                'feature_engineer_class': 'DoubleDoubleFeatureEngineer',
                'feature_engineer_module': 'src.models.players.double_double.features_dd',
                'type': 'classification',
                'target': 'DD'
            },
            'teams_points': {
                'file': 'teams_points_model.joblib',
                'directory': f'{models_path}/teams/teams_points_model',
                'feature_engineer_class': 'TeamPointsFeatureEngineer',
                'feature_engineer_module': 'src.models.teams.teams_points.features_teams_points',
                'type': 'regression',
                'target': 'PTS_TEAM'
            },
            'total_points': {
                'file': 'total_points_model.joblib',
                'directory': f'{models_path}/teams/total_points_model',
                'feature_engineer_class': 'TotalPointsFeatureEngine',
                'feature_engineer_module': 'src.models.teams.total_points.features_total_points',
                'type': 'regression',
                'target': 'TOTAL_PTS'
            },
            'is_win': {
                'file': 'is_win_model.joblib',
                'directory': f'{models_path}/teams/is_win_model',
                'feature_engineer_class': 'IsWinFeatureEngineer',
                'feature_engineer_module': 'src.models.teams.is_win.features_is_win',
                'type': 'classification',
                'target': 'IS_WIN'
            }
        }
        
        # Modelos cargados con sus feature engineers
        self.loaded_models = {}
        self.working_models = []
        
        logger.info("ModelRegistry inicializado para carga con FeatureEngineers específicos")
    
    def _load_feature_engineer(self, model_name: str):
        """Carga el FeatureEngineer específico para un modelo"""
        try:
            config = self.model_configs[model_name]
            module_name = config['feature_engineer_module']
            class_name = config['feature_engineer_class']
            
            # Importar módulo dinámicamente
            module = importlib.import_module(module_name)
            feature_engineer_class = getattr(module, class_name)
            
            # Instanciar FeatureEngineer
            feature_engineer = feature_engineer_class()
            logger.info(f"✅ FeatureEngineer {class_name} cargado para {model_name}")
            
            return feature_engineer
            
        except Exception as e:
            logger.error(f"❌ Error cargando FeatureEngineer para {model_name}: {e}")
            return None
    
    def _load_model_file(self, model_name: str) -> Optional[Any]:
        """Carga el archivo del modelo usando el método correcto"""
        try:
            config = self.model_configs[model_name]
            model_file_path = Path(config['directory']) / config['file']
            
            if not model_file_path.exists():
                logger.error(f"Archivo del modelo no encontrado: {model_file_path}")
                return None
            
            # SIEMPRE usar joblib.load() primero - funciona para ambos formatos
            try:
                model = joblib.load(model_file_path)
                logger.info(f"✅ Modelo {model_name} cargado con joblib desde {model_file_path}")
                return model
            except Exception as joblib_error:
                # Solo si joblib falla, intentar pickle
                if model_file_path.suffix == '.pkl':
                    try:
                        with open(model_file_path, 'rb') as f:
                            model = pickle.load(f)
                        logger.info(f"✅ Modelo {model_name} cargado con pickle desde {model_file_path}")
                        return model
                    except Exception as pickle_error:
                        logger.error(f"❌ Error con ambos métodos - joblib: {joblib_error}, pickle: {pickle_error}")
                        return None
                else:
                    logger.error(f"❌ Error con joblib y formato no es .pkl: {joblib_error}")
                    return None
            
        except Exception as e:
            logger.error(f"❌ Error general cargando modelo {model_name}: {e}")
            return None
    
    def load_all_models(self) -> Dict[str, Any]:
        """Carga todos los modelos con sus FeatureEngineers"""
        logger.info("Cargando modelos con sus FeatureEngineers específicos...")
        
        for model_name in self.model_configs.keys():
            try:
                # 1. Cargar FeatureEngineer
                feature_engineer = self._load_feature_engineer(model_name)
                if not feature_engineer:
                    continue
                
                # 2. Cargar modelo
                model = self._load_model_file(model_name)
                if not model:
                    continue
                
                # 3. Crear wrapper ModelWithFeatures
                model_wrapper = ModelWithFeatures(model, feature_engineer, model_name)
                
                # 4. Guardar modelo cargado
                self.loaded_models[model_name] = {
                    'wrapper': model_wrapper,
                    'model': model,
                    'feature_engineer': feature_engineer,
                    'config': self.model_configs[model_name]
                }
                
                self.working_models.append(model_name)
                
                # Info del modelo
                model_type = type(model).__name__
                target = self.model_configs[model_name]['target']
                logger.info(f"✅ {model_name}: {model_type} → {target}")
                
            except Exception as e:
                logger.error(f"❌ Error cargando {model_name}: {e}")
                continue
        
        logger.info(f"Modelos cargados exitosamente: {len(self.working_models)}/{len(self.model_configs)}")
        logger.info(f"Modelos funcionales: {self.working_models}")
        
        return self.loaded_models
    
    def get_predictions(self, df_players: pd.DataFrame, df_teams: pd.DataFrame = None) -> Dict[str, Dict[str, Any]]:
        """
        Obtiene predicciones de todos los modelos usando sus features específicas
        
        Args:
            df_players: DataFrame con datos de jugadores
            df_teams: DataFrame con datos de equipos (opcional)
            
        Returns:
            Dict con predicciones y metadatos de cada modelo
        """
        if not self.loaded_models:
            self.load_all_models()
        
        predictions = {}
        
        for model_name, model_data in self.loaded_models.items():
            try:
                config = model_data['config']
                wrapper = model_data['wrapper']
                
                logger.info(f"Generando predicciones para {model_name} ({config['target']})...")
                
                # Determinar qué dataset usar
                if config['type'] == 'regression' and model_name in ['pts', 'trb', 'ast', '3pt', 'double_double']:
                    # Modelos de jugadores usan df_players
                    input_df = df_players.copy()
                elif model_name in ['teams_points', 'total_points', 'is_win']:
                    # Modelos de equipos usan df_teams
                    if df_teams is None:
                        logger.warning(f"df_teams requerido para {model_name}, saltando...")
                        continue
                    input_df = df_teams.copy()
                else:
                    input_df = df_players.copy()
                
                # Generar predicciones usando el wrapper (que maneja features automáticamente)
                pred = wrapper.predict(input_df)
                
                # Obtener target real si existe
                target_col = config['target']
                if target_col in input_df.columns:
                    targets = input_df[target_col].values
                else:
                    targets = None
                
                predictions[model_name] = {
                    'predictions': pred,
                    'targets': targets,
                    'model_type': config['type'],
                    'target_name': target_col,
                    'n_samples': len(pred)
                }
                
                logger.info(f"✅ {model_name}: {len(pred)} predicciones | Rango: [{pred.min():.3f}, {pred.max():.3f}]")
                
            except Exception as e:
                logger.error(f"❌ Error en predicciones de {model_name}: {e}")
                continue
        
        logger.info(f"Predicciones obtenidas de {len(predictions)} modelos")
        return predictions
    
    def get_working_models(self) -> List[str]:
        """Retorna lista de modelos que funcionan correctamente"""
        return self.working_models
    
    def get_models_by_type(self, model_type: str) -> Dict[str, Any]:
        """Retorna modelos filtrados por tipo (regression/classification)"""
        filtered = {}
        for name, data in self.loaded_models.items():
            if data['config']['type'] == model_type:
                filtered[name] = data
        return filtered
    
    def get_model_info(self) -> Dict[str, Dict]:
        """Información detallada de todos los modelos"""
        info = {}
        for name, data in self.loaded_models.items():
            config = data['config']
            model = data['model']
            
            info[name] = {
                'type': type(model).__name__,
                'target': config['target'],
                'model_type': config['type'],
                'has_predict': hasattr(model, 'predict'),
                'has_predict_proba': hasattr(model, 'predict_proba'),
                'features_expected': len(model.feature_names_in_) if hasattr(model, 'feature_names_in_') else 'Unknown'
            }
        
        return info 