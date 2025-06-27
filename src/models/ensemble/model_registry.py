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
        """Generar features y predecir"""
        try:
            # Si el modelo tiene su propio método predict que maneja features
            if hasattr(self.model, 'predict') and hasattr(self.model, 'feature_engineer'):
                return self.model.predict(df)
            
            # Si tenemos un feature engineer separado
            if self.feature_engineer is not None:
                features_df, feature_names = self._prepare_features(df)
                if not features_df.empty and feature_names:
                    X = features_df[feature_names].fillna(0)
                    return self.model.predict(X)
            
            # Fallback: intentar predicción directa
            return self.model.predict(df)
            
        except Exception as e:
            logger.error(f"Error en predicción para {self.model_name}: {e}")
            # Retornar predicciones dummy en caso de error
            return np.zeros(len(df))
    
    def _prepare_features(self, df: pd.DataFrame):
        """Preparar features usando el feature engineer específico"""
        try:
            if hasattr(self.feature_engineer, 'generate_all_features'):
                features = self.feature_engineer.generate_all_features(df)
                if features:
                    optimal_features = self.feature_engineer.get_optimal_feature_set(df)
                    return df, optimal_features
            elif hasattr(self.feature_engineer, 'transform'):
                return self.feature_engineer.transform(df), []
            else:
                return df, []
        except Exception as e:
            logger.warning(f"Error preparando features para {self.model_name}: {e}")
            return df, []


class ModelRegistry:
    """Registro centralizado de modelos NBA con FeatureEngineers"""
    
    def __init__(self, models_base_path: str = "results"):
        self.models_base_path = Path(models_base_path)
        self.loaded_models = {}
        self.model_metadata = {}
        self.failed_models = {}
        
        self.model_paths = {
            'pts_player': 'players/pts_model',  # Directorio existe
            'trb_player': 'players/trb_model',  # Directorio existe
            'ast_player': 'players/ast_model',  # Directorio existe
            'triples_player': 'players/3pt_model',  # Directorio existe (CORREGIDO)
            'double_double_player': 'players/double_double_model',  # Directorio existe (CORREGIDO)
            'teams_points': 'teams/teams_points_model',  # Directorio existe
            'total_points': 'teams/total_points_model',  # Directorio existe
            'is_win': 'teams/is_win_model'  # Directorio existe
        }
        
        # Mapeo de archivos específicos por modelo (CORREGIDO basado en archivos reales)
        self.model_files = {
                    'pts_player': 'xgboost_pts_model.joblib',  # Existe en players/pts_model/
        'trb_player': 'xgboost_trb_model.joblib',  # Existe en players/trb_model/
        'ast_player': 'xgboost_ast_model.joblib',  # Existe en players/ast_model/
            'triples_player': '3pt_model.joblib',  # Existe en players/3pt_model/ (CORREGIDO)
            'double_double_player': 'dd_model.joblib',  # Existe en players/double_double_model/ (CORREGIDO)
                    'teams_points': 'teams_points_model.joblib',  # Existe en teams/teams_points_model/
        'total_points': 'total_points_model.joblib',  # Existe en teams/total_points_model/
        'is_win': 'is_win_model.joblib'  # Existe en teams/is_win_model/
        }
        
        # Mapeo de FeatureEngineers por modelo
        self.feature_engineers = {
            'pts_player': 'src.models.players.pts.features_pts.PointsFeatureEngineer',
            'trb_player': 'src.models.players.trb.features_trb.ReboundsFeatureEngineer',
            'ast_player': 'src.models.players.ast.features_ast.AssistsFeatureEngineer',
            'triples_player': 'src.models.players.triples.features_triples.ThreePointsFeatureEngineer',
            'double_double_player': 'src.models.players.double_double.features_dd.DoubleDoubleFeatureEngineer',
            'teams_points': 'src.models.teams.teams_points.features_teams_points.TeamPointsFeatureEngineer',
            'total_points': 'src.models.teams.total_points.features_total_points.TotalPointsFeatureEngine',
            'is_win': 'src.models.teams.is_win.features_is_win.IsWinFeatureEngineer'
        }
    
    def _import_feature_engineer(self, module_path: str):
        """Importar dinámicamente un FeatureEngineer"""
        try:
            module_name, class_name = module_path.rsplit('.', 1)
            module = __import__(module_name, fromlist=[class_name])
            feature_engineer_class = getattr(module, class_name)
            return feature_engineer_class()
        except Exception as e:
            logger.warning(f"No se pudo importar FeatureEngineer desde {module_path}: {e}")
            return None
    
    def discover_available_models(self) -> Dict[str, Dict[str, Any]]:
        """Descubrir modelos disponibles"""
        available_models = {}
        
        for model_name, relative_path in self.model_paths.items():
            model_path = self.models_base_path / relative_path
            
            if model_path.exists():
                available_models[model_name] = {
                    'status': 'available',
                    'path': str(model_path)
                }
            else:
                available_models[model_name] = {
                    'status': 'not_found',
                    'path': str(model_path)
                }
        
        return available_models
    
    def load_model(self, model_name: str) -> Optional[Any]:
        """Cargar un modelo específico con su FeatureEngineer"""
        if model_name in self.loaded_models:
            return self.loaded_models[model_name]
        
        if model_name not in self.model_paths:
            logger.error(f"Modelo '{model_name}' no reconocido")
            return None
        
        model_path = self.models_base_path / self.model_paths[model_name]
        
        if not model_path.exists():
            logger.error(f"Directorio del modelo no encontrado: {model_path}")
            return None
        
        try:
            # Cargar el modelo principal
            model = self._load_model_file(model_name, model_path)
            if model is None:
                return None
            
            # Cargar FeatureEngineer específico
            feature_engineer = None
            if model_name in self.feature_engineers:
                feature_engineer = self._import_feature_engineer(self.feature_engineers[model_name])
            
            # Crear wrapper que combina modelo y feature engineer
            model_wrapper = ModelWithFeatures(model, feature_engineer, model_name)
            
            self.loaded_models[model_name] = model_wrapper
            self.model_metadata[model_name] = {
                'loaded_at': datetime.now(),
                'path': str(model_path),
                'has_feature_engineer': feature_engineer is not None
            }
            
            logger.info(f"Modelo {model_name} cargado exitosamente con FeatureEngineer: {feature_engineer is not None}")
            return model_wrapper
            
        except Exception as e:
            logger.error(f"Error cargando modelo {model_name}: {e}")
            self.failed_models[model_name] = str(e)
            return None
    
    def _load_model_file(self, model_name: str, model_path: Path) -> Optional[Any]:
        """Cargar archivo de modelo con manejo robusto de formatos"""
        # Usar archivo específico si está definido y no es None
        if model_name in self.model_files and self.model_files[model_name] is not None:
            model_file = model_path / self.model_files[model_name]
            
            if not model_file.exists():
                logger.error(f"Archivo específico del modelo no encontrado: {model_file}")
                # Buscar archivo alternativo
                model_files = list(model_path.glob('*.pkl')) + list(model_path.glob('*.joblib'))
                if model_files:
                    model_file = model_files[0]
                    logger.info(f"Usando archivo alternativo: {model_file}")
                else:
                    logger.error(f"No se encontraron archivos de modelo en {model_path}")
                    return None
        else:
            # Buscar archivo principal del modelo automáticamente
            model_files = list(model_path.glob('*.pkl')) + list(model_path.glob('*.joblib'))
            
            if not model_files:
                logger.error(f"No se encontraron archivos de modelo en {model_path}")
                return None
            
            # Cargar el primer archivo encontrado
            model_file = model_files[0]
            logger.info(f"Auto-detectado archivo: {model_file}")
        
        # Intentar cargar el modelo con diferentes métodos
        model = None
        
        if model_file.suffix == '.pkl':
            # Intentar pickle primero
            try:
                with open(model_file, 'rb') as f:
                    loaded_data = pickle.load(f)
                
                # Verificar si es un diccionario (formato legacy) o modelo directo
                if isinstance(loaded_data, dict):
                    # Formato diccionario - buscar el modelo dentro
                    if 'stacking_model' in loaded_data:
                        model = loaded_data['stacking_model']
                        logger.info(f"Modelo {model_name} extraído de diccionario (stacking_model)")
                    elif 'model' in loaded_data:
                        model = loaded_data['model']
                        logger.info(f"Modelo {model_name} extraído de diccionario (model)")
                    elif hasattr(loaded_data, 'predict'):
                        # El diccionario en sí es el modelo
                        model = loaded_data
                        logger.info(f"Modelo {model_name} cargado como diccionario completo")
                    else:
                        # Buscar cualquier objeto con método predict en el diccionario
                        for key, value in loaded_data.items():
                            if hasattr(value, 'predict'):
                                model = value
                                logger.info(f"Modelo {model_name} extraído de diccionario ({key})")
                                break
                else:
                    # Formato objeto directo
                    model = loaded_data
                    logger.info(f"Modelo {model_name} cargado como objeto directo")
                    
            except Exception as e:
                logger.warning(f"Error con pickle para {model_name}: {e}")
                # Intentar joblib como respaldo
                try:
                    loaded_data = joblib.load(model_file)
                    
                    # Aplicar la misma lógica para joblib
                    if isinstance(loaded_data, dict):
                        if 'stacking_model' in loaded_data:
                            model = loaded_data['stacking_model']
                            logger.info(f"Modelo {model_name} extraído de diccionario joblib (stacking_model)")
                        elif 'model' in loaded_data:
                            model = loaded_data['model']
                            logger.info(f"Modelo {model_name} extraído de diccionario joblib (model)")
                        else:
                            for key, value in loaded_data.items():
                                if hasattr(value, 'predict'):
                                    model = value
                                    logger.info(f"Modelo {model_name} extraído de diccionario joblib ({key})")
                                    break
                    else:
                        model = loaded_data
                        logger.info(f"Modelo {model_name} cargado con joblib como respaldo")
                        
                except Exception as e2:
                    logger.error(f"Error también con joblib para {model_name}: {e2}")
                    return None
        else:
            # Intentar joblib primero para archivos .joblib
            try:
                loaded_data = joblib.load(model_file)
                
                # Aplicar lógica de verificación de formato
                if isinstance(loaded_data, dict):
                    if 'stacking_model' in loaded_data:
                        model = loaded_data['stacking_model']
                        logger.info(f"Modelo {model_name} extraído de diccionario joblib (stacking_model)")
                    elif 'model' in loaded_data:
                        model = loaded_data['model']
                        logger.info(f"Modelo {model_name} extraído de diccionario joblib (model)")
                    else:
                        for key, value in loaded_data.items():
                            if hasattr(value, 'predict'):
                                model = value
                                logger.info(f"Modelo {model_name} extraído de diccionario joblib ({key})")
                                break
                else:
                    model = loaded_data
                    
            except Exception as e:
                logger.warning(f"Error con joblib para {model_name}: {e}")
                # Intentar pickle como respaldo  
                try:
                    with open(model_file, 'rb') as f:
                        loaded_data = pickle.load(f)
                    
                    # Aplicar lógica de verificación de formato
                    if isinstance(loaded_data, dict):
                        if 'stacking_model' in loaded_data:
                            model = loaded_data['stacking_model']
                            logger.info(f"Modelo {model_name} extraído de diccionario pickle (stacking_model)")
                        elif 'model' in loaded_data:
                            model = loaded_data['model']
                            logger.info(f"Modelo {model_name} extraído de diccionario pickle (model)")
                        else:
                            for key, value in loaded_data.items():
                                if hasattr(value, 'predict'):
                                    model = value
                                    logger.info(f"Modelo {model_name} extraído de diccionario pickle ({key})")
                                    break
                    else:
                        model = loaded_data
                        logger.info(f"Modelo {model_name} cargado con pickle como respaldo")
                        
                except Exception as e2:
                    logger.error(f"Error también con pickle para {model_name}: {e2}")
                    return None
        
        # Verificar que el modelo final tiene método predict
        if model is None:
            logger.error(f"No se pudo extraer un modelo válido de {model_name}")
            return None
        
        if not hasattr(model, 'predict'):
            logger.error(f"El objeto cargado para {model_name} no tiene método predict")
            return None
        
        return model
    
    def load_all_models(self) -> Dict[str, Any]:
        """Cargar todos los modelos disponibles"""
        available_models = self.discover_available_models()
        
        for model_name, model_info in available_models.items():
            if model_info['status'] == 'available':
                self.load_model(model_name)
        
        return self.loaded_models
    
    def get_model(self, model_name: str) -> Optional[Any]:
        """Obtener modelo cargado"""
        return self.loaded_models.get(model_name)
    
    def get_registry_summary(self) -> Dict[str, Any]:
        """Obtener resumen del registro"""
        return {
            'models_loaded': len(self.loaded_models),
            'models_failed': len(self.failed_models),
            'loaded_models': list(self.loaded_models.keys()),
            'failed_models': list(self.failed_models.keys()),
            'models_with_feature_engineers': [
                name for name, metadata in self.model_metadata.items() 
                if metadata.get('has_feature_engineer', False)
            ]
        } 