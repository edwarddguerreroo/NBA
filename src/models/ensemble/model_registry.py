"""
Registro de Modelos NBA
======================

Sistema centralizado para cargar, inicializar y gestionar todos los modelos
individuales del sistema NBA.
"""

import os
import pickle
import joblib
import json
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
import pandas as pd
import numpy as np
from datetime import datetime
import warnings

from config.logging_config import NBALogger
logger = NBALogger.get_logger(__name__)

warnings.filterwarnings('ignore')


class ModelRegistry:
    """Registro centralizado de modelos NBA"""
    
    def __init__(self, models_base_path: str = "results"):
        self.models_base_path = Path(models_base_path)
        self.loaded_models = {}
        self.model_metadata = {}
        self.failed_models = {}
        
        self.model_paths = {
            'pts_player': 'players/pts_model',
            'trb_player': 'players/trb_model', 
            'ast_player': 'players/ast_model',
            'triples_player': 'players/3pt_model',
            'double_double_player': 'players/double_double_model',
            'teams_points': 'teams/teams_points_model',
            'total_points': 'teams/total_points_model',
            'is_win': 'teams/is_win_model'
        }
        
        # Mapeo de archivos específicos por modelo
        self.model_files = {
            'pts_player': 'xgboost_pts_model.pkl',
            'trb_player': 'xgboost_trb_model.pkl',
            'ast_player': 'xgboost_ast_model.pkl', 
            'triples_player': '3pt_model.joblib',
            'double_double_player': 'dd_model.joblib',
            'teams_points': 'teams_points_model.pkl',
            'total_points': 'total_points_model.pkl',
            'is_win': 'is_win_model.pkl'
        }
    
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
        """Cargar un modelo específico"""
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
            # Usar archivo específico si está definido
            if model_name in self.model_files:
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
                # Buscar archivo principal del modelo
                model_files = list(model_path.glob('*.pkl')) + list(model_path.glob('*.joblib'))
                
                if not model_files:
                    logger.error(f"No se encontraron archivos de modelo en {model_path}")
                    return None
                
                # Cargar el primer archivo encontrado
                model_file = model_files[0]
            
            # Intentar cargar el modelo con diferentes métodos
            model = None
            
            if model_file.suffix == '.pkl':
                # Intentar pickle primero
                try:
                    with open(model_file, 'rb') as f:
                        model = pickle.load(f)
                except Exception as e:
                    logger.warning(f"Error con pickle para {model_name}: {e}")
                    # Intentar joblib como respaldo
                    try:
                        model = joblib.load(model_file)
                        logger.info(f"Modelo {model_name} cargado con joblib como respaldo")
                    except Exception as e2:
                        logger.error(f"Error también con joblib para {model_name}: {e2}")
                        return None
            else:
                # Intentar joblib primero
                try:
                    model = joblib.load(model_file)
                except Exception as e:
                    logger.warning(f"Error con joblib para {model_name}: {e}")
                    # Intentar pickle como respaldo  
                    try:
                        with open(model_file, 'rb') as f:
                            model = pickle.load(f)
                        logger.info(f"Modelo {model_name} cargado con pickle como respaldo")
                    except Exception as e2:
                        logger.error(f"Error también con pickle para {model_name}: {e2}")
                        return None
            
            if model is None:
                logger.error(f"No se pudo cargar el modelo {model_name} con ningún método")
                return None
            
            self.loaded_models[model_name] = model
            self.model_metadata[model_name] = {
                'loaded_at': datetime.now(),
                'path': str(model_path),
                'file': str(model_file)
            }
            
            logger.info(f"Modelo {model_name} cargado exitosamente")
            return model
            
        except Exception as e:
            logger.error(f"Error cargando modelo {model_name}: {e}")
            self.failed_models[model_name] = str(e)
            return None
    
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
            'failed_models': list(self.failed_models.keys())
        } 