"""
Configuración del Ensemble NBA
=============================

Configuración centralizada para el sistema de ensemble con:
- Parámetros de Optuna para optimización bayesiana
- División cronológica estricta
- Configuración de K-Fold temporal
- Métricas por tipo de modelo
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
import optuna
from sklearn.model_selection import TimeSeriesSplit
import numpy as np


@dataclass
class ModelConfig:
    """Configuración para un modelo específico"""
    name: str
    type: str  # 'regression' o 'classification'
    model_class: str
    feature_engineer_class: str
    weight: float = 1.0
    enabled: bool = True
    target_column: str = ""
    metrics: List[str] = None
    min_samples: int = 1000


@dataclass 
class EnsembleConfig:
    """Configuración principal del ensemble"""
    
    # Configuración de validación temporal
    temporal_validation: Dict[str, Any] = None
    
    # Configuración de K-Fold
    kfold_config: Dict[str, Any] = None
    
    # Configuración de Optuna
    optuna_config: Dict[str, Any] = None
    
    # Modelos disponibles
    models_config: Dict[str, ModelConfig] = None
    
    # Meta-learners configuration
    meta_learners_config: Dict[str, Any] = None
    
    # Ensemble strategies
    ensemble_strategies: List[str] = None
    
    def __post_init__(self):
        """Inicializar configuraciones por defecto"""
        if self.temporal_validation is None:
            self.temporal_validation = {
                'method': 'chronological',
                'validation_split': 0.2,
                'test_split': 0.1,
                'min_train_days': 180,
                'min_val_days': 60,
                'gap_days': 7  # Gap entre train/val para evitar leakage
            }
        
        if self.kfold_config is None:
            self.kfold_config = {
                'n_splits': 5,
                'shuffle': False,  # Mantener orden cronológico
                'random_state': 42,
                'use_time_series_split': True
            }
        
        if self.optuna_config is None:
            self.optuna_config = {
                'n_trials': 100,
                'timeout': 3600,  # 1 hora máximo
                'n_jobs': -1,
                'sampler': 'TPE',
                'pruner': 'MedianPruner',
                'direction': 'minimize',  # Para MAE/RMSE
                'study_name': 'nba_ensemble_optimization'
            }
        
        if self.models_config is None:
            self.models_config = self._create_default_models_config()
        
        if self.meta_learners_config is None:
            self.meta_learners_config = self._create_meta_learners_config()
        
        if self.ensemble_strategies is None:
            self.ensemble_strategies = [
                'weighted_average',
                'stacking_regression',
                'stacking_classification', 
                'voting_regression',
                'voting_classification',
                'hierarchical_ensemble',
                'adaptive_weights'
            ]
    
    def _create_default_models_config(self) -> Dict[str, ModelConfig]:
        """Crear configuración por defecto de modelos"""
        return {
            'pts_player': ModelConfig(
                name='pts_player',
                type='regression',
                model_class='StackingPTSModel',
                feature_engineer_class='PointsFeatureEngineer',
                target_column='PTS',
                weight=1.0,
                metrics=['mae', 'rmse', 'r2', 'mape'],
                min_samples=2000
            ),
            'trb_player': ModelConfig(
                name='trb_player', 
                type='regression',
                model_class='StackingTRBModel',
                feature_engineer_class='ReboundsFeatureEngineer',
                target_column='TRB',
                weight=1.0,
                metrics=['mae', 'rmse', 'r2', 'mape'],
                min_samples=2000
            ),
            'ast_player': ModelConfig(
                name='ast_player',
                type='regression', 
                model_class='StackingASTModel',
                feature_engineer_class='AssistsFeatureEngineer',
                target_column='AST',
                weight=1.0,
                metrics=['mae', 'rmse', 'r2', 'mape'],
                min_samples=2000
            ),
            'triples_player': ModelConfig(
                name='triples_player',
                type='regression',
                model_class='Stacking3PTModel', 
                feature_engineer_class='TriplesFeatureEngineer',
                target_column='3P',
                weight=0.8,  # Menor peso por mayor variabilidad
                metrics=['mae', 'rmse', 'r2', 'accuracy_±1'],
                min_samples=1500
            ),
            'double_double_player': ModelConfig(
                name='double_double_player',
                type='classification',
                model_class='DoubleDoubleAdvancedModel',
                feature_engineer_class='DoubleDoubleFeatureEngineer',
                target_column='double_double',
                weight=0.7,  # Menor peso por menor volumen de casos positivos
                metrics=['accuracy', 'precision', 'recall', 'f1', 'auc_roc'],
                min_samples=1000
            ),
            'teams_points': ModelConfig(
                name='teams_points',
                type='regression',
                model_class='TeamPointsModel',
                feature_engineer_class='TeamPointsFeatureEngineer', 
                target_column='PTS',
                weight=1.0,
                metrics=['mae', 'rmse', 'r2', 'mape'],
                min_samples=1000
            ),
            'total_points': ModelConfig(
                name='total_points',
                type='regression',
                model_class='NBATotalPointsPredictor',
                feature_engineer_class='TotalPointsFeatureEngine',
                target_column='total_points',
                weight=1.0,
                metrics=['mae', 'rmse', 'r2', 'mape'],
                min_samples=1000
            ),
            'is_win': ModelConfig(
                name='is_win',
                type='classification',
                model_class='IsWinModel',
                feature_engineer_class='IsWinFeatureEngineer',
                target_column='is_win', 
                weight=1.0,
                metrics=['accuracy', 'precision', 'recall', 'f1', 'auc_roc'],
                min_samples=1000
            )
        }
    
    def _create_meta_learners_config(self) -> Dict[str, Any]:
        """Configuración de meta-learners"""
        return {
            'regression': {
                'xgboost': {
                    'n_estimators': 200,
                    'max_depth': 4,
                    'learning_rate': 0.05,
                    'subsample': 0.8,
                    'colsample_bytree': 0.8,
                    'random_state': 42
                },
                'lightgbm': {
                    'n_estimators': 200,
                    'max_depth': 4,
                    'learning_rate': 0.05,
                    'subsample': 0.8,
                    'feature_fraction': 0.8,
                    'random_state': 42
                },
                'ridge': {
                    'alpha': 1.0,
                    'random_state': 42
                },
                'random_forest': {
                    'n_estimators': 100,
                    'max_depth': 6,
                    'min_samples_split': 10,
                    'random_state': 42
                }
            },
            'classification': {
                'xgboost': {
                    'n_estimators': 200,
                    'max_depth': 4,
                    'learning_rate': 0.05,
                    'subsample': 0.8,
                    'colsample_bytree': 0.8,
                    'random_state': 42,
                    'objective': 'binary:logistic'
                },
                'lightgbm': {
                    'n_estimators': 200,
                    'max_depth': 4,
                    'learning_rate': 0.05,
                    'subsample': 0.8,
                    'feature_fraction': 0.8,
                    'random_state': 42,
                    'objective': 'binary'
                },
                'logistic': {
                    'C': 1.0,
                    'random_state': 42,
                    'max_iter': 1000
                },
                'random_forest': {
                    'n_estimators': 100,
                    'max_depth': 6,
                    'min_samples_split': 10,
                    'random_state': 42
                }
            }
        }
    
    def get_optuna_directions(self) -> Dict[str, str]:
        """Obtener direcciones de optimización por métrica"""
        return {
            'mae': 'minimize',
            'rmse': 'minimize', 
            'mape': 'minimize',
            'r2': 'maximize',
            'accuracy': 'maximize',
            'precision': 'maximize',
            'recall': 'maximize',
            'f1': 'maximize',
            'auc_roc': 'maximize'
        }
    
    def create_optuna_study(self, study_name: str = None, 
                           metric: str = 'mae') -> optuna.Study:
        """Crear estudio de Optuna"""
        if study_name is None:
            study_name = f"{self.optuna_config['study_name']}_{metric}"
        
        direction = self.get_optuna_directions().get(metric, 'minimize')
        
        # Configurar sampler
        if self.optuna_config['sampler'] == 'TPE':
            sampler = optuna.samplers.TPESampler(
                seed=42,
                n_startup_trials=10
            )
        else:
            sampler = optuna.samplers.RandomSampler(seed=42)
        
        # Configurar pruner
        if self.optuna_config['pruner'] == 'MedianPruner':
            pruner = optuna.pruners.MedianPruner(
                n_startup_trials=5,
                n_warmup_steps=10
            )
        else:
            pruner = optuna.pruners.NopPruner()
        
        study = optuna.create_study(
            study_name=study_name,
            direction=direction,
            sampler=sampler,
            pruner=pruner
        )
        
        return study
    
    def create_time_series_split(self, n_splits: int = None) -> TimeSeriesSplit:
        """Crear TimeSeriesSplit para validación cronológica"""
        if n_splits is None:
            n_splits = self.kfold_config['n_splits']
        
        return TimeSeriesSplit(
            n_splits=n_splits,
            gap=self.temporal_validation['gap_days']
        )
    
    def get_enabled_models(self) -> Dict[str, ModelConfig]:
        """Obtener solo modelos habilitados"""
        return {
            name: config for name, config in self.models_config.items()
            if config.enabled
        }
    
    def get_models_by_type(self, model_type: str) -> Dict[str, ModelConfig]:
        """Obtener modelos por tipo (regression/classification)"""
        return {
            name: config for name, config in self.get_enabled_models().items()
            if config.type == model_type
        }
    
    def update_model_weight(self, model_name: str, weight: float):
        """Actualizar peso de un modelo"""
        if model_name in self.models_config:
            self.models_config[model_name].weight = weight
    
    def disable_model(self, model_name: str):
        """Deshabilitar un modelo"""
        if model_name in self.models_config:
            self.models_config[model_name].enabled = False
    
    def get_metrics_for_type(self, model_type: str) -> List[str]:
        """Obtener métricas por tipo de modelo"""
        if model_type == 'regression':
            return ['mae', 'rmse', 'r2', 'mape']
        else:  # classification
            return ['accuracy', 'precision', 'recall', 'f1', 'auc_roc'] 