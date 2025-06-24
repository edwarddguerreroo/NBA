"""
Modelo XGBoost para Predicción de Puntos NBA
===========================================

Modelo avanzado con optimización bayesiana, validación cruzada y early stopping
para predecir puntos que anotará un jugador en su próximo partido.

Arquitectura:
- XGBoost con hiperparámetros optimizados
- Validación cruzada temporal
- Optimización bayesiana con Optuna
- Early stopping para prevenir overfitting
- Métricas de evaluación robustas
- NUEVO: Stacking ensemble con múltiples modelos ML/DL
"""

import json
import logging
import os
import warnings
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
import platform
import threading
import time

import joblib
import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from optuna.samplers import TPESampler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import (
    RandomForestRegressor, 
    ExtraTreesRegressor, 
    GradientBoostingRegressor,
    VotingRegressor,
    StackingRegressor,
    HistGradientBoostingRegressor
)
from sklearn.linear_model import (
    Ridge, 
    Lasso, 
    ElasticNet, 
    LinearRegression,
    BayesianRidge
)
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor

# Importar el feature engineer
from .features_pts import PointsFeatureEngineer

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


class TimeoutError(Exception):
    """Excepción personalizada para timeouts"""
    pass


def timeout_wrapper(func, timeout_seconds, *args, **kwargs):
    """
    Wrapper para ejecutar función con timeout compatible con Windows.
    
    Args:
        func: Función a ejecutar
        timeout_seconds: Tiempo límite en segundos
        *args, **kwargs: Argumentos para la función
        
    Returns:
        Resultado de la función o None si hay timeout
        
    Raises:
        TimeoutError: Si la función excede el tiempo límite
    """
    result = [None]
    exception = [None]
    
    def target():
        try:
            result[0] = func(*args, **kwargs)
        except Exception as e:
            exception[0] = e
    
    thread = threading.Thread(target=target)
    thread.daemon = True
    thread.start()
    thread.join(timeout_seconds)
    
    if thread.is_alive():
        # El thread sigue corriendo, hay timeout
        logger.warning(f"Función {func.__name__} excedió timeout de {timeout_seconds}s")
        raise TimeoutError(f"Función excedió tiempo límite de {timeout_seconds} segundos")
    
    if exception[0]:
        raise exception[0]
    
    return result[0]


class StackingPTSModel:
    """
    REVOLUCIONARIO: Modelo de Stacking Ensemble para Predicción de Puntos NBA
    
    Combina múltiples algoritmos de ML/DL con:
    - Regularización avanzada en todos los modelos
    - Early stopping inteligente
    - Validación cruzada temporal estricta
    - Optimización bayesiana por modelo
    - Meta-learner adaptativo
    """
    
    def __init__(self,
                 n_trials: int = 75,
                 cv_folds: int = 5,
                 early_stopping_rounds: int = 30,
                 random_state: int = 42,
                 enable_neural_networks: bool = True,
                 enable_gpu: bool = False,
                 enable_svr: bool = True):
        """
        Inicializa el modelo de stacking ensemble.
        
        Args:
            n_trials: Número de trials para optimización bayesiana por modelo
            cv_folds: Número de folds para validación cruzada
            early_stopping_rounds: Rounds para early stopping
            random_state: Semilla para reproducibilidad
            enable_neural_networks: Si incluir redes neuronales
            enable_gpu: Si usar GPU para modelos compatibles
            enable_svr: Si incluir SVR (puede ser lento en datasets grandes)
        """
        self.n_trials = n_trials
        self.cv_folds = cv_folds
        self.early_stopping_rounds = early_stopping_rounds
        self.random_state = random_state
        self.enable_neural_networks = enable_neural_networks
        self.enable_gpu = enable_gpu
        self.enable_svr = enable_svr
        
        # Componentes principales
        self.feature_engineer = PointsFeatureEngineer()
        self.scaler = StandardScaler()
        
        # Modelos base
        self.base_models = {}
        self.trained_base_models = {}
        self.best_params_per_model = {}
        
        # Modelo de stacking
        self.stacking_model = None
        self.meta_learner = None
        
        # Métricas y resultados
        self.training_metrics = {}
        self.validation_metrics = {}
        self.cv_scores = {}
        self.feature_importance = {}
        
        # Features utilizadas
        self.selected_features = []
        self.feature_names = []
        
        # Estado del modelo
        self.is_trained = False
        
        # Configurar modelos base
        self._setup_base_models()
        
        logger.info("Modelo Stacking PTS inicializado con configuración revolucionaria")
        logger.info(f"Modelos habilitados: NN={enable_neural_networks}, SVR={enable_svr}, GPU={enable_gpu}")
    
    def _setup_base_models(self):
        """
        PARTE 1: Configuración de modelos base con regularización BALANCEADA para evitar underfitting.
        OPTIMIZADO: Solo modelos de ALTO rendimiento basado en análisis de resultados.
        """
        logger.info("Configurando modelos base ULTRA-OPTIMIZADOS con regularización balanceada...")
        
        # 1. XGBoost con regularización BALANCEADA (MAE: 1.705, Importancia: 28.0% - MAYOR peso)
        self.base_models['xgboost'] = {
            'model_class': xgb.XGBRegressor,
            'param_space': {
                'n_estimators': (200, 1200),
                'learning_rate': (0.02, 0.25),
                'max_depth': (4, 12),
                'min_child_weight': (1, 8),
                'subsample': (0.7, 0.95),
                'colsample_bytree': (0.7, 0.95),
                'reg_alpha': (0.01, 5.0),
                'reg_lambda': (0.01, 8.0),
                'gamma': (0.0, 3.0)
            },
            'fixed_params': {
                'objective': 'reg:squarederror',
                'eval_metric': 'mae',
                'random_state': self.random_state,
                'verbosity': 0,
                'n_jobs': -1
            },
            'early_stopping': True
        }
        
        # 2. LightGBM con regularización BALANCEADA (MAE: 1.705, Importancia: 18.2%)
        self.base_models['lightgbm'] = {
            'model_class': lgb.LGBMRegressor,
            'param_space': {
                'n_estimators': (200, 1200),
                'learning_rate': (0.02, 0.25),
                'max_depth': (4, 12),
                'num_leaves': (20, 200),
                'min_child_samples': (3, 30),
                'subsample': (0.7, 0.95),
                'colsample_bytree': (0.7, 0.95),
                'reg_alpha': (0.01, 5.0),
                'reg_lambda': (0.01, 8.0),
                'min_split_gain': (0.0, 0.5)
            },
            'fixed_params': {
                'objective': 'regression',
                'metric': 'mae',
                'random_state': self.random_state,
                'verbosity': -1,
                'n_jobs': -1
            },
            'early_stopping': True
        }
        
        # 3. CatBoost con regularización BALANCEADA (MAE: 1.680 - MEJOR individual, Importancia: 17.4%)
        self.base_models['catboost'] = {
            'model_class': cb.CatBoostRegressor,
            'param_space': {
                'iterations': (200, 1200),
                'learning_rate': (0.02, 0.25),
                'depth': (4, 10),
                'l2_leaf_reg': (0.5, 15),
                'border_count': (64, 255),
                'subsample': (0.7, 0.95),
                'rsm': (0.7, 0.95)
            },
            'fixed_params': {
                'loss_function': 'MAE',
                'random_seed': self.random_state,
                'verbose': False,
                'allow_writing_files': False,
                'thread_count': -1
            },
            'early_stopping': True
        }
        
        # 4. Gradient Boosting con mayor capacidad (MAE: 1.711, Importancia: 20.3%)
        self.base_models['gradient_boosting'] = {
            'model_class': GradientBoostingRegressor,
            'param_space': {
                'n_estimators': (100, 500),
                'learning_rate': (0.02, 0.25),
                'max_depth': (4, 10),
                'min_samples_split': (2, 15),
                'min_samples_leaf': (1, 8),
                'subsample': (0.7, 0.95),
                'max_features': (0.4, 0.95),
                'alpha': (0.1, 0.7)
            },
            'fixed_params': {
                'random_state': self.random_state,
                'validation_fraction': 0.1,
                'n_iter_no_change': self.early_stopping_rounds,
                'tol': 1e-5
            },
            'early_stopping': True
        }
        
        # 5. Ridge Regression (MAE: 1.963, Importancia: 12.9% - Estabilizador lineal)
        self.base_models['ridge'] = {
            'model_class': Ridge,
            'param_space': {
                'alpha': (0.01, 50.0),
                'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga']
            },
            'fixed_params': {
                'random_state': self.random_state,
                'max_iter': 3000
            },
            'early_stopping': False
        }
        
        # ELIMINADOS por bajo rendimiento:
        # - Random Forest (MAE: 1.740, Importancia: 4.4% - MUY BAJA)
        # - Extra Trees (MAE: 1.859, Importancia: 8.4%)
        # - Neural Network (MAE: 1.815, Importancia: 5.4%)
        # - Lasso (MAE: 1.961, Importancia: 13.0%)
        # - Elastic Net (MAE: 1.951, Importancia: 5.1%)
        # - Hist Gradient Boosting (MAE: 1.715, Importancia: 4.8%)
        
        logger.info(f"Configurados {len(self.base_models)} modelos base ULTRA-OPTIMIZADOS")
        logger.info("Ensemble FINAL: XGBoost, LightGBM, CatBoost, Gradient Boosting, Ridge")
        logger.info("Eliminado: Random Forest (MAE: 1.740, Importancia: 4.4%)")
    
    def _optimize_single_model(self, model_name: str, X_train, y_train, X_val, y_val) -> Dict[str, Any]:
        """
        PARTE 2: Optimización bayesiana para un modelo individual.
        
        Args:
            model_name: Nombre del modelo a optimizar
            X_train: Datos de entrenamiento
            y_train: Target de entrenamiento
            X_val: Datos de validación
            y_val: Target de validación
            
        Returns:
            Dict con mejores parámetros y score
        """
        logger.info(f"Optimizando hiperparámetros para {model_name}...")
        
        model_config = self.base_models[model_name]
        
        def objective(trial):
            # Construir parámetros del trial
            params = model_config['fixed_params'].copy()
            
            for param_name, param_range in model_config['param_space'].items():
                if isinstance(param_range, tuple) and len(param_range) == 2:
                    if isinstance(param_range[0], int):
                        params[param_name] = trial.suggest_int(param_name, *param_range)
                    elif isinstance(param_range[0], float):
                        params[param_name] = trial.suggest_float(param_name, *param_range)
                elif isinstance(param_range, list):
                    params[param_name] = trial.suggest_categorical(param_name, param_range)
            
            # Crear y entrenar modelo
            model = model_config['model_class'](**params)
            
            # Calcular pesos adaptativos
            sample_weights = self._calculate_adaptive_weights(y_train.values)
            
            try:
                # Función de entrenamiento con timeout
                def train_model():
                    # Entrenar con early stopping si es compatible
                    if model_config['early_stopping'] and hasattr(model, 'fit'):
                        if 'xgb' in model_name.lower():
                            # XGBoost con manejo de versiones diferentes y más robusto
                            try:
                                # Intentar con early_stopping_rounds (versiones más antiguas)
                                model.fit(
                                    X_train, y_train,
                                    sample_weight=sample_weights,
                                    eval_set=[(X_val, y_val)],
                                    early_stopping_rounds=self.early_stopping_rounds,
                                    verbose=False
                                )
                            except (TypeError, ValueError) as xgb_error:
                                # Si falla, usar sin early_stopping_rounds (versiones nuevas)
                                try:
                                    model.fit(
                                        X_train, y_train,
                                        sample_weight=sample_weights,
                                        eval_set=[(X_val, y_val)],
                                        verbose=False
                                    )
                                except Exception as xgb_error2:
                                    # Como último recurso, entrenar sin eval_set
                                    try:
                                        model.fit(X_train, y_train, sample_weight=sample_weights)
                                    except Exception as xgb_error3:
                                        logger.warning(f"Error entrenando XGBoost en optimización: {xgb_error3}")
                                        return float('inf')
                        elif 'lightgbm' in model_name.lower():
                            try:
                                model.fit(
                                    X_train, y_train,
                                    sample_weight=sample_weights,
                                    eval_set=[(X_val, y_val)],
                                    callbacks=[
                                        lgb.early_stopping(self.early_stopping_rounds),
                                        lgb.log_evaluation(0)
                                    ]
                                )
                            except Exception as lgb_error:
                                # Fallback sin early stopping
                                model.fit(X_train, y_train, sample_weight=sample_weights)
                        elif 'catboost' in model_name.lower():
                            try:
                                model.fit(
                                    X_train, y_train,
                                    sample_weight=sample_weights,
                                    eval_set=(X_val, y_val),
                                    early_stopping_rounds=self.early_stopping_rounds,
                                    verbose=False
                                )
                            except Exception as cb_error:
                                # Fallback sin early stopping
                                model.fit(X_train, y_train, sample_weight=sample_weights)
                        else:
                            # Para modelos con early stopping nativo (sklearn)
                            if model_name in ['neural_network']:
                                # MLPRegressor no acepta sample_weight
                                model.fit(X_train, y_train)
                            else:
                                model.fit(X_train, y_train, sample_weight=sample_weights)
                    else:
                        # Entrenamiento estándar
                        if 'svr' in model_name.lower():
                            # Para SVR, usar subset de datos si es muy grande
                            if len(X_train) > 5000:
                                subset_idx = np.random.choice(len(X_train), 5000, replace=False)
                                X_train_subset = X_train.iloc[subset_idx]
                                y_train_subset = y_train.iloc[subset_idx]
                                sample_weights_subset = sample_weights[subset_idx]
                                model.fit(X_train_subset, y_train_subset, sample_weight=sample_weights_subset)
                                return X_train_subset, y_train_subset
                            else:
                                model.fit(X_train, y_train, sample_weight=sample_weights)
                                return X_train, y_train
                        elif model_name in ['neural_network']:
                            # MLPRegressor no acepta sample_weight
                            model.fit(X_train, y_train)
                            return X_train, y_train
                        else:
                            model.fit(X_train, y_train, sample_weight=sample_weights)
                            return X_train, y_train
                
                # Ejecutar entrenamiento con timeout
                if 'svr' in model_name.lower():
                    # Timeout específico para SVR
                    train_result = timeout_wrapper(train_model, 120)  # 2 minutos
                    if train_result:
                        X_train_used, y_train_used = train_result
                    else:
                        X_train_used, y_train_used = X_train, y_train
                else:
                    # Sin timeout para otros modelos
                    train_result = train_model()
                    if train_result:
                        X_train_used, y_train_used = train_result
                    else:
                        X_train_used, y_train_used = X_train, y_train
                
                # Predecir y evaluar
                y_pred = model.predict(X_val)
                
                # Calcular score con pesos adaptativos
                val_weights = self._calculate_adaptive_weights(y_val.values)
                weighted_errors = np.abs(y_pred - y_val.values) * val_weights
                weighted_mae = np.mean(weighted_errors)
                
                # Penalizaciones REDUCIDAS para evitar underfitting
                high_scoring_mask = y_val.values >= 20
                if high_scoring_mask.sum() > 0:
                    high_scoring_errors = y_pred[high_scoring_mask] - y_val.values[high_scoring_mask]
                    underestimation_penalty = np.mean(np.maximum(-high_scoring_errors, 0)) * 0.3  # REDUCIDO de 0.5
                else:
                    underestimation_penalty = 0
                
                # Penalización por overfitting MÁS SUAVE
                train_pred = model.predict(X_train_used)
                train_mae = mean_absolute_error(y_train_used, train_pred)
                overfitting_penalty = max(0, train_mae - weighted_mae) * 0.1  # REDUCIDO de 0.2
                
                # Penalización por complejidad MÁS SUAVE
                complexity_penalty = 0
                if 'max_depth' in params:
                    complexity_penalty += (params['max_depth'] / 20.0) * 0.02  # REDUCIDO
                if 'n_estimators' in params:
                    complexity_penalty += (params['n_estimators'] / 2000.0) * 0.02  # REDUCIDO
                elif 'max_iter' in params:
                    complexity_penalty += (params['max_iter'] / 1000.0) * 0.02  # REDUCIDO
                
                final_score = weighted_mae + underestimation_penalty + overfitting_penalty + complexity_penalty
                
                return final_score
                
            except (Exception, TimeoutError) as e:
                logger.warning(f"Error en optimización de {model_name}: {e}")
                return float('inf')
        
        # Crear estudio Optuna con timeout
        study = optuna.create_study(
            direction='minimize',
            sampler=TPESampler(seed=self.random_state)
        )
        
        # SILENCIAR COMPLETAMENTE OPTUNA
        optuna.logging.set_verbosity(optuna.logging.CRITICAL)
        
        # Configurar timeout específico por modelo
        if 'svr' in model_name.lower():
            timeout_seconds = 600  # 10 minutos para SVR
            n_trials_model = min(self.n_trials, 30)  # Reducir trials para SVR
        elif 'neural_network' in model_name.lower():
            timeout_seconds = 900  # 15 minutos para redes neuronales
            n_trials_model = min(self.n_trials, 40)  # Reducir trials para NN
        else:
            timeout_seconds = 1200  # 20 minutos para otros modelos
            n_trials_model = self.n_trials
        
        # Optimizar con timeout SIN VERBOSIDAD
        try:
            study.optimize(
                objective, 
                n_trials=n_trials_model, 
                timeout=timeout_seconds,
                show_progress_bar=False,
                catch=(Exception,)
            )
        except Exception as e:
            logger.error(f"Error en optimización de {model_name}: {e}")
            # Usar parámetros por defecto si falla la optimización
            best_params = model_config['fixed_params']
            best_score = float('inf')
        else:
            best_params = {**model_config['fixed_params'], **study.best_params}
            best_score = study.best_value
        
        logger.info(f"{model_name} - Mejor MAE: {best_score:.4f}")
        
        return {
            'best_params': best_params,
            'best_score': best_score,
            'study': study if 'study' in locals() else None
        }
    
    def _train_base_models(self, X_train, y_train, X_val, y_val) -> Dict[str, Any]:
        """
        PARTE 2: Entrenamiento de todos los modelos base con optimización.
        
        Args:
            X_train: Datos de entrenamiento
            y_train: Target de entrenamiento
            X_val: Datos de validación
            y_val: Target de validación
            
        Returns:
            Dict con modelos entrenados y métricas
        """
        logger.info("Entrenando modelos base con optimización bayesiana...")
        
        trained_models = {}
        optimization_results = {}
        
        for i, model_name in enumerate(self.base_models.keys(), 1):
            logger.info(f"[{i}/{len(self.base_models)}] Procesando {model_name}...")
            
            try:
                # Optimizar hiperparámetros
                opt_result = self._optimize_single_model(
                    model_name, X_train, y_train, X_val, y_val
                )
                
                # Entrenar modelo final con mejores parámetros
                best_params = opt_result['best_params']
                model_config = self.base_models[model_name]
                final_model = model_config['model_class'](**best_params)
                
                # Calcular pesos adaptativos
                sample_weights = self._calculate_adaptive_weights(y_train.values)
                
                # Entrenar modelo final
                if model_config['early_stopping'] and hasattr(final_model, 'fit'):
                    if 'xgb' in model_name.lower():
                        # XGBoost con manejo de versiones diferentes y más robusto
                        try:
                            # Intentar con early_stopping_rounds (versiones más antiguas)
                            final_model.fit(
                                X_train, y_train,
                                sample_weight=sample_weights,
                                eval_set=[(X_val, y_val)],
                                early_stopping_rounds=self.early_stopping_rounds,
                                verbose=False
                            )
                        except (TypeError, ValueError) as xgb_error:
                            # Si falla, usar sin early_stopping_rounds (versiones nuevas)
                            try:
                                final_model.fit(
                                    X_train, y_train,
                                    sample_weight=sample_weights,
                                    eval_set=[(X_val, y_val)],
                                    verbose=False
                                )
                            except Exception as xgb_error2:
                                # Como último recurso, entrenar sin eval_set
                                try:
                                    final_model.fit(X_train, y_train, sample_weight=sample_weights)
                                except Exception as xgb_error3:
                                    logger.error(f"Error entrenando XGBoost {model_name}: {xgb_error3}")
                                    continue
                    elif 'lightgbm' in model_name.lower():
                        try:
                            final_model.fit(
                                X_train, y_train,
                                sample_weight=sample_weights,
                                eval_set=[(X_val, y_val)],
                                callbacks=[
                                    lgb.early_stopping(self.early_stopping_rounds),
                                    lgb.log_evaluation(0)
                                ]
                            )
                        except Exception as lgb_error:
                            # Fallback sin early stopping
                            final_model.fit(X_train, y_train, sample_weight=sample_weights)
                    elif 'catboost' in model_name.lower():
                        try:
                            final_model.fit(
                                X_train, y_train,
                                sample_weight=sample_weights,
                                eval_set=(X_val, y_val),
                                early_stopping_rounds=self.early_stopping_rounds,
                                verbose=False
                            )
                        except Exception as cb_error:
                            # Fallback sin early stopping
                            final_model.fit(X_train, y_train, sample_weight=sample_weights)
                    else:
                        # Para modelos sklearn con early stopping
                        if model_name in ['neural_network']:
                            # MLPRegressor no acepta sample_weight
                            final_model.fit(X_train, y_train)
                        else:
                            final_model.fit(X_train, y_train, sample_weight=sample_weights)
                else:
                    # Entrenamiento estándar
                    if model_name in ['neural_network']:
                        # MLPRegressor no acepta sample_weight
                        final_model.fit(X_train, y_train)
                    else:
                        final_model.fit(X_train, y_train, sample_weight=sample_weights)
                
                # Evaluar modelo
                train_pred = final_model.predict(X_train)
                val_pred = final_model.predict(X_val)
                
                model_metrics = {
                    'train_mae': mean_absolute_error(y_train, train_pred),
                    'val_mae': mean_absolute_error(y_val, val_pred),
                    'train_rmse': np.sqrt(mean_squared_error(y_train, train_pred)),
                    'val_rmse': np.sqrt(mean_squared_error(y_val, val_pred)),
                    'train_r2': r2_score(y_train, train_pred),
                    'val_r2': r2_score(y_val, val_pred)
                }
                
                trained_models[model_name] = final_model
                optimization_results[model_name] = {
                    'best_params': best_params,
                    'best_score': opt_result['best_score'],
                    'metrics': model_metrics
                }
                
                logger.info(f"{model_name} completado - MAE: {model_metrics['val_mae']:.3f}")
                
            except Exception as e:
                logger.error(f"Error entrenando {model_name}: {e}")
                continue
        
        self.trained_base_models = trained_models
        self.best_params_per_model = {
            name: result['best_params'] 
            for name, result in optimization_results.items()
        }
        
        logger.info(f"Entrenamiento completado: {len(trained_models)}/{len(self.base_models)} modelos exitosos")
        
        return {
            'trained_models': trained_models,
            'optimization_results': optimization_results
        }
    
    def _setup_stacking_model(self):
        """
        PARTE 2: Configuración del modelo de stacking con meta-learner optimizado.
        """
        logger.info("Configurando modelo de stacking con meta-learner adaptativo...")
        
        if not self.trained_base_models:
            raise ValueError("Modelos base no entrenados. Ejecutar _train_base_models primero.")
        
        # Preparar estimadores base para stacking
        base_estimators = [
            (name, model) for name, model in self.trained_base_models.items()
        ]
        
        # Meta-learner: Ridge con regularización BALANCEADA para evitar underfitting
        meta_learner = Ridge(
            alpha=0.5,  # REDUCIDO de 1.0 - menos regularización
            random_state=self.random_state,
            max_iter=3000  # AUMENTADO para mejor convergencia
        )
        
        # Crear modelo de stacking con configuración CORREGIDA
        # Usar un número específico de folds en lugar de TimeSeriesSplit
        self.stacking_model = StackingRegressor(
            estimators=base_estimators,
            final_estimator=meta_learner,
            cv=3,  # Usar número fijo en lugar de TimeSeriesSplit
            n_jobs=1,  # CAMBIADO: usar 1 job para evitar problemas de paralelización
            passthrough=False  # No pasar features originales al meta-learner
        )
        
        logger.info(f"Stacking configurado con {len(base_estimators)} modelos base")
    
    def _temporal_cross_validate_stacking(self, X, y, df_with_dates) -> Dict[str, float]:
        """
        PARTE 2: Validación cruzada temporal específica para stacking ensemble.
        
        Args:
            X: Features
            y: Target
            df_with_dates: DataFrame con fechas para validación temporal
            
        Returns:
            Dict con métricas de validación cruzada
        """
        logger.info("Iniciando validación cruzada temporal para stacking...")
        
        # Asegurar que las fechas estén en formato datetime
        dates = pd.to_datetime(df_with_dates['Date'])
        
        # Crear splits temporales basados en fechas
        unique_dates = sorted(dates.unique())
        n_dates = len(unique_dates)
        
        # Dividir en períodos temporales
        fold_size = n_dates // self.cv_folds
        
        cv_results = {
            'mae_scores': [],
            'rmse_scores': [],
            'r2_scores': [],
            'fold_details': [],
            'base_model_scores': {name: [] for name in self.trained_base_models.keys()}
        }
        
        for fold in range(self.cv_folds):
            # Definir fechas de entrenamiento y validación
            val_start_idx = fold * fold_size
            val_end_idx = min((fold + 1) * fold_size, n_dates)
            
            if val_end_idx >= n_dates:
                val_end_idx = n_dates
            
            # Fechas de validación
            val_dates = unique_dates[val_start_idx:val_end_idx]
            
            # Fechas de entrenamiento (solo fechas anteriores)
            if fold == 0:
                continue  # Skip primer fold si no hay datos de entrenamiento
            
            train_dates = unique_dates[:val_start_idx]
            
            # Crear máscaras
            train_mask = dates.isin(train_dates)
            val_mask = dates.isin(val_dates)
            
            if train_mask.sum() == 0 or val_mask.sum() == 0:
                continue
            
            # Dividir datos
            X_train_fold = X[train_mask]
            y_train_fold = y[train_mask]
            X_val_fold = X[val_mask]
            y_val_fold = y[val_mask]
            
            # Entrenar modelos base para este fold
            fold_base_models = {}
            for model_name, model_config in self.base_models.items():
                try:
                    # Usar mejores parámetros encontrados
                    if model_name in self.best_params_per_model:
                        params = self.best_params_per_model[model_name]
                    else:
                        params = model_config['fixed_params']
                    
                    # Crear modelo con manejo de errores robusto
                    try:
                        fold_model = model_config['model_class'](**params)
                    except Exception as model_creation_error:
                        logger.warning(f"Error creando modelo {model_name} fold {fold}: {model_creation_error}")
                        continue
                    
                    sample_weights = self._calculate_adaptive_weights(y_train_fold.values)
                    
                    # Entrenar con o sin sample_weight según el modelo
                    try:
                        if model_name in ['neural_network']:
                            # MLPRegressor no acepta sample_weight
                            fold_model.fit(X_train_fold, y_train_fold)
                        elif 'xgb' in model_name.lower():
                            # XGBoost con manejo de versiones diferentes y más robusto
                            try:
                                # Intentar con early_stopping_rounds (versiones más antiguas)
                                fold_model.fit(
                                    X_train_fold, y_train_fold,
                                    sample_weight=sample_weights,
                                    eval_set=[(X_val_fold, y_val_fold)],
                                    early_stopping_rounds=self.early_stopping_rounds,
                                    verbose=False
                                )
                            except (TypeError, ValueError) as xgb_error:
                                # Si falla, usar sin early_stopping_rounds (versiones nuevas)
                                try:
                                    fold_model.fit(
                                        X_train_fold, y_train_fold,
                                        sample_weight=sample_weights,
                                        eval_set=[(X_val_fold, y_val_fold)],
                                        verbose=False
                                    )
                                except Exception as xgb_error2:
                                    # Como último recurso, entrenar sin eval_set
                                    try:
                                        fold_model.fit(X_train_fold, y_train_fold, sample_weight=sample_weights)
                                    except Exception as xgb_error3:
                                        logger.warning(f"Error entrenando XGBoost {model_name} fold {fold}: {xgb_error3}")
                                        continue
                        elif 'lightgbm' in model_name.lower():
                            try:
                                fold_model.fit(
                                    X_train_fold, y_train_fold,
                                    sample_weight=sample_weights,
                                    eval_set=[(X_val_fold, y_val_fold)],
                                    callbacks=[
                                        lgb.early_stopping(self.early_stopping_rounds),
                                        lgb.log_evaluation(0)
                                    ]
                                )
                            except Exception as lgb_error:
                                # Fallback sin early stopping
                                fold_model.fit(X_train_fold, y_train_fold, sample_weight=sample_weights)
                        elif 'catboost' in model_name.lower():
                            try:
                                fold_model.fit(
                                    X_train_fold, y_train_fold,
                                    sample_weight=sample_weights,
                                    eval_set=(X_val_fold, y_val_fold),
                                    early_stopping_rounds=self.early_stopping_rounds,
                                    verbose=False
                                )
                            except Exception as cb_error:
                                # Fallback sin early stopping
                                fold_model.fit(X_train_fold, y_train_fold, sample_weight=sample_weights)
                        else:
                            # Otros modelos sí aceptan sample_weight
                            fold_model.fit(X_train_fold, y_train_fold, sample_weight=sample_weights)
                    except Exception as training_error:
                        logger.warning(f"Error entrenando modelo {model_name} fold {fold}: {training_error}")
                        continue
                    
                    # Verificar que el modelo se entrenó correctamente antes de agregarlo
                    if hasattr(fold_model, 'predict'):
                        fold_base_models[model_name] = fold_model
                    else:
                        logger.warning(f"Modelo {model_name} fold {fold} no tiene método predict")
                        continue
                    
                    # Evaluar modelo base individual con manejo robusto de errores
                    try:
                        # Verificar que los datos de validación son válidos
                        if len(X_val_fold) == 0 or len(y_val_fold) == 0:
                            logger.warning(f"Datos de validación vacíos para {model_name} fold {fold}")
                            continue
                        
                        # Verificar que no hay NaN en los datos
                        if X_val_fold.isnull().any().any() or y_val_fold.isnull().any():
                            logger.warning(f"Datos de validación con NaN para {model_name} fold {fold}")
                            continue
                        
                        # Realizar predicción con manejo de errores específico
                        base_pred = fold_model.predict(X_val_fold)
                        
                        # Verificar que las predicciones son válidas
                        if base_pred is None or len(base_pred) == 0:
                            logger.warning(f"Predicciones vacías para {model_name} fold {fold}")
                            continue
                        
                        # Verificar que no hay NaN en las predicciones
                        if np.isnan(base_pred).any():
                            logger.warning(f"Predicciones con NaN para {model_name} fold {fold}")
                            continue
                        
                        # Calcular MAE solo si todo es válido
                        base_mae = mean_absolute_error(y_val_fold, base_pred)
                        cv_results['base_model_scores'][model_name].append(base_mae)
                        
                    except Exception as prediction_error:
                        logger.warning(f"Error prediciendo con modelo {model_name} fold {fold}: {str(prediction_error)}")
                        # Remover el modelo si no puede predecir
                        if model_name in fold_base_models:
                            del fold_base_models[model_name]
                        continue
                    
                except Exception as e:
                    logger.warning(f"Error general en modelo {model_name} fold {fold}: {str(e)}")
                    continue
            
            # Crear stacking temporal para este fold con configuración CORREGIDA
            if len(fold_base_models) >= 2:
                try:
                    fold_estimators = [(name, model) for name, model in fold_base_models.items()]
                    fold_meta_learner = Ridge(alpha=1.0, random_state=self.random_state)
                    
                    fold_stacking = StackingRegressor(
                        estimators=fold_estimators,
                        final_estimator=fold_meta_learner,
                        cv=3,  # Usar número fijo
                        n_jobs=1  # CAMBIADO: usar 1 job para evitar problemas
                    )
                    
                    # Entrenar stacking
                    fold_stacking.fit(X_train_fold, y_train_fold)
                    
                    # Predecir con stacking
                    y_pred_fold = fold_stacking.predict(X_val_fold)
            
                    # Calcular métricas
                    mae = mean_absolute_error(y_val_fold, y_pred_fold)
                    rmse = np.sqrt(mean_squared_error(y_val_fold, y_pred_fold))
                    r2 = r2_score(y_val_fold, y_pred_fold)
                    
                    # Métricas de precisión por rango
                    accuracy_1pt = np.mean(np.abs(y_pred_fold - y_val_fold) <= 1) * 100
                    accuracy_2pts = np.mean(np.abs(y_pred_fold - y_val_fold) <= 2) * 100
                    accuracy_3pts = np.mean(np.abs(y_pred_fold - y_val_fold) <= 3) * 100
                    
                    cv_results['mae_scores'].append(mae)
                    cv_results['rmse_scores'].append(rmse)
                    cv_results['r2_scores'].append(r2)
                    
                    fold_detail = {
                        'fold': fold + 1,
                        'mae': mae,
                        'rmse': rmse,
                        'r2': r2,
                        'accuracy_1pt': accuracy_1pt,
                        'accuracy_2pts': accuracy_2pts,
                        'accuracy_3pts': accuracy_3pts,
                        'train_size': len(X_train_fold),
                        'val_size': len(X_val_fold),
                        'train_date_range': f"{train_dates[0]} a {train_dates[-1]}",
                        'val_date_range': f"{val_dates[0]} a {val_dates[-1]}",
                        'n_base_models': len(fold_base_models)
                    }
                    
                    cv_results['fold_details'].append(fold_detail)
            
                    # Solo mostrar progreso cada 2 folds
                    if fold % 2 == 0 or fold == self.cv_folds - 1:
                        logger.info(f"CV Fold {fold + 1}/{self.cv_folds}: MAE={mae:.3f}, R²={r2:.3f}, Modelos={len(fold_base_models)}")
                
                except Exception as stacking_error:
                    logger.warning(f"Error en stacking fold {fold}: {stacking_error}")
                    logger.info(f"Fold {fold + 1} saltado - {len(fold_base_models)} modelos disponibles")
                    continue
            else:
                logger.warning(f"Fold {fold + 1} saltado - solo {len(fold_base_models)} modelos disponibles (mínimo 2)")
        
        # Calcular estadísticas finales
        if cv_results['mae_scores']:
            cv_results.update({
                'mae_mean': np.mean(cv_results['mae_scores']),
                'mae_std': np.std(cv_results['mae_scores']),
                'rmse_mean': np.mean(cv_results['rmse_scores']),
                'rmse_std': np.std(cv_results['rmse_scores']),
                'r2_mean': np.mean(cv_results['r2_scores']),
                'r2_std': np.std(cv_results['r2_scores']),
                'n_folds': len(cv_results['mae_scores'])
            })
            
            # Estadísticas de modelos base
            for model_name, scores in cv_results['base_model_scores'].items():
                if scores:
                    cv_results[f'{model_name}_mae_mean'] = np.mean(scores)
                    cv_results[f'{model_name}_mae_std'] = np.std(scores)
        
        logger.info(f"Validación cruzada completada: {len(cv_results['mae_scores'])} folds válidos")
        
        return cv_results
    
    def _prepare_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """
        Prepara las características usando el feature engineer especializado.
        
        Args:
            df: DataFrame con datos NBA
            
        Returns:
            Tuple[pd.DataFrame, List[str]]: DataFrame procesado y lista de features
        """
        logger.info("Preparando características especializadas para predicción de puntos...")
        
        # Trabajar con una copia
        df_features = df.copy()
        
        # Generar todas las características
        features = self.feature_engineer.generate_all_features(df_features)
        
        if not features:
            raise ValueError("No se pudieron generar características válidas")
        
        # Obtener conjunto óptimo de características
        optimal_features = self.feature_engineer.get_optimal_feature_set(
            df_features, max_features=35
        )
        
        logger.info(f"Características seleccionadas: {len(optimal_features)}")
        logger.info(f"Top 10 características: {optimal_features[:10]}")
        
        # Verificar que todas las features existen en el DataFrame
        missing_features = [f for f in optimal_features if f not in df_features.columns]
        if missing_features:
            logger.error(f"Features faltantes en DataFrame procesado: {missing_features}")
            # Filtrar solo las features que existen
            optimal_features = [f for f in optimal_features if f in df_features.columns]
            logger.info(f"Features corregidas: {len(optimal_features)}")
        
        return df_features, optimal_features
    
    def _temporal_split(self, df: pd.DataFrame, test_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        División temporal de datos para evitar data leakage.
        
        Args:
            df: DataFrame con datos
            test_size: Proporción de datos para prueba
            
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: DataFrames de train y test
        """
        if 'Date' not in df.columns:
            raise ValueError("Columna 'Date' requerida para división temporal")
        
        # Ordenar por fecha
        df_sorted = df.sort_values('Date').reset_index(drop=True)
        
        # División temporal
        split_idx = int(len(df_sorted) * (1 - test_size))
        train_df = df_sorted.iloc[:split_idx].copy()
        test_df = df_sorted.iloc[split_idx:].copy()
        
        logger.info(f"División temporal: {len(train_df)} entrenamiento, {len(test_df)} prueba")
        logger.info(f"Fecha corte: {train_df['Date'].max()} -> {test_df['Date'].min()}")
        
        return train_df, test_df
    
    def train(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        PARTE 3: Entrenamiento completo del modelo de stacking ensemble.
        
        Args:
            df: DataFrame con datos NBA
            
        Returns:
            Dict[str, float]: Métricas de validación
        """
        logger.info("Iniciando entrenamiento completo del modelo Stacking PTS...")
        
        # Verificar target
        if 'PTS' not in df.columns:
            raise ValueError("Columna 'PTS' (target) no encontrada en el dataset")
        
        # Verificar y ordenar datos cronológicamente
        if 'Date' in df.columns:
            logger.info("Verificando orden cronológico de datos de entrada...")
            df['Date'] = pd.to_datetime(df['Date'])
            if not df['Date'].is_monotonic_increasing:
                logger.info("Ordenando datos cronológicamente...")
                df = df.sort_values(['Player', 'Date']).reset_index(drop=True)
            logger.info("Datos en orden cronológico confirmado")
        else:
            logger.warning("Columna 'Date' no encontrada - no se puede verificar orden cronológico")
        
        # Preparar características
        df_features, feature_names = self._prepare_features(df)
        self.selected_features = feature_names
        self.feature_names = feature_names
        
        # División temporal
        train_df, test_df = self._temporal_split(df_features)
        
        # Preparar datos de entrenamiento
        X_train = train_df[feature_names]
        y_train = train_df['PTS']
        X_test = test_df[feature_names]
        y_test = test_df['PTS']
        
        # Verificar datos válidos
        X_train = X_train.fillna(0)
        X_test = X_test.fillna(0)
        
        logger.info(f"Datos de entrenamiento: {X_train.shape}")
        logger.info(f"Datos de prueba: {X_test.shape}")
        logger.info(f"Target stats - Media: {y_train.mean():.2f}, Std: {y_train.std():.2f}")
        
        # División para optimización de hiperparámetros
        split_idx = int(len(X_train) * 0.8)
        X_opt_train = X_train.iloc[:split_idx]
        y_opt_train = y_train.iloc[:split_idx]
        X_opt_val = X_train.iloc[split_idx:]
        y_opt_val = y_train.iloc[split_idx:]
        
        # PASO 1: Entrenar modelos base con optimización bayesiana
        logger.info("PASO 1/5: Entrenando modelos base...")
        base_results = self._train_base_models(X_opt_train, y_opt_train, X_opt_val, y_opt_val)
        
        # PASO 2: Configurar modelo de stacking
        logger.info("PASO 2/5: Configurando stacking...")
        self._setup_stacking_model()
        
        # PASO 3: Validación cruzada temporal del stacking
        logger.info("PASO 3/5: Validación cruzada temporal...")
        cv_results = self._temporal_cross_validate_stacking(X_train, y_train, train_df)
        self.cv_scores = cv_results
        
        # PASO 4: Entrenamiento final del stacking con todos los datos
        logger.info("PASO 4/5: Entrenamiento final del stacking...")
        
        # Calcular pesos adaptativos para entrenamiento final
        final_weights = self._calculate_adaptive_weights(y_train.values)
        
        # Entrenar stacking final con manejo robusto de errores
        try:
            # Intentar con sample_weight primero
            self.stacking_model.fit(X_train, y_train, sample_weight=final_weights)
        except TypeError:
            # Si el stacking no acepta sample_weight, entrenar sin él
            logger.warning("StackingRegressor no acepta sample_weight, entrenando sin pesos")
            try:
                self.stacking_model.fit(X_train, y_train)
            except Exception as stacking_error:
                logger.error(f"Error en entrenamiento de stacking: {stacking_error}")
                # Como último recurso, crear un stacking más simple
                logger.info("Creando stacking simplificado como respaldo...")
                
                # Crear un stacking más simple sin paralelización
                simple_meta_learner = Ridge(alpha=1.0, random_state=self.random_state)
                simple_estimators = [(name, model) for name, model in self.trained_base_models.items()]
                
                self.stacking_model = StackingRegressor(
                    estimators=simple_estimators,
                    final_estimator=simple_meta_learner,
                    cv=2,  # Menos folds para ser más simple
                    n_jobs=1,  # Sin paralelización
                    passthrough=False
                )
                
                # Entrenar el stacking simplificado
                self.stacking_model.fit(X_train, y_train)
                logger.info("Stacking simplificado entrenado exitosamente")
        except Exception as general_error:
            logger.error(f"Error general en stacking: {general_error}")
            raise RuntimeError(f"No se pudo entrenar el modelo de stacking: {general_error}")
        
        # PASO 5: Evaluación final
        logger.info("PASO 5/5: Evaluación final...")
        
        # Predicciones de entrenamiento y prueba
        train_pred = self.stacking_model.predict(X_train)
        test_pred = self.stacking_model.predict(X_test)
        
        # Aplicar corrección de sesgo a predicciones de prueba
        test_pred_corrected = self._apply_bias_correction(test_pred, test_df)
        
        # Métricas de entrenamiento
        self.training_metrics = {
            'mae': mean_absolute_error(y_train, train_pred),
            'rmse': np.sqrt(mean_squared_error(y_train, train_pred)),
            'r2': r2_score(y_train, train_pred)
        }
        
        # Métricas de validación (con corrección de sesgo)
        self.validation_metrics = {
            'mae': mean_absolute_error(y_test, test_pred_corrected),
            'rmse': np.sqrt(mean_squared_error(y_test, test_pred_corrected)),
            'r2': r2_score(y_test, test_pred_corrected),
            'mae_raw': mean_absolute_error(y_test, test_pred),  # Sin corrección
            'rmse_raw': np.sqrt(mean_squared_error(y_test, test_pred)),
            'r2_raw': r2_score(y_test, test_pred)
        }
        
        # Métricas de evaluación por rangos de puntuación
        range_metrics = self._calculate_range_metrics(y_test.values, test_pred_corrected)
        self.validation_metrics.update(range_metrics)
        
        # Calcular accuracy aproximado
        accuracy_1pt = np.mean(np.abs(test_pred_corrected - y_test) <= 1) * 100
        accuracy_2pts = np.mean(np.abs(test_pred_corrected - y_test) <= 2) * 100
        accuracy_3pts = np.mean(np.abs(test_pred_corrected - y_test) <= 3) * 100
        
        # Agregar accuracies a métricas de validación
        self.validation_metrics.update({
            'accuracy_1pt': accuracy_1pt,
            'accuracy_2pts': accuracy_2pts,
            'accuracy_3pts': accuracy_3pts
        })
        
        # Feature importance del stacking (meta-learner)
        if hasattr(self.stacking_model.final_estimator_, 'coef_'):
            # Para modelos lineales como Ridge
            try:
                # Usar estimators (tuplas originales) en lugar de estimators_ (modelos entrenados)
                base_model_names = [name for name, _ in self.stacking_model.estimators]
                meta_importance = dict(zip(
                    base_model_names,
                    np.abs(self.stacking_model.final_estimator_.coef_)
                ))
                self.feature_importance = meta_importance
            except Exception as importance_error:
                logger.warning(f"No se pudo calcular feature importance del meta-learner: {importance_error}")
                # Usar nombres de modelos base como fallback
                self.feature_importance = {name: 1.0 for name in self.trained_base_models.keys()}
        
        # Marcar como entrenado
        self.is_trained = True
        
        # Mostrar resultados FINALES
        logger.info("=" * 60)
        logger.info("ENTRENAMIENTO COMPLETADO EXITOSAMENTE")
        logger.info("=" * 60)
        logger.info(f"MAE: {self.validation_metrics['mae']:.4f}")
        logger.info(f"RMSE: {self.validation_metrics['rmse']:.4f}")
        logger.info(f"R²: {self.validation_metrics['r2']:.4f}")
        logger.info(f"Accuracy ±1pt: {accuracy_1pt:.1f}%")
        logger.info(f"Accuracy ±2pts: {accuracy_2pts:.1f}%")
        logger.info(f"Accuracy ±3pts: {accuracy_3pts:.1f}%")
        logger.info("=" * 60)
        
        return self.validation_metrics
    
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        PARTE 3: Realizar predicciones con el modelo de stacking entrenado.
        
        Args:
            df: DataFrame con datos para predicción
            
        Returns:
            np.ndarray: Predicciones de puntos
        """
        if not self.is_trained or self.stacking_model is None:
            raise ValueError("Modelo no entrenado. Ejecutar train() primero.")
        
        # Preparar características
        df_features, _ = self._prepare_features(df)
        X = df_features[self.selected_features].fillna(0)
        
        # Predecir con stacking
        predictions = self.stacking_model.predict(X)
        
        # Aplicar corrección de sesgo
        corrected_predictions = self._apply_bias_correction(predictions, df_features)
        
        return corrected_predictions
    
    def predict_with_base_models(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        PARTE 3: Realizar predicciones con todos los modelos base y stacking.
        
        Args:
            df: DataFrame con datos para predicción
            
        Returns:
            Dict con predicciones de cada modelo base y stacking
        """
        if not self.is_trained:
            raise ValueError("Modelo no entrenado. Ejecutar train() primero.")
        
        # Preparar características
        df_features, _ = self._prepare_features(df)
        X = df_features[self.selected_features].fillna(0)
        
        predictions = {}
        
        # Predicciones de modelos base
        for model_name, model in self.trained_base_models.items():
            try:
                base_pred = model.predict(X)
                predictions[model_name] = base_pred
            except Exception as e:
                logger.warning(f"Error prediciendo con {model_name}: {e}")
                predictions[model_name] = np.zeros(len(X))
        
        # Predicción del stacking
        if self.stacking_model is not None:
            stacking_pred = self.stacking_model.predict(X)
            stacking_pred_corrected = self._apply_bias_correction(stacking_pred, df_features)
            predictions['stacking'] = stacking_pred
            predictions['stacking_corrected'] = stacking_pred_corrected
        
        return predictions
    
    def get_model_performance_summary(self) -> Dict[str, Any]:
        """
        PARTE 3: Obtener resumen de rendimiento de todos los modelos.
        
        Returns:
            Dict con resumen de rendimiento
        """
        if not self.is_trained:
            raise ValueError("Modelo no entrenado.")
        
        summary = {
            'stacking_performance': {
                'training': self.training_metrics,
                'validation': self.validation_metrics,
                'cross_validation': self.cv_scores
            },
            'base_models_performance': {},
            'model_importance_in_stacking': self.feature_importance,
            'configuration': {
                'n_base_models': len(self.trained_base_models),
                'base_model_names': list(self.trained_base_models.keys()),
                'n_features': len(self.selected_features),
                'n_trials_per_model': self.n_trials,
                'cv_folds': self.cv_folds,
                'early_stopping_rounds': self.early_stopping_rounds
            }
        }
        
        # Agregar rendimiento individual de modelos base si está disponible
        if hasattr(self, 'cv_scores') and self.cv_scores:
            for model_name in self.trained_base_models.keys():
                if f'{model_name}_mae_mean' in self.cv_scores:
                    summary['base_models_performance'][model_name] = {
                        'cv_mae_mean': self.cv_scores[f'{model_name}_mae_mean'],
                        'cv_mae_std': self.cv_scores[f'{model_name}_mae_std']
                    }
        
        return summary
    
    def save_model(self, filepath: str):
        """
        PARTE 3: Guardar modelo completo de stacking.
        
        Args:
            filepath: Ruta donde guardar el modelo
        """
        if not self.is_trained:
            raise ValueError("Modelo no entrenado.")
        
        # Crear directorio si no existe
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        joblib.dump(self.stacking_model, filepath)
        logger.info(f"Modelo de stacking (solo objeto) guardado en: {filepath}")
        
        # Guardar metadata por separado si es necesario para debugging
        metadata_path = filepath.replace('.pkl', '_metadata.pkl')
        model_metadata = {
            'trained_base_models': self.trained_base_models,
            'best_params_per_model': self.best_params_per_model,
            'feature_engineer': self.feature_engineer,
            'scaler': self.scaler,
            'selected_features': self.selected_features,
            'feature_importance': self.feature_importance,
            'training_metrics': self.training_metrics,
            'validation_metrics': self.validation_metrics,
            'cv_scores': self.cv_scores,
            'model_config': {
                'n_trials': self.n_trials,
                'cv_folds': self.cv_folds,
                'early_stopping_rounds': self.early_stopping_rounds,
                'random_state': self.random_state,
                'enable_neural_networks': self.enable_neural_networks,
                'enable_gpu': self.enable_gpu,
                'enable_svr': self.enable_svr
            },
            'is_trained': self.is_trained
        }
        joblib.dump(model_metadata, metadata_path)
        logger.info(f"Metadata guardada en: {metadata_path}")
    
    def load_model(self, filepath: str):
        """
        PARTE 3: Cargar modelo completo de stacking.
        
        Args:
            filepath: Ruta del modelo a cargar
        """
        try:
            # Intentar cargar modelo directo (nuevo formato)
            self.stacking_model = joblib.load(filepath)
            self.is_trained = True
            logger.info(f"Modelo de stacking (objeto directo) cargado desde: {filepath}")
            
            # Intentar cargar metadata si existe
            metadata_path = filepath.replace('.pkl', '_metadata.pkl')
            if os.path.exists(metadata_path):
                metadata = joblib.load(metadata_path)
                self.trained_base_models = metadata.get('trained_base_models', {})
                self.best_params_per_model = metadata.get('best_params_per_model', {})
                self.feature_engineer = metadata.get('feature_engineer', PointsFeatureEngineer())
                self.scaler = metadata.get('scaler', StandardScaler())
                self.selected_features = metadata.get('selected_features', [])
                self.feature_importance = metadata.get('feature_importance', {})
                self.training_metrics = metadata.get('training_metrics', {})
                self.validation_metrics = metadata.get('validation_metrics', {})
                self.cv_scores = metadata.get('cv_scores', {})
                
                # Restaurar configuración
                config = metadata.get('model_config', {})
                self.n_trials = config.get('n_trials', self.n_trials)
                self.cv_folds = config.get('cv_folds', self.cv_folds)
                logger.info(f"Metadata cargada desde: {metadata_path}")
            
        except Exception as e:
            # Fallback: intentar cargar formato antiguo (diccionario)
            logger.warning(f"Error cargando modelo directo, intentando formato legacy: {e}")
            try:
                model_data = joblib.load(filepath)
                if isinstance(model_data, dict) and 'stacking_model' in model_data:
                    self.stacking_model = model_data['stacking_model']
                    self.trained_base_models = model_data.get('trained_base_models', {})
                    self.best_params_per_model = model_data.get('best_params_per_model', {})
                    self.feature_engineer = model_data.get('feature_engineer', PointsFeatureEngineer())
                    self.scaler = model_data.get('scaler', StandardScaler())
                    self.selected_features = model_data.get('selected_features', [])
                    self.feature_importance = model_data.get('feature_importance', {})
                    self.training_metrics = model_data.get('training_metrics', {})
                    self.validation_metrics = model_data.get('validation_metrics', {})
                    self.cv_scores = model_data.get('cv_scores', {})
                    self.is_trained = model_data.get('is_trained', True)
                    logger.info(f"Modelo legacy (diccionario) cargado desde: {filepath}")
                else:
                    raise ValueError("Formato de archivo no reconocido")
            except Exception as e2:
                raise ValueError(f"No se pudo cargar el modelo. Error formato directo: {e}, Error formato legacy: {e2}")
    
    def generate_report(self) -> Dict:
        """
        PARTE 3: Generar reporte completo del modelo de stacking.
        
        Returns:
            Dict: Reporte completo con métricas y configuración
        """
        if not self.is_trained:
            raise ValueError("Modelo no entrenado.")
        
        report = {
            'model_info': {
                'type': 'Stacking Ensemble Regressor',
                'target': 'PTS (Puntos)',
                'n_base_models': len(self.trained_base_models),
                'base_models': list(self.trained_base_models.keys()),
                'meta_learner': type(self.stacking_model.final_estimator_).__name__,
                'features_count': len(self.selected_features),
                'stacking_cv_folds': 3
            },
            'performance': {
                'stacking': {
                'training': self.training_metrics,
                'validation': self.validation_metrics,
                'cross_validation': self.cv_scores
            },
                'base_models': self.get_model_performance_summary()['base_models_performance']
            },
            'model_importance': self.feature_importance,
            'best_hyperparameters': self.best_params_per_model,
            'model_config': {
                'n_trials_per_model': self.n_trials,
                'cv_folds': self.cv_folds,
                'early_stopping_rounds': self.early_stopping_rounds,
                'random_state': self.random_state,
                'enable_neural_networks': self.enable_neural_networks,
                'enable_gpu': self.enable_gpu,
                'enable_svr': self.enable_svr
            },
            'feature_selection': {
                'total_features': len(self.selected_features),
                'top_10_features': self.selected_features[:10] if len(self.selected_features) >= 10 else self.selected_features
            }
        }
        
        return report
    
    def _calculate_adaptive_weights(self, y_true: np.ndarray) -> np.ndarray:
        """
        Calcula pesos adaptativos BALANCEADOS basados en el rango de puntos.
        BALANCEADO: Pesos moderados para evitar underfitting mientras se enfoca en alto scoring.
        
        Args:
            y_true: Valores reales de puntos
            
        Returns:
            np.ndarray: Pesos adaptativos para cada muestra
        """
        weights = np.ones_like(y_true, dtype=float)
        
        # ESTRATEGIA DE PESOS BALANCEADA (evitar underfitting)
        # 0-5 puntos: peso normal (1.0) - jugadores suplentes
        # 5-10 puntos: peso ligeramente mayor (1.2) - jugadores rotación
        # 10-15 puntos: peso moderado (1.5) - jugadores importantes
        # 15-20 puntos: peso alto (2.0) - jugadores clave
        # 20-25 puntos: peso muy alto (2.8) - estrellas
        # 25-30 puntos: peso extremo (4.0) - superstrellas
        # 30+ puntos: peso máximo (5.5) - actuaciones elite
        
        weights = np.where(y_true >= 30, 5.5, weights)      # Elite performances (REDUCIDO de 7.0)
        weights = np.where((y_true >= 25) & (y_true < 30), 4.0, weights)  # Superstars (REDUCIDO de 5.0)
        weights = np.where((y_true >= 20) & (y_true < 25), 2.8, weights)  # Stars (REDUCIDO de 3.5)
        weights = np.where((y_true >= 15) & (y_true < 20), 2.0, weights)  # Key players (REDUCIDO de 2.5)
        weights = np.where((y_true >= 10) & (y_true < 15), 1.5, weights)  # Important players (REDUCIDO de 1.8)
        weights = np.where((y_true >= 5) & (y_true < 10), 1.2, weights)   # Rotation players (REDUCIDO de 1.3)
        
        # Peso adicional para casos extremos MODERADO (40+ puntos)
        weights = np.where(y_true >= 40, 7.0, weights)  # REDUCIDO de 10.0
        
        # Solo mostrar estadísticas en modo debug si es necesario
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Pesos adaptativos BALANCEADOS calculados - Distribución:")
            logger.debug(f"  Peso 1.0 (0-5pts): {np.sum(weights == 1.0)} muestras")
            logger.debug(f"  Peso 1.2 (5-10pts): {np.sum(weights == 1.2)} muestras")
            logger.debug(f"  Peso 1.5 (10-15pts): {np.sum(weights == 1.5)} muestras")
            logger.debug(f"  Peso 2.0 (15-20pts): {np.sum(weights == 2.0)} muestras")
            logger.debug(f"  Peso 2.8 (20-25pts): {np.sum(weights == 2.8)} muestras")
            logger.debug(f"  Peso 4.0 (25-30pts): {np.sum(weights == 4.0)} muestras")
            logger.debug(f"  Peso 5.5 (30-40pts): {np.sum((weights == 5.5) & (y_true < 40))} muestras")
            logger.debug(f"  Peso 7.0 (40+pts): {np.sum(weights == 7.0)} muestras")
        
        return weights

    def _calculate_range_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Dict[str, float]]:
        """
        Calcula métricas de evaluación por rangos de puntuación MEJORADAS.
        NUEVO: Rangos más específicos y métricas especializadas para alto scoring.
        
        Args:
            y_true: Valores reales de puntos
            y_pred: Predicciones de puntos
            
        Returns:
            Dict[str, Dict[str, float]]: Métricas de evaluación por rangos de puntuación
        """
        range_metrics = {}
        
        # RANGOS MEJORADOS Y MÁS ESPECÍFICOS
        ranges = [
            (0, 5, "Suplentes"),
            (5, 10, "Rotación"),
            (10, 15, "Importantes"),
            (15, 20, "Clave"),
            (20, 25, "Estrellas"),
            (25, 30, "Superstrellas"),
            (30, 40, "Elite"),
            (40, float('inf'), "Histórico")
        ]
        
        for start, end, label in ranges:
            range_mask = (y_true >= start) & (y_true < end)
            range_samples = np.sum(range_mask)
            
            if range_samples > 0:
                range_y_true = y_true[range_mask]
                range_y_pred = y_pred[range_mask]
                
                # Métricas básicas
                range_mae = mean_absolute_error(range_y_true, range_y_pred)
                range_rmse = np.sqrt(mean_squared_error(range_y_true, range_y_pred))
                range_r2 = r2_score(range_y_true, range_y_pred)
                
                # NUEVAS MÉTRICAS ESPECÍFICAS PARA ALTO SCORING
                
                # 1. Sesgo de predicción (bias)
                bias = np.mean(range_y_pred - range_y_true)
                
                # 2. Porcentaje de subestimación
                underestimation_pct = np.mean(range_y_pred < range_y_true) * 100
                
                # 3. Error promedio en subestimación
                underestimated_mask = range_y_pred < range_y_true
                if underestimated_mask.sum() > 0:
                    avg_underestimation = np.mean(range_y_true[underestimated_mask] - range_y_pred[underestimated_mask])
                else:
                    avg_underestimation = 0
                
                # 4. Accuracy por tolerancia específica para el rango
                if start >= 20:  # Para alto scoring, tolerancia mayor
                    tolerance = 4
                elif start >= 10:  # Para scoring medio, tolerancia media
                    tolerance = 3
                else:  # Para bajo scoring, tolerancia menor
                    tolerance = 2
                
                accuracy_range = np.mean(np.abs(range_y_pred - range_y_true) <= tolerance) * 100
                
                # 5. Coeficiente de variación del error
                error_cv = np.std(np.abs(range_y_pred - range_y_true)) / (range_mae + 1e-6)
                
                # 6. Métricas específicas para rangos altos
                if start >= 20:
                    # Capacidad de predecir juegos explosivos (30+ puntos)
                    explosive_games_true = np.sum(range_y_true >= 30)
                    explosive_games_pred = np.sum(range_y_pred >= 30)
                    
                    # CORREGIDO: Cálculo correcto de explosive_recall
                    if explosive_games_true > 0:
                        # Recall: cuántos juegos explosivos reales predecimos correctamente
                        correctly_pred_explosive = np.sum((range_y_true >= 30) & (range_y_pred >= 27))  # Tolerancia de 3 puntos
                        explosive_recall = correctly_pred_explosive / explosive_games_true
                    else:
                        explosive_recall = 0.0
                    
                    # Error en predicciones de alto impacto
                    high_impact_mask = range_y_true >= 25
                    if high_impact_mask.sum() > 0:
                        high_impact_mae = mean_absolute_error(
                            range_y_true[high_impact_mask], 
                            range_y_pred[high_impact_mask]
                        )
                    else:
                        high_impact_mae = 0
                else:
                    explosive_recall = 0.0
                    high_impact_mae = 0.0
                
                range_metrics[f"{start}-{end}_{label}"] = {
                    'mae': range_mae,
                    'rmse': range_rmse,
                    'r2': range_r2,
                    'bias': bias,
                    'underestimation_pct': underestimation_pct,
                    'avg_underestimation': avg_underestimation,
                    'accuracy_tolerance': accuracy_range,
                    'error_cv': error_cv,
                    'explosive_recall': explosive_recall,
                    'high_impact_mae': high_impact_mae,
                    'samples': range_samples
                }
        
        # MÉTRICAS AGREGADAS PARA ALTO SCORING
        high_scoring_mask = y_true >= 20
        if high_scoring_mask.sum() > 0:
            high_scoring_metrics = self._calculate_high_scoring_aggregate_metrics(
                y_true[high_scoring_mask], 
                y_pred[high_scoring_mask]
            )
            range_metrics['HIGH_SCORING_AGGREGATE'] = high_scoring_metrics
        
        return range_metrics
    
    def _calculate_high_scoring_aggregate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        NUEVO: Calcula métricas agregadas específicas para jugadores de alto scoring (20+ puntos).
        
        Args:
            y_true: Valores reales de puntos (solo 20+)
            y_pred: Predicciones de puntos (solo 20+)
            
        Returns:
            Dict[str, float]: Métricas agregadas para alto scoring
        """
        metrics = {}
        
        # Métricas básicas
        metrics['mae'] = mean_absolute_error(y_true, y_pred)
        metrics['rmse'] = np.sqrt(mean_squared_error(y_true, y_pred))
        metrics['r2'] = r2_score(y_true, y_pred)
        
        # Sesgo sistemático
        metrics['systematic_bias'] = np.mean(y_pred - y_true)
        
        # Distribución de errores
        errors = y_pred - y_true
        metrics['error_std'] = np.std(errors)
        metrics['error_skewness'] = self._calculate_skewness(errors)
        
        # Capacidad de predecir diferentes niveles
        for threshold in [25, 30, 35, 40]:
            true_above = np.sum(y_true >= threshold)
            pred_above = np.sum(y_pred >= threshold)
            
            if true_above > 0:
                # Recall: qué porcentaje de juegos de threshold+ puntos predecimos
                correctly_pred_above = np.sum((y_true >= threshold) & (y_pred >= threshold - 3))
                recall = correctly_pred_above / true_above
                
                # Precision: de los que predecimos threshold+, qué porcentaje son correctos
                if pred_above > 0:
                    precision = correctly_pred_above / pred_above
                else:
                    precision = 0
                
                metrics[f'recall_{threshold}plus'] = recall
                metrics[f'precision_{threshold}plus'] = precision
            else:
                metrics[f'recall_{threshold}plus'] = 0
                metrics[f'precision_{threshold}plus'] = 0
        
        # Métricas de calibración
        metrics['mean_prediction'] = np.mean(y_pred)
        metrics['mean_actual'] = np.mean(y_true)
        metrics['prediction_calibration'] = metrics['mean_prediction'] / metrics['mean_actual']
        
        # Consistencia de predicción
        metrics['prediction_std'] = np.std(y_pred)
        metrics['actual_std'] = np.std(y_true)
        metrics['variance_ratio'] = metrics['prediction_std'] / metrics['actual_std']
        
        metrics['samples'] = len(y_true)
        
        return metrics
    
    def _calculate_skewness(self, data: np.ndarray) -> float:
        """
        Calcula la asimetría (skewness) de los datos.
        
        Args:
            data: Array de datos
            
        Returns:
            float: Valor de skewness
        """
        try:
            mean = np.mean(data)
            std = np.std(data)
            if std == 0:
                return 0
            skewness = np.mean(((data - mean) / std) ** 3)
            return skewness
        except:
            return 0

    def _apply_bias_correction(self, predictions: np.ndarray, features_df: pd.DataFrame = None) -> np.ndarray:
        """
        NUEVO: Aplica corrección de sesgo para reducir subestimación en alto scoring.
        
        Args:
            predictions: Predicciones originales del modelo
            features_df: DataFrame con features para contexto (opcional)
            
        Returns:
            np.ndarray: Predicciones corregidas
        """
        corrected_predictions = predictions.copy()
        
        # CORRECCIÓN BÁSICA PARA ALTO SCORING
        high_scoring_mask = predictions >= 20
        medium_scoring_mask = (predictions >= 15) & (predictions < 20)
        
        # Corrección progresiva basada en nivel de predicción
        corrected_predictions[high_scoring_mask] += 0.8  # Corrección para alto scoring
        corrected_predictions[medium_scoring_mask] += 0.4  # Corrección menor para medio scoring
        
        # CORRECCIÓN ADICIONAL BASADA EN FEATURES (si están disponibles)
        if features_df is not None:
            # Si hay features de uso, ajustar por usage rate
            if 'usage_rate_5g' in features_df.columns:
                high_usage_mask = features_df['usage_rate_5g'] > 25
                corrected_predictions[high_usage_mask] += 0.3
            
            # Si hay features de minutos, ajustar por minutos altos
            if 'mp_hist_avg_5g' in features_df.columns:
                high_minutes_mask = features_df['mp_hist_avg_5g'] > 32
                corrected_predictions[high_minutes_mask] += 0.2
        
            # Corrección basada en ensemble confidence si está disponible
            if 'ensemble_confidence_score' in features_df.columns:
                confidence_scores = features_df['ensemble_confidence_score'].values
                
                # Corrección adaptativa basada en confianza
                high_confidence_mask = confidence_scores > 0.7
                medium_confidence_mask = (confidence_scores >= 0.4) & (confidence_scores <= 0.7)
                low_confidence_mask = confidence_scores < 0.4
                
                # Corrección diferenciada por confianza
                high_conf_high_pred = high_confidence_mask & (predictions >= 25)
                corrected_predictions[high_conf_high_pred] += 0.3
                
                med_conf_high_pred = medium_confidence_mask & (predictions >= 20)
                corrected_predictions[med_conf_high_pred] += 0.6
                
                low_conf_high_pred = low_confidence_mask & (predictions >= 18)
                corrected_predictions[low_conf_high_pred] += 1.0
        
        # Asegurar que las predicciones no sean negativas ni excesivamente altas
        corrected_predictions = np.clip(corrected_predictions, 0, 80)
        
        # Log de correcciones aplicadas solo en modo debug
        corrections_applied = np.sum(corrected_predictions != predictions)
        if logger.isEnabledFor(logging.DEBUG) and corrections_applied > 0:
            avg_correction = np.mean(corrected_predictions - predictions)
            logger.debug(f"Corrección de sesgo aplicada a {corrections_applied} predicciones")
            logger.debug(f"Corrección promedio: +{avg_correction:.2f} puntos")
        
        return corrected_predictions
    

# Mantener compatibilidad con el modelo original
class XGBoostPTSModel(StackingPTSModel):
    """
    Clase de compatibilidad que hereda del nuevo modelo de stacking.
    Mantiene la interfaz original pero usa el stacking internamente.
    """
    
    def __init__(self, **kwargs):
        """
        Inicializa el modelo manteniendo compatibilidad con la interfaz original.
        """
        # Mapear parámetros antiguos a nuevos
        if 'n_trials' not in kwargs:
            kwargs['n_trials'] = 75  # AUMENTADO de 50 para mejor optimización
        
        # Deshabilitar SVR por defecto para evitar problemas de timeout
        if 'enable_svr' not in kwargs:
            kwargs['enable_svr'] = False
            logger.info("SVR deshabilitado por defecto en modo compatibilidad para evitar timeouts")
        
        super().__init__(**kwargs)
        
        # Mantener atributos para compatibilidad
        self.model = None  # Se asignará al stacking_model después del entrenamiento
        self.best_params = None  # Se asignará a los parámetros del mejor modelo base
        
        logger.info("Modelo XGBoost PTS inicializado con stacking ensemble BALANCEADO")
    
    def train(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Entrenamiento manteniendo compatibilidad con la interfaz original.
        """
        # Llamar al método de entrenamiento del stacking
        result = super().train(df)
        
        # Asignar para compatibilidad
        self.model = self.stacking_model
        if self.best_params_per_model:
            # Usar parámetros del modelo con mejor rendimiento
            best_model_name = min(
                self.best_params_per_model.keys(),
                key=lambda x: self.cv_scores.get(f'{x}_mae_mean', float('inf'))
            )
            self.best_params = self.best_params_per_model[best_model_name]
        
        # NUEVO: Calcular y guardar cutoff_date para compatibilidad con trainer
        if 'Date' in df.columns:
            df_sorted = df.sort_values('Date').reset_index(drop=True)
            split_idx = int(len(df_sorted) * 0.8)  # Mismo split que en _temporal_split
            self.cutoff_date = df_sorted.iloc[split_idx]['Date']
            logger.info(f"Cutoff date establecido: {self.cutoff_date}")
        else:
            # Si no hay columna Date, usar fecha actual como fallback
            from datetime import datetime
            self.cutoff_date = datetime.now()
            logger.warning("No se encontró columna 'Date', usando fecha actual como cutoff_date")
        
        return result
    
    def get_feature_importance(self, top_n: int = 20) -> Dict[str, float]:
        """
        Obtener importancia de características para compatibilidad.
        
        Args:
            top_n: Número de características más importantes a retornar
            
        Returns:
            Dict[str, float]: Diccionario con importancia de características
        """
        if not self.is_trained:
            raise ValueError("Modelo no entrenado. Ejecutar train() primero.")
        
        # Intentar obtener feature importance del stacking ensemble
        if hasattr(self, 'feature_importance') and self.feature_importance:
            # Si tenemos importancia del meta-learner (importancia de modelos base)
            stacking_importance = self.feature_importance.copy()
        else:
            stacking_importance = {}
        
        # Intentar obtener feature importance de modelos base individuales
        feature_importance_combined = {}
        
        # Combinar importancia de todos los modelos base
        for model_name, model in self.trained_base_models.items():
            try:
                if hasattr(model, 'feature_importances_'):
                    # Para modelos tree-based (XGBoost, LightGBM, RandomForest, etc.)
                    importances = model.feature_importances_
                    feature_names = self.selected_features
                    
                    for feature, importance in zip(feature_names, importances):
                        if feature not in feature_importance_combined:
                            feature_importance_combined[feature] = 0
                        feature_importance_combined[feature] += importance
                        
                elif hasattr(model, 'coef_'):
                    # Para modelos lineales (Ridge, Lasso, etc.)
                    importances = np.abs(model.coef_)
                    feature_names = self.selected_features
                    
                    for feature, importance in zip(feature_names, importances):
                        if feature not in feature_importance_combined:
                            feature_importance_combined[feature] = 0
                        feature_importance_combined[feature] += importance
                        
            except Exception as e:
                logger.warning(f"No se pudo obtener feature importance de {model_name}: {e}")
                continue
        
        # Normalizar importancias combinadas
        if feature_importance_combined:
            total_importance = sum(feature_importance_combined.values())
            if total_importance > 0:
                feature_importance_combined = {
                    feature: importance / total_importance 
                    for feature, importance in feature_importance_combined.items()
                }
        
        # Si no hay importancia de features, usar importancia uniforme
        if not feature_importance_combined:
            feature_importance_combined = {
                feature: 1.0 / len(self.selected_features) 
                for feature in self.selected_features
            }
        
        # Ordenar por importancia y tomar top_n
        sorted_features = sorted(
            feature_importance_combined.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        top_features = dict(sorted_features[:top_n])
        
        logger.info(f"Feature importance calculada para top {len(top_features)} características")
        
        return top_features

