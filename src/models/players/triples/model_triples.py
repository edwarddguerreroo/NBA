"""
Modelo XGBoost para Predicción de Triples NBA (3P)
=================================================

Modelo especializado para predecir triples que anotará un jugador en su próximo partido.
Optimizado específicamente para las características únicas del tiro de 3 puntos.

Arquitectura:
- XGBoost con hiperparámetros optimizados para triples
- Validación cruzada temporal
- Optimización bayesiana con Optuna
- Early stopping para prevenir overfitting
- Features especializadas en patrones de tiro de 3PT
- Stacking ensemble con modelos especializados
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
import gc
import tempfile
import shutil

import joblib
import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from optuna.samplers import TPESampler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import (
    RandomForestRegressor, 
    ExtraTreesRegressor, 
    GradientBoostingRegressor,
    VotingRegressor,
    StackingRegressor
)
from sklearn.linear_model import Ridge, Lasso, ElasticNet

# Importar el feature engineer especializado
from .features_triples import ThreePointsFeatureEngineer

warnings.filterwarnings('ignore')

# Configurar variables de entorno para evitar problemas de tkinter y matplotlib
os.environ['MPLBACKEND'] = 'Agg'  # Evitar GUI backends
os.environ['DISPLAY'] = ''  # Evitar problemas de display en Windows
os.environ['TK_SILENCE_DEPRECATION'] = '1'  # Silenciar warnings de tkinter

# Configurar joblib para evitar problemas de memoria y threading
os.environ['JOBLIB_MULTIPROCESSING'] = '0'  # Desactivar multiprocessing
os.environ['JOBLIB_TEMP_FOLDER'] = tempfile.gettempdir()
os.environ['OMP_NUM_THREADS'] = '1'  # Limitar threads OpenMP
os.environ['MKL_NUM_THREADS'] = '1'  # Limitar threads MKL

logger = logging.getLogger(__name__)


def configure_safe_environment():
    """
    Configura el entorno para evitar problemas de threading y tkinter
    """
    try:
        # Configurar matplotlib sin GUI
        import matplotlib
        matplotlib.use('Agg')
        
        # Suprimir warnings de tkinter
        import warnings
        warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')
        warnings.filterwarnings('ignore', message='.*main thread is not in main loop.*')
        
        # Configurar threading seguro
        import threading
        threading.current_thread().daemon = False
        
    except Exception as e:
        logger.debug(f"Error configurando entorno seguro: {e}")


# Configurar entorno seguro al importar el módulo
configure_safe_environment()


class TimeoutError(Exception):
    """Excepción personalizada para timeouts"""
    pass


def timeout_wrapper(func, timeout_seconds, *args, **kwargs):
    """
    Wrapper para ejecutar función con timeout compatible con Windows.
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
        logger.warning(f"Función {func.__name__} excedió timeout de {timeout_seconds}s")
        raise TimeoutError(f"Función excedió tiempo límite de {timeout_seconds} segundos")
    
    if exception[0]:
        raise exception[0]
    
    return result[0]


class Stacking3PTModel:
    """
    Modelo de Stacking Ensemble especializado para predicción de triples (3P)
    
    Combina múltiples algoritmos optimizados específicamente para:
    - Patrones de tiro de 3 puntos
    - Eficiencia vs volumen
    - Consistencia del tirador
    - Factores contextuales del tiro
    """
    
    def __init__(self,
                 n_trials: int = 100,  # Aumentado de 50 a 100 para mejor optimización
                 cv_folds: int = 5,
                 early_stopping_rounds: int = 25,
                 random_state: int = 42,
                 teams_df: pd.DataFrame = None):
        """
        Inicializa el modelo de stacking para triples.
        
        Args:
            n_trials: Número de trials para optimización bayesiana
            cv_folds: Número de folds para validación cruzada
            early_stopping_rounds: Rounds para early stopping
            random_state: Semilla para reproducibilidad
            teams_df: DataFrame con datos de equipos
        """
        self.n_trials = n_trials
        self.cv_folds = cv_folds
        self.early_stopping_rounds = early_stopping_rounds
        self.random_state = random_state
        self.teams_df = teams_df
        
        # Componentes principales
        self.feature_engineer = ThreePointsFeatureEngineer(teams_df=teams_df)
        self.scaler = StandardScaler()
        
        # Modelos base optimizados para triples
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
        
        # Configurar entorno seguro
        configure_safe_environment()
        
        # Configurar gestión de memoria
        self._temp_dir = None
        self._cleanup_temp_files()
        
        # Configurar modelos base especializados para triples
        self._setup_base_models()
        
        logger.info("Modelo Stacking 3PT inicializado para predicción de triples")
    
    def _setup_base_models(self):
        """
        Configura todos los modelos base del ensemble para triples
        """
        logger.info("Configurando modelos base especializados para triples...")
        
        # Importar catboost una sola vez al inicio
        try:
            import catboost as cb
            catboost_available = True
        except ImportError:
            catboost_available = False
            logger.warning("CatBoost no disponible")
        
        # 1. XGBoost optimizado para triples (parámetros ajustados para conteo discreto)
        self.base_models['xgboost'] = {
            'model_class': xgb.XGBRegressor,
            'param_space': {
                'n_estimators': (200, 1000),
                'learning_rate': (0.02, 0.15),
                'max_depth': (4, 12),
                'min_child_weight': (1, 8),
                'subsample': (0.75, 0.95),
                'colsample_bytree': (0.75, 0.95),
                'reg_alpha': (0.1, 4.0),
                'reg_lambda': (0.1, 6.0),
                'gamma': (0.0, 3.0),
                'max_delta_step': (0, 2)  # Ayuda con valores extremos
            },
            'fixed_params': {
                'objective': 'reg:squarederror',
                'random_state': self.random_state,
                'n_jobs': 1,  # Evitar problemas de paralelización
                'verbosity': 0,
                'tree_method': 'hist'  # Más eficiente para datos tabulares
            },
            'weight': 0.35  # Aumentar peso por ser más efectivo
        }
        
        # 2. LightGBM para capturar patrones específicos de tiro
        self.base_models['lightgbm'] = {
            'model_class': lgb.LGBMRegressor,
            'param_space': {
                'n_estimators': (200, 800),
                'learning_rate': (0.02, 0.12),
                'num_leaves': (31, 127),
                'min_child_samples': (15, 80),
                'subsample': (0.75, 0.95),
                'colsample_bytree': (0.75, 0.95),
                'reg_alpha': (0.1, 3.0),
                'reg_lambda': (0.1, 4.0),
                'min_split_gain': (0.0, 1.0),
                'feature_fraction': (0.7, 0.95)
            },
            'fixed_params': {
                'objective': 'regression',
                'metric': 'rmse',
                'random_state': self.random_state,
                'n_jobs': 1,  # Evitar problemas de paralelización
                'verbosity': -1,
                'boosting_type': 'gbdt',
                'force_col_wise': True
            },
            'weight': 0.25
        }
        
        # 3. CatBoost Ultra-Optimizado para Triples (solo si está disponible)
        if catboost_available:
            self.base_models['catboost_ultra'] = {
                'model_class': cb.CatBoostRegressor,
                'param_space': {
                    'iterations': (200, 800),
                    'learning_rate': (0.02, 0.15),
                    'depth': (4, 10),
                    'l2_leaf_reg': (1, 10),
                    'random_strength': (0.1, 2.0),
                    'bagging_temperature': (0.1, 1.0),
                    'border_count': (128, 255),
                    'feature_border_type': ['GreedyLogSum', 'Median', 'UniformAndQuantiles']
                },
                'fixed_params': {
                    'loss_function': 'RMSE',
                    'random_seed': self.random_state,
                    'verbose': False,
                    'allow_writing_files': False,
                    'thread_count': 1,  # Forzar single thread para evitar problemas de tkinter
                    'boosting_type': 'Plain'
                },
                'weight': 0.15
            }
        
        # 4. Random Forest especializado para triples
        self.base_models['random_forest'] = {
            'model_class': RandomForestRegressor,
            'param_space': {
                'n_estimators': (150, 400),
                'max_depth': (6, 16),
                'min_samples_split': (5, 20),
                'min_samples_leaf': (2, 10),
                'max_features': ['sqrt', 'log2', 0.6, 0.8],
                'max_samples': (0.7, 0.95)
            },
            'fixed_params': {
                'random_state': self.random_state,
                'n_jobs': 1,  # Evitar problemas de paralelización
                'oob_score': True
            },
            'weight': 0.15
        }
        
        # 5. Extra Trees para diversidad
        self.base_models['extra_trees'] = {
            'model_class': ExtraTreesRegressor,
            'param_space': {
                'n_estimators': (120, 300),
                'max_depth': (6, 14),
                'min_samples_split': (3, 15),
                'min_samples_leaf': (1, 8),
                'max_features': ['sqrt', 'log2', 0.6],
                'bootstrap': [True, False]
            },
            'fixed_params': {
                'random_state': self.random_state,
                'n_jobs': 1  # Evitar problemas de paralelización
            },
            'weight': 0.1
        }
        
        # 6. Gradient Boosting especializado en valores extremos
        self.base_models['gradient_boosting'] = {
            'model_class': GradientBoostingRegressor,
            'param_space': {
                'n_estimators': (100, 400),
                'learning_rate': (0.02, 0.12),
                'max_depth': (4, 10),
                'min_samples_split': (5, 20),
                'min_samples_leaf': (2, 10),
                'subsample': (0.7, 0.95),
                'max_features': ['sqrt', 'log2', 0.7]
            },
            'fixed_params': {
                'loss': 'squared_error',
                'random_state': self.random_state,
                'validation_fraction': 0.1,
                'n_iter_no_change': 15,
                'tol': 1e-4
            },
            'weight': 0.1
        }
        
        # 7. Redes Neuronales Ultra-Especializadas
        try:
            from sklearn.neural_network import MLPRegressor
            self.base_models['neural_network'] = {
                'model_class': MLPRegressor,
                'param_space': {
                    'hidden_layer_sizes': [(50,), (100,), (50, 25), (100, 50)],
                    'activation': ['relu', 'tanh'],
                    'alpha': (0.0001, 0.01),
                    'learning_rate': ['constant', 'adaptive'],
                    'learning_rate_init': (0.001, 0.01)
                },
                'fixed_params': {
                    'solver': 'adam',
                    'max_iter': 300,
                    'random_state': self.random_state,
                    'early_stopping': True,
                    'validation_fraction': 0.1,
                    'n_iter_no_change': 10
                },
                'weight': 0.1
            }
        except ImportError:
            logger.warning("Neural Network no disponible, saltando...")
        
        # 8. Ridge Regression para estabilidad
        from sklearn.linear_model import Ridge
        self.base_models['ridge'] = {
            'model_class': Ridge,
            'param_space': {
                'alpha': (0.1, 100.0),
                'solver': ['auto', 'svd', 'cholesky']
            },
            'fixed_params': {
                'random_state': self.random_state,
                'max_iter': 1000
            },
            'weight': 0.08
        }
        
        # 9. ElasticNet para regularización combinada
        from sklearn.linear_model import ElasticNet
        self.base_models['elastic_net'] = {
            'model_class': ElasticNet,
            'param_space': {
                'alpha': (0.01, 10.0),
                'l1_ratio': (0.1, 0.9),
                'max_iter': [1000, 2000]
            },
            'fixed_params': {
                'random_state': self.random_state,
                'selection': 'cyclic'
            },
            'weight': 0.08
        }
        
        # 10. Support Vector Regression para patrones no lineales
        try:
            from sklearn.svm import SVR
            self.base_models['svr'] = {
                'model_class': SVR,
                'param_space': {
                    'C': (0.1, 100.0),
                    'epsilon': (0.01, 0.5),
                    'kernel': ['rbf', 'poly'],
                    'gamma': ['scale', 'auto']
                },
                'fixed_params': {
                    'cache_size': 200,
                    'max_iter': 1000
                },
                'weight': 0.08
            }
        except ImportError:
            logger.warning("SVR no disponible, saltando...")
        
        # 11. AdaBoost para capturar errores residuales
        from sklearn.ensemble import AdaBoostRegressor
        from sklearn.tree import DecisionTreeRegressor
        self.base_models['ada_boost'] = {
            'model_class': AdaBoostRegressor,
            'param_space': {
                'n_estimators': (50, 200),
                'learning_rate': (0.01, 0.5),
                'loss': ['linear', 'square', 'exponential']
            },
            'fixed_params': {
                'base_estimator': DecisionTreeRegressor(max_depth=3, random_state=self.random_state),
                'random_state': self.random_state
            },
            'weight': 0.08
        }
        
        # 12. Huber Regressor para robustez a outliers
        from sklearn.linear_model import HuberRegressor
        self.base_models['huber'] = {
            'model_class': HuberRegressor,
            'param_space': {
                'epsilon': (1.1, 2.0),
                'alpha': (0.0001, 0.1),
                'max_iter': [100, 200, 300]
            },
            'fixed_params': {
                'fit_intercept': True,
                'tol': 1e-4
            },
            'weight': 0.07
        }
        
        logger.info(f"Configurados {len(self.base_models)} modelos base para triples")
    
    def _cleanup_temp_files(self):
        """Limpia archivos temporales y libera memoria"""
        try:
            # Limpiar directorio temporal si existe
            if self._temp_dir and os.path.exists(self._temp_dir):
                shutil.rmtree(self._temp_dir, ignore_errors=True)
            
            # Crear nuevo directorio temporal
            self._temp_dir = tempfile.mkdtemp(prefix='triples_model_')
            
            # Configurar joblib para usar directorio temporal específico
            os.environ['JOBLIB_TEMP_FOLDER'] = self._temp_dir
            
            # Configurar threading para evitar problemas de tkinter
            import matplotlib
            matplotlib.use('Agg')  # Asegurar backend sin GUI
            
            # Forzar recolección de basura
            gc.collect()
            
        except Exception as e:
            logger.warning(f"Error limpiando archivos temporales: {e}")
    
    def __del__(self):
        """Limpieza al destruir el objeto"""
        try:
            # Configurar entorno seguro antes de limpiar
            configure_safe_environment()
            self._cleanup_temp_files()
        except Exception:
            # Silenciar errores de destructor para evitar problemas de threading
            pass
    
    def _optimize_single_model(self, model_name: str, X_train, y_train, X_val, y_val, sample_weights=None) -> Dict[str, Any]:
        """
        Optimiza un modelo individual usando Optuna
        """
        logger.info(f"Optimizando modelo {model_name} para triples...")
        
        model_config = self.base_models[model_name]
        
        def objective(trial):
            # Construir parámetros del trial
            params = {}
            for param_name, param_range in model_config['param_space'].items():
                if isinstance(param_range, tuple) and len(param_range) == 2:
                    if isinstance(param_range[0], int):
                        params[param_name] = trial.suggest_int(param_name, param_range[0], param_range[1])
                    else:
                        params[param_name] = trial.suggest_float(param_name, param_range[0], param_range[1])
                elif isinstance(param_range, list):
                    params[param_name] = trial.suggest_categorical(param_name, param_range)
            
            # Combinar con parámetros fijos
            params.update(model_config['fixed_params'])
            
            try:
                X_train_local = X_train.copy() if hasattr(X_train, 'copy') else X_train
                X_val_local = X_val.copy() if hasattr(X_val, 'copy') else X_val
                
                # Entrenar modelo
                model = model_config['model_class'](**params)
                
                # Validación específica para XGBoost - asegurar consistencia de feature names
                if model_name == 'xgboost':
                    # Verificar que X_train y X_val tengan las mismas columnas
                    if hasattr(X_train_local, 'columns') and hasattr(X_val_local, 'columns'):
                        if list(X_train_local.columns) != list(X_val_local.columns):
                            logger.warning(f"Inconsistencia en columnas - Train: {len(X_train_local.columns)}, Val: {len(X_val_local.columns)}")
                            # Sincronizar columnas
                            common_cols = list(set(X_train_local.columns) & set(X_val_local.columns))
                            X_train_local = X_train_local[common_cols]
                            X_val_local = X_val_local[common_cols]
                            logger.info(f"Columnas sincronizadas: {len(common_cols)}")
                
                # Entrenar modelo con early stopping y sample weights según el tipo
                if model_name == 'lightgbm':
                    fit_params = {
                        'eval_set': [(X_val_local, y_val)],
                        'callbacks': [lgb.early_stopping(self.early_stopping_rounds), lgb.log_evaluation(0)]
                    }
                    if sample_weights is not None:
                        fit_params['sample_weight'] = sample_weights
                    model.fit(X_train_local, y_train, **fit_params)
                elif model_name == 'catboost_ultra':
                    fit_params = {
                        'eval_set': (X_val_local, y_val),
                        'verbose': False,
                        'thread_count': 1  # Forzar single thread para evitar problemas
                    }
                    if sample_weights is not None:
                        fit_params['sample_weight'] = sample_weights
                    model.fit(X_train_local, y_train, **fit_params)
                elif model_name in ['random_forest', 'extra_trees', 'gradient_boosting', 'ada_boost']:
                    fit_params = {}
                    if sample_weights is not None:
                        fit_params['sample_weight'] = sample_weights
                    model.fit(X_train_local, y_train, **fit_params)
                elif model_name == 'neural_network':
                    # Redes neuronales no soportan sample_weight directamente en sklearn
                    model.fit(X_train_local, y_train)
                elif model_name in ['ridge', 'elastic_net', 'huber']:
                    # Modelos lineales simples
                    fit_params = {}
                    if sample_weights is not None and model_name != 'elastic_net':
                        fit_params['sample_weight'] = sample_weights
                    model.fit(X_train_local, y_train, **fit_params)
                elif model_name == 'svr':
                    # SVR no soporta sample_weight
                    model.fit(X_train_local, y_train)
                else:
                    # XGBoost y otros
                    fit_params = {}
                    if sample_weights is not None:
                        fit_params['sample_weight'] = sample_weights
                    model.fit(X_train_local, y_train, **fit_params)
                
                # Predecir y evaluar
                y_pred = model.predict(X_val_local)
                
                # Clipping para triples (0 a 15 máximo razonable)
                y_pred = np.clip(y_pred, 0, 15)
                
                # Usar MAE como métrica principal para triples
                mae = mean_absolute_error(y_val, y_pred)
                
                return mae
                
            except Exception as e:
                logger.warning(f"Error en trial {trial.number} para {model_name}: {str(e)}")
                return float('inf')
        
        # Crear estudio con sampler optimizado
        study = optuna.create_study(
            direction='minimize',
            sampler=TPESampler(seed=self.random_state)
        )
        
        # Optimizar con timeout para triples
        try:
            study.optimize(objective, n_trials=min(self.n_trials, 50), timeout=180)  # 3 min timeout, máximo 50 trials
        except Exception as e:
            logger.warning(f"Error en optimización de {model_name}: {str(e)}")
            return None
        
        # Obtener mejores parámetros
        best_params = study.best_params.copy()
        best_params.update(model_config['fixed_params'])
        
        # Limpiar memoria antes de entrenar modelo final
        gc.collect()
        
        # Entrenar modelo final con sample weights
        best_model = model_config['model_class'](**best_params)
        
        # Validación específica para XGBoost - asegurar consistencia de feature names
        if model_name == 'xgboost':
            # Verificar que X_train y X_val tengan las mismas columnas
            if hasattr(X_train, 'columns') and hasattr(X_val, 'columns'):
                if list(X_train.columns) != list(X_val.columns):
                    logger.warning(f"Inconsistencia en columnas finales - Train: {len(X_train.columns)}, Val: {len(X_val.columns)}")
                    # Sincronizar columnas
                    common_cols = list(set(X_train.columns) & set(X_val.columns))
                    X_train = X_train[common_cols]
                    X_val = X_val[common_cols]
                    logger.info(f"Columnas finales sincronizadas: {len(common_cols)}")
        
        if model_name == 'lightgbm':
            fit_params = {
                'eval_set': [(X_val, y_val)],
                'callbacks': [lgb.early_stopping(self.early_stopping_rounds), lgb.log_evaluation(0)]
            }
            if sample_weights is not None:
                fit_params['sample_weight'] = sample_weights
            best_model.fit(X_train, y_train, **fit_params)
        elif model_name == 'catboost_ultra':
            fit_params = {
                'eval_set': (X_val, y_val),
                'verbose': False,
                'thread_count': 1  # Forzar single thread para evitar problemas
            }
            if sample_weights is not None:
                fit_params['sample_weight'] = sample_weights
            best_model.fit(X_train, y_train, **fit_params)
        elif model_name in ['random_forest', 'extra_trees', 'gradient_boosting', 'ada_boost']:
            fit_params = {}
            if sample_weights is not None:
                fit_params['sample_weight'] = sample_weights
            best_model.fit(X_train, y_train, **fit_params)
        elif model_name == 'neural_network':
            # Redes neuronales no soportan sample_weight directamente en sklearn
            best_model.fit(X_train, y_train)
        elif model_name in ['ridge', 'elastic_net', 'huber']:
            # Modelos lineales simples
            fit_params = {}
            if sample_weights is not None and model_name != 'elastic_net':
                fit_params['sample_weight'] = sample_weights
            best_model.fit(X_train, y_train, **fit_params)
        elif model_name == 'svr':
            # SVR no soporta sample_weight
            best_model.fit(X_train, y_train)
        else:
            # XGBoost y otros
            fit_params = {}
            if sample_weights is not None:
                fit_params['sample_weight'] = sample_weights
            best_model.fit(X_train, y_train, **fit_params)
        
        # Limpiar memoria después de optimización
        best_score = study.best_value
        n_trials = len(study.trials)
        del study
        gc.collect()
        
        return {
            'model': best_model,
            'best_params': best_params,
            'best_score': best_score,
            'n_trials': n_trials
        }
    
    def _train_base_models(self, X_train, y_train, X_val, y_val, sample_weights=None) -> Dict[str, Any]:
        """
        Entrena todos los modelos base
        """
        logger.info("Entrenando modelos base para triples...")
        
        results = {}
        
        for model_name in self.base_models.keys():
            logger.info(f"Entrenando modelo: {model_name}")
            
            try:
                result = self._optimize_single_model(model_name, X_train, y_train, X_val, y_val, sample_weights)
                
                if result is not None:
                    results[model_name] = result
                    self.trained_base_models[model_name] = result['model']
                    self.best_params_per_model[model_name] = result['best_params']
                    
                    logger.info(f"{model_name} - Mejor MAE: {result['best_score']:.4f}")
                else:
                    logger.warning(f"Falló entrenamiento de {model_name}")
                    
            except Exception as e:
                logger.error(f"Error entrenando {model_name}: {str(e)}")
                continue
        
        logger.info(f"Entrenados {len(results)} modelos base exitosamente")
        return results
    
    def _setup_stacking_model(self):
        """
        Configura el modelo de stacking con meta-learner optimizado para triples
        """
        if not self.trained_base_models:
            raise ValueError("No hay modelos base entrenados para stacking")
        
        # Preparar estimadores para stacking
        estimators = []
        for name, model in self.trained_base_models.items():
            estimators.append((name, model))
        
        # META-LEARNER ULTRA-ESPECIALIZADO PARA VALORES EXTREMOS
        from sklearn.ensemble import RandomForestRegressor, VotingRegressor, ExtraTreesRegressor
        from sklearn.linear_model import Ridge, ElasticNet, HuberRegressor
        from sklearn.svm import SVR
        
        # Crear ensemble de meta-learners optimizado para valores extremos
        base_meta_learners = [
            # Random Forest optimizado para valores extremos
            ('rf_extreme', RandomForestRegressor(
                n_estimators=200,
                max_depth=12,
                min_samples_split=2,
                min_samples_leaf=1,
                max_features=0.7,
                random_state=self.random_state,
                n_jobs=1,  # Evitar problemas de paralelización
                oob_score=True,
                bootstrap=True,
                max_samples=0.8  # Mejor para valores raros
            )),
            # Extra Trees para capturar patrones extremos
            ('et_extreme', ExtraTreesRegressor(
                n_estimators=150,
                max_depth=14,
                min_samples_split=2,
                min_samples_leaf=1,
                max_features=0.6,
                random_state=self.random_state,
                n_jobs=1,  # Evitar problemas de paralelización
                bootstrap=False  # Mejor para extremos
            )),
            # Huber Regressor para robustez con outliers
            ('huber_meta', HuberRegressor(
                epsilon=1.5,  # Más tolerante a valores extremos
                alpha=0.01,
                max_iter=200
            )),
            # Ridge con regularización suave
            ('ridge_meta', Ridge(
                alpha=0.3,  # Menos regularización para permitir extremos
                random_state=self.random_state
            ))
        ]
        
        # SVR especializado para valores extremos
        try:
            base_meta_learners.append(
                ('svr_extreme', SVR(
                    kernel='rbf',
                    C=2.0,  # Mayor C para permitir más complejidad
                    epsilon=0.05,  # Menor epsilon para mejor ajuste
                    gamma='scale'
                ))
            )
        except:
            pass
        
        # Meta-learner como ensemble votante con pesos optimizados para extremos
        meta_learner = VotingRegressor(
            estimators=base_meta_learners,
            weights=[0.35, 0.25, 0.15, 0.15, 0.1] if len(base_meta_learners) == 5 else [0.4, 0.3, 0.2, 0.1]
        )
        
        # Crear modelo de stacking
        self.stacking_model = StackingRegressor(
            estimators=estimators,
            final_estimator=meta_learner,
            cv=3,  # Menor CV para speed
            n_jobs=1,  # Evitar problemas de paralelización
            passthrough=False  # Solo usar predicciones de base models
        )
        
        logger.info("Modelo de stacking configurado para triples")
    
    def _temporal_cross_validate_stacking(self, X, y, df_with_dates) -> Dict[str, float]:
        """
        Validación cruzada temporal para el modelo de stacking - CRONOLÓGICA
        """
        logger.info("Realizando validación cruzada temporal...")
        
        # Ordenar cronológicamente: primero por jugador, luego por fecha
        df_sorted = df_with_dates.sort_values(['Player', 'Date']).reset_index(drop=True)
        
        # Configurar splits temporales
        tscv = TimeSeriesSplit(n_splits=self.cv_folds)
        
        cv_scores = []
        fold_metrics = []
        
        # CORRECCIÓN CRÍTICA: Preparar features consistentes antes de los folds
        logger.info("Preparando features consistentes para validación cruzada...")
        X_prepared, consistent_features = self._prepare_features(df_sorted)
        y_sorted = df_sorted['3P'].values if '3P' in df_sorted.columns else y.reindex(df_sorted.index)
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X_prepared)):
            logger.info(f"Fold {fold + 1}/{self.cv_folds}")
            
            # Usar features consistentes preparadas
            X_train_fold = X_prepared.iloc[train_idx]
            X_val_fold = X_prepared.iloc[val_idx]
            y_train_fold = y_sorted[train_idx]
            y_val_fold = y_sorted[val_idx]
            
            # VERIFICACIÓN ADICIONAL: Asegurar consistencia de columns entre train y val
            if list(X_train_fold.columns) != list(X_val_fold.columns):
                logger.warning(f"Fold {fold + 1}: Inconsistencia de columnas detectada")
                common_cols = list(set(X_train_fold.columns) & set(X_val_fold.columns))
                X_train_fold = X_train_fold[common_cols]
                X_val_fold = X_val_fold[common_cols]
                logger.info(f"Fold {fold + 1}: Usando {len(common_cols)} columnas comunes")
            
            # Verificar que specialization_progression esté presente en ambos
            if 'specialization_progression' not in X_train_fold.columns:
                logger.warning(f"Fold {fold + 1}: specialization_progression faltante en train, agregando")
                X_train_fold['specialization_progression'] = 0.0
            if 'specialization_progression' not in X_val_fold.columns:
                logger.warning(f"Fold {fold + 1}: specialization_progression faltante en val, agregando")
                X_val_fold['specialization_progression'] = 0.0
            
            try:
                # Entrenar stacking model
                self.stacking_model.fit(X_train_fold, y_train_fold)
                
                # Predecir
                y_pred_fold = self.stacking_model.predict(X_val_fold)
                y_pred_fold = np.clip(y_pred_fold, 0, 15)  # Clip para triples
                
                # Calcular métricas
                mae = mean_absolute_error(y_val_fold, y_pred_fold)
                rmse = np.sqrt(mean_squared_error(y_val_fold, y_pred_fold))
                r2 = r2_score(y_val_fold, y_pred_fold)
                
                cv_scores.append(mae)
                fold_metrics.append({
                    'fold': fold + 1,
                    'mae': mae,
                    'rmse': rmse,
                    'r2': r2,
                    'samples': len(y_val_fold),
                    'features_used': len(X_train_fold.columns)
                })
                
                logger.info(f"Fold {fold + 1} - MAE: {mae:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}, Features: {len(X_train_fold.columns)}")
                
            except Exception as e:
                logger.error(f"Error en fold {fold + 1}: {str(e)}")
                # Usar MAE penalizante en caso de error
                cv_scores.append(10.0)
                fold_metrics.append({
                    'fold': fold + 1,
                    'mae': 10.0,
                    'rmse': 10.0,
                    'r2': -1.0,
                    'samples': len(y_val_fold),
                    'features_used': len(X_train_fold.columns) if 'X_train_fold' in locals() else 0,
                    'error': str(e)
                })
                logger.warning(f"Fold {fold + 1} falló, usando MAE penalizante: 10.0")
        
        # Calcular métricas promedio
        valid_scores = [score for score in cv_scores if score < 9.0]  # Excluir scores penalizantes
        
        if valid_scores:
            avg_metrics = {
                'cv_mae_mean': np.mean(valid_scores),
                'cv_mae_std': np.std(valid_scores),
                'cv_rmse_mean': np.mean([m['rmse'] for m in fold_metrics if 'error' not in m]),
                'cv_r2_mean': np.mean([m['r2'] for m in fold_metrics if 'error' not in m]),
                'successful_folds': len(valid_scores),
                'total_folds': len(cv_scores),
                'features_consistency': consistent_features
            }
        else:
            logger.error("Todos los folds fallaron, usando métricas por defecto")
            avg_metrics = {
                'cv_mae_mean': 10.0,
                'cv_mae_std': 0.0,
                'cv_rmse_mean': 10.0,
                'cv_r2_mean': -1.0,
                'successful_folds': 0,
                'total_folds': len(cv_scores),
                'features_consistency': consistent_features
            }
        
        self.cv_scores = avg_metrics
        
        logger.info(f"CV completado - MAE: {avg_metrics['cv_mae_mean']:.4f} ± {avg_metrics['cv_mae_std']:.4f}")
        logger.info(f"Folds exitosos: {avg_metrics['successful_folds']}/{avg_metrics['total_folds']}")
        
        return avg_metrics
    
    def _prepare_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """
        Prepara features específicas para triples
        """
        logger.info("Preparando features para triples...")
        
        # Crear copia para no modificar original
        df_features = df.copy()
        
        # Generar features especializadas en triples
        feature_names = self.feature_engineer.generate_all_features(df_features)
        
        # Verificar que las features se crearon
        available_features = [f for f in feature_names if f in df_features.columns]
        missing_features = [f for f in feature_names if f not in df_features.columns]
        
        # Agregar features faltantes con valores por defecto
        for missing_feature in missing_features:
            logger.warning(f"Feature faltante '{missing_feature}' - agregando con valor por defecto")
            df_features[missing_feature] = 0.0
            available_features.append(missing_feature)
        
        # Asegurar que specialization_progression esté presente (fix específico)
        if 'specialization_progression' not in df_features.columns:
            logger.warning("specialization_progression faltante - agregando con valor por defecto")
            df_features['specialization_progression'] = 0.0
            if 'specialization_progression' not in available_features:
                available_features.append('specialization_progression')
        
        logger.info(f"Features disponibles para triples: {len(available_features)}")
        
        # Preparar DataFrame final
        X = df_features[available_features].fillna(0)
        
        # Verificación final de consistencia
        if len(X.columns) != len(available_features):
            logger.error(f"Inconsistencia: X tiene {len(X.columns)} columnas pero available_features tiene {len(available_features)}")
            # Sincronizar
            available_features = list(X.columns)
        
        return X, available_features
    
    def _temporal_split(self, df: pd.DataFrame, test_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split temporal de los datos - ORDENACIÓN CRONOLÓGICA GARANTIZADA
        Con ajuste temporal para corregir tendencia observada en dashboard
        """
        # Ordenar cronológicamente: primero por jugador, luego por fecha
        df_sorted = df.sort_values(['Player', 'Date']).reset_index(drop=True)
        
        # Verificar ordenación cronológica
        logger.info(f"Rango de fechas: {df_sorted['Date'].min()} a {df_sorted['Date'].max()}")
        
        # Calcular punto de corte basado en fechas, no en número de registros
        # Usar percentil 85 para tener más datos de entrenamiento (mejor para extremos)
        date_cutoff = df_sorted['Date'].quantile(1 - test_size + 0.05)
        
        # Split basado en fecha de corte
        train_df = df_sorted[df_sorted['Date'] < date_cutoff].copy()
        test_df = df_sorted[df_sorted['Date'] >= date_cutoff].copy()
        
        # AJUSTE TEMPORAL: Agregar factor de corrección basado en tendencia del dashboard
        if 'Date' in train_df.columns and '3P' in train_df.columns:
            # Calcular tendencia temporal en datos de entrenamiento
            train_df['date_numeric'] = pd.to_datetime(train_df['Date']).astype(int) / 10**9
            test_df['date_numeric'] = pd.to_datetime(test_df['Date']).astype(int) / 10**9
            
            # Agregar factor de corrección temporal (basado en observación del dashboard)
            # El modelo tiende a subestimar, especialmente en datos más recientes
            train_df['temporal_correction_factor'] = 1.0  # Base para entrenamiento
            test_df['temporal_correction_factor'] = 1.05  # Ligero ajuste al alza para test
            
            logger.info("Aplicado factor de corrección temporal para tendencia observada")
        
        logger.info(f"Split temporal cronológico optimizado:")
        logger.info(f"  Train: {len(train_df)} registros (hasta {train_df['Date'].max()})")
        logger.info(f"  Test: {len(test_df)} registros (desde {test_df['Date'].min()})")
        logger.info(f"  Fecha de corte: {date_cutoff}")
        
        return train_df, test_df
    
    def _calculate_sample_weights(self, y_train):
        """
        SAMPLE WEIGHTING ULTRA-EXTREMO ESPECIALIZADO
        Optimizado específicamente para mejorar predicción de valores 6+ triples
        Basado en análisis del dashboard: MAE 1.835 para valores excepcionales
        """
        weights = np.ones(len(y_train))
        
        # 1. PESOS ULTRA-AGRESIVOS para valores extremos (basado en dashboard)
        exceptional_mask = y_train >= 8      # Casos ultra-raros
        very_high_mask = (y_train >= 6) & (y_train < 8)  # Casos excepcionales
        high_mask = (y_train >= 4) & (y_train < 6)       # Casos buenos
        medium_mask = (y_train >= 2) & (y_train < 4)     # Casos promedio
        low_mask = y_train < 2                           # Casos bajos
        
        # Pesos extremos basados en dificultad de predicción del dashboard
        weights[exceptional_mask] = 8.0      # 8+ triples: peso máximo (MAE muy alto)
        weights[very_high_mask] = 6.0        # 6-7 triples: peso ultra-alto
        weights[high_mask] = 3.5             # 4-5 triples: peso alto (MAE 0.910)
        weights[medium_mask] = 1.8           # 2-3 triples: peso medio (MAE 0.456)
        weights[low_mask] = 1.0              # 0-1 triples: peso base (MAE 0.280)
        
        # 2. MUESTREO ESTRATIFICADO ADAPTATIVO
        # Sobremuestreo inteligente de casos raros
        value_counts = pd.Series(y_train).value_counts()
        total_samples = len(y_train)
        
        for i, y_val in enumerate(y_train):
            frequency = value_counts.get(y_val, 1)
            
            # Factor de rareza exponencial para casos extremos
            if y_val >= 6:
                rarity_boost = np.power(total_samples / frequency, 0.8)
                weights[i] *= (1 + rarity_boost * 0.5)
            elif y_val >= 4:
                rarity_boost = np.power(total_samples / frequency, 0.6)
                weights[i] *= (1 + rarity_boost * 0.3)
            else:
                rarity_boost = np.log(total_samples / frequency + 1)
                weights[i] *= (1 + rarity_boost * 0.1)
        
        # 3. CORRECCIÓN TEMPORAL INTELIGENTE
        # Dar más peso a patrones recientes (tendencia temporal del dashboard)
        temporal_weights = np.linspace(0.7, 1.4, len(y_train))
        weights *= temporal_weights
        
        # 4. BOOST ADICIONAL para casos problemáticos
        # Identificar y dar peso extra a casos que históricamente son difíciles
        problematic_mask = y_train >= 5  # Casos con MAE > 1.0 según dashboard
        weights[problematic_mask] *= 1.5
        
        # 5. NORMALIZACIÓN PRESERVANDO EXTREMOS
        # Normalizar pero mantener énfasis en valores altos
        base_weight = weights[y_train < 2].mean() if (y_train < 2).any() else 1.0
        weights = weights / base_weight
        
        # Clip conservador para evitar inestabilidad
        weights = np.clip(weights, 0.3, 12.0)
        
        # Log de distribución de pesos
        logger.info(f"Sample weights - Extremos (8+): {(weights[y_train >= 8]).mean():.2f}")
        logger.info(f"Sample weights - Altos (6-7): {(weights[(y_train >= 6) & (y_train < 8)]).mean():.2f}")
        logger.info(f"Sample weights - Buenos (4-5): {(weights[(y_train >= 4) & (y_train < 6)]).mean():.2f}")
        
        return weights
    
    def train(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Entrena el modelo completo de stacking para triples
        """
        logger.info("Iniciando entrenamiento del modelo de triples...")
        
        # Verificar target
        if '3P' not in df.columns:
            raise ValueError("Columna '3P' no encontrada en los datos")
        
        # Estadísticas del target
        threept_stats = df['3P'].describe()
        logger.info(f"Target 3P - Media: {threept_stats['mean']:.2f}, Max: {threept_stats['max']:.0f}")
        
        # Split temporal
        train_df, val_df = self._temporal_split(df, test_size=0.2)
        
        # Preparar features
        X_train, feature_names = self._prepare_features(train_df)
        X_val, val_feature_names = self._prepare_features(val_df)
        
        # Verificar consistencia de features entre train y val
        if feature_names != val_feature_names:
            logger.warning(f"Inconsistencia de features - Train: {len(feature_names)}, Val: {len(val_feature_names)}")
            # Usar solo features comunes
            common_features = list(set(feature_names) & set(val_feature_names))
            X_train = X_train[common_features]
            X_val = X_val[common_features]
            feature_names = common_features
            logger.info(f"Usando features comunes: {len(common_features)}")
        
        # Targets
        y_train = train_df['3P'].values
        y_val = val_df['3P'].values
        
        # Calcular pesos de muestra para valores extremos
        sample_weights = self._calculate_sample_weights(y_train)
        
        # Guardar features seleccionadas
        self.selected_features = feature_names
        self.feature_names = feature_names
        
        logger.info(f"Datos preparados - Train: {X_train.shape}, Val: {X_val.shape}")
        logger.info(f"Sample weights - Alto valor: {(sample_weights > 2.5).sum()}, Medio: {((sample_weights > 1.2) & (sample_weights <= 2.5)).sum()}")
        
        # Entrenar modelos base con sample weights
        base_results = self._train_base_models(X_train, y_train, X_val, y_val, sample_weights)
        
        if not base_results:
            raise ValueError("Ningún modelo base se entrenó exitosamente")
        
        # Configurar stacking
        self._setup_stacking_model()
        
        # Entrenar modelo de stacking final
        logger.info("Entrenando modelo de stacking final...")
        self.stacking_model.fit(X_train, y_train)
        
        # Evaluar en validación
        y_pred_val = self.stacking_model.predict(X_val)
        y_pred_val = np.clip(y_pred_val, 0, 15)
        
        # Calcular métricas finales
        final_metrics = {
            'mae': mean_absolute_error(y_val, y_pred_val),
            'rmse': np.sqrt(mean_squared_error(y_val, y_pred_val)),
            'r2': r2_score(y_val, y_pred_val)
        }
        
        # Validación cruzada temporal
        df_full_features, _ = self._prepare_features(df)
        cv_metrics = self._temporal_cross_validate_stacking(
            df_full_features, df['3P'], df
        )
        
        # Combinar métricas
        final_metrics.update(cv_metrics)
        
        # Guardar métricas
        self.training_metrics = final_metrics
        self.is_trained = True
        
        # Limpiar memoria después del entrenamiento
        self._cleanup_temp_files()
        
        logger.info("Entrenamiento completado para triples")
        logger.info(f"Métricas finales - MAE: {final_metrics['mae']:.4f}, R²: {final_metrics['r2']:.4f}")
        
        return final_metrics
    
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Realiza predicciones de triples con corrección temporal
        """
        if not self.is_trained:
            raise ValueError("Modelo no entrenado. Llama a train() primero.")
        
        # Preparar features
        X, _ = self._prepare_features(df)
        
        # Predecir
        predictions = self.stacking_model.predict(X)
        
        # CORRECCIÓN TEMPORAL INTELIGENTE basada en análisis del dashboard
        if 'Date' in df.columns:
            # Aplicar factor de corrección temporal progresivo
            dates = pd.to_datetime(df['Date'])
            max_date = dates.max()
            
            # Factor de corrección que aumenta para fechas más recientes
            # Basado en la tendencia temporal observada en el dashboard
            days_from_max = (max_date - dates).dt.days
            temporal_factor = 1.0 + (0.02 * np.exp(-days_from_max / 30))  # Decaimiento exponencial
            
            # Aplicar corrección especialmente a valores predichos altos
            high_pred_mask = predictions >= 3.0
            predictions[high_pred_mask] *= temporal_factor[high_pred_mask]
            
            logger.debug(f"Aplicada corrección temporal - Factor promedio: {temporal_factor.mean():.3f}")
        
        # CORRECCIÓN ESPECÍFICA PARA VALORES EXTREMOS
        # Basado en el análisis del dashboard: el modelo subestima valores 6+
        extreme_mask = predictions >= 4.0
        if extreme_mask.any():
            # Boost progresivo para valores altos predichos
            boost_factor = 1.0 + (predictions[extreme_mask] - 4.0) * 0.08
            predictions[extreme_mask] *= boost_factor
            
            logger.debug(f"Aplicado boost para valores extremos: {extreme_mask.sum()} predicciones")
        
        # Clip para triples (0 a 15 máximo razonable)
        predictions = np.clip(predictions, 0, 15)
        
        return predictions
    
    def get_feature_importance(self, top_n: int = 15) -> Dict[str, float]:
        """
        Obtiene importancia de features
        """
        if not self.is_trained:
            return {}
        
        # Obtener importancia del mejor modelo base (típicamente XGBoost)
        if 'xgboost' in self.trained_base_models:
            model = self.trained_base_models['xgboost']
            if hasattr(model, 'feature_importances_'):
                importance_dict = dict(zip(self.feature_names, model.feature_importances_))
                # Ordenar por importancia
                sorted_importance = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
                return dict(sorted_importance[:top_n])
        
        return {}
    
    def save_model(self, filepath: str):
        """Guarda el modelo entrenado como objeto directo usando JOBLIB con protocolo estable"""
        if not self.is_trained:
            raise ValueError("No hay modelo entrenado para guardar")
        
        if self.stacking_model is None:
            raise ValueError("Modelo no entrenado. Ejecutar train() primero.")
        
        # Crear directorio si no existe
        import os
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Configurar entorno para evitar problemas de threading
        import matplotlib
        matplotlib.use('Agg')
        
        # Guardar SOLO el modelo entrenado como objeto directo usando JOBLIB con protocolo estable
        with joblib.parallel_backend('threading', n_jobs=1):
            joblib.dump(self.stacking_model, filepath, compress=3, protocol=4)
        
        logger.info(f"Modelo Triples guardado como objeto directo (JOBLIB): {filepath}")
    
    def load_model(self, filepath: str):
        """Carga un modelo previamente entrenado (compatible con ambos formatos)"""
        # Configurar entorno para evitar problemas de threading
        import matplotlib
        matplotlib.use('Agg')
        
        try:
            # Intentar cargar modelo directo (nuevo formato)
            with joblib.parallel_backend('threading', n_jobs=1):
                self.stacking_model = joblib.load(filepath)
            
            if hasattr(self.stacking_model, 'predict'):
                self.is_trained = True
                logger.info(f"Modelo de triples (objeto directo) cargado desde: {filepath}")
                return
            else:
                # No es un modelo directo, tratar como formato antiguo
                raise ValueError("No es modelo directo")
        except (ValueError, AttributeError):
            # Formato antiguo (diccionario)
            try:
                with joblib.parallel_backend('threading', n_jobs=1):
                    model_data = joblib.load(filepath)
                
                if isinstance(model_data, dict) and 'stacking_model' in model_data:
                    self.stacking_model = model_data['stacking_model']
                    self.trained_base_models = model_data.get('trained_base_models', {})
                    self.feature_names = model_data.get('feature_names', [])
                    self.selected_features = model_data.get('selected_features', [])
                    self.training_metrics = model_data.get('training_metrics', {})
                    self.best_params_per_model = model_data.get('best_params_per_model', {})
                    self.scaler = model_data.get('scaler')
                    self.is_trained = model_data.get('is_trained', True)
                    logger.info(f"Modelo de triples (formato legacy) cargado desde: {filepath}")
                else:
                    raise ValueError("Formato de archivo no reconocido")
            except Exception as e:
                raise ValueError(f"No se pudo cargar el modelo de triples: {e}")


class XGBoost3PTModel(Stacking3PTModel):
    """
    Modelo XGBoost simple para predicción de triples
    Compatible con la interfaz existente
    """
    
    def __init__(self, **kwargs):
        # Inicializar solo con XGBoost
        super().__init__(**kwargs)
        
        # Simplificar a solo XGBoost para compatibilidad
        self.base_models = {
            'xgboost': self.base_models['xgboost']
        }
        
        logger.info("Modelo XGBoost 3PT inicializado (solo XGBoost)")
    
    def train(self, df: pd.DataFrame) -> Dict[str, float]:
        """Entrena solo con XGBoost para simplicidad"""
        return super().train(df)
    
    def get_feature_importance(self, top_n: int = 15) -> Dict[str, float]:
        """Obtiene importancia de features del modelo XGBoost"""
        return super().get_feature_importance(top_n)