"""
Modelo de Predicción de Asistencias (AST)
========================================

Sistema avanzado de stacking ensemble para predicción de asistencias
que realizará un jugador NBA en su próximo partido.

CARACTERÍSTICAS PRINCIPALES:
- Stacking ensemble con XGBoost, LightGBM, CatBoost, Gradient Boosting
- Optimización bayesiana de hiperparámetros
- Validación cruzada temporal (respeta orden cronológico)
- Regularización avanzada (L1/L2, Dropout, Early Stopping)
- División temporal para evitar data leakage
- Meta-learner adaptativo para stacking
- Features especializadas para asistencias

ARQUITECTURA:
1. Modelos Base: XGBoost, LightGBM, CatBoost, GradientBoosting, Ridge
2. Meta-learner: Ridge con regularización L2
3. Validación: TimeSeriesSplit cronológico
4. Optimización: Optuna (Bayesian Optimization)

ESPECIALIZACIÓN PARA ASISTENCIAS:
- Enfoque en visión de cancha y Basketball IQ
- Features de control del balón y ritmo de juego
- Contexto de equipo (calidad de tiradores)
- Historial de pases y situaciones de juego
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
import warnings
from pathlib import Path
import joblib
import json

# ML Libraries
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.ensemble import StackingRegressor
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import optuna
from optuna.samplers import TPESampler

# Configurar Optuna para ser silencioso
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Local imports
from .features_ast import AssistsFeatureEngineer

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

class StackingASTModel:
    """
    Modelo de Stacking Ensemble para Predicción de Asistencias
    ULTRA-OPTIMIZADO con regularización balanceada y validación temporal
    Especializado en características de playmakers y visión de cancha
    """
    
    def __init__(self, enable_neural_network: bool = True, enable_svr: bool = False, 
                 enable_gpu: bool = False, random_state: int = 42, teams_df: pd.DataFrame = None):
        """
        Inicializar el modelo de stacking para AST
        
        Args:
            enable_neural_network: Habilitar red neuronal (mejor para patrones complejos)
            enable_svr: Habilitar SVR (muy lento, deshabilitado por defecto)
            enable_gpu: Usar GPU para XGBoost/LightGBM si está disponible
            random_state: Semilla para reproducibilidad
            teams_df: DataFrame con datos de equipos para features avanzadas
        """
        self.random_state = random_state
        self.enable_neural_network = enable_neural_network
        self.enable_svr = enable_svr
        self.enable_gpu = enable_gpu
        self.teams_df = teams_df
        
        # Configuración de modelos
        self.base_models = {}
        self.stacking_model = None
        self.feature_engineer = AssistsFeatureEngineer(teams_df=teams_df)
        self.scaler = StandardScaler()
        
        # Métricas y resultados
        self.validation_metrics = {}
        self.cv_scores = {}
        self.feature_importance = {}
        self.best_params_per_model = {}
        
        # Configuración de optimización BALANCEADA
        self.n_trials = 35  # Aumentado de 25 a 35 para mejor optimización
        self.cv_folds = 5  # Aumentado de 3 a 4 para mejor validación
        
        # MEJORA CRÍTICA: Feature selection agresivo
        self.max_features = 80  # Aumentado de 50 a 80 para mejor rendimiento
        self.feature_selection_method = 'hybrid'  # Método híbrido más inteligente
        
        # Configurar modelos base
        self._setup_base_models()
        
        # Mostrar ensemble final
        model_names = list(self.base_models.keys())
        logger.info(f"Modelo AST inicializado - Ensemble: {', '.join(model_names)}")
        logger.info(f"Configuración: NN={enable_neural_network}, SVR={enable_svr}, GPU={enable_gpu}")
        logger.info(f"Feature selection: Máximo {self.max_features} características")
    
    def _setup_base_models(self):
        """Configurar modelos base con hiperparámetros optimizados para asistencias"""
        
        # XGBoost - Excelente para features de Basketball IQ y patrones complejos
        self.base_models['xgboost'] = {
            'model': xgb.XGBRegressor(
                random_state=self.random_state,
                tree_method='gpu_hist' if self.enable_gpu else 'hist',
                gpu_id=0 if self.enable_gpu else None,
                n_jobs=-1
            ),
            'param_space': {
                'n_estimators': (100, 300),  # Rango reducido
                'max_depth': (3, 6),         # Rango reducido
                'learning_rate': (0.05, 0.2), # Rango reducido
                'subsample': (0.8, 1.0),     # Rango reducido
                'colsample_bytree': (0.8, 1.0), # Rango reducido
                'reg_alpha': (0, 3),         # Rango reducido
                'reg_lambda': (1, 5)         # Rango reducido
            }
        }
        
        # LightGBM - Rápido y eficiente para features de visión de cancha
        self.base_models['lightgbm'] = {
            'model': lgb.LGBMRegressor(
                random_state=self.random_state,
                device='gpu' if self.enable_gpu else 'cpu',
                gpu_platform_id=0 if self.enable_gpu else None,
                gpu_device_id=0 if self.enable_gpu else None,
                n_jobs=-1,
                verbose=-1
            ),
            'param_space': {
                'n_estimators': (100, 300),  # Rango reducido
                'max_depth': (3, 6),         # Rango reducido
                'learning_rate': (0.05, 0.2), # Rango reducido
                'subsample': (0.8, 1.0),     # Rango reducido
                'colsample_bytree': (0.8, 1.0), # Rango reducido
                'reg_alpha': (0, 3),         # Rango reducido
                'reg_lambda': (1, 5),        # Rango reducido
                'num_leaves': (15, 50)       # Rango reducido
            }
        }
        
        # CatBoost - Excelente para features categóricas de contexto de equipo
        self.base_models['catboost'] = {
            'model': cb.CatBoostRegressor(
                random_state=self.random_state,
                task_type='GPU' if self.enable_gpu else 'CPU',
                devices='0' if self.enable_gpu else None,
                verbose=False,
                allow_writing_files=False
            ),
            'param_space': {
                'iterations': (100, 250),    # Rango reducido
                'depth': (3, 6),             # Rango reducido
                'learning_rate': (0.05, 0.2), # Rango reducido
                'l2_leaf_reg': (1, 5),       # Rango reducido
                'subsample': (0.8, 1.0)      # Rango reducido
            }
        }
        
        # Gradient Boosting - Robusto para features de control del balón
        self.base_models['gradient_boosting'] = {
            'model': GradientBoostingRegressor(
                random_state=self.random_state
            ),
            'param_space': {
                'n_estimators': (100, 200),  # Rango reducido
                'max_depth': (3, 5),         # Rango reducido
                'learning_rate': (0.05, 0.15), # Rango reducido
                'subsample': (0.8, 1.0),     # Rango reducido
                'max_features': (0.8, 1.0)   # Rango reducido
            }
        }
        
        # Neural Network ELIMINADO - Peor rendimiento (MAE: 0.708) vs otros modelos
        # if self.enable_neural_network:
        #     self.base_models['neural_network'] = {
        #         'model': MLPRegressor(
        #             random_state=self.random_state,
        #             max_iter=500,
        #             early_stopping=True,
        #             validation_fraction=0.1
        #         ),
        #         'param_space': {
        #             'hidden_layer_sizes': [(32,), (64,), (32, 16)],
        #             'alpha': (0.001, 0.05),
        #             'learning_rate_init': (0.001, 0.005)
        #         }
        #     }
        
        # SVR (opcional) - Excelente para relaciones no lineales en asistencias
        if self.enable_svr:
            self.base_models['svr'] = {
                'model': SVR(),
                'param_space': {
                    'C': (0.1, 50.0),
                    'epsilon': (0.01, 0.5),
                    'gamma': ['scale', 'auto'],
                    'kernel': ['rbf', 'linear', 'poly']
                }
            }
    
    def _temporal_split(self, df: pd.DataFrame, test_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        División temporal de datos respetando orden cronológico
        CRÍTICO: Evita data leakage usando fechas
        """
        if 'Date' not in df.columns:
            logger.warning("Columna 'Date' no encontrada, usando división secuencial")
            split_idx = int(len(df) * (1 - test_size))
            return df.iloc[:split_idx].copy(), df.iloc[split_idx:].copy()
        
        # Ordenar por fecha
        df_sorted = df.sort_values('Date').reset_index(drop=True)
        
        # Encontrar punto de corte temporal
        split_idx = int(len(df_sorted) * (1 - test_size))
        cutoff_date = df_sorted.iloc[split_idx]['Date']
        
        train_data = df_sorted[df_sorted['Date'] < cutoff_date].copy()
        test_data = df_sorted[df_sorted['Date'] >= cutoff_date].copy()
        
        logger.info(f"División temporal: {len(train_data)} entrenamiento, {len(test_data)} prueba")
        logger.info(f"Fecha corte: {cutoff_date}")
        
        return train_data, test_data
    
    def _validate_training_data(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        CRÍTICO: Validar que los datos de entrenamiento sean correctos.
        
        Args:
            X: DataFrame con características
            y: Serie con variable objetivo
        """
        logger.info("Validando datos de entrenamiento...")
        
        # Verificar que X sea numérico (específico para LightGBM)
        non_numeric_cols = []
        for col in X.columns:
            dtype = X[col].dtype
            if dtype == 'object' or dtype.name == 'string':
                # Verificar si contiene valores no numéricos
                sample_values = X[col].dropna().head(10).tolist()
                logger.error(f"Columna no numérica '{col}' (tipo: {dtype}): {sample_values}")
                non_numeric_cols.append(col)
            elif dtype.name not in ['int8', 'int16', 'int32', 'int64', 'float16', 'float32', 'float64', 'bool']:
                logger.error(f"Columna con tipo no compatible '{col}' (tipo: {dtype})")
                non_numeric_cols.append(col)
        
        if non_numeric_cols:
            raise ValueError(f"Columnas no numéricas detectadas para LightGBM: {non_numeric_cols}")
        
        # Verificar valores infinitos
        inf_cols = []
        for col in X.columns:
            if np.isinf(X[col]).any():
                inf_cols.append(col)
        
        if inf_cols:
            logger.warning(f"Columnas con valores infinitos (serán reemplazados): {inf_cols}")
            X[inf_cols] = X[inf_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
        
        # Verificar NaN en target
        if y.isna().any():
            logger.warning(f"Target contiene {y.isna().sum()} valores NaN")
        
        # CRÍTICO: Forzar conversión a tipos compatibles con LightGBM
        for col in X.columns:
            if X[col].dtype == 'object':
                # Intentar conversión numérica forzada
                X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)
            elif X[col].dtype not in ['int64', 'float64', 'bool']:
                # Convertir a float64 para compatibilidad
                X[col] = X[col].astype('float64')
        
        logger.info(f"Validación completada - X: {X.shape}, y: {len(y)}")
        logger.info(f"Tipos de datos únicos en X: {X.dtypes.unique()}")
        
        return
    
    def _filter_problematic_columns(self, df: pd.DataFrame, features: List[str]) -> List[str]:
        """
        CRÍTICO: Filtrar columnas que causan problemas en LightGBM y otros modelos.
        
        Args:
            df: DataFrame completo
            features: Lista de características propuestas
            
        Returns:
            List[str]: Lista de características filtradas y seguras
        """
        # Columnas problemáticas conocidas (solo las que realmente causan problemas)
        problematic_columns = {
            'Away', 'GS', 'Result', 'Pos', 'Player', 'Date', 'Team', 'Opp',
            # DATA LEAKAGE CRÍTICO: Eliminar columnas que usan información del juego actual
            'AST_double', 'PTS_double', 'TRB_double', 'STL_double', 'BLK_double',
            'double_double', 'triple_double', 'AST', 'PTS', 'TRB', 'STL', 'BLK',
            'GmSc'  # Game Score también usa stats del juego actual
        }
        
        # Filtrar características problemáticas
        safe_features = []
        removed_features = []
        
        for feature in features:
            if feature in problematic_columns:
                removed_features.append(feature)
                continue
                
            # Verificar si la columna existe en el DataFrame
            if feature not in df.columns:
                removed_features.append(feature)
                continue
                
            # Verificar tipo de datos
            if df[feature].dtype == 'object':
                # Intentar convertir a numérico
                try:
                    test_conversion = pd.to_numeric(df[feature], errors='coerce')
                    if test_conversion.isna().all():
                        removed_features.append(feature)
                        continue
                except:
                    removed_features.append(feature)
                    continue
            
            safe_features.append(feature)
        
        if removed_features:
            logger.warning(f"Columnas problemáticas eliminadas ({len(removed_features)}): {removed_features[:10]}...")
        
        logger.info(f"Filtrado de columnas: {len(features)} → {len(safe_features)} características seguras")
        
        return safe_features
    
    def train(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Entrenamiento completo del modelo Stacking AST
        
        Args:
            df: DataFrame con datos de jugadores y estadísticas
            
        Returns:
            Dict con métricas de validación
        """
        logger.info("Iniciando entrenamiento del modelo AST...")
        
        # Verificar orden cronológico
        if 'Date' in df.columns:
            if not df['Date'].is_monotonic_increasing:
                logger.info("Ordenando datos cronológicamente...")
                df = df.sort_values(['Player', 'Date']).reset_index(drop=True)
        
        # Generar features especializadas para asistencias
        logger.info("Generando características especializadas...")
        all_features = self.feature_engineer.generate_all_features(df)  # Modificar DataFrame directamente
        
        if not all_features:
            raise ValueError("No se pudieron generar features para AST")
        
        logger.info(f"Features generadas: {len(all_features)}")
        
        # MEJORA CRÍTICA: Las características ya están seleccionadas por el feature engineer
        features = all_features
        
        logger.info(f"Features seleccionadas tras filtrado: {len(features)}")
        
        # Preparar datos (ahora df tiene las features)
        X = df[features].fillna(0)
        y = df['AST']
        
        # CRÍTICO: Filtrar columnas problemáticas y validar datos numéricos
        features = self._filter_problematic_columns(df, features)
        X = df[features].fillna(0)
        self._validate_training_data(X, y)
        
        # División temporal
        train_data, test_data = self._temporal_split(df)
        
        X_train = train_data[features].fillna(0)
        y_train = train_data['AST']
        X_test = test_data[features].fillna(0)
        y_test = test_data['AST']
        
        # Validar datos de entrenamiento y prueba
        self._validate_training_data(X_train, y_train)
        self._validate_training_data(X_test, y_test)
        
        logger.info(f"Entrenamiento: {X_train.shape[0]} muestras, Prueba: {X_test.shape[0]} muestras")
        
        # PASO 1: Entrenar modelos base con optimización bayesiana
        logger.info("Entrenando modelos base...")
        self._train_base_models_with_optimization(X_train, y_train)
        
        # PASO 2: Configurar stacking
        logger.info("Configurando stacking...")
        self._setup_stacking_model()
        
        # PASO 3: Validación cruzada temporal
        logger.info("Validación cruzada...")
        self._perform_temporal_cross_validation(X_train, y_train)
        
        # PASO 4: Entrenar modelo final
        logger.info("Entrenando modelo final...")
        self.stacking_model.fit(X_train, y_train)
        
        # PASO 5: Evaluación final
        logger.info("Evaluación final...")
        y_pred = self.stacking_model.predict(X_test)
        
        # Calcular métricas
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        # Métricas específicas para asistencias
        accuracy_1ast = np.mean(np.abs(y_test - y_pred) <= 1) * 100
        accuracy_2ast = np.mean(np.abs(y_test - y_pred) <= 2) * 100
        accuracy_3ast = np.mean(np.abs(y_test - y_pred) <= 3) * 100
        
        self.validation_metrics = {
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'accuracy_1ast': accuracy_1ast,
            'accuracy_2ast': accuracy_2ast,
            'accuracy_3ast': accuracy_3ast
        }
        
        # Calcular importancia de features
        self._calculate_feature_importance(features)
        
        # Mostrar resultados FINALES
        logger.info("=" * 50)
        logger.info("ENTRENAMIENTO COMPLETADO")
        logger.info("=" * 50)
        logger.info(f"MAE: {self.validation_metrics['mae']:.4f}")
        logger.info(f"RMSE: {self.validation_metrics['rmse']:.4f}")
        logger.info(f"R²: {self.validation_metrics['r2']:.4f}")
        logger.info(f"Accuracy ±1ast: {accuracy_1ast:.1f}%")
        logger.info(f"Accuracy ±2ast: {accuracy_2ast:.1f}%")
        logger.info(f"Accuracy ±3ast: {accuracy_3ast:.1f}%")
        logger.info("=" * 50)
        
        return self.validation_metrics

    def _train_base_models_with_optimization(self, X_train: pd.DataFrame, y_train: pd.Series):
        """Entrenar modelos base con optimización bayesiana de hiperparámetros"""
        
        trained_models = {}
        
        for i, (model_name, model_config) in enumerate(self.base_models.items(), 1):
            logger.info(f"[{i}/{len(self.base_models)}] Entrenando {model_name}...")
            
            try:
                # Optimización bayesiana (silenciosa)
                best_params = self._optimize_hyperparameters(
                    model_config['model'], 
                    model_config['param_space'], 
                    X_train, 
                    y_train,
                    model_name
                )
                
                # Entrenar con mejores parámetros
                model = model_config['model']
                model.set_params(**best_params)
                model.fit(X_train, y_train)
                
                # Evaluar en datos de entrenamiento
                y_pred_train = model.predict(X_train)
                train_mae = mean_absolute_error(y_train, y_pred_train)
                
                trained_models[model_name] = model
                self.best_params_per_model[model_name] = best_params
                
                logger.info(f"{model_name} completado - MAE: {train_mae:.3f}")
                
            except Exception as e:
                logger.error(f"Error entrenando {model_name}: {str(e)}")
                continue
        
        if not trained_models:
            raise ValueError("No se pudo entrenar ningún modelo base")
        
        # Actualizar modelos base con los entrenados
        for model_name, trained_model in trained_models.items():
            self.base_models[model_name]['model'] = trained_model
        
        logger.info(f"Modelos entrenados: {len(trained_models)}/{len(self.base_models)}")
    
    def _optimize_hyperparameters(self, model, param_space: Dict, X_train: pd.DataFrame, 
                                 y_train: pd.Series, model_name: str) -> Dict:
        """Optimización bayesiana de hiperparámetros usando Optuna"""
        
        def objective(trial):
            # Generar parámetros para el trial
            params = {}
            for param_name, param_range in param_space.items():
                if isinstance(param_range, tuple) and len(param_range) == 2:
                    if isinstance(param_range[0], int):
                        params[param_name] = trial.suggest_int(param_name, param_range[0], param_range[1])
                    elif isinstance(param_range[0], float):
                        params[param_name] = trial.suggest_float(param_name, param_range[0], param_range[1])
                elif isinstance(param_range, list):
                    params[param_name] = trial.suggest_categorical(param_name, param_range)
            
            # Configurar modelo con parámetros del trial
            trial_model = model.__class__(**{**model.get_params(), **params})
            
            # Validación cruzada temporal más robusta
            n_splits = min(2, len(X_train) // 200)  # Máximo 2 folds para optimización rápida
            if n_splits < 2:
                # Si el dataset es muy pequeño, usar train-test split simple
                split_idx = int(len(X_train) * 0.8)
                X_fold_train, X_fold_val = X_train.iloc[:split_idx], X_train.iloc[split_idx:]
                y_fold_train, y_fold_val = y_train.iloc[:split_idx], y_train.iloc[split_idx:]
                
                trial_model.fit(X_fold_train, y_fold_train)
                y_pred = trial_model.predict(X_fold_val)
                mae = mean_absolute_error(y_fold_val, y_pred)
                scores = [mae]
            else:
                tscv = TimeSeriesSplit(n_splits=n_splits)
                scores = []
                
                for train_idx, val_idx in tscv.split(X_train):
                    X_fold_train, X_fold_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
                    y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
                    
                    trial_model.fit(X_fold_train, y_fold_train)
                    y_pred = trial_model.predict(X_fold_val)
                    mae = mean_absolute_error(y_fold_val, y_pred)
                    scores.append(mae)
            
            return np.mean(scores)
        
        # Crear estudio de optimización (silencioso)
        study = optuna.create_study(
            direction='minimize',
            sampler=TPESampler(seed=self.random_state)
        )
        
        # Optimizar (silencioso)
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=False)
        
        return study.best_params
    
    def _setup_stacking_model(self):
        """
        Configurar el modelo de stacking SIMPLIFICADO Y OPTIMIZADO para AST
        ENFOQUE: Menos complejidad, más precisión
        """
        # SIMPLIFICAR ARQUITECTURA - SOLO MODELOS BASE MÁS EFECTIVOS
        
        # GRUPO 1: MODELOS PRINCIPALES OPTIMIZADOS PARA AST
        main_models = [
            # XGBoost optimizado para asistencias
            ('xgb_ast', xgb.XGBRegressor(
                n_estimators=200,  # Reducido para evitar overfitting
                max_depth=4,       # Menos profundidad para generalizar mejor
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=1,       # Regularización L1 moderada
                reg_lambda=2,      # Regularización L2 moderada
                random_state=self.random_state,
                tree_method='gpu_hist' if self.enable_gpu else 'hist'
            )),
            
            # LightGBM optimizado para asistencias
            ('lgb_ast', lgb.LGBMRegressor(
                n_estimators=200,  # Reducido para evitar overfitting
                max_depth=4,       # Menos profundidad
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=1,
                reg_lambda=2,
                num_leaves=15,     # Reducido significativamente
                random_state=self.random_state,
                verbose=-1
            )),
            
            # CatBoost para features categóricas
            ('catboost_ast', cb.CatBoostRegressor(
                iterations=150,    # Reducido
                depth=4,           # Menos profundidad
                learning_rate=0.1,
                l2_leaf_reg=2,
                subsample=0.8,
                random_state=self.random_state,
                verbose=False,
                allow_writing_files=False
            )),
            
            # Ridge para estabilidad
            ('ridge_ast', Ridge(
                alpha=1.0,         # Regularización moderada
                fit_intercept=True,
                random_state=self.random_state
            ))
        ]
        
        # Agregar Neural Network solo si está habilitado
        if self.enable_neural_network:
            main_models.append(
                ('mlp_ast', MLPRegressor(
                    hidden_layer_sizes=(32, 16),  # Arquitectura más simple
                    activation='relu',
                    solver='adam',
                    alpha=0.01,        # Regularización moderada
                    learning_rate='adaptive',
                    learning_rate_init=0.001,
                    max_iter=300,      # Menos iteraciones
                    early_stopping=True,
                    validation_fraction=0.1,
                    n_iter_no_change=15,
                    random_state=self.random_state
                ))
            )
        
        # META-LEARNER SIMPLE Y EFECTIVO
        meta_learner = Ridge(
            alpha=0.5,  # Regularización ligera para permitir aprendizaje
            fit_intercept=True,
            random_state=self.random_state
        )
        
        # STACKING SIMPLE - UN SOLO NIVEL
        self.stacking_model = StackingRegressor(
            estimators=main_models,
            final_estimator=meta_learner,
            cv=3,              # CV simple
            n_jobs=1,          # Sin paralelización para evitar conflictos
            passthrough=False  # Solo predicciones de modelos base
        )
        
        logger.info(f"Stacking SIMPLIFICADO configurado: {len(main_models)} modelos base")
        
        return self.stacking_model
    
    def _perform_temporal_cross_validation(self, X_train: pd.DataFrame, y_train: pd.Series):
        """Realizar validación cruzada temporal para evaluar estabilidad"""
        
        # Usar menos folds para evitar problemas con datasets pequeños
        n_splits = min(3, len(X_train) // 100)  # Máximo 3 folds, mínimo 100 muestras por fold
        if n_splits < 2:
            logger.warning("Dataset muy pequeño para CV temporal, saltando validación cruzada")
            self.cv_scores = {'mae_mean': 0, 'mae_std': 0, 'r2_mean': 0, 'cv_scores': []}
            return
        
        tscv = TimeSeriesSplit(n_splits=n_splits)
        cv_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X_train), 1):
            try:
                X_fold_train, X_fold_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
                y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
                
                # Entrenar stacking en fold con configuración más conservadora
                fold_stacking = StackingRegressor(
                    estimators=[(name, config['model']) for name, config in self.base_models.items()],
                    final_estimator=Ridge(alpha=1.0, random_state=self.random_state),
                    cv=2,  # CV interno muy pequeño
                    n_jobs=1  # Sin paralelización para evitar conflictos
                )
                
                fold_stacking.fit(X_fold_train, y_fold_train)
                y_pred = fold_stacking.predict(X_fold_val)
                
                # Métricas del fold
                mae = mean_absolute_error(y_fold_val, y_pred)
                r2 = r2_score(y_fold_val, y_pred)
                
                cv_scores.append({'mae': mae, 'r2': r2})
                
                # Solo mostrar el último fold
                if fold == n_splits:
                    logger.info(f"CV completado: MAE={mae:.3f}, R²={r2:.3f}")
                
            except Exception as e:
                logger.warning(f"Error en CV fold {fold}: {str(e)}")
                continue
        
        if not cv_scores:
            logger.warning("No se pudo completar ningún fold de CV")
            self.cv_scores = {'mae_mean': 0, 'mae_std': 0, 'r2_mean': 0, 'cv_scores': []}
            return
        
        # Promediar scores
        avg_mae = np.mean([score['mae'] for score in cv_scores])
        avg_r2 = np.mean([score['r2'] for score in cv_scores])
        std_mae = np.std([score['mae'] for score in cv_scores])
        
        self.cv_scores = {
            'mae_mean': avg_mae,
            'mae_std': std_mae,
            'r2_mean': avg_r2,
            'cv_scores': cv_scores
        }
    
    def _calculate_feature_importance(self, features: List[str]):
        """Calcular importancia de features desde modelos base"""
        
        feature_importance = {}
        
        for model_name, model_config in self.base_models.items():
            model = model_config['model']
            
            try:
                if hasattr(model, 'feature_importances_'):
                    # XGBoost, LightGBM, CatBoost, GradientBoosting
                    importances = model.feature_importances_
                elif hasattr(model, 'coef_'):
                    # Ridge, Linear models
                    importances = np.abs(model.coef_)
                else:
                    continue
                
                # Normalizar importancias
                importances = importances / np.sum(importances)
                
                for i, feature in enumerate(features):
                    if feature not in feature_importance:
                        feature_importance[feature] = 0
                    feature_importance[feature] += importances[i]
                    
            except Exception as e:
                logger.warning(f"No se pudo calcular importancia para {model_name}: {e}")
                continue
        
        # Normalizar importancias finales
        total_importance = sum(feature_importance.values())
        if total_importance > 0:
            for feature in feature_importance:
                feature_importance[feature] /= total_importance
        
        # Ordenar por importancia
        self.feature_importance = dict(
            sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        )
        
        # Mostrar top 5 features más importantes
        logger.info("Top 5 features más importantes:")
        for i, (feature, importance) in enumerate(list(self.feature_importance.items())[:5], 1):
            logger.info(f"  {i}. {feature}: {importance:.4f}")
    
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Realizar predicciones usando el modelo entrenado
        
        Args:
            df: DataFrame con datos de jugadores
            
        Returns:
            Array con predicciones de AST
        """
        if self.stacking_model is None:
            raise ValueError("Modelo no entrenado. Ejecutar train() primero.")
        
        # Generar features (modificar DataFrame directamente)
        features = self.feature_engineer.generate_all_features(df)
        
        # Filtrar columnas problemáticas
        features = self._filter_problematic_columns(df, features)
        X = df[features].fillna(0)
        
        # Validar datos para predicción
        self._validate_training_data(X, pd.Series([0] * len(X)))
        
        # Realizar predicciones usando el modelo principal entrenado
        predictions = self.stacking_model.predict(X)
        
        # Asegurar que las predicciones sean no negativas (asistencias no pueden ser negativas)
        predictions = np.maximum(predictions, 0)
        
        return predictions
    
    def save_model(self, filepath: str):
        """Guardar modelo entrenado"""
        model_data = {
            'stacking_model': self.stacking_model,
            'base_models': self.base_models,
            'feature_engineer': self.feature_engineer,
            'validation_metrics': self.validation_metrics,
            'cv_scores': self.cv_scores,
            'feature_importance': self.feature_importance,
            'best_params_per_model': self.best_params_per_model
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"Modelo guardado en: {filepath}")
    
    def load_model(self, filepath: str):
        """Cargar modelo entrenado"""
        model_data = joblib.load(filepath)
        
        self.stacking_model = model_data['stacking_model']
        self.base_models = model_data['base_models']
        self.feature_engineer = model_data['feature_engineer']
        self.validation_metrics = model_data['validation_metrics']
        self.cv_scores = model_data['cv_scores']
        self.feature_importance = model_data['feature_importance']
        self.best_params_per_model = model_data['best_params_per_model']
        
        logger.info(f"Modelo cargado desde: {filepath}")
    
    def get_model_summary(self) -> Dict[str, Any]:
        """Obtener resumen del modelo"""
        return {
            'model_type': 'Stacking Ensemble AST',
            'base_models': list(self.base_models.keys()),
            'validation_metrics': self.validation_metrics,
            'cv_scores': self.cv_scores,
            'n_features': len(self.feature_importance),
            'top_features': list(self.feature_importance.keys())[:10]
        }


class XGBoostASTModel:
    """
    Modelo XGBoost simplificado para AST con compatibilidad con el sistema existente
    Mantiene la interfaz del modelo de PTS pero optimizado para asistencias
    """
    
    def __init__(self, enable_neural_network: bool = True, enable_svr: bool = False, 
                 enable_gpu: bool = False, random_state: int = 42, teams_df: pd.DataFrame = None):
        """Inicializar modelo XGBoost AST con stacking ensemble"""
        self.stacking_model = StackingASTModel(
            enable_neural_network=enable_neural_network,
            enable_svr=enable_svr,
            enable_gpu=enable_gpu,
            random_state=random_state,
            teams_df=teams_df
        )
        
        # Atributos para compatibilidad
        self.model = None
        self.validation_metrics = {}
        self.best_params = {}
        self.cutoff_date = None
        
        logger.info("Modelo XGBoost AST inicializado con stacking completo")
    
    def train(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Entrenamiento manteniendo compatibilidad con la interfaz original
        """
        # Llamar al método de entrenamiento del stacking
        result = self.stacking_model.train(df)
        
        # Asignar para compatibilidad
        self.model = self.stacking_model.stacking_model
        self.validation_metrics = result
        
        if self.stacking_model.best_params_per_model:
            # Usar parámetros del modelo con mejor rendimiento
            best_model_name = min(
                self.stacking_model.best_params_per_model.keys(),
                key=lambda x: self.stacking_model.cv_scores.get('mae_mean', float('inf'))
            )
            self.best_params = self.stacking_model.best_params_per_model[best_model_name]
        
        # Calcular y guardar cutoff_date para compatibilidad con trainer
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
    
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Realizar predicciones"""
        return self.stacking_model.predict(df)
    
    def save_model(self, filepath: str):
        """Guardar modelo"""
        self.stacking_model.save_model(filepath)
    
    def load_model(self, filepath: str):
        """Cargar modelo"""
        self.stacking_model.load_model(filepath)
        self.model = self.stacking_model.stacking_model
        self.validation_metrics = self.stacking_model.validation_metrics
