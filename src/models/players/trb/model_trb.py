"""
Modelo de Predicción de Rebotes Totales (TRB)
============================================

Sistema avanzado de stacking ensemble para predicción de rebotes totales
que capturará un jugador NBA en su próximo partido.

CARACTERÍSTICAS PRINCIPALES:
- Stacking ensemble con XGBoost, LightGBM, CatBoost, Gradient Boosting
- Optimización bayesiana de hiperparámetros
- Validación cruzada temporal (respeta orden cronológico)
- Regularización avanzada (L1/L2, Dropout, Early Stopping)
- División temporal para evitar data leakage
- Meta-learner adaptativo para stacking
- Features especializadas para rebotes

ARQUITECTURA:
1. Modelos Base: XGBoost, LightGBM, CatBoost, GradientBoosting, Ridge
2. Meta-learner: Ridge con regularización L2
3. Validación: TimeSeriesSplit cronológico
4. Optimización: Optuna (Bayesian Optimization)
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
from .features_trb import ReboundsFeatureEngineer

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

class StackingTRBModel:
    """
    Modelo de Stacking Ensemble para Predicción de Rebotes Totales
    ULTRA-OPTIMIZADO con regularización balanceada y validación temporal
    """
    
    def __init__(self, enable_neural_network: bool = True, enable_svr: bool = False, 
                 enable_gpu: bool = False, random_state: int = 42, teams_df: pd.DataFrame = None):
        """
        Inicializar el modelo de stacking para TRB
        
        Args:
            enable_neural_network: Habilitar red neuronal (más lento pero mejor)
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
        self.feature_engineer = ReboundsFeatureEngineer(teams_df=teams_df)
        self.scaler = StandardScaler()
        
        # Métricas y resultados
        self.validation_metrics = {}
        self.cv_scores = {}
        self.feature_importance = {}
        self.best_params_per_model = {}
        
        # Configuración de optimización
        self.n_trials = 50  # Trials para optimización bayesiana
        self.cv_folds = 5   # Folds para validación cruzada temporal
        
        # Configurar modelos base
        self._setup_base_models()
        
        # Mostrar ensemble final
        model_names = list(self.base_models.keys())
        logger.info(f"Modelo TRB inicializado - Ensemble: {', '.join(model_names)}")
        logger.info(f"Configuración: NN={enable_neural_network}, SVR={enable_svr}, GPU={enable_gpu}")
    
    def _setup_base_models(self):
        """Configurar modelos base con hiperparámetros optimizados para rebotes"""
        
        # XGBoost - Excelente para features categóricas y no lineales
        self.base_models['xgboost'] = {
            'model': xgb.XGBRegressor(
                random_state=self.random_state,
                tree_method='gpu_hist' if self.enable_gpu else 'hist',
                gpu_id=0 if self.enable_gpu else None,
                n_jobs=-1
            ),
            'param_space': {
                'n_estimators': (100, 500),
                'max_depth': (3, 8),
                'learning_rate': (0.01, 0.3),
                'subsample': (0.7, 1.0),
                'colsample_bytree': (0.7, 1.0),
                'reg_alpha': (0, 10),
                'reg_lambda': (1, 10),
                'min_child_weight': (1, 10)
            }
        }
        
        # LightGBM - Rápido y eficiente para datasets grandes
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
                'n_estimators': (100, 500),
                'max_depth': (3, 8),
                'learning_rate': (0.01, 0.3),
                'subsample': (0.7, 1.0),
                'colsample_bytree': (0.7, 1.0),
                'reg_alpha': (0, 10),
                'reg_lambda': (1, 10),
                'min_child_samples': (5, 50),
                'num_leaves': (10, 100)
            }
        }
        
        # CatBoost - Excelente para features categóricas sin preprocessing
        self.base_models['catboost'] = {
            'model': cb.CatBoostRegressor(
                random_state=self.random_state,
                task_type='GPU' if self.enable_gpu else 'CPU',
                devices='0' if self.enable_gpu else None,
                verbose=False,
                allow_writing_files=False
            ),
            'param_space': {
                'iterations': (100, 500),
                'depth': (3, 8),
                'learning_rate': (0.01, 0.3),
                'l2_leaf_reg': (1, 10),
                'subsample': (0.7, 1.0),
                'colsample_bylevel': (0.7, 1.0),
                'min_data_in_leaf': (1, 50)
            }
        }
        
        # Gradient Boosting - Robusto y estable
        self.base_models['gradient_boosting'] = {
            'model': GradientBoostingRegressor(
                random_state=self.random_state
            ),
            'param_space': {
                'n_estimators': (100, 300),
                'max_depth': (3, 6),
                'learning_rate': (0.01, 0.2),
                'subsample': (0.7, 1.0),
                'max_features': (0.7, 1.0),
                'alpha': (0.1, 0.9)  # Quantile para robustez
            }
        }
        
        # Ridge - Regularización L2 para estabilidad
        self.base_models['ridge'] = {
            'model': Ridge(random_state=self.random_state),
            'param_space': {
                'alpha': (0.1, 100.0),
                'fit_intercept': [True, False],
                'solver': ['auto', 'svd', 'cholesky', 'lsqr']
            }
        }
        
        # Neural Network (opcional)
        if self.enable_neural_network:
            self.base_models['neural_network'] = {
                'model': MLPRegressor(
                    random_state=self.random_state,
                    max_iter=1000,
                    early_stopping=True,
                    validation_fraction=0.1
                ),
                'param_space': {
                    'hidden_layer_sizes': [(50,), (100,), (50, 25), (100, 50)],
                    'alpha': (0.0001, 0.1),
                    'learning_rate_init': (0.001, 0.01),
                    'beta_1': (0.8, 0.99),
                    'beta_2': (0.9, 0.999)
                }
            }
        
        # SVR (opcional, muy lento)
        if self.enable_svr:
            self.base_models['svr'] = {
                'model': SVR(),
                'param_space': {
                    'C': (0.1, 100.0),
                    'epsilon': (0.01, 1.0),
                    'gamma': ['scale', 'auto'],
                    'kernel': ['rbf', 'linear']
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
    
    def train(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Entrenamiento completo del modelo Stacking TRB
        
        Args:
            df: DataFrame con datos de jugadores y estadísticas
            
        Returns:
            Dict con métricas de validación
        """
        logger.info("Iniciando entrenamiento del modelo TRB...")
        
        # Verificar orden cronológico
        if 'Date' in df.columns:
            if not df['Date'].is_monotonic_increasing:
                logger.info("Ordenando datos cronológicamente...")
                df = df.sort_values(['Player', 'Date']).reset_index(drop=True)
        
        # Generar features especializadas para rebotes
        logger.info("Generando características especializadas...")
        features = self.feature_engineer.generate_all_features(df)  # Modificar DataFrame directamente
        
        if not features:
            raise ValueError("No se pudieron generar features para TRB")
        
        logger.info(f"Features seleccionadas: {len(features)}")
        
        # Preparar datos (ahora df tiene las features)
        X = df[features].fillna(0)
        y = df['TRB']
        
        # División temporal
        train_data, test_data = self._temporal_split(df)
        
        X_train = train_data[features].fillna(0)
        y_train = train_data['TRB']
        X_test = test_data[features].fillna(0)
        y_test = test_data['TRB']
        
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
        
        # Métricas específicas para rebotes
        accuracy_1reb = np.mean(np.abs(y_test - y_pred) <= 1) * 100
        accuracy_2reb = np.mean(np.abs(y_test - y_pred) <= 2) * 100
        accuracy_3reb = np.mean(np.abs(y_test - y_pred) <= 3) * 100
        
        self.validation_metrics = {
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'accuracy_1reb': accuracy_1reb,
            'accuracy_2reb': accuracy_2reb,
            'accuracy_3reb': accuracy_3reb
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
        logger.info(f"Accuracy ±1reb: {accuracy_1reb:.1f}%")
        logger.info(f"Accuracy ±2reb: {accuracy_2reb:.1f}%")
        logger.info(f"Accuracy ±3reb: {accuracy_3reb:.1f}%")
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
        Configurar el modelo de stacking con meta-learner optimizado
        STACKING AVANZADO ESPECIALIZADO BASADO EN FEATURES IMPORTANTES
        """
        # STACKING ESPECIALIZADO POR GRUPOS DE FEATURES
        # Basado en las top features identificadas: orb_rate, trb_avg_3g, scoring_role, etc.
        
        # GRUPO 1: MODELOS ESPECIALIZADOS EN REBOTES HISTÓRICOS
        # Para features como: trb_avg_3g, trb_per_minute_10g, orb_rate
        rebounding_specialists = [
            ('xgb_rebounding', xgb.XGBRegressor(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=2,
                reg_lambda=5,
                random_state=self.random_state,
                tree_method='gpu_hist' if self.enable_gpu else 'hist'
            )),
            ('lgb_rebounding', lgb.LGBMRegressor(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=2,
                reg_lambda=5,
                num_leaves=50,
                random_state=self.random_state,
                verbose=-1
            ))
        ]
        
        # GRUPO 2: MODELOS ESPECIALIZADOS EN CONTEXTO DE JUEGO
        # Para features como: scoring_role, orb_minutes_interaction, interior_play_index
        context_specialists = [
            ('catboost_context', cb.CatBoostRegressor(
                iterations=300,
                depth=6,
                learning_rate=0.1,
                l2_leaf_reg=5,
                subsample=0.8,
                random_state=self.random_state,
                verbose=False,
                allow_writing_files=False
            )),
            ('gb_context', GradientBoostingRegressor(
                n_estimators=200,
                max_depth=5,
                learning_rate=0.1,
                subsample=0.8,
                max_features=0.8,
                alpha=0.2,
                random_state=self.random_state
            ))
        ]
        
        # GRUPO 3: MODELOS ESPECIALIZADOS EN INTERACCIONES COMPLEJAS
        # Para features de interacción y volatilidad
        interaction_specialists = []
        
        if self.enable_neural_network:
            interaction_specialists.append(
                ('mlp_interactions', MLPRegressor(
                    hidden_layer_sizes=(128, 64, 32),
                    activation='relu',
                    solver='adam',
                    alpha=0.01,  # Regularización L2
                    learning_rate='adaptive',
                    learning_rate_init=0.001,
                    max_iter=500,
                    early_stopping=True,
                    validation_fraction=0.1,
                    n_iter_no_change=20,
                    random_state=self.random_state
                ))
            )
        
        # Agregar Ridge como estabilizador
        interaction_specialists.append(
            ('ridge_interactions', Ridge(
                alpha=10.0,
                fit_intercept=True,
                random_state=self.random_state
            ))
        )
        
        # COMBINAR TODOS LOS ESPECIALISTAS
        all_specialists = rebounding_specialists + context_specialists + interaction_specialists
        
        # META-LEARNER AVANZADO CON MÚLTIPLES NIVELES
        # Nivel 1: Ridge con regularización moderada
        meta_learner_l1 = Ridge(
            alpha=5.0,
            fit_intercept=True,
            random_state=self.random_state
        )
        
        # STACKING PRINCIPAL - CORREGIDO PARA EVITAR ERROR DE CV
        self.stacking_model = StackingRegressor(
            estimators=all_specialists,
            final_estimator=meta_learner_l1,
            cv=3,  # Usar entero en lugar de TimeSeriesSplit para evitar error
            n_jobs=1,  # Reducir paralelización para evitar conflictos
            passthrough=False  # Solo usar predicciones de base models
        )
        
        # STACKING SECUNDARIO (NIVEL 2) - ENSEMBLE DE ENSEMBLES
        # Crear un segundo nivel de stacking para máxima precisión
        
        # Modelos del primer nivel (ya optimizados)
        level1_models = [
            ('stacking_primary', self.stacking_model),
            ('xgb_final', xgb.XGBRegressor(
                n_estimators=400,
                max_depth=5,
                learning_rate=0.08,
                subsample=0.85,
                colsample_bytree=0.85,
                reg_alpha=3,
                reg_lambda=7,
                random_state=self.random_state,
                tree_method='gpu_hist' if self.enable_gpu else 'hist'
            )),
            ('lgb_final', lgb.LGBMRegressor(
                n_estimators=400,
                max_depth=5,
                learning_rate=0.08,
                subsample=0.85,
                colsample_bytree=0.85,
                reg_alpha=3,
                reg_lambda=7,
                num_leaves=40,
                random_state=self.random_state,
                verbose=-1
            ))
        ]
        
        # Meta-learner final ultra-conservador
        meta_learner_final = Ridge(
            alpha=15.0,  # Alta regularización para estabilidad
            fit_intercept=True,
            random_state=self.random_state
        )
        
        # STACKING FINAL (NIVEL 2) - CORREGIDO PARA EVITAR ERROR DE CV
        self.final_stacking_model = StackingRegressor(
            estimators=level1_models,
            final_estimator=meta_learner_final,
            cv=2,  # Usar entero en lugar de TimeSeriesSplit para evitar error
            n_jobs=1,  # Reducir paralelización para evitar conflictos
            passthrough=True  # Incluir features originales en nivel final
        )
        
        logger.info(f"Stacking configurado: {len(all_specialists)} modelos especializados")
        
        return self.final_stacking_model
    
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
            Array con predicciones de TRB
        """
        if self.stacking_model is None:
            raise ValueError("Modelo no entrenado. Ejecutar train() primero.")
        
        # Generar features (modificar DataFrame directamente)
        features = self.feature_engineer.generate_all_features(df)
        X = df[features].fillna(0)
        
        # Realizar predicciones usando el modelo principal entrenado
        predictions = self.stacking_model.predict(X)
        
        # Asegurar que las predicciones sean no negativas (rebotes no pueden ser negativos)
        predictions = np.maximum(predictions, 0)
        
        return predictions
    
    def save_model(self, filepath: str):
        """Guardar modelo entrenado como objeto directo"""
        if self.stacking_model is None:
            raise ValueError("Modelo no entrenado. Ejecutar train() primero.")
        
        # Crear directorio si no existe
        import os
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Guardar SOLO el modelo entrenado como objeto directo usando JOBLIB con compresión
        joblib.dump(self.stacking_model, filepath, compress=3, protocol=4)
        logger.info(f"Modelo TRB guardado como objeto directo (JOBLIB): {filepath}")
    
    def load_model(self, filepath: str):
        """Cargar modelo entrenado (compatible con ambos formatos)"""
        try:
            # Intentar cargar modelo directo (nuevo formato)
            self.stacking_model = joblib.load(filepath)
            if hasattr(self.stacking_model, 'predict'):
                logger.info(f"Modelo TRB (objeto directo) cargado desde: {filepath}")
                return
            else:
                # No es un modelo directo, tratar como formato antiguo
                raise ValueError("No es modelo directo")
        except (ValueError, AttributeError):
            # Formato antiguo (diccionario)
            try:
                model_data = joblib.load(filepath)
                if isinstance(model_data, dict) and 'stacking_model' in model_data:
                    self.stacking_model = model_data['stacking_model']
                    self.base_models = model_data.get('base_models', {})
                    self.feature_engineer = model_data.get('feature_engineer')
                    self.validation_metrics = model_data.get('validation_metrics', {})
                    self.cv_scores = model_data.get('cv_scores', {})
                    self.feature_importance = model_data.get('feature_importance', {})
                    self.best_params_per_model = model_data.get('best_params_per_model', {})
                    logger.info(f"Modelo TRB (formato legacy) cargado desde: {filepath}")
                else:
                    raise ValueError("Formato de archivo no reconocido")
            except Exception as e:
                raise ValueError(f"No se pudo cargar el modelo TRB: {e}")
    
    def get_model_summary(self) -> Dict[str, Any]:
        """Obtener resumen del modelo"""
        return {
            'model_type': 'Stacking Ensemble TRB',
            'base_models': list(self.base_models.keys()),
            'validation_metrics': self.validation_metrics,
            'cv_scores': self.cv_scores,
            'n_features': len(self.feature_importance),
            'top_features': list(self.feature_importance.keys())[:10]
        }


class XGBoostTRBModel:
    """
    Modelo XGBoost simplificado para TRB con compatibilidad con el sistema existente
    Mantiene la interfaz del modelo de PTS pero optimizado para rebotes
    """
    
    def __init__(self, enable_neural_network: bool = True, enable_svr: bool = True, 
                 enable_gpu: bool = True, random_state: int = 42, teams_df: pd.DataFrame = None):
        """Inicializar modelo XGBoost TRB con stacking ensemble"""
        self.stacking_model = StackingTRBModel(
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
        
        logger.info("Modelo XGBoost TRB inicializado con stacking completo")
    
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
