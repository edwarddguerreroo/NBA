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
from src.models.players.triples.features_triples import ThreePointsFeatureEngineer

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


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
                 n_trials: int = 50,
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
        
        # Configurar modelos base especializados para triples
        self._setup_base_models()
        
        logger.info("Modelo Stacking 3PT inicializado para predicción de triples")
    
    def _setup_base_models(self):
        """
        Configuración de modelos base optimizados específicamente para triples
        """
        logger.info("Configurando modelos base especializados para triples...")
        
        # 1. XGBoost optimizado para triples (parámetros ajustados para conteo discreto)
        self.base_models['xgboost'] = {
            'model_class': xgb.XGBRegressor,
            'param_space': {
                'n_estimators': (150, 800),
                'learning_rate': (0.03, 0.2),
                'max_depth': (3, 10),
                'min_child_weight': (1, 6),
                'subsample': (0.7, 0.9),
                'colsample_bytree': (0.7, 0.9),
                'reg_alpha': (0.1, 3.0),
                'reg_lambda': (0.1, 5.0),
                'gamma': (0.0, 2.0)
            },
            'fixed_params': {
                'objective': 'reg:squarederror',
                'random_state': self.random_state,
                'n_jobs': -1,
                'verbosity': 0
            },
            'weight': 0.3  # Mayor peso por ser efectivo en conteo
        }
        
        # 2. LightGBM para capturar patrones específicos de tiro
        self.base_models['lightgbm'] = {
            'model_class': lgb.LGBMRegressor,
            'param_space': {
                'n_estimators': (150, 600),
                'learning_rate': (0.03, 0.15),
                'num_leaves': (20, 100),
                'min_child_samples': (10, 50),
                'subsample': (0.7, 0.9),
                'colsample_bytree': (0.7, 0.9),
                'reg_alpha': (0.1, 2.0),
                'reg_lambda': (0.1, 3.0)
            },
            'fixed_params': {
                'objective': 'regression',
                'metric': 'rmse',
                'random_state': self.random_state,
                'n_jobs': -1,
                'verbosity': -1
            },
            'weight': 0.25
        }
        
        # 3. Random Forest para robustez
        self.base_models['random_forest'] = {
            'model_class': RandomForestRegressor,
            'param_space': {
                'n_estimators': (100, 300),
                'max_depth': (5, 15),
                'min_samples_split': (2, 10),
                'min_samples_leaf': (1, 5),
                'max_features': (0.6, 0.9)
            },
            'fixed_params': {
                'random_state': self.random_state,
                'n_jobs': -1
            },
            'weight': 0.2
        }
        
        # 4. Extra Trees para diversidad
        self.base_models['extra_trees'] = {
            'model_class': ExtraTreesRegressor,
            'param_space': {
                'n_estimators': (100, 250),
                'max_depth': (5, 12),
                'min_samples_split': (2, 8),
                'min_samples_leaf': (1, 4)
            },
            'fixed_params': {
                'random_state': self.random_state,
                'n_jobs': -1
            },
            'weight': 0.15
        }
        
        # 5. Ridge para regularización lineal
        self.base_models['ridge'] = {
            'model_class': Ridge,
            'param_space': {
                'alpha': (0.1, 10.0)
            },
            'fixed_params': {
                'random_state': self.random_state
            },
            'weight': 0.1
        }
        
        logger.info(f"Configurados {len(self.base_models)} modelos base para triples")
    
    def _optimize_single_model(self, model_name: str, X_train, y_train, X_val, y_val) -> Dict[str, Any]:
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
                # Entrenar modelo
                model = model_config['model_class'](**params)
                
                # Entrenar con early stopping si es compatible
                if model_name in ['xgboost', 'lightgbm']:
                    model.fit(
                        X_train, y_train,
                        eval_set=[(X_val, y_val)],
                        early_stopping_rounds=self.early_stopping_rounds,
                        verbose=False
                    )
                else:
                    model.fit(X_train, y_train)
                
                # Predecir y evaluar
                y_pred = model.predict(X_val)
                
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
            study.optimize(objective, n_trials=self.n_trials, timeout=300)  # 5 min timeout
        except Exception as e:
            logger.warning(f"Error en optimización de {model_name}: {str(e)}")
            return None
        
        # Obtener mejores parámetros
        best_params = study.best_params.copy()
        best_params.update(model_config['fixed_params'])
        
        # Entrenar modelo final
        best_model = model_config['model_class'](**best_params)
        if model_name in ['xgboost', 'lightgbm']:
            best_model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                early_stopping_rounds=self.early_stopping_rounds,
                verbose=False
            )
        else:
            best_model.fit(X_train, y_train)
        
        return {
            'model': best_model,
            'best_params': best_params,
            'best_score': study.best_value,
            'n_trials': len(study.trials)
        }
    
    def _train_base_models(self, X_train, y_train, X_val, y_val) -> Dict[str, Any]:
        """
        Entrena todos los modelos base
        """
        logger.info("Entrenando modelos base para triples...")
        
        results = {}
        
        for model_name in self.base_models.keys():
            logger.info(f"Entrenando modelo: {model_name}")
            
            try:
                result = self._optimize_single_model(model_name, X_train, y_train, X_val, y_val)
                
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
        
        # Meta-learner optimizado para conteo de triples
        meta_learner = Ridge(alpha=1.0, random_state=self.random_state)
        
        # Crear modelo de stacking
        self.stacking_model = StackingRegressor(
            estimators=estimators,
            final_estimator=meta_learner,
            cv=3,  # Menor CV para speed
            n_jobs=-1,
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
        X_sorted = X.reindex(df_sorted.index)
        y_sorted = y.reindex(df_sorted.index)
        
        # Configurar splits temporales
        tscv = TimeSeriesSplit(n_splits=self.cv_folds)
        
        cv_scores = []
        fold_metrics = []
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X_sorted)):
            logger.info(f"Fold {fold + 1}/{self.cv_folds}")
            
            X_train_fold = X_sorted.iloc[train_idx]
            X_val_fold = X_sorted.iloc[val_idx]
            y_train_fold = y_sorted.iloc[train_idx]
            y_val_fold = y_sorted.iloc[val_idx]
            
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
                'samples': len(y_val_fold)
            })
            
            logger.info(f"Fold {fold + 1} - MAE: {mae:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}")
        
        # Calcular métricas promedio
        avg_metrics = {
            'cv_mae_mean': np.mean(cv_scores),
            'cv_mae_std': np.std(cv_scores),
            'cv_rmse_mean': np.mean([m['rmse'] for m in fold_metrics]),
            'cv_r2_mean': np.mean([m['r2'] for m in fold_metrics])
        }
        
        self.cv_scores = avg_metrics
        
        logger.info(f"CV MAE: {avg_metrics['cv_mae_mean']:.4f} ± {avg_metrics['cv_mae_std']:.4f}")
        
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
        
        if missing_features:
            logger.warning(f"Features faltantes: {len(missing_features)}")
        
        logger.info(f"Features disponibles para triples: {len(available_features)}")
        
        # Preparar DataFrame final
        X = df_features[available_features].fillna(0)
        
        return X, available_features
    
    def _temporal_split(self, df: pd.DataFrame, test_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split temporal de los datos - ORDENACIÓN CRONOLÓGICA GARANTIZADA
        """
        # Ordenar cronológicamente: primero por jugador, luego por fecha
        df_sorted = df.sort_values(['Player', 'Date']).reset_index(drop=True)
        
        # Verificar ordenación cronológica
        logger.info(f"Rango de fechas: {df_sorted['Date'].min()} a {df_sorted['Date'].max()}")
        
        # Calcular punto de corte basado en fechas, no en número de registros
        # Usar percentil 80 de fechas para asegurar división temporal real
        date_cutoff = df_sorted['Date'].quantile(1 - test_size)
        
        # Split basado en fecha de corte
        train_df = df_sorted[df_sorted['Date'] < date_cutoff].copy()
        test_df = df_sorted[df_sorted['Date'] >= date_cutoff].copy()
        
        logger.info(f"Split temporal cronológico:")
        logger.info(f"  Train: {len(train_df)} registros (hasta {train_df['Date'].max()})")
        logger.info(f"  Test: {len(test_df)} registros (desde {test_df['Date'].min()})")
        logger.info(f"  Fecha de corte: {date_cutoff}")
        
        return train_df, test_df
    
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
        X_val, _ = self._prepare_features(val_df)
        
        # Targets
        y_train = train_df['3P'].values
        y_val = val_df['3P'].values
        
        # Guardar features seleccionadas
        self.selected_features = feature_names
        self.feature_names = feature_names
        
        logger.info(f"Datos preparados - Train: {X_train.shape}, Val: {X_val.shape}")
        
        # Entrenar modelos base
        base_results = self._train_base_models(X_train, y_train, X_val, y_val)
        
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
        
        logger.info("Entrenamiento completado para triples")
        logger.info(f"Métricas finales - MAE: {final_metrics['mae']:.4f}, R²: {final_metrics['r2']:.4f}")
        
        return final_metrics
    
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Realiza predicciones de triples
        """
        if not self.is_trained:
            raise ValueError("Modelo no entrenado. Llama a train() primero.")
        
        # Preparar features
        X, _ = self._prepare_features(df)
        
        # Predecir
        predictions = self.stacking_model.predict(X)
        
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
        """Guarda el modelo entrenado"""
        if not self.is_trained:
            raise ValueError("No hay modelo entrenado para guardar")
        
        model_data = {
            'stacking_model': self.stacking_model,
            'trained_base_models': self.trained_base_models,
            'feature_names': self.feature_names,
            'selected_features': self.selected_features,
            'training_metrics': self.training_metrics,
            'best_params_per_model': self.best_params_per_model,
            'scaler': self.scaler,
            'is_trained': self.is_trained
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"Modelo de triples guardado en: {filepath}")
    
    def load_model(self, filepath: str):
        """Carga un modelo previamente entrenado"""
        model_data = joblib.load(filepath)
        
        self.stacking_model = model_data['stacking_model']
        self.trained_base_models = model_data['trained_base_models']
        self.feature_names = model_data['feature_names']
        self.selected_features = model_data['selected_features']
        self.training_metrics = model_data['training_metrics']
        self.best_params_per_model = model_data['best_params_per_model']
        self.scaler = model_data['scaler']
        self.is_trained = model_data['is_trained']
        
        logger.info(f"Modelo de triples cargado desde: {filepath}")


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
