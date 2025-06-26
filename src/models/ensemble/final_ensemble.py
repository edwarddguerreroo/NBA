"""
Ensemble Final NBA
==================

Sistema de ensemble avanzado que combina todos los modelos individuales
en una predicción final refinada usando técnicas de meta-learning.

Características principales:
- Stacking separado para regresión y clasificación
- Optimización bayesiana con Optuna
- División cronológica estricta
- Validación cruzada temporal
- Manejo automático de modelos faltantes
- Métricas especializadas por tipo de problema
"""

import os
import sys
import warnings
import joblib
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Union
from datetime import datetime, timedelta
import json

# Core libraries
import numpy as np
import pandas as pd

# ML libraries
from sklearn.ensemble import StackingRegressor, StackingClassifier, VotingRegressor, VotingClassifier
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)
from sklearn.linear_model import Ridge, LogisticRegression
import xgboost as xgb
import lightgbm as lgb

# Optuna for Bayesian optimization
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

# Local imports
from .ensemble_config import EnsembleConfig, ModelConfig
from .model_registry import ModelRegistry
from config.logging_config import NBALogger

# Configuración
warnings.filterwarnings('ignore')
logger = NBALogger.get_logger(__name__)


class FinalEnsembleModel:
    """
    Modelo de ensemble final que combina todos los modelos individuales
    usando técnicas avanzadas de meta-learning y optimización bayesiana
    """
    
    def __init__(self, config: Optional[EnsembleConfig] = None,
                 models_path: str = "results"):
        """
        Inicializar ensemble final
        
        Args:
            config: Configuración del ensemble
            models_path: Ruta a los modelos entrenados
        """
        self.config = config or EnsembleConfig()
        self.models_path = models_path
        
        # Componentes principales
        self.model_registry = ModelRegistry(models_path)
        self.loaded_models = {}
        
        # Ensemble models
        self.regression_ensemble = None
        self.classification_ensemble = None
        
        # Meta-learners optimizados
        self.optimized_meta_learners = {}
        
        # Scalers
        self.regression_scaler = StandardScaler()
        self.classification_scaler = StandardScaler()
        
        # Estado del entrenamiento
        self.is_trained = False
        self.training_history = {}
        self.optimization_results = {}
        
        # Métricas
        self.validation_scores = {}
        self.feature_importance = {}
        
        logger.info("FinalEnsembleModel inicializado")
    
    def load_individual_models(self) -> Dict[str, Any]:
        """
        Cargar todos los modelos individuales disponibles
        
        Returns:
            Dict con modelos cargados exitosamente
        """
        logger.info("Cargando modelos individuales...")
        
        # Descubrir modelos disponibles
        available_models = self.model_registry.discover_available_models()
        
        # Cargar modelos habilitados
        enabled_models = self.config.get_enabled_models()
        loaded_count = 0
        
        for model_name, model_config in enabled_models.items():
            if model_name in available_models and available_models[model_name]['status'] == 'available':
                model = self.model_registry.load_model(model_name)
                if model:
                    self.loaded_models[model_name] = {
                        'model': model,
                        'config': model_config,
                        'type': model_config.type
                    }
                    loaded_count += 1
                    logger.info(f"✓ {model_name} cargado ({model_config.type})")
                else:
                    logger.warning(f"✗ Error cargando {model_name}")
            else:
                logger.warning(f"✗ {model_name} no disponible")
        
        logger.info(f"Modelos cargados: {loaded_count}/{len(enabled_models)}")
        
        return self.loaded_models
    
    def prepare_ensemble_data(self, df_players: pd.DataFrame, 
                            df_teams: pd.DataFrame) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.Series]]:
        """
        Preparar datos para ensemble separando por tipo de modelo
        
        Args:
            df_players: DataFrame con datos de jugadores
            df_teams: DataFrame con datos de equipos
            
        Returns:
            Tuple con (features_dict, targets_dict) por tipo de modelo
        """
        logger.info("Preparando datos para ensemble...")
        
        features_dict = {'regression': pd.DataFrame(), 'classification': pd.DataFrame()}
        targets_dict = {'regression': pd.Series(), 'classification': pd.Series()}
        
        # Generar predicciones de modelos base
        base_predictions = self._generate_base_predictions(df_players, df_teams)
        
        # Organizar por tipo de modelo
        regression_models = self.config.get_models_by_type('regression')
        classification_models = self.config.get_models_by_type('classification')
        
        # Preparar features de regresión
        if regression_models and base_predictions:
            reg_features = []
            reg_targets = None
            feature_names = []
            
            for model_name, model_config in regression_models.items():
                if model_name in base_predictions:
                    pred_data = base_predictions[model_name]
                    reg_features.append(pred_data['predictions'])
                    feature_names.append(model_name)
                    
                    if reg_targets is None:
                        reg_targets = pred_data['targets']
            
            if reg_features:
                features_dict['regression'] = pd.DataFrame(
                    np.column_stack(reg_features),
                    columns=feature_names
                )
                targets_dict['regression'] = pd.Series(reg_targets)
        
        # Preparar features de clasificación
        if classification_models and base_predictions:
            clf_features = []
            clf_targets = None
            feature_names = []
            
            for model_name, model_config in classification_models.items():
                if model_name in base_predictions:
                    pred_data = base_predictions[model_name]
                    
                    # Para clasificación, usar probabilidades si están disponibles
                    if 'probabilities' in pred_data and pred_data['probabilities'] is not None:
                        clf_features.append(pred_data['probabilities'][:, 1])  # Clase positiva
                    else:
                        clf_features.append(pred_data['predictions'])
                    
                    feature_names.append(model_name)
                    
                    if clf_targets is None:
                        clf_targets = pred_data['targets']
            
            if clf_features:
                features_dict['classification'] = pd.DataFrame(
                    np.column_stack(clf_features),
                    columns=feature_names
                )
                targets_dict['classification'] = pd.Series(clf_targets)
        
        logger.info(f"Datos preparados - Regresión: {len(features_dict['regression'])}, "
                   f"Clasificación: {len(features_dict['classification'])}")
        
        return features_dict, targets_dict
    
    def _generate_base_predictions(self, df_players: pd.DataFrame, 
                                 df_teams: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        """
        Generar predicciones base de todos los modelos individuales
        
        Args:
            df_players: DataFrame con datos de jugadores
            df_teams: DataFrame con datos de equipos
            
        Returns:
            Dict con predicciones de cada modelo
        """
        base_predictions = {}
        
        for model_name, model_info in self.loaded_models.items():
            try:
                model = model_info['model']
                model_config = model_info['config']
                
                # Seleccionar datos apropiados
                if 'player' in model_name:
                    data = df_players
                else:
                    data = df_teams
                
                # Verificar que tenemos datos
                if data.empty:
                    logger.warning(f"Sin datos para {model_name}")
                    continue
                
                # Verificar que existe la columna target
                target_col = model_config.target_column
                if target_col not in data.columns:
                    logger.warning(f"Columna target '{target_col}' no encontrada para {model_name}")
                    continue
                
                # Generar predicciones - manejo robusto para diferentes tipos de modelos
                predictions = None
                try:
                    # Verificar que el modelo tiene método predict
                    if not hasattr(model, 'predict'):
                        logger.error(f"Modelo {model_name} no tiene método predict")
                        continue
                    
                    # Si es un ModelWithFeatures wrapper, usar su método predict integrado
                    if hasattr(model, 'model_name') and hasattr(model, 'feature_engineer'):
                        # Es un wrapper ModelWithFeatures - maneja automáticamente las features
                        logger.info(f"Usando ModelWithFeatures wrapper para {model_name}")
                        predictions = model.predict(data)
                        
                    # Si el modelo es de un tipo específico conocido, usar su método predict personalizado
                    elif hasattr(model, '__class__') and 'StackingPTSModel' in str(model.__class__):
                        # Es nuestro modelo stacking personalizado
                        predictions = model.predict(data)
                        
                    elif hasattr(model, '__class__') and any(ml_lib in str(model.__class__) for ml_lib in ['xgboost', 'lightgbm', 'catboost', 'sklearn']):
                        # Es un modelo ML estándar - usar predict directo
                        # Para estos modelos, necesitamos solo las features, no todo el DataFrame
                        if hasattr(model, 'feature_names_in_'):
                            # El modelo tiene información de features específicas
                            model_features = model.feature_names_in_
                            common_features = [f for f in model_features if f in data.columns]
                            if common_features:
                                predictions = model.predict(data[common_features])
                            else:
                                logger.warning(f"No hay features comunes para {model_name} con feature_names_in_")
                                continue
                        else:
                            # Intentar predict directo - puede fallar si hay incompatibilidad de features
                            predictions = model.predict(data)
                    else:
                        # Modelo genérico - intentar predict directo primero
                        predictions = model.predict(data)
                        
                except Exception as pred_error:
                    logger.warning(f"Error en predict directo para {model_name}: {pred_error}")
                    # Estrategias de fallback para diferentes tipos de errores
                    try:
                        # Estrategia 1: Si es error de features, intentar con subset
                        if hasattr(model, 'feature_names_in_'):
                            model_features = model.feature_names_in_
                            common_features = [f for f in model_features if f in data.columns]
                            if common_features:
                                predictions = model.predict(data[common_features])
                                logger.info(f"Predicción exitosa para {model_name} con {len(common_features)} features comunes")
                            else:
                                logger.error(f"No hay features comunes para {model_name}")
                                continue
                        else:
                            # Estrategia 2: Intentar con datos numéricos solamente
                            numeric_data = data.select_dtypes(include=[np.number])
                            if not numeric_data.empty:
                                predictions = model.predict(numeric_data)
                                logger.info(f"Predicción exitosa para {model_name} con datos numéricos")
                            else:
                                logger.error(f"No hay datos numéricos para {model_name}")
                                continue
                                
                    except Exception as e2:
                        logger.error(f"Error definitivo en predict para {model_name}: {e2}")
                        continue
                
                # Verificar que las predicciones son válidas
                if predictions is None:
                    logger.error(f"Predicciones None para {model_name}")
                    continue
                
                if len(predictions) == 0:
                    logger.error(f"Predicciones vacías para {model_name}")
                    continue
                
                if np.isnan(predictions).all():
                    logger.error(f"Predicciones todas NaN para {model_name}")
                    continue
                
                # Para modelos de clasificación, también obtener probabilidades
                probabilities = None
                if model_config.type == 'classification' and hasattr(model, 'predict_proba'):
                    try:
                        # Si es un wrapper, usar el modelo interno
                        if hasattr(model, 'model') and hasattr(model.model, 'predict_proba'):
                            if hasattr(model.model, 'feature_names_in_'):
                                model_features = model.model.feature_names_in_
                                common_features = [f for f in model_features if f in data.columns]
                                if common_features:
                                    probabilities = model.model.predict_proba(data[common_features])
                                else:
                                    probabilities = None
                            else:
                                probabilities = model.model.predict_proba(data)
                        else:
                            # Modelo directo
                            if hasattr(model, 'feature_names_in_'):
                                model_features = model.feature_names_in_
                                common_features = [f for f in model_features if f in data.columns]
                                if common_features:
                                    probabilities = model.predict_proba(data[common_features])
                                else:
                                    probabilities = None
                            else:
                                probabilities = model.predict_proba(data)
                    except Exception as prob_error:
                        logger.warning(f"No se pudieron obtener probabilidades para {model_name}: {prob_error}")
                        probabilities = None
                
                base_predictions[model_name] = {
                    'predictions': predictions,
                    'probabilities': probabilities,
                    'targets': data[target_col].values,
                    'model_type': model_config.type
                }
                
                logger.info(f"✓ Predicciones generadas para {model_name}: {len(predictions)} muestras")
                
            except Exception as e:
                logger.error(f"Error general generando predicciones para {model_name}: {e}")
                continue
        
        logger.info(f"Predicciones base generadas para {len(base_predictions)}/{len(self.loaded_models)} modelos")
        return base_predictions
    
    def optimize_meta_learners(self, features_dict: Dict[str, pd.DataFrame],
                             targets_dict: Dict[str, pd.Series]) -> Dict[str, Any]:
        """
        Optimizar meta-learners usando Optuna
        
        Args:
            features_dict: Features por tipo de modelo
            targets_dict: Targets por tipo de modelo
            
        Returns:
            Resultados de optimización
        """
        logger.info("Iniciando optimización bayesiana de meta-learners...")
        
        optimization_results = {}
        
        # Optimizar meta-learners de regresión
        if not features_dict['regression'].empty:
            logger.info("Optimizando meta-learners de regresión...")
            reg_results = self._optimize_regression_meta_learners(
                features_dict['regression'], targets_dict['regression']
            )
            optimization_results['regression'] = reg_results
        
        # Optimizar meta-learners de clasificación
        if not features_dict['classification'].empty:
            logger.info("Optimizando meta-learners de clasificación...")
            clf_results = self._optimize_classification_meta_learners(
                features_dict['classification'], targets_dict['classification']
            )
            optimization_results['classification'] = clf_results
        
        self.optimization_results = optimization_results
        logger.info("Optimización bayesiana completada")
        
        return optimization_results
    
    def _optimize_regression_meta_learners(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Optimizar meta-learners para regresión"""
        
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                'max_depth': trial.suggest_int('max_depth', 3, 8),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
                'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0),
                'random_state': 42
            }
            
            meta_learner = xgb.XGBRegressor(**params)
            tscv = self.config.create_time_series_split()
            scores = cross_val_score(meta_learner, X, y, cv=tscv, scoring='neg_mean_absolute_error')
            
            return -scores.mean()
        
        # Crear y ejecutar estudio
        study = self.config.create_optuna_study('regression_meta_learner', 'mae')
        study.optimize(objective, n_trials=self.config.optuna_config['n_trials'])
        
        # Entrenar meta-learner final
        best_params = study.best_params
        best_meta_learner = xgb.XGBRegressor(**best_params)
        best_meta_learner.fit(X, y)
        
        self.optimized_meta_learners['regression'] = best_meta_learner
        
        return {
            'best_params': best_params,
            'best_score': study.best_value,
            'n_trials': len(study.trials),
            'meta_learner': best_meta_learner
        }
    
    def _optimize_classification_meta_learners(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Optimizar meta-learners para clasificación"""
        
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                'max_depth': trial.suggest_int('max_depth', 3, 8),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
                'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0),
                'random_state': 42,
                'objective': 'binary:logistic'
            }
            
            meta_learner = xgb.XGBClassifier(**params)
            tscv = self.config.create_time_series_split()
            scores = cross_val_score(meta_learner, X, y, cv=tscv, scoring='f1')
            
            return scores.mean()
        
        # Crear y ejecutar estudio
        study = self.config.create_optuna_study('classification_meta_learner', 'f1')
        study.optimize(objective, n_trials=self.config.optuna_config['n_trials'])
        
        # Entrenar meta-learner final
        best_params = study.best_params
        best_meta_learner = xgb.XGBClassifier(**best_params)
        best_meta_learner.fit(X, y)
        
        self.optimized_meta_learners['classification'] = best_meta_learner
        
        return {
            'best_params': best_params,
            'best_score': study.best_value,
            'n_trials': len(study.trials),
            'meta_learner': best_meta_learner
        }
    
    def build_final_ensembles(self, features_dict: Dict[str, pd.DataFrame],
                            targets_dict: Dict[str, pd.Series]) -> Dict[str, Any]:
        """
        Construir ensembles finales usando meta-learners optimizados
        
        Args:
            features_dict: Features por tipo
            targets_dict: Targets por tipo
            
        Returns:
            Ensembles construidos
        """
        logger.info("Construyendo ensembles finales...")
        
        ensemble_results = {}
        
        # Construir ensemble de regresión
        if not features_dict['regression'].empty:
            X_reg = features_dict['regression']
            y_reg = targets_dict['regression']
            
            # Meta-learner optimizado o por defecto
            meta_learner = self.optimized_meta_learners.get('regression', 
                                                          xgb.XGBRegressor(random_state=42))
            
            # Escalar features
            X_reg_scaled = self.regression_scaler.fit_transform(X_reg)
            
            # Entrenar meta-learner directamente (enfoque simplificado)
            meta_learner.fit(X_reg_scaled, y_reg)
            self.regression_ensemble = meta_learner
            
            # Evaluar
            reg_score = meta_learner.score(X_reg_scaled, y_reg)
            ensemble_results['regression'] = {
                'ensemble': self.regression_ensemble,
                'score': reg_score,
                'n_models': len(X_reg.columns)
            }
            
            logger.info(f"✓ Ensemble de regresión: R² = {reg_score:.4f}")
        
        # Construir ensemble de clasificación
        if not features_dict['classification'].empty:
            X_clf = features_dict['classification']
            y_clf = targets_dict['classification']
            
            # Meta-learner optimizado o por defecto
            meta_learner = self.optimized_meta_learners.get('classification',
                                                          LogisticRegression(random_state=42))
            
            # Escalar features
            X_clf_scaled = self.classification_scaler.fit_transform(X_clf)
            
            # Entrenar meta-learner directamente
            meta_learner.fit(X_clf_scaled, y_clf)
            self.classification_ensemble = meta_learner
            
            # Evaluar
            clf_score = meta_learner.score(X_clf_scaled, y_clf)
            ensemble_results['classification'] = {
                'ensemble': self.classification_ensemble,
                'score': clf_score,
                'n_models': len(X_clf.columns)
            }
            
            logger.info(f"✓ Ensemble de clasificación: Accuracy = {clf_score:.4f}")
        
        return ensemble_results
    
    def train(self, df_players: pd.DataFrame, df_teams: pd.DataFrame) -> Dict[str, Any]:
        """
        Entrenar ensemble completo
        
        Args:
            df_players: DataFrame con datos de jugadores
            df_teams: DataFrame con datos de equipos
            
        Returns:
            Resultados del entrenamiento
        """
        logger.info("=== INICIANDO ENTRENAMIENTO DE ENSEMBLE FINAL ===")
        
        start_time = datetime.now()
        
        # 1. Cargar modelos individuales
        self.load_individual_models()
        
        if not self.loaded_models:
            raise ValueError("No se cargaron modelos individuales")
        
        # 2. Preparar datos
        features_dict, targets_dict = self.prepare_ensemble_data(df_players, df_teams)
        
        # 3. Optimizar meta-learners
        optimization_results = self.optimize_meta_learners(features_dict, targets_dict)
        
        # 4. Construir ensembles finales
        ensemble_results = self.build_final_ensembles(features_dict, targets_dict)
        
        # 5. Evaluar rendimiento
        validation_results = self._evaluate_ensembles(features_dict, targets_dict)
        
        # Guardar resultados
        end_time = datetime.now()
        training_duration = (end_time - start_time).total_seconds()
        
        self.training_history = {
            'start_time': start_time,
            'end_time': end_time,
            'duration_seconds': training_duration,
            'models_loaded': len(self.loaded_models),
            'optimization_results': optimization_results,
            'ensemble_results': ensemble_results,
            'validation_results': validation_results
        }
        
        self.is_trained = True
        
        logger.info(f"=== ENTRENAMIENTO COMPLETADO ({training_duration:.1f}s) ===")
        
        return self.training_history
    
    def _evaluate_ensembles(self, features_dict: Dict[str, pd.DataFrame],
                          targets_dict: Dict[str, pd.Series]) -> Dict[str, Any]:
        """Evaluar rendimiento de ensembles"""
        results = {}
        
        # Evaluar regresión
        if self.regression_ensemble and not features_dict['regression'].empty:
            X = self.regression_scaler.transform(features_dict['regression'])
            y = targets_dict['regression']
            
            y_pred = self.regression_ensemble.predict(X)
            
            results['regression'] = {
                'mae': mean_absolute_error(y, y_pred),
                'rmse': np.sqrt(mean_squared_error(y, y_pred)),
                'r2': r2_score(y, y_pred)
            }
        
        # Evaluar clasificación
        if self.classification_ensemble and not features_dict['classification'].empty:
            X = self.classification_scaler.transform(features_dict['classification'])
            y = targets_dict['classification']
            
            y_pred = self.classification_ensemble.predict(X)
            y_proba = self.classification_ensemble.predict_proba(X)[:, 1]
            
            results['classification'] = {
                'accuracy': accuracy_score(y, y_pred),
                'precision': precision_score(y, y_pred, average='weighted'),
                'recall': recall_score(y, y_pred, average='weighted'),
                'f1': f1_score(y, y_pred, average='weighted'),
                'auc_roc': roc_auc_score(y, y_proba)
            }
        
        return results
    
    def predict(self, df_players: pd.DataFrame, df_teams: pd.DataFrame,
                model_type: str = 'both') -> Dict[str, np.ndarray]:
        """
        Realizar predicciones con ensemble entrenado
        
        Args:
            df_players: DataFrame con datos de jugadores
            df_teams: DataFrame con datos de equipos
            model_type: 'regression', 'classification' o 'both'
            
        Returns:
            Dict con predicciones por tipo
        """
        if not self.is_trained:
            raise ValueError("El ensemble debe ser entrenado primero")
        
        predictions = {}
        
        # Preparar features para ensemble
        features_dict, _ = self.prepare_ensemble_data(df_players, df_teams)
        
        # Predicciones de regresión
        if (model_type in ['regression', 'both'] and 
            self.regression_ensemble and 
            not features_dict['regression'].empty):
            
            X_reg = self.regression_scaler.transform(features_dict['regression'])
            pred_reg = self.regression_ensemble.predict(X_reg)
            predictions['regression'] = pred_reg
        
        # Predicciones de clasificación
        if (model_type in ['classification', 'both'] and 
            self.classification_ensemble and 
            not features_dict['classification'].empty):
            
            X_clf = self.classification_scaler.transform(features_dict['classification'])
            pred_clf = self.classification_ensemble.predict(X_clf)
            proba_clf = self.classification_ensemble.predict_proba(X_clf)
            
            predictions['classification'] = pred_clf
            predictions['classification_proba'] = proba_clf
        
        return predictions
    
    def save_ensemble(self, filepath: str):
        """Guardar ensemble entrenado como objeto directo usando JOBLIB"""
        if not self.is_trained:
            raise ValueError("El ensemble debe ser entrenado antes de guardarlo")
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        # Determinar cuál ensemble guardar (priorizar regression)
        if self.regression_ensemble is not None:
            model_to_save = self.regression_ensemble
            logger.info("Guardando ensemble de regresión como modelo principal")
        elif self.classification_ensemble is not None:
            model_to_save = self.classification_ensemble
            logger.info("Guardando ensemble de clasificación como modelo principal")
        else:
            raise ValueError("No hay ensemble entrenado para guardar")
        
        # Guardar SOLO el modelo ensemble principal usando JOBLIB con compresión
        joblib.dump(model_to_save, filepath, compress=3)
        logger.info(f"Ensemble guardado como objeto directo (JOBLIB): {filepath}")
    
    @classmethod
    def load_ensemble(cls, filepath: str, models_path: str = "results"):
        """Cargar ensemble desde archivo con compatibilidad para ambos formatos (JOBLIB prioritario)"""
        try:
            # Intentar cargar con JOBLIB primero (formato estándar)
            try:
                ensemble_data = joblib.load(filepath)
                logger.info("Archivo cargado con JOBLIB exitosamente")
            except Exception as joblib_error:
                # Fallback a pickle para archivos legacy
                logger.warning(f"Error con JOBLIB, intentando pickle: {joblib_error}")
                with open(filepath, 'rb') as f:
                    ensemble_data = pickle.load(f)
                logger.info("Archivo cargado con pickle como fallback")
            
            # Verificar formato del archivo cargado
            if isinstance(ensemble_data, dict) and 'config' in ensemble_data:
                # Formato diccionario completo (legacy)
                config = ensemble_data.get('config')
                ensemble = cls(config=config, models_path=models_path)
                
                # Restaurar estado completo
                ensemble.regression_ensemble = ensemble_data.get('regression_ensemble')
                ensemble.classification_ensemble = ensemble_data.get('classification_ensemble')
                ensemble.regression_scaler = ensemble_data.get('regression_scaler', StandardScaler())
                ensemble.classification_scaler = ensemble_data.get('classification_scaler', StandardScaler())
                ensemble.optimized_meta_learners = ensemble_data.get('optimized_meta_learners', {})
                ensemble.training_history = ensemble_data.get('training_history', {})
                ensemble.optimization_results = ensemble_data.get('optimization_results', {})
                ensemble.validation_scores = ensemble_data.get('validation_scores', {})
                ensemble.feature_importance = ensemble_data.get('feature_importance', {})
                ensemble.is_trained = ensemble_data.get('is_trained', True)
                
                logger.info(f"Ensemble (formato legacy diccionario) cargado desde: {filepath}")
                
            elif hasattr(ensemble_data, 'predict'):
                # Formato objeto directo (nuevo estándar)
                ensemble = cls(models_path=models_path)
                
                # Asignar como ensemble de regresión por defecto
                ensemble.regression_ensemble = ensemble_data
                ensemble.regression_scaler = StandardScaler()  # Inicializar scaler por defecto
                ensemble.is_trained = True
                logger.info(f"Ensemble (objeto directo JOBLIB) cargado desde: {filepath}")
                
            else:
                raise ValueError("Formato de archivo no reconocido")
            
            return ensemble
            
        except Exception as e:
            logger.error(f"Error cargando ensemble desde {filepath}: {e}")
            raise ValueError(f"No se pudo cargar el ensemble: {e}")
    
    def get_training_summary(self) -> Dict[str, Any]:
        """Obtener resumen del entrenamiento"""
        if not self.is_trained:
            return {"status": "not_trained"}
        
        return {
            "status": "trained",
            "training_duration": self.training_history.get('duration_seconds', 0),
            "models_loaded": self.training_history.get('models_loaded', 0),
            "has_regression_ensemble": self.regression_ensemble is not None,
            "has_classification_ensemble": self.classification_ensemble is not None,
            "optimization_results": self.optimization_results,
            "validation_results": self.training_history.get('validation_results', {})
        }


# Classes auxiliares para crear estimators dummy
class DummyRegressor:
    """Regressor dummy que retorna predicciones pre-calculadas"""
    
    def __init__(self, predictions):
        self.predictions = predictions
    
    def fit(self, X, y):
        return self
    
    def predict(self, X):
        return self.predictions[:len(X)]


class DummyClassifier:
    """Classifier dummy que retorna predicciones pre-calculadas"""
    
    def __init__(self, predictions):
        self.predictions = predictions
        self.classes_ = np.unique(predictions)
    
    def fit(self, X, y):
        return self
    
    def predict(self, X):
        return self.predictions[:len(X)]
    
    def predict_proba(self, X):
        pred = self.predictions[:len(X)]
        proba = np.zeros((len(pred), len(self.classes_)))
        for i, p in enumerate(pred):
            proba[i, int(p)] = 1.0
        return proba 