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
import pickle

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
                 models_path: str = "trained_models"):
        """
        Inicializar ensemble final
        
        Args:
            config: Configuración del ensemble
            models_path: Ruta a los modelos entrenados (nueva ubicación: trained_models/)
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
        Cargar todos los modelos individuales con sus FeatureEngineers
        
        Returns:
            Dict con modelos cargados exitosamente
        """
        logger.info("Cargando modelos individuales con FeatureEngineers...")
        
        # Usar el nuevo registry para cargar modelos con feature engineers
        self.loaded_models = self.model_registry.load_all_models()
        
        loaded_count = len(self.loaded_models)
        total_configured = len(self.model_registry.model_configs)
        
        logger.info(f"Modelos cargados: {loaded_count}/{total_configured}")
        
        # Log detallado de modelos cargados
        for model_name, model_data in self.loaded_models.items():
            config = model_data['config']
            logger.info(f"✓ {model_name}: {config['type']} → {config['target']}")
        
        return self.loaded_models
    
    def prepare_ensemble_data(self, df_players: pd.DataFrame, 
                            df_teams: pd.DataFrame) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.Series]]:
        """
        Preparar datos para ensemble usando predicciones de modelos individuales
        
        Args:
            df_players: DataFrame con datos de jugadores
            df_teams: DataFrame con datos de equipos
            
        Returns:
            Tuple con (features_dict, targets_dict) por tipo de modelo
        """
        logger.info("Preparando datos para ensemble usando predicciones de modelos individuales...")
        
        features_dict = {'regression': pd.DataFrame(), 'classification': pd.DataFrame()}
        targets_dict = {'regression': pd.Series(), 'classification': pd.Series()}
        
        # Generar predicciones de modelos base usando el nuevo registry
        base_predictions_categorized = self.model_registry.get_predictions(df_players, df_teams)
        
        if not base_predictions_categorized:
            logger.error("No se pudieron generar predicciones de modelos base")
            return features_dict, targets_dict
        
        # Aplanar la estructura categorizada para compatibilidad
        base_predictions = {}
        if isinstance(base_predictions_categorized, dict):
            for category, models in base_predictions_categorized.items():
                if isinstance(models, dict):
                    # Es estructura categorizada
                    base_predictions.update(models)
                else:
                    # Es estructura plana
                    base_predictions = base_predictions_categorized
                    break
        else:
            base_predictions = base_predictions_categorized
        
        # CORRECCIÓN CRÍTICA: Alinear predicciones de diferentes granularidades
        base_predictions = self._alinear_predicciones_ensemble(base_predictions)
        
        # Separar modelos por tipo usando el nuevo ModelRegistry
        regression_models = self.model_registry.get_models_by_type('regression')
        classification_models = self.model_registry.get_models_by_type('classification')
        
        # Preparar features de regresión
        if regression_models and base_predictions:
            reg_features = []
            reg_targets = None
            feature_names = []
            reference_target = None
            
            for model_name in regression_models.keys():
                if model_name in base_predictions:
                    pred_data = base_predictions[model_name]
                    
                    # Verificar que las predicciones tienen el tamaño correcto
                    if pred_data['n_samples'] > 0:
                        reg_features.append(pred_data['predictions'])
                        feature_names.append(f"pred_{model_name}")
                        
                        # Usar targets del primer modelo como referencia
                        if reg_targets is None and pred_data['targets'] is not None:
                            reg_targets = pred_data['targets']
                            reference_target = pred_data['target_name']
                            logger.info(f"Usando {model_name} como referencia para targets de regresión ({reference_target})")
            
            if reg_features:
                # Verificar que todas las features tienen el mismo tamaño
                feature_sizes = [len(f) for f in reg_features]
                if len(set(feature_sizes)) == 1:  # Todos del mismo tamaño
                    features_dict['regression'] = pd.DataFrame(
                        np.column_stack(reg_features),
                        columns=feature_names
                    )
                    if reg_targets is not None:
                        targets_dict['regression'] = pd.Series(reg_targets)
                    logger.info(f"Features de regresión preparadas: {len(feature_names)} modelos, {len(reg_features[0])} muestras")
                    logger.info(f"Modelos de regresión: {feature_names}")
                else:
                    logger.error(f"Tamaños de features de regresión incompatibles: {feature_sizes}")
        
        # Preparar features de clasificación
        if classification_models and base_predictions:
            clf_features = []
            clf_targets = None
            feature_names = []
            reference_target = None
            
            for model_name in classification_models.keys():
                if model_name in base_predictions:
                    pred_data = base_predictions[model_name]
                    
                    # Verificar que las predicciones tienen el tamaño correcto
                    if pred_data['n_samples'] > 0:
                        clf_features.append(pred_data['predictions'])
                        feature_names.append(f"pred_{model_name}")
                        
                        # Usar targets del primer modelo como referencia
                        if clf_targets is None and pred_data['targets'] is not None:
                            clf_targets = pred_data['targets']
                            reference_target = pred_data['target_name']
                            logger.info(f"Usando {model_name} como referencia para targets de clasificación ({reference_target})")
            
            if clf_features:
                # Verificar que todas las features tienen el mismo tamaño
                feature_sizes = [len(f) for f in clf_features]
                if len(set(feature_sizes)) == 1:  # Todos del mismo tamaño
                    features_dict['classification'] = pd.DataFrame(
                        np.column_stack(clf_features),
                        columns=feature_names
                    )
                    if clf_targets is not None:
                        targets_dict['classification'] = pd.Series(clf_targets)
                    logger.info(f"Features de clasificación preparadas: {len(feature_names)} modelos, {len(clf_features[0])} muestras")
                    logger.info(f"Modelos de clasificación: {feature_names}")
                else:
                    logger.error(f"Tamaños de features de clasificación incompatibles: {feature_sizes}")
        
        logger.info(f"RESUMEN: Datos preparados - Regresión: {len(features_dict['regression'])} features, "
                   f"Clasificación: {len(features_dict['classification'])} features")
        
        # Log de modelos disponibles
        logger.info(f"Predicciones generadas por: {list(base_predictions.keys())}")
        logger.info(f"Modelos de regresión disponibles: {list(regression_models.keys())}")
        logger.info(f"Modelos de clasificación disponibles: {list(classification_models.keys())}")
        
        return features_dict, targets_dict
    
    def _alinear_predicciones_ensemble(self, base_predictions):
        """
        Alinear predicciones de jugadores y equipos para el ensemble
        Convierte predicciones de nivel jugador (55,901) a nivel equipo (5,226)
        
        PROBLEMA: Los modelos de jugadores generan 55,901 predicciones (nivel jugador-juego)
                  Los modelos de equipos generan 5,226 predicciones (nivel equipo-juego)
                  
        SOLUCIÓN: Agregar las predicciones de jugadores al nivel de equipo
        """
        predictions_alineadas = {}
        
        logger.info("🔧 Alineando predicciones de diferentes granularidades...")
        
        for model_name, pred_data in base_predictions.items():
            if not isinstance(pred_data, dict) or 'predictions' not in pred_data:
                predictions_alineadas[model_name] = pred_data
                continue
                
            predictions = pred_data['predictions']
            n_samples = len(predictions)
            
            logger.info(f"  {model_name}: {n_samples} predicciones")
            
            # Si es de nivel jugador (>10,000), agregar a nivel equipo (5,226)
            if n_samples > 10000:  # Asumimos que es nivel jugador
                logger.info(f"    Convirtiendo de nivel jugador a nivel equipo...")
                
                # Calcular jugadores por equipo-juego
                target_size = 5226  # Tamaño objetivo (equipos)
                team_size = n_samples // target_size  # ~10-11 jugadores por equipo-juego
                
                team_predictions = []
                
                for i in range(0, len(predictions), team_size):
                    group = predictions[i:i+team_size]
                    if len(group) > 0:
                        # Agregar según el tipo de estadística
                        if model_name in ['pts', 'trb', 'ast']:
                            # Estadísticas aditivas: sumar las predicciones de jugadores
                            team_pred = np.sum(group)
                        elif model_name in ['3pt']:
                            # Estadísticas de triples: sumar también (total de triples del equipo)
                            team_pred = np.sum(group)
                        elif model_name == 'double_double':
                            # Double-doubles: proporción o cantidad total
                            team_pred = np.sum(group)  # Total de double-doubles del equipo
                        else:
                            # Por defecto: promedio
                            team_pred = np.mean(group)
                        
                        team_predictions.append(team_pred)
                
                # Ajustar al tamaño exacto
                if len(team_predictions) > target_size:
                    team_predictions = team_predictions[:target_size]
                elif len(team_predictions) < target_size:
                    # Extender con el promedio si faltan
                    avg_pred = np.mean(team_predictions) if team_predictions else 0
                    team_predictions.extend([avg_pred] * (target_size - len(team_predictions)))
                
                # Actualizar predicción alineada
                pred_data_aligned = pred_data.copy()
                pred_data_aligned['predictions'] = np.array(team_predictions)
                pred_data_aligned['n_samples'] = len(team_predictions)
                
                # Recalcular estadísticas
                pred_data_aligned['prediction_stats'] = {
                    'min': float(np.min(team_predictions)),
                    'max': float(np.max(team_predictions)),
                    'mean': float(np.mean(team_predictions)),
                    'std': float(np.std(team_predictions))
                }
                
                predictions_alineadas[model_name] = pred_data_aligned
                logger.info(f"    ✅ {n_samples} → {len(team_predictions)} predicciones")
                
            else:
                # Ya es nivel equipo, mantener tal como está
                predictions_alineadas[model_name] = pred_data
                logger.info(f"    ✅ Mantenido (ya es nivel equipo)")
        
        # Verificar alineación final
        tamaños = [pred_data.get('n_samples', 0) for pred_data in predictions_alineadas.values() if isinstance(pred_data, dict)]
        tamaños_unicos = set(tamaños)
        
        if len(tamaños_unicos) == 1:
            logger.info(f"✅ Todas las predicciones alineadas: {list(tamaños_unicos)[0]} muestras")
        else:
            logger.warning(f"⚠️  Tamaños aún desalineados: {tamaños_unicos}")
        
        return predictions_alineadas
    
    def _generate_base_predictions_legacy(self, df_players: pd.DataFrame, 
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
        joblib.dump(model_to_save, filepath, compress=3, protocol=4)
        logger.info(f"Ensemble guardado como objeto directo (JOBLIB): {filepath}")
    
    @classmethod
    def load_ensemble(cls, filepath: str, models_path: str = "trained_models"):
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