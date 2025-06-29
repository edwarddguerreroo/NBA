"""
Trainer Completo para Ensemble NBA Final
=======================================

Trainer que integra carga de datos, entrenamiento del ensemble final
y generación completa de métricas y visualizaciones para el sistema completo NBA.

Características:
- Integración completa con data loader
- Entrenamiento automatizado del ensemble final
- Generación de dashboard PNG unificado con todas las métricas
- Métricas detalladas para regresión y clasificación
- Análisis de feature importance del ensemble
- Validación cruzada temporal
- Manejo robusto de features para todos los modelos
"""

import json
import logging
import os
import warnings
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    classification_report
)

# ML Libraries
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import StackingRegressor, StackingClassifier
from sklearn.linear_model import Ridge, LogisticRegression
import xgboost as xgb
import lightgbm as lgb

# Imports del proyecto
from src.preprocessing.data_loader import NBADataLoader

# Import del sistema de logging unificado
from config.logging_config import configure_trainer_logging, NBALogger

warnings.filterwarnings('ignore')
logger = configure_trainer_logging('ensemble')

# Configurar estilo de visualizaciones optimizado para PNG
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10


class EnsembleNBATrainer:
    """
    Trainer completo para ensemble NBA final.
    
    Integra carga de datos, entrenamiento, evaluación y visualizaciones del ensemble completo.
    """
    
    def __init__(self,
                 game_data_path: str,
                 biometrics_path: str,
                 teams_path: str,
                 models_path: str = "results",
                 output_dir: str = "results/ensemble_model",
                 cv_folds: int = 5,
                 random_state: int = 42):
        """
        Inicializa el trainer completo para el ensemble NBA.
        
        Args:
            game_data_path: Ruta a datos de partidos
            biometrics_path: Ruta a datos biométricos
            teams_path: Ruta a datos de equipos
            models_path: Ruta base donde están los modelos individuales
            output_dir: Directorio de salida para resultados
            cv_folds: Folds para validación cruzada
            random_state: Semilla para reproducibilidad
        """
        self.game_data_path = game_data_path
        self.biometrics_path = biometrics_path
        self.teams_path = teams_path
        self.models_path = models_path
        self.output_dir = os.path.normpath(output_dir)
        self.random_state = random_state
        self.cv_folds = cv_folds
        
        # Crear directorio de salida con manejo robusto
        try:
            os.makedirs(self.output_dir, exist_ok=True)
            logger.info(f"Directorio de salida creado/verificado: {self.output_dir}")
        except Exception as e:
            logger.error(f"Error creando directorio {self.output_dir}: {e}")
            # Crear directorio alternativo en caso de error
            self.output_dir = os.path.normpath("results_ensemble_model")
            os.makedirs(self.output_dir, exist_ok=True)
            logger.info(f"Usando directorio alternativo: {self.output_dir}")
        
        # Componentes principales
        self.data_loader = NBADataLoader(
            game_data_path, biometrics_path, teams_path
        )
        
        # Datos y resultados
        self.df_players = None
        self.df_teams = None
        self.training_results = None
        self.predictions = None
        self.loaded_models = None
        self.ensemble_model = None
        
        logger.info(f"Trainer Ensemble NBA inicializado | Output: {self.output_dir}")
    
    def load_and_prepare_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Carga y prepara todos los datos necesarios.
        
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: Datos de jugadores y equipos preparados
        """
        logger.info("Cargando datos NBA para ensemble")
        
        # Cargar datos usando el data loader
        self.df_players, self.df_teams = self.data_loader.load_data()
        
        # Estadísticas básicas de los datos
        logger.info(f"Datos cargados:")
        logger.info(f"  - Jugadores: {len(self.df_players)} registros")
        logger.info(f"  - Equipos: {len(self.df_teams)} registros")
        logger.info(f"  - Jugadores únicos: {self.df_players['Player'].nunique()}")
        logger.info(f"  - Equipos únicos: {self.df_players['Team'].nunique()}")
        
        return self.df_players, self.df_teams
    
    def _setup_final_ensemble(self):
        """Configura el FinalEnsemble con modelos individuales"""
        logger.info("Configurando FinalEnsemble con modelos individuales...")
        
        from src.models.ensemble.final_ensemble import FinalEnsembleModel
        from src.models.ensemble.ensemble_config import EnsembleConfig
        
        # Crear configuración para ensemble
        config = EnsembleConfig()
        
        # Inicializar FinalEnsemble
        self.final_ensemble = FinalEnsembleModel(
            config=config,
            models_path=self.models_path
        )
        
        # Cargar modelos individuales
        loaded_models = self.final_ensemble.load_individual_models()
        
        logger.info(f"FinalEnsemble configurado con {len(loaded_models)} modelos individuales")
        return loaded_models
    
    def create_ensemble_features_with_final_ensemble(self) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.Series]]:
        """
        Usar FinalEnsemble para generar features basadas en predicciones de modelos individuales.
        
        Returns:
            Tuple: Features dict y targets dict por tipo de modelo
        """
        logger.info("Generando features del ensemble usando FinalEnsemble...")
        
        # Usar el FinalEnsemble para preparar datos
        features_dict, targets_dict = self.final_ensemble.prepare_ensemble_data(
            self.df_players, self.df_teams
        )
        
        logger.info(f"Features generadas por FinalEnsemble:")
        logger.info(f"  - Regresión: {len(features_dict['regression'])} features")
        logger.info(f"  - Clasificación: {len(features_dict['classification'])} features")
        
        return features_dict, targets_dict

    def train_ensemble(self, X: pd.DataFrame, targets: Dict[str, pd.Series]) -> Dict[str, Any]:
        """
        Entrena el ensemble final.
        
        Args:
            X: Features del ensemble
            targets: Diccionario de targets
            
        Returns:
            Dict: Resultados del entrenamiento
        """
        logger.info("Entrenando ensemble final")
        
        if X.empty or not targets:
            raise ValueError("Features o targets vacíos")
        
        start_time = datetime.now()
        ensemble_models = {}
        ensemble_results = {}
        
        # Entrenar modelos para cada target
        for target_name, target_series in targets.items():
            try:
                logger.info(f"Entrenando ensemble para {target_name}")
                
                # Determinar tipo de problema
                is_classification = 'double_double' in target_name
                
                # Alinear datos
                valid_idx = target_series.notna()
                X_target = X.loc[valid_idx]
                y_target = target_series.loc[valid_idx]
                
                if len(X_target) < 100:
                    logger.warning(f"{target_name}: Pocos datos válidos ({len(X_target)})")
                    continue
                
                # Entrenar modelo específico
                if is_classification:
                    model_result = self._train_classification_ensemble(X_target, y_target, target_name)
                else:
                    model_result = self._train_regression_ensemble(X_target, y_target, target_name)
                
                if model_result:
                    ensemble_models[target_name] = model_result['model']
                    ensemble_results[target_name] = model_result['metrics']
                    
                    logger.info(f"✅ {target_name}: Ensemble entrenado")
                
            except Exception as e:
                logger.error(f"Error entrenando ensemble para {target_name}: {str(e)}")
        
        training_duration = (datetime.now() - start_time).total_seconds()
        
        # Compilar resultados
        results = {
            'ensemble_models': ensemble_models,
            'individual_results': ensemble_results,
            'training_duration_seconds': training_duration,
            'n_features': X.shape[1],
            'feature_names': list(X.columns),
            'targets_trained': list(ensemble_models.keys())
        }
        
        self.ensemble_model = ensemble_models
        self.training_results = results
        
        logger.info(f"Ensemble completado en {training_duration:.1f}s")
        
        return results

    def _train_regression_ensemble(self, X: pd.DataFrame, y: pd.Series, 
                                 target_name: str) -> Optional[Dict]:
        """Entrena ensemble para regresión con métricas realistas"""
        try:
            from sklearn.model_selection import train_test_split
            
            # ARREGLAR OVERFITTING: Dividir en train/test
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=self.random_state
            )
            
            base_models = [
                ('ridge', Ridge(alpha=1.0, random_state=self.random_state)),
                ('xgb', xgb.XGBRegressor(n_estimators=100, max_depth=4, 
                                       learning_rate=0.1, random_state=self.random_state)),
                ('lgb', lgb.LGBMRegressor(n_estimators=100, max_depth=4,
                                       learning_rate=0.1, random_state=self.random_state,
                                       verbose=-1))
            ]
            
            meta_model = Ridge(alpha=0.1, random_state=self.random_state)
            
            stacking_regressor = StackingRegressor(
                estimators=base_models,
                final_estimator=meta_model,
                cv=min(5, len(X_train) // 100),
                n_jobs=1
            )
            
            # Entrenar solo con datos de entrenamiento
            stacking_regressor.fit(X_train, y_train)
            
            # Evaluar con datos de prueba (métricas realistas)
            y_pred = stacking_regressor.predict(X_test)
            
            metrics = {
                'mae': float(mean_absolute_error(y_test, y_pred)),
                'rmse': float(np.sqrt(mean_squared_error(y_test, y_pred))),
                'r2': float(r2_score(y_test, y_pred)),
                'n_samples_train': int(len(y_train)),
                'n_samples_test': int(len(y_test))
            }
            
            # Re-entrenar con todos los datos para el modelo final
            stacking_regressor.fit(X, y)
            
            return {
                'model': stacking_regressor,
                'metrics': metrics,
                'predictions': y_pred
            }
            
        except Exception as e:
            logger.error(f"Error en regresión ensemble para {target_name}: {str(e)}")
            return None

    def _train_classification_ensemble(self, X: pd.DataFrame, y: pd.Series, 
                                     target_name: str) -> Optional[Dict]:
        """Entrena ensemble para clasificación con métricas realistas"""
        try:
            from sklearn.model_selection import train_test_split
            
            # ARREGLAR OVERFITTING: Dividir en train/test
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=self.random_state, stratify=y
            )
            
            base_models = [
                ('ridge', LogisticRegression(random_state=self.random_state, max_iter=1000)),
                ('xgb', xgb.XGBClassifier(n_estimators=100, max_depth=4,
                                        learning_rate=0.1, random_state=self.random_state)),
                ('lgb', lgb.LGBMClassifier(n_estimators=100, max_depth=4,
                                        learning_rate=0.1, random_state=self.random_state,
                                        verbose=-1))
            ]
            
            meta_model = LogisticRegression(random_state=self.random_state, max_iter=1000)
            
            stacking_classifier = StackingClassifier(
                estimators=base_models,
                final_estimator=meta_model,
                cv=min(5, len(X_train) // 100),
                n_jobs=1
            )
            
            # Entrenar solo con datos de entrenamiento
            stacking_classifier.fit(X_train, y_train)
            
            # Evaluar con datos de prueba (métricas realistas)
            y_pred = stacking_classifier.predict(X_test)
            
            metrics = {
                'accuracy': float(accuracy_score(y_test, y_pred)),
                'precision': float(precision_score(y_test, y_pred, average='weighted', zero_division=0)),
                'recall': float(recall_score(y_test, y_pred, average='weighted', zero_division=0)),
                'f1': float(f1_score(y_test, y_pred, average='weighted', zero_division=0)),
                'n_samples_train': int(len(y_train)),
                'n_samples_test': int(len(y_test))
            }
            
            # Re-entrenar con todos los datos para el modelo final
            stacking_classifier.fit(X, y)
            
            return {
                'model': stacking_classifier,
                'metrics': metrics,
                'predictions': y_pred
            }
            
        except Exception as e:
            logger.error(f"Error en clasificación ensemble para {target_name}: {str(e)}")
            return None

    def generate_all_visualizations(self):
        """
        Genera una visualización completa en PNG con todas las métricas del ensemble.
        """
        logger.info("Generando visualizaciones del ensemble")
        
        if not self.training_results:
            logger.warning("No hay resultados para visualizar")
            return
        
        # Crear figura principal
        fig = plt.figure(figsize=(20, 14))
        fig.suptitle('Dashboard Completo - Ensemble NBA Final', fontsize=20, fontweight='bold', y=0.98)
        
        # Crear grid de subplots
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
        
        # 1. Resumen de métricas del ensemble
        ax1 = fig.add_subplot(gs[0, 0])
        self._plot_ensemble_metrics_summary(ax1)
        
        # 2. Comparación de modelos individuales
        ax2 = fig.add_subplot(gs[0, 1:3])
        self._plot_models_comparison(ax2)
        
        # 3. Distribución de targets
        ax3 = fig.add_subplot(gs[0, 3])
        self._plot_targets_distribution(ax3)
        
        # 4. Feature importance del ensemble
        ax4 = fig.add_subplot(gs[1, 0:2])
        self._plot_ensemble_feature_importance(ax4)
        
        # 5. Análisis de regresión
        ax5 = fig.add_subplot(gs[1, 2])
        self._plot_regression_analysis(ax5)
        
        # 6. Análisis de clasificación
        ax6 = fig.add_subplot(gs[1, 3])
        self._plot_classification_analysis(ax6)
        
        # 7. Análisis temporal del ensemble
        ax7 = fig.add_subplot(gs[2, 0:2])
        self._plot_temporal_ensemble_analysis(ax7)
        
        # 8. Análisis de estabilidad de predicciones
        ax8 = fig.add_subplot(gs[2, 2:4])
        self._plot_prediction_stability(ax8)
        
        # Guardar dashboard completo
        dashboard_path = os.path.join(self.output_dir, "ensemble_model_dashboard_complete.png")
        plt.savefig(dashboard_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        
        logger.info(f"Dashboard guardado: {dashboard_path}")

    def _plot_ensemble_metrics_summary(self, ax):
        """Gráfico resumen de métricas del ensemble"""
        if not self.training_results or not self.training_results.get('individual_results'):
            ax.text(0.5, 0.5, 'Métricas del\nensemble\nno disponibles', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Métricas del Ensemble')
            ax.axis('off')
            return
        
        results = self.training_results['individual_results']
        
        # Crear tabla de métricas
        metric_data = []
        for target, metrics in results.items():
            if 'mae' in metrics:  # Regresión
                metric_data.append([target.replace('_target', ''), 'MAE', f"{metrics['mae']:.3f}"])
                metric_data.append([target.replace('_target', ''), 'R²', f"{metrics.get('r2', 0):.3f}"])
            elif 'accuracy' in metrics:  # Clasificación
                metric_data.append([target.replace('_target', ''), 'Acc', f"{metrics['accuracy']:.3f}"])
                metric_data.append([target.replace('_target', ''), 'F1', f"{metrics.get('f1', 0):.3f}"])
        
        if metric_data:
            # Convertir a DataFrame para mejor manejo
            df_metrics = pd.DataFrame(metric_data, columns=['Target', 'Métrica', 'Valor'])
            
            # Mostrar como tabla
            table = ax.table(cellText=df_metrics.values,
                            colLabels=df_metrics.columns,
                            cellLoc='center',
                            loc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1, 1.5)
            
            # Estilo
            for i in range(len(df_metrics) + 1):
                for j in range(3):
                    cell = table[(i, j)]
                    if i == 0:
                        cell.set_facecolor('#40466e')
                        cell.set_text_props(weight='bold', color='white')
                    else:
                        cell.set_facecolor('#f1f1f2')
        
        ax.axis('off')
        ax.set_title('Métricas del Ensemble', fontweight='bold', pad=20)

    def _plot_models_comparison(self, ax):
        """Comparación de modelos individuales cargados"""
        if not self.loaded_models:
            ax.text(0.5, 0.5, 'Modelos\nno cargados', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Comparación de Modelos')
            return
        
        model_names = list(self.loaded_models.keys())
        model_sizes = [info['size_mb'] for info in self.loaded_models.values()]
        model_features = [info['n_features'] for info in self.loaded_models.values()]
        
        x = np.arange(len(model_names))
        width = 0.35
        
        # Normalizar tamaños para visualización
        max_size = max(model_sizes) if model_sizes else 1
        normalized_sizes = [size / max_size * 100 for size in model_sizes]
        
        bars1 = ax.bar(x - width/2, normalized_sizes, width, label='Tamaño (% del máx)', alpha=0.7)
        bars2 = ax.bar(x + width/2, model_features, width, label='Número de Features', alpha=0.7)
        
        ax.set_xlabel('Modelos')
        ax.set_ylabel('Valores')
        ax.set_title('Comparación de Modelos Cargados', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=45)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

    def _plot_targets_distribution(self, ax):
        """Distribución de targets disponibles"""
        if not hasattr(self, 'df_players') or self.df_players is None:
            ax.text(0.5, 0.5, 'Datos\nno disponibles', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Distribución de Targets')
            return
        
        targets = ['PTS', 'TRB', 'AST', '3P']
        available_targets = [t for t in targets if t in self.df_players.columns]
        
        if available_targets:
            # Calcular medias
            means = [self.df_players[t].mean() for t in available_targets]
            
            bars = ax.bar(available_targets, means, color=['lightcoral', 'lightgreen', 'lightblue', 'gold'][:len(available_targets)])
            
            # Agregar valores en las barras
            for bar, mean in zip(bars, means):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                       f'{mean:.1f}', ha='center', va='bottom')
            
            ax.set_ylabel('Promedio')
            ax.set_title('Promedio de Stats Principales', fontweight='bold')
            ax.grid(axis='y', alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'Targets\nno encontrados', ha='center', va='center', transform=ax.transAxes)

    def _plot_ensemble_feature_importance(self, ax):
        """Feature importance del ensemble (predicciones de modelos base)"""
        ax.text(0.5, 0.5, 'Feature Importance\ndel Ensemble\n(Basado en predicciones\nde modelos base)', 
               ha='center', va='center', transform=ax.transAxes, fontsize=12)
        ax.set_title('Feature Importance del Ensemble', fontweight='bold')
        ax.axis('off')

    def _plot_regression_analysis(self, ax):
        """Análisis de resultados de regresión"""
        if not self.training_results or not self.training_results.get('individual_results'):
            ax.text(0.5, 0.5, 'Análisis de\nregresión\nno disponible', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Análisis de Regresión')
            return
        
        regression_results = {k: v for k, v in self.training_results['individual_results'].items() 
                            if 'mae' in v}
        
        if regression_results:
            targets = list(regression_results.keys())
            r2_scores = [results['r2'] for results in regression_results.values()]
            
            bars = ax.bar(targets, r2_scores, color='steelblue', alpha=0.7)
            
            for bar, r2 in zip(bars, r2_scores):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{r2:.3f}', ha='center', va='bottom')
            
            ax.set_ylabel('R² Score')
            ax.set_title('R² por Target de Regresión', fontweight='bold')
            ax.set_xticklabels([t.replace('_target', '') for t in targets], rotation=45)
            ax.grid(axis='y', alpha=0.3)

    def _plot_classification_analysis(self, ax):
        """Análisis de resultados de clasificación"""
        if not self.training_results or not self.training_results.get('individual_results'):
            ax.text(0.5, 0.5, 'Análisis de\nclasificación\nno disponible', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Análisis de Clasificación')
            return
        
        classification_results = {k: v for k, v in self.training_results['individual_results'].items() 
                                if 'accuracy' in v}
        
        if classification_results:
            targets = list(classification_results.keys())
            accuracies = [results['accuracy'] for results in classification_results.values()]
            
            bars = ax.bar(targets, accuracies, color='orange', alpha=0.7)
            
            for bar, acc in zip(bars, accuracies):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{acc:.3f}', ha='center', va='bottom')
            
            ax.set_ylabel('Accuracy')
            ax.set_title('Accuracy por Target de Clasificación', fontweight='bold')
            ax.set_xticklabels([t.replace('_target', '') for t in targets], rotation=45)
            ax.grid(axis='y', alpha=0.3)

    def _plot_temporal_ensemble_analysis(self, ax):
        """Análisis temporal del ensemble"""
        ax.text(0.5, 0.5, 'Análisis Temporal\ndel Ensemble\n(En desarrollo)', 
               ha='center', va='center', transform=ax.transAxes, fontsize=12)
        ax.set_title('Análisis Temporal del Ensemble', fontweight='bold')
        ax.axis('off')

    def _plot_prediction_stability(self, ax):
        """Análisis de estabilidad de predicciones"""
        ax.text(0.5, 0.5, 'Análisis de Estabilidad\nde Predicciones\n(En desarrollo)', 
               ha='center', va='center', transform=ax.transAxes, fontsize=12)
        ax.set_title('Estabilidad de Predicciones', fontweight='bold')
        ax.axis('off')

    def save_results(self):
        """
        Guarda todos los resultados del entrenamiento en archivos.
        """
        logger.info("Guardando resultados del ensemble")
        
        # Guardar ensemble completo
        if self.ensemble_model:
            ensemble_path = os.path.join(self.output_dir, "ensemble_nba_final.joblib")
            joblib.dump(self.ensemble_model, ensemble_path)
            logger.info(f"Ensemble guardado: {ensemble_path}")
        
        # Guardar métricas en JSON
        if self.training_results:
            # Preparar resultados para JSON
            json_results = {}
            for key, value in self.training_results.items():
                if key == 'ensemble_models':
                    # No incluir modelos en JSON, solo metadata
                    json_results[key] = {k: type(v).__name__ for k, v in value.items()}
                elif isinstance(value, dict):
                    json_results[key] = {k: float(v) if isinstance(v, (np.integer, np.floating)) else v 
                                       for k, v in value.items()}
                elif isinstance(value, (np.integer, np.floating)):
                    json_results[key] = float(value)
            else:
                    json_results[key] = value
            
            metrics_path = os.path.join(self.output_dir, "ensemble_training_results.json")
            with open(metrics_path, 'w') as f:
                json.dump(json_results, f, indent=2, default=str)
            logger.info(f"Métricas guardadas: {metrics_path}")
        
        # Guardar información de modelos cargados
        if self.loaded_models:
            models_info = {}
            for name, info in self.loaded_models.items():
                models_info[name] = {
                    'type': info['type'],
                    'target': info['target'], 
                    'path': info['path'],
                    'size_mb': info['size_mb'],
                    'n_features': info['n_features'],
                    'model_class': info['model_class'],
                    'feature_class': info['feature_class']
                }
            
            models_path = os.path.join(self.output_dir, "loaded_models_info.json")
            with open(models_path, 'w') as f:
                json.dump(models_info, f, indent=2)
            logger.info(f"Info modelos guardada: {models_path}")
        
        logger.info(f"Todos los archivos guardados en: {self.output_dir}")

    def run_complete_training(self):
        """
        Ejecuta el pipeline completo de entrenamiento del ensemble usando FinalEnsemble.
        """
        logger.info("Iniciando entrenamiento completo del ensemble NBA con modelos individuales")
        
        try:
            # 1. Cargar y preparar datos
            self.load_and_prepare_data()
            
            # 2. Configurar FinalEnsemble con modelos individuales
            self._setup_final_ensemble()
            
            # 3. Entrenar el FinalEnsemble completo
            logger.info("Entrenando FinalEnsemble...")
            ensemble_results = self.final_ensemble.train(self.df_players, self.df_teams)
            
            # 4. Guardar el ensemble entrenado
            ensemble_path = os.path.join(self.output_dir, "ensemble_nba_final.joblib")
            self.final_ensemble.save_ensemble(ensemble_path)
            logger.info(f"FinalEnsemble guardado: {ensemble_path}")
            
            # 5. Compilar resultados para compatibilidad
            self.training_results = {
                'ensemble_type': 'FinalEnsemble',
                'models_loaded': len(self.final_ensemble.loaded_models),
                'training_duration_seconds': ensemble_results.get('training_duration', 0),
                'validation_scores': ensemble_results.get('validation_scores', {}),
                'ensemble_models': ensemble_results.get('ensemble_models', {}),
                'final_ensemble_results': ensemble_results
            }
            
            # 6. Guardar resultados adicionales
            self.save_results()
            
            # Resumen final
            logger.info("\n" + "="*60)
            logger.info("RESUMEN FINAL DEL ENSEMBLE NBA")
            logger.info("="*60)
            logger.info(f"FinalEnsemble entrenado exitosamente")
            logger.info(f"Modelos individuales cargados: {len(self.final_ensemble.loaded_models)}")
            logger.info(f"Tiempo de entrenamiento: {ensemble_results.get('training_duration', 0):.1f}s")
            
            # Mostrar métricas si están disponibles
            if 'validation_scores' in ensemble_results:
                val_scores = ensemble_results['validation_scores']
                logger.info("Métricas de validación:")
                for metric_type, scores in val_scores.items():
                    if isinstance(scores, dict):
                        for metric, value in scores.items():
                            logger.info(f"  {metric_type} {metric}: {value:.3f}")
            
            logger.info("="*60)
            
            return self.training_results
            
        except Exception as e:
            logger.error(f"Error en entrenamiento del ensemble: {str(e)}")
            logger.error(f"Tipo de error: {type(e).__name__}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise


def main():
    """
    Función principal para ejecutar el entrenamiento completo del ensemble.
    """
    # Configurar logging ultra-silencioso para librerías externas
    logging.basicConfig(level=logging.ERROR)
    
    # Silenciar librerías específicas
    for lib in ['sklearn', 'xgboost', 'lightgbm', 'optuna']:
        logging.getLogger(lib).setLevel(logging.ERROR)
    
    # Rutas de datos (ajustar según tu configuración)
    game_data_path = "data/players.csv"
    biometrics_path = "data/height.csv"  
    teams_path = "data/teams.csv"
    
    # Crear y ejecutar trainer
    trainer = EnsembleNBATrainer(
        game_data_path=game_data_path,
        biometrics_path=biometrics_path,
        teams_path=teams_path,
        models_path="results",
        output_dir="results/ensemble_model",
        cv_folds=5,
        random_state=42
    )
    
    # Ejecutar entrenamiento completo
    results = trainer.run_complete_training()
    
    logger.info("¡Entrenamiento del ensemble completado!")
    logger.info(f"Resultados guardados en: {trainer.output_dir}")
    
    return results


if __name__ == "__main__":
    results = main() 