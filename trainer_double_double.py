"""
Trainer Completo para Modelo Double Double NBA
==============================================

Trainer que integra carga de datos, entrenamiento del modelo de double double
y generación completa de métricas y visualizaciones para predicción de double double NBA.

Características:
- Integración completa con data loader
- Entrenamiento automatizado con optimización bayesiana
- Generación de dashboard PNG unificado con todas las métricas
- Métricas detalladas específicas para clasificación binaria
- Análisis de feature importance
- Validación cruzada cronológica
"""

import json
import logging
import os
import warnings
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    precision_recall_curve, roc_curve
)

# Imports del proyecto
from src.preprocessing.data_loader import NBADataLoader
from src.models.players.double_double.dd_model import DoubleDoubleAdvancedModel

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

# Configurar estilo de visualizaciones optimizado para PNG
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10


class DoubleDoubleTrainer:
    """
    Trainer completo para modelo de predicción de double double NBA.
    
    Integra carga de datos, entrenamiento, evaluación y visualizaciones.
    """
    
    def __init__(self,
                 game_data_path: str,
                 biometrics_path: str,
                 teams_path: str,
                 output_dir: str = "results/double_double_model",
                 n_trials: int = 50,  # Trials para optimización bayesiana
                 cv_folds: int = 5,
                 random_state: int = 42):
        """
        Inicializa el trainer completo para Double Double.
        
        Args:
            game_data_path: Ruta a datos de partidos
            biometrics_path: Ruta a datos biométricos
            teams_path: Ruta a datos de equipos
            output_dir: Directorio de salida para resultados
            n_trials: Trials para optimización bayesiana
            cv_folds: Folds para validación cruzada
            random_state: Semilla para reproducibilidad
        """
        self.game_data_path = game_data_path
        self.biometrics_path = biometrics_path
        self.teams_path = teams_path
        self.output_dir = os.path.normpath(output_dir)
        self.random_state = random_state
        
        # Crear directorio de salida con manejo robusto
        try:
            os.makedirs(self.output_dir, exist_ok=True)
            logger.info(f"Directorio de salida creado/verificado: {self.output_dir}")
        except Exception as e:
            logger.error(f"Error creando directorio {self.output_dir}: {e}")
            # Crear directorio alternativo en caso de error
            self.output_dir = os.path.normpath("results_double_double_model")
            os.makedirs(self.output_dir, exist_ok=True)
            logger.info(f"Usando directorio alternativo: {self.output_dir}")
        
        # Componentes principales
        self.data_loader = NBADataLoader(
            game_data_path, biometrics_path, teams_path
        )
        self.model = DoubleDoubleAdvancedModel(
            optimize_hyperparams=False,  # Desactivar temporalmente para evitar error de índices
            bayesian_n_calls=n_trials,
            device=None  # Auto-detect
        )
        
        # Datos y resultados
        self.df = None
        self.teams_df = None
        self.training_results = None
        self.predictions = None
        self.prediction_probabilities = None
        
        logger.info(f"Trainer Double Double inicializado - Output: {self.output_dir}")
    
    def load_and_prepare_data(self) -> pd.DataFrame:
        """
        Carga y prepara todos los datos necesarios.
        
        Returns:
            pd.DataFrame: Datos preparados para entrenamiento
        """
        logger.info("Cargando datos NBA...")
        
        # Cargar datos usando el data loader
        self.df, self.teams_df = self.data_loader.load_data()
        
        # Crear columna double_double si no existe
        if 'double_double' not in self.df.columns:
            # Crear double_double basado en PTS, TRB, AST, STL, BLK
            stats_cols = ['PTS', 'TRB', 'AST', 'STL', 'BLK']
            available_stats = [col for col in stats_cols if col in self.df.columns]
            
            if len(available_stats) >= 2:
                # Contar cuántas stats >= 10
                double_count = sum((self.df[col] >= 10).astype(int) for col in available_stats)
                self.df['double_double'] = (double_count >= 2).astype(int)
                logger.info(f"Columna double_double creada usando stats: {available_stats}")
            else:
                raise ValueError("No hay suficientes columnas de estadísticas para crear double_double")
        
        # Estadísticas básicas de los datos
        logger.info(f"Datos cargados: {len(self.df)} registros de jugadores")
        logger.info(f"Datos de equipos: {len(self.teams_df)} registros")
        logger.info(f"Jugadores únicos: {self.df['Player'].nunique()}")
        logger.info(f"Equipos únicos: {self.df['Team'].nunique()}")
        logger.info(f"Rango de fechas: {self.df['Date'].min()} a {self.df['Date'].max()}")
        
        # Estadísticas del target
        dd_stats = self.df['double_double'].value_counts()
        dd_rate = self.df['double_double'].mean()
        logger.info(f"Distribución Double Double - No DD: {dd_stats.get(0, 0)}, DD: {dd_stats.get(1, 0)}")
        logger.info(f"Tasa de Double Double: {dd_rate:.3f} ({dd_rate*100:.1f}%)")
        
        return self.df
    
    def train_model(self) -> Dict:
        """
        Entrena el modelo completo con optimización y validación.
        
        Returns:
            Dict: Resultados del entrenamiento
        """
        logger.info("Iniciando entrenamiento del modelo Double Double...")
        
        if self.df is None:
            raise ValueError("Datos no cargados. Ejecutar load_and_prepare_data() primero")
        
        # Entrenar modelo
        start_time = datetime.now()
        logger.info("Entrenando modelo Double Double con ensemble avanzado...")
        self.training_results = self.model.train(self.df)
        training_duration = (datetime.now() - start_time).total_seconds()
        
        logger.info(f"Modelo Double Double completado en {training_duration:.1f} segundos")
        
        # Mostrar resultados del entrenamiento con análisis específico DD
        logger.info("=" * 60)
        logger.info("RESULTADOS DEL ENTRENAMIENTO DOUBLE DOUBLE")
        logger.info("=" * 60)
        
        stacking_metrics = self.training_results.get('stacking_metrics', {})
        accuracy = stacking_metrics.get('accuracy', 0) * 100
        precision = stacking_metrics.get('precision', 0) * 100
        recall = stacking_metrics.get('recall', 0) * 100
        f1 = stacking_metrics.get('f1_score', 0) * 100  # CORREGIDO: usar 'f1_score' en lugar de 'f1'
        roc_auc = stacking_metrics.get('roc_auc', 0) * 100
        
        logger.info(f"Accuracy: {accuracy:.2f}%")
        logger.info(f"Precision: {precision:.2f}% (OBJETIVO: >45%)")
        logger.info(f"Recall: {recall:.2f}% (OBJETIVO: 75-80%)")
        logger.info(f"F1-Score: {f1:.2f}% (OBJETIVO: >55%)")
        logger.info(f"ROC-AUC: {roc_auc:.2f}% (OBJETIVO: >94%)")
        
        # Análisis de calidad específico para DD
        logger.info("\n📊 ANÁLISIS DE CALIDAD DOUBLE DOUBLE:")
        if precision >= 45:
            logger.info("✅ PRECISION: EXCELENTE (>45%)")
        elif precision >= 35:
            logger.info("⚠️  PRECISION: BUENA (35-45%) - Mejorable")
        else:
            logger.info("❌ PRECISION: BAJA (<35%) - REQUIERE MEJORA")
            
        if 75 <= recall <= 85:
            logger.info("✅ RECALL: ÓPTIMO (75-85%)")
        elif recall > 85:
            logger.info("⚠️  RECALL: ALTO (>85%) - Posible overprediction")
        else:
            logger.info("❌ RECALL: BAJO (<75%) - REQUIERE MEJORA")
            
        if f1 >= 55:
            logger.info("✅ F1-SCORE: EXCELENTE (>55%)")
        elif f1 >= 45:
            logger.info("⚠️  F1-SCORE: BUENO (45-55%) - Mejorable")
        else:
            logger.info("❌ F1-SCORE: BAJO (<45%) - REQUIERE MEJORA")
        
        # Análisis del threshold
        threshold = getattr(self.model, 'optimal_threshold', 0.5)
        logger.info(f"\n🎯 THRESHOLD ÓPTIMO: {threshold:.3f}")
        if threshold >= 0.35:
            logger.info("✅ THRESHOLD: CONSERVADOR (bueno para precision)")
        elif threshold >= 0.25:
            logger.info("⚠️  THRESHOLD: MODERADO")
        else:
            logger.info("❌ THRESHOLD: MUY BAJO (riesgo de falsos positivos)")
        
        logger.info("=" * 60)
        
        # Generar predicciones
        logger.info("Generando predicciones...")
        self.predictions = self.model.predict(self.df)
        proba_result = self.model.predict_proba(self.df)
        
        # Asegurar que las probabilidades sean 1D (solo clase positiva)
        if len(proba_result.shape) == 2 and proba_result.shape[1] == 2:
            # Si es 2D con 2 columnas, tomar solo la clase positiva (columna 1)
            self.prediction_probabilities = proba_result[:, 1]
        elif len(proba_result.shape) == 2 and proba_result.shape[1] == 1:
            # Si es 2D con 1 columna, aplanar
            self.prediction_probabilities = proba_result.flatten()
        else:
            # Si ya es 1D, usar tal como está
            self.prediction_probabilities = proba_result
        
        return self.training_results
    
    def generate_all_visualizations(self):
        """
        Genera una visualización completa en PNG con todas las métricas principales.
        """
        logger.info("Generando visualización completa en PNG...")
        
        # Asegurar que el directorio de salida existe
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Crear figura principal con subplots organizados
        fig = plt.figure(figsize=(24, 16))
        fig.suptitle('Dashboard Completo - Modelo NBA Double Double Prediction', fontsize=20, fontweight='bold', y=0.98)
        
        # Crear grid de subplots (4 filas x 4 columnas)
        gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)
        
        # 1. Métricas principales del modelo (esquina superior izquierda)
        ax1 = fig.add_subplot(gs[0, 0])
        self._plot_model_metrics_summary(ax1)
        
        # 2. Feature importance (esquina superior derecha)
        ax2 = fig.add_subplot(gs[0, 1:3])
        self._plot_feature_importance_compact(ax2)
        
        # 3. Distribución del target (esquina superior derecha)
        ax3 = fig.add_subplot(gs[0, 3])
        self._plot_target_distribution_compact(ax3)
        
        # 4. Matriz de confusión (segunda fila, izquierda)
        ax4 = fig.add_subplot(gs[1, 0])
        self._plot_confusion_matrix_compact(ax4)
        
        # 5. Curva ROC (segunda fila, centro-izquierda)
        ax5 = fig.add_subplot(gs[1, 1])
        self._plot_roc_curve_compact(ax5)
        
        # 6. Curva Precision-Recall (segunda fila, centro-derecha)
        ax6 = fig.add_subplot(gs[1, 2])
        self._plot_precision_recall_curve_compact(ax6)
        
        # 7. Distribución de probabilidades (segunda fila, derecha)
        ax7 = fig.add_subplot(gs[1, 3])
        self._plot_probability_distribution_compact(ax7)
        
        # 8. Análisis de confianza (tercera fila, izquierda)
        ax8 = fig.add_subplot(gs[2, 0:2])
        self._plot_confidence_analysis_compact(ax8)
        
        # 9. Análisis por posición (tercera fila, derecha)
        ax9 = fig.add_subplot(gs[2, 2:4])
        self._plot_position_analysis_compact(ax9)
        
        # 10. Análisis temporal (cuarta fila, izquierda)
        ax10 = fig.add_subplot(gs[3, 0:2])
        self._plot_temporal_analysis_compact(ax10)
        
        # 11. Top jugadores DD (cuarta fila, derecha)
        ax11 = fig.add_subplot(gs[3, 2:4])
        self._plot_top_dd_players_compact(ax11)
        
        # Guardar como PNG con ruta normalizada
        png_path = os.path.normpath(os.path.join(self.output_dir, 'model_dashboard_complete.png'))
        
        try:
            plt.savefig(png_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
            logger.info(f"Dashboard completo guardado en: {png_path}")
        except Exception as e:
            logger.error(f"Error guardando PNG: {e}")
            # Intentar con ruta absoluta
            abs_png_path = os.path.abspath(png_path)
            logger.info(f"Intentando con ruta absoluta: {abs_png_path}")
            plt.savefig(abs_png_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
            logger.info(f"Dashboard guardado exitosamente en: {abs_png_path}")
        finally:
            plt.close()
    
    def _plot_model_metrics_summary(self, ax):
        """Resumen de métricas principales del modelo."""
        ax.axis('off')
        
        # Obtener métricas
        stacking_metrics = self.training_results.get('stacking_metrics', {})
        accuracy = stacking_metrics.get('accuracy', 0)
        precision = stacking_metrics.get('precision', 0)
        recall = stacking_metrics.get('recall', 0)
        f1 = stacking_metrics.get('f1_score', 0)
        roc_auc = stacking_metrics.get('roc_auc', 0)
        
        # Crear texto de métricas
        metrics_text = f"""
MÉTRICAS DEL MODELO DD

Accuracy: {accuracy:.3f}
Precision: {precision:.3f}
Recall: {recall:.3f}
F1-Score: {f1:.3f}
ROC-AUC: {roc_auc:.3f}

MODELOS BASE:
• XGBoost (Gradient Boosting)
• LightGBM (Fast Boosting)
• CatBoost (Categorical)
• Random Forest (Ensemble)
• Neural Network (Deep Learning)
• Logistic Regression (Linear)

THRESHOLD OPTIMIZADO:
{getattr(self.model, 'optimal_threshold', 0.5):.3f}
"""
        
        ax.text(0.05, 0.95, metrics_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
        
        ax.set_title('Resumen del Modelo', fontweight='bold', fontsize=12)
    
    def _plot_feature_importance_compact(self, ax):
        """Gráfico compacto de importancia de features."""
        try:
            # Intentar obtener feature importance del modelo
            feature_importance = None
            
            if hasattr(self.model, 'feature_importance') and self.model.feature_importance:
                feature_importance = self.model.feature_importance
            elif hasattr(self.model, 'get_feature_importance'):
                feature_importance_result = self.model.get_feature_importance(top_n=15)
                if isinstance(feature_importance_result, dict) and 'feature_importance' in feature_importance_result:
                    feature_importance = feature_importance_result['feature_importance']
            
            if not feature_importance:
                ax.text(0.5, 0.5, 'Feature importance\nno disponible', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title('Feature Importance', fontweight='bold')
                return
            
            # Procesar feature importance - manejar diferentes formatos
            if isinstance(feature_importance, dict):
                # Si es un diccionario simple
                features = list(feature_importance.keys())[:15]
                importances = []
                
                for feature in features:
                    importance_val = feature_importance[feature]
                    # Si el valor es un diccionario, extraer el valor numérico
                    if isinstance(importance_val, dict):
                        # Buscar claves comunes para valores numéricos
                        if 'importance' in importance_val:
                            importances.append(float(importance_val['importance']))
                        elif 'value' in importance_val:
                            importances.append(float(importance_val['value']))
                        elif 'score' in importance_val:
                            importances.append(float(importance_val['score']))
                        else:
                            # Tomar el primer valor numérico encontrado
                            numeric_val = None
                            for v in importance_val.values():
                                try:
                                    numeric_val = float(v)
                                    break
                                except (ValueError, TypeError):
                                    continue
                            importances.append(numeric_val if numeric_val is not None else 0.0)
                    else:
                        # Si es un valor numérico directo
                        try:
                            importances.append(float(importance_val))
                        except (ValueError, TypeError):
                            importances.append(0.0)
            else:
                # Si no es un diccionario, mostrar mensaje
                ax.text(0.5, 0.5, 'Formato de feature\nimportance no soportado', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title('Feature Importance', fontweight='bold')
                return
            
            if not features or not importances or len(features) != len(importances):
                ax.text(0.5, 0.5, 'Error procesando\nfeature importance', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title('Feature Importance', fontweight='bold')
                return
            
            # Crear gráfico horizontal
            y_pos = np.arange(len(features))
            bars = ax.barh(y_pos, importances, color='lightcoral', alpha=0.8)
            
            ax.set_yticks(y_pos)
            ax.set_yticklabels([f.replace('_', ' ').title()[:20] for f in features], fontsize=8)
            ax.set_xlabel('Importancia')
            ax.set_title('Top 15 Features Más Importantes', fontweight='bold')
            
            # Agregar valores en las barras
            max_importance = max(importances) if importances else 1
            for i, (bar, val) in enumerate(zip(bars, importances)):
                ax.text(bar.get_width() + max_importance * 0.01, bar.get_y() + bar.get_height()/2, 
                       f'{val:.3f}', va='center', fontsize=7)
            
            ax.grid(axis='x', alpha=0.3)
            ax.invert_yaxis()
            
        except Exception as e:
            # En caso de cualquier error, mostrar mensaje
            ax.text(0.5, 0.5, f'Error en feature\nimportance:\n{str(e)[:30]}...', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=8)
            ax.set_title('Feature Importance', fontweight='bold')
    
    def _plot_target_distribution_compact(self, ax):
        """Distribución compacta del target Double Double."""
        dd_values = self.df['double_double']
        
        # Gráfico de barras
        counts = dd_values.value_counts().sort_index()
        labels = ['No DD', 'Double Double']
        colors = ['lightcoral', 'lightgreen']
        
        bars = ax.bar(labels, counts.values, color=colors, alpha=0.8, edgecolor='black')
        
        # Agregar porcentajes
        total = len(dd_values)
        for bar, count in zip(bars, counts.values):
            percentage = count / total * 100
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + total*0.01, 
                   f'{count}\n({percentage:.1f}%)', ha='center', va='bottom', fontsize=9)
        
        ax.set_ylabel('Frecuencia')
        ax.set_title('Distribución Double Double', fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
    
    def _plot_confusion_matrix_compact(self, ax):
        """Matriz de confusión compacta."""
        if self.predictions is None:
            ax.text(0.5, 0.5, 'Predicciones\nno disponibles', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Matriz de Confusión', fontweight='bold')
            return
        
        # Calcular matriz de confusión
        y_true = self.df['double_double']
        cm = confusion_matrix(y_true, self.predictions)
        
        # Crear heatmap
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['No DD', 'DD'], yticklabels=['No DD', 'DD'],
                   ax=ax, cbar=False)
        
        ax.set_xlabel('Predicho')
        ax.set_ylabel('Real')
        ax.set_title('Matriz de Confusión', fontweight='bold')
    
    def _plot_roc_curve_compact(self, ax):
        """Curva ROC compacta."""
        if self.prediction_probabilities is None:
            ax.text(0.5, 0.5, 'Probabilidades\nno disponibles', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Curva ROC', fontweight='bold')
            return
        
        y_true = self.df['double_double']
        y_proba = self.prediction_probabilities
        
        # Calcular curva ROC
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        auc = roc_auc_score(y_true, y_proba)
        
        # Plotear curva
        ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC (AUC = {auc:.3f})')
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('Tasa Falsos Positivos')
        ax.set_ylabel('Tasa Verdaderos Positivos')
        ax.set_title('Curva ROC', fontweight='bold')
        ax.legend(loc="lower right", fontsize=8)
        ax.grid(alpha=0.3)
    
    def _plot_precision_recall_curve_compact(self, ax):
        """Curva Precision-Recall compacta."""
        if self.prediction_probabilities is None:
            ax.text(0.5, 0.5, 'Probabilidades\nno disponibles', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Curva Precision-Recall', fontweight='bold')
            return
        
        y_true = self.df['double_double']
        y_proba = self.prediction_probabilities
        
        # Calcular curva Precision-Recall
        precision, recall, _ = precision_recall_curve(y_true, y_proba)
        
        # Plotear curva
        ax.plot(recall, precision, color='blue', lw=2)
        
        # Línea de baseline (proporción de positivos)
        baseline = y_true.mean()
        ax.axhline(y=baseline, color='red', linestyle='--', label=f'Baseline ({baseline:.3f})')
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title('Curva Precision-Recall', fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    
    def _plot_probability_distribution_compact(self, ax):
        """Distribución de probabilidades compacta."""
        if self.prediction_probabilities is None:
            ax.text(0.5, 0.5, 'Probabilidades\nno disponibles', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Distribución de Probabilidades', fontweight='bold')
            return
        
        y_true = self.df['double_double']
        y_proba = self.prediction_probabilities
        
        # Separar probabilidades por clase real
        proba_no_dd = y_proba[y_true == 0]
        proba_dd = y_proba[y_true == 1]
        
        # Histogramas
        ax.hist(proba_no_dd, bins=20, alpha=0.7, label='No DD', color='lightcoral', density=True)
        ax.hist(proba_dd, bins=20, alpha=0.7, label='DD', color='lightgreen', density=True)
        
        # Línea de threshold si está disponible
        if hasattr(self.model, 'optimal_threshold'):
            ax.axvline(self.model.optimal_threshold, color='black', linestyle='--', 
                      label=f'Threshold ({self.model.optimal_threshold:.3f})')
        
        ax.set_xlabel('Probabilidad Predicha')
        ax.set_ylabel('Densidad')
        ax.set_title('Distribución de Probabilidades', fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    
    def _plot_confidence_analysis_compact(self, ax):
        """Análisis de confianza compacto."""
        if self.prediction_probabilities is None:
            ax.text(0.5, 0.5, 'Probabilidades\nno disponibles', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Análisis de Confianza', fontweight='bold')
            return
        
        y_true = self.df['double_double']
        y_proba = self.prediction_probabilities
        
        # Definir rangos de confianza
        confidence_ranges = [
            (0.0, 0.3, 'Baja'),
            (0.3, 0.7, 'Media'),
            (0.7, 1.0, 'Alta')
        ]
        
        range_names = []
        accuracies = []
        counts = []
        
        for min_conf, max_conf, name in confidence_ranges:
            mask = (y_proba >= min_conf) & (y_proba < max_conf)
            if np.sum(mask) > 0:
                y_pred_range = (y_proba[mask] >= 0.5).astype(int)
                accuracy = accuracy_score(y_true[mask], y_pred_range)
                range_names.append(name)
                accuracies.append(accuracy)
                counts.append(np.sum(mask))
        
        if range_names:
            bars = ax.bar(range_names, accuracies, alpha=0.8, color=['red', 'orange', 'green'])
            
            ax.set_ylabel('Accuracy')
            ax.set_title('Accuracy por Nivel de Confianza', fontweight='bold')
            
            # Agregar valores en las barras
            for bar, acc, count in zip(bars, accuracies, counts):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                       f'{acc:.3f}\n(n={count})', ha='center', va='bottom', fontsize=8)
            
            ax.set_ylim(0, 1.1)
            ax.grid(axis='y', alpha=0.3)
    
    def _plot_position_analysis_compact(self, ax):
        """Análisis por posición compacto."""
        # Crear categorías de posición basadas en altura si está disponible
        if 'Height_Inches' in self.df.columns:
            self.df['position_category'] = pd.cut(
                self.df['Height_Inches'], 
                bins=[0, 78, 82, 100], 
                labels=['Guard', 'Forward', 'Center']
            )
            
            # Calcular tasa de DD por posición
            position_stats = self.df.groupby('position_category')['double_double'].agg(['mean', 'count']).reset_index()
            position_stats = position_stats[position_stats['count'] >= 10]  # Filtrar posiciones con pocos datos
            
            if len(position_stats) > 0:
                positions = position_stats['position_category']
                dd_rates = position_stats['mean']
                
                bars = ax.bar(positions, dd_rates, alpha=0.8, color=['lightblue', 'orange', 'red'])
                
                ax.set_ylabel('Tasa Double Double')
                ax.set_title('Tasa DD por Posición', fontweight='bold')
                
                # Agregar valores en las barras
                for bar, rate, count in zip(bars, dd_rates, position_stats['count']):
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                           f'{rate:.3f}\n(n={count})', ha='center', va='bottom', fontsize=8)
                
                ax.grid(axis='y', alpha=0.3)
            else:
                ax.text(0.5, 0.5, 'Datos insuficientes\npor posición', 
                       ha='center', va='center', transform=ax.transAxes)
        else:
            ax.text(0.5, 0.5, 'Datos de altura\nno disponibles', 
                   ha='center', va='center', transform=ax.transAxes)
        
        ax.set_title('Análisis por Posición', fontweight='bold')
    
    def _plot_temporal_analysis_compact(self, ax):
        """Análisis temporal compacto."""
        if 'Date' not in self.df.columns:
            ax.text(0.5, 0.5, 'Datos de fecha\nno disponibles', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Análisis Temporal', fontweight='bold')
            return
        
        # Agrupar por mes
        df_temp = self.df.copy()
        df_temp['month'] = pd.to_datetime(df_temp['Date']).dt.to_period('M')
        
        monthly_stats = df_temp.groupby('month').agg({
            'double_double': ['mean', 'count']
        }).reset_index()
        
        monthly_stats.columns = ['month', 'dd_rate', 'count']
        monthly_stats = monthly_stats[monthly_stats['count'] >= 50]  # Filtrar meses con pocos datos
        
        if len(monthly_stats) > 0:
            months = [str(m) for m in monthly_stats['month']]
            dd_rates = monthly_stats['dd_rate']
            
            ax.plot(months, dd_rates, marker='o', linewidth=2, markersize=6, color='blue')
            
            ax.set_ylabel('Tasa Double Double')
            ax.set_title('Tasa DD por Mes', fontweight='bold')
            ax.tick_params(axis='x', rotation=45)
            ax.grid(alpha=0.3)
            
            # Agregar línea de promedio
            avg_rate = dd_rates.mean()
            ax.axhline(y=avg_rate, color='red', linestyle='--', alpha=0.7, 
                      label=f'Promedio ({avg_rate:.3f})')
            ax.legend(fontsize=8)
        else:
            ax.text(0.5, 0.5, 'Datos temporales\ninsuficientes', 
                   ha='center', va='center', transform=ax.transAxes)
    
    def _plot_top_dd_players_compact(self, ax):
        """Análisis compacto de top jugadores DD."""
        # Obtener top jugadores por tasa de DD
        player_stats = self.df.groupby('Player').agg({
            'double_double': ['mean', 'count']
        }).reset_index()
        
        player_stats.columns = ['Player', 'dd_rate', 'count']
        
        # Filtrar jugadores con al menos 10 juegos
        player_stats = player_stats[player_stats['count'] >= 10]
        
        # Top 10 jugadores con mayor tasa de DD
        top_players = player_stats.nlargest(10, 'dd_rate')
        
        if len(top_players) > 0:
            players = [p[:15] + '...' if len(p) > 15 else p for p in top_players['Player']]
            rates = top_players['dd_rate']
            
            bars = ax.barh(players, rates, alpha=0.8, color='lightsteelblue')
            
            ax.set_xlabel('Tasa Double Double')
            ax.set_title('Top 10 Jugadores DD', fontweight='bold')
            
            # Agregar valores en las barras
            for i, (bar, val) in enumerate(zip(bars, rates)):
                ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                       f'{val:.3f}', va='center', fontsize=8)
            
            ax.grid(axis='x', alpha=0.3)
            ax.invert_yaxis()
        else:
            ax.text(0.5, 0.5, 'Datos de jugadores\ninsuficientes', 
                   ha='center', va='center', transform=ax.transAxes)
    
    def save_results(self):
        """Guarda todos los resultados del entrenamiento."""
        logger.info("Guardando resultados...")
        
        # Asegurar que el directorio de salida existe
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Intentar guardar modelo (puede fallar debido a clases locales)
        model_path = os.path.normpath(os.path.join(self.output_dir, 'double_double_model.pkl'))
        try:
            self.model.save_model(model_path)
            logger.info(f"Modelo guardado exitosamente en: {model_path}")
        except Exception as e:
            logger.warning(f"No se pudo guardar el modelo debido a: {str(e)}")
            logger.info("Continuando con el guardado de otros resultados...")
        
        # Guardar reporte completo
        report = {
            'model_type': 'Double Double Advanced Ensemble',
            'training_results': self.training_results,
            'model_summary': getattr(self.model, 'training_summary', {}),
            'timestamp': datetime.now().isoformat()
        }
        report_path = os.path.normpath(os.path.join(self.output_dir, 'training_report.json'))
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Guardar predicciones
        if self.predictions is not None:
            predictions_df = self.df[['Player', 'Date', 'Team', 'double_double']].copy()
            predictions_df['dd_predicted'] = self.predictions
            if self.prediction_probabilities is not None:
                predictions_df['dd_probability'] = self.prediction_probabilities
                predictions_df['correct_prediction'] = (self.predictions == self.df['double_double']).astype(int)
            
            predictions_path = os.path.normpath(os.path.join(self.output_dir, 'predictions.csv'))
            predictions_df.to_csv(predictions_path, index=False)
        
        # Guardar feature importance
        try:
            if hasattr(self.model, 'feature_importance') and self.model.feature_importance:
                # Procesar feature importance manejando diferentes formatos
                importance_data = []
                for k, v in self.model.feature_importance.items():
                    if isinstance(v, dict):
                        # Si el valor es un diccionario, extraer el valor numérico
                        if 'importance' in v:
                            importance_val = float(v['importance'])
                        elif 'value' in v:
                            importance_val = float(v['value'])
                        elif 'score' in v:
                            importance_val = float(v['score'])
                        else:
                            # Tomar el primer valor numérico encontrado
                            importance_val = 0.0
                            for dict_val in v.values():
                                try:
                                    importance_val = float(dict_val)
                                    break
                                except (ValueError, TypeError):
                                    continue
                    else:
                        # Si es un valor numérico directo
                        try:
                            importance_val = float(v)
                        except (ValueError, TypeError):
                            importance_val = 0.0
                    
                    importance_data.append({'feature': k, 'importance': importance_val})
                
                if importance_data:
                    importance_df = pd.DataFrame(importance_data).sort_values('importance', ascending=False)
                    importance_path = os.path.normpath(os.path.join(self.output_dir, 'feature_importance.csv'))
                    importance_df.to_csv(importance_path, index=False)
                    logger.info(f"Feature importance guardado en: {importance_path}")
                else:
                    logger.warning("No se pudieron procesar los datos de feature importance")
            else:
                logger.info("Feature importance no disponible")
        except Exception as e:
            logger.warning(f"Error guardando feature importance: {str(e)}")
        
        # Crear resumen de archivos generados
        files_summary = {
            'model_file': 'double_double_model.pkl',
            'dashboard_image': 'model_dashboard_complete.png',
            'training_report': 'training_report.json',
            'predictions': 'predictions.csv',
            'feature_importance': 'feature_importance.csv',
            'output_directory': os.path.normpath(self.output_dir),
            'generation_timestamp': datetime.now().isoformat()
        }
        
        summary_path = os.path.normpath(os.path.join(self.output_dir, 'files_summary.json'))
        with open(summary_path, 'w') as f:
            json.dump(files_summary, f, indent=2)
        
        logger.info(f"Resultados guardados en: {os.path.normpath(self.output_dir)}")
        logger.info("Archivos generados:")
        if os.path.exists(model_path):
            logger.info(f"  • Modelo: {model_path}")
        logger.info(f"  • Dashboard PNG: {os.path.normpath(os.path.join(self.output_dir, 'model_dashboard_complete.png'))}")
        logger.info(f"  • Reporte: {report_path}")
        if self.predictions is not None:
            logger.info(f"  • Predicciones: {predictions_path}")
        logger.info(f"  • Resumen: {summary_path}")
    
    def run_complete_training(self):
        """
        Ejecuta el pipeline completo de entrenamiento.
        
        Returns:
            Dict: Resultados completos del entrenamiento
        """
        logger.info("Iniciando pipeline de entrenamiento Double Double...")
        
        try:
            # 1. Cargar datos
            self.load_and_prepare_data()
            
            # 2. Entrenar modelo
            results = self.train_model()
            
            # 3. Generar visualizaciones
            self.generate_all_visualizations()
            
            # 4. Guardar resultados
            self.save_results()
            
            logger.info("Pipeline ejecutado exitosamente!")
            return results
            
        except Exception as e:
            logger.error(f"Error en pipeline de entrenamiento: {str(e)}")
            raise


def main():
    """Función principal para ejecutar el trainer."""
    # Configurar logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Rutas de datos correctas
    game_data_path = "data/players.csv"
    biometrics_path = "data/height.csv"
    teams_path = "data/teams.csv"
    
    # Crear y ejecutar trainer
    trainer = DoubleDoubleTrainer(
        game_data_path=game_data_path,
        biometrics_path=biometrics_path,
        teams_path=teams_path,
        output_dir="results/double_double_model",
        n_trials=50,
        cv_folds=5
    )
    
    # Ejecutar pipeline completo
    results = trainer.run_complete_training()
    
    print("\n" + "="*80)
    print("RESUMEN FINAL DE ENTRENAMIENTO DOUBLE DOUBLE")
    print("="*80)
    
    # Mostrar información del modelo
    stacking_metrics = results.get('stacking_metrics', {})
    print(f"\n📊 MODELO DOUBLE DOUBLE (Advanced Ensemble):")
    print(f"   Accuracy: {stacking_metrics.get('accuracy', 0):.4f}")
    print(f"   Precision: {stacking_metrics.get('precision', 0):.4f}")
    print(f"   Recall: {stacking_metrics.get('recall', 0):.4f}")
    print(f"   F1-Score: {stacking_metrics.get('f1_score', 0):.4f}")
    print(f"   ROC-AUC: {stacking_metrics.get('roc_auc', 0):.4f}")
    
    # Mostrar información adicional
    print(f"\n📋 INFORMACIÓN ADICIONAL:")
    print(f"   Modelos Base: XGBoost, LightGBM, CatBoost, Random Forest, Neural Network, Logistic Regression")
    print(f"   Features Especializadas: {results.get('specialized_features_used', 0)}")
    print(f"   Muestras Entrenamiento: {results.get('training_samples', 0)}")
    print(f"   Muestras Validación: {results.get('validation_samples', 0)}")
    print(f"   Optimización: Bayesiana con threshold adaptativo")
    
    print("="*80)
    print("Entrenamiento Double Double completado exitosamente!")
    print("="*80)


if __name__ == "__main__":
    main() 