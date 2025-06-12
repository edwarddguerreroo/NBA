"""
Trainer Completo para Modelo NBA Is Win
=======================================

Trainer que integra carga de datos, entrenamiento del modelo de victorias
y generación completa de métricas y visualizaciones para predicción de victorias NBA.

Características:
- Integración completa con data loader
- Entrenamiento automatizado con optimización bayesiana
- Generación de dashboard PNG unificado con todas las métricas
- Métricas detalladas específicas para clasificación de victorias
- Análisis de feature importance
- Validación cruzada estratificada
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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report

# Imports del proyecto
from src.preprocessing.data_loader import NBADataLoader
from src.models.teams.is_win.model_is_win import IsWinModel

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

# Configurar estilo de visualizaciones optimizado para PNG
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10


class IsWinTrainer:
    """
    Trainer completo para modelo de predicción de victorias NBA.
    
    Integra carga de datos, entrenamiento, evaluación y visualizaciones.
    """
    
    def __init__(self,
                 game_data_path: str,
                 biometrics_path: str,
                 teams_path: str,
                 output_dir: str = "results/is_win_model",
                 n_trials: int = 50,  # Trials para optimización bayesiana
                 cv_folds: int = 5,
                 random_state: int = 42):
        """
        Inicializa el trainer completo para predicción de victorias.
        
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
            self.output_dir = os.path.normpath("results_is_win_model")
            os.makedirs(self.output_dir, exist_ok=True)
            logger.info(f"Usando directorio alternativo: {self.output_dir}")
        
        # Componentes principales
        self.data_loader = NBADataLoader(
            game_data_path, biometrics_path, teams_path
        )
        self.model = IsWinModel(
            optimize_hyperparams=True,
            device=None,
            bayesian_n_calls=max(50, n_trials),  # Asegurar mínimo de 50 calls
            min_memory_gb=2.0
        )
        
        # Datos y resultados
        self.df = None
        self.teams_df = None
        self.training_results = None
        self.predictions = None
        self.probabilities = None
        
        logger.info(f"Trainer Is Win inicializado - Output: {self.output_dir}")
    
    def load_and_prepare_data(self) -> pd.DataFrame:
        """
        Carga y prepara todos los datos necesarios.
        
        Returns:
            pd.DataFrame: Datos preparados para entrenamiento
        """
        logger.info("Cargando datos NBA...")
        
        # Cargar datos usando el data loader
        self.df, self.teams_df = self.data_loader.load_data()
        
        # Procesar datos específicos para modelo de victorias
        teams_data = self.teams_df.copy()
        
        # Crear variable target 'is_win' basada en 'Result'
        if 'Result' in teams_data.columns:
            teams_data['is_win'] = teams_data['Result'].str.startswith('W').astype(int)
        else:
            raise ValueError("Columna 'Result' no encontrada en datos de equipos")
        
        # Estadísticas básicas de los datos
        logger.info(f"Datos cargados: {len(teams_data)} registros de equipos")
        logger.info(f"Equipos únicos: {teams_data['Team'].nunique()}")
        logger.info(f"Rango de fechas: {teams_data['Date'].min()} a {teams_data['Date'].max()}")
        
        # Estadísticas del target
        win_rate = teams_data['is_win'].mean()
        total_games = len(teams_data)
        wins = teams_data['is_win'].sum()
        losses = total_games - wins
        
        logger.info(f"Estadísticas de victorias:")
        logger.info(f"  - Total partidos: {total_games}")
        logger.info(f"  - Victorias: {wins} ({win_rate:.1%})")
        logger.info(f"  - Derrotas: {losses} ({1-win_rate:.1%})")
        
        self.df = teams_data
        return self.df
    
    def train_model(self) -> Dict:
        """
        Entrena el modelo completo con optimización y validación.
        
        Returns:
            Dict: Resultados del entrenamiento
        """
        logger.info("Iniciando entrenamiento del modelo Is Win...")
        
        if self.df is None:
            raise ValueError("Datos no cargados. Ejecutar load_and_prepare_data() primero")
        
        # Entrenar modelo
        start_time = datetime.now()
        logger.info("Entrenando modelo Is Win con ensemble optimizado...")
        self.training_results = self.model.train(self.df, validation_split=0.2)
        training_duration = (datetime.now() - start_time).total_seconds()
        
        logger.info(f"Modelo Is Win completado en {training_duration:.1f} segundos")
        
        # Mostrar resultados del entrenamiento
        logger.info("=" * 50)
        logger.info("RESULTADOS DEL ENTRENAMIENTO IS WIN")
        logger.info("=" * 50)
        
        if 'accuracy' in self.training_results:
            logger.info(f"Accuracy: {self.training_results['accuracy']:.4f}")
        if 'precision' in self.training_results:
            logger.info(f"Precision: {self.training_results['precision']:.4f}")
        if 'recall' in self.training_results:
            logger.info(f"Recall: {self.training_results['recall']:.4f}")
        if 'f1_score' in self.training_results:
            logger.info(f"F1-Score: {self.training_results['f1_score']:.4f}")
        if 'auc_roc' in self.training_results:
            logger.info(f"AUC-ROC: {self.training_results['auc_roc']:.4f}")
        
        logger.info("=" * 50)
        
        # Generar predicciones
        logger.info("Generando predicciones...")
        self.probabilities = self.model.predict_proba(self.df)
        self.predictions = self.model.predict(self.df)
        
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
        fig.suptitle('Dashboard Completo - Modelo NBA Is Win Prediction', fontsize=20, fontweight='bold', y=0.98)
        
        # Crear grid de subplots (4 filas x 4 columnas)
        gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)
        
        # 1. Métricas principales del modelo
        ax1 = fig.add_subplot(gs[0, 0])
        self._plot_model_metrics_summary(ax1)
        
        # 2. Feature importance
        ax2 = fig.add_subplot(gs[0, 1:3])
        self._plot_feature_importance_compact(ax2)
        
        # 3. Distribución de clases
        ax3 = fig.add_subplot(gs[0, 3])
        self._plot_class_distribution_compact(ax3)
        
        # 4. Matriz de confusión
        ax4 = fig.add_subplot(gs[1, 0])
        self._plot_confusion_matrix_compact(ax4)
        
        # 5. Curva ROC
        ax5 = fig.add_subplot(gs[1, 1])
        self._plot_roc_curve_compact(ax5)
        
        # 6. Distribución de probabilidades
        ax6 = fig.add_subplot(gs[1, 2:4])
        self._plot_probability_distribution_compact(ax6)
        
        # 7. Accuracy por rango de confianza
        ax7 = fig.add_subplot(gs[2, 0:2])
        self._plot_confidence_analysis_compact(ax7)
        
        # 8. Análisis por equipos
        ax8 = fig.add_subplot(gs[2, 2:4])
        self._plot_team_performance_compact(ax8)
        
        # 9. Análisis temporal
        ax9 = fig.add_subplot(gs[3, 0:2])
        self._plot_temporal_analysis_compact(ax9)
        
        # 10. Validación cruzada
        ax10 = fig.add_subplot(gs[3, 2:4])
        self._plot_cv_results_compact(ax10)
        
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
        accuracy = self.training_results.get('accuracy', 0)
        precision = self.training_results.get('precision', 0)
        recall = self.training_results.get('recall', 0)
        f1 = self.training_results.get('f1_score', 0)
        auc = self.training_results.get('auc_roc', 0)
        
        # Crear texto de métricas
        metrics_text = f"""
MÉTRICAS DEL MODELO IS WIN

Accuracy: {accuracy:.3f}
Precision: {precision:.3f}
Recall: {recall:.3f}
F1-Score: {f1:.3f}
AUC-ROC: {auc:.3f}

MODELO ENSEMBLE:
• XGBoost
• LightGBM
• Random Forest
• Neural Network
• Logistic Regression
• SVM
"""
        
        ax.text(0.05, 0.95, metrics_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7))
        
        ax.set_title('Resumen del Modelo', fontweight='bold', fontsize=12)
    
    def _plot_feature_importance_compact(self, ax):
        """Gráfico compacto de importancia de features."""
        try:
            # Obtener feature importance del modelo
            importance_dict = self.model.get_feature_importance(top_n=15)
            
            if not importance_dict or 'feature_importance' not in importance_dict:
                ax.text(0.5, 0.5, 'Feature importance\nno disponible', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title('Feature Importance', fontweight='bold')
                return
            
            # Extraer top features
            feature_data = importance_dict['feature_importance']
            top_features = dict(list(feature_data.items())[:15])
            
            features = list(top_features.keys())
            importances = list(top_features.values())
            
            # Crear gráfico horizontal
            y_pos = np.arange(len(features))
            bars = ax.barh(y_pos, importances, color='lightcoral', alpha=0.8)
            
            ax.set_yticks(y_pos)
            ax.set_yticklabels([f.replace('_', ' ').title()[:20] for f in features], fontsize=8)
            ax.set_xlabel('Importancia')
            ax.set_title('Top 15 Features Más Importantes', fontweight='bold')
            
            # Agregar valores en las barras
            for i, (bar, val) in enumerate(zip(bars, importances)):
                ax.text(bar.get_width() + max(importances) * 0.01, bar.get_y() + bar.get_height()/2, 
                       f'{val:.3f}', va='center', fontsize=7)
            
            ax.grid(axis='x', alpha=0.3)
            ax.invert_yaxis()
            
        except Exception as e:
            ax.text(0.5, 0.5, f'Error cargando\nfeature importance:\n{str(e)}', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Feature Importance', fontweight='bold')
    
    def _plot_class_distribution_compact(self, ax):
        """Distribución compacta de clases."""
        win_counts = self.df['is_win'].value_counts()
        
        # Gráfico de barras
        bars = ax.bar(['Derrotas', 'Victorias'], [win_counts[0], win_counts[1]], 
                     color=['red', 'green'], alpha=0.7)
        
        # Agregar valores en las barras
        for bar, count in zip(bars, win_counts.values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 10,
                   f'{count}\n({count/len(self.df):.1%})',
                   ha='center', va='bottom', fontsize=9)
        
        ax.set_ylabel('Cantidad de Partidos')
        ax.set_title('Distribución de Victorias/Derrotas', fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
    
    def _plot_confusion_matrix_compact(self, ax):
        """Matriz de confusión compacta."""
        if self.predictions is None:
            ax.text(0.5, 0.5, 'Predicciones\nno disponibles', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Matriz de Confusión', fontweight='bold')
            return
        
        # Calcular matriz de confusión
        cm = confusion_matrix(self.df['is_win'], self.predictions)
        
        # Crear heatmap
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                   xticklabels=['Derrota', 'Victoria'],
                   yticklabels=['Derrota', 'Victoria'])
        
        ax.set_xlabel('Predicción')
        ax.set_ylabel('Real')
        ax.set_title('Matriz de Confusión', fontweight='bold')
    
    def _plot_roc_curve_compact(self, ax):
        """Curva ROC compacta."""
        if self.probabilities is None:
            ax.text(0.5, 0.5, 'Probabilidades\nno disponibles', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Curva ROC', fontweight='bold')
            return
        
        from sklearn.metrics import roc_curve, auc
        
        # Calcular curva ROC
        fpr, tpr, _ = roc_curve(self.df['is_win'], self.probabilities[:, 1])
        roc_auc = auc(fpr, tpr)
        
        # Plotear curva ROC
        ax.plot(fpr, tpr, color='darkorange', lw=2, 
               label=f'ROC curve (AUC = {roc_auc:.3f})')
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', alpha=0.5)
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('Tasa de Falsos Positivos')
        ax.set_ylabel('Tasa de Verdaderos Positivos')
        ax.set_title('Curva ROC', fontweight='bold')
        ax.legend(loc="lower right")
        ax.grid(alpha=0.3)
    
    def _plot_probability_distribution_compact(self, ax):
        """Distribución de probabilidades predichas."""
        if self.probabilities is None:
            ax.text(0.5, 0.5, 'Probabilidades\nno disponibles', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Distribución de Probabilidades', fontweight='bold')
            return
        
        # Separar por clase real
        win_probs = self.probabilities[self.df['is_win'] == 1, 1]
        loss_probs = self.probabilities[self.df['is_win'] == 0, 1]
        
        # Histogramas
        ax.hist(loss_probs, bins=30, alpha=0.5, label='Derrotas Reales', color='red')
        ax.hist(win_probs, bins=30, alpha=0.5, label='Victorias Reales', color='green')
        
        ax.set_xlabel('Probabilidad de Victoria Predicha')
        ax.set_ylabel('Frecuencia')
        ax.set_title('Distribución de Probabilidades por Clase Real', fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)
    
    def _plot_confidence_analysis_compact(self, ax):
        """Análisis de confianza de predicciones."""
        if self.probabilities is None:
            ax.text(0.5, 0.5, 'Probabilidades\nno disponibles', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Análisis de Confianza', fontweight='bold')
            return
        
        # Definir rangos de confianza
        confidence_ranges = [
            (0.5, 0.6, 'Baja'),
            (0.6, 0.7, 'Media'),
            (0.7, 0.8, 'Alta'),
            (0.8, 0.9, 'Muy Alta'),
            (0.9, 1.0, 'Extrema')
        ]
        
        range_names = []
        accuracies = []
        counts = []
        
        for min_conf, max_conf, name in confidence_ranges:
            # Filtrar predicciones en este rango de confianza
            max_probs = np.max(self.probabilities, axis=1)
            mask = (max_probs >= min_conf) & (max_probs < max_conf)
            
            if np.sum(mask) > 0:
                range_accuracy = accuracy_score(self.df['is_win'][mask], self.predictions[mask])
                range_names.append(name)
                accuracies.append(range_accuracy)
                counts.append(np.sum(mask))
        
        if range_names:
            bars = ax.bar(range_names, accuracies, color='lightblue', alpha=0.8)
            
            # Agregar valores en las barras
            for bar, acc, count in zip(bars, accuracies, counts):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                       f'{acc:.3f}\n(n={count})', ha='center', va='bottom', fontsize=8)
            
            ax.set_ylabel('Accuracy')
            ax.set_title('Accuracy por Rango de Confianza', fontweight='bold')
            ax.set_ylim(0, 1)
            ax.grid(axis='y', alpha=0.3)
    
    def _plot_team_performance_compact(self, ax):
        """Análisis de rendimiento por equipo."""
        # Calcular win rate por equipo
        team_performance = self.df.groupby('Team').agg({
            'is_win': ['mean', 'count']
        }).reset_index()
        
        team_performance.columns = ['Team', 'WinRate', 'Games']
        
        # Filtrar equipos con al menos 20 juegos
        team_performance = team_performance[team_performance['Games'] >= 20]
        
        # Top 10 y bottom 10 equipos
        top_teams = team_performance.nlargest(10, 'WinRate')
        
        if len(top_teams) > 0:
            teams = [t[:3] for t in top_teams['Team']]  # Abreviar nombres
            win_rates = top_teams['WinRate']
            
            bars = ax.bar(teams, win_rates, color='lightsteelblue', alpha=0.8)
            
            # Agregar valores en las barras
            for bar, wr in zip(bars, win_rates):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                       f'{wr:.2f}', ha='center', va='bottom', fontsize=8)
            
            ax.set_ylabel('Win Rate')
            ax.set_title('Top 10 Equipos por Win Rate', fontweight='bold')
            ax.set_ylim(0, 1)
            plt.setp(ax.get_xticklabels(), rotation=45)
            ax.grid(axis='y', alpha=0.3)
    
    def _plot_temporal_analysis_compact(self, ax):
        """Análisis temporal de victorias."""
        # Agrupar por mes
        df_copy = self.df.copy()
        df_copy['month'] = pd.to_datetime(df_copy['Date']).dt.to_period('M')
        
        monthly_stats = df_copy.groupby('month').agg({
            'is_win': 'mean'
        }).reset_index()
        
        if len(monthly_stats) > 0:
            months = [str(m) for m in monthly_stats['month']]
            win_rates = monthly_stats['is_win']
            
            ax.plot(months, win_rates, marker='o', linewidth=2, markersize=4)
            
            ax.set_ylabel('Win Rate')
            ax.set_title('Win Rate por Mes', fontweight='bold')
            ax.set_ylim(0, 1)
            plt.setp(ax.get_xticklabels(), rotation=45)
            ax.grid(alpha=0.3)
    
    def _plot_cv_results_compact(self, ax):
        """Gráfico compacto de resultados de validación cruzada."""
        try:
            # Intentar obtener resultados de CV del modelo
            cv_results = getattr(self.model, '_cv_results', None)
            
            if cv_results is None:
                ax.text(0.5, 0.5, 'Resultados CV\nno disponibles', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title('Validación Cruzada', fontweight='bold')
                return
            
            # Extraer métricas de CV
            if isinstance(cv_results, dict) and 'cv_scores' in cv_results:
                fold_scores = cv_results['cv_scores']
                accuracy_scores = [fold.get('accuracy', 0) for fold in fold_scores]
                
                folds = range(1, len(accuracy_scores) + 1)
                
                bars = ax.bar(folds, accuracy_scores, alpha=0.8, color='purple')
                
                # Agregar valores en las barras
                for bar, acc in zip(bars, accuracy_scores):
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                           f'{acc:.3f}', ha='center', va='bottom', fontsize=8)
                
                ax.set_xlabel('Fold')
                ax.set_ylabel('Accuracy')
                ax.set_title('Validación Cruzada por Fold', fontweight='bold')
                ax.set_xticks(folds)
                ax.set_xticklabels([f'Fold {i}' for i in folds])
                
                # Agregar promedio
                avg_acc = np.mean(accuracy_scores)
                ax.axhline(y=avg_acc, color='red', linestyle='--', 
                          label=f'Promedio: {avg_acc:.3f}')
                ax.legend()
                ax.grid(axis='y', alpha=0.3)
            else:
                ax.text(0.5, 0.5, 'Formato CV\nno compatible', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title('Validación Cruzada', fontweight='bold')
                
        except Exception as e:
            ax.text(0.5, 0.5, f'Error en CV:\n{str(e)}', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Validación Cruzada', fontweight='bold')
    
    def save_results(self):
        """Guarda todos los resultados del entrenamiento."""
        logger.info("Guardando resultados...")
        
        # Asegurar que el directorio de salida existe
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Guardar modelo
        model_path = os.path.normpath(os.path.join(self.output_dir, 'is_win_model.pkl'))
        self.model.save_model(model_path)
        
        # Guardar reporte completo
        report = {
            'model_type': 'NBA Is Win Ensemble',
            'training_results': self.training_results,
            'model_summary': self.model.get_training_summary() if hasattr(self.model, 'get_training_summary') else {},
            'timestamp': datetime.now().isoformat()
        }
        report_path = os.path.normpath(os.path.join(self.output_dir, 'training_report.json'))
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Guardar predicciones
        if self.predictions is not None and self.probabilities is not None:
            predictions_df = self.df[['Team', 'Date', 'Opp', 'is_win']].copy()
            predictions_df['win_predicted'] = self.predictions
            predictions_df['win_probability'] = self.probabilities[:, 1]
            predictions_df['correct_prediction'] = (predictions_df['is_win'] == predictions_df['win_predicted']).astype(int)
            
            predictions_path = os.path.normpath(os.path.join(self.output_dir, 'predictions.csv'))
            predictions_df.to_csv(predictions_path, index=False)
        
        # Guardar feature importance
        try:
            importance_dict = self.model.get_feature_importance(top_n=50)
            if importance_dict and 'feature_importance' in importance_dict:
                importance_df = pd.DataFrame([
                    {'feature': k, 'importance': v} 
                    for k, v in importance_dict['feature_importance'].items()
                ]).sort_values('importance', ascending=False)
                
                importance_path = os.path.normpath(os.path.join(self.output_dir, 'feature_importance.csv'))
                importance_df.to_csv(importance_path, index=False)
        except Exception as e:
            logger.warning(f"No se pudo guardar feature importance: {e}")
        
        # Crear resumen de archivos generados
        files_summary = {
            'model_file': 'is_win_model.pkl',
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
        logger.info("Iniciando pipeline de entrenamiento Is Win...")
        
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
    trainer = IsWinTrainer(
        game_data_path=game_data_path,
        biometrics_path=biometrics_path,
        teams_path=teams_path,
        output_dir="results/is_win_model",
        n_trials=20,  # Reducido para pruebas más rápidas
        cv_folds=5
    )
    
    # Ejecutar pipeline completo
    results = trainer.run_complete_training()
    
    print("\n" + "="*80)
    print("RESUMEN FINAL DE ENTRENAMIENTO IS WIN")
    print("="*80)
    
    # Mostrar información del modelo
    print(f"\n📊 MODELO IS WIN (Ensemble de Clasificación):")
    if 'accuracy' in results:
        print(f"   Accuracy: {results['accuracy']:.4f}")
    if 'precision' in results:
        print(f"   Precision: {results['precision']:.4f}")
    if 'recall' in results:
        print(f"   Recall: {results['recall']:.4f}")
    if 'f1_score' in results:
        print(f"   F1-Score: {results['f1_score']:.4f}")
    if 'auc_roc' in results:
        print(f"   AUC-ROC: {results['auc_roc']:.4f}")
    
    # Mostrar información adicional
    print(f"\n📋 INFORMACIÓN ADICIONAL:")
    print(f"   Modelos Base: XGBoost, LightGBM, Random Forest")
    print(f"   Neural Network, Logistic Regression, SVM")
    print(f"   Meta-learner: Ensemble optimizado")
    print(f"   Validación: Cruzada estratificada (5 folds)")
    print(f"   Optimización: Bayesiana con GPU support")
    
    print("="*80)
    print("Entrenamiento Is Win completado exitosamente!")
    print("="*80)


if __name__ == "__main__":
    main() 