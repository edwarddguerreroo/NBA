"""
Trainer Completo para Modelo XGBoost PTS
========================================

Trainer que integra carga de datos, entrenamiento del modelo XGBoost
y generación completa de métricas y visualizaciones para predicción de puntos NBA.

Características:
- Integración completa con data loader
- Entrenamiento automatizado con optimización bayesiana
- Generación de dashboard PNG unificado con todas las métricas
- Métricas detalladas y reportes completos
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
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Imports del proyecto
from src.preprocessing.data_loader import NBADataLoader
from src.models.players.pts.model_pts import XGBoostPTSModel

# Import del sistema de logging unificado
from config.logging_config import configure_trainer_logging, NBALogger

warnings.filterwarnings('ignore')
logger = configure_trainer_logging('pts')

# Configurar estilo de visualizaciones optimizado para PNG
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10


class XGBoostPTSTrainer:
    """
    Trainer completo para modelo XGBoost de predicción de puntos NBA.
    
    Integra carga de datos, entrenamiento, evaluación y visualizaciones.
    """
    
    def __init__(self,
                 game_data_path: str,
                 biometrics_path: str,
                 teams_path: str,
                 output_dir: str = "results/pts_model",
                 n_trials: int = 100,
                 cv_folds: int = 5,
                 random_state: int = 42):
        """
        Inicializa el trainer completo.
        
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
            self.output_dir = os.path.normpath("results_pts_model")
            os.makedirs(self.output_dir, exist_ok=True)
            logger.info(f"Usando directorio alternativo: {self.output_dir}")
        
        # Componentes principales
        self.data_loader = NBADataLoader(
            game_data_path, biometrics_path, teams_path
        )
        self.model = XGBoostPTSModel(
            n_trials=n_trials,
            cv_folds=cv_folds,
            random_state=random_state
        )
        
        # Datos y resultados
        self.df = None
        self.teams_df = None
        self.training_results = None
        self.predictions = None
        
        logger.info(f"Trainer XGBoost PTS inicializado | Output: {self.output_dir}")
    
    def load_and_prepare_data(self) -> pd.DataFrame:
        """
        Carga y prepara todos los datos necesarios.
        
        Returns:
            pd.DataFrame: Datos preparados para entrenamiento
        """
        logger.info("Cargando datos NBA")
        
        # Cargar datos usando el data loader
        self.df, self.teams_df = self.data_loader.load_data()
        
        # Estadísticas básicas de los datos
        logger.info(f"Datos cargados: {len(self.df)} registros de jugadores")
        logger.info(f"Datos de equipos: {len(self.teams_df)} registros")
        logger.info(f"Jugadores únicos: {self.df['Player'].nunique()}")
        logger.info(f"Equipos únicos: {self.df['Team'].nunique()}")
        logger.info(f"Rango de fechas: {self.df['Date'].min()} a {self.df['Date'].max()}")
        
        # Verificar target
        if 'PTS' not in self.df.columns:
            raise ValueError("Columna 'PTS' no encontrada en los datos")
        
        # Estadísticas del target
        pts_stats = self.df['PTS'].describe()
        logger.info(f"Estadísticas PTS - Media: {pts_stats['mean']:.2f}, "
                   f"Mediana: {pts_stats['50%']:.2f}, "
                   f"Max: {pts_stats['max']:.0f}")
        
        return self.df
    
    def train_model(self) -> Dict:
        """
        Entrena el modelo completo con optimización y validación.
        
        Returns:
            Dict: Resultados del entrenamiento
        """
        logger.info("Iniciando entrenamiento del modelo XGBoost PTS")
        
        if self.df is None:
            raise ValueError("Datos no cargados. Ejecutar load_and_prepare_data() primero")
        
        # Entrenar modelo
        start_time = datetime.now()
        self.training_results = self.model.train(self.df)
        training_duration = (datetime.now() - start_time).total_seconds()
        
        logger.info(f"Entrenamiento completado | Duración: {training_duration:.1f} segundos")
        
        # Generar predicciones en conjunto de prueba
        self.predictions = self.model.predict(self.df)
        
        # Compilar resultados completos
        results = {
            'training_metrics': self.model.training_metrics,
            'validation_metrics': self.model.validation_metrics,
            'cv_scores': self.model.cv_scores,
            'feature_importance': self.model.get_feature_importance(20),
            'best_params': self.model.best_params,
            'training_duration_seconds': training_duration,
            'model_info': {
                'n_features': len(self.model.selected_features),
                'n_samples': len(self.df),
                'target_mean': self.df['PTS'].mean(),
                'target_std': self.df['PTS'].std()
            }
        }
        
        return results
    
    def generate_all_visualizations(self):
        """
        Genera una visualización completa en PNG con todas las métricas principales.
        """
        logger.info("Generando visualización completa en PNG")
        
        # Asegurar que el directorio de salida existe
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Crear figura principal con subplots organizados
        fig = plt.figure(figsize=(24, 16))
        fig.suptitle('Dashboard Completo - Modelo NBA PTS Prediction', fontsize=20, fontweight='bold', y=0.98)
        
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
        
        # 4. Predicciones vs Reales (segunda fila, izquierda)
        ax4 = fig.add_subplot(gs[1, 0:2])
        self._plot_predictions_vs_actual_compact(ax4)
        
        # 5. Residuos (segunda fila, derecha)
        ax5 = fig.add_subplot(gs[1, 2:4])
        self._plot_residuals_compact(ax5)
        
        # 6. Validación cruzada (tercera fila, izquierda)
        ax6 = fig.add_subplot(gs[2, 0:2])
        self._plot_cv_results_compact(ax6)
        
        # 7. Análisis por rangos de puntos (tercera fila, derecha)
        ax7 = fig.add_subplot(gs[2, 2:4])
        self._plot_points_range_analysis_compact(ax7)
        
        # 8. Análisis temporal (cuarta fila, izquierda)
        ax8 = fig.add_subplot(gs[3, 0:2])
        self._plot_temporal_analysis_compact(ax8)
        
        # 9. Top jugadores predicciones (cuarta fila, derecha)
        ax9 = fig.add_subplot(gs[3, 2:4])
        self._plot_top_players_analysis_compact(ax9)
        
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
        val_metrics = self.training_results.get('validation_metrics', {})
        cv_metrics = self.training_results.get('cv_scores', {})
        
        mae = val_metrics.get('mae', 0)
        rmse = val_metrics.get('rmse', 0)
        r2 = val_metrics.get('r2', 0)
        accuracy_3pts = val_metrics.get('accuracy_3pts', 0)
        
        cv_mae_mean = cv_metrics.get('mae_mean', 0)
        cv_r2_mean = cv_metrics.get('r2_mean', 0)
        
        # Crear texto con métricas
        metrics_text = f"""
MÉTRICAS DEL MODELO

Validación:
• MAE: {mae:.3f}
• RMSE: {rmse:.3f}
• R²: {r2:.3f}
• Accuracy ±3pts: {accuracy_3pts:.1f}%

Validación Cruzada:
• MAE CV: {cv_mae_mean:.3f}
• R² CV: {cv_r2_mean:.3f}

Estado: PRODUCCIÓN
        """
        
        ax.text(0.05, 0.95, metrics_text, transform=ax.transAxes, fontsize=11,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
        
        ax.set_title('Resumen de Rendimiento', fontsize=14, fontweight='bold')
    
    def _plot_feature_importance_compact(self, ax):
        """Feature importance compacta."""
        if not hasattr(self.model, 'get_feature_importance'):
            ax.text(0.5, 0.5, 'Feature importance no disponible', 
                   ha='center', va='center', transform=ax.transAxes)
            return
        
        # Obtener top 10 features
        top_features = self.model.get_feature_importance(10)
        features = list(top_features.keys())
        importance = list(top_features.values())
        
        # Gráfico horizontal
        y_pos = np.arange(len(features))
        bars = ax.barh(y_pos, importance, color='steelblue', alpha=0.7)
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(features, fontsize=9)
        ax.set_xlabel('Importancia', fontsize=10)
        ax.set_title('Top 10 Características Más Importantes', fontsize=12, fontweight='bold')
        
        # Agregar valores en las barras
        for i, (bar, val) in enumerate(zip(bars, importance)):
            ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                   f'{val:.3f}', va='center', fontsize=8)
        
        ax.grid(axis='x', alpha=0.3)
    
    def _plot_target_distribution_compact(self, ax):
        """Distribución del target compacta."""
        # Histograma de puntos
        ax.hist(self.df['PTS'], bins=30, alpha=0.7, color='orange', edgecolor='black')
        
        # Líneas de estadísticas
        mean_pts = self.df['PTS'].mean()
        median_pts = self.df['PTS'].median()
        
        ax.axvline(mean_pts, color='red', linestyle='--', linewidth=2, 
                  label=f'Media: {mean_pts:.1f}')
        ax.axvline(median_pts, color='green', linestyle='--', linewidth=2,
                  label=f'Mediana: {median_pts:.1f}')
        
        ax.set_xlabel('Puntos', fontsize=10)
        ax.set_ylabel('Frecuencia', fontsize=10)
        ax.set_title('Distribución de Puntos', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
    
    def _plot_predictions_vs_actual_compact(self, ax):
        """Análisis de predicciones vs valores reales compacto."""
        if self.predictions is None:
            ax.text(0.5, 0.5, 'Predicciones no disponibles', 
                   ha='center', va='center', transform=ax.transAxes)
            return
        
        y_true = self.df['PTS'].values
        y_pred = self.predictions
        
        # Scatter plot predicciones vs reales
        ax.scatter(y_true, y_pred, alpha=0.5, s=10)
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
        ax.set_xlabel('Puntos Reales', fontsize=10)
        ax.set_ylabel('Puntos Predichos', fontsize=10)
        ax.set_title('Predicciones vs Valores Reales', fontsize=12, fontweight='bold')
        
        # Calcular R²
        r2 = r2_score(y_true, y_pred)
        ax.text(0.05, 0.95, f'R² = {r2:.3f}', transform=ax.transAxes, 
               fontsize=11, bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        ax.grid(alpha=0.3)
    
    def _plot_residuals_compact(self, ax):
        """Análisis de residuos compacto."""
        if self.predictions is None:
            ax.text(0.5, 0.5, 'Residuos no disponibles', 
                   ha='center', va='center', transform=ax.transAxes)
            return
        
        y_true = self.df['PTS'].values
        y_pred = self.predictions
        residuals = y_pred - y_true
        
        # Histograma de residuos
        ax.hist(residuals, bins=30, alpha=0.7, color='purple', edgecolor='black')
        ax.axvline(0, color='red', linestyle='--', linewidth=2)
        ax.axvline(residuals.mean(), color='orange', linestyle='--', linewidth=2,
                  label=f'Media: {residuals.mean():.3f}')
        
        ax.set_xlabel('Residuos (Predicho - Real)', fontsize=10)
        ax.set_ylabel('Frecuencia', fontsize=10)
        ax.set_title('Distribución de Residuos', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
    
    def _plot_cv_results_compact(self, ax):
        """Resultados de validación cruzada compactos."""
        if not hasattr(self.model, 'cv_scores') or self.model.cv_scores is None:
            ax.text(0.5, 0.5, 'Resultados de validación cruzada no disponibles', 
                   ha='center', va='center', transform=ax.transAxes)
            return
        
        cv_scores = self.model.cv_scores
        
        # Preparar datos para gráfico
        metrics = []
        means = []
        stds = []
        
        if 'mae_mean' in cv_scores:
            metrics.append('MAE')
            means.append(cv_scores['mae_mean'])
            stds.append(cv_scores.get('mae_std', 0))
        
        if 'rmse_mean' in cv_scores:
            metrics.append('RMSE')
            means.append(cv_scores['rmse_mean'])
            stds.append(cv_scores.get('rmse_std', 0))
        
        if 'r2_mean' in cv_scores:
            metrics.append('R²')
            means.append(cv_scores['r2_mean'])
            stds.append(cv_scores.get('r2_std', 0))
        
        if metrics:
            x_pos = np.arange(len(metrics))
            bars = ax.bar(x_pos, means, yerr=stds, capsize=5, 
                         color=['salmon', 'lightblue', 'lightgreen'][:len(metrics)])
            ax.set_xticks(x_pos)
            ax.set_xticklabels(metrics)
            ax.set_title('Validación Cruzada (Media ± Std)', fontsize=12, fontweight='bold')
            ax.set_ylabel('Valor', fontsize=10)
            
            # Agregar valores en las barras
            for bar, mean_val in zip(bars, means):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                       f'{mean_val:.3f}', ha='center', va='bottom', fontsize=9)
        
        ax.grid(axis='y', alpha=0.3)
    
    def _plot_points_range_analysis_compact(self, ax):
        """Análisis de predicciones por rango de puntos compacto."""
        if self.predictions is None:
            ax.text(0.5, 0.5, 'Predicciones no disponibles', 
                   ha='center', va='center', transform=ax.transAxes)
            return
        
        y_true = self.df['PTS'].values
        y_pred = self.predictions
        
        # Definir rangos de puntos
        ranges = [(0, 10), (10, 15), (15, 20), (20, 25), (25, 30), (30, 50)]
        range_labels = ['0-10', '10-15', '15-20', '20-25', '25-30', '30+']
        
        accuracies_1pt = []
        accuracies_2pts = []
        sample_counts = []
        
        for start, end in ranges:
            if end == 50:  # Para el último rango (30+)
                mask = y_true >= start
            else:
                mask = (y_true >= start) & (y_true < end)
            
            if mask.sum() > 0:
                range_y_true = y_true[mask]
                range_y_pred = y_pred[mask]
                
                acc_1pt = np.mean(np.abs(range_y_pred - range_y_true) <= 1) * 100
                acc_2pts = np.mean(np.abs(range_y_pred - range_y_true) <= 2) * 100
                
                accuracies_1pt.append(acc_1pt)
                accuracies_2pts.append(acc_2pts)
                sample_counts.append(mask.sum())
            else:
                accuracies_1pt.append(0)
                accuracies_2pts.append(0)
                sample_counts.append(0)
        
        # Gráfico de barras
        x_pos = np.arange(len(range_labels))
        width = 0.35
        
        bars1 = ax.bar(x_pos - width/2, accuracies_1pt, width, label='±1 punto', color='lightblue', alpha=0.7)
        bars2 = ax.bar(x_pos + width/2, accuracies_2pts, width, label='±2 puntos', color='lightgreen', alpha=0.7)
        
        ax.set_xticks(x_pos)
        ax.set_xticklabels(range_labels)
        ax.set_xlabel('Rango de Puntos', fontsize=10)
        ax.set_ylabel('Accuracy (%)', fontsize=10)
        ax.set_title('Accuracy por Rango de Puntos', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        
        # Agregar conteos de muestras
        for i, count in enumerate(sample_counts):
            if count > 0:
                ax.text(i, max(accuracies_1pt[i], accuracies_2pts[i]) + 5, 
                       f'n={count}', ha='center', va='bottom', fontsize=8)
        
        ax.grid(axis='y', alpha=0.3)
    
    def _plot_temporal_analysis_compact(self, ax):
        """Análisis temporal compacto."""
        if self.df is None:
            ax.text(0.5, 0.5, 'Datos no cargados', 
                   ha='center', va='center', transform=ax.transAxes)
            return
        
        # Promedio de puntos por mes
        monthly_pts = self.df.groupby(self.df['Date'].dt.to_period('M'))['PTS'].mean()
        
        # Limitar a últimos 12 meses para mejor visualización
        if len(monthly_pts) > 12:
            monthly_pts = monthly_pts.tail(12)
        
        ax.plot(range(len(monthly_pts)), monthly_pts.values, 
               marker='o', linewidth=2, color='purple', markersize=6)
        
        # Configurar etiquetas del eje x
        ax.set_xticks(range(len(monthly_pts)))
        ax.set_xticklabels([str(period)[-7:] for period in monthly_pts.index], 
                          rotation=45, fontsize=8)
        
        ax.set_title('Tendencia Temporal de Puntos', fontsize=12, fontweight='bold')
        ax.set_xlabel('Mes', fontsize=10)
        ax.set_ylabel('Puntos Promedio', fontsize=10)
        
        # Agregar línea de tendencia
        if len(monthly_pts) > 2:
            z = np.polyfit(range(len(monthly_pts)), monthly_pts.values, 1)
            p = np.poly1d(z)
            ax.plot(range(len(monthly_pts)), p(range(len(monthly_pts))), 
                   "r--", alpha=0.8, linewidth=1, label=f'Tendencia')
            ax.legend(fontsize=9)
        
        ax.grid(alpha=0.3)
    
    def _plot_top_players_analysis_compact(self, ax):
        """Análisis de top jugadores compacto."""
        if self.df is None:
            ax.text(0.5, 0.5, 'Datos no cargados', 
                   ha='center', va='center', transform=ax.transAxes)
            return
        
        # Top 10 jugadores por puntos promedio (mínimo 10 partidos)
        player_stats = self.df.groupby('Player')['PTS'].agg(['mean', 'count']).reset_index()
        player_stats = player_stats[player_stats['count'] >= 10]
        top_scorers = player_stats.nlargest(10, 'mean')
        
        if len(top_scorers) == 0:
            ax.text(0.5, 0.5, 'No hay suficientes datos de jugadores', 
                   ha='center', va='center', transform=ax.transAxes)
            return
        
        # Gráfico horizontal
        y_pos = np.arange(len(top_scorers))
        bars = ax.barh(y_pos, top_scorers['mean'].values, color='gold', alpha=0.7)
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels([name[:15] + '' if len(name) > 15 else name 
                           for name in top_scorers['Player'].values], fontsize=8)
        ax.set_xlabel('Puntos Promedio', fontsize=10)
        ax.set_title('Top 10 Anotadores (≥10 partidos)', fontsize=12, fontweight='bold')
        
        # Agregar valores en las barras
        for i, (bar, val) in enumerate(zip(bars, top_scorers['mean'].values)):
            ax.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height()/2, 
                   f'{val:.1f}', va='center', fontsize=8)
        
        ax.grid(axis='x', alpha=0.3)
    
    def save_results(self):
        """Guarda todos los resultados del entrenamiento."""
        logger.info("Guardando resultados completos")
        
        # Asegurar que el directorio de salida existe
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Guardar modelo en trained_models/
        model_path = os.path.normpath(os.path.join('trained_models', 'xgboost_pts_model.joblib'))
        os.makedirs('trained_models', exist_ok=True)
        self.model.save_model(model_path)
        
        # Guardar reporte completo
        report = self.model.generate_report()
        report_path = os.path.normpath(os.path.join(self.output_dir, 'training_report.json'))
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Guardar predicciones
        if self.predictions is not None:
            predictions_df = self.df[['Player', 'Date', 'Team', 'PTS']].copy()
            predictions_df['PTS_predicted'] = self.predictions
            predictions_df['error'] = self.predictions - self.df['PTS']
            predictions_df['abs_error'] = np.abs(predictions_df['error'])
            
            predictions_path = os.path.normpath(os.path.join(self.output_dir, 'predictions.csv'))
            predictions_df.to_csv(predictions_path, index=False)
        
        # Guardar feature importance
        if hasattr(self.model, 'get_feature_importance'):
            try:
                feature_importance = self.model.get_feature_importance(20)
                importance_df = pd.DataFrame([
                {'feature': k, 'importance': v} 
                    for k, v in feature_importance.items()
            ]).sort_values('importance', ascending=False)
            
                importance_path = os.path.normpath(os.path.join(self.output_dir, 'feature_importance.csv'))
                importance_df.to_csv(importance_path, index=False)
            except Exception as e:
                logger.warning(f"No se pudo guardar feature importance: {e}")
        
        # Crear resumen de archivos generados
        files_summary = {
            'model_file': 'trained_models/xgboost_pts_model.joblib',
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
        logger.info("Iniciando pipeline completo de entrenamiento XGBoost PTS")
        
        try:
            # 1. Cargar datos
            self.load_and_prepare_data()
            
            # 2. Entrenar modelo
            results = self.train_model()
            
            # 3. Generar visualizaciones
            self.generate_all_visualizations()
            
            # 4. Guardar resultados
            self.save_results()
            
            logger.info("Pipeline completo ejecutado exitosamente!")
            return results
            
        except Exception as e:
            logger.error(f"Error en pipeline de entrenamiento: {str(e)}")
            raise


def main():
    """
    Función principal para ejecutar el entrenamiento completo de PTS.
    """
    # Configurar logging ultra-silencioso
    logging.basicConfig(
        level=logging.ERROR,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Solo mensajes críticos del trainer principal
    main_logger = logging.getLogger(__name__)
    main_logger.setLevel(logging.WARNING)
    
    # Silenciar librerías externas
    logging.getLogger('sklearn').setLevel(logging.ERROR)
    logging.getLogger('xgboost').setLevel(logging.ERROR)
    logging.getLogger('lightgbm').setLevel(logging.ERROR)
    logging.getLogger('catboost').setLevel(logging.ERROR)
    logging.getLogger('optuna').setLevel(logging.ERROR)
    
    # Rutas de datos (ajustar según tu configuración)
    game_data_path = "data/players.csv"
    biometrics_path = "data/height.csv"
    teams_path = "data/teams.csv"
    
    # Crear y ejecutar trainer
    trainer = XGBoostPTSTrainer(
        game_data_path=game_data_path,
        biometrics_path=biometrics_path,
        teams_path=teams_path,
        output_dir="results/pts_model",
        n_trials=20,
        cv_folds=5,
        random_state=42
    )
    
    # Ejecutar pipeline completo
    results = trainer.run_complete_training()
    
    print("Entrenamiento PTS Model completado!")
    print(f"Resultados: {trainer.output_dir}")
    
    return results


if __name__ == "__main__":
    main() 