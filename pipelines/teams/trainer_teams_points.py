"""
Trainer Completo para Modelo NBA Teams Points
=============================================

Trainer que integra carga de datos, entrenamiento del modelo de puntos por equipo
y generación completa de métricas y visualizaciones para predicción de puntos NBA.

Características:
- Integración completa con data loader
- Entrenamiento automatizado con optimización bayesiana
- Generación de dashboard PNG unificado con todas las métricas
- Métricas detalladas específicas para predicción de puntos por equipo
- Análisis de feature importance
- Validación cruzada temporal
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
from src.models.teams.teams_points.model_teams_points import TeamPointsModel

# Import del sistema de logging unificado
from config.logging_config import configure_trainer_logging, NBALogger

warnings.filterwarnings('ignore')
logger = configure_trainer_logging('teams_points')

# Configurar estilo de visualizaciones optimizado para PNG
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10


class TeamsPointsTrainer:
    """
    Trainer completo para modelo de predicción de puntos por equipo NBA.
    
    Integra carga de datos, entrenamiento, evaluación y visualizaciones.
    """
    
    def __init__(self,
                 game_data_path: str,
                 biometrics_path: str,
                 teams_path: str,
                 output_dir: str = "results/teams_points_model",
                 n_trials: int = 50,  # Trials para optimización bayesiana
                 cv_folds: int = 5,
                 random_state: int = 42):
        """
        Inicializa el trainer completo para predicción de puntos por equipo.
        
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
            self.output_dir = os.path.normpath("results_teams_points_model")
            os.makedirs(self.output_dir, exist_ok=True)
            logger.info(f"Usando directorio alternativo: {self.output_dir}")
        
        # Componentes principales
        self.data_loader = NBADataLoader(
            game_data_path, biometrics_path, teams_path
        )
        self.model = TeamPointsModel(
            optimize_hyperparams=True
        )
        
        # Datos y resultados
        self.df = None
        self.teams_df = None
        self.training_results = None
        self.predictions = None
        
        logger.info(f"Trainer Teams Points inicializado | Output: {self.output_dir}")
    
    def load_and_prepare_data(self) -> pd.DataFrame:
        """
        Carga y prepara todos los datos necesarios.
        
        Returns:
            pd.DataFrame: Datos preparados para entrenamiento
        """
        logger.info("Cargando datos NBA")
        
        # Cargar datos usando el data loader
        self.df, self.teams_df = self.data_loader.load_data()
        
        # Usar datos de equipos para predicción de puntos
        teams_data = self.teams_df.copy()
        
        # Verificar que existe la columna target
        if 'PTS' not in teams_data.columns:
            raise ValueError("Columna 'PTS' no encontrada en datos de equipos")
        
        # Estadísticas básicas de los datos
        logger.info(f"Datos cargados: {len(teams_data)} registros de equipos")
        logger.info(f"Equipos únicos: {teams_data['Team'].nunique()}")
        logger.info(f"Rango de fechas: {teams_data['Date'].min()} a {teams_data['Date'].max()}")
        
        # Estadísticas del target
        pts_stats = teams_data['PTS'].describe()
        logger.info(f"Estadísticas PTS por equipo:")
        logger.info(f"  | Media: {pts_stats['mean']:.1f}")
        logger.info(f"  | Mediana: {pts_stats['50%']:.1f}")
        logger.info(f"  | Min/Max: {pts_stats['min']:.0f}/{pts_stats['max']:.0f}")
        logger.info(f"  | Desv. Estándar: {pts_stats['std']:.1f}")
        
        self.df = teams_data
        return self.df
    
    def train_model(self) -> Dict:
        """
        Entrena el modelo completo con optimización y validación.
        
        Returns:
            Dict: Resultados del entrenamiento
        """
        logger.info("Iniciando entrenamiento del modelo Teams Points")
        
        if self.df is None:
            raise ValueError("Datos no cargados. Ejecutar load_and_prepare_data() primero")
        
        # Entrenar modelo
        start_time = datetime.now()
        logger.info("Entrenando modelo Teams Points con ensemble optimizado")
        self.training_results = self.model.train(self.df, validation_split=0.2)
        training_duration = (datetime.now() - start_time).total_seconds()
        
        logger.info(f"Modelo Teams Points completado | Duración: {training_duration:.1f} segundos")
        
        # Mostrar resultados del entrenamiento
        logger.info("=" * 50)
        logger.info("RESULTADOS DEL ENTRENAMIENTO TEAMS POINTS")
        logger.info("=" * 50)
        
        if 'mae' in self.training_results:
            logger.info(f"MAE: {self.training_results['mae']:.3f}")
        if 'rmse' in self.training_results:
            logger.info(f"RMSE: {self.training_results['rmse']:.3f}")
        if 'r2' in self.training_results:
            logger.info(f"R²: {self.training_results['r2']:.3f}")
        
        # Métricas específicas para puntos
        if 'accuracy_5pts' in self.training_results:
            logger.info(f"Accuracy ±5pts: {self.training_results['accuracy_5pts']:.1f}%")
        if 'accuracy_10pts' in self.training_results:
            logger.info(f"Accuracy ±10pts: {self.training_results['accuracy_10pts']:.1f}%")
        
        logger.info("=" * 50)
        
        # Generar predicciones
        logger.info("Generando predicciones")
        self.predictions = self.model.predict(self.df)
        
        # Calcular métricas finales
        if hasattr(self.model, 'cutoff_date'):
            test_data = self.df[self.df['Date'] >= self.model.cutoff_date].copy()
            if len(test_data) > 0:
                test_indices = test_data.index
                y_true = test_data['PTS'].values
                y_pred = self.predictions[test_indices] if len(self.predictions) == len(self.df) else self.predictions[:len(y_true)]
                
                # Ajustar dimensiones si es necesario
                min_len = min(len(y_true), len(y_pred))
                y_true = y_true[:min_len]
                y_pred = y_pred[:min_len]
                
                mae = np.mean(np.abs(y_true - y_pred))
                rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
                r2 = 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)
                
                # Métricas específicas para puntos
                accuracy_5pts = np.mean(np.abs(y_true - y_pred) <= 5) * 100
                accuracy_10pts = np.mean(np.abs(y_true - y_pred) <= 10) * 100
                
                logger.info(f"Métricas finales | MAE: {mae:.3f}, RMSE: {rmse:.3f}, R²: {r2:.3f}")
                logger.info(f"Accuracy ±5pts: {accuracy_5pts:.1f}%, ±10pts: {accuracy_10pts:.1f}%")
                
                self.training_results.update({
                    'final_mae': mae,
                    'final_rmse': rmse,
                    'final_r2': r2,
                    'final_accuracy_5pts': accuracy_5pts,
                    'final_accuracy_10pts': accuracy_10pts
                })
        
        return self.training_results
    
    def generate_all_visualizations(self):
        """
        Genera una visualización completa en PNG con todas las métricas principales.
        """
        logger.info("Generando visualización completa en PNG")
        
        # Asegurar que el directorio de salida existe
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Crear figura principal con subplots organizados
        fig = plt.figure(figsize=(24, 16))
        fig.suptitle('Dashboard Completo - Modelo NBA Teams Points Prediction', fontsize=20, fontweight='bold', y=0.98)
        
        # Crear grid de subplots (4 filas x 4 columnas)
        gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)
        
        # 1. Métricas principales del modelo
        ax1 = fig.add_subplot(gs[0, 0])
        self._plot_model_metrics_summary(ax1)
        
        # 2. Feature importance
        ax2 = fig.add_subplot(gs[0, 1:3])
        self._plot_feature_importance_compact(ax2)
        
        # 3. Distribución del target
        ax3 = fig.add_subplot(gs[0, 3])
        self._plot_target_distribution_compact(ax3)
        
        # 4. Predicciones vs Reales
        ax4 = fig.add_subplot(gs[1, 0:2])
        self._plot_predictions_vs_actual_compact(ax4)
        
        # 5. Residuos
        ax5 = fig.add_subplot(gs[1, 2:4])
        self._plot_residuals_compact(ax5)
        
        # 6. Análisis por rangos de puntos
        ax6 = fig.add_subplot(gs[2, 0:2])
        self._plot_points_range_analysis_compact(ax6)
        
        # 7. Top equipos ofensivos
        ax7 = fig.add_subplot(gs[2, 2:4])
        self._plot_top_offensive_teams_compact(ax7)
        
        # 8. Análisis temporal
        ax8 = fig.add_subplot(gs[3, 0:2])
        self._plot_temporal_analysis_compact(ax8)
        
        # 9. Validación cruzada
        ax9 = fig.add_subplot(gs[3, 2:4])
        self._plot_cv_results_compact(ax9)
        
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
        mae = self.training_results.get('mae', 0)
        rmse = self.training_results.get('rmse', 0)
        r2 = self.training_results.get('r2', 0)
        accuracy_5pts = self.training_results.get('accuracy_5pts', 0)
        accuracy_10pts = self.training_results.get('accuracy_10pts', 0)
        
        # Crear texto de métricas
        metrics_text = f"""
MÉTRICAS DEL MODELO TEAMS POINTS

MAE: {mae:.3f}
RMSE: {rmse:.3f}
R²: {r2:.3f}

ACCURACY PUNTOS:
±5 pts: {accuracy_5pts:.1f}%
±10 pts: {accuracy_10pts:.1f}%

MODELOS BASE:
• XGBoost (Offense Engine)
• LightGBM (Tempo Control)
• CatBoost (Team Chemistry)
• Random Forest (Consistency)
• Neural Network (Patterns)
• Ridge (Stability)
"""
        
        ax.text(0.05, 0.95, metrics_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7))
        
        ax.set_title('Resumen del Modelo', fontweight='bold', fontsize=12)
    
    def _plot_feature_importance_compact(self, ax):
        """Gráfico compacto de importancia de features."""
        try:
            # Debug logging
            logger.info("DEBUG: Iniciando _plot_feature_importance_compact")
            logger.info(f"DEBUG: Modelo entrenado: {getattr(self.model, 'is_trained', False)}")
            logger.info(f"DEBUG: Tiene get_feature_importance: {hasattr(self.model, 'get_feature_importance')}")
            
            # Obtener feature importance del modelo
            top_features = {}
            
            if hasattr(self.model, 'get_feature_importance'):
                logger.info("DEBUG: Llamando get_feature_importance...")
                importance_dict = self.model.get_feature_importance(top_n=20)
                logger.info(f"DEBUG: Resultado importance_dict: {type(importance_dict)}")
                logger.info(f"DEBUG: Claves en importance_dict: {list(importance_dict.keys()) if importance_dict else 'None'}")
                
                if importance_dict and 'top_features' in importance_dict and importance_dict['top_features']:
                    logger.info(f"DEBUG: top_features encontrado, longitud: {len(importance_dict['top_features'])}")
                    # Convertir la lista de diccionarios a un diccionario simple
                    top_features_list = importance_dict['top_features'][:20]
                    top_features = {item['feature']: item['importance'] for item in top_features_list}
                    logger.info(f"DEBUG: top_features convertido: {len(top_features)} items")
                elif importance_dict and 'feature_importance' in importance_dict:
                    logger.info("DEBUG: Usando feature_importance")
                    top_features = dict(list(importance_dict['feature_importance'].items())[:20])
                elif importance_dict and isinstance(importance_dict, dict):
                    logger.info("DEBUG: Usando dict directo")
                    top_features = dict(list(importance_dict.items())[:20])
                else:
                    logger.warning("DEBUG: No se pudo procesar importance_dict")
            else:
                logger.warning("DEBUG: Modelo no tiene get_feature_importance")
            
            # Si no se obtuvo feature importance, crear uno básico
            if not top_features and hasattr(self.model, 'feature_columns') and len(self.model.feature_columns) > 0:
                logger.info("DEBUG: Creando feature importance básico")
                features = self.model.feature_columns[:20]
                importances = [1.0/len(features)] * len(features)
                top_features = {f: imp for f, imp in zip(features, importances)}
                logger.info(f"DEBUG: Feature importance básico creado: {len(top_features)} items")
            
            # Si aún no hay datos, mostrar mensaje
            if not top_features:
                logger.warning("DEBUG: No hay top_features, mostrando mensaje de no disponible")
                ax.text(0.5, 0.5, 'Feature importance\nno disponible', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title('Feature Importance', fontweight='bold')
                return
            
            logger.info(f"DEBUG: Creando plot con {len(top_features)} features")
            features = list(top_features.keys())
            importances = list(top_features.values())
            
            # Crear gráfico horizontal
            y_pos = np.arange(len(features))
            bars = ax.barh(y_pos, importances, color='lightcoral', alpha=0.8)
            
            ax.set_yticks(y_pos)
            ax.set_yticklabels([f.replace('_', ' ').title()[:25] for f in features], fontsize=7)
            ax.set_xlabel('Importancia')
            ax.set_title('Top 20 Features Más Importantes', fontweight='bold')
            
            # Agregar valores en las barras
            for i, (bar, val) in enumerate(zip(bars, importances)):
                ax.text(bar.get_width() + max(importances) * 0.01, bar.get_y() + bar.get_height()/2, 
                       f'{val:.3f}', va='center', fontsize=7)
            
            ax.grid(axis='x', alpha=0.3)
            ax.invert_yaxis()
            logger.info("DEBUG: Plot de feature importance completado exitosamente")
            
        except Exception as e:
            logger.error(f"DEBUG: Error en _plot_feature_importance_compact: {e}")
            import traceback
            logger.error(f"DEBUG: Traceback: {traceback.format_exc()}")
            ax.text(0.5, 0.5, f'Error cargando\nfeature importance:\n{str(e)}', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Feature Importance', fontweight='bold')
    
    def _plot_target_distribution_compact(self, ax):
        """Distribución compacta del target PTS."""
        pts_values = self.df['PTS']
        
        # Histograma
        ax.hist(pts_values, bins=25, alpha=0.7, color='lightcoral', edgecolor='black')
        
        # Estadísticas
        mean_pts = pts_values.mean()
        median_pts = pts_values.median()
        
        ax.axvline(mean_pts, color='red', linestyle='--', label=f'Media: {mean_pts:.1f}')
        ax.axvline(median_pts, color='blue', linestyle='--', label=f'Mediana: {median_pts:.1f}')
        
        ax.set_xlabel('Puntos por Equipo (PTS)')
        ax.set_ylabel('Frecuencia')
        ax.set_title('Distribución de PTS por Equipo', fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    
    def _plot_predictions_vs_actual_compact(self, ax):
        """Gráfico compacto de predicciones vs valores reales."""
        if self.predictions is None:
            ax.text(0.5, 0.5, 'Predicciones\nno disponibles', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Predicciones vs Reales', fontweight='bold')
            return
        
        # Usar datos de prueba si están disponibles
        if hasattr(self.model, 'cutoff_date'):
            test_data = self.df[self.df['Date'] >= self.model.cutoff_date].copy()
            if len(test_data) > 0:
                test_indices = test_data.index
                y_true = test_data['PTS'].values
                y_pred = self.predictions[test_indices] if len(self.predictions) == len(self.df) else self.predictions[:len(y_true)]
                
                # Ajustar dimensiones si es necesario
                min_len = min(len(y_true), len(y_pred))
                y_true = y_true[:min_len]
                y_pred = y_pred[:min_len]
            else:
                # Usar todos los datos
                y_true = self.df['PTS'].values
                y_pred = self.predictions
        else:
            # Usar todos los datos
            y_true = self.df['PTS'].values
            y_pred = self.predictions
        
        # Scatter plot
        ax.scatter(y_true, y_pred, alpha=0.6, s=20, color='coral')
        
        # Línea perfecta
        min_val = min(min(y_true), min(y_pred))
        max_val = max(max(y_true), max(y_pred))
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Predicción Perfecta')
        
        ax.set_xlabel('PTS Real')
        ax.set_ylabel('PTS Predicho')
        ax.set_title('Predicciones vs Valores Reales', fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)
        
        # Agregar R²
        r2 = r2_score(y_true, y_pred)
        ax.text(0.05, 0.95, f'R² = {r2:.3f}', transform=ax.transAxes, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    def _plot_residuals_compact(self, ax):
        """Gráfico compacto de residuos."""
        if self.predictions is None:
            ax.text(0.5, 0.5, 'Predicciones\nno disponibles', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Análisis de Residuos', fontweight='bold')
            return
        
        # Calcular residuos
        y_true = self.df['PTS'].values
        y_pred = self.predictions
        residuals = y_true - y_pred
        
        # Scatter plot de residuos
        ax.scatter(y_pred, residuals, alpha=0.6, s=20, color='orange')
        ax.axhline(y=0, color='red', linestyle='--', lw=2)
        
        ax.set_xlabel('PTS Predicho')
        ax.set_ylabel('Residuos (Real - Predicho)')
        ax.set_title('Análisis de Residuos', fontweight='bold')
        ax.grid(alpha=0.3)
        
        # Agregar estadísticas de residuos
        mae = np.mean(np.abs(residuals))
        ax.text(0.05, 0.95, f'MAE = {mae:.3f}', transform=ax.transAxes, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    def _plot_points_range_analysis_compact(self, ax):
        """Análisis compacto por rangos de puntos."""
        if self.predictions is None:
            ax.text(0.5, 0.5, 'Predicciones\nno disponibles', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Análisis por Rangos de Puntos', fontweight='bold')
            return
        
        y_true = self.df['PTS'].values
        y_pred = self.predictions
        
        # Definir rangos de puntos
        ranges = [
            (70, 90, 'Bajo (70-90)'),
            (91, 110, 'Medio (91-110)'),
            (111, 130, 'Alto (111-130)'),
            (131, 160, 'Elite (131+)')
        ]
        
        range_names = []
        range_maes = []
        range_counts = []
        
        for min_pts, max_pts, name in ranges:
            mask = (y_true >= min_pts) & (y_true <= max_pts)
            if np.sum(mask) > 0:
                range_mae = np.mean(np.abs(y_true[mask] - y_pred[mask]))
                range_names.append(name)
                range_maes.append(range_mae)
                range_counts.append(np.sum(mask))
        
        if range_names:
            # Crear gráfico de barras
            bars = ax.bar(range_names, range_maes, alpha=0.8, color=['lightblue', 'lightgreen', 'orange', 'red'])
            
            ax.set_ylabel('MAE')
            ax.set_title('MAE por Rango de Puntos', fontweight='bold')
            ax.tick_params(axis='x', rotation=45)
            
            # Agregar valores en las barras
            for bar, mae, count in zip(bars, range_maes, range_counts):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                       f'{mae:.2f}\n(n={count})', ha='center', va='bottom', fontsize=8)
            
            ax.grid(axis='y', alpha=0.3)
    
    def _plot_top_offensive_teams_compact(self, ax):
        """Análisis compacto de top equipos ofensivos."""
        # Obtener promedio de puntos por equipo
        team_stats = self.df.groupby('Team').agg({
            'PTS': ['mean', 'count']
        }).reset_index()
        
        team_stats.columns = ['Team', 'mean', 'count']
        
        # Filtrar equipos con al menos 20 juegos
        team_stats = team_stats[team_stats['count'] >= 20]
        
        # Top 10 equipos ofensivos
        top_teams = team_stats.nlargest(10, 'mean')
        
        if len(top_teams) > 0:
            teams = [t[:3] for t in top_teams['Team']]  # Abreviar nombres
            means = top_teams['mean']
            
            bars = ax.barh(teams, means, alpha=0.8, color='lightsteelblue')
            
            ax.set_xlabel('Promedio PTS')
            ax.set_title('Top 10 Equipos Ofensivos', fontweight='bold')
            
            # Agregar valores en las barras
            for i, (bar, val) in enumerate(zip(bars, means)):
                ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2, 
                       f'{val:.1f}', va='center', fontsize=8)
            
            ax.grid(axis='x', alpha=0.3)
            ax.invert_yaxis()
    
    def _plot_temporal_analysis_compact(self, ax):
        """Análisis temporal compacto."""
        # Agrupar por mes
        df_copy = self.df.copy()
        df_copy['month'] = pd.to_datetime(df_copy['Date']).dt.to_period('M')
        
        monthly_stats = df_copy.groupby('month').agg({
            'PTS': 'mean'
        }).reset_index()
        
        if len(monthly_stats) > 0:
            months = [str(m) for m in monthly_stats['month']]
            avg_pts = monthly_stats['PTS']
            
            ax.plot(months, avg_pts, marker='o', linewidth=2, markersize=4)
            
            ax.set_ylabel('Promedio PTS')
            ax.set_title('Promedio de Puntos por Mes', fontweight='bold')
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
                mae_scores = [fold.get('mae', 0) for fold in fold_scores]
                
                folds = range(1, len(mae_scores) + 1)
                
                bars = ax.bar(folds, mae_scores, alpha=0.8, color='purple')
                
                # Agregar valores en las barras
                for bar, mae in zip(bars, mae_scores):
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                           f'{mae:.2f}', ha='center', va='bottom', fontsize=8)
                
                ax.set_xlabel('Fold')
                ax.set_ylabel('MAE')
                ax.set_title('Validación Cruzada por Fold', fontweight='bold')
                ax.set_xticks(folds)
                ax.set_xticklabels([f'Fold {i}' for i in folds])
                
                # Agregar promedio
                avg_mae = np.mean(mae_scores)
                ax.axhline(y=avg_mae, color='red', linestyle='--', 
                          label=f'Promedio: {avg_mae:.3f}')
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
        logger.info("Guardando resultados")
        
        # Asegurar que el directorio de salida existe
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Guardar modelo en .joblib/
        model_path = os.path.normpath(os.path.join('.joblib', 'teams_points_model.joblib'))
        os.makedirs('.joblib', exist_ok=True)
        if hasattr(self.model, 'save_model'):
            self.model.save_model(model_path)
        else:
            # Backup: usar joblib
            joblib.dump(self.model, model_path)
        
        # Guardar reporte completo
        report = {
            'model_type': 'NBA Teams Points Ensemble',
            'training_results': self.training_results,
            'model_summary': self.model.get_training_summary() if hasattr(self.model, 'get_training_summary') else {},
            'timestamp': datetime.now().isoformat()
        }
        report_path = os.path.normpath(os.path.join(self.output_dir, 'training_report.json'))
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Guardar predicciones
        if self.predictions is not None:
            predictions_df = self.df[['Team', 'Date', 'Opp', 'PTS']].copy()
            predictions_df['PTS_predicted'] = self.predictions
            predictions_df['error'] = self.predictions - self.df['PTS']
            predictions_df['abs_error'] = np.abs(predictions_df['error'])
            
            predictions_path = os.path.normpath(os.path.join(self.output_dir, 'predictions.csv'))
            predictions_df.to_csv(predictions_path, index=False)
        
        # Guardar feature importance
        try:
            if hasattr(self.model, 'get_feature_importance'):
                importance_dict = self.model.get_feature_importance(top_n=50)
                if importance_dict and 'top_features' in importance_dict and importance_dict['top_features']:
                    # Convertir la lista de diccionarios a DataFrame
                    importance_df = pd.DataFrame(importance_dict['top_features'])
                    
                    importance_path = os.path.normpath(os.path.join(self.output_dir, 'feature_importance.csv'))
                    importance_df.to_csv(importance_path, index=False)
                    logger.info(f"Feature importance guardado en: {importance_path}")
                else:
                    logger.warning("Feature importance está vacío o no tiene 'top_features'")
            else:
                logger.warning("El modelo no tiene método 'get_feature_importance'")
        except Exception as e:
            logger.error(f"Error al guardar feature importance: {e}")
            # Intentar crear un archivo básico de feature importance
            try:
                if hasattr(self.model, 'feature_columns'):
                    basic_df = pd.DataFrame({
                        'feature': self.model.feature_columns[:50],
                        'importance': [1.0/len(self.model.feature_columns)] * min(50, len(self.model.feature_columns))
                    })
                    importance_path = os.path.normpath(os.path.join(self.output_dir, 'feature_importance.csv'))
                    basic_df.to_csv(importance_path, index=False)
                    logger.info(f"Feature importance básico guardado en: {importance_path}")
            except Exception as e2:
                logger.error(f"No se pudo crear feature importance básico: {e2}")
        
        # Crear resumen de archivos generados
        files_summary = {
            'model_file': '.joblib/teams_points_model.joblib',
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
        logger.info(f"  • Feature Importance: {os.path.normpath(os.path.join(self.output_dir, 'feature_importance.csv'))}")
        logger.info(f"  • Resumen: {summary_path}")
    
    def run_complete_training(self):
        """
        Ejecuta el pipeline completo de entrenamiento.
        
        Returns:
            Dict: Resultados completos del entrenamiento
        """
        logger.info("Iniciando pipeline de entrenamiento Teams Points")
        
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
    """
    Función principal para ejecutar el entrenamiento completo de TEAMS_POINTS.
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
    trainer = TeamsPointsTrainer(
        game_data_path=game_data_path,
        biometrics_path=biometrics_path,
        teams_path=teams_path,
        output_dir="results/teams_points_model",
        n_trials=50,
        cv_folds=5,
        random_state=42
    )
    
    # Ejecutar pipeline completo
    results = trainer.run_complete_training()
    
    print("Entrenamiento Teams Points Model completado!")
    print(f"Resultados: {trainer.output_dir}")
    
    return results


if __name__ == "__main__":
    main() 