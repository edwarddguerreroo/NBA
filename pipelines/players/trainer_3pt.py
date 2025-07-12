"""
Trainer Completo para Modelo XGBoost 3PT
========================================

Trainer que integra carga de datos, entrenamiento del modelo XGBoost
y generación completa de métricas y visualizaciones para predicción de triples NBA.

Características:
- Integración completa con data loader
- Entrenamiento automatizado con optimización bayesiana
- Generación de dashboard PNG unificado con todas las métricas
- Métricas detalladas y reportes completos
- Análisis de feature importance específico para triples
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
from src.models.players.triples.model_triples import XGBoost3PTModel

# Configuración de warnings y logging
# Import del sistema de logging unificado
from config.logging_config import configure_trainer_logging, NBALogger

warnings.filterwarnings('ignore')
logger = configure_trainer_logging('3pt')

# Configurar estilo de visualizaciones optimizado para PNG
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10


class XGBoost3PTTrainer:
    """
    Trainer completo para modelo XGBoost de predicción de triples NBA.
    
    Integra carga de datos, entrenamiento, evaluación y visualizaciones específicas para 3PT.
    """
    
    def __init__(self,
                 game_data_path: str,
                 biometrics_path: str,
                 teams_path: str,
                 output_dir: str = "results/3pt_model",
                 n_trials: int = 75,
                 cv_folds: int = 5,
                 random_state: int = 42):
        """
        Inicializa el trainer completo para triples.
        
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
            self.output_dir = os.path.normpath("results_3pt_model")
            os.makedirs(self.output_dir, exist_ok=True)
            logger.info(f"Usando directorio alternativo: {self.output_dir}")
        
        # Componentes principales
        self.data_loader = NBADataLoader(
            game_data_path, biometrics_path, teams_path
        )
        
        # Cargar datos para pasar teams_df al modelo
        self.df, self.teams_df = self.data_loader.load_data()
        
        self.model = XGBoost3PTModel(
            n_trials=n_trials,
            cv_folds=cv_folds,
            random_state=random_state,
            teams_df=self.teams_df
        )
        
        # Resultados
        self.training_results = None
        self.predictions = None
        
        logger.info(f"Trainer XGBoost 3PT inicializado | Output: {self.output_dir}")
    
    def load_and_prepare_data(self) -> pd.DataFrame:
        """
        Carga y prepara todos los datos necesarios para triples.
        
        Returns:
            pd.DataFrame: Datos preparados para entrenamiento
        """
        print("Cargando datos NBA")
        
        # Los datos ya están cargados en __init__, solo verificamos
        if self.df is None or self.teams_df is None:
            self.df, self.teams_df = self.data_loader.load_data()
        
        # Verificar target específico para triples
        if '3P' not in self.df.columns:
            raise ValueError("Columna '3P' no encontrada en los datos")
        
        print(f"Datos preparados: {len(self.df)} registros | {self.df['Player'].nunique()} jugadores")
        
        return self.df
    
    def train_model(self) -> Dict:
        """
        Entrena el modelo completo con optimización y validación para triples.
        
        Returns:
            Dict: Resultados del entrenamiento
        """
        print("Entrenando modelo ultra-perfeccionado")
        
        if self.df is None:
            raise ValueError("Datos no cargados. Ejecutar load_and_prepare_data() primero")
        
        # Entrenar modelo
        start_time = datetime.now()
        self.training_results = self.model.train(self.df)
        training_duration = (datetime.now() - start_time).total_seconds()
        
        print(f"Entrenamiento completado en {training_duration:.1f}s")
        
        # Generar predicciones en conjunto de prueba
        self.predictions = self.model.predict(self.df)
        
        # Compilar resultados completos
        results = {
            'training_metrics': self.model.training_metrics,
            'validation_metrics': getattr(self.model, 'validation_metrics', {}),
            'cv_scores': getattr(self.model, 'cv_scores', {}),
            'feature_importance': self.model.get_feature_importance(20),
            'best_params': getattr(self.model, 'best_params_per_model', {}),
            'training_duration_seconds': training_duration,
            'model_info': {
                'n_features': len(self.model.selected_features),
                'n_samples': len(self.df),
                'target_mean': self.df['3P'].mean(),
                'target_std': self.df['3P'].std(),
                'target_max': self.df['3P'].max()
            }
        }
        
        return results
    
    def generate_all_visualizations(self):
        """
        Genera una visualización completa en PNG con todas las métricas específicas para triples.
        """
        print("Generando visualizaciones")
        
        # Asegurar que el directorio de salida existe
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Crear figura principal con subplots organizados
        fig = plt.figure(figsize=(24, 16))
        fig.suptitle('Dashboard Completo - Modelo NBA 3PT Prediction', fontsize=20, fontweight='bold', y=0.98)
        
        # Crear grid de subplots (4 filas x 4 columnas)
        gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)
        
        # 1. Métricas principales del modelo (esquina superior izquierda)
        ax1 = fig.add_subplot(gs[0, 0])
        self._plot_model_metrics_summary(ax1)
        
        # 2. Feature importance (esquina superior derecha)
        ax2 = fig.add_subplot(gs[0, 1:3])
        self._plot_feature_importance_compact(ax2)
        
        # 3. Distribución del target (3P)
        ax3 = fig.add_subplot(gs[0, 3])
        self._plot_target_distribution_compact(ax3)
        
        # 4. Predicciones vs Real (fila 2, izquierda)
        ax4 = fig.add_subplot(gs[1, 0:2])
        self._plot_predictions_vs_actual_compact(ax4)
        
        # 5. Análisis de residuos
        ax5 = fig.add_subplot(gs[1, 2])
        self._plot_residuos_compact(ax5)
        
        # 6. Resultados CV
        ax6 = fig.add_subplot(gs[1, 3])
        self._plot_cv_results_compact(ax6)
        
        # 7. Análisis por rango de triples (fila 3)
        ax7 = fig.add_subplot(gs[2, 0:2])
        self._plot_threepoint_range_analysis_compact(ax7)
        
        # 8. Análisis temporal (fila 3, derecha)
        ax8 = fig.add_subplot(gs[2, 2:4])
        self._plot_temporal_analysis_compact(ax8)
        
        # 9. Top tiradores analysis (fila 4, izquierda)
        ax9 = fig.add_subplot(gs[3, 0:2])
        self._plot_top_shooters_analysis_compact(ax9)
        
        # 10. Análisis de eficiencia vs volumen (fila 4, derecha)
        ax10 = fig.add_subplot(gs[3, 2:4])
        self._plot_efficiency_vs_volume_compact(ax10)
        
        # Guardar dashboard completo
        dashboard_path = os.path.join(self.output_dir, "3pt_model_dashboard_complete.png")
        plt.savefig(dashboard_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        
        print(f"Dashboard guardado: {dashboard_path}")
    
    def _plot_model_metrics_summary(self, ax):
        """Gráfico resumen de métricas principales del modelo"""
        if not self.training_results:
            ax.text(0.5, 0.5, 'No hay resultados\nde entrenamiento', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_title('Métricas del Modelo 3PT')
            return
        
        # Obtener métricas principales
        metrics = self.training_results.get('training_metrics', {})
        
        # Crear tabla de métricas
        metric_data = []
        if 'mae' in metrics:
            metric_data.append(['MAE', f"{metrics['mae']:.3f}"])
        if 'rmse' in metrics:
            metric_data.append(['RMSE', f"{metrics['rmse']:.3f}"])
        if 'r2' in metrics:
            metric_data.append(['R²', f"{metrics['r2']:.3f}"])
        if 'cv_mae_mean' in metrics:
            metric_data.append(['CV MAE', f"{metrics['cv_mae_mean']:.3f}"])
        
        # Agregar información del modelo
        model_info = self.training_results.get('model_info', {})
        if 'n_features' in model_info:
            metric_data.append(['Features', f"{model_info['n_features']}"])
        if 'target_mean' in model_info:
            metric_data.append(['3P Media', f"{model_info['target_mean']:.2f}"])
        
        # Verificar que tengamos datos para mostrar
        if not metric_data:
            ax.text(0.5, 0.5, 'Métricas\nno disponibles', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Métricas del Modelo 3PT', fontweight='bold', pad=20)
            ax.axis('off')
            return
        
        # Crear tabla
        table = ax.table(cellText=metric_data,
                        colLabels=['Métrica', 'Valor'],
                        cellLoc='center',
                        loc='center',
                        colWidths=[0.5, 0.4])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        # Estilo de la tabla
        for i in range(len(metric_data) + 1):
            for j in range(2):
                cell = table[(i, j)]
                if i == 0:  # Header
                    cell.set_facecolor('#40466e')
                    cell.set_text_props(weight='bold', color='white')
                else:
                    cell.set_facecolor('#f1f1f2')
        
        ax.axis('off')
        ax.set_title('Métricas del Modelo 3PT', fontweight='bold', pad=20)
    
    def _plot_feature_importance_compact(self, ax):
        """Gráfico compacto de feature importance específico para triples"""
        feature_importance = self.training_results.get('feature_importance', {})
        
        if not feature_importance:
            ax.text(0.5, 0.5, 'Feature importance\nno disponible', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Top Features para 3PT')
            return
        
        # Tomar top 12 features para triples
        top_features = list(feature_importance.items())[:12]
        features, importances = zip(*top_features)
        
        # Crear barplot horizontal
        bars = ax.barh(range(len(features)), importances, color='steelblue', alpha=0.7)
        ax.set_yticks(range(len(features)))
        ax.set_yticklabels([f.replace('threept_', '3PT_').replace('_', ' ').title() 
                           for f in features], fontsize=9)
        ax.set_xlabel('Importancia')
        ax.set_title('Top Features para Predicción de 3PT', fontweight='bold')
        
        # Agregar valores en las barras
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax.text(width + 0.001, bar.get_y() + bar.get_height()/2, 
                   f'{importances[i]:.3f}', ha='left', va='center', fontsize=8)
        
        ax.grid(axis='x', alpha=0.3)
        ax.invert_yaxis()
    
    def _plot_target_distribution_compact(self, ax):
        """Distribución del target (3P) específica para triples"""
        if '3P' not in self.df.columns:
            ax.text(0.5, 0.5, 'Target 3P\nno disponible', 
                   ha='center', va='center', transform=ax.transAxes)
            return
        
        threept_values = self.df['3P'].values
        
        # Histograma con bins específicos para triples (0-12)
        bins = range(int(threept_values.max()) + 2)
        counts, _, patches = ax.hist(threept_values, bins=bins, alpha=0.7, color='skyblue', edgecolor='black')
        
        # Colorear barras según frecuencia
        for i, (patch, count) in enumerate(zip(patches, counts)):
            if i <= 2:  # 0-2 triples (común)
                patch.set_facecolor('lightcoral')
            elif i <= 5:  # 3-5 triples (buen juego)
                patch.set_facecolor('lightgreen')
            else:  # 6+ triples (excepcional)
                patch.set_facecolor('gold')
        
        ax.set_xlabel('Triples Anotados (3P)')
        ax.set_ylabel('Frecuencia')
        ax.set_title('Distribución de Triples', fontweight='bold')
        
        # Agregar estadísticas
        mean_3p = threept_values.mean()
        ax.axvline(mean_3p, color='red', linestyle='--', alpha=0.8, label=f'Media: {mean_3p:.2f}')
        ax.legend()
        ax.grid(alpha=0.3)
    
    def _plot_predictions_vs_actual_compact(self, ax):
        """Scatter plot de predicciones vs valores reales para triples"""
        if self.predictions is None or '3P' not in self.df.columns:
            ax.text(0.5, 0.5, 'Predicciones\nno disponibles', 
                   ha='center', va='center', transform=ax.transAxes)
            return
        
        actual = self.df['3P'].values
        predicted = self.predictions
        
        # Scatter plot
        ax.scatter(actual, predicted, alpha=0.6, s=30, color='steelblue')
        
        # Línea de referencia perfecta
        max_val = max(actual.max(), predicted.max())
        ax.plot([0, max_val], [0, max_val], 'r--', alpha=0.8, label='Predicción Perfecta')
        
        # Métricas en el gráfico
        mae = mean_absolute_error(actual, predicted)
        r2 = r2_score(actual, predicted)
        
        ax.text(0.05, 0.95, f'MAE: {mae:.3f}\nR²: {r2:.3f}', 
               transform=ax.transAxes, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax.set_xlabel('3P Reales')
        ax.set_ylabel('3P Predichos')
        ax.set_title('Predicciones vs Valores Reales (3PT)', fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)
    
    def _plot_residuos_compact(self, ax):
        """Análisis de residuos para triples"""
        if self.predictions is None or '3P' not in self.df.columns:
            ax.text(0.5, 0.5, 'Residuos\nno disponibles', 
                   ha='center', va='center', transform=ax.transAxes)
            return
        
        actual = self.df['3P'].values
        predicted = self.predictions
        residuos = actual - predicted
        
        # Scatter de residuos
        ax.scatter(predicted, residuos, alpha=0.6, s=20, color='darkgreen')
        ax.axhline(y=0, color='red', linestyle='--', alpha=0.8)
        
        ax.set_xlabel('3P Predichos')
        ax.set_ylabel('Residuos')
        ax.set_title('Análisis de Residuos', fontweight='bold')
        ax.grid(alpha=0.3)
    
    def _plot_cv_results_compact(self, ax):
        """Resultados de validación cruzada"""
        cv_scores = self.training_results.get('cv_scores', {})
        
        if not cv_scores:
            ax.text(0.5, 0.5, 'CV scores\nno disponibles', 
                   ha='center', va='center', transform=ax.transAxes)
            return
        
        # Extraer métricas de CV
        metrics_data = []
        labels = []
        
        if 'cv_mae_mean' in cv_scores:
            metrics_data.append(cv_scores['cv_mae_mean'])
            labels.append('MAE')
        if 'cv_rmse_mean' in cv_scores:
            metrics_data.append(cv_scores['cv_rmse_mean'])
            labels.append('RMSE')
        if 'cv_r2_mean' in cv_scores:
            metrics_data.append(cv_scores['cv_r2_mean'])
            labels.append('R²')
        
        if metrics_data:
            bars = ax.bar(labels, metrics_data, color=['lightcoral', 'lightblue', 'lightgreen'][:len(metrics_data)])
            
            # Agregar valores en las barras
            for bar, value in zip(bars, metrics_data):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{value:.3f}', ha='center', va='bottom')
        
        ax.set_title('Métricas de Validación Cruzada', fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
    
    def _plot_threepoint_range_analysis_compact(self, ax):
        """Análisis por rangos de triples específico"""
        if self.predictions is None or '3P' not in self.df.columns:
            ax.text(0.5, 0.5, 'Análisis de rangos\nno disponible', 
                   ha='center', va='center', transform=ax.transAxes)
            return
        
        actual = self.df['3P'].values
        predicted = self.predictions
        
        # Definir rangos específicos para triples
        ranges = [
            (0, 1, '0-1 (Malo)'),
            (2, 3, '2-3 (Promedio)'),
            (4, 5, '4-5 (Bueno)'),
            (6, 15, '6+ (Excepcional)')
        ]
        
        range_mae = []
        range_labels = []
        range_counts = []
        
        for min_val, max_val, label in ranges:
            mask = (actual >= min_val) & (actual <= max_val)
            if mask.sum() > 0:
                mae = mean_absolute_error(actual[mask], predicted[mask])
                range_mae.append(mae)
                range_labels.append(label)
                range_counts.append(mask.sum())
        
        # Crear gráfico de barras
        bars = ax.bar(range_labels, range_mae, 
                     color=['lightcoral', 'khaki', 'lightgreen', 'gold'][:len(range_mae)])
        
        # Agregar conteos en las barras
        for bar, count, mae in zip(bars, range_counts, range_mae):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                   f'MAE: {mae:.3f}\nn={count}', ha='center', va='bottom', fontsize=9)
        
        ax.set_ylabel('MAE')
        ax.set_title('Precisión por Rango de Triples', fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        plt.setp(ax.get_xticklabels(), rotation=15)
    
    def _plot_temporal_analysis_compact(self, ax):
        """Análisis temporal para triples"""
        if 'Date' not in self.df.columns or self.predictions is None:
            ax.text(0.5, 0.5, 'Análisis temporal\nno disponible', 
                   ha='center', va='center', transform=ax.transAxes)
            return
        
        # Crear DataFrame temporal
        df_temp = self.df.copy()
        df_temp['Predicted_3P'] = self.predictions
        df_temp['Date'] = pd.to_datetime(df_temp['Date'])
        
        # Agregar por mes
        df_monthly = df_temp.groupby(df_temp['Date'].dt.to_period('M')).agg({
            '3P': 'mean',
            'Predicted_3P': 'mean'
        }).reset_index()
        
        # Plot
        x_pos = range(len(df_monthly))
        ax.plot(x_pos, df_monthly['3P'], 'b-o', label='Real', markersize=4)
        ax.plot(x_pos, df_monthly['Predicted_3P'], 'r-s', label='Predicho', markersize=4)
        
        ax.set_xlabel('Período')
        ax.set_ylabel('Promedio 3P')
        ax.set_title('Tendencia Temporal de Triples', fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)
        
        # Rotar etiquetas
        if len(df_monthly) > 0:
            ax.set_xticks(x_pos[::max(1, len(x_pos)//6)])
            ax.set_xticklabels([str(p) for p in df_monthly['Date'][::max(1, len(x_pos)//6)]], 
                              rotation=45)
    
    def _plot_top_shooters_analysis_compact(self, ax):
        """Análisis de top tiradores de triples"""
        if '3P' not in self.df.columns or self.predictions is None:
            ax.text(0.5, 0.5, 'Análisis de tiradores\nno disponible', 
                   ha='center', va='center', transform=ax.transAxes)
            return
        
        # Agregar por jugador (mínimo 10 juegos)
        df_players = self.df.copy()
        df_players['Predicted_3P'] = self.predictions
        
        player_stats = df_players.groupby('Player').agg({
            '3P': ['mean', 'count'],
            'Predicted_3P': 'mean'
        }).round(3)
        
        # Aplanar columnas
        player_stats.columns = ['3P_Real', 'Games', '3P_Pred']
        player_stats = player_stats[player_stats['Games'] >= 10]  # Mínimo 10 juegos
        
        # Top 10 tiradores por promedio real
        top_shooters = player_stats.nlargest(10, '3P_Real')
        
        if len(top_shooters) > 0:
            x_pos = range(len(top_shooters))
            width = 0.35
            
            bars1 = ax.bar([x - width/2 for x in x_pos], top_shooters['3P_Real'], 
                          width, label='Real', color='steelblue', alpha=0.7)
            bars2 = ax.bar([x + width/2 for x in x_pos], top_shooters['3P_Pred'], 
                          width, label='Predicho', color='orange', alpha=0.7)
            
            ax.set_xlabel('Jugadores')
            ax.set_ylabel('Promedio 3P')
            ax.set_title('Top 10 Tiradores de Triples', fontweight='bold')
            ax.set_xticks(x_pos)
            ax.set_xticklabels([name[:10] for name in top_shooters.index], rotation=45)
            ax.legend()
            ax.grid(axis='y', alpha=0.3)
    
    def _plot_efficiency_vs_volume_compact(self, ax):
        """Análisis de eficiencia vs volumen para triples"""
        if '3P' not in self.df.columns or '3PA' not in self.df.columns:
            ax.text(0.5, 0.5, 'Análisis eficiencia\nvs volumen\nno disponible', 
                   ha='center', va='center', transform=ax.transAxes)
            return
        
        # Agregar por jugador
        player_stats = self.df.groupby('Player').agg({
            '3P': 'sum',
            '3PA': 'sum',
            'Player': 'count'
        }).rename(columns={'Player': 'Games'})
        
        # Filtrar jugadores con suficientes intentos
        player_stats = player_stats[player_stats['3PA'] >= 30]  # Mínimo 30 intentos
        player_stats['3P%'] = player_stats['3P'] / player_stats['3PA']
        player_stats['3PA_per_game'] = player_stats['3PA'] / player_stats['Games']
        
        if len(player_stats) > 0:
            # Scatter plot
            scatter = ax.scatter(player_stats['3PA_per_game'], player_stats['3P%'], 
                               alpha=0.6, s=50, c=player_stats['3P'], cmap='viridis')
            
            ax.set_xlabel('Intentos 3PA por Juego')
            ax.set_ylabel('Eficiencia 3P%')
            ax.set_title('Eficiencia vs Volumen de Triples', fontweight='bold')
            ax.grid(alpha=0.3)
            
            # Colorbar
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label('Total 3P Anotados')
    
    def save_results(self):
        """
        Guarda todos los resultados del entrenamiento en archivos.
        """
        print("Guardando resultados")
        
        # Guardar modelo entrenado en .joblib/
        model_path = os.path.join('.joblib', "3pt_model.joblib")
        os.makedirs('.joblib', exist_ok=True)
        self.model.save_model(model_path)
        
        # Guardar métricas en JSON
        if self.training_results:
            metrics_path = os.path.join(self.output_dir, "3pt_training_results.json")
            
            # Convertir numpy arrays a listas para JSON serialization
            json_results = {}
            for key, value in self.training_results.items():
                if isinstance(value, dict):
                    json_results[key] = {k: float(v) if isinstance(v, (np.integer, np.floating)) else v 
                                       for k, v in value.items()}
                elif isinstance(value, (np.integer, np.floating)):
                    json_results[key] = float(value)
                else:
                    json_results[key] = value
            
            with open(metrics_path, 'w') as f:
                json.dump(json_results, f, indent=2, default=str)
            
        
        # Guardar predicciones
        if self.predictions is not None:
            predictions_df = self.df[['Player', 'Date', '3P']].copy()
            predictions_df['Predicted_3P'] = self.predictions
            predictions_df['Residual'] = predictions_df['3P'] - predictions_df['Predicted_3P']
            
            predictions_path = os.path.join(self.output_dir, "3pt_predictions.csv")
            predictions_df.to_csv(predictions_path, index=False)
        
        # Guardar feature importance
        feature_importance = self.model.get_feature_importance(50)
        if feature_importance:
            importance_df = pd.DataFrame(list(feature_importance.items()), 
                                       columns=['Feature', 'Importance'])
            importance_path = os.path.join(self.output_dir, "3pt_feature_importance.csv")
            importance_df.to_csv(importance_path, index=False)
        
        print(f"Archivos guardados en: {self.output_dir}")
    
    def run_complete_training(self):
        """
        Ejecuta el pipeline completo de entrenamiento para triples.
        """
        print("Iniciando entrenamiento ultra-perfeccionado")
        
        try:
            # 1. Cargar y preparar datos
            self.load_and_prepare_data()
            
            # 2. Entrenar modelo
            results = self.train_model()
            
            # 3. Generar visualizaciones
            self.generate_all_visualizations()
            
            # 4. Guardar resultados
            self.save_results()
            
            # Resumen final
            if results:
                training_metrics = results.get('training_metrics', {})
                print("\nRESUMEN FINAL DEL MODELO 3PT")
                print(f"MAE: {training_metrics.get('mae', 'N/A'):.4f}")
                print(f"RMSE: {training_metrics.get('rmse', 'N/A'):.4f}")
                print(f"R²: {training_metrics.get('r2', 'N/A'):.4f}")
                print(f"Features: {results.get('model_info', {}).get('n_features', 'N/A')}")
                print(f"Tiempo: {results.get('training_duration_seconds', 0):.1f}s")
            
            return results
            
        except Exception as e:
            print(f"Error: {str(e)}")
            raise


def main():
    """
    Función principal para ejecutar el entrenamiento completo de triples.
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
    trainer = XGBoost3PTTrainer(
        game_data_path=game_data_path,
        biometrics_path=biometrics_path,
        teams_path=teams_path,
        output_dir="results/3pt_model",
        n_trials=150,  
        cv_folds=5,
        random_state=42
    )
    
    # Ejecutar entrenamiento completo
    results = trainer.run_complete_training()
    
    print("¡Entrenamiento ultra-perfeccionado completado!")
    print(f" Resultados: {trainer.output_dir}")
    
    return results


if __name__ == "__main__":
    results = main() 