"""
Trainer Completo para Modelo XGBoost AST
========================================

Trainer que integra carga de datos, entrenamiento del modelo XGBoost
y generación completa de métricas y visualizaciones para predicción de asistencias NBA.

Características:
- Integración completa con data loader
- Entrenamiento automatizado con optimización bayesiana
- Generación de dashboard PNG unificado con todas las métricas
- Métricas detalladas específicas para asistencias
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
from src.models.players.ast.model_ast import XGBoostASTModel

# Import del sistema de logging unificado
from config.logging_config import configure_trainer_logging, NBALogger

warnings.filterwarnings('ignore')
logger = configure_trainer_logging('ast')

# Configurar estilo de visualizaciones optimizado para PNG
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10


class XGBoostASTTrainer:
    """
    Trainer completo para modelo XGBoost de predicción de asistencias NBA.
    
    Integra carga de datos, entrenamiento, evaluación y visualizaciones.
    """
    
    def __init__(self,
                 game_data_path: str,
                 biometrics_path: str,
                 teams_path: str,
                 output_dir: str = "results/ast_model",
                 n_trials: int = 50,  # Trials para optimización bayesiana
                 cv_folds: int = 5,
                 random_state: int = 42):
        """
        Inicializa el trainer completo para AST.
        
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
            self.output_dir = os.path.normpath("results_ast_model")
            os.makedirs(self.output_dir, exist_ok=True)
            logger.info(f"Usando directorio alternativo: {self.output_dir}")
        
        # Componentes principales
        self.data_loader = NBADataLoader(
            game_data_path, biometrics_path, teams_path
        )
        self.model = XGBoostASTModel(
            enable_neural_network=True,
            enable_svr=False,  # Deshabilitado para mayor velocidad
            enable_gpu=False,
            random_state=random_state,
            teams_df=None  # Se asignará después de cargar datos
        )
        
        # Configurar parámetros de optimización
        self.model.stacking_model.n_trials = n_trials
        self.model.stacking_model.cv_folds = cv_folds
        
        # Datos y resultados
        self.df = None
        self.teams_df = None
        self.training_results = None
        self.predictions = None
        
        logger.info(f"Trainer XGBoost AST inicializado | Output: {self.output_dir}")
    
    def load_and_prepare_data(self) -> pd.DataFrame:
        """
        Carga y prepara todos los datos necesarios.
        
        Returns:
            pd.DataFrame: Datos preparados para entrenamiento
        """
        logger.info("Cargando datos NBA")
        
        # Cargar datos usando el data loader
        self.df, self.teams_df = self.data_loader.load_data()
        
        # ASIGNAR DATOS DE EQUIPOS AL MODELO
        self.model.stacking_model.teams_df = self.teams_df
        self.model.stacking_model.feature_engineer.teams_df = self.teams_df
        logger.info("Datos de equipos asignados al modelo para features avanzadas")
        
        # Estadísticas básicas de los datos
        logger.info(f"Datos cargados: {len(self.df)} registros de jugadores")
        logger.info(f"Datos de equipos: {len(self.teams_df)} registros")
        logger.info(f"Jugadores únicos: {self.df['Player'].nunique()}")
        logger.info(f"Equipos únicos: {self.df['Team'].nunique()}")
        logger.info(f"Rango de fechas: {self.df['Date'].min()} a {self.df['Date'].max()}")
        
        # Verificar target
        if 'AST' not in self.df.columns:
            raise ValueError("Columna 'AST' no encontrada en los datos")
        
        # Estadísticas del target
        ast_stats = self.df['AST'].describe()
        logger.info(f"Estadísticas AST - Media: {ast_stats['mean']:.2f}, "
                   f"Mediana: {ast_stats['50%']:.2f}, "
                   f"Max: {ast_stats['max']:.0f}")
        
        return self.df
    
    def train_model(self) -> Dict:
        """
        Entrena el modelo completo con optimización y validación.
        
        Returns:
            Dict: Resultados del entrenamiento
        """
        logger.info("Iniciando entrenamiento del modelo XGBoost AST")
        
        if self.df is None:
            raise ValueError("Datos no cargados. Ejecutar load_and_prepare_data() primero")
        
        # Entrenar modelo
        start_time = datetime.now()
        logger.info("Entrenando modelo AST con stacking ensemble")
        self.training_results = self.model.train(self.df)
        training_duration = (datetime.now() - start_time).total_seconds()
        
        logger.info(f"Modelo AST completado | Duración: {training_duration:.1f} segundos")
        
        # Mostrar resultados del entrenamiento
        logger.info("=" * 50)
        logger.info("RESULTADOS DEL ENTRENAMIENTO AST")
        logger.info("=" * 50)
        logger.info(f"MAE: {self.training_results.get('mae', 0):.4f}")
        logger.info(f"RMSE: {self.training_results.get('rmse', 0):.4f}")
        logger.info(f"R²: {self.training_results.get('r2', 0):.4f}")
        logger.info(f"Accuracy ±1ast: {self.training_results.get('accuracy_1ast', 0):.1f}%")
        logger.info(f"Accuracy ±2ast: {self.training_results.get('accuracy_2ast', 0):.1f}%")
        logger.info(f"Accuracy ±3ast: {self.training_results.get('accuracy_3ast', 0):.1f}%")
        logger.info("=" * 50)
        
        # Generar predicciones
        logger.info("Generando predicciones")
        self.predictions = self.model.predict(self.df)
        
        # Calcular métricas finales en datos de prueba
        test_data = self.df[self.df['Date'] >= self.model.cutoff_date].copy()
        if len(test_data) > 0:
            # Obtener índices de los datos de prueba para alinear predicciones
            test_indices = test_data.index
            
            y_true = test_data['AST'].values
            # Alinear predicciones con los datos de prueba usando los índices
            y_pred = self.predictions[test_indices] if len(self.predictions) == len(self.df) else self.predictions[:len(y_true)]
            
            # Verificar que las dimensiones coincidan
            if len(y_true) != len(y_pred):
                logger.warning(f"Dimensiones no coinciden: y_true={len(y_true)}, y_pred={len(y_pred)}")
                # Ajustar al tamaño menor para evitar errores
                min_len = min(len(y_true), len(y_pred))
                y_true = y_true[:min_len]
                y_pred = y_pred[:min_len]
                logger.info(f"Ajustado a dimensión común: {min_len}")
            
            mae = np.mean(np.abs(y_true - y_pred))
            rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
            r2 = 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)
            
            # Métricas específicas para asistencias
            accuracy_1ast = np.mean(np.abs(y_true - y_pred) <= 1) * 100
            accuracy_2ast = np.mean(np.abs(y_true - y_pred) <= 2) * 100
            accuracy_3ast = np.mean(np.abs(y_true - y_pred) <= 3) * 100
            
            logger.info(f"Métricas finales | MAE: {mae:.3f}, RMSE: {rmse:.3f}, R²: {r2:.3f}")
            logger.info(f"Accuracy ±1ast: {accuracy_1ast:.1f}%, ±2ast: {accuracy_2ast:.1f}%, ±3ast: {accuracy_3ast:.1f}%")
            
            self.training_results.update({
                'final_mae': mae,
                'final_rmse': rmse,
                'final_r2': r2,
                'final_accuracy_1ast': accuracy_1ast,
                'final_accuracy_2ast': accuracy_2ast,
                'final_accuracy_3ast': accuracy_3ast
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
        fig.suptitle('Dashboard Completo - Modelo NBA AST Prediction', fontsize=20, fontweight='bold', y=0.98)
        
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
        
        # 7. Análisis por rangos de asistencias (tercera fila, derecha)
        ax7 = fig.add_subplot(gs[2, 2:4])
        self._plot_assists_range_analysis_compact(ax7)
        
        # 8. Análisis temporal (cuarta fila, izquierda)
        ax8 = fig.add_subplot(gs[3, 0:2])
        self._plot_temporal_analysis_compact(ax8)
        
        # 9. Top pasadores predicciones (cuarta fila, derecha)
        ax9 = fig.add_subplot(gs[3, 2:4])
        self._plot_top_passers_analysis_compact(ax9)
        
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
        accuracy_1ast = self.training_results.get('accuracy_1ast', 0)
        accuracy_2ast = self.training_results.get('accuracy_2ast', 0)
        accuracy_3ast = self.training_results.get('accuracy_3ast', 0)
        
        # Crear texto de métricas
        metrics_text = f"""
MÉTRICAS DEL MODELO AST

MAE: {mae:.3f}
RMSE: {rmse:.3f}
R²: {r2:.3f}

ACCURACY ASISTENCIAS:
±1 ast: {accuracy_1ast:.1f}%
±2 ast: {accuracy_2ast:.1f}%
±3 ast: {accuracy_3ast:.1f}%

MODELOS BASE:
• XGBoost (Court Vision)
• LightGBM (Court Vision)
• CatBoost (Basketball IQ)
• Gradient Boosting (Basketball IQ)
• Neural Network (Interactions)
• Ridge (Stabilizer)
"""
        
        ax.text(0.05, 0.95, metrics_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7))
        
        ax.set_title('Resumen del Modelo', fontweight='bold', fontsize=12)
    
    def _plot_feature_importance_compact(self, ax):
        """Gráfico compacto de importancia de features."""
        if not hasattr(self.model.stacking_model, 'feature_importance') or not self.model.stacking_model.feature_importance:
            ax.text(0.5, 0.5, 'Feature importance\nno disponible', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Feature Importance', fontweight='bold')
            return
        
        # Obtener top 15 features
        importance_dict = self.model.stacking_model.feature_importance
        top_features = dict(list(importance_dict.items())[:15])
        
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
    
    def _plot_target_distribution_compact(self, ax):
        """Distribución compacta del target AST."""
        ast_values = self.df['AST']
        
        # Histograma
        ax.hist(ast_values, bins=20, alpha=0.7, color='lightcoral', edgecolor='black')
        
        # Estadísticas
        mean_ast = ast_values.mean()
        median_ast = ast_values.median()
        
        ax.axvline(mean_ast, color='red', linestyle='--', label=f'Media: {mean_ast:.1f}')
        ax.axvline(median_ast, color='blue', linestyle='--', label=f'Mediana: {median_ast:.1f}')
        
        ax.set_xlabel('Asistencias (AST)')
        ax.set_ylabel('Frecuencia')
        ax.set_title('Distribución de AST', fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    
    def _plot_predictions_vs_actual_compact(self, ax):
        """Gráfico compacto de predicciones vs valores reales."""
        if self.predictions is None:
            ax.text(0.5, 0.5, 'Predicciones\nno disponibles', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Predicciones vs Reales', fontweight='bold')
            return
        
        # Usar datos de prueba
        test_data = self.df[self.df['Date'] >= self.model.cutoff_date].copy()
        if len(test_data) == 0:
            ax.text(0.5, 0.5, 'Datos de prueba\nno disponibles', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Predicciones vs Reales', fontweight='bold')
            return
        
        test_indices = test_data.index
        y_true = test_data['AST'].values
        y_pred = self.predictions[test_indices] if len(self.predictions) == len(self.df) else self.predictions[:len(y_true)]
        
        # Ajustar dimensiones si es necesario
        min_len = min(len(y_true), len(y_pred))
        y_true = y_true[:min_len]
        y_pred = y_pred[:min_len]
        
        # Scatter plot
        ax.scatter(y_true, y_pred, alpha=0.6, s=20, color='coral')
        
        # Línea perfecta
        min_val = min(min(y_true), min(y_pred))
        max_val = max(max(y_true), max(y_pred))
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Predicción Perfecta')
        
        ax.set_xlabel('AST Real')
        ax.set_ylabel('AST Predicho')
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
        
        # Usar datos de prueba
        test_data = self.df[self.df['Date'] >= self.model.cutoff_date].copy()
        if len(test_data) == 0:
            ax.text(0.5, 0.5, 'Datos de prueba\nno disponibles', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Análisis de Residuos', fontweight='bold')
            return
        
        test_indices = test_data.index
        y_true = test_data['AST'].values
        y_pred = self.predictions[test_indices] if len(self.predictions) == len(self.df) else self.predictions[:len(y_true)]
        
        # Ajustar dimensiones si es necesario
        min_len = min(len(y_true), len(y_pred))
        y_true = y_true[:min_len]
        y_pred = y_pred[:min_len]
        
        residuals = y_true - y_pred
        
        # Scatter plot de residuos
        ax.scatter(y_pred, residuals, alpha=0.6, s=20, color='orange')
        ax.axhline(y=0, color='red', linestyle='--', lw=2)
        
        ax.set_xlabel('AST Predicho')
        ax.set_ylabel('Residuos (Real - Predicho)')
        ax.set_title('Análisis de Residuos', fontweight='bold')
        ax.grid(alpha=0.3)
        
        # Agregar estadísticas de residuos
        mae = np.mean(np.abs(residuals))
        ax.text(0.05, 0.95, f'MAE = {mae:.3f}', transform=ax.transAxes, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    def _plot_cv_results_compact(self, ax):
        """Gráfico compacto de resultados de validación cruzada."""
        cv_scores = self.model.stacking_model.cv_scores
        
        if not cv_scores or 'cv_scores' not in cv_scores:
            ax.text(0.5, 0.5, 'Resultados CV\nno disponibles', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Validación Cruzada', fontweight='bold')
            return
        
        # Extraer MAE de cada fold
        fold_scores = cv_scores['cv_scores']
        mae_scores = [fold['mae'] for fold in fold_scores]
        r2_scores = [fold['r2'] for fold in fold_scores]
        
        folds = range(1, len(mae_scores) + 1)
        
        # Crear gráfico de barras
        x = np.arange(len(folds))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, mae_scores, width, label='MAE', alpha=0.8, color='orange')
        
        # Crear segundo eje para R²
        ax2 = ax.twinx()
        bars2 = ax2.bar(x + width/2, r2_scores, width, label='R²', alpha=0.8, color='purple')
        
        ax.set_xlabel('Fold')
        ax.set_ylabel('MAE', color='orange')
        ax2.set_ylabel('R²', color='purple')
        ax.set_title('Validación Cruzada por Fold', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([f'Fold {i}' for i in folds])
        
        # Agregar valores en las barras
        for bar, val in zip(bars1, mae_scores):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                   f'{val:.3f}', ha='center', va='bottom', fontsize=8)
        
        for bar, val in zip(bars2, r2_scores):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{val:.3f}', ha='center', va='bottom', fontsize=8)
        
        # Agregar promedios
        avg_mae = cv_scores.get('mae_mean', 0)
        avg_r2 = cv_scores.get('r2_mean', 0)
        ax.text(0.02, 0.98, f'Promedio MAE: {avg_mae:.3f}\nPromedio R²: {avg_r2:.3f}', 
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        ax.grid(alpha=0.3)
    
    def _plot_assists_range_analysis_compact(self, ax):
        """Análisis compacto por rangos de asistencias."""
        if self.predictions is None:
            ax.text(0.5, 0.5, 'Predicciones\nno disponibles', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Análisis por Rangos de Asistencias', fontweight='bold')
            return
        
        # Usar datos de prueba
        test_data = self.df[self.df['Date'] >= self.model.cutoff_date].copy()
        if len(test_data) == 0:
            ax.text(0.5, 0.5, 'Datos de prueba\nno disponibles', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Análisis por Rangos de Asistencias', fontweight='bold')
            return
        
        test_indices = test_data.index
        y_true = test_data['AST'].values
        y_pred = self.predictions[test_indices] if len(self.predictions) == len(self.df) else self.predictions[:len(y_true)]
        
        # Ajustar dimensiones si es necesario
        min_len = min(len(y_true), len(y_pred))
        y_true = y_true[:min_len]
        y_pred = y_pred[:min_len]
        
        # Definir rangos de asistencias
        ranges = [
            (0, 2, 'Bajo (0-2)'),
            (3, 5, 'Medio (3-5)'),
            (6, 8, 'Alto (6-8)'),
            (9, 20, 'Elite (9+)')
        ]
        
        range_names = []
        range_maes = []
        range_counts = []
        
        for min_ast, max_ast, name in ranges:
            mask = (y_true >= min_ast) & (y_true <= max_ast)
            if np.sum(mask) > 0:
                range_mae = np.mean(np.abs(y_true[mask] - y_pred[mask]))
                range_names.append(name)
                range_maes.append(range_mae)
                range_counts.append(np.sum(mask))
        
        if range_names:
            # Crear gráfico de barras
            bars = ax.bar(range_names, range_maes, alpha=0.8, color=['lightblue', 'lightgreen', 'orange', 'red'])
            
            ax.set_ylabel('MAE')
            ax.set_title('MAE por Rango de Asistencias', fontweight='bold')
            ax.tick_params(axis='x', rotation=45)
            
            # Agregar valores en las barras
            for bar, mae, count in zip(bars, range_maes, range_counts):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                       f'{mae:.3f}\n(n={count})', ha='center', va='bottom', fontsize=8)
            
            ax.grid(axis='y', alpha=0.3)
    
    def _plot_temporal_analysis_compact(self, ax):
        """Análisis temporal compacto."""
        if self.predictions is None:
            ax.text(0.5, 0.5, 'Predicciones\nno disponibles', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Análisis Temporal', fontweight='bold')
            return
        
        # Usar datos de prueba
        test_data = self.df[self.df['Date'] >= self.model.cutoff_date].copy()
        if len(test_data) == 0:
            ax.text(0.5, 0.5, 'Datos de prueba\nno disponibles', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Análisis Temporal', fontweight='bold')
            return
        
        test_indices = test_data.index
        y_true = test_data['AST'].values
        y_pred = self.predictions[test_indices] if len(self.predictions) == len(self.df) else self.predictions[:len(y_true)]
        
        # Ajustar dimensiones si es necesario
        min_len = min(len(y_true), len(y_pred))
        test_data = test_data.iloc[:min_len]
        y_true = y_true[:min_len]
        y_pred = y_pred[:min_len]
        
        # Agrupar por mes
        test_data_copy = test_data.copy()
        test_data_copy['month'] = pd.to_datetime(test_data_copy['Date']).dt.to_period('M')
        test_data_copy['y_true'] = y_true
        test_data_copy['y_pred'] = y_pred
        test_data_copy['abs_error'] = np.abs(y_true - y_pred)
        
        monthly_stats = test_data_copy.groupby('month').agg({
            'abs_error': 'mean',
            'y_true': 'count'
        }).reset_index()
        
        if len(monthly_stats) > 0:
            months = [str(m) for m in monthly_stats['month']]
            mae_by_month = monthly_stats['abs_error']
            
            bars = ax.bar(months, mae_by_month, alpha=0.8, color='lightcoral')
            
            ax.set_ylabel('MAE')
            ax.set_title('MAE por Mes', fontweight='bold')
            ax.tick_params(axis='x', rotation=45)
            
            # Agregar valores en las barras
            for bar, mae in zip(bars, mae_by_month):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                       f'{mae:.3f}', ha='center', va='bottom', fontsize=8)
            
            ax.grid(axis='y', alpha=0.3)
    
    def _plot_top_passers_analysis_compact(self, ax):
        """Análisis compacto de top pasadores."""
        # Obtener top pasadores por promedio
        player_stats = self.df.groupby('Player').agg({
            'AST': ['mean', 'count']
        }).reset_index()
        
        player_stats.columns = ['Player', 'mean', 'count']
        
        # Filtrar jugadores con al menos 10 juegos
        player_stats = player_stats[player_stats['count'] >= 10]
        
        # Top 10 pasadores
        top_passers = player_stats.nlargest(10, 'mean')
        
        if len(top_passers) > 0:
            players = [p[:15] + '' if len(p) > 15 else p for p in top_passers['Player']]
            means = top_passers['mean']
            
            bars = ax.barh(players, means, alpha=0.8, color='lightsteelblue')
            
            ax.set_xlabel('Promedio AST')
            ax.set_title('Top 10 Pasadores', fontweight='bold')
            
            # Agregar valores en las barras
            for i, (bar, val) in enumerate(zip(bars, means)):
                ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2, 
                       f'{val:.1f}', va='center', fontsize=8)
            
            ax.grid(axis='x', alpha=0.3)
            ax.invert_yaxis()
    
    def save_results(self):
        """Guarda todos los resultados del entrenamiento."""
        logger.info("Guardando resultados")
        
        # Asegurar que el directorio de salida existe
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Guardar modelo en .joblib/
        model_path = os.path.normpath(os.path.join('.joblib', 'ast_model.joblib'))
        os.makedirs('.joblib', exist_ok=True)
        self.model.save_model(model_path)
        
        # Guardar reporte completo
        report = {
            'model_type': 'XGBoost AST Stacking Ensemble',
            'training_results': self.training_results,
            'model_summary': self.model.stacking_model.get_model_summary(),
            'timestamp': datetime.now().isoformat()
        }
        report_path = os.path.normpath(os.path.join(self.output_dir, 'training_report.json'))
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Guardar predicciones
        if self.predictions is not None:
            predictions_df = self.df[['Player', 'Date', 'Team', 'AST']].copy()
            predictions_df['AST_predicted'] = self.predictions
            predictions_df['error'] = self.predictions - self.df['AST']
            predictions_df['abs_error'] = np.abs(predictions_df['error'])
            
            predictions_path = os.path.normpath(os.path.join(self.output_dir, 'predictions.csv'))
            predictions_df.to_csv(predictions_path, index=False)
        
        # Guardar feature importance
        if hasattr(self.model.stacking_model, 'feature_importance') and self.model.stacking_model.feature_importance:
            importance_df = pd.DataFrame([
                {'feature': k, 'importance': v} 
                for k, v in self.model.stacking_model.feature_importance.items()
            ]).sort_values('importance', ascending=False)
            
            importance_path = os.path.normpath(os.path.join(self.output_dir, 'feature_importance.csv'))
            importance_df.to_csv(importance_path, index=False)
        
        # Crear resumen de archivos generados
        files_summary = {
            'model_file': '.joblib/ast_model.joblib',
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
        logger.info("Iniciando pipeline de entrenamiento AST")
        
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
    Función principal para ejecutar el entrenamiento completo de AST.
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
    trainer = XGBoostASTTrainer(
        game_data_path=game_data_path,
        biometrics_path=biometrics_path,
        teams_path=teams_path,
        output_dir="results/ast_model",
        n_trials=20,
        cv_folds=5,
        random_state=42
    )
    
    # Ejecutar pipeline completo
    results = trainer.run_complete_training()
    
    print("Entrenamiento AST Model completado!")
    print(f"Resultados: {trainer.output_dir}")
    
    return results


if __name__ == "__main__":
    main() 