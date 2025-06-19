"""
Pipeline de Entrenamiento para Modelos de Deep Learning 3P
=========================================================

Pipeline especializado para entrenar y evaluar modelos de Deep Learning
para predicción de triples (3P) en la NBA.

Características:
- Integración con ThreePointsFeatureEngineer para features especializadas
- Soporte para múltiples arquitecturas DL (Transformer, LSTM, GNN, VAE, etc.)
- Validación cruzada y métricas específicas para triples
- Optimización de hiperparámetros
- Guardado automático de modelos

Autor: AI Basketball Analytics Expert
Fecha: 2025-06-10
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Imports del proyecto
from src.preprocessing.data_loader import NBADataLoader
from src.models.players.triples.features_triples import ThreePointsFeatureEngineer
from src.models.players.triples.dl_triples import (
    DLTrainer, DLConfig, Hybrid3PPredictor
)


class TriplesDeepLearningPipeline:
    """
    Pipeline completo para entrenamiento de modelos DL de triples.
    """
    
    def __init__(self, 
                 data_path: str = "data/nba_games_2018_2024.csv",
                 cache_dir: str = "cache/triples_dl",
                 results_dir: str = "results/triples_dl"):
        """
        Inicializa el pipeline.
        
        Args:
            data_path: Ruta al archivo de datos
            cache_dir: Directorio para cache
            results_dir: Directorio para resultados
        """
        self.data_path = data_path
        self.cache_dir = Path(cache_dir)
        self.results_dir = Path(results_dir)
        
        # Crear directorios
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Componentes
        self.data_loader = NBADataLoader(
            game_data_path="data/players.csv",
            biometrics_path="data/height.csv", 
            teams_path="data/teams.csv"
        )
        self.feature_engineer = ThreePointsFeatureEngineer()
        self.dl_config = DLConfig()
        
        # Datos
        self.df_raw = None
        self.df_features = None
        self.feature_columns = None
        
        logger.info("Pipeline de Deep Learning 3P inicializado")
    
    def load_and_prepare_data(self, 
                            min_games: int = 20,
                            correlation_threshold: float = 0.95,
                            max_features: int = 60) -> pd.DataFrame:
        """
        Carga y prepara los datos con features especializadas.
        
        Args:
            min_games: Mínimo de juegos por jugador
            correlation_threshold: Umbral para eliminación de features correlacionadas
            max_features: Máximo número de features a mantener
            
        Returns:
            DataFrame con features preparadas
        """
        logger.info("Cargando y preparando datos...")
        
        # Cargar datos
        df_players, df_teams = self.data_loader.load_data()
        self.df_raw = df_players  # Usar datos de jugadores
        logger.info(f"Datos cargados: {len(self.df_raw)} registros")
        
        # Filtrar jugadores con suficientes juegos
        player_games = self.df_raw.groupby('Player').size()
        valid_players = player_games[player_games >= min_games].index
        self.df_raw = self.df_raw[self.df_raw['Player'].isin(valid_players)]
        logger.info(f"Después del filtro de {min_games} juegos: {len(self.df_raw)} registros")
        
        # Configurar feature engineer con parámetros más conservadores
        self.feature_engineer = ThreePointsFeatureEngineer(
            correlation_threshold=0.98,  # Más conservador
            max_features=100,  # Permitir más features para filtrar después
            teams_df=df_teams
        )
        
        # Generar features especializadas
        logger.info("Generando features especializadas para triples...")
        try:
            all_features = self.feature_engineer.generate_all_features(self.df_raw)
            self.df_features = self.df_raw.copy()
            logger.info(f"Features generadas: {len(all_features)} features")
            
            # OPTIMIZACIÓN: Filtrar features por porcentaje de NaN
            logger.info("Aplicando filtro optimizado de NaN...")
            
            nan_analysis = []
            for feature in all_features:
                if feature in self.df_features.columns:
                    total_records = len(self.df_features)
                    nan_count = self.df_features[feature].isna().sum()
                    nan_percentage = (nan_count / total_records) * 100
                    
                    nan_analysis.append({
                        'feature': feature,
                        'nan_percentage': nan_percentage
                    })
            
            # Usar solo features con menos del 10% de NaN
            good_features = [item['feature'] for item in nan_analysis if item['nan_percentage'] < 10.0]
            logger.info(f"Features con <10% NaN: {len(good_features)} de {len(all_features)}")
            
            # Actualizar lista de features
            self.feature_columns = good_features
            
            # Verificar datos válidos
            required_cols = self.feature_columns + ['3P']
            df_clean = self.df_features.dropna(subset=required_cols)
            logger.info(f"Registros válidos después de filtro optimizado: {len(df_clean)}")
            
            if len(df_clean) == 0:
                raise ValueError("No hay registros válidos después de filtro optimizado")
            
        except Exception as e:
            logger.warning(f"Error generando features avanzadas: {e}")
            # Fallback: usar features básicas
            logger.info("Usando features básicas de triples...")
            
            # Buscar features básicas relacionadas con triples
            basic_features = []
            
            # Features básicas estándar de NBA
            standard_features = ['3PA', '3P%', 'FG', 'FGA', 'FG%', '2P', '2PA', '2P%', 
                                'FT', 'FTA', 'FT%', 'ORB', 'DRB', 'TRB', 'AST', 'STL', 
                                'BLK', 'TOV', 'PF', 'PTS', 'MP']
            
            for col in standard_features:
                if col in self.df_raw.columns and col != '3P':  # Excluir target
                    basic_features.append(col)
            
            # Buscar features que empiecen con 'threept_' si existen
            threept_features = [col for col in self.df_raw.columns 
                               if col.startswith('threept_') and col != '3P']
            basic_features.extend(threept_features)
            
            # Eliminar duplicados
            self.feature_columns = list(set(basic_features))
            self.df_features = self.df_raw.copy()
        
        logger.info(f"Features finales configuradas: {len(self.feature_columns)} features")
        logger.info(f"DataFrame final: {self.df_features.shape}")
        
        # Verificación final: asegurar que tenemos datos válidos
        required_cols = self.feature_columns + ['3P']
        df_clean = self.df_features.dropna(subset=required_cols)
        logger.info(f"Registros válidos después de limpieza: {len(df_clean)}")
        
        if len(df_clean) == 0:
            raise ValueError("No hay registros válidos después de preparar features")
        
        return self.df_features
    
    def train_model(self,
                   model_type: str = "hybrid",
                   test_size: float = 0.2,
                   validation_split: float = 0.2,
                   save_model: bool = True) -> Dict:
        """
        Entrena un modelo específico.
        
        Args:
            model_type: Tipo de modelo a entrenar
            test_size: Proporción de datos para test
            validation_split: Proporción de datos para validación
            save_model: Si guardar el modelo entrenado
            
        Returns:
            Resultados del entrenamiento
        """
        if self.df_features is None:
            raise ValueError("Debe cargar los datos primero con load_and_prepare_data()")
        
        logger.info(f"Entrenando modelo {model_type}")
        
        # Configurar modelo con el número real de features
        config = self.dl_config.get_config(model_type)
        actual_features = len(self.feature_columns)
        config.input_features = actual_features
        logger.info(f"Configurando modelo {model_type} con {actual_features} features")
        
        # Crear trainer
        trainer = DLTrainer(config, model_type)
        
        # División train/test
        from sklearn.model_selection import train_test_split
        train_df, test_df = train_test_split(
            self.df_features, 
            test_size=test_size, 
            random_state=config.random_seed,
            stratify=None  # No estratificar para regresión
        )
        
        logger.info(f"División de datos: {len(train_df)} train, {len(test_df)} test")
        
        # Entrenar
        save_path = None
        if save_model:
            save_path = str(self.results_dir / f"model_{model_type}_3p.pth")
        
        training_results = trainer.train(
            train_df, 
            feature_columns=self.feature_columns,
            validation_split=validation_split,
            save_path=save_path
        )
        
        # Evaluar en test
        test_predictions = trainer.predict(test_df)
        test_actual = test_df['3P'].values
        
        # Calcular métricas de test
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        
        test_mae = mean_absolute_error(test_actual, test_predictions)
        test_rmse = np.sqrt(mean_squared_error(test_actual, test_predictions))
        test_r2 = r2_score(test_actual, test_predictions)
        
        # Métricas específicas de triples
        test_accuracy_exact = np.mean(test_predictions.round() == test_actual) * 100
        test_accuracy_1 = np.mean(np.abs(test_predictions - test_actual) <= 1.0) * 100
        test_accuracy_2 = np.mean(np.abs(test_predictions - test_actual) <= 2.0) * 100
        
        test_metrics = {
            'mae': test_mae,
            'rmse': test_rmse,
            'r2': test_r2,
            'accuracy_exact_3p': test_accuracy_exact,
            'accuracy_1_3p': test_accuracy_1,
            'accuracy_2_3p': test_accuracy_2
        }
        
        logger.info(f"Métricas de test - MAE: {test_mae:.4f}, "
                   f"RMSE: {test_rmse:.4f}, R²: {test_r2:.4f}")
        logger.info(f"Exactitud exacta: {test_accuracy_exact:.2f}%, "
                   f"±1: {test_accuracy_1:.2f}%, ±2: {test_accuracy_2:.2f}%")
        
        # Compilar resultados
        results = {
            'model_type': model_type,
            'training_results': training_results,
            'test_metrics': test_metrics,
            'feature_count': len(self.feature_columns),
            'train_size': len(train_df),
            'test_size': len(test_df)
        }
        
        # Guardar resultados
        results_file = self.results_dir / f"results_{model_type}_3p.json"
        import json
        with open(results_file, 'w') as f:
            # Convertir numpy arrays a listas para JSON
            json_results = results.copy()
            for key, value in json_results.get('test_metrics', {}).items():
                if isinstance(value, np.ndarray):
                    json_results['test_metrics'][key] = value.tolist()
            
            json.dump(json_results, f, indent=2, default=str)
        
        return results
    
    def compare_models(self, 
                      model_types: List[str] = None,
                      cv_folds: int = 5) -> pd.DataFrame:
        """
        Compara diferentes modelos usando validación cruzada.
        
        Args:
            model_types: Lista de tipos de modelos a comparar
            cv_folds: Número de folds para validación cruzada
            
        Returns:
            DataFrame con comparación de resultados
        """
        if self.df_features is None:
            raise ValueError("Debe cargar los datos primero con load_and_prepare_data()")
        
        if model_types is None:
            model_types = ["transformer", "lstm", "gnn", "hybrid"]
        
        logger.info(f"Comparando {len(model_types)} modelos con CV de {cv_folds} folds")
        
        results = []
        
        for model_type in model_types:
            logger.info(f"Evaluando modelo {model_type}")
            
            try:
                # Configurar modelo
                config = self.dl_config.get_config(model_type)
                actual_features = len(self.feature_columns)
                config.input_features = actual_features
                logger.info(f"CV - Configurando modelo {model_type} con {actual_features} features")
                
                # Crear trainer
                trainer = DLTrainer(config, model_type)
                
                # Validación cruzada usando las features ya configuradas en load_and_prepare_data
                cv_results = trainer.cross_validate(
                    self.df_features, 
                    cv_folds=cv_folds,
                    feature_columns=self.feature_columns  # Usar features específicas configuradas
                )
                
                # Extraer métricas promedio
                cv_summary = cv_results['cv_summary']
                
                result_row = {
                    'model_type': model_type,
                    'mae_mean': cv_summary['mae_mean'],
                    'mae_std': cv_summary['mae_std'],
                    'rmse_mean': cv_summary.get('rmse_mean', 0),
                    'rmse_std': cv_summary.get('rmse_std', 0),
                    'r2_mean': cv_summary.get('r2_mean', 0),
                    'r2_std': cv_summary.get('r2_std', 0),
                    'feature_count': cv_results.get('feature_count', actual_features),
                    'cv_folds': cv_folds
                }
                
                results.append(result_row)
                
                logger.info(f"Modelo {model_type} completado: "
                           f"MAE={cv_summary['mae_mean']:.4f}±{cv_summary['mae_std']:.4f}")
                
            except Exception as e:
                logger.error(f"Error entrenando {model_type}: {e}")
                
                # Agregar resultado con error
                error_row = {
                    'model_type': model_type,
                    'mae_mean': float('inf'),
                    'mae_std': 0,
                    'rmse_mean': float('inf'),
                    'rmse_std': 0,
                    'r2_mean': -float('inf'),
                    'r2_std': 0,
                    'feature_count': len(self.feature_columns),
                    'cv_folds': cv_folds,
                    'error': str(e)
                }
                results.append(error_row)
        
        # Crear DataFrame de resultados
        results_df = pd.DataFrame(results)
        
        # Ordenar por MAE (mejor primero)
        results_df = results_df.sort_values('mae_mean')
        
        # Guardar resultados
        results_file = self.results_dir / "model_comparison_3p.csv"
        results_df.to_csv(results_file, index=False)
        logger.info(f"Comparación guardada en {results_file}")
        
        return results_df
    
    def run_full_pipeline(self,
                         min_games: int = 20,
                         correlation_threshold: float = 0.95,
                         max_features: int = 60,
                         model_types: List[str] = None) -> Dict:
        """
        Ejecuta el pipeline completo.
        
        Args:
            min_games: Mínimo de juegos por jugador
            correlation_threshold: Umbral para eliminación de features correlacionadas
            max_features: Máximo número de features
            model_types: Lista de modelos a entrenar
            
        Returns:
            Resultados completos del pipeline
        """
        logger.info("Iniciando pipeline completo de Deep Learning 3P")
        
        # 1. Cargar y preparar datos
        self.load_and_prepare_data(
            min_games=min_games,
            correlation_threshold=correlation_threshold,
            max_features=max_features
        )
        
        # 2. Comparar modelos
        if model_types is None:
            model_types = ["transformer", "lstm", "gnn", "hybrid"]
        
        comparison_df = self.compare_models(model_types)
        
        # 3. Entrenar mejor modelo
        if not comparison_df.empty:
            best_model = comparison_df.iloc[0]['model_type']
            logger.info(f"Mejor modelo: {best_model}")
            
            best_model_results = self.train_model(
                model_type=best_model,
                save_model=True
            )
        else:
            best_model_results = None
        
        # 4. Compilar resultados finales
        final_results = {
            'data_summary': {
                'total_records': len(self.df_features),
                'feature_count': len(self.feature_columns),
                'players': self.df_features['Player'].nunique(),
                'date_range': [
                    str(self.df_features['Date'].min()),
                    str(self.df_features['Date'].max())
                ]
            },
            'model_comparison': comparison_df.to_dict('records'),
            'best_model_results': best_model_results
        }
        
        # Guardar resultados finales
        final_results_file = self.results_dir / "final_results_3p.json"
        import json
        with open(final_results_file, 'w') as f:
            json.dump(final_results, f, indent=2, default=str)
        
        logger.info(f"Pipeline completado. Resultados en {self.results_dir}")
        
        return final_results


def main():
    """Función principal para ejecutar el pipeline."""
    
    # Configuración
    DATA_PATH = "data/nba_games_2018_2024.csv"
    
    # Crear pipeline
    pipeline = TriplesDeepLearningPipeline(
        data_path=DATA_PATH,
        cache_dir="cache/triples_dl",
        results_dir="results/triples_dl"
    )
    
    # Ejecutar pipeline completo
    results = pipeline.run_full_pipeline(
        min_games=20,
        correlation_threshold=0.95,
        max_features=60,
        model_types=["transformer", "lstm", "gnn", "hybrid"]
    )
    
    print("\n" + "="*50)
    print("PIPELINE DE DEEP LEARNING 3P COMPLETADO")
    print("="*50)
    
    # Mostrar resumen
    data_summary = results['data_summary']
    print(f"\nDatos procesados:")
    print(f"- Registros totales: {data_summary['total_records']:,}")
    print(f"- Features generadas: {data_summary['feature_count']}")
    print(f"- Jugadores únicos: {data_summary['players']:,}")
    print(f"- Rango de fechas: {data_summary['date_range'][0]} a {data_summary['date_range'][1]}")
    
    # Mostrar mejores modelos
    if results['model_comparison']:
        print(f"\nTop 3 modelos por MAE:")
        for i, model in enumerate(results['model_comparison'][:3]):
            print(f"{i+1}. {model['model_type']}: "
                  f"MAE = {model['mae_mean']:.4f} (±{model['mae_std']:.4f})")
    
    # Mostrar métricas del mejor modelo
    if results['best_model_results']:
        best_results = results['best_model_results']
        test_metrics = best_results['test_metrics']
        print(f"\nMejor modelo ({best_results['model_type']}) - Métricas de test:")
        print(f"- MAE: {test_metrics['mae']:.4f}")
        print(f"- RMSE: {test_metrics['rmse']:.4f}")
        print(f"- R²: {test_metrics['r2']:.4f}")
        print(f"- Exactitud exacta: {test_metrics['accuracy_exact_3p']:.2f}%")
        print(f"- Exactitud ±1: {test_metrics['accuracy_1_3p']:.2f}%")
        print(f"- Exactitud ±2: {test_metrics['accuracy_2_3p']:.2f}%")


if __name__ == "__main__":
    main() 