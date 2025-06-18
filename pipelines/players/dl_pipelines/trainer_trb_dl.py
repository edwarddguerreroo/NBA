"""
Ejecutor del Trainer de Deep Learning para Rebotes NBA
======================================================

Pipeline completo para entrenar modelos de Deep Learning especializados
en predicción de rebotes usando arquitecturas avanzadas.
"""

import sys
import os
from pathlib import Path
import logging
import pandas as pd
import numpy as np
import torch
from datetime import datetime
import json

# Agregar el directorio raíz al path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.preprocessing.data_loader import NBADataLoader
from src.models.players.trb.features_trb import ReboundsFeatureEngineer
from src.models.players.trb.dl_trb.config import DLConfig, TransformerConfig, LSTMConfig, GNNConfig, VAEConfig, EnsembleConfig, HybridConfig
from src.models.players.trb.dl_trb.trainer import DLTrainer
from src.models.players.trb.dl_trb.evaluate import ModelEvaluator
from config.logging_config import configure_model_logging

# Configurar logging
logger = configure_model_logging("trb_dl_pipeline")


class TRBDeepLearningPipeline:
    """
    Pipeline completo para entrenamiento de modelos de Deep Learning TRB.
    
    Características:
    - Múltiples arquitecturas de DL (Transformer, LSTM, GNN, VAE, Ensemble, Hybrid)
    - Optimización automática de hiperparámetros
    - Evaluación exhaustiva con métricas especializadas
    - Comparación entre modelos
    - Guardado automático de resultados
    """
    
    def __init__(self, model_type: str = "transformer", use_gpu: bool = True):
        """
        Inicializa el pipeline.
        
        Args:
            model_type: Tipo de modelo DL a entrenar
            use_gpu: Si usar GPU cuando esté disponible
        """
        self.model_type = model_type
        self.use_gpu = use_gpu and torch.cuda.is_available()
        
        # Configurar dispositivo
        self.device = "cuda" if self.use_gpu else "cpu"
        logger.info(f"Pipeline DL TRB inicializado: modelo={model_type}, dispositivo={self.device}")
        
        # Inicializar componentes
        self.data_loader = NBADataLoader(
            game_data_path="data/players.csv",
            biometrics_path="data/height.csv", 
            teams_path="data/teams.csv"
        )
        self.feature_engineer = None
        self.trainer = None
        self.evaluator = None
        
        # Configuraciones disponibles
        self.config_map = {
            "transformer": TransformerConfig,
            "multiscale_transformer": TransformerConfig,
            "lstm": LSTMConfig,
            "hierarchical_lstm": LSTMConfig,
            "conv_lstm": LSTMConfig,
            "gnn": GNNConfig,
            "hierarchical_gnn": GNNConfig,
            "vae": VAEConfig,
            "conditional_vae": VAEConfig,
            "beta_vae": VAEConfig,
            "sequential_vae": VAEConfig,
            "ensemble": EnsembleConfig,
            "hierarchical_ensemble": EnsembleConfig,
            "hybrid": HybridConfig,
            "multiscale_hybrid": HybridConfig,
            "adaptive_hybrid": HybridConfig
        }
        
        # Resultados
        self.results = {}
        
    def load_and_prepare_data(self) -> pd.DataFrame:
        """Carga y prepara los datos con features especializadas."""
        logger.info("Cargando datos NBA...")
        
        # Cargar datos
        df_players, df_teams = self.data_loader.load_data()
        
        # Verificar datos
        if df_players is None or df_players.empty:
            raise ValueError("No se pudieron cargar los datos de jugadores")
        
        logger.info(f"Datos cargados: {len(df_players)} registros de jugadores")
        
        # Inicializar feature engineer
        self.feature_engineer = ReboundsFeatureEngineer(
            correlation_threshold=0.95,
            max_features=30
        )
        
        # Generar features especializadas
        logger.info("Generando features especializadas para Deep Learning TRB...")
        features = self.feature_engineer.generate_all_features(df_players)
        
        logger.info(f"Features generadas: {len(features)}")
        
        # Actualizar número de features en configuración
        self.num_features = len(features)
        self.feature_columns = features  # Guardar las features generadas
        
        # Filtrar datos válidos
        df_clean = df_players.dropna(subset=['TRB'])
        
        # Ordenar cronológicamente
        df_clean = df_clean.sort_values(['Player', 'Date'])
        
        logger.info(f"Datos preparados: {len(df_clean)} registros válidos")
        
        return df_clean
    
    def create_config(self) -> object:
        """Crea la configuración específica para el modelo."""
        if self.model_type not in self.config_map:
            raise ValueError(f"Tipo de modelo no soportado: {self.model_type}")
        
        config_class = self.config_map[self.model_type]
        
        # Configuraciones específicas por modelo
        if self.model_type == "transformer":
            config = config_class(
                device=self.device,
                batch_size=128,
                learning_rate=1e-3,
                num_epochs=100,
                patience=15,
                d_model=256,
                nhead=8,
                num_encoder_layers=6,
                dim_feedforward=1024,
                sequence_length=15,
                input_features=self.num_features,  # Ajustado al número real de features de TRB
                target_feature="TRB"
            )
        elif self.model_type == "lstm":
            config = config_class(
                device=self.device,
                batch_size=128,
                learning_rate=1e-3,
                num_epochs=100,
                patience=15,
                hidden_size=64,
                num_layers=3,
                bidirectional=True,
                sequence_length=15,
                input_features=self.num_features,
                target_feature="TRB"
            )
        elif self.model_type == "gnn":
            config = config_class(
                device=self.device,
                batch_size=64,
                learning_rate=1e-3,
                num_epochs=100,
                patience=15,
                hidden_dim=64,
                num_gnn_layers=4,
                gnn_type="GAT",
                num_attention_heads=8,
                input_features=self.num_features,
                target_feature="TRB"
            )
        elif self.model_type == "vae":
            config = config_class(
                device=self.device,
                batch_size=128,
                learning_rate=1e-3,
                num_epochs=100,
                patience=15,
                latent_dim=32,
                encoder_dims=[256, 128, 64],
                decoder_dims=[64, 128, 256],
                input_features=self.num_features,
                target_feature="TRB"
            )
        elif self.model_type == "ensemble":
            config = config_class(
                device=self.device,
                batch_size=128,
                learning_rate=1e-3,
                num_epochs=100,
                patience=15,
                num_experts=6,
                expert_types=["temporal", "team", "individual", "matchup", "situational", "physical"],
                input_features=self.num_features,
                target_feature="TRB"
            )
        elif self.model_type == "hybrid":
            config = config_class(
                device=self.device,
                batch_size=128,
                learning_rate=1e-3,
                num_epochs=100,
                patience=15,
                input_features=self.num_features,
                target_feature="TRB"
            )
        else:
            config = config_class(
                device=self.device,
                input_features=self.num_features,
                target_feature="TRB"
            )
        
        logger.info(f"Configuración creada para modelo {self.model_type}")
        return config
    
    def train_model(self, df: pd.DataFrame) -> dict:
        """Entrena el modelo de Deep Learning."""
        logger.info(f"Iniciando entrenamiento del modelo {self.model_type}...")
        
        # Crear configuración
        config = self.create_config()
        
        # Inicializar trainer
        self.trainer = DLTrainer(config, self.model_type)
        
        # Entrenar modelo
        train_results = self.trainer.train(df, self.feature_columns)
        
        logger.info(f"Entrenamiento completado - Mejor pérdida val: {train_results.get('best_val_loss', 'N/A')}")
        
        return train_results
    
    def evaluate_model(self, df: pd.DataFrame) -> dict:
        """Evalúa el modelo entrenado."""
        if self.trainer is None:
            raise ValueError("Modelo no entrenado. Ejecutar train_model primero.")
        
        logger.info("Evaluando modelo...")
        
        # Inicializar evaluador
        self.evaluator = ModelEvaluator(self.trainer.model, self.trainer.config)
        
        # Evaluar modelo
        eval_results = self.evaluator.evaluate(df)
        
        logger.info(f"Evaluación completada - MAE: {eval_results.get('mae', 'N/A'):.3f}")
        
        return eval_results
    
    def save_results(self, output_dir: str = "results/trb_dl_model"):
        """Guarda los resultados del entrenamiento y evaluación."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Guardar resultados
        results_file = os.path.join(output_dir, f"{self.model_type}_results.json")
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        # Guardar modelo
        if self.trainer and self.trainer.model:
            model_file = os.path.join(output_dir, f"{self.model_type}_model.pth")
            self.trainer.model.save_model(model_file)
        
        logger.info(f"Resultados guardados en: {output_dir}")
    
    def run_complete_pipeline(self) -> dict:
        """Ejecuta el pipeline completo de entrenamiento y evaluación."""
        logger.info("=== INICIANDO PIPELINE COMPLETO DE DEEP LEARNING TRB ===")
        
        try:
            # 1. Cargar y preparar datos
            df = self.load_and_prepare_data()
            
            # 2. Entrenar modelo
            train_results = self.train_model(df)
            
            # 3. Evaluar modelo
            eval_results = self.evaluate_model(df)
            
            # 4. Combinar resultados
            self.results = {
                'model_type': self.model_type,
                'device': self.device,
                'timestamp': datetime.now().isoformat(),
                'data_info': {
                    'total_samples': len(df),
                    'features_count': len(self.feature_columns)
                },
                'training': train_results,
                'evaluation': eval_results
            }
            
            # 5. Guardar resultados
            self.save_results()
            
            # 6. Log resumen final
            self._log_final_summary(self.results)
            
            return self.results
            
        except Exception as e:
            logger.error(f"Error en pipeline: {str(e)}")
            raise
    
    def _log_final_summary(self, results: dict):
        """Log del resumen final de resultados."""
        logger.info("=== RESUMEN FINAL DEL ENTRENAMIENTO ===")
        logger.info(f"Modelo: {results['model_type']}")
        logger.info(f"Dispositivo: {results['device']}")
        
        eval_metrics = results.get('evaluation', {})
        logger.info(f"MAE: {eval_metrics.get('mae', 'N/A'):.3f}")
        logger.info(f"RMSE: {eval_metrics.get('rmse', 'N/A'):.3f}")
        logger.info(f"R²: {eval_metrics.get('r2', 'N/A'):.3f}")
        logger.info(f"Precisión ±1 rebote: {eval_metrics.get('accuracy_1trb', 'N/A'):.1f}%")
        logger.info(f"Precisión ±2 rebotes: {eval_metrics.get('accuracy_2trb', 'N/A'):.1f}%")
        logger.info(f"Precisión ±3 rebotes: {eval_metrics.get('accuracy_3trb', 'N/A'):.1f}%")
        
        logger.info("=== PIPELINE COMPLETADO EXITOSAMENTE ===")


def main():
    """Función principal para ejecutar el pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Pipeline de Deep Learning para TRB NBA')
    parser.add_argument('--model', type=str, default='transformer',
                       choices=['transformer', 'lstm', 'gnn', 'vae', 'ensemble', 'hybrid'],
                       help='Tipo de modelo a entrenar')
    parser.add_argument('--no-gpu', action='store_true',
                       help='Forzar uso de CPU (no GPU)')
    
    args = parser.parse_args()
    
    # Crear y ejecutar pipeline
    pipeline = TRBDeepLearningPipeline(
        model_type=args.model,
        use_gpu=not args.no_gpu
    )
    
    results = pipeline.run_complete_pipeline()
    
    print("\n=== RESULTADOS FINALES ===")
    print(f"Modelo: {results['model_type']}")
    eval_metrics = results.get('evaluation', {})
    print(f"MAE: {eval_metrics.get('mae', 'N/A'):.3f}")
    print(f"R²: {eval_metrics.get('r2', 'N/A'):.3f}")
    print(f"Precisión ±2 rebotes: {eval_metrics.get('accuracy_2trb', 'N/A'):.1f}%")


if __name__ == "__main__":
    main() 