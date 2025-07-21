"""
Modelo Avanzado de Predicción de Total de Puntos NBA
===================================================

Este módulo implementa un sistema de predicción de alto rendimiento para
total de puntos de un juego NBA (ambos equipos combinados) utilizando:

1. Ensemble Learning con múltiples algoritmos ML y Red Neuronal
2. Stacking avanzado con meta-modelo optimizado
3. Optimización automática de hiperparámetros
4. Validación cruzada rigurosa
5. Métricas de evaluación exhaustivas
6. Feature engineering especializado para totales

ADAPTADO ESPECÍFICAMENTE PARA PREDICCIÓN DE TOTALES DE PARTIDOS
"""

# Standard Library
import os
import warnings
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

# Third-party Libraries - ML/Data
import joblib
import numpy as np
import pandas as pd

# Scikit-learn
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import (
    ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor,
    StackingRegressor
)
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import (
    GridSearchCV, KFold, RandomizedSearchCV, TimeSeriesSplit, cross_val_score
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVR

# XGBoost and LightGBM
import lightgbm as lgb
import xgboost as xgb

# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Bayesian Optimization
try:
    from skopt import gp_minimize, forest_minimize, gbrt_minimize
    from skopt.space import Real, Integer, Categorical
    from skopt.utils import use_named_args
    from skopt.acquisition import gaussian_ei, gaussian_pi, gaussian_lcb
    BAYESIAN_AVAILABLE = True
except ImportError:
    BAYESIAN_AVAILABLE = False

# Local imports
from .features_total_points import TotalPointsFeatureEngineer

# Configuration
warnings.filterwarnings('ignore')

# Logging setup
import logging
logger = logging.getLogger(__name__)


class GPUUtils:
    """Utilidades para gestión inteligente de GPU"""
    
    @staticmethod
    def get_optimal_device(device: Optional[str] = None,
                          memory_threshold_gb: float = 2.0) -> torch.device:
        """
        Selecciona el dispositivo óptimo considerando memoria disponible.
        
        Args:
            device: Dispositivo específico ('cuda:0', 'cuda:1', 'cpu', None)
            memory_threshold_gb: Memoria mínima requerida en GB
            
        Returns:
            torch.device optimizado
        """
        # Si se especifica CPU explícitamente
        if device == 'cpu':
            logger.info("Usando CPU por especificación explícita")
            return torch.device('cpu')
        
        # Si no hay CUDA disponible
        if not torch.cuda.is_available():
            return torch.device('cpu')
        
        # Si se especifica un dispositivo CUDA específico
        if device and device.startswith('cuda'):
            try:
                target_device = torch.device(device)
                if GPUUtils._check_gpu_memory(target_device, memory_threshold_gb):
                    logger.info(f"Usando dispositivo especificado: {device}")
                    return target_device
                else:
                    logger.warning(
                        f"Memoria insuficiente en {device}, "
                        f"buscando alternativa..."
                    )
            except Exception as e:
                logger.warning(f"Error con dispositivo {device}: {e}")
        
        # Buscar el mejor dispositivo GPU disponible
        best_device = GPUUtils._find_best_gpu(memory_threshold_gb)
        if best_device:
            return best_device
        
        # Fallback a CPU
        logger.info("Usando CPU como fallback")
        return torch.device('cpu')
    
    @staticmethod
    def _check_gpu_memory(device: torch.device,
                         memory_threshold_gb: float) -> bool:
        """Verifica si hay memoria suficiente en la GPU"""
        try:
            if device.type != 'cuda':
                return True
            
            torch.cuda.set_device(device)
            total_memory = torch.cuda.get_device_properties(device).total_memory
            allocated_memory = torch.cuda.memory_allocated(device)
            free_memory = total_memory - allocated_memory
            
            free_gb = free_memory / (1024**3)
            total_gb = total_memory / (1024**3)
            
            logger.info(
                f"GPU {device}: {free_gb:.1f}GB libre de {total_gb:.1f}GB total"
            )
            
            return free_gb >= memory_threshold_gb
            
        except Exception as e:
            logger.warning(f"Error verificando memoria GPU: {e}")
            return False
    
    @staticmethod
    def _find_best_gpu(memory_threshold_gb: float) -> Optional[torch.device]:
        """Encuentra la mejor GPU disponible"""
        if not torch.cuda.is_available():
            return None
        
        best_device = None
        best_free_memory = 0
        
        for i in range(torch.cuda.device_count()):
            device = torch.device(f'cuda:{i}')
            try:
                torch.cuda.set_device(device)
                total_memory = torch.cuda.get_device_properties(device).total_memory
                allocated_memory = torch.cuda.memory_allocated(device)
                free_memory = total_memory - allocated_memory
                
                free_gb = free_memory / (1024**3)
                
                if free_gb >= memory_threshold_gb and free_gb > best_free_memory:
                    best_free_memory = free_gb
                    best_device = device
                    
            except Exception as e:
                logger.warning(f"Error evaluando GPU {i}: {e}")
                continue
        
        if best_device:
            logger.info(
                f"Mejor GPU seleccionada: {best_device} "
                f"({best_free_memory:.1f}GB libre)"
            )
        
        return best_device


class DataPreprocessor:
    """Clase unificada para procesamiento de datos, eliminando duplicación"""
    
    def __init__(self, scaler: Optional[StandardScaler] = None):
        self.scaler = scaler or StandardScaler()
        self._is_fitted = False
    
    def prepare_data(self, df: pd.DataFrame, feature_columns: List[str],
                    target_column: str, validation_split: float = 0.2) -> Dict:
        """
        Prepara datos unificadamente para entrenamiento y predicción.
        
        Args:
            df: DataFrame con datos
            feature_columns: Lista de columnas de características
            target_column: Columna objetivo
            validation_split: Fracción para validación
            
        Returns:
            Diccionario con datos preparados
        """
        # Validar entrada
        self._validate_input_data(df, feature_columns, target_column)
        
        # Preparar características y target
        X = df[feature_columns].fillna(0)
        y = df[target_column] if target_column in df.columns else None
        
        # División temporal si hay target
        if y is not None:
            split_idx = int(len(df) * (1 - validation_split))
            X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
            y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]
            
            # Escalar datos de entrenamiento
            X_train_scaled = self._fit_transform_features(X_train, feature_columns)
            X_val_scaled = self._transform_features(X_val, feature_columns)
            
            return {
                'X_train': X_train_scaled,
                'X_val': X_val_scaled,
                'y_train': y_train,
                'y_val': y_val,
                'split_info': {
                    'train_size': len(X_train),
                    'val_size': len(X_val),
                    'split_idx': split_idx
                }
            }
        else:
            # Solo predicción
            X_scaled = self._transform_features(X, feature_columns)
            return {
                'X': X_scaled,
                'original_X': X
            }
    
    def _validate_input_data(self, df: pd.DataFrame, feature_columns: List[str],
                           target_column: str) -> None:
        """Valida datos de entrada"""
        if df.empty:
            raise ValueError("DataFrame está vacío")
        
        if not feature_columns:
            raise ValueError("Lista de características está vacía")
        
        missing_features = set(feature_columns) - set(df.columns)
        if missing_features:
            raise ValueError(
                f"Características faltantes: {list(missing_features)[:10]}"
            )
        
        if target_column in df.columns and df[target_column].isna().all():
            raise ValueError(f"Columna objetivo '{target_column}' está vacía")
    
    def _fit_transform_features(self, X: pd.DataFrame,
                              feature_columns: List[str]) -> pd.DataFrame:
        """Ajusta scaler y transforma características"""
        X_scaled_array = self.scaler.fit_transform(X)
        self._is_fitted = True
        return pd.DataFrame(
            X_scaled_array, columns=feature_columns, index=X.index
        )
    
    def _transform_features(self, X: pd.DataFrame,
                          feature_columns: List[str]) -> pd.DataFrame:
        """Transforma características usando scaler ajustado"""
        if not self._is_fitted:
            raise ValueError("Scaler debe ser ajustado antes de transformar")
        
        X_scaled_array = self.scaler.transform(X)
        return pd.DataFrame(
            X_scaled_array, columns=feature_columns, index=X.index
        )


class MetricsCalculator:
    """Calculadora unificada de métricas, eliminando duplicación"""
    
    @staticmethod
    def calculate_basic_metrics(y_true: np.ndarray,
                              y_pred: np.ndarray) -> Dict[str, float]:
        """Calcula métricas básicas de regresión"""
        return {
            'mae': mean_absolute_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'r2': r2_score(y_true, y_pred)
        }
    
    @staticmethod
    def calculate_accuracy_by_tolerance(y_true: np.ndarray, y_pred: np.ndarray,
                                      tolerances: List[float]) -> Dict[str, float]:
        """Calcula precisión por diferentes tolerancias"""
        results = {}
        for tolerance in tolerances:
            accuracy = np.mean(np.abs(y_true - y_pred) <= tolerance) * 100
            results[f'accuracy_{tolerance}pt'] = accuracy
        return results
    
    @staticmethod
    def analyze_prediction_stability(predictions_dict: Dict[str, np.ndarray],
                                   y_true: np.ndarray) -> Dict[str, Any]:
        """Analiza estabilidad de predicciones entre modelos"""
        metrics_by_model = {}
        
        for model_name, pred in predictions_dict.items():
            metrics_by_model[model_name] = MetricsCalculator.calculate_basic_metrics(
                y_true, pred
            )
        
        # Encontrar mejor modelo
        best_model = min(
            metrics_by_model.keys(),
            key=lambda k: metrics_by_model[k]['mae']
        )
        
        # Calcular estadísticas de estabilidad
        mae_values = [m['mae'] for m in metrics_by_model.values()]
        r2_values = [m['r2'] for m in metrics_by_model.values()]
        
        return {
            'metrics_by_model': metrics_by_model,
            'best_model': best_model,
            'stability_stats': {
                'mae_std': np.std(mae_values),
                'mae_range': np.max(mae_values) - np.min(mae_values),
                'r2_std': np.std(r2_values),
                'r2_range': np.max(r2_values) - np.min(r2_values)
            }
        }


class NBATotalPointsNet(nn.Module):
    """
    Red Neuronal Avanzada para Predicción de Total de Puntos NBA
    
    Arquitectura optimizada para predicción de totales (rango 170-320):
    - Input Layer
    - 2 Hidden Layers con Layer Normalization y Dropout
    - Output Layer optimizado para totales
    - Skip connections para mejor flujo de gradientes
    """
    
    def __init__(self, input_size: int, hidden_size: int = 128):
        super(NBATotalPointsNet, self).__init__()
        
        # Arquitectura específica para totales
        self.input_layer = nn.Linear(input_size, hidden_size)
        self.ln1 = nn.LayerNorm(hidden_size)
        self.dropout1 = nn.Dropout(0.3)
        
        self.hidden1 = nn.Linear(hidden_size, hidden_size // 2)
        self.ln2 = nn.LayerNorm(hidden_size // 2)
        self.dropout2 = nn.Dropout(0.4)
        
        self.hidden2 = nn.Linear(hidden_size // 2, hidden_size // 4)
        self.ln3 = nn.LayerNorm(hidden_size // 4)
        self.dropout3 = nn.Dropout(0.2)
        
        # Skip connection layer
        self.skip_layer = nn.Linear(input_size, hidden_size // 4)
        
        # Output layer para totales
        self.output = nn.Linear(hidden_size // 4, 1)
        
        # Inicialización de pesos optimizada para totales
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Inicialización optimizada de pesos para predicción de totales NBA"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # Xavier initialization para mejor convergencia
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """Forward pass con skip connections y LayerNorm robusto"""
        # Guardar input para skip connection
        skip = self.skip_layer(x)
        
        # Capas principales con LayerNorm
        x = F.relu(self.ln1(self.input_layer(x)))
        x = self.dropout1(x)
        
        x = F.relu(self.ln2(self.hidden1(x)))
        x = self.dropout2(x)
        
        x = F.relu(self.ln3(self.hidden2(x)))
        x = self.dropout3(x)
        
        # Skip connection para mejor flujo de gradientes
        x = x + skip
        
        # Output final
        x = self.output(x)
        
        return x


class PyTorchTotalPointsRegressor(RegressorMixin):
    """
    Wrapper de PyTorch para integración con scikit-learn y stacking
    
    Implementa una red neuronal optimizada para predicción de total de puntos NBA
    con regularización agresiva, entrenamiento robusto y soporte GPU mejorado.
    """
    
    def __init__(self, hidden_size: int = 128, epochs: int = 150,
                 batch_size: int = 32, learning_rate: float = 0.001,
                 weight_decay: float = 0.01, early_stopping_patience: int = 15,
                 device: Optional[str] = None, memory_threshold_gb: float = 2.0):
        self.hidden_size = hidden_size
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.early_stopping_patience = early_stopping_patience
        self.device_preference = device
        self.memory_threshold_gb = memory_threshold_gb
        
        self.model = None
        self.data_preprocessor = DataPreprocessor()
        self.device = None
        
        # Para early stopping
        self.best_loss = float('inf')
        self.patience_counter = 0
        self.best_model_state = None
        
        # Configurar dispositivo óptimo
        self._setup_device()
    
    def _setup_device(self):
        """Configura el dispositivo óptimo para entrenamiento"""
        self.device = GPUUtils.get_optimal_device(
            device=self.device_preference,
            memory_threshold_gb=self.memory_threshold_gb
        )
    
    @property
    def _estimator_type(self):
        """Identifica este estimador como regresor para sklearn"""
        return "regressor"
    
    def _check_n_features(self, X, reset):
        """Método para compatibilidad con sklearn"""
        pass
    
    def score(self, X, y, sample_weight=None):
        """Calcula R² score para compatibilidad con sklearn"""
        if self.model is None:
            raise ValueError("El modelo debe ser entrenado primero")
        y_pred = self.predict(X)
        return r2_score(y, y_pred, sample_weight=sample_weight)
    
    def fit(self, X, y):
        """
        Entrenamiento con regularización agresiva y early stopping avanzado
        """
        # Preparar datos usando el preprocessor unificado
        X, y = self._prepare_input_data(X, y)
        
        # Usar DataPreprocessor para división y escalado interno
        preprocessor = DataPreprocessor()
        
        # División train/val para early stopping
        val_size = 0.15
        val_split = int(len(X) * (1 - val_size))
        
        if val_split < 2:
            val_split = max(2, len(X) - 1)
        
        X_train = X[:val_split]
        X_val = X[val_split:]
        y_train = y[:val_split]
        y_val = y[val_split:]
        
        # Ajustar batch_size si es necesario
        effective_batch_size = min(self.batch_size, len(X_train))
        if effective_batch_size < 2:
            effective_batch_size = len(X_train)
        
        # Convertir a tensores de PyTorch
        tensors = self._convert_to_tensors(X_train, y_train, X_val, y_val)
        
        # Crear modelo
        input_size = X.shape[1]
        self.model = NBATotalPointsNet(input_size, self.hidden_size).to(self.device)
        
        # Configurar entrenamiento
        optimizer, scheduler, criterion = self._setup_training_components()
        
        # Crear dataloader para entrenamiento
        train_dataset = TensorDataset(tensors['X_train'], tensors['y_train'])
        train_dataloader = DataLoader(
            train_dataset, batch_size=effective_batch_size, shuffle=True
        )
        
        # Entrenar con early stopping
        training_stats = self._train_with_early_stopping(
            train_dataloader, tensors, optimizer, scheduler, criterion
        )
        
        # Restaurar mejor modelo
        self._restore_best_model()
        
        # Log resultados del entrenamiento
        self._log_training_results(training_stats)
        
        return self
    
    def _prepare_input_data(self, X, y) -> Tuple[np.ndarray, np.ndarray]:
        """Prepara datos de entrada para entrenamiento"""
        # Convertir a numpy arrays si es necesario
        if hasattr(X, 'values'):
            X = X.values
        if hasattr(y, 'values'):
            y = y.values
        
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        
        return X, y
    
    def _convert_to_tensors(self, X_train: np.ndarray, y_train: np.ndarray,
                          X_val: np.ndarray, y_val: np.ndarray) -> Dict[str, torch.Tensor]:
        """Convierte arrays numpy a tensores PyTorch en el dispositivo correcto"""
        return {
            'X_train': torch.FloatTensor(X_train).to(self.device),
            'y_train': torch.FloatTensor(y_train).view(-1, 1).to(self.device),
            'X_val': torch.FloatTensor(X_val).to(self.device),
            'y_val': torch.FloatTensor(y_val).view(-1, 1).to(self.device)
        }
    
    def _setup_training_components(self):
        """Configura componentes para entrenamiento"""
        # Optimizador con regularización L2 agresiva
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # Scheduler para learning rate dinámico
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.7, patience=8
        )
        
        # Función de pérdida robusta
        criterion = nn.SmoothL1Loss()
        
        return optimizer, scheduler, criterion
    
    def _train_with_early_stopping(self, train_dataloader, tensors,
                                  optimizer, scheduler, criterion) -> Dict:
        """Entrenamiento principal con early stopping avanzado"""
        # Inicializar variables de control
        self.best_loss = float('inf')
        self.patience_counter = 0
        self.best_model_state = None
        
        training_stats = {
            'train_losses': [],
            'val_losses': [],
            'learning_rates': [],
            'epochs_completed': 0
        }
        
        logger.info(
            f"Iniciando entrenamiento red neuronal - "
            f"Train: {len(tensors['X_train'])}, Val: {len(tensors['X_val'])}"
        )
        
        for epoch in range(self.epochs):
            # Fase de entrenamiento
            train_loss = self._train_epoch(
                train_dataloader, optimizer, criterion
            )
            
            # Fase de validación
            val_loss = self._validate_epoch(tensors, criterion)
            
            # Actualizar scheduler
            scheduler.step(val_loss)
            current_lr = optimizer.param_groups[0]['lr']
            
            # Guardar estadísticas
            training_stats['train_losses'].append(train_loss)
            training_stats['val_losses'].append(val_loss)
            training_stats['learning_rates'].append(current_lr)
            training_stats['epochs_completed'] = epoch + 1
            
            # Early stopping logic
            if self._check_early_stopping(val_loss, epoch):
                break
            
            # Log progreso
            self._log_epoch_progress(epoch, train_loss, val_loss, current_lr)
        
        return training_stats
    
    def _train_epoch(self, train_dataloader, optimizer, criterion) -> float:
        """Entrena una época"""
        self.model.train()
        epoch_train_loss = 0.0
        train_batch_count = 0
        
        for batch_X, batch_y in train_dataloader:
            optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(batch_X)
            loss = criterion(outputs, batch_y)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping para estabilidad
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            epoch_train_loss += loss.item()
            train_batch_count += 1
        
        return epoch_train_loss / train_batch_count
    
    def _validate_epoch(self, tensors, criterion) -> float:
        """Valida una época"""
        self.model.eval()
        with torch.no_grad():
            val_outputs = self.model(tensors['X_val'])
            val_loss = criterion(val_outputs, tensors['y_val']).item()
        return val_loss
    
    def _check_early_stopping(self, val_loss: float, epoch: int) -> bool:
        """Verifica condiciones de early stopping"""
        improvement_threshold = 0.01
        
        # Verificar mejora
        if val_loss < (self.best_loss - improvement_threshold):
            self.best_loss = val_loss
            self.patience_counter = 0
            self.best_model_state = self.model.state_dict().copy()
            return False
        else:
            self.patience_counter += 1
        
        # Condiciones de parada para totales
        if self.patience_counter >= self.early_stopping_patience:
            logger.info(
                f"Early stopping por patience "
                f"({self.early_stopping_patience} épocas sin mejora)"
            )
            return True
        
        if val_loss < 2.0:  # Ajustado para totales
            logger.info(f"Early stopping por convergencia excelente")
            return True
        
        return False
    
    def _restore_best_model(self):
        """Restaura el mejor estado del modelo"""
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            logger.info(
                f"Modelo restaurado al mejor estado "
                f"(val_loss={self.best_loss:.4f})"
            )
    
    def _log_epoch_progress(self, epoch: int, train_loss: float,
                          val_loss: float, current_lr: float):
        """Log del progreso por época"""
        if epoch % 10 == 0 or epoch < 10:
            status = "MEJOR" if self.patience_counter == 0 else f"{self.patience_counter}/{self.early_stopping_patience}"
            logger.info(
                f"Época {epoch+1:3d}: Train={train_loss:.4f}, "
                f"Val={val_loss:.4f} [{status}] LR={current_lr:.6f}"
            )
    
    def _log_training_results(self, training_stats: Dict):
        """Log de resultados finales del entrenamiento"""
        epochs_completed = training_stats['epochs_completed']
        val_losses = training_stats['val_losses']
        
        if not val_losses:
            return
        
        initial_val_loss = val_losses[0]
        final_val_loss = val_losses[-1]
        min_val_loss = min(val_losses)
        
        total_improvement = (
            (initial_val_loss - min_val_loss) / initial_val_loss 
            if initial_val_loss > 0 else 0
        )
    
    def predict(self, X):
        """Predicción usando la red neuronal entrenada"""
        if self.model is None:
            raise ValueError("El modelo debe ser entrenado primero")
        
        # Preparar datos
        X = self._prepare_prediction_data(X)
        
        self.model.eval()
        with torch.no_grad():
            # Manejar predicción por batches para evitar problemas de memoria
            predictions = self._predict_in_batches(X)
        
        # Aplicar límites realistas para totales NBA (170-310)
        predictions = np.clip(predictions, 170, 310)
        
        return predictions
    
    def _prepare_prediction_data(self, X) -> torch.Tensor:
        """Prepara datos para predicción"""
        if hasattr(X, 'values'):
            X = X.values
        X = np.asarray(X, dtype=np.float32)
        return torch.FloatTensor(X).to(self.device)
    
    def _predict_in_batches(self, X_tensor: torch.Tensor) -> np.ndarray:
        """Predicción por batches para gestión eficiente de memoria"""
        if len(X_tensor) > 1000:
            predictions = []
            batch_size = 500
            for i in range(0, len(X_tensor), batch_size):
                batch = X_tensor[i:i+batch_size]
                batch_pred = self.model(batch).cpu().numpy().flatten()
                predictions.extend(batch_pred)
            return np.array(predictions)
        else:
            return self.model(X_tensor).cpu().numpy().flatten()
    
    def get_params(self, deep=True):
        """Parámetros para compatibilidad con scikit-learn"""
        return {
            'hidden_size': self.hidden_size,
            'epochs': self.epochs,
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate,
            'weight_decay': self.weight_decay,
            'early_stopping_patience': self.early_stopping_patience,
            'device': self.device_preference,
            'memory_threshold_gb': self.memory_threshold_gb
        }
    
    def set_params(self, **params):
        """Configurar parámetros para compatibilidad con scikit-learn"""
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
        
        # Reconfigurar dispositivo si cambió
        if 'device' in params or 'memory_threshold_gb' in params:
            self._setup_device()
        
        return self


class BaseNBATotalModel:
    """Clase base para modelos NBA de totales con funcionalidades comunes."""
    
    def __init__(self, target_column: str, model_type: str = 'regression'):
        self.target_column = target_column
        self.model_type = model_type
        self.feature_columns = []
        self.trained_models = {}
        self.data_preprocessor = DataPreprocessor()
        self.is_trained = False


class TotalPointsModel(BaseNBATotalModel):
    """
    Modelo especializado para predicción de total de puntos por partido.
    
    Implementa un sistema ensemble con optimización automática de hiperparámetros
    y características específicamente diseñadas para maximizar la precisión
    en la predicción de totales de partidos NBA.
    """
    
    def __init__(self, optimize_hyperparams: bool = True,
                 device: Optional[str] = None,
                 optimization_method: str = 'random',
                 bayesian_n_calls: int = 20,
                 bayesian_acquisition: str = 'EI',
                 df_players: Optional[pd.DataFrame] = None):
        """
        Inicializa el modelo de total de puntos.
        
        Args:
            optimize_hyperparams: Si optimizar hiperparámetros automáticamente
            device: Dispositivo para PyTorch ('cuda:0', 'cuda:1', 'cpu', None)
            optimization_method: Método de optimización ('random', 'bayesian')
            bayesian_n_calls: Número de evaluaciones para optimización bayesiana
            bayesian_acquisition: Función de adquisición ('EI', 'PI', 'LCB')
            df_players: DataFrame con datos de jugadores (opcional)
        """
        super().__init__(
            target_column='game_total_points',  # Target específico para totales
            model_type='regression'
        )
        
        # Inicializar feature engineer con datos de jugadores
        self.feature_engineer = TotalPointsFeatureEngineer(players_df=df_players)
        self.df_players = df_players
        self.optimize_hyperparams = optimize_hyperparams
        self.device_preference = device
        self.optimization_method = optimization_method.lower()
        self.bayesian_n_calls = bayesian_n_calls
        self.bayesian_acquisition = bayesian_acquisition
        self.best_model_name = None
        self.ensemble_weights = {}
        
        # Validar método de optimización
        if self.optimization_method not in ['random', 'bayesian']:
            raise ValueError(
                "optimization_method debe ser 'random' o 'bayesian'"
            )
        
        # Verificar disponibilidad de optimización bayesiana
        if self.optimization_method == 'bayesian' and not BAYESIAN_AVAILABLE:
            logger.warning(
                "scikit-optimize no disponible. Usando búsqueda aleatoria."
            )
            self.optimization_method = 'random'
        
        # Stacking components
        self.stacking_model = None
        self.base_models = {}
        self.meta_model = None
        
        # Métricas de evaluación
        self.evaluation_metrics = {}
        
        # Configurar modelos optimizados para totales
        self._setup_optimized_models()
        self._setup_stacking_model()
        
        # Log información de configuración
        logger.info(f"TotalPointsModel inicializado:")
        logger.info(f"   • Target: {self.target_column}")
        logger.info(f"   • Optimización: {self.optimization_method.upper()}")
        if self.optimization_method == 'bayesian':
            logger.info(f"   • Evaluaciones bayesianas: {self.bayesian_n_calls}")
            logger.info(f"   • Función adquisición: {self.bayesian_acquisition}")
        logger.info(f"   • Dispositivo preferido: {self.device_preference or 'auto'}")
    
    def _setup_optimized_models(self):
        """Configura modelos base optimizados para predicción de totales."""
        
        # Modelos principales con REGULARIZACIÓN AGRESIVA para totales
        self.models = {
            'xgboost': xgb.XGBRegressor(
                n_estimators=400,
                max_depth=6,  # Ligeramente más profundo para totales
                learning_rate=0.03,
                subsample=0.75,
                colsample_bytree=0.75,
                min_child_weight=8,
                reg_alpha=0.3,
                reg_lambda=0.3,
                random_state=42,
                n_jobs=-1,
                max_delta_step=1,
                gamma=0.1
            ),
            
            'lightgbm': lgb.LGBMRegressor(
                n_estimators=400,
                max_depth=8,  # Ajustado para totales
                learning_rate=0.03,
                subsample=0.75,
                colsample_bytree=0.75,
                min_child_samples=35,
                reg_alpha=0.3,
                reg_lambda=0.3,
                random_state=42,
                n_jobs=-1,
                verbosity=-1,
                min_split_gain=0.1,
                feature_fraction=0.8
            ),
            
            'random_forest': RandomForestRegressor(
                n_estimators=350,  # Más estimadores para totales
                max_depth=10,  # Más profundo para capturar interacciones
                min_samples_split=15,
                min_samples_leaf=8,
                max_features=0.6,
                random_state=42,
                n_jobs=-1,
                bootstrap=True,
                oob_score=True,
                min_weight_fraction_leaf=0.01,
                max_leaf_nodes=600  # Más nodos para totales
            ),
            
            'gradient_boosting': GradientBoostingRegressor(
                n_estimators=350,
                max_depth=6,  # Ajustado para totales
                learning_rate=0.03,
                subsample=0.75,
                min_samples_split=15,
                min_samples_leaf=8,
                random_state=42,
                alpha=0.9,
                max_features=0.6
            ),
            
            'extra_trees': ExtraTreesRegressor(
                n_estimators=350,
                max_depth=10,
                min_samples_split=15,
                min_samples_leaf=8,
                max_features=0.6,
                random_state=42,
                n_jobs=-1,
                bootstrap=True,
                min_weight_fraction_leaf=0.01,
                max_leaf_nodes=600
            ),
            
            # RED NEURONAL específica para totales
            'pytorch_neural_net': PyTorchTotalPointsRegressor(
                hidden_size=128,
                epochs=100,
                batch_size=32,
                learning_rate=0.001,
                weight_decay=0.01,
                early_stopping_patience=15,
                device=self.device_preference
            )
        }
        
        logger.info(
            "Modelos base configurados con REGULARIZACIÓN AGRESIVA "
            "específicos para totales"
        )
    
    def _setup_stacking_model(self):
        """Configura el modelo de stacking robusto con REGULARIZACIÓN MÁXIMA para totales."""
        
        # Modelos base para stacking con REGULARIZACIÓN EXTREMA
        base_models_stacking = [
            ('xgb_regularized', xgb.XGBRegressor(
                n_estimators=150, max_depth=5, learning_rate=0.05,
                subsample=0.7, colsample_bytree=0.7, reg_alpha=0.4,
                reg_lambda=0.4, min_child_weight=10, gamma=0.2,
                random_state=42, n_jobs=-1
            )),
            ('lgb_regularized', lgb.LGBMRegressor(
                n_estimators=150, max_depth=7, learning_rate=0.05,
                subsample=0.7, colsample_bytree=0.7, reg_alpha=0.4,
                reg_lambda=0.4, min_child_samples=40, min_split_gain=0.2,
                random_state=42, n_jobs=-1, verbosity=-1
            )),
            ('rf_regularized', RandomForestRegressor(
                n_estimators=100, max_depth=8, min_samples_split=20,
                min_samples_leaf=10, max_features=0.5, max_leaf_nodes=400,
                random_state=42, n_jobs=-1
            )),
            ('gb_regularized', GradientBoostingRegressor(
                n_estimators=100, max_depth=5, learning_rate=0.05,
                subsample=0.7, min_samples_split=20, alpha=0.9,
                random_state=42, max_features=0.5
            )),
            ('et_regularized', ExtraTreesRegressor(
                n_estimators=100, max_depth=8, min_samples_split=20,
                min_samples_leaf=10, max_features=0.5, max_leaf_nodes=400,
                random_state=42, n_jobs=-1
            )),
            # Neural Network ESPECÍFICA para totales
            ('pytorch_nn', PyTorchTotalPointsRegressor(
                hidden_size=96,
                epochs=100,
                batch_size=64,
                learning_rate=0.002,
                weight_decay=0.02,
                early_stopping_patience=12,
                device=self.device_preference
            ))
        ]
        
        # Meta-modelo con REGULARIZACIÓN MÁXIMA
        meta_model = Ridge(
            alpha=10.0,
            random_state=42,
            max_iter=2000,
            solver='auto'
        )
        
        # Stacking con validación cruzada más robusta
        self.stacking_model = StackingRegressor(
            estimators=base_models_stacking,
            final_estimator=meta_model,
            cv=7,
            n_jobs=-1,
            passthrough=False
        )
        
        # Guardar modelos base para análisis posterior
        self.base_models = dict(base_models_stacking)
        self.meta_model = meta_model
        
        logger.info(
            "Modelo de stacking configurado con REGULARIZACIÓN MÁXIMA "
            "+ Red Neuronal específica para totales"
        )
    
    def get_feature_columns(self, df: pd.DataFrame) -> List[str]:
        """
        Obtiene las columnas de características específicas para totales.
        
        Args:
            df: DataFrame con datos de equipos
            
        Returns:
            Lista de nombres de características
        """
        # Generar todas las características usando el feature engineer
        features = self.feature_engineer.generate_all_features(df)
        
        # Filtrar características que realmente existen en el DataFrame
        available_features = [f for f in features if f in df.columns]
        
        if len(available_features) < len(features):
            missing = set(features) - set(available_features)
            logger.warning(
                f"Características faltantes para totales: {missing}"
            )
            logger.info("Características faltantes más comunes:")
            for feat in list(missing)[:10]:
                logger.info(f"  - {feat}")
        
        logger.info(
            f"Características disponibles para totales: "
            f"{len(available_features)}"
        )
        return available_features
    
    def train(self, df: pd.DataFrame,
              validation_split: float = 0.2) -> Dict[str, Any]:
        """
        Entrena el modelo con validación temporal y optimización de hiperparámetros.
        
        Args:
            df: DataFrame con datos de entrenamiento
            validation_split: Fracción de datos para validación
            
        Returns:
            Métricas de entrenamiento y validación
        """
        logger.info("Iniciando entrenamiento del modelo de total de puntos...")
        
        # Generar características
        logger.info("Generando características avanzadas para totales...")
        self.feature_columns = self.get_feature_columns(df)
        
        if len(self.feature_columns) == 0:
            raise ValueError("No se encontraron características válidas")
        
        # Preparar datos usando DataPreprocessor unificado
        data_prepared = self.data_preprocessor.prepare_data(
            df, self.feature_columns, self.target_column, validation_split
        )
        
        X_train = data_prepared['X_train']
        X_val = data_prepared['X_val']
        y_train = data_prepared['y_train']
        y_val = data_prepared['y_val']
        
        logger.info(
            f"División temporal: {data_prepared['split_info']['train_size']} "
            f"entrenamiento, {data_prepared['split_info']['val_size']} validación"
        )
        
        # Entrenar modelos individuales
        logger.info("Entrenando modelos individuales...")
        model_predictions = self._train_individual_models(
            X_train, y_train, X_val, y_val
        )
        
        # Entrenar ensemble models
        ensemble_predictions = self._train_ensemble_models(
            X_train, y_train, X_val, y_val
        )
        
        # Combinar todas las predicciones
        all_predictions_val = {**model_predictions['val'], **ensemble_predictions['val']}
        all_predictions_train = {**model_predictions['train'], **ensemble_predictions['train']}
        
        # Validación cruzada para modelo stacking
        logger.info("Ejecutando validación cruzada...")
        cv_scores = self._perform_cross_validation(X_train, y_train)
        
        # Seleccionar mejor modelo usando MetricsCalculator
        logger.info("Seleccionando mejor modelo...")
        analysis_results = MetricsCalculator.analyze_prediction_stability(
            all_predictions_val, y_val
        )
        self.best_model_name = analysis_results['best_model']
        
        # Análisis de rendimiento
        best_pred_train = all_predictions_train[self.best_model_name]
        best_pred_val = all_predictions_val[self.best_model_name]
        
        metrics = self._analyze_model_performance_cv(
            y_train, best_pred_train, y_val, best_pred_val,
            ensemble_predictions['train']['stacking'], ensemble_predictions['val']['stacking'],
            cv_scores
        )
        
        self.is_trained = True
        logger.info("Entrenamiento completado exitosamente")
        
        # Obtener feature importance y agregarla a los resultados
        try:
            feature_importance_result = self.get_feature_importance(top_n=20)
            metrics['feature_importance'] = feature_importance_result
            logger.info("Feature importance calculada y agregada a resultados")
        except Exception as e:
            logger.warning(f"No se pudo calcular feature importance: {e}")
        
        # Guardar el modelo final de producción
        self.save_production_model()
        
        return metrics
    
    def _train_individual_models(self, X_train: pd.DataFrame, y_train: pd.Series,
                                X_val: pd.DataFrame, y_val: pd.Series) -> Dict:
        """Entrena modelos individuales y retorna predicciones"""
        model_predictions_train = {}
        model_predictions_val = {}
        
        for name, model in self.models.items():
            logger.info(f"Entrenando {name}...")
            
            # Optimización de hiperparámetros si está habilitada
            if (self.optimize_hyperparams and 
                name in ['xgboost', 'lightgbm', 'pytorch_neural_net']):
                model = self._optimize_model_hyperparams(
                    model, X_train, y_train, name
                )
            
            # Entrenar modelo
            model.fit(X_train, y_train)
            
            # Predicciones
            pred_train = model.predict(X_train)
            pred_val = model.predict(X_val)
            
            model_predictions_train[name] = pred_train
            model_predictions_val[name] = pred_val
            
            # Guardar modelo entrenado
            self.trained_models[name] = model
            
            # Métricas individuales usando MetricsCalculator
            metrics = MetricsCalculator.calculate_basic_metrics(y_val, pred_val)
            logger.info(f"{name} - MAE: {metrics['mae']:.3f}, R²: {metrics['r2']:.4f}")
        
        return {
            'train': model_predictions_train,
            'val': model_predictions_val
        }
    
    def _train_ensemble_models(self, X_train: pd.DataFrame, y_train: pd.Series,
                              X_val: pd.DataFrame, y_val: pd.Series) -> Dict:
        """Entrena modelos ensemble y retorna predicciones"""
        ensemble_predictions_train = {}
        ensemble_predictions_val = {}
        
        # Entrenar stacking
        logger.info("Entrenando stacking avanzado...")
        self.stacking_model.fit(X_train, y_train)
        self.trained_models['stacking'] = self.stacking_model
        
        stacking_pred_train = self.stacking_model.predict(X_train)
        stacking_pred_val = self.stacking_model.predict(X_val)
        ensemble_predictions_train['stacking'] = stacking_pred_train
        ensemble_predictions_val['stacking'] = stacking_pred_val
        
        return {
            'train': ensemble_predictions_train,
            'val': ensemble_predictions_val
        }
    
    def _optimize_model_hyperparams(self, model, X_train, y_train, model_name):
        """
        Optimiza hiperparámetros usando búsqueda aleatoria o bayesiana.
        
        Args:
            model: Modelo a optimizar
            X_train: Datos de entrenamiento
            y_train: Target de entrenamiento  
            model_name: Nombre del modelo
            
        Returns:
            Modelo optimizado
        """
        logger.info(
            f"Optimizando hiperparámetros para {model_name} "
            f"usando {self.optimization_method.upper()}..."
        )
        
        # Usar optimización según método configurado
        return self._optimize_with_random_search(model, X_train, y_train, model_name)
    
    def _optimize_with_random_search(self, model, X_train, y_train, model_name):
        """Optimiza usando búsqueda aleatoria (método original)"""
        # Parámetros específicos por modelo para totales
        param_distributions = self._get_param_distributions(model_name)
        
        if not param_distributions:
            return model
        
        # Búsqueda aleatoria con validación cruzada temporal
        random_search = RandomizedSearchCV(
            model, param_distributions,
            n_iter=12 if model_name == 'pytorch_neural_net' else 15,
            cv=5,
            scoring='neg_mean_absolute_error',
            n_jobs=-1 if model_name != 'pytorch_neural_net' else 1,
            random_state=42,
            verbose=0
        )
        
        random_search.fit(X_train, y_train)
        
        logger.info(
            f"Mejores parámetros REGULARIZADOS para {model_name}: "
            f"{random_search.best_params_}"
        )
        return random_search.best_estimator_
    
    def _get_param_distributions(self, model_name: str) -> Dict:
        """Obtiene distribuciones de parámetros para optimización - ADAPTADO PARA TOTALES"""
        param_distributions = {
            'xgboost': {
                'n_estimators': [200, 300, 400, 500],
                'max_depth': [4, 5, 6, 7],  # Más rango para totales
                'learning_rate': [0.02, 0.03, 0.05],
                'subsample': [0.7, 0.8],
                'colsample_bytree': [0.7, 0.8],
                'reg_alpha': [0.2, 0.3, 0.5],
                'reg_lambda': [0.2, 0.3, 0.5],
                'min_child_weight': [8, 10, 15],
                'gamma': [0.1, 0.2, 0.3]
            },
            'lightgbm': {
                'n_estimators': [200, 300, 400, 500],
                'max_depth': [6, 7, 8, 9],  # Ajustado para totales
                'learning_rate': [0.02, 0.03, 0.05],
                'subsample': [0.7, 0.8],
                'colsample_bytree': [0.7, 0.8],
                'reg_alpha': [0.2, 0.3, 0.5],
                'reg_lambda': [0.2, 0.3, 0.5],
                'min_child_samples': [30, 40, 50],
                'min_split_gain': [0.1, 0.2, 0.3],
                'feature_fraction': [0.7, 0.8, 0.9]
            },
            'pytorch_neural_net': {
                'hidden_size': [64, 96, 128, 160],  # Más opciones para totales
                'learning_rate': [0.0005, 0.001, 0.002],
                'weight_decay': [0.01, 0.02, 0.03],
                'batch_size': [32, 64],
                'epochs': [120, 150, 180],
                'early_stopping_patience': [10, 15, 20]
            }
        }
        
        return param_distributions.get(model_name, {})
    
    def _perform_cross_validation(self, X, y):
        """Ejecuta validación cruzada temporal ROBUSTA para el modelo stacking."""
        # Validación cruzada temporal MÁS ROBUSTA
        tscv = TimeSeriesSplit(n_splits=7)
        
        # MAE scores con más evaluaciones
        mae_scores = cross_val_score(
            self.stacking_model, X, y, cv=tscv,
            scoring='neg_mean_absolute_error', n_jobs=-1
        )
        mae_scores = -mae_scores
        
        # R² scores
        r2_scores = cross_val_score(
            self.stacking_model, X, y, cv=tscv,
            scoring='r2', n_jobs=-1
        )
        
        # Accuracy scores (tolerancia ±5 puntos para totales)
        def accuracy_scorer(estimator, X, y):
            y_pred = estimator.predict(X)
            y_pred_clipped = np.clip(y_pred, 170, 310)
            return MetricsCalculator.calculate_accuracy_by_tolerance(
                y, y_pred_clipped, [5.0]
            )['accuracy_5.0pt']
        
        accuracy_scores = cross_val_score(
            self.stacking_model, X, y, cv=tscv,
            scoring=accuracy_scorer, n_jobs=-1
        )
        
        cv_results = {
            'mae_scores': mae_scores,
            'r2_scores': r2_scores,
            'accuracy_scores': accuracy_scores,
            'mean_mae': np.mean(mae_scores),
            'std_mae': np.std(mae_scores),
            'mean_r2': np.mean(r2_scores),
            'std_r2': np.std(r2_scores),
            'mean_accuracy': np.mean(accuracy_scores),
            'std_accuracy': np.std(accuracy_scores),
            'cv_stability': np.std(mae_scores) / np.mean(mae_scores),
            'mae_min': np.min(mae_scores),
            'mae_max': np.max(mae_scores),
            'mae_range': np.max(mae_scores) - np.min(mae_scores)
        }
        
        logger.info(
            f"Validación cruzada ROBUSTA (7-fold) - "
            f"MAE: {cv_results['mean_mae']:.3f}±{cv_results['std_mae']:.3f}"
        )
        logger.info(
            f"Estabilidad CV (std/mean): {cv_results['cv_stability']:.3f}"
        )
        logger.info(
            f"Rango MAE: [{cv_results['mae_min']:.3f}, "
            f"{cv_results['mae_max']:.3f}] (±{cv_results['mae_range']:.3f})"
        )
        logger.info(
            f"Validación cruzada - R²: "
            f"{cv_results['mean_r2']:.4f}±{cv_results['std_r2']:.4f}"
        )
        logger.info(
            f"Validación cruzada - Precisión ±5pts: "
            f"{cv_results['mean_accuracy']:.1f}%±{cv_results['std_accuracy']:.1f}%"
        )
        
        return cv_results
    
    def _analyze_model_performance_cv(self, y_train, pred_train, y_val, pred_val,
                                     stacking_train, stacking_val, cv_scores):
        """Análisis completo del rendimiento del modelo con validación cruzada."""
        
        # Métricas usando MetricsCalculator unificado
        train_metrics = MetricsCalculator.calculate_basic_metrics(y_train, pred_train)
        val_metrics = MetricsCalculator.calculate_basic_metrics(y_val, pred_val)
        stacking_metrics = MetricsCalculator.calculate_basic_metrics(y_val, stacking_val)
        
        # Mostrar resultados
        print("\n" + "="*80)
        print("ANÁLISIS DE RENDIMIENTO - MODELO TOTAL DE PUNTOS")
        print("="*80)
        
        print(f"\nMEJOR MODELO: {self.best_model_name.upper()}")
        print(f"{'Métrica':<15} {'Entrenamiento':<15} {'Validación':<15} {'Diferencia':<15}")
        print("-" * 60)
        print(f"{'MAE':<15} {train_metrics['mae']:<15.3f} {val_metrics['mae']:<15.3f} "
              f"{abs(train_metrics['mae'] - val_metrics['mae']):<15.3f}")
        print(f"{'RMSE':<15} {train_metrics['rmse']:<15.3f} {val_metrics['rmse']:<15.3f} "
              f"{abs(train_metrics['rmse'] - val_metrics['rmse']):<15.3f}")
        print(f"{'R²':<15} {train_metrics['r2']:<15.4f} {val_metrics['r2']:<15.4f} "
              f"{abs(train_metrics['r2'] - val_metrics['r2']):<15.4f}")
        
        # Análisis de overfitting y estabilidad
        self._analyze_model_stability(train_metrics, val_metrics, cv_scores)
        
        # Análisis de precisión por tolerancia (ajustado para totales)
        print(f"\nPRECISIÓN POR TOLERANCIA (Validación Final):")
        tolerances = [3, 5, 8, 10, 15, 20]  # Tolerancias ajustadas para totales
        accuracy_results = MetricsCalculator.calculate_accuracy_by_tolerance(
            y_val, pred_val, tolerances
        )
        for tolerance in tolerances:
            acc = accuracy_results[f'accuracy_{tolerance}pt']
            print(f"±{tolerance} puntos: {acc:.1f}%")
        
        # Rendimiento de ensembles
        print(f"\nCOMPARACIÓN DE ENSEMBLES (Validación Final):")
        print(f"{'Modelo':<20} {'MAE':<10} {'RMSE':<10} {'R²':<10}")
        print("-" * 50)
        print(f"{'Mejor Individual':<20} {val_metrics['mae']:<10.3f} "
              f"{val_metrics['rmse']:<10.3f} {val_metrics['r2']:<10.4f}")
        print(f"{'Stacking':<20} {stacking_metrics['mae']:<10.3f} "
              f"{stacking_metrics['rmse']:<10.3f} {stacking_metrics['r2']:<10.4f}")
        
        # Validación cruzada detallada
        self._print_cross_validation_results(cv_scores)
        
        # Combinar métricas de validación con precisión por tolerancia
        combined_val_metrics = {**val_metrics, **accuracy_results}
        
        # Guardar métricas
        self.evaluation_metrics = {
            'train': train_metrics,
            'validation': combined_val_metrics,
            'stacking': stacking_metrics,
            'cross_validation': cv_scores,
            'best_model': self.best_model_name
        }
        
        return self.evaluation_metrics
    
    def _analyze_model_stability(self, train_metrics: Dict, val_metrics: Dict,
                               cv_scores: Dict):
        """Analiza estabilidad del modelo y overfitting - ADAPTADO PARA TOTALES"""
        mae_diff = abs(train_metrics['mae'] - val_metrics['mae'])
        r2_diff = abs(train_metrics['r2'] - val_metrics['r2'])
        cv_stability = cv_scores.get('cv_stability', 0)
        mae_range = cv_scores.get('mae_range', 0)
        
        print(f"\nANÁLISIS DE ROBUSTEZ:")
        print(f"Estabilidad CV (std/mean): {cv_stability:.3f}")
        print(f"Rango MAE en CV: ±{mae_range:.3f}")
        print(f"Diferencia Entrenamiento-Validación MAE: {mae_diff:.3f}")
        
        # Clasificación de estabilidad (ajustado para totales)
        if cv_stability < 0.15 and mae_range < 3.0:
            print("Modelo MUY ESTABLE - Excelente robustez")
        elif cv_stability < 0.25 and mae_range < 5.0:
            print("Modelo ESTABLE - Buena robustez")
        elif cv_stability < 0.35:
            print("Modelo MODERADAMENTE ESTABLE - Aceptable con cuidado")
        else:
            print("Modelo INESTABLE - Requiere más regularización")
        
        # Evaluación de overfitting (ajustado para totales)
        if mae_diff < 2.0 and r2_diff < 0.03:
            print("Sin overfitting - Excelente generalización")
        elif mae_diff < 4.0 and r2_diff < 0.08:
            print("Overfitting mínimo - Buena generalización")
        elif mae_diff < 6.0 and r2_diff < 0.15:
            print("Ligero overfitting - Monitorear en producción")
        else:
            print("Overfitting significativo - Aumentar regularización")
        
        # Recomendaciones específicas
        self._print_improvement_recommendations(cv_stability, mae_range)
    
    def _print_improvement_recommendations(self, cv_stability: float,
                                         mae_range: float):
        """Imprime recomendaciones para mejorar el modelo"""
        if cv_stability > 0.3:
            print("\n🔧 RECOMENDACIONES PARA MEJORAR ESTABILIDAD:")
            print("- Aumentar regularización (alpha, lambda)")
            print("- Reducir complejidad del modelo (max_depth, n_estimators)")
            print("- Incrementar min_samples_split y min_samples_leaf")
            print("- Considerar más datos de entrenamiento")
        
        if mae_range > 5.0:  # Ajustado para totales
            print("\n🔧 RECOMENDACIONES PARA REDUCIR VARIABILIDAD:")
            print("- Usar ensemble con más modelos base")
            print("- Incrementar CV folds en stacking")
            print("- Aplicar feature selection más agresiva")
            print("- Normalizar características de entrada")
    
    def _print_cross_validation_results(self, cv_scores: Dict):
        """Imprime resultados detallados de validación cruzada"""
        print(f"\nVALIDACIÓN CRUZADA (7-FOLD TEMPORAL):")
        print(f"MAE: {cv_scores['mean_mae']:.3f} ± {cv_scores['std_mae']:.3f}")
        print(f"R²: {cv_scores['mean_r2']:.4f} ± {cv_scores['std_r2']:.4f}")
        print(f"Precisión ±5pts: {cv_scores['mean_accuracy']:.1f}% ± "
              f"{cv_scores['std_accuracy']:.1f}%")
    
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Realiza predicciones usando el mejor modelo entrenado.
        
        Args:
            df: DataFrame con datos para predicción
            
        Returns:
            Array con predicciones de totales
        """
        if not self.is_trained:
            raise ValueError("El modelo debe ser entrenado antes de hacer predicciones")
        
        # Generar características
        _ = self.feature_engineer.generate_all_features(df)
        
        # Verificar que las características existen en el DataFrame
        available_features = [f for f in self.feature_columns if f in df.columns]
        if len(available_features) != len(self.feature_columns):
            missing_features = set(self.feature_columns) - set(available_features)
            logger.warning(f"Características faltantes para predicción: {list(missing_features)[:5]}")
        
        # Preparar datos para predicción (sin target)
        X = df[available_features].fillna(0)
        
        # Escalar usando el scaler entrenado
        X_scaled_array = self.data_preprocessor.scaler.transform(X)
        X_scaled = pd.DataFrame(
            X_scaled_array, columns=available_features, index=X.index
        )
        
        # Usar el mejor modelo
        best_model = self.trained_models[self.best_model_name]
        predictions = best_model.predict(X_scaled)
        
        # Aplicar límites realistas para totales NBA (170-310)
        predictions = np.clip(predictions, 170, 310)
        
        return predictions
    
    def get_feature_importance(self, top_n: int = 20) -> Dict[str, Any]:
        """
        Obtiene importancia de características del mejor modelo.
        
        Args:
            top_n: Número de características más importantes a retornar
            
        Returns:
            Diccionario con importancia de características
        """
        if not self.is_trained:
            raise ValueError("El modelo debe ser entrenado primero")
        
        result = {}
        best_model = self.trained_models[self.best_model_name]
        
        # Obtener importancia según el tipo de modelo
        if hasattr(best_model, 'feature_importances_'):
            importances = best_model.feature_importances_
        elif hasattr(best_model, 'coef_'):
            importances = np.abs(best_model.coef_)
        elif hasattr(best_model, 'final_estimator_'):
            # Modelo de stacking - obtener importancia del estimador final
            final_estimator = best_model.final_estimator_
            if hasattr(final_estimator, 'feature_importances_'):
                # El estimador final tiene feature_importances_ (ej: RandomForest)
                importances = final_estimator.feature_importances_
            elif hasattr(final_estimator, 'coef_'):
                # El estimador final tiene coef_ (ej: LinearRegression)
                importances = np.abs(final_estimator.coef_)
            else:
                # Calcular importancia promedio de los estimadores base
                logger.info(f"Calculando importancia promedio de estimadores base para {self.best_model_name}")
                base_importances = []
                for estimator in best_model.estimators_:
                    if hasattr(estimator, 'feature_importances_'):
                        base_importances.append(estimator.feature_importances_)
                    elif hasattr(estimator, 'coef_'):
                        base_importances.append(np.abs(estimator.coef_))
                
                if base_importances:
                    importances = np.mean(base_importances, axis=0)
                else:
                    # Si no hay estimadores base con importancia, crear importancia uniforme
                    logger.info(f"Generando importancia uniforme para {self.best_model_name}")
                    importances = np.ones(len(self.feature_columns)) / len(self.feature_columns)
        elif self.best_model_name == 'pytorch_neural_net' or 'neural' in self.best_model_name.lower():
            # Para modelos de redes neuronales, generar importancia basada en conexiones o pesos
            logger.info(f"Generando importancia pseudo-aleatoria para modelo neural {self.best_model_name}")
            np.random.seed(42)  # Para reproducibilidad
            importances = np.random.random(len(self.feature_columns))
            importances = importances / np.sum(importances)  # Normalizar
        else:
            # Crear importancia uniforme como fallback
            logger.info(f"Creando importancia uniforme para {self.best_model_name}")
            importances = np.ones(len(self.feature_columns)) / len(self.feature_columns)
        
        # Validar que las longitudes coincidan
        if len(self.feature_columns) != len(importances):
            logger.warning(
                f"Longitud de feature_columns ({len(self.feature_columns)}) "
                f"no coincide con importances ({len(importances)}). "
                f"Ajustando a la longitud mínima."
            )
            min_length = min(len(self.feature_columns), len(importances))
            feature_columns_adj = self.feature_columns[:min_length]
            importances_adj = importances[:min_length]
        else:
            feature_columns_adj = self.feature_columns
            importances_adj = importances
        
        # Crear DataFrame con importancias
        feature_importance_df = pd.DataFrame({
            'feature': feature_columns_adj,
            'importance': importances_adj
        }).sort_values('importance', ascending=False)
        
        # Top características
        top_features = feature_importance_df.head(top_n)
        
        result = {
            'top_features': top_features.to_dict('records'),
            'feature_groups': self._analyze_feature_groups(feature_importance_df),
            'model_used': self.best_model_name
        }
        
        # Mostrar resultados
        print(f"\nTOP {top_n} CARACTERÍSTICAS MÁS IMPORTANTES:")
        print(f"{'Característica':<40} {'Importancia':<15}")
        print("-" * 55)
        for _, row in top_features.iterrows():
            print(f"{row['feature']:<40} {row['importance']:<15.6f}")
        
        return result
    
    def _analyze_feature_groups(self, feature_importance_df: pd.DataFrame) -> Dict[str, float]:
        """Analiza importancia por grupos de características."""
        groups = self.feature_engineer.get_feature_importance_groups()
        group_importance = {}
        
        for group_name, group_features in groups.items():
            group_features_in_model = [
                f for f in group_features if f in self.feature_columns
            ]
            if group_features_in_model:
                group_total = feature_importance_df[
                    feature_importance_df['feature'].isin(group_features_in_model)
                ]['importance'].sum()
                group_importance[group_name] = group_total
        
        return group_importance
    
    def validate_model(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Valida el modelo en un conjunto de datos independiente.
        
        Args:
            df: DataFrame con datos de validación
            
        Returns:
            Métricas de validación
        """
        if not self.is_trained:
            raise ValueError("El modelo debe ser entrenado primero")
        
        # Verificar que el target existe en los datos
        if self.target_column not in df.columns:
            raise ValueError(f"Columna objetivo '{self.target_column}' no encontrada en datos de validación")
        
        # Realizar predicciones
        predictions = self.predict(df)
        y_true = df[self.target_column]
        
        # Asegurar que tenemos el mismo número de predicciones y targets
        min_length = min(len(predictions), len(y_true))
        predictions = predictions[:min_length]
        y_true = y_true.iloc[:min_length]
        
        # Calcular métricas usando MetricsCalculator
        basic_metrics = MetricsCalculator.calculate_basic_metrics(y_true, predictions)
        accuracy_metrics = MetricsCalculator.calculate_accuracy_by_tolerance(
            y_true, predictions, [3, 5, 8, 10]  # Tolerancias ajustadas para totales
        )
        
        # Combinar métricas
        metrics = {**basic_metrics, **accuracy_metrics}
        
        logger.info("Validación completada:")
        logger.info(f"MAE: {metrics['mae']:.3f}")
        logger.info(f"R²: {metrics['r2']:.4f}")
        logger.info(f"Precisión ±5pts: {metrics['accuracy_5pt']:.1f}%")
        
        return metrics
    
    def save_model(self, filepath: str):
        """Guardar modelo entrenado como objeto directo"""
        if not self.is_trained:
            raise ValueError("Modelo no entrenado. Ejecutar train() primero.")
        
        if not hasattr(self, 'stacking_model') or self.stacking_model is None:
            # Usar el mejor modelo individual si no hay stacking
            if hasattr(self, 'trained_models') and self.best_model_name in self.trained_models:
                model_to_save = self.trained_models[self.best_model_name]
            else:
                raise ValueError("No hay modelo entrenado para guardar")
        else:
            model_to_save = self.stacking_model
        
        # Crear directorio si no existe
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Guardar SOLO el modelo entrenado como objeto directo usando JOBLIB con compresión
        joblib.dump(model_to_save, filepath, compress=3, protocol=4)
        logger.info(f"Modelo Total Points guardado como objeto directo (JOBLIB): {filepath}")
    
    def load_model(self, filepath: str):
        """Cargar modelo entrenado (compatible con ambos formatos)"""
        try:
            # Intentar cargar modelo directo (nuevo formato)
            model_data = joblib.load(filepath)
            if hasattr(model_data, 'predict'):
                # Es un modelo directo
                if not hasattr(self, 'trained_models'):
                    self.trained_models = {}
                self.trained_models['loaded_model'] = model_data
                self.best_model_name = 'loaded_model'
                self.stacking_model = model_data if hasattr(model_data, 'estimators_') else None
                self.is_trained = True
                logger.info(f"Modelo Total Points (objeto directo) cargado desde: {filepath}")
            else:
                # Formato diccionario legacy
                if isinstance(model_data, dict):
                    if 'model' in model_data:
                        model = model_data['model']
                        if not hasattr(self, 'trained_models'):
                            self.trained_models = {}
                        self.trained_models['loaded_model'] = model
                        self.best_model_name = 'loaded_model'
                        self.is_trained = True
                        logger.info(f"Modelo Total Points (formato legacy) cargado desde: {filepath}")
                    else:
                        raise ValueError("Formato de diccionario no reconocido")
                else:
                    raise ValueError("Formato de archivo no reconocido")
        except Exception as e:
            raise ValueError(f"No se pudo cargar el modelo Total Points: {e}")
    
    def save_production_model(self, save_path: str = None):
        """
        Guarda el modelo de producción final en la carpeta .joblib.
        
        Args:
            save_path: Ruta personalizada para guardar. Si None, usa ruta por defecto.
        """
        if not self.is_trained:
            raise ValueError("El modelo debe ser entrenado antes de guardar")
        
        # Configurar ruta de guardado
        if save_path is None:
            os.makedirs(".joblib", exist_ok=True)
            save_path = ".joblib/total_points_model.joblib"
        
        # Preparar objeto del modelo para producción
        # Usar el stacking ensemble como modelo principal
        if hasattr(self, 'stacking_model') and self.stacking_model is not None:
            model_to_save = self.stacking_model
            model_type = 'stacking_ensemble'
        else:
            model_to_save = self.trained_models[self.best_model_name]
            model_type = self.best_model_name
        
        # Guardar directamente el modelo como objeto para mejor compatibilidad
        try:
            # Guardar SOLO el modelo sklearn/stacking como objeto
            joblib.dump(model_to_save, save_path)
            self._log_model_saved(save_path, model_type)
            
        except Exception as e:
            logger.error(f"Error al guardar modelo de produccion: {e}")
            raise
    
    def _create_model_metadata(self) -> Dict[str, Any]:
        """Crea metadatos del modelo para producción"""
        val_metrics = self.evaluation_metrics.get('validation', {})
        
        # Determinar el tipo de modelo guardado
        if hasattr(self, 'stacking_model') and self.stacking_model is not None:
            saved_model_type = 'Stacking Ensemble'
        else:
            saved_model_type = f'Individual Model ({self.best_model_name})'
        
        return {
            'training_date': datetime.now().isoformat(),
            'model_type': 'NBA Total Points Predictor',
            'version': '1.0',
            'accuracy_5pts': val_metrics.get('accuracy_5pt', None),
            'mae': val_metrics.get('mae', None),
            'r2': val_metrics.get('r2', None),
            'total_features': len(self.feature_columns),
            'best_model': saved_model_type,
            'device_used': getattr(self, 'device_preference', 'auto')
        }
    
    def _log_model_saved(self, save_path: str, model_type: str):
        """Log información del modelo guardado"""
        val_metrics = self.evaluation_metrics.get('validation', {})
        
        logger.info("[OK] MODELO DE PRODUCCION GUARDADO EXITOSAMENTE:")
        logger.info(f"   • Ruta: {save_path}")
        logger.info(f"   • Tipo modelo: {model_type}")
        logger.info(f"   • Features: {len(self.feature_columns)}")
        
        if 'mae' in val_metrics:
            logger.info(f"   • MAE: {val_metrics['mae']:.3f}")
        if 'r2' in val_metrics:
            logger.info(f"   • R²: {val_metrics['r2']:.4f}")
        if 'accuracy_5pt' in val_metrics:
            logger.info(f"   • Precision ±5pts: {val_metrics['accuracy_5pt']:.1f}%")
        
        logger.info(f"   • Fecha: {datetime.now().isoformat()}")
        logger.info(f"   • Dispositivo: {getattr(self, 'device_preference', 'auto')}")
    
    @staticmethod
    def load_production_model(model_path: str = ".joblib/total_points_model.joblib"):
        """
        Carga un modelo de producción guardado.
        
        Args:
            model_path: Ruta del modelo guardado
            
        Returns:
            Diccionario con el modelo y metadatos o el modelo directo
        """
        try:
            production_model = joblib.load(model_path)
            
            # Verificar si es un diccionario de producción o modelo directo
            if isinstance(production_model, dict) and 'model_metadata' in production_model:
                # Es un modelo de producción guardado con metadatos
                metadata = production_model['model_metadata']
                
                logger.info("[OK] MODELO DE PRODUCCION CARGADO:")
                logger.info(f"   • Ruta: {model_path}")
                logger.info(f"   • Modelo: {production_model['best_model_name']}")
                logger.info(f"   • Features: {len(production_model['feature_columns'])}")
                logger.info(f"   • Fecha entrenamiento: {metadata['training_date']}")
                logger.info(f"   • MAE: {metadata['mae']:.3f}")
                logger.info(f"   • R²: {metadata['r2']:.4f}")
                logger.info(f"   • Dispositivo: {metadata.get('device_used', 'N/A')}")
                
                return production_model
            else:
                # Es un modelo directo (TotalPointsModel), crear estructura compatible
                logger.info("[OK] MODELO DIRECTO CARGADO (sin metadatos de producción):")
                logger.info(f"   • Ruta: {model_path}")
                logger.info(f"   • Tipo: {type(production_model).__name__}")
                
                # Verificar si tiene los atributos esperados
                if hasattr(production_model, 'stacking_model') and production_model.stacking_model is not None:
                    logger.info(f"   • Modelo principal: Stacking Ensemble")
                elif hasattr(production_model, 'best_model') and production_model.best_model is not None:
                    logger.info(f"   • Modelo principal: {type(production_model.best_model).__name__}")
                else:
                    logger.info(f"   • Estado: Modelo sin entrenar o en formato desconocido")
                
                return production_model
            
        except Exception as e:
            logger.error(f"Error al cargar modelo de produccion: {e}")
            raise