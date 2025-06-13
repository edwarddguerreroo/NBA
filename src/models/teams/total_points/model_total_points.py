"""
Modelo Avanzado de Predicción de Puntos Totales NBA (PTS + PTS_Opp)
===================================================================

Este módulo implementa un sistema de predicción de alto rendimiento para
puntos totales en partidos NBA utilizando:

1. Ensemble Learning con múltiples algoritmos ML y Red Neuronal
2. Stacking avanzado con meta-modelo optimizado
3. Optimización Bayesiana de hiperparámetros
4. Validación cruzada temporal rigurosa
5. Métricas de evaluación exhaustivas
6. Feature engineering especializado para totales

Objetivo: 97%+ de precisión en predicción de puntos totales
"""

# Standard Library
import os
import pickle
import warnings
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
import copy
import sys
from tqdm import tqdm

# Third-party Libraries - ML/Data
import joblib
import numpy as np
import pandas as pd

# Scikit-learn
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.ensemble import (
    ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor,
    StackingRegressor, VotingRegressor, AdaBoostRegressor, BaggingRegressor
)
from sklearn.linear_model import (
    ElasticNet, Lasso, Ridge, HuberRegressor, 
    BayesianRidge, ARDRegression, PassiveAggressiveRegressor
)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import (
    GridSearchCV, KFold, RandomizedSearchCV, TimeSeriesSplit, 
    cross_val_score, cross_validate, train_test_split
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.compose import TransformedTargetRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.svm import SVR, NuSVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, RationalQuadratic, DotProduct
from sklearn.neural_network import MLPRegressor

# XGBoost, LightGBM and CatBoost
import xgboost as xgb
import lightgbm as lgb

# Verificar disponibilidad de CatBoost
CATBOOST_AVAILABLE = False
try:
    import catboost as cat
    CATBOOST_AVAILABLE = True
except ImportError:
    cat = None
    CATBOOST_AVAILABLE = False

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
    warnings.warn("scikit-optimize no disponible. Instalarlo para optimización Bayesiana.")

# Optuna (alternativa moderna a scikit-optimize)
try:
    import optuna
    from optuna.samplers import TPESampler
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

# Local imports
from .features_total_points import TotalPointsFeatureEngine

# Configuration
warnings.filterwarnings('ignore')

# Logging setup
import logging
# No configuramos logging.basicConfig aquí para evitar sobrescribir la configuración global
logger = logging.getLogger(__name__)
# Solo establecemos el nivel para este logger específico
logger.setLevel(logging.INFO)


class GPUManager:
    """Gestión inteligente de recursos GPU para entrenamiento eficiente"""
    
    @staticmethod
    def get_optimal_device(prefer_gpu: bool = True, 
                          memory_threshold_gb: float = 2.0) -> torch.device:
        """
        Selecciona el dispositivo óptimo para entrenamiento.
        
        Args:
            prefer_gpu: Si preferir GPU cuando esté disponible
            memory_threshold_gb: Memoria mínima requerida en GB
            
        Returns:
            torch.device optimizado
        """
        if not prefer_gpu:
            return torch.device('cpu')
        
        if not torch.cuda.is_available():
            logger.info("CUDA no disponible, usando CPU")
            return torch.device('cpu')
        
        # Buscar GPU con suficiente memoria
        for i in range(torch.cuda.device_count()):
            device = torch.device(f'cuda:{i}')
            props = torch.cuda.get_device_properties(i)
            total_memory_gb = props.total_memory / (1024**3)
            
            if total_memory_gb >= memory_threshold_gb:
                logger.info(f"Usando GPU {i}: {props.name} ({total_memory_gb:.1f}GB)")
                return device
        
        logger.info("GPUs sin memoria suficiente, usando CPU")
        return torch.device('cpu')


class TotalPointsNeuralNet(nn.Module):
    """
    Red Neuronal Profunda Optimizada para Predicción de Puntos Totales NBA
    
    Arquitectura especializada para capturar patrones complejos en totales:
    - Múltiples capas con diferentes tamaños
    - Batch Normalization y Dropout para regularización
    - Skip connections para mejor flujo de gradientes
    - Activaciones mixtas para no-linealidad
    """
    
    def __init__(self, input_size: int, hidden_sizes: List[int] = None,
                 dropout_rates: List[float] = None):
        super(TotalPointsNeuralNet, self).__init__()
        
        if hidden_sizes is None:
            hidden_sizes = [256, 128, 64, 32]
        if dropout_rates is None:
            dropout_rates = [0.3, 0.4, 0.3, 0.2]
        
        self.layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        
        # Construir capas
        prev_size = input_size
        for i, (hidden_size, dropout_rate) in enumerate(zip(hidden_sizes, dropout_rates)):
            self.layers.append(nn.Linear(prev_size, hidden_size))
            self.batch_norms.append(nn.BatchNorm1d(hidden_size))
            self.dropouts.append(nn.Dropout(dropout_rate))
            prev_size = hidden_size
        
        # Skip connections
        self.skip1 = nn.Linear(input_size, hidden_sizes[1])
        self.skip2 = nn.Linear(hidden_sizes[1], hidden_sizes[-1])
        
        # Capa de salida
        self.output = nn.Linear(hidden_sizes[-1], 1)
        
        # Inicialización optimizada
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Inicialización de pesos optimizada para regresión de totales"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # He initialization para ReLU
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """Forward pass con arquitectura compleja"""
        # Guardar para skip connections
        input_skip = x
        
        # Primera mitad de la red
        for i in range(len(self.layers) // 2):
            x = self.layers[i](x)
            x = self.batch_norms[i](x)
            x = F.relu(x)
            x = self.dropouts[i](x)
        
        # Primera skip connection
        x = x + self.skip1(input_skip)
        skip_mid = x
        
        # Segunda mitad de la red
        for i in range(len(self.layers) // 2, len(self.layers)):
            x = self.layers[i](x)
            x = self.batch_norms[i](x)
            x = F.leaky_relu(x, 0.1)  # Leaky ReLU para evitar neuronas muertas
            x = self.dropouts[i](x)
        
        # Segunda skip connection
        x = x + self.skip2(skip_mid)
        
        # Salida
        x = self.output(x)
        
        return x


class PyTorchTotalPointsRegressor(BaseEstimator, RegressorMixin):
    """
    Wrapper de PyTorch para integración perfecta con scikit-learn
    
    Implementa una red neuronal profunda optimizada específicamente
    para predicción de puntos totales con regularización avanzada.
    """
    
    def __init__(self, hidden_sizes: List[int] = None, dropout_rates: List[float] = None,
                 epochs: int = 200, batch_size: int = 64, learning_rate: float = 0.001,
                 weight_decay: float = 0.01, early_stopping_patience: int = 20,
                 lr_scheduler: str = 'cosine', gradient_clip: float = 1.0,
                 device: str = 'auto', random_state: int = 42):
        
        self.hidden_sizes = hidden_sizes or [256, 128, 64, 32]
        self.dropout_rates = dropout_rates or [0.3, 0.4, 0.3, 0.2]
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.early_stopping_patience = early_stopping_patience
        self.lr_scheduler = lr_scheduler
        self.gradient_clip = gradient_clip
        self.device = device
        self.random_state = random_state
        
        # Componentes del modelo
        self.model = None
        self.scaler = StandardScaler()
        self._device = None
        self.training_history = []
        
        # Para early stopping
        self.best_loss = float('inf')
        self.patience_counter = 0
        self.best_model_state = None
        
        # Configurar semillas para reproducibilidad
        self._set_random_seeds()
    
    def _set_random_seeds(self):
        """Configura semillas para reproducibilidad"""
        np.random.seed(self.random_state)
        torch.manual_seed(self.random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.random_state)
    
    def fit(self, X, y):
        """Entrena el modelo con validación y early stopping"""
        # Preparar dispositivo
        if self.device == 'auto':
            self._device = GPUManager.get_optimal_device()
        else:
            self._device = torch.device(self.device)
        
        # Preparar datos
        X = self._validate_input(X)
        y = self._validate_target(y)
        
        # Escalar características
        X_scaled = self.scaler.fit_transform(X)
        
        # División train/validation
        val_size = 0.15
        n_val = int(len(X) * val_size)
        indices = np.random.permutation(len(X))
        
        train_idx, val_idx = indices[n_val:], indices[:n_val]
        
        X_train = X_scaled[train_idx]
        y_train = y[train_idx]
        X_val = X_scaled[val_idx]
        y_val = y[val_idx]
        
        # Convertir a tensores
        X_train_tensor = torch.FloatTensor(X_train).to(self._device)
        y_train_tensor = torch.FloatTensor(y_train.reshape(-1, 1)).to(self._device)
        X_val_tensor = torch.FloatTensor(X_val).to(self._device)
        y_val_tensor = torch.FloatTensor(y_val.reshape(-1, 1)).to(self._device)
        
        # Crear modelo
        input_size = X.shape[1]
        self.model = TotalPointsNeuralNet(
            input_size, self.hidden_sizes, self.dropout_rates
        ).to(self._device)
        
        # Configurar optimizador y scheduler
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        if self.lr_scheduler == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.epochs
            )
        elif self.lr_scheduler == 'reduce':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', patience=10, factor=0.5
            )
        else:
            scheduler = None
        
        # Loss function
        criterion = nn.MSELoss()
        
        # Crear DataLoader
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True
        )
        
        # Entrenamiento
        self.training_history = []
        
        for epoch in range(self.epochs):
            # Training
            self.model.train()
            train_loss = 0.0
            
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                
                loss.backward()
                
                # Gradient clipping
                if self.gradient_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.gradient_clip
                    )
                
                optimizer.step()
                train_loss += loss.item() * batch_X.size(0)
            
            train_loss /= len(X_train)
            
            # Validation
            self.model.eval()
            with torch.no_grad():
                val_outputs = self.model(X_val_tensor)
                val_loss = criterion(val_outputs, y_val_tensor).item()
            
            # Guardar historial
            self.training_history.append({
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'lr': optimizer.param_groups[0]['lr']
            })
            
            # Learning rate scheduling
            if scheduler:
                if self.lr_scheduler == 'reduce':
                    scheduler.step(val_loss)
                else:
                    scheduler.step()
            
            # Early stopping
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                self.patience_counter = 0
                self.best_model_state = copy.deepcopy(self.model.state_dict())
            else:
                self.patience_counter += 1
                
            if self.patience_counter >= self.early_stopping_patience:
                logger.info(f"Early stopping en época {epoch}")
                break
            
            # Log progreso
            if epoch % 10 == 0:
                logger.info(
                    f"Época {epoch}/{self.epochs} - "
                    f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
                )
        
        # Restaurar mejor modelo
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
        
        return self
    
    def predict(self, X):
        """Realiza predicciones"""
        if self.model is None:
            raise ValueError("El modelo debe ser entrenado primero")
        
        X = self._validate_input(X)
        X_scaled = self.scaler.transform(X)
        X_tensor = torch.FloatTensor(X_scaled).to(self._device)
        
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(X_tensor).cpu().numpy().flatten()
        
        return predictions
    
    def _validate_input(self, X):
        """Valida y convierte entrada a numpy array"""
        if isinstance(X, pd.DataFrame):
            X = X.values
        elif not isinstance(X, np.ndarray):
            X = np.array(X)
        
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        return X
    
    def _validate_target(self, y):
        """Valida y convierte target a numpy array"""
        if isinstance(y, pd.Series):
            y = y.values
        elif not isinstance(y, np.ndarray):
            y = np.array(y)
        
        return y.flatten()


class BlendingRegressor(BaseEstimator, RegressorMixin):
    """
    Implementación avanzada de Blending para regresión.
    
    Combina múltiples modelos base usando una estrategia de ponderación optimizada
    que puede adaptarse a diferentes conjuntos de datos y escenarios.
    """
    
    def __init__(self, base_models=None, weights=None, weight_optimization='auto', 
                 meta_model=None, cv=5, random_state=None):
        """
        Inicializa el BlendingRegressor.
        
        Args:
            base_models: Lista de modelos base (instancias de sklearn o compatibles)
            weights: Pesos para cada modelo base (None para optimización automática)
            weight_optimization: Método de optimización de pesos ('auto', 'grid', 'bayesian', None)
            meta_model: Modelo meta para combinar predicciones (None para usar Ridge)
            cv: Número de folds para validación cruzada en optimización
            random_state: Semilla para reproducibilidad
        """
        self.base_models = base_models or []
        self.weights = weights
        self.weight_optimization = weight_optimization
        self.meta_model = meta_model
        self.cv = cv
        self.random_state = random_state
        
        # Atributos internos
        self._fitted_models = None
        self._optimized_weights = None
        self._meta_model_fitted = None
        self._model_names = None
        
    def fit(self, X, y):
        """
        Entrena el modelo de blending completo.
        
        Args:
            X: Features de entrenamiento
            y: Target de entrenamiento
            
        Returns:
            self: Instancia entrenada
        """
        if len(self.base_models) == 0:
            raise ValueError("Se requiere al menos un modelo base")
        
        # Guardar nombres de modelos para logging
        self._model_names = [
            type(model).__name__ if hasattr(model, '__class__') else str(model)
            for model in self.base_models
        ]
        
        # Entrenar modelos base
        self._fitted_models = []
        for i, model in enumerate(self.base_models):
            try:
                model_clone = clone(model)
                model_clone.fit(X, y)
                self._fitted_models.append(model_clone)
            except Exception as e:
                logger.warning(f"Error entrenando modelo {self._model_names[i]}: {str(e)}")
                # Usar un modelo simple como fallback
                from sklearn.linear_model import Ridge
                fallback = Ridge(alpha=1.0, random_state=self.random_state)
                fallback.fit(X, y)
                self._fitted_models.append(fallback)
        
        # Generar predicciones de modelos base
        base_predictions = np.column_stack([
            model.predict(X) for model in self._fitted_models
        ])
        
        # Optimizar pesos si es necesario
        if self.weights is None or self.weight_optimization:
            self._optimize_weights(X, y, base_predictions)
        else:
            # Usar pesos proporcionados
            if len(self.weights) != len(self._fitted_models):
                raise ValueError(f"Número de pesos ({len(self.weights)}) no coincide con número de modelos ({len(self._fitted_models)})")
            self._optimized_weights = np.array(self.weights) / np.sum(self.weights)  # Normalizar
        
        # Entrenar meta-modelo si se proporciona
        if self.meta_model is not None:
            self._meta_model_fitted = clone(self.meta_model)
            self._meta_model_fitted.fit(base_predictions, y)
        
        return self
    
    def _optimize_weights(self, X, y, base_predictions):
        """
        Optimiza los pesos para cada modelo base.
        
        Args:
            X: Features de entrenamiento
            y: Target de entrenamiento
            base_predictions: Predicciones de modelos base
        """
        n_models = len(self._fitted_models)
        
        if self.weight_optimization == 'grid' or (self.weight_optimization == 'auto' and n_models <= 3):
            # Optimización por grid search para pocos modelos
            self._grid_search_weights(base_predictions, y)
        elif self.weight_optimization == 'bayesian' or (self.weight_optimization == 'auto' and n_models > 3):
            # Optimización bayesiana para muchos modelos
            self._bayesian_optimize_weights(base_predictions, y)
        else:
            # Pesos basados en rendimiento individual (fallback)
            self._performance_based_weights(X, y)
    
    def _grid_search_weights(self, base_predictions, y):
        """Optimiza pesos mediante búsqueda en cuadrícula"""
        n_models = base_predictions.shape[1]
        
        # Para 2 modelos, podemos hacer una búsqueda lineal simple
        if n_models == 2:
            best_mae = float('inf')
            best_weight = 0.5
            
            for w1 in np.linspace(0, 1, 101):  # 0.00, 0.01, ..., 1.00
                w2 = 1 - w1
                pred = w1 * base_predictions[:, 0] + w2 * base_predictions[:, 1]
                mae = mean_absolute_error(y, pred)
                
                if mae < best_mae:
                    best_mae = mae
                    best_weight = w1
            
            self._optimized_weights = np.array([best_weight, 1 - best_weight])
            logger.info(f"Pesos optimizados por grid search: {self._optimized_weights}")
            
        # Para 3 modelos, usamos una cuadrícula 2D
        elif n_models == 3:
            best_mae = float('inf')
            best_weights = np.ones(3) / 3
            
            for w1 in np.linspace(0, 1, 11):  # 0.0, 0.1, ..., 1.0
                for w2 in np.linspace(0, 1 - w1, int(11 * (1 - w1)) + 1):
                    w3 = 1 - w1 - w2
                    if w3 < 0:
                        continue
                    
                    weights = np.array([w1, w2, w3])
                    pred = base_predictions.dot(weights)
                    mae = mean_absolute_error(y, pred)
                    
                    if mae < best_mae:
                        best_mae = mae
                        best_weights = weights
            
            self._optimized_weights = best_weights
            logger.info(f"Pesos optimizados por grid search: {self._optimized_weights}")
            
        else:
            # Para más de 3 modelos, usamos optimización convexa
            from scipy.optimize import minimize
            
            def objective(weights):
                # Normalizar pesos para que sumen 1
                weights = weights / np.sum(weights)
                pred = base_predictions.dot(weights)
                return mean_absolute_error(y, pred)
            
            # Inicializar con pesos iguales
            initial_weights = np.ones(n_models) / n_models
            
            # Restricciones: pesos no negativos que suman 1
            bounds = [(0, 1) for _ in range(n_models)]
            
            # Optimizar
            result = minimize(
                objective,
                initial_weights,
                bounds=bounds,
                method='SLSQP',
                options={'disp': False}
            )
            
            # Normalizar y guardar pesos optimizados
            self._optimized_weights = result.x / np.sum(result.x)
            logger.info(f"Pesos optimizados por optimización convexa: {self._optimized_weights}")
    
    def _bayesian_optimize_weights(self, base_predictions, y):
        """Optimiza pesos mediante optimización bayesiana"""
        try:
            from skopt import gp_minimize
            from skopt.space import Real
            
            n_models = base_predictions.shape[1]
            
            # Definir espacio de búsqueda
            space = [Real(0.0, 1.0, name=f'w{i}') for i in range(n_models)]
            
            # Función objetivo
            def objective(weights):
                # Normalizar pesos para que sumen 1
                weights = np.array(weights) / np.sum(weights)
                pred = base_predictions.dot(weights)
                return mean_absolute_error(y, pred)
            
            # Optimizar
            res = gp_minimize(
                objective,
                space,
                n_calls=50,
                random_state=self.random_state,
                verbose=False
            )
            
            # Normalizar y guardar pesos optimizados
            self._optimized_weights = np.array(res.x) / np.sum(res.x)
            logger.info(f"Pesos optimizados por optimización bayesiana: {self._optimized_weights}")
            
        except ImportError:
            logger.warning("scikit-optimize no disponible, usando pesos basados en rendimiento")
            self._performance_based_weights(None, y)
    
    def _performance_based_weights(self, X, y):
        """Asigna pesos basados en el rendimiento individual de cada modelo"""
        n_models = len(self._fitted_models)
        
        if X is not None and y is not None:
            # Usar validación cruzada para evaluar modelos
            from sklearn.model_selection import KFold
            
            kf = KFold(n_splits=self.cv, shuffle=True, random_state=self.random_state)
            mae_scores = np.zeros(n_models)
            
            for model_idx, model in enumerate(self._fitted_models):
                cv_maes = []
                
                for train_idx, val_idx in kf.split(X):
                    X_train, X_val = X[train_idx], X[val_idx]
                    y_train, y_val = y[train_idx], y[val_idx]
                    
                    model_clone = clone(model)
                    model_clone.fit(X_train, y_train)
                    pred = model_clone.predict(X_val)
                    cv_maes.append(mean_absolute_error(y_val, pred))
                
                mae_scores[model_idx] = np.mean(cv_maes)
            
            # Convertir MAE a pesos (menor MAE = mayor peso)
            # Usar transformación inversa y softmax
            weights = 1.0 / (mae_scores + 1e-10)
            self._optimized_weights = weights / np.sum(weights)
            
        else:
            # Fallback a pesos iguales
            self._optimized_weights = np.ones(n_models) / n_models
        
        logger.info(f"Pesos basados en rendimiento: {self._optimized_weights}")
    
    def predict(self, X):
        """
        Genera predicciones combinando modelos base con pesos optimizados.
        
        Args:
            X: Features para predicción
            
        Returns:
            array: Predicciones combinadas
        """
        if self._fitted_models is None:
            raise ValueError("El modelo no ha sido entrenado. Llamar a fit() primero.")
        
        # Generar predicciones de modelos base
        base_predictions = np.column_stack([
            model.predict(X) for model in self._fitted_models
        ])
        
        # Si hay un meta-modelo, usarlo
        if self._meta_model_fitted is not None:
            return self._meta_model_fitted.predict(base_predictions)
        
        # De lo contrario, usar combinación ponderada
        return base_predictions.dot(self._optimized_weights)
    
    def get_model_weights(self):
        """Retorna los pesos optimizados y nombres de modelos"""
        if self._optimized_weights is None:
            return None
        
        return dict(zip(self._model_names, self._optimized_weights))


class NBATotalPointsPredictor:
    """
    Sistema completo de predicción de puntos totales NBA
    
    Combina múltiples algoritmos de ML con stacking avanzado,
    optimización Bayesiana y validación temporal rigurosa.
    """
    
    def __init__(self, optimize_hyperparams: bool = True,
                 optimization_method: str = 'bayesian',
                 n_optimization_trials: int = 50,
                 use_neural_network: bool = True,
                 device: str = 'auto',
                 random_state: int = 42):
        """
        Inicializa el predictor de puntos totales.
        
        Args:
            optimize_hyperparams: Si optimizar hiperparámetros automáticamente
            optimization_method: Método de optimización ('bayesian', 'optuna', 'random')
            n_optimization_trials: Número de trials para optimización
            use_neural_network: Si incluir red neuronal en el ensemble
            device: Dispositivo para red neuronal ('auto', 'cpu', 'cuda')
            random_state: Semilla para reproducibilidad
        """
        self.optimize_hyperparams = optimize_hyperparams
        self.optimization_method = optimization_method
        self.n_optimization_trials = n_optimization_trials
        self.use_neural_network = use_neural_network
        self.device = device
        self.random_state = random_state
        
        # Componentes del modelo
        self.feature_engineer = TotalPointsFeatureEngine()
        self.models = {}
        self.optimized_models = {}
        self.ensemble_models = {}
        self.feature_columns = None
        self.scaler = StandardScaler()
        
        # Resultados de entrenamiento
        self.training_results = {}
        self.feature_importance = {}
        self.optimization_history = {}
        
        # Configurar modelos
        self._setup_models()
    
    def _setup_models(self):
        """Configura los modelos base y ensemble"""
        self.models = {}
        self.optimized_models = {}
        self.ensemble_models = {}
        
        # Configurar semilla aleatoria
        np.random.seed(self.random_state)
        
        # Modelos base - Regresores lineales
        self.models['ridge'] = Ridge(alpha=1.0, random_state=self.random_state)
        self.models['lasso'] = Lasso(alpha=0.01, random_state=self.random_state)
        self.models['elastic_net'] = ElasticNet(alpha=0.01, l1_ratio=0.5, random_state=self.random_state)
        self.models['bayesian_ridge'] = BayesianRidge()
        self.models['huber'] = HuberRegressor(epsilon=1.35)
        
        # Modelos base - Árboles y boosting
        self.models['random_forest'] = RandomForestRegressor(
            n_estimators=200, max_depth=15, random_state=self.random_state, n_jobs=-1
        )
        self.models['extra_trees'] = ExtraTreesRegressor(
            n_estimators=200, max_depth=15, random_state=self.random_state, n_jobs=-1
        )
        
        # Modelos avanzados con hiperparámetros mejorados
        self.models['xgboost'] = xgb.XGBRegressor(
            n_estimators=300, 
            max_depth=8,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.5,
            reg_lambda=1.0,
            random_state=self.random_state,
            n_jobs=-1
        )
        
        self.models['lightgbm'] = lgb.LGBMRegressor(
            n_estimators=300,
            num_leaves=50,
            learning_rate=0.05,
            feature_fraction=0.8,
            bagging_fraction=0.8,
            min_child_samples=20,
            random_state=self.random_state,
            n_jobs=-1,
            verbose=-1
        )
        
        if CATBOOST_AVAILABLE:
            self.models['catboost'] = cat.CatBoostRegressor(
                iterations=300,
                depth=8,
                learning_rate=0.05,
                l2_leaf_reg=3,
                random_state=self.random_state,
                verbose=False
            )
        
        # Red neuronal
        if self.use_neural_network:
            self.models['neural_network'] = PyTorchTotalPointsRegressor(
                hidden_sizes=[256, 192, 128, 64],
                dropout_rates=[0.3, 0.3, 0.2, 0.2],
                epochs=250,
                batch_size=128,
                learning_rate=0.001,
                weight_decay=0.01,
                early_stopping_patience=25,
                lr_scheduler='cosine',
                gradient_clip=1.0,
                device=self.device,
                random_state=self.random_state
            )
        
        # Modelos ensemble
        self.ensemble_models['voting'] = VotingRegressor(
            estimators=[
                ('xgboost', self.models.get('xgboost')),
                ('lightgbm', self.models.get('lightgbm')),
                ('random_forest', self.models.get('random_forest')),
                ('ridge', self.models.get('ridge'))
            ]
        )
        
        # Stacking con meta-regressor optimizado
        self.ensemble_models['stacking'] = StackingRegressor(
            estimators=[
                ('xgboost', self.models.get('xgboost')),
                ('lightgbm', self.models.get('lightgbm')),
                ('catboost', self.models.get('catboost')) if 'catboost' in self.models else 
                            ('extra_trees', self.models.get('extra_trees')),
                ('random_forest', self.models.get('random_forest')),
                ('neural_network', self.models.get('neural_network')) if 'neural_network' in self.models else 
                                ('huber', self.models.get('huber')),
                ('ridge', self.models.get('ridge'))
            ],
            final_estimator=xgb.XGBRegressor(
                n_estimators=200,
                max_depth=5,
                learning_rate=0.03,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=self.random_state
            ),
            cv=5,
            n_jobs=-1
        )
        
        # Blending personalizado con calibración
        base_models = [
            self.models.get('xgboost'),
            self.models.get('lightgbm'),
            self.models.get('random_forest'),
            self.models.get('ridge'),
            self.models.get('huber')
        ]
        
        # Filtrar modelos None
        base_models = [model for model in base_models if model is not None]
        
        # Crear blending con calibración isotónica
        self.ensemble_models['blending'] = BlendingRegressor(base_models, self.random_state)
    
    def train(self, df_teams: pd.DataFrame, df_players: pd.DataFrame = None,
              validation_split: float = 0.2) -> Dict[str, Any]:
        """
        Entrena el sistema completo de predicción.
        
        Args:
            df_teams: DataFrame con datos de equipos
            df_players: DataFrame con datos de jugadores (opcional)
            validation_split: Proporción de datos para validación
            
        Returns:
            Diccionario con resultados de entrenamiento
        """
        logger.info("="*80)
        logger.info("INICIANDO ENTRENAMIENTO DE MODELO DE PUNTOS TOTALES")
        logger.info("="*80)
        
        # Crear features
        logger.info("\n[1/6] Generando features...")
        df_features = self.feature_engineer.create_features(df_teams, df_players)
        
        # Preparar datos
        logger.info("\n[2/6] Preparando datos para entrenamiento...")
        X, y = self.feature_engineer.prepare_features(df_features, target_col='total_points')
        self.feature_columns = list(X.columns)
        
        # División temporal
        split_idx = int(len(X) * (1 - validation_split))
        X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]
        
        logger.info(f"Datos de entrenamiento: {X_train.shape}")
        logger.info(f"Datos de validación: {X_val.shape}")
        
        # Escalar datos
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        # Entrenar modelos individuales
        logger.info("\n[3/6] Entrenando modelos individuales...")
        individual_results = self._train_individual_models(
            X_train_scaled, y_train, X_val_scaled, y_val
        )
        
        # Optimización de hiperparámetros
        if self.optimize_hyperparams:
            logger.info("\n[4/6] Optimizando hiperparámetros...")
            self._optimize_hyperparameters(X_train_scaled, y_train)
        
        # Entrenar modelos ensemble
        logger.info("\n[5/6] Entrenando modelos ensemble...")
        ensemble_results = self._train_ensemble_models(
            X_train_scaled, y_train, X_val_scaled, y_val
        )
        
        # Validación cruzada temporal
        logger.info("\n[6/6] Realizando validación cruzada temporal...")
        cv_results = self._perform_temporal_cross_validation(X, y)
        
        # Compilar resultados
        self.training_results = {
            'individual_models': individual_results,
            'ensemble_models': ensemble_results,
            'cross_validation': cv_results,
            'feature_importance': self._calculate_feature_importance(),
            'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Generar reporte
        self._generate_training_report()
        
        return self.training_results
    
    def _train_individual_models(self, X_train, y_train, X_val, y_val) -> Dict:
        """Entrena todos los modelos individuales"""
        results = {}
        
        for name, model in tqdm(self.models.items(), desc="Entrenando modelos"):
            try:
                # Entrenar modelo
                if name == 'neural_network':
                    # La red neuronal espera arrays numpy
                    model.fit(X_train, y_train.values)
                else:
                    # Para LightGBM, no pasamos feature_name para evitar advertencias
                    model.fit(X_train, y_train)
                
                # Predicciones
                train_pred = model.predict(X_train)
                val_pred = model.predict(X_val)
                
                # Métricas
                results[name] = {
                    'train_mae': mean_absolute_error(y_train, train_pred),
                    'train_rmse': np.sqrt(mean_squared_error(y_train, train_pred)),
                    'train_r2': r2_score(y_train, train_pred),
                    'val_mae': mean_absolute_error(y_val, val_pred),
                    'val_rmse': np.sqrt(mean_squared_error(y_val, val_pred)),
                    'val_r2': r2_score(y_val, val_pred),
                    'predictions': val_pred
                }
                
                # Guardar modelo optimizado
                self.optimized_models[name] = model
                
            except Exception as e:
                logger.error(f"Error entrenando {name}: {str(e)}")
                results[name] = {'error': str(e)}
        
        return results
    
    def _optimize_hyperparameters(self, X_train, y_train):
        """Optimiza hiperparámetros de los mejores modelos"""
        
        # Seleccionar mejores modelos para optimizar
        best_models = ['xgboost', 'lightgbm', 'catboost', 'random_forest']
        
        for model_name in best_models:
            if model_name not in self.models:
                continue
            
            logger.info(f"Optimizando {model_name}...")
            
            if self.optimization_method == 'bayesian' and BAYESIAN_AVAILABLE:
                optimized = self._bayesian_optimization(
                    model_name, X_train, y_train
                )
            elif self.optimization_method == 'optuna' and OPTUNA_AVAILABLE:
                optimized = self._optuna_optimization(
                    model_name, X_train, y_train
                )
            else:
                optimized = self._random_search_optimization(
                    model_name, X_train, y_train
                )
            
            if optimized is not None:
                self.optimized_models[model_name] = optimized
    
    def _bayesian_optimization(self, model_name: str, X_train, y_train):
        """Optimización Bayesiana de hiperparámetros"""
        
        # Definir espacio de búsqueda
        if model_name == 'xgboost':
            space = [
                Real(0.01, 0.3, name='learning_rate'),
                Integer(100, 500, name='n_estimators'),
                Integer(3, 10, name='max_depth'),
                Real(0.5, 1.0, name='subsample'),
                Real(0.5, 1.0, name='colsample_bytree'),
                Real(0.0, 10.0, name='reg_alpha'),
                Real(0.0, 10.0, name='reg_lambda')
            ]
        elif model_name == 'lightgbm':
            space = [
                Real(0.01, 0.3, name='learning_rate'),
                Integer(100, 500, name='n_estimators'),
                Integer(20, 100, name='num_leaves'),
                Real(0.5, 1.0, name='feature_fraction'),
                Real(0.5, 1.0, name='bagging_fraction'),
                Integer(5, 30, name='min_child_samples')
            ]
        elif model_name == 'catboost':
            space = [
                Real(0.01, 0.3, name='learning_rate'),
                Integer(100, 500, name='iterations'),
                Integer(4, 10, name='depth'),
                Real(0.0, 10.0, name='l2_leaf_reg')
            ]
        elif model_name == 'random_forest':
            space = [
                Integer(100, 500, name='n_estimators'),
                Integer(10, 50, name='max_depth'),
                Integer(2, 20, name='min_samples_split'),
                Integer(1, 10, name='min_samples_leaf'),
                Real(0.5, 1.0, name='max_features')
            ]
        else:
            return None
        
        # Función objetivo
        @use_named_args(space)
        def objective(**params):
            # Crear modelo con parámetros
            if model_name == 'xgboost':
                model = xgb.XGBRegressor(
                    **params,
                    random_state=self.random_state,
                    n_jobs=-1
                )
            elif model_name == 'lightgbm':
                # Para LightGBM, convertir X_train a DataFrame para mantener nombres de características
                if not isinstance(X_train, pd.DataFrame):
                    X_train_df = pd.DataFrame(X_train, columns=self.feature_columns)
                else:
                    X_train_df = X_train
                
                model = lgb.LGBMRegressor(
                    **params,
                    random_state=self.random_state,
                    n_jobs=-1,
                    verbose=-1
                )
                
                # Validación cruzada para evitar overfitting
                cv = KFold(n_splits=3, shuffle=True, random_state=self.random_state)
                cv_scores = cross_val_score(
                    model, X_train_df, y_train, 
                    cv=cv, scoring='neg_mean_absolute_error', 
                    n_jobs=-1
                )
                
                # Retornar el MAE negativo promedio (para minimizar)
                return -cv_scores.mean()
                
            elif model_name == 'catboost':
                model = cat.CatBoostRegressor(
                    **params,
                    random_seed=self.random_state,
                    verbose=False
                )
            elif model_name == 'random_forest':
                model = RandomForestRegressor(
                    **params,
                    random_state=self.random_state,
                    n_jobs=-1
                )
            else:
                return float('inf')
            
            # Validación cruzada para evitar overfitting
            if model_name != 'lightgbm':  # Ya manejado para LightGBM
                cv = KFold(n_splits=3, shuffle=True, random_state=self.random_state)
                cv_scores = cross_val_score(
                    model, X_train, y_train, 
                    cv=cv, scoring='neg_mean_absolute_error', 
                    n_jobs=-1
                )
                
                # Retornar el MAE negativo promedio (para minimizar)
                return -cv_scores.mean()
        
        # Ejecutar optimización
        try:
            logger.info(f"Iniciando optimización bayesiana para {model_name}...")
            
            result = gp_minimize(
                objective,
                space,
                n_calls=self.n_optimization_trials,
                random_state=self.random_state,
                verbose=False
            )
            
            # Extraer mejores parámetros
            best_params = dict(zip([dim.name for dim in space], result.x))
            logger.info(f"Mejores parámetros para {model_name}: {best_params}")
            
            # Crear modelo optimizado
            if model_name == 'xgboost':
                optimized_model = xgb.XGBRegressor(
                    **best_params,
                    random_state=self.random_state,
                    n_jobs=-1
                )
            elif model_name == 'lightgbm':
                optimized_model = lgb.LGBMRegressor(
                    **best_params,
                    random_state=self.random_state,
                    n_jobs=-1,
                    verbose=-1
                )
            elif model_name == 'catboost':
                optimized_model = cat.CatBoostRegressor(
                    **best_params,
                    random_seed=self.random_state,
                    verbose=False
                )
            elif model_name == 'random_forest':
                optimized_model = RandomForestRegressor(
                    **best_params,
                    random_state=self.random_state,
                    n_jobs=-1
                )
            else:
                return None
            
            # Entrenar modelo con todos los datos
            if model_name == 'lightgbm':
                # Para LightGBM, convertir X_train a DataFrame para mantener nombres de características
                if not isinstance(X_train, pd.DataFrame):
                    X_train_df = pd.DataFrame(X_train, columns=self.feature_columns)
                else:
                    X_train_df = X_train
                
                optimized_model.fit(X_train_df, y_train)
            else:
                optimized_model.fit(X_train, y_train)
            
            # Guardar historial de optimización
            self.optimization_history[model_name] = {
                'best_params': best_params,
                'best_score': -result.fun,
                'all_scores': [-y for y in result.func_vals],
                'n_trials': self.n_optimization_trials
            }
            
            return optimized_model
            
        except Exception as e:
            logger.error(f"Error en optimización bayesiana para {model_name}: {str(e)}")
            return None
    
    def _optuna_optimization(self, model_name: str, X_train, y_train):
        """Optimización con Optuna (más moderno que scikit-optimize)"""
        
        def objective(trial):
            if model_name == 'xgboost':
                params = {
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                    'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 10.0),
                    'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 10.0)
                }
                model = xgb.XGBRegressor(**params, random_state=self.random_state, n_jobs=-1)
            
            elif model_name == 'lightgbm':
                params = {
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                    'num_leaves': trial.suggest_int('num_leaves', 20, 100),
                    'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
                    'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0),
                    'min_child_samples': trial.suggest_int('min_child_samples', 5, 30)
                }
                model = lgb.LGBMRegressor(**params, random_state=self.random_state, n_jobs=-1, verbose=-1, feature_name='auto')
            
            else:
                return None
            
            # Validación cruzada
            cv_scores = cross_val_score(
                model, X_train, y_train,
                cv=TimeSeriesSplit(n_splits=3),
                scoring='neg_mean_absolute_error',
                n_jobs=-1
            )
            
            return -cv_scores.mean()
        
        # Crear estudio
        study = optuna.create_study(
            direction='minimize',
            sampler=TPESampler(seed=self.random_state)
        )
        
        # Optimizar
        study.optimize(
            objective,
            n_trials=self.n_optimization_trials,
            show_progress_bar=True
        )
        
        # Crear modelo con mejores parámetros
        best_params = study.best_params
        
        if model_name == 'xgboost':
            optimized_model = xgb.XGBRegressor(
                **best_params,
                random_state=self.random_state,
                n_jobs=-1
            )
        elif model_name == 'lightgbm':
            optimized_model = lgb.LGBMRegressor(
                **best_params,
                random_state=self.random_state,
                n_jobs=-1,
                verbose=-1,
                feature_name='auto'  # Usar nombres automáticos para evitar advertencias
            )
        
        # Entrenar
        optimized_model.fit(X_train, y_train)
        
        # Guardar historial
        self.optimization_history[model_name] = {
            'best_params': best_params,
            'best_score': -study.best_value,
            'n_trials': len(study.trials)
        }
        
        return optimized_model
    
    def _random_search_optimization(self, model_name: str, X_train, y_train):
        """Optimización con búsqueda aleatoria como fallback"""
        
        # Definir distribuciones de parámetros
        if model_name == 'xgboost':
            param_dist = {
                'learning_rate': [0.01, 0.05, 0.1, 0.2, 0.3],
                'n_estimators': [100, 200, 300, 400, 500],
                'max_depth': [3, 5, 7, 9],
                'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
                'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0]
            }
        elif model_name == 'lightgbm':
            param_dist = {
                'learning_rate': [0.01, 0.05, 0.1, 0.2, 0.3],
                'n_estimators': [100, 200, 300, 400, 500],
                'num_leaves': [20, 31, 50, 70, 100],
                'feature_fraction': [0.6, 0.7, 0.8, 0.9, 1.0],
                'bagging_fraction': [0.6, 0.7, 0.8, 0.9, 1.0]
            }
        else:
            return None
        
        # Crear modelo base
        if model_name == 'xgboost':
            base_model = xgb.XGBRegressor(random_state=self.random_state, n_jobs=-1)
        elif model_name == 'lightgbm':
            base_model = lgb.LGBMRegressor(random_state=self.random_state, n_jobs=-1, verbose=-1, feature_name='auto')
        
        # Búsqueda aleatoria
        random_search = RandomizedSearchCV(
            base_model,
            param_distributions=param_dist,
            n_iter=min(self.n_optimization_trials, 20),
            cv=TimeSeriesSplit(n_splits=3),
            scoring='neg_mean_absolute_error',
            n_jobs=-1,
            random_state=self.random_state
        )
        
        random_search.fit(X_train, y_train)
        
        # Guardar historial
        self.optimization_history[model_name] = {
            'best_params': random_search.best_params_,
            'best_score': -random_search.best_score_
        }
        
        return random_search.best_estimator_
    
    def _train_ensemble_models(self, X_train, y_train, X_val, y_val) -> Dict:
        """Entrena modelos ensemble avanzados"""
        results = {}
        
        # Seleccionar mejores modelos para ensemble
        best_models = []
        for name in ['xgboost', 'lightgbm', 'catboost', 'random_forest', 'extra_trees']:
            if name in self.optimized_models:
                best_models.append((name, self.optimized_models[name]))
        
        # Voting Regressor
        logger.info("Entrenando Voting Regressor...")
        voting_regressor = VotingRegressor(estimators=best_models)
        voting_regressor.fit(X_train, y_train)
        
        voting_pred_train = voting_regressor.predict(X_train)
        voting_pred_val = voting_regressor.predict(X_val)
        
        results['voting'] = {
            'train_mae': mean_absolute_error(y_train, voting_pred_train),
            'val_mae': mean_absolute_error(y_val, voting_pred_val),
            'train_rmse': np.sqrt(mean_squared_error(y_train, voting_pred_train)),
            'val_rmse': np.sqrt(mean_squared_error(y_val, voting_pred_val)),
            'train_r2': r2_score(y_train, voting_pred_train),
            'val_r2': r2_score(y_val, voting_pred_val)
        }
        
        self.ensemble_models['voting'] = voting_regressor
        
        # Stacking Regressor
        logger.info("Entrenando Stacking Regressor...")
        
        # Meta-learner
        meta_learner = xgb.XGBRegressor(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=3,
            random_state=self.random_state
        )
        
        stacking_regressor = StackingRegressor(
            estimators=best_models,
            final_estimator=meta_learner,
            cv=3  # Para generar predicciones out-of-fold
        )
        
        stacking_regressor.fit(X_train, y_train)
        
        stacking_pred_train = stacking_regressor.predict(X_train)
        stacking_pred_val = stacking_regressor.predict(X_val)
        
        results['stacking'] = {
            'train_mae': mean_absolute_error(y_train, stacking_pred_train),
            'val_mae': mean_absolute_error(y_val, stacking_pred_val),
            'train_rmse': np.sqrt(mean_squared_error(y_train, stacking_pred_train)),
            'val_rmse': np.sqrt(mean_squared_error(y_val, stacking_pred_val)),
            'train_r2': r2_score(y_train, stacking_pred_train),
            'val_r2': r2_score(y_val, stacking_pred_val)
        }
        
        self.ensemble_models['stacking'] = stacking_regressor
        
        # Blending personalizado con calibración
        logger.info("Creando Blending Ensemble...")
        
        # Calcular pesos basados en MAE de validación
        model_scores = {}
        for name, result in self.training_results.get('individual_models', {}).items():
            if 'val_mae' in result:
                model_scores[name] = 1 / (result['val_mae'] + 1e-6)
        
        # Normalizar pesos
        total_score = sum(model_scores.values())
        weights = {name: score/total_score for name, score in model_scores.items()}
        
        # Crear predicciones ponderadas
        blend_pred_val = np.zeros_like(y_val)
        for name, weight in weights.items():
            if name in self.optimized_models:
                pred = self.optimized_models[name].predict(X_val)
                blend_pred_val += weight * pred
        
        results['blending'] = {
            'val_mae': mean_absolute_error(y_val, blend_pred_val),
            'val_rmse': np.sqrt(mean_squared_error(y_val, blend_pred_val)),
            'val_r2': r2_score(y_val, blend_pred_val),
            'weights': weights
        }
        
        return results
    
    def _perform_temporal_cross_validation(self, X, y) -> Dict:
        """Realiza validación cruzada temporal"""
        
        cv_results = {}
        
        # Modelos a evaluar
        models_to_evaluate = {
            'xgboost': self.optimized_models.get('xgboost', self.models.get('xgboost')),
            'lightgbm': self.optimized_models.get('lightgbm', self.models.get('lightgbm')),
            'stacking': self.ensemble_models.get('stacking')
        }
        
        # Filtrar modelos None
        models_to_evaluate = {name: model for name, model in models_to_evaluate.items() 
                             if model is not None}
        
        # Configurar validación cruzada temporal
        tscv = TimeSeriesSplit(n_splits=5)
        
        for name, model in models_to_evaluate.items():
            logger.info(f"Validación cruzada temporal para {name}...")
            
            try:
                # Definir métricas a evaluar
                scoring = {
                    'neg_mean_absolute_error': 'neg_mean_absolute_error',
                    'neg_root_mean_squared_error': 'neg_root_mean_squared_error',
                    'r2': 'r2'
                }
                
                # Preparar datos para LightGBM si es necesario
                if name == 'lightgbm' and not isinstance(X, pd.DataFrame):
                    X_df = pd.DataFrame(X, columns=self.feature_columns)
                    cv_scores = cross_validate(
                        model, X_df, y, cv=tscv, scoring=scoring, 
                        return_train_score=True, n_jobs=-1
                    )
                else:
                    # Ejecutar validación cruzada
                    cv_scores = cross_validate(
                        model, X, y, cv=tscv, scoring=scoring, 
                        return_train_score=True, n_jobs=-1
                    )
                
                # Guardar resultados
                cv_results[name] = {
                    'train_mae': -cv_scores['train_neg_mean_absolute_error'].mean(),
                    'val_mae': -cv_scores['test_neg_mean_absolute_error'].mean(),
                    'train_rmse': -cv_scores['train_neg_root_mean_squared_error'].mean(),
                    'val_rmse': -cv_scores['test_neg_root_mean_squared_error'].mean(),
                    'train_r2': cv_scores['train_r2'].mean(),
                    'val_r2': cv_scores['test_r2'].mean(),
                    'mae_std': cv_scores['test_neg_mean_absolute_error'].std(),
                    'rmse_std': cv_scores['test_neg_root_mean_squared_error'].std(),
                    'r2_std': cv_scores['test_r2'].std()
                }
            except Exception as e:
                logger.error(f"Error en validación cruzada para {name}: {str(e)}")
                cv_results[name] = {'error': str(e)}
        
        return cv_results
    
    def _calculate_feature_importance(self) -> Dict:
        """Calcula importancia de características"""
        importance_dict = {}
        
        # Modelos basados en árboles
        tree_models = ['xgboost', 'lightgbm', 'catboost', 'random_forest', 'extra_trees']
        
        for name in tree_models:
            if name in self.optimized_models:
                model = self.optimized_models[name]
                
                if hasattr(model, 'feature_importances_'):
                    importance_dict[name] = pd.DataFrame({
                        'feature': self.feature_columns,
                        'importance': model.feature_importances_
                    }).sort_values('importance', ascending=False)
        
        # Importancia promedio
        if importance_dict:
            avg_importance = pd.DataFrame({'feature': self.feature_columns})
            avg_importance['importance'] = 0
            
            for name, df in importance_dict.items():
                feature_imp = df.set_index('feature')['importance']
                avg_importance['importance'] += feature_imp.reindex(avg_importance['feature']).fillna(0)
            
            avg_importance['importance'] /= len(importance_dict)
            avg_importance = avg_importance.sort_values('importance', ascending=False)
            
            importance_dict['average'] = avg_importance
        
        return importance_dict
    
    def _generate_training_report(self):
        """Genera reporte detallado de entrenamiento"""
        
        logger.info("\n" + "="*80)
        logger.info("REPORTE DE ENTRENAMIENTO - MODELO DE PUNTOS TOTALES")
        logger.info("="*80)
        
        # Mejores modelos individuales
        if 'individual_models' in self.training_results:
            logger.info("\nMEJORES MODELOS INDIVIDUALES:")
            
            model_performance = []
            for name, metrics in self.training_results['individual_models'].items():
                if 'val_mae' in metrics:
                    model_performance.append({
                        'model': name,
                        'val_mae': metrics['val_mae'],
                        'val_rmse': metrics['val_rmse'],
                        'val_r2': metrics['val_r2']
                    })
            
            model_performance.sort(key=lambda x: x['val_mae'])
            
            for i, perf in enumerate(model_performance[:5]):
                logger.info(
                    f"{i+1}. {perf['model']:15s} - "
                    f"MAE: {perf['val_mae']:.2f}, "
                    f"RMSE: {perf['val_rmse']:.2f}, "
                    f"R²: {perf['val_r2']:.3f}"
                )
        
        # Modelos ensemble
        if 'ensemble_models' in self.training_results:
            logger.info("\nMODELOS ENSEMBLE:")
            
            for name, metrics in self.training_results['ensemble_models'].items():
                if 'val_mae' in metrics:
                    logger.info(
                        f"{name:15s} - "
                        f"MAE: {metrics['val_mae']:.2f}, "
                        f"RMSE: {metrics['val_rmse']:.2f}, "
                        f"R²: {metrics['val_r2']:.3f}"
                    )
        
        # Validación cruzada
        if 'cross_validation' in self.training_results:
            logger.info("\nVALIDACIÓN CRUZADA TEMPORAL:")
            
            for name, metrics in self.training_results['cross_validation'].items():
                logger.info(
                    f"{name:15s} - "
                    f"MAE: {metrics['val_mae']:.2f} (±{metrics['mae_std']:.2f}), "
                    f"R²: {metrics['val_r2']:.3f} (±{metrics['r2_std']:.3f})"
                )
        
        # Top features
        if 'feature_importance' in self.training_results:
            if 'average' in self.training_results['feature_importance']:
                logger.info("\nTOP 10 CARACTERÍSTICAS MÁS IMPORTANTES:")
                
                top_features = self.training_results['feature_importance']['average'].head(10)
                for idx, row in top_features.iterrows():
                    logger.info(f"{row['feature']:40s} - {row['importance']:.4f}")
        
        logger.info("="*80 + "\n")
    
    def predict(self, df_teams: pd.DataFrame, df_players: pd.DataFrame = None,
                model_name: str = 'stacking') -> np.ndarray:
        """
        Realiza predicciones usando el modelo especificado.
        
        Args:
            df_teams: DataFrame con datos de equipos
            df_players: DataFrame con datos de jugadores (opcional)
            model_name: Nombre del modelo a usar ('stacking', 'voting', 'xgboost', etc.)
            
        Returns:
            Array con predicciones
        """
        # Generar features
        df_features = self.feature_engineer.create_features(df_teams, df_players)
        
        # Preparar datos
        X = df_features[self.feature_columns] if self.feature_columns else df_features
        X_scaled = self.scaler.transform(X)
        
        # Convertir a DataFrame para mantener nombres de características
        X_scaled_df = pd.DataFrame(X_scaled, columns=self.feature_columns)
        
        # Seleccionar modelo
        if model_name in self.ensemble_models:
            model = self.ensemble_models[model_name]
        elif model_name in self.optimized_models:
            model = self.optimized_models[model_name]
        elif model_name in self.models:
            model = self.models[model_name]
        else:
            raise ValueError(f"Modelo '{model_name}' no encontrado")
        
        # Realizar predicción
        if model_name == 'lightgbm':
            predictions = model.predict(X_scaled_df)
        else:
            predictions = model.predict(X_scaled_df)  # Usar DataFrame para todos para consistencia
        
        return predictions
    
    def save_model(self, filepath: str = 'models/total_points_predictor.pkl'):
        """Guarda el modelo completo"""
        
        # Crear directorio si no existe
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Preparar objeto para guardar
        model_data = {
            'feature_engineer': self.feature_engineer,
            'feature_columns': self.feature_columns,
            'scaler': self.scaler,
            'optimized_models': self.optimized_models,
            'ensemble_models': self.ensemble_models,
            'training_results': self.training_results,
            'optimization_history': self.optimization_history,
            'metadata': {
                'version': '1.0',
                'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'n_features': len(self.feature_columns) if self.feature_columns else 0
            }
        }
        
        # Guardar
        joblib.dump(model_data, filepath)
        logger.info(f"Modelo guardado en: {filepath}")
    
    @staticmethod
    def load_model(filepath: str = 'models/total_points_predictor.pkl'):
        """Carga un modelo guardado"""
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Archivo no encontrado: {filepath}")
        
        # Cargar datos
        model_data = joblib.load(filepath)
        
        # Crear instancia
        predictor = NBATotalPointsPredictor()
        
        # Restaurar componentes
        predictor.feature_engineer = model_data['feature_engineer']
        predictor.feature_columns = model_data['feature_columns']
        predictor.scaler = model_data['scaler']
        predictor.optimized_models = model_data['optimized_models']
        predictor.ensemble_models = model_data['ensemble_models']
        predictor.training_results = model_data['training_results']
        predictor.optimization_history = model_data.get('optimization_history', {})
        
        logger.info(f"Modelo cargado desde: {filepath}")
        
        return predictor
