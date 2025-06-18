"""
Módulo de Deep Learning para Predicción de Puntos (PTS)
=======================================================

Este módulo contiene arquitecturas avanzadas de Deep Learning especializadas
en la predicción de puntos en la NBA, incluyendo:

- Transformers para secuencias temporales
- Graph Neural Networks para relaciones jugador-equipo
- LSTM bidireccionales con attention
- Variational Autoencoders para feature learning
- Ensembles especializados

"""

from .base import BaseDLModel
from .transformer import BasketballTransformer
from .gnn import PlayerTeamGNN
from .lstm_attention import BiLSTMAttention
from .vae import BasketballVAE
from .ensemble import SpecializedEnsemble
from .hybrid import HybridPTSPredictor
from .trainer import DLTrainer
from .config import DLConfig

__version__ = "1.0.0"
__author__ = "AI Basketball Analytics Expert"

__all__ = [
    'BaseDLModel',
    'BasketballTransformer',
    'PlayerTeamGNN',
    'BiLSTMAttention',
    'BasketballVAE',
    'SpecializedEnsemble',
    'HybridPTSPredictor',
    'DLTrainer',
    'DLConfig'
] 