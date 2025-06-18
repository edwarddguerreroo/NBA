"""
Módulo de Deep Learning para Predicción de Rebotes (TRB)
========================================================

Este módulo contiene arquitecturas avanzadas de Deep Learning especializadas
en la predicción de rebotes en la NBA, incluyendo:

- Transformers para secuencias temporales
- Graph Neural Networks para relaciones jugador-equipo
- LSTM bidireccionales con attention
- Variational Autoencoders para feature learning
- Ensembles especializados

Autor: AI Basketball Analytics Expert
Fecha: 2025-06-10
"""

from .base import BaseDLModel
from .transformer import BasketballTransformer
from .gnn import PlayerTeamGNN
from .lstm_attention import BiLSTMAttention
from .vae import BasketballVAE
from .ensemble import SpecializedEnsemble
from .hybrid import HybridTRBPredictor
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
    'HybridTRBPredictor',
    'DLTrainer',
    'DLConfig'
] 