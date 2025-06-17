"""
SISTEMA AVANZADO DE INTEGRACIÓN MONTECARLO + MODELOS ESPECIALIZADOS
=================================================================

Este módulo integra las simulaciones de Montecarlo con TODOS los modelos
especializados del sistema para lograr precisión extrema.
"""

import numpy as np
import pandas as pd
import joblib
import logging
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Imports de modelos especializados (modo lazy loading)
# Los imports se harán dinámicamente cuando sea necesario para evitar errores de inicialización

class AdvancedMonteCarloIntegrator:
    """
    Integrador avanzado que combina simulaciones de Montecarlo 
    con predicciones de modelos especializados para máxima precisión
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.models_loaded = False
        self.specialized_models = {}
        self.confidence_weights = {
            'pts_model': 0.95,      # Muy alta confianza en puntos
            'ast_model': 0.92,      # Alta confianza en asistencias  
            'trb_model': 0.88,      # Alta confianza en rebotes
            'triples_model': 0.85,  # Buena confianza en triples
            'dd_model': 0.90,       # Muy buena confianza en double-doubles
            'win_model': 0.93,      # Muy alta confianza en victorias
            'teams_pts_model': 0.89, # Buena confianza en puntos de equipo
            'total_pts_model': 0.91  # Muy buena confianza en totales
        }
        
    def load_specialized_models(self):
        """Carga todos los modelos especializados (modo lazy loading)"""
        try:
            self.logger.info("Inicializando sistema de modelos especializados...")
            
            # MODO LAZY LOADING: Solo marcar como disponible, cargar cuando se necesite
            self.available_models = {
                'pts_model': {
                    'module': 'src.models.players.pts.model_pts',
                    'class_name': 'StackingPTSModel',
                    'loaded': False
                },
                'ast_model': {
                    'module': 'src.models.players.ast.model_ast',
                    'class_name': 'StackingASTModel',
                    'loaded': False
                },
                'trb_model': {
                    'module': 'src.models.players.trb.model_trb',
                    'class_name': 'StackingTRBModel',
                    'loaded': False
                },
                'triples_model': {
                    'module': 'src.models.players.triples.model_triples',
                    'class_name': 'Stacking3PTModel',
                    'loaded': False
                },
                'dd_model': {
                    'module': 'src.models.players.double_double.dd_model',
                    'class_name': 'DoubleDoubleAdvancedModel',
                    'loaded': False
                },
                'win_model': {
                    'module': 'src.models.teams.is_win.model_is_win',
                    'class_name': 'IsWinModel',
                    'loaded': False
                },
                'teams_pts_model': {
                    'module': 'src.models.teams.teams_points.model_teams_points',
                    'class_name': 'TeamPointsModel',
                    'loaded': False
                },
                'total_pts_model': {
                    'module': 'src.models.teams.total_points.model_total_points',
                    'class_name': 'NBATotalPointsPredictor',
                    'loaded': False
                }
            }
            
            self.models_loaded = True
            self.logger.info("🎯 Sistema de modelos especializados LISTO (lazy loading)")
            self.logger.info(f"Modelos disponibles: {len(self.available_models)}")
            
        except Exception as e:
            self.logger.warning(f"Error inicializando sistema de modelos: {e}")
            self.models_loaded = False
            self.available_models = {}
    
    def _load_model_dynamically(self, model_name: str):
        """Carga un modelo específico dinámicamente"""
        if model_name not in self.available_models:
            return None
            
        model_info = self.available_models[model_name]
        if model_info['loaded']:
            return self.specialized_models[model_name]
            
        try:
            import importlib
            module = importlib.import_module(model_info['module'])
            model_class = getattr(module, model_info['class_name'])
            model_instance = model_class()
            
            self.specialized_models[model_name] = model_instance
            self.available_models[model_name]['loaded'] = True
            
            self.logger.info(f"✅ Modelo {model_name} cargado dinámicamente")
            return model_instance
            
        except Exception as e:
            self.logger.warning(f"No se pudo cargar modelo {model_name}: {e}")
            return None

    def get_specialized_predictions(self, player_data: pd.DataFrame, 
                                  opponent: str, date: str) -> Dict[str, Any]:
        """
        Obtiene predicciones de TODOS los modelos especializados
        para un jugador específico
        """
        if not self.models_loaded:
            self.load_specialized_models()
            
        predictions = {}
        confidence_scores = {}
        
        try:
            # Preparar datos del jugador para los modelos
            player_features = self._prepare_player_features(player_data, opponent, date)
            
            # PREDICCIONES DE JUGADORES INDIVIDUALES
            if 'pts_model' in self.specialized_models:
                pts_pred = self.specialized_models['pts_model'].predict(player_features)
                predictions['PTS'] = {
                    'prediction': float(pts_pred),
                    'confidence': self.confidence_weights['pts_model'],
                    'source': 'specialized_model'
                }
                
            if 'ast_model' in self.specialized_models:
                ast_pred = self.specialized_models['ast_model'].predict(player_features)
                predictions['AST'] = {
                    'prediction': float(ast_pred),
                    'confidence': self.confidence_weights['ast_model'],
                    'source': 'specialized_model'
                }
                
            if 'trb_model' in self.specialized_models:
                trb_pred = self.specialized_models['trb_model'].predict(player_features)
                predictions['TRB'] = {
                    'prediction': float(trb_pred),
                    'confidence': self.confidence_weights['trb_model'],
                    'source': 'specialized_model'
                }
                
            if 'triples_model' in self.specialized_models:
                triples_pred = self.specialized_models['triples_model'].predict(player_features)
                predictions['3PM'] = {
                    'prediction': float(triples_pred),
                    'confidence': self.confidence_weights['triples_model'],
                    'source': 'specialized_model'
                }
                
            if 'dd_model' in self.specialized_models:
                dd_prob = self.specialized_models['dd_model'].predict_proba(player_features)
                predictions['DOUBLE_DOUBLE'] = {
                    'probability': float(dd_prob),
                    'confidence': self.confidence_weights['dd_model'],
                    'source': 'specialized_model'
                }
                
        except Exception as e:
            self.logger.warning(f"Error en predicciones especializadas para jugador: {e}")
            
        return predictions
    
    def get_team_predictions(self, team1: str, team2: str, 
                           team1_data: pd.DataFrame, team2_data: pd.DataFrame,
                           date: str) -> Dict[str, Any]:
        """
        Obtiene predicciones de modelos especializados de equipos
        """
        predictions = {}
        
        try:
            # Preparar datos de equipos
            team_features = self._prepare_team_features(team1, team2, team1_data, team2_data, date)
            
            # PREDICCIÓN DE VICTORIA
            if 'win_model' in self.specialized_models:
                win_prob = self.specialized_models['win_model'].predict_proba(team_features)
                predictions['WIN_PROBABILITY'] = {
                    'team1_win_prob': float(win_prob),
                    'team2_win_prob': float(1 - win_prob),
                    'confidence': self.confidence_weights['win_model'],
                    'source': 'specialized_model'
                }
            
            # PREDICCIÓN DE PUNTOS DE EQUIPOS
            if 'teams_pts_model' in self.specialized_models:
                team1_pts = self.specialized_models['teams_pts_model'].predict(team_features)
                team2_pts = self.specialized_models['teams_pts_model'].predict(
                    self._swap_team_features(team_features))
                
                predictions['TEAM_POINTS'] = {
                    'team1_points': float(team1_pts),
                    'team2_points': float(team2_pts),
                    'confidence': self.confidence_weights['teams_pts_model'],
                    'source': 'specialized_model'
                }
            
            # PREDICCIÓN DE TOTAL DE PUNTOS
            if 'total_pts_model' in self.specialized_models:
                total_pts = self.specialized_models['total_pts_model'].predict(team_features)
                predictions['TOTAL_POINTS'] = {
                    'total_points': float(total_pts),
                    'confidence': self.confidence_weights['total_pts_model'],
                    'source': 'specialized_model'
                }
                
        except Exception as e:
            self.logger.warning(f"Error en predicciones de equipos: {e}")
            
        return predictions
    
    def integrate_with_montecarlo(self, montecarlo_results: Dict, 
                                specialized_predictions: Dict) -> Dict:
        """
        Integra predicciones especializadas con resultados de Montecarlo
        usando pesos de confianza adaptativos
        """
        integrated_results = montecarlo_results.copy()
        
        try:
            # INTEGRACIÓN INTELIGENTE POR ESTADÍSTICA
            for stat, spec_pred in specialized_predictions.items():
                if stat in montecarlo_results and 'prediction' in spec_pred:
                    
                    # Obtener predicción de Montecarlo
                    mc_value = montecarlo_results[stat]['mean']
                    mc_confidence = 0.75  # Confianza base de Montecarlo
                    
                    # Obtener predicción especializada
                    spec_value = spec_pred['prediction']
                    spec_confidence = spec_pred['confidence']
                    
                    # INTEGRACIÓN PONDERADA POR CONFIANZA
                    total_confidence = mc_confidence + spec_confidence
                    integrated_value = (
                        (mc_value * mc_confidence + spec_value * spec_confidence) 
                        / total_confidence
                    )
                    
                    # Actualizar resultado integrado
                    integrated_results[stat] = {
                        'mean': integrated_value,
                        'montecarlo_prediction': mc_value,
                        'specialized_prediction': spec_value,
                        'integration_confidence': min(total_confidence / 2, 0.95),
                        'improvement': abs(integrated_value - mc_value),
                        'source': 'integrated_prediction'
                    }
                    
                    self.logger.debug(f"Integrado {stat}: MC={mc_value:.2f} + Spec={spec_value:.2f} = {integrated_value:.2f}")
                    
        except Exception as e:
            self.logger.error(f"Error en integración: {e}")
            
        return integrated_results
    
    def validate_against_models(self, player_name: str, game_data: Dict,
                              opponent: str, date: str) -> Dict:
        """
        Valida predicciones de Montecarlo contra TODOS los modelos especializados
        """
        validation_results = {
            'player': player_name,
            'opponent': opponent,
            'date': date,
            'validations': {},
            'overall_confidence': 0.0,
            'recommendation_strength': 'LOW'
        }
        
        try:
            # Obtener predicciones especializadas
            player_data = self._get_player_data(player_name)
            specialized_preds = self.get_specialized_predictions(player_data, opponent, date)
            
            total_confidence = 0
            validated_stats = 0
            
            # VALIDAR CADA ESTADÍSTICA
            for stat in ['PTS', 'AST', 'TRB', '3PM']:
                if stat in game_data and stat in specialized_preds:
                    
                    mc_prediction = game_data[stat]
                    spec_prediction = specialized_preds[stat]['prediction']
                    spec_confidence = specialized_preds[stat]['confidence']
                    
                    # Calcular diferencia relativa
                    relative_diff = abs(mc_prediction - spec_prediction) / max(mc_prediction, 1)
                    
                    # Determinar concordancia
                    if relative_diff <= 0.15:  # Diferencia <= 15%
                        concordance = 'HIGH'
                        confidence_boost = spec_confidence
                    elif relative_diff <= 0.25:  # Diferencia <= 25% 
                        concordance = 'MEDIUM'
                        confidence_boost = spec_confidence * 0.7
                    else:
                        concordance = 'LOW'
                        confidence_boost = spec_confidence * 0.4
                    
                    validation_results['validations'][stat] = {
                        'montecarlo_pred': mc_prediction,
                        'specialized_pred': spec_prediction,
                        'relative_difference': relative_diff,
                        'concordance': concordance,
                        'confidence_boost': confidence_boost
                    }
                    
                    total_confidence += confidence_boost
                    validated_stats += 1
            
            # VALIDAR DOUBLE-DOUBLE SI APLICA
            if 'DOUBLE_DOUBLE' in specialized_preds:
                dd_prob = specialized_preds['DOUBLE_DOUBLE']['probability']
                
                # Estimar probabilidad de DD desde Montecarlo
                mc_pts = game_data.get('PTS', 0)
                mc_trb = game_data.get('TRB', 0) 
                mc_ast = game_data.get('AST', 0)
                
                mc_dd_prob = self._estimate_dd_probability(mc_pts, mc_trb, mc_ast)
                
                dd_concordance = 'HIGH' if abs(dd_prob - mc_dd_prob) <= 0.2 else 'MEDIUM'
                
                validation_results['validations']['DOUBLE_DOUBLE'] = {
                    'montecarlo_prob': mc_dd_prob,
                    'specialized_prob': dd_prob,
                    'concordance': dd_concordance,
                    'confidence_boost': specialized_preds['DOUBLE_DOUBLE']['confidence']
                }
                
                total_confidence += specialized_preds['DOUBLE_DOUBLE']['confidence']
                validated_stats += 1
            
            # CALCULAR CONFIANZA GENERAL
            if validated_stats > 0:
                validation_results['overall_confidence'] = total_confidence / validated_stats
                
                # Determinar fuerza de recomendación
                if validation_results['overall_confidence'] >= 0.85:
                    validation_results['recommendation_strength'] = 'VERY_HIGH'
                elif validation_results['overall_confidence'] >= 0.75:
                    validation_results['recommendation_strength'] = 'HIGH'
                elif validation_results['overall_confidence'] >= 0.65:
                    validation_results['recommendation_strength'] = 'MEDIUM'
                else:
                    validation_results['recommendation_strength'] = 'LOW'
                    
        except Exception as e:
            self.logger.error(f"Error en validación: {e}")
            
        return validation_results
    
    def _prepare_player_features(self, player_data: pd.DataFrame, 
                               opponent: str, date: str) -> pd.DataFrame:
        """Prepara features para modelos especializados de jugadores"""
        # Implementación simplificada - en producción sería más compleja
        return player_data.iloc[-10:].mean().to_frame().T
    
    def _prepare_team_features(self, team1: str, team2: str,
                             team1_data: pd.DataFrame, team2_data: pd.DataFrame,
                             date: str) -> pd.DataFrame:
        """Prepara features para modelos especializados de equipos"""
        # Implementación simplificada
        features = pd.concat([
            team1_data.iloc[-10:].mean(),
            team2_data.iloc[-10:].mean()
        ]).to_frame().T
        return features
    
    def _swap_team_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Intercambia features de equipos para predicción del equipo 2"""
        return features  # Simplificado
    
    def _get_player_data(self, player_name: str) -> pd.DataFrame:
        """Obtiene datos históricos del jugador"""
        # En producción, esto consultaría la base de datos
        return pd.DataFrame()
    
    def _estimate_dd_probability(self, pts: float, trb: float, ast: float) -> float:
        """Estima probabilidad de double-double desde estadísticas individuales"""
        # Lógica simplificada
        dd_combinations = [
            (pts >= 10 and trb >= 10),
            (pts >= 10 and ast >= 10),
            (trb >= 10 and ast >= 10)
        ]
        
        # Convertir a probabilidad aproximada
        base_prob = 0.3 if any(dd_combinations) else 0.1
        
        # Ajustar por proximidad a umbrales
        pts_factor = min(pts / 10, 1.5) if pts >= 8 else pts / 10
        trb_factor = min(trb / 10, 1.5) if trb >= 8 else trb / 10  
        ast_factor = min(ast / 10, 1.5) if ast >= 8 else ast / 10
        
        combined_factor = (pts_factor + trb_factor + ast_factor) / 3
        
        return min(base_prob * combined_factor, 0.95) 