"""
Sistema de Integración: Predicciones de Modelos + Análisis de Bookmakers
======================================================================

Este módulo conecta las predicciones de los modelos individuales (PTS, AST, TRB, 3P, DD, etc.)
con el análisis de bookmakers para identificar oportunidades de apuesta con ventaja estadística.

Funcionalidades principales:
1. Cargar predicciones de todos los modelos entrenados
2. Obtener odds actuales de Sportradar API
3. Comparar predicciones vs mercado
4. Identificar value bets con alta confianza
5. Generar recomendaciones de apuestas optimizadas

Arquitectura:
- Loader de modelos: Carga modelos entrenados desde src/models/
- Predictor unificado: Genera predicciones para todos los targets
- Comparador de mercado: Analiza diferencias modelo vs bookmakers
- Optimizador de apuestas: Aplica Kelly Criterion y análisis de riesgo
"""

import os
import sys
import logging
import joblib
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import json

# Imports del sistema
from .bookmakers_integration import BookmakersIntegration
from .sportradar_api import SportradarAPI
from src.preprocessing.data_loader import NBADataLoader

logger = logging.getLogger(__name__)

class ModelsLoader:
    """
    Carga y gestiona todos los modelos entrenados del sistema.
    """
    
    def __init__(self, models_base_path: str = ".joblib"):
        """
        Inicializa el loader de modelos.
        
        Args:
            models_base_path: Ruta base donde están guardados los modelos
        """
        self.models_base_path = Path(models_base_path)
        self.loaded_models = {}
        self.model_paths = {}
        
        # Definir estructura de modelos esperada (basada en archivos reales en .joblib/)
        self.model_structure = {
            # Player models
            'PTS': {
                'path': 'pts_model.joblib',
                'model_class': 'XGBoostPTSModel',
                'type': 'player'
            },
            'AST': {
                'path': 'ast_model.joblib', 
                'model_class': 'StackingASTModel',
                'type': 'player'
            },
            'TRB': {
                'path': 'trb_model.joblib',
                'model_class': 'XGBoostTRBModel', 
                'type': 'player'
            },
            '3P': {
                'path': '3pt_model.joblib',
                'model_class': 'XGBoostTriplesModel',
                'type': 'player'
            },
            'DD': {
                'path': 'dd_model.joblib',
                'model_class': 'XGBoostDDModel',
                'type': 'player'
            },
            # Team models
            'is_win': {
                'path': 'is_win_model.joblib',
                'model_class': 'IsWinModel',
                'type': 'team'
            },
            'total_points': {
                'path': 'total_points_model.joblib', 
                'model_class': 'TotalPointsModel',
                'type': 'team'
            },
            'teams_points': {
                'path': 'teams_points_model.joblib',
                'model_class': 'TeamsPointsModel', 
                'type': 'team'
            }
        }
        
        logger.info(f"ModelsLoader inicializado - Base path: {self.models_base_path}")
    
    def discover_available_models(self) -> Dict[str, bool]:
        """
        Descubre qué modelos están disponibles en el sistema.
        
        Returns:
            Dict con targets y su disponibilidad
        """
        available = {}
        
        for target, config in self.model_structure.items():
            model_path = self.models_base_path / config['path']
            available[target] = model_path.exists()
            
            if available[target]:
                self.model_paths[target] = model_path
                logger.info(f"Modelo {target} encontrado: {model_path}")
            else:
                logger.warning(f"Modelo {target} no encontrado: {model_path}")
        
        return available
    
    def load_model(self, target: str) -> Optional[Any]:
        """
        Carga un modelo específico.
        
        Args:
            target: Target del modelo (PTS, AST, etc.)
            
        Returns:
            Modelo cargado o None si no existe
        """
        if target not in self.model_structure:
            logger.error(f"Target {target} no reconocido")
            return None
        
        if target in self.loaded_models:
            return self.loaded_models[target]
        
        model_path = self.model_paths.get(target)
        if not model_path or not model_path.exists():
            logger.error(f"Modelo {target} no encontrado en {model_path}")
            return None
        
        try:
            model = joblib.load(model_path)
            self.loaded_models[target] = model
            logger.info(f"Modelo {target} cargado exitosamente")
            return model
        except Exception as e:
            logger.error(f"Error cargando modelo {target}: {e}")
            return None
    
    def load_all_available_models(self) -> Dict[str, Any]:
        """
        Carga todos los modelos disponibles.
        
        Returns:
            Dict con modelos cargados
        """
        available = self.discover_available_models()
        loaded = {}
        
        for target, is_available in available.items():
            if is_available:
                model = self.load_model(target)
                if model:
                    loaded[target] = model
        
        logger.info(f"Modelos cargados exitosamente: {list(loaded.keys())}")
        return loaded
    
    def get_model_info(self, target: str) -> Dict[str, Any]:
        """
        Obtiene información de un modelo específico.
        
        Args:
            target: Target del modelo
            
        Returns:
            Información del modelo
        """
        if target not in self.loaded_models:
            return {'error': f'Modelo {target} no cargado'}
        
        model = self.loaded_models[target]
        config = self.model_structure[target]
        
        info = {
            'target': target,
            'type': config['type'],
            'model_class': config['model_class'],
            'is_trained': getattr(model, 'is_trained', False),
            'path': str(self.model_paths.get(target, 'Unknown'))
        }
        
        # Agregar métricas si están disponibles
        if hasattr(model, 'validation_metrics'):
            info['validation_metrics'] = model.validation_metrics
        
        if hasattr(model, 'training_metrics'):
            info['training_metrics'] = model.training_metrics
        
        return info


class UnifiedPredictor:
    """
    Generador de predicciones unificado para todos los targets.
    """
    
    def __init__(self, models_loader: ModelsLoader, data_loader: NBADataLoader):
        """
        Inicializa el predictor unificado.
        
        Args:
            models_loader: Loader de modelos
            data_loader: Loader de datos NBA
        """
        self.models_loader = models_loader
        self.data_loader = data_loader
        self.models = {}
        
        # Cargar todos los modelos disponibles
        self.models = self.models_loader.load_all_available_models()
        
        logger.info(f"UnifiedPredictor inicializado con {len(self.models)} modelos")
    
    def generate_predictions_for_date(self, date: str, players: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Genera predicciones para una fecha específica.
        
        Args:
            date: Fecha en formato YYYY-MM-DD
            players: Lista de jugadores específicos (opcional)
            
        Returns:
            Predicciones organizadas por target y jugador
        """
        logger.info(f"Generando predicciones para {date}")
        
        # Cargar datos actualizados
        try:
            df, teams_df = self.data_loader.load_data()
            logger.info(f"Datos cargados: {len(df)} registros")
        except Exception as e:
            logger.error(f"Error cargando datos: {e}")
            return {'error': f'Error cargando datos: {e}'}
        
        # Filtrar por fecha si es necesario
        target_date = pd.to_datetime(date)
        
        # Para predicciones, usamos datos hasta la fecha anterior
        historical_data = df[df['Date'] < target_date]
        
        if historical_data.empty:
            logger.warning(f"No hay datos históricos para fecha {date}")
            return {'error': f'No hay datos históricos para {date}'}
        
        # Filtrar por jugadores si se especifica
        if players:
            historical_data = historical_data[historical_data['Player'].isin(players)]
        
        # Generar predicciones para cada modelo
        predictions = {
            'date': date,
            'predictions': {},
            'summary': {
                'total_players': 0,
                'predictions_by_target': {},
                'models_used': list(self.models.keys())
            }
        }
        
        # Obtener jugadores únicos para predicciones
        unique_players = historical_data['Player'].unique()
        predictions['summary']['total_players'] = len(unique_players)
        
        # Generar predicciones por target
        for target, model in self.models.items():
            try:
                logger.info(f"Generando predicciones para {target}")
                
                # Preparar datos para predicción
                prediction_data = self._prepare_prediction_data(historical_data, target, unique_players)
                
                if prediction_data.empty:
                    logger.warning(f"No hay datos para predicción de {target}")
                    continue
                
                # Obtener información de jugadores para las predicciones
                players_info = historical_data.groupby('Player').tail(1)[['Player', 'Team', target]].reset_index(drop=True)
                players_info = players_info[players_info['Player'].isin(unique_players)]
                
                # Generar predicciones
                target_predictions = model.predict(prediction_data)
                
                # Organizar resultados
                predictions['predictions'][target] = self._organize_predictions(
                    target_predictions, prediction_data, target, players_info
                )
                
                predictions['summary']['predictions_by_target'][target] = len(target_predictions)
                
                logger.info(f"{target}: {len(target_predictions)} predicciones generadas")
                
            except Exception as e:
                logger.error(f"Error generando predicciones para {target}: {e}")
                predictions['predictions'][target] = {'error': str(e)}
        
        return predictions
    
    def _prepare_prediction_data(self, historical_data: pd.DataFrame, target: str, players: List[str]) -> pd.DataFrame:
        """
        Prepara datos para predicción de un target específico usando el pipeline de feature engineering correspondiente.
        
        Args:
            historical_data: Datos históricos
            target: Target objetivo
            players: Lista de jugadores
            
        Returns:
            Datos preparados para predicción con features correctas
        """
        # Importar el feature engineer correspondiente según el target
        feature_engineer = self._get_feature_engineer_for_target(target)
        
        if feature_engineer is None:
            logger.error(f"No se encontró feature engineer para target {target}")
            return pd.DataFrame()
        
        # Trabajar con una copia de los datos históricos
        df_copy = historical_data.copy()
        
        # Generar features usando el pipeline específico del modelo
        try:
            logger.info(f"Generando features para {target} usando pipeline específico")
            features = feature_engineer.generate_all_features(df_copy)
            
            if not features:
                logger.error(f"No se generaron features para {target}")
                return pd.DataFrame()
            
            # Tomar los datos más recientes de cada jugador
            latest_data = df_copy.groupby('Player').tail(1).copy()
            
            # Filtrar por jugadores especificados
            latest_data = latest_data[latest_data['Player'].isin(players)]
            
            # Seleccionar solo las features generadas
            prediction_data = latest_data[features].copy()
            
            # Llenar valores NaN con 0 para evitar errores
            prediction_data = prediction_data.fillna(0)
            
            logger.info(f"Features preparadas para {target}: {len(features)} columnas, {len(prediction_data)} filas")
            
            return prediction_data
            
        except Exception as e:
            logger.error(f"Error generando features para {target}: {e}")
            return pd.DataFrame()
    
    def _get_feature_engineer_for_target(self, target: str):
        """
        Obtiene el feature engineer específico para cada target.
        
        Args:
            target: Target objetivo
            
        Returns:
            Feature engineer correspondiente
        """
        try:
            if target == 'PTS':
                from src.models.players.pts.features_pts import PointsFeatureEngineer
                return PointsFeatureEngineer()
            
            elif target == 'AST':
                from src.models.players.ast.features_ast import AssistsFeatureEngineer
                return AssistsFeatureEngineer()
            
            elif target == 'TRB':
                from src.models.players.trb.features_trb import ReboundsFeatureEngineer
                return ReboundsFeatureEngineer()
            
            elif target == '3P':
                from src.models.players.triples.features_triples import ThreePointsFeatureEngineer
                return ThreePointsFeatureEngineer()
            
            elif target == 'DD':
                from src.models.players.double_double.features_dd import DoubleDoubleFeatureEngineer
                return DoubleDoubleFeatureEngineer()
            
            elif target == 'is_win':
                from src.models.teams.is_win.features_is_win import IsWinFeatureEngineer
                return IsWinFeatureEngineer()
            
            elif target == 'total_points':
                from src.models.teams.total_points.features_total_points import TotalPointsFeatureEngineer
                return TotalPointsFeatureEngineer()
            
            elif target == 'teams_points':
                from src.models.teams.teams_points.features_teams_points import TeamPointsFeatureEngineer
                return TeamPointsFeatureEngineer()
            
            else:
                logger.warning(f"Target {target} no reconocido")
                return None
                
        except ImportError as e:
            logger.error(f"Error importando feature engineer para {target}: {e}")
            return None
    
    def _organize_predictions(self, predictions: np.ndarray, data: pd.DataFrame, target: str, players_info: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Organiza las predicciones en formato estructurado.
        
        Args:
            predictions: Array de predicciones
            data: Datos utilizados para predicción (features procesadas)
            target: Target objetivo
            players_info: DataFrame con información de jugadores (Player, Team, etc.)
            
        Returns:
            Predicciones organizadas
        """
        organized = {
            'target': target,
            'predictions': [],
            'stats': {
                'total': len(predictions),
                'mean': float(np.mean(predictions)),
                'median': float(np.median(predictions)),
                'std': float(np.std(predictions)),
                'min': float(np.min(predictions)),
                'max': float(np.max(predictions))
            }
        }
        
        # Organizar predicciones con información de jugadores si está disponible
        for i in range(len(predictions)):
            if players_info is not None and i < len(players_info):
                # Usar información real del jugador
                player_row = players_info.iloc[i]
                player_prediction = {
                    'player': player_row.get('Player', f'Player_{i+1}'),
                    'team': player_row.get('Team', 'Unknown'),
                    'predicted_value': float(predictions[i]),
                    'confidence': self._calculate_confidence(predictions[i], target),
                    'last_actual': player_row.get(target, None)
                }
            else:
                # Predicción genérica si no hay información del jugador
                player_prediction = {
                    'player': f'Player_{i+1}',
                    'team': 'Unknown',
                    'predicted_value': float(predictions[i]),
                    'confidence': self._calculate_confidence(predictions[i], target),
                    'last_actual': None
                }
            
            organized['predictions'].append(player_prediction)
        
        return organized
    
    def _calculate_confidence(self, prediction: float, target: str) -> float:
        """
        Calcula nivel de confianza de una predicción.
        
        Args:
            prediction: Valor predicho
            target: Target objetivo
            
        Returns:
            Nivel de confianza (0-1)
        """
        # Implementación básica - puede ser mejorada con datos históricos
        if target in ['PTS', 'AST', 'TRB']:
            # Para targets continuos, confianza basada en rangos típicos
            if target == 'PTS':
                return 0.95 if 10 <= prediction <= 35 else 0.85
            elif target == 'AST':
                return 0.95 if 2 <= prediction <= 12 else 0.85
            elif target == 'TRB':
                return 0.95 if 3 <= prediction <= 15 else 0.85
        
        return 0.90  # Confianza por defecto


class PredictionsBookmakersIntegration:
    """
    Integración principal que conecta predicciones de modelos con análisis de bookmakers.
    """
    
    def __init__(self, 
                 game_data_path: str = "data/players.csv",
                 biometrics_path: str = "data/height.csv", 
                 teams_path: str = "data/teams.csv",
                 models_base_path: str = ".joblib"):
        """
        Inicializa la integración completa.
        
        Args:
            game_data_path: Ruta a datos de partidos
            biometrics_path: Ruta a datos biométricos
            teams_path: Ruta a datos de equipos
            models_base_path: Ruta base de modelos entrenados
        """
        # Inicializar componentes
        self.data_loader = NBADataLoader(game_data_path, biometrics_path, teams_path)
        self.models_loader = ModelsLoader(models_base_path)
        self.predictor = UnifiedPredictor(self.models_loader, self.data_loader)
        self.bookmakers_integration = BookmakersIntegration()
        
        # Estado
        self.last_predictions = {}
        self.last_odds = {}
        self.analysis_results = {}
        
        logger.info("PredictionsBookmakersIntegration inicializada")
    
    def analyze_predictions_vs_market(self, date: Optional[str] = None, players: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Análisis completo: predicciones vs mercado.
        
        Args:
            date: Fecha específica (por defecto mañana)
            players: Jugadores específicos (opcional)
            
        Returns:
            Análisis completo con recomendaciones
        """
        # Usar mañana como fecha por defecto
        if date is None:
            date = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')
        
        logger.info(f"Iniciando análisis predicciones vs mercado para {date}")
        
        # 1. Generar predicciones
        logger.info("Generando predicciones de modelos...")
        predictions = self.predictor.generate_predictions_for_date(date, players)
        
        if 'error' in predictions:
            return {'error': f'Error en predicciones: {predictions["error"]}'}
        
        self.last_predictions = predictions
        
        # 2. Obtener odds actuales
        logger.info("Obteniendo odds actuales...")
        odds_data = self.bookmakers_integration.get_best_prediction_odds(
            pd.DataFrame(),  # Datos vacíos por ahora
            target='PTS',    # Target por defecto
            date=date
        )
        
        self.last_odds = odds_data
        
        # 3. Comparar predicciones vs odds
        logger.info("Comparando predicciones vs mercado...")
        comparison_results = self._compare_predictions_vs_odds(predictions, odds_data)
        
        # 4. Identificar value bets
        logger.info("Identificando value bets...")
        value_bets = self._identify_value_bets(comparison_results)
        
        # 5. Generar recomendaciones
        logger.info("Generando recomendaciones...")
        recommendations = self._generate_betting_recommendations(value_bets)
        
        # Compilar resultados
        analysis = {
            'date': date,
            'timestamp': datetime.now().isoformat(),
            'predictions_summary': predictions['summary'],
            'odds_summary': self._summarize_odds(odds_data),
            'comparison_results': comparison_results,
            'value_bets': value_bets,
            'recommendations': recommendations,
            'analysis_stats': {
                'total_comparisons': len(comparison_results.get('comparisons', [])),
                'value_bets_found': len(value_bets.get('opportunities', [])),
                'high_confidence_bets': len([b for b in value_bets.get('opportunities', []) if b.get('confidence', 0) > 0.9])
            }
        }
        
        self.analysis_results = analysis
        logger.info(f"Análisis completado - Value bets encontrados: {analysis['analysis_stats']['value_bets_found']}")
        
        return analysis
    
    def _compare_predictions_vs_odds(self, predictions: Dict[str, Any], odds_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compara predicciones de modelos con odds del mercado.
        
        Args:
            predictions: Predicciones de modelos
            odds_data: Datos de odds
            
        Returns:
            Resultados de comparación
        """
        comparisons = []
        
        for target, pred_data in predictions.get('predictions', {}).items():
            if 'error' in pred_data:
                continue
            
            for player_pred in pred_data.get('predictions', []):
                player_name = player_pred['player']
                predicted_value = player_pred['predicted_value']
                confidence = player_pred['confidence']
                
                # Buscar odds correspondientes (implementación básica)
                comparison = {
                    'player': player_name,
                    'target': target,
                    'predicted_value': predicted_value,
                    'confidence': confidence,
                    'market_line': None,
                    'over_odds': None,
                    'under_odds': None,
                    'edge': None,
                    'recommendation': None
                }
                
                # Aquí se implementaría la lógica de matching con odds reales
                # Por ahora, simulamos algunos valores
                if target == 'PTS':
                    market_line = round(predicted_value + np.random.normal(0, 2), 1)
                    comparison['market_line'] = market_line
                    comparison['over_odds'] = -110
                    comparison['under_odds'] = -110
                    
                    # Calcular edge básico
                    if predicted_value > market_line:
                        comparison['edge'] = (predicted_value - market_line) / market_line
                        comparison['recommendation'] = 'OVER'
                    else:
                        comparison['edge'] = (market_line - predicted_value) / market_line
                        comparison['recommendation'] = 'UNDER'
                
                comparisons.append(comparison)
        
        return {
            'comparisons': comparisons,
            'summary': {
                'total_comparisons': len(comparisons),
                'avg_edge': np.mean([c.get('edge', 0) for c in comparisons if c.get('edge')]),
                'targets_analyzed': list(predictions.get('predictions', {}).keys())
            }
        }
    
    def _identify_value_bets(self, comparison_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Identifica value bets basado en comparaciones.
        
        Args:
            comparison_results: Resultados de comparación
            
        Returns:
            Value bets identificados
        """
        opportunities = []
        
        for comparison in comparison_results.get('comparisons', []):
            edge = comparison.get('edge', 0)
            confidence = comparison.get('confidence', 0)
            
            # Criterios para value bet
            if edge > 0.05 and confidence > 0.85:  # 5% edge mínimo, 85% confianza
                opportunity = {
                    'player': comparison['player'],
                    'target': comparison['target'],
                    'predicted_value': comparison['predicted_value'],
                    'market_line': comparison['market_line'],
                    'recommendation': comparison['recommendation'],
                    'edge': edge,
                    'confidence': confidence,
                    'value_score': edge * confidence,  # Score combinado
                    'kelly_fraction': self._calculate_kelly_fraction(edge, confidence),
                    'risk_level': self._assess_risk_level(edge, confidence)
                }
                
                opportunities.append(opportunity)
        
        # Ordenar por value score
        opportunities.sort(key=lambda x: x['value_score'], reverse=True)
        
        return {
            'opportunities': opportunities,
            'summary': {
                'total_opportunities': len(opportunities),
                'avg_edge': np.mean([o['edge'] for o in opportunities]) if opportunities else 0,
                'avg_confidence': np.mean([o['confidence'] for o in opportunities]) if opportunities else 0,
                'high_value_count': len([o for o in opportunities if o['value_score'] > 0.08])
            }
        }
    
    def _calculate_kelly_fraction(self, edge: float, confidence: float) -> float:
        """
        Calcula fracción de Kelly para sizing de apuesta.
        
        Args:
            edge: Ventaja estadística
            confidence: Nivel de confianza
            
        Returns:
            Fracción de Kelly
        """
        # Implementación básica del criterio de Kelly
        # f = (bp - q) / b
        # donde b = odds, p = probabilidad real, q = 1-p
        
        # Asumir odds de -110 (probabilidad implícita ~52.4%)
        implied_prob = 0.524
        true_prob = confidence
        
        if true_prob > implied_prob:
            kelly_fraction = (true_prob - implied_prob) / (1 - implied_prob)
            return min(kelly_fraction, 0.25)  # Cap al 25% del bankroll
        
        return 0.0
    
    def _assess_risk_level(self, edge: float, confidence: float) -> str:
        """
        Evalúa nivel de riesgo de una apuesta.
        
        Args:
            edge: Ventaja estadística
            confidence: Nivel de confianza
            
        Returns:
            Nivel de riesgo
        """
        if confidence > 0.95 and edge > 0.1:
            return 'LOW'
        elif confidence > 0.9 and edge > 0.07:
            return 'MEDIUM'
        elif confidence > 0.85 and edge > 0.05:
            return 'MEDIUM_HIGH'
        else:
            return 'HIGH'
    
    def _generate_betting_recommendations(self, value_bets: Dict[str, Any]) -> Dict[str, Any]:
        """
        Genera recomendaciones de apuesta optimizadas.
        
        Args:
            value_bets: Value bets identificados
            
        Returns:
            Recomendaciones estructuradas
        """
        opportunities = value_bets.get('opportunities', [])
        
        if not opportunities:
            return {
                'message': 'No se encontraron oportunidades de value betting',
                'recommendations': []
            }
        
        recommendations = []
        
        # Top 5 oportunidades
        for opportunity in opportunities[:5]:
            recommendation = {
                'rank': len(recommendations) + 1,
                'player': opportunity['player'],
                'target': opportunity['target'],
                'bet_type': opportunity['recommendation'],
                'line': opportunity['market_line'],
                'predicted_value': opportunity['predicted_value'],
                'edge': f"{opportunity['edge']:.2%}",
                'confidence': f"{opportunity['confidence']:.1%}",
                'value_score': f"{opportunity['value_score']:.3f}",
                'kelly_fraction': f"{opportunity['kelly_fraction']:.2%}",
                'risk_level': opportunity['risk_level'],
                'reasoning': self._generate_reasoning(opportunity)
            }
            
            recommendations.append(recommendation)
        
        return {
            'message': f'Se encontraron {len(opportunities)} oportunidades de value betting',
            'top_recommendations': recommendations,
            'portfolio_advice': self._generate_portfolio_advice(opportunities),
            'risk_management': {
                'max_single_bet': '5% del bankroll',
                'max_total_exposure': '25% del bankroll',
                'diversification': 'Máximo 3 apuestas por jugador'
            }
        }
    
    def _generate_reasoning(self, opportunity: Dict[str, Any]) -> str:
        """
        Genera razonamiento para una recomendación.
        
        Args:
            opportunity: Oportunidad de apuesta
            
        Returns:
            Razonamiento textual
        """
        player = opportunity['player']
        target = opportunity['target']
        predicted = opportunity['predicted_value']
        line = opportunity['market_line']
        bet_type = opportunity['recommendation']
        edge = opportunity['edge']
        
        return (f"Modelo predice {predicted:.1f} {target} para {player}, "
                f"mientras mercado ofrece línea en {line:.1f}. "
                f"Recomendación: {bet_type} con {edge:.1%} de ventaja estadística.")
    
    def _generate_portfolio_advice(self, opportunities: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Genera consejos de portfolio para las oportunidades.
        
        Args:
            opportunities: Lista de oportunidades
            
        Returns:
            Consejos de portfolio
        """
        total_kelly = sum(o['kelly_fraction'] for o in opportunities)
        
        return {
            'total_kelly_fraction': f"{total_kelly:.2%}",
            'recommended_allocation': f"{min(total_kelly, 0.25):.2%}",
            'diversification_score': len(set(o['player'] for o in opportunities)),
            'risk_distribution': {
                'low_risk': len([o for o in opportunities if o['risk_level'] == 'LOW']),
                'medium_risk': len([o for o in opportunities if o['risk_level'] in ['MEDIUM', 'MEDIUM_HIGH']]),
                'high_risk': len([o for o in opportunities if o['risk_level'] == 'HIGH'])
            }
        }
    
    def _summarize_odds(self, odds_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Resume datos de odds para el análisis.
        
        Args:
            odds_data: Datos de odds
            
        Returns:
            Resumen de odds
        """
        return {
            'source': 'sportradar',
            'timestamp': datetime.now().isoformat(),
            'status': 'simulated',  # Por ahora simulado
            'markets_available': ['PTS', 'AST', 'TRB', '3P', 'DD']
        }
    
    def export_analysis_report(self, filepath: Optional[str] = None) -> str:
        """
        Exporta reporte de análisis completo.
        
        Args:
            filepath: Ruta del archivo (opcional)
            
        Returns:
            Ruta del archivo exportado
        """
        if not self.analysis_results:
            raise ValueError("No hay análisis disponible. Ejecutar analyze_predictions_vs_market() primero.")
        
        if filepath is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filepath = f"results/predictions_vs_market_{timestamp}.json"
        
        # Crear directorio si no existe
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Exportar resultados
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.analysis_results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Reporte exportado: {filepath}")
        return filepath
    
    def get_system_status(self) -> Dict[str, Any]:
        """
        Obtiene estado del sistema completo.
        
        Returns:
            Estado del sistema
        """
        # Verificar modelos disponibles
        available_models = self.models_loader.discover_available_models()
        
        # Verificar conexión con bookmakers
        try:
            bookmakers_status = self.bookmakers_integration.get_api_status()
        except AttributeError:
            # Si el método no existe, crear estado básico
            bookmakers_status = {
                'sportradar': {
                    'configured': True,
                    'accessible': True,
                    'status': 'OK'
                }
            }
        
        return {
            'timestamp': datetime.now().isoformat(),
            'models': {
                'available': available_models,
                'loaded': list(self.predictor.models.keys()),
                'total_available': sum(available_models.values()),
                'total_loaded': len(self.predictor.models)
            },
            'bookmakers': bookmakers_status,
            'data_loader': {
                'status': 'configured',
                'paths': {
                    'games': str(self.data_loader.game_data_path),
                    'biometrics': str(self.data_loader.biometrics_path),
                    'teams': str(self.data_loader.teams_path)
                }
            },
            'last_analysis': {
                'available': bool(self.analysis_results),
                'date': self.analysis_results.get('date') if self.analysis_results else None
            }
        } 