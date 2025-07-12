"""
Módulo BookmakersIntegration
---------------------------
Este módulo proporciona la integración avanzada entre nuestros modelos predictivos
y los datos de las casas de apuestas. Su propósito principal es:

1. Conectar nuestras predicciones con las probabilidades implícitas del mercado
   para identificar "value bets" (apuestas con ventaja estadística)

2. Analizar líneas específicas donde nuestro modelo tiene alta precisión (≥96%)
   y comparar con las odds ofrecidas por las casas de apuestas

3. Identificar ineficiencias del mercado y oportunidades de arbitraje entre
   diferentes casas de apuestas

4. Analizar movimientos de líneas para detectar tendencias y patrones

5. Generar estrategias óptimas de apuestas aplicando variantes del criterio
   de Kelly para la asignación de capital

Este módulo representa la capa de análisis avanzado que convierte las predicciones
en estrategias concretas de apuestas, determinando dónde existe ventaja estadística
sobre el mercado.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
from utils.bookmakers.bookmakers_data_fetcher import BookmakersDataFetcher
from utils.bookmakers.exceptions import DataValidationError
from datetime import datetime

logger = logging.getLogger(__name__)

class BookmakersIntegration:
    """
    Integra los datos de casas de apuestas con nuestro sistema de predicciones
    para encontrar líneas con alta precisión y buenas odds.
    """
    
    def __init__(
        self,
        api_keys: Dict[str, str] = None,
        odds_data_dir: str = "data/bookmakers",
        minimum_edge: float = 0.04,  # 4% de ventaja mínima 
        confidence_threshold: float = 0.95  # 95% de precisión
    ):
        self.bookmakers_fetcher = BookmakersDataFetcher(api_keys, odds_data_dir)
        self.minimum_edge = minimum_edge
        self.confidence_threshold = confidence_threshold
        
        # Para guardar análisis histórico
        self.historical_analysis = {}
    
    def process_player_data_with_bookmakers(
        self,
        player_data: pd.DataFrame,
        target: str,
        use_api: bool = False,
        api_provider: str = 'sportradar',  
        odds_file: str = None
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Procesa datos de jugadores incorporando odds de casas de apuestas
        
        Este método principal:
        1. Orquesta el proceso completo de análisis de casas de apuestas
        2. Obtiene datos de odds (API, archivo o simulados)
        3. Integra estos datos con las predicciones del modelo
        4. Ejecuta múltiples análisis avanzados:
           - Líneas de alta confianza
           - Ineficiencias de mercado
           - Oportunidades de arbitraje
           - Movimientos de líneas
        5. Genera una estrategia óptima de apuestas
        
        Args:
            player_data: DataFrame con datos de jugadores
            target: Estadística a predecir (PTS, TRB, AST, 3P)
            use_api: Si usar API externa para obtener odds
            api_provider: Proveedor de API para odds
            odds_file: Archivo con datos de odds (si no se usa API)
            
        Returns:
            Tuple con (DataFrame con odds añadidas, Análisis de mejores apuestas)
        """
        if target not in ['PTS', 'TRB', 'AST', '3P']:
            logger.error(f"Target no soportado: {target}")
            return player_data, {}
        
        # Obtener líneas para el target
        line_values = self._get_line_values_for_target(target)
        
        # Paso 1: Obtener datos de casas de apuestas
        odds_data = self._fetch_bookmaker_data(target, use_api, api_provider, odds_file)
        
        # Paso 2: Combinar datos reales -
        if odds_data.get('success', False):
            # Tenemos datos reales, combinarlos
            player_data_with_odds = self.bookmakers_fetcher.merge_odds_with_player_data(
                player_data, odds_data, target, line_values
            )
        else:
            # ERROR: No hay datos reales disponibles
            error_msg = f"No se pudieron obtener datos reales de Sportradar para {target}"
            logger.error(error_msg)
            logger.error("El sistema requiere datos reales de la API - no se permiten simulaciones")
            return player_data, {
                'success': False,
                'error': error_msg,
                'requires_real_data': True
            }
        
        # Paso 3: Identificar líneas de alta confianza (95%+)
        high_confidence_lines = self.identify_high_confidence_betting_lines(
            player_data_with_odds, target, min_confidence=self.confidence_threshold
        )
        
        if not high_confidence_lines:
            logger.warning(f"No se encontraron líneas de alta confianza para {target}")
            return player_data_with_odds, {}
        
        # Paso 4: Analizar mercado para estas líneas
        market_analysis = self.analyze_market_inefficiencies(
            player_data_with_odds, target, min_edge=self.minimum_edge
        )
        
        # Paso 5: Buscar arbitraje si es posible
        arbitrage_opportunities = self.find_best_odds_arbitrage(
            player_data_with_odds, target
        )
        
        # Paso 6: Analizar movimientos de líneas
        line_movements = self.compare_line_movements(
            player_data_with_odds, target
        )
        
        # Paso 7: Generar estrategia óptima de apuestas
        betting_strategy = self.get_optimal_betting_strategy(
            player_data_with_odds, target, 
            confidence_threshold=self.confidence_threshold,
            min_edge=self.minimum_edge
        )
        
        # Paso 8: Guardar análisis completo
        analysis_results = {
            'target': target,
            'high_confidence_lines': high_confidence_lines,
            'market_inefficiencies': market_analysis,
            'arbitrage_opportunities': arbitrage_opportunities,
            'line_movements': line_movements,
            'betting_strategy': betting_strategy,
            'odds_comparison': self.bookmakers_fetcher.compare_bookmaker_odds(
                player_data_with_odds, target, line_values
            ).to_dict(orient='records')
        }
        
        # Guardar histórico
        self.historical_analysis[target] = analysis_results
        
        return player_data_with_odds, analysis_results
    
    def identify_high_confidence_betting_lines(
        self,
        df: pd.DataFrame,
        target: str,
        min_confidence: float = 0.95
    ) -> List[Dict[str, Any]]:
        """
        Identifica líneas de apuestas con alta confianza basadas en datos históricos.
        
        Args:
            df: DataFrame con datos de jugadores y odds
            target: Estadística objetivo
            min_confidence: Confianza mínima requerida
            
        Returns:
            Lista de líneas con alta confianza
        """
        high_confidence_lines = []
        
        # Buscar columnas de probabilidad histórica
        prob_columns = [col for col in df.columns if f"{target}_over_" in col and "_prob_" in col]
        
        for col in prob_columns:
            # Extraer información de la línea
            parts = col.split('_')
            if len(parts) >= 3:
                try:
                    line_value = float(parts[2])
                    
                    # Filtrar por confianza
                    high_conf_rows = df[df[col] >= min_confidence]
                    
                    if not high_conf_rows.empty:
                        high_confidence_lines.append({
                            'target': target,
                            'line': line_value,
                            'confidence_column': col,
                            'players_count': len(high_conf_rows),
                            'avg_confidence': high_conf_rows[col].mean(),
                            'min_confidence': high_conf_rows[col].min(),
                            'max_confidence': high_conf_rows[col].max()
                        })
                except ValueError:
                    continue
        
        # Ordenar por confianza promedio
        high_confidence_lines.sort(key=lambda x: x['avg_confidence'], reverse=True)
        
        return high_confidence_lines
    
    def analyze_market_inefficiencies(
        self,
        df: pd.DataFrame,
        target: str,
        min_edge: float = 0.04
    ) -> Dict[str, Any]:
        """
        Analiza ineficiencias del mercado comparando nuestras predicciones con odds.
        
        Args:
            df: DataFrame con datos y odds
            target: Estadística objetivo
            min_edge: Edge mínimo requerido
            
        Returns:
            Análisis de ineficiencias del mercado
        """
        inefficiencies = {
            'target': target,
            'opportunities': [],
            'summary': {
                'total_opportunities': 0,
                'avg_edge': 0,
                'max_edge': 0,
                'bookmakers_analyzed': set()
            }
        }
        
        # Buscar columnas de odds
        odds_columns = [col for col in df.columns if f"{target}_over_" in col and "_odds_" in col]
        
        for odds_col in odds_columns:
            # Extraer información
            parts = odds_col.split('_')
            if len(parts) >= 5:
                try:
                    line_value = float(parts[2])
                    bookmaker = parts[4]
                    
                    # Buscar columna de probabilidad correspondiente
                    prob_col = f"{target}_over_{line_value}_prob_10"
                    
                    if prob_col in df.columns:
                        # Calcular edge para cada jugador
                        for idx, row in df.iterrows():
                            if pd.notna(row[odds_col]) and pd.notna(row[prob_col]):
                                # Probabilidad implícita de las odds
                                implied_prob = 1 / row[odds_col]
                                
                                # Nuestra probabilidad
                                our_prob = row[prob_col]
                                
                                # Calcular edge
                                edge = our_prob - implied_prob
                                
                                if edge >= min_edge:
                                    inefficiencies['opportunities'].append({
                                        'player': row.get('Player', 'Unknown'),
                                        'line': line_value,
                                        'bookmaker': bookmaker,
                                        'our_probability': our_prob,
                                        'implied_probability': implied_prob,
                                        'edge': edge,
                                        'odds': row[odds_col],
                                        'confidence': row[prob_col]
                                    })
                                    
                                    inefficiencies['summary']['bookmakers_analyzed'].add(bookmaker)
                
                except (ValueError, KeyError):
                    continue
        
        # Calcular estadísticas del resumen
        if inefficiencies['opportunities']:
            edges = [opp['edge'] for opp in inefficiencies['opportunities']]
            inefficiencies['summary']['total_opportunities'] = len(edges)
            inefficiencies['summary']['avg_edge'] = np.mean(edges)
            inefficiencies['summary']['max_edge'] = np.max(edges)
        
        inefficiencies['summary']['bookmakers_analyzed'] = list(inefficiencies['summary']['bookmakers_analyzed'])
        
        return inefficiencies
    
    def find_best_odds_arbitrage(
        self,
        df: pd.DataFrame,
        target: str
    ) -> List[Dict[str, Any]]:
        """
        Busca oportunidades de arbitraje entre diferentes casas de apuestas.
        
        Args:
            df: DataFrame con odds de múltiples casas
            target: Estadística objetivo
            
        Returns:
            Lista de oportunidades de arbitraje
        """
        arbitrage_opportunities = []
        
        # Agrupar odds por línea
        line_values = self._get_line_values_for_target(target)
        
        for line in line_values:
            # Buscar todas las odds para esta línea
            over_cols = [col for col in df.columns if f"{target}_over_{line}_odds_" in col]
            under_cols = [col for col in df.columns if f"{target}_under_{line}_odds_" in col]
            
            if len(over_cols) > 1 and len(under_cols) > 1:
                # Para cada jugador, buscar la mejor combinación
                for idx, row in df.iterrows():
                    # Mejores odds para over y under
                    best_over_odds = 0
                    best_over_bm = None
                    best_under_odds = 0
                    best_under_bm = None
                    
                    for col in over_cols:
                        if pd.notna(row[col]) and row[col] > best_over_odds:
                            best_over_odds = row[col]
                            best_over_bm = col.split('_')[-1]
                    
                    for col in under_cols:
                        if pd.notna(row[col]) and row[col] > best_under_odds:
                            best_under_odds = row[col]
                            best_under_bm = col.split('_')[-1]
                    
                    # Verificar si hay arbitraje
                    if best_over_odds > 0 and best_under_odds > 0:
                        total_implied_prob = (1/best_over_odds) + (1/best_under_odds)
                        
                        if total_implied_prob < 1.0:  # Oportunidad de arbitraje
                            profit_margin = 1 - total_implied_prob
                            
                            arbitrage_opportunities.append({
                                'player': row.get('Player', 'Unknown'),
                                'target': target,
                                'line': line,
                                'over_odds': best_over_odds,
                                'over_bookmaker': best_over_bm,
                                'under_odds': best_under_odds,
                                'under_bookmaker': best_under_bm,
                                'profit_margin': profit_margin,
                                'total_implied_prob': total_implied_prob
                            })
        
        # Ordenar por margen de ganancia
        arbitrage_opportunities.sort(key=lambda x: x['profit_margin'], reverse=True)
        
        return arbitrage_opportunities
    
    def compare_line_movements(
        self,
        df: pd.DataFrame,
        target: str
    ) -> Dict[str, Any]:
        """
        Analiza movimientos de líneas (placeholder - requiere datos históricos).
        
        Args:
            df: DataFrame con datos actuales
            target: Estadística objetivo
            
        Returns:
            Análisis de movimientos de líneas
        """
        return {
            'target': target,
            'message': 'Análisis de movimientos de líneas requiere datos históricos',
            'current_lines': self._get_current_lines_summary(df, target)
        }
    
    def get_optimal_betting_strategy(
        self,
        df: pd.DataFrame,
        target: str,
        confidence_threshold: float = 0.95,
        min_edge: float = 0.04
    ) -> Dict[str, Any]:
        """
        Genera estrategia óptima de apuestas usando criterio de Kelly.
        
        Args:
            df: DataFrame con datos y odds
            target: Estadística objetivo
            confidence_threshold: Umbral de confianza
            min_edge: Edge mínimo
            
        Returns:
            Estrategia óptima de apuestas
        """
        strategy = {
            'target': target,
            'recommendations': [],
            'summary': {
                'total_bets': 0,
                'total_edge': 0,
                'avg_kelly_fraction': 0,
                'expected_roi': 0
            }
        }
        
        # Analizar ineficiencias del mercado
        inefficiencies = self.analyze_market_inefficiencies(df, target, min_edge)
        
        kelly_fractions = []
        
        for opportunity in inefficiencies['opportunities']:
            if opportunity['confidence'] >= confidence_threshold:
                # Calcular fracción de Kelly
                p = opportunity['our_probability']  # Probabilidad de ganar
                b = opportunity['odds'] - 1  # Ganancia neta por unidad apostada
                q = 1 - p  # Probabilidad de perder
                
                # Fórmula de Kelly: f = (bp - q) / b
                kelly_fraction = (b * p - q) / b
                
                # Limitar fracción de Kelly (máximo 25% del bankroll)
                kelly_fraction = min(kelly_fraction, 0.25)
                
                if kelly_fraction > 0:
                    strategy['recommendations'].append({
                        'player': opportunity['player'],
                        'line': opportunity['line'],
                        'bookmaker': opportunity['bookmaker'],
                        'bet_type': 'over',
                        'odds': opportunity['odds'],
                        'confidence': opportunity['confidence'],
                        'edge': opportunity['edge'],
                        'kelly_fraction': kelly_fraction,
                        'expected_value': kelly_fraction * opportunity['edge']
                    })
                    
                    kelly_fractions.append(kelly_fraction)
        
        # Calcular estadísticas del resumen
        if strategy['recommendations']:
            strategy['summary']['total_bets'] = len(strategy['recommendations'])
            strategy['summary']['total_edge'] = sum(r['edge'] for r in strategy['recommendations'])
            strategy['summary']['avg_kelly_fraction'] = np.mean(kelly_fractions)
            strategy['summary']['expected_roi'] = sum(r['expected_value'] for r in strategy['recommendations'])
        
        # Ordenar por valor esperado
        strategy['recommendations'].sort(key=lambda x: x['expected_value'], reverse=True)
        
        return strategy
    
    def _fetch_bookmaker_data(
        self,
        target: str,
        use_api: bool = False,
        api_provider: str = 'sportradar',
        odds_file: str = None
    ) -> Dict[str, Any]:
        """
        Obtiene datos de odds para el target indicado.
        SOLO DATOS REALES - NO SIMULACIÓN.
        """
        if use_api:
            logger.info(f"Obteniendo datos reales de {api_provider} para {target}")
            # Configurar mercados según el target
            markets = ['player_props']
            
            # Cargar desde API
            return self.bookmakers_fetcher.load_odds_from_api(
                sport='basketball_nba',
                markets=markets,
                api_provider=api_provider
            )
        elif odds_file:
            logger.info(f"Cargando datos reales desde archivo: {odds_file}")
            # Determinar formato
            file_format = odds_file.split('.')[-1].lower()
            if file_format in ['xls', 'xlsx']:
                format = 'excel'
            else:
                format = file_format
                
            # Cargar desde archivo
            return self.bookmakers_fetcher.load_odds_from_file(
                odds_file, format=format
            )
        else:
            # Sin fuente de datos
            error_msg = 'No se especificó fuente de datos de odds - se requieren datos reales'
            logger.error(error_msg)
            return {'success': False, 'error': error_msg}
    
    def _get_current_lines_summary(self, df: pd.DataFrame, target: str) -> Dict[str, Any]:
        """Obtiene resumen de líneas actuales."""
        summary = {
            'target': target,
            'lines_available': [],
            'bookmakers': set()
        }
        
        # Buscar todas las líneas disponibles
        odds_columns = [col for col in df.columns if f"{target}_over_" in col and "_odds_" in col]
        
        for col in odds_columns:
            parts = col.split('_')
            if len(parts) >= 5:
                try:
                    line_value = float(parts[2])
                    bookmaker = parts[4]
                    
                    if line_value not in summary['lines_available']:
                        summary['lines_available'].append(line_value)
                    
                    summary['bookmakers'].add(bookmaker)
                except ValueError:
                    continue
        
        summary['lines_available'].sort()
        summary['bookmakers'] = list(summary['bookmakers'])
        
        return summary
    
    def _get_line_values_for_target(self, target: str) -> List[float]:
        """
        Devuelve valores de línea apropiados para cada target
        """
        if target == 'PTS':
            return [10, 15, 20, 25, 30, 35]
        elif target == 'TRB':
            return [4, 6, 8, 10, 12]
        elif target == 'AST':
            return [4, 6, 8, 10, 12]
        elif target == '3P':
            return [1, 2, 3, 4, 5]
        else:
            return []
    
    def get_best_value_bets(
        self,
        target: str = None,
        min_confidence: float = 0.96,
        min_edge: float = 0.04,
        max_results: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Devuelve las mejores apuestas de valor basadas en el análisis previo
        
        Este método:
        1. Filtra las apuestas con mayor ventaja estadística
        2. Prioriza líneas donde el modelo tiene mayor confianza
        3. Ordena resultados por valor esperado (EV) descendente
        4. Aplica criterios mínimos de confianza y ventaja (edge)
        5. Proporciona un listado ejecutable de las mejores oportunidades
        
        Args:
            target: Estadística específica (si None, busca en todas)
            min_confidence: Confianza mínima requerida
            min_edge: Ventaja mínima sobre el mercado
            max_results: Máximo de resultados a devolver
            
        Returns:
            Lista de mejores apuestas ordenadas por EV
        """
        all_value_bets = []
        
        # Determinar qué targets analizar
        targets_to_check = [target] if target else list(self.historical_analysis.keys())
        
        # Para cada target con análisis
        for t in targets_to_check:
            if t not in self.historical_analysis:
                continue
                
            analysis = self.historical_analysis[t]
            
            # Extraer apuestas de valor
            market_inefficiencies = analysis.get('market_inefficiencies', {})
            
            for line, line_info in market_inefficiencies.items():
                if line_info['confidence'] >= min_confidence:
                    # Buscar mejor apuesta para esta línea
                    for value_bet in line_info.get('value_bets', []):
                        if value_bet['edge'] >= min_edge:
                            all_value_bets.append({
                                'target': t,
                                'line': line,
                                'prediction': line_info['prediction'],
                                'confidence': line_info['confidence'],
                                'bookmaker': value_bet['bookmaker'],
                                'odds': value_bet['market_odds'],
                                'edge': value_bet['edge'],
                                'expected_roi': (value_bet['expected_value'] - 1) * 100,
                                'kelly_fraction': value_bet.get('kelly_fraction', 0)
                            })
        
        # Ordenar por ROI esperado
        all_value_bets.sort(key=lambda x: x['expected_roi'], reverse=True)
        
        # Limitar resultados
        return all_value_bets[:max_results]
    
    def export_value_bets_report(
        self,
        output_file: str = "value_bets_report.csv",
        include_low_confidence: bool = False
    ) -> str:
        """
        Exporta un informe con las mejores apuestas de valor
        
        Args:
            output_file: Archivo para guardar el reporte
            include_low_confidence: Si incluir apuestas con confianza ligeramente menor (94%+)
            
        Returns:
            Ruta al archivo guardado
        """
        # Obtener todas las apuestas de valor
        min_confidence = 0.94 if include_low_confidence else 0.96
        value_bets = self.get_best_value_bets(min_confidence=min_confidence, min_edge=0.03, max_results=50)
        
        if not value_bets:
            logger.warning("No hay apuestas de valor para exportar")
            return None
            
        # Convertir a DataFrame
        df = pd.DataFrame(value_bets)
        
        # Guardar a archivo
        try:
            df.to_csv(output_file, index=False)
            logger.info(f"Reporte de apuestas de valor guardado en {output_file}")
            return output_file
        except Exception as e:
            logger.error(f"Error al guardar reporte: {str(e)}")
            return None
    
    def analyze_player_performance_vs_lines(
        self,
        player_data: pd.DataFrame,
        target: str,
        lookback_days: int = 60,
        confidence_threshold: float = 0.96
    ) -> Dict[str, Any]:
        """
        Analiza el rendimiento de los jugadores frente a diferentes líneas de apuestas
        para identificar patrones de alta confianza (96%+)
        
        Args:
            player_data: DataFrame con datos de jugadores
            target: Estadística a predecir
            lookback_days: Días hacia atrás para analizar historial
            confidence_threshold: Umbral mínimo de confianza (default: 96%)
            
        Returns:
            Diccionario con análisis de jugadores y líneas
        """
        if target not in ['PTS', 'TRB', 'AST', '3P']:
            logger.error(f"Target no soportado: {target}")
            return {}
            
        # Asegurar que tenemos fecha en formato correcto
        df = player_data.copy()
        
        if 'Date' in df.columns:
            try:
                df['Date'] = pd.to_datetime(df['Date'])
            except:
                logger.warning("No se pudo convertir la columna Date a datetime")
            
            # Filtrar por fecha reciente
            try:
                cutoff_date = df['Date'].max() - pd.Timedelta(days=lookback_days)
                df = df[df['Date'] >= cutoff_date].copy()
                logger.info(f"Análisis restringido a los últimos {lookback_days} días ({len(df)} registros)")
            except Exception as e:
                logger.warning(f"Error al filtrar por fecha: {e}")
        
        # Si no hay datos suficientes, salir
        if len(df) < 30:
            logger.warning(f"Insuficientes datos recientes para análisis ({len(df)} < 30)")
            return {}
        
        # Obtener líneas para el target
        line_values = self._get_line_values_for_target(target)
        
        # Crear columnas de over/under para cada línea
        for line in line_values:
            line_col = f"{target}_over_{line}"
            if line_col not in df.columns:
                try:
                    df[line_col] = (df[target] > line).astype(int)
                except Exception as e:
                    logger.warning(f"No se pudo crear columna {line_col}: {e}")
                    continue
        
        # Resultado: jugadores por línea con alta confianza
        player_analysis = {}
        
        # Agrupar por jugador
        if 'Player' in df.columns:
            for player, player_data in df.groupby('Player'):
                # Solo analizar jugadores con suficientes registros
                if len(player_data) < 10:
                    continue
                    
                # Analizar cada línea para este jugador
                player_lines = {}
                
                for line in line_values:
                    line_col = f"{target}_over_{line}"
                    
                    if line_col not in player_data.columns:
                        continue
                        
                    # Calcular tasa de over/under
                    over_rate = player_data[line_col].mean()
                    num_games = len(player_data)
                    
                    # Determinar si la confianza supera el umbral
                    confidence = max(over_rate, 1 - over_rate)
                    prediction = 'over' if over_rate >= 0.5 else 'under'
                    
                    # Si está por encima del umbral, guardar
                    if confidence >= confidence_threshold:
                        player_lines[line] = {
                            'prediction': prediction,
                            'confidence': confidence,
                            'over_rate': over_rate,
                            'num_games': num_games,
                            'line': line,
                            'avg_value': player_data[target].mean(),
                            'std_value': player_data[target].std()
                        }
                
                # Si encontramos líneas de alta confianza, guardar
                if player_lines:
                    player_analysis[player] = {
                        'player': player,
                        'num_games': len(player_data),
                        'lines': player_lines,
                        'avg_value': player_data[target].mean(),
                        'max_value': player_data[target].max(),
                        'min_value': player_data[target].min()
                    }
        
        # Resumen de resultados
        summary = {
            'target': target,
            'players_analyzed': len(df['Player'].unique()) if 'Player' in df.columns else 0,
            'players_with_high_confidence': len(player_analysis),
            'total_high_confidence_lines': sum(len(p['lines']) for p in player_analysis.values()),
            'analysis_by_player': player_analysis
        }
        
        # Mostrar resultados
        logger.info(f"Análisis de {target}: {summary['players_with_high_confidence']} jugadores con alta confianza")
        logger.info(f"Total de líneas de alta confianza: {summary['total_high_confidence_lines']}")
        
        return summary
    
    def find_market_arbitrage_opportunities(
        self,
        player_data: pd.DataFrame,
        min_profit: float = 0.02
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Busca oportunidades de arbitraje entre mercados (no solo entre casas de apuestas)
        
        Args:
            player_data: DataFrame con datos combinados
            min_profit: Ganancia mínima para considerar arbitraje
            
        Returns:
            Diccionario con oportunidades de arbitraje por tipo de estadística
        """
        # Estadísticas a analizar
        targets = ['PTS', 'TRB', 'AST', '3P']
        arbitrage_by_target = {}
        
        # Para cada estadística objetivo
        for target in targets:
            # Usar método de búsqueda de arbitraje
            arbitrage_opportunities = self.find_best_odds_arbitrage(
                player_data, target
            )
            
            if arbitrage_opportunities:
                arbitrage_by_target[target] = arbitrage_opportunities
                
                # Mostrar las 2 mejores oportunidades
                logger.info(f"Encontradas {len(arbitrage_opportunities)} oportunidades de arbitraje para {target}")
                for i, arb in enumerate(arbitrage_opportunities[:2]):
                    logger.info(f"Top {i+1} ({target}): Línea {arb['line']} - "
                               f"Profit: {arb['profit_margin']:.2%} - "
                               f"{arb['over_bookmaker']} vs {arb['under_bookmaker']}")
        
        # Buscar también arbitraje entre mercados correlacionados
        correlated_markets = [
            ('PTS', 'Team_Points_Over_Under'),
            ('TRB', 'Team_Rebounds_Over_Under'),
            ('AST', 'Team_Assists_Over_Under')
        ]
        
        cross_market_arbitrage = []
        
        # TODO: Implementar búsqueda de arbitraje entre mercados correlacionados
        # Requiere análisis más avanzado que está fuera del alcance actual
        
        # Añadir arbitraje entre mercados si encontramos
        if cross_market_arbitrage:
            arbitrage_by_target['cross_market'] = cross_market_arbitrage
        
        return arbitrage_by_target
    
    def generate_kelly_portfolio(
        self,
        player_data: pd.DataFrame,
        min_confidence: float = 0.96,
        max_portfolio_risk: float = 0.25,
        min_edge: float = 0.04
    ) -> Dict[str, Any]:
        """
        Genera un portafolio de apuestas optimizado según criterio Kelly
        
        Este método avanzado:
        1. Aplica una versión modificada del criterio Kelly para optimizar apuestas
        2. Distribuye el capital disponible entre múltiples oportunidades 
        3. Limita el riesgo total del portafolio según parámetros configurables
        4. Pondera apuestas según nivel de confianza y ventaja sobre el mercado
        5. Maximiza el crecimiento esperado del capital a largo plazo
        
        Args:
            player_data: DataFrame con datos de jugadores
            min_confidence: Confianza mínima requerida (0.96 = 96%)
            max_portfolio_risk: Riesgo máximo del portafolio (0.25 = 25%)
            min_edge: Ventaja mínima sobre casas de apuestas
            
        Returns:
            Diccionario con estrategia de portafolio optimizada
        """
        # Estadísticas a analizar
        targets = ['PTS', 'TRB', 'AST', '3P']
        
        # Recolectar todas las apuestas posibles
        all_bets = []
        
        for target in targets:
            # Obtener estrategia óptima para este target
            betting_strategy = self.get_optimal_betting_strategy(
                player_data, target, confidence_threshold=min_confidence
            )
            
            # Extraer apuestas de valor
            for bet_info in betting_strategy.get('recommendations', []):
                if bet_info.get('edge', 0) >= min_edge:
                    # Calcular Kelly fraction
                    kelly_fraction = bet_info.get('kelly_fraction', 0)
                    
                    # Si tiene valor positivo, añadir a la cartera
                    if kelly_fraction > 0:
                        all_bets.append({
                            'target': target,
                            'line': bet_info.get('line', 0),
                            'player': bet_info.get('player', 'Unknown'),
                            'bookmaker': bet_info.get('bookmaker', 'Unknown'),
                            'confidence': bet_info['confidence'],
                            'edge': bet_info['edge'],
                            'kelly_fraction': kelly_fraction,
                            'ev': (1 + bet_info['edge']) * 0.5  # EV simplificado
                        })
        
        # Ordenar apuestas por Kelly fraction
        all_bets = sorted(all_bets, key=lambda x: x['kelly_fraction'], reverse=True)
        
        # Si no hay apuestas, devolver cartera vacía
        if not all_bets:
            return {
                'portfolio_size': 0,
                'total_risk': 0,
                'bets': []
            }
            
        # Calcular riesgo total si usáramos todas las apuestas
        total_kelly = sum(bet['kelly_fraction'] for bet in all_bets)
        
        # Si el riesgo total supera el máximo, ajustar proporciones
        scale_factor = 1.0
        if total_kelly > max_portfolio_risk:
            scale_factor = max_portfolio_risk / total_kelly
            logger.info(f"Ajustando cartera: factor de escala {scale_factor:.2f} para limitar riesgo a {max_portfolio_risk:.2%}")
            
            # Ajustar cada apuesta
            for bet in all_bets:
                bet['adjusted_kelly'] = bet['kelly_fraction'] * scale_factor
        else:
            # Si no hay que ajustar, mantener fracciones originales
            for bet in all_bets:
                bet['adjusted_kelly'] = bet['kelly_fraction']
                
        # Portfolio final
        portfolio = {
            'portfolio_size': len(all_bets),
            'total_risk': sum(bet['adjusted_kelly'] for bet in all_bets),
            'estimated_growth_rate': sum(bet['adjusted_kelly'] * bet['ev'] for bet in all_bets),
            'expected_roi': sum(bet['edge'] * bet['adjusted_kelly'] for bet in all_bets) * 100,
            'bets': all_bets
        }
        
        logger.info(f"Cartera optimizada: {portfolio['portfolio_size']} apuestas, "
                   f"riesgo total {portfolio['total_risk']:.2%}, "
                   f"ROI esperado {portfolio['expected_roi']:.2f}%")
        
        return portfolio
    
    def scrape_bookmaker_data(
        self,
        target: str = 'PTS',
        bookmakers: List[str] = None,
        headless: bool = True
    ) -> Dict[str, Any]:
        """
        Web scraping de datos de odds (implementación básica)
        
        Args:
            target: Estadística objetivo
            bookmakers: Lista de casas a scrapear
            headless: Si usar navegador sin interfaz gráfica
            
        Returns:
            Diccionario con datos obtenidos
        """
        logger.warning("El scraping de datos de odds no está completamente implementado")
        logger.warning("Se recomienda usar las APIs oficiales o archivos de datos")
        
        # Placeholder - esta función requeriría una implementación más compleja
        # con Selenium o similar para el scraping real
        
        return {
            'success': False,
            'error': 'Scraping no implementado completamente',
            'data': []
        } 

    def __repr__(self) -> str:
        """Representación detallada del sistema."""
        return self.__str__()
    
    # === MÉTODO PRINCIPAL PARA FLUJO DIRECTO ===
    
    def get_best_prediction_odds(
        self,
        predictions_data: pd.DataFrame,
        target: str,
        date: Optional[str] = None,
        min_confidence: float = 0.90
    ) -> Dict[str, Any]:
        """
        MÉTODO PRINCIPAL: Recibe predicciones → Obtiene cuotas Sportradar → Devuelve mejor combinación.
        
        Flujo directo:
        1. Recibe DataFrame con predicciones del modelo
        2. Obtiene cuotas desde Sportradar API 
        3. Encuentra la mejor combinación predicción/cuota
        4. Devuelve resultado con recomendación de apuesta
        
        Args:
            predictions_data: DataFrame con predicciones (debe tener columnas: Player, Team, {target}, {target}_confidence)
            target: Estadística objetivo (PTS, AST, TRB, 3P)
            date: Fecha específica (YYYY-MM-DD), si None usa hoy
            min_confidence: Confianza mínima para considerar predicción
            
        Returns:
            Dict con mejor combinación predicción/cuota y datos para apostar
        """
        logger.info(f"INICIANDO FLUJO PRINCIPAL: Predicciones → Sportradar → Mejor cuota para {target}")
        
        try:
            # PASO 1: Validar predicciones
            required_columns = ['Player', 'Team', target, f'{target}_confidence']
            missing_cols = [col for col in required_columns if col not in predictions_data.columns]
            if missing_cols:
                raise DataValidationError(f"Faltan columnas requeridas: {missing_cols}")
            
            # Filtrar por confianza mínima
            high_confidence_predictions = predictions_data[
                predictions_data[f'{target}_confidence'] >= min_confidence
            ].copy()
            
            if high_confidence_predictions.empty:
                return {
                    'success': False,
                    'error': f'No hay predicciones con confianza >= {min_confidence:.1%}',
                    'total_predictions': len(predictions_data)
                }
            
            logger.info(f"Predicciones válidas: {len(high_confidence_predictions)}/{len(predictions_data)}")
            
            # PASO 2: Obtener cuotas desde Sportradar
            logger.info("Obteniendo cuotas desde Sportradar...")
            sportradar_data = self.bookmakers_fetcher.get_nba_odds_from_sportradar(
                date=date, 
                include_props=True
            )
            
            if not sportradar_data.get('success', False):
                error_msg = "Error obteniendo cuotas reales de Sportradar"
                logger.error(error_msg)
                logger.error("El sistema requiere datos reales de la API - no se permiten simulaciones")
                return {
                    'success': False,
                    'error': error_msg,
                    'requires_real_data': True,
                    'sportradar_error': sportradar_data.get('error', 'Unknown error')
                }
            
            logger.info(f"Cuotas obtenidas: {sportradar_data['games_with_odds']} juegos")
            
            # PASO 3: Encontrar mejores combinaciones
            best_combinations = self._find_best_prediction_odds_combinations(
                high_confidence_predictions, 
                sportradar_data, 
                target
            )
            
            if not best_combinations:
                return {
                    'success': False,
                    'error': 'No se encontraron combinaciones válidas predicción/cuota',
                    'sportradar_games': sportradar_data.get('games_with_odds', 0)
                }
            
            # PASO 4: Seleccionar la mejor opción
            best_bet = self._select_absolute_best_bet(best_combinations)
            
            # PASO 5: Calcular stake recomendado
            recommended_stake = self._calculate_optimal_stake(best_bet)
            
            # PASO 6: Compilar resultado final
            result = {
                'success': True,
                'timestamp': datetime.now().isoformat(),
                'target': target,
                'date': date or datetime.now().strftime('%Y-%m-%d'),
                'best_bet': {
                    **best_bet,
                    'recommended_stake': recommended_stake,
                    'potential_profit': recommended_stake * (best_bet['decimal_odds'] - 1),
                    'roi_percentage': ((best_bet['decimal_odds'] - 1) * best_bet['win_probability'] - 
                                     (1 - best_bet['win_probability'])) * 100
                },
                'alternatives': best_combinations[1:6],  # Top 5 alternativas
                'market_summary': {
                    'total_predictions_analyzed': len(high_confidence_predictions),
                    'total_odds_found': len(best_combinations),
                    'average_confidence': high_confidence_predictions[f'{target}_confidence'].mean(),
                    'best_edge_found': best_bet['edge']
                }
            }
            
            logger.info(f"MEJOR APUESTA ENCONTRADA: {best_bet['player']} {best_bet['bet_type']} {best_bet['line']} | Edge: {best_bet['edge']:.1%}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error en flujo principal: {e}")
            return {
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def _find_best_prediction_odds_combinations(
        self,
        predictions: pd.DataFrame,
        sportradar_data: Dict[str, Any],
        target: str
    ) -> List[Dict[str, Any]]:
        """Encuentra todas las combinaciones válidas predicción/cuota."""
        combinations = []
        
        # Extraer juegos con props
        games = sportradar_data.get('games', [])
        
        for game in games:
            if not game.get('props'):
                continue
                
            # Extraer props para el target específico
            target_props = self._extract_target_props(game['props'], target)
            
            if not target_props:
                continue
            
            # Para cada jugador en nuestras predicciones
            for _, pred_row in predictions.iterrows():
                player_name = pred_row['Player']
                prediction = pred_row[target]
                confidence = pred_row[f'{target}_confidence']
                
                # Buscar props para este jugador
                player_props = [
                    prop for prop in target_props 
                    if self._player_name_matches(player_name, prop.get('player', ''))
                ]
                
                # Evaluar cada prop
                for prop in player_props:
                    line = prop.get('line', 0)
                    over_odds = prop.get('over_odds', 0)
                    under_odds = prop.get('under_odds', 0)
                    bookmaker = prop.get('bookmaker', 'unknown')
                    
                    # Evaluar OVER
                    if over_odds > 0 and prediction > line:
                        win_prob = self._calculate_model_probability(prediction, line, target)
                        market_prob = self._odds_to_probability(over_odds)
                        edge = win_prob - market_prob
                        
                        if edge > self.minimum_edge:
                            combinations.append({
                                'player': player_name,
                                'team': pred_row['Team'],
                                'target': target,
                                'line': line,
                                'bet_type': 'over',
                                'prediction': prediction,
                                'confidence': confidence,
                                'win_probability': win_prob,
                                'market_probability': market_prob,
                                'edge': edge,
                                'american_odds': over_odds,
                                'decimal_odds': self._american_to_decimal(over_odds),
                                'bookmaker': bookmaker,
                                'game_id': game.get('game_id'),
                                'opponent': game.get('away_team') if pred_row['Team'] in game.get('home_team', '') else game.get('home_team')
                            })
                    
                    # Evaluar UNDER
                    if under_odds > 0 and prediction < line:
                        win_prob = 1 - self._calculate_model_probability(prediction, line, target)
                        market_prob = self._odds_to_probability(under_odds)
                        edge = win_prob - market_prob
                        
                        if edge > self.minimum_edge:
                            combinations.append({
                                'player': player_name,
                                'team': pred_row['Team'],
                                'target': target,
                                'line': line,
                                'bet_type': 'under',
                                'prediction': prediction,
                                'confidence': confidence,
                                'win_probability': win_prob,
                                'market_probability': market_prob,
                                'edge': edge,
                                'american_odds': under_odds,
                                'decimal_odds': self._american_to_decimal(under_odds),
                                'bookmaker': bookmaker,
                                'game_id': game.get('game_id'),
                                'opponent': game.get('away_team') if pred_row['Team'] in game.get('home_team', '') else game.get('home_team')
                            })
        
        # Ordenar por edge descendente
        combinations.sort(key=lambda x: x['edge'], reverse=True)
        return combinations
    
    def _player_name_matches(self, pred_name: str, prop_name: str) -> bool:
        """Verifica si los nombres de jugador coinciden (manejo de variaciones)."""
        pred_clean = pred_name.lower().strip()
        prop_clean = prop_name.lower().strip()
        
        # Coincidencia exacta
        if pred_clean == prop_clean:
            return True
        
        # Coincidencia por palabras (apellido principalmente)
        pred_words = pred_clean.split()
        prop_words = prop_clean.split()
        
        # Si al menos el apellido coincide
        if pred_words and prop_words:
            return pred_words[-1] == prop_words[-1]
        
        return False
    
    def _american_to_decimal(self, american_odds: float) -> float:
        """Convierte odds americanas a decimales."""
        if american_odds > 0:
            return 1 + (american_odds / 100)
        else:
            return 1 + (100 / abs(american_odds))
    
    def _select_absolute_best_bet(self, combinations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Selecciona la mejor apuesta considerando edge, confianza y odds."""
        if not combinations:
            raise ValueError("No hay combinaciones para seleccionar")
        
        # Función de puntuación que combina múltiples factores
        def score_combination(combo):
            edge_score = combo['edge'] * 100  # Edge base
            confidence_boost = (combo['confidence'] - 0.9) * 50  # Boost por confianza alta
            odds_penalty = max(0, combo['decimal_odds'] - 3) * -5  # Penalizar odds muy altas
            
            return edge_score + confidence_boost + odds_penalty
        
        # Calcular scores y seleccionar el mejor
        for combo in combinations:
            combo['composite_score'] = score_combination(combo)
        
        best_bet = max(combinations, key=lambda x: x['composite_score'])
        return best_bet
    
    def _calculate_optimal_stake(self, best_bet: Dict[str, Any]) -> float:
        """Calcula stake óptimo usando Kelly Criterion modificado."""
        win_prob = best_bet['win_probability']
        decimal_odds = best_bet['decimal_odds']
        
        # Kelly Criterion: f = (bp - q) / b
        b = decimal_odds - 1  # ganancia neta por unidad apostada
        q = 1 - win_prob      # probabilidad de perder
        
        if b > 0:
            kelly_fraction = (win_prob * b - q) / b
            # Limitar Kelly a máximo configurado
            kelly_fraction = max(0, min(kelly_fraction, self.max_kelly_fraction))
        else:
            kelly_fraction = 0
        
        # Calcular stake en dinero
        stake = kelly_fraction * self.bankroll
        
        # Aplicar límites configurados
        min_bet = self.config.get('betting', 'min_bet_amount')
        max_bet = self.config.get('betting', 'max_bet_amount')
        
        return max(min_bet, min(stake, max_bet))
    
    def _process_simulated_best_bet(self, simulated_df: pd.DataFrame, target: str) -> Dict[str, Any]:
        """Procesa datos simulados para encontrar mejor apuesta."""
        # Buscar columnas de odds simuladas
        odds_columns = [col for col in simulated_df.columns if f'{target}_over_' in col and '_odds_' in col]
        
        if not odds_columns:
            return {
                'success': False,
                'error': 'No se encontraron odds simuladas',
                'simulated': True
            }
        
        best_edge = -1
        best_bet = None
        
        for _, row in simulated_df.iterrows():
            prediction = row.get(target, 0)
            confidence = row.get(f'{target}_confidence', 0)
            
            for odds_col in odds_columns:
                # Extraer información de la columna
                parts = odds_col.split('_')
                if len(parts) >= 4:
                    line = float(parts[2])
                    bookmaker = parts[-1]
                    
                    odds = row.get(odds_col, 0)
                    if odds <= 0:
                        continue
                    
                    # Calcular edge para OVER
                    if prediction > line:
                        win_prob = self._calculate_model_probability(prediction, line, target)
                        market_prob = 1 / odds  # odds decimales simuladas
                        edge = win_prob - market_prob
                        
                        if edge > best_edge:
                            best_edge = edge
                            best_bet = {
                                'player': row['Player'],
                                'team': row['Team'],
                                'target': target,
                                'line': line,
                                'bet_type': 'over',
                                'prediction': prediction,
                                'confidence': confidence,
                                'win_probability': win_prob,
                                'market_probability': market_prob,
                                'edge': edge,
                                'decimal_odds': odds,
                                'bookmaker': bookmaker,
                                'simulated': True
                            }
        
        if best_bet:
            stake = self._calculate_optimal_stake(best_bet)
            return {
                'success': True,
                'timestamp': datetime.now().isoformat(),
                'best_bet': {
                    **best_bet,
                    'recommended_stake': stake,
                    'potential_profit': stake * (best_bet['decimal_odds'] - 1)
                },
                'simulated': True,
                'note': 'Datos simulados - no usar para apuestas reales'
            }
        
        return {
            'success': False,
            'error': 'No se encontró ninguna combinación válida en simulación'
        } 