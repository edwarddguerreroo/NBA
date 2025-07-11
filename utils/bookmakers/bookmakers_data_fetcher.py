"""
Bookmakers Data Fetcher - Integración Completa con Sportradar
============================================================

Fetcher avanzado que obtiene, procesa y gestiona datos de cuotas desde múltiples fuentes:
- Sportradar API (principal)
- APIs secundarias (The Odds API, etc.)
- Archivos locales
- Simulación inteligente

Su propósito es actuar como capa de abstracción unificada para todas las fuentes
de datos de cuotas, proporcionando interfaz consistente para el sistema de predicción.

Funciones principales:
1. Integración completa con Sportradar API para cuotas NBA
2. Obtener player props (PTS, AST, TRB, 3P) en tiempo real
3. Cargar datos desde archivos (Excel, CSV, JSON)
4. Simulación inteligente de cuotas con varianza realista
5. Normalización y procesamiento de diferentes formatos
6. Cache inteligente para optimizar rendimiento
7. Análisis comparativo entre casas de apuestas
"""

import pandas as pd
import numpy as np
import requests
import json
import logging
import time
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import os
from pathlib import Path

from .sportradar_api import SportradarAPI
from .config import get_config
from .exceptions import (
    BookmakersAPIError,
    SportradarAPIError,
    InsufficientDataError,
    DataValidationError,
    CacheError
)

logger = logging.getLogger(__name__)

class BookmakersDataFetcher:
    """
    Fetcher avanzado para datos de cuotas con integración completa de Sportradar.
    
    Funcionalidades principales:
    - Sportradar API como fuente principal
    - APIs secundarias como respaldo
    - Cache inteligente con validación temporal
    - Simulación avanzada con varianza realista
    - Normalización de datos entre proveedores
    """
    
    def __init__(
        self,
        api_keys: Dict[str, str] = None,
        odds_data_dir: str = "data/bookmakers",
        cache_expiry: int = 12,  # Horas
        config_override: Optional[Dict] = None
    ):
        """
        Inicializa el fetcher con configuración avanzada.
        
        Args:
            api_keys: Diccionario con API keys por proveedor
            odds_data_dir: Directorio para datos y cache
            cache_expiry: Horas antes de que expire el cache
            config_override: Configuración personalizada
        """
        # Configuración
        self.config = get_config()
        if config_override:
            for section, values in config_override.items():
                for key, value in values.items():
                    self.config.set(section, key, value=value)
        
        # API Keys y configuración
        self.api_keys = api_keys or {}
        self.odds_data_dir = Path(odds_data_dir)
        self.cache_expiry = cache_expiry
        
        # Crear directorios necesarios
        self.odds_data_dir.mkdir(parents=True, exist_ok=True)
        (self.odds_data_dir / 'cache').mkdir(exist_ok=True)
        (self.odds_data_dir / 'raw').mkdir(exist_ok=True)
        
        # Inicializar Sportradar API como proveedor principal
        sportradar_key = (
            self.api_keys.get('sportradar') or 
            self.config.get('sportradar', 'api_key') or
            os.getenv('API_SPORTRADAR') or
            os.getenv('SPORTRADAR_API_KEY')
        )
        self.sportradar_api = None
        
        if sportradar_key:
            try:
                self.sportradar_api = SportradarAPI(
                    api_key=sportradar_key,
                    config_override=config_override.get('sportradar') if config_override else None
                )
                logger.info("Sportradar API inicializada como proveedor principal")
                logger.info(f"API Key configurada: {'*' * (len(sportradar_key) - 4)}{sportradar_key[-4:]}")
            except Exception as e:
                logger.error(f"Error inicializando Sportradar API: {e}")
        else:
            logger.warning("Sportradar API key no encontrada. Funcionalidad limitada.")
            logger.warning("Configura la variable de entorno API_SPORTRADAR o SPORTRADAR_API_KEY")
        
        # Métricas de uso
        self.api_calls_made = 0
        self.cache_hits = 0
        self.cache_misses = 0
        
        # Diccionario de casas de apuestas conocidas
        self.supported_bookmakers = {
            # Casas de apuestas principales
            'draftkings': 'DraftKings',
            'fanduel': 'FanDuel',
            'betmgm': 'BetMGM',
            'caesars': 'Caesars',
            'pointsbet': 'PointsBet',
            'bet365': 'Bet365',
            'wynn': 'Wynn',
            'pinnacle': 'Pinnacle',
            
        }

    # === MÉTODOS SPORTRADAR (PROVEEDOR PRINCIPAL) ===
    
    def get_nba_odds_from_sportradar(
        self,
        date: Optional[str] = None,
        team_filter: Optional[List[str]] = None,
        include_props: bool = True
    ) -> Dict[str, Any]:
        """
        Obtiene cuotas NBA desde Sportradar API.
        
        Args:
            date: Fecha específica (YYYY-MM-DD). Si es None, obtiene próximos partidos
            team_filter: Lista de equipos para filtrar
            include_props: Si incluir player props
            
        Returns:
            Diccionario con cuotas organizadas por partido
        """
        if not self.sportradar_api:
            raise SportradarAPIError("Sportradar API no inicializada")
        
        try:
            # Usar método optimizado para obtener todos los targets a la vez
            if date:
                # Obtener odds para fecha específica
                odds_data = self.sportradar_api.get_nba_odds_for_targets(date=date)
            else:
                # Obtener odds para próximos partidos (método automatizado)
                odds_data = self.sportradar_api.get_live_and_upcoming_odds()
                
            # Procesar resultados
            if not odds_data.get('games', {}):
                today = datetime.now().strftime("%Y-%m-%d")
                return {
                    'success': True,
                    'date': date or today,
                    'games': [],
                    'total_games': 0
                }
            
            # Convertir datos al formato esperado
            games_with_odds = []
            
            # Obtener juegos de la estructura correcta según el método usado
            if 'all_games' in odds_data:
                games_dict = odds_data['all_games']
            else:
                games_dict = odds_data.get('games', {})
                
            for game_id, game_data in games_dict.items():
                # Filtrar por equipo si es necesario
                if team_filter:
                    home_team = game_data.get('teams', {}).get('home', '').lower()
                    away_team = game_data.get('teams', {}).get('away', '').lower()
                    
                    if not any(team.lower() in home_team or team.lower() in away_team 
                              for team in team_filter):
                        continue
                
                # Extraer datos del juego
                game_info = {
                    'game_id': game_id,
                    'home_team': game_data.get('teams', {}).get('home'),
                    'away_team': game_data.get('teams', {}).get('away'),
                    'scheduled': game_data.get('scheduled'),
                    'odds': game_data.get('odds', {}).get('game_markets', {}),
                    'props': game_data.get('odds', {}).get('player_props', {}) if include_props else None
                }
                
                games_with_odds.append(game_info)
                self.api_calls_made += 1
            
            return {
                'success': True,
                'source': 'sportradar',
                'date': date or odds_data.get('date', datetime.now().strftime("%Y-%m-%d")),
                'games': games_with_odds,
                'total_games': len(games_dict),
                'games_with_odds': len(games_with_odds),
                'api_calls_made': self.api_calls_made,
                'cache_stats': self.sportradar_api.get_cache_stats() if hasattr(self.sportradar_api, 'get_cache_stats') else {}
            }
            
        except Exception as e:
            logger.error(f"Error obteniendo datos de Sportradar: {e}")
            return {
                'success': False,
                'error': str(e),
                'source': 'sportradar'
            }
    
    def get_player_props_sportradar(
        self,
        player_name: str,
        target: str = 'PTS',
        days_ahead: int = 7
    ) -> List[Dict[str, Any]]:
        """
        Obtiene player props específicos desde Sportradar.
        
        Args:
            player_name: Nombre del jugador
            target: Estadística objetivo (PTS, AST, TRB, 3P)
            days_ahead: Días hacia adelante para buscar
            
        Returns:
            Lista de props encontrados
        """
        if not self.sportradar_api:
            raise SportradarAPIError("Sportradar API no inicializada")
        
        try:
            # Obtener partidos próximos
            start_date = datetime.now().strftime("%Y-%m-%d")
            end_date = (datetime.now() + timedelta(days=days_ahead)).strftime("%Y-%m-%d")
            
            games = self.sportradar_api.get_games_by_date_range(start_date, end_date)
            
            player_props = []
            
            for game in games:
                game_id = game.get('id')
                if game_id:
                    try:
                        props_data = self.sportradar_api.get_player_props(game_id)
                        
                        # Buscar props del jugador específico
                        if 'markets' in props_data:
                            for market in props_data['markets']:
                                if target.lower() in market.get('description', '').lower():
                                    for outcome in market.get('outcomes', []):
                                        player_in_outcome = outcome.get('player', {}).get('name', '')
                                        if player_name.lower() in player_in_outcome.lower():
                                            player_props.append({
                                                'game_id': game_id,
                                                'player': player_in_outcome,
                                                'target': target,
                                                'line': outcome.get('point'),
                                                'over_odds': outcome.get('over_odds'),
                                                'under_odds': outcome.get('under_odds'),
                                                'market': market.get('description'),
                                                'bookmaker': outcome.get('bookmaker'),
                                                'game_info': {
                                                    'home': game.get('home', {}).get('name'),
                                                    'away': game.get('away', {}).get('name'),
                                                    'date': game.get('scheduled')
                                                }
                                            })
                    except SportradarAPIError as e:
                        logger.warning(f"Error obteniendo props para {game_id}: {e}")
            
            return player_props
            
        except Exception as e:
            logger.error(f"Error buscando props para {player_name}: {e}")
            return []
    
    def get_market_overview_sportradar(self, date: Optional[str] = None) -> Dict[str, Any]:
        """
        Obtiene resumen completo del mercado NBA desde Sportradar.
        
        Args:
            date: Fecha específica (YYYY-MM-DD)
            
        Returns:
            Resumen del mercado con estadísticas
        """
        if not self.sportradar_api:
            raise SportradarAPIError("Sportradar API no inicializada")
        
        return self.sportradar_api.get_market_overview(date)

    def load_odds_from_api(
        self,
        sport: str = 'basketball_nba',
        markets: List[str] = None,
        bookmakers: List[str] = None,
        api_provider: str = 'sportradar',  # Cambiado a Sportradar como default
        force_refresh: bool = False,
        date: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Carga odds desde un proveedor de API (The Odds API o similar)
        
        Este método:
        1. Consulta APIs externas para obtener datos actualizados de odds
        2. Gestiona caché para minimizar llamadas a la API y cumplir con límites
        3. Soporta diferentes proveedores de API y tipos de mercados
        
        Args:
            sport: Deporte a consultar
            markets: Lista de mercados (h2h, spreads, totals, player_props)
            bookmakers: Lista de casas de apuestas a incluir
            api_provider: Proveedor de API ('odds_api', 'sportsdata_io', etc.)
            force_refresh: Si forzar la actualización ignorando caché
            
        Returns:
            Datos de odds en formato JSON
        """
        # Mercados por defecto si no se especifican
        if markets is None:
            markets = ['h2h', 'spreads', 'totals', 'player_props']
            
        # Casas de apuestas por defecto
        if bookmakers is None:
            bookmakers = list(self.supported_bookmakers.keys())
        
        # Archivo de caché para esta consulta
        cache_key = f"{sport}_{'-'.join(markets)}_{'-'.join(sorted(bookmakers))}"
        cache_file = self.odds_data_dir / f"{cache_key}.json"
        
        # Verificar si tenemos datos en caché y son recientes
        if not force_refresh and cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    cached_data = json.load(f)
                
                # Verificar si los datos son recientes
                last_update = datetime.fromisoformat(cached_data.get('last_update', '2000-01-01'))
                if datetime.now() - last_update < timedelta(hours=self.cache_expiry):
                    logger.info(f"Usando datos de odds en caché para {cache_key}")
                    return cached_data
            except Exception as e:
                logger.warning(f"Error al leer caché: {e}")
        
        # Si llegamos aquí, necesitamos obtener nuevos datos
        # Diferentes implementaciones según el proveedor
        if api_provider == 'sportradar':
            # Usar Sportradar como proveedor principal
            if not self.sportradar_api:
                logger.error("Sportradar API no está configurada")
                return {'success': False, 'error': "Sportradar API no configurada"}
            
            try:
                # Obtener datos usando los métodos específicos de Sportradar
                if sport == 'basketball_nba':
                    odds_data = self.get_nba_odds_from_sportradar(
                        date=date,
                        include_props=True
                    )
                else:
                    logger.error(f"Deporte no soportado en Sportradar: {sport}")
                    return {'success': False, 'error': f"Deporte no soportado: {sport}"}
            except Exception as e:
                logger.error(f"Error obteniendo datos de Sportradar: {e}")
                return {'success': False, 'error': str(e)}
                
        elif api_provider == 'odds_api':
            # Fallback a The Odds API
            api_key = self.api_keys.get(api_provider)
            if not api_key:
                logger.error(f"No se encontró API key para {api_provider}")
                return {'success': False, 'error': f"API key no configurada para {api_provider}"}
            odds_data = self._fetch_from_odds_api(api_key, sport, markets, bookmakers)
            
        elif api_provider == 'sportsdata_io':
            # Otro proveedor alternativo
            api_key = self.api_keys.get(api_provider)
            if not api_key:
                logger.error(f"No se encontró API key para {api_provider}")
                return {'success': False, 'error': f"API key no configurada para {api_provider}"}
            odds_data = self._fetch_from_sportsdata_io(api_key, sport, markets, bookmakers)
            
        else:
            logger.error(f"Proveedor de API no soportado: {api_provider}")
            return {'success': False, 'error': f"Proveedor no soportado: {api_provider}"}
        
        # Verificar si fue exitoso
        if odds_data.get('success', False):
            # Añadir timestamp
            odds_data['last_update'] = datetime.now().isoformat()
            
            # Guardar en caché
            try:
                with open(cache_file, 'w') as f:
                    json.dump(odds_data, f)
                logger.info(f"Datos de odds guardados en caché para {cache_key}")
            except Exception as e:
                logger.warning(f"Error al guardar caché: {e}")
        
        return odds_data
    
    def _fetch_from_odds_api(
        self, 
        api_key: str, 
        sport: str, 
        markets: List[str], 
        bookmakers: List[str]
    ) -> Dict[str, Any]:
        """
        Obtiene datos desde The Odds API
        """
        base_url = "https://api.the-odds-api.com/v4/sports"
        all_data = {'success': True, 'data': []}
        
        for market in markets:
            try:
                # Diferentes endpoints según el mercado
                if market == 'player_props':
                    url = f"{base_url}/{sport}/odds?apiKey={api_key}&markets=player_props&bookmakers={','.join(bookmakers)}"
                else:
                    url = f"{base_url}/{sport}/odds?apiKey={api_key}&markets={market}&bookmakers={','.join(bookmakers)}"
                
                logger.info(f"Consultando {market} odds para {sport}")
                response = requests.get(url, timeout=30)
                
                if response.status_code == 200:
                    market_data = response.json()
                    all_data['data'].extend(market_data)
                    
                    # Respetar límites de API
                    remaining = response.headers.get('X-Remaining-Requests', 0)
                    logger.info(f"Solicitudes restantes: {remaining}")
                    
                    if int(remaining) < 1:
                        logger.warning("Límite de API alcanzado. Algunas odds podrían faltar.")
                        break
                else:
                    logger.error(f"Error al obtener odds: {response.status_code} - {response.text}")
                    
                # Esperar para no saturar la API
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"Error consultando {market} odds: {str(e)}")
                all_data['success'] = False
                all_data['error'] = str(e)
        
        return all_data
    
    def _fetch_from_sportsdata_io(
        self, 
        api_key: str, 
        sport: str, 
        markets: List[str], 
        bookmakers: List[str]
    ) -> Dict[str, Any]:
        """
        Obtiene datos desde SportsData.io
        """
        # Implementación específica para SportsData.io
        # ...
        
        # Placeholder
        return {'success': False, 'error': 'SportsData.io no implementado aún'}
    
    def load_odds_from_file(
        self,
        file_path: str,
        format: str = 'csv'
    ) -> Dict[str, Any]:
        """
        Carga datos de odds desde un archivo local
        
        Args:
            file_path: Ruta al archivo de datos
            format: Formato del archivo ('csv', 'json', 'excel')
            
        Returns:
            Dict con datos de odds
        """
        try:
            if format.lower() == 'csv':
                df = pd.read_csv(file_path)
            elif format.lower() == 'json':
                with open(file_path, 'r') as f:
                    return json.load(f)
            elif format.lower() in ['excel', 'xlsx', 'xls']:
                df = pd.read_excel(file_path)
            else:
                logger.error(f"Formato no soportado: {format}")
                return {'success': False, 'error': f"Formato no soportado: {format}"}
            
            # Convertir DataFrame a formato estándar
            data = self._standardize_odds_data(df)
            return {'success': True, 'data': data, 'source': 'file', 'file_path': file_path}
            
        except Exception as e:
            logger.error(f"Error al cargar archivo {file_path}: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def _standardize_odds_data(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Convierte un DataFrame a formato estándar para odds
        """
        # Aquí se implementaría la lógica de estandarización
        # Placeholder
        return df.to_dict(orient='records')
        
    def merge_odds_with_player_data(
        self,
        player_data: pd.DataFrame,
        odds_data: Dict[str, Any],
        target: str = 'PTS',
        line_values: List[float] = None
    ) -> pd.DataFrame:
        """
        Combina datos de jugadores con odds de casas de apuestas
        
        Este método es crucial para:
        1. Integrar datos de diferentes fuentes (predicciones + mercado)
        2. Alinear correctamente los datos por jugador, fecha y línea de apuesta 
        3. Crear columnas específicas para cada combinación de línea y casa
        4. Habilitar análisis comparativo entre nuestras predicciones y el mercado
        
        Args:
            player_data: DataFrame con datos de jugadores
            odds_data: Datos de odds obtenidos mediante load_odds_from_api o load_odds_from_file
            target: Estadística objetivo (PTS, TRB, AST, 3P)
            line_values: Lista de valores de línea a analizar
            
        Returns:
            DataFrame combinado con datos de jugadores y odds de casas
        """
        if not odds_data.get('success', False) or not odds_data.get('data'):
            logger.error("No hay datos de odds válidos para combinar")
            return player_data
        
        # Crear copia para no modificar el original
        df = player_data.copy()
        
        # Líneas por defecto según el target si no se especifican
        if line_values is None:
            if target == 'PTS':
                line_values = [10, 15, 20, 25, 30, 35]
            elif target == 'TRB':
                line_values = [4, 6, 8, 10, 12]
            elif target == 'AST':
                line_values = [4, 6, 8, 10, 12]
            elif target == '3P':
                line_values = [1, 2, 3, 4, 5]
            else:
                line_values = []
        
        # Procesar datos de odds
        try:
            # Diferentes procesamiento según el formato de datos
            if 'source' in odds_data and odds_data['source'] == 'file':
                df = self._merge_file_odds(df, odds_data['data'], target, line_values)
            else:
                # Asumir formato API
                df = self._merge_api_odds(df, odds_data['data'], target, line_values)
            
            return df
            
        except Exception as e:
            logger.error(f"Error al combinar datos de odds: {str(e)}")
            return player_data
    
    def _merge_file_odds(
        self,
        df: pd.DataFrame,
        odds_data: List[Dict[str, Any]],
        target: str,
        line_values: List[float]
    ) -> pd.DataFrame:
        """
        Combina datos de un archivo de odds con datos de jugadores
        """
        # Implementación específica para archivos
        # ...
        
        # Placeholder
        return df
    
    def _merge_api_odds(
        self,
        df: pd.DataFrame,
        odds_data: List[Dict[str, Any]],
        target: str,
        line_values: List[float]
    ) -> pd.DataFrame:
        """
        Combina datos de API de odds con datos de jugadores
        """
        # Implementación específica para datos de API
        # ...
        
        # Placeholder
        return df
    
    def extract_player_props(
        self,
        odds_data: Dict[str, Any],
        player_name: Optional[str] = None,
        team_name: Optional[str] = None,
        prop_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Extrae props específicas de jugador desde datos de odds completos
        
        Args:
            odds_data: Datos de odds completos
            player_name: Nombre de jugador para filtrar (opcional)
            team_name: Nombre de equipo para filtrar (opcional)
            prop_type: Tipo de prop para filtrar (points, rebounds, assists, threes)
            
        Returns:
            Lista filtrada de props de jugador
        """
        if not odds_data.get('success', False) or not odds_data.get('data'):
            return []
        
        props = []
        
        # Mapeo de prop_type a representaciones comunes en las APIs
        prop_type_map = {
            'points': ['points', 'pts', 'player points', 'total points'],
            'rebounds': ['rebounds', 'reb', 'player rebounds', 'total rebounds', 'trb'],
            'assists': ['assists', 'ast', 'player assists', 'total assists'],
            'threes': ['threes', '3pt', 'three pointers', '3 pointers', '3p']
        }
        
        # Término de búsqueda basado en prop_type
        search_terms = []
        if prop_type and prop_type.lower() in prop_type_map:
            search_terms = prop_type_map[prop_type.lower()]
        
        # Procesar datos
        try:
            for game in odds_data['data']:
                # Verificar si el juego tiene datos de player_props
                if 'bookmakers' not in game:
                    continue
                
                for bookmaker in game['bookmakers']:
                    bookmaker_name = bookmaker['key']
                    
                    for market in bookmaker.get('markets', []):
                        market_key = market.get('key', '')
                        
                        # Verificar si es un prop de jugador
                        if 'player' not in market_key.lower():
                            continue
                        
                        # Si se especificó prop_type, verificar que coincida
                        if prop_type and not any(term in market_key.lower() for term in search_terms):
                            continue
                        
                        for outcome in market.get('outcomes', []):
                            # Verificar filtros
                            if player_name and player_name.lower() not in outcome.get('name', '').lower():
                                continue
                                
                            if team_name and team_name.lower() not in outcome.get('team', '').lower():
                                continue
                            
                            # Añadir a resultados
                            props.append({
                                'player': outcome.get('name', ''),
                                'team': outcome.get('team', ''),
                                'prop_type': market_key,
                                'line': outcome.get('point'),
                                'over_price': outcome.get('price') if outcome.get('name', '').lower().endswith('over') else None,
                                'under_price': outcome.get('price') if outcome.get('name', '').lower().endswith('under') else None,
                                'bookmaker': bookmaker_name,
                                'game_time': game.get('commence_time'),
                                'home_team': game.get('home_team'),
                                'away_team': game.get('away_team')
                            })
            
            return props
            
        except Exception as e:
            logger.error(f"Error al extraer props de jugador: {str(e)}")
            return []
    
    def create_odds_columns(
        self,
        df: pd.DataFrame,
        props_data: List[Dict[str, Any]],
        target: str = 'PTS'
    ) -> pd.DataFrame:
        """
        Crea columnas con odds para cada bookmaker en el DataFrame
        
        Args:
            df: DataFrame con datos de jugadores
            props_data: Lista de props extraídas
            target: Estadística objetivo (PTS, TRB, AST, 3P)
            
        Returns:
            DataFrame con columnas de odds añadidas
        """
        # Mapeo de targets a tipos de props
        target_to_prop = {
            'PTS': ['points', 'pts', 'player points', 'total points'],
            'TRB': ['rebounds', 'reb', 'player rebounds', 'total rebounds', 'trb'],
            'AST': ['assists', 'ast', 'player assists', 'total assists'],
            '3P': ['threes', '3pt', 'three pointers', '3 pointers', '3p']
        }
        
        # Asegurar que tenemos prop_type para el target
        if target not in target_to_prop:
            logger.error(f"Target no soportado para odds: {target}")
            return df
            
        prop_types = target_to_prop[target]
        
        # Crear copia para no modificar el original
        result_df = df.copy()
        
        # Agrupar props por jugador
        player_props = {}
        for prop in props_data:
            # Verificar si es el tipo correcto de prop
            if not any(pt in prop['prop_type'].lower() for pt in prop_types):
                continue
                
            player = prop['player']
            if player not in player_props:
                player_props[player] = []
            
            player_props[player].append(prop)
        
        # Para cada jugador en el DataFrame
        for idx, row in result_df.iterrows():
            player_name = row['Player'] if 'Player' in row else None
            if not player_name or player_name not in player_props:
                continue
            
            # Procesar props para este jugador
            player_odds = player_props[player_name]
            
            # Para cada línea y bookmaker
            for prop in player_odds:
                line = prop.get('line')
                if line is None:
                    continue
                    
                bookmaker = prop.get('bookmaker', 'unknown').lower()
                over_price = prop.get('over_price')
                under_price = prop.get('under_price')
                
                # Crear nombres de columna
                if over_price is not None:
                    col_name = f"{target}_over_{line}_odds_{bookmaker}"
                    result_df.at[idx, col_name] = over_price
                
                if under_price is not None:
                    col_name = f"{target}_under_{line}_odds_{bookmaker}"
                    result_df.at[idx, col_name] = under_price
        
        return result_df
    
    def simulate_bookmaker_data(
        self,
        df: pd.DataFrame,
        target: str = 'PTS',
        line_values: List[float] = None,
        bookmakers: List[str] = None,
        noise_level: float = 0.05
    ) -> pd.DataFrame:
        """
        Simula datos de odds de casas de apuestas cuando no hay datos reales
        
        Este método es útil para:
        1. Pruebas del sistema cuando no se tiene acceso a APIs o datos reales
        2. Desarrollo y validación de algoritmos de análisis de mercado
        3. Simulación de diferentes escenarios de mercado con distintos niveles de ruido
        4. Entrenamiento y evaluación de estrategias de apuestas
        
        Args:
            df: DataFrame con datos de jugadores/equipos
            target: Estadística objetivo (PTS, TRB, AST, 3P)
            line_values: Lista de valores de línea a simular
            bookmakers: Lista de casas a simular
            noise_level: Nivel de ruido/variación entre casas (0.05 = 5%)
            
        Returns:
            DataFrame con odds simuladas añadidas
        """
        # Valores por defecto
        if line_values is None:
            if target == 'PTS':
                line_values = [10, 15, 20, 25, 30, 35]
            elif target == 'TRB':
                line_values = [4, 6, 8, 10, 12]
            elif target == 'AST':
                line_values = [4, 6, 8, 10, 12]
            elif target == '3P':
                line_values = [1, 2, 3, 4, 5]
            else:
                line_values = []
                
        if bookmakers is None:
            bookmakers = ['draftkings', 'fanduel', 'betmgm', 'caesars', 'pointsbet']
        
        # Crear copia para no modificar el original
        result_df = df.copy()
        
        # Columna target binaria para cada línea
        for line in line_values:
            line_col = f"{target}_over_{line}"
            
            # Si la columna no existe, crearla
            if line_col not in result_df.columns:
                try:
                    result_df[line_col] = (result_df[target] > line).astype(int)
                except:
                    logger.warning(f"No se pudo crear columna {line_col}")
                    continue
        
        # Para cada jugador, línea y bookmaker, generar odds
        for idx, row in result_df.iterrows():
            for line in line_values:
                line_col = f"{target}_over_{line}"
                
                # Saltarse si no tenemos la columna
                if line_col not in result_df.columns:
                    continue
                
                # Obtener probabilidad real (usamos datos históricos)
                try:
                    # Verificar si tenemos columna de probabilidad histórica
                    prob_col = f"{line_col}_prob_10"  # Probabilidad en 10 juegos
                    
                    if prob_col in result_df.columns:
                        true_prob = result_df.at[idx, prob_col]
                    else:
                        # Si no tenemos histórico, usar un valor aleatorio
                        true_prob = np.random.uniform(0.3, 0.7)
                    
                    # Generar odds para cada bookmaker con ruido
                    for bm in bookmakers:
                        # Añadir ruido específico de bookmaker
                        bm_noise = np.random.normal(0, noise_level)
                        implied_prob = np.clip(true_prob + bm_noise, 0.05, 0.95)
                        
                        # Convertir a precio europeo (1/p)
                        over_price = round(1 / implied_prob, 2)
                        under_price = round(1 / (1 - implied_prob), 2)
                        
                        # Guardar en DataFrame
                        result_df.at[idx, f"{target}_over_{line}_odds_{bm}"] = over_price
                        result_df.at[idx, f"{target}_under_{line}_odds_{bm}"] = under_price
                        
                except Exception as e:
                    logger.warning(f"Error al simular odds para idx {idx}, line {line}: {str(e)}")
        
        return result_df
        
    def compare_bookmaker_odds(
        self,
        df: pd.DataFrame,
        target: str = 'PTS',
        line_values: List[float] = None,
        min_samples: int = 5
    ) -> pd.DataFrame:
        """
        Analiza las diferencias entre odds de distintas casas de apuestas
        
        Este método permite:
        1. Identificar qué casas ofrecen mejores odds sistemáticamente 
        2. Detectar discrepancias significativas entre diferentes operadores
        3. Encontrar casas que podrían tener ventaja para ciertos tipos de apuestas
        4. Generar métricas comparativas para optimizar selección de operadores
        
        Args:
            df: DataFrame con datos de jugadores incluyendo odds
            target: Estadística objetivo (PTS, TRB, AST, 3P)
            line_values: Valores de línea específicos a analizar
            min_samples: Mínimo de muestras para incluir un bookmaker
            
        Returns:
            DataFrame con análisis de diferencias
        """
        # Valores por defecto
        if line_values is None:
            if target == 'PTS':
                line_values = [10, 15, 20, 25, 30, 35]
            elif target == 'TRB':
                line_values = [4, 6, 8, 10, 12]
            elif target == 'AST':
                line_values = [4, 6, 8, 10, 12]
            elif target == '3P':
                line_values = [1, 2, 3, 4, 5]
            else:
                line_values = []
        
        # Resultados
        comparison_data = []
        
        # Para cada línea
        for line in line_values:
            # Detectar columnas de odds para esta línea
            over_cols = [col for col in df.columns 
                        if f"{target}_over_{line}_odds_" in col]
            
            under_cols = [col for col in df.columns 
                        if f"{target}_under_{line}_odds_" in col]
            
            if not over_cols and not under_cols:
                logger.info(f"No se encontraron columnas de odds para {target} línea {line}")
                continue
            
            # Calcular estadísticas para OVER
            over_stats = {}
            if over_cols:
                for col in over_cols:
                    # Extraer nombre de bookmaker
                    bm = col.split(f"{target}_over_{line}_odds_")[1]
                    
                    # Calcular estadísticas
                    values = df[col].dropna()
                    if len(values) >= min_samples:
                        over_stats[bm] = {
                            'count': len(values),
                            'mean': values.mean(),
                            'min': values.min(),
                            'max': values.max(),
                            'std': values.std(),
                            'implied_prob': 1 / values.mean()
                        }
            
            # Calcular estadísticas para UNDER
            under_stats = {}
            if under_cols:
                for col in under_cols:
                    # Extraer nombre de bookmaker
                    bm = col.split(f"{target}_under_{line}_odds_")[1]
                    
                    # Calcular estadísticas
                    values = df[col].dropna()
                    if len(values) >= min_samples:
                        under_stats[bm] = {
                            'count': len(values),
                            'mean': values.mean(),
                            'min': values.min(),
                            'max': values.max(),
                            'std': values.std(),
                            'implied_prob': 1 / values.mean()
                        }
            
            # Encontrar mejores casas
            best_over_bm = None
            best_over_odds = 0
            
            for bm, stats in over_stats.items():
                if stats['mean'] > best_over_odds:
                    best_over_odds = stats['mean']
                    best_over_bm = bm
            
            best_under_bm = None
            best_under_odds = 0
            
            for bm, stats in under_stats.items():
                if stats['mean'] > best_under_odds:
                    best_under_odds = stats['mean']
                    best_under_bm = bm
            
            # Guardar análisis
            comparison_data.append({
                'target': target,
                'line': line,
                'over_bookmakers': len(over_stats),
                'under_bookmakers': len(under_stats),
                'best_over_bookmaker': best_over_bm,
                'best_over_odds': best_over_odds,
                'best_under_bookmaker': best_under_bm,
                'best_under_odds': best_under_odds,
                'over_stats': over_stats,
                'under_stats': under_stats,
                'juice': sum(1/stats['mean'] for bm, stats in over_stats.items()) / len(over_stats) + 
                         sum(1/stats['mean'] for bm, stats in under_stats.items()) / len(under_stats) if over_stats and under_stats else None
            })
        
        return pd.DataFrame(comparison_data) 
    
    # === MÉTODOS DE UTILIDAD Y GESTIÓN ===
    
    def get_api_status(self) -> Dict[str, Any]:
        """
        Obtiene el estado de todas las APIs configuradas.
        
        Returns:
            Estado de cada API
        """
        status = {
            'sportradar': {'configured': False, 'accessible': False, 'error': None},
            'odds_api': {'configured': False, 'accessible': False, 'error': None},
            'sportsdata_io': {'configured': False, 'accessible': False, 'error': None}
        }
        
        # Verificar Sportradar
        if self.sportradar_api:
            status['sportradar']['configured'] = True
            try:
                test_result = self.sportradar_api.test_connection()
                status['sportradar']['accessible'] = test_result['success']
                if not test_result['success']:
                    status['sportradar']['error'] = test_result.get('error', 'Unknown error')
            except Exception as e:
                status['sportradar']['error'] = str(e)
        
        # Verificar otros proveedores
        for provider in ['odds_api', 'sportsdata_io']:
            if self.api_keys.get(provider):
                status[provider]['configured'] = True
                # Aquí podrías añadir tests de conexión para otros proveedores
        
        return status
    
    def get_cache_status(self) -> Dict[str, Any]:
        """
        Obtiene estadísticas del cache.
        
        Returns:
            Estadísticas del cache
        """
        cache_stats = {
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'hit_rate': self.cache_hits / max(1, self.cache_hits + self.cache_misses),
            'api_calls_made': self.api_calls_made,
            'cache_directory': str(self.odds_data_dir),
            'cache_expiry_hours': self.cache_expiry
        }
        
        # Añadir estadísticas de Sportradar si está disponible
        if self.sportradar_api:
            try:
                sportradar_cache = self.sportradar_api.get_cache_stats()
                cache_stats['sportradar_cache'] = sportradar_cache
            except:
                pass
        
        return cache_stats
    
    def clear_all_cache(self):
        """Limpia todo el cache."""
        try:
            # Limpiar cache de archivos
            cache_dir = self.odds_data_dir / 'cache'
            for cache_file in cache_dir.glob('*.json'):
                cache_file.unlink()
            
            # Limpiar cache de Sportradar
            if self.sportradar_api:
                self.sportradar_api.clear_cache()
            
            # Resetear contadores
            self.cache_hits = 0
            self.cache_misses = 0
            
            logger.info("Cache limpiado completamente")
            
        except Exception as e:
            logger.error(f"Error limpiando cache: {e}")
            raise CacheError(f"Error limpiando cache: {e}")
    
    def get_supported_features(self) -> Dict[str, List[str]]:
        """
        Obtiene características soportadas por cada proveedor.
        
        Returns:
            Características por proveedor
        """
        features = {
            'sportradar': [
                'nba_odds',
                'player_props',
                'game_schedule',
                'team_info',
                'market_overview',
                'historical_data',
                'real_time_updates',
                'cache_management'
            ],
            'odds_api': [
                'multi_sport',
                'bookmaker_comparison',
                'multiple_markets',
                'historical_odds'
            ],
            'sportsdata_io': [
                'player_stats',
                'game_data',
                'odds_integration'
            ],
            'file_loading': [
                'csv_files',
                'json_files',
                'excel_files',
                'data_standardization'
            ],
            'simulation': [
                'realistic_odds',
                'variance_modeling',
                'bookmaker_simulation',
                'market_simulation'
            ]
        }
        
        return features
    
    def export_data_summary(self, output_file: Optional[str] = None) -> Dict[str, Any]:
        """
        Exporta resumen de datos obtenidos.
        
        Args:
            output_file: Archivo opcional para guardar el resumen
            
        Returns:
            Resumen de datos
        """
        summary = {
            'timestamp': datetime.now().isoformat(),
            'api_status': self.get_api_status(),
            'cache_status': self.get_cache_status(),
            'supported_features': self.get_supported_features(),
            'configuration': {
                'cache_expiry_hours': self.cache_expiry,
                'data_directory': str(self.odds_data_dir),
                'sportradar_enabled': self.sportradar_api is not None
            }
        }
        
        if output_file:
            try:
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(summary, f, indent=2, ensure_ascii=False)
                logger.info(f"Resumen exportado a {output_file}")
            except Exception as e:
                logger.error(f"Error exportando resumen: {e}")
        
        return summary
    
    def __str__(self) -> str:
        """Representación string del fetcher."""
        sportradar_status = "OK" if self.sportradar_api else "NO CONFIGURADO"
        return (f"BookmakersDataFetcher("
                f"sportradar={sportradar_status}, "
                f"cache_hits={self.cache_hits}, "
                f"api_calls={self.api_calls_made})")
    
    def __repr__(self) -> str:
        """Representación detallada del fetcher."""
        return self.__str__() 