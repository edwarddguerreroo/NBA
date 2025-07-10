"""
Sportradar API Client para NBA
=============================

Cliente completo para la API de Sportradar que obtiene:
- Cuotas de casas de apuestas para partidos NBA
- Líneas de player props (PTS, AST, TRB, 3P)
- Información de partidos y equipos
- Datos históricos de cuotas

Documentación API: https://developer.sportradar.com/docs/read/basketball/NBA_v8
"""

import time
import json
import logging
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from urllib.parse import urljoin
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .config import get_config
from .exceptions import (
    SportradarAPIError, 
    RateLimitError, 
    AuthenticationError, 
    NetworkError,
    DataValidationError,
    create_http_error
)

logger = logging.getLogger(__name__)


class SportradarAPI:
    """
    Cliente avanzado para la API de Sportradar NBA.
    
    Proporciona acceso completo a:
    - Schedules y partidos
    - Cuotas de casas de apuestas
    - Player props y líneas
    - Datos de equipos y jugadores
    """
    
    def __init__(self, api_key: Optional[str] = None, config_override: Optional[Dict] = None):
        """
        Inicializa el cliente de Sportradar.
        
        Args:
            api_key: API key de Sportradar. Si no se proporciona, se obtiene de config
            config_override: Configuración personalizada para sobrescribir defaults
        """
        # Configuración
        self.config = get_config()
        if config_override:
            for key, value in config_override.items():
                self.config.set('sportradar', key, value=value)
        
        # API Key
        self.api_key = api_key or self.config.get('sportradar', 'api_key')
        if not self.api_key:
            raise AuthenticationError("API key de Sportradar requerida")
        
        # URLs base según documentación oficial
        self.base_url = self.config.get_api_url('basketball')
        self.odds_base_url = self.config.get_api_url('prematch')
        self.player_props_url = self.config.get_api_url('player_props')
        self.live_odds_url = self.config.get_api_url('live_odds')
        
        # Configuración de red
        self.timeout = self.config.get('sportradar', 'timeout')
        self.retry_attempts = self.config.get('sportradar', 'retry_attempts')
        self.retry_delay = self.config.get('sportradar', 'retry_delay')
        
        # Sport IDs
        self.sport_id = self.config.get_sport_id('basketball')
        
        # Rate limiting
        self.rate_limit_calls = self.config.get('sportradar', 'rate_limit_calls')
        self.rate_limit_period = self.config.get('sportradar', 'rate_limit_period')
        self.last_calls = []
        
        # Configurar sesión HTTP con retry strategy
        self.session = self._setup_session()
        
        # Cache simple en memoria
        self._cache = {}
        self._cache_duration = 300  # 5 minutos
        
        logger.info(f"SportradarAPI inicializada | Base URL: {self.base_url}")
    
    def _setup_session(self) -> requests.Session:
        """Configura sesión HTTP con estrategia de retry."""
        session = requests.Session()
        
        # Configurar retry strategy
        retry_strategy = Retry(
            total=self.retry_attempts,
            backoff_factor=self.retry_delay,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "OPTIONS"]
        )
        
        # Configurar adapter
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        # Headers por defecto
        session.headers.update({
            'User-Agent': 'NBA-Prediction-System/1.0',
            'Accept': 'application/json',
            'Content-Type': 'application/json'
        })
        
        return session
    
    def _check_rate_limit(self):
        """Verifica y aplica rate limiting."""
        now = time.time()
        
        # Limpiar llamadas antiguas
        self.last_calls = [call_time for call_time in self.last_calls 
                          if now - call_time < self.rate_limit_period]
        
        # Verificar límite
        if len(self.last_calls) >= self.rate_limit_calls:
            sleep_time = self.rate_limit_period - (now - self.last_calls[0])
            if sleep_time > 0:
                logger.warning(f"Rate limit alcanzado. Esperando {sleep_time:.1f}s")
                time.sleep(sleep_time)
        
        # Registrar llamada actual
        self.last_calls.append(now)
    
    def _get_cache_key(self, endpoint: str, params: Dict) -> str:
        """Genera clave de cache para endpoint y parámetros."""
        params_str = json.dumps(params, sort_keys=True)
        return f"{endpoint}:{hash(params_str)}"
    
    def _is_cache_valid(self, cache_key: str) -> bool:
        """Verifica si entrada de cache es válida."""
        if cache_key not in self._cache:
            return False
        
        cached_time = self._cache[cache_key]['timestamp']
        return (time.time() - cached_time) < self._cache_duration
    
    def _make_request(self, endpoint: str, params: Optional[Dict] = None,
                     use_cache: bool = True, use_odds_api: bool = False, expect_xml: bool = False) -> Dict[str, Any]:
        """
        Realiza petición HTTP a la API de Sportradar.
        
        Args:
            endpoint: Endpoint de la API
            params: Parámetros de query
            use_cache: Si usar cache
            use_odds_api: Si usar la API de Odds en lugar de Sports
            
        Returns:
            Respuesta JSON de la API
            
        Raises:
            SportradarAPIError: Error en la API
        """
        params = params or {}
        params['api_key'] = self.api_key
        
        # Verificar cache
        cache_key = self._get_cache_key(endpoint, params)
        if use_cache and self._is_cache_valid(cache_key):
            logger.debug(f"Cache hit para {endpoint}")
            return self._cache[cache_key]['data']
        
        # Rate limiting
        self._check_rate_limit()
        
        # Construir URL - usar base URL apropiada
        base_url = self.odds_base_url if use_odds_api else self.base_url
        url = urljoin(base_url, endpoint)
        
        try:
            logger.debug(f"GET {url} | Params: {len(params)} items")
            
            response = self.session.get(
                url, 
                params=params, 
                timeout=self.timeout
            )
            
            # Manejo de errores HTTP
            if response.status_code != 200:
                self._handle_http_error(response, endpoint)
            
            # Parsear respuesta
            try:
                if expect_xml:
                    # Parsear XML y convertir a diccionario
                    data = self._parse_xml_response(response.text)
                else:
                    data = response.json()
            except (json.JSONDecodeError, ET.ParseError) as e:
                raise SportradarAPIError(
                    f"Respuesta inválida: {e}",
                    response.status_code,
                    endpoint=endpoint
                )
            
            # Guardar en cache
            if use_cache:
                self._cache[cache_key] = {
                    'data': data,
                    'timestamp': time.time()
                }
            
            logger.debug(f"Respuesta exitosa de {endpoint}")
            return data
            
        except requests.exceptions.Timeout:
            raise NetworkError(f"Timeout en petición a {endpoint}", url, self.timeout)
        except requests.exceptions.ConnectionError as e:
            raise NetworkError(f"Error de conexión: {e}", url)
        except requests.exceptions.RequestException as e:
            raise SportradarAPIError(f"Error en petición: {e}", endpoint=endpoint)
    
    def _handle_http_error(self, response: requests.Response, endpoint: str):
        """Maneja errores HTTP específicos."""
        status_code = response.status_code
        
        # Intentar obtener mensaje de error de la respuesta
        try:
            error_data = response.json()
            message = error_data.get('message', f"Error HTTP {status_code}")
        except:
            message = f"Error HTTP {status_code}"
        
        # Rate limiting específico
        if status_code == 429:
            retry_after = response.headers.get('Retry-After')
            if retry_after:
                try:
                    retry_after = int(retry_after)
                except ValueError:
                    retry_after = None
            raise RateLimitError(retry_after)
        
        # Crear excepción apropiada
        raise create_http_error(
            status_code, 
            message, 
            response_data=error_data if 'error_data' in locals() else None
        )
    
    # === MÉTODOS PRINCIPALES DE LA API ===
    
    def get_daily_odds_schedule(self, date: str, sport_id: int = None) -> Dict[str, Any]:
        """
        Obtiene el schedule con odds disponibles para una fecha específica usando Player Props v2 API.
        
        Args:
            date: Fecha en formato YYYY-MM-DD
            sport_id: ID del deporte (default: basketball from config)
            
        Returns:
            Schedule con sport events que tienen odds
        """
        if sport_id is None:
            sport_id = self.sport_id
            
        # Usar Player Props v2 API para daily schedules
        endpoint = self.config.get_endpoint('daily_schedules', 
                                           sport_id=f"sr:sport:{sport_id}", date=date)
        
        # Construir URL manualmente para Player Props v2 API
        player_props_url = self.config.get('sportradar', 'player_props_url')
        url = urljoin(player_props_url, endpoint)
        params = {'api_key': self.api_key}
        
        # Rate limiting
        self._check_rate_limit()
        
        try:
            logger.debug(f"GET {url} | Daily Schedule Player Props v2")
            
            response = self.session.get(
                url, 
                params=params, 
                timeout=self.timeout
            )
            
            if response.status_code != 200:
                self._handle_http_error(response, endpoint)
            
            try:
                # Intentar parsear como JSON primero
                data = response.json()
            except json.JSONDecodeError as e:
                raise SportradarAPIError(
                    f"Respuesta JSON inválida: {e}",
                    response.status_code,
                    endpoint=endpoint
                )
            
            logger.debug(f"Respuesta exitosa de daily schedule para {date}")
            return data
            
        except requests.exceptions.Timeout:
            raise NetworkError(f"Timeout en petición a {endpoint}", url, self.timeout)
        except requests.exceptions.ConnectionError as e:
            raise NetworkError(f"Error de conexión: {e}", url)
        except requests.exceptions.RequestException as e:
            raise SportradarAPIError(f"Error en petición: {e}", endpoint=endpoint)
    
    def get_schedule(self, date: Optional[str] = None, season: str = "2024") -> Dict[str, Any]:
        """
        Obtiene el schedule de partidos NBA.
        
        Args:
            date: Fecha específica (YYYY-MM-DD). Si es None, obtiene schedule completo
            season: Temporada (ej: "2024" para 2024-25)
            
        Returns:
            Datos del schedule
        """
        if date:
            # Schedule para fecha específica
            endpoint = f"games/{date}/schedule.json"
        else:
            # Schedule completo de temporada
            endpoint = f"seasons/{season}/schedule.json"
        
        return self._make_request(endpoint)
    
    def get_odds(self, sport_event_id: str, format: str = "american") -> Dict[str, Any]:
        """
        Obtiene cuotas para un partido específico usando Odds Comparison API.
        
        Args:
            sport_event_id: ID del sport event de Sportradar (ej: sr:sport_event:12345)
            format: Formato de odds ("us", "eu", "uk")
            
        Returns:
            Datos de cuotas del partido
        """
        # Usar endpoint configurado para Sport Event Markets
        endpoint = self.config.get_endpoint('sport_event_markets', 
                                           sport_event_id=sport_event_id)
        
        return self._make_request(endpoint, use_odds_api=True, expect_xml=False)
    
    def get_player_props(self, sport_event_id: str, format: str = "us") -> Dict[str, Any]:
        """
        Obtiene player props para un partido usando Odds Comparison Player Props API.
        
        Args:
            sport_event_id: ID del sport event de Sportradar
            format: Formato de odds ("us", "eu", "uk")
            
        Returns:
            Datos de player props
        """
        # Usar endpoint configurado para Player Props
        endpoint = self.config.get_endpoint('sport_event_player_props', 
                                           sport_event_id=sport_event_id)
        
        # Construir URL manualmente para player props ya que usa API diferente
        url = urljoin(self.player_props_url, endpoint)
        params = {'api_key': self.api_key}
        
        # Rate limiting
        self._check_rate_limit()
        
        try:
            logger.debug(f"GET {url} | Player Props")
            
            response = self.session.get(
                url, 
                params=params, 
                timeout=self.timeout
            )
            
            if response.status_code != 200:
                self._handle_http_error(response, endpoint)
            
            try:
                # Intentar parsear como JSON primero
                data = response.json()
            except json.JSONDecodeError as e:
                raise SportradarAPIError(
                    f"Respuesta JSON inválida: {e}",
                    response.status_code,
                    endpoint=endpoint
                )
            
            logger.debug(f"Respuesta exitosa de player props para {sport_event_id}")
            return data
            
        except requests.exceptions.Timeout:
            raise NetworkError(f"Timeout en petición a {endpoint}", url, self.timeout)
        except requests.exceptions.ConnectionError as e:
            raise NetworkError(f"Error de conexión: {e}", url)
        except requests.exceptions.RequestException as e:
            raise SportradarAPIError(f"Error en petición: {e}", endpoint=endpoint)
    
    def get_teams(self) -> Dict[str, Any]:
        """
        Obtiene información de todos los equipos NBA.
        
        Returns:
            Datos de equipos
        """
        endpoint = "league/hierarchy.json"
        return self._make_request(endpoint)
    
    def get_game_info(self, sport_event_id: str) -> Dict[str, Any]:
        """
        Obtiene información detallada de un partido usando Sports API.
        
        Args:
            sport_event_id: ID del sport event
            
        Returns:
            Información del partido
        """
        # Usar la API de Sports para información del partido
        endpoint = f"sport_events/{sport_event_id}/summary.json"
        return self._make_request(endpoint, use_odds_api=False)
    
    def get_bookmakers(self) -> Dict[str, Any]:
        """
        Obtiene lista de casas de apuestas disponibles.
        
        Returns:
            Lista de bookmakers
        """
        endpoint = "en/books"
        return self._make_request(endpoint, use_odds_api=True, expect_xml=False)
    
    def get_sports(self) -> Dict[str, Any]:
        """
        Obtiene lista de deportes disponibles.
        
        Returns:
            Lista de deportes
        """
        endpoint = "en/sports"
        return self._make_request(endpoint, use_odds_api=True, expect_xml=False)
    
    def get_sport_competitions(self, sport_id: int) -> Dict[str, Any]:
        """
        Obtiene lista de competiciones para un deporte específico.
        
        Args:
            sport_id: ID del deporte
            
        Returns:
            Lista de competiciones
        """
        endpoint = f"en/sports/{sport_id}/competitions"
        return self._make_request(endpoint, use_odds_api=True, expect_xml=False)
    
    def get_competition_schedules(self, competition_id: str, offset: int = 0, 
                                 limit: int = 50, start: int = 0) -> Dict[str, Any]:
        """
        Obtiene schedules para una competición específica.
        
        Args:
            competition_id: ID de la competición (ej: sr:competition:17)
            offset: Offset para paginación
            limit: Límite de resultados
            start: Inicio para paginación
            
        Returns:
            Schedules de la competición
        """
        endpoint = f"en/competitions/{competition_id}/schedules"
        params = {
            'offset': offset,
            'limit': limit,
            'start': start
        }
        return self._make_request(endpoint, params=params, use_odds_api=True, expect_xml=False)
    
    # === MÉTODOS DE ANÁLISIS AVANZADO ===
    
    def get_games_by_date_range(self, start_date: str, end_date: str) -> List[Dict[str, Any]]:
        """
        Obtiene partidos en un rango de fechas.
        
        Args:
            start_date: Fecha inicio (YYYY-MM-DD)
            end_date: Fecha fin (YYYY-MM-DD)
            
        Returns:
            Lista de partidos
        """
        games = []
        
        try:
            start = datetime.strptime(start_date, "%Y-%m-%d")
            end = datetime.strptime(end_date, "%Y-%m-%d")
        except ValueError as e:
            raise DataValidationError(f"Formato de fecha inválido: {e}")
        
        current = start
        while current <= end:
            date_str = current.strftime("%Y-%m-%d")
            try:
                schedule = self.get_schedule(date_str)
                if 'games' in schedule:
                    games.extend(schedule['games'])
            except SportradarAPIError as e:
                logger.warning(f"Error obteniendo schedule para {date_str}: {e}")
            
            current += timedelta(days=1)
        
        return games
    
    def get_odds_for_multiple_games(self, sport_event_ids: List[str], 
                                   include_props: bool = True, 
                                   format: str = "us") -> Dict[str, Dict[str, Any]]:
        """
        Obtiene cuotas para múltiples partidos.
        
        Args:
            sport_event_ids: Lista de IDs de sport events
            include_props: Si incluir player props
            format: Formato de odds ("us", "eu", "uk")
            
        Returns:
            Diccionario con cuotas por sport_event_id
        """
        results = {}
        
        for sport_event_id in sport_event_ids:
            try:
                # Obtener cuotas principales
                odds = self.get_odds(sport_event_id, format)
                results[sport_event_id] = {'odds': odds}
                
                # Obtener player props si se solicita
                if include_props:
                    try:
                        props = self.get_player_props(sport_event_id, format)
                        results[sport_event_id]['props'] = props
                    except SportradarAPIError as e:
                        logger.warning(f"Error obteniendo props para {sport_event_id}: {e}")
                        results[sport_event_id]['props'] = None
                
            except SportradarAPIError as e:
                logger.error(f"Error obteniendo datos para {sport_event_id}: {e}")
                results[sport_event_id] = {'error': str(e)}
        
        return results
    
    def search_games_by_teams(self, home_team: str, away_team: str,
                             date_range_days: int = 7) -> List[Dict[str, Any]]:
        """
        Busca partidos entre equipos específicos.
        
        Args:
            home_team: Nombre o código del equipo local
            away_team: Nombre o código del equipo visitante
            date_range_days: Días hacia adelante para buscar
            
        Returns:
            Lista de partidos encontrados
        """
        end_date = datetime.now() + timedelta(days=date_range_days)
        start_date = datetime.now().strftime("%Y-%m-%d")
        end_date_str = end_date.strftime("%Y-%m-%d")
        
        games = self.get_games_by_date_range(start_date, end_date_str)
        
        # Filtrar por equipos
        matching_games = []
        for game in games:
            home = game.get('home', {}).get('name', '').lower()
            away = game.get('away', {}).get('name', '').lower()
            
            if (home_team.lower() in home and away_team.lower() in away):
                matching_games.append(game)
        
        return matching_games
    
    def get_market_overview(self, date: Optional[str] = None) -> Dict[str, Any]:
        """
        Obtiene resumen del mercado para una fecha.
        
        Args:
            date: Fecha (YYYY-MM-DD). Si es None, usa fecha actual
            
        Returns:
            Resumen del mercado
        """
        if not date:
            date = datetime.now().strftime("%Y-%m-%d")
        
        # Obtener schedule del día
        schedule = self.get_schedule(date)
        
        if 'games' not in schedule:
            return {'date': date, 'games': [], 'total_games': 0}
        
        # Obtener cuotas para todos los partidos
        games_with_odds = []
        for game in schedule['games']:
            game_id = game.get('id')
            if game_id:
                try:
                    odds = self.get_odds(game_id)
                    game_data = {
                        'game': game,
                        'odds': odds,
                        'id': game_id
                    }
                    games_with_odds.append(game_data)
                except SportradarAPIError as e:
                    logger.warning(f"No se pudieron obtener odds para {game_id}: {e}")
        
        return {
            'date': date,
            'games': games_with_odds,
            'total_games': len(schedule['games']),
            'games_with_odds': len(games_with_odds)
        }
    
    def get_nba_odds_today(self, format: str = "us", include_props: bool = True) -> Dict[str, Any]:
        """
        Obtiene odds para todos los partidos NBA de hoy.
        
        Args:
            format: Formato de odds ("us", "eu", "uk")
            include_props: Si incluir player props
            
        Returns:
            Diccionario con odds de partidos NBA del día
        """
        today = datetime.now().strftime("%Y-%m-%d")
        
        try:
            # Obtener schedule del día para NBA (sport_id = 2)
            schedule = self.get_daily_odds_schedule(today, sport_id=2)
            
            if 'sport_events' not in schedule:
                return {
                    'date': today,
                    'sport_events': [],
                    'odds_data': {},
                    'message': 'No hay partidos NBA programados para hoy'
                }
            
            # Extraer sport_event_ids
            sport_event_ids = [event['id'] for event in schedule['sport_events']]
            
            # Obtener odds para todos los partidos
            odds_data = self.get_odds_for_multiple_games(
                sport_event_ids, 
                include_props=include_props, 
                format=format
            )
            
            return {
                'date': today,
                'sport_events': schedule['sport_events'],
                'odds_data': odds_data,
                'total_games': len(sport_event_ids),
                'message': f'Odds obtenidas para {len(sport_event_ids)} partidos NBA'
            }
            
        except SportradarAPIError as e:
            logger.error(f"Error obteniendo odds NBA para {today}: {e}")
            return {
                'date': today,
                'error': str(e),
                'sport_events': [],
                'odds_data': {}
            }
    
    # === MÉTODOS ESPECÍFICOS PARA TARGETS DEL MODELO ===
    
    def get_specific_nba_odds(self, sport_event_id: str, targets: List[str] = None, 
                             format: str = "us") -> Dict[str, Any]:
        """
        Obtiene odds específicas para los targets del modelo de predicción NBA.
        
        Args:
            sport_event_id: ID del sport event
            targets: Lista de targets específicos a obtener:
                   - 'moneyline': Victoria/derrota (is_win)
                   - 'totals': Total de puntos del partido (total_points)
                   - 'spreads': Spread de puntos
                   - 'team_totals': Puntos por equipo (teams_points)
                   - 'player_points': Puntos de jugadores (PTS)
                   - 'player_assists': Asistencias de jugadores (AST)
                   - 'player_rebounds': Rebotes de jugadores (TRB)
                   - 'player_threes': Triples de jugadores (3P)
                   - 'player_combos': Combinaciones (double_double, etc.)
            format: Formato de odds ("us", "eu", "uk")
            
        Returns:
            Diccionario estructurado con odds por target
        """
        if targets is None:
            targets = ['moneyline', 'totals', 'spreads', 'team_totals', 
                      'player_points', 'player_assists', 'player_rebounds', 
                      'player_threes', 'player_combos']
        
        result = {
            'sport_event_id': sport_event_id,
            'targets': {},
            'metadata': {
                'format': format,
                'timestamp': datetime.now().isoformat(),
                'requested_targets': targets
            }
        }
        
        try:
            # Obtener odds principales del juego
            if any(target in ['moneyline', 'totals', 'spreads', 'team_totals'] for target in targets):
                game_odds = self.get_odds(sport_event_id, format)
                result['targets']['game_markets'] = self._extract_game_markets(game_odds, targets)
            
            # Obtener player props si se solicitan
            if any(target.startswith('player_') for target in targets):
                player_props = self.get_player_props(sport_event_id, format)
                result['targets']['player_props'] = self._extract_player_markets(player_props, targets)
                
        except SportradarAPIError as e:
            logger.error(f"Error obteniendo odds específicas para {sport_event_id}: {e}")
            result['error'] = str(e)
        
        return result
    
    def _extract_game_markets(self, odds_data: Dict[str, Any], targets: List[str]) -> Dict[str, Any]:
        """
        Extrae mercados de juego específicos según los targets solicitados.
        
        Args:
            odds_data: Datos completos de odds del juego
            targets: Targets específicos solicitados
            
        Returns:
            Diccionario con mercados extraídos
        """
        markets = {}
        
        if 'markets' not in odds_data:
            return markets
        
        for market in odds_data['markets']:
            market_name = market.get('name', '').lower()
            
            # Moneyline (is_win target)
            if 'moneyline' in targets and any(keyword in market_name for keyword in ['moneyline', 'match winner', 'winner']):
                markets['moneyline'] = {
                    'market_name': market.get('name'),
                    'outcomes': market.get('outcomes', []),
                    'target_mapping': 'is_win'
                }
            
            # Totals (total_points target)
            elif 'totals' in targets and any(keyword in market_name for keyword in ['total points', 'over/under', 'totals']):
                markets['totals'] = {
                    'market_name': market.get('name'),
                    'outcomes': market.get('outcomes', []),
                    'target_mapping': 'total_points'
                }
            
            # Spreads
            elif 'spreads' in targets and any(keyword in market_name for keyword in ['spread', 'handicap', 'point spread']):
                markets['spreads'] = {
                    'market_name': market.get('name'),
                    'outcomes': market.get('outcomes', []),
                    'target_mapping': 'point_spread'
                }
            
            # Team Totals (teams_points target)
            elif 'team_totals' in targets and any(keyword in market_name for keyword in ['team total', 'team points']):
                if 'team_totals' not in markets:
                    markets['team_totals'] = []
                markets['team_totals'].append({
                    'market_name': market.get('name'),
                    'outcomes': market.get('outcomes', []),
                    'target_mapping': 'teams_points'
                })
        
        return markets
    
    def _extract_player_markets(self, props_data: Dict[str, Any], targets: List[str]) -> Dict[str, Any]:
        """
        Extrae mercados de jugadores específicos según los targets solicitados.
        
        Args:
            props_data: Datos completos de player props
            targets: Targets específicos solicitados
            
        Returns:
            Diccionario con mercados de jugadores extraídos
        """
        player_markets = {}
        
        if 'markets' not in props_data:
            return player_markets
        
        for market in props_data['markets']:
            market_name = market.get('name', '').lower()
            
            # Player Points (PTS target)
            if 'player_points' in targets and any(keyword in market_name for keyword in ['player points', 'points over/under', 'total points']):
                if 'player_points' not in player_markets:
                    player_markets['player_points'] = []
                player_markets['player_points'].append({
                    'market_name': market.get('name'),
                    'outcomes': market.get('outcomes', []),
                    'target_mapping': 'PTS'
                })
            
            # Player Assists (AST target)
            elif 'player_assists' in targets and any(keyword in market_name for keyword in ['assists', 'player assists']):
                if 'player_assists' not in player_markets:
                    player_markets['player_assists'] = []
                player_markets['player_assists'].append({
                    'market_name': market.get('name'),
                    'outcomes': market.get('outcomes', []),
                    'target_mapping': 'AST'
                })
            
            # Player Rebounds (TRB target)
            elif 'player_rebounds' in targets and any(keyword in market_name for keyword in ['rebounds', 'total rebounds', 'player rebounds']):
                if 'player_rebounds' not in player_markets:
                    player_markets['player_rebounds'] = []
                player_markets['player_rebounds'].append({
                    'market_name': market.get('name'),
                    'outcomes': market.get('outcomes', []),
                    'target_mapping': 'TRB'
                })
            
            # Player Threes (3P target)
            elif 'player_threes' in targets and any(keyword in market_name for keyword in ['3-pointers', 'three pointers', 'threes made']):
                if 'player_threes' not in player_markets:
                    player_markets['player_threes'] = []
                player_markets['player_threes'].append({
                    'market_name': market.get('name'),
                    'outcomes': market.get('outcomes', []),
                    'target_mapping': '3P'
                })
            
            # Player Combos (double_double, etc.)
            elif 'player_combos' in targets and any(keyword in market_name for keyword in ['double-double', 'triple-double', 'double double']):
                if 'player_combos' not in player_markets:
                    player_markets['player_combos'] = []
                player_markets['player_combos'].append({
                    'market_name': market.get('name'),
                    'outcomes': market.get('outcomes', []),
                    'target_mapping': 'double_double'
                })
        
        return player_markets
    
    def _parse_xml_response(self, xml_text: str) -> Dict[str, Any]:
        """
        Parsea respuesta XML de Sportradar y convierte a diccionario.
        
        Args:
            xml_text: Texto XML de la respuesta
            
        Returns:
            Diccionario con datos parseados
        """
        try:
            root = ET.fromstring(xml_text)
            
            # Extraer namespace si existe
            namespace = ''
            if root.tag.startswith('{'):
                namespace = root.tag.split('}')[0] + '}'
            
            # Procesar según el tipo de respuesta
            if root.tag.endswith('books'):
                # Lista de bookmakers
                books = []
                for book in root.findall(f'{namespace}book'):
                    books.append({
                        'id': book.get('id'),
                        'name': book.get('name')
                    })
                return {'books': books}
            
            elif root.tag.endswith('markets'):
                # Mercados de odds
                markets = []
                for market in root.findall(f'{namespace}market'):
                    market_data = {
                        'id': market.get('id'),
                        'name': market.get('name'),
                        'outcomes': []
                    }
                    
                    for outcome in market.findall(f'{namespace}outcome'):
                        outcome_data = {
                            'id': outcome.get('id'),
                            'name': outcome.get('name'),
                            'odds': outcome.get('odds')
                        }
                        market_data['outcomes'].append(outcome_data)
                    
                    markets.append(market_data)
                
                return {'markets': markets}
            
            else:
                # Respuesta genérica - convertir a dict básico
                result = {}
                for child in root:
                    tag_name = child.tag.replace(namespace, '')
                    if child.text:
                        result[tag_name] = child.text
                    else:
                        result[tag_name] = {attr: child.get(attr) for attr in child.attrib}
                
                return result
                
        except ET.ParseError as e:
            logger.error(f"Error parseando XML: {e}")
            raise
    
    def get_nba_odds_for_targets(self, date: str = None, targets: List[str] = None, 
                                format: str = "us") -> Dict[str, Any]:
        """
        Obtiene odds específicas para targets del modelo NBA en una fecha determinada.
        
        Args:
            date: Fecha en formato YYYY-MM-DD (None para hoy)
            targets: Lista de targets específicos (None para todos)
            format: Formato de odds ("us", "eu", "uk")
            
        Returns:
            Diccionario con odds estructuradas por target para todos los partidos del día
        """
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")
        
        if targets is None:
            targets = ['moneyline', 'totals', 'team_totals', 'player_points', 
                      'player_assists', 'player_rebounds', 'player_threes']
        
        logger.info(f"Obteniendo odds NBA para targets específicos: {date}")
        
        result = {
            'date': date,
            'targets_requested': targets,
            'games': {},
            'summary': {
                'total_games': 0,
                'games_with_odds': 0,
                'targets_found': {}
            },
            'metadata': {
                'format': format,
                'timestamp': datetime.now().isoformat()
            }
        }
        
        try:
            # Obtener schedule del día
            schedule = self.get_daily_odds_schedule(date, sport_id=2)
            
            if 'sport_events' not in schedule:
                result['message'] = 'No hay partidos NBA programados para esta fecha'
                return result
            
            sport_events = schedule['sport_events']
            result['summary']['total_games'] = len(sport_events)
            
            # Procesar cada partido
            for event in sport_events:
                sport_event_id = event['id']
                
                try:
                    # Obtener odds específicas para este partido
                    game_odds = self.get_specific_nba_odds(sport_event_id, targets, format)
                    
                    # Estructurar datos del partido
                    game_info = {
                        'sport_event_id': sport_event_id,
                        'teams': {
                            'home': event.get('competitors', [{}])[0].get('name', 'Unknown'),
                            'away': event.get('competitors', [{}])[1].get('name', 'Unknown') if len(event.get('competitors', [])) > 1 else 'Unknown'
                        },
                        'scheduled': event.get('scheduled'),
                        'odds': game_odds.get('targets', {}),
                        'has_odds': not game_odds.get('error')
                    }
                    
                    result['games'][sport_event_id] = game_info
                    
                    if game_info['has_odds']:
                        result['summary']['games_with_odds'] += 1
                        
                        # Contar targets encontrados
                        for target_category in game_odds.get('targets', {}):
                            if target_category not in result['summary']['targets_found']:
                                result['summary']['targets_found'][target_category] = 0
                            result['summary']['targets_found'][target_category] += 1
                
                except SportradarAPIError as e:
                    logger.warning(f"Error obteniendo odds para {sport_event_id}: {e}")
                    result['games'][sport_event_id] = {
                        'sport_event_id': sport_event_id,
                        'error': str(e),
                        'has_odds': False
                    }
            
            result['message'] = f'Procesados {len(sport_events)} partidos, {result["summary"]["games_with_odds"]} con odds'
            
        except SportradarAPIError as e:
            logger.error(f"Error obteniendo schedule para {date}: {e}")
            result['error'] = str(e)
            result['message'] = f'Error obteniendo datos: {e}'
        
        return result
    
    def extract_odds_for_predictions(self, odds_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extrae y estructura odds específicamente para el sistema de predicciones.
        
        Args:
            odds_data: Datos de odds obtenidos de get_nba_odds_for_targets
            
        Returns:
            Diccionario estructurado para facilitar la integración con el modelo
        """
        predictions_data = {
            'date': odds_data.get('date'),
            'games_count': odds_data.get('summary', {}).get('total_games', 0),
            'games': {}
        }
        
        for game_id, game_data in odds_data.get('games', {}).items():
            if not game_data.get('has_odds'):
                continue
                
            game_predictions = {
                'teams': game_data.get('teams', {}),
                'scheduled': game_data.get('scheduled'),
                'targets': {}
            }
            
            odds = game_data.get('odds', {})
            
            # Extraer odds para targets específicos del modelo
            
            # 1. Game-level targets
            if 'game_markets' in odds:
                game_markets = odds['game_markets']
                
                # is_win (moneyline)
                if 'moneyline' in game_markets:
                    game_predictions['targets']['is_win'] = {
                        'market': game_markets['moneyline']['market_name'],
                        'odds': game_markets['moneyline']['outcomes']
                    }
                
                # total_points (totals)
                if 'totals' in game_markets:
                    game_predictions['targets']['total_points'] = {
                        'market': game_markets['totals']['market_name'],
                        'odds': game_markets['totals']['outcomes']
                    }
                
                # teams_points (team totals)
                if 'team_totals' in game_markets:
                    game_predictions['targets']['teams_points'] = game_markets['team_totals']
            
            # 2. Player-level targets
            if 'player_props' in odds:
                player_props = odds['player_props']
                
                # Mapear cada tipo de prop a su target correspondiente
                target_mapping = {
                    'player_points': 'PTS',
                    'player_assists': 'AST', 
                    'player_rebounds': 'TRB',
                    'player_threes': '3P',
                    'player_combos': 'double_double'
                }
                
                for prop_type, target_name in target_mapping.items():
                    if prop_type in player_props:
                        game_predictions['targets'][target_name] = player_props[prop_type]
            
            predictions_data['games'][game_id] = game_predictions
        
        return predictions_data
    
    # === UTILIDADES ===
    
    def clear_cache(self):
        """Limpia el cache en memoria."""
        self._cache.clear()
        logger.info("Cache limpiado")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas del cache."""
        return {
            'entries': len(self._cache),
            'cache_duration': self._cache_duration,
            'oldest_entry': min([v['timestamp'] for v in self._cache.values()]) if self._cache else None,
            'newest_entry': max([v['timestamp'] for v in self._cache.values()]) if self._cache else None
        }
    
    def test_connection(self) -> Dict[str, Any]:
        """
        Prueba la conexión con la API.
        
        Returns:
            Resultado del test
        """
        try:
            start_time = time.time()
            teams = self.get_teams()
            response_time = time.time() - start_time
            
            return {
                'success': True,
                'response_time': response_time,
                'api_accessible': True,
                'teams_count': len(teams.get('conferences', [])),
                'message': "Conexión exitosa con Sportradar API"
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'api_accessible': False,
                'message': f"Error de conexión: {e}"
            }
    
    def __str__(self) -> str:
        """Representación string del cliente."""
        return f"SportradarAPI(base_url={self.base_url}, cache_entries={len(self._cache)})" 