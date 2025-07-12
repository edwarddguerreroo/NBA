"""
Configuración para la integración con Sportradar API.
Actualizada según documentación oficial de Sportradar.
"""

import os
from configparser import ConfigParser
from typing import Dict, Any, Optional, List
import logging
import json

# Intentar cargar python-dotenv para variables de entorno
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # Si no está disponible python-dotenv, continuar sin él
    pass

logger = logging.getLogger(__name__)

# Configuración por defecto basada en documentación oficial de Sportradar
DEFAULT_CONFIG = {
    'sportradar': {
        # URLs base corregidas según documentación oficial
        'base_url': 'https://api.sportradar.com/basketball/trial/v8/en',
        'odds_base_url': 'https://api.sportradar.com/oddscomparison-prematch/trial/v2/',
        'player_props_url': 'https://api.sportradar.com/oddscomparison-player-props/trial/v2/',
        'live_odds_url': 'https://api.sportradar.com/oddscomparison-live-odds/trial/v2/',
        'regular_odds_url': 'https://api.sportradar.com/oddscomparison-trial/v1/',
        
        # API Key
        'api_key': '',
        
        # Sport IDs según documentación
        'basketball_sport_id': '2',  # Basketball incluye NBA
        'nba_sport_id': '2',
        
        # Configuración de red
        'timeout': 30,
        'retry_attempts': 3,
        'retry_delay': 1,
        
        # Rate limiting (trial account según doc)
        'rate_limit_calls': 5,
        'rate_limit_period': 1,
        
        # Configuración de cache optimizado
        'cache_duration': 300,  # 5 minutos por defecto
        'cache_enabled': True,
        'cache_type': 'memory_disk',  # memory, disk, memory_disk
        'cache_max_size_mb': 100,     # Máximo 100MB cache en memoria
        'cache_persistence': True,    # Persistir cache en disco
        'cache_compression': True,    # Comprimir datos de cache
        
        # Formatos soportados según documentación
        'supported_formats': ['json', 'xml'],
        'default_format': 'json',  # JSON por defecto, XML disponible
        'default_language': 'en',
        
        # Endpoints según documentación oficial - Odds Comparison v2 - Prematch
        'endpoints': {
            # Libros/casas de apuestas
            'books': 'en/books',
            # Deportes
            'sports': 'en/sports',
            # Competiciones por deporte
            'sport_competitions': 'en/sports/{sport_id}/competitions',
            # Categorías por deporte
            'sport_categories': 'en/sports/{sport_id}/categories',
            
            # Sport Event Markets - Odds para eventos específicos
            'sport_event_markets': 'en/sport_events/{sport_event_id}/sport_event_markets',
            # Competition Markets - Odds por competición
            'competition_markets': 'en/competitions/{competition_id}/sport_event_markets',
            # Schedule Markets - Odds por fecha
            'schedule_markets': 'en/sports/{sport_id}/schedules/{date}/sport_event_markets',
            
            # Schedules - Calendarios
            'competition_schedules': 'en/competitions/{competition_id}/schedules',
            'sport_schedules': 'en/sports/{sport_id}/schedules/{date}/schedules',
            
            # Mappings
            'competition_mappings': 'en/competitions/mappings',
            'competitor_mappings': 'en/competitors/mappings',
            'player_mappings': 'en/players/mappings',
            'sport_event_mappings': 'en/sport_events/mappings'
        },
        
        # Market IDs para NBA según documentación oficial
        'market_ids': {
            # Mercados de equipo (Prematch API)
            '1x2': 1,  # Winner/Moneyline → is_win
            'total_incl_overtime': 225,  # Total points (incl. overtime) → total_points
            'home_total_incl_overtime': 227,  # Home total → teams_points
            'away_total_incl_overtime': 228,  # Away total → teams_points
            
            # Player Props Markets (Player Props API) - IDs oficiales
            'total_points': 'sr:market:921',  # Puntos totales (incluye horas extras) → PTS
            'total_assists': 'sr:market:922',  # Asistencias totales (incluidas horas extras) → AST
            'total_rebounds': 'sr:market:923',  # Rebotes totales (incluye tiempo extra) → TRB
            'total_3pt_field_goals': 'sr:market:924',  # Total de goles de campo de 3 puntos → 3P
            'total_steals': 'sr:market:8000',  # Robos totales (incluidas horas extras)
            'total_blocks': 'sr:market:8001',  # Total de bloques (incl. horas extras)
            'total_turnovers': 'sr:market:8002',  # Rotación total (incluidas horas extras)
            'points_plus_rebounds': 'sr:market:8003',  # Total de puntos más rebotes
            'points_plus_assists': 'sr:market:8004',  # Total de puntos más asistencias
            'rebounds_plus_assists': 'sr:market:8005',  # Rebotes totales más asistencias
            'points_assists_rebounds': 'sr:market:8006',  # Total de puntos más asistencias más rebotes
            'blocks_plus_steals': 'sr:market:8007',  # Total de bloqueos más robos
            'double_double': 'sr:market:8008',  # Doble doble (incluye horas extras)
            'triple_double': 'sr:market:8009'  # Triple doble (incluye tiempo extra)
        },
        
        # Mapeo de targets del sistema a mercados de Sportradar
        'target_to_market': {
            # Targets de jugadores (Player Props API)
            'PTS': 'total_points',
            'AST': 'total_assists', 
            'TRB': 'total_rebounds',
            '3P': 'total_3pt_field_goals',
            'double_double': 'double_double',
            
            # Targets de equipos (Prematch API)
            'is_win': '1x2',
            'total_points': 'total_incl_overtime',
            'teams_points': ['home_total_incl_overtime', 'away_total_incl_overtime']
        }
    },
    
    # Player Props API v2 - Configuración especializada para player props
    'player_props_v2': {
        'base_url': 'https://api.sportradar.com/oddscomparison-player-props/trial/v2/',
        'api_key': '',  # Se carga desde entorno
        'timeout': 30,
        'retry_attempts': 3,
        'retry_delay': 1,
        
        # Rate limiting específico para Player Props API
        'rate_limit_calls': 5,
        'rate_limit_period': 1,
        
        # Cache optimizado para player props
        'cache_duration': 300,  # 5 minutos
        'cache_enabled': True,
        
        # Endpoints específicos para Player Props API
        'endpoints': {
            # Básicos
            'sports': 'en/sports',
            'books': 'en/books',
            'sport_competitions': 'en/sports/{sport_id}/competitions',
            'sport_categories': 'en/sports/{sport_id}/categories',
            'sport_stages': 'en/sports/{sport_id}/stages',
            
            # Schedules
            'competition_schedules': 'en/competitions/{competition_id}/schedules',
            'sport_schedules': 'en/sports/{sport_id}/schedules/{date}/schedules',
            
            # Player Props específicos
            'competition_player_props': 'en/competitions/{competition_id}/players_props',
            'event_player_props': 'en/sport_events/{event_id}/players_props',
            'schedule_player_props': 'en/sports/{sport_id}/schedules/{date}/players_props',
            'players_props_changelog': 'en/players_props_changelog'
        },
        
        # Competiciones NBA para Player Props
        'nba_competition_ids': {
            'nba': 'sr:competition:132',  # NBA (temporada baja - player_props: false)
            'wnba': 'sr:competition:486',  # WNBA (activa - player_props: true)
            'ncaa': 'sr:competition:648'   # NCAA (player_props: false)
        },
        
        # Mapeo de targets a mercados específicos de Player Props
        'target_to_market': {
            'PTS': ['total_points', 'player_points', 'points'],
            'AST': ['total_assists', 'player_assists', 'assists'],
            'TRB': ['total_rebounds', 'player_rebounds', 'rebounds'],
            '3P': ['total_3pt_field_goals', 'player_threes', 'three_pointers', 'threes'],
            'double_double': ['double_double', 'dd']
        }
    },
    'betting': {
        'minimum_edge': 0.04,          # 4% mínimo de ventaja
        'confidence_threshold': 0.96,  # 96% confianza mínima
        'max_kelly_fraction': 0.25,    # Máximo 25% del bankroll
        'min_odds': 1.5,               # Odds mínimas aceptables
        'max_odds': 10.0,              # Odds máximas aceptables
        'default_bankroll': 1000.0,    # Bankroll por defecto
        'min_bet_amount': 10.0,        # Apuesta mínima
        'max_bet_amount': 500.0        # Apuesta máxima
    },
    'data': {
        'cache_enabled': True,
        'cache_duration_hours': 1,
        'cache_directory': 'data/cache/bookmakers',
        'cache_cleanup_enabled': True,
        'cache_cleanup_interval_hours': 24,
        'cache_max_entries': 10000,
        'cache_stats_enabled': True,
        'simulate_when_no_data': False,  # DESHABILITADO - Solo datos reales
        'simulation_variance': 0.15,    # 15% varianza en simulación
        'min_historical_games': 5       # Mínimo de juegos para análisis
    },
    'analysis': {
        'lookback_days': 60,
        'min_samples_per_line': 30,
        'correlation_threshold': 0.3,
        'outlier_detection': True,
        'adaptive_thresholds': True
    },
    'logging': {
        'level': 'INFO',
        'file_logging': True,
        'log_directory': 'logs/bookmakers'
    }
}


class BookmakersConfig:
    """
    Gestión centralizada de configuración para el módulo bookmakers.
    
    Maneja configuración de:
    - APIs (Sportradar, etc.)
    - Parámetros de betting
    - Configuración de datos y cache
    - Variables de entorno
    """
    
    def __init__(self, config_file: Optional[str] = None):
        """
        Inicializa la configuración.
        
        Args:
            config_file: Ruta opcional a archivo de configuración JSON
        """
        self.config = self._load_default_config()
        
        # Cargar configuración desde archivo si se proporciona
        if config_file and os.path.exists(config_file):
            self._load_from_file(config_file)
        
        # Cargar variables de entorno
        self._load_from_environment()
        
        # Validar configuración
        self._validate_config()
    
    def _load_default_config(self) -> Dict[str, Any]:
        """Carga configuración por defecto basada en documentación oficial."""
        return DEFAULT_CONFIG.copy()
    
    def _load_from_file(self, config_file: str):
        """Carga configuración desde archivo JSON."""
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                file_config = json.load(f)
            
            # Merge recursivo de configuraciones
            self._merge_config(self.config, file_config)
            logger.info(f"Configuración cargada desde {config_file}")
            
        except Exception as e:
            logger.warning(f"Error cargando configuración desde {config_file}: {e}")
    
    def _load_from_environment(self):
        """Carga configuración desde variables de entorno."""
        env_mappings = {
            'SPORTRADAR_API': ['sportradar', 'api_key'], 
            'SPORTRADAR_BASE_URL': ['sportradar', 'base_url'],
            'SPORTRADAR_PLAYER_PROPS_API': ['player_props_v2', 'api_key'],  # Nueva API key para Player Props
            'BETTING_MIN_EDGE': ['betting', 'minimum_edge'],
            'BETTING_CONFIDENCE_THRESHOLD': ['betting', 'confidence_threshold'],
            'BETTING_MAX_KELLY': ['betting', 'max_kelly_fraction'],
            'CACHE_ENABLED': ['data', 'cache_enabled'],
            'CACHE_DURATION_HOURS': ['data', 'cache_duration_hours'],
            'LOG_LEVEL': ['logging', 'level']
        }
        
        for env_var, config_path in env_mappings.items():
            value = os.getenv(env_var)
            if value is not None:
                # Convertir tipos apropiados
                if config_path[1] in ['minimum_edge', 'confidence_threshold', 'max_kelly_fraction']:
                    value = float(value)
                elif config_path[1] in ['cache_duration_hours']:
                    value = int(value)
                elif config_path[1] in ['cache_enabled']:
                    value = value.lower() in ('true', '1', 'yes', 'on')
                
                # Asignar valor
                self._set_nested_value(self.config, config_path, value)
                logger.info(f"Configuración {env_var} cargada desde entorno")
    
    def _merge_config(self, target: Dict, source: Dict):
        """Merge recursivo de diccionarios de configuración."""
        for key, value in source.items():
            if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                self._merge_config(target[key], value)
            else:
                target[key] = value
    
    def _set_nested_value(self, config: Dict, path: list, value: Any):
        """Establece valor en configuración anidada."""
        for key in path[:-1]:
            config = config.setdefault(key, {})
        config[path[-1]] = value
    
    def _validate_config(self):
        """Valida la configuración cargada."""
        # Validar API key de Sportradar
        if not self.config['sportradar']['api_key']:
            logger.warning("API key de Sportradar no configurada. Algunas funciones no estarán disponibles.")
        
        # Validar API key de Player Props
        if not self.config['player_props_v2']['api_key']:
            logger.warning("API key de Player Props v2 no configurada. Funciones de player props no estarán disponibles.")
        
        # Validar parámetros de betting
        betting = self.config['betting']
        if betting['minimum_edge'] <= 0 or betting['minimum_edge'] >= 1:
            raise ValueError("minimum_edge debe estar entre 0 y 1")
        
        if betting['confidence_threshold'] <= 0 or betting['confidence_threshold'] >= 1:
            raise ValueError("confidence_threshold debe estar entre 0 y 1")
        
        if betting['max_kelly_fraction'] <= 0 or betting['max_kelly_fraction'] > 1:
            raise ValueError("max_kelly_fraction debe estar entre 0 y 1")
        
        # Crear directorios necesarios
        self._create_directories()
    
    def _create_directories(self):
        """Crea directorios necesarios para cache y logs."""
        directories = [
            self.config['data']['cache_directory'],
            self.config['logging']['log_directory']
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
    
    def get(self, *keys) -> Any:
        """
        Obtiene valor de configuración usando claves anidadas.
        
        Args:
            *keys: Claves anidadas (ej: 'sportradar', 'api_key')
            
        Returns:
            Valor de configuración
        """
        value = self.config
        for key in keys:
            value = value[key]
        return value
    
    def set(self, *keys, value: Any):
        """
        Establece valor de configuración usando claves anidadas.
        
        Args:
            *keys: Claves anidadas
            value: Valor a establecer
        """
        config = self.config
        for key in keys[:-1]:
            config = config.setdefault(key, {})
        config[keys[-1]] = value
    
    def save_to_file(self, config_file: str):
        """
        Guarda configuración actual a archivo JSON.
        
        Args:
            config_file: Ruta del archivo de configuración
        """
        try:
            os.makedirs(os.path.dirname(config_file), exist_ok=True)
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=2, ensure_ascii=False)
            logger.info(f"Configuración guardada en {config_file}")
        except Exception as e:
            logger.error(f"Error guardando configuración: {e}")
    
    def get_sportradar_config(self) -> Dict[str, Any]:
        """Obtiene configuración específica de Sportradar."""
        return self.config['sportradar'].copy()
    
    def get_betting_config(self) -> Dict[str, Any]:
        """Obtiene configuración específica de betting."""
        return self.config['betting'].copy()
    
    def get_data_config(self) -> Dict[str, Any]:
        """Obtiene configuración específica de datos."""
        return self.config['data'].copy()
    
    def is_api_configured(self, api_type: str = 'sportradar') -> bool:
        """
        Verifica si la API está configurada correctamente.
        
        Args:
            api_type: Tipo de API ('sportradar', 'player_props_v2', 'both')
            
        Returns:
            True si la API está configurada
        """
        if api_type == 'sportradar':
            return bool(self.config['sportradar']['api_key'])
        elif api_type == 'player_props_v2':
            return bool(self.config['player_props_v2']['api_key'])
        elif api_type == 'both':
            return (bool(self.config['sportradar']['api_key']) and 
                   bool(self.config['player_props_v2']['api_key']))
        else:
            return bool(self.config['sportradar']['api_key'])
    
    def get_api_url(self, api_type: str = 'prematch') -> str:
        """
        Obtiene URL base para tipo de API específico.
        
        Args:
            api_type: Tipo de API ('prematch', 'player_props_v2', 'live_odds', 'basketball')
            
        Returns:
            URL base para el tipo de API
        """
        if api_type == 'player_props_v2':
            return self.config['player_props_v2']['base_url']
        elif api_type == 'prematch':
            return self.config['sportradar']['odds_base_url']
        elif api_type == 'live_odds':
            return self.config['sportradar']['live_odds_url']
        elif api_type == 'basketball':
            return self.config['sportradar']['base_url']
        else:
            # Default to prematch
            return self.config['sportradar']['odds_base_url']
    
    def get_endpoint(self, endpoint_name: str, api_type: str = 'sportradar', **kwargs) -> str:
        """
        Obtiene endpoint formateado con parámetros.
        
        Args:
            endpoint_name: Nombre del endpoint
            api_type: Tipo de API ('sportradar', 'player_props_v2')
            **kwargs: Parámetros para formatear endpoint
            
        Returns:
            Endpoint formateado
        """
        if api_type == 'player_props_v2':
            endpoints = self.config['player_props_v2']['endpoints']
        else:
            endpoints = self.config['sportradar']['endpoints']
            
        endpoint_template = endpoints.get(endpoint_name, '')
        
        try:
            return endpoint_template.format(**kwargs)
        except KeyError as e:
            logger.error(f"Parámetro faltante para endpoint {endpoint_name}: {e}")
            return endpoint_template
    
    def get_market_id(self, market_name: str) -> Optional[str]:
        """
        Obtiene ID de mercado por nombre.
        
        Args:
            market_name: Nombre del mercado
            
        Returns:
            ID del mercado o None si no existe
        """
        market_ids = self.config['sportradar']['market_ids']
        return market_ids.get(market_name)
    
    def get_target_markets(self, target: str) -> list:
        """
        Obtiene mercados asociados a un target del sistema.
        
        Args:
            target: Target del sistema (PTS, AST, etc.)
            
        Returns:
            Lista de nombres de mercados
        """
        target_mapping = self.config['sportradar']['target_to_market']
        markets = target_mapping.get(target, [])
        
        # Asegurar que siempre devuelva una lista
        if isinstance(markets, str):
            return [markets]
        return markets if isinstance(markets, list) else []
    
    def get_sport_id(self, sport: str = 'basketball') -> int:
        """
        Obtiene Sport ID para Sportradar.
        
        Args:
            sport: Deporte ('basketball', 'nba')
            
        Returns:
            Sport ID
        """
        sport_key = f"{sport}_sport_id"
        return int(self.config['sportradar'].get(sport_key, 2))  # Default basketball = 2
    
    def get_player_props_config(self) -> Dict[str, Any]:
        """Obtiene configuración específica de Player Props API."""
        return self.config['player_props_v2'].copy()
    
    def get_nba_competition_id(self, competition_key: str = 'nba') -> str:
        """
        Obtiene ID de competición NBA para Player Props API.
        
        Args:
            competition_key: Clave de competición ('nba', 'wnba', 'ncaa')
            
        Returns:
            ID de competición
        """
        return self.config['player_props_v2']['nba_competition_ids'].get(competition_key, 'sr:competition:132')
    
    def get_player_props_target_markets(self, target: str) -> List[str]:
        """
        Obtiene mercados de Player Props API para un target específico.
        
        Args:
            target: Target del sistema (PTS, AST, TRB, 3P, double_double)
            
        Returns:
            Lista de nombres de mercados para Player Props API
        """
        target_mapping = self.config['player_props_v2']['target_to_market']
        markets = target_mapping.get(target, [])
        
        # Asegurar que siempre devuelva una lista
        if isinstance(markets, str):
            return [markets]
        return markets if isinstance(markets, list) else []
    
    def __str__(self) -> str:
        """Representación string de la configuración (sin API keys)."""
        safe_config = self.config.copy()
        if 'api_key' in safe_config.get('sportradar', {}):
            safe_config['sportradar']['api_key'] = '***'
        if 'api_key' in safe_config.get('player_props_v2', {}):
            safe_config['player_props_v2']['api_key'] = '***'
        return json.dumps(safe_config, indent=2)


# Instancia global de configuración
_global_config = None

def get_config() -> BookmakersConfig:
    """Obtiene la instancia global de configuración."""
    global _global_config
    if _global_config is None:
        _global_config = BookmakersConfig()
    return _global_config

def set_config(config: BookmakersConfig):
    """Establece la instancia global de configuración."""
    global _global_config
    _global_config = config 