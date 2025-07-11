"""
Configuración para la integración con Sportradar API.
Actualizada según documentación oficial de Sportradar.
"""

import os
from configparser import ConfigParser
from typing import Dict, Any, Optional
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
        'odds_base_url': 'https://api.sportradar.com/oddscomparison-prematch/trial/v2',
        'player_props_url': 'https://api.sportradar.com/oddscomparison-player-props/trial/v2',
        'live_odds_url': 'https://api.sportradar.com/oddscomparison-live-odds/trial/v2',
        'regular_odds_url': 'https://api.sportradar.com/oddscomparison-trial/v1',
        
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
        
        # Endpoints según documentación oficial - SIN extensión, usar Accept: application/json
        'endpoints': {
            # Libros/casas de apuestas
            'books': 'en/books',
            # Player props por competición
            'competition_player_props': 'en/competitions/{competition_id}/player_props',
            # Calendario de player props por competición
            'competition_schedule': 'en/competitions/{competition_id}/schedule',
            # Player props por evento
            'sport_event_player_props': 'en/sport_events/{sport_event_id}/player_props',
            # Player props por fecha
            'date_player_props': 'en/sports/{sport_id}/schedule/{date}/player_props',
            # Calendario por fecha
            'date_schedule': 'en/sports/{sport_id}/schedule/{date}/schedule',
            # Changelog
            'player_props_changelog': 'en/player_props_changelog',
            # Deportes
            'sports': 'en/sports',
            # Competiciones
            'sport_competitions': 'en/sports/{sport_id}/competitions',
            # Categorías
            'sport_categories': 'en/sports/{sport_id}/categories',
            # Stages
            'sport_stages': 'en/sports/{sport_id}/stages',
            # Mapping
            'competition_mappings': 'en/mappings/competitions',
            'competitor_mappings': 'en/mappings/competitors',
            'player_mappings': 'en/mappings/players',
            'sport_event_mappings': 'en/mappings/sport_events'
        },
        
        # Market IDs para NBA según documentación oficial
        'market_ids': {
            # Mercados de equipo (Prematch API)
            '1x2': 1,  # Winner/Moneyline → is_win
            'total_incl_overtime': 225,  # Total points (incl. overtime) → total_points
            'home_total_incl_overtime': 227,  # Home total → teams_points
            'away_total_incl_overtime': 228,  # Away total → teams_points
            
            # Player Props Markets (Player Props API)
            'total_points': 'sr:market:921',  # Total points (incl. overtime) → PTS
            'total_assists': 'sr:market:922',  # Total assists (incl. overtime) → AST
            'total_rebounds': 'sr:market:923',  # Total rebounds (incl. overtime) → TRB
            'total_3pt_field_goals': 'sr:market:924',  # Total 3-point field goals → 3P
            'double_double': 'sr:market:8008'  # Double double → double_double
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
        'simulate_when_no_data': True,
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
            'SPORTRADAR_API_KEY': ['sportradar', 'api_key'],
            'API_SPORTRADAR': ['sportradar', 'api_key'],  # Soporte para ambos formatos
            'SPORTRADAR_BASE_URL': ['sportradar', 'base_url'],
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
    
    def is_api_configured(self) -> bool:
        """Verifica si la API está configurada correctamente."""
        return bool(self.config['sportradar']['api_key'])
    
    def get_api_url(self, api_type: str = 'prematch') -> str:
        """
        Obtiene URL base para tipo de API específico.
        
        Args:
            api_type: Tipo de API ('prematch', 'player_props', 'live_odds', 'basketball')
            
        Returns:
            URL base para el tipo de API
        """
        url_map = {
            'prematch': 'odds_base_url',
            'player_props': 'player_props_url', 
            'live_odds': 'live_odds_url',
            'basketball': 'base_url'
        }
        
        url_key = url_map.get(api_type, 'odds_base_url')
        return self.config['sportradar'][url_key]
    
    def get_endpoint(self, endpoint_name: str, **kwargs) -> str:
        """
        Obtiene endpoint formateado con parámetros.
        
        Args:
            endpoint_name: Nombre del endpoint
            **kwargs: Parámetros para formatear endpoint
            
        Returns:
            Endpoint formateado
        """
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
    
    def __str__(self) -> str:
        """Representación string de la configuración (sin API keys)."""
        safe_config = self.config.copy()
        if 'api_key' in safe_config.get('sportradar', {}):
            safe_config['sportradar']['api_key'] = '***'
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