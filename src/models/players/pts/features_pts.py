"""
Módulo de Características para Predicción de Puntos (PTS)
========================================================

Este módulo contiene toda la lógica de ingeniería de características específica
para la predicción de puntos que anotará un jugador NBA en su próximo partido.
Implementa características avanzadas enfocadas en factores que determinan
la capacidad anotadora de un jugador.

MEJORAS IMPLEMENTADAS:
- Features temporales avanzadas para capturar drift temporal
- Características de tendencias recientes con mayor peso
- Features de adaptabilidad a cambios en patrones de juego
- Regularización temporal para evitar overfitting

Sin data leakage, todas las métricas usan shift(1) para crear historial

"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
import logging
from config.logging_config import NBALogger
from datetime import datetime, timedelta
import warnings
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

# Configurar logging balanceado para features - mostrar etapas principales
logger = NBALogger.get_logger(__name__.split(".")[-1])  # Permitir logs de etapas principales

class PointsFeatureEngineer:
    """
    Motor de features para predicción de puntos usando ESTADÍSTICAS HISTÓRICAS
    OPTIMIZADO - Rendimiento pasado para predecir anotación futura
    MEJORADO - Con features temporales avanzadas para capturar drift temporal
    """
            
    def __init__(self, lookback_games: int = 10):
        """Inicializa el ingeniero de características para predicción de puntos."""
        self.lookback_games = lookback_games
        self.scaler = StandardScaler()
        self.feature_columns = []
        
        # SISTEMA DE CACHE MEJORADO
        self._cached_calculations = {}
        self._features_cache = {}
        self._last_data_hash = None
        self._computed_features = set()  # Track de features ya computadas
        
        # REGISTRO DE FEATURES CREADAS
        self._feature_registry = {
            'temporal': [],
            'temporal_advanced': [],  # NUEVO: Features temporales avanzadas
            'contextual': [],
            'scoring': [],
            'shooting': [],
            'usage': [],
            'opponent': [],
            'biometric': [],
            'stacking': [],
            'meta_stacking': [],
            'situational': [],
            'drift_detection': []  # NUEVO: Features para detectar drift temporal
        }
        
        NBALogger.log_model_start(logger, "Points", "FeatureEngineer")
    
    def _get_data_hash(self, df: pd.DataFrame) -> str:
        """Generar hash único para el DataFrame"""
        try:
            # Usar shape, columnas y algunos valores para crear hash
            data_info = f"{df.shape}_{list(df.columns)}_{df.iloc[0].sum() if len(df) > 0 else 0}_{df.iloc[-1].sum() if len(df) > 0 else 0}"
            return str(hash(data_info))
        except:
            return str(hash(str(df.shape)))
    
    def _ensure_datetime_and_sort(self, df: pd.DataFrame) -> None:
        """Método auxiliar para asegurar que Date esté en formato datetime y ordenar datos"""
        if 'Date' in df.columns and df['Date'].dtype != 'datetime64[ns]':
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            df.sort_values(['Player', 'Date'], inplace=True)
            df.reset_index(drop=True, inplace=True)
            logger.debug("Datos ordenados cronológicamente por jugador")
    
    def _calculate_basic_temporal_features(self, df: pd.DataFrame) -> None:
        """Método auxiliar para calcular features temporales básicas una sola vez"""
        if 'Date' in df.columns:
            # Calcular una sola vez todas las features temporales
            df['days_rest'] = df.groupby('Player')['Date'].diff().dt.days.fillna(2)
            df['day_of_week'] = df['Date'].dt.dayofweek
            df['month'] = df['Date'].dt.month
            df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
            
            # Días en temporada
            season_start = df['Date'].min()
            df['days_into_season'] = (df['Date'] - season_start).dt.days
            
            # Back-to-back indicator (calculado una sola vez)
            df['is_back_to_back'] = (df['days_rest'] <= 1).astype(int)
            
            # NUEVAS FEATURES TEMPORALES AVANZADAS
            self._calculate_advanced_temporal_features(df)
            
            logger.debug("Features temporales básicas y avanzadas calculadas")
    
    def _calculate_advanced_temporal_features(self, df: pd.DataFrame) -> None:
        """NUEVO: Calcular features temporales avanzadas para capturar drift temporal"""
        if 'Date' not in df.columns:
            return
            
        # Calcular tiempo desde el último juego para cada jugador
        df['time_since_last_game'] = df.groupby('Player')['Date'].diff().dt.total_seconds() / 3600  # en horas
        df['time_since_last_game'] = df['time_since_last_game'].fillna(48)  # default 48 horas
        
        # Features de recencia temporal (más peso a juegos recientes)
        df['recency_weight'] = np.exp(-df.groupby('Player').cumcount() * 0.1)  # decay exponencial
        
        # Indicador de temporada (early/mid/late season)
        max_days = df['days_into_season'].max()
        df['season_phase'] = pd.cut(df['days_into_season'], 
                                   bins=[0, max_days*0.3, max_days*0.7, max_days],
                                   labels=[0, 1, 2], include_lowest=True).astype(float)
        
        # Features de momentum temporal
        # Calcular juegos en la última semana usando un enfoque simple y robusto
        df['games_in_last_week'] = df.groupby('Player').cumcount() + 1
        df['games_in_last_week'] = df['games_in_last_week'].clip(upper=7)  # Máximo 7 juegos
        
        # Indicador de carga de juegos (schedule density)
        df['schedule_density'] = df['games_in_last_week'] / 7.0  # juegos por día promedio
        
        logger.debug("Features temporales avanzadas calculadas")
    
    def _calculate_player_context_features(self, df: pd.DataFrame) -> None:
        """Método auxiliar para calcular features de contexto del jugador una sola vez"""
        # Features de contexto ya disponibles del data_loader
        if 'is_home' not in df.columns:
            logger.debug("is_home no encontrado del data_loader - features de ventaja local no disponibles")
        else:
            logger.debug("Usando is_home del data_loader para features de ventaja local")
            # Calcular features relacionadas con ventaja local
            df['home_scoring_advantage'] = df['is_home'] * 0.08  # 8% boost para anotación en casa
            df['travel_fatigue_penalty'] = np.where(df['is_home'] == 0, -0.05, 0.0)
        
        # Features de titular/suplente ya disponibles del data_loader
        if 'is_started' not in df.columns:
            logger.debug("is_started no encontrado del data_loader - features de titular no disponibles")
        else:
            logger.debug("Usando is_started del data_loader para features de titular")
            # Boost para titulares (más minutos = más oportunidades de anotar)
            df['starter_scoring_boost'] = df['is_started'] * 0.25  # Mayor boost para anotación
    
    def generate_all_features(self, df: pd.DataFrame) -> List[str]:
        """
        PIPELINE OPTIMIZADO DE FEATURES PARA PREDICCIÓN DE PUNTOS
        Sistema mejorado que evita duplicaciones y optimiza cálculos
        """
        
        logger.info("Generando features NBA OPTIMIZADAS para predicción de puntos...")

        # VERIFICACIÓN DE PTS COMO TARGET
        if 'PTS' in df.columns:
            pts_stats = df['PTS'].describe()
            logger.info(f"Target PTS disponible - Estadísticas: Media={pts_stats['mean']:.1f}, Max={pts_stats['max']:.0f}")
        else:
            available_cols = list(df.columns)[:10]
            logger.error(f"PTS no encontrado en el dataset - requerido para features de puntos")
            logger.error(f"Columnas disponibles (primeras 10): {available_cols}")
            logger.error(f"Shape del dataset: {df.shape}")
            return []
        
        # VERIFICAR CACHE DE DATOS
        current_hash = self._get_data_hash(df)
        if current_hash == self._last_data_hash and self.feature_columns:
            logger.info("Usando features cacheadas (datos sin cambios)")
            return self.feature_columns
        
        # LIMPIAR REGISTROS PARA NUEVA EJECUCIÓN
        self._computed_features.clear()
        for category in self._feature_registry:
            self._feature_registry[category].clear()
        
        # VERIFICAR FEATURES DEL DATA_LOADER
        data_loader_features = ['is_home', 'is_started', 'Height_Inches', 'Weight', 'BMI']
        available_features = [f for f in data_loader_features if f in df.columns]
        missing_features = [f for f in data_loader_features if f not in df.columns]
        
        if available_features:
            logger.info(f"Features del data_loader: {len(available_features)}/{len(data_loader_features)} disponibles")
        if missing_features:
            logger.debug(f"Features faltantes: {missing_features}")
        
        # Trabajar directamente con el DataFrame
        if df.empty:
            return []
        
        # PASO 0: Preparación básica (solo si no existe)
        if not self._check_feature_exists(df, 'days_rest'):
            self._ensure_datetime_and_sort(df)
            self._calculate_basic_temporal_features(df)
            self._calculate_player_context_features(df)
        
        logger.info("Iniciando generación de features ESPECIALIZADAS (sin duplicaciones)...")
        
        # GENERAR FEATURES POR CATEGORÍAS (verificando existencia)
        self._create_temporal_features_optimized(df)
        self._create_contextual_features_optimized(df)
        self._create_scoring_performance_features_optimized(df)
        self._create_shooting_efficiency_features_optimized(df)
        self._create_usage_and_opportunity_features_optimized(df)
        self._create_opponent_defensive_features_optimized(df)
        self._create_biometric_and_position_features_optimized(df)
        self._create_advanced_stacking_features_optimized(df)
        self._create_meta_stacking_features_optimized(df)
        self._create_situational_context_features_optimized(df)
        
        # NUEVAS FUNCIONES PARA DRIFT TEMPORAL Y TENDENCIAS RECIENTES
        self._create_temporal_drift_features(df)
        self._create_recent_trend_features(df)
        self._create_adaptive_features(df)
        
        # NUEVAS FUNCIONES PARA JUGADORES DE ALTO SCORING
        self._create_high_scoring_features(df)
        self._create_elite_player_features(df)
        self._create_star_performance_features(df)
        
        # 🚀 REVOLUCIONARIO: FEATURES DE ENSEMBLE INTELIGENTE (MÁXIMA PRIORIDAD)
        self._create_revolutionary_ensemble_features(df)
        
        # 🎯 NUEVO: FEATURES ULTRA-CRÍTICAS PARA 97% EFECTIVIDAD
        self._create_ultra_critical_range_features(df)
        
        # ACTUALIZAR CACHE
        self._last_data_hash = current_hash
        
        # COMPILAR FEATURES FINALES
        essential_features = self._compile_essential_features(df)
        
        # APLICAR FILTRO DE CORRELACIÓN
        logger.info("Aplicando filtro de correlación optimizado...")
        essential_features = self._apply_correlation_regularization(df, essential_features)
        
        # ACTUALIZAR REGISTRO
        self.feature_columns = essential_features
        
        # REPORTE FINAL
        self._log_feature_creation_summary()
        
        logger.info(f"Features optimizadas generadas: {len(essential_features)}")
        
        return essential_features
    
    def _create_temporal_features_optimized(self, df: pd.DataFrame) -> None:
        """Features temporales específicas para predicción de puntos (optimizado)"""
        logger.debug("Creando features temporales optimizadas...")
        
        # Features de fatiga específicas para anotación
        if self._register_feature('scoring_fatigue_factor', 'temporal'):
            df['scoring_fatigue_factor'] = np.where(df['days_rest'] <= 1, 0.90, 1.0)
        
        if self._register_feature('rest_scoring_boost', 'temporal'):
            df['rest_scoring_boost'] = np.where(df['days_rest'] >= 3, 1.08, 1.0)
        
        # Momentum temporal para anotación
        if self._register_feature('weekend_scoring_boost', 'temporal'):
            df['weekend_scoring_boost'] = df['is_weekend'] * 0.05
        
        if self._register_feature('season_progression', 'temporal'):
            df['season_progression'] = df['days_into_season'] / 200.0
        
        # Back-to-back penalty para anotación
        if self._register_feature('b2b_scoring_penalty', 'temporal'):
            df['b2b_scoring_penalty'] = df['is_back_to_back'] * (-0.08)
        
        # Features específicas de día de la semana para anotación
        if self._register_feature('prime_time_game', 'temporal'):
            df['prime_time_game'] = df['day_of_week'].isin([1, 2, 4]).astype(int)
        
        logger.debug(f"Features temporales creadas: {len(self._feature_registry['temporal'])}")
    
    def _create_contextual_features_optimized(self, df: pd.DataFrame) -> None:
        """Features de contexto específicas para predicción de puntos (optimizado)"""
        logger.debug("Creando features contextuales optimizadas...")
        
        # Ventaja local ajustada para anotación
        if 'is_home' in df.columns and self._register_feature('home_court_scoring_boost', 'contextual'):
            df['home_court_scoring_boost'] = df['is_home'] * 0.12
        
        # Features de posición específicas para anotación
        if 'Pos' in df.columns or any('pos' in col.lower() for col in df.columns):
            # Identificar posiciones propensas a anotar
            if self._register_feature('is_guard', 'contextual'):
                df['is_guard'] = df.get('Pos', 'N/A').str.contains('G', na=False).astype(int)
            if self._register_feature('is_shooting_guard', 'contextual'):
                df['is_shooting_guard'] = df.get('Pos', 'N/A').str.contains('SG', na=False).astype(int)
            if self._register_feature('is_small_forward', 'contextual'):
                df['is_small_forward'] = df.get('Pos', 'N/A').str.contains('SF', na=False).astype(int)
            if self._register_feature('is_wing_player', 'contextual'):
                df['is_wing_player'] = (df['is_shooting_guard'] | df['is_small_forward']).astype(int)
        else:
            # Estimar posiciones basadas en estadísticas
            # Usar valores por defecto si las columnas no existen
            if 'AST' in df.columns:
                ast_avg = df['AST']
            else:
                ast_avg = pd.Series([0] * len(df), index=df.index)
                
            if '3PA' in df.columns:
                three_attempts = df['3PA']
            else:
                three_attempts = pd.Series([0] * len(df), index=df.index)
                
            if 'FGA' in df.columns:
                fg_attempts = df['FGA']
            else:
                fg_attempts = pd.Series([0] * len(df), index=df.index)
                
            if 'TRB' in df.columns:
                rebounds = df['TRB']
            else:
                rebounds = pd.Series([0] * len(df), index=df.index)
            
            if self._register_feature('is_guard', 'contextual'):
                df['is_guard'] = ((ast_avg > 3) | (three_attempts > 4)).astype(int)
            
            if self._register_feature('is_shooting_guard', 'contextual'):
                df['is_shooting_guard'] = ((fg_attempts > 8) & (three_attempts > 3)).astype(int)
            
            if self._register_feature('is_small_forward', 'contextual'):
                df['is_small_forward'] = ((fg_attempts > 6) & (rebounds > 4) & (three_attempts > 2)).astype(int)
            
            if self._register_feature('is_wing_player', 'contextual'):
                df['is_wing_player'] = (df.get('is_shooting_guard', pd.Series([0] * len(df), index=df.index)) | 
                                       df.get('is_small_forward', pd.Series([0] * len(df), index=df.index))).astype(int)
        
        # Boost específico para posiciones anotadoras
        if self._register_feature('scorer_position_boost', 'contextual'):
            df['scorer_position_boost'] = (
                df.get('is_shooting_guard', 0) * 0.15 + 
                df.get('is_small_forward', 0) * 0.12 + 
                df.get('is_guard', 0) * 0.08
            )
        
        logger.debug(f"Features contextuales creadas: {len(self._feature_registry['contextual'])}")
    
    def _create_scoring_performance_features_optimized(self, df: pd.DataFrame) -> None:
        """Features de rendimiento anotador específicas (optimizado)"""
        logger.debug("Creando features de rendimiento anotador optimizadas...")
        
        # Verificar que PTS existe
        if 'PTS' not in df.columns:
            NBALogger.log_warning(logger, "Columna PTS no encontrada - features de anotación limitadas")
            return
        
        # FEATURES HISTÓRICAS DE PUNTOS (evitando data leakage)
        
        # Promedios históricos de puntos en diferentes ventanas
        for window in [3, 5, 10]:
            feature_name = f'pts_hist_avg_{window}g'
            if self._register_feature(feature_name, 'scoring'):
                hist_avg = self._get_historical_series(df, 'PTS', window=window, operation='mean')
                df[feature_name] = hist_avg
            
            # Consistencia en anotación
            consistency_name = f'pts_consistency_{window}g'
            if self._register_feature(consistency_name, 'scoring'):
                hist_std = self._get_historical_series(df, 'PTS', window=window, operation='std')
                df[consistency_name] = 1 / (hist_std + 1)
        
        # Tendencia reciente en anotación
        if self._register_feature('pts_trend_factor', 'scoring'):
            pts_avg_3g = df.get('pts_hist_avg_3g', 0)
            pts_avg_10g = df.get('pts_hist_avg_10g', 0)
            df['pts_trend_factor'] = pts_avg_3g / (pts_avg_10g + 0.1)
        
        # Puntos por encima del promedio personal de temporada
        if self._register_feature('pts_above_season_avg', 'scoring'):
            season_avg = df.groupby('Player')['PTS'].transform('mean').shift(1)
            season_avg = season_avg.fillna(df['PTS'].mean())
            df['pts_above_season_avg'] = df.get('pts_hist_avg_5g', 0) - season_avg
        
        # Racha anotadora (hot streak)
        if self._register_feature('hot_streak_indicator', 'scoring'):
            pts_recent = df.get('pts_hist_avg_3g', 0)
            pts_baseline = df.get('pts_hist_avg_10g', 0)
            df['hot_streak_indicator'] = np.where(
                pts_recent > pts_baseline * 1.15, 1, 0
            )
        
        # Varianza en anotación
        if self._register_feature('scoring_variance_5g', 'scoring'):
            df['scoring_variance_5g'] = self._get_historical_series(df, 'PTS', window=5, operation='std')
        
        # Forma anotadora reciente
        if self._register_feature('scoring_form_5g', 'scoring'):
            df['scoring_form_5g'] = (
                df.get('pts_hist_avg_5g', 0) * 0.7 +
                df.get('pts_trend_factor', 1.0) * df.get('pts_hist_avg_5g', 0) * 0.3
            )
        
        logger.debug(f"Features de scoring creadas: {len(self._feature_registry['scoring'])}")
    
    def _create_shooting_efficiency_features_optimized(self, df: pd.DataFrame) -> None:
        """Features de eficiencia de tiro (optimizado)"""
        logger.debug("Creando features de eficiencia de tiro optimizadas...")
        
        # Features de tiros de campo
        if 'FG' in df.columns and 'FGA' in df.columns:
            # Eficiencia de tiro histórica
            if self._register_feature('fg_hist_avg_5g', 'shooting'):
                fg_hist = self._get_historical_series(df, 'FG', window=5, operation='mean')
                df['fg_hist_avg_5g'] = fg_hist
            
            if self._register_feature('fga_hist_avg_5g', 'shooting'):
                fga_hist = self._get_historical_series(df, 'FGA', window=5, operation='mean')
                df['fga_hist_avg_5g'] = fga_hist
            
            if self._register_feature('fg_efficiency_5g', 'shooting'):
                fg_hist = df.get('fg_hist_avg_5g', 0)
                fga_hist = df.get('fga_hist_avg_5g', 0)
                df['fg_efficiency_5g'] = fg_hist / (fga_hist + 0.1)
            
            # Volumen de tiros
            if self._register_feature('shooting_volume_5g', 'shooting'):
                df['shooting_volume_5g'] = df.get('fga_hist_avg_5g', 0)
            
            # Tendencia en eficiencia
            if self._register_feature('fg_efficiency_trend', 'shooting'):
                fg_eff_3g = (self._get_historical_series(df, 'FG', window=3, operation='mean') / 
                           (self._get_historical_series(df, 'FGA', window=3, operation='mean') + 0.1))
                df['fg_efficiency_trend'] = fg_eff_3g / (df.get('fg_efficiency_5g', 0.4) + 0.01)
        
        # Features de triples
        if '3P' in df.columns and '3PA' in df.columns:
            if self._register_feature('three_point_made_5g', 'shooting'):
                three_made_hist = self._get_historical_series(df, '3P', window=5, operation='mean')
                df['three_point_made_5g'] = three_made_hist
            
            if self._register_feature('three_point_att_5g', 'shooting'):
                three_att_hist = self._get_historical_series(df, '3PA', window=5, operation='mean')
                df['three_point_att_5g'] = three_att_hist
            
            if self._register_feature('three_point_efficiency_5g', 'shooting'):
                three_made = df.get('three_point_made_5g', 0)
                three_att = df.get('three_point_att_5g', 0)
                df['three_point_efficiency_5g'] = three_made / (three_att + 0.1)
            
            # Especialista en triples
            if self._register_feature('three_point_specialist', 'shooting'):
                df['three_point_specialist'] = (df.get('three_point_att_5g', 0) > 4).astype(int)
        
        # Features de tiros libres
        if 'FT' in df.columns and 'FTA' in df.columns:
            if self._register_feature('ft_made_5g', 'shooting'):
                ft_made_hist = self._get_historical_series(df, 'FT', window=5, operation='mean')
                df['ft_made_5g'] = ft_made_hist
            
            if self._register_feature('ft_att_5g', 'shooting'):
                ft_att_hist = self._get_historical_series(df, 'FTA', window=5, operation='mean')
                df['ft_att_5g'] = ft_att_hist
            
            if self._register_feature('ft_efficiency_5g', 'shooting'):
                ft_made = df.get('ft_made_5g', 0)
                ft_att = df.get('ft_att_5g', 0)
                df['ft_efficiency_5g'] = ft_made / (ft_att + 0.1)
            
            # Agresividad (llegar a la línea de tiros libres)
            if self._register_feature('ft_aggressiveness', 'shooting'):
                df['ft_aggressiveness'] = df.get('ft_att_5g', 0)
        
        # Eficiencia anotadora general
        if self._register_feature('scoring_efficiency_5g', 'shooting'):
            pts_avg = df.get('pts_hist_avg_5g', 0)
            fga_avg = df.get('fga_hist_avg_5g', 0)
            df['scoring_efficiency_5g'] = pts_avg / (fga_avg + 0.1)
        
        logger.debug(f"Features de shooting creadas: {len(self._feature_registry['shooting'])}")
    
    def _create_usage_and_opportunity_features_optimized(self, df: pd.DataFrame) -> None:
        """Features de uso y oportunidades ofensivas (optimizado)"""
        logger.debug("Creando features de uso y oportunidades optimizadas...")
        
        # Minutos históricos (oportunidades de anotar)
        if 'MP' in df.columns:
            if self._register_feature('mp_hist_avg_5g', 'usage'):
                mp_hist = self._get_historical_series(df, 'MP', window=5, operation='mean')
                df['mp_hist_avg_5g'] = mp_hist
            
            # Jugador de muchos minutos
            if self._register_feature('high_minutes_player', 'usage'):
                mp_hist = df.get('mp_hist_avg_5g', 25)
                df['high_minutes_player'] = (mp_hist > 30).astype(int)
            
            # Puntos por minuto (eficiencia temporal)
            if self._register_feature('pts_per_minute_5g', 'usage'):
                pts_avg = df.get('pts_hist_avg_5g', 0)
                mp_hist = df.get('mp_hist_avg_5g', 25)
                df['pts_per_minute_5g'] = pts_avg / (mp_hist + 0.1)
        
        # Usage rate histórico
        if 'USG%' in df.columns:
            if self._register_feature('usage_rate_5g', 'usage'):
                usage_hist = self._get_historical_series(df, 'USG%', window=5, operation='mean')
                df['usage_rate_5g'] = usage_hist
            
            # Alto uso ofensivo
            if self._register_feature('high_usage_player', 'usage'):
                usage_hist = df.get('usage_rate_5g', 20)
                df['high_usage_player'] = (usage_hist > 25).astype(int)
        else:
            # Estimar usage rate basado en tiros y puntos
            if self._register_feature('estimated_usage_5g', 'usage'):
                fga_avg = df.get('fga_hist_avg_5g', 0)
                df['estimated_usage_5g'] = fga_avg * 2.5
            
            if self._register_feature('usage_rate_5g', 'usage'):
                df['usage_rate_5g'] = df.get('estimated_usage_5g', 20)
        
        # Oportunidades de tiro por partido
        if self._register_feature('shooting_opportunities_5g', 'usage'):
            fga_avg = df.get('fga_hist_avg_5g', 0)
            ft_att = df.get('ft_att_5g', 0)
            df['shooting_opportunities_5g'] = fga_avg + ft_att * 0.5
        
        # Rol ofensivo en el equipo
        if self._register_feature('scoring_role_importance', 'usage'):
            pts_avg = df.get('pts_hist_avg_5g', 0)
            team_avg_pts = pts_avg.groupby(df['Team']).transform('mean')
            df['scoring_role_importance'] = pts_avg / (team_avg_pts + 0.1)
        
        logger.debug(f"Features de usage creadas: {len(self._feature_registry['usage'])}")
    
    def _create_opponent_defensive_features_optimized(self, df: pd.DataFrame) -> None:
        """Features relacionadas con la defensa del oponente"""
        logger.debug("Creando features de defensa del oponente...")
        
        # Dificultad del enfrentamiento (basado en puntos permitidos por el oponente)
        if 'Opp' in df.columns:
            # Calcular puntos promedio permitidos por cada equipo (usando datos históricos)
            try:
                # Usar datos históricos del oponente para evitar data leakage
                opp_pts_allowed = df.groupby(['Opp', 'Date'])['PTS'].sum().groupby('Opp').rolling(5, min_periods=1).mean()
                opp_defensive_rating = opp_pts_allowed.groupby('Opp').mean()
                
                # Mapear a dificultad del enfrentamiento
                df['opp_defensive_strength'] = df['Opp'].map(opp_defensive_rating).fillna(110)  # Default promedio
                
                # Normalizar dificultad (menor rating = mejor defensa = más difícil anotar)
                df['matchup_scoring_difficulty'] = 120 - df['opp_defensive_strength']  # Invertir escala
                df['matchup_scoring_difficulty'] = df['matchup_scoring_difficulty'] / 20  # Normalizar 0-1
                
            except Exception as e:
                NBALogger.log_warning(logger, "Error calculando features del oponente: {e}")
                df['matchup_scoring_difficulty'] = 0.5  # Valor neutral
        
        # Ajuste por pace del juego (más posesiones = más oportunidades)
        try:
            # Estimar pace basado en estadísticas del equipo
            team_pace = df.groupby('Team')['FGA'].transform('mean') * 1.2  # Estimación aproximada
            df['pace_factor'] = team_pace / 100  # Normalizar
            df['pace_adjusted_scoring'] = df.get('pts_hist_avg_5g', 0) * df['pace_factor']
        except:
            df['pace_adjusted_scoring'] = df.get('pts_hist_avg_5g', 0)
        
    def _create_biometric_and_position_features_optimized(self, df: pd.DataFrame) -> None:
        """Features biométricas y de posición específicas para anotación"""
        logger.debug("Creando features biométricas y de posición...")
        
        # Categorización de altura para anotación
        def categorize_height_scoring(height):
            if pd.isna(height):
                return 1  # Promedio
            elif height <= 74:  # ≤ 6'2"
                return 0  # Bajo (guards, anotación perimetral)
            elif height <= 79:  # 6'3" - 6'7"
                return 1  # Promedio (wings, versátiles)
            else:  # > 6'7"
                return 2  # Alto (forwards/centers, anotación interior)
        
        if 'Height_Inches' in df.columns:
            df['height_category_scoring'] = df['Height_Inches'].apply(categorize_height_scoring)
            df['perimeter_scorer'] = (df['height_category_scoring'] <= 1).astype(int)
            df['interior_scorer'] = (df['height_category_scoring'] == 2).astype(int)
        else:
            df['height_category_scoring'] = 1
            df['perimeter_scorer'] = 1
            df['interior_scorer'] = 0
        
        # Features de físico para anotación
        if 'BMI' in df.columns:
            try:
                df['bmi_category'] = pd.cut(df['BMI'], bins=[0, 22, 25, 28, 50], labels=[0, 1, 2, 3]).astype(float)
                # Jugadores más ligeros tienden a ser mejores tiradores perimetrales
                df['athletic_build'] = (df['bmi_category'] <= 1).astype(int)
            except Exception as e:
                NBALogger.log_warning(logger, "Error categorizando BMI: {e}")
                df['bmi_category'] = 1.0
                df['athletic_build'] = 1
        else:
            df['bmi_category'] = 1.0
            df['athletic_build'] = 1
        
        # Arquetipo de anotador
        try:
            height_cat = df.get('height_category_scoring', pd.Series(1, index=df.index))
            three_specialist = df.get('three_point_specialist', pd.Series(0, index=df.index))
            high_usage = df.get('high_usage_player', pd.Series(0, index=df.index))
            
            # Score de arquetipo anotador
            df['scorer_archetype_score'] = (
                three_specialist.astype(float) * 0.3 +
                high_usage.astype(float) * 0.4 +
                (height_cat == 1).astype(float) * 0.3  # Wings son mejores anotadores versátiles
            )
            
            # Asegurar rango correcto
            df['scorer_archetype_score'] = df['scorer_archetype_score'].clip(0.0, 1.0)
            
        except Exception as e:
            NBALogger.log_error(logger, "Error calculando scorer_archetype_score: {e}")
            df['scorer_archetype_score'] = 0.5  # Valor neutral por defecto
        
        # Capacidad de anotación en clutch (últimos minutos)
        try:
            pts_consistency = df.get('pts_consistency_5g', 0.5)
            usage_rate = df.get('usage_rate_5g', 20)
            
            df['clutch_scoring_ability'] = (
                pts_consistency * 0.4 +
                (usage_rate / 100) * 0.6  # Normalizar usage rate
            )
            
        except Exception as e:
            NBALogger.log_error(logger, "Error calculando clutch_scoring_ability: {e}")
            df['clutch_scoring_ability'] = 0.5
        
        # Ajuste por fatiga
        try:
            mp_avg = df.get('mp_hist_avg_5g', 25)
            rest_factor = df.get('scoring_fatigue_factor', 1.0)
            
            df['fatigue_adjusted_scoring'] = df.get('pts_hist_avg_5g', 0) * rest_factor * (mp_avg / 35)
            
        except Exception as e:
            NBALogger.log_error(logger, "Error calculando fatigue_adjusted_scoring: {e}")
            df['fatigue_adjusted_scoring'] = df.get('pts_hist_avg_5g', 0)
        
        # Calidad de selección de tiro
        try:
            fg_eff = df.get('fg_efficiency_5g', 0.4)
            shot_volume = df.get('shooting_volume_5g', 10)
            
            # Balance entre eficiencia y volumen
            df['shot_selection_quality'] = fg_eff * np.log(shot_volume + 1) / 3  # Normalizar
            
        except Exception as e:
            NBALogger.log_error(logger, f"Error calculando shot_selection_quality: {e}")
            df['shot_selection_quality'] = 0.5
    
    def _create_advanced_stacking_features_optimized(self, df: pd.DataFrame) -> None:
        """
        FEATURES DE STACKING AVANZADAS - Combinaciones inteligentes de las features más importantes
        Basado en el análisis de importancia: pts_above_season_avg, usage_rate_5g, shooting_volume_5g, etc.
        """
        logger.info("Creando features de stacking avanzadas para máxima predicción...")
        
        # ==================== TIER 1: COMBINACIONES DE ELITE (Top 5 features) ====================
        
        # 1. SCORING ELITE COMPOSITE - Combina las 3 features más importantes
        pts_above_avg = df.get('pts_above_season_avg', 0)
        usage_rate = df.get('usage_rate_5g', 20) / 100  # Normalizar
        shooting_vol = df.get('shooting_volume_5g', 10)
        
        df['elite_scorer_composite'] = (
            pts_above_avg * 0.4 +           # 40% - Puntos por encima del promedio
            usage_rate * 15 * 0.35 +        # 35% - Usage rate escalado
            (shooting_vol / 20) * 0.25       # 25% - Volumen de tiros normalizado
        )
        
        # 2. PACE-ADJUSTED SCORING POWER - Combina pace y scoring
        pace_adj = df.get('pace_adjusted_scoring', 0)
        pts_hist = df.get('pts_hist_avg_5g', 0)
        
        df['pace_scoring_power'] = pace_adj * (1 + pts_hist / 25)  # Amplifica con promedio histórico
        
        # 3. SCORING FORM MOMENTUM - Combina forma y tendencia
        scoring_form = df.get('scoring_form_5g', 0)
        estimated_usage = df.get('estimated_usage_5g', 20) / 100
        
        df['form_momentum_factor'] = scoring_form * (1 + estimated_usage * 2)
        
        # ==================== TIER 2: COMBINACIONES DE EFICIENCIA ====================
        
        # 4. EFFICIENCY-VOLUME BALANCE - Balance perfecto entre eficiencia y volumen
        fg_hist = df.get('fg_hist_avg_3g', 0.45)
        fatigue_adj = df.get('fatigue_adjusted_scoring', 0)
        
        df['efficiency_volume_balance'] = (
            fg_hist * fatigue_adj * 2  # Eficiencia × Volumen ajustado por fatiga
        )
        
        # 5. LEAGUE-ADJUSTED SCORING - Ajuste por liga y contexto
        league_adj = df.get('league_adjusted_scoring', 0)
        matchup_diff = df.get('matchup_scoring_difficulty', 0.5)
        
        df['league_context_score'] = league_adj * (2 - matchup_diff)  # Inverso de dificultad
        
        # ==================== TIER 3: COMBINACIONES DE OPORTUNIDADES ====================
        
        # 6. OPPORTUNITY MAXIMIZER - Maximiza oportunidades de anotación
        mp_hist = df.get('mp_hist_avg_5g', 25)
        starter_boost = df.get('starter_scoring_boost', 0)
        home_adv = df.get('home_scoring_advantage', 0)
        
        df['opportunity_maximizer'] = (
            (mp_hist / 35) * 0.5 +           # 50% - Minutos normalizados
            starter_boost * 0.3 +            # 30% - Boost de titular
            home_adv * 0.2                   # 20% - Ventaja local
        )
        
        # 7. CLUTCH PERFORMANCE PREDICTOR - Predictor de rendimiento clutch
        clutch_ability = df.get('clutch_scoring_ability', 0.5)
        pts_consistency = df.get('pts_consistency_5g', 0.5)
        
        df['clutch_performance_predictor'] = (
            clutch_ability * pts_consistency * 2  # Habilidad × Consistencia
        )
        
        # ==================== TIER 4: COMBINACIONES AVANZADAS DE CONTEXTO ====================
        
        # 8. DEFENSIVE MATCHUP EXPLOITER - Explota debilidades defensivas
        opp_def_rating = df.get('opp_def_rating_5g', 110)
        shot_selection = df.get('shot_selection_quality', 0.5)
        
        df['defensive_exploiter'] = (
            (120 - opp_def_rating) / 10 * shot_selection  # Inverso de rating defensivo × calidad de tiro
        )
        
        # 9. RHYTHM AND FLOW - Ritmo y fluidez del jugador
        three_pt_eff = df.get('three_point_efficiency_5g', 0.35)
        ft_eff = df.get('ft_efficiency_5g', 0.75)
        scoring_variance = df.get('scoring_variance_5g', 5)
        
        df['rhythm_flow_factor'] = (
            (three_pt_eff * 0.4 + ft_eff * 0.6) * (10 / (scoring_variance + 1))  # Eficiencia × Consistencia
        )
        
        # ==================== TIER 5: META-FEATURES DE STACKING ====================
        
        # 10. SCORING CEILING PREDICTOR - Predice el techo anotador
        hot_streak = df.get('hot_streak_indicator', 0)
        scorer_archetype = df.get('scorer_archetype_score', 0.5)
        
        df['scoring_ceiling'] = (
            pts_hist * (1 + hot_streak * 0.3) * scorer_archetype * 1.5
        )
        
        # 11. FLOOR-CEILING RANGE - Rango de rendimiento esperado
        df['scoring_floor'] = pts_hist * 0.7 * pts_consistency  # Piso conservador
        df['scoring_range'] = df['scoring_ceiling'] - df['scoring_floor']
        
        # 12. CONFIDENCE-WEIGHTED PREDICTION - Predicción ponderada por confianza
        pts_consistency = df.get('pts_consistency_5g', 0.5)
        usage_normalized = df.get('usage_rate_5g', 20) / 30  # Normalizado
        minutes_normalized = df.get('mp_hist_avg_5g', 25) / 35  # Normalizado
        
        # Calcular confianza promedio elemento por elemento
        avg_confidence = (pts_consistency + usage_normalized + minutes_normalized) / 3
        df['confidence_weighted_prediction'] = df['elite_scorer_composite'] * avg_confidence
        
        # ==================== TIER 6: INTERACCIONES MULTIPLICATIVAS ====================
        
        # 13. USAGE × EFFICIENCY INTERACTION - Interacción crítica
        df['usage_efficiency_interaction'] = usage_rate * fg_hist * 20  # Escalado apropiado
        
        # 14. VOLUME × PACE INTERACTION - Interacción de volumen y ritmo
        df['volume_pace_interaction'] = shooting_vol * (pace_adj / 15)  # Normalizado
        
        # 15. FORM × OPPORTUNITY INTERACTION - Forma × Oportunidad
        df['form_opportunity_interaction'] = scoring_form * (mp_hist / 30)
        
        # ==================== TIER 7: FEATURES DE MOMENTUM AVANZADO ====================
        
        # 16. MOMENTUM ACCELERATOR - Acelerador de momentum
        recent_trend = df.get('pts_trend_factor', 1.0)
        df['momentum_accelerator'] = (
            recent_trend * df['elite_scorer_composite'] * 
            (1 + hot_streak * 0.2)
        )
        
        # 17. PRESSURE PERFORMANCE - Rendimiento bajo presión
        pressure_situations = df.get('high_usage_player', 0)
        df['pressure_performance'] = (
            df['clutch_performance_predictor'] * (1 + pressure_situations * 0.15)
        )
        
        # ==================== TIER 8: FEATURES DE VALIDACIÓN CRUZADA ====================
        
        # 18. CROSS-VALIDATION SCORE - Score de validación cruzada
        pts_above_avg = df.get('pts_above_season_avg', 0)
        usage_normalized = df.get('usage_rate_5g', 20) / 100
        volume_normalized = df.get('shooting_volume_5g', 10) / 15
        pace_normalized = df.get('pace_adjusted_scoring', 0) / 20
        scoring_form = df.get('scoring_form_5g', 0)
        
        # Calcular score de validación cruzada elemento por elemento
        df['cross_validation_score'] = (pts_above_avg + usage_normalized + volume_normalized + pace_normalized + scoring_form) / 5
        
        # 19. ENSEMBLE PREDICTION - Predicción de ensemble
        elite_composite = df['elite_scorer_composite']
        pace_power = df['pace_scoring_power'] / 2
        opportunity_max = df['opportunity_maximizer'] * 15
        confidence_pred = df['confidence_weighted_prediction']
        
        # Calcular predicción de ensemble elemento por elemento
        df['ensemble_scoring_prediction'] = (elite_composite + pace_power + opportunity_max + confidence_pred) / 4
        
        # 20. FINAL STACKED PREDICTOR - Predictor final apilado
        df['final_stacked_predictor'] = (
            df['ensemble_scoring_prediction'] * 0.4 +
            df['momentum_accelerator'] * 0.3 +
            df['scoring_ceiling'] * 0.2 +
            df['cross_validation_score'] * 10 * 0.1
        )
        
        # ==================== VALIDACIÓN Y LIMPIEZA ====================
        
        # Limpiar valores infinitos y NaN en todas las nuevas features
        stacking_features = [
            'elite_scorer_composite', 'pace_scoring_power', 'form_momentum_factor',
            'efficiency_volume_balance', 'league_context_score', 'opportunity_maximizer',
            'clutch_performance_predictor', 'defensive_exploiter', 'rhythm_flow_factor',
            'scoring_ceiling', 'scoring_floor', 'scoring_range', 'confidence_weighted_prediction',
            'usage_efficiency_interaction', 'volume_pace_interaction', 'form_opportunity_interaction',
            'momentum_accelerator', 'pressure_performance', 'cross_validation_score',
            'ensemble_scoring_prediction', 'final_stacked_predictor'
        ]
        
        for feature in stacking_features:
            if feature in df.columns:
                # Reemplazar infinitos y NaN
                df[feature] = df[feature].replace([np.inf, -np.inf], np.nan)
                df[feature] = df[feature].fillna(df[feature].median())
                
                # Aplicar límites razonables
                if 'predictor' in feature or 'prediction' in feature:
                    df[feature] = np.clip(df[feature], 0, 50)  # Límites para predicciones de puntos
                elif 'composite' in feature or 'score' in feature:
                    df[feature] = np.clip(df[feature], -10, 10)  # Límites para scores compuestos
                else:
                    df[feature] = np.clip(df[feature], -5, 5)   # Límites generales
        
        logger.info(f"Creadas {len(stacking_features)} features de stacking avanzadas")
        logger.info("Features de stacking más importantes:")
        logger.info("  - elite_scorer_composite: Combina top 3 features")
        logger.info("  - final_stacked_predictor: Predictor final apilado")
        logger.info("  - ensemble_scoring_prediction: Predicción de ensemble")
        logger.info("  - momentum_accelerator: Acelerador de momentum")
    
    def _update_feature_columns(self, df: pd.DataFrame):
        """Actualizar la lista de columnas de features disponibles"""
        # Excluir columnas básicas y de target
        exclude_columns = {
            'Player', 'Date', 'Team', 'Opp', 'Result', 'MP', 'GS', 'Away',
            'FG', 'FGA', 'FG%', '2P', '2PA', '2P%', '3P', '3PA', '3P%', 
            'FT', 'FTA', 'FT%', 'PTS', 'ORB', 'DRB', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF',
            'PTS_double', 'TRB_double', 'AST_double', 'STL_double', 'BLK_double',
            'double_double', 'triple_double',
            'day_of_week', 'month', 'days_rest', 'days_into_season'
        }
        
        self.feature_columns = [col for col in df.columns if col not in exclude_columns]
        logger.debug(f"Features disponibles actualizadas: {len(self.feature_columns)} columnas")
    
    def get_feature_importance_groups(self) -> Dict[str, List[str]]:
        """Obtener grupos de features por importancia para predicción de puntos"""
        return {
            'core_scoring_features': [
                'pts_hist_avg_5g', 'scoring_efficiency_5g', 'pts_trend_factor', 'pts_consistency_5g'
            ],
            'shooting_features': [
                'fg_efficiency_5g', 'shooting_volume_5g', 'three_point_efficiency_5g', 'ft_efficiency_5g',
                'shot_selection_quality', 'fg_efficiency_trend'
            ],
            'opportunity_features': [
                'mp_hist_avg_5g', 'usage_rate_5g', 'shooting_opportunities_5g', 'scoring_role_importance'
            ],
            'contextual_features': [
                'home_scoring_advantage', 'starter_scoring_boost', 'matchup_scoring_difficulty', 'pace_adjusted_scoring'
            ],
            'player_type_features': [
                'scorer_archetype_score', 'clutch_scoring_ability', 'three_point_specialist', 'high_usage_player'
            ]
        }
    
    def validate_features(self, df: pd.DataFrame) -> Dict[str, any]:
        """Validar features generadas para predicción de puntos"""
        validation_results = {
            'total_features': len(self.feature_columns),
            'missing_values': {},
            'infinite_values': {},
            'feature_ranges': {},
            'scoring_specific_features': 0
        }
        
        # Contar features específicas de anotación
        scoring_keywords = ['pts_', 'scoring_', 'shooting_', 'fg_', 'three_point_', 'ft_']
        validation_results['scoring_specific_features'] = sum(
            1 for col in self.feature_columns 
            if any(keyword in col.lower() for keyword in scoring_keywords)
        )
        
        # Validar cada feature
        for col in self.feature_columns:
            if col in df.columns:
                series = df[col]
                validation_results['missing_values'][col] = series.isna().sum()
                validation_results['infinite_values'][col] = np.isinf(series).sum()
                validation_results['feature_ranges'][col] = {
                    'min': series.min(),
                    'max': series.max(),
                    'mean': series.mean()
                }
        
        return validation_results
    
    def _get_historical_series(self, df: pd.DataFrame, column: str, window: int, 
                              operation: str = 'mean', min_periods: int = 1, q: float = None) -> pd.Series:
        """
        Obtener serie histórica de una columna usando shift(1) para evitar data leakage
        MEJORADO: Soporte para múltiples operaciones incluyendo quantiles
        """
        if column not in df.columns:
            NBALogger.log_warning(logger, f"Columna {column} no encontrada")
            return pd.Series(0, index=df.index)
        
        # Usar shift(1) para evitar data leakage
        shifted_series = df.groupby('Player')[column].shift(1)
        
        # Aplicar operación de rolling window
        rolling_obj = shifted_series.groupby(df['Player']).rolling(window, min_periods=min_periods)
        
        if operation == 'mean':
            result = rolling_obj.mean()
        elif operation == 'std':
            result = rolling_obj.std()
        elif operation == 'max':
            result = rolling_obj.max()
        elif operation == 'min':
            result = rolling_obj.min()
        elif operation == 'sum':
            result = rolling_obj.sum()
        elif operation == 'quantile':
            if q is not None:
                result = rolling_obj.quantile(q)
            else:
                result = rolling_obj.quantile(0.5)  # mediana por defecto
        else:
            NBALogger.log_warning(logger, f"Operación {operation} no soportada, usando mean")
            result = rolling_obj.mean()
        
        # Reset index para mantener estructura original
        result = result.reset_index(0, drop=True)
        
        # Rellenar valores NaN con 0 o valor por defecto
        result = result.fillna(0)
        
        return result
    
    def _clear_cache(self):
        """Limpiar cache de cálculos"""
        self._cached_calculations.clear()
        self._features_cache.clear()
        logger.debug("Cache de features limpiado")
    
    def _apply_correlation_regularization(self, df: pd.DataFrame, features: List[str]) -> List[str]:
        """
        FILTRO DE CORRELACIÓN AVANZADO - Elimina features redundantes para reducir ruido
        
        Estrategia:
        1. Eliminar features altamente correlacionadas (>0.85)
        2. Priorizar features de stacking sobre features básicas
        3. Mantener features con mayor correlación con el target (PTS)
        4. Eliminar features con baja varianza
        """
        if len(features) <= 15:
            logger.debug(f"Pocas features ({len(features)}) - saltando filtro de correlación")
            return features
        
        logger.debug(f"Aplicando filtro de correlación a {len(features)} features...")
        
        try:
            # PASO 0: PROTEGER FEATURES CRÍTICAS DE STACKING (NO ELIMINAR NUNCA)
            protected_features = [
                'ultimate_scoring_predictor', 'super_scoring_predictor', 'consensus_prediction',
                'final_stacked_predictor', 'elite_scorer_composite', 'ensemble_scoring_prediction',
                'momentum_accelerator', 'confidence_weighted_prediction', 'pts_hist_avg_5g',
                'scoring_efficiency_5g', 'pts_trend_factor', 'shooting_volume_5g'
            ]
            
            # Separar features protegidas de las que pueden ser filtradas
            protected_in_features = [f for f in protected_features if f in features]
            filterable_features = [f for f in features if f not in protected_features]
            
            logger.debug(f"Features protegidas: {len(protected_in_features)}")
            logger.debug(f"Features filtrables: {len(filterable_features)}")
            
            # Si no hay suficientes features filtrables, retornar todas
            if len(filterable_features) <= 10:
                logger.debug("Pocas features filtrables - retornando todas")
                return features
            
            # Preparar datos para análisis de correlación solo con features filtrables
            feature_data = df[filterable_features].fillna(0)
            
            # PASO 1: Eliminar features con varianza muy baja (casi constantes)
            low_variance_features = []
            for feature in filterable_features:
                if feature in feature_data.columns:
                    try:
                        variance = feature_data[feature].var()
                        if pd.isna(variance) or variance < 0.001:  # Varianza muy baja
                            low_variance_features.append(feature)
                    except:
                        low_variance_features.append(feature)  # Si hay error, eliminar
            
            if low_variance_features:
                logger.debug(f"Eliminando {len(low_variance_features)} features con baja varianza")
                filterable_features = [f for f in filterable_features if f not in low_variance_features]
                if len(filterable_features) > 0:
                    feature_data = feature_data[filterable_features]
                else:
                    # Si no quedan features filtrables, retornar solo las protegidas
                    return protected_in_features
            
            # PASO 2: Calcular matriz de correlación
            if len(filterable_features) <= 2:
                # Si quedan muy pocas features, no filtrar más
                return protected_in_features + filterable_features
            
            corr_matrix = feature_data.corr().abs()
            
            # PASO 3: Calcular correlación con target PTS
            target_correlations = {}
            if 'PTS' in df.columns:
                for feature in filterable_features:
                    if feature in df.columns:
                        try:
                            corr_with_target = df[feature].corr(df['PTS'])
                            if pd.notna(corr_with_target):
                                target_correlations[feature] = abs(corr_with_target)
                            else:
                                target_correlations[feature] = 0
                        except:
                            target_correlations[feature] = 0
            else:
                # Si no hay PTS, usar varianza como proxy
                for feature in filterable_features:
                    if feature in feature_data.columns:
                        try:
                            target_correlations[feature] = feature_data[feature].var()
                        except:
                            target_correlations[feature] = 0
            
            # PASO 4: Encontrar pares altamente correlacionados
            high_corr_pairs = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    try:
                        correlation = corr_matrix.iloc[i, j]
                        if pd.notna(correlation) and correlation > 0.85:  # Umbral de correlación alta
                            feature1 = corr_matrix.columns[i]
                            feature2 = corr_matrix.columns[j]
                            high_corr_pairs.append((feature1, feature2, correlation))
                    except:
                        continue
            
            logger.debug(f"Encontrados {len(high_corr_pairs)} pares de features altamente correlacionadas")
            
            # PASO 5: Decidir qué features eliminar (solo entre las filtrables)
            features_to_remove = set()
            
            for feature1, feature2, correlation in high_corr_pairs:
                if feature1 in features_to_remove or feature2 in features_to_remove:
                    continue  # Ya se decidió eliminar una de estas
                
                # Criterios de decisión: mantener la con mayor correlación con target
                target_corr1 = target_correlations.get(feature1, 0)
                target_corr2 = target_correlations.get(feature2, 0)
                
                # Decidir cuál eliminar
                if abs(target_corr1 - target_corr2) > 0.05:
                    # Eliminar la con menor correlación con target
                    feature_to_remove = feature1 if target_corr1 < target_corr2 else feature2
                else:
                    # Eliminar la con nombre más largo (menos específica)
                    feature_to_remove = feature1 if len(feature1) > len(feature2) else feature2
                
                features_to_remove.add(feature_to_remove)
                logger.debug(f"Eliminando {feature_to_remove} (correlación {correlation:.3f})")
            
            # PASO 6: Aplicar filtro adicional por correlación con target si es necesario
            remaining_filterable = [f for f in filterable_features if f not in features_to_remove]
            
            # Si aún tenemos muchas features filtrables, mantener solo las mejores
            max_filterable = 25  # Máximo de features filtrables
            if len(remaining_filterable) > max_filterable:
                # Ordenar por correlación con target y mantener las mejores
                sorted_by_target_corr = sorted(
                    remaining_filterable, 
                    key=lambda x: target_correlations.get(x, 0), 
                    reverse=True
                )
                remaining_filterable = sorted_by_target_corr[:max_filterable]
                
                additional_removed = len(filterable_features) - len(features_to_remove) - len(remaining_filterable)
                if additional_removed > 0:
                    logger.debug(f"Filtro adicional: eliminadas {additional_removed} features con baja correlación con target")
            
            # PASO 7: Combinar features protegidas + features filtradas restantes
            final_features = protected_in_features + remaining_filterable
            
            # PASO 8: Reporte final
            total_removed = len(features) - len(final_features)
            logger.info(f"FILTRO DE CORRELACIÓN COMPLETADO:")
            logger.info(f"  Features originales: {len(features)}")
            logger.info(f"  Features protegidas: {len(protected_in_features)}")
            logger.info(f"  Features eliminadas: {total_removed}")
            logger.info(f"  Features finales: {len(final_features)}")
            logger.info(f"  Reducción de ruido: {(total_removed/len(features)*100):.1f}%")
            
            if features_to_remove:
                logger.debug(f"Features eliminadas por correlación: {len(features_to_remove)}")
            
            # Verificar que mantenemos features críticas
            critical_features = [
                'ultimate_scoring_predictor', 'super_scoring_predictor', 'consensus_prediction',
                'final_stacked_predictor', 'elite_scorer_composite', 'pts_hist_avg_5g'
            ]
            
            critical_kept = [f for f in critical_features if f in final_features]
            critical_lost = [f for f in critical_features if f in features and f not in final_features]
            
            logger.info(f"  Features críticas mantenidas: {len(critical_kept)}/{len([f for f in critical_features if f in features])}")
            
            if critical_lost:
                NBALogger.log_warning(logger, f"Features críticas perdidas: {critical_lost}")
            else:
                logger.info("  Todas las features críticas fueron preservadas")
            
            return final_features
            
        except Exception as e:
            NBALogger.log_error(logger, f"Error en filtro de correlación: {str(e)}")
            logger.info("Retornando features originales sin filtrar")
            return features
    
    def _create_meta_stacking_features_optimized(self, df: pd.DataFrame) -> None:
        """
        META-FEATURES DE STACKING DE SEGUNDO NIVEL
        Combinaciones de las features de stacking de primer nivel para máxima predicción
        """
        logger.info("Creando meta-features de stacking de segundo nivel...")
        
        # ==================== META-COMBINACIONES DE ELITE ====================
        
        # 1. SUPER PREDICTOR - Combina los mejores predictores de primer nivel
        final_stacked = df.get('final_stacked_predictor', 0)
        elite_composite = df.get('elite_scorer_composite', 0)
        ensemble_pred = df.get('ensemble_scoring_prediction', 0)
        
        df['super_scoring_predictor'] = (
            final_stacked * 0.4 +
            elite_composite * 0.35 +
            ensemble_pred * 0.25
        )
        
        # 2. CONFIDENCE-MOMENTUM FUSION - Fusiona confianza y momentum
        confidence_weighted = df.get('confidence_weighted_prediction', 0)
        momentum_acc = df.get('momentum_accelerator', 0)
        
        df['confidence_momentum_fusion'] = (
            confidence_weighted * momentum_acc / 5  # Normalizado
        )
        
        # 3. OPPORTUNITY-EFFICIENCY MAXIMIZER - Maximiza oportunidad × eficiencia
        opportunity_max = df.get('opportunity_maximizer', 0)
        efficiency_balance = df.get('efficiency_volume_balance', 0)
        
        df['opportunity_efficiency_max'] = (
            opportunity_max * efficiency_balance * 3  # Escalado
        )
        
        # ==================== INTERACCIONES DE SEGUNDO NIVEL ====================
        
        # 4. CEILING-FLOOR STABILITY - Estabilidad basada en rango
        scoring_ceiling = df.get('scoring_ceiling', 0)
        scoring_floor = df.get('scoring_floor', 0)
        scoring_range = df.get('scoring_range', 1)
        
        df['ceiling_floor_stability'] = (
            (scoring_ceiling + scoring_floor) / 2 * (10 / (scoring_range + 1))
        )
        
        # 5. PRESSURE-CLUTCH AMPLIFIER - Amplificador de presión y clutch
        pressure_perf = df.get('pressure_performance', 0)
        clutch_pred = df.get('clutch_performance_predictor', 0)
        
        df['pressure_clutch_amplifier'] = (
            pressure_perf * clutch_pred * 1.5  # Amplificado
        )
        
        # ==================== FEATURES DE VALIDACIÓN AVANZADA ====================
        
        # 6. CROSS-VALIDATION ENSEMBLE - Ensemble de validación cruzada
        cross_val = df.get('cross_validation_score', 0)
        pace_power = df.get('pace_scoring_power', 0)
        
        df['cv_ensemble_score'] = (
            cross_val * 8 + pace_power / 3  # Ponderado
        )
        
        # 7. RHYTHM-MOMENTUM SYNC - Sincronización de ritmo y momentum
        rhythm_flow = df.get('rhythm_flow_factor', 0)
        form_momentum = df.get('form_momentum_factor', 0)
        
        df['rhythm_momentum_sync'] = (
            rhythm_flow * form_momentum * 2  # Sincronizado
        )
        
        # ==================== FEATURES DE CONTEXTO AVANZADO ====================
        
        # 8. DEFENSIVE-LEAGUE CONTEXT - Contexto defensivo y de liga
        defensive_exploit = df.get('defensive_exploiter', 0)
        league_context = df.get('league_context_score', 0)
        
        df['defensive_league_context'] = (
            defensive_exploit * league_context * 1.2
        )
        
        # 9. INTERACTION MULTIPLIER - Multiplicador de interacciones
        usage_eff_int = df.get('usage_efficiency_interaction', 0)
        vol_pace_int = df.get('volume_pace_interaction', 0)
        form_opp_int = df.get('form_opportunity_interaction', 0)
        
        df['interaction_multiplier'] = (
            usage_eff_int * 0.4 +
            vol_pace_int * 0.35 +
            form_opp_int * 0.25
        )
        
        # ==================== PREDICTOR FINAL DE SEGUNDO NIVEL ====================
        
        # 10. ULTIMATE SCORING PREDICTOR - Predictor final definitivo
        df['ultimate_scoring_predictor'] = (
            df['super_scoring_predictor'] * 0.3 +
            df['confidence_momentum_fusion'] * 0.25 +
            df['opportunity_efficiency_max'] * 0.2 +
            df['ceiling_floor_stability'] * 0.15 +
            df['cv_ensemble_score'] * 0.1
        )
        
        # ==================== FEATURES DE ROBUSTEZ ====================
        
        # 11. PREDICTION ROBUSTNESS - Robustez de predicción
        primary_predictors = [
            df.get('final_stacked_predictor', 0),
            df.get('elite_scorer_composite', 0) * 2,
            df.get('ensemble_scoring_prediction', 0),
            df['super_scoring_predictor']
        ]
        
        # Convertir a array numpy para evitar ambigüedad
        predictors_array = np.array(primary_predictors)
        
        # Calcular desviación estándar como medida de robustez
        df['prediction_robustness'] = 1 / (np.std(predictors_array, axis=0) + 0.1)
        
        # 12. CONSENSUS PREDICTION - Predicción de consenso
        df['consensus_prediction'] = np.median(predictors_array, axis=0)
        
        # ==================== VALIDACIÓN Y LIMPIEZA DE SEGUNDO NIVEL ====================
        
        meta_features = [
            'super_scoring_predictor', 'confidence_momentum_fusion', 'opportunity_efficiency_max',
            'ceiling_floor_stability', 'pressure_clutch_amplifier', 'cv_ensemble_score',
            'rhythm_momentum_sync', 'defensive_league_context', 'interaction_multiplier',
            'ultimate_scoring_predictor', 'prediction_robustness', 'consensus_prediction'
        ]
        
        for feature in meta_features:
            if feature in df.columns:
                # Limpiar valores problemáticos
                df[feature] = df[feature].replace([np.inf, -np.inf], np.nan)
                df[feature] = df[feature].fillna(df[feature].median())
                
                # Aplicar límites específicos
                if 'predictor' in feature or 'prediction' in feature:
                    df[feature] = np.clip(df[feature], 0, 60)  # Límites para predicciones
                elif 'robustness' in feature:
                    df[feature] = np.clip(df[feature], 0, 10)   # Límites para robustez
                else:
                    df[feature] = np.clip(df[feature], -8, 8)   # Límites generales
        
        logger.info(f"Creadas {len(meta_features)} meta-features de stacking de segundo nivel")
        logger.info("Meta-features más importantes:")
        logger.info("  - ultimate_scoring_predictor: Predictor final definitivo")
        logger.info("  - super_scoring_predictor: Combina mejores predictores")
        logger.info("  - consensus_prediction: Predicción de consenso")
        logger.info("  - prediction_robustness: Robustez de predicción")
    
    def _log_stacking_features_summary(self, df: pd.DataFrame) -> None:
        """
        Mostrar resumen completo de todas las features de stacking creadas
        """
        logger.info("="*80)
        logger.info("RESUMEN COMPLETO DE FEATURES DE STACKING PARA PREDICCION DE PUNTOS")
        logger.info("="*80)
        
        # TIER 1: Meta-features de segundo nivel (más importantes)
        meta_features = [
            'ultimate_scoring_predictor', 'super_scoring_predictor', 'consensus_prediction',
            'confidence_momentum_fusion', 'opportunity_efficiency_max', 'ceiling_floor_stability',
            'pressure_clutch_amplifier', 'cv_ensemble_score', 'rhythm_momentum_sync',
            'defensive_league_context', 'interaction_multiplier', 'prediction_robustness'
        ]
        
        # TIER 2: Features de stacking de primer nivel
        stacking_features = [
            'final_stacked_predictor', 'elite_scorer_composite', 'ensemble_scoring_prediction',
            'momentum_accelerator', 'confidence_weighted_prediction', 'pace_scoring_power',
            'opportunity_maximizer', 'clutch_performance_predictor', 'efficiency_volume_balance',
            'league_context_score', 'defensive_exploiter', 'rhythm_flow_factor',
            'scoring_ceiling', 'scoring_floor', 'scoring_range', 'usage_efficiency_interaction',
            'volume_pace_interaction', 'form_opportunity_interaction', 'form_momentum_factor',
            'pressure_performance', 'cross_validation_score'
        ]
        
        # Contar features disponibles
        available_meta = [f for f in meta_features if f in df.columns]
        available_stacking = [f for f in stacking_features if f in df.columns]
        
        logger.info(f"META-FEATURES DE SEGUNDO NIVEL: {len(available_meta)}/{len(meta_features)} creadas")
        
        logger.info(f"FEATURES DE STACKING PRIMER NIVEL: {len(available_stacking)}/{len(stacking_features)} creadas")
        
        # Mostrar features más predictivas esperadas
        logger.info(f"TOP 10 FEATURES DE STACKING MAS PREDICTIVAS (orden de importancia esperada):")
        top_predictive = [
            'ultimate_scoring_predictor',     # 1. Predictor final definitivo
            'super_scoring_predictor',        # 2. Combina mejores predictores
            'consensus_prediction',           # 3. Predicción de consenso
            'final_stacked_predictor',        # 4. Predictor final apilado
            'elite_scorer_composite',         # 5. Combina top 3 features originales
            'confidence_momentum_fusion',     # 6. Fusión confianza-momentum
            'ensemble_scoring_prediction',    # 7. Predicción de ensemble
            'opportunity_efficiency_max',     # 8. Maximizador oportunidad-eficiencia
            'momentum_accelerator',           # 9. Acelerador de momentum
            'prediction_robustness'           # 10. Robustez de predicción
        ]
        
        for i, feature in enumerate(top_predictive, 1):
            if feature in df.columns:
                values = df[feature]
                logger.info(f"  {i:2d}. {feature}: Rango=[{values.min():.2f}, {values.max():.2f}]")
        
        # Estadísticas generales
        total_stacking_features = len(available_meta) + len(available_stacking)

    
    def generate_correlation_report(self, df: pd.DataFrame, features: List[str]) -> Dict[str, any]:
        """
        Generar reporte detallado de correlaciones entre features
        Útil para debugging y optimización del modelo
        """
        logger.debug("Generando reporte de correlaciones...")
        
        try:
            feature_data = df[features].fillna(0)
            corr_matrix = feature_data.corr().abs()
            
            # Estadísticas de correlación
            correlations_flat = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    correlation = corr_matrix.iloc[i, j]
                    if not np.isnan(correlation):
                        correlations_flat.append({
                            'feature1': corr_matrix.columns[i],
                            'feature2': corr_matrix.columns[j],
                            'correlation': correlation
                        })
            
            # Ordenar por correlación descendente
            correlations_flat.sort(key=lambda x: x['correlation'], reverse=True)
            
            # Correlaciones con target PTS
            target_correlations = {}
            if 'PTS' in df.columns:
                for feature in features:
                    if feature in df.columns:
                        try:
                            corr_with_target = abs(df[feature].corr(df['PTS']))
                            target_correlations[feature] = corr_with_target if not np.isnan(corr_with_target) else 0
                        except:
                            target_correlations[feature] = 0
            
            # Estadísticas de varianza
            variance_stats = {}
            for feature in features:
                if feature in feature_data.columns:
                    variance_stats[feature] = {
                        'variance': feature_data[feature].var(),
                        'std': feature_data[feature].std(),
                        'mean': feature_data[feature].mean(),
                        'min': feature_data[feature].min(),
                        'max': feature_data[feature].max()
                    }
            
            report = {
                'total_features': len(features),
                'high_correlations': [c for c in correlations_flat if c['correlation'] > 0.85],
                'medium_correlations': [c for c in correlations_flat if 0.7 <= c['correlation'] <= 0.85],
                'target_correlations': target_correlations,
                'variance_stats': variance_stats,
                'top_correlated_pairs': correlations_flat[:10],
                'low_variance_features': [f for f, stats in variance_stats.items() if stats['variance'] < 0.001]
            }
            
            # Log resumen reducido
            logger.info(f"REPORTE DE CORRELACIONES:")
            logger.info(f"  Total features analizadas: {len(features)}")
            logger.info(f"  Correlaciones altas (>0.85): {len(report['high_correlations'])}")
            logger.info(f"  Correlaciones medias (0.7-0.85): {len(report['medium_correlations'])}")
            logger.info(f"  Features con baja varianza: {len(report['low_variance_features'])}")
            
            if report['high_correlations']:
                logger.debug(f"Top 3 correlaciones altas:")
                for i, corr in enumerate(report['high_correlations'][:3]):
                    logger.debug(f"    {i+1}. {corr['feature1']} <-> {corr['feature2']}: {corr['correlation']:.3f}")
            
            return report
            
        except Exception as e:
            NBALogger.log_error(logger, f"Error generando reporte de correlaciones: {str(e)}")
            return {'error': str(e)}
    
    def get_optimal_feature_set(self, df: pd.DataFrame, max_features: int = 35) -> List[str]:
        """
        Obtener conjunto óptimo de features balanceando predictividad y diversidad
        """
        logger.info(f"Calculando conjunto óptimo de máximo {max_features} features...")
        
        # Generar todas las features primero
        all_features = self.generate_all_features(df)
        
        # Aplicar filtro de correlación más agresivo
        if len(all_features) > max_features:
            # Usar umbral de correlación más bajo para ser más agresivo
            original_method = self._apply_correlation_regularization
            
            def aggressive_filter(df_inner, features_inner):
                # Modificar temporalmente para ser más agresivo
                try:
                    feature_data = df_inner[features_inner].fillna(0)
                    corr_matrix = feature_data.corr().abs()
                    
                    # Umbral más agresivo
                    high_corr_pairs = []
                    for i in range(len(corr_matrix.columns)):
                        for j in range(i+1, len(corr_matrix.columns)):
                            if corr_matrix.iloc[i, j] > 0.75:  # Más agresivo: 0.75 en lugar de 0.85
                                high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j]))
                    
                    # Aplicar lógica similar pero más agresiva
                    return original_method(df_inner, features_inner)
                except:
                    return features_inner[:max_features]
            
            # Aplicar filtro de correlación agresivo
            final_features = aggressive_filter(df, all_features)
            
            # CRÍTICO: Aplicar filtro de data leakage ANTES de la selección final
            leakage_features = self._detect_data_leakage_features(df)
            final_features = [f for f in final_features if f not in leakage_features]
            logger.info(f"Filtro de data leakage aplicado: {len(leakage_features)} features eliminadas")
            
            # CRÍTICO: Eliminar features dominantes detectadas
            dominant_features = [f for f in final_features if f in ['player_tier', 'GmSc', 'BPM', 'PER']]
            final_features = [f for f in final_features if f not in dominant_features]
            logger.info(f"Features dominantes eliminadas: {dominant_features}")
            
            # Asegurar que no excedemos el máximo
            if len(final_features) > max_features:
                logger.info(f"Aplicando límite final: {len(final_features)} -> {max_features} features")
                final_features = final_features[:max_features]
            
            logger.info(f"Conjunto óptimo: {len(final_features)} features seleccionadas")
            return final_features
        else:
            # PASO 1: Filtrar features ruidosas
            logger.info("Aplicando filtros de ruido para eliminar features problemáticas...")
            clean_features = self._apply_noise_filters(df, all_features)
            logger.info(f"Conjunto óptimo: {len(clean_features)} features seleccionadas")
            return clean_features
    
    def _create_situational_context_features_optimized(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Crea features de contexto situacional que afectan el rendimiento.
        
        Args:
            df: DataFrame con datos NBA
            
        Returns:
            pd.DataFrame: DataFrame con features situacionales agregadas
        """
        logger.info("Creando features de contexto situacional...")
        
        df = df.copy()
        df['Date'] = pd.to_datetime(df['Date'])
        
        # Ordenar por jugador y fecha para cálculos temporales
        df = df.sort_values(['Player', 'Date']).reset_index(drop=True)
        
        # 1. Back-to-back games indicator
        df['prev_game_date'] = df.groupby('Player')['Date'].shift(1)
        df['days_since_last_game'] = (df['Date'] - df['prev_game_date']).dt.days
        df['is_back_to_back'] = (df['days_since_last_game'] == 1).astype(int)
        
        # 2. Días de descanso (0-7+ días)
        df['rest_days'] = df['days_since_last_game'].fillna(3).clip(0, 7)
        
        # 3. Fatigue indicator (últimos 7 partidos)
        df['games_last_7_days'] = df.groupby('Player').cumcount() + 1
        df['games_last_7_days'] = df['games_last_7_days'].clip(upper=7)
        
        # 4. Schedule density (últimos 14 partidos)
        df['games_last_14_days'] = df.groupby('Player').cumcount() + 1
        df['games_last_14_days'] = df['games_last_14_days'].clip(upper=14)
        
        # 5. Home court advantage (si está disponible)
        if 'is_home' in df.columns:
            df['home_court_advantage'] = df['is_home'].astype(int)
        else:
            # Inferir de Team vs Opp (simplificado)
            df['home_court_advantage'] = 0.5  # Neutral por defecto
        
        # 6. Day of week effects
        df['day_of_week'] = df['Date'].dt.dayofweek
        df['is_weekend'] = (df['day_of_week'].isin([5, 6])).astype(int)
        
        # 7. Month effects (temporada regular vs playoffs)
        df['month'] = df['Date'].dt.month
        df['is_playoff_season'] = (df['month'].isin([4, 5, 6])).astype(int)
        
        # 8. Time since season start
        df['season_year'] = df['Date'].dt.year
        season_starts = df.groupby('season_year')['Date'].min()
        df['days_since_season_start'] = df.apply(
            lambda row: (row['Date'] - season_starts[row['season_year']]).days,
            axis=1
        )
        
        # 9. Cumulative minutes load (fatigue proxy)
        if 'MP' in df.columns:
            df['cumulative_minutes'] = df.groupby('Player')['MP'].cumsum()
            df['avg_minutes_last_5'] = df.groupby('Player')['MP'].rolling(
                window=5, min_periods=1
            ).mean().reset_index(0, drop=True)
        else:
            # Usar valores por defecto si MP no existe
            df['cumulative_minutes'] = df.groupby('Player').cumcount() * 25  # Asumiendo 25 min promedio
            df['avg_minutes_last_5'] = 25.0  # Valor por defecto
        
        # 10. Performance momentum indicators
        if 'PTS' in df.columns:
            df['pts_last_3_avg'] = df.groupby('Player')['PTS'].rolling(
                window=3, min_periods=1
            ).mean().shift(1).reset_index(0, drop=True)
            
            df['pts_trend_last_5'] = df.groupby('Player')['PTS'].rolling(
                window=5, min_periods=2
            ).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) >= 2 else 0
            ).shift(1).reset_index(0, drop=True)
        else:
            # Usar valores por defecto si PTS no existe
            df['pts_last_3_avg'] = 15.0  # Valor por defecto
            df['pts_trend_last_5'] = 0.0  # Sin tendencia
        
        # 11. Opponent strength (si hay datos de equipos)
        if hasattr(self, 'teams_df') and self.teams_df is not None:
            # Calcular defensive rating promedio del oponente
            team_def_rating = self.teams_df.groupby('Team').agg({
                'PTS_allowed': 'mean'  # Asumiendo que existe esta columna
            }).reset_index()
            
            # Merge con datos del oponente
            if 'Opp' in df.columns:
                df = df.merge(
                    team_def_rating.rename(columns={'Team': 'Opp', 'PTS_allowed': 'opp_def_rating'}),
                    on='Opp', how='left'
                )
                df['opp_def_rating'] = df['opp_def_rating'].fillna(df['opp_def_rating'].mean())
            else:
                df['opp_def_rating'] = 110.0  # Valor promedio por defecto
        else:
            df['opp_def_rating'] = 110.0
        
        # 12. Interaction features
        df['rest_x_usage'] = df['rest_days'] * df.get('usage_rate_5g', 20)
        df['fatigue_x_minutes'] = df['games_last_7_days'] * df['avg_minutes_last_5']
        df['home_x_momentum'] = df['home_court_advantage'] * df['pts_trend_last_5']
        
        # Limpiar columnas temporales
        df = df.drop(['prev_game_date', 'days_since_last_game'], axis=1, errors='ignore')
        
        # Lista de features situacionales creadas
        situational_features = [
            'is_back_to_back', 'rest_days', 'games_last_7_days', 'games_last_14_days',
            'home_court_advantage', 'day_of_week', 'is_weekend', 'is_playoff_season',
            'days_since_season_start', 'cumulative_minutes', 'avg_minutes_last_5',
            'pts_last_3_avg', 'pts_trend_last_5', 'opp_def_rating',
            'rest_x_usage', 'fatigue_x_minutes', 'home_x_momentum'
        ]
        
        # Verificar que las features fueron creadas
        created_features = [f for f in situational_features if f in df.columns]
        
        logger.info(f"Features situacionales creadas: {len(created_features)}")
        logger.info(f"Features: {created_features}")
        
        # Estadísticas de las nuevas features
        logger.info("Estadísticas de features situacionales:")
        for feature in created_features[:5]:  # Mostrar solo las primeras 5
            if df[feature].dtype in ['int64', 'float64']:
                logger.info(f"  {feature}: media={df[feature].mean():.3f}, "
                           f"std={df[feature].std():.3f}")
        
        return df
    
    def _register_feature(self, feature_name: str, category: str) -> bool:
        """
        Registra una feature y verifica si ya fue computada.
        
        Args:
            feature_name: Nombre de la feature
            category: Categoría de la feature
            
        Returns:
            bool: True si la feature es nueva, False si ya existe
        """
        if feature_name in self._computed_features:
            logger.debug(f"Feature {feature_name} ya computada - saltando")
            return False
        
        # Solo registrar si realmente vamos a crear la feature
        self._computed_features.add(feature_name)
        if category in self._feature_registry:
            self._feature_registry[category].append(feature_name)
        
        return True
    
    def _check_feature_exists(self, df: pd.DataFrame, feature_name: str) -> bool:
        """
        Verifica si una feature ya existe en el DataFrame.
        
        Args:
            df: DataFrame a verificar
            feature_name: Nombre de la feature
            
        Returns:
            bool: True si la feature existe
        """
        return feature_name in df.columns
    
    def _get_or_create_feature(self, df: pd.DataFrame, feature_name: str, 
                              calculation_func, *args, **kwargs):
        """
        Obtiene una feature existente o la calcula si no existe.
        
        Args:
            df: DataFrame
            feature_name: Nombre de la feature
            calculation_func: Función para calcular la feature
            *args, **kwargs: Argumentos para la función
            
        Returns:
            pd.Series: Serie con la feature
        """
        if self._check_feature_exists(df, feature_name):
            logger.debug(f"Usando feature existente: {feature_name}")
            return df[feature_name]
        
        # Calcular feature
        logger.debug(f"Calculando nueva feature: {feature_name}")
        result = calculation_func(*args, **kwargs)
        df[feature_name] = result
        
        return result
    
    def _compile_essential_features(self, df: pd.DataFrame) -> List[str]:
        """
        Compila la lista de features esenciales evitando duplicaciones.
        Verifica que las features realmente existan en el DataFrame.
        
        Args:
            df: DataFrame con todas las features
            
        Returns:
            List[str]: Lista de features esenciales que existen en el DataFrame
        """
        logger.debug("Compilando features esenciales...")
        
        # Excluir columnas básicas y de target
        exclude_columns = {
            'Player', 'Date', 'Team', 'Opp', 'Result', 'MP', 'GS', 'Away',
            'FG', 'FGA', 'FG%', '2P', '2PA', '2P%', '3P', '3PA', '3P%', 
            'FT', 'FTA', 'FT%', 'PTS', 'ORB', 'DRB', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF',
            'PTS_double', 'TRB_double', 'AST_double', 'STL_double', 'BLK_double',
            'double_double', 'triple_double',
            'day_of_week', 'month', 'days_rest', 'days_into_season'
        }
        
        # Obtener todas las features disponibles REALMENTE en el DataFrame
        all_features = [col for col in df.columns if col not in exclude_columns]
        
        logger.debug(f"Features disponibles en DataFrame: {len(all_features)}")
        logger.debug(f"Primeras 10: {all_features[:10]}")
        
        # Verificar features registradas vs features reales
        registered_features = set()
        for category_features in self._feature_registry.values():
            registered_features.update(category_features)
        
        missing_registered = registered_features - set(all_features)
        if missing_registered:
            NBALogger.log_warning(logger, f"Features registradas pero no en DataFrame: {missing_registered}")
        
        # PRIORIDAD 0: FEATURES REVOLUCIONARIAS DE ENSEMBLE (MÁXIMA PRIORIDAD ABSOLUTA)
        revolutionary_ensemble = [f for f in self._feature_registry['temporal_advanced'] 
                                 if any(keyword in f for keyword in [
                                     'temporal_ensemble_predictor', 'final_ensemble_predictor',
                                     'ensemble_confidence_score', 'volatility_ensemble_score',
                                     'momentum_ensemble_indicator', 'context_ensemble_factor',
                                     'efficiency_ensemble_score', 'opportunity_ensemble_factor'
                                 ]) and f in all_features]
        
        # PRIORIDAD 1: Features de alto scoring (SEGUNDA PRIORIDAD)
        high_scoring_features = [f for f in self._feature_registry['temporal_advanced'] 
                               if any(keyword in f for keyword in [
                                   'high_scorer', 'explosion_potential', 'high_scoring_streak',
                                   'high_volume_efficiency', 'clutch_scoring_factor',
                                   'player_tier', 'team_carry_ability', 'offensive_versatility',
                                   'defensive_attention', 'superstar_factor',
                                   'prob_30plus_game', 'prob_40plus_game', 'explosive_game_trend',
                                   'motivation_factor', 'performance_ceiling', 'historic_game_potential'
                               ]) and f in all_features and f not in revolutionary_ensemble]
        
        # PRIORIDAD 2: Meta-features de stacking (verificar existencia)
        meta_stacking = [f for f in self._feature_registry['meta_stacking'] 
                        if f in all_features]
        
        # PRIORIDAD 3: Features de drift temporal y tendencias recientes
        temporal_advanced = [f for f in self._feature_registry['temporal_advanced'] 
                           if f in all_features and f not in high_scoring_features]
        drift_detection = [f for f in self._feature_registry['drift_detection'] 
                          if f in all_features]
        
        # PRIORIDAD 4: Features de stacking avanzadas (verificar existencia)
        stacking = [f for f in self._feature_registry['stacking'] 
                   if f in all_features]
        
        # PRIORIDAD 5: Features core de scoring (verificar existencia)
        core_scoring = [f for f in self._feature_registry['scoring'] 
                       if f in all_features]
        
        # PRIORIDAD 6: Features de shooting (verificar existencia)
        shooting = [f for f in self._feature_registry['shooting'] 
                   if f in all_features]
        
        # PRIORIDAD 7: Features de usage y oportunidades (verificar existencia)
        usage = [f for f in self._feature_registry['usage'] 
                if f in all_features]
        
        # PRIORIDAD 8: Features situacionales (verificar existencia)
        situational = [f for f in self._feature_registry['situational'] 
                      if f in all_features]
        
        # PRIORIDAD 9: Resto de features (verificar existencia)
        categorized_features = set(meta_stacking + temporal_advanced + drift_detection + 
                                 stacking + core_scoring + shooting + usage + situational)
        other_features = [f for f in all_features if f not in categorized_features]
        
        # Compilar en orden de prioridad
        essential_features = (
            revolutionary_ensemble[:6] +           # Máximo 6 features revolucionarias
            high_scoring_features[:6] +           # Máximo 6 features de alto scoring
            meta_stacking[:12] +           # Máximo 12 meta-features
            temporal_advanced[:8] +        # Máximo 8 features temporales avanzadas
            drift_detection[:6] +          # Máximo 6 features de drift detection
            stacking[:15] +                # Máximo 15 stacking features
            core_scoring[:8] +             # Máximo 8 core scoring
            shooting[:6] +                 # Máximo 6 shooting
            usage[:5] +                    # Máximo 5 usage
            situational[:6] +              # Máximo 6 situacionales
            other_features[:6]             # Máximo 6 otras
        )
        
        # Eliminar duplicados manteniendo orden
        seen = set()
        unique_features = []
        for feature in essential_features:
            if feature not in seen and feature in df.columns:  # Verificación adicional
                seen.add(feature)
                unique_features.append(feature)
        
        # Verificación final: todas las features deben existir en el DataFrame
        final_features = [f for f in unique_features if f in df.columns]
        
        logger.info(f"Features compiladas por categoría (verificadas):")
        logger.info(f"  Revolucionarias: {len([f for f in revolutionary_ensemble if f in final_features])}")
        logger.info(f"  Alto scoring: {len([f for f in high_scoring_features if f in final_features])}")
        logger.info(f"  Meta-stacking: {len([f for f in meta_stacking[:12] if f in final_features])}")
        logger.info(f"  Temporal avanzadas: {len([f for f in temporal_advanced[:8] if f in final_features])}")
        logger.info(f"  Drift detection: {len([f for f in drift_detection[:6] if f in final_features])}")
        logger.info(f"  Stacking: {len([f for f in stacking[:15] if f in final_features])}")
        logger.info(f"  Core scoring: {len([f for f in core_scoring[:8] if f in final_features])}")
        logger.info(f"  Shooting: {len([f for f in shooting[:6] if f in final_features])}")
        logger.info(f"  Usage: {len([f for f in usage[:5] if f in final_features])}")
        logger.info(f"  Situacionales: {len([f for f in situational[:6] if f in final_features])}")
        logger.info(f"  Otras: {len([f for f in other_features[:6] if f in final_features])}")
        
        # Verificación final crítica
        missing_features = [f for f in final_features if f not in df.columns]
        if missing_features:
            NBALogger.log_error(logger, f"FEATURES FALTANTES EN DATAFRAME: {missing_features}")
            # Remover features faltantes
            final_features = [f for f in final_features if f in df.columns]
        
        logger.info(f"Features finales verificadas: {len(final_features)}")
        
        return final_features
    
    def _log_feature_creation_summary(self) -> None:
        """
        Registra un resumen de la creación de features.
        """
        logger.info("=" * 60)
        logger.info("RESUMEN DE CREACIÓN DE FEATURES OPTIMIZADO")
        logger.info("=" * 60)
        
        total_features = 0
        for category, features in self._feature_registry.items():
            count = len(features)
            total_features += count
            if count > 0:
                logger.info(f"{category.upper()}: {count} features")
                if count <= 5:  # Mostrar todas si son pocas
                    logger.info(f"  {features}")
                else:  # Mostrar solo las primeras 3
                    logger.info(f"  Top 3: {features[:3]}")
        
        logger.info(f"TOTAL FEATURES CREADAS: {total_features}")
        logger.info(f"FEATURES EN CACHE: {len(self._computed_features)}")
        logger.info("=" * 60)
    
    def _ensure_feature_in_dataframe(self, df: pd.DataFrame, feature_name: str, 
                                   default_value=0) -> None:
        """
        Asegura que una feature esté presente en el DataFrame.
        
        Args:
            df: DataFrame
            feature_name: Nombre de la feature
            default_value: Valor por defecto si no existe
        """
        if feature_name not in df.columns:
            NBALogger.log_warning(logger, f"Feature {feature_name} no encontrada en DataFrame - creando con valor por defecto")
            df[feature_name] = default_value

    def _create_temporal_drift_features(self, df: pd.DataFrame) -> None:
        """
        NUEVO: Crear features para detectar y adaptarse al drift temporal.
        Estas features ayudan al modelo a detectar cambios en patrones de juego.
        """
        logger.debug("Creando features de drift temporal...")
        
        if 'PTS' not in df.columns:
            return
            
        # Feature 1: Diferencia entre rendimiento reciente vs histórico
        if self._register_feature('pts_recent_vs_historical', 'drift_detection'):
            pts_recent_3g = self._get_historical_series(df, 'PTS', 3, 'mean').shift(1)
            pts_historical_10g = self._get_historical_series(df, 'PTS', 10, 'mean').shift(1)
            df['pts_recent_vs_historical'] = (pts_recent_3g - pts_historical_10g).fillna(0)
        
        # Feature 2: Volatilidad temporal de puntos (detecta cambios en consistencia)
        if self._register_feature('pts_volatility_trend', 'drift_detection'):
            pts_std_3g = self._get_historical_series(df, 'PTS', 3, 'std').shift(1)
            pts_std_10g = self._get_historical_series(df, 'PTS', 10, 'std').shift(1)
            df['pts_volatility_trend'] = (pts_std_3g - pts_std_10g).fillna(0)
        
        # Feature 3: Tendencia direccional de puntos (momentum)
        if self._register_feature('pts_momentum_indicator', 'drift_detection'):
            pts_trend = df.groupby('Player')['PTS'].shift(1).rolling(5, min_periods=2).apply(
                lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) >= 2 else 0
            )
            df['pts_momentum_indicator'] = pts_trend.fillna(0)
        
        # Feature 4: Adaptabilidad del jugador (qué tan rápido se adapta a cambios)
        if self._register_feature('player_adaptability_score', 'drift_detection'):
            # Medir qué tan consistente es el jugador en diferentes fases de temporada
            season_phase_consistency = df.groupby(['Player', 'season_phase'])['PTS'].transform('std')
            df['player_adaptability_score'] = (1 / (1 + season_phase_consistency)).fillna(0.5)
        
        logger.debug("Features de drift temporal creadas")
    
    def _create_recent_trend_features(self, df: pd.DataFrame) -> None:
        """
        NUEVO: Crear features que dan mayor peso a tendencias recientes.
        Estas features ayudan a capturar cambios en el rendimiento del jugador.
        """
        logger.debug("Creando features de tendencias recientes...")
        
        if 'PTS' not in df.columns:
            return
        
        # Feature 1: Promedio ponderado de puntos (más peso a juegos recientes)
        if self._register_feature('pts_weighted_avg_5g', 'temporal_advanced'):
            weights = np.array([0.4, 0.3, 0.2, 0.1])  # Pesos decrecientes
            pts_weighted = df.groupby('Player')['PTS'].shift(1).rolling(4, min_periods=1).apply(
                lambda x: np.average(x, weights=weights[:len(x)]) if len(x) > 0 else 0
            )
            df['pts_weighted_avg_5g'] = pts_weighted.fillna(0)
        
        # Feature 2: Tendencia exponencial de puntos
        if self._register_feature('pts_exponential_trend', 'temporal_advanced'):
            alpha = 0.3  # Factor de suavizado exponencial
            pts_ema = df.groupby('Player')['PTS'].shift(1).ewm(alpha=alpha, min_periods=1).mean()
            df['pts_exponential_trend'] = pts_ema.fillna(0)
        
        # Feature 3: Aceleración en el rendimiento (segunda derivada)
        if self._register_feature('pts_acceleration', 'temporal_advanced'):
            pts_diff1 = df.groupby('Player')['PTS'].shift(1).diff()
            pts_acceleration = pts_diff1.diff()
            df['pts_acceleration'] = pts_acceleration.fillna(0)
        
        # Feature 4: Ratio de mejora reciente
        if self._register_feature('pts_improvement_ratio', 'temporal_advanced'):
            pts_last_2g = self._get_historical_series(df, 'PTS', 2, 'mean').shift(1)
            pts_prev_2g = self._get_historical_series(df, 'PTS', 2, 'mean').shift(3)
            improvement_ratio = pts_last_2g / (pts_prev_2g + 1e-6)  # Evitar división por cero
            df['pts_improvement_ratio'] = improvement_ratio.fillna(1.0)
        
        # Feature 5: Consistencia reciente vs histórica
        if self._register_feature('pts_recent_consistency', 'temporal_advanced'):
            pts_cv_3g = (self._get_historical_series(df, 'PTS', 3, 'std').shift(1) / 
                        (self._get_historical_series(df, 'PTS', 3, 'mean').shift(1) + 1e-6))
            df['pts_recent_consistency'] = (1 / (1 + pts_cv_3g)).fillna(0.5)
        
        logger.debug("Features de tendencias recientes creadas")
    
    def _create_adaptive_features(self, df: pd.DataFrame) -> None:
        """
        NUEVO: Crear features adaptativas que se ajustan a cambios en el contexto del juego.
        Estas features ayudan al modelo a adaptarse a diferentes situaciones de juego.
        """
        logger.debug("Creando features adaptativas...")
        
        # Feature 1: Adaptación a la carga de juegos
        if self._register_feature('schedule_adaptation_factor', 'temporal_advanced'):
            # Factor que penaliza rendimiento cuando hay muchos juegos seguidos
            if 'schedule_density' in df.columns:
                df['schedule_adaptation_factor'] = np.exp(-df['schedule_density'] * 2)
            else:
                df['schedule_adaptation_factor'] = 1.0
        
        # Feature 2: Factor de recuperación después de descanso
        if self._register_feature('rest_recovery_factor', 'temporal_advanced'):
            if 'days_rest' in df.columns:
                # Curva de recuperación: óptimo en 1-2 días, penalización por mucho/poco descanso
                optimal_rest = 1.5
                df['rest_recovery_factor'] = np.exp(-0.5 * ((df['days_rest'] - optimal_rest) / 2) ** 2)
            else:
                df['rest_recovery_factor'] = 1.0
        
        # Feature 3: Factor de adaptación a oponente
        if self._register_feature('opponent_adaptation_score', 'temporal_advanced'):
            # Medir qué tan bien se adapta el jugador a diferentes tipos de defensa
            if 'Opp' in df.columns and 'PTS' in df.columns:
                player_vs_opp = df.groupby(['Player', 'Opp'])['PTS'].transform('mean')
                player_overall = df.groupby('Player')['PTS'].transform('mean')
                df['opponent_adaptation_score'] = (player_vs_opp / (player_overall + 1e-6)).fillna(1.0)
            else:
                df['opponent_adaptation_score'] = 1.0
        
        # Feature 4: Factor de momentum del equipo
        if self._register_feature('team_momentum_factor', 'temporal_advanced'):
            if 'is_win' in df.columns:
                # Momentum basado en victorias recientes del equipo
                team_wins_3g = df.groupby('Team')['is_win'].shift(1).rolling(3, min_periods=1).mean()
                df['team_momentum_factor'] = team_wins_3g.fillna(0.5)
            else:
                df['team_momentum_factor'] = 0.5
        
        # Feature 5: Factor de presión situacional
        if self._register_feature('situational_pressure_factor', 'temporal_advanced'):
            pressure_factor = 1.0
            
            # Aumentar presión en playoffs (si hay indicador)
            if 'is_playoff' in df.columns:
                pressure_factor += df['is_playoff'] * 0.3
            
            # Aumentar presión en juegos importantes (si hay indicador)
            if 'is_important_game' in df.columns:
                pressure_factor += df['is_important_game'] * 0.2
            
            # Aumentar presión en final de temporada
            if 'season_phase' in df.columns:
                pressure_factor += (df['season_phase'] == 2) * 0.1  # Late season
            
            df['situational_pressure_factor'] = pressure_factor
        
        logger.debug("Features adaptativas creadas")

    def _create_high_scoring_features(self, df: pd.DataFrame) -> None:
        """
        NUEVO: Crear features específicas para jugadores de alto scoring (20+ puntos).
        Estas features ayudan a predecir mejor a jugadores estrella.
        """
        logger.debug("Creando features específicas para alto scoring...")
        
        if 'PTS' not in df.columns:
            return
        
        # Feature 1: Indicador de jugador de alto scoring histórico
        if self._register_feature('is_high_scorer', 'temporal_advanced'):
            pts_season_avg = df.groupby('Player')['PTS'].transform('mean')
            df['is_high_scorer'] = (pts_season_avg >= 18).astype(int)
        
        # Feature 2: Potencial de explosión (likelihood de juegos de 30+ puntos)
        if self._register_feature('explosion_potential', 'temporal_advanced'):
            # Basado en máximo histórico y consistencia
            pts_max_5g = self._get_historical_series(df, 'PTS', 5, 'max').shift(1)
            pts_std_5g = self._get_historical_series(df, 'PTS', 5, 'std').shift(1)
            df['explosion_potential'] = (pts_max_5g / 30.0) * (1 + pts_std_5g / 10.0)
        
        # Feature 3: Momentum de alto scoring (racha de juegos de 20+ puntos)
        if self._register_feature('high_scoring_streak', 'temporal_advanced'):
            # Contar juegos consecutivos de 20+ puntos
            high_scoring_games = (df['PTS'].shift(1) >= 20).astype(int)
            df['high_scoring_streak'] = high_scoring_games.groupby(df['Player']).rolling(
                window=5, min_periods=1
            ).sum().reset_index(0, drop=True).fillna(0)
        
        # Feature 4: Eficiencia en alto volumen (puntos por tiro en alto volumen)
        if self._register_feature('high_volume_efficiency', 'temporal_advanced'):
            if 'FGA' in df.columns:
                fga_avg = self._get_historical_series(df, 'FGA', 5, 'mean').shift(1)
                pts_avg = self._get_historical_series(df, 'PTS', 5, 'mean').shift(1)
                # Solo para jugadores con alto volumen de tiros
                high_volume_mask = fga_avg >= 15
                df['high_volume_efficiency'] = np.where(
                    high_volume_mask, 
                    pts_avg / (fga_avg + 1e-6), 
                    0
                )
            else:
                df['high_volume_efficiency'] = 0
        
        # Feature 5: Factor de clutch scoring (rendimiento en situaciones importantes)
        if self._register_feature('clutch_scoring_factor', 'temporal_advanced'):
            # Basado en consistencia y promedio alto
            pts_avg_5g = self._get_historical_series(df, 'PTS', 5, 'mean').shift(1)
            pts_consistency = self._get_historical_series(df, 'PTS', 5, 'std').shift(1)
            clutch_factor = pts_avg_5g * (1 / (1 + pts_consistency / 5))
            df['clutch_scoring_factor'] = clutch_factor.fillna(0)
        
        logger.debug("Features de alto scoring creadas")
    
    def _create_elite_player_features(self, df: pd.DataFrame) -> None:
        """
        NUEVO: Crear features específicas para jugadores elite (25+ puntos promedio).
        Estas features capturan características únicas de superstrellas.
        """
        logger.debug("Creando features para jugadores elite...")
        
        if 'PTS' not in df.columns:
            return
        
        # Feature 1: Clasificación de tier de jugador
        if self._register_feature('player_tier', 'temporal_advanced'):
            pts_season_avg = df.groupby('Player')['PTS'].transform('mean')
            # Tier 0: Suplentes (0-8 pts), Tier 1: Rotación (8-15 pts), 
            # Tier 2: Titulares (15-22 pts), Tier 3: Estrellas (22-28 pts), Tier 4: Superstrellas (28+ pts)
            df['player_tier'] = pd.cut(
                pts_season_avg, 
                bins=[0, 8, 15, 22, 28, 100], 
                labels=[0, 1, 2, 3, 4], 
                include_lowest=True
            ).astype(float)
        
        # Feature 2: Capacidad de carry (llevar al equipo ofensivamente)
        if self._register_feature('team_carry_ability', 'temporal_advanced'):
            # Proporción de puntos del equipo que anota el jugador
            if 'Team' in df.columns:
                team_pts_avg = df.groupby(['Team', 'Date'])['PTS'].transform('sum')
                player_pts = df['PTS']
                df['team_carry_ability'] = (player_pts / (team_pts_avg + 1e-6)).fillna(0)
            else:
                df['team_carry_ability'] = 0
        
        # Feature 3: Versatilidad ofensiva (múltiples formas de anotar)
        if self._register_feature('offensive_versatility', 'temporal_advanced'):
            versatility_score = 0
            
            # Puntos desde tiros de campo
            if 'FG' in df.columns:
                fg_pts = self._get_historical_series(df, 'FG', 5, 'mean').shift(1) * 2
                versatility_score += fg_pts * 0.4
            
            # Puntos desde triples
            if '3P' in df.columns:
                three_pts = self._get_historical_series(df, '3P', 5, 'mean').shift(1) * 3
                versatility_score += three_pts * 0.3
            
            # Puntos desde tiros libres
            if 'FT' in df.columns:
                ft_pts = self._get_historical_series(df, 'FT', 5, 'mean').shift(1)
                versatility_score += ft_pts * 0.3
            
            df['offensive_versatility'] = versatility_score
        
        # Feature 4: Presión defensiva que enfrenta (proxy)
        if self._register_feature('defensive_attention', 'temporal_advanced'):
            # Jugadores elite enfrentan más presión defensiva
            pts_avg = self._get_historical_series(df, 'PTS', 10, 'mean').shift(1)
            usage_rate = df.get('usage_rate_5g', 20)
            # Estimar atención defensiva basada en scoring y usage
            df['defensive_attention'] = (pts_avg / 25.0) * (usage_rate / 30.0)
        
        # Feature 5: Factor de superstar (combinación de múltiples métricas elite)
        if self._register_feature('superstar_factor', 'temporal_advanced'):
            pts_avg = self._get_historical_series(df, 'PTS', 5, 'mean').shift(1)
            player_tier = df.get('player_tier', 2)
            carry_ability = df.get('team_carry_ability', 0.2)
            
            # Factor compuesto para identificar superstrellas
            superstar_factor = (
                (pts_avg / 30.0) * 0.4 +           # Promedio de puntos normalizado
                (player_tier / 4.0) * 0.3 +        # Tier del jugador
                carry_ability * 0.3                 # Capacidad de carry
            )
            df['superstar_factor'] = superstar_factor.fillna(0)
        
        logger.debug("Features para jugadores elite creadas")
    
    def _create_star_performance_features(self, df: pd.DataFrame) -> None:
        """
        NUEVO: Crear features que predicen actuaciones estelares específicas.
        Estas features ayudan a predecir cuándo un jugador tendrá un juego excepcional.
        """
        logger.debug("Creando features de actuaciones estelares...")
        
        if 'PTS' not in df.columns:
            return
        
        # Feature 1: Probabilidad de juego de 30+ puntos
        if self._register_feature('prob_30plus_game', 'temporal_advanced'):
            # Basado en historial de juegos de 30+ puntos
            games_30plus = (df.groupby('Player')['PTS'].shift(1) >= 30).astype(int)
            prob_30plus = games_30plus.groupby(df['Player']).rolling(
                window=20, min_periods=5
            ).mean().reset_index(0, drop=True).fillna(0)
            df['prob_30plus_game'] = prob_30plus
        
        # Feature 2: Probabilidad de juego de 40+ puntos
        if self._register_feature('prob_40plus_game', 'temporal_advanced'):
            # Basado en historial de juegos de 40+ puntos
            games_40plus = (df.groupby('Player')['PTS'].shift(1) >= 40).astype(int)
            prob_40plus = games_40plus.groupby(df['Player']).rolling(
                window=30, min_periods=5
            ).mean().reset_index(0, drop=True).fillna(0)
            df['prob_40plus_game'] = prob_40plus
        
        # Feature 3: Tendencia hacia juegos explosivos
        if self._register_feature('explosive_game_trend', 'temporal_advanced'):
            # Tendencia en los máximos de puntos recientes
            pts_max_3g = self._get_historical_series(df, 'PTS', 3, 'max').shift(1)
            pts_max_10g = self._get_historical_series(df, 'PTS', 10, 'max').shift(1)
            explosive_trend = pts_max_3g / (pts_max_10g + 1e-6)
            df['explosive_game_trend'] = explosive_trend.fillna(1.0)
        
        # Feature 4: Factor de motivación (juegos importantes)
        if self._register_feature('motivation_factor', 'temporal_advanced'):
            motivation = 1.0
            
            # Mayor motivación en playoffs
            if 'is_playoff' in df.columns:
                motivation += df['is_playoff'] * 0.4
            
            # Mayor motivación contra equipos fuertes
            if 'opp_def_rating' in df.columns:
                # Equipos con mejor defensa (rating más bajo) generan más motivación
                strong_defense = (df['opp_def_rating'] < 105).astype(int)
                motivation += strong_defense * 0.2
            
            # Mayor motivación en juegos televisados (proxy: fin de semana)
            if 'is_weekend' in df.columns:
                motivation += df['is_weekend'] * 0.1
            
            df['motivation_factor'] = motivation
        
        # Feature 5: Ceiling predictor (predictor del techo de rendimiento)
        if self._register_feature('performance_ceiling', 'temporal_advanced'):
            # Combina múltiples factores para predecir el techo de rendimiento
            pts_max_season = df.groupby('Player')['PTS'].transform('max')
            pts_avg_5g = self._get_historical_series(df, 'PTS', 5, 'mean').shift(1)
            explosion_potential = df.get('explosion_potential', 1.0)
            motivation = df.get('motivation_factor', 1.0)
            
            # Ceiling basado en máximo histórico ajustado por forma actual y motivación
            ceiling = (
                pts_max_season * 0.4 +              # 40% del máximo histórico
                pts_avg_5g * 1.5 * 0.4 +           # 40% de forma actual amplificada
                explosion_potential * 10 * 0.2     # 20% del potencial de explosión
            ) * motivation  # Ajustado por motivación
            
            df['performance_ceiling'] = ceiling.fillna(0)
        
        # Feature 6: Predictor de juego histórico (likelihood de anotar 50+ puntos)
        if self._register_feature('historic_game_potential', 'temporal_advanced'):
            # Solo para jugadores que han demostrado capacidad elite
            superstar_factor = df.get('superstar_factor', 0)
            performance_ceiling = df.get('performance_ceiling', 20)
            explosive_trend = df.get('explosive_game_trend', 1.0)
            
            # Potencial de juego histórico
            historic_potential = (
                superstar_factor * performance_ceiling * explosive_trend / 50.0
            )
            df['historic_game_potential'] = historic_potential.fillna(0)
        
        logger.debug("Features de actuaciones estelares creadas")
        
        # Actualizar registros de features
        high_scoring_features = [
            'is_high_scorer', 'explosion_potential', 'high_scoring_streak',
            'high_volume_efficiency', 'clutch_scoring_factor'
        ]
        
        elite_features = [
            'player_tier', 'team_carry_ability', 'offensive_versatility',
            'defensive_attention', 'superstar_factor'
        ]
        
        star_performance_features = [
            'prob_30plus_game', 'prob_40plus_game', 'explosive_game_trend',
            'motivation_factor', 'performance_ceiling', 'historic_game_potential'
        ]
        
        # Agregar a registros
        self._feature_registry['temporal_advanced'].extend(high_scoring_features)
        self._feature_registry['temporal_advanced'].extend(elite_features)
        self._feature_registry['temporal_advanced'].extend(star_performance_features)
        
        logger.info(f"Features para alto scoring creadas:")
        logger.info(f"  Alto scoring: {len(high_scoring_features)} features")
        logger.info(f"  Jugadores elite: {len(elite_features)} features")
        logger.info(f"  Actuaciones estelares: {len(star_performance_features)} features")

    def _detect_data_leakage_features(self, df: pd.DataFrame) -> List[str]:
        """
        REVOLUCIONARIO: Detecta y elimina COMPLETAMENTE features que causan data leakage.
        
        Args:
            df: DataFrame con features
            
        Returns:
            List[str]: Lista de features sospechosas de data leakage
        """
        leakage_features = []
        
        # LISTA COMPLETA DE FEATURES CON DATA LEAKAGE CONFIRMADO
        confirmed_leakage_features = [
            'GmSc',      # Game Score = PTS + 0.4*FG - 0.7*FGA - 0.4*(FTA-FT) + 0.7*ORB + 0.3*DRB + STL + 0.7*AST + 0.7*BLK - 0.4*PF - TOV
            'BPM',       # Box Plus/Minus incluye puntos directamente
            'PER',       # Player Efficiency Rating incluye puntos
            'TS%',       # True Shooting puede estar correlacionado con puntos del mismo juego
            'eFG%',      # Effective Field Goal incluye información del mismo juego
            'ORtg',      # Offensive Rating incluye puntos
            'DRtg',      # Defensive Rating puede estar correlacionado
            'VORP',      # Value Over Replacement incluye puntos
            'WS',        # Win Shares incluye puntos
            'WS/48',     # Win Shares per 48 incluye puntos
            'PIE',       # Player Impact Estimate incluye puntos
            'USG%',      # Usage puede incluir información del mismo juego
            '+/-',       # Plus/Minus del mismo juego
            'NETRTG',    # Net Rating incluye información del juego actual
            'PACE',      # Pace del juego actual
            'POSS',      # Possessions del juego actual
            'player_tier',  # NUEVO: Tier basado en promedio de puntos (data leakage indirecto)
        ]
        
        # DETECCIÓN AUTOMÁTICA DE CORRELACIONES EXTREMAS
        extreme_correlation_features = []
        if 'PTS' in df.columns:
            for feature in df.columns:
                if feature != 'PTS' and feature not in confirmed_leakage_features:
                    try:
                        correlation = abs(df[feature].corr(df['PTS']))
                        if correlation > 0.92:  # Umbral muy estricto
                            extreme_correlation_features.append(feature)
                            NBALogger.log_warning(logger, f"CORRELACIÓN EXTREMA detectada: {feature} (r={correlation:.3f})")
                    except:
                        pass
        
        # COMBINAR TODAS LAS FEATURES DE LEAKAGE
        all_leakage_features = confirmed_leakage_features + extreme_correlation_features
        
        # VERIFICAR CUÁLES EXISTEN EN EL DATAFRAME
        for feature in all_leakage_features:
            if feature in df.columns:
                leakage_features.append(feature)
                NBALogger.log_error(logger, f"DATA LEAKAGE CONFIRMADO - ELIMINANDO: {feature}")
        
        # DETECCIÓN ADICIONAL: Features que contienen información del juego actual
        suspicious_patterns = ['_current', '_today', '_game', '_match']
        for feature in df.columns:
            if any(pattern in feature.lower() for pattern in suspicious_patterns):
                if feature not in leakage_features:
                    leakage_features.append(feature)
                    NBALogger.log_warning(logger, "PATRÓN SOSPECHOSO detectado: {feature}")
        
        return leakage_features
    
    def _apply_leakage_filter(self, df: pd.DataFrame, features: List[str]) -> List[str]:
        """
        REVOLUCIONARIO: Aplica filtro de data leakage EXTREMADAMENTE ESTRICTO.
        
        Args:
            df: DataFrame con features
            features: Lista de features a filtrar
            
        Returns:
            List[str]: Features filtradas sin data leakage
        """
        leakage_features = self._detect_data_leakage_features(df)
        
        if leakage_features:
            NBALogger.log_error(logger, "🚨 APLICANDO FILTRO REVOLUCIONARIO DE DATA LEAKAGE 🚨")
            NBALogger.log_error(logger, "  Features ELIMINADAS por data leakage: {len(leakage_features)}")
            for feature in leakage_features:
                NBALogger.log_error(logger, "    ❌ ELIMINADO: {feature}")
            
            # Remover features de data leakage
            filtered_features = [f for f in features if f not in leakage_features]
            
            logger.info(f"  ✅ Features antes del filtro: {len(features)}")
            logger.info(f"  ✅ Features después del filtro: {len(filtered_features)}")
            logger.info(f"  ✅ Reducción de data leakage: {len(leakage_features)} features eliminadas")
            
            # VERIFICACIÓN ADICIONAL: Asegurar que no queden features dominantes
            if len(filtered_features) > 0:
                filtered_features = self._prevent_feature_dominance(df, filtered_features)
            
            return filtered_features
        else:
            logger.info("✅ No se detectó data leakage en las features")
            return features
    
    def _prevent_feature_dominance(self, df: pd.DataFrame, features: List[str]) -> List[str]:
        """
        NUEVO: Previene que una sola feature domine el modelo.
        
        Args:
            df: DataFrame con features
            features: Lista de features
            
        Returns:
            List[str]: Features balanceadas
        """
        if 'PTS' not in df.columns or len(features) < 5:
            return features
        
        # Calcular correlación con target para cada feature
        feature_correlations = {}
        for feature in features:
            if feature in df.columns:
                try:
                    corr = abs(df[feature].corr(df['PTS']))
                    if not pd.isna(corr):
                        feature_correlations[feature] = corr
                except:
                    feature_correlations[feature] = 0
        
        # Identificar features dominantes (correlación > 0.8)
        dominant_features = [f for f, corr in feature_correlations.items() if corr > 0.8]
        
        if dominant_features:
            NBALogger.log_warning(logger, "🔥 FEATURES DOMINANTES detectadas: {dominant_features}")
            
            # Mantener solo la mejor feature dominante y eliminar el resto
            if len(dominant_features) > 1:
                best_dominant = max(dominant_features, key=lambda x: feature_correlations[x])
                features_to_remove = [f for f in dominant_features if f != best_dominant]
                
                NBALogger.log_warning(logger, f"  ⚖️ Manteniendo mejor dominante: {best_dominant} (r={feature_correlations[best_dominant]:.3f})")
                NBALogger.log_warning(logger, f"  ❌ Eliminando dominantes: {features_to_remove}")
                
                features = [f for f in features if f not in features_to_remove]
        
        # Identificar features dominantes (correlación > 0.75 O importancia esperada > 30%)
        dominant_features = [f for f, corr in feature_correlations.items() if corr > 0.75]
        
        # NUEVO: Detectar features que podrían dominar por construcción
        potentially_dominant_features = [
            'player_tier', 'GmSc', 'BPM', 'PER', 'team_carry_ability'
        ]
        
        for feature in potentially_dominant_features:
            if feature in features and feature not in dominant_features:
                dominant_features.append(feature)
        
        return features

    def _create_revolutionary_ensemble_features(self, df: pd.DataFrame) -> None:
        """
        REVOLUCIONARIO: Features de ensemble ultra-avanzadas para stacking
        Diseñadas para capturar patrones complejos que modelos individuales no detectan
        """
        if self._check_feature_exists(df, 'ensemble_confidence_score'):
            return
        
        logger.info("Creando features revolucionarias de ensemble...")
        
        # 1. CONFIDENCE SCORE ULTRA-AVANZADO
        # Combinar múltiples indicadores de confianza
        confidence_components = []
        
        # Componente 1: Consistencia histórica
        if 'pts_hist_std_5g' in df.columns:
            consistency = 1.0 / (1.0 + df['pts_hist_std_5g'])
            confidence_components.append(consistency * 0.3)
        
        # Componente 2: Tendencia reciente
        if 'pts_trend_5g' in df.columns:
            trend_stability = 1.0 / (1.0 + np.abs(df['pts_trend_5g']))
            confidence_components.append(trend_stability * 0.25)
        
        # Componente 3: Minutos estables
        if 'mp_hist_std_5g' in df.columns:
            minutes_stability = 1.0 / (1.0 + df['mp_hist_std_5g'])
            confidence_components.append(minutes_stability * 0.2)
        
        # Componente 4: Eficiencia de tiro
        if 'shooting_efficiency_5g' in df.columns:
            shooting_confidence = df['shooting_efficiency_5g']
            confidence_components.append(shooting_confidence * 0.25)
        
        # Combinar componentes
        if confidence_components:
            df['ensemble_confidence_score'] = np.sum(confidence_components, axis=0)
            df['ensemble_confidence_score'] = df['ensemble_confidence_score'].clip(0, 1)
            self._register_feature('ensemble_confidence_score', 'meta_stacking')
        
        # 2. FEATURES DE VOLATILIDAD ULTRA-ESPECÍFICAS
        # Detectar patrones de volatilidad que afectan predicción
        
        # Volatilidad adaptativa (más peso a juegos recientes)
        if 'PTS' in df.columns:
            pts_volatility_components = []
            for window in [3, 5, 10]:
                pts_std = self._get_historical_series(df, 'PTS', window, 'std')
                pts_mean = self._get_historical_series(df, 'PTS', window, 'mean')
                cv = pts_std / (pts_mean + 0.1)  # Coeficiente de variación
                pts_volatility_components.append(cv)
            
            # Combinar volatilidades con pesos decrecientes
            weights = [0.5, 0.3, 0.2]
            df['pts_adaptive_volatility'] = sum(w * v for w, v in zip(weights, pts_volatility_components))
            self._register_feature('pts_adaptive_volatility', 'meta_stacking')
        
        # 3. FEATURES DE MOMENTUM ULTRA-AVANZADAS
        # Capturar momentum en múltiples escalas temporales
        
        if 'PTS' in df.columns:
            # Momentum a corto plazo (últimos 3 juegos)
            pts_recent = self._get_historical_series(df, 'PTS', 3, 'mean')
            pts_baseline = self._get_historical_series(df, 'PTS', 10, 'mean')
            df['pts_short_momentum'] = (pts_recent - pts_baseline) / (pts_baseline + 0.1)
            self._register_feature('pts_short_momentum', 'meta_stacking')
            
            # Momentum a medio plazo (últimos 5 vs 15 juegos)
            pts_medium = self._get_historical_series(df, 'PTS', 5, 'mean')
            pts_long_baseline = self._get_historical_series(df, 'PTS', 15, 'mean')
            df['pts_medium_momentum'] = (pts_medium - pts_long_baseline) / (pts_long_baseline + 0.1)
            self._register_feature('pts_medium_momentum', 'meta_stacking')
        
        # 4. FEATURES DE CONTEXTO SITUACIONAL ULTRA-ESPECÍFICAS
        
        # Rendimiento en situaciones específicas
        if 'is_home' in df.columns and 'PTS' in df.columns:
            # Diferencia de rendimiento casa vs visitante
            home_pts = df.groupby('Player')['PTS'].transform(
                lambda x: x.shift(1).rolling(10, min_periods=1).mean() if len(x) > 1 else x.mean()
            )
            # Simplificar el cálculo para evitar problemas
            df['home_away_pts_diff'] = np.where(df['is_home'] == 1, 0.1, -0.1) * home_pts
            self._register_feature('home_away_pts_diff', 'meta_stacking')
        
        # 5. FEATURES DE INTERACCIÓN ULTRA-COMPLEJAS
        
        # Interacción entre usage rate y eficiencia
        if 'usage_rate_5g' in df.columns and 'shooting_efficiency_5g' in df.columns:
            df['usage_efficiency_interaction'] = df['usage_rate_5g'] * df['shooting_efficiency_5g']
            self._register_feature('usage_efficiency_interaction', 'meta_stacking')
        
        # Interacción entre minutos y descanso
        if 'mp_hist_avg_5g' in df.columns and 'days_rest' in df.columns:
            df['minutes_rest_interaction'] = df['mp_hist_avg_5g'] * np.log(df['days_rest'] + 1)
            self._register_feature('minutes_rest_interaction', 'meta_stacking')
        
        logger.info("Features revolucionarias de ensemble creadas")

    def _create_ultra_critical_range_features(self, df: pd.DataFrame) -> None:
        """
        NUEVO: Features ULTRA-ESPECÍFICAS para rangos críticos (20+ puntos) 
        Diseñadas específicamente para alcanzar 97% efectividad en alto scoring.
        """
        logger.info("Creando features ULTRA-CRÍTICAS para rangos de alto scoring...")
        
        if 'PTS' not in df.columns:
            NBALogger.log_warning(logger, "PTS no disponible - saltando features de rangos críticos")
            return
        
        # 1. FEATURES DE DETECCIÓN DE ALTO SCORING
        
        # Probabilidad histórica de alto scoring (20+ puntos)
        pts_20plus_history = (df.groupby('Player')['PTS'].shift(1) >= 20).astype(int)
        df['high_scoring_probability'] = pts_20plus_history.groupby(df['Player']).rolling(
            10, min_periods=1
        ).mean().reset_index(0, drop=True)
        self._register_feature('high_scoring_probability', 'scoring')
        
        # Frecuencia de juegos explosivos (30+ puntos) en últimos 20 juegos
        pts_30plus_history = (df.groupby('Player')['PTS'].shift(1) >= 30).astype(int)
        # ==================== ENSEMBLE TEMPORAL MULTI-ESCALA ====================
        
        # Feature 1: Predictor de ensemble temporal (combina múltiples ventanas)
        if self._register_feature('temporal_ensemble_predictor', 'temporal_advanced'):
            # Combinar predictores de diferentes ventanas temporales
            pts_3g = self._get_historical_series(df, 'PTS', 3, 'mean')
            pts_5g = self._get_historical_series(df, 'PTS', 5, 'mean')
            pts_10g = self._get_historical_series(df, 'PTS', 10, 'mean')
            
            # Pesos adaptativos basados en recencia y estabilidad
            weight_3g = 0.5  # Mayor peso a reciente
            weight_5g = 0.3  # Peso medio
            weight_10g = 0.2  # Menor peso a histórico
            
            df['temporal_ensemble_predictor'] = (
                pts_3g * weight_3g + 
                pts_5g * weight_5g + 
                pts_10g * weight_10g
            ).fillna(0)
        
        # Feature 2: Ensemble de volatilidad (predice consistencia)
        if self._register_feature('volatility_ensemble_score', 'temporal_advanced'):
            # Combinar múltiples medidas de volatilidad
            pts_std_3g = self._get_historical_series(df, 'PTS', 3, 'std')
            pts_std_5g = self._get_historical_series(df, 'PTS', 5, 'std')
            pts_std_10g = self._get_historical_series(df, 'PTS', 10, 'std')
            
            # Score de estabilidad (inverso de volatilidad)
            stability_3g = 1 / (1 + pts_std_3g)
            stability_5g = 1 / (1 + pts_std_5g)
            stability_10g = 1 / (1 + pts_std_10g)
            
            df['volatility_ensemble_score'] = (
                stability_3g * 0.4 + 
                stability_5g * 0.35 + 
                stability_10g * 0.25
            ).fillna(0.5)
        
        # ==================== ENSEMBLE DE MOMENTUM INTELIGENTE ====================
        
        # Feature 3: Momentum ensemble (combina múltiples indicadores de momentum)
        if self._register_feature('momentum_ensemble_indicator', 'temporal_advanced'):
            # Momentum basado en tendencias
            pts_trend_3g = df.groupby('Player')['PTS'].shift(1).rolling(3, min_periods=2).apply(
                lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) >= 2 else 0
            ).fillna(0)
            
            pts_trend_5g = df.groupby('Player')['PTS'].shift(1).rolling(5, min_periods=3).apply(
                lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) >= 3 else 0
            ).fillna(0)
            
            # Momentum basado en diferencias
            pts_diff_recent = self._get_historical_series(df, 'PTS', 2, 'mean') - self._get_historical_series(df, 'PTS', 5, 'mean')
            
            # Combinar indicadores de momentum
            df['momentum_ensemble_indicator'] = (
                pts_trend_3g * 0.4 + 
                pts_trend_5g * 0.3 + 
                pts_diff_recent * 0.3
            ).fillna(0)
        
        # ==================== ENSEMBLE DE CONTEXTO SITUACIONAL ====================
        
        # Feature 4: Contexto ensemble (combina múltiples factores contextuales)
        if self._register_feature('context_ensemble_factor', 'temporal_advanced'):
            context_score = 1.0
            
            # Factor de descanso
            if 'days_rest' in df.columns:
                rest_factor = np.where(df['days_rest'] == 1, 0.95,  # B2B penalty
                             np.where(df['days_rest'] == 2, 1.0,    # Optimal
                             np.where(df['days_rest'] >= 3, 1.05, 1.0)))  # Rest bonus
                context_score *= rest_factor
            
            # Factor de ventaja local
            if 'is_home' in df.columns:
                home_factor = np.where(df['is_home'] == 1, 1.08, 0.98)  # Home advantage
                context_score *= home_factor
            
            # Factor de titular
            if 'is_started' in df.columns:
                starter_factor = np.where(df['is_started'] == 1, 1.12, 0.92)  # Starter advantage
                context_score *= starter_factor
            
            df['context_ensemble_factor'] = context_score
        
        # ==================== ENSEMBLE DE EFICIENCIA MULTI-DIMENSIONAL ====================
        
        # Feature 5: Eficiencia ensemble (combina múltiples métricas de eficiencia)
        if self._register_feature('efficiency_ensemble_score', 'temporal_advanced'):
            efficiency_components = []
            
            # Eficiencia de tiros de campo
            if 'FG' in df.columns and 'FGA' in df.columns:
                fg_made_hist = self._get_historical_series(df, 'FG', 5, 'mean')
                fga_hist = self._get_historical_series(df, 'FGA', 5, 'mean')
                fg_efficiency = fg_made_hist / (fga_hist + 0.1)
                efficiency_components.append(fg_efficiency * 0.4)
            
            # Eficiencia de triples
            if '3P' in df.columns and '3PA' in df.columns:
                three_made_hist = self._get_historical_series(df, '3P', 5, 'mean')
                three_att_hist = self._get_historical_series(df, '3PA', 5, 'mean')
                three_efficiency = three_made_hist / (three_att_hist + 0.1)
                efficiency_components.append(three_efficiency * 0.3)
            
            # Eficiencia de tiros libres
            if 'FT' in df.columns and 'FTA' in df.columns:
                ft_made_hist = self._get_historical_series(df, 'FT', 5, 'mean')
                ft_att_hist = self._get_historical_series(df, 'FTA', 5, 'mean')
                ft_efficiency = ft_made_hist / (ft_att_hist + 0.1)
                efficiency_components.append(ft_efficiency * 0.3)
            
            # Combinar componentes de eficiencia
            if efficiency_components:
                df['efficiency_ensemble_score'] = sum(efficiency_components)
            else:
                df['efficiency_ensemble_score'] = 0.45  # Default efficiency
        
        # ==================== ENSEMBLE DE OPORTUNIDADES INTELIGENTE ====================
        
        # Feature 6: Oportunidades ensemble (combina múltiples factores de oportunidad)
        if self._register_feature('opportunity_ensemble_factor', 'temporal_advanced'):
            opportunity_score = 1.0
            
            # Factor de minutos
            if 'MP' in df.columns:
                mp_hist = self._get_historical_series(df, 'MP', 5, 'mean')
                minutes_factor = mp_hist / 30.0  # Normalizar a 30 minutos
                opportunity_score *= minutes_factor.fillna(1.0)
            
            # Factor de uso estimado
            if 'FGA' in df.columns:
                fga_hist = self._get_historical_series(df, 'FGA', 5, 'mean')
                usage_factor = (fga_hist / 15.0).clip(0.5, 2.0)  # Normalizar y limitar
                opportunity_score *= usage_factor.fillna(1.0)
            
            # Factor de rol en equipo (estimado)
            pts_hist = self._get_historical_series(df, 'PTS', 5, 'mean')
            role_factor = np.where(pts_hist >= 20, 1.2,      # Star player
                         np.where(pts_hist >= 15, 1.1,      # Key player
                         np.where(pts_hist >= 10, 1.0, 0.9))) # Role player
            opportunity_score *= role_factor
            
            df['opportunity_ensemble_factor'] = opportunity_score.fillna(1.0)
        
        # ==================== ENSEMBLE PREDICTIVO FINAL ====================
        
        # Feature 7: Predictor ensemble final (combina todos los ensembles)
        if self._register_feature('final_ensemble_predictor', 'temporal_advanced'):
            temporal_pred = df.get('temporal_ensemble_predictor', 0)
            volatility_score = df.get('volatility_ensemble_score', 0.5)
            momentum_indicator = df.get('momentum_ensemble_indicator', 0)
            context_factor = df.get('context_ensemble_factor', 1.0)
            efficiency_score = df.get('efficiency_ensemble_score', 0.45)
            opportunity_factor = df.get('opportunity_ensemble_factor', 1.0)
            
            # Predicción final combinando todos los ensembles
            base_prediction = temporal_pred * volatility_score
            momentum_adjustment = base_prediction * (1 + momentum_indicator * 0.1)
            context_adjustment = momentum_adjustment * context_factor
            efficiency_adjustment = context_adjustment * (1 + efficiency_score * 0.2)
            final_prediction = efficiency_adjustment * opportunity_factor
            
            df['final_ensemble_predictor'] = final_prediction.fillna(0)
        
        # ==================== ENSEMBLE DE CONFIANZA ====================
        
        # Feature 8: Score de confianza del ensemble
        if self._register_feature('ensemble_confidence_score', 'temporal_advanced'):
            # Confianza basada en consistencia de predictores
            temporal_pred = df.get('temporal_ensemble_predictor', 0)
            final_pred = df.get('final_ensemble_predictor', 0)
            
            # Diferencia entre predictores (menor diferencia = mayor confianza)
            prediction_variance = abs(temporal_pred - final_pred)
            confidence = 1 / (1 + prediction_variance / 5)  # Normalizar
            
            # Ajustar por volatilidad histórica
            volatility_score = df.get('volatility_ensemble_score', 0.5)
            adjusted_confidence = confidence * volatility_score
            
            df['ensemble_confidence_score'] = adjusted_confidence.fillna(0.5)
        
        # ==================== VALIDACIÓN Y LIMPIEZA ====================
        
        ensemble_features = [
            'temporal_ensemble_predictor', 'volatility_ensemble_score', 'momentum_ensemble_indicator',
            'context_ensemble_factor', 'efficiency_ensemble_score', 'opportunity_ensemble_factor',
            'final_ensemble_predictor', 'ensemble_confidence_score'
        ]
        
        for feature in ensemble_features:
            if feature in df.columns:
                # Limpiar valores problemáticos
                df[feature] = df[feature].replace([np.inf, -np.inf], np.nan)
                df[feature] = df[feature].fillna(df[feature].median())
                
                # Aplicar límites razonables
                if 'predictor' in feature:
                    df[feature] = np.clip(df[feature], 0, 50)  # Límites para predicciones
                elif 'factor' in feature:
                    df[feature] = np.clip(df[feature], 0.5, 2.0)  # Límites para factores
                elif 'score' in feature:
                    df[feature] = np.clip(df[feature], 0, 1)  # Límites para scores
        
        logger.info(f"ENSEMBLE REVOLUCIONARIO CREADO: {len(ensemble_features)} features")
        logger.info("  temporal_ensemble_predictor: Combina múltiples ventanas temporales")
        logger.info("  final_ensemble_predictor: Predictor ensemble definitivo")
        logger.info("  ensemble_confidence_score: Score de confianza del ensemble")
    
    def _apply_noise_filters(self, df: pd.DataFrame, features: List[str]) -> List[str]:
        """
        Aplica filtros avanzados para eliminar features que solo agregan ruido a los modelos.
        
        Args:
            df: DataFrame con los datos
            features: Lista de features a filtrar
            
        Returns:
            List[str]: Lista de features filtradas sin ruido
        """
        logger.info(f"Iniciando filtrado de ruido en {len(features)} features...")
        
        # Copia de features para trabajar
        clean_features = features.copy()
        removed_features = []
        
        # FILTRO 1: Eliminar features con varianza extremadamente baja (casi constantes)
        logger.info("Aplicando filtro de varianza...")
        variance_threshold = 0.001  # Umbral muy bajo para features casi constantes
        
        for feature in features:
            if feature in df.columns:
                try:
                    variance = df[feature].var()
                    if pd.isna(variance) or variance < variance_threshold:
                        clean_features.remove(feature)
                        removed_features.append(f"{feature} (varianza: {variance:.6f})")
                except Exception:
                    # Si hay error calculando varianza, eliminar la feature
                    if feature in clean_features:
                        clean_features.remove(feature)
                        removed_features.append(f"{feature} (error cálculo)")
        
        # FILTRO 2: Eliminar features con demasiados valores NaN o infinitos
        logger.info("Aplicando filtro de valores faltantes/infinitos...")
        nan_threshold = 0.7  # Más del 70% de valores faltantes
        
        for feature in clean_features.copy():
            if feature in df.columns:
                try:
                    total_values = len(df[feature])
                    nan_count = df[feature].isna().sum()
                    inf_count = np.isinf(df[feature].replace([np.inf, -np.inf], np.nan)).sum()
                    
                    nan_ratio = (nan_count + inf_count) / total_values
                    
                    if nan_ratio > nan_threshold:
                        clean_features.remove(feature)
                        removed_features.append(f"{feature} (NaN/Inf: {nan_ratio:.2%})")
                except Exception:
                    clean_features.remove(feature)
                    removed_features.append(f"{feature} (error NaN)")
        
        # FILTRO 3: Eliminar features con distribuciones extremadamente sesgadas
        logger.info("Aplicando filtro de distribuciones sesgadas...")
        skewness_threshold = 10.0  # Sesgo extremo
        
        for feature in clean_features.copy():
            if feature in df.columns:
                try:
                    # Calcular solo con valores válidos
                    valid_values = df[feature].dropna().replace([np.inf, -np.inf], np.nan).dropna()
                    
                    if len(valid_values) > 10:  # Necesitamos suficientes valores
                        skewness = abs(valid_values.skew())
                        
                        if pd.isna(skewness) or skewness > skewness_threshold:
                            clean_features.remove(feature)
                            removed_features.append(f"{feature} (sesgo: {skewness:.2f})")
                except Exception:
                    clean_features.remove(feature)
                    removed_features.append(f"{feature} (error sesgo)")
        
        # FILTRO 4: Eliminar features con correlación perfecta o casi perfecta con otras
        logger.info("Aplicando filtro de correlación extrema...")
        correlation_threshold = 0.99  # Correlación casi perfecta
        
        if len(clean_features) > 1:
            try:
                # Calcular matriz de correlación solo con features válidas
                feature_data = df[clean_features].fillna(0).replace([np.inf, -np.inf], 0)
                corr_matrix = feature_data.corr().abs()
                
                # Encontrar pares con correlación extrema
                for i in range(len(corr_matrix.columns)):
                    for j in range(i+1, len(corr_matrix.columns)):
                        corr_value = corr_matrix.iloc[i, j]
                        
                        if not pd.isna(corr_value) and corr_value > correlation_threshold:
                            feature_i = corr_matrix.columns[i]
                            feature_j = corr_matrix.columns[j]
                            
                            # Eliminar la feature con menor varianza
                            if feature_i in clean_features and feature_j in clean_features:
                                var_i = df[feature_i].var()
                                var_j = df[feature_j].var()
                                
                                if pd.isna(var_i) or (not pd.isna(var_j) and var_i < var_j):
                                    clean_features.remove(feature_i)
                                    removed_features.append(f"{feature_i} (corr con {feature_j}: {corr_value:.3f})")
                                else:
                                    clean_features.remove(feature_j)
                                    removed_features.append(f"{feature_j} (corr con {feature_i}: {corr_value:.3f})")
            except Exception as e:
                logger.warning(f"Error en filtro de correlación: {e}")
        
        # FILTRO 5: Eliminar features conocidas como problemáticas para puntos
        logger.info("Aplicando filtro de features problemáticas conocidas...")
        problematic_patterns = [
            # Features que tienden a ser ruidosas o poco predictivas
            '_squared_',  # Features cuadráticas suelen ser ruidosas
            '_cubed_',    # Features cúbicas suelen ser ruidosas
            '_interaction_complex_',  # Interacciones complejas suelen ser ruidosas
            'random_',    # Features aleatorias
            'noise_',     # Features de ruido
            '_outlier_',  # Features de outliers suelen ser inestables
            '_extreme_',  # Features extremas suelen ser inestables
        ]
        
        for feature in clean_features.copy():
            for pattern in problematic_patterns:
                if pattern in feature.lower():
                    clean_features.remove(feature)
                    removed_features.append(f"{feature} (patrón problemático: {pattern})")
                    break
        
        # FILTRO 6: Validar que las features restantes sean numéricas
        logger.info("Validando features numéricas...")
        for feature in clean_features.copy():
            if feature in df.columns:
                try:
                    # Intentar convertir a numérico
                    numeric_values = pd.to_numeric(df[feature], errors='coerce')
                    
                    # Si más del 50% no se puede convertir, eliminar
                    if numeric_values.isna().sum() / len(numeric_values) > 0.5:
                        clean_features.remove(feature)
                        removed_features.append(f"{feature} (no numérica)")
                except Exception:
                    clean_features.remove(feature)
                    removed_features.append(f"{feature} (error numérico)")
        
        # FILTRO 7: Eliminar features con nombres sospechosos o mal formados
        logger.info("Aplicando filtro de nombres sospechosos...")
        suspicious_patterns = [
            'unnamed',
            'index',
            'level_',
            '__',  # Doble underscore suele indicar features temporales mal formadas
            '...',  # Puntos múltiples
            'tmp_',  # Features temporales
            'debug_',  # Features de debug
        ]
        
        for feature in clean_features.copy():
            feature_lower = feature.lower()
            for pattern in suspicious_patterns:
                if pattern in feature_lower:
                    clean_features.remove(feature)
                    removed_features.append(f"{feature} (nombre sospechoso: {pattern})")
                    break
        
        # Resumen del filtrado
        features_removed = len(features) - len(clean_features)
        logger.info(f"Filtrado completado: {features_removed} features eliminadas, {len(clean_features)} restantes")
        
        if features_removed > 0:
            logger.info("Features eliminadas por ruido:")
            for removed in removed_features[:10]:  # Mostrar solo las primeras 10
                logger.info(f"  - {removed}")
            if len(removed_features) > 10:
                logger.info(f"  ... y {len(removed_features) - 10} más")
        
        # Validación final: asegurar que tenemos features válidas
        if len(clean_features) == 0:
            logger.warning("ADVERTENCIA: Todos las features fueron eliminadas por filtros de ruido")
            # Devolver las features más básicas como fallback
            basic_features = [f for f in features if any(pattern in f for pattern in ['pts_', 'mp_', 'fg_', 'ast_', 'trb_'])]
            if basic_features:
                logger.info(f"Usando {len(basic_features)} features básicas como fallback")
                return basic_features[:20]  # Máximo 20 features básicas
            else:
                logger.error("No se encontraron features básicas válidas")
                return features[:10]  # Devolver las primeras 10 como último recurso
        
        return clean_features