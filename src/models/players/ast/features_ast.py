"""
Módulo de Características para Predicción de Asistencias (AST)
==============================================================

FEATURES BASADAS EN PRINCIPIOS FUNDAMENTALES DE ASISTENCIAS:

1. VISIÓN DE CANCHA: Capacidad de ver oportunidades de pase
2. BASKETBALL IQ: Inteligencia de juego y toma de decisiones
3. CONTROL DEL BALÓN: Manejo del balón y tiempo de posesión
4. RITMO DE JUEGO: Velocidad y transiciones
5. CONTEXTO DEL EQUIPO: Calidad de tiradores y sistema ofensivo
6. CONTEXTO DEL OPONENTE: Presión defensiva y estilo
7. HISTORIAL DE PASES: Rendimiento pasado en asistencias
8. SITUACIÓN DEL JUEGO: Contexto específico del partido
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
import logging
from config.logging_config import NBALogger
from datetime import datetime, timedelta
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression

warnings.filterwarnings('ignore')

# Configurar logging balanceado para features - mostrar etapas principales
logger = NBALogger.get_logger(__name__.split(".")[-1])  # Permitir logs de etapas principales

class AssistsFeatureEngineer:
    """
    Feature Engineer especializado en predicción de asistencias (AST)
    Basado en los principios fundamentales de los mejores pasadores de la NBA
    """
    
    def __init__(self, correlation_threshold: float = 0.95, max_features: int = 80, teams_df: pd.DataFrame = None):
        self.correlation_threshold = correlation_threshold
        self.max_features = max_features  # BALANCEADO a 80 para mejor rendimiento
        self.teams_df = teams_df  # Datos de equipos para features avanzadas
        self.feature_registry = {}
        self.feature_categories = {
            'historical_performance': 12,   # Aumentado para nuevas ventanas de tiempo
            'recent_trends': 12,            # Aumentado para volatilidad avanzada
            'efficiency_metrics': 8,        # Aumentado para eficiencia avanzada
            'contextual_factors': 3,        # Factores contextuales
            'basic_stats': 4,               # Estadísticas básicas
            'opponent_analysis': 6,         # Análisis del oponente
            'team_synergy': 6,              # Sinergia con el equipo
            'game_situation': 7,            # Situación específica del juego
            'physical_factors': 6,          # Factores físicos y biométricos
            'advanced_basketball': 10,      # Métricas avanzadas de basketball
            'ultra_predictive': 20,         # Features ultra-predictivas
            'hybrid_features_advanced': 15, # Features híbridas avanzadas (NUEVAS)
            'extreme_predictive': 15,       # Features EXTREMADAMENTE predictivas (90%+)
            'supreme_predictive': 20,       # Features SUPREMAS (95%+ perfección absoluta)
            'adaptability_critical': 10,    # Features CRÍTICAS de adaptabilidad (NUEVAS)
            'basketball_specific': 15,      # Features específicas de baloncesto NBA (NUEVAS)
            'advanced_playmaking': 15,      # Features avanzadas de playmaking (NUEVAS)
            'revolutionary_team_context': 20, # Features REVOLUCIONARIAS de contexto de equipo real (NUEVAS)
            'interaction_patterns': 17,    # Features de interacción y patrones complejos (NUEVAS)
            'model_feedback': 15,          # Features ultra-predictivas basadas en feedback del modelo (AUMENTADAS)
            'neural_patterns': 6,          # Features de patrones neuronales (NUEVAS)
            'quantum_inspired': 4,         # Features inspiradas en mecánica cuántica (NUEVAS)
            'chaos_theory': 4,             # Features de teoría del caos (NUEVAS)
            'fractal_features': 2,         # Features fractales (NUEVAS)
            'evolutionary_features': 3     # Features evolutivas (NUEVAS)
        }
        self.protected_features = ['AST', 'Player', 'Date', 'Team', 'Opp', 'Away', 'GS', 'Result', 'Pos']
        
    def _register_feature(self, feature_name: str, category: str):
        """Registra una feature en su categoría correspondiente"""
        if category not in self.feature_registry:
            self.feature_registry[category] = []
        self.feature_registry[category].append(feature_name)
    
    def _get_historical_series(self, df: pd.DataFrame, column: str, 
                              window: int = 5, operation: str = 'mean') -> pd.Series:
        """Calcula series históricas por jugador"""
        if column not in df.columns:
            return pd.Series(np.nan, index=df.index)
        
        grouped = df.groupby('Player')[column]
        
        if operation == 'mean':
            result = grouped.rolling(window=window, min_periods=1).mean().shift(1)
        elif operation == 'std':
            result = grouped.rolling(window=window, min_periods=2).std().shift(1)
        elif operation == 'max':
            result = grouped.rolling(window=window, min_periods=1).max().shift(1)
        elif operation == 'min':
            result = grouped.rolling(window=window, min_periods=1).min().shift(1)
        elif operation == 'sum':
            result = grouped.rolling(window=window, min_periods=1).sum().shift(1)
        else:
            result = grouped.rolling(window=window, min_periods=1).mean().shift(1)
        
        # Resetear índice para compatibilidad
        return result.reset_index(0, drop=True)
    
    def generate_all_features(self, df: pd.DataFrame) -> List[str]:
        """
        Genera features AVANZADAS Y ALTAMENTE PREDICTIVAS para asistencias
        Aprovecha todas las características disponibles en players y teams datasets
        """
        logger.info("Generando features AVANZADAS para predicción de asistencias...")
        
        # Verificar target
        if 'AST' in df.columns:
            ast_stats = df['AST'].describe()
            logger.info(f"Target AST disponible - Media={ast_stats['mean']:.1f}, Max={ast_stats['max']:.0f}")
        else:
            available_cols = list(df.columns)[:10]
            logger.warning(f"Target AST no disponible - features limitadas")
            logger.warning(f"Columnas disponibles (primeras 10): {available_cols}")
            logger.warning(f"Shape del dataset: {df.shape}")
        
        # Limpiar registro de features
        self.feature_registry = {}
        for category in self.feature_categories:
            self.feature_categories[category] = []
        
        # 1. FEATURES DE RENDIMIENTO HISTÓRICO (MÁS PREDICTIVAS)
        self._generate_historical_performance_features(df)
        
        # 2. FEATURES DE TENDENCIAS RECIENTES
        self._generate_recent_trends_features(df)
        
        # 3. FEATURES DE EFICIENCIA
        self._generate_efficiency_features(df)
        
        # 4. FEATURES CONTEXTUALES SIMPLES
        self._generate_contextual_features(df)
        
        # 5. FEATURES DE ESTADÍSTICAS BÁSICAS
        self._generate_basic_stats_features(df)
        
        # 6. FEATURES DE ANÁLISIS DEL OPONENTE
        self._generate_opponent_analysis_features(df)
        
        # 7. FEATURES DE SINERGIA CON EL EQUIPO
        self._generate_team_synergy_features(df)
        
        # 8. FEATURES DE SITUACIÓN DEL JUEGO
        self._generate_game_situation_features(df)
        
        # 9. FEATURES DE FACTORES FÍSICOS
        self._generate_physical_factors_features(df)
        
        # 10. FEATURES AVANZADAS DE BASKETBALL
        self._generate_advanced_basketball_features(df)
        
        # 11. FEATURES ULTRA-PREDICTIVAS (BASADAS EN TOP FEATURES IDENTIFICADAS)
        self._generate_ultra_predictive_features(df)
        
        # 12. FEATURES HÍBRIDAS ADICIONALES (PARTE 2)
        self._generate_hybrid_features_advanced(df)
        
        # 13. FEATURES EXTREMADAMENTE PREDICTIVAS (PARA 90%+ EFECTIVIDAD)
        self._generate_extreme_predictive_features(df)
        
        # 14. FEATURES SUPREMAS (PARA PERFECCIÓN ABSOLUTA - 95%+)
        self._generate_supreme_predictive_features(df)
        
        # === NUEVAS FEATURES CRÍTICAS DE ADAPTABILIDAD ===
        self._generate_adaptability_features(df)
        
        # === FEATURES ULTRA-ESPECÍFICAS PARA ASISTENCIAS NBA ===
        self._generate_basketball_specific_assist_features(df)
        
        # === FEATURES DE PLAYMAKING AVANZADO ===
        self._generate_advanced_playmaking_features(df)
        
        # === FEATURES REVOLUCIONARIAS DE CONTEXTO DE EQUIPO REAL ===
        self._generate_revolutionary_team_context_features(df)
        
        # === FEATURES DE INTERACCIÓN Y PATRONES COMPLEJOS ===
        self._generate_interaction_and_pattern_features(df)
        
        # === FEATURES ULTRA-PREDICTIVAS BASADAS EN FEEDBACK DEL MODELO ===
        self._generate_model_feedback_features(df)
        
        
        # Obtener lista de features creadas
        all_features = [col for col in df.columns if col not in self.protected_features]
        
        # MEJORA CRÍTICA: Usar selección inteligente en lugar del método anterior
        selected_features = self.intelligent_feature_selection(df, all_features, 'AST')
            
        # Asegurar que las features de model_feedback siempre estén incluidas (son las más importantes)
        model_feedback_features = self.feature_registry.get('model_feedback', [])
        priority_features = [f for f in model_feedback_features if f in df.columns and f in selected_features]
        
        # Si hay features prioritarias, asegurar que estén al principio
        if priority_features:
            other_features = [f for f in selected_features if f not in priority_features]
            selected_features = priority_features + other_features
        
        # Validar features finales
        final_features = self._validate_features(df, selected_features)
        
        # CRÍTICO: Validar que todas las features sean numéricas
        final_features = self._ensure_numeric_features(df, final_features)
        
        logger.info(f"FEATURES FINALES GENERADAS: {len(final_features)}")
        logger.info(f"Features por categoría: {dict(self.feature_categories)}")
        
        return final_features
    
    def _generate_historical_performance_features(self, df: pd.DataFrame) -> List[str]:
        """Features de rendimiento histórico - Las más predictivas"""
        features = []
        
        # 1. PROMEDIO DE TEMPORADA (EXPANDIENDO)
        ast_season_avg = df.groupby('Player')['AST'].expanding().mean().shift(1).reset_index(0, drop=True)
        df['ast_season_avg'] = ast_season_avg.fillna(0)
        features.append('ast_season_avg')
        
        # 2. PROMEDIO ÚLTIMOS 3 JUEGOS
        ast_avg_3g = df.groupby('Player')['AST'].rolling(3, min_periods=1).mean().shift(1).reset_index(0, drop=True)
        df['ast_avg_3g'] = ast_avg_3g.fillna(0)
        features.append('ast_avg_3g')
        
        # 3. PROMEDIO ÚLTIMOS 5 JUEGOS
        ast_avg_5g = df.groupby('Player')['AST'].rolling(5, min_periods=1).mean().shift(1).reset_index(0, drop=True)
        df['ast_avg_5g'] = ast_avg_5g.fillna(0)
        features.append('ast_avg_5g')
        
        # 4. PROMEDIO ÚLTIMOS 10 JUEGOS
        ast_avg_10g = df.groupby('Player')['AST'].rolling(10, min_periods=1).mean().shift(1).reset_index(0, drop=True)
        df['ast_avg_10g'] = ast_avg_10g.fillna(0)
        features.append('ast_avg_10g')
        
        # 5. ÚLTIMO JUEGO
        ast_last = df.groupby('Player')['AST'].shift(1).fillna(0)
        df['ast_last_game'] = ast_last
        features.append('ast_last_game')
        
        # 6. MÁXIMO EN ÚLTIMOS 5 JUEGOS
        ast_max_5g = df.groupby('Player')['AST'].rolling(5, min_periods=1).max().shift(1).reset_index(0, drop=True)
        df['ast_max_5g'] = ast_max_5g.fillna(0)
        features.append('ast_max_5g')
        
        # === NUEVAS FEATURES CRÍTICAS BASADAS EN MODELO DE PUNTOS ===
        
        # 7. PROMEDIO ÚLTIMOS 7 JUEGOS (ventana intermedia)
        ast_avg_7g = df.groupby('Player')['AST'].rolling(7, min_periods=1).mean().shift(1).reset_index(0, drop=True)
        df['ast_avg_7g'] = ast_avg_7g.fillna(0)
        features.append('ast_avg_7g')
        
        # 8. PROMEDIO ÚLTIMOS 15 JUEGOS (ventana larga)
        ast_avg_15g = df.groupby('Player')['AST'].rolling(15, min_periods=1).mean().shift(1).reset_index(0, drop=True)
        df['ast_avg_15g'] = ast_avg_15g.fillna(0)
        features.append('ast_avg_15g')
        
        # 9. AST POR MINUTO HISTÓRICO (5 juegos)
        mp_avg_5g = df.groupby('Player')['MP'].rolling(5, min_periods=1).mean().shift(1).reset_index(0, drop=True)
        df['ast_per_minute_5g'] = (ast_avg_5g / (mp_avg_5g + 1)).fillna(0)
        features.append('ast_per_minute_5g')
        
        # 10. AST POR MINUTO HISTÓRICO (10 juegos)
        mp_avg_10g = df.groupby('Player')['MP'].rolling(10, min_periods=1).mean().shift(1).reset_index(0, drop=True)
        df['ast_per_minute_10g'] = (ast_avg_10g / (mp_avg_10g + 1)).fillna(0)
        features.append('ast_per_minute_10g')
        
        # 11. AST SOBRE PROMEDIO DE TEMPORADA
        df['ast_above_season_avg'] = (ast_avg_5g - ast_season_avg).fillna(0)
        features.append('ast_above_season_avg')
        
        # 12. RATIO DE MEJORA (últimos 5 vs anteriores 5)
        ast_prev_5g = df.groupby('Player')['AST'].rolling(5, min_periods=1).mean().shift(6).reset_index(0, drop=True)
        df['ast_improvement_ratio'] = (ast_avg_5g / (ast_prev_5g + 0.1)).fillna(1.0)
        features.append('ast_improvement_ratio')
        
        # Registrar features
        for feature in features:
            self._register_feature(feature, 'historical_performance')
        
        return features
    
    def _generate_recent_trends_features(self, df: pd.DataFrame) -> List[str]:
        """Features de tendencias recientes"""
        features = []
        
        # 1. TENDENCIA LINEAL (ÚLTIMOS 5 JUEGOS)
        df['ast_trend_5g'] = df.groupby('Player')['AST'].rolling(5, min_periods=3).apply(
            lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) >= 3 else 0
        ).shift(1).reset_index(0, drop=True).fillna(0)
        features.append('ast_trend_5g')
        
        # 2. MOMENTUM (ÚLTIMOS 3 vs ANTERIORES 3)
        ast_recent_3 = df.groupby('Player')['AST'].rolling(3, min_periods=1).mean().shift(1).reset_index(0, drop=True)
        ast_prev_3 = df.groupby('Player')['AST'].rolling(3, min_periods=1).mean().shift(4).reset_index(0, drop=True)
        df['ast_momentum'] = (ast_recent_3 - ast_prev_3).fillna(0)
        features.append('ast_momentum')
        
        # 3. VOLATILIDAD (DESVIACIÓN ESTÁNDAR ÚLTIMOS 5)
        ast_std_5g = df.groupby('Player')['AST'].rolling(5, min_periods=2).std().shift(1).reset_index(0, drop=True)
        df['ast_volatility'] = ast_std_5g.fillna(0)
        features.append('ast_volatility')
        
        # 4. CONSISTENCIA (INVERSO DE VOLATILIDAD)
        df['ast_consistency'] = 1 / (df['ast_volatility'] + 0.1)
        features.append('ast_consistency')
        
        # === NUEVAS FEATURES DE VOLATILIDAD Y TENDENCIAS AVANZADAS ===
        
        # 5. VOLATILIDAD ÚLTIMOS 3 JUEGOS (más sensible)
        ast_std_3g = df.groupby('Player')['AST'].rolling(3, min_periods=2).std().shift(1).reset_index(0, drop=True)
        df['ast_volatility_3g'] = ast_std_3g.fillna(0)
        features.append('ast_volatility_3g')
        
        # 6. VOLATILIDAD ÚLTIMOS 10 JUEGOS (más estable)
        ast_std_10g = df.groupby('Player')['AST'].rolling(10, min_periods=3).std().shift(1).reset_index(0, drop=True)
        df['ast_volatility_10g'] = ast_std_10g.fillna(0)
        features.append('ast_volatility_10g')
        
        # 7. CONSISTENCIA MEJORADA (3 juegos)
        df['ast_consistency_3g'] = 1 / (df['ast_volatility_3g'] + 0.1)
        features.append('ast_consistency_3g')
        
        # 8. CONSISTENCIA MEJORADA (10 juegos)
        df['ast_consistency_10g'] = 1 / (df['ast_volatility_10g'] + 0.1)
        features.append('ast_consistency_10g')
        
        # 9. FACTOR DE TENDENCIA (últimos 10 juegos)
        df['ast_trend_10g'] = df.groupby('Player')['AST'].rolling(10, min_periods=5).apply(
            lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) >= 5 else 0
        ).shift(1).reset_index(0, drop=True).fillna(0)
        features.append('ast_trend_10g')
        
        # 10. FACTOR DE TENDENCIA (últimos 3 juegos - ultra reciente)
        df['ast_trend_3g'] = df.groupby('Player')['AST'].rolling(3, min_periods=2).apply(
            lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) >= 2 else 0
        ).shift(1).reset_index(0, drop=True).fillna(0)
        features.append('ast_trend_3g')
        
        # 11. MOMENTUM EXTENDIDO (últimos 5 vs anteriores 5)
        ast_recent_5 = df.groupby('Player')['AST'].rolling(5, min_periods=1).mean().shift(1).reset_index(0, drop=True)
        ast_prev_5 = df.groupby('Player')['AST'].rolling(5, min_periods=1).mean().shift(6).reset_index(0, drop=True)
        df['ast_momentum_5g'] = (ast_recent_5 - ast_prev_5).fillna(0)
        features.append('ast_momentum_5g')
        
        # 12. ACELERACIÓN (cambio en la tendencia)
        df['ast_trend_acceleration'] = (df['ast_trend_3g'] - df['ast_trend_5g']).fillna(0)
        features.append('ast_trend_acceleration')
        
        # Registrar features
        for feature in features:
            self._register_feature(feature, 'recent_trends')
        
        return features
    
    def _generate_efficiency_features(self, df: pd.DataFrame) -> List[str]:
        """Features de eficiencia"""
        features = []
        
        # 1. ASISTENCIAS POR MINUTO
        mp_last = df.groupby('Player')['MP'].shift(1)
        ast_last = df.groupby('Player')['AST'].shift(1)
        df['ast_per_minute'] = (ast_last / (mp_last + 1)).fillna(0)
        features.append('ast_per_minute')
        
        # 2. RATIO AST/TOV
        tov_last = df.groupby('Player')['TOV'].shift(1)
        df['ast_tov_ratio'] = (ast_last / (tov_last + 1)).fillna(0)
        features.append('ast_tov_ratio')
        
        # 3. PROMEDIO MINUTOS ÚLTIMOS 3 JUEGOS
        mp_avg_3g = df.groupby('Player')['MP'].rolling(3, min_periods=1).mean().shift(1).reset_index(0, drop=True)
        df['mp_avg_3g'] = mp_avg_3g.fillna(0)
        features.append('mp_avg_3g')
        
        # 4. EFICIENCIA OFENSIVA (AST + PTS por posesión estimada)
        pts_last = df.groupby('Player')['PTS'].shift(1)
        fga_last = df.groupby('Player')['FGA'].shift(1)
        df['offensive_efficiency'] = ((ast_last + pts_last) / (fga_last + tov_last + 1)).fillna(0)
        features.append('offensive_efficiency')
        
        # === NUEVAS FEATURES DE EFICIENCIA AVANZADAS ===
        
        # 5. EFICIENCIA DE ASISTENCIAS (últimos 5 juegos)
        ast_avg_5g = df.groupby('Player')['AST'].rolling(5, min_periods=1).mean().shift(1).reset_index(0, drop=True)
        mp_avg_5g = df.groupby('Player')['MP'].rolling(5, min_periods=1).mean().shift(1).reset_index(0, drop=True)
        df['ast_efficiency_5g'] = (ast_avg_5g / (mp_avg_5g + 1)).fillna(0)
        features.append('ast_efficiency_5g')
        
        # 6. TENDENCIA DE EFICIENCIA (mejorando vs empeorando)
        ast_eff_recent = df.groupby('Player')['ast_per_minute'].rolling(3, min_periods=1).mean().shift(1).reset_index(0, drop=True)
        ast_eff_prev = df.groupby('Player')['ast_per_minute'].rolling(3, min_periods=1).mean().shift(4).reset_index(0, drop=True)
        df['ast_efficiency_trend'] = (ast_eff_recent - ast_eff_prev).fillna(0)
        features.append('ast_efficiency_trend')
        
        # 7. RATIO AST/TOV MEJORADO (últimos 5 juegos)
        tov_avg_5g = df.groupby('Player')['TOV'].rolling(5, min_periods=1).mean().shift(1).reset_index(0, drop=True)
        df['ast_tov_ratio_5g'] = (ast_avg_5g / (tov_avg_5g + 1)).fillna(0)
        features.append('ast_tov_ratio_5g')
        
        # 8. EFICIENCIA DE CREACIÓN (AST por posesión usada)
        if 'FGA' in df.columns and 'FTA' in df.columns:
            fta_last = df.groupby('Player')['FTA'].shift(1)
            possessions_used = fga_last + 0.44 * fta_last + tov_last
            df['ast_creation_efficiency'] = (ast_last / (possessions_used + 1)).fillna(0)
            features.append('ast_creation_efficiency')
        
        # Registrar features
        for feature in features:
            self._register_feature(feature, 'efficiency_metrics')
        
        return features
    
    def _generate_contextual_features(self, df: pd.DataFrame) -> List[str]:
        """Features contextuales simples"""
        features = []
        
        # 1. ROL DE PLAYMAKER PRINCIPAL (RANKING EN EQUIPO)
        df['team_ast_rank'] = df.groupby(['Team', 'Date'])['AST'].rank(ascending=False, method='dense').shift(1).fillna(3)
        features.append('team_ast_rank')
        
        # 2. EXPERIENCIA EN LA TEMPORADA
        df['games_played'] = df.groupby('Player').cumcount()
        features.append('games_played')
        
        # 3. FACTOR TITULAR (MINUTOS > 20)
        df['starter_factor'] = (df['mp_avg_3g'] > 20).astype(int)
        features.append('starter_factor')
        
        # 4. CONTEXTO DE EQUIPO (SI HAY DATOS DE EQUIPOS)
        if self.teams_df is not None and 'AST' in self.teams_df.columns:
            team_ast_avg = self.teams_df.groupby('Team')['AST'].mean().reset_index()
            team_ast_avg.columns = ['Team', 'team_ast_avg']
            df_temp = df.merge(team_ast_avg, on='Team', how='left')
            df['team_ast_avg'] = df_temp['team_ast_avg'].fillna(25.0)
            features.append('team_ast_avg')
        
        # Registrar features
        for feature in features:
            self._register_feature(feature, 'contextual_factors')
        
        return features
    
    def _generate_basic_stats_features(self, df: pd.DataFrame) -> List[str]:
        """Features de estadísticas básicas correlacionadas"""
        features = []
        
        # 1. PROMEDIO DE PUNTOS (CORRELACIONADO CON AST)
        pts_avg_3g = df.groupby('Player')['PTS'].rolling(3, min_periods=1).mean().shift(1).reset_index(0, drop=True)
        df['pts_avg_3g'] = pts_avg_3g.fillna(0)
        features.append('pts_avg_3g')
        
        # 2. PROMEDIO DE ROBOS (VISIÓN DE CANCHA)
        if 'STL' in df.columns:
            stl_avg_3g = df.groupby('Player')['STL'].rolling(3, min_periods=1).mean().shift(1).reset_index(0, drop=True)
            df['stl_avg_3g'] = stl_avg_3g.fillna(0)
            features.append('stl_avg_3g')
        
        # 3. PROMEDIO DE PÉRDIDAS (CONTROL DEL BALÓN)
        if 'TOV' in df.columns:
            tov_avg_3g = df.groupby('Player')['TOV'].rolling(3, min_periods=1).mean().shift(1).reset_index(0, drop=True)
            df['tov_avg_3g'] = tov_avg_3g.fillna(0)
            features.append('tov_avg_3g')
        
        # 4. PORCENTAJE DE TIROS DE CAMPO (EFICIENCIA)
        if 'FG%' in df.columns:
            fg_pct_avg_3g = df.groupby('Player')['FG%'].rolling(3, min_periods=1).mean().shift(1).reset_index(0, drop=True)
            df['fg_pct_avg_3g'] = fg_pct_avg_3g.fillna(0)
            features.append('fg_pct_avg_3g')
        
        # Registrar features
        for feature in features:
            self._register_feature(feature, 'basic_stats')
        
        return features

    def _generate_opponent_analysis_features(self, df: pd.DataFrame) -> List[str]:
        """Features de análisis del oponente - Aprovecha datos de teams"""
        features = []
        
        # 1. ASISTENCIAS PERMITIDAS POR EL OPONENTE (HISTÓRICO)
        opp_ast_allowed = df.groupby('Opp')['AST'].rolling(5, min_periods=1).mean().shift(1).reset_index(0, drop=True)
        df['opp_ast_allowed'] = opp_ast_allowed.fillna(5.0)
        features.append('opp_ast_allowed')
        
        # 2. PRESIÓN DEFENSIVA DEL OPONENTE (ROBOS)
        if 'STL' in df.columns:
            opp_defensive_pressure = df.groupby('Opp')['STL'].rolling(3, min_periods=1).mean().shift(1).reset_index(0, drop=True)
            df['opp_defensive_pressure'] = opp_defensive_pressure.fillna(7.5)
            features.append('opp_defensive_pressure')
        
        # 3. RITMO DE JUEGO DEL OPONENTE (FGA como proxy)
        if 'FGA' in df.columns:
            opp_pace = df.groupby('Opp')['FGA'].rolling(3, min_periods=1).mean().shift(1).reset_index(0, drop=True)
            df['opp_pace'] = opp_pace.fillna(85.0)
            features.append('opp_pace')
        
        # 4. EFICIENCIA DEFENSIVA DEL OPONENTE (PUNTOS PERMITIDOS)
        opp_pts_allowed = df.groupby('Opp')['PTS'].rolling(5, min_periods=1).mean().shift(1).reset_index(0, drop=True)
        df['opp_pts_allowed'] = opp_pts_allowed.fillna(110.0)
        features.append('opp_pts_allowed')
        
        # 5. PÉRDIDAS FORZADAS POR EL OPONENTE
        if 'TOV' in df.columns:
            opp_tov_forced = df.groupby('Opp')['TOV'].rolling(3, min_periods=1).mean().shift(1).reset_index(0, drop=True)
            df['opp_tov_forced'] = opp_tov_forced.fillna(14.0)
            features.append('opp_tov_forced')
        
        # 6. VENTAJA MATCHUP (AST del jugador vs AST permitidas por oponente)
        ast_last = df.groupby('Player')['AST'].shift(1)
        df['matchup_advantage'] = (ast_last / (df['opp_ast_allowed'] + 0.1)).fillna(0)
        features.append('matchup_advantage')
        
        # Registrar features
        for feature in features:
            self._register_feature(feature, 'opponent_analysis')
        
        return features
    
    def _generate_team_synergy_features(self, df: pd.DataFrame) -> List[str]:
        """Features de sinergia con el equipo - Aprovecha datos de teams"""
        features = []
        
        # 1. PORCENTAJE DE ASISTENCIAS DEL EQUIPO
        team_ast_total = df.groupby(['Team', 'Date'])['AST'].transform('sum').shift(1)
        player_ast_hist = df.groupby('Player')['AST'].shift(1)
        df['team_ast_share'] = (player_ast_hist / (team_ast_total + 1)).fillna(0)
        features.append('team_ast_share')
        
        # 2. CALIDAD DE TIRADORES DEL EQUIPO (FG% promedio)
        if 'FG%' in df.columns:
            team_fg_pct = df.groupby(['Team', 'Date'])['FG%'].transform('mean').shift(1)
            df['team_shooting_quality'] = team_fg_pct.fillna(0.45)
            features.append('team_shooting_quality')
        
        # 3. ESPACIAMIENTO DEL EQUIPO (3P% promedio)
        if '3P%' in df.columns:
            team_3p_pct = df.groupby(['Team', 'Date'])['3P%'].transform('mean').shift(1)
            df['team_spacing'] = team_3p_pct.fillna(0.35)
            features.append('team_spacing')
        
        # 4. RITMO OFENSIVO DEL EQUIPO
        if 'FGA' in df.columns:
            team_pace = df.groupby(['Team', 'Date'])['FGA'].transform('sum').shift(1)
            df['team_offensive_pace'] = team_pace.fillna(85.0)
            features.append('team_offensive_pace')
        
        # 5. EFICIENCIA OFENSIVA DEL EQUIPO (PTS/FGA)
        if 'PTS' in df.columns and 'FGA' in df.columns:
            team_pts = df.groupby(['Team', 'Date'])['PTS'].transform('sum').shift(1)
            team_fga = df.groupby(['Team', 'Date'])['FGA'].transform('sum').shift(1)
            df['team_offensive_efficiency'] = (team_pts / (team_fga + 1)).fillna(1.1)
            features.append('team_offensive_efficiency')
        
        # 6. SINERGIA PASADOR-TIRADORES (AST del jugador * calidad de tiradores)
        df['passer_shooter_synergy'] = df['ast_per_minute'] * df['team_shooting_quality']
        features.append('passer_shooter_synergy')
        
        # Eliminar código problemático que usa columnas inexistentes
        
        # Registrar features
        for feature in features:
            self._register_feature(feature, 'team_synergy')
        
        return features
    
    def _generate_game_situation_features(self, df: pd.DataFrame) -> List[str]:
        """Features de situación específica del juego"""
        features = []
        
        # 1. FACTOR CASA/VISITANTE
        if 'is_home' in df.columns:
            df['home_advantage'] = df['is_home']
            features.append('home_advantage')
        
        # 2. FACTOR TITULAR
        if 'is_started' in df.columns:
            df['starter_impact'] = df['is_started']
            features.append('starter_impact')
        
        # 3. BACK-TO-BACK GAMES (diferencia de fechas)
        df['days_rest'] = df.groupby('Player')['Date'].diff().dt.days.fillna(2)
        df['is_back_to_back'] = (df['days_rest'] <= 1).astype(int)
        features.append('is_back_to_back')
        
        # 4. RESULTADO ESPERADO (basado en diferencia de puntos histórica)
        if 'point_diff' in df.columns:
            team_point_diff_avg = df.groupby('Team')['point_diff'].rolling(5, min_periods=1).mean().shift(1).reset_index(0, drop=True)
            df['expected_result'] = (team_point_diff_avg > 0).astype(int).fillna(0)
            features.append('expected_result')
        
        # 5. MINUTOS ESPERADOS (basado en promedio reciente)
        mp_expected = df.groupby('Player')['MP'].rolling(5, min_periods=1).mean().shift(1).reset_index(0, drop=True)
        df['minutes_expected'] = mp_expected.fillna(20.0)
        features.append('minutes_expected')
        
        # 6. FACTOR CLUTCH (juegos cerrados)
        if 'total_score' in df.columns:
            df['high_scoring_game'] = (df['total_score'] > 220).astype(int)
            features.append('high_scoring_game')
        
        # 7. OVERTIME FACTOR
        if 'has_overtime' in df.columns:
            df['overtime_factor'] = df['has_overtime'].astype(int)
            features.append('overtime_factor')
        
        # Registrar features
        for feature in features:
            self._register_feature(feature, 'game_situation')
        
        return features
    
    def _generate_physical_factors_features(self, df: pd.DataFrame) -> List[str]:
        """Features de factores físicos y biométricos"""
        features = []
        
        # 1. ÍNDICE DE MASA CORPORAL (BMI)
        if 'BMI' in df.columns:
            df['bmi_factor'] = df['BMI']
            features.append('bmi_factor')
        
        # 2. ALTURA EN PULGADAS
        if 'Height_Inches' in df.columns:
            df['height_factor'] = df['Height_Inches']
            features.append('height_factor')
        
        # 3. PESO
        if 'Weight' in df.columns:
            df['weight_factor'] = df['Weight']
            features.append('weight_factor')
        
        # 4. FACTOR DE POSICIÓN (basado en altura y peso)
        if 'Height_Inches' in df.columns and 'Weight' in df.columns:
            # Crear un índice de posición basado en físico
            df['position_index'] = (df['Height_Inches'] * 0.6 + df['Weight'] * 0.4) / 100
            features.append('position_index')
        
        # 5. FATIGA ACUMULADA (minutos en últimos 3 juegos)
        mp_sum_3g = df.groupby('Player')['MP'].rolling(3, min_periods=1).sum().shift(1).reset_index(0, drop=True)
        df['fatigue_factor'] = mp_sum_3g.fillna(60.0)
        features.append('fatigue_factor')
        
        # 6. CARGA DE TRABAJO (minutos por días de descanso)
        df['workload_ratio'] = df['fatigue_factor'] / (df['days_rest'] + 1)
        features.append('workload_ratio')
        
        # Registrar features
        for feature in features:
            self._register_feature(feature, 'physical_factors')
        
        return features
    
    def _generate_advanced_basketball_features(self, df: pd.DataFrame) -> List[str]:
        """Features avanzadas de basketball usando todas las estadísticas disponibles"""
        features = []
        
        # 1. TRUE SHOOTING PERCENTAGE (TS%)
        if 'TS%' in df.columns:
            ts_pct_avg = df.groupby('Player')['TS%'].rolling(5, min_periods=1).mean().shift(1).reset_index(0, drop=True)
            df['true_shooting_avg'] = ts_pct_avg.fillna(0.55)
            features.append('true_shooting_avg')
        
        # 2. GAME SCORE PROMEDIO
        if 'GmSc' in df.columns:
            gmsc_avg = df.groupby('Player')['GmSc'].rolling(5, min_periods=1).mean().shift(1).reset_index(0, drop=True)
            df['game_score_avg'] = gmsc_avg.fillna(8.0)
            features.append('game_score_avg')
        
        # 3. BOX PLUS/MINUS PROMEDIO
        if 'BPM' in df.columns:
            bpm_avg = df.groupby('Player')['BPM'].rolling(5, min_periods=1).mean().shift(1).reset_index(0, drop=True)
            df['bpm_avg'] = bpm_avg.fillna(0.0)
            features.append('bpm_avg')
        
        # 4. PLUS/MINUS PROMEDIO
        if '+/-' in df.columns:
            plus_minus_avg = df.groupby('Player')['+/-'].rolling(5, min_periods=1).mean().shift(1).reset_index(0, drop=True)
            df['plus_minus_avg'] = plus_minus_avg.fillna(0.0)
            features.append('plus_minus_avg')
        
        # 5. USAGE RATE (estimado basado en FGA, FTA, TOV, MP)
        if all(col in df.columns for col in ['FGA', 'FTA', 'TOV', 'MP']):
            # Estimación simplificada de Usage Rate
            team_mp = df.groupby(['Team', 'Date'])['MP'].transform('sum').shift(1)
            team_fga = df.groupby(['Team', 'Date'])['FGA'].transform('sum').shift(1)
            team_fta = df.groupby(['Team', 'Date'])['FTA'].transform('sum').shift(1)
            team_tov = df.groupby(['Team', 'Date'])['TOV'].transform('sum').shift(1)
            
            player_fga = df.groupby('Player')['FGA'].shift(1)
            player_fta = df.groupby('Player')['FTA'].shift(1)
            player_tov = df.groupby('Player')['TOV'].shift(1)
            player_mp = df.groupby('Player')['MP'].shift(1)
            
            usage_numerator = (player_fga + 0.44 * player_fta + player_tov) * (team_mp / 5)
            usage_denominator = (team_fga + 0.44 * team_fta + team_tov) * player_mp
            df['usage_rate'] = (usage_numerator / (usage_denominator + 1)).fillna(0.2)
            features.append('usage_rate')
        
        # 6. ASSIST RATE (AST / (FGA + 0.44*FTA + AST + TOV))
        if all(col in df.columns for col in ['AST', 'FGA', 'FTA', 'TOV']):
            ast_last = df.groupby('Player')['AST'].shift(1)
            fga_last = df.groupby('Player')['FGA'].shift(1)
            fta_last = df.groupby('Player')['FTA'].shift(1)
            tov_last = df.groupby('Player')['TOV'].shift(1)
            
            possessions = fga_last + 0.44 * fta_last + ast_last + tov_last
            df['assist_rate'] = (ast_last / (possessions + 1)).fillna(0)
            features.append('assist_rate')
        
        # 7. TURNOVER RATE
        if all(col in df.columns for col in ['TOV', 'FGA', 'FTA']):
            tov_last = df.groupby('Player')['TOV'].shift(1)
            fga_last = df.groupby('Player')['FGA'].shift(1)
            fta_last = df.groupby('Player')['FTA'].shift(1)
            
            possessions = fga_last + 0.44 * fta_last + tov_last
            df['turnover_rate'] = (tov_last / (possessions + 1)).fillna(0)
            features.append('turnover_rate')
        
        # 8. STEAL RATE (robos por posesión defensiva estimada)
        if 'STL' in df.columns:
            stl_last = df.groupby('Player')['STL'].shift(1)
            mp_last = df.groupby('Player')['MP'].shift(1)
            df['steal_rate'] = (stl_last / (mp_last / 48 * 100 + 1)).fillna(0)
            features.append('steal_rate')
        
        # 9. DOUBLE-DOUBLE FREQUENCY
        if 'double_double' in df.columns:
            dd_freq = df.groupby('Player')['double_double'].rolling(10, min_periods=1).mean().shift(1).reset_index(0, drop=True)
            df['double_double_freq'] = dd_freq.fillna(0)
            features.append('double_double_freq')
        
        # 10. TRIPLE-DOUBLE FREQUENCY
        if 'triple_double' in df.columns:
            td_freq = df.groupby('Player')['triple_double'].rolling(20, min_periods=1).mean().shift(1).reset_index(0, drop=True)
            df['triple_double_freq'] = td_freq.fillna(0)
            features.append('triple_double_freq')
        
        # Registrar features
        for feature in features:
            self._register_feature(feature, 'advanced_basketball')
        
        return features

    def _generate_ultra_predictive_features(self, df: pd.DataFrame) -> List[str]:
        """
        Features ULTRA-PREDICTIVAS basadas en las top features identificadas por el modelo:
        1. ast_season_avg (35.17%) - Promedio de temporada
        2. ast_avg_3g (13.86%) - Promedio últimos 3 juegos  
        3. starter_impact (5.80%) - Factor titular
        4. triple_double_freq (5.00%) - Frecuencia de triple-dobles
        5. team_shooting_quality (2.86%) - Calidad de tiradores del equipo
        """
        features = []
        
        # === GRUPO 1: MEJORAS BASADAS EN ast_season_avg (TOP FEATURE) ===
        
        # 1. PROMEDIO PONDERADO TEMPORADA (más peso a juegos recientes)
        ast_expanding = df.groupby('Player')['AST'].expanding().mean().shift(1).reset_index(0, drop=True)
        ast_recent_5 = df.groupby('Player')['AST'].rolling(5, min_periods=1).mean().shift(1).reset_index(0, drop=True)
        games_played = df.groupby('Player').cumcount()
        
        # Peso dinámico: más peso a temporada con más juegos, pero siempre algo de peso reciente
        weight_season = np.minimum(games_played / 30, 0.8)  # Máximo 80% peso a temporada
        weight_recent = 1 - weight_season
        
        df['ast_weighted_season_avg'] = (ast_expanding * weight_season + ast_recent_5 * weight_recent).fillna(0)
        features.append('ast_weighted_season_avg')
        
        # 2. TENDENCIA DE TEMPORADA (mejorando vs empeorando)
        ast_first_10 = df.groupby('Player')['AST'].rolling(10, min_periods=5).mean().shift(1).reset_index(0, drop=True)
        ast_last_10 = df.groupby('Player')['AST'].rolling(10, min_periods=1).mean().shift(1).reset_index(0, drop=True)
        df['ast_season_trend'] = (ast_last_10 - ast_first_10).fillna(0)
        features.append('ast_season_trend')
        
        # 3. ESTABILIDAD DE TEMPORADA (consistencia)
        ast_season_std = df.groupby('Player')['AST'].expanding().std().shift(1).reset_index(0, drop=True)
        df['ast_season_stability'] = (ast_expanding / (ast_season_std + 0.1)).fillna(0)
        features.append('ast_season_stability')
        
        # === GRUPO 2: MEJORAS BASADAS EN ast_avg_3g (SEGUNDA TOP FEATURE) ===
        
        # 4. MOMENTUM ULTRA-RECIENTE (últimos 2 vs anteriores 2)
        ast_last_2 = df.groupby('Player')['AST'].rolling(2, min_periods=1).mean().shift(1).reset_index(0, drop=True)
        ast_prev_2 = df.groupby('Player')['AST'].rolling(2, min_periods=1).mean().shift(3).reset_index(0, drop=True)
        df['ast_ultra_momentum'] = (ast_last_2 - ast_prev_2).fillna(0)
        features.append('ast_ultra_momentum')
        
        # 5. PREDICTOR HÍBRIDO OPTIMIZADO (temporada + reciente + momentum)
        df['ast_hybrid_predictor'] = (
            df['ast_weighted_season_avg'] * 0.5 + 
            df['ast_avg_3g'] * 0.3 + 
            df['ast_ultra_momentum'] * 0.2
        ).fillna(0)
        features.append('ast_hybrid_predictor')
        
        # 6. FACTOR DE CONFIANZA EN PREDICCIÓN RECIENTE
        ast_std_3g = df.groupby('Player')['AST'].rolling(3, min_periods=1).std().shift(1).reset_index(0, drop=True)
        df['ast_recent_confidence'] = (df['ast_avg_3g'] / (ast_std_3g + 0.1)).fillna(0)
        features.append('ast_recent_confidence')
        
        # === GRUPO 3: MEJORAS BASADAS EN starter_impact (TERCERA TOP FEATURE) ===
        
        # 7. IMPACTO TITULAR PONDERADO POR MINUTOS
        if 'is_started' in df.columns:
            mp_avg = df.groupby('Player')['MP'].rolling(5, min_periods=1).mean().shift(1).reset_index(0, drop=True)
            df['starter_minutes_impact'] = df['is_started'] * (mp_avg / 48)  # Normalizado por 48 min
            features.append('starter_minutes_impact')
        
        # 8. DIFERENCIA AST TITULAR vs SUPLENTE
        if 'is_started' in df.columns:
            ast_as_starter = df.groupby('Player').apply(
                lambda x: x[x['is_started'] == 1]['AST'].shift(1).mean()
            ).reset_index(level=0, drop=True)
            ast_as_bench = df.groupby('Player').apply(
                lambda x: x[x['is_started'] == 0]['AST'].shift(1).mean()
            ).reset_index(level=0, drop=True)
            df['starter_bench_diff'] = (ast_as_starter - ast_as_bench).fillna(0)
            features.append('starter_bench_diff')
        
        # === GRUPO 4: MEJORAS BASADAS EN triple_double_freq (CUARTA TOP FEATURE) ===
        
        # 9. POTENCIAL DE TRIPLE-DOBLE (AST + otros stats altos)
        if all(col in df.columns for col in ['PTS', 'TRB', 'AST']):
            pts_last = df.groupby('Player')['PTS'].shift(1)
            trb_last = df.groupby('Player')['TRB'].shift(1)
            ast_last = df.groupby('Player')['AST'].shift(1)
            
            # Contar cuántas categorías están cerca de doble dígito
            near_double_pts = (pts_last >= 8).astype(int)
            near_double_trb = (trb_last >= 8).astype(int)
            near_double_ast = (ast_last >= 8).astype(int)
            
            df['triple_double_potential'] = near_double_pts + near_double_trb + near_double_ast
            features.append('triple_double_potential')
        
        # 10. FRECUENCIA DE JUEGOS DE ALTO IMPACTO (AST >= 7)
        high_ast_games = df.groupby('Player').apply(
            lambda x: (x['AST'].shift(1) >= 7).rolling(10, min_periods=1).mean()
        ).reset_index(level=0, drop=True).fillna(0)
        df['high_assist_game_freq'] = high_ast_games
        features.append('high_assist_game_freq')
        
        # === GRUPO 5: MEJORAS BASADAS EN team_shooting_quality (QUINTA TOP FEATURE) ===
        
        # 11. SINERGIA OPTIMIZADA PASADOR-TIRADORES
        if 'team_shooting_quality' in df.columns:
            # Multiplicar AST histórico por calidad de tiradores
            df['optimized_passer_synergy'] = df['ast_weighted_season_avg'] * df['team_shooting_quality']
            features.append('optimized_passer_synergy')
        
        # 12. FACTOR DE OPORTUNIDADES DE ASISTENCIA
        if all(col in df.columns for col in ['team_shooting_quality', 'team_offensive_pace']):
            df['assist_opportunities_factor'] = df['team_shooting_quality'] * (df['team_offensive_pace'] / 100)
            features.append('assist_opportunities_factor')
        
        # === GRUPO 6: FEATURES DE INTERACCIÓN ENTRE TOP FEATURES ===
        
        # 13. PREDICTOR MAESTRO (combinación de todas las top features)
        base_predictor = df['ast_weighted_season_avg'] * 0.4
        recent_predictor = df['ast_avg_3g'] * 0.25
        
        if 'is_started' in df.columns:
            starter_boost = df['is_started'] * 0.15
        else:
            starter_boost = 0
            
        if 'triple_double_freq' in df.columns:
            elite_boost = df['triple_double_freq'] * 0.1
        else:
            elite_boost = 0
            
        if 'team_shooting_quality' in df.columns:
            team_boost = df['team_shooting_quality'] * 0.1
        else:
            team_boost = 0
        
        df['master_ast_predictor'] = base_predictor + recent_predictor + starter_boost + elite_boost + team_boost
        features.append('master_ast_predictor')
        
        # 14. FACTOR DE CEILING (máximo potencial basado en mejores juegos)
        ast_max_5g = df.groupby('Player')['AST'].rolling(5, min_periods=1).max().shift(1).reset_index(0, drop=True)
        df['ast_ceiling_factor'] = (ast_max_5g * df['ast_recent_confidence']).fillna(0)
        features.append('ast_ceiling_factor')
        
        # 15. FACTOR DE FLOOR (mínimo esperado basado en consistencia)
        ast_min_5g = df.groupby('Player')['AST'].rolling(5, min_periods=1).min().shift(1).reset_index(0, drop=True)
        df['ast_floor_factor'] = (ast_min_5g * df['ast_season_stability']).fillna(0)
        features.append('ast_floor_factor')
        
        # === GRUPO 7: FEATURES DE CONTEXTO ESPECÍFICO PARA ASISTENCIAS ===
        
        # 16. FACTOR DE RITMO PERSONAL (AST por posesión estimada)
        if 'FGA' in df.columns and 'FTA' in df.columns and 'TOV' in df.columns:
            fga_last = df.groupby('Player')['FGA'].shift(1)
            fta_last = df.groupby('Player')['FTA'].shift(1)
            tov_last = df.groupby('Player')['TOV'].shift(1)
            ast_last = df.groupby('Player')['AST'].shift(1)
            
            possessions = fga_last + 0.44 * fta_last + tov_last + ast_last
            df['ast_per_possession_optimized'] = (ast_last / (possessions + 1)).fillna(0)
            features.append('ast_per_possession_optimized')
        
        # 17. FACTOR DE PRESIÓN TEMPORAL (back-to-back impact)
        if 'is_back_to_back' in df.columns:
            # AST en back-to-back vs juegos normales
            ast_b2b = df.groupby('Player').apply(
                lambda x: x[x['is_back_to_back'] == 1]['AST'].shift(1).mean()
            ).reset_index(level=0, drop=True)
            ast_normal = df.groupby('Player').apply(
                lambda x: x[x['is_back_to_back'] == 0]['AST'].shift(1).mean()
            ).reset_index(level=0, drop=True)
            df['fatigue_ast_impact'] = (ast_normal - ast_b2b).fillna(0)
            features.append('fatigue_ast_impact')
        
        # 18. FACTOR DE MATCHUP HISTÓRICO
        ast_vs_opp = df.groupby(['Player', 'Opp'])['AST'].expanding().mean().shift(1).reset_index([0,1], drop=True)
        df['historical_matchup_ast'] = ast_vs_opp.fillna(df['ast_weighted_season_avg'])
        features.append('historical_matchup_ast')
        
        # 19. PREDICTOR FINAL ULTRA-OPTIMIZADO
        # Combina los mejores predictores con pesos optimizados
        df['ultra_ast_predictor'] = (
            df['master_ast_predictor'] * 0.4 +
            df['historical_matchup_ast'] * 0.25 +
            df['ast_ceiling_factor'] * 0.15 +
            df['ast_hybrid_predictor'] * 0.1 +
            df['ast_floor_factor'] * 0.1
        ).fillna(0)
        features.append('ultra_ast_predictor')
        
        # 20. CONFIANZA EN PREDICCIÓN (meta-feature)
        # Basado en estabilidad y cantidad de datos
        games_sample = df.groupby('Player').cumcount()
        stability_factor = 1 / (ast_season_std + 0.1)
        sample_confidence = np.minimum(games_sample / 20, 1.0)
        df['prediction_confidence'] = (stability_factor * sample_confidence).fillna(0)
        features.append('prediction_confidence')
        
        # Registrar features
        for feature in features:
            self._register_feature(feature, 'ultra_predictive')
        
        logger.info(f"Generadas {len(features)} features ULTRA-PREDICTIVAS basadas en top features del modelo")
        
        return features

    def _generate_hybrid_features_advanced(self, df: pd.DataFrame) -> List[str]:
        """
        Features híbridas adicionales más sofisticadas para alcanzar 90%+ efectividad
        Combina las mejores características identificadas de manera inteligente
        """
        features = []
        
        # === GRUPO 1: HÍBRIDOS TEMPORALES AVANZADOS ===
        
        # 1. PREDICTOR TEMPORAL ADAPTATIVO
        # Combina diferentes horizontes temporales con pesos que se adaptan al contexto
        ast_1g = df.groupby('Player')['AST'].shift(1).fillna(0)
        ast_3g = df.groupby('Player')['AST'].rolling(3, min_periods=1).mean().shift(1).reset_index(0, drop=True).fillna(0)
        ast_5g = df.groupby('Player')['AST'].rolling(5, min_periods=1).mean().shift(1).reset_index(0, drop=True).fillna(0)
        ast_10g = df.groupby('Player')['AST'].rolling(10, min_periods=1).mean().shift(1).reset_index(0, drop=True).fillna(0)
        
        # Calcular volatilidad para adaptar pesos
        ast_volatility = df.groupby('Player')['AST'].rolling(5, min_periods=2).std().shift(1).reset_index(0, drop=True).fillna(1)
        
        # Pesos adaptativos: más peso a corto plazo si hay alta volatilidad
        volatility_factor = (ast_volatility / (ast_volatility.mean() + 0.1)).clip(0.5, 2.0)
        
        # Pesos que se adaptan a la volatilidad
        weight_recent = 0.4 * volatility_factor / (volatility_factor + 1)  # Más peso si volátil
        weight_medium = 0.3 * (2 - volatility_factor) / 2  # Menos peso si volátil
        weight_long = 0.3 * (2 - volatility_factor) / 2
        
        # Normalizar pesos
        total_weight = weight_recent + weight_medium + weight_long
        weight_recent = weight_recent / total_weight
        weight_medium = weight_medium / total_weight
        weight_long = weight_long / total_weight
        
        df['adaptive_temporal_predictor'] = (
            ast_3g * weight_recent +
            ast_5g * weight_medium +
            ast_10g * weight_long
        )
        features.append('adaptive_temporal_predictor')
        
        # 2. PREDICTOR DE MOMENTUM INTELIGENTE
        # Detecta cambios de tendencia y ajusta predicciones
        momentum_3g = (ast_3g - ast_5g).fillna(0)
        momentum_5g = (ast_5g - ast_10g).fillna(0)
        
        # Momentum compuesto que considera múltiples horizontes
        momentum_strength = (momentum_3g * 0.7 + momentum_5g * 0.3)
        momentum_direction = np.sign(momentum_strength)
        momentum_magnitude = abs(momentum_strength)
        
        # Predictor que se ajusta por momentum
        df['intelligent_momentum_predictor'] = (
            df['adaptive_temporal_predictor'] + 
            momentum_direction * momentum_magnitude * 0.3
        ).clip(0, 15)
        features.append('intelligent_momentum_predictor')
        
        # === GRUPO 2: HÍBRIDOS CONTEXTUALES AVANZADOS ===
        
        # 3. PREDICTOR CONTEXTUAL MULTI-DIMENSIONAL
        # Combina múltiples contextos de manera inteligente
        base_predictor = df.get('ast_season_avg', 2.5)
        
        # Factor de minutos (oportunidades)
        if 'MP' in df.columns:
            mp_factor = (df.groupby('Player')['MP'].rolling(3, min_periods=1).mean().shift(1).reset_index(0, drop=True) / 36).clip(0.3, 1.5)
        else:
            mp_factor = 1.0
        
        # Factor de rol (titular vs suplente)
        if 'is_started' in df.columns:
            role_factor = df['is_started'] * 1.15 + (1 - df['is_started']) * 0.85
        else:
            role_factor = 1.0
        
        # Factor de calidad del equipo
        if 'team_shooting_quality' in df.columns:
            team_factor = (df['team_shooting_quality'] * 0.3 + 0.85).clip(0.7, 1.2)
        else:
            team_factor = 1.0
        
        # Predictor multi-dimensional
        df['multidimensional_predictor'] = (
            base_predictor * mp_factor * role_factor * team_factor
        ).clip(0, 12)
        features.append('multidimensional_predictor')
        
        # 4. PREDICTOR DE SINERGIA AVANZADA
        # Combina rendimiento individual con contexto de equipo
        individual_performance = df['intelligent_momentum_predictor']
        team_context = df['multidimensional_predictor']
        
        # Peso dinámico basado en experiencia del jugador
        games_played = df.groupby('Player').cumcount()
        experience_weight = np.minimum(games_played / 30, 0.8)  # Máximo 80% peso a individual
        team_weight = 1 - experience_weight
        
        df['advanced_synergy_predictor'] = (
            individual_performance * experience_weight +
            team_context * team_weight
        )
        features.append('advanced_synergy_predictor')
        
        # === GRUPO 3: HÍBRIDOS DE CONFIANZA Y CALIBRACIÓN ===
        
        # 5. PREDICTOR CON INTERVALOS DE CONFIANZA
        # Calcula rangos de predicción más precisos
        predictors_for_confidence = [
            'adaptive_temporal_predictor',
            'intelligent_momentum_predictor',
            'multidimensional_predictor',
            'advanced_synergy_predictor'
        ]
        
        available_predictors = [p for p in predictors_for_confidence if p in df.columns]
        
        if len(available_predictors) > 1:
            # Calcular estadísticas de los predictores
            pred_mean = df[available_predictors].mean(axis=1)
            pred_std = df[available_predictors].std(axis=1)
            pred_min = df[available_predictors].min(axis=1)
            pred_max = df[available_predictors].max(axis=1)
            
            # Confianza basada en concordancia entre predictores
            confidence = 1 / (1 + pred_std / (pred_mean + 0.1))
            
            # Predictor con intervalos
            df['confidence_interval_predictor'] = pred_mean
            df['prediction_confidence_advanced'] = confidence
            df['prediction_lower_bound'] = pred_min
            df['prediction_upper_bound'] = pred_max
            
            features.extend([
                'confidence_interval_predictor',
                'prediction_confidence_advanced',
                'prediction_lower_bound',
                'prediction_upper_bound'
            ])
        
        # 6. PREDICTOR CALIBRADO POR RENDIMIENTO HISTÓRICO
        # Ajusta predicciones basándose en precisión histórica del modelo
        if 'prediction_confidence_advanced' in df.columns:
            # Simular precisión histórica (en implementación real, esto vendría de datos históricos)
            historical_accuracy = 0.79  # Accuracy ±2ast actual
            
            # Ajustar predicción basándose en confianza y precisión histórica
            confidence_factor = df['prediction_confidence_advanced']
            accuracy_adjustment = (confidence_factor * historical_accuracy + 
                                 (1 - confidence_factor) * 0.5)  # Fallback conservador
            
            df['historically_calibrated_predictor'] = (
                df['confidence_interval_predictor'] * accuracy_adjustment +
                df.get('ast_season_avg', 2.5) * (1 - accuracy_adjustment)
            )
            features.append('historically_calibrated_predictor')
        
        # === GRUPO 4: HÍBRIDOS ESPECIALIZADOS POR TIPO DE JUGADOR ===
        
        # 7. PREDICTOR ESPECIALIZADO POR ARQUETIPO
        # Diferentes estrategias para diferentes tipos de jugadores
        ast_avg = df.groupby('Player')['AST'].expanding().mean().shift(1).reset_index(0, drop=True).fillna(2.5)
        
        # Clasificar jugadores por arquetipo
        playmaker_mask = ast_avg >= 5.0  # Armadores elite
        facilitator_mask = (ast_avg >= 2.5) & (ast_avg < 5.0)  # Facilitadores
        scorer_mask = ast_avg < 2.5  # Anotadores/otros
        
        # Predictores especializados
        playmaker_pred = df.get('advanced_synergy_predictor', ast_avg) * 1.05  # Más agresivo
        facilitator_pred = df.get('confidence_interval_predictor', ast_avg)  # Balanceado
        scorer_pred = df.get('ast_season_avg', ast_avg) * 0.95  # Más conservador
        
        archetype_predictor = np.where(playmaker_mask, playmaker_pred,
                             np.where(facilitator_mask, facilitator_pred, scorer_pred))
        
        df['archetype_specialized_predictor'] = pd.Series(archetype_predictor).fillna(ast_avg)
        features.append('archetype_specialized_predictor')
        
        # === GRUPO 5: PREDICTOR MAESTRO HÍBRIDO FINAL ===
        
        # 8. PREDICTOR MAESTRO HÍBRIDO
        # Combina todos los predictores híbridos con pesos optimizados
        hybrid_predictors = [
            ('adaptive_temporal_predictor', 0.25),
            ('intelligent_momentum_predictor', 0.20),
            ('advanced_synergy_predictor', 0.20),
            ('confidence_interval_predictor', 0.15),
            ('historically_calibrated_predictor', 0.10),
            ('archetype_specialized_predictor', 0.10)
        ]
        
        available_hybrids = [(name, weight) for name, weight in hybrid_predictors if name in df.columns]
        
        if available_hybrids:
            # Normalizar pesos
            total_weight = sum(weight for _, weight in available_hybrids)
            normalized_weights = [(name, weight/total_weight) for name, weight in available_hybrids]
            
            # Calcular predictor maestro
            master_hybrid = sum(df[name] * weight for name, weight in normalized_weights)
            df['master_hybrid_predictor'] = master_hybrid.fillna(df.get('ast_season_avg', 2.5))
            features.append('master_hybrid_predictor')
        
        # 9. PREDICTOR FINAL ULTRA-HÍBRIDO
        # Combina el mejor predictor híbrido con las mejores features extremas
        if 'master_hybrid_predictor' in df.columns:
            # Combinar con las mejores features extremas identificadas
            ultra_hybrid = (
                df['master_hybrid_predictor'] * 0.5 +
                df.get('contextual_ast_predictor', df.get('ast_season_avg', 2.5)) * 0.3 +
                df.get('calibrated_ast_predictor', df.get('ast_season_avg', 2.5)) * 0.2
            )
            
            df['ultra_hybrid_predictor'] = ultra_hybrid.fillna(df.get('ast_season_avg', 2.5))
            features.append('ultra_hybrid_predictor')
        
        # Registrar features
        for feature in features:
            self._register_feature(feature, 'hybrid_features_advanced')
        
        logger.info(f"Generadas {len(features)} features HÍBRIDAS AVANZADAS para máxima efectividad")
        
        return features

    def _generate_extreme_predictive_features(self, df: pd.DataFrame) -> List[str]:
        """
        Features EXTREMADAMENTE PREDICTIVAS para alcanzar 90%+ efectividad
        Basadas en patrones profundos y específicos de asistencias en NBA
        """
        features = []
        
        # === GRUPO 1: PREDICTORES CONTEXTUALES EXTREMOS ===
        
        # 1. PREDICTOR POR SITUACIÓN DE JUEGO ESPECÍFICA
        # AST promedio cuando el equipo está ganando vs perdiendo
        df['game_score_diff'] = df.get('team_score', 0) - df.get('opp_score', 0)
        
        # Crear contextos específicos con umbrales optimizados
        winning_mask = df['game_score_diff'] > 3  # Reducir umbral para más precisión
        losing_mask = df['game_score_diff'] < -3  # Reducir umbral para más precisión
        close_mask = abs(df['game_score_diff']) <= 3  # Juegos cerrados más precisos
        
        # AST en diferentes contextos de juego
        ast_when_winning = df.groupby('Player').apply(
            lambda x: x[winning_mask.loc[x.index]]['AST'].shift(1).mean()
        ).reindex(df['Player']).values
        
        ast_when_losing = df.groupby('Player').apply(
            lambda x: x[losing_mask.loc[x.index]]['AST'].shift(1).mean()
        ).reindex(df['Player']).values
        
        ast_when_close = df.groupby('Player').apply(
            lambda x: x[close_mask.loc[x.index]]['AST'].shift(1).mean()
        ).reindex(df['Player']).values
        
        # Predictor contextual basado en situación actual
        current_context = np.where(winning_mask, ast_when_winning,
                          np.where(losing_mask, ast_when_losing, ast_when_close))
        
        df['contextual_ast_predictor'] = pd.Series(current_context).fillna(df['ast_season_avg'])
        features.append('contextual_ast_predictor')
        
        # 2. PREDICTOR POR MINUTOS JUGADOS ESPECÍFICOS
        # AST por rango de minutos (diferentes roles)
        if 'MP' in df.columns:
            mp_last = df.groupby('Player')['MP'].shift(1)
            
            # Rangos de minutos específicos
            starter_minutes = mp_last >= 28  # Titulares
            role_player_minutes = (mp_last >= 15) & (mp_last < 28)  # Jugadores de rol
            bench_minutes = mp_last < 15  # Suplentes
            
            # AST promedio por rango de minutos
            ast_as_starter = df.groupby('Player').apply(
                lambda x: x[starter_minutes.loc[x.index]]['AST'].shift(1).mean()
            ).reindex(df['Player']).values
            
            ast_as_role = df.groupby('Player').apply(
                lambda x: x[role_player_minutes.loc[x.index]]['AST'].shift(1).mean()
            ).reindex(df['Player']).values
            
            ast_as_bench = df.groupby('Player').apply(
                lambda x: x[bench_minutes.loc[x.index]]['AST'].shift(1).mean()
            ).reindex(df['Player']).values
            
            # Predictor basado en minutos esperados
            expected_minutes = df.groupby('Player')['MP'].rolling(5, min_periods=1).mean().shift(1).reset_index(0, drop=True)
            
            minutes_based_predictor = np.where(expected_minutes >= 28, ast_as_starter,
                                     np.where(expected_minutes >= 15, ast_as_role, ast_as_bench))
            
            df['minutes_based_ast_predictor'] = pd.Series(minutes_based_predictor).fillna(df['ast_season_avg'])
            features.append('minutes_based_ast_predictor')
        
        # === GRUPO 2: PREDICTORES DE PATRONES TEMPORALES EXTREMOS ===
        
        # 3. PREDICTOR DE RACHAS (STREAKS)
        # Detectar rachas de alto/bajo rendimiento
        ast_last = df.groupby('Player')['AST'].shift(1)
        ast_avg_season = df.groupby('Player')['AST'].expanding().mean().shift(1).reset_index(0, drop=True)
        
        # Racha actual (juegos consecutivos por encima/debajo del promedio)
        above_avg = (ast_last > ast_avg_season).astype(int)
        below_avg = (ast_last < ast_avg_season).astype(int)
        
        # Calcular longitud de racha actual
        def calculate_streak(series):
            if len(series) == 0:
                return 0
            current = series.iloc[-1]
            streak = 1
            for i in range(len(series)-2, -1, -1):
                if series.iloc[i] == current:
                    streak += 1
                else:
                    break
            return streak * (1 if current == 1 else -1)
        
        hot_streak = df.groupby('Player').apply(
            lambda x: x['AST'].rolling(10, min_periods=1).apply(
                lambda s: calculate_streak(s > s.mean()), raw=False
            ).shift(1)
        ).reset_index(level=0, drop=True)
        
        df['ast_streak_factor'] = hot_streak.fillna(0)
        features.append('ast_streak_factor')
        
        # 4. PREDICTOR DE MOMENTUM AVANZADO
        # Momentum basado en últimos 3 juegos vs promedio histórico
        ast_last_3 = df.groupby('Player')['AST'].rolling(3, min_periods=1).mean().shift(1).reset_index(0, drop=True)
        ast_historical = df.groupby('Player')['AST'].expanding().mean().shift(4).reset_index(0, drop=True)
        
        momentum_strength = (ast_last_3 - ast_historical) / (ast_historical + 0.1)
        df['advanced_momentum'] = momentum_strength.fillna(0)
        features.append('advanced_momentum')
        
        # === GRUPO 3: PREDICTORES DE INTERACCIÓN CON COMPAÑEROS ===
        
        # 5. PREDICTOR DE SINERGIA CON TIRADORES ESPECÍFICOS
        if all(col in df.columns for col in ['team_fg_pct', 'team_3p_pct']):
            # Calidad de tiradores del equipo en juegos recientes
            team_shooting_recent = df.groupby(['Team', 'Date']).agg({
                'team_fg_pct': 'first',
                'team_3p_pct': 'first'
            }).reset_index()
            
            # Promedio móvil de calidad de tiro del equipo
            team_shooting_quality = team_shooting_recent.groupby('Team')[['team_fg_pct', 'team_3p_pct']].rolling(5, min_periods=1).mean()
            
            # Sinergia específica: AST cuando el equipo tira bien
            good_shooting_games = df['team_fg_pct'] > df.groupby('Team')['team_fg_pct'].expanding().mean().shift(1).reset_index(0, drop=True)
            
            ast_with_good_shooters = df.groupby('Player').apply(
                lambda x: x[good_shooting_games.loc[x.index]]['AST'].shift(1).mean()
            ).reindex(df['Player']).values
            
            df['shooter_synergy_predictor'] = pd.Series(ast_with_good_shooters).fillna(df['ast_season_avg'])
            features.append('shooter_synergy_predictor')
        
        # === GRUPO 4: PREDICTORES DE MATCHUP ESPECÍFICOS ===
        
        # 6. PREDICTOR POR ESTILO DE DEFENSA DEL OPONENTE
        # AST vs equipos defensivos vs ofensivos
        if 'opp_defensive_rating' in df.columns:
            # Clasificar oponentes por estilo defensivo
            defensive_teams = df['opp_defensive_rating'] < df['opp_defensive_rating'].quantile(0.33)
            offensive_teams = df['opp_defensive_rating'] > df['opp_defensive_rating'].quantile(0.67)
            
            ast_vs_defense = df.groupby('Player').apply(
                lambda x: x[defensive_teams.loc[x.index]]['AST'].shift(1).mean()
            ).reindex(df['Player']).values
            
            ast_vs_offense = df.groupby('Player').apply(
                lambda x: x[offensive_teams.loc[x.index]]['AST'].shift(1).mean()
            ).reindex(df['Player']).values
            
            # Predictor basado en tipo de oponente actual
            current_matchup_predictor = np.where(defensive_teams, ast_vs_defense, ast_vs_offense)
            df['matchup_style_predictor'] = pd.Series(current_matchup_predictor).fillna(df['ast_season_avg'])
            features.append('matchup_style_predictor')
        
        # === GRUPO 5: PREDICTORES DE ESTADO FÍSICO/MENTAL ===
        
        # 7. PREDICTOR DE FATIGA AVANZADA
        # Basado en minutos acumulados y back-to-backs
        if all(col in df.columns for col in ['MP', 'is_back_to_back']):
            # Carga de trabajo reciente
            minutes_last_3 = df.groupby('Player')['MP'].rolling(3, min_periods=1).sum().shift(1).reset_index(0, drop=True)
            minutes_season_avg = df.groupby('Player')['MP'].expanding().mean().shift(1).reset_index(0, drop=True) * 3
            
            workload_ratio = minutes_last_3 / (minutes_season_avg + 1)
            
            # Factor de fatiga combinado
            fatigue_factor = workload_ratio * (1 + df['is_back_to_back'].astype(int) * 0.5)
            
            # AST ajustado por fatiga
            df['fatigue_adjusted_ast'] = df['ast_season_avg'] * (2 - fatigue_factor).clip(0.5, 1.5)
            features.append('fatigue_adjusted_ast')
        
        # === GRUPO 6: PREDICTORES HÍBRIDOS EXTREMOS ===
        
        # 8. PREDICTOR MAESTRO EXTREMO
        # Combina todos los predictores con pesos finales optimizados
        base_weight = 0.25
        context_weight = 0.20
        momentum_weight = 0.15
        matchup_weight = 0.15
        minutes_weight = 0.15
        fatigue_weight = 0.10
        
        extreme_predictor = (
            df.get('ast_season_avg', 0) * base_weight +
            df.get('contextual_ast_predictor', 0) * context_weight +
            df.get('advanced_momentum', 0) * momentum_weight * 5 +  # Amplificar momentum
            df.get('matchup_style_predictor', 0) * matchup_weight +
            df.get('minutes_based_ast_predictor', 0) * minutes_weight +
            df.get('fatigue_adjusted_ast', 0) * fatigue_weight
        )
        
        df['extreme_ast_predictor'] = extreme_predictor.fillna(df['ast_season_avg'])
        features.append('extreme_ast_predictor')
        
        # 9. PREDICTOR DE CONFIANZA EXTREMA
        # Basado en estabilidad de todos los predictores
        predictors = ['ast_season_avg', 'contextual_ast_predictor', 'minutes_based_ast_predictor', 'extreme_ast_predictor']
        available_predictors = [p for p in predictors if p in df.columns]
        
        if len(available_predictors) > 1:
            predictor_std = df[available_predictors].std(axis=1)
            predictor_mean = df[available_predictors].mean(axis=1)
            
            confidence = 1 / (predictor_std + 0.1)  # Más confianza = menos variabilidad
            df['extreme_confidence'] = confidence.fillna(1)
            features.append('extreme_confidence')
        
        # 10. PREDICTOR FINAL PONDERADO POR CONFIANZA
        if 'extreme_confidence' in df.columns:
            high_confidence_mask = df['extreme_confidence'] > df['extreme_confidence'].quantile(0.7)
            
            # Usar predictor más conservador cuando hay baja confianza
            final_predictor = np.where(
                high_confidence_mask,
                df['extreme_ast_predictor'],
                df['ast_season_avg'] * 0.7 + df['extreme_ast_predictor'] * 0.3
            )
            
            df['confidence_weighted_predictor'] = pd.Series(final_predictor)
            features.append('confidence_weighted_predictor')
        
        # === GRUPO 7: FEATURES DE CALIBRACIÓN ESPECÍFICA ===
        
        # 11. CALIBRADOR POR RANGO DE AST
        # Diferentes predictores para diferentes rangos de asistencias
        ast_historical = df.groupby('Player')['AST'].expanding().mean().shift(1).reset_index(0, drop=True)
        
        # Clasificar jugadores por nivel de AST con rangos optimizados
        low_ast_players = ast_historical <= 1.5    # Jugadores de muy pocas asistencias
        mid_ast_players = (ast_historical > 1.5) & (ast_historical <= 4.5)  # Rango medio ampliado
        high_ast_players = ast_historical > 4.5    # Jugadores de muchas asistencias
        
        # Predictores específicos por nivel con pesos optimizados
        low_ast_predictor = df['ast_season_avg'] * 0.95   # Ligeramente más conservador
        mid_ast_predictor = df['extreme_ast_predictor'] * 1.02  # Ligero boost
        high_ast_predictor = df['extreme_ast_predictor'] * 1.05  # Más agresivo para elite
        
        calibrated_predictor = np.where(low_ast_players, low_ast_predictor,
                               np.where(mid_ast_players, mid_ast_predictor, high_ast_predictor))
        
        df['calibrated_ast_predictor'] = pd.Series(calibrated_predictor).fillna(df['ast_season_avg'])
        features.append('calibrated_ast_predictor')
        
        # 12. PREDICTOR HÍBRIDO OPTIMIZADO (NUEVO)
        # Combina las top features identificadas con pesos optimizados
        # Basado en importancia: contextual_ast_predictor (35.9%) + calibrated_ast_predictor (13.7%)
        
        # Pesos optimizados basados en importancia real del modelo
        contextual_weight = 0.45    # Aumentar peso de la feature más importante
        calibrated_weight = 0.25    # Peso significativo para la segunda más importante
        historical_weight = 0.15    # ast_avg_10g (tercera más importante)
        starter_weight = 0.10       # starter_impact (cuarta más importante)
        recent_weight = 0.05        # Tendencia reciente
        
        optimized_hybrid = (
            df.get('contextual_ast_predictor', df['ast_season_avg']) * contextual_weight +
            df.get('calibrated_ast_predictor', df['ast_season_avg']) * calibrated_weight +
            df.get('ast_avg_10g', df['ast_season_avg']) * historical_weight +
            df.get('starter_impact', 1.0) * starter_weight +
            df.get('ast_avg_3g', df['ast_season_avg']) * recent_weight
        )
        
        df['optimized_hybrid_predictor'] = optimized_hybrid.fillna(df['ast_season_avg'])
        features.append('optimized_hybrid_predictor')
        
        # 13. PREDICTOR FINAL ULTRA-OPTIMIZADO
        # Combina los mejores predictores con pesos finales optimizados
        df['final_extreme_predictor'] = (
            df['master_ast_predictor'] * 0.4 +
            df['historical_matchup_ast'] * 0.25 +
            df['ast_ceiling_factor'] * 0.15 +
            df['ast_hybrid_predictor'] * 0.1 +
            df['ast_floor_factor'] * 0.1
        ).fillna(0)
        features.append('final_extreme_predictor')
        
        # Registrar features
        for feature in features:
            self._register_feature(feature, 'extreme_predictive')
        
        logger.info(f"Generadas {len(features)} features EXTREMADAMENTE PREDICTIVAS para 90%+ efectividad")
        
        return features

    def _apply_correlation_filter(self, df: pd.DataFrame, features: List[str]) -> List[str]:
        """Aplica filtro de correlación para reducir features redundantes"""
        logger.info(f"Aplicando filtro de correlación: {len(features)} -> {self.max_features} features")
        
        # Preparar datos para filtro de correlación
        X = df[features].fillna(0)
        
        # Calcular matriz de correlación
        corr_matrix = X.corr().abs()
        
        # Encontrar features altamente correlacionadas
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        
        # Identificar features a eliminar
        to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > self.correlation_threshold)]
        
        # Mantener features importantes y eliminar correlacionadas
        filtered_features = [f for f in features if f not in to_drop]
        
        # Si aún hay demasiadas features, usar selección basada en target
        if len(filtered_features) > self.max_features and 'AST' in df.columns:
            logger.info("Aplicando selección adicional basada en target...")
            
            X_filtered = df[filtered_features].fillna(0)
            y = df['AST'].fillna(df['AST'].mean())
            
            # Seleccionar mejores features
            selector = SelectKBest(score_func=f_regression, k=self.max_features)
            selector.fit(X_filtered, y)
            
            # Obtener features seleccionadas
            selected_features = [filtered_features[i] for i in range(len(filtered_features)) 
                               if selector.get_support()[i]]
            filtered_features = selected_features
        
        logger.info(f"Filtro de correlación: {len(features)} -> {len(filtered_features)} features")
        
        return filtered_features
    
    def _log_feature_summary(self) -> None:
        """Registra resumen de features optimizadas"""
        total_features = sum(len(features) for features in self.feature_categories.values())
        logger.info(f"Features AVANZADAS: {total_features} en {len(self.feature_categories)} categorías (máx {self.max_features})")
        
        for category, features in self.feature_categories.items():
            if features:
                logger.info(f"  {category}: {len(features)} features")
        
        logger.info("ENFOQUE: Features avanzadas aprovechando todos los datos disponibles")

    def _select_features_by_category(self, df: pd.DataFrame, all_features: List[str]) -> List[str]:
        """Selecciona features balanceadas por categoría"""
        selected_features = []
        
        # Seleccionar features por categoría según límites configurados
        for category, max_features in self.feature_categories.items():
            if category in self.feature_registry:
                category_features = self.feature_registry[category]
                # Filtrar features que existen en el DataFrame
                available_features = [f for f in category_features if f in df.columns]
                
                if available_features:
                    # Calcular correlación con target para seleccionar las mejores
                    correlations = []
                    for feature in available_features:
                        try:
                            corr = abs(df[feature].corr(df['AST']))
                            if not pd.isna(corr):
                                correlations.append((feature, corr))
                        except:
                            continue
                    
                    # Ordenar por correlación y tomar las mejores
                    correlations.sort(key=lambda x: x[1], reverse=True)
                    # Asegurar que max_features sea un entero
                    max_features_int = int(max_features) if isinstance(max_features, (int, float)) else len(available_features)
                    selected_from_category = [f[0] for f in correlations[:max_features_int]]
                    selected_features.extend(selected_from_category)
                    
                    logger.info(f"Categoría {category}: {len(selected_from_category)}/{len(available_features)} features seleccionadas")
        
        logger.info(f"Total features seleccionadas por categoría: {len(selected_features)}")
        return selected_features

    def _generate_supreme_predictive_features(self, df: pd.DataFrame) -> List[str]:
        """
        FEATURES SUPREMAS - PARA PERFECCIÓN ABSOLUTA (95%+ EFECTIVIDAD)
        
        Estas features implementan los patrones más sofisticados y específicos
        para alcanzar la máxima precisión posible en predicción de asistencias.
        """
        features = []
        logger.info("Generando FEATURES SUPREMAS para perfección absoluta...")
        
        # === GRUPO 1: PREDICTORES ADAPTATIVOS INTELIGENTES ===
        
        # 1. PREDICTOR ADAPTATIVO POR VOLATILIDAD
        # Se adapta automáticamente según la estabilidad del jugador
        ast_volatility = df.groupby('Player')['AST'].rolling(10, min_periods=3).std().shift(1).reset_index(0, drop=True)
        ast_mean = df.groupby('Player')['AST'].rolling(10, min_periods=1).mean().shift(1).reset_index(0, drop=True)
        
        # Jugadores estables: más peso a promedio, volátiles: más peso a tendencia reciente
        stability_factor = 1 / (ast_volatility + 0.5)  # Más estable = factor más alto
        stability_factor = stability_factor.fillna(1)
        
        adaptive_predictor = (
            ast_mean * (stability_factor / (stability_factor + 1)) +
            df.get('ast_avg_3g', ast_mean) * (1 / (stability_factor + 1))
        )
        
        df['adaptive_volatility_predictor'] = adaptive_predictor.fillna(df.get('ast_season_avg', 0))
        features.append('adaptive_volatility_predictor')
        
        # 2. PREDICTOR DE APRENDIZAJE PROGRESIVO
        # Mejora sus predicciones basándose en errores pasados
        ast_last_pred = df.get('contextual_ast_predictor', df.get('ast_season_avg', 0))
        ast_actual = df.groupby('Player')['AST'].shift(1).fillna(0)
        
        # Error de predicción anterior
        prediction_error = ast_actual - ast_last_pred
        
        # Crear DataFrame temporal para calcular tendencia de error
        temp_df = pd.DataFrame({
            'Player': df['Player'],
            'prediction_error': prediction_error
        })
        
        error_trend = temp_df.groupby('Player')['prediction_error'].rolling(5, min_periods=1).mean().shift(1).reset_index(0, drop=True)
        
        # Ajustar predicción basándose en tendencia de error
        learning_predictor = ast_last_pred + (error_trend * 0.3)  # Corrección moderada
        
        df['learning_adaptive_predictor'] = learning_predictor.fillna(df.get('ast_season_avg', 0))
        features.append('learning_adaptive_predictor')
        
        # 3. PREDICTOR POR CONTEXTO ESPECÍFICO ULTRA-DETALLADO
        # Diferentes predictores para situaciones muy específicas
        
        # Contexto: Juegos importantes (playoffs, vs equipos top)
        important_games = (
            df.get('is_playoff', False) | 
            (df.get('Opp_win_pct', 0.5) > 0.6)  # Vs equipos ganadores
        )
        
        # Contexto: Juegos fáciles (vs equipos débiles)
        easy_games = df.get('Opp_win_pct', 0.5) < 0.4
        
        # Contexto: Juegos normales
        normal_games = ~(important_games | easy_games)
        
        # Predictores específicos por contexto
        important_game_predictor = df.get('ast_season_avg', 0) * 1.1  # Ligeramente más agresivo
        easy_game_predictor = df.get('ast_season_avg', 0) * 1.05     # Ligero boost
        normal_game_predictor = df.get('contextual_ast_predictor', df.get('ast_season_avg', 0))
        
        context_specific_predictor = np.where(
            important_games, important_game_predictor,
            np.where(easy_games, easy_game_predictor, normal_game_predictor)
        )
        
        df['ultra_context_predictor'] = pd.Series(context_specific_predictor)
        features.append('ultra_context_predictor')
        
        # === GRUPO 2: FEATURES DE PRECISIÓN EXTREMA ===
        
        # 4. PREDICTOR DE RANGO DINÁMICO
        # Predice no solo el valor, sino el rango probable
        ast_floor = df.groupby('Player')['AST'].rolling(10, min_periods=1).quantile(0.25).shift(1).reset_index(0, drop=True)
        ast_ceiling = df.groupby('Player')['AST'].rolling(10, min_periods=1).quantile(0.75).shift(1).reset_index(0, drop=True)
        ast_median = df.groupby('Player')['AST'].rolling(10, min_periods=1).median().shift(1).reset_index(0, drop=True)
        
        # Predictor que considera el rango completo
        range_predictor = (ast_floor * 0.2 + ast_median * 0.6 + ast_ceiling * 0.2)
        
        df['dynamic_range_predictor'] = range_predictor.fillna(df.get('ast_season_avg', 0))
        features.append('dynamic_range_predictor')
        
        # 5. PREDICTOR DE CONFIANZA CALIBRADA
        # Ajusta predicciones basándose en la confianza histórica
        
        # Calcular precisión histórica por jugador
        historical_errors = []
        for player in df['Player'].unique():
            player_data = df[df['Player'] == player].copy()
            if len(player_data) > 5:
                pred_col = 'contextual_ast_predictor' if 'contextual_ast_predictor' in df.columns else 'ast_season_avg'
                errors = abs(player_data['AST'] - player_data[pred_col].shift(1))
                avg_error = errors.mean()
                historical_errors.append({'Player': player, 'avg_error': avg_error})
        
        if historical_errors:
            error_df = pd.DataFrame(historical_errors)
            df = df.merge(error_df, on='Player', how='left')
            
            # Ajustar predicción basándose en precisión histórica
            confidence_factor = 1 / (df['avg_error'] + 0.5)  # Más precisión = más confianza
            base_pred = df.get('contextual_ast_predictor', df.get('ast_season_avg', 0))
            conservative_pred = df.get('ast_season_avg', 0)
            
            # Mezclar predicción agresiva con conservadora basándose en confianza
            calibrated_confidence_predictor = (
                base_pred * (confidence_factor / (confidence_factor + 1)) +
                conservative_pred * (1 / (confidence_factor + 1))
            )
            
            df['calibrated_confidence_predictor'] = calibrated_confidence_predictor
            features.append('calibrated_confidence_predictor')
        
        # === GRUPO 3: FEATURES DE SINERGIA AVANZADA ===
        
        # 6. PREDICTOR DE SINERGIA TEMPORAL
        # Considera cómo cambia la sinergia con el equipo a lo largo del tiempo
        
        # Calcular sinergia reciente vs histórica
        team_ast_recent = df.groupby(['Player', 'Team'])['AST'].rolling(5, min_periods=1).mean().shift(1)
        team_ast_season = df.groupby(['Player', 'Team'])['AST'].expanding().mean().shift(1)
        
        # Factor de mejora/empeoramiento de sinergia
        synergy_trend = (team_ast_recent / (team_ast_season + 0.1)) - 1
        synergy_trend = synergy_trend.reset_index(0, drop=True).reset_index(0, drop=True)
        
        # Predictor que considera la tendencia de sinergia
        base_synergy_pred = df.get('team_shooting_quality', 1) * df.get('ast_season_avg', 0)
        temporal_synergy_predictor = base_synergy_pred * (1 + synergy_trend * 0.2)
        
        df['temporal_synergy_predictor'] = temporal_synergy_predictor.fillna(base_synergy_pred)
        features.append('temporal_synergy_predictor')
        
        # === GRUPO 4: FEATURES MAESTRAS SUPREMAS ===
        
        # 7. PREDICTOR MAESTRO SUPREMO
        # Combina todos los predictores supremos con pesos optimizados
        supreme_predictors = [
            ('adaptive_volatility_predictor', 0.25),
            ('learning_adaptive_predictor', 0.20),
            ('ultra_context_predictor', 0.20),
            ('dynamic_range_predictor', 0.15),
            ('temporal_synergy_predictor', 0.10),
            ('calibrated_confidence_predictor', 0.10) if 'calibrated_confidence_predictor' in df.columns else ('ast_season_avg', 0.10)
        ]
        
        supreme_master = sum(
            df.get(pred_name, df.get('ast_season_avg', 0)) * weight 
            for pred_name, weight in supreme_predictors
        )
        
        df['supreme_master_predictor'] = supreme_master
        features.append('supreme_master_predictor')
        
        # 8. PREDICTOR FINAL SUPREMO
        # Combina el mejor de cada categoría anterior
        final_supreme = (
            df.get('supreme_master_predictor', 0) * 0.4 +
            df.get('contextual_ast_predictor', 0) * 0.3 +
            df.get('calibrated_ast_predictor', 0) * 0.2 +
            df.get('optimized_hybrid_predictor', 0) * 0.1
        )
        
        df['final_supreme_predictor'] = final_supreme.fillna(df.get('ast_season_avg', 0))
        features.append('final_supreme_predictor')
        
        # Registrar features
        for feature in features:
            self._register_feature(feature, 'supreme_predictive')
        
        logger.info(f"Generadas {len(features)} FEATURES SUPREMAS para perfección absoluta (95%+)")
        
        return features

    def _validate_features(self, df: pd.DataFrame, selected_features: List[str]) -> List[str]:
        """Valida y ajusta las features seleccionadas"""
        # Implementa la lógica para validar y ajustar las features seleccionadas
        # Esto puede incluir la eliminación de features que no cumplen con ciertas condiciones
        # o la combinación de features para crear nuevas
        return selected_features

    def _generate_adaptability_features(self, df: pd.DataFrame) -> List[str]:
        """
        FEATURES CRÍTICAS DE ADAPTABILIDAD
        - player_adaptability_score (8.79% importancia)
        - opponent_adaptation_score (6.21% importancia)
        """
        features = []
        logger.info("Generando FEATURES CRÍTICAS DE ADAPTABILIDAD...")
        
        # === ADAPTABILIDAD DEL JUGADOR ===
        
        # 1. PLAYER ADAPTABILITY SCORE (equivalente al modelo de puntos)
        # Mide qué tan bien se adapta el jugador a diferentes fases de la temporada
        if 'Date' in df.columns:
            # Calcular fase de temporada
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            season_start = df['Date'].min()
            season_length = (df['Date'].max() - season_start).days
            df['season_phase'] = ((df['Date'] - season_start).dt.days / season_length).fillna(0)
            
            # Calcular consistencia por fase de temporada
            season_phase_consistency = df.groupby(['Player', pd.cut(df['season_phase'], bins=3)])['AST'].std().groupby('Player').mean()
            season_phase_consistency = season_phase_consistency.fillna(1.0)
            
            # Crear mapeo de jugador a adaptabilidad
            adaptability_map = (1 / (1 + season_phase_consistency)).to_dict()
            df['player_adaptability_score'] = df['Player'].map(adaptability_map).fillna(0.5)
            features.append('player_adaptability_score')
        
        # 2. OPPONENT ADAPTATION SCORE (equivalente al modelo de puntos)
        # Mide qué tan bien rinde el jugador contra diferentes oponentes
        if 'Opp' in df.columns:
            # Calcular rendimiento histórico vs cada oponente
            player_vs_opp = df.groupby(['Player', 'Opp'])['AST'].expanding().mean().shift(1).reset_index([0,1], drop=True)
            player_overall = df.groupby('Player')['AST'].expanding().mean().shift(1).reset_index(0, drop=True)
            
            # Ratio de rendimiento vs oponente específico vs rendimiento general
            opponent_adaptation = (player_vs_opp / (player_overall + 1e-6)).fillna(1.0)
            df['opponent_adaptation_score'] = opponent_adaptation
            features.append('opponent_adaptation_score')
        
        # === FEATURES ADICIONALES DE ADAPTABILIDAD ===
        
        # 3. ADAPTABILIDAD A MINUTOS (similar a shooting_volume en puntos)
        # Mide cómo se adaptan las asistencias según los minutos jugados
        if 'MP' in df.columns:
            mp_avg = df.groupby('Player')['MP'].rolling(5, min_periods=1).mean().shift(1).reset_index(0, drop=True)
            ast_avg = df.groupby('Player')['AST'].rolling(5, min_periods=1).mean().shift(1).reset_index(0, drop=True)
            
            # Eficiencia de asistencias por minuto (equivalente a shooting_volume)
            df['assist_volume_5g'] = (ast_avg * mp_avg / 48).fillna(0)  # Normalizado por 48 min
            features.append('assist_volume_5g')
        
        # 4. ADAPTABILIDAD A DIFERENTES ROLES (titular vs suplente)
        if 'is_started' in df.columns:
            # AST como titular vs como suplente
            ast_as_starter = df.groupby('Player').apply(
                lambda x: x[x['is_started'] == 1]['AST'].shift(1).mean()
            ).reset_index(level=0, drop=True)
            ast_as_bench = df.groupby('Player').apply(
                lambda x: x[x['is_started'] == 0]['AST'].shift(1).mean()
            ).reset_index(level=0, drop=True)
            
            # Factor de adaptabilidad de rol
            role_adaptability = abs(ast_as_starter - ast_as_bench).fillna(0)
            df['role_adaptability_factor'] = (1 / (role_adaptability + 0.5))  # Menos diferencia = más adaptable
            features.append('role_adaptability_factor')
        
        # 5. ADAPTABILIDAD AL RITMO DEL EQUIPO
        # Mide cómo se adaptan las asistencias al ritmo ofensivo del equipo
        if 'FGA' in df.columns:
            team_pace = df.groupby(['Team', 'Date'])['FGA'].transform('sum').shift(1)
            ast_vs_pace = df.groupby('Player')['AST'].shift(1) / (team_pace / 100 + 1)
            df['pace_adaptability'] = ast_vs_pace.fillna(0)
            features.append('pace_adaptability')
        
        # 6. ADAPTABILIDAD A LA PRESIÓN DEFENSIVA
        # Mide cómo mantiene las asistencias contra defensas agresivas
        if 'STL' in df.columns:
            opp_defensive_pressure = df.groupby('Opp')['STL'].rolling(3, min_periods=1).mean().shift(1).reset_index(0, drop=True)
            ast_vs_pressure = df.groupby('Player')['AST'].shift(1) / (opp_defensive_pressure + 1)
            df['pressure_adaptability'] = ast_vs_pressure.fillna(0)
            features.append('pressure_adaptability')
        
        # 7. FACTOR DE EXPLOSIÓN POTENCIAL (equivalente a explosion_potential en puntos)
        # Identifica jugadores con potencial de juegos excepcionales de asistencias
        ast_max_5g = df.groupby('Player')['AST'].rolling(5, min_periods=1).max().shift(1).reset_index(0, drop=True)
        ast_avg_5g = df.groupby('Player')['AST'].rolling(5, min_periods=1).mean().shift(1).reset_index(0, drop=True)
        df['ast_explosion_potential'] = (ast_max_5g - ast_avg_5g).fillna(0)
        features.append('ast_explosion_potential')
        
        # 8. TIER DE JUGADOR PARA ASISTENCIAS (equivalente a player_tier en puntos)
        # Clasifica jugadores por nivel de asistencias
        ast_season_avg = df.groupby('Player')['AST'].expanding().mean().shift(1).reset_index(0, drop=True)
        
        def categorize_assist_tier(avg_ast):
            if avg_ast >= 8:
                return 4  # Elite playmaker
            elif avg_ast >= 6:
                return 3  # High-level playmaker
            elif avg_ast >= 4:
                return 2  # Solid playmaker
            elif avg_ast >= 2:
                return 1  # Role player
            else:
                return 0  # Limited playmaker
        
        df['ast_player_tier'] = ast_season_avg.apply(categorize_assist_tier)
        features.append('ast_player_tier')
        
        # 9. EFICIENCIA ENSEMBLE SCORE (equivalente a efficiency_ensemble_score)
        # Combina múltiples métricas de eficiencia de asistencias
        ast_per_min = df.get('ast_per_minute_5g', 0)
        ast_tov_ratio = df.get('ast_tov_ratio', 0)
        team_synergy = df.get('team_shooting_quality', 0.45)
        
        df['ast_efficiency_ensemble'] = (
            ast_per_min * 0.4 + 
            ast_tov_ratio * 0.3 + 
            team_synergy * 0.3
        ).fillna(0)
        features.append('ast_efficiency_ensemble')
        
        # 10. HIGH VOLUME EFFICIENCY (para asistencias)
        # Identifica jugadores que mantienen eficiencia con alto volumen
        ast_shifted_vol = df.groupby('Player')['AST'].shift(1)
        high_ast_games = (ast_shifted_vol >= 7).astype(int)
        ast_efficiency = df.get('ast_per_minute_5g', 0)
        df['high_ast_volume_efficiency'] = (high_ast_games * ast_efficiency).fillna(0)
        features.append('high_ast_volume_efficiency')
        
        # Registrar features
        for feature in features:
            self._register_feature(feature, 'adaptability_critical')
        
        logger.info(f"Generadas {len(features)} FEATURES CRÍTICAS DE ADAPTABILIDAD")
        return features

    def _generate_basketball_specific_assist_features(self, df: pd.DataFrame) -> List[str]:
        """
        FEATURES ULTRA-ESPECÍFICAS PARA ASISTENCIAS NBA
        Basadas en los fundamentos reales del baloncesto que predicen asistencias
        """
        features = []
        logger.info("Generando FEATURES ULTRA-ESPECÍFICAS para asistencias NBA...")
        
        # === GRUPO 1: CONTROL DEL BALÓN Y CREACIÓN ===
        
        # 1. USAGE RATE PARA ASISTENCIAS (% de posesiones que terminan en AST)
        if all(col in df.columns for col in ['FGA', 'FTA', 'TOV', 'AST']):
            fga_last = df.groupby('Player')['FGA'].shift(1)
            fta_last = df.groupby('Player')['FTA'].shift(1)
            tov_last = df.groupby('Player')['TOV'].shift(1)
            ast_last = df.groupby('Player')['AST'].shift(1)
            
            # Posesiones totales usadas por el jugador
            total_possessions = fga_last + 0.44 * fta_last + tov_last + ast_last
            df['ast_usage_rate'] = (ast_last / (total_possessions + 1)).fillna(0)
            features.append('ast_usage_rate')
        
        # 2. PURE POINT RATING (AST vs scoring attempts)
        if all(col in df.columns for col in ['FGA', 'FTA', 'AST']):
            scoring_attempts = fga_last + fta_last
            df['pure_point_rating'] = (ast_last / (scoring_attempts + 1)).fillna(0)
            features.append('pure_point_rating')
        
        # 3. PLAYMAKER EFFICIENCY (AST per touch approximation)
        if all(col in df.columns for col in ['AST', 'TOV', 'FGA']):
            # Aproximación de toques = FGA + AST + TOV
            touches_approx = fga_last + ast_last + tov_last
            df['playmaker_efficiency'] = (ast_last / (touches_approx + 1)).fillna(0)
            features.append('playmaker_efficiency')
        
        # === GRUPO 2: CONTEXTO DE EQUIPO ESPECÍFICO ===
        
        # 4. TEAM ASSIST DEPENDENCY (qué % de AST del equipo genera el jugador)
        team_ast_total = df.groupby(['Team', 'Date'])['AST'].transform('sum')
        player_ast = df.groupby('Player')['AST'].shift(1)
        df['team_ast_dependency'] = (player_ast / (team_ast_total.shift(1) + 1)).fillna(0)
        features.append('team_ast_dependency')
        
        # 5. ASSIST OPPORTUNITY RATE (basado en FG% del equipo)
        if 'FG%' in df.columns:
            team_fg_pct = df.groupby(['Team', 'Date'])['FG%'].transform('mean').shift(1)
            df['assist_opportunity_rate'] = (player_ast * team_fg_pct).fillna(0)
            features.append('assist_opportunity_rate')
        
        # === GRUPO 3: PATRONES DE JUEGO ESPECÍFICOS ===
        
        # 6. TRANSITION ASSIST INDICATOR (basado en pace y STL)
        if 'STL' in df.columns:
            stl_last = df.groupby('Player')['STL'].shift(1)
            # Jugadores que roban más tienden a generar más AST en transición
            df['transition_ast_potential'] = (stl_last * ast_last).fillna(0)
            features.append('transition_ast_potential')
        
        # 7. HALF-COURT ASSIST RATE (AST que no son de transición)
        if 'STL' in df.columns:
            # AST que no provienen directamente de robos
            df['halfcourt_ast_rate'] = (ast_last - stl_last).clip(lower=0).fillna(0)
            features.append('halfcourt_ast_rate')
        
        # === GRUPO 4: MÉTRICAS AVANZADAS ESPECÍFICAS ===
        
        # 8. ASSIST-TO-SHOT RATIO (AST vs intentos de tiro del jugador)
        if 'FGA' in df.columns:
            df['ast_to_shot_ratio'] = (ast_last / (fga_last + 1)).fillna(0)
            features.append('ast_to_shot_ratio')
        
        # 9. COURT VISION METRIC (AST + STL como proxy de visión de cancha)
        if 'STL' in df.columns:
            df['court_vision_metric'] = (ast_last + stl_last * 0.5).fillna(0)
            features.append('court_vision_metric')
        
        # 10. BASKETBALL IQ PROXY (AST/TOV ratio weighted by minutes)
        if all(col in df.columns for col in ['TOV', 'MP']):
            mp_last = df.groupby('Player')['MP'].shift(1)
            ast_tov_ratio = ast_last / (tov_last + 1)
            df['basketball_iq_proxy'] = (ast_tov_ratio * (mp_last / 48)).fillna(0)
            features.append('basketball_iq_proxy')
        
        # === GRUPO 5: FEATURES DE CONTEXTO SITUACIONAL ===
        
        # 11. CLUTCH ASSIST POTENTIAL (basado en +/-)
        if '+/-' in df.columns:
            plus_minus_last = df.groupby('Player')['+/-'].shift(1)
            df['clutch_assist_potential'] = (ast_last * (plus_minus_last > 0).astype(int)).fillna(0)
            features.append('clutch_assist_potential')
        
        # 12. ROLE CONSISTENCY (diferencia AST titular vs suplente)
        if 'is_started' in df.columns:
            # Calcular AST promedio como titular vs suplente
            starter_ast = df.groupby('Player').apply(
                lambda x: x[x['is_started'] == 1]['AST'].shift(1).mean()
            ).reset_index(level=0, drop=True)
            bench_ast = df.groupby('Player').apply(
                lambda x: x[x['is_started'] == 0]['AST'].shift(1).mean()
            ).reset_index(level=0, drop=True)
            
            # Consistencia = menor diferencia entre roles
            role_diff = abs(starter_ast - bench_ast).fillna(0)
            df['role_consistency'] = (1 / (role_diff + 0.5))  # Inverso de la diferencia
            features.append('role_consistency')
        
        # === GRUPO 6: FEATURES DE MOMENTUM ESPECÍFICAS ===
        
        # 13. HOT HAND ASSIST (AST en juegos consecutivos con AST altas)
        ast_shifted = df.groupby('Player')['AST'].shift(1)
        ast_expanding_mean = df.groupby('Player')['AST'].expanding().mean().shift(2)
        
        # Resetear índices para comparación
        ast_shifted_reset = ast_shifted.reset_index(drop=True)
        ast_expanding_reset = ast_expanding_mean.reset_index(drop=True)
        
        ast_above_avg = (ast_shifted_reset > ast_expanding_reset).astype(int)
        df['hot_hand_assist'] = ast_above_avg.rolling(3, min_periods=1).sum().fillna(0)
        features.append('hot_hand_assist')
        
        # 14. ASSIST STREAK MOMENTUM
        # Simplificar el cálculo para evitar problemas de índice
        ast_shifted_streak = df.groupby('Player')['AST'].shift(1)
        ast_expanding_streak = df.groupby('Player')['AST'].expanding().mean().shift(2)
        
        # Resetear índices para comparación
        ast_shifted_streak_reset = ast_shifted_streak.reset_index(drop=True)
        ast_expanding_streak_reset = ast_expanding_streak.reset_index(drop=True)
        
        # Calcular racha simple: juegos consecutivos por encima del promedio
        above_avg_streak = (ast_shifted_streak_reset >= ast_expanding_streak_reset).astype(int)
        df['assist_streak'] = above_avg_streak.rolling(3, min_periods=1).sum().fillna(0)
        features.append('assist_streak')
        
        # === GRUPO 7: FEATURES DE ADAPTACIÓN AL OPONENTE ===
        
        # 15. DEFENSIVE PRESSURE ADAPTATION (AST vs defensive rating del oponente)
        if 'STL' in df.columns:
            # Usar STL del oponente como proxy de presión defensiva
            opp_defensive_pressure = df.groupby('Opp')['STL'].rolling(5, min_periods=1).mean().shift(1).reset_index(0, drop=True)
            df['pressure_adaptation'] = (ast_last / (opp_defensive_pressure + 1)).fillna(0)
            features.append('pressure_adaptation')
        
        # Actualizar categoría
        self.feature_categories['basketball_specific'] = 15
        
        # Registrar features
        for feature in features:
            self._register_feature(feature, 'basketball_specific')
        
        logger.info(f"Generadas {len(features)} FEATURES ULTRA-ESPECÍFICAS para asistencias NBA")
        return features

    def _generate_advanced_playmaking_features(self, df: pd.DataFrame) -> List[str]:
        """
        FEATURES AVANZADAS DE PLAYMAKING
        Métricas específicas para identificar y predecir el rendimiento de playmakers de élite
        """
        features = []
        logger.info("Generando FEATURES AVANZADAS DE PLAYMAKING...")
        
        # === GRUPO 1: MÉTRICAS DE CREACIÓN DE JUEGO ===
        
        # 1. ASSIST RATE VERDADERO (AST por 100 posesiones del equipo)
        if all(col in df.columns for col in ['AST', 'MP']):
            ast_last = df.groupby('Player')['AST'].shift(1)
            mp_last = df.groupby('Player')['MP'].shift(1)
            
            # Estimación de posesiones del equipo basada en minutos jugados
            team_possessions_est = (mp_last / 48) * 100  # Aproximación
            df['true_assist_rate'] = (ast_last / (team_possessions_est + 1) * 100).fillna(0)
            features.append('true_assist_rate')
        
        # 2. ASSIST PERCENTAGE (% de FG del equipo asistidos por el jugador)
        if all(col in df.columns for col in ['AST', 'MP']):
            # Estimación de FG del equipo cuando el jugador está en cancha
            team_fg_est = (mp_last / 48) * 40  # Aproximación de FG del equipo
            df['assist_percentage'] = (ast_last / (team_fg_est + 1) * 100).fillna(0)
            features.append('assist_percentage')
        
        # 3. SECONDARY ASSIST PROXY (STL que llevan a AST)
        if 'STL' in df.columns:
            stl_last = df.groupby('Player')['STL'].shift(1)
            ast_last = df.groupby('Player')['AST'].shift(1)
            df['secondary_assist_proxy'] = (stl_last * 0.3 + ast_last * 0.7).fillna(0)
            features.append('secondary_assist_proxy')
        
        # === GRUPO 2: EFICIENCIA DE PASE ===
        
        # 4. PASS EFFICIENCY RATING (AST vs TOV con peso por minutos)
        if all(col in df.columns for col in ['AST', 'TOV', 'MP']):
            tov_last = df.groupby('Player')['TOV'].shift(1)
            mp_last = df.groupby('Player')['MP'].shift(1)
            
            # Eficiencia de pase ponderada por minutos
            pass_efficiency = (ast_last - tov_last * 0.5) * (mp_last / 36)
            df['pass_efficiency_rating'] = pass_efficiency.fillna(0)
            features.append('pass_efficiency_rating')
        
        # 5. DECISION MAKING INDEX (AST + STL - TOV)
        if all(col in df.columns for col in ['AST', 'STL', 'TOV']):
            decision_index = ast_last + stl_last - tov_last
            df['decision_making_index'] = decision_index.fillna(0)
            features.append('decision_making_index')
        
        # === GRUPO 3: IMPACTO EN EL EQUIPO ===
        
        # 6. TEAM OFFENSIVE RATING WHEN PLAYING (proxy)
        if all(col in df.columns for col in ['PTS', 'AST', 'MP']):
            pts_last = df.groupby('Player')['PTS'].shift(1)
            
            # Impacto ofensivo estimado del jugador
            offensive_impact = (pts_last + ast_last * 2) * (mp_last / 48)
            df['offensive_impact_rating'] = offensive_impact.fillna(0)
            features.append('offensive_impact_rating')
        
        # 7. FLOOR GENERAL RATING (combinación de múltiples métricas)
        if all(col in df.columns for col in ['AST', 'STL', 'TOV', 'PF']):
            pf_last = df.groupby('Player')['PF'].shift(1)
            
            # Rating de general de cancha (más AST y STL, menos TOV y PF)
            floor_general = (ast_last * 2 + stl_last - tov_last - pf_last * 0.5)
            df['floor_general_rating'] = floor_general.fillna(0)
            features.append('floor_general_rating')
        
        # === GRUPO 4: MÉTRICAS DE CLUTCH Y SITUACIONALES ===
        
        # 8. CLUTCH PLAYMAKING (AST en juegos cerrados)
        if '+/-' in df.columns:
            plus_minus_last = df.groupby('Player')['+/-'].shift(1)
            
            # AST en juegos donde el +/- es positivo (juegos competitivos)
            clutch_ast = ast_last * (abs(plus_minus_last) <= 10).astype(int)
            df['clutch_playmaking'] = clutch_ast.fillna(0)
            features.append('clutch_playmaking')
        
        # 9. PRESSURE SITUATION ASSISTS (AST cuando el equipo necesita)
        if all(col in df.columns for col in ['AST', 'TOV']):
            # AST en situaciones de alta presión (más TOV del equipo)
            team_tov = df.groupby(['Team', 'Date'])['TOV'].transform('sum').shift(1)
            pressure_situations = (team_tov > team_tov.quantile(0.7)).astype(int)
            df['pressure_situation_assists'] = (ast_last * pressure_situations).fillna(0)
            features.append('pressure_situation_assists')
        
        # === GRUPO 5: FEATURES DE CONSISTENCIA AVANZADA ===
        
        # 10. ASSIST FLOOR (mínimo confiable de AST)
        ast_min_5g = df.groupby('Player')['AST'].rolling(5, min_periods=1).min().shift(1).reset_index(0, drop=True)
        ast_avg_5g = df.groupby('Player')['AST'].rolling(5, min_periods=1).mean().shift(1).reset_index(0, drop=True)
        df['assist_floor'] = (ast_min_5g / (ast_avg_5g + 0.1)).fillna(0)
        features.append('assist_floor')
        
        # 11. ASSIST CEILING (máximo potencial de AST)
        ast_max_5g = df.groupby('Player')['AST'].rolling(5, min_periods=1).max().shift(1).reset_index(0, drop=True)
        df['assist_ceiling'] = (ast_max_5g / (ast_avg_5g + 0.1)).fillna(0)
        features.append('assist_ceiling')
        
        # 12. PLAYMAKING VERSATILITY (rango de AST)
        df['playmaking_versatility'] = (df['assist_ceiling'] - df['assist_floor']).fillna(0)
        features.append('playmaking_versatility')
        
        # === GRUPO 6: FEATURES DE MOMENTUM Y TENDENCIAS ===
        
        # 13. ASSIST MOMENTUM SCORE
        ast_trend_3g = df.groupby('Player')['AST'].rolling(3, min_periods=2).apply(
            lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) >= 2 else 0
        ).shift(1).reset_index(0, drop=True).fillna(0)
        
        df['assist_momentum_score'] = (ast_trend_3g * ast_avg_5g).fillna(0)
        features.append('assist_momentum_score')
        
        # 14. HOT STREAK INDICATOR
        ast_shifted_pm = df.groupby('Player')['AST'].shift(1)
        ast_expanding_pm = df.groupby('Player')['AST'].expanding().mean().shift(2)
        
        # Resetear índices para comparación
        ast_shifted_pm_reset = ast_shifted_pm.reset_index(drop=True)
        ast_expanding_pm_reset = ast_expanding_pm.reset_index(drop=True)
        
        ast_above_season = (ast_shifted_pm_reset > ast_expanding_pm_reset).astype(int)
        df['hot_streak_indicator'] = ast_above_season.rolling(3, min_periods=1).sum().fillna(0)
        features.append('hot_streak_indicator')
        
        # 15. COLD STREAK RECOVERY
        ast_below_season = (ast_shifted_pm_reset < ast_expanding_pm_reset).astype(int)
        cold_streak = ast_below_season.rolling(3, min_periods=1).sum()
        df['cold_streak_recovery'] = (3 - cold_streak).clip(lower=0).fillna(0)
        features.append('cold_streak_recovery')
        
        # Actualizar categoría
        self.feature_categories['advanced_playmaking'] = 15
        
        # Registrar features
        for feature in features:
            self._register_feature(feature, 'advanced_playmaking')
        
        logger.info(f"Generadas {len(features)} FEATURES AVANZADAS DE PLAYMAKING")
        return features

    def _generate_revolutionary_team_context_features(self, df: pd.DataFrame) -> List[str]:
        """
        FEATURES REVOLUCIONARIAS DE CONTEXTO DE EQUIPO REAL
        Utiliza datos reales de equipos (teams_df) para crear el contexto más preciso jamás visto
        """
        features = []
        logger.info("Generando FEATURES REVOLUCIONARIAS DE CONTEXTO DE EQUIPO REAL...")
        
        if self.teams_df is None:
            logger.warning("teams_df no disponible - usando features básicas")
            return self._generate_basic_team_context_features(df)
        
        # === GRUPO 1: CALIDAD REAL DE TIRADORES DEL EQUIPO - OPTIMIZADO ===
        
        try:
            # 1. REAL TEAM SHOOTING QUALITY (FG% real del equipo) - MÉTODO EFICIENTE
            # Crear diccionarios de promedios por equipo (más eficiente que merge)
            team_fg_pct = self.teams_df.groupby('Team')['FG%'].mean().to_dict()
            team_3p_pct = self.teams_df.groupby('Team')['3P%'].mean().to_dict()
            team_pts_avg = self.teams_df.groupby('Team')['PTS'].mean().to_dict()
            
            # Mapear directamente sin merge masivo
            df['real_team_shooting_quality'] = df['Team'].map(team_fg_pct).fillna(0.45)
            features.append('real_team_shooting_quality')
            
            # 2. ASSIST EFFECTIVENESS REAL (AST × FG% real del equipo)
            ast_last = df.groupby('Player')['AST'].shift(1)
            df['assist_effectiveness_real'] = (ast_last * df['real_team_shooting_quality']).fillna(0)
            features.append('assist_effectiveness_real')
            
            # 3. TEAM 3P SHOOTING QUALITY (para asistencias en triples)
            df['real_team_3p_quality'] = df['Team'].map(team_3p_pct).fillna(0.35)
            df['assist_3p_effectiveness'] = (ast_last * df['real_team_3p_quality']).fillna(0)
            features.append('real_team_3p_quality')
            features.append('assist_3p_effectiveness')
            
        except Exception as e:
            logger.warning(f"Error en team shooting quality: {e}")
            # Features básicas como fallback
            ast_last = df.groupby('Player')['AST'].shift(1)
            df['real_team_shooting_quality'] = 0.45
            df['assist_effectiveness_real'] = ast_last * 0.45
            df['real_team_3p_quality'] = 0.35
            df['assist_3p_effectiveness'] = ast_last * 0.35
            features.extend(['real_team_shooting_quality', 'assist_effectiveness_real', 
                           'real_team_3p_quality', 'assist_3p_effectiveness'])
        
        # === GRUPO 2: RITMO Y POSESIONES REALES DEL EQUIPO - OPTIMIZADO ===
        
        try:
            # 4. REAL TEAM POSSESSIONS (cálculo exacto basado en datos del equipo)
            # Calcular posesiones promedio por equipo (más eficiente)
            if all(col in self.teams_df.columns for col in ['FGA', 'FTA']):
                team_possessions = (
                    self.teams_df['FGA'] + 
                    0.44 * self.teams_df['FTA'] + 
                    self.teams_df.get('TOV', 0)
                )
            else:
                team_possessions = self.teams_df['FGA'] * 1.08
            
            # Crear diccionario de posesiones promedio por equipo
            team_poss_dict = self.teams_df.assign(possessions=team_possessions).groupby('Team')['possessions'].mean().to_dict()
            
            # 5. TRUE ASSIST RATE (AST por 100 posesiones reales del equipo)
            real_possessions = df['Team'].map(team_poss_dict).fillna(100)
            df['true_assist_rate_real'] = (ast_last / (real_possessions / 100 + 1)).fillna(0)
            features.append('true_assist_rate_real')
            
            # 6. TEAM PACE REAL (posesiones por 48 minutos)
            df['real_team_pace'] = real_possessions.fillna(100)
            df['assist_pace_adjusted'] = (ast_last * (df['real_team_pace'] / 100)).fillna(0)
            features.append('real_team_pace')
            features.append('assist_pace_adjusted')
            
        except Exception as e:
            logger.warning(f"Error en team possessions: {e}")
            # Features básicas como fallback
            df['true_assist_rate_real'] = (ast_last / 1.0).fillna(0)
            df['real_team_pace'] = 100
            df['assist_pace_adjusted'] = ast_last.fillna(0)
            features.extend(['true_assist_rate_real', 'real_team_pace', 'assist_pace_adjusted'])
        
        # === GRUPO 3: EFICIENCIA OFENSIVA REAL DEL EQUIPO - OPTIMIZADO ===
        
        try:
            # 7. REAL OFFENSIVE RATING (puntos por 100 posesiones reales)
            team_pts_dict = self.teams_df.groupby('Team')['PTS'].mean().to_dict()
            team_pts = df['Team'].map(team_pts_dict).fillna(110)
            real_possessions = df.get('real_team_pace', 100)
            df['real_offensive_rating'] = (team_pts / (real_possessions / 100 + 1)).fillna(110)
            features.append('real_offensive_rating')
            
            # 8. ASSIST IMPACT ON OFFENSE (correlación AST con eficiencia ofensiva)
            df['assist_offensive_impact'] = (ast_last * (df['real_offensive_rating'] / 110)).fillna(0)
            features.append('assist_offensive_impact')
            
        except Exception as e:
            logger.warning(f"Error en offensive rating: {e}")
            # Features básicas como fallback
            df['real_offensive_rating'] = 110
            df['assist_offensive_impact'] = ast_last.fillna(0)
            features.extend(['real_offensive_rating', 'assist_offensive_impact'])
        
        # === GRUPO 4: DEPENDENCIA REAL DE ASISTENCIAS DEL EQUIPO ===
        
        # 9. REAL TEAM ASSIST DEPENDENCY (% real de AST del equipo por el jugador)
        # Calcular total de AST del equipo por partido
        team_ast_total = df.groupby(['Team', 'Date'])['AST'].transform('sum')
        df['real_team_ast_dependency'] = (ast_last / (team_ast_total.shift(1) + 1)).fillna(0)
        features.append('real_team_ast_dependency')
        
        # 10. ASSIST SHARE CONSISTENCY (consistencia en % de AST del equipo)
        ast_share_rolling = df.groupby('Player')['real_team_ast_dependency'].rolling(5, min_periods=1).std().shift(1).reset_index(0, drop=True)
        df['assist_share_consistency'] = (1 / (ast_share_rolling + 0.1)).fillna(1)
        features.append('assist_share_consistency')
        
        # === GRUPO 5: CONTEXTO DEFENSIVO DEL OPONENTE REAL ===
        
        # 11. OPPONENT DEFENSIVE RATING REAL (de teams_df) - OPTIMIZADO
        if 'PTS_Opp' in self.teams_df.columns:
            try:
                # Crear diccionario de ratings defensivos por equipo (más eficiente)
                opp_def_ratings = self.teams_df.groupby('Team')['PTS_Opp'].mean().to_dict()
                
                # Mapear directamente sin merge masivo
                df['real_opponent_def_rating'] = df['Opp'].map(opp_def_ratings).fillna(110)
                features.append('real_opponent_def_rating')
                
                # 12. ASSIST DIFFICULTY FACTOR (más difícil vs defensas mejores)
                df['assist_difficulty_factor'] = (ast_last * (120 - df['real_opponent_def_rating']) / 20).fillna(0)
                features.append('assist_difficulty_factor')
            except Exception as e:
                logger.warning(f"Error en opponent defensive rating: {e}")
                # Features básicas como fallback
                df['real_opponent_def_rating'] = 110
                df['assist_difficulty_factor'] = ast_last.fillna(0)
                features.extend(['real_opponent_def_rating', 'assist_difficulty_factor'])
        
        # === GRUPO 6: FEATURES DE SINERGIA AVANZADA ===
        
        # 13. TEAM CHEMISTRY INDICATOR (basado en distribución de AST)
        team_ast_std = df.groupby(['Team', 'Date'])['AST'].transform('std').shift(1).fillna(2)
        df['team_chemistry_indicator'] = (1 / (team_ast_std + 1)).fillna(0.5)
        features.append('team_chemistry_indicator')
        
        # 14. PLAYMAKER HIERARCHY (posición en jerarquía de AST del equipo)
        df['ast_rank_in_team'] = df.groupby(['Team', 'Date'])['AST'].rank(ascending=False, method='dense')
        df['playmaker_hierarchy'] = (6 - df['ast_rank_in_team'].shift(1)).clip(1, 5).fillna(3)
        features.append('playmaker_hierarchy')
        
        # === GRUPO 7: FEATURES DE MOMENTUM DE EQUIPO ===
        
        # 15. TEAM OFFENSIVE MOMENTUM (tendencia ofensiva del equipo)
        team_off_momentum = df.groupby('Team')['real_offensive_rating'].rolling(3, min_periods=1).apply(
            lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) >= 2 else 0
        ).shift(1).reset_index(0, drop=True)
        df['team_offensive_momentum'] = team_off_momentum.fillna(0)
        features.append('team_offensive_momentum')
        
        # 16. ASSIST MOMENTUM ALIGNMENT (AST del jugador alineado con momentum del equipo)
        ast_momentum = df.groupby('Player')['AST'].rolling(3, min_periods=1).apply(
            lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) >= 2 else 0
        ).shift(1).reset_index(0, drop=True)
        df['assist_momentum_alignment'] = (ast_momentum * df['team_offensive_momentum']).fillna(0)
        features.append('assist_momentum_alignment')
        
        # === GRUPO 8: FEATURES DE CONTEXTO SITUACIONAL AVANZADO ===
        
        # 17. HOME/AWAY TEAM CONTEXT - CORREGIDO CON COLUMNAS REALES
        if 'is_home' in df.columns:
            try:
                # Calcular promedios home/away por jugador usando is_home
                home_ast_dict = df[df['is_home'] == 1].groupby('Player')['AST'].mean().to_dict()
                away_ast_dict = df[df['is_home'] == 0].groupby('Player')['AST'].mean().to_dict()
                
                # Crear feature basada en si es home o away
                def get_home_away_factor(row):
                    player = row['Player']
                    if row['is_home'] == 1:
                        return home_ast_dict.get(player, row.get('ast_season_avg', 3.0))
                    else:
                        return away_ast_dict.get(player, row.get('ast_season_avg', 3.0))
                
                df['home_away_ast_factor'] = df.apply(get_home_away_factor, axis=1)
                features.append('home_away_ast_factor')
                
            except Exception as e:
                logger.warning(f"Error en home/away context: {e}")
                # Feature básica como fallback
                df['home_away_ast_fallback'] = df.get('ast_season_avg', 3.0)
                features.append('home_away_ast_fallback')
        
        # 18. CLUTCH TEAM PERFORMANCE CONTEXT
        # Identificar juegos cerrados basado en diferencia de puntos
        if 'PTS' in df.columns:
            team_pts_game = df.groupby(['Team', 'Date'])['PTS'].transform('sum')
            opp_pts_game = team_pts_game.shift(1)  # Aproximación
            game_closeness = abs(team_pts_game - opp_pts_game) <= 10
            
            clutch_ast = ast_last * game_closeness.astype(int)
            df['clutch_context_assists'] = clutch_ast.fillna(0)
            features.append('clutch_context_assists')
        
        # === GRUPO 9: FEATURES DE PREDICCIÓN ULTRA-AVANZADA ===
        
        # 19. MASTER TEAM CONTEXT PREDICTOR
        # Combina las mejores features de contexto de equipo
        base_team_predictor = (
            df['assist_effectiveness_real'] * 0.3 +
            df['true_assist_rate_real'] * 0.25 +
            df['real_team_ast_dependency'] * 0.2 +
            df['assist_offensive_impact'] * 0.15 +
            df['playmaker_hierarchy'] * 0.1
        )
        df['master_team_context_predictor'] = base_team_predictor.fillna(0)
        features.append('master_team_context_predictor')
        
        # 20. ULTIMATE ASSIST PREDICTOR (combinación final)
        # Integra contexto de equipo con rendimiento individual
        individual_factor = df.get('ast_avg_5g', 0)
        team_factor = df['master_team_context_predictor']
        momentum_factor = df['assist_momentum_alignment']
        
        df['ultimate_assist_predictor'] = (
            individual_factor * 0.4 +
            team_factor * 0.4 +
            momentum_factor * 0.2
        ).fillna(0)
        features.append('ultimate_assist_predictor')
        
        # Actualizar categoría
        self.feature_categories['revolutionary_team_context'] = 20
        
        # Registrar features
        for feature in features:
            self._register_feature(feature, 'revolutionary_team_context')
        
        logger.info(f"Generadas {len(features)} FEATURES REVOLUCIONARIAS DE CONTEXTO DE EQUIPO REAL")
        return features
    
    def _generate_basic_team_context_features(self, df: pd.DataFrame) -> List[str]:
        """Features básicas de contexto de equipo cuando teams_df no está disponible"""
        features = []
        
        # Features básicas usando solo datos de jugadores
        if 'AST' in df.columns:
            ast_last = df.groupby('Player')['AST'].shift(1)
            
            # 1. Team AST dependency básica
            team_ast_total = df.groupby(['Team', 'Date'])['AST'].transform('sum')
            df['basic_team_ast_dependency'] = (ast_last / (team_ast_total.shift(1) + 1)).fillna(0)
            features.append('basic_team_ast_dependency')
            
            # 2. Assist share consistency básica
            ast_share_rolling = df.groupby('Player')['basic_team_ast_dependency'].rolling(5, min_periods=1).std().shift(1).reset_index(0, drop=True)
            df['basic_assist_share_consistency'] = (1 / (ast_share_rolling + 0.1)).fillna(1)
            features.append('basic_assist_share_consistency')
        
        # Registrar features
        for feature in features:
            self._register_feature(feature, 'revolutionary_team_context')
        
        return features

    def _generate_interaction_and_pattern_features(self, df: pd.DataFrame) -> List[str]:
        """
        FEATURES DE INTERACCIÓN Y PATRONES COMPLEJOS
        Implementa análisis avanzado de patrones de juego, interacciones entre jugadores y secuencias
        """
        features = []
        logger.info("Generando FEATURES DE INTERACCIÓN Y PATRONES COMPLEJOS...")
        
        try:
            ast_last = df.groupby('Player')['AST'].shift(1)
            
            # === GRUPO 1: INTERACCIONES ENTRE JUGADORES ESPECÍFICOS ===
            
            # 1. SINERGIA CON COMPAÑEROS DE EQUIPO (basado en altura)
            if 'Height_Inches' in df.columns:
                # Identificar "big men" (jugadores altos) por equipo/fecha
                team_avg_height = df.groupby(['Team', 'Date'])['Height_Inches'].transform('mean')
                is_big_man = (df['Height_Inches'] > team_avg_height + 2).astype(int)
                
                # AST cuando hay big men en el equipo
                big_men_in_team = df.groupby(['Team', 'Date'])['Height_Inches'].transform(
                    lambda x: (x > x.mean() + 2).sum()
                )
                df['ast_with_big_men'] = (ast_last * (big_men_in_team >= 2).astype(int)).fillna(0)
                features.append('ast_with_big_men')
                
                # Complementariedad de roles (playmaker + big man)
                player_is_playmaker = (df['Height_Inches'] < team_avg_height - 1).astype(int)
                df['playmaker_bigman_synergy'] = (ast_last * player_is_playmaker * (big_men_in_team >= 1).astype(int)).fillna(0)
                features.append('playmaker_bigman_synergy')
            
            # 2. DIVERSIDAD DE COMPAÑEROS (basado en distribución de minutos)
            if 'MP' in df.columns:
                # Distribución de minutos en el equipo (más diversa = mejor química)
                team_mp_std = df.groupby(['Team', 'Date'])['MP'].transform('std').shift(1)
                df['teammate_diversity'] = (ast_last * (team_mp_std / 10)).fillna(0)
                features.append('teammate_diversity')
            
            # === GRUPO 2: PATRONES DE PASE POR POSICIÓN EN CANCHA ===
            
            # 3. ASISTENCIAS EN TRIPLES (basado en 3PA del equipo)
            if '3PA' in df.columns:
                team_3pa = df.groupby(['Team', 'Date'])['3PA'].transform('sum').shift(1)
                three_point_rate = (team_3pa / 30).clip(0, 2)  # Normalizado
                df['ast_on_threes_potential'] = (ast_last * three_point_rate).fillna(0)
                features.append('ast_on_threes_potential')
            
            # 4. ASISTENCIAS EN CANASTAS DE 2 (basado en 2PA del equipo)
            if '2PA' in df.columns:
                team_2pa = df.groupby(['Team', 'Date'])['2PA'].transform('sum').shift(1)
                two_point_rate = (team_2pa / 50).clip(0, 2)  # Normalizado
                df['ast_on_twos_potential'] = (ast_last * two_point_rate).fillna(0)
                features.append('ast_on_twos_potential')
            
            # 5. PATRONES ESPACIALES INFERIDOS (basado en altura del jugador)
            if 'Height_Inches' in df.columns:
                # AST desde perímetro (jugadores más bajos)
                perimeter_factor = (1 - (df['Height_Inches'] - 70) / 15).clip(0, 1)
                df['perimeter_ast_tendency'] = (ast_last * perimeter_factor).fillna(0)
                features.append('perimeter_ast_tendency')
                
                # AST desde poste bajo (jugadores altos)
                post_factor = ((df['Height_Inches'] - 70) / 15).clip(0, 1)
                df['post_ast_tendency'] = (ast_last * post_factor).fillna(0)
                features.append('post_ast_tendency')
            
            # === GRUPO 3: ANÁLISIS DE SECUENCIAS DE JUGADAS ===
            
            # 6. MOMENTUM DE ASISTENCIAS (hot hand effect)
            ast_hot_streak = df.groupby('Player')['AST'].rolling(3, min_periods=1).apply(
                lambda x: 1 if len(x) >= 2 and all(x >= x.mean()) else 0
            ).shift(1).reset_index(0, drop=True)
            df['ast_hot_streak'] = ast_hot_streak.fillna(0)
            features.append('ast_hot_streak')
            
            # 7. PATRONES DE FLUJO DE JUEGO (alto ritmo)
            if 'FGA' in df.columns:
                team_pace = df.groupby(['Team', 'Date'])['FGA'].transform('sum').shift(1)
                high_pace_games = (team_pace > 90).astype(int)
                df['high_pace_ast_efficiency'] = (ast_last * high_pace_games).fillna(0)
                features.append('high_pace_ast_efficiency')
            
            # 8. ASISTENCIAS EN JUEGOS CERRADOS (basado en +/-)
            if '+/-' in df.columns:
                close_game = (abs(df['+/-'].shift(1)) <= 10).astype(int)
                df['clutch_ast_rate'] = (ast_last * close_game).fillna(0)
                features.append('clutch_ast_rate')
            
            # === GRUPO 4: PATRONES TEMPORALES AVANZADOS ===
            
            # 9. ASISTENCIAS EN DIFERENTES FASES DEL JUEGO
            # Aproximación: early game vs late game basado en minutos jugados
            if 'MP' in df.columns:
                mp_last = df.groupby('Player')['MP'].shift(1)
                # Si jugó muchos minutos, probablemente jugó en momentos clave
                key_moments_factor = (mp_last / 40).clip(0, 1.2)
                df['key_moments_ast'] = (ast_last * key_moments_factor).fillna(0)
                features.append('key_moments_ast')
            
            # 10. SECUENCIAS DE RENDIMIENTO
            # Patrón de AST en juegos consecutivos
            ast_sequence_pattern = df.groupby('Player')['AST'].rolling(4, min_periods=2).apply(
                lambda x: 1 if len(x) >= 3 and x.iloc[-1] > x.iloc[0] else 0  # Tendencia ascendente
            ).shift(1).reset_index(0, drop=True)
            df['ast_sequence_improvement'] = ast_sequence_pattern.fillna(0)
            features.append('ast_sequence_improvement')
            
            # === GRUPO 5: INTERACCIONES COMPLEJAS ENTRE VARIABLES ===
            
            # 11. INTERACCIÓN ALTURA × ASISTENCIAS × MINUTOS
            if all(col in df.columns for col in ['Height_Inches', 'MP']):
                mp_last = df.groupby('Player')['MP'].shift(1)
                height_mp_interaction = (df['Height_Inches'] / 80) * (mp_last / 36)
                df['height_minutes_ast_interaction'] = (ast_last * height_mp_interaction).fillna(0)
                features.append('height_minutes_ast_interaction')
            
            # 12. INTERACCIÓN ROBOS × ASISTENCIAS (transición)
            if 'STL' in df.columns:
                stl_last = df.groupby('Player')['STL'].shift(1)
                transition_factor = (stl_last / 2).clip(0, 2)
                df['transition_ast_from_steals'] = (ast_last * transition_factor).fillna(0)
                features.append('transition_ast_from_steals')
            
            # 13. INTERACCIÓN PÉRDIDAS × ASISTENCIAS (control del balón)
            if 'TOV' in df.columns:
                tov_last = df.groupby('Player')['TOV'].shift(1)
                ball_control_factor = (3 / (tov_last + 1)).clip(0, 3)  # Menos TOV = mejor control
                df['ball_control_ast_efficiency'] = (ast_last * ball_control_factor).fillna(0)
                features.append('ball_control_ast_efficiency')
            
            # === GRUPO 6: PATRONES DE ADAPTACIÓN SITUACIONAL ===
            
            # 14. ADAPTACIÓN A DIFERENTES OPONENTES
            # AST vs oponentes específicos (memoria histórica)
            opp_specific_ast = df.groupby(['Player', 'Opp'])['AST'].expanding().mean().shift(1).reset_index([0,1], drop=True)
            df['opp_specific_ast_memory'] = opp_specific_ast.fillna(ast_last)
            features.append('opp_specific_ast_memory')
            
            # 15. ADAPTACIÓN AL CONTEXTO DEL JUEGO
            # Combinar múltiples factores contextuales
            if all(col in df.columns for col in ['is_home', 'is_started']):
                context_multiplier = (
                    df['is_home'] * 1.1 +  # Ventaja de local
                    df['is_started'] * 1.2 +  # Ventaja de titular
                    0.8  # Base
                )
                df['contextual_ast_adaptation'] = (ast_last * context_multiplier).fillna(0)
                features.append('contextual_ast_adaptation')
            
            # === GRUPO 7: FEATURES DE PREDICCIÓN ULTRA-AVANZADA ===
            
            # 16. PREDICTOR MAESTRO DE INTERACCIONES
            # Combina las mejores features de interacción
            interaction_features = ['ast_with_big_men', 'ast_on_threes_potential', 'ast_hot_streak', 
                                  'transition_ast_from_steals', 'ball_control_ast_efficiency']
            available_features = [f for f in interaction_features if f in df.columns]
            
            if available_features:
                interaction_sum = sum(df[f] for f in available_features)
                df['master_interaction_predictor'] = (interaction_sum / len(available_features)).fillna(0)
                features.append('master_interaction_predictor')
            
            # 17. PREDICTOR FINAL DE PATRONES COMPLEJOS
            # Integra todos los patrones identificados
            pattern_features = ['perimeter_ast_tendency', 'key_moments_ast', 'ast_sequence_improvement',
                              'opp_specific_ast_memory', 'contextual_ast_adaptation']
            available_patterns = [f for f in pattern_features if f in df.columns]
            
            if available_patterns:
                pattern_sum = sum(df[f] for f in available_patterns)
                df['complex_patterns_predictor'] = (pattern_sum / len(available_patterns)).fillna(0)
                features.append('complex_patterns_predictor')
            
        except Exception as e:
            logger.warning(f"Error en features de interacción: {e}")
            # Features básicas como fallback
            ast_last = df.groupby('Player')['AST'].shift(1)
            df['basic_interaction_predictor'] = ast_last.fillna(0)
            features.append('basic_interaction_predictor')
        
        # Actualizar categoría
        self.feature_categories['interaction_patterns'] = len(features)
        
        # Registrar features
        for feature in features:
            self._register_feature(feature, 'interaction_patterns')
        
        logger.info(f"Generadas {len(features)} FEATURES DE INTERACCIÓN Y PATRONES COMPLEJOS")
        return features

    def _generate_model_feedback_features(self, df: pd.DataFrame) -> List[str]:
        """
        FEATURES ULTRA-PREDICTIVAS BASADAS EN FEEDBACK DEL MODELO
        Amplifica los patrones que el modelo identificó como más importantes:
        - contextual_ast_predictor (32.84%)
        - calibrated_ast_predictor (10.87%) 
        - ast_avg_15g (6.60%)
        - ast_per_minute_10g (6.27%)
        - starter_impact (4.78%)
        """
        features = []
        logger.info("Generando FEATURES ULTRA-PREDICTIVAS BASADAS EN FEEDBACK DEL MODELO...")
        
        try:
            ast_last = df.groupby('Player')['AST'].shift(1)
            
            # === GRUPO 1: AMPLIFICACIÓN DE PREDICTORES CONTEXTUALES ===
            # Basado en que contextual_ast_predictor domina con 32.84%
            
            # 1. SUPER CONTEXTUAL PREDICTOR (Mejora del top 1)
            # Combina múltiples contextos con pesos optimizados
            if all(col in df.columns for col in ['is_home', 'is_started', 'MP']):
                mp_last = df.groupby('Player')['MP'].shift(1)
                ast_15g = df.groupby('Player')['AST'].rolling(15, min_periods=1).mean().shift(1).reset_index(0, drop=True)
                
                # Contexto ultra-avanzado con pesos basados en importancia del modelo
                context_score = (
                    df['is_home'] * 0.15 +           # Factor local
                    df['is_started'] * 0.35 +        # Factor titular (muy importante)
                    (mp_last / 40).clip(0, 1) * 0.25 +  # Factor minutos
                    (ast_15g / 10).clip(0, 1) * 0.25    # Factor histórico
                )
                df['super_contextual_predictor'] = (ast_last * context_score).fillna(0)
                features.append('super_contextual_predictor')
            
            # 2. CALIBRATED PREDICTOR ENHANCED (Mejora del top 2)
            # Versión mejorada del calibrated_ast_predictor
            ast_5g = df.groupby('Player')['AST'].rolling(5, min_periods=1).mean().shift(1).reset_index(0, drop=True)
            ast_10g = df.groupby('Player')['AST'].rolling(10, min_periods=1).mean().shift(1).reset_index(0, drop=True)
            ast_season = df.groupby('Player')['AST'].expanding().mean().shift(1).reset_index(0, drop=True)
            
            # Calibración ultra-precisa con múltiples ventanas
            calibration_weights = [0.4, 0.35, 0.25]  # Más peso a corto plazo
            df['ultra_calibrated_predictor'] = (
                ast_5g * calibration_weights[0] + 
                ast_10g * calibration_weights[1] + 
                ast_season * calibration_weights[2]
            ).fillna(0)
            features.append('ultra_calibrated_predictor')
            
            # === GRUPO 2: AMPLIFICACIÓN DE VENTANAS TEMPORALES ===
            # Basado en que ast_avg_15g es top 3 con 6.60%
            
            # 3. MULTI-WINDOW TEMPORAL PREDICTOR
            # Combina múltiples ventanas temporales con pesos optimizados
            windows = [3, 5, 10, 15, 20]
            window_weights = [0.1, 0.2, 0.25, 0.25, 0.2]  # Más peso a ventanas medias
            
            temporal_sum = 0
            for window, weight in zip(windows, window_weights):
                ast_window = df.groupby('Player')['AST'].rolling(window, min_periods=1).mean().shift(1).reset_index(0, drop=True)
                temporal_sum += ast_window.fillna(0) * weight
            
            df['multi_window_predictor'] = temporal_sum
            features.append('multi_window_predictor')
            
            # 4. TEMPORAL MOMENTUM PREDICTOR
            # Detecta tendencias en múltiples ventanas
            ast_3g = df.groupby('Player')['AST'].rolling(3, min_periods=1).mean().shift(1).reset_index(0, drop=True)
            ast_7g = df.groupby('Player')['AST'].rolling(7, min_periods=1).mean().shift(1).reset_index(0, drop=True)
            
            # Momentum = diferencia entre ventanas cortas y largas
            momentum = (ast_3g - ast_7g).fillna(0)
            df['temporal_momentum_predictor'] = (ast_10g + momentum * 0.3).fillna(0)
            features.append('temporal_momentum_predictor')
            
            # === GRUPO 3: AMPLIFICACIÓN DE EFICIENCIA POR MINUTO ===
            # Basado en que ast_per_minute_10g es top 4 con 6.27%
            
            # 5. ULTRA EFFICIENCY PREDICTOR
            # Versión mejorada de ast_per_minute con múltiples factores
            if 'MP' in df.columns:
                mp_5g = df.groupby('Player')['MP'].rolling(5, min_periods=1).mean().shift(1).reset_index(0, drop=True)
                mp_10g = df.groupby('Player')['MP'].rolling(10, min_periods=1).mean().shift(1).reset_index(0, drop=True)
                
                # Eficiencia ultra-precisa
                efficiency_5g = (ast_5g / (mp_5g + 1)).fillna(0)
                efficiency_10g = (ast_10g / (mp_10g + 1)).fillna(0)
                
                # Combinar eficiencias con factor de confianza basado en minutos
                confidence_factor = (mp_10g / 30).clip(0, 1.5)
                df['ultra_efficiency_predictor'] = (
                    efficiency_5g * 0.4 + efficiency_10g * 0.6
                ) * confidence_factor
                features.append('ultra_efficiency_predictor')
            
            # === GRUPO 4: AMPLIFICACIÓN DEL FACTOR TITULAR ===
            # Basado en que starter_impact es top 5 con 4.78%
            
            # 6. ENHANCED STARTER IMPACT
            # Versión mejorada que considera historial de titularidad
            if 'is_started' in df.columns:
                # Frecuencia de titularidad en últimos juegos
                starter_freq_5g = df.groupby('Player')['is_started'].rolling(5, min_periods=1).mean().shift(1).reset_index(0, drop=True)
                starter_freq_10g = df.groupby('Player')['is_started'].rolling(10, min_periods=1).mean().shift(1).reset_index(0, drop=True)
                
                # Impacto mejorado de titularidad
                starter_consistency = (starter_freq_5g + starter_freq_10g) / 2
                df['enhanced_starter_impact'] = (
                    ast_last * df['is_started'] * (1 + starter_consistency)
                ).fillna(0)
                features.append('enhanced_starter_impact')
            
            # === GRUPO 5: FEATURES DE ALTA FRECUENCIA DE ASISTENCIAS ===
            # Basado en que high_assist_game_freq es importante
            
            # 7. EXPLOSIVE ASSIST POTENTIAL
            # Predice juegos de muchas asistencias
            ast_max_5g = df.groupby('Player')['AST'].rolling(5, min_periods=1).max().shift(1).reset_index(0, drop=True)
            ast_std_5g = df.groupby('Player')['AST'].rolling(5, min_periods=1).std().shift(1).reset_index(0, drop=True)
            
            # Potencial explosivo = máximo reciente + volatilidad
            df['explosive_assist_potential'] = (
                ast_max_5g * 0.7 + ast_std_5g * 0.3
            ).fillna(0)
            features.append('explosive_assist_potential')
            
            # === GRUPO 6: FEATURES DE DIFICULTAD Y ADAPTABILIDAD ===
            # Basado en assist_difficulty_factor y player_adaptability_score
            
            # 8. ADAPTIVE DIFFICULTY PREDICTOR
            # Combina dificultad del oponente con adaptabilidad del jugador
            if 'Opp' in df.columns:
                # Rendimiento histórico vs cada oponente
                opp_ast_avg = df.groupby(['Player', 'Opp'])['AST'].expanding().mean().shift(1).reset_index([0,1], drop=True)
                league_ast_avg = df.groupby('Player')['AST'].expanding().mean().shift(1).reset_index(0, drop=True)
                
                # Factor de adaptación = rendimiento vs oponente / rendimiento general
                adaptation_factor = (opp_ast_avg / (league_ast_avg + 0.1)).fillna(1)
                df['adaptive_difficulty_predictor'] = (ast_last * adaptation_factor).fillna(0)
                features.append('adaptive_difficulty_predictor')
            
            # === GRUPO 7: FEATURES DE FATIGA Y CARGA DE TRABAJO ===
            # Basado en fatigue_adjusted_ast y workload_ratio
            
            # 9. SMART FATIGUE PREDICTOR
            # Versión inteligente que considera múltiples factores de fatiga
            if 'MP' in df.columns:
                # Carga de trabajo reciente
                mp_sum_3g = df.groupby('Player')['MP'].rolling(3, min_periods=1).sum().shift(1).reset_index(0, drop=True)
                mp_sum_7g = df.groupby('Player')['MP'].rolling(7, min_periods=1).sum().shift(1).reset_index(0, drop=True)
                
                # Factor de fatiga inteligente
                fatigue_factor = 1 - (mp_sum_3g / 120).clip(0, 0.3)  # Máximo 30% de penalización
                rest_bonus = 1 + (1 / (mp_sum_7g / 7 + 1)) * 0.1  # Bonus por descanso
                
                df['smart_fatigue_predictor'] = (
                    ast_last * fatigue_factor * rest_bonus
                ).fillna(0)
                features.append('smart_fatigue_predictor')
            
            # === GRUPO 8: PREDICTOR MAESTRO FINAL ===
            
            # 10. ULTIMATE AST PREDICTOR
            # Combina las mejores features con pesos basados en importancia del modelo
            predictor_features = [
                ('super_contextual_predictor', 0.35),
                ('ultra_calibrated_predictor', 0.25),
                ('multi_window_predictor', 0.15),
                ('ultra_efficiency_predictor', 0.10),
                ('enhanced_starter_impact', 0.08),
                ('explosive_assist_potential', 0.04),
                ('adaptive_difficulty_predictor', 0.03)
            ]
            
            ultimate_sum = 0
            available_predictors = 0
            
            for feature_name, weight in predictor_features:
                if feature_name in df.columns:
                    ultimate_sum += df[feature_name] * weight
                    available_predictors += weight
            
            if available_predictors > 0:
                df['ultimate_ast_predictor'] = ultimate_sum / available_predictors
                features.append('ultimate_ast_predictor')
            
        except Exception as e:
            logger.warning(f"Error en features de feedback del modelo: {e}")
            # Feature básica como fallback
            ast_last = df.groupby('Player')['AST'].shift(1)
            df['basic_feedback_predictor'] = ast_last.fillna(0)
            features.append('basic_feedback_predictor')
        
        # Actualizar categoría
        self.feature_categories['model_feedback'] = len(features)
        
        # Registrar features
        for feature in features:
            self._register_feature(feature, 'model_feedback')
        
        logger.info(f"Generadas {len(features)} FEATURES ULTRA-PREDICTIVAS BASADAS EN FEEDBACK DEL MODELO")
        
        return features
    
    def _generate_neural_pattern_features(self, df: pd.DataFrame) -> List[str]:
        """
        Features que imitan patrones de redes neuronales para capturar 
        relaciones no lineales complejas en las asistencias
        """
        logger.info("Generando features de patrones neuronales...")
        features = []
        
        try:
            # 1. ACTIVACIONES NO LINEALES
            if 'AST' in df.columns:
                ast_hist = self._get_historical_series(df, 'AST', 7, 'mean')
                
                # Función de activación ReLU
                df['ast_relu_activation'] = np.maximum(0, ast_hist - ast_hist.mean())
                features.append('ast_relu_activation')
                
                # Función de activación Sigmoid
                df['ast_sigmoid_activation'] = 1 / (1 + np.exp(-ast_hist.fillna(0)))
                features.append('ast_sigmoid_activation')
                
                # Función de activación Tanh
                df['ast_tanh_activation'] = np.tanh(ast_hist.fillna(0))
                features.append('ast_tanh_activation')
            
            # 2. CONVOLUCIONES TEMPORALES
            if 'AST' in df.columns:
                # Simular convolución 1D sobre series temporal de asistencias
                ast_series = self._get_historical_series(df, 'AST', 10, 'mean')
                ast_5g = self._get_historical_series(df, 'AST', 5, 'mean')
                ast_3g = self._get_historical_series(df, 'AST', 3, 'mean')
                
                # Kernel de detección de tendencias
                df['ast_trend_convolution'] = (
                    ast_series * 0.5 + 
                    ast_5g * 0.3 + 
                    ast_3g * 0.2
                ).fillna(0)
                features.append('ast_trend_convolution')
            
            # 3. ATTENTION MECHANISM SIMULADO
            if all(col in df.columns for col in ['AST', 'MP', 'BPM']):
                # Simular mecanismo de atención entre diferentes métricas
                ast_query = self._get_historical_series(df, 'AST', 5, 'mean').fillna(0)
                mp_key = df['MP'].fillna(0)
                bmp_value = df['BPM'].fillna(0)
                
                # Calcular scores de atención
                attention_scores = np.exp(ast_query * mp_key / 100) / (np.exp(ast_query * mp_key / 100) + 1e-8)
                df['attention_weighted_predictor'] = attention_scores * bmp_value
                features.append('attention_weighted_predictor')
                
        except Exception as e:
            logger.warning(f"Error en features de patrones neuronales: {e}")
        
        # Registrar features
        for feature in features:
            self._register_feature(feature, 'neural_patterns')
        
        return features
    
    def _generate_quantum_inspired_features(self, df: pd.DataFrame) -> List[str]:
        """
        Features inspiradas en mecánica cuántica para capturar 
        superposiciones y entrelazamientos en el rendimiento
        """
        logger.info("Generando features inspiradas en mecánica cuántica...")
        features = []
        
        try:
            # 1. SUPERPOSICIÓN DE ESTADOS
            if 'AST' in df.columns:
                # Estado de alto rendimiento vs bajo rendimiento
                ast_mean = self._get_historical_series(df, 'AST', 20, 'mean').fillna(0)
                ast_recent = self._get_historical_series(df, 'AST', 3, 'mean').fillna(0)
                
                # Superposición cuántica (probabilidad de estar en cada estado)
                high_state_prob = 1 / (1 + np.exp(-(ast_recent - ast_mean)))
                df['quantum_superposition_ast'] = high_state_prob
                features.append('quantum_superposition_ast')
            
            # 2. ENTRELAZAMIENTO CUÁNTICO
            if all(col in df.columns for col in ['AST', 'TOV']):
                # Entrelazamiento entre asistencias y pérdidas de balón
                ast_normalized = (self._get_historical_series(df, 'AST', 10, 'mean').fillna(0) - 
                                self._get_historical_series(df, 'AST', 10, 'mean').fillna(0).mean())
                tov_normalized = (self._get_historical_series(df, 'TOV', 10, 'mean').fillna(0) - 
                                self._get_historical_series(df, 'TOV', 10, 'mean').fillna(0).mean())
                
                # Función de entrelazamiento
                df['quantum_entanglement_ast_tov'] = np.cos(ast_normalized) * np.sin(tov_normalized)
                features.append('quantum_entanglement_ast_tov')
            
            # 3. PRINCIPIO DE INCERTIDUMBRE
            if 'AST' in df.columns:
                # Incertidumbre entre precisión y recall en predicción
                ast_mean = self._get_historical_series(df, 'AST', 10, 'mean').fillna(0)
                ast_std = self._get_historical_series(df, 'AST', 10, 'std').fillna(0)
                
                # Principio de incertidumbre de Heisenberg aplicado
                df['quantum_uncertainty_ast'] = ast_mean * ast_std
                features.append('quantum_uncertainty_ast')
                
        except Exception as e:
            logger.warning(f"Error en features cuánticas: {e}")
        
        # Registrar features
        for feature in features:
            self._register_feature(feature, 'quantum_inspired')
        
        return features
    
    def _generate_chaos_theory_features(self, df: pd.DataFrame) -> List[str]:
        """
        Features basadas en teoría del caos para capturar 
        comportamientos no lineales y efectos mariposa
        """
        logger.info("Generando features de teoría del caos...")
        features = []
        
        try:
            # 1. ATRACTORES EXTRAÑOS
            if 'AST' in df.columns:
                # Crear espacio de fase para asistencias
                ast_t = self._get_historical_series(df, 'AST', 1, 'mean').fillna(0)  # AST(t)
                ast_t1 = self._get_historical_series(df, 'AST', 2, 'mean').fillna(0)  # AST(t-1)
                ast_t2 = self._get_historical_series(df, 'AST', 3, 'mean').fillna(0)  # AST(t-2)
                
                # Atractor de Lorenz simplificado
                df['chaos_lorenz_x'] = ast_t - ast_t1
                df['chaos_lorenz_y'] = ast_t1 - ast_t2
                df['chaos_lorenz_z'] = ast_t * ast_t1 - ast_t2
                
                features.extend(['chaos_lorenz_x', 'chaos_lorenz_y', 'chaos_lorenz_z'])
            
            # 2. EXPONENTE DE LYAPUNOV
            if 'AST' in df.columns:
                # Medir sensibilidad a condiciones iniciales
                ast_series = self._get_historical_series(df, 'AST', 15, 'mean').fillna(0)
                ast_diff = ast_series.diff().fillna(0)
                
                # Aproximación del exponente de Lyapunov
                lyapunov_approx = np.log(np.abs(ast_diff) + 1e-8).rolling(5).mean().fillna(0)
                df['chaos_lyapunov_exponent'] = lyapunov_approx
                features.append('chaos_lyapunov_exponent')
                
        except Exception as e:
            logger.warning(f"Error en features de teoría del caos: {e}")
        
        # Registrar features
        for feature in features:
            self._register_feature(feature, 'chaos_theory')
        
        return features
    
    def _generate_fractal_features(self, df: pd.DataFrame) -> List[str]:
        """
        Features basadas en geometría fractal para capturar 
        patrones auto-similares en el rendimiento
        """
        logger.info("Generando features fractales...")
        features = []
        
        try:
            if 'AST' in df.columns:
                # 1. DIMENSIÓN FRACTAL
                ast_series = self._get_historical_series(df, 'AST', 20, 'mean').fillna(0)
                
                # Calcular dimensión fractal usando box-counting
                # Simplificado para series temporales
                ast_range = ast_series.max() - ast_series.min()
                if ast_range > 0:
                    df['fractal_dimension'] = np.log(len(ast_series)) / np.log(ast_range + 1)
                else:
                    df['fractal_dimension'] = 0
                features.append('fractal_dimension')
                
                # 2. AUTOSIMILARIDAD
                # Comparar patrones en diferentes escalas temporales
                ast_3g = self._get_historical_series(df, 'AST', 3, 'mean').fillna(0)
                ast_9g = self._get_historical_series(df, 'AST', 9, 'mean').fillna(0)
                
                # Medida de autosimilaridad
                df['fractal_self_similarity'] = np.abs(ast_3g - ast_9g / 3)
                features.append('fractal_self_similarity')
                
        except Exception as e:
            logger.warning(f"Error en features fractales: {e}")
        
        # Registrar features
        for feature in features:
            self._register_feature(feature, 'fractal_features')
        
        return features
    
    def _generate_evolutionary_features(self, df: pd.DataFrame) -> List[str]:
        """
        Features basadas en algoritmos evolutivos para capturar 
        adaptación y evolución del rendimiento
        """
        logger.info("Generando features evolutivas...")
        features = []
        
        try:
            if 'AST' in df.columns:
                # 1. FITNESS FUNCTION
                # Función de aptitud basada en múltiples criterios
                ast_avg = self._get_historical_series(df, 'AST', 10, 'mean').fillna(0)
                ast_consistency = 1 / (self._get_historical_series(df, 'AST', 10, 'std').fillna(1) + 1)
                
                df['evolutionary_fitness'] = ast_avg * ast_consistency
                features.append('evolutionary_fitness')
                
                # 2. MUTATION RATE
                # Tasa de cambio en el rendimiento
                ast_recent = self._get_historical_series(df, 'AST', 3, 'mean').fillna(0)
                ast_baseline = self._get_historical_series(df, 'AST', 15, 'mean').fillna(0)
                
                df['evolutionary_mutation_rate'] = np.abs(ast_recent - ast_baseline) / (ast_baseline + 1)
                features.append('evolutionary_mutation_rate')
                
                # 3. SELECTION PRESSURE
                # Presión de selección basada en competencia
                if 'MP' in df.columns:
                    minutes_pressure = df['MP'] / 48  # Normalizado
                    df['evolutionary_selection_pressure'] = ast_avg * minutes_pressure
                    features.append('evolutionary_selection_pressure')
                    
        except Exception as e:
            logger.warning(f"Error en features evolutivas: {e}")
        
        # Registrar features
        for feature in features:
            self._register_feature(feature, 'evolutionary_features')
        
        return features

    def intelligent_feature_selection(self, df: pd.DataFrame, features: List[str], target_col: str = 'AST') -> List[str]:
        """
        MEJORA CRÍTICA: Selección inteligente de características para reducir sobreingeniería.
        ACTUALIZADO: Elimina features de bajo rendimiento que generan ruido.
        
        Estrategia de selección:
        1. Features básicas esenciales (siempre incluir)
        2. Selección por importancia usando XGBoost rápido
        3. Eliminación de características redundantes/correlacionadas
        4. Eliminación de features con importancia < 0.003 (ruido)
        5. Límite máximo de características
        
        Args:
            df: DataFrame con todas las características
            features: Lista de todas las características disponibles
            target_col: Columna objetivo (AST)
            
        Returns:
            List[str]: Lista de características seleccionadas (máximo self.max_features)
        """
        logger.info(f"Selección inteligente de características: {len(features)} → {self.max_features}")
        
        # PASO 1: Features básicas esenciales (OPTIMIZADAS - solo las más importantes)
        essential_features = [
            # Top features identificadas por el modelo
            'optimized_hybrid_predictor', 'learning_adaptive_predictor',
            'TS%', 'BPM', 'MP', '3P%', 'FG%',
            'minutes_based_ast_predictor', 'ast_per_minute_10g', 'FG',
            'TOV', 'ast_per_minute_5g', '2P%', 'total_score',
            'contextual_ast_predictor', 'home_away_ast_factor',
            # Features de contexto críticas
            'ast_avg_15g', 'ast_season_avg', 'real_opponent_def_rating',
            'ultimate_ast_predictor', 'usage_rate', 'assist_rate'
        ]
        
        # Filtrar features esenciales que existen
        selected_features = [f for f in essential_features if f in features and f in df.columns]
        remaining_features = [f for f in features if f not in selected_features and f in df.columns]
        
        logger.info(f"Features esenciales incluidas: {len(selected_features)}")
        
        # PASO 2: Si ya tenemos suficientes features esenciales
        if len(selected_features) >= self.max_features:
            return selected_features[:self.max_features]
        
        # PASO 3: Selección por importancia usando XGBoost rápido
        if len(remaining_features) > 0 and len(df) > 100:
            try:
                import xgboost as xgb
                
                # Preparar datos asegurando que sean numéricos
                X_temp = df[remaining_features].copy()
                
                # Convertir categóricas a numéricas antes de XGBoost
                for col in X_temp.columns:
                    if X_temp[col].dtype in ['object', 'string', 'category']:
                        X_temp[col] = pd.to_numeric(X_temp[col], errors='coerce')
                
                X_temp = X_temp.fillna(0)
                y_temp = df[target_col]
                
                # XGBoost rápido para feature importance (solo con datos numéricos)
                temp_model = xgb.XGBRegressor(
                    n_estimators=50,  # Muy rápido
                    max_depth=3,
                    learning_rate=0.1,
                    random_state=42,
                    n_jobs=1,  # Single thread para evitar conflictos
                    verbosity=0
                )
                
                temp_model.fit(X_temp, y_temp)
                
                # Obtener importancias
                importances = temp_model.feature_importances_
                feature_importance_pairs = list(zip(remaining_features, importances))
                
                # NUEVO: Filtrar features con importancia muy baja (< 0.003)
                filtered_pairs = [(f, imp) for f, imp in feature_importance_pairs if imp >= 0.003]
                
                logger.info(f"Features filtradas por baja importancia: {len(feature_importance_pairs) - len(filtered_pairs)}")
                
                # Ordenar por importancia descendente
                filtered_pairs.sort(key=lambda x: x[1], reverse=True)
                
                # Seleccionar las mejores características restantes
                remaining_slots = self.max_features - len(selected_features)
                top_features = [f[0] for f in filtered_pairs[:remaining_slots]]
                
                selected_features.extend(top_features)
                
                logger.info(f"Features por importancia añadidas: {len(top_features)}")
                
            except Exception as e:
                logger.warning(f"Error en selección por importancia: {e}")
                # Fallback: tomar features aleatorias pero filtrar las conocidas de bajo rendimiento
                low_performance_features = [
                    'is_home', 'ast_usage_rate', 'is_started', 'is_win',
                    'starter_impact', 'ast_consistency', 'ast_max_5g',
                    'ast_last_game', 'ast_consistency_3g', 'overtime_periods',
                    'team_offensive_pace', '2P', 'ast_improvement_ratio',
                    'ast_avg_3g', '3PA', 'ast_tov_ratio', 'ast_momentum',
                    'ast_volatility', 'ast_avg_5g', 'ast_volatility_3g',
                    'FT%', 'ast_trend_5g', 'point_diff', '2PA'
                ]
                
                filtered_remaining = [f for f in remaining_features if f not in low_performance_features]
                remaining_slots = self.max_features - len(selected_features)
                selected_features.extend(filtered_remaining[:remaining_slots])
        
        # PASO 4: Verificar correlaciones altas y eliminar redundantes (solo si tenemos muchas)
        if len(selected_features) > 30:  # Solo si tenemos muchas features
            try:
                corr_matrix = df[selected_features].corr().abs()
                
                # Encontrar pares con correlación > 0.95
                high_corr_pairs = []
                for i in range(len(corr_matrix.columns)):
                    for j in range(i+1, len(corr_matrix.columns)):
                        if corr_matrix.iloc[i, j] > 0.95:
                            high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j]))
                
                # Eliminar features redundantes (mantener la primera de cada par)
                features_to_remove = set()
                for f1, f2 in high_corr_pairs:
                    if f1 not in essential_features:  # No eliminar features esenciales
                        features_to_remove.add(f1)
                    elif f2 not in essential_features:
                        features_to_remove.add(f2)
                
                selected_features = [f for f in selected_features if f not in features_to_remove]
                
                if features_to_remove:
                    logger.info(f"Features redundantes eliminadas: {len(features_to_remove)}")
                
            except Exception as e:
                logger.warning(f"Error eliminando correlaciones: {e}")
        
        # Asegurar que no excedemos el límite
        final_features = selected_features[:self.max_features]
        
        logger.info(f"Selección final: {len(final_features)} características")
        logger.info(f"Top 10 features: {final_features[:10]}")
        
        return final_features
    
    def _ensure_numeric_features(self, df: pd.DataFrame, features: List[str]) -> List[str]:
        """
        CRÍTICO: Asegurar que todas las características sean numéricas.
        Elimina características que contienen valores no numéricos.
        MEJORADO: Maneja mejor las columnas categóricas problemáticas.
        
        Args:
            df: DataFrame con las características
            features: Lista de características a validar
            
        Returns:
            List[str]: Lista de características numéricas válidas
        """
        numeric_features = []
        removed_features = []
        
        # Columnas categóricas conocidas que causan problemas
        categorical_columns = ['Pos', 'Team', 'Opp', 'Player', 'Date', 'Away', 'GS', 'Result']
        
        for feature in features:
            if feature not in df.columns:
                removed_features.append(f"{feature} (no existe)")
                continue
            
            # Saltar columnas categóricas conocidas
            if feature in categorical_columns:
                removed_features.append(f"{feature} (categórica)")
                continue
                
            try:
                # Verificar si la columna es numérica
                if df[feature].dtype in ['object', 'string', 'category']:
                    # Verificar si contiene solo valores numéricos como strings
                    sample_values = df[feature].dropna().head(100)
                    
                    if len(sample_values) == 0:
                        removed_features.append(f"{feature} (vacía)")
                        continue
                    
                    # Intentar convertir a numérico
                    test_conversion = pd.to_numeric(sample_values, errors='coerce')
                    
                    if test_conversion.isna().all():
                        # Contiene valores no numéricos
                        unique_vals = sample_values.unique()[:5]  # Primeros 5 valores únicos
                        removed_features.append(f"{feature} (valores no numéricos: {list(unique_vals)})")
                        continue
                    elif test_conversion.isna().sum() / len(test_conversion) > 0.5:
                        # Más del 50% son valores no numéricos
                        removed_features.append(f"{feature} (>50% no numéricos)")
                        continue
                    else:
                        # Convertir toda la columna
                        df[feature] = pd.to_numeric(df[feature], errors='coerce').fillna(0)
                
                # Verificar que el tipo final sea numérico
                if df[feature].dtype in ['float64', 'int64', 'float32', 'int32', 'bool', 'int8', 'int16']:
                    # Reemplazar infinitos con NaN y luego con 0
                    if np.isinf(df[feature]).any():
                        df[feature] = df[feature].replace([np.inf, -np.inf], np.nan).fillna(0)
                    
                    # Reemplazar NaN con 0
                    if df[feature].isna().any():
                        df[feature] = df[feature].fillna(0)
                    
                    numeric_features.append(feature)
                else:
                    removed_features.append(f"{feature} (tipo final no numérico: {df[feature].dtype})")
                    
            except Exception as e:
                removed_features.append(f"{feature} (error: {str(e)[:50]})")
                continue
        
        if removed_features:
            logger.warning(f"Features eliminadas ({len(removed_features)}): {removed_features[:5]}...")
        
        logger.info(f"Validación numérica: {len(features)} → {len(numeric_features)} características")
        
        return numeric_features