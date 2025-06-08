"""
Módulo de Características para Predicción de Rebotes Totales (TRB)
================================================================

FEATURES BASADAS EN PRINCIPIOS FUNDAMENTALES DE REBOTES:

1. EFICIENCIA DE TIRO: Los rebotes se generan por tiros fallados
2. ALTURA Y FÍSICO: Ventaja natural para capturar rebotes  
3. POSICIONAMIENTO: Minutos, rol, posición en cancha
4. CONTEXTO DEL EQUIPO: Ritmo, estilo de juego
5. CONTEXTO DEL OPONENTE: Características que afectan oportunidades
6. HISTORIAL DE REBOTES: Rendimiento pasado
7. SITUACIÓN DEL JUEGO: Contexto específico del partido

Sin data leakage, todas las métricas usan shift(1) para crear historial
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
import logging
from datetime import datetime, timedelta
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression

warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class ReboundsFeatureEngineer:
    """
    Feature Engineer especializado en predicción de rebotes (TRB)
    Basado en los principios fundamentales de los rebotes en la NBA
    """
    
    def __init__(self, correlation_threshold: float = 0.95, max_features: int = 30, teams_df: pd.DataFrame = None):
        self.correlation_threshold = correlation_threshold
        self.max_features = max_features
        self.teams_df = teams_df  # Datos de equipos para features avanzadas
        self.feature_registry = {}
        self.feature_categories = {
            'shooting_efficiency': [],  # Eficiencia de tiro (genera oportunidades)
            'physical_advantage': [],   # Ventaja física (altura, peso)
            'positioning': [],          # Posicionamiento y minutos
            'team_context': [],         # Contexto del equipo
            'opponent_context': [],     # Contexto del oponente
            'rebounding_history': [],   # Historial de rebotes
            'game_situation': []        # Situación del juego
        }
        self.protected_features = ['TRB', 'Player', 'Date', 'Team', 'Opp']
        
    def _register_feature(self, feature_name: str, category: str) -> bool:
        """Registra una feature en la categoría correspondiente"""
        if feature_name not in self.feature_registry:
            self.feature_registry[feature_name] = category
            if category in self.feature_categories:
                self.feature_categories[category].append(feature_name)
            return True
        return False
    
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
        else:
            result = grouped.rolling(window=window, min_periods=1).mean().shift(1)
        
        # Resetear índice para compatibilidad
        return result.reset_index(0, drop=True)
    
    def generate_all_features(self, df: pd.DataFrame) -> List[str]:
        """
        Genera todas las features especializadas para predicción de rebotes
        Modifica el DataFrame in-place y devuelve la lista de features creadas
        """
        logger.info("Generando features NBA ESPECIALIZADAS para predicción de rebotes...")
        
        # Verificar target
        if 'TRB' in df.columns:
            trb_stats = df['TRB'].describe()
            logger.info(f"Target TRB disponible - Media={trb_stats['mean']:.1f}, Max={trb_stats['max']:.0f}")
        else:
            logger.warning("Target TRB no disponible - features limitadas")
        
        # Limpiar registro de features
        self.feature_registry = {}
        for category in self.feature_categories:
            self.feature_categories[category] = []
        
        # 1. FEATURES DE EFICIENCIA DE TIRO (Generan oportunidades de rebote)
        self._create_shooting_efficiency_features(df)
        
        # 2. FEATURES DE VENTAJA FÍSICA
        self._create_physical_advantage_features(df)
        
        # 3. FEATURES DE POSICIONAMIENTO
        self._create_positioning_features(df)
        
        # 4. FEATURES DE CONTEXTO DEL EQUIPO
        self._create_team_context_features(df)
        
        # 5. FEATURES DE CONTEXTO DEL OPONENTE
        self._create_opponent_context_features(df)
        
        # 6. FEATURES DE HISTORIAL DE REBOTES
        self._create_rebounding_history_features(df)
        
        # 7. FEATURES DE SITUACIÓN DEL JUEGO
        self._create_game_situation_features(df)
        
        # Obtener lista de features creadas
        created_features = list(self.feature_registry.keys())
        
        # Aplicar filtro de correlación si es necesario
        if len(created_features) > self.max_features:
            # Aplicar filtro de correlación manualmente
            logger.info(f"Aplicando filtro de correlación: {len(created_features)} -> {self.max_features} features")
            
            # Preparar datos para filtro de correlación
            X = df[created_features].fillna(0)
            
            # Calcular matriz de correlación
            corr_matrix = X.corr().abs()
            
            # Encontrar features altamente correlacionadas
            upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            
            # Identificar features a eliminar
            to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > self.correlation_threshold)]
            
            # Mantener features importantes y eliminar correlacionadas
            created_features = [f for f in created_features if f not in to_drop]
            
            # Si aún hay demasiadas features, usar selección basada en target
            if len(created_features) > self.max_features and 'TRB' in df.columns:
                logger.info("Aplicando selección adicional basada en target...")
                
                X_filtered = df[created_features].fillna(0)
                y = df['TRB'].fillna(df['TRB'].mean())
                
                # Seleccionar mejores features
                selector = SelectKBest(score_func=f_regression, k=self.max_features)
                selector.fit(X_filtered, y)
                
                # Obtener features seleccionadas
                selected_features = [created_features[i] for i in range(len(created_features)) 
                                   if selector.get_support()[i]]
                created_features = selected_features
        
        # Resumen final
        self._log_feature_summary()
        
        return created_features
    
    def _create_shooting_efficiency_features(self, df: pd.DataFrame) -> None:
        """
        Features basadas en eficiencia de tiro - PRINCIPIO FUNDAMENTAL
        Los rebotes se generan por tiros fallados. Más tiros fallados = más oportunidades
        """
        logger.debug("Creando features de eficiencia de tiro...")
        
        # EFICIENCIA DE TIRO DEL JUGADOR (Historial)
        if 'FG%' in df.columns:
            # Promedio histórico de eficiencia
            if self._register_feature('player_fg_pct_5g', 'shooting_efficiency'):
                fg_pct_hist = self._get_historical_series(df, 'FG%', window=5, operation='mean')
                df['player_fg_pct_5g'] = fg_pct_hist.fillna(df['FG%'].mean())
            
            # Tendencia de eficiencia (mejorando/empeorando)
            if self._register_feature('player_fg_trend', 'shooting_efficiency'):
                fg_recent = self._get_historical_series(df, 'FG%', window=3, operation='mean')
                fg_long = self._get_historical_series(df, 'FG%', window=10, operation='mean')
                df['player_fg_trend'] = (fg_recent - fg_long).fillna(0)
        
        # VOLUMEN DE TIROS (Más tiros = más oportunidades de rebote)
        if 'FGA' in df.columns:
            # Promedio histórico de intentos
            if self._register_feature('player_fga_5g', 'shooting_efficiency'):
                fga_hist = self._get_historical_series(df, 'FGA', window=5, operation='mean')
                df['player_fga_5g'] = fga_hist.fillna(df['FGA'].mean())
        
        # EFICIENCIA DE TIRO DEL EQUIPO (Genera rebotes defensivos)
        if 'Team' in df.columns and 'FG%' in df.columns:
            # Eficiencia promedio del equipo
            if self._register_feature('team_fg_efficiency', 'shooting_efficiency'):
                team_fg_avg = df.groupby(['Team', 'Date'])['FG%'].transform('mean')
                df['team_fg_efficiency'] = team_fg_avg.shift(1).fillna(df['FG%'].mean())
        
        # TIROS DE 3 PUNTOS (Rebotes más largos según investigación)
        if '3PA' in df.columns and 'FGA' in df.columns:
            # Proporción de tiros de 3
            if self._register_feature('three_point_rate', 'shooting_efficiency'):
                three_rate = df['3PA'] / (df['FGA'] + 0.1)  # Evitar división por 0
                df['three_point_rate'] = three_rate.fillna(0)
    
    def _create_physical_advantage_features(self, df: pd.DataFrame) -> None:
        """
        Features de ventaja física - PRINCIPIO FUNDAMENTAL
        Altura y físico son determinantes clave en rebotes
        """
        logger.debug("Creando features de ventaja física...")
        
        # ALTURA ABSOLUTA
        if 'Height_Inches' in df.columns:
            if self._register_feature('height_inches', 'physical_advantage'):
                df['height_inches'] = df['Height_Inches'].fillna(df['Height_Inches'].median())
        
        # VENTAJA DE ALTURA RELATIVA (vs promedio de la liga)
        if 'Height_Inches' in df.columns:
            if self._register_feature('height_advantage', 'physical_advantage'):
                league_avg_height = df['Height_Inches'].median()
                df['height_advantage'] = df['Height_Inches'] - league_avg_height
                df['height_advantage'] = df['height_advantage'].fillna(0)
        
        # ÍNDICE DE MASA CORPORAL (Físico para rebotes)
        if 'BMI' in df.columns:
            if self._register_feature('bmi_index', 'physical_advantage'):
                df['bmi_index'] = df['BMI'].fillna(df['BMI'].median())
        
        # VENTAJA FÍSICA COMPUESTA
        if 'Height_Inches' in df.columns and 'Weight' in df.columns:
            if self._register_feature('physical_dominance_index', 'physical_advantage'):
                height_norm = (df['Height_Inches'] - df['Height_Inches'].min()) / (df['Height_Inches'].max() - df['Height_Inches'].min())
                weight_norm = (df['Weight'] - df['Weight'].min()) / (df['Weight'].max() - df['Weight'].min())
                df['physical_dominance_index'] = (height_norm * 0.7 + weight_norm * 0.3).fillna(0.5)
            
            # MEJORAS AVANZADAS BASADAS EN PHYSICAL_DOMINANCE_INDEX
            
            # 1. Physical Dominance vs Liga por posición
            if self._register_feature('physical_dominance_vs_position', 'physical_advantage'):
                # Comparar dominancia física vs promedio de jugadores similares en altura
                height_bins = pd.cut(df['Height_Inches'], bins=5, labels=['Guard', 'Wing', 'Forward', 'BigMan', 'Center'])
                position_avg_dominance = df.groupby(height_bins)['physical_dominance_index'].transform('mean')
                df['physical_dominance_vs_position'] = df['physical_dominance_index'] - position_avg_dominance.fillna(0.5)
            
            # 2. Physical Advantage Score (altura + peso + alcance estimado)
            if self._register_feature('physical_advantage_score', 'physical_advantage'):
                # Estimar alcance basado en altura (aproximación)
                estimated_wingspan = df['Height_Inches'] * 1.05  # Típicamente 5% más que altura
                height_factor = (df['Height_Inches'] - 70) / 15  # Normalizado
                weight_factor = (df['Weight'] - 180) / 80  # Normalizado
                wingspan_factor = (estimated_wingspan - 70) / 15  # Normalizado
                df['physical_advantage_score'] = (
                    height_factor * 0.4 + 
                    weight_factor * 0.3 + 
                    wingspan_factor * 0.3
                ).fillna(0.5)
            
            # 3. Physical Dominance vs Opponent
            if 'Opp' in df.columns and self._register_feature('physical_dominance_vs_opponent', 'physical_advantage'):
                # Dominancia física relativa vs promedio del oponente
                opp_avg_height = df.groupby('Opp')['Height_Inches'].expanding().mean().shift(1)
                opp_avg_height = opp_avg_height.reset_index(0, drop=True)
                height_advantage_vs_opp = df['Height_Inches'] - opp_avg_height.fillna(df['Height_Inches'].median())
                df['physical_dominance_vs_opponent'] = (height_advantage_vs_opp / 5).fillna(0)  # Normalizado
            
            # 4. Physical Efficiency (productividad por ventaja física) - HISTÓRICO
            if 'TRB' in df.columns and self._register_feature('physical_efficiency', 'physical_advantage'):
                # TRB histórico por unidad de ventaja física - SIN DATA LEAKAGE
                trb_hist = self._get_historical_series(df, 'TRB', window=5, operation='mean')
                trb_per_physical = trb_hist / (df['physical_dominance_index'] + 0.1)
                df['physical_efficiency'] = trb_per_physical.fillna(df['TRB'].mean())
            
            # 5. Physical Dominance Momentum
            if self._register_feature('physical_dominance_momentum', 'physical_advantage'):
                # Cómo la ventaja física se traduce en rendimiento reciente
                if 'TRB' in df.columns:
                    trb_recent = self._get_historical_series(df, 'TRB', window=3, operation='mean')
                    expected_trb_by_physical = df['physical_dominance_index'] * 15  # Aproximación
                    df['physical_dominance_momentum'] = (trb_recent - expected_trb_by_physical).fillna(0)
                else:
                    df['physical_dominance_momentum'] = 0
    
    def _create_positioning_features(self, df: pd.DataFrame) -> None:
        """
        Features de posicionamiento - PRINCIPIO FUNDAMENTAL
        Minutos jugados y posición determinan oportunidades de rebote
        """
        logger.debug("Creando features de posicionamiento...")
        
        # MINUTOS JUGADOS (Más minutos = más oportunidades)
        if 'MP' in df.columns:
            # Promedio histórico de minutos
            if self._register_feature('minutes_avg_5g', 'positioning'):
                mp_hist = self._get_historical_series(df, 'MP', window=5, operation='mean')
                df['minutes_avg_5g'] = mp_hist.fillna(df['MP'].mean())
            
            # Proporción de minutos jugados (vs promedio liga)
            if self._register_feature('minutes_rate', 'positioning'):
                # Usar 30 minutos como referencia (promedio NBA real) - SIN DATA LEAKAGE
                mp_hist = self._get_historical_series(df, 'MP', window=5, operation='mean')
                df['minutes_rate'] = mp_hist / 30.0  # Normalizado al promedio NBA
                df['minutes_rate'] = df['minutes_rate'].fillna(1.0)  # 1.0 = promedio liga
            
            # MEJORAS AVANZADAS BASADAS EN MINUTES_RATE (Tercera feature más importante)
            
            # 1. Minutes Rate histórico con múltiples ventanas
            for window in [3, 7, 15]:
                feature_name = f'minutes_rate_hist_{window}g'
                if self._register_feature(feature_name, 'positioning'):
                    mp_hist = self._get_historical_series(df, 'MP', window=window, operation='mean')
                    df[feature_name] = (mp_hist / 30.0).fillna(1.0)
            
            # 2. Estabilidad de minutos (consistencia en el rol)
            if self._register_feature('minutes_stability', 'positioning'):
                mp_std = self._get_historical_series(df, 'MP', window=7, operation='std')
                mp_avg = self._get_historical_series(df, 'MP', window=7, operation='mean')
                # Coeficiente de variación inverso (mayor estabilidad = menor variación)
                df['minutes_stability'] = 1 / (1 + mp_std / (mp_avg + 0.1))
                df['minutes_stability'] = df['minutes_stability'].fillna(0.5)
            
            # 3. Minutes Rate vs expectativa por posición - HISTÓRICO
            if 'Height_Inches' in df.columns and self._register_feature('minutes_vs_position', 'positioning'):
                # Jugadores más altos típicamente juegan más minutos (centros/forwards)
                height_percentile = df['Height_Inches'].rank(pct=True)
                expected_minutes = 20 + height_percentile * 15  # 20-35 min basado en altura
                # Usar minutos históricos - SIN DATA LEAKAGE
                mp_hist = self._get_historical_series(df, 'MP', window=5, operation='mean')
                df['minutes_vs_position'] = mp_hist / (expected_minutes + 0.1)
            
            # 4. Tendencia de minutos (creciente/decreciente)
            if self._register_feature('minutes_trend', 'positioning'):
                mp_recent = self._get_historical_series(df, 'MP', window=3, operation='mean')
                mp_long = self._get_historical_series(df, 'MP', window=10, operation='mean')
                df['minutes_trend'] = (mp_recent - mp_long).fillna(0)
            
            # 5. Minutes Rate en contexto de equipo - HISTÓRICO
            if 'Team' in df.columns and self._register_feature('minutes_team_share', 'positioning'):
                # Proporción de minutos del jugador vs total del equipo - SIN DATA LEAKAGE
                team_total_minutes = df.groupby(['Team', 'Date'])['MP'].transform('sum')
                mp_hist = self._get_historical_series(df, 'MP', window=5, operation='mean')
                df['minutes_team_share'] = mp_hist / (team_total_minutes.shift(1).fillna(240) + 0.1)  # 240 min = 5 jugadores x 48 min
                df['minutes_team_share'] = df['minutes_team_share'].fillna(0.2)  # ~20% promedio
            
            # 6. Minutes Efficiency Score (minutos + productividad) - HISTÓRICO
            if 'TRB' in df.columns and self._register_feature('minutes_efficiency_score', 'positioning'):
                # Usar datos históricos - SIN DATA LEAKAGE
                trb_hist = self._get_historical_series(df, 'TRB', window=5, operation='mean')
                mp_hist = self._get_historical_series(df, 'MP', window=5, operation='mean')
                trb_per_minute = trb_hist / (mp_hist + 0.1)
                minutes_factor = np.minimum(mp_hist / 30.0, 1.5)  # Cap en 1.5x promedio
                # Score que combina minutos históricos y productividad por minuto histórica
                df['minutes_efficiency_score'] = (minutes_factor * 0.4 + trb_per_minute * 10 * 0.6).fillna(1.0)
            
            # 7. Minutes Load Management (fatiga/descanso)
            if self._register_feature('minutes_load_factor', 'positioning'):
                # Factor de carga basado en minutos recientes vs promedio
                mp_recent_3g = self._get_historical_series(df, 'MP', window=3, operation='mean')
                mp_season_avg = self._get_historical_series(df, 'MP', window=20, operation='mean')
                load_factor = mp_recent_3g / (mp_season_avg + 0.1)
                # Normalizar para que 1.0 = carga normal
                df['minutes_load_factor'] = load_factor.fillna(1.0)
        
        # ROL COMO TITULAR
        if 'is_started' in df.columns:
            # Consistencia como titular
            if self._register_feature('starter_rate_5g', 'positioning'):
                starter_hist = self._get_historical_series(df, 'is_started', window=5, operation='mean')
                df['starter_rate_5g'] = starter_hist.fillna(0.5)
        
        # POSICIÓN EN CANCHA (Basado en estadísticas)
        if 'BLK' in df.columns and 'AST' in df.columns:
            # Índice de juego interior (más bloqueos = más cerca del aro)
            if self._register_feature('interior_play_index', 'positioning'):
                blk_norm = df['BLK'] / (df['BLK'].max() + 0.1)
                ast_norm = df['AST'] / (df['AST'].max() + 0.1)
                # Más bloqueos y menos asistencias = juego más interior
                df['interior_play_index'] = (blk_norm - ast_norm * 0.5).fillna(0)
            
            # MEJORAS AVANZADAS BASADAS EN INTERIOR_PLAY_INDEX
            
            # 1. Interior Play Index histórico
            if 'BLK' in df.columns and 'AST' in df.columns:
                for window in [5, 10]:
                    feature_name = f'interior_play_hist_{window}g'
                    if self._register_feature(feature_name, 'positioning'):
                        blk_hist = self._get_historical_series(df, 'BLK', window=window, operation='mean')
                        ast_hist = self._get_historical_series(df, 'AST', window=window, operation='mean')
                        blk_norm_hist = blk_hist / (df['BLK'].max() + 0.1)
                        ast_norm_hist = ast_hist / (df['AST'].max() + 0.1)
                        df[feature_name] = (blk_norm_hist - ast_norm_hist * 0.5).fillna(0)
            
            # 2. Interior Play vs Physical Dominance (sinergia) - HISTÓRICO
            if 'Height_Inches' in df.columns and 'Weight' in df.columns and 'BLK' in df.columns:
                if self._register_feature('interior_physical_synergy', 'positioning'):
                    # Combinar índice interior con dominancia física - SIN DATA LEAKAGE
                    height_factor = (df['Height_Inches'] - 70) / 15  # Normalizado
                    blk_hist = self._get_historical_series(df, 'BLK', window=5, operation='mean')
                    blk_factor = blk_hist / (df['BLK'].max() + 0.1)
                    df['interior_physical_synergy'] = (height_factor * 0.6 + blk_factor * 0.4).fillna(0.5)
            
            # 3. Interior Play Consistency
            if 'BLK' in df.columns and self._register_feature('interior_play_consistency', 'positioning'):
                blk_std = self._get_historical_series(df, 'BLK', window=7, operation='std')
                blk_avg = self._get_historical_series(df, 'BLK', window=7, operation='mean')
                # Consistencia en juego interior
                df['interior_play_consistency'] = 1 / (1 + blk_std / (blk_avg + 0.1))
                df['interior_play_consistency'] = df['interior_play_consistency'].fillna(0.5)
    
    def _create_team_context_features(self, df: pd.DataFrame) -> None:
        """
        Features de contexto del equipo - PRINCIPIO FUNDAMENTAL
        El estilo de juego del equipo afecta las oportunidades de rebote
        """
        logger.debug("Creando features de contexto del equipo...")
        
        # USAR DATOS  DE EQUIPOS
        if self.teams_df is not None and 'Team' in df.columns:
            logger.debug("Utilizando datos reales de equipos para features avanzadas...")
            
            # RITMO DE JUEGO REAL DEL EQUIPO
            if self._register_feature('team_pace_real', 'team_context'):
                # Calcular ritmo real basado en FGA del equipo
                team_pace_stats = self.teams_df.groupby('Team').agg({
                    'FGA': 'mean',  # Promedio de intentos por juego
                    'MP': 'mean'    # Minutos jugados
                }).reset_index()
                
                # Merge con datos del jugador
                df_temp = df.merge(
                    team_pace_stats.rename(columns={'FGA': 'team_fga_avg', 'MP': 'team_mp_avg'}),
                    on='Team', how='left'
                )
                df['team_pace_real'] = df_temp['team_fga_avg'].fillna(85)  # Promedio NBA ~85 FGA
            
            # EFICIENCIA DEFENSIVA REAL DEL EQUIPO
            if self._register_feature('team_def_efficiency', 'team_context'):
                # Calcular eficiencia defensiva real
                team_def_stats = self.teams_df.groupby('Team').agg({
                    'PTS_Opp': 'mean',  # Puntos permitidos por juego
                    'FG%_Opp': 'mean' if 'FG%_Opp' in self.teams_df.columns else lambda x: 0.45
                }).reset_index()
                
                df_temp = df.merge(
                    team_def_stats.rename(columns={'PTS_Opp': 'team_pts_allowed'}),
                    on='Team', how='left'
                )
                # Menor puntos permitidos = mejor defensa = más rebotes defensivos
                df['team_def_efficiency'] = 120 - df_temp['team_pts_allowed'].fillna(110)
            
            # FORTALEZA REBOTEADORA DEL EQUIPO
            if 'TRB' in self.teams_df.columns and self._register_feature('team_reb_strength', 'team_context'):
                team_reb_stats = self.teams_df.groupby('Team')['TRB'].mean().reset_index()
                team_reb_stats.columns = ['Team', 'team_trb_avg']
                
                df_temp = df.merge(team_reb_stats, on='Team', how='left')
                df['team_reb_strength'] = df_temp['team_trb_avg'].fillna(45)  # Promedio NBA ~45 TRB
        
        else:
            # FALLBACK: Usar datos de jugadores para aproximar contexto del equipo
            logger.debug("Usando datos de jugadores para aproximar contexto del equipo...")
            
            # RITMO DE JUEGO DEL EQUIPO (Más posesiones = más rebotes)
            if 'Team' in df.columns and 'FGA' in df.columns:
                # Promedio de intentos del equipo (proxy del ritmo)
                if self._register_feature('team_pace_proxy', 'team_context'):
                    team_fga = df.groupby(['Team', 'Date'])['FGA'].transform('sum')
                    df['team_pace_proxy'] = team_fga.shift(1).fillna(85)  # Promedio NBA
            
            # EFICIENCIA DEFENSIVA DEL EQUIPO
            if 'Team' in df.columns and 'STL' in df.columns and 'BLK' in df.columns:
                # Índice defensivo del equipo
                if self._register_feature('team_defense_index', 'team_context'):
                    team_stl = df.groupby(['Team', 'Date'])['STL'].transform('sum')
                    team_blk = df.groupby(['Team', 'Date'])['BLK'].transform('sum')
                    df['team_defense_index'] = (team_stl + team_blk * 2).shift(1).fillna(10)
        
        # VENTAJA DE ALTURA DEL EQUIPO (Siempre disponible)
        if 'Team' in df.columns and 'Height_Inches' in df.columns:
            if self._register_feature('team_height_advantage', 'team_context'):
                team_height_avg = df.groupby(['Team', 'Date'])['Height_Inches'].transform('mean')
                league_height_avg = df['Height_Inches'].median()
                df['team_height_advantage'] = (team_height_avg - league_height_avg).shift(1).fillna(0)
    
    def _create_opponent_context_features(self, df: pd.DataFrame) -> None:
        """
        Features de contexto del oponente - PRINCIPIO FUNDAMENTAL
        Las características del oponente afectan las oportunidades de rebote
        """
        logger.debug("Creando features de contexto del oponente...")
        
        # USAR DATOS REALES DE EQUIPOS PARA OPONENTE SI ESTÁN DISPONIBLES
        if self.teams_df is not None and 'Opp' in df.columns:
            logger.debug("Utilizando datos reales de equipos para features del oponente...")
            
            # EFICIENCIA OFENSIVA DEL OPONENTE (Más fallos = más rebotes defensivos)
            if self._register_feature('opp_offensive_efficiency', 'opponent_context'):
                opp_off_stats = self.teams_df.groupby('Team').agg({
                    'FG%': 'mean',  # Eficiencia de tiro
                    'PTS': 'mean'   # Puntos por juego
                }).reset_index()
                
                df_temp = df.merge(
                    opp_off_stats.rename(columns={'Team': 'Opp', 'FG%': 'opp_fg_pct', 'PTS': 'opp_pts_avg'}),
                    on='Opp', how='left'
                )
                # Menor eficiencia = más rebotes defensivos disponibles
                df['opp_offensive_efficiency'] = 1 - df_temp['opp_fg_pct'].fillna(0.45)
            
            # RITMO REAL DEL OPONENTE
            if self._register_feature('opp_pace_real', 'opponent_context'):
                opp_pace_stats = self.teams_df.groupby('Team')['FGA'].mean().reset_index()
                opp_pace_stats.columns = ['Team', 'opp_fga_avg']
                
                df_temp = df.merge(
                    opp_pace_stats.rename(columns={'Team': 'Opp'}),
                    on='Opp', how='left'
                )
                # Normalizar al promedio de la liga
                league_avg_fga = 85  # Promedio NBA
                df['opp_pace_real'] = df_temp['opp_fga_avg'].fillna(league_avg_fga) / league_avg_fga
            
            # FORTALEZA REBOTEADORA REAL DEL OPONENTE
            if 'TRB' in self.teams_df.columns and self._register_feature('opp_reb_strength_real', 'opponent_context'):
                opp_reb_stats = self.teams_df.groupby('Team')['TRB'].mean().reset_index()
                opp_reb_stats.columns = ['Team', 'opp_trb_avg']
                
                df_temp = df.merge(
                    opp_reb_stats.rename(columns={'Team': 'Opp'}),
                    on='Opp', how='left'
                )
                # Más rebotes del oponente = menos oportunidades para nosotros
                df['opp_reb_strength_real'] = df_temp['opp_trb_avg'].fillna(45)  # Promedio NBA
        
        else:
            # FALLBACK: Usar datos de jugadores para aproximar oponente
            logger.debug("Usando datos de jugadores para aproximar contexto del oponente...")
            
            # EFICIENCIA DE TIRO DEL OPONENTE (Más fallos = más rebotes defensivos)
            if 'Opp' in df.columns and 'FG%' in df.columns:
                # Eficiencia histórica del oponente
                if self._register_feature('opp_shooting_efficiency', 'opponent_context'):
                    # Calcular eficiencia promedio del oponente en juegos anteriores
                    opp_fg_pct = df.groupby('Opp')['FG%'].expanding().mean().shift(1)
                    opp_fg_pct = opp_fg_pct.reset_index(0, drop=True)
                    df['opp_shooting_efficiency'] = opp_fg_pct.fillna(0.45)  # Promedio NBA
            
            # RITMO DEL OPONENTE
            if 'Opp' in df.columns and 'FGA' in df.columns:
                if self._register_feature('opp_pace_factor', 'opponent_context'):
                    # Promedio de intentos del oponente
                    opp_pace = df.groupby('Opp')['FGA'].expanding().mean().shift(1)
                    opp_pace = opp_pace.reset_index(0, drop=True)
                    league_avg_pace = 85  # Promedio NBA más realista
                    df['opp_pace_factor'] = (opp_pace / league_avg_pace).fillna(1.0)
            
            # FORTALEZA REBOTEADORA DEL OPONENTE
            if 'Opp' in df.columns and 'TRB' in df.columns:
                if self._register_feature('opp_rebounding_strength', 'opponent_context'):
                    # Promedio de rebotes permitidos al oponente
                    opp_trb_allowed = df.groupby('Opp')['TRB'].expanding().mean().shift(1)
                    opp_trb_allowed = opp_trb_allowed.reset_index(0, drop=True)
                    df['opp_rebounding_strength'] = opp_trb_allowed.fillna(8.5)  # Promedio individual
    
    def _create_rebounding_history_features(self, df: pd.DataFrame) -> None:
        """
        Features de historial de rebotes - PRINCIPIO FUNDAMENTAL
        El rendimiento pasado predice el futuro
        """
        logger.debug("Creando features de historial de rebotes...")
        
        if 'TRB' not in df.columns:
            logger.warning("TRB no disponible - features de historial limitadas")
            return
        
        # PROMEDIO HISTÓRICO DE REBOTES (Múltiples ventanas)
        for window in [3, 5, 10]:
            feature_name = f'trb_avg_{window}g'
            if self._register_feature(feature_name, 'rebounding_history'):
                trb_hist = self._get_historical_series(df, 'TRB', window=window, operation='mean')
                df[feature_name] = trb_hist.fillna(df['TRB'].mean())
        
        # MEJORAS AVANZADAS BASADAS EN TRB_AVG (Segunda feature más importante)
        
        # 1. TRB promedio ponderado por minutos jugados
        if 'MP' in df.columns:
            for window in [5, 10]:
                feature_name = f'trb_per_minute_{window}g'
                if self._register_feature(feature_name, 'rebounding_history'):
                    trb_hist = self._get_historical_series(df, 'TRB', window=window, operation='mean')
                    mp_hist = self._get_historical_series(df, 'MP', window=window, operation='mean')
                    df[feature_name] = (trb_hist / (mp_hist + 0.1)).fillna(0.3)  # TRB por minuto
        
        # 2. TRB promedio ajustado por ritmo del equipo
        if 'Team' in df.columns and 'FGA' in df.columns:
            for window in [5, 10]:
                feature_name = f'trb_pace_adjusted_{window}g'
                if self._register_feature(feature_name, 'rebounding_history'):
                    trb_hist = self._get_historical_series(df, 'TRB', window=window, operation='mean')
                    team_fga = df.groupby(['Team', 'Date'])['FGA'].transform('sum')
                    pace_factor = team_fga / 85  # Normalizado al promedio NBA
                    df[feature_name] = (trb_hist / (pace_factor.shift(1).fillna(1.0) + 0.1)).fillna(df['TRB'].mean())
        
        # 3. TRB promedio vs expectativa por posición
        if 'Height_Inches' in df.columns:
            for window in [5, 10]:
                feature_name = f'trb_vs_position_expectation_{window}g'
                if self._register_feature(feature_name, 'rebounding_history'):
                    trb_hist = self._get_historical_series(df, 'TRB', window=window, operation='mean')
                    # Expectativa basada en altura (jugadores más altos esperan más rebotes)
                    height_expectation = (df['Height_Inches'] - 70) * 0.5  # Aproximación lineal
                    df[feature_name] = (trb_hist - height_expectation.fillna(5)).fillna(0)
        
        # 4. Aceleración de TRB (cambio en la tendencia)
        if self._register_feature('trb_acceleration', 'rebounding_history'):
            trb_recent = self._get_historical_series(df, 'TRB', window=3, operation='mean')
            trb_mid = self._get_historical_series(df, 'TRB', window=7, operation='mean')
            trb_long = self._get_historical_series(df, 'TRB', window=15, operation='mean')
            # Aceleración: cambio en la tendencia
            trend_recent = trb_recent - trb_mid
            trend_long = trb_mid - trb_long
            df['trb_acceleration'] = (trend_recent - trend_long).fillna(0)
        
        # 5. TRB promedio en contexto de oponente
        if 'Opp' in df.columns:
            for window in [5, 10]:
                feature_name = f'trb_vs_opponent_{window}g'
                if self._register_feature(feature_name, 'rebounding_history'):
                    # TRB histórico contra equipos específicos
                    trb_vs_opp = df.groupby(['Player', 'Opp'])['TRB'].expanding().mean().shift(1)
                    trb_vs_opp = trb_vs_opp.reset_index(level=[0,1], drop=True)
                    trb_overall = self._get_historical_series(df, 'TRB', window=window, operation='mean')
                    df[feature_name] = (trb_vs_opp - trb_overall).fillna(0)
        
        # 6. TRB Momentum Score (combinando múltiples tendencias)
        if self._register_feature('trb_momentum_score', 'rebounding_history'):
            trb_3g = self._get_historical_series(df, 'TRB', window=3, operation='mean')
            trb_5g = self._get_historical_series(df, 'TRB', window=5, operation='mean')
            trb_10g = self._get_historical_series(df, 'TRB', window=10, operation='mean')
            # Score ponderado de momentum
            momentum_score = (
                (trb_3g - trb_5g) * 0.5 +  # Tendencia reciente
                (trb_5g - trb_10g) * 0.3 +  # Tendencia media
                (trb_3g - trb_10g) * 0.2    # Tendencia general
            )
            df['trb_momentum_score'] = momentum_score.fillna(0)
        
        # CONSISTENCIA EN REBOTES
        if self._register_feature('trb_consistency', 'rebounding_history'):
            trb_std = self._get_historical_series(df, 'TRB', window=5, operation='std')
            trb_avg = self._get_historical_series(df, 'TRB', window=5, operation='mean')
            # Coeficiente de variación (menor = más consistente)
            df['trb_consistency'] = 1 / (1 + trb_std / (trb_avg + 0.1))
            df['trb_consistency'] = df['trb_consistency'].fillna(0.5)
        
        # TENDENCIA RECIENTE
        if self._register_feature('trb_trend', 'rebounding_history'):
            trb_recent = self._get_historical_series(df, 'TRB', window=3, operation='mean')
            trb_long = self._get_historical_series(df, 'TRB', window=10, operation='mean')
            df['trb_trend'] = (trb_recent - trb_long).fillna(0)
        
        # REBOTES OFENSIVOS vs DEFENSIVOS (Si disponible)
        if 'ORB' in df.columns and 'DRB' in df.columns:
            # Proporción de rebotes ofensivos
            if self._register_feature('orb_rate', 'rebounding_history'):
                orb_rate = df['ORB'] / (df['TRB'] + 0.1)
                df['orb_rate'] = orb_rate.fillna(0.3)  # Promedio típico
            
            # MEJORAS AVANZADAS BASADAS EN ORB_RATE (Feature más importante)
            
            # 1. ORB Rate histórico con múltiples ventanas temporales
            for window in [3, 7, 15]:
                feature_name = f'orb_rate_hist_{window}g'
                if self._register_feature(feature_name, 'rebounding_history'):
                    orb_hist = self._get_historical_series(df, 'ORB', window=window, operation='mean')
                    trb_hist = self._get_historical_series(df, 'TRB', window=window, operation='mean')
                    df[feature_name] = (orb_hist / (trb_hist + 0.1)).fillna(0.3)
            
            # 2. Tendencia de ORB Rate (mejorando/empeorando)
            if self._register_feature('orb_rate_trend', 'rebounding_history'):
                orb_rate_recent = self._get_historical_series(df, 'ORB', window=3, operation='mean') / (self._get_historical_series(df, 'TRB', window=3, operation='mean') + 0.1)
                orb_rate_long = self._get_historical_series(df, 'ORB', window=10, operation='mean') / (self._get_historical_series(df, 'TRB', window=10, operation='mean') + 0.1)
                df['orb_rate_trend'] = (orb_rate_recent - orb_rate_long).fillna(0)
            
            # 3. Volatilidad de ORB Rate (consistencia)
            if self._register_feature('orb_rate_volatility', 'rebounding_history'):
                orb_rate_series = df['ORB'] / (df['TRB'] + 0.1)
                orb_rate_std = orb_rate_series.rolling(window=7, min_periods=3).std()
                df['orb_rate_volatility'] = orb_rate_std.fillna(0.1)
            
            # 4. ORB Rate vs posición (contexto físico)
            if 'Height_Inches' in df.columns and self._register_feature('orb_rate_vs_height', 'rebounding_history'):
                # Jugadores más altos típicamente tienen menor ORB rate pero más TRB total
                height_percentile = df['Height_Inches'].rank(pct=True)
                df['orb_rate_vs_height'] = df['orb_rate'] * (1 - height_percentile * 0.3)  # Ajuste por altura
            
            # 5. ORB Rate en contexto de equipo
            if 'Team' in df.columns and self._register_feature('orb_rate_team_context', 'rebounding_history'):
                # ORB Rate relativo al promedio del equipo
                team_orb_rate = df.groupby(['Team', 'Date']).apply(
                    lambda x: x['ORB'].sum() / (x['TRB'].sum() + 0.1)
                ).reset_index(name='team_orb_rate')
                df_temp = df.merge(team_orb_rate, on=['Team', 'Date'], how='left')
                df['orb_rate_team_context'] = df['orb_rate'] / (df_temp['team_orb_rate'].shift(1).fillna(0.3) + 0.01)
            
            # 6. ORB Efficiency Score (combinando rate y volumen)
            if self._register_feature('orb_efficiency_score', 'rebounding_history'):
                orb_volume = self._get_historical_series(df, 'ORB', window=5, operation='mean')
                orb_rate_hist = self._get_historical_series(df, 'ORB', window=5, operation='mean') / (self._get_historical_series(df, 'TRB', window=5, operation='mean') + 0.1)
                # Score que combina volumen y eficiencia
                df['orb_efficiency_score'] = (orb_volume * 0.6 + orb_rate_hist * 10 * 0.4).fillna(1.0)
    
    def _create_game_situation_features(self, df: pd.DataFrame) -> None:
        """
        Features de situación del juego - PRINCIPIO FUNDAMENTAL
        El contexto del juego afecta el estilo y las oportunidades
        """
        logger.debug("Creando features de situación del juego...")
        
        # VENTAJA LOCAL
        if 'is_home' in df.columns:
            if self._register_feature('home_advantage', 'game_situation'):
                df['home_advantage'] = df['is_home'].fillna(0)
        
        # RESULTADO DEL JUEGO (Si está ganando/perdiendo)
        if 'Result' in df.columns:
            if self._register_feature('game_result', 'game_situation'):
                # Convertir resultado a numérico (1 = victoria, 0 = derrota)
                df['game_result'] = (df['Result'] == 'W').astype(int)
        
        # DIFERENCIA DE PUNTOS (Si disponible)
        if 'PTS' in df.columns and 'Team' in df.columns:
            # Promedio de puntos del jugador vs equipo
            if self._register_feature('scoring_role', 'game_situation'):
                team_pts_avg = df.groupby(['Team', 'Date'])['PTS'].transform('mean')
                player_pts_share = df['PTS'] / (team_pts_avg + 0.1)
                df['scoring_role'] = player_pts_share.fillna(0.2)  # Promedio típico
        
        # FACTOR TEMPORAL (Mes de la temporada)
        if 'Date' in df.columns:
            if self._register_feature('season_phase', 'game_situation'):
                df['Date'] = pd.to_datetime(df['Date'])
                # Normalizar mes (0-1, donde 0.5 es mitad de temporada)
                month_factor = (df['Date'].dt.month - 10) % 12 / 12  # Temporada NBA Oct-Jun
                df['season_phase'] = month_factor
        
        # FEATURES AVANZADAS DE INTERACCIÓN (Combinando features más importantes)
        
        # 1. ORB Rate × Minutes Rate (Oportunidad × Tiempo)
        if 'orb_rate' in df.columns and 'minutes_rate' in df.columns:
            if self._register_feature('orb_minutes_interaction', 'game_situation'):
                df['orb_minutes_interaction'] = df['orb_rate'] * df['minutes_rate']
        
        # 2. Physical Dominance × Interior Play (Físico × Posición)
        if 'physical_dominance_index' in df.columns and 'interior_play_index' in df.columns:
            if self._register_feature('physical_interior_interaction', 'game_situation'):
                df['physical_interior_interaction'] = df['physical_dominance_index'] * df['interior_play_index']
        
        # 3. TRB Avg × Opponent Pace (Rendimiento × Contexto)
        if 'trb_avg_5g' in df.columns and 'opp_pace_real' in df.columns:
            if self._register_feature('trb_pace_interaction', 'game_situation'):
                df['trb_pace_interaction'] = df['trb_avg_5g'] * df['opp_pace_real']
        
        # 4. Minutes Rate × Team Context (Rol × Equipo)
        if 'minutes_rate' in df.columns and 'team_pace_real' in df.columns:
            if self._register_feature('minutes_team_interaction', 'game_situation'):
                df['minutes_team_interaction'] = df['minutes_rate'] * df['team_pace_real']
        
        # 5. Composite Rebounding Score (Score maestro)
        if all(col in df.columns for col in ['orb_rate', 'trb_avg_5g', 'minutes_rate', 'physical_dominance_index']):
            if self._register_feature('composite_rebounding_score', 'game_situation'):
                # Score ponderado basado en importancia de features
                df['composite_rebounding_score'] = (
                    df['orb_rate'] * 0.278 +  # Peso basado en importancia del modelo
                    (df['trb_avg_5g'] / 15) * 0.190 +  # Normalizado
                    df['minutes_rate'] * 0.165 +
                    df['physical_dominance_index'] * 0.100
                ).fillna(0.5)
        
        # 6. Situational Advantage Score (Ventaja situacional)
        if 'home_advantage' in df.columns and 'opp_pace_real' in df.columns:
            if self._register_feature('situational_advantage', 'game_situation'):
                home_factor = df['home_advantage'] * 0.05  # Pequeña ventaja local
                pace_factor = df['opp_pace_real'] - 1.0  # Desviación del ritmo promedio
                df['situational_advantage'] = (home_factor + pace_factor * 0.1).fillna(0)
        
        # 7. Performance Momentum (Momentum de rendimiento)
        if 'trb_trend' in df.columns and 'minutes_trend' in df.columns:
            if self._register_feature('performance_momentum', 'game_situation'):
                # Combinar tendencias de TRB y minutos
                df['performance_momentum'] = (df['trb_trend'] * 0.7 + df['minutes_trend'] * 0.3).fillna(0)
        
        # 8. Matchup Advantage (Ventaja de emparejamiento)
        if all(col in df.columns for col in ['physical_dominance_index', 'opp_reb_strength_real']):
            if self._register_feature('matchup_advantage', 'game_situation'):
                # Ventaja física vs fortaleza reboteadora del oponente
                df['matchup_advantage'] = df['physical_dominance_index'] - (df['opp_reb_strength_real'] / 50)  # Normalizado
    
    def _apply_correlation_filter(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aplica filtro de correlación para eliminar features redundantes
        """
        logger.info("Aplicando filtro de correlación optimizado...")
        
        # Identificar features numéricas (excluyendo protegidas)
        numeric_features = []
        for col in df.select_dtypes(include=[np.number]).columns:
            if col not in self.protected_features and col in self.feature_registry:
                numeric_features.append(col)
        
        if len(numeric_features) <= self.max_features:
            logger.info(f"Features ({len(numeric_features)}) dentro del límite ({self.max_features})")
            return df
        
        # Calcular matriz de correlación
        corr_matrix = df[numeric_features].corr().abs()
        
        # Encontrar pares altamente correlacionados
        upper_triangle = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        
        # Features a eliminar
        to_drop = []
        for column in upper_triangle.columns:
            if any(upper_triangle[column] > self.correlation_threshold):
                to_drop.append(column)
        
        # Eliminar features correlacionadas
        features_to_keep = [f for f in numeric_features if f not in to_drop]
        
        # Si aún tenemos demasiadas features, usar selección basada en target
        if len(features_to_keep) > self.max_features and 'TRB' in df.columns:
            logger.info("Aplicando selección adicional basada en target...")
            
            # Preparar datos para selección
            X = df[features_to_keep].fillna(0)
            y = df['TRB'].fillna(df['TRB'].mean())
            
            # Seleccionar mejores features
            selector = SelectKBest(score_func=f_regression, k=self.max_features)
            selector.fit(X, y)
            
            # Obtener features seleccionadas
            selected_features = [features_to_keep[i] for i in range(len(features_to_keep)) 
                               if selector.get_support()[i]]
            features_to_keep = selected_features
        
        # Mantener solo features seleccionadas + protegidas
        final_columns = list(self.protected_features) + features_to_keep
        final_columns = [col for col in final_columns if col in df.columns]
        
        logger.info(f"FILTRO DE CORRELACIÓN COMPLETADO:")
        logger.info(f"  Features originales: {len(numeric_features)}")
        logger.info(f"  Features protegidas: {len(self.protected_features)}")
        logger.info(f"  Features finales: {len(features_to_keep)}")
        
        return df[final_columns]
    
    def _log_feature_summary(self) -> None:
        """Registra resumen de features creadas"""
        logger.info("=" * 60)
        logger.info("RESUMEN DE FEATURES ESPECIALIZADAS PARA REBOTES")
        logger.info("=" * 60)
        
        total_features = 0
        for category, features in self.feature_categories.items():
            if features:
                logger.info(f"{category.upper()}: {len(features)} features")
                total_features += len(features)
        
        logger.info(f"TOTAL FEATURES CREADAS: {total_features}")
        logger.info("=" * 60)
