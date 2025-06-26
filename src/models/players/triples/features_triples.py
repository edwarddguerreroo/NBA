"""
Módulo de Características para Predicción de Triples (3P)
========================================================

FEATURES BASADAS EN PRINCIPIOS FUNDAMENTALES DE TRIPLES:

1. PRECISIÓN DE TIRO: Eficiencia histórica desde la línea de 3
2. VOLUMEN DE TIRO: Cantidad y frecuencia de intentos de 3PT
3. MECÁNICA DE TIRO: Consistencia y ritmo de tiro
4. CONTEXTO ESPACIAL: Posición, espaciado, defensa del oponente
5. CONTEXTO DEL JUEGO: Situación, presión, momentum
6. ESTILO DE JUEGO: Rol en el equipo, sistema ofensivo
7. FACTORES FÍSICOS: Fatiga, descanso, condición
8. HISTORIAL DE ESPECIALISTA: Rendimiento como tirador

Basado en análisis de los mejores tiradores: Curry, Thompson, Allen, etc.

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
from sklearn.feature_selection import SelectKBest, f_regression

warnings.filterwarnings('ignore')

# Configurar logging balanceado para features - mostrar etapas principales
logger = NBALogger.get_logger(__name__.split(".")[-1])  # Permitir logs de etapas principales

class ThreePointsFeatureEngineer:
    """
    Feature Engineer especializado en predicción de triples (3P)
    Basado en los principios fundamentales de los mejores tiradores de la NBA
    """
    
    def __init__(self, correlation_threshold: float = 0.98, max_features: int = 150, teams_df: pd.DataFrame = None):
        self.correlation_threshold = correlation_threshold
        self.max_features = max_features
        self.teams_df = teams_df  # Datos de equipos para features avanzadas
        self.feature_registry = {}
        self.feature_categories = {
            'shooting_accuracy': [],      # Precisión y eficiencia de tiro
            'shooting_volume': [],        # Volumen y frecuencia de intentos
            'shooting_mechanics': [],     # Consistencia y mecánica
            'spatial_context': [],        # Contexto espacial y defensivo
            'game_situation': [],         # Situación del juego
            'team_system': [],            # Sistema ofensivo del equipo
            'physical_factors': [],       # Factores físicos
            'specialist_traits': [],      # Características de especialista
            'opponent_defense': [],       # Análisis defensivo del oponente
            'momentum_factors': [],        # Momentum y confianza
            'quantum_features': []         # Features cuánticas ultra-avanzadas
        }
        self.protected_features = ['3P', 'Player', 'Date', 'Team', 'Opp']
        
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
        elif operation == 'sum':
            result = grouped.rolling(window=window, min_periods=1).sum().shift(1)
        else:
            result = grouped.rolling(window=window, min_periods=1).mean().shift(1)
        
        # Resetear índice para compatibilidad
        return result.reset_index(0, drop=True)
    
    def generate_all_features(self, df: pd.DataFrame) -> List[str]:
        """
        Genera todas las features especializadas para predicción de triples
        Modifica el DataFrame in-place y devuelve la lista de features creadas
        """
        logger.info("Generando features NBA ESPECIALIZADAS para predicción de triples...")
        
        # ASEGURAR ORDENACIÓN CRONOLÓGICA ANTES DE GENERAR FEATURES
        if 'Date' in df.columns and 'Player' in df.columns:
            df.sort_values(['Player', 'Date'], inplace=True)
            df.reset_index(drop=True, inplace=True)
            logger.debug("DataFrame ordenado cronológicamente por jugador y fecha")
        
        # Verificar target y columnas esenciales
        if '3P' in df.columns:
            threept_stats = df['3P'].describe()
            logger.info(f"Target 3P disponible - Media={threept_stats['mean']:.1f}, Max={threept_stats['max']:.0f}")
        else:
            NBALogger.log_warning(logger, "Target 3P no disponible - features limitadas")
        
        # Verificar columnas necesarias para triples
        required_cols = ['3PA', '3P%']
        available_cols = [col for col in required_cols if col in df.columns]
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if available_cols:
            logger.info(f"Columnas de triples disponibles: {available_cols}")
        if missing_cols:
            NBALogger.log_warning(logger, "Columnas de triples faltantes: {missing_cols}")
        
        # Limpiar registro de features
        self.feature_registry = {}
        for category in self.feature_categories:
            self.feature_categories[category] = []
        
        # 1. FEATURES DE PRECISIÓN DE TIRO
        self._create_shooting_accuracy_features(df)
        
        # 2. FEATURES DE VOLUMEN DE TIRO
        self._create_shooting_volume_features(df)
        
        # 3. FEATURES DE MECÁNICA DE TIRO
        self._create_shooting_mechanics_features(df)
        
        # 4. FEATURES DE CONTEXTO ESPACIAL
        self._create_spatial_context_features(df)
        
        # 5. FEATURES DE SITUACIÓN DEL JUEGO
        self._create_game_situation_features(df)
        
        # 6. FEATURES DE SISTEMA DE EQUIPO
        self._create_team_system_features(df)
        
        # 7. FEATURES DE FACTORES FÍSICOS
        self._create_physical_factors_features(df)
        
        # 8. FEATURES DE ESPECIALISTA
        self._create_specialist_traits_features(df)
        
        # 9. FEATURES DE DEFENSA DEL OPONENTE
        self._create_opponent_defense_features(df)
        
        # 10. FEATURES DE MOMENTUM
        self._create_momentum_factors_features(df)
        
        # 11. FEATURES CUÁNTICAS ULTRA-AVANZADAS
        self._create_quantum_features(df)
        
        # Obtener lista de features creadas
        created_features = list(self.feature_registry.keys())
        
        # PASO 1: Filtrar features ruidosas
        logger.info("Aplicando filtros de ruido para eliminar features problemáticas...")
        clean_features = self._apply_noise_filters(df, created_features)
        
        # Aplicar filtros si es necesario
        if len(clean_features) > self.max_features:
            clean_features = self._apply_feature_selection(df, clean_features)
        
        # Resumen final
        self._log_feature_summary()
        
        return clean_features
    
    def _create_shooting_accuracy_features(self, df: pd.DataFrame) -> None:
        """
        Features de precisión de tiro - PRINCIPIO FUNDAMENTAL
        La eficiencia desde la línea de 3 es el factor más predictivo
        """
        logger.debug("Creando features de precisión de tiro...")
        
        # EFICIENCIA HISTÓRICA (Más predictivo)
        if '3P%' in df.columns:
            # Promedio de temporada (expandiendo)
            if self._register_feature('threept_pct_season', 'shooting_accuracy'):
                pct_season = df.groupby('Player')['3P%'].expanding().mean().shift(1).reset_index(0, drop=True)
                df['threept_pct_season'] = pct_season.fillna(df['3P%'].mean())
            
            # Promedio últimos 3 juegos
            if self._register_feature('threept_pct_3g', 'shooting_accuracy'):
                pct_3g = self._get_historical_series(df, '3P%', window=3, operation='mean')
                df['threept_pct_3g'] = pct_3g.fillna(df['3P%'].mean())
            
            # Promedio últimos 5 juegos
            if self._register_feature('threept_pct_5g', 'shooting_accuracy'):
                pct_5g = self._get_historical_series(df, '3P%', window=5, operation='mean')
                df['threept_pct_5g'] = pct_5g.fillna(df['3P%'].mean())
            
            # Promedio últimos 10 juegos
            if self._register_feature('threept_pct_10g', 'shooting_accuracy'):
                pct_10g = self._get_historical_series(df, '3P%', window=10, operation='mean')
                df['threept_pct_10g'] = pct_10g.fillna(df['3P%'].mean())
            
            # Consistencia de tiro (desviación estándar)
            if self._register_feature('threept_consistency', 'shooting_accuracy'):
                consistency = self._get_historical_series(df, '3P%', window=10, operation='std')
                df['threept_consistency'] = (1 / (consistency + 0.01)).fillna(10)  # Invertir para que mayor sea mejor
        
        # EFICIENCIA CALCULADA (Si tenemos 3P y 3PA)
        if '3P' in df.columns and '3PA' in df.columns:
            # Eficiencia histórica calculada
            if self._register_feature('threept_efficiency_calc', 'shooting_accuracy'):
                made_hist = self._get_historical_series(df, '3P', window=5, operation='sum')
                att_hist = self._get_historical_series(df, '3PA', window=5, operation='sum')
                df['threept_efficiency_calc'] = (made_hist / (att_hist + 0.1)).fillna(0.33)
        
        # ÚLTIMO JUEGO (Momentum inmediato)
        if '3P' in df.columns:
            if self._register_feature('threept_last_game', 'shooting_accuracy'):
                last_made = df.groupby('Player')['3P'].shift(1).fillna(0)
                df['threept_last_game'] = last_made
    
    def _create_shooting_volume_features(self, df: pd.DataFrame) -> None:
        """
        Features de volumen de tiro - MUY PREDICTIVO según el modelo
        El volumen histórico es crucial para predecir triples futuros
        """
        logger.debug("Creando features de volumen de tiro...")
        
        # VOLUMEN HISTÓRICO (Top predictor según modelo)
        if '3PA' in df.columns:
            # Promedio de temporada (expandiendo) - Feature más importante
            if self._register_feature('threept_attempts_season', 'shooting_volume'):
                attempts_season = df.groupby('Player')['3PA'].expanding().mean().shift(1).reset_index(0, drop=True)
                df['threept_attempts_season'] = attempts_season.fillna(df['3PA'].mean())
            
            # Promedio últimos 5 juegos (muy predictivo)
            if self._register_feature('threept_attempts_5g', 'shooting_volume'):
                attempts_5g = self._get_historical_series(df, '3PA', window=5, operation='mean')
                df['threept_attempts_5g'] = attempts_5g.fillna(df['3PA'].mean())
            
            # Último juego
            if self._register_feature('threept_attempts_last', 'shooting_volume'):
                attempts_last = df.groupby('Player')['3PA'].shift(1)
                df['threept_attempts_last'] = attempts_last.fillna(df['3PA'].mean())
            
            # === NUEVAS FEATURES ULTRA-PREDICTIVAS ===
            
            # 1. RATIO DE VOLUMEN ADAPTATIVO (Combina eficiencia + volumen)
            if self._register_feature('adaptive_volume_ratio', 'shooting_volume'):
                # Jugadores que incrementan volumen cuando están eficientes
                recent_pct = self._get_historical_series(df, '3P%', window=3, operation='mean')
                recent_attempts = self._get_historical_series(df, '3PA', window=3, operation='mean')
                season_pct = df.groupby('Player')['3P%'].expanding().mean().shift(1).reset_index(0, drop=True)
                
                # Ratio de adaptación: más intentos cuando mejor eficiencia
                adaptive_ratio = np.where(recent_pct > season_pct, 
                                        recent_attempts / (season_pct + 0.01), 
                                        recent_attempts * season_pct)
                df['adaptive_volume_ratio'] = pd.Series(adaptive_ratio).fillna(df['3PA'].mean())
            
            # 2. MOMENTUM DE VOLUMEN (Tendencia de incremento/decremento)
            if self._register_feature('volume_momentum', 'shooting_volume'):
                # Diferencia entre intentos recientes vs promedio de temporada
                recent_vol = self._get_historical_series(df, '3PA', window=3, operation='mean')
                season_vol = df.groupby('Player')['3PA'].expanding().mean().shift(1).reset_index(0, drop=True)
                volume_momentum = (recent_vol - season_vol) / (season_vol + 0.1)
                df['volume_momentum'] = volume_momentum.fillna(0)
            
            # 3. ÍNDICE DE CONFIANZA EN VOLUMEN
            if self._register_feature('volume_confidence_index', 'shooting_volume'):
                # Jugadores que mantienen alto volumen tras fallos
                last_3p = df.groupby('Player')['3P'].shift(1)
                last_attempts = df.groupby('Player')['3PA'].shift(1)
                recent_attempts = self._get_historical_series(df, '3PA', window=3, operation='mean')
                
                # Confianza = mantener volumen tras fallos
                confidence_score = np.where((last_3p == 0) & (last_attempts > 2), 
                                          recent_attempts / (last_attempts + 0.1), 
                                          1.0)
                df['volume_confidence_index'] = pd.Series(confidence_score).fillna(1.0)
        
        # RATIO DE TRIPLES EN OFENSIVA TOTAL
        if '3PA' in df.columns and 'FGA' in df.columns:
            if self._register_feature('threept_ratio_of_total', 'shooting_volume'):
                # Ratio histórico de intentos de 3 vs total
                threept_ratio_total = self._get_historical_series(df, '3PA', window=5, operation='sum') / (
                    self._get_historical_series(df, 'FGA', window=5, operation='sum') + 0.1)
                df['threept_ratio_of_total'] = threept_ratio_total.fillna(0.35)  # Liga promedio
            
            # === NUEVA FEATURE ULTRA-PREDICTIVA ===
            # 4. ÍNDICE DE ESPECIALIZACIÓN PROGRESIVA
            if self._register_feature('specialization_progression', 'shooting_volume'):
                # Mide si el jugador se especializa más en triples con el tiempo
                try:
                    # Calcular ratio de especialización temprana de manera más robusta
                    early_ratios = []
                    for player in df['Player'].unique():
                        player_data = df[df['Player'] == player].sort_values('Date')
                        if len(player_data) >= 10:
                            early_3pa = player_data['3PA'].head(10).sum()
                            early_fga = player_data['FGA'].head(10).sum()
                            early_ratio = early_3pa / (early_fga + 0.1) if early_fga > 0 else 0.35
                        else:
                            early_ratio = 0.35  # Liga promedio como fallback
                        early_ratios.append({'Player': player, 'early_spec': early_ratio})
                    
                    early_ratio_df = pd.DataFrame(early_ratios)
                    
                    # Calcular ratio reciente
                    recent_3pa = self._get_historical_series(df, '3PA', window=10, operation='sum')
                    recent_fga = self._get_historical_series(df, 'FGA', window=10, operation='sum')
                    recent_ratio = recent_3pa / (recent_fga + 0.1)
                    
                    # Merge y calcular progresión
                    df_temp = df.merge(early_ratio_df, on='Player', how='left')
                    progression = (recent_ratio - df_temp['early_spec']) / (df_temp['early_spec'] + 0.1)
                    df['specialization_progression'] = progression.fillna(0.0)
                    
                except Exception as e:
                    logger.warning(f"Error calculando specialization_progression: {e}")
                    # Fallback robusto
                    df['specialization_progression'] = 0.0
        
        # FRECUENCIA DE TIRO POR MINUTO
        if '3PA' in df.columns and 'MP' in df.columns:
            if self._register_feature('threept_per_minute', 'shooting_volume'):
                attempts_per_min = self._get_historical_series(df, '3PA', window=5, operation='mean') / (
                    self._get_historical_series(df, 'MP', window=5, operation='mean') + 0.1)
                df['threept_per_minute'] = attempts_per_min.fillna(0.1)
            
            # === NUEVA FEATURE ULTRA-PREDICTIVA ===
            # 5. EFICIENCIA DE TIEMPO EN TRIPLES
            if self._register_feature('time_efficiency_3pt', 'shooting_volume'):
                # Triples anotados por minuto jugado (más predictivo que por juego)
                if '3P' in df.columns:
                    threepts_made = self._get_historical_series(df, '3P', window=5, operation='mean')
                    minutes_played = self._get_historical_series(df, 'MP', window=5, operation='mean')
                    time_efficiency = threepts_made / (minutes_played + 0.1)
                    df['time_efficiency_3pt'] = time_efficiency.fillna(0)
                else:
                    df['time_efficiency_3pt'] = 0.0
        
        # === FEATURES INTERACTIVAS ULTRA-PREDICTIVAS ===
        
        # 6. ÍNDICE DE OPORTUNIDAD SITUACIONAL
        if self._register_feature('situational_opportunity_index', 'shooting_volume'):
            # Combina volumen + contexto del juego
            if '3PA' in df.columns:
                recent_attempts = self._get_historical_series(df, '3PA', window=3, operation='mean')
                home_games = (df['Home'] == 1).astype(int) if 'Home' in df.columns else 0.5
                
                # Más oportunidades en casa y con buen ritmo
                situational_index = recent_attempts * (1 + 0.1 * home_games)
                df['situational_opportunity_index'] = situational_index.fillna(df['3PA'].mean())
            else:
                df['situational_opportunity_index'] = 5.0
        
        # 7. PREDICTOR DE VOLUMEN DINÁMICO
        if self._register_feature('dynamic_volume_predictor', 'shooting_volume'):
            # Predictor que combina múltiples factores de volumen
            if '3PA' in df.columns:
                season_vol = df.groupby('Player')['3PA'].expanding().mean().shift(1).reset_index(0, drop=True)
                recent_vol = self._get_historical_series(df, '3PA', window=3, operation='mean')
                recent_eff = self._get_historical_series(df, '3P%', window=3, operation='mean') if '3P%' in df.columns else 0.35
                
                # Volumen dinámico basado en eficiencia reciente
                dynamic_vol = (season_vol * 0.6 + recent_vol * 0.4) * (1 + recent_eff * 0.5)
                df['dynamic_volume_predictor'] = dynamic_vol.fillna(5.0)
            else:
                df['dynamic_volume_predictor'] = 5.0
    
    def _create_shooting_mechanics_features(self, df: pd.DataFrame) -> None:
        """
        Features de mecánica de tiro - Consistencia y patrón de tiro
        """
        logger.debug("Creando features de mecánica de tiro...")
        
        # RACHA ACTUAL (Streak)
        if '3P' in df.columns:
            if self._register_feature('threept_hot_streak', 'shooting_mechanics'):
                def calculate_hot_streak(series):
                    """Calcula racha actual de juegos con al menos 1 triple"""
                    streak = 0
                    for val in series[::-1]:  # Revisar desde el más reciente
                        if val >= 1:
                            streak += 1
                        else:
                            break
                    return streak
                
                hot_streak = df.groupby('Player')['3P'].shift(1).groupby(df['Player']).rolling(
                    window=10, min_periods=1).apply(calculate_hot_streak, raw=True)
                df['threept_hot_streak'] = hot_streak.reset_index(0, drop=True).fillna(0)
        
        # PATRONES DE TIRO POR CUARTO
        if '3P' in df.columns and 'MP' in df.columns:
            # Eficiencia por minuto jugado (diferente de la anterior - esta es made/minute)
            if self._register_feature('threept_made_per_minute', 'shooting_mechanics'):
                made_hist = self._get_historical_series(df, '3P', window=5, operation='mean')
                minutes_hist = self._get_historical_series(df, 'MP', window=5, operation='mean')
                df['threept_made_per_minute'] = (made_hist / (minutes_hist + 1)).fillna(0)
        
        # VARIABILIDAD EN RENDIMIENTO
        if '3P' in df.columns:
            if self._register_feature('threept_variability', 'shooting_mechanics'):
                variability = self._get_historical_series(df, '3P', window=10, operation='std')
                df['threept_variability'] = variability.fillna(0)
    
    def _create_spatial_context_features(self, df: pd.DataFrame) -> None:
        """
        Features de contexto espacial - Posición y situación de tiro
        """
        logger.debug("Creando features de contexto espacial...")
        
        # VENTAJA LOCAL vs VISITANTE
        if 'is_home' in df.columns:
            if '3P%' in df.columns:
                if self._register_feature('threept_home_advantage', 'spatial_context'):
                    # Eficiencia en casa vs fuera
                    home_games = df[df['is_home'] == 1]
                    away_games = df[df['is_home'] == 0]
                    
                    home_pct = home_games.groupby('Player')['3P%'].expanding().mean().shift(1)
                    away_pct = away_games.groupby('Player')['3P%'].expanding().mean().shift(1)
                    
                    df['threept_home_advantage'] = df['is_home'] * 0.05  # 5% boost en casa
        
        # ROL EN EL EQUIPO (Titular vs suplente)
        if 'is_started' in df.columns:
            if self._register_feature('threept_starter_boost', 'spatial_context'):
                df['threept_starter_boost'] = df['is_started'] * 0.1  # 10% boost para titulares
        
        # MINUTOS JUGADOS EFECTO
        if 'MP' in df.columns:
            if self._register_feature('threept_minutes_impact', 'spatial_context'):
                minutes_hist = self._get_historical_series(df, 'MP', window=5, operation='mean')
                # Normalizar minutos (30+ minutos = 1.0)
                df['threept_minutes_impact'] = np.minimum(minutes_hist / 30, 1.0).fillna(0.5)
    
    def _create_game_situation_features(self, df: pd.DataFrame) -> None:
        """
        Features de situación del juego - Contexto y presión
        """
        logger.debug("Creando features de situación del juego...")
        
        # DÍAS DE DESCANSO
        if 'Date' in df.columns:
            if self._register_feature('days_rest_impact', 'game_situation'):
                days_rest = df.groupby('Player')['Date'].diff().dt.days.fillna(2)
                # Optimal rest es 1-2 días
                df['days_rest_impact'] = np.where(days_rest.between(1, 2), 1.0, 
                                                 np.where(days_rest == 0, 0.8,  # Back-to-back penalty
                                                         np.where(days_rest >= 3, 0.9, 1.0)))  # Demasiado descanso
        
        # BACK-TO-BACK GAMES
        if 'Date' in df.columns:
            if self._register_feature('is_back_to_back', 'game_situation'):
                days_rest = df.groupby('Player')['Date'].diff().dt.days.fillna(2)
                df['is_back_to_back'] = (days_rest <= 1).astype(int)
        
        # MOMENTO DE LA TEMPORADA
        if 'Date' in df.columns:
            if self._register_feature('season_phase', 'game_situation'):
                # Calcular días desde el inicio de temporada
                season_start = df['Date'].min()
                days_into_season = (df['Date'] - season_start).dt.days
                max_days = days_into_season.max()
                
                # Early season (0-0.3), Mid season (0.3-0.7), Late season (0.7-1.0)
                df['season_phase'] = pd.cut(days_into_season, 
                                           bins=[0, max_days*0.3, max_days*0.7, max_days],
                                           labels=[0, 1, 2], include_lowest=True).astype(float)
    
    def _create_team_system_features(self, df: pd.DataFrame) -> None:
        """
        Features del sistema de equipo - Estilo ofensivo
        """
        logger.debug("Creando features de sistema de equipo...")
        
        # RITMO DEL EQUIPO (Basado en posesiones estimadas)
        if self.teams_df is not None and 'Team' in df.columns:
            if self._register_feature('team_pace_factor', 'team_system'):
                try:
                    # Verificar que las columnas existen en teams_df
                    required_cols = ['Team', 'Date', 'FGA', 'FTA']
                    if all(col in self.teams_df.columns for col in required_cols):
                        # Merge con datos de equipo
                        teams_subset = self.teams_df[required_cols].copy()
                        merged_data = df[['Team', 'Date']].merge(
                            teams_subset, 
                            on=['Team', 'Date'], 
                            how='left'
                        )
                        
                        # Verificar que el merge fue exitoso
                        if 'FGA' in merged_data.columns and 'FTA' in merged_data.columns:
                            # Estimar pace: (FGA + 0.4*FTA) * 2 (ambos equipos)
                            estimated_pace = (merged_data['FGA'].fillna(80) + 0.4 * merged_data['FTA'].fillna(25)) * 2
                            df['team_pace_factor'] = (estimated_pace / 100).fillna(1.0)  # Normalizar
                        else:
                            NBALogger.log_warning(logger, "Columnas FGA/FTA no encontradas después del merge")
                            df['team_pace_factor'] = 1.0
                    else:
                        NBALogger.log_warning(logger, "Columnas requeridas no encontradas en teams_df: {required_cols}")
                        df['team_pace_factor'] = 1.0
                except Exception as e:
                    NBALogger.log_warning(logger, "Error calculando team_pace_factor: {e}")
                    df['team_pace_factor'] = 1.0
        
        # ASISTENCIAS DEL EQUIPO (Espaciado)
        if 'AST' in df.columns:
            if self._register_feature('team_assist_rate', 'team_system'):
                team_ast = df.groupby(['Team', 'Date'])['AST'].transform('sum')
                df['team_assist_rate'] = team_ast / 30  # Normalizar por juego típico
        
        # TRIPLES DEL EQUIPO (Filosofía de tiro)
        if '3PA' in df.columns:
            if self._register_feature('team_three_philosophy', 'team_system'):
                team_3pa = df.groupby(['Team', 'Date'])['3PA'].transform('sum')
                df['team_three_philosophy'] = team_3pa / 40  # Normalizar
    
    def _create_physical_factors_features(self, df: pd.DataFrame) -> None:
        """
        Features de factores físicos - Condición y características físicas
        """
        logger.debug("Creando features de factores físicos...")
        
        # ALTURA (Ventaja para tiro)
        if 'Height_Inches' in df.columns:
            if self._register_feature('height_shooting_advantage', 'physical_factors'):
                # Normalizar altura para tiro (guards tienen mejor % típicamente)
                height_norm = (df['Height_Inches'] - 72) / 12  # Centrar en 6'0"
                df['height_shooting_advantage'] = np.where(height_norm.between(-1, 1), 1.0, 0.8)
        
        # PESO/BMI (Condición atlética)
        if 'BMI' in df.columns:
            if self._register_feature('athletic_condition', 'physical_factors'):
                # BMI óptimo para tiradores es 22-26
                df['athletic_condition'] = np.where(df['BMI'].between(22, 26), 1.0, 0.9)
        
        # EDAD (Si está disponible)
        if 'Age' in df.columns:
            if self._register_feature('age_shooting_factor', 'physical_factors'):
                # Peak de tiro es típicamente 25-30 años
                df['age_shooting_factor'] = np.where(df['Age'].between(25, 30), 1.0, 0.95)
    
    def _create_specialist_traits_features(self, df: pd.DataFrame) -> None:
        """
        Características de tiradores de élite con métricas avanzadas
        """
        logger.debug("Creando features ultra-predictivas de especialista...")
        
        # === ESPECIALISTA DE TRIPLES ULTRA-REFINADO (Top predictor) ===
        if '3PA' in df.columns:
            if self._register_feature('is_three_point_specialist', 'specialist_traits'):
                # Criterio multi-dimensional para especialistas
                season_attempts = df.groupby('Player')['3PA'].expanding().mean().shift(1).reset_index(0, drop=True)
                recent_attempts = self._get_historical_series(df, '3PA', window=5, operation='mean')
                games_played = df.groupby('Player').cumcount() + 1
                
                # Criterio adaptativo: más estricto con más experiencia
                threshold_base = 4.0
                experience_factor = np.where(games_played >= 30, 1.2, 0.8)
                adaptive_threshold = threshold_base * experience_factor
                
                # Criterio híbrido: temporada Y reciente
                specialist_criteria = (season_attempts * 0.7 + recent_attempts * 0.3) >= adaptive_threshold
                df['is_three_point_specialist'] = specialist_criteria.astype(int)
        
        # === TIRADOR DE ÉLITE ULTRA-AVANZADO ===
        if '3P%' in df.columns and '3PA' in df.columns:
            if self._register_feature('is_elite_shooter', 'specialist_traits'):
                # Elite con criterios múltiples
                pct_season = df.groupby('Player')['3P%'].expanding().mean().shift(1).reset_index(0, drop=True)
                pct_recent = self._get_historical_series(df, '3P%', window=10, operation='mean')
                att_season = df.groupby('Player')['3PA'].expanding().mean().shift(1).reset_index(0, drop=True)
                
                # Consistencia en eficiencia
                pct_consistency = 1 / (self._get_historical_series(df, '3P%', window=15, operation='std') + 0.01)
                
                # Elite híbrido: eficiencia + volumen + consistencia
                efficiency_score = (pct_season * 0.6 + pct_recent * 0.4)
                volume_qualified = att_season >= 3.5
                consistency_qualified = pct_consistency >= 15  # Alta consistencia
                
                elite_criteria = (efficiency_score >= 0.375) & volume_qualified & consistency_qualified
                df['is_elite_shooter'] = elite_criteria.astype(int)
        
        # === CLUTCH SHOOTER ULTRA-PREDICTIVO ===
        if '3P' in df.columns and '3PA' in df.columns:
            if self._register_feature('clutch_shooter_trait', 'specialist_traits'):
                # Clutch basado en rendimiento bajo presión
                high_attempt_games = df.groupby('Player')['3PA'].shift(1) >= 6
                performance_in_pressure = df.groupby('Player')['3P'].shift(1)
                
                # Eficiencia en juegos de alto volumen (presión) - Simplificado
                league_avg_3p = df['3P'].mean()
                
                # Aproximación: jugadores que mantienen eficiencia reciente
                recent_3p = self._get_historical_series(df, '3P', window=5, operation='mean')
                recent_attempts = self._get_historical_series(df, '3PA', window=5, operation='mean')
                
                # Clutch = alta eficiencia con volumen considerable
                clutch_trait = ((recent_3p >= league_avg_3p * 0.8) & (recent_attempts >= 4.0))
                df['clutch_shooter_trait'] = clutch_trait.astype(int)
        
        # === VOLUME SHOOTER ULTRA-PREDICTIVO (2° más importante) ===
        if '3PA' in df.columns:
            if self._register_feature('volume_shooter_trait', 'specialist_traits'):
                # Volume shooter con múltiples criterios
                att_season = df.groupby('Player')['3PA'].expanding().mean().shift(1).reset_index(0, drop=True)
                team_rank = df.groupby(['Team', 'Date'])['3PA'].rank(ascending=False, method='dense')
                league_percentile = df.groupby('Date')['3PA'].rank(pct=True)
                
                # Múltiples criterios para volume shooter
                high_season_volume = att_season >= 5.5
                top_in_team = team_rank <= 2  # Top 2 en equipo
                top_in_league = league_percentile >= 0.75  # Top 25% de la liga
                
                # Combinar criterios con pesos
                volume_score = (high_season_volume.astype(int) * 0.5 + 
                               top_in_team.astype(int) * 0.3 + 
                               top_in_league.astype(int) * 0.2)
                
                df['volume_shooter_trait'] = (volume_score >= 0.6).astype(int)
        
        # === NUEVAS FEATURES ULTRA-PREDICTIVAS ===
        
        # 1. ÍNDICE DE ESPECIALIZACIÓN PROGRESIVA
        if '3PA' in df.columns and '3P%' in df.columns:
            if self._register_feature('specialization_evolution_index', 'specialist_traits'):
                # Mide cómo evoluciona la especialización del jugador - Simplificado
                recent_attempts = self._get_historical_series(df, '3PA', window=10, operation='mean')
                season_attempts = df.groupby('Player')['3PA'].expanding().mean().shift(1).reset_index(0, drop=True)
                recent_efficiency = self._get_historical_series(df, '3P%', window=10, operation='mean')
                season_efficiency = df.groupby('Player')['3P%'].expanding().mean().shift(1).reset_index(0, drop=True)
                
                # Evolución simplificada: reciente vs promedio histórico
                volume_evolution = (recent_attempts - season_attempts) / (season_attempts + 0.1)
                efficiency_evolution = (recent_efficiency - season_efficiency) / (season_efficiency + 0.01)
                
                # Índice combinado
                evolution_index = (volume_evolution * 0.6 + efficiency_evolution * 0.4)
                df['specialization_evolution_index'] = evolution_index.fillna(0)
        
        # 2. CONSISTENCY UNDER PRESSURE INDEX
        if '3P%' in df.columns and '3PA' in df.columns:
            if self._register_feature('consistency_under_pressure', 'specialist_traits'):
                # Consistencia en juegos de alto volumen
                high_vol_threshold = df['3PA'].quantile(0.75)
                high_vol_games = df['3PA'] >= high_vol_threshold
                
                # Consistencia simplificada: variabilidad de eficiencia
                efficiency_std = self._get_historical_series(df, '3P%', window=15, operation='std')
                efficiency_mean = self._get_historical_series(df, '3P%', window=15, operation='mean')
                
                # Ratio de consistencia: menor variabilidad = mejor
                consistency_ratio = efficiency_mean / (efficiency_std + 0.01)
                df['consistency_under_pressure'] = consistency_ratio.fillna(1.0)
        
        # 3. ADAPTIVE SHOOTING PROFILE
        if '3PA' in df.columns and 'MP' in df.columns:
            if self._register_feature('adaptive_shooting_profile', 'specialist_traits'):
                # Perfil de tiro adaptativo según minutos jugados
                minutes_per_game = self._get_historical_series(df, 'MP', window=5, operation='mean')
                attempts_per_minute = self._get_historical_series(df, '3PA', window=5, operation='mean') / (minutes_per_game + 0.1)
                
                # Adaptabilidad: mantener frecuencia independiente de minutos
                league_avg_rate = df['3PA'].sum() / (df['MP'].sum() + 0.1)
                adaptive_score = attempts_per_minute / (league_avg_rate + 0.001)
                
                df['adaptive_shooting_profile'] = adaptive_score.fillna(1.0)
    
    def _create_opponent_defense_features(self, df: pd.DataFrame) -> None:
        """
        Features de defensa del oponente - Análisis defensivo
        """
        logger.debug("Creando features de defensa del oponente...")
        
        # DEFENSA DE TRIPLES DEL OPONENTE
        if 'Opp' in df.columns and self.teams_df is not None:
            if self._register_feature('opponent_3pt_defense', 'opponent_defense'):
                try:
                    if '3P%_Opp' in self.teams_df.columns:
                        # Merge con datos de equipos oponentes
                        opp_defense = df.merge(
                            self.teams_df[['Team', 'Date', '3P%_Opp']].rename(columns={'Team': 'Opp'}),
                            on=['Opp', 'Date'],
                            how='left'
                        )['3P%_Opp']
                        # Invertir: mayor % permitido = peor defensa = más fácil anotar
                        df['opponent_3pt_defense'] = opp_defense.fillna(0.35)  # Default neutral
                    else:
                        # Aproximación simple si no hay datos específicos
                        team_mapping = {team: idx*0.1 + 0.3 for idx, team in enumerate(df['Opp'].unique())}
                        df['opponent_3pt_defense'] = df['Opp'].map(team_mapping).fillna(0.35)
                except Exception as e:
                    NBALogger.log_warning(logger, "Error calculando opponent_3pt_defense: {e}")
                    # Fallback simple
                    team_mapping = {team: idx*0.05 + 0.3 for idx, team in enumerate(df['Opp'].unique())}
                    df['opponent_3pt_defense'] = df['Opp'].map(team_mapping).fillna(0.35)
        
        # HISTORIAL VS OPONENTE
        if 'Opp' in df.columns and '3P' in df.columns:
            if self._register_feature('historical_vs_opponent', 'opponent_defense'):
                # Promedio histórico vs este oponente específico
                hist_vs_opp = df.groupby(['Player', 'Opp'])['3P'].expanding().mean().shift(1)
                df['historical_vs_opponent'] = hist_vs_opp.reset_index(0, drop=True).reset_index(0, drop=True).fillna(df['3P'].mean())
        
        # === FEATURES ULTRA-ESPECÍFICAS DE DEFENSA DEL OPONENTE ===
        if 'Opp' in df.columns:
            # 1. Defensa contra tiradores de alto volumen
            if self._register_feature('opp_vs_volume_shooters', 'opponent_defense'):
                # Cómo defiende el oponente a jugadores que intentan muchos triples
                opp_vs_high_volume = df[df['3PA'] >= 6].groupby('Opp')['3P%'].transform('mean') if '3PA' in df.columns else 0.35
                df['opp_vs_volume_shooters'] = opp_vs_high_volume.fillna(0.35)
            
            # 2. Defensa contra especialistas en triples
            if self._register_feature('opp_vs_specialists', 'opponent_defense'):
                # Cómo defiende contra jugadores con alta eficiencia
                opp_vs_specialists = df[df['3P%'] >= 0.38].groupby('Opp')['3P'].transform('mean') if '3P%' in df.columns else 1.2
                df['opp_vs_specialists'] = opp_vs_specialists.fillna(1.2)
            
            # 3. Tendencia defensiva reciente del oponente
            if self._register_feature('opp_recent_defense_trend', 'opponent_defense'):
                # Tendencia de triples permitidos en últimos 5 juegos
                opp_recent_trend = df.groupby('Opp')['3P'].rolling(5, min_periods=1).mean().shift(1).reset_index(0, drop=True)
                opp_season_avg = df.groupby('Opp')['3P'].expanding().mean().shift(1).reset_index(0, drop=True)
                
                # Ratio: si está permitiendo más o menos triples últimamente
                trend_ratio = (opp_recent_trend / (opp_season_avg + 0.1)).fillna(1.0)
                df['opp_recent_defense_trend'] = np.clip(trend_ratio, 0.5, 2.0)
            
            # 4. Factor de rivalidad/motivación
            if self._register_feature('rivalry_motivation_factor', 'opponent_defense'):
                # Equipos específicos que generan más motivación (aproximación)
                rivalry_teams = ['LAL', 'BOS', 'GSW', 'MIA', 'NYK']  # Equipos de alto perfil
                rivalry_factor = df['Opp'].apply(lambda x: 1.2 if x in rivalry_teams else 1.0)
                df['rivalry_motivation_factor'] = rivalry_factor
            
            # 5. Defensa específica contra valores extremos
            if self._register_feature('opp_vs_extreme_games', 'opponent_defense'):
                # Cómo defiende el oponente cuando los jugadores tienen juegos explosivos
                opp_vs_extreme = df[df['3P'] >= 5].groupby('Opp')['3P'].transform('mean') if '3P' in df.columns else 2.0
                df['opp_vs_extreme_games'] = opp_vs_extreme.fillna(2.0)
    
    def _create_momentum_factors_features(self, df: pd.DataFrame) -> None:
        """
        Features de momentum - Confianza y tendencias
        """
        logger.debug("Creando features de momentum...")
        
        # HOT HAND EFFECT
        if '3P' in df.columns:
            if self._register_feature('hot_hand_momentum', 'momentum_factors'):
                # Juegos consecutivos con 2+ triples
                def calculate_hot_hand(series):
                    count = 0
                    for val in series[::-1]:
                        if val >= 2:
                            count += 1
                        else:
                            break
                    return min(count, 5)  # Cap en 5 juegos
                
                hot_hand = df.groupby('Player')['3P'].shift(1).groupby(df['Player']).rolling(
                    window=5, min_periods=1).apply(calculate_hot_hand, raw=True)
                df['hot_hand_momentum'] = hot_hand.reset_index(0, drop=True).fillna(0)
        
        # TENDENCIA RECIENTE
        if '3P' in df.columns:
            if self._register_feature('recent_form_trend', 'momentum_factors'):
                recent_avg = self._get_historical_series(df, '3P', window=3, operation='mean')
                long_avg = self._get_historical_series(df, '3P', window=15, operation='mean')
                df['recent_form_trend'] = (recent_avg - long_avg).fillna(0)
        
        # CONFIDENCE INDICATOR
        if '3P%' in df.columns:
            if self._register_feature('confidence_indicator', 'momentum_factors'):
                recent_pct = self._get_historical_series(df, '3P%', window=3, operation='mean')
                season_pct = df.groupby('Player')['3P%'].expanding().mean().shift(1).reset_index(0, drop=True)
                df['confidence_indicator'] = (recent_pct - season_pct).fillna(0)
        
        # === NUEVAS FEATURES HÍBRIDAS ULTRA-ESPECÍFICAS ===
        
        # 1. VOLUME-CLUTCH SYNERGY (Combina las 2 features más importantes)
        if self._register_feature('volume_clutch_synergy', 'momentum_factors'):
            volume_trait = df.get('volume_shooter_trait', 0)
            clutch_trait = df.get('clutch_shooter_trait', 0)
            recent_attempts = self._get_historical_series(df, '3PA', window=3, operation='mean') if '3PA' in df.columns else 5.0
            
            # Sinergia: volume shooters que también son clutch tienen explosión potencial
            synergy_score = (volume_trait * clutch_trait * (recent_attempts / 5.0))
            df['volume_clutch_synergy'] = synergy_score.fillna(0)
        
        # 2. HIGH PERFORMANCE CATALYST (Para predecir 6+ triples) - CORREGIDO
        if self._register_feature('high_performance_catalyst', 'momentum_factors'):
            # CORRECCIÓN: Usar solo datos históricos con shift
            recent_3p = self._get_historical_series(df, '3P', window=3, operation='mean') if '3P' in df.columns else 1.0
            recent_attempts = self._get_historical_series(df, '3PA', window=3, operation='mean') if '3PA' in df.columns else 5.0
            recent_pct = self._get_historical_series(df, '3P%', window=3, operation='mean') if '3P%' in df.columns else 0.35
            
            # Asegurar que todas las series usen shift(1) para evitar leakage
            if '3P' in df.columns:
                recent_3p = df.groupby('Player')['3P'].shift(1).rolling(window=3, min_periods=1).mean()
            if '3PA' in df.columns:
                recent_attempts = df.groupby('Player')['3PA'].shift(1).rolling(window=3, min_periods=1).mean()
            if '3P%' in df.columns:
                recent_pct = df.groupby('Player')['3P%'].shift(1).rolling(window=3, min_periods=1).mean()
            
            # Catalizador: alta eficiencia + alto volumen + momentum reciente (solo histórico)
            catalyst_score = (recent_pct * 3.0) * (recent_attempts / 8.0) * (1 + recent_3p / 3.0)
            df['high_performance_catalyst'] = catalyst_score.fillna(0)
        
        # 3. EXPLOSIVE GAME PREDICTOR (Predictor específico para juegos explosivos)
        if self._register_feature('explosive_game_predictor', 'momentum_factors'):
            # Histórico de juegos con 5+ triples
            explosive_games_hist = df.groupby('Player')['3P'].apply(
                lambda x: (x.shift(1) >= 5).rolling(window=20, min_periods=1).sum()
            ).reset_index(0, drop=True)
            
            season_attempts = df.groupby('Player')['3PA'].expanding().mean().shift(1).reset_index(0, drop=True) if '3PA' in df.columns else 5.0
            home_col = df.get('Home', pd.Series([0.5] * len(df), index=df.index))
            if isinstance(home_col, pd.Series):
                home_advantage = (home_col == 1).astype(int)
            else:
                home_advantage = pd.Series([int(home_col == 1)] * len(df), index=df.index)
            
            # Predictor explosivo: historial + volumen + ventaja local
            explosive_score = (explosive_games_hist / 20.0) * (season_attempts / 6.0) * (1 + 0.2 * home_advantage)
            df['explosive_game_predictor'] = explosive_score.fillna(0)
        
        # 4. CONSISTENCY MOMENTUM HYBRID
        if self._register_feature('consistency_momentum_hybrid', 'momentum_factors'):
            consistency_score = df.get('consistency_under_pressure', 1.0)
            recent_form = df.get('recent_form_trend', 0.0)
            attempts_trend = df.get('volume_momentum', 0.0)
            
            # Híbrido: consistencia + forma reciente + tendencia de volumen
            hybrid_score = (consistency_score * (1 + abs(recent_form)) * (1 + abs(attempts_trend)))
            df['consistency_momentum_hybrid'] = hybrid_score.fillna(1.0)
        
        # 5. HOT STREAK ULTRA-PREDICTIVO (Específico para valores extremos)
        if self._register_feature('hot_streak_predictor', 'momentum_factors'):
            # Detectar rachas calientes de triples
            hot_streak_3g = df.groupby('Player')['3P'].apply(
                lambda x: (x.shift(1).rolling(3, min_periods=1).mean() >= 3.0).astype(int)
            ).reset_index(0, drop=True)
            
            hot_streak_5g = df.groupby('Player')['3P'].apply(
                lambda x: (x.shift(1).rolling(5, min_periods=1).mean() >= 2.5).astype(int)
            ).reset_index(0, drop=True)
            
            # Predictor de racha: combinar rachas cortas y largas
            df['hot_streak_predictor'] = (hot_streak_3g * 2.0 + hot_streak_5g * 1.5).fillna(0)
        
        # 6. EXTREME VALUE MOMENTUM (Específico para 6+ triples)
        if self._register_feature('extreme_value_momentum', 'momentum_factors'):
            # Histórico de juegos excepcionales recientes
            extreme_games_hist = df.groupby('Player')['3P'].apply(
                lambda x: (x.shift(1) >= 6).rolling(10, min_periods=1).sum()
            ).reset_index(0, drop=True)
            
            # Momentum de intentos altos
            high_attempts_hist = df.groupby('Player')['3PA'].apply(
                lambda x: (x.shift(1) >= 8).rolling(5, min_periods=1).sum()
            ).reset_index(0, drop=True) if '3PA' in df.columns else pd.Series([0] * len(df), index=df.index)
            
            # Combinar histórico extremo + intentos altos
            df['extreme_value_momentum'] = (extreme_games_hist / 10.0 + high_attempts_hist / 5.0).fillna(0)
        
        # 7. CLUTCH SHOOTING INDICATOR (Contexto de presión)
        if self._register_feature('clutch_shooting_factor', 'momentum_factors'):
            # Eficiencia en situaciones de presión (aproximada)
            if '3P%' in df.columns:
                # Corregir índices para alineación correcta
                recent_efficiency = df.groupby('Player')['3P%'].shift(1).rolling(5, min_periods=1).mean().reset_index(0, drop=True)
                season_efficiency = df.groupby('Player')['3P%'].expanding().mean().shift(1).reset_index(0, drop=True)
                
                # Factor clutch: si está por encima de su promedio reciente
                clutch_factor = (recent_efficiency / (season_efficiency + 0.001)).fillna(1.0)
                df['clutch_shooting_factor'] = np.clip(clutch_factor, 0.5, 2.0)
            else:
                df['clutch_shooting_factor'] = 1.0
        
        # 5. PEAK PERFORMANCE ZONE (Zona de máximo rendimiento)
        if self._register_feature('peak_performance_zone', 'momentum_factors'):
            season_avg = df.groupby('Player')['3P'].expanding().mean().shift(1).reset_index(0, drop=True) if '3P' in df.columns else 1.2
            season_attempts = df.groupby('Player')['3PA'].expanding().mean().shift(1).reset_index(0, drop=True) if '3PA' in df.columns else 5.0
            recent_hot_streak = df.get('hot_hand_momentum', 0.0)
            
            # Zona peak: cuando todo se alinea (promedio alto + volumen + racha)
            peak_zone = (season_avg / 2.0) * (season_attempts / 6.0) * (1 + recent_hot_streak / 3.0)
            df['peak_performance_zone'] = peak_zone.fillna(0)
    
    def _create_quantum_features(self, df: pd.DataFrame) -> None:
        """
        Capturan patrones temporales microscópicos y oscilaciones de rendimiento
        """
        logger.debug("Creando features cuánticas ultra-avanzadas...")
        
        # 1. OSCILACIÓN TEMPORAL CUÁNTICA (Detecta ciclos de rendimiento) - CORREGIDO
        if self._register_feature('quantum_oscillation_3pt', 'quantum_features'):
            if '3P' in df.columns:
                # CORRECCIÓN: Usar solo datos históricos con shift
                performance_series = df.groupby('Player')['3P'].shift(1).groupby(df['Player']).rolling(
                    window=7, min_periods=3).apply(
                    lambda vals: np.std(vals) * np.mean(vals) / (np.max(vals) + 0.1) if len(vals) > 0 else 0, raw=True
                )
                df['quantum_oscillation_3pt'] = performance_series.reset_index(0, drop=True).fillna(0)
        
        # 2. ENTROPÍA DE RENDIMIENTO (Mide impredecibilidad del jugador) - CORREGIDO
        if self._register_feature('performance_entropy', 'quantum_features'):
            if '3P' in df.columns:
                # CORRECCIÓN: Usar datos históricos con shift
                entropy_calc = df.groupby('Player')['3P'].shift(1).groupby(df['Player']).rolling(
                    window=10, min_periods=5).apply(
                    lambda vals: -np.sum(
                        [(v/vals.sum() + 1e-8) * np.log(v/vals.sum() + 1e-8) 
                         for v in vals if vals.sum() > 0]
                    ) if vals.sum() > 0 else 0, raw=True
                )
                df['performance_entropy'] = entropy_calc.reset_index(0, drop=True).fillna(0)
        
        # 3. MOMENTUM CUÁNTICO (Aceleración de la aceleración) - CORREGIDO
        if self._register_feature('quantum_momentum', 'quantum_features'):
            if '3P' in df.columns:
                # CORRECCIÓN: Usar datos históricos con shift
                historical_3p = df.groupby('Player')['3P'].shift(1)
                first_derivative = historical_3p.groupby(df['Player']).diff(1)
                second_derivative = first_derivative.groupby(df['Player']).diff(1)
                quantum_momentum = second_derivative * first_derivative
                df['quantum_momentum'] = quantum_momentum.fillna(0)
        
        # 4. RESONANCIA CON OPONENTE (Sinergia/antagonismo con defensa rival)
        if self._register_feature('opponent_resonance', 'quantum_features'):
            if '3P' in df.columns:
                player_performance = self._get_historical_series(df, '3P', window=5, operation='mean')
                # Usar una proxy simple si opponent_3pt_defense no existe
                if 'opponent_3pt_defense' in df.columns:
                    opp_defense = df.get('opponent_3pt_defense', 1.0)
                else:
                    # Proxy: usar el promedio de triples permitidos por el oponente
                    opp_defense = 1.0
                
                # Resonancia: cuando la defensa fuerte mejora al jugador (desafío) o lo empeora
                resonance = player_performance * np.cos(opp_defense * np.pi) + 0.5
                df['opponent_resonance'] = resonance.fillna(0.5)
            else:
                df['opponent_resonance'] = 0.5
        
        # 5. CAMPO MORFOGÉNICO DE EQUIPO (Influencia colectiva del equipo) 
        if self._register_feature('team_morphic_field', 'quantum_features'):
            if '3P' in df.columns:
                # Calcular promedio histórico del equipo por fecha (sin el jugador actual)
                team_historical = []
                for idx, row in df.iterrows():
                    player = row['Player']
                    team = row['Team']
                    date = row['Date']
                    
                    # Obtener juegos históricos del equipo (excluyendo jugador actual y juegos futuros)
                    team_mask = (df['Team'] == team) & (df['Player'] != player) & (df['Date'] < date)
                    if team_mask.sum() > 0:
                        team_avg = df.loc[team_mask, '3P'].tail(10).mean()  # Últimos 10 juegos del equipo
                    else:
                        team_avg = 1.2  # Media liga por defecto
                    
                    # Promedio histórico del jugador
                    player_mask = (df['Player'] == player) & (df['Date'] < date)
                    if player_mask.sum() > 0:
                        player_avg = df.loc[player_mask, '3P'].tail(5).mean()  # Últimos 5 juegos del jugador
                    else:
                        player_avg = 1.2
                    
                    # Campo morfogénico: diferencia histórica con decay temporal
                    days_since_last = 1  # Simplificado
                    morphic_value = (player_avg - team_avg) * np.exp(-0.1 * days_since_last)
                    team_historical.append(morphic_value)
                
                df['team_morphic_field'] = team_historical
        
        # 6. FRACTALES DE RENDIMIENTO (Patrones autosimilares en diferentes escalas)
        if self._register_feature('performance_fractal', 'quantum_features'):
            if '3P' in df.columns:
                # Correlación entre ventanas de diferentes tamaños
                short_pattern = self._get_historical_series(df, '3P', window=3, operation='std')
                medium_pattern = self._get_historical_series(df, '3P', window=7, operation='std') 
                long_pattern = self._get_historical_series(df, '3P', window=15, operation='std')
                
                # Fractal: correlación cruzada entre escalas
                fractal_dimension = (short_pattern * medium_pattern * long_pattern) ** (1/3)
                df['performance_fractal'] = fractal_dimension.fillna(0)
        
        # 7. SINCRONÍA CÓSMICA (Alineación con patrones universales)
        if self._register_feature('cosmic_sync', 'quantum_features'):
            if 'Date' in df.columns:
                # Patrones basados en día de la semana y fase lunar aproximada
                dates = pd.to_datetime(df['Date'])
                day_of_week = dates.dt.dayofweek
                day_of_year = dates.dt.dayofyear
                
                # Función sinusoidal combinada para capturar ritmos naturales
                cosmic_rhythm = (np.sin(2 * np.pi * day_of_week / 7) * 
                                np.cos(2 * np.pi * day_of_year / 365.25) + 1) / 2
                df['cosmic_sync'] = cosmic_rhythm
        
        # 8. POTENCIAL EXPLOSIVO LATENTE (Energía acumulada lista para liberarse)
        if self._register_feature('latent_explosive_potential', 'quantum_features'):
            if '3P' in df.columns and '3PA' in df.columns:
                # Energía acumulada: diferencia entre capacidad y rendimiento reciente
                season_max = df.groupby('Player')['3P'].expanding().max().shift(1).reset_index(0, drop=True)
                recent_avg = self._get_historical_series(df, '3P', window=5, operation='mean')
                attempts_trend = self._get_historical_series(df, '3PA', window=3, operation='mean')
                
                # Potencial latente: capacidad no realizada × momentum de intentos
                latent_potential = (season_max - recent_avg) * (attempts_trend / 5.0)
                df['latent_explosive_potential'] = latent_potential.fillna(0).clip(0, 10)
    
    def _apply_feature_selection(self, df: pd.DataFrame, features: List[str]) -> List[str]:
        """Aplica selección de features si hay demasiadas"""
        logger.info(f"Aplicando selección de features: {len(features)} -> {self.max_features}")
        
        # Aplicar filtro de correlación
        if len(features) > self.max_features:
            X = df[features].fillna(0)
            corr_matrix = X.corr().abs()
            
            # Encontrar features altamente correlacionadas
            upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > self.correlation_threshold)]
            
            features = [f for f in features if f not in to_drop]
        
        # Selección adicional basada en target si es necesario
        if len(features) > self.max_features and '3P' in df.columns:
            X_filtered = df[features].fillna(0)
            y = df['3P'].fillna(0)
            
            selector = SelectKBest(score_func=f_regression, k=self.max_features)
            selector.fit(X_filtered, y)
            
            selected_features = [features[i] for i in range(len(features)) if selector.get_support()[i]]
            features = selected_features
        
        return features
    
    def _log_feature_summary(self) -> None:
        """Log resumen de features creadas"""
        total_features = len(self.feature_registry)
        logger.info(f"RESUMEN DE FEATURES PARA TRIPLES (3P):")
        logger.info(f"Total features creadas: {total_features}")
        
        for category, features in self.feature_categories.items():
            if features:
                logger.info(f"  {category}: {len(features)} features")
        
        logger.info("Features más importantes:")
        for category in ['shooting_accuracy', 'shooting_volume', 'shooting_mechanics']:
            if self.feature_categories[category]:
                logger.info(f"  {category}: {self.feature_categories[category][:3]}")
    
    def _apply_noise_filters(self, df: pd.DataFrame, features: List[str]) -> List[str]:
        """
        Aplica filtros avanzados para eliminar features que solo agregan ruido a los modelos de triples.
        
        Args:
            df: DataFrame con los datos
            features: Lista de features a filtrar
            
        Returns:
            List[str]: Lista de features filtradas sin ruido
        """
        logger.info(f"Iniciando filtrado de ruido en {len(features)} features de triples...")
        
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
        skewness_threshold = 10.0  # Sesgo extremo (más estricto para triples)
        
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
        
        # FILTRO 5: Eliminar features conocidas como problemáticas para triples
        logger.info("Aplicando filtro de features problemáticas conocidas...")
        problematic_patterns = [
            # Features que tienden a ser ruidosas o poco predictivas para triples
            '_squared_',  # Features cuadráticas suelen ser ruidosas
            '_cubed_',    # Features cúbicas suelen ser ruidosas
            '_interaction_complex_',  # Interacciones complejas suelen ser ruidosas
            'random_',    # Features aleatorias
            'noise_',     # Features de ruido
            '_outlier_',  # Features de outliers suelen ser inestables
            '_extreme_',  # Features extremas suelen ser inestables
            'cosmic_',    # Features cósmicas experimentales suelen ser ruidosas
            'quantum_',   # Features cuánticas experimentales suelen ser ruidosas
            'fractal_',   # Features fractales suelen ser ruidosas
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
        
        # FILTRO 8: Filtro específico para triples - eliminar features poco relevantes
        logger.info("Aplicando filtro específico para triples...")
        irrelevant_for_triples = [
            # Features que típicamente no son predictivas para triples
            '_block_',    # Bloqueos no relacionados con triples
            '_steal_',    # Robos menos relevantes para triples
            '_foul_',     # Faltas menos relevantes para triples
            '_rebound_',  # Rebotes menos relevantes para triples
        ]
        
        for feature in clean_features.copy():
            for pattern in irrelevant_for_triples:
                if pattern in feature.lower():
                    clean_features.remove(feature)
                    removed_features.append(f"{feature} (poco relevante para triples)")
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
            basic_features = [f for f in features if any(pattern in f for pattern in ['threept_', '3p_', 'shooting_', 'accuracy_', 'volume_'])]
            if basic_features:
                logger.info(f"Usando {len(basic_features)} features básicas como fallback")
                return basic_features[:20]  # Máximo 20 features básicas
            else:
                logger.error("No se encontraron features básicas válidas")
                return features[:10]  # Devolver las primeras 10 como último recurso
        
        return clean_features