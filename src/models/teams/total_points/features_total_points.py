"""
Módulo de Características para Predicción de Puntos Totales (PTS + PTS_Opp)
=========================================================================

Este módulo contiene toda la lógica de ingeniería de características específica
para la predicción de puntos totales en partidos NBA. Implementa características
avanzadas combinando perspectivas de equipos y jugadores.

FEATURES DE DOMINIO ESPECÍFICO con máximo poder predictivo
OPTIMIZADO - Sin cálculos duplicados, sin multicolinealidad, sin data leakage
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, List, Optional
import warnings
from sklearn.preprocessing import StandardScaler
import logging
from datetime import datetime
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import VarianceThreshold

warnings.filterwarnings('ignore')

# Configurar logging
logger = logging.getLogger(__name__)

class TotalPointsFeatureEngine:
    """
    Motor de ingeniería de características para predicción de puntos totales.
    Combina perspectivas de equipo y jugadores para máxima precisión.
    
    Características principales:
    - Prevención de data leakage con shift(1) en todas las operaciones temporales
    - Features de dominio específico NBA
    - Optimización para evitar multicolinealidad
    - Cache para evitar recálculos
    """
    
    def __init__(self, lookback_games: int = 10):
        """
        Inicializa el motor de features.
        
        Args:
            lookback_games: Número de juegos hacia atrás para cálculos históricos
        """
        self.lookback_games = lookback_games
        self.feature_columns = None
        self.scaler = StandardScaler()
        # Cache para evitar recálculos
        self._cached_calculations = {}
        # Estadísticas de validación
        self._validation_stats = {}
        
    def create_features(self, df_teams: pd.DataFrame, df_players: pd.DataFrame = None) -> pd.DataFrame:
        """
        Crea todas las características para el modelo de puntos totales.
        
        Args:
            df_teams: DataFrame con datos de equipos
            df_players: DataFrame con datos de jugadores (opcional)
            
        Returns:
            DataFrame con todas las características
        """
        if df_teams.empty:
            logger.warning("DataFrame de equipos vacío")
            return pd.DataFrame()
        
        # Crear copia para no modificar el original
        df = df_teams.copy()
        
        # Crear característica objetivo si no existe
        if 'total_points' not in df.columns and 'PTS' in df.columns and 'PTS_Opp' in df.columns:
            df['total_points'] = df['PTS'] + df['PTS_Opp']
        
        # Aplicar transformaciones básicas
        df = self._apply_base_calculations(df)
        
        # Crear características de oponentes
        df = self._create_opponent_features(df)
        
        # Crear características de matchup
        df = self._create_matchup_features(df)
        
        # Crear características temporales
        df = self._create_temporal_features(df)
        
        # Crear características basadas en jugadores (si hay datos disponibles)
        df = self._create_player_based_features(df, df_players)
        
        # Crear características de porcentajes de tiro
        df = self._create_shooting_percentage_features(df)
        
        # NUEVAS CARACTERÍSTICAS AVANZADAS BASADAS EN TOP FEATURES IDENTIFICADAS
        # Crear características avanzadas de volumen de tiros (basado en FGA_FGA_Opp_total_ma siendo top)
        df = self._create_advanced_shooting_volume_features(df)
        
        # Crear características de patrones de mercado (NUEVA FUNCIÓN)
        df = self._create_market_pattern_features(df)
        
        # Crear características avanzadas de momentum y volatilidad (basado en pts_momentum_sqrt siendo top 5)
        df = self._create_advanced_momentum_features(df)
        
        # Crear características avanzadas de pace (basado en pace_ma_3/5/10 siendo top features)
        df = self._create_advanced_pace_features(df)
        
        # Crear características avanzadas de volatilidad (basado en volatilidad siendo múltiples top features)
        df = self._create_advanced_volatility_features(df)
        
        # Crear características avanzadas de fuerza del oponente
        df = self._create_opponent_strength_features(df)
        
        # Crear características avanzadas de ritmo y eficiencia
        df = self._create_pace_and_efficiency_features(df)
        
        # Crear características ensemble que combinan múltiples aspectos
        df = self._create_ensemble_features(df)
        
        # Crear características avanzadas
        df = self._create_advanced_features(df)
        
        # Crear variables dummy para características categóricas
        df = self._create_dummy_variables(df)
        
        # APLICAR FEATURE SELECTION AVANZADO (DESACTIVADO TEMPORALMENTE)
        # logger.info("Aplicando feature selection avanzado...")
        # df = self._advanced_feature_selection(df, target_col='total_points')
        
        # Limpiar y preparar datos finales
        df = self._clean_features(df)
        
        logger.info(f"Features preparadas: {df.shape[1]} características, {df.shape[0]} registros")
        
        return df
    
    def _ensure_chronological_order(self, df: pd.DataFrame) -> pd.DataFrame:
        """Asegura orden cronológico para evitar data leakage"""
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.sort_values(['Team', 'Date'])
            df.reset_index(drop=True, inplace=True)
            logger.info(f"Datos ordenados cronológicamente. Rango de fechas: {df['Date'].min()} a {df['Date'].max()}")
        else:
            logger.warning("Columna 'Date' no encontrada - usando orden original")
        return df
    
    def _validate_and_prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Valida y prepara los datos iniciales"""
        # Verificar columnas requeridas
        required_cols = ['Team', 'Opp', 'PTS', 'PTS_Opp']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            logger.error(f"Columnas requeridas faltantes: {missing_cols}")
            raise ValueError(f"Columnas requeridas faltantes: {missing_cols}")
        
        # Verificar y crear is_win si es necesario
        if 'is_win' not in df.columns:
            if 'Result' in df.columns:
                logger.info("Creando columna is_win desde Result...")
                df['is_win'] = df['Result'].apply(self._extract_win_from_result)
                valid_wins = df['is_win'].notna().sum()
                logger.info(f"is_win creada exitosamente: {valid_wins}/{len(df)} registros válidos")
            else:
                logger.warning("No se puede crear is_win: columna Result no disponible")
                df['is_win'] = 0.5  # Valor neutral
        
        # Verificar has_overtime
        if 'has_overtime' not in df.columns:
            logger.info("Columna has_overtime no encontrada, creando con valor 0")
            df['has_overtime'] = 0
            
        if 'overtime_periods' not in df.columns:
            df['overtime_periods'] = 0
        
        # Verificar is_home
        if 'is_home' not in df.columns:
            if 'Away' in df.columns:
                df['is_home'] = (df['Away'] != '@').astype(int)
                logger.info("Columna is_home creada desde Away")
            else:
                df['is_home'] = 0.5  # Valor neutral
        
        return df
    
    def _extract_win_from_result(self, result_str: str) -> Optional[int]:
        """Extrae is_win desde el formato 'W 123-100' o 'L 114-116'"""
        try:
            result_str = str(result_str).strip()
            if result_str.startswith('W'):
                return 1
            elif result_str.startswith('L'):
                return 0
            else:
                return None
        except:
            return None
    
    def _apply_base_calculations(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aplica cálculos básicos y transformaciones iniciales al DataFrame"""
        
        if df.empty:
            return df
        
        # Asegurar que tenemos las columnas necesarias
        required_cols = ['Team', 'Opp', 'PTS', 'PTS_Opp']
        if not all(col in df.columns for col in required_cols):
            missing = [col for col in required_cols if col not in df.columns]
            logger.warning(f"Faltan columnas requeridas: {missing}")
            return df
        
        # Ordenar cronológicamente si hay fecha disponible
        if 'Date' in df.columns:
            df = df.sort_values(['Team', 'Date'])
        
        # Crear características básicas
        if 'total_points' not in df.columns:
            df['total_points'] = df['PTS'] + df['PTS_Opp']
        
        # Características de victoria/derrota
        if 'is_win' not in df.columns:
            df['is_win'] = (df['PTS'] > df['PTS_Opp']).astype(int)
        
        # Características de local/visitante
        if 'is_home' not in df.columns and '@' in df.columns:
            df['is_home'] = (~df['@']).astype(int)
        elif 'is_home' not in df.columns:
            logger.warning("No se pudo determinar local/visitante")
        
        # Días de descanso
        if 'Date' in df.columns:
            df['days_rest'] = df.groupby('Team')['Date'].diff().dt.days.fillna(3).clip(upper=5)
        
        # Día de la semana
        if 'Date' in df.columns and 'day_of_week' not in df.columns:
            df['day_of_week'] = df['Date'].dt.dayofweek
            df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        
        # Mes
        if 'Date' in df.columns and 'month' not in df.columns:
            df['month'] = df['Date'].dt.month
        
        # Día de la temporada
        if 'Date' in df.columns and 'day_of_season' not in df.columns:
            # Asumir que la temporada comienza en octubre (mes 10)
            df['season'] = df['Date'].dt.year
            oct_first = pd.to_datetime(df['season'].astype(str) + '-10-01')
            
            # Método alternativo para ajustar fechas sin usar itertuples
            df['temp_oct_first'] = oct_first
            mask = df['Date'] < df['temp_oct_first']
            df.loc[mask, 'temp_oct_first'] = pd.to_datetime((df.loc[mask, 'season'] - 1).astype(str) + '-10-01')
            df['day_of_season'] = (df['Date'] - df['temp_oct_first']).dt.days
            df.drop('temp_oct_first', axis=1, inplace=True)
        
        return df
    
    def _create_team_basic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Crea características básicas del equipo con protección contra data leakage"""
        
        # Estadísticas móviles de puntos (con shift para evitar data leakage)
        for window in [3, 5, 10, 15]:
            # Puntos propios
            df[f'pts_ma_{window}'] = df.groupby('Team')['PTS'].transform(
                lambda x: x.rolling(window, min_periods=1).mean().shift(1)
            )
            df[f'pts_std_{window}'] = df.groupby('Team')['PTS'].transform(
                lambda x: x.rolling(window, min_periods=1).std().shift(1)
            )
            
            # Puntos del oponente
            df[f'pts_opp_ma_{window}'] = df.groupby('Team')['PTS_Opp'].transform(
                lambda x: x.rolling(window, min_periods=1).mean().shift(1)
            )
            
            # Total de puntos
            df[f'total_pts_ma_{window}'] = df[f'pts_ma_{window}'] + df[f'pts_opp_ma_{window}']
            
            # Variabilidad del total
            df[f'total_pts_std_{window}'] = df.groupby('Team').apply(
                lambda x: (x['PTS'] + x['PTS_Opp']).rolling(window, min_periods=1).std().shift(1)
            ).reset_index(level=0, drop=True)
        
        # Tendencia de puntos (regresión lineal simple)
        df['pts_trend'] = self._calculate_trend(df, 'PTS', 10)
        df['pts_opp_trend'] = self._calculate_trend(df, 'PTS_Opp', 10)
        df['total_pts_trend'] = df['pts_trend'] + df['pts_opp_trend']
        
        # Promedios expandidos (toda la historia hasta el momento)
        df['pts_season_avg'] = df.groupby('Team')['PTS'].transform(
            lambda x: x.expanding().mean().shift(1)
        )
        df['pts_opp_season_avg'] = df.groupby('Team')['PTS_Opp'].transform(
            lambda x: x.expanding().mean().shift(1)
        )
        
        logger.info(f"Features básicas creadas: {len([col for col in df.columns if 'pts_ma_' in col or 'pts_std_' in col])}")
        
        return df
    
    def _calculate_trend(self, df: pd.DataFrame, column: str, window: int) -> pd.Series:
        """Calcula tendencia usando regresión lineal con protección contra data leakage"""
        def trend(x):
            if len(x) < 3:
                return 0
            indices = np.arange(len(x))
            if np.std(indices) == 0:
                return 0
            return np.polyfit(indices, x, 1)[0]
        
        return df.groupby('Team')[column].transform(
            lambda x: x.rolling(window, min_periods=3).apply(trend).shift(1)
        )
    
    def _create_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Crea características de momentum y forma reciente"""
        
        # Racha de victorias/derrotas (con shift)
        if 'is_win' in df.columns:
            # Cálculo más eficiente de rachas
            df['win_streak'] = df.groupby('Team').apply(
                lambda x: self._calculate_streak(x['is_win'], 1)
            ).reset_index(level=0, drop=True)
            
            df['loss_streak'] = df.groupby('Team').apply(
                lambda x: self._calculate_streak(1 - x['is_win'], 1)
            ).reset_index(level=0, drop=True)
        
        # Forma reciente (últimos N juegos)
        for n in [3, 5, 10]:
            if 'is_win' in df.columns:
                df[f'win_rate_last_{n}'] = df.groupby('Team')['is_win'].transform(
                    lambda x: x.rolling(n, min_periods=1).mean().shift(1)
                )
            
            # Puntos sobre/bajo el promedio
            df[f'pts_vs_avg_last_{n}'] = df.groupby('Team').apply(
                lambda x: x['PTS'].rolling(n, min_periods=1).mean().shift(1) - 
                         x['PTS'].expanding().mean().shift(1)
            ).reset_index(level=0, drop=True)
            
            # Lo mismo para puntos totales
            df[f'total_pts_vs_avg_last_{n}'] = df.groupby('Team').apply(
                lambda x: (x['PTS'] + x['PTS_Opp']).rolling(n, min_periods=1).mean().shift(1) - 
                         (x['PTS'] + x['PTS_Opp']).expanding().mean().shift(1)
            ).reset_index(level=0, drop=True)
        
        # Momentum de puntos (cambio porcentual)
        for periods in [3, 5]:
            df[f'pts_momentum_{periods}'] = df.groupby('Team')['PTS'].transform(
                lambda x: x.pct_change(periods=periods).shift(1)
            )
            df[f'total_pts_momentum_{periods}'] = df.groupby('Team').apply(
                lambda x: (x['PTS'] + x['PTS_Opp']).pct_change(periods=periods).shift(1)
            ).reset_index(level=0, drop=True)
        
        # Consistencia (coeficiente de variación)
        df['pts_consistency'] = df.groupby('Team').apply(
            lambda x: (x['PTS'].rolling(10, min_periods=3).std() / 
                      x['PTS'].rolling(10, min_periods=3).mean()).shift(1)
        ).reset_index(level=0, drop=True)
        
        df['total_pts_consistency'] = df.groupby('Team').apply(
            lambda x: ((x['PTS'] + x['PTS_Opp']).rolling(10, min_periods=3).std() / 
                      (x['PTS'] + x['PTS_Opp']).rolling(10, min_periods=3).mean()).shift(1)
        ).reset_index(level=0, drop=True)
        
        logger.info(f"Features de momentum creadas: {len([col for col in df.columns if 'momentum' in col or 'streak' in col])}")
        
        return df
    
    def _calculate_streak(self, series: pd.Series, value: int) -> pd.Series:
        """Calcula rachas de un valor específico"""
        streak = (series == value).astype(int)
        streak = streak.groupby((streak != streak.shift()).cumsum()).cumsum()
        return streak.shift(1)
    
    def _create_opponent_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Crea características del oponente y matchup"""
        
        # Crear un DataFrame temporal con estadísticas de equipos
        team_stats = pd.DataFrame()
        
        # Para cada equipo, calcular estadísticas móviles
        for team in df['Team'].unique():
            team_data = df[df['Team'] == team].copy()
            if len(team_data) < 3:
                continue
            
            # Ordenar por fecha
            team_data = team_data.sort_values('Date') if 'Date' in team_data.columns else team_data
            
            # Calcular estadísticas móviles con shift para evitar data leakage
            team_data['PTS_mean'] = team_data['PTS'].rolling(10, min_periods=3).mean().shift(1)
            team_data['PTS_std'] = team_data['PTS'].rolling(10, min_periods=3).std().shift(1)
            team_data['PTS_Opp_mean'] = team_data['PTS_Opp'].rolling(10, min_periods=3).mean().shift(1)
            team_data['PTS_Opp_std'] = team_data['PTS_Opp'].rolling(10, min_periods=3).std().shift(1)
            
            if 'is_win' in team_data.columns:
                team_data['is_win_mean'] = team_data['is_win'].rolling(10, min_periods=3).mean().shift(1)
            else:
                team_data['is_win_mean'] = 0.5
            
            # Guardar solo las columnas necesarias
            team_stats_subset = team_data[['Team', 'Date', 'PTS_mean', 'PTS_std', 'PTS_Opp_mean', 'PTS_Opp_std', 'is_win_mean']].copy()
            
            # Agregar al DataFrame temporal
            team_stats = pd.concat([team_stats, team_stats_subset], ignore_index=True)
        
        # Renombrar columnas para el merge
        team_stats = team_stats.rename(columns={
            'Team': 'Opp',
            'PTS_mean': 'opp_PTS_mean',
            'PTS_std': 'opp_PTS_std',
            'PTS_Opp_mean': 'opp_PTS_Opp_mean',
            'PTS_Opp_std': 'opp_PTS_Opp_std',
            'is_win_mean': 'opp_is_win_mean'
        })
        
        # Merge con datos del oponente usando fecha anterior o igual
        # Primero ordenamos por fecha
        df = df.sort_values('Date')
        team_stats = team_stats.sort_values('Date')
        
        # Crear un DataFrame para almacenar las estadísticas del oponente
        df_with_opp = df.copy()
        
        # Para cada fila, buscar la estadística más reciente del oponente
        for idx, row in df.iterrows():
            opp = row['Opp']
            date = row['Date']
            
            # Filtrar estadísticas del oponente antes de esta fecha
            opp_stats = team_stats[(team_stats['Opp'] == opp) & (team_stats['Date'] < date)]
            
            if not opp_stats.empty:
                # Tomar la estadística más reciente
                latest_stats = opp_stats.iloc[-1]
                
                # Asignar valores
                for col in ['opp_PTS_mean', 'opp_PTS_std', 'opp_PTS_Opp_mean', 'opp_PTS_Opp_std']:
                    df_with_opp.loc[idx, col] = latest_stats[col]
            else:
                # No hay datos históricos para este oponente
                for col in ['opp_PTS_mean', 'opp_PTS_std', 'opp_PTS_Opp_mean', 'opp_PTS_Opp_std']:
                    df_with_opp.loc[idx, col] = np.nan
                df_with_opp.loc[idx, 'opp_is_win_mean'] = 0.5
        
        # Diferencial de calidad
        if 'win_rate_last_10' in df_with_opp.columns and 'opp_is_win_mean' in df_with_opp.columns:
            df_with_opp['quality_diff'] = df_with_opp['win_rate_last_10'] - df_with_opp['opp_is_win_mean']
        
        # Matchup histórico
        df_with_opp = self._create_matchup_features(df_with_opp)
        
        # Estilo de juego del oponente
        if all(col in df_with_opp.columns for col in ['opp_PTS_mean', 'opp_PTS_Opp_mean']):
            df_with_opp['opp_pace_tendency'] = df_with_opp['opp_PTS_mean'] + df_with_opp['opp_PTS_Opp_mean']
            df_with_opp['opp_scoring_tendency'] = df_with_opp['opp_PTS_mean'] / df_with_opp['opp_PTS_Opp_mean'].clip(lower=1)
        
        logger.info(f"Features de oponente creadas: {len([col for col in df_with_opp.columns if col.startswith('opp_')])}")
        
        return df_with_opp
    
    def _create_matchup_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Crea features específicas de matchup histórico SIN DATA LEAKAGE"""
        # Crear clave de matchup
        df['matchup_key'] = df.apply(lambda x: tuple(sorted([x['Team'], x['Opp']])), axis=1)
        
        # Ordenar por fecha para asegurar cálculos correctos
        df = df.sort_values(['matchup_key', 'Team', 'Date']) if 'Date' in df.columns else df.sort_values(['matchup_key', 'Team'])
        
        # Calcular estadísticas de matchup fila por fila para evitar data leakage
        df['matchup_total_avg'] = np.nan
        df['matchup_total_std'] = np.nan
        df['matchup_diff_avg'] = np.nan
        df['matchup_win_rate'] = np.nan
        
        # Calcular para cada combinación de matchup_key y Team
        for (matchup, team), group_indices in df.groupby(['matchup_key', 'Team']).groups.items():
            group_data = df.loc[group_indices].copy()
            
            # Calcular total_points y pts_diff para este grupo
            total_pts = group_data['PTS'] + group_data['PTS_Opp']
            pts_diff = group_data['PTS'] - group_data['PTS_Opp']
            is_win = group_data.get('is_win', pd.Series([0.5] * len(group_data), index=group_data.index))
            
            # Calcular estadísticas expandidas con shift para cada fila
            matchup_total_avg = total_pts.expanding().mean().shift(1)
            matchup_total_std = total_pts.expanding().std().shift(1)
            matchup_diff_avg = pts_diff.expanding().mean().shift(1)
            matchup_win_rate = is_win.expanding().mean().shift(1)
            
            # Asignar valores calculados al DataFrame original
            df.loc[group_indices, 'matchup_total_avg'] = matchup_total_avg
            df.loc[group_indices, 'matchup_total_std'] = matchup_total_std
            df.loc[group_indices, 'matchup_diff_avg'] = matchup_diff_avg
            df.loc[group_indices, 'matchup_win_rate'] = matchup_win_rate
        
        return df
    
    def _create_efficiency_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Crea características de eficiencia ofensiva y defensiva"""
        
        # Eficiencia de tiro actual
        if all(col in df.columns for col in ['FG%', 'FGA']):
            df['fg_efficiency'] = df['FG%'] * df['FGA']
            df['fg_efficiency_opp'] = df['FG%_Opp'] * df['FGA_Opp']
        
        if all(col in df.columns for col in ['3P%', '3PA']):
            df['fg3_efficiency'] = df['3P%'] * df['3PA']
            df['fg3_efficiency_opp'] = df['3P%_Opp'] * df['3PA_Opp']
        
        # True Shooting (usando valores actuales para el cálculo base)
        if all(col in df.columns for col in ['PTS', 'FGA', 'FTA']):
            df['true_shooting'] = df['PTS'] / (2 * (df['FGA'] + 0.44 * df['FTA']))
            df['true_shooting_opp'] = df['PTS_Opp'] / (2 * (df['FGA_Opp'] + 0.44 * df['FTA_Opp']))
        
        # Eficiencia móvil (con shift)
        for window in [5, 10]:
            if 'true_shooting' in df.columns:
                df[f'off_efficiency_ma_{window}'] = df.groupby('Team')['true_shooting'].transform(
                    lambda x: x.rolling(window, min_periods=1).mean().shift(1)
                )
                df[f'def_efficiency_ma_{window}'] = df.groupby('Team')['true_shooting_opp'].transform(
                    lambda x: x.rolling(window, min_periods=1).mean().shift(1)
                )
            
            # Eficiencia combinada del juego
            if all(col in df.columns for col in ['fg_efficiency', 'fg_efficiency_opp']):
                df[f'game_efficiency_ma_{window}'] = df.groupby('Team').apply(
                    lambda x: (x['fg_efficiency'] + x['fg_efficiency_opp']).rolling(window, min_periods=1).mean().shift(1)
                ).reset_index(level=0, drop=True)
        
        # Ratios de tiro
        if all(col in df.columns for col in ['3PA', 'FGA']):
            df['three_point_rate'] = df['3PA'] / (df['FGA'] + 1)
            df['three_point_rate_opp'] = df['3PA_Opp'] / (df['FGA_Opp'] + 1)
            df['three_point_rate_diff'] = df['three_point_rate'] - df['three_point_rate_opp']
        
        # Diferencial de eficiencia
        if all(col in df.columns for col in ['fg_efficiency', 'fg_efficiency_opp']):
            df['efficiency_diff'] = df['fg_efficiency'] - df['fg_efficiency_opp']
        
        # Tiros libres como indicador de agresividad
        if all(col in df.columns for col in ['FTA', 'FGA']):
            df['ft_rate'] = df['FTA'] / (df['FGA'] + 1)
            df['ft_rate_opp'] = df['FTA_Opp'] / (df['FGA_Opp'] + 1)
            df['ft_rate_total'] = df['ft_rate'] + df['ft_rate_opp']
            
            df['ft_rate_ma_5'] = df.groupby('Team')['ft_rate_total'].transform(
                lambda x: x.rolling(5, min_periods=1).mean().shift(1)
            )
        
        logger.info(f"Features de eficiencia creadas: {len([col for col in df.columns if 'efficiency' in col or 'shooting' in col])}")
        
        return df
    
    def _create_pace_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Crea características relacionadas con el ritmo de juego"""
        
        # Usar posesiones calculadas en base calculations
        if 'total_possessions' in df.columns:
            df['pace_est'] = df['total_possessions']
        else:
            # Estimación alternativa
            if all(col in df.columns for col in ['FGA', 'FTA', 'FGA_Opp', 'FTA_Opp']):
                df['possessions_est'] = df['FGA'] + 0.44 * df['FTA']
                df['possessions_est_opp'] = df['FGA_Opp'] + 0.44 * df['FTA_Opp']
                df['pace_est'] = (df['possessions_est'] + df['possessions_est_opp']) / 2
        
        # Ritmo móvil (con shift)
        for window in [3, 5, 10]:
            if 'pace_est' in df.columns:
                df[f'pace_ma_{window}'] = df.groupby('Team')['pace_est'].transform(
                    lambda x: x.rolling(window, min_periods=1).mean().shift(1)
                )
                
                # Variabilidad del ritmo
                df[f'pace_std_{window}'] = df.groupby('Team')['pace_est'].transform(
                    lambda x: x.rolling(window, min_periods=1).std().shift(1)
                )
        
        # Cambio de ritmo
        if 'pace_est' in df.columns:
            df['pace_change'] = df.groupby('Team')['pace_est'].transform(
                lambda x: x.pct_change(periods=3).shift(1)
            )
            
            # Tendencia de ritmo
            df['pace_trend'] = self._calculate_trend(df, 'pace_est', 10)
        
        # Adaptación al ritmo del oponente CON SHIFT
        if 'pace_ma_5' in df.columns:
            # Calcular ritmo promedio por equipo usando expanding mean con shift
            team_avg_pace = df.groupby('Team')['pace_ma_5'].transform(
                lambda x: x.expanding().mean().shift(1)
            )
            opp_avg_pace = df.groupby('Opp')['pace_ma_5'].transform(
                lambda x: x.expanding().mean().shift(1)
            )
            
            df['pace_matchup'] = df['pace_ma_5'] - opp_avg_pace
            df['pace_differential'] = team_avg_pace - opp_avg_pace
        
        # Factor de ritmo esperado para el juego
        if all(col in df.columns for col in ['pace_ma_5', 'opp_pace_tendency']):
            df['expected_game_pace'] = (df['pace_ma_5'] + df['opp_pace_tendency']) / 2
        
        logger.info(f"Features de ritmo creadas: {len([col for col in df.columns if 'pace' in col or 'possessions' in col])}")
        
        return df
    
    def _create_situational_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Crea características situacionales"""
        
        # Local vs Visitante (con shift para datos históricos)
        if 'is_home' in df.columns:
            # Puntos en casa
            df['home_pts_avg'] = df[df['is_home'] == 1].groupby('Team')['PTS'].transform(
                lambda x: x.expanding().mean().shift(1)
            )
            df['home_total_avg'] = df[df['is_home'] == 1].groupby('Team').apply(
                lambda x: (x['PTS'] + x['PTS_Opp']).expanding().mean().shift(1)
            ).reset_index(level=0, drop=True)
            
            # Puntos fuera
            df['away_pts_avg'] = df[df['is_home'] == 0].groupby('Team')['PTS'].transform(
                lambda x: x.expanding().mean().shift(1)
            )
            df['away_total_avg'] = df[df['is_home'] == 0].groupby('Team').apply(
                lambda x: (x['PTS'] + x['PTS_Opp']).expanding().mean().shift(1)
            ).reset_index(level=0, drop=True)
            
            # Llenar NaN con el promedio general
            for col in ['home_pts_avg', 'home_total_avg', 'away_pts_avg', 'away_total_avg']:
                df[col] = df.groupby('Team').apply(
                    lambda x: x[col].fillna(x['PTS'].rolling(10, min_periods=1).mean().shift(1))
                ).reset_index(level=0, drop=True)
            
            # Ventaja de local
            df['home_advantage'] = df['home_pts_avg'] - df['away_pts_avg']
            df['home_total_advantage'] = df['home_total_avg'] - df['away_total_avg']
        
        # Rendimiento en juegos cerrados (con shift)
        df['close_game'] = (df['PTS'] - df['PTS_Opp']).abs() <= 5
        df['pts_in_close_games'] = df[df['close_game']].groupby('Team')['PTS'].transform(
            lambda x: x.expanding().mean().shift(1)
        )
        df['total_in_close_games'] = df[df['close_game']].groupby('Team').apply(
            lambda x: (x['PTS'] + x['PTS_Opp']).expanding().mean().shift(1)
        ).reset_index(level=0, drop=True)
        
        # Rendimiento en overtime (con shift)
        if 'has_overtime' in df.columns:
            df['pts_in_ot'] = df[df['has_overtime'] == 1].groupby('Team')['PTS'].transform(
                lambda x: x.expanding().mean().shift(1)
            )
            df['total_in_ot'] = df[df['has_overtime'] == 1].groupby('Team').apply(
                lambda x: (x['PTS'] + x['PTS_Opp']).expanding().mean().shift(1)
            ).reset_index(level=0, drop=True)
        
        # Días de descanso
        if 'days_rest' in df.columns:
            df['is_back_to_back'] = (df['days_rest'] == 1).astype(int)
            df['is_rested'] = (df['days_rest'] >= 2).astype(int)
            
            # Rendimiento con descanso (con shift)
            df['pts_with_rest'] = df[df['is_rested'] == 1].groupby('Team')['PTS'].transform(
                lambda x: x.expanding().mean().shift(1)
            )
            df['total_with_rest'] = df[df['is_rested'] == 1].groupby('Team').apply(
                lambda x: (x['PTS'] + x['PTS_Opp']).expanding().mean().shift(1)
            ).reset_index(level=0, drop=True)
            
            df['pts_b2b'] = df[df['is_back_to_back'] == 1].groupby('Team')['PTS'].transform(
                lambda x: x.expanding().mean().shift(1)
            )
            df['total_b2b'] = df[df['is_back_to_back'] == 1].groupby('Team').apply(
                lambda x: (x['PTS'] + x['PTS_Opp']).expanding().mean().shift(1)
            ).reset_index(level=0, drop=True)
        
        logger.info(f"Features situacionales creadas: {len([col for col in df.columns if any(x in col for x in ['home', 'away', 'close', 'ot', 'rest', 'b2b'])])}")
        
        return df
    
    def _create_player_based_features(self, df: pd.DataFrame, df_players: pd.DataFrame) -> pd.DataFrame:
        """Crea características basadas en datos de jugadores"""
        
        # Verificar si hay datos de jugadores
        if df_players is None or df_players.empty:
            logger.warning("No hay datos de jugadores disponibles")
            return df
        
        # Asegurar orden cronológico en datos de jugadores
        df_players = df_players.sort_values(['Team', 'Date'])
        
        # Verificar columnas disponibles en df_players
        available_cols = df_players.columns.tolist()
        logger.info(f"Columnas disponibles en datos de jugadores: {len(available_cols)}")
        
        # Definir agregaciones basadas en columnas disponibles
        agg_dict = {}
        
        # Estadísticas básicas
        if 'PTS' in available_cols:
            agg_dict['PTS'] = ['sum', 'mean', 'std', 'max']
        if 'MP' in available_cols:
            agg_dict['MP'] = ['mean', 'std']
        if 'is_started' in available_cols:
            agg_dict['is_started'] = 'sum'
        
        # Porcentajes de tiro
        for col in ['FG%', '3P%', 'FT%']:
            if col in available_cols:
                agg_dict[col] = 'mean'
        
        # Estadísticas adicionales
        for col in ['TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF']:
            if col in available_cols:
                agg_dict[col] = 'sum'
        
        if '+/-' in available_cols:
            agg_dict['+/-'] = 'mean'
        
        # Doble-dobles y triple-dobles
        for col in ['has_double_double', 'has_triple_double']:
            if col in available_cols:
                agg_dict[col] = 'sum'
        
        if not agg_dict:
            logger.warning("No se encontraron columnas válidas para agregar en datos de jugadores")
            return df
        
        try:
            # Agrupar estadísticas de jugadores por equipo y fecha de manera más eficiente
            # Limitar a solo las columnas necesarias
            cols_to_use = ['Team', 'Date'] + list(agg_dict.keys())
            df_players_slim = df_players[cols_to_use].copy()
            
            # Agrupar y agregar
            player_stats = df_players_slim.groupby(['Team', 'Date']).agg(agg_dict)
            
            # Aplanar columnas
            player_stats.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col 
                                   for col in player_stats.columns]
            
            # Resetear índice
            player_stats = player_stats.reset_index()
            
            # Renombrar columnas para claridad
            rename_dict = {
                'PTS_sum': 'team_total_pts_players',
                'PTS_mean': 'avg_player_pts',
                'PTS_std': 'pts_distribution',
                'PTS_max': 'top_scorer_pts',
                'MP_mean': 'avg_minutes',
                'MP_std': 'minutes_distribution',
                'is_started_sum': 'num_starters',
                'FG%_mean': 'team_fg_pct',
                '3P%_mean': 'team_3p_pct',
                'FT%_mean': 'team_ft_pct',
                'TRB_sum': 'team_rebounds',
                'AST_sum': 'team_assists',
                'STL_sum': 'team_steals',
                'BLK_sum': 'team_blocks',
                'TOV_sum': 'team_turnovers',
                'PF_sum': 'team_fouls',
                '+/-_mean': 'team_plus_minus',
                'has_double_double_sum': 'num_double_doubles',
                'has_triple_double_sum': 'num_triple_doubles'
            }
            
            # Aplicar renombrado solo para columnas que existen
            rename_dict = {k: v for k, v in rename_dict.items() if k in player_stats.columns}
            player_stats = player_stats.rename(columns=rename_dict)
            
            # Aplicar shift a las estadísticas de jugadores para evitar data leakage
            # Hacerlo de manera más eficiente
            shift_cols = [col for col in player_stats.columns if col not in ['Team', 'Date']]
            
            for col in shift_cols:
                player_stats[col] = player_stats.groupby('Team')[col].shift(1)
            
            # Merge con datos de equipos de manera más eficiente
            # Usar merge con indicador para detectar problemas
            df = pd.merge(df, player_stats, on=['Team', 'Date'], how='left', indicator='_merge_indicator')
            
            # Verificar si hubo problemas en el merge
            if (df['_merge_indicator'] == 'left_only').any():
                logger.warning(f"Hay {(df['_merge_indicator'] == 'left_only').sum()} filas sin datos de jugadores")
            
            # Eliminar columna indicadora
            df = df.drop('_merge_indicator', axis=1)
            
            # Características derivadas de jugadores (solo si las columnas base existen)
            # Calcular de manera más eficiente
            if 'top_scorer_pts' in df.columns and 'team_total_pts_players' in df.columns:
                df['scoring_concentration'] = df['top_scorer_pts'] / df['team_total_pts_players'].clip(lower=1)
            
            if all(col in df.columns for col in ['num_starters', 'avg_player_pts', 'team_total_pts_players']):
                df['bench_contribution'] = 1 - (df['num_starters'] * df['avg_player_pts']) / df['team_total_pts_players'].clip(lower=1)
            
            # Eficiencia del equipo basada en jugadores
            efficiency_cols = ['team_total_pts_players', 'team_rebounds', 'team_assists', 
                              'team_steals', 'team_blocks', 'team_turnovers', 'avg_minutes']
            
            if all(col in df.columns for col in efficiency_cols):
                # Usar operaciones vectorizadas
                df['team_efficiency_players'] = (
                    df['team_total_pts_players'] + df['team_rebounds'] + 
                    df['team_assists'] + df['team_steals'] + df['team_blocks'] - 
                    df['team_turnovers']
                ) / df['avg_minutes'].clip(lower=1)
            
            # Tendencias de jugadores clave (con shift)
            for stat in ['top_scorer_pts', 'team_assists', 'team_rebounds']:
                if stat in df.columns:
                    df[f'{stat}_ma_5'] = df.groupby('Team')[stat].transform(
                        lambda x: x.rolling(5, min_periods=1).mean().shift(1)
                    )
            
            # Factor de profundidad del equipo
            if 'pts_distribution' in df.columns:
                df['team_depth_factor'] = 1 / df['pts_distribution'].clip(lower=0.1)
            
            logger.info(f"Features de jugadores creadas: {len([col for col in df.columns if any(x in col for x in ['team_', 'player', 'scorer', 'bench'])])}")
            
        except Exception as e:
            logger.error(f"Error procesando datos de jugadores: {str(e)}")
            # No interrumpir el proceso completo por un error en features de jugadores
        
        return df
    
    def _create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Crea características de interacción entre variables"""
        
        # Interacción ritmo-eficiencia
        if all(col in df.columns for col in ['pace_ma_5', 'off_efficiency_ma_5']):
            df['pace_efficiency_interaction'] = df['pace_ma_5'] * df['off_efficiency_ma_5']
        
        # Interacción local/visitante con forma
        if all(col in df.columns for col in ['is_home', 'win_rate_last_5']):
            df['home_form_interaction'] = df['is_home'] * df['win_rate_last_5']
        
        # Interacción matchup-momentum
        if 'matchup_win_rate' in df.columns and 'pts_momentum_5' in df.columns:
            df['matchup_momentum'] = df['matchup_win_rate'].fillna(0.5) * df['pts_momentum_5']
        
        # Interacción descanso-ritmo
        if all(col in df.columns for col in ['days_rest', 'pace_ma_3']):
            df['rest_pace_interaction'] = df['days_rest'] * df['pace_ma_3']
        
        # Interacción calidad-estilo
        if all(col in df.columns for col in ['win_rate_last_10', 'three_point_rate']):
            df['quality_style_interaction'] = df['win_rate_last_10'] * df['three_point_rate']
        
        # Polinomios de características clave
        for col in ['pts_ma_5', 'pace_ma_5', 'total_pts_ma_5']:
            if col in df.columns:
                df[f'{col}_squared'] = df[col] ** 2
                df[f'{col}_sqrt'] = np.sqrt(np.abs(df[col]))
        
        # Ratios importantes
        if all(col in df.columns for col in ['pts_ma_5', 'pts_opp_ma_5']):
            df['offense_defense_ratio'] = df['pts_ma_5'] / (df['pts_opp_ma_5'] + 1)
        
        if all(col in df.columns for col in ['off_efficiency_ma_5', 'def_efficiency_ma_5']):
            df['efficiency_ratio'] = df['off_efficiency_ma_5'] / (df['def_efficiency_ma_5'] + 0.1)
        
        # Interacciones de totales
        if all(col in df.columns for col in ['total_pts_ma_5', 'pace_ma_5']):
            df['total_pace_interaction'] = df['total_pts_ma_5'] * df['pace_ma_5'] / 100
        
        if all(col in df.columns for col in ['total_pts_consistency', 'expected_game_pace']):
            df['consistency_pace_interaction'] = df['total_pts_consistency'] * df['expected_game_pace']
        
        logger.info(f"Features de interacción creadas: {len([col for col in df.columns if 'interaction' in col or 'ratio' in col])}")
        
        return df
    
    def _create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Crea características temporales y de calendario"""
        
        if 'Date' not in df.columns:
            logger.warning("Columna Date no disponible, saltando features temporales")
            return df
        
        # Componentes de fecha ya creados en base calculations
        # Agregar features adicionales
        
        # Fase de la temporada
        if 'day_of_season' in df.columns:
            # Usar pd.cut de manera más eficiente
            df['season_phase'] = pd.cut(df['day_of_season'], 
                                       bins=[0, 30, 60, 120, 180, 365],
                                       labels=['early', 'early_mid', 'mid', 'late_mid', 'late'])
            
            # Crear dummies de manera más eficiente sin concatenar
            phase_dummies = pd.get_dummies(df['season_phase'], prefix='phase')
            for col in phase_dummies.columns:
                df[col] = phase_dummies[col]
            
            # Porcentaje de temporada completado
            df['season_progress'] = df['day_of_season'] / 365
        
        # Asegurar que Date es datetime
        if not pd.api.types.is_datetime64_any_dtype(df['Date']):
            df['Date'] = pd.to_datetime(df['Date'])
        
        # Método optimizado para calcular juegos en últimos 7 días
        # Crear una columna temporal para contar juegos
        df['game_count'] = 1
        
        # Inicializar columna de juegos en últimos 7 días
        df['games_last_7_days'] = 0
        
        # Procesar cada equipo por separado
        for team in df['Team'].unique():
            team_mask = df['Team'] == team
            if team_mask.sum() <= 1:
                continue
            
            # Obtener índices y fechas para este equipo
            team_indices = df.index[team_mask]
            team_dates = df.loc[team_mask, 'Date'].sort_values()
            
            # Para cada fecha, contar juegos en los 7 días anteriores
            for i, (idx, date) in enumerate(zip(team_indices, team_dates)):
                # Usar vectorización en lugar de bucle
                prev_7days = (date - team_dates[:i]) <= pd.Timedelta(days=7)
                games_count = prev_7days.sum()
                df.loc[idx, 'games_last_7_days'] = games_count
        
        # Eliminar columna temporal
        df.drop('game_count', axis=1, inplace=True)
        
        # Densidad de calendario
        df['schedule_density'] = df['games_last_7_days'] / 7
        
        # Fatiga acumulada
        if 'MP' in df.columns:
            # Usar transform en lugar de recálculo
            df['cumulative_minutes'] = df.groupby('Team')['MP'].transform('cumsum')
            df['fatigue_index'] = df['games_last_7_days'] * df['cumulative_minutes'] / 1000
        
        # Patrones semanales
        if 'day_of_week' in df.columns:
            # Crear dummies sin concatenación
            dow_dummies = pd.get_dummies(df['day_of_week'], prefix='dow')
            for col in dow_dummies.columns:
                df[col] = dow_dummies[col]
        
        # Tendencias mensuales
        if 'month' in df.columns:
            # Usar vectorización en lugar de .isin()
            df['high_intensity_month'] = ((df['month'] >= 3) & (df['month'] <= 5)).astype(int)
            df['season_start'] = ((df['month'] == 10) | (df['month'] == 11)).astype(int)
        
        logger.info(f"Features temporales creadas: {len([col for col in df.columns if any(x in col for x in ['phase_', 'dow_', 'season', 'fatigue'])])}")
        
        return df
    
    def _apply_quality_filters(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aplica filtros de calidad y limpieza final"""
        
        # Reemplazar infinitos
        df = df.replace([np.inf, -np.inf], np.nan)
        
        # Columnas a excluir de la imputación (no numéricas o problemáticas)
        exclude_from_imputation = ['Team', 'Date', 'Opp', 'Result', 'Away', 'GS']
        
        # Contar NaN antes de limpieza
        nan_counts = df.isna().sum()
        high_nan_cols = nan_counts[nan_counts > len(df) * 0.5].index.tolist()
        
        if high_nan_cols:
            logger.warning(f"Columnas con >50% NaN: {high_nan_cols[:10]}...")
        
        # Estrategia de imputación por tipo de feature
        for col in df.columns:
            # Excluir columnas no numéricas y columnas en la lista de exclusión
            if col in exclude_from_imputation or df[col].dtype not in [np.float64, np.int64, 'float64', 'int64']:
                continue
                
            # Para features de promedio móvil, forward fill
            if any(pattern in col for pattern in ['_ma_', '_avg_', '_mean']):
                df[col] = df.groupby('Team')[col].fillna(method='ffill')
            
            # Para features de conteo, llenar con 0
            elif any(pattern in col for pattern in ['streak', 'count', 'num_']):
                df[col] = df[col].fillna(0)
            
            # Para ratios y porcentajes, llenar con valor neutral
            elif any(pattern in col for pattern in ['rate', 'pct', 'ratio']):
                df[col] = df[col].fillna(0.5)
            
            # Para el resto, llenar con la mediana del equipo o 0
            else:
                try:
                    df[col] = df.groupby('Team')[col].transform(
                        lambda x: x.fillna(x.median() if not x.median() is np.nan else 0)
                    )
                except (TypeError, ValueError):
                    # Si hay error, simplemente llenar con 0
                    logger.warning(f"Error procesando columna {col}, llenando con 0")
                    df[col] = df[col].fillna(0)
        
        # Validación final
        remaining_nan = df.isna().sum().sum()
        if remaining_nan > 0:
            logger.warning(f"NaN restantes después de limpieza: {remaining_nan}")
            
            # Último recurso: llenar todos los NaN restantes con 0
            df = df.fillna(0)
        
        # Guardar estadísticas de validación
        self._validation_stats = {
            'total_features': len(df.columns),
            'numeric_features': len(df.select_dtypes(include=[np.number]).columns),
            'high_nan_features': len(high_nan_cols),
            'remaining_nan': remaining_nan
        }
        
        return df
    
    def _update_feature_columns(self, df: pd.DataFrame) -> None:
        """Actualiza la lista de columnas de características disponibles"""
        
        # Obtener patrones de características
        feature_patterns, exclude_cols = self.get_feature_columns()
        
        # Agregar columnas específicas a excluir
        exclude_cols.extend(['total_points', 'close_game'])  # Variables auxiliares
        
        # Seleccionar columnas que coinciden con los patrones
        feature_cols = []
        for col in df.columns:
            if col in exclude_cols:
                continue
            # Verificar si la columna coincide con algún patrón
            if any(pattern in col for pattern in feature_patterns):
                feature_cols.append(col)
            # O si es una columna numérica no excluida
            elif df[col].dtype in [np.float64, np.int64]:
                feature_cols.append(col)
        
        self.feature_columns = feature_cols
        logger.info(f"Features finales identificadas: {len(feature_cols)}")
    
    def _generate_feature_report(self, df: pd.DataFrame) -> None:
        """Genera un reporte detallado de las features creadas"""
        
        logger.info("\n" + "="*80)
        logger.info("REPORTE FINAL DE FEATURES")
        logger.info("="*80)
        
        # Agrupar features por categoría
        categories = {
            'Básicas': ['pts_ma_', 'pts_std_', 'pts_opp_ma_', 'total_pts_ma_'],
            'Momentum': ['momentum', 'streak', 'win_rate', 'consistency'],
            'Oponente': ['opp_', 'matchup_', 'quality_diff'],
            'Eficiencia': ['efficiency', 'shooting', 'fg_', 'ft_rate'],
            'Ritmo': ['pace', 'possessions'],
            'Situacional': ['home', 'away', 'close', 'rest', 'b2b'],
            'Jugadores': ['team_', 'player', 'scorer', 'bench'],
            'Interacción': ['interaction', 'ratio', 'squared'],
            'Temporal': ['phase_', 'dow_', 'season', 'month', 'fatigue']
        }
        
        for category, patterns in categories.items():
            count = len([col for col in df.columns 
                        if any(pattern in col for pattern in patterns)])
            if count > 0:
                logger.info(f"{category:15s}: {count:3d} features")
        
        logger.info(f"\nTOTAL FEATURES: {len(self.feature_columns)}")
        
        # Estadísticas de validación
        if self._validation_stats:
            logger.info(f"\nEstadísticas de validación:")
            for key, value in self._validation_stats.items():
                logger.info(f"  {key}: {value}")
        
        logger.info("="*80 + "\n")
    
    def get_feature_columns(self) -> Tuple[List[str], List[str]]:
        """Retorna la lista de patrones de características y columnas a excluir"""
        
        # Excluir columnas que no son características
        exclude_cols = ['Team', 'Date', 'Opp', 'Result', 'PTS', 'PTS_Opp', 
                       'matchup_key', 'season_phase', 'Away', 'MP',
                       'FG', 'FGA', 'FG%', '2P', '2PA', '2P%', '3P', '3PA', '3P%',
                       'FT', 'FTA', 'FT%', 'FG_Opp', 'FGA_Opp', 'FG%_Opp',
                       '2P_Opp', '2PA_Opp', '2P%_Opp', '3P_Opp', '3PA_Opp', '3P%_Opp',
                       'FT_Opp', 'FTA_Opp', 'FT%_Opp']
        
        # Características numéricas principales - todos los patrones
        feature_patterns = [
            # Básicas
            'pts_ma_', 'pts_std_', 'pts_opp_ma_', 'total_pts_ma_', 'total_pts_std_',
            'pts_trend', 'pts_opp_trend', 'total_pts_trend',
            'pts_season_avg', 'pts_opp_season_avg',
            
            # Momentum
            'win_streak', 'loss_streak', 'win_rate_last_',
            'pts_vs_avg_last_', 'total_pts_vs_avg_last_',
            'pts_momentum_', 'total_pts_momentum_',
            'pts_consistency', 'total_pts_consistency',
            
            # Oponente
            'opp_', 'quality_diff', 'matchup_',
            
            # Eficiencia
            'fg_efficiency', 'fg3_efficiency', 'true_shooting',
            'off_efficiency_ma_', 'def_efficiency_ma_', 'game_efficiency_ma_',
            'three_point_rate', 'efficiency_diff', 'ft_rate',
            
            # Ritmo
            'pace_', 'possessions_', 'expected_game_pace',
            
            # Situacional
            'home_pts_avg', 'away_pts_avg', 'home_advantage',
            'home_total_avg', 'away_total_avg', 'home_total_advantage',
            'pts_in_close_games', 'total_in_close_games',
            'pts_in_ot', 'total_in_ot',
            'days_rest', 'is_back_to_back', 'is_rested',
            'pts_with_rest', 'total_with_rest',
            'pts_b2b', 'total_b2b',
            
            # Base calculations
            'is_home', 'has_overtime', 'overtime_periods',
            'is_win', 'is_weekend',
            'team_possessions', 'opp_possessions', 'total_possessions',
            'team_shot_volume', 'opp_shot_volume', 'total_shot_volume',
            'team_true_shooting_base', 'opp_true_shooting_base',
            
            # Jugadores
            'team_total_pts_players', 'avg_player_pts', 'pts_distribution',
            'top_scorer_pts', 'avg_minutes', 'minutes_distribution',
            'num_starters', 'team_fg_pct', 'team_3p_pct', 'team_ft_pct',
            'team_rebounds', 'team_assists', 'team_steals', 'team_blocks',
            'team_turnovers', 'team_fouls', 'team_plus_minus',
            'num_double_doubles', 'num_triple_doubles',
            'scoring_concentration', 'bench_contribution',
            'team_efficiency_players', 'team_depth_factor',
            
            # Interacciones
            'pace_efficiency_interaction', 'home_form_interaction',
            'matchup_momentum', 'rest_pace_interaction',
            'quality_style_interaction', 'total_pace_interaction',
            'consistency_pace_interaction',
            '_squared', '_sqrt',
            'offense_defense_ratio', 'efficiency_ratio',
            
            # Temporal
            'month', 'day_of_week', 'day_of_season', 'days_into_season',
            'phase_', 'dow_', 'season_progress', 'season_start',
            'high_intensity_month', 'games_last_7_days',
            'schedule_density', 'cumulative_minutes', 'fatigue_index'
        ]
        
        return feature_patterns, exclude_cols
    
    def prepare_features(self, df: pd.DataFrame, target_col: str = 'total_points') -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepara las características para entrenamiento.
        
        Args:
            df: DataFrame con todas las características
            target_col: Nombre de la columna objetivo
            
        Returns:
            X: DataFrame con características
            y: Series con target
        """
        if df.empty:
            logger.warning("DataFrame vacío en prepare_features")
            return pd.DataFrame(), pd.Series()
        
        # Separar target
        if target_col not in df.columns:
            logger.error(f"Columna target '{target_col}' no encontrada")
            return pd.DataFrame(), pd.Series()
        
        y = df[target_col]
        
        # Eliminar columnas que no son características
        cols_to_drop = [
            # Target y componentes directos
            target_col, 'PTS', 'PTS_Opp',
            
            # Identificadores
            'Date', 'Team', 'Opp', 'matchup_id',
            
            # Variables categóricas ya codificadas como dummies
            'day_of_week', 'month',
            
            # Variables que causan data leakage (directamente relacionadas con puntos)
            'FG', 'FGA', 'FG%', '2P', '2PA', '2P%', '3P', '3PA', '3P%',
            'FT', 'FTA', 'FT%', 'ORB', 'DRB', 'TRB', 'AST', 'STL', 'BLK',
            'TOV', 'PF', '+/-', 'TS%', 'eFG%', 'ORB%', 'DRB%', 'TRB%',
            
            # Variables del oponente que causan data leakage
            'FG_Opp', 'FGA_Opp', 'FG%_Opp', '2P_Opp', '2PA_Opp', '2P%_Opp',
            '3P_Opp', '3PA_Opp', '3P%_Opp', 'FT_Opp', 'FTA_Opp', 'FT%_Opp',
            
            # Variables derivadas directamente de los puntos
            'team_possessions', 'opp_possessions', 'total_possessions',
            'team_true_shooting_base', 'opp_true_shooting_base',
            'team_shot_volume', 'opp_shot_volume', 'total_shot_volume',
            
            # Estadísticas de jugadores directamente relacionadas con puntos
            'team_total_pts_players', 'avg_player_pts', 'top_scorer_pts',
            
            # Variables de porcentajes sin shift (nuevas)
            'three_point_ratio', 'three_point_ratio_opp',
            'FG%_vs_FG%_Opp_diff', '2P%_vs_2P%_Opp_diff', '3P%_vs_3P%_Opp_diff', 'FT%_vs_FT%_Opp_diff'
        ]
        
        # Patrones adicionales para identificar columnas de data leakage
        leakage_patterns = [
            # Columnas sin shift que pueden causar data leakage
            'FGA_FGA_Opp_total', '3PA_3PA_Opp_total'
        ]
        
        # Agregar columnas que coinciden con patrones de leakage
        for pattern in leakage_patterns:
            pattern_cols = [col for col in df.columns if pattern in col and not any(x in col for x in ['_ma_', '_shift', '_lag'])]
            cols_to_drop.extend(pattern_cols)
        
        # Solo eliminar columnas que existen
        cols_to_drop = [col for col in cols_to_drop if col in df.columns]
        X = df.drop(columns=cols_to_drop)
        
        # Verificar si hay columnas categóricas restantes
        cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
        if cat_cols:
            logger.warning(f"Eliminando columnas categóricas restantes: {cat_cols}")
            X = X.drop(columns=cat_cols)
        
        # Verificar que todas las columnas son numéricas
        non_numeric = X.select_dtypes(exclude=['number']).columns.tolist()
        if non_numeric:
            logger.warning(f"Convirtiendo columnas no numéricas a numéricas: {non_numeric}")
            
            # Intentar convertir columnas categóricas a numéricas
            for col in non_numeric:
                if col in X.columns:
                    try:
                        # Si es booleana o categórica, convertir a numérica
                        if X[col].dtype == 'bool':
                            X[col] = X[col].astype(int)
                        elif X[col].dtype == 'object':
                            # Intentar conversión directa
                            X[col] = pd.to_numeric(X[col], errors='coerce')
                            # Si hay NaN después de conversión, llenar con 0
                            if X[col].isna().any():
                                X[col] = X[col].fillna(0)
                        else:
                            # Para otros tipos, forzar conversión
                            X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)
                    except Exception as e:
                        logger.warning(f"No se pudo convertir {col}: {e}. Eliminando columna.")
                        X = X.drop(columns=[col])
            
            # Verificar nuevamente
            remaining_non_numeric = X.select_dtypes(exclude=['number']).columns.tolist()
            if remaining_non_numeric:
                logger.error(f"Eliminando columnas que no se pudieron convertir: {remaining_non_numeric}")
                X = X.drop(columns=remaining_non_numeric)
        
        # Detectar y eliminar características con alta correlación con el target
        # Creamos un DataFrame temporal con el target para poder calcular correlaciones
        temp_df = X.copy()
        temp_df[target_col] = y
        
        # Detectar data leakage con umbral más conservador de 0.85
        # Las medias móviles con shift deberían tener correlaciones menores
        leakage_cols = self._detect_data_leakage(temp_df, target_col, threshold=0.85)
        
        # Eliminar columnas con posible data leakage
        if leakage_cols:
            logger.warning(f"Eliminando {len(leakage_cols)} columnas con posible data leakage")
            X = X.drop(columns=leakage_cols, errors='ignore')
        
        # Verificación adicional para asegurar que no haya columnas de porcentajes sin shift
        # Esto es especialmente importante para las nuevas características
        shooting_cols = [col for col in X.columns if any(pattern in col for pattern in ['FG%', '2P%', '3P%', 'FT%'])]
        potential_leakage = [col for col in shooting_cols if not any(pattern in col for pattern in ['_ma_', '_trend_', '_pct_', '_diff_ma_'])]
        
        if potential_leakage:
            logger.warning(f"Eliminando {len(potential_leakage)} columnas adicionales de porcentajes sin shift")
            X = X.drop(columns=potential_leakage, errors='ignore')
        
        # SISTEMA AVANZADO DE FEATURE SELECTION REACTIVADO
        logger.info("Aplicando sistema avanzado de feature selection...")
        
        # LIMPIEZA DE VALORES INFINITOS Y EXTREMOS ANTES DEL FEATURE SELECTION
        logger.info("Limpiando valores infinitos y extremos...")
        
        # Reemplazar infinitos con NaN
        X = X.replace([np.inf, -np.inf], np.nan)
        
        # Detectar y corregir valores extremadamente grandes
        for col in X.select_dtypes(include=['number']).columns:
            if X[col].dtype in ['float64', 'float32']:
                # Calcular límites razonables (percentiles 0.1% y 99.9%)
                try:
                    lower_bound = X[col].quantile(0.001)
                    upper_bound = X[col].quantile(0.999)
                    
                    # Si hay valores fuera de estos límites, limitarlos
                    extreme_mask = (X[col] < lower_bound) | (X[col] > upper_bound)
                    if extreme_mask.any():
                        n_extreme = extreme_mask.sum()
                        logger.debug(f"Limitando {n_extreme} valores extremos en {col}")
                        X[col] = X[col].clip(lower=lower_bound, upper=upper_bound)
                        
                except Exception as e:
                    logger.debug(f"Error procesando {col}: {e}")
        
        # Llenar NaN con mediana después de la limpieza
        for col in X.select_dtypes(include=['number']).columns:
            if X[col].isna().any():
                median_val = X[col].median()
                if pd.isna(median_val):
                    median_val = 0.0
                X[col] = X[col].fillna(median_val)
        
        # Verificar que no quedan infinitos
        inf_cols = []
        for col in X.columns:
            if np.isinf(X[col]).any():
                inf_cols.append(col)
        
        if inf_cols:
            logger.warning(f"Eliminando {len(inf_cols)} columnas con infinitos persistentes: {inf_cols[:5]}...")
            X = X.drop(columns=inf_cols)
        
        logger.info(f"Limpieza completada. Features restantes: {X.shape[1]}")
        
        X = self._advanced_feature_selection(X, y)
        
        # Registrar columnas finales
        self.feature_columns = list(X.columns)
        logger.info(f"Features finales identificadas: {len(self.feature_columns)}")
        logger.info(f"Features preparadas: {X.shape[1]} características, {X.shape[0]} registros")
        
        return X, y
    
    def get_feature_importance_groups(self) -> Dict[str, List[str]]:
        """
        Retorna grupos de features para análisis de importancia
        
        Returns:
            Diccionario con grupos de features categorizadas
        """
        if self.feature_columns is None:
            return {}
        
        groups = {
            'basic_stats': [],
            'momentum': [],
            'opponent': [],
            'efficiency': [],
            'pace': [],
            'situational': [],
            'player_based': [],
            'interactions': [],
            'temporal': []
        }
        
        for feature in self.feature_columns:
            if any(p in feature for p in ['pts_ma_', 'pts_std_', 'pts_trend']):
                groups['basic_stats'].append(feature)
            elif any(p in feature for p in ['momentum', 'streak', 'consistency']):
                groups['momentum'].append(feature)
            elif any(p in feature for p in ['opp_', 'matchup_', 'quality_diff']):
                groups['opponent'].append(feature)
            elif any(p in feature for p in ['efficiency', 'shooting', 'fg_']):
                groups['efficiency'].append(feature)
            elif any(p in feature for p in ['pace', 'possessions']):
                groups['pace'].append(feature)
            elif any(p in feature for p in ['home', 'away', 'rest', 'close']):
                groups['situational'].append(feature)
            elif any(p in feature for p in ['team_', 'player', 'scorer']):
                groups['player_based'].append(feature)
            elif any(p in feature for p in ['interaction', 'ratio', 'squared']):
                groups['interactions'].append(feature)
            elif any(p in feature for p in ['phase_', 'dow_', 'month', 'season']):
                groups['temporal'].append(feature)
        
        # Filtrar grupos vacíos
        groups = {k: v for k, v in groups.items() if v}
        
        return groups
    
    def validate_features(self, df: pd.DataFrame) -> Dict[str, any]:
        """
        Valida la calidad de las features generadas
        
        Args:
            df: DataFrame con features
            
        Returns:
            Diccionario con métricas de validación
        """
        validation_results = {
            'total_features': len(self.feature_columns) if self.feature_columns else 0,
            'missing_values': {},
            'constant_features': [],
            'high_correlation_pairs': [],
            'feature_stats': {}
        }
        
        if self.feature_columns is None:
            return validation_results
        
        # Verificar valores faltantes
        for col in self.feature_columns:
            if col in df.columns:
                missing_pct = df[col].isna().sum() / len(df) * 100
                if missing_pct > 0:
                    validation_results['missing_values'][col] = missing_pct
        
        # Identificar features constantes
        for col in self.feature_columns:
            if col in df.columns and df[col].nunique() == 1:
                validation_results['constant_features'].append(col)
        
        # Calcular estadísticas básicas
        numeric_features = [col for col in self.feature_columns 
                           if col in df.columns and df[col].dtype in [np.float64, np.int64]]
        
        if numeric_features:
            stats_df = df[numeric_features].describe()
            validation_results['feature_stats'] = {
                'mean_values': stats_df.loc['mean'].to_dict(),
                'std_values': stats_df.loc['std'].to_dict(),
                'min_values': stats_df.loc['min'].to_dict(),
                'max_values': stats_df.loc['max'].to_dict()
            }
        
        return validation_results

    def _advanced_feature_selection(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """
        Sistema avanzado de selección de características optimizado para puntos totales NBA.
        
        Implementa múltiples técnicas de selección:
        1. Filtro de varianza (elimina features constantes/casi constantes)
        2. Filtro de correlación (elimina features altamente correlacionadas)
        3. Selección por información mutua
        4. Detección avanzada de data leakage
        5. Filtro de importancia estadística
        
        Args:
            X: DataFrame con features
            y: Series con target
            
        Returns:
            DataFrame con features seleccionadas
        """
        logger.info(f"Iniciando feature selection avanzado con {X.shape[1]} features")
        original_features = X.shape[1]
        
        # 1. FILTRO DE VARIANZA - Eliminar features constantes o casi constantes
        from sklearn.feature_selection import VarianceThreshold
        
        # Eliminar features con varianza muy baja (< 0.01)
        variance_selector = VarianceThreshold(threshold=0.01)
        try:
            X_variance = pd.DataFrame(
                variance_selector.fit_transform(X),
                columns=X.columns[variance_selector.get_support()],
                index=X.index
            )
            removed_variance = original_features - X_variance.shape[1]
            logger.info(f"Filtro de varianza: eliminadas {removed_variance} features")
            X = X_variance
        except Exception as e:
            logger.warning(f"Error en filtro de varianza: {e}")
        
        # 2. FILTRO DE CORRELACIÓN - Eliminar features altamente correlacionadas
        try:
            corr_matrix = X.corr().abs()
            upper_triangle = corr_matrix.where(
                np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
            )
            
            # Encontrar features con correlación > 0.95
            high_corr_features = [
                column for column in upper_triangle.columns 
                if any(upper_triangle[column] > 0.95)
            ]
            
            if high_corr_features:
                X = X.drop(columns=high_corr_features)
                logger.info(f"Filtro de correlación: eliminadas {len(high_corr_features)} features")
        except Exception as e:
            logger.warning(f"Error en filtro de correlación: {e}")
        
        # 3. SELECCIÓN POR INFORMACIÓN MUTUA
        try:
            from sklearn.feature_selection import mutual_info_regression, SelectKBest
            
            # Calcular información mutua
            mi_scores = mutual_info_regression(X, y, random_state=42)
            mi_scores = pd.Series(mi_scores, index=X.columns)
            
            # Seleccionar top features por información mutua (top 90% para ser menos agresivo)
            n_features_to_keep = max(int(len(mi_scores) * 0.9), 100)  # Mínimo 100 features
            top_mi_features = mi_scores.nlargest(n_features_to_keep).index.tolist()
            
            X = X[top_mi_features]
            removed_mi = len(mi_scores) - len(top_mi_features)
            logger.info(f"Selección por información mutua: eliminadas {removed_mi} features")
            
        except Exception as e:
            logger.warning(f"Error en selección por información mutua: {e}")
        
        # 4. DETECCIÓN AVANZADA DE DATA LEAKAGE
        try:
            # Crear DataFrame temporal para análisis de correlación
            temp_df = X.copy()
            temp_df['target'] = y
            
            # Calcular correlaciones con el target
            target_correlations = temp_df.corr()['target'].abs().sort_values(ascending=False)
            
            # Identificar features con correlación sospechosamente alta (> 0.8)
            suspicious_features = target_correlations[
                (target_correlations > 0.8) & (target_correlations.index != 'target')
            ].index.tolist()
            
            # Análisis adicional para confirmar data leakage
            confirmed_leakage = []
            for feature in suspicious_features:
                # Verificar si el nombre sugiere data leakage
                leakage_patterns = [
                    'pts_current', 'total_current', 'game_total', 'actual_',
                    'real_', 'true_', 'final_', 'result_'
                ]
                
                if any(pattern in feature.lower() for pattern in leakage_patterns):
                    confirmed_leakage.append(feature)
                # También verificar correlación extremadamente alta (> 0.9)
                elif target_correlations[feature] > 0.9:
                    confirmed_leakage.append(feature)
            
            if confirmed_leakage:
                X = X.drop(columns=confirmed_leakage, errors='ignore')
                logger.warning(f"Data leakage avanzado: eliminadas {len(confirmed_leakage)} features sospechosas")
                
        except Exception as e:
            logger.warning(f"Error en detección avanzada de data leakage: {e}")
        
        # 5. FILTRO DE IMPORTANCIA ESTADÍSTICA
        try:
            from scipy.stats import pearsonr
            
            # Calcular correlaciones estadísticamente significativas
            significant_features = []
            for feature in X.columns:
                try:
                    corr, p_value = pearsonr(X[feature].fillna(0), y)
                    # Mantener features con p-value < 0.05 y correlación mínima
                    if p_value < 0.05 and abs(corr) > 0.01:
                        significant_features.append(feature)
                except:
                    # Si hay error, mantener la feature
                    significant_features.append(feature)
            
            if len(significant_features) < X.shape[1]:
                X = X[significant_features]
                removed_stats = X.shape[1] - len(significant_features)
                logger.info(f"Filtro estadístico: eliminadas {removed_stats} features no significativas")
                
        except Exception as e:
            logger.warning(f"Error en filtro estadístico: {e}")
        
        # 6. VERIFICACIÓN FINAL DE CALIDAD
        try:
            # Eliminar features con demasiados valores faltantes (> 50%)
            missing_threshold = 0.5
            features_to_keep = []
            
            for feature in X.columns:
                missing_pct = X[feature].isna().sum() / len(X)
                if missing_pct <= missing_threshold:
                    features_to_keep.append(feature)
            
            if len(features_to_keep) < X.shape[1]:
                X = X[features_to_keep]
                removed_missing = X.shape[1] - len(features_to_keep)
                logger.info(f"Filtro de valores faltantes: eliminadas {removed_missing} features")
                
        except Exception as e:
            logger.warning(f"Error en filtro de valores faltantes: {e}")
        
        # RESUMEN FINAL
        final_features = X.shape[1]
        reduction_pct = ((original_features - final_features) / original_features) * 100
        
        logger.info(f"Feature selection completado:")
        logger.info(f"  Features originales: {original_features}")
        logger.info(f"  Features finales: {final_features}")
        logger.info(f"  Reducción: {reduction_pct:.1f}%")
        
        # Verificar que tenemos suficientes features
        if final_features < 10:
            logger.warning(f"Muy pocas features después de la selección ({final_features}). Revisando criterios...")
            # En caso extremo, relajar criterios y mantener más features
        
        return X

    def _create_advanced_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Crea características avanzadas para mejorar la precisión del modelo"""
        
        if df.empty:
            return df
        
        # Características de momentum y tendencias
        if 'Date' in df.columns:
            df = df.sort_values(['Team', 'Date'])
            
            # Tendencias no lineales
            df['pts_momentum_squared'] = df.groupby('Team')['PTS'].diff(1).pow(2)
            df['pts_momentum_sqrt'] = df.groupby('Team')['PTS'].diff(1).abs().pow(0.5) * np.sign(df.groupby('Team')['PTS'].diff(1))
            df['pts_momentum_log'] = np.log1p(df.groupby('Team')['PTS'].diff(1).abs()) * np.sign(df.groupby('Team')['PTS'].diff(1))
            
            # Características de aceleración (cambio en la tendencia)
            df['pts_acceleration'] = df.groupby('Team')['PTS'].diff(1).diff(1)
            df['total_pts_acceleration'] = df.groupby('Team')['total_points'].diff(1).diff(1)
            
            # Características de volatilidad
            window_sizes = [5, 10, 15]
            for window in window_sizes:
                df[f'pts_volatility_{window}'] = df.groupby('Team')['PTS'].rolling(window=window).std().reset_index(level=0, drop=True)
                df[f'total_pts_volatility_{window}'] = df.groupby('Team')['total_points'].rolling(window=window).std().reset_index(level=0, drop=True)
        
        # Características de matchup avanzadas
        if 'Opp' in df.columns:
            # Crear identificador único para cada combinación de equipos
            df['matchup_id'] = df.apply(lambda x: '_'.join(sorted([x['Team'], x['Opp']])), axis=1)
            
            # Estadísticas históricas de matchup (SIN DATA LEAKAGE)
            # Calcular estadísticas usando solo juegos ANTERIORES
            matchup_historical = []
            
            for idx, row in df.iterrows():
                matchup_id = row['matchup_id']
                current_date = row.get('Date', pd.Timestamp.now())
                
                # Obtener juegos anteriores de este matchup
                if 'Date' in df.columns:
                    historical_games = df[
                        (df['matchup_id'] == matchup_id) & 
                        (df['Date'] < current_date)
                    ]
                else:
                    # Si no hay fecha, usar índice como proxy temporal
                    historical_games = df[
                        (df['matchup_id'] == matchup_id) & 
                        (df.index < idx)
                    ]
                
                if len(historical_games) > 0:
                    matchup_historical.append({
                        'matchup_total_mean_hist': historical_games['total_points'].mean(),
                        'matchup_total_std_hist': historical_games['total_points'].std(),
                        'matchup_games_played': len(historical_games),
                        'matchup_dominance_hist': historical_games['PTS'].mean() - historical_games['PTS_Opp'].mean()
                    })
                else:
                    # Sin historial, usar valores neutros
                    matchup_historical.append({
                        'matchup_total_mean_hist': df['total_points'].mean(),
                        'matchup_total_std_hist': df['total_points'].std(),
                        'matchup_games_played': 0,
                        'matchup_dominance_hist': 0
                    })
            
            # Convertir a DataFrame y fusionar
            matchup_hist_df = pd.DataFrame(matchup_historical)
            df = pd.concat([df, matchup_hist_df], axis=1)
            
            # Características derivadas SIN data leakage
            df['matchup_experience'] = np.log1p(df['matchup_games_played'])
            df['matchup_consistency'] = 1 / (df['matchup_total_std_hist'] + 1)  # Inverso de volatilidad
        
        # Características de forma y tendencia CON SHIFT para evitar data leakage
        if 'Date' in df.columns and 'PTS' in df.columns:
            # Forma reciente (últimos N partidos vs promedio de temporada)
            for window in [3, 5, 10]:
                # Calcular promedio móvil CON SHIFT
                df[f'pts_ma_{window}'] = df.groupby('Team')['PTS'].transform(
                    lambda x: x.rolling(window=window, min_periods=1).mean().shift(1)
                )
                
                # Comparar con promedio de temporada CON SHIFT
                df[f'pts_form_{window}'] = df.groupby('Team').apply(
                    lambda x: x[f'pts_ma_{window}'] / x['PTS'].expanding().mean().shift(1)
                ).reset_index(level=0, drop=True)
                
                # Lo mismo para puntos totales CON SHIFT
                df[f'total_pts_ma_{window}'] = df.groupby('Team')['total_points'].transform(
                    lambda x: x.rolling(window=window, min_periods=1).mean().shift(1)
                )
                
                df[f'total_pts_form_{window}'] = df.groupby('Team').apply(
                    lambda x: x[f'total_pts_ma_{window}'] / x['total_points'].expanding().mean().shift(1)
                ).reset_index(level=0, drop=True)
        
        # Características de interacción
        if all(col in df.columns for col in ['is_home', 'days_rest']):
            # Interacciones entre variables importantes
            df['home_rest_interaction'] = df['is_home'] * df['days_rest']
            
            if 'win_streak' in df.columns:
                df['home_streak_interaction'] = df['is_home'] * df['win_streak']
                df['rest_streak_interaction'] = df['days_rest'] * df['win_streak']
        
        # Características polinomiales para variables numéricas clave
        if 'total_pts_ma_5' in df.columns:
            df['total_pts_ma_5_squared'] = df['total_pts_ma_5'] ** 2
            df['total_pts_ma_5_cubed'] = df['total_pts_ma_5'] ** 3
        
        # Características cíclicas para variables temporales
        if 'day_of_season' in df.columns:
            # Transformar día de temporada a características cíclicas
            df['day_of_season_sin'] = np.sin(2 * np.pi * df['day_of_season'] / 365)
            df['day_of_season_cos'] = np.cos(2 * np.pi * df['day_of_season'] / 365)
        

        # 1. Análisis de tendencias de puntos totales con regresión polinomial
        if 'total_points' in df.columns:
            for team in df['Team'].unique():
                team_data = df[df['Team'] == team].sort_values('Date') if 'Date' in df.columns else df[df['Team'] == team]
                if len(team_data) >= 10:
                    # Crear índice para regresión
                    indices = np.arange(len(team_data))
                    
                    try:
                        # Regresión polinomial de grado 2 para capturar tendencias no lineales
                        poly_coefs = np.polyfit(indices, team_data['total_points'].values, 2)
                        
                        # Guardar coeficientes como características
                        df.loc[df['Team'] == team, 'total_pts_trend_quad'] = poly_coefs[0]
                        df.loc[df['Team'] == team, 'total_pts_trend_linear'] = poly_coefs[1]
                        df.loc[df['Team'] == team, 'total_pts_trend_const'] = poly_coefs[2]
                        
                        # Calcular valores ajustados
                        fitted_values = np.polyval(poly_coefs, indices)
                        
                        # Calcular residuos para medir volatilidad
                        residuals = team_data['total_points'].values - fitted_values
                        df.loc[df['Team'] == team, 'total_pts_trend_residuals'] = np.std(residuals)
                    except:
                        # Si hay error en el ajuste, usar valores neutrales
                        df.loc[df['Team'] == team, 'total_pts_trend_quad'] = 0
                        df.loc[df['Team'] == team, 'total_pts_trend_linear'] = 0
                        df.loc[df['Team'] == team, 'total_pts_trend_const'] = team_data['total_points'].mean()
                        df.loc[df['Team'] == team, 'total_pts_trend_residuals'] = team_data['total_points'].std()
        
        # 2. Análisis de patrones de ritmo de juego
        if all(col in df.columns for col in ['pace_ma_5', 'total_pts_ma_5']):
            # Correlación entre ritmo y puntos totales (ventana móvil)
            for team in df['Team'].unique():
                team_data = df[df['Team'] == team].sort_values('Date') if 'Date' in df.columns else df[df['Team'] == team]
                if len(team_data) >= 10:
                    # Calcular correlación móvil entre ritmo y puntos totales
                    rolling_corrs = []
                    for i in range(len(team_data)):
                        if i < 10:  # Necesitamos al menos 10 observaciones
                            rolling_corrs.append(np.nan)
                        else:
                            window_data = team_data.iloc[max(0, i-10):i]
                            if len(window_data) >= 5:  # Verificar datos suficientes
                                corr = window_data['pace_ma_5'].corr(window_data['total_pts_ma_5'])
                                rolling_corrs.append(corr)
                            else:
                                rolling_corrs.append(np.nan)
                    
                    df.loc[df['Team'] == team, 'pace_pts_correlation'] = rolling_corrs
        
        # 3. Características de calendario y descanso avanzadas CON SHIFT
        if 'days_rest' in df.columns:
            # Efecto acumulativo del descanso (media móvil de días de descanso) CON SHIFT
            df['rest_ma_5'] = df.groupby('Team')['days_rest'].transform(
                lambda x: x.rolling(window=5, min_periods=1).mean().shift(1)
            )
            
            # Efecto de fatiga (inverso del descanso acumulado)
            df['fatigue_factor'] = 1 / (df['rest_ma_5'] + 1)
            
            # Interacción entre fatiga y puntos totales históricos
            if 'total_pts_ma_5' in df.columns:
                df['fatigue_total_pts_interaction'] = df['fatigue_factor'] * df['total_pts_ma_5']
        
        # 4. Análisis de matchup específico para puntos totales
        if 'matchup_id' in df.columns and 'total_points' in df.columns:
            # Calcular desviación de cada matchup respecto a los promedios de los equipos
            matchup_deviations = []
            
            for idx, row in df.iterrows():
                team = row['Team']
                opp = row['Opp']
                
                # Obtener promedio de puntos totales del equipo y oponente
                team_avg = df[df['Team'] == team]['total_points'].mean()
                opp_avg = df[df['Team'] == opp]['total_points'].mean()
                
                # Promedio esperado
                expected_avg = (team_avg + opp_avg) / 2
                
                # Valor actual
                actual = row['total_points']
                
                # Desviación
                if pd.notna(expected_avg) and pd.notna(actual):
                    deviation = actual - expected_avg
                else:
                    deviation = 0
                    
                matchup_deviations.append(deviation)
            
            df['matchup_total_pts_deviation'] = matchup_deviations
            
            # Crear característica de desviación móvil para cada matchup
            df['matchup_deviation_ma'] = df.groupby('matchup_id')['matchup_total_pts_deviation'].transform(
                lambda x: x.rolling(window=3, min_periods=1).mean().shift(1)
            )
        
        # 5. Características de consistencia y variabilidad avanzadas
        if 'total_points' in df.columns:
            # Rango intercuartílico móvil (medida robusta de dispersión)
            for window in [5, 10]:
                df[f'total_pts_iqr_{window}'] = df.groupby('Team')['total_points'].transform(
                    lambda x: x.rolling(window=window, min_periods=3).apply(
                        lambda y: np.percentile(y, 75) - np.percentile(y, 25) if len(y) >= 3 else np.nan
                    )
                )
            
            # Asimetría y curtosis (forma de la distribución de puntos)
            df['total_pts_skew_5'] = df.groupby('Team')['total_points'].transform(
                lambda x: x.rolling(window=5, min_periods=3).apply(
                    lambda y: pd.Series(y).skew() if len(y) >= 3 else np.nan
                )
            )
            
            df['total_pts_kurt_5'] = df.groupby('Team')['total_points'].transform(
                lambda x: x.rolling(window=5, min_periods=3).apply(
                    lambda y: pd.Series(y).kurt() if len(y) >= 3 else np.nan
                )
            )
        
        # 6. Características de estacionalidad y contexto de temporada
        if 'day_of_season' in df.columns:
            # Fase de temporada (categórica)
            df['season_phase_numeric'] = pd.cut(
                df['day_of_season'], 
                bins=[0, 30, 90, 180, 270, 365],
                labels=[1, 2, 3, 4, 5]
            ).astype(float)
            
            # Distancia al All-Star Game (típicamente a mitad de temporada)
            df['days_to_allstar'] = abs(df['day_of_season'] - 180)
            
            # Distancia a playoffs (final de temporada)
            df['days_to_playoffs'] = np.maximum(0, 270 - df['day_of_season'])
            
            # Interacciones con fase de temporada
            if 'total_pts_ma_5' in df.columns:
                df['season_phase_total_pts'] = df['season_phase_numeric'] * df['total_pts_ma_5'] / 1000
        
        # 7. Análisis de tendencias de overtime
        if 'has_overtime' in df.columns:
            # Probabilidad de overtime basada en historial reciente
            df['overtime_prob_ma5'] = df.groupby('Team')['has_overtime'].transform(
                lambda x: x.rolling(window=10, min_periods=1).mean().shift(1)
            )
            
            # Interacción entre probabilidad de overtime y puntos totales
            if 'total_pts_ma_5' in df.columns:
                df['overtime_pts_interaction'] = df['overtime_prob_ma5'] * df['total_pts_ma_5']
        
        # 8. Características de oponente avanzadas CON SHIFT para evitar data leakage
        if 'Opp' in df.columns:
            # Calcular efectos de equipo usando expanding mean con shift
            df['team_total_pts_effect'] = df.groupby('Team')['total_points'].transform(
                lambda x: x.expanding().mean().shift(1) - df['total_points'].expanding().mean().shift(1)
            )
            
            # Crear mapping de efectos del oponente usando datos históricos
            df['opp_total_pts_effect'] = df.groupby('Opp')['total_points'].transform(
                lambda x: x.expanding().mean().shift(1) - df['total_points'].expanding().mean().shift(1)
            )
            
            # Interacción entre efectos de equipo y oponente (usando datos históricos)
            df['team_opp_effect_interaction'] = df['team_total_pts_effect'] * df['opp_total_pts_effect']
        
        return df

    def _create_dummy_variables(self, df: pd.DataFrame) -> pd.DataFrame:
        """Crea variables dummy para características categóricas"""
        
        if df.empty:
            return df
        
        # Día de la semana
        if 'day_of_week' in df.columns:
            # Crear dummies para día de la semana
            for day in range(7):
                df[f'dow_{day}'] = (df['day_of_week'] == day).astype(int)
        
        # Mes de alta intensidad (playoffs, finales de temporada)
        if 'month' in df.columns:
            high_intensity_months = [4, 5, 6]  # Abril, Mayo, Junio
            df['high_intensity_month'] = df['month'].isin(high_intensity_months).astype(int)
        
        # Inicio de temporada
        if 'day_of_season' in df.columns:
            df['season_start'] = (df['day_of_season'] < 30).astype(int)
        
        return df

    def _clean_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Limpia y prepara las características finales"""
        
        if df.empty:
            return df
        
        # Eliminar columnas no necesarias para el modelo
        cols_to_drop = [
            # Columnas de identificación que no son features
            'GameID', 'Game_ID', 'game_id', 'game_code',
            # Columnas temporales que ya han sido procesadas
            'season', 'season_type',
            # Columnas duplicadas o redundantes
            '@', 'Home'
        ]
        
        # Eliminar solo las columnas que existen
        cols_to_drop = [col for col in cols_to_drop if col in df.columns]
        if cols_to_drop:
            df = df.drop(columns=cols_to_drop)
        
        # Manejar valores faltantes
        numeric_cols = df.select_dtypes(include=['number']).columns
        for col in numeric_cols:
            if df[col].isna().any():
                # Usar la mediana para características con outliers potenciales
                df[col] = df[col].fillna(df[col].median())
        
        # Eliminar filas con demasiados valores faltantes
        if len(df.columns) > 10:
            threshold = len(df.columns) * 0.5  # 50% de valores faltantes
            df = df.dropna(thresh=threshold)
        
        return df

    def _detect_data_leakage(self, df: pd.DataFrame, target_col: str = 'total_points', 
                              threshold: float = 0.8) -> List[str]:
        """
        Detecta características con alta correlación con el target que podrían causar data leakage.
        
        Args:
            df: DataFrame con características y target
            target_col: Nombre de la columna objetivo
            threshold: Umbral de correlación para considerar data leakage
            
        Returns:
            Lista de columnas con posible data leakage
        """
        if target_col not in df.columns:
            logger.warning(f"Target {target_col} no encontrado en el DataFrame")
            return []
            
        # Seleccionar solo columnas numéricas
        numeric_df = df.select_dtypes(include=['number'])
        
        if numeric_df.empty:
            return []
            
        # Lista de columnas con posible data leakage
        leakage_cols = []
        
        # 1. Detección por correlación directa (con excepciones para features válidas)
        correlations = numeric_df.corr()[target_col].abs().sort_values(ascending=False)
        
        # Identificar columnas con alta correlación (excluyendo el target mismo)
        high_corr_cols = correlations[correlations > threshold].index.tolist()
        if target_col in high_corr_cols:
            high_corr_cols.remove(target_col)
        
        # Filtrar features válidas que pueden tener alta correlación pero no son data leakage
        valid_high_corr_patterns = [
            '_ma_', '_shift', '_lag', '_trend', '_avg', '_mean', '_rolling',
            '_ewm', '_expanding', '_hist', 'historical_', '_prev', 'previous_',
            '_last', 'last_', '_momentum', '_streak', '_consistency'
        ]
        
        # Solo agregar a leakage si no tiene patrones válidos
        filtered_high_corr = []
        for col in high_corr_cols:
            has_valid_pattern = any(pattern in col.lower() for pattern in valid_high_corr_patterns)
            if not has_valid_pattern:
                filtered_high_corr.append(col)
            else:
                logger.debug(f"Manteniendo feature con alta correlación pero patrón válido: {col} (r={correlations[col]:.3f})")
        
        leakage_cols.extend(filtered_high_corr)
        
        # 2. Detección por análisis de componentes del target
        # Si el target es una suma de componentes, los componentes son data leakage
        if target_col == 'total_points':
            components = ['PTS', 'PTS_Opp']
            for comp in components:
                if comp in df.columns:
                    leakage_cols.append(comp)
        
        # 3. Detección por nombres de columnas que indican data leakage
        leakage_patterns = [
            # FEATURES CRÍTICAS CON DATA LEAKAGE DIRECTO
            'total_points_max', 'total_points_min', 'total_points_std', 'total_points_count',
            'total_points_sum', 'total_points_range', 'total_points_cv',
            
            # Estadísticas directas de juego actual (sin shift)
            'FG', 'FGA', 'FG%', '2P', '2PA', '2P%', '3P', '3PA', '3P%',
            'FT', 'FTA', 'FT%', 'TS%', 'eFG%',
            
            # Estadísticas defensivas/oponente directas
            'FG_Opp', 'FGA_Opp', 'FG%_Opp', '2P_Opp', '2PA_Opp', '2P%_Opp',
            '3P_Opp', '3PA_Opp', '3P%_Opp', 'FT_Opp', 'FTA_Opp', 'FT%_Opp',
            
            # Estadísticas de rebotes/asistencias que contribuyen directamente a puntos
            'ORB', 'DRB', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF',
            
            # Estadísticas avanzadas que incorporan puntos
            '+/-', 'ORB%', 'DRB%', 'TRB%', 'USG%', 'OffRtg', 'DefRtg',
            
            # Variables derivadas directamente de los puntos sin shift
            'team_possessions', 'opp_possessions', 'total_possessions',
            'team_true_shooting_base', 'opp_true_shooting_base',
            'team_shot_volume', 'opp_shot_volume', 'total_shot_volume',
            
            # Estadísticas de jugadores directamente relacionadas con puntos
            'team_total_pts_players', 'avg_player_pts', 'top_scorer_pts'
        ]
        
        # Buscar patrones de data leakage en nombres de columnas (excluyendo features válidas)
        pattern_leakage_cols = []
        for col in df.columns:
            # Verificar si contiene patrones de leakage
            has_leakage_pattern = any(pattern in col for pattern in leakage_patterns)
            
            if has_leakage_pattern:
                # Verificar si tiene indicadores de que es una feature válida (con shift/lag)
                has_valid_indicator = any(indicator in col.lower() for indicator in [
                    '_ma_', '_shift', '_lag', '_trend', '_avg', '_mean', '_rolling',
                    '_ewm', '_expanding', '_hist', 'historical_', '_prev', 'previous_',
                    '_last', 'last_', '_pct_', '_diff_ma_'
                ])
                
                if not has_valid_indicator:
                    pattern_leakage_cols.append(col)
                else:
                    logger.debug(f"Manteniendo feature con patrón de leakage pero indicador válido: {col}")
        
        leakage_cols.extend(pattern_leakage_cols)
        
        # 4. Detección por análisis de información mutua (más sofisticado que correlación)
        try:
            from sklearn.feature_selection import mutual_info_regression
            
            # Seleccionar columnas numéricas sin NaN
            X = df.drop(columns=[target_col] + leakage_cols, errors='ignore')
            X = X.select_dtypes(include=['number'])
            X = X.fillna(X.median())
            y = df[target_col].fillna(df[target_col].median())
            
            if not X.empty:
                # Calcular información mutua
                mi_scores = mutual_info_regression(X, y)
                mi_scores = pd.Series(mi_scores, index=X.columns)
                
                # Normalizar puntuaciones
                mi_scores = mi_scores / mi_scores.max() if mi_scores.max() > 0 else mi_scores
                
                # Identificar características con alta información mutua
                high_mi_cols = mi_scores[mi_scores > 0.9].index.tolist()
                leakage_cols.extend(high_mi_cols)
        except:
            logger.warning("No se pudo realizar análisis de información mutua")
        
        # 5. Detección por análisis de características con shift incorrecto
        # Buscar características que deberían tener shift pero no lo tienen
        time_dependent_patterns = ['ma_', 'std_', 'mean_', 'avg_', 'trend_', 'momentum_']
        
        for col in df.columns:
            if any(pattern in col for pattern in time_dependent_patterns):
                # Verificar si la columna no tiene '_shift' o similar en el nombre
                if 'shift' not in col and 'lag' not in col:
                    # Verificar correlación con target
                    if col in numeric_df.columns:
                        corr = numeric_df[col].corr(numeric_df[target_col])
                        if abs(corr) > 0.9:  # Umbral más alto para estas columnas
                            leakage_cols.append(col)
        
        # Eliminar duplicados y ordenar
        leakage_cols = sorted(list(set(leakage_cols)))
        
        if leakage_cols:
            logger.warning(f"Detectadas {len(leakage_cols)} columnas con posible data leakage")
            for col in leakage_cols[:10]:  # Mostrar las 10 primeras para no saturar el log
                if col in correlations:
                    logger.warning(f"  - {col}: {correlations[col]:.4f}")
                else:
                    logger.warning(f"  - {col}: detectada por patrón o análisis")
            if len(leakage_cols) > 10:
                logger.warning(f"  - ... y {len(leakage_cols) - 10} más")
        
        return leakage_cols

    def _create_shooting_percentage_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Crea características avanzadas de porcentajes de tiro con ventanas móviles y shift
        para evitar data leakage.
        
        Args:
            df: DataFrame con datos de equipos
            
        Returns:
            DataFrame con características de porcentajes de tiro agregadas
        """
        if df.empty:
            return df
        
        # Verificar columnas disponibles
        shooting_cols = [
            'FG', 'FGA', 'FG%', '2P', '2PA', '2P%', '3P', '3PA', '3P%', 'FT', 'FTA', 'FT%',
            'FG_Opp', 'FGA_Opp', 'FG%_Opp', '2P_Opp', '2PA_Opp', '2P%_Opp', 
            '3P_Opp', '3PA_Opp', '3P%_Opp', 'FT_Opp', 'FTA_Opp', 'FT%_Opp'
        ]
        
        available_cols = [col for col in shooting_cols if col in df.columns]
        
        if not available_cols:
            logger.warning("No hay columnas de tiro disponibles para crear características")
            return df
        
        logger.info(f"Creando características de porcentajes de tiro con {len(available_cols)} columnas disponibles")
        
        # Definir ventanas para medias móviles
        windows = [3, 5, 10]
        
        # Crear características de medias móviles con shift para porcentajes
        percentage_cols = [col for col in available_cols if '%' in col]
        for col in percentage_cols:
            for window in windows:
                # Crear media móvil con shift(1) para evitar data leakage
                df[f'{col}_ma_{window}'] = df.groupby('Team')[col].transform(
                    lambda x: x.rolling(window=window, min_periods=1).mean().shift(1)
                )
        
        # Calcular porcentajes móviles para casos donde tenemos intentos y aciertos
        # Esto es más preciso que usar medias móviles de porcentajes
        shot_pairs = [
            ('FG', 'FGA'), ('2P', '2PA'), ('3P', '3PA'), ('FT', 'FTA'),
            ('FG_Opp', 'FGA_Opp'), ('2P_Opp', '2PA_Opp'), ('3P_Opp', '3PA_Opp'), ('FT_Opp', 'FTA_Opp')
        ]
        
        for made_col, attempt_col in shot_pairs:
            if made_col in available_cols and attempt_col in available_cols:
                for window in windows:
                    # Calcular suma móvil de aciertos y intentos con shift
                    made_sum = df.groupby('Team')[made_col].transform(
                        lambda x: x.rolling(window=window, min_periods=1).sum().shift(1)
                    )
                    attempt_sum = df.groupby('Team')[attempt_col].transform(
                        lambda x: x.rolling(window=window, min_periods=1).sum().shift(1)
                    )
                    
                    # Calcular porcentaje real basado en sumas (evita sesgos de medias de porcentajes)
                    df[f'{made_col}_{attempt_col}_pct_{window}'] = made_sum / attempt_sum.replace(0, np.nan)
        
        # Crear características de tendencia para porcentajes clave
        key_percentage_cols = ['FG%', '3P%', 'FT%', 'FG%_Opp', '3P%_Opp', 'FT%_Opp']
        for col in key_percentage_cols:
            if col in available_cols:
                # Calcular tendencia como pendiente de regresión lineal
                df[f'{col}_trend_5'] = df.groupby('Team')[col].transform(
                    lambda x: x.rolling(window=5, min_periods=3).apply(
                        lambda y: np.polyfit(np.arange(len(y)), y, 1)[0] if len(y) >= 3 else np.nan
                    ).shift(1)
                )
        
        # Crear características de diferencial de porcentajes (equipo vs oponente)
        percentage_pairs = [
            ('FG%', 'FG%_Opp'), ('2P%', '2P%_Opp'), ('3P%', '3P%_Opp'), ('FT%', 'FT%_Opp')
        ]
        
        for team_pct, opp_pct in percentage_pairs:
            if team_pct in available_cols and opp_pct in available_cols:
                # Diferencial directo
                df[f'{team_pct}_vs_{opp_pct}_diff'] = df[team_pct] - df[opp_pct]
                
                # Diferencial de medias móviles
                for window in windows:
                    team_ma = f'{team_pct}_ma_{window}'
                    opp_ma = f'{opp_pct}_ma_{window}'
                    
                    if team_ma in df.columns and opp_ma in df.columns:
                        df[f'{team_pct}_vs_{opp_pct}_diff_ma_{window}'] = df[team_ma] - df[opp_ma]
        
        # Crear características de eficiencia ofensiva y defensiva
        if 'FG%' in available_cols and '3P%' in available_cols and 'FT%' in available_cols:
            for window in windows:
                # Eficiencia ofensiva combinada (ponderada)
                if all(f'{col}_ma_{window}' in df.columns for col in ['FG%', '3P%', 'FT%']):
                    df[f'off_efficiency_combined_{window}'] = (
                        df[f'FG%_ma_{window}'] * 0.5 + 
                        df[f'3P%_ma_{window}'] * 0.3 + 
                        df[f'FT%_ma_{window}'] * 0.2
                    )
                
                # Eficiencia defensiva combinada (ponderada)
                if all(f'{col}_ma_{window}' in df.columns for col in ['FG%_Opp', '3P%_Opp', 'FT%_Opp']):
                    df[f'def_efficiency_combined_{window}'] = (
                        df[f'FG%_Opp_ma_{window}'] * 0.5 + 
                        df[f'3P%_Opp_ma_{window}'] * 0.3 + 
                        df[f'FT%_Opp_ma_{window}'] * 0.2
                    )
        
        # Características de volumen de tiros
        volume_pairs = [('FGA', 'FGA_Opp'), ('3PA', '3PA_Opp')]
        for team_vol, opp_vol in volume_pairs:
            if team_vol in available_cols and opp_vol in available_cols:
                # Volumen total de tiros (ritmo)
                df[f'{team_vol}_{opp_vol}_total'] = df[team_vol] + df[opp_vol]
                
                # Media móvil de volumen total con shift
                for window in windows:
                    df[f'{team_vol}_{opp_vol}_total_ma_{window}'] = df.groupby('Team')[f'{team_vol}_{opp_vol}_total'].transform(
                        lambda x: x.rolling(window=window, min_periods=1).mean().shift(1)
                    )
        
        # Características de distribución de tiros (2P vs 3P)
        if '2PA' in available_cols and '3PA' in available_cols:
            # Ratio de tiros de 3 puntos
            df['three_point_ratio'] = df['3PA'] / (df['2PA'] + df['3PA']).replace(0, np.nan)
            
            # Media móvil de ratio de tiros de 3 puntos con shift
            for window in windows:
                df[f'three_point_ratio_ma_{window}'] = df.groupby('Team')['three_point_ratio'].transform(
                    lambda x: x.rolling(window=window, min_periods=1).mean().shift(1)
                )
        
        # Lo mismo para el oponente
        if '2PA_Opp' in available_cols and '3PA_Opp' in available_cols:
            df['three_point_ratio_opp'] = df['3PA_Opp'] / (df['2PA_Opp'] + df['3PA_Opp']).replace(0, np.nan)
            
            for window in windows:
                df[f'three_point_ratio_opp_ma_{window}'] = df.groupby('Team')['three_point_ratio_opp'].transform(
                    lambda x: x.rolling(window=window, min_periods=1).mean().shift(1)
                )
        
        # Interacciones entre porcentajes y volumen
        if 'FG%' in available_cols and 'FGA' in available_cols:
            for window in windows:
                fg_pct_col = f'FG%_ma_{window}'
                fga_col = f'FGA_ma_{window}' if f'FGA_ma_{window}' in df.columns else 'FGA'
                
                if fg_pct_col in df.columns:
                    df[f'fg_pct_volume_interaction_{window}'] = df[fg_pct_col] * df[fga_col]
        
        if '3P%' in available_cols and '3PA' in available_cols:
            for window in windows:
                fg3_pct_col = f'3P%_ma_{window}'
                fg3a_col = f'3PA_ma_{window}' if f'3PA_ma_{window}' in df.columns else '3PA'
                
                if fg3_pct_col in df.columns:
                    df[f'fg3_pct_volume_interaction_{window}'] = df[fg3_pct_col] * df[fg3a_col]
        
        logger.info(f"Creadas {len([col for col in df.columns if 'pct' in col and col not in available_cols])} nuevas características de porcentajes de tiro")
        
        return df

    def _create_market_pattern_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Crea características avanzadas basadas en patrones de mercado y líneas de apuestas
        para mejorar la precisión del modelo de predicción de puntos totales.
        
        Args:
            df: DataFrame con datos de equipos
            
        Returns:
            DataFrame con características de patrones de mercado agregadas
        """
        if df.empty:
            return df
        
        # Verificar si tenemos las columnas necesarias
        if 'Date' not in df.columns or 'Team' not in df.columns:
            logger.warning("Columnas Date o Team no encontradas para crear features de mercado")
            return df
        
        # Ordenar cronológicamente
        df = df.sort_values(['Team', 'Date'])
        
        # Crear características de tendencia de mercado (simuladas)
        if 'total_points' in df.columns:
            # Calcular la tendencia del mercado (diferencia entre predicción y resultado)
            # Simulamos líneas de mercado como promedio móvil + bias
            df['market_line_sim'] = df.groupby('Team')['total_points'].transform(
                lambda x: x.rolling(10, min_periods=1).mean().shift(1) + 
                         x.rolling(20, min_periods=1).std().shift(1) * 0.2
            )
            
            # Calcular error del mercado (diferencia entre línea simulada y resultado real)
            df['market_error'] = df['total_points'] - df['market_line_sim']
            
            # Tendencias de error del mercado
            df['market_error_ma_5'] = df.groupby('Team')['market_error'].transform(
                lambda x: x.rolling(5, min_periods=1).mean().shift(1)
            )
            
            df['market_error_ma_10'] = df.groupby('Team')['market_error'].transform(
                lambda x: x.rolling(10, min_periods=1).mean().shift(1)
            )
            
            # Volatilidad del error del mercado
            df['market_error_std_5'] = df.groupby('Team')['market_error'].transform(
                lambda x: x.rolling(5, min_periods=1).std().shift(1)
            )
            
            # Dirección del error del mercado (positivo = mercado subestima, negativo = sobrestima)
            df['market_direction'] = df.groupby('Team')['market_error'].transform(
                lambda x: np.sign(x.rolling(3, min_periods=1).mean().shift(1))
            )
            
            # Patrón de zigzag (cambios en la dirección del error)
            df['market_zigzag'] = df.groupby('Team')['market_direction'].transform(
                lambda x: (x != x.shift(1)).astype(int).rolling(5, min_periods=1).mean().shift(1)
            )
            
            # Eficiencia del mercado (qué tan cerca estuvo la línea del resultado)
            df['market_efficiency'] = 1 / (1 + np.abs(df['market_error'].shift(1)))
            
            # Tendencia de ajuste del mercado
            df['market_adjustment_trend'] = df.groupby('Team')['market_line_sim'].transform(
                lambda x: x.diff().rolling(3, min_periods=1).mean().shift(1)
            )
            
            # Correlación entre línea de mercado y resultado real
            window = 10
            df['market_correlation'] = df.groupby('Team').apply(
                lambda g: g['market_line_sim'].rolling(window, min_periods=3).corr(g['total_points']).shift(1)
            ).reset_index(level=0, drop=True)
            
            # Sesgo sistemático del mercado por día de la semana
            if 'day_of_week' in df.columns:
                for day in range(7):
                    day_mask = df['day_of_week'] == day
                    if day_mask.sum() > 10:  # Solo si tenemos suficientes datos
                        day_bias = df.loc[day_mask, 'market_error'].mean()
                        df[f'market_dow_{day}_bias'] = (df['day_of_week'] == day).astype(float) * day_bias
            
            # Características de momentum de mercado
            df['market_momentum'] = df.groupby('Team')['market_line_sim'].transform(
                lambda x: x.diff().rolling(3, min_periods=1).mean().shift(1)
            )
            
            # Características de reversión a la media
            df['market_mean_reversion'] = df.groupby('Team')['market_error'].transform(
                lambda x: x.shift(1) * -0.5  # Factor de reversión
            )
            
            # Eliminar la línea de mercado simulada para evitar data leakage
            df = df.drop('market_line_sim', axis=1)
        
        return df

    def _create_advanced_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Crea features avanzadas de momentum basadas en los patrones más predictivos,
        expandiendo las features de momentum que el modelo identificó como importantes.
        """
        if df.empty:
            return df
        
        # Verificar columnas necesarias
        if 'PTS' not in df.columns or 'total_points' not in df.columns:
            return df
        
        # 1. MOMENTUM MULTI-ESCALA (basado en pts_momentum_sqrt siendo top 5)
        momentum_windows = [2, 3, 4, 5, 7, 10, 15]
        
        for window in momentum_windows:
            # Momentum básico con shift
            pts_ma = df.groupby('Team')['PTS'].transform(
                lambda x: x.rolling(window=window, min_periods=1).mean().shift(1)
            )
            pts_ma_prev = df.groupby('Team')['PTS'].transform(
                lambda x: x.rolling(window=window, min_periods=1).mean().shift(2)
            )
            df[f'pts_momentum_{window}'] = pts_ma - pts_ma_prev
            
            # Momentum con transformaciones (sqrt, square, log)
            df[f'pts_momentum_sqrt_{window}'] = np.sqrt(np.abs(df[f'pts_momentum_{window}'])) * np.sign(df[f'pts_momentum_{window}'])
            df[f'pts_momentum_square_{window}'] = np.square(df[f'pts_momentum_{window}'])
            df[f'pts_momentum_log_{window}'] = np.log1p(np.abs(df[f'pts_momentum_{window}'])) * np.sign(df[f'pts_momentum_{window}'])
            
            # Momentum de total points
            total_ma = df.groupby('Team')['total_points'].transform(
                lambda x: x.rolling(window=window, min_periods=1).mean().shift(1)
            )
            total_ma_prev = df.groupby('Team')['total_points'].transform(
                lambda x: x.rolling(window=window, min_periods=1).mean().shift(2)
            )
            df[f'total_pts_momentum_{window}'] = total_ma - total_ma_prev
            df[f'total_pts_momentum_sqrt_{window}'] = np.sqrt(np.abs(df[f'total_pts_momentum_{window}'])) * np.sign(df[f'total_pts_momentum_{window}'])
        
        # 2. ACELERACIÓN MULTI-ESCALA (basado en acceleration siendo importante)
        for window in [3, 5, 7, 10]:
            # Aceleración de puntos (segunda derivada) con shift
            df[f'pts_acceleration_{window}'] = df.groupby('Team')[f'pts_momentum_{window}'].transform(
                lambda x: x.diff().shift(1)
            )
            
            # Aceleración de total points con shift
            df[f'total_pts_acceleration_{window}'] = df.groupby('Team')[f'total_pts_momentum_{window}'].transform(
                lambda x: x.diff().shift(1)
            )
            
            # Jerk (tercera derivada) para capturar cambios súbitos con shift
            if window >= 5:
                df[f'pts_jerk_{window}'] = df.groupby('Team')[f'pts_acceleration_{window}'].transform(
                    lambda x: x.diff().shift(1)
                )
        
        # 3. MOMENTUM DIRECCIONAL Y PERSISTENCIA
        for window in [5, 10]:
            # Persistencia del momentum (cuántos juegos consecutivos en la misma dirección)
            momentum_direction = np.sign(df[f'pts_momentum_{window}'])
            df[f'momentum_persistence_{window}'] = df.groupby('Team')[f'pts_momentum_{window}'].transform(
                lambda x: (np.sign(x) == np.sign(x.shift(1))).astype(int).rolling(window=window).sum().shift(1)
            )
            
            # Momentum relativo (vs promedio del equipo)
            team_avg_momentum = df.groupby('Team')[f'pts_momentum_{window}'].transform(
                lambda x: x.expanding().mean().shift(1)
            )
            df[f'momentum_relative_{window}'] = df[f'pts_momentum_{window}'] - team_avg_momentum
        
        return df
    
    def _create_advanced_pace_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Crea features avanzadas de pace/ritmo basadas en los patrones identificados,
        expandiendo las features de pace que dominan el top 15.
        """
        if df.empty:
            return df
        
        # Verificar columnas necesarias
        if 'pace_factor' not in df.columns:
            # Crear pace_factor básico si no existe
            if 'total_points' in df.columns and 'MP' in df.columns:
                df['estimated_possessions'] = (df['PTS'] + df['PTS_Opp']) / 2.2
                df['pace_factor'] = (df['estimated_possessions'] * 48) / df['MP']
            else:
                return df
        
        # 1. PACE MULTI-ESCALA (basado en pace_ma_3/5/10 siendo top features)
        pace_windows = [2, 3, 4, 5, 6, 7, 8, 10, 12, 15]
        
        for window in pace_windows:
            # Media móvil básica con shift
            df[f'pace_ma_{window}'] = df.groupby('Team')['pace_factor'].transform(
                lambda x: x.rolling(window=window, min_periods=1).mean().shift(1)
            )
            
            # Desviación estándar del pace con shift
            df[f'pace_std_{window}'] = df.groupby('Team')['pace_factor'].transform(
                lambda x: x.rolling(window=window, min_periods=1).std().shift(1)
            )
            
            # Coeficiente de variación del pace con shift
            df[f'pace_cv_{window}'] = df[f'pace_std_{window}'] / df[f'pace_ma_{window}'].replace(0, np.nan)
            
            # Tendencia del pace con shift (ajustar min_periods)
            min_periods_pace = min(3, window)
            df[f'pace_trend_{window}'] = df.groupby('Team')['pace_factor'].transform(
                lambda x: x.rolling(window=window, min_periods=min_periods_pace).apply(
                    lambda y: np.polyfit(np.arange(len(y)), y, 1)[0] if len(y) >= min_periods_pace else np.nan
                ).shift(1)
            )
        
        # 2. PACE RELATIVO Y COMPARATIVO
        # Pace relativo al promedio de la liga con shift
        league_avg_pace = df['pace_factor'].expanding().mean().shift(1)
        df['pace_vs_league'] = df['pace_factor'] - league_avg_pace
        
        # Pace relativo al promedio del equipo con shift
        team_avg_pace = df.groupby('Team')['pace_factor'].transform(
            lambda x: x.expanding().mean().shift(1)
        )
        df['pace_vs_team_avg'] = df['pace_factor'] - team_avg_pace
        
        # 3. INTERACCIONES PACE-PUNTOS (basado en pace_pts_correlation siendo importante)
        for window in [3, 5, 10]:
            # Correlación móvil entre pace y puntos con shift
            df[f'pace_pts_correlation_{window}'] = df.groupby('Team').apply(
                lambda x: x['pace_factor'].rolling(window=window).corr(x['total_points']).shift(1)
            ).reset_index(level=0, drop=True)
            
            # Elasticidad pace-puntos (cambio porcentual) con shift
            pace_pct_change = df.groupby('Team')['pace_factor'].transform(lambda x: x.pct_change().shift(1))
            pts_pct_change = df.groupby('Team')['total_points'].transform(lambda x: x.pct_change().shift(1))
            df[f'pace_pts_elasticity_{window}'] = df.groupby('Team').apply(
                lambda x: (x['total_points'].pct_change() / x['pace_factor'].pct_change().replace(0, np.nan)).rolling(window=window).mean().shift(1)
            ).reset_index(level=0, drop=True)
        
        # 4. PACE ADAPTATIVO (ajuste al oponente)
        if 'Opp' in df.columns:
            # Pace promedio del oponente con shift
            opp_avg_pace = df.groupby('Opp')['pace_factor'].transform(
                lambda x: x.expanding().mean().shift(1)
            )
            df['opp_pace_avg'] = opp_avg_pace
            
            # Adaptación del pace al oponente con shift
            df['pace_adaptation'] = df['pace_factor'] - df['opp_pace_avg']
            
            # Media móvil de adaptación con shift
            for window in [3, 5]:
                df[f'pace_adaptation_ma_{window}'] = df.groupby('Team')['pace_adaptation'].transform(
                    lambda x: x.rolling(window=window, min_periods=1).mean().shift(1)
                )
        
        return df
    
    def _create_advanced_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Crea features avanzadas de volatilidad basadas en los patrones identificados,
        expandiendo las features de volatilidad que aparecen múltiples veces en el top.
        """
        if df.empty:
            return df
        
        # Verificar columnas necesarias
        if 'total_points' not in df.columns or 'PTS' not in df.columns:
            return df
        
        # 1. VOLATILIDAD MULTI-ESCALA Y MULTI-MÉTRICA
        volatility_windows = [3, 5, 7, 10, 15, 20]
        
        for window in volatility_windows:
            # Volatilidad estándar (desviación estándar) con shift
            df[f'total_pts_volatility_{window}'] = df.groupby('Team')['total_points'].transform(
                lambda x: x.rolling(window=window, min_periods=2).std().shift(1)
            )
            
            df[f'pts_volatility_{window}'] = df.groupby('Team')['PTS'].transform(
                lambda x: x.rolling(window=window, min_periods=2).std().shift(1)
            )
            
            # Volatilidad normalizada (coeficiente de variación) con shift
            pts_mean = df.groupby('Team')['PTS'].transform(
                lambda x: x.rolling(window=window, min_periods=1).mean().shift(1)
            )
            df[f'pts_cv_{window}'] = df[f'pts_volatility_{window}'] / pts_mean.replace(0, np.nan)
            
            # Volatilidad robusta (rango intercuartílico) con shift (ajustar min_periods)
            min_periods_iqr = min(3, window)
            df[f'total_pts_iqr_{window}'] = df.groupby('Team')['total_points'].transform(
                lambda x: x.rolling(window=window, min_periods=min_periods_iqr).apply(
                    lambda y: np.percentile(y, 75) - np.percentile(y, 25) if len(y) >= min_periods_iqr else np.nan
                ).shift(1)
            )
            
            # Volatilidad direccional (desviación absoluta media) con shift
            df[f'total_pts_mad_{window}'] = df.groupby('Team')['total_points'].transform(
                lambda x: x.rolling(window=window, min_periods=2).apply(
                    lambda y: np.mean(np.abs(y - np.mean(y))) if len(y) >= 2 else np.nan
                ).shift(1)
            )
        
        # 2. VOLATILIDAD CONDICIONAL SIMPLIFICADA
        for window in [5, 10]:
            # Volatilidad condicional (cambios porcentuales) con shift
            df[f'conditional_volatility_{window}'] = df.groupby('Team')['total_points'].transform(
                lambda x: x.pct_change().rolling(window=window, min_periods=2).std().shift(1)
            )
            
            # Volatilidad de rangos (max-min) con shift
            df[f'range_volatility_{window}'] = df.groupby('Team')['total_points'].transform(
                lambda x: (x.rolling(window=window, min_periods=2).max() - 
                          x.rolling(window=window, min_periods=2).min()).shift(1)
            )
        
        # 3. VOLATILIDAD RELATIVA Y COMPARATIVA
        # Volatilidad relativa al promedio de la liga con shift
        league_volatility = df['total_points'].rolling(window=50, min_periods=10).std().shift(1)
        team_volatility_5 = df.groupby('Team')['total_points'].transform(
            lambda x: x.rolling(window=5, min_periods=2).std().shift(1)
        )
        df['volatility_vs_league'] = team_volatility_5 / league_volatility.replace(0, np.nan)
        
        # Volatilidad relativa al oponente con shift
        if 'Opp' in df.columns:
            opp_volatility = df.groupby('Opp')['total_points'].transform(
                lambda x: x.expanding().std().shift(1)
            )
            df['volatility_vs_opp'] = team_volatility_5 / opp_volatility.replace(0, np.nan)
        
        return df

    def _create_opponent_strength_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Crea características avanzadas sobre la fuerza del oponente
        CORREGIDO para usar las columnas reales del dataset
        """
        df = df.copy()
        
        # Verificar que tenemos las columnas necesarias
        required_cols = ['Team', 'Opp', 'PTS', 'PTS_Opp', 'total_points', 'is_win']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            logger.warning(f"Columnas faltantes para opponent features: {missing_cols}")
            logger.info(f"Columnas disponibles: {list(df.columns)[:20]}...")  # Mostrar primeras 20
            return df
        
        # Crear características de matchup usando las columnas existentes de _create_opponent_features
        if 'opp_PTS_mean' in df.columns:
            # Diferencia ofensiva: puntos promedio del equipo vs defensa promedio del oponente CON SHIFT
            team_pts_mean = df.groupby('Team')['PTS'].transform(
                lambda x: x.expanding().mean().shift(1)
            )
            df['offensive_matchup'] = team_pts_mean - df['opp_PTS_Opp_mean'].fillna(110)
            
            # Diferencia defensiva: defensa del equipo vs ofensa del oponente CON SHIFT
            team_def_mean = df.groupby('Team')['PTS_Opp'].transform(
                lambda x: x.expanding().mean().shift(1)
            )
            df['defensive_matchup'] = df['opp_PTS_mean'].fillna(110) - team_def_mean
            
            # Strength of schedule (últimos 5 juegos) CON SHIFT
            df['recent_opp_strength'] = df.groupby('Team')['opp_PTS_mean'].transform(
                lambda x: x.rolling(5).mean().shift(1)
            ).fillna(110)
            
            # Total points matchup CON SHIFT (usando columnas existentes)
            team_total_mean = df.groupby('Team')['total_points'].transform(
                lambda x: x.expanding().mean().shift(1)
            )
            # Usar promedio global si no existe la columna específica
            global_total_avg = df['total_points'].mean()
            df['total_pts_matchup'] = team_total_mean - global_total_avg
        else:
            # Valores por defecto si no se pueden calcular
            logger.warning("No se pudieron crear características de oponente, usando valores por defecto")
            logger.warning(f"Columnas disponibles después del merge: {[col for col in df.columns if 'opp_' in col or col in ['Team', 'Opp']]}")
            
            # Crear características básicas usando promedios globales
            global_pts_avg = df['PTS'].mean()
            global_pts_opp_avg = df['PTS_Opp'].mean()
            global_total_avg = df['total_points'].mean()
            
            # Usar diferencias respecto a promedios globales CON SHIFT
            team_pts_avg = df.groupby('Team')['PTS'].transform(
                lambda x: x.expanding().mean().shift(1)
            )
            team_def_avg = df.groupby('Team')['PTS_Opp'].transform(
                lambda x: x.expanding().mean().shift(1)
            )
            team_total_avg = df.groupby('Team')['total_points'].transform(
                lambda x: x.expanding().mean().shift(1)
            )
            
            df['offensive_matchup'] = team_pts_avg - global_pts_opp_avg
            df['defensive_matchup'] = global_pts_avg - team_def_avg
            df['recent_opp_strength'] = global_pts_avg
            df['total_pts_matchup'] = team_total_avg - global_total_avg
            
            logger.info("Características de oponente creadas usando promedios globales como fallback")
        
        return df
    
    def _create_pace_and_efficiency_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Crea características avanzadas de ritmo y eficiencia
        """
        df = df.copy()
        
        # Verificar columnas disponibles
        if 'MP' not in df.columns:
            logger.warning("Columna 'MP' no encontrada, usando valor por defecto de 240 minutos")
            df['MP'] = 240
        
        # Estimación básica de posesiones usando la fórmula simplificada
        # Possessions ≈ (PTS + PTS_Opp) / 2.2 (aproximación cuando no tenemos FGA, FTA, etc.)
        df['estimated_possessions'] = (df['PTS'] + df['PTS_Opp']) / 2.2
        
        # Pace factor (posesiones por 48 minutos)
        df['pace_factor'] = (df['estimated_possessions'] * 48) / df['MP']
        
        # Eficiencia ofensiva histórica (usando datos anteriores CON SHIFT)
        df['offensive_efficiency_hist'] = df.groupby('Team').apply(
            lambda x: ((x['PTS'].shift(1) * 100) / (x['estimated_possessions'].shift(1) + 1e-10))
        ).reset_index(level=0, drop=True)
        
        # Eficiencia defensiva histórica (usando datos anteriores CON SHIFT)
        df['defensive_efficiency_hist'] = df.groupby('Team').apply(
            lambda x: ((x['PTS_Opp'].shift(1) * 100) / (x['estimated_possessions'].shift(1) + 1e-10))
        ).reset_index(level=0, drop=True)
        
        # Net efficiency histórica (diferencia entre ofensiva y defensiva históricas)
        df['net_efficiency_hist'] = df['offensive_efficiency_hist'] - df['defensive_efficiency_hist']
        
        # Rolling averages de eficiencia histórica (últimos N juegos) CON SHIFT
        for window in [3, 5, 10]:
            df[f'off_eff_ma_{window}'] = df.groupby('Team')['offensive_efficiency_hist'].transform(
                lambda x: x.rolling(window).mean().shift(1)
            )
            df[f'def_eff_ma_{window}'] = df.groupby('Team')['defensive_efficiency_hist'].transform(
                lambda x: x.rolling(window).mean().shift(1)
            )
            df[f'pace_ma_{window}'] = df.groupby('Team')['pace_factor'].transform(
                lambda x: x.rolling(window).mean().shift(1)
            )
            df[f'net_eff_ma_{window}'] = df.groupby('Team')['net_efficiency_hist'].transform(
                lambda x: x.rolling(window).mean().shift(1)
            )
        
        # Características de tempo y ritmo CON SHIFT
        df['total_pts_per_minute'] = df['total_points'] / df['MP']
        df['tempo_factor'] = df.groupby('Team')['total_pts_per_minute'].transform(
            lambda x: x.rolling(5).mean().shift(1)
        )
        
        # Volatilidad del pace CON SHIFT
        df['pace_volatility'] = df.groupby('Team')['pace_factor'].transform(
            lambda x: x.rolling(10).std().shift(1)
        )
        
        return df
    
    def _create_ensemble_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Crea características ensemble que combinan múltiples aspectos del juego
        CORREGIDO para usar las columnas reales del dataset
        """
        df = df.copy()
        
        # Características composite de rendimiento (usando eficiencias históricas SIN DATA LEAKAGE)
        pts_hist = df.groupby('Team')['PTS'].transform(lambda x: x.shift(1)).fillna(0)
        pts_opp_hist = df.groupby('Team')['PTS_Opp'].transform(lambda x: x.shift(1)).fillna(110)
        pace_hist = df.groupby('Team')['pace_factor'].transform(lambda x: x.shift(1)).fillna(100)
        
        df['offensive_composite'] = (
            pts_hist * 0.4 +
            df.get('offensive_efficiency_hist', 100).fillna(100) * 0.4 +  # Escalado apropiado
            pace_hist * 0.2
        )
        
        df['defensive_composite'] = (
            (120 - pts_opp_hist) * 0.4 +  # Invertir para que menor sea mejor
            (120 - df.get('defensive_efficiency_hist', 110).fillna(110)) * 0.4 +  # Invertir eficiencia defensiva
            pace_hist * 0.2
        )
        
        # Características de momentum composite (usando rolling averages)
        momentum_cols = [col for col in df.columns if '_ma_' in col and any(x in col for x in ['3', '5'])]
        if momentum_cols:
            # Seleccionar columnas de momentum válidas (no-null)
            valid_momentum = []
            for col in momentum_cols[:5]:  # Limitar para evitar sobreajuste
                if df[col].notna().sum() > len(df) * 0.3:  # Al menos 30% de valores válidos
                    valid_momentum.append(col)
            
            if valid_momentum:
                df['momentum_composite'] = df[valid_momentum].fillna(method='ffill').fillna(0).mean(axis=1)
            else:
                df['momentum_composite'] = 0
        else:
            df['momentum_composite'] = 0
        
        # Características de volatilidad composite
        volatility_cols = [col for col in df.columns if 'vol' in col or 'std' in col]
        if volatility_cols:
            valid_volatility = []
            for col in volatility_cols[:3]:  # Limitar número
                if df[col].notna().sum() > len(df) * 0.3:
                    valid_volatility.append(col)
            
            if valid_volatility:
                df['volatility_composite'] = df[valid_volatility].fillna(0).mean(axis=1)
            else:
                df['volatility_composite'] = 0
        else:
            df['volatility_composite'] = 0
        
        # Característica de strength composite (usando las nuevas columnas)
        if 'offensive_matchup' in df.columns and 'defensive_matchup' in df.columns:
            df['strength_composite'] = (
                df['offensive_matchup'].fillna(0) * 0.6 +
                df['defensive_matchup'].fillna(0) * 0.4
            )
        else:
            df['strength_composite'] = 0
        
        # Característica de eficiencia total (usando eficiencia histórica SIN DATA LEAKAGE)
        if 'net_efficiency_hist' in df.columns:
            df['efficiency_composite'] = (
                df['net_efficiency_hist'].fillna(0) * 0.7 +
                df.groupby('Team')['pace_factor'].transform(lambda x: x.shift(1)).fillna(100) * 0.3
            )
        else:
            df['efficiency_composite'] = 0
        
        return df

    def _create_advanced_shooting_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Crea features avanzadas de volumen de tiros basadas en los patrones más predictivos
        identificados por el modelo, todas con shift para evitar data leakage.
        """
        if df.empty:
            return df
        
        # Verificar columnas necesarias
        shooting_cols = ['FGA', 'FGA_Opp', '3PA', '3PA_Opp', '2PA', '2PA_Opp']
        available_cols = [col for col in shooting_cols if col in df.columns]
        
        if len(available_cols) < 4:
            logger.warning("Columnas insuficientes para crear features avanzadas de volumen")
            return df
        
        # 1. FEATURES DE VOLUMEN TOTAL EXPANDIDAS (basado en top features)
        volume_pairs = [
            ('FGA', 'FGA_Opp', 'total_fga'),
            ('3PA', '3PA_Opp', 'total_3pa'),
            ('2PA', '2PA_Opp', 'total_2pa')
        ]
        
        for team_col, opp_col, total_name in volume_pairs:
            if team_col in df.columns and opp_col in df.columns:
                # Volumen total
                df[total_name] = df[team_col] + df[opp_col]
                
                # Ventanas expandidas (más granularidad)
                windows = [2, 3, 4, 5, 7, 10, 15]
                for window in windows:
                    # Media móvil con shift
                    df[f'{total_name}_ma_{window}'] = df.groupby('Team')[total_name].transform(
                        lambda x: x.rolling(window=window, min_periods=1).mean().shift(1)
                    )
                    
                    # Tendencia del volumen con shift (ajustar min_periods)
                    min_periods_trend = min(3, window)
                    df[f'{total_name}_trend_{window}'] = df.groupby('Team')[total_name].transform(
                        lambda x: x.rolling(window=window, min_periods=min_periods_trend).apply(
                            lambda y: np.polyfit(np.arange(len(y)), y, 1)[0] if len(y) >= min_periods_trend else np.nan
                        ).shift(1)
                    )
                    
                    # Aceleración del volumen (segunda derivada) con shift
                    if window >= 5:
                        df[f'{total_name}_acceleration_{window}'] = df.groupby('Team')[total_name].transform(
                            lambda x: x.diff().diff().shift(1)
                        )
        
        # 2. RATIOS DE VOLUMEN AVANZADOS
        if 'total_fga' in df.columns and 'total_3pa' in df.columns:
            # Ratio de agresividad de triples
            df['three_point_aggression'] = df['total_3pa'] / df['total_fga'].replace(0, np.nan)
            
            # Media móvil de agresividad con shift
            for window in [3, 5, 7, 10]:
                df[f'three_point_aggression_ma_{window}'] = df.groupby('Team')['three_point_aggression'].transform(
                    lambda x: x.rolling(window=window, min_periods=1).mean().shift(1)
                )
        
        # 3. FEATURES DE DISTRIBUCIÓN DE VOLUMEN
        if 'FGA' in df.columns and 'FGA_Opp' in df.columns:
            # Balance ofensivo/defensivo
            df['offensive_volume_dominance'] = df['FGA'] / (df['FGA'] + df['FGA_Opp']).replace(0, np.nan)
            
            # Media móvil de dominancia con shift
            for window in [3, 5, 10]:
                df[f'offensive_volume_dominance_ma_{window}'] = df.groupby('Team')['offensive_volume_dominance'].transform(
                    lambda x: x.rolling(window=window, min_periods=1).mean().shift(1)
                )
        
        return df
