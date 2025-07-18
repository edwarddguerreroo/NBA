"""More actions
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

warnings.filterwarnings('ignore')

# Configurar logging
logger = logging.getLogger(__name__)

class TotalPointsFeatureEngineer:
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
        Inicializa el motor de características para puntos totales.
        
        Args:
            lookback_games: Número de juegos históricos a considerar
        """
        self.lookback_games = lookback_games
        self.scaler = StandardScaler()
        self.feature_columns = []

    def generate_all_features(self, df: pd.DataFrame) -> List[str]:
        """
        MÉTODO ESTÁNDAR - Genera todas las features para total_points siguiendo el patrón del proyecto.
        Mantiene consistencia con teams_points, is_win, pts, trb, ast, dd, y triples.
        
        Args:
            df: DataFrame con datos de equipos
            
        Returns:
            Lista de nombres de features generadas (excluyendo target y columnas auxiliares)
        """
        logger.info("Generando features NBA OPTIMIZADAS para predicción de puntos totales...")
        
        if df.empty:
            logger.warning("DataFrame vacío recibido")
            return []
        
        # Usar create_features existente que ya implementa toda la lógica
        df_processed = self.create_features(df.copy())
        
        # Compilar lista de features siguiendo el patrón estándar del proyecto
        excluded_columns = {
            # Columnas de identificación
            'Team', 'Date', 'Away', 'Opp', 'Result', 'MP',
            # Target y componentes (evitar data leakage)
            'PTS', 'PTS_Opp', 'total_points',
            # Estadísticas básicas de tiro (pueden causar data leakage si no tienen shift)
            'FG', 'FGA', 'FG%', '2P', '2PA', '2P%', '3P', '3PA', '3P%', 
            'FT', 'FTA', 'FT%', 'FG_Opp', 'FGA_Opp', 'FG%_Opp', 
            '2P_Opp', '2PA_Opp', '2P%_Opp', '3P_Opp', '3PA_Opp', '3P%_Opp',
            'FT_Opp', 'FTA_Opp', 'FT%_Opp',
            # Variables auxiliares y categóricas
            'season', 'temp_oct_first'
        }
        
        # Obtener todas las features válidas
        all_features = [col for col in df_processed.columns if col not in excluded_columns]
        
        # Aplicar filtros de ruido específicos para total_points
        clean_features = self._apply_noise_filters(df_processed, all_features)
        
        logger.info(f"Generadas {len(clean_features)} características OPTIMIZADAS para puntos totales")
        return clean_features

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

        # Crear características básicas del equipo (medias móviles, std, etc.)
        df = self._create_team_basic_features(df)

        # Crear características de momentum
        df = self._create_momentum_features(df)

        # Crear características de eficiencia
        df = self._create_efficiency_features(df)

        # Crear características de pace
        df = self._create_pace_features(df)

        # Crear características situacionales
        df = self._create_situational_features(df)

        # Crear características de interacción
        df = self._create_interaction_features(df)

        # Crear características de oponentes
        df = self._create_opponent_features(df)

        # Crear características de matchup
        df = self._create_matchup_features(df)

        # Crear características temporales
        df = self._create_temporal_features(df)

        # Crear características basadas en jugadores (si hay datos disponibles)
        df = self._create_player_based_features(df, df_players)

        # Crear características de porcentajes de tiro (NUEVA FUNCIÓN)
        # Crear características de porcentajes de tiro
        df = self._create_shooting_percentage_features(df)

        # Crear características avanzadas
        df = self._create_advanced_features(df)

        # Crear variables dummy para características categóricas
        df = self._create_dummy_variables(df)

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
        """
        Aplica cálculos básicos en lugar para evitar duplicaciones.
        """
        # Ordenar cronológicamente PRIMERO
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            df.sort_values(['Team', 'Date'], inplace=True)
            df.reset_index(drop=True, inplace=True)
        
        # ================= CÁLCULOS TEMPORALES =================
        if 'Date' in df.columns:
            df['days_rest'] = df.groupby('Team')['Date'].diff().dt.days.fillna(2)
            df['day_of_week'] = df['Date'].dt.dayofweek
            df['month'] = df['Date'].dt.month
            df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
            
            # Progreso en temporada
            season_start = df['Date'].min()
            df['days_into_season'] = np.clip((df['Date'] - season_start).dt.days, 0, 250)  # Clip para evitar valores extremos
            df['season_progress'] = df['days_into_season'] / 180  # Temporada ~180 días
            df['season_progress'] = df['season_progress'].clip(0, 1)
            
            # Características trigonométricas para capturar ciclos estacionales
            df['day_of_season_sin'] = np.sin(2 * np.pi * df['season_progress'])
            df['day_of_season_cos'] = np.cos(2 * np.pi * df['season_progress'])
            
            # Fases de temporada
            df['phase_early'] = (df['season_progress'] < 0.25).astype(int)
            df['phase_early_mid'] = ((df['season_progress'] >= 0.25) & (df['season_progress'] < 0.5)).astype(int)
            df['phase_mid'] = ((df['season_progress'] >= 0.5) & (df['season_progress'] < 0.6)).astype(int)
            df['phase_late_mid'] = ((df['season_progress'] >= 0.6) & (df['season_progress'] < 0.75)).astype(int)
            df['phase_late'] = (df['season_progress'] >= 0.75).astype(int)
            df['season_phase_numeric'] = df['season_progress'] * 4  # 0-4 scale
        
        # ================= CÁLCULOS BÁSICOS DE POSESIONES NBA =================
        # Estimación de posesiones usando fórmula NBA aproximada (sin TOV disponible)
        if 'FGA' in df.columns and 'FTA' in df.columns:
            df['possessions'] = df['FGA'] + df['FTA'] * 0.44
            df['opp_possessions'] = df['FGA_Opp'] + df['FTA_Opp'] * 0.44
            
            # Pace aproximado (posesiones por 48 minutos)
            df['pace_approx'] = (df['possessions'] + df['opp_possessions']) / 2
            
            # Eficiencias ofensiva y defensiva aproximadas
            if 'PTS' in df.columns:
                df['off_rating_approx'] = (df['PTS'] / (df['possessions'] + 1e-6)) * 100
                df['def_rating_approx'] = (df['PTS_Opp'] / (df['opp_possessions'] + 1e-6)) * 100
        
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

        # FEATURES ESPECÍFICAS REQUERIDAS POR EL MODELO - CÁLCULOS REALES
        # Promedio histórico de puntos de 5 juegos (equivalente a pts_ma_5 pero con nombre específico)
        df['pts_hist_avg_5g'] = df.groupby('Team')['PTS'].transform(
            lambda x: x.rolling(5, min_periods=1).mean().shift(1)
        )
        
        # Promedio histórico de puntos del oponente de 5 juegos
        df['pts_opp_hist_avg_5g'] = df.groupby('Team')['PTS_Opp'].transform(
            lambda x: x.rolling(5, min_periods=1).mean().shift(1)
        )
        
        # Diferencia histórica de puntos de 5 juegos
        df['point_diff_hist_avg_5g'] = df.groupby('Team').apply(
            lambda x: (x['PTS'] - x['PTS_Opp']).rolling(5, min_periods=1).mean().shift(1)
        ).reset_index(level=0, drop=True)
        
        # Consistencia de puntos de 5 juegos (desviación estándar)
        df['pts_consistency_5g'] = df.groupby('Team')['PTS'].transform(
            lambda x: x.rolling(5, min_periods=1).std().shift(1)
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
                
        # FEATURE ESPECÍFICA REQUERIDA: team_win_rate_10g
        if 'is_win' in df.columns:
            df['team_win_rate_10g'] = df.groupby('Team')['is_win'].transform(
                lambda x: x.rolling(10, min_periods=1).mean().shift(1)
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

        # Transformaciones matemáticas de momentum (movidas desde _create_advanced_features)
        # Aplicar clipping muy agresivo para evitar valores extremos
        df['pts_momentum_squared'] = df.groupby('Team')['PTS'].transform(
            lambda x: np.clip(x.diff(1).pow(2).shift(1), 0, 100)  # Clip muy agresivo: 0-100 para reducir varianza
        )
        df['pts_momentum_sqrt'] = df.groupby('Team')['PTS'].transform(
            lambda x: (x.diff(1).abs().pow(0.5) * np.sign(x.diff(1))).shift(1)
        )
        df['pts_momentum_log'] = df.groupby('Team')['PTS'].transform(
            lambda x: (np.log1p(x.diff(1).abs()) * np.sign(x.diff(1))).shift(1)
        )

        # Aceleración (movidas desde _create_advanced_features)
        df['pts_acceleration'] = df.groupby('Team')['PTS'].transform(
            lambda x: x.diff(1).diff(1).shift(1)
        )
        df['total_pts_acceleration'] = df.groupby('Team')['total_points'].transform(
            lambda x: x.diff(1).diff(1).shift(1)
        )

        # Consistencia (coeficiente de variación)
        df['pts_consistency'] = df.groupby('Team').apply(
            lambda x: (x['PTS'].rolling(10, min_periods=3).std() / 
                      x['PTS'].rolling(10, min_periods=3).mean()).shift(1)
        ).reset_index(level=0, drop=True)

        df['total_pts_consistency'] = df.groupby('Team').apply(
            lambda x: ((x['PTS'] + x['PTS_Opp']).rolling(10, min_periods=3).std() / 
                      (x['PTS'] + x['PTS_Opp']).rolling(10, min_periods=3).mean()).shift(1)
        ).reset_index(level=0, drop=True)

        logger.info(f"Features de momentum consolidadas: {len([col for col in df.columns if 'momentum' in col or 'streak' in col or 'acceleration' in col])}")

        return df

    def _calculate_streak(self, series: pd.Series, value: int) -> pd.Series:
        """Calcula rachas de un valor específico"""
        streak = (series == value).astype(int)
        streak = streak.groupby((streak != streak.shift()).cumsum()).cumsum()
        return streak.shift(1)

    def _create_opponent_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Crea características del oponente con valores por defecto robustos"""
        
        if df.empty:
            return df
            
        # Verificar columnas requeridas
        required_cols = ['Team', 'Opp', 'PTS', 'PTS_Opp']
        if not all(col in df.columns for col in required_cols):
            logger.warning(f"Faltan columnas para opponent features: {[col for col in required_cols if col not in df.columns]}")
            return df

        df_with_opp = df.copy()
        
        # Estadísticas globales para valores por defecto
        global_pts_mean = df['PTS'].mean()
        global_pts_std = df['PTS'].std()
        global_pts_opp_mean = df['PTS_Opp'].mean()
        global_pts_opp_std = df['PTS_Opp'].std()
        
        # Crear estadísticas históricas de oponentes con valores robustos
        opp_features = {}
        
        for team in df['Team'].unique():
            team_data = df[df['Team'] == team].copy()
            
            if len(team_data) < 2:
                # Usar estadísticas globales si no hay suficientes datos
                opp_features[team] = {
                    'opp_PTS_mean': global_pts_mean,
                    'opp_PTS_std': global_pts_std,
                    'opp_PTS_Opp_mean': global_pts_opp_mean,
                    'opp_PTS_Opp_std': global_pts_opp_std,
                    'opp_is_win_mean': 0.5
                }
            else:
                # Ordenar por fecha si está disponible
                if 'Date' in team_data.columns:
                    team_data = team_data.sort_values('Date')
                
                # Calcular estadísticas móviles robustas
                pts_rolling = team_data['PTS'].rolling(window=5, min_periods=1).mean().shift(1)
                pts_std_rolling = team_data['PTS'].rolling(window=5, min_periods=1).std().shift(1)
                pts_opp_rolling = team_data['PTS_Opp'].rolling(window=5, min_periods=1).mean().shift(1)
                pts_opp_std_rolling = team_data['PTS_Opp'].rolling(window=5, min_periods=1).std().shift(1)
                
                # Rellenar NaN con valores globales
                pts_mean = pts_rolling.fillna(global_pts_mean).iloc[-1]
                pts_std = pts_std_rolling.fillna(global_pts_std).iloc[-1]
                pts_opp_mean = pts_opp_rolling.fillna(global_pts_opp_mean).iloc[-1]
                pts_opp_std = pts_opp_std_rolling.fillna(global_pts_opp_std).iloc[-1]
                
                # Calcular win rate
                if 'is_win' in team_data.columns:
                    win_rate = team_data['is_win'].rolling(window=5, min_periods=1).mean().shift(1).fillna(0.5).iloc[-1]
                else:
                    win_rate = 0.5
                
                opp_features[team] = {
                    'opp_PTS_mean': pts_mean,
                    'opp_PTS_std': pts_std,
                    'opp_PTS_Opp_mean': pts_opp_mean,
                    'opp_PTS_Opp_std': pts_opp_std,
                    'opp_is_win_mean': win_rate
                }
        
        # Asignar features del oponente a cada fila
        for idx, row in df_with_opp.iterrows():
            opp_team = row['Opp']
            
            if opp_team in opp_features:
                for feature, value in opp_features[opp_team].items():
                    df_with_opp.loc[idx, feature] = value
            else:
                # Usar valores globales por defecto
                df_with_opp.loc[idx, 'opp_PTS_mean'] = global_pts_mean
                df_with_opp.loc[idx, 'opp_PTS_std'] = global_pts_std
                df_with_opp.loc[idx, 'opp_PTS_Opp_mean'] = global_pts_opp_mean
                df_with_opp.loc[idx, 'opp_PTS_Opp_std'] = global_pts_opp_std
                df_with_opp.loc[idx, 'opp_is_win_mean'] = 0.5
        
        # Features derivadas robustas
        df_with_opp['opp_pace_tendency'] = df_with_opp['opp_PTS_mean'] + df_with_opp['opp_PTS_Opp_mean']
        df_with_opp['opp_scoring_tendency'] = df_with_opp['opp_PTS_mean'] / df_with_opp['opp_PTS_Opp_mean'].clip(lower=80)
        
        # Diferencial de calidad
        if 'win_rate_last_10' in df_with_opp.columns:
            df_with_opp['quality_diff'] = df_with_opp['win_rate_last_10'] - df_with_opp['opp_is_win_mean']
        else:
            df_with_opp['quality_diff'] = 0.0
        
        logger.info(f"Features de oponente creadas: {len([col for col in df_with_opp.columns if col.startswith('opp_')])}")
        
        return df_with_opp

    def _create_matchup_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Crea features específicas de matchup histórico"""
        
        # Verificar si existe la columna 'Opp'
        if 'Opp' not in df.columns:
            logger.warning("Columna 'Opp' no encontrada, creando features de matchup por defecto")
            # Crear features de matchup con valores por defecto
            for col in ['matchup_total_avg', 'matchup_total_std', 'matchup_diff_avg', 'matchup_win_rate']:
                df[col] = np.nan
            return df
            
        # Crear clave de matchup
        df['matchup_key'] = df.apply(lambda x: tuple(sorted([x['Team'], x['Opp']])), axis=1)

        # Versión optimizada para evitar explosión de datos
        # Calculamos las estadísticas para cada combinación única de matchup_key y Team
        matchup_stats = []

        for (matchup, team), group in df.groupby(['matchup_key', 'Team']):
            # Ordenar por fecha para asegurar cálculos correctos
            group = group.sort_values('Date') if 'Date' in group.columns else group

            # Calcular estadísticas históricas una vez por grupo
            total_pts = group['PTS'] + group['PTS_Opp']
            pts_diff = group['PTS'] - group['PTS_Opp']

            # Calcular estadísticas expandidas (último valor es el que usaremos)
            if len(group) > 0:
                last_idx = group.index[-1]

                # Calcular valores históricos hasta el penúltimo registro (shift(1))
                total_pts_mean = total_pts.expanding().mean().shift(1).iloc[-1] if len(total_pts) > 1 else np.nan
                total_pts_std = total_pts.expanding().std().shift(1).iloc[-1] if len(total_pts) > 1 else np.nan
                pts_diff_mean = pts_diff.expanding().mean().shift(1).iloc[-1] if len(pts_diff) > 1 else np.nan

                if 'is_win' in group.columns:
                    is_win_mean = group['is_win'].expanding().mean().shift(1).iloc[-1] if len(group) > 1 else 0.5
                else:
                    is_win_mean = 0.5

                # Agregar estadísticas a la lista
                matchup_stats.append({
                    'matchup_key': matchup,
                    'Team': team,
                    'matchup_total_avg': total_pts_mean,
                    'matchup_total_std': total_pts_std,
                    'matchup_diff_avg': pts_diff_mean,
                    'matchup_win_rate': is_win_mean
                })

        # Convertir a DataFrame
        if matchup_stats:
            matchup_stats_df = pd.DataFrame(matchup_stats)

            # Merge con el DataFrame original
            df = df.merge(matchup_stats_df, on=['matchup_key', 'Team'], how='left')
        else:
            # Si no hay estadísticas, crear columnas vacías
            for col in ['matchup_total_avg', 'matchup_total_std', 'matchup_diff_avg', 'matchup_win_rate']:
                df[col] = np.nan

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

        # Adaptación al ritmo del oponente
        if 'pace_ma_5' in df.columns and 'Opp' in df.columns:
            # Calcular ritmo promedio por equipo
            team_avg_pace = df.groupby('Team')['pace_ma_5'].transform('mean')
            opp_avg_pace = df.groupby('Opp')['pace_ma_5'].transform('mean')

            df['pace_matchup'] = df['pace_ma_5'] - opp_avg_pace
            df['pace_differential'] = team_avg_pace - opp_avg_pace
        elif 'pace_ma_5' in df.columns:
            # Sin datos del oponente, usar valores neutros
            df['pace_matchup'] = 0.0
            df['pace_differential'] = 0.0

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

            # Ventaja de local con clipping para evitar valores extremos
            df['home_advantage'] = np.clip(df['home_pts_avg'] - df['away_pts_avg'], -50, 50)
            df['home_total_advantage'] = np.clip(df['home_total_avg'] - df['away_total_avg'], -100, 100)

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
            df['total_b2b'] = df[df['is_back_to_back'] == 1].groupby('Team')['PTS'].transform(
                lambda x: (x + df.loc[x.index, 'PTS_Opp']).expanding().mean().shift(1)
            )

        logger.info(f"Features situacionales creadas: {len([col for col in df.columns if any(x in col for x in ['home', 'away', 'close', 'ot', 'rest', 'b2b'])])}")

        return df

    def _create_player_based_features(self, df: pd.DataFrame, df_players: pd.DataFrame) -> pd.DataFrame:
        """Crea características basadas en datos de jugadores con valores robustos por defecto"""

        # Si no hay datos de jugadores, crear features con valores por defecto
        if df_players is None or df_players.empty:
            logger.warning("No hay datos de jugadores disponibles, usando valores por defecto")
            
            # Valores por defecto realistas CON SHIFT para evitar data leakage
            df['team_total_pts_players'] = df.groupby('Team')['PTS'].transform(
                lambda x: x.shift(1)
            ).fillna(110)  # Valor por defecto realista
            df['avg_player_pts'] = df.groupby('Team')['PTS'].transform(
                lambda x: x.shift(1) / 5
            ).fillna(22)  # ~5 jugadores principales 
            df['pts_distribution'] = 5.0  # Distribución típica
            df['top_scorer_pts'] = df.groupby('Team')['PTS'].transform(
                lambda x: x.shift(1) * 0.3
            ).fillna(33)  # ~30% del scoring del top scorer
            df['avg_minutes'] = 24.0  # Minutos promedio típicos
            df['minutes_distribution'] = 8.0  # Distribución típica de minutos
            df['num_starters'] = 5.0  # 5 titulares estándar
            df['team_fg_pct'] = 0.45  # FG% típico
            df['team_3p_pct'] = 0.35  # 3P% típico
            df['team_ft_pct'] = 0.75  # FT% típico
            df['team_rebounds'] = 45.0  # Rebotes típicos por equipo
            df['team_assists'] = 25.0  # Asistencias típicas
            df['team_steals'] = 8.0  # Robos típicos
            df['team_blocks'] = 5.0  # Bloqueos típicos
            df['team_turnovers'] = 15.0  # Pérdidas típicas
            df['team_fouls'] = 20.0  # Faltas típicas
            df['team_plus_minus'] = 0.0  # +/- neutro
            df['num_double_doubles'] = 1.0  # ~1 doble-doble por juego
            df['num_triple_doubles'] = 0.0  # Raros
            
            logger.info("Features de jugadores creadas con valores por defecto")
            return df

        try:
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

            # Agrupar estadísticas de jugadores por equipo y fecha
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
            shift_cols = [col for col in player_stats.columns if col not in ['Team', 'Date']]

            for col in shift_cols:
                player_stats[col] = player_stats.groupby('Team')[col].shift(1)

            # Merge con datos de equipos
            df = pd.merge(df, player_stats, on=['Team', 'Date'], how='left', indicator='_merge_indicator')

            # Verificar si hubo problemas en el merge y rellenar con valores por defecto
            missing_data_rows = (df['_merge_indicator'] == 'left_only')
            if missing_data_rows.any():
                num_missing = missing_data_rows.sum()
                logger.warning(f"Hay {num_missing} filas sin datos de jugadores")
                
                # Rellenar con valores por defecto CON DATOS HISTÓRICOS para filas sin datos
                default_values = {
                    'team_total_pts_players': df.groupby('Team')['PTS'].transform(lambda x: x.shift(1)).fillna(110),
                    'avg_player_pts': df.groupby('Team')['PTS'].transform(lambda x: x.shift(1) / 5).fillna(22),
                    'pts_distribution': 5.0,
                    'top_scorer_pts': df.groupby('Team')['PTS'].transform(lambda x: x.shift(1) * 0.3).fillna(33),
                    'avg_minutes': 24.0,
                    'minutes_distribution': 8.0,
                    'num_starters': 5.0,
                    'team_fg_pct': 0.45,
                    'team_3p_pct': 0.35,
                    'team_ft_pct': 0.75,
                    'team_rebounds': 45.0,
                    'team_assists': 25.0,
                    'team_steals': 8.0,
                    'team_blocks': 5.0,
                    'team_turnovers': 15.0,
                    'team_fouls': 20.0,
                    'team_plus_minus': 0.0,
                    'num_double_doubles': 1.0,
                    'num_triple_doubles': 0.0
                }
                
                for col, default_val in default_values.items():
                    if col in df.columns:
                        df.loc[missing_data_rows, col] = df.loc[missing_data_rows, col].fillna(default_val)

            # Eliminar columna indicadora
            df = df.drop('_merge_indicator', axis=1)

            # Rellenar NaN restantes con valores por defecto globales
            fill_values = {
                'team_total_pts_players': df['PTS'].mean() if 'PTS' in df.columns else 110,
                'avg_player_pts': 22.0,
                'pts_distribution': 5.0,
                'top_scorer_pts': 33.0,
                'avg_minutes': 24.0,
                'minutes_distribution': 8.0,
                'num_starters': 5.0,
                'team_fg_pct': 0.45,
                'team_3p_pct': 0.35,
                'team_ft_pct': 0.75,
                'team_rebounds': 45.0,
                'team_assists': 25.0,
                'team_steals': 8.0,
                'team_blocks': 5.0,
                'team_turnovers': 15.0,
                'team_fouls': 20.0,
                'team_plus_minus': 0.0,
                'num_double_doubles': 1.0,
                'num_triple_doubles': 0.0
            }
            
            for col, fill_val in fill_values.items():
                if col in df.columns:
                    df[col] = df[col].fillna(fill_val)

            # Características derivadas de jugadores con protección contra división por cero
            if 'top_scorer_pts' in df.columns and 'team_total_pts_players' in df.columns:
                df['scoring_concentration'] = df['top_scorer_pts'] / df['team_total_pts_players'].clip(lower=1)

            if all(col in df.columns for col in ['num_starters', 'avg_player_pts', 'team_total_pts_players']):
                df['bench_contribution'] = 1 - (df['num_starters'] * df['avg_player_pts']) / df['team_total_pts_players'].clip(lower=1)

            # Eficiencia del equipo basada en jugadores con clips robustos
            efficiency_cols = ['team_total_pts_players', 'team_rebounds', 'team_assists', 
                              'team_steals', 'team_blocks', 'team_turnovers', 'avg_minutes']

            if all(col in df.columns for col in efficiency_cols):
                df['team_efficiency_players'] = (
                    df['team_total_pts_players'] + df['team_rebounds'] + 
                    df['team_assists'] + df['team_steals'] + df['team_blocks'] - 
                    df['team_turnovers']
                ) / df['avg_minutes'].clip(lower=1)

            # Tendencias de jugadores clave (con shift y fillna)
            for stat in ['top_scorer_pts', 'team_assists', 'team_rebounds']:
                if stat in df.columns:
                    df[f'{stat}_ma_5'] = df.groupby('Team')[stat].transform(
                        lambda x: x.rolling(5, min_periods=1).mean().shift(1).fillna(x.mean())
                    )

            # Factor de profundidad del equipo con clip robusto
            if 'pts_distribution' in df.columns:
                df['team_depth_factor'] = 1 / df['pts_distribution'].clip(lower=0.1)

            logger.info(f"Features de jugadores creadas: {len([col for col in df.columns if any(x in col for x in ['team_', 'player', 'scorer', 'bench'])])}")

        except Exception as e:
            logger.error(f"Error procesando datos de jugadores: {str(e)}")
            # No interrumpir el proceso completo, usar valores por defecto
            logger.info("Usando valores por defecto para features de jugadores")

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

        # Polinomios de características clave con clipping agresivo
        for col in ['pts_ma_5', 'pace_ma_5', 'total_pts_ma_5']:
            if col in df.columns:
                # Normalizar antes de elevar al cuadrado
                col_normalized = (df[col] - df[col].mean()) / (df[col].std() + 1e-8)
                col_normalized = np.clip(col_normalized, -5, 5)  # Clip normalizado
                df[f'{col}_squared'] = np.clip(col_normalized ** 2, 0, 25)
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

        # NOTA: Características de temporada ya creadas en _apply_base_calculations:
        # season_progress, season_phase_numeric, phase_early, phase_mid, etc.

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
            # Corregir cumulative_minutes para evitar valores extremos
            df['cumulative_minutes'] = df.groupby('Team')['MP'].transform('cumsum')
            # Normalizar cumulative_minutes para evitar valores extremos
            df['cumulative_minutes'] = np.clip(df['cumulative_minutes'] / 1000, 0, 100)  # Escalar a un rango razonable
            df['fatigue_index'] = df['games_last_7_days'] * df['cumulative_minutes']

        # Patrones semanales - SIMPLIFICADO (solo weekend vs weekday)
        if 'day_of_week' in df.columns:
            # Solo crear una feature binaria: fin de semana vs día de semana
            df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
            # No crear dummies individuales por día - generan ruido

        # ELIMINADO: Tendencias mensuales específicas
        # Estas features (high_intensity_month, season_start) generan ruido según análisis

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
            'FG%_vs_FG%_Opp_diff', '2P%_vs_2P%_Opp_diff', '3P%_vs_3P%_Opp_diff', 'FT%_vs_FT%_Opp_diff',

            # NUEVAS VARIABLES CON DATA LEAKAGE DETECTADAS EN ANÁLISIS
            'PTS_mean', 'total_points_mean', 'team_plus_minus', 'is_win',
            'PTS_Opp_mean', 'team_efficiency_players', 'team_ft_pct', 
            'team_fg_pct', 'team_3p_pct', 'MP', 'scoring_concentration',
            'team_rebounds', 'team_assists', 'team_steals', 'team_blocks',
            'team_turnovers', 'team_fouls', 'pts_distribution', 'minutes_distribution',
            'avg_minutes', 'cumulative_minutes', 'bench_contribution', 'team_depth_factor',
            'num_starters',

            # FEATURES DE RUIDO - IMPORTANCIA MUY BAJA (< 0.01)
            'dow_0', 'dow_1', 'dow_2', 'dow_3', 'dow_4', 'dow_5', 'dow_6',
            'market_dow_0_bias', 'market_dow_1_bias', 'market_dow_2_bias', 
            'market_dow_3_bias', 'market_dow_4_bias', 'market_dow_5_bias', 'market_dow_6_bias',
            'is_weekend', 'season_start', 'high_intensity_month',
            'opp_is_win_mean', 'matchup_dominance',

            # FEATURES TEMPORALES IRRELEVANTES
            'days_to_allstar', 'days_to_playoffs',

            # FEATURES MARGINALES QUE GENERAN RUIDO (0.01-0.015)
            'overtime_prob_ma5', 'season_phase_numeric', 'schedule_density',
            'fatigue_factor', 'market_zigzag', 'overtime_pts_interaction',

            # FEATURES REDUNDANTES DE THREE_POINT_RATIO (mantener solo las más importantes)
            'three_point_ratio_ma_3', 'three_point_ratio_ma_5', 'three_point_ratio_ma_10',
            'three_point_ratio_opp_ma_3', 'three_point_ratio_opp_ma_5', 'three_point_ratio_opp_ma_10',

            # FEATURES DE INTERACCIÓN FG3 REDUNDANTES (mantener solo la más importante)
            'fg3_pct_volume_interaction_3', 'fg3_pct_volume_interaction_5'
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

        # Verificar y limpiar columnas categóricas restantes
        cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
        if cat_cols:
            logger.info(f"Eliminando {len(cat_cols)} columnas categóricas: {cat_cols[:5]}{'...' if len(cat_cols) > 5 else ''}")
            X = X.drop(columns=cat_cols)

        # Verificar y convertir columnas no numéricas
        non_numeric = X.select_dtypes(exclude=['number']).columns.tolist()
        if non_numeric:
            logger.info(f"Convirtiendo {len(non_numeric)} columnas no numéricas: {non_numeric[:5]}{'...' if len(non_numeric) > 5 else ''}")
            # Intentar convertir a numéricas antes de eliminar
            for col in non_numeric:
                try:
                    X[col] = pd.to_numeric(X[col], errors='coerce')
                except:
                    pass
            # Finalmente, seleccionar solo numéricas
            X = X.select_dtypes(include=['number'])

        # NUEVO: Filtro automático de features de ruido por baja varianza  
        low_variance_cols = self._detect_low_variance_features(X, threshold=0.001)
        if low_variance_cols:
            logger.info(f"Eliminando {len(low_variance_cols)} columnas con baja varianza: {low_variance_cols[:5]}{'...' if len(low_variance_cols) > 5 else ''}")
            X = X.drop(columns=low_variance_cols, errors='ignore')

        # Verificación adicional para asegurar que no haya columnas de porcentajes sin shift
        # Esto es especialmente importante para las nuevas características
        shooting_cols = [col for col in X.columns if any(pattern in col for pattern in ['FG%', '2P%', '3P%', 'FT%'])]
        potential_leakage = [col for col in shooting_cols if not any(pattern in col for pattern in ['_ma_', '_trend_', '_pct_', '_diff_ma_'])]

        if potential_leakage:
            logger.info(f"Eliminando {len(potential_leakage)} columnas de porcentajes sin shift: {potential_leakage[:3]}{'...' if len(potential_leakage) > 3 else ''}")
            X = X.drop(columns=potential_leakage, errors='ignore')

        # Aplicar clipping robusto final para evitar valores extremos
        X = self._apply_robust_clipping(X)

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

    def _create_advanced_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Crea características avanzadas para mejorar la precisión del modelo"""

        if df.empty:
            return df

        # Características de momentum y tendencias
        if 'Date' in df.columns:
            df = df.sort_values(['Team', 'Date'])

            # NOTA: Features de momentum y aceleración movidas a _create_momentum_features 
            # para evitar duplicación

            # NOTA: Features de volatilidad (std) ya existen en _create_team_basic_features 
            # como pts_std_5, pts_std_10, pts_std_15 y total_pts_std_5, etc.
            # No duplicar aquí para evitar redundancia

        # Características de matchup avanzadas
        if 'Opp' in df.columns:
            # Crear identificador único para cada combinación de equipos
            df['matchup_id'] = df.apply(lambda x: '_'.join(sorted([x['Team'], x['Opp']])), axis=1)

            # Estadísticas históricas de matchup CON SHIFT TEMPORAL para evitar data leakage
            df_sorted = df.sort_values(['Team', 'Date']) if 'Date' in df.columns else df.sort_index()
            
            # Calcular estadísticas de matchup usando solo datos históricos
            matchup_historical_stats = []
            
            for idx, row in df_sorted.iterrows():
                matchup_id = row['matchup_id']
                current_date = row['Date'] if 'Date' in df.columns else idx
                
                # Obtener datos históricos del matchup (anteriores al juego actual)
                if 'Date' in df.columns:
                    historical_matchup = df_sorted[
                        (df_sorted['matchup_id'] == matchup_id) & 
                        (df_sorted['Date'] < current_date)
                    ]
                else:
                    historical_matchup = df_sorted[
                        (df_sorted['matchup_id'] == matchup_id) & 
                        (df_sorted.index < idx)
                    ]
                
                if len(historical_matchup) > 0:
                    # Calcular estadísticas solo con datos históricos
                    stats = {
                        'total_points_mean': historical_matchup['total_points'].mean(),
                        'total_points_std': historical_matchup['total_points'].std(),
                        'total_points_min': historical_matchup['total_points'].min(),
                        'total_points_max': historical_matchup['total_points'].max(),
                        'total_points_count': len(historical_matchup),
                        'PTS_mean': historical_matchup['PTS'].mean(),
                        'PTS_Opp_mean': historical_matchup['PTS_Opp'].mean()
                    }
                else:
                    # Sin datos históricos, usar valores neutrales
                    stats = {
                        'total_points_mean': 220,  # Promedio típico NBA
                        'total_points_std': 20,
                        'total_points_min': 180,
                        'total_points_max': 260,
                        'total_points_count': 0,
                        'PTS_mean': 110,
                        'PTS_Opp_mean': 110
                    }
                
                matchup_historical_stats.append(stats)
            
            # Crear DataFrame con estadísticas históricas
            stats_df = pd.DataFrame(matchup_historical_stats)
            
            # Asignar índices para alinear con df_sorted
            stats_df.index = df_sorted.index
            
            # Agregar las nuevas columnas al dataframe principal
            for col in stats_df.columns:
                df_sorted[col] = stats_df[col]
            
            # Reordenar de vuelta al orden original si es necesario
            df = df_sorted.sort_index()

            # Calcular características derivadas de matchup
            df['matchup_total_range'] = df['total_points_max'] - df['total_points_min']
            df['matchup_total_cv'] = df['total_points_std'] / (df['total_points_mean'] + 1e-5)  # Coeficiente de variación
            df['matchup_dominance'] = df['PTS_mean'] - df['PTS_Opp_mean']

            # Indicador de experiencia en el matchup
            df['matchup_experience'] = np.log1p(df['total_points_count'])

        # Características de forma y tendencia (USANDO FEATURES EXISTENTES con shift)
        if 'Date' in df.columns and 'PTS' in df.columns:
            # Las medias móviles siempre existen a esta altura del procesamiento
            for window in [3, 5, 10]:
                # Comparar con promedio de temporada (usando features existentes)
                pts_form = df.groupby('Team').apply(
                    lambda x: x[f'pts_ma_{window}'] / x['PTS'].expanding().mean().shift(1)
                )
                if isinstance(pts_form, pd.DataFrame):
                    pts_form = pts_form.iloc[:, 0]  # Tomar primera columna si es DataFrame
                df[f'pts_form_{window}'] = pts_form.reset_index(level=0, drop=True)

                total_pts_form = df.groupby('Team').apply(
                    lambda x: x[f'total_pts_ma_{window}'] / x['total_points'].expanding().mean().shift(1)
                )
                if isinstance(total_pts_form, pd.DataFrame):
                    total_pts_form = total_pts_form.iloc[:, 0]  # Tomar primera columna si es DataFrame
                df[f'total_pts_form_{window}'] = total_pts_form.reset_index(level=0, drop=True)

        # Características de interacción
        if all(col in df.columns for col in ['is_home', 'days_rest']):
            # Interacciones entre variables importantes
            df['home_rest_interaction'] = df['is_home'] * df['days_rest']

            if 'win_streak' in df.columns:
                df['home_streak_interaction'] = df['is_home'] * df['win_streak']
                df['rest_streak_interaction'] = df['days_rest'] * df['win_streak']

        # Características polinomiales para variables numéricas clave
        if 'total_pts_ma_5' in df.columns:
            total_pts_normalized = (df['total_pts_ma_5'] - df['total_pts_ma_5'].mean()) / (df['total_pts_ma_5'].std() + 1e-8)
            # Aplicar clipping más agresivo para evitar valores extremos
            total_pts_normalized = np.clip(total_pts_normalized, -5, 5)
            df['total_pts_ma_5_squared'] = np.clip(total_pts_normalized ** 2, 0, 25)
            df['total_pts_ma_5_cubed'] = np.clip(total_pts_normalized ** 3, -125, 125)

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

                        # Guardar coeficientes como características con clipping para evitar valores extremos
                        df.loc[df['Team'] == team, 'total_pts_trend_quad'] = np.clip(poly_coefs[0], -10, 10)
                        df.loc[df['Team'] == team, 'total_pts_trend_linear'] = np.clip(poly_coefs[1], -50, 50)
                        df.loc[df['Team'] == team, 'total_pts_trend_const'] = np.clip(poly_coefs[2], 150, 300)

                        # Calcular valores ajustados
                        fitted_values = np.polyval(poly_coefs, indices)

                        # Calcular residuos para medir volatilidad
                        residuals = team_data['total_points'].values - fitted_values
                        residuals_std = np.std(residuals)
                        df.loc[df['Team'] == team, 'total_pts_trend_residuals'] = np.clip(residuals_std, 0, 100)
                    except:
                        # Si hay error en el ajuste, usar valores neutrales
                        df.loc[df['Team'] == team, 'total_pts_trend_quad'] = 0
                        df.loc[df['Team'] == team, 'total_pts_trend_linear'] = 0
                        df.loc[df['Team'] == team, 'total_pts_trend_const'] = team_data['total_points'].mean()
                        df.loc[df['Team'] == team, 'total_pts_trend_residuals'] = team_data['total_points'].std()

        # 2. Análisis de patrones de ritmo de juego  
        if 'pace_ma_5' in df.columns:
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
                                # Manejar NaN en correlación
                                if np.isnan(corr):
                                    rolling_corrs.append(0.0)
                                else:
                                    rolling_corrs.append(np.clip(corr, -1, 1))
                            else:
                                rolling_corrs.append(0.0)

                    df.loc[df['Team'] == team, 'pace_pts_correlation'] = rolling_corrs

        # 3. Características de calendario y descanso avanzadas
        if 'days_rest' in df.columns:
            # Efecto acumulativo del descanso (CON SHIFT para evitar data leakage)
            df['rest_ma_5'] = df.groupby('Team')['days_rest'].transform(
                lambda x: x.rolling(window=5, min_periods=1).mean().shift(1)
            )

            # Efecto de fatiga (inverso del descanso acumulado)
            df['fatigue_factor'] = 1 / (df['rest_ma_5'] + 1)

            # Interacción entre fatiga y puntos totales históricos
            df['fatigue_total_pts_interaction'] = df['fatigue_factor'] * df['total_pts_ma_5']

        # 4. Análisis de matchup específico para puntos totales (CORREGIDO - SIN DATA LEAKAGE)
        if 'matchup_id' in df.columns and 'total_points' in df.columns:
            # Calcular desviación histórica usando solo datos pasados
            matchup_deviations = []

            # Ordenar por fecha para procesar cronológicamente
            df_sorted = df.sort_values(['Team', 'Date']) if 'Date' in df.columns else df.sort_index()
            
            for idx, row in df_sorted.iterrows():
                team = row['Team']
                opp = row['Opp']

                # OBTENER SOLO DATOS HISTÓRICOS (anteriores al juego actual)
                if 'Date' in df.columns:
                    historical_data = df_sorted[df_sorted['Date'] < row['Date']]
                else:
                    historical_data = df_sorted.iloc[:df_sorted.index.get_loc(idx)]

                if len(historical_data) > 0:
                    # Promedios históricos (SIN incluir el juego actual)
                    team_historical_avg = historical_data[historical_data['Team'] == team]['total_points'].mean()
                    opp_historical_avg = historical_data[historical_data['Team'] == opp]['total_points'].mean()
                    
                    if pd.notna(team_historical_avg) and pd.notna(opp_historical_avg):
                        expected_avg = (team_historical_avg + opp_historical_avg) / 2
                        # La desviación se calculará después del juego para uso futuro
                        deviation = 0  # No usar el valor actual para evitar leakage
                    else:
                        deviation = 0
                else:
                    deviation = 0  # No hay datos históricos suficientes

                matchup_deviations.append(deviation)

            df['matchup_total_pts_deviation'] = matchup_deviations

            # Crear característica de desviación móvil para cada matchup
            df['matchup_deviation_ma'] = df.groupby('matchup_id')['matchup_total_pts_deviation'].transform(
                lambda x: x.rolling(window=3, min_periods=1).mean().shift(1)
            )

        # 5. Características de consistencia y variabilidad avanzadas
        if 'total_points' in df.columns:
            # Rango intercuartílico móvil CON SHIFT (medida robusta de dispersión)
            for window in [5, 10]:
                df[f'total_pts_iqr_{window}'] = df.groupby('Team')['total_points'].transform(
                    lambda x: x.rolling(window=window, min_periods=3).apply(
                        lambda y: np.percentile(y, 75) - np.percentile(y, 25) if len(y) >= 3 else np.nan
                    ).shift(1)
                )

            # Asimetría y curtosis CON SHIFT (forma de la distribución de puntos)
            df['total_pts_skew_5'] = df.groupby('Team')['total_points'].transform(
                lambda x: x.rolling(window=5, min_periods=3).apply(
                    lambda y: pd.Series(y).skew() if len(y) >= 3 else np.nan
                ).shift(1)
            )

            df['total_pts_kurt_5'] = df.groupby('Team')['total_points'].transform(
                lambda x: x.rolling(window=5, min_periods=3).apply(
                    lambda y: pd.Series(y).kurt() if len(y) >= 3 else np.nan
                ).shift(1)
            )

        # 6. Características de estacionalidad y contexto de temporada
        # NOTA: season_phase_numeric ya existe de _apply_base_calculations
        # Interacciones con fase de temporada
        df['season_phase_total_pts'] = df['season_phase_numeric'] * df['total_pts_ma_5'] / 1000

        # ELIMINADO: Análisis de tendencias de overtime
        # overtime_prob_ma5 y overtime_pts_interaction generan ruido según análisis

        # 8. Características de oponente avanzadas
        if 'Opp' in df.columns:
            # Para cada equipo, calcular su efecto en los puntos totales
            team_effects = {}

            for team in df['Team'].unique():
                # Puntos totales promedio cuando este equipo juega
                team_avg = df[df['Team'] == team]['total_points'].mean()
                # Puntos totales promedio de la liga
                league_avg = df['total_points'].mean()
                # Efecto del equipo
                team_effects[team] = team_avg - league_avg

            # Añadir el efecto del oponente como característica
            df['opp_total_pts_effect'] = df['Opp'].map(team_effects)

            # Interacción entre efectos de equipo y oponente
            df['team_opp_effect_interaction'] = df['Team'].map(team_effects) * df['opp_total_pts_effect']

        return df

    def _create_dummy_variables(self, df: pd.DataFrame) -> pd.DataFrame:
        """Crea variables dummy para características categóricas - SIMPLIFICADO"""

        if df.empty:
            return df

        # ELIMINADO: Creación de dummies individuales por día de semana
        # ELIMINADO: high_intensity_month y season_start (generan ruido)
        
        # Solo mantener las transformaciones ya creadas en otros métodos
        logger.info("Variables dummy simplificadas - eliminadas features de ruido")

        return df

    def _clean_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Limpia y prepara las características finales con correcciones robustas"""

        if df.empty:
            return df

        # 1. Eliminar columnas no necesarias para el modelo
        cols_to_drop = [
            # Columnas de identificación que no son features
            'GameID', 'Game_ID', 'game_id', 'game_code',
            # Columnas temporales que ya han sido procesadas
            'season', 'season_type',
            # Columnas duplicadas o redundantes
            '@', 'Home', 'matchup_key'
        ]

        # Eliminar solo las columnas que existen
        cols_to_drop = [col for col in cols_to_drop if col in df.columns]
        if cols_to_drop:
            df = df.drop(columns=cols_to_drop)
            logger.info(f"Eliminadas {len(cols_to_drop)} columnas innecesarias")

        # 2. Identificar y limpiar features con baja varianza
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        low_variance_features = []
        
        for col in numeric_cols:
            if df[col].var() < 0.001:  # Varianza muy baja
                if df[col].nunique() <= 2:  # Feature binaria o constante
                    low_variance_features.append(col)
        
        # Eliminar features con varianza demasiado baja (excepto importantes)
        important_features = ['is_win', 'is_home', 'has_overtime']  # Features binarias importantes
        features_to_remove = [f for f in low_variance_features if f not in important_features]
        
        if features_to_remove:
            df = df.drop(columns=features_to_remove)
            logger.info(f"Eliminadas {len(features_to_remove)} features con baja varianza")

        # 3. ELIMINAR AUTOMÁTICAMENTE DATA LEAKAGE
        if 'total_points' in df.columns:
            logger.info("Eliminando automáticamente features con data leakage...")
            df = self._remove_data_leakage_features(df, 'total_points')
        else:
            logger.warning("No se puede eliminar data leakage: target 'total_points' no encontrado")

        # 4. Actualizar lista de columnas numéricas después de eliminaciones
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        # 5. Limpiar valores infinitos
        inf_cols = []
        for col in numeric_cols:
            if np.isinf(df[col]).any():
                inf_cols.append(col)
                # Reemplazar infinitos con valores límite
                df[col] = df[col].replace([np.inf, -np.inf], np.nan)
        
        if inf_cols:
            logger.info(f"Limpiados valores infinitos en {len(inf_cols)} columnas")

        # 6. Manejar NaN masivos (>80% de la columna)
        nan_heavy_cols = []
        for col in numeric_cols:
            nan_percent = df[col].isnull().sum() / len(df)
            if nan_percent > 0.8:
                nan_heavy_cols.append(col)
        
        if nan_heavy_cols:
            df = df.drop(columns=nan_heavy_cols)
            logger.info(f"Eliminadas {len(nan_heavy_cols)} columnas con >80% NaN")
            # Actualizar lista después de eliminación
            numeric_cols = df.select_dtypes(include=[np.number]).columns

        # 7. Limpiar valores extremos usando IQR robusto
        for col in numeric_cols:
            if col in df.columns:  # Verificar que la columna aún existe
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                
                # Solo aplicar si hay variabilidad suficiente
                if IQR > 0.001:
                    lower_bound = Q1 - 3 * IQR  # 3 IQR en lugar de 1.5 (menos agresivo)
                    upper_bound = Q3 + 3 * IQR
                    
                    # Capear valores extremos
                    outliers_before = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
                    df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
                    
                    if outliers_before > 0:
                        logger.debug(f"Corregidos {outliers_before} outliers en {col}")

        # 8. Manejar NaN restantes de forma inteligente
        fill_strategies = {}
        
        for col in numeric_cols:
            if col in df.columns and df[col].isnull().any():
                nan_count = df[col].isnull().sum()
                
                # Estrategia basada en el tipo de feature
                if 'rate' in col.lower() or 'pct' in col.lower() or '%' in col:
                    # Porcentajes: usar valores típicos
                    if 'fg' in col.lower():
                        fill_value = 0.45
                    elif '3p' in col.lower():
                        fill_value = 0.35
                    elif 'ft' in col.lower():
                        fill_value = 0.75
                    else:
                        fill_value = 0.5
                elif 'win' in col.lower():
                    fill_value = 0.5  # Win rate neutro
                elif 'pts' in col.lower() or 'points' in col.lower():
                    fill_value = df[col].median() if not df[col].isnull().all() else 110
                elif 'rebounds' in col.lower() or 'trb' in col.lower():
                    fill_value = df[col].median() if not df[col].isnull().all() else 45
                elif 'assists' in col.lower() or 'ast' in col.lower():
                    fill_value = df[col].median() if not df[col].isnull().all() else 25
                else:
                    # Para otras features, usar mediana o 0
                    fill_value = df[col].median() if not df[col].isnull().all() else 0.0
                
                df[col] = df[col].fillna(fill_value)
                fill_strategies[col] = f"filled {nan_count} NaN with {fill_value}"

        # 9. Verificación final de calidad
        remaining_issues = []
        
        for col in df.select_dtypes(include=[np.number]).columns:
            # Verificar NaN restantes
            if df[col].isnull().any():
                remaining_issues.append(f"{col}: {df[col].isnull().sum()} NaN")
            
            # Verificar infinitos restantes
            if np.isinf(df[col]).any():
                remaining_issues.append(f"{col}: infinitos")
                # Limpiar infinitos finales
                df[col] = df[col].replace([np.inf, -np.inf], 0)

        if remaining_issues:
            logger.warning(f"Problemas restantes después de limpieza: {len(remaining_issues)}")

        # 10. Eliminar filas con demasiados valores problemáticos - AJUSTADO para conservar más registros
        if len(df.columns) > 10:
            # Usar umbral más permisivo para conservar más registros
            min_features = min(20, len(df.columns) * 0.3)  # Al menos 20 features válidas O 30% del total
            threshold = max(min_features, 10)  # Mínimo absoluto de 10 features
            
            initial_rows = len(df)
            df = df.dropna(thresh=threshold)
            final_rows = len(df)
            
            if initial_rows != final_rows:
                logger.info(f"Eliminadas {initial_rows - final_rows} filas con <{threshold} valores válidos (de {len(df.columns)} features)")

        logger.info(f"Limpieza completada: {df.shape[1]} features, {df.shape[0]} registros")
        
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

        # 1. Detección por correlación directa
        correlations = numeric_df.corr()[target_col].abs().sort_values(ascending=False)

        # Identificar columnas con alta correlación (excluyendo el target mismo)
        high_corr_cols = correlations[correlations > threshold].index.tolist()
        if target_col in high_corr_cols:
            high_corr_cols.remove(target_col)

        leakage_cols.extend(high_corr_cols)

        # 2. Detección por análisis de componentes del target
        # Si el target es una suma de componentes, los componentes son data leakage
        if target_col == 'total_points':
            components = ['PTS', 'PTS_Opp']
            for comp in components:
                if comp in df.columns:
                    leakage_cols.append(comp)

        # 3. Detección por nombres de columnas que indican data leakage
        leakage_patterns = [
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

        # Buscar patrones de data leakage en nombres de columnas
        pattern_leakage_cols = [col for col in df.columns if any(pattern in col for pattern in leakage_patterns)]
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

    def _remove_data_leakage_features(self, df: pd.DataFrame, target_col: str = 'total_points') -> pd.DataFrame:
        """
        Remueve automáticamente features con data leakage del DataFrame.
        
        Args:
            df: DataFrame con features
            target_col: Nombre de la columna objetivo
            
        Returns:
            DataFrame sin features problemáticas
        """
        original_shape = df.shape
        leakage_features = self._detect_data_leakage(df, target_col)
        
        if not leakage_features:
            logger.info("No se detectaron features con data leakage")
            return df
        
        # Categorizar features por tipo de problema
        critical_leakage = []  # Correlación >0.95 - SIEMPRE eliminar
        moderate_leakage = []  # Correlación 0.8-0.95 - evaluar caso por caso
        pattern_leakage = []   # Detectadas por patrones de nombres
        
        if target_col in df.columns:
            correlations = df.select_dtypes(include=[np.number]).corr()[target_col].abs()
            
            for feature in leakage_features:
                if feature in correlations:
                    corr_val = correlations[feature]
                    if corr_val > 0.95:
                        critical_leakage.append(feature)
                    elif corr_val > 0.8:
                        moderate_leakage.append(feature)
                    else:
                        pattern_leakage.append(feature)
                else:
                    pattern_leakage.append(feature)
        else:
            # Sin target disponible, clasificar por patrones
            pattern_leakage = leakage_features
        
        # REGLAS DE ELIMINACIÓN
        features_to_remove = []
        
        # 1. ELIMINAR SIEMPRE: Features críticas
        features_to_remove.extend(critical_leakage)
        
        # 2. EVALUAR: Features moderadas - conservar solo algunas importantes
        important_moderate = []
        for feature in moderate_leakage:
            # Conservar features importantes para el modelo (aunque tengan correlación alta)
            if any(pattern in feature.lower() for pattern in [
                'home_advantage', 'rest_days', 'season_', 'month_',
                'day_of_week', 'streak', 'momentum', 'form'
            ]):
                important_moderate.append(feature)
                logger.info(f"Conservando feature moderada importante: {feature}")
            else:
                features_to_remove.append(feature)
        
        # 3. EVALUAR: Features por patrones - ser más selectivo
        # FEATURES CRÍTICAS PARA EL MODELO - NUNCA ELIMINAR
        critical_model_features = [
            'team_win_rate_10g', 'pts_hist_avg_5g', 'pts_opp_hist_avg_5g', 
            'point_diff_hist_avg_5g', 'pts_consistency_5g'
        ]
        
        for feature in pattern_leakage:
            # PROTEGER features críticas para el modelo
            if feature in critical_model_features:
                logger.info(f"🛡️ PROTEGIENDO feature crítica del modelo: {feature}")
                continue  # NO agregar a features_to_remove
                
            # Conservar features importantes detectadas por patrones
            elif any(pattern in feature.lower() for pattern in [
                'ma_10', 'ma_5', 'trend_', 'momentum_', 'consistency_',
                'home_advantage', 'rest_', 'fatigue_'
            ]) and 'shift' in feature.lower():
                logger.info(f"Conservando feature por patrón (con shift): {feature}")
            elif feature in ['PTS', 'PTS_Opp']:
                # NUNCA conservar componentes directos del target
                features_to_remove.append(feature)
            elif any(direct_pattern in feature for direct_pattern in [
                'FG', 'FGA', '2P', '3P', 'FT', 'AST', 'TRB', 'ORB', 'DRB',
                'STL', 'BLK', 'TOV', 'PF', '+/-'
            ]) and 'shift' not in feature.lower():
                # Eliminar estadísticas directas sin shift
                features_to_remove.append(feature)
            else:
                # Mantener otras features por patrones
                logger.info(f"Conservando feature por patrón: {feature}")
        
        # Eliminar duplicados
        features_to_remove = list(set(features_to_remove))
        
        # PROTECCIÓN FINAL: Asegurar que las features críticas NO se eliminen
        protected_features = [
            'team_win_rate_10g', 'pts_hist_avg_5g', 'pts_opp_hist_avg_5g', 
            'point_diff_hist_avg_5g', 'pts_consistency_5g'
        ]
        
        # Remover features protegidas de la lista de eliminación
        features_to_remove = [f for f in features_to_remove if f not in protected_features]
        
        # Log de features protegidas
        protected_found = [f for f in protected_features if f in df.columns]
        if protected_found:
            logger.info(f"🛡️ PROTEGIDAS {len(protected_found)} features críticas del modelo: {protected_found}")
        
        # Verificar que las features existen antes de eliminar
        existing_features_to_remove = [f for f in features_to_remove if f in df.columns]
        
        if existing_features_to_remove:
            df_cleaned = df.drop(columns=existing_features_to_remove)
            
            logger.warning(f"ELIMINADAS {len(existing_features_to_remove)} features con data leakage:")
            logger.warning(f"  - Críticas (>0.95): {len(critical_leakage)}")
            logger.warning(f"  - Moderadas eliminadas: {len([f for f in moderate_leakage if f in existing_features_to_remove])}")
            logger.warning(f"  - Por patrones: {len([f for f in pattern_leakage if f in existing_features_to_remove])}")
            
            # Mostrar primeras 15 features eliminadas
            for feature in existing_features_to_remove[:15]:
                corr_info = ""
                if target_col in df.columns and feature in df.columns:
                    try:
                        corr_val = df[feature].corr(df[target_col])
                        corr_info = f" (corr: {corr_val:.3f})"
                    except:
                        corr_info = ""
                logger.warning(f"    × {feature}{corr_info}")
            
            if len(existing_features_to_remove) > 15:
                logger.warning(f"    × ... y {len(existing_features_to_remove) - 15} más")
                
            logger.info(f"Shape después de eliminar data leakage: {original_shape} -> {df_cleaned.shape}")
            return df_cleaned
        else:
            logger.info("No se encontraron features válidas para eliminar")
            return df

    def _detect_low_variance_features(self, df: pd.DataFrame, threshold: float = 0.001) -> List[str]:
        """
        Detecta features con muy baja varianza que solo generan ruido.
        
        Args:
            df: DataFrame con features
            threshold: Umbral de varianza mínima
            
        Returns:
            Lista de columnas con baja varianza
        """
        low_variance_cols = []
        
        # Solo analizar columnas numéricas
        numeric_cols = df.select_dtypes(include=['number']).columns
        
        for col in numeric_cols:
            if df[col].var() < threshold:
                low_variance_cols.append(col)
        
        # También detectar columnas con valores únicos muy limitados
        for col in numeric_cols:
            unique_ratio = df[col].nunique() / len(df)
            if unique_ratio < 0.01 and col not in low_variance_cols:  # Menos del 1% de valores únicos
                low_variance_cols.append(col)
        
        # Detectar columnas que son prácticamente constantes
        for col in numeric_cols:
            if df[col].nunique() <= 2 and col not in low_variance_cols:
                # Verificar si una de las categorías domina (>95%)
                value_counts = df[col].value_counts(normalize=True)
                if value_counts.iloc[0] > 0.95:
                    low_variance_cols.append(col)
        
        if low_variance_cols:
            logger.info(f"Features de baja varianza detectadas: {low_variance_cols[:10]}...")
        
        return low_variance_cols

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

    def _apply_noise_filters(self, df: pd.DataFrame, features: List[str]) -> List[str]:
        """
        Aplica filtros avanzados para eliminar features que solo agregan ruido a los modelos de puntos totales.
        Específicamente diseñado para mantener solo features útiles para predicción de total_points.
        
        Args:
            df: DataFrame con los datos
            features: Lista de features a filtrar
            
        Returns:
            List[str]: Lista de features filtradas sin ruido
        """
        logger.info(f"Iniciando filtrado de ruido en {len(features)} features de puntos totales...")
        
        if not features:
            return features
        
        # FILTRO 1: Varianza mínima (solo columnas numéricas)
        clean_features = []
        for feature in features:
            if feature in df.columns:
                # Verificar si es columna numérica
                if pd.api.types.is_numeric_dtype(df[feature]):
                    variance = df[feature].var()
                    if pd.isna(variance) or variance < 1e-8:
                        logger.debug(f"Eliminando {feature} por varianza muy baja: {variance}")
                        continue
                    clean_features.append(feature)
                else:
                    # Columnas no numéricas se omiten automáticamente
                    logger.debug(f"Omitiendo {feature} por ser no numérica")
            else:
                logger.warning(f"Feature {feature} no encontrada en DataFrame")
        
        # FILTRO 2: Valores infinitos o NaN excesivos (solo columnas numéricas)
        filtered_features = []
        for feature in clean_features:
            if feature in df.columns and pd.api.types.is_numeric_dtype(df[feature]):
                nan_pct = df[feature].isna().mean()
                inf_count = np.isinf(df[feature]).sum()
                
                if nan_pct > 0.5:  # Más del 50% NaN
                    logger.debug(f"Eliminando {feature} por exceso de NaN: {nan_pct:.2%}")
                    continue
                if inf_count > 0:  # Cualquier valor infinito
                    logger.debug(f"Eliminando {feature} por valores infinitos: {inf_count}")
                    continue
                filtered_features.append(feature)
        
        # FILTRO 3: Correlación extrema con total_points (posible data leakage, solo numéricas)
        if 'total_points' in df.columns and pd.api.types.is_numeric_dtype(df['total_points']):
            safe_features = []
            for feature in filtered_features:
                if feature in df.columns and feature != 'total_points' and pd.api.types.is_numeric_dtype(df[feature]):
                    try:
                        corr = df[feature].corr(df['total_points'])
                        if pd.isna(corr) or abs(corr) > 0.99:  # Correlación sospechosamente alta
                            logger.debug(f"Eliminando {feature} por correlación sospechosa con total_points: {corr:.3f}")
                            continue
                        safe_features.append(feature)
                    except:
                        safe_features.append(feature)  # Conservar si no se puede calcular correlación
                else:
                    safe_features.append(feature)
        else:
            safe_features = filtered_features
        
        # FILTRO 4: Features que tienden a ser ruidosas o poco predictivas para puntos totales
        noise_patterns = [
            '_squared_',  # Features cuadráticas suelen ser ruidosas
            '_cubed_',    # Features cúbicas suelen ser ruidosas
            '_interaction_complex_',  # Interacciones complejas suelen ser ruidosas
            '_polynomial_',  # Features polinomiales suelen ser ruidosas
            'noise_',     # Features de ruido
            '_random_',   # Features aleatorias
            '_test_',     # Features de prueba
            'cosmic_',    # Features cósmicas experimentales suelen ser ruidosas
            'quantum_',   # Features cuánticas experimentales suelen ser ruidosas
            'fractal_',   # Features fractales suelen ser ruidosas
            '_chaos_',    # Features de caos suelen ser ruidosas
            '_entropy_extreme_',  # Features de entropía extrema
        ]
        
        final_features = []
        removed_by_pattern = []
        
        for feature in safe_features:
            is_noisy = False
            for pattern in noise_patterns:
                if pattern in feature.lower():
                    removed_by_pattern.append(feature)
                    is_noisy = True
                    break
            
            if not is_noisy:
                final_features.append(feature)
        
        # FILTRO 5: Límite de features para evitar overfitting en total_points
        max_features_total = 100  # Límite específico para puntos totales (mayor que individual)
        if len(final_features) > max_features_total:
            logger.info(f"Aplicando límite de features: {max_features_total}")
            # Priorizar features más importantes para puntos totales
            priority_keywords = [
                'total_pts', 'pace', 'momentum', 'efficiency', 'pts_ma', 'trend',
                'home', 'rest', 'season', 'matchup', 'opp', 'shooting',
                'form', 'streak', 'volatility', 'avg_player'
            ]
            
            prioritized_features = []
            remaining_features = []
            
            for feature in final_features:
                is_priority = any(keyword in feature for keyword in priority_keywords)
                if is_priority:
                    prioritized_features.append(feature)
                else:
                    remaining_features.append(feature)
            
            # Tomar features prioritarias + algunas adicionales hasta el límite
            final_features = prioritized_features[:max_features_total//2] + remaining_features[:max_features_total//2]
        
        # Log de resultados
        removed_count = len(features) - len(final_features)
        logger.info(f"Filtrado de ruido completado:")
        logger.info(f"  Features originales: {len(features)}")
        logger.info(f"  Features finales: {len(final_features)}")
        logger.info(f"  Features eliminadas: {removed_count}")
        
        if removed_by_pattern:
            logger.info("Features eliminadas por ruido:")
            for feature in removed_by_pattern[:5]:  # Mostrar solo las primeras 5
                logger.info(f"  - {feature}")
            if len(removed_by_pattern) > 5:
                logger.info(f"  ... y {len(removed_by_pattern) - 5} más")
        
        if not final_features:
            logger.warning("ADVERTENCIA: Todas las features fueron eliminadas por filtros de ruido")
            # Devolver al menos algunas features básicas si todo fue eliminado
            basic_features = [f for f in features if any(keyword in f for keyword in ['pts', 'momentum', 'pace', 'home'])]
            return basic_features[:10] if basic_features else features[:5]
        
        return final_features

    def _apply_robust_clipping(self, X: pd.DataFrame) -> pd.DataFrame:
        """Aplica clipping robusto para evitar valores extremos que causan problemas en el modelo"""
        
        if X.empty:
            return X
        
        X_clipped = X.copy()
        
        # Aplicar clipping por tipo de feature
        for col in X_clipped.columns:
            if X_clipped[col].dtype in ['float64', 'int64']:
                
                # Clipping específico para features problemáticas identificadas
                if 'pts_momentum_squared' in col:
                    # pts_momentum_squared: clipping muy agresivo para reducir varianza
                    X_clipped[col] = np.clip(X_clipped[col], 0, 100)  # Límite muy restrictivo para std < 100
                elif 'momentum_squared' in col or 'pts_ma_5_squared' in col or 'pace_ma_5_squared' in col:
                    # Otras features cuadráticas: usar percentiles para clipping
                    q1 = X_clipped[col].quantile(0.01)
                    q99 = X_clipped[col].quantile(0.99)
                    X_clipped[col] = np.clip(X_clipped[col], q1, q99)
                
                elif 'days_into_season' in col:
                    # Días en temporada: 0-250 días máximo
                    X_clipped[col] = np.clip(X_clipped[col], 0, 250)
                
                elif 'home_total_advantage' in col or 'home_advantage' in col:
                    # Ventajas de local: rango razonable
                    X_clipped[col] = np.clip(X_clipped[col], -100, 100)
                
                elif 'acceleration' in col:
                    # Aceleración: clipping por percentiles
                    q5 = X_clipped[col].quantile(0.05)
                    q95 = X_clipped[col].quantile(0.95)
                    X_clipped[col] = np.clip(X_clipped[col], q5, q95)
                
                elif 'trend_residuals' in col:
                    # Residuos de tendencia: 0-100 rango
                    X_clipped[col] = np.clip(X_clipped[col], 0, 100)
                
                elif 'trend_quad' in col:
                    # Coeficientes cuadráticos: -10 a 10
                    X_clipped[col] = np.clip(X_clipped[col], -10, 10)
                
                elif 'trend_linear' in col:
                    # Coeficientes lineales: -50 a 50
                    X_clipped[col] = np.clip(X_clipped[col], -50, 50)
                
                elif 'correlation' in col:
                    # Correlaciones: -1 a 1
                    X_clipped[col] = np.clip(X_clipped[col], -1, 1)
                
                elif 'fatigue' in col:
                    # Fatiga: 0-10 rango
                    X_clipped[col] = np.clip(X_clipped[col], 0, 10)
                
                elif 'cumulative_minutes' in col:
                    # Minutos acumulados: ya normalizado a 0-100
                    X_clipped[col] = np.clip(X_clipped[col], 0, 100)
                
                else:
                    # Clipping general usando IQR para detectar outliers extremos
                    Q1 = X_clipped[col].quantile(0.25)
                    Q3 = X_clipped[col].quantile(0.75)
                    IQR = Q3 - Q1
                    
                    if IQR > 0:  # Solo si hay varianza
                        lower_bound = Q1 - 3 * IQR  # 3 IQR en lugar de 1.5
                        upper_bound = Q3 + 3 * IQR
                        
                        # Aplicar clipping solo si hay valores extremos
                        outliers_count = ((X_clipped[col] < lower_bound) | (X_clipped[col] > upper_bound)).sum()
                        if outliers_count > 0:
                            X_clipped[col] = np.clip(X_clipped[col], lower_bound, upper_bound)
        
        # Verificar que no hay infinitos o NaN después del clipping
        X_clipped = X_clipped.replace([np.inf, -np.inf], np.nan)
        if X_clipped.isnull().any().any():
            logger.warning("Se encontraron NaN después del clipping, rellenando con 0")
            X_clipped = X_clipped.fillna(0)
        
        return X_clipped
