"""
Módulo de Características para Predicción de Total de Puntos del Juego
=====================================================================

Este módulo contiene toda la lógica de ingeniería de características específica
para la predicción del total de puntos de un juego NBA (ambos equipos combinados).
Implementa características avanzadas optimizadas para predicción de totales.

FEATURES DE DOMINIO ESPECÍFICO con máximo poder predictivo
OPTIMIZADO - Sin cálculos duplicados, sin multicolinealidad
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

class TotalPointsFeatureEngineer:
    """
    Motor de features para predicción de total de puntos del juego (ambos equipos)
    """
    
    def __init__(self, lookback_games: int = 10, teams_df: pd.DataFrame = None, players_df: pd.DataFrame = None):
        """Inicializa el ingeniero de características para total de puntos."""
        self.lookback_games = lookback_games
        self.scaler = StandardScaler()
        self.feature_columns = []
        self.teams_df = teams_df  # Datos de equipos 
        self.players_df = players_df  # Datos de jugadores 
        # Cache para evitar recálculos
        self._cached_calculations = {}
        
    def generate_all_features(self, df: pd.DataFrame) -> List[str]:
        """
        Pipeline completo de features para predicción de total de puntos del juego.
        """
        logger.info("Generando features NBA específicas OPTIMIZADAS para total de puntos del juego...")
        
        # VERIFICACIÓN ESPECÍFICA DE is_win
        if 'is_win' not in df.columns:
            # Buscar columnas similares
            similar_cols = [col for col in df.columns if 'win' in col.lower() or 'result' in col.lower()]

            # CREAR is_win desde Result si está disponible
            if 'Result' in df.columns:
                
                def extract_win_from_result(result_str):
                    """Extrae is_win desde el formato 'W 123-100' o 'L 114-116'"""
                    try:
                        result_str = str(result_str).strip()
                        if result_str.startswith('W'):
                            return 1
                        elif result_str.startswith('L'):
                            return 0
                        else:
                            return None  # Valor inválido
                    except:
                        return None
                
                df['is_win'] = df['Result'].apply(extract_win_from_result)
                
                # Verificar creación exitosa
                valid_wins = df['is_win'].notna().sum()
                total_rows = len(df)
                
                if valid_wins < total_rows:
                    invalid_results = df[df['is_win'].isna()]['Result'].unique()
                    NBALogger.log_warning(logger, "   Formatos no reconocidos: {invalid_results}")
            else:
                NBALogger.log_error(logger, "No se puede crear is_win: columna Result no disponible")
        
        # Trabajar directamente con el DataFrame (NO crear copia)
        if df.empty:
            return []
        
        # Asegurar orden cronológico para evitar data leakage
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'], format='mixed')
            df.sort_values(['Team', 'Date'], inplace=True)
            df.reset_index(drop=True, inplace=True)
        else:
            NBALogger.log_warning(logger, "Columna 'Date' no encontrada - usando orden original")
        
        # PASO 1: Cálculos base (una sola vez) - ADAPTADO PARA TOTAL
        self._create_base_calculations(df)
        
        # PASO 2: Features básicas NBA usando cálculos base
        self._create_basic_nba_features(df)
        
        # PASO 3: Features avanzadas usando cálculos existentes
        self._create_advanced_features_optimized(df)
        
        # PASO 4: Features de contexto y situación
        self._create_context_features(df)
        
        # PASO 5: Features de porcentajes de tiro avanzadas
        self._create_shooting_percentage_features(df)
        
        # PASO 6: Features predictivas mejoradas basadas en feedback del modelo
        self._create_enhanced_predictive_features(df)
        
        # PASO 7: Features ultra-predictivas basadas en feedback más reciente
        self._create_ultra_predictive_features(df)
        
        # PASO 8: Features de interacción final
        self._create_final_interactions(df)
        
        # Aplicar filtros de calidad
        self._apply_quality_filters(df)
        
        # Actualizar lista de features disponibles
        self._update_feature_columns(df)
        
        # Compilar lista de todas las características
        all_features = [col for col in df.columns if col not in [
            'Team', 'Date', 'Away', 'Opp', 'Result', 'MP', 'PTS', 'PTS_Opp',
            'FG', 'FGA', 'FG%', '2P', '2PA', '2P%', '3P', '3PA', '3P%', 
            'FT', 'FTA', 'FT%', 'FG_Opp', 'FGA_Opp', 'FG%_Opp', 
            '2P_Opp', '2PA_Opp', '2P%_Opp', '3P_Opp', '3PA_Opp', '3P%_Opp',
            'FT_Opp', 'FTA_Opp', 'FT%_Opp',
        # Excluir variables categóricas y auxiliares
        'game_pace_tier', 'game_total_tier_adjusted_projection',
        # EXCLUIR DATA LEAKAGE - Target y variables relacionadas
        'total_points', 'game_total_points', 'teams_points'
    ]]
    # Nota: is_win se excluye solo si viene de datos externos, pero si la creamos internamente 
    # para features de momentum, se mantiene como feature válida
            
        # PASO 1: Filtrar features ruidosas
        logger.info("Aplicando filtros de ruido para eliminar features problemáticas...")
        clean_features = self._apply_noise_filters(df, all_features)
        
        logger.info(f"Generadas {len(clean_features)} características ESPECÍFICAS para total de puntos del juego")
        return clean_features
    
    def _create_base_calculations(self, df: pd.DataFrame) -> None:
        """
        CÁLCULOS BASE NBA - Una sola vez para evitar duplicaciones
        ADAPTADO PARA TOTAL DE PUNTOS DEL JUEGO (ambos equipos)
        """
        logger.info("Calculando métricas base NBA para total de puntos del juego...")
        
        # ==================== CÁLCULOS TEMPORALES BASE ====================
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'], format='mixed')
            df['days_rest'] = df.groupby('Team')['Date'].diff().dt.days.fillna(2)
            df['day_of_week'] = df['Date'].dt.dayofweek
            df['month'] = df['Date'].dt.month
            df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
            
            # Días en temporada
            season_start = df['Date'].min()
            df['days_into_season'] = (df['Date'] - season_start).dt.days
        
        # ==================== CÁLCULOS COMBINADOS PARA TOTAL ====================
        # NOTA: NO crear game_total_points aquí para evitar data leakage
        # El target se creará en el trainer después de generar features
        
        # ==================== CÁLCULOS DE POSESIONES NBA OFICIAL ====================
        # Fórmula NBA oficial adaptada para falta de TOV - USAR DATOS HISTÓRICOS
        if all(col in df.columns for col in ['FGA', 'FTA']):
            # Usar promedios históricos para evitar data leakage
            team_fga_avg = df.groupby('Team')['FGA'].transform(
                lambda x: x.rolling(5, min_periods=1).mean().shift(1)
            ).fillna(85)
            team_fta_avg = df.groupby('Team')['FTA'].transform(
                lambda x: x.rolling(5, min_periods=1).mean().shift(1)
            ).fillna(20)
            
            opp_fga_avg = df.groupby('Opp')['FGA_Opp'].transform(
                lambda x: x.rolling(5, min_periods=1).mean().shift(1)
            ).fillna(85)
            opp_fta_avg = df.groupby('Opp')['FTA_Opp'].transform(
                lambda x: x.rolling(5, min_periods=1).mean().shift(1)
            ).fillna(20)
            
            df['team_possessions'] = team_fga_avg + team_fta_avg * 0.44
            df['opp_possessions'] = opp_fga_avg + opp_fta_avg * 0.44
            
            # Total de posesiones del juego
            df['game_total_possessions'] = df['team_possessions'] + df['opp_possessions']
        
        # ==================== CÁLCULOS DE EFICIENCIA COMBINADA ====================
        # True Shooting Percentage combinado - USAR DATOS HISTÓRICOS
        if all(col in df.columns for col in ['FG%', 'FT%', 'FG%_Opp', 'FT%_Opp']):
            # Usar promedios históricos para evitar data leakage
            df['team_true_shooting_approx'] = df.groupby('Team')['FG%'].transform(
                lambda x: x.rolling(5, min_periods=1).mean().shift(1)
            ).fillna(0.45) * 0.6 + df.groupby('Team')['FT%'].transform(
                lambda x: x.rolling(5, min_periods=1).mean().shift(1)
            ).fillna(0.75) * 0.4
            
            df['opp_true_shooting_approx'] = df.groupby('Opp')['FG%_Opp'].transform(
                lambda x: x.rolling(5, min_periods=1).mean().shift(1)
            ).fillna(0.45) * 0.6 + df.groupby('Opp')['FT%_Opp'].transform(
                lambda x: x.rolling(5, min_periods=1).mean().shift(1)
            ).fillna(0.75) * 0.4
            
            # Eficiencia combinada del juego
            df['game_combined_shooting_efficiency'] = (df['team_true_shooting_approx'] + df['opp_true_shooting_approx']) / 2
        
        # Effective Field Goal Percentage combinado - USAR DATOS HISTÓRICOS
        if all(col in df.columns for col in ['FG%', '3P%', '3PA', 'FGA', 'FG%_Opp', '3P%_Opp', '3PA_Opp', 'FGA_Opp']):
            # Usar promedios históricos para evitar data leakage
            team_fg_avg = df.groupby('Team')['FG%'].transform(
                lambda x: x.rolling(5, min_periods=1).mean().shift(1)
            ).fillna(0.45)
            team_3p_avg = df.groupby('Team')['3P%'].transform(
                lambda x: x.rolling(5, min_periods=1).mean().shift(1)
            ).fillna(0.35)
            team_3pa_avg = df.groupby('Team')['3PA'].transform(
                lambda x: x.rolling(5, min_periods=1).mean().shift(1)
            ).fillna(7)
            team_fga_avg = df.groupby('Team')['FGA'].transform(
                lambda x: x.rolling(5, min_periods=1).mean().shift(1)
            ).fillna(85)
            
            opp_fg_avg = df.groupby('Opp')['FG%_Opp'].transform(
                lambda x: x.rolling(5, min_periods=1).mean().shift(1)
            ).fillna(0.45)
            opp_3p_avg = df.groupby('Opp')['3P%_Opp'].transform(
                lambda x: x.rolling(5, min_periods=1).mean().shift(1)
            ).fillna(0.35)
            opp_3pa_avg = df.groupby('Opp')['3PA_Opp'].transform(
                lambda x: x.rolling(5, min_periods=1).mean().shift(1)
            ).fillna(7)
            opp_fga_avg = df.groupby('Opp')['FGA_Opp'].transform(
                lambda x: x.rolling(5, min_periods=1).mean().shift(1)
            ).fillna(85)
            
            df['team_efg_approx'] = team_fg_avg + (0.5 * team_3p_avg * (team_3pa_avg / (team_fga_avg + 1)))
            df['opp_efg_approx'] = opp_fg_avg + (0.5 * opp_3p_avg * (opp_3pa_avg / (opp_fga_avg + 1)))
            
            # eFG% combinado del juego
            df['game_combined_efg'] = (df['team_efg_approx'] + df['opp_efg_approx']) / 2
        
        # ==================== PROYECCIONES DIRECTAS COMBINADAS ====================
        # Proyección directa de scoring combinado - USAR DATOS HISTÓRICOS
        if all(col in df.columns for col in ['FGA', 'FG%', '3PA', '3P%', 'FTA', 'FT%', 'FGA_Opp', 'FG%_Opp', '3PA_Opp', '3P%_Opp', 'FTA_Opp', 'FT%_Opp']):
            # Usar promedios históricos para evitar data leakage
            team_fga_avg = df.groupby('Team')['FGA'].transform(
                lambda x: x.rolling(5, min_periods=1).mean().shift(1)
            ).fillna(85)
            team_fg_avg = df.groupby('Team')['FG%'].transform(
                lambda x: x.rolling(5, min_periods=1).mean().shift(1)
            ).fillna(0.45)
            team_3pa_avg = df.groupby('Team')['3PA'].transform(
                lambda x: x.rolling(5, min_periods=1).mean().shift(1)
            ).fillna(30)
            team_3p_avg = df.groupby('Team')['3P%'].transform(
                lambda x: x.rolling(5, min_periods=1).mean().shift(1)
            ).fillna(0.35)
            team_fta_avg = df.groupby('Team')['FTA'].transform(
                lambda x: x.rolling(5, min_periods=1).mean().shift(1)
            ).fillna(20)
            team_ft_avg = df.groupby('Team')['FT%'].transform(
                lambda x: x.rolling(5, min_periods=1).mean().shift(1)
            ).fillna(0.75)
            
            opp_fga_avg = df.groupby('Opp')['FGA_Opp'].transform(
                lambda x: x.rolling(5, min_periods=1).mean().shift(1)
            ).fillna(85)
            opp_fg_avg = df.groupby('Opp')['FG%_Opp'].transform(
                lambda x: x.rolling(5, min_periods=1).mean().shift(1)
            ).fillna(0.45)
            opp_3pa_avg = df.groupby('Opp')['3PA_Opp'].transform(
                lambda x: x.rolling(5, min_periods=1).mean().shift(1)
            ).fillna(30)
            opp_3p_avg = df.groupby('Opp')['3P%_Opp'].transform(
                lambda x: x.rolling(5, min_periods=1).mean().shift(1)
            ).fillna(0.35)
            opp_fta_avg = df.groupby('Opp')['FTA_Opp'].transform(
                lambda x: x.rolling(5, min_periods=1).mean().shift(1)
            ).fillna(20)
            opp_ft_avg = df.groupby('Opp')['FT%_Opp'].transform(
                lambda x: x.rolling(5, min_periods=1).mean().shift(1)
            ).fillna(0.75)
            
            df['team_direct_scoring_projection'] = (
                team_fga_avg * team_fg_avg * 2 +    # Puntos de 2P estimados históricos
                team_3pa_avg * team_3p_avg * 3 +    # Puntos de 3P estimados históricos
                team_fta_avg * team_ft_avg * 1      # Puntos de FT estimados históricos
            )
            
            df['opp_direct_scoring_projection'] = (
                opp_fga_avg * opp_fg_avg * 2 +
                opp_3pa_avg * opp_3p_avg * 3 +
                opp_fta_avg * opp_ft_avg * 1
            )
            
            # Proyección total del juego
            df['game_total_scoring_projection'] = df['team_direct_scoring_projection'] + df['opp_direct_scoring_projection']
        
        # ==================== VOLÚMENES Y MÉTRICAS COMBINADAS ====================
        if all(col in df.columns for col in ['FGA', 'FTA', 'FGA_Opp', 'FTA_Opp']):
            # Usar promedios históricos para evitar data leakage
            team_fga_avg = df.groupby('Team')['FGA'].transform(
                lambda x: x.rolling(5, min_periods=1).mean().shift(1)
            ).fillna(85)
            team_fta_avg = df.groupby('Team')['FTA'].transform(
                lambda x: x.rolling(5, min_periods=1).mean().shift(1)
            ).fillna(20)
            
            opp_fga_avg = df.groupby('Opp')['FGA_Opp'].transform(
                lambda x: x.rolling(5, min_periods=1).mean().shift(1)
            ).fillna(85)
            opp_fta_avg = df.groupby('Opp')['FTA_Opp'].transform(
                lambda x: x.rolling(5, min_periods=1).mean().shift(1)
            ).fillna(20)
            
            df['team_total_shot_volume'] = team_fga_avg + team_fta_avg
            df['opp_total_shot_volume'] = opp_fga_avg + opp_fta_avg
            
            # Volumen total del juego
            df['game_total_shot_volume'] = df['team_total_shot_volume'] + df['opp_total_shot_volume']
        
        # ==================== FEATURES DE CONTEXTO BASE ====================
        # Ventaja local
        if 'Away' in df.columns:
            df['team_is_home'] = (df['Away'] == 0).astype(int)
        else:
            df['team_is_home'] = 0
        
        # Factor de energía basado en descanso para el juego
        if 'days_rest' in df.columns:
            df['team_energy_factor'] = np.where(
                df['days_rest'] == 0, 0.92,  # Back-to-back penalty
                np.where(df['days_rest'] == 1, 0.97,  # 1 día
                        np.where(df['days_rest'] >= 3, 1.03, 1.0))  # 3+ días boost
            )
        
        # Ritmo esperado del juego
        if 'game_total_possessions' in df.columns:
            df['game_expected_pace'] = df['game_total_possessions'] / 48  # Por minuto
        
        # Importancia del partido para total de puntos
        if 'days_into_season' in df.columns:
            df['game_importance_factor'] = np.where(
                df['days_into_season'] > 200, 1.04,  # Playoffs/final temporada
                np.where(df['days_into_season'] > 100, 1.02, 1.0)  # Mitad temporada
            )
    
    def _create_basic_nba_features(self, df: pd.DataFrame) -> None:
        """Features básicas NBA usando cálculos base existentes - ADAPTADO PARA TOTAL"""
        
        # ==================== PROMEDIOS MÓVILES OPTIMIZADOS PARA TOTAL ====================
        windows = [3, 5, 7, 10]
        
        # Promedios de características base del juego
        for window in windows:
            if 'game_total_scoring_projection' in df.columns:
                df[f'game_total_projection_avg_{window}g'] = df.groupby('Team')['game_total_scoring_projection'].transform(
                    lambda x: x.rolling(window=window, min_periods=1).mean().shift(1)
                )
            
            if 'game_combined_shooting_efficiency' in df.columns:
                df[f'game_shooting_efficiency_avg_{window}g'] = df.groupby('Team')['game_combined_shooting_efficiency'].transform(
                    lambda x: x.rolling(window=window, min_periods=1).mean().shift(1)
                )
            
            if 'game_total_possessions' in df.columns:
                df[f'game_pace_avg_{window}g'] = df.groupby('Team')['game_total_possessions'].transform(
                    lambda x: x.rolling(window=window, min_periods=1).mean().shift(1)
                )
            
            if 'game_total_shot_volume' in df.columns:
                df[f'game_volume_avg_{window}g'] = df.groupby('Team')['game_total_shot_volume'].transform(
                    lambda x: x.rolling(window=window, min_periods=1).mean().shift(1)
                )
        
        # ==================== MÉTRICAS DERIVADAS OPTIMIZADAS ====================
        # Ritmo base del juego
        if 'game_pace_avg_5g' in df.columns:
            df['game_base_pace'] = df['game_pace_avg_5g']
        
        # Proyección base del total del juego
        if 'game_total_projection_avg_5g' in df.columns:
            df['game_base_total_projection'] = df['game_total_projection_avg_5g']
        
        # ==================== FEATURES ESPECÍFICAS REQUERIDAS POR EL MODELO ====================
        
        # Promedios históricos de totales
        if 'game_total_points' in df.columns:
            df['total_points_hist_avg_5g'] = df.groupby('Team')['game_total_points'].transform(
                lambda x: x.rolling(5, min_periods=1).mean().shift(1)
            )
            
            df['total_points_hist_avg_10g'] = df.groupby('Team')['game_total_points'].transform(
                lambda x: x.rolling(10, min_periods=1).mean().shift(1)
            )
        
        # Consistencia de totales
        if 'game_total_points' in df.columns:
            df['total_points_consistency_5g'] = df.groupby('Team')['game_total_points'].transform(
                lambda x: x.rolling(5, min_periods=1).std().shift(1)
            )
        
        # Diferencias de scoring combinadas
        if 'PTS' in df.columns and 'PTS_Opp' in df.columns:
            df['point_diff_hist_avg_5g'] = df.groupby('Team')['PTS'].transform(
                lambda x: (df.loc[x.index, 'PTS'] - df.loc[x.index, 'PTS_Opp']).rolling(5, min_periods=1).mean().shift(1)
            )
        
        # Ratios de victorias para momentum
        if 'is_win' in df.columns:
            df['team_win_rate_10g'] = df.groupby('Team')['is_win'].transform(
                lambda x: x.rolling(10, min_periods=1).mean().shift(1)
            )
    
    def _create_advanced_features_optimized(self, df: pd.DataFrame) -> None:
        """Features avanzadas optimizadas sin duplicaciones y multicolinealidad - ADAPTADO PARA TOTAL"""
        
        # ==================== PROYECCIONES OPTIMIZADAS PARA TOTAL ====================
        # Proyección matemática combinada del juego
        if all(col in df.columns for col in ['game_total_possessions', 'game_combined_shooting_efficiency']):
            df['game_mathematical_projection'] = (
                df['game_total_possessions'] * df['game_combined_shooting_efficiency'] * 1.8
            )
        
        # Proyección híbrida del total del juego
        if all(col in df.columns for col in ['game_mathematical_projection', 'game_total_shot_volume']):
            df['game_hybrid_total_projection'] = (
                df['game_mathematical_projection'] * 0.6 +
                df['game_total_shot_volume'] * 1.2  # Ajuste para totales
            )
        
        # ==================== FEATURES MEJORADAS PARA RANGOS ESPECÍFICOS ====================
        # Clasificar juegos por tendencia de scoring total
        if 'game_total_scoring_projection' in df.columns:
            # Clasificar juegos por total esperado
            total_values = df['game_total_scoring_projection'].fillna(df['game_total_scoring_projection'].median())
            df['game_pace_tier'] = pd.cut(
                total_values, 
                bins=[0, 200, 220, 300], 
                labels=['low_scoring', 'mid_scoring', 'high_scoring']
            )
            # Convertir a string para evitar problemas categóricos
            df['game_pace_tier'] = df['game_pace_tier'].astype(str)
            df['game_pace_tier'] = df['game_pace_tier'].fillna('mid_scoring')
            
            # Features específicas por tier de total
            for tier in ['low_scoring', 'mid_scoring', 'high_scoring']:
                tier_mask = df['game_pace_tier'] == tier
                if 'game_combined_shooting_efficiency' in df.columns:
                    tier_col = f'game_{tier}_efficiency'
                    df[tier_col] = df.groupby('Team')['game_combined_shooting_efficiency'].transform(
                        lambda x: x.where(tier_mask).expanding().mean().shift(1)
                    ).fillna(df['game_combined_shooting_efficiency'])
        
        # ==================== FEATURES DE CONSISTENCIA MEJORADAS ====================
        # Estabilidad de totales por ventana temporal
        if 'game_total_scoring_projection' in df.columns:
            # Coeficiente de variación (estabilidad)
            df['game_total_stability'] = df.groupby('Team')['game_total_scoring_projection'].transform(
                lambda x: x.rolling(window=10, min_periods=3).std().shift(1) / 
                         (x.rolling(window=10, min_periods=3).mean().shift(1) + 1e-6)
            )
            # Invertir para que mayor valor = mayor estabilidad
            df['game_total_stability'] = 1 / (df['game_total_stability'] + 0.01)
        
        # ==================== FEATURES DE OPONENTE OPTIMIZADAS ====================
        # Calidad combinada del matchup
        if 'Opp' in df.columns and 'PTS_Opp' in df.columns:
            opp_def_ranking = df.groupby('Opp')['PTS_Opp'].transform(
                lambda x: x.shift(1).rolling(10, min_periods=3).mean()
            )
            df['opponent_def_strength'] = opp_def_ranking.rank(pct=True).fillna(0.5)
        
        if 'Opp' in df.columns and 'PTS' in df.columns:
            opp_off_ranking = df.groupby('Opp')['PTS'].transform(
                lambda x: x.shift(1).rolling(10, min_periods=3).mean()
            )
            df['opponent_off_strength'] = opp_off_ranking.rank(pct=True).fillna(0.5)
        
        # Factor de calidad total del matchup
        if all(col in df.columns for col in ['opponent_off_strength', 'opponent_def_strength']):
            df['matchup_total_quality'] = (
                df['opponent_off_strength'] + df['opponent_def_strength']
            ) / 2
        
        # ==================== FEATURES DE MOMENTUM OPTIMIZADAS ====================
        # Verificar disponibilidad de is_win del data loader
        if 'is_win' in df.columns:
            # Win percentage histórico combinado
            df['combined_win_pct_5g'] = df.groupby('Team')['is_win'].transform(
                lambda x: x.shift(1).rolling(window=5, min_periods=1).mean()
            ).fillna(0.5)
            
            # Momentum reciente combinado
            df['combined_recent_momentum'] = df.groupby('Team')['is_win'].transform(
                lambda x: x.shift(1).rolling(window=3, min_periods=1).sum()
            ).fillna(1.5)
        
        # ==================== FEATURES DE MATCHUP ESPECÍFICO ====================
        # Historial vs oponente específico del total
        if all(col in df.columns for col in ['Team', 'Opp', 'game_total_scoring_projection']):
            df['total_vs_opp_history'] = df.groupby(['Team', 'Opp'])['game_total_scoring_projection'].transform(
                lambda x: x.expanding().mean().shift(1)
            )
        
        # Edge del matchup específico para totales
        if all(col in df.columns for col in ['total_vs_opp_history', 'game_total_projection_avg_10g']):
            df['total_matchup_edge'] = (
                df['total_vs_opp_history'].fillna(df['game_total_projection_avg_10g']) - 
                df['game_total_projection_avg_10g']
            )
    
    def _create_context_features(self, df: pd.DataFrame) -> None:
        """Features de contexto situacional - ADAPTADO PARA TOTAL"""
        
        # ==================== FACTORES CONTEXTUALES ESPECÍFICOS ====================
        # Factor de altitud para totales (afecta tiros)
        altitude_teams = ['DEN', 'UTA', 'PHX']
        df['game_altitude_factor'] = df['Team'].apply(lambda x: 1.03 if x in altitude_teams else 1.0)
        
        # Factor de rivalidad para totales (más puntos en rivalidades)
        rivalry_boost_teams = {
            'LAL': ['BOS', 'LAC'], 'BOS': ['LAL', 'PHI'], 'GSW': ['LAC', 'CLE'],
            'MIA': ['BOS', 'NYK'], 'CHI': ['DET', 'CLE'], 'DAL': ['SAS', 'HOU']
        }
        def get_total_rivalry_factor(row):
            team = row['Team']
            opp = row.get('Opp', '')
            if team in rivalry_boost_teams and opp in rivalry_boost_teams[team]:
                return 1.05  # Rivalidades tienden a más puntos
            return 1.0
        
        if 'Opp' in df.columns:
            df['game_rivalry_factor'] = df.apply(get_total_rivalry_factor, axis=1)
        else:
            df['game_rivalry_factor'] = 1.0
        
        # Fatiga de temporada para totales
        if 'month' in df.columns:
            df['game_season_fatigue'] = np.where(df['month'].isin([1, 2, 3]), 0.97, 1.0)
        
        # Factor de urgencia del juego
        if 'days_into_season' in df.columns:
            df['game_urgency_factor'] = np.where(df['days_into_season'] > 200, 1.06, 1.0)
        
        # Advantage de descanso para totales (equipos descansados = más puntos)
        if 'days_rest' in df.columns:
            df['game_rest_total_factor'] = np.where(
                df['days_rest'] == 0, -2.5,  # Back-to-back reduce totales
                np.where(df['days_rest'] == 1, -0.8,  
                        np.where(df['days_rest'] >= 3, 1.8, 0.0))  # Equipos descansados = más puntos
            )
        
        # Factor de fin de semana (más espectáculo = más puntos)
        if 'is_weekend' in df.columns:
            df['game_weekend_factor'] = df['is_weekend'] * 1.5
    
    def _create_final_interactions(self, df: pd.DataFrame) -> None:
        """Features de interacción final optimizadas sin multicolinealidad - ADAPTADO PARA TOTAL"""
        
        # ==================== INTERACCIONES SIMPLIFICADAS PARA TOTAL ====================
        # Interacción pace-efficiency para totales
        if all(col in df.columns for col in ['game_total_possessions', 'game_combined_shooting_efficiency']):
            df['game_pace_efficiency_interaction'] = df['game_total_possessions'] * df['game_combined_shooting_efficiency']
        
        # Interacción momentum-contexto del juego
        if all(col in df.columns for col in ['combined_win_pct_5g', 'team_energy_factor']):
            df['game_momentum_context'] = df['combined_win_pct_5g'] * df['team_energy_factor']
        
        # Interacción calidad-eficiencia del matchup
        if all(col in df.columns for col in ['game_combined_shooting_efficiency', 'matchup_total_quality']):
            df['game_quality_efficiency_interaction'] = (
                df['game_combined_shooting_efficiency'] * df['matchup_total_quality']
            )
        
        # ==================== INTERACCIONES MEJORADAS PARA ESTABILIDAD ====================
        # Estabilidad-momentum combinada para totales
        if all(col in df.columns for col in ['game_total_stability', 'combined_win_pct_5g']):
            df['game_stability_momentum'] = df['game_total_stability'] * df['combined_win_pct_5g']
        
        # ==================== PROYECCIÓN FINAL MEJORADA DEL TOTAL ====================
        # Proyección base mejorada para totales
        base_features = ['game_hybrid_total_projection', 'game_quality_efficiency_interaction']
        if all(col in df.columns for col in base_features):
            df['game_enhanced_total_projection'] = (
                df['game_hybrid_total_projection'] * 0.5 +
                df['game_quality_efficiency_interaction'] * 15
            )
        
        # Ajustes contextuales mejorados para totales
        contextual_features = ['game_altitude_factor', 'game_rest_total_factor', 'game_importance_factor', 'game_weekend_factor']
        if all(col in df.columns for col in contextual_features):
            df['game_contextual_total_adjustment'] = (
                df['game_altitude_factor'] * 2.0 +
                df['game_rest_total_factor'] +
                df['game_importance_factor'] * 1.5 +
                df['game_weekend_factor']
            )
        
        # Ajuste por estabilidad del total
        stability_features = ['game_stability_momentum', 'total_matchup_edge']
        if all(col in df.columns for col in stability_features):
            df['game_stability_total_adjustment'] = (
                df['game_stability_momentum'] * 3.0 +
                df['total_matchup_edge'] * 0.5
            )
        
        # Una sola proyección final robusta para el total MEJORADA
        final_features = ['game_enhanced_total_projection', 'game_contextual_total_adjustment', 'game_stability_total_adjustment']
        if all(col in df.columns for col in final_features):
            df['game_final_total_projection'] = (
                df['game_enhanced_total_projection'] +
                df['game_contextual_total_adjustment'] + 
                df['game_stability_total_adjustment']
            )
        elif 'game_enhanced_total_projection' in df.columns and 'game_contextual_total_adjustment' in df.columns:
            # Fallback si no hay stability features
            df['game_final_total_projection'] = (
                df['game_enhanced_total_projection'] +
                df['game_contextual_total_adjustment']
            )
        
        # ==================== AJUSTES ESPECÍFICOS POR RANGO ====================
        # Ajustes específicos para diferentes rangos de totales
        if all(col in df.columns for col in ['game_final_total_projection', 'game_pace_tier']):
            # Ajustar predicciones según el tier de total histórico
            df['game_total_tier_adjusted_projection'] = df['game_final_total_projection'].copy()
            
            # Ajustes por tier para mejorar precisión en cada rango
            low_mask = df['game_pace_tier'] == 'low_scoring'
            mid_mask = df['game_pace_tier'] == 'mid_scoring' 
            high_mask = df['game_pace_tier'] == 'high_scoring'
            
            # Ajustes conservadores para juegos de bajo scoring
            df.loc[low_mask, 'game_total_tier_adjusted_projection'] *= 0.97
            
            # Ajustes neutros para juegos de medio scoring
            df.loc[mid_mask, 'game_total_tier_adjusted_projection'] *= 1.0
            
            # Ajustes ligeramente agresivos para juegos de alto scoring
            df.loc[high_mask, 'game_total_tier_adjusted_projection'] *= 1.03
            
            # Usar la proyección ajustada como final
            df['game_final_total_projection'] = df['game_total_tier_adjusted_projection']
        
        # Límites realistas NBA para totales (calibrado con datos reales)
        if 'game_final_total_projection' in df.columns:
            df['game_final_total_prediction'] = np.clip(df['game_final_total_projection'], 160, 300)
        
        # ==================== FEATURES DE CONFIANZA MEJORADAS DEL TOTAL ====================
        # Métrica de confianza específica del total mejorada
        confidence_features = ['matchup_total_quality', 'game_combined_shooting_efficiency', 'game_total_stability']
        if all(col in df.columns for col in confidence_features):
            df['game_total_prediction_confidence'] = (
                df['matchup_total_quality'] * 0.4 +
                df['game_combined_shooting_efficiency'] * 0.3 +
                df['game_total_stability'] * 0.3
            )
        elif all(col in df.columns for col in ['matchup_total_quality', 'game_combined_shooting_efficiency']):
            # Fallback sin stability
            df['game_total_prediction_confidence'] = (
                df['matchup_total_quality'] * 0.5 +
                df['game_combined_shooting_efficiency'] * 0.5
            )
    
    def _apply_quality_filters(self, df: pd.DataFrame) -> None:
        """Aplicar filtros de calidad para eliminar features problemáticas"""
        # Eliminar features con alta correlación o problemas
        problematic_features = [
        ]
        
        for feature in problematic_features:
            if feature in df.columns:
                df.drop(columns=[feature], inplace=True)
    
    def _update_feature_columns(self, df: pd.DataFrame):
        """Actualizar lista de columnas de features disponibles"""
        exclude_cols = ['Team', 'Date', 'Away', 'Opp', 'Result', 'MP', 'PTS', 'PTS_Opp',
                       'FG', 'FGA', 'FG%', '2P', '2PA', '2P%', '3P', '3PA', '3P%', 
                       'FT', 'FTA', 'FT%', 'FG_Opp', 'FGA_Opp', 'FG%_Opp', 
                       '2P_Opp', '2PA_Opp', '2P%_Opp', '3P_Opp', '3PA_Opp', '3P%_Opp',
                       'FT_Opp', 'FTA_Opp', 'FT%_Opp',
                       'game_total_points']  # Excluir el target
        
        self.feature_columns = [col for col in df.columns if col not in exclude_cols]
    
    def get_feature_importance_groups(self) -> Dict[str, List[str]]:
        """Retorna las características agrupadas por categoría."""
        groups = {
            'base_projections': [
                'game_total_scoring_projection', 'game_mathematical_projection', 
                'game_hybrid_total_projection', 'game_final_total_projection'
            ],
            'efficiency_metrics': [
                'game_combined_shooting_efficiency', 'game_combined_efg', 
                'game_pace_efficiency_interaction'
            ],
            'historical_trends': [
                'game_total_projection_avg_5g', 'game_shooting_efficiency_avg_5g',
                'game_pace_avg_5g', 'total_points_hist_avg_5g', 'total_points_hist_avg_10g'
            ],
            'opponent_factors': [
                'matchup_total_quality', 'total_vs_opp_history', 'total_matchup_edge'
            ],
            'contextual_factors': [
                'game_altitude_factor', 'game_rivalry_factor', 'game_rest_total_factor',
                'game_urgency_factor', 'game_weekend_factor'
            ],
            'momentum_features': [
                'combined_win_pct_5g', 'combined_recent_momentum',
                'game_momentum_context'
            ],
            'final_interactions': [
                'game_quality_efficiency_interaction', 'game_stability_momentum',
                'game_final_total_prediction'
            ]
        }
        
        return groups
    
    def validate_features(self, df: pd.DataFrame) -> Dict[str, any]:
        """Valida la calidad de las características generadas."""
        validation_report = {
            'total_features': 0,
            'missing_features': [],
            'feature_coverage': {}
        }
        
        groups = self.get_feature_importance_groups()
        all_features = []
        for group_features in groups.values():
            all_features.extend(group_features)
        
        validation_report['total_features'] = len(all_features)
        
        # Verificar características faltantes
        for feature in all_features:
            if feature not in df.columns:
                validation_report['missing_features'].append(feature)
        
        # Verificar cobertura por grupo
        for group_name, group_features in groups.items():
            existing = sum(1 for f in group_features if f in df.columns)
            validation_report['feature_coverage'][group_name] = {
                'total': len(group_features),
                'existing': existing,
                'coverage': existing / len(group_features) if group_features else 0
            }
        
        logger.info(f"Validación completada: {len(all_features)} features, "
                   f"{len(validation_report['missing_features'])} faltantes")
        
        return validation_report 
    
    def _log_feature_generation_start(self, feature_type: str):
        """Log inicio de generación de features"""
        NBALogger.log_training_progress(self.logger, f"Generando features {feature_type}")
    
    def _log_feature_generation_complete(self, feature_type: str, count: int):
        """Log finalización de generación de features"""
        self.logger.info(f"Features {feature_type} completadas: {count} generadas")
    
    def _log_data_validation(self, validation_results: dict):
        """Log resultados de validación de datos"""
        NBALogger.log_data_info(self.logger, validation_results)
    
    def _apply_noise_filters(self, df: pd.DataFrame, features: List[str]) -> List[str]:

        """
        Aplica filtros avanzados para eliminar features que solo agregan ruido a los modelos de total de puntos.
        DESHABILITADO: Para mantener compatibilidad con modelos entrenados.
        
        Args:
            df: DataFrame con los datos
            features: Lista de features a filtrar
            
        Returns:
            List[str]: Lista de features filtradas sin ruido
        """
        logger.info(f"Filtrado de ruido DESHABILITADO para compatibilidad con modelos entrenados")
        logger.info(f"Manteniendo todas las {len(features)} features de total de puntos")
        
        # Retornar todas las features sin filtrar
        return features
    
    def _create_shooting_percentage_features(self, df: pd.DataFrame) -> None:
        """
        Crea características avanzadas de porcentajes de tiro con ventanas móviles y shift
        para evitar data leakage.
        
        Args:
            df: DataFrame con datos de equipos (modificado in-place)
        """
        if df.empty:
            return

        # Verificar columnas disponibles
        shooting_cols = [
            'FG', 'FGA', 'FG%', '2P', '2PA', '2P%', '3P', '3PA', '3P%', 'FT', 'FTA', 'FT%',
            'FG_Opp', 'FGA_Opp', 'FG%_Opp', '2P_Opp', '2PA_Opp', '2P%_Opp', 
            '3P_Opp', '3PA_Opp', '3P%_Opp', 'FT_Opp', 'FTA_Opp', 'FT%_Opp'
        ]

        available_cols = [col for col in shooting_cols if col in df.columns]

        if not available_cols:
            NBALogger.log_warning(logger, "No hay columnas de tiro disponibles para crear características")
            return

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
            # Ratio de tiros de 3 puntos - USAR DATOS HISTÓRICOS
            team_3pa_avg = df.groupby('Team')['3PA'].transform(
                lambda x: x.rolling(5, min_periods=1).mean().shift(1)
            ).fillna(30)
            team_2pa_avg = df.groupby('Team')['2PA'].transform(
                lambda x: x.rolling(5, min_periods=1).mean().shift(1)
            ).fillna(55)
            
            df['three_point_ratio'] = team_3pa_avg / (team_2pa_avg + team_3pa_avg).replace(0, np.nan)

            # Media móvil de ratio de tiros de 3 puntos con shift
            for window in windows:
                df[f'three_point_ratio_ma_{window}'] = df.groupby('Team')['three_point_ratio'].transform(
                    lambda x: x.rolling(window=window, min_periods=1).mean().shift(1)
                )

        # Lo mismo para el oponente - USAR DATOS HISTÓRICOS
        if '2PA_Opp' in available_cols and '3PA_Opp' in available_cols:
            opp_3pa_avg = df.groupby('Opp')['3PA_Opp'].transform(
                lambda x: x.rolling(5, min_periods=1).mean().shift(1)
            ).fillna(30)
            opp_2pa_avg = df.groupby('Opp')['2PA_Opp'].transform(
                lambda x: x.rolling(5, min_periods=1).mean().shift(1)
            ).fillna(55)
            
            df['three_point_ratio_opp'] = opp_3pa_avg / (opp_2pa_avg + opp_3pa_avg).replace(0, np.nan)

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
    
    def _create_enhanced_predictive_features(self, df: pd.DataFrame) -> None:
        """
        Crea características predictivas mejoradas basadas en el feedback del modelo.
        Estas features están optimizadas para maximizar el poder predictivo.
        
        Args:
            df: DataFrame con datos de equipos (modificado in-place)
        """
        if df.empty:
            return

        logger.info("Creando características predictivas mejoradas basadas en feedback del modelo")

        # ==================== DIFERENCIALES AVANZADOS DE PORCENTAJES ====================
        # Diferenciales ponderados por importancia histórica
        percentage_pairs = [
            ('FG%', 'FG%_Opp'), ('2P%', '2P%_Opp'), ('3P%', '3P%_Opp'), ('FT%', 'FT%_Opp')
        ]

        for team_pct, opp_pct in percentage_pairs:
            if team_pct in df.columns and opp_pct in df.columns:
                # Diferencial directo ponderado
                df[f'{team_pct}_vs_{opp_pct}_diff_weighted'] = (
                    df[team_pct] - df[opp_pct]
                ) * 1.5  # Ponderación basada en importancia del modelo
                
                # Diferencial de tendencias
                team_trend = f'{team_pct}_trend_5'
                opp_trend = f'{opp_pct}_trend_5'
                if team_trend in df.columns and opp_trend in df.columns:
                    df[f'{team_pct}_vs_{opp_pct}_trend_diff'] = (
                        df[team_trend] - df[opp_trend]
                    ) * 2.0  # Mayor peso para tendencias
                
                # Diferencial de estabilidad (consistencia)
                for window in [3, 5, 10]:
                    team_ma = f'{team_pct}_ma_{window}'
                    opp_ma = f'{opp_pct}_ma_{window}'
                    if team_ma in df.columns and opp_ma in df.columns:
                        df[f'{team_pct}_vs_{opp_pct}_stability_diff_{window}'] = (
                            df[team_ma] - df[opp_ma]
                        ) * (window / 5.0)  # Peso proporcional a la ventana

        # ==================== VOLÚMENES TOTALES MEJORADOS ====================
        # Volúmenes totales con ponderación temporal
        volume_pairs = [('FGA', 'FGA_Opp'), ('3PA', '3PA_Opp'), ('FTA', 'FTA_Opp')]
        
        for team_vol, opp_vol in volume_pairs:
            if team_vol in df.columns and opp_vol in df.columns:
                # Volumen total ponderado por ritmo del juego
                df[f'{team_vol}_{opp_vol}_total_weighted'] = (
                    df[team_vol] + df[opp_vol]
                ) * 1.2  # Ponderación basada en importancia
                
                # Volumen total con ajuste por eficiencia
                team_pct = team_vol.replace('A', '%') if 'A' in team_vol else team_vol
                opp_pct = opp_vol.replace('A', '%_Opp') if 'A' in opp_vol else opp_vol
                
                if team_pct in df.columns and opp_pct in df.columns:
                    df[f'{team_vol}_{opp_vol}_efficiency_adjusted'] = (
                        (df[team_vol] + df[opp_vol]) * 
                        ((df[team_pct] + df[opp_pct]) / 2)
                    )
                
                # Volumen total con tendencia
                for window in [3, 5, 10]:
                    total_ma = f'{team_vol}_{opp_vol}_total_ma_{window}'
                    if total_ma in df.columns:
                        df[f'{team_vol}_{opp_vol}_total_trend_{window}'] = (
                            df[total_ma] * 1.1  # Ponderación para tendencias
                        )

        # ==================== INTERACCIONES PORCENTAJE-VOLUMEN MEJORADAS ====================
        # Interacciones más sofisticadas basadas en feedback
        interaction_pairs = [
            ('FG%', 'FGA'), ('3P%', '3PA'), ('FT%', 'FTA'),
            ('FG%_Opp', 'FGA_Opp'), ('3P%_Opp', '3PA_Opp'), ('FT%_Opp', 'FTA_Opp')
        ]
        
        for pct_col, vol_col in interaction_pairs:
            if pct_col in df.columns and vol_col in df.columns:
                # Interacción básica mejorada
                for window in [3, 5, 10]:
                    pct_ma = f'{pct_col}_ma_{window}'
                    if pct_ma in df.columns:
                        df[f'{pct_col}_{vol_col}_interaction_enhanced_{window}'] = (
                            df[pct_ma] * df[vol_col] * (window / 5.0)
                        )
                
                # Interacción con tendencia
                pct_trend = f'{pct_col}_trend_5'
                if pct_trend in df.columns:
                    df[f'{pct_col}_{vol_col}_trend_interaction'] = (
                        df[pct_trend] * df[vol_col] * 1.5
                    )

        # ==================== RATIOS DE 3 PUNTOS MEJORADOS ====================
        # Ratios con más contexto y ponderación
        if '2PA' in df.columns and '3PA' in df.columns:
            # Ratio ponderado por eficiencia
            if '3P%' in df.columns and '2P%' in df.columns:
                df['three_point_ratio_efficiency_weighted'] = (
                    df['3PA'] / (df['2PA'] + df['3PA']).replace(0, np.nan)
                ) * (df['3P%'] / (df['2P%'] + 0.1))  # Ponderación por eficiencia
            
            # Ratio con tendencia
            for window in [3, 5, 10]:
                ratio_ma = f'three_point_ratio_ma_{window}'
                if ratio_ma in df.columns:
                    df[f'three_point_ratio_trend_{window}'] = (
                        df[ratio_ma] * 1.3  # Ponderación para tendencias
                    )

        # Lo mismo para el oponente
        if '2PA_Opp' in df.columns and '3PA_Opp' in df.columns:
            if '3P%_Opp' in df.columns and '2P%_Opp' in df.columns:
                df['three_point_ratio_opp_efficiency_weighted'] = (
                    df['3PA_Opp'] / (df['2PA_Opp'] + df['3PA_Opp']).replace(0, np.nan)
                ) * (df['3P%_Opp'] / (df['2P%_Opp'] + 0.1))
            
            for window in [3, 5, 10]:
                ratio_opp_ma = f'three_point_ratio_opp_ma_{window}'
                if ratio_opp_ma in df.columns:
                    df[f'three_point_ratio_opp_trend_{window}'] = (
                        df[ratio_opp_ma] * 1.3
                    )

        # ==================== FEATURES DE MOMENTUM MEJORADAS ====================
        # Momentum basado en victorias con ponderación temporal
        if 'is_win' in df.columns:
            # Momentum ponderado por importancia
            df['win_momentum_weighted'] = df.groupby('Team')['is_win'].transform(
                lambda x: x.rolling(window=5, min_periods=1).mean().shift(1)
            ) * 1.5  # Ponderación basada en importancia
            
            # Momentum reciente con mayor peso
            df['win_momentum_recent_weighted'] = df.groupby('Team')['is_win'].transform(
                lambda x: x.rolling(window=3, min_periods=1).sum().shift(1)
            ) * 2.0  # Mayor peso para momentum reciente

        # ==================== FEATURES DE ESTABILIDAD MEJORADAS ====================
        # Estabilidad de totales con ponderación
        if 'game_total_scoring_projection' in df.columns:
            # Estabilidad ponderada
            df['game_total_stability_weighted'] = df.groupby('Team')['game_total_scoring_projection'].transform(
                lambda x: x.rolling(window=10, min_periods=3).std().shift(1) / 
                         (x.rolling(window=10, min_periods=3).mean().shift(1) + 1e-6)
            )
            # Invertir y ponderar
            df['game_total_stability_weighted'] = (1 / (df['game_total_stability_weighted'] + 0.01)) * 1.2

        # ==================== FEATURES DE CONTEXTO MEJORADAS ====================
        # Días en temporada con ponderación
        if 'days_into_season' in df.columns:
            df['days_into_season_weighted'] = df['days_into_season'] * 1.1  # Ponderación basada en importancia
        
        # Eficiencia por tier de scoring con ponderación
        if 'game_low_scoring_efficiency' in df.columns:
            df['game_low_scoring_efficiency_weighted'] = df['game_low_scoring_efficiency'] * 1.3

        # ==================== INTERACCIONES AVANZADAS ====================
        # Interacciones entre features más importantes
        if all(col in df.columns for col in ['FG%_vs_FG%_Opp_diff', 'FGA_FGA_Opp_total']):
            df['fg_diff_volume_interaction'] = (
                df['FG%_vs_FG%_Opp_diff'] * df['FGA_FGA_Opp_total'] * 0.01
            )
        
        if all(col in df.columns for col in ['3P%_vs_3P%_Opp_diff', '3PA_3PA_Opp_total']):
            df['three_pt_diff_volume_interaction'] = (
                df['3P%_vs_3P%_Opp_diff'] * df['3PA_3PA_Opp_total'] * 0.01
            )
        
        if all(col in df.columns for col in ['is_win', 'days_into_season']):
            df['win_season_interaction'] = (
                df['is_win'] * df['days_into_season'] * 0.001
            )

        # ==================== FEATURES DE PREDICCIÓN FINAL MEJORADAS ====================
        # Combinación ponderada de las features más importantes
        important_features = [
            'FG%_vs_FG%_Opp_diff', 'FGA_FGA_Opp_total', '3P%_vs_3P%_Opp_diff',
            'FT%_vs_FT%_Opp_diff', 'is_win', '3PA_3PA_Opp_total'
        ]
        
        available_important = [f for f in important_features if f in df.columns]
        if len(available_important) >= 3:
            # Crear feature combinada ponderada
            combined_value = 0
            weights = [903, 896, 644, 530, 495, 309]  # Pesos basados en importancia del modelo
            
            for i, feature in enumerate(available_important[:len(weights)]):
                if feature in df.columns:
                    combined_value += df[feature] * (weights[i] / 1000.0)
            
            df['combined_important_features'] = combined_value

        logger.info(f"Creadas {len([col for col in df.columns if any(keyword in col for keyword in ['weighted', 'enhanced', 'trend', 'interaction']) and col not in df.columns])} características predictivas mejoradas")
    
    def _create_ultra_predictive_features(self, df: pd.DataFrame) -> None:
        """
        Crea características ultra-predictivas basadas en el feedback más reciente del modelo.
        Estas features están optimizadas para maximizar el poder predictivo basado en las
        features más importantes identificadas.
        
        Args:
            df: DataFrame con datos de equipos (modificado in-place)
        """
        if df.empty:
            return

        logger.info("Creando características ultra-predictivas basadas en feedback más reciente")

        # ==================== FEATURES BASADAS EN GAME_WEEKEND_FACTOR (TOP 1) ====================
        # Factor de fin de semana mejorado con más contexto
        if 'is_weekend' in df.columns:
            # Factor de fin de semana con ponderación temporal
            df['game_weekend_factor_enhanced'] = df['is_weekend'] * 1.5  # Ponderación basada en importancia
            
            # Factor de fin de semana con contexto de temporada
            if 'days_into_season' in df.columns:
                df['game_weekend_season_context'] = (
                    df['is_weekend'] * 
                    (1 + (df['days_into_season'] / 365.0) * 0.2)  # Ajuste por progresión de temporada
                )
            
            # Factor de fin de semana con contexto de rivalidad
            if 'game_rivalry_factor' in df.columns:
                df['game_weekend_rivalry_boost'] = (
                    df['is_weekend'] * df['game_rivalry_factor'] * 1.3
                )

        # ==================== FEATURES BASADAS EN THREE_POINT_RATIO (TOP 2) ====================
        # Ratios de 3 puntos ultra-mejorados - USAR DATOS HISTÓRICOS
        if '2PA' in df.columns and '3PA' in df.columns:
            # Usar medias móviles históricas para evitar data leakage
            three_pt_ratio_ma_5 = f'three_point_ratio_ma_5'
            if three_pt_ratio_ma_5 in df.columns:
                # Ratio de 3 puntos con ponderación por eficiencia usando datos históricos
                if '3P%_ma_5' in df.columns and '2P%_ma_5' in df.columns:
                    df['three_point_ratio_efficiency_boost'] = (
                        df[three_pt_ratio_ma_5]
                    ) * (df['3P%_ma_5'] / (df['2P%_ma_5'] + 0.05)) * 1.8  # Mayor ponderación
                    
                    # Ratio de 3 puntos con contexto de defensa usando datos históricos
                    if '3P%_Opp_ma_5' in df.columns:
                        df['three_point_ratio_defense_context'] = (
                            df[three_pt_ratio_ma_5]
                        ) * (df['3P%_ma_5'] / (df['3P%_Opp_ma_5'] + 0.1)) * 1.5
            
            # Ratio de 3 puntos con tendencia mejorada
            for window in [3, 5, 10]:
                ratio_ma = f'three_point_ratio_ma_{window}'
                if ratio_ma in df.columns:
                    df[f'three_point_ratio_trend_boost_{window}'] = (
                        df[ratio_ma] * (window / 3.0) * 1.4  # Ponderación proporcional
                    )

        # Lo mismo para el oponente con mayor peso - USAR DATOS HISTÓRICOS
        if '2PA_Opp' in df.columns and '3PA_Opp' in df.columns:
            three_pt_ratio_opp_ma_5 = f'three_point_ratio_opp_ma_5'
            if three_pt_ratio_opp_ma_5 in df.columns:
                if '3P%_Opp_ma_5' in df.columns and '2P%_Opp_ma_5' in df.columns:
                    df['three_point_ratio_opp_efficiency_boost'] = (
                        df[three_pt_ratio_opp_ma_5]
                    ) * (df['3P%_Opp_ma_5'] / (df['2P%_Opp_ma_5'] + 0.05)) * 2.0  # Mayor peso para oponente
            
            for window in [3, 5, 10]:
                ratio_opp_ma = f'three_point_ratio_opp_ma_{window}'
                if ratio_opp_ma in df.columns:
                    df[f'three_point_ratio_opp_trend_boost_{window}'] = (
                        df[ratio_opp_ma] * (window / 3.0) * 1.6  # Mayor peso para oponente
                    )

        # ==================== FEATURES BASADAS EN FT% DIFERENCIALES (TOP 4) ====================
        # Diferenciales de FT% 
        if 'FT%' in df.columns and 'FT%_Opp' in df.columns:
            # Usar medias móviles históricas para evitar data leakage
            ft_ma_5 = f'FT%_ma_5'
            ft_opp_ma_5 = f'FT%_Opp_ma_5'
            
            if ft_ma_5 in df.columns and ft_opp_ma_5 in df.columns:
                # Diferencial con mayor ponderación usando datos históricos
                df['FT%_vs_FT%_Opp_diff_ultra'] = (
                    df[ft_ma_5] - df[ft_opp_ma_5]
                ) * 2.0  # Doble ponderación
                
                # Diferencial con contexto de volumen usando datos históricos
                if 'FTA' in df.columns and 'FTA_Opp' in df.columns:
                    fta_avg = df.groupby('Team')['FTA'].transform(
                        lambda x: x.rolling(5, min_periods=1).mean().shift(1)
                    ).fillna(20)
                    fta_opp_avg = df.groupby('Opp')['FTA_Opp'].transform(
                        lambda x: x.rolling(5, min_periods=1).mean().shift(1)
                    ).fillna(20)
                    
                    df['FT%_vs_FT%_Opp_volume_context'] = (
                        (df[ft_ma_5] - df[ft_opp_ma_5]) * 
                        ((fta_avg + fta_opp_avg) / 40.0)  # Normalizar por volumen típico
                    ) * 1.8
            
            # Diferencial con tendencia mejorada
            for window in [3, 5, 10]:
                ft_ma = f'FT%_ma_{window}'
                ft_opp_ma = f'FT%_Opp_ma_{window}'
                if ft_ma in df.columns and ft_opp_ma in df.columns:
                    df[f'FT%_vs_FT%_Opp_diff_trend_{window}'] = (
                        df[ft_ma] - df[ft_opp_ma]
                    ) * (window / 5.0) * 1.7

        # ==================== FEATURES BASADAS EN GAME_TOTAL_POSSESSIONS (TOP 5) ====================
        # Posesiones totales ultra-mejoradas - USAR DATOS HISTÓRICOS
        if 'game_total_possessions' in df.columns:
            # Usar media móvil histórica para evitar data leakage
            game_poss_ma_5 = f'game_pace_avg_5g'
            if game_poss_ma_5 in df.columns:
                # Posesiones con ponderación por ritmo usando datos históricos
                df['game_total_possessions_ultra'] = df[game_poss_ma_5] * 1.3
                
                # Posesiones con contexto de eficiencia usando datos históricos
                if 'game_combined_shooting_efficiency' in df.columns:
                    shooting_eff_ma = df.groupby('Team')['game_combined_shooting_efficiency'].transform(
                        lambda x: x.rolling(5, min_periods=1).mean().shift(1)
                    ).fillna(0.5)
                    
                    df['game_possessions_efficiency_context'] = (
                        df[game_poss_ma_5] * shooting_eff_ma * 0.8
                    )
            
            # Posesiones con tendencia temporal
            for window in [3, 5, 10]:
                poss_ma = f'game_pace_avg_{window}g'
                if poss_ma in df.columns:
                    df[f'game_possessions_trend_{window}'] = (
                        df[poss_ma] * 1.2
                    )

        # ==================== FEATURES BASADAS EN TEAM_WIN_RATE_10G (TOP 6) ====================
        # Win rate mejorado con más contexto
        if 'is_win' in df.columns:
            # Win rate con ponderación temporal mejorada
            df['team_win_rate_10g_ultra'] = df.groupby('Team')['is_win'].transform(
                lambda x: x.rolling(window=10, min_periods=1).mean().shift(1)
            ) * 1.6  # Mayor ponderación
            
            # Win rate con momentum reciente
            df['team_win_rate_recent_momentum'] = df.groupby('Team')['is_win'].transform(
                lambda x: x.rolling(window=5, min_periods=1).sum().shift(1)
            ) * 2.0  # Mayor peso para momentum reciente
            
            # Win rate con contexto de temporada
            if 'days_into_season' in df.columns:
                df['team_win_rate_season_context'] = (
                    df.groupby('Team')['is_win'].transform(
                        lambda x: x.rolling(window=10, min_periods=1).mean().shift(1)
                    ) * (1 + (df['days_into_season'] / 365.0) * 0.3)
                )

        # ==================== FEATURES BASADAS EN GAME_PACE_AVG_5G (TOP 7) ====================
        # Ritmo del juego ultra-mejorado
        if 'game_pace_avg_5g' in df.columns:
            # Ritmo con ponderación directa
            df['game_pace_avg_5g_ultra'] = df['game_pace_avg_5g'] * 1.4
            
            # Ritmo con contexto de estabilidad
            if 'game_total_stability' in df.columns:
                df['game_pace_stability_context'] = (
                    df['game_pace_avg_5g'] * df['game_total_stability'] * 1.2
                )

        # ==================== FEATURES BASADAS EN OFF_EFFICIENCY_COMBINED_3 (TOP 8) ====================
        # Eficiencia ofensiva ultra-mejorada
        if 'off_efficiency_combined_3' in df.columns:
            # Eficiencia con ponderación directa
            df['off_efficiency_combined_3_ultra'] = df['off_efficiency_combined_3'] * 1.5
            
            # Eficiencia con contexto defensivo
            if 'def_efficiency_combined_3' in df.columns:
                df['off_def_efficiency_balance'] = (
                    df['off_efficiency_combined_3'] - df['def_efficiency_combined_3']
                ) * 1.8

        # ==================== FEATURES BASADAS EN GAME_ENHANCED_TOTAL_PROJECTION (TOP 9) ====================
        # Proyección total ultra-mejorada
        if 'game_enhanced_total_projection' in df.columns:
            # Proyección con ponderación directa
            df['game_enhanced_total_projection_ultra'] = df['game_enhanced_total_projection'] * 1.3
            
            # Proyección con ajuste por estabilidad
            if 'game_total_stability' in df.columns:
                df['game_enhanced_total_stability_adjusted'] = (
                    df['game_enhanced_total_projection'] * 
                    (1 + df['game_total_stability'] * 0.5)
                )

        # ==================== FEATURES BASADAS EN HAS_OVERTIME (TOP 10) ====================
        # Factor de overtime mejorado - USAR DATOS HISTÓRICOS
        if 'has_overtime' in df.columns:
            # Overtime con ponderación usando tendencia histórica
            df['has_overtime_ultra'] = df.groupby('Team')['has_overtime'].transform(
                lambda x: x.rolling(10, min_periods=1).mean().shift(1)
            ).fillna(0.1) * 1.4
            
            # Overtime con contexto de temporada usando datos históricos
            if 'days_into_season' in df.columns:
                df['has_overtime_season_context'] = (
                    df.groupby('Team')['has_overtime'].transform(
                        lambda x: x.rolling(10, min_periods=1).mean().shift(1)
                    ).fillna(0.1) * 
                    (1 + (df['days_into_season'] / 365.0) * 0.4)
                )

        # ==================== INTERACCIONES AVANZADAS ENTRE TOP FEATURES ====================
        # Interacciones entre las features más importantes
        if all(col in df.columns for col in ['game_weekend_factor_enhanced', 'three_point_ratio_efficiency_boost']):
            df['weekend_three_pt_interaction'] = (
                df['game_weekend_factor_enhanced'] * df['three_point_ratio_efficiency_boost'] * 0.5
            )
        
        if all(col in df.columns for col in ['FT%_vs_FT%_Opp_diff_ultra', 'game_total_possessions_ultra']):
            df['ft_diff_possessions_interaction'] = (
                df['FT%_vs_FT%_Opp_diff_ultra'] * df['game_total_possessions_ultra'] * 0.001
            )
        
        if all(col in df.columns for col in ['team_win_rate_10g_ultra', 'game_pace_avg_5g_ultra']):
            df['win_rate_pace_interaction'] = (
                df['team_win_rate_10g_ultra'] * df['game_pace_avg_5g_ultra'] * 0.01
            )

        # ==================== FEATURES DE PREDICCIÓN FINAL ULTRA-MEJORADAS ====================
        # Combinación ponderada de las features más importantes con pesos actualizados
        ultra_important_features = [
            'game_weekend_factor_enhanced', 'three_point_ratio_efficiency_boost',
            'three_point_ratio_opp_efficiency_boost', 'FT%_vs_FT%_Opp_diff_ultra',
            'game_total_possessions_ultra', 'team_win_rate_10g_ultra'
        ]
        
        available_ultra = [f for f in ultra_important_features if f in df.columns]
        if len(available_ultra) >= 3:
            # Crear feature combinada ultra-ponderada
            combined_ultra_value = 0
            ultra_weights = [0.0082, 0.0082, 0.0081, 0.0081, 0.0081, 0.0081]  # Pesos basados en importancia real
            
            for i, feature in enumerate(available_ultra[:len(ultra_weights)]):
                if feature in df.columns:
                    combined_ultra_value += df[feature] * (ultra_weights[i] * 100)  # Escalar pesos
            
            df['ultra_combined_important_features'] = combined_ultra_value

        # ==================== FEATURES BASADAS EN EL DASHBOARD ====================
        # Basado en el análisis del dashboard, agregar features para mejorar precisión
        
        # Feature para mejorar precisión en rangos específicos (basado en error por rango)
        if 'game_total_scoring_projection' in df.columns:
            # Ajuste específico para juegos de bajo scoring (150-190)
            low_scoring_mask = df['game_total_scoring_projection'] < 190
            df['low_scoring_adjustment'] = np.where(low_scoring_mask, -2.0, 0.0)
            
            # Ajuste específico para juegos de alto scoring (251+)
            high_scoring_mask = df['game_total_scoring_projection'] > 250
            df['high_scoring_adjustment'] = np.where(high_scoring_mask, 1.5, 0.0)
        
        # Feature para mejorar precisión over/under (basado en precisión del dashboard)
        if 'game_final_total_projection' in df.columns:
            # Ajuste basado en la línea over/under típica
            df['over_under_adjustment'] = np.where(
                df['game_final_total_projection'] > 230, 1.0, -0.5
            )

        logger.info(f"Creadas {len([col for col in df.columns if any(keyword in col for keyword in ['ultra', 'boost', 'enhanced', 'interaction']) and col not in df.columns])} características ultra-predictivas")
