"""
Módulo de Características para Predicción de Victorias (is_win)
==============================================================

Este módulo contiene toda la lógica de ingeniería de características específica
para la predicción de victorias de un equipo NBA por partido. Implementa características
avanzadas enfocadas en factores que determinan el resultado de un partido.

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

class IsWinFeatureEngineer:
    """
    Motor de features para predicción de victoria/derrota usando ESTADÍSTICAS HISTÓRICAS
    OPTIMIZADO - Rendimiento pasado para predecir juegos futuros
    """
    
    def __init__(self, lookback_games: int = 10):
        """Inicializa el ingeniero de características para predicción de victorias."""
        self.lookback_games = lookback_games
        self.scaler = StandardScaler()
        self.feature_columns = []
        # Cache para evitar recálculos
        self._cached_calculations = {}
        # Cache para features generadas
        self._features_cache = {}
        self._last_data_hash = None
        
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
            df.sort_values(['Team', 'Date'], inplace=True)
            df.reset_index(drop=True, inplace=True)
            logger.debug("Datos ordenados cronológicamente por equipo")
    
    def _calculate_basic_temporal_features(self, df: pd.DataFrame) -> None:
        """Método auxiliar para calcular features temporales básicas una sola vez"""
        if 'Date' in df.columns:
            # Calcular una sola vez todas las features temporales
            df['days_rest'] = df.groupby('Team')['Date'].diff().dt.days.fillna(2)
            df['day_of_week'] = df['Date'].dt.dayofweek
            df['month'] = df['Date'].dt.month
            df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
            
            # Días en temporada
            season_start = df['Date'].min()
            df['days_into_season'] = (df['Date'] - season_start).dt.days
            
            # Back-to-back indicator (calculado una sola vez)
            df['is_back_to_back'] = (df['days_rest'] <= 1).astype(int)
            
            logger.debug("Features temporales básicas calculadas")
    
    def _calculate_home_away_features(self, df: pd.DataFrame) -> None:
        """Método auxiliar para calcular features de ventaja local una sola vez"""
        if 'is_home' not in df.columns:
            logger.debug("is_home no encontrado del data_loader - features de ventaja local no disponibles")
            return
        
        logger.debug("Usando is_home del data_loader para features de ventaja local")
        
        # Calcular features relacionadas con ventaja local
        df['home_advantage'] = df['is_home'] * 0.06  # 6% boost histórico NBA
        df['travel_penalty'] = np.where(df['is_home'] == 0, -0.02, 0.0)
        
        # Back-to-back road penalty (usando is_back_to_back ya calculado)
        if 'is_back_to_back' in df.columns:
            df['road_b2b_penalty'] = np.where(
                (df['is_home'] == 0) & (df['is_back_to_back'] == 1), -0.04, 0.0
            )
    
    def generate_all_features(self, df: pd.DataFrame) -> List[str]:
        """
        PIPELINE SIMPLIFICADO DE FEATURES ANTI-OVERFITTING
        Usar solo estadísticas básicas históricas - MENOS COMPLEJIDAD
        CON CACHE para evitar regeneración innecesaria
        """
        
        # Verificar cache primero
        data_hash = self._get_data_hash(df)
        if data_hash == self._last_data_hash and self.feature_columns:
            logger.debug("Usando features desde cache (sin regenerar)")
            return self.feature_columns
        
        # Solo mostrar log principal la primera vez o cuando cambian los datos
        if self._last_data_hash is None:
            logger.info("Generando features NBA SIMPLIFICADAS anti-overfitting...")
        else:
            logger.debug("Regenerando features (datos cambiaron)")

        # VERIFICACIÓN DE is_win COMO TARGET (ya viene del dataset)
        if 'is_win' in df.columns:
            win_distribution = df['is_win'].value_counts().to_dict()
            if self._last_data_hash is None:  # Solo log la primera vez
                logger.info(f"Target is_win disponible - Distribución: {win_distribution}")
        else:
            NBALogger.log_error(logger, "is_win no encontrado en el dataset - requerido para features de victoria")
            return []
        
        # VERIFICAR FEATURES DEL DATA_LOADER (consolidado en un solo mensaje)
        data_loader_features = ['is_home', 'has_overtime', 'overtime_periods']
        available_features = [f for f in data_loader_features if f in df.columns]
        missing_features = [f for f in data_loader_features if f not in df.columns]
        
        if self._last_data_hash is None:  # Solo log la primera vez
            if available_features:
                logger.info(f"Features del data_loader: {len(available_features)}/{len(data_loader_features)} disponibles")
            if missing_features:
                logger.debug(f"Features faltantes: {missing_features}")
        
        # Trabajar directamente con el DataFrame
        if df.empty:
            return []
        
        # PASO 0: Preparación básica (una sola vez)
        self._ensure_datetime_and_sort(df)
        self._calculate_basic_temporal_features(df)
        self._calculate_home_away_features(df)
        
        if self._last_data_hash is None:  # Solo log la primera vez
            logger.info("Iniciando generación de features SIMPLIFICADAS...")
        
        # SOLO FEATURES BÁSICAS ESENCIALES - NO ULTRA COMPLEJAS
        self._create_temporal_features_simple(df)
        self._create_contextual_features_simple(df)
        self._create_performance_features_simple(df)
        self._create_efficiency_features_simple(df)
        self._create_win_features_simple(df)
        self._create_opponent_features_simple(df)
        
        # Actualizar lista de features disponibles
        self._update_feature_columns(df)
        
        # Compilar lista de características BÁSICAS ÚNICAMENTE
        all_features = [col for col in df.columns if col not in [
            # Columnas básicas del dataset
            'Team', 'Date', 'Away', 'Opp', 'Result', 'MP',
            # Estadísticas del juego actual (NO USAR - data leakage)
            'FG', 'FGA', 'FG%', '2P', '2PA', '2P%', '3P', '3PA', '3P%', 
            'FT', 'FTA', 'FT%', 'PTS', 'ORB', 'DRB', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF',
            'FG_Opp', 'FGA_Opp', 'FG%_Opp', '2P_Opp', '2PA_Opp', '2P%_Opp', 
            '3P_Opp', '3PA_Opp', '3P%_Opp', 'FT_Opp', 'FTA_Opp', 'FT%_Opp', 'PTS_Opp',
            'ORB_Opp', 'DRB_Opp', 'TRB_Opp', 'AST_Opp', 'STL_Opp', 'BLK_Opp', 'TOV_Opp', 'PF_Opp',
            # Target variable
            'is_win',
            # Columnas auxiliares temporales
            'day_of_week', 'month', 'days_rest', 'is_home', 'is_away_numeric'
        ]]
        
        # FILTRAR SOLO FEATURES ESENCIALES - MÁXIMO 30 FEATURES
        essential_features = []
        
        # PRIORIDAD 1: Features de win rate básicas (máximo 8)
        win_features = [col for col in all_features if any(keyword in col for keyword in [
            'team_win_rate_5g', 'team_win_rate_10g', 'weighted_win_rate_5g', 'weighted_win_rate_10g',
            'home_win_rate_10g', 'away_win_rate_10g', 'win_momentum_5g', 'win_momentum_10g'
        ])]
        essential_features.extend(win_features[:8])
        
        # PRIORIDAD 2: Features de rendimiento básicas (máximo 8)
        performance_features = [col for col in all_features if any(keyword in col for keyword in [
            'pts_hist_avg_5g', 'pts_hist_avg_10g', 'pts_opp_hist_avg_5g', 'pts_opp_hist_avg_10g',
            'point_diff_hist_avg_5g', 'point_diff_hist_avg_10g', 'pts_consistency_5g', 'pts_consistency_10g'
        ])]
        essential_features.extend([f for f in performance_features[:8] if f not in essential_features])
        
        # PRIORIDAD 3: Features de eficiencia básicas (máximo 6)
        efficiency_features = [col for col in all_features if any(keyword in col for keyword in [
            'fg_pct_hist_avg_5g', 'fg_pct_hist_avg_10g', '3p_pct_hist_avg_5g', 
            '3p_pct_hist_avg_10g', 'ft_pct_hist_avg_5g', 'ft_pct_hist_avg_10g'
        ])]
        essential_features.extend([f for f in efficiency_features[:6] if f not in essential_features])
        
        # PRIORIDAD 4: Features contextuales básicas (máximo 6)
        contextual_features = [col for col in all_features if any(keyword in col for keyword in [
            'is_back_to_back', 'days_rest', 'home_advantage', 'travel_penalty', 
            'road_b2b_penalty', 'is_weekend'
        ])]
        essential_features.extend([f for f in contextual_features[:6] if f not in essential_features])
        
        # PRIORIDAD 5: Features de momentum y tendencias (máximo 4)
        momentum_features = [col for col in all_features if any(keyword in col for keyword in [
            'win_streak', 'loss_streak', 'recent_form', 'momentum_score'
        ])]
        essential_features.extend([f for f in momentum_features[:4] if f not in essential_features])
        
        # PRIORIDAD 6: Features de oponente básicas (máximo 4)
        opponent_features = [col for col in all_features if any(keyword in col for keyword in [
            'opponent_recent_form', 'opponent_season_record', 'last_vs_opp_result', 'revenge_motivation'
        ])]
        essential_features.extend([f for f in opponent_features[:4] if f not in essential_features])
        
        # PRIORIDAD 7: CARACTERÍSTICAS BÁSICAS ADICIONALES (máximo 4)
        additional_features = [col for col in all_features if any(keyword in col for keyword in [
            'energy_factor', 'altitude_advantage', 'rest_advantage', 'season_fatigue_factor'
        ])]
        essential_features.extend([f for f in additional_features[:4] if f not in essential_features])
        
        # FILTRADO DE FEATURES QUE GENERAN RUIDO
        filtered_features = self._filter_noisy_features(essential_features, df)
        
        # AÑADIR FEATURES CONTEXTUALES DE ALTA PRECISIÓN
        precision_features = self._create_precision_features(df)
        
        # Combinar features filtradas con features de precisión
        combined_features = filtered_features + [f for f in precision_features if f not in filtered_features]
        
        # Límite de características optimizadas para precisión
        final_features = combined_features[:28]
        
        # Actualizar la lista final
        self.feature_columns = final_features
        
        # Actualizar cache
        self._last_data_hash = data_hash
        
        # Limpiar cache para liberar memoria
        self._clear_cache()

        # Solo log detallado la primera vez
        if data_hash != self._last_data_hash or len(final_features) != len(self._features_cache.get('last_features', [])):
            logger.info(f"Features SIMPLIFICADAS generadas: {len(final_features)} características (máximo 30)")
            logger.info(f"Pipeline simplificado para evitar inconsistencias entre entrenamiento y predicción")
            self._features_cache['last_features'] = final_features.copy()
        
        return self.feature_columns
    
    def _create_temporal_features_simple(self, df: pd.DataFrame) -> None:
        """Features temporales básicas disponibles antes del juego"""
        # Solo agregar features adicionales aquí
        if 'days_rest' in df.columns:
            # Factor de energía basado en descanso
            df['energy_factor'] = np.where(
                df['days_rest'] == 0, 0.85,  # Back-to-back penalty
                np.where(df['days_rest'] == 1, 0.92,  # 1 día
                        np.where(df['days_rest'] >= 3, 1.08, 1.0))  # 3+ días boost
            )
    
    def _create_contextual_features_simple(self, df: pd.DataFrame) -> None:
        """Features contextuales disponibles antes del juego"""
        # Las features básicas de home/away ya fueron calculadas en _calculate_home_away_features
        
        # Ventaja de altitud para equipos específicos
        altitude_teams = ['DEN', 'UTA', 'PHX']
        df['altitude_advantage'] = df['Team'].apply(
            lambda x: 0.025 if x in altitude_teams else 0.0
        )
        
        # Rest advantage específico (usando days_rest ya calculado)
        if 'days_rest' in df.columns:
            df['rest_advantage'] = np.where(
                df['days_rest'] == 0, -0.15,  # Penalización back-to-back
                np.where(df['days_rest'] == 1, -0.05,
                        np.where(df['days_rest'] >= 3, 0.08, 0.0))
            )
        
        # Season fatigue factor
        if 'month' in df.columns:
            df['season_fatigue_factor'] = np.where(
                df['month'].isin([1, 2, 3]), -0.015,  # Fatiga final temporada
                np.where(df['month'].isin([11, 12]), 0.01, 0.0)  # Boost inicio
            )
        
        # Weekend boost
        if 'is_weekend' in df.columns:
            df['weekend_boost'] = df['is_weekend'] * 0.01
    
    def _create_performance_features_simple(self, df: pd.DataFrame) -> None:
        """Features de rendimiento BÁSICAS únicamente - ANTI-OVERFITTING"""
        # Solo ventanas básicas: 5 y 10 juegos
        basic_windows = [5, 10]
        
        for window in basic_windows:
            # Puntos históricos básicos
            pts_hist_avg = self._get_historical_series(df, 'PTS', window, 'mean')
            df[f'pts_hist_avg_{window}g'] = pts_hist_avg
            
            pts_opp_hist_avg = self._get_historical_series(df, 'PTS_Opp', window, 'mean')
            df[f'pts_opp_hist_avg_{window}g'] = pts_opp_hist_avg
            
            # Diferencial básico de puntos
            df[f'point_diff_hist_avg_{window}g'] = pts_hist_avg - pts_opp_hist_avg
            
            # Consistencia básica en puntos
            pts_std = self._get_historical_series(df, 'PTS', window, 'std', min_periods=2)
            df[f'pts_consistency_{window}g'] = 1 / (pts_std.fillna(1) + 1)
    
    def _create_efficiency_features_simple(self, df: pd.DataFrame) -> None:
        """Features de eficiencia BÁSICAS únicamente - ANTI-OVERFITTING"""
        # Solo ventanas básicas: 5 y 10 juegos
        basic_windows = [5, 10]
        
        # Solo estadísticas básicas de tiros
        shooting_stats = ['FG%', '3P%', 'FT%']
        for stat in shooting_stats:
            if stat in df.columns:
                for window in basic_windows:
                    # Porcentajes históricos básicos
                    stat_hist_avg = self._get_historical_series(df, stat, window, 'mean')
                    df[f'{stat.lower().replace("%", "_pct")}_hist_avg_{window}g'] = stat_hist_avg
    
    def _create_win_features_simple(self, df: pd.DataFrame) -> None:
        """Features de victoria BÁSICAS únicamente - ANTI-OVERFITTING"""
        # Solo ventanas básicas: 5 y 10 juegos
        basic_windows = [5, 10]
        
        for window in basic_windows:
            # Win rate histórico básico
            df[f'team_win_rate_{window}g'] = (
                df.groupby('Team')['is_win'].shift(1)
                .rolling(window=window, min_periods=1).mean()
            ).fillna(0.5)
            
            # Weighted win rate básico (solo para ventana de 5)
            if window == 5:
                wins_shifted = df.groupby('Team')['is_win'].shift(1).fillna(0)
                
                def simple_weighted_mean(x):
                    try:
                        x_clean = pd.to_numeric(x, errors='coerce').dropna()
                        if len(x_clean) == 0:
                            return 0.5
                        # Pesos simples: más reciente = más peso
                        weights = np.linspace(0.5, 1.0, len(x_clean))
                        weights = weights / weights.sum()
                        return float(np.average(x_clean, weights=weights))
                    except:
                        return 0.5
                
                df[f'weighted_win_rate_{window}g'] = (
                    wins_shifted.rolling(window=window, min_periods=1)
                    .apply(simple_weighted_mean, raw=False)
                )
                
                # Win momentum básico
                if window >= 5:
                    first_half = wins_shifted.rolling(window=3, min_periods=1).mean()
                    second_half = wins_shifted.shift(2).rolling(window=3, min_periods=1).mean()
                    df[f'win_momentum_{window}g'] = first_half - second_half
        
        # Home/Away win rates básicos - SIMPLIFICADO para evitar errores
        if 'is_home' in df.columns:
            window = 10
            
            # Método simplificado: usar win rate general con ajuste por contexto
            general_win_rate = df.groupby('Team')['is_win'].shift(1).rolling(window=window, min_periods=1).mean().fillna(0.5)
            
            # Home win rate: win rate general + boost de casa
            df[f'home_win_rate_{window}g'] = general_win_rate + 0.05  # 5% boost histórico
            df[f'home_win_rate_{window}g'] = df[f'home_win_rate_{window}g'].clip(0, 1)  # Mantener en rango [0,1]
            
            # Away win rate: win rate general - penalización visitante
            df[f'away_win_rate_{window}g'] = general_win_rate - 0.05  # 5% penalización
            df[f'away_win_rate_{window}g'] = df[f'away_win_rate_{window}g'].clip(0, 1)  # Mantener en rango [0,1]
            
        else:
            # Si no hay información de is_home, crear features con valores neutros
            window = 10
            df[f'home_win_rate_{window}g'] = 0.5
            df[f'away_win_rate_{window}g'] = 0.5
    
    def _create_opponent_features_simple(self, df: pd.DataFrame) -> None:
        """Features de oponente BÁSICAS únicamente - ANTI-OVERFITTING"""
        if 'Opp' not in df.columns or 'is_win' not in df.columns:
            return
            
        # Recent form del oponente básico
        opp_recent_form = df.groupby('Opp')['is_win'].transform(
            lambda x: x.shift(1).rolling(5, min_periods=2).mean()
        )
        df['opponent_recent_form'] = 1 - opp_recent_form.fillna(0.5)
        
        # Win rate del oponente en temporada básico
        opp_season_record = df.groupby('Opp')['is_win'].transform(
            lambda x: x.shift(1).expanding().mean()
        )
        df['opponent_season_record'] = 1 - opp_season_record.fillna(0.5)
        
        # Último resultado vs este oponente
        df['last_vs_opp_result'] = df.groupby(['Team', 'Opp'])['is_win'].transform(
            lambda x: x.shift(1).tail(1).iloc[0] if len(x.shift(1).dropna()) > 0 else 0.5
        ).fillna(0.5)
        
        # Revenge factor básico
        df['revenge_motivation'] = np.where(
            df['last_vs_opp_result'] == 0, 0.04,  # Perdieron último vs este rival
            np.where(df['last_vs_opp_result'] == 1, -0.02, 0.0)  # Ganaron último vs este rival
        )
        
    def _create_precision_features(self, df: pd.DataFrame) -> List[str]:
        """
        Crea features contextuales específicas para mejorar la precisión en situaciones críticas
        """
        precision_features = []
        
        # 1. CLUTCH TIME PERFORMANCE (últimos 5 minutos)
        if 'pts_hist_avg_5g' in df.columns and 'pts_opp_hist_avg_5g' in df.columns:
            # Capacidad de cerrar juegos
            df['clutch_factor'] = np.where(
                (df['pts_hist_avg_5g'] - df['pts_opp_hist_avg_5g']) > 3, 0.08,  # Dominantes
                np.where((df['pts_hist_avg_5g'] - df['pts_opp_hist_avg_5g']) < -3, -0.08, 0.0)  # Débiles
            )
            precision_features.append('clutch_factor')
        
        # 2. MOMENTUM SHIFT INDICATOR
        if 'team_win_rate_5g' in df.columns and 'team_win_rate_10g' in df.columns:
            # Tendencia reciente vs histórica
            df['momentum_shift'] = df['team_win_rate_5g'] - df['team_win_rate_10g']
            precision_features.append('momentum_shift')
        
        # 3. DEFENSIVE PRESSURE RATING
        if 'pts_opp_hist_avg_10g' in df.columns:
            # Calidad defensiva vs promedio liga (asumiendo ~110 pts/juego)
            league_avg_pts = 110
            df['defensive_pressure'] = (league_avg_pts - df['pts_opp_hist_avg_10g']) / 20.0  # Normalizado
            precision_features.append('defensive_pressure')
        
        # 4. CONSISTENCY UNDER PRESSURE
        if 'pts_consistency_5g' in df.columns and 'opponent_recent_form' in df.columns:
            # Consistencia contra equipos fuertes
            df['pressure_consistency'] = df['pts_consistency_5g'] * (1 - df['opponent_recent_form'])
            precision_features.append('pressure_consistency')
        
        # 5. STRATEGIC ADVANTAGE
        if 'home_advantage' in df.columns and 'travel_penalty' in df.columns:
            # Ventaja estratégica combinada
            df['strategic_edge'] = df['home_advantage'] - df['travel_penalty']
            precision_features.append('strategic_edge')
        
        # 6. MOTIVATION FACTOR
        if 'revenge_motivation' in df.columns and 'last_vs_opp_result' in df.columns:
            # Factor motivacional intensificado
            df['motivation_intensity'] = df['revenge_motivation'] * (1 + abs(0.5 - df['last_vs_opp_result']))
            precision_features.append('motivation_intensity')
        
        # 7. FORM DIFFERENTIAL
        if 'team_win_rate_5g' in df.columns and 'opponent_recent_form' in df.columns:
            # Diferencia de forma entre equipos
            df['form_advantage'] = df['team_win_rate_5g'] - (1 - df['opponent_recent_form'])
            precision_features.append('form_advantage')
        
        logger.info(f"Features de precisión creadas: {len(precision_features)}")
        return precision_features
    
    def _update_feature_columns(self, df: pd.DataFrame):
        """Actualizar lista de columnas de features históricas"""
        exclude_cols = [
            # Columnas básicas del dataset
            'Team', 'Date', 'Away', 'Opp', 'Result', 'MP',
            
            # Estadísticas del juego actual (usadas solo para crear historial)
            'FG', 'FGA', 'FG%', '2P', '2PA', '2P%', '3P', '3PA', '3P%', 
            'FT', 'FTA', 'FT%', 'PTS', 'ORB', 'DRB', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF',
            'FG_Opp', 'FGA_Opp', 'FG%_Opp', '2P_Opp', '2PA_Opp', '2P%_Opp', 
            '3P_Opp', '3PA_Opp', '3P%_Opp', 'FT_Opp', 'FTA_Opp', 'FT%_Opp', 'PTS_Opp',
            'ORB_Opp', 'DRB_Opp', 'TRB_Opp', 'AST_Opp', 'STL_Opp', 'BLK_Opp', 'TOV_Opp', 'PF_Opp',
            # Target variable
            'is_win',
            # Columnas auxiliares temporales
            'day_of_week', 'month', 'days_rest', 'is_home', 'is_away_numeric'
        ]
        
        self.feature_columns = [col for col in df.columns if col not in exclude_cols]
    
    def get_feature_importance_groups(self) -> Dict[str, List[str]]:
        """Retorna las características agrupadas por categoría HISTÓRICAS."""
        groups = {
            'temporal_context': [
                'day_of_week', 'month', 'is_weekend', 'days_into_season',
                'days_rest', 'energy_factor', 'season_fatigue_factor'
            ],
            
            'home_advantage': [
                'is_home', 'home_advantage', 'travel_penalty', 'road_b2b_penalty',
                'altitude_advantage', 'weekend_boost'
            ],
            
            'performance_historical': [
                'net_rating_hist_avg_3g', 'net_rating_hist_avg_5g', 'net_rating_hist_avg_7g', 'net_rating_hist_avg_10g',
                'point_diff_hist_avg_3g', 'point_diff_hist_avg_5g', 'point_diff_hist_avg_7g', 'point_diff_hist_avg_10g',
                'off_rating_hist_avg_3g', 'off_rating_hist_avg_5g', 'off_rating_hist_avg_7g', 'off_rating_hist_avg_10g',
                'def_rating_hist_avg_3g', 'def_rating_hist_avg_5g', 'def_rating_hist_avg_7g', 'def_rating_hist_avg_10g'
            ],
            
            'efficiency_historical': [
                'fg_diff_hist_avg_3g', 'fg_diff_hist_avg_5g', 'fg_diff_hist_avg_7g',
                'three_diff_hist_avg_3g', 'three_diff_hist_avg_5g', 'three_diff_hist_avg_7g',
                'ft_diff_hist_avg_3g', 'ft_diff_hist_avg_5g', 'ft_diff_hist_avg_7g',
                'ts_diff_hist_avg_3g', 'ts_diff_hist_avg_5g', 'ts_diff_hist_avg_7g',
                'efg_diff_hist_avg_3g', 'efg_diff_hist_avg_5g', 'efg_diff_hist_avg_7g'
            ],
            
            'momentum_factors': [
                'team_win_rate_3g', 'team_win_rate_5g', 'team_win_rate_7g', 'team_win_rate_10g', 'team_win_rate_15g',
                'recent_form', 'win_streak', 'last_3_wins', 'power_score'
            ],
            
            'opponent_quality': [
                'opponent_recent_form', 'opponent_season_record', 'opponent_power_rating',
                'last_vs_opp_result', 'revenge_motivation', 'power_mismatch'
            ],
            
            'clutch_performance': [
                'clutch_win_rate', 'overtime_win_rate'
            ],
            
            'advanced_metrics': [
                'four_factors_dominance', 'performance_volatility', 'consistency_score',
                'home_road_split', 'desperation_index'
            ]
        }
        
        return groups
    
    def validate_features(self, df: pd.DataFrame) -> Dict[str, any]:
        """Valida la calidad de las características históricas generadas."""
        validation_report = {
            'total_features': 0,
            'missing_features': [],
            'feature_coverage': {},
            'target_analysis': {}
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
        
        # Análisis del target is_win si existe
        if 'is_win' in df.columns:
            validation_report['target_analysis'] = {
                'total_games': len(df),
                'wins': df['is_win'].sum(),
                'losses': (df['is_win'] == 0).sum(),
                'win_rate': df['is_win'].mean(),
                'missing_target': df['is_win'].isna().sum()
            }
        
        logger.info(f"Validación completada: {len(all_features)} features históricas, "
                   f"{len(validation_report['missing_features'])} faltantes")
        
        return validation_report
    
    def _get_historical_series(self, df: pd.DataFrame, column: str, window: int, 
                              operation: str = 'mean', min_periods: int = 1) -> pd.Series:
        """
        Método auxiliar para obtener series históricas con cache para evitar recálculos
        
        Args:
            df: DataFrame con los datos
            column: Nombre de la columna a procesar
            window: Ventana temporal
            operation: Operación a realizar ('mean', 'std', 'sum', 'var')
            min_periods: Períodos mínimos para el cálculo
        
        Returns:
            Serie histórica calculada con shift(1)
        """
        cache_key = f"{column}_{window}_{operation}_{min_periods}"
        
        if cache_key in self._cached_calculations:
            return self._cached_calculations[cache_key]
        
        if column not in df.columns:
            NBALogger.log_warning(logger, "Columna {column} no encontrada para cálculo histórico")
            return pd.Series(index=df.index, dtype=float).fillna(0.5 if operation == 'mean' else 0.0)
        
        # Calcular serie histórica con shift(1)
        shifted_series = df.groupby('Team')[column].shift(1)
        
        if operation == 'mean':
            result = shifted_series.rolling(window=window, min_periods=min_periods).mean()
        elif operation == 'std':
            result = shifted_series.rolling(window=window, min_periods=min_periods).std()
        elif operation == 'sum':
            result = shifted_series.rolling(window=window, min_periods=min_periods).sum()
        elif operation == 'var':
            result = shifted_series.rolling(window=window, min_periods=min_periods).var()
        elif operation == 'expanding_mean':
            result = shifted_series.expanding(min_periods=min_periods).mean()
        else:
            raise ValueError(f"Operación {operation} no soportada")
        
        # Guardar en cache
        self._cached_calculations[cache_key] = result
        
        return result
    
    def _clear_cache(self):
        """Limpiar cache de cálculos para liberar memoria"""
        self._cached_calculations.clear()
        logger.info("Cache de cálculos limpiado")
    
    def _log_feature_generation_start(self, feature_type: str):
        """Log inicio de generación de features"""
        NBALogger.log_training_progress(self.logger, f"Generando features {feature_type}")
    
    def _log_feature_generation_complete(self, feature_type: str, count: int):
        """Log finalización de generación de features"""
        self.logger.info(f"Features {feature_type} completadas: {count} generadas")
    
    def _log_data_validation(self, validation_results: dict):
        """Log resultados de validación de datos"""
        NBALogger.log_data_info(self.logger, validation_results)
    
    def _filter_noisy_features(self, feature_list: List[str], df: pd.DataFrame) -> List[str]:
        """
        Filtra features que pueden generar ruido basándose en:
        - Correlación con el target
        - Varianza
        - Valores faltantes
        - Estabilidad temporal
        """
        if 'is_win' not in df.columns:
            return feature_list
        
        filtered_features = []
        noise_threshold_corr = 0.01  # Correlación mínima con target
        noise_threshold_var = 1e-6   # Varianza mínima
        missing_threshold = 0.5      # Máximo 50% valores faltantes
        
        logger.info(f"Filtrando {len(feature_list)} features para eliminar ruido...")
        
        for feature in feature_list:
            if feature not in df.columns:
                continue
                
            feature_series = df[feature]
            
            # 1. Verificar valores faltantes
            missing_ratio = feature_series.isnull().sum() / len(feature_series)
            if missing_ratio > missing_threshold:
                logger.debug(f"Descartada {feature}: {missing_ratio:.2%} valores faltantes")
                continue
            
            # 2. Verificar varianza
            feature_clean = feature_series.dropna()
            if len(feature_clean) < 10:  # Muy pocos valores válidos
                logger.debug(f"Descartada {feature}: muy pocos valores válidos")
                continue
                
            variance = feature_clean.var()
            if variance < noise_threshold_var:
                logger.debug(f"Descartada {feature}: varianza muy baja ({variance:.6f})")
                continue
            
            # 3. Verificar correlación con target
            target_series = df['is_win']
            
            # Alinear series eliminando NaN
            combined = pd.concat([feature_series, target_series], axis=1).dropna()
            if len(combined) < 50:  # Muy pocas observaciones para correlación confiable
                logger.debug(f"Descartada {feature}: muy pocas observaciones válidas")
                continue
            
            correlation = abs(combined.iloc[:, 0].corr(combined.iloc[:, 1]))
            if pd.isna(correlation) or correlation < noise_threshold_corr:
                logger.debug(f"Descartada {feature}: correlación muy baja ({correlation:.4f})")
                continue
            
            # 4. Verificar valores extremos (outliers excesivos)
            q99 = feature_clean.quantile(0.99)
            q01 = feature_clean.quantile(0.01)
            outlier_ratio = ((feature_clean > q99) | (feature_clean < q01)).sum() / len(feature_clean)
            if outlier_ratio > 0.1:  # Más del 10% son outliers
                logger.debug(f"Descartada {feature}: demasiados outliers ({outlier_ratio:.2%})")
                continue
            
            # Feature pasó todos los filtros
            filtered_features.append(feature)
        
        removed_count = len(feature_list) - len(filtered_features)
        logger.info(f"Filtrado completado: {len(filtered_features)} features conservadas, "
                   f"{removed_count} descartadas por ruido")
        
        return filtered_features
