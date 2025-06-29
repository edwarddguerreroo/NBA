#!/usr/bin/env python3
"""
Script de Debug Detallado para Features de Puntos Totales NBA
===========================================================

Este script analiza exhaustivamente el proceso de generación de features
para identificar y solucionar todos los problemas antes del entrenamiento.

Análisis incluido:
- Validación de datos de entrada
- Detección de data leakage
- Identificación de valores NaN/Infinitos
- Verificación de tipos de datos
- Análisis de correlaciones sospechosas
- Validación de consistencia temporal
- Verificación de feature names
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
import logging
from typing import Dict, List, Tuple, Any
import os

# Imports del proyecto
from src.preprocessing.data_loader import NBADataLoader
from src.models.teams.total_points.features_total_points import TotalPointsFeatureEngine

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

warnings.filterwarnings('ignore')

class TotalPointsFeaturesDebugger:
    """
    Debugger exhaustivo para features de puntos totales NBA
    """
    
    def __init__(self, output_dir: str = "debug_results"):
        """
        Inicializa el debugger
        
        Args:
            output_dir: Directorio para guardar resultados de debug
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Estadísticas de debug
        self.debug_stats = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'issues_found': [],
            'warnings': [],
            'data_quality': {},
            'feature_stats': {},
            'recommendations': []
        }
        
        logger.info("="*80)
        logger.info("INICIANDO DEBUG EXHAUSTIVO DE FEATURES DE PUNTOS TOTALES")
        logger.info("="*80)
    
    def run_complete_debug(self):
        """Ejecuta debug completo del sistema de features"""
        
        try:
            # 1. CARGAR Y VALIDAR DATOS
            logger.info("\n[1/10] Cargando y validando datos...")
            df_teams, df_players = self._load_and_validate_data()
            
            # 2. ANALIZAR DATOS DE ENTRADA
            logger.info("\n[2/10] Analizando calidad de datos de entrada...")
            self._analyze_input_data_quality(df_teams, df_players)
            
            # 3. GENERAR FEATURES PASO A PASO
            logger.info("\n[3/10] Generando features paso a paso...")
            df_features = self._debug_feature_generation(df_teams, df_players)
            
            # 4. VALIDAR FEATURES GENERADAS
            logger.info("\n[4/10] Validando features generadas...")
            self._validate_generated_features(df_features)
            
            # 5. DETECTAR DATA LEAKAGE
            logger.info("\n[5/10] Detectando data leakage...")
            self._detect_data_leakage(df_features)
            
            # 6. PREPARAR FEATURES PARA MODELO
            logger.info("\n[6/10] Preparando features para modelo...")
            X, y = self._debug_feature_preparation(df_features)
            
            # 7. VALIDAR DATOS FINALES
            logger.info("\n[7/10] Validando datos finales...")
            self._validate_final_data(X, y)
            
            # 8. ANALIZAR CORRELACIONES
            logger.info("\n[8/10] Analizando correlaciones...")
            self._analyze_correlations(X, y)
            
            # 9. VERIFICAR CONSISTENCIA TEMPORAL
            logger.info("\n[9/10] Verificando consistencia temporal...")
            self._verify_temporal_consistency(df_features)
            
            # 10. GENERAR REPORTE FINAL
            logger.info("\n[10/10] Generando reporte final...")
            self._generate_final_report()
            
        except Exception as e:
            logger.error(f"Error durante el debug: {str(e)}")
            self.debug_stats['issues_found'].append(f"Error crítico: {str(e)}")
            raise
    
    def _load_and_validate_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Carga y valida datos iniciales"""
        
        try:
            # Cargar datos directamente para evitar problemas del loader
            logger.info("Cargando datos directamente...")
            
            # Cargar teams.csv
            try:
                df_teams = pd.read_csv('data/teams.csv')
                logger.info(f"✓ teams.csv cargado: {len(df_teams)} filas, {len(df_teams.columns)} columnas")
            except Exception as e:
                logger.error(f"✗ Error cargando teams.csv: {e}")
                raise
            
            # Cargar players.csv
            try:
                df_players = pd.read_csv('data/players.csv')
                logger.info(f"✓ players.csv cargado: {len(df_players)} filas, {len(df_players.columns)} columnas")
            except Exception as e:
                logger.warning(f"⚠ Error cargando players.csv: {e}")
                df_players = None
            
            logger.info(f"✓ Datos cargados exitosamente")
            logger.info(f"  - Equipos: {len(df_teams)} registros, {len(df_teams.columns)} columnas")
            logger.info(f"  - Jugadores: {len(df_players) if df_players is not None else 0} registros")
            
            # Validar columnas esenciales para puntos totales
            required_cols = ['Team', 'Opp', 'PTS', 'PTS_Opp', 'Date']
            missing_cols = [col for col in required_cols if col not in df_teams.columns]
            
            if missing_cols:
                issue = f"Columnas esenciales faltantes: {missing_cols}"
                self.debug_stats['issues_found'].append(issue)
                logger.error(f"✗ {issue}")
            else:
                logger.info(f"✓ Todas las columnas esenciales presentes")
            
            # Validar tipos de datos
            if 'Date' in df_teams.columns:
                if not pd.api.types.is_datetime64_any_dtype(df_teams['Date']):
                    df_teams['Date'] = pd.to_datetime(df_teams['Date'], errors='coerce')
                    logger.info("✓ Columna Date convertida a datetime")
            
            # Crear target si no existe
            if 'total_points' not in df_teams.columns:
                df_teams['total_points'] = df_teams['PTS'] + df_teams['PTS_Opp']
                logger.info("✓ Variable target 'total_points' creada")
            
            # Estadísticas del target
            target_stats = df_teams['total_points'].describe()
            logger.info(f"✓ Estadísticas target 'total_points':")
            logger.info(f"  - Media: {target_stats['mean']:.1f}")
            logger.info(f"  - Min/Max: {target_stats['min']:.0f}/{target_stats['max']:.0f}")
            logger.info(f"  - Desv. Est: {target_stats['std']:.1f}")
            
            # Estadísticas básicas
            self.debug_stats['data_quality']['teams_count'] = len(df_teams)
            self.debug_stats['data_quality']['players_count'] = len(df_players) if df_players is not None else 0
            self.debug_stats['data_quality']['teams_columns'] = len(df_teams.columns)
            self.debug_stats['data_quality']['date_range'] = f"{df_teams['Date'].min()} to {df_teams['Date'].max()}"
            self.debug_stats['data_quality']['target_range'] = f"{target_stats['min']:.0f} - {target_stats['max']:.0f}"
            
            return df_teams, df_players
            
        except Exception as e:
            issue = f"Error cargando datos: {str(e)}"
            self.debug_stats['issues_found'].append(issue)
            logger.error(f"✗ {issue}")
            raise
    
    def _analyze_input_data_quality(self, df_teams: pd.DataFrame, df_players: pd.DataFrame):
        """Analiza calidad de datos de entrada"""
        
        logger.info("Analizando calidad de datos de entrada...")
        
        # Análisis de df_teams
        logger.info("\n--- ANÁLISIS DE DATOS DE EQUIPOS ---")
        
        # NaN por columna
        nan_summary = df_teams.isna().sum()
        high_nan_cols = nan_summary[nan_summary > len(df_teams) * 0.1].sort_values(ascending=False)
        
        if not high_nan_cols.empty:
            logger.warning(f"Columnas con >10% NaN:")
            for col, count in high_nan_cols.head(10).items():
                pct = (count / len(df_teams)) * 100
                logger.warning(f"  - {col}: {count} ({pct:.1f}%)")
                self.debug_stats['warnings'].append(f"Columna {col} tiene {pct:.1f}% NaN")
        else:
            logger.info("✓ No hay columnas con >10% NaN")
        
        # Validar datos numéricos esenciales
        numeric_cols = ['PTS', 'PTS_Opp', 'FGA', 'FGA_Opp', 'FTA', 'FTA_Opp']
        for col in numeric_cols:
            if col in df_teams.columns:
                # Verificar rangos razonables
                col_data = df_teams[col]
                if col.startswith('PTS'):
                    # Puntos: rango razonable 60-180
                    outliers = col_data[(col_data < 60) | (col_data > 180)]
                    if len(outliers) > 0:
                        logger.warning(f"  - {col}: {len(outliers)} valores fuera de rango 60-180")
                elif col.startswith('FGA'):
                    # Field goals attempted: rango razonable 60-120
                    outliers = col_data[(col_data < 60) | (col_data > 120)]
                    if len(outliers) > 0:
                        logger.warning(f"  - {col}: {len(outliers)} valores fuera de rango 60-120")
                elif col.startswith('FTA'):
                    # Free throws attempted: rango razonable 10-40
                    outliers = col_data[(col_data < 5) | (col_data > 50)]
                    if len(outliers) > 0:
                        logger.warning(f"  - {col}: {len(outliers)} valores fuera de rango 5-50")
        
        # Análisis temporal
        if 'Date' in df_teams.columns:
            date_gaps = self._check_date_gaps(df_teams)
            logger.info(f"✓ Análisis temporal completado. Gaps encontrados: {sum(date_gaps.values())}")
        
        # Análisis de df_players (si existe)
        if df_players is not None:
            logger.info("\n--- ANÁLISIS DE DATOS DE JUGADORES ---")
            player_nan_summary = df_players.isna().sum()
            high_player_nan = player_nan_summary[player_nan_summary > len(df_players) * 0.1]
            
            if not high_player_nan.empty:
                logger.warning(f"Columnas de jugadores con >10% NaN: {len(high_player_nan)}")
            else:
                logger.info("✓ Datos de jugadores en buen estado")
        
        # Guardar estadísticas
        self.debug_stats['data_quality']['high_nan_columns'] = len(high_nan_cols)
        self.debug_stats['data_quality']['teams_numeric_cols'] = len([c for c in df_teams.columns if df_teams[c].dtype in ['int64', 'float64']])
    
    def _check_date_gaps(self, df: pd.DataFrame) -> Dict[str, int]:
        """Verifica gaps en fechas por equipo"""
        
        gaps_by_team = {}
        
        for team in df['Team'].unique():
            team_data = df[df['Team'] == team].sort_values('Date')
            if len(team_data) > 1:
                date_diffs = team_data['Date'].diff().dt.days
                # Considerar gap si hay más de 5 días entre juegos
                gaps = date_diffs[date_diffs > 5]
                gaps_by_team[team] = len(gaps)
        
        return gaps_by_team
    
    def _debug_feature_generation(self, df_teams: pd.DataFrame, df_players: pd.DataFrame) -> pd.DataFrame:
        """Genera features paso a paso para debug"""
        
        logger.info("Generando features de puntos totales paso a paso...")
        
        # Crear copia para debug
        df_debug = df_teams.copy()
        initial_shape = df_debug.shape
        
        logger.info(f"Datos iniciales: {initial_shape[0]} filas, {initial_shape[1]} columnas")
        
        try:
            # Inicializar feature engine
            feature_engine = TotalPointsFeatureEngine(lookback_games=10)
            
            # Generar features completas
            logger.info("Ejecutando generación de features completa...")
            features = feature_engine.generate_all_features(df_debug, df_players)
            
            logger.info(f"✓ Generación completada: {len(features)} features creadas")
            
            # Validar shape final
            final_shape = df_debug.shape
            logger.info(f"Datos finales: {final_shape[0]} filas, {final_shape[1]} columnas")
            
            if final_shape[0] != initial_shape[0]:
                warning = f"ALERTA: Número de filas cambió de {initial_shape[0]} a {final_shape[0]}"
                logger.warning(warning)
                self.debug_stats['warnings'].append(warning)
            
            # Análisis de features por grupo
            feature_groups = feature_engine.get_feature_importance_groups()
            logger.info("\n--- FEATURES POR GRUPO ---")
            for group, group_features in feature_groups.items():
                logger.info(f"  - {group}: {len(group_features)} features")
                
            # Guardar estadísticas
            self.debug_stats['feature_stats']['total_features'] = len(features)
            self.debug_stats['feature_stats']['feature_groups'] = {k: len(v) for k, v in feature_groups.items()}
            self.debug_stats['feature_stats']['features_list'] = features
            
            return df_debug
            
        except Exception as e:
            error = f"Error en generación de features: {str(e)}"
            logger.error(error)
            self.debug_stats['issues_found'].append(error)
            raise
    
    def _validate_generated_features(self, df: pd.DataFrame):
        """Valida features generadas"""
        
        logger.info("Validando features generadas...")
        
        # Identificar features (excluir columnas originales)
        exclude_cols = [
            'Team', 'Date', 'Away', 'Opp', 'Result', 'MP', 'PTS', 'PTS_Opp', 'total_points',
            'FG', 'FGA', 'FG%', '2P', '2PA', '2P%', '3P', '3PA', '3P%', 
            'FT', 'FTA', 'FT%', 'FG_Opp', 'FGA_Opp', 'FG%_Opp', 
            '2P_Opp', '2PA_Opp', '2P%_Opp', '3P_Opp', '3PA_Opp', '3P%_Opp',
            'FT_Opp', 'FTA_Opp', 'FT%_Opp',
            # Excluir categóricas
            'total_points_tier', 'team_scoring_tier', 'opp_scoring_tier'
        ]
        
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        logger.info(f"Validando {len(feature_cols)} features...")
        
        # 1. Verificar NaN
        nan_counts = df[feature_cols].isna().sum()
        high_nan_features = nan_counts[nan_counts > len(df) * 0.3]
        
        if not high_nan_features.empty:
            logger.warning(f"Features con >30% NaN:")
            for feature, count in high_nan_features.head(10).items():
                pct = (count / len(df)) * 100
                logger.warning(f"  - {feature}: {pct:.1f}%")
                self.debug_stats['warnings'].append(f"Feature {feature} tiene {pct:.1f}% NaN")
        else:
            logger.info("✓ No hay features con >30% NaN")
        
        # 2. Verificar infinitos
        numeric_features = df[feature_cols].select_dtypes(include=[np.number])
        inf_counts = np.isinf(numeric_features).sum()
        inf_features = inf_counts[inf_counts > 0]
        
        if not inf_features.empty:
            logger.warning(f"Features con valores infinitos:")
            for feature, count in inf_features.head(10).items():
                logger.warning(f"  - {feature}: {count} infinitos")
                self.debug_stats['warnings'].append(f"Feature {feature} tiene {count} infinitos")
        else:
            logger.info("✓ No hay features con valores infinitos")
        
        # 3. Verificar varianza cero
        zero_var_features = numeric_features.columns[numeric_features.var() == 0]
        
        if len(zero_var_features) > 0:
            logger.warning(f"Features con varianza cero: {len(zero_var_features)}")
            for feature in zero_var_features[:5]:
                logger.warning(f"  - {feature}")
                self.debug_stats['warnings'].append(f"Feature {feature} tiene varianza cero")
        else:
            logger.info("✓ No hay features con varianza cero")
        
        # 4. Verificar features específicas de puntos totales
        expected_features = [
            'game_total_scoring_projection',
            'game_combined_efficiency',
            'total_points_trend_5g',
            'pace_efficiency_interaction',
            'home_rivalry_boost'
        ]
        
        missing_expected = [f for f in expected_features if f not in feature_cols]
        if missing_expected:
            logger.warning(f"Features esperadas faltantes: {missing_expected}")
            self.debug_stats['warnings'].append(f"Features esperadas faltantes: {missing_expected}")
        else:
            logger.info("✓ Todas las features esperadas están presentes")
        
        # Guardar estadísticas de validación
        self.debug_stats['feature_stats']['high_nan_count'] = len(high_nan_features)
        self.debug_stats['feature_stats']['inf_features_count'] = len(inf_features)
        self.debug_stats['feature_stats']['zero_var_count'] = len(zero_var_features)
        self.debug_stats['feature_stats']['missing_expected'] = missing_expected
    
    def _detect_data_leakage(self, df: pd.DataFrame):
        """Detecta posible data leakage"""
        
        logger.info("Detectando data leakage...")
        
        # Verificar que no haya correlación perfecta con target
        if 'total_points' in df.columns:
            feature_cols = [col for col in df.columns if col not in [
                'Team', 'Date', 'Away', 'Opp', 'Result', 'MP', 'PTS', 'PTS_Opp', 'total_points'
            ]]
            
            numeric_features = df[feature_cols].select_dtypes(include=[np.number])
            
            if not numeric_features.empty:
                # Calcular correlaciones con target
                correlations = numeric_features.corrwith(df['total_points']).abs()
                
                # Buscar correlaciones sospechosamente altas
                suspicious_corr = correlations[correlations > 0.99]
                
                if not suspicious_corr.empty:
                    logger.warning(f"Features con correlación >0.99 con target:")
                    for feature, corr in suspicious_corr.items():
                        logger.warning(f"  - {feature}: {corr:.4f}")
                        self.debug_stats['warnings'].append(f"Posible data leakage: {feature} (corr={corr:.4f})")
                else:
                    logger.info("✓ No se detectó data leakage obvio")
        
        # Verificar features que podrían usar datos futuros
        future_leak_patterns = ['_next', '_future', '_ahead']
        potential_leaks = []
        
        for col in df.columns:
            for pattern in future_leak_patterns:
                if pattern in col.lower():
                    potential_leaks.append(col)
                    break
        
        if potential_leaks:
            logger.warning(f"Features con patrones sospechosos: {potential_leaks}")
            self.debug_stats['warnings'].append(f"Features sospechosas: {potential_leaks}")
        else:
            logger.info("✓ No se encontraron patrones sospechosos en nombres de features")
    
    def _debug_feature_preparation(self, df_features: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepara features para modelo"""
        
        logger.info("Preparando features para modelo...")
        
        # Verificar target
        if 'total_points' not in df_features.columns:
            error = "Target 'total_points' no encontrado"
            logger.error(error)
            self.debug_stats['issues_found'].append(error)
            raise ValueError(error)
        
        # Separar features y target
        exclude_cols = [
            'Team', 'Date', 'Away', 'Opp', 'Result', 'MP', 'PTS', 'PTS_Opp', 'total_points',
            'FG', 'FGA', 'FG%', '2P', '2PA', '2P%', '3P', '3PA', '3P%', 
            'FT', 'FTA', 'FT%', 'FG_Opp', 'FGA_Opp', 'FG%_Opp', 
            '2P_Opp', '2PA_Opp', '2P%_Opp', '3P_Opp', '3PA_Opp', '3P%_Opp',
            'FT_Opp', 'FTA_Opp', 'FT%_Opp',
            # Excluir categóricas
            'total_points_tier', 'team_scoring_tier', 'opp_scoring_tier'
        ]
        
        feature_cols = [col for col in df_features.columns if col not in exclude_cols]
        
        # Filtrar solo numéricas
        X_raw = df_features[feature_cols].select_dtypes(include=[np.number])
        y = df_features['total_points']
        
        logger.info(f"Features iniciales: {len(feature_cols)}")
        logger.info(f"Features numéricas: {len(X_raw.columns)}")
        
        # Limpiar datos
        # 1. Eliminar filas con target NaN
        valid_target_mask = y.notna()
        X_clean = X_raw[valid_target_mask]
        y_clean = y[valid_target_mask]
        
        logger.info(f"Filas con target válido: {len(y_clean)}")
        
        # 2. Eliminar features con demasiados NaN
        nan_threshold = 0.5
        low_nan_features = X_clean.columns[X_clean.isna().mean() <= nan_threshold]
        X_clean = X_clean[low_nan_features]
        
        logger.info(f"Features con <{nan_threshold*100:.0f}% NaN: {len(X_clean.columns)}")
        
        # 3. Rellenar NaN restantes
        X_clean = X_clean.fillna(X_clean.median())
        
        # 4. Eliminar features con varianza cero
        from sklearn.feature_selection import VarianceThreshold
        var_selector = VarianceThreshold(threshold=0)
        X_clean = pd.DataFrame(
            var_selector.fit_transform(X_clean),
            columns=X_clean.columns[var_selector.get_support()],
            index=X_clean.index
        )
        
        logger.info(f"Features finales: {len(X_clean.columns)}")
        
        # Guardar estadísticas
        self.debug_stats['feature_stats']['final_feature_count'] = len(X_clean.columns)
        self.debug_stats['feature_stats']['valid_samples'] = len(y_clean)
        self.debug_stats['feature_stats']['final_features'] = list(X_clean.columns)
        
        return X_clean, y_clean
    
    def _validate_final_data(self, X: pd.DataFrame, y: pd.Series):
        """Valida datos finales"""
        
        logger.info("Validando datos finales...")
        
        # Verificar shapes
        logger.info(f"Shape final: X={X.shape}, y={y.shape}")
        
        if X.shape[0] != y.shape[0]:
            error = f"Mismatch en número de muestras: X={X.shape[0]}, y={y.shape[0]}"
            logger.error(error)
            self.debug_stats['issues_found'].append(error)
        
        # Verificar NaN
        x_nan_count = X.isna().sum().sum()
        y_nan_count = y.isna().sum()
        
        if x_nan_count > 0:
            logger.warning(f"X contiene {x_nan_count} valores NaN")
            self.debug_stats['warnings'].append(f"X contiene {x_nan_count} NaN")
        
        if y_nan_count > 0:
            logger.warning(f"y contiene {y_nan_count} valores NaN")
            self.debug_stats['warnings'].append(f"y contiene {y_nan_count} NaN")
        
        if x_nan_count == 0 and y_nan_count == 0:
            logger.info("✓ No hay valores NaN en datos finales")
        
        # Verificar infinitos
        x_inf_count = np.isinf(X).sum().sum()
        y_inf_count = np.isinf(y).sum()
        
        if x_inf_count > 0:
            logger.warning(f"X contiene {x_inf_count} valores infinitos")
            self.debug_stats['warnings'].append(f"X contiene {x_inf_count} infinitos")
        
        if y_inf_count > 0:
            logger.warning(f"y contiene {y_inf_count} valores infinitos")
            self.debug_stats['warnings'].append(f"y contiene {y_inf_count} infinitos")
        
        if x_inf_count == 0 and y_inf_count == 0:
            logger.info("✓ No hay valores infinitos en datos finales")
        
        # Estadísticas del target
        target_stats = y.describe()
        logger.info(f"Estadísticas del target:")
        logger.info(f"  - Media: {target_stats['mean']:.1f}")
        logger.info(f"  - Min/Max: {target_stats['min']:.0f}/{target_stats['max']:.0f}")
        logger.info(f"  - Desv. Estándar: {target_stats['std']:.1f}")
        
        # Verificar rango razonable del target
        if target_stats['min'] < 150 or target_stats['max'] > 300:
            warning = f"Target fuera de rango esperado (150-300): {target_stats['min']:.0f}-{target_stats['max']:.0f}"
            logger.warning(warning)
            self.debug_stats['warnings'].append(warning)
        else:
            logger.info("✓ Target en rango esperado (150-300)")
        
        # Verificar features con varianza muy baja
        low_var_threshold = 0.01
        low_var_features = X.columns[X.var() < low_var_threshold]
        
        if len(low_var_features) > 0:
            logger.warning(f"Features con varianza muy baja (<{low_var_threshold}): {len(low_var_features)}")
            for feature in low_var_features[:5]:
                logger.warning(f"  - {feature}: var={X[feature].var():.6f}")
        else:
            logger.info("✓ Todas las features tienen varianza adecuada")
        
        # Guardar estadísticas finales
        self.debug_stats['data_quality']['final_samples'] = len(y)
        self.debug_stats['data_quality']['final_features'] = len(X.columns)
        self.debug_stats['data_quality']['target_mean'] = float(target_stats['mean'])
        self.debug_stats['data_quality']['target_std'] = float(target_stats['std'])
    
    def _analyze_correlations(self, X: pd.DataFrame, y: pd.Series):
        """Analiza correlaciones"""
        
        logger.info("Analizando correlaciones...")
        
        # Correlación con target
        target_correlations = X.corrwith(y).abs().sort_values(ascending=False)
        
        logger.info("Top 10 features por correlación con target:")
        for i, (feature, corr) in enumerate(target_correlations.head(10).items()):
            logger.info(f"  {i+1:2d}. {feature}: {corr:.4f}")
        
        # Features con baja correlación
        low_corr_threshold = 0.01
        low_corr_features = target_correlations[target_correlations < low_corr_threshold]
        
        if len(low_corr_features) > 0:
            logger.warning(f"Features con correlación muy baja (<{low_corr_threshold}): {len(low_corr_features)}")
            self.debug_stats['warnings'].append(f"{len(low_corr_features)} features con correlación <{low_corr_threshold}")
        
        # Correlaciones entre features (multicolinealidad)
        if len(X.columns) <= 100:  # Solo para datasets pequeños
            corr_matrix = X.corr().abs()
            high_corr_pairs = []
            
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    if corr_matrix.iloc[i, j] > 0.95:
                        high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_matrix.iloc[i, j]))
            
            if high_corr_pairs:
                logger.warning(f"Pares de features con correlación >0.95: {len(high_corr_pairs)}")
                for feat1, feat2, corr in high_corr_pairs[:5]:
                    logger.warning(f"  - {feat1} vs {feat2}: {corr:.4f}")
                self.debug_stats['warnings'].append(f"{len(high_corr_pairs)} pares con alta correlación")
            else:
                logger.info("✓ No hay alta multicolinealidad entre features")
        else:
            logger.info("Dataset demasiado grande para análisis completo de correlaciones")
        
        # Guardar estadísticas
        self.debug_stats['feature_stats']['top_correlations'] = {
            feature: float(corr) for feature, corr in target_correlations.head(10).items()
        }
        self.debug_stats['feature_stats']['low_corr_count'] = len(low_corr_features)
    
    def _verify_temporal_consistency(self, df: pd.DataFrame):
        """Verifica consistencia temporal"""
        
        logger.info("Verificando consistencia temporal...")
        
        if 'Date' not in df.columns:
            logger.warning("No hay columna Date para verificación temporal")
            return
        
        # Verificar orden temporal
        df_sorted = df.sort_values(['Team', 'Date'])
        
        # Verificar que las features históricas no usen datos futuros
        rolling_features = [col for col in df.columns if any(x in col for x in ['_avg_', '_trend_', '_momentum'])]
        
        logger.info(f"Verificando {len(rolling_features)} features temporales...")
        
        temporal_issues = 0
        
        for team in df['Team'].unique()[:3]:  # Verificar solo algunas muestras
            team_data = df_sorted[df_sorted['Team'] == team]
            
            if len(team_data) < 5:
                continue
                
            # Verificar que features históricas cambien gradualmente
            for feature in rolling_features[:5]:  # Solo verificar algunas features
                if feature in team_data.columns:
                    feature_values = team_data[feature].dropna()
                    
                    if len(feature_values) > 3:
                        # Verificar cambios extremos (posible data leakage)
                        changes = feature_values.diff().abs()
                        extreme_changes = changes[changes > feature_values.std() * 3]
                        
                        if len(extreme_changes) > len(feature_values) * 0.1:
                            temporal_issues += 1
        
        if temporal_issues > 0:
            logger.warning(f"Posibles problemas temporales detectados: {temporal_issues}")
            self.debug_stats['warnings'].append(f"{temporal_issues} posibles problemas temporales")
        else:
            logger.info("✓ Consistencia temporal verificada")
    
    def _generate_final_report(self):
        """Genera reporte final de debug"""
        
        logger.info("\n" + "="*80)
        logger.info("REPORTE FINAL DE DEBUG")
        logger.info("="*80)
        
        # Resumen de issues
        issues_count = len(self.debug_stats['issues_found'])
        warnings_count = len(self.debug_stats['warnings'])
        
        logger.info(f"\n📊 RESUMEN:")
        logger.info(f"  - Issues críticos: {issues_count}")
        logger.info(f"  - Warnings: {warnings_count}")
        logger.info(f"  - Features finales: {self.debug_stats['feature_stats'].get('final_feature_count', 0)}")
        logger.info(f"  - Muestras válidas: {self.debug_stats['feature_stats'].get('valid_samples', 0)}")
        
        # Mostrar issues críticos
        if issues_count > 0:
            logger.error(f"\n❌ ISSUES CRÍTICOS ({issues_count}):")
            for i, issue in enumerate(self.debug_stats['issues_found'], 1):
                logger.error(f"  {i}. {issue}")
        
        # Mostrar warnings importantes
        if warnings_count > 0:
            logger.warning(f"\n⚠️  WARNINGS ({min(warnings_count, 10)}):")
            for i, warning in enumerate(self.debug_stats['warnings'][:10], 1):
                logger.warning(f"  {i}. {warning}")
                
            if warnings_count > 10:
                logger.warning(f"  ... y {warnings_count - 10} warnings más")
        
        # Estado general
        if issues_count == 0:
            if warnings_count <= 5:
                logger.info("\n✅ ESTADO: EXCELENTE - Listo para entrenamiento")
            elif warnings_count <= 15:
                logger.info("\n🟡 ESTADO: BUENO - Se puede proceder con precaución")
            else:
                logger.info("\n🟠 ESTADO: REGULAR - Revisar warnings antes de continuar")
        else:
            logger.error("\n🔴 ESTADO: PROBLEMAS CRÍTICOS - Resolver issues antes de continuar")
        
        # Generar recomendaciones
        self._generate_recommendations()
        
        # Guardar reporte
        self._save_debug_report()
        
        logger.info(f"\n📝 Reporte completo guardado en: {self.output_dir}/")
        logger.info("="*80)
    
    def _generate_recommendations(self):
        """Genera recomendaciones basadas en el análisis"""
        
        recommendations = []
        
        # Basado en issues críticos
        if len(self.debug_stats['issues_found']) > 0:
            recommendations.append("🔴 CRÍTICO: Resolver todos los issues críticos antes de entrenar")
        
        # Basado en warnings
        warnings_count = len(self.debug_stats['warnings'])
        if warnings_count > 20:
            recommendations.append("🟠 Revisar y filtrar features problemáticas")
        
        # Basado en correlaciones
        if self.debug_stats['feature_stats'].get('low_corr_count', 0) > 10:
            recommendations.append("📉 Considerar eliminar features con baja correlación")
        
        # Basado en número de features
        feature_count = self.debug_stats['feature_stats'].get('final_feature_count', 0)
        sample_count = self.debug_stats['feature_stats'].get('valid_samples', 0)
        
        if feature_count > 0 and sample_count > 0:
            ratio = sample_count / feature_count
            if ratio < 10:
                recommendations.append(f"⚠️ Ratio muestras/features bajo ({ratio:.1f}:1) - Considerar reducción dimensional")
        
        # Basado en calidad del target
        target_std = self.debug_stats['data_quality'].get('target_std', 0)
        if target_std < 15:
            recommendations.append("📊 Baja variabilidad en target - Verificar calidad de datos")
        
        # Recomendaciones generales
        if len(self.debug_stats['issues_found']) == 0 and warnings_count <= 10:
            recommendations.append("✅ Datos en buen estado - Proceder con entrenamiento")
            recommendations.append("🚀 Considerar validación cruzada temporal para evaluación")
            recommendations.append("📈 Monitorear performance en datos de validación")
        
        self.debug_stats['recommendations'] = recommendations
        
        # Mostrar recomendaciones
        if recommendations:
            logger.info(f"\n💡 RECOMENDACIONES:")
            for i, rec in enumerate(recommendations, 1):
                logger.info(f"  {i}. {rec}")
    
    def _save_debug_report(self):
        """Guarda reporte de debug en archivo"""
        
        import json
        
        # Guardar estadísticas completas
        report_path = os.path.join(self.output_dir, f"total_points_debug_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(self.debug_stats, f, indent=2, ensure_ascii=False, default=str)
        
        # Guardar resumen en texto
        summary_path = os.path.join(self.output_dir, f"total_points_debug_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
        
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("REPORTE DE DEBUG - FEATURES DE PUNTOS TOTALES NBA\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Timestamp: {self.debug_stats['timestamp']}\n\n")
            
            f.write("RESUMEN:\n")
            f.write(f"- Issues críticos: {len(self.debug_stats['issues_found'])}\n")
            f.write(f"- Warnings: {len(self.debug_stats['warnings'])}\n")
            f.write(f"- Features finales: {self.debug_stats['feature_stats'].get('final_feature_count', 0)}\n")
            f.write(f"- Muestras válidas: {self.debug_stats['feature_stats'].get('valid_samples', 0)}\n\n")
            
            if self.debug_stats['issues_found']:
                f.write("ISSUES CRÍTICOS:\n")
                for i, issue in enumerate(self.debug_stats['issues_found'], 1):
                    f.write(f"{i}. {issue}\n")
                f.write("\n")
            
            if self.debug_stats['warnings']:
                f.write("WARNINGS (top 10):\n")
                for i, warning in enumerate(self.debug_stats['warnings'][:10], 1):
                    f.write(f"{i}. {warning}\n")
                f.write("\n")
            
            if self.debug_stats['recommendations']:
                f.write("RECOMENDACIONES:\n")
                for i, rec in enumerate(self.debug_stats['recommendations'], 1):
                    f.write(f"{i}. {rec}\n")

def main():
    """Función principal del script de debug"""
    
    print("🏀 INICIANDO DEBUG DE FEATURES DE PUNTOS TOTALES NBA")
    print("=" * 60)
    
    try:
        # Crear debugger
        debugger = TotalPointsFeaturesDebugger()
        
        # Ejecutar debug completo
        debugger.run_complete_debug()
        
        print("\n✅ DEBUG COMPLETADO EXITOSAMENTE")
        
    except Exception as e:
        print(f"\n❌ ERROR DURANTE DEBUG: {str(e)}")
        raise

if __name__ == "__main__":
    main() 