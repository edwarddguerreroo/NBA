#!/usr/bin/env python3
"""
Script de Debug para Features de Total Points
============================================

Este script analiza las features generadas por TotalPointsFeatureEngine
para detectar problemas como valores extremos, NaN, infinitos, etc.
"""

import pandas as pd
import numpy as np
import sys
import os
import logging
from typing import Dict, List, Tuple, Any
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Agregar el directorio src al path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from models.teams.total_points.features_total_points import TotalPointsFeatureEngine
from preprocessing.data_loader import NBADataLoader

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TotalPointsFeatureDebugger:
    """Debugger especializado para features de total_points"""
    
    def __init__(self):
        self.feature_engine = TotalPointsFeatureEngine()
        # Usar NBADataLoader con rutas correctas
        self.data_loader = NBADataLoader(
            game_data_path="data/players.csv",
            biometrics_path="data/height.csv", 
            teams_path="data/teams.csv"
        )
        self.debug_results = {}
        
    def load_test_data(self) -> pd.DataFrame:
        """Cargar datos de prueba"""
        logger.info("Cargando datos de prueba...")
        try:
            # Cargar datos usando NBADataLoader
            df_players, df_teams = self.data_loader.load_data()
            
            if df_teams.empty:
                logger.error("No se pudieron cargar datos de equipos")
                return pd.DataFrame()
            
            logger.info(f"Datos cargados: {df_teams.shape[0]} registros, {df_teams.shape[1]} columnas")
            return df_teams
        except Exception as e:
            logger.error(f"Error cargando datos: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return pd.DataFrame()
    
    def analyze_raw_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analizar datos crudos antes del procesamiento"""
        logger.info("=== ANÁLISIS DE DATOS CRUDOS ===")
        
        analysis = {
            'shape': df.shape,
            'columns': list(df.columns),
            'dtypes': df.dtypes.to_dict(),
            'missing_values': df.isnull().sum().to_dict(),
            'numeric_columns': df.select_dtypes(include=[np.number]).columns.tolist(),
            'categorical_columns': df.select_dtypes(exclude=[np.number]).columns.tolist()
        }
        
        # Verificar columnas críticas
        critical_cols = ['Team', 'Opp', 'PTS', 'PTS_Opp', 'Date']
        missing_critical = [col for col in critical_cols if col not in df.columns]
        analysis['missing_critical_columns'] = missing_critical
        
        # Estadísticas básicas de PTS y PTS_Opp
        if 'PTS' in df.columns and 'PTS_Opp' in df.columns:
            analysis['pts_stats'] = {
                'PTS_mean': df['PTS'].mean(),
                'PTS_std': df['PTS'].std(),
                'PTS_min': df['PTS'].min(),
                'PTS_max': df['PTS'].max(),
                'PTS_Opp_mean': df['PTS_Opp'].mean(),
                'PTS_Opp_std': df['PTS_Opp'].std(),
                'PTS_Opp_min': df['PTS_Opp'].min(),
                'PTS_Opp_max': df['PTS_Opp'].max()
            }
            
            # Crear total_points para análisis
            df['total_points'] = df['PTS'] + df['PTS_Opp']
            analysis['total_points_stats'] = {
                'mean': df['total_points'].mean(),
                'std': df['total_points'].std(),
                'min': df['total_points'].min(),
                'max': df['total_points'].max(),
                'median': df['total_points'].median()
            }
        
        self._print_analysis_section("DATOS CRUDOS", analysis)
        return analysis
    
    def generate_and_analyze_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Generar features y analizarlas"""
        logger.info("=== GENERACIÓN Y ANÁLISIS DE FEATURES ===")
        
        try:
            # Generar features
            logger.info("Generando features...")
            df_features = self.feature_engine.create_features(df.copy())
            
            if df_features.empty:
                logger.error("No se generaron features")
                return pd.DataFrame(), {}
            
            logger.info(f"Features generadas: {df_features.shape[1]} columnas, {df_features.shape[0]} filas")
            
            # Analizar features generadas
            analysis = self._analyze_features_quality(df_features)
            
            return df_features, analysis
            
        except Exception as e:
            logger.error(f"Error generando features: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return pd.DataFrame(), {}
    
    def _analyze_features_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analizar calidad de las features"""
        analysis = {
            'total_features': df.shape[1],
            'total_rows': df.shape[0],
            'numeric_features': len(df.select_dtypes(include=[np.number]).columns),
            'categorical_features': len(df.select_dtypes(exclude=[np.number]).columns)
        }
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        # 1. Detectar valores problemáticos
        problematic_features = {}
        
        for col in numeric_cols:
            issues = []
            
            # NaN
            nan_count = df[col].isnull().sum()
            nan_pct = (nan_count / len(df)) * 100
            if nan_count > 0:
                issues.append(f"NaN: {nan_count} ({nan_pct:.1f}%)")
            
            # Infinitos
            inf_count = np.isinf(df[col]).sum()
            if inf_count > 0:
                issues.append(f"Infinitos: {inf_count}")
            
            # Valores extremos (fuera de 5 desviaciones estándar)
            if df[col].std() > 0:
                mean_val = df[col].mean()
                std_val = df[col].std()
                extreme_count = ((df[col] - mean_val).abs() > 5 * std_val).sum()
                if extreme_count > 0:
                    issues.append(f"Extremos: {extreme_count}")
            
            # Valores muy grandes (>10000 o <-10000)
            large_count = (df[col].abs() > 10000).sum()
            if large_count > 0:
                issues.append(f"Muy grandes: {large_count}")
            
            # Varianza muy baja (features casi constantes)
            if df[col].var() < 0.001:
                issues.append(f"Baja varianza: {df[col].var():.6f}")
            
            if issues:
                problematic_features[col] = {
                    'issues': issues,
                    'mean': df[col].mean(),
                    'std': df[col].std(),
                    'min': df[col].min(),
                    'max': df[col].max(),
                    'range': df[col].max() - df[col].min()
                }
        
        analysis['problematic_features'] = problematic_features
        
        # 2. Top features con problemas más severos
        severe_problems = []
        for col, info in problematic_features.items():
            severity_score = 0
            for issue in info['issues']:
                if 'Infinitos' in issue:
                    severity_score += 10
                elif 'NaN' in issue and '50' in issue:  # >50% NaN
                    severity_score += 8
                elif 'Extremos' in issue:
                    severity_score += 5
                elif 'Muy grandes' in issue:
                    severity_score += 7
            
            if severity_score > 5:
                severe_problems.append((col, severity_score, info))
        
        # Ordenar por severidad
        severe_problems.sort(key=lambda x: x[1], reverse=True)
        analysis['severe_problems'] = severe_problems[:10]  # Top 10
        
        # 3. Estadísticas generales
        analysis['general_stats'] = {
            'features_with_nan': len([col for col in numeric_cols if df[col].isnull().any()]),
            'features_with_inf': len([col for col in numeric_cols if np.isinf(df[col]).any()]),
            'features_with_extremes': len([col for col in numeric_cols if ((df[col] - df[col].mean()).abs() > 5 * df[col].std()).any() if df[col].std() > 0]),
            'low_variance_features': len([col for col in numeric_cols if df[col].var() < 0.001])
        }
        
        self._print_analysis_section("CALIDAD DE FEATURES", analysis)
        return analysis
    
    def test_model_compatibility(self, df_features: pd.DataFrame) -> Dict[str, Any]:
        """Probar compatibilidad con el modelo"""
        logger.info("=== PRUEBA DE COMPATIBILIDAD CON MODELO ===")
        
        try:
            # Simular preparación para modelo
            numeric_cols = df_features.select_dtypes(include=[np.number]).columns.tolist()
            
            # Excluir target y columnas problemáticas
            excluded = ['total_points', 'PTS', 'PTS_Opp', 'Team', 'Date', 'Opp']
            feature_cols = [col for col in numeric_cols if col not in excluded]
            
            if not feature_cols:
                return {'error': 'No hay features válidas para el modelo'}
            
            # Preparar datos como lo haría el modelo
            X = df_features[feature_cols].copy()
            
            # Limpiar datos
            X = X.replace([np.inf, -np.inf], np.nan)
            X = X.fillna(0)
            
            analysis = {
                'available_features': len(feature_cols),
                'features_used': feature_cols[:65] if len(feature_cols) >= 65 else feature_cols,
                'shape_after_cleaning': X.shape,
                'final_nan_count': X.isnull().sum().sum(),
                'final_inf_count': np.isinf(X).sum().sum(),
                'feature_ranges': {}
            }
            
            # Analizar rangos de features finales
            for col in X.columns:
                analysis['feature_ranges'][col] = {
                    'min': float(X[col].min()),
                    'max': float(X[col].max()),
                    'mean': float(X[col].mean()),
                    'std': float(X[col].std())
                }
            
            # Detectar features que podrían causar problemas al modelo
            problematic_for_model = []
            for col in X.columns:
                col_range = X[col].max() - X[col].min()
                if col_range > 1000:  # Rango muy grande
                    problematic_for_model.append(f"{col}: rango muy grande ({col_range:.1f})")
                elif X[col].std() > 100:  # Desviación estándar muy alta
                    problematic_for_model.append(f"{col}: std muy alta ({X[col].std():.1f})")
            
            analysis['problematic_for_model'] = problematic_for_model
            
            self._print_analysis_section("COMPATIBILIDAD CON MODELO", analysis)
            return analysis
            
        except Exception as e:
            logger.error(f"Error en prueba de compatibilidad: {str(e)}")
            return {'error': str(e)}
    
    def generate_feature_correlation_report(self, df_features: pd.DataFrame) -> Dict[str, Any]:
        """Generar reporte de correlaciones entre features"""
        logger.info("=== ANÁLISIS DE CORRELACIONES ===")
        
        try:
            numeric_cols = df_features.select_dtypes(include=[np.number]).columns.tolist()
            
            if 'total_points' not in numeric_cols:
                logger.warning("Target 'total_points' no encontrado")
                return {}
            
            # Calcular correlaciones con el target
            correlations = df_features[numeric_cols].corr()['total_points'].abs().sort_values(ascending=False)
            
            # Excluir el target mismo
            correlations = correlations.drop('total_points', errors='ignore')
            
            analysis = {
                'high_correlation_features': correlations.head(10).to_dict(),
                'low_correlation_features': correlations.tail(10).to_dict(),
                'potential_data_leakage': correlations[correlations > 0.95].to_dict(),
                'useless_features': correlations[correlations < 0.01].to_dict()
            }
            
            self._print_analysis_section("CORRELACIONES", analysis)
            return analysis
            
        except Exception as e:
            logger.error(f"Error en análisis de correlaciones: {str(e)}")
            return {'error': str(e)}
    
    def _print_analysis_section(self, title: str, analysis: Dict[str, Any]):
        """Imprimir sección de análisis"""
        print(f"\n{'='*60}")
        print(f"  {title}")
        print(f"{'='*60}")
        
        if title == "DATOS CRUDOS":
            print(f"📊 Forma del dataset: {analysis['shape']}")
            print(f"📈 Columnas numéricas: {len(analysis['numeric_columns'])}")
            print(f"📝 Columnas categóricas: {len(analysis['categorical_columns'])}")
            
            if analysis['missing_critical_columns']:
                print(f"⚠️  Columnas críticas faltantes: {analysis['missing_critical_columns']}")
            
            if 'total_points_stats' in analysis:
                stats = analysis['total_points_stats']
                print(f"🎯 Total Points - Media: {stats['mean']:.1f}, Rango: [{stats['min']:.1f}, {stats['max']:.1f}]")
        
        elif title == "CALIDAD DE FEATURES":
            print(f"📊 Total features: {analysis['total_features']}")
            print(f"🔢 Features numéricas: {analysis['numeric_features']}")
            
            stats = analysis['general_stats']
            print(f"⚠️  Features con NaN: {stats['features_with_nan']}")
            print(f"♾️  Features con infinitos: {stats['features_with_inf']}")
            print(f"📈 Features con extremos: {stats['features_with_extremes']}")
            print(f"📉 Features con baja varianza: {stats['low_variance_features']}")
            
            if analysis['severe_problems']:
                print(f"\n🚨 FEATURES CON PROBLEMAS SEVEROS:")
                for col, score, info in analysis['severe_problems'][:5]:
                    print(f"   {col} (severidad: {score}): {', '.join(info['issues'])}")
        
        elif title == "COMPATIBILIDAD CON MODELO":
            if 'error' in analysis:
                print(f"❌ Error: {analysis['error']}")
            else:
                print(f"✅ Features disponibles: {analysis['available_features']}")
                print(f"📊 Forma final: {analysis['shape_after_cleaning']}")
                print(f"🧹 NaN finales: {analysis['final_nan_count']}")
                print(f"♾️  Infinitos finales: {analysis['final_inf_count']}")
                
                if analysis['problematic_for_model']:
                    print(f"\n⚠️  Features problemáticas para el modelo:")
                    for issue in analysis['problematic_for_model'][:5]:
                        print(f"   {issue}")
        
        elif title == "CORRELACIONES":
            if 'error' in analysis:
                print(f"❌ Error: {analysis['error']}")
            else:
                print(f"🔗 Features más correlacionadas:")
                for feat, corr in list(analysis['high_correlation_features'].items())[:5]:
                    print(f"   {feat}: {corr:.3f}")
                
                if analysis['potential_data_leakage']:
                    print(f"\n🚨 Posible data leakage (corr > 0.95):")
                    for feat, corr in analysis['potential_data_leakage'].items():
                        print(f"   {feat}: {corr:.3f}")
    
    def run_full_debug(self) -> Dict[str, Any]:
        """Ejecutar debug completo"""
        logger.info("🔍 INICIANDO DEBUG COMPLETO DE TOTAL_POINTS FEATURES")
        
        # 1. Cargar datos
        df = self.load_test_data()
        if df.empty:
            logger.error("No se pudieron cargar datos")
            return {}
        
        # 2. Analizar datos crudos
        raw_analysis = self.analyze_raw_data(df)
        
        # 3. Generar y analizar features
        df_features, feature_analysis = self.generate_and_analyze_features(df)
        
        if df_features.empty:
            logger.error("No se pudieron generar features")
            return {'raw_analysis': raw_analysis}
        
        # 4. Probar compatibilidad con modelo
        model_analysis = self.test_model_compatibility(df_features)
        
        # 5. Análisis de correlaciones
        correlation_analysis = self.generate_feature_correlation_report(df_features)
        
        # 6. Compilar resultados
        results = {
            'raw_data': raw_analysis,
            'features': feature_analysis,
            'model_compatibility': model_analysis,
            'correlations': correlation_analysis,
            'df_features': df_features  # Para inspección manual
        }
        
        self._generate_summary_report(results)
        return results
    
    def _generate_summary_report(self, results: Dict[str, Any]):
        """Generar reporte resumen"""
        print(f"\n{'='*60}")
        print(f"  RESUMEN EJECUTIVO")
        print(f"{'='*60}")
        
        # Contar problemas críticos
        critical_issues = 0
        warnings = 0
        
        if 'features' in results and 'severe_problems' in results['features']:
            critical_issues += len(results['features']['severe_problems'])
        
        if 'model_compatibility' in results and 'problematic_for_model' in results['model_compatibility']:
            warnings += len(results['model_compatibility']['problematic_for_model'])
        
        if 'correlations' in results and 'potential_data_leakage' in results['correlations']:
            critical_issues += len(results['correlations']['potential_data_leakage'])
        
        print(f"🚨 Problemas críticos: {critical_issues}")
        print(f"⚠️  Advertencias: {warnings}")
        
        if critical_issues == 0 and warnings == 0:
            print(f"✅ Todo parece estar en orden!")
        elif critical_issues > 0:
            print(f"❌ Se detectaron problemas críticos que requieren atención inmediata")
        else:
            print(f"⚠️  Se detectaron advertencias que deberían revisarse")
        
        print(f"\n💡 Recomendaciones:")
        if critical_issues > 0:
            print(f"   1. Corregir features con problemas severos")
            print(f"   2. Eliminar features con data leakage")
            print(f"   3. Normalizar features con valores extremos")
        if warnings > 0:
            print(f"   4. Aplicar clipping a features con rangos muy grandes")
            print(f"   5. Considerar transformaciones logarítmicas para features con alta varianza")


def main():
    """Función principal"""
    debugger = TotalPointsFeatureDebugger()
    results = debugger.run_full_debug()
    
    # Guardar resultados para inspección posterior
    if results:
        logger.info("Debug completado. Resultados disponibles en la variable 'results'")
        
        # Opcionalmente, guardar a archivo
        try:
            import pickle
            with open('debug_results.pkl', 'wb') as f:
                pickle.dump(results, f)
            logger.info("Resultados guardados en 'debug_results.pkl'")
        except Exception as e:
            logger.warning(f"No se pudieron guardar resultados: {e}")
    
    return results


if __name__ == "__main__":
    results = main() 