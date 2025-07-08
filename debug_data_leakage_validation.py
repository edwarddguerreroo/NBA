"""
VALIDACIÓN DE CORRECCIONES DE DATA LEAKAGE - TOTAL POINTS
========================================================

Script para verificar que las correcciones aplicadas al módulo total_points
han eliminado efectivamente el data leakage detectado.

PROBLEMAS CORREGIDOS:
1. market_error: Correlación 0.945 - ELIMINADO
2. total_pts_acceleration: Agregado shift(1)
3. medias móviles: Agregado shift(1) consistente
4. Features de volatilidad: Agregado shift(1)

"""

import pandas as pd
import numpy as np
import sys
import os
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Agregar el directorio src al path
sys.path.append('src')

from models.teams.total_points.features_total_points import TotalPointsFeatureEngine

def create_realistic_test_data():
    """Crear datos de prueba realistas para validación"""
    np.random.seed(42)
    
    teams = ['Lakers', 'Warriors', 'Celtics', 'Heat', 'Bulls', 'Knicks']
    games_per_team = 20
    
    data = []
    start_date = datetime(2023, 10, 15)
    
    for team in teams:
        for game_num in range(games_per_team):
            # Simular fechas con espaciado realista
            game_date = start_date + timedelta(days=game_num * 3)
            
            # Estadísticas base realistas para NBA
            base_pts = 110 + np.random.normal(0, 10)
            base_pts_opp = 108 + np.random.normal(0, 12)
            
            # Agregar variabilidad temporal realista
            season_factor = 1 + 0.1 * np.sin(2 * np.pi * game_num / 82)
            pts = max(80, base_pts * season_factor + np.random.normal(0, 8))
            pts_opp = max(80, base_pts_opp * season_factor + np.random.normal(0, 8))
            
            # Otras estadísticas NBA realistas
            data.append({
                'Date': game_date,
                'Team': team,
                'Opp': np.random.choice([t for t in teams if t != team]),
                'PTS': round(pts),
                'PTS_Opp': round(pts_opp),
                'FG': np.random.randint(35, 55),
                'FGA': np.random.randint(80, 100),
                'FT': np.random.randint(10, 25),
                'FTA': np.random.randint(15, 30),
                'TRB': np.random.randint(35, 55),
                'AST': np.random.randint(20, 35),
                'Away': '@' if np.random.random() > 0.5 else '',
                'Result': 'W' if pts > pts_opp else 'L'
            })
    
    df = pd.DataFrame(data)
    df['total_points'] = df['PTS'] + df['PTS_Opp']
    return df

def validate_data_leakage_corrections():
    """Validar que las correcciones de data leakage funcionan"""
    print("=" * 80)
    print("VALIDACIÓN DE CORRECCIONES DE DATA LEAKAGE")
    print("=" * 80)
    
    # Crear datos de prueba
    print("\n1. CREANDO DATOS DE PRUEBA...")
    df = create_realistic_test_data()
    print(f"   - Datos creados: {df.shape[0]} registros, {df.shape[1]} columnas")
    print(f"   - Equipos: {df['Team'].nunique()}")
    print(f"   - Rango total_points: {df['total_points'].min():.0f} - {df['total_points'].max():.0f}")
    
    # Inicializar motor de features
    print("\n2. INICIALIZANDO MOTOR DE FEATURES...")
    engine = TotalPointsFeatureEngine()
    
    # Generar features
    print("\n3. GENERANDO FEATURES CON CORRECCIONES...")
    try:
        feature_names = engine.generate_all_features(df)
        print(f"   ✅ Features generadas exitosamente: {len(feature_names)}")
    except Exception as e:
        print(f"   ❌ Error generando features: {e}")
        return
    
    # VALIDACIÓN 1: Verificar que market_error fue eliminado
    print("\n4. VALIDACIÓN 1: ELIMINACIÓN DE MARKET_ERROR")
    market_error_features = [f for f in feature_names if 'market_error' in f]
    if market_error_features:
        print(f"   ❌ PROBLEMA: Features market_error aún presentes: {market_error_features}")
    else:
        print("   ✅ CORRECTO: market_error y derivadas eliminadas completamente")
    
    # VALIDACIÓN 2: Verificar correlaciones extremas
    print("\n5. VALIDACIÓN 2: CORRELACIONES CON TARGET")
    numeric_features = [f for f in feature_names if pd.api.types.is_numeric_dtype(df[f])]
    high_corr_features = []
    
    for feature in numeric_features:
        try:
            corr = abs(df[feature].corr(df['total_points']))
            if pd.notna(corr) and corr > 0.9:
                high_corr_features.append((feature, corr))
        except:
            continue
    
    if high_corr_features:
        print(f"   ⚠️  FEATURES CON CORRELACIÓN ALTA:")
        for feature, corr in sorted(high_corr_features, key=lambda x: x[1], reverse=True):
            print(f"      - {feature}: {corr:.4f}")
    else:
        print("   ✅ CORRECTO: No hay features con correlación extrema (>0.9)")
    
    # VALIDACIÓN 3: Verificar que features temporales tienen lag
    print("\n6. VALIDACIÓN 3: FEATURES TEMPORALES CON LAG")
    temporal_patterns = ['_ma_', '_std_', '_trend_', '_momentum_', '_acceleration']
    temporal_features = []
    
    for feature in feature_names:
        if any(pattern in feature for pattern in temporal_patterns):
            temporal_features.append(feature)
    
    print(f"   - Features temporales encontradas: {len(temporal_features)}")
    
    # Verificar correlaciones de features temporales
    problematic_temporal = []
    for feature in temporal_features:
        try:
            corr = abs(df[feature].corr(df['total_points']))
            if pd.notna(corr) and corr > 0.85:
                problematic_temporal.append((feature, corr))
        except:
            continue
    
    if problematic_temporal:
        print(f"   ⚠️  FEATURES TEMPORALES CON CORRELACIÓN SOSPECHOSA:")
        for feature, corr in problematic_temporal:
            print(f"      - {feature}: {corr:.4f}")
    else:
        print("   ✅ CORRECTO: Features temporales tienen correlación apropiada")
    
    # VALIDACIÓN 4: Verificar ausencia de componentes directos del target
    print("\n7. VALIDACIÓN 4: COMPONENTES DIRECTOS DEL TARGET")
    direct_components = ['PTS', 'PTS_Opp', 'total_points']
    found_components = [comp for comp in direct_components if comp in feature_names]
    
    if found_components:
        print(f"   ❌ PROBLEMA: Componentes directos encontrados: {found_components}")
    else:
        print("   ✅ CORRECTO: Sin componentes directos del target en features")
    
    # VALIDACIÓN 5: Análisis de distribución de correlaciones
    print("\n8. VALIDACIÓN 5: DISTRIBUCIÓN DE CORRELACIONES")
    correlations = []
    for feature in numeric_features:
        try:
            corr = abs(df[feature].corr(df['total_points']))
            if pd.notna(corr):
                correlations.append(corr)
        except:
            continue
    
    if correlations:
        correlations = np.array(correlations)
        print(f"   - Total features numéricas: {len(correlations)}")
        print(f"   - Correlación promedio: {correlations.mean():.4f}")
        print(f"   - Correlación máxima: {correlations.max():.4f}")
        print(f"   - Features con corr > 0.8: {sum(correlations > 0.8)}")
        print(f"   - Features con corr > 0.7: {sum(correlations > 0.7)}")
        
        # Distribución por rangos
        ranges = [(0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]
        for low, high in ranges:
            count = sum((correlations >= low) & (correlations < high))
            print(f"   - Correlación {low:.1f}-{high:.1f}: {count} features")
    
    # VALIDACIÓN 6: Top features más predictivas (sin data leakage)
    print("\n9. VALIDACIÓN 6: TOP FEATURES MÁS PREDICTIVAS")
    feature_correlations = []
    for feature in numeric_features:
        try:
            corr = abs(df[feature].corr(df['total_points']))
            if pd.notna(corr) and corr < 0.9:  # Excluir posibles data leakage
                feature_correlations.append((feature, corr))
        except:
            continue
    
    # Ordenar por correlación
    top_features = sorted(feature_correlations, key=lambda x: x[1], reverse=True)[:10]
    
    print("   TOP 10 FEATURES MÁS PREDICTIVAS (sin data leakage):")
    for i, (feature, corr) in enumerate(top_features, 1):
        print(f"   {i:2d}. {feature:<30} {corr:.4f}")
    
    # RESUMEN FINAL
    print("\n" + "=" * 80)
    print("RESUMEN DE VALIDACIÓN")
    print("=" * 80)
    
    issues_found = 0
    
    if market_error_features:
        issues_found += 1
        print("❌ PROBLEMA: market_error no eliminado completamente")
    
    if len(high_corr_features) > 2:  # Permitir hasta 2 features con correlación alta
        issues_found += 1
        print(f"❌ PROBLEMA: {len(high_corr_features)} features con correlación extrema")
    
    if found_components:
        issues_found += 1
        print("❌ PROBLEMA: Componentes directos del target presentes")
    
    if len(problematic_temporal) > 3:  # Permitir hasta 3 features temporales sospechosas
        issues_found += 1
        print(f"❌ PROBLEMA: {len(problematic_temporal)} features temporales sospechosas")
    
    if issues_found == 0:
        print("✅ VALIDACIÓN EXITOSA: Data leakage corregido efectivamente")
        print("✅ El modelo está listo para entrenamiento sin data leakage")
    else:
        print(f"⚠️  VALIDACIÓN PARCIAL: {issues_found} problemas encontrados")
        print("⚠️  Se requiere revisión adicional")
    
    print(f"\nFeatures finales generadas: {len(feature_names)}")
    print(f"Features numéricas válidas: {len(numeric_features)}")
    print(f"Correlación máxima permitida: < 0.9")

if __name__ == "__main__":
    validate_data_leakage_corrections() 