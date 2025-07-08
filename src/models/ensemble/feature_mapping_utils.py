
"""
Funciones de Mapeo de Features para Ensemble
"""

import pandas as pd
import numpy as np
from typing import Dict, List

def create_intelligent_feature_mapping(expected_features: List[str], available_features: List[str]) -> Dict[str, str]:
    """
    Crear mapeo inteligente entre features esperadas y disponibles
    """
    mapping = {}
    
    # Mapeo directo para coincidencias exactas
    for feature in expected_features:
        if feature in available_features:
            mapping[feature] = feature
    
    # Mapeo inteligente para features similares
    unmapped_expected = [f for f in expected_features if f not in mapping]
    unmapped_available = [f for f in available_features if f not in mapping.values()]
    
    for expected in unmapped_expected:
        best_match = find_best_feature_match(expected, unmapped_available)
        if best_match:
            mapping[expected] = best_match
            unmapped_available.remove(best_match)
    
    return mapping

def find_best_feature_match(target: str, candidates: List[str]) -> str:
    """
    Encontrar la mejor coincidencia para una feature
    """
    target_lower = target.lower()
    
    # Buscar coincidencias por palabras clave
    for candidate in candidates:
        candidate_lower = candidate.lower()
        
        # Coincidencias de palabras clave principales
        target_words = set(target_lower.split('_'))
        candidate_words = set(candidate_lower.split('_'))
        
        # Calcular overlap
        overlap = len(target_words & candidate_words)
        total_words = len(target_words | candidate_words)
        
        if total_words > 0 and overlap / total_words > 0.5:
            return candidate
    
    return None

def generate_missing_features(df: pd.DataFrame, missing_features: List[str]) -> pd.DataFrame:
    """
    Generar features faltantes con valores por defecto inteligentes
    """
    df_result = df.copy()
    
    for feature in missing_features:
        default_value = get_smart_default_value(feature)
        df_result[feature] = default_value
    
    return df_result

def get_smart_default_value(feature_name: str) -> float:
    """
    Obtener valor por defecto inteligente para una feature
    """
    feature_lower = feature_name.lower()
    
    # Patrones específicos de NBA
    if 'pts' in feature_lower and ('avg' in feature_lower or 'ma_' in feature_lower):
        return 12.0  # Promedio de puntos
    elif 'ast' in feature_lower and ('avg' in feature_lower or 'ma_' in feature_lower):
        return 5.0   # Promedio de asistencias
    elif 'trb' in feature_lower and ('avg' in feature_lower or 'ma_' in feature_lower):
        return 8.0   # Promedio de rebotes
    elif '3p' in feature_lower and ('avg' in feature_lower or 'ma_' in feature_lower):
        return 1.5   # Promedio de triples
    
    # Patrones de eficiencia
    elif 'efficiency' in feature_lower or 'pct' in feature_lower or '%' in feature_lower:
        return 0.45  # 45% eficiencia típica
    elif 'rate' in feature_lower:
        return 0.5   # 50% rate típico
    
    # Patrones de tendencia
    elif 'trend' in feature_lower or 'factor' in feature_lower:
        return 1.0   # Factor neutro
    
    # Patrones booleanos
    elif 'is_' in feature_lower or feature_lower.startswith('has_'):
        return 0.0   # False por defecto
    
    # Por defecto
    else:
        return 0.0

def apply_feature_mapping_and_fill(df: pd.DataFrame, expected_features: List[str], 
                                 feature_mapping: Dict[str, str]) -> pd.DataFrame:
    """
    Aplicar mapeo de features y llenar faltantes
    """
    result_df = pd.DataFrame(index=df.index)
    
    for expected_feature in expected_features:
        if expected_feature in feature_mapping and feature_mapping[expected_feature] in df.columns:
            # Usar feature mapeada
            result_df[expected_feature] = df[feature_mapping[expected_feature]]
        else:
            # Generar valor por defecto
            result_df[expected_feature] = get_smart_default_value(expected_feature)
    
    # Limpiar datos
    result_df = result_df.replace([np.inf, -np.inf], 0).fillna(0)
    
    return result_df
