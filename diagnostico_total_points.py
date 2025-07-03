#!/usr/bin/env python3
"""
DIAGNÓSTICO TOTAL_POINTS - ANÁLISIS DE VALORES EXTREMOS
====================================================

Script para identificar la causa raíz de los valores extremos
en las predicciones del modelo total_points.
"""

import pandas as pd
import numpy as np
import joblib
import logging
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from config.logging_config import NBALogger
logger = NBALogger.get_logger(__name__)

def main():
    logger.info("🔍 DIAGNÓSTICO TOTAL_POINTS - ANÁLISIS DE VALORES EXTREMOS")
    logger.info("=" * 80)
    
    try:
        # PASO 1: Cargar datos originales
        logger.info("📊 PASO 1: Cargando datos...")
        from src.preprocessing.data_loader import NBADataLoader
        data_loader = NBADataLoader('data/players.csv', 'data/height.csv', 'data/teams.csv')
        df_players, df_teams = data_loader.load_data()
        
        logger.info(f"   Datos cargados: {len(df_teams)} equipos")
        
        # PASO 2: Cargar modelo y feature engineer
        logger.info("🤖 PASO 2: Cargando modelo total_points...")
        model_path = Path("trained_models/total_points_model.joblib")
        
        if not model_path.exists():
            logger.error(f"   ❌ Modelo no encontrado en {model_path}")
            return
        
        model = joblib.load(model_path)
        logger.info(f"   ✅ Modelo cargado: {type(model).__name__}")
        
        # Información del modelo
        if hasattr(model, 'n_features_in_'):
            logger.info(f"   Features esperadas: {model.n_features_in_}")
        if hasattr(model, 'feature_names_in_'):
            logger.info(f"   Primeras features: {list(model.feature_names_in_[:10])}")
        
        # PASO 3: Cargar FeatureEngineer
        logger.info("🔧 PASO 3: Cargando FeatureEngineer...")
        from src.models.teams.total_points.features_total_points import TotalPointsFeatureEngine
        feature_engineer = TotalPointsFeatureEngine()
        
        # PASO 4: Generar features paso a paso
        logger.info("⚙️ PASO 4: Generando features paso a paso...")
        df_work = df_teams.copy()
        
        logger.info("   4.1 Ejecutando create_features()...")
        try:
            features_df = feature_engineer.create_features(df_work)
            logger.info(f"   ✅ Features generadas: {features_df.shape}")
            logger.info(f"   Columnas: {list(features_df.columns[:10])}...")
            
            # Análisis estadístico de features
            logger.info("   📈 Análisis estadístico de features:")
            for col in features_df.select_dtypes(include=[np.number]).columns[:15]:
                values = features_df[col]
                logger.info(f"      {col}: min={values.min():.2f}, max={values.max():.2f}, mean={values.mean():.2f}, std={values.std():.2f}")
            
        except Exception as e:
            logger.error(f"   ❌ Error generando features: {e}")
            import traceback
            logger.error(f"   Traceback: {traceback.format_exc()}")
            return
        
        # PASO 5: Análisis de compatibilidad con modelo
        logger.info("🔄 PASO 5: Análisis de compatibilidad...")
        
        # Verificar dimensiones
        expected_features = model.n_features_in_ if hasattr(model, 'n_features_in_') else "Unknown"
        actual_features = features_df.shape[1]
        
        logger.info(f"   Features esperadas por modelo: {expected_features}")
        logger.info(f"   Features generadas: {actual_features}")
        
        if hasattr(model, 'feature_names_in_'):
            expected_names = list(model.feature_names_in_)
            available_names = list(features_df.columns)
            
            # Features faltantes
            missing = [f for f in expected_names if f not in available_names]
            # Features extra
            extra = [f for f in available_names if f not in expected_names]
            
            logger.info(f"   Features faltantes: {len(missing)}")
            if missing:
                logger.info(f"      Primeras 10: {missing[:10]}")
            
            logger.info(f"   Features extra: {len(extra)}")
            if extra:
                logger.info(f"      Primeras 10: {extra[:10]}")
        
        # PASO 6: Intentar predicción con features originales
        logger.info("🎯 PASO 6: Predicción con features sin procesar...")
        
        try:
            # Preparar datos para el modelo
            if hasattr(model, 'feature_names_in_'):
                # Usar solo las features que el modelo espera
                expected_features = model.feature_names_in_
                available_features = [f for f in expected_features if f in features_df.columns]
                missing_features = [f for f in expected_features if f not in features_df.columns]
                
                logger.info(f"   Features disponibles: {len(available_features)}/{len(expected_features)}")
                
                if missing_features:
                    logger.warning(f"   Creando features faltantes con valores por defecto...")
                    for feature in missing_features[:5]:  # Solo las primeras 5 para probar
                        features_df[feature] = 0.0
                
                # Preparar X con el orden correcto
                X = features_df[expected_features].fillna(0)
            else:
                # Usar las primeras N features numéricas
                numeric_cols = features_df.select_dtypes(include=[np.number]).columns
                if expected_features != "Unknown":
                    X = features_df[numeric_cols[:expected_features]].fillna(0)
                else:
                    X = features_df[numeric_cols].fillna(0)
            
            logger.info(f"   X preparado: {X.shape}")
            logger.info(f"   Valores X - min: {X.values.min():.2f}, max: {X.values.max():.2f}")
            
            # Predicción
            predictions = model.predict(X)
            
            logger.info(f"   ✅ Predicción exitosa: {len(predictions)} valores")
            logger.info(f"   Rango predicciones: [{predictions.min():.2f}, {predictions.max():.2f}]")
            logger.info(f"   Media: {predictions.mean():.2f}")
            logger.info(f"   Desviación estándar: {predictions.std():.2f}")
            
            # ANÁLISIS DE VALORES EXTREMOS
            logger.info("🚨 PASO 7: Análisis de valores extremos...")
            
            # Encontrar valores extremos
            q01 = np.percentile(predictions, 1)
            q99 = np.percentile(predictions, 99)
            
            logger.info(f"   Percentil 1%: {q01:.2f}")
            logger.info(f"   Percentil 99%: {q99:.2f}")
            
            # Índices de valores extremos
            extreme_low = np.where(predictions < q01)[0]
            extreme_high = np.where(predictions > q99)[0]
            
            logger.info(f"   Valores extremos bajos: {len(extreme_low)}")
            logger.info(f"   Valores extremos altos: {len(extreme_high)}")
            
            if len(extreme_low) > 0:
                logger.info("   📊 Análisis de casos extremos bajos:")
                for i in extreme_low[:3]:  # Primeros 3 casos
                    logger.info(f"      Índice {i}: predicción={predictions[i]:.2f}")
                    logger.info(f"         Features: {dict(zip(X.columns[:5], X.iloc[i][:5]))}")
            
            if len(extreme_high) > 0:
                logger.info("   📊 Análisis de casos extremos altos:")
                for i in extreme_high[:3]:  # Primeros 3 casos
                    logger.info(f"      Índice {i}: predicción={predictions[i]:.2f}")
                    logger.info(f"         Features: {dict(zip(X.columns[:5], X.iloc[i][:5]))}")
            
            # PASO 8: Análisis de features problemáticas
            logger.info("🔍 PASO 8: Identificando features problemáticas...")
            
            # Buscar features con valores extremos
            for col in X.columns:
                values = X[col]
                if values.std() > 0:  # Solo si hay variabilidad
                    q01_feat = np.percentile(values, 1)
                    q99_feat = np.percentile(values, 99)
                    
                    if abs(q01_feat) > 1000 or abs(q99_feat) > 1000:
                        logger.warning(f"   Feature problemática: {col}")
                        logger.warning(f"      Rango: [{q01_feat:.2f}, {q99_feat:.2f}]")
                        logger.warning(f"      Media: {values.mean():.2f}, Std: {values.std():.2f}")
            
        except Exception as e:
            logger.error(f"   ❌ Error en predicción: {e}")
            import traceback
            logger.error(f"   Traceback: {traceback.format_exc()}")
        
        # PASO 9: Recomendaciones
        logger.info("💡 PASO 9: Recomendaciones para solucionar valores extremos:")
        logger.info("   1. Aplicar normalización/estandarización a features numéricas")
        logger.info("   2. Detectar y limitar outliers en features de entrada")
        logger.info("   3. Aplicar clipping a predicciones finales")
        logger.info("   4. Verificar que el modelo fue entrenado con datos similares")
        logger.info("   5. Considerar re-entrenar el modelo con mejor preprocesamiento")
        
    except Exception as e:
        logger.error(f"❌ Error en diagnóstico: {e}")
        import traceback
        logger.error(f"Traceback completo: {traceback.format_exc()}")

if __name__ == "__main__":
    main() 