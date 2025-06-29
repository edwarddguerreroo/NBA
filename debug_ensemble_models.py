"""
Debug Específico para Errores de Arrays Ambiguos en Ensemble NBA
===============================================================

Script para identificar y solucionar el problema específico de arrays ambiguos
que está afectando a todos los modelos durante la generación de predicciones.
"""

import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from src.preprocessing.data_loader import NBADataLoader

def debug_array_ambiguous_error():
    """Debuggear específicamente el error de arrays ambiguos"""
    print("🔍 DEBUG: Análisis de Arrays Ambiguos en Modelos NBA")
    print("=" * 60)
    
    # Configurar modelos existentes
    models_to_test = {
        'double_double': 'results/players/double_double_model/dd_model.joblib',
        '3pt': 'results/3pt_model/3pt_model.joblib',
        'pts': 'results/players/pts_model/xgboost_pts_model.joblib',
        'trb': 'results/players/trb_model/xgboost_trb_model.joblib',
        'ast': 'results/players/ast_model/xgboost_ast_model.joblib'
    }
    
    # Cargar datos
    print("📊 Cargando datos NBA...")
    try:
        data_loader = NBADataLoader("data/players.csv", "data/height.csv", "data/teams.csv")
        df_players, df_teams = data_loader.load_data()
        print(f"✅ Datos cargados: {len(df_players)} jugadores, {len(df_teams)} equipos")
    except Exception as e:
        print(f"❌ Error cargando datos: {e}")
        return
    
    # Tomar muestra pequeña para debuggear
    df_sample = df_players.head(100).copy()
    print(f"🔬 Usando muestra de {len(df_sample)} registros para debug")
    
    # Probar cada modelo
    for model_name, model_path in models_to_test.items():
        print(f"\n🧪 Probando modelo: {model_name}")
        print(f"   Archivo: {model_path}")
        
        try:
            # Verificar si existe el archivo
            if not Path(model_path).exists():
                print(f"   ❌ Archivo no existe")
                continue
            
            # Cargar modelo
            print(f"   📁 Cargando modelo...")
            model = joblib.load(model_path)
            print(f"   ✅ Modelo cargado: {type(model).__name__}")
            
            # Verificar features esperadas
            expected_features = getattr(model, 'feature_names_in_', None)
            if expected_features is None:
                print(f"   ⚠️ Sin feature_names_in_")
                continue
            
            print(f"   📋 Features esperadas: {len(expected_features)}")
            
            # Verificar features disponibles en datos
            available_features = [f for f in expected_features if f in df_sample.columns]
            missing_features = [f for f in expected_features if f not in df_sample.columns]
            
            print(f"   ✅ Features disponibles: {len(available_features)}/{len(expected_features)}")
            print(f"   ❌ Features faltantes: {len(missing_features)}")
            
            # Mostrar algunas features esperadas vs disponibles
            if len(expected_features) > 0:
                print(f"   📝 Primeras 10 features esperadas:")
                for i, feature in enumerate(expected_features[:10]):
                    status = "✅" if feature in df_sample.columns else "❌"
                    print(f"      {status} {feature}")
                
                print(f"   📝 Columnas disponibles en datos (primeras 10):")
                for col in df_sample.columns[:10]:
                    print(f"      📊 {col}")
            
            # Crear DataFrame de prueba
            print(f"   🔧 Preparando datos de prueba...")
            X_test = pd.DataFrame()
            
            for feature in expected_features:
                if feature in df_sample.columns:
                    # Limpiar datos
                    clean_data = pd.to_numeric(df_sample[feature], errors='coerce').fillna(0)
                    clean_data = clean_data.replace([np.inf, -np.inf], 0)
                    X_test[feature] = clean_data
                else:
                    # Feature faltante - rellenar con 0
                    X_test[feature] = 0
            
            # Verificar que no hay problemas en los datos
            print(f"   🔎 Verificando calidad de datos...")
            print(f"      - Shape: {X_test.shape}")
            print(f"      - NaN: {X_test.isna().sum().sum()}")
            print(f"      - Inf: {np.isinf(X_test.select_dtypes(include=[np.number])).sum().sum()}")
            
            # Verificar tipos de datos
            non_numeric_cols = []
            for col in X_test.columns:
                if not pd.api.types.is_numeric_dtype(X_test[col]):
                    non_numeric_cols.append(col)
            
            if non_numeric_cols:
                print(f"      ⚠️ Columnas no numéricas: {len(non_numeric_cols)}")
                for col in non_numeric_cols[:5]:  # Mostrar solo las primeras 5
                    print(f"         - {col}: {X_test[col].dtype}")
            else:
                print(f"      ✅ Todas las columnas son numéricas")
            
            # DIAGNÓSTICO ESPECÍFICO DEL ERROR
            print(f"   🚨 DIAGNÓSTICO ESPECÍFICO DE ARRAYS AMBIGUOS:")
            
            # Intentar diferentes métodos que podrían causar el error
            test_operations = [
                ("X_test.shape", lambda: X_test.shape),
                ("X_test.empty", lambda: X_test.empty),
                ("len(X_test)", lambda: len(X_test)),
                ("X_test is None", lambda: X_test is None),
                ("X_test.values", lambda: X_test.values.shape),
                ("model.predict(X_test[:1])", lambda: model.predict(X_test.iloc[:1])),
                ("model.predict(X_test[:5])", lambda: model.predict(X_test.iloc[:5]))
            ]
            
            for op_name, operation in test_operations:
                try:
                    result = operation()
                    print(f"      ✅ {op_name}: {result}")
                except Exception as op_error:
                    error_msg = str(op_error)
                    if "ambiguous" in error_msg.lower() or "truth value" in error_msg.lower():
                        print(f"      🎯 {op_name}: ENCONTRADO ARRAY AMBIGUO!")
                        print(f"         Error: {error_msg}")
                        
                        # INVESTIGACIÓN PROFUNDA
                        if "predict" in op_name:
                            print(f"         🔬 INVESTIGACIÓN PROFUNDA DEL MODELO:")
                            try:
                                print(f"            - Tipo modelo: {type(model)}")
                                print(f"            - Atributos: {[attr for attr in dir(model) if not attr.startswith('_')][:10]}")
                                
                                # Si es StackingRegressor/Classifier, revisar estimadores
                                if hasattr(model, 'estimators_'):
                                    print(f"            - Es ensemble con {len(model.estimators_)} estimadores")
                                    for i, estimator in enumerate(model.estimators_):
                                        print(f"              Estimador {i}: {type(estimator)}")
                                
                                # Probar con datos muy simples
                                simple_X = pd.DataFrame(np.zeros((1, len(expected_features))), columns=expected_features)
                                try:
                                    simple_pred = model.predict(simple_X)
                                    print(f"            ✅ Predicción simple funciona: {simple_pred}")
                                except Exception as simple_error:
                                    print(f"            ❌ Predicción simple falla: {simple_error}")
                                    
                            except Exception as deep_error:
                                print(f"            ❌ Error en investigación profunda: {deep_error}")
                    else:
                        print(f"      ❌ {op_name}: {error_msg}")
                        
        except Exception as model_error:
            print(f"   ❌ Error general con modelo {model_name}: {model_error}")
    
    print(f"\n🏁 Debug completado")

if __name__ == "__main__":
    debug_array_ambiguous_error() 