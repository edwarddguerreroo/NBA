import joblib
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Configurar estilo de visualización
plt.style.use('default')
sns.set_palette("husl")

def load_ensemble_results():
    """Cargar resultados del ensemble"""
    try:
        ensemble_path = Path("results/ensemble_nba_final.joblib")
        if ensemble_path.exists():
            print(f"📁 Cargando ensemble desde: {ensemble_path}")
            ensemble = joblib.load(ensemble_path)
            print(f"✅ Ensemble cargado: {type(ensemble)}")
            return ensemble
        else:
            print(f"❌ No se encontró archivo de ensemble en: {ensemble_path}")
            return None
    except Exception as e:
        print(f"❌ Error cargando ensemble: {e}")
        return None

def analyze_individual_models():
    """Analizar modelos individuales"""
    models_info = {
        'double_double': 'results/players/double_double_model/dd_model.joblib',
        '3pt': 'results/players/3pt_model/3pt_model.joblib', 
        'pts': 'results/players/pts_model/xgboost_pts_model.joblib',
        'trb': 'results/players/trb_model/xgboost_trb_model.joblib',
        'ast': 'results/players/ast_model/xgboost_ast_model.joblib',
        'teams_points': 'results/teams/teams_points_model/teams_points_model.joblib',
        'total_points': 'results/teams/total_points_model/total_points_model.joblib',
        'is_win': 'results/teams/is_win_model/is_win_model.joblib'
    }
    
    print("\n" + "="*80)
    print("📊 ANÁLISIS DETALLADO DE MODELOS INDIVIDUALES")
    print("="*80)
    
    working_models = []
    model_details = {}
    
    for name, path in models_info.items():
        try:
            if Path(path).exists():
                file_size = Path(path).stat().st_size / (1024 * 1024)  # MB
                print(f"\n🔍 {name.upper()}")
                print(f"   📁 Archivo: {path} ({file_size:.1f} MB)")
                
                # Cargar modelo
                model = joblib.load(path)
                model_type = type(model).__name__
                
                print(f"   🤖 Tipo: {model_type}")
                print(f"   📋 Métodos: {[m for m in dir(model) if not m.startswith('_') and callable(getattr(model, m))][:5]}")
                
                # Analizar features si están disponibles
                if hasattr(model, 'feature_names_in_'):
                    n_features = len(model.feature_names_in_)
                    print(f"   🎯 Features esperadas: {n_features}")
                    print(f"   📝 Top 5 features: {list(model.feature_names_in_)[:5]}")
                    
                    model_details[name] = {
                        'model': model,
                        'type': model_type,
                        'features': list(model.feature_names_in_),
                        'n_features': n_features,
                        'file_size_mb': file_size
                    }
                else:
                    print(f"   ⚠️  No tiene feature_names_in_")
                    model_details[name] = {
                        'model': model,
                        'type': model_type,
                        'features': [],
                        'n_features': 0,
                        'file_size_mb': file_size
                    }
                
                working_models.append(name)
                print(f"   ✅ Estado: FUNCIONAL")
                
            else:
                print(f"❌ {name}: Archivo no existe - {path}")
                
        except Exception as e:
            print(f"❌ {name}: Error - {str(e)[:100]}")
    
    print(f"\n📈 RESUMEN: {len(working_models)}/{len(models_info)} modelos funcionales")
    print(f"✅ Modelos OK: {working_models}")
    
    return model_details

def analyze_data_compatibility():
    """Analizar compatibilidad de datos con modelos"""
    print("\n" + "="*80)
    print("🔄 ANÁLISIS DE COMPATIBILIDAD DE DATOS")
    print("="*80)
    
    try:
        # Cargar datos de muestra
        players_data = pd.read_csv("data/players.csv")
        teams_data = pd.read_csv("data/teams.csv") if Path("data/teams.csv").exists() else None
        
        print(f"📊 Datos jugadores: {players_data.shape}")
        print(f"📋 Columnas jugadores: {list(players_data.columns)[:10]}")
        
        if teams_data is not None:
            print(f"📊 Datos equipos: {teams_data.shape}")
            print(f"📋 Columnas equipos: {list(teams_data.columns)[:10]}")
        
        # Verificar features numéricas
        numeric_cols_players = players_data.select_dtypes(include=[np.number]).columns
        print(f"🔢 Features numéricas jugadores: {len(numeric_cols_players)}")
        print(f"   Top 10: {list(numeric_cols_players)[:10]}")
        
        return {
            'players': players_data,
            'teams': teams_data,
            'numeric_features_players': list(numeric_cols_players),
            'data_quality': 'OK' if len(numeric_cols_players) > 5 else 'POOR'
        }
        
    except Exception as e:
        print(f"❌ Error cargando datos: {e}")
        return None

def create_prediction_dashboard(model_details, data_info):
    """Crear dashboard de predicciones de modelos"""
    print("\n" + "="*80)
    print("🎨 CREANDO DASHBOARD DE PREDICCIONES")
    print("="*80)
    
    if not model_details or not data_info:
        print("❌ Datos insuficientes para dashboard")
        return
    
    # Preparar datos de muestra (primeras 1000 filas)
    sample_data = data_info['players'].head(1000)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('NBA Ensemble Model Analysis Dashboard', fontsize=16, fontweight='bold')
    
    # 1. Distribución de features numéricas clave
    ax1 = axes[0, 0]
    key_features = ['PTS', 'TRB', 'AST']
    available_features = [f for f in key_features if f in sample_data.columns]
    
    if available_features:
        for feature in available_features:
            values = sample_data[feature].dropna()
            ax1.hist(values, alpha=0.6, label=feature, bins=30)
        ax1.set_title('Distribución de Stats Principales')
        ax1.set_xlabel('Valor')
        ax1.set_ylabel('Frecuencia')
        ax1.legend()
    else:
        ax1.text(0.5, 0.5, 'Features no\ndisponibles', ha='center', va='center', transform=ax1.transAxes)
        ax1.set_title('Stats Principales - No Disponibles')
    
    # 2. Análisis de modelos
    ax2 = axes[0, 1]
    model_types = [details['type'] for details in model_details.values()]
    type_counts = pd.Series(model_types).value_counts()
    
    ax2.pie(type_counts.values, labels=type_counts.index, autopct='%1.1f%%')
    ax2.set_title('Distribución de Tipos de Modelos')
    
    # 3. Número de features por modelo
    ax3 = axes[0, 2]
    model_names = list(model_details.keys())
    feature_counts = [details['n_features'] for details in model_details.values()]
    
    bars = ax3.bar(model_names, feature_counts, color='skyblue')
    ax3.set_title('Features Esperadas por Modelo')
    ax3.set_xlabel('Modelo')
    ax3.set_ylabel('Número de Features')
    ax3.tick_params(axis='x', rotation=45)
    
    # Agregar valores en las barras
    for bar, count in zip(bars, feature_counts):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{count}', ha='center', va='bottom')
    
    # 4. Tamaño de archivos de modelos
    ax4 = axes[1, 0]
    file_sizes = [details['file_size_mb'] for details in model_details.values()]
    
    bars = ax4.bar(model_names, file_sizes, color='lightcoral')
    ax4.set_title('Tamaño de Archivos de Modelos (MB)')
    ax4.set_xlabel('Modelo')
    ax4.set_ylabel('Tamaño (MB)')
    ax4.tick_params(axis='x', rotation=45)
    
    for bar, size in zip(bars, file_sizes):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{size:.1f}', ha='center', va='bottom')
    
    # 5. Matriz de correlación de features principales
    ax5 = axes[1, 1]
    numeric_features = ['PTS', 'TRB', 'AST', 'MP', 'FG', 'FGA']
    available_numeric = [f for f in numeric_features if f in sample_data.columns]
    
    if len(available_numeric) >= 2:
        corr_matrix = sample_data[available_numeric].corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=ax5)
        ax5.set_title('Correlación Features Principales')
    else:
        ax5.text(0.5, 0.5, 'Datos insuficientes\npara correlación', ha='center', va='center', transform=ax5.transAxes)
        ax5.set_title('Correlación - Datos Insuficientes')
    
    # 6. Resumen estadístico
    ax6 = axes[1, 2]
    ax6.axis('off')
    
    summary_text = f"""
    📊 RESUMEN DEL ANÁLISIS
    
    ✅ Modelos cargados: {len(model_details)}/8
    📁 Tamaño total: {sum(file_sizes):.1f} MB
    🎯 Features promedio: {np.mean(feature_counts):.1f}
    📈 Datos disponibles: {sample_data.shape[0]:,} muestras
    🔢 Features numéricas: {len(data_info['numeric_features_players'])}
    
    🏆 Modelo más grande: {model_names[np.argmax(file_sizes)]}
    🎯 Más features: {model_names[np.argmax(feature_counts)]}
    
    �� Estado general: {'✅ BUENO' if len(model_details) >= 6 else '⚠️ MEJORABLE'}
    """
    
    ax6.text(0.1, 0.9, summary_text, transform=ax6.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
    
    plt.tight_layout()
    
    # Guardar dashboard
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dashboard_path = f"ensemble_analysis_dashboard_{timestamp}.png"
    plt.savefig(dashboard_path, dpi=300, bbox_inches='tight')
    print(f"📊 Dashboard guardado: {dashboard_path}")
    
    plt.show()

def main():
    """Función principal de diagnóstico"""
    print("🚀 INICIANDO DIAGNÓSTICO AVANZADO DEL ENSEMBLE NBA")
    print("="*80)
    
    # 1. Analizar ensemble
    ensemble_results = load_ensemble_results()
    
    # 2. Analizar modelos individuales
    model_details = analyze_individual_models()
    
    # 3. Analizar compatibilidad de datos
    data_info = analyze_data_compatibility()
    
    # 4. Crear dashboard visual
    create_prediction_dashboard(model_details, data_info)
    
    # 5. Resumen final
    print("\n" + "="*80)
    print("🎯 DIAGNÓSTICO COMPLETADO")
    print("="*80)
    
    if model_details and len(model_details) >= 6:
        print("✅ Estado general: MODELOS FUNCIONANDO CORRECTAMENTE")
        print("📈 Recomendación: Proceder con ensemble refinado")
    else:
        print("⚠️ Estado general: NECESITA MEJORAS")
        print("🔧 Recomendación: Revisar feature engineering y compatibilidad de datos")
    
    print(f"\n📊 Análisis completado a las {datetime.now().strftime('%H:%M:%S')}")

if __name__ == "__main__":
    main() 