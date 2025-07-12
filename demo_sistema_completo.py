#!/usr/bin/env python3
"""
Demo Sistema Completo Modelo vs Mercado
=======================================

Este script demuestra el funcionamiento completo del sistema de comparación
modelo vs mercado con filtros más permisivos para mostrar resultados.

Características demostradas:
1. Obtención de datos reales del mercado
2. Comparación con predicciones del modelo
3. Cálculo de ventajas estadísticas
4. Identificación de oportunidades
5. Análisis de valor esperado y Kelly fraction
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime
from utils.bookmakers.bookmakers_integration import BookmakersIntegration
from utils.bookmakers.config import get_config
import os

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_realistic_predictions():
    """
    Crea predicciones más realistas basadas en los datos del mercado.
    """
    # Predicciones ajustadas para mostrar oportunidades
    predictions = [
        {
            'player_name': 'Gobert, Rudy',
            'team': 'MIN',
            'predicted_PTS': 8.2,   # Predicción por debajo del mercado
            'predicted_AST': 0.8,
            'predicted_TRB': 12.5,  # Predicción por encima del mercado
            'confidence': 0.89,
            'model_version': 'v2.1'
        },
        {
            'player_name': 'Edwards, Anthony',
            'team': 'MIN',
            'predicted_PTS': 25.8,
            'predicted_AST': 4.2,
            'predicted_TRB': 5.8,
            'confidence': 0.92,
            'model_version': 'v2.1'
        },
        {
            'player_name': 'Davis, Anthony',
            'team': 'LAL',
            'predicted_PTS': 26.5,  # Predicción por encima del mercado
            'predicted_AST': 2.8,
            'predicted_TRB': 11.2,
            'confidence': 0.88,
            'model_version': 'v2.1'
        },
        {
            'player_name': 'James, LeBron',
            'team': 'LAL',
            'predicted_PTS': 21.5,
            'predicted_AST': 9.5,   # Predicción por encima del mercado
            'predicted_TRB': 7.8,
            'confidence': 0.91,
            'model_version': 'v2.1'
        }
    ]
    
    return pd.DataFrame(predictions)

def demo_sistema_completo():
    """
    Demostración completa del sistema modelo vs mercado.
    """
    print("🎯 DEMO SISTEMA COMPLETO MODELO VS MERCADO")
    print("=" * 60)
    
    # Configurar API keys
    config = get_config()
    api_keys = {
        'sportradar': config.get('sportradar', 'api_key') or os.getenv('SPORTRADAR_API')
    }
    
    if not api_keys['sportradar']:
        print("❌ ERROR: API key de Sportradar no configurada")
        return
    
    # Inicializar sistema con filtros más permisivos
    print("📊 Inicializando sistema con filtros permisivos...")
    bookmakers = BookmakersIntegration(
        api_keys=api_keys,
        minimum_edge=0.01,  # 1% de ventaja mínima (muy permisivo)
        confidence_threshold=0.80  # 80% de confianza mínima
    )
    
    # Crear predicciones realistas
    print("🤖 Creando predicciones realistas del modelo...")
    model_predictions = create_realistic_predictions()
    print(f"Predicciones creadas para {len(model_predictions)} jugadores")
    
    # Mostrar predicciones
    print("\n📋 PREDICCIONES DEL MODELO:")
    for _, pred in model_predictions.iterrows():
        print(f"  {pred['player_name']}: PTS={pred['predicted_PTS']:.1f}, "
              f"AST={pred['predicted_AST']:.1f}, TRB={pred['predicted_TRB']:.1f} "
              f"(conf: {pred['confidence']:.0%})")
    
    # Sport event ID del ejemplo que funciona
    sport_event_id = "sr:sport_event:59850122"
    
    print(f"\n🎯 ANALIZANDO EVENTO: {sport_event_id}")
    print("=" * 50)
    
    # Analizar cada target
    targets = ['PTS', 'AST', 'TRB']
    all_opportunities = []
    
    for target in targets:
        print(f"\n🔍 ANALIZANDO {target}:")
        print("-" * 30)
        
        try:
            # Comparar modelo vs mercado con filtros permisivos
            result = bookmakers.compare_model_vs_market(
                model_predictions=model_predictions,
                sport_event_id=sport_event_id,
                target=target,
                min_edge=0.005,  # 0.5% de ventaja mínima
                min_confidence=0.80  # 80% de confianza mínima
            )
            
            if result.get('success', False):
                summary = result['summary']
                opportunities = result.get('opportunities', [])
                
                print(f"✅ Jugadores analizados: {summary['total_players_analyzed']}")
                print(f"🎯 Jugadores con ventaja: {summary['players_with_edges']}")
                
                if opportunities:
                    print(f"💰 Oportunidades encontradas: {len(opportunities)}")
                    print(f"📊 Mejor ventaja: {summary['max_edge']:.2%}")
                    
                    # Mostrar mejores oportunidades
                    for i, opp in enumerate(opportunities[:2], 1):
                        print(f"\n  {i}. {opp['player']} - {opp['bet_type'].upper()} {opp['line']}")
                        print(f"     Predicción: {opp['predicted_value']:.1f}")
                        print(f"     Prob. modelo: {opp['model_probability']:.1%}")
                        print(f"     Prob. mercado: {opp['market_probability']:.1%}")
                        print(f"     Ventaja: {opp['edge_percentage']:.2f}%")
                        print(f"     Odds: {opp['odds']['decimal']:.2f}")
                        print(f"     Valor esperado: {opp['expected_value']:.4f}")
                        print(f"     Kelly fraction: {opp['kelly_fraction']:.2%}")
                        print(f"     Score: {opp['opportunity_score']:.4f}")
                    
                    # Agregar a lista global
                    all_opportunities.extend(opportunities)
                else:
                    print("⚠️  No se encontraron oportunidades")
            else:
                print(f"❌ Error: {result.get('error', 'Unknown error')}")
                
        except Exception as e:
            print(f"❌ Error procesando {target}: {e}")
            continue
    
    # Resumen final
    print(f"\n🏆 RESUMEN FINAL")
    print("=" * 40)
    
    if all_opportunities:
        # Ordenar por score
        all_opportunities.sort(key=lambda x: x['opportunity_score'], reverse=True)
        
        print(f"✅ Total de oportunidades encontradas: {len(all_opportunities)}")
        print(f"🎯 Mejor oportunidad general:")
        
        best = all_opportunities[0]
        print(f"   {best['player']} - {best['target']} {best['bet_type'].upper()} {best['line']}")
        print(f"   Ventaja: {best['edge_percentage']:.2f}%")
        print(f"   Odds: {best['odds']['decimal']:.2f}")
        print(f"   Kelly fraction: {best['kelly_fraction']:.2%}")
        print(f"   Valor esperado: {best['expected_value']:.4f}")
        
        # Análisis de distribución
        edges = [opp['edge_percentage'] for opp in all_opportunities]
        print(f"\n📊 Distribución de ventajas:")
        print(f"   Promedio: {np.mean(edges):.2f}%")
        print(f"   Máximo: {max(edges):.2f}%")
        print(f"   Mínimo: {min(edges):.2f}%")
        
    else:
        print("⚠️  No se encontraron oportunidades con los filtros actuales")
        print("💡 Sugerencias:")
        print("   - Reducir min_edge a 0.001 (0.1%)")
        print("   - Reducir min_confidence a 0.70 (70%)")
        print("   - Ajustar las predicciones del modelo")

def demo_analisis_mercado():
    """
    Demuestra el análisis del mercado sin comparación con modelo.
    """
    print("\n🔍 DEMO ANÁLISIS PURO DEL MERCADO")
    print("=" * 50)
    
    # Configurar API keys
    config = get_config()
    api_keys = {
        'sportradar': config.get('sportradar', 'api_key') or os.getenv('SPORTRADAR_API')
    }
    
    bookmakers = BookmakersIntegration(api_keys=api_keys)
    
    # Obtener datos del mercado directamente
    sport_event_id = "sr:sport_event:59850122"
    
    try:
        market_data = bookmakers.bookmakers_fetcher.sportradar_api.get_player_props(sport_event_id)
        
        if market_data.get('success', False):
            print("✅ Datos del mercado obtenidos exitosamente")
            
            # Mostrar resumen
            market_summary = market_data.get('market_summary', {})
            print(f"📊 Jugadores en el mercado: {market_summary.get('total_players', 0)}")
            print(f"🎯 Targets disponibles: {market_summary.get('targets_available', [])}")
            print(f"🏪 Casas de apuestas: {len(market_summary.get('bookmakers', []))}")
            
            # Mostrar algunos ejemplos
            players = market_data.get('players', {})
            print(f"\n📋 EJEMPLOS DEL MERCADO:")
            
            for player_name, player_data in list(players.items())[:3]:
                print(f"\n  {player_name}:")
                targets = player_data.get('targets', {})
                
                for target, target_data in targets.items():
                    lines = target_data.get('lines', [])
                    if lines:
                        line = lines[0]  # Primera línea
                        print(f"    {target}: {line['value']} "
                              f"(Over: {line['over']['best_odds']['decimal']:.2f}, "
                              f"Under: {line['under']['best_odds']['decimal']:.2f})")
        else:
            print(f"❌ Error obteniendo datos del mercado: {market_data.get('error', 'Unknown error')}")
            
    except Exception as e:
        print(f"❌ Error en análisis del mercado: {e}")

def main():
    """
    Función principal que ejecuta la demostración completa.
    """
    print("🚀 DEMOSTRACIÓN COMPLETA DEL SISTEMA")
    print("Comparación Modelo vs Mercado NBA")
    print("=" * 60)
    
    # Verificar configuración
    config = get_config()
    api_key = config.get('sportradar', 'api_key') or os.getenv('SPORTRADAR_API')
    
    if not api_key:
        print("❌ ERROR: API key de Sportradar no configurada")
        print("Configura la variable de entorno SPORTRADAR_API")
        return
    
    print(f"✅ API Key configurada: {'*' * (len(api_key) - 4)}{api_key[-4:]}")
    
    try:
        # Ejecutar demostraciones
        demo_sistema_completo()
        demo_analisis_mercado()
        
        print("\n🎉 DEMOSTRACIÓN COMPLETADA")
        print("=" * 60)
        
        print("\n💡 PRÓXIMOS PASOS:")
        print("1. Integrar con tus modelos reales de predicción")
        print("2. Ajustar filtros según tu estrategia de riesgo")
        print("3. Implementar sistema de alertas automáticas")
        print("4. Crear dashboard para monitoreo en tiempo real")
        print("5. Backtesting con datos históricos")
        
    except Exception as e:
        print(f"❌ Error en demostración: {e}")
        logger.exception("Error detallado:")

if __name__ == "__main__":
    main() 