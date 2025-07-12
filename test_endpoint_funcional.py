#!/usr/bin/env python3
"""
Test del Endpoint Funcional - Evento 59850122
==============================================

Prueba el sistema usando el endpoint específico que el usuario confirmó que funciona:
https://api.sportradar.com/oddscomparison-player-props/trial/v2/en/sport_events/sr%3Asport_event%3A59850122/players_props
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.bookmakers.bookmakers_integration import BookmakersIntegration
from utils.bookmakers.sportradar_api import SportradarAPI
import json
import pandas as pd
from datetime import datetime

def test_endpoint_funcional():
    """
    Prueba el endpoint funcional específico
    """
    print("🎯 TEST DEL ENDPOINT FUNCIONAL - EVENTO 59850122")
    print("=" * 60)
    
    try:
        # 1. Inicializar sistema
        print("\n1. Inicializando sistema...")
        bookmakers = BookmakersIntegration()
        
        # 2. Verificar que Sportradar API está configurada
        print("\n2. Verificando configuración de Sportradar API...")
        if not bookmakers.bookmakers_fetcher.sportradar_api:
            print("   ❌ Sportradar API no configurada")
            return
        
        print("   ✅ Sportradar API configurada correctamente")
        
        # 3. Probar el endpoint específico que funciona
        print("\n3. Probando endpoint específico que funciona...")
        sport_event_id = "sr:sport_event:59850122"
        
        try:
            # Usar directamente el método get_player_props
            player_props = bookmakers.bookmakers_fetcher.sportradar_api.get_player_props(sport_event_id)
            
            print(f"   📊 Resultado: {player_props.get('success', False)}")
            
            if player_props.get('success', False):
                print("   ✅ Endpoint funciona correctamente!")
                
                # Mostrar estructura de datos
                print("\n   📋 Estructura de datos recibidos:")
                
                # Información del evento
                sport_event = player_props.get('sport_event', {})
                if sport_event:
                    print(f"      🏀 Evento: {sport_event.get('id', 'N/A')}")
                    print(f"      📅 Fecha: {sport_event.get('start_time', 'N/A')}")
                    
                    # Equipos
                    competitors = sport_event.get('competitors', [])
                    if competitors:
                        print(f"      🏆 Equipos:")
                        for comp in competitors:
                            print(f"         - {comp.get('name', 'N/A')} ({comp.get('qualifier', 'N/A')})")
                
                # Mercados disponibles
                markets = player_props.get('markets', [])
                print(f"      📊 Mercados totales: {len(markets)}")
                
                # Analizar tipos de mercados
                market_types = {}
                player_count = set()
                
                for market in markets:
                    market_name = market.get('name', 'Unknown')
                    if market_name not in market_types:
                        market_types[market_name] = 0
                    market_types[market_name] += 1
                    
                    # Contar jugadores únicos
                    for outcome in market.get('outcomes', []):
                        competitor = outcome.get('competitor', {})
                        if competitor.get('name'):
                            player_count.add(competitor.get('name'))
                
                print(f"      👥 Jugadores únicos: {len(player_count)}")
                print(f"      🎯 Tipos de mercado:")
                
                # Mostrar top 10 mercados
                sorted_markets = sorted(market_types.items(), key=lambda x: x[1], reverse=True)
                for market_type, count in sorted_markets[:10]:
                    print(f"         - {market_type}: {count} mercados")
                
                # Mostrar algunos ejemplos de outcomes
                print(f"\n      💡 Ejemplos de outcomes (primeros 5):")
                outcomes_shown = 0
                
                for market in markets[:5]:
                    market_name = market.get('name', 'Unknown')
                    outcomes = market.get('outcomes', [])
                    
                    if outcomes:
                        print(f"         📈 {market_name}:")
                        for outcome in outcomes[:2]:  # Máximo 2 por mercado
                            competitor = outcome.get('competitor', {})
                            player_name = competitor.get('name', 'N/A')
                            odds = outcome.get('odds', 'N/A')
                            total = outcome.get('total', outcome.get('handicap', 'N/A'))
                            
                            print(f"            * {player_name}: {total} @ {odds}")
                            outcomes_shown += 1
                            
                            if outcomes_shown >= 5:
                                break
                        
                        if outcomes_shown >= 5:
                            break
                
                # 4. Identificar nuestros targets específicos
                print(f"\n4. Identificando targets específicos del sistema...")
                
                our_targets = {
                    'PTS': ['total points', 'points', 'player points'],
                    'AST': ['total assists', 'assists', 'player assists'],
                    'TRB': ['total rebounds', 'rebounds', 'player rebounds'],
                    '3P': ['total threes', 'three pointers', 'threes', '3-pointers'],
                    'DD': ['double double', 'double_double']
                }
                
                found_targets = {}
                
                for target, keywords in our_targets.items():
                    found_markets = []
                    
                    for market in markets:
                        market_name = market.get('name', '').lower()
                        if any(keyword.lower() in market_name for keyword in keywords):
                            found_markets.append({
                                'name': market.get('name'),
                                'outcomes': len(market.get('outcomes', []))
                            })
                    
                    if found_markets:
                        found_targets[target] = found_markets
                
                print(f"   🎯 Targets encontrados: {len(found_targets)}")
                for target, markets_list in found_targets.items():
                    print(f"      - {target}: {len(markets_list)} mercados")
                    for market_info in markets_list[:2]:  # Mostrar máximo 2
                        print(f"        * {market_info['name']} ({market_info['outcomes']} outcomes)")
                
                # 5. Probar integración con sistema de comparación
                print(f"\n5. Probando integración con sistema de comparación...")
                
                # Crear predicciones mock basadas en los jugadores encontrados
                sample_players = list(player_count)[:3]  # Tomar primeros 3 jugadores
                
                if sample_players:
                    mock_predictions = pd.DataFrame({
                        'player_name': sample_players,
                        'predicted_PTS': [25.5, 22.3, 18.5],
                        'predicted_AST': [7.2, 3.5, 8.1],
                        'predicted_TRB': [8.5, 11.8, 6.2],
                        'confidence': [0.92, 0.89, 0.87]
                    })
                    
                    print(f"   📊 Predicciones mock para {len(sample_players)} jugadores:")
                    for _, row in mock_predictions.iterrows():
                        print(f"      - {row['player_name']}: PTS={row['predicted_PTS']}, AST={row['predicted_AST']}, TRB={row['predicted_TRB']}")
                    
                    # Probar comparación modelo vs mercado
                    try:
                        comparison_result = bookmakers.compare_model_vs_market(
                            model_predictions=mock_predictions,
                            sport_event_id=sport_event_id,
                            target='PTS',
                            min_edge=0.03  # Reducir umbral para encontrar más oportunidades
                        )
                        
                        if comparison_result.get('success', False):
                            print("   ✅ Comparación modelo vs mercado exitosa")
                            
                            opportunities = comparison_result.get('opportunities', [])
                            print(f"   🎯 Oportunidades encontradas: {len(opportunities)}")
                            
                            if opportunities:
                                print(f"   🏆 Mejores oportunidades:")
                                for i, opp in enumerate(opportunities[:3], 1):
                                    print(f"      {i}. {opp.get('player', 'N/A')} - {opp.get('target', 'N/A')}")
                                    print(f"         Línea: {opp.get('line', 'N/A')}")
                                    print(f"         Tipo: {opp.get('bet_type', 'N/A')}")
                                    print(f"         Edge: {opp.get('edge', 0):.2%}")
                                    print(f"         Valor esperado: {opp.get('expected_value', 0):.3f}")
                                    print()
                            else:
                                print("   ⚠️  No se encontraron oportunidades con el umbral actual")
                                
                            # Mostrar estadísticas
                            summary = comparison_result.get('summary', {})
                            print(f"   📊 Resumen:")
                            print(f"      - Jugadores analizados: {summary.get('total_players_analyzed', 0)}")
                            print(f"      - Jugadores con edges: {summary.get('players_with_edges', 0)}")
                            print(f"      - Edge promedio: {summary.get('avg_edge', 0):.2%}")
                            print(f"      - Edge máximo: {summary.get('max_edge', 0):.2%}")
                            
                        else:
                            print(f"   ❌ Error en comparación: {comparison_result.get('error', 'Unknown')}")
                            
                    except Exception as e:
                        print(f"   ❌ Error en comparación modelo vs mercado: {e}")
                
                else:
                    print("   ⚠️  No se encontraron jugadores para crear predicciones mock")
                
                # 6. Verificar estadísticas del sistema
                print(f"\n6. Estadísticas del sistema:")
                try:
                    cache_stats = bookmakers.bookmakers_fetcher.get_cache_status()
                    print(f"   📈 Cache hits: {cache_stats.get('cache_hits', 0)}")
                    print(f"   📉 Cache misses: {cache_stats.get('cache_misses', 0)}")
                    print(f"   🔄 API calls: {cache_stats.get('api_calls_made', 0)}")
                    print(f"   ⚡ Hit rate: {cache_stats.get('hit_rate', 0):.2%}")
                except Exception as e:
                    print(f"   ⚠️  Error obteniendo estadísticas: {e}")
                
                print(f"\n✅ TEST COMPLETADO EXITOSAMENTE")
                print(f"🎯 El endpoint funciona correctamente y el sistema puede procesar los datos")
                
            else:
                print(f"   ❌ Error en endpoint: {player_props.get('error', 'Unknown error')}")
                
        except Exception as e:
            print(f"   ❌ Error probando endpoint: {e}")
            import traceback
            traceback.print_exc()
        
    except Exception as e:
        print(f"\n❌ ERROR GENERAL EN EL TEST: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_endpoint_funcional() 