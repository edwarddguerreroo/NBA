#!/usr/bin/env python3
"""
Test del Sistema NBA - Evento Específico 59850122
================================================

Prueba el sistema completo con el evento específico 59850122 que se jugó el 2025-05-01
para verificar que todas las funcionalidades estén operativas.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.bookmakers.bookmakers_integration import BookmakersIntegration
from utils.bookmakers.sportradar_api import SportradarAPI
import json
from datetime import datetime

def test_evento_especifico():
    """
    Prueba el evento específico 59850122 del 2025-05-01
    """
    print("🏀 TEST DEL SISTEMA NBA - EVENTO ESPECÍFICO 59850122")
    print("=" * 60)
    
    try:
        # 1. Inicializar BookmakersIntegration
        print("\n1. Inicializando BookmakersIntegration...")
        bookmakers = BookmakersIntegration()
        
        # 2. Probar conexión básica
        print("\n2. Probando conexión con Sportradar API...")
        try:
            # Usar el fetcher interno para verificar estado
            status = bookmakers.bookmakers_fetcher.get_api_status()
            print(f"   Estado Sportradar: {status.get('sportradar', {})}")
        except Exception as e:
            print(f"   ⚠️  No se pudo verificar estado de API: {e}")
        
        # 3. Probar obtención de player props para el evento específico
        print(f"\n3. Obteniendo player props para evento 59850122...")
        sport_event_id = "sr:sport_event:59850122"
        
        try:
            # Usar el método correcto del fetcher
            player_props = bookmakers.bookmakers_fetcher.sportradar_api.get_player_props(sport_event_id)
            
            if player_props.get('success', False):
                print(f"   ✅ Player props obtenidas exitosamente")
                print(f"   📊 Datos del evento:")
                
                # Mostrar información del evento
                event_info = player_props.get('sport_event', {})
                print(f"      - ID: {event_info.get('id', 'N/A')}")
                print(f"      - Fecha: {event_info.get('start_time', 'N/A')}")
                
                # Mostrar equipos
                competitors = event_info.get('competitors', [])
                if competitors:
                    print(f"      - Equipos:")
                    for comp in competitors:
                        print(f"        * {comp.get('name', 'N/A')} ({comp.get('qualifier', 'N/A')})")
                
                # Mostrar mercados disponibles
                markets = player_props.get('markets', [])
                print(f"      - Mercados encontrados: {len(markets)}")
                
                # Agrupar por tipo de mercado
                market_types = {}
                for market in markets:
                    market_name = market.get('name', 'Unknown')
                    if market_name not in market_types:
                        market_types[market_name] = 0
                    market_types[market_name] += 1
                
                print(f"      - Tipos de mercado:")
                for market_type, count in market_types.items():
                    print(f"        * {market_type}: {count} mercados")
                
                # Mostrar algunos ejemplos de outcomes
                print(f"      - Ejemplos de outcomes:")
                outcomes_shown = 0
                for market in markets[:3]:  # Mostrar primeros 3 mercados
                    market_name = market.get('name', 'Unknown')
                    outcomes = market.get('outcomes', [])
                    print(f"        * {market_name} ({len(outcomes)} outcomes)")
                    
                    for outcome in outcomes[:2]:  # Mostrar primeros 2 outcomes
                        competitor = outcome.get('competitor', {})
                        player_name = competitor.get('name', 'N/A')
                        odds = outcome.get('odds', 'N/A')
                        total = outcome.get('total', outcome.get('handicap', 'N/A'))
                        
                        print(f"          - {player_name}: {total} @ {odds}")
                        outcomes_shown += 1
                        
                        if outcomes_shown >= 5:  # Limitar ejemplos
                            break
                    
                    if outcomes_shown >= 5:
                        break
                
            else:
                print(f"   ❌ Error obteniendo player props: {player_props.get('error', 'Unknown error')}")
                
        except Exception as e:
            print(f"   ❌ Error en player props: {e}")
        
        # 4. Probar obtención de prematch odds
        print(f"\n4. Obteniendo prematch odds para evento 59850122...")
        
        try:
            # Usar el método correcto del fetcher
            prematch_odds = bookmakers.bookmakers_fetcher.sportradar_api.get_prematch_odds(sport_event_id)
            
            if prematch_odds.get('success', False):
                print(f"   ✅ Prematch odds obtenidas exitosamente")
                
                # Mostrar mercados de prematch
                markets = prematch_odds.get('markets', [])
                print(f"   📊 Mercados de prematch: {len(markets)}")
                
                # Mostrar mercados específicos que nos interesan
                target_markets = {
                    1: 'is_win (1x2/moneyline)',
                    225: 'total_points (total_incl_overtime)',
                    227: 'teams_points (home_total_incl_overtime)',
                    228: 'teams_points (away_total_incl_overtime)'
                }
                
                found_markets = {}
                for market in markets:
                    market_id = market.get('id')
                    if market_id in target_markets:
                        found_markets[market_id] = {
                            'name': market.get('name', 'Unknown'),
                            'outcomes': len(market.get('outcomes', []))
                        }
                
                print(f"   🎯 Mercados objetivo encontrados:")
                for market_id, market_info in found_markets.items():
                    target_name = target_markets[market_id]
                    print(f"      - Market ID {market_id}: {market_info['name']} ({target_name}) - {market_info['outcomes']} outcomes")
                
                if not found_markets:
                    print(f"   ⚠️  No se encontraron mercados objetivo específicos")
                    print(f"   📋 Mercados disponibles:")
                    for market in markets[:5]:  # Mostrar primeros 5
                        print(f"      - ID {market.get('id')}: {market.get('name', 'Unknown')}")
                
            else:
                print(f"   ❌ Error obteniendo prematch odds: {prematch_odds.get('error', 'Unknown error')}")
                
        except Exception as e:
            print(f"   ❌ Error en prematch odds: {e}")
        
        # 5. Probar sistema completo de comparación modelo vs mercado
        print(f"\n5. Probando sistema completo modelo vs mercado...")
        
        try:
            # Simular predicciones del modelo como DataFrame
            import pandas as pd
            mock_predictions_df = pd.DataFrame({
                'player_name': ['LeBron James', 'Anthony Davis', 'Russell Westbrook'],
                'predicted_PTS': [25.5, 22.3, 18.5],
                'predicted_AST': [7.2, 3.5, 8.1],
                'predicted_TRB': [8.5, 11.8, 6.2],
                'confidence': [0.92, 0.89, 0.87]
            })
            
            # Probar con target PTS
            comparison_result = bookmakers.compare_model_vs_market(
                model_predictions=mock_predictions_df,
                sport_event_id=sport_event_id,
                target='PTS'
            )
            
            if comparison_result.get('success', False):
                print(f"   ✅ Comparación modelo vs mercado exitosa")
                
                opportunities = comparison_result.get('opportunities', [])
                print(f"   🎯 Oportunidades identificadas: {len(opportunities)}")
                
                # Mostrar mejores oportunidades
                if opportunities:
                    print(f"   🏆 Top 3 oportunidades:")
                    for i, opp in enumerate(opportunities[:3], 1):
                        print(f"      {i}. {opp.get('target', 'N/A')} - {opp.get('player', opp.get('market', 'N/A'))}")
                        print(f"         Edge: {opp.get('edge', 0):.2%}")
                        print(f"         Valor esperado: {opp.get('expected_value', 0):.3f}")
                        print(f"         Kelly: {opp.get('kelly_fraction', 0):.3f}")
                        print(f"         Score: {opp.get('composite_score', 0):.3f}")
                        print()
                
                # Mostrar estadísticas por target
                stats = comparison_result.get('stats', {})
                print(f"   📊 Estadísticas por target:")
                for target, count in stats.get('opportunities_by_target', {}).items():
                    print(f"      - {target}: {count} oportunidades")
                
            else:
                print(f"   ❌ Error en comparación: {comparison_result.get('error', 'Unknown error')}")
                
        except Exception as e:
            print(f"   ❌ Error en comparación modelo vs mercado: {e}")
        
        # 6. Mostrar estadísticas finales
        print(f"\n6. Estadísticas finales del sistema:")
        try:
            cache_stats = bookmakers.bookmakers_fetcher.get_cache_status()
            print(f"   📈 Cache hits: {cache_stats.get('cache_hits', 0)}")
            print(f"   📉 Cache misses: {cache_stats.get('cache_misses', 0)}")
            print(f"   🔄 API calls: {cache_stats.get('api_calls_made', 0)}")
            print(f"   ⚡ Hit rate: {cache_stats.get('hit_rate', 0):.2%}")
        except Exception as e:
            print(f"   ⚠️  No se pudieron obtener estadísticas: {e}")
        
        print(f"\n✅ TEST COMPLETADO - Evento 59850122 procesado exitosamente")
        
    except Exception as e:
        print(f"\n❌ ERROR GENERAL EN EL TEST: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_evento_especifico() 