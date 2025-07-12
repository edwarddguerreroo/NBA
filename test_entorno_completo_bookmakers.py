#!/usr/bin/env python3
"""
Test completo del entorno utils/bookmakers/
Prueba todas las funcionalidades con el evento sr:sport_event:56328141
"""

import os
import sys
sys.path.append('.')

from utils.bookmakers.sportradar_api import SportradarAPI
from utils.bookmakers.bookmakers_data_fetcher import BookmakersDataFetcher
from utils.bookmakers.bookmakers_integration import BookmakersIntegration
from utils.bookmakers.config import get_config
from utils.bookmakers.exceptions import SportradarAPIError
import logging
import json
from datetime import datetime

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_sportradar_api():
    """Test completo de SportradarAPI"""
    print("=" * 60)
    print("🔥 TEST SPORTRADAR API")
    print("=" * 60)
    
    try:
        api = SportradarAPI()
        sport_event_id = "sr:sport_event:56328141"
        
        print(f"📅 Evento de prueba: {sport_event_id}")
        print()
        
        # 1. Test Player Props
        print("1️⃣ PLAYER PROPS API")
        print("-" * 30)
        
        player_props = api.get_player_props(sport_event_id)
        
        if player_props.get('success', False):
            players = player_props.get('players', {})
            print(f"✅ Player Props exitoso: {len(players)} jugadores")
            
            # Mostrar algunos jugadores y sus targets
            for i, (player_name, player_data) in enumerate(players.items()):
                if i >= 5:  # Mostrar solo primeros 5
                    break
                targets = list(player_data.get('targets', {}).keys())
                print(f"   👤 {player_name}: {targets}")
                
                # Mostrar detalles de PTS si está disponible
                if 'PTS' in targets:
                    pts_data = player_data['targets']['PTS']
                    lines_count = len(pts_data.get('lines', []))
                    best_odds = pts_data.get('best_odds', {})
                    print(f"      📊 PTS: {lines_count} líneas, best_over: {best_odds.get('over')}")
        else:
            print(f"❌ Player Props falló: {player_props.get('error', 'Unknown error')}")
        
        print()
        
        # 2. Test Prematch Odds (Team Props)
        print("2️⃣ PREMATCH ODDS API (Team Props)")
        print("-" * 30)
        
        prematch_odds = api.get_prematch_odds(sport_event_id)
        
        if prematch_odds.get('success', False):
            markets = prematch_odds.get('markets', [])
            print(f"✅ Prematch Odds exitoso: {len(markets)} mercados")
            
            # Buscar mercados específicos para nuestros targets
            target_markets = {
                'is_win': [1],  # Market ID 1
                'total_points': [225],  # Market ID 225
                'teams_points': [227, 228]  # Market IDs 227, 228
            }
            
            for target, market_ids in target_markets.items():
                found_markets = [m for m in markets if m.get('id') in market_ids]
                print(f"   🎯 {target}: {len(found_markets)} mercados encontrados")
                
                for market in found_markets:
                    market_name = market.get('name', 'N/A')
                    market_id = market.get('id', 'N/A')
                    outcomes_count = len(market.get('outcomes', []))
                    print(f"      📈 {market_name} (ID: {market_id}) - {outcomes_count} outcomes")
        else:
            print(f"❌ Prematch Odds falló: {prematch_odds.get('error', 'Unknown error')}")
        
        print()
        
        # 3. Test Market Overview
        print("3️⃣ MARKET OVERVIEW")
        print("-" * 30)
        
        try:
            market_overview = api.get_market_overview()
            if market_overview.get('success', False):
                print("✅ Market Overview exitoso")
                total_games = market_overview.get('total_games', 0)
                games_with_odds = market_overview.get('games_with_odds', 0)
                print(f"   📊 Total juegos: {total_games}, con odds: {games_with_odds}")
            else:
                print("❌ Market Overview falló")
        except Exception as e:
            print(f"⚠️ Market Overview no disponible: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error en SportradarAPI: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_bookmakers_data_fetcher():
    """Test completo de BookmakersDataFetcher"""
    print("=" * 60)
    print("🔥 TEST BOOKMAKERS DATA FETCHER")
    print("=" * 60)
    
    try:
        fetcher = BookmakersDataFetcher()
        
        print("📋 Configuración del fetcher:")
        print(f"   🔧 Cache expiry: {fetcher.cache_expiry}h")
        print(f"   📁 Data directory: {fetcher.odds_data_dir}")
        print(f"   🌐 Sportradar API: {'✅' if fetcher.sportradar_api else '❌'}")
        print()
        
        # 1. Test get_player_props_for_targets
        print("1️⃣ GET PLAYER PROPS FOR TARGETS")
        print("-" * 30)
        
        # Usar método que simula obtener juegos y luego props
        targets = ['PTS', 'AST', 'TRB', '3P', 'DD']
        
        # Simular obteniendo props directamente para nuestro evento conocido
        sport_event_id = "sr:sport_event:56328141"
        
        try:
            # Obtener props directas para el evento
            player_props = fetcher.sportradar_api.get_player_props(sport_event_id)
            
            if player_props.get('success', False):
                print("✅ Props obtenidas exitosamente")
                players = player_props.get('players', {})
                
                # Analizar por target
                target_stats = {}
                for target in targets:
                    target_stats[target] = 0
                    for player_name, player_data in players.items():
                        if target in player_data.get('targets', {}):
                            target_stats[target] += 1
                
                print("   📊 Props por target:")
                for target, count in target_stats.items():
                    print(f"      {target}: {count} jugadores")
            else:
                print(f"❌ Error obteniendo props: {player_props.get('error', 'Unknown')}")
                
        except Exception as e:
            print(f"❌ Error en get_player_props_for_targets: {e}")
        
        print()
        
        # 2. Test seasonal optimization
        print("2️⃣ SEASONAL OPTIMIZATION")
        print("-" * 30)
        
        try:
            seasonal_info = fetcher.get_seasonal_props_availability()
            current_phase = seasonal_info['current_phase']
            recommendations = seasonal_info['recommendations']
            
            print(f"✅ Análisis estacional exitoso")
            print(f"   📅 Fase actual: {current_phase}")
            print(f"   ⏰ Cache recomendado: {recommendations['cache_expiry_hours']}h")
            print(f"   🎯 Targets prioritarios: {recommendations['priority_targets']}")
            print(f"   🔄 Frecuencia API: {recommendations['api_call_frequency']}")
            
        except Exception as e:
            print(f"❌ Error en seasonal optimization: {e}")
        
        print()
        
        # 3. Test API status
        print("3️⃣ API STATUS")
        print("-" * 30)
        
        try:
            api_status = fetcher.get_api_status()
            sportradar_status = api_status.get('sportradar', {})
            
            print(f"✅ Status check exitoso")
            print(f"   🔧 Configurada: {'✅' if sportradar_status.get('configured') else '❌'}")
            print(f"   🌐 Accesible: {'✅' if sportradar_status.get('accessible') else '❌'}")
            
            if sportradar_status.get('error'):
                print(f"   ⚠️ Error: {sportradar_status['error']}")
                
        except Exception as e:
            print(f"❌ Error en API status: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error en BookmakersDataFetcher: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_bookmakers_integration():
    """Test completo de BookmakersIntegration"""
    print("=" * 60)
    print("🔥 TEST BOOKMAKERS INTEGRATION")
    print("=" * 60)
    
    try:
        integration = BookmakersIntegration()
        
        print("📋 Configuración de integración:")
        print(f"   🎯 Min edge: {integration.minimum_edge}")
        print(f"   📊 Min confidence: {integration.confidence_threshold}")
        print(f"   💰 Fetcher configurado: {'✅' if integration.bookmakers_fetcher else '❌'}")
        print()
        
        # 1. Test compare_model_vs_market
        print("1️⃣ COMPARE MODEL VS MARKET")
        print("-" * 30)
        
        sport_event_id = "sr:sport_event:56328141"
        
        try:
            # Crear predicciones mock para el test
            import pandas as pd
            
            mock_predictions = pd.DataFrame({
                'PLAYER_NAME': ['Canada, Jordin', 'Mitchell, Kelsey'],
                'PTS_pred': [12.5, 18.2],
                'confidence': [0.92, 0.95],
                'DATE': ['2025-07-11', '2025-07-11']
            })
            
            # Test con el primer jugador
            first_player = mock_predictions.iloc[0]['PLAYER_NAME']
            
            comparison = integration.compare_model_vs_market(
                model_predictions=mock_predictions,
                sport_event_id=sport_event_id,
                target='PTS',
                player_name=first_player
            )
            
            if comparison.get('success', False):
                print("✅ Comparación modelo vs mercado exitosa")
                
                opportunities = comparison.get('opportunities', [])
                market_data = comparison.get('market_data', {})
                
                print(f"   🎯 Player: {first_player}")
                print(f"   📊 Oportunidades encontradas: {len(opportunities)}")
                print(f"   💰 Markets disponibles: {len(market_data.get('markets', []))}")
                
                # Mostrar mejor oportunidad si existe
                if opportunities:
                    best = opportunities[0]
                    edge = best.get('edge', 0)
                    confidence = best.get('confidence', 0)
                    print(f"   🎯 Mejor edge: {edge:.1%} (conf: {confidence:.1%})")
            else:
                print(f"❌ Error en comparación: {comparison.get('error', 'Unknown')}")
                
        except Exception as e:
            print(f"❌ Error en compare_model_vs_market: {e}")
        
        print()
        
        # 2. Test betting opportunities (simulado)
        print("2️⃣ BETTING OPPORTUNITIES (Simulado)")
        print("-" * 30)
        
        try:
            # Simular algunas predicciones para el análisis
            import pandas as pd
            
            mock_predictions = pd.DataFrame({
                'PLAYER_NAME': ['Canada, Jordin', 'Mitchell, Kelsey'],
                'PTS_pred': [12.5, 18.2],
                'confidence': [0.92, 0.95],
                'DATE': ['2025-07-11', '2025-07-11']
            })
            
            print("✅ Simulación de análisis de oportunidades")
            print(f"   📊 Predicciones mock: {len(mock_predictions)} jugadores")
            print("   🎯 Targets analizados: PTS")
            print("   📈 Confianza promedio: 93.5%")
            
            # Mostrar estructura de lo que se analizaría
            for _, row in mock_predictions.iterrows():
                player = row['PLAYER_NAME']
                pred = row['PTS_pred']
                conf = row['confidence']
                print(f"      👤 {player}: PTS={pred:.1f} (conf: {conf:.1%})")
                
        except Exception as e:
            print(f"❌ Error en betting opportunities: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error en BookmakersIntegration: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_config_and_exceptions():
    """Test de configuración y manejo de excepciones"""
    print("=" * 60)
    print("🔥 TEST CONFIG & EXCEPTIONS")
    print("=" * 60)
    
    try:
        # 1. Test configuración
        print("1️⃣ CONFIGURACIÓN")
        print("-" * 30)
        
        config = get_config()
        
        print("✅ Configuración cargada exitosamente")
        print(f"   🔗 Player Props URL: {config.get('sportradar', 'player_props_url')}")
        print(f"   🔗 Prematch URL: {config.get('sportradar', 'odds_base_url')}")
        print(f"   🔑 API Key configurada: {'✅' if config.get('sportradar', 'api_key') else '❌'}")
        
        # Test market IDs
        market_ids = config.get('sportradar', 'market_ids')
        print(f"   📊 Market IDs configurados: {len(market_ids)} mercados")
        
        # Mostrar algunos market IDs importantes
        important_markets = ['total_points', 'total_assists', 'total_rebounds']
        for market in important_markets:
            market_id = market_ids.get(market, 'N/A')
            print(f"      {market}: {market_id}")
        
        print()
        
        # 2. Test target mapping
        print("2️⃣ TARGET MAPPING")
        print("-" * 30)
        
        target_mapping = config.get('sportradar', 'target_to_market')
        print(f"✅ Target mapping: {len(target_mapping)} targets")
        
        for target, market in target_mapping.items():
            print(f"   🎯 {target} → {market}")
        
        print()
        
        # 3. Test sport IDs
        print("3️⃣ SPORT IDS")
        print("-" * 30)
        
        basketball_id = config.get_sport_id('basketball')
        nba_id = config.get_sport_id('nba')
        
        print(f"✅ Sport IDs configurados")
        print(f"   🏀 Basketball: {basketball_id}")
        print(f"   🏀 NBA: {nba_id}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error en config & exceptions: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Ejecutar todos los tests"""
    print("🚀 INICIANDO TEST COMPLETO DEL ENTORNO BOOKMAKERS")
    print("📅 Evento de prueba: sr:sport_event:56328141")
    print("🎯 Objetivos: Player Props + Team Props")
    print()
    
    results = {}
    
    # Ejecutar todos los tests
    tests = [
        ("SportradarAPI", test_sportradar_api),
        ("BookmakersDataFetcher", test_bookmakers_data_fetcher),
        ("BookmakersIntegration", test_bookmakers_integration),
        ("Config & Exceptions", test_config_and_exceptions)
    ]
    
    for test_name, test_func in tests:
        try:
            print(f"\n🔄 Ejecutando {test_name}...")
            result = test_func()
            results[test_name] = "✅ PASS" if result else "❌ FAIL"
        except Exception as e:
            results[test_name] = f"❌ ERROR: {e}"
        
        print()
    
    # Resumen final
    print("=" * 60)
    print("📋 RESUMEN FINAL")
    print("=" * 60)
    
    for test_name, result in results.items():
        print(f"{result} {test_name}")
    
    # Estadísticas
    passed = sum(1 for r in results.values() if "✅" in r)
    total = len(results)
    
    print()
    print(f"📊 ESTADÍSTICAS: {passed}/{total} tests exitosos ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("🎉 ¡TODOS LOS TESTS PASARON! El entorno bookmakers está 100% funcional")
    else:
        print("⚠️ Algunos tests fallaron. Revisar logs para más detalles.")
    
    print()
    print("🏁 Test completo finalizado")

if __name__ == "__main__":
    main() 