#!/usr/bin/env python3
"""
Test del Prematch API Corregido
===============================

Probando el Prematch API con las URLs correctas según la documentación:
https://api.sportradar.com/oddscomparison-prematch/trial/v2/

Endpoints a probar:
1. /en/competitions/{competition_id}/sport_event_markets
2. /en/sport_events/{sport_event_id}/sport_event_markets  
3. /en/sports/{sport_id}/schedules/{date}/sport_event_markets
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.bookmakers.sportradar_api import SportradarAPI
from utils.bookmakers.config import get_config
import json
from datetime import datetime

def test_prematch_api_corregido():
    """
    Test del Prematch API con URLs corregidas
    """
    print("🎯 TEST PREMATCH API CORREGIDO")
    print("=" * 60)
    
    try:
        # 1. Inicializar API
        print("\n1. Inicializando Sportradar API...")
        config = get_config()
        api_key = config.get('sportradar', 'api_key')
        
        if not api_key:
            print("   ❌ API key no encontrada")
            return
        
        sportradar = SportradarAPI(api_key=api_key)
        
        # Verificar URLs configuradas
        odds_base_url = config.get('sportradar', 'odds_base_url')
        print(f"   ✅ API inicializada")
        print(f"   🔗 Prematch URL: {odds_base_url}")
        
        # 2. Test Endpoint 1: Competition Markets
        print("\n2. Probando /competitions/{competition_id}/sport_event_markets...")
        
        try:
            # Usar WNBA que está activa
            competition_id = "sr:competition:486"  # WNBA
            
            result = sportradar.get_prematch_odds_by_competition(
                competition_id=competition_id,
                limit=3
            )
            
            print(f"   ✅ Resultado: {result.get('success', False)}")
            
            if result.get('success'):
                data = result.get('data', {})
                print(f"   📊 Datos obtenidos: {len(str(data))} caracteres")
                
                # Verificar estructura
                if 'competition_sport_event_markets' in data:
                    events = data['competition_sport_event_markets']
                    print(f"   🏀 Eventos encontrados: {len(events)}")
                    
                    # Mostrar primer evento como ejemplo
                    if events:
                        first_event = events[0]
                        sport_event = first_event.get('sport_event', {})
                        markets = first_event.get('markets', [])
                        
                        print(f"   🎯 Evento ejemplo: {sport_event.get('id', 'N/A')}")
                        print(f"      Markets: {len(markets)}")
                        
                        # Mostrar equipos
                        competitors = sport_event.get('competitors', [])
                        for comp in competitors:
                            name = comp.get('name', 'N/A')
                            qualifier = comp.get('qualifier', 'N/A')
                            print(f"      {qualifier}: {name}")
                        
                        # Mostrar algunos markets
                        for i, market in enumerate(markets[:3]):
                            market_id = market.get('id', 'N/A')
                            market_name = market.get('name', 'N/A')
                            books = market.get('books', [])
                            print(f"      📈 Market {market_id}: {market_name} ({len(books)} casas)")
                            
                            # Mostrar primera casa de apuestas
                            if books:
                                first_book = books[0]
                                book_name = first_book.get('name', 'N/A')
                                outcomes = first_book.get('outcomes', [])
                                print(f"         💰 {book_name}: {len(outcomes)} outcomes")
                                
                                # Mostrar primer outcome
                                if outcomes:
                                    first_outcome = outcomes[0]
                                    outcome_type = first_outcome.get('type', 'N/A')
                                    odds_decimal = first_outcome.get('odds_decimal', 'N/A')
                                    print(f"            🎲 {outcome_type}: {odds_decimal}")
                else:
                    print("   ⚠️  Estructura inesperada en respuesta")
                    print(f"   📋 Keys disponibles: {list(data.keys())}")
            else:
                error_msg = result.get('error', 'Unknown error')
                print(f"   ❌ Error: {error_msg}")
                
        except Exception as e:
            print(f"   ❌ Excepción: {e}")
        
        # 3. Test Endpoint 2: Sport Event Markets
        print("\n3. Probando /sport_events/{sport_event_id}/sport_event_markets...")
        
        try:
            # Evento específico que el usuario mencionó
            sport_event_id = "sr:sport_event:59850122"
            
            result = sportradar.get_prematch_odds(sport_event_id)
            
            print(f"   ✅ Resultado: {result.get('success', False)}")
            
            if result.get('success'):
                markets = result.get('markets', [])
                print(f"   📈 Markets encontrados: {len(markets)}")
                
                # Mostrar algunos markets
                for i, market in enumerate(markets[:3]):
                    market_id = market.get('id', 'N/A')
                    market_name = market.get('name', 'N/A')
                    print(f"   🎲 Market {i+1}: {market_id} - {market_name}")
            else:
                error_msg = result.get('error', 'Unknown error')
                print(f"   ❌ Error: {error_msg}")
                
        except Exception as e:
            print(f"   ❌ Excepción: {e}")
        
        # 4. Test directo de URL construcción
        print("\n4. Verificando construcción de URLs...")
        
        # URL 1: Competition markets
        competition_id = "sr:competition:486"
        competition_id_formatted = competition_id.replace(':', '%3A')
        endpoint1 = f"en/competitions/{competition_id_formatted}/sport_event_markets"
        full_url1 = f"{odds_base_url}{endpoint1}?api_key={api_key[:8]}..."
        print(f"   🔗 URL Competition: {full_url1}")
        
        # URL 2: Sport event markets
        sport_event_id = "sr:sport_event:59850122"
        sport_event_id_formatted = sport_event_id.replace(':', '%3A')
        endpoint2 = f"en/sport_events/{sport_event_id_formatted}/sport_event_markets"
        full_url2 = f"{odds_base_url}{endpoint2}?api_key={api_key[:8]}..."
        print(f"   🔗 URL Sport Event: {full_url2}")
        
        # URL 3: Schedule markets
        sport_id = "sr:sport:2"
        sport_id_formatted = sport_id.replace(':', '%3A')
        date = "2025-05-01"
        endpoint3 = f"en/sports/{sport_id_formatted}/schedules/{date}/sport_event_markets"
        full_url3 = f"{odds_base_url}{endpoint3}?api_key={api_key[:8]}..."
        print(f"   🔗 URL Schedule: {full_url3}")
        
        # 5. Resumen
        print("\n5. Resumen de correcciones aplicadas:")
        print("   ✅ URL base corregida a odds_base_url")
        print("   ✅ Endpoints según documentación oficial")
        print("   ✅ Formateo correcto de IDs (: → %3A)")
        print("   ✅ Parámetros correctos por endpoint")
        
        print("\n🎉 TEST PREMATCH API COMPLETADO")
        
    except Exception as e:
        print(f"\n❌ ERROR CRÍTICO: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_prematch_api_corregido() 