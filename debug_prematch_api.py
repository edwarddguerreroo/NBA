#!/usr/bin/env python3
"""
Debug específico para Prematch API
Prueba los endpoints exactos que funcionan en Swagger
"""

import os
import sys
sys.path.append('.')

import requests
import json
import logging
from urllib.parse import urljoin
from utils.bookmakers.config import get_config

# Configurar logging detallado
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_prematch_endpoints():
    """Test directo de los endpoints de Prematch API que funcionan en Swagger"""
    
    print("=" * 80)
    print("🔍 DEBUG PREMATCH API - ENDPOINTS ESPECÍFICOS")
    print("=" * 80)
    print()
    
    # Configuración
    config = get_config()
    api_key = config.get('sportradar', 'api_key')
    base_url = "https://api.sportradar.com/oddscomparison-prematch/trial/v2/"
    
    print(f"🔑 API Key: {'*' * 20}{api_key[-4:]}")
    print(f"🌐 Base URL: {base_url}")
    print()
    
    # Endpoints a probar (exactos de Swagger)
    endpoints_to_test = [
        {
            "name": "Competition Markets",
            "url": "https://api.sportradar.com/oddscomparison-prematch/trial/v2/en/competitions/sr%3Acompetition%3A486/sport_event_markets",
            "params": {"offset": 0, "limit": 5, "start": 5, "api_key": api_key},
            "description": "Mercados por competición (WNBA)"
        },
        {
            "name": "Sport Event Markets",
            "url": "https://api.sportradar.com/oddscomparison-prematch/trial/v2/en/sport_events/sr%3Asport_event%3A56328141/sport_event_markets",
            "params": {"limit": 5, "api_key": api_key},
            "description": "Mercados por evento específico"
        },
        {
            "name": "Schedule Markets",
            "url": "https://api.sportradar.com/oddscomparison-prematch/trial/v2/en/sports/sr%3Asport%3A2/schedules/2025-07-12/sport_event_markets",
            "params": {"limit": 5, "api_key": api_key},
            "description": "Mercados por fecha y deporte"
        }
    ]
    
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'NBA-Prediction-System/1.0',
        'Accept': 'application/json',
        'Accept-Encoding': 'gzip, deflate'
    })
    
    results = []
    
    for i, endpoint in enumerate(endpoints_to_test, 1):
        print(f"{i}️⃣ PROBANDO: {endpoint['name']}")
        print("-" * 50)
        print(f"📝 Descripción: {endpoint['description']}")
        print(f"🔗 URL: {endpoint['url']}")
        print(f"📋 Params: {endpoint['params']}")
        print()
        
        try:
            # Hacer petición
            print("🚀 Enviando petición...")
            response = session.get(
                endpoint['url'],
                params=endpoint['params'],
                timeout=30
            )
            
            print(f"📊 Status Code: {response.status_code}")
            print(f"📏 Content Length: {len(response.content)} bytes")
            print(f"🕒 Response Time: {response.elapsed.total_seconds():.2f}s")
            
            # Headers importantes
            content_type = response.headers.get('Content-Type', 'N/A')
            print(f"📄 Content-Type: {content_type}")
            
            if response.status_code == 200:
                try:
                    data = response.json()
                    print("✅ JSON válido recibido")
                    
                    # Analizar estructura
                    if isinstance(data, dict):
                        keys = list(data.keys())
                        print(f"🔑 Keys principales: {keys}")
                        
                        # Buscar sport_event_markets
                        if 'sport_event_markets' in data:
                            markets = data['sport_event_markets']
                            print(f"📈 Sport Event Markets encontrados: {len(markets)}")
                            
                            # Mostrar algunos markets
                            for j, market in enumerate(markets[:3]):
                                sport_event = market.get('sport_event', {})
                                event_id = sport_event.get('id', 'N/A')
                                event_name = f"{sport_event.get('competitors', [{}])[0].get('name', 'N/A')} vs {sport_event.get('competitors', [{}])[-1].get('name', 'N/A')}" if sport_event.get('competitors') else 'N/A'
                                
                                markets_count = len(market.get('markets', []))
                                print(f"   🏀 Market {j+1}: {event_name} ({event_id}) - {markets_count} markets")
                                
                                # Mostrar algunos markets específicos
                                for k, market_detail in enumerate(market.get('markets', [])[:2]):
                                    market_id = market_detail.get('id', 'N/A')
                                    market_name = market_detail.get('name', 'N/A')
                                    outcomes_count = len(market_detail.get('outcomes', []))
                                    print(f"      📊 {market_name} (ID: {market_id}) - {outcomes_count} outcomes")
                        else:
                            print("⚠️ No se encontró 'sport_event_markets' en la respuesta")
                            print(f"📋 Estructura recibida: {json.dumps(data, indent=2)[:500]}...")
                    
                    results.append({
                        'endpoint': endpoint['name'],
                        'success': True,
                        'status_code': response.status_code,
                        'data_size': len(response.content),
                        'markets_found': len(data.get('sport_event_markets', [])) if isinstance(data, dict) else 0
                    })
                    
                except json.JSONDecodeError as e:
                    print(f"❌ Error decodificando JSON: {e}")
                    print(f"📄 Contenido recibido: {response.text[:500]}...")
                    results.append({
                        'endpoint': endpoint['name'],
                        'success': False,
                        'status_code': response.status_code,
                        'error': f"JSON decode error: {e}"
                    })
            else:
                print(f"❌ Error HTTP {response.status_code}")
                print(f"📄 Respuesta: {response.text[:500]}...")
                results.append({
                    'endpoint': endpoint['name'],
                    'success': False,
                    'status_code': response.status_code,
                    'error': f"HTTP {response.status_code}: {response.text[:200]}"
                })
                
        except Exception as e:
            print(f"❌ Excepción: {e}")
            results.append({
                'endpoint': endpoint['name'],
                'success': False,
                'error': str(e)
            })
        
        print()
        print("=" * 50)
        print()
    
    # Resumen final
    print("📋 RESUMEN DE RESULTADOS")
    print("=" * 50)
    
    for result in results:
        status = "✅ SUCCESS" if result['success'] else "❌ FAILED"
        print(f"{status} {result['endpoint']}")
        
        if result['success']:
            print(f"   📊 Status: {result['status_code']}")
            print(f"   📏 Data: {result['data_size']} bytes")
            print(f"   📈 Markets: {result['markets_found']}")
        else:
            print(f"   ❌ Error: {result['error']}")
        print()
    
    # Estadísticas
    successful = sum(1 for r in results if r['success'])
    total = len(results)
    
    print(f"📊 ESTADÍSTICAS FINALES: {successful}/{total} endpoints exitosos")
    
    if successful == total:
        print("🎉 ¡Todos los endpoints funcionan correctamente!")
    else:
        print("⚠️ Algunos endpoints tienen problemas")

def compare_with_sportradar_api():
    """Comparar con lo que hace SportradarAPI internamente"""
    
    print("=" * 80)
    print("🔍 COMPARACIÓN CON SPORTRADAR API INTERNA")
    print("=" * 80)
    print()
    
    try:
        from utils.bookmakers.sportradar_api import SportradarAPI
        
        api = SportradarAPI()
        sport_event_id = "sr:sport_event:56328141"
        
        print(f"🎯 Probando con evento: {sport_event_id}")
        print()
        
        # 1. Test get_prematch_odds
        print("1️⃣ MÉTODO get_prematch_odds")
        print("-" * 30)
        
        try:
            prematch_result = api.get_prematch_odds(sport_event_id)
            
            print(f"✅ Método ejecutado")
            print(f"📊 Success: {prematch_result.get('success', False)}")
            print(f"📈 Markets: {len(prematch_result.get('markets', []))}")
            
            if prematch_result.get('success', False):
                markets = prematch_result.get('markets', [])
                print(f"📋 Markets encontrados: {len(markets)}")
                
                for i, market in enumerate(markets[:3]):
                    market_id = market.get('id', 'N/A')
                    market_name = market.get('name', 'N/A')
                    outcomes = len(market.get('outcomes', []))
                    print(f"   📊 Market {i+1}: {market_name} (ID: {market_id}) - {outcomes} outcomes")
            else:
                print(f"❌ Error: {prematch_result.get('error', 'Unknown error')}")
                
        except Exception as e:
            print(f"❌ Excepción en get_prematch_odds: {e}")
        
        print()
        
        # 2. Test get_prematch_odds_by_competition
        print("2️⃣ MÉTODO get_prematch_odds_by_competition")
        print("-" * 30)
        
        try:
            competition_result = api.get_prematch_odds_by_competition('sr:competition:486')
            
            print(f"✅ Método ejecutado")
            print(f"📊 Success: {competition_result.get('success', False)}")
            
            if competition_result.get('success', False):
                sport_event_markets = competition_result.get('sport_event_markets', [])
                print(f"📋 Sport Event Markets: {len(sport_event_markets)}")
                
                for i, event_market in enumerate(sport_event_markets[:2]):
                    sport_event = event_market.get('sport_event', {})
                    event_id = sport_event.get('id', 'N/A')
                    markets_count = len(event_market.get('markets', []))
                    print(f"   🏀 Event {i+1}: {event_id} - {markets_count} markets")
            else:
                print(f"❌ Error: {competition_result.get('error', 'Unknown error')}")
                
        except Exception as e:
            print(f"❌ Excepción en get_prematch_odds_by_competition: {e}")
            
    except Exception as e:
        print(f"❌ Error importando SportradarAPI: {e}")

if __name__ == "__main__":
    print("🚀 INICIANDO DEBUG DE PREMATCH API")
    print()
    
    # Test 1: Endpoints directos
    test_prematch_endpoints()
    
    print("\n" + "=" * 80 + "\n")
    
    # Test 2: Comparación con API interna
    compare_with_sportradar_api()
    
    print("\n�� Debug completado") 