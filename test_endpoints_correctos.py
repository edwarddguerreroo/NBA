#!/usr/bin/env python3
"""
Test de Endpoints Correctos - NBA Sistema
=========================================

Usando los endpoints correctos que el usuario confirmó:

1. Player Props API v2 (para jugadores):
   https://api.sportradar.com/oddscomparison-player-props/trial/v2/en/sports/sr%3Asport%3A2/schedules/2025-05-01/players_props

2. Prematch API (para equipos):
   https://api.sportradar.com/oddscomparison-prematch/trial/v2/en/competitions/sr%3Acompetition%3A486/sport_event_markets

Market IDs correctos:
- Player Props: sr:market:921 (PTS), sr:market:922 (AST), sr:market:923 (TRB), sr:market:924 (3P), sr:market:8008 (DD)
- Team Markets: 1 (is_win), 225 (total_points), 227/228 (teams_points)
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.bookmakers.sportradar_api import SportradarAPI
from utils.bookmakers.config import get_config
import json
from datetime import datetime

def test_endpoints_correctos():
    """
    Test usando los endpoints correctos especificados por el usuario
    """
    print("🎯 TEST DE ENDPOINTS CORRECTOS")
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
        print(f"   ✅ API inicializada con key: {'*' * (len(api_key) - 4)}{api_key[-4:]}")
        
        # 2. Test Player Props API v2 - Endpoint por fecha
        print("\n2. Probando Player Props API v2 (endpoint por fecha)...")
        
        try:
            # Usar el endpoint exacto que funciona
            date = "2025-05-01"
            sport_id = 2  # Basketball
            
            # Método correcto: get_player_props_by_date
            player_props_result = sportradar.get_player_props_by_date(date=date, sport_id=sport_id)
            
            print(f"   ✅ Resultado: {player_props_result.get('success', False)}")
            
            if player_props_result.get('success'):
                events = player_props_result.get('events', [])
                print(f"   📊 Eventos encontrados: {len(events)}")
                
                # Mostrar algunos eventos
                for i, event in enumerate(events[:3]):
                    event_id = event.get('sport_event_id', 'N/A')
                    props = event.get('props', {})
                    print(f"   🏀 Evento {i+1}: {event_id}")
                    print(f"      Props disponibles: {len(props)}")
                    
                    # Mostrar info del evento
                    event_info = event.get('event_info', {})
                    competitors = event_info.get('competitors', [])
                    for comp in competitors[:2]:
                        name = comp.get('name', 'N/A')
                        qualifier = comp.get('qualifier', 'N/A')
                        print(f"      {qualifier}: {name}")
            else:
                print(f"   ⚠️  Error: {player_props_result.get('error', 'Unknown')}")
                
        except Exception as e:
            print(f"   ❌ Error en Player Props: {e}")
        
        # 3. Test Prematch API - Competition markets
        print("\n3. Probando Prematch API (competition markets)...")
        
        try:
            # Usar WNBA que está activa (sr:competition:486)
            competition_id = "sr:competition:486"  # WNBA
            
            # Método correcto: get_prematch_odds_by_competition
            prematch_result = sportradar.get_prematch_odds_by_competition(
                competition_id=competition_id,
                limit=5
            )
            
            print(f"   ✅ Resultado: {prematch_result.get('success', False)}")
            
            if prematch_result.get('success'):
                data = prematch_result.get('data', {})
                print(f"   📊 Datos obtenidos: {len(str(data))} caracteres")
                
                # Verificar estructura
                if 'competition_sport_event_markets' in data:
                    events = data['competition_sport_event_markets']
                    print(f"   🏀 Eventos con markets: {len(events)}")
                    
                    # Mostrar algunos eventos
                    for i, event in enumerate(events[:2]):
                        sport_event = event.get('sport_event', {})
                        event_id = sport_event.get('id', 'N/A')
                        markets = event.get('markets', [])
                        
                        print(f"   🎯 Evento {i+1}: {event_id}")
                        print(f"      Markets disponibles: {len(markets)}")
                        
                        # Mostrar equipos
                        competitors = sport_event.get('competitors', [])
                        for comp in competitors:
                            name = comp.get('name', 'N/A')
                            qualifier = comp.get('qualifier', 'N/A')
                            print(f"      {qualifier}: {name}")
                        
                        # Mostrar algunos markets
                        for market in markets[:3]:
                            market_id = market.get('id', 'N/A')
                            market_name = market.get('name', 'N/A')
                            books = market.get('books', [])
                            print(f"      📈 Market {market_id}: {market_name} ({len(books)} casas)")
                            
                            # Mostrar casas de apuestas
                            for book in books[:2]:
                                book_name = book.get('name', 'N/A')
                                outcomes = book.get('outcomes', [])
                                print(f"         💰 {book_name}: {len(outcomes)} outcomes")
                else:
                    print("   ⚠️  Estructura de datos inesperada")
            else:
                print(f"   ⚠️  Error: {prematch_result.get('error', 'Unknown')}")
                
        except Exception as e:
            print(f"   ❌ Error en Prematch API: {e}")
        
        # 4. Test directo del evento específico (Player Props)
        print("\n4. Probando Player Props para evento específico...")
        
        try:
            # Evento específico que el usuario mencionó
            sport_event_id = "sr:sport_event:59850122"
            
            # Método directo: get_player_props
            direct_props = sportradar.get_player_props(sport_event_id)
            
            print(f"   ✅ Resultado: {direct_props.get('success', False)}")
            
            if direct_props.get('success'):
                data = direct_props.get('data', {})
                print(f"   📊 Datos obtenidos: {len(str(data))} caracteres")
                
                # Verificar si hay props
                if 'players_props' in data:
                    props = data['players_props']
                    print(f"   👥 Player props: {len(props)}")
                else:
                    print("   ⚠️  Sin player props (evento histórico)")
                
                # Verificar info del evento
                if 'sport_event' in data:
                    event = data['sport_event']
                    print(f"   🏀 Evento: {event.get('id', 'N/A')}")
                    
                    competitors = event.get('competitors', [])
                    for comp in competitors:
                        name = comp.get('name', 'N/A')
                        qualifier = comp.get('qualifier', 'N/A')
                        print(f"      {qualifier}: {name}")
            else:
                print(f"   ⚠️  Error: {direct_props.get('error', 'Unknown')}")
                
        except Exception as e:
            print(f"   ❌ Error en Props directo: {e}")
        
        # 5. Resumen de Market IDs
        print("\n5. Market IDs confirmados:")
        print("   Player Props API:")
        print("   - sr:market:921: total_points → PTS")
        print("   - sr:market:922: total_assists → AST") 
        print("   - sr:market:923: total_rebounds → TRB")
        print("   - sr:market:924: total_3pt_field_goals → 3P")
        print("   - sr:market:8008: double_double → DD")
        print("")
        print("   Prematch API:")
        print("   - Market ID 1: winner → is_win")
        print("   - Market ID 225: total_incl_overtime → total_points")
        print("   - Market ID 227: home_total_incl_overtime → teams_points")
        print("   - Market ID 228: away_total_incl_overtime → teams_points")
        
        print("\n🎉 TEST DE ENDPOINTS COMPLETADO")
        
    except Exception as e:
        print(f"\n❌ ERROR CRÍTICO: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_endpoints_correctos() 