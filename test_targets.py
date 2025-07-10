#!/usr/bin/env python3
"""
Test completo para verificar que todos los métodos funcionen correctamente
para obtener cuotas de nuestros targets NBA.
"""

import sys
sys.path.append('.')

from utils.bookmakers.sportradar_api import SportradarAPI
from utils.bookmakers.config import BookmakersConfig

def test_basic_endpoints():
    """Prueba endpoints básicos."""
    print("=" * 60)
    print("TEST DE ENDPOINTS BÁSICOS")
    print("=" * 60)
    
    api = SportradarAPI()
    
    # Test 1: Bookmakers
    print("\n=== TEST 1: BOOKMAKERS ===")
    try:
        bookmakers = api.get_bookmakers()
        books = bookmakers.get('books', [])
        print(f"✅ Bookmakers obtenidos: {len(books)}")
        for book in books[:3]:
            print(f"  - {book['name']} ({book['id']})")
    except Exception as e:
        print(f"❌ Error en bookmakers: {e}")
        return False
    
    # Test 2: Sports
    print("\n=== TEST 2: SPORTS ===")
    try:
        sports = api.get_sports()
        sports_list = sports.get('sports', [])
        print(f"✅ Sports obtenidos: {len(sports_list)}")
        
        basketball_id = None
        for sport in sports_list:
            print(f"  - {sport['name']} (ID: {sport['id']})")
            if 'basketball' in sport['name'].lower():
                basketball_id = sport['id']
                print(f"    *** BASKETBALL FOUND: {basketball_id} ***")
        
        if not basketball_id:
            print("❌ No se encontró Basketball")
            return False
            
    except Exception as e:
        print(f"❌ Error en sports: {e}")
        return False
    
    # Test 3: Basketball Competitions
    print(f"\n=== TEST 3: BASKETBALL COMPETITIONS (ID: {basketball_id}) ===")
    try:
        competitions = api.get_sport_competitions(basketball_id)
        comp_list = competitions.get('competitions', [])
        print(f"✅ Competitions obtenidas: {len(comp_list)}")
        
        nba_competition_id = None
        for comp in comp_list:
            print(f"  - {comp['name']} (ID: {comp['id']})")
            if 'nba' in comp['name'].lower():
                nba_competition_id = comp['id']
                print(f"    *** NBA FOUND: {nba_competition_id} ***")
                
        if not nba_competition_id:
            print("❌ No se encontró NBA")
            return False
            
    except Exception as e:
        print(f"❌ Error en competitions: {e}")
        return False
    
    # Test 4: NBA Schedules
    print(f"\n=== TEST 4: NBA SCHEDULES (ID: {nba_competition_id}) ===")
    try:
        schedules = api.get_competition_schedules(nba_competition_id, limit=5)
        schedule_list = schedules.get('schedules', [])
        print(f"✅ Schedules obtenidos: {len(schedule_list)}")
        
        sport_event_ids = []
        for schedule in schedule_list:
            event = schedule.get('sport_event', {})
            event_id = event.get('id', 'N/A')
            status = event.get('status', 'N/A')
            print(f"  - {event_id} - Status: {status}")
            if event_id != 'N/A':
                sport_event_ids.append(event_id)
                
        print(f"Sport Event IDs encontrados: {len(sport_event_ids)}")
        return sport_event_ids[:2]  # Retornar primeros 2 para testing
        
    except Exception as e:
        print(f"❌ Error en schedules: {e}")
        return False

def test_odds_endpoints(sport_event_ids):
    """Prueba endpoints de odds."""
    print("\n" + "=" * 60)
    print("TEST DE ENDPOINTS DE ODDS")
    print("=" * 60)
    
    api = SportradarAPI()
    
    for i, sport_event_id in enumerate(sport_event_ids):
        print(f"\n=== TEST ODDS {i+1}: {sport_event_id} ===")
        
        # Test Sport Event Markets
        try:
            markets = api.get_odds(sport_event_id)
            print(f"✅ Markets obtenidos para {sport_event_id}")
            
            # Mostrar estructura
            if 'markets' in markets:
                print(f"  - Markets disponibles: {len(markets['markets'])}")
                for market in markets['markets'][:3]:
                    print(f"    - {market.get('name', 'N/A')}")
            else:
                print(f"  - Estructura: {list(markets.keys())}")
                
        except Exception as e:
            print(f"❌ Error en markets para {sport_event_id}: {e}")
        
        # Test Player Props
        try:
            props = api.get_player_props(sport_event_id)
            print(f"✅ Player Props obtenidos para {sport_event_id}")
            
            # Mostrar estructura
            if 'markets' in props:
                print(f"  - Props disponibles: {len(props['markets'])}")
            else:
                print(f"  - Estructura: {list(props.keys())}")
                
        except Exception as e:
            print(f"❌ Error en props para {sport_event_id}: {e}")

def test_target_methods():
    """Prueba métodos específicos para targets."""
    print("\n" + "=" * 60)
    print("TEST DE MÉTODOS PARA TARGETS")
    print("=" * 60)
    
    api = SportradarAPI()
    
    # Test método específico para NBA odds
    print("\n=== TEST NBA ODDS TODAY ===")
    try:
        nba_odds = api.get_nba_odds_today()
        print(f"✅ NBA Odds Today obtenidos")
        print(f"  - Fecha: {nba_odds.get('date', 'N/A')}")
        print(f"  - Total games: {nba_odds.get('total_games', 0)}")
        print(f"  - Message: {nba_odds.get('message', 'N/A')}")
    except Exception as e:
        print(f"❌ Error en NBA odds today: {e}")
    
    # Test método para targets específicos
    print("\n=== TEST NBA ODDS FOR TARGETS ===")
    try:
        targets = ['PTS', 'AST', 'TRB', 'is_win', 'total_points']
        target_odds = api.get_nba_odds_for_targets(targets=targets)
        print(f"✅ NBA Odds for Targets obtenidos")
        print(f"  - Total games: {target_odds.get('total_games', 0)}")
        print(f"  - Games with odds: {target_odds.get('games_with_odds', 0)}")
        print(f"  - Targets found: {target_odds.get('targets_found', {})}")
    except Exception as e:
        print(f"❌ Error en NBA odds for targets: {e}")

def main():
    """Función principal."""
    print("NBA TARGETS ODDS - TEST COMPLETO")
    print("=" * 60)
    
    # Test 1: Endpoints básicos
    sport_event_ids = test_basic_endpoints()
    
    if sport_event_ids:
        # Test 2: Endpoints de odds
        test_odds_endpoints(sport_event_ids)
    
    # Test 3: Métodos específicos para targets
    test_target_methods()
    
    print("\n" + "=" * 60)
    print("TEST COMPLETADO")
    print("=" * 60)

if __name__ == "__main__":
    main() 