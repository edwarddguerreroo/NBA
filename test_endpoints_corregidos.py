"""
Test de Endpoints Corregidos: Player Props v2 API
================================================

Este script prueba los endpoints corregidos que usan Player Props v2 API
en lugar de los endpoints que requieren permisos especiales.
"""

import os
import sys
import logging
from datetime import datetime, timedelta

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_endpoints_que_funcionan():
    """
    Prueba los endpoints de Player Props v2 API que SÍ funcionan.
    """
    print("=" * 70)
    print("TEST: Endpoints Player Props v2 API que SÍ Funcionan")
    print("=" * 70)
    
    try:
        from utils.bookmakers.sportradar_api import SportradarAPI
        from utils.bookmakers.config import get_config
        
        # Obtener configuración
        config = get_config()
        api_key = config.get('sportradar', 'api_key') or os.getenv('SPORTRADAR_API')
        
        if not api_key:
            print("❌ ERROR: No se encontró API key")
            return False
        
        print(f"✅ API Key encontrada: {'*' * (len(api_key) - 4)}{api_key[-4:]}")
        
        # Inicializar API
        api = SportradarAPI(api_key=api_key)
        print("✅ SportradarAPI inicializada")
        
        resultados = {}
        
        # Test 1: Player Props por fecha (método nuevo)
        print(f"\n🔍 Test 1: Player Props por fecha")
        try:
            today = datetime.now().strftime("%Y-%m-%d")
            result = api.get_player_props_by_date(today)
            
            if result.get('success', False):
                events_count = len(result.get('events', []))
                print(f"✅ Éxito: {events_count} eventos con props encontrados")
                resultados['props_by_date'] = {'status': 'success', 'events': events_count}
            else:
                print(f"⚠️  Sin eventos: {result.get('error', 'No events for today')}")
                resultados['props_by_date'] = {'status': 'no_events', 'events': 0}
                
        except Exception as e:
            print(f"❌ Excepción: {e}")
            resultados['props_by_date'] = {'status': 'exception', 'error': str(e)}
        
        # Test 2: Player Props por competición (WNBA activa)
        print(f"\n🔍 Test 2: Player Props por competición (WNBA)")
        try:
            result = api.get_player_props_by_competition('sr:competition:486')
            
            if result.get('success', False):
                events_count = len(result.get('events', []))
                print(f"✅ Éxito: {events_count} eventos WNBA con props")
                
                # Mostrar algunos eventos
                for i, event in enumerate(result.get('events', [])[:2]):
                    event_info = event.get('event_info', {})
                    competitors = event_info.get('competitors', [])
                    if len(competitors) >= 2:
                        home = next((c['name'] for c in competitors if c.get('qualifier') == 'home'), 'Unknown')
                        away = next((c['name'] for c in competitors if c.get('qualifier') == 'away'), 'Unknown')
                        status = event_info.get('status', 'unknown')
                        print(f"   Evento {i+1}: {away} @ {home} ({status})")
                
                resultados['props_by_competition'] = {'status': 'success', 'events': events_count}
            else:
                print(f"⚠️  Sin eventos: {result.get('error', 'No events for WNBA')}")
                resultados['props_by_competition'] = {'status': 'no_events', 'events': 0}
                
        except Exception as e:
            print(f"❌ Excepción: {e}")
            resultados['props_by_competition'] = {'status': 'exception', 'error': str(e)}
        
        # Test 3: Schedule Markets (que ya funcionaba)
        print(f"\n🔍 Test 3: Schedule Markets")
        try:
            today = datetime.now().strftime("%Y-%m-%d")
            result = api.get_schedule_markets(sport_id=2, date=today)
            
            if result and 'sport_events' in result:
                events_count = len(result['sport_events'])
                print(f"✅ Éxito: {events_count} eventos con markets")
                resultados['schedule_markets'] = {'status': 'success', 'events': events_count}
            else:
                print(f"⚠️  Sin eventos: No sport_events found")
                resultados['schedule_markets'] = {'status': 'no_events', 'events': 0}
                
        except Exception as e:
            print(f"❌ Excepción: {e}")
            resultados['schedule_markets'] = {'status': 'exception', 'error': str(e)}
        
        return resultados
        
    except Exception as e:
        print(f"❌ Error general: {e}")
        return {}

def test_metodos_corregidos():
    """
    Prueba los métodos get_odds y get_player_props corregidos.
    """
    print("\n" + "=" * 70)
    print("TEST: Métodos Corregidos con Estrategias Alternativas")
    print("=" * 70)
    
    try:
        from utils.bookmakers.sportradar_api import SportradarAPI
        from utils.bookmakers.config import get_config
        
        config = get_config()
        api_key = config.get('sportradar', 'api_key') or os.getenv('SPORTRADAR_API')
        api = SportradarAPI(api_key=api_key)
        
        # Obtener un sport_event_id de ejemplo de WNBA
        print(f"🔍 Obteniendo sport_event_id de ejemplo...")
        try:
            wnba_schedules = api.get_competition_schedules('sr:competition:486', limit=3)
            
            test_event_id = None
            if 'schedules' in wnba_schedules and wnba_schedules['schedules']:
                for schedule in wnba_schedules['schedules']:
                    event = schedule.get('sport_event', {})
                    if event.get('id'):
                        test_event_id = event['id']
                        competitors = event.get('competitors', [])
                        if len(competitors) >= 2:
                            home = next((c['name'] for c in competitors if c.get('qualifier') == 'home'), 'Unknown')
                            away = next((c['name'] for c in competitors if c.get('qualifier') == 'away'), 'Unknown')
                            print(f"   Usando evento: {away} @ {home} (ID: {test_event_id})")
                        break
            
            if not test_event_id:
                print("⚠️  No se encontró sport_event_id de ejemplo")
                return {'status': 'no_test_event'}
            
        except Exception as e:
            print(f"❌ Error obteniendo evento de ejemplo: {e}")
            return {'status': 'error_getting_test_event'}
        
        resultados = {}
        
        # Test 1: get_player_props corregido
        print(f"\n🔍 Test 1: get_player_props corregido")
        try:
            result = api.get_player_props(test_event_id)
            
            if result.get('success', False):
                method = result.get('method', 'unknown')
                print(f"✅ Éxito: Props obtenidas usando método {method}")
                resultados['get_player_props'] = {'status': 'success', 'method': method}
            else:
                error = result.get('error', 'Unknown error')
                print(f"⚠️  Sin props: {error}")
                resultados['get_player_props'] = {'status': 'no_props', 'error': error}
                
        except Exception as e:
            print(f"❌ Excepción: {e}")
            resultados['get_player_props'] = {'status': 'exception', 'error': str(e)}
        
        # Test 2: get_odds corregido
        print(f"\n🔍 Test 2: get_odds corregido")
        try:
            result = api.get_odds(test_event_id)
            
            if result.get('success', False):
                method = result.get('method', 'unknown')
                print(f"✅ Éxito: Odds obtenidas usando método {method}")
                resultados['get_odds'] = {'status': 'success', 'method': method}
            else:
                error = result.get('error', 'Unknown error')
                print(f"⚠️  Sin odds: {error}")
                resultados['get_odds'] = {'status': 'no_odds', 'error': error}
                
        except Exception as e:
            print(f"❌ Excepción: {e}")
            resultados['get_odds'] = {'status': 'exception', 'error': str(e)}
        
        return resultados
        
    except Exception as e:
        print(f"❌ Error general: {e}")
        return {}

def test_integracion_targets():
    """
    Prueba la integración completa para nuestros targets.
    """
    print("\n" + "=" * 70)
    print("TEST: Integración Completa para Targets (PTS, AST, TRB, 3P)")
    print("=" * 70)
    
    try:
        from utils.bookmakers.bookmakers_data_fetcher import BookmakersDataFetcher
        
        # Inicializar fetcher
        fetcher = BookmakersDataFetcher()
        print("✅ BookmakersDataFetcher inicializado")
        
        # Test usando WNBA que está activa
        print(f"\n🔍 Probando obtener props para targets usando WNBA activa...")
        
        targets = ['PTS', 'AST', 'TRB', '3P']
        resultados = {}
        
        for target in targets:
            print(f"\n   🎯 Target: {target}")
            try:
                # Usar método optimizado
                props_data = fetcher.get_optimized_props_for_targets(
                    targets=[target]
                )
                
                if props_data.get('success', False):
                    games_total = props_data['summary']['total_games']
                    games_with_props = props_data['summary']['games_with_props']
                    target_props = props_data['summary']['props_by_target'][target]
                    optimization_info = props_data.get('optimization_info', {})
                    
                    print(f"   ✅ Éxito para {target}:")
                    print(f"      📊 Juegos totales: {games_total}")
                    print(f"      📊 Juegos con props: {games_with_props}")
                    print(f"      📊 Props {target}: {target_props}")
                    print(f"      🔧 Optimización: {optimization_info.get('season_phase', 'N/A')}")
                    
                    resultados[target] = {
                        'status': 'success',
                        'games_total': games_total,
                        'games_with_props': games_with_props,
                        'target_props': target_props
                    }
                else:
                    error = props_data.get('error', 'Unknown error')
                    print(f"   ❌ Error para {target}: {error}")
                    resultados[target] = {'status': 'error', 'error': error}
                    
            except Exception as e:
                print(f"   ❌ Excepción para {target}: {e}")
                resultados[target] = {'status': 'exception', 'error': str(e)}
        
        return resultados
        
    except Exception as e:
        print(f"❌ Error general: {e}")
        return {}

def main():
    """
    Función principal de pruebas.
    """
    print("🚀 TEST COMPLETO: Endpoints Corregidos Player Props v2 API")
    print("=" * 70)
    
    # Test 1: Endpoints que funcionan
    resultado_endpoints = test_endpoints_que_funcionan()
    
    # Test 2: Métodos corregidos
    resultado_metodos = test_metodos_corregidos()
    
    # Test 3: Integración para targets
    resultado_targets = test_integracion_targets()
    
    # Resumen final
    print("\n" + "=" * 70)
    print("RESUMEN FINAL")
    print("=" * 70)
    
    # Analizar resultados de endpoints
    if resultado_endpoints:
        endpoints_ok = sum(1 for r in resultado_endpoints.values() if r.get('status') == 'success')
        print(f"📊 Endpoints funcionando: {endpoints_ok}/{len(resultado_endpoints)}")
        
        for endpoint, result in resultado_endpoints.items():
            status = result.get('status', 'unknown')
            events = result.get('events', 0)
            if status == 'success':
                print(f"   ✅ {endpoint}: {events} eventos")
            else:
                print(f"   ⚠️  {endpoint}: {status}")
    
    # Analizar resultados de métodos
    if resultado_metodos:
        metodos_ok = sum(1 for r in resultado_metodos.values() if r.get('status') == 'success')
        print(f"\n🔧 Métodos corregidos: {metodos_ok}/{len(resultado_metodos)}")
        
        for metodo, result in resultado_metodos.items():
            status = result.get('status', 'unknown')
            method = result.get('method', 'N/A')
            if status == 'success':
                print(f"   ✅ {metodo}: usando {method}")
            else:
                print(f"   ⚠️  {metodo}: {status}")
    
    # Analizar resultados de targets
    if resultado_targets:
        targets_ok = sum(1 for r in resultado_targets.values() if r.get('status') == 'success')
        print(f"\n🎯 Targets funcionando: {targets_ok}/4")
        
        for target, result in resultado_targets.items():
            status = result.get('status', 'unknown')
            props = result.get('target_props', 0)
            if status == 'success':
                print(f"   ✅ {target}: {props} props")
            else:
                print(f"   ❌ {target}: {status}")
    
    # Score total
    total_tests = len(resultado_endpoints or {}) + len(resultado_metodos or {}) + 4  # 4 targets
    successful_tests = (
        sum(1 for r in (resultado_endpoints or {}).values() if r.get('status') == 'success') +
        sum(1 for r in (resultado_metodos or {}).values() if r.get('status') == 'success') +
        sum(1 for r in (resultado_targets or {}).values() if r.get('status') == 'success')
    )
    
    if total_tests > 0:
        success_rate = (successful_tests / total_tests) * 100
        print(f"\n🎯 SCORE FINAL: {successful_tests}/{total_tests} ({success_rate:.1f}%)")
        
        if success_rate >= 70:
            print(f"\n🎉 ENDPOINTS CORREGIDOS FUNCIONANDO")
            print("   ✅ Player Props v2 API operativa")
            print("   ✅ Estrategias alternativas implementadas")
            print("   ✅ Sistema adaptado a limitaciones de cuenta trial")
            return True
        else:
            print(f"\n⚠️  CORRECCIONES PARCIALES")
            print("   ⚠️  Algunos endpoints necesitan más trabajo")
    
    return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 