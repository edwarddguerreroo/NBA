#!/usr/bin/env python3
"""
Test Completo de Todos los Endpoints de Sportradar
================================================

Este script prueba todos los endpoints disponibles de Sportradar para identificar:
- Cuáles funcionan correctamente
- Cuáles dan error 403 (no autorizados)
- Cuáles requieren parámetros específicos
- Qué datos están disponibles con la API key actual

Esto nos ayudará a configurar el sistema para usar solo los endpoints que funcionan.
"""

import os
import sys
import json
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def test_all_sportradar_endpoints():
    """Prueba todos los endpoints de Sportradar disponibles."""
    
    print("🔍 PRUEBA COMPLETA DE ENDPOINTS SPORTRADAR")
    print("=" * 60)
    
    # Verificar API key
    api_key = os.getenv('SPORTRADAR_API')
    if not api_key:
        print("❌ ERROR: Variable SPORTRADAR_API no configurada")
        return False
    
    print(f"✅ API Key: {'*' * 36}{api_key[-4:]}")
    
    # Importar módulos
    try:
        from utils.bookmakers import SportradarAPI
        api = SportradarAPI(api_key=api_key)
        print("✅ SportradarAPI inicializada")
    except Exception as e:
        print(f"❌ Error inicializando API: {e}")
        return False
    
    # Definir todos los endpoints para probar
    endpoints_to_test = [
        # === ENDPOINTS BÁSICOS ===
        {
            'name': 'Test Connection',
            'method': 'test_connection',
            'params': {},
            'description': 'Prueba básica de conexión'
        },
        
        # === SPORTS API (Basketball) ===
        {
            'name': 'League Hierarchy',
            'method': '_make_request',
            'params': {'endpoint': 'league/hierarchy.json'},
            'description': 'Información de equipos NBA'
        },
        {
            'name': 'Teams',
            'method': 'get_teams',
            'params': {},
            'description': 'Lista de equipos NBA'
        },
        {
            'name': 'Current Season Schedule',
            'method': '_make_request',
            'params': {'endpoint': 'games/2024/REG/schedule.json'},
            'description': 'Calendario temporada regular 2024'
        },
        {
            'name': 'Current Season Schedule 2025',
            'method': '_make_request',
            'params': {'endpoint': 'games/2025/REG/schedule.json'},
            'description': 'Calendario temporada regular 2025'
        },
        {
            'name': 'Playoffs Schedule 2024',
            'method': '_make_request',
            'params': {'endpoint': 'games/2024/PST/schedule.json'},
            'description': 'Calendario playoffs 2024'
        },
        
        # === ODDS COMPARISON API ===
        {
            'name': 'Sports List',
            'method': 'get_sports',
            'params': {},
            'description': 'Lista de deportes disponibles'
        },
        {
            'name': 'Bookmakers List',
            'method': 'get_bookmakers',
            'params': {},
            'description': 'Lista de casas de apuestas'
        },
        {
            'name': 'Basketball Competitions',
            'method': 'get_sport_competitions',
            'params': {'sport_id': 2},
            'description': 'Competiciones de basketball'
        },
        
        # === DAILY SCHEDULES (diferentes fechas) ===
        {
            'name': 'Daily Schedule Today',
            'method': 'get_daily_odds_schedule',
            'params': {'date': datetime.now().strftime('%Y-%m-%d')},
            'description': 'Schedule de hoy'
        },
        {
            'name': 'Daily Schedule Yesterday',
            'method': 'get_daily_odds_schedule',
            'params': {'date': (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')},
            'description': 'Schedule de ayer'
        },
        {
            'name': 'Daily Schedule Tomorrow',
            'method': 'get_daily_odds_schedule',
            'params': {'date': (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')},
            'description': 'Schedule de mañana'
        },
        {
            'name': 'Daily Schedule Next Week',
            'method': 'get_daily_odds_schedule',
            'params': {'date': (datetime.now() + timedelta(days=7)).strftime('%Y-%m-%d')},
            'description': 'Schedule próxima semana'
        },
        
        # === FECHAS DE TEMPORADA NBA ===
        {
            'name': 'Schedule Oct 2024',
            'method': 'get_daily_odds_schedule',
            'params': {'date': '2024-10-15'},
            'description': 'Inicio temporada 2024-25'
        },
        {
            'name': 'Schedule Nov 2024',
            'method': 'get_daily_odds_schedule',
            'params': {'date': '2024-11-15'},
            'description': 'Temporada regular nov 2024'
        },
        {
            'name': 'Schedule Dec 2024',
            'method': 'get_daily_odds_schedule',
            'params': {'date': '2024-12-15'},
            'description': 'Temporada regular dic 2024'
        },
        {
            'name': 'Schedule Jan 2025',
            'method': 'get_daily_odds_schedule',
            'params': {'date': '2025-01-15'},
            'description': 'Temporada regular ene 2025'
        },
        {
            'name': 'Schedule Feb 2025',
            'method': 'get_daily_odds_schedule',
            'params': {'date': '2025-02-15'},
            'description': 'Temporada regular feb 2025'
        },
        {
            'name': 'Schedule Mar 2025',
            'method': 'get_daily_odds_schedule',
            'params': {'date': '2025-03-15'},
            'description': 'Temporada regular mar 2025'
        },
        {
            'name': 'Schedule Apr 2025',
            'method': 'get_daily_odds_schedule',
            'params': {'date': '2025-04-15'},
            'description': 'Final temporada regular 2025'
        },
        
        # === MÉTODOS ESPECÍFICOS DEL SISTEMA ===
        {
            'name': 'Next NBA Games',
            'method': 'get_next_nba_games',
            'params': {'days_ahead': 7, 'max_games': 10},
            'description': 'Próximos juegos NBA'
        },
        {
            'name': 'NBA Odds for Targets',
            'method': 'get_nba_odds_for_targets',
            'params': {'date': '2025-01-15'},
            'description': 'Odds para targets específicos'
        },
        {
            'name': 'Live and Upcoming Odds',
            'method': 'get_live_and_upcoming_odds',
            'params': {},
            'description': 'Odds en vivo y próximos'
        },
        {
            'name': 'Automated Predictions Data',
            'method': 'get_automated_predictions_data',
            'params': {'max_games': 5, 'days_ahead': 3},
            'description': 'Datos automatizados para predicciones'
        },
        
        # === ENDPOINTS DIRECTOS (URLs manuales) ===
        {
            'name': 'Direct Basketball API',
            'method': '_make_request',
            'params': {'endpoint': 'tournaments.json'},
            'description': 'Torneos de basketball'
        },
        {
            'name': 'Direct Season Info',
            'method': '_make_request',
            'params': {'endpoint': 'seasons.json'},
            'description': 'Información de temporadas'
        },
    ]
    
    # Ejecutar pruebas
    results = []
    successful_endpoints = []
    failed_endpoints = []
    
    print(f"\n🚀 Probando {len(endpoints_to_test)} endpoints...")
    print("-" * 60)
    
    for i, endpoint_test in enumerate(endpoints_to_test, 1):
        name = endpoint_test['name']
        method = endpoint_test['method']
        params = endpoint_test['params']
        description = endpoint_test['description']
        
        print(f"\n[{i:2d}/{len(endpoints_to_test)}] {name}")
        print(f"      📝 {description}")
        
        try:
            # Ejecutar el método
            if hasattr(api, method):
                method_func = getattr(api, method)
                if params:
                    result = method_func(**params)
                else:
                    result = method_func()
            else:
                print(f"      ❌ Método {method} no existe")
                failed_endpoints.append(name)
                continue
            
            # Analizar resultado
            if isinstance(result, dict):
                if result.get('success', True):
                    # Contar datos si es posible
                    data_count = 0
                    if 'data' in result:
                        if isinstance(result['data'], list):
                            data_count = len(result['data'])
                        elif isinstance(result['data'], dict):
                            data_count = len(result['data'])
                    elif 'games' in result:
                        if isinstance(result['games'], list):
                            data_count = len(result['games'])
                        elif isinstance(result['games'], dict):
                            data_count = len(result['games'])
                    elif 'teams' in result:
                        if isinstance(result['teams'], list):
                            data_count = len(result['teams'])
                    elif 'conferences' in result:
                        data_count = len(result['conferences'])
                    elif 'divisions' in result:
                        data_count = len(result['divisions'])
                    
                    print(f"      ✅ ÉXITO - {data_count} elementos")
                    successful_endpoints.append({
                        'name': name,
                        'method': method,
                        'params': params,
                        'data_count': data_count,
                        'description': description
                    })
                else:
                    error_msg = result.get('error', 'Error desconocido')
                    print(f"      ❌ ERROR - {error_msg}")
                    failed_endpoints.append(name)
            else:
                print(f"      ✅ ÉXITO - Respuesta no estándar")
                successful_endpoints.append({
                    'name': name,
                    'method': method,
                    'params': params,
                    'data_count': 'N/A',
                    'description': description
                })
            
        except Exception as e:
            error_str = str(e)
            if '403' in error_str or 'Forbidden' in error_str:
                print(f"      🚫 ERROR 403 - No autorizado")
            elif '401' in error_str:
                print(f"      🔐 ERROR 401 - No autenticado")
            elif '404' in error_str:
                print(f"      🔍 ERROR 404 - No encontrado")
            elif '429' in error_str:
                print(f"      ⏱️  ERROR 429 - Rate limit")
            else:
                print(f"      ❌ ERROR - {error_str}")
            
            failed_endpoints.append(name)
        
        # Pausa para no saturar la API
        time.sleep(0.5)
    
    # Resumen final
    print("\n" + "=" * 60)
    print("📊 RESUMEN DE PRUEBAS")
    print("=" * 60)
    
    print(f"✅ ENDPOINTS EXITOSOS: {len(successful_endpoints)}")
    for endpoint in successful_endpoints:
        print(f"   • {endpoint['name']} - {endpoint['data_count']} elementos")
    
    print(f"\n❌ ENDPOINTS FALLIDOS: {len(failed_endpoints)}")
    for endpoint_name in failed_endpoints:
        print(f"   • {endpoint_name}")
    
    # Guardar resultados
    results_data = {
        'timestamp': datetime.now().isoformat(),
        'api_key_suffix': api_key[-4:],
        'total_tested': len(endpoints_to_test),
        'successful': len(successful_endpoints),
        'failed': len(failed_endpoints),
        'successful_endpoints': successful_endpoints,
        'failed_endpoints': failed_endpoints
    }
    
    with open('sportradar_endpoints_results.json', 'w') as f:
        json.dump(results_data, f, indent=2)
    
    print(f"\n💾 Resultados guardados en: sportradar_endpoints_results.json")
    
    # Recomendaciones
    print("\n" + "=" * 60)
    print("💡 RECOMENDACIONES")
    print("=" * 60)
    
    if successful_endpoints:
        print("✅ Endpoints funcionales encontrados:")
        for endpoint in successful_endpoints[:5]:  # Mostrar top 5
            print(f"   • {endpoint['name']}: {endpoint['description']}")
        
        print("\n🔧 Configurar el sistema para usar solo estos endpoints")
    else:
        print("❌ No se encontraron endpoints funcionales")
        print("🔧 Verificar:")
        print("   • API key válida")
        print("   • Tipo de cuenta (trial/production)")
        print("   • Permisos de la API key")
    
    return len(successful_endpoints) > 0

if __name__ == "__main__":
    success = test_all_sportradar_endpoints()
    sys.exit(0 if success else 1) 