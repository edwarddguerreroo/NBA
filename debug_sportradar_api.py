#!/usr/bin/env python3
"""
Debug Sportradar API - Verificar Estado y Endpoints
==================================================

Script para diagnosticar problemas con la API de Sportradar
y verificar qué endpoints están disponibles.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.bookmakers.sportradar_api import SportradarAPI
from utils.bookmakers.config import get_config
import json
from datetime import datetime

def debug_sportradar_api():
    """
    Debug completo de la API de Sportradar
    """
    print("🔍 DEBUG SPORTRADAR API")
    print("=" * 50)
    
    try:
        # 1. Verificar configuración
        print("\n1. Verificando configuración...")
        config = get_config()
        
        # Obtener API key
        api_key = (
            config.get('sportradar', 'api_key') or
            os.getenv('SPORTRADAR_API')
        )
        
        if api_key:
            print(f"   ✅ API Key encontrada: {'*' * (len(api_key) - 4)}{api_key[-4:]}")
        else:
            print("   ❌ API Key no encontrada")
            return
        
        # 2. Inicializar API
        print("\n2. Inicializando Sportradar API...")
        sportradar = SportradarAPI(api_key=api_key)
        
        # 3. Verificar URLs configuradas
        print("\n3. Verificando URLs configuradas...")
        print(f"   Player Props URL: {config.get('sportradar', 'player_props_url')}")
        print(f"   Odds Base URL: {config.get('sportradar', 'odds_base_url')}")
        print(f"   Basketball URL: {config.get('sportradar', 'base_url')}")
        
        # 4. Probar conexión básica
        print("\n4. Probando conexión básica...")
        try:
            connection_test = sportradar.test_connection()
            print(f"   Resultado: {connection_test}")
        except Exception as e:
            print(f"   ❌ Error en test de conexión: {e}")
        
        # 5. Probar endpoints específicos
        print("\n5. Probando endpoints específicos...")
        
        # 5.1. Probar endpoint de schedule general
        print("\n   5.1. Probando schedule general...")
        try:
            # Usar fecha actual para ver partidos próximos
            today = datetime.now().strftime('%Y-%m-%d')
            schedule_result = sportradar.get_schedule(date=today)
            print(f"        Resultado schedule: {schedule_result.get('success', False)}")
            if not schedule_result.get('success', False):
                print(f"        Error: {schedule_result.get('error', 'Unknown')}")
        except Exception as e:
            print(f"        ❌ Error en schedule: {e}")
        
        # 5.2. Probar endpoint de player props con evento específico
        print("\n   5.2. Probando player props con evento 59850122...")
        try:
            sport_event_id = "sr:sport_event:59850122"
            player_props = sportradar.get_player_props(sport_event_id)
            print(f"        Resultado player props: {player_props.get('success', False)}")
            if not player_props.get('success', False):
                print(f"        Error: {player_props.get('error', 'Unknown')}")
        except Exception as e:
            print(f"        ❌ Error en player props: {e}")
        
        # 5.3. Probar endpoint de prematch
        print("\n   5.3. Probando prematch odds con evento 59850122...")
        try:
            sport_event_id = "sr:sport_event:59850122"
            prematch_odds = sportradar.get_prematch_odds(sport_event_id)
            print(f"        Resultado prematch: {prematch_odds.get('success', False)}")
            if not prematch_odds.get('success', False):
                print(f"        Error: {prematch_odds.get('error', 'Unknown')}")
        except Exception as e:
            print(f"        ❌ Error en prematch: {e}")
        
        # 6. Probar con eventos más recientes
        print("\n6. Probando con eventos más recientes...")
        
        # 6.1. Obtener schedule de hoy
        print("\n   6.1. Obteniendo schedule de hoy...")
        try:
            today = datetime.now().strftime('%Y-%m-%d')
            schedule_today = sportradar.get_schedule(date=today)
            
            if schedule_today.get('success', False):
                games = schedule_today.get('games', [])
                print(f"        ✅ Encontrados {len(games)} partidos para hoy")
                
                # Mostrar algunos partidos
                for i, game in enumerate(games[:3]):
                    print(f"        Partido {i+1}: {game.get('home', 'N/A')} vs {game.get('away', 'N/A')}")
                    print(f"                   ID: {game.get('id', 'N/A')}")
                    print(f"                   Fecha: {game.get('scheduled', 'N/A')}")
                    
                    # Probar player props con este evento
                    if game.get('id'):
                        try:
                            props_test = sportradar.get_player_props(game['id'])
                            print(f"                   Props disponibles: {props_test.get('success', False)}")
                        except Exception as e:
                            print(f"                   Error props: {e}")
                    print()
            else:
                print(f"        ❌ Error obteniendo schedule: {schedule_today.get('error', 'Unknown')}")
        except Exception as e:
            print(f"        ❌ Error general en schedule: {e}")
        
        # 7. Verificar límites de API
        print("\n7. Verificando límites de API...")
        try:
            cache_stats = sportradar.get_cache_stats()
            print(f"   📊 Estadísticas:")
            print(f"      - Requests realizados: {cache_stats.get('requests_made', 0)}")
            print(f"      - Cache hits: {cache_stats.get('cache_hits', 0)}")
            print(f"      - Cache misses: {cache_stats.get('cache_misses', 0)}")
        except Exception as e:
            print(f"   ⚠️  No se pudieron obtener estadísticas: {e}")
        
        print("\n✅ DEBUG COMPLETADO")
        
    except Exception as e:
        print(f"\n❌ ERROR GENERAL EN DEBUG: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_sportradar_api() 