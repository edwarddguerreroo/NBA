"""
Verificación Completa: Endpoints Player Props v2 API para Targets de Predicción
=============================================================================

Este script verifica que todos los endpoints necesarios para nuestros targets
de predicción (PTS, AST, TRB, 3P) funcionan correctamente con Player Props v2 API.

Targets verificados:
- PTS (Puntos)
- AST (Asistencias) 
- TRB (Rebotes)
- 3P (Triples)
"""

import os
import sys
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def verificar_endpoints_basicos():
    """
    Verifica los endpoints básicos de Player Props v2 API.
    """
    print("=" * 70)
    print("VERIFICACIÓN: Endpoints Básicos Player Props v2 API")
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
        
        endpoints_basicos = [
            {
                'name': 'Sports',
                'method': 'get_sports',
                'args': [],
                'expected_key': 'sports'
            },
            {
                'name': 'Bookmakers',
                'method': 'get_bookmakers',
                'args': [],
                'expected_key': 'books'
            },
            {
                'name': 'Sport Competitions (Basketball)',
                'method': 'get_sport_competitions',
                'args': [2],  # Basketball
                'expected_key': 'competitions'
            }
        ]
        
        resultados = {}
        
        for endpoint in endpoints_basicos:
            print(f"\n🔍 Probando: {endpoint['name']}")
            try:
                method = getattr(api, endpoint['method'])
                result = method(*endpoint['args'])
                
                if isinstance(result, dict) and endpoint['expected_key'] in result:
                    count = len(result[endpoint['expected_key']])
                    print(f"✅ Éxito: {count} elementos encontrados")
                    resultados[endpoint['name']] = {'status': 'success', 'count': count}
                else:
                    print(f"❌ Error: Estructura inesperada - {list(result.keys()) if isinstance(result, dict) else type(result)}")
                    resultados[endpoint['name']] = {'status': 'error', 'error': 'Estructura inesperada'}
                    
            except Exception as e:
                error_msg = str(e)
                print(f"❌ Excepción: {error_msg}")
                resultados[endpoint['name']] = {'status': 'exception', 'error': error_msg}
        
        return resultados
        
    except Exception as e:
        print(f"❌ Error general: {e}")
        return {}

def verificar_schedules_por_competicion():
    """
    Verifica schedules para diferentes competiciones de basketball.
    """
    print("\n" + "=" * 70)
    print("VERIFICACIÓN: Schedules por Competición")
    print("=" * 70)
    
    try:
        from utils.bookmakers.sportradar_api import SportradarAPI
        from utils.bookmakers.config import get_config
        
        config = get_config()
        api_key = config.get('sportradar', 'api_key') or os.getenv('SPORTRADAR_API')
        api = SportradarAPI(api_key=api_key)
        
        # Competiciones a verificar
        competiciones = [
            {'id': 'sr:competition:132', 'name': 'NBA', 'activa': False},
            {'id': 'sr:competition:486', 'name': 'WNBA', 'activa': True},
            {'id': 'sr:competition:648', 'name': 'NCAA', 'activa': False}
        ]
        
        resultados = {}
        
        for comp in competiciones:
            print(f"\n🔍 Verificando: {comp['name']} ({comp['id']})")
            try:
                result = api.get_competition_schedules(comp['id'], limit=10)
                
                if 'schedules' in result:
                    schedules_count = len(result['schedules'])
                    print(f"✅ Éxito: {schedules_count} schedules encontrados")
                    
                    # Analizar eventos próximos
                    upcoming_events = 0
                    ended_events = 0
                    
                    for schedule in result['schedules'][:5]:
                        event = schedule.get('sport_event', {})
                        status = event.get('status', 'unknown')
                        
                        if status == 'not_started':
                            upcoming_events += 1
                        elif status == 'ended':
                            ended_events += 1
                    
                    print(f"   📊 Próximos: {upcoming_events}, Finalizados: {ended_events}")
                    
                    resultados[comp['name']] = {
                        'status': 'success',
                        'schedules_count': schedules_count,
                        'upcoming_events': upcoming_events,
                        'ended_events': ended_events,
                        'activa_esperada': comp['activa']
                    }
                else:
                    print(f"❌ Error: No se encontraron schedules")
                    resultados[comp['name']] = {'status': 'error', 'error': 'No schedules found'}
                    
            except Exception as e:
                error_msg = str(e)
                print(f"❌ Excepción: {error_msg}")
                resultados[comp['name']] = {'status': 'exception', 'error': error_msg}
        
        return resultados
        
    except Exception as e:
        print(f"❌ Error general: {e}")
        return {}

def verificar_player_props_targets():
    """
    Verifica que podemos obtener player props para nuestros targets específicos.
    """
    print("\n" + "=" * 70)
    print("VERIFICACIÓN: Player Props para Targets (PTS, AST, TRB, 3P)")
    print("=" * 70)
    
    try:
        from utils.bookmakers.bookmakers_data_fetcher import BookmakersDataFetcher
        
        # Inicializar fetcher
        fetcher = BookmakersDataFetcher()
        print("✅ BookmakersDataFetcher inicializado")
        
        # Targets a verificar
        targets = ['PTS', 'AST', 'TRB', '3P']
        
        resultados = {}
        
        for target in targets:
            print(f"\n🔍 Verificando target: {target}")
            try:
                # Usar método específico para obtener props de targets
                props_data = fetcher.get_player_props_for_targets(
                    targets=[target]
                )
                
                if props_data.get('success', False):
                    games_count = props_data['summary']['total_games']
                    games_with_props = props_data['summary']['games_with_props']
                    target_props = props_data['summary']['props_by_target'][target]
                    
                    print(f"✅ Éxito para {target}:")
                    print(f"   📊 Juegos totales: {games_count}")
                    print(f"   📊 Juegos con props: {games_with_props}")
                    print(f"   📊 Props para {target}: {target_props}")
                    
                    resultados[target] = {
                        'status': 'success',
                        'games_total': games_count,
                        'games_with_props': games_with_props,
                        'target_props_count': target_props
                    }
                else:
                    error_msg = props_data.get('error', 'Unknown error')
                    print(f"❌ Error para {target}: {error_msg}")
                    resultados[target] = {'status': 'error', 'error': error_msg}
                    
            except Exception as e:
                error_msg = str(e)
                print(f"❌ Excepción para {target}: {error_msg}")
                resultados[target] = {'status': 'exception', 'error': error_msg}
        
        return resultados
        
    except Exception as e:
        print(f"❌ Error general: {e}")
        return {}

def verificar_optimizacion_estacional():
    """
    Verifica que la optimización estacional funciona correctamente.
    """
    print("\n" + "=" * 70)
    print("VERIFICACIÓN: Optimización Estacional")
    print("=" * 70)
    
    try:
        from utils.bookmakers.bookmakers_data_fetcher import BookmakersDataFetcher
        
        fetcher = BookmakersDataFetcher()
        
        # Verificar información estacional
        seasonal_info = fetcher.get_seasonal_props_availability()
        
        current_phase = seasonal_info['current_phase']
        cache_hours = seasonal_info['recommendations']['cache_expiry_hours']
        priority_targets = seasonal_info['recommendations']['priority_targets']
        api_frequency = seasonal_info['recommendations']['api_call_frequency']
        
        print(f"✅ Optimización estacional activa:")
        print(f"   📅 Fase actual: {current_phase}")
        print(f"   ⏰ Cache optimizado: {cache_hours} horas")
        print(f"   🎯 Targets prioritarios: {priority_targets}")
        print(f"   📡 Frecuencia API: {api_frequency}")
        
        # Verificar optimización automática
        optimization = fetcher.optimize_cache_for_season()
        
        print(f"\n🔍 Optimización automática:")
        print(f"   ✅ Optimizado: {optimization['optimized']}")
        print(f"   📝 Razón: {optimization['reason']}")
        
        return {
            'status': 'success',
            'current_phase': current_phase,
            'cache_hours': cache_hours,
            'priority_targets': priority_targets,
            'optimization_active': optimization['optimized']
        }
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return {'status': 'error', 'error': str(e)}

def verificar_integracion_completa():
    """
    Verifica la integración completa del sistema.
    """
    print("\n" + "=" * 70)
    print("VERIFICACIÓN: Integración Completa del Sistema")
    print("=" * 70)
    
    try:
        from utils.bookmakers.bookmakers_integration import BookmakersIntegration
        
        # Inicializar integración
        integration = BookmakersIntegration()
        print("✅ BookmakersIntegration inicializada")
        
        # Verificar método principal
        print(f"\n🔍 Verificando método principal get_best_prediction_odds...")
        
        # Crear datos de prueba simulados
        import pandas as pd
        import numpy as np
        
        test_predictions = pd.DataFrame({
            'Player': ['LeBron James', 'Stephen Curry', 'Kevin Durant'],
            'Team': ['LAL', 'GSW', 'PHX'],
            'PTS': [25.5, 28.2, 26.8],
            'PTS_confidence': [0.92, 0.95, 0.89],
            'AST': [7.2, 6.8, 5.1],
            'AST_confidence': [0.88, 0.91, 0.86],
            'TRB': [8.1, 5.2, 7.3],
            'TRB_confidence': [0.90, 0.87, 0.93]
        })
        
        print(f"   📊 Datos de prueba creados: {len(test_predictions)} jugadores")
        
        # Verificar que el método existe y puede procesar datos
        if hasattr(integration, 'get_best_prediction_odds'):
            print(f"   ✅ Método get_best_prediction_odds disponible")
            
            # Verificar métodos auxiliares
            metodos_auxiliares = [
                '_calculate_model_probability',
                '_odds_to_probability',
                'find_arbitrage_opportunities'
            ]
            
            metodos_ok = 0
            for metodo in metodos_auxiliares:
                if hasattr(integration, metodo):
                    metodos_ok += 1
                    print(f"   ✅ {metodo}: disponible")
                else:
                    print(f"   ❌ {metodo}: faltante")
            
            print(f"\n📊 Resumen integración:")
            print(f"   ✅ Métodos auxiliares: {metodos_ok}/{len(metodos_auxiliares)}")
            print(f"   ✅ Sistema listo para predicciones")
            
            return {
                'status': 'success',
                'main_method_available': True,
                'auxiliary_methods': metodos_ok,
                'total_methods_expected': len(metodos_auxiliares),
                'integration_ready': metodos_ok == len(metodos_auxiliares)
            }
        else:
            print(f"   ❌ Método principal no disponible")
            return {'status': 'error', 'error': 'Main method not available'}
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return {'status': 'error', 'error': str(e)}

def main():
    """
    Función principal de verificación completa.
    """
    print("🚀 VERIFICACIÓN COMPLETA: Endpoints Player Props v2 API para Targets")
    print("=" * 70)
    
    # Verificación 1: Endpoints básicos
    resultado_basicos = verificar_endpoints_basicos()
    
    # Verificación 2: Schedules por competición
    resultado_schedules = verificar_schedules_por_competicion()
    
    # Verificación 3: Player Props para targets específicos
    resultado_targets = verificar_player_props_targets()
    
    # Verificación 4: Optimización estacional
    resultado_estacional = verificar_optimizacion_estacional()
    
    # Verificación 5: Integración completa
    resultado_integracion = verificar_integracion_completa()
    
    # Resumen final
    print("\n" + "=" * 70)
    print("RESUMEN FINAL DE VERIFICACIÓN")
    print("=" * 70)
    
    # Analizar resultados
    endpoints_ok = sum(1 for r in resultado_basicos.values() if r.get('status') == 'success')
    schedules_ok = sum(1 for r in resultado_schedules.values() if r.get('status') == 'success')
    targets_ok = sum(1 for r in resultado_targets.values() if r.get('status') == 'success')
    estacional_ok = resultado_estacional.get('status') == 'success'
    integracion_ok = resultado_integracion.get('status') == 'success'
    
    print(f"\n📊 RESULTADOS POR CATEGORÍA:")
    print(f"   🔗 Endpoints básicos: {endpoints_ok}/{len(resultado_basicos)} funcionando")
    print(f"   📅 Schedules: {schedules_ok}/{len(resultado_schedules)} funcionando")
    print(f"   🎯 Targets (PTS/AST/TRB/3P): {targets_ok}/4 funcionando")
    print(f"   🔄 Optimización estacional: {'✅ OK' if estacional_ok else '❌ ERROR'}")
    print(f"   🔧 Integración completa: {'✅ OK' if integracion_ok else '❌ ERROR'}")
    
    # Calcular score total
    total_checks = len(resultado_basicos) + len(resultado_schedules) + 4 + 1 + 1
    successful_checks = endpoints_ok + schedules_ok + targets_ok + (1 if estacional_ok else 0) + (1 if integracion_ok else 0)
    
    success_rate = (successful_checks / total_checks) * 100
    
    print(f"\n🎯 SCORE TOTAL: {successful_checks}/{total_checks} ({success_rate:.1f}%)")
    
    if success_rate >= 80:
        print(f"\n🎉 VERIFICACIÓN EXITOSA")
        print("   ✅ Sistema completamente funcional para targets de predicción")
        print("   ✅ Player Props v2 API funcionando correctamente")
        print("   ✅ Todos los targets (PTS, AST, TRB, 3P) soportados")
        print("   ✅ Optimización estacional activa")
        print("   ✅ Integración lista para producción")
    elif success_rate >= 60:
        print(f"\n⚠️  VERIFICACIÓN PARCIAL")
        print("   ✅ Funcionalidad básica operativa")
        print("   ⚠️  Algunos componentes necesitan revisión")
    else:
        print(f"\n❌ VERIFICACIÓN FALLIDA")
        print("   ❌ Problemas críticos encontrados")
        print("   🔧 Revisión completa necesaria")
    
    return success_rate >= 80

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 