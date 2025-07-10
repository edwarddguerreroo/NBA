"""
Script de Debug para Sportradar API
===================================

Script completo para probar y depurar la implementación de sportradar_api.py
Incluye tests de conectividad, autenticación, endpoints y manejo de errores.
"""

import sys
import os
import json
import traceback
from datetime import datetime, timedelta
from typing import Dict, Any, List

# Agregar el directorio raíz al path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from utils.bookmakers.sportradar_api import SportradarAPI
from utils.bookmakers.exceptions import SportradarAPIError, RateLimitError, AuthenticationError, NetworkError
from utils.bookmakers.config import get_config

class SportradarDebugger:
    """
    Debugger completo para la API de Sportradar
    """
    
    def __init__(self):
        self.api = None
        self.test_results = {}
        self.errors_found = []
        
    def setup_api(self, api_key: str = None) -> bool:
        """
        Configura la API con la clave proporcionada o desde config
        """
        print("=" * 60)
        print("CONFIGURANDO API DE SPORTRADAR")
        print("=" * 60)
        
        try:
            if api_key:
                print(f"Usando API key proporcionada: {api_key[:10]}...")
                self.api = SportradarAPI(api_key=api_key)
            else:
                print("Intentando cargar API key desde configuración...")
                config = get_config()
                configured_key = config.get('sportradar', 'api_key')
                
                if not configured_key:
                    print("❌ ERROR: No se encontró API key en configuración")
                    print("Por favor, configura tu API key en utils/bookmakers/config.py")
                    return False
                
                print(f"API key encontrada en config: {configured_key[:10]}...")
                self.api = SportradarAPI()
            
            print("✅ API configurada correctamente")
            return True
            
        except Exception as e:
            print(f"❌ ERROR configurando API: {e}")
            self.errors_found.append(f"Setup error: {e}")
            return False
    
    def test_basic_connectivity(self) -> bool:
        """
        Test básico de conectividad
        """
        print("\n" + "=" * 60)
        print("TEST 1: CONECTIVIDAD BÁSICA")
        print("=" * 60)
        
        try:
            result = self.api.test_connection()
            
            if result['success']:
                print("✅ Conexión exitosa con Sportradar API")
                print(f"   Tiempo de respuesta: {result['response_time']:.2f}s")
                print(f"   Equipos encontrados: {result.get('teams_count', 'N/A')}")
                self.test_results['connectivity'] = True
                return True
            else:
                print("❌ Fallo en la conexión")
                print(f"   Error: {result.get('error', 'Unknown')}")
                self.test_results['connectivity'] = False
                self.errors_found.append(f"Connectivity error: {result.get('error')}")
                return False
                
        except Exception as e:
            print(f"❌ Excepción en test de conectividad: {e}")
            print(f"   Traceback: {traceback.format_exc()}")
            self.test_results['connectivity'] = False
            self.errors_found.append(f"Connectivity exception: {e}")
            return False
    
    def test_schedule_endpoints(self) -> bool:
        """
        Test de endpoints de schedule
        """
        print("\n" + "=" * 60)
        print("TEST 2: ENDPOINTS DE SCHEDULE")
        print("=" * 60)
        
        success = True
        
        # Test 1: Schedule de hoy
        try:
            today = datetime.now().strftime("%Y-%m-%d")
            print(f"Probando schedule para {today}...")
            
            schedule = self.api.get_schedule(today)
            
            if 'games' in schedule:
                games_count = len(schedule['games'])
                print(f"✅ Schedule obtenido: {games_count} juegos")
                self.test_results['schedule_today'] = True
            else:
                print("⚠️ Schedule obtenido pero sin juegos")
                print(f"   Keys disponibles: {list(schedule.keys())}")
                self.test_results['schedule_today'] = True  # No es error si no hay juegos
                
        except Exception as e:
            print(f"❌ Error en schedule de hoy: {e}")
            self.test_results['schedule_today'] = False
            self.errors_found.append(f"Schedule today error: {e}")
            success = False
        
        # Test 2: Schedule de odds
        try:
            print(f"Probando odds schedule para {today}...")
            
            odds_schedule = self.api.get_daily_odds_schedule(today)
            
            if 'sport_events' in odds_schedule:
                events_count = len(odds_schedule['sport_events'])
                print(f"✅ Odds schedule obtenido: {events_count} eventos")
                self.test_results['odds_schedule'] = True
            else:
                print("⚠️ Odds schedule obtenido pero sin eventos")
                print(f"   Keys disponibles: {list(odds_schedule.keys())}")
                self.test_results['odds_schedule'] = True
                
        except Exception as e:
            print(f"❌ Error en odds schedule: {e}")
            self.test_results['odds_schedule'] = False
            self.errors_found.append(f"Odds schedule error: {e}")
            success = False
        
        return success
    
    def test_odds_endpoints(self) -> bool:
        """
        Test de endpoints de odds
        """
        print("\n" + "=" * 60)
        print("TEST 3: ENDPOINTS DE ODDS")
        print("=" * 60)
        
        success = True
        
        # Primero obtener un sport_event_id de prueba
        test_sport_event_id = None
        
        try:
            today = datetime.now().strftime("%Y-%m-%d")
            schedule = self.api.get_daily_odds_schedule(today)
            
            if 'sport_events' in schedule and len(schedule['sport_events']) > 0:
                test_sport_event_id = schedule['sport_events'][0]['id']
                print(f"Usando sport_event_id de prueba: {test_sport_event_id}")
            else:
                # Usar un ID de ejemplo si no hay juegos hoy
                test_sport_event_id = "sr:sport_event:12345"
                print(f"No hay juegos hoy, usando ID de ejemplo: {test_sport_event_id}")
                
        except Exception as e:
            print(f"⚠️ No se pudo obtener sport_event_id real: {e}")
            test_sport_event_id = "sr:sport_event:12345"
            print(f"Usando ID de ejemplo: {test_sport_event_id}")
        
        # Test 1: Odds principales
        try:
            print("Probando endpoint de odds principales...")
            
            odds = self.api.get_odds(test_sport_event_id)
            
            if isinstance(odds, dict):
                print("✅ Odds obtenidas correctamente")
                print(f"   Keys disponibles: {list(odds.keys())}")
                
                if 'markets' in odds:
                    markets_count = len(odds['markets'])
                    print(f"   Mercados encontrados: {markets_count}")
                
                self.test_results['odds_main'] = True
            else:
                print(f"⚠️ Respuesta inesperada: {type(odds)}")
                self.test_results['odds_main'] = False
                
        except Exception as e:
            print(f"❌ Error en odds principales: {e}")
            print(f"   Tipo de error: {type(e).__name__}")
            self.test_results['odds_main'] = False
            self.errors_found.append(f"Main odds error: {e}")
            success = False
        
        # Test 2: Player props
        try:
            print("Probando endpoint de player props...")
            
            props = self.api.get_player_props(test_sport_event_id)
            
            if isinstance(props, dict):
                print("✅ Player props obtenidas correctamente")
                print(f"   Keys disponibles: {list(props.keys())}")
                
                if 'markets' in props:
                    markets_count = len(props['markets'])
                    print(f"   Mercados de props encontrados: {markets_count}")
                
                self.test_results['player_props'] = True
            else:
                print(f"⚠️ Respuesta inesperada: {type(props)}")
                self.test_results['player_props'] = False
                
        except Exception as e:
            print(f"❌ Error en player props: {e}")
            print(f"   Tipo de error: {type(e).__name__}")
            self.test_results['player_props'] = False
            self.errors_found.append(f"Player props error: {e}")
            success = False
        
        return success
    
    def test_specific_targets(self) -> bool:
        """
        Test de métodos específicos para targets del modelo
        """
        print("\n" + "=" * 60)
        print("TEST 4: MÉTODOS ESPECÍFICOS PARA TARGETS")
        print("=" * 60)
        
        success = True
        
        # Test 1: get_specific_nba_odds
        try:
            print("Probando get_specific_nba_odds...")
            
            # Obtener un sport_event_id
            today = datetime.now().strftime("%Y-%m-%d")
            schedule = self.api.get_daily_odds_schedule(today)
            
            if 'sport_events' in schedule and len(schedule['sport_events']) > 0:
                test_event_id = schedule['sport_events'][0]['id']
            else:
                test_event_id = "sr:sport_event:12345"
            
            targets = ['moneyline', 'totals', 'player_points']
            specific_odds = self.api.get_specific_nba_odds(test_event_id, targets)
            
            if isinstance(specific_odds, dict) and 'targets' in specific_odds:
                print("✅ Odds específicas obtenidas")
                print(f"   Targets encontrados: {list(specific_odds['targets'].keys())}")
                self.test_results['specific_odds'] = True
            else:
                print(f"⚠️ Estructura inesperada: {list(specific_odds.keys()) if isinstance(specific_odds, dict) else type(specific_odds)}")
                self.test_results['specific_odds'] = False
                
        except Exception as e:
            print(f"❌ Error en odds específicas: {e}")
            self.test_results['specific_odds'] = False
            self.errors_found.append(f"Specific odds error: {e}")
            success = False
        
        # Test 2: get_nba_odds_for_targets
        try:
            print("Probando get_nba_odds_for_targets...")
            
            targets = ['moneyline', 'totals']
            odds_for_targets = self.api.get_nba_odds_for_targets(targets=targets)
            
            if isinstance(odds_for_targets, dict) and 'games' in odds_for_targets:
                games_count = len(odds_for_targets['games'])
                print(f"✅ Odds para targets obtenidas: {games_count} juegos")
                print(f"   Summary: {odds_for_targets.get('summary', {})}")
                self.test_results['odds_for_targets'] = True
            else:
                print(f"⚠️ Estructura inesperada en odds_for_targets")
                self.test_results['odds_for_targets'] = False
                
        except Exception as e:
            print(f"❌ Error en odds_for_targets: {e}")
            self.test_results['odds_for_targets'] = False
            self.errors_found.append(f"Odds for targets error: {e}")
            success = False
        
        return success
    
    def test_error_handling(self) -> bool:
        """
        Test de manejo de errores
        """
        print("\n" + "=" * 60)
        print("TEST 5: MANEJO DE ERRORES")
        print("=" * 60)
        
        success = True
        
        # Test 1: Sport event ID inválido
        try:
            print("Probando con sport_event_id inválido...")
            
            invalid_id = "invalid_sport_event_id"
            odds = self.api.get_odds(invalid_id)
            
            print("⚠️ No se generó error con ID inválido (puede ser normal)")
            
        except SportradarAPIError as e:
            print(f"✅ Error manejado correctamente: {e}")
            self.test_results['error_handling'] = True
        except Exception as e:
            print(f"❌ Error no manejado correctamente: {type(e).__name__}: {e}")
            self.errors_found.append(f"Error handling failed: {e}")
            success = False
        
        # Test 2: Rate limiting (simulado)
        try:
            print("Verificando configuración de rate limiting...")
            
            rate_limit_calls = self.api.rate_limit_calls
            rate_limit_period = self.api.rate_limit_period
            
            print(f"✅ Rate limiting configurado: {rate_limit_calls} llamadas / {rate_limit_period}s")
            self.test_results['rate_limiting'] = True
            
        except Exception as e:
            print(f"❌ Error en configuración de rate limiting: {e}")
            self.errors_found.append(f"Rate limiting error: {e}")
            success = False
        
        return success
    
    def test_cache_functionality(self) -> bool:
        """
        Test de funcionalidad de cache
        """
        print("\n" + "=" * 60)
        print("TEST 6: FUNCIONALIDAD DE CACHE")
        print("=" * 60)
        
        try:
            # Verificar cache stats
            cache_stats = self.api.get_cache_stats()
            print(f"✅ Cache stats obtenidas: {cache_stats}")
            
            # Test de limpieza de cache
            self.api.clear_cache()
            cache_stats_after = self.api.get_cache_stats()
            print(f"✅ Cache limpiada: {cache_stats_after}")
            
            self.test_results['cache'] = True
            return True
            
        except Exception as e:
            print(f"❌ Error en funcionalidad de cache: {e}")
            self.test_results['cache'] = False
            self.errors_found.append(f"Cache error: {e}")
            return False
    
    def generate_debug_report(self) -> Dict[str, Any]:
        """
        Genera un reporte completo de debug
        """
        print("\n" + "=" * 60)
        print("REPORTE DE DEBUG")
        print("=" * 60)
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results.values() if result)
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'total_tests': total_tests,
                'passed_tests': passed_tests,
                'failed_tests': total_tests - passed_tests,
                'success_rate': f"{(passed_tests/total_tests*100):.1f}%" if total_tests > 0 else "0%"
            },
            'test_results': self.test_results,
            'errors_found': self.errors_found,
            'recommendations': self._generate_recommendations()
        }
        
        print(f"Tests ejecutados: {total_tests}")
        print(f"Tests exitosos: {passed_tests}")
        print(f"Tests fallidos: {total_tests - passed_tests}")
        print(f"Tasa de éxito: {report['summary']['success_rate']}")
        
        if self.errors_found:
            print(f"\nErrores encontrados ({len(self.errors_found)}):")
            for i, error in enumerate(self.errors_found, 1):
                print(f"  {i}. {error}")
        
        if report['recommendations']:
            print(f"\nRecomendaciones ({len(report['recommendations'])}):")
            for i, rec in enumerate(report['recommendations'], 1):
                print(f"  {i}. {rec}")
        
        return report
    
    def _generate_recommendations(self) -> List[str]:
        """
        Genera recomendaciones basadas en los errores encontrados
        """
        recommendations = []
        
        if not self.test_results.get('connectivity', True):
            recommendations.append("Verificar conectividad a internet y configuración de red")
            recommendations.append("Verificar que la API key de Sportradar sea válida")
        
        if not self.test_results.get('schedule_today', True):
            recommendations.append("Verificar formato de URLs en configuración")
            recommendations.append("Revisar endpoints de Sports API en documentación")
        
        if not self.test_results.get('odds_schedule', True):
            recommendations.append("Verificar configuración de Odds API URLs")
            recommendations.append("Confirmar acceso a Odds Comparison API")
        
        if not self.test_results.get('odds_main', True):
            recommendations.append("Revisar formato de sport_event_id")
            recommendations.append("Verificar endpoints de Odds API")
        
        if not self.test_results.get('player_props', True):
            recommendations.append("Verificar acceso a Player Props API")
            recommendations.append("Revisar configuración de player_props_url")
        
        if len(self.errors_found) > 3:
            recommendations.append("Considerar revisar la documentación oficial de Sportradar")
            recommendations.append("Verificar que la suscripción incluya todos los endpoints necesarios")
        
        return recommendations
    
    def save_debug_report(self, report: Dict[str, Any], filename: str = None) -> str:
        """
        Guarda el reporte de debug en un archivo JSON
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"sportradar_debug_report_{timestamp}.json"
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            
            print(f"\n✅ Reporte guardado en: {filename}")
            return filename
            
        except Exception as e:
            print(f"\n❌ Error guardando reporte: {e}")
            return ""

def main():
    """
    Función principal del debugger
    """
    print("SPORTRADAR API DEBUGGER")
    print("=" * 60)
    print("Este script probará todos los aspectos de la implementación de Sportradar API")
    print("=" * 60)
    
    debugger = SportradarDebugger()
    
    # Usar configuración automáticamente desde .env
    print("\nUsando configuración desde .env...")
    
    # Ejecutar tests
    if not debugger.setup_api(None):
        print("\n❌ No se pudo configurar la API. Abortando tests.")
        return
    
    # Ejecutar todos los tests
    tests = [
        debugger.test_basic_connectivity,
        debugger.test_schedule_endpoints,
        debugger.test_odds_endpoints,
        debugger.test_specific_targets,
        debugger.test_error_handling,
        debugger.test_cache_functionality
    ]
    
    for test in tests:
        try:
            test()
        except KeyboardInterrupt:
            print("\n\n⚠️ Debug interrumpido por el usuario")
            break
        except Exception as e:
            print(f"\n❌ Error inesperado en test: {e}")
            debugger.errors_found.append(f"Unexpected test error: {e}")
    
    # Generar reporte final
    report = debugger.generate_debug_report()
    
    # Guardar reporte
    save_report = input("\n¿Guardar reporte en archivo JSON? (y/n): ").strip().lower()
    if save_report == 'y':
        debugger.save_debug_report(report)
    
    print("\n" + "=" * 60)
    print("DEBUG COMPLETADO")
    print("=" * 60)

if __name__ == "__main__":
    main() 