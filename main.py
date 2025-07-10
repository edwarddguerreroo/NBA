"""
Sistema de Predicciones NBA - Punto de Entrada Principal
======================================================

Sistema avanzado de machine learning para predicción de estadísticas NBA
con integración automatizada de APIs de casas de apuestas.

Características:
- Predicciones automatizadas de PTS, AST, TRB, 3P, double_double
- Integración con Sportradar para odds en tiempo real
- Procesamiento automatizado sin input manual
- Arquitectura modular y escalable
"""

import logging
import sys
from datetime import datetime
from typing import Dict, Any

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/main.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)


def test_automated_odds_integration():
    """
    Prueba la integración automatizada de odds sin necesidad de especificar
    event_id ni date manualmente.
    """
    print("=== PRUEBA DE INTEGRACIÓN AUTOMATIZADA ===")
    print("Sin especificar event_id ni date - TODO AUTOMÁTICO")
    print()
    
    try:
        # Importar la integración automatizada
        from utils.bookmakers.bookmakers_integration import AutomatedBookmakersIntegration
        
        # Inicializar
        print("1. Inicializando integración automatizada...")
        integration = AutomatedBookmakersIntegration()
        print("   ✓ Integración lista")
        print()
        
        # Escaneo rápido
        print("2. Escaneo rápido del mercado...")
        quick_scan = integration.quick_market_scan(game_limit=3)
        
        if 'error' not in quick_scan:
            print(f"   ✓ Partidos verificados: {quick_scan['games_checked']}")
            print(f"   ✓ Con odds disponibles: {quick_scan['summary']['total_with_odds']}")
        else:
            print(f"   ⚠ Error en escaneo: {quick_scan['error']}")
        print()
        
        # Datos para predicciones automáticos
        print("3. Obteniendo datos automáticos para predicciones...")
        predictions_data = integration.get_ready_to_predict_data(max_games=5, days_ahead=3)
        
        if 'error' not in predictions_data:
            auto_info = predictions_data.get('automation_info', {})
            print(f"   ✓ Partidos encontrados: {auto_info.get('games_found', 0)}")
            print(f"   ✓ Con odds disponibles: {auto_info.get('games_with_odds', 0)}")
            print(f"   ✓ Método: {auto_info.get('method', 'N/A')}")
            print(f"   ✓ Automático: {not auto_info.get('requires_manual_input', True)}")
        else:
            print(f"   ⚠ Error: {predictions_data['error']}")
        print()
        
        # Mostrar ejemplos de partidos
        if 'games' in predictions_data and predictions_data['games']:
            print("4. Ejemplos de partidos encontrados automáticamente:")
            count = 0
            for game_id, game_data in predictions_data['games'].items():
                if count >= 2:
                    break
                    
                teams = game_data.get('teams', {})
                date = game_data.get('date', 'N/A')
                has_odds = game_data.get('has_odds', False)
                
                print(f"   • {teams.get('away', 'TBD')} @ {teams.get('home', 'TBD')}")
                print(f"     📅 Fecha: {date}")
                print(f"     🎯 Odds: {'Disponibles' if has_odds else 'No disponibles'}")
                print(f"     🆔 ID: {game_id}")
                print()
                count += 1
        
        print("✅ PROCESO AUTOMATIZADO COMPLETADO")
        print("   Todos los datos se obtuvieron sin input manual")
        return True
        
    except ImportError as e:
        print(f"❌ Error de importación: {e}")
        print("   Verifica que las dependencias estén instaladas")
        return False
    except Exception as e:
        print(f"❌ Error en prueba automatizada: {e}")
        return False


def test_sportradar_direct():
    """
    Prueba directa de la API de Sportradar con métodos automatizados.
    """
    print("=== PRUEBA DIRECTA API SPORTRADAR ===")
    print("Métodos automatizados sin event_id/date manual")
    print()
    
    try:
        from utils.bookmakers.sportradar_api import SportradarAPI
        
        # Inicializar API
        print("1. Inicializando API Sportradar...")
        api = SportradarAPI()
        print("   ✓ API lista")
        print()
        
        # Próximos partidos automático
        print("2. Obteniendo próximos partidos automáticamente...")
        next_games = api.get_next_nba_games(days_ahead=3, max_games=5)
        
        if 'error' not in next_games:
            print(f"   ✓ Partidos encontrados: {next_games['total_found']}")
            print(f"   ✓ Periodo: {next_games['search_period']['start_date']} a {next_games['search_period']['end_date']}")
        else:
            print(f"   ⚠ Error: {next_games['error']}")
        print()
        
        # Odds automáticas
        print("3. Obteniendo odds automáticamente...")
        auto_odds = api.get_odds_for_next_games(days_ahead=2, max_games=3)
        
        if 'error' not in auto_odds:
            summary = auto_odds.get('summary', {})
            print(f"   ✓ Partidos procesados: {summary.get('games_processed', 0)}")
            print(f"   ✓ Con odds: {summary.get('games_with_odds', 0)}")
        else:
            print(f"   ⚠ Error: {auto_odds['error']}")
        print()
        
        # Verificación rápida
        print("4. Verificación rápida...")
        quick_check = api.quick_odds_check(game_limit=2)
        
        if 'error' not in quick_check:
            print(f"   ✓ Partidos verificados: {quick_check['games_checked']}")
            print(f"   ✓ Odds disponibles: {quick_check['summary']['has_live_odds']}")
        else:
            print(f"   ⚠ Error: {quick_check['error']}")
        print()
        
        print("✅ PRUEBA DIRECTA COMPLETADA")
        return True
        
    except Exception as e:
        print(f"❌ Error en prueba directa: {e}")
        return False


def main():
    """
    Función principal del sistema.
    """
    print("🏀 SISTEMA DE PREDICCIONES NBA")
    print("===============================")
    print(f"📅 Iniciado: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Ejecutar pruebas automatizadas
    print("🔄 EJECUTANDO PRUEBAS AUTOMATIZADAS")
    print("Verificando integración de odds sin input manual...")
    print()
    
    # Prueba de integración automatizada
    integration_success = test_automated_odds_integration()
    print()
    print("-" * 50)
    print()
    
    # Prueba directa de API
    api_success = test_sportradar_direct()
    print()
    print("-" * 50)
    print()
    
    # Resumen final
    print("📋 RESUMEN DE PRUEBAS")
    print(f"   Integración automatizada: {'✅ OK' if integration_success else '❌ FALLO'}")
    print(f"   API directa automatizada: {'✅ OK' if api_success else '❌ FALLO'}")
    print()
    
    if integration_success and api_success:
        print("🎉 SISTEMA LISTO PARA USAR")
        print("   Puedes obtener odds automáticamente sin especificar event_id ni date")
        print()
        print("📖 MÉTODOS DISPONIBLES:")
        print("   • integration.get_ready_to_predict_data() - Datos listos para predicciones")
        print("   • integration.quick_market_scan() - Escaneo rápido del mercado")
        print("   • api.get_automated_predictions_data() - Proceso completamente automatizado")
        print("   • api.get_next_nba_games() - Próximos partidos automáticamente")
        print("   • api.get_odds_for_next_games() - Odds automáticas")
        print("   • api.quick_odds_check() - Verificación rápida")
    else:
        print("⚠️  ALGUNOS COMPONENTES REQUIEREN ATENCIÓN")
        print("   Revisa los logs para más detalles")
    
    print()
    print("🏁 PROCESO COMPLETADO")


if __name__ == "__main__":
    main() 