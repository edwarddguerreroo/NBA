#!/usr/bin/env python3
"""
Test de Integración: Predicciones + Bookmakers
==============================================

Script de prueba para demostrar la integración completa entre:
1. Predicciones de modelos entrenados (PTS, AST, TRB, 3P, DD, etc.)
2. Análisis de odds de bookmakers (Sportradar API)
3. Identificación de value bets con ventaja estadística
4. Recomendaciones de apuestas optimizadas

Este script demuestra el flujo completo del sistema sin necesidad de input manual.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.bookmakers.predictions_integration import PredictionsBookmakersIntegration
from datetime import datetime, timedelta
import json

def test_system_status():
    """
    Prueba el estado del sistema completo.
    """
    print("🔍 1. VERIFICANDO ESTADO DEL SISTEMA")
    print("=" * 50)
    
    try:
        # Inicializar integración
        integration = PredictionsBookmakersIntegration()
        
        # Obtener estado del sistema
        status = integration.get_system_status()
        
        print(f"📊 Estado del Sistema:")
        print(f"   • Timestamp: {status['timestamp']}")
        print()
        
        # Modelos disponibles
        models = status['models']
        print(f"🤖 Modelos:")
        print(f"   • Disponibles: {models['total_available']}/{len(models['available'])}")
        print(f"   • Cargados: {models['total_loaded']}")
        print()
        
        for target, available in models['available'].items():
            status_emoji = "✅" if available else "❌"
            loaded_emoji = "🟢" if target in models['loaded'] else "🔴"
            print(f"   {status_emoji} {target}: Disponible | {loaded_emoji} Cargado")
        
        print()
        
        # Bookmakers
        bookmakers = status['bookmakers']
        print(f"🏪 Bookmakers:")
        for provider, provider_status in bookmakers.items():
            configured = "✅" if provider_status.get('configured', False) else "❌"
            accessible = "✅" if provider_status.get('accessible', False) else "❌"
            print(f"   • {provider}: Configurado {configured} | Accesible {accessible}")
        
        print()
        
        # Data Loader
        data_loader = status['data_loader']
        print(f"📁 Data Loader:")
        print(f"   • Status: {data_loader['status']}")
        print(f"   • Paths configurados: {len(data_loader['paths'])}")
        
        print()
        print("✅ VERIFICACIÓN DE ESTADO COMPLETADA")
        return True
        
    except Exception as e:
        print(f"❌ ERROR en verificación de estado: {e}")
        return False

def test_predictions_generation():
    """
    Prueba la generación de predicciones.
    """
    print("🎯 2. GENERANDO PREDICCIONES DE MODELOS")
    print("=" * 50)
    
    try:
        # Inicializar integración
        integration = PredictionsBookmakersIntegration()
        
        # Generar predicciones para mañana
        tomorrow = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')
        print(f"📅 Generando predicciones para: {tomorrow}")
        print()
        
        # Generar predicciones (limitadas para prueba)
        predictions = integration.predictor.generate_predictions_for_date(
            date=tomorrow,
            players=None  # Todos los jugadores disponibles
        )
        
        if 'error' in predictions:
            print(f"❌ Error en predicciones: {predictions['error']}")
            return False
        
        # Mostrar resumen
        summary = predictions['summary']
        print(f"📊 Resumen de Predicciones:")
        print(f"   • Fecha: {predictions['date']}")
        print(f"   • Jugadores analizados: {summary['total_players']}")
        print(f"   • Modelos utilizados: {len(summary['models_used'])}")
        print()
        
        # Mostrar predicciones por target
        print(f"🎯 Predicciones por Target:")
        for target, count in summary['predictions_by_target'].items():
            print(f"   • {target}: {count} predicciones")
        
        print()
        
        # Mostrar ejemplos de predicciones
        print(f"📋 Ejemplos de Predicciones:")
        count = 0
        for target, pred_data in predictions['predictions'].items():
            if 'error' in pred_data or count >= 2:
                continue
            
            # Mostrar top 3 predicciones de este target
            top_predictions = pred_data['predictions'][:3]
            print(f"   🏀 {target}:")
            
            for pred in top_predictions:
                player = pred['player']
                team = pred['team']
                value = pred['predicted_value']
                confidence = pred['confidence']
                
                print(f"      • {player} ({team}): {value:.1f} (conf: {confidence:.1%})")
            
            print()
            count += 1
        
        print("✅ GENERACIÓN DE PREDICCIONES COMPLETADA")
        return True
        
    except Exception as e:
        print(f"❌ ERROR en generación de predicciones: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_market_analysis():
    """
    Prueba el análisis completo predicciones vs mercado.
    """
    print("📈 3. ANÁLISIS PREDICCIONES VS MERCADO")
    print("=" * 50)
    
    try:
        # Inicializar integración
        integration = PredictionsBookmakersIntegration()
        
        # Ejecutar análisis completo
        tomorrow = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')
        print(f"🔍 Analizando mercado para: {tomorrow}")
        print()
        
        analysis = integration.analyze_predictions_vs_market(
            date=tomorrow,
            players=None  # Todos los jugadores
        )
        
        if 'error' in analysis:
            print(f"❌ Error en análisis: {analysis['error']}")
            return False
        
        # Mostrar resumen del análisis
        stats = analysis['analysis_stats']
        print(f"📊 Estadísticas del Análisis:")
        print(f"   • Comparaciones realizadas: {stats['total_comparisons']}")
        print(f"   • Value bets encontrados: {stats['value_bets_found']}")
        print(f"   • Alta confianza: {stats['high_confidence_bets']}")
        print()
        
        # Mostrar recomendaciones
        recommendations = analysis['recommendations']
        print(f"💡 Recomendaciones:")
        print(f"   • {recommendations['message']}")
        print()
        
        if recommendations.get('top_recommendations'):
            print(f"🏆 Top Recomendaciones:")
            for rec in recommendations['top_recommendations'][:3]:
                player = rec['player']
                target = rec['target']
                bet_type = rec['bet_type']
                line = rec['line']
                edge = rec['edge']
                confidence = rec['confidence']
                risk = rec['risk_level']
                
                print(f"   {rec['rank']}. {player} - {target} {bet_type} {line}")
                print(f"      Edge: {edge} | Confianza: {confidence} | Riesgo: {risk}")
                print(f"      {rec['reasoning']}")
                print()
        
        # Portfolio advice
        portfolio = recommendations.get('portfolio_advice', {})
        if portfolio:
            print(f"💼 Consejos de Portfolio:")
            print(f"   • Asignación total recomendada: {portfolio.get('recommended_allocation', 'N/A')}")
            print(f"   • Diversificación: {portfolio.get('diversification_score', 0)} jugadores")
            
            risk_dist = portfolio.get('risk_distribution', {})
            print(f"   • Distribución de riesgo:")
            print(f"     - Bajo: {risk_dist.get('low_risk', 0)}")
            print(f"     - Medio: {risk_dist.get('medium_risk', 0)}")
            print(f"     - Alto: {risk_dist.get('high_risk', 0)}")
            print()
        
        print("✅ ANÁLISIS DE MERCADO COMPLETADO")
        return True
        
    except Exception as e:
        print(f"❌ ERROR en análisis de mercado: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_report_export():
    """
    Prueba la exportación de reportes.
    """
    print("📄 4. EXPORTACIÓN DE REPORTES")
    print("=" * 50)
    
    try:
        # Inicializar integración
        integration = PredictionsBookmakersIntegration()
        
        # Ejecutar análisis primero
        tomorrow = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')
        analysis = integration.analyze_predictions_vs_market(date=tomorrow)
        
        if 'error' in analysis:
            print(f"⚠️  Saltando exportación - Error en análisis: {analysis['error']}")
            return False
        
        # Exportar reporte
        report_path = integration.export_analysis_report()
        
        print(f"📁 Reporte exportado: {report_path}")
        
        # Verificar que el archivo existe
        if os.path.exists(report_path):
            file_size = os.path.getsize(report_path)
            print(f"   • Tamaño: {file_size} bytes")
            print(f"   • Formato: JSON")
            
            # Mostrar estructura del reporte
            with open(report_path, 'r', encoding='utf-8') as f:
                report_data = json.load(f)
            
            print(f"   • Secciones: {list(report_data.keys())}")
            print()
            
            print("✅ EXPORTACIÓN DE REPORTE COMPLETADA")
            return True
        else:
            print(f"❌ Archivo de reporte no encontrado: {report_path}")
            return False
        
    except Exception as e:
        print(f"❌ ERROR en exportación de reporte: {e}")
        return False

def main():
    """
    Función principal del test.
    """
    print("🏀 TEST DE INTEGRACIÓN PREDICCIONES + BOOKMAKERS")
    print("=" * 60)
    print(f"📅 Iniciado: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Ejecutar todas las pruebas
    tests = [
        ("Estado del Sistema", test_system_status),
        ("Generación de Predicciones", test_predictions_generation),
        ("Análisis de Mercado", test_market_analysis),
        ("Exportación de Reportes", test_report_export)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"🔄 Ejecutando: {test_name}")
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"❌ ERROR CRÍTICO en {test_name}: {e}")
            results.append((test_name, False))
        
        print()
        print("-" * 60)
        print()
    
    # Resumen final
    print("📋 RESUMEN DE PRUEBAS")
    print("=" * 30)
    
    passed = 0
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {test_name}")
        if success:
            passed += 1
    
    print()
    print(f"📊 Resultado: {passed}/{len(results)} pruebas exitosas")
    
    if passed == len(results):
        print("🎉 TODAS LAS PRUEBAS PASARON - SISTEMA LISTO")
        print()
        print("💡 El sistema está listo para:")
        print("   • Generar predicciones automáticamente")
        print("   • Obtener odds en tiempo real")
        print("   • Identificar value bets")
        print("   • Generar recomendaciones optimizadas")
        print("   • Exportar reportes completos")
    else:
        print("⚠️  ALGUNAS PRUEBAS FALLARON - REVISAR CONFIGURACIÓN")
        print()
        print("🔧 Posibles soluciones:")
        print("   • Verificar que los modelos estén entrenados")
        print("   • Comprobar configuración de Sportradar API")
        print("   • Validar rutas de datos")
        print("   • Revisar logs para errores específicos")
    
    return passed == len(results)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 