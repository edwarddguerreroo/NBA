#!/usr/bin/env python3
"""
Prueba Rápida de Orquestación - Módulo Bookmakers
================================================

Script simple para verificar que todos los componentes estén correctamente orquestados.
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime

# Añadir el directorio raíz al path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

def test_imports():
    """Prueba las importaciones básicas."""
    print("🔍 Probando importaciones...")
    
    try:
        from utils.bookmakers import (
            SportradarAPI,
            BookmakersDataFetcher,
            BookmakersIntegration,
            BookmakersConfig
        )
        print("✅ Importaciones exitosas")
        return True
    except ImportError as e:
        print(f"❌ Error en importaciones: {e}")
        return False

def test_configuration():
    """Prueba la configuración."""
    print("\n🔍 Probando configuración...")
    
    try:
        from utils.bookmakers import BookmakersConfig
        
        config = BookmakersConfig()
        
        # Verificar algunos valores clave
        assert config.get('sportradar', 'timeout') == 30
        assert config.get('betting', 'minimum_edge') == 0.04
        
        # Verificar URLs
        base_url = config.get_api_url('basketball')
        assert 'api.sportradar.com' in base_url
        
        print("✅ Configuración correcta")
        return True
    except Exception as e:
        print(f"❌ Error en configuración: {e}")
        return False

def test_sportradar_api():
    """Prueba la API de Sportradar."""
    print("\n🔍 Probando Sportradar API...")
    
    try:
        from utils.bookmakers import SportradarAPI
        
        # Crear instancia con API key de prueba
        api = SportradarAPI(api_key="test_key_12345")
        
        # Verificar que se inicializó correctamente
        assert hasattr(api, 'session')
        assert hasattr(api, '_cache')
        assert api.api_key == "test_key_12345"
        
        # Test de conexión (debería fallar con key falsa, pero no crashear)
        connection_test = api.test_connection()
        assert isinstance(connection_test, dict)
        
        print("✅ Sportradar API inicializada correctamente")
        return True
    except Exception as e:
        print(f"❌ Error en Sportradar API: {e}")
        return False

def test_data_fetcher():
    """Prueba el BookmakersDataFetcher."""
    print("\n🔍 Probando BookmakersDataFetcher...")
    
    try:
        from utils.bookmakers import BookmakersDataFetcher
        
        # Crear instancia
        fetcher = BookmakersDataFetcher(
            api_keys={'sportradar': 'test_key_12345'},
            odds_data_dir='data/bookmakers/test'
        )
        
        # Verificar componentes
        assert hasattr(fetcher, 'sportradar_api')
        assert fetcher.sportradar_api is not None
        
        # Verificar métodos principales
        assert hasattr(fetcher, 'get_nba_odds_from_sportradar')
        assert hasattr(fetcher, 'simulate_bookmaker_data')
        
        # Test de estado
        api_status = fetcher.get_api_status()
        assert isinstance(api_status, dict)
        assert 'sportradar' in api_status
        
        print("✅ BookmakersDataFetcher funcionando")
        return True
    except Exception as e:
        print(f"❌ Error en BookmakersDataFetcher: {e}")
        return False

def test_integration():
    """Prueba el BookmakersIntegration."""
    print("\n🔍 Probando BookmakersIntegration...")
    
    try:
        from utils.bookmakers import BookmakersIntegration
        
        # Crear instancia
        integration = BookmakersIntegration(
            api_keys={'sportradar': 'test_key_12345'}
        )
        
        # Verificar componentes
        assert hasattr(integration, 'bookmakers_fetcher')
        assert hasattr(integration, 'minimum_edge')
        assert hasattr(integration, 'confidence_threshold')
        
        # Verificar métodos principales
        assert hasattr(integration, 'identify_high_confidence_betting_lines')
        assert hasattr(integration, 'analyze_market_inefficiencies')
        assert hasattr(integration, 'get_optimal_betting_strategy')
        
        # Test con datos simulados
        test_data = create_test_data()
        
        # Test de líneas de alta confianza
        high_confidence = integration.identify_high_confidence_betting_lines(
            test_data, 'PTS', min_confidence=0.90
        )
        assert isinstance(high_confidence, list)
        
        print("✅ BookmakersIntegration funcionando")
        return True
    except Exception as e:
        print(f"❌ Error en BookmakersIntegration: {e}")
        return False

def test_complete_workflow():
    """Prueba el flujo completo."""
    print("\n🔍 Probando flujo completo...")
    
    try:
        from utils.bookmakers import BookmakersIntegration
        
        # Crear instancia
        integration = BookmakersIntegration(
            api_keys={'sportradar': 'test_key_12345'}
        )
        
        # Crear datos de predicciones
        predictions_data = create_predictions_data()
        
        # Ejecutar flujo principal
        result = integration.get_best_prediction_odds(
            predictions_data, 
            target='PTS',
            min_confidence=0.85
        )
        
        # Verificar resultado
        assert isinstance(result, dict)
        
        if result.get('success'):
            print(f"✅ Flujo completo exitoso - Mejor apuesta encontrada")
            print(f"   Player: {result['best_bet']['player']}")
            print(f"   Edge: {result['best_bet']['edge']:.2%}")
        else:
            print(f"✅ Flujo completo manejó caso sin datos correctamente")
        
        return True
    except Exception as e:
        print(f"❌ Error en flujo completo: {e}")
        return False

def create_test_data():
    """Crea datos de prueba."""
    np.random.seed(42)
    
    data = []
    players = ['LeBron James', 'Stephen Curry', 'Kevin Durant']
    
    for player in players:
        player_data = {
            'Player': player,
            'Team': f'Team_{player.split()[1]}',
            'PTS': np.random.uniform(22, 32),
            'PTS_confidence': np.random.uniform(0.88, 0.96),
            'PTS_over_25_prob_10': np.random.uniform(0.7, 0.95),
            'PTS_over_25_odds_draftkings': np.random.uniform(1.8, 2.2),
            'PTS_under_25_odds_draftkings': np.random.uniform(1.8, 2.2),
        }
        data.append(player_data)
    
    return pd.DataFrame(data)

def create_predictions_data():
    """Crea datos de predicciones."""
    np.random.seed(42)
    
    data = []
    players = ['LeBron James', 'Stephen Curry', 'Kevin Durant']
    
    for player in players:
        data.append({
            'Player': player,
            'Team': f'Team_{player.split()[1]}',
            'PTS': np.random.uniform(22, 32),
            'PTS_confidence': np.random.uniform(0.88, 0.96),
        })
    
    return pd.DataFrame(data)

def main():
    """Función principal."""
    print("🚀 PRUEBA RÁPIDA DE ORQUESTACIÓN - MÓDULO BOOKMAKERS")
    print("=" * 60)
    
    tests = [
        ("Importaciones", test_imports),
        ("Configuración", test_configuration),
        ("Sportradar API", test_sportradar_api),
        ("Data Fetcher", test_data_fetcher),
        ("Integration", test_integration),
        ("Flujo Completo", test_complete_workflow),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append(result)
        except Exception as e:
            print(f"💥 Error inesperado en {test_name}: {e}")
            results.append(False)
    
    # Resumen
    print("\n" + "=" * 60)
    print("📊 RESUMEN DE PRUEBAS")
    print("=" * 60)
    
    passed = sum(results)
    total = len(results)
    
    print(f"Pruebas exitosas: {passed}/{total}")
    print(f"Tasa de éxito: {(passed/total)*100:.1f}%")
    
    if passed == total:
        print("\n🎉 ¡TODAS LAS PRUEBAS PASARON!")
        print("✅ El módulo bookmakers está correctamente orquestado")
    else:
        print(f"\n⚠️  {total - passed} pruebas fallaron")
        print("❌ Revisar los errores arriba")
    
    print(f"\nFecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 