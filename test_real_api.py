#!/usr/bin/env python3
"""
Test de Validación de API Real de Sportradar
==========================================

Este script valida que el sistema funcione exclusivamente con datos reales
de la API de Sportradar, sin simulaciones ni datos inventados.

Requiere:
- Variable de entorno SPORTRADAR_API_KEY configurada
- Conexión a internet
- API key válida de Sportradar
"""

import os
import sys
import logging
from datetime import datetime
from typing import Dict, Any

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_api_key_configuration():
    """Valida que la API key esté configurada correctamente."""
    logger.info("🔍 Validando configuración de API key...")
    
    api_key = os.getenv('SPORTRADAR_API')
    if not api_key:
        logger.error("❌ Variable de entorno SPORTRADAR_API no configurada")
        logger.error("Configura: export SPORTRADAR_API=tu_api_key_aqui")
        return False
    
    if len(api_key) < 20:
        logger.error("❌ API key parece inválida (muy corta)")
        return False
    
    logger.info(f"✅ API key configurada: {'*' * (len(api_key) - 4)}{api_key[-4:]}")
    return True

def test_sportradar_api_initialization():
    """Valida que la API de Sportradar se inicialice correctamente."""
    logger.info("🔍 Probando inicialización de Sportradar API...")
    
    try:
        from utils.bookmakers import SportradarAPI
        
        # Inicializar con API key real
        api_key = os.getenv('SPORTRADAR_API')
        api = SportradarAPI(api_key=api_key)
        
        logger.info("✅ Sportradar API inicializada correctamente")
        return True, api
        
    except Exception as e:
        logger.error(f"❌ Error inicializando Sportradar API: {e}")
        return False, None

def test_sportradar_connection(api):
    """Valida que la conexión con Sportradar funcione."""
    logger.info("🔍 Probando conexión con Sportradar...")
    
    try:
        test_result = api.test_connection()
        
        if test_result.get('success', False):
            logger.info("✅ Conexión con Sportradar exitosa")
            return True
        else:
            error_msg = test_result.get('error', 'Unknown error')
            logger.error(f"❌ Error en conexión: {error_msg}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Excepción en test de conexión: {e}")
        return False

def test_bookmakers_data_fetcher():
    """Valida que el BookmakersDataFetcher funcione con datos reales."""
    logger.info("🔍 Probando BookmakersDataFetcher...")
    
    try:
        from utils.bookmakers import BookmakersDataFetcher
        
        # Inicializar fetcher (debería usar la API key del entorno)
        fetcher = BookmakersDataFetcher()
        
        # Verificar que se inicializó correctamente
        if fetcher.sportradar_api is None:
            logger.error("❌ Sportradar API no se inicializó en el fetcher")
            return False
        
        logger.info("✅ BookmakersDataFetcher inicializado correctamente")
        return True, fetcher
        
    except Exception as e:
        logger.error(f"❌ Error inicializando BookmakersDataFetcher: {e}")
        return False, None

def test_real_data_fetching(fetcher):
    """Valida que se puedan obtener datos reales."""
    logger.info("🔍 Probando obtención de datos reales...")
    
    try:
        # Intentar obtener datos NBA reales
        odds_data = fetcher.get_nba_odds_from_sportradar(
            include_props=True
        )
        
        if odds_data.get('success', False):
            games_count = odds_data.get('games_with_odds', 0)
            logger.info(f"✅ Datos reales obtenidos: {games_count} juegos con odds")
            return True
        else:
            error_msg = odds_data.get('error', 'Unknown error')
            logger.error(f"❌ Error obteniendo datos reales: {error_msg}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Excepción obteniendo datos reales: {e}")
        return False

def test_no_simulation_fallback():
    """Valida que el sistema NO use simulación como fallback."""
    logger.info("🔍 Validando que no se use simulación...")
    
    try:
        from utils.bookmakers import BookmakersIntegration
        import pandas as pd
        
        # Crear datos de prueba
        test_data = pd.DataFrame({
            'Player': ['Test Player'],
            'Team': ['TEST'],
            'PTS': [25.0],
            'PTS_confidence': [0.95]
        })
        
        integration = BookmakersIntegration()
        
        # Intentar procesar sin especificar fuente de datos
        result_df, analysis = integration.process_player_data_with_bookmakers(
            test_data, 
            target='PTS',
            use_api=False,  # No usar API
            odds_file=None  # No usar archivo
        )
        
        # Debería fallar sin simulación
        if analysis.get('success', True):
            logger.error("❌ El sistema no falló como esperado - podría estar usando simulación")
            return False
        
        if 'requires_real_data' in analysis:
            logger.info("✅ El sistema correctamente requiere datos reales")
            return True
        else:
            logger.error("❌ El sistema no indica que requiere datos reales")
            return False
            
    except Exception as e:
        logger.error(f"❌ Error en test de no simulación: {e}")
        return False

def test_api_integration():
    """Valida la integración completa con API real."""
    logger.info("🔍 Probando integración completa con API real...")
    
    try:
        from utils.bookmakers import BookmakersIntegration
        import pandas as pd
        
        # Crear datos de prueba
        test_data = pd.DataFrame({
            'Player': ['LeBron James', 'Stephen Curry'],
            'Team': ['LAL', 'GSW'],
            'PTS': [25.0, 28.0],
            'PTS_confidence': [0.95, 0.97]
        })
        
        integration = BookmakersIntegration()
        
        # Procesar con API real
        result_df, analysis = integration.process_player_data_with_bookmakers(
            test_data, 
            target='PTS',
            use_api=True,
            api_provider='sportradar'
        )
        
        if analysis.get('success', True) and 'requires_real_data' not in analysis:
            logger.info("✅ Integración con API real exitosa")
            logger.info(f"Análisis generado: {len(analysis)} elementos")
            return True
        else:
            error_msg = analysis.get('error', 'Unknown error')
            logger.error(f"❌ Error en integración: {error_msg}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Excepción en integración: {e}")
        return False

def main():
    """Función principal de pruebas."""
    logger.info("🚀 INICIANDO VALIDACIÓN DE API REAL DE SPORTRADAR")
    logger.info("=" * 60)
    
    tests = [
        ("Configuración API Key", test_api_key_configuration),
        ("Inicialización Sportradar API", test_sportradar_api_initialization),
        ("Inicialización BookmakersDataFetcher", test_bookmakers_data_fetcher),
        ("No simulación como fallback", test_no_simulation_fallback),
        ("Integración completa con API", test_api_integration),
    ]
    
    results = []
    api = None
    fetcher = None
    
    for test_name, test_func in tests:
        logger.info(f"\n📋 {test_name}...")
        
        try:
            if test_func == test_sportradar_api_initialization:
                success, api = test_func()
            elif test_func == test_sportradar_connection:
                success = test_func(api)
            elif test_func == test_bookmakers_data_fetcher:
                success, fetcher = test_func()
            elif test_func == test_real_data_fetching:
                success = test_func(fetcher)
            else:
                success = test_func()
                
            results.append((test_name, success))
            
        except Exception as e:
            logger.error(f"❌ Error en {test_name}: {e}")
            results.append((test_name, False))
    
    # Pruebas adicionales si hay API disponible
    if api:
        logger.info(f"\n📋 Conexión con Sportradar...")
        connection_success = test_sportradar_connection(api)
        results.append(("Conexión Sportradar", connection_success))
    
    if fetcher:
        logger.info(f"\n📋 Obtención de datos reales...")
        data_success = test_real_data_fetching(fetcher)
        results.append(("Datos reales", data_success))
    
    # Resumen final
    logger.info("\n" + "=" * 60)
    logger.info("📊 RESUMEN DE VALIDACIÓN")
    logger.info("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        logger.info(f"{status} - {test_name}")
        if success:
            passed += 1
    
    logger.info(f"\n📈 RESULTADO: {passed}/{total} pruebas exitosas")
    
    if passed == total:
        logger.info("🎉 ¡TODAS LAS VALIDACIONES EXITOSAS!")
        logger.info("El sistema está configurado correctamente para usar datos reales de Sportradar")
        return True
    else:
        logger.error("⚠️  ALGUNAS VALIDACIONES FALLARON")
        logger.error("Revisa la configuración antes de usar el sistema")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 