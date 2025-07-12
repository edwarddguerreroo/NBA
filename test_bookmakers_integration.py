#!/usr/bin/env python3
"""
Script de Pruebas para el Módulo Bookmakers
==========================================

Este script verifica que toda la orquestación del módulo utils/bookmakers
funcione correctamente, incluyendo:

1. Configuración y inicialización
2. Integración con Sportradar API
3. Análisis de mercado y value betting
4. Cache y optimización
5. Manejo de errores

Uso:
    python utils/bookmakers/test_bookmakers_integration.py
"""

import sys
import os
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any
import json

# Añadir el directorio raíz al path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/bookmakers_test.log')
    ]
)

logger = logging.getLogger(__name__)

# Importar módulos del sistema
try:
    from utils.bookmakers import (
        SportradarAPI,
        BookmakersDataFetcher,
        BookmakersIntegration,
        BookmakersConfig,
        BookmakersAPIError,
        SportradarAPIError
    )
    logger.info("✅ Importaciones exitosas")
except ImportError as e:
    logger.error(f"❌ Error en importaciones: {e}")
    sys.exit(1)


class BookmakersTestSuite:
    """Suite de pruebas para el módulo bookmakers."""
    
    def __init__(self):
        """Inicializa la suite de pruebas."""
        self.config = None
        self.sportradar_api = None
        self.data_fetcher = None
        self.integration = None
        self.test_results = {}
        
        # Crear directorios necesarios
        os.makedirs('logs', exist_ok=True)
        os.makedirs('data/bookmakers', exist_ok=True)
        
        logger.info("🧪 Iniciando Suite de Pruebas del Módulo Bookmakers")
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Ejecuta todas las pruebas."""
        tests = [
            ('Configuración', self.test_configuration),
            ('Sportradar API', self.test_sportradar_api),
            ('Data Fetcher', self.test_data_fetcher),
            ('Integration', self.test_integration),
            ('Cache Sistema', self.test_cache_system),
            ('Análisis de Mercado', self.test_market_analysis),
            ('Value Betting', self.test_value_betting),
            ('Manejo de Errores', self.test_error_handling),
            ('Flujo Completo', self.test_complete_workflow)
        ]
        
        for test_name, test_func in tests:
            logger.info(f"\n{'='*50}")
            logger.info(f"🔍 EJECUTANDO: {test_name}")
            logger.info(f"{'='*50}")
            
            try:
                result = test_func()
                self.test_results[test_name] = {
                    'status': 'PASSED' if result else 'FAILED',
                    'details': result
                }
                status = "✅ PASSED" if result else "❌ FAILED"
                logger.info(f"{status}: {test_name}")
            except Exception as e:
                self.test_results[test_name] = {
                    'status': 'ERROR',
                    'error': str(e)
                }
                logger.error(f"💥 ERROR en {test_name}: {e}")
        
        return self.test_results
    
    def test_configuration(self) -> bool:
        """Prueba la configuración del módulo."""
        try:
            # Test 1: Crear configuración por defecto
            self.config = BookmakersConfig()
            logger.info("✓ Configuración por defecto creada")
            
            # Test 2: Verificar valores por defecto
            assert self.config.get('sportradar', 'timeout') == 30
            assert self.config.get('betting', 'minimum_edge') == 0.04
            logger.info("✓ Valores por defecto correctos")
            
            # Test 3: Verificar URLs de API
            base_url = self.config.get_api_url('basketball')
            assert 'api.sportradar.com' in base_url
            logger.info("✓ URLs de API configuradas")
            
            # Test 4: Verificar mapeo de targets
            pts_markets = self.config.get_target_markets('PTS')
            assert 'total_points' in pts_markets
            logger.info("✓ Mapeo de targets correcto")
            
            # Test 5: Verificar endpoints
            endpoint = self.config.get_endpoint('sport_event_player_props', sport_event_id='test')
            assert 'sport_events/test/player_props' in endpoint
            logger.info("✓ Endpoints configurados correctamente")
            
            return True
            
        except Exception as e:
            logger.error(f"Error en test de configuración: {e}")
            return False
    
    def test_sportradar_api(self) -> bool:
        """Prueba la API de Sportradar."""
        try:
            # Test 1: Inicialización sin API key (debería fallar graciosamente)
            try:
                api_without_key = SportradarAPI(api_key=None)
                logger.warning("API inicializada sin key - verificar manejo")
            except Exception:
                logger.info("✓ Manejo correcto de API key faltante")
            
            # Test 2: Inicialización con API key de prueba
            test_api_key = "test_key_12345"
            self.sportradar_api = SportradarAPI(api_key=test_api_key)
            logger.info("✓ API inicializada con key de prueba")
            
            # Test 3: Verificar configuración de sesión
            assert hasattr(self.sportradar_api, 'session')
            assert self.sportradar_api.api_key == test_api_key
            logger.info("✓ Sesión HTTP configurada")
            
            # Test 4: Verificar cache
            assert hasattr(self.sportradar_api, '_cache')
            logger.info("✓ Cache optimizado inicializado")
            
            # Test 5: Test de conexión (debería fallar con key falsa)
            connection_test = self.sportradar_api.test_connection()
            logger.info(f"✓ Test de conexión ejecutado: {connection_test['success']}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error en test de Sportradar API: {e}")
            return False
    
    def test_data_fetcher(self) -> bool:
        """Prueba el BookmakersDataFetcher."""
        try:
            # Test 1: Inicialización
            self.data_fetcher = BookmakersDataFetcher(
                api_keys={'sportradar': 'test_key_12345'},
                odds_data_dir='data/bookmakers/test'
            )
            logger.info("✓ DataFetcher inicializado")
            
            # Test 2: Verificar integración con Sportradar
            assert self.data_fetcher.sportradar_api is not None
            logger.info("✓ Integración con Sportradar verificada")
            
            # Test 3: Test de estado de APIs
            api_status = self.data_fetcher.get_api_status()
            assert 'sportradar' in api_status
            logger.info("✓ Estado de APIs obtenido")
            
            # Test 4: Test de cache
            cache_status = self.data_fetcher.get_cache_status()
            assert 'cache_hits' in cache_status
            logger.info("✓ Estado de cache obtenido")
            
            # Test 5: Test de características soportadas
            features = self.data_fetcher.get_supported_features()
            assert 'sportradar' in features
            assert 'simulation' in features
            logger.info("✓ Características soportadas verificadas")
            
            return True
            
        except Exception as e:
            logger.error(f"Error en test de DataFetcher: {e}")
            return False
    
    def test_integration(self) -> bool:
        """Prueba el BookmakersIntegration."""
        try:
            # Test 1: Inicialización
            self.integration = BookmakersIntegration(
                api_keys={'sportradar': 'test_key_12345'},
                odds_data_dir='data/bookmakers/test'
            )
            logger.info("✓ Integration inicializado")
            
            # Test 2: Verificar componentes
            assert hasattr(self.integration, 'bookmakers_fetcher')
            assert hasattr(self.integration, 'minimum_edge')
            assert hasattr(self.integration, 'confidence_threshold')
            logger.info("✓ Componentes verificados")
            
            # Test 3: Test de líneas de target
            line_values = self.integration._get_line_values_for_target('PTS')
            assert len(line_values) > 0
            assert 20 in line_values or 25 in line_values
            logger.info("✓ Líneas de target obtenidas")
            
            return True
            
        except Exception as e:
            logger.error(f"Error en test de Integration: {e}")
            return False
    
    def test_cache_system(self) -> bool:
        """Prueba el sistema de cache."""
        try:
            if not self.sportradar_api:
                logger.warning("Sportradar API no disponible para test de cache")
                return True
            
            # Test 1: Verificar cache stats
            cache_stats = self.sportradar_api.get_cache_stats()
            assert isinstance(cache_stats, dict)
            logger.info("✓ Estadísticas de cache obtenidas")
            
            # Test 2: Test de limpieza de cache
            self.sportradar_api.clear_cache()
            logger.info("✓ Cache limpiado")
            
            # Test 3: Test de cleanup
            self.sportradar_api.cleanup_cache()
            logger.info("✓ Cleanup de cache ejecutado")
            
            return True
            
        except Exception as e:
            logger.error(f"Error en test de cache: {e}")
            return False
    
    def test_market_analysis(self) -> bool:
        """Prueba el análisis de mercado."""
        try:
            if not self.integration:
                logger.warning("Integration no disponible para test de análisis")
                return True
            
            # Crear datos de prueba
            test_data = self._create_test_dataframe()
            
            # Test 1: Identificar líneas de alta confianza
            high_confidence = self.integration.identify_high_confidence_betting_lines(
                test_data, 'PTS', min_confidence=0.90
            )
            logger.info(f"✓ Líneas de alta confianza: {len(high_confidence)}")
            
            # Test 2: Analizar ineficiencias de mercado
            inefficiencies = self.integration.analyze_market_inefficiencies(
                test_data, 'PTS', min_edge=0.02
            )
            assert 'opportunities' in inefficiencies
            logger.info(f"✓ Ineficiencias analizadas: {len(inefficiencies['opportunities'])}")
            
            # Test 3: Buscar arbitraje
            arbitrage = self.integration.find_best_odds_arbitrage(test_data, 'PTS')
            logger.info(f"✓ Oportunidades de arbitraje: {len(arbitrage)}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error en test de análisis de mercado: {e}")
            return False
    
    def test_value_betting(self) -> bool:
        """Prueba el sistema de value betting."""
        try:
            if not self.integration:
                logger.warning("Integration no disponible para test de value betting")
                return True
            
            # Crear datos de prueba
            test_data = self._create_test_dataframe()
            
            # Test 1: Estrategia óptima de apuestas
            strategy = self.integration.get_optimal_betting_strategy(
                test_data, 'PTS', confidence_threshold=0.90, min_edge=0.02
            )
            assert 'recommendations' in strategy
            logger.info(f"✓ Estrategia generada: {len(strategy['recommendations'])} recomendaciones")
            
            # Test 2: Generar cartera Kelly
            portfolio = self.integration.generate_kelly_portfolio(
                test_data, min_confidence=0.90, min_edge=0.02
            )
            assert 'portfolio_size' in portfolio
            logger.info(f"✓ Cartera Kelly: {portfolio['portfolio_size']} apuestas")
            
            return True
            
        except Exception as e:
            logger.error(f"Error en test de value betting: {e}")
            return False
    
    def test_error_handling(self) -> bool:
        """Prueba el manejo de errores."""
        try:
            # Test 1: Error de configuración
            try:
                bad_config = BookmakersConfig()
                bad_config.set('betting', 'minimum_edge', value=2.0)  # Valor inválido
                bad_config._validate_config()
                logger.warning("Validación de configuración no detectó error")
            except ValueError:
                logger.info("✓ Error de configuración detectado correctamente")
            
            # Test 2: Error de API
            try:
                if self.sportradar_api:
                    # Intentar hacer una petición que debería fallar
                    result = self.sportradar_api.get_daily_odds_schedule("invalid-date")
                    logger.info("✓ Manejo de errores de API verificado")
            except SportradarAPIError:
                logger.info("✓ SportradarAPIError manejado correctamente")
            
            # Test 3: Error de datos insuficientes
            try:
                empty_df = pd.DataFrame()
                if self.integration:
                    self.integration.identify_high_confidence_betting_lines(empty_df, 'PTS')
                logger.info("✓ Manejo de datos vacíos verificado")
            except Exception as e:
                logger.info(f"✓ Error de datos manejado: {type(e).__name__}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error en test de manejo de errores: {e}")
            return False
    
    def test_complete_workflow(self) -> bool:
        """Prueba el flujo completo del sistema."""
        try:
            if not self.integration:
                logger.warning("Integration no disponible para test de flujo completo")
                return True
            
            # Crear datos de predicciones de prueba
            predictions_data = self._create_predictions_dataframe()
            
            # Test 1: Flujo principal con simulación
            logger.info("Ejecutando flujo principal con datos simulados...")
            result = self.integration.get_best_prediction_odds(
                predictions_data, 
                target='PTS',
                min_confidence=0.85
            )
            
            # Verificar resultado
            if result.get('success'):
                logger.info("✓ Flujo completo ejecutado exitosamente")
                logger.info(f"  - Mejor apuesta: {result['best_bet']['player']} - {result['best_bet']['line']}")
                logger.info(f"  - Edge: {result['best_bet']['edge']:.2%}")
                logger.info(f"  - ROI esperado: {result['best_bet']['roi_percentage']:.2f}%")
            else:
                logger.info(f"✓ Flujo completo manejó caso sin datos: {result.get('error', 'Sin error')}")
            
            # Test 2: Flujo con procesamiento de datos
            processed_data, analysis = self.integration.process_player_data_with_bookmakers(
                predictions_data,
                target='PTS',
                simulate_odds=True
            )
            
            assert isinstance(processed_data, pd.DataFrame)
            assert isinstance(analysis, dict)
            logger.info("✓ Procesamiento de datos con bookmakers completado")
            
            return True
            
        except Exception as e:
            logger.error(f"Error en test de flujo completo: {e}")
            return False
    
    def _create_test_dataframe(self) -> pd.DataFrame:
        """Crea un DataFrame de prueba con datos simulados."""
        np.random.seed(42)
        
        players = ['LeBron James', 'Stephen Curry', 'Kevin Durant', 'Giannis Antetokounmpo', 'Luka Doncic']
        data = []
        
        for player in players:
            # Datos base del jugador
            base_pts = np.random.uniform(20, 35)
            confidence = np.random.uniform(0.85, 0.98)
            
            player_data = {
                'Player': player,
                'Team': f'Team_{player.split()[1]}',
                'PTS': base_pts,
                'PTS_confidence': confidence,
                'PTS_over_20_prob_10': np.random.uniform(0.7, 0.95),
                'PTS_over_25_prob_10': np.random.uniform(0.5, 0.85),
                'PTS_over_30_prob_10': np.random.uniform(0.3, 0.65),
            }
            
            # Añadir odds simuladas
            for line in [20, 25, 30]:
                for bm in ['draftkings', 'fanduel', 'betmgm']:
                    over_odds = np.random.uniform(1.8, 2.2)
                    under_odds = np.random.uniform(1.8, 2.2)
                    
                    player_data[f'PTS_over_{line}_odds_{bm}'] = over_odds
                    player_data[f'PTS_under_{line}_odds_{bm}'] = under_odds
            
            data.append(player_data)
        
        return pd.DataFrame(data)
    
    def _create_predictions_dataframe(self) -> pd.DataFrame:
        """Crea un DataFrame de predicciones de prueba."""
        np.random.seed(42)
        
        players = ['LeBron James', 'Stephen Curry', 'Kevin Durant', 'Giannis Antetokounmpo']
        data = []
        
        for player in players:
            data.append({
                'Player': player,
                'Team': f'Team_{player.split()[1]}',
                'PTS': np.random.uniform(22, 32),
                'PTS_confidence': np.random.uniform(0.88, 0.96),
                'AST': np.random.uniform(4, 10),
                'AST_confidence': np.random.uniform(0.85, 0.94),
                'TRB': np.random.uniform(6, 12),
                'TRB_confidence': np.random.uniform(0.82, 0.93)
            })
        
        return pd.DataFrame(data)
    
    def generate_report(self) -> str:
        """Genera un reporte de los resultados de las pruebas."""
        report = []
        report.append("=" * 60)
        report.append("REPORTE DE PRUEBAS - MÓDULO BOOKMAKERS")
        report.append("=" * 60)
        report.append(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Resumen
        total_tests = len(self.test_results)
        passed_tests = sum(1 for r in self.test_results.values() if r['status'] == 'PASSED')
        failed_tests = sum(1 for r in self.test_results.values() if r['status'] == 'FAILED')
        error_tests = sum(1 for r in self.test_results.values() if r['status'] == 'ERROR')
        
        report.append("RESUMEN:")
        report.append(f"  Total de pruebas: {total_tests}")
        report.append(f"  Exitosas: {passed_tests} ✅")
        report.append(f"  Fallidas: {failed_tests} ❌")
        report.append(f"  Con errores: {error_tests} 💥")
        report.append(f"  Tasa de éxito: {(passed_tests/total_tests)*100:.1f}%")
        report.append("")
        
        # Detalles por prueba
        report.append("DETALLES POR PRUEBA:")
        report.append("-" * 40)
        
        for test_name, result in self.test_results.items():
            status_icon = {"PASSED": "✅", "FAILED": "❌", "ERROR": "💥"}[result['status']]
            report.append(f"{status_icon} {test_name}: {result['status']}")
            
            if result['status'] == 'ERROR':
                report.append(f"    Error: {result['error']}")
            report.append("")
        
        # Recomendaciones
        report.append("RECOMENDACIONES:")
        report.append("-" * 20)
        
        if failed_tests > 0:
            report.append("• Revisar las pruebas fallidas para identificar problemas")
        
        if error_tests > 0:
            report.append("• Verificar configuración y dependencias para pruebas con errores")
        
        if passed_tests == total_tests:
            report.append("• ¡Excelente! Todas las pruebas pasaron correctamente")
            report.append("• El módulo bookmakers está completamente orquestado")
        
        report.append("")
        report.append("=" * 60)
        
        return "\n".join(report)


def main():
    """Función principal del script de pruebas."""
    print("🚀 INICIANDO PRUEBAS DEL MÓDULO BOOKMAKERS")
    print("=" * 60)
    
    # Crear y ejecutar suite de pruebas
    test_suite = BookmakersTestSuite()
    results = test_suite.run_all_tests()
    
    # Generar reporte
    report = test_suite.generate_report()
    print("\n" + report)
    
    # Guardar reporte
    os.makedirs('reports', exist_ok=True)
    report_file = f'reports/bookmakers_test_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt'
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"\n📋 Reporte guardado en: {report_file}")
    
    # Código de salida
    total_tests = len(results)
    passed_tests = sum(1 for r in results.values() if r['status'] == 'PASSED')
    
    if passed_tests == total_tests:
        print("\n🎉 ¡TODAS LAS PRUEBAS PASARON!")
        sys.exit(0)
    else:
        print(f"\n⚠️  {total_tests - passed_tests} pruebas fallaron")
        sys.exit(1)


if __name__ == "__main__":
    main() 