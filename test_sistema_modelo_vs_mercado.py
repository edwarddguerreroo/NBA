#!/usr/bin/env python3
"""
Script de Prueba - Sistema Modelo vs Mercado COMPLETO
====================================================

Prueba la integración completa del sistema para TODOS los targets disponibles:

PLAYER TARGETS:
- PTS: Puntos del jugador
- AST: Asistencias del jugador
- TRB: Rebotes del jugador
- 3P: Triples del jugador
- DD: Double-double del jugador

TEAM/GAME TARGETS:
- is_win: Victoria del equipo (1x2/moneyline)
- total_points: Puntos totales del partido
- teams_points: Puntos por equipo (home/away)

Utiliza el endpoint real de Sportradar que funciona:
sr:sport_event:59850122 (Lakers vs Nuggets)
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from utils.bookmakers.bookmakers_integration import BookmakersIntegration

def create_mock_predictions_all_targets():
    """
    Crea predicciones simuladas para TODOS los targets disponibles.
    
    Returns:
        DataFrame con predicciones para todos los targets
    """
    print("Creando predicciones simuladas para TODOS los targets...")
    
    # Jugadores de ejemplo del partido Lakers vs Nuggets
    players_data = [
        # Lakers
        {'Player': 'LeBron James', 'Team': 'LAL', 'Position': 'SF'},
        {'Player': 'Anthony Davis', 'Team': 'LAL', 'Position': 'PF'},
        {'Player': 'Russell Westbrook', 'Team': 'LAL', 'Position': 'PG'},
        {'Player': 'Austin Reaves', 'Team': 'LAL', 'Position': 'SG'},
        {'Player': 'Rui Hachimura', 'Team': 'LAL', 'Position': 'PF'},
        
        # Nuggets
        {'Player': 'Nikola Jokic', 'Team': 'DEN', 'Position': 'C'},
        {'Player': 'Jamal Murray', 'Team': 'DEN', 'Position': 'PG'},
        {'Player': 'Michael Porter Jr.', 'Team': 'DEN', 'Position': 'SF'},
        {'Player': 'Aaron Gordon', 'Team': 'DEN', 'Position': 'PF'},
        {'Player': 'Kentavious Caldwell-Pope', 'Team': 'DEN', 'Position': 'SG'},
    ]
    
    predictions = []
    
    for player in players_data:
        # Predicciones específicas por jugador y posición
        if player['Player'] == 'LeBron James':
            pred = {
                'predicted_PTS': 25.8, 'predicted_AST': 7.2, 'predicted_TRB': 8.1, 
                'predicted_3P': 2.1, 'predicted_DD': 0.75
            }
        elif player['Player'] == 'Anthony Davis':
            pred = {
                'predicted_PTS': 22.5, 'predicted_AST': 3.8, 'predicted_TRB': 11.2, 
                'predicted_3P': 1.2, 'predicted_DD': 0.65
            }
        elif player['Player'] == 'Nikola Jokic':
            pred = {
                'predicted_PTS': 26.3, 'predicted_AST': 9.8, 'predicted_TRB': 12.4, 
                'predicted_3P': 1.8, 'predicted_DD': 0.88
            }
        elif player['Player'] == 'Jamal Murray':
            pred = {
                'predicted_PTS': 19.7, 'predicted_AST': 5.6, 'predicted_TRB': 4.2, 
                'predicted_3P': 2.8, 'predicted_DD': 0.15
            }
        elif player['Player'] == 'Russell Westbrook':
            pred = {
                'predicted_PTS': 16.8, 'predicted_AST': 8.1, 'predicted_TRB': 6.7, 
                'predicted_3P': 1.1, 'predicted_DD': 0.45
            }
        else:
            # Predicciones genéricas para otros jugadores
            if player['Position'] == 'C':
                pred = {
                    'predicted_PTS': np.random.uniform(12, 18), 'predicted_AST': np.random.uniform(2, 4), 
                    'predicted_TRB': np.random.uniform(8, 12), 'predicted_3P': np.random.uniform(0.5, 1.5), 
                    'predicted_DD': np.random.uniform(0.3, 0.6)
                }
            elif player['Position'] == 'PG':
                pred = {
                    'predicted_PTS': np.random.uniform(14, 20), 'predicted_AST': np.random.uniform(5, 8), 
                    'predicted_TRB': np.random.uniform(3, 6), 'predicted_3P': np.random.uniform(1.5, 3), 
                    'predicted_DD': np.random.uniform(0.1, 0.3)
                }
            else:
                pred = {
                    'predicted_PTS': np.random.uniform(10, 16), 'predicted_AST': np.random.uniform(2, 5), 
                    'predicted_TRB': np.random.uniform(4, 8), 'predicted_3P': np.random.uniform(1, 2.5), 
                    'predicted_DD': np.random.uniform(0.05, 0.25)
                }
        
        # Combinar datos del jugador con predicciones
        player_prediction = {**player, **pred}
        
        # Agregar confianza alta para todos los targets
        player_prediction.update({
            'confidence': np.random.uniform(0.92, 0.98),
            'date': datetime.now().strftime('%Y-%m-%d')
        })
        
        predictions.append(player_prediction)
    
    # Agregar predicciones de EQUIPOS/PARTIDOS
    team_predictions = [
        {
            'Team': 'LAL',
            'Opponent': 'DEN',
            'predicted_is_win': 0.45,      # Lakers 45% probabilidad de ganar
            'predicted_total_points': 224.5, # Total puntos del partido
            'predicted_teams_points': 112.8, # Puntos de Lakers
            'confidence': 0.94,
            'date': datetime.now().strftime('%Y-%m-%d')
        },
        {
            'Team': 'DEN', 
            'Opponent': 'LAL',
            'predicted_is_win': 0.55,      # Nuggets 55% probabilidad de ganar
            'predicted_total_points': 224.5, # Total puntos del partido (mismo)
            'predicted_teams_points': 111.7, # Puntos de Nuggets
            'confidence': 0.94,
            'date': datetime.now().strftime('%Y-%m-%d')
        }
    ]
    
    # Combinar predicciones de jugadores y equipos
    all_predictions = predictions + team_predictions
    
    df = pd.DataFrame(all_predictions)
    
    print(f"✅ Predicciones creadas para {len(predictions)} jugadores y {len(team_predictions)} equipos")
    print(f"   Targets de jugadores: PTS, AST, TRB, 3P, DD")
    print(f"   Targets de equipos: is_win, total_points, teams_points")
    
    return df

def test_all_targets_integration():
    """
    Prueba la integración completa para TODOS los targets disponibles.
    """
    print("="*80)
    print("PRUEBA SISTEMA MODELO VS MERCADO - TODOS LOS TARGETS")
    print("="*80)
    
    # Inicializar sistema
    print("\n1. Inicializando sistema de bookmakers...")
    bookmakers = BookmakersIntegration()
    
    # Crear predicciones simuladas
    print("\n2. Generando predicciones simuladas...")
    predictions_df = create_mock_predictions_all_targets()
    
    # Evento de prueba (Lakers vs Nuggets)
    sport_event_id = "sr:sport_event:59850122"
    
    # Lista de TODOS los targets disponibles
    all_targets = [
        # Player targets
        'PTS', 'AST', 'TRB', '3P', 'DD',
        # Team/Game targets
        'is_win', 'total_points', 'teams_points'
    ]
    
    print(f"\n3. Probando {len(all_targets)} targets: {all_targets}")
    
    # Probar cada target
    results_summary = {
        'successful_targets': [],
        'failed_targets': [],
        'total_opportunities': 0
    }
    
    for target in all_targets:
        print(f"\n   Probando target: {target}")
        print(f"   {'='*50}")
        
        try:
            # Comparar modelo vs mercado
            comparison = bookmakers.compare_model_vs_market(
                model_predictions=predictions_df,
                sport_event_id=sport_event_id,
                target=target,
                min_edge=0.03,  # 3% edge mínimo
                min_confidence=0.90
            )
            
            if comparison.get('success', False):
                opportunities = len(comparison.get('opportunities', []))
                results_summary['successful_targets'].append(target)
                results_summary['total_opportunities'] += opportunities
                
                print(f"   ✅ {target}: {opportunities} oportunidades encontradas")
                
                # Mostrar resumen del target
                summary = comparison.get('summary', {})
                print(f"      - Jugadores/equipos analizados: {summary.get('total_players_analyzed', 0)}")
                print(f"      - Con ventaja: {summary.get('players_with_edges', 0)}")
                print(f"      - Edge máximo: {summary.get('max_edge', 0):.3f}")
                
                # Mostrar mejores oportunidades
                if opportunities > 0:
                    best_opps = comparison['opportunities'][:2]  # Top 2
                    for i, opp in enumerate(best_opps, 1):
                        entity = opp.get('player', opp.get('team', 'Unknown'))
                        edge = opp.get('edge', 0)
                        line = opp.get('line', 'N/A')
                        print(f"      {i}. {entity} | Línea: {line} | Edge: {edge:.3f}")
                
            else:
                results_summary['failed_targets'].append(target)
                error = comparison.get('error', 'Unknown error')
                print(f"   ❌ {target}: {error}")
                
        except Exception as e:
            results_summary['failed_targets'].append(target)
            print(f"   ❌ {target}: Error - {str(e)}")
    
    # Resumen final
    print(f"\n{'='*80}")
    print("RESUMEN FINAL")
    print(f"{'='*80}")
    print(f"✅ Targets exitosos: {len(results_summary['successful_targets'])}/{len(all_targets)}")
    print(f"   {results_summary['successful_targets']}")
    print(f"❌ Targets fallidos: {len(results_summary['failed_targets'])}")
    print(f"   {results_summary['failed_targets']}")
    print(f"📊 Total oportunidades: {results_summary['total_opportunities']}")
    
    # Análisis por categoría
    player_targets = [t for t in results_summary['successful_targets'] if t in ['PTS', 'AST', 'TRB', '3P', 'DD']]
    team_targets = [t for t in results_summary['successful_targets'] if t in ['is_win', 'total_points', 'teams_points']]
    
    print(f"\n📈 Análisis por categoría:")
    print(f"   Player targets exitosos: {len(player_targets)}/5 {player_targets}")
    print(f"   Team targets exitosos: {len(team_targets)}/3 {team_targets}")
    
    if len(results_summary['successful_targets']) >= 6:
        print(f"\n🎉 ÉXITO: Sistema funciona para la mayoría de targets!")
    elif len(results_summary['successful_targets']) >= 3:
        print(f"\n⚠️  PARCIAL: Sistema funciona para algunos targets")
    else:
        print(f"\n❌ FALLO: Sistema necesita correcciones")
    
    return results_summary

if __name__ == "__main__":
    try:
        results = test_all_targets_integration()
        print(f"\n✅ Prueba completada. Resultados: {len(results['successful_targets'])} exitosos")
    except Exception as e:
        print(f"\n❌ Error en la prueba: {e}")
        import traceback
        traceback.print_exc() 