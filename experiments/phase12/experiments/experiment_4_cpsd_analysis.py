"""
Experiment 4: CPSD Algorithm Component Analysis
Systematic ablation study of Citation Paradigm Shift Detection (CPSD) algorithm components

Research Questions:
1. How do individual CPSD layers contribute to citation paradigm shift detection?
2. What is the value of the ensemble approach vs individual layers?
3. Are current ensemble weights optimal, or can they be improved?

Primary Hypothesis: Ensemble approach outperforms individual layers

Researcher: AI Research Assistant
Date: June 17, 2025
"""

import sys
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any
from datetime import datetime
from collections import defaultdict

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from core.data_processing import process_domain_data
from core.shift_signal_detection import (
    detect_citation_acceleration_shifts,
    detect_citation_regime_changes,
    detect_citation_bursts,
    detect_citation_binary_segmentation,
    ensemble_citation_shift_integration
)


def load_ground_truth_data(domains: List[str]) -> Dict[str, List[int]]:
    """Load ground truth paradigm shifts for validation."""
    ground_truth = {}
    
    for domain in domains:
        truth_file = Path(f"validation/{domain}_groundtruth.json")
        if truth_file.exists():
            with open(truth_file, 'r') as f:
                data = json.load(f)
                # Extract transition years from ground truth periods
                transitions = []
                for period in data.get('historical_periods', []):
                    if 'start_year' in period:
                        transitions.append(period['start_year'])
                ground_truth[domain] = sorted(transitions)
        else:
            print(f"⚠️ Ground truth file not found for {domain}")
            ground_truth[domain] = []
            
    return ground_truth


def create_citation_time_series(domain_data) -> Dict[int, float]:
    """Create citation time series from domain data."""
    citation_series = defaultdict(float)
    
    # Aggregate citations by year
    for paper in domain_data.papers:
        year = paper.pub_year
        citation_series[year] += paper.cited_by_count
        
    return dict(citation_series)


def run_individual_layer(citations: np.ndarray, years_array: np.ndarray, 
                       layer_name: str) -> Tuple[List[int], List[float]]:
    """Run individual CPSD layer detection."""
    if layer_name == 'gradient':
        detected_years = detect_citation_acceleration_shifts(citations, years_array)
        confidence_scores = [0.8] * len(detected_years)
        
    elif layer_name == 'regime':
        detected_years = detect_citation_regime_changes(citations, years_array)
        confidence_scores = [0.7] * len(detected_years)
        
    elif layer_name == 'burst':
        detected_years = detect_citation_bursts(citations, years_array)
        confidence_scores = [0.6] * len(detected_years)
        
    elif layer_name == 'binary_seg':
        detected_years = detect_citation_binary_segmentation(citations, years_array)
        confidence_scores = [0.5] * len(detected_years)
        
    else:
        raise ValueError(f"Unknown layer: {layer_name}")
        
    return detected_years, confidence_scores


def run_ensemble_layers(citations: np.ndarray, years_array: np.ndarray,
                      ensemble_weights: Dict[str, float]) -> Tuple[List[int], List[float]]:
    """Run full ensemble with specified weights."""
    # Run all individual layers
    gradient_shifts = detect_citation_acceleration_shifts(citations, years_array)
    regime_shifts = detect_citation_regime_changes(citations, years_array)
    burst_shifts = detect_citation_bursts(citations, years_array)
    binary_seg_shifts = detect_citation_binary_segmentation(citations, years_array)
    
    # Ensemble integration
    final_shifts, confidence_scores = ensemble_citation_shift_integration(
        gradient_shifts, regime_shifts, burst_shifts, binary_seg_shifts,
        ensemble_weights=ensemble_weights
    )
    
    return final_shifts, confidence_scores


def calculate_temporal_accuracy(detected_years: List[int], 
                              ground_truth_years: List[int], 
                              tolerance: int = 2) -> Dict[str, float]:
    """Calculate temporal accuracy metrics against ground truth."""
    if not ground_truth_years or not detected_years:
        return {
            'temporal_accuracy': 0.0,
            'mean_temporal_error': float('inf'),
            'f1_score': 0.0,
            'precision': 0.0,
            'recall': 0.0
        }
    
    # Calculate matches within tolerance
    true_positives = 0
    temporal_errors = []
    
    for gt_year in ground_truth_years:
        min_error = min([abs(gt_year - det_year) for det_year in detected_years] + [float('inf')])
        if min_error <= tolerance:
            true_positives += 1
            temporal_errors.append(min_error)
    
    # Calculate precision, recall, F1
    precision = true_positives / len(detected_years) if detected_years else 0.0
    recall = true_positives / len(ground_truth_years) if ground_truth_years else 0.0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    temporal_accuracy = true_positives / len(ground_truth_years) if ground_truth_years else 0.0
    mean_temporal_error = np.mean(temporal_errors) if temporal_errors else float('inf')
    
    return {
        'temporal_accuracy': temporal_accuracy,
        'mean_temporal_error': mean_temporal_error,
        'f1_score': f1_score,
        'precision': precision,
        'recall': recall
    }


def run_cpsd_component_experiment():
    """Run comprehensive CPSD component analysis."""
    print("🔬 EXPERIMENT 4: CPSD ALGORITHM COMPONENT ANALYSIS")
    print("=" * 80)
    print("Research Question: How do individual CPSD layers contribute to detection performance?")
    
    domains = [
        'natural_language_processing',
        'computer_vision', 
        'deep_learning',
        'machine_learning',
        'applied_mathematics',
        'machine_translation',
        'computer_science',
        'art'
    ]
    
    # Layer configurations to test
    layer_configs = {
        'layer_1_gradient': 'gradient',
        'layer_2_regime': 'regime', 
        'layer_3_burst': 'burst',
        'layer_4_binary': 'binary_seg'
    }
    
    # Ensemble weight configurations
    weight_configs = {
        'performance_weights': {"gradient": 0.4, "regime": 0.3, "burst": 0.2, "binary_seg": 0.1},
        'equal_weights': {"gradient": 0.25, "regime": 0.25, "burst": 0.25, "binary_seg": 0.25},
        'conservative_weights': {"gradient": 0.2, "regime": 0.4, "burst": 0.133, "binary_seg": 0.267},
        'aggressive_weights': {"gradient": 0.6, "regime": 0.1, "burst": 0.25, "binary_seg": 0.05}
    }
    
    # Load ground truth data
    ground_truth_data = load_ground_truth_data(domains)
    
    results = {
        'metadata': {
            'experiment_name': 'CPSD Algorithm Component Analysis',
            'domains_tested': len(domains),
            'layer_configurations': len(layer_configs),
            'ensemble_configurations': len(weight_configs),
            'analysis_date': datetime.now().isoformat()
        },
        'domain_results': [],
        'summary': {}
    }
    
    # Run experiments across domains
    layer_performance = defaultdict(list)
    ensemble_performance = defaultdict(list)
    
    for domain in domains:
        ground_truth = ground_truth_data.get(domain, [])
        if not ground_truth:
            print(f"⚠️ Skipping {domain} - no ground truth data")
            continue
            
        print(f"\n🔬 Analyzing: {domain}")
        
        # Load domain data
        try:
            processing_result = process_domain_data(domain)
            if not processing_result.success:
                raise RuntimeError(f"Failed to process domain data: {processing_result.error_message}")
            domain_data = processing_result.domain_data
        except Exception as e:
            print(f"  ❌ Failed to load domain data: {e}")
            continue
        
        # Create citation time series
        citation_series = create_citation_time_series(domain_data)
        if not citation_series:
            print(f"  ❌ No citation data available")
            continue
        
        years = sorted(citation_series.keys())
        citation_values = np.array([citation_series[year] for year in years])
        years_array = np.array(years)
        
        print(f"  📊 Citation time series: {len(years)} years ({min(years)}-{max(years)})")
        
        domain_result = {
            'domain': domain,
            'ground_truth_count': len(ground_truth),
            'layer_results': {},
            'ensemble_results': {}
        }
        
        # Test individual layers
        print(f"  🧪 Testing individual layers...")
        for config_name, layer_name in layer_configs.items():
            try:
                detected_years, confidence_scores = run_individual_layer(
                    citation_values, years_array, layer_name
                )
                
                temporal_metrics = calculate_temporal_accuracy(detected_years, ground_truth)
                
                layer_result = {
                    'detected_count': len(detected_years),
                    'detected_years': detected_years,
                    'f1_score': temporal_metrics['f1_score'],
                    'temporal_accuracy': temporal_metrics['temporal_accuracy'],
                    'mean_confidence': np.mean(confidence_scores) if confidence_scores else 0.0
                }
                
                domain_result['layer_results'][config_name] = layer_result
                layer_performance[config_name].append(temporal_metrics['f1_score'])
                
                print(f"    {config_name}: {len(detected_years)} detections, F1={temporal_metrics['f1_score']:.3f}")
                
            except Exception as e:
                print(f"    ❌ {config_name} failed: {e}")
                domain_result['layer_results'][config_name] = {'error': str(e)}
        
        # Test ensemble configurations
        print(f"  🎯 Testing ensemble configurations...")
        for weight_name, weights in weight_configs.items():
            try:
                detected_years, confidence_scores = run_ensemble_layers(
                    citation_values, years_array, weights
                )
                
                temporal_metrics = calculate_temporal_accuracy(detected_years, ground_truth)
                
                ensemble_result = {
                    'detected_count': len(detected_years),
                    'detected_years': detected_years,
                    'f1_score': temporal_metrics['f1_score'],
                    'temporal_accuracy': temporal_metrics['temporal_accuracy'],
                    'mean_confidence': np.mean(confidence_scores) if confidence_scores else 0.0,
                    'weights': weights
                }
                
                domain_result['ensemble_results'][weight_name] = ensemble_result
                ensemble_performance[weight_name].append(temporal_metrics['f1_score'])
                
                print(f"    {weight_name}: {len(detected_years)} detections, F1={temporal_metrics['f1_score']:.3f}")
                
            except Exception as e:
                print(f"    ❌ {weight_name} failed: {e}")
                domain_result['ensemble_results'][weight_name] = {'error': str(e)}
        
        results['domain_results'].append(domain_result)
    
    # Calculate summary statistics
    print(f"\n📊 CROSS-DOMAIN ANALYSIS")
    print("=" * 50)
    
    layer_summary = {}
    for config_name, f1_scores in layer_performance.items():
        if f1_scores:
            layer_summary[config_name] = {
                'mean_f1': np.mean(f1_scores),
                'std_f1': np.std(f1_scores),
                'domains': len(f1_scores)
            }
            print(f"  {config_name}: F1 = {np.mean(f1_scores):.3f} ± {np.std(f1_scores):.3f}")
    
    ensemble_summary = {}
    for weight_name, f1_scores in ensemble_performance.items():
        if f1_scores:
            ensemble_summary[weight_name] = {
                'mean_f1': np.mean(f1_scores),
                'std_f1': np.std(f1_scores),
                'domains': len(f1_scores)
            }
            print(f"  {weight_name}: F1 = {np.mean(f1_scores):.3f} ± {np.std(f1_scores):.3f}")
    
    # Find best configurations
    if layer_summary:
        best_layer = max(layer_summary.items(), key=lambda x: x[1]['mean_f1'])
        print(f"\n🏆 Best Individual Layer: {best_layer[0]} (F1 = {best_layer[1]['mean_f1']:.3f})")
    
    if ensemble_summary:
        best_ensemble = max(ensemble_summary.items(), key=lambda x: x[1]['mean_f1'])
        print(f"🏆 Best Ensemble: {best_ensemble[0]} (F1 = {best_ensemble[1]['mean_f1']:.3f})")
        
        # Calculate ensemble value
        if layer_summary and ensemble_summary:
            best_individual_f1 = max(layer_summary.values(), key=lambda x: x['mean_f1'])['mean_f1']
            best_ensemble_f1 = max(ensemble_summary.values(), key=lambda x: x['mean_f1'])['mean_f1']
            ensemble_improvement = best_ensemble_f1 - best_individual_f1
            
            print(f"\n💡 Ensemble Value: +{ensemble_improvement:.3f} F1 improvement")
            if ensemble_improvement > 0.05:
                print("   ✅ Substantial benefit - Ensemble justified")
            elif ensemble_improvement > 0:
                print("   📈 Moderate benefit - Ensemble marginally useful")
            else:
                print("   ❌ No benefit - Individual layer preferred")
    
    results['summary'] = {
        'layer_performance': layer_summary,
        'ensemble_performance': ensemble_summary
    }
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path("experiments/phase12/results/experiment_4_cpsd_analysis")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    results_file = results_dir / f"cpsd_component_analysis_{timestamp}.json"
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n💾 Results saved: {results_file}")
    print("✅ EXPERIMENT 4 COMPLETE!")
    
    return results


if __name__ == "__main__":
    results = run_cpsd_component_experiment() 