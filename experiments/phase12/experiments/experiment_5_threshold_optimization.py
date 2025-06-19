"""
Experiment 5: Validation Threshold Optimization
Testing optimal validation thresholds for paradigm detection

Research Questions:
1. What are the optimal validation thresholds for citation-validated vs direction-only acceptance?
2. How do threshold gaps affect citation dependency and detection quality?
3. Are current thresholds (0.5/0.7) optimal?

Primary Hypothesis: Current thresholds provide optimal precision-recall balance

Researcher: AI Research Assistant
Date: June 17, 2025
"""

import sys
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime
from collections import defaultdict

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from core.data_processing import process_domain_data
from core.shift_signal_detection import (
    detect_research_direction_changes,
    cluster_direction_signals_by_proximity,
    detect_citation_structural_breaks,
    validate_direction_with_citation
)
from core.integration import SensitivityConfig


def load_ground_truth_data(domains: List[str]) -> Dict[str, List[int]]:
    """Load ground truth paradigm shifts for validation."""
    ground_truth = {}
    
    for domain in domains:
        truth_file = Path(f"validation/{domain}_groundtruth.json")
        if truth_file.exists():
            with open(truth_file, 'r') as f:
                data = json.load(f)
                transitions = []
                for period in data.get('historical_periods', []):
                    if 'start_year' in period:
                        transitions.append(period['start_year'])
                ground_truth[domain] = sorted(transitions)
        else:
            print(f"⚠️ Ground truth file not found for {domain}")
            ground_truth[domain] = []
            
    return ground_truth


def calculate_validation_metrics(validated_signals: List, ground_truth: List[int]) -> Dict[str, float]:
    """Calculate validation metrics."""
    if not ground_truth or not validated_signals:
        return {'precision': 0.0, 'recall': 0.0, 'f1_score': 0.0}
    
    detected_years = [s.year for s in validated_signals]
    
    # Calculate matches within 2-year tolerance
    true_positives = 0
    for gt_year in ground_truth:
        min_error = min([abs(gt_year - det_year) for det_year in detected_years] + [float('inf')])
        if min_error <= 2:
            true_positives += 1
    
    precision = true_positives / len(detected_years) if detected_years else 0.0
    recall = true_positives / len(ground_truth) if ground_truth else 0.0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {'precision': precision, 'recall': recall, 'f1_score': f1_score}


def test_threshold_combination(domain_data, domain_name: str, ground_truth: List[int],
                             citation_threshold: float, direction_threshold: float) -> Dict[str, Any]:
    """Test specific threshold combination."""
    
    # Generate signals with optimal settings from previous experiments
    raw_direction_signals = detect_research_direction_changes(
        domain_data, sensitivity_threshold=0.3  # From Experiment 1
    )
    
    # Create sensitivity config with 3-year clustering (from Experiment 2)
    sensitivity_config = SensitivityConfig(granularity=3)  # Granularity 3 = 3-year clustering
    clustered_direction_signals = cluster_direction_signals_by_proximity(
        raw_direction_signals, sensitivity_config
    )
    
    # Generate citation signals
    citation_signals = detect_citation_structural_breaks(domain_data, domain_name)
    
    # Create custom config with test thresholds
    custom_config = SensitivityConfig(granularity=3)  # Base config
    custom_config.citation_validated_threshold = citation_threshold
    custom_config.direction_only_threshold = direction_threshold
    custom_config.citation_boost = 0.3
    custom_config.breakthrough_bonus = 0.2
    
    # Run validation with custom thresholds
    validated_signals = validate_direction_with_citation(
        clustered_direction_signals, citation_signals, domain_data, domain_name, custom_config
    )
    
    # Analyze validation pathways
    citation_path_signals = [s for s in validated_signals if "validated" in s.signal_type]
    direction_path_signals = [s for s in validated_signals if "only" in s.signal_type]
    
    # Calculate metrics
    metrics = calculate_validation_metrics(validated_signals, ground_truth)
    
    return {
        'citation_threshold': citation_threshold,
        'direction_threshold': direction_threshold,
        'threshold_gap': direction_threshold - citation_threshold,
        'total_clustered': len(clustered_direction_signals),
        'total_validated': len(validated_signals),
        'citation_path_count': len(citation_path_signals),
        'direction_path_count': len(direction_path_signals),
        'acceptance_rate': len(validated_signals) / len(clustered_direction_signals) if clustered_direction_signals else 0.0,
        'citation_path_rate': len(citation_path_signals) / len(clustered_direction_signals) if clustered_direction_signals else 0.0,
        **metrics
    }


def run_threshold_optimization_experiment():
    """Run validation threshold optimization experiment."""
    
    print("🔬 EXPERIMENT 5: VALIDATION THRESHOLD OPTIMIZATION")
    print("=" * 80)
    print("Research Question: What are the optimal validation thresholds for paradigm detection?")
    
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
    
    # Define threshold combinations to test
    threshold_combinations = [
        (0.3, 0.5),  # Permissive
        (0.4, 0.6),  # Moderate-permissive
        (0.5, 0.7),  # Current default
        (0.6, 0.8),  # Conservative
        (0.7, 0.9),  # Very conservative
        (0.4, 0.4),  # Equal thresholds
        (0.3, 0.9),  # Wide gap
    ]
    
    # Load ground truth data
    ground_truth_data = load_ground_truth_data(domains)
    
    results = {
        'metadata': {
            'experiment_name': 'Validation Threshold Optimization',
            'domains_tested': len(domains),
            'threshold_combinations': len(threshold_combinations),
            'analysis_date': datetime.now().isoformat()
        },
        'domain_results': [],
        'summary': {}
    }
    
    all_results = []
    
    # Test each domain
    for domain in domains:
        ground_truth = ground_truth_data.get(domain, [])
        if not ground_truth:
            print(f"⚠️ Skipping {domain} - no ground truth data")
            continue
            
        print(f"\n🔬 Testing: {domain}")
        
        # Load domain data
        try:
            processing_result = process_domain_data(domain)
            if not processing_result.success:
                raise RuntimeError(f"Failed to process domain data: {processing_result.error_message}")
            domain_data = processing_result.domain_data
        except Exception as e:
            print(f"  ❌ Failed to load domain data: {e}")
            continue
        
        domain_results = []
        
        # Test each threshold combination
        for cite_thresh, dir_thresh in threshold_combinations:
            try:
                result = test_threshold_combination(
                    domain_data, domain, ground_truth, cite_thresh, dir_thresh
                )
                result['domain'] = domain
                domain_results.append(result)
                all_results.append(result)
                
                print(f"  {cite_thresh:.1f}/{dir_thresh:.1f}: F1={result['f1_score']:.3f}, Accept={result['acceptance_rate']:.1%}")
                
            except Exception as e:
                print(f"  ❌ {cite_thresh:.1f}/{dir_thresh:.1f} failed: {e}")
        
        # Find best for this domain
        if domain_results:
            best_domain = max(domain_results, key=lambda x: x['f1_score'])
            print(f"  🏆 Best: {best_domain['citation_threshold']:.1f}/{best_domain['direction_threshold']:.1f} (F1={best_domain['f1_score']:.3f})")
        
        results['domain_results'].append({
            'domain': domain,
            'results': domain_results,
            'best_combination': best_domain if domain_results else None
        })
    
    # Cross-domain analysis
    print(f"\n📊 CROSS-DOMAIN ANALYSIS")
    print("=" * 50)
    
    # Aggregate by threshold combination
    threshold_performance = defaultdict(list)
    for result in all_results:
        key = f"{result['citation_threshold']:.1f}/{result['direction_threshold']:.1f}"
        threshold_performance[key].append(result)
    
    # Calculate summary statistics
    summary_stats = {}
    for combo, domain_results in threshold_performance.items():
        f1_scores = [r['f1_score'] for r in domain_results]
        acceptance_rates = [r['acceptance_rate'] for r in domain_results]
        citation_rates = [r['citation_path_rate'] for r in domain_results]
        
        summary_stats[combo] = {
            'mean_f1': np.mean(f1_scores),
            'std_f1': np.std(f1_scores),
            'mean_acceptance': np.mean(acceptance_rates),
            'mean_citation_dependency': np.mean(citation_rates),
            'domains': len(domain_results),
            'citation_threshold': domain_results[0]['citation_threshold'],
            'direction_threshold': domain_results[0]['direction_threshold'],
            'threshold_gap': domain_results[0]['threshold_gap']
        }
        
        print(f"  {combo}: F1={np.mean(f1_scores):.3f}±{np.std(f1_scores):.3f}, Accept={np.mean(acceptance_rates):.1%}")
    
    # Find optimal configurations
    best_f1_combo = max(summary_stats.items(), key=lambda x: x[1]['mean_f1'])
    best_balance_combo = max(summary_stats.items(), 
                           key=lambda x: x[1]['mean_f1'] * x[1]['mean_acceptance'])
    
    print(f"\n🏆 OPTIMAL CONFIGURATIONS:")
    print(f"  Best F1: {best_f1_combo[0]} (F1={best_f1_combo[1]['mean_f1']:.3f})")
    print(f"  Best Balance: {best_balance_combo[0]} (F1={best_balance_combo[1]['mean_f1']:.3f}, Accept={best_balance_combo[1]['mean_acceptance']:.1%})")
    
    # Compare with current default
    current_default = "0.5/0.7"
    if current_default in summary_stats:
        current_perf = summary_stats[current_default]
        best_perf = best_f1_combo[1]
        improvement = best_perf['mean_f1'] - current_perf['mean_f1']
        
        print(f"\n📊 CURRENT DEFAULT ANALYSIS:")
        print(f"  Current (0.5/0.7): F1={current_perf['mean_f1']:.3f}")
        print(f"  Optimal: F1={best_perf['mean_f1']:.3f}")
        print(f"  Potential improvement: {improvement:+.3f}")
        
        if improvement > 0.01:
            print("  💡 Recommendation: Consider updating default thresholds")
        else:
            print("  ✅ Recommendation: Current defaults are near-optimal")
    
    # Threshold gap analysis
    gap_effects = defaultdict(list)
    for combo, stats in summary_stats.items():
        gap = stats['threshold_gap']
        gap_effects[gap].append(stats['mean_f1'])
    
    print(f"\n📏 THRESHOLD GAP ANALYSIS:")
    for gap in sorted(gap_effects.keys()):
        f1_scores = gap_effects[gap]
        print(f"  Gap {gap:.1f}: F1={np.mean(f1_scores):.3f} (Citation dependency varies)")
    
    results['summary'] = {
        'threshold_performance': summary_stats,
        'optimal_f1': best_f1_combo,
        'optimal_balance': best_balance_combo,
        'gap_effects': dict(gap_effects)
    }
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path("experiments/phase12/results/experiment_5_threshold_optimization")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    results_file = results_dir / f"threshold_optimization_{timestamp}.json"
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n💾 Results saved: {results_file}")
    print("✅ EXPERIMENT 5 COMPLETE!")
    
    return results


if __name__ == "__main__":
    results = run_threshold_optimization_experiment() 