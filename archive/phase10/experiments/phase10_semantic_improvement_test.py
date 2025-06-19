#!/usr/bin/env python3
"""
Phase 10 IMPROVEMENT-002 Testing Framework

Tests data-driven semantic pattern discovery against hardcoded pattern baseline.
Compares performance, signal quality, and computational efficiency.
"""

import sys
import time
import json
import tracemalloc
from pathlib import Path
from typing import Dict, List, Any, Tuple
from collections import defaultdict

# Add project root to path for core module imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from core.data_processing import process_domain_data
from core.shift_signal_detection import detect_shift_signals


def test_semantic_improvement(domain_name: str) -> Dict[str, Any]:
    """
    Test improved semantic detection performance for a domain.
    
    Args:
        domain_name: Name of domain to test
        
    Returns:
        Dictionary with improvement test results
    """
    print(f"\n🧪 TESTING SEMANTIC IMPROVEMENT: {domain_name}")
    print("=" * 70)
    
    try:
        # Load domain data
        processing_result = process_domain_data(domain_name)
        if not processing_result.success:
            raise ValueError(f"Failed to load domain data: {processing_result.error_message}")
        domain_data = processing_result.domain_data
        
        # Test improved semantic detection
        print("  🧠 Testing IMPROVED data-driven semantic detection...")
        tracemalloc.start()
        start_time = time.time()
        
        improved_signals, _ = detect_shift_signals(
            domain_data, 
            domain_name,
            use_citation=False,
            use_semantic=True,  # This will use improved detection
            use_direction=False
        )
        
        end_time = time.time()
        current_memory, peak_memory = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        # Extract semantic signals from improved approach
        improved_semantic_signals = [s for s in improved_signals if 
                                   "semantic" in s.signal_type.lower()]
        
        # Calculate metrics
        improvement_metrics = {
            'domain_name': domain_name,
            'improved_signal_count': len(improved_semantic_signals),
            'improved_signal_years': [s.year for s in improved_semantic_signals],
            'improved_confidence_scores': [s.confidence for s in improved_semantic_signals],
            'improved_evidence_strength': [s.evidence_strength for s in improved_semantic_signals],
            'improved_paradigm_significance': [s.paradigm_significance for s in improved_semantic_signals],
            'improved_computational_time': end_time - start_time,
            'improved_peak_memory_mb': peak_memory / 1024 / 1024,
            'signal_types': [s.signal_type for s in improved_semantic_signals],
            'transition_descriptions': [s.transition_description for s in improved_semantic_signals]
        }
        
        # Quality analysis
        if improved_semantic_signals:
            improvement_metrics.update({
                'improved_avg_confidence': sum(improvement_metrics['improved_confidence_scores']) / len(improved_semantic_signals),
                'improved_max_confidence': max(improvement_metrics['improved_confidence_scores']),
                'improved_min_confidence': min(improvement_metrics['improved_confidence_scores']),
                'improved_avg_evidence_strength': sum(improvement_metrics['improved_evidence_strength']) / len(improved_semantic_signals),
                'improved_avg_paradigm_significance': sum(improvement_metrics['improved_paradigm_significance']) / len(improved_semantic_signals)
            })
        else:
            improvement_metrics.update({
                'improved_avg_confidence': 0.0,
                'improved_max_confidence': 0.0,
                'improved_min_confidence': 0.0,
                'improved_avg_evidence_strength': 0.0,
                'improved_avg_paradigm_significance': 0.0
            })
        
        print(f"  📊 IMPROVED SEMANTIC RESULTS:")
        print(f"      Semantic signals detected: {improvement_metrics['improved_signal_count']}")
        print(f"      Signal years: {improvement_metrics['improved_signal_years']}")
        print(f"      Signal types: {set(improvement_metrics['signal_types'])}")
        print(f"      Average confidence: {improvement_metrics['improved_avg_confidence']:.3f}")
        print(f"      Average paradigm significance: {improvement_metrics['improved_avg_paradigm_significance']:.3f}")
        print(f"      Computational time: {improvement_metrics['improved_computational_time']:.3f}s")
        print(f"      Peak memory: {improvement_metrics['improved_peak_memory_mb']:.1f}MB")
        
        return improvement_metrics
        
    except Exception as e:
        print(f"  ❌ Error testing semantic improvement for {domain_name}: {e}")
        import traceback
        traceback.print_exc()
        return {
            'domain_name': domain_name,
            'error': str(e),
            'improved_signal_count': -1,
            'improved_computational_time': -1,
            'improved_peak_memory_mb': -1
        }


def compare_with_baseline(improvement_results: Dict[str, Dict[str, Any]], 
                         baseline_file: str) -> Dict[str, Any]:
    """
    Compare improvement results with baseline performance.
    
    Args:
        improvement_results: Results from improved semantic detection
        baseline_file: Path to baseline results file
        
    Returns:
        Comparative analysis results
    """
    print(f"\n📊 COMPARING WITH BASELINE: {baseline_file}")
    print("=" * 60)
    
    # Load baseline results
    try:
        with open(baseline_file, 'r') as f:
            baseline_data = json.load(f)
        baseline_results = baseline_data['results']
    except Exception as e:
        print(f"  ❌ Error loading baseline: {e}")
        return {'error': str(e)}
    
    comparison_results = {}
    
    for domain in improvement_results.keys():
        if 'error' in improvement_results[domain]:
            continue
            
        baseline_metrics = baseline_results.get(domain, {})
        improvement_metrics = improvement_results[domain]
        
        if 'error' in baseline_metrics:
            continue
        
        # Calculate improvements
        baseline_count = baseline_metrics.get('signal_count', 0)
        improved_count = improvement_metrics.get('improved_signal_count', 0)
        
        baseline_confidence = baseline_metrics.get('avg_confidence', 0.0)
        improved_confidence = improvement_metrics.get('improved_avg_confidence', 0.0)
        
        baseline_significance = baseline_metrics.get('avg_paradigm_significance', 0.0)
        improved_significance = improvement_metrics.get('improved_avg_paradigm_significance', 0.0)
        
        baseline_time = baseline_metrics.get('computational_time', 0.0)
        improved_time = improvement_metrics.get('improved_computational_time', 0.0)
        
        # Performance improvements
        signal_count_change = improved_count - baseline_count
        signal_count_improvement = ((improved_count - baseline_count) / max(baseline_count, 1)) * 100
        
        confidence_change = improved_confidence - baseline_confidence
        confidence_improvement = ((improved_confidence - baseline_confidence) / max(baseline_confidence, 0.001)) * 100
        
        significance_change = improved_significance - baseline_significance
        significance_improvement = ((improved_significance - baseline_significance) / max(baseline_significance, 0.001)) * 100
        
        time_change = improved_time - baseline_time
        time_improvement = ((baseline_time - improved_time) / max(baseline_time, 0.001)) * 100  # Negative = faster
        
        comparison_results[domain] = {
            'baseline_signals': baseline_count,
            'improved_signals': improved_count,
            'signal_count_change': signal_count_change,
            'signal_count_improvement_pct': signal_count_improvement,
            
            'baseline_confidence': baseline_confidence,
            'improved_confidence': improved_confidence,
            'confidence_change': confidence_change,
            'confidence_improvement_pct': confidence_improvement,
            
            'baseline_significance': baseline_significance,
            'improved_significance': improved_significance,
            'significance_change': significance_change,
            'significance_improvement_pct': significance_improvement,
            
            'baseline_time': baseline_time,
            'improved_time': improved_time,
            'time_change': time_change,
            'time_improvement_pct': time_improvement,
            
            'baseline_years': baseline_metrics.get('signal_years', []),
            'improved_years': improvement_metrics.get('improved_signal_years', []),
            'improved_signal_types': list(set(improvement_metrics.get('signal_types', []))),
            'new_signals_detected': list(set(improvement_metrics.get('improved_signal_years', [])) - set(baseline_metrics.get('signal_years', []))),
            'lost_signals': list(set(baseline_metrics.get('signal_years', [])) - set(improvement_metrics.get('improved_signal_years', [])))
        }
        
        print(f"  {domain:30}")
        print(f"    Signals: {baseline_count:2d} → {improved_count:2d} ({signal_count_change:+d}, {signal_count_improvement:+6.1f}%)")
        print(f"    Confidence: {baseline_confidence:.3f} → {improved_confidence:.3f} ({confidence_change:+.3f}, {confidence_improvement:+6.1f}%)")
        print(f"    Significance: {baseline_significance:.3f} → {improved_significance:.3f} ({significance_change:+.3f}, {significance_improvement:+6.1f}%)")
        print(f"    Time: {baseline_time:.3f}s → {improved_time:.3f}s ({time_change:+.3f}s, {time_improvement:+6.1f}%)")
        if improvement_metrics.get('improved_signal_types'):
            print(f"    Signal types: {', '.join(improvement_metrics['improved_signal_types'])}")
        print()
    
    return comparison_results


def test_all_domains_semantic_improvement() -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
    """
    Test semantic improvement across all domains.
    
    Returns:
        Tuple of (improvement_results, comparison_analysis)
    """
    # Define test domains
    test_domains = [
        'natural_language_processing',
        'deep_learning', 
        'computer_vision',
        'machine_learning',
        'machine_translation'
    ]
    
    improvement_results = {}
    
    print("🚀 PHASE 10 IMPROVEMENT-002 TESTING")
    print("Testing data-driven semantic pattern discovery vs hardcoded baseline")
    print("=" * 80)
    
    for domain in test_domains:
        improvement_results[domain] = test_semantic_improvement(domain)
    
    # Compare with baseline
    baseline_file = "experiments/phase10_results/semantic_detection_baseline.json"
    comparison_analysis = compare_with_baseline(improvement_results, baseline_file)
    
    # Summary statistics
    print(f"\n📋 IMPROVEMENT-002 SUMMARY:")
    print("=" * 50)
    
    total_baseline_signals = 0
    total_improved_signals = 0
    successful_domains = 0
    domains_with_improvements = 0
    total_new_signal_types = set()
    
    for domain, comparison in comparison_analysis.items():
        if 'error' not in comparison:
            total_baseline_signals += comparison['baseline_signals']
            total_improved_signals += comparison['improved_signals']
            successful_domains += 1
            
            if comparison['signal_count_change'] > 0:
                domains_with_improvements += 1
            
            total_new_signal_types.update(comparison['improved_signal_types'])
    
    if successful_domains > 0:
        avg_signal_improvement = ((total_improved_signals - total_baseline_signals) / max(total_baseline_signals, 1)) * 100
        
        print(f"  Total signals: {total_baseline_signals} → {total_improved_signals} ({avg_signal_improvement:+.1f}%)")
        print(f"  Successful domains: {successful_domains}/{len(test_domains)}")
        print(f"  Domains with improvements: {domains_with_improvements}/{successful_domains}")
        print(f"  New signal types discovered: {', '.join(sorted(total_new_signal_types))}")
        
        # Identify best improvements
        best_domain = None
        best_improvement = -float('inf')
        for domain, comparison in comparison_analysis.items():
            if 'error' not in comparison and comparison['signal_count_improvement_pct'] > best_improvement:
                best_improvement = comparison['signal_count_improvement_pct']
                best_domain = domain
        
        if best_domain:
            print(f"  Best improvement: {best_domain} (+{best_improvement:.1f}% signals)")
    
    return improvement_results, comparison_analysis


def save_improvement_results(improvement_results: Dict[str, Dict[str, Any]], 
                           comparison_analysis: Dict[str, Any]) -> str:
    """
    Save semantic improvement test results.
    
    Args:
        improvement_results: Improvement test results
        comparison_analysis: Baseline comparison analysis
        
    Returns:
        Path to saved results file
    """
    from datetime import datetime
    
    # Create experiments directory if it doesn't exist
    Path("experiments/phase10_results").mkdir(parents=True, exist_ok=True)
    
    # Add metadata
    results_data = {
        'metadata': {
            'test_date': datetime.now().isoformat(),
            'test_type': 'semantic_improvement_validation',
            'algorithm_version': 'improved_data_driven_semantic_detection',
            'description': 'IMPROVEMENT-002 validation: Data-driven vs hardcoded semantic patterns',
            'improvements_tested': [
                'TF-IDF breakthrough term detection',
                'LDA topic modeling paradigm shifts',
                'N-gram evolution pattern analysis',
                'Semantic similarity drift detection'
            ]
        },
        'improvement_results': improvement_results,
        'baseline_comparison': comparison_analysis
    }
    
    # Save to file
    results_file = "experiments/phase10_results/semantic_improvement_results.json"
    with open(results_file, 'w') as f:
        json.dump(results_data, f, indent=2)
    
    print(f"\n💾 IMPROVEMENT-002 RESULTS SAVED: {results_file}")
    return results_file


if __name__ == "__main__":
    # Test semantic improvements
    improvement_results, comparison_analysis = test_all_domains_semantic_improvement()
    
    # Save results
    results_file = save_improvement_results(improvement_results, comparison_analysis)
    
    print(f"\n✅ IMPROVEMENT-002 TESTING COMPLETE")
    print(f"   Results saved to: {results_file}")
    print(f"   Data-driven semantic pattern discovery validation complete") 