#!/usr/bin/env python3
"""
Phase 10 Improvement Testing Framework

Tests IMPROVEMENT-001: Citation Detection Fundamental Fixes
Compares improved algorithm against baseline performance.
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


def test_citation_detection_improvements(domain_name: str) -> Dict[str, Any]:
    """
    Test improved citation detection performance for a domain.
    
    Args:
        domain_name: Name of domain to test
        
    Returns:
        Dictionary with improvement test results
    """
    print(f"\n🔬 TESTING IMPROVEMENTS: {domain_name}")
    print("=" * 60)
    
    try:
        # Load domain data
        processing_result = process_domain_data(domain_name)
        if not processing_result.success:
            raise ValueError(f"Failed to load domain data: {processing_result.error_message}")
        domain_data = processing_result.domain_data
        
        # Start memory and time tracking
        tracemalloc.start()
        start_time = time.time()
        
        # Run IMPROVED citation detection (citation signals only)
        shift_signals, transition_evidence = detect_shift_signals(
            domain_data, 
            domain_name,
            use_citation=True,
            use_semantic=False,
            use_direction=False
        )
        
        # Stop tracking
        end_time = time.time()
        current_memory, peak_memory = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        # Extract citation signals (including validated ones)
        citation_signals = [s for s in shift_signals if 
                           s.signal_type == "citation_disruption" or 
                           s.signal_type.endswith("_citation_disruption")]
        
        # Calculate metrics
        improvement_metrics = {
            'domain_name': domain_name,
            'signal_count': len(citation_signals),
            'signal_years': [s.year for s in citation_signals],
            'confidence_scores': [s.confidence for s in citation_signals],
            'evidence_strength': [s.evidence_strength for s in citation_signals],
            'paradigm_significance': [s.paradigm_significance for s in citation_signals],
            'computational_time': end_time - start_time,
            'peak_memory_mb': peak_memory / 1024 / 1024,
            'total_papers': len(domain_data.papers),
            'year_range': domain_data.year_range,
            'years_span': domain_data.year_range[1] - domain_data.year_range[0] + 1,
            'improvement_version': 'phase10_improvement_001'
        }
        
        # Signal quality analysis
        if citation_signals:
            improvement_metrics.update({
                'avg_confidence': sum(improvement_metrics['confidence_scores']) / len(citation_signals),
                'max_confidence': max(improvement_metrics['confidence_scores']),
                'min_confidence': min(improvement_metrics['confidence_scores']),
                'avg_evidence_strength': sum(improvement_metrics['evidence_strength']) / len(citation_signals),
                'avg_paradigm_significance': sum(improvement_metrics['paradigm_significance']) / len(citation_signals)
            })
        else:
            improvement_metrics.update({
                'avg_confidence': 0.0,
                'max_confidence': 0.0,
                'min_confidence': 0.0,
                'avg_evidence_strength': 0.0,
                'avg_paradigm_significance': 0.0
            })
        
        print(f"  📊 IMPROVEMENT RESULTS:")
        print(f"      Citation signals detected: {improvement_metrics['signal_count']}")
        print(f"      Signal years: {improvement_metrics['signal_years']}")
        print(f"      Average confidence: {improvement_metrics['avg_confidence']:.3f}")
        print(f"      Computational time: {improvement_metrics['computational_time']:.3f}s")
        print(f"      Peak memory: {improvement_metrics['peak_memory_mb']:.1f}MB")
        
        return improvement_metrics
        
    except Exception as e:
        print(f"  ❌ Error testing improvements for {domain_name}: {e}")
        return {
            'domain_name': domain_name,
            'error': str(e),
            'signal_count': -1,
            'computational_time': -1,
            'peak_memory_mb': -1,
            'improvement_version': 'phase10_improvement_001'
        }


def compare_with_baseline(improvement_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compare improvement results with baseline results.
    
    Args:
        improvement_results: Results from improved algorithm
        
    Returns:
        Dictionary with comparison analysis
    """
    print(f"\n📊 IMPROVEMENT vs BASELINE COMPARISON")
    print("=" * 60)
    
    # Load baseline results
    baseline_file = "experiments/phase10_results/citation_detection_baseline.json"
    try:
        with open(baseline_file, 'r') as f:
            baseline_data = json.load(f)
        baseline_results = baseline_data['results']
    except Exception as e:
        print(f"❌ Could not load baseline results: {e}")
        return {'error': str(e)}
    
    # Comparison analysis
    comparison = {
        'methodology': 'IMPROVEMENT-001: Citation Detection Fundamental Fixes',
        'improvements_implemented': [
            'Removed useless penalty optimization (proven p=1.0000)',
            'Sparse time series (eliminates signal dilution)',
            'Lower confidence thresholds (0.03 vs 0.1+ baseline)',
            'Enhanced sensitivity (2.0x vs 1.2x confidence scaling)',
            'Reduced minimum gap (3 vs 4 years)'
        ],
        'domain_comparisons': {},
        'overall_improvements': {}
    }
    
    # Domain-by-domain comparison
    total_baseline_signals = 0
    total_improved_signals = 0
    total_baseline_time = 0
    total_improved_time = 0
    successful_domains = 0
    
    for domain in improvement_results.keys():
        if 'error' in improvement_results[domain] or 'error' in baseline_results.get(domain, {}):
            continue
            
        baseline = baseline_results[domain]
        improved = improvement_results[domain]
        
        signal_improvement = improved['signal_count'] - baseline['signal_count']
        time_change = improved['computational_time'] - baseline['computational_time']
        memory_change = improved['peak_memory_mb'] - baseline['peak_memory_mb']
        
        domain_comparison = {
            'baseline_signals': baseline['signal_count'],
            'improved_signals': improved['signal_count'],
            'signal_improvement': signal_improvement,
            'signal_improvement_pct': (signal_improvement / max(baseline['signal_count'], 1)) * 100,
            'baseline_time': baseline['computational_time'],
            'improved_time': improved['computational_time'],
            'time_change': time_change,
            'time_change_pct': (time_change / baseline['computational_time']) * 100 if baseline['computational_time'] > 0 else 0,
            'memory_change_mb': memory_change,
            'baseline_years': baseline.get('signal_years', []),
            'improved_years': improved.get('signal_years', [])
        }
        
        comparison['domain_comparisons'][domain] = domain_comparison
        
        total_baseline_signals += baseline['signal_count']
        total_improved_signals += improved['signal_count']
        total_baseline_time += baseline['computational_time']
        total_improved_time += improved['computational_time']
        successful_domains += 1
        
        print(f"  {domain:30}")
        print(f"    Signals: {baseline['signal_count']:2d} → {improved['signal_count']:2d} ({signal_improvement:+d}, {domain_comparison['signal_improvement_pct']:+.0f}%)")
        print(f"    Time:    {baseline['computational_time']:.3f}s → {improved['computational_time']:.3f}s ({time_change:+.3f}s, {domain_comparison['time_change_pct']:+.1f}%)")
        
        if improved['signal_count'] > 0:
            print(f"    Years:   {improved['signal_years']}")
            print(f"    Avg Confidence: {improved['avg_confidence']:.3f}")
    
    # Overall improvement statistics
    if successful_domains > 0:
        total_signal_improvement = total_improved_signals - total_baseline_signals
        total_time_change = total_improved_time - total_baseline_time
        
        comparison['overall_improvements'] = {
            'total_baseline_signals': total_baseline_signals,
            'total_improved_signals': total_improved_signals,
            'total_signal_improvement': total_signal_improvement,
            'avg_signal_improvement': total_signal_improvement / successful_domains,
            'total_time_change_s': total_time_change,
            'avg_time_change_s': total_time_change / successful_domains,
            'domains_tested': successful_domains,
            'domains_with_improvements': sum(1 for d in comparison['domain_comparisons'].values() 
                                           if d['signal_improvement'] > 0)
        }
        
        print(f"\n  📋 OVERALL IMPROVEMENTS:")
        print(f"    Total signals: {total_baseline_signals} → {total_improved_signals} ({total_signal_improvement:+d})")
        print(f"    Average per domain: {total_signal_improvement/successful_domains:+.1f} signals")
        print(f"    Domains with improvements: {comparison['overall_improvements']['domains_with_improvements']}/{successful_domains}")
        print(f"    Total time change: {total_time_change:+.3f}s")
        
        # Success assessment
        if total_signal_improvement > 0:
            print(f"    🎉 SUCCESS: IMPROVEMENT-001 provides measurable benefits!")
        else:
            print(f"    ⚠️  No improvement detected - further analysis needed")
    
    return comparison


def test_all_improvements() -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
    """
    Test improvements across all domains and compare with baseline.
    
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
    
    print("🚀 PHASE 10 IMPROVEMENT TESTING")
    print("Testing IMPROVEMENT-001: Citation Detection Fundamental Fixes")
    print("=" * 80)
    
    for domain in test_domains:
        improvement_results[domain] = test_citation_detection_improvements(domain)
    
    # Compare with baseline
    comparison_analysis = compare_with_baseline(improvement_results)
    
    return improvement_results, comparison_analysis


def save_improvement_results(improvement_results: Dict[str, Dict[str, Any]], 
                           comparison_analysis: Dict[str, Any]) -> str:
    """
    Save improvement test results.
    
    Args:
        improvement_results: Improvement test results
        comparison_analysis: Comparison with baseline
        
    Returns:
        Path to saved results file
    """
    from datetime import datetime
    
    # Create results directory if it doesn't exist
    Path("experiments/phase10_results").mkdir(parents=True, exist_ok=True)
    
    # Add metadata
    results_data = {
        'metadata': {
            'test_date': datetime.now().isoformat(),
            'test_type': 'improvement_001_citation_detection_fixes',
            'algorithm_version': 'phase10_improved',
            'description': 'IMPROVEMENT-001: Citation Detection Fundamental Fixes',
            'baseline_comparison': True
        },
        'improvement_results': improvement_results,
        'comparison_analysis': comparison_analysis
    }
    
    # Save to file
    results_file = "experiments/phase10_results/improvement_001_test_results.json"
    with open(results_file, 'w') as f:
        json.dump(results_data, f, indent=2)
    
    print(f"\n💾 IMPROVEMENT RESULTS SAVED: {results_file}")
    return results_file


if __name__ == "__main__":
    # Test improvements
    improvement_results, comparison_analysis = test_all_improvements()
    
    # Save results
    results_file = save_improvement_results(improvement_results, comparison_analysis)
    
    print(f"\n✅ IMPROVEMENT-001 TESTING COMPLETE")
    print(f"   Results saved to: {results_file}")
    
    # Quick success assessment
    if 'error' not in comparison_analysis:
        overall = comparison_analysis.get('overall_improvements', {})
        signal_improvement = overall.get('total_signal_improvement', 0)
        
        if signal_improvement > 0:
            print(f"   🎉 SUCCESS: +{signal_improvement} citation signals detected!")
            print(f"   📈 Ready for next improvement phase")
        else:
            print(f"   ⚠️  No improvement - need further analysis")
    else:
        print(f"   ❌ Testing failed - check error logs") 