#!/usr/bin/env python3
"""
Phase 10 Baseline Measurement Framework

Captures current citation detection performance before implementing improvements.
This provides the baseline for controlled experiments to validate each improvement.
"""

import sys
import time
import json
import tracemalloc
from pathlib import Path
from typing import Dict, List, Any
from collections import defaultdict

# Add project root to path for core module imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from core.data_processing import process_domain_data
from core.shift_signal_detection import detect_shift_signals


def measure_citation_detection_performance(domain_name: str) -> Dict[str, Any]:
    """
    Measure baseline citation detection performance for a domain.
    
    Args:
        domain_name: Name of domain to test
        
    Returns:
        Dictionary with performance metrics
    """
    print(f"\n🔍 MEASURING BASELINE: {domain_name}")
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
        
        # Run current citation detection (citation signals only)
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
        
        # Extract citation signals
        citation_signals = [s for s in shift_signals if s.signal_type == "citation_disruption"]
        
        # Calculate metrics
        performance_metrics = {
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
            'years_span': domain_data.year_range[1] - domain_data.year_range[0] + 1
        }
        
        # Signal quality analysis
        if citation_signals:
            performance_metrics.update({
                'avg_confidence': sum(performance_metrics['confidence_scores']) / len(citation_signals),
                'max_confidence': max(performance_metrics['confidence_scores']),
                'min_confidence': min(performance_metrics['confidence_scores']),
                'avg_evidence_strength': sum(performance_metrics['evidence_strength']) / len(citation_signals),
                'avg_paradigm_significance': sum(performance_metrics['paradigm_significance']) / len(citation_signals)
            })
        else:
            performance_metrics.update({
                'avg_confidence': 0.0,
                'max_confidence': 0.0,
                'min_confidence': 0.0,
                'avg_evidence_strength': 0.0,
                'avg_paradigm_significance': 0.0
            })
        
        print(f"  📊 BASELINE RESULTS:")
        print(f"      Citation signals detected: {performance_metrics['signal_count']}")
        print(f"      Signal years: {performance_metrics['signal_years']}")
        print(f"      Average confidence: {performance_metrics['avg_confidence']:.3f}")
        print(f"      Computational time: {performance_metrics['computational_time']:.3f}s")
        print(f"      Peak memory: {performance_metrics['peak_memory_mb']:.1f}MB")
        
        return performance_metrics
        
    except Exception as e:
        print(f"  ❌ Error measuring baseline for {domain_name}: {e}")
        return {
            'domain_name': domain_name,
            'error': str(e),
            'signal_count': -1,
            'computational_time': -1,
            'peak_memory_mb': -1
        }


def measure_all_domains_baseline() -> Dict[str, Dict[str, Any]]:
    """
    Measure baseline performance across all domains.
    
    Returns:
        Dictionary mapping domain names to performance metrics
    """
    # Define test domains (focus on domains with known citation issues)
    test_domains = [
        'natural_language_processing',
        'deep_learning', 
        'computer_vision',
        'machine_learning',
        'machine_translation'
    ]
    
    baseline_results = {}
    
    print("🚀 PHASE 10 BASELINE MEASUREMENT")
    print("Measuring current citation detection performance before improvements")
    print("=" * 80)
    
    for domain in test_domains:
        baseline_results[domain] = measure_citation_detection_performance(domain)
    
    # Summary statistics
    print(f"\n📋 BASELINE SUMMARY:")
    print("=" * 50)
    
    total_signals = 0
    domains_with_signals = 0
    total_time = 0
    
    for domain, metrics in baseline_results.items():
        if 'error' not in metrics:
            signal_count = metrics['signal_count']
            comp_time = metrics['computational_time']
            
            print(f"  {domain:30}: {signal_count:2d} signals, {comp_time:.3f}s")
            
            total_signals += signal_count
            total_time += comp_time
            if signal_count > 0:
                domains_with_signals += 1
    
    print(f"  {'TOTAL':30}: {total_signals:2d} signals, {total_time:.3f}s")
    print(f"  Domains with citation signals: {domains_with_signals}/{len(test_domains)}")
    
    # Identify domains with zero citation signals (improvement targets)
    zero_signal_domains = [domain for domain, metrics in baseline_results.items() 
                          if metrics.get('signal_count', -1) == 0]
    
    if zero_signal_domains:
        print(f"  ⚠️  Zero citation signals: {', '.join(zero_signal_domains)}")
        print(f"     These are primary targets for IMPROVEMENT-001")
    
    return baseline_results


def save_baseline_results(baseline_results: Dict[str, Dict[str, Any]]) -> str:
    """
    Save baseline results for comparison with improvements.
    
    Args:
        baseline_results: Baseline measurement results
        
    Returns:
        Path to saved baseline file
    """
    from datetime import datetime
    
    # Create experiments directory if it doesn't exist
    Path("experiments/phase10_results").mkdir(parents=True, exist_ok=True)
    
    # Add metadata
    baseline_data = {
        'metadata': {
            'measurement_date': datetime.now().isoformat(),
            'measurement_type': 'citation_detection_baseline',
            'algorithm_version': 'original_phase9',
            'description': 'Baseline measurement before Phase 10 improvements',
            'focus': 'citation_disruption_detection_only'
        },
        'results': baseline_results
    }
    
    # Save to file
    baseline_file = "experiments/phase10_results/citation_detection_baseline.json"
    with open(baseline_file, 'w') as f:
        json.dump(baseline_data, f, indent=2)
    
    print(f"\n💾 BASELINE SAVED: {baseline_file}")
    return baseline_file


if __name__ == "__main__":
    # Measure baseline performance
    baseline_results = measure_all_domains_baseline()
    
    # Save results
    baseline_file = save_baseline_results(baseline_results)
    
    print(f"\n✅ BASELINE MEASUREMENT COMPLETE")
    print(f"   Results saved to: {baseline_file}")
    print(f"   Ready to implement IMPROVEMENT-001: Citation Detection Fixes") 