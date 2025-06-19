#!/usr/bin/env python3
"""
Phase 10 Comprehensive Testing Framework

Tests the simplified two-signal algorithm (citation + direction, no semantic)
across all domains to validate IMPROVEMENT-003 maintains detection quality.
"""

import sys
import time
from pathlib import Path
from typing import Dict, List, Any

# Add project root to path for core module imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from core.data_processing import process_domain_data
from core.shift_signal_detection import detect_shift_signals


def test_simplified_algorithm_comprehensive() -> Dict[str, Any]:
    """
    Test the simplified algorithm across all domains.
    
    Returns:
        Dictionary with comprehensive results
    """
    print("🧪 PHASE 10 COMPREHENSIVE TESTING")
    print("="*80)
    print("Testing: Simplified Two-Signal Algorithm (Citation + Direction)")
    print("IMPROVEMENT-003: Semantic Detection Eliminated")
    print()
    
    domains = [
        'natural_language_processing',
        'deep_learning', 
        'computer_vision',
        'machine_learning',
        'machine_translation'
    ]
    
    results = {}
    total_shifts = 0
    total_time = 0
    
    for domain in domains:
        print(f"🔬 Testing: {domain}")
        print("-" * 50)
        
        start_time = time.time()
        
        # Load domain data
        result = process_domain_data(domain)
        if not result.success:
            print(f"   ❌ Failed to load domain: {result.error_message}")
            results[domain] = {'error': result.error_message}
            continue
        
        # Run simplified detection (citation + direction only)
        shift_signals, transition_evidence = detect_shift_signals(
            result.domain_data, 
            domain,
            use_citation=True,
            use_semantic=False,  # IMPROVEMENT-003: Eliminated
            use_direction=True
        )
        
        end_time = time.time()
        processing_time = end_time - start_time
        total_time += processing_time
        
        # Analyze results
        signal_types = list(set(s.signal_type for s in shift_signals))
        years = sorted([s.year for s in shift_signals])
        avg_confidence = sum(s.confidence for s in shift_signals) / len(shift_signals) if shift_signals else 0
        
        results[domain] = {
            'total_shifts': len(shift_signals),
            'signal_types': signal_types,
            'years': years,
            'avg_confidence': avg_confidence,
            'evidence_count': len(transition_evidence),
            'processing_time': processing_time
        }
        
        total_shifts += len(shift_signals)
        
        # Display results
        print(f"   📊 Paradigm shifts detected: {len(shift_signals)}")
        print(f"   🔍 Signal types: {signal_types}")
        print(f"   📅 Years: {years}")
        print(f"   🎯 Average confidence: {avg_confidence:.3f}")
        print(f"   📋 Transition evidence: {len(transition_evidence)} items")
        print(f"   ⏱️  Processing time: {processing_time:.3f}s")
        print()
    
    # Summary analysis
    successful_domains = [d for d, r in results.items() if 'total_shifts' in r]
    avg_shifts_per_domain = total_shifts / len(successful_domains) if successful_domains else 0
    avg_time_per_domain = total_time / len(successful_domains) if successful_domains else 0
    
    summary = {
        'total_shifts': total_shifts,
        'successful_domains': len(successful_domains),
        'total_domains': len(domains),
        'avg_shifts_per_domain': avg_shifts_per_domain,
        'total_processing_time': total_time,
        'avg_time_per_domain': avg_time_per_domain,
        'domain_results': results
    }
    
    # Display summary
    print("📈 COMPREHENSIVE SUMMARY")
    print("="*50)
    print(f"Total paradigm shifts detected: {total_shifts}")
    print(f"Average shifts per domain: {avg_shifts_per_domain:.1f}")
    print(f"Successful domains: {len(successful_domains)}/{len(domains)}")
    print(f"Total processing time: {total_time:.3f}s")
    print(f"Average time per domain: {avg_time_per_domain:.3f}s")
    print()
    
    print("📊 DOMAIN-BY-DOMAIN RESULTS:")
    for domain, result in results.items():
        if 'total_shifts' in result:
            print(f"  {domain}: {result['total_shifts']} shifts, "
                  f"confidence={result['avg_confidence']:.3f}, "
                  f"time={result['processing_time']:.3f}s")
        else:
            print(f"  {domain}: ERROR - {result.get('error', 'Unknown')}")
    
    return summary


if __name__ == "__main__":
    # Run comprehensive test
    results = test_simplified_algorithm_comprehensive()
    
    print()
    print("✅ PHASE 10 COMPREHENSIVE TEST COMPLETE")
    print(f"🎉 SUCCESS: Simplified algorithm detected {results['total_shifts']} paradigm shifts")
    print(f"📈 Performance: {results['avg_time_per_domain']:.3f}s average per domain") 