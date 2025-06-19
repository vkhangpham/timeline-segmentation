#!/usr/bin/env python3
"""
Citation Detection Improvement Experiment (Phase 10+)

This experiment compares the original citation detection method with the improved 
implementation that addresses under-sensitivity issues identified in Phase 10 analysis.

Key refinements tested:
1. Square root scaling instead of log scaling (less aggressive)
2. Lower Cohen's d threshold (0.3 instead of 0.5)
3. More lenient domain-specific confidence floors
4. Hybrid approach preserving original detection capability
"""

import sys
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
import pandas as pd

# Add the project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Add current directory to path
sys.path.insert(0, str(Path.cwd()))

from core.data_processing import process_domain_data
from core.shift_signal_detection import (
    detect_shift_signals, 
    detect_citation_structural_breaks
)
from core.shift_signal_detection_original import detect_citation_structural_breaks


def run_citation_comparison_experiment() -> Dict:
    """
    Run comparative experiment between original and improved citation detection.
    
    Returns:
        Dictionary containing comprehensive experiment results
    """
    print("🚀 CITATION DETECTION IMPROVEMENT EXPERIMENT")
    print("=" * 60)
    print("Comparing original vs improved citation detection methods")
    print("Focus domains: Applied Mathematics, Computer Science, Machine Learning, Deep Learning")
    print("=" * 60)
    
    # Target domains for focused analysis
    target_domains = [
        'applied_mathematics',
        'computer_science', 
        'machine_learning',
        'deep_learning'
    ]
    
    experiment_results = {
        'metadata': {
            'experiment_name': 'Citation Detection Improvement Comparison',
            'experiment_date': datetime.now().isoformat(),
            'target_domains': target_domains,
            'methods_compared': ['original', 'improved'],
            'focus': 'Citation signal detection under-sensitivity fix'
        },
        'domain_results': {},
        'summary_statistics': {},
        'improvement_analysis': {}
    }
    
    # Run comparison for each target domain
    for domain_name in target_domains:
        print(f"\n🔍 ANALYZING DOMAIN: {domain_name}")
        print("-" * 50)
        
        try:
            # Load domain data
            result = process_domain_data(domain_name)
            if not result.success:
                print(f"   ❌ Error loading {domain_name}: {result.error_message}")
                experiment_results['domain_results'][domain_name] = {
                    'error': result.error_message,
                    'status': 'failed'
                }
                continue
            
            domain_data = result.domain_data
            print(f"   📊 Loaded {len(domain_data.papers)} papers, {len(domain_data.citations)} citations")
            
            # Method 1: Original citation detection
            print(f"\n   🔬 ORIGINAL METHOD:")
            start_time = time.time()
            original_signals = detect_citation_structural_breaks(domain_data, domain_name)
            original_runtime = time.time() - start_time
            
            print(f"   ⏱️  Original runtime: {original_runtime:.3f}s")
            print(f"   📈 Original signals detected: {len(original_signals)}")
            
            # Method 2: Refined citation detection  
            print(f"\n   🔧 REFINED METHOD:")
            start_time = time.time()
            refined_signals = detect_citation_structural_breaks(domain_data, domain_name)
            refined_runtime = time.time() - start_time
            
            print(f"   ⏱️  Refined runtime: {refined_runtime:.3f}s")
            print(f"   📈 Refined signals detected: {len(refined_signals)}")
            
            # Analysis
            improvement_count = len(refined_signals) - len(original_signals)
            improvement_percent = (improvement_count / max(len(original_signals), 1)) * 100
            
            print(f"\n   📊 COMPARISON RESULTS:")
            print(f"   🔢 Signal count improvement: +{improvement_count} signals ({improvement_percent:+.1f}%)")
            print(f"   ⚡ Runtime change: {refined_runtime/max(original_runtime, 0.001):.2f}x")
            
            # Detailed signal analysis
            original_years = [s.year for s in original_signals]
            refined_years = [s.year for s in refined_signals]
            
            new_detections = [year for year in refined_years if year not in original_years]
            preserved_detections = [year for year in refined_years if year in original_years]
            lost_detections = [year for year in original_years if year not in refined_years]
            
            print(f"   🆕 New detections: {new_detections}")
            print(f"   ✅ Preserved detections: {preserved_detections}")
            print(f"   ❌ Lost detections: {lost_detections}")
            
            # Store detailed results
            experiment_results['domain_results'][domain_name] = {
                'original': {
                    'signal_count': len(original_signals),
                    'runtime_seconds': original_runtime,
                    'detected_years': original_years,
                    'signals': [serialize_signal(s) for s in original_signals]
                },
                'refined': {
                    'signal_count': len(refined_signals),
                    'runtime_seconds': refined_runtime,
                    'detected_years': refined_years,
                    'signals': [serialize_signal(s) for s in refined_signals]
                },
                'comparison': {
                    'signal_count_change': improvement_count,
                    'signal_count_change_percent': improvement_percent,
                    'runtime_ratio': refined_runtime / max(original_runtime, 0.001),
                    'new_detections': new_detections,
                    'preserved_detections': preserved_detections,
                    'lost_detections': lost_detections,
                    'detection_preservation_rate': len(preserved_detections) / max(len(original_years), 1),
                    'detection_expansion_rate': len(new_detections) / max(len(original_years), 1)
                }
            }
            
        except Exception as e:
            print(f"   ⚠️ Error processing domain {domain_name}: {e}")
            experiment_results['domain_results'][domain_name] = {
                'error': str(e),
                'status': 'failed'
            }
    
    # Calculate summary statistics
    successful_domains = [d for d in experiment_results['domain_results'].values() 
                         if 'error' not in d]
    
    if successful_domains:
        original_total = sum(d['original']['signal_count'] for d in successful_domains)
        refined_total = sum(d['refined']['signal_count'] for d in successful_domains)
        
        experiment_results['summary_statistics'] = {
            'successful_domains': len(successful_domains),
            'total_domains_tested': len(target_domains),
            'original_total_signals': original_total,
            'refined_total_signals': refined_total,
            'total_improvement': refined_total - original_total,
            'total_improvement_percent': ((refined_total - original_total) / max(original_total, 1)) * 100,
            'average_signals_per_domain_original': original_total / len(successful_domains),
            'average_signals_per_domain_refined': refined_total / len(successful_domains),
            'average_runtime_original': sum(d['original']['runtime_seconds'] for d in successful_domains) / len(successful_domains),
            'average_runtime_refined': sum(d['refined']['runtime_seconds'] for d in successful_domains) / len(successful_domains)
        }
        
        # Improvement analysis
        experiment_results['improvement_analysis'] = {
            'domains_with_improvement': len([d for d in successful_domains if d['comparison']['signal_count_change'] > 0]),
            'domains_with_decline': len([d for d in successful_domains if d['comparison']['signal_count_change'] < 0]),
            'domains_unchanged': len([d for d in successful_domains if d['comparison']['signal_count_change'] == 0]),
            'max_improvement': max([d['comparison']['signal_count_change'] for d in successful_domains]),
            'min_improvement': min([d['comparison']['signal_count_change'] for d in successful_domains]),
            'average_preservation_rate': sum(d['comparison']['detection_preservation_rate'] for d in successful_domains) / len(successful_domains),
            'average_expansion_rate': sum(d['comparison']['detection_expansion_rate'] for d in successful_domains) / len(successful_domains)
        }
    
    return experiment_results


def serialize_signal(signal) -> Dict:
    """Convert a ShiftSignal to a serializable dictionary."""
    return {
        'year': signal.year,
        'confidence': signal.confidence,
        'signal_type': signal.signal_type,
        'evidence_strength': signal.evidence_strength,
        'supporting_evidence': list(signal.supporting_evidence),
        'transition_description': signal.transition_description,
        'paradigm_significance': signal.paradigm_significance
    }


def save_experiment_results(results: Dict, output_dir: str = "experiments/phase10/results") -> str:
    """Save experiment results to JSON file."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    filename = f"citation_improvement_experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    filepath = output_path / filename
    
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 EXPERIMENT RESULTS SAVED:")
    print(f"   📁 File: {filepath}")
    
    return str(filepath)


def generate_experiment_summary(results: Dict) -> None:
    """Generate a comprehensive summary of experiment results."""
    print("\n" + "=" * 60)
    print("🧪 CITATION IMPROVEMENT EXPERIMENT SUMMARY")
    print("=" * 60)
    
    stats = results.get('summary_statistics', {})
    analysis = results.get('improvement_analysis', {})
    
    print(f"📊 OVERALL PERFORMANCE:")
    print(f"   Domains tested: {stats.get('total_domains_tested', 0)}")
    print(f"   Successful analyses: {stats.get('successful_domains', 0)}")
    print(f"   Original total signals: {stats.get('original_total_signals', 0)}")
    print(f"   Refined total signals: {stats.get('refined_total_signals', 0)}")
    print(f"   Total improvement: +{stats.get('total_improvement', 0)} signals ({stats.get('total_improvement_percent', 0):+.1f}%)")
    
    print(f"\n🔬 DETECTION ANALYSIS:")
    print(f"   Domains with improvement: {analysis.get('domains_with_improvement', 0)}")
    print(f"   Domains with decline: {analysis.get('domains_with_decline', 0)}")
    print(f"   Domains unchanged: {analysis.get('domains_unchanged', 0)}")
    print(f"   Maximum improvement: +{analysis.get('max_improvement', 0)} signals")
    print(f"   Average preservation rate: {analysis.get('average_preservation_rate', 0):.1%}")
    print(f"   Average expansion rate: {analysis.get('average_expansion_rate', 0):.1%}")
    
    print(f"\n⚡ PERFORMANCE METRICS:")
    print(f"   Average runtime (original): {stats.get('average_runtime_original', 0):.3f}s")
    print(f"   Average runtime (refined): {stats.get('average_runtime_refined', 0):.3f}s")
    
    # Domain-by-domain breakdown
    print(f"\n📋 DOMAIN-BY-DOMAIN RESULTS:")
    for domain_name, domain_results in results.get('domain_results', {}).items():
        if 'error' not in domain_results:
            comp = domain_results['comparison']
            print(f"   {domain_name}:")
            print(f"     Signals: {domain_results['original']['signal_count']} → {domain_results['refined']['signal_count']} ({comp['signal_count_change']:+d})")
            print(f"     New detections: {comp['new_detections']}")
            if comp['lost_detections']:
                print(f"     Lost detections: {comp['lost_detections']}")


def main():
    """Main experiment execution."""
    print("Starting Citation Detection Improvement Experiment...")
    
    # Run the comparative experiment
    experiment_results = run_citation_comparison_experiment()
    
    # Save results
    results_file = save_experiment_results(experiment_results)
    
    # Generate summary
    generate_experiment_summary(experiment_results)
    
    print(f"\n✅ EXPERIMENT COMPLETED")
    print(f"📁 Results saved to: {results_file}")
    print("🔍 Review the detailed logs above for analysis insights")


if __name__ == "__main__":
    main() 