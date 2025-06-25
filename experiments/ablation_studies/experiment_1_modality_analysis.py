#!/usr/bin/env python3
"""
Experiment 1: Signal Detection Modality Analysis

Research Question: How much does each detection modality contribute to final performance?

This experiment evaluates three detection configurations:
1. Direction-only detection (no citation validation)
2. Citation-only detection (gradient analysis alone)  
3. Combined detection (current fusion mechanism)

Follows functional programming principles with fail-fast error handling.
"""

import os
import sys
import time
from typing import Dict, List, Tuple

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from experiment_utils import (
    load_test_domains, evaluate_segmentation_configuration,
    ExperimentResult, save_experiment_results, print_experiment_summary,
    calculate_statistical_significance, TEST_DOMAINS
)
from core.algorithm_config import AlgorithmConfig


def run_modality_analysis_single_domain(
    domain_name: str,
    domain_data,
    base_config: AlgorithmConfig
) -> List[ExperimentResult]:
    """
    Run modality analysis for a single domain.
    
    Pure function that evaluates all modality conditions for one domain.
    
    Args:
        domain_name: Name of the domain being tested
        domain_data: DomainData object for the domain
        base_config: Base algorithm configuration
        
    Returns:
        List of ExperimentResult objects for all conditions
        
    Raises:
        ValueError: If any evaluation fails
    """
    print(f"\n🔬 Running modality analysis for {domain_name}")
    
    results = []
    
    # Condition 1: Direction-only detection
    print("  🎯 Testing direction-only detection...")
    start_time = time.time()
    
    try:
        score, metrics = evaluate_segmentation_configuration(
            domain_data, base_config, use_citation=False, use_direction=True
        )
        execution_time = time.time() - start_time
        
        results.append(ExperimentResult(
            experiment_name="modality_analysis",
            domain=domain_name,
            condition="direction_only",
            score=score,
            consensus_score=metrics['consensus_score'],
            difference_score=metrics['difference_score'],
            num_segments=metrics['num_segments'],
            execution_time=execution_time,
            metadata={
                'use_citation': False,
                'use_direction': True,
                'segment_sizes': metrics['segment_sizes'],
                'detailed_metrics': metrics
            }
        ))
        
        print(f"    ✅ Direction-only: score={score:.3f} ({metrics['num_segments']} segments, {execution_time:.1f}s)")
        
    except Exception as e:
        raise ValueError(f"Direction-only evaluation failed for {domain_name}: {str(e)}")
    
    # Condition 2: Citation-only detection  
    print("  📈 Testing citation-only detection...")
    start_time = time.time()
    
    try:
        score, metrics = evaluate_segmentation_configuration(
            domain_data, base_config, use_citation=True, use_direction=False
        )
        execution_time = time.time() - start_time
        
        results.append(ExperimentResult(
            experiment_name="modality_analysis",
            domain=domain_name,
            condition="citation_only",
            score=score,
            consensus_score=metrics['consensus_score'],
            difference_score=metrics['difference_score'],
            num_segments=metrics['num_segments'],
            execution_time=execution_time,
            metadata={
                'use_citation': True,
                'use_direction': False,
                'segment_sizes': metrics['segment_sizes'],
                'detailed_metrics': metrics
            }
        ))
        
        print(f"    ✅ Citation-only: score={score:.3f} ({metrics['num_segments']} segments, {execution_time:.1f}s)")
        
    except Exception as e:
        raise ValueError(f"Citation-only evaluation failed for {domain_name}: {str(e)}")
    
    # Condition 3: Combined detection (baseline)
    print("  🔄 Testing combined detection...")
    start_time = time.time()
    
    try:
        score, metrics = evaluate_segmentation_configuration(
            domain_data, base_config, use_citation=True, use_direction=True
        )
        execution_time = time.time() - start_time
        
        results.append(ExperimentResult(
            experiment_name="modality_analysis",
            domain=domain_name,
            condition="combined",
            score=score,
            consensus_score=metrics['consensus_score'],
            difference_score=metrics['difference_score'],
            num_segments=metrics['num_segments'],
            execution_time=execution_time,
            metadata={
                'use_citation': True,
                'use_direction': True,
                'segment_sizes': metrics['segment_sizes'],
                'detailed_metrics': metrics
            }
        ))
        
        print(f"    ✅ Combined: score={score:.3f} ({metrics['num_segments']} segments, {execution_time:.1f}s)")
        
    except Exception as e:
        raise ValueError(f"Combined evaluation failed for {domain_name}: {str(e)}")
    
    return results


def analyze_modality_contributions(results: List[ExperimentResult]) -> Dict[str, any]:
    """
    Analyze the contribution of each modality across domains.
    
    Pure function that calculates modality-specific insights from results.
    
    Args:
        results: List of all experiment results
        
    Returns:
        Dictionary with analysis insights
    """
    # Group results by domain and condition
    domain_results = {}
    for result in results:
        if result.domain not in domain_results:
            domain_results[result.domain] = {}
        domain_results[result.domain][result.condition] = result
    
    # Calculate relative contributions
    analysis = {
        'domain_analysis': {},
        'overall_patterns': {},
        'statistical_tests': {}
    }
    
    direction_scores = []
    citation_scores = []
    combined_scores = []
    
    for domain, conditions in domain_results.items():
        if all(cond in conditions for cond in ['direction_only', 'citation_only', 'combined']):
            dir_score = conditions['direction_only'].score
            cit_score = conditions['citation_only'].score
            comb_score = conditions['combined'].score
            
            # Calculate improvements
            dir_vs_cit = ((dir_score - cit_score) / cit_score * 100) if cit_score > 0 else 0
            comb_vs_dir = ((comb_score - dir_score) / dir_score * 100) if dir_score > 0 else 0
            comb_vs_cit = ((comb_score - cit_score) / cit_score * 100) if cit_score > 0 else 0
            
            analysis['domain_analysis'][domain] = {
                'direction_score': dir_score,
                'citation_score': cit_score,
                'combined_score': comb_score,
                'direction_vs_citation_improvement': dir_vs_cit,
                'combined_vs_direction_improvement': comb_vs_dir,
                'combined_vs_citation_improvement': comb_vs_cit,
                'best_modality': max([('direction', dir_score), ('citation', cit_score), ('combined', comb_score)], 
                                   key=lambda x: x[1])[0]
            }
            
            direction_scores.append(dir_score)
            citation_scores.append(cit_score)
            combined_scores.append(comb_score)
    
    # Overall patterns
    if direction_scores and citation_scores and combined_scores:
        analysis['overall_patterns'] = {
            'mean_direction_score': float(sum(direction_scores) / len(direction_scores)),
            'mean_citation_score': float(sum(citation_scores) / len(citation_scores)),
            'mean_combined_score': float(sum(combined_scores) / len(combined_scores)),
            'direction_vs_citation_preference': 'direction' if sum(direction_scores) > sum(citation_scores) else 'citation',
            'fusion_adds_value': sum(combined_scores) > max(sum(direction_scores), sum(citation_scores))
        }
        
        # Statistical significance tests
        try:
            analysis['statistical_tests']['direction_vs_citation'] = calculate_statistical_significance(
                direction_scores, citation_scores
            )
            analysis['statistical_tests']['combined_vs_direction'] = calculate_statistical_significance(
                combined_scores, direction_scores
            )
            analysis['statistical_tests']['combined_vs_citation'] = calculate_statistical_significance(
                combined_scores, citation_scores
            )
        except Exception as e:
            analysis['statistical_tests']['error'] = str(e)
    
    return analysis


def run_modality_analysis_experiment() -> str:
    """
    Run the complete modality analysis experiment.
    
    Main function that orchestrates the full experiment across all test domains.
    
    Returns:
        Path to saved results file
        
    Raises:
        ValueError: If experiment fails
    """
    print("🧪 EXPERIMENT 1: SIGNAL DETECTION MODALITY ANALYSIS")
    print("=" * 70)
    print("Research Question: How much does each detection modality contribute to final performance?")
    print(f"Test domains: {', '.join(TEST_DOMAINS)}")
    print("\nConditions:")
    print("  1. Direction-only detection (no citation validation)")
    print("  2. Citation-only detection (gradient analysis alone)")
    print("  3. Combined detection (current fusion mechanism)")
    
    # Load test domains
    try:
        domain_data = load_test_domains()
    except Exception as e:
        raise ValueError(f"Failed to load test domains: {str(e)}")
    
    # Use default algorithm configuration as baseline
    base_config = AlgorithmConfig(granularity=3)
    
    # Run experiment for all domains
    all_results = []
    for domain_name in TEST_DOMAINS:
        try:
            domain_results = run_modality_analysis_single_domain(
                domain_name, domain_data[domain_name], base_config
            )
            all_results.extend(domain_results)
        except Exception as e:
            raise ValueError(f"Experiment failed for domain {domain_name}: {str(e)}")
    
    # Analyze results
    analysis = analyze_modality_contributions(all_results)
    
    # Print summary
    print_experiment_summary(all_results, "modality_analysis")
    
    # Print detailed analysis
    print("\n📊 MODALITY CONTRIBUTION ANALYSIS")
    print("=" * 50)
    
    for domain, domain_analysis in analysis['domain_analysis'].items():
        print(f"\n🔍 {domain}:")
        print(f"   Direction: {domain_analysis['direction_score']:.3f}")
        print(f"   Citation:  {domain_analysis['citation_score']:.3f}")
        print(f"   Combined:  {domain_analysis['combined_score']:.3f}")
        print(f"   Best: {domain_analysis['best_modality']}")
        print(f"   Direction vs Citation: {domain_analysis['direction_vs_citation_improvement']:+.1f}%")
        print(f"   Combined vs Direction: {domain_analysis['combined_vs_direction_improvement']:+.1f}%")
    
    if 'overall_patterns' in analysis:
        patterns = analysis['overall_patterns']
        print(f"\n📈 OVERALL PATTERNS:")
        print(f"   Mean scores - Direction: {patterns['mean_direction_score']:.3f}, "
              f"Citation: {patterns['mean_citation_score']:.3f}, "
              f"Combined: {patterns['mean_combined_score']:.3f}")
        print(f"   Preference: {patterns['direction_vs_citation_preference']}")
        print(f"   Fusion adds value: {patterns['fusion_adds_value']}")
    
    # Save results
    additional_metadata = {
        'experiment_description': 'Signal detection modality analysis comparing direction-only, citation-only, and combined detection',
        'test_domains': TEST_DOMAINS,
        'base_config': base_config.__dict__,
        'analysis': analysis
    }
    
    try:
        results_file = save_experiment_results(all_results, "modality_analysis", additional_metadata)
        return results_file
    except Exception as e:
        raise ValueError(f"Failed to save results: {str(e)}")


if __name__ == "__main__":
    try:
        results_file = run_modality_analysis_experiment()
        print(f"\n✅ Experiment 1 completed successfully!")
        print(f"📄 Results saved to: {results_file}")
    except Exception as e:
        print(f"\n❌ Experiment 1 failed: {str(e)}")
        sys.exit(1) 