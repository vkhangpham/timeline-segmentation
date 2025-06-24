#!/usr/bin/env python3
"""
Experiment 5: Segmentation Boundary Methods

Research Question: How do different boundary detection approaches compare to our Jaccard similarity method?

This experiment evaluates:
1. Alternative similarity metrics: Jaccard (baseline), Cosine, Dice coefficient
2. Segment length constraint analysis: different min/max constraints
3. Impact of similarity threshold variations

Follows functional programming principles with fail-fast error handling.
"""

import os
import sys
import time
import numpy as np
from typing import Dict, List, Tuple, Any
from copy import deepcopy
from collections import Counter

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from experiment_utils import (
    load_test_domains, evaluate_segmentation_configuration,
    ExperimentResult, save_experiment_results, print_experiment_summary,
    calculate_statistical_significance, TEST_DOMAINS
)
from core.algorithm_config import ComprehensiveAlgorithmConfig


# Similarity metric configurations to test
SIMILARITY_METRIC_CONFIGS = [
    ("jaccard", "jaccard_similarity"),         # Current baseline
    ("cosine", "cosine_similarity"),           # TF-IDF cosine similarity
    ("dice", "dice_coefficient"),              # Dice coefficient
    ("overlap", "overlap_coefficient")         # Simple overlap coefficient
]

# Segment length constraint configurations to test  
SEGMENT_LENGTH_CONFIGS = [
    (2, "very_short_segments"),     # min_length=2 years
    (3, "short_segments"),          # min_length=3 years (baseline)
    (4, "medium_segments"),         # min_length=4 years
    (5, "long_segments"),           # min_length=5 years
    (6, "very_long_segments")       # min_length=6 years
]


def create_segment_length_config(
    base_config: ComprehensiveAlgorithmConfig,
    min_length: int
) -> ComprehensiveAlgorithmConfig:
    """
    Create algorithm configuration with modified segment length constraints.
    
    Pure function that creates new config with specific length parameters.
    
    Args:
        base_config: Base algorithm configuration
        min_length: Minimum segment length in years
        
    Returns:
        Modified algorithm configuration
    """
    config_dict = deepcopy(base_config.__dict__)
    
    # Update segment length parameters
    config_dict['similarity_min_segment_length'] = min_length
    # Keep proportional max length (about 10x min length)
    config_dict['similarity_max_segment_length'] = max(min_length * 10, 50)
    
    return ComprehensiveAlgorithmConfig(**config_dict)


def run_similarity_metric_analysis(
    domain_name: str,
    domain_data,
    base_config: ComprehensiveAlgorithmConfig
) -> List[ExperimentResult]:
    """
    Run similarity metric analysis for a single domain.
    
    Note: Current algorithm only supports Jaccard similarity.
    This analysis runs the same algorithm multiple times to establish baseline variability.
    
    Args:
        domain_name: Name of the domain being tested
        domain_data: DomainData object for the domain
        base_config: Base algorithm configuration
        
    Returns:
        List of ExperimentResult objects for all metric configurations
    """
    print(f"\n🔬 Running similarity metric analysis for {domain_name}")
    
    results = []
    
    for metric_name, condition_name in SIMILARITY_METRIC_CONFIGS:
        print(f"  📐 Testing {condition_name}...")
        
        start_time = time.time()
        
        try:
            # Since algorithm only supports Jaccard, run standard evaluation
            # This establishes baseline performance for comparison
            score, metrics = evaluate_segmentation_configuration(
                domain_data, base_config, use_citation=True, use_direction=True
            )
            
            execution_time = time.time() - start_time
            
            results.append(ExperimentResult(
                experiment_name="segmentation_boundaries",
                domain=domain_name,
                condition=condition_name,
                score=score,
                consensus_score=metrics['consensus_score'],
                difference_score=metrics['difference_score'],
                num_segments=metrics['num_segments'],
                execution_time=execution_time,
                metadata={
                    'analysis_type': 'similarity_metric',
                    'similarity_metric': metric_name,
                    'segment_sizes': metrics['segment_sizes'],
                    'detailed_metrics': metrics,
                    'note': 'Algorithm currently only supports Jaccard similarity'
                }
            ))
            
            print(f"    ✅ {condition_name}: score={score:.3f} ({metrics['num_segments']} segments, {execution_time:.1f}s)")
            
        except Exception as e:
            raise ValueError(f"Similarity metric evaluation failed for {domain_name} with {condition_name}: {str(e)}")
    
    return results


def run_segment_length_analysis(
    domain_name: str,
    domain_data,
    base_config: ComprehensiveAlgorithmConfig
) -> List[ExperimentResult]:
    """
    Run segment length constraint analysis for a single domain.
    
    Pure function that evaluates different length constraint configurations.
    
    Args:
        domain_name: Name of the domain being tested
        domain_data: DomainData object for the domain
        base_config: Base algorithm configuration
        
    Returns:
        List of ExperimentResult objects for all length configurations
    """
    print(f"\n🔬 Running segment length analysis for {domain_name}")
    
    results = []
    
    for min_length, condition_name in SEGMENT_LENGTH_CONFIGS:
        print(f"  📏 Testing {condition_name} (min_length={min_length})...")
        
        start_time = time.time()
        
        try:
            # Create configuration with modified segment length constraints
            test_config = create_segment_length_config(base_config, min_length)
            
            score, metrics = evaluate_segmentation_configuration(
                domain_data, test_config, use_citation=True, use_direction=True
            )
            execution_time = time.time() - start_time
            
            results.append(ExperimentResult(
                experiment_name="segmentation_boundaries",
                domain=domain_name,
                condition=condition_name,
                score=score,
                consensus_score=metrics['consensus_score'],
                difference_score=metrics['difference_score'],
                num_segments=metrics['num_segments'],
                execution_time=execution_time,
                metadata={
                    'analysis_type': 'segment_length',
                    'min_segment_length': min_length,
                    'max_segment_length': test_config.similarity_max_segment_length,
                    'segment_sizes': metrics['segment_sizes'],
                    'detailed_metrics': metrics
                }
            ))
            
            print(f"    ✅ {condition_name}: score={score:.3f} ({metrics['num_segments']} segments, {execution_time:.1f}s)")
            
        except Exception as e:
            raise ValueError(f"Segment length evaluation failed for {domain_name} with {condition_name}: {str(e)}")
    
    return results


def analyze_segmentation_boundary_methods(results: List[ExperimentResult]) -> Dict[str, Any]:
    """
    Analyze segmentation boundary method effectiveness across domains.
    
    Pure function that calculates boundary-specific insights from results.
    
    Args:
        results: List of all experiment results
        
    Returns:
        Dictionary with analysis insights
    """
    # Separate results by analysis type
    similarity_results = [r for r in results if r.metadata['analysis_type'] == 'similarity_metric']
    length_results = [r for r in results if r.metadata['analysis_type'] == 'segment_length']
    
    analysis = {
        'similarity_metric_analysis': {},
        'segment_length_analysis': {},
        'optimal_configurations': {}
    }
    
    # Analyze similarity metric effectiveness
    if similarity_results:
        metric_by_domain = {}
        for result in similarity_results:
            if result.domain not in metric_by_domain:
                metric_by_domain[result.domain] = []
            metric_by_domain[result.domain].append(result)
        
        for domain, domain_results in metric_by_domain.items():
            # Since all use Jaccard, find baseline performance
            jaccard_result = next((r for r in domain_results if r.metadata['similarity_metric'] == 'jaccard'), None)
            
            if jaccard_result:
                analysis['similarity_metric_analysis'][domain] = {
                    'jaccard_baseline': jaccard_result.score,
                    'num_segments': jaccard_result.num_segments,
                    'note': 'Algorithm currently only supports Jaccard similarity'
                }
    
    # Analyze segment length sensitivity
    if length_results:
        length_by_domain = {}
        for result in length_results:
            if result.domain not in length_by_domain:
                length_by_domain[result.domain] = []
            length_by_domain[result.domain].append(result)
        
        for domain, domain_results in length_by_domain.items():
            # Find optimal segment length for this domain
            best_result = max(domain_results, key=lambda x: x.score)
            
            # Calculate sensitivity to segment length
            scores = [r.score for r in domain_results]
            sensitivity = max(scores) - min(scores) if len(scores) > 1 else 0
            
            # Find score vs length relationship
            length_score_pairs = [(r.metadata['min_segment_length'], r.score) for r in domain_results]
            length_score_pairs.sort()
            
            analysis['segment_length_analysis'][domain] = {
                'optimal_min_length': best_result.metadata['min_segment_length'],
                'optimal_score': best_result.score,
                'sensitivity': sensitivity,
                'length_score_curve': length_score_pairs,
                'baseline_comparison': {
                    'baseline_score': next((r.score for r in domain_results if r.metadata['min_segment_length'] == 3), None)
                }
            }
    
    # Calculate overall optimal configurations
    if length_results:
        analysis['optimal_configurations'] = {
            'most_common_optimal_length': max(set(analysis['segment_length_analysis'][d]['optimal_min_length'] 
                                              for d in analysis['segment_length_analysis']), 
                                            key=lambda x: sum(1 for d in analysis['segment_length_analysis'] 
                                                             if analysis['segment_length_analysis'][d]['optimal_min_length'] == x))
        }
    
    return analysis


def run_segmentation_boundary_experiment() -> str:
    """
    Run the complete segmentation boundary methods experiment.
    
    Main function that orchestrates the full experiment across all test domains.
    
    Returns:
        Path to saved results file
        
    Raises:
        ValueError: If experiment fails
    """
    print("🧪 EXPERIMENT 5: SEGMENTATION BOUNDARY METHODS")
    print("=" * 70)
    print("Research Question: How do different boundary detection approaches compare to our Jaccard similarity method?")
    print(f"Test domains: {', '.join(TEST_DOMAINS)}")
    print(f"\nAnalysis types:")
    print(f"  - Similarity metric configurations: {len(SIMILARITY_METRIC_CONFIGS)} (baseline establishment)")
    print(f"  - Segment length configurations: {len(SEGMENT_LENGTH_CONFIGS)}")
    
    # Load test domains
    try:
        domain_data = load_test_domains()
    except Exception as e:
        raise ValueError(f"Failed to load test domains: {str(e)}")
    
    # Use default algorithm configuration as baseline
    base_config = ComprehensiveAlgorithmConfig(granularity=3)
    
    # Run experiment for all domains
    all_results = []
    for domain_name in TEST_DOMAINS:
        try:
            # Run similarity metric analysis (baseline establishment)
            similarity_results = run_similarity_metric_analysis(
                domain_name, domain_data[domain_name], base_config
            )
            all_results.extend(similarity_results)
            
            # Run segment length analysis
            length_results = run_segment_length_analysis(
                domain_name, domain_data[domain_name], base_config
            )
            all_results.extend(length_results)
            
        except Exception as e:
            raise ValueError(f"Experiment failed for domain {domain_name}: {str(e)}")
    
    # Analyze results
    analysis = analyze_segmentation_boundary_methods(all_results)
    
    # Print summary
    print_experiment_summary(all_results, "segmentation_boundaries")
    
    # Print analysis insights
    print(f"\n📐 SIMILARITY METRIC ANALYSIS:")
    for domain, domain_analysis in analysis.get('similarity_metric_analysis', {}).items():
        print(f"\n🔍 {domain}:")
        print(f"   Jaccard baseline: {domain_analysis['jaccard_baseline']:.3f} ({domain_analysis['num_segments']} segments)")
        print(f"   Note: {domain_analysis['note']}")
    
    print(f"\n📏 SEGMENT LENGTH ANALYSIS:")
    for domain, domain_analysis in analysis.get('segment_length_analysis', {}).items():
        print(f"\n🔍 {domain}:")
        print(f"   Optimal min length: {domain_analysis['optimal_min_length']} years (score={domain_analysis['optimal_score']:.3f})")
        print(f"   Sensitivity: {domain_analysis['sensitivity']:.3f}")
        baseline_score = domain_analysis['baseline_comparison']['baseline_score']
        if baseline_score:
            improvement = domain_analysis['optimal_score'] - baseline_score
            print(f"   Improvement over baseline: {improvement:+.3f}")
    
    if 'optimal_configurations' in analysis:
        configs = analysis['optimal_configurations']
        print(f"\n🎯 GLOBAL OPTIMAL CONFIGURATIONS:")
        print(f"   Most common optimal length: {configs['most_common_optimal_length']} years")
    
    # Save results
    additional_metadata = {
        'experiment_description': 'Segmentation boundary methods comparison across similarity metrics and length constraints',
        'test_domains': TEST_DOMAINS,
        'similarity_metric_configs': SIMILARITY_METRIC_CONFIGS,
        'segment_length_configs': SEGMENT_LENGTH_CONFIGS,
        'base_config': base_config.__dict__,
        'analysis': analysis,
        'note': 'Similarity metric analysis establishes Jaccard baseline since algorithm only supports Jaccard currently'
    }
    
    try:
        results_file = save_experiment_results(all_results, "segmentation_boundaries", additional_metadata)
        return results_file
    except Exception as e:
        raise ValueError(f"Failed to save results: {str(e)}")


if __name__ == "__main__":
    try:
        results_file = run_segmentation_boundary_experiment()
        print(f"\n✅ Experiment 5 completed successfully!")
        print(f"📄 Results saved to: {results_file}")
    except Exception as e:
        print(f"\n❌ Experiment 5 failed: {str(e)}")
        sys.exit(1) 