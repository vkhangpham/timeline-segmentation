#!/usr/bin/env python3
"""
Experiment 2: Temporal Window Sensitivity Analysis

Research Question: How sensitive is algorithm performance to different temporal window configurations?

This experiment evaluates:
1. Direction window sizes: [4, 6, 8, 10] years with different split ratios
2. Citation scales: individual [1], [3], [5] and combinations [1,3], [3,5], [1,5], [1,3,5]

Follows functional programming principles with fail-fast error handling.
"""

import os
import sys
import time
from typing import Dict, List, Tuple, Any
from copy import deepcopy

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from experiment_utils import (
    load_test_domains, evaluate_segmentation_configuration,
    ExperimentResult, save_experiment_results, print_experiment_summary,
    calculate_statistical_significance, TEST_DOMAINS
)
from core.algorithm_config import AlgorithmConfig


# Direction window configurations to test
DIRECTION_WINDOW_CONFIGS = [
    2,   # 2 years
    3,   # 3 years (baseline) 
    4,   # 4 years
    5,   # 5 years
    6,   # 6 years
]

# Citation scale configurations to test
CITATION_SCALE_CONFIGS = [
    [1],         # 1-year only
    [3],         # 3-year only
    [5],         # 5-year only
    [1, 3],      # 1+3 year combination
    [3, 5],      # 3+5 year combination  
    [1, 5],      # 1+5 year combination
    [1, 3, 5],   # Full multi-scale (baseline)
]


def create_direction_window_config(
    base_config: AlgorithmConfig,
    window_size: int
) -> AlgorithmConfig:
    """
    Create algorithm configuration with modified direction window settings.
    
    Pure function that creates new config with specific direction window parameters.
    
    Args:
        base_config: Base algorithm configuration
        window_size: Direction window size in years
        
    Returns:
        Modified algorithm configuration
    """
    config_dict = deepcopy(base_config.__dict__)
    
    # Update direction window parameters
    config_dict['direction_window_size'] = window_size
    
    return AlgorithmConfig(**config_dict)


def create_citation_scale_config(
    base_config: AlgorithmConfig,
    citation_scales: List[int]
) -> AlgorithmConfig:
    """
    Create algorithm configuration with modified citation scale settings.
    
    Pure function that creates new config with specific citation scale parameters.
    
    Args:
        base_config: Base algorithm configuration
        citation_scales: List of citation analysis scales in years
        
    Returns:
        Modified algorithm configuration
    """
    config_dict = deepcopy(base_config.__dict__)
    
    # Update citation scale parameters
    config_dict['citation_analysis_scales'] = citation_scales
    
    return AlgorithmConfig(**config_dict)


def run_direction_window_analysis(
    domain_name: str,
    domain_data,
    base_config: AlgorithmConfig
) -> List[ExperimentResult]:
    """
    Run direction window analysis for a single domain.
    
    Pure function that evaluates all direction window configurations.
    
    Args:
        domain_name: Name of the domain being tested
        domain_data: DomainData object for the domain
        base_config: Base algorithm configuration
        
    Returns:
        List of ExperimentResult objects for all window configurations
        
    Raises:
        ValueError: If any evaluation fails
    """
    print(f"\n🔬 Running direction window analysis for {domain_name}")
    
    results = []
    
    for total_years in DIRECTION_WINDOW_CONFIGS:
        condition_name = f"dir_window_{total_years}y"
        print(f"  🎯 Testing {condition_name}...")
        
        start_time = time.time()
        
        try:
            # Create configuration with modified direction window
            test_config = create_direction_window_config(base_config, total_years)
            
            score, metrics = evaluate_segmentation_configuration(
                domain_data, test_config, use_citation=True, use_direction=True
            )
            execution_time = time.time() - start_time
            
            results.append(ExperimentResult(
                experiment_name="temporal_windows",
                domain=domain_name,
                condition=condition_name,
                score=score,
                consensus_score=metrics['consensus_score'],
                difference_score=metrics['difference_score'],
                num_segments=metrics['num_segments'],
                execution_time=execution_time,
                metadata={
                    'window_type': 'direction',
                    'total_years': total_years,
                    'segment_sizes': metrics['segment_sizes'],
                    'detailed_metrics': metrics
                }
            ))
            
            print(f"    ✅ {condition_name}: score={score:.3f} ({metrics['num_segments']} segments, {execution_time:.1f}s)")
            
        except Exception as e:
            raise ValueError(f"Direction window evaluation failed for {domain_name} with {condition_name}: {str(e)}")
    
    return results


def run_citation_scale_analysis(
    domain_name: str,
    domain_data,
    base_config: AlgorithmConfig
) -> List[ExperimentResult]:
    """
    Run citation scale analysis for a single domain.
    
    Pure function that evaluates all citation scale configurations.
    
    Args:
        domain_name: Name of the domain being tested
        domain_data: DomainData object for the domain
        base_config: Base algorithm configuration
        
    Returns:
        List of ExperimentResult objects for all scale configurations
        
    Raises:
        ValueError: If any evaluation fails
    """
    print(f"\n🔬 Running citation scale analysis for {domain_name}")
    
    results = []
    
    for citation_scales in CITATION_SCALE_CONFIGS:
        scales_str = "+".join(map(str, citation_scales))
        condition_name = f"cit_scales_{scales_str}y"
        print(f"  📈 Testing {condition_name}...")
        
        start_time = time.time()
        
        try:
            # Create configuration with modified citation scales
            test_config = create_citation_scale_config(base_config, citation_scales)
            
            score, metrics = evaluate_segmentation_configuration(
                domain_data, test_config, use_citation=True, use_direction=True
            )
            execution_time = time.time() - start_time
            
            results.append(ExperimentResult(
                experiment_name="temporal_windows",
                domain=domain_name,
                condition=condition_name,
                score=score,
                consensus_score=metrics['consensus_score'],
                difference_score=metrics['difference_score'],
                num_segments=metrics['num_segments'],
                execution_time=execution_time,
                metadata={
                    'window_type': 'citation',
                    'citation_scales': citation_scales,
                    'num_scales': len(citation_scales),
                    'segment_sizes': metrics['segment_sizes'],
                    'detailed_metrics': metrics
                }
            ))
            
            print(f"    ✅ {condition_name}: score={score:.3f} ({metrics['num_segments']} segments, {execution_time:.1f}s)")
            
        except Exception as e:
            raise ValueError(f"Citation scale evaluation failed for {domain_name} with {condition_name}: {str(e)}")
    
    return results


def analyze_temporal_sensitivity(results: List[ExperimentResult]) -> Dict[str, Any]:
    """
    Analyze temporal window sensitivity patterns across domains.
    
    Pure function that calculates window-specific insights from results.
    
    Args:
        results: List of all experiment results
        
    Returns:
        Dictionary with analysis insights
    """
    # Separate direction and citation results
    direction_results = [r for r in results if r.metadata['window_type'] == 'direction']
    citation_results = [r for r in results if r.metadata['window_type'] == 'citation']
    
    analysis = {
        'direction_window_analysis': {},
        'citation_scale_analysis': {},
        'optimal_configurations': {},
        'sensitivity_patterns': {}
    }
    
    # Analyze direction window sensitivity
    if direction_results:
        direction_by_domain = {}
        for result in direction_results:
            if result.domain not in direction_by_domain:
                direction_by_domain[result.domain] = []
            direction_by_domain[result.domain].append(result)
        
        for domain, domain_results in direction_by_domain.items():
            # Find best configuration for this domain
            best_result = max(domain_results, key=lambda x: x.score)
            
            # Calculate sensitivity (score variance)
            scores = [r.score for r in domain_results]
            sensitivity = max(scores) - min(scores) if len(scores) > 1 else 0
            
            analysis['direction_window_analysis'][domain] = {
                'best_config': best_result.condition,
                'best_score': best_result.score,
                'total_years': best_result.metadata['total_years'],
                'sensitivity': sensitivity,
                'all_scores': {r.condition: r.score for r in domain_results}
            }
    
    # Analyze citation scale sensitivity
    if citation_results:
        citation_by_domain = {}
        for result in citation_results:
            if result.domain not in citation_by_domain:
                citation_by_domain[result.domain] = []
            citation_by_domain[result.domain].append(result)
        
        for domain, domain_results in citation_by_domain.items():
            # Find best configuration for this domain
            best_result = max(domain_results, key=lambda x: x.score)
            
            # Calculate sensitivity (score variance)
            scores = [r.score for r in domain_results]
            sensitivity = max(scores) - min(scores) if len(scores) > 1 else 0
            
            analysis['citation_scale_analysis'][domain] = {
                'best_config': best_result.condition,
                'best_score': best_result.score,
                'citation_scales': best_result.metadata['citation_scales'],
                'num_scales': best_result.metadata['num_scales'],
                'sensitivity': sensitivity,
                'all_scores': {r.condition: r.score for r in domain_results}
            }
    
    # Calculate overall patterns
    if direction_results and citation_results:
        # Find globally optimal configurations
        all_direction_scores = [r.score for r in direction_results]
        all_citation_scores = [r.score for r in citation_results]
        
        analysis['sensitivity_patterns'] = {
            'direction_mean_sensitivity': float(sum(analysis['direction_window_analysis'][d]['sensitivity'] 
                                                 for d in analysis['direction_window_analysis']) / 
                                                len(analysis['direction_window_analysis'])) if analysis['direction_window_analysis'] else 0,
            'citation_mean_sensitivity': float(sum(analysis['citation_scale_analysis'][d]['sensitivity'] 
                                               for d in analysis['citation_scale_analysis']) / 
                                               len(analysis['citation_scale_analysis'])) if analysis['citation_scale_analysis'] else 0,
            'direction_score_range': (min(all_direction_scores), max(all_direction_scores)),
            'citation_score_range': (min(all_citation_scores), max(all_citation_scores))
        }
    
    return analysis


def run_temporal_window_experiment() -> str:
    """
    Run the complete temporal window sensitivity experiment.
    
    Main function that orchestrates the full experiment across all test domains.
    
    Returns:
        Path to saved results file
        
    Raises:
        ValueError: If experiment fails
    """
    print("🧪 EXPERIMENT 2: TEMPORAL WINDOW SENSITIVITY ANALYSIS")
    print("=" * 70)
    print("Research Question: How sensitive is algorithm performance to different temporal window configurations?")
    print(f"Test domains: {', '.join(TEST_DOMAINS)}")
    print(f"\nDirection window configurations: {len(DIRECTION_WINDOW_CONFIGS)}")
    print(f"Citation scale configurations: {len(CITATION_SCALE_CONFIGS)}")
    
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
            # Run direction window analysis
            direction_results = run_direction_window_analysis(
                domain_name, domain_data[domain_name], base_config
            )
            all_results.extend(direction_results)
            
            # Run citation scale analysis
            citation_results = run_citation_scale_analysis(
                domain_name, domain_data[domain_name], base_config
            )
            all_results.extend(citation_results)
            
        except Exception as e:
            raise ValueError(f"Experiment failed for domain {domain_name}: {str(e)}")
    
    # Analyze results
    analysis = analyze_temporal_sensitivity(all_results)
    
    # Print summary
    print_experiment_summary(all_results, "temporal_windows")
    
    # Print detailed analysis
    print("\n📊 TEMPORAL WINDOW SENSITIVITY ANALYSIS")
    print("=" * 60)
    
    print("\n🎯 DIRECTION WINDOW ANALYSIS:")
    for domain, domain_analysis in analysis.get('direction_window_analysis', {}).items():
        print(f"\n🔍 {domain}:")
        print(f"   Best config: {domain_analysis['best_config']} (score={domain_analysis['best_score']:.3f})")
        print(f"   Optimal: {domain_analysis['total_years']} years")
        print(f"   Sensitivity: {domain_analysis['sensitivity']:.3f}")
    
    print("\n📈 CITATION SCALE ANALYSIS:")
    for domain, domain_analysis in analysis.get('citation_scale_analysis', {}).items():
        print(f"\n🔍 {domain}:")
        print(f"   Best config: {domain_analysis['best_config']} (score={domain_analysis['best_score']:.3f})")
        print(f"   Optimal scales: {domain_analysis['citation_scales']}")
        print(f"   Sensitivity: {domain_analysis['sensitivity']:.3f}")
    
    if 'sensitivity_patterns' in analysis:
        patterns = analysis['sensitivity_patterns']
        print(f"\n📊 OVERALL SENSITIVITY PATTERNS:")
        print(f"   Direction mean sensitivity: {patterns['direction_mean_sensitivity']:.3f}")
        print(f"   Citation mean sensitivity: {patterns['citation_mean_sensitivity']:.3f}")
        print(f"   Direction score range: {patterns['direction_score_range'][0]:.3f} - {patterns['direction_score_range'][1]:.3f}")
        print(f"   Citation score range: {patterns['citation_score_range'][0]:.3f} - {patterns['citation_score_range'][1]:.3f}")
    
    # Save results
    additional_metadata = {
        'experiment_description': 'Temporal window sensitivity analysis for direction and citation detection',
        'test_domains': TEST_DOMAINS,
        'direction_window_configs': DIRECTION_WINDOW_CONFIGS,
        'citation_scale_configs': CITATION_SCALE_CONFIGS,
        'base_config': base_config.__dict__,
        'analysis': analysis
    }
    
    try:
        results_file = save_experiment_results(all_results, "temporal_windows", additional_metadata)
        return results_file
    except Exception as e:
        raise ValueError(f"Failed to save results: {str(e)}")


if __name__ == "__main__":
    try:
        results_file = run_temporal_window_experiment()
        print(f"\n✅ Experiment 2 completed successfully!")
        print(f"📄 Results saved to: {results_file}")
    except Exception as e:
        print(f"\n❌ Experiment 2 failed: {str(e)}")
        sys.exit(1) 