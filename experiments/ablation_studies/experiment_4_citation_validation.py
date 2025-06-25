#!/usr/bin/env python3
"""
Experiment 4: Citation Validation Strategy Comparison

Research Question: How do different fusion and validation strategies affect detection performance?

This experiment evaluates:
1. Boost factor analysis: β ∈ [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2]
2. Validation window analysis: w_cit ∈ [1, 2, 3, 4, 5] years
3. Alternative fusion methods: additive vs multiplicative, early vs late fusion

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


# Citation boost factor configurations to test
BOOST_FACTOR_CONFIGS = [
    (0.0, "no_boost"),           # No citation boosting
    (0.2, "minimal_boost"),      # Minimal boost
    (0.4, "low_boost"),          # Low boost
    (0.6, "moderate_boost"),     # Moderate boost
    (0.8, "standard_boost"),     # Standard boost (baseline)
    (1.0, "maximum_boost")       # Maximum boost
]

# Citation validation window configurations to test
VALIDATION_WINDOW_CONFIGS = [
    (1, "narrow_window"),     # 1-year window
    (2, "small_window"),      # 2-year window
    (3, "standard_window"),   # 3-year window (baseline)
    (4, "wide_window"),       # 4-year window
    (5, "very_wide_window")   # 5-year window
]

# Remove fusion method configs - not configurable in current algorithm


def create_boost_factor_config(
    base_config: AlgorithmConfig,
    boost_factor: float
) -> AlgorithmConfig:
    """
    Create algorithm configuration with modified citation boost factor.
    
    Pure function that creates new config with specific boost factor parameter.
    
    Args:
        base_config: Base algorithm configuration
        boost_factor: Citation confidence boost factor (β)
        
    Returns:
        Modified algorithm configuration
    """
    config_dict = deepcopy(base_config.__dict__)
    
    # Update citation boost factor
    config_dict['citation_boost'] = boost_factor
    
    return AlgorithmConfig(**config_dict)


def create_validation_window_config(
    base_config: AlgorithmConfig,
    window_size: int
) -> AlgorithmConfig:
    """
    Create algorithm configuration with modified citation validation window.
    
    Pure function that creates new config with specific validation window parameter.
    
    Args:
        base_config: Base algorithm configuration
        window_size: Citation validation window size (w_cit)
        
    Returns:
        Modified algorithm configuration
    """
    config_dict = deepcopy(base_config.__dict__)
    
    # Update citation validation window
    config_dict['citation_support_window'] = window_size
    
    return AlgorithmConfig(**config_dict)


def run_boost_factor_analysis(
    domain_name: str,
    domain_data,
    base_config: AlgorithmConfig
) -> List[ExperimentResult]:
    """
    Run citation boost factor analysis for a single domain.
    
    Pure function that evaluates all boost factor configurations.
    
    Args:
        domain_name: Name of the domain being tested
        domain_data: DomainData object for the domain
        base_config: Base algorithm configuration
        
    Returns:
        List of ExperimentResult objects for all boost factor configurations
        
    Raises:
        ValueError: If any evaluation fails
    """
    print(f"\n🔬 Running citation boost factor analysis for {domain_name}")
    
    results = []
    
    for boost_factor, condition_name in BOOST_FACTOR_CONFIGS:
        print(f"  📈 Testing {condition_name} (β={boost_factor:.1f})...")
        
        start_time = time.time()
        
        try:
            # Create configuration with modified boost factor
            test_config = create_boost_factor_config(base_config, boost_factor)
            
            score, metrics = evaluate_segmentation_configuration(
                domain_data, test_config, use_citation=True, use_direction=True
            )
            execution_time = time.time() - start_time
            
            results.append(ExperimentResult(
                experiment_name="citation_validation",
                domain=domain_name,
                condition=condition_name,
                score=score,
                consensus_score=metrics['consensus_score'],
                difference_score=metrics['difference_score'],
                num_segments=metrics['num_segments'],
                execution_time=execution_time,
                metadata={
                    'analysis_type': 'boost_factor',
                    'boost_factor': boost_factor,
                    'segment_sizes': metrics['segment_sizes'],
                    'detailed_metrics': metrics
                }
            ))
            
            print(f"    ✅ {condition_name}: score={score:.3f} ({metrics['num_segments']} segments, {execution_time:.1f}s)")
            
        except Exception as e:
            raise ValueError(f"Boost factor evaluation failed for {domain_name} with {condition_name}: {str(e)}")
    
    return results


def run_validation_window_analysis(
    domain_name: str,
    domain_data,
    base_config: AlgorithmConfig
) -> List[ExperimentResult]:
    """
    Run citation validation window analysis for a single domain.
    
    Pure function that evaluates all validation window configurations.
    
    Args:
        domain_name: Name of the domain being tested
        domain_data: DomainData object for the domain
        base_config: Base algorithm configuration
        
    Returns:
        List of ExperimentResult objects for all window configurations
        
    Raises:
        ValueError: If any evaluation fails
    """
    print(f"\n🔬 Running citation validation window analysis for {domain_name}")
    
    results = []
    
    for window_size, condition_name in VALIDATION_WINDOW_CONFIGS:
        print(f"  🪟 Testing {condition_name} (w_cit={window_size})...")
        
        start_time = time.time()
        
        try:
            # Create configuration with modified validation window
            test_config = create_validation_window_config(base_config, window_size)
            
            score, metrics = evaluate_segmentation_configuration(
                domain_data, test_config, use_citation=True, use_direction=True
            )
            execution_time = time.time() - start_time
            
            results.append(ExperimentResult(
                experiment_name="citation_validation",
                domain=domain_name,
                condition=condition_name,
                score=score,
                consensus_score=metrics['consensus_score'],
                difference_score=metrics['difference_score'],
                num_segments=metrics['num_segments'],
                execution_time=execution_time,
                metadata={
                    'analysis_type': 'validation_window',
                    'validation_window': window_size,
                    'segment_sizes': metrics['segment_sizes'],
                    'detailed_metrics': metrics
                }
            ))
            
            print(f"    ✅ {condition_name}: score={score:.3f} ({metrics['num_segments']} segments, {execution_time:.1f}s)")
            
        except Exception as e:
            raise ValueError(f"Validation window evaluation failed for {domain_name} with {condition_name}: {str(e)}")
    
    return results


def analyze_citation_validation_strategies(results: List[ExperimentResult]) -> Dict[str, Any]:
    """
    Analyze citation validation strategy effectiveness across domains.
    
    Pure function that calculates validation-specific insights from results.
    
    Args:
        results: List of all experiment results
        
    Returns:
        Dictionary with analysis insights
    """
    # Separate results by analysis type
    boost_factor_results = [r for r in results if r.metadata['analysis_type'] == 'boost_factor']
    validation_window_results = [r for r in results if r.metadata['analysis_type'] == 'validation_window']
    
    analysis = {
        'boost_factor_analysis': {},
        'validation_window_analysis': {},
        'optimal_strategies': {}
    }
    
    # Analyze boost factor sensitivity
    if boost_factor_results:
        boost_by_domain = {}
        for result in boost_factor_results:
            if result.domain not in boost_by_domain:
                boost_by_domain[result.domain] = []
            boost_by_domain[result.domain].append(result)
        
        for domain, domain_results in boost_by_domain.items():
            # Find optimal boost factor for this domain
            best_result = max(domain_results, key=lambda x: x.score)
            
            # Calculate sensitivity to boost factor
            scores = [r.score for r in domain_results]
            sensitivity = max(scores) - min(scores) if len(scores) > 1 else 0
            
            # Find score vs boost factor relationship
            boost_score_pairs = [(r.metadata['boost_factor'], r.score) for r in domain_results]
            boost_score_pairs.sort()
            
            analysis['boost_factor_analysis'][domain] = {
                'optimal_boost_factor': best_result.metadata['boost_factor'],
                'optimal_score': best_result.score,
                'sensitivity': sensitivity,
                'boost_score_curve': boost_score_pairs,
                'baseline_comparison': {
                    'baseline_score': next((r.score for r in domain_results if r.metadata['boost_factor'] == 0.8), None),
                    'no_boost_score': next((r.score for r in domain_results if r.metadata['boost_factor'] == 0.0), None)
                }
            }
    
    # Analyze validation window sensitivity
    if validation_window_results:
        window_by_domain = {}
        for result in validation_window_results:
            if result.domain not in window_by_domain:
                window_by_domain[result.domain] = []
            window_by_domain[result.domain].append(result)
        
        for domain, domain_results in window_by_domain.items():
            # Find optimal window size for this domain
            best_result = max(domain_results, key=lambda x: x.score)
            
            # Calculate sensitivity to window size
            scores = [r.score for r in domain_results]
            sensitivity = max(scores) - min(scores) if len(scores) > 1 else 0
            
            # Find score vs window size relationship
            window_score_pairs = [(r.metadata['validation_window'], r.score) for r in domain_results]
            window_score_pairs.sort()
            
            analysis['validation_window_analysis'][domain] = {
                'optimal_window_size': best_result.metadata['validation_window'],
                'optimal_score': best_result.score,
                'sensitivity': sensitivity,
                'window_score_curve': window_score_pairs,
                'baseline_comparison': {
                    'baseline_score': next((r.score for r in domain_results if r.metadata['validation_window'] == 3), None)
                }
            }
    
    # Calculate overall optimal strategies
    if boost_factor_results and validation_window_results:
        # Find globally optimal configurations per domain
        domain_optimal_configs = {}
        for domain in TEST_DOMAINS:
            domain_optimal_configs[domain] = {
                'optimal_boost_factor': analysis['boost_factor_analysis'].get(domain, {}).get('optimal_boost_factor'),
                'optimal_window_size': analysis['validation_window_analysis'].get(domain, {}).get('optimal_window_size'),
            }
        
        analysis['optimal_strategies'] = {
            'domain_specific_configs': domain_optimal_configs,
            'most_common_boost_factor': max(set(analysis['boost_factor_analysis'][d]['optimal_boost_factor'] 
                                              for d in analysis['boost_factor_analysis']), 
                                          key=lambda x: sum(1 for d in analysis['boost_factor_analysis'] 
                                                           if analysis['boost_factor_analysis'][d]['optimal_boost_factor'] == x)),
            'most_common_window_size': max(set(analysis['validation_window_analysis'][d]['optimal_window_size'] 
                                             for d in analysis['validation_window_analysis']), 
                                         key=lambda x: sum(1 for d in analysis['validation_window_analysis'] 
                                                          if analysis['validation_window_analysis'][d]['optimal_window_size'] == x)),
        }
    
    return analysis


def run_citation_validation_experiment() -> str:
    """
    Run the complete citation validation strategy experiment.
    
    Main function that orchestrates the full experiment across all test domains.
    
    Returns:
        Path to saved results file
        
    Raises:
        ValueError: If experiment fails
    """
    print("🧪 EXPERIMENT 4: CITATION VALIDATION STRATEGY COMPARISON")
    print("=" * 70)
    print("Research Question: How do different fusion and validation strategies affect detection performance?")
    print(f"Test domains: {', '.join(TEST_DOMAINS)}")
    print(f"\nAnalysis types:")
    print(f"  - Boost factor configurations: {len(BOOST_FACTOR_CONFIGS)}")
    print(f"  - Validation window configurations: {len(VALIDATION_WINDOW_CONFIGS)}")
    
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
            # Run boost factor analysis
            boost_results = run_boost_factor_analysis(
                domain_name, domain_data[domain_name], base_config
            )
            all_results.extend(boost_results)
            
            # Run validation window analysis
            window_results = run_validation_window_analysis(
                domain_name, domain_data[domain_name], base_config
            )
            all_results.extend(window_results)
            
        except Exception as e:
            raise ValueError(f"Experiment failed for domain {domain_name}: {str(e)}")
    
    # Analyze results
    analysis = analyze_citation_validation_strategies(all_results)
    
    # Print summary
    print_experiment_summary(all_results, "citation_validation")
    
    # Print detailed analysis
    print("\n📊 CITATION VALIDATION STRATEGY ANALYSIS")
    print("=" * 60)
    
    print("\n📈 BOOST FACTOR ANALYSIS:")
    for domain, domain_analysis in analysis.get('boost_factor_analysis', {}).items():
        print(f"\n🔍 {domain}:")
        print(f"   Optimal boost factor: {domain_analysis['optimal_boost_factor']:.1f} (score={domain_analysis['optimal_score']:.3f})")
        print(f"   Sensitivity: {domain_analysis['sensitivity']:.3f}")
        baseline = domain_analysis['baseline_comparison']['baseline_score']
        no_boost = domain_analysis['baseline_comparison']['no_boost_score']
        if baseline and no_boost:
            print(f"   Baseline vs no-boost: {baseline:.3f} vs {no_boost:.3f} ({(baseline-no_boost):+.3f})")
    
    print("\n🪟 VALIDATION WINDOW ANALYSIS:")
    for domain, domain_analysis in analysis.get('validation_window_analysis', {}).items():
        print(f"\n🔍 {domain}:")
        print(f"   Optimal window size: {domain_analysis['optimal_window_size']} years (score={domain_analysis['optimal_score']:.3f})")
        print(f"   Sensitivity: {domain_analysis['sensitivity']:.3f}")
    
    if 'optimal_strategies' in analysis:
        strategies = analysis['optimal_strategies']
        print(f"\n🎯 GLOBAL OPTIMAL STRATEGIES:")
        print(f"   Most common optimal boost factor: {strategies['most_common_boost_factor']:.1f}")
        print(f"   Most common optimal window size: {strategies['most_common_window_size']} years")
    
    # Save results
    additional_metadata = {
        'experiment_description': 'Citation validation strategy comparison across boost factors and window sizes',
        'test_domains': TEST_DOMAINS,
        'boost_factor_configs': BOOST_FACTOR_CONFIGS,
        'validation_window_configs': VALIDATION_WINDOW_CONFIGS,
        'base_config': base_config.__dict__,
        'analysis': analysis
    }
    
    try:
        results_file = save_experiment_results(all_results, "citation_validation", additional_metadata)
        return results_file
    except Exception as e:
        raise ValueError(f"Failed to save results: {str(e)}")


if __name__ == "__main__":
    try:
        results_file = run_citation_validation_experiment()
        print(f"\n✅ Experiment 4 completed successfully!")
        print(f"📄 Results saved to: {results_file}")
    except Exception as e:
        print(f"\n❌ Experiment 4 failed: {str(e)}")
        sys.exit(1) 