#!/usr/bin/env python3
"""
Experiment 3: Keyword Filtering Impact Assessment

Research Question: What is the value of conservative keyword filtering across different data quality scenarios?

This experiment evaluates:
1. No filtering (all keywords as-is)
2. Conservative filtering (current approach, p_min=0.10)
3. Aggressive filtering (higher thresholds: 0.15, 0.20, 0.25)
4. Analysis across domains with varying keyword annotation quality

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
from core.algorithm_config import ComprehensiveAlgorithmConfig


# Keyword filtering configurations to test
FILTERING_CONFIGS = [
    (0.01, "minimal_filtering"),        # Minimal filtering (near disabled)
    (0.05, "light_filtering"),          # Light filtering  
    (0.10, "conservative_filtering"),   # Conservative filtering (baseline)
    (0.15, "moderate_filtering"),       # Moderate filtering
    (0.20, "aggressive_filtering"),     # Aggressive filtering
    (0.25, "very_aggressive_filtering") # Very aggressive filtering
]


def create_filtering_config(
    base_config: ComprehensiveAlgorithmConfig,
    min_papers_ratio: float
) -> ComprehensiveAlgorithmConfig:
    """
    Create algorithm configuration with modified keyword filtering settings.
    
    Pure function that creates new config with specific filtering parameters.
    
    Args:
        base_config: Base algorithm configuration
        min_papers_ratio: Minimum keyword frequency ratio (0.0 = no filtering)
        
    Returns:
        Modified algorithm configuration
    """
    config_dict = deepcopy(base_config.__dict__)
    
    # Update keyword filtering parameters
    config_dict['keyword_min_papers_ratio'] = min_papers_ratio
    config_dict['keyword_filtering_enabled'] = min_papers_ratio > 0.0
    
    return ComprehensiveAlgorithmConfig(**config_dict)


def analyze_keyword_retention(
    domain_data,
    min_papers_ratio: float
) -> Dict[str, Any]:
    """
    Analyze keyword retention statistics for a given filtering threshold.
    
    Pure function that calculates filtering impact without modifying data.
    
    Args:
        domain_data: DomainData object
        min_papers_ratio: Minimum keyword frequency ratio
        
    Returns:
        Dictionary with keyword retention statistics
    """
    # Collect all keywords and their frequencies
    keyword_counts = {}
    total_papers = len(domain_data.papers)
    
    for paper in domain_data.papers:
        for keyword in paper.keywords:
            keyword_counts[keyword] = keyword_counts.get(keyword, 0) + 1
    
    # Calculate retention statistics
    total_keywords = len(keyword_counts)
    min_occurrences = min_papers_ratio * total_papers
    
    retained_keywords = {k: v for k, v in keyword_counts.items() if v >= min_occurrences}
    retained_count = len(retained_keywords)
    
    # Calculate keyword coverage per paper
    papers_with_keywords = 0
    total_keyword_retention = 0
    
    for paper in domain_data.papers:
        paper_keywords = len(paper.keywords)
        if paper_keywords > 0:
            papers_with_keywords += 1
            retained_for_paper = sum(1 for kw in paper.keywords if kw in retained_keywords)
            total_keyword_retention += retained_for_paper / paper_keywords if paper_keywords > 0 else 0
    
    avg_retention_per_paper = total_keyword_retention / papers_with_keywords if papers_with_keywords > 0 else 0
    
    return {
        'total_unique_keywords': total_keywords,
        'retained_keywords': retained_count,
        'retention_rate': retained_count / total_keywords if total_keywords > 0 else 0,
        'min_occurrences_threshold': min_occurrences,
        'avg_retention_per_paper': avg_retention_per_paper,
        'keyword_frequency_stats': {
            'mean': sum(keyword_counts.values()) / len(keyword_counts) if keyword_counts else 0,
            'max': max(keyword_counts.values()) if keyword_counts else 0,
            'min': min(keyword_counts.values()) if keyword_counts else 0
        }
    }


def run_filtering_analysis_single_domain(
    domain_name: str,
    domain_data,
    base_config: ComprehensiveAlgorithmConfig
) -> List[ExperimentResult]:
    """
    Run keyword filtering analysis for a single domain.
    
    Pure function that evaluates all filtering configurations for one domain.
    
    Args:
        domain_name: Name of the domain being tested
        domain_data: DomainData object for the domain
        base_config: Base algorithm configuration
        
    Returns:
        List of ExperimentResult objects for all filtering configurations
        
    Raises:
        ValueError: If any evaluation fails
    """
    print(f"\n🔬 Running keyword filtering analysis for {domain_name}")
    
    results = []
    
    for min_papers_ratio, condition_name in FILTERING_CONFIGS:
        print(f"  🔍 Testing {condition_name} (ratio={min_papers_ratio:.2f})...")
        
        start_time = time.time()
        
        try:
            # Analyze keyword retention for this threshold
            retention_stats = analyze_keyword_retention(domain_data, min_papers_ratio)
            
            # Create configuration with modified filtering
            test_config = create_filtering_config(base_config, min_papers_ratio)
            
            score, metrics = evaluate_segmentation_configuration(
                domain_data, test_config, use_citation=True, use_direction=True
            )
            execution_time = time.time() - start_time
            
            results.append(ExperimentResult(
                experiment_name="keyword_filtering",
                domain=domain_name,
                condition=condition_name,
                score=score,
                consensus_score=metrics['consensus_score'],
                difference_score=metrics['difference_score'],
                num_segments=metrics['num_segments'],
                execution_time=execution_time,
                metadata={
                    'min_papers_ratio': min_papers_ratio,
                    'filtering_enabled': min_papers_ratio > 0.0,
                    'retention_stats': retention_stats,
                    'segment_sizes': metrics['segment_sizes'],
                    'detailed_metrics': metrics
                }
            ))
            
            print(f"    ✅ {condition_name}: score={score:.3f} ({metrics['num_segments']} segments, "
                  f"retained {retention_stats['retention_rate']:.1%} keywords, {execution_time:.1f}s)")
            
        except Exception as e:
            raise ValueError(f"Filtering evaluation failed for {domain_name} with {condition_name}: {str(e)}")
    
    return results


def analyze_filtering_effectiveness(results: List[ExperimentResult]) -> Dict[str, Any]:
    """
    Analyze keyword filtering effectiveness across domains.
    
    Pure function that calculates filtering-specific insights from results.
    
    Args:
        results: List of all experiment results
        
    Returns:
        Dictionary with analysis insights
    """
    # Group results by domain
    domain_results = {}
    for result in results:
        if result.domain not in domain_results:
            domain_results[result.domain] = []
        domain_results[result.domain].append(result)
    
    analysis = {
        'domain_analysis': {},
        'filtering_patterns': {},
        'quality_vs_retention_analysis': {}
    }
    
    all_retention_rates = []
    all_scores = []
    minimal_filtering_scores = []
    conservative_scores = []
    
    for domain, domain_results_list in domain_results.items():
        # Sort by p_min threshold
        domain_results_list.sort(key=lambda x: x.metadata['min_papers_ratio'])
        
        # Find best configuration for this domain
        best_result = max(domain_results_list, key=lambda x: x.score)
        
        # Get baseline scores for comparison
        minimal_filtering_result = next((r for r in domain_results_list if r.metadata['min_papers_ratio'] == 0.01), None)
        conservative_result = next((r for r in domain_results_list if r.metadata['min_papers_ratio'] == 0.10), None)
        
        # Calculate filtering benefits
        filtering_benefit = 0
        if minimal_filtering_result and conservative_result:
            filtering_benefit = conservative_result.score - minimal_filtering_result.score
            minimal_filtering_scores.append(minimal_filtering_result.score)
            conservative_scores.append(conservative_result.score)
        
        # Analyze score vs retention relationship
        score_retention_pairs = [(r.score, r.metadata['retention_stats']['retention_rate']) 
                                for r in domain_results_list]
        
        analysis['domain_analysis'][domain] = {
            'best_config': best_result.condition,
            'best_score': best_result.score,
            'best_min_papers_ratio': best_result.metadata['min_papers_ratio'],
            'filtering_benefit': filtering_benefit,
            'score_vs_retention': score_retention_pairs,
            'optimal_retention_rate': best_result.metadata['retention_stats']['retention_rate'],
            'keyword_quality_indicators': {
                'total_keywords': best_result.metadata['retention_stats']['total_unique_keywords'],
                'avg_retention_per_paper': best_result.metadata['retention_stats']['avg_retention_per_paper']
            }
        }
        
        # Collect data for overall analysis
        for result in domain_results_list:
            all_retention_rates.append(result.metadata['retention_stats']['retention_rate'])
            all_scores.append(result.score)
    
    # Overall patterns analysis
    if minimal_filtering_scores and conservative_scores:
        analysis['filtering_patterns'] = {
            'mean_minimal_filtering_score': float(sum(minimal_filtering_scores) / len(minimal_filtering_scores)),
            'mean_conservative_score': float(sum(conservative_scores) / len(conservative_scores)),
            'mean_filtering_benefit': float(sum(conservative_scores) / len(conservative_scores) - 
                                          sum(minimal_filtering_scores) / len(minimal_filtering_scores)),
            'filtering_helps_domains': sum(1 for i in range(len(minimal_filtering_scores)) 
                                         if conservative_scores[i] > minimal_filtering_scores[i]),
            'total_domains_tested': len(minimal_filtering_scores)
        }
    
    # Quality vs retention correlation analysis
    if len(all_retention_rates) > 3 and len(all_scores) > 3:
        import numpy as np
        correlation = np.corrcoef(all_retention_rates, all_scores)[0, 1] if len(all_retention_rates) == len(all_scores) else 0
        
        analysis['quality_vs_retention_analysis'] = {
            'score_retention_correlation': float(correlation),
            'optimal_retention_range': {
                'min': min(analysis['domain_analysis'][d]['optimal_retention_rate'] 
                          for d in analysis['domain_analysis']),
                'max': max(analysis['domain_analysis'][d]['optimal_retention_rate'] 
                          for d in analysis['domain_analysis']),
                'mean': sum(analysis['domain_analysis'][d]['optimal_retention_rate'] 
                           for d in analysis['domain_analysis']) / len(analysis['domain_analysis'])
            }
        }
    
    return analysis


def run_keyword_filtering_experiment() -> str:
    """
    Run the complete keyword filtering impact experiment.
    
    Main function that orchestrates the full experiment across all test domains.
    
    Returns:
        Path to saved results file
        
    Raises:
        ValueError: If experiment fails
    """
    print("🧪 EXPERIMENT 3: KEYWORD FILTERING IMPACT ASSESSMENT")
    print("=" * 70)
    print("Research Question: What is the value of conservative keyword filtering across different data quality scenarios?")
    print(f"Test domains: {', '.join(TEST_DOMAINS)}")
    print("\nFiltering configurations:")
    for min_papers_ratio, condition_name in FILTERING_CONFIGS:
        print(f"  - {condition_name}: ratio={min_papers_ratio:.2f}")
    
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
            domain_results = run_filtering_analysis_single_domain(
                domain_name, domain_data[domain_name], base_config
            )
            all_results.extend(domain_results)
        except Exception as e:
            raise ValueError(f"Experiment failed for domain {domain_name}: {str(e)}")
    
    # Analyze results
    analysis = analyze_filtering_effectiveness(all_results)
    
    # Print summary
    print_experiment_summary(all_results, "keyword_filtering")
    
    # Print detailed analysis
    print("\n📊 KEYWORD FILTERING EFFECTIVENESS ANALYSIS")
    print("=" * 60)
    
    for domain, domain_analysis in analysis['domain_analysis'].items():
        print(f"\n🔍 {domain}:")
        print(f"   Best config: {domain_analysis['best_config']} (score={domain_analysis['best_score']:.3f})")
        print(f"   Optimal ratio: {domain_analysis['best_min_papers_ratio']:.2f}")
        print(f"   Filtering benefit: {domain_analysis['filtering_benefit']:+.3f}")
        print(f"   Optimal retention rate: {domain_analysis['optimal_retention_rate']:.1%}")
        print(f"   Total keywords: {domain_analysis['keyword_quality_indicators']['total_keywords']}")
    
    if 'filtering_patterns' in analysis:
        patterns = analysis['filtering_patterns']
        print(f"\n📈 OVERALL FILTERING PATTERNS:")
        print(f"   Minimal filtering mean score: {patterns['mean_minimal_filtering_score']:.3f}")
        print(f"   Conservative filtering mean score: {patterns['mean_conservative_score']:.3f}")
        print(f"   Mean filtering benefit: {patterns['mean_filtering_benefit']:+.3f}")
        print(f"   Domains helped by filtering: {patterns['filtering_helps_domains']}/{patterns['total_domains_tested']}")
    
    if 'quality_vs_retention_analysis' in analysis:
        quality_analysis = analysis['quality_vs_retention_analysis']
        print(f"\n🎯 QUALITY VS RETENTION ANALYSIS:")
        print(f"   Score-retention correlation: {quality_analysis['score_retention_correlation']:.3f}")
        opt_range = quality_analysis['optimal_retention_range']
        print(f"   Optimal retention range: {opt_range['min']:.1%} - {opt_range['max']:.1%} (mean: {opt_range['mean']:.1%})")
    
    # Save results
    additional_metadata = {
        'experiment_description': 'Keyword filtering impact assessment across different data quality scenarios',
        'test_domains': TEST_DOMAINS,
        'filtering_configs': FILTERING_CONFIGS,
        'base_config': base_config.__dict__,
        'analysis': analysis
    }
    
    try:
        results_file = save_experiment_results(all_results, "keyword_filtering", additional_metadata)
        return results_file
    except Exception as e:
        raise ValueError(f"Failed to save results: {str(e)}")


if __name__ == "__main__":
    try:
        results_file = run_keyword_filtering_experiment()
        print(f"\n✅ Experiment 3 completed successfully!")
        print(f"📄 Results saved to: {results_file}")
    except Exception as e:
        print(f"\n❌ Experiment 3 failed: {str(e)}")
        sys.exit(1) 