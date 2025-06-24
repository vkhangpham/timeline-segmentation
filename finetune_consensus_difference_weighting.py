#!/usr/bin/env python3
"""
Consensus-Difference Weighting Fine-Tuning (Phase 15)

Based on baseline comparison findings showing difference-only optimization dominates,
this script systematically explores optimal weighting between consensus and difference metrics.

Key findings to investigate:
- difference_only: 0.611 average score (best)
- bayesian_optimization (60/40): 0.353 average score  
- consensus_only: 0.136 average score (worst)

This suggests optimal weighting should heavily favor difference metrics.
"""

import os
import sys
import json
import time
import numpy as np
from typing import Dict, Any, List, Tuple
from dataclasses import dataclass

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.data_loader import load_domain_data
from core.data_models import DomainData, Paper
from core.algorithm_config import ComprehensiveAlgorithmConfig
from core.consensus_difference_metrics import consensus_score, difference_score
from core.integration import run_change_detection
from optimize_segmentation_bayesian import (
    SuppressOutput, 
    convert_dataframe_to_domain_data,
    optimize_consensus_difference_parameters_bayesian
)


@dataclass
class WeightingExperiment:
    """Configuration for a weighting experiment."""
    consensus_weight: float
    difference_weight: float
    name: str
    description: str
    
    def __post_init__(self):
        """Validate weights sum to 1.0"""
        total = self.consensus_weight + self.difference_weight
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Weights must sum to 1.0, got {total}")


def evaluate_consensus_difference_score_weighted(
    domain_data: DomainData, 
    config: ComprehensiveAlgorithmConfig,
    consensus_weight: float = 0.6,
    difference_weight: float = 0.4
) -> Tuple[float, Dict[str, Any]]:
    """
    Evaluate consensus-difference score with custom weighting.
    
    Args:
        domain_data: Domain data for evaluation
        config: Algorithm configuration
        consensus_weight: Weight for consensus component (0.0-1.0)
        difference_weight: Weight for difference component (0.0-1.0)
    
    Returns:
        Tuple of (weighted_score, detailed_metrics)
    """
    try:
        with SuppressOutput(suppress_stdout=True):
            # Run segmentation algorithm
            segmentation_results, change_detection_result = run_change_detection(
                domain_data.domain_name, 
                granularity=config.granularity, 
                algorithm_config=config
            )
            
            # Extract segments
            if segmentation_results and 'segments' in segmentation_results:
                segments = segmentation_results['segments']
                segment_papers = []
                for segment_years in segments:
                    start_year, end_year = segment_years
                    segment_paper_list = [
                        paper for paper in domain_data.papers
                        if start_year <= paper.pub_year <= end_year
                    ]
                    if segment_paper_list:
                        segment_papers.append(tuple(segment_paper_list))
                
                if not segment_papers:
                    segment_papers = [tuple(domain_data.papers)]
            else:
                segment_papers = [tuple(domain_data.papers)]

        # Calculate consensus and difference scores
        if len(segment_papers) < 2:
            consensus_result = consensus_score(segment_papers[0])
            avg_consensus = consensus_result.value
            avg_difference = 0.0
            consensus_scores = [consensus_result.value]
            difference_scores = []
        else:
            consensus_scores = []
            difference_scores = []
            
            for segment in segment_papers:
                cons_result = consensus_score(segment)
                consensus_scores.append(cons_result.value)
            
            for i in range(len(segment_papers) - 1):
                diff_result = difference_score(segment_papers[i], segment_papers[i + 1])
                difference_scores.append(diff_result.value)
            
            avg_consensus = float(np.mean(consensus_scores))
            avg_difference = float(np.mean(difference_scores))

        # Apply custom weighting
        weighted_score = consensus_weight * avg_consensus + difference_weight * avg_difference

        detailed_metrics = {
            'consensus_score': avg_consensus,
            'difference_score': avg_difference,
            'weighted_score': weighted_score,
            'consensus_weight': consensus_weight,
            'difference_weight': difference_weight,
            'num_segments': len(segment_papers),
            'individual_consensus_scores': consensus_scores,
            'individual_difference_scores': difference_scores
        }

        return weighted_score, detailed_metrics

    except Exception as e:
        print(f"    ⚠️ Evaluation failed: {e}")
        return 0.0, {'error': str(e)}


def test_weighting_strategy(
    domain_data: DomainData, 
    domain_name: str,
    weighting_experiment: WeightingExperiment,
    optimization_evaluations: int = 30
) -> Dict[str, Any]:
    """
    Test a specific weighting strategy using Bayesian optimization.
    
    Args:
        domain_data: Domain data for testing
        domain_name: Name of the domain
        weighting_experiment: Weighting configuration to test
        optimization_evaluations: Number of Bayesian optimization evaluations
    
    Returns:
        Dictionary with optimization results and weighting analysis
    """
    print(f"  🎯 Testing {weighting_experiment.name} ({weighting_experiment.consensus_weight:.1f}/{weighting_experiment.difference_weight:.1f})...")
    
    # Temporarily modify the evaluation function to use custom weighting
    original_evaluation = None
    
    try:
        # Run optimization with custom weighting by monkey-patching the evaluation
        import optimize_segmentation_bayesian as opt_module
        
        # Store original function
        original_evaluation = opt_module.consensus_difference_evaluation_bayesian
        
        # Create custom evaluation function with our weighting
        def custom_evaluation(params, domain_data_inner, domain_name_inner):
            direction_threshold, validation_threshold, similarity_min_segment_length, similarity_max_segment_length = params
            
            if similarity_min_segment_length >= similarity_max_segment_length:
                return -1000.0
            
            config = opt_module.research_vector_to_config(
                direction_threshold, validation_threshold, 
                similarity_min_segment_length, similarity_max_segment_length
            )
            
            score, _ = evaluate_consensus_difference_score_weighted(
                domain_data_inner, config,
                weighting_experiment.consensus_weight,
                weighting_experiment.difference_weight
            )
            
            return -score  # Negative for minimization
        
        # Replace evaluation function
        opt_module.consensus_difference_evaluation_bayesian = custom_evaluation
        
        # Run optimization
        start_time = time.time()
        result = optimize_consensus_difference_parameters_bayesian(
            domain_data, domain_name, max_evaluations=optimization_evaluations
        )
        execution_time = time.time() - start_time
        
        if result.get("optimization_successful", False):
            # Get detailed evaluation with best parameters
            best_params = result["best_parameters"]
            config = ComprehensiveAlgorithmConfig(
                direction_threshold=best_params["direction_threshold"],
                validation_threshold=best_params["validation_threshold"],
                similarity_min_segment_length=best_params["similarity_min_segment_length"],
                similarity_max_segment_length=best_params["similarity_max_segment_length"]
            )
            
            final_score, detailed_metrics = evaluate_consensus_difference_score_weighted(
                domain_data, config,
                weighting_experiment.consensus_weight,
                weighting_experiment.difference_weight
            )
            
            return {
                'weighting_experiment': weighting_experiment.name,
                'consensus_weight': weighting_experiment.consensus_weight,
                'difference_weight': weighting_experiment.difference_weight,
                'optimization_successful': True,
                'best_score': final_score,
                'best_parameters': best_params,
                'execution_time': execution_time,
                'evaluations': optimization_evaluations,
                'detailed_metrics': detailed_metrics
            }
        else:
            return {
                'weighting_experiment': weighting_experiment.name,
                'optimization_successful': False,
                'error': result.get('error', 'Unknown optimization failure')
            }
    
    finally:
        # Restore original evaluation function
        if original_evaluation:
            opt_module.consensus_difference_evaluation_bayesian = original_evaluation


def run_weighting_fine_tuning(domains: List[str]) -> Dict[str, Any]:
    """
    Run systematic weighting fine-tuning across multiple domains.
    
    Args:
        domains: List of domain names to test
    
    Returns:
        Dictionary with comprehensive fine-tuning results
    """
    print("🔬 CONSENSUS-DIFFERENCE WEIGHTING FINE-TUNING")
    print("=" * 70)
    
    # Define weighting experiments based on baseline comparison insights
    weighting_experiments = [
        # Current approach
        WeightingExperiment(0.6, 0.4, "current_60_40", "Current balanced approach"),
        
        # Difference-heavy approaches (based on baseline findings)
        WeightingExperiment(0.4, 0.6, "difference_60", "Difference-favored (60%)"),
        WeightingExperiment(0.3, 0.7, "difference_70", "Difference-heavy (70%)"),
        WeightingExperiment(0.2, 0.8, "difference_80", "Difference-dominant (80%)"),
        WeightingExperiment(0.1, 0.9, "difference_90", "Almost difference-only (90%)"),
        
        # Pure approaches for comparison
        WeightingExperiment(1.0, 0.0, "consensus_only", "Consensus-only baseline"),
        WeightingExperiment(0.0, 1.0, "difference_only", "Difference-only baseline"),
        
        # Consensus-heavy approaches (for completeness)
        WeightingExperiment(0.7, 0.3, "consensus_70", "Consensus-heavy (70%)"),
        WeightingExperiment(0.8, 0.2, "consensus_80", "Consensus-dominant (80%)"),
    ]
    
    print(f"Testing {len(weighting_experiments)} weighting strategies across {len(domains)} domains:")
    for exp in weighting_experiments:
        print(f"  • {exp.name}: {exp.consensus_weight:.1f}/{exp.difference_weight:.1f} - {exp.description}")
    
    all_results = {}
    
    for domain in domains:
        print(f"\n📊 DOMAIN: {domain.upper()}")
        print("-" * 50)
        
        try:
            # Load domain data
            df = load_domain_data(domain)
            if df is None or df.empty:
                print(f"❌ No data available for {domain}")
                continue
            
            domain_data = convert_dataframe_to_domain_data(df, domain)
            print(f"✅ Loaded {len(domain_data.papers)} papers ({domain_data.year_range[0]}-{domain_data.year_range[1]})")
            
            domain_results = {}
            
            for experiment in weighting_experiments:
                try:
                    result = test_weighting_strategy(domain_data, domain, experiment, optimization_evaluations=30)
                    domain_results[experiment.name] = result
                    
                    if result.get('optimization_successful', False):
                        score = result['best_score']
                        cons_score = result['detailed_metrics']['consensus_score']
                        diff_score = result['detailed_metrics']['difference_score']
                        print(f"    ✅ {experiment.name}: {score:.3f} (C:{cons_score:.3f}, D:{diff_score:.3f})")
                    else:
                        print(f"    ❌ {experiment.name}: FAILED")
                        
                except Exception as e:
                    print(f"    ⚠️ {experiment.name}: Error - {e}")
                    domain_results[experiment.name] = {
                        'weighting_experiment': experiment.name,
                        'optimization_successful': False,
                        'error': str(e)
                    }
            
            all_results[domain] = domain_results
            
        except Exception as e:
            print(f"❌ Error processing {domain}: {e}")
            all_results[domain] = {'error': str(e)}
    
    return all_results


def analyze_weighting_results(results: Dict[str, Any]) -> None:
    """
    Analyze and summarize weighting fine-tuning results.
    
    Args:
        results: Results from run_weighting_fine_tuning
    """
    print(f"\n📊 WEIGHTING FINE-TUNING ANALYSIS")
    print("=" * 70)
    
    # Collect scores by weighting strategy
    strategy_scores = {}
    successful_domains = []
    
    for domain, domain_results in results.items():
        if 'error' not in domain_results:
            successful_domains.append(domain)
            
            for strategy, result in domain_results.items():
                if result.get('optimization_successful', False):
                    if strategy not in strategy_scores:
                        strategy_scores[strategy] = []
                    strategy_scores[strategy].append(result['best_score'])
    
    if not strategy_scores:
        print("❌ No successful results to analyze")
        return
    
    # Calculate averages and rankings
    strategy_averages = []
    for strategy, scores in strategy_scores.items():
        avg_score = np.mean(scores)
        std_score = np.std(scores)
        strategy_averages.append((strategy, avg_score, std_score, len(scores)))
    
    # Sort by average score
    strategy_averages.sort(key=lambda x: x[1], reverse=True)
    
    print(f"Results across {len(successful_domains)} domains:")
    print(f"Strategy               | Avg Score | Std Dev | Domains | Improvement")
    print("-" * 70)
    
    best_score = strategy_averages[0][1] if strategy_averages else 0
    
    for i, (strategy, avg_score, std_score, domain_count) in enumerate(strategy_averages, 1):
        improvement = ((avg_score - strategy_averages[-1][1]) / strategy_averages[-1][1] * 100) if strategy_averages[-1][1] > 0 else 0
        print(f"{i:2d}. {strategy:<15} | {avg_score:9.3f} | {std_score:7.3f} | {domain_count:7d} | {improvement:+6.1f}%")
    
    # Identify optimal weighting
    best_strategy, best_avg, best_std, best_domains = strategy_averages[0]
    print(f"\n🏆 OPTIMAL WEIGHTING: {best_strategy}")
    print(f"   Average Score: {best_avg:.3f} ± {best_std:.3f}")
    print(f"   Tested on {best_domains} domains")
    
    # Compare with current approach
    current_result = next((x for x in strategy_averages if x[0] == 'current_60_40'), None)
    if current_result:
        current_avg = current_result[1]
        improvement = ((best_avg - current_avg) / current_avg * 100) if current_avg > 0 else 0
        print(f"   Improvement over current (60/40): {improvement:+.1f}%")
    
    # Show domain-by-domain breakdown for top strategies
    print(f"\n📋 TOP 3 STRATEGIES DOMAIN BREAKDOWN:")
    for strategy, avg_score, std_score, domain_count in strategy_averages[:3]:
        print(f"\n{strategy} (avg: {avg_score:.3f}):")
        for domain in successful_domains:
            domain_results = results[domain]
            if strategy in domain_results and domain_results[strategy].get('optimization_successful', False):
                score = domain_results[strategy]['best_score']
                cons = domain_results[strategy]['detailed_metrics']['consensus_score']
                diff = domain_results[strategy]['detailed_metrics']['difference_score']
                print(f"  {domain:<25}: {score:.3f} (C:{cons:.3f}, D:{diff:.3f})")


def save_weighting_results(results: Dict[str, Any]) -> None:
    """Save weighting fine-tuning results to JSON file."""
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    output_file = f"results/weighting_finetuning_{timestamp}.json"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    save_data = {
        'weighting_finetuning_results': results,
        'metadata': {
            'finetuning_date': time.strftime('%Y-%m-%d %H:%M:%S'),
            'framework': 'Phase 15 Consensus-Difference Metrics',
            'optimization_method': 'Bayesian Optimization',
            'evaluations_per_strategy': 30,
            'motivation': 'Baseline comparison showed difference-only optimization dominates (0.611 vs 0.353 for 60/40 weighting)'
        }
    }
    
    with open(output_file, 'w') as f:
        json.dump(save_data, f, indent=2)
    
    print(f"\n💾 Weighting fine-tuning results saved to {output_file}")


def main():
    """Main function."""
    domains_to_test = [
        'machine_learning',
        'computer_vision', 
        'natural_language_processing'
    ]
    
    if len(sys.argv) > 1:
        domains_to_test = sys.argv[1:]
    
    print("🔬 Starting systematic weighting fine-tuning based on baseline comparison insights...")
    print("📋 Key insight: difference_only (0.611) >> bayesian_60_40 (0.353) >> consensus_only (0.136)")
    print("🎯 Goal: Find optimal consensus-difference weighting for maximum performance")
    
    # Run fine-tuning
    results = run_weighting_fine_tuning(domains_to_test)
    
    # Analyze results
    analyze_weighting_results(results)
    
    # Save results
    save_weighting_results(results)


if __name__ == "__main__":
    main() 