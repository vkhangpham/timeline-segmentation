#!/usr/bin/env python3
"""
Optimizer Trace Analysis Experiment - Phase 16 PARETO-01

Analyzes Bayesian optimization traces to understand how aggregation methods
affect optimization effectiveness and convergence patterns.

Following Rule 4: Tests on real data subsets before full implementation.
Following Rule 6: Fail-fast error handling with no fallbacks.
Following Rule 7: Comprehensive logging for terminal analysis.
"""

import json
import time
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
import os
import sys
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from sklearn.preprocessing import StandardScaler
from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args
import matplotlib.pyplot as plt

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from core.data_processing import process_domain_data
from core.algorithm_config import ComprehensiveAlgorithmConfig
from core.integration import run_change_detection
from core.consensus_difference_metrics import evaluate_segmentation_quality


@dataclass
class OptimizationTrace:
    """Single optimization trajectory record."""
    domain: str
    aggregation_method: str
    evaluations: List[Dict[str, Any]]
    best_score: float
    convergence_iteration: int
    final_consensus: float
    final_difference: float
    runtime: float
    n_evaluations: int


@dataclass
class TraceAnalysisResult:
    """Results of optimizer trace comparison."""
    linear_traces: List[OptimizationTrace]
    harmonic_traces: List[OptimizationTrace]
    convergence_comparison: Dict[str, Any]
    exploration_comparison: Dict[str, Any]
    efficiency_comparison: Dict[str, Any]
    landscape_utilization: Dict[str, Any]


def create_optimization_objective(domain_name: str, aggregation_method: str):
    """
    Create optimization objective function for given domain and aggregation method.
    
    Following Rule 6: Fail-fast - no try-catch blocks.
    """
    # Load domain data once (Rule 3: Real data only)
    result = process_domain_data(domain_name)
    if not result.success:
        raise ValueError(f"Failed to load domain data for {domain_name}: {result.error_message}")
    domain_data = result.domain_data
    
    # Consensus-difference weights from FEATURE-00
    consensus_weight = 0.1
    difference_weight = 0.9
    
    def objective(params):
        """Objective function for Bayesian optimization."""
        start_time = time.time()
        
        # Create configuration
        config = ComprehensiveAlgorithmConfig(
            direction_threshold=params[0],
            validation_threshold=params[1],
            similarity_min_segment_length=int(params[2]),
            similarity_max_segment_length=int(params[3]),
            keyword_min_papers_ratio=0.05,  # From FEATURE-01
            tfidf_max_features=10000,  # From FEATURE-02
        )
        
        # Run segmentation
        segmentation_results, change_detection_result = run_change_detection(
            domain_name, algorithm_config=config
        )
        
        if not segmentation_results:
            return 0.0
        
        # Extract segments
        segments = [(seg[0], seg[1]) for seg in segmentation_results['segments']]
        
        if not segments:
            # Failed segmentation - return worst possible score
            return 0.0
        
        # Convert segments to segment_papers for evaluation
        segment_papers = []
        for start_year, end_year in segments:
            papers_in_segment = []
            for paper in domain_data.papers:
                if start_year <= paper.pub_year <= end_year:
                    papers_in_segment.append(paper)
            if papers_in_segment:  # Only add non-empty segments
                segment_papers.append(tuple(papers_in_segment))
        
        if not segment_papers:
            return 0.0
        
        # Evaluate with specified aggregation method
        result = evaluate_segmentation_quality(
            segment_papers,
            final_combination_weights=(consensus_weight, difference_weight),
            aggregation_method=aggregation_method
        )
        
        execution_time = time.time() - start_time
        
        # Store detailed evaluation info for analysis
        return result.final_score
    
    return objective


def run_bayesian_optimization_trace(domain_name: str, aggregation_method: str, 
                                  n_calls: int = 30, random_state: int = 42) -> OptimizationTrace:
    """
    Run Bayesian optimization and capture detailed trace.
    
    Following Rule 4: Limited evaluations for iterative testing.
    """
    print(f"  🎯 Running {aggregation_method} optimization for {domain_name}")
    
    # Define search space
    dimensions = [
        Real(0.1, 0.9, name='direction_threshold'),
        Real(0.3, 0.95, name='validation_threshold'), 
        Real(3, 5, name='min_segment_length'),
        Real(10, 30, name='max_segment_length')
    ]
    
    # Create objective function
    objective = create_optimization_objective(domain_name, aggregation_method)
    
    start_time = time.time()
    
    # Store evaluations for trace analysis
    evaluations = []
    
    def traced_objective(params):
        """Wrapper to capture evaluation details."""
        score = objective(params)
        evaluations.append({
            'params': list(params),
            'score': score,
            'evaluation_index': len(evaluations)
        })
        return -score  # Minimize negative (maximize positive)
    
    # Run Bayesian optimization
    result = gp_minimize(
        func=traced_objective,
        dimensions=dimensions,
        n_calls=n_calls,
        random_state=random_state,
        acq_func='EI',  # Expected Improvement
        n_initial_points=5
    )
    
    runtime = time.time() - start_time
    
    # Find convergence point (when improvement becomes minimal)
    scores = [eval_info['score'] for eval_info in evaluations]
    best_scores = np.maximum.accumulate(scores)
    improvements = np.diff(best_scores)
    
    # Convergence when improvement < 1% for 3 consecutive evaluations
    convergence_iteration = len(evaluations)
    for i in range(2, len(improvements)):
        if all(imp < 0.01 for imp in improvements[i-2:i+1]):
            convergence_iteration = i + 1
            break
    
    # Get final evaluation details
    best_idx = np.argmax(scores)
    best_evaluation = evaluations[best_idx]
    
    return OptimizationTrace(
        domain=domain_name,
        aggregation_method=aggregation_method,
        evaluations=evaluations,
        best_score=best_evaluation['score'],
        convergence_iteration=convergence_iteration,
        final_consensus=0.0,  # Will be filled by detailed analysis
        final_difference=0.0,  # Will be filled by detailed analysis
        runtime=runtime,
        n_evaluations=len(evaluations)
    )


def analyze_convergence_patterns(linear_traces: List[OptimizationTrace], 
                               harmonic_traces: List[OptimizationTrace]) -> Dict[str, Any]:
    """Analyze convergence characteristics between aggregation methods."""
    
    def get_convergence_stats(traces):
        convergence_iterations = [t.convergence_iteration for t in traces]
        final_scores = [t.best_score for t in traces]
        
        return {
            'mean_convergence_iter': np.mean(convergence_iterations),
            'std_convergence_iter': np.std(convergence_iterations),
            'mean_final_score': np.mean(final_scores),
            'std_final_score': np.std(final_scores),
            'convergence_efficiency': np.mean([s/c for s, c in zip(final_scores, convergence_iterations)])
        }
    
    linear_stats = get_convergence_stats(linear_traces)
    harmonic_stats = get_convergence_stats(harmonic_traces)
    
    return {
        'linear': linear_stats,
        'harmonic': harmonic_stats,
        'convergence_speed_ratio': harmonic_stats['mean_convergence_iter'] / linear_stats['mean_convergence_iter'],
        'score_improvement_ratio': linear_stats['mean_final_score'] / harmonic_stats['mean_final_score'],
        'efficiency_ratio': linear_stats['convergence_efficiency'] / harmonic_stats['convergence_efficiency']
    }


def analyze_exploration_patterns(linear_traces: List[OptimizationTrace], 
                               harmonic_traces: List[OptimizationTrace]) -> Dict[str, Any]:
    """Analyze exploration vs exploitation patterns."""
    
    def get_exploration_stats(traces):
        all_params = []
        all_scores = []
        
        for trace in traces:
            for eval_info in trace.evaluations:
                all_params.append(eval_info['params'])
                all_scores.append(eval_info['score'])
        
        params_array = np.array(all_params)
        scores_array = np.array(all_scores)
        
        # Calculate parameter space coverage
        param_ranges = []
        for dim in range(params_array.shape[1]):
            param_range = np.max(params_array[:, dim]) - np.min(params_array[:, dim])
            param_ranges.append(param_range)
        
        # Calculate score diversity (exploration breadth)
        score_diversity = np.std(scores_array)
        
        # Calculate parameter clustering (exploitation intensity)
        param_distances = []
        for i in range(len(all_params)):
            for j in range(i+1, len(all_params)):
                dist = np.linalg.norm(np.array(all_params[i]) - np.array(all_params[j]))
                param_distances.append(dist)
        
        mean_param_distance = np.mean(param_distances)
        
        return {
            'param_coverage': np.mean(param_ranges),
            'score_diversity': score_diversity,
            'mean_param_distance': mean_param_distance,
            'exploration_efficiency': score_diversity / np.mean(param_ranges)
        }
    
    linear_stats = get_exploration_stats(linear_traces)
    harmonic_stats = get_exploration_stats(harmonic_traces)
    
    return {
        'linear': linear_stats,
        'harmonic': harmonic_stats,
        'coverage_ratio': harmonic_stats['param_coverage'] / linear_stats['param_coverage'],
        'diversity_ratio': harmonic_stats['score_diversity'] / linear_stats['score_diversity'],
        'distance_ratio': harmonic_stats['mean_param_distance'] / linear_stats['mean_param_distance']
    }


def run_optimizer_trace_experiment(domains: List[str], n_runs_per_method: int = 3) -> TraceAnalysisResult:
    """
    Run comprehensive optimizer trace analysis experiment.
    
    Following Rule 4: Limited runs for iterative testing.
    """
    print(f"🚀 Starting Optimizer Trace Analysis")
    print(f"📋 Domains: {domains}")
    print(f"🔄 Runs per method: {n_runs_per_method}")
    
    linear_traces = []
    harmonic_traces = []
    
    for domain in domains:
        print(f"\n🔍 Processing domain: {domain}")
        
        # Run multiple optimization traces for each method
        for run_idx in range(n_runs_per_method):
            print(f"  🔄 Run {run_idx + 1}/{n_runs_per_method}")
            
            # Linear aggregation trace
            linear_trace = run_bayesian_optimization_trace(
                domain, "linear", n_calls=30, random_state=42 + run_idx
            )
            linear_traces.append(linear_trace)
            
            # Harmonic aggregation trace  
            harmonic_trace = run_bayesian_optimization_trace(
                domain, "harmonic", n_calls=30, random_state=42 + run_idx + 100
            )
            harmonic_traces.append(harmonic_trace)
    
    print(f"\n📊 Analyzing convergence patterns")
    convergence_analysis = analyze_convergence_patterns(linear_traces, harmonic_traces)
    
    print(f"📊 Analyzing exploration patterns")
    exploration_analysis = analyze_exploration_patterns(linear_traces, harmonic_traces)
    
    # Additional analyses would go here...
    efficiency_analysis = {'placeholder': 'efficiency_analysis'}
    landscape_analysis = {'placeholder': 'landscape_analysis'}
    
    return TraceAnalysisResult(
        linear_traces=linear_traces,
        harmonic_traces=harmonic_traces,
        convergence_comparison=convergence_analysis,
        exploration_comparison=exploration_analysis,
        efficiency_comparison=efficiency_analysis,
        landscape_utilization=landscape_analysis
    )


def save_trace_results(results: TraceAnalysisResult, output_dir: Path):
    """Save trace analysis results to JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"optimizer_trace_analysis_{timestamp}.json"
    filepath = output_dir / filename
    
    # Convert traces to serializable format
    def trace_to_dict(trace):
        return {
            'domain': trace.domain,
            'aggregation_method': trace.aggregation_method,
            'evaluations': trace.evaluations,
            'best_score': trace.best_score,
            'convergence_iteration': trace.convergence_iteration,
            'final_consensus': trace.final_consensus,
            'final_difference': trace.final_difference,
            'runtime': trace.runtime,
            'n_evaluations': trace.n_evaluations
        }
    
    output_data = {
        'linear_traces': [trace_to_dict(t) for t in results.linear_traces],
        'harmonic_traces': [trace_to_dict(t) for t in results.harmonic_traces],
        'convergence_comparison': results.convergence_comparison,
        'exploration_comparison': results.exploration_comparison,
        'efficiency_comparison': results.efficiency_comparison,
        'landscape_utilization': results.landscape_utilization
    }
    
    with open(filepath, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"📄 Results saved to: {filepath}")
    return filepath


def print_trace_summary(results: TraceAnalysisResult):
    """Print comprehensive summary of trace analysis."""
    print(f"\n🎉 Optimizer Trace Analysis Complete!")
    print(f"📊 CONVERGENCE ANALYSIS:")
    
    conv = results.convergence_comparison
    print(f"  Linear aggregation:")
    print(f"    Mean convergence: {conv['linear']['mean_convergence_iter']:.1f} iterations")
    print(f"    Mean final score: {conv['linear']['mean_final_score']:.3f}")
    print(f"    Convergence efficiency: {conv['linear']['convergence_efficiency']:.4f}")
    
    print(f"  Harmonic aggregation:")
    print(f"    Mean convergence: {conv['harmonic']['mean_convergence_iter']:.1f} iterations")
    print(f"    Mean final score: {conv['harmonic']['mean_final_score']:.3f}")
    print(f"    Convergence efficiency: {conv['harmonic']['convergence_efficiency']:.4f}")
    
    print(f"  Comparative ratios:")
    print(f"    Convergence speed ratio (H/L): {conv['convergence_speed_ratio']:.3f}")
    print(f"    Score improvement ratio (L/H): {conv['score_improvement_ratio']:.3f}")
    print(f"    Efficiency ratio (L/H): {conv['efficiency_ratio']:.3f}")
    
    print(f"\n📊 EXPLORATION ANALYSIS:")
    expl = results.exploration_comparison
    print(f"  Linear aggregation:")
    print(f"    Parameter coverage: {expl['linear']['param_coverage']:.3f}")
    print(f"    Score diversity: {expl['linear']['score_diversity']:.3f}")
    print(f"    Mean param distance: {expl['linear']['mean_param_distance']:.3f}")
    
    print(f"  Harmonic aggregation:")
    print(f"    Parameter coverage: {expl['harmonic']['param_coverage']:.3f}")
    print(f"    Score diversity: {expl['harmonic']['score_diversity']:.3f}")
    print(f"    Mean param distance: {expl['harmonic']['mean_param_distance']:.3f}")
    
    print(f"  Comparative ratios:")
    print(f"    Coverage ratio (H/L): {expl['coverage_ratio']:.3f}")
    print(f"    Diversity ratio (H/L): {expl['diversity_ratio']:.3f}")
    print(f"    Distance ratio (H/L): {expl['distance_ratio']:.3f}")


def main():
    """Main experiment execution."""
    print("🚀 Starting Optimizer Trace Analysis Experiment")
    
    # Test domains (Rule 4: Small subset for iterative testing)
    test_domains = ['applied_mathematics', 'computer_vision']
    
    try:
        # Run experiment
        results = run_optimizer_trace_experiment(test_domains, n_runs_per_method=2)
        
        # Save results
        output_dir = Path("experiments/metric_evaluation/results")
        save_trace_results(results, output_dir)
        
        # Print summary
        print_trace_summary(results)
        
    except Exception as e:
        print(f"🚨 Experiment failed: {e}")
        raise


if __name__ == "__main__":
    main() 