"""Timeline evaluation analysis, display, and I/O functions.

This module contains functions for cross-domain analysis, result display,
and saving/loading evaluation results.
"""

import json
from pathlib import Path
from typing import Dict, Optional

from ..utils.logging import get_logger
from .evaluation import DomainEvaluationSummary


def save_evaluation_result(evaluation_result, verbose: bool = False):
    """Save evaluation result to JSON file.

    Args:
        evaluation_result: DomainEvaluationSummary object
        verbose: Enable verbose logging
    """
    logger = get_logger(__name__, verbose, evaluation_result.domain_name)

    # Create results directory
    results_dir = Path("results/evaluation")
    results_dir.mkdir(parents=True, exist_ok=True)

    # Find algorithm and baseline results
    algorithm_result = evaluation_result.get_result("Algorithm")
    baseline_results = [r for r in evaluation_result.results if r.name in ["5-year", "10-year"]]
    
    if algorithm_result is None:
        logger.error("No algorithm result found in evaluation summary")
        return

    # Prepare data for JSON serialization
    result_data = {
        "domain_name": evaluation_result.domain_name,
        "algorithm_result": {
            "objective_score": algorithm_result.objective_score,
            "raw_objective_score": algorithm_result.raw_objective_score,
            "penalty": algorithm_result.penalty,
            "cohesion_score": algorithm_result.cohesion_score,
            "separation_score": algorithm_result.separation_score,
            "num_segments": algorithm_result.num_segments,
            "num_transitions": algorithm_result.num_transitions,
            "boundary_years": algorithm_result.boundary_years,
            "methodology": algorithm_result.methodology,
            "details": algorithm_result.details,
        },
        "baseline_results": [],
        "all_methods_metrics": {
            "algorithm_metrics": {
                "method_name": algorithm_result.name,
                "objective_score": algorithm_result.objective_score,
                "gemini_boundary_f1": algorithm_result.reference_metrics.get("gemini_boundary_f1", 0.0),
                "gemini_segment_f1": algorithm_result.reference_metrics.get("gemini_segment_f1", 0.0),
                "perplexity_boundary_f1": algorithm_result.reference_metrics.get("perplexity_boundary_f1", 0.0),
                "perplexity_segment_f1": algorithm_result.reference_metrics.get("perplexity_segment_f1", 0.0),
            },
            "baseline_metrics": [],
            "tolerance": evaluation_result.tolerance,
            "details": {},
        },
        "ranking": evaluation_result.get_ranking(),
        "summary": "",  # Will be generated if needed
    }

    # Add baseline results
    for baseline in baseline_results:
        baseline_data = {
            "baseline_name": baseline.name,
            "objective_score": baseline.objective_score,
            "raw_objective_score": baseline.raw_objective_score,
            "penalty": baseline.penalty,
            "cohesion_score": baseline.cohesion_score,
            "separation_score": baseline.separation_score,
            "num_segments": baseline.num_segments,
            "boundary_years": baseline.boundary_years,
        }
        result_data["baseline_results"].append(baseline_data)

    # Add baseline metrics
    for baseline in baseline_results:
        baseline_metric_data = {
            "method_name": baseline.name,
            "objective_score": baseline.objective_score,
            "gemini_boundary_f1": baseline.reference_metrics.get("gemini_boundary_f1", 0.0),
            "gemini_segment_f1": baseline.reference_metrics.get("gemini_segment_f1", 0.0),
            "perplexity_boundary_f1": baseline.reference_metrics.get("perplexity_boundary_f1", 0.0),
            "perplexity_segment_f1": baseline.reference_metrics.get("perplexity_segment_f1", 0.0),
        }
        result_data["all_methods_metrics"]["baseline_metrics"].append(baseline_metric_data)

    # Save to file
    output_file = results_dir / f"{evaluation_result.domain_name}_evaluation.json"
    with open(output_file, "w") as f:
        json.dump(result_data, f, indent=2)

    if verbose:
        logger.info(f"Results saved: {output_file}")


def display_cross_domain_analysis(
    domain_results: Dict[str, Optional[object]], verbose: bool = False
):
    """Display cross-domain analysis in a concise tabular format.

    Args:
        domain_results: Dictionary mapping domain names to DomainEvaluationSummary objects
        verbose: Enable verbose output
    """
    print("\n" + "=" * 80)
    print("CROSS-DOMAIN EVALUATION ANALYSIS")
    print("=" * 80)

    # Filter successful results
    successful_results = {
        domain: result
        for domain, result in domain_results.items()
        if result is not None
    }

    if not successful_results:
        print("No successful evaluations found.")
        return

    print(f"\nDomains evaluated: {len(successful_results)}")
    print(f"Domains: {', '.join(sorted(successful_results.keys()))}")

    # Collect metrics for each method across domains
    method_metrics = {
        "Algorithm": {"objective": [], "gemini_boundary_f1": [], "gemini_segment_f1": [], "perplexity_boundary_f1": [], "perplexity_segment_f1": []},
        "5-year": {"objective": [], "gemini_boundary_f1": [], "gemini_segment_f1": [], "perplexity_boundary_f1": [], "perplexity_segment_f1": []},
        "10-year": {"objective": [], "gemini_boundary_f1": [], "gemini_segment_f1": [], "perplexity_boundary_f1": [], "perplexity_segment_f1": []},
    }

    for domain_name, result in successful_results.items():
        if result is None:
            continue

        # Extract method results from the unified structure
        for eval_result in result.results:
            if eval_result.name in method_metrics:
                method_metrics[eval_result.name]["objective"].append(eval_result.objective_score)
                method_metrics[eval_result.name]["gemini_boundary_f1"].append(
                    eval_result.reference_metrics.get("gemini_boundary_f1", 0.0)
                )
                method_metrics[eval_result.name]["gemini_segment_f1"].append(
                    eval_result.reference_metrics.get("gemini_segment_f1", 0.0)
                )
                method_metrics[eval_result.name]["perplexity_boundary_f1"].append(
                    eval_result.reference_metrics.get("perplexity_boundary_f1", 0.0)
                )
                method_metrics[eval_result.name]["perplexity_segment_f1"].append(
                    eval_result.reference_metrics.get("perplexity_segment_f1", 0.0)
                )

    # Calculate averages
    method_averages = {}
    for method_name, metrics in method_metrics.items():
        method_averages[method_name] = {}
        for metric_name, values in metrics.items():
            if values:
                method_averages[method_name][metric_name] = sum(values) / len(values)
            else:
                method_averages[method_name][metric_name] = 0.0

    # Find best scores for highlighting
    best_objective = max(method_averages.values(), key=lambda x: x.get("objective", 0))["objective"]
    best_gemini_boundary = max(method_averages.values(), key=lambda x: x.get("gemini_boundary_f1", 0))["gemini_boundary_f1"]
    best_gemini_segment = max(method_averages.values(), key=lambda x: x.get("gemini_segment_f1", 0))["gemini_segment_f1"]
    best_perplexity_boundary = max(method_averages.values(), key=lambda x: x.get("perplexity_boundary_f1", 0))["perplexity_boundary_f1"]
    best_perplexity_segment = max(method_averages.values(), key=lambda x: x.get("perplexity_segment_f1", 0))["perplexity_segment_f1"]

    # Display table
    print(f"\nCross-Domain Performance Summary ({len(successful_results)} domains)")
    print("-" * 120)
    
    header = f"{'Method':<12} {'Objective':<12} {'Gem-Bound':<12} {'Gem-Seg':<10} {'Perp-Bound':<12} {'Perp-Seg':<10}"
    print(header)
    print("-" * len(header))

    for method_name in ["Algorithm", "5-year", "10-year"]:
        if method_name in method_averages and any(method_averages[method_name].values()):
            avg = method_averages[method_name]
            
            # Add highlighting for best scores
            obj_str = f"{avg['objective']:.3f}"
            if abs(avg['objective'] - best_objective) < 0.001:
                obj_str += " ⭐"
                
            gb_str = f"{avg['gemini_boundary_f1']:.3f}"
            if abs(avg['gemini_boundary_f1'] - best_gemini_boundary) < 0.001:
                gb_str += " ⭐"
                
            gs_str = f"{avg['gemini_segment_f1']:.3f}"
            if abs(avg['gemini_segment_f1'] - best_gemini_segment) < 0.001:
                gs_str += " ⭐"

            pb_str = f"{avg['perplexity_boundary_f1']:.3f}"
            if abs(avg['perplexity_boundary_f1'] - best_perplexity_boundary) < 0.001:
                pb_str += " ⭐"

            ps_str = f"{avg['perplexity_segment_f1']:.3f}"
            if abs(avg['perplexity_segment_f1'] - best_perplexity_segment) < 0.001:
                ps_str += " ⭐"

            print(f"{method_name:<12} {obj_str:<12} {gb_str:<12} {gs_str:<10} {pb_str:<12} {ps_str:<10}")

    print("\n⭐ = Best score in category")
    print("Gem = Gemini reference, Perp = Perplexity reference")
    print("Bound = Boundary F1, Seg = Segment F1")


def display_final_evaluation_summary(
    evaluation_result: DomainEvaluationSummary, verbose: bool = False
):
    """Display final evaluation results in concise tabular format.

    Args:
        evaluation_result: Domain evaluation summary with unified structure
        verbose: Enable verbose output
    """
    print("\n" + "=" * 80)
    print(f"FINAL EVALUATION RESULTS: {evaluation_result.domain_name.upper()}")
    print("=" * 80)

    # Collect all methods from the unified structure
    all_methods = evaluation_result.results

    # Display objective scores table
    print("\nObjective Scores:")
    print("-" * 40)
    print(f"{'Method':<15} {'Objective Score':<15}")
    print("-" * 40)
    
    for method in all_methods:
        print(f"{method.name:<15} {method.objective_score:<15.3f}")

    # Display metrics vs both references
    print(f"\nAuto-metrics vs References (tolerance = {evaluation_result.tolerance} years):")
    print("-" * 80)
    
    # Header
    header = f"{'Method':<12} {'Gemini Boundary':<15} {'Gemini Segment':<14} {'Perplexity Boundary':<19} {'Perplexity Segment':<16}"
    print(header)
    print("-" * len(header))
    
    # Data rows
    for method in all_methods:
        gemini_boundary = method.reference_metrics.get("gemini_boundary_f1", 0.0)
        gemini_segment = method.reference_metrics.get("gemini_segment_f1", 0.0)
        perplexity_boundary = method.reference_metrics.get("perplexity_boundary_f1", 0.0)
        perplexity_segment = method.reference_metrics.get("perplexity_segment_f1", 0.0)
        
        print(f"{method.name:<12} "
              f"{gemini_boundary:<15.3f} "
              f"{gemini_segment:<14.3f} "
              f"{perplexity_boundary:<19.3f} "
              f"{perplexity_segment:<16.3f}")

    # Find best scores for summary
    best_objective = max(method.objective_score for method in all_methods)
    best_gemini_boundary = max(method.reference_metrics.get("gemini_boundary_f1", 0.0) for method in all_methods)
    best_gemini_segment = max(method.reference_metrics.get("gemini_segment_f1", 0.0) for method in all_methods)
    best_perplexity_boundary = max(method.reference_metrics.get("perplexity_boundary_f1", 0.0) for method in all_methods)
    best_perplexity_segment = max(method.reference_metrics.get("perplexity_segment_f1", 0.0) for method in all_methods)

    print(f"\nBest Scores:")
    print(f"  Objective: {best_objective:.3f}")
    print(f"  Gemini Boundary F1: {best_gemini_boundary:.3f}")
    print(f"  Gemini Segment F1: {best_gemini_segment:.3f}")
    print(f"  Perplexity Boundary F1: {best_perplexity_boundary:.3f}")
    print(f"  Perplexity Segment F1: {best_perplexity_segment:.3f}")

    if verbose:
        print(f"\nDetailed Information:")
        # Display details from any method that has them
        for method in all_methods:
            if method.details:
                print(f"\n{method.name} Details:")
                for key, value in method.details.items():
                    print(f"  {key}: {value}")
