"""Timeline evaluation analysis, display, and I/O functions.

This module contains functions for cross-domain analysis, result display,
and saving/loading evaluation results.
"""

import json
from pathlib import Path
from typing import Dict, Optional
from collections import defaultdict

from ..utils.logging import get_logger
from .evaluation import ComprehensiveEvaluationResult
from .metrics import calculate_f1_score_between_methods


def save_evaluation_result(
    evaluation_result: ComprehensiveEvaluationResult, verbose: bool = False
):
    """Save evaluation result to JSON file.

    Args:
        evaluation_result: ComprehensiveEvaluationResult object
        verbose: Enable verbose logging
    """
    logger = get_logger(__name__, verbose, evaluation_result.domain_name)

    # Create results directory
    results_dir = Path("results/evaluation")
    results_dir.mkdir(parents=True, exist_ok=True)

    # Prepare data for JSON serialization
    result_data = {
        "domain_name": evaluation_result.domain_name,
        "algorithm_result": {
            "objective_score": evaluation_result.algorithm_result.objective_score,
            "raw_objective_score": evaluation_result.algorithm_result.raw_objective_score,
            "penalty": evaluation_result.algorithm_result.penalty,
            "cohesion_score": evaluation_result.algorithm_result.cohesion_score,
            "separation_score": evaluation_result.algorithm_result.separation_score,
            "num_segments": evaluation_result.algorithm_result.num_segments,
            "num_transitions": evaluation_result.algorithm_result.num_transitions,
            "boundary_years": evaluation_result.algorithm_result.boundary_years,
            "methodology": evaluation_result.algorithm_result.methodology,
            "details": evaluation_result.algorithm_result.details,
        },
        "baseline_results": [],
        "auto_metrics": {
            "boundary_f1": evaluation_result.auto_metrics.boundary_f1,
            "boundary_precision": evaluation_result.auto_metrics.boundary_precision,
            "boundary_recall": evaluation_result.auto_metrics.boundary_recall,
            "segment_f1": evaluation_result.auto_metrics.segment_f1,
            "segment_precision": evaluation_result.auto_metrics.segment_precision,
            "segment_recall": evaluation_result.auto_metrics.segment_recall,
            "tolerance": evaluation_result.auto_metrics.tolerance,
            "details": evaluation_result.auto_metrics.details,
        },
        "ranking": evaluation_result.ranking,
        "summary": evaluation_result.summary,
    }

    # Add baseline results
    for baseline in evaluation_result.baseline_results:
        baseline_data = {
            "baseline_name": baseline.baseline_name,
            "objective_score": baseline.objective_score,
            "raw_objective_score": baseline.raw_objective_score,
            "penalty": baseline.penalty,
            "cohesion_score": baseline.cohesion_score,
            "separation_score": baseline.separation_score,
            "num_segments": baseline.num_segments,
            "boundary_years": baseline.boundary_years,
        }
        result_data["baseline_results"].append(baseline_data)

    # Save to file
    output_file = results_dir / f"{evaluation_result.domain_name}_evaluation.json"
    with open(output_file, "w") as f:
        json.dump(result_data, f, indent=2)

    logger.info(f"Evaluation results saved to {output_file}")


def display_evaluation_summary(
    evaluation_result: ComprehensiveEvaluationResult, verbose: bool = False
):
    """Display evaluation summary.

    Args:
        evaluation_result: ComprehensiveEvaluationResult object
        verbose: Enable verbose logging
    """
    logger = get_logger(__name__, verbose, evaluation_result.domain_name)

    print(f"\n{'='*60}")
    print(f"EVALUATION SUMMARY: {evaluation_result.domain_name}")
    print(f"{'='*60}")

    # Algorithm result
    alg_result = evaluation_result.algorithm_result
    print(f"\nALGORITHM RESULT:")
    print(f"-" * 20)
    print(
        f"Objective Score: {alg_result.objective_score:.3f} (Raw: {alg_result.raw_objective_score:.3f}, Penalty: {alg_result.penalty:.3f})"
    )
    print(f"Cohesion Score: {alg_result.cohesion_score:.3f}")
    print(f"Separation Score: {alg_result.separation_score:.3f}")
    print(f"Number of Segments: {alg_result.num_segments}")
    print(f"Boundary Years: {alg_result.boundary_years}")

    # Baseline results
    if evaluation_result.baseline_results:
        print(f"\nBASELINE RESULTS:")
        print(f"-" * 20)
        for baseline in evaluation_result.baseline_results:
            print(
                f"{baseline.baseline_name}: {baseline.objective_score:.3f} "
                f"(Raw: {baseline.raw_objective_score:.3f}, Penalty: {baseline.penalty:.3f}, "
                f"{baseline.num_segments} segments)"
            )

    # Auto-metrics
    auto_metrics = evaluation_result.auto_metrics
    print(f"\nAUTO-METRICS (vs Manual):")
    print(f"-" * 30)
    print(
        f"Boundary F1: {auto_metrics.boundary_f1:.3f} "
        f"(P: {auto_metrics.boundary_precision:.3f}, R: {auto_metrics.boundary_recall:.3f})"
    )
    print(
        f"Segment F1: {auto_metrics.segment_f1:.3f} "
        f"(P: {auto_metrics.segment_precision:.3f}, R: {auto_metrics.segment_recall:.3f})"
    )
    print(f"Tolerance: ±{auto_metrics.tolerance} years")

    # Ranking
    print(f"\nRANKING (by Objective Score):")
    print(f"-" * 30)
    sorted_ranking = sorted(
        evaluation_result.ranking.items(), key=lambda x: x[1], reverse=True
    )
    for i, (name, score) in enumerate(sorted_ranking):
        print(f"{i+1}. {name}: {score:.3f}")

    print(f"\n{'='*60}")


def display_cross_domain_analysis(
    domain_results: Dict[str, Optional[object]], verbose: bool = False
):
    """Display cross-domain analysis and rankings.

    Args:
        domain_results: Dictionary mapping domain names to their evaluation results
        verbose: Enable verbose logging
    """
    logger = get_logger(__name__, verbose, "cross_domain")

    if len(domain_results) < 2:
        logger.info("Cross-domain analysis requires at least 2 domains")
        return

    # Collect metrics across domains
    method_scores = defaultdict(list)
    method_auto_metrics = defaultdict(lambda: defaultdict(list))
    method_boundaries = defaultdict(list)  # Store boundary years for F1 calculation
    manual_boundaries = []  # Store manual baseline boundaries

    for domain_name, result in domain_results.items():
        if result is None:
            continue

        # Collect objective scores
        method_scores["Algorithm"].append(result.algorithm_result.objective_score)
        method_boundaries["Algorithm"].append(result.algorithm_result.boundary_years)

        for baseline in result.baseline_results:
            method_scores[baseline.baseline_name].append(baseline.objective_score)
            method_boundaries[baseline.baseline_name].append(baseline.boundary_years)

            # Store manual boundaries for F1 calculation
            if baseline.baseline_name == "Manual":
                manual_boundaries.append(baseline.boundary_years)

        # Collect auto-metrics (Algorithm vs Manual)
        auto_metrics = result.auto_metrics
        method_auto_metrics["Algorithm"]["boundary_f1"].append(auto_metrics.boundary_f1)
        method_auto_metrics["Algorithm"]["boundary_precision"].append(
            auto_metrics.boundary_precision
        )
        method_auto_metrics["Algorithm"]["boundary_recall"].append(
            auto_metrics.boundary_recall
        )
        method_auto_metrics["Algorithm"]["segment_f1"].append(auto_metrics.segment_f1)
        method_auto_metrics["Algorithm"]["segment_precision"].append(
            auto_metrics.segment_precision
        )
        method_auto_metrics["Algorithm"]["segment_recall"].append(
            auto_metrics.segment_recall
        )

    # Calculate F1 scores for all methods vs manual baseline
    method_f1_scores = defaultdict(lambda: defaultdict(list))

    for method_name, boundaries_list in method_boundaries.items():
        if method_name == "Manual":
            continue  # Skip manual vs manual comparison

        for i, method_boundaries_single in enumerate(boundaries_list):
            if i < len(manual_boundaries):
                manual_boundaries_single = manual_boundaries[i]
                f1_result = calculate_f1_score_between_methods(
                    method_boundaries_single, manual_boundaries_single, tolerance=2
                )

                for metric_name, value in f1_result.items():
                    method_f1_scores[method_name][metric_name].append(value)

    # Calculate averages
    method_avg_scores = {}
    method_std_scores = {}

    for method, scores in method_scores.items():
        if scores:
            avg_score = sum(scores) / len(scores)
            std_score = (sum((x - avg_score) ** 2 for x in scores) / len(scores)) ** 0.5
            method_avg_scores[method] = avg_score
            method_std_scores[method] = std_score

    # Calculate F1 score averages
    method_f1_avg = {}
    method_f1_std = {}

    for method, metrics in method_f1_scores.items():
        method_f1_avg[method] = {}
        method_f1_std[method] = {}

        for metric_name, values in metrics.items():
            if values:
                avg_val = sum(values) / len(values)
                std_val = (sum((x - avg_val) ** 2 for x in values) / len(values)) ** 0.5
                method_f1_avg[method][metric_name] = avg_val
                method_f1_std[method][metric_name] = std_val

    # Calculate auto-metrics averages
    auto_metrics_avg = {}
    auto_metrics_std = {}

    for method, metrics in method_auto_metrics.items():
        auto_metrics_avg[method] = {}
        auto_metrics_std[method] = {}

        for metric_name, values in metrics.items():
            if values:
                avg_val = sum(values) / len(values)
                std_val = (sum((x - avg_val) ** 2 for x in values) / len(values)) ** 0.5
                auto_metrics_avg[method][metric_name] = avg_val
                auto_metrics_std[method][metric_name] = std_val

    # Display results
    print(f"\n{'='*80}")
    print(f"CROSS-DOMAIN ANALYSIS ({len(domain_results)} domains)")
    print(f"{'='*80}")

    # Domain breakdown
    print(f"\nDOMAIN BREAKDOWN:")
    print(f"-" * 40)
    for domain_name, result in domain_results.items():
        if result is None:
            print(f"{domain_name}: FAILED")
            continue

        algo_score = result.algorithm_result.objective_score
        auto_f1 = result.auto_metrics.boundary_f1
        print(f"{domain_name}: Algorithm={algo_score:.3f}, Boundary F1={auto_f1:.3f}")

    # Average objective scores
    print(f"\nAVERAGE OBJECTIVE SCORES:")
    print(f"-" * 40)
    sorted_methods = sorted(method_avg_scores.items(), key=lambda x: x[1], reverse=True)

    for i, (method, avg_score) in enumerate(sorted_methods):
        std_score = method_std_scores.get(method, 0.0)
        print(f"{i+1}. {method}: {avg_score:.3f} (±{std_score:.3f})")

    # F1 scores vs Manual baseline
    print(f"\nF1 SCORES VS MANUAL BASELINE:")
    print(f"-" * 40)

    # Sort methods by boundary F1 score
    boundary_f1_rankings = []
    for method, f1_metrics in method_f1_avg.items():
        if "boundary_f1" in f1_metrics:
            boundary_f1_rankings.append((method, f1_metrics["boundary_f1"]))

    boundary_f1_rankings.sort(key=lambda x: x[1], reverse=True)

    for i, (method, boundary_f1) in enumerate(boundary_f1_rankings):
        std_f1 = method_f1_std.get(method, {}).get("boundary_f1", 0.0)

        # Get segment F1 for this method
        segment_f1 = method_f1_avg.get(method, {}).get("segment_f1", 0.0)
        segment_f1_std = method_f1_std.get(method, {}).get("segment_f1", 0.0)

        print(
            f"{i+1}. {method}: Boundary F1={boundary_f1:.3f} (±{std_f1:.3f}), Segment F1={segment_f1:.3f} (±{segment_f1_std:.3f})"
        )

    # Auto-metrics for Algorithm (original implementation)
    if "Algorithm" in auto_metrics_avg:
        print(f"\nAUTO-METRICS (Algorithm vs Manual - Original Implementation):")
        print(f"-" * 60)
        algo_metrics = auto_metrics_avg["Algorithm"]
        algo_std = auto_metrics_std["Algorithm"]

        boundary_f1 = algo_metrics.get("boundary_f1", 0.0)
        boundary_f1_std = algo_std.get("boundary_f1", 0.0)
        boundary_p = algo_metrics.get("boundary_precision", 0.0)
        boundary_p_std = algo_std.get("boundary_precision", 0.0)
        boundary_r = algo_metrics.get("boundary_recall", 0.0)
        boundary_r_std = algo_std.get("boundary_recall", 0.0)

        segment_f1 = algo_metrics.get("segment_f1", 0.0)
        segment_f1_std = algo_std.get("segment_f1", 0.0)
        segment_p = algo_metrics.get("segment_precision", 0.0)
        segment_p_std = algo_std.get("segment_precision", 0.0)
        segment_r = algo_metrics.get("segment_recall", 0.0)
        segment_r_std = algo_std.get("segment_recall", 0.0)

        print(f"Boundary F1: {boundary_f1:.3f} (±{boundary_f1_std:.3f})")
        print(f"  Precision: {boundary_p:.3f} (±{boundary_p_std:.3f})")
        print(f"  Recall:    {boundary_r:.3f} (±{boundary_r_std:.3f})")
        print(f"Segment F1:  {segment_f1:.3f} (±{segment_f1_std:.3f})")
        print(f"  Precision: {segment_p:.3f} (±{segment_p_std:.3f})")
        print(f"  Recall:    {segment_r:.3f} (±{segment_r_std:.3f})")

    # Performance summary
    print(f"\nPERFORMANCE SUMMARY:")
    print(f"-" * 30)

    # Best method by objective score
    if len(sorted_methods) > 0:
        best_method, best_score = sorted_methods[0]
        print(f"Best Method (Objective Score): {best_method} ({best_score:.3f})")

    # Best method by F1 score
    if len(boundary_f1_rankings) > 0:
        best_f1_method, best_f1_score = boundary_f1_rankings[0]
        print(f"Best Method (F1 vs Manual): {best_f1_method} ({best_f1_score:.3f})")

    # Algorithm performance
    if "Algorithm" in method_avg_scores:
        algo_score = method_avg_scores["Algorithm"]
        algo_rank = next(
            (
                i + 1
                for i, (method, _) in enumerate(sorted_methods)
                if method == "Algorithm"
            ),
            None,
        )
        print(f"Algorithm Rank (Objective): #{algo_rank} ({algo_score:.3f})")

        # Algorithm F1 rank
        algo_f1_rank = next(
            (
                i + 1
                for i, (method, _) in enumerate(boundary_f1_rankings)
                if method == "Algorithm"
            ),
            None,
        )
        algo_f1_score = method_f1_avg.get("Algorithm", {}).get("boundary_f1", 0.0)
        print(f"Algorithm Rank (F1): #{algo_f1_rank} ({algo_f1_score:.3f})")
    # Save cross-domain results
    save_cross_domain_results(
        domain_results, method_avg_scores, auto_metrics_avg, method_f1_avg, verbose
    )

    print(f"\n{'='*80}")


def save_cross_domain_results(
    domain_results: Dict[str, Optional[object]],
    method_avg_scores: Dict[str, float],
    auto_metrics_avg: Dict[str, Dict[str, float]],
    method_f1_avg: Dict[str, Dict[str, float]],
    verbose: bool = False,
):
    """Save cross-domain analysis results.

    Args:
        domain_results: Dictionary mapping domain names to their evaluation results
        method_avg_scores: Average objective scores for each method
        auto_metrics_avg: Average auto-metrics for each method
        method_f1_avg: Average F1 scores for each method vs manual baseline
        verbose: Enable verbose logging
    """
    logger = get_logger(__name__, verbose, "cross_domain")

    results_dir = Path("results/evaluation")
    results_dir.mkdir(parents=True, exist_ok=True)

    # Prepare cross-domain summary
    cross_domain_data = {
        "num_domains": len(domain_results),
        "domains": list(domain_results.keys()),
        "method_averages": method_avg_scores,
        "auto_metrics_averages": auto_metrics_avg,
        "method_f1_averages": method_f1_avg,
        "domain_breakdown": {},
    }

    # Add domain breakdown
    for domain_name, result in domain_results.items():
        if result is None:
            cross_domain_data["domain_breakdown"][domain_name] = {"status": "failed"}
            continue

        domain_data = {
            "status": "success",
            "algorithm_score": result.algorithm_result.objective_score,
            "baseline_scores": {
                baseline.baseline_name: baseline.objective_score
                for baseline in result.baseline_results
            },
            "auto_metrics": {
                "boundary_f1": result.auto_metrics.boundary_f1,
                "boundary_precision": result.auto_metrics.boundary_precision,
                "boundary_recall": result.auto_metrics.boundary_recall,
                "segment_f1": result.auto_metrics.segment_f1,
                "segment_precision": result.auto_metrics.segment_precision,
                "segment_recall": result.auto_metrics.segment_recall,
            },
        }
        cross_domain_data["domain_breakdown"][domain_name] = domain_data

    # Save to file
    output_file = results_dir / "cross_domain_analysis.json"
    with open(output_file, "w") as f:
        json.dump(cross_domain_data, f, indent=2)

    logger.info(f"Cross-domain analysis saved to {output_file}")
