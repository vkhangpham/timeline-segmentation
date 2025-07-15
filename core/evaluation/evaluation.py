"""Timeline evaluation module with objective function scoring and baseline comparisons.

This module provides comprehensive evaluation capabilities for timeline segmentation
including objective function scoring, baseline creation, and auto-metrics calculation.
"""

import json
from pathlib import Path
from typing import Dict, List, NamedTuple

from ..data.data_models import AcademicPeriod, TimelineAnalysisResult
from ..optimization.objective_function import compute_objective_function
from ..optimization.penalty import create_penalty_config_from_algorithm_config
from ..utils.config import AlgorithmConfig
from ..utils.logging import get_logger


# =============================================================================
# DATA MODELS
# =============================================================================


class EvaluationResult(NamedTuple):
    """Result of timeline evaluation with objective function score."""

    objective_score: float
    raw_objective_score: float  # Before penalty
    penalty: float  # Penalty applied
    cohesion_score: float
    separation_score: float
    num_segments: int
    num_transitions: int
    boundary_years: List[int]
    methodology: str
    details: Dict[str, str]


class BaselineResult(NamedTuple):
    """Result of baseline evaluation."""

    baseline_name: str
    objective_score: float
    raw_objective_score: float  # Before penalty
    penalty: float  # Penalty applied
    cohesion_score: float
    separation_score: float
    num_segments: int
    boundary_years: List[int]
    academic_periods: List[AcademicPeriod]


class AutoMetricResult(NamedTuple):
    """Result of auto-metric evaluation."""

    boundary_f1: float
    boundary_precision: float
    boundary_recall: float
    segment_f1: float
    segment_precision: float
    segment_recall: float
    tolerance: int
    details: Dict[str, str]


class ComprehensiveEvaluationResult(NamedTuple):
    """Complete evaluation result with all metrics."""

    domain_name: str
    algorithm_result: EvaluationResult
    baseline_results: List[BaselineResult]
    auto_metrics: AutoMetricResult
    ranking: Dict[str, float]  # baseline_name -> objective_score
    summary: str


# =============================================================================
# CONFIGURATION AND PENALTY CALCULATION
# =============================================================================


# =============================================================================
# CORE EVALUATION LOGIC
# =============================================================================


def evaluate_timeline_result(
    timeline_result: TimelineAnalysisResult,
    algorithm_config: AlgorithmConfig,
    verbose: bool = False,
) -> EvaluationResult:
    """Evaluate a timeline result using the unified penalty system.

    Args:
        timeline_result: Timeline analysis result to evaluate
        algorithm_config: Algorithm configuration (may be overridden by timeline config)
        verbose: Enable verbose logging

    Returns:
        EvaluationResult with unified penalty-based scoring
    """
    logger = get_logger(__name__, verbose)

    academic_periods = list(timeline_result.periods)

    # Use algorithm config from timeline result if available (for consistency)
    config_to_use = algorithm_config
    if timeline_result.algorithm_config is not None:
        config_to_use = timeline_result.algorithm_config
        if verbose:
            logger.info(
                "Using algorithm config from timeline result for consistent evaluation"
            )

    # Create penalty configuration from the appropriate config
    penalty_config = create_penalty_config_from_algorithm_config(config_to_use)

    # Compute objective function with unified penalty system
    obj_result = compute_objective_function(
        academic_periods=academic_periods,
        algorithm_config=config_to_use,
        penalty_config=penalty_config,
        verbose=verbose,
    )

    # Extract boundary years from periods
    boundary_years = []
    for i, period in enumerate(academic_periods):
        if i == 0:
            boundary_years.append(period.start_year)
        boundary_years.append(period.end_year)
    boundary_years = sorted(set(boundary_years))

    if verbose:
        logger.info(
            f"Eval: {obj_result.num_segments} segments, final={obj_result.final_score:.3f}, raw={obj_result.raw_score:.3f}, penalty={obj_result.penalty:.3f}"
        )

    return EvaluationResult(
        objective_score=obj_result.final_score,
        raw_objective_score=obj_result.raw_score,
        penalty=obj_result.penalty,
        cohesion_score=obj_result.cohesion_score,
        separation_score=obj_result.separation_score,
        num_segments=obj_result.num_segments,
        num_transitions=obj_result.num_transitions,
        boundary_years=boundary_years,
        methodology=obj_result.methodology,
        details={
            "cohesion_details": obj_result.cohesion_details,
            "separation_details": obj_result.separation_details,
        },
    )


def run_comprehensive_evaluation(
    domain_name: str,
    timeline_result: TimelineAnalysisResult,
    algorithm_config: AlgorithmConfig,
    data_directory: str = "resources",
    verbose: bool = False,
) -> ComprehensiveEvaluationResult:
    """Run comprehensive evaluation including all baselines and auto-metrics.

    Args:
        domain_name: Domain name to evaluate
        timeline_result: Timeline result from pipeline
        algorithm_config: Algorithm configuration
        data_directory: Directory containing data files
        verbose: Enable verbose logging

    Returns:
        ComprehensiveEvaluationResult with all evaluation metrics
    """
    logger = get_logger(__name__, verbose, domain_name)

    if verbose:
        logger.info(f"Comprehensive eval: {domain_name}")

    # Import baseline and metrics functions
    from .baselines import (
        create_gemini_baseline,
        create_manual_baseline,
        create_fixed_year_baseline,
    )
    from .metrics import calculate_boundary_f1, calculate_segment_f1

    # 1. Evaluate algorithm result
    algorithm_result = evaluate_timeline_result(
        timeline_result=timeline_result,
        algorithm_config=algorithm_config,
        verbose=verbose,
    )

    # 2. Create and evaluate baselines
    baseline_results = []

    # Gemini baseline
    try:
        gemini_baseline = create_gemini_baseline(
            domain_name=domain_name,
            algorithm_config=algorithm_config,
            data_directory=data_directory,
            verbose=verbose,
        )
        baseline_results.append(gemini_baseline)
    except (FileNotFoundError, ValueError) as e:
        if verbose:
            logger.warning(f"Gemini baseline failed: {e}")

    # Manual baseline
    try:
        manual_baseline = create_manual_baseline(
            domain_name=domain_name,
            algorithm_config=algorithm_config,
            data_directory=data_directory,
            verbose=verbose,
        )
        baseline_results.append(manual_baseline)
    except (FileNotFoundError, ValueError) as e:
        if verbose:
            logger.warning(f"Manual baseline failed: {e}")

    # 5-year baseline
    try:
        five_year_baseline = create_fixed_year_baseline(
            domain_name=domain_name,
            algorithm_config=algorithm_config,
            year_interval=5,
            data_directory=data_directory,
            verbose=verbose,
        )
        baseline_results.append(five_year_baseline)
    except (ValueError, RuntimeError) as e:
        if verbose:
            logger.warning(f"5-year baseline failed: {e}")

    # 10-year baseline
    try:
        ten_year_baseline = create_fixed_year_baseline(
            domain_name=domain_name,
            algorithm_config=algorithm_config,
            year_interval=10,
            data_directory=data_directory,
            verbose=verbose,
        )
        baseline_results.append(ten_year_baseline)
    except (ValueError, RuntimeError) as e:
        if verbose:
            logger.warning(f"10-year baseline failed: {e}")

    # 3. Calculate auto-metrics against manual baseline
    auto_metrics = None
    manual_baseline_result = None

    for baseline in baseline_results:
        if baseline.baseline_name == "Manual":
            manual_baseline_result = baseline
            break

    if manual_baseline_result is not None:
        # Extract segments from timeline result
        algorithm_segments = []
        for period in timeline_result.periods:
            algorithm_segments.append((period.start_year, period.end_year))

        # Extract segments from manual baseline
        manual_segments = []
        for period in manual_baseline_result.academic_periods:
            manual_segments.append((period.start_year, period.end_year))

        # Calculate boundary F1
        boundary_f1, boundary_precision, boundary_recall = calculate_boundary_f1(
            predicted_boundaries=algorithm_result.boundary_years,
            ground_truth_boundaries=manual_baseline_result.boundary_years,
            tolerance=2,
        )

        # Calculate segment F1
        segment_f1, segment_precision, segment_recall = calculate_segment_f1(
            predicted_segments=algorithm_segments,
            ground_truth_segments=manual_segments,
            max_segments_per_match=3,
        )

        auto_metrics = AutoMetricResult(
            boundary_f1=boundary_f1,
            boundary_precision=boundary_precision,
            boundary_recall=boundary_recall,
            segment_f1=segment_f1,
            segment_precision=segment_precision,
            segment_recall=segment_recall,
            tolerance=2,
            details={
                "algorithm_boundaries": str(algorithm_result.boundary_years),
                "manual_boundaries": str(manual_baseline_result.boundary_years),
                "algorithm_segments": str(algorithm_segments),
                "manual_segments": str(manual_segments),
            },
        )
    else:
        # Create default auto-metrics if manual baseline not available
        auto_metrics = AutoMetricResult(
            boundary_f1=0.0,
            boundary_precision=0.0,
            boundary_recall=0.0,
            segment_f1=0.0,
            segment_precision=0.0,
            segment_recall=0.0,
            tolerance=2,
            details={"error": "Manual baseline not available"},
        )

    # 4. Create ranking
    ranking = {"Algorithm": algorithm_result.objective_score}
    for baseline in baseline_results:
        ranking[baseline.baseline_name] = baseline.objective_score

    # 5. Generate summary
    summary_parts = [
        f"Evaluation Results for {domain_name}:",
        f"Algorithm Score: {algorithm_result.objective_score:.3f}",
    ]

    if baseline_results:
        summary_parts.append("Baseline Scores:")
        for baseline in baseline_results:
            summary_parts.append(
                f"  {baseline.baseline_name}: {baseline.objective_score:.3f}"
            )

    if auto_metrics and manual_baseline_result:
        summary_parts.extend(
            [
                f"Auto-Metrics vs Manual:",
                f"  Boundary F1: {auto_metrics.boundary_f1:.3f}",
                f"  Segment F1: {auto_metrics.segment_f1:.3f}",
            ]
        )

    summary = "\n".join(summary_parts)

    if verbose:
        logger.info("Comprehensive eval complete")
        logger.info(summary)

    return ComprehensiveEvaluationResult(
        domain_name=domain_name,
        algorithm_result=algorithm_result,
        baseline_results=baseline_results,
        auto_metrics=auto_metrics,
        ranking=ranking,
        summary=summary,
    )


def run_single_evaluation(
    domain_name: str,
    algorithm_config: AlgorithmConfig,
    data_directory: str = "resources",
    timeline_file: str = None,
    verbose: bool = False,
) -> bool:
    """Run evaluation for a single domain.

    Args:
        domain_name: Domain name to evaluate
        algorithm_config: Algorithm configuration
        data_directory: Directory containing data files
        timeline_file: Optional path to existing timeline file
        verbose: Enable verbose logging

    Returns:
        True if evaluation succeeded, False otherwise
    """
    logger = get_logger(__name__, verbose, domain_name)

    try:
        # Get timeline result - either from file or by running the algorithm
        if timeline_file:
            logger.info(f"Loading timeline: {timeline_file}")
            from ..data.data_processing import load_timeline_from_file

            timeline_result = load_timeline_from_file(
                timeline_file=timeline_file,
                algorithm_config=algorithm_config,
                data_directory=data_directory,
                verbose=verbose,
            )

            # Verify domain name matches
            if timeline_result.domain_name != domain_name:
                logger.warning(
                    f"Domain name mismatch: file contains '{timeline_result.domain_name}' but evaluating '{domain_name}'"
                )
                # Update domain name to match the requested evaluation
                timeline_result = TimelineAnalysisResult(
                    domain_name=domain_name,
                    periods=timeline_result.periods,
                    confidence=timeline_result.confidence,
                    boundary_years=timeline_result.boundary_years,
                    narrative_evolution=timeline_result.narrative_evolution,
                )
        else:
            # Run segmentation-only pipeline to get timeline result
            logger.info(f"Running segmentation pipeline for {domain_name}")
            from ..pipeline.orchestrator import analyze_timeline

            timeline_result = analyze_timeline(
                domain_name=domain_name,
                algorithm_config=algorithm_config,
                data_directory=data_directory,
                segmentation_only=True,
                verbose=verbose,
            )

        # Run comprehensive evaluation
        logger.info(f"Comprehensive eval: {domain_name}")
        evaluation_result = run_comprehensive_evaluation(
            domain_name=domain_name,
            timeline_result=timeline_result,
            algorithm_config=algorithm_config,
            data_directory=data_directory,
            verbose=verbose,
        )

        # Save and display results
        from .analysis import save_evaluation_result, display_evaluation_summary

        save_evaluation_result(evaluation_result, verbose)
        display_evaluation_summary(evaluation_result, verbose)

        return True

    except Exception as e:
        logger.error(f"Evaluation failed for {domain_name}: {e}")
        return False


def run_all_domains_evaluation(
    algorithm_config: AlgorithmConfig,
    data_directory: str = "resources",
    verbose: bool = False,
) -> bool:
    """Run evaluation for all available domains using existing timeline files.

    Args:
        algorithm_config: Algorithm configuration
        data_directory: Directory containing data files
        verbose: Enable verbose logging

    Returns:
        True if at least one domain succeeded, False otherwise
    """
    logger = get_logger(__name__, verbose, "all_domains")

    from ..utils.general import (
        discover_available_timeline_domains,
        get_timeline_file_path,
    )

    # Discover domains from timeline files instead of resources directory
    domains = discover_available_timeline_domains(verbose)
    if not domains:
        logger.error("No timeline files found in results/timelines directory")
        logger.error("Please run timeline analysis first to generate timeline files")
        return False

    successful = []
    failed = []
    domain_results = {}

    logger.info(f"COMPREHENSIVE EVALUATION")
    logger.info("=" * 50)
    logger.info(
        f"Processing {len(domains)} domains with existing timeline files: {', '.join(domains)}"
    )

    for domain in domains:
        logger.info(f"Processing {domain}...")

        # Get timeline file path
        timeline_file = get_timeline_file_path(domain, verbose)
        if not timeline_file:
            logger.error(f"Timeline file not found for {domain}")
            failed.append(domain)
            domain_results[domain] = None
            continue

        # Load domain-specific config
        domain_config = AlgorithmConfig.from_config_file(domain_name=domain)

        # Run evaluation using existing timeline file
        evaluation_success = run_single_evaluation(
            domain_name=domain,
            algorithm_config=domain_config,
            data_directory=data_directory,
            timeline_file=timeline_file,  # Always use existing timeline file
            verbose=verbose,
        )

        if evaluation_success:
            successful.append(domain)

            # Load the evaluation result for cross-domain analysis
            try:
                results_file = Path("results/evaluation") / f"{domain}_evaluation.json"
                if results_file.exists():
                    with open(results_file, "r") as f:
                        eval_data = json.load(f)

                    # Convert back to result objects for analysis
                    from types import SimpleNamespace

                    # Create algorithm result
                    algorithm_result = SimpleNamespace(
                        objective_score=eval_data["algorithm_result"][
                            "objective_score"
                        ],
                        raw_objective_score=eval_data["algorithm_result"].get(
                            "raw_objective_score",
                            eval_data["algorithm_result"]["objective_score"],
                        ),
                        penalty=eval_data["algorithm_result"].get("penalty", 0.0),
                        cohesion_score=eval_data["algorithm_result"]["cohesion_score"],
                        separation_score=eval_data["algorithm_result"][
                            "separation_score"
                        ],
                        num_segments=eval_data["algorithm_result"]["num_segments"],
                        boundary_years=eval_data["algorithm_result"]["boundary_years"],
                    )

                    # Create baseline results
                    baseline_results = []
                    for baseline_data in eval_data["baseline_results"]:
                        baseline_result = SimpleNamespace(
                            baseline_name=baseline_data["baseline_name"],
                            objective_score=baseline_data["objective_score"],
                            raw_objective_score=baseline_data.get(
                                "raw_objective_score", baseline_data["objective_score"]
                            ),
                            penalty=baseline_data.get("penalty", 0.0),
                            cohesion_score=baseline_data["cohesion_score"],
                            separation_score=baseline_data["separation_score"],
                            num_segments=baseline_data["num_segments"],
                            boundary_years=baseline_data["boundary_years"],
                        )
                        baseline_results.append(baseline_result)

                    # Create auto-metrics
                    auto_metrics = SimpleNamespace(
                        boundary_f1=eval_data["auto_metrics"]["boundary_f1"],
                        boundary_precision=eval_data["auto_metrics"][
                            "boundary_precision"
                        ],
                        boundary_recall=eval_data["auto_metrics"]["boundary_recall"],
                        segment_f1=eval_data["auto_metrics"]["segment_f1"],
                        segment_precision=eval_data["auto_metrics"][
                            "segment_precision"
                        ],
                        segment_recall=eval_data["auto_metrics"]["segment_recall"],
                    )

                    # Create comprehensive result
                    comprehensive_result = SimpleNamespace(
                        domain_name=domain,
                        algorithm_result=algorithm_result,
                        baseline_results=baseline_results,
                        auto_metrics=auto_metrics,
                    )

                    domain_results[domain] = comprehensive_result

            except Exception as e:
                logger.warning(f"Could not load evaluation results for {domain}: {e}")
                domain_results[domain] = None
        else:
            failed.append(domain)
            domain_results[domain] = None

    logger.info(f"EVALUATION COMPLETE")
    logger.info("=" * 30)
    logger.info(f"Success: {len(successful)}/{len(domains)} domains")

    if successful:
        logger.info(f"Processed: {', '.join(successful)}")
        logger.info("Results saved in 'results/evaluation/' directory")

    if failed:
        logger.warning(f"Failed: {', '.join(failed)}")

    # Display cross-domain analysis if we have multiple successful domains
    if len(successful) > 1:
        from .analysis import display_cross_domain_analysis

        display_cross_domain_analysis(domain_results, verbose)

    return len(successful) > 0


def run_baseline_only_evaluation(
    domain_name: str,
    baseline_type: str,
    algorithm_config: AlgorithmConfig,
    data_directory: str = "resources",
    verbose: bool = False,
) -> bool:
    """Run evaluation for a specific baseline only.

    Args:
        domain_name: Domain name to evaluate
        baseline_type: Type of baseline (gemini, manual, 5-year, 10-year)
        algorithm_config: Algorithm configuration
        data_directory: Directory containing data files
        verbose: Enable verbose logging

    Returns:
        True if evaluation succeeded, False otherwise
    """
    logger = get_logger(__name__, verbose, domain_name)

    from .baselines import (
        create_gemini_baseline,
        create_manual_baseline,
        create_fixed_year_baseline,
    )

    try:
        if baseline_type == "gemini":
            baseline_result = create_gemini_baseline(
                domain_name=domain_name,
                algorithm_config=algorithm_config,
                data_directory=data_directory,
                verbose=verbose,
            )
        elif baseline_type == "manual":
            baseline_result = create_manual_baseline(
                domain_name=domain_name,
                algorithm_config=algorithm_config,
                data_directory=data_directory,
                verbose=verbose,
            )
        elif baseline_type == "5-year":
            baseline_result = create_fixed_year_baseline(
                domain_name=domain_name,
                algorithm_config=algorithm_config,
                year_interval=5,
                data_directory=data_directory,
                verbose=verbose,
            )
        elif baseline_type == "10-year":
            baseline_result = create_fixed_year_baseline(
                domain_name=domain_name,
                algorithm_config=algorithm_config,
                year_interval=10,
                data_directory=data_directory,
                verbose=verbose,
            )
        else:
            logger.error(f"Invalid baseline type: {baseline_type}")
            return False

        # Display baseline result
        print(f"\n{'='*50}")
        print(f"BASELINE EVALUATION: {domain_name} ({baseline_type})")
        print(f"{'='*50}")
        print(f"Objective Score: {baseline_result.objective_score:.3f}")
        print(f"Cohesion Score: {baseline_result.cohesion_score:.3f}")
        print(f"Separation Score: {baseline_result.separation_score:.3f}")
        print(f"Number of Segments: {baseline_result.num_segments}")
        print(f"Boundary Years: {baseline_result.boundary_years}")

        return True

    except Exception as e:
        logger.error(f"Baseline evaluation failed for {domain_name}: {e}")
        return False
