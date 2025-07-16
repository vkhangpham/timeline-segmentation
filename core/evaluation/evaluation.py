"""Timeline evaluation module with objective function scoring and dual reference evaluation.

This module provides core evaluation capabilities for timeline segmentation
including objective function scoring and comprehensive evaluation against dual references.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from ..data.data_models import AcademicPeriod, TimelineAnalysisResult
from ..optimization.objective_function import compute_objective_function
from ..optimization.penalty import create_penalty_config_from_algorithm_config
from ..utils.config import AlgorithmConfig
from ..utils.logging import get_logger


# =============================================================================
# UNIFIED DATA MODEL
# =============================================================================


@dataclass(frozen=True)
class EvaluationResult:
    """Unified result structure for all evaluation types.
    
    Replaces EvaluationResult, BaselineResult, MethodMetrics, AllMethodsMetrics,
    and FinalEvaluationResult with a single flexible structure.
    """
    
    # Core identification
    name: str  # "Algorithm", "5-year", "Gemini", "Perplexity", etc.
    domain_name: Optional[str] = None
    
    # Core metrics (always present)
    objective_score: float = 0.0
    boundary_years: List[int] = field(default_factory=list)
    num_segments: int = 0
    
    # Detailed objective breakdown (optional)
    raw_objective_score: Optional[float] = None
    penalty: Optional[float] = None
    cohesion_score: Optional[float] = None
    separation_score: Optional[float] = None
    num_transitions: Optional[int] = None
    
    # Reference comparison metrics (optional)
    reference_metrics: Dict[str, float] = field(default_factory=dict)
    # Example: {"gemini_boundary_f1": 0.85, "perplexity_segment_f1": 0.72}
    
    # Implementation details (optional)
    academic_periods: Optional[List[AcademicPeriod]] = None
    methodology: Optional[str] = None
    details: Dict[str, str] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate evaluation result data."""
        if not self.name:
            raise ValueError("Evaluation result must have a name")
        if self.num_segments < 0:
            raise ValueError(f"Invalid segment count: {self.num_segments}")
        if self.penalty is not None and self.penalty < 0:
            raise ValueError(f"Invalid penalty: {self.penalty}")


@dataclass(frozen=True)
class DomainEvaluationSummary:
    """Summary of evaluation results for a single domain."""
    
    domain_name: str
    results: List[EvaluationResult]
    tolerance: int = 2
    
    def get_result(self, name: str) -> Optional[EvaluationResult]:
        """Get result by name."""
        for result in self.results:
            if result.name == name:
                return result
        return None
    
    def get_ranking(self) -> Dict[str, float]:
        """Get ranking by objective score."""
        return {result.name: result.objective_score for result in self.results}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def calculate_method_metrics_against_references(
    method_name: str,
    method_boundary_years: List[int],
    method_segments: List[Tuple[int, int]],
    objective_score: float,
    gemini_reference: Optional[EvaluationResult],
    perplexity_reference: Optional[EvaluationResult],
    tolerance: int = 2,
) -> EvaluationResult:
    """Calculate metrics for a method against both references.

    Args:
        method_name: Name of the method
        method_boundary_years: Boundary years from the method
        method_segments: Segments from the method
        objective_score: Objective score of the method
        gemini_reference: Gemini reference baseline result
        perplexity_reference: Perplexity reference baseline result
        tolerance: Year tolerance for boundary matching

    Returns:
        EvaluationResult with scores against both references
    """
    from .metrics import calculate_boundary_f1, calculate_segment_f1

    # Initialize metrics
    gemini_boundary_f1 = gemini_segment_f1 = 0.0
    perplexity_boundary_f1 = perplexity_segment_f1 = 0.0

    # Calculate metrics against Gemini reference
    if gemini_reference:
        gemini_segments = []
        for period in gemini_reference.academic_periods:
            gemini_segments.append((period.start_year, period.end_year))

        gemini_boundary_f1, _, _ = calculate_boundary_f1(
            predicted_boundaries=method_boundary_years,
            ground_truth_boundaries=gemini_reference.boundary_years,
            tolerance=tolerance,
        )

        gemini_segment_f1, _, _ = calculate_segment_f1(
            predicted_segments=method_segments,
            ground_truth_segments=gemini_segments,
            max_segments_per_match=3,
        )

    # Calculate metrics against Perplexity reference
    if perplexity_reference:
        perplexity_segments = []
        for period in perplexity_reference.academic_periods:
            perplexity_segments.append((period.start_year, period.end_year))

        perplexity_boundary_f1, _, _ = calculate_boundary_f1(
            predicted_boundaries=method_boundary_years,
            ground_truth_boundaries=perplexity_reference.boundary_years,
            tolerance=tolerance,
        )

        perplexity_segment_f1, _, _ = calculate_segment_f1(
            predicted_segments=method_segments,
            ground_truth_segments=perplexity_segments,
            max_segments_per_match=3,
        )

    return EvaluationResult(
        name=method_name,
        objective_score=objective_score,
        boundary_years=method_boundary_years,
        num_segments=len(method_segments),
        reference_metrics={
            "gemini_boundary_f1": gemini_boundary_f1,
            "gemini_segment_f1": gemini_segment_f1,
            "perplexity_boundary_f1": perplexity_boundary_f1,
            "perplexity_segment_f1": perplexity_segment_f1,
        },
        academic_periods=None, # No direct academic periods for this helper
        methodology=None,
        details={},
    )


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

    return EvaluationResult(
        name="Algorithm",
        domain_name=timeline_result.domain_name,
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
        academic_periods=academic_periods,
    )


def run_final_evaluation(
    domain_name: str,
    timeline_result: TimelineAnalysisResult,
    algorithm_config: AlgorithmConfig,
    data_directory: str = "resources",
    use_cache: bool = True,
    verbose: bool = False,
) -> DomainEvaluationSummary:
    """Run final evaluation with all methods against both references.

    This evaluation system provides comprehensive metrics:
    - Algorithm, 5-year, 10-year methods (3 total)
    - Each method gets 4 scores: boundary F1 and segment F1 vs both Gemini and Perplexity
    - Only uses fixed-year baselines as actual baselines
    - Treats Gemini and Perplexity as references

    Args:
        domain_name: Domain name to evaluate
        timeline_result: Timeline result from pipeline
        algorithm_config: Algorithm configuration
        data_directory: Directory containing data files
        use_cache: Enable baseline result caching
        verbose: Enable verbose logging

    Returns:
        DomainEvaluationSummary with comprehensive dual reference evaluation
    """
    logger = get_logger(__name__, verbose, domain_name)

    # Use algorithm config from timeline result for consistency
    eval_algorithm_config = algorithm_config
    if timeline_result.algorithm_config is not None:
        eval_algorithm_config = timeline_result.algorithm_config

    # 1. Evaluate algorithm result using consistent config
    algorithm_result = evaluate_timeline_result(
        timeline_result=timeline_result,
        algorithm_config=eval_algorithm_config,
        verbose=verbose,
    )

    # 2. Load shared academic data once
    from .baselines import (
        load_shared_academic_data,
        load_gemini_reference,
        load_perplexity_reference,
        create_fixed_year_baseline,
    )

    academic_years, min_data_year, max_data_year = load_shared_academic_data(
        domain_name=domain_name,
        algorithm_config=eval_algorithm_config,
        data_directory=data_directory,
        verbose=verbose,
    )

    # 3. Create only 5-year and 10-year baselines
    baseline_results = []

    # 5-year baseline
    try:
        five_year_baseline = create_fixed_year_baseline(
            domain_name=domain_name,
            year_interval=5,
            academic_years=academic_years,
            min_data_year=min_data_year,
            max_data_year=max_data_year,
            algorithm_config=eval_algorithm_config,
            use_cache=use_cache,
            verbose=verbose,
        )
        baseline_results.append(five_year_baseline)
    except ValueError as e:
        if verbose:
            logger.warning(f"5-year baseline failed: {e}")

    # 10-year baseline
    try:
        ten_year_baseline = create_fixed_year_baseline(
            domain_name=domain_name,
            year_interval=10,
            academic_years=academic_years,
            min_data_year=min_data_year,
            max_data_year=max_data_year,
            algorithm_config=eval_algorithm_config,
            use_cache=use_cache,
            verbose=verbose,
        )
        baseline_results.append(ten_year_baseline)
    except ValueError as e:
        if verbose:
            logger.warning(f"10-year baseline failed: {e}")

    # 4. Load references (Gemini and Perplexity)
    gemini_reference = load_gemini_reference(
        domain_name=domain_name,
        academic_years=academic_years,
        min_data_year=min_data_year,
        max_data_year=max_data_year,
        algorithm_config=eval_algorithm_config,
        use_cache=use_cache,
        verbose=verbose,
    )

    perplexity_reference = load_perplexity_reference(
        domain_name=domain_name,
        academic_years=academic_years,
        min_data_year=min_data_year,
        max_data_year=max_data_year,
        algorithm_config=eval_algorithm_config,
        use_cache=use_cache,
        verbose=verbose,
    )

    # 5. Calculate metrics for all methods against both references
    
    # Algorithm metrics
    algorithm_segments = []
    for period in timeline_result.periods:
        algorithm_segments.append((period.start_year, period.end_year))

    algorithm_metrics = calculate_method_metrics_against_references(
        method_name="Algorithm",
        method_boundary_years=algorithm_result.boundary_years,
        method_segments=algorithm_segments,
        objective_score=algorithm_result.objective_score,
        gemini_reference=gemini_reference,
        perplexity_reference=perplexity_reference,
        tolerance=2,
    )

    # Baseline metrics
    baseline_metrics = []
    for baseline in baseline_results:
        baseline_segments = []
        for period in baseline.academic_periods:
            baseline_segments.append((period.start_year, period.end_year))

        baseline_method_metrics = calculate_method_metrics_against_references(
            method_name=baseline.name,
            method_boundary_years=baseline.boundary_years,
            method_segments=baseline_segments,
            objective_score=baseline.objective_score,
            gemini_reference=gemini_reference,
            perplexity_reference=perplexity_reference,
            tolerance=2,
        )
        baseline_metrics.append(baseline_method_metrics)

    # Collect detailed information
    details = {
        "algorithm_boundaries": str(algorithm_result.boundary_years),
        "algorithm_segments": str(algorithm_segments),
    }

    if gemini_reference:
        gemini_segments = [(p.start_year, p.end_year) for p in gemini_reference.academic_periods]
        details["gemini_boundaries"] = str(gemini_reference.boundary_years)
        details["gemini_segments"] = str(gemini_segments)
    else:
        details["gemini_error"] = "Gemini reference not available"

    if perplexity_reference:
        perplexity_segments = [(p.start_year, p.end_year) for p in perplexity_reference.academic_periods]
        details["perplexity_boundaries"] = str(perplexity_reference.boundary_years)
        details["perplexity_segments"] = str(perplexity_segments)
    else:
        details["perplexity_error"] = "Perplexity reference not available"

    for i, baseline in enumerate(baseline_results):
        baseline_segments = [(p.start_year, p.end_year) for p in baseline.academic_periods]
        details[f"{baseline.name}_boundaries"] = str(baseline.boundary_years)
        details[f"{baseline.name}_segments"] = str(baseline_segments)

    all_methods_metrics = DomainEvaluationSummary(
        domain_name=domain_name,
        results=[algorithm_metrics] + baseline_metrics,
        tolerance=2,
    )

    # 6. Create ranking (only 3 methods: Algorithm, 5-year, 10-year)
    ranking = {"Algorithm": algorithm_result.objective_score}
    for baseline in baseline_results:
        ranking[baseline.name] = baseline.objective_score

    # 7. Generate comprehensive summary
    summary_parts = [
        f"Final evaluation results for {domain_name}:",
        "",
        "Objective Scores:",
        f"  Algorithm: {algorithm_result.objective_score:.3f}",
    ]

    for baseline in baseline_results:
        summary_parts.append(f"  {baseline.name}: {baseline.objective_score:.3f}")

    summary_parts.extend([
        "",
        "Auto-metrics vs Gemini reference:",
        f"  Algorithm - Boundary F1: {algorithm_metrics.reference_metrics.get('gemini_boundary_f1', 0.0):.3f}, Segment F1: {algorithm_metrics.reference_metrics.get('gemini_segment_f1', 0.0):.3f}",
    ])

    for baseline_metric in baseline_metrics:
        summary_parts.append(
            f"  {baseline_metric.name} - Boundary F1: {baseline_metric.reference_metrics.get('gemini_boundary_f1', 0.0):.3f}, Segment F1: {baseline_metric.reference_metrics.get('gemini_segment_f1', 0.0):.3f}"
        )

    summary_parts.extend([
        "",
        "Auto-metrics vs Perplexity reference:",
        f"  Algorithm - Boundary F1: {algorithm_metrics.reference_metrics.get('perplexity_boundary_f1', 0.0):.3f}, Segment F1: {algorithm_metrics.reference_metrics.get('perplexity_segment_f1', 0.0):.3f}",
    ])

    for baseline_metric in baseline_metrics:
        summary_parts.append(
            f"  {baseline_metric.name} - Boundary F1: {baseline_metric.reference_metrics.get('perplexity_boundary_f1', 0.0):.3f}, Segment F1: {baseline_metric.reference_metrics.get('perplexity_segment_f1', 0.0):.3f}"
        )

    summary = "\n".join(summary_parts)

    return DomainEvaluationSummary(
        domain_name=domain_name,
        results=[algorithm_metrics] + baseline_metrics,
        tolerance=2,
    )


# =============================================================================
# PUBLIC EVALUATION API
# =============================================================================


def evaluate_domains(
    domains: List[str],
    algorithm_config: AlgorithmConfig,
    data_directory: str = "resources",
    timeline_files: Optional[Dict[str, str]] = None,
    baseline_only: Optional[str] = None,
    verbose: bool = False,
) -> Dict[str, Optional[DomainEvaluationSummary]]:
    """Unified evaluation function for one or multiple domains.

    Args:
        domains: List of domain names to evaluate
        algorithm_config: Algorithm configuration
        data_directory: Directory containing data files
        timeline_files: Optional mapping of domain -> timeline file path
        baseline_only: If specified, only evaluate this baseline type (5-year, 10-year)
        verbose: Enable verbose logging

    Returns:
        Dictionary mapping domain names to evaluation results (None if failed)
    """
    logger = get_logger(__name__, verbose)
    results = {}

    for domain in domains:
        try:
            if baseline_only:
                # Handle baseline-only evaluation
                if baseline_only not in ["5-year", "10-year"]:
                    logger.error(f"Invalid baseline type: {baseline_only}")
                    results[domain] = None
                    continue

                from .baselines import load_shared_academic_data, create_fixed_year_baseline

                # Load shared academic data
                academic_years, min_data_year, max_data_year = load_shared_academic_data(
                    domain_name=domain,
                    algorithm_config=algorithm_config,
                    data_directory=data_directory,
                    verbose=verbose,
                )

                # Create baseline
                year_interval = 5 if baseline_only == "5-year" else 10
                baseline_result = create_fixed_year_baseline(
                    domain_name=domain,
                    year_interval=year_interval,
                    academic_years=academic_years,
                    min_data_year=min_data_year,
                    max_data_year=max_data_year,
                    algorithm_config=algorithm_config,
                    use_cache=False,
                    verbose=verbose,
                )

                if baseline_result:
                    # Display baseline result
                    print(f"\n{domain.upper()}: {baseline_result.name} baseline")
                    print(f"Objective Score: {baseline_result.objective_score:.3f}")
                    print(f"  Raw: {baseline_result.raw_objective_score:.3f}, Penalty: {baseline_result.penalty:.3f}")
                    print(f"Segments: {baseline_result.num_segments}")
                    print(f"Boundary Years: {baseline_result.boundary_years}")
                    
                    # Create minimal summary for consistency
                    results[domain] = DomainEvaluationSummary(
                        domain_name=domain,
                        results=[baseline_result],
                        tolerance=2,
                    )
                else:
                    results[domain] = None

            else:
                # Handle full evaluation
                timeline_file = timeline_files.get(domain) if timeline_files else None
                
                # Get timeline result
                if timeline_file:
                    from ..data.data_processing import load_timeline_from_file

                    timeline_result = load_timeline_from_file(
                        timeline_file=timeline_file,
                        algorithm_config=algorithm_config,
                        data_directory=data_directory,
                        verbose=verbose,
                    )

                    # Ensure domain consistency
                    if timeline_result.domain_name != domain:
                        if verbose:
                            logger.warning(f"Domain mismatch: file has '{timeline_result.domain_name}', using '{domain}'")
                        
                        from ..data.data_models import TimelineAnalysisResult
                        timeline_result = TimelineAnalysisResult(
                            domain_name=domain,
                            periods=timeline_result.periods,
                            confidence=timeline_result.confidence,
                            boundary_years=timeline_result.boundary_years,
                            narrative_evolution=timeline_result.narrative_evolution,
                            algorithm_config=timeline_result.algorithm_config,
                        )
                else:
                    # Run segmentation pipeline
                    from ..pipeline.orchestrator import analyze_timeline

                    timeline_result = analyze_timeline(
                        domain_name=domain,
                        algorithm_config=algorithm_config,
                        data_directory=data_directory,
                        segmentation_only=True,
                        verbose=verbose,
                    )

                # Run full evaluation
                evaluation_result = run_final_evaluation(
                    domain_name=domain,
                    timeline_result=timeline_result,
                    algorithm_config=algorithm_config,
                    data_directory=data_directory,
                    use_cache=True,
                    verbose=verbose,
                )

                # Save and display results
                if evaluation_result:
                    from .analysis import save_evaluation_result, display_final_evaluation_summary
                    save_evaluation_result(evaluation_result, verbose)
                    display_final_evaluation_summary(evaluation_result, verbose)

                results[domain] = evaluation_result

        except Exception as e:
            logger.error(f"Evaluation failed for {domain}: {e}")
            results[domain] = None

    return results
