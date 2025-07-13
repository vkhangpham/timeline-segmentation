"""Optimization utilities for parameter tuning with caching and trial scoring.

Simple functions for domain-specific parameter optimization with cached data loading.
"""

import time
from typing import Dict, Any, List
import dataclasses

from ..data.data_models import AcademicYear
from ..data.data_processing import load_domain_data
from ..pipeline.orchestrator import analyze_timeline
from ..evaluation.evaluation import evaluate_timeline_result
from ..utils.config import AlgorithmConfig
from ..utils.logging import get_logger
from .penalty import PenaltyConfig, create_penalty_config_from_dict


# Global cache for academic years
_cached_academic_years: Dict[str, List[AcademicYear]] = {}


def load_cached_academic_years(
    domain_name: str,
    algorithm_config: AlgorithmConfig,
    data_directory: str = "resources",
    verbose: bool = False,
) -> List[AcademicYear]:
    """Load and cache academic years for a domain.

    Args:
        domain_name: Name of the domain
        algorithm_config: Algorithm configuration for data loading
        data_directory: Directory containing domain data
        verbose: Enable verbose logging

    Returns:
        List of AcademicYear objects
    """
    logger = get_logger(__name__, verbose, domain_name)

    cache_key = f"{domain_name}_{data_directory}"

    if cache_key in _cached_academic_years:
        if verbose:
            logger.info(f"Using cached academic years for {domain_name}")
        return _cached_academic_years[cache_key]

    if verbose:
        logger.info(f"Loading academic years for {domain_name}")

    success, academic_years, error_message = load_domain_data(
        domain_name=domain_name,
        algorithm_config=algorithm_config,
        data_directory=data_directory,
        min_papers_per_year=5,
        apply_year_filtering=True,
        verbose=verbose,
    )

    if not success:
        raise RuntimeError(f"Failed to load data for {domain_name}: {error_message}")

    if not academic_years:
        raise RuntimeError(f"No academic years found for {domain_name}")

    _cached_academic_years[cache_key] = academic_years

    if verbose:
        logger.info(f"Cached {len(academic_years)} academic years for {domain_name}")

    return academic_years


def create_trial_config(
    base_config: AlgorithmConfig,
    parameter_overrides: Dict[str, Any],
) -> AlgorithmConfig:
    """Create a new AlgorithmConfig with parameter overrides.

    Args:
        base_config: Base configuration to start from
        parameter_overrides: Dictionary of parameters to override

    Returns:
        New AlgorithmConfig with overridden parameters
    """
    # Convert base config to dictionary
    config_dict = dataclasses.asdict(base_config)

    # Apply parameter overrides for any existing AlgorithmConfig field
    for param_name, value in parameter_overrides.items():
        if param_name in config_dict:
            config_dict[param_name] = value

    # Always set direction_threshold_strategy to "fixed" for optimization
    config_dict["direction_threshold_strategy"] = "fixed"

    return AlgorithmConfig(**config_dict)


def get_validation_metrics(
    timeline_result,
    domain_name: str,
    verbose: bool = False,
) -> tuple[float, float]:
    """Get Boundary-F1 and Segment-F1 metrics if manual reference exists.

    Args:
        timeline_result: Timeline result to evaluate
        domain_name: Domain name for reference lookup
        verbose: Enable verbose logging

    Returns:
        Tuple of (boundary_f1, segment_f1) or (0.0, 0.0) if no reference
    """
    logger = get_logger(__name__, verbose, domain_name)

    try:
        from pathlib import Path
        import json

        manual_file = Path(f"data/references/{domain_name}_manual.json")
        if not manual_file.exists():
            if verbose:
                logger.info(f"No manual reference found for {domain_name}")
            return 0.0, 0.0

        with open(manual_file, "r") as f:
            manual_data = json.load(f)

        # Extract segments from manual data
        periods = manual_data["historical_periods"]
        manual_segments = []
        manual_boundaries = []

        for period in periods:
            years_str = period["years"]
            if "-" in years_str:
                start_year, end_year = map(int, years_str.split("-"))
                manual_segments.append((start_year, end_year))
                manual_boundaries.extend([start_year, end_year])

        manual_boundaries = sorted(set(manual_boundaries))

        # Extract algorithm results
        algorithm_segments = []
        algorithm_boundaries = []

        for period in timeline_result.periods:
            algorithm_segments.append((period.start_year, period.end_year))
            algorithm_boundaries.extend([period.start_year, period.end_year])

        algorithm_boundaries = sorted(set(algorithm_boundaries))

        # Calculate metrics
        from ..evaluation.evaluation import calculate_boundary_f1, calculate_segment_f1

        boundary_f1, _, _ = calculate_boundary_f1(
            predicted_boundaries=algorithm_boundaries,
            ground_truth_boundaries=manual_boundaries,
            tolerance=2,
        )

        segment_f1, _, _ = calculate_segment_f1(
            predicted_segments=algorithm_segments,
            ground_truth_segments=manual_segments,
            max_segments_per_match=3,
        )

        if verbose:
            logger.info(
                f"Validation metrics: Boundary-F1={boundary_f1:.3f}, Segment-F1={segment_f1:.3f}"
            )

        return boundary_f1, segment_f1

    except Exception as e:
        if verbose:
            logger.warning(f"Failed to compute validation metrics: {e}")
        return 0.0, 0.0


def score_trial(
    domain_name: str,
    parameter_overrides: Dict[str, Any],
    base_config: AlgorithmConfig,
    trial_id: int,
    optimization_config: Dict[str, Any] = None,
    data_directory: str = "resources",
    verbose: bool = False,
) -> Dict[str, Any]:
    """Score a single trial with given parameters."""

    # Suppress algorithm logging if not in verbose mode
    if not verbose:
        import logging

        # Save original log levels
        original_levels = {}
        loggers_to_suppress = [
            "",  # Root logger
            "core",
            "core.segmentation",
            "core.segmentation.segmentation",
            "core.pipeline.orchestrator",
            "core.data.data_processing",
            "core.segmentation.change_point_detection",
            "core.segmentation.beam_refinement",
            "core.segment_modeling.segment_modeling",
            "core.utils.general",
            "core.utils.logging",
            "core.data",
            "core.pipeline",
        ]

        for logger_name in loggers_to_suppress:
            logger = logging.getLogger(logger_name)
            original_levels[logger_name] = logger.level
            logger.setLevel(logging.ERROR)  # Only show errors

    start_time = time.time()

    try:
        # Load and cache academic years for this domain
        academic_years = load_cached_academic_years(
            domain_name=domain_name,
            algorithm_config=base_config,
            data_directory=data_directory,
            verbose=verbose,
        )

        # Create trial configuration
        trial_config = create_trial_config(base_config, parameter_overrides)

        # Run timeline analysis (segmentation only)
        timeline_result = analyze_timeline(
            domain_name=domain_name,
            algorithm_config=trial_config,
            data_directory=data_directory,
            segmentation_only=True,  # Skip characterization for optimization
            verbose=verbose,
        )

        # Create penalty configuration from optimization config
        penalty_config = create_penalty_config_from_dict(optimization_config or {})

        # Evaluate the result using unified penalty system
        from ..optimization.objective_function import compute_objective_function
        objective_result = compute_objective_function(
            timeline_result.periods,
            trial_config,
            penalty_config=penalty_config,
            verbose=verbose,
        )

        # Use the penalized score as the final objective
        final_objective_score = objective_result.final_score

        # Get validation metrics
        boundary_f1, segment_f1 = get_validation_metrics(
            timeline_result=timeline_result,
            domain_name=domain_name,
            verbose=verbose,
        )

        return {
            "trial_id": trial_id,
            "parameters": parameter_overrides,
            "objective_score": final_objective_score,
            "raw_score": objective_result.raw_score,
            "penalty": objective_result.penalty,
            "scaled_score": objective_result.scaled_score,
            "cohesion_score": objective_result.cohesion_score,
            "separation_score": objective_result.separation_score,
            "num_segments": len(timeline_result.periods),
            "boundary_f1": boundary_f1,
            "segment_f1": segment_f1,
            "runtime_seconds": time.time() - start_time,
            "error_message": None,
        }

    except Exception as e:
        logger = get_logger(__name__, verbose, domain_name)
        fail_score = (
            (optimization_config or {}).get("scoring", {}).get("fail_score", -10.0)
        )
        if verbose:
            logger.error(f"Trial {trial_id} failed: {e}")
        return {
            "trial_id": trial_id,
            "parameters": parameter_overrides,
            "objective_score": fail_score,
            "raw_score": fail_score,
            "penalty": 0.0,
            "scaled_score": 0.0,
            "cohesion_score": 0.0,
            "separation_score": 0.0,
            "num_segments": 0,
            "boundary_f1": 0.0,
            "segment_f1": 0.0,
            "runtime_seconds": time.time() - start_time,
            "error_message": str(e),
        }

    finally:
        # Restore original log levels if they were changed
        if not verbose and "original_levels" in locals():
            for logger_name, original_level in original_levels.items():
                logging.getLogger(logger_name).setLevel(original_level)


def clear_cache():
    """Clear cached academic years."""
    global _cached_academic_years
    _cached_academic_years.clear()
