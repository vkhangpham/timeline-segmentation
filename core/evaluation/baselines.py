"""Timeline evaluation baseline creation.

This module contains functions for creating different types of baselines
for timeline evaluation including Gemini, Manual, and Fixed-year baselines.
"""

import json
from pathlib import Path

from ..data.data_processing import (
    load_domain_data,
    create_academic_periods_from_segments,
)
from ..optimization.objective_function import compute_objective_function
from ..utils.config import AlgorithmConfig
from ..utils.logging import get_logger
from .evaluation import load_penalty_configuration, compute_penalty, BaselineResult


def create_gemini_baseline(
    domain_name: str,
    algorithm_config: AlgorithmConfig,
    data_directory: str = "resources",
    verbose: bool = False,
) -> BaselineResult:
    """Create baseline using Gemini reference timeline.

    Args:
        domain_name: Domain name to evaluate
        algorithm_config: Algorithm configuration
        data_directory: Directory containing data files
        verbose: Enable verbose logging

    Returns:
        BaselineResult with Gemini baseline evaluation
    """
    logger = get_logger(__name__, verbose, domain_name)

    # Load Gemini reference file
    gemini_file = Path(f"data/references/{domain_name}_gemini.json")

    if not gemini_file.exists():
        raise FileNotFoundError(f"Gemini reference file not found: {gemini_file}")

    with open(gemini_file, "r") as f:
        gemini_data = json.load(f)

    # Extract year boundaries from Gemini data
    periods = gemini_data["historical_periods"]
    segments = []

    for period in periods:
        start_year = period["start_year"]
        end_year = period["end_year"]
        segments.append((start_year, end_year))

    if verbose:
        logger.info(f"Gemini baseline: {len(segments)} segments")
        logger.info(f"Segments: {segments}")

    # Load domain data with same processing as pipeline
    success, academic_years, error_message = load_domain_data(
        domain_name=domain_name,
        algorithm_config=algorithm_config,
        data_directory=data_directory,
        verbose=verbose,
    )

    if not success:
        raise RuntimeError(f"Failed to load domain data: {error_message}")

    # Filter academic years to match data range
    data_years = set(year.year for year in academic_years)
    min_data_year = min(data_years)
    max_data_year = max(data_years)

    # Adjust segments to fit within data range
    adjusted_segments = []
    for start_year, end_year in segments:
        # Clip to data range
        adj_start = max(start_year, min_data_year)
        adj_end = min(end_year, max_data_year)

        # Only include segments that have data
        if adj_start <= adj_end and adj_start in data_years:
            adjusted_segments.append((adj_start, adj_end))

    if not adjusted_segments:
        raise ValueError(
            f"No Gemini segments overlap with data range {min_data_year}-{max_data_year}"
        )

    # Create academic periods from segments
    academic_periods = create_academic_periods_from_segments(
        academic_years=tuple(academic_years),
        segments=adjusted_segments,
        algorithm_config=algorithm_config,
    )

    # Compute objective function
    obj_result = compute_objective_function(
        academic_periods=academic_periods,
        algorithm_config=algorithm_config,
        verbose=verbose,
    )

    # Load penalty configuration and compute penalty
    penalty_config = load_penalty_configuration()
    penalty = compute_penalty(academic_periods, penalty_config)

    # Apply penalty to get final score
    final_score = obj_result.final_score - penalty

    # Extract boundary years
    boundary_years = []
    for i, period in enumerate(academic_periods):
        if i == 0:
            boundary_years.append(period.start_year)
        boundary_years.append(period.end_year)
    boundary_years = sorted(set(boundary_years))

    if verbose:
        logger.info(
            f"Gemini baseline raw objective score: {obj_result.final_score:.3f}"
        )
        logger.info(f"Gemini baseline penalty applied: {penalty:.3f}")
        logger.info(f"Gemini baseline final objective score: {final_score:.3f}")

    return BaselineResult(
        baseline_name="Gemini",
        objective_score=final_score,
        raw_objective_score=obj_result.final_score,
        penalty=penalty,
        cohesion_score=obj_result.cohesion_score,
        separation_score=obj_result.separation_score,
        num_segments=len(academic_periods),
        boundary_years=boundary_years,
        academic_periods=academic_periods,
    )


def create_manual_baseline(
    domain_name: str,
    algorithm_config: AlgorithmConfig,
    data_directory: str = "resources",
    verbose: bool = False,
) -> BaselineResult:
    """Create baseline using manual reference timeline.

    Args:
        domain_name: Domain name to evaluate
        algorithm_config: Algorithm configuration
        data_directory: Directory containing data files
        verbose: Enable verbose logging

    Returns:
        BaselineResult with manual baseline evaluation
    """
    logger = get_logger(__name__, verbose, domain_name)

    # Load manual reference file
    manual_file = Path(f"data/references/{domain_name}_manual.json")

    if not manual_file.exists():
        raise FileNotFoundError(f"Manual reference file not found: {manual_file}")

    with open(manual_file, "r") as f:
        manual_data = json.load(f)

    # Extract year boundaries from manual data (different format)
    periods = manual_data["historical_periods"]
    segments = []

    for period in periods:
        years_str = period["years"]
        # Parse years like "1400-1600" or "1946-2025"
        if "-" in years_str:
            start_year, end_year = map(int, years_str.split("-"))
            segments.append((start_year, end_year))

    if verbose:
        logger.info(f"Manual baseline: {len(segments)} segments")
        logger.info(f"Segments: {segments}")

    # Load domain data with same processing as pipeline
    success, academic_years, error_message = load_domain_data(
        domain_name=domain_name,
        algorithm_config=algorithm_config,
        data_directory=data_directory,
        verbose=verbose,
    )

    if not success:
        raise RuntimeError(f"Failed to load domain data: {error_message}")

    # Filter academic years to match data range
    data_years = set(year.year for year in academic_years)
    min_data_year = min(data_years)
    max_data_year = max(data_years)

    # Adjust segments to fit within data range
    adjusted_segments = []
    for start_year, end_year in segments:
        # Clip to data range
        adj_start = max(start_year, min_data_year)
        adj_end = min(end_year, max_data_year)

        # Only include segments that have data
        if adj_start <= adj_end and adj_start in data_years:
            adjusted_segments.append((adj_start, adj_end))

    if not adjusted_segments:
        raise ValueError(
            f"No manual segments overlap with data range {min_data_year}-{max_data_year}"
        )

    # Create academic periods from segments
    academic_periods = create_academic_periods_from_segments(
        academic_years=tuple(academic_years),
        segments=adjusted_segments,
        algorithm_config=algorithm_config,
    )

    # Compute objective function
    obj_result = compute_objective_function(
        academic_periods=academic_periods,
        algorithm_config=algorithm_config,
        verbose=verbose,
    )

    # Load penalty configuration and compute penalty
    penalty_config = load_penalty_configuration()
    penalty = compute_penalty(academic_periods, penalty_config)

    # Apply penalty to get final score
    final_score = obj_result.final_score - penalty

    # Extract boundary years
    boundary_years = []
    for i, period in enumerate(academic_periods):
        if i == 0:
            boundary_years.append(period.start_year)
        boundary_years.append(period.end_year)
    boundary_years = sorted(set(boundary_years))

    if verbose:
        logger.info(
            f"Manual baseline raw objective score: {obj_result.final_score:.3f}"
        )
        logger.info(f"Manual baseline penalty applied: {penalty:.3f}")
        logger.info(f"Manual baseline final objective score: {final_score:.3f}")

    return BaselineResult(
        baseline_name="Manual",
        objective_score=final_score,
        raw_objective_score=obj_result.final_score,
        penalty=penalty,
        cohesion_score=obj_result.cohesion_score,
        separation_score=obj_result.separation_score,
        num_segments=len(academic_periods),
        boundary_years=boundary_years,
        academic_periods=academic_periods,
    )


def create_fixed_year_baseline(
    domain_name: str,
    algorithm_config: AlgorithmConfig,
    year_interval: int,
    data_directory: str = "resources",
    verbose: bool = False,
) -> BaselineResult:
    """Create baseline using fixed year intervals.

    Args:
        domain_name: Domain name to evaluate
        algorithm_config: Algorithm configuration
        year_interval: Year interval (5 or 10)
        data_directory: Directory containing data files
        verbose: Enable verbose logging

    Returns:
        BaselineResult with fixed year baseline evaluation
    """
    logger = get_logger(__name__, verbose, domain_name)

    # Load domain data with same processing as pipeline
    success, academic_years, error_message = load_domain_data(
        domain_name=domain_name,
        algorithm_config=algorithm_config,
        data_directory=data_directory,
        verbose=verbose,
    )

    if not success:
        raise RuntimeError(f"Failed to load domain data: {error_message}")

    # Get data year range
    data_years = sorted(year.year for year in academic_years)
    min_data_year = min(data_years)
    max_data_year = max(data_years)

    # Create fixed year segments
    segments = []
    current_year = min_data_year

    while current_year <= max_data_year:
        end_year = min(current_year + year_interval - 1, max_data_year)
        segments.append((current_year, end_year))
        current_year = end_year + 1

    if verbose:
        logger.info(
            f"{year_interval}-year baseline: {len(segments)} segments (before filtering)"
        )
        logger.info(f"Segments: {segments}")

    # Filter segments to only include those with academic years
    data_years_set = set(year.year for year in academic_years)
    valid_segments = []

    for start_year, end_year in segments:
        # Check if segment has any academic years with data
        segment_years = [
            year for year in range(start_year, end_year + 1) if year in data_years_set
        ]
        if segment_years:
            valid_segments.append((start_year, end_year))
        elif verbose:
            logger.info(
                f"Skipping segment {start_year}-{end_year}: no academic years with data"
            )

    if not valid_segments:
        raise ValueError(f"No valid segments found for {year_interval}-year baseline")

    if verbose:
        logger.info(
            f"{year_interval}-year baseline: {len(valid_segments)} valid segments (after filtering)"
        )
        logger.info(f"Valid segments: {valid_segments}")

    # Create academic periods from valid segments
    academic_periods = create_academic_periods_from_segments(
        academic_years=tuple(academic_years),
        segments=valid_segments,
        algorithm_config=algorithm_config,
    )

    # Compute objective function
    obj_result = compute_objective_function(
        academic_periods=academic_periods,
        algorithm_config=algorithm_config,
        verbose=verbose,
    )

    # Load penalty configuration and compute penalty
    penalty_config = load_penalty_configuration()
    penalty = compute_penalty(academic_periods, penalty_config)

    # Apply penalty to get final score
    final_score = obj_result.final_score - penalty

    # Extract boundary years
    boundary_years = []
    for i, period in enumerate(academic_periods):
        if i == 0:
            boundary_years.append(period.start_year)
        boundary_years.append(period.end_year)
    boundary_years = sorted(set(boundary_years))

    baseline_name = f"{year_interval}-year"

    if verbose:
        logger.info(
            f"{baseline_name} baseline raw objective score: {obj_result.final_score:.3f}"
        )
        logger.info(f"{baseline_name} baseline penalty applied: {penalty:.3f}")
        logger.info(
            f"{baseline_name} baseline final objective score: {final_score:.3f}"
        )

    return BaselineResult(
        baseline_name=baseline_name,
        objective_score=final_score,
        raw_objective_score=obj_result.final_score,
        penalty=penalty,
        cohesion_score=obj_result.cohesion_score,
        separation_score=obj_result.separation_score,
        num_segments=len(academic_periods),
        boundary_years=boundary_years,
        academic_periods=academic_periods,
    )
