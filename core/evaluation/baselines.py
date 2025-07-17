"""Reference and baseline creation for timeline evaluation.

This module contains functions for loading reference timelines (Gemini, Perplexity)
and creating fixed-year baselines for evaluation with shared data loading and caching.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import hashlib

from ..data.data_models import AcademicYear
from ..data.data_processing import (
    load_domain_data,
    create_academic_periods_from_segments,
)
from ..optimization.objective_function import compute_objective_function
from ..optimization.penalty import create_penalty_config_from_algorithm_config
from ..utils.config import AlgorithmConfig
from ..utils.logging import get_logger
from .evaluation import EvaluationResult


# Global cache for reference and baseline results
_cache: Dict[str, EvaluationResult] = {}


def clear_cache():
    """Clear the global cache."""
    global _cache
    _cache.clear()


def compute_config_hash(algorithm_config: AlgorithmConfig) -> str:
    """Compute hash of algorithm configuration for caching.

    Args:
        algorithm_config: Algorithm configuration to hash

    Returns:
        Configuration hash string
    """
    # Convert config to dictionary and hash relevant fields
    config_dict = {
        "cohesion_weight": algorithm_config.cohesion_weight,
        "separation_weight": algorithm_config.separation_weight,
        "top_k_keywords": algorithm_config.top_k_keywords,
        "min_keyword_frequency_ratio": algorithm_config.min_keyword_frequency_ratio,
        "ubiquity_threshold": algorithm_config.ubiquity_threshold,
        "apply_ubiquitous_filtering": algorithm_config.apply_ubiquitous_filtering,
        "min_papers_per_year": algorithm_config.min_papers_per_year,
        "min_paper_per_segment": algorithm_config.min_paper_per_segment,
    }

    config_str = json.dumps(config_dict, sort_keys=True)
    return hashlib.md5(config_str.encode()).hexdigest()[:8]


def calculate_segment_paper_count(
    start_year: int,
    end_year: int,
    academic_years: List[AcademicYear]
) -> int:
    """Calculate total paper count for a segment.
    
    Args:
        start_year: Segment start year (inclusive)
        end_year: Segment end year (inclusive)
        academic_years: List of academic years data
        
    Returns:
        Total paper count in the segment
    """
    total_papers = 0
    for year in academic_years:
        if start_year <= year.year <= end_year:
            total_papers += year.paper_count
    return total_papers


def load_shared_academic_data(
    domain_name: str,
    algorithm_config: AlgorithmConfig,
    data_directory: str = "resources",
    verbose: bool = False,
) -> Tuple[List[AcademicYear], int, int]:
    """Load academic data once for sharing across evaluations.

    Args:
        domain_name: Domain name to load
        algorithm_config: Algorithm configuration for data processing
        data_directory: Directory containing data files
        verbose: Enable verbose logging

    Returns:
        Tuple of (academic_years, min_data_year, max_data_year)

    Raises:
        RuntimeError: If data loading fails
    """
    logger = get_logger(__name__, verbose, domain_name)

    success, academic_years, error_message = load_domain_data(
        domain_name=domain_name,
        algorithm_config=algorithm_config,
        data_directory=data_directory,
        verbose=verbose,
    )

    if not success:
        raise RuntimeError(f"Failed to load domain data: {error_message}")

    if not academic_years:
        raise RuntimeError(f"No academic years found for {domain_name}")

    # Compute data year range
    data_years = set(year.year for year in academic_years)
    min_data_year = min(data_years)
    max_data_year = max(data_years)

    return academic_years, min_data_year, max_data_year


def create_baseline_from_segments(
    baseline_name: str,
    segments: List[Tuple[int, int]],
    academic_years: List[AcademicYear],
    min_data_year: int,
    max_data_year: int,
    algorithm_config: AlgorithmConfig,
    verbose: bool = False,
) -> EvaluationResult:
    """Create baseline result from segments using pre-loaded data.

    Args:
        baseline_name: Name of the baseline
        segments: List of (start_year, end_year) tuples
        academic_years: Pre-loaded academic years data
        min_data_year: Minimum data year available
        max_data_year: Maximum data year available
        algorithm_config: Algorithm configuration
        verbose: Enable verbose logging

    Returns:
        EvaluationResult with evaluation metrics
    """
    logger = get_logger(__name__, verbose)

    # Filter academic years to match data range
    data_years = set(year.year for year in academic_years)

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
            f"No {baseline_name} segments overlap with data range {min_data_year}-{max_data_year}"
        )

    # Filter segments that don't meet minimum paper count threshold
    valid_segments = []
    skipped_segments = []
    
    for start_year, end_year in adjusted_segments:
        segment_paper_count = calculate_segment_paper_count(
            start_year, end_year, academic_years
        )
        
        if segment_paper_count >= algorithm_config.min_paper_per_segment:
            valid_segments.append((start_year, end_year))
        else:
            skipped_segments.append((start_year, end_year, segment_paper_count))
            if verbose:
                logger.info(
                    f"Skipping {baseline_name} segment {start_year}-{end_year}: "
                    f"{segment_paper_count} papers < {algorithm_config.min_paper_per_segment} threshold"
                )
    
    if not valid_segments:
        if verbose:
            logger.warning(
                f"No {baseline_name} segments meet minimum paper count threshold "
                f"({algorithm_config.min_paper_per_segment})"
            )
        # Return minimal baseline with 0 score
        return EvaluationResult(
            name=baseline_name,
            objective_score=0.0,
            boundary_years=[min_data_year, max_data_year],
            num_segments=0,
            raw_objective_score=0.0,
            penalty=0.0,
            cohesion_score=0.0,
            separation_score=0.0,
            academic_periods=[],
        )

    # Create academic periods from valid segments
    academic_periods = create_academic_periods_from_segments(
        academic_years=tuple(academic_years),
        segments=valid_segments,
        algorithm_config=algorithm_config,
    )

    # Create penalty configuration and compute objective function
    penalty_config = create_penalty_config_from_algorithm_config(algorithm_config)

    obj_result = compute_objective_function(
        academic_periods=academic_periods,
        algorithm_config=algorithm_config,
        penalty_config=penalty_config,
        verbose=verbose,
    )

    # Extract boundary years
    boundary_years = []
    for i, period in enumerate(academic_periods):
        if i == 0:
            boundary_years.append(period.start_year)
        boundary_years.append(period.end_year)
    boundary_years = sorted(set(boundary_years))

    return EvaluationResult(
        name=baseline_name,
        objective_score=obj_result.final_score,
        boundary_years=boundary_years,
        num_segments=len(academic_periods),
        raw_objective_score=obj_result.raw_score,
        penalty=obj_result.penalty,
        cohesion_score=obj_result.cohesion_score,
        separation_score=obj_result.separation_score,
        academic_periods=academic_periods,
    )


def load_gemini_reference(
    domain_name: str,
    academic_years: List[AcademicYear],
    min_data_year: int,
    max_data_year: int,
    algorithm_config: AlgorithmConfig,
    use_cache: bool = True,
    verbose: bool = False,
) -> Optional[EvaluationResult]:
    """Load Gemini reference timeline using pre-loaded data.

    Args:
        domain_name: Domain name
        academic_years: Pre-loaded academic years data
        min_data_year: Minimum data year available
        max_data_year: Maximum data year available
        algorithm_config: Algorithm configuration
        use_cache: Enable result caching
        verbose: Enable verbose logging

    Returns:
        EvaluationResult or None if Gemini file not found
    """
    logger = get_logger(__name__, verbose, domain_name)

    # Check cache first
    if use_cache:
        config_hash = compute_config_hash(algorithm_config)
        cache_key = f"gemini_ref_{domain_name}_{config_hash}"
        if cache_key in _cache:
            return _cache[cache_key]

    # Load Gemini reference file
    gemini_file = Path(f"data/references/{domain_name}_gemini.json")
    if not gemini_file.exists():
        if verbose:
            logger.warning(f"Gemini reference file not found: {gemini_file}")
        return None

    with open(gemini_file, "r") as f:
        gemini_data = json.load(f)

    # Extract year boundaries from Gemini data
    periods = gemini_data["historical_periods"]
    segments = []

    for period in periods:
        start_year = period["start_year"]
        end_year = period["end_year"]
        segments.append((start_year, end_year))

    baseline_result = create_baseline_from_segments(
        baseline_name="Gemini",
        segments=segments,
        academic_years=academic_years,
        min_data_year=min_data_year,
        max_data_year=max_data_year,
        algorithm_config=algorithm_config,
        verbose=verbose,
    )

    # Cache result
    if use_cache:
        _cache[cache_key] = baseline_result

    return baseline_result


def load_perplexity_reference(
    domain_name: str,
    academic_years: List[AcademicYear],
    min_data_year: int,
    max_data_year: int,
    algorithm_config: AlgorithmConfig,
    use_cache: bool = True,
    verbose: bool = False,
) -> Optional[EvaluationResult]:
    """Load Perplexity reference timeline using pre-loaded data.

    Args:
        domain_name: Domain name
        academic_years: Pre-loaded academic years data
        min_data_year: Minimum data year available
        max_data_year: Maximum data year available
        algorithm_config: Algorithm configuration
        use_cache: Enable result caching
        verbose: Enable verbose logging

    Returns:
        EvaluationResult or None if Perplexity file not found
    """
    logger = get_logger(__name__, verbose, domain_name)

    # Check cache first
    if use_cache:
        config_hash = compute_config_hash(algorithm_config)
        cache_key = f"perplexity_ref_{domain_name}_{config_hash}"
        if cache_key in _cache:
            return _cache[cache_key]

    # Load Perplexity reference file
    perplexity_file = Path(f"data/references/{domain_name}_perplexity.json")
    if not perplexity_file.exists():
        if verbose:
            logger.warning(f"Perplexity reference file not found: {perplexity_file}")
        return None

    with open(perplexity_file, "r") as f:
        perplexity_data = json.load(f)

    # Extract year boundaries from Perplexity data
    periods = perplexity_data["historical_periods"]
    segments = []

    for period in periods:
        start_year = period["start_year"]
        end_year = period["end_year"]
        segments.append((start_year, end_year))

    baseline_result = create_baseline_from_segments(
        baseline_name="Perplexity",
        segments=segments,
        academic_years=academic_years,
        min_data_year=min_data_year,
        max_data_year=max_data_year,
        algorithm_config=algorithm_config,
        verbose=verbose,
    )

    # Cache result
    if use_cache:
        _cache[cache_key] = baseline_result

    return baseline_result


def create_fixed_year_baseline(
    domain_name: str,
    year_interval: int,
    academic_years: List[AcademicYear],
    min_data_year: int,
    max_data_year: int,
    algorithm_config: AlgorithmConfig,
    use_cache: bool = True,
    verbose: bool = False,
) -> EvaluationResult:
    """Create fixed year interval baseline using pre-loaded data.

    Args:
        domain_name: Domain name
        year_interval: Year interval (5 or 10)
        academic_years: Pre-loaded academic years data
        min_data_year: Minimum data year available
        max_data_year: Maximum data year available
        algorithm_config: Algorithm configuration
        use_cache: Enable result caching
        verbose: Enable verbose logging

    Returns:
        EvaluationResult with fixed year baseline evaluation
    """
    logger = get_logger(__name__, verbose, domain_name)

    # Check cache first
    if use_cache:
        config_hash = compute_config_hash(algorithm_config)
        cache_key = f"{year_interval}year_{domain_name}_{config_hash}"
        if cache_key in _cache:
            return _cache[cache_key]

    # Create fixed year segments
    segments = []
    current_year = min_data_year

    while current_year <= max_data_year:
        end_year = min(current_year + year_interval - 1, max_data_year)
        segments.append((current_year, end_year))
        current_year = end_year + 1

    baseline_result = create_baseline_from_segments(
        baseline_name=f"{year_interval}-year",
        segments=segments,
        academic_years=academic_years,
        min_data_year=min_data_year,
        max_data_year=max_data_year,
        algorithm_config=algorithm_config,
        verbose=verbose,
    )

    # Cache result
    if use_cache:
        _cache[cache_key] = baseline_result

    return baseline_result
