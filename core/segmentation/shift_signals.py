"""
Simplified Shift Signal Detection for Research Timeline Modeling

This module returns simple boundary years instead of complex objects.
This eliminates unnecessary complexity while preserving the core algorithmic functionality.

Key functionality:
- Research direction change detection → boundary years
- Citation gradient analysis for validation → boundary years
- Temporal clustering with configurable granularity control → boundary years
- Signal validation through citation support → boundary years

Returns: List[int] - Simple boundary years for segmentation
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import defaultdict, Counter
from scipy import stats
from scipy.stats import zscore
from scipy.signal import find_peaks

from ..data.data_models import Paper, AcademicYear
from ..utils.logging import get_logger


def detect_boundary_years(
    academic_years: List[AcademicYear],
    domain_name: str,
    algorithm_config,
    use_citation: bool = True,
    use_direction: bool = True,
    precomputed_signals: Optional[Dict[str, List]] = None,
    verbose: bool = False,
) -> List[AcademicYear]:
    """
    Detect paradigm shift boundary years using AcademicYear objects.

    This is the main entry point for shift detection in the simplified architecture.

    Args:
        academic_years: List of AcademicYear objects with papers and temporal data
        domain_name: Name of the domain for context
        algorithm_config: Algorithm configuration
        use_citation: Whether to use citation validation
        use_direction: Whether to use direction change detection
        precomputed_signals: Optional precomputed signals for efficiency
        verbose: Enable verbose logging

    Returns:
        List of AcademicYear objects where paradigm shifts occur
    """
    logger = get_logger(__name__, verbose)

    if verbose:
        logger.info(f"=== SHIFT DETECTION STARTED ===")
        logger.info(f"  Domain: {domain_name}")
        logger.info(f"  Academic years: {len(academic_years)}")
        logger.info(
            f"  Year range: {min(ay.year for ay in academic_years)}-{max(ay.year for ay in academic_years)}"
        )
        logger.info(f"  Use direction detection: {use_direction}")
        logger.info(f"  Use citation detection: {use_citation}")

    # Use precomputed signals if available
    if precomputed_signals:
        direction_years = precomputed_signals.get("direction_years", [])
        citation_years = precomputed_signals.get("citation_years", [])
        if verbose:
            logger.info(
                f"Using precomputed signals: {len(direction_years)} direction, {len(citation_years)} citation"
            )
    else:
        # Detect direction change years
        direction_years = []
        if use_direction:
            if verbose:
                logger.info("  Running direction change detection...")
            direction_years = detect_direction_change_years(
                academic_years, algorithm_config, verbose
            )
            if verbose:
                logger.info(f"  Direction change years: {direction_years}")

        # Detect citation acceleration years
        citation_years = []
        if use_citation:
            if verbose:
                logger.info("  Running citation acceleration detection...")
            citation_years = detect_citation_acceleration_years(
                academic_years, domain_name, algorithm_config, verbose
            )
            if verbose:
                logger.info(f"  Citation acceleration years: {citation_years}")

    # Validate and combine signals
    if verbose:
        logger.info("  Validating and combining signals...")
    boundary_year_ints = validate_and_combine_signals(
        direction_years, citation_years, algorithm_config, verbose
    )

    # Convert boundary year integers to AcademicYear objects
    year_lookup = {
        academic_year.year: academic_year for academic_year in academic_years
    }
    boundary_academic_years = []

    for year in boundary_year_ints:
        if year in year_lookup:
            boundary_academic_years.append(year_lookup[year])
        else:
            logger.warning(f"Boundary year {year} not found in academic years")

    logger.info(
        f"Detected {len(boundary_academic_years)} boundary years: {[ay.year for ay in boundary_academic_years]}"
    )
    return boundary_academic_years


def detect_direction_change_years(
    academic_years: List[AcademicYear], algorithm_config, verbose: bool = False
) -> List[int]:
    """
    Detect research direction change years using vocabulary analysis with AcademicYear objects.

    Uses pre-computed top_keywords from AcademicYear objects directly, eliminating
    redundant keyword extraction and filtering.

    Args:
        academic_years: List of AcademicYear objects with papers and temporal data
        algorithm_config: Algorithm configuration
        verbose: Enable verbose logging

    Returns:
        List of years where significant direction changes occur
    """
    logger = get_logger(__name__, verbose)

    if verbose:
        logger.info(f"    Starting direction change detection...")
        logger.info(f"    Direction threshold: {algorithm_config.direction_threshold}")

    # Use pre-computed top_keywords directly from AcademicYear objects
    year_keywords_map = {}
    for academic_year in academic_years:
        year_keywords_map[academic_year.year] = list(academic_year.top_keywords)

    if verbose:
        logger.info(f"    Year-to-keywords mapping: {len(year_keywords_map)} years")
        if year_keywords_map:
            sample_years = sorted(year_keywords_map.keys())[:3]
            for year in sample_years:
                keywords = year_keywords_map[year]
                logger.info(
                    f"      {year}: {len(keywords)} keywords - {keywords[:5]}..."
                )

    if len(year_keywords_map) < 2:
        logger.warning("Insufficient years for direction change detection")
        return []

    # Calculate vocabulary similarity between consecutive years
    years = sorted(year_keywords_map.keys())
    direction_changes = []

    if verbose:
        logger.info(f"    Analyzing {len(years)} years for direction changes...")

    similarity_scores = []

    for i in range(len(years) - 1):
        year1, year2 = years[i], years[i + 1]
        keywords1 = year_keywords_map[year1]
        keywords2 = year_keywords_map[year2]

        # Calculate Jaccard similarity
        if not keywords1 or not keywords2:
            if verbose:
                logger.warning(
                    f"      {year1}-{year2}: Missing keywords (k1={len(keywords1)}, k2={len(keywords2)})"
                )
            continue

        set1, set2 = set(keywords1), set(keywords2)
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        similarity = intersection / union if union > 0 else 0.0
        similarity_scores.append(similarity)

        if verbose:
            logger.debug(
                f"      {year1}-{year2}: similarity={similarity:.3f}, intersection={intersection}, union={union}"
            )

        # Check for significant direction change
        if similarity < algorithm_config.direction_threshold:
            direction_changes.append(year2)
            if verbose:
                logger.info(
                    f"      DIRECTION CHANGE DETECTED at {year2}: similarity={similarity:.3f} < threshold={algorithm_config.direction_threshold}"
                )

    if verbose:
        if similarity_scores:
            logger.info(
                f"    Similarity scores: min={min(similarity_scores):.3f}, max={max(similarity_scores):.3f}, avg={np.mean(similarity_scores):.3f}"
            )
        else:
            logger.warning("    No similarity scores calculated!")
        logger.info(f"    Raw direction changes: {direction_changes}")

    # Get years array from academic_years for clustering
    all_years = [academic_year.year for academic_year in academic_years]

    # Cluster nearby changes
    clustered_changes = cluster_and_validate_shifts(
        direction_changes, np.array(all_years), min_segment_length=3
    )

    if verbose:
        logger.info(f"    After clustering: {clustered_changes}")

    logger.info(f"Detected {len(clustered_changes)} direction change years")
    return clustered_changes


def detect_citation_acceleration_years(
    academic_years: List[AcademicYear],
    domain_name: str,
    algorithm_config,
    verbose: bool = False,
) -> List[int]:
    """
    Detect citation acceleration years using gradient analysis with AcademicYear objects.

    Args:
        academic_years: List of AcademicYear objects with papers and temporal data
        domain_name: Name of the domain for context
        algorithm_config: Algorithm configuration
        verbose: Enable verbose logging

    Returns:
        List of years where citation acceleration occurs
    """
    logger = get_logger(__name__, verbose)

    # Get citation time series from AcademicYear objects
    year_citations = {}
    for academic_year in academic_years:
        year = academic_year.year
        # Use the pre-computed total_citations from AcademicYear
        year_citations[year] = academic_year.total_citations

    if len(year_citations) < 5:
        logger.warning("Insufficient citation data for acceleration detection")
        return []

    # Convert to arrays for analysis
    years = sorted(year_citations.keys())
    citations = np.array([year_citations[year] for year in years])

    # Multi-scale gradient analysis
    acceleration_years = []
    scales = algorithm_config.citation_analysis_scales

    for window in scales:
        if len(citations) <= window:
            continue

        # Smooth the series
        if window > 1:
            smoothed = moving_average(citations, window)
            # Pad to maintain length
            smoothed = np.pad(smoothed, (window // 2, window // 2), mode="edge")
            smoothed = smoothed[: len(citations)]
        else:
            smoothed = citations

        # Calculate gradient and acceleration
        gradient = np.gradient(smoothed)
        acceleration = np.gradient(gradient)

        # Find significant changes
        grad_threshold = citation_adaptive_threshold(gradient, "gradient")
        accel_threshold = citation_adaptive_threshold(acceleration, "acceleration")

        # Identify significant years
        significant_grads = np.where(np.abs(gradient) > grad_threshold)[0]
        significant_accels = np.where(np.abs(acceleration) > accel_threshold)[0]

        # Add years to acceleration list
        for idx in significant_grads:
            if idx < len(years):
                acceleration_years.append(years[idx])

        for idx in significant_accels:
            if idx < len(years):
                acceleration_years.append(years[idx])

    # Remove duplicates and cluster
    acceleration_years = list(set(acceleration_years))

    # Get years array from academic_years for clustering
    all_years = [academic_year.year for academic_year in academic_years]

    clustered_years = cluster_and_validate_shifts(
        acceleration_years, np.array(all_years), min_segment_length=3
    )

    logger.info(f"Detected {len(clustered_years)} citation acceleration years")
    return clustered_years


def validate_and_combine_signals(
    direction_years: List[int],
    citation_years: List[int],
    algorithm_config,
    verbose: bool = False,
) -> List[int]:
    """
    Validate and combine direction and citation signals.

    Args:
        direction_years: Years with direction changes
        citation_years: Years with citation acceleration
        algorithm_config: Algorithm configuration
        verbose: Enable verbose logging

    Returns:
        List of validated boundary years
    """
    logger = get_logger(__name__, verbose)

    if not direction_years:
        logger.warning("No direction change years found")
        return []

    # If no citation validation requested, return direction years
    if not citation_years:
        logger.info("No citation validation - using direction years only")
        return sorted(direction_years)

    # Validate direction changes with citation support
    validated_years = []
    support_window = algorithm_config.citation_support_window

    for dir_year in direction_years:
        # Check for citation support within window
        has_support = False
        for cit_year in citation_years:
            if abs(cit_year - dir_year) <= support_window:
                has_support = True
                break

        if has_support:
            validated_years.append(dir_year)
            logger.debug(
                f"Validated direction change at {dir_year} with citation support"
            )
        else:
            logger.debug(f"No citation support for direction change at {dir_year}")

    # Apply validation threshold
    if (
        len(validated_years)
        < len(direction_years) * algorithm_config.validation_threshold
    ):
        logger.warning(
            f"Low validation rate: {len(validated_years)}/{len(direction_years)}"
        )

    return sorted(validated_years)


# =============================================================================
# UTILITY FUNCTIONS - Pure helper functions
# =============================================================================


def citation_adaptive_threshold(data: np.ndarray, method: str) -> float:
    """
    Calculate adaptive thresholds based on data characteristics (pure function).

    Args:
        data: Input data array
        method: Threshold method ('gradient' or 'acceleration')

    Returns:
        Adaptive threshold value
    """
    if len(data) == 0:
        return 0.0

    data_std = np.std(data)

    if method == "gradient":
        # For gradient: threshold based on standard deviation
        return data_std * 1.5
    elif method == "acceleration":
        # For acceleration: threshold based on median absolute deviation
        mad = np.median(np.abs(data - np.median(data)))
        return mad * 2.0
    else:
        return data_std


def moving_average(data: np.ndarray, window: int) -> np.ndarray:
    """
    Pure function for calculating moving average.

    Args:
        data: Input data array
        window: Window size for averaging

    Returns:
        Moving average array
    """
    if window >= len(data):
        return np.array([np.mean(data)] * len(data))

    result = []
    for i in range(len(data) - window + 1):
        result.append(np.mean(data[i : i + window]))
    return np.array(result)


def cluster_and_validate_shifts(
    shifts: List[int], years_array: np.ndarray, min_segment_length: int = 3
) -> List[int]:
    """
    Pure function to cluster nearby shifts and validate temporal spacing.

    Args:
        shifts: List of detected shift years
        years_array: Array of all years
        min_segment_length: Minimum years between shifts

    Returns:
        Filtered and validated shifts
    """
    if not shifts:
        return []

    shifts = sorted(list(set(shifts)))

    # Remove shifts too close together
    filtered_shifts = [shifts[0]]

    for shift in shifts[1:]:
        if shift - filtered_shifts[-1] >= min_segment_length:
            filtered_shifts.append(shift)

    # Validate shifts are within years range
    min_year = min(years_array)
    max_year = max(years_array)

    valid_shifts = [s for s in filtered_shifts if min_year <= s <= max_year]

    return valid_shifts
