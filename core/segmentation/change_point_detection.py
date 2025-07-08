"""Dual-metric shift signal detection for research timeline modeling.

This module implements direction change detection using dual-metric formula
and returns simple boundary years for segmentation.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional

from ..data.data_models import AcademicYear
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
    """Detect paradigm shift boundary years using AcademicYear objects.

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

    if precomputed_signals:
        direction_years = precomputed_signals.get("direction_years", [])
        citation_years = precomputed_signals.get("citation_years", [])
        year_confidences = precomputed_signals.get("year_confidences", {})
        if verbose:
            logger.info(
                f"Using precomputed signals: {len(direction_years)} direction, {len(citation_years)} citation"
            )
    else:
        direction_years = []
        year_confidences = {}
        if use_direction:
            if verbose:
                logger.info("  Running direction change detection...")
            direction_years, year_confidences = detect_direction_change_years(
                academic_years, algorithm_config, verbose
            )
            if verbose:
                logger.info(f"  Direction change years: {direction_years}")

        citation_years = []
        if use_citation:
            if verbose:
                logger.info("  Running citation acceleration detection...")
            citation_years = detect_citation_acceleration_years(
                academic_years, domain_name, algorithm_config, verbose
            )
            if verbose:
                logger.info(f"  Citation acceleration years: {citation_years}")

    if verbose:
        logger.info("  Validating and combining signals...")
    boundary_year_ints = validate_and_combine_signals(
        direction_years, citation_years, year_confidences, algorithm_config, verbose
    )

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
) -> Tuple[List[int], Dict[int, float]]:
    """Detect research direction change years using dual-metric vocabulary analysis.

    Args:
        academic_years: List of AcademicYear objects with papers and temporal data
        algorithm_config: Algorithm configuration
        verbose: Enable verbose logging

    Returns:
        Tuple of (years_list, confidence_dict) where confidence_dict maps year to S_dir confidence
    """
    logger = get_logger(__name__, verbose)

    if verbose:
        logger.info(f"    Starting direction change detection...")
        logger.info(f"    Direction threshold: {algorithm_config.direction_threshold}")

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
        return [], {}

    years = sorted(year_keywords_map.keys())
    direction_changes = []
    year_confidences = {}

    if verbose:
        logger.info(f"    Analyzing {len(years)} years for direction changes...")

    similarity_scores = []

    for i in range(len(years) - 1):
        year1, year2 = years[i], years[i + 1]
        keywords_prev = year_keywords_map[year1]
        keywords_curr = year_keywords_map[year2]

        if not keywords_prev or not keywords_curr:
            if verbose:
                logger.warning(
                    f"      {year1}-{year2}: Missing keywords (k_prev={len(keywords_prev)}, k_curr={len(keywords_curr)})"
                )
            continue

        set_prev, set_curr = set(keywords_prev), set(keywords_curr)

        new_keywords = set_curr - set_prev
        shared_keywords = set_curr & set_prev

        novelty = len(new_keywords) / len(set_curr) if len(set_curr) > 0 else 0.0
        overlap = len(shared_keywords) / len(set_prev) if len(set_prev) > 0 else 0.0

        s_dir = novelty * (1 - overlap)
        similarity_scores.append(s_dir)

        year_confidences[year2] = s_dir

        if verbose:
            logger.debug(
                f"      {year1}-{year2}: novelty={novelty:.3f}, overlap={overlap:.3f}, s_dir={s_dir:.3f}"
            )

        if s_dir > algorithm_config.direction_threshold:
            direction_changes.append(year2)
            if verbose:
                logger.info(
                    f"      DIRECTION CHANGE DETECTED at {year2}: s_dir={s_dir:.3f} > threshold={algorithm_config.direction_threshold}"
                )

    if verbose:
        if similarity_scores:
            logger.info(
                f"    Direction scores (s_dir): min={min(similarity_scores):.3f}, max={max(similarity_scores):.3f}, avg={np.mean(similarity_scores):.3f}"
            )
        else:
            logger.warning("    No direction scores calculated!")
        logger.info(f"    Raw direction changes: {direction_changes}")

    all_years = [academic_year.year for academic_year in academic_years]

    clustered_changes = cluster_and_validate_shifts(
        direction_changes, np.array(all_years), min_segment_length=3
    )

    if verbose:
        logger.info(f"    After clustering: {clustered_changes}")

    filtered_confidences = {
        year: year_confidences[year]
        for year in clustered_changes
        if year in year_confidences
    }

    logger.info(f"Detected {len(clustered_changes)} direction change years")
    return clustered_changes, filtered_confidences


def detect_citation_acceleration_years(
    academic_years: List[AcademicYear],
    domain_name: str,
    algorithm_config,
    verbose: bool = False,
) -> List[int]:
    """Detect citation acceleration years using gradient analysis.

    Args:
        academic_years: List of AcademicYear objects with papers and temporal data
        domain_name: Name of the domain for context
        algorithm_config: Algorithm configuration
        verbose: Enable verbose logging

    Returns:
        List of years where citation acceleration occurs
    """
    logger = get_logger(__name__, verbose)

    year_citations = {}
    for academic_year in academic_years:
        year = academic_year.year
        year_citations[year] = academic_year.total_citations

    if len(year_citations) < 5:
        logger.warning("Insufficient citation data for acceleration detection")
        return []

    years = sorted(year_citations.keys())
    citations = np.array([year_citations[year] for year in years])

    acceleration_years = []
    scales = algorithm_config.citation_analysis_scales

    for window in scales:
        if len(citations) <= window:
            continue

        if window > 1:
            smoothed = moving_average(citations, window)
            smoothed = np.pad(smoothed, (window // 2, window // 2), mode="edge")
            smoothed = smoothed[: len(citations)]
        else:
            smoothed = citations

        gradient = np.gradient(smoothed)
        acceleration = np.gradient(gradient)

        grad_threshold = citation_adaptive_threshold(gradient, "gradient")
        accel_threshold = citation_adaptive_threshold(acceleration, "acceleration")

        significant_grads = np.where(np.abs(gradient) > grad_threshold)[0]
        significant_accels = np.where(np.abs(acceleration) > accel_threshold)[0]

        for idx in significant_grads:
            if idx < len(years):
                acceleration_years.append(years[idx])

        for idx in significant_accels:
            if idx < len(years):
                acceleration_years.append(years[idx])

    acceleration_years = list(set(acceleration_years))

    all_years = [academic_year.year for academic_year in academic_years]

    clustered_years = cluster_and_validate_shifts(
        acceleration_years, np.array(all_years), min_segment_length=3
    )

    logger.info(f"Detected {len(clustered_years)} citation acceleration years")
    return clustered_years


def validate_and_combine_signals(
    direction_years: List[int],
    citation_years: List[int],
    year_confidences: Dict[int, float],
    algorithm_config,
    verbose: bool = False,
) -> List[int]:
    """Validate and combine direction and citation signals with confidence boost.

    Args:
        direction_years: Years with direction changes
        citation_years: Years with citation acceleration
        year_confidences: Confidence scores for direction years
        algorithm_config: Algorithm configuration
        verbose: Enable verbose logging

    Returns:
        List of validated boundary years that meet the confidence threshold
    """
    logger = get_logger(__name__, verbose)

    if not direction_years:
        logger.warning("No direction change years found")
        return []

    if verbose:
        logger.info(
            f"  Validating {len(direction_years)} direction years against threshold {algorithm_config.validation_threshold}"
        )
        logger.info(f"  Citation boost rate: {algorithm_config.citation_boost_rate}")
        logger.info(
            f"  Citation support window: ±{algorithm_config.citation_support_window} years"
        )

    validated_years = []
    support_window = algorithm_config.citation_support_window
    beta = algorithm_config.citation_boost_rate
    validation_threshold = algorithm_config.validation_threshold

    for dir_year in direction_years:
        original_confidence = year_confidences.get(dir_year, 0.0)

        has_citation_support = False
        if citation_years:
            for cit_year in citation_years:
                if abs(cit_year - dir_year) <= support_window:
                    has_citation_support = True
                    break

        if has_citation_support:
            boosted_confidence = original_confidence + beta * validation_threshold
            final_confidence = min(boosted_confidence, 1.0)

            if verbose:
                logger.info(
                    f"    Year {dir_year}: original={original_confidence:.3f} + boost({beta}×{validation_threshold:.3f}) = {final_confidence:.3f} [CITATION SUPPORT]"
                )
        else:
            final_confidence = original_confidence

            if verbose:
                logger.info(
                    f"    Year {dir_year}: confidence={final_confidence:.3f} [NO CITATION SUPPORT]"
                )

        if final_confidence >= validation_threshold:
            validated_years.append(dir_year)
            if verbose:
                logger.info(
                    f"    ✓ VALIDATED: {dir_year} (confidence {final_confidence:.3f} ≥ {validation_threshold:.3f})"
                )
        else:
            if verbose:
                logger.info(
                    f"    ✗ REJECTED: {dir_year} (confidence {final_confidence:.3f} < {validation_threshold:.3f})"
                )

    if verbose:
        logger.info(
            f"  Validation results: {len(validated_years)}/{len(direction_years)} years passed threshold"
        )

    logger.info(
        f"Validated {len(validated_years)} boundary years (threshold: {validation_threshold:.3f})"
    )
    return sorted(validated_years)


def citation_adaptive_threshold(data: np.ndarray, method: str) -> float:
    """Calculate adaptive thresholds based on data characteristics.

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
        return data_std * 1.5
    elif method == "acceleration":
        mad = np.median(np.abs(data - np.median(data)))
        return mad * 2.0
    else:
        return data_std


def moving_average(data: np.ndarray, window: int) -> np.ndarray:
    """Calculate moving average.

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
    """Cluster nearby shifts and validate temporal spacing.

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

    filtered_shifts = [shifts[0]]

    for shift in shifts[1:]:
        if shift - filtered_shifts[-1] >= min_segment_length:
            filtered_shifts.append(shift)

    min_year = min(years_array)
    max_year = max(years_array)

    valid_shifts = [s for s in filtered_shifts if min_year <= s <= max_year]

    return valid_shifts
