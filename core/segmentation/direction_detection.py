"""Direction change detection for research timeline modeling.

This module implements direction change detection using frequency-weighted scoring
and adaptive thresholding to identify paradigm shifts in academic domains.
"""

import numpy as np
from typing import List, Dict

from ..data.data_models import AcademicYear
from ..utils.logging import get_logger


# =============================================================================
# MAIN DETECTION FUNCTIONS
# =============================================================================


def detect_direction_change_years_with_citation_boost(
    academic_years: List[AcademicYear],
    citation_years: List[int],
    algorithm_config,
    verbose: bool = False,
) -> List[int]:
    """Detect direction change years with immediate citation boost integration.

    This streamlined approach:
    1. Computes direction scores for all years
    2. Immediately boosts scores if citation support is nearby
    3. Applies single threshold to final boosted scores

    Args:
        academic_years: List of AcademicYear objects with papers and temporal data
        citation_years: List of years with citation acceleration
        algorithm_config: Algorithm configuration
        verbose: Enable verbose logging

    Returns:
        List of boundary years that exceed the final threshold
    """
    logger = get_logger(__name__, verbose)

    if verbose:
        scoring_method = getattr(
            algorithm_config, "direction_scoring_method", "weighted_jaccard"
        )
        logger.info("    Starting streamlined direction change detection...")
        logger.info(f"    Using scoring method: {scoring_method}")

    # Filter years with insufficient publication volume
    eligible_years = _filter_eligible_years(academic_years, algorithm_config)
    if len(eligible_years) < 3:
        logger.warning("Insufficient years for direction change detection")
        return []

    if verbose:
        logger.info(f"    Analyzing {len(eligible_years)} eligible years...")

    # Calculate threshold (adaptive or fixed)
    threshold = _calculate_detection_threshold(
        eligible_years, algorithm_config, verbose
    )

    # Run main detection algorithm
    boundaries = _run_cumulative_detection_algorithm(
        eligible_years, citation_years, algorithm_config, threshold, verbose
    )

    logger.info(f"Detected {len(boundaries)} streamlined boundary years")
    return boundaries


# =============================================================================
# THRESHOLD CALCULATION FUNCTIONS
# =============================================================================


def _calculate_detection_threshold(
    eligible_years: List[AcademicYear], algorithm_config, verbose: bool
) -> float:
    """Calculate detection threshold (adaptive or fixed)."""
    threshold_strategy = getattr(
        algorithm_config, "direction_threshold_strategy", "global_p90"
    )

    if threshold_strategy == "fixed":
        threshold = algorithm_config.direction_change_threshold
        if verbose:
            logger = get_logger(__name__, verbose)
            logger.info(f"    Using fixed threshold: {threshold:.3f}")
    else:
        threshold = _calculate_adaptive_threshold(
            eligible_years, algorithm_config, verbose
        )
        if verbose:
            logger = get_logger(__name__, verbose)
            logger.info(
                f"    Using adaptive threshold ({threshold_strategy}): {threshold:.3f}"
            )

    return threshold


def _calculate_adaptive_threshold(
    eligible_years: List[AcademicYear],
    algorithm_config,
    verbose: bool = False,
) -> float:
    """Calculate adaptive threshold using cumulative scoring simulation.

    This function performs steps 1 and 2 of the adaptive threshold calculation:
    1. Collects BASE scores using cumulative simulation
    2. Calculates adaptive threshold from BASE sample scores

    Args:
        eligible_years: List of eligible AcademicYear objects
        algorithm_config: Algorithm configuration
        verbose: Enable verbose logging

    Returns:
        Calculated adaptive threshold value
    """
    logger = get_logger(__name__, verbose)

    # Step 1: Collect BASE scores using cumulative simulation
    base_sample_scores = _collect_base_scores_for_threshold(
        eligible_years, algorithm_config, verbose
    )

    # Step 2: Calculate adaptive threshold from BASE sample scores
    if not base_sample_scores:
        logger.warning("No base scores collected, using fixed threshold")
        return algorithm_config.direction_change_threshold

    threshold_strategy = getattr(
        algorithm_config, "direction_threshold_strategy", "global_p90"
    )
    threshold = calculate_threshold_from_scores(
        base_sample_scores,
        threshold_strategy,
        algorithm_config.direction_change_threshold,
    )

    if verbose:
        logger.info("    Step 2: Calculated adaptive threshold from BASE scores")
        if base_sample_scores:
            logger.info(
                f"    Base sample score range: {min(base_sample_scores):.3f}-{max(base_sample_scores):.3f}"
            )
        else:
            logger.info("    No base sample scores collected")

    return threshold


def _collect_base_scores_for_threshold(
    eligible_years: List[AcademicYear], algorithm_config, verbose: bool
) -> List[float]:
    """Collect BASE scores using cumulative simulation for threshold calculation."""
    logger = get_logger(__name__, verbose)

    min_baseline_years = getattr(algorithm_config, "min_baseline_period_years", 3)
    sampling_interval = getattr(algorithm_config, "score_distribution_window_years", 3)

    if verbose:
        logger.info(
            f"    Step 1: Collecting BASE scores using cumulative simulation (every {sampling_interval} years)..."
        )

    base_sample_scores = []
    scoring_method = getattr(
        algorithm_config, "direction_scoring_method", "weighted_jaccard"
    )

    # Simulate cumulative scoring at regular intervals
    last_boundary_idx = 0
    for current_idx in range(
        min_baseline_years, len(eligible_years), sampling_interval
    ):
        current_year = eligible_years[current_idx]

        # Ensure minimum baseline period
        if current_idx - last_boundary_idx < min_baseline_years:
            continue

        # Calculate score for this simulation point
        baseline_frequencies = build_cumulative_baseline(
            eligible_years, last_boundary_idx, current_idx
        )
        current_frequencies = normalize_keyword_frequencies(
            current_year.keyword_frequencies
        )
        base_score = calculate_direction_score(
            current_frequencies, baseline_frequencies, scoring_method
        )

        base_sample_scores.append(base_score)

    return base_sample_scores


def calculate_threshold_from_scores(
    all_scores: List[float], strategy: str = "global_p90", fixed_threshold: float = 0.1
) -> float:
    """Calculate threshold based on score distribution and strategy.

    Args:
        all_scores: List of all direction scores
        strategy: Threshold strategy ("fixed", "global_p90", "global_p95", "global_p99")
        fixed_threshold: Fixed threshold value to use when strategy is "fixed"

    Returns:
        Calculated threshold value
    """
    if strategy == "fixed":
        return fixed_threshold
    elif strategy == "global_p90":
        return np.percentile(all_scores, 90) if all_scores else fixed_threshold
    elif strategy == "global_p95":
        return np.percentile(all_scores, 95) if all_scores else fixed_threshold
    elif strategy == "global_p99":
        return np.percentile(all_scores, 99) if all_scores else fixed_threshold
    else:
        return fixed_threshold


# =============================================================================
# MAIN DETECTION ALGORITHM
# =============================================================================


def _run_cumulative_detection_algorithm(
    eligible_years: List[AcademicYear],
    citation_years: List[int],
    algorithm_config,
    threshold: float,
    verbose: bool,
) -> List[int]:
    """Run the main cumulative detection algorithm with citation boost."""
    logger = get_logger(__name__, verbose)

    if verbose:
        logger.info(
            "    Step 3: Running cumulative algorithm with proper boundary updates..."
        )

    # Get algorithm parameters
    scoring_method = getattr(
        algorithm_config, "direction_scoring_method", "weighted_jaccard"
    )
    min_baseline_years = getattr(algorithm_config, "min_baseline_period_years", 3)
    support_window = algorithm_config.citation_support_window_years
    boost_factor = algorithm_config.citation_confidence_boost

    # Detection state
    boundaries = []
    final_year_diagnostics = {}
    last_boundary_idx = 0

    # Process each year
    for current_idx in range(1, len(eligible_years)):
        current_year = eligible_years[current_idx]

        # Skip if insufficient baseline period
        if current_idx - last_boundary_idx < min_baseline_years:
            continue

        # Calculate direction score
        direction_score_result = _calculate_year_direction_score(
            eligible_years, current_idx, last_boundary_idx, scoring_method
        )

        # Apply citation boost if supported
        final_score = _apply_citation_boost(
            direction_score_result["base_score"],
            current_year.year,
            citation_years,
            support_window,
            boost_factor,
        )

        # Create diagnostics
        diagnostics = _create_year_diagnostics(
            current_year,
            direction_score_result,
            final_score,
            citation_years,
            support_window,
            algorithm_config,
            eligible_years,
            current_idx,
            last_boundary_idx,
        )
        final_year_diagnostics[current_year.year] = diagnostics

        # Check if boundary detected
        if final_score > threshold:
            boundaries.append(current_year.year)
            last_boundary_idx = current_idx
            diagnostics["threshold_exceeded"] = True

            if verbose:
                _log_boundary_detection(
                    logger,
                    current_year.year,
                    direction_score_result["base_score"],
                    final_score,
                    threshold,
                    boost_factor,
                    diagnostics,
                )
        else:
            if verbose:
                _log_no_boundary(
                    logger,
                    current_year.year,
                    direction_score_result["base_score"],
                    final_score,
                    threshold,
                    boost_factor,
                    diagnostics,
                )

    # Log final results and save diagnostics
    _finalize_detection_results(
        algorithm_config,
        final_year_diagnostics,
        boundaries,
        citation_years,
        support_window,
        boost_factor,
        eligible_years,
        verbose,
    )

    return boundaries


def _calculate_year_direction_score(
    eligible_years: List[AcademicYear],
    current_idx: int,
    last_boundary_idx: int,
    scoring_method: str,
) -> Dict:
    """Calculate direction score for a specific year."""
    current_year = eligible_years[current_idx]

    # Build baseline from last boundary to current year
    baseline_frequencies = build_cumulative_baseline(
        eligible_years, last_boundary_idx, current_idx
    )

    # Get current year frequencies
    current_frequencies = normalize_keyword_frequencies(
        current_year.keyword_frequencies
    )

    # Calculate base direction score
    base_score = calculate_direction_score(
        current_frequencies, baseline_frequencies, scoring_method
    )

    return {
        "base_score": base_score,
        "baseline_frequencies": baseline_frequencies,
        "current_frequencies": current_frequencies,
    }


def _apply_citation_boost(
    base_score: float,
    current_year: int,
    citation_years: List[int],
    support_window: int,
    boost_factor: float,
) -> float:
    """Apply citation boost if citation support is nearby."""
    has_citation_support = any(
        abs(cit_year - current_year) <= support_window for cit_year in citation_years
    )

    if has_citation_support:
        return min(base_score + boost_factor, 1.0)  # Cap at 1.0
    else:
        return base_score


# =============================================================================
# SCORING FUNCTIONS
# =============================================================================


def calculate_direction_score(
    current_frequencies: Dict[str, float],
    baseline_frequencies: Dict[str, float],
    method: str = "weighted_jaccard",
) -> float:
    """Calculate direction score using specified method.

    Args:
        current_frequencies: Normalized frequencies for current year
        baseline_frequencies: Normalized frequencies for baseline period
        method: Scoring method ("weighted_jaccard", "jensen_shannon")

    Returns:
        Direction score based on chosen method
    """
    if method == "weighted_jaccard":
        return _calculate_weighted_jaccard_score(
            current_frequencies, baseline_frequencies
        )
    elif method == "jensen_shannon":
        return _calculate_jensen_shannon_divergence(
            current_frequencies, baseline_frequencies
        )
    else:
        raise ValueError(f"Unknown scoring method: {method}")


def _calculate_weighted_jaccard_score(
    current_frequencies: Dict[str, float],
    baseline_frequencies: Dict[str, float],
) -> float:
    """Calculate frequency-weighted direction score using weighted Jaccard similarity.

    Args:
        current_frequencies: Normalized frequencies for current year
        baseline_frequencies: Normalized frequencies for baseline period

    Returns:
        Direction score (1 - weighted_jaccard), range [0, 1]
    """
    if not current_frequencies or not baseline_frequencies:
        return 0.0

    all_keywords = set(current_frequencies.keys()) | set(baseline_frequencies.keys())

    intersection = sum(
        min(current_frequencies.get(k, 0), baseline_frequencies.get(k, 0))
        for k in all_keywords
    )

    union = sum(
        max(current_frequencies.get(k, 0), baseline_frequencies.get(k, 0))
        for k in all_keywords
    )

    if union == 0:
        return 0.0

    weighted_jaccard = intersection / union
    return 1.0 - weighted_jaccard  # Convert to direction change score


def _calculate_jensen_shannon_divergence(
    current_frequencies: Dict[str, float],
    baseline_frequencies: Dict[str, float],
) -> float:
    """Calculate Jensen-Shannon Divergence between probability distributions.

    JSD is a symmetric, bounded [0,1] distance metric that measures information
    divergence between probability distributions. It's theoretically superior
    to weighted Jaccard for comparing research vocabulary distributions.

    Args:
        current_frequencies: Normalized frequencies for current year
        baseline_frequencies: Normalized frequencies for baseline period

    Returns:
        Jensen-Shannon Divergence value, range [0, 1]
    """
    if not current_frequencies or not baseline_frequencies:
        return 0.0

    # Get all keywords and create aligned probability vectors
    all_keywords = sorted(
        set(current_frequencies.keys()) | set(baseline_frequencies.keys())
    )

    # Small epsilon to avoid log(0) issues
    epsilon = 1e-10

    # Create probability vectors
    p = np.array([current_frequencies.get(k, 0) + epsilon for k in all_keywords])
    q = np.array([baseline_frequencies.get(k, 0) + epsilon for k in all_keywords])

    # Normalize to ensure they sum to 1
    p = p / np.sum(p)
    q = q / np.sum(q)

    # Compute average distribution M = 0.5 * (P + Q)
    m = 0.5 * (p + q)

    # Compute JSD = 0.5 * KL(P || M) + 0.5 * KL(Q || M)
    from scipy.stats import entropy

    kl_pm = entropy(p, m)
    kl_qm = entropy(q, m)

    jsd = 0.5 * kl_pm + 0.5 * kl_qm

    # Convert from nats to bits and normalize to [0,1]
    jsd_normalized = jsd / np.log(2)

    return min(jsd_normalized, 1.0)  # Ensure bounded [0,1]


# =============================================================================
# BASELINE AND FREQUENCY FUNCTIONS
# =============================================================================


def build_cumulative_baseline(
    academic_years: List[AcademicYear],
    last_boundary_idx: int,
    current_idx: int,
) -> Dict[str, float]:
    """Build frequency-weighted keyword baseline from last boundary to current year.

    Args:
        academic_years: List of AcademicYear objects
        last_boundary_idx: Index of the last detected boundary year
        current_idx: Index of the current year being analyzed

    Returns:
        Dictionary mapping keywords to normalized frequencies in baseline period
    """
    baseline_frequencies = {}
    total_papers = 0

    # Aggregate from last boundary to current
    for year_idx in range(last_boundary_idx, current_idx):
        if year_idx >= len(academic_years):
            continue

        year = academic_years[year_idx]
        total_papers += year.paper_count

        # Weight by paper count for more accurate representation
        for keyword, freq in year.keyword_frequencies.items():
            baseline_frequencies[keyword] = baseline_frequencies.get(keyword, 0) + freq

    # Normalize by total papers in baseline period
    if total_papers > 0:
        for keyword in baseline_frequencies:
            baseline_frequencies[keyword] = baseline_frequencies[keyword] / total_papers

    return baseline_frequencies


def normalize_keyword_frequencies(
    keyword_frequencies: Dict[str, int],
) -> Dict[str, float]:
    """Normalize keyword frequencies to create probability distribution.

    Args:
        keyword_frequencies: Dictionary mapping keywords to raw frequencies

    Returns:
        Dictionary mapping keywords to normalized frequencies (sum to 1.0)
    """
    total_freq = sum(keyword_frequencies.values())

    if total_freq == 0:
        return {}

    return {keyword: freq / total_freq for keyword, freq in keyword_frequencies.items()}


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================


def _filter_eligible_years(
    academic_years: List[AcademicYear], algorithm_config
) -> List[AcademicYear]:
    """Filter years with insufficient publication volume."""
    min_papers_threshold = getattr(algorithm_config, "min_papers_per_year", 100)
    return [ay for ay in academic_years if ay.paper_count >= min_papers_threshold]


def _create_year_diagnostics(
    current_year: AcademicYear,
    direction_score_result: Dict,
    final_score: float,
    citation_years: List[int],
    support_window: int,
    algorithm_config,
    eligible_years: List[AcademicYear],
    current_idx: int,
    last_boundary_idx: int,
) -> Dict:
    """Create diagnostic information for a year."""
    baseline_frequencies = direction_score_result["baseline_frequencies"]
    current_frequencies = direction_score_result["current_frequencies"]

    # Calculate keyword statistics
    baseline_keywords = set(baseline_frequencies.keys())
    current_keywords = set(current_frequencies.keys())
    new_keywords = current_keywords - baseline_keywords
    shared_keywords = current_keywords & baseline_keywords

    # Create baseline period string
    baseline_period_str = (
        f"{eligible_years[last_boundary_idx + 1].year}-{eligible_years[current_idx - 1].year}"
        if current_idx > last_boundary_idx + 1
        else "empty"
    )

    # Check for citation support
    has_citation_support = any(
        abs(cit_year - current_year.year) <= support_window
        for cit_year in citation_years
    )

    return {
        "baseline_period": baseline_period_str,
        "base_score": direction_score_result["base_score"],
        "final_score": final_score,
        "citation_support": has_citation_support,
        "baseline_keywords_count": len(baseline_keywords),
        "current_keywords_count": len(current_keywords),
        "new_keywords_count": len(new_keywords),
        "shared_keywords_count": len(shared_keywords),
        "top_10_new_keywords": sorted(list(new_keywords))[
            : algorithm_config.diagnostic_top_keywords_limit
        ],
        "top_10_shared_keywords": sorted(list(shared_keywords))[
            : algorithm_config.diagnostic_top_keywords_limit
        ],
        "baseline_period_length": current_idx - last_boundary_idx - 1,
        "threshold_exceeded": False,
    }


def _log_boundary_detection(
    logger,
    year: int,
    base_score: float,
    final_score: float,
    threshold: float,
    boost_factor: float,
    diagnostics: Dict,
) -> None:
    """Log boundary detection with details."""
    if diagnostics["citation_support"]:
        logger.info(
            f"      BOUNDARY DETECTED at {year}: "
            f"base={base_score:.3f} + boost={boost_factor:.3f} = {final_score:.3f} > threshold={threshold:.3f} "
            f"baseline_period={diagnostics['baseline_period']}"
        )
    else:
        logger.info(
            f"      BOUNDARY DETECTED at {year}: "
            f"score={final_score:.3f} > threshold={threshold:.3f} "
            f"baseline_period={diagnostics['baseline_period']}"
        )


def _log_no_boundary(
    logger,
    year: int,
    base_score: float,
    final_score: float,
    threshold: float,
    boost_factor: float,
    diagnostics: Dict,
) -> None:
    """Log no boundary detection with details."""
    if diagnostics["citation_support"]:
        logger.info(
            f"      {year}: "
            f"base={base_score:.3f} + boost={boost_factor:.3f} = {final_score:.3f} ≤ threshold={threshold:.3f} "
            f"baseline_period={diagnostics['baseline_period']}"
        )
    else:
        logger.info(
            f"      {year}: "
            f"score={final_score:.3f} ≤ threshold={threshold:.3f} "
            f"baseline_period={diagnostics['baseline_period']}"
        )


def _finalize_detection_results(
    algorithm_config,
    final_year_diagnostics: Dict,
    boundaries: List[int],
    citation_years: List[int],
    support_window: int,
    boost_factor: float,
    eligible_years: List[AcademicYear],
    verbose: bool,
) -> None:
    """Finalize detection results with statistics and diagnostics."""
    if not final_year_diagnostics:
        return

    # Calculate distribution statistics
    distribution_stats = _calculate_distribution_statistics(
        final_year_diagnostics, boundaries, algorithm_config
    )

    if verbose:
        _log_final_statistics(final_year_diagnostics, distribution_stats)

    # Save diagnostics if enabled
    _save_diagnostics_if_enabled(
        algorithm_config,
        final_year_diagnostics,
        distribution_stats,
        citation_years,
        boost_factor,
        support_window,
        eligible_years,
        verbose,
    )


def _calculate_distribution_statistics(
    final_year_diagnostics: Dict, boundaries: List[int], algorithm_config
) -> Dict:
    """Calculate distribution statistics for final results."""
    base_scores = [diag["base_score"] for diag in final_year_diagnostics.values()]
    final_scores = [diag["final_score"] for diag in final_year_diagnostics.values()]
    citation_boosted_count = sum(
        1 for diag in final_year_diagnostics.values() if diag["citation_support"]
    )

    # Handle empty score lists
    if not base_scores:
        base_score_stats = {
            "min": 0.0,
            "max": 0.0,
            "mean": 0.0,
            "std": 0.0,
            "median": 0.0,
            "p90": 0.0,
            "p95": 0.0,
            "p99": 0.0,
        }
    else:
        base_score_stats = {
            "min": min(base_scores),
            "max": max(base_scores),
            "mean": np.mean(base_scores),
            "std": np.std(base_scores),
            "median": np.median(base_scores),
            "p90": np.percentile(base_scores, 90),
            "p95": np.percentile(base_scores, 95),
            "p99": np.percentile(base_scores, 99),
        }

    if not final_scores:
        final_score_stats = {
            "min": 0.0,
            "max": 0.0,
            "mean": 0.0,
            "std": 0.0,
            "median": 0.0,
            "p90": 0.0,
            "p95": 0.0,
            "p99": 0.0,
        }
    else:
        final_score_stats = {
            "min": min(final_scores),
            "max": max(final_scores),
            "mean": np.mean(final_scores),
            "std": np.std(final_scores),
            "median": np.median(final_scores),
            "p90": np.percentile(final_scores, 90),
            "p95": np.percentile(final_scores, 95),
            "p99": np.percentile(final_scores, 99),
        }

    return {
        "count": len(base_scores),
        "base_scores": base_score_stats,
        "final_scores": final_score_stats,
        "years_above_threshold": len(boundaries),
        "threshold_used": None,  # Will be set by caller
        "threshold_strategy": getattr(
            algorithm_config, "direction_threshold_strategy", "global_p90"
        ),
        "citation_boosted_count": citation_boosted_count,
    }


def _log_final_statistics(
    final_year_diagnostics: Dict, distribution_stats: Dict
) -> None:
    """Log final detection statistics."""
    logger = get_logger(__name__, True)

    base_scores = [diag["base_score"] for diag in final_year_diagnostics.values()]
    final_scores = [diag["final_score"] for diag in final_year_diagnostics.values()]

    if base_scores and final_scores:
        logger.info(
            f"    Base scores: min={min(base_scores):.3f}, max={max(base_scores):.3f}, "
            f"avg={np.mean(base_scores):.3f}"
        )
        logger.info(
            f"    Final scores (with boost): min={min(final_scores):.3f}, max={max(final_scores):.3f}, "
            f"avg={np.mean(final_scores):.3f}"
        )
        logger.info(
            f"    Citation boosted: {distribution_stats['citation_boosted_count']}/{len(final_year_diagnostics)} years"
        )
    else:
        logger.info("    No scores to report (empty diagnostics)")
        logger.info(
            f"    Citation boosted: {distribution_stats['citation_boosted_count']}/{len(final_year_diagnostics)} years"
        )


def _save_diagnostics_if_enabled(
    algorithm_config,
    final_year_diagnostics: Dict,
    distribution_stats: Dict,
    citation_years: List[int],
    boost_factor: float,
    support_window: int,
    eligible_years: List[AcademicYear],
    verbose: bool,
) -> None:
    """Save diagnostics if enabled in configuration."""
    save_diagnostics = getattr(algorithm_config, "save_direction_diagnostics", False)
    if save_diagnostics and hasattr(algorithm_config, "domain_name"):
        from core.utils.diagnostics import save_direction_diagnostics

        diagnostics_path = save_direction_diagnostics(
            f"{algorithm_config.domain_name}_streamlined",
            eligible_years,  # Use actual eligible_years instead of empty list
            final_year_diagnostics,
            distribution_stats,
            {
                "citation_years": citation_years,
                "boost_factor": boost_factor,
                "support_window": support_window,
            },
            verbose,
        )

        if verbose:
            logger = get_logger(__name__, verbose)
            logger.info(f"    Saved streamlined diagnostics to: {diagnostics_path}")


# =============================================================================
# LEGACY FUNCTION ALIASES (for backward compatibility)
# =============================================================================


# Legacy function aliases
compute_direction_score_by_method = calculate_direction_score
compute_frequency_weighted_score = _calculate_weighted_jaccard_score
compute_jensen_shannon_divergence = _calculate_jensen_shannon_divergence
compute_adaptive_threshold_score = calculate_threshold_from_scores
normalize_frequencies = normalize_keyword_frequencies
