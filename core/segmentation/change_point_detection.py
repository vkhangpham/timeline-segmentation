"""Direction change detection for research timeline modeling.

This module implements direction change detection using frequency-weighted scoring
and adaptive thresholding to identify paradigm shifts in academic domains.
"""

import numpy as np
from typing import List, Dict, Optional

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
        logger.info("=== SHIFT DETECTION STARTED ===")
        logger.info(f"  Domain: {domain_name}")
        logger.info(f"  Academic years: {len(academic_years)}")
        logger.info(
            f"  Year range: {min(ay.year for ay in academic_years)}-{max(ay.year for ay in academic_years)}"
        )
        logger.info(f"  Use direction detection: {use_direction}")
        logger.info(f"  Use citation detection: {use_citation}")

    if precomputed_signals:
        citation_years = precomputed_signals.get("citation_years", [])
        boundary_years = precomputed_signals.get("boundary_years", [])
        if verbose:
            logger.info(f"Using precomputed signals: {len(boundary_years)} boundaries")
    else:
        # Step 1: Compute citation acceleration years first
        citation_years = []
        if use_citation:
            if verbose:
                logger.info("  Step 1: Computing citation acceleration years...")
            citation_years = detect_citation_acceleration_years(
                academic_years, domain_name, algorithm_config, verbose
            )
            if verbose:
                logger.info(f"  Citation acceleration years: {citation_years}")

        # Step 2: Compute direction scores with immediate citation boost
        boundary_years = []
        if use_direction:
            if verbose:
                logger.info(
                    "  Step 2: Computing direction scores with citation boost..."
                )
            boundary_years = detect_direction_change_years_with_citation_boost(
                academic_years, citation_years, algorithm_config, verbose
            )
            if verbose:
                logger.info(f"  Final boundary years: {boundary_years}")

    year_lookup = {
        academic_year.year: academic_year for academic_year in academic_years
    }
    boundary_academic_years = []

    for year in boundary_years:
        if year in year_lookup:
            boundary_academic_years.append(year_lookup[year])
        else:
            logger.warning(f"Boundary year {year} not found in academic years")

    logger.info(
        f"Detected {len(boundary_academic_years)} boundary years: {[ay.year for ay in boundary_academic_years]}"
    )
    return boundary_academic_years


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

    # --- Simple & Effective MAD-based YoY growth detection -------------------
    # 1. Build year → citation mapping and ensure chronological order
    year_citations = {ay.year: ay.total_citations for ay in academic_years}
    if len(year_citations) < 5:
        logger.warning("Insufficient citation data for acceleration detection")
        return []

    years = np.array(sorted(year_citations.keys()))
    citations = np.array([year_citations[y] for y in years], dtype=float)

    # 2. Remove the most recent L years – citations still accumulating
    recent_lag = 2  # Hardcoded: recent years have incomplete citations
    if len(citations) > recent_lag:
        years = years[:-recent_lag]
        citations = citations[:-recent_lag]

    # 3. Smooth minor noise with a small moving-average
    smoothing_window = 5 # Hardcoded: small window for noise reduction
    if len(citations) >= smoothing_window:
        smoothed = moving_average(citations, smoothing_window)
        smoothed = np.pad(smoothed, (1, 1), mode="edge")[: len(citations)]
    else:
        smoothed = citations

    # 4. Compute year-over-year relative growth g_t = (c_t − c_{t−1}) / c_{t−1}
    growth = np.diff(smoothed) / (smoothed[:-1] + 1e-9)

    if len(growth) == 0:
        return []

    # 5. Robust threshold: median + k * MAD
    median_g = float(np.median(growth))
    mad_g = float(np.median(np.abs(growth - median_g)))
    threshold = (
        median_g + 3.0 * mad_g
    )  # Hardcoded: 3.0 is standard for outlier detection

    # 6. Basic safeguards against early-year low counts
    citation_median = float(np.median(citations))

    # 7. Identify break years, enforcing a cool-down to avoid clusters
    break_years: List[int] = []
    cooldown = 2  # Hardcoded: prevent clustering of detections
    for idx, g in enumerate(growth, start=1):  # start=1 aligns with current year
        year = int(years[idx])

        # Skip years with very small citation base (noise-prone)
        if citations[idx] < 0.1 * citation_median:  # Hardcoded: 10% threshold
            continue

        if g > threshold:
            if not break_years or year - break_years[-1] > cooldown:
                break_years.append(year)

    logger.info(
        "Detected %d citation acceleration years using MAD method: %s",
        len(break_years),
        break_years,
    )
    return break_years


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

    # Aggregate from last boundary + 1 to current - 1
    for year_idx in range(last_boundary_idx + 1, current_idx):
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


def normalize_frequencies(keyword_frequencies: Dict[str, int]) -> Dict[str, float]:
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


def compute_frequency_weighted_score(
    current_frequencies: Dict[str, float],
    baseline_frequencies: Dict[str, float],
) -> float:
    """Compute frequency-weighted direction score using weighted Jaccard similarity.

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


def compute_jensen_shannon_divergence(
    current_frequencies: Dict[str, float],
    baseline_frequencies: Dict[str, float],
) -> float:
    """Compute Jensen-Shannon Divergence between two probability distributions.

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
    # Using scipy.stats.entropy for KL divergence computation
    from scipy.stats import entropy

    kl_pm = entropy(p, m)
    kl_qm = entropy(q, m)

    jsd = 0.5 * kl_pm + 0.5 * kl_qm

    # Convert from nats to bits and normalize to [0,1]
    jsd_normalized = jsd / np.log(2)

    return min(jsd_normalized, 1.0)  # Ensure bounded [0,1]


def compute_direction_score_by_method(
    current_frequencies: Dict[str, float],
    baseline_frequencies: Dict[str, float],
    method: str = "weighted_jaccard",
) -> float:
    """Compute direction score using specified method.

    Args:
        current_frequencies: Normalized frequencies for current year
        baseline_frequencies: Normalized frequencies for baseline period
        method: Scoring method ("weighted_jaccard", "jensen_shannon")

    Returns:
        Direction score based on chosen method
    """
    if method == "weighted_jaccard":
        return compute_frequency_weighted_score(
            current_frequencies, baseline_frequencies
        )
    elif method == "jensen_shannon":
        return compute_jensen_shannon_divergence(
            current_frequencies, baseline_frequencies
        )
    else:
        raise ValueError(f"Unknown scoring method: {method}")


def compute_adaptive_threshold_score(
    all_scores: List[float],
    strategy: str = "global_p90",
    fixed_threshold: float = 0.1,
) -> float:
    """Compute adaptive threshold based on score distribution.

    Args:
        all_scores: List of all direction scores
        strategy: Threshold strategy ("fixed", "global_p90", "global_p95", "global_p99")
        fixed_threshold: Fixed threshold value to use when strategy is "fixed"

    Returns:
        Adaptive threshold value
    """
    if strategy == "fixed":
        return fixed_threshold
    elif strategy == "global_p90":
        if not all_scores:
            return fixed_threshold  # Fallback to fixed threshold
        return np.percentile(all_scores, 90)
    elif strategy == "global_p95":
        if not all_scores:
            return fixed_threshold  # Fallback to fixed threshold
        return np.percentile(all_scores, 95)
    elif strategy == "global_p99":
        if not all_scores:
            return fixed_threshold  # Fallback to fixed threshold
        return np.percentile(all_scores, 99)
    else:
        return fixed_threshold  # Fallback to fixed threshold


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
    min_papers_threshold = getattr(algorithm_config, "min_papers_per_year", 100)
    eligible_years = [
        ay for ay in academic_years if ay.paper_count >= min_papers_threshold
    ]

    if len(eligible_years) < 3:
        logger.warning("Insufficient years for direction change detection")
        return []

    if verbose:
        logger.info(f"    Analyzing {len(eligible_years)} eligible years...")

    # Citation boost parameters
    support_window = algorithm_config.citation_support_window_years
    boost_factor = algorithm_config.citation_confidence_boost

    # Step 1: Cumulative scoring simulation to collect BASE scores for adaptive threshold
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

    # Simulate cumulative scoring at regular intervals to get BASE score distribution
    last_boundary_idx = 0
    for current_idx in range(
        min_baseline_years, len(eligible_years), sampling_interval
    ):
        current_year = eligible_years[current_idx]

        # Ensure minimum baseline period from simulated last boundary
        if current_idx - last_boundary_idx < min_baseline_years:
            continue

        # Build cumulative baseline from simulated last boundary to current year
        baseline_frequencies = build_cumulative_baseline(
            eligible_years, last_boundary_idx, current_idx
        )

        # Get current year frequencies (normalized)
        current_frequencies = normalize_frequencies(current_year.keyword_frequencies)

        # Compute base direction score using cumulative comparison
        base_score = compute_direction_score_by_method(
            current_frequencies, baseline_frequencies, scoring_method
        )

        # Collect ONLY base scores (no citation boost) for threshold calculation
        base_sample_scores.append(base_score)

    # Step 2: Calculate adaptive threshold from BASE sample scores
    if not base_sample_scores:
        logger.warning(
            f"No base scores collected with cumulative simulation, using fixed threshold"
        )
        threshold = algorithm_config.direction_change_threshold
    else:
        threshold_strategy = getattr(
            algorithm_config, "direction_threshold_strategy", "global_p90"
        )
        threshold = compute_adaptive_threshold_score(
            base_sample_scores,
            threshold_strategy,
            algorithm_config.direction_change_threshold,
        )

        if verbose:
            logger.info("    Step 2: Calculated adaptive threshold from BASE scores")
            logger.info(
                f"    Using threshold strategy: {threshold_strategy}, threshold={threshold:.3f}"
            )
            logger.info(
                f"    Base sample score range: {min(base_sample_scores):.3f}-{max(base_sample_scores):.3f}"
            )

    # Step 3: Proper cumulative scoring with immediate boundary updates
    if verbose:
        logger.info(
            "    Step 3: Running cumulative algorithm with proper boundary updates..."
        )

    boundaries = []
    final_year_diagnostics = {}
    last_boundary_idx = 0

    for current_idx in range(1, len(eligible_years)):
        current_year = eligible_years[current_idx]

        # Ensure minimum baseline period from last boundary
        min_baseline_years = getattr(algorithm_config, "min_baseline_period_years", 3)
        if current_idx - last_boundary_idx < min_baseline_years:
            continue

        # Build cumulative baseline from last boundary to current year
        baseline_frequencies = build_cumulative_baseline(
            eligible_years, last_boundary_idx, current_idx
        )

        # Get current year frequencies (normalized)
        current_frequencies = normalize_frequencies(current_year.keyword_frequencies)

        # Compute base direction score using cumulative comparison
        base_score = compute_direction_score_by_method(
            current_frequencies, baseline_frequencies, scoring_method
        )

        # Check for citation support and boost immediately
        has_citation_support = False
        if citation_years:
            for cit_year in citation_years:
                if abs(cit_year - current_year.year) <= support_window:
                    has_citation_support = True
                    break

        # Apply citation boost if supported
        if has_citation_support:
            final_score = base_score + boost_factor
            final_score = min(final_score, 1.0)  # Cap at 1.0
        else:
            final_score = base_score

        # Compute diagnostic information
        baseline_keywords = set(baseline_frequencies.keys())
        current_keywords = set(current_frequencies.keys())
        new_keywords = current_keywords - baseline_keywords
        shared_keywords = current_keywords & baseline_keywords

        baseline_period_str = (
            f"{eligible_years[last_boundary_idx + 1].year}-{eligible_years[current_idx - 1].year}"
            if current_idx > last_boundary_idx + 1
            else "empty"
        )

        final_year_diagnostics[current_year.year] = {
            "baseline_period": baseline_period_str,
            "base_score": base_score,
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

        # Check if this year exceeds threshold (after applying citation boost to base score)
        if final_score > threshold:
            boundaries.append(current_year.year)
            last_boundary_idx = current_idx  # Immediately update for next iterations

            # Update diagnostic
            final_year_diagnostics[current_year.year]["threshold_exceeded"] = True

            if verbose:
                if has_citation_support:
                    logger.info(
                        f"      BOUNDARY DETECTED at {current_year.year}: "
                        f"base={base_score:.3f} + boost={boost_factor:.3f} = {final_score:.3f} > threshold={threshold:.3f} "
                        f"baseline_period={baseline_period_str}"
                    )
                else:
                    logger.info(
                        f"      BOUNDARY DETECTED at {current_year.year}: "
                        f"score={final_score:.3f} > threshold={threshold:.3f} "
                        f"baseline_period={baseline_period_str}"
                    )
        else:
            if verbose:
                if has_citation_support:
                    logger.info(
                        f"      {current_year.year}: "
                        f"base={base_score:.3f} + boost={boost_factor:.3f} = {final_score:.3f} ≤ threshold={threshold:.3f} "
                        f"baseline_period={baseline_period_str}"
                    )
                else:
                    logger.info(
                        f"      {current_year.year}: "
                        f"score={final_score:.3f} ≤ threshold={threshold:.3f} "
                        f"baseline_period={baseline_period_str}"
                    )

    # Compute distribution statistics based on BASE scores (since threshold is calculated from base scores)
    distribution_stats = {}
    if final_year_diagnostics:
        base_scores = [diag["base_score"] for diag in final_year_diagnostics.values()]
        final_scores = [diag["final_score"] for diag in final_year_diagnostics.values()]
        citation_boosted_count = sum(
            1 for diag in final_year_diagnostics.values() if diag["citation_support"]
        )

        distribution_stats = {
            "count": len(base_scores),
            "base_scores": {
                "min": min(base_scores),
                "max": max(base_scores),
                "mean": np.mean(base_scores),
                "std": np.std(base_scores),
                "median": np.median(base_scores),
                "p90": np.percentile(base_scores, 90),
                "p95": np.percentile(base_scores, 95),
                "p99": np.percentile(base_scores, 99),
            },
            "final_scores": {
                "min": min(final_scores),
                "max": max(final_scores),
                "mean": np.mean(final_scores),
                "std": np.std(final_scores),
                "median": np.median(final_scores),
                "p90": np.percentile(final_scores, 90),
                "p95": np.percentile(final_scores, 95),
                "p99": np.percentile(final_scores, 99),
            },
            "years_above_threshold": len(boundaries),
            "threshold_used": threshold,
            "threshold_strategy": threshold_strategy,
            "citation_boosted_count": citation_boosted_count,
        }

    if verbose:
        if final_year_diagnostics:
            cumulative_base_scores = [
                diag["base_score"] for diag in final_year_diagnostics.values()
            ]
            cumulative_final_scores = [
                diag["final_score"] for diag in final_year_diagnostics.values()
            ]
            logger.info(
                f"    Base scores: min={min(cumulative_base_scores):.3f}, max={max(cumulative_base_scores):.3f}, "
                f"avg={np.mean(cumulative_base_scores):.3f}"
            )
            logger.info(
                f"    Final scores (with boost): min={min(cumulative_final_scores):.3f}, max={max(cumulative_final_scores):.3f}, "
                f"avg={np.mean(cumulative_final_scores):.3f}"
            )
            logger.info(
                f"    Threshold (calculated from base scores): {threshold:.3f} "
                f"({threshold_strategy})"
            )
            logger.info(
                f"    Citation boosted: {distribution_stats['citation_boosted_count']}/{len(final_year_diagnostics)} years"
            )
        logger.info(f"    Detected {len(boundaries)} boundaries: {boundaries}")

    # Save diagnostics if enabled
    save_diagnostics = getattr(algorithm_config, "save_direction_diagnostics", False)
    if save_diagnostics and hasattr(algorithm_config, "domain_name"):
        from core.utils.diagnostics import save_direction_diagnostics

        diagnostics_path = save_direction_diagnostics(
            f"{algorithm_config.domain_name}_streamlined",
            academic_years,
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
            logger.info(f"    Saved streamlined diagnostics to: {diagnostics_path}")

    logger.info(f"Detected {len(boundaries)} streamlined boundary years")
    return boundaries
