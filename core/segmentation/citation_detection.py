"""Citation acceleration detection for research timeline modeling.

This module implements citation acceleration detection using regime-specific methods
and adaptive thresholding to identify periods of rapid citation growth.
"""

import numpy as np
from typing import List

from ..data.data_models import AcademicYear
from ..utils.logging import get_logger


# =============================================================================
# MAIN DETECTION FUNCTIONS
# =============================================================================


def detect_citation_acceleration_years(
    academic_years: List[AcademicYear],
    verbose: bool = False,
) -> List[int]:
    """Detect citation acceleration years using regime-aware methods.

    This method addresses the fundamental bias in traditional relative growth analysis
    by adapting detection strategies based on citation regime characteristics.

    Args:
        academic_years: List of AcademicYear objects with papers and temporal data
        verbose: Enable verbose logging

    Returns:
        List of years where citation acceleration occurs
    """
    logger = get_logger(__name__, verbose)

    # Prepare and validate citation data
    year_citations = {ay.year: ay.total_citations for ay in academic_years}
    if len(year_citations) < 5:
        logger.warning("Insufficient citation data for acceleration detection")
        return []

    years = np.array(sorted(year_citations.keys()))
    citations = np.array([year_citations[y] for y in years], dtype=float)

    # Remove recent years (incomplete citations)
    years, citations = _remove_recent_years(years, citations, lag=2)
    if len(citations) < 5:
        logger.warning("Insufficient data after removing recent years")
        return []

    # Apply regime-specific detection
    acceleration_years = []
    for i, year in enumerate(years):
        regime = _classify_citation_regime(citations, i)

        if _is_acceleration_detected(regime, citations, years, i, verbose):
            acceleration_years.append(int(year))

    # Apply cooldown to prevent clustering
    final_years = _apply_cooldown_filter(acceleration_years, cooldown=2)

    if verbose:
        _log_detection_results(logger, final_years)

    return final_years


# =============================================================================
# CITATION REGIME CLASSIFICATION
# =============================================================================


def _classify_citation_regime(
    citations: np.ndarray, index: int, window_size: int = 10
) -> str:
    """Classify citation regime based on local citation statistics.

    Args:
        citations: Array of citation counts
        index: Current index in citations array
        window_size: Size of local window for statistics

    Returns:
        Regime classification: "sparse", "emerging", or "mature"
    """
    local_window = _get_local_window(citations, index, window_size)

    if len(local_window) < 3:
        return "sparse"  # Default for insufficient data

    median_citations = np.median(local_window)

    # Classification thresholds based on citation magnitude
    if median_citations < 100:
        return "sparse"
    elif median_citations < 1000:
        return "emerging"
    else:
        return "mature"


def _get_local_window(
    citations: np.ndarray, index: int, window_size: int
) -> np.ndarray:
    """Extract local window around current index."""
    start_idx = max(0, index - window_size // 2)
    end_idx = min(len(citations), index + window_size // 2 + 1)
    return citations[start_idx:end_idx]


# =============================================================================
# REGIME-SPECIFIC ACCELERATION DETECTION
# =============================================================================


def _is_acceleration_detected(
    regime: str, citations: np.ndarray, years: np.ndarray, index: int, verbose: bool
) -> bool:
    """Detect acceleration using regime-specific methods."""
    if regime == "sparse":
        return _detect_sparse_regime_acceleration(citations, years, index, verbose)
    elif regime == "emerging":
        return _detect_emerging_regime_acceleration(citations, years, index, verbose)
    elif regime == "mature":
        return _detect_mature_regime_acceleration(citations, years, index, verbose)
    else:
        return False


def _detect_sparse_regime_acceleration(
    citations: np.ndarray, years: np.ndarray, index: int, verbose: bool
) -> bool:
    """Detect acceleration in sparse citation regime using absolute changes.

    Args:
        citations: Array of citation counts
        years: Array of years
        index: Current index
        verbose: Enable verbose logging

    Returns:
        True if acceleration detected
    """
    if index < 5:  # Need sufficient history for sparse regime
        return False

    # Calculate changes with larger window for sparse regime
    window_size = min(7, index)
    current_citations = citations[index]
    baseline_citations = np.mean(citations[index - window_size : index])
    absolute_change = current_citations - baseline_citations

    # Dynamic thresholds based on local statistics
    thresholds = _calculate_sparse_regime_thresholds(citations, index)

    # Check if both absolute and relative thresholds are met
    relative_change = absolute_change / (baseline_citations + 1e-9)
    passes_absolute = absolute_change > thresholds["absolute"]
    passes_relative = relative_change > thresholds["relative"]

    # Additional validations
    if not _is_sustained_growth(citations, index, baseline_citations):
        return False

    if not _is_significant_vs_domain_growth(citations, index, absolute_change):
        return False

    return passes_absolute and passes_relative


def _detect_emerging_regime_acceleration(
    citations: np.ndarray, years: np.ndarray, index: int, verbose: bool
) -> bool:
    """Detect acceleration in emerging citation regime using hybrid approach.

    Args:
        citations: Array of citation counts
        years: Array of years
        index: Current index
        verbose: Enable verbose logging

    Returns:
        True if acceleration detected
    """
    if index < 4:
        return False

    # Calculate changes with moderate window for emerging regime
    window_size = min(6, index)
    current_citations = citations[index]
    baseline_citations = np.mean(citations[index - window_size : index])

    absolute_change = current_citations - baseline_citations
    relative_change = absolute_change / (baseline_citations + 1e-9)

    # Adaptive thresholds for emerging regime
    thresholds = _calculate_emerging_regime_thresholds(citations, index)

    # Check for sustained growth
    if not _is_sustained_growth_emerging(citations, index, baseline_citations):
        return False

    return (
        absolute_change > thresholds["absolute"]
        and relative_change > thresholds["relative"]
    )


def _detect_mature_regime_acceleration(
    citations: np.ndarray, years: np.ndarray, index: int, verbose: bool
) -> bool:
    """Detect acceleration in mature citation regime using scale-aware normalization.

    Args:
        citations: Array of citation counts
        years: Array of years
        index: Current index
        verbose: Enable verbose logging

    Returns:
        True if acceleration detected
    """
    if index < 3:
        return False

    # Calculate changes with smaller window for mature regime (more responsive)
    window_size = min(5, index)
    current_citations = citations[index]
    baseline_citations = np.mean(citations[index - window_size : index])

    absolute_change = current_citations - baseline_citations
    relative_change = absolute_change / (baseline_citations + 1e-9)

    # Scale-aware threshold calculation
    thresholds = _calculate_mature_regime_thresholds(citations, index)

    # Check for positive acceleration (second derivative)
    if _has_positive_acceleration(citations, index):
        thresholds["absolute"] *= 0.8  # Lower threshold if accelerating

    return (
        absolute_change > thresholds["absolute"]
        and relative_change > thresholds["relative"]
    )


# =============================================================================
# THRESHOLD CALCULATION HELPERS
# =============================================================================


def _calculate_sparse_regime_thresholds(citations: np.ndarray, index: int) -> dict:
    """Calculate thresholds for sparse citation regime."""
    local_window = citations[max(0, index - 15) : index + 1]
    local_std = np.std(local_window)

    return {
        "absolute": max(50, 3 * local_std),  # 3 standard deviations
        "relative": 1.0,  # 100% increase minimum
    }


def _calculate_emerging_regime_thresholds(citations: np.ndarray, index: int) -> dict:
    """Calculate thresholds for emerging citation regime."""
    local_window = citations[max(0, index - 12) : index + 1]
    local_std = np.std(local_window)

    return {
        "absolute": max(100, 2 * local_std),  # 2 standard deviations
        "relative": 0.5,  # 50% relative increase
    }


def _calculate_mature_regime_thresholds(citations: np.ndarray, index: int) -> dict:
    """Calculate thresholds for mature citation regime."""
    local_window = citations[max(0, index - 10) : index + 1]
    local_median = np.median(local_window)
    local_std = np.std(local_window)

    return {
        "absolute": max(1000, 0.1 * local_median, 1.5 * local_std),
        "relative": 0.2,  # At least 20% increase
    }


# =============================================================================
# VALIDATION HELPERS
# =============================================================================


def _is_sustained_growth(
    citations: np.ndarray, index: int, baseline_citations: float
) -> bool:
    """Check if growth is sustained (not just a temporary spike)."""
    if index >= len(citations) - 2:
        return True  # Can't check future, assume sustained

    next_citations = citations[index + 1]
    next_next_citations = (
        citations[index + 2] if index + 2 < len(citations) else next_citations
    )

    # Reject if it's just a temporary spike
    return not (
        next_citations < baseline_citations and next_next_citations < baseline_citations
    )


def _is_sustained_growth_emerging(
    citations: np.ndarray, index: int, baseline_citations: float
) -> bool:
    """Check sustained growth for emerging regime."""
    if index >= len(citations) - 1:
        return True

    next_citations = citations[index + 1]
    # Must maintain at least 10% above baseline
    return next_citations >= baseline_citations * 1.1


def _is_significant_vs_domain_growth(
    citations: np.ndarray, index: int, absolute_change: float
) -> bool:
    """Check if change is significant relative to overall domain growth."""
    domain_growth = np.median(np.diff(citations[max(0, index - 20) : index + 1]))
    return absolute_change > 3 * domain_growth


def _has_positive_acceleration(citations: np.ndarray, index: int) -> bool:
    """Check for positive acceleration (second derivative)."""
    if index < 2:
        return False

    growth_rates = np.diff(citations[max(0, index - 5) : index + 1])
    if len(growth_rates) < 2:
        return False

    acceleration = np.diff(growth_rates)
    if len(acceleration) == 0:
        return False

    recent_acceleration = acceleration[-1]
    local_median = np.median(citations[max(0, index - 10) : index + 1])

    return recent_acceleration > 0.1 * local_median


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================


def _remove_recent_years(
    years: np.ndarray, citations: np.ndarray, lag: int = 2
) -> tuple:
    """Remove recent years due to incomplete citations."""
    if len(citations) > lag:
        return years[:-lag], citations[:-lag]
    return years, citations


def _apply_cooldown_filter(years: List[int], cooldown: int = 2) -> List[int]:
    """Apply cooldown period to prevent clustering of detections.

    Args:
        years: List of candidate years
        cooldown: Minimum years between detections

    Returns:
        Filtered list of years with cooldown applied
    """
    if not years:
        return years

    filtered_years = []
    last_year = None

    for year in sorted(years):
        if last_year is None or year - last_year > cooldown:
            filtered_years.append(year)
            last_year = year

    return filtered_years


def _log_detection_results(logger, final_years: List[int]) -> None:
    """Log detection results with temporal distribution."""
    modern_count = sum(1 for y in final_years if y >= 2000)
    mid_count = sum(1 for y in final_years if 1980 <= y < 2000)
    early_count = sum(1 for y in final_years if y < 1980)

    logger.info(
        f"Acceleration: {final_years} (early={early_count}, mid={mid_count}, modern={modern_count})"
    )


def calculate_moving_average(data: np.ndarray, window: int) -> np.ndarray:
    """Calculate moving average for smoothing citation data.

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


# Legacy alias for backward compatibility
moving_average = calculate_moving_average
