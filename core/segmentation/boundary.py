"""
Boundary-Based Segmentation with AcademicPeriod Return (REFACTOR-003)

This module implements efficient segmentation using pre-computed AcademicYear
structures and returns AcademicPeriod objects for optimal timeline evaluation.

Key Features:
- Uses pre-computed AcademicYear structures for all temporal data
- Returns AcademicPeriod objects for seamless integration with objective function
- Uses enhanced objective function for fast evaluation via AcademicPeriod structures
- Deterministic and transparent boundary placement
- Scalable optimization strategies for different problem sizes

Following project guidelines:
- Fail-fast error handling (no fallbacks)
- Functional programming approach (pure functions)
- Real data usage (leverages pre-computed academic structures)
- Transparent and explainable results

Architecture:
- boundary.py: Takes validated signals, returns AcademicPeriod objects
- objective_function.py: Takes AcademicPeriod objects, returns scores
- Clear separation of concerns and single responsibility principle
"""

from typing import List, Tuple, Dict, Any
from ..data.data_models import AcademicYear, AcademicPeriod
from ..utils.logging import get_logger

logger = get_logger(__name__)


# =============================================================================
# SIMPLIFIED BOUNDARY FUNCTIONS (for new architecture)
# =============================================================================


def create_segments_from_boundary_years(
    boundary_academic_years: List[AcademicYear],
    academic_years: Tuple[AcademicYear, ...],
    algorithm_config,
    verbose: bool = False,
) -> List[AcademicPeriod]:
    """
    Create AcademicPeriod objects directly from boundary academic years.

    This function takes validated signal AcademicYear objects and creates
    AcademicPeriod objects representing the timeline segments.

    Args:
        boundary_academic_years: AcademicYear objects where paradigm shifts occur
        academic_years: Available academic year data
        algorithm_config: Algorithm configuration
        verbose: Enable verbose logging

    Returns:
        List of AcademicPeriod objects representing the timeline segments
    """
    # Import here to avoid circular import
    from ..data.data_processing import create_academic_periods_from_segments

    logger = get_logger(__name__, verbose)

    if not academic_years:
        return []

    # Get year range from AVAILABLE academic years (not full domain range)
    available_years = sorted([year.year for year in academic_years])
    min_year = available_years[0]
    max_year = available_years[-1]

    if verbose:
        logger.info(f"=== SEGMENTATION STARTED ===")
        logger.info(f"  Available year range: {min_year}-{max_year}")
        logger.info(f"  Total academic years: {len(academic_years)}")
        logger.info(f"  Boundary academic years: {len(boundary_academic_years)}")

    # Extract boundary years from AcademicYear objects
    boundary_years = [ay.year for ay in boundary_academic_years]

    # Sort boundary years
    boundaries = sorted(set(boundary_years))

    if verbose:
        logger.info(f"  Raw boundary years: {boundary_years}")
        logger.info(f"  Sorted unique boundaries: {boundaries}")

    # Filter boundaries to be within available year range
    valid_boundaries = [year for year in boundaries if min_year < year <= max_year]

    if verbose:
        logger.info(f"  Valid boundaries (within range): {valid_boundaries}")

    if not valid_boundaries:
        if verbose:
            logger.info("  No valid boundary years - creating single period")
        # Create a single period covering all available years
        single_period_segments = [(min_year, max_year)]
        return create_academic_periods_from_segments(
            academic_years, single_period_segments
        )

    # Create segment tuples
    segments = []
    start_year = min_year

    for boundary_year in valid_boundaries:
        if boundary_year > start_year:
            segments.append((start_year, boundary_year - 1))
            start_year = boundary_year

    # Add final segment
    if start_year <= max_year:
        segments.append((start_year, max_year))

    if verbose:
        logger.info(f"  Created segments: {segments}")

    logger.info(
        f"Created {len(segments)} segments from {len(valid_boundaries)} boundaries"
    )

    # Convert segments to AcademicPeriod objects
    academic_periods = create_academic_periods_from_segments(academic_years, segments)

    if verbose:
        for i, period in enumerate(academic_periods):
            logger.info(
                f"  Period {i+1}: {period.start_year}-{period.end_year} ({period.total_papers} papers, {period.total_citations} citations)"
            )

    logger.info(f"Generated {len(academic_periods)} AcademicPeriod objects")
    return academic_periods


# =============================================================================
# LEGACY BOUNDARY FUNCTIONS REMOVED
# =============================================================================
# All legacy functions (create_boundary_periods_from_academic_years,
# create_optimized_boundary_periods_from_academic_years, and related helper functions)
# have been removed as they used deprecated ShiftSignal model.


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================


def validate_period_contiguity(academic_periods: List[AcademicPeriod]) -> bool:
    """
    Validate that AcademicPeriod objects are contiguous with no gaps or overlaps.

    Args:
        academic_periods: List of AcademicPeriod objects

    Returns:
        True if periods are perfectly contiguous, False otherwise

    Raises:
        ValueError: If periods list is empty
    """
    if not academic_periods:
        raise ValueError("Cannot validate empty periods list")

    # Sort periods by start year
    sorted_periods = sorted(academic_periods, key=lambda p: p.start_year)

    # Check each period transition
    for i in range(len(sorted_periods) - 1):
        current_end = sorted_periods[i].end_year
        next_start = sorted_periods[i + 1].start_year

        if current_end + 1 != next_start:
            # Gap or overlap detected
            return False

    return True


def get_boundary_transparency_report(
    academic_periods: List[AcademicPeriod], metadata: Dict[str, Any]
) -> str:
    """
    Generate human-readable transparency report for boundary-based segmentation.

    Args:
        academic_periods: List of AcademicPeriod objects
        metadata: Metadata from segmentation functions

    Returns:
        Formatted transparency report string
    """
    report = "=== Boundary-Based Segmentation Report ===\n"
    report += f"Method: {metadata.get('boundary_method', 'Unknown')}\n"
    report += f"Data Source: {metadata.get('data_source', 'Unknown')}\n"
    report += f"Return Type: {metadata.get('return_type', 'Unknown')}\n"
    report += f"Generated {len(academic_periods)} contiguous periods\n"
    report += f"Signal years: {metadata.get('signal_years', [])}\n\n"

    for i, period in enumerate(academic_periods):
        period_data = metadata.get(f"period_{i}", {})
        rationale = period_data.get("boundary_rationale", "No rationale available")

        report += f"Period {i+1}: {period.start_year}-{period.end_year} ({period.end_year - period.start_year + 1} years)\n"
        report += f"  Papers: {period.total_papers}\n"
        report += f"  Top Keywords: {', '.join(period.top_keywords[:5])}\n"
        report += f"  Rationale: {rationale}\n\n"

    if metadata.get("performance_note"):
        report += f"Performance: {metadata['performance_note']}\n"

    if metadata.get("recommendation"):
        report += f"Recommendation: {metadata['recommendation']}\n"

    return report


# Export main functions
__all__ = [
    # Simplified functions for new architecture
    "create_segments_from_boundary_years",
    # Utility functions
    "validate_period_contiguity",
    "get_boundary_transparency_report",
]
