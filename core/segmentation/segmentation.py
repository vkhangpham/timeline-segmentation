"""Boundary-based segmentation with AcademicPeriod objects.

This module implements efficient segmentation using pre-computed AcademicYear
structures and returns AcademicPeriod objects for timeline evaluation.
"""

from typing import List, Tuple, Dict, Any
from ..data.data_models import AcademicYear, AcademicPeriod
from ..utils.logging import get_logger

logger = get_logger(__name__)


def create_segments_from_boundary_years(
    boundary_academic_years: List[AcademicYear],
    academic_years: Tuple[AcademicYear, ...],
    algorithm_config,
    verbose: bool = False,
) -> List[AcademicPeriod]:
    """Create AcademicPeriod objects directly from boundary academic years.

    Args:
        boundary_academic_years: AcademicYear objects where paradigm shifts occur
        academic_years: Available academic year data
        algorithm_config: Algorithm configuration
        verbose: Enable verbose logging

    Returns:
        List of AcademicPeriod objects representing the timeline segments
    """
    from ..data.data_processing import create_academic_periods_from_segments

    logger = get_logger(__name__, verbose)

    if not academic_years:
        if verbose:
            logger.warning("No academic years provided for segmentation")
        return []

    available_years = sorted([year.year for year in academic_years])
    min_year = available_years[0]
    max_year = available_years[-1]

    if verbose:
        logger.info(
            f"Segmentation: {len(academic_years)} years ({min_year}-{max_year}), {len(boundary_academic_years)} boundaries"
        )

    # Handle empty boundary years - create single period
    if not boundary_academic_years:
        single_period_segments = [(min_year, max_year)]
        return create_academic_periods_from_segments(
            academic_years, single_period_segments, algorithm_config
        )

    boundary_years = [ay.year for ay in boundary_academic_years]
    boundaries = sorted(set(boundary_years))

    valid_boundaries = [year for year in boundaries if min_year < year <= max_year]

    if not valid_boundaries:
        single_period_segments = [(min_year, max_year)]
        return create_academic_periods_from_segments(
            academic_years, single_period_segments, algorithm_config
        )

    segments = []
    start_year = min_year

    for boundary_year in valid_boundaries:
        if boundary_year > start_year:
            segments.append((start_year, boundary_year - 1))
            start_year = boundary_year

    if start_year <= max_year:
        segments.append((start_year, max_year))

    logger.info(
        f"Segments: {segments} ({len(segments)} from {len(valid_boundaries)} boundaries)"
    )

    academic_periods = create_academic_periods_from_segments(
        academic_years, segments, algorithm_config
    )

    if verbose:
        for i, period in enumerate(academic_periods):
            logger.info(
                f"  {i+1}: {period.start_year}-{period.end_year} ({period.total_papers}p, {period.total_citations}c)"
            )

    logger.info(f"Generated {len(academic_periods)} AcademicPeriod objects")
    return academic_periods


def validate_period_contiguity(academic_periods: List[AcademicPeriod]) -> bool:
    """Validate that AcademicPeriod objects are contiguous with no gaps or overlaps.

    Args:
        academic_periods: List of AcademicPeriod objects

    Returns:
        True if periods are perfectly contiguous, False otherwise

    Raises:
        ValueError: If periods list is empty
    """
    if not academic_periods:
        raise ValueError("Cannot validate empty periods list")

    sorted_periods = sorted(academic_periods, key=lambda p: p.start_year)

    for i in range(len(sorted_periods) - 1):
        current_end = sorted_periods[i].end_year
        next_start = sorted_periods[i + 1].start_year

        if current_end + 1 != next_start:
            return False

    return True


def get_boundary_transparency_report(
    academic_periods: List[AcademicPeriod], metadata: Dict[str, Any]
) -> str:
    """Generate human-readable transparency report for boundary-based segmentation.

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
        report += f"Period {i+1}: {period.start_year}-{period.end_year}\n"
        report += f"  Papers: {period.total_papers}\n"
        report += f"  Citations: {period.total_citations}\n"
        if period.topic_label:
            report += f"  Topic: {period.topic_label}\n"
        report += "\n"

    return report
