"""Timeline segmentation components for boundary detection and merging."""

from .segmentation import (
    create_segments_from_boundary_years,
    validate_period_contiguity,
    get_boundary_transparency_report,
)

from .change_point_detection import (
    detect_boundary_years,
    detect_direction_change_years,
    detect_citation_acceleration_years,
    validate_and_combine_signals,
)

from .segment_merging import (
    merge_similar_periods,
)

__all__ = [
    "create_segments_from_boundary_years",
    "validate_period_contiguity",
    "get_boundary_transparency_report",
    "detect_boundary_years",
    "detect_direction_change_years",
    "detect_citation_acceleration_years",
    "validate_and_combine_signals",
    "merge_similar_periods",
]
