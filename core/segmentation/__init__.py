"""Timeline segmentation components for boundary detection and merging."""

from .segmentation import (
    create_segments_from_boundary_years,
    validate_period_contiguity,
    get_boundary_transparency_report,
)

from .change_point_detection import detect_boundary_years
from .citation_detection import detect_citation_acceleration_years
from .direction_detection import detect_direction_change_years_with_citation_boost

__all__ = [
    "create_segments_from_boundary_years",
    "validate_period_contiguity",
    "get_boundary_transparency_report",
    "detect_boundary_years",
    "detect_direction_change_years_with_citation_boost",
    "detect_citation_acceleration_years",
]
