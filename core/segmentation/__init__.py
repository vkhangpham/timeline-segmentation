# Segmentation Sub-module
# Handles boundary segmentation, segment modeling, and segment merging

# Import simplified boundary segmentation functions
from .boundary import (
    create_segments_from_boundary_years,
    validate_period_contiguity,
    get_boundary_transparency_report,
)

from .shift_signals import (
    detect_boundary_years,
    detect_direction_change_years,
    detect_citation_acceleration_years,
    validate_and_combine_signals,
)

# Import simplified segment merging
from .merging import (
    merge_similar_periods,
)

# Export all
__all__ = [
    # Simplified boundary segmentation
    "create_segments_from_boundary_years",
    "validate_period_contiguity",
    "get_boundary_transparency_report",
    # Shift signal detection
    "detect_boundary_years",
    "detect_direction_change_years",
    "detect_citation_acceleration_years",
    "validate_and_combine_signals",
    # Simplified segment merging
    "merge_similar_periods",
]
