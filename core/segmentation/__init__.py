# Segmentation Sub-module
# Handles boundary segmentation, segment modeling, and segment merging

# Import data models for segmentation
from ..data.models import (
    SegmentModelingResult,
    SegmentMergingResult,
    MergeDecision
)

# Import boundary segmentation
from .boundary import (
    create_boundary_segments,
    validate_segment_contiguity,
    get_boundary_transparency_report
)

# Import segment modeling
from .modeling import (
    model_segments,
    generate_modeling_summary,
    validate_segment_modeling_result,
    get_modeling_statistics
)

# Import segment merging
from .merging import (
    merge_similar_segments
)

# Export all
__all__ = [
    # Data models
    'SegmentModelingResult',
    'SegmentMergingResult',
    'MergeDecision',
    
    # Boundary segmentation
    'create_boundary_segments',
    'validate_segment_contiguity',
    'get_boundary_transparency_report',
    
    # Segment modeling
    'model_segments',
    'generate_modeling_summary',
    'validate_segment_modeling_result',
    'get_modeling_statistics',
    
    # Segment merging
    'merge_similar_segments'
] 