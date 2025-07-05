# Analysis Sub-module
# Handles objective function evaluation and paper analysis

# Import objective function
from .objective_function import (
    evaluate_timeline_quality,
    compute_objective_function,
    ObjectiveFunctionResult,
    AntiGamingConfig,
    SegmentMetrics,
    TransitionMetrics,
    jaccard_cohesion,
    jensen_shannon_separation,
    compute_jaccard_cohesion,
    compute_jensen_shannon_separation
)

# Import paper analysis
from .paper_analysis import (
    select_representative_papers,
    load_period_context,
    generate_period_label_and_description,
    generate_merged_segment_label_and_description
)

# Export all  
__all__ = [
    # Objective function
    'evaluate_timeline_quality',
    'compute_objective_function',
    'ObjectiveFunctionResult',
    'AntiGamingConfig',
    'SegmentMetrics',
    'TransitionMetrics',
    'jaccard_cohesion',
    'jensen_shannon_separation',
    'compute_jaccard_cohesion',
    'compute_jensen_shannon_separation',
    
    # Paper analysis
    'select_representative_papers',
    'load_period_context',
    'generate_period_label_and_description',
    'generate_merged_segment_label_and_description'
] 