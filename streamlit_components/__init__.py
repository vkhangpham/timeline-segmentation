"""
Streamlit Timeline App Components

This package contains modular components for the Timeline Segmentation Algorithm Dashboard.
Components are organized by functionality to maintain clean separation of concerns.
"""

from .utils import load_all_domains, run_algorithm_with_params
from .analysis_overview import (
    create_keyword_filtering_plot,
    create_direction_signal_detection_plot,
    create_citation_signal_detection_plot,
    create_final_validation_plot,
    create_similarity_segmentation_plot,
)
from .keyword_evolution import (
    prepare_keyword_evolution_data,
    create_keyword_streamgraph,
    create_enhanced_keyword_heatmap,
)
from .decision_analysis import (
    create_decision_tree_analysis,
    create_decision_flow_diagram,
    create_parameter_sensitivity_analysis,
    create_keyword_filtering_impact_analysis,
)

__all__ = [
    # Utilities
    "load_all_domains",
    "run_algorithm_with_params",
    # Analysis Overview (Tab 1)
    "create_keyword_filtering_plot",
    "create_direction_signal_detection_plot",
    "create_citation_signal_detection_plot",
    "create_final_validation_plot",
    "create_similarity_segmentation_plot",
    # Keyword Evolution (Tab 2)
    "prepare_keyword_evolution_data",
    "create_keyword_streamgraph",
    "create_enhanced_keyword_heatmap",
    # Decision Analysis (Tab 3)
    "create_decision_tree_analysis",
    "create_decision_flow_diagram",
    "create_parameter_sensitivity_analysis",
    "create_keyword_filtering_impact_analysis",
] 