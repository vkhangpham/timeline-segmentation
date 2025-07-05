# Results Sub-module
# Handles result management, saving, and display

# Import result manager
from .manager import (
    save_all_results,
    save_shift_signals,
    save_period_signals,
    save_analysis_results,
    ensure_results_directory_structure
)

# Import results display
from .display import (
    display_analysis_summary,
    format_timeline_narrative,
    format_segmentation_details,
    format_confidence_summary,
    print_detailed_results
)

# Export all
__all__ = [
    # Result management
    'save_all_results',
    'save_shift_signals',
    'save_period_signals',
    'save_analysis_results',
    'ensure_results_directory_structure',
    
    # Results display
    'display_analysis_summary',
    'format_timeline_narrative',
    'format_segmentation_details',
    'format_confidence_summary',
    'print_detailed_results'
] 