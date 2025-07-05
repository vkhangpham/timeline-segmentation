# Pipeline Sub-module
# Handles high-level pipeline orchestration and workflow management

# Import pipeline orchestrator
from .orchestrator import (
    run_complete_analysis,
    run_timeline_analysis,
    run_change_detection
)

# Export all
__all__ = [
    'run_complete_analysis',
    'run_timeline_analysis', 
    'run_change_detection'
] 