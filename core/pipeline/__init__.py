# Pipeline Sub-module
# Handles simplified pipeline orchestration and workflow management

# Import simplified pipeline orchestrator
from .orchestrator import (
    analyze_timeline,
    calculate_timeline_confidence,
    generate_narrative_evolution,
)

# Export all simplified functions
__all__ = [
    "analyze_timeline",
    "calculate_timeline_confidence",
    "generate_narrative_evolution",
]
