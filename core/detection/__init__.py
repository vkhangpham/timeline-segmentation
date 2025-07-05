# Detection Sub-module  
# Handles change point detection, shift signals, and period signals

# Import data models for detection
from ..data.models import (
    ShiftSignal,
    ChangeDetectionResult,
    ChangePoint,
    TransitionEvidence,
    PeriodCharacterization
)

# Import change detection
from .change_detection import (
    detect_changes
)

# Import shift signal detection
from .shift_signals import (
    detect_shift_signals,
    validate_direction_with_citation,
    detect_research_direction_changes,
    detect_citation_structural_breaks
)

# Import period signal detection  
from .period_signals import (
    characterize_periods,
    analyze_network_stability,
    measure_community_persistence
)

# Export all
__all__ = [
    # Data models
    'ShiftSignal',
    'ChangeDetectionResult',
    'ChangePoint',
    'TransitionEvidence',
    'PeriodCharacterization',
    
    # Change detection
    'detect_changes',
    
    # Shift signal detection
    'detect_shift_signals',
    'validate_direction_with_citation',
    'detect_research_direction_changes',
    'detect_citation_structural_breaks',
    
    # Period signal detection
    'characterize_periods',
    'analyze_network_stability',
    'measure_community_persistence'
] 