"""
Change point detection for scientific literature analysis.

This module implements Enhanced Shift Signal Detection for identifying
research paradigm shifts using rich citation network and semantic data.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, TYPE_CHECKING

from .data_models import (
    DomainData, ChangePoint, ChangeDetectionResult, ShiftSignal
)
from .shift_signal_detection import detect_shift_signals

if TYPE_CHECKING:
    from .algorithm_config import AlgorithmConfig


def detect_changes(domain_data: DomainData, algorithm_config: 'AlgorithmConfig') -> Tuple[ChangeDetectionResult, List[ShiftSignal]]:
    """
    Perform paradigm shift detection using Enhanced Shift Signal Detection with comprehensive algorithm configuration.
    
    Args:
        domain_data: Domain data with papers and rich citations
        algorithm_config: Comprehensive algorithm configuration controlling all parameters
        
    Returns:
        Change detection results with paradigm shifts
    """    
    # Get domain file name for configuration
    domain_file_name = domain_data.domain_name.lower().replace(' ', '_')
    
    # Detect paradigm shifts using enhanced algorithm with comprehensive algorithm configuration
    shift_signals = detect_shift_signals(
        domain_data, 
        domain_file_name, 
        algorithm_config=algorithm_config
    )
    
    # Convert shift signals to change points for compatibility
    change_points = []
    for signal in shift_signals:
        change_point = ChangePoint(
            year=signal.year,
            confidence=signal.confidence,
            method="enhanced_shift_signal",
            signal_type=signal.signal_type,
            description=signal.transition_description,
            supporting_evidence=tuple(signal.supporting_evidence)
        )
        change_points.append(change_point)
    
    # Calculate overall statistical significance
    if change_points:
        confidences = [cp.confidence for cp in change_points]
        statistical_significance = np.mean(confidences)
    else:
        statistical_significance = 0.0
    
    # Create empty burst periods for backward compatibility
    burst_periods = []
    
    change_result = ChangeDetectionResult(
        domain_name=domain_data.domain_name,
        time_range=domain_data.year_range,
        change_points=tuple(change_points),
        burst_periods=tuple(burst_periods),
        statistical_significance=statistical_significance,
        validation_score=0.0  # Legacy field
    )
    
    return change_result, shift_signals


# =============================================================================
# STATISTICAL SEGMENTATION REMOVED (IMPROVEMENT-002)
# =============================================================================
# Statistical segmentation functions have been completely removed.
# Only similarity segmentation is now used, handled in similarity_segmentation.py module. 