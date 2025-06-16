"""
Segment Modeling for Timeline Analysis

This module handles the complete segment modeling pipeline by:
1. Using period signal detection to characterize periods
2. Processing and validating period characterizations
3. Providing clean interface for integration module

Follows functional programming principles with pure functions and immutable data structures.
"""

from typing import Dict, List, Tuple, Optional, Any
import numpy as np

from .period_signal_detection import characterize_periods
from .data_models import PeriodCharacterization, SegmentModelingResult


def model_segments(
    domain_name: str, 
    segments: List[Tuple[int, int]]
) -> SegmentModelingResult:
    """
    Main function: Model timeline segments using period signal detection.
    
    Args:
        domain_name: Name of the research domain
        segments: List of time segments from shift signal detection
        
    Returns:
        Segment modeling results with period characterizations
    """
    print(f"\n🎯 SEGMENT MODELING: {domain_name}")
    print("=" * 50)
    
    if not segments:
        print("    ⚠️ No segments provided for modeling")
        return SegmentModelingResult(
            domain_name=domain_name,
            segments=tuple(),
            period_characterizations=tuple(),
            modeling_confidence=0.0,
            modeling_summary="No segments to model"
        )
    
    print(f"    📊 Modeling {len(segments)} segments using period signal detection")
    
    # Use period signal detection to characterize periods
    period_characterizations = characterize_periods(
        domain_name=domain_name,
        segments=segments
    )
    
    # Validate and process results
    valid_characterizations = []
    for pc in period_characterizations:
        if pc.confidence > 0.0:  # Only include valid characterizations
            valid_characterizations.append(pc)
        else:
            print(f"    ⚠️ Skipping low-confidence period {pc.period}: confidence={pc.confidence:.3f}")
    
    # Calculate overall modeling confidence
    if valid_characterizations:
        modeling_confidence = np.mean([pc.confidence for pc in valid_characterizations])
    else:
        modeling_confidence = 0.0
    
    # Generate modeling summary
    modeling_summary = generate_modeling_summary(
        segments, valid_characterizations, modeling_confidence
    )
    
    print(f"    ✅ Modeled {len(valid_characterizations)}/{len(segments)} segments successfully")
    print(f"    📈 Overall modeling confidence: {modeling_confidence:.3f}")
    print(f"    📋 {modeling_summary}")
    
    return SegmentModelingResult(
        domain_name=domain_name,
        segments=tuple(segments),
        period_characterizations=tuple(valid_characterizations),
        modeling_confidence=modeling_confidence,
        modeling_summary=modeling_summary
    )


def generate_modeling_summary(
    segments: List[Tuple[int, int]], 
    characterizations: List[PeriodCharacterization],
    confidence: float
) -> str:
    """
    Generate summary of segment modeling results.
    
    Args:
        segments: Original segments
        characterizations: Valid period characterizations
        confidence: Overall modeling confidence
        
    Returns:
        Summary string
    """
    if not characterizations:
        return "No valid period characterizations generated"
    
    # Confidence assessment
    if confidence >= 0.7:
        confidence_level = "EXCELLENT"
    elif confidence >= 0.5:
        confidence_level = "GOOD"
    elif confidence >= 0.3:
        confidence_level = "MODERATE"
    else:
        confidence_level = "LOW"
    
    # Calculate average network metrics
    avg_network_stability = np.mean([pc.network_stability for pc in characterizations])
    avg_community_persistence = np.mean([pc.community_persistence for pc in characterizations])
    
    summary_parts = [
        f"Successfully modeled {len(characterizations)}/{len(segments)} segments",
        f"Confidence: {confidence:.3f} ({confidence_level})",
        f"Avg network stability: {avg_network_stability:.3f}",
        f"Avg community persistence: {avg_community_persistence:.3f}"
    ]
    
    return "; ".join(summary_parts)


def validate_segment_modeling_result(result: SegmentModelingResult) -> bool:
    """
    Validate segment modeling result for quality assurance.
    
    Args:
        result: Segment modeling result to validate
        
    Returns:
        True if result is valid and meets quality criteria
    """
    # Basic validation
    if not result.period_characterizations:
        return False
    
    # Check confidence threshold
    if result.modeling_confidence < 0.1:
        return False
    
    # Check that all characterizations have valid periods
    for pc in result.period_characterizations:
        if pc.period[0] >= pc.period[1]:  # Invalid period
            return False
        if pc.confidence <= 0.0:  # Invalid confidence
            return False
    
    return True


def get_modeling_statistics(result: SegmentModelingResult) -> Dict[str, Any]:
    """
    Get detailed statistics from segment modeling result.
    
    Args:
        result: Segment modeling result
        
    Returns:
        Dictionary with detailed statistics
    """
    if not result.period_characterizations:
        return {
            'total_segments': len(result.segments),
            'valid_characterizations': 0,
            'success_rate': 0.0,
            'confidence_stats': {},
            'network_stats': {}
        }
    
    confidences = [pc.confidence for pc in result.period_characterizations]
    network_stabilities = [pc.network_stability for pc in result.period_characterizations]
    community_persistences = [pc.community_persistence for pc in result.period_characterizations]
    
    return {
        'total_segments': len(result.segments),
        'valid_characterizations': len(result.period_characterizations),
        'success_rate': len(result.period_characterizations) / len(result.segments),
        'confidence_stats': {
            'mean': np.mean(confidences),
            'std': np.std(confidences),
            'min': np.min(confidences),
            'max': np.max(confidences)
        },
        'network_stats': {
            'avg_stability': np.mean(network_stabilities),
            'avg_persistence': np.mean(community_persistences),
            'stability_range': (np.min(network_stabilities), np.max(network_stabilities)),
            'persistence_range': (np.min(community_persistences), np.max(community_persistences))
        }
    } 