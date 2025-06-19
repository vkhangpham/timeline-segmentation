"""
Change point detection for scientific literature analysis.

This module implements Enhanced Shift Signal Detection for identifying
research paradigm shifts using rich citation network and semantic data.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, TYPE_CHECKING

from .data_models import (
    DomainData, ChangePoint, ChangeDetectionResult
)
from .shift_signal_detection import detect_shift_signals

if TYPE_CHECKING:
    from .integration import SensitivityConfig


def detect_changes(domain_data: DomainData, sensitivity_config: 'SensitivityConfig') -> ChangeDetectionResult:
    """
    Perform paradigm shift detection using Enhanced Shift Signal Detection with centralized sensitivity control.
    
    Args:
        domain_data: Domain data with papers and rich citations
        sensitivity_config: Centralized sensitivity configuration controlling all thresholds
        
    Returns:
        Change detection results with paradigm shifts
    """    
    # Get domain file name for configuration
    domain_file_name = domain_data.domain_name.lower().replace(' ', '_')
    
    # Detect paradigm shifts using enhanced algorithm with sensitivity configuration
    shift_signals, transition_evidence, clustering_metadata = detect_shift_signals(
        domain_data, 
        domain_file_name, 
        sensitivity_config=sensitivity_config
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
    
    return ChangeDetectionResult(
        domain_name=domain_data.domain_name,
        time_range=domain_data.year_range,
        change_points=tuple(change_points),
        burst_periods=tuple(burst_periods),
        statistical_significance=statistical_significance,
        validation_score=0.0  # Legacy field
    )


def create_segments_with_confidence(change_years: List[int], time_range: Tuple[int, int], 
                                           statistical_significance: float = 0.5,
                                           domain_name: str = "") -> List[List[int]]:
    """
    Create segments using statistical significance-calibrated algorithm.
    
    Research-backed improvements:
    1. Dynamic minimum segment length based on statistical significance
    2. Conservative merging when significance is low  
    3. Domain-specific calibration using successful domain patterns
    
    Args:
        change_years: List of change point years (may contain duplicates)
        time_range: (min_year, max_year) of data
        statistical_significance: Statistical significance score from change detection
        domain_name: Domain name for logging/debugging
        
    Returns:
        List of [start_year, end_year] segments with research-backed calibration
    """
    min_year, max_year = time_range
    
    # Dynamic minimum segment length based on statistical significance
    # Research insight: successful domains (0.49-0.55 significance) preserve 4-6 year segments
    # Failed domains (0.31-0.38 significance) need more conservative approach
    if statistical_significance >= 0.5:
        # High confidence: allow shorter segments (like successful domains)
        min_segment_length = 4
    elif statistical_significance >= 0.4:
        # Medium confidence: moderate segments
        min_segment_length = 6
    else:
        # Low confidence: more conservative merging, but not too aggressive
        min_segment_length = 8
    
    print(f"📊 Domain: {domain_name}, Statistical Significance: {statistical_significance:.3f}")
    print(f"📏 Calibrated minimum segment length: {min_segment_length} years")
    
    # Step 1: Remove duplicates while preserving order
    unique_points = []
    seen = set()
    for cp in sorted(change_years):
        if cp not in seen and cp > min_year and cp < max_year:  # Valid range check
            unique_points.append(cp)
            seen.add(cp)
    
    if not unique_points:
        # Return single segment covering entire range
        return [[min_year, max_year]]
    
    print(f"🔍 Detected {len(change_years)} change points, {len(unique_points)} unique valid points")
    
    # Step 2: Create initial segments
    segments = []
    start_year = min_year
    
    for cp in unique_points:
        if cp > start_year:
            end_year = cp - 1
            if end_year >= start_year:  # Ensure valid segment
                segments.append([start_year, end_year])
                start_year = cp
    
    # Add final segment
    if start_year <= max_year:
        segments.append([start_year, max_year])
    
    print(f"📋 Initial segments: {len(segments)}")
    
    # Step 3: Apply research-backed merging with statistical significance consideration
    merged_segments = merge_segments_with_confidence(segments, min_segment_length, statistical_significance, domain_name)
    
    print(f"🔗 Final segments after merging: {len(merged_segments)}")
    for i, (start, end) in enumerate(merged_segments):
        length = end - start + 1
        print(f"   Segment {i+1}: {start}-{end} ({length} years)")
    
    return merged_segments


def create_improved_segments(change_years: List[int], time_range: Tuple[int, int], min_segment_length: int = 3) -> List[List[int]]:
    """
    Legacy segment creation function (Phase 6 backup).
    Kept for compatibility and comparison purposes.
    """
    # Redirect to new implementation with default parameters
    return create_segments_with_confidence(change_years, time_range, statistical_significance=0.5)


def merge_segments_with_confidence(segments: List[List[int]], min_length: int, 
                                 statistical_significance: float, domain_name: str) -> List[List[int]]:
    """
    Research-backed segment merging using statistical significance calibration.
    
    Key improvements:
    1. Conservative merging when statistical significance is low
    2. Preserve meaningful boundaries even if segments are short
    3. Avoid creating unrealistic long segments (like 168-year Art segment)
    
    Args:
        segments: List of [start_year, end_year] segments
        min_length: Minimum acceptable segment length
        statistical_significance: Statistical significance to guide merging aggressiveness  
        domain_name: Domain name for debugging
        
    Returns:
        List of merged segments with research-backed calibration
    """
    if not segments:
        return []
    
    # Research insight: when statistical significance is very low, be more conservative
    # Avoid creating extremely long segments like the 168-year Art segment
    max_segment_length = 50 if statistical_significance < 0.4 else 100
    
    merged = []
    i = 0
    
    while i < len(segments):
        current_start, current_end = segments[i]
        segment_length = current_end - current_start + 1
        
        # Check if current segment is acceptable
        if segment_length >= min_length:
            # Segment is acceptable, keep it
            merged.append([current_start, current_end])
            i += 1
            continue
        
        # Current segment is too short, need to merge
        # Research insight: prefer merging with adjacent segment that creates more reasonable length
        
        # Option 1: Merge with previous segment (if exists)
        can_merge_prev = (merged and 
                         (merged[-1][1] - merged[-1][0] + 1 + segment_length) <= max_segment_length)
        
        # Option 2: Merge with next segment (if exists)  
        can_merge_next = (i + 1 < len(segments) and
                         (segment_length + segments[i + 1][1] - segments[i + 1][0] + 1) <= max_segment_length)
        
        if can_merge_prev and (not can_merge_next or statistical_significance < 0.4):
            # Merge backward (conservative approach for low significance)
            prev_start, prev_end = merged[-1]
            merged[-1] = [prev_start, current_end]
            print(f"   ⬅️  Merged segment {current_start}-{current_end} with previous ({prev_start}-{prev_end})")
        elif can_merge_next:
            # Merge forward
            next_start, next_end = segments[i + 1]
            merged_segment = [current_start, next_end]
            merged.append(merged_segment)
            print(f"   ➡️  Merged segment {current_start}-{current_end} with next ({next_start}-{next_end})")
            i += 1  # Skip the next segment since we merged it
        elif merged:
            # Force merge with previous if no other option (but cap the length)
            prev_start, prev_end = merged[-1]
            if (prev_end - prev_start + 1 + segment_length) <= max_segment_length:
                merged[-1] = [prev_start, current_end]
                print(f"   ⬅️  Force merged short segment {current_start}-{current_end} with previous")
            else:
                # Keep short segment rather than create unrealistic long segment
                merged.append([current_start, current_end])
                print(f"   ⚠️  Kept short segment {current_start}-{current_end} to avoid unrealistic length")
        else:
            # No merging options, keep the segment
            merged.append([current_start, current_end])
            print(f"   📌 Kept first segment {current_start}-{current_end} (no merge options)")
        
        i += 1
    
    return merged 