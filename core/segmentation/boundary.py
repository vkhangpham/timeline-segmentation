"""
Boundary-Based Segmentation (REFACTOR-003)

This module implements simple deterministic segmentation using validated shift signals
as direct boundaries, replacing the complex similarity-based approach.

Key Features:
- Uses shift signal years directly as segment boundaries
- No heuristic similarity analysis or length constraints
- Relies on objective function's anti-gaming safeguards for quality control
- Deterministic and transparent boundary placement
- Minimal code footprint (~50 lines vs ~576 lines in similarity_segmentation)

Following project guidelines:
- Fail-fast error handling (no fallbacks)
- Functional programming approach (pure functions)
- Real data usage (leverages validated shift signals)
- Transparent and explainable results
"""

from typing import List, Tuple, Dict, Any
from ..data.models import ShiftSignal, DomainData


def create_boundary_segments(
    validated_signals: List[ShiftSignal],
    domain_data: DomainData
) -> Tuple[List[Tuple[int, int]], Dict[str, Any]]:
    """
    Create contiguous timeline segments using shift signal years as direct boundaries.
    
    This is the simplified replacement for similarity-based segmentation.
    Uses validated shift signals as cut points to create segments, letting the
    objective function's anti-gaming safeguards handle quality control.
    
    Algorithm:
    1. Sort validated signals chronologically
    2. Use signal years as direct boundaries
    3. Create segments: [domain_start, signal1-1], [signal1, signal2-1], ..., [signalN, domain_end]
    4. Return segments with transparency metadata
    
    Args:
        validated_signals: List of validated paradigm shift signals
        domain_data: Domain data containing year range information
        
    Returns:
        Tuple of (segments, transparency_metadata) where:
        - segments: List of (start_year, end_year) tuples for contiguous segments
        - transparency_metadata: Dict explaining boundary placement decisions
        
    Raises:
        ValueError: If validated_signals is empty or domain_data is invalid
        
    Example:
        signals = [ShiftSignal(year=1985), ShiftSignal(year=1995)]
        domain_data = DomainData(year_range=(1980, 2010), ...)
        segments, metadata = create_boundary_segments(signals, domain_data)
        # Returns: [(1980, 1984), (1985, 1994), (1995, 2010)], {...}
    """
    
    if not validated_signals:
        raise ValueError("Cannot create segments from empty validated_signals list")
    
    if not domain_data or not hasattr(domain_data, 'year_range'):
        raise ValueError("domain_data must have valid year_range attribute")
    
    # Get domain year range
    min_year, max_year = domain_data.year_range
    
    if min_year >= max_year:
        raise ValueError(f"Invalid domain year range: {min_year} >= {max_year}")
    
    # Sort signals chronologically for boundary placement
    sorted_signals = sorted(validated_signals, key=lambda s: s.year)
    
    # Validate signals are within domain range
    for signal in sorted_signals:
        if not (min_year <= signal.year <= max_year):
            raise ValueError(f"Signal year {signal.year} outside domain range [{min_year}, {max_year}]")
    
    # Handle single signal case - split domain at signal year
    if len(sorted_signals) == 1:
        signal_year = sorted_signals[0].year
        segments = [
            (min_year, signal_year - 1),
            (signal_year, max_year)
        ]
        
        metadata = {
            "segments_count": 2,
            "signal_years": [signal_year],
            "boundary_method": "direct_signal_boundaries",
            "segment_0": {
                "range": (min_year, signal_year - 1),
                "boundary_rationale": f"Domain start to signal at {signal_year}"
            },
            "segment_1": {
                "range": (signal_year, max_year),
                "boundary_rationale": f"Signal at {signal_year} to domain end"
            }
        }
        
        return segments, metadata
    
    # Multiple signals case - create boundaries at each signal year
    segments = []
    signal_years = [s.year for s in sorted_signals]
    
    # First segment: domain start to first signal
    segments.append((min_year, sorted_signals[0].year - 1))
    
    # Middle segments: signal to next signal
    for i in range(len(sorted_signals) - 1):
        start_year = sorted_signals[i].year
        end_year = sorted_signals[i + 1].year - 1
        segments.append((start_year, end_year))
    
    # Last segment: last signal to domain end
    segments.append((sorted_signals[-1].year, max_year))
    
    # Create transparency metadata
    metadata = {
        "segments_count": len(segments),
        "signal_years": signal_years,
        "boundary_method": "direct_signal_boundaries",
        "algorithm": "deterministic_boundary_placement",
        "anti_gaming_note": "Quality control handled by objective function anti-gaming safeguards"
    }
    
    # Add detailed rationale for each segment
    for i, (start_year, end_year) in enumerate(segments):
        if i == 0:
            rationale = f"Domain start to first signal at {signal_years[0]}"
        elif i == len(segments) - 1:
            rationale = f"Last signal at {signal_years[-1]} to domain end"
        else:
            rationale = f"Signal at {signal_years[i-1]} to signal at {signal_years[i]}"
        
        metadata[f"segment_{i}"] = {
            "range": (start_year, end_year),
            "length": end_year - start_year + 1,
            "boundary_rationale": rationale
        }
    
    return segments, metadata


def validate_segment_contiguity(segments: List[Tuple[int, int]]) -> bool:
    """
    Validate that segments are contiguous with no gaps or overlaps.
    
    Args:
        segments: List of (start_year, end_year) segment tuples
        
    Returns:
        True if segments are perfectly contiguous, False otherwise
        
    Raises:
        ValueError: If segments list is empty
    """
    
    if not segments:
        raise ValueError("Cannot validate empty segments list")
    
    # Sort segments by start year
    sorted_segments = sorted(segments, key=lambda seg: seg[0])
    
    # Check each segment transition
    for i in range(len(sorted_segments) - 1):
        current_end = sorted_segments[i][1]
        next_start = sorted_segments[i + 1][0]
        
        if current_end + 1 != next_start:
            # Gap or overlap detected
            return False
    
    return True


def get_boundary_transparency_report(
    segments: List[Tuple[int, int]], 
    metadata: Dict[str, Any]
) -> str:
    """
    Generate human-readable transparency report for boundary-based segmentation.
    
    Args:
        segments: List of segment tuples
        metadata: Metadata from create_boundary_segments()
        
    Returns:
        Formatted transparency report string
    """
    
    report = "=== Boundary-Based Segmentation Report ===\n"
    report += f"Method: {metadata.get('boundary_method', 'Unknown')}\n"
    report += f"Generated {len(segments)} contiguous segments\n"
    report += f"Signal years: {metadata.get('signal_years', [])}\n\n"
    
    for i, (start, end) in enumerate(segments):
        segment_data = metadata.get(f"segment_{i}", {})
        length = segment_data.get('length', end - start + 1)
        rationale = segment_data.get('boundary_rationale', 'No rationale available')
        
        report += f"Segment {i+1}: {start}-{end} ({length} years)\n"
        report += f"  Rationale: {rationale}\n\n"
    
    if metadata.get('anti_gaming_note'):
        report += f"Quality Control: {metadata['anti_gaming_note']}\n"
    
    return report


# Export main functions
__all__ = [
    'create_boundary_segments',
    'validate_segment_contiguity', 
    'get_boundary_transparency_report'
] 