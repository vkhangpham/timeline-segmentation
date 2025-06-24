"""
Boundary-Based Similarity Segmentation (IMPROVEMENT-002)

This module implements contiguous timeline segmentation using validated signals as centroids
and keyword similarity to determine optimal segment boundaries. Replaces clustering approach
with explainable boundary detection that guarantees non-overlapping segments.

Key Features:
- Validated signals preserved as discrete centroids (no clustering loss)
- Contiguous segments guaranteed by boundary optimization algorithm  
- Complete transparency through similarity-based boundary explanations
- Leverages existing year_keywords infrastructure for simplicity
- Single parameter control for easy adoption

Following project guidelines:
- Fail-fast error handling (no fallbacks)
- Functional programming approach (pure functions)
- Real data usage (leverages existing keyword infrastructure)
- Rigorous validation and transparency
"""

from typing import List, Dict, Tuple, Any, Optional
from dataclasses import dataclass
from core.data_models import ShiftSignal, DomainData
from core.keyword_utils import extract_year_keywords, calculate_jaccard_similarity


def create_similarity_based_segments(
    validated_signals: List[ShiftSignal],
    year_keywords: Dict[int, List[str]],
    domain_data: DomainData,
    min_segment_length: int = 4,
    max_segment_length: int = 50
) -> Tuple[List[Tuple[int, int]], Dict[str, Any]]:
    """
    Create contiguous timeline segments using boundary-based similarity detection with length controls.
    
    Algorithm:
    1. Sort validated signals chronologically 
    2. Find optimal boundaries between adjacent signals based on keyword similarity
    3. Create initial segments from boundaries
    4. Enforce minimum segment length through intelligent merging
    5. Generate transparency metadata for each boundary and merge decision
    
    Args:
        validated_signals: List of validated paradigm shift signals (centroids)
        year_keywords: Dict mapping years to keyword lists (from existing infrastructure)
        domain_data: Domain data containing year range information
        min_segment_length: Minimum acceptable segment length in years
        max_segment_length: Maximum segment length to prevent unrealistic periods
        
    Returns:
        Tuple of (segments, transparency_metadata) where:
        - segments: List of (start_year, end_year) tuples for contiguous segments
        - transparency_metadata: Dict explaining each boundary and merge decision
        
    Raises:
        ValueError: If validated_signals is empty, year_keywords is invalid, or parameters are invalid
        
    Example:
        signals = [ShiftSignal(year=1985), ShiftSignal(year=1995)]
        year_keywords = {1980: ['old'], 1985: ['new'], 1990: ['hybrid'], 1995: ['modern']}
        segments, metadata = create_similarity_based_segments(
            signals, year_keywords, domain_data, min_segment_length=5
        )
        # Returns: [(1960, 1990), (1990, 2020)], {...}
    """
    
    if not validated_signals:
        raise ValueError("Cannot create segments from empty validated_signals list")
    
    if not year_keywords:
        raise ValueError("year_keywords dictionary cannot be empty")
    
    if not 1 <= min_segment_length <= 20:
        raise ValueError(f"min_segment_length must be 1-20 years, got {min_segment_length}")
    
    if not 10 <= max_segment_length <= 200:
        raise ValueError(f"max_segment_length must be 10-200 years, got {max_segment_length}")
    
    if min_segment_length >= max_segment_length:
        raise ValueError(f"min_segment_length ({min_segment_length}) must be < max_segment_length ({max_segment_length})")
    
    # 🔧 NEW: Check if configuration is feasible given signal density
    min_year, max_year = domain_data.year_range
    total_years = max_year - min_year + 1
    num_signals = len(validated_signals)
    
    # If we have too many signals for the minimum segment length, warn but proceed
    theoretical_min_years_needed = num_signals * min_segment_length
    if theoretical_min_years_needed > total_years:
        print(f"    ⚠️  WARNING: {num_signals} signals × {min_segment_length} min length = {theoretical_min_years_needed} years")
        print(f"    ⚠️  But domain only spans {total_years} years. Some segments may need to be shorter.")
        print(f"    ⚠️  Consider reducing min_segment_length or validation_threshold")
    
    # Get domain year range
    min_year, max_year = domain_data.year_range
    
    # Sort signals chronologically for boundary detection
    sorted_signals = sorted(validated_signals, key=lambda s: s.year)
    
    # Handle single signal case - entire domain range
    if len(sorted_signals) == 1:
        single_segment = [(min_year, max_year)]
        single_metadata = {
            "segments_count": 1,
            "signal_years": [sorted_signals[0].year],
            "segment_0": {
                "signal_year": sorted_signals[0].year,
                "segment_range": (min_year, max_year),
                "boundary_rationale": "Single signal - covers entire domain range"
            }
        }
        return single_segment, single_metadata
    
    # Find optimal boundaries between adjacent signals
    boundaries = [min_year]  # Start with domain minimum
    boundary_explanations = []
    
    for i in range(len(sorted_signals) - 1):
        signal_a = sorted_signals[i]
        signal_b = sorted_signals[i + 1]
        
            # Find optimal boundary between these signals
        boundary_year, explanation = find_optimal_boundary(
            signal_a.year, signal_b.year, year_keywords
        )
        
        boundaries.append(boundary_year)
        boundary_explanations.append(explanation)
    
    boundaries.append(max_year)  # End with domain maximum
    
    # Create initial segments from boundaries
    initial_segments = []
    for i in range(len(sorted_signals)):
        start_year = boundaries[i]
        end_year = boundaries[i + 1] - 1 if i < len(sorted_signals) - 1 else boundaries[i + 1]
        initial_segments.append((start_year, end_year, sorted_signals[i].year))  # Include signal year
    
    # Apply minimum segment length enforcement
    final_segments, merge_decisions = enforce_minimum_segment_length(
        initial_segments, min_segment_length, max_segment_length, year_keywords
    )
    
    # Create transparency metadata
    transparency_data = {
        "segments_count": len(final_segments),
        "signal_years": [s.year for s in sorted_signals],
        "boundary_explanations": boundary_explanations,
        "merge_decisions": merge_decisions,
        "min_segment_length": min_segment_length,
        "max_segment_length": max_segment_length
    }
    
    for i, (start_year, end_year) in enumerate(final_segments):
        transparency_data[f"segment_{i}"] = {
            "segment_range": (start_year, end_year),
            "segment_length": end_year - start_year + 1,
            "boundary_rationale": f"Similarity-based boundary with minimum length enforcement"
        }
    
    return final_segments, transparency_data


def find_optimal_boundary(
    signal_year_a: int,
    signal_year_b: int, 
    year_keywords: Dict[int, List[str]]
) -> Tuple[int, str]:
    """
    Find optimal boundary year where keyword similarity switches from signal A to signal B.
    
    UPDATED: Now skips years with empty keyword lists to avoid bias from sparse annotation.
    
    Scans years between the two signals to find the crossover point where
    keywords become more similar to signal B than signal A.
    
    Args:
        signal_year_a: Year of first signal (earlier)
        signal_year_b: Year of second signal (later)
        year_keywords: Dict mapping years to keyword lists
        
    Returns:
        Tuple of (boundary_year, explanation) where:
        - boundary_year: Year where similarity switches
        - explanation: Human-readable rationale for boundary placement
        
    Raises:
        ValueError: If signal_year_a >= signal_year_b
        
    Example:
        boundary, explanation = find_optimal_boundary(1985, 1995, year_keywords)
        # Returns: (1990, "Keywords switch similarity at 1990: sim_to_1985=0.2, sim_to_1995=0.4")
    """
    
    if signal_year_a >= signal_year_b:
        raise ValueError(f"signal_year_a ({signal_year_a}) must be < signal_year_b ({signal_year_b})")
    
    # Get signal keywords for similarity comparison
    signal_a_keywords = year_keywords.get(signal_year_a, [])
    signal_b_keywords = year_keywords.get(signal_year_b, [])
    
    # If no keywords available for signals, use midpoint
    if not signal_a_keywords and not signal_b_keywords:
        midpoint = (signal_year_a + signal_year_b) // 2
        return midpoint, f"No keywords for signals {signal_year_a},{signal_year_b} → midpoint {midpoint}"
    
    # Scan years between signals to find similarity crossover
    best_boundary = (signal_year_a + signal_year_b) // 2  # Default to midpoint
    best_explanation = "midpoint_default"
    
    for test_year in range(signal_year_a + 1, signal_year_b):
        test_keywords = year_keywords.get(test_year, [])
        
        # Skip years with no keywords (IMPROVEMENT: avoids empty-set bias)
        if not test_keywords:
            continue
        
        # Calculate similarities to both signals
        sim_to_a = calculate_jaccard_similarity(test_keywords, signal_a_keywords)
        sim_to_b = calculate_jaccard_similarity(test_keywords, signal_b_keywords)
        
        # Find crossover point where similarity switches
        if sim_to_b > sim_to_a:  
            # Test year is more similar to signal B - potential boundary
            best_boundary = test_year
            best_explanation = f"Keywords switch at {test_year}: sim_to_{signal_year_a}={sim_to_a:.3f}, sim_to_{signal_year_b}={sim_to_b:.3f}"
            break  # Take first crossover point
    
    return best_boundary, best_explanation


def calculate_jaccard_similarity(keywords_a: List[str], keywords_b: List[str]) -> float:
    """
    Calculate Jaccard similarity coefficient between two keyword lists.
    
    Jaccard similarity = |intersection| / |union|
    Ranges from 0.0 (no overlap) to 1.0 (identical sets)
    
    Args:
        keywords_a: First keyword list
        keywords_b: Second keyword list
        
    Returns:
        Jaccard similarity coefficient (0.0 to 1.0)
        
    Example:
        sim = calculate_jaccard_similarity(['a', 'b', 'c'], ['b', 'c', 'd'])
        # Returns: 0.5 (2 intersection / 4 union)
    """
    
    # Convert to sets for efficient set operations
    set_a = set(keywords_a) if keywords_a else set()
    set_b = set(keywords_b) if keywords_b else set()
    
    # Handle empty sets case
    if not set_a and not set_b:
        return 1.0  # Both empty = identical
    
    if not set_a or not set_b:
        return 0.0  # One empty = no similarity
    
    # Calculate Jaccard similarity
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    
    return intersection / union if union > 0 else 0.0


def enforce_minimum_segment_length(
    initial_segments: List[Tuple[int, int, int]],  # (start, end, signal_year)
    min_segment_length: int,
    max_segment_length: int,
    year_keywords: Dict[int, List[str]]
) -> Tuple[List[Tuple[int, int]], List[str]]:
    """
    Enforce minimum segment length through intelligent merging.
    
    Algorithm:
    1. Identify segments that are too short
    2. For each short segment, determine best merge direction using similarity analysis
    3. Merge segments while respecting maximum length constraints
    4. Track all merge decisions for transparency
    
    Args:
        initial_segments: List of (start_year, end_year, signal_year) tuples
        min_segment_length: Minimum acceptable segment length
        max_segment_length: Maximum acceptable segment length
        year_keywords: Year-to-keywords mapping for similarity analysis
        
    Returns:
        Tuple of (final_segments, merge_decisions) where:
        - final_segments: List of (start_year, end_year) tuples after merging
        - merge_decisions: List of human-readable merge decision explanations
    """
    
    if not initial_segments:
        return [], []
    
    # Convert to mutable format for processing
    segments = [(start, end, signal_year) for start, end, signal_year in initial_segments]
    merge_decisions = []
    
    print(f"    📏 Enforcing minimum segment length: {min_segment_length} years")
    print(f"    📊 Initial segments: {len(segments)}")
    
    i = 0
    while i < len(segments):
        start, end, signal_year = segments[i]
        segment_length = end - start + 1
        
        if segment_length >= min_segment_length:
            # Segment is acceptable, move to next
            i += 1
            continue
        
        # Segment is too short, need to merge
        print(f"    ⚠️  Segment {start}-{end} ({segment_length} years) is too short")
        
        # Determine merge direction
        can_merge_left = i > 0
        can_merge_right = i < len(segments) - 1
        
        if not can_merge_left and not can_merge_right:
            # Only segment - keep it even if short
            merge_decisions.append(f"Kept short segment {start}-{end} (only segment)")
            i += 1
            continue
        
        # Choose merge direction based on length constraints and similarity
        merge_left = False
        merge_right = False
        
        if can_merge_left:
            left_start, left_end, left_signal = segments[i-1]
            left_length = left_end - left_start + 1
            merged_left_length = left_length + segment_length
            merge_left = merged_left_length <= max_segment_length
        
        if can_merge_right:
            right_start, right_end, right_signal = segments[i+1]
            right_length = right_end - right_start + 1
            merged_right_length = segment_length + right_length
            merge_right = merged_right_length <= max_segment_length
        
        # Decision logic: prefer direction that creates more balanced segments
        if merge_left and merge_right:
            # Both directions possible - choose based on similarity or balance
            left_similarity = calculate_jaccard_similarity(
                year_keywords.get(signal_year, []),
                year_keywords.get(segments[i-1][2], [])
            )
            right_similarity = calculate_jaccard_similarity(
                year_keywords.get(signal_year, []),
                year_keywords.get(segments[i+1][2], [])
            )
            
            if left_similarity > right_similarity:
                # Merge left (higher similarity)
                merged_segment = (segments[i-1][0], end, segments[i-1][2])
                segments[i-1] = merged_segment
                segments.pop(i)
                merge_decisions.append(
                    f"Merged {start}-{end} ← (similarity {left_similarity:.3f} > {right_similarity:.3f})"
                )
                # Don't increment i, check merged segment
            else:
                # Merge right (higher similarity or equal)
                merged_segment = (start, segments[i+1][1], signal_year)
                segments[i] = merged_segment
                segments.pop(i+1)
                merge_decisions.append(
                    f"Merged {start}-{end} → (similarity {right_similarity:.3f} >= {left_similarity:.3f})"
                )
                # Don't increment i, check merged segment
                
        elif merge_left:
            # Only left merge possible
            merged_segment = (segments[i-1][0], end, segments[i-1][2])
            segments[i-1] = merged_segment
            segments.pop(i)
            merge_decisions.append(f"Merged {start}-{end} ← (only viable direction)")
            # Don't increment i, check merged segment
            
        elif merge_right:
            # Only right merge possible
            merged_segment = (start, segments[i+1][1], signal_year)
            segments[i] = merged_segment
            segments.pop(i+1)
            merge_decisions.append(f"Merged {start}-{end} → (only viable direction)")
            # Don't increment i, check merged segment
            
        else:
            # 🔧 FIXED: Prioritize minimum length over maximum length constraints
            # If we can't merge within max_segment_length, we must still merge to meet minimum
            
            # Force merge in the direction that violates max_segment_length least
            if can_merge_left and can_merge_right:
                # Choose direction that creates smallest violation
                left_start, left_end, left_signal = segments[i-1]
                left_length = left_end - left_start + 1
                merged_left_length = left_length + segment_length
                
                right_start, right_end, right_signal = segments[i+1]
                right_length = right_end - right_start + 1
                merged_right_length = segment_length + right_length
                
                if merged_left_length <= merged_right_length:
                    # Merge left (smaller violation or equal)
                    merged_segment = (segments[i-1][0], end, segments[i-1][2])
                    segments[i-1] = merged_segment
                    segments.pop(i)
                    merge_decisions.append(
                        f"Force-merged {start}-{end} ← (min_length priority, result={merged_left_length} years)"
                    )
                else:
                    # Merge right (smaller violation)
                    merged_segment = (start, segments[i+1][1], signal_year)
                    segments[i] = merged_segment
                    segments.pop(i+1)
                    merge_decisions.append(
                        f"Force-merged {start}-{end} → (min_length priority, result={merged_right_length} years)"
                    )
                    
            elif can_merge_left:
                # Force merge left
                merged_segment = (segments[i-1][0], end, segments[i-1][2])
                segments[i-1] = merged_segment
                segments.pop(i)
                merge_decisions.append(f"Force-merged {start}-{end} ← (min_length priority)")
                
            elif can_merge_right:
                # Force merge right
                merged_segment = (start, segments[i+1][1], signal_year)
                segments[i] = merged_segment
                segments.pop(i+1)
                merge_decisions.append(f"Force-merged {start}-{end} → (min_length priority)")
                
            else:
                # Truly cannot merge (only segment) - keep short segment
                merge_decisions.append(
                    f"Kept short segment {start}-{end} (only segment - cannot merge)"
                )
                i += 1
    
    # Convert back to simple tuples
    final_segments = [(start, end) for start, end, _ in segments]
    
    print(f"    ✅ Final segments: {len(final_segments)}")
    for i, (start, end) in enumerate(final_segments):
        length = end - start + 1
        print(f"       Segment {i+1}: {start}-{end} ({length} years)")
    
    # 🔧 NEW: Validate that we actually enforced minimum segment length
    validation_failures = []
    for i, (start, end) in enumerate(final_segments):
        length = end - start + 1
        if length < min_segment_length:
            validation_failures.append(f"Segment {i+1}: {start}-{end} ({length} years)")
    
    if validation_failures:
        print(f"    ❌ VALIDATION FAILED: {len(validation_failures)} segments still below minimum:")
        for failure in validation_failures:
            print(f"       {failure}")
        raise ValueError(f"Segments shorter than min length: {validation_failures}")
    else:
        print(f"    ✅ Minimum segment length validation passed")
    
    return final_segments, merge_decisions


def validate_segment_contiguity(segments: List[Tuple[int, int]]) -> bool:
    """
    Validate that segments are contiguous with no gaps or overlaps.
    
    Args:
        segments: List of (start_year, end_year) segment tuples
        
    Returns:
        True if segments are perfectly contiguous, False otherwise
        
    Raises:
        ValueError: If segments list is empty or malformed
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


def check_segment_length_feasibility(
    num_signals: int,
    domain_year_span: int,
    min_segment_length: int,
    max_segment_length: int
) -> Tuple[bool, str]:
    """
    Check if the segment length configuration is feasible given signal density.
    
    Args:
        num_signals: Number of validated signals (centroids)
        domain_year_span: Total years in domain
        min_segment_length: Minimum segment length requirement
        max_segment_length: Maximum segment length constraint
        
    Returns:
        Tuple of (is_feasible, explanation)
        
    Example:
        is_feasible, reason = check_segment_length_feasibility(10, 50, 8, 15)
        # Returns: (False, "10 signals × 8 min length = 80 years, but domain spans only 50 years")
    """
    
    min_years_needed = num_signals * min_segment_length
    
    if min_years_needed > domain_year_span:
        explanation = (
            f"Configuration impossible: {num_signals} signals × {min_segment_length} min length = "
            f"{min_years_needed} years, but domain spans only {domain_year_span} years. "
            f"Reduce min_segment_length to {domain_year_span // num_signals} or lower validation_threshold."
        )
        return False, explanation
    
    if min_segment_length >= max_segment_length:
        explanation = (
            f"Configuration invalid: min_segment_length ({min_segment_length}) >= "
            f"max_segment_length ({max_segment_length})"
        )
        return False, explanation
    
    return True, "Configuration is feasible"


def get_segment_transparency_report(
    segments: List[Tuple[int, int]], 
    transparency_data: Dict[str, Any]
) -> str:
    """
    Generate human-readable transparency report for segment decisions.
    
    Args:
        segments: List of segment tuples
        transparency_data: Metadata from create_similarity_based_segments()
        
    Returns:
        Formatted transparency report string
    """
    
    report = "=== Similarity-Based Segmentation Report ===\n"
    report += f"Generated {len(segments)} contiguous segments\n"
    report += f"Signal years: {transparency_data.get('signal_years', [])}\n\n"
    
    for i, (start, end) in enumerate(segments):
        segment_data = transparency_data.get(f"segment_{i}", {})
        signal_year = segment_data.get('signal_year', 'Unknown')
        rationale = segment_data.get('boundary_rationale', 'No rationale available')
        
        report += f"Segment {i+1}: {start}-{end}\n"
        report += f"  Signal Year: {signal_year}\n"
        report += f"  Rationale: {rationale}\n\n"
    
    if 'boundary_explanations' in transparency_data:
        report += "Boundary Decisions:\n"
        for j, explanation in enumerate(transparency_data['boundary_explanations']):
            report += f"  Boundary {j+1}: {explanation}\n"
    
    return report 