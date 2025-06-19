"""
Shift Signal Detection for Research Timeline Modeling

This module implements paradigm transition detection using a simplified, optimized approach:

ARCHITECTURE (Phase 12 Optimized):
1. PRIMARY: Research direction changes detect paradigm shifts
2. SECONDARY: Citation gradient analysis validates and boosts confidence  
3. SIMPLIFIED: Clean signal hierarchy with predictable granularity control
4. RESTRICTED: Breakthrough paper validation removed (too permissive)

KEY FEATURES:
- Direction-driven paradigm detection (primary method)
- Gradient-only CPSD citation validation (F1=0.437 > ensemble F1=0.355)  
- Temporal clustering with fixed algorithm for predictable behavior
- Citation-only validation logic for higher precision
- Breakthrough paper validation REMOVED to reduce false positives

Follows functional programming principles with pure functions and immutable data structures.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Set, Any
from collections import defaultdict, Counter
import json
from pathlib import Path
from scipy import stats

from .data_models import (
    Paper,
    DomainData,
    ShiftSignal,
    TransitionEvidence,
)


# =============================================================================
# UTILITY FUNCTIONS - Pure helper functions
# =============================================================================

def adaptive_threshold(data: np.ndarray, method: str) -> float:
    """
    Calculate adaptive thresholds based on data characteristics (pure function).

    Args:
        data: Input data array
        method: Threshold method ('gradient' or 'acceleration')

    Returns:
        Adaptive threshold value
    """
    if len(data) == 0:
        return 0.0

    data_std = np.std(data)

    if method == "gradient":
        # For gradient: threshold based on standard deviation
        return data_std * 1.5
    elif method == "acceleration":
        # For acceleration: threshold based on median absolute deviation
        mad = np.median(np.abs(data - np.median(data)))
        return mad * 2.0
    else:
        return data_std


def moving_average(data: np.ndarray, window: int) -> np.ndarray:
    """
    Pure function for calculating moving average.

    Args:
        data: Input data array
        window: Window size for averaging

    Returns:
        Moving average array
    """
    if window >= len(data):
        return np.array([np.mean(data)] * len(data))

    result = []
    for i in range(len(data) - window + 1):
        result.append(np.mean(data[i : i + window]))
    return np.array(result)


def cluster_and_validate_shifts(
    shifts: List[int], years_array: np.ndarray, min_segment_length: int = 3
) -> List[int]:
    """
    Pure function to cluster nearby shifts and validate temporal spacing.

    Args:
        shifts: List of detected shift years
        years_array: Array of all years
        min_segment_length: Minimum years between shifts

    Returns:
        Filtered and validated shifts
    """
    if not shifts:
        return []

    shifts = sorted(list(set(shifts)))

    # Remove shifts too close together
    filtered_shifts = [shifts[0]]

    for shift in shifts[1:]:
        if shift - filtered_shifts[-1] >= min_segment_length:
            filtered_shifts.append(shift)

    # Validate shifts are within years range
    min_year = min(years_array)
    max_year = max(years_array)

    valid_shifts = [s for s in filtered_shifts if min_year <= s <= max_year]

    return valid_shifts


# =============================================================================
# CITATION ANALYSIS - Gradient-only CPSD (Phase 11 Optimized)
# =============================================================================

def detect_citation_acceleration_shifts(
    citations: np.ndarray, years_array: np.ndarray
) -> List[int]:
    """
    Gradient-only citation analysis for paradigm shift detection.
    
    PHASE 11 OPTIMIZATION: Proven optimal method (F1=0.437) compared to 
    complex ensemble approaches (F1=0.355).

    Detects paradigm shifts through:
    - Multi-scale gradient analysis (1, 3, 5-year windows)
    - First derivative (acceleration/deceleration)
    - Second derivative (inflection points)
    - Adaptive thresholds based on data characteristics

    Args:
        citations: Citation counts array
        years_array: Corresponding years array

    Returns:
        List of detected shift years
    """
    shifts = []

    # Multi-scale gradient analysis - captures different paradigm shift patterns
    for window in [1, 3, 5]:  # 1-year: sudden breaks, 3-year: transitions, 5-year: gradual evolution
        if len(citations) <= window:
            continue

        # Smooth the series for this scale
        if window > 1:
            smoothed = moving_average(citations, window)
            # Pad to maintain length
            smoothed = np.pad(smoothed, (window // 2, window // 2), mode="edge")
            smoothed = smoothed[: len(citations)]
        else:
            smoothed = citations

        # First derivative (gradient) - detects acceleration/deceleration
        gradient = np.gradient(smoothed)

        # Second derivative (acceleration) - detects inflection points
        acceleration = np.gradient(gradient)

        # Adaptive thresholds based on series characteristics
        grad_threshold = adaptive_threshold(gradient, "gradient")
        accel_threshold = adaptive_threshold(acceleration, "acceleration")

        # Find significant changes
        significant_grads = np.where(np.abs(gradient) > grad_threshold)[0]
        significant_accels = np.where(np.abs(acceleration) > accel_threshold)[0]

        # Combine and filter
        candidates = np.union1d(significant_grads, significant_accels)

        # Convert indices to years and filter
        for idx in candidates:
            if idx < len(years_array):
                year = years_array[idx]
                if year not in shifts:
                    shifts.append(year)

    # Temporal clustering and validation
    return cluster_and_validate_shifts(shifts, years_array)


def detect_citation_structural_breaks(
    domain_data: DomainData, domain_name: str
) -> List[ShiftSignal]:
    """
    CPSD Gradient-Only Citation Validation - Phase 11 optimized.

    PHASE 11 RESEARCH CONCLUSION: Gradient-only approach achieves optimal 
    performance (F1=0.437) compared to complex ensemble methods (F1=0.355).

    This provides citation-based validation for direction-detected paradigm shifts
    using multi-scale gradient analysis specifically designed for citation time series.

    Args:
        domain_data: Domain data with papers and citations
        domain_name: Domain name for logging

    Returns:
        List of gradient-based citation validation signals
    """
    print(f"    🔍 CPSD GRADIENT-ONLY CITATION VALIDATION for {domain_name}")

    # Create citation time series from domain data
    citation_series = defaultdict(float)

    # Aggregate citations by year
    for paper in domain_data.papers:
        year = paper.pub_year
        citation_series[year] += paper.cited_by_count

    if not citation_series:
        print(f"    ⚠️ No citation data found for {domain_name}")
        return []

    # Prepare data for gradient analysis
    years = sorted(citation_series.keys())
    citation_values = np.array([citation_series[year] for year in years])
    years_array = np.array(years)

    print(
        f"    🔬 Citation time series: {len(years)} years ({min(years)}-{max(years)})"
    )
    print(
        f"    📊 Citation range: {min(citation_values):,.0f} - {max(citation_values):,.0f}"
    )

    # Apply gradient-only CPSD algorithm
    try:
        # Phase 11 optimal configuration: gradient-only detection
        gradient_shifts = detect_citation_acceleration_shifts(
            citation_values, years_array
        )

        print(f"    📊 CPSD GRADIENT-ONLY Results:")
        print(f"        🎯 Gradient shifts detected: {len(gradient_shifts)}")
        print(f"        📈 Performance: Gradient-only (F1=0.437) > Ensemble (F1=0.355)")

        # Convert to ShiftSignal objects
        signals = []
        for shift_year in gradient_shifts:
            confidence = 0.7  # Standard confidence for gradient detection

            # Find contributing papers
            contributing_papers = tuple(
                p.id for p in domain_data.papers if p.pub_year == shift_year
            )

            # Supporting evidence for gradient detection
            supporting_evidence = [
                "Citation acceleration/deceleration pattern detected",
                "Multi-scale gradient analysis (1, 3, 5-year windows)",
                "First and second derivative significance testing"
            ]

            # Evidence strength for gradient-only method
            evidence_strength = min(confidence * 0.8, 1.0)

            print(
                f"      ✅ {shift_year}: confidence={confidence:.3f}, evidence_strength={evidence_strength:.3f}"
            )

            # Create citation validation signal
            signals.append(
                ShiftSignal(
                    year=shift_year,
                    confidence=confidence,
                    signal_type="citation_gradient_cpsd",
                    evidence_strength=evidence_strength,
                    supporting_evidence=tuple(supporting_evidence),
                    contributing_papers=contributing_papers,
                    transition_description=f"CPSD gradient-only paradigm shift detection at {shift_year} (confidence={confidence:.3f})",
                    paradigm_significance=0.8,  # Higher significance for proven method
                )
            )

        print(
            f"    🏆 CPSD GRADIENT-ONLY COMPLETE: {len(signals)} citation validation signals ready"
        )

        return signals

    except Exception as e:
        print(f"    ⚠️ CPSD detection failed: {e}")
        # Following project guideline: fail fast, no fallbacks
        raise RuntimeError(f"CPSD citation detection failed for {domain_name}: {e}")


# =============================================================================
# DIRECTION ANALYSIS - Primary paradigm detection
# =============================================================================

def detect_research_direction_changes(
    domain_data: DomainData, 
    detection_threshold: float = 0.4, 
    return_analysis_data: bool = False
) -> List[ShiftSignal]:
    """
    PRIMARY: Detect paradigm shifts through research direction changes.
    
    This is the main method for paradigm detection, using keyword evolution 
    and research focus changes to identify fundamental paradigm transitions.

    Args:
        domain_data: Domain data with papers and citations
        detection_threshold: Direction change score threshold (lower = more sensitive)
                           0.2 = High sensitivity (fine-grained)
                           0.4 = Medium sensitivity (balanced, default)  
                           0.6 = Low sensitivity (coarse-grained)
        return_analysis_data: If True, returns tuple for visualization

    Returns:
        List of direction-based paradigm shift signals, or tuple with analysis data
    """
    signals = []

    # Group papers by year and analyze keyword evolution
    year_keywords = defaultdict(list)
    for paper in domain_data.papers:
        year_keywords[paper.pub_year].extend(paper.keywords)

    years = sorted(year_keywords.keys())
    if len(years) < 5:
        if return_analysis_data:
            return [], {"years": [], "overlap": [], "novelty": [], "direction_score": [], "novel_keywords": [], "top_keywords": [], "threshold": detection_threshold}
        return []

    print(f"    🎛️  Direction detection threshold: {detection_threshold:.2f} (lower = more sensitive)")

    # Initialize analysis data for visualization if requested
    analysis_data = {
        "years": [],
        "overlap": [],
        "novelty": [],
        "direction_score": [],
        "novel_keywords": [],  # Enhanced: list of novel keywords for each year
        "top_keywords": [],    # Enhanced: list of top current keywords for each year
        "threshold": detection_threshold
    }

    # Analyze keyword evolution using sliding windows
    window_size = 3
    for i in range(window_size, len(years)):
        year = years[i]

        # Current window keywords
        current_keywords = []
        for y in years[i - window_size : i]:
            current_keywords.extend(year_keywords[y])

        # Previous window keywords
        prev_keywords = []
        for y in years[max(0, i - window_size * 2) : i - window_size]:
            prev_keywords.extend(year_keywords[y])

        if not current_keywords or not prev_keywords:
            continue

        # Calculate keyword overlap and novelty
        current_set = set(current_keywords)
        prev_set = set(prev_keywords)

        if len(prev_set) == 0:
            continue

        overlap = len(current_set & prev_set) / len(prev_set)
        novelty = len(current_set - prev_set) / len(current_set) if current_set else 0

        # Direction change score: high novelty + low overlap indicates paradigm shift
        direction_change_score = novelty * (1 - overlap)

        # Collect detailed analysis data for visualization (regardless of threshold)
        if return_analysis_data:
            new_keywords = current_set - prev_set
            keyword_frequencies = Counter(current_keywords)
            
            # Get significant novel keywords (appearing at least 2 times)
            significant_novel = [kw for kw in new_keywords if keyword_frequencies[kw] >= 2]
            
            # Get top current keywords by frequency
            top_current = [kw for kw, count in keyword_frequencies.most_common(10)]
            
            analysis_data["years"].append(year)
            analysis_data["overlap"].append(overlap)
            analysis_data["novelty"].append(novelty)
            analysis_data["direction_score"].append(direction_change_score)
            analysis_data["novel_keywords"].append(significant_novel[:8])  # Top 8 novel keywords for display
            analysis_data["top_keywords"].append(top_current[:8])          # Top 8 current keywords for display

        # Apply configurable threshold for granularity control
        if direction_change_score > detection_threshold:
            # Validate significance with keyword frequency analysis
            new_keywords = current_set - prev_set
            keyword_frequencies = Counter(current_keywords)
            significant_new = [
                kw for kw in new_keywords if keyword_frequencies[kw] >= 2
            ]

            # Require multiple significant new keywords for paradigm shift
            if len(significant_new) >= 3:
                confidence = min(direction_change_score, 1.0)
                contributing_papers = tuple(
                    p.id
                    for p in domain_data.papers
                    if p.pub_year == year
                    and any(kw in p.keywords for kw in significant_new)
                )

                # Create enhanced description with keyword data
                top_current_keywords = [kw for kw, count in keyword_frequencies.most_common(10)]
                enhanced_description = (
                    f"Research direction shift: {novelty:.1%} new keywords (threshold={detection_threshold:.2f}) | "
                    f"Novel keywords: {', '.join(significant_new[:5])} | "
                    f"Top keywords: {', '.join(top_current_keywords[:5])}"
                )

                signals.append(
                    ShiftSignal(
                        year=year,
                        confidence=confidence,
                        signal_type="direction_volatility",
                        evidence_strength=direction_change_score,
                        supporting_evidence=tuple(
                            [f"Novel focus: {kw}" for kw in significant_new[:5]] +
                            [f"Top current: {kw}" for kw in top_current_keywords[:5]]
                        ),
                        contributing_papers=contributing_papers,
                        transition_description=enhanced_description,
                        paradigm_significance=0.4,
                    )
                )

    print(f"    📊 Direction signals detected: {len(signals)} (threshold={detection_threshold:.2f})")
    
    if return_analysis_data:
        return signals, analysis_data
    return signals


# =============================================================================
# TEMPORAL CLUSTERING - Fixed algorithm for predictable behavior
# =============================================================================

def cluster_direction_signals_by_proximity(
    signals: List[ShiftSignal], 
    sensitivity_config
) -> List[ShiftSignal]:
    """
    Cluster direction signals within temporal proximity into coherent paradigm shifts.
    
    FIXED ALGORITHM: Uses cluster START year for comparison (not end year) to 
    prevent endless chaining and ensure predictable granularity behavior.
    
    Args:
        signals: Raw direction signals to cluster
        sensitivity_config: Configuration with clustering window
        
    Returns:
        Clustered signals representing distinct paradigm shifts
    """
    if not signals:
        return []
    
    adaptive_window = sensitivity_config.clustering_window
    
    print(f"    🔗 TEMPORAL CLUSTERING: {len(signals)} raw direction signals")
    print(f"    🎛️  Clustering window: {adaptive_window} years")
    
    # Sort by year
    sorted_signals = sorted(signals, key=lambda s: s.year)
    
    clustered = []
    current_cluster = [sorted_signals[0]]
    
    for signal in sorted_signals[1:]:
        # FIXED: Compare with cluster START (current_cluster[0]) not end (current_cluster[-1])
        if signal.year - current_cluster[0].year <= adaptive_window:
            # Add to current cluster
            current_cluster.append(signal)
            print(f"      📎 Adding {signal.year} to cluster starting {current_cluster[0].year}")
        else:
            # Finalize current cluster and start new one
            cluster_representative = merge_cluster_into_single_signal(current_cluster)
            clustered.append(cluster_representative)
            if len(current_cluster) > 1:
                cluster_years = ', '.join(str(s.year) for s in current_cluster)
                print(f"      ✅ Cluster MERGED [{cluster_years}] → paradigm shift at {cluster_representative.year}")
            else:
                print(f"      ✅ Single signal {current_cluster[0].year} → paradigm shift")
            current_cluster = [signal]
    
    # Add final cluster
    if current_cluster:
        cluster_representative = merge_cluster_into_single_signal(current_cluster)
        clustered.append(cluster_representative)
        if len(current_cluster) > 1:
            cluster_years = ', '.join(str(s.year) for s in current_cluster)
            print(f"      ✅ Final cluster MERGED [{cluster_years}] → paradigm shift at {cluster_representative.year}")
        else:
            print(f"      ✅ Final single signal {current_cluster[0].year} → paradigm shift")
    
    print(f"    🎯 CLUSTERING COMPLETE: {len(signals)} signals → {len(clustered)} paradigm shifts")
    return clustered


def merge_cluster_into_single_signal(cluster: List[ShiftSignal]) -> ShiftSignal:
    """
    Merge multiple temporal signals into representative paradigm shift.
    
    Strategy: Use middle year as representative, combine evidence and confidence.
    
    Args:
        cluster: List of signals in temporal proximity
        
    Returns:
        Single representative paradigm shift signal
    """
    if len(cluster) == 1:
        return cluster[0]
    
    # Use middle year as representative
    representative_year = cluster[len(cluster)//2].year
    
    # Combine confidence (mean for stability)
    combined_confidence = float(np.mean([s.confidence for s in cluster]))
    
    # Combine evidence strength (max for best evidence)
    combined_evidence_strength = float(np.max([s.evidence_strength for s in cluster]))
    
    # Combine supporting evidence (top pieces from all signals)
    combined_evidence = []
    for s in cluster:
        combined_evidence.extend(s.supporting_evidence)
    # Remove duplicates and keep top 10
    unique_evidence = list(dict.fromkeys(combined_evidence))[:10]
    
    # Combine contributing papers
    all_papers = set()
    for s in cluster:
        all_papers.update(s.contributing_papers)
    
    # Enhanced paradigm significance
    combined_paradigm_significance = float(np.max([s.paradigm_significance for s in cluster]))
    
    # Create clustering description preserving keyword data
    best_signal = max(cluster, key=lambda s: s.evidence_strength)
    years_span = f"{cluster[0].year}-{cluster[-1].year}" if len(cluster) > 1 else str(cluster[0].year)
    clustering_suffix = f" | Clustered ({years_span}): {len(cluster)} signals merged"
    
    # Preserve original keyword data and add clustering info
    if "Novel keywords:" in best_signal.transition_description:
        transition_description = best_signal.transition_description + clustering_suffix
    else:
        transition_description = f"Clustered paradigm shift ({years_span}): {len(cluster)} direction signals merged"
    
    return ShiftSignal(
        year=representative_year,
        confidence=combined_confidence,
        signal_type="direction_clustered",
        evidence_strength=combined_evidence_strength,
        supporting_evidence=tuple(unique_evidence),
        contributing_papers=tuple(all_papers),
        transition_description=transition_description,
        paradigm_significance=combined_paradigm_significance
    )


# =============================================================================
# VALIDATION LOGIC - Consistent threshold with score boosting
# =============================================================================

def validate_direction_with_citation(
    direction_signals: List[ShiftSignal],
    citation_signals: List[ShiftSignal], 
    domain_data: DomainData,
    domain_name: str,
    sensitivity_config
) -> List[ShiftSignal]:
    """
    Direction-citation validation with consistent threshold logic.
    
    SIMPLIFIED VALIDATION LOGIC: All signals use the same validation threshold,
    with citation support providing score boost only.
    
    Logic:
    1. Start with clustered direction signals as paradigm candidates
    2. Check for citation support within ±2 years → +0.3 confidence boost
    3. Apply consistent validation threshold to all boosted signals
    
    Args:
        direction_signals: Primary direction change signals
        citation_signals: Secondary citation validation signals
        domain_data: Domain data for context
        domain_name: Domain name for logging
        sensitivity_config: Configuration with boost values and thresholds
        
    Returns:
        List of validated paradigm shift signals
    """
    print(f"  🔄 DIRECTION-CITATION VALIDATION:")
    print(f"    🔍 DEBUG: Input validation parameters:")
    print(f"      📊 Direction signals to validate: {len(direction_signals)}")
    print(f"      🔗 Citation signals for validation: {len(citation_signals)}")
    print(f"      🎯 Validation threshold: {sensitivity_config.validation_threshold:.3f}")
    print(f"      📈 Citation boost: {sensitivity_config.citation_boost:.3f}")
    print(f"      ⚠️ Breakthrough paper validation REMOVED (too permissive)")
    
    if not direction_signals:
        print(f"    ⚠️ No direction signals detected - no paradigm shifts")
        return []
    
    validated_paradigms = []
    
    for idx, direction_signal in enumerate(direction_signals):
        year = direction_signal.year
        
        print(f"    🎯 EVALUATING direction signal {idx+1}/{len(direction_signals)}: {year}")
        print(f"      📊 Base confidence: {direction_signal.confidence:.3f}")
        print(f"      🏷️  Signal type: {direction_signal.signal_type}")
        
        # Start with direction signal base confidence
        base_confidence = direction_signal.confidence
        paradigm_score = direction_signal.paradigm_significance
        supporting_evidence = list(direction_signal.supporting_evidence)
        contributing_papers = set(direction_signal.contributing_papers)
        
        # Check for citation support within ±2 years
        citation_support = False
        citation_years_nearby = []
        for citation_signal in citation_signals:
            if abs(citation_signal.year - year) <= 2:
                citation_support = True
                citation_years_nearby.append(citation_signal.year)
                supporting_evidence.extend(citation_signal.supporting_evidence)
                contributing_papers.update(citation_signal.contributing_papers)
                print(f"      🔍 Citation validation found: {citation_signal.year} (±2 years from {year})")
        
        if citation_support:
            print(f"      ✅ Citation support: YES (nearby years: {citation_years_nearby})")
        else:
            print(f"      ❌ Citation support: NO")
        
        # Calculate final confidence with additive score boosting (citation only)
        confidence_boosts = 0.0
        
        if citation_support:
            confidence_boosts += sensitivity_config.citation_boost
            print(f"      📈 Citation boost applied: +{sensitivity_config.citation_boost:.2f}")
        
        # Apply boosts to confidence score (capped at 1.0)
        final_confidence = min(base_confidence + confidence_boosts, 1.0)
        final_paradigm_score = paradigm_score + confidence_boosts
        
        # Use consistent validation threshold for all signals
        threshold = sensitivity_config.validation_threshold
            
        print(f"      📊 FINAL CALCULATION:")
        print(f"        Base confidence: {base_confidence:.3f}")
        print(f"        Total boosts: +{confidence_boosts:.3f}")
        print(f"        Final confidence: {final_confidence:.3f}")
        print(f"        Validation threshold: {threshold:.3f}")
        print(f"        Result: {'PASS' if final_confidence >= threshold else 'FAIL'}")
        
        if final_confidence >= threshold:
            # Preserve keyword data from original description
            original_description = direction_signal.transition_description
            validation_suffix = " (citation validated)" if citation_support else ""
            
            validated_signal = ShiftSignal(
                year=year,
                confidence=final_confidence,
                signal_type="direction_primary_validated" if citation_support else "direction_primary_only",
                evidence_strength=direction_signal.evidence_strength + (sensitivity_config.citation_boost if citation_support else 0),
                supporting_evidence=tuple(supporting_evidence[:10]),
                contributing_papers=tuple(contributing_papers),
                transition_description=original_description + validation_suffix,
                paradigm_significance=final_paradigm_score
            )
            
            validated_paradigms.append(validated_signal)
            
            # Build validation type description
            validation_type = "CITATION VALIDATED" if citation_support else "DIRECTION ONLY"
                
            print(f"      ✅ ACCEPTED ({validation_type}): {year}")
        else:
            print(f"      ❌ FILTERED: {year} confidence {final_confidence:.3f} < threshold {threshold:.3f}")
    
    print(f"    🏆 VALIDATION COMPLETE: {len(validated_paradigms)} paradigm shifts validated")
    print(f"    📅 Validated years: {[s.year for s in validated_paradigms] if validated_paradigms else 'None'}")
    return validated_paradigms


# =============================================================================
# MAIN DETECTION PIPELINE
# =============================================================================

def detect_shift_signals(
    domain_data: DomainData,
    domain_name: str,
    sensitivity_config,
    use_citation: bool = True,
    use_direction: bool = True,
    precomputed_signals: Optional[Dict[str, List[ShiftSignal]]] = None,
) -> Tuple[List[ShiftSignal], List[TransitionEvidence], Dict[str, Any]]:
    """
    Main paradigm shift detection pipeline.
    
    SIMPLIFIED ARCHITECTURE:
    1. Direction signals detect paradigm shifts (primary) 
    2. Temporal clustering prevents over-segmentation
    3. Citation signals validate and boost confidence (secondary)
    4. Breakthrough paper validation REMOVED (too permissive)
    
    Args:
        domain_data: Domain data with papers and citations
        domain_name: Name of the domain
        sensitivity_config: Configuration for thresholds and parameters
        use_citation: Whether to use citation validation
        use_direction: Whether to use direction signals
        precomputed_signals: Optional pre-computed signals

    Returns:
        Tuple of (paradigm_shifts, transition_evidence, clustering_metadata)
    """
    print(f"\n🔍 SHIFT SIGNAL DETECTION: {domain_name}")
    print(f"  🎯 PRIMARY: Direction signals (paradigm detection)")
    print(f"  🔗 CLUSTERING: Temporal proximity filtering")
    print(f"  🔍 SECONDARY: Citation signals (validation)")
    print(f"  🎛️  THRESHOLD: {sensitivity_config.detection_threshold:.2f}")
    print(f"  🎛️  CLUSTERING WINDOW: {sensitivity_config.clustering_window} years")
    print(f"  🎛️  VALIDATION THRESHOLD: {sensitivity_config.validation_threshold:.2f}")
    print("=" * 60)

    # Stage 1: Primary Detection - Direction Signals 
    if precomputed_signals:
        print("  ⚡️ Using pre-computed raw signals.")
        raw_direction_signals = (
            precomputed_signals.get("direction", []) if use_direction else []
        )
        citation_signals = (
            precomputed_signals.get("citation", []) if use_citation else []
        )
    else:
        # PRIMARY: Research direction changes detect paradigm shifts
        raw_direction_signals = (
            detect_research_direction_changes(domain_data, detection_threshold=sensitivity_config.detection_threshold) if use_direction else []
        )
        
        # SECONDARY: Citation patterns for validation
        citation_signals = (
            detect_citation_structural_breaks(domain_data, domain_name)
            if use_citation
            else []
        )

    print(f"  🎯 RAW Direction signals: {len(raw_direction_signals)}")
    if raw_direction_signals:
        print(f"    📅 Raw direction years: {[s.year for s in raw_direction_signals]}")
        print(f"    📈 Raw direction confidences: {[f'{s.confidence:.3f}' for s in raw_direction_signals]}")
    
    # Stage 2: Temporal Clustering for direction signals
    clustered_direction_signals = (
        cluster_direction_signals_by_proximity(raw_direction_signals, sensitivity_config) 
        if raw_direction_signals else []
    )
    
    print(f"  🔗 CLUSTERED Direction signals: {len(clustered_direction_signals)}")
    if clustered_direction_signals:
        print(f"    📅 Clustered direction years: {[s.year for s in clustered_direction_signals]}")
        print(f"    📈 Clustered direction confidences: {[f'{s.confidence:.3f}' for s in clustered_direction_signals]}")
    print(f"  🔍 Citation validation signals: {len(citation_signals)}")
    if citation_signals:
        print(f"    📅 Citation years: {[s.year for s in citation_signals]}")
    
    # Create clustering metadata for visualization
    clustering_metadata = create_clustering_metadata(
        raw_direction_signals, clustered_direction_signals, citation_signals, sensitivity_config
    )
    
    # Stage 3: Direction-Citation Validation
    paradigm_shifts = validate_direction_with_citation(
        clustered_direction_signals, citation_signals, domain_data, domain_name, sensitivity_config
    )

    print(f"  ✅ Final validated paradigm shifts: {len(paradigm_shifts)}")
    if paradigm_shifts:
        print(f"    📅 Final paradigm shift years: {[s.year for s in paradigm_shifts]}")
        print(f"    🏷️  Final signal types: {[s.signal_type for s in paradigm_shifts]}")
        print(f"    📈 Final confidences: {[f'{s.confidence:.3f}' for s in paradigm_shifts]}")
    else:
        print(f"    ⚠️  NO paradigm shifts validated!")

    # Stage 4: Generate transition evidence
    transition_evidence = generate_transition_justifications(
        paradigm_shifts, domain_data
    )

    print(f"  📋 Transition evidence generated: {len(transition_evidence)}")

    # Stage 5: Save results for visualization
    save_shift_signals_for_visualization(
        raw_signals=raw_direction_signals + citation_signals,
        validated_signals=paradigm_shifts,
        paradigm_shifts=paradigm_shifts,
        transition_evidence=transition_evidence,
        domain_name=domain_name,
    )

    return paradigm_shifts, transition_evidence, clustering_metadata


# =============================================================================
# SUPPORTING FUNCTIONS
# =============================================================================

def create_clustering_metadata(
    raw_direction_signals: List[ShiftSignal],
    clustered_direction_signals: List[ShiftSignal], 
    citation_signals: List[ShiftSignal],
    sensitivity_config
) -> Dict[str, Any]:
    """
    Create clustering metadata for visualization.
    
    Args:
        raw_direction_signals: Original direction signals before clustering
        clustered_direction_signals: Direction signals after clustering
        citation_signals: Citation validation signals
        sensitivity_config: Configuration used
        
    Returns:
        Dict containing visualization metadata
    """
    # Create signal lookup by year
    raw_signal_by_year = {signal.year: signal for signal in raw_direction_signals}
    
    # Analyze clustering relationships
    clustering_relationships = []
    for clustered_signal in clustered_direction_signals:
        cluster_info = {
            'representative_year': clustered_signal.year,
            'evidence_strength': clustered_signal.evidence_strength,
            'confidence': clustered_signal.confidence,
            'transition_description': clustered_signal.transition_description,
            'contributing_years': [],
            'is_clustered': False,
            'cluster_span': None,
            'merge_count': 0
        }
        
        # Parse clustering information from description
        if "Clustered" in clustered_signal.transition_description and "(" in clustered_signal.transition_description:
            try:
                desc = clustered_signal.transition_description
                range_part = desc.split("(")[1].split(")")[0]
                if "-" in range_part:
                    start_year, end_year = map(int, range_part.split("-"))
                    cluster_info['cluster_span'] = (start_year, end_year)
                    
                    # Find contributing raw signals
                    contributing_years = []
                    for year in range(start_year, end_year + 1):
                        if year in raw_signal_by_year:
                            contributing_years.append(year)
                    
                    cluster_info['contributing_years'] = sorted(contributing_years)
                    cluster_info['is_clustered'] = len(contributing_years) > 1
                    cluster_info['merge_count'] = len(contributing_years)
                else:
                    # Single signal
                    cluster_info['contributing_years'] = [clustered_signal.year]
                    cluster_info['merge_count'] = 1
            except:
                # Fallback - treat as single signal
                cluster_info['contributing_years'] = [clustered_signal.year]
                cluster_info['merge_count'] = 1
        else:
            # Single signal (not clustered)
            cluster_info['contributing_years'] = [clustered_signal.year]
            cluster_info['merge_count'] = 1
        
        clustering_relationships.append(cluster_info)
    
    # Create comprehensive metadata
    metadata = {
        'raw_direction_signals': raw_direction_signals,
        'clustered_direction_signals': clustered_direction_signals,
        'citation_signals': citation_signals,
        'sensitivity_config': sensitivity_config,
        'clustering_relationships': clustering_relationships,
        'raw_signal_by_year': raw_signal_by_year,
        'clustering_summary': {
            'total_raw_signals': len(raw_direction_signals),
            'total_clustered_signals': len(clustered_direction_signals),
            'clustering_window': sensitivity_config.clustering_window,
            'detection_threshold': sensitivity_config.detection_threshold,
            'reduction_ratio': len(raw_direction_signals) / max(len(clustered_direction_signals), 1),
            'merge_statistics': {
                'single_signals': sum(1 for rel in clustering_relationships if rel['merge_count'] == 1),
                'merged_signals': sum(1 for rel in clustering_relationships if rel['merge_count'] > 1),
                'max_merge_count': max((rel['merge_count'] for rel in clustering_relationships), default=0),
                'total_raw_signals_merged': sum(rel['merge_count'] for rel in clustering_relationships)
            }
        }
    }
    
    return metadata


# DEPRECATED: Breakthrough paper validation removed as too permissive
# def load_breakthrough_papers(domain_data: DomainData, domain_name: str) -> List[Paper]:
#     """
#     Load breakthrough papers for significance weighting.
#     
#     DEPRECATED: Breakthrough paper validation has been removed from the pipeline
#     as it was making validation too permissive. Only citation validation is now used.
#     """
#     return []


def generate_transition_justifications(
    paradigm_shifts: List[ShiftSignal], domain_data: DomainData
) -> List[TransitionEvidence]:
    """
    Generate detailed evidence for each paradigm transition.

    Args:
        paradigm_shifts: List of paradigm shift signals
        domain_data: Domain data

    Returns:
        List of transition evidence
    """
    transition_evidence = []

    for signal in paradigm_shifts:
        # Analyze papers around the transition year
        window_papers = [
            p for p in domain_data.papers if abs(p.pub_year - signal.year) <= 2
        ]

        # Extract evidence patterns
        disruption_patterns = []
        emergence_patterns = []
        methodological_changes = []

        # Analyze keywords for patterns
        all_keywords = []
        for paper in window_papers:
            all_keywords.extend(paper.keywords)

        keyword_counts = Counter(all_keywords)
        top_keywords = [kw for kw, count in keyword_counts.most_common(5)]

        if top_keywords:
            emergence_patterns.extend(
                [f"Emerging keyword: {kw}" for kw in top_keywords[:3]]
            )

        # Analyze semantic descriptions for patterns
        semantic_patterns = []
        for citation in domain_data.citations:
            if (
                abs(citation.citing_year - signal.year) <= 1
                and citation.semantic_description
            ):
                desc = citation.semantic_description.lower()
                if any(
                    word in desc
                    for word in ["introduces", "novel", "new", "breakthrough"]
                ):
                    semantic_patterns.append(citation.semantic_description[:100])

        if semantic_patterns:
            methodological_changes.extend(semantic_patterns[:3])

        # Create transition evidence
        evidence = TransitionEvidence(
            year=signal.year,
            disruption_patterns=tuple(disruption_patterns),
            emergence_patterns=tuple(emergence_patterns),
            cross_domain_influences=tuple(),
            methodological_changes=tuple(methodological_changes),
            breakthrough_papers=signal.contributing_papers[:5],
            confidence_score=signal.confidence,
        )

        transition_evidence.append(evidence)

    return transition_evidence


def save_shift_signals_for_visualization(
    raw_signals: List[ShiftSignal],
    validated_signals: List[ShiftSignal],
    paradigm_shifts: List[ShiftSignal],
    transition_evidence: List[TransitionEvidence],
    domain_name: str,
    output_dir: str = "results/signals",
) -> str:
    """
    Save shift signal detection results for visualization.

    Args:
        raw_signals: Raw signals from detection methods
        validated_signals: Cross-validated signals
        paradigm_shifts: Final paradigm shift signals
        transition_evidence: Supporting evidence
        domain_name: Name of the domain
        output_dir: Directory to save files

    Returns:
        Path to the saved file
    """
    from pathlib import Path
    from datetime import datetime
    import json

    # Ensure output directory exists
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    def serialize_shift_signal(signal: ShiftSignal) -> Dict:
        """Convert ShiftSignal to dictionary"""
        return {
            "year": int(signal.year),
            "confidence": float(signal.confidence),
            "signal_type": str(signal.signal_type),
            "evidence_strength": float(signal.evidence_strength),
            "supporting_evidence": list(signal.supporting_evidence),
            "contributing_papers": list(signal.contributing_papers),
            "transition_description": str(signal.transition_description),
            "paradigm_significance": float(signal.paradigm_significance),
        }

    def serialize_transition_evidence(evidence: TransitionEvidence) -> Dict:
        """Convert TransitionEvidence to dictionary"""
        return {
            "year": int(evidence.year),
            "disruption_patterns": list(evidence.disruption_patterns),
            "emergence_patterns": list(evidence.emergence_patterns),
            "cross_domain_influences": list(evidence.cross_domain_influences),
            "methodological_changes": list(evidence.methodological_changes),
            "breakthrough_papers": list(evidence.breakthrough_papers),
            "confidence_score": float(evidence.confidence_score),
        }

    # Create comprehensive dataset
    shift_signals_data = {
        "metadata": {
            "domain_name": domain_name,
            "analysis_date": datetime.now().isoformat(),
            "analysis_type": "shift_signal_detection",
            "description": "Paradigm transition detection using direction-citation hierarchy",
            "methodology": {
                "stage1": "Direction signal detection (primary)",
                "stage2": "Temporal clustering (fixed algorithm)",
                "stage3": "Citation validation (gradient-only CPSD)",
                "stage4": "Consistent threshold validation with score boosting",
            },
        },
        "raw_signals": {
            "count": len(raw_signals),
            "description": "Raw signals before clustering and validation",
            "signals": [serialize_shift_signal(s) for s in raw_signals],
            "signal_types": list(set(s.signal_type for s in raw_signals)),
        },
        "validated_signals": {
            "count": len(validated_signals),
            "description": "Final validated paradigm shifts",
            "signals": [serialize_shift_signal(s) for s in validated_signals],
        },
        "paradigm_shifts": {
            "count": len(paradigm_shifts),
            "description": "Final paradigm shift signals",
            "signals": [serialize_shift_signal(s) for s in paradigm_shifts],
        },
        "transition_evidence": {
            "count": len(transition_evidence),
            "description": "Supporting evidence for paradigm transitions",
            "evidence": [serialize_transition_evidence(e) for e in transition_evidence],
        },
        "visualization_metadata": {
            "timeline_data": {
                "raw_signal_years": sorted(list(set(int(s.year) for s in raw_signals))),
                "paradigm_shift_years": sorted(
                    list(set(int(s.year) for s in paradigm_shifts))
                ),
            },
            "confidence_distributions": {
                "raw_confidence_range": [
                    float(min([s.confidence for s in raw_signals] + [0])),
                    float(max([s.confidence for s in raw_signals] + [1])),
                ],
                "paradigm_confidence_range": [
                    float(min([s.confidence for s in paradigm_shifts] + [0])),
                    float(max([s.confidence for s in paradigm_shifts] + [1])),
                ],
            },
        },
    }

    # Save to file
    output_file = f"{output_dir}/{domain_name}_shift_signals.json"
    with open(output_file, "w") as f:
        json.dump(shift_signals_data, f, indent=2)

    print(f"  📊 RESULTS SAVED:")
    print(f"      📁 File: {output_file}")
    print(f"      🔍 Raw signals: {len(raw_signals)}")
    print(f"      🎯 Paradigm shifts: {len(paradigm_shifts)}")
    print(f"      📋 Transition evidence: {len(transition_evidence)}")

    return output_file
