"""
Shift Signal Detection for Research Timeline Modeling

This module implements paradigm transition detection using research direction changes
and citation analysis validation.

Key functionality:
- Research direction change detection
- Citation gradient analysis for validation
- Temporal clustering with configurable granularity control
- Signal validation through citation support

Implements functional programming principles with pure functions and immutable data structures.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Set, Any
from collections import defaultdict, Counter
import json
from pathlib import Path
from scipy import stats

from ..data.models import (
    Paper,
    DomainData,
    ShiftSignal,
    TransitionEvidence,
)
from ..utils.filtering import filter_domain_keywords_conservative
from ..utils.logging import get_logger


# =============================================================================
# UTILITY FUNCTIONS - Pure helper functions
# =============================================================================

def citation_adaptive_threshold(data: np.ndarray, method: str) -> float:
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
# CITATION ANALYSIS - Gradient-based shift detection
# =============================================================================

def detect_citation_acceleration_shifts(
    citations: np.ndarray, years_array: np.ndarray, citation_scales: List[int] = None
) -> List[Tuple[int, float]]:
    """
    Citation gradient analysis for paradigm shift detection.

    Uses multi-scale gradient analysis with first and second derivatives
    to detect acceleration/deceleration and inflection points in citation patterns.
    Returns confidence scores based on gradient strength.

    Args:
        citations: Citation counts array
        years_array: Corresponding years array
        citation_scales: List of scales for multi-scale gradient analysis

    Returns:
        List of tuples (shift_year, confidence_score)
    """
    shift_confidences = {}  # year -> confidence mapping

    # Multi-scale gradient analysis - captures different paradigm shift patterns
    if citation_scales is None:
        citation_scales = [1, 3, 5]
    for window in citation_scales:  # 1-year: sudden breaks, 3-year: transitions, 5-year: gradual evolution
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
        grad_threshold = citation_adaptive_threshold(gradient, "gradient")
        accel_threshold = citation_adaptive_threshold(acceleration, "acceleration")

        # Find significant changes with confidence calculation
        significant_grads = np.where(np.abs(gradient) > grad_threshold)[0]
        significant_accels = np.where(np.abs(acceleration) > accel_threshold)[0]

        # Calculate confidence for gradient signals
        for idx in significant_grads:
            if idx < len(years_array):
                year = years_array[idx]
                # Confidence based on how much the gradient exceeds threshold
                confidence = min(np.abs(gradient[idx]) / grad_threshold, 2.0) / 2.0  # Normalize to [0, 1]
                confidence = max(0.3, min(confidence, 0.95))  # Clamp to reasonable range
                
                # Take maximum confidence if year already detected
                if year in shift_confidences:
                    shift_confidences[year] = max(shift_confidences[year], confidence)
                else:
                    shift_confidences[year] = confidence

        # Calculate confidence for acceleration signals
        for idx in significant_accels:
            if idx < len(years_array):
                year = years_array[idx]
                # Confidence based on how much the acceleration exceeds threshold
                confidence = min(np.abs(acceleration[idx]) / accel_threshold, 2.0) / 2.0  # Normalize to [0, 1]
                confidence = max(0.3, min(confidence, 0.95))  # Clamp to reasonable range
                
                # Take maximum confidence if year already detected
                if year in shift_confidences:
                    shift_confidences[year] = max(shift_confidences[year], confidence)
                else:
                    shift_confidences[year] = confidence

    # Convert to list of tuples and apply temporal clustering
    shifts_with_confidence = list(shift_confidences.items())
    
    # Apply temporal clustering but preserve confidence scores
    clustered_years = cluster_and_validate_shifts([year for year, _ in shifts_with_confidence], years_array)
    
    # Return only the clustered years with their confidence scores
    result = []
    for year in clustered_years:
        if year in shift_confidences:
            result.append((year, shift_confidences[year]))
    
    return result


def detect_citation_structural_breaks(
    domain_data: DomainData, domain_name: str, algorithm_config = None, verbose: bool = False
) -> List[ShiftSignal]:
    """
    Citation-based paradigm shift validation using gradient analysis.

    Provides citation-based validation for direction-detected paradigm shifts
    using multi-scale gradient analysis for citation time series.

    Args:
        domain_data: Domain data with papers and citations
        domain_name: Domain name for logging
        algorithm_config: Algorithm configuration including citation analysis scales
        verbose: Enable verbose logging

    Returns:
        List of gradient-based citation validation signals
    """
    logger = get_logger(__name__, verbose)
    
    # Create citation time series from domain data
    citation_series = defaultdict(float)

    # Aggregate citations by year
    for paper in domain_data.papers:
        year = paper.pub_year
        citation_series[year] += paper.cited_by_count

    if not citation_series:
        logger.warning(f"No citation data found for {domain_name}")
        return []

    # Prepare data for gradient analysis
    years = sorted(citation_series.keys())
    citation_values = np.array([citation_series[year] for year in years])
    years_array = np.array(years)

    # Get shifts with calculated confidence scores
    citation_scales = algorithm_config.citation_analysis_scales if algorithm_config else [1, 3, 5]
    gradient_shifts_with_confidence = detect_citation_acceleration_shifts(
        citation_values, years_array, citation_scales
    )
    
    # Convert to ShiftSignal objects
    signals = []
    for shift_year, confidence in gradient_shifts_with_confidence:
        # Find contributing papers
        contributing_papers = tuple(
            p.id for p in domain_data.papers if p.pub_year == shift_year
        )

        # Supporting evidence for gradient detection
        supporting_evidence = [
            "Citation acceleration/deceleration pattern detected",
            "Multi-scale gradient analysis (1, 3, 5-year windows)",
            "First and second derivative significance testing",
            f"Gradient strength-based confidence: {confidence:.3f}"
        ]

        # Evidence strength for gradient-only method
        evidence_strength = min(confidence * 0.8, 1.0)

        # Create citation validation signal
        signals.append(
            ShiftSignal(
                year=shift_year,
                confidence=confidence,  # Now using calculated confidence
                signal_type="citation_gradient_cpsd",
                evidence_strength=evidence_strength,
                supporting_evidence=tuple(supporting_evidence),
                contributing_papers=contributing_papers,
                transition_description=f"(confidence={confidence:.3f})",
                paradigm_significance=min(0.8 * confidence, 0.9),  # Scale paradigm significance with confidence
            )
        )

    logger.info(f"Citation signals detected: {len(signals)}")
    for signal in signals[:5]:
        logger.debug(f"    {signal.year} {signal.transition_description}")
    if len(signals) > 5:
        logger.debug(f"    ... {len(signals) - 5} more signals")

    return signals


# =============================================================================
# DIRECTION ANALYSIS - Primary paradigm detection
# =============================================================================

def detect_research_direction_changes(
    domain_data: DomainData, 
    algorithm_config,
    return_analysis_data: bool = False,
    verbose: bool = False
) -> List[ShiftSignal]:
    """
    Detect paradigm shifts through research direction changes.
    
    Main method for paradigm detection, using keyword evolution and research focus 
    changes to identify fundamental paradigm transitions. Includes keyword filtering
    to reduce noise while preserving genuine paradigm signals.

    Args:
        domain_data: Domain data with papers and citations
        algorithm_config: Algorithm configuration including filtering parameters
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
    
    # Extract thresholds from algorithm config  
    detection_threshold = algorithm_config.direction_threshold
    
    if len(years) < 5:
        if return_analysis_data:
            return [], {"years": [], "overlap": [], "novelty": [], "direction_score": [], "novel_keywords": [], "top_keywords": [], "threshold": detection_threshold}
        return []
    # Initialize analysis data for visualization if requested
    analysis_data = {
        "years": [],
        "overlap": [],
        "novelty": [],
        "direction_score": [],
        "novel_keywords": [],  # Enhanced: list of novel keywords for each year
        "top_keywords": [],    # Enhanced: list of top current keywords for each year
        "threshold": detection_threshold,
        "filtering_enabled": algorithm_config.keyword_filtering_enabled,
        "filtering_activity": []  # Enhanced: detailed filtering activity for visualization
    }

    # Analyze keyword evolution using sliding windows
    window_size = algorithm_config.direction_window_size
    for i in range(window_size, len(years)):
        year = years[i]

        # Current window keywords
        current_keywords = []
        current_papers = []
        for y in years[i - window_size : i]:
            current_keywords.extend(year_keywords[y])
            current_papers.extend([p for p in domain_data.papers if p.pub_year == y])

        # Previous window keywords  
        prev_keywords = []
        prev_papers = []
        for y in years[max(0, i - window_size * 2) : i - window_size]:
            prev_keywords.extend(year_keywords[y])
            prev_papers.extend([p for p in domain_data.papers if p.pub_year == y])
        
        # IMPROVEMENT-001: Apply conservative keyword filtering
        if algorithm_config.keyword_filtering_enabled:
            filtered_current, current_rationale = filter_domain_keywords_conservative(
                current_keywords, current_papers, algorithm_config, domain_data.domain_name
            )
            filtered_prev, prev_rationale = filter_domain_keywords_conservative(
                prev_keywords, prev_papers, algorithm_config, domain_data.domain_name
            )
            
            # Collect filtering activity for visualization
            if return_analysis_data:
                original_current_count = len(current_keywords)
                filtered_current_count = len(filtered_current) 
                
                # Store filtering activity data
                filtering_activity = {
                    "year": year,
                    "original_count": original_current_count,
                    "filtered_count": filtered_current_count,
                    "retention_rate": filtered_current_count / original_current_count if original_current_count > 0 else 1.0,
                    "removed_count": original_current_count - filtered_current_count,
                    "rationale": current_rationale
                }
                analysis_data["filtering_activity"].append(filtering_activity)
            
            # Use filtered keywords for analysis
            current_keywords = filtered_current
            prev_keywords = filtered_prev

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
                kw for kw in new_keywords if keyword_frequencies[kw] >= algorithm_config.keyword_min_frequency
            ]

            # Require multiple significant new keywords for paradigm shift (configurable)
            if len(significant_new) >= algorithm_config.min_significant_keywords:
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
                    f"{novelty:.1%} new keywords (threshold={detection_threshold:.2f}) | "
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

    logger = get_logger(__name__, verbose)
    logger.info(f"Direction signals detected: {len(signals)} (threshold={detection_threshold:.2f})")
    for signal in signals[:5]:
        logger.debug(f"    {signal.year} (confidence={signal.confidence:.2f})")
    if len(signals) > 5:
        logger.debug(f"    ... {len(signals) - 5} more signals")
    
    if return_analysis_data:
        return signals, analysis_data
    return signals

# =============================================================================
# VALIDATION LOGIC - Consistent threshold with score boosting
# =============================================================================

def validate_direction_with_citation(
    direction_signals: List[ShiftSignal],
    citation_signals: List[ShiftSignal], 
    algorithm_config,
    verbose: bool = False
) -> List[ShiftSignal]:
    """
    Simplified direction-citation validation with unified logic.
    
    SIMPLIFIED VALIDATION LOGIC: 
    - Single validation path for all signals
    - Pure functions for decision transparency
    - Configurable parameters (no hardcoded values)
    - Streamlined logging for clarity
    
    Args:
        direction_signals: Primary direction change signals
        citation_signals: Secondary citation validation signals
        domain_data: Domain data for context
        domain_name: Domain name for logging
        algorithm_config: Comprehensive algorithm configuration with all validation parameters
        verbose: Enable verbose logging
        
    Returns:
        List of validated paradigm shift signals
    """
    logger = get_logger(__name__, verbose)
    logger.info("Validating direction signals with citation signals:")
    if not direction_signals:
        logger.info("No direction signals to validate")
        return [], []
    
    validated_paradigms = []
    validation_summary = {'accepted': 0, 'rejected': 0, 'citation_supported': 0}
    
    # Get citation support window (configurable)
    citation_window = algorithm_config.citation_support_window
    citation_boost_rate_rate = algorithm_config.citation_boost_rate
    
    results = []
    # Process each direction signal through simplified validation
    for direction_signal in direction_signals:
        year = direction_signal.year
        base_confidence = direction_signal.confidence
        
        # Step 1: Analyze citation support (pure function)
        citation_support = False
        supporting_citations = []
        
        for citation_signal in citation_signals:
            if abs(citation_signal.year - year) <= citation_window:
                citation_support = True
                supporting_citations.append(citation_signal)
        
        # Step 2: Calculate confidence boost (pure function) - 50% of base confidence
        confidence_boost = (citation_boost_rate_rate * base_confidence) if citation_support else 0.0
        
        # Step 3: Compute final confidence (pure function)
        final_confidence = min(base_confidence + confidence_boost, 1.0)
        
        # Step 4: Apply validation threshold (pure function)
        is_valid = final_confidence >= algorithm_config.validation_threshold
        
        # Step 5: Create validated signal if accepted
        if is_valid:
            # Combine evidence from original signal and citations
            combined_evidence = list(direction_signal.supporting_evidence)
            for citation in supporting_citations:
                combined_evidence.extend(citation.supporting_evidence)
            
            # Determine signal type and description
            if citation_support:
                signal_type = "direction_primary_validated"
                validation_suffix = " (citation validated)"
                validation_summary['citation_supported'] += 1
            else:
                signal_type = "direction_primary_only" 
                validation_suffix = " (direction only)"
            
            validated_signal = ShiftSignal(
                year=year,
                confidence=final_confidence,
                signal_type=signal_type,
                evidence_strength=direction_signal.evidence_strength + confidence_boost,
                supporting_evidence=tuple(combined_evidence[:10]),  # Keep top 10 pieces
                contributing_papers=direction_signal.contributing_papers,
                transition_description=f"{direction_signal.transition_description}{validation_suffix}",
                paradigm_significance=direction_signal.paradigm_significance + confidence_boost
            )
            
            validated_paradigms.append(validated_signal)
            validation_summary['accepted'] += 1
            
            # Generate decision rationale for transparency
            boost_text = f" + boost({confidence_boost:.2f})" if confidence_boost > 0 else ""
            rationale = f"Confidence: {base_confidence:.3f}{boost_text} = {final_confidence:.3f} ≥ threshold({algorithm_config.validation_threshold:.2f})"
            results.append((year, rationale))
        else:
            validation_summary['rejected'] += 1
            rationale = f"Confidence: {base_confidence:.3f} + boost({confidence_boost:.2f}) = {final_confidence:.3f} < threshold({algorithm_config.validation_threshold:.2f})"
            results.append((year, rationale))
    
    for year, rationale in results[:5]:
        logger.debug(f"    {year}: {rationale}")
    if len(results) > 5:
        logger.debug(f"    ... {len(results) - 5} more validation")

    return validated_paradigms, results


# =============================================================================
# MAIN DETECTION PIPELINE
# =============================================================================

def detect_shift_signals(
    domain_data: DomainData,
    domain_name: str,
    algorithm_config,
    use_citation: bool = True,
    use_direction: bool = True,
    precomputed_signals: Optional[Dict[str, List[ShiftSignal]]] = None,
    verbose: bool = False
) -> List[ShiftSignal]:
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
        algorithm_config: Comprehensive algorithm configuration for thresholds and parameters
        use_citation: Whether to use citation validation
        use_direction: Whether to use direction signals
        precomputed_signals: Optional pre-computed signals

    Returns:
        Tuple of (paradigm_shifts, segmentation_metadata)
    """
    # Stage 1: Primary Detection - Direction Signals 
    if precomputed_signals:
        raw_direction_signals = (
            precomputed_signals.get("direction", []) if use_direction else []
        )
        citation_signals = (
            precomputed_signals.get("citation", []) if use_citation else []
        )
    else:
        # PRIMARY: Research direction changes detect paradigm shifts
        raw_direction_signals = (
            detect_research_direction_changes(domain_data, algorithm_config, verbose=verbose) if use_direction else []
        )
        
        # SECONDARY: Citation patterns for validation
        citation_signals = (
            detect_citation_structural_breaks(domain_data, domain_name, algorithm_config, verbose=verbose)
            if use_citation
            else []
        )
    
    # Stage 3: Direction-Citation Validation
    paradigm_shifts, validation_results = validate_direction_with_citation(raw_direction_signals, citation_signals, algorithm_config, verbose=verbose)

    return paradigm_shifts