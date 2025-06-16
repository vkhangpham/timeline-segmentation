"""
Shift Signal Detection for Research Timeline Modeling.

This module implements sophisticated paradigm transition detection using rich data sources
and advanced change point detection methods. Part of Phase 9 fundamental framework that
separates shift signals (paradigm transitions) from period signals (stability characterization).

Based on comprehensive literature review and Phase 8 baseline analysis, this module combines:
- ruptures library for robust change point detection
- Enhanced semantic analysis using citation descriptions and breakthrough papers
- Multi-signal fusion for paradigm vs technical innovation distinction

Follows functional programming principles with pure functions and immutable data structures.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Set
from collections import defaultdict, Counter
import json
from pathlib import Path
import dataclasses

# Import ruptures for change point detection
import ruptures as rpt

# Import our data models
from .data_models import (
    Paper, CitationRelation as Citation, DomainData, 
    ChangePointWithPapers, ShiftSignal, TransitionEvidence
)


# Pure function to load paradigm patterns
def load_paradigm_patterns() -> Dict[str, List[str]]:
    """Load domain-specific paradigm shift patterns."""
    # Enhanced semantic patterns based on Phase 8 success
    patterns = {
        'architectural_shifts': [
            'introduces new architecture', 'revolutionary approach', 'novel architecture',
            'paradigm shift', 'breakthrough', 'first to', 'pioneer', 'foundational'
        ],
        'methodological_shifts': [
            'solves the problem of', 'enables training of', 'overcomes limitations',
            'fundamentally changes', 'transforms', 'revolutionizes'
        ],
        'domain_expansion': [
            'first application to', 'generalizes across', 'extends to',
            'applicable to', 'broader context'
        ],
        'foundational_work': [
            'lays the foundation', 'seminal contribution', 'establishes',
            'defines', 'creates framework', 'theoretical basis'
        ]
    }
    return patterns


def detect_shift_signals(
    domain_data: DomainData, 
    domain_name: str,
    use_citation: bool = True,
    use_semantic: bool = True,
    use_direction: bool = True,
    semantic_confidence_nudge: float = 0.0,
    semantic_temporal_nudge: int = 0,
    precomputed_signals: Optional[Dict[str, List[ShiftSignal]]] = None
) -> Tuple[List[ShiftSignal], List[TransitionEvidence]]:
    """
    Main function for detecting paradigm transition signals.
    
    Implements multi-source signal fusion with paradigm significance filtering.
    
    Args:
        domain_data: Domain data with papers and citations
        domain_name: Name of the domain for configuration
        use_citation: Whether to use citation disruption signal
        use_semantic: Whether to use semantic shift signal
        use_direction: Whether to use research direction signal
        semantic_confidence_nudge: Pct to nudge semantic signal confidence
        semantic_temporal_nudge: Years to nudge semantic signal timing
        precomputed_signals: Optional dict of pre-computed raw signals to bypass detection
        
    Returns:
        Tuple of (paradigm_shifts, transition_evidence)
    """
    print(f"\n🔍 SHIFT SIGNAL DETECTION: {domain_name}")
    print(f"  Configuration: Citation={use_citation}, Semantic={use_semantic}, Direction={use_direction}")
    print(f"  Nudges: Semantic Confidence={semantic_confidence_nudge}, Semantic Temporal={semantic_temporal_nudge}")
    print("=" * 60)
    
    # Stage 1: Individual signal detection
    if precomputed_signals:
        print("  ⚡️ Using pre-computed raw signals.")
        citation_disruptions = precomputed_signals.get('citation', []) if use_citation else []
        semantic_shifts = precomputed_signals.get('semantic', []) if use_semantic else []
        direction_volatility = precomputed_signals.get('direction', []) if use_direction else []
    else:
        citation_disruptions = detect_citation_structural_breaks(domain_data, domain_name) if use_citation else []
        semantic_shifts = detect_vocabulary_regime_changes(domain_data) if use_semantic else []
        direction_volatility = detect_research_direction_changes(domain_data) if use_direction else []
    
    # Apply nudges for sensitivity analysis
    if semantic_confidence_nudge != 0.0 and semantic_shifts:
        print(f"  ⚡️ Nudging semantic confidence by {semantic_confidence_nudge*100:.0f}%")
        semantic_shifts = [
            dataclasses.replace(s, confidence=min(s.confidence * (1 + semantic_confidence_nudge), 1.0))
            for s in semantic_shifts
        ]

    if semantic_temporal_nudge != 0 and semantic_shifts:
        print(f"  ⚡️ Nudging semantic temporal by {semantic_temporal_nudge} year(s)")
        semantic_shifts = [
            dataclasses.replace(s, year=s.year + semantic_temporal_nudge)
            for s in semantic_shifts
        ]
            
    print(f"  📊 Citation disruptions: {len(citation_disruptions)}")
    print(f"  📊 Semantic shifts: {len(semantic_shifts)}")
    print(f"  📊 Direction volatility: {len(direction_volatility)}")
    
    # Stage 2: Signal validation and confidence scoring
    all_signals = citation_disruptions + semantic_shifts + direction_volatility
    validated_signals = cross_validate_signals(all_signals, domain_data)
    
    print(f"  ✅ Validated signals: {len(validated_signals)}")
    
    # Stage 3: Paradigm vs technical distinction
    paradigm_shifts = filter_for_paradigm_significance(validated_signals, domain_data, domain_name)
    
    print(f"  🎯 Paradigm shifts identified: {len(paradigm_shifts)}")
    
    # Stage 4: Transition evidence generation
    transition_evidence = generate_transition_justifications(paradigm_shifts, domain_data)
    
    print(f"  📋 Transition evidence generated: {len(transition_evidence)}")
    
    # Stage 5: Save shift signals for visualization
    save_shift_signals_for_visualization(
        raw_signals=all_signals,
        validated_signals=validated_signals, 
        paradigm_shifts=paradigm_shifts,
        transition_evidence=transition_evidence,
        domain_name=domain_name
    )
    
    return paradigm_shifts, transition_evidence


def save_shift_signals_for_visualization(
    raw_signals: List[ShiftSignal],
    validated_signals: List[ShiftSignal],
    paradigm_shifts: List[ShiftSignal],
    transition_evidence: List[TransitionEvidence],
    domain_name: str,
    output_dir: str = "results/signals"
) -> str:
    """
    Save all shift signal detection results for visualization and analysis.
    
    Args:
        raw_signals: Raw signals from individual detection methods
        validated_signals: Cross-validated signals
        paradigm_shifts: Final paradigm shift signals
        transition_evidence: Supporting evidence for transitions
        domain_name: Name of the domain
        output_dir: Directory to save signal files
        
    Returns:
        Path to the saved shift signals file
    """
    from pathlib import Path
    from datetime import datetime
    import json
    
    # Ensure output directory exists
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    def serialize_shift_signal(signal: ShiftSignal) -> Dict:
        """Convert ShiftSignal to serializable dictionary"""
        return {
            'year': signal.year,
            'confidence': signal.confidence,
            'signal_type': signal.signal_type,
            'evidence_strength': signal.evidence_strength,
            'supporting_evidence': list(signal.supporting_evidence),
            'contributing_papers': list(signal.contributing_papers),
            'transition_description': signal.transition_description,
            'paradigm_significance': signal.paradigm_significance
        }
    
    def serialize_transition_evidence(evidence: TransitionEvidence) -> Dict:
        """Convert TransitionEvidence to serializable dictionary"""
        return {
            'year': evidence.year,
            'disruption_patterns': list(evidence.disruption_patterns),
            'emergence_patterns': list(evidence.emergence_patterns),
            'cross_domain_influences': list(evidence.cross_domain_influences),
            'methodological_changes': list(evidence.methodological_changes),
            'breakthrough_papers': list(evidence.breakthrough_papers),
            'confidence_score': evidence.confidence_score
        }
    
    # Create comprehensive shift signals dataset
    shift_signals_data = {
        'metadata': {
            'domain_name': domain_name,
            'analysis_date': datetime.now().isoformat(),
            'analysis_type': 'shift_signal_detection',
            'description': 'Paradigm transition detection using multi-source signal fusion',
            'methodology': {
                'stage1': 'Individual signal detection (citation, semantic, direction)',
                'stage2': 'Cross-validation and confidence scoring',
                'stage3': 'Paradigm significance filtering',
                'stage4': 'Transition evidence generation'
            }
        },
        'raw_signals': {
            'count': len(raw_signals),
            'description': 'Raw signals from individual detection methods before validation',
            'signals': [serialize_shift_signal(s) for s in raw_signals],
            'signal_types': list(set(s.signal_type for s in raw_signals))
        },
        'validated_signals': {
            'count': len(validated_signals),
            'description': 'Cross-validated signals with combined evidence',
            'signals': [serialize_shift_signal(s) for s in validated_signals],
            'validation_improvements': {
                'multi_signal_bonus_applied': True,
                'temporal_proximity_clustering': True,
                'evidence_combination': True
            }
        },
        'paradigm_shifts': {
            'count': len(paradigm_shifts),
            'description': 'Final paradigm shift signals after significance filtering',
            'signals': [serialize_shift_signal(s) for s in paradigm_shifts],
            'filtering_criteria': {
                'breakthrough_paper_proximity': 'Within 2 years',
                'multi_signal_confidence_boost': '>0.7 threshold',
                'domain_specific_thresholds': 'Applied'
            }
        },
        'transition_evidence': {
            'count': len(transition_evidence),
            'description': 'Supporting evidence and justifications for paradigm transitions',
            'evidence': [serialize_transition_evidence(e) for e in transition_evidence]
        },
        'visualization_metadata': {
            'timeline_data': {
                'raw_signal_years': sorted(list(set(s.year for s in raw_signals))),
                'validated_signal_years': sorted(list(set(s.year for s in validated_signals))),
                'paradigm_shift_years': sorted(list(set(s.year for s in paradigm_shifts)))
            },
            'confidence_distributions': {
                'raw_confidence_range': [
                    min([s.confidence for s in raw_signals] + [0]), 
                    max([s.confidence for s in raw_signals] + [1])
                ],
                'paradigm_confidence_range': [
                    min([s.confidence for s in paradigm_shifts] + [0]), 
                    max([s.confidence for s in paradigm_shifts] + [1])
                ]
            },
            'signal_type_analysis': {
                'raw_types': list(set(s.signal_type for s in raw_signals)),
                'final_types': list(set(s.signal_type for s in paradigm_shifts))
            },
            'filtering_statistics': {
                'raw_to_validated_retention': len(validated_signals) / max(len(raw_signals), 1),
                'validated_to_paradigm_retention': len(paradigm_shifts) / max(len(validated_signals), 1),
                'overall_retention': len(paradigm_shifts) / max(len(raw_signals), 1)
            }
        }
    }
    
    # Save to file
    output_file = f"{output_dir}/{domain_name}_shift_signals.json"
    with open(output_file, 'w') as f:
        json.dump(shift_signals_data, f, indent=2)
    
    print(f"  📊 SHIFT SIGNALS SAVED FOR VISUALIZATION:")
    print(f"      📁 File: {output_file}")
    print(f"      🔍 Raw signals: {len(raw_signals)}")
    print(f"      ✅ Validated signals: {len(validated_signals)}")
    print(f"      🎯 Paradigm shifts: {len(paradigm_shifts)}")
    print(f"      📋 Transition evidence: {len(transition_evidence)}")
    
    return output_file


def estimate_optimal_penalty(normalized_series: np.ndarray, domain_name: str) -> float:
    """
    FUNDAMENTAL SOLUTION: Data-driven penalty estimation for structural break detection.
    
    Automatically determines optimal penalty based on series characteristics:
    - Series variance and volatility
    - Signal-to-noise ratio
    - Temporal density
    - Domain-specific patterns
    
    UPDATED: Reduced over-segmentation by favoring paradigm significance over sensitivity
    
    Args:
        normalized_series: Normalized time series data
        domain_name: Domain name for logging
        
    Returns:
        Optimal penalty value for PELT algorithm
    """
    try:
        # Calculate data characteristics
        series_variance = np.var(normalized_series)
        series_mean = np.mean(normalized_series)
        series_std = np.std(normalized_series)
        
        # Signal-to-noise ratio estimation
        signal_strength = series_mean / (series_std + 1e-6)
        
        # Temporal volatility (adjacent differences)
        if len(normalized_series) > 1:
            temporal_volatility = np.mean(np.abs(np.diff(normalized_series)))
        else:
            temporal_volatility = 0.0
        
        # Coefficient of variation
        cv = series_std / (series_mean + 1e-6)
        
        # Data density (non-zero ratio)
        non_zero_ratio = np.count_nonzero(normalized_series) / len(normalized_series)
        
        print(f"    📊 Data characteristics: variance={series_variance:.3f}, SNR={signal_strength:.3f}, volatility={temporal_volatility:.3f}, CV={cv:.3f}, density={non_zero_ratio:.3f}")
        
        # FUNDAMENTAL ALGORITHM: Adaptive penalty based on data characteristics
        # UPDATED: Higher base penalties to reduce over-segmentation
        
        # Base penalty inversely related to signal strength and density
        # INCREASED: More conservative base penalty calculation
        base_penalty = 2.0 / (signal_strength + 0.2) * (1.2 / (non_zero_ratio + 0.2))
        
        # Adjust for temporal volatility (high volatility = lower penalty for sensitivity)
        # REDUCED: Less penalty reduction for volatility to prevent over-segmentation
        volatility_factor = 1.0 / (temporal_volatility * 5 + 1.0)  # Was 10, now 5
        
        # Adjust for coefficient of variation (high variation = lower penalty)
        # MODERATED: Less aggressive CV adjustment
        cv_factor = 1.0 / (cv + 0.8)  # Was 0.5, now 0.8
        
        # Series length adjustment (longer series can handle lower penalties)
        # CONSERVATIVE: Less penalty reduction for longer series
        length_factor = max(0.7, 1.0 - (len(normalized_series) - 20) / 200.0)  # Was 0.5, 10, 100
        
        # Combine factors
        adaptive_penalty = base_penalty * volatility_factor * cv_factor * length_factor
        
        # Ensure reasonable bounds - INCREASED minimum for paradigm significance
        optimal_penalty = np.clip(adaptive_penalty, 0.8, 6.0)  # Was 0.05-3.0, now 0.8-6.0
        
        print(f"    🎯 Adaptive penalty calculation: base={base_penalty:.3f}, volatility_factor={volatility_factor:.3f}, cv_factor={cv_factor:.3f}, length_factor={length_factor:.3f}")
        print(f"    ✅ Optimal penalty: {optimal_penalty:.3f} (ANTI-OVERSEGMENTATION)")
        
        return optimal_penalty
        
    except Exception as e:
        print(f"    ⚠️ Penalty estimation failed: {e}, using conservative fallback")
        # Fallback to conservative penalty for paradigm detection
        return max(1.0, np.median(normalized_series) * 4.0)  # Was 0.1, 2.0


def detect_citation_structural_breaks(domain_data: DomainData, domain_name: str) -> List[ShiftSignal]:
    """
    Detect structural breaks in citation patterns using ruptures.
    
    FUNDAMENTAL SOLUTION: Data-driven penalty selection and dense time series analysis
    UPDATED: Reduced over-segmentation with paradigm-focused thresholds
    
    Args:
        domain_data: Domain data with papers and citations
        domain_name: Domain name for logging
        
    Returns:
        List of citation disruption signals
    """
    # Create DENSE citation time series (fill gaps with zeros)
    citation_series = defaultdict(float)
    influence_series = defaultdict(float)
    
    # Get full year range 
    all_years = [p.pub_year for p in domain_data.papers]
    min_year, max_year = min(all_years), max(all_years)
    
    # Create dense series with zeros for missing years
    for year in range(min_year, max_year + 1):
        citation_series[year] = 0.0
        influence_series[year] = 0.0
    
    # Fill in actual citation data
    for paper in domain_data.papers:
        year = paper.pub_year
        citation_series[year] += paper.cited_by_count
        # Influence score: citation count weighted by recency
        recency_weight = 1.0 / (2025 - year + 1)
        influence_series[year] += paper.cited_by_count * recency_weight
    
    # Prepare DENSE data for ruptures
    years = sorted(citation_series.keys())
    citation_values = [citation_series[year] for year in years]
    influence_values = [influence_series[year] for year in years]
    
    print(f"    🔬 Dense time series: {len(years)} years ({min_year}-{max_year})")
    print(f"    📊 Citation range: {min(citation_values):,.0f} - {max(citation_values):,.0f}")
    
    # Use ruptures PELT algorithm for optimal segmentation
    signals = []
    
    # Citation structural breaks with DATA-DRIVEN penalty selection
    if len(citation_values) >= 5 and max(citation_values) > 0:
        try:
            # Normalize values for stability
            normalized_citations = np.array(citation_values) / max(citation_values)
            
            # FUNDAMENTAL SOLUTION: Data-driven penalty estimation
            optimal_penalty = estimate_optimal_penalty(normalized_citations, domain_name)
            
            algo = rpt.Pelt(model="l2").fit(normalized_citations.reshape(-1, 1))
            change_points = algo.predict(pen=optimal_penalty)
            
            print(f"    📍 Data-driven penalty {optimal_penalty:.3f}: {len(change_points)-1} change points detected")
            
            # UPDATED: Conservative confidence threshold for paradigm significance
            series_volatility = np.std(normalized_citations)
            dynamic_threshold = max(0.1, series_volatility * 0.8)  # Was 0.02, 0.5
            
            print(f"    🎯 Dynamic confidence threshold: {dynamic_threshold:.3f} (ANTI-OVERSEGMENTATION, based on volatility={series_volatility:.3f})")
            
            # Convert to ShiftSignal objects with PARADIGM-FOCUSED confidence threshold
            accepted_change_points = []
            for cp_idx in change_points[:-1]:  # Exclude last point (end of series)
                if 0 < cp_idx < len(years):
                    year = years[cp_idx]
                    
                    # MINIMUM PERIOD ENFORCEMENT: Ensure at least 4 years between change points
                    if accepted_change_points and year - accepted_change_points[-1] < 4:
                        print(f"      ❌ {year}: Too close to previous change point ({accepted_change_points[-1]}) - minimum 4-year gap required")
                        continue
                    
                    # Calculate confidence based on magnitude of change
                    before_mean = np.mean(normalized_citations[max(0, cp_idx-3):cp_idx])
                    after_mean = np.mean(normalized_citations[cp_idx:min(len(normalized_citations), cp_idx+3)])
                    change_magnitude = abs(after_mean - before_mean)
                    
                    # REDUCED: Less confidence inflation for realistic assessment
                    confidence = min(change_magnitude * 1.2, 1.0)  # Was 2.0, now 1.2
                    
                    # PARADIGM-FOCUSED confidence threshold
                    if confidence > dynamic_threshold:
                        contributing_papers = tuple(p.id for p in domain_data.papers if p.pub_year == year)
                        
                        print(f"      ✅ {year}: magnitude={change_magnitude:.3f}, confidence={confidence:.3f}")
                        
                        signals.append(ShiftSignal(
                            year=year,
                            confidence=confidence,
                            signal_type="citation_disruption",
                            evidence_strength=change_magnitude,
                            supporting_evidence=(f"Citation pattern change: {before_mean:.3f} → {after_mean:.3f}",),
                            contributing_papers=contributing_papers,
                            transition_description=f"Structural break in citation patterns at {year}",
                            paradigm_significance=0.5  # Initial neutral significance
                        ))
                        accepted_change_points.append(year)
                    else:
                        print(f"      ❌ {year}: magnitude={change_magnitude:.3f}, confidence={confidence:.3f} (below paradigm threshold {dynamic_threshold:.3f})")
                    
        except Exception as e:
            print(f"    ⚠️ Citation structural break detection failed: {e}")
    
    # Influence structural breaks with DATA-DRIVEN parameters
    if len(influence_values) >= 5 and max(influence_values) > 0:
        try:
            normalized_influence = np.array(influence_values) / max(influence_values)
            
            # Data-driven penalty for influence (slightly higher)
            influence_penalty = estimate_optimal_penalty(normalized_influence, domain_name) * 1.2
            
            algo = rpt.Pelt(model="l2").fit(normalized_influence.reshape(-1, 1))
            change_points = algo.predict(pen=influence_penalty)
            
            # Adaptive threshold for influence
            influence_volatility = np.std(normalized_influence)
            influence_threshold = max(0.03, influence_volatility * 0.6)
            
            for cp_idx in change_points[:-1]:
                if 0 < cp_idx < len(years):
                    year = years[cp_idx]
                    
                    # Avoid duplicates from citation analysis
                    if not any(s.year == year and s.signal_type == "citation_disruption" for s in signals):
                        before_mean = np.mean(normalized_influence[max(0, cp_idx-3):cp_idx])
                        after_mean = np.mean(normalized_influence[cp_idx:min(len(normalized_influence), cp_idx+3)])
                        change_magnitude = abs(after_mean - before_mean)
                        
                        confidence = min(change_magnitude * 2.0, 1.0)
                        
                        # ADAPTIVE confidence threshold for influence
                        if confidence > influence_threshold:
                            contributing_papers = tuple(p.id for p in domain_data.papers if p.pub_year == year)
                            
                            print(f"      ✅ {year} (influence): magnitude={change_magnitude:.3f}, confidence={confidence:.3f}")
                            
                            signals.append(ShiftSignal(
                                year=year,
                                confidence=confidence,
                                signal_type="citation_disruption",
                                evidence_strength=change_magnitude,
                                supporting_evidence=(f"Influence pattern change: {before_mean:.3f} → {after_mean:.3f}",),
                                contributing_papers=contributing_papers,
                                transition_description=f"Structural break in influence patterns at {year}",
                                paradigm_significance=0.6  # Slightly higher for influence patterns
                            ))
                            
        except Exception as e:
            print(f"    ⚠️ Influence structural break detection failed: {e}")
    
    print(f"    🎯 Total citation disruption signals: {len(signals)}")
    return signals


def detect_vocabulary_regime_changes(domain_data: DomainData) -> List[ShiftSignal]:
    """
    Detect semantic vocabulary shifts using enhanced patterns from Phase 8.
    
    Leverages rich citation descriptions and content abstracts.
    
    Args:
        domain_data: Domain data with papers and citations
        
    Returns:
        List of semantic shift signals
    """
    signals = []
    paradigm_patterns = load_paradigm_patterns()
    
    # Analyze semantic descriptions from citations
    year_descriptions = defaultdict(list)
    for citation in domain_data.citations:
        if citation.semantic_description:
            year_descriptions[citation.citing_year].append(citation.semantic_description.lower())
    
    # Analyze paradigm patterns across time windows
    years = sorted(year_descriptions.keys())
    window_size = 3
    
    for i in range(window_size, len(years) - window_size):
        year = years[i]
        
        # Current window semantic patterns
        current_window = []
        for y in years[i-window_size//2:i+window_size//2+1]:
            current_window.extend(year_descriptions[y])
        
        # Previous window
        prev_window = []
        for y in years[max(0, i-window_size):i]:
            prev_window.extend(year_descriptions[y])
        
        if not current_window or not prev_window:
            continue
        
        # Extract paradigm patterns
        current_patterns = extract_enhanced_semantic_patterns(current_window, paradigm_patterns)
        prev_patterns = extract_enhanced_semantic_patterns(prev_window, paradigm_patterns)
        
        # Detect emerging paradigm patterns
        novel_patterns = []
        paradigm_score = 0.0
        
        for pattern_type, patterns in current_patterns.items():
            for pattern, count in patterns.items():
                prev_count = prev_patterns.get(pattern_type, {}).get(pattern, 0)
                
                # Significant emergence or amplification
                if count >= 2 and (prev_count == 0 or count > prev_count * 2):
                    novel_patterns.append(f"{pattern_type}: {pattern}")
                    # Higher paradigm significance for architectural and foundational patterns
                    if pattern_type in ['architectural_shifts', 'foundational_work']:
                        paradigm_score += 0.3
                    elif pattern_type in ['methodological_shifts']:
                        paradigm_score += 0.2
                    else:
                        paradigm_score += 0.1
        
        # Create signal if significant patterns detected
        if len(novel_patterns) >= 2:  # Require multiple patterns for significance
            confidence = min(len(novel_patterns) / 5.0, 1.0)  # Scale confidence
            paradigm_significance = min(paradigm_score, 1.0)
            contributing_papers = tuple(p.id for p in domain_data.papers 
                                       if abs(p.pub_year - year) <= 1)
            
            signals.append(ShiftSignal(
                year=year,
                confidence=confidence,
                signal_type="semantic_shift",
                evidence_strength=paradigm_score,
                supporting_evidence=tuple(novel_patterns[:5]),  # Top 5 patterns
                contributing_papers=contributing_papers,
                transition_description=f"Semantic paradigm shift: {', '.join(novel_patterns[:3])}",
                paradigm_significance=paradigm_significance
            ))
    
    return signals


def extract_enhanced_semantic_patterns(descriptions: List[str], paradigm_patterns: Dict[str, List[str]]) -> Dict[str, Dict[str, int]]:
    """Extract enhanced semantic patterns using Phase 8 paradigm indicators."""
    patterns = {category: {} for category in paradigm_patterns.keys()}
    
    for desc in descriptions:
        for category, keywords in paradigm_patterns.items():
            for keyword in keywords:
                if keyword in desc:
                    patterns[category][keyword] = patterns[category].get(keyword, 0) + 1
    
    return patterns


def detect_research_direction_changes(domain_data: DomainData) -> List[ShiftSignal]:
    """
    Detect changes in research directions using keyword and topic volatility.
    
    Args:
        domain_data: Domain data with papers and citations
        
    Returns:
        List of direction volatility signals
    """
    signals = []
    
    # Group papers by year and analyze keyword evolution
    year_keywords = defaultdict(list)
    for paper in domain_data.papers:
        year_keywords[paper.pub_year].extend(paper.keywords)
    
    years = sorted(year_keywords.keys())
    if len(years) < 5:
        return []
    
    # Calculate keyword diversity and volatility
    window_size = 3
    for i in range(window_size, len(years)):
        year = years[i]
        
        # Current window keywords
        current_keywords = []
        for y in years[i-window_size:i]:
            current_keywords.extend(year_keywords[y])
        
        # Previous window keywords
        prev_keywords = []
        for y in years[max(0, i-window_size*2):i-window_size]:
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
        
        # High novelty + low overlap indicates direction change
        direction_change_score = novelty * (1 - overlap)
        
        if direction_change_score > 0.4:  # Threshold for significant direction change
            # Count frequency of new keywords to assess significance
            new_keywords = current_set - prev_set
            keyword_frequencies = Counter(current_keywords)
            significant_new = [kw for kw in new_keywords if keyword_frequencies[kw] >= 2]
            
            if len(significant_new) >= 3:  # Multiple significant new keywords
                confidence = min(direction_change_score, 1.0)
                contributing_papers = tuple(p.id for p in domain_data.papers 
                                           if p.pub_year == year and 
                                           any(kw in p.keywords for kw in significant_new))
                
                signals.append(ShiftSignal(
                    year=year,
                    confidence=confidence,
                    signal_type="direction_volatility",
                    evidence_strength=direction_change_score,
                    supporting_evidence=tuple(f"New focus: {kw}" for kw in significant_new[:5]),
                    contributing_papers=contributing_papers,
                    transition_description=f"Research direction shift: {novelty:.1%} new keywords",
                    paradigm_significance=0.4  # Moderate paradigm significance for direction changes
                ))
    
    return signals


def cross_validate_signals(signals: List[ShiftSignal], domain_data: DomainData) -> List[ShiftSignal]:
    """
    Cross-validate signals using multiple evidence sources and temporal proximity.
    
    Args:
        signals: List of detected signals
        domain_data: Domain data for validation
        
    Returns:
        List of validated signals
    """
    if not signals:
        return []
    
    # Group signals by year (within 2-year window)
    signal_groups = defaultdict(list)
    for signal in signals:
        signal_groups[signal.year].append(signal)
        # Also add to nearby years for clustering
        signal_groups[signal.year - 1].append(signal)
        signal_groups[signal.year + 1].append(signal)
    
    validated_signals = []
    processed_years = set()
    
    for year, year_signals in signal_groups.items():
        if year in processed_years or len(year_signals) < 1:
            continue
        
        # Find the year with the most signals in the cluster
        cluster_signals = defaultdict(list)
        for signal in year_signals:
            cluster_signals[signal.year].append(signal)
        
        # Get the year with maximum signals
        best_year = max(cluster_signals.keys(), key=lambda y: len(cluster_signals[y]))
        best_signals = cluster_signals[best_year]
        
        if len(best_signals) >= 1:  # At least one signal required
            # Combine evidence from multiple signal types
            combined_confidence = 0.0
            combined_evidence = []
            combined_papers = set()
            max_paradigm_significance = 0.0
            
            signal_types = set()
            for signal in best_signals:
                signal_types.add(signal.signal_type)
                combined_confidence += signal.confidence
                combined_evidence.extend(signal.supporting_evidence)
                combined_papers.update(signal.contributing_papers)
                max_paradigm_significance = max(max_paradigm_significance, signal.paradigm_significance)
            
            # Multi-signal bonus for cross-validation
            multi_signal_bonus = 0.2 * (len(signal_types) - 1)
            final_confidence = min(combined_confidence + multi_signal_bonus, 1.0)
            
            # Create validated signal
            primary_signal = max(best_signals, key=lambda s: s.confidence)
            
            validated_signal = ShiftSignal(
                year=best_year,
                confidence=final_confidence,
                signal_type=f"validated_{primary_signal.signal_type}",
                evidence_strength=primary_signal.evidence_strength,
                supporting_evidence=tuple(combined_evidence[:10]),  # Top 10 pieces of evidence
                contributing_papers=tuple(combined_papers),
                transition_description=f"Multi-signal paradigm transition: {', '.join(signal_types)}",
                paradigm_significance=max_paradigm_significance
            )
            
            validated_signals.append(validated_signal)
            
            # Mark nearby years as processed to avoid duplicates
            for y in range(best_year - 2, best_year + 3):
                processed_years.add(y)
    
    return validated_signals


def filter_for_paradigm_significance(signals: List[ShiftSignal], domain_data: DomainData, domain_name: str) -> List[ShiftSignal]:
    """
    Filter signals for paradigm significance using breakthrough papers and domain context.
    
    Implements Trial 3: Hierarchical Paradigm Filtering
    
    Args:
        signals: List of validated signals
        domain_data: Domain data
        domain_name: Domain name for configuration
        
    Returns:
        List of paradigm shift signals
    """
    if not signals:
        return []
    
    # Load breakthrough papers for domain
    breakthrough_papers = load_breakthrough_papers(domain_data, domain_name)
    breakthrough_years = {p.pub_year for p in breakthrough_papers}
    
    paradigm_shifts = []
    
    for signal in signals:
        paradigm_score = signal.paradigm_significance
        
        # Boost significance for signals near breakthrough papers
        nearby_breakthroughs = [year for year in breakthrough_years 
                              if abs(year - signal.year) <= 2]
        if nearby_breakthroughs:
            paradigm_score += 0.3
            
        # Boost significance for strong multi-signal evidence
        if signal.signal_type.startswith("validated_") and signal.confidence > 0.7:
            paradigm_score += 0.2
        
        # Apply domain-specific significance thresholds (lowered for better recall)
        significance_threshold = {
            'natural_language_processing': 0.5,  # Was 0.8 → improved recall
            'deep_learning': 0.4,               # Was 0.6 → better detection
            'computer_vision': 0.4,             # Was 0.5 → enhanced sensitivity
            'machine_translation': 0.4,         # Was 0.5 → improved coverage
            'machine_learning': 0.3             # Was 0.4 → maximum sensitivity
        }
        
        threshold = significance_threshold.get(domain_name, 0.5)
        
        if paradigm_score >= threshold:
            # Update signal with final paradigm score
            paradigm_signal = ShiftSignal(
                year=signal.year,
                confidence=signal.confidence,
                signal_type=signal.signal_type,
                evidence_strength=signal.evidence_strength,
                supporting_evidence=signal.supporting_evidence,
                contributing_papers=signal.contributing_papers,
                transition_description=signal.transition_description,
                paradigm_significance=paradigm_score
            )
            paradigm_shifts.append(paradigm_signal)
    
    return paradigm_shifts


def load_breakthrough_papers(domain_data: DomainData, domain_name: str) -> List[Paper]:
    """
    Load breakthrough papers for significance weighting.
    
    Args:
        domain_data: Domain data
        domain_name: Domain name
        
    Returns:
        List of breakthrough papers
    """
    try:
        # Construct file path for breakthrough papers
        breakthrough_file = Path(f"resources/{domain_name}/{domain_name}_breakthrough_papers.jsonl")
        
        if not breakthrough_file.exists():
            print(f"    ⚠️ Breakthrough papers file not found: {breakthrough_file}")
            return []
        
        # Load breakthrough paper IDs
        breakthrough_ids = set()
        with open(breakthrough_file, 'r') as f:
            for line in f:
                if line.strip():
                    paper_data = json.loads(line.strip())
                    paper_id = paper_data.get('openalex_id', '')
                    if paper_id:
                        breakthrough_ids.add(paper_id)
        
        # Filter papers that are breakthrough papers
        breakthrough_papers = [p for p in domain_data.papers if p.id in breakthrough_ids]
        
        print(f"    📚 Loaded {len(breakthrough_papers)} breakthrough papers from {len(breakthrough_ids)} IDs")
        return breakthrough_papers
        
    except Exception as e:
        print(f"    ⚠️ Error loading breakthrough papers: {e}")
        return []


def generate_transition_justifications(paradigm_shifts: List[ShiftSignal], 
                                      domain_data: DomainData) -> List[TransitionEvidence]:
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
        window_papers = [p for p in domain_data.papers 
                        if abs(p.pub_year - signal.year) <= 2]
        
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
            emergence_patterns.extend([f"Emerging keyword: {kw}" for kw in top_keywords[:3]])
        
        # Analyze semantic descriptions for patterns
        semantic_patterns = []
        for citation in domain_data.citations:
            if (abs(citation.citing_year - signal.year) <= 1 and 
                citation.semantic_description):
                desc = citation.semantic_description.lower()
                if any(word in desc for word in ['introduces', 'novel', 'new', 'breakthrough']):
                    semantic_patterns.append(citation.semantic_description[:100])
        
        if semantic_patterns:
            methodological_changes.extend(semantic_patterns[:3])
        
        # Create transition evidence
        evidence = TransitionEvidence(
            year=signal.year,
            disruption_patterns=tuple(disruption_patterns),
            emergence_patterns=tuple(emergence_patterns),
            cross_domain_influences=tuple(),  # Could be enhanced in future
            methodological_changes=tuple(methodological_changes),
            breakthrough_papers=signal.contributing_papers[:5],  # Top 5 contributing papers
            confidence_score=signal.confidence
        )
        
        transition_evidence.append(evidence)
    
    return transition_evidence


def convert_to_change_points(shift_signals: List[ShiftSignal]) -> List[ChangePointWithPapers]:
    """
    Convert shift signals to change points for compatibility with existing pipeline.
    
    Args:
        shift_signals: List of shift signals
        
    Returns:
        List of change points with papers
    """
    change_points = []
    
    for signal in shift_signals:
        change_point = ChangePointWithPapers(
            year=signal.year,
            confidence=signal.confidence,
            method="enhanced_shift_signal",
            signal_type=signal.signal_type,
            description=signal.transition_description,
            supporting_evidence=signal.supporting_evidence,
            contributing_papers=signal.contributing_papers
        )
        change_points.append(change_point)
    
    return change_points


def detect_paradigm_shifts_trial1(domain: str, papers_data: List[Dict],
                                 semantic_data: Optional[List[Dict]] = None,
                                 breakthrough_papers: Optional[List[Dict]] = None) -> List[Dict]:
    """
    Legacy function for detecting paradigm shifts (Trial 1 compatibility).
    
    Args:
        domain: Domain name
        papers_data: List of paper dictionaries
        semantic_data: Optional semantic data
        breakthrough_papers: Optional breakthrough papers
        
    Returns:
        List of paradigm shift dictionaries
    """
    # Convert to domain data format
    papers = []
    for paper_dict in papers_data:
        paper = Paper(
            id=paper_dict.get('id', ''),
            title=paper_dict.get('title', ''),
            content=paper_dict.get('content', ''),
            pub_year=paper_dict.get('pub_year', 0),
            cited_by_count=paper_dict.get('cited_by_count', 0),
            keywords=tuple(paper_dict.get('keywords', [])),
            children=tuple(paper_dict.get('children', [])),
            description=paper_dict.get('description', '')
        )
        papers.append(paper)
    
    # Create domain data
    citations = []
    if semantic_data:
        for sem_data in semantic_data:
            citation = Citation(
                citing_paper_id=sem_data.get('citing_paper_id', ''),
                cited_paper_id=sem_data.get('cited_paper_id', ''),
                citing_year=sem_data.get('citing_year', 0),
                cited_year=sem_data.get('cited_year', 0),
                semantic_description=sem_data.get('semantic_description', '')
            )
            citations.append(citation)
    
    year_range = (min(p.pub_year for p in papers), max(p.pub_year for p in papers)) if papers else (0, 0)
    
    domain_data = DomainData(
        domain_name=domain,
        papers=tuple(papers),
        citations=tuple(citations),
        graph_nodes=tuple(),
        year_range=year_range,
        total_papers=len(papers)
    )
    
    # Detect shift signals
    shift_signals, _ = detect_shift_signals(domain_data, domain)
    
    # Convert to dictionary format for legacy compatibility
    results = []
    for signal in shift_signals:
        result = {
            'year': signal.year,
            'confidence': signal.confidence,
            'type': signal.signal_type,
            'description': signal.transition_description,
            'evidence': list(signal.supporting_evidence),
            'paradigm_significance': signal.paradigm_significance
        }
        results.append(result)
    
    return results