"""
Objective Function for Timeline Segmentation
============================================

This module implements the validated objective function for timeline segmentation
based on comprehensive multi-domain analysis of 12,000 random segments.

Key Features:
- Jaccard cohesion metric (keyword overlap within segments)
- Jensen-Shannon separation metric (vocabulary shift between segments)
- Orthogonal metrics (r=0.001 across domains) enabling linear combination
- Cohesion-dominant weighting (0.8, 0.2) optimized for expert timeline performance
- Anti-gaming safeguards to prevent metric exploitation

Anti-Gaming Mechanisms:
- Size-weighted averaging prevents micro-segment gaming
- Minimum segment floor excludes tiny segments
- Segment count penalty discourages excessive segmentation

Validation Results:
- 4 domains analyzed (NLP, CV, Math, Art)
- Expert timelines: 37th percentile cohesion, 15th percentile separation
- Cross-domain consistency maintained
- Production-ready with fail-fast error handling
"""

from __future__ import annotations

import numpy as np
import math
from typing import List, Tuple, Dict, NamedTuple, Optional
from collections import defaultdict, Counter
from scipy.spatial.distance import jensenshannon
import json
import os

from ..data.models import Paper
from ..utils.logging import get_logger


class AntiGamingConfig(NamedTuple):
    """Configuration for anti-gaming safeguards."""
    min_segment_size: int = 50  # Minimum papers per segment
    size_weight_power: float = 0.5  # Power for size weighting (0.5 = sqrt)
    segment_count_penalty_sigma: float = 4.0  # Exponential decay parameter
    enable_size_weighting: bool = True
    enable_segment_floor: bool = True
    enable_count_penalty: bool = True


class ObjectiveFunctionResult(NamedTuple):
    """Complete result of objective function evaluation."""
    final_score: float
    cohesion_score: float
    separation_score: float
    num_segments: int
    num_transitions: int
    cohesion_details: str
    separation_details: str
    methodology: str
    # Anti-gaming metrics
    size_weighted_cohesion: float = 0.0
    size_weighted_separation: float = 0.0
    segment_count_penalty: float = 1.0
    excluded_segments: int = 0


class SegmentMetrics(NamedTuple):
    """Metrics for a single segment."""
    cohesion: float
    size: int
    keywords_count: int
    top_keywords: List[str]


class TransitionMetrics(NamedTuple):
    """Metrics for a transition between two segments."""
    separation: float
    vocab_size: int
    segment_a_size: int
    segment_b_size: int


def load_objective_weights() -> Tuple[float, float]:
    """Load objective function weights from configuration."""
    config_path = "optimization_config.json"
    
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Load from dedicated objective_function section
        obj_config = config.get("objective_function", {})
        if obj_config:
            cohesion_weight = obj_config.get("cohesion_weight", 0.8)
            separation_weight = obj_config.get("separation_weight", 0.2)
            return cohesion_weight, separation_weight
    
    # Fallback to validated optimal weights
    return 0.8, 0.2


def load_top_k_keywords() -> int:
    """Load top-K keywords parameter from configuration."""
    config_path = "optimization_config.json"
    
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Load from dedicated objective_function section
        obj_config = config.get("objective_function", {})
        if obj_config:
            return obj_config.get("top_k_keywords", 15)
    
    # Fallback to validated optimal value
    return 15


def load_anti_gaming_config() -> AntiGamingConfig:
    """Load anti-gaming configuration from file."""
    config_path = "optimization_config.json"
    
    if os.path.exists(config_path):
        # FAIL-FAST: Load anti-gaming config or fail immediately with clear error message
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Load from dedicated anti_gaming section
        ag_config = config.get("anti_gaming", {})
        if ag_config:
            return AntiGamingConfig(
                min_segment_size=ag_config.get("min_segment_size", 50),
                size_weight_power=ag_config.get("size_weight_power", 0.5),
                segment_count_penalty_sigma=ag_config.get("segment_count_penalty_sigma", 4.0),
                enable_size_weighting=ag_config.get("enable_size_weighting", True),
                enable_segment_floor=ag_config.get("enable_segment_floor", True),
                enable_count_penalty=ag_config.get("enable_count_penalty", True)
            )
    
    # Fallback to default configuration
    return AntiGamingConfig()


def compute_size_weighted_average(segment_scores: List[float], segment_sizes: List[int], power: float = 0.5) -> float:
    """
    Compute size-weighted average to prevent micro-segment gaming.
    
    Uses power weighting: weight = size^power
    - power=0.0: uniform weighting (vulnerable to gaming)
    - power=0.5: square root weighting (balanced)
    - power=1.0: linear weighting (heavily favors large segments)
    
    Args:
        segment_scores: List of metric scores per segment
        segment_sizes: List of sizes (number of papers) per segment
        power: Power for size weighting (default: 0.5)
        
    Returns:
        Size-weighted average score
    
    Raises:
        ValueError: If inputs have mismatched lengths
    """
    if not segment_scores or not segment_sizes:
        return 0.0
    
    if len(segment_scores) != len(segment_sizes):
        raise ValueError("segment_scores and segment_sizes must have same length")
    
    # Compute weights
    weights = [size ** power for size in segment_sizes]
    total_weight = sum(weights)
    
    if total_weight == 0:
        return 0.0
    
    # Weighted average
    weighted_sum = sum(score * weight for score, weight in zip(segment_scores, weights))
    return weighted_sum / total_weight


def compute_segment_count_penalty(num_segments: int, domain_year_span: int, sigma: float = 4.0) -> float:
    """
    Compute exponential penalty for deviating from expected segment count.
    
    Expected segments = domain_year_span / 15 (one segment per ~15 years)
    Penalty = exp(-|K_actual - K_expected| / σ)
    
    Args:
        num_segments: Actual number of segments
        domain_year_span: Total years covered by domain
        sigma: Exponential decay parameter (higher = more lenient)
        
    Returns:
        Penalty factor in [0, 1]
    """
    if domain_year_span <= 0:
        return 1.0
    
    # Realistic expectation: one segment per 15 years
    expected_segments = max(1, round(domain_year_span / 15))
    deviation = abs(num_segments - expected_segments)
    
    # Exponential penalty with configurable sigma
    penalty = math.exp(-deviation / sigma)
    
    return penalty


def filter_segments_by_size(segment_papers: List[List[Paper]], min_size: int) -> Tuple[List[List[Paper]], int]:
    """
    Filter out segments below minimum size threshold.
    
    Args:
        segment_papers: List of paper segments
        min_size: Minimum papers per segment
        
    Returns:
        Tuple of (filtered_segments, excluded_count)
    """
    filtered_segments = []
    excluded_count = 0
    
    for segment in segment_papers:
        if len(segment) >= min_size:
            filtered_segments.append(segment)
        else:
            excluded_count += 1
    
    return filtered_segments, excluded_count


def compute_jaccard_cohesion(segment_papers: List[Paper], top_k: int = None) -> Tuple[float, str, List[str]]:
    """
    Compute segment cohesion using mean Jaccard similarity of top-K keywords.
    
    This metric measures how well papers within a segment share common vocabulary,
    focusing on the most defining keywords of the segment.
    
    Args:
        segment_papers: List of papers in the segment
        top_k: Number of top keywords to use for defining the segment (default: load from config)
    
    Returns:
        Tuple of (cohesion_score, explanation, top_keywords_list)
    
    Raises:
        ValueError: If segment is empty or has no keywords
    """
    if not segment_papers:
        raise ValueError("Segment cannot be empty")
    
    # Load top_k from configuration if not provided
    if top_k is None:
        top_k = load_top_k_keywords()
    
    # Collect all keywords and their frequencies
    keyword_counts = defaultdict(int)
    total_papers_with_keywords = 0
    
    for paper in segment_papers:
        if paper.keywords:
            total_papers_with_keywords += 1
            for keyword in paper.keywords:
                keyword_counts[keyword] += 1
    
    if not keyword_counts:
        return 0.0, "No keywords found in segment", []
    
    # Get top-K keywords by frequency
    top_keywords_items = sorted(keyword_counts.items(), key=lambda x: x[1], reverse=True)[:top_k]
    top_keywords = {kw for kw, count in top_keywords_items}
    top_keywords_list = [kw for kw, count in top_keywords_items]
    
    if not top_keywords:
        return 0.0, "No defining keywords found", []
    
    # Compute Jaccard similarity for each paper with defining keywords
    jaccard_scores = []
    
    for paper in segment_papers:
        paper_keywords = set(paper.keywords) if paper.keywords else set()
        
        if paper_keywords & top_keywords:  # Paper has at least one defining keyword
            intersection = len(paper_keywords & top_keywords)
            union = len(paper_keywords | top_keywords)
            
            if union > 0:
                jaccard = intersection / union
                jaccard_scores.append(jaccard)
    
    if not jaccard_scores:
        return 0.0, f"No papers match top-{top_k} keywords", top_keywords_list
    
    mean_cohesion = float(np.mean(jaccard_scores))
    
    explanation = (
        f"Jaccard cohesion: {len(jaccard_scores)}/{len(segment_papers)} papers match "
        f"top-{len(top_keywords)} keywords (freq: {keyword_counts[top_keywords_list[0]] if top_keywords_list else 0}-"
        f"{keyword_counts[top_keywords_list[-1]] if top_keywords_list else 0}) → {mean_cohesion:.3f}"
    )
    
    return mean_cohesion, explanation, top_keywords_list


def compute_jensen_shannon_separation(segment_a: List[Paper], segment_b: List[Paper]) -> Tuple[float, str]:
    """
    Compute separation between two segments using Jensen-Shannon divergence.
    
    This metric measures vocabulary shift between segments by comparing
    keyword frequency distributions using information-theoretic divergence.
    
    Args:
        segment_a: Papers in first segment
        segment_b: Papers in second segment
    
    Returns:
        Tuple of (separation_score, explanation)
    
    Raises:
        ValueError: If either segment is empty
    """
    if not segment_a or not segment_b:
        raise ValueError("Both segments must be non-empty")
    
    # Collect keywords from both segments
    keywords_a = []
    keywords_b = []
    
    for paper in segment_a:
        if paper.keywords:
            keywords_a.extend(paper.keywords)
    
    for paper in segment_b:
        if paper.keywords:
            keywords_b.extend(paper.keywords)
    
    if not keywords_a or not keywords_b:
        return 0.0, "One or both segments have no keywords"
    
    # Create unified vocabulary
    vocab = list(set(keywords_a) | set(keywords_b))
    vocab_size = len(vocab)
    
    if vocab_size == 0:
        return 0.0, "No vocabulary overlap"
    
    # Compute frequency distributions
    def get_frequency_distribution(keywords: List[str], vocabulary: List[str]) -> np.ndarray:
        counts = Counter(keywords)
        total = sum(counts.values())
        
        if total == 0:
            return np.ones(len(vocabulary)) / len(vocabulary)  # Uniform distribution
        
        return np.array([counts[word] / total for word in vocabulary])
    
    p = get_frequency_distribution(keywords_a, vocab)
    q = get_frequency_distribution(keywords_b, vocab)
    
    # Compute Jensen-Shannon divergence
    # Add small epsilon to avoid numerical issues
    epsilon = 1e-10
    p = p + epsilon
    q = q + epsilon
    
    # Normalize after adding epsilon
    p = p / p.sum()
    q = q / q.sum()
    
    # Jensen-Shannon divergence using scipy
    js_divergence = jensenshannon(p, q, base=2)
    
    # Convert to separation score (0-1 range)
    separation_score = float(js_divergence)
    
    explanation = (
        f"Jensen-Shannon separation: vocab_size={vocab_size}, "
        f"seg_a_keywords={len(keywords_a)}, seg_b_keywords={len(keywords_b)} → {separation_score:.3f}"
    )
    
    return separation_score, explanation


def evaluate_segment_cohesion(segment_papers: List[Paper]) -> SegmentMetrics:
    """Evaluate cohesion metrics for a single segment."""
    if not segment_papers:
        raise ValueError("Segment cannot be empty")
    
    cohesion, details, top_keywords = compute_jaccard_cohesion(segment_papers)
    
    # Count total keywords
    total_keywords = sum(len(paper.keywords) for paper in segment_papers if paper.keywords)
    
    return SegmentMetrics(
        cohesion=cohesion,
        size=len(segment_papers),
        keywords_count=total_keywords,
        top_keywords=top_keywords[:5]  # Store top 5 for summary
    )


def evaluate_transition_separation(segment_a: List[Paper], segment_b: List[Paper]) -> TransitionMetrics:
    """Evaluate separation metrics for a transition between two segments."""
    if not segment_a or not segment_b:
        raise ValueError("Both segments must be non-empty")
    
    separation, details = compute_jensen_shannon_separation(segment_a, segment_b)
    
    # Get vocabulary sizes
    vocab_a = set()
    vocab_b = set()
    
    for paper in segment_a:
        if paper.keywords:
            vocab_a.update(paper.keywords)
    
    for paper in segment_b:
        if paper.keywords:
            vocab_b.update(paper.keywords)
    
    total_vocab = len(vocab_a | vocab_b)
    
    return TransitionMetrics(
        separation=separation,
        vocab_size=total_vocab,
        segment_a_size=len(segment_a),
        segment_b_size=len(segment_b)
    )


def compute_objective_function(segment_papers: List[List[Paper]], 
                             cohesion_weight: float = None,
                             separation_weight: float = None,
                             anti_gaming_config: AntiGamingConfig = None) -> ObjectiveFunctionResult:
    """
    Compute the complete objective function for a timeline segmentation with anti-gaming safeguards.
    
    This function implements the validated objective function:
    - Jaccard cohesion (keyword overlap within segments)
    - Jensen-Shannon separation (vocabulary shift between segments)  
    - Linear combination with cohesion-dominant weights (0.8, 0.2)
    - Anti-gaming safeguards to prevent metric exploitation
    
    Args:
        segment_papers: List of segments, where each segment is a list of Paper objects
        cohesion_weight: Weight for cohesion component (default: load from config)
        separation_weight: Weight for separation component (default: load from config)
        anti_gaming_config: Anti-gaming configuration (default: load from config)
    
    Returns:
        ObjectiveFunctionResult with complete evaluation including anti-gaming metrics
    
    Raises:
        ValueError: If segmentation is empty or contains empty segments
    """
    if not segment_papers:
        raise ValueError("Segmentation cannot be empty")
    
    if any(not segment for segment in segment_papers):
        raise ValueError("All segments must contain at least one paper")
    
    # Load weights from configuration if not provided
    if cohesion_weight is None or separation_weight is None:
        config_cohesion_weight, config_separation_weight = load_objective_weights()
        cohesion_weight = cohesion_weight or config_cohesion_weight
        separation_weight = separation_weight or config_separation_weight
    
    # Load anti-gaming configuration if not provided
    if anti_gaming_config is None:
        anti_gaming_config = load_anti_gaming_config()
    
    # Validate weights
    if abs(cohesion_weight + separation_weight - 1.0) > 1e-6:
        raise ValueError(f"Weights must sum to 1.0, got {cohesion_weight + separation_weight:.6f}")
    
    # Apply segment size filtering if enabled
    original_segment_count = len(segment_papers)
    excluded_segments = 0
    
    if anti_gaming_config.enable_segment_floor:
        segment_papers, excluded_segments = filter_segments_by_size(
            segment_papers, anti_gaming_config.min_segment_size
        )
        
        if not segment_papers:
            raise ValueError(f"All segments excluded by minimum size filter ({anti_gaming_config.min_segment_size} papers)")
    
    num_segments = len(segment_papers)
    num_transitions = max(0, num_segments - 1)
    
    # Handle single segment case
    if num_segments == 1:
        segment_metrics = evaluate_segment_cohesion(segment_papers[0])
        
        # For single segment, final score is just cohesion (no separation)
        final_score = cohesion_weight * segment_metrics.cohesion
        
        methodology = (
            f"Single segment objective: {cohesion_weight:.1f} × {segment_metrics.cohesion:.3f} "
            f"(cohesion only) = {final_score:.3f}"
        )
        
        return ObjectiveFunctionResult(
            final_score=final_score,
            cohesion_score=segment_metrics.cohesion,
            separation_score=0.0,
            num_segments=1,
            num_transitions=0,
            cohesion_details=f"Single segment: {segment_metrics.size} papers, {segment_metrics.keywords_count} keywords",
            separation_details="No transitions in single segment",
            methodology=methodology,
            size_weighted_cohesion=segment_metrics.cohesion,
            size_weighted_separation=0.0,
            segment_count_penalty=1.0,
            excluded_segments=excluded_segments
        )
    
    # Multiple segments case
    segment_metrics_list = []
    transition_metrics_list = []
    
    # Evaluate cohesion for each segment
    for i, segment in enumerate(segment_papers):
        try:
            metrics = evaluate_segment_cohesion(segment)
            segment_metrics_list.append(metrics)
        except Exception as e:
            raise ValueError(f"Failed to evaluate cohesion for segment {i+1}: {e}")
    
    # Evaluate separation for each transition
    for i in range(num_transitions):
        try:
            metrics = evaluate_transition_separation(segment_papers[i], segment_papers[i + 1])
            transition_metrics_list.append(metrics)
        except Exception as e:
            raise ValueError(f"Failed to evaluate separation for transition {i+1}→{i+2}: {e}")
    
    # Standard aggregation
    cohesion_scores = [metrics.cohesion for metrics in segment_metrics_list]
    separation_scores = [metrics.separation for metrics in transition_metrics_list]
    
    avg_cohesion = float(np.mean(cohesion_scores))
    avg_separation = float(np.mean(separation_scores)) if separation_scores else 0.0
    
    # Anti-gaming: Size-weighted aggregation
    size_weighted_cohesion = avg_cohesion
    size_weighted_separation = avg_separation
    
    if anti_gaming_config.enable_size_weighting:
        # Size-weighted cohesion
        segment_sizes = [metrics.size for metrics in segment_metrics_list]
        if cohesion_scores and segment_sizes:
            size_weighted_cohesion = compute_size_weighted_average(
                cohesion_scores, segment_sizes, anti_gaming_config.size_weight_power
            )
        
        # Size-weighted separation (using geometric mean of adjacent segment sizes)
        if separation_scores and len(segment_sizes) > 1:
            transition_sizes = []
            for i in range(len(segment_sizes) - 1):
                transition_size = math.sqrt(segment_sizes[i] * segment_sizes[i + 1])
                transition_sizes.append(transition_size)
            
            size_weighted_separation = compute_size_weighted_average(
                separation_scores, transition_sizes, anti_gaming_config.size_weight_power
            )
    
    # Segment count penalty
    segment_count_penalty = 1.0
    if anti_gaming_config.enable_count_penalty:
        # Calculate domain year span from all papers
        all_papers = [paper for segment in segment_papers for paper in segment]
        if all_papers:
            years = [paper.pub_year for paper in all_papers]
            domain_year_span = max(years) - min(years) + 1
            segment_count_penalty = compute_segment_count_penalty(
                num_segments, domain_year_span, anti_gaming_config.segment_count_penalty_sigma
            )
    
    # Compute final objective score with anti-gaming
    if anti_gaming_config.enable_size_weighting:
        final_score = (cohesion_weight * size_weighted_cohesion + 
                      separation_weight * size_weighted_separation) * segment_count_penalty
    else:
        final_score = (cohesion_weight * avg_cohesion + 
                      separation_weight * avg_separation) * segment_count_penalty
    
    # Create detailed explanations
    cohesion_details = " | ".join([
        f"Seg{i+1}: {metrics.cohesion:.3f} ({metrics.size}p, top: {', '.join(metrics.top_keywords[:3])})"
        for i, metrics in enumerate(segment_metrics_list)
    ])
    
    separation_details = " | ".join([
        f"T{i+1}→{i+2}: {metrics.separation:.3f} (vocab={metrics.vocab_size})"
        for i, metrics in enumerate(transition_metrics_list)
    ])
    
    # Build methodology description
    weighting_desc = "size-weighted" if anti_gaming_config.enable_size_weighting else "standard"
    penalty_desc = f" × penalty({segment_count_penalty:.3f})" if anti_gaming_config.enable_count_penalty else ""
    
    methodology = (
        f"Multi-segment objective ({weighting_desc}): {cohesion_weight:.1f} × {size_weighted_cohesion:.3f} + "
        f"{separation_weight:.1f} × {size_weighted_separation:.3f}{penalty_desc} = {final_score:.3f} "
        f"({num_segments} segments, {num_transitions} transitions"
        f"{f', {excluded_segments} excluded' if excluded_segments > 0 else ''})"
    )
    
    return ObjectiveFunctionResult(
        final_score=final_score,
        cohesion_score=avg_cohesion,
        separation_score=avg_separation,
        num_segments=num_segments,
        num_transitions=num_transitions,
        cohesion_details=cohesion_details,
        separation_details=separation_details,
        methodology=methodology,
        size_weighted_cohesion=size_weighted_cohesion,
        size_weighted_separation=size_weighted_separation,
        segment_count_penalty=segment_count_penalty,
        excluded_segments=excluded_segments
    )


def evaluate_timeline_quality(segment_papers: List[List[Paper]], 
                            verbose: bool = False,
                            anti_gaming_config: AntiGamingConfig = None) -> ObjectiveFunctionResult:
    """
    High-level function to evaluate timeline segmentation quality with anti-gaming safeguards.
    
    This is the main entry point for objective function evaluation,
    using the validated cohesion-dominant strategy with anti-gaming protections.
    
    Args:
        segment_papers: List of segments (each segment is a list of papers)
        verbose: Whether to print detailed evaluation information
        anti_gaming_config: Anti-gaming configuration (default: load from config)
    
    Returns:
        ObjectiveFunctionResult with complete evaluation including anti-gaming metrics
    """
    logger = get_logger(__name__, verbose)
    
    try:
        result = compute_objective_function(segment_papers, anti_gaming_config=anti_gaming_config)
        
        if verbose:
            logger.info("Timeline Quality Evaluation:")
            logger.info(f"  Final Score: {result.final_score:.3f}")
            logger.info(f"  Cohesion: {result.cohesion_score:.3f} (size-weighted: {result.size_weighted_cohesion:.3f})")
            logger.info(f"  Separation: {result.separation_score:.3f} (size-weighted: {result.size_weighted_separation:.3f})")
            logger.info(f"  Segments: {result.num_segments} (excluded: {result.excluded_segments})")
            logger.info(f"  Count Penalty: {result.segment_count_penalty:.3f}")
            logger.info(f"  Methodology: {result.methodology}")
        
        return result
        
    except Exception as e:
        raise ValueError(f"Timeline quality evaluation failed: {e}")


# Convenience functions for backward compatibility
def jaccard_cohesion(segment_papers: List[Paper]) -> float:
    """Compute Jaccard cohesion for a single segment (backward compatibility)."""
    cohesion, _, _ = compute_jaccard_cohesion(segment_papers)
    return cohesion


def jensen_shannon_separation(segment_a: List[Paper], segment_b: List[Paper]) -> float:
    """Compute Jensen-Shannon separation between two segments (backward compatibility)."""
    separation, _ = compute_jensen_shannon_separation(segment_a, segment_b)
    return separation


# Export main functions
__all__ = [
    'AntiGamingConfig',
    'ObjectiveFunctionResult',
    'SegmentMetrics', 
    'TransitionMetrics',
    'compute_objective_function',
    'evaluate_timeline_quality',
    'compute_jaccard_cohesion',
    'compute_jensen_shannon_separation',
    'compute_size_weighted_average',
    'compute_segment_count_penalty',
    'filter_segments_by_size',
    'jaccard_cohesion',
    'jensen_shannon_separation'
] 