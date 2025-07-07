"""
Clean Objective Function Module

This module computes objective function scores for timeline segmentation quality.
Uses pre-computed AcademicPeriod data structures to eliminate redundant computations.

Core functionality:
- Period cohesion evaluation using Jaccard similarity (ORIGINAL ALGORITHM)
- Period separation evaluation using Jensen-Shannon divergence (ORIGINAL ALGORITHM)
- Anti-gaming mechanisms to prevent micro-segmentation abuse
- Comprehensive scoring with configurable weights

Follows fail-fast principles with strict error handling throughout.
"""

import os
import json
import math
from typing import Dict, List, NamedTuple, Tuple
from collections import Counter
import numpy as np
from scipy.spatial.distance import jensenshannon

from ..data.data_models import AcademicPeriod


# =============================================================================
# DATA STRUCTURES
# =============================================================================


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


class PeriodMetrics(NamedTuple):
    """Metrics for a single academic period."""

    cohesion: float
    size: int
    keywords_count: int
    top_keywords: List[str]


class TransitionMetrics(NamedTuple):
    """Metrics for a transition between two periods."""

    separation: float
    vocab_size: int
    period_a_size: int
    period_b_size: int


# =============================================================================
# ANTI-GAMING UTILITIES
# =============================================================================


def compute_size_weighted_average(
    segment_scores: List[float], segment_sizes: List[int], power: float = 0.5
) -> float:
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
    weights = [size**power for size in segment_sizes]
    total_weight = sum(weights)

    if total_weight == 0:
        return 0.0

    # Weighted average
    weighted_sum = sum(score * weight for score, weight in zip(segment_scores, weights))
    return weighted_sum / total_weight


def compute_segment_count_penalty(
    num_segments: int, domain_year_span: int, sigma: float = 4.0
) -> float:
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


def filter_periods_by_size(
    academic_periods: List[AcademicPeriod], min_size: int
) -> Tuple[List[AcademicPeriod], int]:
    """
    Filter out periods below minimum size threshold.

    Args:
        academic_periods: List of AcademicPeriod objects
        min_size: Minimum papers per period

    Returns:
        Tuple of (filtered_periods, excluded_count)
    """
    filtered_periods = []
    excluded_count = 0

    for period in academic_periods:
        if period.total_papers >= min_size:
            filtered_periods.append(period)
        else:
            excluded_count += 1

    return filtered_periods, excluded_count


# =============================================================================
# CORE METRIC COMPUTATION (ORIGINAL ALGORITHM RESTORED)
# =============================================================================


def evaluate_period_cohesion(
    academic_period: AcademicPeriod, top_k: int
) -> PeriodMetrics:
    """
    Evaluate cohesion for an AcademicPeriod using Jaccard similarity (ORIGINAL ALGORITHM).

    Args:
        academic_period: AcademicPeriod with pre-computed keyword data
        top_k: Number of top keywords to use

    Returns:
        PeriodMetrics with cohesion score and metadata
    """
    # Use pre-computed keyword frequencies
    keyword_frequencies = academic_period.combined_keyword_frequencies

    if not keyword_frequencies:
        return PeriodMetrics(
            cohesion=0.0,
            size=academic_period.total_papers,
            keywords_count=0,
            top_keywords=[],
        )

    # Get top-K keywords by frequency
    top_keywords_items = sorted(
        keyword_frequencies.items(), key=lambda x: x[1], reverse=True
    )[:top_k]
    top_keywords_set = {kw for kw, count in top_keywords_items}
    top_keywords = [kw for kw, count in top_keywords_items]

    if not top_keywords_set:
        return PeriodMetrics(
            cohesion=0.0,
            size=academic_period.total_papers,
            keywords_count=len(keyword_frequencies),
            top_keywords=[],
        )

    # Calculate Jaccard cohesion (ORIGINAL ALGORITHM)
    jaccard_scores = []

    for paper in academic_period.get_all_papers():
        paper_keywords = set(paper.keywords) if paper.keywords else set()

        if paper_keywords & top_keywords_set:  # Paper has at least one defining keyword
            intersection = len(paper_keywords & top_keywords_set)
            union = len(paper_keywords | top_keywords_set)

            if union > 0:
                jaccard = intersection / union
                jaccard_scores.append(jaccard)

    if not jaccard_scores:
        cohesion = 0.0
    else:
        cohesion = float(np.mean(jaccard_scores))

    return PeriodMetrics(
        cohesion=cohesion,
        size=academic_period.total_papers,
        keywords_count=len(keyword_frequencies),
        top_keywords=top_keywords,
    )


def evaluate_period_separation(
    period_a: AcademicPeriod, period_b: AcademicPeriod
) -> TransitionMetrics:
    """
    Evaluate separation between two AcademicPeriods using Jensen-Shannon divergence (ORIGINAL ALGORITHM).

    Args:
        period_a: First AcademicPeriod
        period_b: Second AcademicPeriod

    Returns:
        TransitionMetrics with separation score and metadata
    """
    # Collect keywords from both periods
    keywords_a = []
    keywords_b = []

    for paper in period_a.get_all_papers():
        if paper.keywords:
            keywords_a.extend(paper.keywords)

    for paper in period_b.get_all_papers():
        if paper.keywords:
            keywords_b.extend(paper.keywords)

    if not keywords_a or not keywords_b:
        return TransitionMetrics(
            separation=0.0,
            vocab_size=0,
            period_a_size=period_a.total_papers,
            period_b_size=period_b.total_papers,
        )

    # Create unified vocabulary
    vocab = list(set(keywords_a) | set(keywords_b))
    vocab_size = len(vocab)

    if vocab_size == 0:
        return TransitionMetrics(
            separation=0.0,
            vocab_size=0,
            period_a_size=period_a.total_papers,
            period_b_size=period_b.total_papers,
        )

    # Compute frequency distributions
    p = get_frequency_distribution(keywords_a, vocab)
    q = get_frequency_distribution(keywords_b, vocab)

    # Compute Jensen-Shannon divergence (ORIGINAL ALGORITHM)
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
    separation = float(js_divergence) if not np.isnan(js_divergence) else 0.0

    return TransitionMetrics(
        separation=separation,
        vocab_size=vocab_size,
        period_a_size=period_a.total_papers,
        period_b_size=period_b.total_papers,
    )


def get_frequency_distribution(
    keywords: List[str], vocabulary: List[str]
) -> np.ndarray:
    """
    Get frequency distribution for keywords over vocabulary.

    Args:
        keywords: List of keywords (with potential duplicates)
        vocabulary: Complete vocabulary (unique keywords)

    Returns:
        Frequency distribution array
    """
    if not vocabulary:
        return np.array([])

    # Count keyword frequencies
    keyword_counts = Counter(keywords)

    # Build distribution vector
    total = sum(keyword_counts.values())
    if total == 0:
        return np.ones(len(vocabulary)) / len(vocabulary)  # Uniform distribution

    return np.array([keyword_counts[word] / total for word in vocabulary])


def compute_objective_function(
    academic_periods: List[AcademicPeriod],
    algorithm_config,
    verbose: bool = False,
) -> ObjectiveFunctionResult:
    """
    Compute objective function for academic periods with anti-gaming mechanisms.

    This is the main entry point for objective function evaluation. Uses configuration
    from AlgorithmConfig to eliminate redundant parameter loading.

    Args:
        academic_periods: List of AcademicPeriod objects to evaluate
        algorithm_config: AlgorithmConfig with all necessary parameters
        verbose: Enable verbose logging output

    Returns:
        ObjectiveFunctionResult with comprehensive evaluation metrics

    Raises:
        ValueError: If academic_periods is empty or algorithm_config is invalid
    """
    if not academic_periods:
        raise ValueError("academic_periods cannot be empty")

    if not algorithm_config:
        raise ValueError("algorithm_config cannot be None")

    from ..utils.logging import get_logger

    logger = get_logger(__name__, verbose)

    # Extract configuration parameters
    cohesion_weight = algorithm_config.cohesion_weight
    separation_weight = algorithm_config.separation_weight
    top_k = algorithm_config.top_k_keywords
    anti_gaming_config = algorithm_config.get_anti_gaming_config()

    if verbose:
        logger.info(f"Computing objective function for {len(academic_periods)} periods")
        logger.info(
            f"Weights: cohesion={cohesion_weight}, separation={separation_weight}"
        )
        logger.info(f"Top-K keywords: {top_k}")
        logger.info(f"Anti-gaming config: {anti_gaming_config}")

    # Apply period size filtering if enabled
    if anti_gaming_config.enable_segment_floor:
        filtered_periods, excluded_count = filter_periods_by_size(
            academic_periods, anti_gaming_config.min_segment_size
        )

        if verbose and excluded_count > 0:
            logger.info(f"Excluded {excluded_count} periods below size threshold")

        # Use filtered periods for evaluation
        evaluation_periods = filtered_periods
    else:
        evaluation_periods = academic_periods
        excluded_count = 0

    if not evaluation_periods:
        if verbose:
            logger.warning("No periods remaining after filtering")
        return ObjectiveFunctionResult(
            final_score=0.0,
            cohesion_score=0.0,
            separation_score=0.0,
            num_segments=0,
            num_transitions=0,
            cohesion_details="No valid segments",
            separation_details="No valid transitions",
            methodology="Anti-gaming filtering",
            excluded_segments=excluded_count,
        )

    # Handle single period case
    if len(evaluation_periods) == 1:
        period_metrics = evaluate_period_cohesion(evaluation_periods[0], top_k)

        # For single period, final score is just cohesion (no separation)
        final_score = cohesion_weight * period_metrics.cohesion

        methodology = (
            f"Single period objective: {cohesion_weight:.1f} × {period_metrics.cohesion:.3f} "
            f"(cohesion only) = {final_score:.3f}"
        )

        return ObjectiveFunctionResult(
            final_score=final_score,
            cohesion_score=period_metrics.cohesion,
            separation_score=0.0,
            num_segments=1,
            num_transitions=0,
            cohesion_details=f"Single period: {period_metrics.size} papers, {period_metrics.keywords_count} keywords",
            separation_details="No transitions in single period",
            methodology=methodology,
            size_weighted_cohesion=period_metrics.cohesion,
            size_weighted_separation=0.0,
            segment_count_penalty=1.0,
            excluded_segments=excluded_count,
        )

    # Multiple periods case
    period_metrics_list = []
    transition_metrics_list = []

    # Evaluate cohesion for each period
    for i, period in enumerate(evaluation_periods):
        try:
            metrics = evaluate_period_cohesion(period, top_k)
            period_metrics_list.append(metrics)
            if verbose:
                logger.info(
                    f"Period {i+1} cohesion: {metrics.cohesion:.3f} ({metrics.size} papers)"
                )
        except Exception as e:
            raise ValueError(f"Failed to evaluate cohesion for period {i+1}: {e}")

    # Evaluate separation for each transition
    for i in range(len(evaluation_periods) - 1):
        try:
            metrics = evaluate_period_separation(
                evaluation_periods[i], evaluation_periods[i + 1]
            )
            transition_metrics_list.append(metrics)
            if verbose:
                logger.info(
                    f"Transition {i+1}→{i+2} separation: {metrics.separation:.3f}"
                )
        except Exception as e:
            raise ValueError(
                f"Failed to evaluate separation for transition {i+1}→{i+2}: {e}"
            )

    # Standard aggregation
    cohesion_scores = [metrics.cohesion for metrics in period_metrics_list]
    separation_scores = [metrics.separation for metrics in transition_metrics_list]

    avg_cohesion = float(np.mean(cohesion_scores))
    avg_separation = float(np.mean(separation_scores)) if separation_scores else 0.0

    # Anti-gaming: Size-weighted aggregation
    size_weighted_cohesion = avg_cohesion
    size_weighted_separation = avg_separation

    if anti_gaming_config.enable_size_weighting:
        # Size-weighted cohesion
        period_sizes = [metrics.size for metrics in period_metrics_list]
        if cohesion_scores and period_sizes:
            size_weighted_cohesion = compute_size_weighted_average(
                cohesion_scores, period_sizes, anti_gaming_config.size_weight_power
            )

        # Size-weighted separation (using geometric mean of adjacent period sizes)
        if separation_scores and len(period_sizes) > 1:
            transition_sizes = []
            for i in range(len(period_sizes) - 1):
                transition_size = math.sqrt(period_sizes[i] * period_sizes[i + 1])
                transition_sizes.append(transition_size)

            size_weighted_separation = compute_size_weighted_average(
                separation_scores,
                transition_sizes,
                anti_gaming_config.size_weight_power,
            )

    # Segment count penalty
    segment_count_penalty = 1.0
    if anti_gaming_config.enable_count_penalty:
        # Calculate domain year span from academic periods
        if evaluation_periods:
            min_year = min(period.start_year for period in evaluation_periods)
            max_year = max(period.end_year for period in evaluation_periods)
            domain_year_span = max_year - min_year + 1
            segment_count_penalty = compute_segment_count_penalty(
                len(evaluation_periods),
                domain_year_span,
                anti_gaming_config.segment_count_penalty_sigma,
            )

    # Compute final objective score with anti-gaming
    if anti_gaming_config.enable_size_weighting:
        final_score = (
            cohesion_weight * size_weighted_cohesion
            + separation_weight * size_weighted_separation
        ) * segment_count_penalty
    else:
        final_score = (
            cohesion_weight * avg_cohesion + separation_weight * avg_separation
        ) * segment_count_penalty

    # Create detailed explanations
    cohesion_details = " | ".join(
        [
            f"P{i+1}: {metrics.cohesion:.3f} ({metrics.size}p, top: {', '.join(metrics.top_keywords[:3])})"
            for i, metrics in enumerate(period_metrics_list)
        ]
    )

    separation_details = " | ".join(
        [
            f"T{i+1}→{i+2}: {metrics.separation:.3f} (vocab={metrics.vocab_size})"
            for i, metrics in enumerate(transition_metrics_list)
        ]
    )

    # Build methodology description
    weighting_desc = (
        "size-weighted" if anti_gaming_config.enable_size_weighting else "standard"
    )
    penalty_desc = (
        f" × penalty({segment_count_penalty:.3f})"
        if anti_gaming_config.enable_count_penalty
        else ""
    )

    methodology = (
        f"Multi-period objective ({weighting_desc}): {cohesion_weight:.1f} × {size_weighted_cohesion:.3f} + "
        f"{separation_weight:.1f} × {size_weighted_separation:.3f}{penalty_desc} = {final_score:.3f} "
        f"({len(evaluation_periods)} periods, {len(transition_metrics_list)} transitions"
        f"{f', {excluded_count} excluded' if excluded_count > 0 else ''})"
    )

    if verbose:
        logger.info(f"Final objective score: {final_score:.3f}")

    return ObjectiveFunctionResult(
        final_score=final_score,
        cohesion_score=avg_cohesion,
        separation_score=avg_separation,
        num_segments=len(evaluation_periods),
        num_transitions=len(transition_metrics_list),
        cohesion_details=cohesion_details,
        separation_details=separation_details,
        methodology=methodology,
        size_weighted_cohesion=size_weighted_cohesion,
        size_weighted_separation=size_weighted_separation,
        segment_count_penalty=segment_count_penalty,
        excluded_segments=excluded_count,
    )
