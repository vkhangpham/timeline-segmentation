"""Objective function computation for timeline segmentation quality.

This module computes objective function scores using pre-computed AcademicPeriod
data structures for period cohesion and separation evaluation.
"""

from typing import List, NamedTuple, Dict, Any, Optional
from collections import Counter
import numpy as np
from scipy.spatial.distance import jensenshannon

from ..data.data_models import AcademicPeriod
from .penalty import PenaltyConfig, compute_penalized_objective, create_penalty_config_from_dict


class ObjectiveFunctionResult(NamedTuple):
    """Complete result of objective function evaluation."""

    final_score: float
    raw_score: float  # Before penalty
    penalty: float
    scaled_score: float  # 0-1 scaled score
    cohesion_score: float
    separation_score: float
    num_segments: int
    num_transitions: int
    cohesion_details: str
    separation_details: str
    methodology: str


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


def evaluate_period_cohesion(
    academic_period: AcademicPeriod, top_k: int
) -> PeriodMetrics:
    """Evaluate cohesion for an AcademicPeriod using 1-JSD between years and period.

    Args:
        academic_period: AcademicPeriod with pre-computed keyword data
        top_k: Number of top keywords to use (for consistency with original interface)

    Returns:
        PeriodMetrics with cohesion score and metadata
    """
    # If only one year, perfect cohesion
    if len(academic_period.academic_years) <= 1:
        return PeriodMetrics(
            cohesion=1.0,
            size=academic_period.total_papers,
            keywords_count=len(academic_period.combined_keyword_frequencies),
            top_keywords=list(academic_period.top_keywords[:top_k]),
        )

    # Collect all keywords from all years to create period vocabulary
    all_keywords = []
    for academic_year in academic_period.academic_years:
        for paper in academic_year.papers:
            if paper.keywords:
                all_keywords.extend(paper.keywords)

    if not all_keywords:
        return PeriodMetrics(
            cohesion=0.0,
            size=academic_period.total_papers,
            keywords_count=0,
            top_keywords=[],
        )

    # Create vocabulary from all keywords in the period
    period_vocab = list(set(all_keywords))
    
    # Get overall period keyword distribution
    period_distribution = get_frequency_distribution(all_keywords, period_vocab)

    # Calculate JSD between each year and the overall period
    jsd_scores = []
    
    for academic_year in academic_period.academic_years:
        year_keywords = []
        for paper in academic_year.papers:
            if paper.keywords:
                year_keywords.extend(paper.keywords)
        
        if not year_keywords:
            continue
            
        year_distribution = get_frequency_distribution(year_keywords, period_vocab)
        
        # Add small epsilon to avoid issues with zero probabilities
        epsilon = 1e-10
        year_dist_smooth = year_distribution + epsilon
        period_dist_smooth = period_distribution + epsilon
        
        # Normalize to ensure they sum to 1
        year_dist_smooth = year_dist_smooth / year_dist_smooth.sum()
        period_dist_smooth = period_dist_smooth / period_dist_smooth.sum()
        
        # Compute Jensen-Shannon divergence
        js_divergence = jensenshannon(year_dist_smooth, period_dist_smooth, base=2)
        
        if not np.isnan(js_divergence):
            jsd_scores.append(js_divergence)

    if not jsd_scores:
        cohesion = 0.0
    else:
        # Return 1 - average_JSD (so higher values indicate better cohesion)
        avg_jsd = float(np.mean(jsd_scores))
        cohesion = 1.0 - avg_jsd

    # Get top keywords for metadata (using period-level data)
    top_keywords_items = sorted(
        academic_period.combined_keyword_frequencies.items(), 
        key=lambda x: x[1], reverse=True
    )[:top_k]
    top_keywords = [kw for kw, count in top_keywords_items]

    return PeriodMetrics(
        cohesion=cohesion,
        size=academic_period.total_papers,
        keywords_count=len(academic_period.combined_keyword_frequencies),
        top_keywords=top_keywords,
    )


def evaluate_period_separation(
    period_a: AcademicPeriod, period_b: AcademicPeriod
) -> TransitionMetrics:
    """Evaluate separation between two AcademicPeriods using Jensen-Shannon divergence.

    Args:
        period_a: First AcademicPeriod
        period_b: Second AcademicPeriod

    Returns:
        TransitionMetrics with separation score and metadata
    """
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

    vocab = list(set(keywords_a) | set(keywords_b))
    vocab_size = len(vocab)

    if vocab_size == 0:
        return TransitionMetrics(
            separation=0.0,
            vocab_size=0,
            period_a_size=period_a.total_papers,
            period_b_size=period_b.total_papers,
        )

    p = get_frequency_distribution(keywords_a, vocab)
    q = get_frequency_distribution(keywords_b, vocab)

    epsilon = 1e-10
    p = p + epsilon
    q = q + epsilon

    p = p / p.sum()
    q = q / q.sum()

    js_divergence = jensenshannon(p, q, base=2)

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
    """Get frequency distribution for keywords over vocabulary.

    Args:
        keywords: List of keywords (with potential duplicates)
        vocabulary: Complete vocabulary (unique keywords)

    Returns:
        Frequency distribution array
    """
    if not vocabulary:
        return np.array([])

    keyword_counts = Counter(keywords)

    total = sum(keyword_counts.values())
    if total == 0:
        return np.ones(len(vocabulary)) / len(vocabulary)

    return np.array([keyword_counts[word] / total for word in vocabulary])


def compute_objective_function(
    academic_periods: List[AcademicPeriod],
    algorithm_config,
    penalty_config: Optional[PenaltyConfig] = None,
    verbose: bool = False,
) -> ObjectiveFunctionResult:
    """Compute objective function for academic periods with unified penalty system.

    Args:
        academic_periods: List of AcademicPeriod objects to evaluate
        algorithm_config: AlgorithmConfig with cohesion/separation weights and top_k_keywords
        penalty_config: Optional PenaltyConfig; if None, created from algorithm_config
        verbose: Enable verbose logging output

    Returns:
        ObjectiveFunctionResult with cohesion, separation, and penalty evaluation

    Raises:
        ValueError: If academic_periods is empty or algorithm_config is invalid
    """
    if not academic_periods:
        raise ValueError("academic_periods cannot be empty")

    if not algorithm_config:
        raise ValueError("algorithm_config cannot be None")

    from ..utils.logging import get_logger

    logger = get_logger(__name__, verbose)

    # Create penalty config if not provided
    if penalty_config is None:
        # Convert algorithm_config to dict for penalty config creation
        config_dict = {"penalty": {}}
        penalty_config = create_penalty_config_from_dict(config_dict)

    cohesion_weight = algorithm_config.cohesion_weight
    separation_weight = algorithm_config.separation_weight
    top_k = algorithm_config.top_k_keywords

    if verbose:
        logger.info(f"Computing objective function for {len(academic_periods)} periods")
        logger.info(
            f"Weights: cohesion={cohesion_weight}, separation={separation_weight}"
        )
        logger.info(f"Top-K keywords: {top_k}")

    if len(academic_periods) == 1:
        period_metrics = evaluate_period_cohesion(academic_periods[0], top_k)

        base_score = cohesion_weight * period_metrics.cohesion

        # Apply unified penalty system
        penalty_result = compute_penalized_objective(
            base_score, academic_periods, penalty_config
        )

        methodology = (
            f"Single period objective: {cohesion_weight:.1f} × {period_metrics.cohesion:.3f} "
            f"(cohesion only) = {base_score:.3f}, penalty = {penalty_result['penalty']:.3f}, "
            f"final = {penalty_result['penalized_score']:.3f}, scaled = {penalty_result['scaled_score']:.3f}"
        )

        return ObjectiveFunctionResult(
            final_score=penalty_result["penalized_score"],
            raw_score=base_score,
            penalty=penalty_result["penalty"],
            scaled_score=penalty_result["scaled_score"],
            cohesion_score=period_metrics.cohesion,
            separation_score=0.0,
            num_segments=1,
            num_transitions=0,
            cohesion_details=f"Single period: {period_metrics.size} papers, {period_metrics.keywords_count} keywords",
            separation_details="No transitions in single period",
            methodology=methodology,
        )

    period_metrics_list = []
    transition_metrics_list = []

    for i, period in enumerate(academic_periods):
        try:
            metrics = evaluate_period_cohesion(period, top_k)
            period_metrics_list.append(metrics)
            if verbose:
                logger.info(
                    f"Period {i+1} cohesion: {metrics.cohesion:.3f} ({metrics.size} papers)"
                )
        except Exception as e:
            raise ValueError(f"Failed to evaluate cohesion for period {i+1}: {e}")

    for i in range(len(academic_periods) - 1):
        try:
            metrics = evaluate_period_separation(
                academic_periods[i], academic_periods[i + 1]
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

    cohesion_scores = [metrics.cohesion for metrics in period_metrics_list]
    separation_scores = [metrics.separation for metrics in transition_metrics_list]

    avg_cohesion = float(np.mean(cohesion_scores))
    avg_separation = float(np.mean(separation_scores)) if separation_scores else 0.0

    base_score = cohesion_weight * avg_cohesion + separation_weight * avg_separation

    # Apply unified penalty system
    penalty_result = compute_penalized_objective(
        base_score, academic_periods, penalty_config
    )

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

    methodology = (
        f"Multi-period objective: {cohesion_weight:.1f} × {avg_cohesion:.3f} + "
        f"{separation_weight:.1f} × {avg_separation:.3f} = {base_score:.3f}, "
        f"penalty = {penalty_result['penalty']:.3f}, final = {penalty_result['penalized_score']:.3f}, "
        f"scaled = {penalty_result['scaled_score']:.3f} "
        f"({len(academic_periods)} periods, {len(transition_metrics_list)} transitions)"
    )

    if verbose:
        logger.info(f"Final objective score: {penalty_result['penalized_score']:.3f}")
        logger.info(f"Scaled score: {penalty_result['scaled_score']:.3f}")

    return ObjectiveFunctionResult(
        final_score=penalty_result["penalized_score"],
        raw_score=base_score,
        penalty=penalty_result["penalty"],
        scaled_score=penalty_result["scaled_score"],
        cohesion_score=avg_cohesion,
        separation_score=avg_separation,
        num_segments=len(academic_periods),
        num_transitions=len(transition_metrics_list),
        cohesion_details=cohesion_details,
        separation_details=separation_details,
        methodology=methodology,
    )
