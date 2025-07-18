"""Unified penalty system for timeline segmentation optimization.

This module provides a single, consistent penalty computation that can be used
across optimization and beam search to prevent gaming and provide bounded scores.
"""

import math
from typing import List, Dict, Any
from dataclasses import dataclass

from ..data.data_models import AcademicPeriod


@dataclass
class PenaltyConfig:
    """Configuration for unified penalty computation."""

    # Length constraints
    min_period_years: int
    max_period_years: int

    # Segment count constraints
    auto_n_upper: bool
    n_upper_buffer: int

    # Penalty weights
    lambda_short: float
    lambda_long: float
    lambda_count: float

    # Scaling parameters
    enable_scaling: bool
    scaling_factor: float


def compute_timeline_span(academic_periods: List[AcademicPeriod]) -> int:
    """Compute the total span of years covered by academic periods.

    Args:
        academic_periods: List of AcademicPeriod objects

    Returns:
        Total span in years
    """
    if not academic_periods:
        return 0

    min_year = min(period.start_year for period in academic_periods)
    max_year = max(period.end_year for period in academic_periods)

    return max_year - min_year + 1


def compute_auto_n_upper(
    academic_periods: List[AcademicPeriod], penalty_config: PenaltyConfig
) -> int:
    """Compute N_upper automatically based on timeline span.

    Args:
        academic_periods: List of AcademicPeriod objects
        penalty_config: Penalty configuration

    Returns:
        Automatically calculated N_upper value
    """
    timeline_span = compute_timeline_span(academic_periods)
    avg_target_length = (
        penalty_config.min_period_years + penalty_config.max_period_years
    ) / 2

    # Calculate ideal number of segments
    ideal_segments = math.ceil(timeline_span / avg_target_length)

    # Add buffer for flexibility
    return ideal_segments + penalty_config.n_upper_buffer


def compute_unified_penalty(
    academic_periods: List[AcademicPeriod], penalty_config: PenaltyConfig
) -> float:
    """Compute unified additive penalty for segmentation.

    This function computes a single penalty value that combines:
    1. Short period penalty: λ_short * Σ max(0, L_min - L_i)
    2. Long period penalty: λ_long * Σ max(0, L_i - L_max)
    3. Over-segmentation penalty: λ_count * max(0, N - N_upper)

    Args:
        academic_periods: List of AcademicPeriod objects to evaluate
        penalty_config: Penalty configuration

    Returns:
        Total penalty value (non-negative)
    """
    if not academic_periods:
        return 0.0

    num_segments = len(academic_periods)

    # Compute N_upper automatically
    n_upper = compute_auto_n_upper(academic_periods, penalty_config)

    # 1. Short period penalty
    short_penalty = 0.0
    for period in academic_periods:
        period_length = period.end_year - period.start_year + 1
        if period_length < penalty_config.min_period_years:
            short_penalty += penalty_config.lambda_short * (
                penalty_config.min_period_years - period_length
            )

    # 2. Long period penalty
    long_penalty = 0.0
    for period in academic_periods:
        period_length = period.end_year - period.start_year + 1
        if period_length > penalty_config.max_period_years:
            long_penalty += penalty_config.lambda_long * (
                period_length - penalty_config.max_period_years
            )

    # 3. Over-segmentation penalty
    count_penalty = 0.0
    if num_segments > n_upper:
        count_penalty = penalty_config.lambda_count * (num_segments - n_upper)

    total_penalty = short_penalty + long_penalty + count_penalty

    return total_penalty


def compute_scaled_objective(
    raw_objective: float, penalty_config: PenaltyConfig
) -> float:
    """Scale objective score to 0-1 range using sigmoid function.

    Args:
        raw_objective: Raw objective score (can be negative)
        penalty_config: Penalty configuration

    Returns:
        Scaled score in range [0, 1]
    """
    if not penalty_config.enable_scaling:
        return raw_objective

    # Use sigmoid function: 1 / (1 + exp(-k * raw))
    k = penalty_config.scaling_factor

    # Avoid overflow for very negative values
    if raw_objective * k < -500:
        return 0.0

    return 1.0 / (1.0 + math.exp(-k * raw_objective))


def compute_penalized_objective(
    base_objective: float,
    academic_periods: List[AcademicPeriod],
    penalty_config: PenaltyConfig,
) -> Dict[str, float]:
    """Compute final penalized and scaled objective score.

    Args:
        base_objective: Base objective score (cohesion + separation)
        academic_periods: List of AcademicPeriod objects
        penalty_config: Penalty configuration

    Returns:
        Dictionary with raw_score, penalty, penalized_score, scaled_score
    """
    penalty = compute_unified_penalty(academic_periods, penalty_config)
    penalized_score = base_objective - penalty
    scaled_score = compute_scaled_objective(penalized_score, penalty_config)

    return {
        "raw_score": base_objective,
        "penalty": penalty,
        "penalized_score": penalized_score,
        "scaled_score": scaled_score,
    }


def create_penalty_config_from_dict(config_dict: Dict[str, Any]) -> PenaltyConfig:
    """Create PenaltyConfig from configuration dictionary.

    Args:
        config_dict: Dictionary with penalty configuration

    Returns:
        PenaltyConfig instance
        
    Raises:
        KeyError: If required configuration parameters are missing
    """
    # Extract relevant parameters without defaults
    penalty_section = config_dict.get("penalty", {})
    
    if not penalty_section:
        raise KeyError("Missing 'penalty' section in configuration")
    
    required_params = [
        "min_period_years", "max_period_years", "auto_n_upper", "n_upper_buffer",
        "lambda_short", "lambda_long", "lambda_count", "enable_scaling", "scaling_factor"
    ]
    
    for param in required_params:
        if param not in penalty_section:
            raise KeyError(f"Missing required penalty parameter: {param}")

    return PenaltyConfig(
        min_period_years=penalty_section["min_period_years"],
        max_period_years=penalty_section["max_period_years"],
        auto_n_upper=penalty_section["auto_n_upper"],
        n_upper_buffer=penalty_section["n_upper_buffer"],
        lambda_short=penalty_section["lambda_short"],
        lambda_long=penalty_section["lambda_long"],
        lambda_count=penalty_section["lambda_count"],
        enable_scaling=penalty_section["enable_scaling"],
        scaling_factor=penalty_section["scaling_factor"],
    )


def create_penalty_config_from_algorithm_config(algorithm_config) -> PenaltyConfig:
    """Create PenaltyConfig from AlgorithmConfig.

    This ensures optimization uses the same penalty parameters as the main algorithm.

    Args:
        algorithm_config: AlgorithmConfig instance

    Returns:
        PenaltyConfig instance
        
    Raises:
        AttributeError: If required penalty attributes are missing from algorithm_config
    """
    required_attrs = [
        "penalty_min_period_years", "penalty_max_period_years", "penalty_auto_n_upper",
        "penalty_n_upper_buffer", "penalty_lambda_short", "penalty_lambda_long",
        "penalty_lambda_count", "penalty_enable_scaling", "penalty_scaling_factor"
    ]
    
    for attr in required_attrs:
        if not hasattr(algorithm_config, attr):
            raise AttributeError(f"Missing required algorithm config attribute: {attr}")

    return PenaltyConfig(
        min_period_years=algorithm_config.penalty_min_period_years,
        max_period_years=algorithm_config.penalty_max_period_years,
        auto_n_upper=algorithm_config.penalty_auto_n_upper,
        n_upper_buffer=algorithm_config.penalty_n_upper_buffer,
        lambda_short=algorithm_config.penalty_lambda_short,
        lambda_long=algorithm_config.penalty_lambda_long,
        lambda_count=algorithm_config.penalty_lambda_count,
        enable_scaling=algorithm_config.penalty_enable_scaling,
        scaling_factor=algorithm_config.penalty_scaling_factor,
    )
