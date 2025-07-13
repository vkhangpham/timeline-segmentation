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
    min_period_years: int = 3
    max_period_years: int = 14
    
    # Segment count constraints
    auto_n_upper: bool = True
    n_upper_buffer: int = 1
    target_segments_upper: int = 8  # Used only if auto_n_upper is False
    
    # Penalty weights
    lambda_short: float = 0.05
    lambda_long: float = 0.03
    lambda_count: float = 0.02
    
    # Scaling parameters
    enable_scaling: bool = True
    scaling_factor: float = 2.0  # For sigmoid scaling: 1/(1+exp(-k*raw))


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
    academic_periods: List[AcademicPeriod], 
    penalty_config: PenaltyConfig
) -> int:
    """Compute N_upper automatically based on timeline span.
    
    Args:
        academic_periods: List of AcademicPeriod objects
        penalty_config: Penalty configuration
        
    Returns:
        Automatically calculated N_upper value
    """
    timeline_span = compute_timeline_span(academic_periods)
    avg_target_length = (penalty_config.min_period_years + penalty_config.max_period_years) / 2
    
    # Calculate ideal number of segments
    ideal_segments = math.ceil(timeline_span / avg_target_length)
    
    # Add buffer for flexibility
    return ideal_segments + penalty_config.n_upper_buffer


def compute_unified_penalty(
    academic_periods: List[AcademicPeriod],
    penalty_config: PenaltyConfig
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
    
    # Compute N_upper (automatically or from config)
    if penalty_config.auto_n_upper:
        n_upper = compute_auto_n_upper(academic_periods, penalty_config)
    else:
        n_upper = penalty_config.target_segments_upper
    
    # 1. Short period penalty
    short_penalty = 0.0
    for period in academic_periods:
        period_length = period.end_year - period.start_year + 1
        if period_length < penalty_config.min_period_years:
            short_penalty += penalty_config.lambda_short * (penalty_config.min_period_years - period_length)
    
    # 2. Long period penalty
    long_penalty = 0.0
    for period in academic_periods:
        period_length = period.end_year - period.start_year + 1
        if period_length > penalty_config.max_period_years:
            long_penalty += penalty_config.lambda_long * (period_length - penalty_config.max_period_years)
    
    # 3. Over-segmentation penalty
    count_penalty = 0.0
    if num_segments > n_upper:
        count_penalty = penalty_config.lambda_count * (num_segments - n_upper)
    
    total_penalty = short_penalty + long_penalty + count_penalty
    
    return total_penalty


def compute_scaled_objective(
    raw_objective: float,
    penalty_config: PenaltyConfig
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
    penalty_config: PenaltyConfig
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
        "scaled_score": scaled_score
    }


def create_penalty_config_from_dict(config_dict: Dict[str, Any]) -> PenaltyConfig:
    """Create PenaltyConfig from configuration dictionary.
    
    Args:
        config_dict: Dictionary with penalty configuration
        
    Returns:
        PenaltyConfig instance
    """
    # Extract relevant parameters with defaults
    penalty_section = config_dict.get("penalty", {})
    
    return PenaltyConfig(
        min_period_years=penalty_section.get("min_period_years", 3),
        max_period_years=penalty_section.get("max_period_years", 14),
        auto_n_upper=penalty_section.get("auto_n_upper", True),
        n_upper_buffer=penalty_section.get("n_upper_buffer", 1),
        target_segments_upper=penalty_section.get("target_segments_upper", 8),
        lambda_short=penalty_section.get("lambda_short", 0.05),
        lambda_long=penalty_section.get("lambda_long", 0.03),
        lambda_count=penalty_section.get("lambda_count", 0.02),
        enable_scaling=penalty_section.get("enable_scaling", True),
        scaling_factor=penalty_section.get("scaling_factor", 2.0)
    ) 