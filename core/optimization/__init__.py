"""Optimization module for timeline segmentation parameter tuning.

This module provides tools for optimizing segmentation parameters using
Bayesian optimization with cached data loading and parallel execution.
"""

from .objective_function import (
    compute_objective_function,
    ObjectiveFunctionResult,
    PeriodMetrics,
    TransitionMetrics,
)
from .optimization import (
    load_cached_academic_years,
    create_trial_config,
    score_trial,
    get_validation_metrics,
    clear_cache,
)
from .penalty import (
    PenaltyConfig,
    compute_unified_penalty,
    compute_penalized_objective,
    compute_scaled_objective,
    create_penalty_config_from_dict,
    create_penalty_config_from_algorithm_config,
)
from .bayesian_optimizer import run_bayesian_optimization
from .optimization_config import load_config, get_parameter_space

__all__ = [
    # Objective function components
    "compute_objective_function",
    "ObjectiveFunctionResult",
    "PeriodMetrics",
    "TransitionMetrics",
    # Optimization utilities
    "load_cached_academic_years",
    "create_trial_config",
    "score_trial",
    "get_validation_metrics",
    "clear_cache",
    # Unified penalty system
    "PenaltyConfig",
    "compute_unified_penalty",
    "compute_penalized_objective",
    "compute_scaled_objective",
    "create_penalty_config_from_dict",
    "create_penalty_config_from_algorithm_config",
    # Optimization components
    "run_bayesian_optimization",
    "load_config",
    "get_parameter_space",
]
