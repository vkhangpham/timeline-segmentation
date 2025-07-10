"""Objective function evaluation and optimization utilities."""

from .objective_function import (
    compute_objective_function,
    ObjectiveFunctionResult,
    PeriodMetrics,
    TransitionMetrics,
    evaluate_period_cohesion,
    evaluate_period_separation,
)

from .optimization import (
    load_cached_academic_years,
    create_trial_config,
    compute_penalty,
    get_validation_metrics,
    score_trial,
)

__all__ = [
    "compute_objective_function",
    "ObjectiveFunctionResult",
    "PeriodMetrics",
    "TransitionMetrics",
    "evaluate_period_cohesion",
    "evaluate_period_separation",
    "compute_penalty",
    "get_validation_metrics",
    "score_trial",
    "load_cached_academic_years",
    "create_trial_config",
]
