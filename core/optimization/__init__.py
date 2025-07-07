# Analysis Sub-module
# Handles objective function evaluation and paper analysis

# Import objective function
from .objective_function import (
    # Main objective function (AcademicPeriod-based)
    compute_objective_function,
    ObjectiveFunctionResult,
    AntiGamingConfig,
    PeriodMetrics,
    TransitionMetrics,
    # Core metric functions
    evaluate_period_cohesion,
    evaluate_period_separation,
)

# Export all
__all__ = [
    # Main objective function (AcademicPeriod-based)
    "compute_objective_function",
    "ObjectiveFunctionResult",
    "AntiGamingConfig",
    "PeriodMetrics",
    "TransitionMetrics",
    # Core metric functions
    "evaluate_period_cohesion",
    "evaluate_period_separation",
]
