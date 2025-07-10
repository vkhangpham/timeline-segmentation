"""Evaluation module for timeline segmentation quality assessment."""

from .evaluation import (
    evaluate_timeline_result,
    run_comprehensive_evaluation,
    run_single_evaluation,
    run_all_domains_evaluation,
    run_baseline_only_evaluation,
    load_penalty_configuration,
    compute_penalty,
    EvaluationResult,
    BaselineResult,
    AutoMetricResult,
    ComprehensiveEvaluationResult,
)

from .baselines import (
    create_gemini_baseline,
    create_manual_baseline,
    create_fixed_year_baseline,
)

from .metrics import (
    calculate_boundary_f1,
    calculate_segment_f1,
    calculate_f1_score_between_methods,
)

from .analysis import (
    save_evaluation_result,
    display_evaluation_summary,
    display_cross_domain_analysis,
    save_cross_domain_results,
)

__all__ = [
    # Core evaluation functions
    "evaluate_timeline_result",
    "run_comprehensive_evaluation",
    "run_single_evaluation",
    "run_all_domains_evaluation",
    "run_baseline_only_evaluation",
    # Configuration
    "load_penalty_configuration",
    "compute_penalty",
    # Models
    "EvaluationResult",
    "BaselineResult",
    "AutoMetricResult",
    "ComprehensiveEvaluationResult",
    # Baselines
    "create_gemini_baseline",
    "create_manual_baseline",
    "create_fixed_year_baseline",
    # Metrics
    "calculate_boundary_f1",
    "calculate_segment_f1",
    "calculate_f1_score_between_methods",
    # Analysis
    "save_evaluation_result",
    "display_evaluation_summary",
    "display_cross_domain_analysis",
    "save_cross_domain_results",
]
