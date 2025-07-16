"""Evaluation module for timeline segmentation quality assessment."""

from .evaluation import (
    # Core evaluation functions
    evaluate_timeline_result,
    run_final_evaluation,
    evaluate_domains,
    # Helper functions
    calculate_method_metrics_against_references,
    # Unified data models
    EvaluationResult,
    DomainEvaluationSummary,
)

from .baselines import (
    # Reference loading functions
    load_gemini_reference,
    load_perplexity_reference,
    # Baseline creation functions
    create_fixed_year_baseline,
    # Shared data loading and caching
    load_shared_academic_data,
    clear_cache,
    compute_config_hash,
)

from .metrics import (
    calculate_boundary_f1,
    calculate_segment_f1,
)

from .analysis import (
    save_evaluation_result,
    display_final_evaluation_summary,
    display_cross_domain_analysis,
)

__all__ = [
    # Core evaluation functions
    "evaluate_timeline_result",
    "run_final_evaluation",
    "evaluate_domains",
    # Helper functions
    "calculate_method_metrics_against_references",
    # Unified data models
    "EvaluationResult",
    "DomainEvaluationSummary",
    # Reference loading functions
    "load_gemini_reference",
    "load_perplexity_reference",
    # Baseline creation functions
    "create_fixed_year_baseline",
    # Shared data loading and caching
    "load_shared_academic_data",
    "clear_cache",
    "compute_config_hash",
    # Metrics
    "calculate_boundary_f1",
    "calculate_segment_f1",
    # Analysis
    "save_evaluation_result",
    "display_final_evaluation_summary",
    "display_cross_domain_analysis",
]
