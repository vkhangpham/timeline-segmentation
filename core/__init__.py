"""Core package for timeline segmentation algorithm."""

from .optimization.objective_function import (
    compute_objective_function,
    ObjectiveFunctionResult,
)

from .data.data_models import (
    Paper,
    AcademicYear,
    AcademicPeriod,
    TimelineAnalysisResult,
)

from .pipeline.orchestrator import analyze_timeline

from .data.data_processing import (
    load_papers_from_json,
    load_domain_data,
    create_academic_periods_from_segments,
    create_single_academic_period,
)

from .utils.config import AlgorithmConfig

__all__ = [
    # Objective function and evaluation
    "compute_objective_function",
    "ObjectiveFunctionResult",
    # Data models and structures
    "Paper",
    "AcademicYear",
    "AcademicPeriod",
    "TimelineAnalysisResult",
    # Pipeline entry points
    "analyze_timeline",
    # Data processing and loading
    "load_papers_from_json",
    "load_domain_data",
    "create_academic_periods_from_segments",
    "create_single_academic_period",
    # Configuration management
    "AlgorithmConfig",
]
