# Core package for timeline segmentation algorithm
# Simplified architecture with clean data flow

# Import main objective function for easy access
from .optimization.objective_function import (
    compute_objective_function,
    ObjectiveFunctionResult,
)

# Import key data models
from .data.data_models import (
    Paper,
    DomainData,
    AcademicYear,
    AcademicPeriod,
    TimelineAnalysisResult,
)

# Import simplified pipeline orchestration
from .pipeline.orchestrator import analyze_timeline

# Import simplified data processing functions
from .data.data_processing import (
    load_papers_from_json,
    load_domain_data,
    create_academic_periods_from_segments,
    create_single_academic_period,
)

# Import configuration management
from .utils.config import AlgorithmConfig

# Main exports for public API
__all__ = [
    # Core objective function and evaluation
    "compute_objective_function",
    "ObjectiveFunctionResult",
    # Data models and structures
    "Paper",
    "DomainData",
    "AcademicYear",
    "AcademicPeriod",
    "TimelineAnalysisResult",
    # Simplified pipeline entry points
    "analyze_timeline",
    # Simplified data processing and loading
    "load_papers_from_json",
    "load_domain_data",
    "create_academic_periods_from_segments",
    "create_single_academic_period",
    # Configuration management
    "AlgorithmConfig",
]
