"""
Data processing and model definitions.
"""

# Import essential data models only
from .data_models import (
    # Core temporal structures
    AcademicYear,
    AcademicPeriod,
    # Paper and domain models
    Paper,
    CitationRelation,
    DomainData,
    # Result models
    TimelineAnalysisResult,
)

# Import data processing functions
from .data_processing import (
    # Simplified data loading functions
    load_domain_data,
    create_academic_periods_from_segments,
    create_single_academic_period,
    # Utility functions
    load_papers_from_json,
    load_citation_graph,
    filter_papers_by_year_range,
    filter_papers_by_minimum_yearly_count,
    compute_academic_years,
)

# Export all
__all__ = [
    # Core temporal structures
    "AcademicYear",
    "AcademicPeriod",
    # Paper and domain models
    "Paper",
    "CitationRelation",
    "DomainData",
    # Result models
    "TimelineAnalysisResult",
    # Data loading functions
    "load_domain_data",
    "create_academic_periods_from_segments",
    "create_single_academic_period",
    # Utility functions
    "load_papers_from_json",
    "load_citation_graph",
    "filter_papers_by_year_range",
    "filter_papers_by_minimum_yearly_count",
    "compute_academic_years",
]
