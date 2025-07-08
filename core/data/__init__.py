"""Data processing and model definitions for timeline segmentation."""

from .data_models import (
    AcademicYear,
    AcademicPeriod,
    Paper,
    CitationRelation,
    TimelineAnalysisResult,
)

from .data_processing import (
    load_domain_data,
    create_academic_periods_from_segments,
    create_single_academic_period,
    load_papers_from_json,
    load_citation_graph,
    filter_papers_by_year_range,
    compute_academic_years,
)

__all__ = [
    "AcademicYear",
    "AcademicPeriod",
    "Paper",
    "CitationRelation",
    "TimelineAnalysisResult",
    "load_domain_data",
    "create_academic_periods_from_segments",
    "create_single_academic_period",
    "load_papers_from_json",
    "load_citation_graph",
    "filter_papers_by_year_range",
    "compute_academic_years",
]
