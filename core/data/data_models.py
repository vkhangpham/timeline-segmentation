"""
Core data models for scientific publication analysis.

This module defines essential immutable data structures for papers, citations, and temporal data
using functional programming principles. Only includes models that are absolutely necessary.

REFACTORED: Simplified object flow using only AcademicYear and AcademicPeriod as core structures.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any, TYPE_CHECKING
from datetime import datetime

if TYPE_CHECKING:
    from typing import Tuple as TupleType


# ============================================================================
# PAPER AND DOMAIN MODELS (defined first for forward references)
# ============================================================================


@dataclass(frozen=True)
class Paper:
    """Immutable representation of a scientific paper."""

    id: str
    title: str
    content: str
    pub_year: int
    cited_by_count: int
    keywords: Tuple[str, ...]  # Use tuple for immutability
    children: Tuple[str, ...]  # Citing papers
    description: str

    def __post_init__(self):
        """Validate paper data after initialization."""
        if (
            self.pub_year < 1400 or self.pub_year > 2050
        ):  # More lenient range for historical data
            raise ValueError(f"Invalid publication year: {self.pub_year}")
        if self.cited_by_count < 0:
            raise ValueError(f"Invalid citation count: {self.cited_by_count}")


# ============================================================================
# CORE TEMPORAL DATA STRUCTURES
# ============================================================================


@dataclass(frozen=True)
class AcademicYear:
    """
    Pre-computed temporal data structure for a single academic year.

    This structure is computed once during data loading and contains all
    year-level aggregations needed by the algorithm, eliminating repeated
    computations in change detection, objective function, and segmentation.

    Keywords are processed once during creation using the top-K frequency approach.
    """

    year: int
    papers: Tuple[Paper, ...]  # Papers published in this year
    paper_count: int
    total_citations: int  # Sum of cited_by_count for all papers
    keyword_frequencies: Dict[str, int]  # Frequency of each keyword in this year
    top_keywords: Tuple[str, ...]  # Top-K keywords by frequency (configurable K)

    def __post_init__(self):
        """Validate academic year data."""
        if self.year < 1400 or self.year > 2050:
            raise ValueError(f"Invalid academic year: {self.year}")
        if self.paper_count != len(self.papers):
            raise ValueError(
                f"Paper count mismatch: {self.paper_count} vs {len(self.papers)}"
            )
        if self.total_citations < 0:
            raise ValueError(f"Invalid total citations: {self.total_citations}")
        # Allow empty keywords for years with no valid keywords


@dataclass(frozen=True)
class AcademicPeriod:
    """
    Enhanced temporal data structure for an academic period (multiple years).

    REFACTORED: This now includes all characterization data and gets papers from academic_years.
    No redundant storage of papers since they're already in the constituent academic_years.
    """

    start_year: int
    end_year: int
    academic_years: Tuple[
        AcademicYear, ...
    ]  # Constituent years (papers accessible through these)
    total_papers: int  # Sum across all years
    total_citations: int  # Sum across all years
    combined_keyword_frequencies: Dict[str, int]  # Aggregated keyword frequencies
    top_keywords: Tuple[str, ...]  # Most frequent keywords across period

    # Period characterization data (merged from former PeriodCharacterization)
    topic_label: Optional[str] = None
    topic_description: Optional[str] = None
    network_stability: float = 0.0
    community_persistence: float = 0.0
    flow_stability: float = 0.0
    centrality_consensus: float = 0.0
    representative_papers: Tuple[Dict[str, Any], ...] = field(default_factory=tuple)
    network_metrics: Dict[str, float] = field(default_factory=dict)
    confidence: float = 0.0

    def __post_init__(self):
        """Validate academic period data."""
        if self.start_year > self.end_year:
            raise ValueError(
                f"Invalid period range: {self.start_year} > {self.end_year}"
            )

        # Validate year coverage (allow gaps for filtered data)
        actual_years = set(year.year for year in self.academic_years)

        # Check that all academic years are within the period range
        for year in actual_years:
            if not (self.start_year <= year <= self.end_year):
                raise ValueError(
                    f"Academic year {year} is outside period range {self.start_year}-{self.end_year}"
                )

        # Check that we have at least one year in the period
        if not actual_years:
            raise ValueError(
                f"No academic years found for period {self.start_year}-{self.end_year}"
            )

        # Validate aggregated counts
        expected_papers = sum(year.paper_count for year in self.academic_years)
        if self.total_papers != expected_papers:
            raise ValueError(
                f"Paper count mismatch: {self.total_papers} vs {expected_papers}"
            )

        expected_citations = sum(year.total_citations for year in self.academic_years)
        if self.total_citations != expected_citations:
            raise ValueError(
                f"Citation count mismatch: {self.total_citations} vs {expected_citations}"
            )

        # Validate characterization data ranges
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Invalid confidence: {self.confidence}")
        if not 0.0 <= self.network_stability <= 1.0:
            raise ValueError(f"Invalid network stability: {self.network_stability}")

    def get_year_range(self) -> Tuple[int, int]:
        """Get the year range as a tuple."""
        return (self.start_year, self.end_year)

    def get_papers_in_year(self, year: int) -> Tuple[Paper, ...]:
        """Get papers published in a specific year within this period."""
        for academic_year in self.academic_years:
            if academic_year.year == year:
                return academic_year.papers
        return tuple()

    def get_all_papers(self) -> Tuple[Paper, ...]:
        """Get all papers in this period from constituent academic years."""
        all_papers = []
        for academic_year in self.academic_years:
            all_papers.extend(academic_year.papers)
        return tuple(all_papers)

    def get_keyword_frequency(self, keyword: str) -> int:
        """Get the frequency of a keyword across the entire period."""
        return self.combined_keyword_frequencies.get(keyword, 0)

    def is_characterized(self) -> bool:
        """Check if this period has been characterized with topic information."""
        return self.topic_label is not None and self.confidence > 0.0


# ============================================================================
# CITATION AND DOMAIN MODELS
# ============================================================================


@dataclass(frozen=True)
class CitationRelation:
    """Represents a citation relationship with semantic information."""

    citing_paper_id: str
    cited_paper_id: str
    citing_year: int
    cited_year: int
    relation_description: str = ""
    semantic_description: str = ""
    common_topics_count: int = 0
    edge_index: str = ""

    def __post_init__(self):
        """Validate citation data."""
        if self.common_topics_count < 0:
            raise ValueError(f"Invalid common topics count: {self.common_topics_count}")


@dataclass(frozen=True)
class DomainData:
    """Collection of papers and rich citation graph for a research domain."""

    domain_name: str
    papers: Tuple[Paper, ...]
    citations: Tuple[CitationRelation, ...]
    graph_nodes: Tuple[Tuple[str, str], ...]  # (paper_id, graph_description)
    year_range: Tuple[int, int]
    total_papers: int

    def __post_init__(self):
        """Validate domain data consistency."""
        if self.total_papers != len(self.papers):
            raise ValueError(
                f"Paper count mismatch: {self.total_papers} vs {len(self.papers)}"
            )

        # Validate year range
        if self.papers:
            min_year = min(p.pub_year for p in self.papers)
            max_year = max(p.pub_year for p in self.papers)
            if self.year_range != (min_year, max_year):
                raise ValueError(
                    f"Year range mismatch: {self.year_range} vs ({min_year}, {max_year})"
                )


# ============================================================================
# RESULT MODELS
# ============================================================================


@dataclass(frozen=True)
class TimelineAnalysisResult:
    """
    Results from unified timeline analysis.

    SIMPLIFIED: Only contains essential final results with clean data structures.
    """

    domain_name: str
    periods: Tuple[AcademicPeriod, ...]  # The final timeline periods
    confidence: float
    boundary_years: Tuple[int, ...]  # Years where boundaries were placed
    narrative_evolution: str

    def __post_init__(self):
        """Validate results."""
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Invalid confidence: {self.confidence}")
        if not self.periods:
            raise ValueError("Timeline must have at least one period")


# ============================================================================
# LEGACY STUBS REMOVED
# ============================================================================
# All legacy stub models (PeriodCharacterization, SegmentModelingResult, ShiftSignal)
# have been removed as the migration to simplified architecture is complete.


# Export only essential models
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
]
