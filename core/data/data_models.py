"""Core data models for scientific publication analysis.

This module defines immutable data structures for papers, citations, and temporal data
using functional programming principles.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, Any

@dataclass(frozen=True)
class Paper:
    """Immutable representation of a scientific paper."""

    id: str
    title: str
    content: str
    pub_year: int
    cited_by_count: int
    keywords: Tuple[str, ...]
    children: Tuple[str, ...]  # Citing papers
    description: str

    def __post_init__(self):
        """Validate paper data after initialization."""
        if self.pub_year < 1400 or self.pub_year > 2050:
            raise ValueError(f"Invalid publication year: {self.pub_year}")
        if self.cited_by_count < 0:
            raise ValueError(f"Invalid citation count: {self.cited_by_count}")


@dataclass(frozen=True)
class AcademicYear:
    """Pre-computed temporal data structure for a single academic year.

    This structure is computed once during data loading and contains all
    year-level aggregations needed by the algorithm.
    """

    year: int
    papers: Tuple[Paper, ...]
    paper_count: int
    total_citations: int
    keyword_frequencies: Dict[str, int]
    top_keywords: Tuple[str, ...]

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


@dataclass(frozen=True)
class AcademicPeriod:
    """Enhanced temporal data structure for an academic period (multiple years)."""

    start_year: int
    end_year: int
    academic_years: Tuple[AcademicYear, ...]
    total_papers: int
    total_citations: int
    combined_keyword_frequencies: Dict[str, int]
    top_keywords: Tuple[str, ...]

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

        actual_years = set(year.year for year in self.academic_years)

        for year in actual_years:
            if not (self.start_year <= year <= self.end_year):
                raise ValueError(
                    f"Academic year {year} is outside period range {self.start_year}-{self.end_year}"
                )

        if not actual_years:
            raise ValueError(
                f"No academic years found for period {self.start_year}-{self.end_year}"
            )

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

        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Invalid confidence: {self.confidence}")
        if not 0.0 <= self.network_stability <= 1.0:
            raise ValueError(f"Invalid network stability: {self.network_stability}")

    def get_year_range(self) -> Tuple[int, int]:
        """Get the year range as a tuple.
        
        Returns:
            Tuple of (start_year, end_year)
        """
        return (self.start_year, self.end_year)

    def get_papers_in_year(self, year: int) -> Tuple[Paper, ...]:
        """Get papers published in a specific year within this period.
        
        Args:
            year: Year to get papers for
            
        Returns:
            Tuple of papers published in the specified year
        """
        for academic_year in self.academic_years:
            if academic_year.year == year:
                return academic_year.papers
        return tuple()

    def get_all_papers(self) -> Tuple[Paper, ...]:
        """Get all papers in this period from constituent academic years.
        
        Returns:
            Tuple of all papers in the period
        """
        all_papers = []
        for academic_year in self.academic_years:
            all_papers.extend(academic_year.papers)
        return tuple(all_papers)

    def get_keyword_frequency(self, keyword: str) -> int:
        """Get the frequency of a keyword across the entire period.
        
        Args:
            keyword: Keyword to look up
            
        Returns:
            Frequency count of the keyword
        """
        return self.combined_keyword_frequencies.get(keyword, 0)

    def is_characterized(self) -> bool:
        """Check if this period has been characterized with topic information.
        
        Returns:
            True if period has topic characterization
        """
        return self.topic_label is not None and self.confidence > 0.0


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
class TimelineAnalysisResult:
    """Results from unified timeline analysis."""

    domain_name: str
    periods: Tuple[AcademicPeriod, ...]
    confidence: float
    boundary_years: Tuple[int, ...]
    narrative_evolution: str

    def __post_init__(self):
        """Validate results."""
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Invalid confidence: {self.confidence}")
        if not self.periods:
            raise ValueError("Timeline must have at least one period")


__all__ = [
    "AcademicYear",
    "AcademicPeriod",
    "Paper",
    "CitationRelation",
    "TimelineAnalysisResult",
]
