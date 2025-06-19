"""
Core data models for scientific publication analysis.

This module defines immutable data structures for papers, citations, and temporal data
using functional programming principles. All data classes are frozen to ensure immutability.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set, Tuple, Any
from datetime import datetime


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
        if self.pub_year < 0 or self.pub_year > 2030:
            raise ValueError(f"Invalid publication year: {self.pub_year}")
        if self.cited_by_count < 0:
            raise ValueError(f"Invalid citation count: {self.cited_by_count}")


@dataclass(frozen=True)
class CitationRelation:
    """Represents a citation relationship with rich semantic information."""
    
    citing_paper_id: str
    cited_paper_id: str
    citing_year: int
    cited_year: int
    relation_description: str = ""  # Natural language timeline description
    semantic_description: str = ""  # Detailed HOW relationship description  
    common_topics_count: int = 0    # Number of shared topics
    edge_index: str = ""           # Graph edge identifier
    
    def __post_init__(self):
        """Validate citation data."""
        if self.common_topics_count < 0:
            raise ValueError(f"Invalid common topics count: {self.common_topics_count}")


@dataclass(frozen=True)
class DomainData:
    """Collection of papers and rich citation graph for a research domain."""
    
    domain_name: str
    papers: Tuple[Paper, ...]
    citations: Tuple[CitationRelation, ...]  # Now includes rich semantic info
    graph_nodes: Tuple[Tuple[str, str], ...]  # (paper_id, graph_description)
    year_range: Tuple[int, int]
    total_papers: int
    
    def __post_init__(self):
        """Validate domain data consistency."""
        if self.total_papers != len(self.papers):
            raise ValueError(f"Paper count mismatch: {self.total_papers} vs {len(self.papers)}")
        
        # Validate year range
        if self.papers:
            min_year = min(p.pub_year for p in self.papers)
            max_year = max(p.pub_year for p in self.papers)
            if self.year_range != (min_year, max_year):
                raise ValueError(f"Year range mismatch: {self.year_range} vs ({min_year}, {max_year})")


@dataclass(frozen=True)
class TemporalWindow:
    """Represents a time window for analysis."""
    
    start_year: int
    end_year: int
    papers: Tuple[Paper, ...]
    paper_count: int
    avg_citations: float
    total_citations: int
    
    def __post_init__(self):
        """Validate temporal window data."""
        if self.start_year > self.end_year:
            raise ValueError(f"Start year {self.start_year} cannot be after end year {self.end_year}")
        if self.paper_count != len(self.papers):
            raise ValueError(f"Paper count mismatch: {self.paper_count} vs {len(self.papers)}")


@dataclass(frozen=True)
class DataStatistics:
    """Statistical summary of domain data."""
    
    domain_name: str
    total_papers: int
    year_range: Tuple[int, int]
    avg_citations: float
    median_citations: float
    content_completeness: float
    keyword_completeness: float
    citation_network_size: int
    avg_citations_per_paper: float
    top_keywords: Tuple[str, ...]
    most_productive_years: Tuple[Tuple[int, int], ...]  # (year, paper_count)
    
    def __post_init__(self):
        """Validate statistics."""
        if not 0 <= self.content_completeness <= 1:
            raise ValueError(f"Invalid content completeness: {self.content_completeness}")
        if not 0 <= self.keyword_completeness <= 1:
            raise ValueError(f"Invalid keyword completeness: {self.keyword_completeness}")


@dataclass(frozen=True)
class ProcessingResult:
    """Result of data processing operations."""
    
    success: bool
    domain_data: Optional[DomainData]
    statistics: Optional[DataStatistics]
    error_message: Optional[str]
    processing_time_seconds: float
    papers_processed: int
    
    def __post_init__(self):
        """Validate result consistency."""
        if self.success and (self.domain_data is None or self.statistics is None):
            raise ValueError("Successful result must include domain_data and statistics")
        if not self.success and self.error_message is None:
            raise ValueError("Failed result must include error_message")


@dataclass(frozen=True)
class DataSubset:
    """Represents a subset of papers for testing/validation."""
    
    name: str
    papers: Tuple[Paper, ...]
    selection_criteria: str
    subset_size: int
    original_domain: str
    
    def __post_init__(self):
        """Validate subset data."""
        if self.subset_size != len(self.papers):
            raise ValueError(f"Subset size mismatch: {self.subset_size} vs {len(self.papers)}")


@dataclass(frozen=True)
class ValidationEvent:
    """Historical event for validation purposes."""
    
    event_id: str
    title: str
    domain: str
    year: int
    confidence: float
    description: str
    expected_signals: Tuple[str, ...]
    validation_window: Tuple[int, int]  # (start_year, end_year)
    priority: str  # "Critical", "High", "Medium", "Low"
    
    def __post_init__(self):
        """Validate event data."""
        if not 0 <= self.confidence <= 1:
            raise ValueError(f"Invalid confidence: {self.confidence}")
        if self.priority not in ["Critical", "High", "Medium", "Low"]:
            raise ValueError(f"Invalid priority: {self.priority}")


@dataclass(frozen=True)
class KeywordAnalysis:
    """Analysis based on paper keywords and semantic evolution."""
    
    domain_name: str
    time_window: Tuple[int, int]
    keyword_frequencies: Tuple[Tuple[str, int], ...]  # (keyword, frequency)
    emerging_keywords: Tuple[str, ...]  # Keywords with high growth
    relationship_patterns: Tuple[str, ...]  # Common research relationship patterns
    semantic_signals: Tuple[str, ...]  # Innovation signals from citation descriptions


@dataclass(frozen=True)
class ChangePointWithPapers:
    """Enhanced change point that tracks contributing papers."""
    
    year: int
    confidence: float
    method: str  # "kleinberg", "cusum", "semantic"
    signal_type: str  # "citation", "productivity", "keyword", "semantic"
    description: str
    supporting_evidence: Tuple[str, ...]
    contributing_papers: Tuple[str, ...]  # Paper IDs that contributed to this change point


@dataclass(frozen=True)
class BurstPeriodWithPapers:
    """Enhanced burst period that tracks contributing papers."""
    
    start_year: int
    end_year: int
    intensity: float
    signal_type: str
    burst_items: Tuple[str, ...]  # Keywords or semantic patterns in burst
    contributing_papers: Tuple[str, ...]  # Paper IDs that had bursts


@dataclass(frozen=True)
class ChangeDetectionResultWithPapers:
    """Enhanced change detection result with paper tracking."""
    
    domain_name: str
    time_range: Tuple[int, int]
    change_points: Tuple[ChangePointWithPapers, ...]
    burst_periods: Tuple[BurstPeriodWithPapers, ...]
    statistical_significance: float
    validation_score: float
    # Signal-to-paper mappings for representative selection
    citation_burst_papers: Dict[Tuple[int, int], Tuple[str, ...]]
    semantic_change_papers: Dict[Tuple[int, int], Tuple[str, ...]]
    keyword_burst_papers: Dict[Tuple[int, int], Tuple[str, ...]]


# ============================================================================
# CHANGE DETECTION DATA MODELS
# ============================================================================

@dataclass(frozen=True)
class ChangePoint:
    """Represents a detected change point in time series."""
    
    year: int
    confidence: float
    method: str  # "enhanced_shift_signal"
    signal_type: str  # "citation", "semantic", "combined"
    description: str
    supporting_evidence: Tuple[str, ...]


@dataclass(frozen=True)
class BurstPeriod:
    """Represents a burst period (for backward compatibility)."""
    
    start_year: int
    end_year: int
    intensity: float
    signal_type: str
    burst_items: Tuple[str, ...]


@dataclass(frozen=True)
class ChangeDetectionResult:
    """Result of change point detection analysis."""
    
    domain_name: str
    time_range: Tuple[int, int]
    change_points: Tuple[ChangePoint, ...]
    burst_periods: Tuple[BurstPeriod, ...]
    statistical_significance: float
    validation_score: float


# ============================================================================
# SHIFT SIGNAL DETECTION DATA MODELS
# ============================================================================

@dataclass(frozen=True)
class ShiftSignal:
    """Represents a detected paradigm transition signal."""
    
    year: int
    confidence: float
    signal_type: str  # "citation_disruption", "semantic_shift", "cross_domain", "direction_volatility"
    evidence_strength: float
    supporting_evidence: Tuple[str, ...]
    contributing_papers: Tuple[str, ...]
    transition_description: str
    paradigm_significance: float  # 0.0-1.0 scale for paradigm vs technical


@dataclass(frozen=True)
class TransitionEvidence:
    """Evidence supporting a paradigm transition."""
    
    year: int
    disruption_patterns: Tuple[str, ...]
    emergence_patterns: Tuple[str, ...]
    cross_domain_influences: Tuple[str, ...]
    methodological_changes: Tuple[str, ...]
    breakthrough_papers: Tuple[str, ...]
    confidence_score: float


# ============================================================================
# PERIOD SIGNAL DETECTION DATA MODELS
# ============================================================================

@dataclass(frozen=True)
class PeriodCharacterization:
    """Represents a characterized research period with network-based analysis."""
    
    period: Tuple[int, int]
    topic_label: str
    topic_description: str
    network_stability: float
    community_persistence: float
    flow_stability: float
    centrality_consensus: float
    representative_papers: Tuple[Dict[str, Any], ...]
    network_metrics: Dict[str, float]
    confidence: float


# ============================================================================
# SEGMENT MODELING DATA MODELS
# ============================================================================

@dataclass(frozen=True)
class SegmentModelingResult:
    """Results from segment modeling analysis."""
    
    domain_name: str
    segments: Tuple[Tuple[int, int], ...]
    period_characterizations: Tuple[PeriodCharacterization, ...]
    modeling_confidence: float
    modeling_summary: str


# ============================================================================
# SEGMENT MERGING DATA MODELS
# ============================================================================

@dataclass(frozen=True)
class MergeDecision:
    """Represents a decision to merge two consecutive segments."""
    
    segment1_index: int
    segment2_index: int
    semantic_similarity: float
    shift_signal_strength: float
    merge_confidence: float
    merge_justification: str
    merged_period: Tuple[int, int]
    merged_label: str
    merged_description: str


@dataclass(frozen=True)
class SegmentMergingResult:
    """Results from segment merging analysis."""
    
    original_segments: Tuple[PeriodCharacterization, ...]
    merged_segments: Tuple[PeriodCharacterization, ...]
    merge_decisions: Tuple[MergeDecision, ...]
    merging_summary: str


# ============================================================================
# INTEGRATION DATA MODELS
# ============================================================================

@dataclass(frozen=True)
class TimelineAnalysisResult:
    """Results from unified timeline analysis."""
    
    domain_name: str
    period_characterizations: Tuple[PeriodCharacterization, ...]
    merged_period_characterizations: Tuple[PeriodCharacterization, ...]
    merging_result: Optional[SegmentMergingResult]
    unified_confidence: float
    narrative_evolution: str


 