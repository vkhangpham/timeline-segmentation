"""
Conservative Keyword Filtering for Domain-Aware Paradigm Detection

This module implements IMPROVEMENT-001: Conservative keyword filtering to reduce noise
from imperfect keyword annotations. Uses frequency-based and cross-domain analysis
rather than semantic filtering to avoid over-aggressive filtering.

Philosophy:
- Downstream mitigation for imperfect keyword annotation (not root cause fix)
- Conservative approach: better to filter too little than too much
- Preserve recall over precision to avoid degrading performance
- Simple, explainable filtering criteria

Follows functional programming principles with pure functions and immutable data.
"""

from typing import List, Dict, Set, Tuple, Optional
from collections import defaultdict, Counter
import warnings

from .data_models import DomainData, Paper
from .algorithm_config import AlgorithmConfig


def filter_domain_keywords_conservative(
    keywords: List[str], 
    year_papers: List[Paper],
    algorithm_config: AlgorithmConfig,
    domain_name: str
) -> Tuple[List[str], Dict[str, str]]:
    """
    Apply conservative keyword filtering to reduce noise from imperfect annotations.
    
    Philosophy: Conservative filtering that only removes obvious noise while preserving
    potentially valid keywords. Better to keep questionable keywords than risk losing
    genuine paradigm signals.
    
    Args:
        keywords: Raw keywords for the time period
        year_papers: Papers from the time period
        algorithm_config: Configuration with filtering parameters
        domain_name: Domain name for logging/context
        
    Returns:
        Tuple of (filtered_keywords, filtering_rationale)
        filtering_rationale: Dict mapping each decision to explanation
    """
    if not algorithm_config.keyword_filtering_enabled:
        return keywords, {"status": "filtering_disabled"}
    
    if not keywords:
        return keywords, {"status": "no_keywords_to_filter"}
    
    filtering_rationale = {"status": "active_filtering"}
    original_count = len(keywords)
    
    # Step 1: Basic frequency filtering - remove keywords appearing in very few papers
    filtered_keywords = _filter_by_paper_frequency(
        keywords, year_papers, algorithm_config, filtering_rationale
    )
    
    # Step 2: Conservative cross-domain contamination check (future enhancement)
    # Note: For now, we only implement frequency filtering to start conservatively
    # Cross-domain analysis requires access to other domain data which we'll add later
    
    filtered_count = len(filtered_keywords)
    filtering_rationale["original_keywords"] = original_count
    filtering_rationale["filtered_keywords"] = filtered_count
    filtering_rationale["filtering_ratio"] = f"{filtered_count}/{original_count} ({filtered_count/original_count:.1%})"

    return filtered_keywords, filtering_rationale


def _filter_by_paper_frequency(
    keywords: List[str],
    year_papers: List[Paper], 
    algorithm_config: AlgorithmConfig,
    filtering_rationale: Dict[str, str]
) -> List[str]:
    """
    Conservative frequency-based filtering: remove keywords appearing in very few papers.
    
    This addresses noise from keywords that appear in only 1-2 papers within a time window,
    which are unlikely to represent genuine domain-wide paradigm shifts.
    """
    if not year_papers:
        filtering_rationale["frequency_filtering"] = "no_papers_in_period"
        return keywords
    
    # Count how many papers contain each keyword
    keyword_paper_counts = Counter()
    for paper in year_papers:
        paper_keywords = set(paper.keywords)
        for keyword in keywords:
            if keyword in paper_keywords:
                keyword_paper_counts[keyword] += 1
    
    # Calculate minimum paper threshold
    total_papers = len(year_papers)
    min_papers_threshold = max(1, int(total_papers * algorithm_config.keyword_min_papers_ratio))
    
    # Conservative filtering: only remove keywords with very low frequency
    filtered_keywords = []
    removed_keywords = []
    
    for keyword in keywords:
        paper_count = keyword_paper_counts.get(keyword, 0)
        if paper_count >= min_papers_threshold:
            filtered_keywords.append(keyword)
        else:
            removed_keywords.append(f"{keyword}({paper_count}p)")
    
    # Add detailed rationale
    filtering_rationale["frequency_threshold"] = f"{min_papers_threshold} papers (ratio={algorithm_config.keyword_min_papers_ratio:.2f})"
    filtering_rationale["total_papers_in_period"] = total_papers
    
    if removed_keywords:
        filtering_rationale["removed_low_frequency"] = f"{len(removed_keywords)} keywords: {', '.join(removed_keywords[:5])}"
        if len(removed_keywords) > 5:
            filtering_rationale["removed_low_frequency"] += f" (+{len(removed_keywords)-5} more)"
    else:
        filtering_rationale["removed_low_frequency"] = "none"
    
    return filtered_keywords


def analyze_keyword_quality_metrics(
    domain_data: DomainData,
    algorithm_config: AlgorithmConfig
) -> Dict[str, any]:
    """
    Analyze keyword quality metrics for domain to inform filtering decisions.
    
    This provides insights into keyword annotation quality and helps set appropriate
    filtering thresholds without actually modifying the keywords.
    
    Returns metrics that can be used to assess filtering effectiveness.
    """
    if not algorithm_config.keyword_filtering_enabled:
        return {"status": "filtering_disabled"}
    
    # Group papers by year
    papers_by_year = defaultdict(list)
    for paper in domain_data.papers:
        papers_by_year[paper.pub_year].append(paper)
    
    # Analyze keyword distribution patterns
    all_keywords = []
    for paper in domain_data.papers:
        all_keywords.extend(paper.keywords)
    
    keyword_frequencies = Counter(all_keywords)
    total_keywords = len(all_keywords)
    unique_keywords = len(keyword_frequencies)
    
    # Calculate quality metrics
    metrics = {
        "domain_name": domain_data.domain_name,
        "total_papers": len(domain_data.papers),
        "total_keyword_instances": total_keywords,
        "unique_keywords": unique_keywords,
        "avg_keywords_per_paper": total_keywords / len(domain_data.papers) if domain_data.papers else 0,
        "keyword_diversity": unique_keywords / total_keywords if total_keywords > 0 else 0
    }
    
    # Analyze papers without keywords (data quality issue)
    papers_without_keywords = sum(1 for p in domain_data.papers if not p.keywords)
    metrics["papers_without_keywords"] = papers_without_keywords
    metrics["keyword_coverage"] = (len(domain_data.papers) - papers_without_keywords) / len(domain_data.papers) if domain_data.papers else 0
    
    # Analyze frequency distribution
    frequency_counts = Counter(keyword_frequencies.values())
    singleton_keywords = frequency_counts.get(1, 0)  # Keywords appearing only once
    metrics["singleton_keywords"] = singleton_keywords
    metrics["singleton_ratio"] = singleton_keywords / unique_keywords if unique_keywords > 0 else 0
    
    # Low frequency keywords (potential noise)
    low_freq_threshold = max(1, int(len(domain_data.papers) * algorithm_config.keyword_min_papers_ratio))
    low_freq_keywords = sum(1 for count in keyword_frequencies.values() if count < low_freq_threshold)
    metrics["low_frequency_keywords"] = low_freq_keywords
    metrics["low_frequency_ratio"] = low_freq_keywords / unique_keywords if unique_keywords > 0 else 0
    
    return metrics


def validate_filtering_configuration(
    algorithm_config: AlgorithmConfig,
    domain_data: DomainData
) -> List[str]:
    """
    Validate keyword filtering configuration and return warnings/suggestions.
    
    Helps ensure filtering parameters are reasonable for the domain characteristics.
    """
    warnings_list = []
    
    if not algorithm_config.keyword_filtering_enabled:
        return warnings_list
    
    # Check if parameters are too aggressive for domain size
    total_papers = len(domain_data.papers)
    min_papers_threshold = max(1, int(total_papers * algorithm_config.keyword_min_papers_ratio))
    
    if min_papers_threshold > total_papers * 0.1:
        warnings_list.append(f"keyword_min_papers_ratio ({algorithm_config.keyword_min_papers_ratio:.2f}) may be too aggressive for domain size ({total_papers} papers)")
    
    if algorithm_config.cross_domain_contamination_threshold < 0.7:
        warnings_list.append(f"cross_domain_contamination_threshold ({algorithm_config.cross_domain_contamination_threshold:.2f}) is quite aggressive - may remove valid interdisciplinary keywords")
    
    # Check keyword coverage
    papers_with_keywords = sum(1 for p in domain_data.papers if p.keywords)
    keyword_coverage = papers_with_keywords / total_papers if total_papers > 0 else 0
    
    if keyword_coverage < 0.5:
        warnings_list.append(f"Low keyword coverage ({keyword_coverage:.1%}) - keyword filtering may be less effective")
    
    return warnings_list


# Utility function for testing and analysis
def preview_filtering_impact(
    domain_data: DomainData,
    algorithm_config: AlgorithmConfig,
    year_range: Optional[Tuple[int, int]] = None
) -> Dict[str, any]:
    """
    Preview the impact of keyword filtering without actually applying it.
    
    Useful for testing and parameter tuning before full implementation.
    """
    if not algorithm_config.keyword_filtering_enabled:
        return {"status": "filtering_disabled"}
    
    # Filter papers by year range if specified
    if year_range:
        start_year, end_year = year_range
        relevant_papers = [p for p in domain_data.papers if start_year <= p.pub_year <= end_year]
    else:
        relevant_papers = list(domain_data.papers)
    
    # Collect all keywords
    all_keywords = []
    for paper in relevant_papers:
        all_keywords.extend(paper.keywords)
    
    # Apply filtering preview
    filtered_keywords, rationale = filter_domain_keywords_conservative(
        all_keywords, relevant_papers, algorithm_config, domain_data.domain_name
    )
    
    # Calculate impact metrics
    original_unique = len(set(all_keywords))
    filtered_unique = len(set(filtered_keywords))
    
    preview = {
        "domain_name": domain_data.domain_name,
        "year_range": year_range or (min(p.pub_year for p in domain_data.papers), max(p.pub_year for p in domain_data.papers)),
        "papers_analyzed": len(relevant_papers),
        "original_total_keywords": len(all_keywords),
        "original_unique_keywords": original_unique,
        "filtered_total_keywords": len(filtered_keywords),
        "filtered_unique_keywords": filtered_unique,
        "retention_rate": filtered_unique / original_unique if original_unique > 0 else 0,
        "filtering_rationale": rationale
    }
    
    return preview 