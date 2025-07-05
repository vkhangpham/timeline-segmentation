"""
Consolidated Keyword Utilities

Pure functions for keyword extraction and processing shared across all modules.
This consolidates keyword logic to prevent drift and ensure consistency across
experiments. Replaces duplicated keyword processing scattered across multiple modules.

Key Functions:
- Keyword extraction from papers
- Frequency analysis and counting
- Top-K keyword selection
- Keyword distribution metrics
- Jaccard similarity calculations

Follows project guidelines:
- Fail-fast error handling (no fallbacks)
- Functional programming approach (pure functions)
- Real data usage (leverages existing infrastructure)
- Single responsibility principle (consolidated keyword operations)
"""

from typing import Dict, List, Tuple, Counter as CounterType, Optional, Union
from collections import defaultdict, Counter

from ..data.models import DomainData, Paper

# Import YAKE for phrase enrichment
try:
    import yake
except ImportError:
    # Fail-fast: YAKE is required when phrase_enrichment is enabled
    yake = None


def extract_year_keywords(domain_data: DomainData) -> Dict[int, List[str]]:
    """
    Extract year-to-keywords mapping from domain data.
    
    This is the canonical implementation shared across all modules to prevent
    logic drift between similarity segmentation and direction detection.
    
    Args:
        domain_data: Domain data containing papers with keywords
        
    Returns:
        Dict mapping years to lists of keywords for that year
        
    Raises:
        ValueError: If domain_data is invalid or contains no papers
        
    Example:
        year_keywords = extract_year_keywords(domain_data)
        # Returns: {1995: ['neural', 'network'], 1996: ['deep', 'learning'], ...}
    """
    
    if not domain_data:
        raise ValueError("domain_data cannot be None")
    
    if not domain_data.papers:
        raise ValueError("domain_data must contain at least one paper")
    
    year_keywords = defaultdict(list)
    
    # Group papers by year and aggregate keywords  
    for paper in domain_data.papers:
        year = paper.pub_year
        
        # Add paper keywords to year (if any)
        if hasattr(paper, 'keywords') and paper.keywords:
            year_keywords[year].extend(paper.keywords)
    
    # Convert to regular dict and remove duplicates while preserving order
    result = {}
    for year, keywords in year_keywords.items():
        seen = set()
        unique_keywords = []
        for keyword in keywords:
            if keyword not in seen:
                seen.add(keyword)
                unique_keywords.append(keyword)
        result[year] = unique_keywords
    
    return result


def calculate_jaccard_similarity(keywords_a: List[str], keywords_b: List[str]) -> float:
    """
    Calculate Jaccard similarity coefficient between two keyword lists.
    
    FIXED: Returns 0.0 when either set is empty (was incorrectly returning 1.0 
    for both empty, which biased boundary detection toward sparse periods).
    
    Jaccard similarity = |intersection| / |union|
    Ranges from 0.0 (no overlap) to 1.0 (identical sets)
    
    Args:
        keywords_a: First keyword list
        keywords_b: Second keyword list
        
    Returns:
        Jaccard similarity coefficient (0.0 to 1.0)
        
    Example:
        sim = calculate_jaccard_similarity(['a', 'b', 'c'], ['b', 'c', 'd'])
        # Returns: 0.5 (2 intersection / 4 union)
    """
    
    # Convert to sets for efficient set operations
    set_a = set(keywords_a) if keywords_a else set()
    set_b = set(keywords_b) if keywords_b else set()
    
    # Handle empty sets case - FIXED: return 0.0 when either is empty
    if not set_a or not set_b:
        return 0.0  # No similarity when either set is empty
    
    # Calculate Jaccard similarity
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    
    return intersection / union if union > 0 else 0.0 


def yake_phrases(text: str, top_k: int = 10) -> List[str]:
    """
    Extract top-k key phrases from text using YAKE algorithm.
    
    YAKE (Yet Another Keyword Extractor) is an unsupervised method that identifies
    key phrases based on statistical features without requiring training data.
    This function follows fail-fast principles and functional programming style.
    
    Args:
        text: Input text to extract phrases from
        top_k: Number of top phrases to return (default: 10)
        
    Returns:
        List of extracted phrases (strings), ordered by YAKE score (lower is better)
        
    Raises:
        ImportError: If YAKE package is not installed
        ValueError: If text is empty or top_k is invalid
        
    Example:
        phrases = yake_phrases("Neural networks show great promise in machine learning", top_k=5)
        # Returns: ['neural networks', 'machine learning', 'great promise', ...]
    """
    # Fail-fast validation
    if yake is None:
        raise ImportError("YAKE package not installed. Run: pip install yake")
    
    if not text or not text.strip():
        raise ValueError("text cannot be empty")
    
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    
    # YAKE configuration optimized for academic abstracts
    # - lan: language (English)
    # - n: maximum number of words in keyphrase (3 for multi-word concepts)
    # - dedupLim: deduplication threshold (0.7 to avoid near-duplicates)
    # - top: number of phrases to extract
    kw_extractor = yake.KeywordExtractor(
        lan="en",
        n=3,  # Allow up to 3-word phrases
        dedupLim=0.7,
        top=top_k
    )
    
    # Extract keywords - returns list of (phrase, score) tuples
    keywords = kw_extractor.extract_keywords(text)
    
    # Return only the phrases, sorted by YAKE score (lower is better)
    phrases = [phrase for phrase, score in keywords]
    
    return phrases


# =============================================================================
# CONSOLIDATED KEYWORD PROCESSING FUNCTIONS
# =============================================================================

def extract_keywords_from_papers(papers: Union[List[Paper], Tuple[Paper, ...]]) -> List[str]:
    """
    Extract all keywords from a list of Paper objects.
    
    Consolidates keyword extraction logic that was duplicated across:
    - data_processing.py (3 instances)
    - keyword_filtering.py 
    - paper_selection_and_labeling.py
    - shift_signal_detection.py
    
    Args:
        papers: List or tuple of Paper objects
        
    Returns:
        List of all keywords from all papers (includes duplicates for frequency counting)
        
    Raises:
        ValueError: If papers is empty or invalid
        
    Example:
        papers = [Paper(...), Paper(...)]
        all_keywords = extract_keywords_from_papers(papers)
        # Returns: ['neural', 'network', 'learning', 'neural', 'deep', ...]
    """
    if not papers:
        raise ValueError("papers cannot be empty")
    
    all_keywords = []
    for paper in papers:
        if hasattr(paper, 'keywords') and paper.keywords:
            all_keywords.extend(paper.keywords)
    
    return all_keywords


def count_keyword_frequencies(papers: Union[List[Paper], Tuple[Paper, ...]], 
                            return_paper_counts: bool = False) -> Union[Counter, Tuple[Counter, Dict[str, int]]]:
    """
    Count keyword frequencies from papers with optional paper count tracking.
    
    Consolidates keyword frequency counting logic that was duplicated across multiple modules.
    
    Args:
        papers: List or tuple of Paper objects
        return_paper_counts: If True, also return how many papers each keyword appears in
        
    Returns:
        If return_paper_counts=False: Counter of keyword frequencies
        If return_paper_counts=True: Tuple of (keyword_frequencies, keyword_paper_counts)
        
    Example:
        keyword_freq = count_keyword_frequencies(papers)
        # Returns: Counter({'neural': 15, 'network': 12, 'learning': 10, ...})
        
        keyword_freq, paper_counts = count_keyword_frequencies(papers, return_paper_counts=True)
        # Returns: (Counter(...), {'neural': 8, 'network': 7, 'learning': 6, ...})
    """
    if not papers:
        return Counter() if not return_paper_counts else (Counter(), {})
    
    # Extract keywords
    all_keywords = extract_keywords_from_papers(papers)
    keyword_frequencies = Counter(all_keywords)
    
    if not return_paper_counts:
        return keyword_frequencies
    
    # Count how many papers each keyword appears in
    keyword_paper_counts = {}
    keywords_to_count = set(all_keywords)
    
    for keyword in keywords_to_count:
        paper_count = sum(1 for paper in papers 
                         if hasattr(paper, 'keywords') and paper.keywords and keyword in paper.keywords)
        keyword_paper_counts[keyword] = paper_count
    
    return keyword_frequencies, keyword_paper_counts


def get_top_keywords(papers: Union[List[Paper], Tuple[Paper, ...]], 
                    top_k: int = 20) -> List[Tuple[str, int]]:
    """
    Get top-K keywords by frequency from papers.
    
    Consolidates top keyword selection logic duplicated across multiple modules.
    
    Args:
        papers: List or tuple of Paper objects
        top_k: Number of top keywords to return
        
    Returns:
        List of (keyword, frequency) tuples, sorted by frequency descending
        
    Example:
        top_keywords = get_top_keywords(papers, top_k=10)
        # Returns: [('neural', 15), ('network', 12), ('learning', 10), ...]
    """
    if not papers:
        return []
    
    keyword_frequencies = count_keyword_frequencies(papers)
    return keyword_frequencies.most_common(top_k)


def analyze_keyword_distribution(papers: Union[List[Paper], Tuple[Paper, ...]], 
                               min_frequency_threshold: int = 1) -> Dict[str, any]:
    """
    Analyze keyword distribution and quality metrics for papers.
    
    Consolidates keyword analysis logic from keyword_filtering.py and data_processing.py.
    
    Args:
        papers: List or tuple of Paper objects
        min_frequency_threshold: Minimum frequency threshold for analysis
        
    Returns:
        Dictionary with comprehensive keyword distribution metrics
        
    Example:
        metrics = analyze_keyword_distribution(papers)
        # Returns: {'total_keywords': 1000, 'unique_keywords': 250, 'avg_per_paper': 3.2, ...}
    """
    if not papers:
        return {
            'total_papers': 0,
            'total_keywords': 0,
            'unique_keywords': 0,
            'papers_with_keywords': 0,
            'papers_without_keywords': 0,
            'keyword_coverage': 0.0,
            'avg_keywords_per_paper': 0.0,
            'singleton_keywords': 0,
            'low_frequency_keywords': 0
        }
    
    # Basic metrics
    total_papers = len(papers)
    all_keywords = extract_keywords_from_papers(papers)
    total_keywords = len(all_keywords)
    
    # Papers with/without keywords
    papers_with_keywords = sum(1 for paper in papers 
                              if hasattr(paper, 'keywords') and paper.keywords)
    papers_without_keywords = total_papers - papers_with_keywords
    
    # Keyword frequencies
    keyword_frequencies = Counter(all_keywords)
    unique_keywords = len(keyword_frequencies)
    
    # Distribution analysis
    frequency_counts = Counter(keyword_frequencies.values())
    singleton_keywords = frequency_counts.get(1, 0)  # Keywords appearing only once
    low_frequency_keywords = sum(1 for count in keyword_frequencies.values() 
                                if count < min_frequency_threshold)
    
    return {
        'total_papers': total_papers,
        'total_keywords': total_keywords,
        'unique_keywords': unique_keywords,
        'papers_with_keywords': papers_with_keywords,
        'papers_without_keywords': papers_without_keywords,
        'keyword_coverage': papers_with_keywords / total_papers if total_papers > 0 else 0.0,
        'avg_keywords_per_paper': total_keywords / total_papers if total_papers > 0 else 0.0,
        'keyword_diversity': unique_keywords / total_keywords if total_keywords > 0 else 0.0,
        'singleton_keywords': singleton_keywords,
        'singleton_ratio': singleton_keywords / unique_keywords if unique_keywords > 0 else 0.0,
        'low_frequency_keywords': low_frequency_keywords,
        'low_frequency_ratio': low_frequency_keywords / unique_keywords if unique_keywords > 0 else 0.0,
        'frequency_distribution': dict(frequency_counts)
    }


def get_emerging_keywords(papers: Union[List[Paper], Tuple[Paper, ...]], 
                         recent_years: int = 3,
                         top_k: int = 10) -> Tuple[List[str], List[Tuple[str, int]]]:
    """
    Identify emerging keywords from recent papers.
    
    Consolidates emerging keyword logic from data_processing.py.
    
    Args:
        papers: List or tuple of Paper objects
        recent_years: Number of recent years to consider for emerging keywords
        top_k: Number of top emerging keywords to return
        
    Returns:
        Tuple of (emerging_keywords_list, keyword_frequencies)
        
    Example:
        emerging, freqs = get_emerging_keywords(papers, recent_years=3, top_k=10)
        # Returns: (['transformers', 'attention', 'bert'], [('transformers', 45), ...])
    """
    if not papers:
        return [], []
    
    # Find the most recent years in the dataset
    years = [paper.pub_year for paper in papers if hasattr(paper, 'pub_year')]
    if not years:
        return [], []
    
    max_year = max(years)
    cutoff_year = max_year - recent_years + 1
    
    # Filter recent papers
    recent_papers = [paper for paper in papers 
                    if hasattr(paper, 'pub_year') and paper.pub_year >= cutoff_year]
    
    if not recent_papers:
        return [], []
    
    # Get top keywords from recent papers
    top_recent_keywords = get_top_keywords(recent_papers, top_k=top_k)
    emerging_keywords = [kw for kw, count in top_recent_keywords]
    
    return emerging_keywords, top_recent_keywords


def convert_keywords_string_to_list(keywords_str: str, separator: str = '|') -> List[str]:
    """
    Convert separated keyword string to list.
    
    Consolidates keyword string conversion logic from data_processing.py.
    
    Args:
        keywords_str: Separated keywords string
        separator: Separator character (default: '|')
        
    Returns:
        List of keywords
        
    Example:
        keywords = convert_keywords_string_to_list("neural|network|learning")
        # Returns: ['neural', 'network', 'learning']
    """
    if not keywords_str or not keywords_str.strip():
        return []
    
    return [k.strip() for k in keywords_str.split(separator) if k.strip()]


# Export all functions
__all__ = [
    # Original functions
    'extract_year_keywords',
    'calculate_jaccard_similarity', 
    'yake_phrases',
    # Consolidated keyword processing functions
    'extract_keywords_from_papers',
    'count_keyword_frequencies',
    'get_top_keywords',
    'analyze_keyword_distribution',
    'get_emerging_keywords',
    'convert_keywords_string_to_list'
] 