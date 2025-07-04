"""
Shared Keyword Utilities

Pure functions for keyword extraction and processing shared across all modules.
This consolidates keyword logic to prevent drift and ensure consistency across
experiments.

Follows project guidelines:
- Fail-fast error handling (no fallbacks)
- Functional programming approach (pure functions)
- Real data usage (leverages existing infrastructure)
"""

from typing import Dict, List
from collections import defaultdict

from .data_models import DomainData

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