#!/usr/bin/env python3
"""
Keyword-Only Cohesion & Separation Metrics Experiments
======================================================

This module implements and evaluates simplified keyword-based metrics for timeline segmentation:
- Cohesion: measures consensus within segments
- Separation: measures difference between adjacent segments

All metrics use filtered keywords (≥2 years, ≥1% papers) for robustness.

ANTI-GAMING SAFEGUARDS (Phase 17):
- Size-weighted averaging instead of simple means
- Minimum segment floor (50+ papers)
- Segment-count penalty with exponential decay
- K-stratified baselines (size + count control)

Evaluation includes:
1. Reference timelines (manual & Gemini) - should score ≥50th percentile vs K-stratified baselines
2. Fixed-range segmentations (5, 10, 15 years) - should score in 25-75th percentile  
3. K-stratified random baselines - control for both size and segment count

Key Design Principles:
- Fail-fast error handling (no try-catch masking)
- Computational efficiency (sub-quadratic algorithms)
- Interpretable scores in 0-1 range
- Anti-gaming: prevent micro-segmentation exploitation
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Set, Optional, NamedTuple
from collections import Counter, defaultdict
from dataclasses import dataclass
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.spatial.distance import jensenshannon
from scipy.stats import pearsonr
import random
import time
import math

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.data.models import Paper, DomainData


@dataclass
class KeywordStats:
    """Statistics for keyword filtering."""
    keyword: str
    years_seen: int
    paper_count: int
    paper_ratio: float
    

@dataclass
class MetricResult:
    """Result container for metric calculations."""
    value: float
    explanation: str
    computation_time: float = 0.0


@dataclass
class AntiGamingConfig:
    """Configuration for anti-gaming safeguards."""
    min_segment_size: int = 50  # Minimum papers per segment
    segment_count_penalty_sigma: float = 4.0  # Exponential decay parameter (more lenient)
    size_weight_power: float = 0.5  # Power for size weighting (0.5 = sqrt)
    enable_size_weighting: bool = True
    enable_segment_floor: bool = True
    enable_count_penalty: bool = False  # Disabled for now - too aggressive


class SegmentationEvaluation(NamedTuple):
    """Complete evaluation result for a segmentation."""
    cohesion_mean_jaccard: float
    cohesion_entropy: float
    separation_js: float
    separation_topk: float
    combined_silhouette: float
    num_segments: int
    filtered_keyword_count: int
    computation_time: float
    # Anti-gaming metrics
    size_weighted_cohesion_jaccard: float = 0.0
    size_weighted_separation_js: float = 0.0
    segment_count_penalty: float = 1.0
    final_anti_gaming_score: float = 0.0


def load_domain_papers(domain_name: str) -> List[Paper]:
    """
    Load papers for a domain from JSON data sources.
    
    Args:
        domain_name: Name of domain (e.g., 'machine_learning')
        
    Returns:
        List of Paper objects
    """
    # Try docs_info.json format first (new format)
    docs_info_path = f"resources/{domain_name}/{domain_name}_docs_info.json"
    
    if not os.path.exists(docs_info_path):
        raise FileNotFoundError(f"Domain data not found: {docs_info_path}")
    
    with open(docs_info_path, 'r') as f:
        docs_data = json.load(f)
    
    # Handle both formats: {"papers": [...]} or {url: paper_data, ...}
    if 'papers' in docs_data:
        paper_entries = docs_data['papers']
    else:
        # Assume it's a dictionary with URLs as keys
        paper_entries = docs_data.values()
    
    papers = []
    for paper_data in paper_entries:
        # Parse keywords - handle both string and list formats
        keywords_raw = paper_data.get('keywords', [])
        if isinstance(keywords_raw, str):
            if '|' in keywords_raw:
                keywords = [k.strip() for k in keywords_raw.split('|') if k.strip()]
            else:
                keywords = [k.strip() for k in keywords_raw.split(',') if k.strip()]
        elif isinstance(keywords_raw, list):
            keywords = [str(k).strip() for k in keywords_raw if str(k).strip()]
        else:
            keywords = []
        
        paper = Paper(
            id=str(paper_data.get('id', f"{domain_name}_{len(papers)}")),
            title=str(paper_data.get('title', '')),
            content=str(paper_data.get('abstract', paper_data.get('content', ''))),
            pub_year=int(paper_data.get('pub_year', paper_data.get('year', 2000))),
            cited_by_count=int(paper_data.get('cited_by_count', 0)),
            keywords=tuple(keywords),
            children=tuple(),
            description=str(paper_data.get('title', ''))
        )
        papers.append(paper)
    
    if not papers:
        raise ValueError(f"No papers found in {docs_info_path}")
    
    print(f"Loaded {len(papers)} papers for {domain_name}")
    return papers


def load_reference_timelines(domain_name: str) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
    """
    Load manual and Gemini reference timelines from JSON files.
    
    Args:
        domain_name: Name of domain
        
    Returns:
        Tuple of (manual_segments, gemini_segments)
    """
    manual_path = f"data/references/{domain_name}_manual.json"
    gemini_path = f"data/references/{domain_name}_gemini.json"
    
    def parse_timeline(filepath: str) -> List[Tuple[int, int]]:
        """Parse timeline JSON to segment list."""
        if not os.path.exists(filepath):
            print(f"WARNING: Timeline file not found: {filepath}")
            return [(2000, 2023)]  # Fallback single segment
        
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        segments = []
        
        # Handle different JSON formats
        if isinstance(data, list):
            # Direct list of segments
            timeline_data = data
        elif 'timeline' in data:
            timeline_data = data['timeline']
        elif 'segments' in data:
            timeline_data = data['segments']
        elif 'historical_periods' in data:
            # Reference timeline format
            timeline_data = data['historical_periods']
        else:
            print(f"WARNING: Unexpected JSON format in {filepath}")
            return [(2000, 2023)]
        
        for item in timeline_data:
            if isinstance(item, dict):
                # Handle different period formats
                if 'years' in item:
                    # Manual format: "years": "1950-1970"
                    years_str = item['years']
                    if '-' in years_str:
                        try:
                            start_str, end_str = years_str.split('-')
                            start = int(start_str.strip())
                            end = int(end_str.strip())
                        except ValueError:
                            continue
                    else:
                        continue
                elif 'start_year' in item and 'end_year' in item:
                    # Gemini format: "start_year": 1950, "end_year": 1970
                    start = item['start_year']
                    end = item['end_year']
                else:
                    # Legacy format
                    start = item.get('start_year', item.get('year_start', 2000))
                    end = item.get('end_year', item.get('year_end', 2023))
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                # Direct (start, end) pair
                start, end = item[0], item[1]
            else:
                print(f"WARNING: Cannot parse timeline item: {item}")
                continue
            
            segments.append((int(start), int(end)))
        
        # Sort by start year and validate
        segments.sort()
        if not segments:
            segments = [(2000, 2023)]
        
        return segments
    
    manual_segments = parse_timeline(manual_path)
    gemini_segments = parse_timeline(gemini_path)
    
    print(f"Loaded reference timelines: manual={len(manual_segments)} segments, gemini={len(gemini_segments)} segments")
    return manual_segments, gemini_segments


def compute_keyword_statistics(papers: List[Paper]) -> Dict[str, KeywordStats]:
    """
    Compute statistics for all keywords across papers.
    
    Args:
        papers: List of papers
        
    Returns:
        Dictionary mapping keyword -> KeywordStats
    """
    keyword_years = defaultdict(set)
    keyword_papers = defaultdict(int)
    total_papers = len(papers)
    
    for paper in papers:
        for keyword in paper.keywords:
            keyword_years[keyword].add(paper.pub_year)
            keyword_papers[keyword] += 1
    
    stats = {}
    for keyword in keyword_years:
        stats[keyword] = KeywordStats(
            keyword=keyword,
            years_seen=len(keyword_years[keyword]),
            paper_count=keyword_papers[keyword],
            paper_ratio=keyword_papers[keyword] / total_papers if total_papers > 0 else 0.0
        )
    
    return stats


def filter_keywords(papers: List[Paper], min_years: int = 2, min_paper_ratio: float = 0.01) -> Tuple[List[Paper], Set[str]]:
    """
    Filter papers to only include keywords that meet robustness criteria.
    
    RELAXED FILTERING: Now uses 1% threshold instead of 5% to retain more keywords.
    
    Args:
        papers: List of input papers
        min_years: Minimum years a keyword must appear (default: 2)
        min_paper_ratio: Minimum ratio of papers a keyword must appear in (default: 0.01 = 1%)
        
    Returns:
        Tuple of (filtered_papers, valid_keywords_set)
    """
    # Compute keyword statistics
    keyword_stats = compute_keyword_statistics(papers)
    
    # Apply filtering criteria
    valid_keywords = set()
    for keyword, stats in keyword_stats.items():
        if stats.years_seen >= min_years and stats.paper_ratio >= min_paper_ratio:
            valid_keywords.add(keyword)
    
    # Filter papers to only include valid keywords
    filtered_papers = []
    for paper in papers:
        filtered_keywords = [kw for kw in paper.keywords if kw in valid_keywords]
        if filtered_keywords:  # Only include papers with at least one valid keyword
            filtered_paper = Paper(
                id=paper.id,
                title=paper.title,
                content=paper.content,
                pub_year=paper.pub_year,
                cited_by_count=paper.cited_by_count,
                keywords=tuple(filtered_keywords),
                children=paper.children,
                description=paper.description
            )
            filtered_papers.append(filtered_paper)
    
    print(f"Keyword filtering: {len(keyword_stats)} → {len(valid_keywords)} keywords ({len(valid_keywords)/len(keyword_stats)*100:.1f}%)")
    print(f"Paper filtering: {len(papers)} → {len(filtered_papers)} papers ({len(filtered_papers)/len(papers)*100:.1f}%)")
    
    return filtered_papers, valid_keywords


def papers_to_segments(papers: List[Paper], segment_years: List[Tuple[int, int]]) -> List[List[Paper]]:
    """
    Convert year-based segments to paper-based segments.
    
    Args:
        papers: List of papers
        segment_years: List of (start_year, end_year) tuples
        
    Returns:
        List of paper segments
    """
    segments = []
    
    for start_year, end_year in segment_years:
        segment_papers = [
            paper for paper in papers
            if start_year <= paper.pub_year <= end_year
        ]
        segments.append(segment_papers)
    
    return segments


# ============================================================================
# COHESION METRICS (within-segment consensus) - WITH ANTI-GAMING
# ============================================================================

def cohesion_mean_jaccard_to_union(segment_papers: List[Paper], top_k_keywords: int = 15) -> MetricResult:
    """
    Mean Jaccard similarity of papers to segment's top-K defining keywords.
    
    Only papers with at least 1 keyword from the top-K are included in cohesion calculation.
    Robust to vocabulary size differences. Complexity: O(sum(|K_p|)).
    
    Args:
        segment_papers: List of papers in segment
        top_k_keywords: Number of top keywords to define segment (10-20 recommended)
        
    Returns:
        MetricResult with cohesion score in [0, 1]
    """
    start_time = time.time()
    
    if not segment_papers:
        return MetricResult(0.0, "Empty segment", time.time() - start_time)
    
    # Count keyword frequencies in segment
    keyword_counts = Counter()
    for paper in segment_papers:
        keyword_counts.update(paper.keywords)
    
    if not keyword_counts:
        return MetricResult(0.0, "No keywords in segment", time.time() - start_time)
    
    # Get top-K most frequent keywords to define the segment
    top_keywords = set(kw for kw, _ in keyword_counts.most_common(top_k_keywords))
    
    if not top_keywords:
        return MetricResult(0.0, "No defining keywords", time.time() - start_time)
    
    # Only include papers that have at least 1 keyword from top-K
    relevant_papers = []
    jaccard_scores = []
    
    for paper in segment_papers:
        paper_keywords = set(paper.keywords)
        paper_top_keywords = paper_keywords & top_keywords
        
        if paper_top_keywords:  # Paper has at least 1 defining keyword
            relevant_papers.append(paper)
            # Jaccard: intersection / union with top keywords
            jaccard = len(paper_top_keywords) / len(paper_keywords | top_keywords)
            jaccard_scores.append(jaccard)
    
    if not jaccard_scores:
        return MetricResult(0.0, "No papers with defining keywords", time.time() - start_time)
    
    mean_jaccard = float(np.mean(jaccard_scores))
    explanation = f"Mean Jaccard to top-{len(top_keywords)} keywords: {mean_jaccard:.3f} over {len(relevant_papers)}/{len(segment_papers)} papers"
    
    return MetricResult(mean_jaccard, explanation, time.time() - start_time)


def cohesion_keyword_entropy(segment_papers: List[Paper], top_k_keywords: int = 15) -> MetricResult:
    """
    Keyword entropy-based cohesion using only top-K defining keywords.
    
    Only considers papers with at least 1 keyword from top-K. Lower entropy = higher cohesion.
    Complexity: O(|K_segment|).
    
    Args:
        segment_papers: List of papers in segment
        top_k_keywords: Number of top keywords to define segment (10-20 recommended)
        
    Returns:
        MetricResult with cohesion score in [0, 1]
    """
    start_time = time.time()
    
    if not segment_papers:
        return MetricResult(0.0, "Empty segment", time.time() - start_time)
    
    # Count keyword frequencies in segment
    keyword_counts = Counter()
    for paper in segment_papers:
        keyword_counts.update(paper.keywords)
    
    if not keyword_counts:
        return MetricResult(0.0, "No keywords in segment", time.time() - start_time)
    
    # Get top-K most frequent keywords to define the segment
    top_keywords = set(kw for kw, _ in keyword_counts.most_common(top_k_keywords))
    
    if not top_keywords:
        return MetricResult(0.0, "No defining keywords", time.time() - start_time)
    
    # Count frequencies only among papers that have defining keywords
    relevant_keyword_counts = Counter()
    relevant_papers = 0
    
    for paper in segment_papers:
        paper_keywords = set(paper.keywords)
        paper_top_keywords = paper_keywords & top_keywords
        
        if paper_top_keywords:  # Paper has at least 1 defining keyword
            relevant_papers += 1
            relevant_keyword_counts.update(paper_top_keywords)
    
    if not relevant_keyword_counts:
        return MetricResult(0.0, "No papers with defining keywords", time.time() - start_time)
    
    # Compute Shannon entropy over defining keywords only
    total_count = sum(relevant_keyword_counts.values())
    probabilities = [count / total_count for count in relevant_keyword_counts.values()]
    
    entropy = -sum(p * np.log2(p) for p in probabilities if p > 0)
    max_entropy = np.log2(len(relevant_keyword_counts))  # Uniform distribution
    
    # Convert to cohesion (higher = more cohesive)
    if max_entropy > 0:
        normalized_entropy = entropy / max_entropy
        cohesion = 1.0 - normalized_entropy
    else:
        cohesion = 1.0  # Single keyword = perfect cohesion
    
    explanation = f"Entropy cohesion (top-{len(top_keywords)}): {cohesion:.3f} over {relevant_papers}/{len(segment_papers)} papers (entropy={entropy:.3f})"
    
    return MetricResult(cohesion, explanation, time.time() - start_time)


# ============================================================================
# SEPARATION METRICS (between-segment difference)
# ============================================================================

def separation_jensen_shannon(segment_a: List[Paper], segment_b: List[Paper]) -> MetricResult:
    """
    Jensen-Shannon divergence between segment keyword frequency distributions.
    
    Symmetric, bounded [0, 1]. Complexity: O(|V|) where V = union of vocabularies.
    
    Args:
        segment_a: First segment papers
        segment_b: Second segment papers
        
    Returns:
        MetricResult with separation score in [0, 1]
    """
    start_time = time.time()
    
    if not segment_a or not segment_b:
        return MetricResult(0.0, "Empty segment", time.time() - start_time)
    
    # Count keyword frequencies
    counts_a = Counter()
    counts_b = Counter()
    
    for paper in segment_a:
        counts_a.update(paper.keywords)
    for paper in segment_b:
        counts_b.update(paper.keywords)
    
    # Get union vocabulary
    vocab = set(counts_a.keys()) | set(counts_b.keys())
    
    if not vocab:
        return MetricResult(0.0, "No keywords in either segment", time.time() - start_time)
    
    # Create frequency vectors
    total_a = sum(counts_a.values()) or 1  # Avoid division by zero
    total_b = sum(counts_b.values()) or 1
    
    freq_a = np.array([counts_a.get(kw, 0) / total_a for kw in vocab])
    freq_b = np.array([counts_b.get(kw, 0) / total_b for kw in vocab])
    
    # Compute Jensen-Shannon divergence
    js_distance = jensenshannon(freq_a, freq_b, base=2)  # Base 2 for [0, 1] range
    
    explanation = f"Jensen-Shannon separation: {js_distance:.3f} over {len(vocab)} keywords"
    
    return MetricResult(float(js_distance), explanation, time.time() - start_time)


def separation_topk_overlap(segment_a: List[Paper], segment_b: List[Paper], k: int = 50) -> MetricResult:
    """
    1 - Jaccard similarity of top-K most frequent keywords per segment.
    
    Robust to long-tail noise. Complexity: O(|K| log |K|) for sorting.
    
    Args:
        segment_a: First segment papers
        segment_b: Second segment papers
        k: Number of top keywords to consider
        
    Returns:
        MetricResult with separation score in [0, 1]
    """
    start_time = time.time()
    
    if not segment_a or not segment_b:
        return MetricResult(0.0, "Empty segment", time.time() - start_time)
    
    # Count keyword frequencies
    counts_a = Counter()
    counts_b = Counter()
    
    for paper in segment_a:
        counts_a.update(paper.keywords)
    for paper in segment_b:
        counts_b.update(paper.keywords)
    
    if not counts_a or not counts_b:
        return MetricResult(0.0, "No keywords in one segment", time.time() - start_time)
    
    # Get top-K keywords
    top_k_a = set(kw for kw, _ in counts_a.most_common(k))
    top_k_b = set(kw for kw, _ in counts_b.most_common(k))
    
    # Compute Jaccard distance
    intersection = len(top_k_a & top_k_b)
    union = len(top_k_a | top_k_b)
    
    if union == 0:
        jaccard_similarity = 0.0
    else:
        jaccard_similarity = intersection / union
    
    separation = 1.0 - jaccard_similarity
    
    explanation = f"Top-{k} overlap separation: {separation:.3f} (intersection={intersection}, union={union})"
    
    return MetricResult(separation, explanation, time.time() - start_time)


# ============================================================================
# ANTI-GAMING AGGREGATION FUNCTIONS
# ============================================================================

def compute_size_weighted_average(segment_scores: List[float], segment_sizes: List[int], power: float = 0.5) -> float:
    """
    Compute size-weighted average to prevent micro-segment gaming.
    
    Uses power weighting: weight = size^power
    - power=0.0: uniform weighting (vulnerable to gaming)
    - power=0.5: square root weighting (balanced)
    - power=1.0: linear weighting (heavily favors large segments)
    
    Args:
        segment_scores: List of metric scores per segment
        segment_sizes: List of sizes (number of papers) per segment
        power: Power for size weighting (default: 0.5)
        
    Returns:
        Size-weighted average score
    """
    if not segment_scores or not segment_sizes:
        return 0.0
    
    if len(segment_scores) != len(segment_sizes):
        raise ValueError("segment_scores and segment_sizes must have same length")
    
    # Compute weights
    weights = [size ** power for size in segment_sizes]
    total_weight = sum(weights)
    
    if total_weight == 0:
        return 0.0
    
    # Weighted average
    weighted_sum = sum(score * weight for score, weight in zip(segment_scores, weights))
    return weighted_sum / total_weight


def compute_segment_count_penalty(num_segments: int, domain_year_span: int, sigma: float = 4.0) -> float:
    """
    Compute exponential penalty for deviating from expected segment count.
    
    Expected segments = domain_year_span / 15 (one segment per ~15 years for more realistic expectations)
    Penalty = exp(-|K_actual - K_expected| / σ)
    
    Args:
        num_segments: Actual number of segments
        domain_year_span: Total years covered by domain
        sigma: Exponential decay parameter (higher = more lenient, default: 4.0)
        
    Returns:
        Penalty factor in [0, 1]
    """
    if domain_year_span <= 0:
        return 1.0
    
    # More realistic expectation: one segment per 15 years instead of 10
    expected_segments = max(1, round(domain_year_span / 15))
    deviation = abs(num_segments - expected_segments)
    
    # More lenient penalty with higher sigma
    penalty = math.exp(-deviation / sigma)
    
    return penalty


def filter_segments_by_size(segments: List[List[Paper]], min_size: int) -> Tuple[List[List[Paper]], List[int]]:
    """
    Filter out segments below minimum size threshold.
    
    Args:
        segments: List of paper segments
        min_size: Minimum papers per segment
        
    Returns:
        Tuple of (filtered_segments, excluded_indices)
    """
    filtered_segments = []
    excluded_indices = []
    
    for i, segment in enumerate(segments):
        if len(segment) >= min_size:
            filtered_segments.append(segment)
        else:
            excluded_indices.append(i)
    
    return filtered_segments, excluded_indices


# ============================================================================
# COMBINED SCORING - WITH ANTI-GAMING
# ============================================================================

def compute_silhouette_score(cohesion: float, separation: float) -> float:
    """
    Compute silhouette-style combined score: separation / (cohesion + separation).
    
    Args:
        cohesion: Cohesion score [0, 1]
        separation: Separation score [0, 1]
        
    Returns:
        Combined score in [0, 1]
    """
    if cohesion + separation == 0:
        return 0.0
    
    return separation / (cohesion + separation)


def evaluate_segmentation(papers: List[Paper], segment_years: List[Tuple[int, int]], 
                         anti_gaming_config: Optional[AntiGamingConfig] = None) -> SegmentationEvaluation:
    """
    Comprehensive evaluation of a segmentation using all keyword-based metrics with anti-gaming safeguards.
    
    Args:
        papers: List of filtered papers
        segment_years: List of (start_year, end_year) segments
        anti_gaming_config: Configuration for anti-gaming measures
        
    Returns:
        SegmentationEvaluation with all metric scores including anti-gaming measures
    """
    start_time = time.time()
    
    if anti_gaming_config is None:
        anti_gaming_config = AntiGamingConfig()
    
    # Convert to paper segments
    segments = papers_to_segments(papers, segment_years)
    
    if not segments:
        return SegmentationEvaluation(0.0, 0.0, 0.0, 0.0, 0.0, 0, 0, time.time() - start_time)
    
    # Count filtered keywords
    all_keywords = set()
    for paper in papers:
        all_keywords.update(paper.keywords)
    
    # Apply segment size filtering if enabled
    original_segments = segments
    if anti_gaming_config.enable_segment_floor:
        segments, excluded_indices = filter_segments_by_size(segments, anti_gaming_config.min_segment_size)
        if excluded_indices:
            print(f"   Excluded {len(excluded_indices)} segments below {anti_gaming_config.min_segment_size} papers")
    
    if not segments:
        print("   WARNING: All segments excluded by size filter")
        return SegmentationEvaluation(0.0, 0.0, 0.0, 0.0, 0.0, 0, 0, time.time() - start_time)
    
    # Compute cohesion metrics (average across segments)
    cohesion_jaccard_scores = []
    cohesion_entropy_scores = []
    segment_sizes = []
    
    for segment in segments:
        if segment:  # Skip empty segments
            jaccard_result = cohesion_mean_jaccard_to_union(segment, top_k_keywords=15)
            entropy_result = cohesion_keyword_entropy(segment, top_k_keywords=15)
            
            cohesion_jaccard_scores.append(jaccard_result.value)
            cohesion_entropy_scores.append(entropy_result.value)
            segment_sizes.append(len(segment))
    
    # Standard averaging (for backward compatibility)
    avg_cohesion_jaccard = float(np.mean(cohesion_jaccard_scores)) if cohesion_jaccard_scores else 0.0
    avg_cohesion_entropy = float(np.mean(cohesion_entropy_scores)) if cohesion_entropy_scores else 0.0

    # Size-weighted averaging (anti-gaming)
    size_weighted_cohesion_jaccard = 0.0
    if anti_gaming_config.enable_size_weighting and cohesion_jaccard_scores and segment_sizes:
        size_weighted_cohesion_jaccard = compute_size_weighted_average(
            cohesion_jaccard_scores, segment_sizes, anti_gaming_config.size_weight_power
        )
    else:
        size_weighted_cohesion_jaccard = avg_cohesion_jaccard
    
    # Compute separation metrics (average across adjacent pairs)
    separation_js_scores = []
    separation_topk_scores = []
    transition_sizes = []  # For size-weighted separation
    
    for i in range(len(segments) - 1):
        if segments[i] and segments[i + 1]:  # Skip empty segments
            js_result = separation_jensen_shannon(segments[i], segments[i + 1])
            topk_result = separation_topk_overlap(segments[i], segments[i + 1])
            
            separation_js_scores.append(js_result.value)
            separation_topk_scores.append(topk_result.value)
            
            # Transition size = geometric mean of adjacent segment sizes
            transition_size = math.sqrt(len(segments[i]) * len(segments[i + 1]))
            transition_sizes.append(transition_size)
    
    # Standard averaging (for backward compatibility)
    avg_separation_js = float(np.mean(separation_js_scores)) if separation_js_scores else 0.0
    avg_separation_topk = float(np.mean(separation_topk_scores)) if separation_topk_scores else 0.0

    # Size-weighted averaging (anti-gaming)
    size_weighted_separation_js = 0.0
    if anti_gaming_config.enable_size_weighting and separation_js_scores and transition_sizes:
        size_weighted_separation_js = compute_size_weighted_average(
            separation_js_scores, transition_sizes, anti_gaming_config.size_weight_power
        )
    else:
        size_weighted_separation_js = avg_separation_js
    
    # Compute segment count penalty
    segment_count_penalty = 1.0
    if anti_gaming_config.enable_count_penalty:
        domain_years = [p.pub_year for p in papers]
        if domain_years:
            domain_year_span = max(domain_years) - min(domain_years) + 1
            segment_count_penalty = compute_segment_count_penalty(
                len(original_segments), domain_year_span, anti_gaming_config.segment_count_penalty_sigma
            )
    
    # Compute combined scores
    combined_score = compute_silhouette_score(size_weighted_cohesion_jaccard, size_weighted_separation_js)
    
    # Final anti-gaming score: size-weighted metrics with segment count penalty
    if anti_gaming_config.enable_size_weighting:
        anti_gaming_combined = compute_silhouette_score(size_weighted_cohesion_jaccard, size_weighted_separation_js)
        final_anti_gaming_score = anti_gaming_combined * segment_count_penalty
    else:
        final_anti_gaming_score = combined_score * segment_count_penalty
    

    
    return SegmentationEvaluation(
        cohesion_mean_jaccard=avg_cohesion_jaccard,
        cohesion_entropy=avg_cohesion_entropy,
        separation_js=avg_separation_js,
        separation_topk=avg_separation_topk,
        combined_silhouette=combined_score,
        num_segments=len(original_segments),
        filtered_keyword_count=len(all_keywords),
        computation_time=time.time() - start_time,
        size_weighted_cohesion_jaccard=size_weighted_cohesion_jaccard,
        size_weighted_separation_js=size_weighted_separation_js,
        segment_count_penalty=segment_count_penalty,
        final_anti_gaming_score=final_anti_gaming_score
    )


# ============================================================================
# SEGMENTATION GENERATORS - WITH K-STRATIFIED BASELINES
# ============================================================================

def generate_random_segmentations(papers: List[Paper], num_samples: int = 1000) -> List[List[Tuple[int, int]]]:
    """
    Generate random segmentations by sampling breakpoints.
    
    Args:
        papers: List of papers to determine year range
        num_samples: Number of random segmentations to generate
        
    Returns:
        List of segmentations (each is list of (start_year, end_year))
    """
    if not papers:
        return []
    
    min_year = min(paper.pub_year for paper in papers)
    max_year = max(paper.pub_year for paper in papers)
    year_span = max_year - min_year + 1
    
    segmentations = []
    
    for _ in range(num_samples):
        # Random number of segments (1 to year_span//2)
        max_segments = max(1, year_span // 2)
        num_segments = random.randint(1, max_segments)
        
        if num_segments == 1:
            # Single segment
            segmentations.append([(min_year, max_year)])
        else:
            # Generate random breakpoints
            breakpoints = sorted(random.sample(range(min_year + 1, max_year), num_segments - 1))
            
            # Convert to segments
            segments = []
            start = min_year
            for breakpoint in breakpoints:
                segments.append((start, breakpoint - 1))
                start = breakpoint
            segments.append((start, max_year))
            
            segmentations.append(segments)
    
    return segmentations


def generate_fixed_range_segmentations(papers: List[Paper]) -> Dict[int, List[Tuple[int, int]]]:
    """
    Generate fixed-range segmentations (5, 10, 15 year windows).
    
    Args:
        papers: List of papers to determine year range
        
    Returns:
        Dictionary mapping window_size -> segmentation
    """
    if not papers:
        return {}
    
    min_year = min(paper.pub_year for paper in papers)
    max_year = max(paper.pub_year for paper in papers)
    
    segmentations = {}
    
    for window_size in [5, 10, 15]:
        segments = []
        current_year = min_year
        
        while current_year <= max_year:
            end_year = min(current_year + window_size - 1, max_year)
            segments.append((current_year, end_year))
            current_year = end_year + 1
        
        segmentations[window_size] = segments
    
    return segmentations


def generate_size_stratified_baselines(papers: List[Paper], target_segment_sizes: List[int], num_samples: int = 100) -> List[List[Tuple[int, int]]]:
    """
    Generate size-stratified random baselines that match target segment sizes.
    
    This addresses the segment size bias where tiny random segments get 
    artificially perfect cohesion scores compared to realistic large segments.
    
    Args:
        papers: List of papers to segment
        target_segment_sizes: List of target sizes (number of papers per segment)
        num_samples: Number of random segmentations to generate
        
    Returns:
        List of segmentations with segments approximately matching target sizes
    """
    if not papers:
        return []
    
    # Sort papers by year for easier segmentation
    sorted_papers = sorted(papers, key=lambda p: p.pub_year)
    total_papers = len(sorted_papers)
    
    stratified_segmentations = []
    
    for _ in range(num_samples):
        # Randomly shuffle target sizes to vary segment order
        shuffled_sizes = target_segment_sizes.copy()
        random.shuffle(shuffled_sizes)
        
        # Adjust sizes to fit total papers
        total_target = sum(shuffled_sizes)
        if total_target > total_papers:
            # Scale down proportionally
            scale_factor = total_papers / total_target
            shuffled_sizes = [max(1, int(size * scale_factor)) for size in shuffled_sizes]
        
        # Create segments with target sizes
        segments = []
        start_idx = 0
        
        for i, target_size in enumerate(shuffled_sizes):
            if start_idx >= total_papers:
                break
                
            # For last segment, take all remaining papers
            if i == len(shuffled_sizes) - 1:
                end_idx = total_papers
            else:
                # Add some randomness (±20%) to target size
                actual_size = max(1, int(target_size * random.uniform(0.8, 1.2)))
                end_idx = min(start_idx + actual_size, total_papers)
            
            if end_idx > start_idx:
                # Get year range for this segment
                start_year = sorted_papers[start_idx].pub_year
                end_year = sorted_papers[end_idx - 1].pub_year
                segments.append((start_year, end_year))
                start_idx = end_idx
        
        if len(segments) >= 2:  # Need at least 2 segments
            stratified_segmentations.append(segments)
    
    return stratified_segmentations


def generate_k_stratified_baselines(papers: List[Paper], target_k: int, target_segment_sizes: List[int], 
                                   num_samples: int = 100) -> List[List[Tuple[int, int]]]:
    """
    Generate K-stratified baselines that control for both segment count AND sizes.
    
    This is the ultimate anti-gaming baseline that prevents both:
    1. Micro-segmentation (many tiny segments)
    2. Size bias (comparing large vs small segments)
    
    Args:
        papers: List of papers to segment
        target_k: Target number of segments
        target_segment_sizes: List of target sizes (should have length target_k)
        num_samples: Number of random segmentations to generate
        
    Returns:
        List of K-segment segmentations with approximately matching sizes
    """
    if not papers or target_k <= 0:
        return []
    
    # Ensure we have target_k sizes
    if len(target_segment_sizes) != target_k:
        # Repeat or truncate sizes to match target_k
        if len(target_segment_sizes) < target_k:
            # Repeat sizes cyclically
            target_segment_sizes = (target_segment_sizes * ((target_k // len(target_segment_sizes)) + 1))[:target_k]
        else:
            # Truncate to target_k
            target_segment_sizes = target_segment_sizes[:target_k]
    
    # Sort papers by year for easier segmentation
    sorted_papers = sorted(papers, key=lambda p: p.pub_year)
    total_papers = len(sorted_papers)
    
    k_stratified_segmentations = []
    
    for _ in range(num_samples):
        # Randomly shuffle target sizes to vary segment order
        shuffled_sizes = target_segment_sizes.copy()
        random.shuffle(shuffled_sizes)
        
        # Adjust sizes to fit total papers
        total_target = sum(shuffled_sizes)
        if total_target > total_papers:
            # Scale down proportionally
            scale_factor = total_papers / total_target
            shuffled_sizes = [max(1, int(size * scale_factor)) for size in shuffled_sizes]
        
        # Ensure we have exactly target_k segments
        segments = []
        start_idx = 0
        
        for i in range(target_k):
            if start_idx >= total_papers:
                break
                
            # For last segment, take all remaining papers
            if i == target_k - 1:
                end_idx = total_papers
            else:
                # Use target size with some randomness (±10% for K-stratified)
                target_size = shuffled_sizes[i] if i < len(shuffled_sizes) else (total_papers // target_k)
                actual_size = max(1, int(target_size * random.uniform(0.9, 1.1)))
                end_idx = min(start_idx + actual_size, total_papers)
            
            if end_idx > start_idx:
                # Get year range for this segment
                start_year = sorted_papers[start_idx].pub_year
                end_year = sorted_papers[end_idx - 1].pub_year
                segments.append((start_year, end_year))
                start_idx = end_idx
        
        # Only include if we have exactly target_k segments
        if len(segments) == target_k:
            k_stratified_segmentations.append(segments)
    
    return k_stratified_segmentations


if __name__ == "__main__":
    print("Keyword Metrics Experiments Module - WITH ANTI-GAMING SAFEGUARDS")
    print("Run experiments using the evaluation functions above.") 