"""
Consensus & Difference Metrics (Phase 15)
========================================

Pure-function metrics to quantify
A. Consensus inside a segment (C-metrics)
B. Difference between two consecutive segments (D-metrics)

Only robust, explainable signals are used:
    • keywords
    • abstracts / content (TF-IDF embedding)
    • citation edges

Each metric returns **(value, explanation_string)** so callers can surface
human-readable reasoning in dashboards or journals.

All functions raise on invalid input (fail-fast).

Weights are loaded from optimization_config.json to ensure consistency across
all optimization and baseline comparison methods.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, List, Dict, NamedTuple
from collections import Counter
import numpy as np
import json
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import jensenshannon

from core.data_models import Paper

# ---------------------------------------------------------------------------
# Configuration Loading
# ---------------------------------------------------------------------------

def _load_optimization_config():
    """Load optimization configuration from centralized JSON file."""
    config_path = "optimization_config.json"
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Optimization config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    return config

# Global configuration loaded once
_OPTIMIZATION_CONFIG = _load_optimization_config()

# ---------------------------------------------------------------------------
# Helper NamedTuples for explanations
# ---------------------------------------------------------------------------

class MetricResult(NamedTuple):
    value: float
    explanation: str

# ---------------------------------------------------------------------------
# Utility helpers (pure)
# ---------------------------------------------------------------------------

def _extract_keywords(papers: Tuple[Paper, ...]) -> List[str]:
    kw: List[str] = []
    for p in papers:
        if p.keywords:
            kw.extend(p.keywords)
    return kw


def _tfidf_embeddings(texts: List[str]):
    if len(texts) < 2:
        # Single vector or empty – return zeros to avoid failure
        return np.zeros((len(texts), 1))
    vec = TfidfVectorizer(max_features=500, stop_words="english")
    return vec.fit_transform(texts).toarray()

# ---------------------------------------------------------------------------
# C-metrics  (Consensus inside a segment)
# ---------------------------------------------------------------------------

def c1_keyword_jaccard(segment: Tuple[Paper, ...]) -> MetricResult:
    """Average Jaccard keyword overlap between each paper and the segment keyword set."""
    if not segment:
        raise ValueError("segment cannot be empty")
    seg_kw_set = set(_extract_keywords(segment))
    if not seg_kw_set:
        return MetricResult(0.0, "Segment has no keywords → consensus undefined → score 0.0")
    scores = []
    for p in segment:
        p_kw = set(p.keywords) if p.keywords else set()
        if not p_kw:
            continue
        jacc = len(p_kw & seg_kw_set) / len(p_kw | seg_kw_set)
        scores.append(jacc)
    if not scores:
        return MetricResult(0.0, "No papers with keywords → consensus 0.0")
    mean_score = float(np.mean(scores))
    explanation = (
        f"C1 Keyword-Jaccard: mean Jaccard of each paper vs segment keywords over {len(scores)} papers "
        f"= {mean_score:.3f} (Segment keyword count={len(seg_kw_set)})"
    )
    return MetricResult(mean_score, explanation)


def c2_tfidf_cohesion(segment: Tuple[Paper, ...]) -> MetricResult:
    """Mean pairwise cosine similarity of TF-IDF embeddings of abstracts / keywords."""
    n = len(segment)
    if n < 2:
        return MetricResult(0.0, "Segment size <2 → cohesion 0.0")
    texts: List[str] = []
    for p in segment:
        if p.content:
            texts.append(p.content)
        elif p.keywords:
            texts.append(" ".join(p.keywords))
        else:
            texts.append("")
    emb = _tfidf_embeddings(texts)
    if emb.shape[0] < 2:
        return MetricResult(0.0, "Insufficient text for TF-IDF cohesion → 0.0")
    sim = cosine_similarity(emb)
    mean_sim = float((sim.sum() - np.trace(sim)) / (n * (n - 1)))
    explanation = (
        f"C2 TF-IDF Cohesion: mean pairwise cosine similarity across {n} papers = {mean_sim:.3f}"
    )
    return MetricResult(mean_sim, explanation)


def c3_citation_density(segment: Tuple[Paper, ...]) -> MetricResult:
    """Density of citation edges inside the segment (undirected)."""
    n = len(segment)
    if n < 2:
        return MetricResult(0.0, "Segment size <2 → density 0.0")
    id_map = {p.id: p for p in segment}
    edges = 0
    for p in segment:
        if p.children:
            edges += sum(1 for child in p.children if child in id_map)
    undirected_edges = edges / 2  # every edge counted twice (a→b, b←a)
    density = min(1.0, undirected_edges / (n * (n - 1) / 2))
    explanation = (
        f"C3 Citation Density: {int(undirected_edges)} internal edges among {n} papers → {density:.3f}"
    )
    return MetricResult(density, explanation)

# ---------------------------------------------------------------------------
# D-metrics  (Difference between two segments)
# ---------------------------------------------------------------------------

def _freq_vector(keywords: List[str], vocab: List[str]):
    counts = Counter(keywords)
    return np.array([counts.get(t, 0) for t in vocab], dtype=float)


def d1_keyword_js(segment_a: Tuple[Paper, ...], segment_b: Tuple[Paper, ...]) -> MetricResult:
    """Jensen-Shannon divergence between keyword frequency distributions."""
    kw_a = _extract_keywords(segment_a)
    kw_b = _extract_keywords(segment_b)
    vocab = list(set(kw_a) | set(kw_b))
    if not vocab:
        return MetricResult(0.0, "No keywords in either segment → JS 0.0")
    p = _freq_vector(kw_a, vocab)
    q = _freq_vector(kw_b, vocab)
    p = p / (p.sum() or 1)
    q = q / (q.sum() or 1)
    js = float(jensenshannon(p, q, base=2))  # 0-1
    explanation = f"D1 Keyword JS-Divergence over {len(vocab)} keywords = {js:.3f}"
    return MetricResult(js, explanation)


def d2_centroid_distance(segment_a: Tuple[Paper, ...], segment_b: Tuple[Paper, ...]) -> MetricResult:
    """1 – cosine similarity between TF-IDF centroids of two segments."""
    texts_a: List[str] = [p.content or " ".join(p.keywords) for p in segment_a]
    texts_b: List[str] = [p.content or " ".join(p.keywords) for p in segment_b]
    if not texts_a or not texts_b:
        return MetricResult(0.0, "Empty text in one segment → distance 0.0")
    emb = _tfidf_embeddings(texts_a + texts_b)
    k = len(texts_a)
    centroid_a = emb[:k].mean(axis=0)
    centroid_b = emb[k:].mean(axis=0)
    sim = float(cosine_similarity([centroid_a], [centroid_b])[0][0])
    distance = 1.0 - sim
    explanation = f"D2 Centroid Distance (1 – cosine) = {distance:.3f}"
    return MetricResult(distance, explanation)


def d3_cross_citation_ratio(segment_a: Tuple[Paper, ...], segment_b: Tuple[Paper, ...]) -> MetricResult:
    """Ratio of citation edges that cross segments (lower → more different)."""
    ids_a = {p.id for p in segment_a}
    ids_b = {p.id for p in segment_b}
    total_possible = len(segment_a) * len(segment_b)
    if total_possible == 0:
        return MetricResult(0.0, "One segment empty → ratio 0.0")
    cross = 0
    for p in segment_a:
        if p.children:
            cross += sum(1 for c in p.children if c in ids_b)
    for p in segment_b:
        if p.children:
            cross += sum(1 for c in p.children if c in ids_a)
    ratio = cross / total_possible
    explanation = f"D3 Cross-Citation Ratio: {cross}/{total_possible} = {ratio:.3f}"
    return MetricResult(ratio, explanation)

# ---------------------------------------------------------------------------
# Aggregators
# ---------------------------------------------------------------------------

def consensus_score(segment: Tuple[Paper, ...], weights: Tuple[float, float, float]=None) -> MetricResult:
    """Weighted combination of C-metrics."""
    if weights is None:
        # Load default weights from configuration
        consensus_weights = _OPTIMIZATION_CONFIG["consensus_difference_weights"]["consensus_internal_weights"]
        weights = (
            consensus_weights["c1_keyword_jaccard"],
            consensus_weights["c2_tfidf_cohesion"], 
            consensus_weights["c3_citation_density"]
        )
    
    c1 = c1_keyword_jaccard(segment)
    c2 = c2_tfidf_cohesion(segment)
    c3 = c3_citation_density(segment)
    w1, w2, w3 = weights
    score = w1*c1.value + w2*c2.value + w3*c3.value
    expl = " | ".join([c1.explanation, c2.explanation, c3.explanation])
    return MetricResult(score, f"Consensus score {score:.3f} = {w1}*C1 + {w2}*C2 + {w3}*C3 | " + expl)


def difference_score(seg_a: Tuple[Paper, ...], seg_b: Tuple[Paper, ...], weights: Tuple[float,float,float]=None) -> MetricResult:
    if weights is None:
        # Load default weights from configuration
        difference_weights = _OPTIMIZATION_CONFIG["consensus_difference_weights"]["difference_internal_weights"]
        weights = (
            difference_weights["d1_keyword_js"],
            difference_weights["d2_centroid_distance"],
            difference_weights["d3_cross_citation_ratio"]
        )
    
    d1 = d1_keyword_js(seg_a, seg_b)
    d2 = d2_centroid_distance(seg_a, seg_b)
    d3 = d3_cross_citation_ratio(seg_a, seg_b)
    w1,w2,w3 = weights
    score = w1*d1.value + w2*d2.value + w3*(1-d3.value)  # note (1 - ratio) : higher different when fewer cross citations
    expl = " | ".join([d1.explanation, d2.explanation, d3.explanation])
    return MetricResult(score, f"Difference score {score:.3f} = {w1}*D1 + {w2}*D2 + {w3}*(1-D3) | " + expl)

# ---------------------------------------------------------------------------
# Fail-fast validation helper
# ---------------------------------------------------------------------------

def _validate_weights(weights: Tuple[float, float, float]):
    if abs(sum(weights) - 1.0) > 1e-6:
        raise ValueError("weights must sum to 1.0")


# ---------------------------------------------------------------------------
# Comprehensive Segmentation Evaluation
# ---------------------------------------------------------------------------

class SegmentationEvaluationResult(NamedTuple):
    """Complete result of segmentation evaluation."""
    final_score: float
    consensus_score: float
    difference_score: float
    num_segments: int
    consensus_explanation: str
    difference_explanation: str
    individual_consensus_scores: List[float]
    individual_difference_scores: List[float]
    methodology_explanation: str


def evaluate_segmentation_quality(
    segment_papers: List[Tuple[Paper, ...]],
    consensus_weights: Tuple[float, float, float] = None,
    difference_weights: Tuple[float, float, float] = None,
    final_combination_weights: Tuple[float, float] = None
) -> SegmentationEvaluationResult:
    """
    Comprehensive evaluation of segmentation quality using Phase 15 consensus-difference metrics.
    
    This is the single authoritative function for evaluating segment quality.
    All optimization and evaluation code should use this function.
    
    Args:
        segment_papers: List of segments, where each segment is a tuple of Paper objects
        consensus_weights: Weights for C1, C2, C3 metrics (defaults from optimization_config.json)
        difference_weights: Weights for D1, D2, D3 metrics (defaults from optimization_config.json)
        final_combination_weights: Weights for combining consensus and difference scores (defaults from optimization_config.json)
    
    Returns:
        SegmentationEvaluationResult with complete evaluation metrics and explanations
    
    Raises:
        ValueError: If weights don't sum to 1.0 or segments are invalid
    """
    # Load default weights from configuration if not provided
    if consensus_weights is None:
        config_consensus = _OPTIMIZATION_CONFIG["consensus_difference_weights"]["consensus_internal_weights"]
        consensus_weights = (
            config_consensus["c1_keyword_jaccard"],
            config_consensus["c2_tfidf_cohesion"],
            config_consensus["c3_citation_density"]
        )
    
    if difference_weights is None:
        config_difference = _OPTIMIZATION_CONFIG["consensus_difference_weights"]["difference_internal_weights"]
        difference_weights = (
            config_difference["d1_keyword_js"],
            config_difference["d2_centroid_distance"],
            config_difference["d3_cross_citation_ratio"]
        )
    
    if final_combination_weights is None:
        config_final = _OPTIMIZATION_CONFIG["consensus_difference_weights"]["final_combination_weights"]
        final_combination_weights = (
            config_final["consensus_weight"],
            config_final["difference_weight"]
        )
    
    # Validate inputs
    _validate_weights(consensus_weights)
    _validate_weights(difference_weights)
    
    # Validate final combination weights
    if abs(sum(final_combination_weights) - 1.0) > 1e-6:
        raise ValueError(f"final_combination_weights must sum to 1.0, got {sum(final_combination_weights):.6f}")
    
    consensus_final_weight, difference_final_weight = final_combination_weights
    
    if not segment_papers:
        raise ValueError("segment_papers cannot be empty")
    
    if any(not segment for segment in segment_papers):
        raise ValueError("All segments must contain at least one paper")
    
    # Handle single segment case
    if len(segment_papers) == 1:
        consensus_result = consensus_score(segment_papers[0], consensus_weights)
        # For single segment, final score is just the consensus score (difference is 0)
        final_score_single = consensus_final_weight * consensus_result.value + difference_final_weight * 0.0
        return SegmentationEvaluationResult(
            final_score=final_score_single,
            consensus_score=consensus_result.value,
            difference_score=0.0,
            num_segments=1,
            consensus_explanation=consensus_result.explanation,
            difference_explanation="Single segment - no transitions to evaluate",
            individual_consensus_scores=[consensus_result.value],
            individual_difference_scores=[],
            methodology_explanation=f"Single segment evaluation: final_score = {consensus_final_weight}*{consensus_result.value:.3f} + {difference_final_weight}*0.0 = {final_score_single:.3f}"
        )
    
    # Multiple segments case
    consensus_results = []
    difference_results = []
    
    # Calculate consensus for each segment
    for i, segment in enumerate(segment_papers):
        cons_result = consensus_score(segment, consensus_weights)
        consensus_results.append(cons_result)
    
    # Calculate difference between consecutive segments
    for i in range(len(segment_papers) - 1):
        diff_result = difference_score(segment_papers[i], segment_papers[i + 1], difference_weights)
        difference_results.append(diff_result)
    
    # Aggregate scores
    consensus_values = [r.value for r in consensus_results]
    difference_values = [r.value for r in difference_results]
    
    avg_consensus = float(np.mean(consensus_values))
    avg_difference = float(np.mean(difference_values))
    
    # Final score: weighted combination of consensus and difference
    final_score = consensus_final_weight * avg_consensus + difference_final_weight * avg_difference
    
    # Create comprehensive explanations
    consensus_explanations = [f"Segment {i+1}: {r.explanation}" for i, r in enumerate(consensus_results)]
    difference_explanations = [f"Transition {i+1}→{i+2}: {r.explanation}" for i, r in enumerate(difference_results)]
    
    methodology_explanation = (
        f"Multi-segment evaluation: final_score = {consensus_final_weight}*consensus + {difference_final_weight}*difference = "
        f"{consensus_final_weight}*{avg_consensus:.3f} + {difference_final_weight}*{avg_difference:.3f} = {final_score:.3f}"
    )
    
    return SegmentationEvaluationResult(
        final_score=final_score,
        consensus_score=avg_consensus,
        difference_score=avg_difference,
        num_segments=len(segment_papers),
        consensus_explanation=" | ".join(consensus_explanations),
        difference_explanation=" | ".join(difference_explanations),
        individual_consensus_scores=consensus_values,
        individual_difference_scores=difference_values,
        methodology_explanation=methodology_explanation
    ) 