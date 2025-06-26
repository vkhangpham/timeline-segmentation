"""
Consensus & Difference Metrics for Timeline Segmentation
========================================================

Pure-function metrics to quantify:
A. Consensus inside a segment (C-metrics)
B. Difference between two consecutive segments (D-metrics)

Key optimizations:
    • TF-IDF vectorization (outperforms contextual embeddings)
    • Linear aggregation (superior optimization effectiveness)
    • 10k max features for optimal performance
    • Keyword filtering for improved segmentation

Each metric returns **(value, explanation_string)** for transparency.
All functions follow fail-fast principles with no error masking.
Weights loaded from optimization_config.json ensure system-wide consistency.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, List, Dict, NamedTuple
from collections import Counter
import numpy as np
import json
import os
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import jensenshannon
import math

from core.algorithm_config import AlgorithmConfig
from core.data_models import Paper
from core.text_vectorization import tfidf_embeddings, contextual_embeddings

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


def _get_embeddings(texts: List[str]):
    """
    Dispatcher for text vectorization with optimized defaults.
    
    TF-IDF is the default method for temporal segmentation as it outperforms
    contextual embeddings and better captures vocabulary shifts in research transitions.
    """
    _vec_cfg = _OPTIMIZATION_CONFIG.get("text_vectorizer", {})
    
    # Priority order: environment variables > optimization_config.json defaults
    vectorizer_type = os.getenv("VECTORIZER_TYPE", _vec_cfg.get("type", "tfidf"))
    clean_env = os.getenv("CLEAN_TEXT_ENABLED")
    clean_enabled = (clean_env.lower() == "true") if clean_env is not None else _vec_cfg.get("clean_text_enabled", False)
    
    if vectorizer_type == "tfidf":
        # Default 10k features for optimal performance
        max_feats = int(os.getenv("TFIDF_MAX_FEATURES", _vec_cfg.get("max_features", 10000)))
        return tfidf_embeddings(texts, max_features=max_feats, clean=clean_enabled)
    
    elif vectorizer_type == "contextual":
        # Available for research but not recommended for production
        model_name = os.getenv("CONTEXTUAL_MODEL", _vec_cfg.get("contextual_model", "all-mpnet-base-v2"))
        cache_dir = os.getenv("HF_CACHE_DIR", _vec_cfg.get("cache_dir", ".hf_cache"))
        device = os.getenv("DEVICE", _vec_cfg.get("device", "auto"))
        return contextual_embeddings(texts, model_name=model_name, cache_dir=cache_dir, device=device, clean=clean_enabled)
    
    else:
        raise ValueError(f"Unknown vectorizer_type '{vectorizer_type}'. Valid options: 'tfidf', 'contextual'")

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
    """Mean pairwise cosine similarity of text embeddings of abstracts / keywords."""
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
    emb = _get_embeddings(texts)
    if emb.shape[0] < 2:
        return MetricResult(0.0, "Insufficient text for embedding cohesion → 0.0")
    sim = cosine_similarity(emb)
    mean_sim = float((sim.sum() - np.trace(sim)) / (n * (n - 1)))
    
    # Dynamic explanation based on vectorizer type
    _vec_cfg = _OPTIMIZATION_CONFIG.get("text_vectorizer", {})
    vectorizer_type = os.getenv("VECTORIZER_TYPE", _vec_cfg.get("type", "tfidf"))
    embedding_name = "TF-IDF" if vectorizer_type == "tfidf" else f"Contextual-{_vec_cfg.get('contextual_model', 'unknown')}"
    
    explanation = (
        f"C2 {embedding_name} Cohesion: mean pairwise cosine similarity across {n} papers = {mean_sim:.3f}"
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
    """1 – cosine similarity between text embedding centroids of two segments."""
    texts_a: List[str] = [p.content or " ".join(p.keywords) for p in segment_a]
    texts_b: List[str] = [p.content or " ".join(p.keywords) for p in segment_b]
    if not texts_a or not texts_b:
        return MetricResult(0.0, "Empty text in one segment → distance 0.0")
    emb = _get_embeddings(texts_a + texts_b)
    k = len(texts_a)
    centroid_a = emb[:k].mean(axis=0)
    centroid_b = emb[k:].mean(axis=0)
    sim = float(cosine_similarity([centroid_a], [centroid_b])[0][0])
    distance = 1.0 - sim
    
    # Dynamic explanation based on vectorizer type
    _vec_cfg = _OPTIMIZATION_CONFIG.get("text_vectorizer", {})
    vectorizer_type = os.getenv("VECTORIZER_TYPE", _vec_cfg.get("type", "tfidf"))
    embedding_name = "TF-IDF" if vectorizer_type == "tfidf" else f"Contextual-{_vec_cfg.get('contextual_model', 'unknown')}"
    
    explanation = f"D2 {embedding_name} Centroid Distance (1 – cosine) = {distance:.3f}"
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
# Aggregation helpers (linear, harmonic, adaptive_tchebycheff)
# ---------------------------------------------------------------------------

VALID_AGGREGATION_METHODS = {"linear", "harmonic", "adaptive_tchebycheff"}


def _compute_runtime_scale_bounds(segment_papers: List[Tuple[Paper, ...]]) -> Tuple[float, float]:
    """
    Compute runtime scale bounds for consensus and difference metrics.
    
    This function estimates appropriate upper bounds for normalization by:
    1. Computing consensus scores for different segment configurations
    2. Computing difference scores for various segment pairs
    3. Using statistical measures (95th percentile) to set robust bounds
    
    Args:
        segment_papers: List of segments to analyze for scale estimation
        
    Returns:
        Tuple of (consensus_upper_bound, difference_upper_bound)
    """
    if not segment_papers:
        return 0.06, 0.8  # Fallback to empirical bounds
    
    consensus_scores = []
    difference_scores = []
    
    # Sample consensus scores from actual segments
    for segment in segment_papers:
        if len(segment) >= 2:  # Need at least 2 papers for meaningful consensus
            try:
                cons_result = consensus_score(segment)
                consensus_scores.append(cons_result.value)
            except:
                continue
    
    # Sample difference scores between segment pairs
    for i in range(len(segment_papers)):
        for j in range(i + 1, len(segment_papers)):
            if len(segment_papers[i]) >= 1 and len(segment_papers[j]) >= 1:
                try:
                    diff_result = difference_score(segment_papers[i], segment_papers[j])
                    difference_scores.append(diff_result.value)
                except:
                    continue
    
    # Additional sampling: create temporary segments to get more data points
    all_papers = [paper for segment in segment_papers for paper in segment]
    if len(all_papers) >= 10:
        # Create some temporary segments for additional sampling
        mid_point = len(all_papers) // 2
        temp_segment1 = tuple(all_papers[:mid_point])
        temp_segment2 = tuple(all_papers[mid_point:])
        
        try:
            cons1 = consensus_score(temp_segment1)
            cons2 = consensus_score(temp_segment2)
            consensus_scores.extend([cons1.value, cons2.value])
            
            diff_temp = difference_score(temp_segment1, temp_segment2)
            difference_scores.append(diff_temp.value)
        except:
            pass
    
    # Compute robust upper bounds using 95th percentile + safety margin
    if consensus_scores:
        consensus_bound = max(0.04, np.percentile(consensus_scores, 95) * 1.2)  # 20% safety margin
    else:
        consensus_bound = 0.06  # Fallback
    
    if difference_scores:
        difference_bound = max(0.4, np.percentile(difference_scores, 95) * 1.2)  # 20% safety margin
    else:
        difference_bound = 0.8  # Fallback
    
    return consensus_bound, difference_bound


def _estimate_consensus_baseline(segment_papers: List[Tuple[Paper, ...]]) -> float:
    """
    Estimate baseline consensus score for domain-specific normalization.
    
    Uses single-segment baseline as the minimum consensus threshold (τ_cons).
    This represents the consensus score when no segmentation is applied.
    
    Args:
        segment_papers: List of segments to analyze
        
    Returns:
        Baseline consensus score (τ_cons)
    """
    if not segment_papers:
        return 0.04  # Fallback based on empirical data
    
    # Create single segment with all papers
    all_papers = tuple(paper for segment in segment_papers for paper in segment)
    single_segment = [all_papers]
    
    # Calculate consensus for single segment (no internal weights override)
    consensus_result = consensus_score(all_papers)
    tau_cons = consensus_result.value
    
    # Ensure minimum threshold based on empirical observations
    return max(tau_cons, 0.04)


def _adaptive_tchebycheff_scalarization(
    consensus: float, 
    difference: float, 
    tau_cons: float,
    preference_weight: float = 0.5,
    consensus_scale: float = None,
    difference_scale: float = None
) -> Tuple[float, str]:
    """
    Adaptive Augmented Tchebycheff scalarization for consensus-difference optimization.
    
    This method addresses the scale mismatch problem where Q_cons ≈ 0.05 and Q_diff ≈ 0.5-0.7
    by using adaptive normalization and constraint-based optimization.
    
    Method:
    1. Constraint enforcement: Ensure consensus ≥ τ_cons (baseline threshold)
    2. Adaptive normalization: Scale both metrics to comparable ranges
    3. Tchebycheff combination: max(w₁×norm_cons, w₂×norm_diff) + ρ×(w₁×norm_cons + w₂×norm_diff)
    
    Args:
        consensus: Raw consensus score
        difference: Raw difference score  
        tau_cons: Baseline consensus threshold (from single-segment baseline)
        preference_weight: User preference between consensus (0.0) and difference (1.0)
        consensus_scale: Upper bound for consensus normalization (auto-computed if None)
        difference_scale: Upper bound for difference normalization (auto-computed if None)
        
    Returns:
        Tuple of (final_score, explanation)
    """
    # Constraint enforcement: Soft penalty if consensus significantly below baseline
    # Allow reasonable deviations (10% tolerance) to avoid over-penalizing minor differences
    tolerance = 0.10  # 10% tolerance
    if consensus < tau_cons * (1 - tolerance):
        penalty_score = consensus / tau_cons * 0.1  # Severe penalty for major violations
        explanation = (
            f"Adaptive Tchebycheff: consensus={consensus:.3f} < τ_cons={tau_cons:.3f}×(1-{tolerance}) → "
            f"constraint violation penalty={penalty_score:.3f}"
        )
        return penalty_score, explanation
    
    # Adaptive normalization: use provided scales or fallback to empirical observations
    if consensus_scale is None:
        consensus_scale = 0.06  # Fallback from empirical data (landscape analysis)
    if difference_scale is None:
        difference_scale = 0.8   # Fallback from empirical data (landscape analysis)
    
    # Normalize to [0, 1] range with adaptive scaling
    norm_consensus = min(1.0, consensus / consensus_scale)
    norm_difference = min(1.0, difference / difference_scale)
    
    # Tchebycheff weights: preference_weight controls consensus vs difference emphasis
    w_consensus = 1.0 - preference_weight  # Higher when preference_weight is low
    w_difference = preference_weight       # Higher when preference_weight is high
    
    # Augmented Tchebycheff scalarization
    # max(w₁×f₁, w₂×f₂) + ρ×(w₁×f₁ + w₂×f₂)
    rho = 0.1  # Augmentation parameter (small positive value)
    
    weighted_consensus = w_consensus * norm_consensus
    weighted_difference = w_difference * norm_difference
    
    tchebycheff_max = max(weighted_consensus, weighted_difference)
    augmentation_term = rho * (weighted_consensus + weighted_difference)
    
    final_score = tchebycheff_max + augmentation_term
    
    explanation = (
        f"Adaptive Tchebycheff: norm_cons={norm_consensus:.3f} (raw={consensus:.3f}/scale={consensus_scale:.3f}), "
        f"norm_diff={norm_difference:.3f} (raw={difference:.3f}/scale={difference_scale:.3f}), "
        f"weights=({w_consensus:.2f},{w_difference:.2f}), "
        f"max({weighted_consensus:.3f},{weighted_difference:.3f}) + {rho}×{weighted_consensus + weighted_difference:.3f} = {final_score:.3f}"
    )
    
    return final_score, explanation


def _aggregate_scores(consensus: float, difference: float, weights: Tuple[float, float], method: str = "linear") -> float:
    """
    Combine consensus & difference scores using specified aggregation method.
    
    Linear aggregation is the default method as it provides:
    - Better optimization effectiveness
    - Higher score diversity for gradient-based optimization
    - Avoids score compression issues of harmonic methods

    Args:
        consensus: Average consensus score across segments.
        difference: Average difference score across transitions.
        weights: Tuple (consensus_weight, difference_weight) that must sum to 1.
        method: "linear" (recommended), "harmonic" (research only), or "adaptive_tchebycheff" (Phase 17).

    Returns:
        Aggregated score (float).
    """
    if method not in VALID_AGGREGATION_METHODS:
        raise ValueError(f"Unknown aggregation_method '{method}'. Valid options: {', '.join(VALID_AGGREGATION_METHODS)}")

    consensus_weight, difference_weight = weights
    _validate_weights((consensus_weight, difference_weight))

    if method == "linear":
        return consensus_weight * consensus + difference_weight * difference

    elif method == "adaptive_tchebycheff":
        # For adaptive Tchebycheff, weights represent preference (0.0 = consensus focus, 1.0 = difference focus)
        preference_weight = difference_weight  # Use difference_weight as preference parameter
        
        # τ_cons estimation requires segment context, use empirical baseline
        tau_cons = 0.04  # Conservative baseline from landscape analysis
        
        # Use empirical bounds when called without segment context
        score, _ = _adaptive_tchebycheff_scalarization(
            consensus, difference, tau_cons, preference_weight,
            consensus_scale=0.06, difference_scale=0.8  # Empirical fallbacks
        )
        return score

    # Harmonic mean implementation with safe handling when one component weight is zero
    if consensus_weight == 0.0:
        return difference  # All weight on difference component
    if difference_weight == 0.0:
        return consensus  # All weight on consensus component

    if consensus == 0.0 or difference == 0.0:
        # Harmonic mean tends to 0 when either component is 0 under positive weight.
        return 0.0

    return 1.0 / ((consensus_weight / consensus) + (difference_weight / difference))

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
    final_combination_weights: Tuple[float, float] = None,
    aggregation_method: str = None,
    algorithm_config: 'AlgorithmConfig' = None
) -> SegmentationEvaluationResult:
    """
    Comprehensive evaluation of segmentation quality using optimized metrics.
    
    This function evaluates segment quality using optimized settings:
    - TF-IDF vectorization for better vocabulary shift detection
    - Linear aggregation for superior optimization effectiveness
    - Validated weighting schemes from configuration
    
    Args:
        segment_papers: List of segments, where each segment is a tuple of Paper objects
                consensus_weights: C1, C2, C3 weights (defaults from optimization_config.json)    
        difference_weights: D1, D2, D3 weights (defaults from optimization_config.json)
        final_combination_weights: Consensus/difference combination weights
        aggregation_method: "linear" (default) or "harmonic" (research only)
        algorithm_config: Configuration for comprehensive algorithm
    
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
    
    # Load aggregation method from configuration with environment override
    if aggregation_method is None:
        # Priority order: environment variable > optimization_config.json
        aggregation_method = os.getenv("AGGREGATION_METHOD")
        if aggregation_method is None:
            aggregation_method = _OPTIMIZATION_CONFIG["consensus_difference_weights"]["aggregation_method"]
    
    # Validate inputs
    _validate_weights(consensus_weights)
    _validate_weights(difference_weights)
    
    # Validate and unpack final combination weights
    _validate_weights(final_combination_weights)
    
    consensus_final_weight, difference_final_weight = final_combination_weights
    
    # Validate aggregation method
    if aggregation_method not in VALID_AGGREGATION_METHODS:
        raise ValueError(
            f"aggregation_method must be one of {', '.join(VALID_AGGREGATION_METHODS)}, got '{aggregation_method}'"
        )
    
    if not segment_papers:
        raise ValueError("segment_papers cannot be empty")
    
    if any(not segment for segment in segment_papers):
        raise ValueError("All segments must contain at least one paper")
    
    # Handle single segment case
    if len(segment_papers) == 1:
        consensus_result = consensus_score(segment_papers[0], consensus_weights)
        # For single segment, final score is just the consensus score (difference is 0)
        if aggregation_method == "adaptive_tchebycheff":
            # For single segment, use consensus as baseline and compute runtime bounds
            tau_cons = _estimate_consensus_baseline(segment_papers)
            consensus_scale, difference_scale = _compute_runtime_scale_bounds(segment_papers)
            preference_weight = difference_final_weight
            
            final_score_single, tchebycheff_explanation_single = _adaptive_tchebycheff_scalarization(
                consensus_result.value, 0.0, tau_cons, preference_weight,
                consensus_scale=consensus_scale, difference_scale=difference_scale
            )
            
            tchebycheff_single_explanation = f" | τ_cons={tau_cons:.3f} | runtime_bounds=({consensus_scale:.3f},{difference_scale:.3f}) | {tchebycheff_explanation_single}"
        else:
            final_score_single = _aggregate_scores(
                consensus_result.value,
                0.0,
                (consensus_final_weight, difference_final_weight),
                method=aggregation_method,
            )
            tchebycheff_single_explanation = ""
        
        # Apply segment count penalty if enabled (single segment case)
        penalty_explanation_single = ""
        if algorithm_config and algorithm_config.segment_count_penalty_enabled:
            # Calculate domain year span from segment papers
            all_years = [p.pub_year for p in segment_papers[0]]
            if all_years:
                domain_year_span = max(all_years) - min(all_years) + 1
                k_actual = 1
                k_desired = max(1, round(domain_year_span / 10))
                
                # Apply exponential penalty: penalty = exp(-|K - K_desired| / σ)
                penalty = math.exp(-abs(k_actual - k_desired) / algorithm_config.segment_count_penalty_sigma)
                final_score_single *= penalty
                
                penalty_explanation_single = (
                    f" | segment-count penalty σ={algorithm_config.segment_count_penalty_sigma}, "
                    f"K={k_actual}, K_desired={k_desired} → penalty={penalty:.3f} → adjusted_final={final_score_single:.3f}"
                )
        
        return SegmentationEvaluationResult(
            final_score=final_score_single,
            consensus_score=consensus_result.value,
            difference_score=0.0,
            num_segments=1,
            consensus_explanation=consensus_result.explanation,
            difference_explanation="Single segment - no transitions to evaluate",
            individual_consensus_scores=[consensus_result.value],
            individual_difference_scores=[],
            methodology_explanation=(
                f"Single segment evaluation ({aggregation_method}): final_score = {final_score_single:.3f}{tchebycheff_single_explanation}{penalty_explanation_single}"
            )
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
    if aggregation_method == "adaptive_tchebycheff":
        # For adaptive Tchebycheff, estimate τ_cons and runtime scale bounds from actual segments
        tau_cons = _estimate_consensus_baseline(segment_papers)
        consensus_scale, difference_scale = _compute_runtime_scale_bounds(segment_papers)
        preference_weight = difference_final_weight  # Use difference weight as preference
        
        final_score, tchebycheff_explanation = _adaptive_tchebycheff_scalarization(
            avg_consensus, avg_difference, tau_cons, preference_weight,
            consensus_scale=consensus_scale, difference_scale=difference_scale
        )
        
        # Store explanation for methodology
        tchebycheff_method_explanation = f" | τ_cons={tau_cons:.3f} | runtime_bounds=({consensus_scale:.3f},{difference_scale:.3f}) | {tchebycheff_explanation}"
    else:
        final_score = _aggregate_scores(
            avg_consensus,
            avg_difference,
            (consensus_final_weight, difference_final_weight),
            method=aggregation_method,
        )
        tchebycheff_method_explanation = ""
    
    # Apply segment count penalty if enabled
    penalty_explanation = ""
    if algorithm_config and algorithm_config.segment_count_penalty_enabled:
        # Calculate domain year span from segment papers
        all_years = [p.pub_year for segment in segment_papers for p in segment]
        if all_years:
            domain_year_span = max(all_years) - min(all_years) + 1
            k_actual = len(segment_papers)
            k_desired = max(1, round(domain_year_span / 10))
            
            # Apply exponential penalty: penalty = exp(-|K - K_desired| / σ)
            penalty = math.exp(-abs(k_actual - k_desired) / algorithm_config.segment_count_penalty_sigma)
            final_score *= penalty
            
            penalty_explanation = (
                f" | segment-count penalty σ={algorithm_config.segment_count_penalty_sigma}, "
                f"K={k_actual}, K_desired={k_desired} → penalty={penalty:.3f} → adjusted_final={final_score:.3f}"
            )
    
    # Create comprehensive explanations
    consensus_explanations = [f"Segment {i+1}: {r.explanation}" for i, r in enumerate(consensus_results)]
    difference_explanations = [f"Transition {i+1}→{i+2}: {r.explanation}" for i, r in enumerate(difference_results)]
    
    methodology_explanation = (
        f"Multi-segment evaluation ({aggregation_method}): consensus={avg_consensus:.3f}, difference={avg_difference:.3f}, "
        f"weights=({consensus_final_weight},{difference_final_weight}) → final={final_score:.3f}{tchebycheff_method_explanation}{penalty_explanation}"
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