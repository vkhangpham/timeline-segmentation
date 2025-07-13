"""Diagnostic utilities for direction change detection analysis.

This module provides functions to analyze and save intermediate outputs
from the direction change detection algorithm for validation and debugging.
"""

import json
import os
from typing import Dict, List, Any
import numpy as np
from collections import defaultdict
from datetime import datetime

from core.utils.logging import get_logger
from core.data.data_models import AcademicYear


def save_direction_diagnostics(
    domain_name: str,
    academic_years: List[AcademicYear],
    year_diagnostics: Dict[int, Dict[str, Any]],
    distribution_stats: Dict[str, float],
    threshold_analysis: Dict[str, Any],
    verbose: bool = False,
) -> str:
    """Save comprehensive direction change diagnostics to JSON file.

    Args:
        domain_name: Name of the domain being analyzed
        academic_years: List of AcademicYear objects
        year_diagnostics: Per-year diagnostic data
        distribution_stats: Statistical distribution analysis
        threshold_analysis: Threshold analysis results
        verbose: Enable verbose logging

    Returns:
        Path to saved diagnostics file
    """
    logger = get_logger(__name__, verbose)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"direction_diagnostics_{domain_name}_{timestamp}.json"
    filepath = os.path.join("results", "diagnostics", filename)

    # Prepare comprehensive diagnostic data
    diagnostics_data = {
        "domain_name": domain_name,
        "timestamp": timestamp,
        "total_years": len(academic_years),
        "year_range": {
            "start": min(ay.year for ay in academic_years) if academic_years else None,
            "end": max(ay.year for ay in academic_years) if academic_years else None,
        },
        "year_diagnostics": year_diagnostics,
        "distribution_stats": distribution_stats,
        "threshold_analysis": threshold_analysis,
        "metadata": {
            "generated_by": "direction_change_detection_diagnostics",
            "purpose": "Phase 1 diagnostics for OPTIMIZATION-001",
        },
    }

    # Save to file
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(diagnostics_data, f, indent=2, default=str)

    if verbose:
        logger.info(f"Saved direction diagnostics to: {filepath}")

    return filepath


def analyze_keyword_filtering_effects(
    academic_years: List[AcademicYear], domain_name: str, verbose: bool = False
) -> Dict[str, Any]:
    """Analyze effects of keyword filtering pipeline on emerging terms.

    Args:
        academic_years: List of AcademicYear objects
        domain_name: Domain name for analysis
        verbose: Enable verbose logging

    Returns:
        Dictionary containing filtering analysis results
    """
    logger = get_logger(__name__, verbose)

    if verbose:
        logger.info("Analyzing keyword filtering effects...")

    filtering_analysis = {
        "total_years_analyzed": len(academic_years),
        "years_with_keyword_data": 0,
        "avg_keywords_per_year": 0.0,
        "keyword_frequency_distribution": {},
        "emerging_terms_analysis": {},
        "temporal_keyword_evolution": {},
    }

    all_keywords = []
    yearly_keyword_counts = []

    # Analyze keyword distribution across years
    for academic_year in academic_years:
        if academic_year.keyword_frequencies:
            filtering_analysis["years_with_keyword_data"] += 1
            yearly_keyword_counts.append(len(academic_year.keyword_frequencies))
            all_keywords.extend(academic_year.keyword_frequencies.keys())

    if yearly_keyword_counts:
        filtering_analysis["avg_keywords_per_year"] = np.mean(yearly_keyword_counts)
        filtering_analysis["keyword_count_stats"] = {
            "min": min(yearly_keyword_counts),
            "max": max(yearly_keyword_counts),
            "std": np.std(yearly_keyword_counts),
        }

    # Analyze emerging terms (keywords that appear in recent years)
    recent_years = sorted([ay.year for ay in academic_years])[-10:]  # Last 10 years
    emerging_terms = defaultdict(list)

    for academic_year in academic_years:
        if academic_year.year in recent_years and academic_year.keyword_frequencies:
            for keyword, freq in academic_year.keyword_frequencies.items():
                emerging_terms[keyword].append(
                    (academic_year.year, freq, academic_year.paper_count)
                )

    # Identify terms that appear in recent years but not in top_keywords
    missed_emerging_terms = []
    for academic_year in academic_years:
        if academic_year.year in recent_years:
            if academic_year.keyword_frequencies and academic_year.top_keywords:
                all_freq_keywords = set(academic_year.keyword_frequencies.keys())
                top_keywords_set = set(academic_year.top_keywords)
                missed = all_freq_keywords - top_keywords_set

                for missed_term in missed:
                    freq = academic_year.keyword_frequencies[missed_term]
                    freq_ratio = freq / academic_year.paper_count
                    missed_emerging_terms.append(
                        {
                            "year": academic_year.year,
                            "keyword": missed_term,
                            "frequency": freq,
                            "frequency_ratio": freq_ratio,
                            "total_papers": academic_year.paper_count,
                        }
                    )

    filtering_analysis["missed_emerging_terms"] = missed_emerging_terms
    filtering_analysis["emerging_terms_count"] = len(emerging_terms)

    if verbose:
        logger.info(f"Found {len(missed_emerging_terms)} missed emerging terms")
        logger.info(f"Analyzed {len(emerging_terms)} unique emerging terms")

    return filtering_analysis


def compute_adaptive_thresholds(
    s_dir_scores: List[float], verbose: bool = False
) -> Dict[str, float]:
    """Compute various adaptive threshold strategies for S_dir scores.

    Args:
        s_dir_scores: List of S_dir scores from direction change detection
        verbose: Enable verbose logging

    Returns:
        Dictionary of threshold values for different strategies
    """
    logger = get_logger(__name__, verbose)

    if not s_dir_scores:
        logger.warning("No S_dir scores provided for threshold computation")
        return {}

    scores_array = np.array(s_dir_scores)

    thresholds = {
        "fixed_0.1": 0.1,  # Current threshold
        "global_p90": np.percentile(scores_array, 90),
        "global_p95": np.percentile(scores_array, 95),
        "global_p99": np.percentile(scores_array, 99),
        "mean_plus_1std": np.mean(scores_array) + np.std(scores_array),
        "mean_plus_1.5std": np.mean(scores_array) + 1.5 * np.std(scores_array),
        "mean_plus_2std": np.mean(scores_array) + 2 * np.std(scores_array),
        "median_plus_mad": np.median(scores_array)
        + np.median(np.abs(scores_array - np.median(scores_array))),
    }

    if verbose:
        logger.info("Computed adaptive thresholds:")
        for strategy, threshold in thresholds.items():
            logger.info(f"  {strategy}: {threshold:.4f}")

    return thresholds


def analyze_temporal_patterns(
    year_diagnostics: Dict[int, Dict[str, Any]], verbose: bool = False
) -> Dict[str, Any]:
    """Analyze temporal patterns in S_dir scores by decade.

    Args:
        year_diagnostics: Per-year diagnostic data
        verbose: Enable verbose logging

    Returns:
        Dictionary containing temporal pattern analysis
    """
    logger = get_logger(__name__, verbose)

    if verbose:
        logger.info("Analyzing temporal patterns in S_dir scores...")

    # Group years by decade
    decades = defaultdict(list)
    for year, data in year_diagnostics.items():
        decade = (year // 10) * 10
        if "s_dir" in data:
            decades[decade].append(data["s_dir"])

    decade_stats = {}
    for decade, scores in decades.items():
        if scores:
            decade_stats[f"{decade}s"] = {
                "count": len(scores),
                "mean": np.mean(scores),
                "std": np.std(scores),
                "min": min(scores),
                "max": max(scores),
                "p90": np.percentile(scores, 90),
                "p95": np.percentile(scores, 95),
            }

    temporal_analysis = {
        "decade_statistics": decade_stats,
        "trend_analysis": {},
        "score_evolution": {},
    }

    # Analyze trends over time
    if len(decade_stats) > 1:
        decades_sorted = sorted(decade_stats.keys())
        means = [decade_stats[d]["mean"] for d in decades_sorted]
        stds = [decade_stats[d]["std"] for d in decades_sorted]

        temporal_analysis["trend_analysis"] = {
            "mean_trend": "decreasing" if means[-1] < means[0] else "increasing",
            "std_trend": "decreasing" if stds[-1] < stds[0] else "increasing",
            "decades_analyzed": decades_sorted,
        }

    if verbose:
        logger.info(f"Analyzed {len(decade_stats)} decades")
        for decade, stats in decade_stats.items():
            logger.info(f"  {decade}: mean={stats['mean']:.4f}, std={stats['std']:.4f}")

    return temporal_analysis


def generate_keyword_evolution_report(
    academic_years: List[AcademicYear], focus_years: List[int], verbose: bool = False
) -> Dict[str, Any]:
    """Generate detailed keyword evolution report for specific focus years.

    Args:
        academic_years: List of AcademicYear objects
        focus_years: Years to focus analysis on (e.g., known paradigm shift years)
        verbose: Enable verbose logging

    Returns:
        Dictionary containing keyword evolution analysis
    """
    logger = get_logger(__name__, verbose)

    if verbose:
        logger.info(f"Generating keyword evolution report for years: {focus_years}")

    year_map = {ay.year: ay for ay in academic_years}
    evolution_report = {}

    for focus_year in focus_years:
        if focus_year not in year_map:
            continue

        focus_ay = year_map[focus_year]

        # Find previous year for comparison
        prev_year = focus_year - 1
        if prev_year not in year_map:
            continue

        prev_ay = year_map[prev_year]

        # Analyze keyword changes
        prev_keywords = set(prev_ay.top_keywords) if prev_ay.top_keywords else set()
        curr_keywords = set(focus_ay.top_keywords) if focus_ay.top_keywords else set()

        new_keywords = curr_keywords - prev_keywords
        lost_keywords = prev_keywords - curr_keywords
        shared_keywords = curr_keywords & prev_keywords

        evolution_report[focus_year] = {
            "previous_year": prev_year,
            "new_keywords": list(new_keywords),
            "lost_keywords": list(lost_keywords),
            "shared_keywords": list(shared_keywords),
            "new_keywords_count": len(new_keywords),
            "lost_keywords_count": len(lost_keywords),
            "shared_keywords_count": len(shared_keywords),
            "total_keywords_prev": len(prev_keywords),
            "total_keywords_curr": len(curr_keywords),
            "keyword_frequencies": (
                dict(focus_ay.keyword_frequencies)
                if focus_ay.keyword_frequencies
                else {}
            ),
        }

        # Calculate novelty and overlap manually
        if curr_keywords and prev_keywords:
            novelty = len(new_keywords) / len(curr_keywords)
            overlap = len(shared_keywords) / len(prev_keywords)
            s_dir = novelty * (1 - overlap)

            evolution_report[focus_year]["metrics"] = {
                "novelty": novelty,
                "overlap": overlap,
                "s_dir": s_dir,
            }

    if verbose:
        logger.info(f"Generated evolution report for {len(evolution_report)} years")

    return evolution_report
