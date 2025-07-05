#!/usr/bin/env python3
"""
Comprehensive Component Analysis for Anti-Gaming Keyword Metrics
===============================================================

This script runs detailed component analysis of individual cohesion and separation metrics
using the validated anti-gaming safeguards. It provides:

1. Individual metric score distributions across domains
2. Expert timeline percentile placements for each component
3. Correlation analysis between cohesion/separation metrics  
4. Concrete examples of high vs low scoring segments
5. Comparative analysis against K-stratified baselines

VALIDATION CRITERIA:
- Expert timelines should show meaningful differentiation in component scores
- Individual metrics should correlate appropriately with overall quality
- Anti-gaming safeguards should prevent component-level gaming
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import time
import random
from typing import Dict, List, Tuple, Any
from scipy.stats import pearsonr, spearmanr
from collections import defaultdict

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core_metrics import (
    load_domain_papers,
    load_reference_timelines,
    filter_keywords,
    evaluate_segmentation,
    generate_k_stratified_baselines,
    papers_to_segments,
    AntiGamingConfig,
    SegmentationEvaluation,
    cohesion_mean_jaccard_to_union,
    cohesion_keyword_entropy,
    separation_jensen_shannon,
    separation_topk_overlap
)


def analyze_individual_components(domain_name: str, num_baselines: int = 200) -> Dict[str, Any]:
    """
    Analyze individual cohesion and separation components for a domain.
    
    Args:
        domain_name: Name of domain to analyze
        num_baselines: Number of K-stratified baselines to generate
        
    Returns:
        Dictionary with comprehensive component analysis results
    """
    print(f"\n{'='*70}")
    print(f"COMPONENT ANALYSIS: {domain_name.upper()}")
    print(f"{'='*70}")
    
    analysis_start_time = time.time()
    
    # 1. Load and filter data
    print("1. Loading and filtering data...")
    try:
        papers = load_domain_papers(domain_name)
        manual_segments, gemini_segments = load_reference_timelines(domain_name)
        filtered_papers, valid_keywords = filter_keywords(papers, min_years=2, min_paper_ratio=0.01)
        
        print(f"   Papers: {len(papers)} → {len(filtered_papers)} (after filtering)")
        print(f"   Keywords: {len(valid_keywords)} valid")
        print(f"   Year range: {min(p.pub_year for p in papers)}-{max(p.pub_year for p in papers)}")
        
        if len(filtered_papers) < 100:
            print(f"   WARNING: Only {len(filtered_papers)} papers - insufficient for robust analysis")
            return {"success": False, "error": "Insufficient papers"}
            
    except Exception as e:
        print(f"   ERROR: Failed to load data - {e}")
        return {"success": False, "error": str(e)}
    
    # 2. Configure anti-gaming safeguards
    print("\n2. Configuring anti-gaming safeguards...")
    anti_gaming_config = AntiGamingConfig(
        min_segment_size=50,
        size_weight_power=0.5,
        enable_size_weighting=True,
        enable_segment_floor=True,
        enable_count_penalty=False
    )
    
    print(f"   Size-weighted averaging: {anti_gaming_config.enable_size_weighting}")
    print(f"   Minimum segment size: {anti_gaming_config.min_segment_size} papers")
    print(f"   Size weight power: {anti_gaming_config.size_weight_power}")
    
    # 3. Analyze reference timelines
    print("\n3. Analyzing reference timeline components...")
    
    # Convert segments to paper lists
    manual_paper_segments = papers_to_segments(filtered_papers, manual_segments)
    gemini_paper_segments = papers_to_segments(filtered_papers, gemini_segments)
    
    # Filter segments by minimum size
    manual_valid_segments = [seg for seg in manual_paper_segments if len(seg) >= anti_gaming_config.min_segment_size]
    gemini_valid_segments = [seg for seg in gemini_paper_segments if len(seg) >= anti_gaming_config.min_segment_size]
    
    print(f"   Manual segments: {len(manual_paper_segments)} → {len(manual_valid_segments)} (after size filter)")
    print(f"   Gemini segments: {len(gemini_paper_segments)} → {len(gemini_valid_segments)} (after size filter)")
    
    # Analyze individual components for reference timelines
    def analyze_timeline_components(segments, timeline_name):
        """Analyze individual components for a timeline."""
        if not segments:
            return {"error": "No valid segments"}
        
        # Individual cohesion scores
        cohesion_jaccard_scores = []
        cohesion_entropy_scores = []
        
        # Individual separation scores  
        separation_js_scores = []
        separation_topk_scores = []
        
        # Segment-level analysis
        segment_details = []
        
        for i, segment in enumerate(segments):
            # Cohesion metrics
            jaccard_result = cohesion_mean_jaccard_to_union(segment)
            entropy_result = cohesion_keyword_entropy(segment)
            
            cohesion_jaccard_scores.append(jaccard_result.value)
            cohesion_entropy_scores.append(entropy_result.value)
            
            segment_details.append({
                "segment_id": i + 1,
                "size": len(segment),
                "year_range": (min(p.pub_year for p in segment), max(p.pub_year for p in segment)),
                "cohesion_jaccard": jaccard_result.value,
                "cohesion_entropy": entropy_result.value,
                "top_keywords": [kw for kw, count in 
                               sorted([(kw, sum(1 for p in segment if kw in p.keywords)) 
                                     for kw in valid_keywords], 
                                     key=lambda x: x[1], reverse=True)[:10]]
            })
        
        # Separation metrics (between consecutive segments)
        for i in range(len(segments) - 1):
            js_result = separation_jensen_shannon(segments[i], segments[i + 1])
            topk_result = separation_topk_overlap(segments[i], segments[i + 1])
            
            separation_js_scores.append(js_result.value)
            separation_topk_scores.append(topk_result.value)
        
        # Compute statistics
        def compute_stats(scores):
            if not scores:
                return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "median": 0.0}
            return {
                "mean": np.mean(scores),
                "std": np.std(scores),
                "min": np.min(scores),
                "max": np.max(scores),
                "median": np.median(scores),
                "q25": np.percentile(scores, 25),
                "q75": np.percentile(scores, 75)
            }
        
        return {
            "timeline_name": timeline_name,
            "num_segments": len(segments),
            "num_transitions": len(segments) - 1,
            "cohesion_jaccard": {
                "scores": cohesion_jaccard_scores,
                "stats": compute_stats(cohesion_jaccard_scores)
            },
            "cohesion_entropy": {
                "scores": cohesion_entropy_scores,
                "stats": compute_stats(cohesion_entropy_scores)
            },
            "separation_js": {
                "scores": separation_js_scores,
                "stats": compute_stats(separation_js_scores)
            },
            "separation_topk": {
                "scores": separation_topk_scores,
                "stats": compute_stats(separation_topk_scores)
            },
            "segment_details": segment_details
        }
    
    manual_analysis = analyze_timeline_components(manual_valid_segments, "Manual")
    gemini_analysis = analyze_timeline_components(gemini_valid_segments, "Gemini")
    
    # 4. Generate K-stratified baselines for component analysis
    print("\n4. Generating K-stratified baselines for component analysis...")
    
    # Generate baselines for both reference timelines
    manual_k_stratified = []
    gemini_k_stratified = []
    
    if manual_valid_segments:
        manual_sizes = [len(seg) for seg in manual_valid_segments]
        manual_k_stratified = generate_k_stratified_baselines(
            filtered_papers, len(manual_sizes), manual_sizes, num_samples=num_baselines
        )
        print(f"   Generated {len(manual_k_stratified)} manual K-stratified baselines")
    
    if gemini_valid_segments:
        gemini_sizes = [len(seg) for seg in gemini_valid_segments]
        gemini_k_stratified = generate_k_stratified_baselines(
            filtered_papers, len(gemini_sizes), gemini_sizes, num_samples=num_baselines
        )
        print(f"   Generated {len(gemini_k_stratified)} Gemini K-stratified baselines")
    
    # 5. Analyze baseline components
    print("\n5. Analyzing baseline component distributions...")
    
    def analyze_baseline_components(k_stratified_list, baseline_name):
        """Analyze component distributions for K-stratified baselines."""
        if not k_stratified_list:
            return {"error": "No baselines available"}
        
        all_cohesion_jaccard = []
        all_cohesion_entropy = []
        all_separation_js = []
        all_separation_topk = []
        
        for baseline_segments in k_stratified_list:
            # Convert to paper segments and filter by size
            baseline_paper_segments = papers_to_segments(filtered_papers, baseline_segments)
            valid_baseline_segments = [seg for seg in baseline_paper_segments 
                                     if len(seg) >= anti_gaming_config.min_segment_size]
            
            if not valid_baseline_segments:
                continue
            
            # Cohesion scores
            for segment in valid_baseline_segments:
                jaccard_result = cohesion_mean_jaccard_to_union(segment)
                entropy_result = cohesion_keyword_entropy(segment)
                all_cohesion_jaccard.append(jaccard_result.value)
                all_cohesion_entropy.append(entropy_result.value)
            
            # Separation scores
            for i in range(len(valid_baseline_segments) - 1):
                js_result = separation_jensen_shannon(valid_baseline_segments[i], 
                                                    valid_baseline_segments[i + 1])
                topk_result = separation_topk_overlap(valid_baseline_segments[i], 
                                                    valid_baseline_segments[i + 1])
                all_separation_js.append(js_result.value)
                all_separation_topk.append(topk_result.value)
        
        def compute_stats_with_percentiles(scores):
            if not scores:
                return {"mean": 0.0, "std": 0.0, "percentiles": {}}
            return {
                "mean": np.mean(scores),
                "std": np.std(scores),
                "min": np.min(scores),
                "max": np.max(scores),
                "median": np.median(scores),
                "percentiles": {p: np.percentile(scores, p) for p in [5, 10, 25, 50, 75, 90, 95]}
            }
        
        return {
            "baseline_name": baseline_name,
            "num_baselines": len(k_stratified_list),
            "total_segments_analyzed": len(all_cohesion_jaccard),
            "total_transitions_analyzed": len(all_separation_js),
            "cohesion_jaccard": {
                "all_scores": all_cohesion_jaccard,
                "stats": compute_stats_with_percentiles(all_cohesion_jaccard)
            },
            "cohesion_entropy": {
                "all_scores": all_cohesion_entropy,
                "stats": compute_stats_with_percentiles(all_cohesion_entropy)
            },
            "separation_js": {
                "all_scores": all_separation_js,
                "stats": compute_stats_with_percentiles(all_separation_js)
            },
            "separation_topk": {
                "all_scores": all_separation_topk,
                "stats": compute_stats_with_percentiles(all_separation_topk)
            }
        }
    
    manual_baseline_analysis = analyze_baseline_components(manual_k_stratified, "Manual K-Stratified")
    gemini_baseline_analysis = analyze_baseline_components(gemini_k_stratified, "Gemini K-Stratified")
    
    # 6. Compute percentile rankings for expert timelines
    print("\n6. Computing expert timeline percentile rankings...")
    
    def compute_component_percentiles(expert_analysis, baseline_analysis):
        """Compute percentile rankings for expert timeline components."""
        if "error" in expert_analysis or "error" in baseline_analysis:
            return {"error": "Missing data for percentile computation"}
        
        percentiles = {}
        
        # Cohesion Jaccard percentiles
        expert_jaccard_scores = expert_analysis["cohesion_jaccard"]["scores"]
        baseline_jaccard_scores = baseline_analysis["cohesion_jaccard"]["all_scores"]
        
        if expert_jaccard_scores and baseline_jaccard_scores:
            jaccard_percentiles = []
            for score in expert_jaccard_scores:
                percentile = (np.sum(np.array(baseline_jaccard_scores) <= score) / len(baseline_jaccard_scores)) * 100
                jaccard_percentiles.append(percentile)
            percentiles["cohesion_jaccard"] = {
                "individual_percentiles": jaccard_percentiles,
                "mean_percentile": np.mean(jaccard_percentiles)
            }
        
        # Cohesion Entropy percentiles
        expert_entropy_scores = expert_analysis["cohesion_entropy"]["scores"]
        baseline_entropy_scores = baseline_analysis["cohesion_entropy"]["all_scores"]
        
        if expert_entropy_scores and baseline_entropy_scores:
            entropy_percentiles = []
            for score in expert_entropy_scores:
                percentile = (np.sum(np.array(baseline_entropy_scores) <= score) / len(baseline_entropy_scores)) * 100
                entropy_percentiles.append(percentile)
            percentiles["cohesion_entropy"] = {
                "individual_percentiles": entropy_percentiles,
                "mean_percentile": np.mean(entropy_percentiles)
            }
        
        # Separation JS percentiles
        expert_js_scores = expert_analysis["separation_js"]["scores"]
        baseline_js_scores = baseline_analysis["separation_js"]["all_scores"]
        
        if expert_js_scores and baseline_js_scores:
            js_percentiles = []
            for score in expert_js_scores:
                percentile = (np.sum(np.array(baseline_js_scores) <= score) / len(baseline_js_scores)) * 100
                js_percentiles.append(percentile)
            percentiles["separation_js"] = {
                "individual_percentiles": js_percentiles,
                "mean_percentile": np.mean(js_percentiles)
            }
        
        # Separation TopK percentiles
        expert_topk_scores = expert_analysis["separation_topk"]["scores"]
        baseline_topk_scores = baseline_analysis["separation_topk"]["all_scores"]
        
        if expert_topk_scores and baseline_topk_scores:
            topk_percentiles = []
            for score in expert_topk_scores:
                percentile = (np.sum(np.array(baseline_topk_scores) <= score) / len(baseline_topk_scores)) * 100
                topk_percentiles.append(percentile)
            percentiles["separation_topk"] = {
                "individual_percentiles": topk_percentiles,
                "mean_percentile": np.mean(topk_percentiles)
            }
        
        return percentiles
    
    manual_percentiles = compute_component_percentiles(manual_analysis, manual_baseline_analysis)
    gemini_percentiles = compute_component_percentiles(gemini_analysis, gemini_baseline_analysis)
    
    # 7. Correlation analysis
    print("\n7. Performing correlation analysis...")
    
    def compute_correlations(analysis):
        """Compute correlations between different component metrics."""
        if "error" in analysis:
            return {"error": "Missing data for correlation analysis"}
        
        # Extract scores
        jaccard_scores = analysis["cohesion_jaccard"]["scores"]
        entropy_scores = analysis["cohesion_entropy"]["scores"]
        js_scores = analysis["separation_js"]["scores"]
        topk_scores = analysis["separation_topk"]["scores"]
        
        correlations = {}
        
        # Cohesion correlations
        if len(jaccard_scores) > 1 and len(entropy_scores) > 1:
            if len(jaccard_scores) == len(entropy_scores):
                corr_coeff, p_value = pearsonr(jaccard_scores, entropy_scores)
                correlations["cohesion_jaccard_vs_entropy"] = {
                    "correlation": corr_coeff,
                    "p_value": p_value,
                    "significant": p_value < 0.05
                }
        
        # Separation correlations
        if len(js_scores) > 1 and len(topk_scores) > 1:
            if len(js_scores) == len(topk_scores):
                corr_coeff, p_value = pearsonr(js_scores, topk_scores)
                correlations["separation_js_vs_topk"] = {
                    "correlation": corr_coeff,
                    "p_value": p_value,
                    "significant": p_value < 0.05
                }
        
        return correlations
    
    manual_correlations = compute_correlations(manual_analysis)
    gemini_correlations = compute_correlations(gemini_analysis)
    
    # 8. Identify high/low scoring examples
    print("\n8. Identifying high/low scoring examples...")
    
    def identify_extreme_examples(analysis, baseline_analysis):
        """Identify segments with extremely high or low component scores."""
        if "error" in analysis or "error" in baseline_analysis:
            return {"error": "Missing data for example identification"}
        
        examples = {}
        
        # Find segments with highest/lowest cohesion scores
        segment_details = analysis["segment_details"]
        baseline_jaccard = baseline_analysis["cohesion_jaccard"]["stats"]["percentiles"]
        
        if segment_details and baseline_jaccard:
            # High cohesion examples (>90th percentile)
            high_cohesion_threshold = baseline_jaccard.get(90, 0.5)
            high_cohesion_segments = [
                seg for seg in segment_details 
                if seg["cohesion_jaccard"] > high_cohesion_threshold
            ]
            
            # Low cohesion examples (<10th percentile)
            low_cohesion_threshold = baseline_jaccard.get(10, 0.1)
            low_cohesion_segments = [
                seg for seg in segment_details 
                if seg["cohesion_jaccard"] < low_cohesion_threshold
            ]
            
            examples["high_cohesion_segments"] = high_cohesion_segments[:3]  # Top 3
            examples["low_cohesion_segments"] = low_cohesion_segments[:3]   # Bottom 3
        
        return examples
    
    manual_examples = identify_extreme_examples(manual_analysis, manual_baseline_analysis)
    gemini_examples = identify_extreme_examples(gemini_analysis, gemini_baseline_analysis)
    
    analysis_time = time.time() - analysis_start_time
    print(f"\n   Component analysis completed in {analysis_time:.1f}s")
    
    # Return comprehensive results
    return {
        "success": True,
        "domain": domain_name,
        "analysis_time": analysis_time,
        "data_stats": {
            "total_papers": len(papers),
            "filtered_papers": len(filtered_papers),
            "valid_keywords": len(valid_keywords),
            "year_range": (min(p.pub_year for p in papers), max(p.pub_year for p in papers))
        },
        "anti_gaming_config": {
            "min_segment_size": anti_gaming_config.min_segment_size,
            "size_weight_power": anti_gaming_config.size_weight_power,
            "enable_size_weighting": anti_gaming_config.enable_size_weighting,
            "enable_segment_floor": anti_gaming_config.enable_segment_floor
        },
        "reference_timelines": {
            "manual": manual_analysis,
            "gemini": gemini_analysis
        },
        "baseline_distributions": {
            "manual": manual_baseline_analysis,
            "gemini": gemini_baseline_analysis
        },
        "percentile_rankings": {
            "manual": manual_percentiles,
            "gemini": gemini_percentiles
        },
        "correlations": {
            "manual": manual_correlations,
            "gemini": gemini_correlations
        },
        "extreme_examples": {
            "manual": manual_examples,
            "gemini": gemini_examples
        }
    }


def generate_component_analysis_report(results: Dict[str, Any]) -> str:
    """Generate a comprehensive text report from component analysis results."""
    
    if not results.get("success", False):
        return f"Component analysis failed: {results.get('error', 'Unknown error')}"
    
    domain = results["domain"]
    report_lines = []
    
    # Header
    report_lines.append(f"COMPONENT ANALYSIS REPORT: {domain.upper()}")
    report_lines.append("=" * 60)
    
    # Data summary
    data_stats = results["data_stats"]
    report_lines.append(f"\nDATA SUMMARY:")
    report_lines.append(f"  Papers: {data_stats['total_papers']} → {data_stats['filtered_papers']} (filtered)")
    report_lines.append(f"  Keywords: {data_stats['valid_keywords']} valid")
    report_lines.append(f"  Year range: {data_stats['year_range'][0]}-{data_stats['year_range'][1]}")
    
    # Reference timeline analysis
    manual_analysis = results["reference_timelines"]["manual"]
    gemini_analysis = results["reference_timelines"]["gemini"]
    
    report_lines.append(f"\nREFERENCE TIMELINE ANALYSIS:")
    
    for timeline_name, analysis in [("Manual", manual_analysis), ("Gemini", gemini_analysis)]:
        if "error" in analysis:
            report_lines.append(f"  {timeline_name}: {analysis['error']}")
            continue
            
        report_lines.append(f"  {timeline_name} Timeline:")
        report_lines.append(f"    Segments: {analysis['num_segments']}")
        report_lines.append(f"    Transitions: {analysis['num_transitions']}")
        
        # Cohesion metrics
        jaccard_stats = analysis["cohesion_jaccard"]["stats"]
        entropy_stats = analysis["cohesion_entropy"]["stats"]
        report_lines.append(f"    Cohesion Jaccard: μ={jaccard_stats['mean']:.3f}, σ={jaccard_stats['std']:.3f}")
        report_lines.append(f"    Cohesion Entropy: μ={entropy_stats['mean']:.3f}, σ={entropy_stats['std']:.3f}")
        
        # Separation metrics
        if analysis["num_transitions"] > 0:
            js_stats = analysis["separation_js"]["stats"]
            topk_stats = analysis["separation_topk"]["stats"]
            report_lines.append(f"    Separation JS: μ={js_stats['mean']:.3f}, σ={js_stats['std']:.3f}")
            report_lines.append(f"    Separation TopK: μ={topk_stats['mean']:.3f}, σ={topk_stats['std']:.3f}")
    
    # Percentile rankings
    manual_percentiles = results["percentile_rankings"]["manual"]
    gemini_percentiles = results["percentile_rankings"]["gemini"]
    
    report_lines.append(f"\nPERCENTILE RANKINGS VS K-STRATIFIED BASELINES:")
    
    for timeline_name, percentiles in [("Manual", manual_percentiles), ("Gemini", gemini_percentiles)]:
        if "error" in percentiles:
            report_lines.append(f"  {timeline_name}: {percentiles['error']}")
            continue
            
        report_lines.append(f"  {timeline_name} Timeline:")
        
        for metric_name, metric_data in percentiles.items():
            mean_pct = metric_data["mean_percentile"]
            report_lines.append(f"    {metric_name}: {mean_pct:.1f}th percentile")
    
    # Correlation analysis
    manual_corr = results["correlations"]["manual"]
    gemini_corr = results["correlations"]["gemini"]
    
    report_lines.append(f"\nCORRELATION ANALYSIS:")
    
    for timeline_name, corr_data in [("Manual", manual_corr), ("Gemini", gemini_corr)]:
        if "error" in corr_data:
            report_lines.append(f"  {timeline_name}: {corr_data['error']}")
            continue
            
        report_lines.append(f"  {timeline_name} Timeline:")
        
        for corr_name, corr_info in corr_data.items():
            corr_val = corr_info["correlation"]
            significant = corr_info["significant"]
            sig_marker = "**" if significant else ""
            report_lines.append(f"    {corr_name}: r={corr_val:.3f}{sig_marker}")
    
    # High/low examples
    manual_examples = results["extreme_examples"]["manual"]
    gemini_examples = results["extreme_examples"]["gemini"]
    
    report_lines.append(f"\nEXTREME EXAMPLES:")
    
    for timeline_name, examples in [("Manual", manual_examples), ("Gemini", gemini_examples)]:
        if "error" in examples:
            report_lines.append(f"  {timeline_name}: {examples['error']}")
            continue
            
        report_lines.append(f"  {timeline_name} Timeline:")
        
        # High cohesion examples
        high_cohesion = examples.get("high_cohesion_segments", [])
        if high_cohesion:
            report_lines.append(f"    High Cohesion Segments:")
            for seg in high_cohesion:
                report_lines.append(f"      Segment {seg['segment_id']}: {seg['year_range'][0]}-{seg['year_range'][1]}, "
                                  f"size={seg['size']}, jaccard={seg['cohesion_jaccard']:.3f}")
                report_lines.append(f"        Top keywords: {', '.join(seg['top_keywords'][:5])}")
        
        # Low cohesion examples
        low_cohesion = examples.get("low_cohesion_segments", [])
        if low_cohesion:
            report_lines.append(f"    Low Cohesion Segments:")
            for seg in low_cohesion:
                report_lines.append(f"      Segment {seg['segment_id']}: {seg['year_range'][0]}-{seg['year_range'][1]}, "
                                  f"size={seg['size']}, jaccard={seg['cohesion_jaccard']:.3f}")
                report_lines.append(f"        Top keywords: {', '.join(seg['top_keywords'][:5])}")
    
    return "\n".join(report_lines)


def save_component_analysis_results(results: Dict[str, Any], output_dir: str = "results/component_analysis"):
    """Save component analysis results to JSON and text files."""
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert numpy types to native Python types for JSON serialization
    def convert_to_json_serializable(obj):
        """Recursively convert numpy types to JSON serializable types."""
        if isinstance(obj, dict):
            return {k: convert_to_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_json_serializable(item) for item in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj
    
    # Convert results to JSON serializable format
    json_results = convert_to_json_serializable(results)
    
    # Create filename with timestamp
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    domain = results.get("domain", "unknown")
    
    # Save JSON results
    json_filename = f"component_analysis_{domain}_{timestamp}.json"
    json_filepath = os.path.join(output_dir, json_filename)
    
    with open(json_filepath, 'w') as f:
        json.dump(json_results, f, indent=2)
    
    print(f"JSON results saved to: {json_filepath}")
    
    # Save text report
    report_text = generate_component_analysis_report(results)
    
    report_filename = f"component_analysis_{domain}_{timestamp}.txt"
    report_filepath = os.path.join(output_dir, report_filename)
    
    with open(report_filepath, 'w') as f:
        f.write(report_text)
    
    print(f"Text report saved to: {report_filepath}")
    
    # Save latest versions
    latest_json_path = os.path.join(output_dir, f"latest_{domain}_analysis.json")
    latest_report_path = os.path.join(output_dir, f"latest_{domain}_report.txt")
    
    with open(latest_json_path, 'w') as f:
        json.dump(json_results, f, indent=2)
    
    with open(latest_report_path, 'w') as f:
        f.write(report_text)
    
    print(f"Latest versions saved to: {latest_json_path}, {latest_report_path}")


def main():
    """Run component analysis on available domains."""
    print("COMPREHENSIVE COMPONENT ANALYSIS")
    print("=" * 60)
    
    # Available domains for testing
    test_domains = ["natural_language_processing", "computer_vision", "applied_mathematics", "art"]
    
    all_results = {}
    successful_analyses = 0
    
    for domain in test_domains:
        try:
            result = analyze_individual_components(domain, num_baselines=200)
            all_results[domain] = result
            
            if result.get("success", False):
                successful_analyses += 1
                
                # Save individual domain results
                save_component_analysis_results(result)
                
                # Print summary report
                print(f"\nSUMMARY REPORT FOR {domain.upper()}:")
                print("-" * 40)
                print(generate_component_analysis_report(result))
                
        except Exception as e:
            print(f"FAILED to run component analysis for {domain}: {e}")
            all_results[domain] = {"success": False, "error": str(e)}
    
    # Save consolidated results
    consolidated_results = {
        "analysis_timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
        "total_domains": len(test_domains),
        "successful_analyses": successful_analyses,
        "domain_results": all_results
    }
    
    consolidated_filepath = "results/component_analysis/consolidated_component_analysis.json"
    os.makedirs(os.path.dirname(consolidated_filepath), exist_ok=True)
    
    # Convert to JSON serializable
    def convert_to_json_serializable(obj):
        if isinstance(obj, dict):
            return {k: convert_to_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_json_serializable(item) for item in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj
    
    json_consolidated = convert_to_json_serializable(consolidated_results)
    
    with open(consolidated_filepath, 'w') as f:
        json.dump(json_consolidated, f, indent=2)
    
    print(f"\nConsolidated results saved to: {consolidated_filepath}")
    
    # Final summary
    print(f"\n{'='*60}")
    print(f"COMPONENT ANALYSIS SUMMARY")
    print(f"{'='*60}")
    print(f"Successful analyses: {successful_analyses}/{len(test_domains)}")
    
    if successful_analyses == len(test_domains):
        print(f"✓ ALL COMPONENT ANALYSES COMPLETED SUCCESSFULLY")
    else:
        print(f"⚠ {len(test_domains) - successful_analyses} ANALYSES FAILED")


if __name__ == "__main__":
    main() 