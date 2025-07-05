#!/usr/bin/env python3
"""
Anti-Gaming Keyword Metrics Experiment Runner
=============================================

This script runs comprehensive experiments using anti-gaming safeguards:
1. Size-weighted averaging prevents micro-segment gaming
2. Minimum segment floor (50+ papers) excludes tiny segments  
3. K-stratified baselines provide fair comparison

VALIDATION CRITERIA:
- Expert timelines should score ≥50th percentile vs K-stratified baselines
- Anti-gaming metrics should favor realistic segmentations over micro-segmentations
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
from scipy.stats import pearsonr

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core_metrics import (
    load_domain_papers,
    load_reference_timelines,
    filter_keywords,
    evaluate_segmentation,
    generate_k_stratified_baselines,
    papers_to_segments,
    AntiGamingConfig,
    SegmentationEvaluation
)


def run_domain_anti_gaming_experiments(domain_name: str, num_k_stratified_samples: int = 500) -> Dict[str, Any]:
    """
    Run comprehensive anti-gaming experiments for a single domain.
    
    Args:
        domain_name: Name of domain to test
        num_k_stratified_samples: Number of K-stratified baselines to generate
        
    Returns:
        Dictionary with all experiment results
    """
    print(f"\n{'='*70}")
    print(f"ANTI-GAMING EXPERIMENTS: {domain_name.upper()}")
    print(f"{'='*70}")
    
    experiment_start_time = time.time()
    
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
            print(f"   WARNING: Only {len(filtered_papers)} papers - insufficient for robust testing")
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
        enable_count_penalty=False  # Disabled due to over-penalization
    )
    
    print(f"   Size-weighted averaging: {anti_gaming_config.enable_size_weighting}")
    print(f"   Minimum segment size: {anti_gaming_config.min_segment_size} papers")
    print(f"   Size weight power: {anti_gaming_config.size_weight_power}")
    
    # 3. Evaluate reference timelines
    print("\n3. Evaluating reference timelines...")
    manual_eval = evaluate_segmentation(filtered_papers, manual_segments, anti_gaming_config)
    gemini_eval = evaluate_segmentation(filtered_papers, gemini_segments, anti_gaming_config)
    
    print(f"   Manual: score={manual_eval.final_anti_gaming_score:.3f}, "
          f"cohesion={manual_eval.size_weighted_cohesion_jaccard:.3f}, "
          f"separation={manual_eval.size_weighted_separation_js:.3f}, "
          f"segments={manual_eval.num_segments}")
    
    print(f"   Gemini: score={gemini_eval.final_anti_gaming_score:.3f}, "
          f"cohesion={gemini_eval.size_weighted_cohesion_jaccard:.3f}, "
          f"separation={gemini_eval.size_weighted_separation_js:.3f}, "
          f"segments={gemini_eval.num_segments}")
    
    # 4. Generate K-stratified baselines for both reference timelines
    print(f"\n4. Generating K-stratified baselines ({num_k_stratified_samples} samples)...")
    
    # Get segment sizes from reference timelines
    manual_paper_segments = papers_to_segments(filtered_papers, manual_segments)
    gemini_paper_segments = papers_to_segments(filtered_papers, gemini_segments)
    
    manual_sizes = [len(seg) for seg in manual_paper_segments if len(seg) >= anti_gaming_config.min_segment_size]
    gemini_sizes = [len(seg) for seg in gemini_paper_segments if len(seg) >= anti_gaming_config.min_segment_size]
    
    print(f"   Manual segment sizes: {manual_sizes} (K={len(manual_sizes)})")
    print(f"   Gemini segment sizes: {gemini_sizes} (K={len(gemini_sizes)})")
    
    # Generate K-stratified baselines for manual timeline
    manual_k_stratified = []
    if manual_sizes:
        manual_k_stratified = generate_k_stratified_baselines(
            filtered_papers, len(manual_sizes), manual_sizes, num_samples=num_k_stratified_samples
        )
        print(f"   Generated {len(manual_k_stratified)} manual K-stratified baselines")
    
    # Generate K-stratified baselines for Gemini timeline  
    gemini_k_stratified = []
    if gemini_sizes:
        gemini_k_stratified = generate_k_stratified_baselines(
            filtered_papers, len(gemini_sizes), gemini_sizes, num_samples=num_k_stratified_samples
        )
        print(f"   Generated {len(gemini_k_stratified)} Gemini K-stratified baselines")
    
    # 5. Evaluate K-stratified baselines
    print("\n5. Evaluating K-stratified baselines...")
    
    manual_k_scores = []
    gemini_k_scores = []
    
    # Evaluate manual K-stratified baselines
    if manual_k_stratified:
        print(f"   Evaluating manual K-stratified baselines...")
        for i, k_seg in enumerate(manual_k_stratified):
            k_eval = evaluate_segmentation(filtered_papers, k_seg, anti_gaming_config)
            manual_k_scores.append(k_eval.final_anti_gaming_score)
            
            if i % 100 == 0:
                print(f"     Progress: {i+1}/{len(manual_k_stratified)}")
    
    # Evaluate Gemini K-stratified baselines
    if gemini_k_stratified:
        print(f"   Evaluating Gemini K-stratified baselines...")
        for i, k_seg in enumerate(gemini_k_stratified):
            k_eval = evaluate_segmentation(filtered_papers, k_seg, anti_gaming_config)
            gemini_k_scores.append(k_eval.final_anti_gaming_score)
            
            if i % 100 == 0:
                print(f"     Progress: {i+1}/{len(gemini_k_stratified)}")
    
    # 6. Compute percentile rankings
    print("\n6. Computing percentile rankings...")
    
    def compute_percentile_rank(value, distribution):
        """Compute percentile rank of value in distribution."""
        if not distribution:
            return 0.0
        return (np.sum(np.array(distribution) <= value) / len(distribution)) * 100
    
    manual_percentile = compute_percentile_rank(manual_eval.final_anti_gaming_score, manual_k_scores) if manual_k_scores else 0.0
    gemini_percentile = compute_percentile_rank(gemini_eval.final_anti_gaming_score, gemini_k_scores) if gemini_k_scores else 0.0
    
    print(f"   Manual timeline: {manual_percentile:.1f}th percentile vs K-stratified baselines")
    print(f"   Gemini timeline: {gemini_percentile:.1f}th percentile vs K-stratified baselines")
    
    # 7. Validation against criteria
    print("\n7. Validation against anti-gaming criteria...")
    
    manual_passes = manual_percentile >= 50.0
    gemini_passes = gemini_percentile >= 50.0
    
    print(f"   Manual ≥50th percentile: {manual_passes} ({'✓' if manual_passes else '✗'})")
    print(f"   Gemini ≥50th percentile: {gemini_passes} ({'✓' if gemini_passes else '✗'})")
    
    # 8. Statistical summary
    print("\n8. Statistical summary...")
    
    if manual_k_scores:
        manual_stats = {
            'mean': np.mean(manual_k_scores),
            'std': np.std(manual_k_scores),
            'median': np.median(manual_k_scores),
            'p25': np.percentile(manual_k_scores, 25),
            'p75': np.percentile(manual_k_scores, 75)
        }
        print(f"   Manual K-stratified: mean={manual_stats['mean']:.3f}, "
              f"median={manual_stats['median']:.3f}, std={manual_stats['std']:.3f}")
    
    if gemini_k_scores:
        gemini_stats = {
            'mean': np.mean(gemini_k_scores),
            'std': np.std(gemini_k_scores),
            'median': np.median(gemini_k_scores),
            'p25': np.percentile(gemini_k_scores, 25),
            'p75': np.percentile(gemini_k_scores, 75)
        }
        print(f"   Gemini K-stratified: mean={gemini_stats['mean']:.3f}, "
              f"median={gemini_stats['median']:.3f}, std={gemini_stats['std']:.3f}")
    
    experiment_time = time.time() - experiment_start_time
    print(f"\n   Experiment completed in {experiment_time:.1f}s")
    
    # Return comprehensive results
    return {
        "success": True,
        "domain": domain_name,
        "data_stats": {
            "total_papers": len(papers),
            "filtered_papers": len(filtered_papers),
            "valid_keywords": len(valid_keywords),
            "year_range": (min(p.pub_year for p in papers), max(p.pub_year for p in papers))
        },
        "reference_evaluations": {
            "manual": {
                "score": manual_eval.final_anti_gaming_score,
                "cohesion": manual_eval.size_weighted_cohesion_jaccard,
                "separation": manual_eval.size_weighted_separation_js,
                "num_segments": manual_eval.num_segments,
                "percentile": manual_percentile,
                "passes_criteria": manual_passes
            },
            "gemini": {
                "score": gemini_eval.final_anti_gaming_score,
                "cohesion": gemini_eval.size_weighted_cohesion_jaccard,
                "separation": gemini_eval.size_weighted_separation_js,
                "num_segments": gemini_eval.num_segments,
                "percentile": gemini_percentile,
                "passes_criteria": gemini_passes
            }
        },
        "k_stratified_baselines": {
            "manual": {
                "target_sizes": manual_sizes,
                "num_samples": len(manual_k_stratified),
                "scores": manual_k_scores,
                "stats": manual_stats if manual_k_scores else {}
            },
            "gemini": {
                "target_sizes": gemini_sizes,
                "num_samples": len(gemini_k_stratified),
                "scores": gemini_k_scores,
                "stats": gemini_stats if gemini_k_scores else {}
            }
        },
        "anti_gaming_config": {
            "min_segment_size": anti_gaming_config.min_segment_size,
            "size_weight_power": anti_gaming_config.size_weight_power,
            "enable_size_weighting": anti_gaming_config.enable_size_weighting,
            "enable_segment_floor": anti_gaming_config.enable_segment_floor,
            "enable_count_penalty": anti_gaming_config.enable_count_penalty
        },
        "experiment_time": experiment_time
    }


def save_anti_gaming_results(results: Dict[str, Any], output_dir: str = "results/anti_gaming"):
    """Save anti-gaming experiment results to JSON file."""
    
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
    filename = f"anti_gaming_experiments_{timestamp}.json"
    filepath = os.path.join(output_dir, filename)
    
    # Save results
    with open(filepath, 'w') as f:
        json.dump(json_results, f, indent=2)
    
    print(f"Results saved to: {filepath}")
    
    # Also save latest version
    latest_filepath = os.path.join(output_dir, "latest_anti_gaming_results.json")
    with open(latest_filepath, 'w') as f:
        json.dump(json_results, f, indent=2)
    
    print(f"Latest results saved to: {latest_filepath}")


def main():
    """Run anti-gaming experiments on available domains."""
    print("ANTI-GAMING KEYWORD METRICS EXPERIMENTS")
    print("=" * 60)
    
    # Available domains for testing
    test_domains = ["natural_language_processing", "computer_vision", "applied_mathematics", "art"]
    
    all_results = {}
    successful_experiments = 0
    
    for domain in test_domains:
        try:
            result = run_domain_anti_gaming_experiments(domain, num_k_stratified_samples=200)
            all_results[domain] = result
            
            if result.get("success", False):
                successful_experiments += 1
                
        except Exception as e:
            print(f"FAILED to run experiments for {domain}: {e}")
            all_results[domain] = {"success": False, "error": str(e)}
    
    # Save all results
    save_anti_gaming_results(all_results)
    
    # Summary
    print(f"\n{'='*60}")
    print(f"ANTI-GAMING EXPERIMENTS SUMMARY")
    print(f"{'='*60}")
    print(f"Successful experiments: {successful_experiments}/{len(test_domains)}")
    
    # Validation summary
    print(f"\nVALIDATION RESULTS:")
    print(f"Domain                      Manual ≥50%   Gemini ≥50%")
    print(f"-" * 55)
    
    for domain, result in all_results.items():
        if result.get("success", False):
            manual_pass = result["reference_evaluations"]["manual"]["passes_criteria"]
            gemini_pass = result["reference_evaluations"]["gemini"]["passes_criteria"]
            manual_pct = result["reference_evaluations"]["manual"]["percentile"]
            gemini_pct = result["reference_evaluations"]["gemini"]["percentile"]
            
            manual_status = f"✓ {manual_pct:.1f}%" if manual_pass else f"✗ {manual_pct:.1f}%"
            gemini_status = f"✓ {gemini_pct:.1f}%" if gemini_pass else f"✗ {gemini_pct:.1f}%"
            
            print(f"{domain:<25} {manual_status:<12} {gemini_status}")
        else:
            print(f"{domain:<25} ERROR        ERROR")
    
    if successful_experiments == len(test_domains):
        print(f"\n✓ ALL EXPERIMENTS COMPLETED SUCCESSFULLY")
    else:
        print(f"\n⚠ {len(test_domains) - successful_experiments} EXPERIMENTS FAILED")


if __name__ == "__main__":
    main() 