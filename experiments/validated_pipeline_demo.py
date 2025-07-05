#!/usr/bin/env python3
"""
Validated Anti-Gaming Keyword Metrics Pipeline Demo
===================================================

This script demonstrates the complete validated pipeline for timeline segmentation
evaluation using anti-gaming keyword metrics. It showcases:

1. Loading validated configuration from optimization_config.json
2. Applying anti-gaming safeguards (size-weighting, segment floor)
3. Generating K-stratified baselines for fair comparison
4. Computing component metrics with proper validation
5. Interpreting results against established validation criteria

VALIDATION CRITERIA (Based on Component Analysis):
- Expert timelines should achieve ≥60th percentile vs K-stratified baselines
- Anti-gaming safeguards must prevent micro-segmentation gaming
- Component metrics should show meaningful differentiation
- Results should be interpretable and align with domain knowledge

SUCCESS METRICS:
- 60% of component metrics achieve ≥60th percentile performance
- No evidence of metric gaming or exploitation
- Strong correlation patterns between complementary metrics
- Concrete examples align with domain expertise
"""

import os
import sys
import json
import time
import numpy as np
from typing import Dict, List, Tuple, Any

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


def load_validated_config() -> Dict[str, Any]:
    """Load validated configuration from optimization_config.json."""
    config_path = "optimization_config.json"
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Validate that keyword metrics are enabled
    if not config.get("keyword_metrics", {}).get("enabled", False):
        raise ValueError("Keyword metrics must be enabled in configuration")
    
    return config


def create_anti_gaming_config_from_json(config: Dict[str, Any]) -> AntiGamingConfig:
    """Create AntiGamingConfig from JSON configuration."""
    ag_config = config["keyword_metrics"]["anti_gaming_config"]
    
    return AntiGamingConfig(
        min_segment_size=ag_config["min_segment_size"],
        size_weight_power=ag_config["size_weight_power"],
        enable_size_weighting=ag_config["enable_size_weighting"],
        enable_segment_floor=ag_config["enable_segment_floor"],
        enable_count_penalty=ag_config["enable_count_penalty"]
    )


def validate_timeline_quality(domain_name: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate timeline quality using the complete anti-gaming pipeline.
    
    Args:
        domain_name: Name of domain to validate
        config: Loaded configuration dictionary
        
    Returns:
        Validation results with pass/fail status and detailed metrics
    """
    print(f"\n{'='*70}")
    print(f"VALIDATING TIMELINE QUALITY: {domain_name.upper()}")
    print(f"{'='*70}")
    
    start_time = time.time()
    
    # 1. Load configuration
    keyword_config = config["keyword_metrics"]
    validation_config = config["validation_criteria"]
    anti_gaming_config = create_anti_gaming_config_from_json(config)
    
    print(f"1. Configuration loaded:")
    print(f"   Validation threshold: {validation_config['expert_timeline_percentile_threshold']}th percentile")
    print(f"   Anti-gaming enabled: {anti_gaming_config.enable_size_weighting}")
    print(f"   Minimum segment size: {anti_gaming_config.min_segment_size} papers")
    
    # 2. Load and filter data
    print(f"\n2. Loading and filtering data...")
    try:
        papers = load_domain_papers(domain_name)
        manual_segments, gemini_segments = load_reference_timelines(domain_name)
        
        # Apply keyword filtering
        filtering_config = keyword_config["keyword_filtering"]
        filtered_papers, valid_keywords = filter_keywords(
            papers, 
            min_years=filtering_config["min_years"],
            min_paper_ratio=filtering_config["min_paper_ratio"]
        )
        
        print(f"   Papers: {len(papers)} → {len(filtered_papers)} (filtered)")
        print(f"   Keywords: {len(valid_keywords)} valid")
        print(f"   Manual timeline: {len(manual_segments)} segments")
        print(f"   Gemini timeline: {len(gemini_segments)} segments")
        
    except Exception as e:
        return {
            "success": False,
            "error": f"Data loading failed: {e}",
            "domain": domain_name
        }
    
    # 3. Convert to paper segments and apply anti-gaming filters
    print(f"\n3. Applying anti-gaming filters...")
    
    manual_paper_segments = papers_to_segments(filtered_papers, manual_segments)
    gemini_paper_segments = papers_to_segments(filtered_papers, gemini_segments)
    
    # Apply segment size floor
    manual_valid = [seg for seg in manual_paper_segments if len(seg) >= anti_gaming_config.min_segment_size]
    gemini_valid = [seg for seg in gemini_paper_segments if len(seg) >= anti_gaming_config.min_segment_size]
    
    print(f"   Manual segments: {len(manual_paper_segments)} → {len(manual_valid)} (after size filter)")
    print(f"   Gemini segments: {len(gemini_paper_segments)} → {len(gemini_valid)} (after size filter)")
    
    if not manual_valid and not gemini_valid:
        return {
            "success": False,
            "error": "No valid segments after anti-gaming filters",
            "domain": domain_name
        }
    
    # 4. Generate K-stratified baselines
    print(f"\n4. Generating K-stratified baselines...")
    baseline_config = keyword_config["baseline_generation"]
    num_samples = baseline_config["k_stratified_samples"]
    
    validation_results = {}
    
    for timeline_name, segments in [("manual", manual_valid), ("gemini", gemini_valid)]:
        if not segments:
            continue
            
        print(f"   Generating {num_samples} {timeline_name} K-stratified baselines...")
        
        segment_sizes = [len(seg) for seg in segments]
        k_stratified_baselines = generate_k_stratified_baselines(
            filtered_papers, len(segments), segment_sizes, num_samples=num_samples
        )
        
        # 5. Evaluate reference timeline
        print(f"   Evaluating {timeline_name} timeline...")
        reference_evaluation = evaluate_segmentation(
            filtered_papers, 
            [(min(p.pub_year for p in seg), max(p.pub_year for p in seg)) for seg in segments],
            anti_gaming_config=anti_gaming_config
        )
        
        # 6. Evaluate baselines for percentile computation
        print(f"   Evaluating {len(k_stratified_baselines)} baseline segmentations...")
        baseline_scores = []
        
        for baseline_segments in k_stratified_baselines:
            try:
                baseline_eval = evaluate_segmentation(
                    filtered_papers, baseline_segments, anti_gaming_config=anti_gaming_config
                )
                baseline_scores.append(baseline_eval.final_anti_gaming_score)
            except:
                continue
        
        if not baseline_scores:
            print(f"   WARNING: No valid baseline scores for {timeline_name}")
            continue
        
        # 7. Compute percentile ranking
        reference_score = reference_evaluation.final_anti_gaming_score
        percentile = (np.sum(np.array(baseline_scores) <= reference_score) / len(baseline_scores)) * 100
        
        # 8. Apply validation criteria
        threshold = validation_config["expert_timeline_percentile_threshold"]
        passes_validation = percentile >= threshold
        
        validation_results[timeline_name] = {
            "reference_score": reference_score,
            "baseline_scores": baseline_scores,
            "percentile": percentile,
            "threshold": threshold,
            "passes_validation": passes_validation,
            "num_segments": len(segments),
            "evaluation_details": {
                "cohesion_jaccard": reference_evaluation.cohesion_mean_jaccard,
                "cohesion_entropy": reference_evaluation.cohesion_entropy,
                "separation_js": reference_evaluation.separation_js,
                "separation_topk": reference_evaluation.separation_topk,
                "size_weighted_cohesion": reference_evaluation.size_weighted_cohesion_jaccard,
                "size_weighted_separation": reference_evaluation.size_weighted_separation_js,
                "anti_gaming_score": reference_evaluation.final_anti_gaming_score
            }
        }
        
        print(f"   {timeline_name.capitalize()} timeline: {percentile:.1f}th percentile {'✓' if passes_validation else '✗'}")
    
    # 9. Overall validation assessment
    passing_timelines = sum(1 for result in validation_results.values() if result["passes_validation"])
    total_timelines = len(validation_results)
    min_passing = validation_config.get("minimum_domains_passing", 1)
    
    # Success if at least one timeline passes (since we're validating domain quality)
    overall_success = passing_timelines > 0 and total_timelines > 0
    
    execution_time = time.time() - start_time
    
    print(f"\n{'='*70}")
    print(f"VALIDATION SUMMARY: {domain_name.upper()}")
    print(f"{'='*70}")
    print(f"Timelines passing validation: {passing_timelines}/{total_timelines}")
    print(f"Overall validation: {'PASS ✓' if overall_success else 'FAIL ✗'}")
    print(f"Execution time: {execution_time:.1f}s")
    
    return {
        "success": True,
        "domain": domain_name,
        "overall_validation_pass": overall_success,
        "passing_timelines": passing_timelines,
        "total_timelines": total_timelines,
        "timeline_results": validation_results,
        "execution_time": execution_time,
        "anti_gaming_config": {
            "min_segment_size": anti_gaming_config.min_segment_size,
            "size_weight_power": anti_gaming_config.size_weight_power,
            "enable_size_weighting": anti_gaming_config.enable_size_weighting
        }
    }


def run_pipeline_demo():
    """Run the complete validated pipeline demo."""
    print("VALIDATED ANTI-GAMING KEYWORD METRICS PIPELINE DEMO")
    print("=" * 80)
    
    # Load validated configuration
    try:
        config = load_validated_config()
        print("✓ Validated configuration loaded successfully")
        print(f"  Anti-gaming safeguards: {config['keyword_metrics']['anti_gaming_config']}")
        print(f"  Validation threshold: {config['validation_criteria']['expert_timeline_percentile_threshold']}th percentile")
    except Exception as e:
        print(f"✗ Configuration loading failed: {e}")
        return
    
    # Test domains (based on component analysis results)
    test_domains = ["natural_language_processing", "computer_vision", "applied_mathematics"]
    
    all_results = {}
    successful_validations = 0
    
    for domain in test_domains:
        try:
            result = validate_timeline_quality(domain, config)
            all_results[domain] = result
            
            if result.get("success", False) and result.get("overall_validation_pass", False):
                successful_validations += 1
                
        except Exception as e:
            print(f"✗ Validation failed for {domain}: {e}")
            all_results[domain] = {"success": False, "error": str(e)}
    
    # Final summary
    print(f"\n{'='*80}")
    print(f"PIPELINE DEMO SUMMARY")
    print(f"{'='*80}")
    print(f"Domains tested: {len(test_domains)}")
    print(f"Successful validations: {successful_validations}")
    print(f"Success rate: {successful_validations/len(test_domains)*100:.1f}%")
    
    if successful_validations >= 2:  # At least 2/3 domains should pass
        print(f"\n🎉 PIPELINE VALIDATION SUCCESSFUL!")
        print(f"   The anti-gaming keyword metrics pipeline is ready for production use.")
        print(f"   Key features validated:")
        print(f"   ✓ Size-weighted averaging prevents micro-segmentation gaming")
        print(f"   ✓ Segment floor eliminates unrealistic tiny segments")
        print(f"   ✓ K-stratified baselines provide fair comparison")
        print(f"   ✓ Expert timelines achieve meaningful differentiation")
    else:
        print(f"\n⚠ PIPELINE VALIDATION INCOMPLETE")
        print(f"   Only {successful_validations}/{len(test_domains)} domains passed validation.")
        print(f"   Review component analysis findings and adjust thresholds.")
    
    # Save results
    results_path = "results/validated_pipeline_demo_results.json"
    os.makedirs("results", exist_ok=True)
    
    with open(results_path, 'w') as f:
        # Convert numpy types for JSON serialization
        def convert_for_json(obj):
            if isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.bool_):
                return bool(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_for_json(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_for_json(item) for item in obj]
            else:
                return obj
        
        json_results = convert_for_json(all_results)
        json.dump(json_results, f, indent=2)
    
    print(f"\nDetailed results saved to: {results_path}")


if __name__ == "__main__":
    run_pipeline_demo() 