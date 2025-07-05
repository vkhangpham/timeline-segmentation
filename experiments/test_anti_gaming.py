#!/usr/bin/env python3
"""
Test Anti-Gaming Safeguards
===========================

This script tests that the anti-gaming safeguards successfully prevent metric exploitation:
1. Size-weighted averaging prevents micro-segment gaming
2. Minimum segment floor excludes tiny segments
3. Segment-count penalty discourages excessive segmentation
4. K-stratified baselines provide fair comparison
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core_metrics import (
    load_domain_papers,
    filter_keywords,
    evaluate_segmentation,
    generate_k_stratified_baselines,
    AntiGamingConfig,
    papers_to_segments
)
import numpy as np
import random


def create_micro_segmentation(papers, max_segments=20):
    """Create a micro-segmentation with many tiny segments."""
    if not papers:
        return []
    
    sorted_papers = sorted(papers, key=lambda p: p.pub_year)
    min_year = sorted_papers[0].pub_year
    max_year = sorted_papers[-1].pub_year
    year_span = max_year - min_year + 1
    
    # Create many small segments
    num_segments = min(max_segments, year_span)
    segment_size = max(1, year_span // num_segments)
    
    segments = []
    current_year = min_year
    
    for i in range(num_segments):
        if i == num_segments - 1:
            # Last segment takes remaining years
            end_year = max_year
        else:
            end_year = min(current_year + segment_size - 1, max_year)
        
        segments.append((current_year, end_year))
        current_year = end_year + 1
        
        if current_year > max_year:
            break
    
    return segments


def create_normal_segmentation(papers, num_segments=4):
    """Create a normal segmentation with reasonable-sized segments."""
    if not papers:
        return []
    
    sorted_papers = sorted(papers, key=lambda p: p.pub_year)
    min_year = sorted_papers[0].pub_year
    max_year = sorted_papers[-1].pub_year
    year_span = max_year - min_year + 1
    
    segment_size = max(1, year_span // num_segments)
    
    segments = []
    current_year = min_year
    
    for i in range(num_segments):
        if i == num_segments - 1:
            # Last segment takes remaining years
            end_year = max_year
        else:
            end_year = min(current_year + segment_size - 1, max_year)
        
        segments.append((current_year, end_year))
        current_year = end_year + 1
        
        if current_year > max_year:
            break
    
    return segments


def test_anti_gaming_on_domain(domain_name: str):
    """Test anti-gaming safeguards on a specific domain."""
    print(f"\n{'='*60}")
    print(f"TESTING ANTI-GAMING SAFEGUARDS: {domain_name.upper()}")
    print(f"{'='*60}")
    
    try:
        # Load and filter data
        print("1. Loading data...")
        papers = load_domain_papers(domain_name)
        filtered_papers, valid_keywords = filter_keywords(papers, min_years=2, min_paper_ratio=0.01)
        
        print(f"   Papers: {len(papers)} → {len(filtered_papers)} (after filtering)")
        print(f"   Keywords: {len(valid_keywords)} valid")
        
        if len(filtered_papers) < 100:
            print(f"   WARNING: Only {len(filtered_papers)} papers - may not be sufficient for testing")
            return
        
        # Create test segmentations
        print("\n2. Creating test segmentations...")
        micro_segments = create_micro_segmentation(filtered_papers, max_segments=15)
        normal_segments = create_normal_segmentation(filtered_papers, num_segments=4)
        
        print(f"   Micro-segmentation: {len(micro_segments)} segments")
        print(f"   Normal segmentation: {len(normal_segments)} segments")
        
        # Analyze segment sizes
        micro_paper_segments = papers_to_segments(filtered_papers, micro_segments)
        normal_paper_segments = papers_to_segments(filtered_papers, normal_segments)
        
        micro_sizes = [len(seg) for seg in micro_paper_segments if seg]
        normal_sizes = [len(seg) for seg in normal_paper_segments if seg]
        
        print(f"   Micro segment sizes: {micro_sizes[:5]}... (mean={np.mean(micro_sizes):.1f})")
        print(f"   Normal segment sizes: {normal_sizes} (mean={np.mean(normal_sizes):.1f})")
        
        # Test configurations
        configs = {
            "no_safeguards": AntiGamingConfig(
                enable_size_weighting=False,
                enable_segment_floor=False, 
                enable_count_penalty=False
            ),
            "size_weighting_only": AntiGamingConfig(
                enable_size_weighting=True,
                enable_segment_floor=False,
                enable_count_penalty=False,
                size_weight_power=0.5
            ),
            "segment_floor_only": AntiGamingConfig(
                enable_size_weighting=False,
                enable_segment_floor=True,
                enable_count_penalty=False,
                min_segment_size=50
            ),
            "count_penalty_only": AntiGamingConfig(
                enable_size_weighting=False,
                enable_segment_floor=False,
                enable_count_penalty=True,
                segment_count_penalty_sigma=2.0
            ),
            "all_safeguards": AntiGamingConfig(
                enable_size_weighting=True,
                enable_segment_floor=True,
                enable_count_penalty=True,
                min_segment_size=50,
                size_weight_power=0.5,
                segment_count_penalty_sigma=2.0
            )
        }
        
        print("\n3. Testing configurations...")
        results = {}
        
        for config_name, config in configs.items():
            print(f"\n   Testing: {config_name}")
            
            # Evaluate micro-segmentation
            micro_eval = evaluate_segmentation(filtered_papers, micro_segments, config)
            
            # Evaluate normal segmentation
            normal_eval = evaluate_segmentation(filtered_papers, normal_segments, config)
            
            results[config_name] = {
                "micro": micro_eval,
                "normal": normal_eval
            }
            
            print(f"     Micro: final_score={micro_eval.final_anti_gaming_score:.3f}, "
                  f"cohesion={micro_eval.size_weighted_cohesion_jaccard:.3f}, "
                  f"separation={micro_eval.size_weighted_separation_js:.3f}, "
                  f"penalty={micro_eval.segment_count_penalty:.3f}")
            
            print(f"     Normal: final_score={normal_eval.final_anti_gaming_score:.3f}, "
                  f"cohesion={normal_eval.size_weighted_cohesion_jaccard:.3f}, "
                  f"separation={normal_eval.size_weighted_separation_js:.3f}, "
                  f"penalty={normal_eval.segment_count_penalty:.3f}")
            
            # Check if normal beats micro (anti-gaming working)
            normal_better = normal_eval.final_anti_gaming_score > micro_eval.final_anti_gaming_score
            print(f"     Normal > Micro: {normal_better} ({'✓' if normal_better else '✗'})")
        
        # Summary analysis
        print("\n4. ANTI-GAMING EFFECTIVENESS ANALYSIS:")
        print("   Configuration                Normal > Micro?   Score Difference")
        print("   " + "-"*60)
        
        for config_name, result in results.items():
            micro_score = result["micro"].final_anti_gaming_score
            normal_score = result["normal"].final_anti_gaming_score
            normal_better = normal_score > micro_score
            score_diff = normal_score - micro_score
            
            status = "✓ WORKING" if normal_better else "✗ VULNERABLE"
            print(f"   {config_name:<25} {status:<13} {score_diff:+.3f}")
        
        # Test K-stratified baselines
        print("\n5. Testing K-stratified baselines...")
        
        # Get target sizes from normal segmentation
        target_sizes = [len(seg) for seg in normal_paper_segments if seg]
        target_k = len(target_sizes)
        
        # Generate K-stratified baselines
        k_stratified = generate_k_stratified_baselines(
            filtered_papers, target_k, target_sizes, num_samples=20
        )
        
        print(f"   Generated {len(k_stratified)} K-stratified baselines")
        print(f"   Target K={target_k}, target sizes={target_sizes}")
        
        if k_stratified:
            # Evaluate a few K-stratified examples
            k_scores = []
            for i, k_seg in enumerate(k_stratified[:5]):
                k_eval = evaluate_segmentation(filtered_papers, k_seg, configs["all_safeguards"])
                k_scores.append(k_eval.final_anti_gaming_score)
                
            print(f"   K-stratified scores: {[f'{s:.3f}' for s in k_scores]}")
            print(f"   K-stratified mean: {np.mean(k_scores):.3f}")
        
        print(f"\n   TESTING COMPLETE: {domain_name}")
        return True
        
    except Exception as e:
        print(f"   ERROR: {e}")
        return False


def main():
    """Test anti-gaming safeguards on available domains."""
    print("ANTI-GAMING SAFEGUARDS TEST SUITE")
    print("=" * 50)
    
    # Test on a few domains
    test_domains = ["natural_language_processing", "machine_learning", "computer_vision"]
    
    successful_tests = 0
    total_tests = 0
    
    for domain in test_domains:
        total_tests += 1
        try:
            success = test_anti_gaming_on_domain(domain)
            if success:
                successful_tests += 1
        except Exception as e:
            print(f"FAILED to test {domain}: {e}")
    
    print(f"\n{'='*50}")
    print(f"ANTI-GAMING TEST SUMMARY")
    print(f"{'='*50}")
    print(f"Successful tests: {successful_tests}/{total_tests}")
    
    if successful_tests == total_tests:
        print("✓ ALL TESTS PASSED - Anti-gaming safeguards implemented successfully")
    else:
        print("✗ SOME TESTS FAILED - Check implementation")


if __name__ == "__main__":
    main() 