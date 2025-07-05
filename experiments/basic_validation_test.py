#!/usr/bin/env python3
"""
FIXED Smoke Test for Component Analysis
=======================================

Test keyword filtering and metric behavior with SIZE-STRATIFIED baselines
to address the fundamental segment size bias issue.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core_metrics import (
    load_domain_papers,
    load_reference_timelines,
    filter_keywords,
    evaluate_segmentation,
    generate_random_segmentations,
    generate_size_stratified_baselines,
    papers_to_segments
)
import numpy as np

def test_size_stratified_baselines(domain_name: str):
    """Test the fixed approach using size-stratified baselines."""
    print(f"\n{'='*60}")
    print(f"SIZE-STRATIFIED BASELINE TEST: {domain_name.upper()}")
    print(f"{'='*60}")
    
    # Load data with relaxed filtering
    papers = load_domain_papers(domain_name)
    manual_segments, gemini_segments = load_reference_timelines(domain_name)
    filtered_papers, valid_keywords = filter_keywords(papers, min_years=2, min_paper_ratio=0.01)
    
    print(f"Loaded {len(filtered_papers)} papers with {len(valid_keywords)} keywords")
    
    # Convert expert timelines to paper segments to get target sizes
    manual_paper_segments = papers_to_segments(filtered_papers, manual_segments)
    gemini_paper_segments = papers_to_segments(filtered_papers, gemini_segments)
    
    # Get expert segment sizes as targets
    manual_sizes = [len(seg) for seg in manual_paper_segments if seg]
    gemini_sizes = [len(seg) for seg in gemini_paper_segments if seg]
    
    print(f"\n1. EXPERT SEGMENT SIZES:")
    print(f"   Manual: {manual_sizes} (avg: {np.mean(manual_sizes):.0f})")
    print(f"   Gemini: {gemini_sizes} (avg: {np.mean(gemini_sizes):.0f})")
    
    # Evaluate expert timelines
    print(f"\n2. EXPERT TIMELINE SCORES:")
    manual_eval = evaluate_segmentation(filtered_papers, manual_segments)
    gemini_eval = evaluate_segmentation(filtered_papers, gemini_segments)
    
    print(f"   Manual: Jaccard={manual_eval.cohesion_mean_jaccard:.3f}, JS={manual_eval.separation_js:.3f}")
    print(f"   Gemini: Jaccard={gemini_eval.cohesion_mean_jaccard:.3f}, JS={gemini_eval.separation_js:.3f}")
    
    # Generate three types of baselines
    print(f"\n3. BASELINE COMPARISON:")
    
    # A) Original random baselines (biased toward small segments)
    print("   A) Original random baselines (biased):")
    random_segmentations = generate_random_segmentations(filtered_papers, num_samples=50)
    random_evaluations = []
    random_sizes = []
    
    for segments in random_segmentations:
        eval_result = evaluate_segmentation(filtered_papers, segments)
        random_evaluations.append(eval_result)
        
        # Track segment sizes
        paper_segments = papers_to_segments(filtered_papers, segments)
        sizes = [len(seg) for seg in paper_segments if seg]
        random_sizes.extend(sizes)
    
    random_jaccard = [e.cohesion_mean_jaccard for e in random_evaluations]
    random_js = [e.separation_js for e in random_evaluations]
    
    print(f"      Avg segment size: {np.mean(random_sizes):.0f} ± {np.std(random_sizes):.0f}")
    print(f"      Jaccard: {np.mean(random_jaccard):.3f} ± {np.std(random_jaccard):.3f}")
    print(f"      JS: {np.mean(random_js):.3f} ± {np.std(random_js):.3f}")
    
    # B) Size-stratified baselines matching manual timeline
    print("   B) Size-stratified baselines (manual sizes):")
    manual_stratified = generate_size_stratified_baselines(filtered_papers, manual_sizes, num_samples=50)
    manual_stratified_evaluations = []
    manual_stratified_sizes = []
    
    for segments in manual_stratified:
        eval_result = evaluate_segmentation(filtered_papers, segments)
        manual_stratified_evaluations.append(eval_result)
        
        # Track segment sizes
        paper_segments = papers_to_segments(filtered_papers, segments)
        sizes = [len(seg) for seg in paper_segments if seg]
        manual_stratified_sizes.extend(sizes)
    
    manual_strat_jaccard = [e.cohesion_mean_jaccard for e in manual_stratified_evaluations]
    manual_strat_js = [e.separation_js for e in manual_stratified_evaluations]
    
    print(f"      Avg segment size: {np.mean(manual_stratified_sizes):.0f} ± {np.std(manual_stratified_sizes):.0f}")
    print(f"      Jaccard: {np.mean(manual_strat_jaccard):.3f} ± {np.std(manual_strat_jaccard):.3f}")
    print(f"      JS: {np.mean(manual_strat_js):.3f} ± {np.std(manual_strat_js):.3f}")
    
    # C) Size-stratified baselines matching Gemini timeline
    print("   C) Size-stratified baselines (Gemini sizes):")
    gemini_stratified = generate_size_stratified_baselines(filtered_papers, gemini_sizes, num_samples=50)
    gemini_stratified_evaluations = []
    gemini_stratified_sizes = []
    
    for segments in gemini_stratified:
        eval_result = evaluate_segmentation(filtered_papers, segments)
        gemini_stratified_evaluations.append(eval_result)
        
        # Track segment sizes
        paper_segments = papers_to_segments(filtered_papers, segments)
        sizes = [len(seg) for seg in paper_segments if seg]
        gemini_stratified_sizes.extend(sizes)
    
    gemini_strat_jaccard = [e.cohesion_mean_jaccard for e in gemini_stratified_evaluations]
    gemini_strat_js = [e.separation_js for e in gemini_stratified_evaluations]
    
    print(f"      Avg segment size: {np.mean(gemini_stratified_sizes):.0f} ± {np.std(gemini_stratified_sizes):.0f}")
    print(f"      Jaccard: {np.mean(gemini_strat_jaccard):.3f} ± {np.std(gemini_strat_jaccard):.3f}")
    print(f"      JS: {np.mean(gemini_strat_js):.3f} ± {np.std(gemini_strat_js):.3f}")
    
    # Compute percentiles against appropriate baselines
    print(f"\n4. PERCENTILE ANALYSIS (Fixed):")
    
    # Manual vs manual-sized baselines
    manual_jaccard_percentile = (np.sum(np.array(manual_strat_jaccard) <= manual_eval.cohesion_mean_jaccard) / len(manual_strat_jaccard)) * 100
    manual_js_percentile = (np.sum(np.array(manual_strat_js) <= manual_eval.separation_js) / len(manual_strat_js)) * 100
    
    print(f"   Manual vs manual-sized baselines:")
    print(f"     Jaccard: {manual_jaccard_percentile:.1f}th percentile")
    print(f"     JS: {manual_js_percentile:.1f}th percentile")
    
    # Gemini vs gemini-sized baselines
    gemini_jaccard_percentile = (np.sum(np.array(gemini_strat_jaccard) <= gemini_eval.cohesion_mean_jaccard) / len(gemini_strat_jaccard)) * 100
    gemini_js_percentile = (np.sum(np.array(gemini_strat_js) <= gemini_eval.separation_js) / len(gemini_strat_js)) * 100
    
    print(f"   Gemini vs gemini-sized baselines:")
    print(f"     Jaccard: {gemini_jaccard_percentile:.1f}th percentile")
    print(f"     JS: {gemini_js_percentile:.1f}th percentile")
    
    # Summary
    print(f"\n5. BIAS CORRECTION SUMMARY:")
    print(f"   BEFORE (biased): Manual scored {(np.sum(np.array(random_jaccard) <= manual_eval.cohesion_mean_jaccard) / len(random_jaccard)) * 100:.1f}th percentile")
    print(f"   AFTER (fixed):   Manual scored {manual_jaccard_percentile:.1f}th percentile")
    print(f"   IMPROVEMENT: {manual_jaccard_percentile - (np.sum(np.array(random_jaccard) <= manual_eval.cohesion_mean_jaccard) / len(random_jaccard)) * 100:.1f} percentile points")
    
    return {
        'manual_eval': manual_eval,
        'gemini_eval': gemini_eval,
        'manual_percentiles': (manual_jaccard_percentile, manual_js_percentile),
        'gemini_percentiles': (gemini_jaccard_percentile, gemini_js_percentile),
        'bias_corrected': True
    }

def main():
    """Run the fixed smoke test."""
    domain_name = "natural_language_processing"
    
    print("FIXED COMPONENT ANALYSIS SMOKE TEST")
    print("=" * 40)
    print(f"Testing domain: {domain_name}")
    print("ADDRESSING: Segment size bias in metrics")
    
    try:
        results = test_size_stratified_baselines(domain_name)
        
        print(f"\n{'='*60}")
        print("FIXED SMOKE TEST RESULTS")
        print(f"{'='*60}")
        
        manual_jaccard_pct, manual_js_pct = results['manual_percentiles']
        gemini_jaccard_pct, gemini_js_pct = results['gemini_percentiles']
        
        print(f"✅ Size bias corrected using stratified baselines")
        print(f"✅ Manual timeline: {manual_jaccard_pct:.0f}th percentile (Jaccard), {manual_js_pct:.0f}th percentile (JS)")
        print(f"✅ Gemini timeline: {gemini_jaccard_pct:.0f}th percentile (Jaccard), {gemini_js_pct:.0f}th percentile (JS)")
        
        # Check if expert timelines now score reasonably
        if manual_jaccard_pct >= 25 and manual_js_pct >= 25:
            print(f"🎯 VALIDATION PASSED: Expert timelines score ≥25th percentile")
        else:
            print(f"⚠️  VALIDATION MIXED: Some metrics still below 25th percentile")
            
        print(f"✅ Ready for full component analysis with corrected baselines")
        
    except Exception as e:
        print(f"\n❌ FIXED SMOKE TEST FAILED: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 