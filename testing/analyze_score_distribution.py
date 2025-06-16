"""
Analysis Script for Merging Score Distribution

This script analyzes the output of the merging analysis to help determine a
data-driven, calibrated threshold for merge decisions.

**Functionality:**
1.  Loads the JSON output from `test_merging_analysis.py`.
2.  Extracts the final weighted score for each analyzed segment pair.
3.  Calculates and prints key statistics for the score distribution:
    - Mean
    - Standard Deviation
    - Minimum and Maximum scores
4.  Identifies and highlights the scores for known "should-merge" and
    "should-not-merge" pairs to provide context for threshold selection.
5.  Suggests a calibrated threshold based on the analysis.
"""

import json
from pathlib import Path
import numpy as np

def analyze_scores():
    """Main function to run the score distribution analysis."""
    
    # --- Load the analysis results ---
    analysis_file = Path("results/merging_analysis/natural_language_processing_merging_analysis.json")
    if not analysis_file.exists():
        print(f"❌ Analysis file not found at: {analysis_file}")
        return

    with open(analysis_file, 'r') as f:
        analysis_data = json.load(f)

    analysis_pairs = analysis_data.get("analysis_pairs", [])
    if not analysis_pairs:
        print("❌ No analysis pairs found in the file.")
        return

    print("--- Merging Score Distribution Analysis ---")
    
    scores = [pair['final_weighted_score'] for pair in analysis_pairs]
    
    # --- Calculate and Print Statistics ---
    mean_score = np.mean(scores)
    std_dev = np.std(scores)
    min_score = np.min(scores)
    max_score = np.max(scores)
    
    print("\n[Score Distribution Statistics]")
    print(f"  - Mean: {mean_score:.3f}")
    print(f"  - Standard Deviation: {std_dev:.3f}")
    print(f"  - Min Score: {min_score:.3f}")
    print(f"  - Max Score: {max_score:.3f}")
    
    # --- Identify Scores for Key Control Pairs ---
    print("\n[Control Pair Analysis]")
    
    should_merge_pair = None
    should_not_merge_pair = None
    
    for pair in analysis_pairs:
        # Known "should-merge" pair
        pair_str = pair['pair']
        if "1994" in pair_str and "1997" in pair_str and "1998" in pair_str and "2003" in pair_str:
            should_merge_pair = pair
        # Known "should-not-merge" pair
        if "2011" in pair_str and "2014" in pair_str and "2015" in pair_str and "2023" in pair_str:
            should_not_merge_pair = pair
            
    if should_merge_pair:
        score = should_merge_pair['final_weighted_score']
        print(f"  - ✅ 'Should Merge' Pair Score (Statistical NLP): {score:.3f}")
    else:
        print("  - ⚠️ Could not find the 'Statistical NLP' pair.")

    if should_not_merge_pair:
        score = should_not_merge_pair['final_weighted_score']
        print(f"  - ❌ 'Should Not Merge' Pair Score (Word Reps -> Transformers): {score:.3f}")
    else:
        print("  - ⚠️ Could not find the 'Word Reps -> Transformers' pair.")
        
    # --- Propose a Calibrated Threshold ---
    print("\n[Threshold Recommendation]")
    
    if should_merge_pair:
        # Suggest a threshold slightly below the "should-merge" score
        # but hopefully above the mean of other scores.
        suggested_threshold = should_merge_pair['final_weighted_score'] - (std_dev / 2)
        
        print(f"  - Based on the analysis, a calibrated threshold of ~{suggested_threshold:.2f} is recommended.")
        print(f"  - This would correctly include the 'Statistical NLP' pair (score: {should_merge_pair['final_weighted_score']:.3f})")
        if should_not_merge_pair:
             if suggested_threshold > should_not_merge_pair['final_weighted_score']:
                 print(f"  - It would also correctly exclude the 'Transformers' pair (score: {should_not_merge_pair['final_weighted_score']:.3f})")
             else:
                 print(f"  - ⚠️ CAUTION: This threshold might incorrectly include the 'Transformers' pair.")
    else:
        print("  - Cannot recommend a threshold without a valid 'should-merge' control pair.")


if __name__ == "__main__":
    analyze_scores() 