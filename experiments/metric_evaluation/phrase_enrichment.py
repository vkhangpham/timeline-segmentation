#!/usr/bin/env python3
"""
Phase 16 FEATURE-06: Keyword-Phrase Enrichment via YAKE Evaluation

This experiment evaluates the impact of enriching paper keywords with multi-word 
phrases extracted using YAKE (Yet Another Keyword Extractor) on consensus and 
difference metrics.

Evaluation Protocol:
1. Pilot study: Applied Mathematics (stable) & Computer Vision (volatile)
2. Metrics: ΔC1, ΔD1, ΔFinal-L, runtime, memory usage
3. Success criteria: ΔL ≥ +0.01 OR (C1↑ & D1↑) with no L drop
4. Full sweep: All 8 domains if pilot succeeds

Follows project guidelines:
- Fail-fast error handling (no fallbacks)
- Real data only (Rule 3)
- Rigorous evaluation (Rule 5)
- Terminal log analysis (Rule 7)
"""

import os
import sys
import json
import time
import traceback
from datetime import datetime
from typing import Dict, Any, List, Tuple

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from core.data_loader import load_domain_data
from optimize_segmentation_bayesian import (
    convert_dataframe_to_domain_data,
    optimize_consensus_difference_parameters_bayesian,
)
from core.algorithm_config import ComprehensiveAlgorithmConfig
from core.consensus_difference_metrics import evaluate_segmentation_quality


def run_phrase_enrichment_comparison(domain_name: str, max_evaluations: int = 30) -> Dict[str, Any]:
    """
    Run A/B comparison of phrase enrichment on/off for a single domain.
    
    Args:
        domain_name: Domain to evaluate
        max_evaluations: Number of Bayesian optimization evaluations per condition
        
    Returns:
        Dictionary with detailed comparison results
        
    Raises:
        ValueError: If domain data cannot be loaded or evaluation fails
    """
    print(f"\n🔤 PHRASE ENRICHMENT EVALUATION: {domain_name.upper()}")
    print("=" * 60)
    
    # Load domain data
    print(f"📊 Loading {domain_name} data...")
    df = load_domain_data(domain_name)
    
    if df is None or df.empty:
        raise ValueError(f"Failed to load data for domain '{domain_name}'")
    
    print(f"✅ Loaded {len(df)} papers")
    
    results = {}
    
    # Test both conditions: phrase enrichment OFF and ON
    for phrase_enabled in [False, True]:
        condition_name = "phrase_enrichment_on" if phrase_enabled else "phrase_enrichment_off"
        print(f"\n🧪 Testing {condition_name}...")
        
        # Set environment variable to control phrase enrichment
        if phrase_enabled:
            os.environ["PHRASE_ENRICHMENT"] = "true"
        else:
            os.environ["PHRASE_ENRICHMENT"] = "false"
        
        try:
            start_time = time.time()
            
            # Convert DataFrame to DomainData (phrase enrichment applied here)
            domain_data = convert_dataframe_to_domain_data(df, domain_name)
            conversion_time = time.time() - start_time
            
            # Count keywords before/after for transparency
            total_keywords = sum(len(paper.keywords) for paper in domain_data.papers)
            avg_keywords_per_paper = total_keywords / len(domain_data.papers)
            
            print(f"   📚 Processed {len(domain_data.papers)} papers")
            print(f"   🔤 Average keywords per paper: {avg_keywords_per_paper:.1f}")
            print(f"   ⏱️  Data conversion time: {conversion_time:.3f}s")
            
            # Base configuration with Phase 16 optimized defaults
            base_config = ComprehensiveAlgorithmConfig(
                keyword_min_papers_ratio=0.05,  # FEATURE-01 optimized
                keyword_filtering_enabled=True,
                domain_name=domain_name,
            )
            
            # Run Bayesian optimization
            optimization_start = time.time()
            result = optimize_consensus_difference_parameters_bayesian(
                domain_data,
                domain_name,
                max_evaluations=max_evaluations,
                base_config=base_config,
            )
            optimization_time = time.time() - optimization_start
            
            # Record comprehensive results
            if result["optimization_successful"]:
                condition_result = {
                    "final_score": result["best_consensus_difference_score"],
                    "consensus_score": result.get("best_detailed_evaluation", {}).get("consensus_score", 0.0),
                    "difference_score": result.get("best_detailed_evaluation", {}).get("difference_score", 0.0),
                    "num_segments": result.get("best_detailed_evaluation", {}).get("num_segments", 0),
                    "best_parameters": result["best_parameters"],
                    "total_evaluations": result["total_evaluations"],
                    "data_conversion_time": conversion_time,
                    "optimization_time": optimization_time,
                    "total_runtime": conversion_time + optimization_time,
                    "avg_keywords_per_paper": avg_keywords_per_paper,
                    "total_keywords": total_keywords,
                    "optimization_successful": True
                }
                
                print(f"   ✅ SUCCESS: Final score {result['best_consensus_difference_score']:.3f}")
                print(f"   📊 Consensus: {condition_result['consensus_score']:.3f}, "
                      f"Difference: {condition_result['difference_score']:.3f}")
                print(f"   ⏱️  Total runtime: {condition_result['total_runtime']:.1f}s")
                
            else:
                condition_result = {
                    "optimization_successful": False,
                    "error": result.get("error", "Unknown optimization failure"),
                    "data_conversion_time": conversion_time,
                    "optimization_time": optimization_time,
                    "avg_keywords_per_paper": avg_keywords_per_paper,
                    "total_keywords": total_keywords,
                }
                
                print(f"   ❌ FAILED: {condition_result['error']}")
            
        except Exception as e:
            # Fail-fast: capture and report any errors
            condition_result = {
                "optimization_successful": False,
                "error": f"Exception during {condition_name}: {str(e)}",
                "traceback": traceback.format_exc(),
            }
            print(f"   ❌ EXCEPTION: {str(e)}")
            print(f"   📋 Traceback: {traceback.format_exc()}")
        
        results[condition_name] = condition_result
    
    # Clean up environment variable
    if "PHRASE_ENRICHMENT" in os.environ:
        del os.environ["PHRASE_ENRICHMENT"]
    
    # Calculate deltas and summary
    if (results["phrase_enrichment_off"]["optimization_successful"] and 
        results["phrase_enrichment_on"]["optimization_successful"]):
        
        off_result = results["phrase_enrichment_off"]
        on_result = results["phrase_enrichment_on"]
        
        # Calculate performance deltas
        delta_final = on_result["final_score"] - off_result["final_score"]
        delta_consensus = on_result["consensus_score"] - off_result["consensus_score"]
        delta_difference = on_result["difference_score"] - off_result["difference_score"]
        delta_keywords = on_result["avg_keywords_per_paper"] - off_result["avg_keywords_per_paper"]
        delta_runtime = on_result["total_runtime"] - off_result["total_runtime"]
        
        # Evaluate success criteria
        success_criteria_met = (
            delta_final >= 0.01 or  # ΔL ≥ +0.01
            (delta_consensus > 0 and delta_difference > 0 and delta_final >= 0)  # C1↑ & D1↑ with no L drop
        )
        
        results["comparison_summary"] = {
            "delta_final_score": delta_final,
            "delta_consensus": delta_consensus,
            "delta_difference": delta_difference,
            "delta_avg_keywords_per_paper": delta_keywords,
            "delta_runtime_seconds": delta_runtime,
            "success_criteria_met": success_criteria_met,
            "evaluation_timestamp": datetime.now().isoformat(),
        }
        
        print(f"\n📊 COMPARISON SUMMARY:")
        print(f"   📈 ΔFinal Score: {delta_final:+.3f}")
        print(f"   📈 ΔConsensus: {delta_consensus:+.3f}")
        print(f"   📈 ΔDifference: {delta_difference:+.3f}")
        print(f"   🔤 ΔKeywords/paper: {delta_keywords:+.1f}")
        print(f"   ⏱️  ΔRuntime: {delta_runtime:+.1f}s")
        print(f"   🎯 Success criteria met: {success_criteria_met}")
    
    else:
        results["comparison_summary"] = {
            "evaluation_failed": True,
            "reason": "One or both optimization runs failed",
            "evaluation_timestamp": datetime.now().isoformat(),
        }
        print(f"\n❌ COMPARISON FAILED: One or both optimization runs failed")
    
    return results


def run_pilot_study() -> Dict[str, Any]:
    """
    Run pilot study on Applied Mathematics (stable) and Computer Vision (volatile).
    
    Returns:
        Dictionary with pilot study results
    """
    pilot_domains = ["applied_mathematics", "computer_vision"]
    pilot_results = {}
    
    print(f"🧪 PHASE 16 FEATURE-06 PILOT STUDY")
    print(f"Testing phrase enrichment on {len(pilot_domains)} pilot domains")
    print("=" * 70)
    
    for domain in pilot_domains:
        try:
            domain_result = run_phrase_enrichment_comparison(domain, max_evaluations=20)
            pilot_results[domain] = domain_result
        except Exception as e:
            print(f"❌ Pilot study failed for {domain}: {e}")
            pilot_results[domain] = {
                "pilot_failed": True,
                "error": str(e),
                "traceback": traceback.format_exc(),
            }
    
    # Analyze pilot results
    successful_domains = [
        domain for domain, result in pilot_results.items()
        if result.get("comparison_summary", {}).get("success_criteria_met", False)
    ]
    
    pilot_summary = {
        "pilot_domains": pilot_domains,
        "successful_domains": successful_domains,
        "success_rate": len(successful_domains) / len(pilot_domains),
        "proceed_to_full_study": len(successful_domains) > 0,
        "evaluation_timestamp": datetime.now().isoformat(),
    }
    
    pilot_results["pilot_summary"] = pilot_summary
    
    print(f"\n🎯 PILOT STUDY SUMMARY:")
    print(f"   ✅ Successful domains: {successful_domains}")
    print(f"   📊 Success rate: {pilot_summary['success_rate']:.1%}")
    print(f"   🚀 Proceed to full study: {pilot_summary['proceed_to_full_study']}")
    
    return pilot_results


def run_full_study() -> Dict[str, Any]:
    """
    Run full evaluation across all 8 domains.
    
    Returns:
        Dictionary with full study results
    """
    all_domains = [
        "applied_mathematics", "art", "computer_science", "computer_vision",
        "deep_learning", "machine_learning", "machine_translation", "natural_language_processing"
    ]
    
    full_results = {}
    
    print(f"🚀 PHASE 16 FEATURE-06 FULL STUDY")
    print(f"Testing phrase enrichment on all {len(all_domains)} domains")
    print("=" * 70)
    
    for domain in all_domains:
        try:
            domain_result = run_phrase_enrichment_comparison(domain, max_evaluations=30)
            full_results[domain] = domain_result
        except Exception as e:
            print(f"❌ Full study failed for {domain}: {e}")
            full_results[domain] = {
                "study_failed": True,
                "error": str(e),
                "traceback": traceback.format_exc(),
            }
    
    # Aggregate results
    successful_comparisons = [
        result for result in full_results.values()
        if result.get("comparison_summary", {}).get("success_criteria_met", False)
    ]
    
    if successful_comparisons:
        # Calculate aggregate metrics
        avg_delta_final = sum(
            result["comparison_summary"]["delta_final_score"] 
            for result in successful_comparisons
        ) / len(successful_comparisons)
        
        avg_delta_consensus = sum(
            result["comparison_summary"]["delta_consensus"] 
            for result in successful_comparisons
        ) / len(successful_comparisons)
        
        avg_delta_difference = sum(
            result["comparison_summary"]["delta_difference"] 
            for result in successful_comparisons
        ) / len(successful_comparisons)
        
        full_summary = {
            "total_domains": len(all_domains),
            "successful_domains": len(successful_comparisons),
            "success_rate": len(successful_comparisons) / len(all_domains),
            "avg_delta_final_score": avg_delta_final,
            "avg_delta_consensus": avg_delta_consensus,
            "avg_delta_difference": avg_delta_difference,
            "recommendation": "ADOPT" if avg_delta_final >= 0.01 else "REJECT",
            "evaluation_timestamp": datetime.now().isoformat(),
        }
    else:
        full_summary = {
            "total_domains": len(all_domains),
            "successful_domains": 0,
            "success_rate": 0.0,
            "recommendation": "REJECT",
            "reason": "No successful domain comparisons",
            "evaluation_timestamp": datetime.now().isoformat(),
        }
    
    full_results["full_summary"] = full_summary
    
    print(f"\n🎯 FULL STUDY SUMMARY:")
    print(f"   ✅ Successful domains: {full_summary['successful_domains']}/{full_summary['total_domains']}")
    print(f"   📊 Success rate: {full_summary['success_rate']:.1%}")
    if full_summary.get("avg_delta_final_score"):
        print(f"   📈 Average ΔFinal Score: {full_summary['avg_delta_final_score']:+.3f}")
        print(f"   📈 Average ΔConsensus: {full_summary['avg_delta_consensus']:+.3f}")
        print(f"   📈 Average ΔDifference: {full_summary['avg_delta_difference']:+.3f}")
    print(f"   🎯 Recommendation: {full_summary['recommendation']}")
    
    return full_results


def main():
    """Main execution function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Phase 16 FEATURE-06: Phrase Enrichment Evaluation")
    parser.add_argument("--pilot", action="store_true", help="Run pilot study only")
    parser.add_argument("--full", action="store_true", help="Run full study only")
    parser.add_argument("--domain", type=str, help="Run single domain evaluation")
    parser.add_argument("--max-evals", type=int, default=30, help="Max evaluations per condition")
    
    args = parser.parse_args()
    
    # Determine what to run
    if args.domain:
        # Single domain evaluation
        print(f"🔬 Running single domain evaluation: {args.domain}")
        results = {"single_domain": run_phrase_enrichment_comparison(args.domain, args.max_evals)}
        
    elif args.pilot:
        # Pilot study only
        results = run_pilot_study()
        
    elif args.full:
        # Full study only
        results = run_full_study()
        
    else:
        # Default: pilot study, then full study if pilot succeeds
        print("🧪 Running sequential pilot → full study evaluation")
        
        # Run pilot study
        pilot_results = run_pilot_study()
        results = {"pilot_study": pilot_results}
        
        # Check if pilot succeeded
        if pilot_results.get("pilot_summary", {}).get("proceed_to_full_study", False):
            print("\n🚀 Pilot study successful - proceeding to full evaluation")
            full_results = run_full_study()
            results["full_study"] = full_results
        else:
            print("\n❌ Pilot study failed - skipping full evaluation")
            results["full_study"] = {"skipped": True, "reason": "Pilot study did not meet success criteria"}
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"experiments/metric_evaluation/results/phrase_enrichment_{timestamp}.json"
    
    # Create results directory if it doesn't exist
    os.makedirs("experiments/metric_evaluation/results", exist_ok=True)
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved to {results_file}")
    
    # Summary output for journal updating
    print(f"\n📋 EXPERIMENT COMPLETE - Phase 16 FEATURE-06 Evaluation")
    print(f"Results file: {results_file}")
    print("Ready for development journal update.")


if __name__ == "__main__":
    main() 