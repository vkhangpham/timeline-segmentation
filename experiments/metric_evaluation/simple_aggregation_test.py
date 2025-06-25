#!/usr/bin/env python3
"""
Simple Aggregation Method Test (Phase 16, FEATURE-05)
=====================================================

Simplified test to validate that:
1. Configuration infrastructure works correctly
2. Linear vs harmonic methods produce different optimization results
3. Environment override works as expected

This is a focused test following Rule 4 (iterative testing with real data subsets).
"""

import os
import sys
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from optimize_segmentation_bayesian import optimize_single_domain


def test_aggregation_methods(domain: str = "applied_mathematics", max_evaluations: int = 20):
    """
    Test both aggregation methods on a single domain and compare results.
    
    Args:
        domain: Domain to test on
        max_evaluations: Number of optimization evaluations per method
    """
    print(f"🔬 Testing aggregation methods on domain: {domain}")
    print(f"📊 Evaluations per method: {max_evaluations}")
    print("=" * 60)
    
    results = {}
    
    for method in ["linear", "harmonic"]:
        print(f"\n🎯 Testing {method.upper()} aggregation...")
        
        # Set environment variable
        os.environ["AGGREGATION_METHOD"] = method
        
        start_time = time.time()
        
        try:
            result = optimize_single_domain(
                domain_name=domain,
                max_evaluations=max_evaluations,
                keyword_ratio=0.05  # Use Phase 16 default
            )
            
            runtime = time.time() - start_time
            
            # Extract key metrics
            results[method] = {
                "score": result["best_consensus_difference_score"],
                "consensus": result["best_detailed_evaluation"]["consensus_score"],
                "difference": result["best_detailed_evaluation"]["difference_score"],
                "num_segments": result["best_detailed_evaluation"]["num_segments"],
                "runtime": runtime,
                "parameters": result["best_parameters"],
                "methodology": result["best_detailed_evaluation"]["methodology_explanation"]
            }
            
            print(f"    ✅ {method.upper()} Results:")
            print(f"    • Score: {results[method]['score']:.3f}")
            print(f"    • Consensus: {results[method]['consensus']:.3f}")
            print(f"    • Difference: {results[method]['difference']:.3f}")
            print(f"    • Segments: {results[method]['num_segments']}")
            print(f"    • Runtime: {results[method]['runtime']:.1f}s")
            print(f"    • Method confirmed: {'harmonic' if 'harmonic' in results[method]['methodology'].lower() else 'linear'}")
            
        except Exception as e:
            print(f"    ❌ {method.upper()} failed: {e}")
            results[method] = {"error": str(e)}
        
        finally:
            # Clean up environment
            os.environ.pop("AGGREGATION_METHOD", None)
    
    # Compare results
    print("\n📊 COMPARISON RESULTS")
    print("=" * 60)
    
    if "error" in results["linear"]:
        print(f"❌ Linear method failed: {results['linear']['error']}")
        return
    
    if "error" in results["harmonic"]:
        print(f"❌ Harmonic method failed: {results['harmonic']['error']}")
        return
    
    # Calculate differences
    delta_score = results["harmonic"]["score"] - results["linear"]["score"]
    delta_consensus = results["harmonic"]["consensus"] - results["linear"]["consensus"]
    delta_difference = results["harmonic"]["difference"] - results["linear"]["difference"]
    delta_segments = results["harmonic"]["num_segments"] - results["linear"]["num_segments"]
    runtime_ratio = results["harmonic"]["runtime"] / results["linear"]["runtime"] if results["linear"]["runtime"] > 0 else 1.0
    
    print(f"📈 Score Difference (harmonic - linear): {delta_score:+.3f}")
    print(f"🤝 Consensus Difference: {delta_consensus:+.3f}")
    print(f"🔄 Difference Metric Difference: {delta_difference:+.3f}")
    print(f"🏗️  Segments Difference: {delta_segments:+d}")
    print(f"⏱️  Runtime Ratio (harmonic/linear): {runtime_ratio:.2f}x")
    
    # Validate that methods are actually different
    methods_different = (
        abs(delta_score) > 0.001 or
        abs(delta_consensus) > 0.001 or 
        abs(delta_difference) > 0.001 or
        delta_segments != 0
    )
    
    if methods_different:
        print("\n✅ SUCCESS: Methods produce different results as expected!")
        print("🎯 Infrastructure working correctly - ready for full ablation study")
    else:
        print("\n⚠️  WARNING: Methods produce identical results")
        print("🔍 May indicate configuration issue or edge case")
    
    # Check methodology confirmation
    linear_confirmed = "linear" in results["linear"]["methodology"].lower()
    harmonic_confirmed = "harmonic" in results["harmonic"]["methodology"].lower()
    
    if linear_confirmed and harmonic_confirmed:
        print("✅ Methodology reporting confirmed for both methods")
    else:
        print(f"⚠️  Methodology reporting issue: linear={linear_confirmed}, harmonic={harmonic_confirmed}")
    
    return results


def main():
    """Main test runner."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Simple Aggregation Method Test")
    parser.add_argument("--domain", default="applied_mathematics", 
                       help="Domain to test (default: applied_mathematics)")
    parser.add_argument("--max-evaluations", type=int, default=20,
                       help="Number of optimization evaluations per method")
    
    args = parser.parse_args()
    
    print("🎯 Simple Aggregation Method Test (Phase 16, FEATURE-05)")
    print("=" * 60)
    
    try:
        results = test_aggregation_methods(args.domain, args.max_evaluations)
        if results:
            print(f"\n💾 Test completed successfully!")
            return 0
        else:
            print(f"\n❌ Test failed!")
            return 1
            
    except Exception as e:
        print(f"❌ Test failed with exception: {e}")
        return 1


if __name__ == "__main__":
    exit(main()) 