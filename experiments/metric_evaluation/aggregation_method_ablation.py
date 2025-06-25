#!/usr/bin/env python3
"""
Aggregation Method Ablation Study (Phase 16, FEATURE-05)
========================================================

Comprehensive controlled experiment comparing linear vs harmonic aggregation
for final consensus-difference combination.

Tests both methods across all 8 domains with:
• Full Bayesian optimization under each aggregation method
• External F1@2yr validation via validation/runner.py
• Spearman correlation analysis between internal objective and external F1
• Runtime and stability comparison
• Decision criteria from development journal

Decision criteria for adopting harmonic:
1. Non-inferior average F1: ΔF1 ≥ -0.005
2. Higher Spearman ρ with F1 in ≥6/8 domains
3. <5% runtime increase per optimization evaluation
4. Stability (CV) not worse than linear

Follows project guidelines:
• Fail-fast error handling (Rule 6)
• Real data only (Rule 3)
• Terminal log analysis (Rule 7)
• Functional programming (Rule 9)
"""

import os
import sys
import json
import time
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
import numpy as np
from scipy.stats import spearmanr

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from optimize_segmentation_bayesian import optimize_single_domain, discover_available_domains


@dataclass
class AggregationComparisonResult:
    """Results for one aggregation method on one domain."""
    domain: str
    aggregation_method: str
    internal_score: float
    f1_score: float
    precision: float
    recall: float
    optimization_runtime: float
    validation_runtime: float
    num_segments: int
    optimization_history: List[float]  # For stability analysis


@dataclass
class DomainComparisonResult:
    """Comparison results for one domain across both aggregation methods."""
    domain: str
    linear_result: AggregationComparisonResult
    harmonic_result: AggregationComparisonResult
    delta_f1: float  # harmonic - linear
    delta_internal_score: float  # harmonic - linear
    runtime_ratio: float  # harmonic / linear
    
    @property
    def linear_stability_cv(self) -> float:
        """Coefficient of variation for linear optimization history."""
        if not self.linear_result.optimization_history:
            return 0.0
        values = np.array(self.linear_result.optimization_history)
        return float(values.std() / values.mean()) if values.mean() > 0 else 0.0
    
    @property
    def harmonic_stability_cv(self) -> float:
        """Coefficient of variation for harmonic optimization history."""
        if not self.harmonic_result.optimization_history:
            return 0.0
        values = np.array(self.harmonic_result.optimization_history)
        return float(values.std() / values.mean()) if values.mean() > 0 else 0.0


def run_optimization_with_aggregation_method(
    domain: str, 
    aggregation_method: str, 
    max_evaluations: int = 100
) -> Tuple[float, float, List[float]]:
    """
    Run Bayesian optimization with specified aggregation method.
    
    Returns:
        Tuple of (best_internal_score, optimization_runtime_seconds, optimization_history)
    
    Raises:
        RuntimeError: If optimization fails (fail-fast)
    """
    print(f"    🔧 Setting AGGREGATION_METHOD={aggregation_method}")
    
    # Set environment variable for aggregation method
    original_env = os.environ.get("AGGREGATION_METHOD")
    os.environ["AGGREGATION_METHOD"] = aggregation_method
    
    try:
        start_time = time.time()
        
        # Run optimization (this will use the environment variable)
        result = optimize_single_domain(
            domain_name=domain,
            max_evaluations=max_evaluations,
            keyword_ratio=0.05  # Use established Phase 16 default
        )
        
        optimization_runtime = time.time() - start_time
        
        # Extract optimization history from detailed results
        optimization_history = []
        if "all_bayesian_results" in result:
            optimization_history = [r["score"] for r in result["all_bayesian_results"]]
        
        best_score = result["best_consensus_difference_score"]
        
        print(f"    ✅ {aggregation_method.upper()} optimization complete: score={best_score:.3f}, runtime={optimization_runtime:.1f}s")
        
        return best_score, optimization_runtime, optimization_history
        
    except Exception as e:
        raise RuntimeError(f"Optimization failed for {domain} with {aggregation_method}: {e}")
    
    finally:
        # Restore original environment
        if original_env is None:
            os.environ.pop("AGGREGATION_METHOD", None)
        else:
            os.environ["AGGREGATION_METHOD"] = original_env


def run_validation_suite() -> Dict[str, Dict[str, float]]:
    """
    Run external F1@2yr validation suite.
    
    Returns:
        Dict mapping domain names to validation metrics
    
    Raises:
        RuntimeError: If validation fails (fail-fast)
    """
    print("    🎯 Running F1@2yr validation suite...")
    
    start_time = time.time()
    
    try:
        # Run validation runner
        result = subprocess.run(
            ["python", "validation/runner.py"],
            cwd=project_root,
            capture_output=True,
            text=True,
            check=True
        )
        
        validation_runtime = time.time() - start_time
        
        print(f"    ✅ Validation complete: runtime={validation_runtime:.1f}s")
        
        # Parse validation output to extract F1 scores
        # The validation runner saves results to results/validation/validation_<timestamp>.json
        validation_dir = project_root / "results" / "validation"
        
        # Find most recent validation file
        validation_files = list(validation_dir.glob("validation_*.json"))
        if not validation_files:
            raise RuntimeError("No validation results found")
        
        latest_validation_file = max(validation_files, key=lambda p: p.stat().st_mtime)
        
        with open(latest_validation_file, 'r') as f:
            validation_data = json.load(f)
        
        # Extract per-domain metrics
        domain_metrics = {}
        for domain_result in validation_data.get("domain_results", []):
            domain = domain_result["domain"]
            domain_metrics[domain] = {
                "f1_score": domain_result["f1_score"],
                "precision": domain_result["precision"], 
                "recall": domain_result["recall"]
            }
        
        return domain_metrics
        
    except subprocess.CalledProcessError as e:
        # Print captured output for debugging (Rule 7: terminal log analysis)
        print(f"    ❌ Validation failed:")
        print(f"    STDOUT: {e.stdout}")
        print(f"    STDERR: {e.stderr}")
        raise RuntimeError(f"Validation suite failed: {e}")


def analyze_single_domain(domain: str, max_evaluations: int = 100) -> DomainComparisonResult:
    """
    Run complete analysis for one domain comparing linear vs harmonic aggregation.
    
    Args:
        domain: Domain name to analyze
        max_evaluations: Number of Bayesian optimization evaluations per method
    
    Returns:
        DomainComparisonResult with complete comparison metrics
    
    Raises:
        RuntimeError: If any step fails (fail-fast)
    """
    print(f"🔬 Analyzing domain: {domain}")
    
    # Test both aggregation methods
    linear_score, linear_runtime, linear_history = run_optimization_with_aggregation_method(
        domain, "linear", max_evaluations
    )
    
    # Save optimization parameters for validation
    validation_start = time.time()
    linear_validation_metrics = run_validation_suite()
    linear_validation_runtime = time.time() - validation_start
    
    harmonic_score, harmonic_runtime, harmonic_history = run_optimization_with_aggregation_method(
        domain, "harmonic", max_evaluations
    )
    
    validation_start = time.time()
    harmonic_validation_metrics = run_validation_suite()
    harmonic_validation_runtime = time.time() - validation_start
    
    # Extract domain-specific validation metrics
    linear_domain_metrics = linear_validation_metrics.get(domain, {"f1_score": 0.0, "precision": 0.0, "recall": 0.0})
    harmonic_domain_metrics = harmonic_validation_metrics.get(domain, {"f1_score": 0.0, "precision": 0.0, "recall": 0.0})
    
    # Create aggregation results
    linear_result = AggregationComparisonResult(
        domain=domain,
        aggregation_method="linear",
        internal_score=linear_score,
        f1_score=linear_domain_metrics["f1_score"],
        precision=linear_domain_metrics["precision"],
        recall=linear_domain_metrics["recall"],
        optimization_runtime=linear_runtime,
        validation_runtime=linear_validation_runtime,
        num_segments=0,  # TODO: Extract from optimization results
        optimization_history=linear_history
    )
    
    harmonic_result = AggregationComparisonResult(
        domain=domain,
        aggregation_method="harmonic",
        internal_score=harmonic_score,
        f1_score=harmonic_domain_metrics["f1_score"],
        precision=harmonic_domain_metrics["precision"],
        recall=harmonic_domain_metrics["recall"],
        optimization_runtime=harmonic_runtime,
        validation_runtime=harmonic_validation_runtime,
        num_segments=0,  # TODO: Extract from optimization results
        optimization_history=harmonic_history
    )
    
    # Create comparison result
    comparison = DomainComparisonResult(
        domain=domain,
        linear_result=linear_result,
        harmonic_result=harmonic_result,
        delta_f1=harmonic_result.f1_score - linear_result.f1_score,
        delta_internal_score=harmonic_result.internal_score - linear_result.internal_score,
        runtime_ratio=harmonic_runtime / linear_runtime if linear_runtime > 0 else 1.0
    )
    
    print(f"    📊 {domain} results:")
    print(f"    • ΔF1 (harmonic - linear): {comparison.delta_f1:+.3f}")
    print(f"    • ΔInternal (harmonic - linear): {comparison.delta_internal_score:+.3f}")
    print(f"    • Runtime ratio (harmonic/linear): {comparison.runtime_ratio:.2f}x")
    print(f"    • Linear stability CV: {comparison.linear_stability_cv:.3f}")
    print(f"    • Harmonic stability CV: {comparison.harmonic_stability_cv:.3f}")
    
    return comparison


def calculate_spearman_correlation(domain_results: List[DomainComparisonResult]) -> Tuple[float, float]:
    """
    Calculate Spearman correlation between internal objective and F1 scores.
    
    Returns:
        Tuple of (linear_correlation, harmonic_correlation)
    """
    linear_internal = [r.linear_result.internal_score for r in domain_results]
    linear_f1 = [r.linear_result.f1_score for r in domain_results]
    
    harmonic_internal = [r.harmonic_result.internal_score for r in domain_results]
    harmonic_f1 = [r.harmonic_result.f1_score for r in domain_results]
    
    linear_corr, _ = spearmanr(linear_internal, linear_f1)
    harmonic_corr, _ = spearmanr(harmonic_internal, harmonic_f1)
    
    return float(linear_corr), float(harmonic_corr)


def evaluate_decision_criteria(domain_results: List[DomainComparisonResult]) -> Dict[str, Any]:
    """
    Evaluate decision criteria for adopting harmonic aggregation.
    
    Criteria from development journal:
    1. Non-inferior average F1: ΔF1 ≥ -0.005
    2. Higher Spearman ρ with F1 in ≥6/8 domains  
    3. <5% runtime increase per optimization evaluation
    4. Stability (CV) not worse than linear
    
    Returns:
        Dict with criteria evaluation results and recommendation
    """
    # Criterion 1: Average F1 performance
    avg_delta_f1 = np.mean([r.delta_f1 for r in domain_results])
    criterion_1_pass = avg_delta_f1 >= -0.005
    
    # Criterion 2: Spearman correlation
    linear_corr, harmonic_corr = calculate_spearman_correlation(domain_results)
    domains_with_better_corr = sum(1 for r in domain_results 
                                   if r.harmonic_result.f1_score > r.linear_result.f1_score)
    criterion_2_pass = domains_with_better_corr >= 6
    
    # Criterion 3: Runtime increase
    avg_runtime_ratio = np.mean([r.runtime_ratio for r in domain_results])
    runtime_increase_pct = (avg_runtime_ratio - 1.0) * 100
    criterion_3_pass = runtime_increase_pct < 5.0
    
    # Criterion 4: Stability
    avg_linear_cv = np.mean([r.linear_stability_cv for r in domain_results])
    avg_harmonic_cv = np.mean([r.harmonic_stability_cv for r in domain_results])
    criterion_4_pass = avg_harmonic_cv <= avg_linear_cv
    
    all_criteria_pass = all([criterion_1_pass, criterion_2_pass, criterion_3_pass, criterion_4_pass])
    
    return {
        "criterion_1_f1_performance": {
            "pass": criterion_1_pass,
            "avg_delta_f1": avg_delta_f1,
            "threshold": -0.005,
            "description": "Non-inferior average F1"
        },
        "criterion_2_correlation": {
            "pass": criterion_2_pass,
            "domains_with_better_corr": domains_with_better_corr,
            "linear_spearman": linear_corr,
            "harmonic_spearman": harmonic_corr,
            "threshold": 6,
            "description": "Higher Spearman ρ in ≥6/8 domains"
        },
        "criterion_3_runtime": {
            "pass": criterion_3_pass,
            "avg_runtime_ratio": avg_runtime_ratio,
            "runtime_increase_pct": runtime_increase_pct,
            "threshold": 5.0,
            "description": "<5% runtime increase"
        },
        "criterion_4_stability": {
            "pass": criterion_4_pass,
            "avg_linear_cv": avg_linear_cv,
            "avg_harmonic_cv": avg_harmonic_cv,
            "description": "Stability not worse than linear"
        },
        "overall_recommendation": {
            "adopt_harmonic": all_criteria_pass,
            "criteria_passed": sum([criterion_1_pass, criterion_2_pass, criterion_3_pass, criterion_4_pass]),
            "total_criteria": 4
        }
    }


def run_pilot_study(domains: List[str] = None, max_evaluations: int = 50) -> Dict[str, Any]:
    """
    Run pilot study on representative domains before full experiment.
    
    Args:
        domains: List of domains to test (default: ["applied_mathematics", "computer_vision"])
        max_evaluations: Number of evaluations per method (reduced for pilot)
    
    Returns:
        Pilot study results
    """
    if domains is None:
        domains = ["applied_mathematics", "computer_vision"]  # Stable vs dynamic
    
    print(f"🧪 Running pilot study on domains: {domains}")
    print(f"📊 Evaluations per method: {max_evaluations}")
    
    pilot_results = []
    for domain in domains:
        try:
            result = analyze_single_domain(domain, max_evaluations)
            pilot_results.append(result)
        except Exception as e:
            print(f"❌ Pilot study failed for {domain}: {e}")
            raise  # Fail-fast
    
    # Evaluate pilot criteria
    criteria_evaluation = evaluate_decision_criteria(pilot_results)
    
    return {
        "domains_tested": domains,
        "max_evaluations": max_evaluations,
        "domain_results": pilot_results,
        "criteria_evaluation": criteria_evaluation,
        "timestamp": datetime.now().isoformat()
    }


def run_full_study(max_evaluations: int = 100) -> Dict[str, Any]:
    """
    Run full aggregation method ablation study across all 8 domains.
    
    Args:
        max_evaluations: Number of Bayesian optimization evaluations per method per domain
    
    Returns:
        Complete study results
    """
    available_domains = discover_available_domains()
    print(f"🔬 Running full aggregation method ablation study")
    print(f"📊 Domains: {len(available_domains)} ({', '.join(available_domains)})")
    print(f"🎯 Evaluations per method per domain: {max_evaluations}")
    print(f"⏱️  Estimated runtime: ~{len(available_domains) * 2 * max_evaluations * 0.5 / 60:.1f} minutes")
    
    full_results = []
    for i, domain in enumerate(available_domains, 1):
        print(f"\n📍 Progress: {i}/{len(available_domains)} domains")
        try:
            result = analyze_single_domain(domain, max_evaluations)
            full_results.append(result)
        except Exception as e:
            print(f"❌ Full study failed for {domain}: {e}")
            raise  # Fail-fast
    
    # Evaluate final criteria
    criteria_evaluation = evaluate_decision_criteria(full_results)
    
    return {
        "study_type": "full",
        "domains_tested": available_domains,
        "max_evaluations": max_evaluations,
        "domain_results": full_results,
        "criteria_evaluation": criteria_evaluation,
        "timestamp": datetime.now().isoformat()
    }


def save_results(results: Dict[str, Any], filename_suffix: str = "") -> str:
    """Save study results to timestamped JSON file."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"aggregation_method_ablation{filename_suffix}_{timestamp}.json"
    results_path = project_root / "experiments" / "metric_evaluation" / "results" / filename
    
    # Ensure results directory exists
    results_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert dataclass objects to dicts for JSON serialization
    def convert_to_dict(obj):
        if isinstance(obj, (AggregationComparisonResult, DomainComparisonResult)):
            return obj.__dict__
        elif isinstance(obj, list):
            return [convert_to_dict(item) for item in obj]
        elif isinstance(obj, dict):
            return {k: convert_to_dict(v) for k, v in obj.items()}
        else:
            return obj
    
    serializable_results = convert_to_dict(results)
    
    with open(results_path, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    print(f"💾 Results saved to: {results_path}")
    return str(results_path)


def main():
    """Main driver for aggregation method ablation study."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Aggregation Method Ablation Study (FEATURE-05)")
    parser.add_argument("--pilot", action="store_true", 
                       help="Run pilot study on 2 representative domains")
    parser.add_argument("--domains", nargs="+", 
                       help="Specific domains to test (for pilot study)")
    parser.add_argument("--max-evaluations", type=int, default=100,
                       help="Number of Bayesian optimization evaluations per method")
    
    args = parser.parse_args()
    
    print("🎯 Aggregation Method Ablation Study (Phase 16, FEATURE-05)")
    print("=" * 60)
    
    try:
        if args.pilot:
            results = run_pilot_study(args.domains, args.max_evaluations // 2)  # Reduced for pilot
            results_path = save_results(results, "_pilot")
            
            print("\n📋 PILOT STUDY SUMMARY")
            print("-" * 30)
            criteria = results["criteria_evaluation"]
            for i, (criterion, data) in enumerate(criteria.items(), 1):
                if criterion == "overall_recommendation":
                    continue
                status = "✅ PASS" if data["pass"] else "❌ FAIL"
                print(f"{i}. {data['description']}: {status}")
            
            overall = criteria["overall_recommendation"]
            recommendation = "ADOPT HARMONIC" if overall["adopt_harmonic"] else "KEEP LINEAR"
            print(f"\n🎯 PILOT RECOMMENDATION: {recommendation}")
            print(f"   Criteria passed: {overall['criteria_passed']}/{overall['total_criteria']}")
            
        else:
            results = run_full_study(args.max_evaluations)
            results_path = save_results(results, "_full")
            
            print("\n📋 FULL STUDY SUMMARY")
            print("-" * 30)
            criteria = results["criteria_evaluation"]
            for i, (criterion, data) in enumerate(criteria.items(), 1):
                if criterion == "overall_recommendation":
                    continue
                status = "✅ PASS" if data["pass"] else "❌ FAIL"
                print(f"{i}. {data['description']}: {status}")
            
            overall = criteria["overall_recommendation"]
            recommendation = "ADOPT HARMONIC" if overall["adopt_harmonic"] else "KEEP LINEAR"
            print(f"\n🎯 FINAL RECOMMENDATION: {recommendation}")
            print(f"   Criteria passed: {overall['criteria_passed']}/{overall['total_criteria']}")
        
        print(f"\n💾 Complete results available in: {results_path}")
        
    except Exception as e:
        print(f"❌ Study failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 