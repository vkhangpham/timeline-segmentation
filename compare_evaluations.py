#!/usr/bin/env python3
"""
Compare Multiple Evaluation Results

This script compares multiple evaluation output files to analyze
relative performance between different algorithms or configurations.

Usage:
    python compare_evaluations.py validation/deep_learning_evaluation_results.json validation/deep_learning_baseline_evaluation_results.json
    python compare_evaluations.py validation/*_evaluation_results.json
    python compare_evaluations.py --output comparison_report.json validation/*.json
"""

import json
import os
import sys
import argparse
from pathlib import Path
from typing import List, Dict, Any
import glob


def load_evaluation_results(file_paths: List[str]) -> List[Dict[str, Any]]:
    """
    Load multiple evaluation result files.
    
    Args:
        file_paths: List of paths to evaluation result files
        
    Returns:
        List of evaluation result dictionaries
    """
    results = []
    
    for file_path in file_paths:
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                data['_file_path'] = file_path
                data['_file_name'] = os.path.basename(file_path)
                results.append(data)
                print(f"✅ Loaded: {file_path}")
        except Exception as e:
            print(f"❌ Error loading {file_path}: {e}")
    
    return results


def extract_algorithm_info(result: Dict[str, Any]) -> Dict[str, str]:
    """Extract algorithm and domain information from result."""
    file_name = result['_file_name']
    
    # Extract algorithm name
    if 'algorithm' in result:
        algorithm = result['algorithm']
    elif 'baseline' in file_name.lower():
        algorithm = 'baseline'
    else:
        algorithm = 'current'
    
    # Extract domain
    if 'domain' in result:
        domain = result['domain']
    else:
        # Try to extract from filename
        domain = file_name.replace('_evaluation_results.json', '').replace('_baseline', '')
    
    return {'algorithm': algorithm, 'domain': domain}


def compare_evaluation_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compare multiple evaluation results and generate comprehensive report.
    
    Args:
        results: List of evaluation result dictionaries
        
    Returns:
        Comparison report dictionary
    """
    print("\n" + "=" * 80)
    print("MULTI-ALGORITHM EVALUATION COMPARISON")
    print("=" * 80)
    
    if len(results) < 2:
        print("⚠️ Need at least 2 evaluation results for comparison")
        return {}
    
    # Organize results by domain and algorithm
    by_domain = {}
    algorithm_names = set()
    
    for result in results:
        info = extract_algorithm_info(result)
        domain = info['domain']
        algorithm = info['algorithm']
        
        if domain not in by_domain:
            by_domain[domain] = {}
        
        by_domain[domain][algorithm] = result
        algorithm_names.add(algorithm)
    
    algorithm_names = sorted(algorithm_names)
    
    print(f"\n📊 COMPARISON OVERVIEW:")
    print("-" * 30)
    print(f"Domains: {len(by_domain)} ({', '.join(sorted(by_domain.keys()))})")
    print(f"Algorithms: {len(algorithm_names)} ({', '.join(algorithm_names)})")
    print()
    
    # Generate comparison report
    comparison_report = {
        "comparison_type": "multi_algorithm_evaluation",
        "domains": list(by_domain.keys()),
        "algorithms": algorithm_names,
        "domain_comparisons": {},
        "overall_summary": {}
    }
    
    # Compare each domain
    for domain, domain_results in by_domain.items():
        if len(domain_results) < 2:
            print(f"⚠️ Skipping {domain}: Only {len(domain_results)} algorithm(s) available")
            continue
        
        print(f"\n{'='*20} {domain.upper()} COMPARISON {'='*20}")
        
        domain_comparison = compare_domain_results(domain_results, algorithm_names)
        comparison_report["domain_comparisons"][domain] = domain_comparison
    
    # Generate overall summary
    overall_summary = generate_overall_summary(comparison_report["domain_comparisons"])
    comparison_report["overall_summary"] = overall_summary
    
    print(f"\n🏆 OVERALL COMPARISON SUMMARY")
    print("=" * 40)
    print_overall_summary(overall_summary)
    
    return comparison_report


def compare_domain_results(domain_results: Dict[str, Dict], algorithm_names: List[str]) -> Dict[str, Any]:
    """Compare algorithms within a single domain."""
    
    # Create comparison table
    metrics_to_compare = ['precision', 'recall', 'f1_score']
    
    print(f"\n📈 PERFORMANCE METRICS:")
    print("-" * 30)
    
    # Header
    header = f"{'Algorithm':<15}"
    for metric in metrics_to_compare:
        header += f"{metric.replace('_', ' ').title():<12}"
    header += "Assessment"
    print(header)
    print("-" * len(header))
    
    # Algorithm rows
    algorithm_data = {}
    
    for algorithm in algorithm_names:
        if algorithm not in domain_results:
            continue
            
        result = domain_results[algorithm]
        
        # Handle nested evaluation structure for enhanced LLM evaluation
        if 'recall_evaluation' in result:
            # Enhanced evaluation with LLM - metrics are nested
            metrics = result['recall_evaluation']['metrics']
        else:
            # Standard evaluation - metrics are at top level
            metrics = result.get('metrics', {})
        
        # Use enhanced LLM precision if available, otherwise standard
        if 'enhanced_llm_evaluation' in result:
            display_metrics = {
                'precision': result['enhanced_precision'],
                'recall': metrics['recall'],  # Always use GT recall
                'f1_score': metrics['f1_score']  # Use GT F1
            }
        else:
            display_metrics = {
                'precision': metrics.get('precision', 0),
                'recall': metrics.get('recall', 0),
                'f1_score': metrics.get('f1_score', 0)
            }
        
        assessment = result.get('assessment', 'Unknown')
        assessment_short = assessment.split(':')[0] if ':' in assessment else assessment
        
        # Print row
        row = f"{algorithm:<15}"
        for metric in metrics_to_compare:
            value = display_metrics[metric]
            row += f"{value:<12.1%}" if value <= 1.0 else f"{value:<12.3f}"
        row += assessment_short
        print(row)
        
        # Store for comparison
        algorithm_data[algorithm] = {
            'metrics': display_metrics,
            'assessment': assessment,
            'raw_result': result
        }
    
    # Calculate improvements if we have baseline
    improvements = {}
    if 'baseline' in algorithm_data:
        baseline_metrics = algorithm_data['baseline']['metrics']
        
        print(f"\n📊 IMPROVEMENTS OVER BASELINE:")
        print("-" * 35)
        
        for algorithm, data in algorithm_data.items():
            if algorithm == 'baseline':
                continue
                
            algo_metrics = data['metrics']
            algo_improvements = {}
            
            print(f"\n{algorithm.upper()} vs BASELINE:")
            for metric in metrics_to_compare:
                baseline_val = baseline_metrics[metric]
                algo_val = algo_metrics[metric]
                improvement = algo_val - baseline_val
                
                algo_improvements[metric] = improvement
                
                if improvement > 0.05:  # 5% threshold
                    status = "✅"
                elif improvement < -0.05:
                    status = "❌"
                else:
                    status = "⚡"
                
                print(f"  {status} {metric.replace('_', ' ').title()}: {improvement:+.1%}")
            
            improvements[algorithm] = algo_improvements
    
    # Display enhanced criteria if available
    display_enhanced_criteria(domain_results, algorithm_names)
    
    # Generate detailed comparison
    comparison_data = {
        "algorithm_data": algorithm_data,
        "improvements": improvements,
        "best_performing": find_best_performing(algorithm_data),
        "domain_verdict": generate_domain_verdict(algorithm_data, improvements)
    }
    
    print(f"\n🎯 DOMAIN VERDICT: {comparison_data['domain_verdict']}")
    
    return comparison_data


def display_enhanced_criteria(domain_results: Dict[str, Dict], algorithm_names: List[str]) -> None:
    """Display enhanced LLM evaluation criteria if available."""
    enhanced_results = {}
    
    # Check which algorithms have enhanced evaluation
    for algorithm in algorithm_names:
        if algorithm not in domain_results:
            continue
        
        result = domain_results[algorithm]
        if 'enhanced_llm_evaluation' in result and 'criteria_metrics' in result:
            enhanced_results[algorithm] = result
    
    if not enhanced_results:
        return
    
    print(f"\n🔬 ENHANCED VALIDATION CRITERIA COMPARISON:")
    print("-" * 45)
    
    # Create criteria comparison table
    criteria_names = ['good_time_range', 'good_papers', 'good_keywords', 'good_labels']
    criteria_labels = ['⏰ Time Range', '📄 Paper Relevance', '🔖 Keyword Coherence', '🏷️ Label Match']
    
    # Header
    header = f"{'Algorithm':<15}"
    for label in criteria_labels:
        header += f"{label:<20}"
    print(header)
    print("-" * len(header))
    
    # Algorithm rows
    for algorithm in algorithm_names:
        if algorithm not in enhanced_results:
            continue
        
        result = enhanced_results[algorithm]
        criteria_metrics = result['criteria_metrics']
        
        # Get segment count from enhanced LLM evaluation
        llm_summary = result['enhanced_llm_evaluation']['summary']
        segment_count = llm_summary.get('total_segments', 0)
        
        row = f"{algorithm:<15}"
        for criteria_name in criteria_names:
            good_count = criteria_metrics.get(criteria_name, 0)
            if segment_count > 0:
                percentage = good_count / segment_count
                row += f"{good_count}/{segment_count} ({percentage:.0%})"
            else:
                row += f"{good_count}/? (N/A)"
            row += " " * (20 - len(row.split()[-1]))
        
        print(row)
    
    # Three-pillar integration status
    print()
    for algorithm in algorithm_names:
        if algorithm not in enhanced_results:
            continue
        
        result = enhanced_results[algorithm]
        if result.get('three_pillar_labels_used'):
            print(f"✅ {algorithm}: Three-pillar labels integrated for validation")
        else:
            print(f"❌ {algorithm}: No three-pillar integration available")


def find_best_performing(algorithm_data: Dict) -> Dict[str, str]:
    """Find best performing algorithm for each metric."""
    if not algorithm_data:
        return {}
    
    metrics = ['precision', 'recall', 'f1_score']
    best = {}
    
    for metric in metrics:
        best_algo = None
        best_value = -1
        
        for algorithm, data in algorithm_data.items():
            value = data['metrics'].get(metric, 0)
            if value > best_value:
                best_value = value
                best_algo = algorithm
        
        best[metric] = best_algo
    
    return best


def generate_domain_verdict(algorithm_data: Dict, improvements: Dict) -> str:
    """Generate overall verdict for domain comparison."""
    if 'baseline' not in algorithm_data:
        return "No baseline available for comparison"
    
    if not improvements:
        return "No improvements to analyze"
    
    # Count significant improvements across all non-baseline algorithms
    total_improvements = 0
    total_metrics = 0
    
    for algorithm, algo_improvements in improvements.items():
        for metric, improvement in algo_improvements.items():
            total_metrics += 1
            if improvement > 0.05:  # 5% improvement threshold
                total_improvements += 1
    
    improvement_rate = total_improvements / total_metrics if total_metrics > 0 else 0
    
    if improvement_rate >= 0.8:
        return "🎉 MAJOR IMPROVEMENT: Significant advancement over baseline"
    elif improvement_rate >= 0.6:
        return "🚀 SUBSTANTIAL IMPROVEMENT: Clear advancement over baseline"
    elif improvement_rate >= 0.4:
        return "✅ MODERATE IMPROVEMENT: Measurable progress over baseline"
    elif improvement_rate >= 0.2:
        return "⚡ MIXED RESULTS: Some improvements, some similar performance"
    else:
        return "⚠️ LIMITED IMPROVEMENT: Minimal advancement over baseline"


def generate_overall_summary(domain_comparisons: Dict) -> Dict[str, Any]:
    """Generate overall summary across all domains."""
    if not domain_comparisons:
        return {}
    
    # Count verdicts
    verdict_counts = {}
    domain_count = len(domain_comparisons)
    
    for domain, comparison in domain_comparisons.items():
        verdict = comparison.get('domain_verdict', 'Unknown')
        verdict_category = verdict.split(':')[0] if ':' in verdict else verdict
        verdict_counts[verdict_category] = verdict_counts.get(verdict_category, 0) + 1
    
    # Find most common algorithms
    all_algorithms = set()
    for comparison in domain_comparisons.values():
        all_algorithms.update(comparison.get('algorithm_data', {}).keys())
    
    return {
        "total_domains": domain_count,
        "verdict_distribution": verdict_counts,
        "algorithms_compared": sorted(all_algorithms),
        "overall_verdict": determine_overall_verdict(verdict_counts, domain_count)
    }


def determine_overall_verdict(verdict_counts: Dict, total_domains: int) -> str:
    """Determine overall verdict across all domains."""
    if not verdict_counts:
        return "No comparisons available"
    
    major_improvements = verdict_counts.get("🎉 MAJOR IMPROVEMENT", 0)
    substantial_improvements = verdict_counts.get("🚀 SUBSTANTIAL IMPROVEMENT", 0)
    moderate_improvements = verdict_counts.get("✅ MODERATE IMPROVEMENT", 0)
    
    total_improvements = major_improvements + substantial_improvements + moderate_improvements
    improvement_rate = total_improvements / total_domains
    
    if improvement_rate >= 0.8:
        return "🎉 UNIVERSAL SUCCESS: Improvements across nearly all domains"
    elif improvement_rate >= 0.6:
        return "🚀 STRONG SUCCESS: Improvements across majority of domains"
    elif improvement_rate >= 0.4:
        return "✅ MODERATE SUCCESS: Improvements in several domains"
    else:
        return "⚠️ MIXED SUCCESS: Variable performance across domains"


def print_overall_summary(summary: Dict):
    """Print the overall summary in a formatted way."""
    print(f"Total Domains Compared: {summary.get('total_domains', 0)}")
    print(f"Algorithms: {', '.join(summary.get('algorithms_compared', []))}")
    print()
    
    print("Verdict Distribution:")
    for verdict, count in summary.get('verdict_distribution', {}).items():
        print(f"  {verdict}: {count}")
    
    print()
    print(f"Overall Assessment: {summary.get('overall_verdict', 'Unknown')}")


def main():
    parser = argparse.ArgumentParser(description='Compare multiple evaluation results')
    parser.add_argument('files', nargs='+', help='Evaluation result JSON files to compare')
    parser.add_argument('--output', '-o', help='Output file for comparison report JSON')
    
    args = parser.parse_args()
    
    # Expand glob patterns
    file_paths = []
    for pattern in args.files:
        if '*' in pattern or '?' in pattern:
            file_paths.extend(glob.glob(pattern))
        else:
            file_paths.append(pattern)
    
    # Remove duplicates and sort
    file_paths = sorted(set(file_paths))
    
    print(f"🔍 Comparing {len(file_paths)} evaluation results:")
    for path in file_paths:
        print(f"  • {path}")
    print()
    
    # Load results
    results = load_evaluation_results(file_paths)
    
    if len(results) < 2:
        print("❌ Need at least 2 valid evaluation results for comparison")
        sys.exit(1)
    
    # Generate comparison
    comparison_report = compare_evaluation_results(results)
    
    # Save report if requested
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(comparison_report, f, indent=2)
        print(f"\n💾 Comparison report saved to: {args.output}")
    
    print("\n✅ Comparison completed successfully")


if __name__ == "__main__":
    main() 