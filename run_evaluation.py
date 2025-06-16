#!/usr/bin/env python3
"""
Generic Evaluation Script

This script evaluates any comprehensive analysis results file against the validation framework.
Supports both standard and LLM-enhanced evaluation.

Usage:
    python run_evaluation.py results/deep_learning_comprehensive_analysis.json
    python run_evaluation.py results/deep_learning_comprehensive_analysis.json --no-llm
    python run_evaluation.py results/art_comprehensive_analysis.json --domain art
    python run_evaluation.py --all
"""

import json
import os
import sys
from pathlib import Path
import argparse
import glob
from typing import List

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent))

from validation.sanity_metrics import run_sanity_checks
from validation.manual_evaluation import run_evaluation as run_manual_evaluation, run_comprehensive_evaluation
from validation.llm_judge import run_llm_evaluation, evaluate_with_ensemble
import pandas as pd


def discover_available_domains() -> List[str]:
    """
    Automatically discover available domains from the resources directory.
    
    Returns:
        List of domain names found in resources directory
    """
    resources_path = Path("resources")
    
    if not resources_path.exists():
        print("❌ Resources directory not found")
        return []
    
    domains = []
    for item in resources_path.iterdir():
        # Only include directories and exclude system files
        if item.is_dir() and not item.name.startswith('.'):
            domains.append(item.name)
    
    return sorted(domains)


def load_comprehensive_analysis(file_path: str) -> dict:
    """
    Load comprehensive analysis results from any file.
    
    Args:
        file_path: Path to comprehensive analysis results file
        
    Returns:
        Dictionary with comprehensive analysis results
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Results file not found: {file_path}")
    
    with open(file_path, 'r') as f:
        return json.load(f)


def extract_domain_from_results(results: dict, file_path: str) -> str:
    """Extract domain name from results or file path."""
    # Try to get domain from results metadata
    if 'analysis_metadata' in results and 'domain_name' in results['analysis_metadata']:
        return results['analysis_metadata']['domain_name']
    
    # Try to extract from file path
    filename = os.path.basename(file_path)
    if '_comprehensive_analysis.json' in filename:
        return filename.replace('_comprehensive_analysis.json', '')
    
    # Fallback
    return 'unknown'


def get_algorithm_name(results: dict, file_path: str) -> str:
    """Get algorithm name from results or file path."""
    if ('segmentation_results' in results and 
        'method_details' in results['segmentation_results'] and 
        'algorithm' in results['segmentation_results']['method_details']):
        return results['segmentation_results']['method_details']['algorithm']
    
    # Default to comprehensive since these are comprehensive analysis results
    return 'comprehensive'


def extract_segmentation_data(comprehensive_results: dict) -> dict:
    """
    Extract segmentation data from comprehensive analysis results.
    
    Args:
        comprehensive_results: Full comprehensive analysis results
        
    Returns:
        Dictionary with segmentation results in the expected format
    """
    if 'segmentation_results' not in comprehensive_results:
        raise ValueError("No segmentation_results found in comprehensive analysis")
    
    segmentation_data = comprehensive_results['segmentation_results'].copy()
    
    # Ensure compatibility with existing evaluation code
    if 'method_details' not in segmentation_data:
        segmentation_data['method_details'] = {}
    
    return segmentation_data


def extract_three_pillar_data(comprehensive_results: dict) -> dict:
    """
    Extract three-pillar data from comprehensive analysis results.
    
    Args:
        comprehensive_results: Full comprehensive analysis results
        
    Returns:
        Dictionary with three-pillar results in the expected format, or None if not available
    """
    if 'timeline_analysis' not in comprehensive_results:
        return None
    
    timeline_analysis = comprehensive_results['timeline_analysis']
    
    # Convert to the format expected by the LLM evaluation
    three_pillar_data = {
        'metastable_states': timeline_analysis.get('metastable_states', []),
        'state_transitions': timeline_analysis.get('state_transitions', []),
        'pillar_contributions': timeline_analysis.get('pillar_contributions', {}),
        'unified_confidence': timeline_analysis.get('unified_confidence', 0.0),
        'narrative_evolution': timeline_analysis.get('narrative_evolution', '')
    }
    
    return three_pillar_data


def find_all_comprehensive_analysis_results() -> List[str]:
    """
    Find all comprehensive analysis results files in the results directory.
    
    Returns:
        List of paths to comprehensive analysis results files
    """
    pattern = "results/*_comprehensive_analysis.json"
    files = glob.glob(pattern)
    return sorted(files)


def run_all_evaluations(use_llm_judge: bool = True) -> bool:
    """
    Run evaluation for all available comprehensive analysis results files.
    
    Args:
        use_llm_judge: Whether to use enhanced LLM-as-a-judge evaluation
        
    Returns:
        True if all evaluations successful, False if any failed
    """
    comprehensive_files = find_all_comprehensive_analysis_results()
    
    if not comprehensive_files:
        print("❌ No comprehensive analysis results files found in results/ directory")
        
        # Show available domains for helpful guidance
        available_domains = discover_available_domains()
        if available_domains:
            print(f"📁 Available domains in resources/: {', '.join(available_domains)}")
            print("Run timeline analysis first: python run_timeline_analysis.py --domain all")
        else:
            print("❌ No domains found in resources/ directory")
        return False
    
    print("🌍 RUNNING EVALUATION ON ALL AVAILABLE COMPREHENSIVE ANALYSIS RESULTS")
    print("=" * 70)
    print(f"Found {len(comprehensive_files)} comprehensive analysis results files:")
    for file_path in comprehensive_files:
        print(f"  • {os.path.basename(file_path)}")
    print()
    
    successful_evaluations = []
    failed_evaluations = []
    
    for file_path in comprehensive_files:
        print(f"\n{'='*80}")
        print(f"EVALUATING: {os.path.basename(file_path)}")
        print(f"{'='*80}")
        
        try:
            success = run_evaluation_on_file(file_path, domain=None, use_llm_judge=use_llm_judge)
            if success:
                successful_evaluations.append(file_path)
            else:
                failed_evaluations.append(file_path)
        except Exception as e:
            print(f"❌ Error evaluating {file_path}: {e}")
            failed_evaluations.append(file_path)
    
    # Final summary
    print(f"\n🏆 ALL EVALUATIONS COMPLETE")
    print("=" * 45)
    print(f"Successfully evaluated: {len(successful_evaluations)}/{len(comprehensive_files)} files")
    
    if successful_evaluations:
        print("\n✅ Successful evaluations:")
        for file_path in successful_evaluations:
            domain = extract_domain_from_file_path(file_path)
            print(f"  • {domain}")
    
    if failed_evaluations:
        print("\n❌ Failed evaluations:")
        for file_path in failed_evaluations:
            domain = extract_domain_from_file_path(file_path)
            print(f"  • {domain}")
    
    print(f"\n📁 Evaluation results saved in 'validation/' directory")
    
    return len(failed_evaluations) == 0


def extract_domain_from_file_path(file_path: str) -> str:
    """Extract domain name from file path for display purposes."""
    filename = os.path.basename(file_path)
    return filename.replace('_comprehensive_analysis.json', '')


def run_evaluation_on_file(file_path: str, domain: str = None, use_llm_judge: bool = True) -> bool:
    """
    Run complete evaluation framework on a comprehensive analysis results file.
    
    Args:
        file_path: Path to comprehensive analysis results file
        domain: Domain name (auto-detected if not provided)
        use_llm_judge: Whether to use enhanced LLM-as-a-judge evaluation
        
    Returns:
        True if evaluation successful, False if failed at sanity checks
    """
    print("=" * 80)
    print(f"COMPREHENSIVE ANALYSIS EVALUATION: {os.path.basename(file_path)}")
    print("=" * 80)
    print()
    
    # Load results
    print(f"Loading comprehensive analysis results from {file_path}...")
    try:
        comprehensive_results = load_comprehensive_analysis(file_path)
    except Exception as e:
        print(f"❌ Error loading results: {e}")
        return False
    
    # Extract components from comprehensive results
    try:
        segmentation_data = extract_segmentation_data(comprehensive_results)
        three_pillar_data = extract_three_pillar_data(comprehensive_results)
    except Exception as e:
        print(f"❌ Error extracting data from comprehensive results: {e}")
        return False
    
    # Extract domain and algorithm info
    if domain is None:
        domain = extract_domain_from_results(comprehensive_results, file_path)
    
    algorithm_name = get_algorithm_name(comprehensive_results, file_path)
    
    # Load domain data using unified data loader
    try:
        from core.data_loader import load_domain_data
        data_df = load_domain_data(domain, prefer_source="csv")
    except FileNotFoundError:
        print(f"❌ Domain data not found for domain: {domain}")
        print("Available domains can be found in resources/ or data/processed/ directories")
        return False
    
    # Extract metadata for display
    metadata = comprehensive_results.get('analysis_metadata', {})
    year_range = metadata.get('time_range', [data_df['year'].min(), data_df['year'].max()])
    
    print(f"✅ Loaded: {len(data_df)} papers from {year_range[0]}-{year_range[1]}")
    print(f"📊 Algorithm: {algorithm_name}")
    print(f"🎯 Domain: {domain}")
    print(f"📈 Detected: {len(segmentation_data['change_points'])} change points")
    print(f"📋 Segments: {len(segmentation_data['segments'])}")
    print(f"📊 Statistical significance: {segmentation_data['statistical_significance']:.3f}")
    if three_pillar_data:
        print(f"🏛️  Three-pillar analysis: Available")
        print(f"📈 Unified confidence: {three_pillar_data['unified_confidence']:.3f}")
    else:
        print(f"🏛️  Three-pillar analysis: Not available")
    print()
    
    # TIER 1: Automated Sanity Checks
    print("TIER 1: AUTOMATED SANITY CHECKS")
    print("=" * 40)
    print("These checks must pass before proceeding to manual evaluation.")
    print("Passing these does NOT indicate success - they are minimum requirements.")
    print()
    
    # Convert segments to the format expected by sanity checks
    segments = segmentation_data['segments']
    
    # Run sanity checks
    sanity_results = run_sanity_checks(data_df, segments)
    
    # Print detailed sanity check report
    print("SANITY CHECK REPORT")
    print("=" * 40)
    print(f"Total Segments Detected: {len(segments)}")
    print(f"Overall Pass: {'✓ PASS' if sanity_results['overall_pass'] else '✗ FAIL'}")
    print()
    
    print("METRICS:")
    print("-" * 40)
    thresholds = sanity_results['thresholds']
    for metric, value in sanity_results.items():
        if metric in thresholds:
            threshold = thresholds[metric]
            status = "✓" if value >= threshold else "✗"
            print(f"{metric:20}: {value:6.3f} (>= {threshold:.3f}) {status}")
    
    print()
    if not sanity_results['overall_pass']:
        print("FAILURES:")
        print("-" * 40)
        for metric, value in sanity_results.items():
            if metric in thresholds and value < thresholds[metric]:
                print(f"  • {metric}: {value:.3f} < {thresholds[metric]:.3f}")
        print()
        print(f"❌ TIER 1 FAILED: {algorithm_name} algorithm has fundamental issues")
        if use_llm_judge:
            print("⚠️  Proceeding with LLM evaluation despite sanity check failures for comprehensive assessment")
        else:
            print("Cannot proceed to manual evaluation until sanity checks pass.")
            return False
    else:
        print(f"✅ TIER 1 PASSED: {algorithm_name} algorithm meets minimum requirements")
        print()
    
    # TIER 2+: Manual Evaluation Against Ground Truth
    # Always run enhanced LLM evaluation if requested, regardless of sanity check results
    if use_llm_judge:
        print("TIER 2-4: ENHANCED LLM EVALUATION WITH CONCRETE VALIDATION CRITERIA")
        print("=" * 70)
        print("Multi-tier evaluation combining ground truth recall with enhanced LLM validation.")
        print("• Tier 2: Ground truth paradigm shift detection")
        print("• Tier 3: Enhanced LLM evaluation with concrete validation criteria")
        print("• Tier 4: Three-pillar integration with algorithm label validation")
        if not sanity_results['overall_pass']:
            print("• NOTE: Running despite sanity check failures for comprehensive assessment")
        print()
        
        data_path = f"data/processed/{domain}_processed.csv"
        gt_path = f"validation/{domain}_groundtruth.json"
        
        print("🤖 Using ensemble enhanced LLM evaluation with concrete criteria:")
        print("   • Time range sensibility (duration, boundaries)")
        print("   • Paper relevance and coherence")
        print("   • Keyword coherence and theme consistency")
        print("   • Algorithm label validation (comprehensive analysis integration)")
        print("   • Ensemble: llama3.1:8b, llama3.2:3b, gemma3:4b")
        print("   • Majority voting for robust consensus")
        print()
        
        # Run standard ground truth evaluation first
        print("Running ground truth evaluation...")
        gt_evaluation = run_manual_evaluation(segmentation_data, data_path, gt_path)
        
        # Run ensemble enhanced LLM evaluation
        print("Running ensemble enhanced LLM evaluation...")
        algorithm_segments = [tuple(seg) for seg in segmentation_data['segments']]
        
        llm_evaluation = evaluate_with_ensemble(
            algorithm_segments=algorithm_segments,
            data_df=data_df,
            domain=domain,
            models=["llama3.1:8b", "llama3.2:3b", "gemma3:4b"],
            three_pillar_data=three_pillar_data  # Pass the three-pillar data directly
        )
        
        # Combine evaluations
        evaluation_report = {
            'sanity_check_passed': sanity_results['overall_pass'],
            'recall_evaluation': gt_evaluation,
            'enhanced_llm_evaluation': llm_evaluation,
            'enhanced_precision': llm_evaluation['summary']['precision'],
            'criteria_metrics': llm_evaluation['summary']['criteria_metrics'],
            'three_pillar_labels_used': llm_evaluation['summary']['three_pillar_labels_used']
        }
    else:
        # Only run manual evaluation if sanity checks passed
        if not sanity_results['overall_pass']:
            print("❌ Cannot proceed to manual evaluation - sanity checks failed.")
            return False
            
        print("TIER 2: MANUAL EVALUATION AGAINST HIGH-PRECISION GROUND TRUTH")
        print("=" * 60)
        print("Comparing algorithm detections against research-backed paradigm shifts.")
        print("Ground truth has 100% precision but low recall (only major shifts included).")
        print()
        
        data_path = f"data/processed/{domain}_processed.csv"
        gt_path = f"validation/{domain}_groundtruth.json"
        evaluation_report = run_manual_evaluation(segmentation_data, data_path, gt_path)
        evaluation_report['sanity_check_passed'] = sanity_results['overall_pass']
    
    # Assessment and Recommendations
    tier_label = "FINAL ASSESSMENT" if use_llm_judge else "TIER 3: COMPREHENSIVE ASSESSMENT"
    print(f"\n{tier_label}")
    print("=" * 40)
    
    # Handle different report structures for LLM vs standard evaluation
    if use_llm_judge and 'recall_evaluation' in evaluation_report:
        # Enhanced evaluation with LLM - metrics are nested
        metrics = evaluation_report['recall_evaluation']['metrics']
        matching = evaluation_report['recall_evaluation']['matching_results']
    else:
        # Standard evaluation - metrics are at top level
        metrics = evaluation_report['metrics']
        matching = evaluation_report['matching_results']
    
    print(f"{algorithm_name.title()} Algorithm Performance:")
    if use_llm_judge and 'enhanced_llm_evaluation' in evaluation_report:
        # Show both standard and enhanced LLM metrics
        enhanced_precision = evaluation_report['enhanced_precision']
        criteria_metrics = evaluation_report['criteria_metrics']
        
        print(f"  Ground Truth Precision: {metrics['precision']:.1%}")
        print(f"  Ground Truth Recall:    {metrics['recall']:.1%}")
        print(f"  Enhanced LLM Precision: {enhanced_precision:.1%}")
        print(f"  F1 Score (GT):          {metrics['f1_score']:.3f}")
        print()
        print("  Enhanced Validation Criteria:")
        print(f"    ⏰ Good Time Ranges:    {criteria_metrics['good_time_range']}/{len(segmentation_data['segments'])} ({criteria_metrics['good_time_range']/len(segmentation_data['segments']):.1%})")
        print(f"    📄 Good Paper Relevance: {criteria_metrics['good_papers']}/{len(segmentation_data['segments'])} ({criteria_metrics['good_papers']/len(segmentation_data['segments']):.1%})")
        print(f"    🔖 Good Keyword Coherence: {criteria_metrics['good_keywords']}/{len(segmentation_data['segments'])} ({criteria_metrics['good_keywords']/len(segmentation_data['segments']):.1%})")
        if evaluation_report['three_pillar_labels_used']:
            print(f"    🏷️  Good Label Matches:  {criteria_metrics['good_labels']}/{len(segmentation_data['segments'])} ({criteria_metrics['good_labels']/len(segmentation_data['segments']):.1%})")
    else:
        # Standard metrics only
        print(f"  Precision: {metrics['precision']:.1%} ({metrics['true_positives']}/{metrics['true_positives'] + metrics['false_positives']})")
        print(f"  Recall:    {metrics['recall']:.1%} ({metrics['true_positives']}/{metrics['true_positives'] + metrics['false_negatives']})")
        print(f"  F1 Score:  {metrics['f1_score']:.3f}")
    print()
    
    # Conservative assessment using appropriate metrics
    if use_llm_judge and 'enhanced_llm_evaluation' in evaluation_report:
        # Use enhanced LLM precision with ground truth recall
        precision_score = evaluation_report['enhanced_precision']
        recall_score = metrics['recall']  # Still use ground truth recall
        f1_score = metrics['f1_score']  # Use ground truth F1
    else:
        # Use standard metrics
        precision_score = metrics['precision']
        recall_score = metrics['recall']
        f1_score = metrics['f1_score']
    
    # Critical assessment (adjust for sanity check failures)
    print("CRITICAL ASSESSMENT:")
    print("-" * 25)
    
    sanity_passed = evaluation_report.get('sanity_check_passed', True)
    
    if not sanity_passed:
        assessment = f"❌ FUNDAMENTAL ISSUES: {algorithm_name} failed basic sanity checks"
        confidence = "Very low confidence - algorithm has fundamental problems"
    elif precision_score >= 0.85 and recall_score >= 0.85:
        assessment = f"✅ EXCELLENT: {algorithm_name} meets high performance standards"
        confidence = "High confidence in algorithm quality"
    elif precision_score >= 0.67 and recall_score >= 0.67:
        assessment = f"✅ GOOD: {algorithm_name} shows acceptable performance"
        confidence = "Moderate confidence in algorithm quality"
    elif precision_score >= 0.50 or recall_score >= 0.67:
        assessment = f"⚠️ LIMITED: {algorithm_name} has partial success with limitations"
        confidence = "Low confidence, needs improvement"
    else:
        assessment = f"❌ POOR: {algorithm_name} does not meet minimum thresholds"
        confidence = "Very low confidence, requires major improvements"
    
    print(f"Assessment: {assessment}")
    print(f"Confidence: {confidence}")
    print()
    
    # Detailed analysis
    print("DETAILED FINDINGS:")
    print("-" * 20)
    
    if matching['matches']:
        print(f"✅ Successfully detected {len(matching['matches'])} ground truth paradigm shifts:")
        for match in matching['matches']:
            gt_shift = match['ground_truth_shift']
            algo_seg = match['algorithm_segment']
            overlap = match['overlap_ratio']
            print(f"   • {gt_shift['name']}: {gt_shift['start_year']}-{gt_shift['end_year']} "
                  f"vs {algo_seg[0]}-{algo_seg[1]} (overlap: {overlap:.1%})")
    
    if matching['unmatched_ground_truth']:
        print(f"❌ Missed {len(matching['unmatched_ground_truth'])} known paradigm shifts:")
        for gt_shift in matching['unmatched_ground_truth']:
            print(f"   • {gt_shift['name']}: {gt_shift['start_year']}-{gt_shift['end_year']}")
    
    if matching['unmatched_algorithm']:
        print(f"🔍 Algorithm detected {len(matching['unmatched_algorithm'])} additional periods:")
        for idx, (seg_idx, segment) in enumerate(matching['unmatched_algorithm']):
            print(f"   • Period {seg_idx + 1}: {segment[0]}-{segment[1]}")
    
    print()
    
    # Save evaluation results
    output_filename = os.path.basename(file_path).replace('_comprehensive_analysis.json', '_evaluation_results.json')
    output_path = f"validation/{output_filename}"
    
    # Add metadata to evaluation report
    evaluation_report['input_file'] = file_path
    evaluation_report['domain'] = domain
    evaluation_report['algorithm'] = algorithm_name
    evaluation_report['assessment'] = assessment
    evaluation_report['comprehensive_analysis_used'] = True
    
    # Convert any tuple keys to strings for JSON serialization
    def convert_tuples_to_strings(obj):
        if isinstance(obj, dict):
            return {str(key) if isinstance(key, tuple) else key: convert_tuples_to_strings(value) 
                    for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_tuples_to_strings(item) for item in obj]
        else:
            return obj
    
    serializable_report = convert_tuples_to_strings(evaluation_report)
    
    os.makedirs("validation", exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(serializable_report, f, indent=2)
    
    print("=" * 80)
    evaluation_type = "ENSEMBLE ENHANCED LLM EVALUATION" if use_llm_judge else "STANDARD EVALUATION"
    print(f"{evaluation_type} COMPLETED")
    print(f"Input:  {file_path}")
    print(f"Output: {output_path}")
    if use_llm_judge:
        print("📊 Enhanced evaluation includes concrete validation criteria assessment")
        print("🤖 Requires Ollama server with ensemble models: llama3.1:8b, llama3.2:3b, gemma3:4b")
        print("🗳️  Uses majority voting across multiple models for robust consensus")
        print("🏛️  Features three-pillar integration with algorithm label validation")
        if evaluation_report.get('three_pillar_labels_used'):
            print("✅ Three-pillar results successfully integrated for label validation")
        else:
            print("ℹ️  Three-pillar labels not available for validation")
        
        # Display ensemble statistics if available
        if 'enhanced_llm_evaluation' in evaluation_report:
            llm_summary = evaluation_report['enhanced_llm_evaluation']['summary']
            if 'models_successful' in llm_summary:
                models_successful = llm_summary['models_successful']
                models_attempted = llm_summary.get('models_attempted', len(["llama3.1:8b", "llama3.2:3b", "gemma3:4b"]))
                print(f"🔧 Ensemble performance: {models_successful}/{models_attempted} models successful")
        
        if not sanity_passed:
            print("⚠️  Note: Evaluation completed despite sanity check failures for comprehensive assessment")
    print("🏗️  Used comprehensive analysis format for streamlined evaluation")
    print("=" * 80)
    
    # Return True for successful evaluation (even if sanity checks failed when using LLM)
    return True if use_llm_judge else sanity_passed


def main():
    parser = argparse.ArgumentParser(description='Evaluate comprehensive analysis results against validation framework')
    
    # Make input_file optional when using --all
    parser.add_argument('input_file', nargs='?', help='Path to comprehensive analysis results JSON file')
    parser.add_argument('--domain', help='Domain name (auto-detected if not provided)')
    parser.add_argument('--no-llm', action='store_true', help='Disable LLM-enhanced evaluation')
    parser.add_argument('--all', action='store_true', help='Evaluate all available comprehensive analysis results')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.all and args.input_file:
        print("❌ Cannot specify both --all and input_file")
        parser.print_help()
        sys.exit(1)
    
    if not args.all and not args.input_file:
        print("❌ Must specify either input_file or --all")
        parser.print_help()
        sys.exit(1)
    
    use_llm = not args.no_llm
    
    if args.all:
        print("📊 Evaluating all available comprehensive analysis results")
        if use_llm:
            print("🤖 Using ensemble enhanced LLM evaluation with concrete validation criteria")
            print("🗳️  Majority voting across llama3.1:8b, llama3.2:3b, gemma3:4b")
        else:
            print("📈 Using standard evaluation (no LLM)")
        print()
        
        success = run_all_evaluations(use_llm)
    else:
        print(f"📊 Evaluating: {args.input_file}")
        if use_llm:
            print("🤖 Using ensemble enhanced LLM evaluation with concrete validation criteria")
            print("🗳️  Majority voting across llama3.1:8b, llama3.2:3b, gemma3:4b")
        else:
            print("📈 Using standard evaluation (no LLM)")
        print()
        
        success = run_evaluation_on_file(args.input_file, args.domain, use_llm)
    
    if success:
        print("✅ Evaluation completed successfully")
    else:
        print("❌ Evaluation failed")
        sys.exit(1)


if __name__ == "__main__":
    main() 