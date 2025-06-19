#!/usr/bin/env python3
"""
Timeline Analysis Pipeline

This script runs the complete scientific literature time series segmentation pipeline,
implementing the Three-Pillar Architecture with Citation-Aware Topic Inheritance 
and Metastable Knowledge States Framework.

Usage:
    python run_timeline_analysis.py --domain deep_learning
    python run_timeline_analysis.py --domain all
    python run_timeline_analysis.py --help
"""

import argparse
import time

from core.utils import discover_available_domains, ensure_results_directory
from core.integration import run_change_detection, run_timeline_analysis
import json

def display_segmentation_summary(domain_name: str, segmentation_results: dict, change_detection_result):
    """Display a summary of the segmentation results only."""
    
    print(f"\n📊 SEGMENTATION SUMMARY: {domain_name}")
    print("=" * 40)
    
    try:
        segments = segmentation_results.get('segments', [])
        statistical_significance = change_detection_result.statistical_significance if change_detection_result else 0.0
        change_points = change_detection_result.change_points if change_detection_result else []
        
        print(f"📈 Change Points Detected: {len(change_points)}")
        print(f"📏 Timeline Segments Created: {len(segments)}")
        print(f"📊 Statistical Significance: {statistical_significance:.3f}")
        
        # Show change point years
        if change_points:
            change_years = [cp.year for cp in change_points]
            print(f"🎯 Change Point Years: {sorted(change_years)}")
        
        # Show all timeline segments
        print(f"\n📋 TIMELINE SEGMENTS:")
        for i, segment in enumerate(segments):
            start, end = segment
            length = end - start + 1
            print(f"  {i+1}. {start}-{end} ({length} years)")
        
        # Summary statistics
        if segments:
            lengths = [end - start + 1 for start, end in segments]
            avg_length = sum(lengths) / len(lengths)
            min_length = min(lengths)
            max_length = max(lengths)
            print(f"\n📈 Segment Statistics:")
            print(f"  Average length: {avg_length:.1f} years")
            print(f"  Range: {min_length}-{max_length} years")
            
    except Exception as e:
        print(f"❌ Error displaying segmentation results: {str(e)}")


def display_results_summary(domain_name: str, timeline_file: str):
    """Display a summary of the analysis results."""
    
    print(f"\n📊 RESULTS SUMMARY: {domain_name}")
    print("=" * 40)
    
    try:
        with open(timeline_file, 'r') as f:
            results = json.load(f)
        
        # Read from comprehensive analysis format
        period_characterizations = results['timeline_analysis']['final_period_characterizations']
        unified_confidence = results['timeline_analysis']['unified_confidence']
        narrative_evolution = results['timeline_analysis']['narrative_evolution']
        
        print(f"Period Characterizations: {len(period_characterizations)}")
        print(f"Unified Confidence: {unified_confidence:.3f}")
        print(f"Evolution Narrative:\n{narrative_evolution}")
        
        # Show all timeline segments
        print(f"\n📋 KEY TIMELINE PERIODS:")
        for i, period in enumerate(period_characterizations):  # Show all periods
            period_range = f"{period['period'][0]}-{period['period'][1]}"
            topic = period['topic_label']
            confidence = period['confidence']
            description = period['topic_description']
            print(f"  {i+1}. {period_range}: {topic} (confidence: {confidence:.3f}) ")
            print(f"    {description}")
        
    except Exception as e:
        print(f"❌ Error displaying results: {str(e)}")


def run_domain_analysis(domain_name: str, segmentation_only: bool = False, granularity: int = 3) -> bool:
    """Run complete analysis for a single domain with configurable granularity."""
    
    # Map granularity integer to descriptive names for logging
    granularity_names = {
        1: "ultra_coarse",
        2: "coarse", 
        3: "balanced",
        4: "fine",
        5: "ultra_fine"
    }
    
    granularity_name = granularity_names.get(granularity, f"level_{granularity}")
    
    if segmentation_only:
        print(f"\n🔍 RUNNING SEGMENTATION ONLY: {domain_name.upper()}")
        print(f"🎛️  Granularity: {granularity_name} (level {granularity})")
        print("=" * 60)
    else:
        print(f"\n🚀 RUNNING COMPLETE ANALYSIS: {domain_name.upper()}")
        print(f"🎛️  Granularity: {granularity_name} (level {granularity})")
        print("=" * 60)
    
    start_time = time.time()
    
    # Step 1: Change point detection and segmentation with granularity control
    segmentation_results, change_detection_result = run_change_detection(domain_name, granularity=granularity)
    if not segmentation_results:
        return False
    
    if segmentation_only:
        # Display segmentation results only
        display_segmentation_summary(domain_name, segmentation_results, change_detection_result)
        total_time = time.time() - start_time
        print(f"\n✅ {domain_name} segmentation completed in {total_time:.2f} seconds")
        return True
    
    # Step 2: Timeline analysis with period characterization (full analysis only)
    timeline_file = run_timeline_analysis(domain_name, segmentation_results, change_detection_result)
    if not timeline_file:
        return False
    
    # Step 3: Display results
    display_results_summary(domain_name, timeline_file)
    
    total_time = time.time() - start_time
    print(f"\n✅ {domain_name} analysis completed in {total_time:.2f} seconds")
    
    return True


def run_all_domains(segmentation_only: bool = False, granularity: int = 3):
    """Run analysis for all available domains with configurable granularity."""
    
    # Dynamically discover domains from resources directory
    domains = discover_available_domains()
    
    if not domains:
        print("❌ No domains found in resources directory")
        return False
    
    successful_domains = []
    
    analysis_type = "SEGMENTATION" if segmentation_only else "ANALYSIS"
    print(f"🌍 RUNNING CROSS-DOMAIN {analysis_type}")
    print(f"🎛️  Granularity: {granularity}")
    print("=" * 50)
    print(f"Analyzing {len(domains)} domains with Timeline Analysis Framework...")
    print(f"Discovered domains: {', '.join(domains)}")
    print()
    
    for domain in domains:
        print(f"\n📊 Processing domain {len(successful_domains) + 1}/{len(domains)}: {domain}")
        success = run_domain_analysis(domain, segmentation_only, granularity)
        if success:
            successful_domains.append(domain)
    
    # Cross-domain summary
    print(f"\n🏆 CROSS-DOMAIN {analysis_type} COMPLETE")
    print("=" * 45)
    print(f"Successfully processed: {len(successful_domains)}/{len(domains)} domains")
    
    if successful_domains:
        print("\n✅ Processed domains:")
        for domain in successful_domains:
            print(f"  • {domain}")
        
        if not segmentation_only:
            print(f"\n📁 Results saved in 'results/' directory:")
            print(f"  • Comprehensive analysis: {len(successful_domains)} files")
            print(f"\n🎯 Timeline analysis with period characterization complete!")
        else:
            print(f"\n🔍 Segmentation testing complete!")
    
    if len(successful_domains) < len(domains):
        failed_domains = [d for d in domains if d not in successful_domains]
        print(f"\n❌ Failed domains:")
        for domain in failed_domains:
            print(f"  • {domain}")
    
    return len(successful_domains) > 0


def main():
    """Main pipeline execution."""
    
    parser = argparse.ArgumentParser(
        description="Scientific Literature Timeline Analysis Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_timeline_analysis.py --domain deep_learning
  python run_timeline_analysis.py --domain applied_mathematics  
  python run_timeline_analysis.py --domain art
  python run_timeline_analysis.py --domain natural_language_processing
  python run_timeline_analysis.py --domain all

Available domains are automatically discovered from the resources/ directory.
        """
    )
    
    parser.add_argument(
        '--domain', 
        type=str, 
        required=True,
        help='Domain to analyze (use "all" to process all domains, or specify a domain name from resources/ directory)'
    )
    
    parser.add_argument(
        '--segmentation-only',
        action='store_true',
        help='Run only segmentation analysis (skip timeline characterization for faster testing)'
    )
    
    parser.add_argument(
        '--granularity',
        type=int,
        default=3,
        choices=[1, 2, 3, 4, 5],
        help='Timeline granularity control: 1 (ultra_coarse, fewest segments), 2 (coarse), 3 (balanced, default), 4 (fine), 5 (ultra_fine, most segments)'
    )
    
    args = parser.parse_args()
    
    # Ensure results directory exists
    ensure_results_directory()
    
    print("🧪 SCIENTIFIC LITERATURE TIMELINE ANALYSIS")
    print("Enhanced Shift Signal Detection + Temporal Network Period Analysis")
    print("=" * 70)
    
    if args.domain == 'all':
        success = run_all_domains(args.segmentation_only, args.granularity)
    else:
        available_domains = discover_available_domains()
        if args.domain not in available_domains:
            print(f"❌ Invalid domain: {args.domain}")
            print(f"Available domains: {', '.join(available_domains)}")
            return False
        
        success = run_domain_analysis(args.domain, args.segmentation_only, args.granularity)
    
    if success:
        print(f"\n🎉 Analysis complete! Check the 'results/' directory for outputs.")
    else:
        print(f"\n❌ Analysis failed. Check error messages above.")
    
    return success


if __name__ == "__main__":
    main() 