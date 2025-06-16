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
        print(f"Evolution Narrative: {narrative_evolution}")
        
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


def run_domain_analysis(domain_name: str) -> bool:
    """Run complete analysis for a single domain."""
    
    print(f"\n🚀 RUNNING COMPLETE ANALYSIS: {domain_name.upper()}")
    print("=" * 60)
    
    start_time = time.time()
    
    # Step 1: Change point detection and segmentation
    segmentation_results, change_detection_result = run_change_detection(domain_name)
    if not segmentation_results:
        return False
    
    # Step 2: Timeline analysis with period characterization
    timeline_file = run_timeline_analysis(domain_name, segmentation_results, change_detection_result)
    if not timeline_file:
        return False
    
    # Step 3: Display results
    display_results_summary(domain_name, timeline_file)
    
    total_time = time.time() - start_time
    print(f"\n✅ {domain_name} analysis completed in {total_time:.2f} seconds")
    
    return True


def run_all_domains():
    """Run analysis for all available domains."""
    
    # Dynamically discover domains from resources directory
    domains = discover_available_domains()
    
    if not domains:
        print("❌ No domains found in resources directory")
        return False
    
    successful_domains = []
    
    print("🌍 RUNNING CROSS-DOMAIN ANALYSIS")
    print("=" * 50)
    print(f"Analyzing {len(domains)} domains with Timeline Analysis Framework...")
    print(f"Discovered domains: {', '.join(domains)}")
    print()
    
    for domain in domains:
        print(f"\n📊 Processing domain {len(successful_domains) + 1}/{len(domains)}: {domain}")
        success = run_domain_analysis(domain)
        if success:
            successful_domains.append(domain)
    
    # Cross-domain summary
    print(f"\n🏆 CROSS-DOMAIN ANALYSIS COMPLETE")
    print("=" * 45)
    print(f"Successfully processed: {len(successful_domains)}/{len(domains)} domains")
    
    if successful_domains:
        print("\n✅ Processed domains:")
        for domain in successful_domains:
            print(f"  • {domain}")
        
        print(f"\n📁 Results saved in 'results/' directory:")
        print(f"  • Comprehensive analysis: {len(successful_domains)} files")
        
        print(f"\n🎯 Timeline analysis with period characterization complete!")
    
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
    
    args = parser.parse_args()
    
    # Ensure results directory exists
    ensure_results_directory()
    
    print("🧪 SCIENTIFIC LITERATURE TIMELINE ANALYSIS")
    print("Enhanced Shift Signal Detection + Temporal Network Period Analysis")
    print("=" * 70)
    
    if args.domain == 'all':
        success = run_all_domains()
    else:
        available_domains = discover_available_domains()
        if args.domain not in available_domains:
            print(f"❌ Invalid domain: {args.domain}")
            print(f"Available domains: {', '.join(available_domains)}")
            return False
        
        success = run_domain_analysis(args.domain)
    
    if success:
        print(f"\n🎉 Analysis complete! Check the 'results/' directory for outputs.")
    else:
        print(f"\n❌ Analysis failed. Check error messages above.")
    
    return success


if __name__ == "__main__":
    main() 