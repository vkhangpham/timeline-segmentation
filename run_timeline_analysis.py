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
from core.algorithm_config import ComprehensiveAlgorithmConfig
from core.integration import run_change_detection, run_timeline_analysis
import json
import os

def display_segmentation_summary(domain_name: str, segmentation_results: dict, change_detection_result):
    """Display a summary of the segmentation results only."""
    
    print(f"\nSEGMENTATION SUMMARY: {domain_name}")
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


def load_optimized_parameters_if_available(domain_name: str, params_file: str = None) -> dict:
    """Load optimized parameters if available."""
    optimized_params_file = params_file or "results/optimized_parameters.json"
    
    if os.path.exists(optimized_params_file):
        try:
            with open(optimized_params_file, 'r') as f:
                data = json.load(f)
            
            optimized_params = data.get('consensus_difference_optimized_parameters', {})
            if domain_name in optimized_params:
                print(f"📁 Using optimized parameters for {domain_name}: {optimized_params[domain_name]}")
                print(f"📄 Loaded from: {optimized_params_file}")
                return optimized_params[domain_name]
            else:
                print(f"⚠️ No optimized parameters found for {domain_name} in {optimized_params_file}, using defaults")
                return {}
        except Exception as e:
            print(f"❌ Error loading optimized parameters from {optimized_params_file}: {e}")
            return {}
    else:
        print(f"⚠️ No optimized parameters file found at {optimized_params_file}, using defaults")
        return {}


def run_domain_analysis(domain_name: str, segmentation_only: bool = False, granularity: int = 3, algorithm_config: ComprehensiveAlgorithmConfig = None, optimized_params_file: str = None) -> bool:
    """Run complete analysis for a single domain with comprehensive algorithm configuration."""
    start_time = time.time()
    
    # Load optimized parameters if available
    optimized_params = load_optimized_parameters_if_available(domain_name, optimized_params_file)
    
    # Update algorithm config with optimized parameters if available
    if optimized_params and algorithm_config:
        # Get all current config values as a base
        config_kwargs = {
            'granularity': algorithm_config.granularity,
            'domain_name': domain_name,
            'direction_threshold': algorithm_config.direction_threshold,
            'validation_threshold': algorithm_config.validation_threshold,
            'keyword_min_frequency': algorithm_config.keyword_min_frequency,
            'min_significant_keywords': algorithm_config.min_significant_keywords,
            'keyword_filtering_enabled': algorithm_config.keyword_filtering_enabled,
            'keyword_min_papers_ratio': algorithm_config.keyword_min_papers_ratio,
            'citation_boost': algorithm_config.citation_boost,
            'citation_support_window': algorithm_config.citation_support_window,
            'similarity_min_segment_length': algorithm_config.similarity_min_segment_length,
            'similarity_max_segment_length': algorithm_config.similarity_max_segment_length,
        }
        
        # Dynamically override with any available optimized parameters
        for param_name, param_value in optimized_params.items():
            if param_name in config_kwargs:
                config_kwargs[param_name] = param_value
                print(f"   🔧 Using optimized {param_name}: {param_value}")
        
        algorithm_config = ComprehensiveAlgorithmConfig(**config_kwargs)
        print(f"🎯 Using optimized algorithm config: dir={algorithm_config.direction_threshold:.3f}, val={algorithm_config.validation_threshold:.3f}, sim_length={algorithm_config.similarity_min_segment_length}-{algorithm_config.similarity_max_segment_length}")
    
    # Step 1: Change point detection and segmentation with comprehensive configuration
    segmentation_results, change_detection_result = run_change_detection(
        domain_name, 
        granularity=algorithm_config.granularity, 
        algorithm_config=algorithm_config
    )
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


def run_all_domains(segmentation_only: bool = False, granularity: int = 3, algorithm_config: ComprehensiveAlgorithmConfig = None, optimized_params_file: str = None):
    """Run analysis for all available domains with comprehensive algorithm configuration."""
    
    # Create comprehensive configuration (will be overridden per domain)
    if algorithm_config is None:
        algorithm_config = ComprehensiveAlgorithmConfig(granularity=granularity)
    
    # Dynamically discover domains from resources directory
    domains = discover_available_domains()
    
    if not domains:
        print("❌ No domains found in resources directory")
        return False
    
    successful_domains = []
    
    analysis_type = "SEGMENTATION" if segmentation_only else "ANALYSIS"
    print(f"🌍 RUNNING CROSS-DOMAIN {analysis_type}")
    print(f"🎛️  Comprehensive Configuration: Granularity {algorithm_config.granularity}")
    print(f"🔧  Algorithm Parameters: {len([f for f in algorithm_config.__dataclass_fields__])} total parameters")
    print("=" * 50)
    print(f"Analyzing {len(domains)} domains with Enhanced Timeline Analysis Framework...")
    print(f"Discovered domains: {', '.join(domains)}")
    print()
    
    for domain in domains:
        print(f"\n📊 Processing domain {len(successful_domains) + 1}/{len(domains)}: {domain}")
        # Create domain-specific configuration for optimization
        domain_config = ComprehensiveAlgorithmConfig(granularity=granularity, domain_name=domain)
        success = run_domain_analysis(domain, segmentation_only, granularity, domain_config, optimized_params_file)
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
            print(f"🔧 Used comprehensive configuration with {len([f for f in algorithm_config.__dataclass_fields__])} parameters")
        else:
            print(f"\n🔍 Segmentation testing complete!")
    
    if len(successful_domains) < len(domains):
        failed_domains = [d for d in domains if d not in successful_domains]
        print(f"\n❌ Failed domains:")
        for domain in failed_domains:
            print(f"  • {domain}")
    
    return len(successful_domains) > 0


def main():
    """Main pipeline execution with comprehensive algorithm configuration."""
    
    parser = argparse.ArgumentParser(
        description="Scientific Literature Timeline Analysis Pipeline with Comprehensive Algorithm Configuration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_timeline_analysis.py --domain deep_learning
  python run_timeline_analysis.py --domain applied_mathematics  
  python run_timeline_analysis.py --domain art
  python run_timeline_analysis.py --domain natural_language_processing
  python run_timeline_analysis.py --domain all
  python run_timeline_analysis.py --domain computer_vision --granularity 1
  python run_timeline_analysis.py --domain all --granularity 5
  python run_timeline_analysis.py --domain deep_learning --optimized-params-file results/my_custom_params.json
  python run_timeline_analysis.py --domain all --optimized-params-file results/optimized_parameters_bayesian.json

Available domains are automatically discovered from the resources/ directory.
Comprehensive configuration provides access to 12 core algorithm parameters.
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
        help='Timeline granularity control: 1 (ultra_fine, most segments), 2 (fine), 3 (balanced, default), 4 (coarse), 5 (ultra_coarse, fewest segments)'
    )
    
    # Advanced configuration options
    parser.add_argument(
        '--direction-threshold',
        type=float,
        default=None,
        help='Override direction detection threshold (0.1-0.8, lower = more sensitive)'
    )
    
    parser.add_argument(
        '--validation-threshold', 
        type=float,
        default=None,
        help='Override validation threshold (0.5-0.95, higher = more stringent)'
    )
    
    parser.add_argument(
        '--citation-boost',
        type=float,
        default=None,
        help='Override citation support boost (0.0-1.0)'
    )
    
    parser.add_argument(
        '--optimized-params-file',
        type=str,
        default=None,
        help='Path to optimized parameters JSON file (default: results/optimized_parameters.json)'
    )
    

    
    args = parser.parse_args()
    
    # Create comprehensive algorithm configuration
    overrides = {}
    if args.direction_threshold is not None:
        overrides['direction_threshold'] = args.direction_threshold
    if args.validation_threshold is not None:
        overrides['validation_threshold'] = args.validation_threshold  
    if args.citation_boost is not None:
        overrides['citation_boost'] = args.citation_boost
    
    # Determine domain name for configuration
    domain_for_config = args.domain if args.domain != 'all' else None
    
    if overrides:
        print(f"🔧 Using custom parameter overrides: {overrides}")
        algorithm_config = ComprehensiveAlgorithmConfig.create_custom(
            granularity=args.granularity,
            domain_name=domain_for_config,
            overrides=overrides
        )
    else:
        algorithm_config = ComprehensiveAlgorithmConfig(granularity=args.granularity, domain_name=domain_for_config)
    
    # Ensure results directory exists
    ensure_results_directory()
    
    print(f"Active Configuration: {algorithm_config.get_configuration_summary()}")
    
    if args.domain == 'all':
        success = run_all_domains(args.segmentation_only, args.granularity, algorithm_config, args.optimized_params_file)
    else:
        available_domains = discover_available_domains()
        if args.domain not in available_domains:
            print(f"❌ Invalid domain: {args.domain}")
            print(f"Available domains: {', '.join(available_domains)}")
            return False
        
        success = run_domain_analysis(args.domain, args.segmentation_only, args.granularity, algorithm_config, args.optimized_params_file)

    return success


if __name__ == "__main__":
    main() 