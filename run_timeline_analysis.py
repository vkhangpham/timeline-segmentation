#!/usr/bin/env python3
"""
Timeline Segmentation Pipeline

Runs scientific literature timeline segmentation with change point detection,
period characterization, and segment merging.

Usage:
    python run_timeline_analysis.py --domain deep_learning
    python run_timeline_analysis.py --domain all
"""

import argparse
from typing import List

from core.utils import discover_available_domains, ensure_results_directory
from core.algorithm_config import AlgorithmConfig
from core.integration import (
    run_complete_analysis, 
    load_optimized_parameters, 
    display_analysis_summary
)


def run_domain_analysis(domain_name: str, segmentation_only: bool = False, granularity: int = 3, 
                       algorithm_config: AlgorithmConfig = None, optimized_params_file: str = None) -> bool:
    """Run complete analysis for a single domain using the integration layer."""
    
    # Load optimized parameters and update config
    optimized_params = load_optimized_parameters(domain_name, optimized_params_file)
    
    if optimized_params and algorithm_config:
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
        
        for param_name, param_value in optimized_params.items():
            if param_name in config_kwargs:
                config_kwargs[param_name] = param_value
        
        algorithm_config = AlgorithmConfig(**config_kwargs)
    
    # Run complete analysis through integration layer
    results = run_complete_analysis(
        domain_name=domain_name,
        algorithm_config=algorithm_config,
        segmentation_only=segmentation_only,
        save_results=True
    )
    
    # Display results
    display_analysis_summary(results)
    
    return results.get('success', False)


def run_all_domains(segmentation_only: bool = False, granularity: int = 3, 
                   algorithm_config: AlgorithmConfig = None, optimized_params_file: str = None) -> bool:
    """Run analysis for all available domains."""
    if algorithm_config is None:
        algorithm_config = AlgorithmConfig(granularity=granularity)
    
    domains = discover_available_domains()
    if not domains:
        print("No domains found")
        return False
    
    successful = []
    analysis_type = "SEGMENTATION" if segmentation_only else "ANALYSIS"
    
    print(f"CROSS-DOMAIN {analysis_type}")
    print("=" * 50)
    print(f"Processing {len(domains)} domains: {', '.join(domains)}")
    
    for domain in domains:
        print(f"\nProcessing {domain}...")
        domain_config = AlgorithmConfig(granularity=granularity, domain_name=domain)
        if run_domain_analysis(domain, segmentation_only, granularity, domain_config, optimized_params_file):
            successful.append(domain)
    
    print(f"\n{analysis_type} COMPLETE")
    print("=" * 30)
    print(f"Success: {len(successful)}/{len(domains)} domains")
    
    if successful:
        print("Processed:", ", ".join(successful))
        if not segmentation_only:
            print("Results saved in 'results/timelines/' directory")
    
    if len(successful) < len(domains):
        failed = [d for d in domains if d not in successful]
        print("Failed:", ", ".join(failed))
    
    return len(successful) > 0


def main():
    """Main pipeline execution."""
    parser = argparse.ArgumentParser(
        description="Timeline Segmentation Pipeline - Clean architecture with integration layer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_timeline_analysis.py --domain deep_learning
  python run_timeline_analysis.py --domain all
  python run_timeline_analysis.py --domain computer_vision --granularity 1
  python run_timeline_analysis.py --domain applied_mathematics --segmentation-only
        """
    )
    
    parser.add_argument('--domain', type=str, required=True,
                       help='Domain to analyze (use "all" for all domains)')
    parser.add_argument('--segmentation-only', action='store_true',
                       help='Run only segmentation (skip timeline analysis)')
    parser.add_argument('--granularity', type=int, default=3, choices=[1, 2, 3, 4, 5],
                       help='Timeline granularity: 1=fine, 3=balanced, 5=coarse')
    parser.add_argument('--direction-threshold', type=float, default=None,
                       help='Direction detection threshold (0.1-0.8)')
    parser.add_argument('--validation-threshold', type=float, default=None,
                       help='Validation threshold (0.5-0.95)')
    parser.add_argument('--citation-boost', type=float, default=None,
                       help='Citation support boost (0.0-1.0)')
    parser.add_argument('--optimized-params-file', type=str, default=None,
                       help='Path to optimized parameters JSON file')
    
    args = parser.parse_args()
    
    # Build algorithm config
    overrides = {}
    if args.direction_threshold is not None:
        overrides['direction_threshold'] = args.direction_threshold
    if args.validation_threshold is not None:
        overrides['validation_threshold'] = args.validation_threshold  
    if args.citation_boost is not None:
        overrides['citation_boost'] = args.citation_boost
    
    domain_for_config = args.domain if args.domain != 'all' else None
    
    if overrides:
        algorithm_config = AlgorithmConfig.create_custom(
            granularity=args.granularity, domain_name=domain_for_config, overrides=overrides
        )
    else:
        algorithm_config = AlgorithmConfig(granularity=args.granularity, domain_name=domain_for_config)
    
    ensure_results_directory()
    
    if args.domain == 'all':
        success = run_all_domains(args.segmentation_only, args.granularity, algorithm_config, args.optimized_params_file)
    else:
        available_domains = discover_available_domains()
        if args.domain not in available_domains:
            print(f"Invalid domain: {args.domain}")
            print(f"Available: {', '.join(available_domains)}")
            return False
        
        success = run_domain_analysis(args.domain, args.segmentation_only, args.granularity, 
                                    algorithm_config, args.optimized_params_file)

    return success


if __name__ == "__main__":
    main() 