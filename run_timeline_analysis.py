#!/usr/bin/env python3
"""
Timeline Segmentation Pipeline

Runs scientific literature timeline segmentation with change point detection,
period characterization, and segment merging.

Usage:
    python run_timeline_analysis.py --domain deep_learning
    python run_timeline_analysis.py --domain all --verbose
"""

import argparse
from typing import List

from core.utils.general import discover_available_domains, ensure_results_directory
from core.utils.config import AlgorithmConfig
from core.pipeline.orchestrator import run_complete_analysis
from core.utils.parameters import load_optimized_parameters
from core.results.display import display_analysis_summary
from core.utils.logging import configure_global_logging, get_logger


def run_domain_analysis(domain_name: str, segmentation_only: bool = False, granularity: int = 3, 
                       algorithm_config: AlgorithmConfig = None, optimized_params_file: str = None, 
                       verbose: bool = False) -> bool:
    """Run complete analysis for a single domain using specialized modules."""
    logger = get_logger(__name__, verbose)
    
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
            'citation_boost_rate': algorithm_config.citation_boost_rate,
            'citation_support_window': algorithm_config.citation_support_window,
            # Note: similarity_min/max_segment_length parameters deprecated (REFACTOR-003)
        }
        
        for param_name, param_value in optimized_params.items():
            if param_name in config_kwargs:
                config_kwargs[param_name] = param_value
        
        algorithm_config = AlgorithmConfig(**config_kwargs)
    
    # Run complete analysis using pipeline orchestrator
    results = run_complete_analysis(
        domain_name=domain_name,
        algorithm_config=algorithm_config,
        segmentation_only=segmentation_only,
        save_results=True,
        verbose=verbose
    )
    
    # Display results
    display_analysis_summary(results)
    
    return results.get('success', False)


def run_all_domains(segmentation_only: bool = False, granularity: int = 3, 
                   algorithm_config: AlgorithmConfig = None, optimized_params_file: str = None,
                   verbose: bool = False) -> bool:
    """Run analysis for all available domains."""
    logger = get_logger(__name__, verbose)
    
    if algorithm_config is None:
        algorithm_config = AlgorithmConfig(granularity=granularity)
    
    domains = discover_available_domains(verbose)
    if not domains:
        logger.error("No domains found")
        return False
    
    successful = []
    analysis_type = "SEGMENTATION" if segmentation_only else "ANALYSIS"
    
    logger.info(f"CROSS-DOMAIN {analysis_type}")
    logger.info("=" * 50)
    logger.info(f"Processing {len(domains)} domains: {', '.join(domains)}")
    
    for domain in domains:
        logger.info(f"Processing {domain}...")
        domain_config = AlgorithmConfig(granularity=granularity, domain_name=domain)
        if run_domain_analysis(domain, segmentation_only, granularity, domain_config, 
                             optimized_params_file, verbose):
            successful.append(domain)
    
    logger.info(f"{analysis_type} COMPLETE")
    logger.info("=" * 30)
    logger.info(f"Success: {len(successful)}/{len(domains)} domains")
    
    if successful:
        logger.info(f"Processed: {', '.join(successful)}")
        if not segmentation_only:
            logger.info("Results saved in 'results/timelines/' directory")
    
    if len(successful) < len(domains):
        failed = [d for d in domains if d not in successful]
        logger.warning(f"Failed: {', '.join(failed)}")
    
    return len(successful) > 0


def main():
    """Main pipeline execution."""
    parser = argparse.ArgumentParser(
        description="Timeline Segmentation Pipeline - Clean architecture with specialized modules",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_timeline_analysis.py --domain deep_learning
  python run_timeline_analysis.py --domain all --verbose
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
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging (DEBUG level)')
    parser.add_argument('--direction-threshold', type=float, default=None,
                       help='Direction detection threshold (0.1-0.8)')
    parser.add_argument('--validation-threshold', type=float, default=None,
                       help='Validation threshold (0.5-0.95)')
    parser.add_argument('--citation-boost-rate', type=float, default=None,
                       help='Citation support boost (0.0-1.0)')
    parser.add_argument('--optimized-params-file', type=str, default=None,
                       help='Path to optimized parameters JSON file')
    
    args = parser.parse_args()
    
    # Configure global logging based on verbosity
    configure_global_logging(verbose=args.verbose)
    logger = get_logger(__name__, args.verbose)
    
    # Build algorithm config
    overrides = {}
    if args.direction_threshold is not None:
        overrides['direction_threshold'] = args.direction_threshold
    if args.validation_threshold is not None:
        overrides['validation_threshold'] = args.validation_threshold  
    if args.citation_boost_rate is not None:
        overrides['citation_boost_rate'] = args.citation_boost_rate
    
    domain_for_config = args.domain if args.domain != 'all' else None
    
    if overrides:
        algorithm_config = AlgorithmConfig.create_custom(
            granularity=args.granularity, domain_name=domain_for_config, overrides=overrides
        )
    else:
        algorithm_config = AlgorithmConfig(granularity=args.granularity, domain_name=domain_for_config)
    
    ensure_results_directory()
    
    if args.domain == 'all':
        success = run_all_domains(args.segmentation_only, args.granularity, algorithm_config, 
                                args.optimized_params_file, args.verbose)
    else:
        available_domains = discover_available_domains(args.verbose)
        if args.domain not in available_domains:
            logger.error(f"Invalid domain: {args.domain}")
            logger.error(f"Available: {', '.join(available_domains)}")
            return False
        
        success = run_domain_analysis(args.domain, args.segmentation_only, args.granularity, 
                                    algorithm_config, args.optimized_params_file, args.verbose)

    return success


if __name__ == "__main__":
    main() 