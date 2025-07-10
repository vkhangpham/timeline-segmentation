#!/usr/bin/env python3
"""Comprehensive evaluation script for timeline segmentation.

This script provides evaluation capabilities for timeline segmentation results
including objective function scoring, baseline comparisons, and auto-metrics.
"""

import argparse
import sys
from pathlib import Path

from core.utils.general import discover_available_timeline_domains
from core.utils.config import AlgorithmConfig
from core.utils.logging import configure_global_logging, get_logger
from core.evaluation.evaluation import (
    run_single_evaluation,
    run_all_domains_evaluation,
    run_baseline_only_evaluation,
)


def main():
    """Main evaluation execution."""
    parser = argparse.ArgumentParser(
        description="Timeline Segmentation Evaluation - Comprehensive evaluation with baselines and auto-metrics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/run_evaluation.py --domain art --timeline-file results/timelines/art_timeline_analysis.json
  python scripts/run_evaluation.py --domain all --verbose
  python scripts/run_evaluation.py --domain computer_vision --baseline-only manual --timeline-file results/timelines/computer_vision_timeline_analysis.json
  python scripts/run_evaluation.py --domain deep_learning --baseline-only 5-year --timeline-file results/timelines/deep_learning_timeline_analysis.json
        """,
    )
    
    parser.add_argument(
        "--domain",
        type=str,
        required=True,
        help='Domain to evaluate (use "all" for all domains with existing timeline files)',
    )
    parser.add_argument(
        "--baseline-only",
        type=str,
        choices=["gemini", "manual", "5-year", "10-year"],
        help="Run only a specific baseline evaluation",
    )
    parser.add_argument(
        "--data-directory",
        type=str,
        default="resources",
        help="Directory containing data files",
    )
    parser.add_argument(
        "--verbose", 
        action="store_true", 
        help="Enable verbose logging (DEBUG level)"
    )
    parser.add_argument(
        "--timeline-file",
        type=str,
        default=None,
        help="Path to existing timeline JSON file (required for specific domains)",
    )
    
    args = parser.parse_args()
    
    # Validate timeline file usage
    if args.domain != "all":
        # For specific domains, timeline file is required
        if not args.timeline_file:
            print("Error: --timeline-file is required when evaluating a specific domain")
            print("Timeline evaluation requires existing timeline files.")
            print("Example: python scripts/run_evaluation.py --domain art --timeline-file results/timelines/art_timeline_analysis.json")
            return False
        
        # Check if timeline file exists
        timeline_path = Path(args.timeline_file)
        if not timeline_path.exists():
            print(f"Error: Timeline file not found: {args.timeline_file}")
            return False
    else:
        # For "all" domains, timeline file should not be specified
        if args.timeline_file:
            print("Error: --timeline-file cannot be used with --domain all")
            print("The 'all' option automatically discovers and uses timeline files from results/timelines/")
            return False
    
    # Validate baseline-only usage
    if args.baseline_only:
        if args.domain == "all":
            print("Error: --baseline-only cannot be used with --domain all")
            print("Baseline evaluation requires specifying a specific domain.")
            return False
        
        if not args.timeline_file:
            print("Error: --timeline-file is required when using --baseline-only")
            print("Baseline evaluation requires an existing timeline file for comparison.")
            return False
    
    # Configure logging
    configure_global_logging(
        verbose=args.verbose, 
        domain_name=args.domain if args.domain != "all" else None
    )
    logger = get_logger(
        __name__, 
        args.verbose, 
        args.domain if args.domain != "all" else None
    )
    
    # Prepare algorithm config
    domain_for_config = args.domain if args.domain != "all" else None
    algorithm_config = AlgorithmConfig.from_config_file(domain_name=domain_for_config)
    
    # Run evaluation
    success = False
    
    if args.baseline_only:
        # Run baseline-only evaluation
        available_domains = discover_available_timeline_domains(args.verbose)
        if args.domain not in available_domains:
            logger.error(f"Invalid domain: {args.domain}")
            logger.error(f"Available domains with timeline files: {', '.join(available_domains)}")
            return False
        
        success = run_baseline_only_evaluation(
            domain_name=args.domain,
            baseline_type=args.baseline_only,
            algorithm_config=algorithm_config,
            data_directory=args.data_directory,
            verbose=args.verbose,
        )
    elif args.domain == "all":
        # Run comprehensive evaluation for all domains with existing timeline files
        success = run_all_domains_evaluation(
            algorithm_config=algorithm_config,
            data_directory=args.data_directory,
            verbose=args.verbose,
        )
    else:
        # Run comprehensive evaluation for single domain
        available_domains = discover_available_timeline_domains(args.verbose)
        if args.domain not in available_domains:
            logger.warning(f"Domain '{args.domain}' not found in existing timeline files")
            logger.info(f"Available domains with timeline files: {', '.join(available_domains)}")
            logger.info("Proceeding with evaluation using provided timeline file...")
        
        success = run_single_evaluation(
            domain_name=args.domain,
            algorithm_config=algorithm_config,
            data_directory=args.data_directory,
            timeline_file=args.timeline_file,
            verbose=args.verbose,
        )
    
    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 