#!/usr/bin/env python3
"""Comprehensive evaluation script for timeline segmentation.

This script provides evaluation capabilities for timeline segmentation results
including objective function scoring, baseline comparisons, and auto-metrics.
"""

import argparse
import sys
from pathlib import Path

from core.utils.general import discover_available_timeline_domains, get_timeline_file_path
from core.utils.config import AlgorithmConfig
from core.utils.logging import configure_global_logging, get_logger
from core.evaluation.evaluation import evaluate_domains
from core.evaluation.baselines import clear_cache
from core.evaluation.analysis import display_cross_domain_analysis


def main():
    """Main evaluation execution."""
    parser = argparse.ArgumentParser(
        description="Timeline Segmentation Evaluation - Comprehensive evaluation with baselines and auto-metrics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/run_evaluation.py --domain art --timeline-file results/timelines/art_timeline_analysis.json
  python scripts/run_evaluation.py --domain all --verbose
  python scripts/run_evaluation.py --domain computer_vision --baseline-only 5-year
  python scripts/run_evaluation.py --domain deep_learning --baseline-only 10-year
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
        choices=["5-year", "10-year"],
        help="Run only a specific baseline evaluation",
    )
    parser.add_argument(
        "--timeline-file",
        type=str,
        help="Optional path to specific timeline file to evaluate",
    )
    parser.add_argument(
        "--data-directory",
        type=str,
        default="resources",
        help="Directory containing domain data (default: resources)",
    )
    parser.add_argument(
        "--clear-cache",
        action="store_true",
        help="Clear baseline result cache before evaluation",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    # Configure logging
    configure_global_logging(verbose=args.verbose)
    logger = get_logger(__name__, args.verbose)

    # Clear cache if requested
    if args.clear_cache:
        clear_cache()
        logger.info("Baseline cache cleared")

    # Load algorithm configuration
    algorithm_config = AlgorithmConfig.from_config_file()

    # Validate timeline file if provided
    if args.timeline_file and not Path(args.timeline_file).exists():
        logger.error(f"Timeline file not found: {args.timeline_file}")
        return False

    # Prepare domains and timeline files
    if args.domain == "all":
        domains = discover_available_timeline_domains(args.verbose)
        if not domains:
            logger.error("No timeline files found")
            return False
        
        # Build timeline files mapping
        timeline_files = {}
        for domain in domains:
            timeline_file = get_timeline_file_path(domain, args.verbose)
            if timeline_file:
                timeline_files[domain] = timeline_file
                
        print(f"EVALUATING {len(domains)} DOMAINS")
        print("=" * 50)
    else:
        # Single domain
        available_domains = discover_available_timeline_domains(args.verbose)
        if args.domain not in available_domains:
            logger.error(f"Invalid domain: {args.domain}")
            logger.error(f"Available domains: {', '.join(available_domains)}")
            return False
            
        domains = [args.domain]
        timeline_files = {args.domain: args.timeline_file} if args.timeline_file else None

    # Run evaluation
    results = evaluate_domains(
        domains=domains,
        algorithm_config=algorithm_config,
        data_directory=args.data_directory,
        timeline_files=timeline_files,
        baseline_only=args.baseline_only,
        verbose=args.verbose,
    )

    # Handle results
    successful = [d for d, r in results.items() if r is not None]
    failed = [d for d, r in results.items() if r is None]

    if failed:
        print(f"\nFailed domains: {', '.join(failed)}")

    # Cross-domain analysis for multiple successful evaluations
    if len(successful) >= 2 and not args.baseline_only:
        try:
            successful_results = {d: r for d, r in results.items() if r is not None}
            display_cross_domain_analysis(successful_results, args.verbose)
        except Exception as e:
            logger.warning(f"Cross-domain analysis failed: {e}")

    success = len(successful) > 0
    if success:
        logger.info("Evaluation completed successfully")
    else:
        logger.error("Evaluation failed")
        
    return success


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
