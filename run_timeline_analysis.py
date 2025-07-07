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
import json
from typing import List
from pathlib import Path

from core.utils.general import discover_available_domains, ensure_results_directory
from core.utils.config import AlgorithmConfig
from core.pipeline.orchestrator import analyze_timeline
from core.utils.parameters import load_optimized_parameters
from core.results.display import display_analysis_summary
from core.utils.logging import configure_global_logging, get_logger


def run_domain_analysis(
    domain_name: str,
    segmentation_only: bool = False,
    algorithm_config: AlgorithmConfig = None,
    optimized_params_file: str = None,
    verbose: bool = False,
) -> bool:
    """Run complete analysis for a single domain using simplified orchestrator."""
    logger = get_logger(__name__, verbose)

    # Load default config if none provided
    if algorithm_config is None:
        algorithm_config = AlgorithmConfig.from_config_file(domain_name=domain_name)
        if verbose:
            logger.info("No optimized parameters found, using defaults")

    # Load optimized parameters and update config
    optimized_params = load_optimized_parameters(domain_name, optimized_params_file)

    if optimized_params and algorithm_config:
        # Get current config as dict and update with optimized params
        import dataclasses

        config_kwargs = dataclasses.asdict(algorithm_config)

        for param_name, param_value in optimized_params.items():
            if param_name in config_kwargs:
                config_kwargs[param_name] = param_value

        # Update the config with optimized parameters
        algorithm_config = AlgorithmConfig(**config_kwargs)

    try:
        # Run simplified timeline analysis
        timeline_result = analyze_timeline(
            domain_name=domain_name,
            algorithm_config=algorithm_config,
            data_directory="resources",
            verbose=verbose,
        )

        # Save results
        save_timeline_result(timeline_result, domain_name, verbose)

        # Display results
        display_timeline_summary(timeline_result, verbose)

        return True

    except Exception as e:
        logger.error(f"Analysis failed for {domain_name}: {e}")
        return False


def save_timeline_result(timeline_result, domain_name: str, verbose: bool = False):
    """Save timeline analysis results to JSON file."""
    logger = get_logger(__name__, verbose)

    # Create results directory
    results_dir = Path("results/timelines")
    results_dir.mkdir(parents=True, exist_ok=True)

    # Convert timeline result to serializable format
    result_data = {
        "domain_name": timeline_result.domain_name,
        "confidence": timeline_result.confidence,
        "boundary_years": list(timeline_result.boundary_years),
        "narrative_evolution": timeline_result.narrative_evolution,
        "periods": [],
    }

    # Add period information
    for period in timeline_result.periods:
        period_data = {
            "start_year": period.start_year,
            "end_year": period.end_year,
            "total_papers": period.total_papers,
            "total_citations": period.total_citations,
            "top_keywords": list(period.top_keywords),
            "topic_label": period.topic_label,
            "topic_description": period.topic_description,
            "confidence": period.confidence,
            "network_stability": period.network_stability,
            "network_metrics": period.network_metrics,
        }
        result_data["periods"].append(period_data)

    # Save to file
    output_file = results_dir / f"{domain_name}_timeline_analysis.json"
    with open(output_file, "w") as f:
        json.dump(result_data, f, indent=2)

    logger.info(f"Timeline results saved to {output_file}")


def display_timeline_summary(timeline_result, verbose: bool = False):
    """Display timeline analysis summary."""
    logger = get_logger(__name__, verbose)

    print(f"\n{'='*50}")
    print(f"TIMELINE ANALYSIS SUMMARY: {timeline_result.domain_name}")
    print(f"{'='*50}")

    print(f"Overall Confidence: {timeline_result.confidence:.3f}")
    print(f"Boundary Years: {list(timeline_result.boundary_years)}")
    print(f"Number of Periods: {len(timeline_result.periods)}")

    print(f"\nPERIOD BREAKDOWN:")
    print(f"{'-'*40}")

    for i, period in enumerate(timeline_result.periods):
        print(f"Period {i+1}: {period.start_year}-{period.end_year}")
        print(f"  Papers: {period.total_papers}")
        print(f"  Citations: {period.total_citations}")
        print(f"  Topic: {period.topic_label}")
        print(f"  Confidence: {period.confidence:.3f}")
        if period.top_keywords:
            print(f"  Top Keywords: {', '.join(period.top_keywords[:5])}")
        print()

    print(f"NARRATIVE EVOLUTION:")
    print(f"{'-'*40}")
    print(timeline_result.narrative_evolution)
    print()


def run_all_domains(
    segmentation_only: bool = False,
    algorithm_config: AlgorithmConfig = None,
    optimized_params_file: str = None,
    verbose: bool = False,
) -> bool:
    """Run analysis for all available domains."""
    logger = get_logger(__name__, verbose)

    if algorithm_config is None:
        algorithm_config = AlgorithmConfig.from_config_file()

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
        domain_config = AlgorithmConfig.from_config_file(domain_name=domain)
        if run_domain_analysis(
            domain, segmentation_only, domain_config, optimized_params_file, verbose
        ):
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
        description="Timeline Segmentation Pipeline - Simplified architecture with clean data flow",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_timeline_analysis.py --domain deep_learning
  python run_timeline_analysis.py --domain all --verbose
  python run_timeline_analysis.py --domain computer_vision --granularity 1
  python run_timeline_analysis.py --domain applied_mathematics --segmentation-only
        """,
    )

    parser.add_argument(
        "--domain",
        type=str,
        required=True,
        help='Domain to analyze (use "all" for all domains)',
    )
    parser.add_argument(
        "--segmentation-only",
        action="store_true",
        help="Run only segmentation (skip timeline analysis)",
    )
    parser.add_argument(
        "--granularity",
        type=int,
        default=3,
        choices=[1, 2, 3, 4, 5],
        help="Timeline granularity: 1=fine, 3=balanced, 5=coarse",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Enable verbose logging (DEBUG level)"
    )
    parser.add_argument(
        "--direction-threshold",
        type=float,
        default=None,
        help="Direction detection threshold (0.1-0.8)",
    )
    parser.add_argument(
        "--validation-threshold",
        type=float,
        default=None,
        help="Validation threshold (0.5-0.95)",
    )
    parser.add_argument(
        "--citation-boost-rate",
        type=float,
        default=None,
        help="Citation support boost (0.0-1.0)",
    )
    parser.add_argument(
        "--optimized-params-file",
        type=str,
        default=None,
        help="Path to optimized parameters JSON file",
    )

    args = parser.parse_args()

    # Configure global logging based on verbosity
    configure_global_logging(verbose=args.verbose)
    logger = get_logger(__name__, args.verbose)

    # Build algorithm config
    overrides = {}
    if args.direction_threshold is not None:
        overrides["direction_threshold"] = args.direction_threshold
    if args.validation_threshold is not None:
        overrides["validation_threshold"] = args.validation_threshold
    if args.citation_boost_rate is not None:
        overrides["citation_boost_rate"] = args.citation_boost_rate

    domain_for_config = args.domain if args.domain != "all" else None

    # Load the base config and apply overrides if needed
    algorithm_config = AlgorithmConfig.from_config_file(domain_name=domain_for_config)

    if overrides:
        # Create a new config with overrides by reconstructing the dataclass
        import dataclasses

        config_dict = dataclasses.asdict(algorithm_config)
        config_dict.update(overrides)
        algorithm_config = AlgorithmConfig(**config_dict)

    ensure_results_directory()

    if args.domain == "all":
        success = run_all_domains(
            args.segmentation_only,
            algorithm_config,
            args.optimized_params_file,
            args.verbose,
        )
    else:
        available_domains = discover_available_domains(args.verbose)
        if args.domain not in available_domains:
            logger.error(f"Invalid domain: {args.domain}")
            logger.error(f"Available: {', '.join(available_domains)}")
            return False

        success = run_domain_analysis(
            args.domain,
            args.segmentation_only,
            algorithm_config,
            args.optimized_params_file,
            args.verbose,
        )

    return success


if __name__ == "__main__":
    main()
