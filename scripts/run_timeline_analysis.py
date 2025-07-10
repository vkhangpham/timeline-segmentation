#!/usr/bin/env python3
"""Timeline segmentation pipeline for scientific literature analysis.
Provides command-line interface for running timeline analysis with change point detection.
"""

import argparse
import json
from pathlib import Path

from core.utils.general import discover_available_domains, ensure_results_directory
from core.utils.config import AlgorithmConfig
from core.pipeline.orchestrator import analyze_timeline
from core.utils.logging import configure_global_logging, get_logger


def load_optimized_parameters(domain_name: str, verbose: bool = False) -> dict:
    """Load optimized parameters for a domain from optimization results.
    
    Args:
        domain_name: Name of the domain
        verbose: Enable verbose logging
        
    Returns:
        Dictionary of optimized parameters
        
    Raises:
        FileNotFoundError: If optimized parameters file doesn't exist
        ValueError: If parameters file is invalid
    """
    logger = get_logger(__name__, verbose, domain_name)
    
    # Path to optimized parameters file
    params_file = Path(f"results/optimized_params/{domain_name}.json")
    
    if not params_file.exists():
        raise FileNotFoundError(
            f"No optimized parameters found for {domain_name}. "
            f"Run optimization first: python scripts/optimize_domain.py --domain {domain_name}"
        )
    
    try:
        with open(params_file, "r") as f:
            params_data = json.load(f)
        
        if "best_parameters" not in params_data:
            raise ValueError(f"Invalid optimization results file: {params_file}")
            
        optimized_params = params_data["best_parameters"]
        
        if verbose:
            logger.info(f"Loaded optimized parameters from {params_file}")
            logger.info(f"Optimization date: {params_data.get('optimization_date', 'unknown')}")
            logger.info(f"Best objective score: {params_data.get('best_objective_score', 'unknown'):.3f}")
            logger.info(f"Optimized parameters: {optimized_params}")
        else:
            print(f"Using optimized parameters for {domain_name} (score: {params_data.get('best_objective_score', 0):.3f})")
            
        return optimized_params
        
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in optimization results file: {e}")


def run_domain_analysis(
    domain_name: str,
    segmentation_only: bool = False,
    algorithm_config: AlgorithmConfig = None,
    no_save: bool = False,
    verbose: bool = False,
) -> bool:
    """Run complete analysis for a single domain.

    Args:
        domain_name: Name of the domain to analyze
        segmentation_only: Run only segmentation (skip timeline analysis)
        algorithm_config: Algorithm configuration (defaults to config.json)
        no_save: Skip saving results to files
        verbose: Enable verbose logging

    Returns:
        True if analysis succeeded, False otherwise
    """
    logger = get_logger(__name__, verbose, domain_name)

    if algorithm_config is None:
        algorithm_config = AlgorithmConfig.from_config_file(domain_name=domain_name)
        if verbose:
            logger.info("Using default configuration")

    try:
        timeline_result = analyze_timeline(
            domain_name=domain_name,
            algorithm_config=algorithm_config,
            data_directory="resources",
            segmentation_only=segmentation_only,
            verbose=verbose,
        )

        if not no_save:
            save_timeline_result(timeline_result, domain_name, verbose)
        else:
            logger.info("Skipping save (--no-save flag specified)")

        display_timeline_summary(timeline_result, verbose)

        return True

    except Exception as e:
        logger.error(f"Analysis failed for {domain_name}: {e}")
        return False


def save_timeline_result(timeline_result, domain_name: str, verbose: bool = False):
    """Save timeline analysis results to JSON file.

    Args:
        timeline_result: TimelineAnalysisResult object
        domain_name: Name of the domain
        verbose: Enable verbose logging
    """
    logger = get_logger(__name__, verbose, domain_name)

    results_dir = Path("results/timelines")
    results_dir.mkdir(parents=True, exist_ok=True)

    result_data = {
        "domain_name": timeline_result.domain_name,
        "confidence": timeline_result.confidence,
        "boundary_years": list(timeline_result.boundary_years),
        "narrative_evolution": timeline_result.narrative_evolution,
        "periods": [],
    }

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

    output_file = results_dir / f"{domain_name}_timeline_analysis.json"
    with open(output_file, "w") as f:
        json.dump(result_data, f, indent=2)

    logger.info(f"Timeline results saved to {output_file}")


def display_timeline_summary(timeline_result, verbose: bool = False):
    """Display timeline analysis summary.

    Args:
        timeline_result: TimelineAnalysisResult object
        verbose: Enable verbose logging
    """
    logger = get_logger(__name__, verbose, timeline_result.domain_name)

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
    no_save: bool = False,
    verbose: bool = False,
) -> bool:
    """Run analysis for all available domains.

    Args:
        segmentation_only: Run only segmentation (skip timeline analysis)
        algorithm_config: Algorithm configuration (defaults to config.json)
        no_save: Skip saving results to files
        verbose: Enable verbose logging

    Returns:
        True if at least one domain succeeded, False otherwise
    """
    logger = get_logger(__name__, verbose, "all_domains")

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
            domain, segmentation_only, domain_config, no_save, verbose
        ):
            successful.append(domain)

    logger.info(f"{analysis_type} COMPLETE")
    logger.info("=" * 30)
    logger.info(f"Success: {len(successful)}/{len(domains)} domains")

    if successful:
        logger.info(f"Processed: {', '.join(successful)}")
        if not segmentation_only and not no_save:
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
  python scripts/run_timeline_analysis.py --domain deep_learning
  python scripts/run_timeline_analysis.py --domain applied_mathematics --use-optimized
  python scripts/run_timeline_analysis.py --domain all --verbose
  python scripts/run_timeline_analysis.py --domain computer_vision --segmentation-only
  python scripts/run_timeline_analysis.py --domain applied_mathematics --use-optimized --verbose
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
        help="Run only segmentation (skip characterization and merging)",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Skip saving results to files",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Enable verbose logging (DEBUG level)"
    )
    parser.add_argument(
        "--direction-threshold",
        type=float,
        default=None,
        help="Direction change detection threshold (0.4-0.9)",
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
        help="Citation confidence boost (0.1-0.5)",
    )
    parser.add_argument(
        "--use-optimized",
        action="store_true",
        help="Use optimized parameters for the domain (if available)",
    )

    args = parser.parse_args()

    configure_global_logging(
        verbose=args.verbose, domain_name=args.domain if args.domain != "all" else None
    )
    logger = get_logger(
        __name__, args.verbose, args.domain if args.domain != "all" else None
    )

    # Build parameter overrides from command line arguments
    overrides = {}
    if args.direction_threshold is not None:
        overrides["direction_change_threshold"] = args.direction_threshold
    if args.validation_threshold is not None:
        overrides["validation_threshold"] = args.validation_threshold
    if args.citation_boost_rate is not None:
        overrides["citation_confidence_boost"] = args.citation_boost_rate

    domain_for_config = args.domain if args.domain != "all" else None

    algorithm_config = AlgorithmConfig.from_config_file(domain_name=domain_for_config)

    # Apply optimized parameters if requested
    if args.use_optimized and args.domain != "all":
        try:
            optimized_params = load_optimized_parameters(args.domain, args.verbose)
            # Add optimized parameters to overrides (command line args take precedence)
            for key, value in optimized_params.items():
                if key not in overrides:  # Don't override explicit command line arguments
                    overrides[key] = value
            
            # CRITICAL: When using optimized direction_change_threshold, 
            # must set strategy to "fixed" so the algorithm uses it
            if 'direction_change_threshold' in optimized_params and 'direction_threshold_strategy' not in overrides:
                overrides['direction_threshold_strategy'] = 'fixed'
                if args.verbose:
                    logger.info("Set direction_threshold_strategy='fixed' to use optimized direction_change_threshold")
                    
        except FileNotFoundError:
            logger.warning(f"Optimized parameters not found for {args.domain}. Using default config.")
        except ValueError as e:
            logger.error(f"Error loading optimized parameters for {args.domain}: {e}")
            return False

    # Apply all overrides to the configuration
    if overrides:
        import dataclasses

        config_dict = dataclasses.asdict(algorithm_config)
        config_dict.update(overrides)
        algorithm_config = AlgorithmConfig(**config_dict)
        
        if args.verbose:
            logger.info(f"Applied parameter overrides: {overrides}")

    ensure_results_directory()

    if args.domain == "all":
        success = run_all_domains(
            args.segmentation_only,
            algorithm_config,
            args.no_save,
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
            args.no_save,
            args.verbose,
        )

    return success


if __name__ == "__main__":
    main()
