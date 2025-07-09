#!/usr/bin/env python3
"""Comprehensive evaluation script for timeline segmentation.

This script provides evaluation capabilities for timeline segmentation results
including objective function scoring, baseline comparisons, and auto-metrics.
"""

import argparse
import json
from pathlib import Path
import sys

from core.utils.general import discover_available_domains
from core.utils.config import AlgorithmConfig
from core.utils.logging import configure_global_logging, get_logger
from core.pipeline.orchestrator import analyze_timeline
from core.evaluation.evaluation import (
    evaluate_timeline_result,
    create_gemini_baseline,
    create_manual_baseline,
    create_fixed_year_baseline,
    run_comprehensive_evaluation,
)


def run_single_evaluation(
    domain_name: str,
    algorithm_config: AlgorithmConfig,
    data_directory: str = "resources",
    verbose: bool = False,
) -> bool:
    """Run evaluation for a single domain.
    
    Args:
        domain_name: Domain name to evaluate
        algorithm_config: Algorithm configuration
        data_directory: Directory containing data files
        verbose: Enable verbose logging
        
    Returns:
        True if evaluation succeeded, False otherwise
    """
    logger = get_logger(__name__, verbose, domain_name)
    
    try:
        # 1. Run segmentation-only pipeline to get timeline result
        logger.info(f"Running segmentation pipeline for {domain_name}")
        timeline_result = analyze_timeline(
            domain_name=domain_name,
            algorithm_config=algorithm_config,
            data_directory=data_directory,
            segmentation_only=True,
            verbose=verbose,
        )
        
        # 2. Run comprehensive evaluation
        logger.info(f"Running comprehensive evaluation for {domain_name}")
        evaluation_result = run_comprehensive_evaluation(
            domain_name=domain_name,
            timeline_result=timeline_result,
            algorithm_config=algorithm_config,
            data_directory=data_directory,
            verbose=verbose,
        )
        
        # 3. Save results
        save_evaluation_result(evaluation_result, verbose)
        
        # 4. Display summary
        display_evaluation_summary(evaluation_result, verbose)
        
        return True
        
    except Exception as e:
        logger.error(f"Evaluation failed for {domain_name}: {e}")
        return False


def save_evaluation_result(evaluation_result, verbose: bool = False):
    """Save evaluation result to JSON file.
    
    Args:
        evaluation_result: ComprehensiveEvaluationResult object
        verbose: Enable verbose logging
    """
    logger = get_logger(__name__, verbose, evaluation_result.domain_name)
    
    # Create results directory
    results_dir = Path("results/evaluation")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Prepare data for JSON serialization
    result_data = {
        "domain_name": evaluation_result.domain_name,
        "algorithm_result": {
            "objective_score": evaluation_result.algorithm_result.objective_score,
            "cohesion_score": evaluation_result.algorithm_result.cohesion_score,
            "separation_score": evaluation_result.algorithm_result.separation_score,
            "num_segments": evaluation_result.algorithm_result.num_segments,
            "num_transitions": evaluation_result.algorithm_result.num_transitions,
            "boundary_years": evaluation_result.algorithm_result.boundary_years,
            "methodology": evaluation_result.algorithm_result.methodology,
            "details": evaluation_result.algorithm_result.details,
        },
        "baseline_results": [],
        "auto_metrics": {
            "boundary_f1": evaluation_result.auto_metrics.boundary_f1,
            "boundary_precision": evaluation_result.auto_metrics.boundary_precision,
            "boundary_recall": evaluation_result.auto_metrics.boundary_recall,
            "segment_f1": evaluation_result.auto_metrics.segment_f1,
            "segment_precision": evaluation_result.auto_metrics.segment_precision,
            "segment_recall": evaluation_result.auto_metrics.segment_recall,
            "tolerance": evaluation_result.auto_metrics.tolerance,
            "details": evaluation_result.auto_metrics.details,
        },
        "ranking": evaluation_result.ranking,
        "summary": evaluation_result.summary,
    }
    
    # Add baseline results
    for baseline in evaluation_result.baseline_results:
        baseline_data = {
            "baseline_name": baseline.baseline_name,
            "objective_score": baseline.objective_score,
            "cohesion_score": baseline.cohesion_score,
            "separation_score": baseline.separation_score,
            "num_segments": baseline.num_segments,
            "boundary_years": baseline.boundary_years,
        }
        result_data["baseline_results"].append(baseline_data)
    
    # Save to file
    output_file = results_dir / f"{evaluation_result.domain_name}_evaluation.json"
    with open(output_file, "w") as f:
        json.dump(result_data, f, indent=2)
    
    logger.info(f"Evaluation results saved to {output_file}")


def display_evaluation_summary(evaluation_result, verbose: bool = False):
    """Display evaluation summary.
    
    Args:
        evaluation_result: ComprehensiveEvaluationResult object
        verbose: Enable verbose logging
    """
    logger = get_logger(__name__, verbose, evaluation_result.domain_name)
    
    print(f"\n{'='*60}")
    print(f"EVALUATION SUMMARY: {evaluation_result.domain_name}")
    print(f"{'='*60}")
    
    # Algorithm result
    alg_result = evaluation_result.algorithm_result
    print(f"\nALGORITHM RESULT:")
    print(f"-" * 20)
    print(f"Objective Score: {alg_result.objective_score:.3f}")
    print(f"Cohesion Score: {alg_result.cohesion_score:.3f}")
    print(f"Separation Score: {alg_result.separation_score:.3f}")
    print(f"Number of Segments: {alg_result.num_segments}")
    print(f"Boundary Years: {alg_result.boundary_years}")
    
    # Baseline results
    if evaluation_result.baseline_results:
        print(f"\nBASELINE RESULTS:")
        print(f"-" * 20)
        for baseline in evaluation_result.baseline_results:
            print(f"{baseline.baseline_name}: {baseline.objective_score:.3f} "
                  f"({baseline.num_segments} segments)")
    
    # Auto-metrics
    auto_metrics = evaluation_result.auto_metrics
    print(f"\nAUTO-METRICS (vs Manual):")
    print(f"-" * 30)
    print(f"Boundary F1: {auto_metrics.boundary_f1:.3f} "
          f"(P: {auto_metrics.boundary_precision:.3f}, R: {auto_metrics.boundary_recall:.3f})")
    print(f"Segment F1: {auto_metrics.segment_f1:.3f} "
          f"(P: {auto_metrics.segment_precision:.3f}, R: {auto_metrics.segment_recall:.3f})")
    print(f"Tolerance: ±{auto_metrics.tolerance} years")
    
    # Ranking
    print(f"\nRANKING (by Objective Score):")
    print(f"-" * 30)
    sorted_ranking = sorted(evaluation_result.ranking.items(), key=lambda x: x[1], reverse=True)
    for i, (name, score) in enumerate(sorted_ranking):
        rank_symbol = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else f"{i+1}."
        print(f"{rank_symbol} {name}: {score:.3f}")
    
    print(f"\n{'='*60}")


def run_all_domains_evaluation(
    algorithm_config: AlgorithmConfig,
    data_directory: str = "resources",
    verbose: bool = False,
) -> bool:
    """Run evaluation for all available domains.
    
    Args:
        algorithm_config: Algorithm configuration
        data_directory: Directory containing data files
        verbose: Enable verbose logging
        
    Returns:
        True if at least one domain succeeded, False otherwise
    """
    logger = get_logger(__name__, verbose, "all_domains")
    
    domains = discover_available_domains(verbose)
    if not domains:
        logger.error("No domains found")
        return False
    
    successful = []
    failed = []
    
    logger.info(f"COMPREHENSIVE EVALUATION")
    logger.info("=" * 50)
    logger.info(f"Processing {len(domains)} domains: {', '.join(domains)}")
    
    for domain in domains:
        logger.info(f"Processing {domain}...")
        domain_config = AlgorithmConfig.from_config_file(domain_name=domain)
        
        if run_single_evaluation(
            domain_name=domain,
            algorithm_config=domain_config,
            data_directory=data_directory,
            verbose=verbose,
        ):
            successful.append(domain)
        else:
            failed.append(domain)
    
    logger.info(f"EVALUATION COMPLETE")
    logger.info("=" * 30)
    logger.info(f"Success: {len(successful)}/{len(domains)} domains")
    
    if successful:
        logger.info(f"Processed: {', '.join(successful)}")
        logger.info("Results saved in 'results/evaluation/' directory")
    
    if failed:
        logger.warning(f"Failed: {', '.join(failed)}")
    
    return len(successful) > 0


def run_baseline_only_evaluation(
    domain_name: str,
    baseline_type: str,
    algorithm_config: AlgorithmConfig,
    data_directory: str = "resources",
    verbose: bool = False,
) -> bool:
    """Run evaluation for a specific baseline only.
    
    Args:
        domain_name: Domain name to evaluate
        baseline_type: Type of baseline (gemini, manual, 5-year, 10-year)
        algorithm_config: Algorithm configuration
        data_directory: Directory containing data files
        verbose: Enable verbose logging
        
    Returns:
        True if evaluation succeeded, False otherwise
    """
    logger = get_logger(__name__, verbose, domain_name)
    
    try:
        if baseline_type == "gemini":
            baseline_result = create_gemini_baseline(
                domain_name=domain_name,
                algorithm_config=algorithm_config,
                data_directory=data_directory,
                verbose=verbose,
            )
        elif baseline_type == "manual":
            baseline_result = create_manual_baseline(
                domain_name=domain_name,
                algorithm_config=algorithm_config,
                data_directory=data_directory,
                verbose=verbose,
            )
        elif baseline_type == "5-year":
            baseline_result = create_fixed_year_baseline(
                domain_name=domain_name,
                algorithm_config=algorithm_config,
                year_interval=5,
                data_directory=data_directory,
                verbose=verbose,
            )
        elif baseline_type == "10-year":
            baseline_result = create_fixed_year_baseline(
                domain_name=domain_name,
                algorithm_config=algorithm_config,
                year_interval=10,
                data_directory=data_directory,
                verbose=verbose,
            )
        else:
            logger.error(f"Invalid baseline type: {baseline_type}")
            return False
        
        # Display baseline result
        print(f"\n{'='*50}")
        print(f"BASELINE EVALUATION: {domain_name} ({baseline_type})")
        print(f"{'='*50}")
        print(f"Objective Score: {baseline_result.objective_score:.3f}")
        print(f"Cohesion Score: {baseline_result.cohesion_score:.3f}")
        print(f"Separation Score: {baseline_result.separation_score:.3f}")
        print(f"Number of Segments: {baseline_result.num_segments}")
        print(f"Boundary Years: {baseline_result.boundary_years}")
        
        return True
        
    except Exception as e:
        logger.error(f"Baseline evaluation failed for {domain_name}: {e}")
        return False


def main():
    """Main evaluation execution."""
    parser = argparse.ArgumentParser(
        description="Timeline Segmentation Evaluation - Comprehensive evaluation with baselines and auto-metrics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_evaluation.py --domain art
  python run_evaluation.py --domain all --verbose
  python run_evaluation.py --domain computer_vision --baseline-only manual
  python run_evaluation.py --domain deep_learning --baseline-only 5-year
        """,
    )
    
    parser.add_argument(
        "--domain",
        type=str,
        required=True,
        help='Domain to evaluate (use "all" for all domains)',
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
    
    args = parser.parse_args()
    
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
    
    # Prepare algorithm config with overrides
    overrides = {}
    if args.direction_threshold is not None:
        overrides["direction_threshold"] = args.direction_threshold
    if args.validation_threshold is not None:
        overrides["validation_threshold"] = args.validation_threshold
    if args.citation_boost_rate is not None:
        overrides["citation_confidence_boost"] = args.citation_boost_rate
    
    domain_for_config = args.domain if args.domain != "all" else None
    algorithm_config = AlgorithmConfig.from_config_file(domain_name=domain_for_config)
    
    if overrides:
        import dataclasses
        config_dict = dataclasses.asdict(algorithm_config)
        config_dict.update(overrides)
        algorithm_config = AlgorithmConfig(**config_dict)
    
    # Run evaluation
    success = False
    
    if args.baseline_only:
        # Run baseline-only evaluation
        if args.domain == "all":
            logger.error("Cannot run baseline-only evaluation for all domains")
            return False
        
        available_domains = discover_available_domains(args.verbose)
        if args.domain not in available_domains:
            logger.error(f"Invalid domain: {args.domain}")
            logger.error(f"Available: {', '.join(available_domains)}")
            return False
        
        success = run_baseline_only_evaluation(
            domain_name=args.domain,
            baseline_type=args.baseline_only,
            algorithm_config=algorithm_config,
            data_directory=args.data_directory,
            verbose=args.verbose,
        )
    elif args.domain == "all":
        # Run comprehensive evaluation for all domains
        success = run_all_domains_evaluation(
            algorithm_config=algorithm_config,
            data_directory=args.data_directory,
            verbose=args.verbose,
        )
    else:
        # Run comprehensive evaluation for single domain
        available_domains = discover_available_domains(args.verbose)
        if args.domain not in available_domains:
            logger.error(f"Invalid domain: {args.domain}")
            logger.error(f"Available: {', '.join(available_domains)}")
            return False
        
        success = run_single_evaluation(
            domain_name=args.domain,
            algorithm_config=algorithm_config,
            data_directory=args.data_directory,
            verbose=args.verbose,
        )
    
    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 