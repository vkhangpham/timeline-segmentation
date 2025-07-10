#!/usr/bin/env python3
"""Domain-specific parameter optimization for timeline segmentation.

This script uses Bayesian optimization to find optimal parameters for each domain.
"""

import argparse
import csv
import json
import time
from pathlib import Path
from typing import List, Dict, Any, Callable
import sys

from tqdm import tqdm

# Check if we need to suppress logging (if --verbose is not in args)
if '--verbose' not in sys.argv:
    # Import logging utility first and set suppress mode before any core imports
    from core.utils.logging import set_suppress_console_logging
    set_suppress_console_logging(True)

from core.utils.config import AlgorithmConfig
from core.optimization.optimization import score_trial, clear_cache
from core.optimization.bayesian_optimizer import run_bayesian_optimization
from core.optimization.optimization_config import load_config
from core.utils.general import discover_available_domains
from core.utils.logging import configure_global_logging, get_logger


class OptimizationProgressTracker:
    """Tracks optimization progress with tqdm and best score updates."""
    
    def __init__(self, total_trials: int, domain_name: str, optimization_config: Dict[str, Any] = None):
        self.domain_name = domain_name
        self.best_score = float('-inf')
        self.best_params = {}
        self.optimization_config = optimization_config or {}
        
        # Create progress bar
        self.pbar = tqdm(
            total=total_trials,
            desc=f"Optimizing {domain_name}",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] Best: {postfix}",
            postfix="Starting..."
        )
    
    def update(self, trial_result: Dict[str, Any]):
        """Update progress bar with trial result."""
        score = trial_result["objective_score"]
        params = trial_result["parameters"]
        
        # Update best score if improved
        if score > self.best_score:
            self.best_score = score
            self.best_params = params.copy()
        
        # Format current trial parameters for display (always show current, not just best)
        current_params = self._format_params(params)
        current_postfix = f"{score:.3f} | Current: {current_params}"
        
        # If we have a best score, add it to the display
        if self.best_score > float('-inf'):
            best_params = self._format_params(self.best_params)
            current_postfix += f" | Best: {self.best_score:.3f} ({best_params})"
            
        self.pbar.set_postfix_str(current_postfix)
        self.pbar.update(1)
    
    def _format_params(self, params: Dict[str, Any]) -> str:
        """Format parameters for compact display."""
        if not params:
            return "N/A"
        
        # Get all parameters from optimization config for comprehensive display
        param_config = self.optimization_config.get("parameters", {})
        
        formatted_params = []
        for param_name, param_value in params.items():
            # Create short abbreviations for common parameters
            param_abbrevs = {
                "direction_change_threshold": "dir_thresh",
                "score_distribution_window_years": "window",
                "citation_confidence_boost": "cit_boost",
                "outlier_threshold": "outlier",
                "min_segment_length": "min_len",
                "max_segment_length": "max_len",
                "change_point_threshold": "cp_thresh",
                "confidence_threshold": "conf_thresh",
                "smoothing_window": "smooth",
                "regularization_strength": "reg",
                "learning_rate": "lr",
                "batch_size": "batch",
                "num_epochs": "epochs",
            }
            
            abbrev = param_abbrevs.get(param_name, param_name[:8])
            
            # Format value based on type
            if isinstance(param_value, float):
                formatted_value = f"{param_value:.3f}"
            elif isinstance(param_value, int):
                formatted_value = str(param_value)
            else:
                formatted_value = str(param_value)
            
            formatted_params.append(f"{abbrev}={formatted_value}")
        
        return ", ".join(formatted_params)
    
    def close(self):
        """Close progress bar."""
        self.pbar.close()


def create_objective_function(
    domain_name: str,
    base_config: AlgorithmConfig,
    optimization_config: Dict[str, Any],
    progress_tracker: OptimizationProgressTracker = None,
    data_directory: str = "resources",
    verbose: bool = False,
) -> Callable[[Dict[str, Any], int], Dict[str, Any]]:
    """Create objective function for optimization.

    Args:
        domain_name: Name of the domain
        base_config: Base algorithm configuration
        optimization_config: Optimization configuration
        progress_tracker: Progress tracker for tqdm updates
        data_directory: Directory containing domain data
        verbose: Enable verbose logging

    Returns:
        Objective function that takes (parameters, trial_id) and returns trial result dict
    """

    def objective_function(parameters: Dict[str, Any], trial_id: int) -> Dict[str, Any]:
        result = score_trial(
            domain_name=domain_name,
            parameter_overrides=parameters,
            base_config=base_config,
            trial_id=trial_id,
            optimization_config=optimization_config,
            data_directory=data_directory,
            verbose=verbose,  # Only verbose if explicitly requested
        )
        
        # Update progress tracker if provided
        if progress_tracker:
            progress_tracker.update(result)
            
        return result

    return objective_function


def save_trial_results(
    results: List[Dict[str, Any]],
    domain_name: str,
    optimization_config: Dict[str, Any] = None,
    verbose: bool = False,
) -> str:
    """Save trial results to CSV file with dynamic parameter columns.

    Args:
        results: List of trial result dictionaries
        domain_name: Domain name for filename
        optimization_config: Optimization configuration for parameter names
        verbose: Enable verbose logging

    Returns:
        Path to saved CSV file
    """
    logger = get_logger(__name__, verbose, domain_name)

    output_dir = Path("results/optimization_logs")
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_file = output_dir / f"{domain_name}.csv"

    # Base fieldnames that are always present
    base_fieldnames = [
        "trial_id",
        "objective_score",
        "cohesion_score",
        "separation_score",
        "num_segments",
        "boundary_f1",
        "segment_f1",
        "runtime_seconds",
    ]
    
    # Get parameter names dynamically from optimization config
    parameter_names = []
    if optimization_config and "parameters" in optimization_config:
        parameter_names = list(optimization_config["parameters"].keys())
    elif results:
        # Fallback: extract parameter names from actual results
        parameter_names = list(results[0].get("parameters", {}).keys())
    
    # Combine all fieldnames
    fieldnames = base_fieldnames + parameter_names + ["error_message"]

    with open(csv_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for result in results:
            row = {
                "trial_id": result["trial_id"],
                "objective_score": result["objective_score"],
                "cohesion_score": result["cohesion_score"],
                "separation_score": result["separation_score"],
                "num_segments": result["num_segments"],
                "boundary_f1": result["boundary_f1"],
                "segment_f1": result["segment_f1"],
                "runtime_seconds": result["runtime_seconds"],
                "error_message": result["error_message"] or "",
            }

            # Add parameter values dynamically
            for param_name in parameter_names:
                row[param_name] = result["parameters"].get(param_name, "")

            writer.writerow(row)

    if verbose:
        logger.info(f"Saved {len(results)} trial results to {csv_file}")
        logger.info(f"Optimized parameters: {', '.join(parameter_names)}")

    return str(csv_file)


def save_best_config(
    best_result: Dict[str, Any],
    domain_name: str,
    verbose: bool = False,
) -> str:
    """Save best configuration to JSON file.

    Args:
        best_result: Best trial result dictionary
        domain_name: Domain name for filename
        verbose: Enable verbose logging

    Returns:
        Path to saved JSON file
    """
    logger = get_logger(__name__, verbose, domain_name)

    output_dir = Path("results/optimized_params")
    output_dir.mkdir(parents=True, exist_ok=True)

    json_file = output_dir / f"{domain_name}.json"

    # Convert numpy types to native Python types for JSON serialization
    def convert_numpy_types(obj):
        """Convert numpy types to native Python types."""
        import numpy as np

        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(v) for v in obj]
        return obj

    config_data = {
        "domain_name": domain_name,
        "optimization_date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "best_objective_score": best_result["objective_score"],
        "best_parameters": convert_numpy_types(best_result["parameters"]),
        "validation_metrics": {
            "boundary_f1": best_result["boundary_f1"],
            "segment_f1": best_result["segment_f1"],
            "num_segments": best_result["num_segments"],
        },
        "performance": {
            "cohesion_score": best_result["cohesion_score"],
            "separation_score": best_result["separation_score"],
            "runtime_seconds": best_result["runtime_seconds"],
        },
    }

    with open(json_file, "w") as f:
        json.dump(config_data, f, indent=2)

    if verbose:
        logger.info(f"Saved best configuration to {json_file}")

    return str(json_file)


def optimize_domain(
    domain_name: str,
    optimization_config: Dict[str, Any] = None,
    data_directory: str = "resources",
    verbose: bool = False,
) -> tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Optimize parameters for a single domain using Bayesian optimization.

    Args:
        domain_name: Name of the domain to optimize
        optimization_config: Optimization configuration dictionary
        data_directory: Directory containing domain data
        verbose: Enable verbose logging

    Returns:
        Tuple of (best_result, all_results)
    """
    if optimization_config is None:
        optimization_config = load_config(domain_name=domain_name)

    search_config = optimization_config.get("search", {})
    n_calls = search_config.get("n_calls", 50)

    # Only show detailed logs if verbose
    if verbose:
        logger = get_logger(__name__, verbose, domain_name)
        logger.info(f"Starting Bayesian optimization for {domain_name}")
        logger.info(f"Max trials: {n_calls}")
    else:
        print(f"Starting optimization for {domain_name} ({n_calls} trials)")

    # Load base algorithm configuration
    base_config = AlgorithmConfig.from_config_file(domain_name=domain_name)

    # Clear any cached data
    clear_cache()

    # Create progress tracker (only if not verbose)
    progress_tracker = None if verbose else OptimizationProgressTracker(n_calls, domain_name, optimization_config)

    # Create objective function
    objective_function = create_objective_function(
        domain_name=domain_name,
        base_config=base_config,
        optimization_config=optimization_config,
        progress_tracker=progress_tracker,
        data_directory=data_directory,
        verbose=verbose,
    )

    try:
        # Run Bayesian optimization
        if verbose:
            logger.info("Using Bayesian optimization")
            
        result = run_bayesian_optimization(
            config=optimization_config,
            objective_function=objective_function,
            domain_name=domain_name,
            verbose=verbose,
        )

        best_result = result["best_result"]
        all_results = result["all_results"]
        best_score_plot = result.get("best_score_plot")

        # Filter out failed trials for final statistics
        fail_score = optimization_config.get("scoring", {}).get("fail_score", -10.0)
        valid_results = [r for r in all_results if r["objective_score"] > fail_score]

        if not valid_results:
            raise RuntimeError("All optimization trials failed!")

        if verbose:
            logger.info("=== Optimization Complete ===")
            logger.info(f"Total trials: {len(all_results)}")
            logger.info(f"Successful trials: {len(valid_results)}")
            logger.info(f"Best score: {best_result['objective_score']:.3f}")
            logger.info(f"Best parameters: {best_result['parameters']}")
        else:
            print(f"\nOptimization complete! Best score: {best_result['objective_score']:.3f}")
            if best_score_plot:
                print(f"Best-score curve saved to: {best_score_plot}")

        return best_result, all_results
        
    finally:
        # Always close progress tracker if it exists
        if progress_tracker:
            progress_tracker.close()


def main():
    """Main optimization execution."""
    parser = argparse.ArgumentParser(
        description="Domain-specific parameter optimization using Bayesian optimization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/optimize_domain.py --domain deep_learning
  python scripts/optimize_domain.py --domain applied_mathematics --verbose
  python scripts/optimize_domain.py --domain computer_vision --config custom_config.yaml
        """,
    )

    parser.add_argument(
        "--domain",
        type=str,
        required=True,
        help="Domain to optimize",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to optimization config YAML file (default: config/optimization.yaml)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging (default: show only tqdm progress)",
    )

    args = parser.parse_args()

    # Validate domain
    available_domains = discover_available_domains(args.verbose)
    if args.domain not in available_domains:
        print(f"Error: Invalid domain '{args.domain}'")
        print(f"Available domains: {', '.join(available_domains)}")
        return False

    # Load optimization configuration
    optimization_config = load_config(
        config_path=args.config,
        domain_name=args.domain,
    )

    # Configure logging - only verbose if requested
    if args.verbose:
        configure_global_logging(verbose=True, domain_name=args.domain)

    # Run optimization
    best_result, all_results = optimize_domain(
        domain_name=args.domain,
        optimization_config=optimization_config,
        verbose=args.verbose,
    )

    # Save results
    csv_path = save_trial_results(all_results, args.domain, optimization_config, args.verbose)
    json_path = save_best_config(best_result, args.domain, args.verbose)

    print(f"\n{'='*50}")
    print(f"OPTIMIZATION COMPLETE: {args.domain}")
    print(f"{'='*50}")
    print(f"Best objective score: {best_result['objective_score']:.3f}")
    print(f"Best parameters: {best_result['parameters']}")
    print(f"Results saved to:")
    print(f"  - {csv_path}")
    print(f"  - {json_path}")

    return True


if __name__ == "__main__":
    main()
