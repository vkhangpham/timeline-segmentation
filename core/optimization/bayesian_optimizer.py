"""Bayesian optimization for timeline segmentation parameters.

Simple functional approach using scikit-optimize for parameter optimization.
"""

import time
from typing import Dict, Any, List, Callable

from skopt import gp_minimize
from skopt.space import Real, Integer, Categorical

from ..utils.logging import get_logger


def run_bayesian_optimization(
    config: Dict[str, Any],
    objective_function: Callable,
    domain_name: str,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Run Bayesian optimization for parameter tuning.

    Args:
        config: Configuration dictionary from YAML
        objective_function: Function to optimize (returns trial result dict)
        domain_name: Domain name for logging
        verbose: Enable verbose logging

    Returns:
        Dictionary with optimization results
    """
    logger = get_logger(__name__, verbose, domain_name)

    # Extract search parameters
    search_config = config.get("search", {})
    n_calls = search_config.get("n_calls", 50)
    n_initial_points = search_config.get("n_initial_points", 10)

    # Acquisition function and parallelism settings
    acquisition_function = search_config.get("acquisition_function", "EI")

    execution_cfg = config.get("execution", {})
    max_workers = execution_cfg.get("max_workers", 1)
    # scikit-optimize uses n_jobs for parallel objective evaluation (>=0.9)
    n_jobs = max_workers if isinstance(max_workers, int) and max_workers > 1 else 1

    # Build parameter space
    parameter_space = _build_parameter_space(config)
    parameter_names = list(parameter_space.keys())
    dimensions = [parameter_space[name] for name in parameter_names]

    if verbose:
        logger.info(
            f"Starting Bayesian optimization with {len(parameter_names)} parameters"
        )
        logger.info(f"Max calls: {n_calls}, Initial points: {n_initial_points}")

    # Track results
    trial_results = []
    best_score = float("-inf")
    best_score_progress: List[float] = []  # Track incumbent after each trial

    def objective_wrapper(params):
        """Wrapper for objective function."""
        param_dict = dict(zip(parameter_names, params))
        trial_id = len(trial_results)

        result = objective_function(param_dict, trial_id)
        trial_results.append(result)

        score = result["objective_score"]
        nonlocal best_score
        if score > best_score:
            best_score = score
            if verbose:
                logger.info(f"New best score: {score:.3f} at trial {trial_id}")

        # Log incumbent progression after each evaluation
        best_score_progress.append(best_score)

        return -score  # Negative for minimization

    start_time = time.time()

    # Run optimization with configurable acquisition function & parallelism
    gp_minimize(
        func=objective_wrapper,
        dimensions=dimensions,
        n_calls=n_calls,
        n_initial_points=n_initial_points,
        acq_func=acquisition_function,
        n_jobs=n_jobs,
        random_state=42,
    )

    total_time = time.time() - start_time

    # Find best result
    best_result = max(trial_results, key=lambda x: x["objective_score"])

    if verbose:
        logger.info(
            f"Optimization complete: {len(trial_results)} trials in {total_time:.1f}s"
        )
        logger.info(f"Best score: {best_result['objective_score']:.3f}")

    # Save incumbent-curve plot
    try:
        from pathlib import Path
        import matplotlib.pyplot as plt

        output_dir = Path("results/optimization_logs")
        output_dir.mkdir(parents=True, exist_ok=True)
        plot_path = output_dir / f"{domain_name}_best_score.png"

        plt.figure(figsize=(8, 4))
        plt.plot(best_score_progress, marker="o", linewidth=1.2)
        plt.title(f"Best Score vs Trial – {domain_name}")
        plt.xlabel("Trial")
        plt.ylabel("Best Objective Score")
        plt.grid(True, linestyle="--", alpha=0.4)
        plt.tight_layout()
        plt.savefig(plot_path, dpi=150)
        plt.close()
    except Exception as plot_err:
        # Silently ignore plotting errors to keep optimisation robust
        logger.warning(f"Failed to save best-score plot: {plot_err}")
        plot_path = None

    return {
        "best_result": best_result,
        "all_results": trial_results,
        "n_calls": len(trial_results),
        "total_time": total_time,
        "progress": best_score_progress,
        "best_score_plot": str(plot_path) if plot_path else None,
    }


def _build_parameter_space(config: Dict[str, Any]) -> Dict[str, Any]:
    """Build scikit-optimize parameter space from configuration.

    Automatically handles any parameter defined in the YAML, whether it's
    an existing AlgorithmConfig field or not.
    """
    space = {}
    parameters = config.get("parameters", {})

    for param_name, param_spec in parameters.items():
        param_type = param_spec.get("type", "float")
        param_range = param_spec.get("range")
        param_choices = param_spec.get("choices")

        if param_choices:  # Categorical parameter
            space[param_name] = Categorical(param_choices, name=param_name)
        elif param_range:  # Numeric parameter
            low, high = param_range
            if param_type == "int":
                space[param_name] = Integer(low, high, name=param_name)
            else:
                space[param_name] = Real(low, high, name=param_name)

    return space
