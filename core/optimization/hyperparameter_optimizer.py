#!/usr/bin/env python3
"""
Hyperparameter Optimizer with AcademicPeriod Structures
=======================================================

High-performance Bayesian optimization for timeline segmentation hyperparameters.
Uses pre-computed AcademicPeriod structures for fast objective function evaluation.

Key Features:
- Gaussian Process optimization with scikit-optimize
- Fast evaluation using pre-computed AcademicPeriod structures
- Comprehensive result tracking and persistence
- Configurable search spaces and acquisition functions
- Progress monitoring and convergence analysis

Optimizes:
- direction_threshold: Threshold for detecting directional changes
- validation_threshold: Threshold for validating shift signals

Performance:
- Leverages pre-computed academic structures for efficient evaluation
- Optimized boundary segmentation with objective function integration
"""

import numpy as np
import json
import pickle
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
import warnings

# Suppress skopt warnings
warnings.filterwarnings("ignore", category=UserWarning, module="skopt")

from skopt import gp_minimize
from skopt.space import Real
from skopt.acquisition import gaussian_ei

import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.data.data_processing import load_domain_data
from core.optimization.objective_function import compute_objective_function
from core.segmentation.shift_signals import detect_boundary_years
from core.segmentation.boundary import create_segments_from_boundary_years
from core.utils.config import AlgorithmConfig


@dataclass
class HyperparameterConfig:
    """Configuration for hyperparameter optimization."""

    # Search space
    direction_threshold_range: Tuple[float, float] = (0.05, 0.8)
    validation_threshold_range: Tuple[float, float] = (0.2, 0.9)

    # Optimization settings
    n_calls: int = 50
    n_initial_points: int = 10
    acquisition_function: str = "EI"  # Expected Improvement
    random_state: int = 42

    # Evaluation settings
    use_keyword_cache: bool = True
    use_boundary_optimization: bool = True
    boundary_search_window: int = 2

    # Fast mode settings (for initial exploration)
    fast_mode: bool = False
    fast_mode_paper_fraction: float = 0.3
    fast_mode_disable_boundary_opt: bool = True


@dataclass
class OptimizationResult:
    """Result of a single hyperparameter evaluation."""

    direction_threshold: float
    validation_threshold: float
    final_score: float
    cohesion_score: float
    separation_score: float
    num_segments: int
    num_signals_detected: int
    num_signals_validated: int
    evaluation_time: float
    used_keyword_cache: bool
    used_boundary_optimization: bool

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class HyperparameterOptimizer:
    """
    High-performance Bayesian optimizer for timeline segmentation hyperparameters.
    """

    def __init__(self, domain_name: str, config: HyperparameterConfig = None):
        """
        Initialize hyperparameter optimizer.

        Args:
            domain_name: Name of domain to optimize
            config: Optimization configuration
        """
        self.domain_name = domain_name
        self.config = config or HyperparameterConfig()

        # Load domain data using the new API
        print(f"Loading domain data for {domain_name}...")

        # Create a dummy config for loading (we'll override parameters later)
        from core.utils.config import AlgorithmConfig

        dummy_config = AlgorithmConfig.from_config_file(domain_name=domain_name)

        success, academic_years, error_message = load_domain_data(
            domain_name=domain_name, algorithm_config=dummy_config, verbose=True
        )

        if not success:
            raise RuntimeError(f"Failed to load {domain_name}: {error_message}")

        self.academic_years = academic_years

        # Keyword cache is no longer needed with AcademicPeriod objects
        self.keyword_cache = None

        # Fast mode: create paper subset
        self.fast_mode_papers = None
        if self.config.fast_mode:
            self.fast_mode_papers = self._create_fast_mode_subset()

        # Optimization tracking
        self.results: List[OptimizationResult] = []
        self.best_result: Optional[OptimizationResult] = None
        self.start_time = None

        # Define search space
        self.search_space = [
            Real(
                self.config.direction_threshold_range[0],
                self.config.direction_threshold_range[1],
                name="direction_threshold",
            ),
            Real(
                self.config.validation_threshold_range[0],
                self.config.validation_threshold_range[1],
                name="validation_threshold",
            ),
        ]

        print(f"Optimizer initialized for {domain_name}")
        print(
            f"Search space: direction_threshold {self.config.direction_threshold_range}"
        )
        print(
            f"             validation_threshold {self.config.validation_threshold_range}"
        )
        print(f"Fast mode: {'enabled' if self.config.fast_mode else 'disabled'}")
        print(
            f"Boundary optimization: {'enabled' if self.config.use_boundary_optimization else 'disabled'}"
        )

    def _create_fast_mode_subset(self) -> List:
        """Fast mode not supported with new API - returns empty list."""
        print("Fast mode not supported with new academic year API")
        return []

    def _evaluate_hyperparameters(
        self, direction_threshold: float, validation_threshold: float
    ) -> OptimizationResult:
        """
        Evaluate a single hyperparameter configuration.

        Args:
            direction_threshold: Threshold for detecting directional changes
            validation_threshold: Threshold for validating shift signals

        Returns:
            OptimizationResult with complete evaluation metrics
        """
        start_time = time.time()

        # Use academic years directly (no fast mode support for now)
        academic_years = self.academic_years

        # Create algorithm config by loading from file and overriding specific parameters
        from core.utils.config import AlgorithmConfig
        from dataclasses import replace

        # Load base config from file
        base_config = AlgorithmConfig.from_config_file(domain_name=self.domain_name)

        # Create new config with overridden parameters for optimization
        algo_config = replace(
            base_config,
            direction_threshold=direction_threshold,
            validation_threshold=validation_threshold,
            boundary_optimization_enabled=(
                self.config.use_boundary_optimization
                and not self.config.fast_mode_disable_boundary_opt
            ),
            boundary_search_window=self.config.boundary_search_window,
        )

        try:
            # Step 1: Detect and validate shift signals
            validated_signals = detect_boundary_years(
                academic_years=academic_years,
                domain_name=self.domain_name,
                algorithm_config=algo_config,
                verbose=False,
            )

            if not validated_signals:
                # No valid signals found
                return OptimizationResult(
                    direction_threshold=direction_threshold,
                    validation_threshold=validation_threshold,
                    final_score=0.0,
                    cohesion_score=0.0,
                    separation_score=0.0,
                    num_segments=1,
                    num_signals_detected=0,
                    num_signals_validated=0,
                    evaluation_time=time.time() - start_time,
                    used_keyword_cache=False,  # No longer used
                    used_boundary_optimization=False,
                )

            # Step 2: Create segments using academic years
            periods = create_segments_from_boundary_years(
                boundary_academic_years=validated_signals,
                academic_years=academic_years,
                algorithm_config=algo_config,
                verbose=False,
            )

            # Step 3: Evaluate timeline quality using AcademicPeriod objects
            result = compute_objective_function(periods, algo_config, verbose=False)

            return OptimizationResult(
                direction_threshold=direction_threshold,
                validation_threshold=validation_threshold,
                final_score=result.final_score,
                cohesion_score=result.cohesion_score,
                separation_score=result.separation_score,
                num_segments=result.num_segments,
                num_signals_detected=len(validated_signals),
                num_signals_validated=len(validated_signals),
                evaluation_time=time.time() - start_time,
                used_keyword_cache=False,  # No longer used
                used_boundary_optimization=algo_config.boundary_optimization_enabled,
            )

        except Exception as e:
            print(
                f"Error evaluating hyperparameters ({direction_threshold:.3f}, {validation_threshold:.3f}): {e}"
            )
            return OptimizationResult(
                direction_threshold=direction_threshold,
                validation_threshold=validation_threshold,
                final_score=-1.0,  # Penalty for failed evaluation
                cohesion_score=0.0,
                separation_score=0.0,
                num_segments=0,
                num_signals_detected=0,
                num_signals_validated=0,
                evaluation_time=time.time() - start_time,
                used_keyword_cache=False,  # No longer used
                used_boundary_optimization=False,
            )

    def _create_fast_domain_data(self):
        """Create domain data object for fast mode."""
        from core.data.data_models import DomainData

        if not self.fast_mode_papers:
            return self.domain_data

        # Create new domain data with subset of papers
        years = [p.pub_year for p in self.fast_mode_papers]
        year_range = (min(years), max(years))

        return DomainData(
            domain_name=self.domain_data.domain_name,
            papers=self.fast_mode_papers,
            citations=self.domain_data.citations,  # Keep all citations
            graph_nodes=self.domain_data.graph_nodes,  # Keep all graph nodes
            year_range=year_range,
            total_papers=len(self.fast_mode_papers),
        )

    def _objective_function(self, params) -> float:
        """
        Objective function for Bayesian optimization.

        Args:
            params: List of parameters [direction_threshold, validation_threshold]

        Returns:
            Negative final score (for minimization)
        """
        direction_threshold, validation_threshold = params
        result = self._evaluate_hyperparameters(
            direction_threshold, validation_threshold
        )
        self.results.append(result)

        # Update best result
        if (
            self.best_result is None
            or result.final_score > self.best_result.final_score
        ):
            self.best_result = result

        # Print progress
        elapsed = time.time() - self.start_time
        print(f"Evaluation {len(self.results)}/{self.config.n_calls}:")
        print(
            f"  Params: direction={direction_threshold:.3f}, validation={validation_threshold:.3f}"
        )
        print(
            f"  Score: {result.final_score:.4f} (cohesion={result.cohesion_score:.3f}, separation={result.separation_score:.3f})"
        )
        print(
            f"  Signals: {result.num_signals_detected} detected, {result.num_signals_validated} validated, {result.num_segments} segments"
        )
        print(f"  Time: {result.evaluation_time:.2f}s (total: {elapsed:.1f}s)")
        print(f"  Best so far: {self.best_result.final_score:.4f}")
        print()

        # Return negative score for minimization
        return -result.final_score

    def optimize(self) -> Dict[str, Any]:
        """
        Run Bayesian optimization to find best hyperparameters.

        Returns:
            Dictionary with optimization results
        """
        print(f"Starting Bayesian optimization for {self.domain_name}")
        print(f"Configuration: {self.config}")
        print()

        self.start_time = time.time()

        # Run optimization
        result = gp_minimize(
            func=self._objective_function,
            dimensions=self.search_space,
            n_calls=self.config.n_calls,
            n_initial_points=self.config.n_initial_points,
            acq_func=self.config.acquisition_function,
            random_state=self.config.random_state,
        )

        total_time = time.time() - self.start_time

        # Extract best parameters
        best_direction_threshold = result.x[0]
        best_validation_threshold = result.x[1]
        best_score = -result.fun

        # Create comprehensive results
        optimization_results = {
            "domain_name": self.domain_name,
            "config": asdict(self.config),
            "best_hyperparameters": {
                "direction_threshold": best_direction_threshold,
                "validation_threshold": best_validation_threshold,
            },
            "best_score": best_score,
            "best_result": self.best_result.to_dict() if self.best_result else None,
            "optimization_info": {
                "total_evaluations": len(self.results),
                "total_time": total_time,
                "average_time_per_evaluation": (
                    total_time / len(self.results) if self.results else 0
                ),
                "convergence_values": [-y for y in result.func_vals],
                "parameter_history": [[x[0], x[1]] for x in result.x_iters],
            },
            "all_results": [r.to_dict() for r in self.results],
            "timestamp": datetime.now().isoformat(),
        }

        print(f"Optimization complete!")
        print(
            f"Best hyperparameters: direction_threshold={best_direction_threshold:.3f}, validation_threshold={best_validation_threshold:.3f}"
        )
        print(f"Best score: {best_score:.4f}")
        print(f"Total time: {total_time:.1f}s ({len(self.results)} evaluations)")
        print(f"Average time per evaluation: {total_time/len(self.results):.2f}s")

        return optimization_results

    def save_results(
        self,
        results: Dict[str, Any],
        output_dir: str = "results/hyperparameter_optimization",
    ):
        """Save optimization results to files."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save JSON results
        json_path = (
            output_path
            / f"hyperparameter_optimization_{self.domain_name}_{timestamp}.json"
        )
        with open(json_path, "w") as f:
            json.dump(results, f, indent=2)

        # Save latest results
        latest_path = output_path / f"latest_{self.domain_name}_optimization.json"
        with open(latest_path, "w") as f:
            json.dump(results, f, indent=2)

        print(f"Results saved to {json_path}")
        print(f"Latest results saved to {latest_path}")

        return json_path, latest_path


def main():
    """Run hyperparameter optimization with keyword cache."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Hyperparameter optimization with keyword cache"
    )
    parser.add_argument(
        "--domain",
        type=str,
        default="applied_mathematics",
        help="Domain name to optimize",
    )
    parser.add_argument(
        "--n-calls", type=int, default=30, help="Number of optimization calls"
    )
    parser.add_argument(
        "--fast-mode", action="store_true", help="Use fast mode for initial exploration"
    )
    parser.add_argument("--no-cache", action="store_true", help="Disable keyword cache")
    parser.add_argument(
        "--no-boundary-opt", action="store_true", help="Disable boundary optimization"
    )

    args = parser.parse_args()

    # Create configuration
    config = HyperparameterConfig(
        n_calls=args.n_calls,
        fast_mode=args.fast_mode,
        use_keyword_cache=not args.no_cache,
        use_boundary_optimization=not args.no_boundary_opt,
    )

    print(f"Hyperparameter Optimization with Keyword Cache")
    print(f"Domain: {args.domain}")
    print(f"Configuration: {config}")
    print()

    # Activate conda environment
    import subprocess

    try:
        subprocess.run(["conda", "activate", "timeline"], shell=True, check=False)
    except:
        pass

    # Run optimization
    optimizer = HyperparameterOptimizer(args.domain, config)
    results = optimizer.optimize()

    # Save results
    json_path, latest_path = optimizer.save_results(results)

    print(f"\nOptimization complete! Results saved to {json_path}")

    return results


if __name__ == "__main__":
    results = main()
