#!/usr/bin/env python3
"""
Bayesian Consensus-Difference Optimization Runner

This script uses Bayesian optimization (Gaussian Process with Expected Improvement) 
to efficiently find parameters that maximize consensus-difference scores.

Key Features:
- Adaptive Tchebycheff scalarization for balanced consensus-difference optimization
- TF-IDF vectorization with configurable max features
- Keyword filtering with configurable ratio
- Runtime bounds computation for domain-specific normalization

Uses robust C-metrics (consensus within segments) and D-metrics (difference between segments)
with fail-fast error handling and transparent decision rationale.
"""

import os
import sys
import time
import json
import numpy as np
import logging
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass
from tqdm import tqdm
import argparse
import concurrent.futures as _cf

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import Bayesian optimization dependencies
try:
    from skopt import gp_minimize
    from skopt.space import Real, Integer
    from skopt.acquisition import gaussian_ei
    from skopt.utils import use_named_args
    import skopt
    print("scikit-optimize imported successfully")
except ImportError:
    print("ERROR: scikit-optimize not installed. Install with: pip install scikit-optimize")
    sys.exit(1)

from core.data_loader import load_domain_data
from core.data_models import DomainData, Paper
from core.algorithm_config import AlgorithmConfig
from core.consensus_difference_metrics import (
    evaluate_segmentation_quality,
    SegmentationEvaluationResult,
)


def load_optimization_config():
    """Load optimization configuration from centralized JSON file."""
    config_path = "optimization_config.json"
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Optimization config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    return config


# Removed verbose metadata function - using simplified format


# Path to store consensus-difference-optimized parameters (Bayesian optimization)
CONSENSUS_DIFFERENCE_PARAMS_FILE = "results/optimization/optimized_parameters_bayesian.json"


class SuppressOutput:
    """Context manager to suppress stdout output during optimization."""
    
    def __init__(self, suppress_stdout=True):
        self.suppress_stdout = suppress_stdout
        self.original_stdout = None
        
    def __enter__(self):
        if self.suppress_stdout:
            self.original_stdout = sys.stdout
            sys.stdout = open(os.devnull, 'w')
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.suppress_stdout and self.original_stdout:
            sys.stdout.close()
            sys.stdout = self.original_stdout


@dataclass
class BayesianOptimizationConfig:
    """Configuration for Bayesian optimization."""
    
    # Load parameter bounds from centralized configuration
    def __post_init__(self):
        config = load_optimization_config()
        params = config["optimization_parameters"]
        
        self.direction_threshold_bounds = tuple(params["direction_threshold_bounds"])
        self.validation_threshold_bounds = tuple(params["validation_threshold_bounds"])
        self.similarity_min_segment_length_bounds = tuple(params["similarity_min_segment_length_bounds"])
        self.similarity_max_segment_length_bounds = tuple(params["similarity_max_segment_length_bounds"])
    
    # Bayesian optimization settings
    n_calls: int = 100  # Number of function evaluations (vs 10,000 for grid search)
    n_initial_points: int = 20  # Random exploration before GP model
    acq_func: str = "EI"  # Expected Improvement acquisition function
    random_state: int = 42
    
    def get_search_space(self):
        """Get scikit-optimize search space definition."""
        return [
            Real(self.direction_threshold_bounds[0], self.direction_threshold_bounds[1], 
                 name='direction_threshold'),
            Real(self.validation_threshold_bounds[0], self.validation_threshold_bounds[1], 
                 name='validation_threshold'),
            Integer(self.similarity_min_segment_length_bounds[0], self.similarity_min_segment_length_bounds[1], 
                   name='similarity_min_segment_length'),
            Integer(self.similarity_max_segment_length_bounds[0], self.similarity_max_segment_length_bounds[1], 
                   name='similarity_max_segment_length')
        ]


def research_vector_to_config(
    direction_threshold: float, 
    validation_threshold: float, 
    similarity_min_segment_length: int,
    similarity_max_segment_length: int,
    base_config: Optional[AlgorithmConfig] = None
) -> AlgorithmConfig:
    """Convert parameter values to algorithm configuration."""
    if base_config is None:
        base_config = AlgorithmConfig()

    return AlgorithmConfig(
        granularity=base_config.granularity,
        direction_threshold=float(direction_threshold),
        validation_threshold=float(validation_threshold),
        similarity_min_segment_length=int(similarity_min_segment_length),
        similarity_max_segment_length=int(similarity_max_segment_length),
        # Keep existing parameters at their defaults
        keyword_min_frequency=base_config.keyword_min_frequency,
        min_significant_keywords=base_config.min_significant_keywords,
        keyword_filtering_enabled=base_config.keyword_filtering_enabled,
        keyword_min_papers_ratio=base_config.keyword_min_papers_ratio,
        citation_boost=base_config.citation_boost,
        citation_support_window=base_config.citation_support_window,
        domain_name=base_config.domain_name,
    )


def consensus_difference_evaluation_bayesian(
    params: List[float],
    domain_data: DomainData,
    domain_name: str,
    base_config: Optional[AlgorithmConfig] = None,
) -> float:
    """
    Consensus & Difference evaluation function for Bayesian optimization.
    
    Args:
        params: [direction_threshold, validation_threshold, similarity_min_segment_length, similarity_max_segment_length]
        domain_data: Domain data for evaluation
        domain_name: Name of the domain
    
    Returns:
        Negative consensus-difference score (for minimization)
    """
    direction_threshold, validation_threshold, similarity_min_segment_length, similarity_max_segment_length = params
    
    # Validate parameter constraints
    if similarity_min_segment_length >= similarity_max_segment_length:
        return -1000.0  # Heavy penalty for invalid combinations
    
    config = research_vector_to_config(
        direction_threshold,
        validation_threshold,
        similarity_min_segment_length,
        similarity_max_segment_length,
        base_config=base_config,
    )

    try:
        # Run actual algorithm with suppressed output
        with SuppressOutput(suppress_stdout=True):
            # Import and run change detection to get segments
            from core.integration import run_change_detection
            
            segmentation_results, change_detection_result, shift_signals = run_change_detection(
                domain_data.domain_name, 
                granularity=config.granularity, 
                algorithm_config=config
            )
            
            # Extract segment information from results
            if segmentation_results and 'segments' in segmentation_results:
                segments = segmentation_results['segments']
                
                # Convert segments (year tuples) to lists of papers
                segment_papers = []
                for segment_years in segments:
                    start_year, end_year = segment_years
                    segment_paper_list = [
                        paper for paper in domain_data.papers
                        if start_year <= paper.pub_year <= end_year
                    ]
                    if segment_paper_list:  # Only add non-empty segments
                        segment_papers.append(tuple(segment_paper_list))
                
                # If no valid segments, create single segment with all papers
                if not segment_papers:
                    segment_papers = [tuple(domain_data.papers)]
            else:
                # No segments found, create single segment with all papers
                segment_papers = [tuple(domain_data.papers)]

        # Evaluate segmentation quality using comprehensive metrics from consensus_difference_metrics.py
        # Weights and aggregation method are now loaded automatically from optimization_config.json
        evaluation_result = evaluate_segmentation_quality(segment_papers, algorithm_config=config)
        
        # Return negative score for minimization (scikit-optimize minimizes by default)
        return -evaluation_result.final_score

    except Exception as e:
        # Return large penalty for failed evaluations
        return -1000.0


def optimize_consensus_difference_parameters_bayesian(
    domain_data: DomainData,
    domain_name: str,
    max_evaluations: int = 100,
    random_seed: int = None,
    base_config: Optional[AlgorithmConfig] = None,
) -> Dict[str, Any]:
    """
    Optimize parameters using Bayesian optimization for consensus-difference metrics.

    Returns:
        Dictionary with optimization results in same format as grid search
    """
    # Use domain-specific random seed
    if random_seed is None:
        domain_seed = hash(domain_name) % (2**31)
        if domain_seed < 0:
            domain_seed = -domain_seed
    else:
        domain_seed = random_seed + hash(domain_name) % 1000

    np.random.seed(domain_seed)

    # Ensure minimum evaluations for Bayesian optimization
    if max_evaluations < 20:
        print(f"    WARNING: Increasing evaluations from {max_evaluations} to 20 (minimum for Bayesian optimization)")
        max_evaluations = 20
    
    bo_config = BayesianOptimizationConfig(
        n_calls=max_evaluations,
        n_initial_points=min(20, max_evaluations),
        random_state=domain_seed
    )
    search_space = bo_config.get_search_space()

    print(f"    Optimizing {domain_name} using Bayesian optimization")
    print(f"    Domain seed: {domain_seed}")
    print(f"    Max evaluations: {max_evaluations}")
    print(f"    Algorithm: Gaussian Process + Expected Improvement")
    print(f"    Parameters: direction_threshold, validation_threshold, similarity_min_segment_length, similarity_max_segment_length")

    bounds = [
        bo_config.direction_threshold_bounds,
        bo_config.validation_threshold_bounds,
        bo_config.similarity_min_segment_length_bounds,
        bo_config.similarity_max_segment_length_bounds,
    ]
    
    print(f"    Direction threshold: [{bounds[0][0]:.3f}, {bounds[0][1]:.3f}]")
    print(f"    Validation threshold: [{bounds[1][0]:.3f}, {bounds[1][1]:.3f}]")
    print(f"    Min segment length: [{bounds[2][0]}, {bounds[2][1]}]")
    print(f"    Max segment length: [{bounds[3][0]}, {bounds[3][1]}]")

    # Track all evaluations for compatibility with grid search output format
    all_results = []
    best_score = -1000.0
    best_params = None
    evaluation_count = 0

    # Define objective function with progress tracking
    def objective_with_tracking(params):
        nonlocal evaluation_count, best_score, best_params, all_results
        
        evaluation_count += 1
        
        # Get negative score (for minimization)
        neg_score = consensus_difference_evaluation_bayesian(
            params, domain_data, domain_name, base_config=base_config
        )
        actual_score = -neg_score  # Convert back to positive for tracking
        
        # Update progress
        pbar.set_description(f"    Eval {evaluation_count}: dir={params[0]:.3f}, val={params[1]:.3f}, sim_len={int(params[2])}-{int(params[3])}, score={actual_score:.3f}")
        
        # Track best result
        if actual_score > best_score:
            best_score = actual_score
            best_params = params.copy()
            tqdm.write(f"    NEW BEST: Score={best_score:.3f} (dir={params[0]:.3f}, val={params[1]:.3f}, sim_len={int(params[2])}-{int(params[3])})")
        
        # Store result for tracking
        detailed_evaluation = {
            'score': actual_score,
            'consensus_score': 0.0,  # Will be filled in final verification
            'difference_score': 0.0
        }
        
        all_results.append({
            "direction_threshold": params[0],
            "validation_threshold": params[1],
            "similarity_min_segment_length": int(params[2]),
            "similarity_max_segment_length": int(params[3]),
            "score": actual_score,
            "detailed_evaluation": detailed_evaluation,
        })
        
        # Update progress bar
        pbar.update(1)
        
        return neg_score

    print(f"    Starting Bayesian optimization...")

    # Create progress bar
    pbar = tqdm(total=max_evaluations, desc="    Bayesian Opt", unit="eval", 
                bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]")

    try:
        # Run Bayesian optimization
        result = gp_minimize(
            func=objective_with_tracking,
            dimensions=search_space,
            n_calls=max_evaluations,
            n_initial_points=bo_config.n_initial_points,
            acq_func=bo_config.acq_func,
            random_state=domain_seed,
            verbose=False
        )
        
        # Close progress bar
        pbar.close()
        
        # Get detailed evaluation for best parameters using new metrics
        config = research_vector_to_config(*best_params, base_config=base_config)
        with SuppressOutput(suppress_stdout=True):
            # Run segmentation with best parameters
            from core.integration import run_change_detection
            
            segmentation_results, change_detection_result, shift_signals = run_change_detection(
                domain_data.domain_name, 
                granularity=config.granularity, 
                algorithm_config=config
            )
            
            # Extract segment information
            if segmentation_results and 'segments' in segmentation_results:
                segments = segmentation_results['segments']
                segment_papers = []
                for segment_years in segments:
                    start_year, end_year = segment_years
                    segment_paper_list = [
                        paper for paper in domain_data.papers
                        if start_year <= paper.pub_year <= end_year
                    ]
                    if segment_paper_list:
                        segment_papers.append(tuple(segment_paper_list))
                
                if not segment_papers:
                    segment_papers = [tuple(domain_data.papers)]
            else:
                segment_papers = [tuple(domain_data.papers)]
        
        # Get detailed evaluation using comprehensive metrics from consensus_difference_metrics.py
        # Aggregation method loaded automatically from optimization_config.json with env override support
        detailed_evaluation = evaluate_segmentation_quality(segment_papers, algorithm_config=config)
        final_best_score = detailed_evaluation.final_score
        
        final_best_detailed_evaluation = {
            'score': detailed_evaluation.final_score,
            'consensus_score': detailed_evaluation.consensus_score,
            'difference_score': detailed_evaluation.difference_score,
            'num_segments': detailed_evaluation.num_segments
        }
        
        if abs(final_best_score - best_score) > 0.001:
            print(f"    Final verification: {best_score:.3f} -> {final_best_score:.3f}")
            best_score = final_best_score

        # Prepare results in same format as grid search
        results = {
            "domain": domain_name,
            "best_parameters": {
                "direction_threshold": float(best_params[0]),
                "validation_threshold": float(best_params[1]),
                "similarity_min_segment_length": int(best_params[2]),
                "similarity_max_segment_length": int(best_params[3]),
            },
            "best_consensus_difference_score": float(best_score),
            "total_evaluations": evaluation_count,
            "domain_seed": domain_seed,
            "optimization_successful": True,
            "optimization_type": "bayesian_optimization",
            "algorithm_details": {
                "acquisition_function": bo_config.acq_func,
                "n_initial_points": bo_config.n_initial_points,
                "surrogate_model": "Gaussian Process",
                "library": "scikit-optimize"
            },
            "all_bayesian_results": all_results,
            "best_detailed_evaluation": final_best_detailed_evaluation,
            "convergence_info": {
                "func_vals": [-score for score in result.func_vals],  # Convert back to positive
                "x_iters": result.x_iters,
                "best_iteration": np.argmax([-score for score in result.func_vals])
            }
        }

        print(f"    Bayesian optimization complete: score={best_score:.3f}")
        print(f"    Best params: direction={best_params[0]:.3f}, validation={best_params[1]:.3f}, similarity_length={int(best_params[2])}-{int(best_params[3])}")

        return results

    except Exception as e:
        pbar.close()
        print(f"    ERROR: Bayesian optimization failed: {e}")
        return {
            "domain": domain_name,
            "optimization_successful": False,
            "error": str(e),
            "optimization_type": "bayesian_optimization"
        }


def save_consensus_difference_optimized_parameters(
    optimization_results: Dict[str, Dict[str, Any]],
    filename_suffix: str = ""
) -> None:
    """Save consensus-difference-optimized parameters to JSON file in simplified format."""
    
    # Allow custom filename suffix to distinguish between optimization methods
    if filename_suffix:
        base_name = CONSENSUS_DIFFERENCE_PARAMS_FILE.replace('.json', f'_{filename_suffix}.json')
    else:
        base_name = CONSENSUS_DIFFERENCE_PARAMS_FILE
    
    # Ensure results directory exists
    os.makedirs(os.path.dirname(base_name), exist_ok=True)

    # Format for saving - simplified format with only essential information
    formatted_results = {}
    
    for domain, result in optimization_results.items():
        if result.get("optimization_successful", False):
            # Save only essential information
            best_eval = result.get("best_detailed_evaluation", {})
            formatted_results[domain] = {
                "parameters": result["best_parameters"],
                "score": best_eval.get("score", 0.0),
                "consensus_score": best_eval.get("consensus_score", 0.0),
                "difference_score": best_eval.get("difference_score", 0.0),
                "num_segments": best_eval.get("num_segments", 0)
            }

    # Save simplified data
    save_data = {
        "optimized_parameters": formatted_results,
        "timestamp": time.strftime('%Y-%m-%d %H:%M:%S')
    }

    # Save timestamped version
    with open(base_name, "w") as f:
        json.dump(save_data, f, indent=2)

    # Save latest version in results root
    latest_path = "results/optimized_parameters_bayesian.json"
    os.makedirs("results", exist_ok=True)
    with open(latest_path, "w") as f:
        json.dump(save_data, f, indent=2)

    print(f"Optimized parameters saved to {base_name}")
    print(f"Latest version saved to {latest_path}")
    print(f"Saved results for {len(formatted_results)} domains")


def convert_dataframe_to_domain_data(df, domain_name: str) -> DomainData:
    """
    Convert DataFrame to DomainData object with all required fields.
    
    Phase 16 FEATURE-06: Supports phrase enrichment via YAKE when enabled in configuration.
    """
    # Load phrase enrichment configuration
    config = load_optimization_config()
    phrase_config = config.get("phrase_enrichment", {})
    
    # Priority order: environment variable > optimization_config.json
    phrase_enabled = os.getenv("PHRASE_ENRICHMENT")
    if phrase_enabled is not None:
        phrase_enabled = phrase_enabled.lower() == "true"
    else:
        phrase_enabled = phrase_config.get("enabled", False)
    
    top_k_phrases = phrase_config.get("top_k_phrases", 10)
    
    # Import YAKE utility if phrase enrichment is enabled
    if phrase_enabled:
        from core.keyword_utils import yake_phrases
        print(f"YAKE phrase enrichment enabled (top_k={top_k_phrases})")
    
    papers = []
    for idx, row in df.iterrows():
        # Extract keywords properly - handle pipe-separated format
        keywords_raw = row.get("keywords", "")
        if isinstance(keywords_raw, str):
            # Handle both comma and pipe separated keywords
            if "|" in keywords_raw:
                keywords_list = [k.strip() for k in keywords_raw.split("|") if k.strip()]
            else:
                keywords_list = [k.strip() for k in keywords_raw.split(",") if k.strip()]
        elif isinstance(keywords_raw, list):
            keywords_list = [str(k).strip() for k in keywords_raw if str(k).strip()]
        else:
            keywords_list = []

        # Phase 16 FEATURE-06: Apply phrase enrichment if enabled
        if phrase_enabled:
            content = str(row.get("content", row.get("abstract", "")))
            if content and content.strip():
                try:
                    # Extract YAKE phrases and append to keywords
                    extracted_phrases = yake_phrases(content, top_k=top_k_phrases)
                    keywords_list.extend(extracted_phrases)
                except Exception as e:
                    # Fail-fast: phrase extraction errors are critical
                    raise ValueError(f"YAKE phrase extraction failed for paper {idx} in {domain_name}: {e}")

        paper = Paper(
            id=str(row.get("id", f"{domain_name}_{idx}")),
            title=str(row.get("title", "")),
            content=str(row.get("content", row.get("abstract", ""))),
            pub_year=int(row.get("pub_year", row.get("year", 2000))),
            cited_by_count=int(row.get("cited_by_count", row.get("citations", 0))),
            keywords=tuple(keywords_list),
            children=tuple(),
            description=str(row.get("title", "")),
        )
        papers.append(paper)

    # Calculate year range
    if papers:
        min_year = min(p.pub_year for p in papers)
        max_year = max(p.pub_year for p in papers)
        year_range = (min_year, max_year)
    else:
        year_range = (2000, 2023)

    # Create domain data
    domain_data = DomainData(
        domain_name=domain_name,
        papers=tuple(papers),
        citations=tuple(),
        graph_nodes=tuple(),
        year_range=year_range,
        total_papers=len(papers),
    )

    return domain_data


def optimize_single_domain(
    domain_name: str,
    max_evaluations: int = 100,
    keyword_ratio: float = 0.05,
) -> Dict[str, Any]:
    """Optimize research quality parameters for a single domain using Bayesian optimization."""
    print(f"\nBAYESIAN OPTIMIZATION: {domain_name.upper()}")
    print("=" * 50)

    try:
        # Load configuration and determine data source
        config = load_optimization_config()
        citation_config = config.get("citation_enrichment", {})
        
        # Priority order: environment variable > optimization_config.json
        citation_enabled = os.getenv("CITATION_EDGE_ENRICHMENT")
        if citation_enabled is not None:
            citation_enabled = citation_enabled.lower() == "true"
        else:
            citation_enabled = citation_config.get("enabled", True)
        
        # Load domain data using appropriate method
        if citation_enabled:
            print(f"Loading citation-enriched {domain_name} data...")
            from core.data_loader import load_domain_data_enriched
            
            apply_year_filtering = citation_config.get("apply_year_filtering", False)
            domain_data = load_domain_data_enriched(
                domain=domain_name,
                apply_year_filtering=apply_year_filtering
            )
            print(f"Loaded {len(domain_data.papers)} papers ({domain_data.year_range[0]}-{domain_data.year_range[1]}) with citations")
        else:
            print(f"Loading CSV-based {domain_name} data...")
            df = load_domain_data(domain_name)

            if df is None or df.empty:
                print(f"ERROR: No data available for {domain_name}")
                return {"optimization_successful": False, "error": "No data available"}

            print(f"Loaded {len(df)} papers")
            
            # Convert to DomainData (without citations)
            domain_data = convert_dataframe_to_domain_data(df, domain_name)
            print(f"Processed {len(domain_data.papers)} papers ({domain_data.year_range[0]}-{domain_data.year_range[1]})")

        # Build base configuration with keyword ratio override
        base_config_override = AlgorithmConfig(
            keyword_min_papers_ratio=keyword_ratio,
            keyword_filtering_enabled=True,
            domain_name=domain_name,
        )

        # Run Bayesian optimization
        start_time = time.time()
        result = optimize_consensus_difference_parameters_bayesian(
            domain_data,
            domain_name,
            max_evaluations=max_evaluations,
            base_config=base_config_override,
        )
        execution_time = time.time() - start_time

        # Display results
        if result["optimization_successful"]:
            params = result["best_parameters"]
            score = result["best_consensus_difference_score"]

            print(f"\nOPTIMIZATION SUCCESS:")
            print(f"   Direction threshold: {params['direction_threshold']:.3f}")
            print(f"   Validation threshold: {params['validation_threshold']:.3f}")
            print(f"   Min segment length: {params['similarity_min_segment_length']}")
            print(f"   Max segment length: {params['similarity_max_segment_length']}")
            print(f"   Final score: {score:.3f}")
            print(f"   Execution time: {execution_time:.1f}s")
            print(f"   Total evaluations: {result['total_evaluations']}")            
            # Display key metrics
            if "best_detailed_evaluation" in result:
                detailed_eval = result["best_detailed_evaluation"]
                print(f"\nRESULTS SUMMARY:")
                print(f"   Consensus score: {detailed_eval['consensus_score']:.3f}")
                print(f"   Difference score: {detailed_eval['difference_score']:.3f}")
                print(f"   Number of segments: {detailed_eval['num_segments']}")

            # Add execution time to result
            result["execution_time"] = execution_time
        else:
            print(f"ERROR: Bayesian optimization failed for {domain_name}")

        return result

    except Exception as e:
        print(f"ERROR: Error optimizing {domain_name}: {e}")
        return {"optimization_successful": False, "error": str(e)}


def discover_available_domains() -> List[str]:
    """
    Dynamically discover all available domains by scanning the data/processed directory.
    
    Returns:
        List of domain names that have processed data files
    """
    processed_data_dir = "data/processed"
    
    if not os.path.exists(processed_data_dir):
        print(f"WARNING: Processed data directory not found: {processed_data_dir}")
        return []
    
    available_domains = []
    
    try:
        # Scan for *_processed.csv files
        for filename in os.listdir(processed_data_dir):
            if filename.endswith('_processed.csv'):
                # Extract domain name by removing the _processed.csv suffix
                domain_name = filename.replace('_processed.csv', '')
                available_domains.append(domain_name)
        
        # Sort domains alphabetically for consistent ordering
        available_domains.sort()
        
        if available_domains:
            print(f"Discovered {len(available_domains)} available domains:")
            for domain in available_domains:
                print(f"  {domain}")
        else:
            print("WARNING: No processed data files found in data/processed/")
        
        return available_domains
        
    except Exception as e:
        print(f"ERROR: Error scanning for available domains: {e}")
        return []


def main():
    """Main function."""
    # Discover available domains dynamically
    all_available_domains = discover_available_domains()
    
    if not all_available_domains:
        print("ERROR: No domains available for optimization. Please ensure processed data files exist in data/processed/")
        return
    
    if len(sys.argv) > 1:
        # Parse arguments
        requested_domains = []
        max_evals = 300
        
        for arg in sys.argv[1:]:
            if arg.startswith("--max-evals="):
                max_evals = int(arg.split("=")[1])
            elif not arg.startswith("--"):
                requested_domains.append(arg)
        
        if not requested_domains:
            print("Usage: python optimize_segmentation_bayesian.py [domain1] [domain2] ... [--max-evals=100]")
            print(f"Available domains: {', '.join(all_available_domains)}")
            return
        
        # Validate requested domains exist
        domains_to_optimize = []
        for domain in requested_domains:
            if domain in all_available_domains:
                domains_to_optimize.append(domain)
            else:
                print(f"WARNING: Domain '{domain}' not found in available domains. Skipping.")
        
        if not domains_to_optimize:
            print("ERROR: None of the requested domains are available.")
            print(f"Available domains: {', '.join(all_available_domains)}")
            return
            
        results = {}
        for domain in domains_to_optimize:
            result = optimize_single_domain(domain, max_evals)
            results[domain] = result

        # Save results if any successful
        successful_results = {
            domain: result
            for domain, result in results.items()
            if result.get("optimization_successful", False)
        }
        if successful_results:
            save_consensus_difference_optimized_parameters(successful_results)
            print(f"Finished optimization for {len(successful_results)} out of {len(domains_to_optimize)} domains")
        else:
            print("ERROR: No domains were successfully optimized.")
    else:
        # Optimize all available domains with default evaluations
        max_evals = 100  
        print(f"Running Bayesian optimization on all {len(all_available_domains)} available domains with {max_evals} evaluations each")
        print(f"Domains to optimize: {', '.join(all_available_domains)}")
        
        results = {}
        for domain in all_available_domains:
            result = optimize_single_domain(domain, max_evals)
            results[domain] = result
            
        # Save results
        successful_results = {
            domain: result
            for domain, result in results.items()
            if result.get("optimization_successful", False)
        }
        if successful_results:
            save_consensus_difference_optimized_parameters(successful_results)
            print(f"Finished optimization for {len(successful_results)} out of {len(all_available_domains)} domains")
        else:
            print("ERROR: No domains were successfully optimized.")


if __name__ == "__main__":
    import argparse, concurrent.futures as _cf

    parser = argparse.ArgumentParser(description="Bayesian optimisation for timeline segmentation")
    parser.add_argument("domains", nargs="*", help="Domain names to optimise (default: all)")
    parser.add_argument("--max-evals", type=int, default=100, dest="max_evals", help="Max GP evaluations per domain")
    parser.add_argument("--keyword-ratio", type=float, default=0.05, help="keyword_min_papers_ratio override (0.01–0.5)")
    parser.add_argument("--no-save", action="store_true", dest="no_save", help="Skip saving optimized parameter JSON (useful for experiments)")
    parser.add_argument("--suffix", type=str, default=None, help="Suffix for saved parameter JSON (optional)")
    parser.add_argument("--parallel", type=int, default=1, dest="n_workers", help="Number of parallel workers")
    parser.add_argument("--tfidf_max_features", type=int, default=None, help="Override tfidf_max_features for this run")
    parser.add_argument("--clean-text", action="store_true", dest="clean_text_enabled", help="Enable HTML/stop-word cleaning before TF-IDF")
    parser.add_argument("--silent", action="store_true", dest="silent", help="Suppress verbose output (still prints errors)")
    parser.add_argument("--best-out", type=str, default=None, dest="best_out", help="Path to write best score JSON for sweep scripts")

    args = parser.parse_args()

    all_domains = discover_available_domains()
    if not all_domains:
        sys.exit(1)

    domains_to_run = args.domains if args.domains else all_domains
    invalid = [d for d in domains_to_run if d not in all_domains]
    if invalid:
        print(f"ERROR: Invalid domains requested: {', '.join(invalid)}")
        sys.exit(1)

    if not args.silent:
        # Load config to show actual settings
        config = load_optimization_config()
        aggregation_method = config["consensus_difference_weights"]["aggregation_method"]
        vectorizer_type = config["text_vectorizer"]["type"]
        max_features = config["text_vectorizer"]["max_features"]
        
        print(f"Optimizing {len(domains_to_run)} domains:")
        print(f"  Keyword ratio: {args.keyword_ratio:.2f}")
        print(f"  Max evaluations: {args.max_evals}")
        print(f"  Workers: {args.n_workers}")
        print(f"  Aggregation: {aggregation_method}")
        print(f"  Vectorizer: {vectorizer_type} (max_features={max_features})")

    # Set environment variables for downstream metric calculations
    if args.tfidf_max_features is not None:
        os.environ["TFIDF_MAX_FEATURES"] = str(args.tfidf_max_features)
    if args.clean_text_enabled:
        os.environ["CLEAN_TEXT_ENABLED"] = "true"

    results = {}
    if args.n_workers > 1:
        with _cf.ProcessPoolExecutor(max_workers=args.n_workers) as pool:
            futures = {
                pool.submit(optimize_single_domain, d, args.max_evals, args.keyword_ratio): d for d in domains_to_run
            }
            for fut in _cf.as_completed(futures):
                domain = futures[fut]
                results[domain] = fut.result()
    else:
        for d in domains_to_run:
            results[d] = optimize_single_domain(d, args.max_evals, args.keyword_ratio)

    successful = {d: r for d, r in results.items() if r.get("optimization_successful", False)}
    if not successful:
        print("ERROR: No successful optimizations")
        sys.exit(1)

    # Optionally write best-score JSON for sweep aggregation
    if args.best_out:
        try:
            from pathlib import Path; import json as _json
            out_p = Path(args.best_out)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            # Assume `successful` has at least one domain
            dom, res = next(iter(successful.items()))
            best = {
                "domain": dom,
                "tfidf_max_features": int(os.getenv("TFIDF_MAX_FEATURES", 500)),
                "final_score": res.get("best_detailed_evaluation", {}).get("score")
            }
            with open(out_p, "w", encoding="utf-8") as f:
                _json.dump(best, f)
        except Exception as e:
            if not args.silent:
                print(f"WARNING: Failed to write best-out JSON: {e}")

    if not args.no_save:
        suffix = args.suffix if args.suffix else f"kwr{int(args.keyword_ratio*100):02d}"
        save_consensus_difference_optimized_parameters(successful, filename_suffix=suffix)
        if not args.silent:
            print(f"Finished optimization for {len(successful)} domains. Saved with suffix '{suffix}'.")
    else:
        if not args.silent:
            print(f"Finished optimization for {len(successful)} domains. Saving suppressed by --no-save flag.")