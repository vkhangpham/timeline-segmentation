#!/usr/bin/env python3
"""
Bayesian Consensus-Difference Optimization Runner

This script uses Bayesian optimization (Gaussian Process with Expected Improvement) 
to efficiently find parameters that maximize consensus-difference scores using Phase 15 metrics. 

Uses the new robust C-metrics (consensus within segments) and D-metrics (difference between segments)
instead of the semantic citation approach. Designed to be 95%+ faster than grid search.

Follows the Phase 15 consensus-difference metrics framework.
"""

import os
import sys
import time
import json
import numpy as np
import logging
from datetime import datetime
from typing import Dict, Any, List, Tuple
from dataclasses import dataclass
from tqdm import tqdm

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import Bayesian optimization dependencies
try:
    from skopt import gp_minimize
    from skopt.space import Real, Integer
    from skopt.acquisition import gaussian_ei
    from skopt.utils import use_named_args
    import skopt
    print("✅ scikit-optimize imported successfully")
except ImportError:
    print("❌ scikit-optimize not installed. Install with: pip install scikit-optimize")
    sys.exit(1)

from core.data_loader import load_domain_data
from core.data_models import DomainData, Paper
from core.algorithm_config import ComprehensiveAlgorithmConfig
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


def _get_config_weights_for_metadata():
    """Get configuration weights formatted for metadata storage."""
    try:
        config = load_optimization_config()
        weights = config["consensus_difference_weights"]
        
        return {
            'consensus_weight': weights['final_combination_weights']['consensus_weight'],
            'difference_weight': weights['final_combination_weights']['difference_weight'],
            'internal_consensus_weights': [
                weights['consensus_internal_weights']['c1_keyword_jaccard'],
                weights['consensus_internal_weights']['c2_tfidf_cohesion'],
                weights['consensus_internal_weights']['c3_citation_density']
            ],
            'internal_difference_weights': [
                weights['difference_internal_weights']['d1_keyword_js'],
                weights['difference_internal_weights']['d2_centroid_distance'],
                weights['difference_internal_weights']['d3_cross_citation_ratio']
            ],
            'configuration_source': 'optimization_config.json (centralized weights)'
        }
    except Exception as e:
        # Fallback in case of config loading issues
        return {
            'consensus_weight': 0.8,
            'difference_weight': 0.2,
            'internal_consensus_weights': [0.4, 0.4, 0.2],
            'internal_difference_weights': [0.4, 0.4, 0.2],
            'configuration_source': 'fallback defaults (config load failed)',
            'error': str(e)
        }


# Path to store consensus-difference-optimized parameters (Bayesian optimization)
CONSENSUS_DIFFERENCE_PARAMS_FILE = "results/optimized_parameters_bayesian.json"


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
    base_config: ComprehensiveAlgorithmConfig = None
) -> ComprehensiveAlgorithmConfig:
    """Convert parameter values to algorithm configuration."""
    if base_config is None:
        base_config = ComprehensiveAlgorithmConfig()

    return ComprehensiveAlgorithmConfig(
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
    params: List[float], domain_data: DomainData, domain_name: str
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
        direction_threshold, validation_threshold, 
        similarity_min_segment_length, similarity_max_segment_length
    )

    try:
        # Run actual algorithm with suppressed output
        with SuppressOutput(suppress_stdout=True):
            # Import and run change detection to get segments
            from core.integration import run_change_detection
            
            segmentation_results, change_detection_result = run_change_detection(
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
        # Weights are now loaded automatically from optimization_config.json
        evaluation_result = evaluate_segmentation_quality(segment_papers)
        
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
        print(f"    ⚠️  Increasing evaluations from {max_evaluations} to 20 (minimum for Bayesian optimization)")
        max_evaluations = 20
    
    bo_config = BayesianOptimizationConfig(
        n_calls=max_evaluations,
        n_initial_points=min(20, max_evaluations),
        random_state=domain_seed
    )
    search_space = bo_config.get_search_space()

    print(f"    🎯 Optimizing {domain_name} for CONSENSUS-DIFFERENCE QUALITY using BAYESIAN OPTIMIZATION...")
    print(f"    🎲 Using domain seed: {domain_seed}")
    print(f"    📊 Parameter space: 4D")
    print(f"    🔍 Max evaluations: {max_evaluations}")
    print(f"    🧠 Algorithm: Gaussian Process + Expected Improvement")
    print(f"    📋 Optimizing: direction_threshold, validation_threshold, similarity_min_segment_length, similarity_max_segment_length")

    bounds = [
        bo_config.direction_threshold_bounds,
        bo_config.validation_threshold_bounds,
        bo_config.similarity_min_segment_length_bounds,
        bo_config.similarity_max_segment_length_bounds,
    ]
    
    print(f"    📍 Direction threshold range: {bounds[0][0]:.3f} - {bounds[0][1]:.3f}")
    print(f"    📍 Validation threshold range: {bounds[1][0]:.3f} - {bounds[1][1]:.3f}")
    print(f"    📍 Similarity min segment length range: {bounds[2][0]} - {bounds[2][1]}")
    print(f"    📍 Similarity max segment length range: {bounds[3][0]} - {bounds[3][1]}")

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
        neg_score = consensus_difference_evaluation_bayesian(params, domain_data, domain_name)
        actual_score = -neg_score  # Convert back to positive for tracking
        
        # Update progress
        pbar.set_description(f"    🔍 Eval {evaluation_count}: dir={params[0]:.3f}, val={params[1]:.3f}, sim_len={int(params[2])}-{int(params[3])}, score={actual_score:.3f}")
        
        # Track best result
        if actual_score > best_score:
            best_score = actual_score
            best_params = params.copy()
            tqdm.write(f"    🎯 NEW BEST: Score={best_score:.3f} (dir={params[0]:.3f}, val={params[1]:.3f}, sim_len={int(params[2])}-{int(params[3])})")
        
        # Store result for compatibility
        detailed_evaluation = {
            'score': actual_score,
            'consensus_score': 0.0,  # Will be filled in final verification
            'difference_score': 0.0,
            'consensus_explanation': "Bayesian optimization evaluation",
            'difference_explanation': "Bayesian optimization evaluation",
            'config_weights': _get_config_weights_for_metadata()
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

    print(f"    🌱 Starting Bayesian optimization...")

    # Create progress bar
    pbar = tqdm(total=max_evaluations, desc="    🔍 Bayesian Opt", unit="eval", 
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
        config = research_vector_to_config(*best_params)
        with SuppressOutput(suppress_stdout=True):
            # Run segmentation with best parameters
            from core.integration import run_change_detection
            
            segmentation_results, change_detection_result = run_change_detection(
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
        detailed_evaluation = evaluate_segmentation_quality(segment_papers)
        final_best_score = detailed_evaluation.final_score
        
        final_best_detailed_evaluation = {
            'score': detailed_evaluation.final_score,
            'consensus_score': detailed_evaluation.consensus_score,
            'difference_score': detailed_evaluation.difference_score,
            'num_segments': detailed_evaluation.num_segments,
            'consensus_explanation': detailed_evaluation.consensus_explanation,
            'difference_explanation': detailed_evaluation.difference_explanation,
            'methodology_explanation': detailed_evaluation.methodology_explanation,
            'individual_consensus_scores': detailed_evaluation.individual_consensus_scores,
            'individual_difference_scores': detailed_evaluation.individual_difference_scores,
            'config_weights': _get_config_weights_for_metadata()
        }
        
        if abs(final_best_score - best_score) > 0.001:
            print(f"    🔄 Final verification: {best_score:.3f} → {final_best_score:.3f}")
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

        print(f"    ✅ Bayesian optimization complete: score={best_score:.3f}")
        print(f"    🎯 Best params: direction={best_params[0]:.3f}, validation={best_params[1]:.3f}, similarity_min_length={int(best_params[2])} - {int(best_params[3])}")

        return results

    except Exception as e:
        pbar.close()
        print(f"    ❌ Bayesian optimization failed: {e}")
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
    """Save consensus-difference-optimized parameters to JSON file with detailed explanations."""
    
    # Allow custom filename suffix to distinguish between optimization methods
    if filename_suffix:
        base_name = CONSENSUS_DIFFERENCE_PARAMS_FILE.replace('.json', f'_{filename_suffix}.json')
    else:
        base_name = CONSENSUS_DIFFERENCE_PARAMS_FILE
    
    # Ensure results directory exists
    os.makedirs(os.path.dirname(base_name), exist_ok=True)

    # Format for saving - include detailed explanations
    formatted_results = {}
    detailed_evaluations = {}
    
    for domain, result in optimization_results.items():
        if result.get("optimization_successful", False):
            formatted_results[domain] = result["best_parameters"]
            # TRANSPARENCY: Save detailed explanations
            if "best_detailed_evaluation" in result:
                detailed_evaluations[domain] = result["best_detailed_evaluation"]

    # Add metadata
    save_data = {
        "consensus_difference_optimized_parameters": formatted_results,
        "detailed_evaluations": detailed_evaluations,  # TRANSPARENCY: Save explanations
        "metadata": {
            "optimization_date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_domains": len(formatted_results),
            "parameter_space": "4D (direction_threshold, validation_threshold, similarity_min_segment_length, similarity_max_segment_length)",
            "optimization_method": "Bayesian Optimization (Gaussian Process + Expected Improvement)",
            "objective_function": "Consensus-Difference Score (consensus_within_segments + difference_between_segments)",
            "optimization_type": "bayesian_optimization",
            "metrics_framework": "Phase 15 C-metrics (consensus) + D-metrics (difference)",
            "transparency_features": "Full explanations for consensus and difference scores with individual metric breakdowns",
        },
    }

    with open(base_name, "w") as f:
        json.dump(save_data, f, indent=2)

    print(f"💾 Consensus-difference-optimized parameters saved to {base_name}")
    print(f"📋 Detailed explanations saved for {len(detailed_evaluations)} domains")


def convert_dataframe_to_domain_data(df, domain_name: str) -> DomainData:
    """Convert DataFrame to DomainData object with all required fields."""
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


def optimize_single_domain(domain_name: str, max_evaluations: int = 100) -> Dict[str, Any]:
    """Optimize research quality parameters for a single domain using Bayesian optimization."""
    print(f"\n🎯 BAYESIAN CONSENSUS-DIFFERENCE OPTIMIZATION: {domain_name.upper()}")
    print("=" * 70)

    try:
        # Load domain data
        print(f"📊 Loading {domain_name} data...")
        df = load_domain_data(domain_name)

        if df is None or df.empty:
            print(f"❌ No data available for {domain_name}")
            return {"optimization_successful": False, "error": "No data available"}

        print(f"✅ Loaded {len(df)} papers")

        # Convert to DomainData
        domain_data = convert_dataframe_to_domain_data(df, domain_name)
        print(f"📚 Processed {len(domain_data.papers)} papers ({domain_data.year_range[0]}-{domain_data.year_range[1]})")

        # Run Bayesian optimization
        start_time = time.time()
        result = optimize_consensus_difference_parameters_bayesian(
            domain_data, domain_name, max_evaluations=max_evaluations
        )
        execution_time = time.time() - start_time

        # Display results
        if result["optimization_successful"]:
            params = result["best_parameters"]
            score = result["best_consensus_difference_score"]

            print(f"\n✅ BAYESIAN OPTIMIZATION SUCCESS:")
            print(f"   🎯 Direction threshold: {params['direction_threshold']:.3f}")
            print(f"   🎯 Validation threshold: {params['validation_threshold']:.3f}")
            print(f"   🎯 Similarity min segment length: {params['similarity_min_segment_length']}")
            print(f"   🎯 Similarity max segment length: {params['similarity_max_segment_length']}")
            print(f"   📈 Consensus-difference score: {score:.3f}")
            print(f"   ⏱️  Execution time: {execution_time:.1f}s")
            print(f"   🔄 Total evaluations: {result['total_evaluations']}")            
            # TRANSPARENCY: Display detailed explanations
            if "best_detailed_evaluation" in result:
                detailed_eval = result["best_detailed_evaluation"]
                print(f"\n📋 DETAILED CONSENSUS-DIFFERENCE ANALYSIS:")
                print(f"   📊 Consensus Score: {detailed_eval['consensus_score']:.3f}")
                print(f"   🔄 Difference Score: {detailed_eval['difference_score']:.3f}")
                print(f"   🏗️  Number of Segments: {detailed_eval['num_segments']}")
                print(f"   📝 Consensus Analysis:")
                print(f"      {detailed_eval['consensus_explanation']}")
                print(f"   📝 Difference Analysis:")
                print(f"      {detailed_eval['difference_explanation']}")
                print(f"   ⚖️  Configuration Weights:")
                print(f"      • Final Score: Balanced average (50/50) of consensus and difference")
                print(f"      • Internal Consensus Weights: C1={detailed_eval['config_weights']['internal_consensus_weights'][0]:.1f}, C2={detailed_eval['config_weights']['internal_consensus_weights'][1]:.1f}, C3={detailed_eval['config_weights']['internal_consensus_weights'][2]:.1f}")
                print(f"      • Internal Difference Weights: D1={detailed_eval['config_weights']['internal_difference_weights'][0]:.1f}, D2={detailed_eval['config_weights']['internal_difference_weights'][1]:.1f}, D3={detailed_eval['config_weights']['internal_difference_weights'][2]:.1f}")

            # Add execution time to result
            result["execution_time"] = execution_time
        else:
            print(f"❌ Bayesian optimization failed for {domain_name}")

        return result

    except Exception as e:
        print(f"❌ Error optimizing {domain_name}: {e}")
        return {"optimization_successful": False, "error": str(e)}


def discover_available_domains() -> List[str]:
    """
    Dynamically discover all available domains by scanning the data/processed directory.
    
    Returns:
        List of domain names that have processed data files
    """
    processed_data_dir = "data/processed"
    
    if not os.path.exists(processed_data_dir):
        print(f"⚠️ Processed data directory not found: {processed_data_dir}")
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
            print(f"🔍 Discovered {len(available_domains)} available domains:")
            for domain in available_domains:
                print(f"  • {domain}")
        else:
            print("⚠️ No processed data files found in data/processed/")
        
        return available_domains
        
    except Exception as e:
        print(f"❌ Error scanning for available domains: {e}")
        return []


def main():
    """Main function."""
    # Discover available domains dynamically
    all_available_domains = discover_available_domains()
    
    if not all_available_domains:
        print("❌ No domains available for optimization. Please ensure processed data files exist in data/processed/")
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
                print(f"⚠️ Domain '{domain}' not found in available domains. Skipping.")
        
        if not domains_to_optimize:
            print("❌ None of the requested domains are available.")
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
        else:
            print("❌ No domains were successfully optimized.")
    else:
        # Optimize all available domains with default evaluations
        max_evals = 200  
        print(f"🚀 Running Bayesian consensus-difference optimization on all {len(all_available_domains)} available domains with {max_evals} evaluations each")
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
            print(f"✅ Successfully optimized {len(successful_results)} out of {len(all_available_domains)} domains")
        else:
            print("❌ No domains were successfully optimized.")


if __name__ == "__main__":
    main() 