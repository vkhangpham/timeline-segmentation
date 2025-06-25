#!/usr/bin/env python3
"""
FEATURE-04: Contextual Embedding A/B Test Driver
===============================================

Compares TF-IDF vs contextual embeddings (SBERT/SPECTER) on pilot domains,
then full sweep. Follows Phase 16 fail-fast and real-data principles.

Usage:
    python experiments/metric_evaluation/contextual_embeddings.py --domains computer_vision applied_mathematics
    python experiments/metric_evaluation/contextual_embeddings.py --full-sweep
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parents[2]))

from core.data_loader import load_domain_data
from core.consensus_difference_metrics import evaluate_segmentation_quality
from core.integration import run_change_detection, timeline_analysis


def run_single_domain_comparison(
    domain: str, 
    models_to_test: List[str] = None,
    max_iterations: int = 20
) -> Dict:
    """
    Run TF-IDF vs contextual embedding comparison on a single domain.
    
    Args:
        domain: Domain name (e.g., 'computer_vision')
        models_to_test: List of contextual models to test against TF-IDF
        max_iterations: Number of Bayesian optimization iterations (reduced for speed)
    
    Returns:
        Dictionary with comparison results
    """
    if models_to_test is None:
        models_to_test = ["all-mpnet-base-v2", "all-MiniLM-L6-v2"]
    
    print(f"\n=== Domain: {domain} ===")
    
    # Load optimized parameters for this domain
    try:
        with open("results/optimized_parameters_bayesian.json", "r") as f:
            optimized_data = json.load(f)
        
        optimized_params = optimized_data["consensus_difference_optimized_parameters"]
        
        if domain not in optimized_params:
            raise ValueError(f"No optimized parameters found for domain {domain}")
        
        params = optimized_params[domain]
        print(f"Using optimized parameters: {params}")
        
    except Exception as e:
        print(f"ERROR: Could not load optimized parameters: {e}")
        raise
    
    results = {"domain": domain, "comparisons": {}}
    
    # Test TF-IDF baseline
    print("\n--- Testing TF-IDF Baseline ---")
    os.environ["VECTORIZER_TYPE"] = "tfidf"
    
    start_time = time.time()
    
    # Load domain data and convert to Paper objects  
    from core.data_loader import load_and_validate_domain_data
    from core.data_models import Paper
    
    df = load_and_validate_domain_data(domain)
    
    # Convert DataFrame to Paper objects
    papers = []
    for _, row in df.iterrows():
        keywords = []
        if pd.notna(row['keywords']) and row['keywords']:
            keywords = [k.strip() for k in str(row['keywords']).split('|') if k.strip()]
        
        children = []
        if pd.notna(row['children']) and row['children']:
            children = [c.strip() for c in str(row['children']).split('|') if c.strip()]
        
        paper = Paper(
            id=str(row['id']),
            title=str(row['title']) if pd.notna(row['title']) else "",
            content=str(row['content']) if pd.notna(row['content']) else "",
            pub_year=int(row['year']),
            cited_by_count=int(row['cited_by_count']) if pd.notna(row['cited_by_count']) else 0,
            keywords=tuple(keywords),
            children=tuple(children),
            description=""
        )
        papers.append(paper)
    
    # Create fixed test segments (use existing segmentation approach from optimized parameters)
    # For simplicity, create 3-4 segments based on year ranges
    year_min, year_max = df['year'].min(), df['year'].max()
    span = year_max - year_min
    
    if span <= 15:
        # Short span: 2 segments
        mid_year = year_min + span // 2
        segments = [
            (year_min, mid_year),
            (mid_year + 1, year_max)
        ]
    elif span <= 30:
        # Medium span: 3 segments
        segment_size = span // 3
        segments = [
            (year_min, year_min + segment_size),
            (year_min + segment_size + 1, year_min + 2 * segment_size),
            (year_min + 2 * segment_size + 1, year_max)
        ]
    else:
        # Long span: 4 segments
        segment_size = span // 4
        segments = [
            (year_min, year_min + segment_size),
            (year_min + segment_size + 1, year_min + 2 * segment_size),
            (year_min + 2 * segment_size + 1, year_min + 3 * segment_size),
            (year_min + 3 * segment_size + 1, year_max)
        ]
    
    # Convert to segment papers
    segment_papers = []
    for start_year, end_year in segments:
        period_papers = tuple([
            paper for paper in papers 
            if start_year <= paper.pub_year <= end_year
        ])
        if period_papers:  # Only add non-empty segments
            segment_papers.append(period_papers)
    
    if not segment_papers:
        raise ValueError(f"No valid segments with papers found for domain {domain}")
    
    print(f"Created {len(segment_papers)} segments: {[(min(p.pub_year for p in seg), max(p.pub_year for p in seg)) for seg in segment_papers]}")
    
    # Evaluate segmentation quality using consensus-difference metrics
    segmentation_eval = evaluate_segmentation_quality(segment_papers)
    tfidf_time = time.time() - start_time
    
    results["comparisons"]["tfidf"] = {
        "final_score": segmentation_eval.final_score,
        "consensus_score": segmentation_eval.consensus_score,
        "difference_score": segmentation_eval.difference_score,
        "num_segments": segmentation_eval.num_segments,
        "runtime_seconds": tfidf_time,
        "model_type": "tfidf"
    }
    
    print(f"TF-IDF: L={segmentation_eval.final_score:.3f}, segments={segmentation_eval.num_segments}, time={tfidf_time:.1f}s")
    
    # Test each contextual model
    for model_name in models_to_test:
        print(f"\n--- Testing Contextual Model: {model_name} ---")
        
        # Configure environment for contextual embeddings
        os.environ["VECTORIZER_TYPE"] = "contextual"
        os.environ["CONTEXTUAL_MODEL"] = model_name
        
        try:
            start_time = time.time()
            
            # Evaluate the same segments with contextual embeddings (environment already set)
            segmentation_eval_ctx = evaluate_segmentation_quality(segment_papers)
            contextual_time = time.time() - start_time
            
            results["comparisons"][model_name] = {
                "final_score": segmentation_eval_ctx.final_score,
                "consensus_score": segmentation_eval_ctx.consensus_score,
                "difference_score": segmentation_eval_ctx.difference_score,
                "num_segments": segmentation_eval_ctx.num_segments,
                "runtime_seconds": contextual_time,
                "model_type": "contextual",
                "model_name": model_name
            }
            
            improvement = ((segmentation_eval_ctx.final_score - segmentation_eval.final_score) 
                          / segmentation_eval.final_score) * 100
            
            print(f"{model_name}: L={segmentation_eval_ctx.final_score:.3f} ({improvement:+.1f}%), "
                  f"segments={segmentation_eval_ctx.num_segments}, time={contextual_time:.1f}s")
            
        except Exception as e:
            print(f"ERROR in {model_name}: {e}")
            results["comparisons"][model_name] = {
                "error": str(e),
                "model_type": "contextual",
                "model_name": model_name
            }
            # Fail-fast: re-raise to stop execution
            raise
    
    # Clean up environment
    os.environ.pop("VECTORIZER_TYPE", None)
    os.environ.pop("CONTEXTUAL_MODEL", None)
    
    return results


def run_pilot_study(domains: List[str]) -> Dict:
    """Run pilot study on selected domains."""
    print("=== FEATURE-04 Contextual Embedding Pilot Study ===")
    
    all_results = {
        "experiment": "contextual_embeddings_pilot",
        "timestamp": time.strftime("%Y%m%d_%H%M%S"),
        "domains": domains,
        "results": {}
    }
    
    for domain in domains:
        try:
            domain_results = run_single_domain_comparison(domain)
            all_results["results"][domain] = domain_results
        except Exception as e:
            print(f"FATAL ERROR in domain {domain}: {e}")
            raise  # Fail-fast
    
    return all_results


def run_full_sweep() -> Dict:
    """Run full sweep across all eight domains."""
    domains = [
        "computer_vision", "applied_mathematics", "computer_science", 
        "deep_learning", "machine_learning", "natural_language_processing",
        "machine_translation", "art"
    ]
    
    print("=== FEATURE-04 Contextual Embedding Full Sweep ===")
    
    return run_pilot_study(domains)


def main():
    parser = argparse.ArgumentParser(description="Contextual Embedding A/B Test")
    parser.add_argument("--domains", nargs="+", 
                       help="Specific domains to test (e.g., computer_vision applied_mathematics)")
    parser.add_argument("--full-sweep", action="store_true",
                       help="Run full sweep across all eight domains")
    parser.add_argument("--models", nargs="+", 
                       default=["all-mpnet-base-v2", "all-MiniLM-L6-v2"],
                       help="Contextual models to test")
    
    args = parser.parse_args()
    
    if args.full_sweep:
        results = run_full_sweep()
    elif args.domains:
        results = run_pilot_study(args.domains)
    else:
        print("ERROR: Must specify either --domains or --full-sweep")
        return 1
    
    # Save results
    os.makedirs("experiments/metric_evaluation/results", exist_ok=True)
    output_file = f"experiments/metric_evaluation/results/contextual_embeddings_{results['timestamp']}.json"
    
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n=== Results saved to {output_file} ===")
    
    # Summary analysis
    print("\n=== SUMMARY ===")
    for domain, domain_results in results["results"].items():
        tfidf_score = domain_results["comparisons"]["tfidf"]["final_score"]
        print(f"\n{domain}:")
        print(f"  TF-IDF baseline: {tfidf_score:.3f}")
        
        for model_name, model_results in domain_results["comparisons"].items():
            if model_name == "tfidf":
                continue
            if "error" in model_results:
                print(f"  {model_name}: ERROR - {model_results['error']}")
            else:
                score = model_results["final_score"]
                improvement = ((score - tfidf_score) / tfidf_score) * 100
                print(f"  {model_name}: {score:.3f} ({improvement:+.1f}%)")
    
    return 0


if __name__ == "__main__":
    sys.exit(main()) 