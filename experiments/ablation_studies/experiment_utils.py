"""
Utility Functions for Ablation Studies

Pure functions and common utilities shared across all ablation experiments.
Follows functional programming principles with immutable data and fail-fast error handling.
"""

import os
import sys
import time
import json
import numpy as np
from typing import Dict, Any, List, Tuple, NamedTuple
from dataclasses import dataclass
from pathlib import Path

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from core.data_loader import load_domain_data
from core.data_models import DomainData, Paper
from core.algorithm_config import AlgorithmConfig
from core.consensus_difference_metrics import evaluate_segmentation_quality
from core.similarity_segmentation import create_similarity_based_segments
from core.keyword_utils import extract_year_keywords


# Test domains for ablation studies (representing diverse characteristics)
TEST_DOMAINS = [
    "machine_learning",     # Established field, good keyword quality
    "deep_learning",        # Rapidly evolving, challenging segmentation  
    "applied_mathematics",  # Long temporal span, mature terminology
    "art"                   # Different citation culture, diverse keywords
]


class ExperimentResult(NamedTuple):
    """Immutable result structure for ablation experiments."""
    experiment_name: str
    domain: str
    condition: str
    score: float
    consensus_score: float
    difference_score: float
    num_segments: int
    execution_time: float
    metadata: Dict[str, Any]


class AblationStudyConfig(NamedTuple):
    """Configuration for ablation study execution."""
    test_domains: List[str]
    output_directory: str
    random_seed: int
    statistical_significance_threshold: float
    

def convert_dataframe_to_domain_data(df, domain_name: str) -> DomainData:
    """
    Convert DataFrame to DomainData object with all required fields.
    
    Pure function that transforms raw data into standardized domain data format.
    
    Args:
        df: DataFrame with paper data
        domain_name: Name of the domain
        
    Returns:
        DomainData object with properly formatted papers
        
    Raises:
        ValueError: If DataFrame is invalid or missing required fields
    """
    if df is None or df.empty:
        raise ValueError(f"DataFrame for domain '{domain_name}' is None or empty")
    
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
        raise ValueError(f"No valid papers found for domain '{domain_name}'")

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


def load_test_domains() -> Dict[str, DomainData]:
    """
    Load all test domains for ablation studies.
    
    Pure function that loads and validates domain data for experimental use.
    
    Returns:
        Dictionary mapping domain names to DomainData objects
        
    Raises:
        ValueError: If any test domain cannot be loaded
    """
    domain_data = {}
    
    for domain_name in TEST_DOMAINS:
        print(f"📊 Loading {domain_name} data...")
        # Use correct relative paths from the experiments/ablation_studies directory
        df = load_domain_data(domain_name, 
                            prefer_source="csv",
                            processed_dir="../../data/processed", 
                            resources_dir="../../resources")
        
        if df is None or df.empty:
            raise ValueError(f"Failed to load data for domain '{domain_name}'")
        
        domain_data[domain_name] = convert_dataframe_to_domain_data(df, domain_name)
        print(f"✅ Loaded {len(domain_data[domain_name].papers)} papers "
              f"({domain_data[domain_name].year_range[0]}-{domain_data[domain_name].year_range[1]})")
    
    return domain_data


def evaluate_segmentation_configuration(
    domain_data: DomainData,
    algorithm_config: AlgorithmConfig,
    use_citation: bool = True,
    use_direction: bool = True
) -> Tuple[float, Dict[str, Any]]:
    """
    Evaluate a specific algorithmic configuration for ablation studies.
    
    Pure function that runs segmentation and returns evaluation metrics.
    
    Args:
        domain_data: Domain data to evaluate on
        algorithm_config: Algorithm configuration to test
        use_citation: Whether to enable citation validation
        use_direction: Whether to enable direction detection
        
    Returns:
        Tuple of (consensus_difference_score, detailed_metrics)
        
    Raises:
        ValueError: If configuration is invalid or evaluation fails
    """
    if not use_citation and not use_direction:
        raise ValueError("At least one detection modality (citation or direction) must be enabled")
    
    try:
        # Import required modules for direct shift signal detection
        from core.shift_signal_detection import detect_shift_signals
        
        # Run shift signal detection with modality controls
        shift_signals = detect_shift_signals(
            domain_data, 
            domain_data.domain_name, 
            algorithm_config,
            use_citation=use_citation,
            use_direction=use_direction
        )
        
        # Create segments from shift signals using similarity segmentation
        if shift_signals:
            # Extract year keywords for similarity analysis
            year_keywords = extract_year_keywords(domain_data)
            
            # Create similarity-based segments with length controls
            similarity_segments, similarity_metadata = create_similarity_based_segments(
                shift_signals, 
                year_keywords, 
                domain_data,
                min_segment_length=algorithm_config.similarity_min_segment_length,
                max_segment_length=algorithm_config.similarity_max_segment_length
            )
            
            # Convert segments to year tuples for evaluation
            segments = similarity_segments
        else:
            # No signals found, create single segment spanning entire domain
            segments = [(domain_data.year_range[0], domain_data.year_range[1])]
        
        # Convert segments (year tuples) to lists of papers for evaluation
        segment_papers = []
        for start_year, end_year in segments:
            segment_paper_list = [
                paper for paper in domain_data.papers
                if start_year <= paper.pub_year <= end_year
            ]
            if segment_paper_list:  # Only add non-empty segments
                segment_papers.append(tuple(segment_paper_list))
        
        # If no valid segments, create single segment with all papers
        if not segment_papers:
            segment_papers = [tuple(domain_data.papers)]

        # Evaluate segmentation quality using consensus-difference metrics
        evaluation_result = evaluate_segmentation_quality(segment_papers)
        
        # Create detailed metrics dictionary
        detailed_metrics = {
            'consensus_score': evaluation_result.consensus_score,
            'difference_score': evaluation_result.difference_score,
            'num_segments': evaluation_result.num_segments,
            'individual_consensus_scores': evaluation_result.individual_consensus_scores,
            'individual_difference_scores': evaluation_result.individual_difference_scores,
            'consensus_explanation': evaluation_result.consensus_explanation,
            'difference_explanation': evaluation_result.difference_explanation,
            'methodology_explanation': evaluation_result.methodology_explanation,
            'segment_sizes': [len(seg) for seg in segment_papers],
            'shift_signals': [(s.year, s.confidence, s.signal_type) for s in shift_signals],
            'segments': segments,
            'modality_settings': {
                'use_citation': use_citation,
                'use_direction': use_direction,
                'num_shift_signals': len(shift_signals)
            }
        }
        
        return evaluation_result.final_score, detailed_metrics
        
    except Exception as e:
        raise ValueError(f"Segmentation evaluation failed for domain {domain_data.domain_name}: {str(e)}")


def create_experiment_output_directory(experiment_name: str) -> str:
    """
    Create output directory for experiment results.
    
    Pure function that creates directory structure for experiment outputs.
    
    Args:
        experiment_name: Name of the experiment
        
    Returns:
        Path to created output directory
    """
    base_dir = "experiments/ablation_studies/results"
    experiment_dir = os.path.join(base_dir, experiment_name)
    
    # Create directory if it doesn't exist
    os.makedirs(experiment_dir, exist_ok=True)
    
    return experiment_dir


def save_experiment_results(
    results: List[ExperimentResult],
    experiment_name: str,
    additional_metadata: Dict[str, Any] = None
) -> str:
    """
    Save experiment results to JSON file with metadata.
    
    Pure function that serializes experiment results for later analysis.
    
    Args:
        results: List of experiment results
        experiment_name: Name of the experiment
        additional_metadata: Optional additional metadata
        
    Returns:
        Path to saved results file
    """
    output_dir = create_experiment_output_directory(experiment_name)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"{experiment_name}_results_{timestamp}.json")
    
    # Helper function to make data JSON serializable
    def make_serializable(obj):
        if isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [make_serializable(item) for item in obj]
        elif isinstance(obj, tuple):
            return [make_serializable(item) for item in obj]
        elif isinstance(obj, (bool, int, float, str, type(None))):
            return obj
        elif hasattr(obj, '__dict__'):
            return make_serializable(obj.__dict__)
        else:
            return str(obj)
    
    # Convert results to serializable format
    serializable_results = []
    for result in results:
        serializable_results.append({
            'experiment_name': result.experiment_name,
            'domain': result.domain,
            'condition': result.condition,
            'score': result.score,
            'consensus_score': result.consensus_score,
            'difference_score': result.difference_score,
            'num_segments': result.num_segments,
            'execution_time': result.execution_time,
            'metadata': make_serializable(result.metadata)
        })
    
    # Create complete output data
    output_data = {
        'experiment_name': experiment_name,
        'timestamp': timestamp,
        'results': serializable_results,
        'summary': {
            'total_conditions': len(set(r.condition for r in results)),
            'total_domains': len(set(r.domain for r in results)),
            'total_evaluations': len(results),
            'mean_score': float(np.mean([r.score for r in results])),
            'std_score': float(np.std([r.score for r in results]))
        },
        'metadata': make_serializable(additional_metadata) if additional_metadata else {}
    }
    
    # Save results
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"💾 Experiment results saved to {output_file}")
    return output_file


def calculate_statistical_significance(
    condition_a_scores: List[float],
    condition_b_scores: List[float],
    alpha: float = 0.05
) -> Dict[str, Any]:
    """
    Calculate statistical significance between two experimental conditions.
    
    Pure function that performs statistical testing on experimental results.
    
    Args:
        condition_a_scores: Scores from first condition
        condition_b_scores: Scores from second condition
        alpha: Significance threshold
        
    Returns:
        Dictionary with statistical test results
    """
    from scipy import stats
    
    if len(condition_a_scores) != len(condition_b_scores):
        raise ValueError("Condition score lists must have equal length for paired testing")
    
    # Paired t-test (since we're comparing same domains across conditions)
    t_stat, p_value = stats.ttest_rel(condition_a_scores, condition_b_scores)
    
    # Effect size (Cohen's d for paired samples)
    differences = np.array(condition_a_scores) - np.array(condition_b_scores)
    effect_size = np.mean(differences) / np.std(differences, ddof=1)
    
    # Bootstrap confidence interval for difference
    bootstrap_diffs = []
    n_bootstrap = 1000
    for _ in range(n_bootstrap):
        sample_indices = np.random.choice(len(differences), len(differences), replace=True)
        bootstrap_diffs.append(np.mean(differences[sample_indices]))
    
    ci_lower = np.percentile(bootstrap_diffs, 100 * alpha / 2)
    ci_upper = np.percentile(bootstrap_diffs, 100 * (1 - alpha / 2))
    
    return {
        't_statistic': float(t_stat),
        'p_value': float(p_value),
        'is_significant': p_value < alpha,
        'effect_size': float(effect_size),
        'mean_difference': float(np.mean(differences)),
        'confidence_interval': (float(ci_lower), float(ci_upper)),
        'alpha': alpha
    }


def print_experiment_summary(results: List[ExperimentResult], experiment_name: str) -> None:
    """
    Print formatted summary of experiment results.
    
    Pure function that generates human-readable experiment summary.
    
    Args:
        results: List of experiment results
        experiment_name: Name of the experiment
    """
    print(f"\n📊 {experiment_name.upper()} SUMMARY")
    print("=" * 70)
    
    # Group results by condition
    conditions = {}
    for result in results:
        if result.condition not in conditions:
            conditions[result.condition] = []
        conditions[result.condition].append(result)
    
    # Print condition summaries
    for condition, condition_results in conditions.items():
        scores = [r.score for r in condition_results]
        times = [r.execution_time for r in condition_results]
        segments = [r.num_segments for r in condition_results]
        
        print(f"\n🔍 {condition}:")
        print(f"   Score: {np.mean(scores):.3f} ± {np.std(scores):.3f}")
        print(f"   Time: {np.mean(times):.1f}s ± {np.std(times):.1f}s") 
        print(f"   Segments: {np.mean(segments):.1f} ± {np.std(segments):.1f}")
        
        for domain_result in condition_results:
            print(f"     {domain_result.domain}: {domain_result.score:.3f} "
                  f"({domain_result.num_segments} segments)")
    
    # Find best condition
    condition_means = {cond: np.mean([r.score for r in results]) 
                      for cond, results in conditions.items()}
    best_condition = max(condition_means, key=condition_means.get)
    
    print(f"\n🏆 Best condition: {best_condition} (score={condition_means[best_condition]:.3f})") 