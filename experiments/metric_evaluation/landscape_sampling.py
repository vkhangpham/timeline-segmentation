#!/usr/bin/env python3
"""
Landscape Sampling Experiment - Phase 16 PARETO-01

Measures optimization surface characteristics for linear vs harmonic aggregation
to prove/disprove claims about landscape smoothness and optimization effectiveness.

Following Rule 4: Tests on real data subsets before full implementation.
Following Rule 6: Fail-fast error handling with no fallbacks.
Following Rule 7: Comprehensive logging for terminal analysis.
"""

import json
import time
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

from core.data_processing import process_domain_data
from core.algorithm_config import ComprehensiveAlgorithmConfig
from core.integration import run_change_detection
from core.consensus_difference_metrics import evaluate_segmentation_quality


@dataclass
class LandscapePoint:
    """Single point in parameter landscape with measured characteristics."""
    direction_threshold: float
    validation_threshold: float
    min_segment_length: int
    max_segment_length: int
    consensus_score: float
    difference_score: float
    linear_aggregate: float
    harmonic_aggregate: float
    execution_time: float
    num_segments: int
    success: bool


@dataclass
class LandscapeSmoothness:
    """Smoothness metrics for optimization landscape."""
    spearman_correlation: float  # Between parameter distance and score distance
    local_variance: float  # Variance within epsilon-neighborhoods
    gradient_roughness: float  # Variance of numerical gradients
    num_local_optima: int  # Estimated local optima count


def generate_latin_hypercube_samples(n_samples: int = 400) -> List[Dict[str, float]]:
    """
    Generate Latin hypercube samples over parameter space.
    
    Following Rule 9: Functional programming approach with pure function.
    """
    print(f"🎯 Generating {n_samples} Latin hypercube samples over parameter space")
    
    # Parameter bounds from timeline segmentation paper
    bounds = {
        'direction_threshold': (0.1, 0.4),
        'validation_threshold': (0.3, 0.45),
        'min_segment_length': (3, 5),
        'max_segment_length': (10, 30)
    }
    
    # Generate uniform samples in [0,1]^4
    np.random.seed(42)  # Reproducible sampling
    uniform_samples = np.random.random((n_samples, 4))
    
    # Latin hypercube sampling: permute each dimension
    for i in range(4):
        uniform_samples[:, i] = (np.random.permutation(n_samples) + uniform_samples[:, i]) / n_samples
    
    # Scale to parameter bounds
    samples = []
    param_names = list(bounds.keys())
    
    for i in range(n_samples):
        sample = {}
        for j, param in enumerate(param_names):
            low, high = bounds[param]
            if param in ['min_segment_length', 'max_segment_length']:
                sample[param] = int(low + uniform_samples[i, j] * (high - low))
            else:
                sample[param] = low + uniform_samples[i, j] * (high - low)
        
        # Ensure min_segment_length <= max_segment_length
        if sample['min_segment_length'] > sample['max_segment_length']:
            sample['min_segment_length'], sample['max_segment_length'] = \
                sample['max_segment_length'], sample['min_segment_length']
        
        samples.append(sample)
    
    print(f"✅ Generated {len(samples)} valid parameter combinations")
    return samples


def evaluate_single_point(domain_name: str, params: Dict[str, float]) -> LandscapePoint:
    """
    Evaluate single point in parameter landscape.
    
    Following Rule 6: Fail-fast - no try-catch blocks.
    """
    start_time = time.time()
    
    # Load domain data (Rule 3: Real data only)
    result = process_domain_data(domain_name)
    if not result.success:
        raise ValueError(f"Failed to load domain data for {domain_name}: {result.error_message}")
    
    domain_data = result.domain_data
    
    # Create configuration
    config = ComprehensiveAlgorithmConfig(
        direction_threshold=params['direction_threshold'],
        validation_threshold=params['validation_threshold'],
        similarity_min_segment_length=params['min_segment_length'],
        similarity_max_segment_length=params['max_segment_length'],
        # Use optimized defaults for other parameters
        keyword_min_papers_ratio=0.05,  # From FEATURE-01
        tfidf_max_features=10000,  # From FEATURE-02
    )
    
    # Consensus-difference weights from FEATURE-00
    consensus_weight = 0.1
    difference_weight = 0.9
    
    # Run segmentation
    segmentation_results, change_detection_result = run_change_detection(domain_name, algorithm_config=config)
    
    if not segmentation_results or not segmentation_results.get('segments'):
        # Failed segmentation - return failure point
        return LandscapePoint(
            direction_threshold=params['direction_threshold'],
            validation_threshold=params['validation_threshold'],
            min_segment_length=params['min_segment_length'],
            max_segment_length=params['max_segment_length'],
            consensus_score=0.0,
            difference_score=0.0,
            linear_aggregate=0.0,
            harmonic_aggregate=0.0,
            execution_time=time.time() - start_time,
            num_segments=0,
            success=False
        )
    
    # Extract segments from results
    segments = [(seg[0], seg[1]) for seg in segmentation_results['segments']]
    
    # Convert segments to segment_papers for evaluation
    segment_papers = []
    for start_year, end_year in segments:
        papers_in_segment = []
        for paper in domain_data.papers:
            if start_year <= paper.pub_year <= end_year:
                papers_in_segment.append(paper)
        if papers_in_segment:  # Only add non-empty segments
            segment_papers.append(tuple(papers_in_segment))
    
    if not segment_papers:
        # No valid segments with papers - return failure point
        return LandscapePoint(
            direction_threshold=params['direction_threshold'],
            validation_threshold=params['validation_threshold'],
            min_segment_length=params['min_segment_length'],
            max_segment_length=params['max_segment_length'],
            consensus_score=0.0,
            difference_score=0.0,
            linear_aggregate=0.0,
            harmonic_aggregate=0.0,
            execution_time=time.time() - start_time,
            num_segments=0,
            success=False
        )
    
    # Evaluate with linear aggregation
    linear_result = evaluate_segmentation_quality(
        segment_papers,
        final_combination_weights=(consensus_weight, difference_weight),
        aggregation_method="linear"
    )
    
    # Evaluate with harmonic aggregation
    harmonic_result = evaluate_segmentation_quality(
        segment_papers,
        final_combination_weights=(consensus_weight, difference_weight),
        aggregation_method="harmonic"
    )
    
    execution_time = time.time() - start_time
    
    return LandscapePoint(
        direction_threshold=params['direction_threshold'],
        validation_threshold=params['validation_threshold'],
        min_segment_length=params['min_segment_length'],
        max_segment_length=params['max_segment_length'],
        consensus_score=linear_result.consensus_score,
        difference_score=linear_result.difference_score,
        linear_aggregate=linear_result.final_score,
        harmonic_aggregate=harmonic_result.final_score,
        execution_time=execution_time,
        num_segments=len(segments),
        success=True
    )


def calculate_landscape_smoothness(points: List[LandscapePoint], 
                                 aggregation_method: str = 'linear') -> LandscapeSmoothness:
    """
    Calculate smoothness metrics for optimization landscape.
    
    Following Rule 5: Critical quality evaluation with precise metrics.
    """
    print(f"📊 Calculating smoothness metrics for {aggregation_method} aggregation")
    
    # Filter successful points only
    valid_points = [p for p in points if p.success]
    if len(valid_points) < 10:
        raise ValueError(f"Insufficient valid points for analysis: {len(valid_points)}")
    
    # Extract parameter vectors and scores
    param_vectors = []
    scores = []
    
    for point in valid_points:
        param_vector = [
            point.direction_threshold,
            point.validation_threshold,
            point.min_segment_length,
            point.max_segment_length
        ]
        param_vectors.append(param_vector)
        
        if aggregation_method == 'linear':
            scores.append(point.linear_aggregate)
        else:
            scores.append(point.harmonic_aggregate)
    
    param_vectors = np.array(param_vectors)
    scores = np.array(scores)
    
    # Standardize parameters for distance calculation
    scaler = StandardScaler()
    param_vectors_scaled = scaler.fit_transform(param_vectors)
    
    # 1. Spearman correlation between parameter distance and score distance
    param_distances = pdist(param_vectors_scaled)
    score_distances = pdist(scores.reshape(-1, 1))
    spearman_corr, _ = spearmanr(param_distances, score_distances)
    
    # 2. Local variance (average variance within epsilon-neighborhoods)
    nn = NearestNeighbors(n_neighbors=min(10, len(valid_points)//2))
    nn.fit(param_vectors_scaled)
    
    local_variances = []
    for i, point in enumerate(param_vectors_scaled):
        distances, indices = nn.kneighbors([point])
        neighbor_scores = scores[indices[0]]
        if len(neighbor_scores) > 1:
            local_variances.append(np.var(neighbor_scores))
    
    local_variance = np.mean(local_variances) if local_variances else 0.0
    
    # 3. Gradient roughness (variance of numerical gradients)
    gradients = []
    for i in range(len(param_vectors_scaled)):
        # Find nearest neighbors for gradient estimation
        distances, indices = nn.kneighbors([param_vectors_scaled[i]])
        if len(indices[0]) >= 2:
            neighbor_params = param_vectors_scaled[indices[0][1:]]  # Exclude self
            neighbor_scores = scores[indices[0][1:]]
            
            # Estimate gradient using finite differences
            if len(neighbor_params) > 0:
                param_diffs = neighbor_params - param_vectors_scaled[i]
                score_diffs = neighbor_scores - scores[i]
                
                # Simple gradient estimate
                for j in range(len(param_diffs)):
                    if np.linalg.norm(param_diffs[j]) > 1e-8:
                        gradient = score_diffs[j] / np.linalg.norm(param_diffs[j])
                        gradients.append(gradient)
    
    gradient_roughness = np.var(gradients) if gradients else 0.0
    
    # 4. Local optima estimation (simplified)
    # Count points that are local maxima in their neighborhood
    local_optima_count = 0
    for i in range(len(param_vectors_scaled)):
        distances, indices = nn.kneighbors([param_vectors_scaled[i]])
        neighbor_scores = scores[indices[0]]
        if scores[i] >= np.max(neighbor_scores):
            local_optima_count += 1
    
    return LandscapeSmoothness(
        spearman_correlation=spearman_corr if not np.isnan(spearman_corr) else 0.0,
        local_variance=local_variance,
        gradient_roughness=gradient_roughness,
        num_local_optima=local_optima_count
    )


def analyze_consensus_difference_distribution(points: List[LandscapePoint]) -> Dict[str, float]:
    """
    Analyze distribution imbalance between consensus and difference scores.
    
    Following Rule 5: Quantitative analysis with specific metrics.
    """
    print("📈 Analyzing consensus-difference distribution characteristics")
    
    valid_points = [p for p in points if p.success and p.consensus_score > 0 and p.difference_score > 0]
    
    if len(valid_points) < 5:
        raise ValueError(f"Insufficient valid points for distribution analysis: {len(valid_points)}")
    
    consensus_scores = [p.consensus_score for p in valid_points]
    difference_scores = [p.difference_score for p in valid_points]
    
    # Distribution statistics
    consensus_mean = np.mean(consensus_scores)
    consensus_std = np.std(consensus_scores)
    difference_mean = np.mean(difference_scores)
    difference_std = np.std(difference_scores)
    
    # Imbalance metrics
    imbalance_ratios = [abs(c - d) / (c + d) for c, d in zip(consensus_scores, difference_scores)]
    mean_imbalance = np.mean(imbalance_ratios)
    
    # Scale difference
    consensus_cv = consensus_std / consensus_mean if consensus_mean > 0 else 0
    difference_cv = difference_std / difference_mean if difference_mean > 0 else 0
    
    # Correlation between consensus and difference
    correlation = np.corrcoef(consensus_scores, difference_scores)[0, 1]
    
    return {
        'consensus_mean': consensus_mean,
        'consensus_std': consensus_std,
        'consensus_cv': consensus_cv,
        'difference_mean': difference_mean,
        'difference_std': difference_std,
        'difference_cv': difference_cv,
        'mean_imbalance_ratio': mean_imbalance,
        'consensus_difference_correlation': correlation,
        'scale_ratio': difference_mean / consensus_mean if consensus_mean > 0 else 0,
    }


def run_landscape_sampling_experiment(domains: List[str], n_samples: int = 400) -> Dict[str, Any]:
    """
    Run complete landscape sampling experiment.
    
    Following Rule 4: Test on representative subset before full implementation.
    Following Rule 7: Comprehensive logging for analysis.
    """
    print("🚀 Starting Landscape Sampling Experiment")
    print(f"📋 Domains: {domains}")
    print(f"📊 Samples per domain: {n_samples}")
    
    # Generate parameter samples
    param_samples = generate_latin_hypercube_samples(n_samples)
    
    results = {}
    
    for domain in domains:
        print(f"\n🔍 Processing domain: {domain}")
        domain_start_time = time.time()
        
        domain_points = []
        
        for i, params in enumerate(param_samples):
            if i % 50 == 0:
                print(f"  📈 Progress: {i}/{len(param_samples)} ({100*i/len(param_samples):.1f}%)")
            
            try:
                point = evaluate_single_point(domain, params)
                domain_points.append(point)
                
                if not point.success:
                    print(f"  ⚠️  Failed evaluation at params: {params}")
                    
            except Exception as e:
                # Rule 6: Fail fast - let errors propagate
                print(f"  🚨 Error evaluating {params}: {e}")
                raise
        
        # Calculate metrics for this domain
        successful_points = [p for p in domain_points if p.success]
        success_rate = len(successful_points) / len(domain_points)
        
        print(f"  ✅ Success rate: {success_rate:.1%} ({len(successful_points)}/{len(domain_points)})")
        
        if len(successful_points) < 10:
            print(f"  ⚠️  Insufficient successful points for analysis: {len(successful_points)}")
            continue
        
        # Smoothness analysis
        linear_smoothness = calculate_landscape_smoothness(domain_points, 'linear')
        harmonic_smoothness = calculate_landscape_smoothness(domain_points, 'harmonic')
        
        # Distribution analysis
        distribution_stats = analyze_consensus_difference_distribution(domain_points)
        
        domain_time = time.time() - domain_start_time
        
        results[domain] = {
            'points': [
                {
                    'direction_threshold': p.direction_threshold,
                    'validation_threshold': p.validation_threshold,
                    'min_segment_length': p.min_segment_length,
                    'max_segment_length': p.max_segment_length,
                    'consensus_score': p.consensus_score,
                    'difference_score': p.difference_score,
                    'linear_aggregate': p.linear_aggregate,
                    'harmonic_aggregate': p.harmonic_aggregate,
                    'execution_time': p.execution_time,
                    'num_segments': p.num_segments,
                    'success': p.success
                }
                for p in domain_points
            ],
            'success_rate': success_rate,
            'linear_smoothness': {
                'spearman_correlation': linear_smoothness.spearman_correlation,
                'local_variance': linear_smoothness.local_variance,
                'gradient_roughness': linear_smoothness.gradient_roughness,
                'num_local_optima': linear_smoothness.num_local_optima
            },
            'harmonic_smoothness': {
                'spearman_correlation': harmonic_smoothness.spearman_correlation,
                'local_variance': harmonic_smoothness.local_variance,
                'gradient_roughness': harmonic_smoothness.gradient_roughness,
                'num_local_optima': harmonic_smoothness.num_local_optima
            },
            'distribution_stats': distribution_stats,
            'execution_time': domain_time
        }
        
        print(f"  🏁 Domain completed in {domain_time:.1f}s")
        print(f"  📊 Linear smoothness: ρ={linear_smoothness.spearman_correlation:.3f}, "
              f"local_var={linear_smoothness.local_variance:.6f}, "
              f"optima={linear_smoothness.num_local_optima}")
        print(f"  📊 Harmonic smoothness: ρ={harmonic_smoothness.spearman_correlation:.3f}, "
              f"local_var={harmonic_smoothness.local_variance:.6f}, "
              f"optima={harmonic_smoothness.num_local_optima}")
    
    return results


def main():
    """Main execution following Rule 4: Test on representative subset."""
    
    # Start with three representative domains (Rule 4)
    test_domains = ['applied_mathematics', 'computer_vision', 'deep_learning']
    
    # Generate timestamp for results file
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    
    try:
        # Run experiment
        results = run_landscape_sampling_experiment(test_domains, n_samples=400)
        
        # Save results (Rule 7: Complete documentation)
        results_file = f"experiments/metric_evaluation/results/landscape_sampling_{timestamp}.json"
        Path(results_file).parent.mkdir(parents=True, exist_ok=True)
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n🎉 Experiment completed successfully!")
        print(f"📄 Results saved to: {results_file}")
        
        # Summary analysis
        print(f"\n📊 SUMMARY ANALYSIS:")
        for domain, data in results.items():
            print(f"\n{domain.upper()}:")
            print(f"  Success rate: {data['success_rate']:.1%}")
            print(f"  Linear landscape: ρ={data['linear_smoothness']['spearman_correlation']:.3f}, "
                  f"optima={data['linear_smoothness']['num_local_optima']}")
            print(f"  Harmonic landscape: ρ={data['harmonic_smoothness']['spearman_correlation']:.3f}, "
                  f"optima={data['harmonic_smoothness']['num_local_optima']}")
            print(f"  C-D scale ratio: {data['distribution_stats']['scale_ratio']:.2f}")
            print(f"  Mean imbalance: {data['distribution_stats']['mean_imbalance_ratio']:.3f}")
        
    except Exception as e:
        print(f"🚨 Experiment failed: {e}")
        raise  # Rule 6: Fail fast


if __name__ == "__main__":
    main() 