#!/usr/bin/env python3
"""
Phase 11: Comprehensive Citation Detection Analysis

This experiment systematically analyzes citation detection performance across ALL domains
and explores alternative algorithms to PELT for paradigm shift detection.

Research Questions:
1. Why are citation detection improvements minimal across domains?
2. Is PELT fundamentally limited for citation time series analysis?
3. What alternative algorithms might work better?
4. Are there domain-specific patterns that require different approaches?
"""

import sys
import json
import time
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

# Add the project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(Path.cwd()))

from core.data_processing import process_domain_data
from core.shift_signal_detection import (
    detect_citation_structural_breaks
)
from core.shift_signal_detection_original import detect_citation_structural_breaks
from core.utils import discover_available_domains

# Import alternative algorithms
try:
    from scipy import signal
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    import ruptures as rpt
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("⚠️ Some advanced analysis libraries not available")


def analyze_citation_time_series_properties(domain_data, domain_name: str) -> Dict:
    """
    Analyze fundamental properties of citation time series to understand PELT limitations.
    """
    print(f"    📊 ANALYZING TIME SERIES PROPERTIES: {domain_name}")
    
    # Create citation time series
    citation_series = defaultdict(float)
    for paper in domain_data.papers:
        citation_series[paper.pub_year] += paper.cited_by_count
    
    # Get dense and sparse representations
    years = sorted(citation_series.keys())
    sparse_values = [citation_series[year] for year in years]
    
    # Fill dense series with zeros for missing years
    min_year, max_year = min(years), max(years)
    dense_years = list(range(min_year, max_year + 1))
    dense_values = [citation_series.get(year, 0) for year in dense_years]
    
    # Calculate properties
    properties = {
        'domain_name': domain_name,
        'temporal_coverage': max_year - min_year + 1,
        'data_density': len(years) / (max_year - min_year + 1),
        'sparsity_level': 1 - (len(years) / (max_year - min_year + 1)),
        
        # Citation distribution properties
        'citation_range': (min(sparse_values), max(sparse_values)),
        'citation_variance': np.var(sparse_values),
        'citation_skewness': calculate_skewness(sparse_values),
        'citation_kurtosis': calculate_kurtosis(sparse_values),
        
        # Temporal structure
        'zero_years_count': (max_year - min_year + 1) - len(years),
        'max_gap_years': calculate_max_gap(years),
        'avg_gap_years': calculate_avg_gap(years),
        
        # Volatility measures
        'sparse_volatility': np.std(np.diff(sparse_values)) if len(sparse_values) > 1 else 0,
        'dense_volatility': np.std(np.diff(dense_values)) if len(dense_values) > 1 else 0,
        
        # Trend analysis
        'has_exponential_growth': detect_exponential_pattern(sparse_values, years),
        'dominant_frequency': analyze_dominant_frequency(sparse_values) if SCIPY_AVAILABLE else None,
        'autocorrelation_lag1': calculate_autocorrelation(sparse_values, lag=1),
        
        # PELT-specific challenges
        'extreme_value_ratio': max(sparse_values) / (np.mean(sparse_values) + 1e-6),
        'change_point_detectability': assess_change_point_detectability(sparse_values),
        'noise_to_signal_ratio': calculate_noise_to_signal_ratio(sparse_values)
    }
    
    print(f"      📈 Coverage: {properties['temporal_coverage']} years, Density: {properties['data_density']:.2f}")
    print(f"      📊 Citation range: {properties['citation_range'][0]:,.0f} - {properties['citation_range'][1]:,.0f}")
    print(f"      🔄 Extreme ratio: {properties['extreme_value_ratio']:.1f}x, Volatility: {properties['sparse_volatility']:.2f}")
    
    return properties


def detect_changes_alternative_methods(domain_data, domain_name: str) -> Dict:
    """
    Test alternative change point detection methods beyond PELT.
    """
    print(f"    🔬 TESTING ALTERNATIVE ALGORITHMS: {domain_name}")
    
    # Prepare time series
    citation_series = defaultdict(float)
    for paper in domain_data.papers:
        citation_series[paper.pub_year] += paper.cited_by_count
    
    years = sorted(citation_series.keys())
    values = [citation_series[year] for year in years]
    
    if len(values) < 10:  # Need minimum data for meaningful analysis
        return {'error': 'Insufficient data for alternative methods'}
    
    results = {'domain_name': domain_name, 'methods': {}}
    
    # Method 1: PELT (baseline)
    try:
        normalized_values = np.array(values) / max(values)
        algo_pelt = rpt.Pelt(model="l2").fit(normalized_values.reshape(-1, 1))
        pelt_changes = algo_pelt.predict(pen=1.0)
        results['methods']['pelt'] = {
            'change_points': [years[cp-1] for cp in pelt_changes[:-1] if 0 < cp < len(years)],
            'count': len(pelt_changes) - 1,
            'success': True
        }
        print(f"      📍 PELT: {len(pelt_changes)-1} change points")
    except Exception as e:
        results['methods']['pelt'] = {'error': str(e), 'success': False}
    
    # Method 2: Binary Segmentation
    try:
        algo_binseg = rpt.Binseg(model="l2").fit(normalized_values.reshape(-1, 1))
        binseg_changes = algo_binseg.predict(n_bkps=min(5, len(values)//10))
        results['methods']['binary_segmentation'] = {
            'change_points': [years[cp-1] for cp in binseg_changes[:-1] if 0 < cp < len(years)],
            'count': len(binseg_changes) - 1,
            'success': True
        }
        print(f"      📍 BinSeg: {len(binseg_changes)-1} change points")
    except Exception as e:
        results['methods']['binary_segmentation'] = {'error': str(e), 'success': False}
    
    # Method 3: Window-based (sliding window)
    try:
        window_changes = detect_changes_sliding_window(values, years, window_size=5, threshold=0.5)
        results['methods']['sliding_window'] = {
            'change_points': window_changes,
            'count': len(window_changes),
            'success': True
        }
        print(f"      📍 Sliding Window: {len(window_changes)} change points")
    except Exception as e:
        results['methods']['sliding_window'] = {'error': str(e), 'success': False}
    
    # Method 4: Z-score based detection
    try:
        zscore_changes = detect_changes_zscore(values, years, threshold=2.0)
        results['methods']['zscore'] = {
            'change_points': zscore_changes,
            'count': len(zscore_changes),
            'success': True
        }
        print(f"      📍 Z-Score: {len(zscore_changes)} change points")
    except Exception as e:
        results['methods']['zscore'] = {'error': str(e), 'success': False}
    
    # Method 5: Percentile-based detection (regime changes)
    try:
        percentile_changes = detect_changes_percentile_regime(values, years, percentile=90)
        results['methods']['percentile_regime'] = {
            'change_points': percentile_changes,
            'count': len(percentile_changes),
            'success': True
        }
        print(f"      📍 Percentile Regime: {len(percentile_changes)} change points")
    except Exception as e:
        results['methods']['percentile_regime'] = {'error': str(e), 'success': False}
    
    # Method 6: Gradient-based detection
    try:
        gradient_changes = detect_changes_gradient(values, years, sensitivity=0.3)
        results['methods']['gradient'] = {
            'change_points': gradient_changes,
            'count': len(gradient_changes),
            'success': True
        }
        print(f"      📍 Gradient: {len(gradient_changes)} change points")
    except Exception as e:
        results['methods']['gradient'] = {'error': str(e), 'success': False}
    
    return results


def detect_changes_sliding_window(values: List[float], years: List[int], 
                                window_size: int = 5, threshold: float = 0.5) -> List[int]:
    """Sliding window change detection."""
    changes = []
    for i in range(window_size, len(values) - window_size):
        before_window = values[max(0, i-window_size):i]
        after_window = values[i:min(len(values), i+window_size)]
        
        if len(before_window) > 0 and len(after_window) > 0:
            before_mean = np.mean(before_window)
            after_mean = np.mean(after_window)
            
            # Relative change
            if before_mean > 0:
                relative_change = abs(after_mean - before_mean) / before_mean
                if relative_change > threshold:
                    changes.append(years[i])
    
    return changes


def detect_changes_zscore(values: List[float], years: List[int], threshold: float = 2.0) -> List[int]:
    """Z-score based change detection."""
    if len(values) < 3:
        return []
    
    # Calculate rolling z-scores
    changes = []
    rolling_mean = np.mean(values[:3])
    rolling_std = np.std(values[:3])
    
    for i in range(3, len(values)):
        current_value = values[i]
        zscore = abs(current_value - rolling_mean) / (rolling_std + 1e-6)
        
        if zscore > threshold:
            changes.append(years[i])
        
        # Update rolling statistics
        window = values[max(0, i-5):i+1]
        rolling_mean = np.mean(window)
        rolling_std = np.std(window)
    
    return changes


def detect_changes_percentile_regime(values: List[float], years: List[int], percentile: float = 90) -> List[int]:
    """Percentile-based regime change detection."""
    if len(values) < 5:
        return []
    
    # Calculate percentile threshold
    threshold = np.percentile(values, percentile)
    
    changes = []
    in_high_regime = False
    
    for i, (value, year) in enumerate(zip(values, years)):
        was_high_regime = in_high_regime
        in_high_regime = value > threshold
        
        # Regime change detected
        if was_high_regime != in_high_regime and i > 0:
            changes.append(year)
    
    return changes


def detect_changes_gradient(values: List[float], years: List[int], sensitivity: float = 0.3) -> List[int]:
    """Gradient-based change detection."""
    if len(values) < 3:
        return []
    
    # Calculate gradients
    gradients = np.gradient(values)
    grad_std = np.std(gradients)
    threshold = sensitivity * grad_std
    
    changes = []
    for i in range(1, len(gradients)-1):
        # Look for significant gradient changes
        grad_change = abs(gradients[i+1] - gradients[i-1])
        if grad_change > threshold:
            changes.append(years[i])
    
    return changes


def calculate_skewness(values: List[float]) -> float:
    """Calculate skewness of the distribution."""
    if len(values) < 3:
        return 0.0
    
    mean_val = np.mean(values)
    std_val = np.std(values)
    if std_val == 0:
        return 0.0
    
    skew = np.mean([((x - mean_val) / std_val) ** 3 for x in values])
    return skew


def calculate_kurtosis(values: List[float]) -> float:
    """Calculate kurtosis of the distribution."""
    if len(values) < 4:
        return 0.0
    
    mean_val = np.mean(values)
    std_val = np.std(values)
    if std_val == 0:
        return 0.0
    
    kurt = np.mean([((x - mean_val) / std_val) ** 4 for x in values]) - 3
    return kurt


def calculate_max_gap(years: List[int]) -> int:
    """Calculate maximum gap between consecutive years."""
    if len(years) < 2:
        return 0
    gaps = [years[i+1] - years[i] for i in range(len(years)-1)]
    return max(gaps)


def calculate_avg_gap(years: List[int]) -> float:
    """Calculate average gap between consecutive years."""
    if len(years) < 2:
        return 0.0
    gaps = [years[i+1] - years[i] for i in range(len(years)-1)]
    return np.mean(gaps)


def detect_exponential_pattern(values: List[float], years: List[int]) -> bool:
    """Detect if the series follows exponential growth pattern."""
    if len(values) < 5:
        return False
    
    # Try to fit exponential pattern
    try:
        log_values = np.log(np.array(values) + 1)  # +1 to handle zeros
        correlation = np.corrcoef(years, log_values)[0, 1]
        return abs(correlation) > 0.7  # Strong correlation with log-linear pattern
    except:
        return False


def analyze_dominant_frequency(values: List[float]) -> Optional[float]:
    """Analyze dominant frequency in the time series."""
    if not SCIPY_AVAILABLE or len(values) < 10:
        return None
    
    try:
        # Remove trend
        detrended = signal.detrend(values)
        
        # Compute power spectral density
        freqs, psd = signal.periodogram(detrended)
        
        # Find dominant frequency
        dominant_idx = np.argmax(psd[1:]) + 1  # Skip DC component
        return freqs[dominant_idx]
    except:
        return None


def calculate_autocorrelation(values: List[float], lag: int = 1) -> float:
    """Calculate autocorrelation at given lag."""
    if len(values) <= lag:
        return 0.0
    
    try:
        return np.corrcoef(values[:-lag], values[lag:])[0, 1]
    except:
        return 0.0


def assess_change_point_detectability(values: List[float]) -> float:
    """Assess how detectable change points are in the series."""
    if len(values) < 6:
        return 0.0
    
    # Calculate potential change point strength
    detectability_scores = []
    
    for i in range(2, len(values)-2):
        before = values[:i]
        after = values[i:]
        
        before_mean = np.mean(before)
        after_mean = np.mean(after)
        before_std = np.std(before)
        after_std = np.std(after)
        
        # Cohen's d effect size
        pooled_std = np.sqrt((before_std**2 + after_std**2) / 2)
        if pooled_std > 0:
            effect_size = abs(after_mean - before_mean) / pooled_std
            detectability_scores.append(effect_size)
    
    return max(detectability_scores) if detectability_scores else 0.0


def calculate_noise_to_signal_ratio(values: List[float]) -> float:
    """Calculate noise to signal ratio."""
    if len(values) < 3:
        return 1.0
    
    # Estimate signal as smoothed version
    signal_estimate = np.convolve(values, np.ones(3)/3, mode='same')
    
    # Estimate noise as residual
    noise = np.array(values) - signal_estimate
    
    signal_power = np.var(signal_estimate)
    noise_power = np.var(noise)
    
    if signal_power == 0:
        return float('inf')
    
    return noise_power / signal_power


def run_comprehensive_analysis():
    """Run comprehensive citation detection analysis across all domains."""
    print("🚀 PHASE 11: COMPREHENSIVE CITATION DETECTION ANALYSIS")
    print("=" * 70)
    print("Analyzing citation detection across ALL domains + testing PELT alternatives")
    print("=" * 70)
    
    # Discover all available domains
    available_domains = discover_available_domains()
    print(f"📊 Found {len(available_domains)} domains: {', '.join(available_domains)}")
    
    # Results storage
    results = {
        'metadata': {
            'experiment_name': 'Phase 11 Comprehensive Citation Analysis',
            'experiment_date': datetime.now().isoformat(),
            'total_domains': len(available_domains),
            'research_questions': [
                'Why are citation detection improvements minimal?',
                'Is PELT fundamentally limited for citation analysis?',
                'What alternative algorithms work better?',
                'Are there domain-specific patterns?'
            ]
        },
        'domain_analyses': {},
        'cross_domain_patterns': {},
        'algorithm_comparison': {},
        'recommendations': {}
    }
    
    domain_properties = []
    algorithm_results = []
    
    # Analyze each domain
    for i, domain_name in enumerate(available_domains):
        print(f"\n🔍 DOMAIN {i+1}/{len(available_domains)}: {domain_name}")
        print("-" * 50)
        
        try:
            # Load domain data
            result = process_domain_data(domain_name)
            if not result.success:
                print(f"   ❌ Failed to load {domain_name}: {result.error_message}")
                continue
            
            domain_data = result.domain_data
            print(f"   📊 Loaded {len(domain_data.papers)} papers, {len(domain_data.citations)} citations")
            
            # Analyze time series properties
            properties = analyze_citation_time_series_properties(domain_data, domain_name)
            domain_properties.append(properties)
            
            # Test original citation detection
            print(f"\n   🔬 ORIGINAL CITATION DETECTION:")
            start_time = time.time()
            original_signals = detect_citation_structural_breaks(domain_data, domain_name)
            original_runtime = time.time() - start_time
            print(f"   ⏱️  Original: {len(original_signals)} signals in {original_runtime:.3f}s")
            
            # Test refined citation detection
            print(f"\n   🔧 REFINED CITATION DETECTION:")
            start_time = time.time()
            refined_signals = detect_citation_structural_breaks(domain_data, domain_name)
            refined_runtime = time.time() - start_time
            print(f"   ⏱️  Refined: {len(refined_signals)} signals in {refined_runtime:.3f}s")
            
            # Test alternative algorithms
            print(f"\n   🧪 ALTERNATIVE ALGORITHMS:")
            alternative_results = detect_changes_alternative_methods(domain_data, domain_name)
            algorithm_results.append(alternative_results)
            
            # Store domain analysis
            results['domain_analyses'][domain_name] = {
                'properties': properties,
                'original_detection': {
                    'signal_count': len(original_signals),
                    'runtime': original_runtime,
                    'detected_years': [s.year for s in original_signals]
                },
                'refined_detection': {
                    'signal_count': len(refined_signals),
                    'runtime': refined_runtime,
                    'detected_years': [s.year for s in refined_signals]
                },
                'alternative_methods': alternative_results,
                'improvement_analysis': {
                    'signal_change': len(refined_signals) - len(original_signals),
                    'signal_change_percent': ((len(refined_signals) - len(original_signals)) / max(len(original_signals), 1)) * 100,
                    'runtime_improvement': (original_runtime - refined_runtime) / original_runtime * 100 if original_runtime > 0 else 0
                }
            }
            
        except Exception as e:
            print(f"   ❌ Error analyzing {domain_name}: {e}")
            results['domain_analyses'][domain_name] = {'error': str(e)}
    
    # Cross-domain pattern analysis
    print(f"\n📊 CROSS-DOMAIN PATTERN ANALYSIS")
    print("=" * 50)
    results['cross_domain_patterns'] = analyze_cross_domain_patterns(domain_properties, algorithm_results)
    
    # Algorithm comparison
    print(f"\n🏆 ALGORITHM COMPARISON")
    print("=" * 50)
    results['algorithm_comparison'] = compare_algorithms_across_domains(algorithm_results)
    
    # Generate recommendations
    print(f"\n💡 RECOMMENDATIONS")
    print("=" * 50)
    results['recommendations'] = generate_recommendations(results)
    
    return results


def analyze_cross_domain_patterns(domain_properties: List[Dict], algorithm_results: List[Dict]) -> Dict:
    """Analyze patterns across domains to understand PELT limitations."""
    
    patterns = {
        'temporal_coverage_analysis': {},
        'sparsity_impact': {},
        'volatility_patterns': {},
        'extreme_value_impact': {},
        'detectability_analysis': {}
    }
    
    if not domain_properties:
        return patterns
    
    # Group domains by characteristics
    high_coverage = [p for p in domain_properties if p['temporal_coverage'] > 50]
    low_coverage = [p for p in domain_properties if p['temporal_coverage'] <= 50]
    
    sparse_domains = [p for p in domain_properties if p['sparsity_level'] > 0.3]
    dense_domains = [p for p in domain_properties if p['sparsity_level'] <= 0.3]
    
    high_volatility = [p for p in domain_properties if p['sparse_volatility'] > np.median([p['sparse_volatility'] for p in domain_properties])]
    low_volatility = [p for p in domain_properties if p['sparse_volatility'] <= np.median([p['sparse_volatility'] for p in domain_properties])]
    
    patterns['temporal_coverage_analysis'] = {
        'high_coverage_domains': len(high_coverage),
        'low_coverage_domains': len(low_coverage),
        'avg_coverage_high': np.mean([p['temporal_coverage'] for p in high_coverage]) if high_coverage else 0,
        'avg_coverage_low': np.mean([p['temporal_coverage'] for p in low_coverage]) if low_coverage else 0
    }
    
    patterns['sparsity_impact'] = {
        'sparse_domains': len(sparse_domains),
        'dense_domains': len(dense_domains),
        'avg_sparsity_sparse': np.mean([p['sparsity_level'] for p in sparse_domains]) if sparse_domains else 0,
        'avg_sparsity_dense': np.mean([p['sparsity_level'] for p in dense_domains]) if dense_domains else 0
    }
    
    patterns['volatility_patterns'] = {
        'high_volatility_domains': len(high_volatility),
        'low_volatility_domains': len(low_volatility),
        'avg_volatility_high': np.mean([p['sparse_volatility'] for p in high_volatility]) if high_volatility else 0,
        'avg_volatility_low': np.mean([p['sparse_volatility'] for p in low_volatility]) if low_volatility else 0
    }
    
    # Analyze extreme value impact
    extreme_ratios = [p['extreme_value_ratio'] for p in domain_properties]
    patterns['extreme_value_impact'] = {
        'median_extreme_ratio': np.median(extreme_ratios),
        'max_extreme_ratio': max(extreme_ratios),
        'domains_with_extreme_values': len([p for p in domain_properties if p['extreme_value_ratio'] > 100])
    }
    
    # Analyze detectability
    detectability_scores = [p['change_point_detectability'] for p in domain_properties]
    patterns['detectability_analysis'] = {
        'median_detectability': np.median(detectability_scores),
        'low_detectability_domains': len([p for p in domain_properties if p['change_point_detectability'] < 0.5]),
        'high_detectability_domains': len([p for p in domain_properties if p['change_point_detectability'] > 1.0])
    }
    
    print(f"   📊 Coverage: {patterns['temporal_coverage_analysis']['high_coverage_domains']} high, {patterns['temporal_coverage_analysis']['low_coverage_domains']} low")
    print(f"   📊 Sparsity: {patterns['sparsity_impact']['sparse_domains']} sparse, {patterns['sparsity_impact']['dense_domains']} dense")
    print(f"   📊 Detectability: {patterns['detectability_analysis']['low_detectability_domains']} low, {patterns['detectability_analysis']['high_detectability_domains']} high")
    
    return patterns


def compare_algorithms_across_domains(algorithm_results: List[Dict]) -> Dict:
    """Compare different algorithms across all domains."""
    
    if not algorithm_results:
        return {}
    
    # Count successes and signal counts for each method
    method_performance = defaultdict(lambda: {'successes': 0, 'total_signals': 0, 'domains': []})
    
    for result in algorithm_results:
        if 'methods' not in result:
            continue
            
        domain_name = result['domain_name']
        
        for method_name, method_result in result['methods'].items():
            if method_result.get('success', False):
                method_performance[method_name]['successes'] += 1
                method_performance[method_name]['total_signals'] += method_result['count']
                method_performance[method_name]['domains'].append(domain_name)
    
    # Calculate summary statistics
    comparison = {}
    total_domains = len(algorithm_results)
    
    for method_name, perf in method_performance.items():
        comparison[method_name] = {
            'success_rate': perf['successes'] / total_domains,
            'total_signals': perf['total_signals'],
            'avg_signals_per_domain': perf['total_signals'] / max(perf['successes'], 1),
            'successful_domains': perf['domains']
        }
        
        print(f"   🏆 {method_name}: {perf['successes']}/{total_domains} domains, {perf['total_signals']} total signals")
    
    return comparison


def generate_recommendations(results: Dict) -> Dict:
    """Generate actionable recommendations based on analysis."""
    
    recommendations = {
        'pelt_limitations': [],
        'alternative_approaches': [],
        'domain_specific_strategies': [],
        'implementation_priority': []
    }
    
    # Analyze PELT performance
    pelt_success_rate = results['algorithm_comparison'].get('pelt', {}).get('success_rate', 0)
    pelt_avg_signals = results['algorithm_comparison'].get('pelt', {}).get('avg_signals_per_domain', 0)
    
    if pelt_success_rate < 0.8:
        recommendations['pelt_limitations'].append("PELT shows low success rate across domains")
    
    if pelt_avg_signals < 2:
        recommendations['pelt_limitations'].append("PELT produces very few signals per domain - potential under-sensitivity")
    
    # Compare alternative methods
    best_method = None
    best_score = 0
    
    for method, perf in results['algorithm_comparison'].items():
        if method != 'pelt':
            score = perf['success_rate'] * perf['avg_signals_per_domain']
            if score > best_score:
                best_score = score
                best_method = method
    
    if best_method and best_score > pelt_success_rate * pelt_avg_signals:
        recommendations['alternative_approaches'].append(f"Consider {best_method} as primary algorithm - shows better performance")
    
    # Analyze cross-domain patterns
    patterns = results.get('cross_domain_patterns', {})
    
    if patterns.get('detectability_analysis', {}).get('low_detectability_domains', 0) > len(results['domain_analyses']) / 2:
        recommendations['domain_specific_strategies'].append("Many domains have low change point detectability - consider ensemble methods")
    
    if patterns.get('sparsity_impact', {}).get('sparse_domains', 0) > len(results['domain_analyses']) / 2:
        recommendations['domain_specific_strategies'].append("High sparsity across domains - consider gap-filling or different time series representations")
    
    # Implementation priorities
    if len(recommendations['pelt_limitations']) > 0:
        recommendations['implementation_priority'].append("HIGH: Address PELT limitations")
    
    if best_method:
        recommendations['implementation_priority'].append(f"MEDIUM: Implement {best_method} as alternative")
    
    recommendations['implementation_priority'].append("LOW: Fine-tune existing approaches")
    
    print(f"   💡 Key finding: PELT success rate {pelt_success_rate:.1%}, avg signals {pelt_avg_signals:.1f}")
    if best_method:
        print(f"   💡 Best alternative: {best_method} (score: {best_score:.2f})")
    
    return recommendations


def save_comprehensive_results(results: Dict) -> str:
    """Save comprehensive analysis results."""
    output_dir = Path("experiments/phase11/results")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = output_dir / f"comprehensive_citation_analysis_{timestamp}.json"
    
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 COMPREHENSIVE RESULTS SAVED:")
    print(f"   📁 File: {filepath}")
    
    return str(filepath)


def main():
    """Main experiment execution."""
    print("Starting Phase 11 Comprehensive Citation Detection Analysis...")
    
    # Run comprehensive analysis
    results = run_comprehensive_analysis()
    
    # Save results
    results_file = save_comprehensive_results(results)
    
    print(f"\n✅ PHASE 11 ANALYSIS COMPLETED")
    print(f"📁 Results saved to: {results_file}")
    print("\n🎯 Key insights will help determine if PELT should be replaced")


if __name__ == "__main__":
    main() 