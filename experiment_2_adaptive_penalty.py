#!/usr/bin/env python3
"""
Experiment 2: Adaptive Penalty Validation Study
Academic Study of Timeline Segmentation Algorithm

This experiment systematically evaluates the impact of penalty estimation methods
on segmentation quality, comparing fixed vs adaptive approaches across domains.

Research Question:
Does adaptive penalty estimation significantly improve segmentation quality 
compared to fixed penalty approaches?

Author: Research Team
Date: 2024
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict
import time
from scipy import stats

# Import our core modules
import sys
sys.path.append('.')
from core.data_models import DomainData
from core.shift_signal_detection import detect_shift_signals
from core.change_detection import detect_changes, create_improved_segments_with_confidence
from core.data_processing import process_domain_data

@dataclass
class PenaltyCondition:
    """Penalty estimation experimental condition configuration."""
    name: str
    method: str  # 'fixed' or 'adaptive'
    penalty_value: Optional[float] = None  # For fixed methods
    bounds: Optional[Tuple[float, float]] = None  # For adaptive methods
    description: str = ""

@dataclass
class SegmentationResult:
    """Results for a single penalty condition experiment."""
    condition_name: str
    domain_name: str
    segment_count: int
    segments: List[List[int]]  # [start_year, end_year] pairs
    average_segment_length: float
    over_segmentation_rate: float  # % segments < min_length
    under_segmentation_rate: float  # % missing reference boundaries
    statistical_significance: float
    detected_change_points: List[int]
    temporal_error: float  # Mean distance to nearest reference boundary
    execution_time: float
    raw_penalty_values: List[float]  # For adaptive methods
    domain_characteristics: Dict  # Series variance, volatility, etc.

class AdaptivePenaltyExperiment:
    """
    Comprehensive experimental framework for evaluating adaptive penalty estimation
    against fixed penalty approaches in timeline segmentation.
    """
    
    def __init__(self):
        # Use all 8 domains for comprehensive analysis
        self.domains = [
            'natural_language_processing',
            'deep_learning', 
            'computer_vision',
            'machine_learning',
            'machine_translation',
            'computer_science',
            'art',
            'applied_mathematics'
        ]
        
        # Define experimental conditions for penalty estimation
        self.conditions = [
            PenaltyCondition(
                name="Fixed_Low",
                method="fixed",
                penalty_value=0.5,
                description="Low fixed penalty - should create over-segmentation"
            ),
            PenaltyCondition(
                name="Fixed_Medium", 
                method="fixed",
                penalty_value=1.0,
                description="Medium fixed penalty - traditional baseline"
            ),
            PenaltyCondition(
                name="Fixed_High",
                method="fixed",
                penalty_value=2.0, 
                description="High fixed penalty - should create under-segmentation"
            ),
            PenaltyCondition(
                name="Adaptive_Standard",
                method="adaptive",
                bounds=(0.8, 6.0),
                description="Current algorithm bounds - standard adaptive approach"
            ),
            PenaltyCondition(
                name="Adaptive_Conservative",
                method="adaptive", 
                bounds=(1.0, 6.0),
                description="Conservative adaptive - higher minimum penalty"
            ),
            PenaltyCondition(
                name="Adaptive_Sensitive",
                method="adaptive",
                bounds=(0.5, 4.0),
                description="Sensitive adaptive - lower bounds for more detection"
            )
        ]
        
        # Load reference periods for evaluation
        self.reference_periods = self._load_reference_periods()
        
        # Results storage
        self.results: List[SegmentationResult] = []
        
        print(f"🔬 EXPERIMENT 2: ADAPTIVE PENALTY VALIDATION STUDY")
        print(f"   📊 Testing {len(self.conditions)} penalty conditions across {len(self.domains)} domains")
        print(f"   🎯 Total experiments: {len(self.conditions) * len(self.domains)}")
    
    def _load_reference_periods(self) -> Dict[str, List[Tuple[int, int]]]:
        """Load reference period boundaries for temporal alignment evaluation."""
        reference_periods = {}
        
        for domain in self.domains:
            try:
                validation_file = f"validation/{domain}_groundtruth.json" 
                with open(validation_file, 'r') as f:
                    data = json.load(f)
                
                periods = []
                for period in data:
                    start_year = period['start_year']
                    end_year = period['end_year']
                    periods.append((start_year, end_year))
                
                reference_periods[domain] = periods
                print(f"   📚 Loaded {len(periods)} reference periods for {domain}")
                
            except Exception as e:
                print(f"   ⚠️ Could not load reference periods for {domain}: {e}")
                reference_periods[domain] = []
        
        return reference_periods
    
    def run_single_condition(self, domain_name: str, condition: PenaltyCondition) -> SegmentationResult:
        """
        Run a single penalty estimation experiment for one domain.
        
        Returns detailed segmentation quality metrics for analysis.
        """
        start_time = time.time()
        
        print(f"\n🔬 PENALTY CONDITION: {condition.name}")
        print(f"   Domain: {domain_name}")
        print(f"   Method: {condition.method}")
        if condition.method == "fixed":
            print(f"   Penalty: {condition.penalty_value}")
        else:
            print(f"   Bounds: {condition.bounds}")
        
        try:
            # Load domain data
            result = process_domain_data(domain_name)
            if not result.success:
                raise Exception(f"Failed to load domain data: {result.error_message}")
            
            domain_data = result.domain_data
            
            # Modify penalty estimation based on condition
            if condition.method == "fixed":
                # Temporarily override adaptive penalty function
                segments, change_points, significance, penalties, characteristics = self._run_fixed_penalty_experiment(
                    domain_data, condition.penalty_value
                )
            else:
                # Use adaptive penalty with specified bounds
                segments, change_points, significance, penalties, characteristics = self._run_adaptive_penalty_experiment(
                    domain_data, condition.bounds
                )
            
            # Calculate segmentation quality metrics
            segment_count = len(segments)
            avg_length = np.mean([seg[1] - seg[0] + 1 for seg in segments]) if segments else 0
            
            # Over-segmentation: segments shorter than 3 years (too granular)
            min_length = 3
            short_segments = [seg for seg in segments if (seg[1] - seg[0] + 1) < min_length]
            over_seg_rate = len(short_segments) / max(len(segments), 1) * 100
            
            # Under-segmentation: missing reference boundaries
            under_seg_rate = self._calculate_under_segmentation_rate(segments, domain_name)
            
            # Temporal alignment error
            temporal_error = self._calculate_temporal_error(change_points, domain_name)
            
            print(f"   ✅ Results: {segment_count} segments, avg={avg_length:.1f}y, over-seg={over_seg_rate:.1f}%, temporal_error={temporal_error:.1f}y")
            
            return SegmentationResult(
                condition_name=condition.name,
                domain_name=domain_name,
                segment_count=segment_count,
                segments=segments,
                average_segment_length=avg_length,
                over_segmentation_rate=over_seg_rate,
                under_segmentation_rate=under_seg_rate,
                statistical_significance=significance,
                detected_change_points=change_points,
                temporal_error=temporal_error,
                execution_time=time.time() - start_time,
                raw_penalty_values=penalties,
                domain_characteristics=characteristics
            )
            
        except Exception as e:
            print(f"   ❌ Error in {condition.name} for {domain_name}: {e}")
            return SegmentationResult(
                condition_name=condition.name,
                domain_name=domain_name,
                segment_count=0,
                segments=[],
                average_segment_length=0.0,
                over_segmentation_rate=0.0,
                under_segmentation_rate=0.0,
                statistical_significance=0.0,
                detected_change_points=[],
                temporal_error=999.0,  # High error for failed experiments
                execution_time=time.time() - start_time,
                raw_penalty_values=[],
                domain_characteristics={}
            )
    
    def _run_fixed_penalty_experiment(self, domain_data: DomainData, penalty_value: float) -> Tuple:
        """Run experiment with fixed penalty value."""
        
        # Patch the penalty estimation function temporarily
        import core.shift_signal_detection as ssd
        original_estimate_penalty = ssd.estimate_optimal_penalty
        
        def fixed_penalty_override(normalized_series, domain_name):
            return penalty_value
            
        ssd.estimate_optimal_penalty = fixed_penalty_override
        
        try:
            # Run shift signal detection with fixed penalty
            shift_signals, _ = detect_shift_signals(domain_data, domain_data.domain_name)
            
            # Extract change points
            change_points = [s.year for s in shift_signals]
            
            # Create segments using standard merging (no penalty impact here)
            segments = create_improved_segments_with_confidence(
                change_points, domain_data.year_range, 
                statistical_significance=0.5,  # Standard value
                domain_name=domain_data.domain_name
            )
            
            # Calculate characteristics for analysis
            characteristics = self._calculate_domain_characteristics(domain_data)
            
            return segments, change_points, 0.5, [penalty_value], characteristics
            
        finally:
            # Restore original function
            ssd.estimate_optimal_penalty = original_estimate_penalty
    
    def _run_adaptive_penalty_experiment(self, domain_data: DomainData, bounds: Tuple[float, float]) -> Tuple:
        """Run experiment with adaptive penalty within specified bounds."""
        
        # Patch the penalty estimation function temporarily  
        import core.shift_signal_detection as ssd
        original_estimate_penalty = ssd.estimate_optimal_penalty
        
        def bounded_adaptive_penalty(normalized_series, domain_name):
            # Calculate adaptive penalty using original algorithm
            adaptive_penalty = original_estimate_penalty(normalized_series, domain_name)
            # Apply experimental bounds
            return np.clip(adaptive_penalty, bounds[0], bounds[1])
            
        ssd.estimate_optimal_penalty = bounded_adaptive_penalty
        
        try:
            # Run shift signal detection with bounded adaptive penalty
            shift_signals, _ = detect_shift_signals(domain_data, domain_data.domain_name)
            
            # Extract change points
            change_points = [s.year for s in shift_signals]
            
            # Create segments
            segments = create_improved_segments_with_confidence(
                change_points, domain_data.year_range,
                statistical_significance=0.5,
                domain_name=domain_data.domain_name
            )
            
            # Calculate characteristics
            characteristics = self._calculate_domain_characteristics(domain_data)
            
            # Get penalty values that were actually used (approximate)
            penalties = [bounded_adaptive_penalty(np.array([1.0]), domain_data.domain_name)]
            
            return segments, change_points, 0.5, penalties, characteristics
            
        finally:
            # Restore original function
            ssd.estimate_optimal_penalty = original_estimate_penalty
    
    def _calculate_domain_characteristics(self, domain_data: DomainData) -> Dict:
        """Calculate domain characteristics for analysis."""
        
        # Create citation time series
        citation_series = defaultdict(float)
        all_years = [p.pub_year for p in domain_data.papers]
        if not all_years:
            return {}
            
        min_year, max_year = min(all_years), max(all_years)
        
        for year in range(min_year, max_year + 1):
            citation_series[year] = 0.0
        
        for paper in domain_data.papers:
            citation_series[paper.pub_year] += paper.cited_by_count
        
        # Calculate characteristics
        values = list(citation_series.values())
        if not values or max(values) == 0:
            return {}
            
        normalized_values = np.array(values) / max(values)
        
        return {
            'series_variance': float(np.var(normalized_values)),
            'series_mean': float(np.mean(normalized_values)),
            'series_std': float(np.std(normalized_values)),
            'temporal_volatility': float(np.mean(np.abs(np.diff(normalized_values)))) if len(normalized_values) > 1 else 0.0,
            'coefficient_variation': float(np.std(normalized_values) / (np.mean(normalized_values) + 1e-6)),
            'non_zero_ratio': float(np.count_nonzero(normalized_values) / len(normalized_values)),
            'series_length': len(normalized_values)
        }
    
    def _calculate_under_segmentation_rate(self, segments: List[List[int]], domain_name: str) -> float:
        """Calculate under-segmentation rate based on missing reference boundaries."""
        
        if domain_name not in self.reference_periods:
            return 0.0
            
        reference_boundaries = []
        for start, end in self.reference_periods[domain_name]:
            if start > min([seg[0] for seg in segments]):  # Not the first period
                reference_boundaries.append(start)
        
        if not reference_boundaries:
            return 0.0
            
        # Check how many reference boundaries are missing
        detected_boundaries = set()
        for seg in segments[1:]:  # Skip first segment
            detected_boundaries.add(seg[0])
        
        missing_boundaries = 0
        for ref_boundary in reference_boundaries:
            # Allow ±2 year tolerance
            if not any(abs(ref_boundary - det) <= 2 for det in detected_boundaries):
                missing_boundaries += 1
        
        return (missing_boundaries / len(reference_boundaries)) * 100
    
    def _calculate_temporal_error(self, change_points: List[int], domain_name: str) -> float:
        """Calculate mean temporal error to nearest reference boundary."""
        
        if domain_name not in self.reference_periods or not change_points:
            return 0.0
            
        reference_boundaries = []
        for start, end in self.reference_periods[domain_name]:
            if start > 0:  # Not the first period  
                reference_boundaries.append(start)
        
        if not reference_boundaries:
            return 0.0
            
        errors = []
        for cp in change_points:
            min_error = min(abs(cp - ref) for ref in reference_boundaries)
            errors.append(min_error)
        
        return np.mean(errors) if errors else 0.0
    
    def run_full_experiment(self) -> pd.DataFrame:
        """
        Run the complete penalty estimation ablation study.
        
        Returns comprehensive results dataframe for analysis.
        """
        print("=" * 80)
        print("🧪 EXPERIMENT 2: ADAPTIVE PENALTY VALIDATION STUDY")
        print("=" * 80)
        
        all_results = []
        
        for domain in self.domains:
            print(f"\n📊 DOMAIN: {domain}")
            print("-" * 40)
            
            for condition in self.conditions:
                result = self.run_single_condition(domain, condition)
                all_results.append(result)
                self.results.append(result)
        
        # Convert to DataFrame for analysis
        results_df = pd.DataFrame([
            {
                'Domain': r.domain_name,
                'Condition': r.condition_name,
                'Method': 'Fixed' if 'Fixed' in r.condition_name else 'Adaptive',
                'Segment_Count': r.segment_count,
                'Average_Length': r.average_segment_length,
                'Over_Segmentation_Rate': r.over_segmentation_rate,
                'Under_Segmentation_Rate': r.under_segmentation_rate,
                'Temporal_Error': r.temporal_error,
                'Statistical_Significance': r.statistical_significance,
                'Execution_Time': r.execution_time,
                'Domain_Variance': r.domain_characteristics.get('series_variance', 0),
                'Domain_Volatility': r.domain_characteristics.get('temporal_volatility', 0),
                'Domain_CV': r.domain_characteristics.get('coefficient_variation', 0)
            }
            for r in all_results
        ])
        
        return results_df
    
    def create_comprehensive_visualizations(self, results_df: pd.DataFrame) -> None:
        """Create comprehensive visualizations for penalty estimation analysis."""
        
        # Set up matplotlib style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Create output directory
        output_dir = Path("experiment_results/experiment_2_visualizations")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Segmentation Quality Comparison (Fixed vs Adaptive)
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Segment count comparison
        sns.boxplot(data=results_df, x='Method', y='Segment_Count', ax=axes[0,0])
        axes[0,0].set_title('Segment Count Distribution\n(Fixed vs Adaptive)', fontweight='bold')
        axes[0,0].set_ylabel('Number of Segments')
        
        # Average segment length comparison
        sns.boxplot(data=results_df, x='Method', y='Average_Length', ax=axes[0,1])
        axes[0,1].set_title('Average Segment Length\n(Fixed vs Adaptive)', fontweight='bold')
        axes[0,1].set_ylabel('Average Length (Years)')
        
        # Over-segmentation rate comparison
        sns.boxplot(data=results_df, x='Method', y='Over_Segmentation_Rate', ax=axes[1,0])
        axes[1,0].set_title('Over-Segmentation Rate\n(% segments < 3 years)', fontweight='bold')
        axes[1,0].set_ylabel('Over-Segmentation Rate (%)')
        
        # Temporal error comparison
        sns.boxplot(data=results_df, x='Method', y='Temporal_Error', ax=axes[1,1])
        axes[1,1].set_title('Temporal Alignment Error\n(Years to nearest reference)', fontweight='bold')
        axes[1,1].set_ylabel('Mean Temporal Error (Years)')
        
        plt.suptitle('Experiment 2: Fixed vs Adaptive Penalty Comparison', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_dir / 'fixed_vs_adaptive_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Detailed Condition Analysis Heatmap
        pivot_segments = results_df.pivot(index='Domain', columns='Condition', values='Segment_Count')
        
        plt.figure(figsize=(14, 8))
        sns.heatmap(pivot_segments, annot=True, fmt='d', cmap='YlOrRd', cbar_kws={'label': 'Segment Count'})
        plt.title('Segment Count by Domain and Penalty Condition', fontsize=16, fontweight='bold')
        plt.xlabel('Penalty Condition')
        plt.ylabel('Research Domain')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(output_dir / 'penalty_condition_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Domain Characteristics vs Performance
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Variance vs Over-segmentation
        for method in ['Fixed', 'Adaptive']:
            method_data = results_df[results_df['Method'] == method]
            axes[0].scatter(method_data['Domain_Variance'], method_data['Over_Segmentation_Rate'], 
                           label=method, alpha=0.7, s=60)
        axes[0].set_xlabel('Domain Variance')
        axes[0].set_ylabel('Over-Segmentation Rate (%)')
        axes[0].set_title('Domain Variance vs Over-Segmentation', fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Volatility vs Temporal Error
        for method in ['Fixed', 'Adaptive']:
            method_data = results_df[results_df['Method'] == method]
            axes[1].scatter(method_data['Domain_Volatility'], method_data['Temporal_Error'],
                           label=method, alpha=0.7, s=60)
        axes[1].set_xlabel('Domain Volatility')
        axes[1].set_ylabel('Temporal Error (Years)')
        axes[1].set_title('Domain Volatility vs Temporal Error', fontweight='bold')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # CV vs Segment Count
        for method in ['Fixed', 'Adaptive']:
            method_data = results_df[results_df['Method'] == method]
            axes[2].scatter(method_data['Domain_CV'], method_data['Segment_Count'],
                           label=method, alpha=0.7, s=60)
        axes[2].set_xlabel('Coefficient of Variation')
        axes[2].set_ylabel('Segment Count')
        axes[2].set_title('Domain CV vs Segment Count', fontweight='bold')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        plt.suptitle('Domain Characteristics vs Penalty Method Performance', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_dir / 'domain_characteristics_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 4. Penalty Sensitivity Analysis
        conditions_order = ['Fixed_Low', 'Fixed_Medium', 'Fixed_High', 'Adaptive_Sensitive', 'Adaptive_Standard', 'Adaptive_Conservative']
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        
        # Quality metrics across conditions
        metrics = ['Segment_Count', 'Average_Length', 'Over_Segmentation_Rate', 'Temporal_Error']
        metric_titles = ['Segment Count', 'Average Length (Years)', 'Over-Segmentation Rate (%)', 'Temporal Error (Years)']
        
        for i, (metric, title) in enumerate(zip(metrics, metric_titles)):
            ax = axes[i//2, i%2]
            
            # Calculate means and stds for each condition
            condition_stats = []
            for condition in conditions_order:
                if condition in results_df['Condition'].values:
                    condition_data = results_df[results_df['Condition'] == condition][metric]
                    condition_stats.append({
                        'Condition': condition,
                        'Mean': condition_data.mean(),
                        'Std': condition_data.std()
                    })
            
            if condition_stats:
                stats_df = pd.DataFrame(condition_stats)
                bars = ax.bar(stats_df['Condition'], stats_df['Mean'], 
                             yerr=stats_df['Std'], capsize=5, alpha=0.8)
                
                # Color bars by method type
                for j, condition in enumerate(stats_df['Condition']):
                    if 'Fixed' in condition:
                        bars[j].set_color('lightcoral')
                    else:
                        bars[j].set_color('steelblue')
            
            ax.set_title(title, fontweight='bold')
            ax.set_xlabel('Penalty Condition')
            ax.set_ylabel(title)
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3)
        
        plt.suptitle('Penalty Sensitivity Analysis Across Conditions', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_dir / 'penalty_sensitivity_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Created 4 comprehensive visualizations in {output_dir}")
        print(f"   1. Fixed vs Adaptive Comparison")
        print(f"   2. Penalty Condition Heatmap")
        print(f"   3. Domain Characteristics Analysis")
        print(f"   4. Penalty Sensitivity Analysis")
    
    def calculate_statistical_significance(self, results_df: pd.DataFrame) -> Dict:
        """Calculate statistical significance of adaptive vs fixed penalty approaches."""
        
        fixed_data = results_df[results_df['Method'] == 'Fixed']
        adaptive_data = results_df[results_df['Method'] == 'Adaptive']
        
        metrics = ['Segment_Count', 'Average_Length', 'Over_Segmentation_Rate', 'Temporal_Error']
        significance_results = {}
        
        for metric in metrics:
            fixed_values = fixed_data[metric].values
            adaptive_values = adaptive_data[metric].values
            
            # Perform t-test
            t_stat, p_value = stats.ttest_ind(fixed_values, adaptive_values)
            
            # Calculate effect size (Cohen's d)
            pooled_std = np.sqrt(((len(fixed_values) - 1) * np.var(fixed_values, ddof=1) + 
                                 (len(adaptive_values) - 1) * np.var(adaptive_values, ddof=1)) / 
                                (len(fixed_values) + len(adaptive_values) - 2))
            
            cohens_d = (np.mean(adaptive_values) - np.mean(fixed_values)) / pooled_std
            
            significance_results[metric] = {
                't_statistic': float(t_stat),
                'p_value': float(p_value),
                'cohens_d': float(cohens_d) if not np.isnan(cohens_d) else 0.0,
                'fixed_mean': float(np.mean(fixed_values)),
                'adaptive_mean': float(np.mean(adaptive_values)),
                'significant': bool(p_value < 0.05)
            }
        
        return significance_results
    
    def save_results(self, results_df: pd.DataFrame, significance_results: Dict) -> Path:
        """Save comprehensive experimental results for academic paper."""
        
        output_dir = Path("experiment_results/experiment_2")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save raw results
        results_df.to_csv(output_dir / "penalty_experiment_results.csv", index=False)
        
        # Save significance analysis
        with open(output_dir / "statistical_significance.json", 'w') as f:
            json.dump(significance_results, f, indent=2)
        
        # Create summary statistics
        summary_stats = {
            'experimental_overview': {
                'total_experiments': len(results_df),
                'domains_tested': len(results_df['Domain'].unique()),
                'conditions_tested': len(results_df['Condition'].unique()),
                'fixed_conditions': len(results_df[results_df['Method'] == 'Fixed']),
                'adaptive_conditions': len(results_df[results_df['Method'] == 'Adaptive'])
            },
            'performance_comparison': {
                'fixed_method_stats': {
                    'avg_segments': float(results_df[results_df['Method'] == 'Fixed']['Segment_Count'].mean()),
                    'avg_length': float(results_df[results_df['Method'] == 'Fixed']['Average_Length'].mean()),
                    'avg_over_seg': float(results_df[results_df['Method'] == 'Fixed']['Over_Segmentation_Rate'].mean()),
                    'avg_temporal_error': float(results_df[results_df['Method'] == 'Fixed']['Temporal_Error'].mean())
                },
                'adaptive_method_stats': {
                    'avg_segments': float(results_df[results_df['Method'] == 'Adaptive']['Segment_Count'].mean()),
                    'avg_length': float(results_df[results_df['Method'] == 'Adaptive']['Average_Length'].mean()),
                    'avg_over_seg': float(results_df[results_df['Method'] == 'Adaptive']['Over_Segmentation_Rate'].mean()),
                    'avg_temporal_error': float(results_df[results_df['Method'] == 'Adaptive']['Temporal_Error'].mean())
                }
            },
            'best_performing_conditions': {
                'lowest_over_segmentation': results_df.loc[results_df['Over_Segmentation_Rate'].idxmin()][['Condition', 'Domain', 'Over_Segmentation_Rate']].to_dict(),
                'lowest_temporal_error': results_df.loc[results_df['Temporal_Error'].idxmin()][['Condition', 'Domain', 'Temporal_Error']].to_dict(),
                'optimal_segment_count': results_df.loc[(results_df['Segment_Count'] >= 4) & (results_df['Segment_Count'] <= 8)].groupby('Condition')['Domain'].count().to_dict()
            }
        }
        
        with open(output_dir / "summary_statistics.json", 'w') as f:
            json.dump(summary_stats, f, indent=2)
        
        print(f"📊 COMPREHENSIVE RESULTS SAVED: {output_dir}")
        print(f"   • Experimental results: penalty_experiment_results.csv")
        print(f"   • Statistical significance: statistical_significance.json")
        print(f"   • Summary statistics: summary_statistics.json")
        
        return output_dir

def main():
    """Run Experiment 2 with comprehensive analysis."""
    
    # Initialize experiment
    experiment = AdaptivePenaltyExperiment()
    
    # Run full experimental matrix
    results_df = experiment.run_full_experiment()
    
    # Calculate statistical significance
    significance_results = experiment.calculate_statistical_significance(results_df)
    
    # Create comprehensive visualizations
    experiment.create_comprehensive_visualizations(results_df)
    
    # Save all results
    output_dir = experiment.save_results(results_df, significance_results)
    
    # Print key findings for immediate review
    print("\n" + "=" * 80)
    print("🔬 EXPERIMENT 2 KEY RESEARCH FINDINGS")
    print("=" * 80)
    
    # Method comparison
    fixed_stats = results_df[results_df['Method'] == 'Fixed']
    adaptive_stats = results_df[results_df['Method'] == 'Adaptive']
    
    print(f"\n📊 FIXED vs ADAPTIVE PENALTY COMPARISON:")
    print(f"   Fixed Method    - Avg Segments: {fixed_stats['Segment_Count'].mean():.1f}, Temporal Error: {fixed_stats['Temporal_Error'].mean():.1f}y")
    print(f"   Adaptive Method - Avg Segments: {adaptive_stats['Segment_Count'].mean():.1f}, Temporal Error: {adaptive_stats['Temporal_Error'].mean():.1f}y")
    
    print(f"\n🎯 STATISTICAL SIGNIFICANCE RESULTS:")
    for metric, result in significance_results.items():
        significance = "SIGNIFICANT" if result['significant'] else "NOT SIGNIFICANT"
        print(f"   {metric:25s}: p={result['p_value']:.4f}, Cohen's d={result['cohens_d']:.3f} ({significance})")
    
    print(f"\n📁 Detailed results available in: {output_dir}")

if __name__ == "__main__":
    main() 