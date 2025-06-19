#!/usr/bin/env python3
"""
Experiment 3: Paradigm Significance Filtering Impact Analysis
Tests whether paradigm filtering shows sensitivity where penalty optimization does not.
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List
from dataclasses import dataclass
import time
from scipy import stats

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))  # Add project root to path
from core.data_models import DomainData
from core.shift_signal_detection import detect_shift_signals
from core.change_detection import create_segments_with_confidence
from core.data_processing import process_domain_data

@dataclass
class FilteringCondition:
    name: str
    paradigm_filtering: bool
    breakthrough_weighting: bool
    description: str

class ParadigmFilteringExperiment:
    """Test paradigm filtering sensitivity across domains."""
    
    def __init__(self):
        # Phase-10: focus on the five technical domains processed in two-signal architecture
        self.domains = [
            'natural_language_processing',
            'deep_learning',
            'computer_vision',
            'machine_learning',
            'machine_translation'
        ]
        
        self.conditions = [
            FilteringCondition("No_Filtering", False, False, "No filtering - raw signals"),
            FilteringCondition("Pattern_Only", True, False, "Pattern filtering only"),
            FilteringCondition("Breakthrough_Only", False, True, "Breakthrough weighting only"),
            FilteringCondition("Full_Filtering", True, True, "Complete filtering")
        ]
        
        print(f"🔬 EXPERIMENT 3: PARADIGM FILTERING IMPACT ANALYSIS")
        print(f"   📊 Testing {len(self.conditions)} conditions across {len(self.domains)} domains")
        print(f"   🎯 Key Question: Does filtering show sensitivity unlike penalty optimization?")
    
    def run_single_condition(self, domain_name: str, condition: FilteringCondition) -> Dict:
        """Run single filtering experiment."""
        start_time = time.time()
        
        print(f"\n🔬 CONDITION: {condition.name} - Domain: {domain_name}")
        
        try:
            result = process_domain_data(domain_name)
            if not result.success:
                raise Exception(f"Failed to load domain data: {result.error_message}")
            
            domain_data = result.domain_data
            
            # Apply experimental filtering
            metrics = self._run_filtering_test(domain_data, condition)
            
            print(f"   ✅ Paradigm shifts: {metrics['paradigm_shifts']}, "
                  f"Effectiveness: {metrics['effectiveness']:.3f}")
            
            return {
                'Domain': domain_name,
                'Condition': condition.name,
                'Paradigm_Shifts': metrics['paradigm_shifts'],
                'Avg_Significance': metrics['avg_significance'],
                'Breakthrough_Alignment': metrics['breakthrough_alignment'],
                'Filtering_Effectiveness': metrics['effectiveness'],
                'Execution_Time': time.time() - start_time
            }
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            return {
                'Domain': domain_name, 'Condition': condition.name,
                'Paradigm_Shifts': 0, 'Avg_Significance': 0.0,
                'Breakthrough_Alignment': 0.0, 'Filtering_Effectiveness': 0.0,
                'Execution_Time': time.time() - start_time
            }
    
    def _run_filtering_test(self, domain_data: DomainData, condition: FilteringCondition) -> Dict:
        """Test filtering condition and calculate metrics."""
        
        # Get standard signals as baseline
        standard_signals, _ = detect_shift_signals(
            domain_data,
            domain_data.domain_name,
            use_citation=True,
            use_semantic=False,
            use_direction=True
        )
        
        # Apply experimental filtering
        if condition.name == "No_Filtering":
            paradigm_shifts = standard_signals
        else:
            # Modify filtering behavior
            import core.shift_signal_detection as ssd
            original_filter = ssd.filter_for_paradigm_significance
            
            def experimental_filter(signals, domain_data, domain_name):
                if condition.name == "Pattern_Only":
                    # Apply threshold filtering only
                    return [s for s in signals if s.paradigm_significance >= 0.4]
                elif condition.name == "Breakthrough_Only":
                    # Apply breakthrough weighting only  
                    breakthrough_papers = ssd.load_breakthrough_papers(domain_data, domain_name)
                    breakthrough_years = {p.pub_year for p in breakthrough_papers}
                    
                    boosted_signals = []
                    for signal in signals:
                        score = signal.paradigm_significance
                        if any(abs(signal.year - by) <= 2 for by in breakthrough_years):
                            score += 0.3
                        
                        boosted_signal = ssd.ShiftSignal(
                            year=signal.year, confidence=signal.confidence,
                            signal_type=signal.signal_type, 
                            evidence_strength=signal.evidence_strength,
                            supporting_evidence=signal.supporting_evidence,
                            contributing_papers=signal.contributing_papers,
                            transition_description=signal.transition_description,
                            paradigm_significance=score
                        )
                        boosted_signals.append(boosted_signal)
                    return boosted_signals
                else:  # Full_Filtering
                    return original_filter(signals, domain_data, domain_name)
            
            ssd.filter_for_paradigm_significance = experimental_filter
            
            try:
                paradigm_shifts, _ = detect_shift_signals(
                    domain_data,
                    domain_data.domain_name,
                    use_citation=True,
                    use_semantic=False,
                    use_direction=True
                )
            finally:
                ssd.filter_for_paradigm_significance = original_filter
        
        # Calculate metrics
        paradigm_count = len(paradigm_shifts)
        avg_significance = np.mean([s.paradigm_significance for s in paradigm_shifts]) if paradigm_shifts else 0.0
        breakthrough_alignment = self._calc_breakthrough_alignment(paradigm_shifts, domain_data)
        effectiveness = (avg_significance + breakthrough_alignment/100) / 2
        
        return {
            'paradigm_shifts': paradigm_count,
            'avg_significance': avg_significance,
            'breakthrough_alignment': breakthrough_alignment,
            'effectiveness': effectiveness
        }
    
    def _calc_breakthrough_alignment(self, paradigm_shifts: List, domain_data: DomainData) -> float:
        """Calculate breakthrough paper alignment percentage."""
        if not paradigm_shifts:
            return 0.0
        
        try:
            import core.shift_signal_detection as ssd
            breakthrough_papers = ssd.load_breakthrough_papers(domain_data, domain_data.domain_name)
            breakthrough_years = {p.pub_year for p in breakthrough_papers}
            
            aligned_count = sum(1 for signal in paradigm_shifts
                              if any(abs(signal.year - by) <= 2 for by in breakthrough_years))
            
            return (aligned_count / len(paradigm_shifts)) * 100
        except Exception:
            return 0.0
    
    def run_full_experiment(self) -> pd.DataFrame:
        """Run complete filtering experiment."""
        print("=" * 80)
        print("🧪 EXPERIMENT 3: PARADIGM FILTERING IMPACT ANALYSIS")
        print("=" * 80)
        
        all_results = []
        
        for domain in self.domains:
            print(f"\n📊 DOMAIN: {domain}")
            for condition in self.conditions:
                result = self.run_single_condition(domain, condition)
                all_results.append(result)
        
        return pd.DataFrame(all_results)
    
    def create_visualizations(self, results_df: pd.DataFrame) -> None:
        """Create key visualizations."""
        plt.style.use('default')
        sns.set_palette("husl")
        
        output_dir = Path("../results/experiment_3_visualizations")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Filtering Impact Comparison
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        sns.boxplot(data=results_df, x='Condition', y='Paradigm_Shifts', ax=axes[0,0])
        axes[0,0].set_title('Paradigm Shift Count by Filtering Condition', fontweight='bold')
        axes[0,0].tick_params(axis='x', rotation=45)
        
        sns.boxplot(data=results_df, x='Condition', y='Avg_Significance', ax=axes[0,1])
        axes[0,1].set_title('Average Paradigm Significance', fontweight='bold')
        axes[0,1].tick_params(axis='x', rotation=45)
        
        sns.boxplot(data=results_df, x='Condition', y='Breakthrough_Alignment', ax=axes[1,0])
        axes[1,0].set_title('Breakthrough Paper Alignment (%)', fontweight='bold')
        axes[1,0].tick_params(axis='x', rotation=45)
        
        sns.boxplot(data=results_df, x='Condition', y='Filtering_Effectiveness', ax=axes[1,1])
        axes[1,1].set_title('Overall Filtering Effectiveness', fontweight='bold')
        axes[1,1].tick_params(axis='x', rotation=45)
        
        plt.suptitle('Experiment 3: Paradigm Filtering Impact Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_dir / 'filtering_impact_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Effectiveness Heatmap
        pivot_effectiveness = results_df.pivot(index='Domain', columns='Condition', values='Filtering_Effectiveness')
        
        plt.figure(figsize=(12, 8))
        sns.heatmap(pivot_effectiveness, annot=True, fmt='.3f', cmap='RdYlGn')
        plt.title('Filtering Effectiveness by Domain and Condition', fontsize=16, fontweight='bold')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(output_dir / 'filtering_effectiveness_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Created visualizations in {output_dir}")
    
    def calculate_statistical_significance(self, results_df: pd.DataFrame) -> Dict:
        """Calculate statistical significance of filtering effects."""
        conditions = results_df['Condition'].unique()
        metrics = ['Paradigm_Shifts', 'Avg_Significance', 'Filtering_Effectiveness']
        significance_results = {}
        
        for metric in metrics:
            condition_groups = [results_df[results_df['Condition'] == cond][metric].values for cond in conditions]
            f_stat, p_value = stats.f_oneway(*condition_groups)
            
            significance_results[metric] = {
                'f_statistic': float(f_stat),
                'p_value': float(p_value),
                'significant': bool(p_value < 0.05)
            }
        
        return significance_results
    
    def save_results(self, results_df: pd.DataFrame, significance_results: Dict) -> Path:
        """Save experimental results."""
        output_dir = Path("../results/experiment_3")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        results_df.to_csv(output_dir / "filtering_experiment_results.csv", index=False)
        
        with open(output_dir / "statistical_significance.json", 'w') as f:
            json.dump(significance_results, f, indent=2)
        
        summary_stats = {
            'experimental_overview': {
                'total_experiments': len(results_df),
                'conditions_tested': len(results_df['Condition'].unique())
            },
            'performance_comparison': {
                condition: {
                    'avg_paradigm_shifts': float(results_df[results_df['Condition'] == condition]['Paradigm_Shifts'].mean()),
                    'avg_effectiveness': float(results_df[results_df['Condition'] == condition]['Filtering_Effectiveness'].mean())
                }
                for condition in results_df['Condition'].unique()
            }
        }
        
        with open(output_dir / "summary_statistics.json", 'w') as f:
            json.dump(summary_stats, f, indent=2)
        
        print(f"📊 RESULTS SAVED: {output_dir}")
        return output_dir

def create_comprehensive_visualizations():
    """Create all visualizations for the filtering experiment"""
    # Create output directory
    output_dir = Path("../results/experiment_3_visualizations")
    output_dir.mkdir(parents=True, exist_ok=True)

def main():
    """Run the comprehensive filtering impact experiment"""
    print("🔬 Paradigm Filtering Effectiveness Experiment")
    print("=" * 60)
    
    # Create output directory
    output_dir = Path("../results/experiment_3")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    experiment = ParadigmFilteringExperiment()
    results_df = experiment.run_full_experiment()
    significance_results = experiment.calculate_statistical_significance(results_df)
    experiment.create_visualizations(results_df)
    output_dir = experiment.save_results(results_df, significance_results)
    
    # Print key findings
    print("\n" + "=" * 80)
    print("🔬 EXPERIMENT 3 KEY RESEARCH FINDINGS")
    print("=" * 80)
    
    print(f"\n📊 FILTERING CONDITION COMPARISON:")
    for condition in results_df['Condition'].unique():
        condition_data = results_df[results_df['Condition'] == condition]
        print(f"   {condition:15s} - Avg Paradigm Shifts: {condition_data['Paradigm_Shifts'].mean():.1f}, "
              f"Effectiveness: {condition_data['Filtering_Effectiveness'].mean():.3f}")
    
    print(f"\n🎯 STATISTICAL SIGNIFICANCE:")
    for metric in ['Paradigm_Shifts', 'Filtering_Effectiveness']:
        result = significance_results[metric]
        significance = "SIGNIFICANT" if result['significant'] else "NOT SIGNIFICANT"
        print(f"   {metric:25s}: p={result['p_value']:.4f} ({significance})")
    
    print(f"\n🔬 RESEARCH IMPLICATIONS:")
    paradigm_variation = results_df.groupby('Condition')['Paradigm_Shifts'].std().mean()
    effectiveness_variation = results_df.groupby('Condition')['Filtering_Effectiveness'].std().mean()
    
    if paradigm_variation > 2 or effectiveness_variation > 0.1:
        print("   ✅ FILTERING SHOWS SENSITIVITY - Unlike penalty optimization!")
        print("   📈 Different filtering conditions produce measurably different results")
    else:
        print("   🚨 FILTERING ALSO SHOWS ROBUSTNESS - Similar to penalty optimization")
        print("   🔄 Algorithm may have universal robustness across all components")

if __name__ == "__main__":
    main() 