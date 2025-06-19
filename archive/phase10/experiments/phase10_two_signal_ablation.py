#!/usr/bin/env python3
"""
Phase 10 Experiment 1 (Two-Signal Ablation)

Runs the original multi-source ablation framework but restricts the signal
conditions to the current two-signal architecture:
    1. Citation-Only
    2. Direction-Only
    3. Citation + Direction (Fusion)

Domains: limited to the five high-priority technical domains processed in
Phase 10 comprehensive testing.
The script automatically writes results/visualisations via the underlying
MultiSourceAblationExperiment utilities but saves to dedicated Phase 10 locations.
"""
# ... existing code ...
import sys
from pathlib import Path
import json

# Add project root to path for core module imports
sys.path.append(str(Path(__file__).parent.parent.parent))

# Add phase9 experiments to path for base class import
sys.path.append(str(Path(__file__).parent.parent / "phase9" / "experiments"))

from experiments.phase9.experiments.experiment_1_multi_source_ablation import MultiSourceAblationExperiment, ExperimentCondition


class Phase10TwoSignalAblation(MultiSourceAblationExperiment):
    """Phase 10 specific ablation experiment that saves to dedicated directories."""
    
    def __init__(self):
        super().__init__()
        # Update output directories for Phase 10
        self.output_dir = Path("../results/phase10_two_signal_ablation")
        self.visualizations_dir = Path("../results/phase10_two_signal_visualizations")
        
        # Create directories if they don't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.visualizations_dir.mkdir(parents=True, exist_ok=True)
        
        # Restrict to two-signal conditions only
        self.conditions = [
            ExperimentCondition("Citation_Only", True, False, False, 
                              "Citation disruption signals only"),
            ExperimentCondition("Direction_Only", False, False, True,
                              "Direction volatility signals only"), 
            ExperimentCondition("Citation_Direction", True, False, True,
                              "Citation + direction signals")
        ]
        
        # Phase-10 focus: five technical domains
        self.domains = [
            'natural_language_processing',
            'deep_learning',
            'computer_vision',
            'machine_learning',
            'machine_translation'
        ]
    
    def create_visualizations(self, results_df, evaluation_df):
        """Override to save visualizations to Phase 10 directory."""
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # Set up output directory
        viz_dir = self.visualizations_dir
        viz_dir.mkdir(parents=True, exist_ok=True)
        
        # Call parent method but capture plots and save to our directory
        original_savefig = plt.savefig
        
        def custom_savefig(*args, **kwargs):
            if '../results/experiment_1_visualizations' in str(args[0]):
                # Replace with phase10 visualization path
                new_path = str(args[0]).replace('../results/experiment_1_visualizations',
                                              '../results/phase10_two_signal_visualizations')
                args = (new_path,) + args[1:]
            original_savefig(*args, **kwargs)
        
        plt.savefig = custom_savefig
        
        try:
            super().create_visualizations(results_df, evaluation_df)
        finally:
            plt.savefig = original_savefig
    
    def save_results(self, results_df, evaluation_df, complementarity_analysis, filtering_analysis):
        """Override to save results to Phase 10 directory."""
        
        # Save raw results
        results_df.to_csv(self.output_dir / "raw_results.csv", index=False)
        if not evaluation_df.empty:
            evaluation_df.to_csv(self.output_dir / "evaluation_metrics.csv", index=False)
        
        # Save all analyses
        with open(self.output_dir / "complementarity_analysis.json", 'w') as f:
            json.dump(complementarity_analysis, f, indent=2)
            
        with open(self.output_dir / "filtering_analysis.json", 'w') as f:
            json.dump(filtering_analysis, f, indent=2)
        
        # Create comprehensive summary statistics
        summary_stats = {
            'experiment_overview': {
                'total_experiments': len(results_df),
                'domains_tested': len(self.domains),
                'conditions_tested': len(self.conditions),
                'successful_experiments': len(results_df[results_df['Paradigm_Signals'] >= 0])
            },
            'signal_contribution': {
                'avg_signals_by_condition': results_df.groupby('Condition')['Paradigm_Signals'].mean().to_dict(),
                'std_signals_by_condition': results_df.groupby('Condition')['Paradigm_Signals'].std().to_dict(),
                'avg_segments_by_condition': results_df.groupby('Condition')['Segment_Count'].mean().to_dict(),
                'signal_productivity_ranking': results_df.groupby('Condition')['Paradigm_Signals'].mean().sort_values(ascending=False).to_dict()
            },
            'filtering_mechanisms': {
                'subadditive_domains': filtering_analysis['subadditive_domains'],
                'subadditive_ratio': len(filtering_analysis['subadditive_domains']) / len(self.domains),
                'key_insights': filtering_analysis['filtering_insights']
            },
            'domain_analysis': {
                'signal_variance_by_domain': results_df.groupby('Domain')['Paradigm_Signals'].std().to_dict(),
                'avg_signals_by_domain': results_df.groupby('Domain')['Paradigm_Signals'].mean().to_dict()
            }
        }
        
        # Add evaluation metrics only if we have evaluation data
        if not evaluation_df.empty:
            summary_stats['performance_metrics'] = {
                'avg_f1_by_condition': evaluation_df.groupby('Condition')['F1_Score'].mean().to_dict(),
                'avg_precision_by_condition': evaluation_df.groupby('Condition')['Precision'].mean().to_dict(),
                'avg_recall_by_condition': evaluation_df.groupby('Condition')['Recall'].mean().to_dict(),
                'domain_performance': {
                    'avg_f1_by_domain': evaluation_df.groupby('Domain')['F1_Score'].mean().to_dict(),
                    'best_condition_by_domain': evaluation_df.loc[evaluation_df.groupby('Domain')['F1_Score'].idxmax()][['Domain', 'Condition', 'F1_Score']].to_dict('records')
                }
            }
        
        with open(self.output_dir / "summary_statistics.json", 'w') as f:
            json.dump(summary_stats, f, indent=2)
        
        print(f"\n📊 PHASE 10 RESULTS SAVED: {self.output_dir}")
        print(f"   • Raw results: raw_results.csv")
        print(f"   • Filtering analysis: filtering_analysis.json") 
        print(f"   • Complementarity analysis: complementarity_analysis.json")
        print(f"   • Summary statistics: summary_statistics.json")
        print(f"   • Visualizations: {self.visualizations_dir}")
        if not evaluation_df.empty:
            print(f"   • Evaluation metrics: evaluation_metrics.csv")
        
        return self.output_dir


def main():
    """Execute Phase 10 two-signal ablation study."""
    print("=== Phase 10 Two-Signal Ablation Study ===")
    print("Testing Citation + Direction architecture across 5 domains\n")
    
    experiment = Phase10TwoSignalAblation()
    
    # Run full experimental matrix (5 domains × 3 conditions = 15 experiments)
    results_df = experiment.run_full_experiment()
    
    # Calculate evaluation metrics
    evaluation_df = experiment.calculate_evaluation_metrics(results_df)
    
    # Analyze filtering mechanisms
    filtering_analysis = experiment.analyze_filtering_mechanisms(results_df)
    
    # Analyze signal complementarity
    complementarity_analysis = experiment.analyze_signal_complementarity(results_df)
    
    # Create comprehensive visualizations
    experiment.create_visualizations(results_df, evaluation_df)
    
    # Save all results
    output_dir = experiment.save_results(results_df, evaluation_df, complementarity_analysis, filtering_analysis)
    
    print("\n✅ Phase 10 two-signal ablation study completed successfully!")
    print(f"📊 Results saved to: {output_dir}")
    print(f"🎨 Visualizations saved to: {experiment.visualizations_dir}")


if __name__ == "__main__":
    main() 