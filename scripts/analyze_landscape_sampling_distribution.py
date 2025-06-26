#!/usr/bin/env python3
"""
Landscape Sampling Distribution Analysis

Analyzes the statistical attributes and distributions of consensus and difference scores
from landscape sampling results, with comprehensive visualizations and statistical tests.

This script follows the development guidelines:
- Uses real data (no mock data)
- Implements fail-fast error handling
- Provides transparent analysis with detailed explanations
- Generates comprehensive statistical insights
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
from scipy.stats import pearsonr, spearmanr
import warnings
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
import sys
import os

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
warnings.filterwarnings('ignore', category=FutureWarning)

@dataclass
class ScoreStatistics:
    """Statistical summary for a score distribution."""
    mean: float
    std: float
    cv: float  # Coefficient of variation
    median: float
    q25: float
    q75: float
    iqr: float
    min_val: float
    max_val: float
    skewness: float
    kurtosis: float
    
    def __str__(self):
        return (f"Mean: {self.mean:.4f}, Std: {self.std:.4f}, CV: {self.cv:.4f}, "
                f"Median: {self.median:.4f}, IQR: {self.iqr:.4f}, "
                f"Skew: {self.skewness:.4f}, Kurt: {self.kurtosis:.4f}")

@dataclass
class DomainAnalysis:
    """Complete analysis results for a domain."""
    domain_name: str
    num_points: int
    success_rate: float
    consensus_stats: ScoreStatistics
    difference_stats: ScoreStatistics
    linear_aggregate_stats: ScoreStatistics
    harmonic_aggregate_stats: ScoreStatistics
    consensus_difference_correlation: float
    consensus_difference_correlation_pvalue: float
    spearman_correlation: float
    spearman_pvalue: float
    scale_ratio: float
    mean_imbalance_ratio: float


def load_landscape_data(file_path: str) -> Dict[str, Any]:
    """Load landscape sampling data with fail-fast error handling."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Landscape sampling file not found: {file_path}")
    
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in landscape sampling file: {e}")
    
    if not data:
        raise ValueError("Empty landscape sampling data")
    
    return data


def calculate_score_statistics(scores: List[float]) -> ScoreStatistics:
    """Calculate comprehensive statistics for a score distribution."""
    if not scores:
        raise ValueError("Cannot calculate statistics for empty score list")
    
    scores_array = np.array(scores)
    
    return ScoreStatistics(
        mean=float(np.mean(scores_array)),
        std=float(np.std(scores_array, ddof=1)),
        cv=float(np.std(scores_array, ddof=1) / np.mean(scores_array)) if np.mean(scores_array) != 0 else 0.0,
        median=float(np.median(scores_array)),
        q25=float(np.percentile(scores_array, 25)),
        q75=float(np.percentile(scores_array, 75)),
        iqr=float(np.percentile(scores_array, 75) - np.percentile(scores_array, 25)),
        min_val=float(np.min(scores_array)),
        max_val=float(np.max(scores_array)),
        skewness=float(stats.skew(scores_array)),
        kurtosis=float(stats.kurtosis(scores_array))
    )


def analyze_domain_scores(domain_name: str, domain_data: Dict[str, Any]) -> DomainAnalysis:
    """Analyze score distributions for a single domain."""
    points = domain_data.get('points', [])
    if not points:
        raise ValueError(f"No evaluation points found for domain {domain_name}")
    
    # Extract scores from successful evaluations
    successful_points = [p for p in points if p.get('success', False)]
    if not successful_points:
        raise ValueError(f"No successful evaluations found for domain {domain_name}")
    
    consensus_scores = [p['consensus_score'] for p in successful_points]
    difference_scores = [p['difference_score'] for p in successful_points]
    linear_scores = [p['linear_aggregate'] for p in successful_points]
    harmonic_scores = [p['harmonic_aggregate'] for p in successful_points]
    
    # Calculate statistics
    consensus_stats = calculate_score_statistics(consensus_scores)
    difference_stats = calculate_score_statistics(difference_scores)
    linear_stats = calculate_score_statistics(linear_scores)
    harmonic_stats = calculate_score_statistics(harmonic_scores)
    
    # Calculate correlations
    consensus_diff_corr, consensus_diff_pval = pearsonr(consensus_scores, difference_scores)
    spearman_corr, spearman_pval = spearmanr(consensus_scores, difference_scores)
    
    # Calculate scale ratio and imbalance
    scale_ratio = np.mean(difference_scores) / np.mean(consensus_scores) if np.mean(consensus_scores) != 0 else float('inf')
    mean_imbalance_ratio = np.mean(difference_scores) / (np.mean(consensus_scores) + np.mean(difference_scores))
    
    return DomainAnalysis(
        domain_name=domain_name,
        num_points=len(successful_points),
        success_rate=len(successful_points) / len(points),
        consensus_stats=consensus_stats,
        difference_stats=difference_stats, 
        linear_aggregate_stats=linear_stats,
        harmonic_aggregate_stats=harmonic_stats,
        consensus_difference_correlation=consensus_diff_corr,
        consensus_difference_correlation_pvalue=consensus_diff_pval,
        spearman_correlation=spearman_corr,
        spearman_pvalue=spearman_pval,
        scale_ratio=scale_ratio,
        mean_imbalance_ratio=mean_imbalance_ratio
    )


def create_distribution_plots(domain_analyses: List[DomainAnalysis], output_dir: str):
    """Create comprehensive distribution analysis plots."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up the plotting grid
    fig = plt.figure(figsize=(20, 16))
    
    # Load raw data for plotting
    file_path = "experiments/metric_evaluation/results/landscape_sampling_20250624_150534.json"
    raw_data = load_landscape_data(file_path)
    
    # Create subplots
    gs = fig.add_gridspec(4, 3, hspace=0.3, wspace=0.3)
    
    # Plot 1: Score distributions by domain
    ax1 = fig.add_subplot(gs[0, :])
    plot_score_distributions(raw_data, ax1)
    
    # Plot 2: Consensus vs Difference scatter plots
    ax2 = fig.add_subplot(gs[1, :])
    plot_consensus_vs_difference(raw_data, ax2)
    
    # Plot 3: Correlation analysis
    ax3 = fig.add_subplot(gs[2, 0])
    plot_correlation_analysis(domain_analyses, ax3)
    
    # Plot 4: Statistical summary
    ax4 = fig.add_subplot(gs[2, 1])
    plot_statistical_summary(domain_analyses, ax4)
    
    # Plot 5: Aggregation method comparison
    ax5 = fig.add_subplot(gs[2, 2])
    plot_aggregation_comparison(raw_data, ax5)
    
    # Plot 6: Score variance analysis
    ax6 = fig.add_subplot(gs[3, 0])
    plot_score_variance(domain_analyses, ax6)
    
    # Plot 7: Normality tests
    ax7 = fig.add_subplot(gs[3, 1])
    plot_normality_tests(raw_data, ax7)
    
    # Plot 8: Scale ratio analysis
    ax8 = fig.add_subplot(gs[3, 2])
    plot_scale_ratio_analysis(domain_analyses, ax8)
    
    plt.suptitle('Landscape Sampling Distribution Analysis', fontsize=20, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/landscape_distribution_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()


def plot_score_distributions(raw_data: Dict[str, Any], ax):
    """Plot score distributions for all domains."""
    domains = list(raw_data.keys())
    colors = sns.color_palette("husl", len(domains))
    
    consensus_data = []
    difference_data = []
    domain_labels = []
    
    for domain, color in zip(domains, colors):
        points = raw_data[domain]['points']
        successful_points = [p for p in points if p.get('success', False)]
        
        consensus_scores = [p['consensus_score'] for p in successful_points]
        difference_scores = [p['difference_score'] for p in successful_points]
        
        consensus_data.extend(consensus_scores)
        difference_data.extend(difference_scores)
        domain_labels.extend([f"{domain.replace('_', ' ').title()} (Consensus)"] * len(consensus_scores))
        domain_labels.extend([f"{domain.replace('_', ' ').title()} (Difference)"] * len(difference_scores))
    
    # Create violin plots
    all_scores = consensus_data + difference_data
    all_labels = domain_labels
    
    df = pd.DataFrame({'Score': all_scores, 'Type': all_labels})
    sns.violinplot(data=df, x='Type', y='Score', ax=ax)
    ax.set_title('Score Distributions by Domain and Type', fontweight='bold')
    ax.set_xlabel('Domain and Score Type')
    ax.set_ylabel('Score Value')
    ax.tick_params(axis='x', rotation=45)


def plot_consensus_vs_difference(raw_data: Dict[str, Any], ax):
    """Plot consensus vs difference scores with regression lines."""
    domains = list(raw_data.keys())
    colors = sns.color_palette("husl", len(domains))
    
    for domain, color in zip(domains, colors):
        points = raw_data[domain]['points']
        successful_points = [p for p in points if p.get('success', False)]
        
        consensus_scores = [p['consensus_score'] for p in successful_points]
        difference_scores = [p['difference_score'] for p in successful_points]
        
        # Scatter plot
        ax.scatter(consensus_scores, difference_scores, 
                  alpha=0.6, color=color, s=30,
                  label=f"{domain.replace('_', ' ').title()}")
        
        # Add regression line
        if len(consensus_scores) > 1:
            z = np.polyfit(consensus_scores, difference_scores, 1)
            p = np.poly1d(z)
            x_line = np.linspace(min(consensus_scores), max(consensus_scores), 100)
            ax.plot(x_line, p(x_line), color=color, linestyle='--', alpha=0.8)
    
    ax.set_xlabel('Consensus Score')
    ax.set_ylabel('Difference Score')
    ax.set_title('Consensus vs Difference Scores', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)


def plot_correlation_analysis(domain_analyses: List[DomainAnalysis], ax):
    """Plot correlation analysis between consensus and difference scores."""
    domains = [analysis.domain_name.replace('_', ' ').title() for analysis in domain_analyses]
    pearson_corrs = [analysis.consensus_difference_correlation for analysis in domain_analyses]
    spearman_corrs = [analysis.spearman_correlation for analysis in domain_analyses]
    
    x = np.arange(len(domains))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, pearson_corrs, width, label='Pearson', alpha=0.8)
    bars2 = ax.bar(x + width/2, spearman_corrs, width, label='Spearman', alpha=0.8)
    
    ax.set_xlabel('Domain')
    ax.set_ylabel('Correlation Coefficient')
    ax.set_title('Consensus-Difference Score Correlations', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(domains, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.3f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),  # 3 points vertical offset
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=8)


def plot_statistical_summary(domain_analyses: List[DomainAnalysis], ax):
    """Plot key statistical measures."""
    domains = [analysis.domain_name.replace('_', ' ').title() for analysis in domain_analyses]
    
    # Create heatmap data
    metrics = ['Consensus CV', 'Difference CV', 'Scale Ratio', 'Success Rate']
    data = []
    
    for analysis in domain_analyses:
        row = [
            analysis.consensus_stats.cv,
            analysis.difference_stats.cv, 
            min(analysis.scale_ratio, 20),  # Cap scale ratio for visualization
            analysis.success_rate
        ]
        data.append(row)
    
    df = pd.DataFrame(data, index=domains, columns=metrics)
    
    sns.heatmap(df, annot=True, fmt='.3f', cmap='viridis', ax=ax)
    ax.set_title('Statistical Summary Heatmap', fontweight='bold')
    ax.set_xlabel('Statistical Measure')
    ax.set_ylabel('Domain')


def plot_aggregation_comparison(raw_data: Dict[str, Any], ax):
    """Compare linear vs harmonic aggregation methods."""
    domains = list(raw_data.keys())
    
    linear_means = []
    harmonic_means = []
    domain_names = []
    
    for domain in domains:
        points = raw_data[domain]['points']
        successful_points = [p for p in points if p.get('success', False)]
        
        linear_scores = [p['linear_aggregate'] for p in successful_points]
        harmonic_scores = [p['harmonic_aggregate'] for p in successful_points]
        
        linear_means.append(np.mean(linear_scores))
        harmonic_means.append(np.mean(harmonic_scores))
        domain_names.append(domain.replace('_', ' ').title())
    
    x = np.arange(len(domain_names))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, linear_means, width, label='Linear', alpha=0.8)
    bars2 = ax.bar(x + width/2, harmonic_means, width, label='Harmonic', alpha=0.8)
    
    ax.set_xlabel('Domain')
    ax.set_ylabel('Mean Aggregate Score')
    ax.set_title('Linear vs Harmonic Aggregation', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(domain_names, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)


def plot_score_variance(domain_analyses: List[DomainAnalysis], ax):
    """Plot coefficient of variation for different score types."""
    domains = [analysis.domain_name.replace('_', ' ').title() for analysis in domain_analyses]
    
    consensus_cv = [analysis.consensus_stats.cv for analysis in domain_analyses]
    difference_cv = [analysis.difference_stats.cv for analysis in domain_analyses]
    linear_cv = [analysis.linear_aggregate_stats.cv for analysis in domain_analyses]
    harmonic_cv = [analysis.harmonic_aggregate_stats.cv for analysis in domain_analyses]
    
    x = np.arange(len(domains))
    width = 0.2
    
    ax.bar(x - 1.5*width, consensus_cv, width, label='Consensus', alpha=0.8)
    ax.bar(x - 0.5*width, difference_cv, width, label='Difference', alpha=0.8)
    ax.bar(x + 0.5*width, linear_cv, width, label='Linear Agg.', alpha=0.8)
    ax.bar(x + 1.5*width, harmonic_cv, width, label='Harmonic Agg.', alpha=0.8)
    
    ax.set_xlabel('Domain')
    ax.set_ylabel('Coefficient of Variation')
    ax.set_title('Score Variability Analysis', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(domains, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)


def plot_normality_tests(raw_data: Dict[str, Any], ax):
    """Test and visualize normality of score distributions."""
    domains = list(raw_data.keys())
    
    normality_results = []
    test_names = ['Consensus Shapiro', 'Difference Shapiro', 'Consensus KS', 'Difference KS']
    
    for domain in domains:
        points = raw_data[domain]['points']
        successful_points = [p for p in points if p.get('success', False)]
        
        consensus_scores = np.array([p['consensus_score'] for p in successful_points])
        difference_scores = np.array([p['difference_score'] for p in successful_points])
        
        # Shapiro-Wilk test (for smaller samples)
        if len(consensus_scores) <= 5000:
            shapiro_cons_stat, shapiro_cons_p = stats.shapiro(consensus_scores)
            shapiro_diff_stat, shapiro_diff_p = stats.shapiro(difference_scores)
        else:
            shapiro_cons_p = shapiro_diff_p = 0.0  # Fail for large samples
        
        # Kolmogorov-Smirnov test against normal distribution
        ks_cons_stat, ks_cons_p = stats.kstest(consensus_scores, 'norm', 
                                              args=(np.mean(consensus_scores), np.std(consensus_scores)))
        ks_diff_stat, ks_diff_p = stats.kstest(difference_scores, 'norm',
                                              args=(np.mean(difference_scores), np.std(difference_scores)))
        
        # Convert p-values to -log10 scale for visualization
        row = [
            -np.log10(max(shapiro_cons_p, 1e-10)),
            -np.log10(max(shapiro_diff_p, 1e-10)),
            -np.log10(max(ks_cons_p, 1e-10)),
            -np.log10(max(ks_diff_p, 1e-10))
        ]
        normality_results.append(row)
    
    domain_names = [domain.replace('_', ' ').title() for domain in domains]
    df = pd.DataFrame(normality_results, index=domain_names, columns=test_names)
    
    sns.heatmap(df, annot=True, fmt='.2f', cmap='RdYlBu_r', ax=ax)
    ax.set_title('Normality Tests (-log10 p-values)', fontweight='bold')
    ax.set_xlabel('Test Type')
    ax.set_ylabel('Domain')
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)  # p=0.05 threshold line


def plot_scale_ratio_analysis(domain_analyses: List[DomainAnalysis], ax):
    """Analyze the scale differences between consensus and difference scores."""
    domains = [analysis.domain_name.replace('_', ' ').title() for analysis in domain_analyses]
    scale_ratios = [analysis.scale_ratio for analysis in domain_analyses]
    imbalance_ratios = [analysis.mean_imbalance_ratio for analysis in domain_analyses]
    
    # Create scatter plot
    scatter = ax.scatter(scale_ratios, imbalance_ratios, s=100, alpha=0.7, c=range(len(domains)), cmap='viridis')
    
    # Add domain labels
    for i, domain in enumerate(domains):
        ax.annotate(domain, (scale_ratios[i], imbalance_ratios[i]), 
                   xytext=(5, 5), textcoords='offset points', fontsize=9)
    
    ax.set_xlabel('Scale Ratio (Difference/Consensus)')
    ax.set_ylabel('Mean Imbalance Ratio')
    ax.set_title('Scale and Imbalance Analysis', fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Add reference lines
    ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Balanced (0.5)')
    ax.axvline(x=1.0, color='red', linestyle='--', alpha=0.5, label='Equal Scale (1.0)')
    ax.legend()


def generate_summary_report(domain_analyses: List[DomainAnalysis], output_dir: str):
    """Generate a comprehensive text summary report."""
    report_path = f"{output_dir}/landscape_analysis_report.txt"
    
    with open(report_path, 'w') as f:
        f.write("LANDSCAPE SAMPLING DISTRIBUTION ANALYSIS REPORT\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("OVERALL SUMMARY\n")
        f.write("-" * 20 + "\n")
        total_points = sum(analysis.num_points for analysis in domain_analyses)
        avg_success_rate = np.mean([analysis.success_rate for analysis in domain_analyses])
        f.write(f"Total evaluation points analyzed: {total_points:,}\n")
        f.write(f"Average success rate: {avg_success_rate:.3f}\n")
        f.write(f"Domains analyzed: {len(domain_analyses)}\n\n")
        
        # Cross-domain correlation analysis
        all_pearson_corrs = [analysis.consensus_difference_correlation for analysis in domain_analyses]
        all_spearman_corrs = [analysis.spearman_correlation for analysis in domain_analyses]
        
        f.write("CROSS-DOMAIN CORRELATION PATTERNS\n")
        f.write("-" * 35 + "\n")
        f.write(f"Mean Pearson correlation (consensus vs difference): {np.mean(all_pearson_corrs):.4f} ± {np.std(all_pearson_corrs):.4f}\n")
        f.write(f"Mean Spearman correlation (consensus vs difference): {np.mean(all_spearman_corrs):.4f} ± {np.std(all_spearman_corrs):.4f}\n")
        
        # Statistical significance summary
        significant_pearson = sum(1 for analysis in domain_analyses if analysis.consensus_difference_correlation_pvalue < 0.05)
        significant_spearman = sum(1 for analysis in domain_analyses if analysis.spearman_pvalue < 0.05)
        f.write(f"Statistically significant Pearson correlations: {significant_pearson}/{len(domain_analyses)}\n")
        f.write(f"Statistically significant Spearman correlations: {significant_spearman}/{len(domain_analyses)}\n\n")
        
        # Domain-specific analysis
        for analysis in domain_analyses:
            f.write(f"DOMAIN: {analysis.domain_name.upper().replace('_', ' ')}\n")
            f.write("-" * (len(analysis.domain_name) + 8) + "\n")
            f.write(f"Evaluation points: {analysis.num_points:,}\n")
            f.write(f"Success rate: {analysis.success_rate:.3f}\n\n")
            
            f.write("Score Statistics:\n")
            f.write(f"  Consensus: {analysis.consensus_stats}\n")
            f.write(f"  Difference: {analysis.difference_stats}\n")
            f.write(f"  Linear Aggregate: {analysis.linear_aggregate_stats}\n")
            f.write(f"  Harmonic Aggregate: {analysis.harmonic_aggregate_stats}\n\n")
            
            f.write("Relationship Analysis:\n")
            f.write(f"  Pearson correlation: {analysis.consensus_difference_correlation:.4f} (p={analysis.consensus_difference_correlation_pvalue:.6f})\n")
            f.write(f"  Spearman correlation: {analysis.spearman_correlation:.4f} (p={analysis.spearman_pvalue:.6f})\n")
            f.write(f"  Scale ratio (diff/cons): {analysis.scale_ratio:.2f}\n")
            f.write(f"  Mean imbalance ratio: {analysis.mean_imbalance_ratio:.4f}\n\n")
            
            # Interpretation
            f.write("Interpretation:\n")
            if abs(analysis.consensus_difference_correlation) > 0.7:
                strength = "strong"
            elif abs(analysis.consensus_difference_correlation) > 0.3:
                strength = "moderate"
            else:
                strength = "weak"
                
            direction = "positive" if analysis.consensus_difference_correlation > 0 else "negative"
            f.write(f"  - {strength.title()} {direction} correlation between consensus and difference scores\n")
            
            if analysis.consensus_stats.cv > 0.2:
                f.write(f"  - High consensus score variability (CV={analysis.consensus_stats.cv:.3f})\n")
            if analysis.difference_stats.cv > 0.2:
                f.write(f"  - High difference score variability (CV={analysis.difference_stats.cv:.3f})\n")
                
            if analysis.scale_ratio > 5:
                f.write(f"  - Large scale imbalance: difference scores {analysis.scale_ratio:.1f}x larger than consensus scores\n")
            
            f.write("\n" + "="*60 + "\n\n")


def main():
    """Main analysis function."""
    print("🔍 LANDSCAPE SAMPLING DISTRIBUTION ANALYSIS")
    print("=" * 50)
    
    # Define file path
    file_path = "experiments/metric_evaluation/results/landscape_sampling_20250624_150534.json"
    output_dir = "results/landscape_analysis"
    
    try:
        # Load data
        print(f"📊 Loading landscape sampling data from {file_path}...")
        raw_data = load_landscape_data(file_path)
        print(f"✅ Loaded data for {len(raw_data)} domains")
        
        # Analyze each domain
        domain_analyses = []
        for domain_name, domain_data in raw_data.items():
            print(f"🔬 Analyzing {domain_name}...")
            analysis = analyze_domain_scores(domain_name, domain_data)
            domain_analyses.append(analysis)
            print(f"   📈 {analysis.num_points:,} points, {analysis.success_rate:.1%} success rate")
            print(f"   🔗 Consensus-Difference correlation: {analysis.consensus_difference_correlation:.3f}")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate visualizations
        print(f"📊 Creating comprehensive distribution plots...")
        create_distribution_plots(domain_analyses, output_dir)
        print(f"✅ Plots saved to {output_dir}/landscape_distribution_analysis.png")
        
        # Generate summary report
        print(f"📝 Generating summary report...")
        generate_summary_report(domain_analyses, output_dir)
        print(f"✅ Report saved to {output_dir}/landscape_analysis_report.txt")
        
        # Print key insights
        print("\n🎯 KEY INSIGHTS:")
        print("-" * 20)
        
        total_points = sum(analysis.num_points for analysis in domain_analyses)
        avg_correlation = np.mean([analysis.consensus_difference_correlation for analysis in domain_analyses])
        high_var_domains = [analysis.domain_name for analysis in domain_analyses 
                           if analysis.consensus_stats.cv > 0.2 or analysis.difference_stats.cv > 0.2]
        
        print(f"📊 Analyzed {total_points:,} evaluation points across {len(domain_analyses)} domains")
        print(f"🔗 Average consensus-difference correlation: {avg_correlation:.3f}")
        print(f"⚡ High-variability domains: {', '.join(high_var_domains) if high_var_domains else 'None'}")
        
        # Scale ratio insights
        scale_ratios = [analysis.scale_ratio for analysis in domain_analyses]
        avg_scale_ratio = np.mean(scale_ratios)
        print(f"⚖️  Average scale ratio (difference/consensus): {avg_scale_ratio:.1f}x")
        
        print(f"\n✅ Analysis complete! Results saved in {output_dir}/")
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        raise


if __name__ == "__main__":
    main() 