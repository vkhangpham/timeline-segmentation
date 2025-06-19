#!/usr/bin/env python3
"""
Phase 10 Enhanced Visualization Experiment

Creates dedicated timeline visualizations for the two-signal architecture:
1. Citation disruption timeline with change points
2. Direction volatility timeline with change points  
3. Combined timeline showing both signals

Tests all domains with detailed signal-specific timelines.
"""

import sys
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from typing import Dict, List, Any, Tuple
from collections import defaultdict

# Add project root to path for core module imports
project_root = Path('.').resolve().parent.parent.parent
sys.path.append(str(project_root))

from core.data_processing import process_domain_data
from core.shift_signal_detection import detect_shift_signals, detect_citation_structural_breaks, detect_research_direction_changes
from core.data_models import DomainData, ShiftSignal


class Phase10EnhancedVisualizationExperiment:
    """Enhanced Phase 10 experiment with detailed signal timeline visualizations."""
    
    def __init__(self):
        self.output_dir = Path("../results/phase10_enhanced_visualizations")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Test all domains for comprehensive analysis
        self.domains = [
            'natural_language_processing',
            'deep_learning', 
            'computer_vision',
            'machine_learning',
            'machine_translation',
            'applied_mathematics',
            'computer_science',
            'art'
        ]
        
        # Configure matplotlib for high-quality plots
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        self.setup_plot_style()
    
    def setup_plot_style(self):
        """Configure matplotlib style for publication-quality plots."""
        plt.rcParams.update({
            'figure.figsize': (14, 8),
            'font.size': 11,
            'axes.titlesize': 14,
            'axes.labelsize': 12,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10,
            'figure.dpi': 300
        })
    
    def extract_citation_timeline_data(self, domain_data: DomainData) -> Tuple[Dict, List[ShiftSignal]]:
        """Extract citation data and change points for timeline visualization."""
        # Create citation time series (same logic as in detection function)
        citation_series = defaultdict(float)
        influence_series = defaultdict(float)
        
        for paper in domain_data.papers:
            year = paper.pub_year
            citation_series[year] += paper.cited_by_count
            # Influence score: citation count weighted by recency
            recency_weight = 1.0 / (2025 - year + 1)
            influence_series[year] += paper.cited_by_count * recency_weight
        
        # Detect citation disruption signals
        citation_signals = detect_citation_structural_breaks(domain_data, "timeline_viz")
        
        return {
            'years': sorted(citation_series.keys()),
            'citations': [citation_series[year] for year in sorted(citation_series.keys())],
            'influence': [influence_series[year] for year in sorted(citation_series.keys())]
        }, citation_signals
    
    def extract_direction_timeline_data(self, domain_data: DomainData) -> Tuple[Dict, List[ShiftSignal]]:
        """Extract direction volatility data and change points for timeline visualization."""
        # Group papers by year and analyze keyword evolution
        year_keywords = defaultdict(list)
        year_papers = defaultdict(int)
        
        for paper in domain_data.papers:
            year_keywords[paper.pub_year].extend(paper.keywords)
            year_papers[paper.pub_year] += 1
        
        years = sorted(year_keywords.keys())
        
        # Calculate keyword diversity and novelty scores for each year
        novelty_scores = []
        overlap_scores = []
        direction_change_scores = []
        
        window_size = 3
        for i, year in enumerate(years):
            if i < window_size:
                novelty_scores.append(0.0)
                overlap_scores.append(1.0)
                direction_change_scores.append(0.0)
                continue
            
            # Current window keywords
            current_keywords = []
            for y in years[i-window_size:i]:
                current_keywords.extend(year_keywords[y])
            
            # Previous window keywords
            prev_keywords = []
            for y in years[max(0, i-window_size*2):i-window_size]:
                prev_keywords.extend(year_keywords[y])
            
            if not current_keywords or not prev_keywords:
                novelty_scores.append(0.0)
                overlap_scores.append(1.0)
                direction_change_scores.append(0.0)
                continue
            
            # Calculate metrics
            current_set = set(current_keywords)
            prev_set = set(prev_keywords)
            
            overlap = len(current_set & prev_set) / len(prev_set) if prev_set else 0
            novelty = len(current_set - prev_set) / len(current_set) if current_set else 0
            direction_change = novelty * (1 - overlap)
            
            novelty_scores.append(novelty)
            overlap_scores.append(overlap)
            direction_change_scores.append(direction_change)
        
        # Detect direction volatility signals
        direction_signals = detect_research_direction_changes(domain_data)
        
        return {
            'years': years,
            'paper_counts': [year_papers[year] for year in years],
            'novelty_scores': novelty_scores,
            'overlap_scores': overlap_scores,
            'direction_change_scores': direction_change_scores
        }, direction_signals
    
    def create_citation_timeline_plot(self, domain_name: str, timeline_data: Dict, citation_signals: List[ShiftSignal]):
        """Create detailed citation timeline visualization with change points."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), sharex=True)
        
        years = timeline_data['years']
        citations = timeline_data['citations']
        influence = timeline_data['influence']
        
        # Plot 1: Raw citation counts
        ax1.plot(years, citations, 'b-', linewidth=2, alpha=0.7, label='Citation Count')
        ax1.fill_between(years, citations, alpha=0.3, color='blue')
        
        # Add change points
        for signal in citation_signals:
            ax1.axvline(x=signal.year, color='red', linestyle='--', alpha=0.8, linewidth=2)
            ax1.annotate(f'{signal.year}\n(conf: {signal.confidence:.2f})', 
                        xy=(signal.year, max(citations)*0.8), 
                        xytext=(signal.year, max(citations)*0.9),
                        ha='center', va='bottom', fontsize=9,
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7),
                        arrowprops=dict(arrowstyle='->', color='red'))
        
        ax1.set_title(f'{domain_name.replace("_", " ").title()} - Citation Timeline with Change Points', 
                     fontsize=14, fontweight='bold')
        ax1.set_ylabel('Total Citations', fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Plot 2: Influence-weighted citations
        ax2.plot(years, influence, 'g-', linewidth=2, alpha=0.7, label='Influence Score')
        ax2.fill_between(years, influence, alpha=0.3, color='green')
        
        # Add change points to influence plot
        for signal in citation_signals:
            ax2.axvline(x=signal.year, color='red', linestyle='--', alpha=0.8, linewidth=2)
        
        ax2.set_xlabel('Year', fontsize=12)
        ax2.set_ylabel('Influence Score', fontsize=12)
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # Add summary text
        fig.text(0.02, 0.02, f'Change Points Detected: {len(citation_signals)}', 
                fontsize=10, ha='left', va='bottom',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f'{domain_name}_citation_timeline.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_direction_timeline_plot(self, domain_name: str, timeline_data: Dict, direction_signals: List[ShiftSignal]):
        """Create detailed direction volatility timeline visualization with change points."""
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(16, 12), sharex=True)
        
        years = timeline_data['years']
        paper_counts = timeline_data['paper_counts']
        novelty_scores = timeline_data['novelty_scores']
        overlap_scores = timeline_data['overlap_scores']
        direction_change_scores = timeline_data['direction_change_scores']
        
        # Plot 1: Paper count over time
        ax1.bar(years, paper_counts, alpha=0.6, color='skyblue', label='Papers per Year')
        ax1.set_title(f'{domain_name.replace("_", " ").title()} - Research Direction Evolution', 
                     fontsize=14, fontweight='bold')
        ax1.set_ylabel('Papers per Year', fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Plot 2: Keyword novelty and overlap
        ax2.plot(years, novelty_scores, 'r-', linewidth=2, alpha=0.8, label='Keyword Novelty')
        ax2.plot(years, overlap_scores, 'b-', linewidth=2, alpha=0.8, label='Keyword Overlap')
        ax2.fill_between(years, novelty_scores, alpha=0.3, color='red')
        ax2.fill_between(years, overlap_scores, alpha=0.3, color='blue')
        ax2.set_ylabel('Score (0-1)', fontsize=12)
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # Plot 3: Direction change scores with change points
        ax3.plot(years, direction_change_scores, 'purple', linewidth=2, alpha=0.8, 
                label='Direction Change Score')
        ax3.fill_between(years, direction_change_scores, alpha=0.3, color='purple')
        
        # Add threshold line
        ax3.axhline(y=0.4, color='orange', linestyle=':', alpha=0.8, 
                   label='Detection Threshold (0.4)')
        
        # Add change points
        for signal in direction_signals:
            ax3.axvline(x=signal.year, color='red', linestyle='--', alpha=0.8, linewidth=2)
            ax3.annotate(f'{signal.year}\n(conf: {signal.confidence:.2f})', 
                        xy=(signal.year, max(direction_change_scores)*0.8), 
                        xytext=(signal.year, max(direction_change_scores)*0.9),
                        ha='center', va='bottom', fontsize=9,
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7),
                        arrowprops=dict(arrowstyle='->', color='red'))
        
        ax3.set_xlabel('Year', fontsize=12)
        ax3.set_ylabel('Direction Change Score', fontsize=12)
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        
        # Add summary text
        fig.text(0.02, 0.02, f'Change Points Detected: {len(direction_signals)}', 
                fontsize=10, ha='left', va='bottom',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7))
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f'{domain_name}_direction_timeline.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_combined_timeline_plot(self, domain_name: str, citation_data: Dict, direction_data: Dict, 
                                    citation_signals: List[ShiftSignal], direction_signals: List[ShiftSignal]):
        """Create combined timeline showing both signal types."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 12))
        
        # Citation timeline (top left)
        years_cit = citation_data['years']
        citations = citation_data['citations']
        
        ax1.plot(years_cit, citations, 'b-', linewidth=2, alpha=0.7)
        ax1.fill_between(years_cit, citations, alpha=0.3, color='blue')
        
        for signal in citation_signals:
            ax1.axvline(x=signal.year, color='red', linestyle='--', alpha=0.8, linewidth=2)
        
        ax1.set_title('Citation Disruption Signals', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Citations', fontsize=11)
        ax1.grid(True, alpha=0.3)
        
        # Direction timeline (top right)
        years_dir = direction_data['years']
        direction_scores = direction_data['direction_change_scores']
        
        ax2.plot(years_dir, direction_scores, 'purple', linewidth=2, alpha=0.7)
        ax2.fill_between(years_dir, direction_scores, alpha=0.3, color='purple')
        ax2.axhline(y=0.4, color='orange', linestyle=':', alpha=0.8)
        
        for signal in direction_signals:
            ax2.axvline(x=signal.year, color='red', linestyle='--', alpha=0.8, linewidth=2)
        
        ax2.set_title('Direction Volatility Signals', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Direction Change Score', fontsize=11)
        ax2.grid(True, alpha=0.3)
        
        # Combined signal timeline (bottom spanning both columns)
        ax3.remove()
        ax4.remove()
        ax_combined = fig.add_subplot(2, 1, 2)
        
        # Normalize both signals for comparison
        if citations:
            norm_citations = np.array(citations) / max(citations)
            ax_combined.plot(years_cit, norm_citations, 'b-', linewidth=2, alpha=0.7, 
                           label='Citation (Normalized)')
        
        if direction_scores:
            norm_direction = np.array(direction_scores) / max(direction_scores) if max(direction_scores) > 0 else direction_scores
            ax_combined.plot(years_dir, norm_direction, 'purple', linewidth=2, alpha=0.7, 
                           label='Direction Change (Normalized)')
        
        # Add all change points
        all_signal_years = set()
        for signal in citation_signals:
            ax_combined.axvline(x=signal.year, color='red', linestyle='--', alpha=0.6, linewidth=1.5)
            all_signal_years.add(signal.year)
        
        for signal in direction_signals:
            ax_combined.axvline(x=signal.year, color='darkred', linestyle='-', alpha=0.6, linewidth=1.5)
            all_signal_years.add(signal.year)
        
        ax_combined.set_title(f'{domain_name.replace("_", " ").title()} - Combined Signal Timeline', 
                            fontsize=14, fontweight='bold')
        ax_combined.set_xlabel('Year', fontsize=12)
        ax_combined.set_ylabel('Normalized Signal Strength', fontsize=12)
        ax_combined.grid(True, alpha=0.3)
        ax_combined.legend()
        
        # Add summary statistics
        summary_text = f"""
        Citation Signals: {len(citation_signals)}
        Direction Signals: {len(direction_signals)}
        Total Unique Years: {len(all_signal_years)}
        Signal Density: {len(all_signal_years)}/{len(set(years_cit + years_dir))} years
        """
        
        fig.text(0.02, 0.02, summary_text.strip(), fontsize=10, ha='left', va='bottom',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f'{domain_name}_combined_timeline.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def run_enhanced_visualization_experiment(self):
        """Run the enhanced visualization experiment across all domains."""
        results = []
        
        print("🎨 Starting Phase 10 Enhanced Visualization Experiment")
        print("=" * 60)
        
        for domain_name in self.domains:
            print(f"\n📊 Processing {domain_name}...")
            
            try:
                # Load domain data
                processing_result = process_domain_data(domain_name, data_directory="../../../resources")
                if not processing_result.success:
                    raise Exception(f"Failed to process domain: {processing_result.error_message}")
                
                domain_data = processing_result.domain_data
                
                # Extract timeline data for both signal types
                citation_data, citation_signals = self.extract_citation_timeline_data(domain_data)
                direction_data, direction_signals = self.extract_direction_timeline_data(domain_data)
                
                # Create visualizations
                self.create_citation_timeline_plot(domain_name, citation_data, citation_signals)
                self.create_direction_timeline_plot(domain_name, direction_data, direction_signals)
                self.create_combined_timeline_plot(domain_name, citation_data, direction_data, 
                                                 citation_signals, direction_signals)
                
                # Collect results
                results.append({
                    'domain': domain_name,
                    'citation_signals': len(citation_signals),
                    'direction_signals': len(direction_signals),
                    'total_signals': len(citation_signals) + len(direction_signals),
                    'timespan_years': max(citation_data['years']) - min(citation_data['years']) if citation_data['years'] else 0,
                    'total_papers': len(domain_data.papers),
                    'total_citations': sum(citation_data['citations']),
                    'citation_signal_years': [s.year for s in citation_signals],
                    'direction_signal_years': [s.year for s in direction_signals]
                })
                
                print(f"   ✅ Citation signals: {len(citation_signals)}")
                print(f"   ✅ Direction signals: {len(direction_signals)}")
                print(f"   ✅ Visualizations saved")
                
            except Exception as e:
                print(f"   ❌ Error processing {domain_name}: {e}")
                results.append({
                    'domain': domain_name,
                    'error': str(e)
                })
        
        # Save comprehensive results
        self.save_experiment_results(results)
        
        return results
    
    def save_experiment_results(self, results: List[Dict]):
        """Save comprehensive experiment results."""
        # Save detailed results as JSON
        with open(self.output_dir / "experiment_results.json", 'w') as f:
            json.dump(results, f, indent=2)
        
        # Create summary DataFrame and save as CSV
        summary_data = []
        for result in results:
            if 'error' not in result:
                summary_data.append({
                    'Domain': result['domain'],
                    'Citation_Signals': result['citation_signals'],
                    'Direction_Signals': result['direction_signals'],
                    'Total_Signals': result['total_signals'],
                    'Timespan_Years': result['timespan_years'],
                    'Total_Papers': result['total_papers'],
                    'Total_Citations': result['total_citations'],
                    'Signal_Density': result['total_signals'] / result['timespan_years'] if result['timespan_years'] > 0 else 0
                })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(self.output_dir / "experiment_summary.csv", index=False)
        
        # Create domain comparison visualization only if we have data
        if len(summary_df) > 0:
            self.create_domain_comparison_plot(summary_df)
        else:
            print("⚠️  No successful results to create domain comparison plot")
        
        print(f"\n📊 Results saved to: {self.output_dir}")
        print(f"   • Experiment results: experiment_results.json")
        print(f"   • Summary table: experiment_summary.csv")
        print(f"   • Individual timeline plots: {len([f for f in self.output_dir.glob('*_timeline.png')])} files")
    
    def create_domain_comparison_plot(self, summary_df: pd.DataFrame):
        """Create comparison plot across all domains."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        domains = summary_df['Domain'].str.replace('_', ' ').str.title()
        
        # Plot 1: Signal counts by domain
        x_pos = np.arange(len(domains))
        width = 0.35
        
        ax1.bar(x_pos - width/2, summary_df['Citation_Signals'], width, 
               label='Citation Signals', color='blue', alpha=0.7)
        ax1.bar(x_pos + width/2, summary_df['Direction_Signals'], width,
               label='Direction Signals', color='purple', alpha=0.7)
        
        ax1.set_title('Signal Counts by Domain', fontweight='bold')
        ax1.set_ylabel('Number of Signals')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(domains, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Signal density
        ax2.bar(domains, summary_df['Signal_Density'], color='green', alpha=0.7)
        ax2.set_title('Signal Density (Signals per Year)', fontweight='bold')
        ax2.set_ylabel('Signals per Year')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Total papers vs signals
        ax3.scatter(summary_df['Total_Papers'], summary_df['Total_Signals'], 
                   s=100, alpha=0.7, color='red')
        for i, domain in enumerate(domains):
            ax3.annotate(domain, (summary_df.iloc[i]['Total_Papers'], summary_df.iloc[i]['Total_Signals']),
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        ax3.set_title('Papers vs Signals Relationship', fontweight='bold')
        ax3.set_xlabel('Total Papers')
        ax3.set_ylabel('Total Signals')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Timespan coverage
        ax4.bar(domains, summary_df['Timespan_Years'], color='orange', alpha=0.7)
        ax4.set_title('Temporal Coverage by Domain', fontweight='bold')
        ax4.set_ylabel('Years Covered')
        ax4.tick_params(axis='x', rotation=45)
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'domain_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()


def main():
    """Execute the enhanced visualization experiment."""
    print("🎨 Phase 10 Enhanced Timeline Visualization Experiment")
    print("Creating dedicated timelines for citation and direction signals")
    print("=" * 60)
    
    experiment = Phase10EnhancedVisualizationExperiment()
    results = experiment.run_enhanced_visualization_experiment()
    
    # Print summary statistics
    successful_domains = [r for r in results if 'error' not in r]
    total_citation_signals = sum(r['citation_signals'] for r in successful_domains)
    total_direction_signals = sum(r['direction_signals'] for r in successful_domains)
    
    print("\n" + "=" * 60)
    print("🎯 EXPERIMENT SUMMARY")
    print("=" * 60)
    print(f"Domains processed: {len(successful_domains)}/{len(results)}")
    print(f"Total citation signals: {total_citation_signals}")
    print(f"Total direction signals: {total_direction_signals}")
    if len(successful_domains) > 0:
        print(f"Average signals per domain: {(total_citation_signals + total_direction_signals)/len(successful_domains):.1f}")
    else:
        print("Average signals per domain: N/A (no successful domains)")
    print(f"Visualizations created: {len(list(experiment.output_dir.glob('*.png')))}")
    print(f"\n📊 Results saved to: {experiment.output_dir}")
    
    print("\n✅ Enhanced visualization experiment completed successfully!")


if __name__ == "__main__":
    main() 