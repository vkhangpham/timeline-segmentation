"""
Timeline Analysis Visualization Module

This module provides comprehensive visualization capabilities for timeline analysis results,
including shift signals, period signals, and integrated timeline views.

Features:
- Interactive shift signal visualization with confidence levels
- Period characterization visualization with network metrics
- Combined timeline view showing paradigm transitions and periods
- Statistical analysis dashboards
- Ground truth comparison and evaluation metrics
- Export capabilities for presentations

Follows functional programming principles with pure visualization functions.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle
import seaborn as sns
from datetime import datetime

# Set style for professional visualizations
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


def visualize_domain_timeline(domain_name: str, 
                            signals_dir: str = "results/signals",
                            output_dir: str = "results/visualizations") -> str:
    """
    Create comprehensive timeline visualization for a domain.
    
    Args:
        domain_name: Name of the domain to visualize
        signals_dir: Directory containing signal files
        output_dir: Directory to save visualizations
        
    Returns:
        Path to the main visualization file
    """
    print(f"\nCREATING TIMELINE VISUALIZATION: {domain_name}")
    
    # Ensure output directory exists
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Load signal data
    shift_signals_data = load_shift_signals(domain_name, signals_dir)
    period_signals_data = load_period_signals(domain_name, signals_dir)
    
    if not shift_signals_data and not period_signals_data:
        print(f"No signal data found for {domain_name}")
        return ""
    
    # Create comprehensive visualization
    fig = plt.figure(figsize=(20, 16))
    
    # Main timeline view (top half)
    ax_main = plt.subplot2grid((4, 3), (0, 0), colspan=3, rowspan=2)
    create_main_timeline_view(ax_main, shift_signals_data, period_signals_data, domain_name)
    
    # Shift signals analysis (bottom left)
    ax_shift = plt.subplot2grid((4, 3), (2, 0), colspan=1)
    create_shift_signals_plot(ax_shift, shift_signals_data)
    
    # Period metrics analysis (bottom center)
    ax_period = plt.subplot2grid((4, 3), (2, 1), colspan=1)
    create_period_metrics_plot(ax_period, period_signals_data)
    
    # Confidence analysis (bottom right)
    ax_confidence = plt.subplot2grid((4, 3), (2, 2), colspan=1)
    create_confidence_analysis_plot(ax_confidence, shift_signals_data, period_signals_data)
    
    # Statistics summary (bottom row)
    ax_stats = plt.subplot2grid((4, 3), (3, 0), colspan=3)
    create_statistics_summary(ax_stats, shift_signals_data, period_signals_data, domain_name)
    
    plt.tight_layout()
    
    # Save visualization
    output_file = f"{output_dir}/{domain_name}_timeline_visualization.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    # Create individual detailed visualizations
    create_detailed_shift_visualization(shift_signals_data, domain_name, output_dir)
    create_detailed_period_visualization(period_signals_data, domain_name, output_dir)
    create_interactive_dashboard(shift_signals_data, period_signals_data, domain_name, output_dir)
    
    print(f"Main visualization saved: {output_file}")
    return output_file


def load_shift_signals(domain_name: str, signals_dir: str) -> Optional[Dict]:
    """Load shift signals data from JSON file."""
    shift_file = Path(f"{signals_dir}/{domain_name}_shift_signals.json")
    if not shift_file.exists():
        print(f"Shift signals file not found: {shift_file}")
        return None
    
    try:
        with open(shift_file, 'r') as f:
            data = json.load(f)
        print(f"Loaded shift signals: {data['paradigm_shifts']['count']} paradigm shifts")
        return data
    except Exception as e:
        print(f"Error loading shift signals: {e}")
        return None


def load_period_signals(domain_name: str, signals_dir: str) -> Optional[Dict]:
    """Load period signals data from JSON file."""
    period_file = Path(f"{signals_dir}/{domain_name}_period_signals.json")
    if not period_file.exists():
        print(f"Period signals file not found: {period_file}")
        return None
    
    try:
        with open(period_file, 'r') as f:
            data = json.load(f)
        print(f"Loaded period signals: {data['period_characterizations']['count']} periods")
        return data
    except Exception as e:
        print(f"Error loading period signals: {e}")
        return None


def create_main_timeline_view(ax, shift_data: Optional[Dict], period_data: Optional[Dict], domain_name: str):
    """Create the main timeline visualization showing periods and transitions."""
    ax.set_title(f'{domain_name.replace("_", " ").title()} - Research Timeline Evolution', 
                fontsize=16, fontweight='bold', pad=20)
    
    # Determine timeline bounds
    min_year, max_year = get_timeline_bounds(shift_data, period_data)
    if min_year == max_year:
        return
    
    # Draw period blocks
    if period_data:
        periods = period_data['period_characterizations']['characterizations']
        colors = plt.cm.Set3(np.linspace(0, 1, len(periods)))
        
        for i, (period, color) in enumerate(zip(periods, colors)):
            start_year, end_year = period['period']
            confidence = period['confidence']
            label = period['topic_label']
            
            # Draw period rectangle with transparency based on confidence
            alpha = 0.3 + 0.5 * confidence  # 0.3 to 0.8 alpha range
            rect = Rectangle((start_year, i * 0.8), end_year - start_year, 0.6, 
                           facecolor=color, alpha=alpha, edgecolor='black', linewidth=1)
            ax.add_patch(rect)
            
            # Add period label
            mid_year = (start_year + end_year) / 2
            ax.text(mid_year, i * 0.8 + 0.3, label, 
                   ha='center', va='center', fontsize=10, fontweight='bold',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
            
            # Add confidence indicator
            ax.text(end_year - 1, i * 0.8 + 0.1, f'{confidence:.2f}', 
                   ha='right', va='bottom', fontsize=8, style='italic')
    
    # Draw paradigm shift markers
    if shift_data:
        paradigm_shifts = shift_data['paradigm_shifts']['signals']
        
        for shift in paradigm_shifts:
            year = shift['year']
            confidence = shift['confidence']
            signal_type = shift['signal_type']
            
            # Draw vertical line for paradigm shift
            line_height = len(periods) * 0.8 if period_data else 1
            ax.axvline(x=year, color='red', linestyle='--', linewidth=2, alpha=0.8)
            
            # Add shift marker
            marker_size = 50 + 100 * confidence  # Size based on confidence
            ax.scatter(year, line_height + 0.2, s=marker_size, c='red', 
                      marker='v', alpha=0.8, edgecolors='darkred', linewidth=2)
            
            # Add year label
            ax.text(year, line_height + 0.5, str(year), 
                   ha='center', va='bottom', fontsize=9, fontweight='bold',
                   bbox=dict(boxstyle="round,pad=0.2", facecolor='red', alpha=0.7, edgecolor='darkred'))
    
    # Formatting
    ax.set_xlim(min_year - 2, max_year + 2)
    ax.set_ylim(-0.5, (len(periods) if period_data else 1) * 0.8 + 1)
    ax.set_xlabel('Year', fontsize=12, fontweight='bold')
    ax.set_ylabel('Research Periods', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Add legend
    legend_elements = []
    if period_data:
        legend_elements.append(patches.Patch(color='lightblue', alpha=0.6, label='Research Periods'))
    if shift_data:
        legend_elements.append(plt.Line2D([0], [0], color='red', linestyle='--', linewidth=2, label='Paradigm Shifts'))
    
    if legend_elements:
        ax.legend(handles=legend_elements, loc='upper right')


def create_shift_signals_plot(ax, shift_data: Optional[Dict]):
    """Create shift signals analysis plot."""
    if not shift_data:
        ax.text(0.5, 0.5, 'No Shift Signals Data', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Shift Signals Analysis')
        return
    
    ax.set_title('Paradigm Shift Detection', fontsize=12, fontweight='bold')
    
    # Get signal counts by stage
    raw_count = shift_data['raw_signals']['count']
    validated_count = shift_data['validated_signals']['count']
    paradigm_count = shift_data['paradigm_shifts']['count']
    
    stages = ['Raw\nSignals', 'Validated\nSignals', 'Paradigm\nShifts']
    counts = [raw_count, validated_count, paradigm_count]
    colors = ['lightcoral', 'orange', 'darkred']
    
    bars = ax.bar(stages, counts, color=colors, alpha=0.7, edgecolor='black')
    
    # Add count labels on bars
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
               f'{count}', ha='center', va='bottom', fontweight='bold')
    
    # Add retention rates
    if raw_count > 0:
        val_retention = validated_count / raw_count
        par_retention = paradigm_count / raw_count
        ax.text(0.5, 0.8, f'Validation: {val_retention:.1%}', transform=ax.transAxes, 
               ha='center', fontsize=9)
        ax.text(0.5, 0.7, f'Final: {par_retention:.1%}', transform=ax.transAxes, 
               ha='center', fontsize=9)
    
    ax.set_ylabel('Signal Count')
    ax.grid(True, alpha=0.3)


def create_period_metrics_plot(ax, period_data: Optional[Dict]):
    """Create period network metrics analysis plot."""
    if not period_data:
        ax.text(0.5, 0.5, 'No Period Signals Data', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Period Network Metrics')
        return
    
    ax.set_title('Network Stability Metrics', fontsize=12, fontweight='bold')
    
    # Extract network metrics
    periods = period_data['period_characterizations']['characterizations']
    if not periods:
        return
    
    metrics = ['network_stability', 'community_persistence', 'flow_stability', 'centrality_consensus']
    metric_labels = ['Network\nStability', 'Community\nPersistence', 'Flow\nStability', 'Centrality\nConsensus']
    
    # Calculate average metrics across all periods
    avg_metrics = []
    for metric in metrics:
        values = [p[metric] for p in periods if metric in p]
        avg_metrics.append(np.mean(values) if values else 0)
    
    # Create radar-like bar chart
    bars = ax.bar(metric_labels, avg_metrics, color='skyblue', alpha=0.7, edgecolor='navy')
    
    # Add value labels
    for bar, value in zip(bars, avg_metrics):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
               f'{value:.2f}', ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    ax.set_ylim(0, 1)
    ax.set_ylabel('Average Score')
    ax.grid(True, alpha=0.3)
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')


def create_confidence_analysis_plot(ax, shift_data: Optional[Dict], period_data: Optional[Dict]):
    """Create confidence analysis plot."""
    ax.set_title('Confidence Analysis', fontsize=12, fontweight='bold')
    
    confidences = []
    labels = []
    colors = []
    
    # Collect shift signal confidences
    if shift_data and shift_data['paradigm_shifts']['signals']:
        shift_confidences = [s['confidence'] for s in shift_data['paradigm_shifts']['signals']]
        confidences.extend(shift_confidences)
        labels.extend(['Shift'] * len(shift_confidences))
        colors.extend(['red'] * len(shift_confidences))
    
    # Collect period signal confidences
    if period_data and period_data['period_characterizations']['characterizations']:
        period_confidences = [p['confidence'] for p in period_data['period_characterizations']['characterizations']]
        confidences.extend(period_confidences)
        labels.extend(['Period'] * len(period_confidences))
        colors.extend(['blue'] * len(period_confidences))
    
    if not confidences:
        ax.text(0.5, 0.5, 'No Confidence Data', ha='center', va='center', transform=ax.transAxes)
        return
    
    # Create box plot
    data_dict = {'Confidence': confidences, 'Type': labels}
    df = pd.DataFrame(data_dict)
    
    if len(set(labels)) > 1:
        sns.boxplot(data=df, x='Type', y='Confidence', ax=ax, palette=['red', 'blue'])
    else:
        ax.hist(confidences, bins=10, alpha=0.7, color=colors[0], edgecolor='black')
        ax.set_xlabel('Confidence')
        ax.set_ylabel('Frequency')
    
    ax.grid(True, alpha=0.3)


def create_statistics_summary(ax, shift_data: Optional[Dict], period_data: Optional[Dict], domain_name: str):
    """Create statistics summary panel."""
    ax.set_title('Analysis Statistics Summary', fontsize=12, fontweight='bold')
    ax.axis('off')
    
    stats_text = f"Domain: {domain_name.replace('_', ' ').title()}\n"
    stats_text += f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n"
    
    if shift_data:
        stats_text += "SHIFT SIGNAL DETECTION:\n"
        stats_text += f"• Raw signals detected: {shift_data['raw_signals']['count']}\n"
        stats_text += f"• Validated signals: {shift_data['validated_signals']['count']}\n"
        stats_text += f"• Paradigm shifts identified: {shift_data['paradigm_shifts']['count']}\n"
        
        if shift_data['paradigm_shifts']['signals']:
            years = [s['year'] for s in shift_data['paradigm_shifts']['signals']]
            stats_text += f"• Paradigm shift years: {', '.join(map(str, sorted(years)))}\n"
        stats_text += "\n"
    
    if period_data:
        stats_text += "PERIOD CHARACTERIZATION:\n"
        stats_text += f"• Input segments: {period_data['input_segments']['count']}\n"
        stats_text += f"• Successfully characterized: {period_data['period_characterizations']['count']}\n"
        
        if period_data['period_characterizations']['characterizations']:
            avg_conf = period_data['period_characterizations']['confidence_statistics']['mean_confidence']
            stats_text += f"• Average confidence: {avg_conf:.3f}\n"
            
            total_papers = period_data['detailed_analysis']['network_statistics']['total_papers_analyzed']
            breakthrough_papers = period_data['detailed_analysis']['network_statistics']['total_breakthrough_papers']
            stats_text += f"• Papers analyzed: {total_papers} ({breakthrough_papers} breakthrough)\n"
    
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=10, 
           verticalalignment='top', fontfamily='monospace',
           bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgray', alpha=0.8))


def create_detailed_shift_visualization(shift_data: Optional[Dict], domain_name: str, output_dir: str):
    """Create detailed shift signals visualization."""
    if not shift_data:
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'{domain_name.replace("_", " ").title()} - Detailed Shift Signal Analysis', 
                fontsize=16, fontweight='bold')
    
    # Timeline of all signals
    ax = axes[0, 0]
    ax.set_title('Signal Detection Timeline')
    
    # Plot raw, validated, and paradigm signals
    signal_types = ['raw_signals', 'validated_signals', 'paradigm_shifts']
    colors = ['lightcoral', 'orange', 'darkred']
    labels = ['Raw', 'Validated', 'Paradigm']
    
    for signal_type, color, label in zip(signal_types, colors, labels):
        if signal_type in shift_data and shift_data[signal_type]['signals']:
            years = [s['year'] for s in shift_data[signal_type]['signals']]
            confidences = [s['confidence'] for s in shift_data[signal_type]['signals']]
            ax.scatter(years, [label] * len(years), s=[c*100 for c in confidences], 
                      c=color, alpha=0.7, label=label)
    
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('Year')
    
    # Confidence distribution
    ax = axes[0, 1]
    ax.set_title('Confidence Distribution')
    
    for signal_type, color, label in zip(signal_types, colors, labels):
        if signal_type in shift_data and shift_data[signal_type]['signals']:
            confidences = [s['confidence'] for s in shift_data[signal_type]['signals']]
            ax.hist(confidences, bins=10, alpha=0.6, color=color, label=label, density=True)
    
    ax.legend()
    ax.set_xlabel('Confidence')
    ax.set_ylabel('Density')
    ax.grid(True, alpha=0.3)
    
    # Signal types analysis
    ax = axes[1, 0]
    ax.set_title('Signal Types Distribution')
    
    if shift_data['paradigm_shifts']['signals']:
        signal_types_count = {}
        for signal in shift_data['paradigm_shifts']['signals']:
            sig_type = signal['signal_type']
            signal_types_count[sig_type] = signal_types_count.get(sig_type, 0) + 1
        
        ax.pie(signal_types_count.values(), labels=signal_types_count.keys(), autopct='%1.1f%%')
    
    # Evidence strength analysis
    ax = axes[1, 1]
    ax.set_title('Evidence Strength vs Confidence')
    
    if shift_data['paradigm_shifts']['signals']:
        evidence_strengths = [s['evidence_strength'] for s in shift_data['paradigm_shifts']['signals']]
        confidences = [s['confidence'] for s in shift_data['paradigm_shifts']['signals']]
        ax.scatter(evidence_strengths, confidences, alpha=0.7, s=50)
        ax.set_xlabel('Evidence Strength')
        ax.set_ylabel('Confidence')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_file = f"{output_dir}/{domain_name}_detailed_shift_analysis.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Detailed shift analysis saved: {output_file}")


def create_detailed_period_visualization(period_data: Optional[Dict], domain_name: str, output_dir: str):
    """Create detailed period characterization visualization."""
    if not period_data:
        return
    
    periods = period_data['period_characterizations']['characterizations']
    if not periods:
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'{domain_name.replace("_", " ").title()} - Detailed Period Analysis', 
                fontsize=16, fontweight='bold')
    
    # Network metrics over time
    ax = axes[0, 0]
    ax.set_title('Network Metrics Evolution')
    
    years = [(p['period'][0] + p['period'][1]) / 2 for p in periods]  # Mid-point years
    metrics = ['network_stability', 'community_persistence', 'flow_stability', 'centrality_consensus']
    metric_labels = ['Stability', 'Persistence', 'Flow', 'Consensus']
    
    for metric, label in zip(metrics, metric_labels):
        values = [p[metric] for p in periods]
        ax.plot(years, values, marker='o', label=label, linewidth=2)
    
    ax.legend()
    ax.set_xlabel('Year (Period Midpoint)')
    ax.set_ylabel('Metric Value')
    ax.grid(True, alpha=0.3)
    
    # Confidence vs Period Duration
    ax = axes[0, 1]
    ax.set_title('Confidence vs Period Duration')
    
    durations = [p['period'][1] - p['period'][0] + 1 for p in periods]
    confidences = [p['confidence'] for p in periods]
    
    ax.scatter(durations, confidences, alpha=0.7, s=50)
    ax.set_xlabel('Period Duration (Years)')
    ax.set_ylabel('Confidence')
    ax.grid(True, alpha=0.3)
    
    # Network density analysis
    ax = axes[1, 0]
    ax.set_title('Network Characteristics')
    
    if all('network_metrics' in p for p in periods):
        densities = [p['network_metrics'].get('density', 0) for p in periods]
        node_counts = [p['network_metrics'].get('number_of_nodes', 0) for p in periods]
        
        ax.scatter(node_counts, densities, alpha=0.7, s=50)
        ax.set_xlabel('Number of Papers')
        ax.set_ylabel('Network Density')
        ax.grid(True, alpha=0.3)
    
    # Period labels timeline
    ax = axes[1, 1]
    ax.set_title('Period Labels Timeline')
    ax.axis('off')
    
    timeline_text = ""
    for i, period in enumerate(periods):
        start, end = period['period']
        label = period['topic_label']
        conf = period['confidence']
        timeline_text += f"{start}-{end}: {label} (conf: {conf:.2f})\n"
    
    ax.text(0.05, 0.95, timeline_text, transform=ax.transAxes, fontsize=10,
           verticalalignment='top', fontfamily='monospace',
           bbox=dict(boxstyle="round,pad=0.5", facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    output_file = f"{output_dir}/{domain_name}_detailed_period_analysis.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Detailed period analysis saved: {output_file}")


def create_interactive_dashboard(shift_data: Optional[Dict], period_data: Optional[Dict], 
                               domain_name: str, output_dir: str):
    """Create an interactive HTML dashboard."""
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>{domain_name.replace('_', ' ').title()} - Timeline Analysis Dashboard</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
            .header {{ background-color: #2c3e50; color: white; padding: 20px; border-radius: 10px; }}
            .section {{ background-color: white; margin: 20px 0; padding: 20px; border-radius: 10px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }}
            .metric {{ display: inline-block; margin: 10px; padding: 15px; background-color: #ecf0f1; border-radius: 5px; }}
            .timeline {{ background-color: #3498db; color: white; padding: 10px; margin: 5px 0; border-radius: 5px; }}
            .shift {{ background-color: #e74c3c; color: white; padding: 10px; margin: 5px 0; border-radius: 5px; }}
            table {{ width: 100%; border-collapse: collapse; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>{domain_name.replace('_', ' ').title()} - Timeline Analysis Dashboard</h1>
            <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
    """
    
    # Add shift signals section
    if shift_data:
        html_content += f"""
        <div class="section">
            <h2>Paradigm Shift Detection</h2>
            <div class="metric">
                <strong>Raw Signals:</strong> {shift_data['raw_signals']['count']}
            </div>
            <div class="metric">
                <strong>Validated Signals:</strong> {shift_data['validated_signals']['count']}
            </div>
            <div class="metric">
                <strong>Paradigm Shifts:</strong> {shift_data['paradigm_shifts']['count']}
            </div>
            
            <h3>Detected Paradigm Shifts:</h3>
        """
        
        for shift in shift_data['paradigm_shifts']['signals']:
            html_content += f"""
            <div class="shift">
                <strong>{shift['year']}</strong> - {shift['transition_description']} 
                (Confidence: {shift['confidence']:.3f})
            </div>
            """
        
        html_content += "</div>"
    
    # Add period signals section
    if period_data:
        html_content += f"""
        <div class="section">
            <h2>Period Characterization</h2>
            <div class="metric">
                <strong>Periods Analyzed:</strong> {period_data['period_characterizations']['count']}
            </div>
            <div class="metric">
                <strong>Average Confidence:</strong> {period_data['period_characterizations']['confidence_statistics']['mean_confidence']:.3f}
            </div>
            <div class="metric">
                <strong>Papers Analyzed:</strong> {period_data['detailed_analysis']['network_statistics']['total_papers_analyzed']}
            </div>
            
            <h3>Research Periods:</h3>
            <table>
                <tr>
                    <th>Period</th>
                    <th>Label</th>
                    <th>Confidence</th>
                    <th>Network Stability</th>
                    <th>Description</th>
                </tr>
        """
        
        for period in period_data['period_characterizations']['characterizations']:
            start, end = period['period']
            html_content += f"""
                <tr>
                    <td>{start}-{end}</td>
                    <td>{period['topic_label']}</td>
                    <td>{period['confidence']:.3f}</td>
                    <td>{period['network_stability']:.3f}</td>
                    <td>{period['topic_description'][:100]}...</td>
                </tr>
            """
        
        html_content += "</table></div>"
    
    html_content += """
        <div class="section">
            <h2>Visualization Files</h2>
            <p>The following visualization files have been generated:</p>
            <ul>
                <li>Main Timeline Visualization</li>
                <li>Detailed Shift Analysis</li>
                <li>Detailed Period Analysis</li>
            </ul>
        </div>
    </body>
    </html>
    """
    
    # Save HTML dashboard
    dashboard_file = f"{output_dir}/{domain_name}_dashboard.html"
    with open(dashboard_file, 'w') as f:
        f.write(html_content)
    
    print(f"  📱 Interactive dashboard saved: {dashboard_file}")


def get_timeline_bounds(shift_data: Optional[Dict], period_data: Optional[Dict]) -> Tuple[int, int]:
    """Get the overall timeline bounds from signal data."""
    min_year, max_year = 2025, 1970  # Default bounds
    
    if shift_data and shift_data['paradigm_shifts']['signals']:
        shift_years = [s['year'] for s in shift_data['paradigm_shifts']['signals']]
        min_year = min(min_year, min(shift_years))
        max_year = max(max_year, max(shift_years))
    
    if period_data and period_data['period_characterizations']['characterizations']:
        for period in period_data['period_characterizations']['characterizations']:
            start, end = period['period']
            min_year = min(min_year, start)
            max_year = max(max_year, end)
    
    return min_year, max_year


def visualize_all_domains(signals_dir: str = "results/signals", 
                         output_dir: str = "results/visualizations") -> List[str]:
    """
    Create visualizations for all domains with signal data.
    
    Args:
        signals_dir: Directory containing signal files
        output_dir: Directory to save visualizations
        
    Returns:
        List of created visualization files
    """
    print(f"\nCREATING VISUALIZATIONS FOR ALL DOMAINS")
    print("=" * 60)
    
    signals_path = Path(signals_dir)
    if not signals_path.exists():
        print(f"Signals directory not found: {signals_dir}")
        return []
    
    # Find all domains with signal files
    domains = set()
    for file in signals_path.glob("*_shift_signals.json"):
        domain = file.stem.replace("_shift_signals", "")
        domains.add(domain)
    
    for file in signals_path.glob("*_period_signals.json"):
        domain = file.stem.replace("_period_signals", "")
        domains.add(domain)
    
    if not domains:
        print("No signal files found")
        return []
    
    print(f"Found {len(domains)} domains: {', '.join(sorted(domains))}")
    
    visualization_files = []
    for domain in sorted(domains):
        try:
            main_viz = visualize_domain_timeline(domain, signals_dir, output_dir)
            if main_viz:
                visualization_files.append(main_viz)
        except Exception as e:
            print(f"Error creating visualization for {domain}: {e}")
    
    print(f"\nCreated {len(visualization_files)} visualizations")
    return visualization_files


def visualize_timeline_comparison(domain_name: str, 
                                results_dir: str = "results",
                                validation_dir: str = "validation",
                                output_dir: str = "results/visualizations") -> str:
    """
    Create comprehensive timeline comparison visualization showing generated vs ground truth.
    
    Args:
        domain_name: Name of the domain to visualize
        results_dir: Directory containing analysis results
        validation_dir: Directory containing ground truth data
        output_dir: Directory to save visualizations
        
    Returns:
        Path to the main comparison visualization file
    """
    print(f"\nCREATING TIMELINE COMPARISON VISUALIZATION: {domain_name}")
    print("=" * 70)
    
    # Ensure output directory exists
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Load generated results and reference data
    generated_data = load_generated_timeline_data(domain_name, results_dir)
    reference_data = load_reference_data(domain_name)
    
    if not generated_data:
        print(f"No generated timeline data found for {domain_name}")
        return ""
    
    if not reference_data:
        print(f"No reference data found for {domain_name}")
        return ""
    
    # Create comprehensive comparison visualization
    fig = plt.figure(figsize=(24, 18))
    
    # Main timeline comparison (top section)
    ax_comparison = plt.subplot2grid((5, 4), (0, 0), colspan=4, rowspan=2)
    create_timeline_comparison_plot(ax_comparison, generated_data, reference_data, domain_name)
    
    # Detailed metrics comparison (middle section)
    ax_metrics = plt.subplot2grid((5, 4), (2, 0), colspan=2)
    create_metrics_comparison_plot(ax_metrics, generated_data, reference_data)
    
    # Period accuracy analysis (middle right)
    ax_accuracy = plt.subplot2grid((5, 4), (2, 2), colspan=2)
    create_accuracy_analysis_plot(ax_accuracy, generated_data, reference_data)
    
    # Boundary detection analysis (bottom left)
    ax_boundaries = plt.subplot2grid((5, 4), (3, 0), colspan=2)
    create_boundary_detection_plot(ax_boundaries, generated_data, reference_data)
    
    # Performance summary (bottom right)
    ax_performance = plt.subplot2grid((5, 4), (3, 2), colspan=2)
    create_performance_summary_plot(ax_performance, generated_data, reference_data)
    
    # Detailed statistics table (bottom row)
    ax_stats = plt.subplot2grid((5, 4), (4, 0), colspan=4)
    create_detailed_comparison_table(ax_stats, generated_data, reference_data, domain_name)
    
    plt.tight_layout()
    
    # Save comprehensive comparison
    output_file = f"{output_dir}/{domain_name}_timeline_comparison.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    # Create additional detailed visualizations
    create_period_alignment_visualization(generated_data, reference_data, domain_name, output_dir)
    create_signal_validation_visualization(generated_data, reference_data, domain_name, output_dir)
    create_performance_metrics_dashboard(generated_data, reference_data, domain_name, output_dir)
    
    print(f"  Timeline comparison saved: {output_file}")
    return output_file


def load_generated_timeline_data(domain_name: str, results_dir: str) -> Optional[Dict]:
    """Load generated timeline analysis results."""
    results_file = Path(f"{results_dir}/{domain_name}_comprehensive_analysis.json")
    if not results_file.exists():
        print(f"  Generated results file not found: {results_file}")
        return None
    
    try:
        with open(results_file, 'r') as f:
            data = json.load(f)
        
        segments = data.get('segmentation_results', {}).get('segments', [])
        periods = data.get('timeline_analysis', {}).get('original_period_characterizations', [])
        
        print(f"  Loaded generated timeline: {len(segments)} segments, {len(periods)} characterized periods")
        return data
    except Exception as e:
        print(f"  Error loading generated results: {e}")
        return None


def load_reference_data(domain_name: str, validation_dir: str = "data/references") -> Optional[Dict]:
    """Load reference timeline data from data/references."""
    ref_file = Path(f"{validation_dir}/{domain_name}_gemini.json")
    if not ref_file.exists():
        print(f"  Reference file not found: {ref_file}")
        return None

    try:
        with open(ref_file, 'r') as f:
            data = json.load(f)
        
        periods = data.get('historical_periods', [])
        print(f"  Loaded reference data: {len(periods)} historical periods")
        return data
    except Exception as e:
        print(f"  Error loading reference data: {e}")
        return None


def create_timeline_comparison_plot(ax, generated_data: Dict, reference_data: Dict, domain_name: str):
    """Create side-by-side timeline comparison showing generated vs reference periods."""
    ax.set_title(f'{domain_name.replace("_", " ").title()} - Timeline Comparison: Generated vs Reference', 
                fontsize=16, fontweight='bold', pad=20)
    
    # Extract data
    generated_periods = generated_data.get('timeline_analysis', {}).get('original_period_characterizations', [])
    ref_periods = reference_data.get('historical_periods', [])
    
    # Determine timeline bounds
    all_years = []
    for period in generated_periods:
        all_years.extend(period['period'])
    for period in ref_periods:
        all_years.extend([period['start_year'], period['end_year']])
    
    if not all_years:
        ax.text(0.5, 0.5, 'No timeline data available', ha='center', va='center', transform=ax.transAxes)
        return
    
    min_year, max_year = min(all_years), max(all_years)
    
    # Plot reference periods (top half)
    ref_colors = plt.cm.Set1(np.linspace(0, 1, len(ref_periods)))
    for i, (period, color) in enumerate(zip(ref_periods, ref_colors)):
        start_year = period['start_year']
        end_year = period['end_year']
        duration = end_year - start_year + 1
        
        # Draw period rectangle
        y_pos = 1.2 + i * 0.25
        rect = Rectangle((start_year, y_pos), duration, 0.2, 
                        facecolor=color, alpha=0.7, edgecolor='black', linewidth=1)
        ax.add_patch(rect)
        
        # Add period label
        mid_year = (start_year + end_year) / 2
        ax.text(mid_year, y_pos + 0.1, period['period_name'][:30] + '...' if len(period['period_name']) > 30 else period['period_name'], 
               ha='center', va='center', fontsize=9, fontweight='bold',
               bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8))
        
        # Add duration label
        ax.text(end_year + 1, y_pos + 0.05, f'{duration}y', 
               ha='left', va='center', fontsize=8, style='italic')
    
    # Plot generated periods (bottom half)
    gen_colors = plt.cm.Set2(np.linspace(0, 1, len(generated_periods)))
    for i, (period, color) in enumerate(zip(generated_periods, gen_colors)):
        start_year, end_year = period['period']
        duration = end_year - start_year + 1
        confidence = period['confidence']
        
        # Draw period rectangle with transparency based on confidence
        y_pos = 0.2 + i * 0.25
        alpha = 0.4 + 0.5 * confidence
        rect = Rectangle((start_year, y_pos), duration, 0.2, 
                        facecolor=color, alpha=alpha, edgecolor='black', linewidth=1)
        ax.add_patch(rect)
        
        # Add period label
        mid_year = (start_year + end_year) / 2
        label = period['topic_label'][:25] + '...' if len(period['topic_label']) > 25 else period['topic_label']
        ax.text(mid_year, y_pos + 0.1, label, 
               ha='center', va='center', fontsize=9, fontweight='bold',
               bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8))
        
        # Add confidence and duration
        ax.text(end_year + 1, y_pos + 0.05, f'{duration}y, C:{confidence:.2f}', 
               ha='left', va='center', fontsize=8, style='italic')
    
    # Add paradigm shift markers from generated data
    if 'segmentation_results' in generated_data:
        change_points = generated_data['segmentation_results'].get('change_points', [])
        for cp in change_points:
            ax.axvline(x=cp, color='red', linestyle='--', linewidth=2, alpha=0.8)
            ax.text(cp, len(generated_periods) * 0.25 + 0.5, str(cp), 
                   ha='center', va='bottom', fontsize=9, rotation=90,
                   bbox=dict(boxstyle="round,pad=0.2", facecolor='red', alpha=0.7))
    
    # Formatting
    ax.set_xlim(min_year - 5, max_year + 15)
    ax.set_ylim(0, max(len(generated_periods) * 0.25 + 0.8, len(ref_periods) * 0.25 + 1.4))
    ax.set_xlabel('Year', fontsize=12, fontweight='bold')
    ax.set_ylabel('Timeline Periods', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Add section labels
    ax.text(min_year - 3, len(ref_periods) * 0.25 + 1.3, 'REFERENCE\nDATA', 
           ha='center', va='center', fontsize=12, fontweight='bold',
           bbox=dict(boxstyle="round,pad=0.5", facecolor='lightblue', alpha=0.8))
    ax.text(min_year - 3, len(generated_periods) * 0.25 / 2 + 0.3, 'GENERATED', 
           ha='center', va='center', fontsize=12, fontweight='bold',
           bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgreen', alpha=0.8))
    
    # Add legend
    legend_elements = [
        patches.Patch(color='lightblue', alpha=0.7, label='Reference Periods'),
        patches.Patch(color='lightgreen', alpha=0.7, label='Generated Periods'),
        plt.Line2D([0], [0], color='red', linestyle='--', linewidth=2, label='Detected Change Points')
    ]
    ax.legend(handles=legend_elements, loc='upper right')


def create_metrics_comparison_plot(ax, generated_data: Dict, ground_truth_data: Dict):
    """Create metrics comparison between generated and ground truth."""
    ax.set_title('Timeline Metrics Comparison', fontsize=12, fontweight='bold')
    
    # Calculate metrics
    gen_periods = generated_data.get('timeline_analysis', {}).get('original_period_characterizations', [])
    gt_periods = ground_truth_data.get('historical_periods', [])
    
    if not gen_periods or not gt_periods:
        ax.text(0.5, 0.5, 'Insufficient data for comparison', ha='center', va='center', transform=ax.transAxes)
        return
    
    # Calculate comparison metrics
    gen_count = len(gen_periods)
    gt_count = len(gt_periods)
    
    gen_durations = [p['period'][1] - p['period'][0] + 1 for p in gen_periods]
    gt_durations = [p['end_year'] - p['start_year'] + 1 for p in gt_periods]
    
    avg_gen_duration = np.mean(gen_durations)
    avg_gt_duration = np.mean(gt_durations)
    
    gen_confidences = [p['confidence'] for p in gen_periods]
    avg_confidence = np.mean(gen_confidences)
    
    # Create comparison chart
    metrics = ['Period Count', 'Avg Duration', 'Confidence']
    generated_values = [gen_count, avg_gen_duration, avg_confidence * 20]  # Scale confidence for visibility
    gt_values = [gt_count, avg_gt_duration, 20]  # Ground truth "confidence" = 20 for scale
    
    x = np.arange(len(metrics))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, generated_values, width, label='Generated', alpha=0.7, color='skyblue')
    bars2 = ax.bar(x + width/2, gt_values, width, label='Ground Truth', alpha=0.7, color='lightcoral')
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:  # Only label non-zero bars
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                       f'{height:.1f}' if metrics[bars.index(bar) % len(metrics)] != 'Period Count' else f'{int(height)}',
                       ha='center', va='bottom', fontweight='bold')
    
    ax.set_ylabel('Value')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add special note for confidence
    ax.text(0.5, 0.9, 'Note: Confidence scaled ×20 for visibility', 
           transform=ax.transAxes, ha='center', fontsize=9, style='italic')


def create_accuracy_analysis_plot(ax, generated_data: Dict, ground_truth_data: Dict):
    """Create period accuracy analysis plot."""
    ax.set_title('Period Boundary Accuracy Analysis', fontsize=12, fontweight='bold')
    
    # Calculate boundary matching accuracy
    gen_boundaries = set()
    gen_periods = generated_data.get('timeline_analysis', {}).get('original_period_characterizations', [])
    for period in gen_periods:
        gen_boundaries.add(period['period'][0])  # Start year
        gen_boundaries.add(period['period'][1])  # End year
    
    gt_boundaries = set()
    gt_periods = ground_truth_data.get('historical_periods', [])
    for period in gt_periods:
        gt_boundaries.add(period['start_year'])
        gt_boundaries.add(period['end_year'])
    
    if not gen_boundaries or not gt_boundaries:
        ax.text(0.5, 0.5, 'No boundary data available', ha='center', va='center', transform=ax.transAxes)
        return
    
    # Calculate accuracy metrics with tolerance
    tolerances = [0, 1, 2, 3, 5]
    accuracies = []
    
    for tolerance in tolerances:
        matches = 0
        for gt_boundary in gt_boundaries:
            for gen_boundary in gen_boundaries:
                if abs(gt_boundary - gen_boundary) <= tolerance:
                    matches += 1
                    break
        accuracy = matches / len(gt_boundaries)
        accuracies.append(accuracy)
    
    # Plot accuracy vs tolerance
    ax.plot(tolerances, accuracies, marker='o', linewidth=3, markersize=8, color='green')
    ax.fill_between(tolerances, accuracies, alpha=0.3, color='green')
    
    # Add value labels
    for i, (tolerance, accuracy) in enumerate(zip(tolerances, accuracies)):
        ax.text(tolerance, accuracy + 0.02, f'{accuracy:.2f}', 
               ha='center', va='bottom', fontweight='bold')
    
    ax.set_xlabel('Tolerance (Years)')
    ax.set_ylabel('Boundary Accuracy')
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3)
    
    # Add exact match indicator
    exact_accuracy = accuracies[0]
    ax.axhline(y=exact_accuracy, color='red', linestyle='--', alpha=0.7)
    ax.text(max(tolerances), exact_accuracy + 0.05, f'Exact: {exact_accuracy:.2f}', 
           ha='right', va='bottom', color='red', fontweight='bold')


def create_boundary_detection_plot(ax, generated_data: Dict, ground_truth_data: Dict):
    """Create boundary detection analysis."""
    ax.set_title('Change Point Detection Analysis', fontsize=12, fontweight='bold')
    
    # Get detected change points
    change_points = generated_data.get('segmentation_results', {}).get('change_points', [])
    
    # Get ground truth boundaries
    gt_boundaries = []
    gt_periods = ground_truth_data.get('historical_periods', [])
    for i, period in enumerate(gt_periods[:-1]):  # Exclude last period end
        gt_boundaries.append(period['end_year'] + 1)  # Transition point
    
    if not change_points and not gt_boundaries:
        ax.text(0.5, 0.5, 'No change points detected', ha='center', va='center', transform=ax.transAxes)
        return
    
    # Calculate detection performance
    true_positives = 0
    false_positives = 0
    tolerance = 3  # 3-year tolerance window
    
    detected_matches = set()
    for cp in change_points:
        matched = False
        for gt_boundary in gt_boundaries:
            if abs(cp - gt_boundary) <= tolerance and gt_boundary not in detected_matches:
                true_positives += 1
                detected_matches.add(gt_boundary)
                matched = True
                break
        if not matched:
            false_positives += 1
    
    false_negatives = len(gt_boundaries) - true_positives
    
    # Create performance visualization
    categories = ['True Positives', 'False Positives', 'False Negatives']
    values = [true_positives, false_positives, false_negatives]
    colors = ['green', 'orange', 'red']
    
    bars = ax.bar(categories, values, color=colors, alpha=0.7, edgecolor='black')
    
    # Add value labels
    for bar, value in zip(bars, values):
        if value > 0:
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.1,
                   str(value), ha='center', va='bottom', fontweight='bold')
    
    # Calculate and display metrics
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    metrics_text = f'Precision: {precision:.3f}\nRecall: {recall:.3f}\nF1-Score: {f1:.3f}'
    ax.text(0.02, 0.98, metrics_text, transform=ax.transAxes, fontsize=10, 
           verticalalignment='top', fontfamily='monospace',
           bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgray', alpha=0.8))
    
    ax.set_ylabel('Count')
    ax.grid(True, alpha=0.3)


def create_performance_summary_plot(ax, generated_data: Dict, ground_truth_data: Dict):
    """Create overall performance summary."""
    ax.set_title('Overall Performance Summary', fontsize=12, fontweight='bold')
    
    # Calculate various performance metrics
    gen_periods = generated_data.get('timeline_analysis', {}).get('original_period_characterizations', [])
    gt_periods = ground_truth_data.get('historical_periods', [])
    
    if not gen_periods or not gt_periods:
        ax.text(0.5, 0.5, 'Insufficient data for performance analysis', ha='center', va='center', transform=ax.transAxes)
        return
    
    # Temporal coverage accuracy
    gen_span = max([p['period'][1] for p in gen_periods]) - min([p['period'][0] for p in gen_periods])
    gt_span = max([p['end_year'] for p in gt_periods]) - min([p['start_year'] for p in gt_periods])
    coverage_accuracy = min(gen_span, gt_span) / max(gen_span, gt_span) if max(gen_span, gt_span) > 0 else 0
    
    # Period count accuracy
    count_accuracy = min(len(gen_periods), len(gt_periods)) / max(len(gen_periods), len(gt_periods))
    
    # Average confidence
    avg_confidence = np.mean([p['confidence'] for p in gen_periods])
    
    # Statistical significance
    stat_sig = generated_data.get('analysis_metadata', {}).get('methodology', {}).get('statistical_significance', 0)
    
    # Create radar-like visualization
    metrics = ['Temporal\nCoverage', 'Period Count\nAccuracy', 'Avg\nConfidence', 'Statistical\nSignificance']
    values = [coverage_accuracy, count_accuracy, avg_confidence, stat_sig]
    
    # Normalize statistical significance to 0-1 scale
    values[3] = min(values[3], 1.0)
    
    bars = ax.bar(metrics, values, color=['skyblue', 'lightgreen', 'gold', 'lightcoral'], 
                 alpha=0.7, edgecolor='black')
    
    # Add value labels
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
               f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    ax.set_ylim(0, 1.1)
    ax.set_ylabel('Score (0-1)')
    ax.grid(True, alpha=0.3)
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    
    # Add overall performance score
    overall_score = np.mean(values)
    ax.text(0.5, 0.9, f'Overall Score: {overall_score:.3f}', 
           transform=ax.transAxes, ha='center', fontsize=12, fontweight='bold',
           bbox=dict(boxstyle="round,pad=0.4", facecolor='yellow', alpha=0.8))


def create_detailed_comparison_table(ax, generated_data: Dict, ground_truth_data: Dict, domain_name: str):
    """Create detailed statistics comparison table."""
    ax.set_title('Detailed Comparison Statistics', fontsize=12, fontweight='bold')
    ax.axis('off')
    
    # Gather comprehensive statistics
    gen_periods = generated_data.get('timeline_analysis', {}).get('original_period_characterizations', [])
    gt_periods = ground_truth_data.get('historical_periods', [])
    change_points = generated_data.get('segmentation_results', {}).get('change_points', [])
    
    stats_text = f"DOMAIN: {domain_name.replace('_', ' ').title()}\n"
    stats_text += f"ANALYSIS DATE: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n"
    
    stats_text += "TIMELINE COMPARISON:\n"
    stats_text += f"• Ground Truth Periods: {len(gt_periods)}\n"
    stats_text += f"• Generated Periods: {len(gen_periods)}\n"
    stats_text += f"• Change Points Detected: {len(change_points)}\n"
    
    if gt_periods:
        gt_span = max([p['end_year'] for p in gt_periods]) - min([p['start_year'] for p in gt_periods])
        gt_avg_duration = np.mean([p['end_year'] - p['start_year'] + 1 for p in gt_periods])
        stats_text += f"• Ground Truth Span: {gt_span} years\n"
        stats_text += f"• Ground Truth Avg Duration: {gt_avg_duration:.1f} years\n"
    
    if gen_periods:
        gen_span = max([p['period'][1] for p in gen_periods]) - min([p['period'][0] for p in gen_periods])
        gen_avg_duration = np.mean([p['period'][1] - p['period'][0] + 1 for p in gen_periods])
        gen_avg_confidence = np.mean([p['confidence'] for p in gen_periods])
        stats_text += f"• Generated Span: {gen_span} years\n"
        stats_text += f"• Generated Avg Duration: {gen_avg_duration:.1f} years\n"
        stats_text += f"• Average Confidence: {gen_avg_confidence:.3f}\n"
    
    # Add method details
    method_details = generated_data.get('analysis_metadata', {}).get('methodology', {})
    if method_details:
        stats_text += f"\nMETHODOLOGY:\n"
        stats_text += f"• Statistical Significance: {method_details.get('statistical_significance', 0):.3f}\n"
        stats_text += f"• Total Papers Analyzed: {generated_data.get('analysis_metadata', {}).get('total_papers_analyzed', 0)}\n"
        stats_text += f"• Shift Detection: {method_details.get('shift_detection', 'Unknown')}\n"
        stats_text += f"• Period Characterization: {method_details.get('period_characterization', 'Unknown')}\n"
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10, 
           verticalalignment='top', fontfamily='monospace',
           bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgray', alpha=0.8))


def create_period_alignment_visualization(generated_data: Dict, ground_truth_data: Dict, 
                                        domain_name: str, output_dir: str):
    """Create detailed period alignment visualization."""
    fig, ax = plt.subplots(1, 1, figsize=(20, 12))
    fig.suptitle(f'{domain_name.replace("_", " ").title()} - Period Alignment Analysis', 
                fontsize=16, fontweight='bold')
    
    # Similar to main comparison but with more detail and alignment lines
    gen_periods = generated_data.get('timeline_analysis', {}).get('original_period_characterizations', [])
    gt_periods = ground_truth_data.get('historical_periods', [])
    
    # Plot periods with alignment indicators
    # ... (Implementation continues with detailed alignment visualization)
    
    plt.tight_layout()
    output_file = f"{output_dir}/{domain_name}_period_alignment.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"  Period alignment visualization saved: {output_file}")


def create_signal_validation_visualization(generated_data: Dict, ground_truth_data: Dict, 
                                         domain_name: str, output_dir: str):
    """Create signal validation against ground truth events."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'{domain_name.replace("_", " ").title()} - Signal Validation Analysis', 
                fontsize=16, fontweight='bold')
    
    # ... (Implementation for signal validation visualization)
    
    plt.tight_layout()
    output_file = f"{output_dir}/{domain_name}_signal_validation.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"  Signal validation visualization saved: {output_file}")


def create_performance_metrics_dashboard(generated_data: Dict, ground_truth_data: Dict, 
                                       domain_name: str, output_dir: str):
    """Create comprehensive performance metrics dashboard."""
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    fig.suptitle(f'{domain_name.replace("_", " ").title()} - Performance Metrics Dashboard', 
                fontsize=16, fontweight='bold')
    
    # ... (Implementation for comprehensive performance dashboard)
    
    plt.tight_layout()
    output_file = f"{output_dir}/{domain_name}_performance_dashboard.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"  Performance dashboard saved: {output_file}")


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description="Timeline Analysis Visualization")
    parser.add_argument('--domain', type=str, help='Domain to visualize (or "all" for all domains)')
    parser.add_argument('--signals-dir', type=str, default='results/signals', help='Signals directory')
    parser.add_argument('--output-dir', type=str, default='results/visualizations', help='Output directory')
    
    args = parser.parse_args()
    
    if args.domain == 'all':
        visualize_all_domains(args.signals_dir, args.output_dir)
    elif args.domain:
        visualize_domain_timeline(args.domain, args.signals_dir, args.output_dir)
    else:
        print("Please specify --domain or use --domain all") 