import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Any, Set
import time

# Import core algorithm modules
from core.data_loader import discover_available_domains
from core.data_processing import process_domain_data
from core.algorithm_config import ComprehensiveAlgorithmConfig
from core.shift_signal_detection import (
    detect_shift_signals,
    detect_research_direction_changes,
    detect_citation_structural_breaks,
)
from core.change_detection import create_segments_with_confidence
from core.data_models import DomainData


# Suppress PyTorch warnings
import torch
torch.classes.__path__ = []

# Configure Streamlit page
st.set_page_config(
    page_title="Timeline Segmentation Algorithm Dashboard",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for styling
st.markdown(
    """
<style>
    .metric-container {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .success-message {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 0.75rem;
        border-radius: 0.25rem;
        margin: 1rem 0;
    }
    .info-message {
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        color: #0c5460;
        padding: 0.75rem;
        border-radius: 0.25rem;
        margin: 1rem 0;
    }
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_data
def load_all_domains() -> Dict[str, DomainData]:
    """Load data for all available domains."""
    # Discover available domains dynamically
    available_domains = discover_available_domains()

    domain_data = {}

    for domain in available_domains.keys():
        try:
            # Use proper data processing function
            processing_result = process_domain_data(domain)
            if processing_result.success:
                domain_data[domain] = processing_result.domain_data
            else:
                st.warning(
                    f"Failed to process {domain}: {processing_result.error_message}"
                )
        except Exception as e:
            st.warning(f"Failed to load {domain}: {e}")

    return domain_data


def extract_citation_time_series(
    domain_data: DomainData,
) -> Tuple[List[int], List[float]]:
    """Extract citation time series from domain data."""
    citation_series = defaultdict(float)

    for paper in domain_data.papers:
        year = paper.pub_year
        citation_series[year] += paper.cited_by_count

    if not citation_series:
        return [], []

    years = sorted(citation_series.keys())
    citation_values = [citation_series[year] for year in years]

    return years, citation_values


def run_algorithm_with_params(
    domain_data: DomainData, domain_name: str, algorithm_config: ComprehensiveAlgorithmConfig
) -> Tuple[Any, Any, List[List[int]], Dict]:
    """Run the algorithm with specified parameters and return all signal data for visualization."""

    # Get raw direction signals data WITH analysis data for visualization
    raw_direction_signals, keyword_analysis = detect_research_direction_changes(
        domain_data, algorithm_config.direction_threshold, return_analysis_data=True
    )

    # Get citation signals
    citation_signals = detect_citation_structural_breaks(domain_data, domain_name)

    # Create precomputed signals dict to avoid re-running detection
    precomputed_signals = {
        "direction": raw_direction_signals,
        "citation": citation_signals,
    }

    # Detect shift signals using precomputed signals - now returns clustering metadata
    shift_signals, transition_evidence, clustering_metadata = detect_shift_signals(
        domain_data,
        domain_name,
        algorithm_config=algorithm_config,
        precomputed_signals=precomputed_signals,
    )

    # Convert to change points and create segments
    change_years = [signal.year for signal in shift_signals]

    # Calculate statistical significance from shift signals
    if shift_signals:
        statistical_significance = np.mean(
            [signal.confidence for signal in shift_signals]
        )
    else:
        statistical_significance = 0.0

    # Create segments with statistical significance calibration
    segments = create_segments_with_confidence(
        change_years, domain_data.year_range, statistical_significance, domain_name
    )

    # Return clustering metadata with all visualization data
    enhanced_signal_data = {
        "clustering_metadata": clustering_metadata,
        "keyword_analysis": keyword_analysis,
    }

    return shift_signals, transition_evidence, segments, enhanced_signal_data

def create_comprehensive_validation_plot(
    domain_name: str,
    validated_signals: List,
    enhanced_signal_data: Dict,
    algorithm_config: ComprehensiveAlgorithmConfig,
    domain_data: DomainData,
):
    """Create a comprehensive visualization showing the complete validation process."""

    clustering_metadata = enhanced_signal_data.get("clustering_metadata", {})
    keyword_analysis = enhanced_signal_data.get("keyword_analysis", {})

    citation_signals = clustering_metadata.get("citation_signals", [])

    # Create subplots for comprehensive view
    fig = make_subplots(
        rows=3,
        cols=1,
        subplot_titles=(
            "Step 1: Raw Direction Signal Detection",
            "Step 2: Citation Signal Detection",
            "Step 3: Final Validation with Consistent Threshold",
        ),
        vertical_spacing=0.15,
        shared_xaxes=False,
        row_heights=[0.3, 0.3, 0.4],
        specs=[
            [{"secondary_y": False}],
            [{"secondary_y": False}],  # No secondary y-axis needed for citation only
            [{"secondary_y": False}],
        ],
    )

    # STEP 1: Raw Direction Signal Detection
    if keyword_analysis and keyword_analysis.get("years"):
        all_years = keyword_analysis["years"]
        all_scores = keyword_analysis["direction_score"]
        detection_threshold = keyword_analysis["threshold"]
        novel_keywords_data = keyword_analysis.get("novel_keywords", [])
        top_keywords_data = keyword_analysis.get("top_keywords", [])

        # Define a professional color palette for Step 1
        line_color = "#636E72"  # Soothing dark gray for the base line
        marker_color = "#636E72" # Same as line color for consistency
        highlight_color = "#E17055" # Professional orange/salmon for detected signals
        threshold_color = "#D73027"  # Consistent professional red for thresholds

        # Create enhanced hover text with keyword information
        hover_texts = []
        for i, (year, score) in enumerate(zip(all_years, all_scores)):
            # Get keyword data for this year
            novel_keywords = novel_keywords_data[i] if i < len(novel_keywords_data) else []
            top_keywords = top_keywords_data[i] if i < len(top_keywords_data) else []
            
            # Format novel keywords
            novel_text = ", ".join(novel_keywords[:5]) if novel_keywords else "None"
            if len(novel_keywords) > 5:
                novel_text += f" (+{len(novel_keywords)-5} more)"
            
            # Format top keywords  
            top_text = ", ".join(top_keywords[:5]) if top_keywords else "None"
            if len(top_keywords) > 5:
                top_text += f" (+{len(top_keywords)-5} more)"
            
            hover_text = (
                f"<b>Year:</b> {year}<br>"
                f"<b>Direction Score:</b> {score:.3f}<br>"
                f"<b>Novel Keywords:</b> {novel_text}<br>"
                f"<b>Top Keywords:</b> {top_text}"
            )
            hover_texts.append(hover_text)

        # All raw signals line with markers and enhanced hover
        fig.add_trace(
            go.Scatter(
                x=all_years,
                y=all_scores,
                mode="lines+markers",
                name="Raw Direction Score",
                line=dict(color=line_color, width=2),
                marker=dict(size=5, color=marker_color, opacity=0.7),
                hovertemplate="%{hovertext}<extra></extra>",
                hovertext=hover_texts,
            ),
            row=1,
            col=1,
        )

        # Detected signals (above threshold) - highlighted markers with enhanced hover
        detected_years = []
        detected_scores = []
        detected_hover_texts = []
        for i, (year, score) in enumerate(zip(all_years, all_scores)):
            if score >= detection_threshold:
                detected_years.append(year)
                detected_scores.append(score)
                
                # Get keyword data for detected signals
                novel_keywords = novel_keywords_data[i] if i < len(novel_keywords_data) else []
                top_keywords = top_keywords_data[i] if i < len(top_keywords_data) else []
                
                # Format novel keywords
                novel_text = ", ".join(novel_keywords[:5]) if novel_keywords else "None"
                if len(novel_keywords) > 5:
                    novel_text += f" (+{len(novel_keywords)-5} more)"
                
                # Format top keywords  
                top_text = ", ".join(top_keywords[:5]) if top_keywords else "None"
                if len(top_keywords) > 5:
                    top_text += f" (+{len(top_keywords)-5} more)"
                
                detected_hover_text = (
                    f"<b>Year:</b> {year}<br>"
                    f"<b>Direction Score:</b> {score:.3f}<br>"
                    f"<b>Status:</b> Detected<br>"
                    f"<b>Novel Keywords:</b> {novel_text}<br>"
                    f"<b>Top Keywords:</b> {top_text}"
                )
                detected_hover_texts.append(detected_hover_text)

        if detected_years:
            fig.add_trace(
                go.Scatter(
                    x=detected_years,
                    y=detected_scores,
                    mode="markers",
                    name="Detected Signal",
                    marker=dict(
                        size=10,
                        color=highlight_color,
                        symbol="circle",
                        opacity=1.0,
                        line=dict(width=1.5, color=line_color),
                    ),
                    hovertemplate="%{hovertext}<extra></extra>",
                    hovertext=detected_hover_texts,
                ),
                row=1,
                col=1,
            )

        # Detection threshold line with improved label placement
        fig.add_hline(
            y=detection_threshold,
            line=dict(color=threshold_color, width=2, dash="dash"),
            annotation_text=f"Detection Threshold ({detection_threshold:.2f})",
            annotation_position="top left",
            row=1,
        )

    # STEP 2: Citation Signal Detection
    # Extract citation time series from domain data
    citation_series = defaultdict(float)
    for paper in domain_data.papers:
        if paper.pub_year and paper.cited_by_count:
            citation_series[paper.pub_year] += paper.cited_by_count

    if citation_series:
        # Get sorted years and citation counts
        sorted_years = sorted(citation_series.keys())
        citation_counts = [citation_series[year] for year in sorted_years]
        
        # Calculate adaptive threshold (using statistical approach)
        # Use mean + 1.5 * standard deviation as adaptive threshold
        mean_citations = np.mean(citation_counts)
        std_citations = np.std(citation_counts)
        adaptive_threshold = mean_citations + 1.5 * std_citations
        
        # Add citation time series line with professional color scheme
        # Using a sophisticated teal-based palette inspired by scientific journals
        line_color = "#636E72"  # Soothing dark gray for the base line
        marker_color = "#636E72" # Same as line color for consistency
        highlight_color = "#FFF287"  # Bright yellow for highlighted points
        threshold_color = "#D73027"  # Professional red for threshold (from RdYlBu palette)

        fig.add_trace(
            go.Scatter(
                x=sorted_years,
                y=citation_counts,
                mode="lines+markers",
                name="Total Citations",
                line=dict(color=line_color, width=2.5),
                marker=dict(size=5, color=marker_color, opacity=0.7),
                hovertemplate="<b>Year:</b> %{x}<br><b>Total Citations:</b> %{y:,.0f}<extra></extra>",
            ),
            row=2,
            col=1,
        )
        
        # Add adaptive threshold line with professional styling
        fig.add_hline(
            y=adaptive_threshold,
            line=dict(color=threshold_color, width=2, dash="dash"),
            annotation_text=f"Adaptive Threshold ({adaptive_threshold:,.0f})",
            annotation_position="top left",
            row=2,
        )
        
        # Highlight years above threshold with enhanced markers
        above_threshold_years = []
        above_threshold_counts = []
        
        for year, count in zip(sorted_years, citation_counts):
            if count > adaptive_threshold:
                above_threshold_years.append(year)
                above_threshold_counts.append(count)
        
        if above_threshold_years:
            fig.add_trace(
                go.Scatter(
                    x=above_threshold_years,
                    y=above_threshold_counts,
                    mode="markers",
                    name="Above Threshold",
                    marker=dict(
                        size=12,
                        color=highlight_color,  # Darker teal for strong contrast
                        symbol="circle",
                        opacity=1.0,  # Full opacity for highlighting
                        line=dict(width=2, color=line_color),
                    ),
                    hovertemplate="<b>Year:</b> %{x}<br><b>Citations:</b> %{y:,.0f}<br><b>Status:</b> Above Threshold<extra></extra>",
                ),
                row=2,
                col=1,
            )

    # STEP 3: Final Validation with Consistent Threshold
    # Show ALL direction signals (both validated and filtered) for complete picture
    all_direction_years = []
    all_direction_confidences = []
    all_direction_status = []
    all_direction_hover = []
    
    # Get all clustered direction signals from metadata
    clustered_direction_signals = clustering_metadata.get('clustered_direction_signals', [])
    validated_years = {s.year for s in validated_signals}
    
    for direction_signal in clustered_direction_signals:
        year = direction_signal.year
        base_confidence = direction_signal.confidence
        
        # Check if this signal was validated
        is_validated = year in validated_years
        
        if is_validated:
            # Find the validated signal to get final confidence
            validated_signal = next((s for s in validated_signals if s.year == year), None)
            final_confidence = validated_signal.confidence if validated_signal else base_confidence
            status = "Validated"
            color = "green"
            symbol = "circle"
        else:
            # Calculate what the final confidence would have been (for display)
            # Check for citation support
            citation_support = any(abs(cs.year - year) <= 2 for cs in citation_signals)
            
            confidence_boosts = 0.0
            if citation_support:
                confidence_boosts += algorithm_config.citation_boost

            final_confidence = min(base_confidence + confidence_boosts, 1.0)
            status = "Filtered"
            color = "red" 
            symbol = "x"
        
        all_direction_years.append(year)
        all_direction_confidences.append(final_confidence)
        all_direction_status.append(status)
        
        # Create detailed hover text
        hover_text = f"<b>Year:</b> {year}<br>"
        hover_text += f"<b>Base Confidence:</b> {base_confidence:.3f}<br>"
        
        # Add boost information
        citation_support = any(abs(cs.year - year) <= 2 for cs in citation_signals)
        
        boost_info = []
        if citation_support:
            boost_info.append(f"Citation: +{algorithm_config.citation_boost:.2f}")

        if boost_info:
            hover_text += f"<b>Boosts:</b> {', '.join(boost_info)}<br>"
            
        hover_text += f"<b>Final Confidence:</b> {final_confidence:.3f}<br>"
        hover_text += f"<b>Threshold:</b> {algorithm_config.validation_threshold:.3f}<br>"
        hover_text += f"<b>Status:</b> {status}"
        
        all_direction_hover.append(hover_text)

    # Plot validation results
    for status in ["Validated", "Filtered"]:
        status_indices = [i for i, s in enumerate(all_direction_status) if s == status]
        if status_indices:
            status_years = [all_direction_years[i] for i in status_indices]
            status_scores = [all_direction_confidences[i] for i in status_indices]
            status_hover = [all_direction_hover[i] for i in status_indices]

            color = "green" if status == "Validated" else "red"
            symbol = "circle" if status == "Validated" else "x"
            marker_size = 14 if status == "Validated" else 10

            fig.add_trace(
                go.Scatter(
                    x=status_years,
                    y=status_scores,
                    mode="markers",
                    name=f"{status} Signals",
                    marker=dict(
                        size=marker_size,
                        color=color,
                        symbol=symbol,
                        line=dict(width=2, color="black"),
                        opacity=1.0 if status == "Validated" else 0.7,
                    ),
                    hovertemplate="%{hovertext}<extra></extra>",
                    hovertext=status_hover,
                ),
                row=3,
                col=1,
            )

    # Validation threshold line
    fig.add_hline(
        y=algorithm_config.validation_threshold,
        line=dict(color="red", width=2, dash="dash"),
        annotation_text=f"Validation Threshold ({algorithm_config.validation_threshold:.2f})",
        annotation_position="top left",
        row=3,
    )

    # Update layout
    fig.update_layout(
        height=800,
        showlegend=True,
        title_text=f"Complete Validation Process: {domain_name.replace('_', ' ').title()}",
        legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02),
    )

    # Update axes labels
    fig.update_xaxes(title_text="Year", row=3, col=1)
    fig.update_yaxes(title_text="Direction Score", row=1, col=1)
    fig.update_yaxes(title_text="Total Citations", row=2, col=1)
    fig.update_yaxes(title_text="Final Confidence", row=3, col=1)

    return fig


def create_improved_segments_plot(
    domain_name: str,
    segments: List[List[int]],
    domain_data: DomainData,
    shift_signals: List,
    algorithm_config: ComprehensiveAlgorithmConfig,
):
    """Create an improved timeline visualization showing segments with better details."""

    fig = go.Figure()

    # Enhanced color palette
    colors = px.colors.qualitative.Set2

    # Calculate segment statistics
    segment_lengths = [end - start + 1 for start, end in segments]
    avg_length = np.mean(segment_lengths) if segment_lengths else 0

    for i, (start_year, end_year) in enumerate(segments):
        segment_length = end_year - start_year + 1
        color = colors[i % len(colors)]

        # Determine segment quality based on length
        if segment_length >= avg_length * 1.5:
            quality = "Long Period"
            opacity = 0.8
        elif segment_length >= avg_length * 0.7:
            quality = "Standard Period"
            opacity = 0.7
        else:
            quality = "Short Period"
            opacity = 0.6

        # Add segment bar with enhanced information
        fig.add_trace(
            go.Bar(
                x=[segment_length],
                y=[f"Period {i+1}"],
                orientation="h",
                name=f"Period {i+1}",
                marker_color=color,
                marker_line=dict(width=1, color="black"),
                opacity=opacity,
                text=f"{start_year}-{end_year}",
                textposition="inside",
                base=start_year,
                hovertemplate=f"<b>Period {i+1}</b><br>"
                + f"Years: {start_year}-{end_year}<br>"
                + f"Duration: {segment_length} years<br>"
                + f"Quality: {quality}<br>"
                + f"<extra></extra>",
            )
        )

    # Add paradigm shift markers
    change_years = [signal.year for signal in shift_signals]
    for i, year in enumerate(change_years):
        # Find corresponding signal for details
        signal = shift_signals[i] if i < len(shift_signals) else None

        fig.add_vline(
            x=year,
            line=dict(color="#D73027", width=3, dash="dot")
        )

    # Update layout with enhanced styling
    fig.update_layout(
        title=dict(
            text=f"Timeline Segmentation Results: {domain_name.replace('_', ' ').title()}",
            x=0.5,
            font=dict(size=20),
        ),
        xaxis_title="Year",
        yaxis_title="Historical Periods",
        height=max(400, len(segments) * 80),
        showlegend=False,  # Hide legend for cleaner look
        margin=dict(t=80, b=60, l=150, r=60),
        plot_bgcolor="rgba(248,249,250,0.8)",
        font=dict(size=12),
    )

    # Enhanced x-axis with proper range
    if segments:
        all_years = [year for start, end in segments for year in [start, end]]
        min_year, max_year = min(all_years), max(all_years)
        buffer = max(5, (max_year - min_year) * 0.1)
        fig.update_xaxes(
            range=[min_year - buffer, max_year + buffer],
            showgrid=True,
            gridwidth=1,
            gridcolor="lightgray",
        )

    # Add summary annotation
    summary_text = f"Configuration: Granularity {algorithm_config.granularity} | "
    summary_text += f"Detection: {algorithm_config.direction_threshold:.2f} | "
    summary_text += f"Validation: {algorithm_config.validation_threshold:.2f}<br>"
    summary_text += (
        f"Results: {len(segments)} periods, {len(shift_signals)} paradigm shifts"
    )

    fig.add_annotation(
        xref="paper",
        yref="paper",
        x=0.02,
        y=0.98,
        text=summary_text,
        showarrow=False,
        font=dict(size=10),
        bgcolor="rgba(255,255,255,0.8)",
        bordercolor="gray",
        borderwidth=1,
    )

    return fig


def prepare_keyword_evolution_data(
    domain_data: DomainData, top_n: int = 12
) -> pd.DataFrame:
    """Prepares data for keyword evolution visualizations."""
    keyword_counts_by_year = defaultdict(Counter)
    for paper in domain_data.papers:
        if paper.pub_year and paper.keywords:
            keyword_counts_by_year[paper.pub_year].update(paper.keywords)

    if not keyword_counts_by_year:
        return pd.DataFrame()

    # Create DataFrame and get top keywords
    df_yearly = pd.DataFrame.from_dict(keyword_counts_by_year, orient="index").fillna(0)
    total_counts = df_yearly.sum().sort_values(ascending=False)
    top_keywords = total_counts.head(top_n).index

    if top_keywords.empty:
        return pd.DataFrame()

    df_top_keywords = df_yearly[top_keywords].sort_index()

    # Ensure full year range coverage
    min_year, max_year = domain_data.year_range
    full_year_range = range(min_year, max_year + 1)
    df_top_keywords = df_top_keywords.reindex(full_year_range, fill_value=0)

    # Apply smoothing
    df_smoothed = df_top_keywords.rolling(window=3, min_periods=1, center=True).mean()

    return df_smoothed


def create_keyword_streamgraph(df_keywords: pd.DataFrame, domain_name: str):
    """Creates a streamgraph of keyword evolution using a professional palette."""
    if df_keywords.empty:
        return go.Figure()

    fig = go.Figure()

    # Use a professional, colorblind-friendly palette from Plotly
    colors = px.colors.qualitative.Plotly

    for i, keyword in enumerate(df_keywords.columns):
        fig.add_trace(go.Scatter(
            x=df_keywords.index,
            y=df_keywords[keyword],
            mode='lines',
            line=dict(width=0.5, color=colors[i % len(colors)]),
            stackgroup='one',  # Key property for creating a streamgraph
            name=keyword,
            hovertemplate=f"<b>Keyword:</b> {keyword}<br>" +
                          "<b>Year:</b> %{x}<br>" +
                          "<b>Prominence:</b> %{y:.2f}<extra></extra>"
        ))

    fig.update_layout(
        title=f"Keyword Prominence Streamgraph: {domain_name.replace('_', ' ').title()}",
        xaxis_title="Year",
        yaxis_title="Smoothed Prominence",
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=500,
    )
    return fig


def create_enhanced_keyword_heatmap(
    df_keywords: pd.DataFrame, domain_name: str, shift_signals: List
):
    """Create an enhanced keyword heatmap with paradigm shift markers."""
    if df_keywords.empty:
        return go.Figure()

    fig = px.imshow(
        df_keywords.T,
        labels=dict(x="Year", y="Keyword", color="Prominence"),
        x=df_keywords.index,
        y=df_keywords.columns,
        aspect="auto",
        color_continuous_scale="Viridis",
    )

    # Add paradigm shift markers as vertical lines
    shift_years = [signal.year for signal in shift_signals]
    for year in shift_years:
        fig.add_vline(
            x=year, line=dict(color="red", width=2, dash="solid"), opacity=0.8
        )

    fig.update_layout(
        title=f"Keyword Evolution with Paradigm Shifts: {domain_name.replace('_', ' ').title()}",
        xaxis_title="Year",
        yaxis_title="Keywords",
        height=500,
        font=dict(size=12),
    )

    fig.update_xaxes(tickmode="linear", dtick=10)
    return fig


def create_decision_tree_analysis(
    domain_name: str,
    validated_signals: List,
    enhanced_signal_data: Dict,
    algorithm_config: ComprehensiveAlgorithmConfig,
    domain_data: DomainData,
):
    """Create comprehensive decision tree analysis for algorithm transparency."""
    
    clustering_metadata = enhanced_signal_data.get("clustering_metadata", {})
    clustered_direction_signals = clustering_metadata.get('clustered_direction_signals', [])
    citation_signals = clustering_metadata.get("citation_signals", [])
    
    # Extract decision details for each signal
    decision_details = []
    validated_years = {s.year for s in validated_signals}
    
    for direction_signal in clustered_direction_signals:
        year = direction_signal.year
        base_confidence = direction_signal.confidence
        
        # Check for citation support
        citation_support = any(abs(cs.year - year) <= 2 for cs in citation_signals)
        citation_years = [cs.year for cs in citation_signals if abs(cs.year - year) <= 2]
        
        # Calculate confidence boosts
        confidence_boosts = 0.0
        boost_details = []
        
        if citation_support:
            confidence_boosts += algorithm_config.citation_boost
            boost_details.append(f"Citation Support (+{algorithm_config.citation_boost:.2f})")
        
        # Final confidence calculation
        final_confidence = min(base_confidence + confidence_boosts, 1.0)
        
        # Decision outcome
        is_validated = year in validated_years
        decision_outcome = "ACCEPTED" if is_validated else "REJECTED"
        
        # Decision rationale
        if final_confidence >= algorithm_config.validation_threshold:
            if is_validated:
                rationale = f"✅ Passed threshold ({final_confidence:.3f} ≥ {algorithm_config.validation_threshold:.3f})"
            else:
                rationale = f"🔄 Should pass but not in final results - check algorithm"
        else:
            rationale = f"❌ Below threshold ({final_confidence:.3f} < {algorithm_config.validation_threshold:.3f})"
        
        decision_details.append({
            'year': year,
            'base_confidence': base_confidence,
            'citation_support': citation_support,
            'citation_years': citation_years,
            'confidence_boosts': confidence_boosts,
            'boost_details': boost_details,
            'final_confidence': final_confidence,
            'threshold': algorithm_config.validation_threshold,
            'decision_outcome': decision_outcome,
            'rationale': rationale,
            'signal_type': direction_signal.signal_type,
            'evidence_strength': direction_signal.evidence_strength,
            'supporting_evidence': list(direction_signal.supporting_evidence)[:3]
        })
    
    return decision_details


def create_decision_flow_diagram(decision_details: List[Dict], domain_name: str):
    """Create a decision flow diagram showing the algorithm's decision process."""
    
    fig = go.Figure()
    
    # Create a flowchart-style visualization
    y_positions = list(range(len(decision_details)))
    years = [d['year'] for d in decision_details]
    base_confidences = [d['base_confidence'] for d in decision_details]
    final_confidences = [d['final_confidence'] for d in decision_details]
    outcomes = [d['decision_outcome'] for d in decision_details]
    
    # Base confidence bars
    fig.add_trace(go.Bar(
        x=base_confidences,
        y=y_positions,
        orientation='h',
        name='Base Confidence',
        marker_color='lightblue',
        opacity=0.7,
        text=[f"Base: {conf:.3f}" for conf in base_confidences],
        textposition='inside',
        hovertemplate="<b>Year:</b> %{customdata}<br>" +
                      "<b>Base Confidence:</b> %{x:.3f}<extra></extra>",
        customdata=years
    ))
    
    # Final confidence bars
    fig.add_trace(go.Bar(
        x=final_confidences,
        y=y_positions,
        orientation='h',
        name='Final Confidence',
        marker_color=['green' if outcome == 'ACCEPTED' else 'red' for outcome in outcomes],
        opacity=0.8,
        text=[f"Final: {conf:.3f}" for conf in final_confidences],
        textposition='inside',
        hovertemplate="<b>Year:</b> %{customdata}<br>" +
                      "<b>Final Confidence:</b> %{x:.3f}<br>" +
                      "<b>Outcome:</b> %{text}<extra></extra>",
        customdata=years
    ))
    
    # Add threshold line
    threshold = decision_details[0]['threshold'] if decision_details else 0.8
    fig.add_vline(
        x=threshold,
        line=dict(color="red", width=3, dash="dash"),
        annotation_text=f"Validation Threshold ({threshold:.2f})"
    )
    
    fig.update_layout(
        title=f"Decision Flow Analysis: {domain_name.replace('_', ' ').title()}",
        xaxis_title="Confidence Score",
        yaxis_title="Signals (by chronological order)",
        yaxis=dict(
            tickmode='array',
            tickvals=y_positions,
            ticktext=[f"{year} ({outcome})" for year, outcome in zip(years, outcomes)]
        ),
        height=max(400, len(decision_details) * 40),
        barmode='overlay',
        showlegend=True
    )
    
    return fig


def create_parameter_sensitivity_analysis(decision_details: List[Dict], algorithm_config: ComprehensiveAlgorithmConfig):
    """Create parameter sensitivity analysis showing how changes affect outcomes."""
    
    if not decision_details:
        return go.Figure()
    
    # Test different threshold values
    test_thresholds = np.arange(0.5, 1.0, 0.05)
    sensitivity_results = []
    
    for threshold in test_thresholds:
        accepted_count = sum(1 for d in decision_details if d['final_confidence'] >= threshold)
        sensitivity_results.append({
            'threshold': threshold,
            'accepted_signals': accepted_count,
            'acceptance_rate': accepted_count / len(decision_details)
        })
    
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=("Threshold Impact on Signal Acceptance", "Boost Impact Analysis"),
        vertical_spacing=0.12
    )
    
    # Threshold sensitivity
    thresholds = [r['threshold'] for r in sensitivity_results]
    accepted_counts = [r['accepted_signals'] for r in sensitivity_results]
    
    fig.add_trace(go.Scatter(
        x=thresholds,
        y=accepted_counts,
        mode='lines+markers',
        name='Accepted Signals',
        line=dict(color='blue', width=3),
        marker=dict(size=8),
        hovertemplate="<b>Threshold:</b> %{x:.2f}<br>" +
                      "<b>Accepted Signals:</b> %{y}<extra></extra>"
    ), row=1, col=1)
    
    # Current threshold marker
    current_threshold = algorithm_config.validation_threshold
    current_accepted = sum(1 for d in decision_details if d['final_confidence'] >= current_threshold)
    
    fig.add_trace(go.Scatter(
        x=[current_threshold],
        y=[current_accepted],
        mode='markers',
        name='Current Setting',
        marker=dict(size=15, color='red', symbol='star'),
        hovertemplate="<b>Current Threshold:</b> %{x:.2f}<br>" +
                      "<b>Current Accepted:</b> %{y}<extra></extra>"
    ), row=1, col=1)
    
    # Boost impact analysis
    boost_values = np.arange(0.0, 0.6, 0.05)
    boost_impact = []
    
    for boost in boost_values:
        # Recalculate with different boost
        new_accepted = 0
        for d in decision_details:
            new_final_confidence = min(d['base_confidence'] + (boost if d['citation_support'] else 0), 1.0)
            if new_final_confidence >= current_threshold:
                new_accepted += 1
        boost_impact.append(new_accepted)
    
    fig.add_trace(go.Scatter(
        x=boost_values,
        y=boost_impact,
        mode='lines+markers',
        name='Citation Boost Impact',
        line=dict(color='green', width=3),
        marker=dict(size=6),
        hovertemplate="<b>Citation Boost:</b> %{x:.2f}<br>" +
                      "<b>Accepted Signals:</b> %{y}<extra></extra>"
    ), row=2, col=1)
    
    # Current boost marker
    current_boost = algorithm_config.citation_boost
    fig.add_trace(go.Scatter(
        x=[current_boost],
        y=[current_accepted],
        mode='markers',
        name='Current Boost',
        marker=dict(size=15, color='red', symbol='star'),
        hovertemplate="<b>Current Boost:</b> %{x:.2f}<br>" +
                      "<b>Current Accepted:</b> %{y}<extra></extra>"
    ), row=2, col=1)
    
    fig.update_layout(
        title="Parameter Sensitivity Analysis",
        height=600,
        showlegend=True
    )
    
    fig.update_xaxes(title_text="Validation Threshold", row=1, col=1)
    fig.update_xaxes(title_text="Citation Boost Value", row=2, col=1)
    fig.update_yaxes(title_text="Accepted Signals", row=1, col=1)
    fig.update_yaxes(title_text="Accepted Signals", row=2, col=1)
    
    return fig


def main():
    """Main Streamlit application."""

    st.title("🔬 Timeline Segmentation Algorithm Dashboard")

    st.sidebar.header("🎛️ Algorithm Configuration")

    # Load domain data
    domain_data_dict = load_all_domains()

    # Domain selection
    selected_domain = st.sidebar.selectbox(
        "📊 Select Domain",
        options=list(domain_data_dict.keys()),
        index=0,
        format_func=lambda x: x.replace("_", " ").title(),
    )

    # Configuration mode
    st.sidebar.subheader("⚙️ Configuration Mode")
    config_mode = st.sidebar.radio(
        "Configuration Type",
        ["🎯 Simple (Granularity)", "🔬 Advanced (All Parameters)"],
        index=0,
        help="Choose configuration complexity level"
    )

    if config_mode == "🎯 Simple (Granularity)":
        # Simple granularity interface
        st.sidebar.subheader("🎯 Granularity Level")
        granularity = st.sidebar.slider(
            "Granularity (1=Ultra-Fine, 5=Ultra-Coarse)",
            min_value=1,
            max_value=5,
            value=3,
            step=1,
            help="Primary control for segment count. Lower values = more segments.",
        )

        # Create comprehensive config from granularity
        algorithm_config = ComprehensiveAlgorithmConfig(granularity=granularity)

        # Display the derived parameters
        st.sidebar.info(
            f"""
        **Comprehensive Configuration ({len([f for f in algorithm_config.__dataclass_fields__])} parameters):**
        - Direction Threshold: {algorithm_config.direction_threshold:.2f}
        - Clustering Window: {algorithm_config.clustering_window} years
        - Validation Threshold: {algorithm_config.validation_threshold:.2f}
        - Citation Boost: {algorithm_config.citation_boost:.2f}
        - Citation Support Window: ±{algorithm_config.citation_support_window} years
        - Keyword Min Frequency: {algorithm_config.keyword_min_frequency}
        """
        )

    elif config_mode == "🔬 Advanced (All Parameters)":
        # Advanced parameter controls with comprehensive config
        st.sidebar.subheader("🔬 Comprehensive Parameter Control")
        
        granularity = st.sidebar.slider(
            "Base Granularity",
            min_value=1,
            max_value=5,
            value=3,
            step=1,
            help="Base granularity level for presets",
        )
        
        # Create expandable sections for parameter groups
        with st.sidebar.expander("🎯 Detection Parameters", expanded=True):
            direction_threshold = st.sidebar.slider(
                "Direction Threshold",
                min_value=0.1,
                max_value=0.8,
                value=0.4,
                step=0.05,
                help="Lower values = more sensitive detection",
            )
            
            direction_window_years = st.sidebar.slider(
                "Direction Window (Years)",
                min_value=1,
                max_value=10,
                value=3,
                step=1,
                help="Sliding window size for keyword evolution analysis",
            )
            
            keyword_min_frequency = st.sidebar.slider(
                "Keyword Min Frequency",
                min_value=1,
                max_value=5,
                value=2,
                step=1,
                help="Minimum frequency for keyword significance",
            )
            
            min_significant_keywords = st.sidebar.slider(
                "Min Significant Keywords",
                min_value=1,
                max_value=10,
                value=3,
                step=1,
                help="Minimum number of significant new keywords for paradigm shift",
            )

        with st.sidebar.expander("🔗 Citation Parameters"):
            citation_boost = st.sidebar.slider(
                "Citation Support Boost",
                min_value=0.1,
                max_value=0.6,
                value=0.3,
                step=0.05,
                help="Score boost for citation support",
            )
            
            citation_support_window = st.sidebar.slider(
                "Citation Support Window",
                min_value=1,
                max_value=5,
                value=2,
                step=1,
                help="Time window (±years) for citation support validation",
            )
            
            citation_gradient_multiplier = st.sidebar.slider(
                "Citation Gradient Multiplier",
                min_value=1.0,
                max_value=3.0,
                value=1.5,
                step=0.1,
                help="Multiplier for gradient threshold in citation analysis",
            )

        with st.sidebar.expander("🕒 Temporal Parameters"):
            clustering_window = st.sidebar.slider(
                "Clustering Window (Years)",
                min_value=1,
                max_value=10,
                value=3,
                step=1,
                help="Window for grouping nearby signals",
            )
            
            validation_threshold = st.sidebar.slider(
                "Validation Threshold",
                min_value=0.5,
                max_value=0.95,
                value=0.8,
                step=0.05,
                help="Consistent threshold for all signals",
            )

        # Create comprehensive config with overrides
        overrides = {
            'direction_threshold': direction_threshold,
            'direction_window_years': direction_window_years,
            'keyword_min_frequency': keyword_min_frequency,
            'min_significant_keywords': min_significant_keywords,
            'citation_boost': citation_boost,
            'citation_support_window': citation_support_window,
            'citation_gradient_multiplier': citation_gradient_multiplier,
            'clustering_window': clustering_window,
            'validation_threshold': validation_threshold,
        }
        
        algorithm_config = ComprehensiveAlgorithmConfig.create_custom(
            granularity=granularity,
            overrides=overrides
        )
    # Get domain data
    domain_data = domain_data_dict[selected_domain]

    # Run algorithm
    with st.spinner("🔄 Running algorithm..."):
        shift_signals, transition_evidence, segments, enhanced_signal_data = (
            run_algorithm_with_params(domain_data, selected_domain, algorithm_config)
        )

    # Main visualization tabs
    tab1, tab2, tab3 = st.tabs(
        [
            "🔬 Analysis Overview",
            "🔍 Keyword Evolution", 
            "🌳 Decision Tree Analysis",
        ]
    )

    with tab1:
        st.subheader("Complete Validation Process Visualization")
        validation_fig = create_comprehensive_validation_plot(
            selected_domain, shift_signals, enhanced_signal_data, algorithm_config, domain_data
        )
        st.plotly_chart(validation_fig, use_container_width=True)

        st.subheader("Timeline Segmentation Results")
        segments_fig = create_improved_segments_plot(
            selected_domain, segments, domain_data, shift_signals, algorithm_config
        )
        st.plotly_chart(segments_fig, use_container_width=True)

    with tab2:
        st.subheader("Keyword Evolution Analysis")

        # Prepare keyword data
        df_keyword_evolution = prepare_keyword_evolution_data(domain_data, top_n=12)

        if not df_keyword_evolution.empty:
            # Add the new streamgraph
            streamgraph_fig = create_keyword_streamgraph(df_keyword_evolution, selected_domain)
            st.plotly_chart(streamgraph_fig, use_container_width=True)
            
            # Enhanced keyword heatmap with paradigm shifts
            keyword_fig = create_enhanced_keyword_heatmap(
                df_keyword_evolution, selected_domain, shift_signals
            )
            st.plotly_chart(keyword_fig, use_container_width=True)

            # Keyword change analysis
            st.markdown("### 🔍 Keyword Change Analysis")

            # Extract keyword changes around paradigm shifts
            shift_years = [s.year for s in shift_signals]

            if shift_years:
                keyword_changes = []

                for signal in shift_signals:
                    year = signal.year

                    # Get keywords before and after the shift
                    before_years = [y for y in df_keyword_evolution.index if y < year][
                        -3:
                    ]
                    after_years = [y for y in df_keyword_evolution.index if y >= year][
                        :3
                    ]

                    if before_years and after_years:
                        before_keywords = (
                            df_keyword_evolution.loc[before_years]
                            .mean()
                            .sort_values(ascending=False)
                        )
                        after_keywords = (
                            df_keyword_evolution.loc[after_years]
                            .mean()
                            .sort_values(ascending=False)
                        )

                        # Find emerging and declining keywords
                        emerging = []
                        declining = []

                        for keyword in df_keyword_evolution.columns:
                            before_avg = before_keywords.get(keyword, 0)
                            after_avg = after_keywords.get(keyword, 0)

                            if after_avg > before_avg * 1.5 and after_avg > 2:
                                emerging.append(keyword)
                            elif before_avg > after_avg * 1.5 and before_avg > 2:
                                declining.append(keyword)

                        keyword_changes.append(
                            {
                                "Paradigm Shift": year,
                                "Emerging Keywords": (
                                    ", ".join(emerging[:5]) if emerging else "None"
                                ),
                                "Declining Keywords": (
                                    ", ".join(declining[:5]) if declining else "None"
                                ),
                                "Confidence": f"{signal.confidence:.3f}",
                            }
                        )

                if keyword_changes:
                    df_changes = pd.DataFrame(keyword_changes)
                    st.dataframe(df_changes, use_container_width=True, hide_index=True)
                else:
                    st.info(
                        "No significant keyword changes detected around paradigm shifts."
                    )
            else:
                st.info("No paradigm shifts detected to analyze keyword changes.")
        else:
            st.warning("Insufficient keyword data for evolution analysis.")

    with tab3:
        st.subheader("🌳 Algorithm Decision Tree Analysis")
        st.markdown("**Complete transparency into how the algorithm makes decisions**")
        
        # Generate decision analysis
        decision_details = create_decision_tree_analysis(
            selected_domain, shift_signals, enhanced_signal_data, algorithm_config, domain_data
        )
        
        if decision_details:
            # Decision Flow Diagram
            st.markdown("### 🔄 Decision Flow Visualization")
            decision_flow_fig = create_decision_flow_diagram(decision_details, selected_domain)
            st.plotly_chart(decision_flow_fig, use_container_width=True)
            
            # Parameter Sensitivity Analysis
            st.markdown("### 📊 Parameter Sensitivity Analysis")
            sensitivity_fig = create_parameter_sensitivity_analysis(decision_details, algorithm_config)
            st.plotly_chart(sensitivity_fig, use_container_width=True)
            
            # Detailed Decision Breakdown Table
            st.markdown("### 📋 Detailed Decision Breakdown")
            
            # Prepare data for table
            table_data = []
            for detail in decision_details:
                table_data.append({
                    'Year': detail['year'],
                    'Base Confidence': f"{detail['base_confidence']:.3f}",
                    'Citation Support': "✅ Yes" if detail['citation_support'] else "❌ No",
                    'Confidence Boosts': f"+{detail['confidence_boosts']:.3f}" if detail['confidence_boosts'] > 0 else "None",
                    'Final Confidence': f"{detail['final_confidence']:.3f}",
                    'Threshold': f"{detail['threshold']:.3f}",
                    'Decision': detail['decision_outcome'],
                    'Rationale': detail['rationale']
                })
            
            df_decisions = pd.DataFrame(table_data)
            
            # Color-code the table
            def color_decision(val):
                if 'ACCEPTED' in str(val):
                    return 'background-color: #d4edda'
                elif 'REJECTED' in str(val):
                    return 'background-color: #f8d7da'
                return ''
            
            # Apply styling and display
            styled_df = df_decisions.style.applymap(color_decision, subset=['Decision'])
            st.dataframe(styled_df, use_container_width=True, hide_index=True)
            
            # Interactive "What-If" Analysis
            st.markdown("### 🎯 Interactive What-If Analysis")
            st.markdown("**Explore how parameter changes would affect decisions without re-running the algorithm**")
            
            col1, col2 = st.columns(2)
            
            with col1:
                test_detection_threshold = st.slider(
                    "Test Detection Threshold",
                    min_value=0.1, max_value=0.8, value=algorithm_config.direction_threshold, step=0.05,
                    help="How would changing detection threshold affect initial signal detection?"
                )
                
                test_validation_threshold = st.slider(
                    "Test Validation Threshold", 
                    min_value=0.5, max_value=0.95, value=algorithm_config.validation_threshold, step=0.05,
                    help="How would changing validation threshold affect final acceptance?"
                )
            
            with col2:
                test_citation_boost = st.slider(
                    "Test Citation Boost",
                    min_value=0.0, max_value=0.6, value=algorithm_config.citation_boost, step=0.05,
                    help="How would changing citation boost affect confidence?"
                )
                
                test_clustering_window = st.slider(
                    "Test Clustering Window",
                    min_value=1, max_value=10, value=algorithm_config.clustering_window, step=1,
                    help="How would changing clustering window affect temporal grouping?"
                )
            
            # Calculate what-if results
            if st.button("🔄 Calculate What-If Results"):
                what_if_results = []
                
                for detail in decision_details:
                    # Recalculate with new parameters
                    new_final_confidence = min(
                        detail['base_confidence'] + (test_citation_boost if detail['citation_support'] else 0), 
                        1.0
                    )
                    new_decision = "ACCEPTED" if new_final_confidence >= test_validation_threshold else "REJECTED"
                    
                    change_indicator = ""
                    if new_decision != detail['decision_outcome']:
                        if new_decision == "ACCEPTED":
                            change_indicator = "📈 Now Accepted"
                        else:
                            change_indicator = "📉 Now Rejected"
                    else:
                        change_indicator = "➡️ No Change"
                    
                    what_if_results.append({
                        'Year': detail['year'],
                        'Original Decision': detail['decision_outcome'],
                        'New Final Confidence': f"{new_final_confidence:.3f}",
                        'New Decision': new_decision,
                        'Change': change_indicator
                    })
                
                df_what_if = pd.DataFrame(what_if_results)
                
                # Color-code changes
                def color_change(val):
                    if 'Now Accepted' in str(val):
                        return 'background-color: #d1ecf1; color: #0c5460'
                    elif 'Now Rejected' in str(val):
                        return 'background-color: #f8d7da; color: #721c24'
                    return ''
                
                styled_what_if = df_what_if.style.applymap(color_change, subset=['Change'])
                st.dataframe(styled_what_if, use_container_width=True, hide_index=True)
                
                # Summary of changes
                changes_count = sum(1 for r in what_if_results if 'No Change' not in r['Change'])
                if changes_count > 0:
                    st.info(f"💡 **Impact Summary**: {changes_count} out of {len(what_if_results)} signals would change decisions with these parameters")
                else:
                    st.success("✅ **No Impact**: These parameter changes would not affect any decisions")
            
            # Algorithm Configuration Summary
            st.markdown("### ⚙️ Current Algorithm Configuration")
            config_col1, config_col2 = st.columns(2)
            
            with config_col1:
                st.metric("Direction Threshold", f"{algorithm_config.direction_threshold:.2f}")
                st.metric("Clustering Window", f"{algorithm_config.clustering_window} years")
            
            with config_col2:
                st.metric("Validation Threshold", f"{algorithm_config.validation_threshold:.2f}")
                st.metric("Citation Boost", f"{algorithm_config.citation_boost:.2f}")
            
            # Decision Summary Statistics
            accepted_count = sum(1 for d in decision_details if d['decision_outcome'] == 'ACCEPTED')
            citation_supported = sum(1 for d in decision_details if d['citation_support'])
            
            st.markdown("### 📈 Decision Summary Statistics")
            
            metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
            
            with metric_col1:
                st.metric("Total Signals Evaluated", len(decision_details))
            
            with metric_col2:
                st.metric("Signals Accepted", accepted_count)
            
            with metric_col3:
                st.metric("Citation Supported", citation_supported)
            
            with metric_col4:
                acceptance_rate = (accepted_count / len(decision_details)) * 100 if decision_details else 0
                st.metric("Acceptance Rate", f"{acceptance_rate:.1f}%")
                
        else:
            st.info("No signals detected to analyze. Try adjusting the algorithm parameters to detect signals.")


if __name__ == "__main__":
    main()
