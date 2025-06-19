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
from core.integration import SensitivityConfig
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
    domain_data: DomainData, domain_name: str, sensitivity_config: SensitivityConfig
) -> Tuple[Any, Any, List[List[int]], Dict]:
    """Run the algorithm with specified parameters and return all signal data for visualization."""

    # Get raw direction signals data WITH analysis data for visualization
    raw_direction_signals, keyword_analysis = detect_research_direction_changes(
        domain_data, sensitivity_config.detection_threshold, return_analysis_data=True
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
        sensitivity_config=sensitivity_config,
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
    sensitivity_config: SensitivityConfig,
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
                confidence_boosts += sensitivity_config.citation_boost

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
            boost_info.append(f"Citation: +{sensitivity_config.citation_boost:.2f}")

        if boost_info:
            hover_text += f"<b>Boosts:</b> {', '.join(boost_info)}<br>"
            
        hover_text += f"<b>Final Confidence:</b> {final_confidence:.3f}<br>"
        hover_text += f"<b>Threshold:</b> {sensitivity_config.validation_threshold:.3f}<br>"
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
        y=sensitivity_config.validation_threshold,
        line=dict(color="red", width=2, dash="dash"),
        annotation_text=f"Validation Threshold ({sensitivity_config.validation_threshold:.2f})",
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
    sensitivity_config: SensitivityConfig,
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
    summary_text = f"Configuration: Granularity {sensitivity_config.granularity} | "
    summary_text += f"Detection: {sensitivity_config.detection_threshold:.2f} | "
    summary_text += f"Validation: {sensitivity_config.validation_threshold:.2f}<br>"
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
    use_advanced = st.sidebar.checkbox("🔬 Advanced Parameter Control", value=False)

    if not use_advanced:
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

        # Create sensitivity config from granularity
        sensitivity_config = SensitivityConfig(granularity=granularity)

        # Display the derived parameters
        st.sidebar.info(
            f"""
        **Derived Parameters:**
        - Detection Threshold: {sensitivity_config.detection_threshold:.2f}
        - Clustering Window: {sensitivity_config.clustering_window} years
        - Validation Threshold: {sensitivity_config.validation_threshold:.2f}
        - Citation Boost: {sensitivity_config.citation_boost:.2f}
        """
        )

    else:
        # Advanced parameter controls
        st.sidebar.subheader("🎯 Direct Parameter Control")

        detection_threshold = st.sidebar.slider(
            "Detection Threshold",
            min_value=0.1,
            max_value=0.8,
            value=0.4,
            step=0.05,
            help="Lower values = more sensitive detection",
        )

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
            value=0.9,
            step=0.05,
            help="Consistent threshold for all signals",
        )

        citation_boost = st.sidebar.slider(
            "Citation Support Boost",
            min_value=0.1,
            max_value=0.6,
            value=0.3,
            step=0.05,
            help="Score boost for citation support",
        )

        # Create custom sensitivity config
        sensitivity_config = SensitivityConfig.create_custom(
            detection_threshold=detection_threshold,
            validation_threshold=validation_threshold,
            citation_boost=citation_boost,
            clustering_window=clustering_window,
        )

    # Get domain data
    domain_data = domain_data_dict[selected_domain]

    # Run algorithm
    with st.spinner("🔄 Running algorithm..."):
        shift_signals, transition_evidence, segments, enhanced_signal_data = (
            run_algorithm_with_params(domain_data, selected_domain, sensitivity_config)
        )

    # Main visualization tabs
    tab1, tab2 = st.tabs(
        [
            "🔬 Analysis Overview",
            "🔍 Keyword Evolution",
        ]
    )

    with tab1:
        st.subheader("Complete Validation Process Visualization")
        validation_fig = create_comprehensive_validation_plot(
            selected_domain, shift_signals, enhanced_signal_data, sensitivity_config, domain_data
        )
        st.plotly_chart(validation_fig, use_container_width=True)

        st.subheader("Timeline Segmentation Results")
        segments_fig = create_improved_segments_plot(
            selected_domain, segments, domain_data, shift_signals, sensitivity_config
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


if __name__ == "__main__":
    main()
