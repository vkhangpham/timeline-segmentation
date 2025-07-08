"""Stage 2: Change Point Detection
Visualizes direction change and citation acceleration detection algorithms.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import time
from collections import Counter
from core.segmentation.change_point_detection import (
    detect_boundary_years,
    detect_direction_change_years,
    detect_citation_acceleration_years,
    validate_and_combine_signals
)


def run_change_detection():
    """Run change point detection and store results."""
    if st.session_state.academic_years is None:
        st.error("Please load data first in the Data Exploration page")
        return False
        
    if st.session_state.boundary_years is None:
        with st.spinner("Running change point detection..."):
            start_time = time.time()
            
            boundary_academic_years = detect_boundary_years(
                academic_years=st.session_state.academic_years,
                domain_name=st.session_state.selected_domain,
                algorithm_config=st.session_state.algorithm_config,
                use_citation=True,
                use_direction=True,
                verbose=False
            )
            
            detection_time = time.time() - start_time
            st.session_state.timing_data["Change Detection"] = detection_time
            st.session_state.boundary_years = boundary_academic_years
            
            st.success(f"✅ Change detection completed in {detection_time:.2f}s")
    
    return True


def calculate_direction_signals(academic_years, algorithm_config):
    """Calculate direction change signals for visualization."""
    year_keywords_map = {}
    for academic_year in academic_years:
        year_keywords_map[academic_year.year] = list(academic_year.top_keywords)
    
    years = sorted(year_keywords_map.keys())
    direction_data = []
    
    for i in range(len(years) - 1):
        year1, year2 = years[i], years[i + 1]
        keywords_prev = year_keywords_map[year1]
        keywords_curr = year_keywords_map[year2]
        
        if not keywords_prev or not keywords_curr:
            continue
            
        set_prev, set_curr = set(keywords_prev), set(keywords_curr)
        
        new_keywords = set_curr - set_prev
        shared_keywords = set_curr & set_prev
        
        novelty = len(new_keywords) / len(set_curr) if len(set_curr) > 0 else 0.0
        overlap = len(shared_keywords) / len(set_prev) if len(set_prev) > 0 else 0.0
        s_dir = novelty * (1 - overlap)
        
        is_boundary = s_dir > algorithm_config.direction_threshold
        
        direction_data.append({
            'year': year2,
            'novelty': novelty,
            'overlap': overlap,
            's_dir': s_dir,
            'is_boundary': is_boundary,
            'threshold': algorithm_config.direction_threshold,
            'new_keywords_count': len(new_keywords),
            'shared_keywords_count': len(shared_keywords),
            'total_keywords_prev': len(set_prev),
            'total_keywords_curr': len(set_curr)
        })
    
    return direction_data


def create_direction_change_chart(direction_data):
    """Create direction change visualization."""
    if not direction_data:
        return None
        
    df = pd.DataFrame(direction_data)
    
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=(
            "Direction Change Score (S_dir = novelty × (1 - overlap))",
            "Novelty Score (new keywords / total keywords)",
            "Overlap Score (shared keywords / previous keywords)"
        ),
        vertical_spacing=0.08,
        shared_xaxes=True
    )
    
    # Direction change score
    fig.add_trace(
        go.Scatter(
            x=df['year'],
            y=df['s_dir'],
            mode='lines+markers',
            name='S_dir',
            line=dict(color='#2E86AB', width=2),
            marker=dict(size=8),
            hovertemplate='<b>Year:</b> %{x}<br><b>S_dir:</b> %{y:.3f}<extra></extra>'
        ),
        row=1, col=1
    )
    
    # Threshold line
    fig.add_hline(
        y=df['threshold'].iloc[0],
        line_dash="dash",
        line_color="red",
        row=1, col=1,
        annotation_text=f"Threshold: {df['threshold'].iloc[0]}"
    )
    
    # Boundary markers
    boundary_years = df[df['is_boundary']]['year'].tolist()
    boundary_scores = df[df['is_boundary']]['s_dir'].tolist()
    
    fig.add_trace(
        go.Scatter(
            x=boundary_years,
            y=boundary_scores,
            mode='markers',
            name='Detected Boundaries',
            marker=dict(color='red', size=12, symbol='diamond'),
            hovertemplate='<b>Boundary Year:</b> %{x}<br><b>S_dir:</b> %{y:.3f}<extra></extra>'
        ),
        row=1, col=1
    )
    
    # Novelty score
    fig.add_trace(
        go.Scatter(
            x=df['year'],
            y=df['novelty'],
            mode='lines+markers',
            name='Novelty',
            line=dict(color='#A23B72', width=2),
            marker=dict(size=6),
            hovertemplate='<b>Year:</b> %{x}<br><b>Novelty:</b> %{y:.3f}<extra></extra>',
            showlegend=False
        ),
        row=2, col=1
    )
    
    # Overlap score
    fig.add_trace(
        go.Scatter(
            x=df['year'],
            y=df['overlap'],
            mode='lines+markers',
            name='Overlap',
            line=dict(color='#F18F01', width=2),
            marker=dict(size=6),
            hovertemplate='<b>Year:</b> %{x}<br><b>Overlap:</b> %{y:.3f}<extra></extra>',
            showlegend=False
        ),
        row=3, col=1
    )
    
    fig.update_layout(
        height=600,
        title_text="Direction Change Detection Analysis",
        showlegend=True
    )
    
    fig.update_xaxes(title_text="Year", row=3, col=1)
    fig.update_yaxes(title_text="S_dir Score", row=1, col=1)
    fig.update_yaxes(title_text="Novelty", row=2, col=1)
    fig.update_yaxes(title_text="Overlap", row=3, col=1)
    
    return fig


def create_citation_acceleration_chart(academic_years):
    """Create citation acceleration visualization."""
    years = [ay.year for ay in academic_years]
    citations = [ay.total_citations for ay in academic_years]
    
    # Calculate moving averages
    window_sizes = [1, 3, 5]
    
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=(
            "Citation Counts Over Time",
            "Citation Growth Rate (Year-over-Year %)"
        ),
        vertical_spacing=0.1
    )
    
    # Raw citations
    fig.add_trace(
        go.Scatter(
            x=years,
            y=citations,
            mode='lines+markers',
            name='Citations',
            line=dict(color='#1f77b4', width=2),
            marker=dict(size=8),
            hovertemplate='<b>Year:</b> %{x}<br><b>Citations:</b> %{y}<extra></extra>'
        ),
        row=1, col=1
    )
    
    # Moving averages
    colors = ['#ff7f0e', '#2ca02c', '#d62728']
    for i, window in enumerate(window_sizes):
        if len(citations) >= window:
            ma_citations = np.convolve(citations, np.ones(window)/window, mode='valid')
            ma_years = years[window-1:]
            
            fig.add_trace(
                go.Scatter(
                    x=ma_years,
                    y=ma_citations,
                    mode='lines',
                    name=f'MA-{window}',
                    line=dict(color=colors[i], width=1, dash='dash'),
                    hovertemplate=f'<b>Year:</b> %{{x}}<br><b>MA-{window}:</b> %{{y:.0f}}<extra></extra>'
                ),
                row=1, col=1
            )
    
    # Growth rate
    if len(citations) > 1:
        growth_rates = []
        growth_years = []
        for i in range(1, len(citations)):
            if citations[i-1] > 0:
                growth_rate = ((citations[i] - citations[i-1]) / citations[i-1]) * 100
                growth_rates.append(growth_rate)
                growth_years.append(years[i])
        
        if growth_rates:
            fig.add_trace(
                go.Scatter(
                    x=growth_years,
                    y=growth_rates,
                    mode='lines+markers',
                    name='Growth Rate',
                    line=dict(color='#9467bd', width=2),
                    marker=dict(size=6),
                    hovertemplate='<b>Year:</b> %{x}<br><b>Growth:</b> %{y:.1f}%<extra></extra>',
                    showlegend=False
                ),
                row=2, col=1
            )
            
            # Add zero line
            fig.add_hline(y=0, line_dash="dot", line_color="gray", row=2, col=1)
    
    fig.update_layout(
        height=500,
        title_text="Citation Acceleration Analysis"
    )
    
    fig.update_xaxes(title_text="Year")
    fig.update_yaxes(title_text="Citations", row=1, col=1)
    fig.update_yaxes(title_text="Growth Rate (%)", row=2, col=1)
    
    return fig


def show_detailed_year_analysis(direction_data, academic_years):
    """Show detailed analysis for a specific year."""
    st.subheader("🔍 Detailed Year Analysis")
    
    if not direction_data:
        st.info("No direction change data available")
        return
    
    df = pd.DataFrame(direction_data)
    years_with_data = df['year'].tolist()
    
    selected_year = st.selectbox("Select Year for Detailed Analysis", years_with_data)
    
    # Find year data
    year_data = df[df['year'] == selected_year].iloc[0]
    
    # Find corresponding academic years
    current_ay = next((ay for ay in academic_years if ay.year == selected_year), None)
    prev_ay = next((ay for ay in academic_years if ay.year == selected_year - 1), None)
    
    if current_ay and prev_ay:
        col1, col2 = st.columns(2)
        
        with col1:
            st.write(f"**Year {selected_year} Analysis:**")
            st.write(f"- S_dir Score: {year_data['s_dir']:.3f}")
            st.write(f"- Novelty: {year_data['novelty']:.3f}")
            st.write(f"- Overlap: {year_data['overlap']:.3f}")
            st.write(f"- Is Boundary: {'✅ Yes' if year_data['is_boundary'] else '❌ No'}")
            st.write(f"- Threshold: {year_data['threshold']:.3f}")
            
            st.write(f"**Keyword Counts:**")
            st.write(f"- New Keywords: {year_data['new_keywords_count']}")
            st.write(f"- Shared Keywords: {year_data['shared_keywords_count']}")
            st.write(f"- Previous Total: {year_data['total_keywords_prev']}")
            st.write(f"- Current Total: {year_data['total_keywords_curr']}")
        
        with col2:
            # Keyword comparison
            prev_keywords = set(prev_ay.top_keywords[:20])
            curr_keywords = set(current_ay.top_keywords[:20])
            
            new_keywords = curr_keywords - prev_keywords
            lost_keywords = prev_keywords - curr_keywords
            shared_keywords = curr_keywords & prev_keywords
            
            st.write(f"**Keyword Changes ({selected_year-1} → {selected_year}):**")
            
            if new_keywords:
                st.write("**New Keywords:**")
                for kw in sorted(list(new_keywords)[:10]):
                    st.write(f"+ {kw}")
            
            if lost_keywords:
                st.write("**Lost Keywords:**")
                for kw in sorted(list(lost_keywords)[:10]):
                    st.write(f"- {kw}")
            
            if shared_keywords:
                st.write("**Shared Keywords:**")
                for kw in sorted(list(shared_keywords)[:5]):
                    st.write(f"= {kw}")


def show_algorithm_explanation():
    """Show algorithm explanation with examples."""
    with st.expander("📖 Algorithm Details", expanded=False):
        st.write("""
        **Direction Change Detection Algorithm:**
        
        1. **Dual-Metric Formula:** S_dir = novelty × (1 - overlap)
           - novelty = |new_keywords| / |current_keywords|
           - overlap = |shared_keywords| / |previous_keywords|
        
        2. **Citation Acceleration Detection:**
           - Multi-scale gradient analysis on citation counts
           - Scales: [1, 3, 5] year windows
           - Adaptive thresholding based on data distribution
        
        3. **Signal Validation:**
           - Combines direction and citation signals
           - Requires validation_threshold for final boundary
           - Clusters nearby boundary years (min 3-year segments)
        """)
        
        st.write("""
        **Example Calculation:**
        - Previous year keywords: {A, B, C, D, E} (5 keywords)
        - Current year keywords: {C, D, E, F, G, H} (6 keywords)
        - New keywords: {F, G, H} (3 keywords)
        - Shared keywords: {C, D, E} (3 keywords)
        
        - novelty = 3/6 = 0.5
        - overlap = 3/5 = 0.6
        - S_dir = 0.5 × (1 - 0.6) = 0.5 × 0.4 = 0.2
        
        If direction_threshold = 0.1, then 0.2 > 0.1 → Boundary detected!
        """)


def show_change_detection():
    """Main change detection page function."""
    st.header("🔍 Stage 2: Change Point Detection")
    st.write("Analyze paradigm shift detection using direction change and citation acceleration.")
    
    # Check data availability
    if st.session_state.academic_years is None:
        st.error("Please load data first in the Data Exploration page")
        return
    
    academic_years = st.session_state.academic_years
    
    # Control panel
    col1, col2 = st.columns([3, 1])
    with col1:
        st.subheader("🎯 Detection Results")
    with col2:
        if st.button("🔄 Run Detection", type="primary"):
            st.session_state.boundary_years = None  # Force re-run
            run_change_detection()
    
    # Run detection if needed
    if not run_change_detection():
        return
    
    boundary_years = st.session_state.boundary_years
    
    # Results summary
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Boundary Years Detected", len(boundary_years))
    with col2:
        st.metric("Years Analyzed", len(academic_years))
    with col3:
        detection_rate = len(boundary_years) / len(academic_years) * 100 if academic_years else 0
        st.metric("Detection Rate", f"{detection_rate:.1f}%")
    
    # Boundary years list
    if boundary_years:
        boundary_year_list = [ay.year for ay in boundary_years]
        st.write(f"**Detected Boundary Years:** {boundary_year_list}")
    
    # Direction change analysis
    st.subheader("📊 Direction Change Analysis")
    direction_data = calculate_direction_signals(academic_years, st.session_state.algorithm_config)
    
    if direction_data:
        direction_fig = create_direction_change_chart(direction_data)
        if direction_fig:
            st.plotly_chart(direction_fig, use_container_width=True)
        
        # Statistics
        df = pd.DataFrame(direction_data)
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Avg S_dir", f"{df['s_dir'].mean():.3f}")
        with col2:
            st.metric("Max S_dir", f"{df['s_dir'].max():.3f}")
        with col3:
            st.metric("Above Threshold", len(df[df['is_boundary']]))
        with col4:
            st.metric("Threshold", f"{df['threshold'].iloc[0]:.3f}")
    
    # Citation acceleration analysis
    st.subheader("📈 Citation Acceleration Analysis")
    citation_fig = create_citation_acceleration_chart(academic_years)
    st.plotly_chart(citation_fig, use_container_width=True)
    
    # Detailed analysis tabs
    tab1, tab2 = st.tabs(["📋 Detection Summary", "🔍 Year Analysis"])
    
    with tab1:
        if direction_data:
            # Direction change summary table
            df = pd.DataFrame(direction_data)
            
            # Filter for high scores
            high_scores = df[df['s_dir'] > df['threshold'].iloc[0] * 0.7].copy()
            high_scores = high_scores.sort_values('s_dir', ascending=False)
            
            if not high_scores.empty:
                st.write("**Years with High Direction Change Scores:**")
                display_df = high_scores[['year', 's_dir', 'novelty', 'overlap', 'is_boundary']].copy()
                display_df['s_dir'] = display_df['s_dir'].round(3)
                display_df['novelty'] = display_df['novelty'].round(3)
                display_df['overlap'] = display_df['overlap'].round(3)
                
                st.dataframe(display_df, use_container_width=True, hide_index=True)
    
    with tab2:
        show_detailed_year_analysis(direction_data, academic_years)
    
    # Algorithm explanation
    show_algorithm_explanation() 