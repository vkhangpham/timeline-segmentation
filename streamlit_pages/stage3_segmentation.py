"""Stage 3: Segmentation
Visualizes conversion of boundary years into academic periods.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from collections import Counter
from core.segmentation.segmentation import create_segments_from_boundary_years


def run_segmentation():
    """Run segmentation and store results."""
    if st.session_state.boundary_years is None:
        st.error("Please run change detection first in Stage 2")
        return False
        
    if st.session_state.initial_periods is None:
        with st.spinner("Creating segments from boundary years..."):
            start_time = time.time()
            
            initial_periods = create_segments_from_boundary_years(
                boundary_academic_years=st.session_state.boundary_years,
                academic_years=tuple(st.session_state.academic_years),
                algorithm_config=st.session_state.algorithm_config,
                verbose=False
            )
            
            segmentation_time = time.time() - start_time
            st.session_state.timing_data["Segmentation"] = segmentation_time
            st.session_state.initial_periods = initial_periods
            
            st.success(f"✅ Segmentation completed in {segmentation_time:.2f}s")
    
    return True


def create_timeline_visualization(academic_years, boundary_years, periods):
    """Create visual timeline showing segmentation."""
    years = [ay.year for ay in academic_years]
    papers = [ay.paper_count for ay in academic_years]
    citations = [ay.total_citations for ay in academic_years]
    
    min_year = min(years)
    max_year = max(years)
    
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=(
            "Academic Timeline with Detected Boundaries",
            "Papers per Year by Period",
            "Citations per Year by Period"
        ),
        vertical_spacing=0.08,
        shared_xaxes=True
    )
    
    # Color palette for periods
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
    
    # Timeline with boundaries
    fig.add_trace(
        go.Scatter(
            x=years,
            y=[1] * len(years),
            mode='markers',
            name='Academic Years',
            marker=dict(
                size=8,
                color='lightblue',
                line=dict(width=1, color='darkblue')
            ),
            hovertemplate='<b>Year:</b> %{x}<extra></extra>',
            showlegend=False
        ),
        row=1, col=1
    )
    
    # Boundary markers
    boundary_year_list = [ay.year for ay in boundary_years]
    if boundary_year_list:
        fig.add_trace(
            go.Scatter(
                x=boundary_year_list,
                y=[1] * len(boundary_year_list),
                mode='markers',
                name='Boundaries',
                marker=dict(
                    size=15,
                    color='red',
                    symbol='diamond',
                    line=dict(width=2, color='darkred')
                ),
                hovertemplate='<b>Boundary Year:</b> %{x}<extra></extra>'
            ),
            row=1, col=1
        )
    
    # Period segments
    for i, period in enumerate(periods):
        color = colors[i % len(colors)]
        period_years = list(range(period.start_year, period.end_year + 1))
        
        # Papers by period
        period_papers = []
        period_citations = []
        for year in period_years:
            ay = next((ay for ay in academic_years if ay.year == year), None)
            if ay:
                period_papers.append(ay.paper_count)
                period_citations.append(ay.total_citations)
            else:
                period_papers.append(0)
                period_citations.append(0)
        
        fig.add_trace(
            go.Scatter(
                x=period_years,
                y=period_papers,
                mode='lines+markers',
                name=f'Period {i+1}',
                line=dict(color=color, width=3),
                marker=dict(size=8),
                hovertemplate=f'<b>Period {i+1}</b><br><b>Year:</b> %{{x}}<br><b>Papers:</b> %{{y}}<extra></extra>',
                showlegend=(i == 0)  # Only show legend for first trace
            ),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=period_years,
                y=period_citations,
                mode='lines+markers',
                name=f'Period {i+1}',
                line=dict(color=color, width=3),
                marker=dict(size=8),
                hovertemplate=f'<b>Period {i+1}</b><br><b>Year:</b> %{{x}}<br><b>Citations:</b> %{{y}}<extra></extra>',
                showlegend=False
            ),
            row=3, col=1
        )
        
        # Add period labels
        mid_year = (period.start_year + period.end_year) / 2
        fig.add_annotation(
            x=mid_year,
            y=1.1,
            text=f"P{i+1}",
            showarrow=False,
            font=dict(size=12, color=color),
            row=1, col=1
        )
    
    # Add vertical lines for boundaries
    for boundary_year in boundary_year_list:
        fig.add_vline(
            x=boundary_year,
            line_dash="dash",
            line_color="red",
            opacity=0.7
        )
    
    fig.update_layout(
        height=700,
        title_text="Academic Timeline Segmentation"
    )
    
    fig.update_xaxes(title_text="Year", row=3, col=1)
    fig.update_yaxes(title_text="Timeline", row=1, col=1, showticklabels=False)
    fig.update_yaxes(title_text="Papers", row=2, col=1)
    fig.update_yaxes(title_text="Citations", row=3, col=1)
    
    return fig


def create_period_summary_table(periods):
    """Create summary table of periods."""
    period_data = []
    for i, period in enumerate(periods):
        period_data.append({
            'Period': i + 1,
            'Start Year': period.start_year,
            'End Year': period.end_year,
            'Duration (Years)': period.end_year - period.start_year + 1,
            'Total Papers': period.total_papers,
            'Total Citations': period.total_citations,
            'Avg Papers/Year': round(period.total_papers / (period.end_year - period.start_year + 1), 1),
            'Avg Citations/Year': round(period.total_citations / (period.end_year - period.start_year + 1), 1),
            'Top Keywords': ', '.join(period.top_keywords[:5]) + ("..." if len(period.top_keywords) > 5 else "")
        })
    
    return pd.DataFrame(period_data)


def create_period_comparison_chart(periods):
    """Create comparison chart of period characteristics."""
    if not periods:
        return None
    
    period_names = [f"Period {i+1}\n({p.start_year}-{p.end_year})" for i, p in enumerate(periods)]
    total_papers = [p.total_papers for p in periods]
    total_citations = [p.total_citations for p in periods]
    durations = [p.end_year - p.start_year + 1 for p in periods]
    papers_per_year = [p.total_papers / (p.end_year - p.start_year + 1) for p in periods]
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Total Papers by Period",
            "Total Citations by Period", 
            "Period Duration (Years)",
            "Average Papers per Year"
        )
    )
    
    # Total papers
    fig.add_trace(
        go.Bar(
            x=period_names,
            y=total_papers,
            name='Papers',
            marker_color='lightblue',
            hovertemplate='<b>%{x}</b><br>Papers: %{y}<extra></extra>'
        ),
        row=1, col=1
    )
    
    # Total citations
    fig.add_trace(
        go.Bar(
            x=period_names,
            y=total_citations,
            name='Citations',
            marker_color='orange',
            hovertemplate='<b>%{x}</b><br>Citations: %{y}<extra></extra>'
        ),
        row=1, col=2
    )
    
    # Duration
    fig.add_trace(
        go.Bar(
            x=period_names,
            y=durations,
            name='Duration',
            marker_color='lightgreen',
            hovertemplate='<b>%{x}</b><br>Duration: %{y} years<extra></extra>'
        ),
        row=2, col=1
    )
    
    # Papers per year
    fig.add_trace(
        go.Bar(
            x=period_names,
            y=papers_per_year,
            name='Papers/Year',
            marker_color='lightcoral',
            hovertemplate='<b>%{x}</b><br>Papers/Year: %{y:.1f}<extra></extra>'
        ),
        row=2, col=2
    )
    
    fig.update_layout(
        height=500,
        title_text="Period Characteristics Comparison",
        showlegend=False
    )
    
    return fig


def create_keyword_distribution_chart(periods):
    """Create keyword distribution visualization."""
    if not periods:
        return None
    
    # Collect top keywords from each period
    period_keywords = {}
    for i, period in enumerate(periods):
        period_keywords[f"Period {i+1}"] = dict(list(period.combined_keyword_frequencies.items())[:15])
    
    # Create stacked bar chart
    all_keywords = set()
    for period_kws in period_keywords.values():
        all_keywords.update(period_kws.keys())
    
    # Limit to most frequent keywords across all periods
    keyword_totals = Counter()
    for period_kws in period_keywords.values():
        keyword_totals.update(period_kws)
    
    top_keywords = [kw for kw, count in keyword_totals.most_common(20)]
    
    fig = go.Figure()
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
    
    for i, (period_name, period_kws) in enumerate(period_keywords.items()):
        keyword_counts = [period_kws.get(kw, 0) for kw in top_keywords]
        
        fig.add_trace(
            go.Bar(
                x=top_keywords,
                y=keyword_counts,
                name=period_name,
                marker_color=colors[i % len(colors)],
                hovertemplate=f'<b>{period_name}</b><br><b>Keyword:</b> %{{x}}<br><b>Count:</b> %{{y}}<extra></extra>'
            )
        )
    
    fig.update_layout(
        title="Keyword Distribution Across Periods",
        xaxis_title="Keywords",
        yaxis_title="Frequency",
        barmode='group',
        height=400,
        xaxis_tickangle=-45
    )
    
    return fig


def show_period_details(periods):
    """Show detailed information for selected period."""
    st.subheader("🔍 Period Details")
    
    if not periods:
        st.info("No periods available")
        return
    
    period_options = [f"Period {i+1} ({p.start_year}-{p.end_year})" for i, p in enumerate(periods)]
    selected_period_idx = st.selectbox("Select Period for Details", range(len(periods)), format_func=lambda x: period_options[x])
    
    selected_period = periods[selected_period_idx]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write(f"**Period {selected_period_idx + 1} Overview:**")
        st.write(f"- **Time Range:** {selected_period.start_year} - {selected_period.end_year}")
        st.write(f"- **Duration:** {selected_period.end_year - selected_period.start_year + 1} years")
        st.write(f"- **Academic Years:** {len(selected_period.academic_years)}")
        st.write(f"- **Total Papers:** {selected_period.total_papers:,}")
        st.write(f"- **Total Citations:** {selected_period.total_citations:,}")
        st.write(f"- **Avg Papers/Year:** {selected_period.total_papers / (selected_period.end_year - selected_period.start_year + 1):.1f}")
        st.write(f"- **Unique Keywords:** {len(selected_period.combined_keyword_frequencies)}")
    
    with col2:
        st.write("**Top 15 Keywords:**")
        for i, keyword in enumerate(selected_period.top_keywords[:15], 1):
            freq = selected_period.combined_keyword_frequencies.get(keyword, 0)
            st.write(f"{i:2d}. {keyword} ({freq})")
    
    # Year-by-year breakdown
    st.write("**Year-by-Year Breakdown:**")
    year_data = []
    for ay in selected_period.academic_years:
        year_data.append({
            'Year': ay.year,
            'Papers': ay.paper_count,
            'Citations': ay.total_citations,
            'Top Keywords': ', '.join(ay.top_keywords[:5])
        })
    
    year_df = pd.DataFrame(year_data)
    st.dataframe(year_df, use_container_width=True, hide_index=True)


def show_segmentation_quality_metrics(periods):
    """Show quality metrics for the segmentation."""
    st.subheader("📊 Segmentation Quality Metrics")
    
    if not periods:
        st.info("No periods available for quality analysis")
        return
    
    # Calculate metrics
    total_papers = sum(p.total_papers for p in periods)
    total_citations = sum(p.total_citations for p in periods)
    
    durations = [p.end_year - p.start_year + 1 for p in periods]
    avg_duration = sum(durations) / len(durations)
    duration_std = (sum((d - avg_duration) ** 2 for d in durations) / len(durations)) ** 0.5
    
    paper_distributions = [p.total_papers / total_papers for p in periods if total_papers > 0]
    paper_balance = 1 - max(paper_distributions) if paper_distributions else 0
    
    # Display metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Number of Periods", len(periods))
    with col2:
        st.metric("Avg Period Duration", f"{avg_duration:.1f} years")
    with col3:
        st.metric("Duration Consistency", f"{1/(1+duration_std):.2f}")
    with col4:
        st.metric("Paper Distribution Balance", f"{paper_balance:.2f}")
    
    # Period size distribution
    sizes = [p.total_papers for p in periods]
    
    fig = go.Figure()
    fig.add_trace(
        go.Box(
            y=sizes,
            name="Period Sizes",
            boxpoints="all",
            jitter=0.3,
            pointpos=-1.8,
            marker_color='lightblue'
        )
    )
    
    fig.update_layout(
        title="Distribution of Period Sizes (Papers)",
        yaxis_title="Number of Papers",
        height=300
    )
    
    st.plotly_chart(fig, use_container_width=True)


def show_segmentation():
    """Main segmentation page function."""
    st.header("✂️ Stage 3: Segmentation")
    st.write("Convert boundary years into contiguous academic periods.")
    
    # Check prerequisites
    if st.session_state.academic_years is None:
        st.error("Please load data first in the Data Exploration page")
        return
    
    if st.session_state.boundary_years is None:
        st.error("Please run change detection first in Stage 2")
        return
    
    # Control panel
    col1, col2 = st.columns([3, 1])
    with col1:
        st.subheader("🎯 Segmentation Results")
    with col2:
        if st.button("🔄 Run Segmentation", type="primary"):
            st.session_state.initial_periods = None  # Force re-run
            run_segmentation()
    
    # Run segmentation if needed
    if not run_segmentation():
        return
    
    periods = st.session_state.initial_periods
    boundary_years = st.session_state.boundary_years
    academic_years = st.session_state.academic_years
    
    # Results summary
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Periods Created", len(periods))
    with col2:
        st.metric("Boundary Years Used", len(boundary_years))
    with col3:
        avg_duration = sum(p.end_year - p.start_year + 1 for p in periods) / len(periods) if periods else 0
        st.metric("Avg Period Duration", f"{avg_duration:.1f} years")
    
    # Timeline visualization
    st.subheader("📈 Timeline Visualization")
    timeline_fig = create_timeline_visualization(academic_years, boundary_years, periods)
    st.plotly_chart(timeline_fig, use_container_width=True)
    
    # Period summary
    st.subheader("📋 Period Summary")
    if periods:
        summary_df = create_period_summary_table(periods)
        st.dataframe(summary_df, use_container_width=True, hide_index=True)
    
    # Comparison charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Period Comparison")
        comparison_fig = create_period_comparison_chart(periods)
        if comparison_fig:
            st.plotly_chart(comparison_fig, use_container_width=True)
    
    with col2:
        st.subheader("🏷️ Keyword Distribution")
        keyword_fig = create_keyword_distribution_chart(periods)
        if keyword_fig:
            st.plotly_chart(keyword_fig, use_container_width=True)
    
    # Detailed analysis tabs
    tab1, tab2, tab3 = st.tabs(["🔍 Period Details", "📊 Quality Metrics", "📖 Algorithm Info"])
    
    with tab1:
        show_period_details(periods)
    
    with tab2:
        show_segmentation_quality_metrics(periods)
    
    with tab3:
        st.write("""
        **Segmentation Algorithm:**
        
        1. **Input Processing:** Takes boundary years from change detection
        2. **Segment Creation:** Creates contiguous periods between boundaries
        3. **Academic Year Aggregation:** Combines academic years within each period
        4. **Keyword Consolidation:** Merges keyword frequencies across years
        5. **Validation:** Ensures no gaps or overlaps between periods
        
        **Key Features:**
        - Contiguous periods with no gaps
        - Automatic handling of edge cases
        - Pre-computed aggregations for efficiency
        - Immutable data structures for reliability
        """)
        
        if periods:
            boundary_year_list = [ay.year for ay in boundary_years]
            st.write(f"**Detected Boundaries:** {boundary_year_list}")
            st.write(f"**Resulting Periods:**")
            for i, period in enumerate(periods):
                st.write(f"- Period {i+1}: {period.start_year}-{period.end_year} ({period.end_year - period.start_year + 1} years)")
    
    # Data export
    st.subheader("💾 Export Results")
    if st.button("Export Segmentation Results"):
        segmentation_data = {
            "domain": st.session_state.selected_domain,
            "boundary_years": [ay.year for ay in boundary_years],
            "periods": [
                {
                    "period_id": i + 1,
                    "start_year": p.start_year,
                    "end_year": p.end_year,
                    "duration": p.end_year - p.start_year + 1,
                    "total_papers": p.total_papers,
                    "total_citations": p.total_citations,
                    "top_keywords": list(p.top_keywords[:10])
                }
                for i, p in enumerate(periods)
            ]
        }
        
        st.download_button(
            label="Download Segmentation JSON",
            data=pd.Series(segmentation_data).to_json(indent=2),
            file_name=f"{st.session_state.selected_domain}_segmentation_results.json",
            mime="application/json"
        ) 