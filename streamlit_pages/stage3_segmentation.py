"""Stage 3: Segmentation & Refinement
Visualizes conversion of boundary years into academic periods and beam search refinement.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from collections import Counter
from core.segmentation.segmentation import create_segments_from_boundary_years
from core.segmentation.beam_refinement import beam_search_refinement


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
                verbose=False,
            )

            segmentation_time = time.time() - start_time
            st.session_state.timing_data["Segmentation"] = segmentation_time
            st.session_state.initial_periods = initial_periods

            st.success(f"✅ Initial segmentation completed in {segmentation_time:.2f}s")

    return True


def run_beam_refinement():
    """Run beam search refinement and store results."""
    if st.session_state.initial_periods is None:
        st.error("Please run segmentation first")
        return False

    if st.session_state.refined_periods is None:
        with st.spinner("Running beam search refinement..."):
            start_time = time.time()

            refined_periods = beam_search_refinement(
                initial_periods=st.session_state.initial_periods,
                academic_years=tuple(st.session_state.academic_years),
                algorithm_config=st.session_state.algorithm_config,
                verbose=False,
            )

            refinement_time = time.time() - start_time
            st.session_state.timing_data["Beam Refinement"] = refinement_time
            st.session_state.refined_periods = refined_periods

            st.success(f"✅ Beam search refinement completed in {refinement_time:.2f}s")

    return True


def create_timeline_visualization(academic_years, boundary_years, periods):
    """Create visual timeline showing segmentation."""
    years = [ay.year for ay in academic_years]
    papers = [ay.paper_count for ay in academic_years]
    citations = [ay.total_citations for ay in academic_years]

    min_year = min(years)
    max_year = max(years)

    fig = make_subplots(
        rows=3,
        cols=1,
        subplot_titles=(
            "Academic Timeline with Detected Boundaries",
            "Papers per Year by Period",
            "Citations per Year by Period",
        ),
        vertical_spacing=0.08,
        shared_xaxes=True,
    )

    # Color palette for periods
    colors = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
    ]

    # Timeline with boundaries
    fig.add_trace(
        go.Scatter(
            x=years,
            y=[1] * len(years),
            mode="markers",
            name="Academic Years",
            marker=dict(
                size=8, color="lightblue", line=dict(width=1, color="darkblue")
            ),
            hovertemplate="<b>Year:</b> %{x}<extra></extra>",
            showlegend=False,
        ),
        row=1,
        col=1,
    )

    # Boundary markers
    boundary_year_list = [ay.year for ay in boundary_years]
    if boundary_year_list:
        fig.add_trace(
            go.Scatter(
                x=boundary_year_list,
                y=[1] * len(boundary_year_list),
                mode="markers",
                name="Boundaries",
                marker=dict(
                    size=15,
                    color="red",
                    symbol="diamond",
                    line=dict(width=2, color="darkred"),
                ),
                hovertemplate="<b>Boundary Year:</b> %{x}<extra></extra>",
            ),
            row=1,
            col=1,
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
                mode="lines+markers",
                name=f"Period {i+1}",
                line=dict(color=color, width=3),
                marker=dict(size=8),
                hovertemplate=f"<b>Period {i+1}</b><br><b>Year:</b> %{{x}}<br><b>Papers:</b> %{{y}}<extra></extra>",
                showlegend=(i == 0),  # Only show legend for first trace
            ),
            row=2,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=period_years,
                y=period_citations,
                mode="lines+markers",
                name=f"Period {i+1}",
                line=dict(color=color, width=3),
                marker=dict(size=8),
                hovertemplate=f"<b>Period {i+1}</b><br><b>Year:</b> %{{x}}<br><b>Citations:</b> %{{y}}<extra></extra>",
                showlegend=False,
            ),
            row=3,
            col=1,
        )

        # Add period labels
        mid_year = (period.start_year + period.end_year) / 2
        fig.add_annotation(
            x=mid_year,
            y=1.1,
            text=f"P{i+1}",
            showarrow=False,
            font=dict(size=12, color=color),
            row=1,
            col=1,
        )

    # Add vertical lines for boundaries
    for boundary_year in boundary_year_list:
        fig.add_vline(x=boundary_year, line_dash="dash", line_color="red", opacity=0.7)

    fig.update_layout(height=700, title_text="Academic Timeline Segmentation")

    fig.update_xaxes(title_text="Year", row=3, col=1)
    fig.update_yaxes(title_text="Timeline", row=1, col=1, showticklabels=False)
    fig.update_yaxes(title_text="Papers", row=2, col=1)
    fig.update_yaxes(title_text="Citations", row=3, col=1)

    return fig


def create_period_summary_table(periods):
    """Create summary table of periods."""
    period_data = []
    for i, period in enumerate(periods):
        period_data.append(
            {
                "Period": i + 1,
                "Start Year": period.start_year,
                "End Year": period.end_year,
                "Duration (Years)": period.end_year - period.start_year + 1,
                "Total Papers": period.total_papers,
                "Total Citations": period.total_citations,
                "Avg Papers/Year": round(
                    period.total_papers / (period.end_year - period.start_year + 1), 1
                ),
                "Avg Citations/Year": round(
                    period.total_citations / (period.end_year - period.start_year + 1),
                    1,
                ),
                "Top Keywords": ", ".join(period.top_keywords[:5])
                + ("..." if len(period.top_keywords) > 5 else ""),
            }
        )

    return pd.DataFrame(period_data)


def create_period_comparison_chart(periods):
    """Create comparison chart of period characteristics."""
    if not periods:
        return None

    period_names = [
        f"Period {i+1}\n({p.start_year}-{p.end_year})" for i, p in enumerate(periods)
    ]
    total_papers = [p.total_papers for p in periods]
    total_citations = [p.total_citations for p in periods]
    durations = [p.end_year - p.start_year + 1 for p in periods]
    papers_per_year = [
        p.total_papers / (p.end_year - p.start_year + 1) for p in periods
    ]

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Total Papers by Period",
            "Total Citations by Period",
            "Period Duration (Years)",
            "Average Papers per Year",
        ),
    )

    # Total papers
    fig.add_trace(
        go.Bar(
            x=period_names,
            y=total_papers,
            name="Papers",
            marker_color="lightblue",
            hovertemplate="<b>%{x}</b><br>Papers: %{y}<extra></extra>",
        ),
        row=1,
        col=1,
    )

    # Total citations
    fig.add_trace(
        go.Bar(
            x=period_names,
            y=total_citations,
            name="Citations",
            marker_color="orange",
            hovertemplate="<b>%{x}</b><br>Citations: %{y}<extra></extra>",
        ),
        row=1,
        col=2,
    )

    # Duration
    fig.add_trace(
        go.Bar(
            x=period_names,
            y=durations,
            name="Duration",
            marker_color="lightgreen",
            hovertemplate="<b>%{x}</b><br>Duration: %{y} years<extra></extra>",
        ),
        row=2,
        col=1,
    )

    # Papers per year
    fig.add_trace(
        go.Bar(
            x=period_names,
            y=papers_per_year,
            name="Papers/Year",
            marker_color="lightcoral",
            hovertemplate="<b>%{x}</b><br>Papers/Year: %{y:.1f}<extra></extra>",
        ),
        row=2,
        col=2,
    )

    fig.update_layout(
        height=500, title_text="Period Characteristics Comparison", showlegend=False
    )

    return fig


def create_keyword_distribution_chart(periods):
    """Create keyword distribution visualization."""
    if not periods:
        return None

    # Collect top keywords from each period
    period_keywords = {}
    for i, period in enumerate(periods):
        period_keywords[f"Period {i+1}"] = dict(
            list(period.combined_keyword_frequencies.items())[:15]
        )

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

    colors = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
    ]

    for i, (period_name, period_kws) in enumerate(period_keywords.items()):
        keyword_counts = [period_kws.get(kw, 0) for kw in top_keywords]

        fig.add_trace(
            go.Bar(
                x=top_keywords,
                y=keyword_counts,
                name=period_name,
                marker_color=colors[i % len(colors)],
                hovertemplate=f"<b>{period_name}</b><br><b>Keyword:</b> %{{x}}<br><b>Count:</b> %{{y}}<extra></extra>",
            )
        )

    fig.update_layout(
        title="Keyword Distribution Across Periods",
        xaxis_title="Keywords",
        yaxis_title="Frequency",
        barmode="group",
        height=400,
        xaxis_tickangle=-45,
    )

    return fig


def show_period_details(periods):
    """Show detailed information for selected period."""
    st.subheader("🔍 Period Details")

    if not periods:
        st.info("No periods available")
        return

    period_options = [
        f"Period {i+1} ({p.start_year}-{p.end_year})" for i, p in enumerate(periods)
    ]
    selected_period_idx = st.selectbox(
        "Select Period for Details",
        range(len(periods)),
        format_func=lambda x: period_options[x],
    )

    selected_period = periods[selected_period_idx]

    col1, col2 = st.columns(2)

    with col1:
        st.write(f"**Period {selected_period_idx + 1} Overview:**")
        st.write(
            f"- **Time Range:** {selected_period.start_year} - {selected_period.end_year}"
        )
        st.write(
            f"- **Duration:** {selected_period.end_year - selected_period.start_year + 1} years"
        )
        st.write(f"- **Academic Years:** {len(selected_period.academic_years)}")
        st.write(f"- **Total Papers:** {selected_period.total_papers:,}")
        st.write(f"- **Total Citations:** {selected_period.total_citations:,}")
        st.write(
            f"- **Avg Papers/Year:** {selected_period.total_papers / (selected_period.end_year - selected_period.start_year + 1):.1f}"
        )
        st.write(
            f"- **Unique Keywords:** {len(selected_period.combined_keyword_frequencies)}"
        )

    with col2:
        st.write("**Top 15 Keywords:**")
        for i, keyword in enumerate(selected_period.top_keywords[:15], 1):
            freq = selected_period.combined_keyword_frequencies.get(keyword, 0)
            st.write(f"{i:2d}. {keyword} ({freq})")

    # Year-by-year breakdown
    st.write("**Year-by-Year Breakdown:**")
    year_data = []
    for ay in selected_period.academic_years:
        year_data.append(
            {
                "Year": ay.year,
                "Papers": ay.paper_count,
                "Citations": ay.total_citations,
                "Top Keywords": ", ".join(ay.top_keywords[:5]),
            }
        )

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
    duration_std = (
        sum((d - avg_duration) ** 2 for d in durations) / len(durations)
    ) ** 0.5

    paper_distributions = [
        p.total_papers / total_papers for p in periods if total_papers > 0
    ]
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
            marker_color="lightblue",
        )
    )

    fig.update_layout(
        title="Distribution of Period Sizes (Papers)",
        yaxis_title="Number of Papers",
        height=300,
    )

    st.plotly_chart(fig, use_container_width=True)


def create_refinement_comparison_chart(initial_periods, refined_periods):
    """Create before/after comparison of beam search refinement."""
    if not initial_periods or not refined_periods:
        return None

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Initial: Period Timeline",
            "Refined: Period Timeline",
            "Initial: Period Metrics",
            "Refined: Period Metrics",
        ),
        specs=[
            [{"secondary_y": False}, {"secondary_y": False}],
            [{"secondary_y": False}, {"secondary_y": False}],
        ],
        vertical_spacing=0.15,
    )

    colors = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
    ]

    # Initial periods timeline
    for i, period in enumerate(initial_periods):
        color = colors[i % len(colors)]
        fig.add_trace(
            go.Scatter(
                x=[period.start_year, period.end_year],
                y=[i, i],
                mode="lines+markers",
                name=f"Initial P{i+1}",
                line=dict(color=color, width=6),
                marker=dict(size=10),
                hovertemplate=f"<b>Initial Period {i+1}</b><br>"
                + f"Years: {period.start_year}-{period.end_year}<br>"
                + f"Papers: {period.total_papers}<extra></extra>",
                showlegend=False,
            ),
            row=1,
            col=1,
        )

    # Refined periods timeline
    for i, period in enumerate(refined_periods):
        color = colors[i % len(colors)]
        fig.add_trace(
            go.Scatter(
                x=[period.start_year, period.end_year],
                y=[i, i],
                mode="lines+markers",
                name=f"Refined P{i+1}",
                line=dict(color=color, width=6),
                marker=dict(size=10),
                hovertemplate=f"<b>Refined Period {i+1}</b><br>"
                + f"Years: {period.start_year}-{period.end_year}<br>"
                + f"Papers: {period.total_papers}<extra></extra>",
                showlegend=False,
            ),
            row=1,
            col=2,
        )

    # Initial periods metrics
    initial_papers = [p.total_papers for p in initial_periods]
    initial_durations = [p.end_year - p.start_year + 1 for p in initial_periods]

    fig.add_trace(
        go.Bar(
            x=[f"P{i+1}" for i in range(len(initial_periods))],
            y=initial_papers,
            name="Initial Papers",
            marker_color="lightblue",
            hovertemplate="<b>%{x}</b><br>Papers: %{y}<extra></extra>",
            showlegend=False,
        ),
        row=2,
        col=1,
    )

    # Refined periods metrics
    refined_papers = [p.total_papers for p in refined_periods]
    refined_durations = [p.end_year - p.start_year + 1 for p in refined_periods]

    fig.add_trace(
        go.Bar(
            x=[f"P{i+1}" for i in range(len(refined_periods))],
            y=refined_papers,
            name="Refined Papers",
            marker_color="lightcoral",
            hovertemplate="<b>%{x}</b><br>Papers: %{y}<extra></extra>",
            showlegend=False,
        ),
        row=2,
        col=2,
    )

    fig.update_layout(
        height=600,
        title_text="Beam Search Refinement: Before vs After Comparison",
        showlegend=False,
    )

    fig.update_xaxes(title_text="Year", row=1, col=1)
    fig.update_xaxes(title_text="Year", row=1, col=2)
    fig.update_xaxes(title_text="Period", row=2, col=1)
    fig.update_xaxes(title_text="Period", row=2, col=2)

    fig.update_yaxes(title_text="Period Index", row=1, col=1, showticklabels=False)
    fig.update_yaxes(title_text="Period Index", row=1, col=2, showticklabels=False)
    fig.update_yaxes(title_text="Papers", row=2, col=1)
    fig.update_yaxes(title_text="Papers", row=2, col=2)

    return fig


def show_refinement_analysis(initial_periods, refined_periods):
    """Show detailed analysis of beam search refinement."""
    st.subheader("🔍 Beam Search Refinement Analysis")

    # Configuration summary
    config = st.session_state.algorithm_config
    col1, col2 = st.columns(2)

    with col1:
        st.write("**Beam Search Configuration:**")
        st.write(f"- Enabled: {config.beam_search_enabled}")
        st.write(f"- Beam Width: {config.beam_width}")
        st.write(f"- Max Splits per Segment: {config.max_splits_per_segment}")
        st.write(f"- Min Period Years: {config.min_period_years}")
        st.write(f"- Max Period Years: {config.max_period_years}")

    with col2:
        st.write("**Refinement Results:**")
        period_change = len(refined_periods) - len(initial_periods)
        st.write(f"- Initial Periods: {len(initial_periods)}")
        st.write(f"- Refined Periods: {len(refined_periods)}")
        st.write(f"- Period Change: {period_change:+d}")

        if "Beam Refinement" in st.session_state.timing_data:
            st.write(
                f"- Processing Time: {st.session_state.timing_data['Beam Refinement']:.2f}s"
            )

    # Show specific changes
    if period_change != 0:
        st.write("**Changes Made:**")
        if period_change > 0:
            st.write(
                f"✂️ **Splits:** {period_change} period(s) were split to improve segmentation quality"
            )
        else:
            st.write(
                f"🔗 **Merges:** {abs(period_change)} period(s) were merged to improve segmentation quality"
            )

        # Try to identify specific changes
        if period_change > 0:
            st.write("**Split Analysis:**")
            # Look for periods that might have been split
            for i, initial_period in enumerate(initial_periods):
                # Find refined periods that fall within this initial period
                overlapping_refined = []
                for j, refined_period in enumerate(refined_periods):
                    if (
                        refined_period.start_year >= initial_period.start_year
                        and refined_period.end_year <= initial_period.end_year
                    ):
                        overlapping_refined.append(j + 1)

                if len(overlapping_refined) > 1:
                    st.write(
                        f"- Initial Period {i+1} ({initial_period.start_year}-{initial_period.end_year}) → Refined Periods {overlapping_refined}"
                    )
    else:
        st.write(
            "**No Changes:** Beam search determined the initial segmentation was optimal"
        )


def create_refinement_metrics_chart(initial_periods, refined_periods):
    """Create metrics comparison chart for refinement."""
    if not initial_periods or not refined_periods:
        return None

    # Calculate metrics
    initial_durations = [p.end_year - p.start_year + 1 for p in initial_periods]
    refined_durations = [p.end_year - p.start_year + 1 for p in refined_periods]

    initial_papers = [p.total_papers for p in initial_periods]
    refined_papers = [p.total_papers for p in refined_periods]

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Period Duration Distribution",
            "Paper Count Distribution",
            "Duration Statistics",
            "Paper Statistics",
        ),
    )

    # Duration distributions
    fig.add_trace(
        go.Histogram(
            x=initial_durations,
            name="Initial",
            marker_color="lightblue",
            opacity=0.7,
            nbinsx=min(10, len(initial_durations)),
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Histogram(
            x=refined_durations,
            name="Refined",
            marker_color="lightcoral",
            opacity=0.7,
            nbinsx=min(10, len(refined_durations)),
        ),
        row=1,
        col=1,
    )

    # Paper distributions
    fig.add_trace(
        go.Histogram(
            x=initial_papers,
            name="Initial",
            marker_color="lightblue",
            opacity=0.7,
            nbinsx=min(10, len(initial_papers)),
            showlegend=False,
        ),
        row=1,
        col=2,
    )

    fig.add_trace(
        go.Histogram(
            x=refined_papers,
            name="Refined",
            marker_color="lightcoral",
            opacity=0.7,
            nbinsx=min(10, len(refined_papers)),
            showlegend=False,
        ),
        row=1,
        col=2,
    )

    # Duration box plots
    fig.add_trace(
        go.Box(
            y=initial_durations,
            name="Initial",
            marker_color="lightblue",
            boxpoints="all",
            jitter=0.3,
            showlegend=False,
        ),
        row=2,
        col=1,
    )

    fig.add_trace(
        go.Box(
            y=refined_durations,
            name="Refined",
            marker_color="lightcoral",
            boxpoints="all",
            jitter=0.3,
            showlegend=False,
        ),
        row=2,
        col=1,
    )

    # Paper box plots
    fig.add_trace(
        go.Box(
            y=initial_papers,
            name="Initial",
            marker_color="lightblue",
            boxpoints="all",
            jitter=0.3,
            showlegend=False,
        ),
        row=2,
        col=2,
    )

    fig.add_trace(
        go.Box(
            y=refined_papers,
            name="Refined",
            marker_color="lightcoral",
            boxpoints="all",
            jitter=0.3,
            showlegend=False,
        ),
        row=2,
        col=2,
    )

    fig.update_layout(
        height=600,
        title_text="Refinement Quality Metrics",
        barmode="overlay",
    )

    fig.update_xaxes(title_text="Duration (Years)", row=1, col=1)
    fig.update_xaxes(title_text="Papers", row=1, col=2)
    fig.update_xaxes(title_text="Segmentation", row=2, col=1)
    fig.update_xaxes(title_text="Segmentation", row=2, col=2)

    fig.update_yaxes(title_text="Count", row=1, col=1)
    fig.update_yaxes(title_text="Count", row=1, col=2)
    fig.update_yaxes(title_text="Duration (Years)", row=2, col=1)
    fig.update_yaxes(title_text="Papers", row=2, col=2)

    return fig


def show_segmentation():
    """Main segmentation page function."""
    st.header("✂️ Stage 3: Segmentation & Refinement")
    st.write(
        "Convert boundary years into academic periods and refine with beam search optimization."
    )

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
        st.subheader("🎯 Segmentation & Refinement Pipeline")

        # Show current status
        if st.session_state.initial_periods is None:
            st.info("📋 Ready to run initial segmentation")
        elif st.session_state.refined_periods is None:
            st.info(
                "📋 Initial segmentation complete. Ready for beam search refinement."
            )
        else:
            st.success("✅ Both segmentation and refinement completed!")

    with col2:
        # Two-step process
        if st.button("🔄 Run Initial Segmentation", type="primary"):
            st.session_state.initial_periods = None  # Force re-run
            st.session_state.refined_periods = None
            st.session_state.characterized_periods = None
            st.session_state.final_periods = None
            run_segmentation()

        if st.session_state.initial_periods is not None:
            if st.button("🔍 Run Beam Refinement", type="secondary"):
                st.session_state.refined_periods = None  # Force re-run
                st.session_state.characterized_periods = None
                st.session_state.final_periods = None
                run_beam_refinement()

    # Step 1: Run initial segmentation
    if not run_segmentation():
        return

    initial_periods = st.session_state.initial_periods
    boundary_years = st.session_state.boundary_years
    academic_years = st.session_state.academic_years

    # Step 2: Run beam search refinement
    if not run_beam_refinement():
        return

    refined_periods = st.session_state.refined_periods

    # Results summary
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Boundary Years", len(boundary_years))
    with col2:
        st.metric("Initial Periods", len(initial_periods))
    with col3:
        st.metric("Refined Periods", len(refined_periods))
    with col4:
        period_change = len(refined_periods) - len(initial_periods)
        st.metric("Period Change", f"{period_change:+d}")

    # Show refinement comparison if both are available
    if initial_periods and refined_periods:
        st.subheader("📊 Beam Search Refinement Comparison")
        comparison_fig = create_refinement_comparison_chart(
            initial_periods, refined_periods
        )
        if comparison_fig:
            st.plotly_chart(comparison_fig, use_container_width=True)

    # Timeline visualization (using refined periods)
    st.subheader("📈 Final Timeline Visualization")
    timeline_fig = create_timeline_visualization(
        academic_years, boundary_years, refined_periods
    )
    st.plotly_chart(timeline_fig, use_container_width=True)

    # Period summary (using refined periods)
    st.subheader("📋 Final Period Summary")
    if refined_periods:
        summary_df = create_period_summary_table(refined_periods)
        st.dataframe(summary_df, use_container_width=True, hide_index=True)

    # Comparison charts
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📊 Period Comparison")
        comparison_fig = create_period_comparison_chart(refined_periods)
        if comparison_fig:
            st.plotly_chart(comparison_fig, use_container_width=True)

    with col2:
        st.subheader("🏷️ Keyword Distribution")
        keyword_fig = create_keyword_distribution_chart(refined_periods)
        if keyword_fig:
            st.plotly_chart(keyword_fig, use_container_width=True)

    # Detailed analysis tabs
    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "🔍 Refinement Analysis",
            "📋 Period Details",
            "📊 Quality Metrics",
            "📖 Algorithm Info",
        ]
    )

    with tab1:
        if initial_periods and refined_periods:
            show_refinement_analysis(initial_periods, refined_periods)

            # Refinement metrics
            st.subheader("📊 Refinement Quality Metrics")
            metrics_fig = create_refinement_metrics_chart(
                initial_periods, refined_periods
            )
            if metrics_fig:
                st.plotly_chart(metrics_fig, use_container_width=True)
        else:
            st.info("Run both initial segmentation and beam refinement to see analysis")

    with tab2:
        show_period_details(refined_periods)

    with tab3:
        show_segmentation_quality_metrics(refined_periods)

    with tab4:
        st.write(
            """
        **Segmentation & Refinement Algorithm:**
        
        **Stage 3A: Initial Segmentation**
        1. **Input Processing:** Takes boundary years from change detection
        2. **Segment Creation:** Creates contiguous periods between boundaries
        3. **Academic Year Aggregation:** Combines academic years within each period
        4. **Keyword Consolidation:** Merges keyword frequencies across years
        5. **Validation:** Ensures no gaps or overlaps between periods
        
        **Stage 3B: Beam Search Refinement**
        1. **Initialization:** Start with initial segmentation as base state
        2. **State Generation:** Generate successor states through merge/split operations
        3. **Evaluation:** Score each state using objective function + length penalties
        4. **Beam Search:** Keep top-k states and iterate until convergence
        5. **Optimization:** Select best final state based on cohesion/separation metrics
        
        **Key Features:**
        - Contiguous periods with no gaps
        - Automatic handling of edge cases
        - Objective function-guided refinement
        - Constraint-aware split/merge operations
        - Pre-computed aggregations for efficiency
        """
        )

        if refined_periods:
            boundary_year_list = [ay.year for ay in boundary_years]
            st.write(f"**Detected Boundaries:** {boundary_year_list}")

            st.write(f"**Initial Periods:**")
            for i, period in enumerate(initial_periods):
                st.write(
                    f"- Period {i+1}: {period.start_year}-{period.end_year} ({period.end_year - period.start_year + 1} years)"
                )

            st.write(f"**Refined Periods:**")
            for i, period in enumerate(refined_periods):
                st.write(
                    f"- Period {i+1}: {period.start_year}-{period.end_year} ({period.end_year - period.start_year + 1} years)"
                )

        # Show beam search configuration
        config = st.session_state.algorithm_config
        st.write("**Beam Search Configuration:**")
        st.write(f"- Enabled: {config.beam_search_enabled}")
        if config.beam_search_enabled:
            st.write(f"- Beam Width: {config.beam_width}")
            st.write(f"- Max Splits per Segment: {config.max_splits_per_segment}")
            st.write(f"- Min Period Years: {config.min_period_years}")
            st.write(f"- Max Period Years: {config.max_period_years}")

    # Data export
    st.subheader("💾 Export Results")
    if st.button("Export Segmentation & Refinement Results"):
        segmentation_data = {
            "domain": st.session_state.selected_domain,
            "boundary_years": [ay.year for ay in boundary_years],
            "initial_periods": [
                {
                    "period_id": i + 1,
                    "start_year": p.start_year,
                    "end_year": p.end_year,
                    "duration": p.end_year - p.start_year + 1,
                    "total_papers": p.total_papers,
                    "total_citations": p.total_citations,
                    "top_keywords": list(p.top_keywords[:10]),
                }
                for i, p in enumerate(initial_periods)
            ],
            "refined_periods": [
                {
                    "period_id": i + 1,
                    "start_year": p.start_year,
                    "end_year": p.end_year,
                    "duration": p.end_year - p.start_year + 1,
                    "total_papers": p.total_papers,
                    "total_citations": p.total_citations,
                    "top_keywords": list(p.top_keywords[:10]),
                }
                for i, p in enumerate(refined_periods)
            ],
            "beam_search_config": {
                "enabled": config.beam_search_enabled,
                "beam_width": config.beam_width,
                "max_splits_per_segment": config.max_splits_per_segment,
                "min_period_years": config.min_period_years,
                "max_period_years": config.max_period_years,
            },
        }

        st.download_button(
            label="Download Segmentation & Refinement JSON",
            data=pd.Series(segmentation_data).to_json(indent=2),
            file_name=f"{st.session_state.selected_domain}_segmentation_refinement_results.json",
            mime="application/json",
        )
