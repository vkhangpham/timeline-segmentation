"""Stage 5: Period Merging
Visualizes intelligent merging of similar adjacent periods.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from collections import Counter
from core.segmentation.segment_merging import merge_similar_periods


def run_merging():
    """Run period merging and store results."""
    if st.session_state.characterized_periods is None:
        st.error("Please run characterization first in Stage 4")
        return False

    if st.session_state.final_periods is None:
        with st.spinner("Merging similar periods..."):
            start_time = time.time()

            # Create a temporary config object with merge threshold
            config = st.session_state.algorithm_config
            merge_threshold = getattr(
                st.session_state, "merge_similarity_threshold", 0.6
            )

            # Create a temporary config-like object with the merge threshold
            class TempConfig:
                def __init__(self, base_config, merge_threshold):
                    # Copy all attributes from base config
                    for attr in dir(base_config):
                        if not attr.startswith("_"):
                            setattr(self, attr, getattr(base_config, attr))
                    # Add merge threshold
                    self.merge_similarity_threshold = merge_threshold

            temp_config = TempConfig(config, merge_threshold)

            final_periods = merge_similar_periods(
                periods=st.session_state.characterized_periods,
                algorithm_config=temp_config,
                verbose=False,
            )

            merging_time = time.time() - start_time
            st.session_state.timing_data["Merging"] = merging_time
            st.session_state.final_periods = final_periods

            st.success(f"✅ Merging completed in {merging_time:.2f}s")

    return True


def calculate_keyword_similarity_matrix(periods):
    """Calculate keyword similarity matrix between periods."""
    if not periods:
        return None, None

    period_names = [f"Period {i+1}" for i in range(len(periods))]
    n_periods = len(periods)

    # Calculate Jaccard similarity matrix
    similarity_matrix = []
    for i in range(n_periods):
        row = []
        for j in range(n_periods):
            if i == j:
                similarity = 1.0
            else:
                keywords_i = set(periods[i].top_keywords[:20])
                keywords_j = set(periods[j].top_keywords[:20])

                intersection = len(keywords_i & keywords_j)
                union = len(keywords_i | keywords_j)

                similarity = intersection / union if union > 0 else 0.0

            row.append(similarity)
        similarity_matrix.append(row)

    return similarity_matrix, period_names


def create_similarity_heatmap(periods):
    """Create keyword similarity heatmap."""
    similarity_matrix, period_names = calculate_keyword_similarity_matrix(periods)

    if similarity_matrix is None:
        return None

    fig = go.Figure(
        data=go.Heatmap(
            z=similarity_matrix,
            x=period_names,
            y=period_names,
            colorscale="RdBu_r",
            zmin=0,
            zmax=1,
            hoverongaps=False,
            hovertemplate="<b>%{x} vs %{y}</b><br>Similarity: %{z:.3f}<extra></extra>",
        )
    )

    fig.update_layout(
        title="Period Keyword Similarity Matrix (Jaccard Index)",
        xaxis_title="Periods",
        yaxis_title="Periods",
        height=500,
    )

    return fig


def create_merging_comparison_chart(initial_periods, final_periods):
    """Create before/after comparison of period merging."""
    if not initial_periods or not final_periods:
        return None

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Before: Period Count by Duration",
            "After: Period Count by Duration",
            "Before: Papers Distribution",
            "After: Papers Distribution",
        ),
    )

    # Before merging - durations
    initial_durations = [p.end_year - p.start_year + 1 for p in initial_periods]
    initial_papers = [p.total_papers for p in initial_periods]

    # After merging - durations
    final_durations = [p.end_year - p.start_year + 1 for p in final_periods]
    final_papers = [p.total_papers for p in final_periods]

    # Duration histograms
    fig.add_trace(
        go.Histogram(
            x=initial_durations,
            name="Initial Durations",
            marker_color="lightcoral",
            opacity=0.7,
            nbinsx=10,
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Histogram(
            x=final_durations,
            name="Final Durations",
            marker_color="lightblue",
            opacity=0.7,
            nbinsx=10,
        ),
        row=1,
        col=2,
    )

    # Papers distribution
    fig.add_trace(
        go.Box(
            y=initial_papers,
            name="Initial Papers",
            marker_color="lightcoral",
            boxpoints="all",
            jitter=0.3,
        ),
        row=2,
        col=1,
    )

    fig.add_trace(
        go.Box(
            y=final_papers,
            name="Final Papers",
            marker_color="lightblue",
            boxpoints="all",
            jitter=0.3,
        ),
        row=2,
        col=2,
    )

    fig.update_layout(
        height=600,
        title_text="Period Merging: Before vs After Comparison",
        showlegend=False,
    )

    return fig


def create_merging_timeline_visualization(initial_periods, final_periods):
    """Create timeline showing merging process."""
    if not initial_periods or not final_periods:
        return None

    fig = make_subplots(
        rows=2,
        cols=1,
        subplot_titles=("Before Merging", "After Merging"),
        vertical_spacing=0.15,
        shared_xaxes=True,
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
    ]

    # Before merging
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
                + f"Papers: {period.total_papers}<br>"
                + f'Topic: {period.topic_label or "N/A"}<extra></extra>',
                showlegend=False,
            ),
            row=1,
            col=1,
        )

        # Add period labels
        mid_year = (period.start_year + period.end_year) / 2
        fig.add_annotation(
            x=mid_year,
            y=i + 0.3,
            text=f"P{i+1}",
            showarrow=False,
            font=dict(size=10),
            row=1,
            col=1,
        )

    # After merging
    for i, period in enumerate(final_periods):
        color = colors[i % len(colors)]
        fig.add_trace(
            go.Scatter(
                x=[period.start_year, period.end_year],
                y=[i, i],
                mode="lines+markers",
                name=f"Final P{i+1}",
                line=dict(color=color, width=8),
                marker=dict(size=12),
                hovertemplate=f"<b>Final Period {i+1}</b><br>"
                + f"Years: {period.start_year}-{period.end_year}<br>"
                + f"Papers: {period.total_papers}<br>"
                + f'Topic: {period.topic_label or "N/A"}<extra></extra>',
                showlegend=False,
            ),
            row=2,
            col=1,
        )

        # Add period labels
        mid_year = (period.start_year + period.end_year) / 2
        fig.add_annotation(
            x=mid_year,
            y=i + 0.3,
            text=f"P{i+1}",
            showarrow=False,
            font=dict(size=12, color=color),
            row=2,
            col=1,
        )

    fig.update_layout(
        height=max(400, len(initial_periods) * 40 + len(final_periods) * 40),
        title_text="Period Merging Timeline Visualization",
    )

    fig.update_xaxes(title_text="Year", row=2, col=1)
    fig.update_yaxes(title_text="Initial Periods", row=1, col=1, showticklabels=False)
    fig.update_yaxes(title_text="Final Periods", row=2, col=1, showticklabels=False)

    return fig


def show_merging_analysis(initial_periods, final_periods):
    """Show detailed merging analysis."""
    st.subheader("🔍 Merging Analysis")

    # Merging statistics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Initial Periods", len(initial_periods))
    with col2:
        st.metric("Final Periods", len(final_periods))
    with col3:
        merged_count = len(initial_periods) - len(final_periods)
        st.metric("Periods Merged", merged_count)
    with col4:
        merge_rate = (
            (merged_count / len(initial_periods)) * 100 if initial_periods else 0
        )
        st.metric("Merge Rate", f"{merge_rate:.1f}%")

    # Quality improvements
    st.write("**Quality Impact:**")

    # Calculate average durations
    initial_avg_duration = (
        sum(p.end_year - p.start_year + 1 for p in initial_periods)
        / len(initial_periods)
        if initial_periods
        else 0
    )
    final_avg_duration = (
        sum(p.end_year - p.start_year + 1 for p in final_periods) / len(final_periods)
        if final_periods
        else 0
    )

    # Calculate duration standard deviation
    initial_durations = [p.end_year - p.start_year + 1 for p in initial_periods]
    final_durations = [p.end_year - p.start_year + 1 for p in final_periods]

    initial_duration_std = (
        (
            sum((d - initial_avg_duration) ** 2 for d in initial_durations)
            / len(initial_durations)
        )
        ** 0.5
        if initial_durations
        else 0
    )
    final_duration_std = (
        (
            sum((d - final_avg_duration) ** 2 for d in final_durations)
            / len(final_durations)
        )
        ** 0.5
        if final_durations
        else 0
    )

    col1, col2 = st.columns(2)

    with col1:
        st.write("**Before Merging:**")
        st.write(f"- Average Duration: {initial_avg_duration:.1f} years")
        st.write(f"- Duration Std Dev: {initial_duration_std:.1f}")
        st.write(f"- Number of Periods: {len(initial_periods)}")

        if initial_periods:
            min_papers = min(p.total_papers for p in initial_periods)
            max_papers = max(p.total_papers for p in initial_periods)
            st.write(f"- Paper Range: {min_papers}-{max_papers}")

    with col2:
        st.write("**After Merging:**")
        st.write(f"- Average Duration: {final_avg_duration:.1f} years")
        st.write(f"- Duration Std Dev: {final_duration_std:.1f}")
        st.write(f"- Number of Periods: {len(final_periods)}")

        if final_periods:
            min_papers = min(p.total_papers for p in final_periods)
            max_papers = max(p.total_papers for p in final_periods)
            st.write(f"- Paper Range: {min_papers}-{max_papers}")

    # Show which periods were merged
    st.write("**Merging Details:**")
    if len(final_periods) < len(initial_periods):
        st.write(
            "Based on keyword similarity and temporal adjacency, the following merging occurred:"
        )

        # Simple heuristic to show which periods likely merged
        # (This is approximated since we don't track exact merging history)
        initial_years = [(p.start_year, p.end_year) for p in initial_periods]
        final_years = [(p.start_year, p.end_year) for p in final_periods]

        for i, (final_start, final_end) in enumerate(final_years):
            # Find initial periods that could have contributed to this final period
            contributing_periods = []
            for j, (init_start, init_end) in enumerate(initial_years):
                if init_start >= final_start and init_end <= final_end:
                    contributing_periods.append(j + 1)

            if len(contributing_periods) > 1:
                st.write(
                    f"- Final Period {i+1} ({final_start}-{final_end}) ← Initial Periods {contributing_periods}"
                )


def show_period_comparison_table(initial_periods, final_periods):
    """Show side-by-side comparison table."""
    st.subheader("📋 Period Comparison Table")

    col1, col2 = st.columns(2)

    with col1:
        st.write("**Initial Periods (Before Merging):**")
        if initial_periods:
            initial_data = []
            for i, period in enumerate(initial_periods):
                initial_data.append(
                    {
                        "Period": i + 1,
                        "Years": f"{period.start_year}-{period.end_year}",
                        "Duration": period.end_year - period.start_year + 1,
                        "Papers": period.total_papers,
                        "Topic": (
                            period.topic_label[:25] + "..."
                            if period.topic_label and len(period.topic_label) > 25
                            else period.topic_label
                        )
                        or "N/A",
                    }
                )

            initial_df = pd.DataFrame(initial_data)
            st.dataframe(initial_df, use_container_width=True, hide_index=True)

    with col2:
        st.write("**Final Periods (After Merging):**")
        if final_periods:
            final_data = []
            for i, period in enumerate(final_periods):
                final_data.append(
                    {
                        "Period": i + 1,
                        "Years": f"{period.start_year}-{period.end_year}",
                        "Duration": period.end_year - period.start_year + 1,
                        "Papers": period.total_papers,
                        "Topic": (
                            period.topic_label[:25] + "..."
                            if period.topic_label and len(period.topic_label) > 25
                            else period.topic_label
                        )
                        or "N/A",
                    }
                )

            final_df = pd.DataFrame(final_data)
            st.dataframe(final_df, use_container_width=True, hide_index=True)


def show_merging():
    """Main merging page function."""
    st.header("🔗 Stage 5: Period Merging")
    st.write("Intelligently merge similar adjacent periods based on keyword overlap.")

    # Check prerequisites
    if st.session_state.characterized_periods is None:
        st.error("Please run characterization first in Stage 4")
        return

    # Control panel
    col1, col2 = st.columns([3, 1])
    with col1:
        st.subheader("🎯 Merging Results")
        st.write(
            "⚠️ **Note:** This stage may also include optimization steps that can take time."
        )
    with col2:
        if st.button("🔄 Run Merging", type="primary"):
            st.session_state.final_periods = None  # Force re-run
            run_merging()

    # Run merging if needed
    if not run_merging():
        return

    initial_periods = st.session_state.characterized_periods
    final_periods = st.session_state.final_periods

    # Results summary
    merged_count = len(initial_periods) - len(final_periods)
    merge_rate = (merged_count / len(initial_periods)) * 100 if initial_periods else 0

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Periods Merged", merged_count)
    with col2:
        st.metric("Final Period Count", len(final_periods))
    with col3:
        st.metric("Merge Rate", f"{merge_rate:.1f}%")

    # Keyword similarity analysis
    st.subheader("📊 Keyword Similarity Analysis")
    similarity_fig = create_similarity_heatmap(initial_periods)
    if similarity_fig:
        st.plotly_chart(similarity_fig, use_container_width=True)

        st.write(
            """
        **Similarity Interpretation:**
        - **Red (0.0-0.3):** Low similarity - periods likely distinct
        - **White (0.3-0.7):** Moderate similarity - potential merge candidates
        - **Blue (0.7-1.0):** High similarity - strong merge candidates
        """
        )

    # Timeline visualization
    st.subheader("📈 Merging Timeline")
    timeline_fig = create_merging_timeline_visualization(initial_periods, final_periods)
    if timeline_fig:
        st.plotly_chart(timeline_fig, use_container_width=True)

    # Before/after comparison
    st.subheader("📊 Before vs After Comparison")
    comparison_fig = create_merging_comparison_chart(initial_periods, final_periods)
    if comparison_fig:
        st.plotly_chart(comparison_fig, use_container_width=True)

    # Detailed analysis tabs
    tab1, tab2, tab3 = st.tabs(
        ["🔍 Merging Analysis", "📋 Period Comparison", "📖 Algorithm Info"]
    )

    with tab1:
        show_merging_analysis(initial_periods, final_periods)

    with tab2:
        show_period_comparison_table(initial_periods, final_periods)

    with tab3:
        st.write(
            """
        **Period Merging Algorithm:**
        
        **Sequential Merging Process:**
        1. **Initialize:** Start with first period in result list
        2. **For each subsequent period:**
           - Calculate keyword overlap with last merged period
           - Use top 10 keywords from each period
           - Compute Jaccard similarity: |intersection| / |union|
        3. **Merging Decision:**
           - If overlap > threshold: Merge with previous period
           - If overlap ≤ threshold: Add as separate period
        
        **Merge Operation:**
        1. **Combine Academic Years:** Concatenate all yearly data
        2. **Merge Keywords:** Add keyword frequency counters
        3. **Weighted Confidence:** Paper-count weighted average
        4. **Topic Labels:** Create "Merged: A & B" format
        5. **Network Metrics:** Weighted average of stability measures
        
        **Key Features:**
        - **Sequential Processing:** One pass through periods
        - **Adjacent Only:** Only merges consecutive periods
        - **Keyword-Based:** Uses top keyword overlap for similarity
        - **Conservative:** Maintains original periods unless clear similarity
        - **Preserves Quality:** Weighted averaging of all metrics
        
        **Benefits:**
        - Reduces over-segmentation
        - Improves period coherence
        - Creates more meaningful historical narratives
        - Balances granularity with interpretability
        """
        )

        if "Merging" in st.session_state.timing_data:
            st.write(
                f"**Last Run Time:** {st.session_state.timing_data['Merging']:.2f} seconds"
            )

        # Show algorithm parameters
        config = st.session_state.algorithm_config
        st.write("**Current Configuration:**")
        similarity_threshold = getattr(config, "merge_similarity_threshold", 0.6)
        st.write(f"- Merge Similarity Threshold: {similarity_threshold}")
        st.write(f"- Minimum Segment Size: {config.anti_gaming_min_segment_size}")
        st.write(f"- Size Weight Power: {config.anti_gaming_size_weight_power}")
        st.write(f"- Cohesion Weight: {config.cohesion_weight}")
        st.write(f"- Separation Weight: {config.separation_weight}")

    # Data export
    st.subheader("💾 Export Results")
    if st.button("Export Merging Results"):
        merging_data = {
            "domain": st.session_state.selected_domain,
            "merging_summary": {
                "initial_periods": len(initial_periods),
                "final_periods": len(final_periods),
                "periods_merged": merged_count,
                "merge_rate": merge_rate,
            },
            "final_periods": [
                {
                    "period_id": i + 1,
                    "start_year": p.start_year,
                    "end_year": p.end_year,
                    "duration": p.end_year - p.start_year + 1,
                    "topic_label": p.topic_label,
                    "topic_description": p.topic_description,
                    "confidence": p.confidence,
                    "total_papers": p.total_papers,
                    "total_citations": p.total_citations,
                    "top_keywords": list(p.top_keywords[:10]),
                }
                for i, p in enumerate(final_periods)
            ],
        }

        st.download_button(
            label="Download Merging JSON",
            data=pd.Series(merging_data).to_json(indent=2),
            file_name=f"{st.session_state.selected_domain}_merging_results.json",
            mime="application/json",
        )
