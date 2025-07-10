"""Evaluation visualization page for timeline analysis results.

This page loads and displays evaluation results comparing the algorithm against baselines
and showing auto-metrics. It reads from saved JSON files to avoid re-running evaluations.
"""

import json
import streamlit as st
from pathlib import Path
from typing import Dict, List, Optional, Any
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from datetime import datetime
import numpy as np

from core.utils.general import discover_available_domains


def load_evaluation_results(domain_name: str) -> Optional[Dict[str, Any]]:
    """Load evaluation results from JSON file.

    Args:
        domain_name: Name of the domain to load results for

    Returns:
        Dictionary containing evaluation results, or None if not found
    """
    results_file = Path(f"results/evaluation/{domain_name}_evaluation.json")
    if results_file.exists():
        with open(results_file, "r") as f:
            return json.load(f)
    return None


def load_timeline_analysis(domain_name: str) -> Optional[Dict[str, Any]]:
    """Load detailed timeline analysis results from JSON file.

    Args:
        domain_name: Name of the domain to load timeline for

    Returns:
        Dictionary containing timeline analysis results, or None if not found
    """
    timeline_file = Path(f"results/timelines/{domain_name}_timeline_analysis.json")
    if timeline_file.exists():
        with open(timeline_file, "r") as f:
            return json.load(f)
    return None


def run_evaluation_for_domain(domain_name: str) -> Optional[Dict[str, Any]]:
    """Run evaluation for a domain if results don't exist.

    Args:
        domain_name: Name of the domain to evaluate

    Returns:
        Dictionary containing evaluation results, or None if failed
    """
    import subprocess
    import sys

    # Run evaluation script
    try:
        result = subprocess.run(
            [sys.executable, "run_evaluation.py", "--domain", domain_name],
            capture_output=True,
            text=True,
            cwd=".",
        )

        if result.returncode == 0:
            return load_evaluation_results(domain_name)
        else:
            st.error(f"Evaluation failed: {result.stderr}")
            return None
    except Exception as e:
        st.error(f"Error running evaluation: {e}")
        return None


def display_evaluation_summary(results: Dict[str, Any], domain_name: str):
    """Display evaluation summary with metrics and comparisons.

    Args:
        results: Dictionary containing evaluation results
        domain_name: Name of the domain being evaluated
    """
    st.subheader(f"🎯 Evaluation Summary: {domain_name}")

    # Algorithm result
    algorithm_result = results.get("algorithm_result", {})

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Algorithm Score",
            f"{algorithm_result.get('objective_score', 0):.3f}",
            help="Higher is better",
        )

    with col2:
        st.metric(
            "Cohesion Score",
            f"{algorithm_result.get('cohesion_score', 0):.3f}",
            help="Intra-period coherence",
        )

    with col3:
        st.metric(
            "Separation Score",
            f"{algorithm_result.get('separation_score', 0):.3f}",
            help="Inter-period distinctness",
        )

    with col4:
        st.metric(
            "Segments",
            algorithm_result.get("num_segments", 0),
            help="Number of time periods",
        )


def display_baseline_comparison(results: Dict[str, Any]):
    """Display baseline comparison with ranking.

    Args:
        results: Dictionary containing evaluation results
    """
    st.subheader("📊 Baseline Comparison")

    # Collect all scores
    algorithm_score = results.get("algorithm_result", {}).get("objective_score", 0)
    baseline_results = results.get("baseline_results", [])

    scores = [("Algorithm", algorithm_score)]
    baseline_dict = {}

    # Handle both list and dict formats for baseline results
    if isinstance(baseline_results, list):
        for baseline_data in baseline_results:
            name = baseline_data.get("baseline_name", "")
            score = baseline_data.get("objective_score", 0)
            scores.append((name.title(), score))
            baseline_dict[name.lower()] = baseline_data
    else:
        for baseline_name, baseline_data in baseline_results.items():
            scores.append(
                (baseline_name.title(), baseline_data.get("objective_score", 0))
            )
            baseline_dict[baseline_name.lower()] = baseline_data

    # Sort by score (higher is better)
    scores.sort(key=lambda x: x[1], reverse=True)

    # Create comparison table
    comparison_data = []
    for rank, (name, score) in enumerate(scores, 1):
        emoji = (
            "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else "🏅"
        )
        num_segments = 0
        if name == "Algorithm":
            num_segments = results.get("algorithm_result", {}).get("num_segments", 0)
        else:
            num_segments = baseline_dict.get(name.lower(), {}).get("num_segments", 0)

        comparison_data.append(
            {
                "Rank": f"{emoji} {rank}",
                "Method": name,
                "Objective Score": f"{score:.3f}",
                "Segments": num_segments,
            }
        )

    df = pd.DataFrame(comparison_data)
    st.dataframe(df, hide_index=True, use_container_width=True)

    # Bar chart comparison
    fig = go.Figure()

    colors = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7"]

    for i, (name, score) in enumerate(scores):
        fig.add_trace(
            go.Bar(
                name=name,
                x=[name],
                y=[score],
                marker_color=colors[i % len(colors)],
                text=[f"{score:.3f}"],
                textposition="auto",
            )
        )

    fig.update_layout(
        title="Objective Score Comparison",
        xaxis_title="Method",
        yaxis_title="Objective Score",
        showlegend=False,
        height=400,
    )

    st.plotly_chart(fig, use_container_width=True)


def display_auto_metrics(results: Dict[str, Any]):
    """Display auto-metrics results.

    Args:
        results: Dictionary containing evaluation results
    """
    st.subheader("🎯 Auto-Metrics (vs Manual Reference)")

    auto_metrics = results.get("auto_metrics", {})
    if not auto_metrics:
        st.info("No auto-metrics available")
        return

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📍 Boundary F1")

        metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
        with metrics_col1:
            st.metric(
                "F1 Score",
                f"{auto_metrics.get('boundary_f1', 0):.3f}",
                help="Boundary detection F1 score",
            )
        with metrics_col2:
            st.metric(
                "Precision",
                f"{auto_metrics.get('boundary_precision', 0):.3f}",
                help="Boundary detection precision",
            )
        with metrics_col3:
            st.metric(
                "Recall",
                f"{auto_metrics.get('boundary_recall', 0):.3f}",
                help="Boundary detection recall",
            )

        st.caption(f"Tolerance: ±{auto_metrics.get('tolerance', 2)} years")

    with col2:
        st.subheader("📊 Segment F1")

        metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
        with metrics_col1:
            st.metric(
                "F1 Score",
                f"{auto_metrics.get('segment_f1', 0):.3f}",
                help="Segment overlap F1 score",
            )
        with metrics_col2:
            st.metric(
                "Precision",
                f"{auto_metrics.get('segment_precision', 0):.3f}",
                help="Segment overlap precision",
            )
        with metrics_col3:
            st.metric(
                "Recall",
                f"{auto_metrics.get('segment_recall', 0):.3f}",
                help="Segment overlap recall",
            )

        st.caption("Max 3 segments per ground truth segment")


def display_detailed_results(results: Dict[str, Any]):
    """Display detailed results in expandable sections.

    Args:
        results: Dictionary containing evaluation results
    """
    st.subheader("📋 Detailed Results")

    # Algorithm details
    with st.expander("🔍 Algorithm Details"):
        algorithm_result = results.get("algorithm_result", {})

        col1, col2 = st.columns(2)
        with col1:
            st.write("**Boundary Years:**")
            boundary_years = algorithm_result.get("boundary_years", [])
            if boundary_years:
                st.write(", ".join(map(str, boundary_years)))
            else:
                st.write("No boundary years found")

        with col2:
            st.write("**Period Count:**")
            st.write(f"{algorithm_result.get('num_segments', 0)} segments")

        st.write("**Scoring Breakdown:**")
        st.write(f"- Cohesion: {algorithm_result.get('cohesion_score', 0):.3f}")
        st.write(f"- Separation: {algorithm_result.get('separation_score', 0):.3f}")
        st.write(f"- Overall: {algorithm_result.get('objective_score', 0):.3f}")

    # Baseline details
    baseline_results = results.get("baseline_results", [])
    if baseline_results:
        with st.expander("📊 Baseline Details"):
            # Handle both list and dict formats
            if isinstance(baseline_results, list):
                for baseline_data in baseline_results:
                    baseline_name = baseline_data.get("baseline_name", "")
                    st.write(f"**{baseline_name.title()} Baseline:**")
                    st.write(
                        f"- Objective Score: {baseline_data.get('objective_score', 0):.3f}"
                    )
                    st.write(f"- Segments: {baseline_data.get('num_segments', 0)}")

                    boundary_years = baseline_data.get("boundary_years", [])
                    if boundary_years:
                        st.write(
                            f"- Boundary Years: {', '.join(map(str, boundary_years))}"
                        )
                    st.write("---")
            else:
                for baseline_name, baseline_data in baseline_results.items():
                    st.write(f"**{baseline_name.title()} Baseline:**")
                    st.write(
                        f"- Objective Score: {baseline_data.get('objective_score', 0):.3f}"
                    )
                    st.write(f"- Segments: {baseline_data.get('num_segments', 0)}")

                    boundary_years = baseline_data.get("boundary_years", [])
                    if boundary_years:
                        st.write(
                            f"- Boundary Years: {', '.join(map(str, boundary_years))}"
                        )
                    st.write("---")


def create_baseline_segments(
    boundary_years: List[int], domain_name: str
) -> List[Dict[str, Any]]:
    """Create baseline segments from boundary years.

    Args:
        boundary_years: List of boundary years
        domain_name: Domain name for context

    Returns:
        List of segment dictionaries
    """
    segments = []
    if len(boundary_years) < 2:
        return segments

    for i in range(len(boundary_years) - 1):
        start_year = boundary_years[i]
        end_year = (
            boundary_years[i + 1] - 1
            if i < len(boundary_years) - 2
            else boundary_years[i + 1]
        )

        segments.append(
            {
                "start_year": start_year,
                "end_year": end_year,
                "period_name": f"{start_year}-{end_year}",
                "top_keywords": [
                    f"{domain_name} research",
                    "academic literature",
                    "scholarly work",
                ],  # Generic keywords for baselines
                "total_papers": 0,  # Not available for baselines
                "duration": end_year - start_year + 1,
            }
        )

    return segments


def plot_detailed_timeline(
    timeline_data: Dict[str, Any], title: str, color: str, y_position: float
) -> go.Figure:
    """Create a detailed timeline plot with segments and keywords.

    Args:
        timeline_data: Timeline data with periods
        title: Title for the timeline
        color: Color for the timeline
        y_position: Y position for the timeline

    Returns:
        Plotly figure traces
    """
    periods = timeline_data.get("periods", [])
    if not periods:
        return []

    traces = []

    # Create segments as horizontal bars
    for i, period in enumerate(periods):
        start_year = period.get("start_year", 0)
        end_year = period.get("end_year", 0)
        duration = end_year - start_year + 1

        # Top keywords for hover info
        keywords = period.get("top_keywords", [])
        keyword_text = "<br>".join(keywords[:5]) if keywords else "No keywords"

        # Papers count
        papers = period.get("total_papers", 0)

        # Create hover text
        hover_text = f"<b>{title}</b><br>Period {i+1}: {start_year}-{end_year}<br>Duration: {duration} years<br>Papers: {papers:,}<br><br><b>Top Keywords:</b><br>{keyword_text}"

        # Add segment bar
        traces.append(
            go.Bar(
                name=f"{title} - Period {i+1}",
                x=[duration],
                y=[y_position],
                base=start_year,
                orientation="h",
                marker=dict(
                    color=color, opacity=0.7, line=dict(color="white", width=1)
                ),
                hovertemplate=hover_text + "<extra></extra>",
                showlegend=False,
                text=f"P{i+1}",
                textposition="inside",
                textfont=dict(color="white", size=10),
            )
        )

    return traces


def show_comprehensive_timeline_visualization(
    results: Dict[str, Any], domain_name: str
):
    """Show comprehensive timeline visualization with all methods and keywords.

    Args:
        results: Dictionary containing evaluation results
        domain_name: Name of the domain
    """
    st.subheader("📈 Comprehensive Timeline Analysis")
    st.write(
        "Detailed visualization of all timeline segmentation methods with keywords for each period"
    )

    # Load detailed timeline analysis
    timeline_analysis = load_timeline_analysis(domain_name)

    # Prepare data for visualization
    all_timelines = []

    # 1. Algorithm timeline (detailed if available)
    if timeline_analysis:
        all_timelines.append(
            {
                "name": "Algorithm",
                "data": timeline_analysis,
                "color": "#FF6B6B",
                "type": "detailed",
            }
        )
    else:
        # Fallback to boundary years only
        algorithm_boundaries = results.get("algorithm_result", {}).get(
            "boundary_years", []
        )
        if algorithm_boundaries:
            segments = create_baseline_segments(algorithm_boundaries, domain_name)
            all_timelines.append(
                {
                    "name": "Algorithm",
                    "data": {"periods": segments},
                    "color": "#FF6B6B",
                    "type": "baseline",
                }
            )

    # 2. Baseline timelines
    baseline_results = results.get("baseline_results", [])
    baseline_colors = ["#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7"]

    if isinstance(baseline_results, list):
        for i, baseline_data in enumerate(baseline_results):
            baseline_name = baseline_data.get("baseline_name", "")
            if baseline_name in ["5-year", "10-year"]:  # Skip for cleaner visualization
                continue

            boundary_years = baseline_data.get("boundary_years", [])
            if boundary_years:
                segments = create_baseline_segments(boundary_years, domain_name)
                all_timelines.append(
                    {
                        "name": baseline_name.title(),
                        "data": {"periods": segments},
                        "color": baseline_colors[i % len(baseline_colors)],
                        "type": "baseline",
                    }
                )

    if not all_timelines:
        st.warning("No timeline data available for visualization")
        return

    # Create the comprehensive timeline plot
    fig = go.Figure()

    # Add each timeline
    for i, timeline in enumerate(all_timelines):
        traces = plot_detailed_timeline(
            timeline["data"], timeline["name"], timeline["color"], i
        )
        for trace in traces:
            fig.add_trace(trace)

    # Update layout
    fig.update_layout(
        title=f"Timeline Segmentation Comparison: {domain_name.title()}",
        xaxis_title="Year",
        yaxis_title="Method",
        yaxis=dict(
            tickvals=list(range(len(all_timelines))),
            ticktext=[timeline["name"] for timeline in all_timelines],
            showgrid=True,
        ),
        height=100 + len(all_timelines) * 80,
        showlegend=False,
        hovermode="closest",
        plot_bgcolor="white",
        font=dict(size=12),
    )

    # Add grid
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="lightgray")
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="lightgray")

    st.plotly_chart(fig, use_container_width=True)

    # Show detailed breakdown
    show_timeline_details(all_timelines, domain_name)


def show_timeline_details(all_timelines: List[Dict], domain_name: str):
    """Show detailed breakdown of each timeline.

    Args:
        all_timelines: List of timeline data
        domain_name: Name of the domain
    """
    st.subheader("📋 Timeline Details")

    for timeline in all_timelines:
        with st.expander(f"🔍 {timeline['name']} Timeline Details"):
            periods = timeline["data"].get("periods", [])

            if not periods:
                st.write("No period data available")
                continue

            # Create summary table
            period_data = []
            for i, period in enumerate(periods):
                period_data.append(
                    {
                        "Period": f"P{i+1}",
                        "Years": f"{period.get('start_year', 0)}-{period.get('end_year', 0)}",
                        "Duration": f"{period.get('end_year', 0) - period.get('start_year', 0) + 1} years",
                        "Papers": (
                            f"{period.get('total_papers', 0):,}"
                            if period.get("total_papers", 0) > 0
                            else "N/A"
                        ),
                        "Top Keywords": ", ".join(period.get("top_keywords", [])[:3]),
                    }
                )

            df = pd.DataFrame(period_data)
            st.dataframe(df, hide_index=True, use_container_width=True)

            # Show keywords in detail
            st.write("**Detailed Keywords by Period:**")
            for i, period in enumerate(periods):
                keywords = period.get("top_keywords", [])
                if keywords:
                    st.write(
                        f"**Period {i+1} ({period.get('start_year', 0)}-{period.get('end_year', 0)}):**"
                    )
                    keyword_str = " • ".join(keywords[:8])  # Show top 8 keywords
                    st.write(f"• {keyword_str}")
                    st.write("")


def show_timeline_visualization(results: Dict[str, Any]):
    """Show timeline visualization comparing algorithm vs baselines.

    Args:
        results: Dictionary containing evaluation results
    """
    st.subheader("📈 Timeline Boundaries Overview")

    # Get all boundary years
    algorithm_boundaries = results.get("algorithm_result", {}).get("boundary_years", [])
    baseline_results = results.get("baseline_results", [])

    # Create timeline plot
    fig = go.Figure()

    # Add algorithm boundaries
    if algorithm_boundaries:
        fig.add_trace(
            go.Scatter(
                x=algorithm_boundaries,
                y=[0] * len(algorithm_boundaries),
                mode="markers",
                name="Algorithm",
                marker=dict(size=10, color="red", symbol="diamond"),
                hovertemplate="Algorithm: %{x}<extra></extra>",
            )
        )

    # Add baseline boundaries
    colors = ["blue", "green", "orange", "purple"]
    y_offset = 0.1

    # Handle both list and dict formats
    if isinstance(baseline_results, list):
        for i, baseline_data in enumerate(baseline_results):
            baseline_name = baseline_data.get("baseline_name", "")
            if baseline_name in [
                "5-year",
                "10-year",
            ]:  # Skip these for cleaner visualization
                continue

            boundaries = baseline_data.get("boundary_years", [])
            if boundaries:
                fig.add_trace(
                    go.Scatter(
                        x=boundaries,
                        y=[y_offset * (i + 1)] * len(boundaries),
                        mode="markers",
                        name=baseline_name.title(),
                        marker=dict(size=8, color=colors[i % len(colors)]),
                        hovertemplate=f"{baseline_name.title()}: %{{x}}<extra></extra>",
                    )
                )
    else:
        for i, (baseline_name, baseline_data) in enumerate(baseline_results.items()):
            if baseline_name in [
                "5-year",
                "10-year",
            ]:  # Skip these for cleaner visualization
                continue

            boundaries = baseline_data.get("boundary_years", [])
            if boundaries:
                fig.add_trace(
                    go.Scatter(
                        x=boundaries,
                        y=[y_offset * (i + 1)] * len(boundaries),
                        mode="markers",
                        name=baseline_name.title(),
                        marker=dict(size=8, color=colors[i % len(colors)]),
                        hovertemplate=f"{baseline_name.title()}: %{{x}}<extra></extra>",
                    )
                )

    fig.update_layout(
        title="Timeline Boundaries Comparison",
        xaxis_title="Year",
        yaxis_title="Method",
        yaxis=dict(showticklabels=False),
        height=300,
        showlegend=True,
    )

    st.plotly_chart(fig, use_container_width=True)


def show_evaluation():
    """Main evaluation page function."""
    st.title("🎯 Timeline Evaluation")
    st.write("Compare algorithm performance against baselines and reference timelines")

    # Domain selection
    available_domains = discover_available_domains()
    if not available_domains:
        st.error("No domains found in resources directory")
        return

    selected_domain = st.selectbox(
        "Select Domain for Evaluation",
        options=available_domains,
        help="Choose the domain to view evaluation results",
    )

    if not selected_domain:
        return

    # Load or run evaluation
    results = load_evaluation_results(selected_domain)

    if results is None:
        st.warning(f"No evaluation results found for {selected_domain}")

        if st.button(f"Run Evaluation for {selected_domain}"):
            with st.spinner("Running evaluation... This may take a few minutes."):
                results = run_evaluation_for_domain(selected_domain)

                if results:
                    st.success("Evaluation completed successfully!")
                    st.rerun()
                else:
                    st.error("Evaluation failed. Please check the logs.")
                    return
        else:
            st.info("Click the button above to run evaluation for this domain.")
            return

    # Display results
    if results:
        # Show metadata
        metadata = results.get("metadata", {})
        if metadata:
            st.caption(f"Evaluation run on: {metadata.get('timestamp', 'Unknown')}")

        # Display all sections
        display_evaluation_summary(results, selected_domain)
        st.divider()

        display_baseline_comparison(results)
        st.divider()

        display_auto_metrics(results)
        st.divider()

        show_comprehensive_timeline_visualization(results, selected_domain)
        st.divider()

        show_timeline_visualization(results)
        st.divider()

        display_detailed_results(results)

        # Download results
        st.subheader("💾 Download Results")
        results_json = json.dumps(results, indent=2)
        st.download_button(
            label="Download Evaluation Results (JSON)",
            data=results_json,
            file_name=f"{selected_domain}_evaluation_results.json",
            mime="application/json",
        )
