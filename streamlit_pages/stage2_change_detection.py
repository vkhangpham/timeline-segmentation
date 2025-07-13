"""Stage 2: Change Point Detection
Visualizes direction change and citation acceleration detection algorithms.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import time
from core.segmentation.change_point_detection import detect_boundary_years
from core.segmentation.citation_detection import (
    detect_citation_acceleration_years,
    moving_average,
)
from core.segmentation.direction_detection import (
    detect_direction_change_years_with_citation_boost,
    calculate_direction_score,
    calculate_threshold_from_scores,
    normalize_keyword_frequencies,
    build_cumulative_baseline,
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
                verbose=False,
            )

            detection_time = time.time() - start_time
            st.session_state.timing_data["Change Detection"] = detection_time
            st.session_state.boundary_years = boundary_academic_years

            # Also compute citation acceleration years for visualization
            citation_years = detect_citation_acceleration_years(
                academic_years=st.session_state.academic_years,
                verbose=False,
            )
            st.session_state.citation_acceleration_years = citation_years

            st.success(f"✅ Change detection completed in {detection_time:.2f}s")

    return True


def calculate_direction_signals_streamlined(academic_years, algorithm_config):
    """Calculate direction change signals using the EXACT same algorithm as the main detection.

    This ensures zero discrepancy between visualization and actual algorithm execution.
    """
    # Use the EXACT same algorithm as the main detection
    citation_years = detect_citation_acceleration_years(academic_years, verbose=False)

    # Create a temporary config with diagnostics enabled to capture the detailed data
    import copy

    temp_config = copy.deepcopy(algorithm_config)
    temp_config.save_direction_diagnostics = True
    temp_config.domain_name = "visualization"

    # Call the EXACT same main algorithm function
    boundary_years = detect_direction_change_years_with_citation_boost(
        academic_years, citation_years, temp_config, verbose=False
    )

    # The algorithm generates diagnostics - we need to capture them
    # Since diagnostics are saved, we'll run it again in verbose mode to get the diagnostics
    # But we'll modify the main function to return diagnostics

    # For now, let's run the algorithm again but capture the diagnostics
    direction_data = []

    # Filter years with sufficient papers (same as main algorithm)
    min_papers_threshold = getattr(algorithm_config, "min_papers_per_year", 100)
    eligible_years = [
        ay for ay in academic_years if ay.paper_count >= min_papers_threshold
    ]

    if len(eligible_years) < 3:
        return []

    # Get parameters from config (same as main algorithm)
    scoring_method = getattr(
        algorithm_config, "direction_scoring_method", "weighted_jaccard"
    )
    min_baseline_years = getattr(algorithm_config, "min_baseline_period_years", 3)
    support_window = algorithm_config.citation_support_window_years
    boost_factor = algorithm_config.citation_confidence_boost

    # Calculate adaptive threshold (same as main algorithm)
    sampling_interval = getattr(algorithm_config, "score_distribution_window_years", 3)
    base_sample_scores = []
    last_boundary_idx = 0

    for current_idx in range(
        min_baseline_years, len(eligible_years), sampling_interval
    ):
        current_year = eligible_years[current_idx]
        if current_idx - last_boundary_idx < min_baseline_years:
            continue

        baseline_frequencies = build_cumulative_baseline(
            eligible_years, last_boundary_idx, current_idx
        )
        current_frequencies = normalize_keyword_frequencies(
            current_year.keyword_frequencies
        )
        base_score = calculate_direction_score(
            current_frequencies, baseline_frequencies, scoring_method
        )
        base_sample_scores.append(base_score)

    # Calculate threshold (same as main algorithm)
    if not base_sample_scores:
        threshold = algorithm_config.direction_change_threshold
    else:
        threshold_strategy = getattr(
            algorithm_config, "direction_threshold_strategy", "global_p90"
        )
        threshold = calculate_threshold_from_scores(
            base_sample_scores,
            threshold_strategy,
            algorithm_config.direction_change_threshold,
        )

    # Run the EXACT same cumulative algorithm logic
    last_boundary_idx = 0
    for current_idx in range(1, len(eligible_years)):
        current_year = eligible_years[current_idx]

        if current_idx - last_boundary_idx < min_baseline_years:
            continue

        # Build cumulative baseline (same as main algorithm)
        baseline_frequencies = build_cumulative_baseline(
            eligible_years, last_boundary_idx, current_idx
        )
        current_frequencies = normalize_keyword_frequencies(
            current_year.keyword_frequencies
        )

        # Compute base direction score (same as main algorithm)
        base_score = calculate_direction_score(
            current_frequencies, baseline_frequencies, scoring_method
        )

        # Check for citation support (same as main algorithm)
        has_citation_support = False
        if citation_years:
            for cit_year in citation_years:
                if abs(cit_year - current_year.year) <= support_window:
                    has_citation_support = True
                    break

        # Apply citation boost (same as main algorithm)
        if has_citation_support:
            final_score = min(base_score + boost_factor, 1.0)
        else:
            final_score = base_score

        # Calculate diagnostic metrics (same as main algorithm)
        baseline_keywords = set(baseline_frequencies.keys())
        current_keywords = set(current_frequencies.keys())
        new_keywords = current_keywords - baseline_keywords
        shared_keywords = current_keywords & baseline_keywords

        baseline_period_str = (
            f"{eligible_years[last_boundary_idx + 1].year}-{eligible_years[current_idx - 1].year}"
            if current_idx > last_boundary_idx + 1
            else "empty"
        )

        # Parse baseline period for visualization
        if baseline_period_str != "empty":
            baseline_start, baseline_end = baseline_period_str.split("-")
            baseline_period_start = int(baseline_start)
            baseline_period_end = int(baseline_end)
        else:
            baseline_period_start = current_year.year - 1
            baseline_period_end = current_year.year - 1

        # Calculate legacy metrics for comparison/visualization
        novelty = len(new_keywords) / len(current_keywords) if current_keywords else 0.0
        overlap = (
            len(shared_keywords) / len(baseline_keywords) if baseline_keywords else 0.0
        )

        direction_data.append(
            {
                "year": current_year.year,
                "direction_score": base_score,
                "scoring_method": scoring_method,
                "novelty": novelty,
                "overlap": overlap,
                "new_keywords_count": len(new_keywords),
                "shared_keywords_count": len(shared_keywords),
                "baseline_keywords_count": len(baseline_keywords),
                "current_keywords_count": len(current_keywords),
                "baseline_period_start": baseline_period_start,
                "baseline_period_end": baseline_period_end,
                "baseline_period_length": current_idx - last_boundary_idx - 1,
                "citation_support": has_citation_support,
                "final_score": final_score,
            }
        )

        # Update last_boundary_idx if this year exceeds threshold (same as main algorithm)
        if final_score > threshold:
            last_boundary_idx = current_idx

    return direction_data


def create_direction_change_chart_streamlined(
    direction_data, citation_years, algorithm_config
):
    """Create direction change visualization using streamlined algorithm results."""
    if not direction_data:
        return None

    df = pd.DataFrame(direction_data)

    # Check if final scores are already calculated (new version)
    if "final_score" in df.columns and "citation_support" in df.columns:
        final_df = df
    else:
        # Legacy fallback - calculate final scores with citation boost
        boost_factor = getattr(algorithm_config, "citation_confidence_boost", 0.5)
        support_window = getattr(algorithm_config, "citation_support_window_years", 2)

        # Calculate final scores with citation boost
        final_scores = []
        for _, row in df.iterrows():
            base_score = row["direction_score"]

            # Check for citation support
            has_citation_support = False
            if citation_years:
                for cit_year in citation_years:
                    if abs(cit_year - row["year"]) <= support_window:
                        has_citation_support = True
                        break

            # Apply boost if supported
            if has_citation_support:
                final_score = min(base_score + boost_factor, 1.0)
            else:
                final_score = base_score

            final_scores.append(
                {"final_score": final_score, "citation_support": has_citation_support}
            )

        # Add final scores to dataframe
        final_df = pd.concat([df, pd.DataFrame(final_scores)], axis=1)

    # Create adaptive threshold (simplified for visualization)
    base_scores = final_df["direction_score"].tolist()
    threshold_strategy = getattr(
        algorithm_config, "direction_threshold_strategy", "global_p90"
    )

    if threshold_strategy == "global_p90":
        threshold = np.percentile(base_scores, 90) if base_scores else 0.1
    elif threshold_strategy == "global_p95":
        threshold = np.percentile(base_scores, 95) if base_scores else 0.1
    elif threshold_strategy == "global_p99":
        threshold = np.percentile(base_scores, 99) if base_scores else 0.1
    else:
        threshold = getattr(algorithm_config, "direction_change_threshold", 0.1)

    fig = make_subplots(
        rows=3,
        cols=1,
        subplot_titles=(
            f"Direction Change Score ({df['scoring_method'].iloc[0]} method)",
            "Final Score (with Citation Boost)",
            "Baseline Period Length (Years)",
        ),
        vertical_spacing=0.08,
        shared_xaxes=True,
    )

    # Base direction scores
    fig.add_trace(
        go.Scatter(
            x=final_df["year"],
            y=final_df["direction_score"],
            mode="lines+markers",
            name="Base Score",
            line=dict(color="#2E86AB", width=2),
            marker=dict(size=8),
            hovertemplate="<b>Year:</b> %{x}<br><b>Base Score:</b> %{y:.3f}<br>"
            + f"<b>Method:</b> {df['scoring_method'].iloc[0]}<extra></extra>",
        ),
        row=1,
        col=1,
    )

    # Final scores with citation boost
    colors = [
        "#A23B72" if support else "#2E86AB" for support in final_df["citation_support"]
    ]
    fig.add_trace(
        go.Scatter(
            x=final_df["year"],
            y=final_df["final_score"],
            mode="lines+markers",
            name="Final Score",
            line=dict(color="#A23B72", width=2),
            marker=dict(color=colors, size=8),
            hovertemplate="<b>Year:</b> %{x}<br><b>Final Score:</b> %{y:.3f}<br>"
            + "<b>Citation Support:</b> %{customdata}<extra></extra>",
            customdata=[
                "Yes" if support else "No" for support in final_df["citation_support"]
            ],
        ),
        row=2,
        col=1,
    )

    # Threshold line on final scores
    fig.add_hline(
        y=threshold,
        line_dash="dash",
        line_color="red",
        row=2,
        col=1,
        annotation_text=f"Threshold: {threshold:.3f}",
    )

    # Baseline period length
    fig.add_trace(
        go.Scatter(
            x=final_df["year"],
            y=final_df["baseline_period_length"],
            mode="lines+markers",
            name="Baseline Length",
            line=dict(color="#F18F01", width=2),
            marker=dict(size=6),
            hovertemplate="<b>Year:</b> %{x}<br><b>Baseline Length:</b> %{y} years<br>"
            + "<b>Period:</b> %{customdata}<extra></extra>",
            customdata=[
                f"{row['baseline_period_start']}-{row['baseline_period_end']}"
                for _, row in final_df.iterrows()
            ],
            showlegend=False,
        ),
        row=3,
        col=1,
    )

    # Mark detected boundaries
    boundary_years = (
        [ay.year for ay in st.session_state.boundary_years]
        if st.session_state.boundary_years
        else []
    )
    if boundary_years:
        boundary_scores = final_df[final_df["year"].isin(boundary_years)][
            "final_score"
        ].tolist()

        fig.add_trace(
            go.Scatter(
                x=boundary_years,
                y=boundary_scores,
                mode="markers",
                name="Detected Boundaries",
                marker=dict(color="red", size=12, symbol="diamond"),
                hovertemplate="<b>Boundary Year:</b> %{x}<br><b>Final Score:</b> %{y:.3f}<extra></extra>",
            ),
            row=2,
            col=1,
        )

    fig.update_layout(
        height=700,
        title_text="Streamlined Direction Change Detection Analysis",
        showlegend=True,
    )

    fig.update_xaxes(title_text="Year", row=3, col=1)
    fig.update_yaxes(title_text="Base Score", row=1, col=1)
    fig.update_yaxes(title_text="Final Score", row=2, col=1)
    fig.update_yaxes(title_text="Years", row=3, col=1)

    return fig


def create_citation_acceleration_chart(academic_years, citation_years=None):
    """Create citation acceleration visualization."""
    years = [ay.year for ay in academic_years]
    citations = [ay.total_citations for ay in academic_years]

    fig = make_subplots(
        rows=2,
        cols=1,
        subplot_titles=(
            "Citation Counts Over Time",
            "Citation Growth Rate (Year-over-Year %)",
        ),
        vertical_spacing=0.1,
    )

    # Raw citations
    fig.add_trace(
        go.Scatter(
            x=years,
            y=citations,
            mode="lines+markers",
            name="Citations",
            line=dict(color="#1f77b4", width=2),
            marker=dict(size=8),
            hovertemplate="<b>Year:</b> %{x}<br><b>Citations:</b> %{y}<extra></extra>",
        ),
        row=1,
        col=1,
    )

    # Smoothed citations (as used in algorithm)
    smoothing_window = 5
    if len(citations) >= smoothing_window:
        smoothed = moving_average(np.array(citations), smoothing_window)
        smoothed = np.pad(smoothed, (1, 1), mode="edge")[: len(citations)]

        fig.add_trace(
            go.Scatter(
                x=years,
                y=smoothed,
                mode="lines",
                name="Smoothed",
                line=dict(color="#ff7f0e", width=2, dash="dash"),
                hovertemplate="<b>Year:</b> %{x}<br><b>Smoothed:</b> %{y:.0f}<extra></extra>",
            ),
            row=1,
            col=1,
        )

    # Growth rate
    if len(citations) > 1:
        growth_rates = []
        growth_years = []
        for i in range(1, len(citations)):
            if citations[i - 1] > 0:
                growth_rate = (
                    (citations[i] - citations[i - 1]) / citations[i - 1]
                ) * 100
                growth_rates.append(growth_rate)
                growth_years.append(years[i])

        if growth_rates:
            fig.add_trace(
                go.Scatter(
                    x=growth_years,
                    y=growth_rates,
                    mode="lines+markers",
                    name="Growth Rate",
                    line=dict(color="#9467bd", width=2),
                    marker=dict(size=6),
                    hovertemplate="<b>Year:</b> %{x}<br><b>Growth:</b> %{y:.1f}%<extra></extra>",
                    showlegend=False,
                ),
                row=2,
                col=1,
            )

            # Add zero line
            fig.add_hline(y=0, line_dash="dot", line_color="gray", row=2, col=1)

    # Mark citation acceleration years
    if citation_years:
        acceleration_growth = []
        for year in citation_years:
            if year in growth_years:
                idx = growth_years.index(year)
                acceleration_growth.append(growth_rates[idx])
            else:
                acceleration_growth.append(0)

        fig.add_trace(
            go.Scatter(
                x=citation_years,
                y=acceleration_growth,
                mode="markers",
                name="Citation Acceleration",
                marker=dict(color="red", size=12, symbol="star"),
                hovertemplate="<b>Acceleration Year:</b> %{x}<br><b>Growth:</b> %{y:.1f}%<extra></extra>",
            ),
            row=2,
            col=1,
        )

    fig.update_layout(height=500, title_text="Citation Acceleration Analysis")

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
    years_with_data = df["year"].tolist()

    selected_year = st.selectbox("Select Year for Detailed Analysis", years_with_data)

    # Find year data
    year_data = df[df["year"] == selected_year].iloc[0]

    # Find corresponding academic year
    current_ay = next((ay for ay in academic_years if ay.year == selected_year), None)

    if current_ay:
        col1, col2 = st.columns(2)

        with col1:
            st.write(f"**Year {selected_year} Analysis:**")
            st.write(f"- Direction Score: {year_data['direction_score']:.3f}")
            st.write(f"- Scoring Method: {year_data['scoring_method']}")
            st.write(
                f"- Baseline Period: {year_data['baseline_period_start']}-{year_data['baseline_period_end']}"
            )
            st.write(f"- Baseline Length: {year_data['baseline_period_length']} years")

            st.write(f"**Keyword Counts:**")
            st.write(f"- Current Keywords: {year_data['current_keywords_count']}")
            st.write(f"- Baseline Keywords: {year_data['baseline_keywords_count']}")
            st.write(f"- New Keywords: {year_data['new_keywords_count']}")
            st.write(f"- Shared Keywords: {year_data['shared_keywords_count']}")

            st.write(f"**Traditional Metrics (for comparison):**")
            st.write(f"- Novelty: {year_data['novelty']:.3f}")
            st.write(f"- Overlap: {year_data['overlap']:.3f}")

        with col2:
            # Show top keywords for current year
            st.write(f"**Top Keywords for {selected_year}:**")
            top_keywords = current_ay.top_keywords[:15]
            for i, kw in enumerate(top_keywords, 1):
                st.write(f"{i}. {kw}")


def show_algorithm_explanation():
    """Show algorithm explanation with current implementation details."""
    with st.expander("📖 Algorithm Details", expanded=False):
        st.write(
            """
        **Current Streamlined Detection Algorithm:**
        
        **1. Citation Acceleration Detection:**
        - MAD-based year-over-year growth analysis
        - Smoothing window: 5 years to reduce noise  
        - Excludes recent 2 years (incomplete citations)
        - Threshold: median + 3.0 × MAD
        - Cooldown: 2 years to prevent clustering
        
        **2. Direction Change Detection:**
        - **Cumulative Baseline Approach**: Compares current year against accumulated baseline from last boundary
        - **Frequency-Weighted Scoring**: Uses actual keyword frequency distributions
        - **Scoring Methods**: 
          - `weighted_jaccard`: 1 - (intersection/union) of frequencies
          - `jensen_shannon`: Information divergence between distributions
        - **Adaptive Thresholding**: Based on score distribution (e.g., 90th percentile)
        
        **3. Immediate Citation Boost Integration:**
        - Direction scores boosted when citation acceleration occurs nearby
        - Final Score = Base Score + Citation Boost (if within support window)
        - Single threshold applied to final boosted scores
        - Support window: ±2 years, Boost factor: configurable
        """
        )

        st.write(
            """
        **Key Improvements over Traditional Methods:**
        - **No separate validation step**: Citation boost integrated immediately
        - **Cumulative baseline**: More stable than year-to-year comparisons
        - **Frequency weighting**: Accounts for publication volume differences
        - **Adaptive thresholds**: Adjusts to domain-specific score distributions
        - **Anti-clustering**: Prevents detection of spurious micro-segments
        """
        )


def show_change_detection():
    """Main change detection page function."""
    st.header("🔍 Stage 2: Change Point Detection")
    st.write(
        "Analyze paradigm shift detection using streamlined direction change and citation acceleration."
    )

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
            st.session_state.citation_acceleration_years = None
            run_change_detection()

    # Run detection if needed
    if not run_change_detection():
        return

    boundary_years = st.session_state.boundary_years
    citation_years = getattr(st.session_state, "citation_acceleration_years", [])

    # Results summary
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Boundary Years Detected", len(boundary_years))
    with col2:
        st.metric("Citation Acceleration Years", len(citation_years))
    with col3:
        detection_rate = (
            len(boundary_years) / len(academic_years) * 100 if academic_years else 0
        )
        st.metric("Detection Rate", f"{detection_rate:.1f}%")

    # Results lists
    if boundary_years:
        boundary_year_list = [ay.year for ay in boundary_years]
        st.write(f"**Detected Boundary Years:** {boundary_year_list}")

    if citation_years:
        st.write(f"**Citation Acceleration Years:** {citation_years}")

    # Direction change analysis
    st.subheader("📊 Streamlined Direction Change Analysis")
    direction_data = calculate_direction_signals_streamlined(
        academic_years, st.session_state.algorithm_config
    )

    if direction_data:
        direction_fig = create_direction_change_chart_streamlined(
            direction_data, citation_years, st.session_state.algorithm_config
        )
        if direction_fig:
            st.plotly_chart(direction_fig, use_container_width=True)

        # Statistics
        df = pd.DataFrame(direction_data)
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Avg Base Score", f"{df['direction_score'].mean():.3f}")
        with col2:
            st.metric("Max Base Score", f"{df['direction_score'].max():.3f}")
        with col3:
            st.metric("Scoring Method", df["scoring_method"].iloc[0])
        with col4:
            years_analyzed = len(df)
            st.metric("Years Analyzed", years_analyzed)

    # Citation acceleration analysis
    st.subheader("📈 Citation Acceleration Analysis")
    citation_fig = create_citation_acceleration_chart(academic_years, citation_years)
    st.plotly_chart(citation_fig, use_container_width=True)

    # Detailed analysis tabs
    tab1, tab2 = st.tabs(["📋 Detection Summary", "🔍 Year Analysis"])

    with tab1:
        if direction_data:
            # Direction change summary table
            df = pd.DataFrame(direction_data)

            # Show all years with their scores
            st.write("**Direction Change Analysis Results:**")
            display_df = df[
                [
                    "year",
                    "direction_score",
                    "baseline_period_start",
                    "baseline_period_end",
                    "baseline_period_length",
                ]
            ].copy()
            display_df["direction_score"] = display_df["direction_score"].round(3)
            display_df.columns = [
                "Year",
                "Score",
                "Baseline Start",
                "Baseline End",
                "Baseline Length",
            ]

            st.dataframe(display_df, use_container_width=True, hide_index=True)

    with tab2:
        show_detailed_year_analysis(direction_data, academic_years)

    # Algorithm explanation
    show_algorithm_explanation()
