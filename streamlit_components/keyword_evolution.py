"""
Keyword Evolution Tab Components

Contains all visualization functions for keyword analysis and evolution.
This corresponds to Tab 2 in the main dashboard.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from collections import defaultdict, Counter
from typing import Dict, List, Optional

from core.algorithm_config import ComprehensiveAlgorithmConfig
from core.data_models import DomainData


def prepare_keyword_evolution_data(
    domain_data: DomainData,
    top_n: int = 12,
    algorithm_config: Optional[ComprehensiveAlgorithmConfig] = None,
    enhanced_signal_data: Optional[Dict] = None,
) -> pd.DataFrame:
    """Prepares data for keyword evolution visualizations with optional keyword filtering."""
    keyword_counts_by_year = defaultdict(Counter)

    # Check if we have filtered keyword data available
    filtered_keywords_available = False
    filtering_activity = {}

    if (
        algorithm_config
        and algorithm_config.keyword_filtering_enabled
        and enhanced_signal_data
        and enhanced_signal_data.get("keyword_analysis", {}).get("filtering_activity")
    ):
        filtering_activity = {
            fa["year"]: fa
            for fa in enhanced_signal_data["keyword_analysis"]["filtering_activity"]
        }
        filtered_keywords_available = True

    for paper in domain_data.papers:
        if paper.pub_year and paper.keywords:
            keywords_to_use = paper.keywords

            # Apply filtering if we have filtered data and filtering is enabled
            if (
                filtered_keywords_available
                and algorithm_config.keyword_filtering_enabled
                and paper.pub_year in filtering_activity
            ):
                # For visualization, we show the effect of filtering by using filtered keywords
                # Note: This is a simplified approach - ideally we'd have the exact filtered keywords per paper
                # For now, we'll use the raw keywords but weight them by retention rate
                retention_rate = filtering_activity[paper.pub_year]["retention_rate"]
                # Apply a sampling based on retention rate for visualization purposes
                if retention_rate < 1.0:
                    # Simulate filtering effect by reducing keyword weights
                    keyword_weights = {}
                    for kw in keywords_to_use:
                        # Keywords that would likely be filtered have lower weights
                        keyword_weights[kw] = retention_rate

                    # Update with weighted keywords (simplified approach)
                    for kw, weight in keyword_weights.items():
                        keyword_counts_by_year[paper.pub_year][kw] += weight
                else:
                    keyword_counts_by_year[paper.pub_year].update(keywords_to_use)
            else:
                keyword_counts_by_year[paper.pub_year].update(keywords_to_use)

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


def create_keyword_streamgraph(
    df_keywords: pd.DataFrame, domain_name: str
) -> go.Figure:
    """Creates a streamgraph of keyword evolution using a professional palette."""
    if df_keywords.empty:
        return go.Figure()

    fig = go.Figure()

    # Use a professional, colorblind-friendly palette from Plotly
    colors = px.colors.qualitative.Plotly

    for i, keyword in enumerate(df_keywords.columns):
        fig.add_trace(
            go.Scatter(
                x=df_keywords.index,
                y=df_keywords[keyword],
                mode="lines",
                line=dict(width=0.5, color=colors[i % len(colors)]),
                stackgroup="one",  # Key property for creating a streamgraph
                name=keyword,
                hovertemplate=f"<b>Keyword:</b> {keyword}<br>"
                + "<b>Year:</b> %{x}<br>"
                + "<b>Prominence:</b> %{y:.2f}<extra></extra>",
            )
        )

    fig.update_layout(
        title=f"Keyword Evolution Streamgraph: {domain_name.replace('_', ' ').title()}",
        xaxis_title="Year",
        yaxis_title="Smoothed Prominence",
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=500,
        template="plotly_white",
    )
    return fig


def create_enhanced_keyword_heatmap(
    df_keywords: pd.DataFrame, domain_name: str, shift_signals: List
) -> go.Figure:
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
        title=f"Keyword Prominence Heatmap: {domain_name.replace('_', ' ').title()}",
    )

    # Add paradigm shift markers if available
    if shift_signals:
        shift_years = [signal.year for signal in shift_signals]
        
        # Add vertical lines for paradigm shifts
        for year in shift_years:
            fig.add_vline(
                x=year,
                line=dict(color="red", width=2, dash="dash"),
                opacity=0.7,
            )
        
        # Add annotation for paradigm shifts
        fig.add_annotation(
            text=f"Red lines indicate {len(shift_years)} paradigm shifts",
            xref="paper", yref="paper",
            x=0.02, y=0.98,
            showarrow=False,
            font=dict(size=10, color="red"),
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="red",
            borderwidth=1,
        )

    fig.update_layout(
        xaxis_title="Year",
        yaxis_title="Keywords",
        height=500,
        font=dict(size=12),
    )

    fig.update_xaxes(tickmode="linear", dtick=10)
    return fig 