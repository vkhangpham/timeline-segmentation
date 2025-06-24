"""
Analysis Overview Tab Components

Contains all visualization functions for the step-by-step algorithm validation process.
This corresponds to Tab 1 in the main dashboard.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from collections import defaultdict, Counter
from typing import Dict, List, Optional

from core.algorithm_config import ComprehensiveAlgorithmConfig
from core.data_models import DomainData


def _extract_segment_keywords(domain_data, start_year: int, end_year: int, top_n: int = 10) -> List[str]:
    """Extract top keywords for a specific time segment."""
    if domain_data is None or not hasattr(domain_data, 'papers') or not domain_data.papers:
        return []
    
    # Filter papers in the segment time range - domain_data is a DomainData object
    segment_papers = [
        paper for paper in domain_data.papers
        if paper.pub_year and start_year <= paper.pub_year <= end_year and paper.keywords
    ]
    
    if not segment_papers:
        return []
    
    # Count keyword frequencies
    keyword_counts = Counter()
    for paper in segment_papers:
        # Handle both tuple and string formats for keywords
        if isinstance(paper.keywords, tuple):
            keywords = paper.keywords
        elif isinstance(paper.keywords, str):
            keywords = [kw.strip() for kw in paper.keywords.split('|') if kw.strip()]
        else:
            keywords = []
            
        for keyword in keywords:
            if keyword:  # Skip empty keywords
                keyword_counts[keyword.lower()] += 1
    
    # Return top N keywords
    return [kw for kw, count in keyword_counts.most_common(top_n)]


def _calculate_keyword_differences(current_keywords: List[str], neighbor_keywords: List[str]) -> Dict[str, List[str]]:
    """Calculate keyword differences between current segment and a neighbor."""
    current_set = set(current_keywords)
    neighbor_set = set(neighbor_keywords)
    
    return {
        'unique_to_current': list(current_set - neighbor_set),
        'unique_to_neighbor': list(neighbor_set - current_set),
        'common': list(current_set & neighbor_set)
    }


def create_segment_keywords_table(segments: List[List[int]], domain_data) -> pd.DataFrame:
    """Create a comprehensive table showing keywords for each segment and differences between them."""
    if not segments or domain_data is None or not hasattr(domain_data, 'papers') or not domain_data.papers:
        return pd.DataFrame()
    
    # Extract keywords for all segments
    segment_data = []
    all_segment_keywords = {}
    
    for i, segment in enumerate(segments):
        start_year, end_year = segment[0], segment[1]
        duration = end_year - start_year + 1
        
        # Get top keywords for this segment
        top_keywords = _extract_segment_keywords(domain_data, start_year, end_year, top_n=10)
        all_segment_keywords[i] = top_keywords
        
        # Count papers in this segment
        papers_in_segment = [
            paper for paper in domain_data.papers
            if paper.pub_year and start_year <= paper.pub_year <= end_year
        ]
        
        # Format top keywords display
        if top_keywords:
            top_keywords_display = ", ".join(top_keywords[:6])
            if len(top_keywords) > 6:
                top_keywords_display += f" (+{len(top_keywords)-6} more)"
        else:
            top_keywords_display = "No keywords found"
        
        # Calculate keyword evolution compared to previous segment
        new_keywords = []
        disappeared_keywords = []
        
        if i > 0:  # Compare with previous segment
            prev_keywords = all_segment_keywords[i-1]
            diff_prev = _calculate_keyword_differences(top_keywords, prev_keywords)
            new_keywords = diff_prev['unique_to_current'][:5]  # Top 5 new keywords
            disappeared_keywords = diff_prev['unique_to_neighbor'][:5]  # Top 5 disappeared
        
        # Format evolution displays
        new_keywords_display = ", ".join(new_keywords) if new_keywords else "None"
        disappeared_keywords_display = ", ".join(disappeared_keywords) if disappeared_keywords else "None"
        
        segment_data.append({
            'Segment': f"Segment {i+1}",
            'Years': f"{start_year}-{end_year}",
            'Duration': f"{duration} years",
            'Papers': len(papers_in_segment),
            'Top Keywords': top_keywords_display,
            'New Keywords': new_keywords_display if i > 0 else "N/A (First segment)",
            'Disappeared Keywords': disappeared_keywords_display if i > 0 else "N/A (First segment)"
        })
    
    # Create DataFrame
    df = pd.DataFrame(segment_data)
    
    return df


def create_keyword_filtering_plot(
    domain_name: str,
    algorithm_config: ComprehensiveAlgorithmConfig,
    enhanced_signal_data: Dict,
) -> go.Figure:
    """Create Step 0: Keyword Filtering visualization."""
    fig = go.Figure()

    keyword_analysis = enhanced_signal_data.get("keyword_analysis", {})

    if algorithm_config.keyword_filtering_enabled:
        # Show keyword filtering activity from the logs
        # Extract filtering data from keyword_analysis if available
        filtering_data = keyword_analysis.get("filtering_activity", [])

        if filtering_data:
            filter_years = [d["year"] for d in filtering_data]
            original_counts = [d["original_count"] for d in filtering_data]
            filtered_counts = [d["filtered_count"] for d in filtering_data]
            retention_rates = [d["retention_rate"] for d in filtering_data]

            # Original keyword counts
            fig.add_trace(
                go.Scatter(
                    x=filter_years,
                    y=original_counts,
                    mode="lines+markers",
                    name="Original Keywords",
                    line=dict(color="#FF7F0E", width=2),
                    marker=dict(size=4, opacity=0.7),
                    hovertemplate="<b>Year:</b> %{x}<br><b>Original Keywords:</b> %{y}<extra></extra>",
                ),
            )

            # Filtered keyword counts
            fig.add_trace(
                go.Scatter(
                    x=filter_years,
                    y=filtered_counts,
                    mode="lines+markers",
                    name="Filtered Keywords",
                    line=dict(color="#2CA02C", width=2),
                    marker=dict(size=4, opacity=0.7),
                    hovertemplate="<b>Year:</b> %{x}<br><b>Filtered Keywords:</b> %{y}<br><b>Retention:</b> %{customdata:.1%}<extra></extra>",
                    customdata=retention_rates,
                ),
            )

            # Add filtering efficiency annotation
            avg_retention = np.mean(retention_rates) if retention_rates else 1.0
            fig.add_annotation(
                xref="paper",
                yref="paper",
                x=0.02,
                y=0.98,
                text=f"🔍 Keyword Filtering: {avg_retention:.1%} average retention<br>"
                f"Configuration: {algorithm_config.keyword_min_papers_ratio:.1%} min ratio, ",
                showarrow=False,
                font=dict(size=10),
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor="gray",
                borderwidth=1,
            )
        else:
            # Show that filtering is enabled but no data available
            fig.add_annotation(
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                text="🔍 Keyword Filtering: ENABLED<br>"
                f"Configuration: {algorithm_config.keyword_min_papers_ratio:.1%} min ratio, "
                "(Filtering activity data not available in visualization)",
                showarrow=False,
                font=dict(size=12),
                bgcolor="rgba(173,216,230,0.8)",
                bordercolor="blue",
                borderwidth=1,
            )
    else:
        # Show that filtering is disabled
        fig.add_annotation(
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            text="🔍 Keyword Filtering: DISABLED<br>"
            "All keywords used without filtering",
            showarrow=False,
            font=dict(size=12),
            bgcolor="rgba(248,249,250,0.8)",
            bordercolor="gray",
            borderwidth=1,
        )

    fig.update_layout(
        xaxis_title="Year",
        yaxis_title="Keyword Count",
        height=400,
        showlegend=True,
        template="plotly_white",
    )

    return fig


def create_direction_signal_detection_plot(
    domain_name: str,
    algorithm_config: ComprehensiveAlgorithmConfig,
    enhanced_signal_data: Dict,
) -> go.Figure:
    """Create Step 1: Raw Direction Signal Detection visualization."""
    fig = go.Figure()

    keyword_analysis = enhanced_signal_data.get("keyword_analysis", {})

    if keyword_analysis and keyword_analysis.get("years"):
        all_years = keyword_analysis["years"]
        all_scores = keyword_analysis["direction_score"]
        detection_threshold = keyword_analysis["threshold"]
        novel_keywords_data = keyword_analysis.get("novel_keywords", [])
        top_keywords_data = keyword_analysis.get("top_keywords", [])

        # Define a professional color palette for Step 1
        line_color = "#636E72"  # Soothing dark gray for the base line
        marker_color = "#636E72"  # Same as line color for consistency
        highlight_color = "#E17055"  # Professional orange/salmon for detected signals
        threshold_color = "#D73027"  # Consistent professional red for thresholds

        # Create enhanced hover text with keyword information
        hover_texts = []
        for i, (year, score) in enumerate(zip(all_years, all_scores)):
            # Get keyword data for this year
            novel_keywords = (
                novel_keywords_data[i] if i < len(novel_keywords_data) else []
            )
            top_keywords = top_keywords_data[i] if i < len(top_keywords_data) else []

            # Format novel keywords
            novel_text = ", ".join(novel_keywords[:5]) if novel_keywords else "None"
            if len(novel_keywords) > 5:
                novel_text += f" (+{len(novel_keywords)-5} more)"

            # Format top keywords
            top_text = ", ".join(top_keywords[:5]) if top_keywords else "None"
            if len(top_keywords) > 5:
                top_text += f" (+{len(top_keywords)-5} more)"

            # Add filtering info if enabled
            filtering_note = ""
            if algorithm_config.keyword_filtering_enabled:
                filtering_note = (
                    "<br><b>Filtering:</b> Applied with conservative settings"
                )

            hover_text = (
                f"<b>Year:</b> {year}<br>"
                f"<b>Direction Score:</b> {score:.3f}<br>"
                f"<b>Novel Keywords:</b> {novel_text}<br>"
                f"<b>Top Keywords:</b> {top_text}{filtering_note}"
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
                novel_keywords = (
                    novel_keywords_data[i] if i < len(novel_keywords_data) else []
                )
                top_keywords = (
                    top_keywords_data[i] if i < len(top_keywords_data) else []
                )

                # Format novel keywords
                novel_text = ", ".join(novel_keywords[:5]) if novel_keywords else "None"
                if len(novel_keywords) > 5:
                    novel_text += f" (+{len(novel_keywords)-5} more)"

                # Format top keywords
                top_text = ", ".join(top_keywords[:5]) if top_keywords else "None"
                if len(top_keywords) > 5:
                    top_text += f" (+{len(top_keywords)-5} more)"

                # Add filtering info if enabled
                filtering_note = ""
                if algorithm_config.keyword_filtering_enabled:
                    filtering_note = "<br><b>Filtering:</b> Applied - keywords passed conservative filter"

                detected_hover_text = (
                    f"<b>Year:</b> {year}<br>"
                    f"<b>Direction Score:</b> {score:.3f}<br>"
                    f"<b>Status:</b> Detected<br>"
                    f"<b>Novel Keywords:</b> {novel_text}<br>"
                    f"<b>Top Keywords:</b> {top_text}{filtering_note}"
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
            )

        # Detection threshold line with improved label placement
        fig.add_hline(
            y=detection_threshold,
            line=dict(color=threshold_color, width=2, dash="dash"),
            annotation_text=f"Detection Threshold ({detection_threshold:.2f})",
            annotation_position="top left",
        )

    fig.update_layout(
        xaxis_title="Year",
        yaxis_title="Direction Score",
        height=400,
        showlegend=True,
        template="plotly_white",
    )

    return fig


def create_citation_signal_detection_plot(
    domain_name: str,
    domain_data: DomainData,
    actual_citation_signals: Optional[List] = None,
) -> go.Figure:
    """Create Step 2: Citation Signal Detection visualization.
    
    TRANSPARENCY FIX: Now shows actual algorithm-detected citation signals
    instead of using a different threshold method than the algorithm.
    """
    fig = go.Figure()

    # Extract citation time series from domain data
    citation_series = defaultdict(float)
    for paper in domain_data.papers:
        if paper.pub_year and paper.cited_by_count:
            citation_series[paper.pub_year] += paper.cited_by_count

    if citation_series:
        # Get sorted years and citation counts
        sorted_years = sorted(citation_series.keys())
        citation_counts = [citation_series[year] for year in sorted_years]

        # Professional color scheme
        line_color = "#636E72"  # Soothing dark gray for the base line
        marker_color = "#636E72"  # Same as line color for consistency
        highlight_color = "#FFF287"  # Bright yellow for highlighted points
        
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
        )

        # TRANSPARENCY FIX: Show actual algorithm-detected citation signals
        if actual_citation_signals:
            detected_years = []
            detected_counts = []
            
            # Get citation data for each detected signal year
            for citation_signal in actual_citation_signals:
                year = int(citation_signal.year) if hasattr(citation_signal, 'year') else citation_signal
                if year in citation_series:
                    detected_years.append(year)
                    detected_counts.append(citation_series[year])
            
            if detected_years:
                fig.add_trace(
                    go.Scatter(
                        x=detected_years,
                        y=detected_counts,
                        mode="markers",
                        name="Algorithm Detected",
                        marker=dict(
                            size=12,
                            color=highlight_color,
                            symbol="circle",
                            opacity=1.0,
                            line=dict(width=2, color=line_color),
                        ),
                        hovertemplate="<b>Year:</b> %{x}<br><b>Citations:</b> %{y:,.0f}<br><b>Status:</b> Algorithm Detected Citation Signal<extra></extra>",
                    ),
                )
                
                # Add note about algorithm method
                fig.add_annotation(
                    x=0.02, y=0.98,
                    xref="paper", yref="paper",
                    text="🔬 Shows citation signals detected by gradient-based CPSD algorithm",
                    showarrow=False,
                    font=dict(size=10, color="darkblue"),
                    bgcolor="rgba(255,255,255,0.8)",
                    bordercolor="darkblue",
                    borderwidth=1
                )
        else:
            # Fallback: Calculate adaptive threshold for comparison (but note the limitation)
            mean_citations = np.mean(citation_counts)
            std_citations = np.std(citation_counts)
            adaptive_threshold = mean_citations + 1.5 * std_citations

            # Add adaptive threshold line with warning
            fig.add_hline(
                y=adaptive_threshold,
                line=dict(color="#D73027", width=2, dash="dash"),
                annotation_text=f"⚠️ Statistical Threshold ({adaptive_threshold:,.0f}) - NOT algorithm method",
                annotation_position="top left",
            )

            # Show statistical threshold points with warning
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
                        name="⚠️ Statistical Only",
                        marker=dict(
                            size=12,
                            color="#FFA500",  # Orange to indicate warning
                            symbol="diamond",
                            opacity=0.7,
                            line=dict(width=2, color=line_color),
                        ),
                        hovertemplate="<b>Year:</b> %{x}<br><b>Citations:</b> %{y:,.0f}<br><b>Status:</b> ⚠️ Statistical threshold only - NOT algorithm detected<extra></extra>",
                    ),
                )
            
            # Add warning about transparency issue
            fig.add_annotation(
                x=0.02, y=0.98,
                xref="paper", yref="paper",
                text="⚠️ TRANSPARENCY ISSUE: Algorithm citation signals not provided to visualization",
                showarrow=False,
                font=dict(size=10, color="red"),
                bgcolor="rgba(255,255,255,0.9)",
                bordercolor="red",
                borderwidth=1
            )

    fig.update_layout(
        xaxis_title="Year",
        yaxis_title="Total Citations",
        height=400,
        showlegend=True,
        template="plotly_white",
    )

    return fig


def create_final_validation_plot(
    domain_name: str,
    validated_signals: List,
    enhanced_signal_data: Dict,
    algorithm_config: ComprehensiveAlgorithmConfig,
) -> go.Figure:
    """Create Step 3: Final Validation with Consistent Threshold visualization."""
    fig = go.Figure()

    clustering_metadata = enhanced_signal_data.get("clustering_metadata", {})
    citation_signals = clustering_metadata.get("citation_signals", [])

    # Show ALL direction signals (both validated and filtered) for complete picture
    all_direction_years = []
    all_direction_confidences = []
    all_direction_status = []
    all_direction_hover = []

    # Get all processed direction signals from metadata (these went through validation)
    clustered_direction_signals = clustering_metadata.get(
        "processed_direction_signals", []
    )
    validated_years = {s.year for s in validated_signals}

    for direction_signal in clustered_direction_signals:
        year = direction_signal.year
        base_confidence = direction_signal.confidence

        # Check if this signal was validated
        is_validated = year in validated_years

        if is_validated:
            # Find the validated signal to get final confidence
            validated_signal = next(
                (s for s in validated_signals if s.year == year), None
            )
            final_confidence = (
                validated_signal.confidence if validated_signal else base_confidence
            )
            status = "Validated"
            color = "green"
            symbol = "circle"
        else:
            # Calculate what the final confidence would have been (for display)
            # Check for citation support
            citation_support = any(abs(cs.year - year) <= 2 for cs in citation_signals)

            confidence_boosts = 0.0
            if citation_support:
                confidence_boosts += (0.5 * base_confidence)  # Dynamic boost: 50% of base confidence

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
            dynamic_boost = 0.5 * base_confidence  # Calculate actual boost amount
            boost_info.append(f"Citation: +{dynamic_boost:.3f} (50% of base)")

        if boost_info:
            hover_text += f"<b>Boosts:</b> {', '.join(boost_info)}<br>"

        hover_text += f"<b>Final Confidence:</b> {final_confidence:.3f}<br>"
        hover_text += (
            f"<b>Threshold:</b> {algorithm_config.validation_threshold:.3f}<br>"
        )
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
            )

    # Validation threshold line
    fig.add_hline(
        y=algorithm_config.validation_threshold,
        line=dict(color="red", width=2, dash="dash"),
        annotation_text=f"Validation Threshold ({algorithm_config.validation_threshold:.2f})",
        annotation_position="top left",
    )

    fig.update_layout(
        xaxis_title="Year",
        yaxis_title="Final Confidence",
        height=400,
        showlegend=True,
        template="plotly_white",
    )

    return fig


def create_similarity_segmentation_plot(
    domain_name: str,
    validated_signals: List,
    enhanced_signal_data: Dict,
    algorithm_config: ComprehensiveAlgorithmConfig,
    segments: Optional[List[List[int]]] = None,
    domain_data: Optional[DomainData] = None,
) -> go.Figure:
    """Create Step 4: Similarity-Based Segmentation visualization (now default approach)."""
    fig = go.Figure()

    # Similarity segmentation is now the default and only approach
    # Get segmentation data from enhanced_signal_data and passed parameter
    similarity_metadata = enhanced_signal_data.get("similarity_metadata", {})
    # Use passed segments parameter, fallback to enhanced_signal_data, then empty list
    if segments is None:
        segments = enhanced_signal_data.get("segments", [])

    if segments and len(segments) > 0 and validated_signals:
        # Show the segmentation transformation process
        signal_years = [signal.year for signal in validated_signals]
        signal_confidences = [signal.confidence for signal in validated_signals]

        # 1. Plot original validated signals as the base data
        fig.add_trace(
            go.Scatter(
                x=signal_years,
                y=[0.3] * len(signal_years),  # Position at 0.3 on y-axis
                mode="markers",
                marker=dict(
                    size=12,
                    color=signal_confidences,
                    colorscale="Reds",
                    symbol="circle",
                    line=dict(width=2, color="black"),
                    colorbar=dict(
                        title="Signal Confidence",
                        x=1.02,
                        y=0.2,
                        len=0.4,
                        thickness=10,
                    ),
                ),
                name="Original Validated Signals",
                hovertemplate="<b>Validated Signal</b><br>"
                "Year: %{x}<br>"
                "Confidence: %{marker.color:.3f}<br>"
                "<extra></extra>",
                showlegend=True,
            ),
        )

        # 2. Show segment boundaries and centroids
        segment_colors = [
            "#3498DB",
            "#27AE60",
            "#F39C12",
            "#9B59B6",
            "#E67E22",
            "#1ABC9C",
        ]

        # Extract keyword data for all segments if domain_data is available
        segment_keywords = {}
        if domain_data is not None and hasattr(domain_data, 'papers') and domain_data.papers:
            for i, segment in enumerate(segments):
                start_year, end_year = segment[0], segment[1]
                segment_keywords[i] = _extract_segment_keywords(domain_data, start_year, end_year, top_n=8)

        for i, segment in enumerate(segments):
            start_year, end_year = segment[0], segment[1]
            color = segment_colors[i % len(segment_colors)]

            # Add segment background rectangle
            fig.add_vrect(
                x0=start_year,
                x1=end_year,
                fillcolor=f"rgba({int(color[1:3], 16)}, {int(color[3:5], 16)}, {int(color[5:7], 16)}, 0.15)",
                layer="below",
                line_width=0,
            )

            # Find which validated signal(s) fall within this segment
            segment_signals = [
                sig
                for sig in validated_signals
                if start_year <= sig.year <= end_year
            ]

            if segment_signals:
                # Calculate centroid (weighted by confidence)
                total_confidence = sum(sig.confidence for sig in segment_signals)
                if total_confidence > 0:
                    centroid_year = (
                        sum(sig.year * sig.confidence for sig in segment_signals)
                        / total_confidence
                    )
                else:
                    centroid_year = sum(sig.year for sig in segment_signals) / len(
                        segment_signals
                    )

                # Build hover template with keyword information
                hover_template = (
                    f"<b>Segment {i+1} Centroid</b><br>"
                    f"Centroid Year: {centroid_year:.1f}<br>"
                    f"Range: {start_year}-{end_year}<br>"
                    f"Duration: {end_year - start_year + 1} years<br>"
                    f"Signals in segment: {len(segment_signals)}<br>"
                    f"Signal years: {[sig.year for sig in segment_signals]}<br>"
                )

                # Add keyword information if available
                if domain_data is not None and hasattr(domain_data, 'papers') and domain_data.papers and i in segment_keywords:
                    current_keywords = segment_keywords[i]
                    
                    # Add current segment keywords
                    if current_keywords:
                        keywords_text = ", ".join(current_keywords[:6])  # Show top 6 keywords
                        if len(current_keywords) > 6:
                            keywords_text += f" (+{len(current_keywords)-6} more)"
                        hover_template += f"<br><br><b>Top Keywords:</b><br>{keywords_text}<br>"
                    else:
                        hover_template += f"<br><br><b>Top Keywords:</b> No keywords found<br>"
                    
                    # Add keyword differences with neighbors
                    neighbor_info = []
                    
                    # Previous segment comparison
                    if i > 0 and (i-1) in segment_keywords:
                        prev_keywords = segment_keywords[i-1]
                        diff_prev = _calculate_keyword_differences(current_keywords, prev_keywords)
                        if diff_prev['unique_to_current']:
                            unique_text = ", ".join(diff_prev['unique_to_current'][:4])
                            if len(diff_prev['unique_to_current']) > 4:
                                unique_text += f" (+{len(diff_prev['unique_to_current'])-4})"
                            neighbor_info.append(f"New vs Prev: {unique_text}")
                    
                    # Next segment comparison
                    if i < len(segments) - 1 and (i+1) in segment_keywords:
                        next_keywords = segment_keywords[i+1]
                        diff_next = _calculate_keyword_differences(current_keywords, next_keywords)
                        if diff_next['unique_to_current']:
                            unique_text = ", ".join(diff_next['unique_to_current'][:4])
                            if len(diff_next['unique_to_current']) > 4:
                                unique_text += f" (+{len(diff_next['unique_to_current'])-4})"
                            neighbor_info.append(f"New vs Next: {unique_text}")
                    
                    if neighbor_info:
                        hover_template += f"<br><b>Keyword Differences:</b><br>" + "<br>".join(neighbor_info)
                    elif len(segments) > 1:
                        hover_template += f"<br><b>Keyword Differences:</b> Similar to neighbors"

                hover_template += "<extra></extra>"

                # Add centroid marker
                fig.add_trace(
                    go.Scatter(
                        x=[centroid_year],
                        y=[0.7],  # Position at 0.7 on y-axis
                        mode="markers+text",
                        marker=dict(
                            size=20,
                            color=color,
                            symbol="diamond",
                            line=dict(width=2, color="black"),
                        ),
                        text=f"C{i+1}",
                        textposition="middle center",
                        textfont=dict(color="white", size=10, family="Arial Black"),
                        name=f"Segment {i+1} Centroid",
                        hovertemplate=hover_template,
                        showlegend=False,
                    ),
                )

                # Draw arrows from original signals to centroid
                for sig in segment_signals:
                    fig.add_annotation(
                        x=sig.year,
                        y=0.3,
                        ax=centroid_year,
                        ay=0.7,
                        xref="x",
                        yref="y",
                        axref="x",
                        ayref="y",
                        arrowhead=2,
                        arrowsize=1,
                        arrowwidth=2,
                        arrowcolor=color,
                        opacity=0.7,
                    )

            # Add boundary markers
            if i < len(segments) - 1:  # Don't add boundary after last segment
                boundary_year = end_year
                fig.add_vline(
                    x=boundary_year,
                    line=dict(color="red", width=2, dash="dot"),
                    annotation_text=f"Boundary {boundary_year}",
                    annotation_position="top",
                    annotation_font=dict(size=8, color="red"),
                )

        # 3. Add merging information if minimum segment length was enforced
        min_length = algorithm_config.similarity_min_segment_length
        if min_length > 0:
            merged_segments = [
                seg for seg in segments if seg[1] - seg[0] + 1 >= min_length
            ]
            if len(merged_segments) < len(segments):
                # Some merging occurred
                fig.add_annotation(
                    xref="paper",
                    yref="paper",
                    x=0.02,
                    y=0.95,
                    text=f"⚙️ Merging Applied: {len(segments) - len(merged_segments)} segment(s) merged to meet {min_length}yr minimum",
                    showarrow=False,
                    font=dict(size=10),
                    bgcolor="rgba(255, 152, 0, 0.1)",
                    bordercolor="#FF9800",
                    borderwidth=1,
                )

        # 4. Add process explanation
        process_note = f"Process: {len(validated_signals)} signals → {len(segments)} segments via similarity boundaries"
        if len(validated_signals) > len(segments):
            process_note += f" (signals preserved as centroids)"

        fig.add_annotation(
            xref="paper",
            yref="paper",
            x=0.02,
            y=0.02,
            text=process_note,
            showarrow=False,
            font=dict(size=10),
            bgcolor="rgba(52, 152, 219, 0.1)",
            bordercolor="#3498DB",
            borderwidth=1,
        )
    else:
        # Show message when no segments are available
        fig.add_annotation(
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            text="🎯 Similarity-Based Segmentation (Default Approach)<br><br>"
            "No validated signals available for segmentation visualization.<br>"
            "Adjust algorithm parameters to detect paradigm shift signals.",
            showarrow=False,
            font=dict(size=12),
            bgcolor="rgba(173,216,230,0.8)",
            bordercolor="blue",
            borderwidth=1,
        )

    fig.update_layout(
        xaxis_title="Year",
        yaxis_title="Similarity Segmentation Process",
        yaxis=dict(
            range=[0, 1],
            showticklabels=False,
            tickmode="array",
            tickvals=[0.3, 0.7],
            ticktext=["Original Signals", "Segment Centroids"],
        ),
        height=400,
        showlegend=True,
        template="plotly_white",
    )

    return fig 