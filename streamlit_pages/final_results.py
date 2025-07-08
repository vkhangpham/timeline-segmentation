"""Final Results Page
Displays the complete timeline analysis result with comprehensive visualizations.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from core.data.data_models import TimelineAnalysisResult


def create_final_timeline_result():
    """Create the final TimelineAnalysisResult object."""
    if st.session_state.final_periods is None:
        return None
    
    periods = st.session_state.final_periods
    academic_years = st.session_state.academic_years
    
    # Calculate boundary years from periods
    boundary_years = []
    for i in range(len(periods) - 1):
        boundary_years.append(periods[i].end_year)
    
    # Calculate overall confidence
    if periods:
        confidence = sum(p.confidence for p in periods) / len(periods)
    else:
        confidence = 0.0
    
    # Generate narrative evolution
    narrative_parts = []
    for i, period in enumerate(periods):
        topic = period.topic_label or f"Period {i+1}"
        years = f"{period.start_year}-{period.end_year}"
        narrative_parts.append(f"Period {i+1} ({years}): {topic}")
    
    narrative_evolution = " → ".join(narrative_parts)
    
    # Create the result object
    result = TimelineAnalysisResult(
        domain_name=st.session_state.selected_domain,
        periods=tuple(periods),
        confidence=confidence,
        boundary_years=tuple(boundary_years),
        narrative_evolution=narrative_evolution
    )
    
    return result


def create_comprehensive_timeline_chart(result):
    """Create comprehensive timeline visualization."""
    if not result or not result.periods:
        return None
    
    # Create the main timeline chart
    fig = make_subplots(
        rows=4, cols=1,
        subplot_titles=(
            "Academic Timeline Evolution",
            "Papers per Year",
            "Citations per Year",
            "Confidence Scores by Period"
        ),
        vertical_spacing=0.08,
        shared_xaxes=True,
        row_heights=[0.3, 0.25, 0.25, 0.2]
    )
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
    
    # Timeline visualization (row 1)
    for i, period in enumerate(result.periods):
        color = colors[i % len(colors)]
        
        # Main period line
        fig.add_trace(
            go.Scatter(
                x=[period.start_year, period.end_year],
                y=[i, i],
                mode='lines+markers',
                name=f"Period {i+1}",
                line=dict(color=color, width=8),
                marker=dict(size=12),
                hovertemplate=f"<b>Period {i+1}</b><br>" +
                            f"Years: {period.start_year}-{period.end_year}<br>" +
                            f"Topic: {period.topic_label or 'N/A'}<br>" +
                            f"Papers: {period.total_papers}<br>" +
                            f"Confidence: {period.confidence:.3f}<extra></extra>",
                showlegend=False
            ),
            row=1, col=1
        )
        
        # Add topic labels
        mid_year = (period.start_year + period.end_year) / 2
        topic_text = period.topic_label[:30] + "..." if period.topic_label and len(period.topic_label) > 30 else period.topic_label or f"Period {i+1}"
        
        fig.add_annotation(
            x=mid_year,
            y=i + 0.3,
            text=topic_text,
            showarrow=False,
            font=dict(size=10, color=color),
            row=1, col=1
        )
    
    # Boundary markers
    for boundary_year in result.boundary_years:
        fig.add_vline(
            x=boundary_year,
            line_dash="dash",
            line_color="red",
            opacity=0.7,
            annotation_text=f"Boundary: {boundary_year}",
            annotation_position="top"
        )
    
    # Papers per year (row 2)
    academic_years = st.session_state.academic_years
    years = [ay.year for ay in academic_years]
    papers = [ay.paper_count for ay in academic_years]
    
    for i, period in enumerate(result.periods):
        color = colors[i % len(colors)]
        period_years = [y for y in years if period.start_year <= y <= period.end_year]
        period_papers = [papers[years.index(y)] for y in period_years]
        
        fig.add_trace(
            go.Scatter(
                x=period_years,
                y=period_papers,
                mode='lines+markers',
                name=f"Period {i+1}",
                line=dict(color=color, width=3),
                marker=dict(size=6),
                hovertemplate=f"<b>Period {i+1}</b><br>Year: %{{x}}<br>Papers: %{{y}}<extra></extra>",
                showlegend=False
            ),
            row=2, col=1
        )
    
    # Citations per year (row 3)
    citations = [ay.total_citations for ay in academic_years]
    
    for i, period in enumerate(result.periods):
        color = colors[i % len(colors)]
        period_years = [y for y in years if period.start_year <= y <= period.end_year]
        period_citations = [citations[years.index(y)] for y in period_years]
        
        fig.add_trace(
            go.Scatter(
                x=period_years,
                y=period_citations,
                mode='lines+markers',
                name=f"Period {i+1}",
                line=dict(color=color, width=3),
                marker=dict(size=6),
                hovertemplate=f"<b>Period {i+1}</b><br>Year: %{{x}}<br>Citations: %{{y}}<extra></extra>",
                showlegend=False
            ),
            row=3, col=1
        )
    
    # Confidence scores (row 4)
    period_names = [f"P{i+1}" for i in range(len(result.periods))]
    confidence_scores = [p.confidence for p in result.periods]
    
    fig.add_trace(
        go.Bar(
            x=period_names,
            y=confidence_scores,
            name='Confidence',
            marker_color=[colors[i % len(colors)] for i in range(len(result.periods))],
            hovertemplate='<b>%{x}</b><br>Confidence: %{y:.3f}<extra></extra>',
            showlegend=False
        ),
        row=4, col=1
    )
    
    fig.update_layout(
        height=800,
        title_text=f"Complete Timeline Analysis: {result.domain_name.replace('_', ' ').title()}"
    )
    
    fig.update_xaxes(title_text="Year", row=4, col=1)
    fig.update_yaxes(title_text="Periods", row=1, col=1, showticklabels=False)
    fig.update_yaxes(title_text="Papers", row=2, col=1)
    fig.update_yaxes(title_text="Citations", row=3, col=1)
    fig.update_yaxes(title_text="Confidence", row=4, col=1)
    
    return fig


def create_period_evolution_sankey():
    """Create Sankey diagram showing topic evolution."""
    if not st.session_state.final_periods:
        return None
    
    periods = st.session_state.final_periods
    
    # Prepare data for Sankey
    labels = []
    source = []
    target = []
    values = []
    
    # Add period labels
    for i, period in enumerate(periods):
        topic = period.topic_label or f"Period {i+1}"
        labels.append(f"P{i+1}: {topic[:20]}...")
    
    # Create connections between adjacent periods
    for i in range(len(periods) - 1):
        source.append(i)
        target.append(i + 1)
        # Use overlap in keywords as connection strength
        keywords_current = set(periods[i].top_keywords[:20])
        keywords_next = set(periods[i+1].top_keywords[:20])
        overlap = len(keywords_current & keywords_next)
        values.append(max(overlap, 1))  # Minimum value of 1
    
    if not source:  # Only one period
        return None
    
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=labels,
            color="lightblue"
        ),
        link=dict(
            source=source,
            target=target,
            value=values,
            color="lightgray"
        )
    )])
    
    fig.update_layout(
        title_text="Topic Evolution Flow",
        font_size=10,
        height=400
    )
    
    return fig


def create_summary_statistics_table(result):
    """Create summary statistics table."""
    if not result or not result.periods:
        return None
    
    summary_data = []
    for i, period in enumerate(result.periods):
        summary_data.append({
            'Period': i + 1,
            'Years': f"{period.start_year}-{period.end_year}",
            'Duration': period.end_year - period.start_year + 1,
            'Topic Label': period.topic_label or "N/A",
            'Papers': period.total_papers,
            'Citations': period.total_citations,
            'Confidence': f"{period.confidence:.3f}",
            'Top Keywords': ', '.join(period.top_keywords[:5]) + ("..." if len(period.top_keywords) > 5 else "")
        })
    
    return pd.DataFrame(summary_data)


def show_algorithm_performance_summary():
    """Show overall algorithm performance summary."""
    st.subheader("⚡ Algorithm Performance Summary")
    
    if st.session_state.timing_data:
        # Create performance visualization
        stages = list(st.session_state.timing_data.keys())
        times = list(st.session_state.timing_data.values())
        
        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                x=stages,
                y=times,
                marker_color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'],
                hovertemplate='<b>%{x}</b><br>Time: %{y:.2f}s<extra></extra>'
            )
        )
        
        fig.update_layout(
            title="Processing Time by Stage",
            xaxis_title="Algorithm Stage",
            yaxis_title="Time (seconds)",
            height=300
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Performance metrics
        total_time = sum(times)
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Processing Time", f"{total_time:.2f}s")
        with col2:
            longest_stage = max(st.session_state.timing_data.items(), key=lambda x: x[1])
            st.metric("Longest Stage", f"{longest_stage[0]}: {longest_stage[1]:.2f}s")
        with col3:
            if st.session_state.final_periods:
                periods_per_second = len(st.session_state.final_periods) / total_time
                st.metric("Periods/Second", f"{periods_per_second:.2f}")
        with col4:
            if st.session_state.academic_years:
                papers_processed = sum(ay.paper_count for ay in st.session_state.academic_years)
                papers_per_second = papers_processed / total_time
                st.metric("Papers/Second", f"{papers_per_second:.0f}")


def show_quality_assessment(result):
    """Show quality assessment of the timeline analysis."""
    st.subheader("📊 Quality Assessment")
    
    if not result or not result.periods:
        st.info("No results available for quality assessment")
        return
    
    # Calculate quality metrics
    periods = result.periods
    avg_confidence = result.confidence
    
    # Period distribution metrics
    durations = [p.end_year - p.start_year + 1 for p in periods]
    avg_duration = sum(durations) / len(durations)
    duration_std = (sum((d - avg_duration) ** 2 for d in durations) / len(durations)) ** 0.5
    
    # Paper distribution
    total_papers = sum(p.total_papers for p in periods)
    paper_balance = 1 - max(p.total_papers / total_papers for p in periods) if total_papers > 0 else 0
    
    # Topic coverage
    topics_labeled = sum(1 for p in periods if p.topic_label)
    topic_coverage = topics_labeled / len(periods) if periods else 0
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Overall Confidence", f"{avg_confidence:.3f}")
    with col2:
        st.metric("Topic Coverage", f"{topic_coverage:.1%}")
    with col3:
        st.metric("Duration Consistency", f"{1/(1+duration_std):.3f}")
    with col4:
        st.metric("Paper Balance", f"{paper_balance:.3f}")
    
    # Quality interpretation
    st.write("**Quality Interpretation:**")
    
    quality_notes = []
    if avg_confidence >= 0.8:
        quality_notes.append("✅ High confidence - strong paradigm detection")
    elif avg_confidence >= 0.6:
        quality_notes.append("⚠️ Moderate confidence - reasonable paradigm detection")
    else:
        quality_notes.append("❌ Low confidence - weak paradigm signals")
    
    if topic_coverage >= 0.9:
        quality_notes.append("✅ Excellent topic coverage - most periods labeled")
    elif topic_coverage >= 0.7:
        quality_notes.append("⚠️ Good topic coverage - majority periods labeled")
    else:
        quality_notes.append("❌ Poor topic coverage - many periods unlabeled")
    
    if 1/(1+duration_std) >= 0.7:
        quality_notes.append("✅ Consistent period durations")
    else:
        quality_notes.append("⚠️ Variable period durations")
    
    for note in quality_notes:
        st.write(f"- {note}")


def show_final_results():
    """Main final results page function."""
    st.header("🎯 Final Results")
    st.write("Complete timeline analysis result with comprehensive visualizations.")
    
    # Check if all stages are complete
    if st.session_state.final_periods is None:
        st.error("Please complete all previous stages first")
        
        # Show progress
        stages_status = [
            ("Data Loading", st.session_state.academic_years is not None),
            ("Change Detection", st.session_state.boundary_years is not None),
            ("Segmentation", st.session_state.initial_periods is not None),
            ("Characterization", st.session_state.characterized_periods is not None),
            ("Merging", st.session_state.final_periods is not None)
        ]
        
        st.write("**Progress Status:**")
        for stage, completed in stages_status:
            status = "✅" if completed else "❌"
            st.write(f"{status} {stage}")
        
        return
    
    # Create final result object
    result = create_final_timeline_result()
    
    if not result:
        st.error("Failed to create final timeline result")
        return
    
    # Results overview
    st.subheader("📈 Timeline Analysis Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Domain", result.domain_name.replace('_', ' ').title())
    with col2:
        st.metric("Periods Identified", len(result.periods))
    with col3:
        st.metric("Overall Confidence", f"{result.confidence:.3f}")
    with col4:
        total_years = max(p.end_year for p in result.periods) - min(p.start_year for p in result.periods) + 1
        st.metric("Years Covered", total_years)
    
    # Main timeline visualization
    st.subheader("🗓️ Complete Timeline Visualization")
    timeline_fig = create_comprehensive_timeline_chart(result)
    if timeline_fig:
        st.plotly_chart(timeline_fig, use_container_width=True)
    
    # Topic evolution flow
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("🔄 Topic Evolution Flow")
        sankey_fig = create_period_evolution_sankey()
        if sankey_fig:
            st.plotly_chart(sankey_fig, use_container_width=True)
        else:
            st.info("Topic evolution flow requires multiple periods")
    
    with col2:
        st.subheader("📋 Period Summary")
        summary_df = create_summary_statistics_table(result)
        if summary_df is not None:
            st.dataframe(summary_df, use_container_width=True, hide_index=True)
    
    # Narrative evolution
    st.subheader("📖 Narrative Evolution")
    st.write(result.narrative_evolution)
    
    # Detailed period descriptions
    st.subheader("📝 Detailed Period Descriptions")
    for i, period in enumerate(result.periods):
        with st.expander(f"Period {i+1}: {period.start_year}-{period.end_year} - {period.topic_label or 'Unlabeled'}"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"**Time Range:** {period.start_year}-{period.end_year}")
                st.write(f"**Duration:** {period.end_year - period.start_year + 1} years")
                st.write(f"**Total Papers:** {period.total_papers:,}")
                st.write(f"**Total Citations:** {period.total_citations:,}")
                st.write(f"**Confidence:** {period.confidence:.3f}")
                
                if hasattr(period, 'network_stability'):
                    st.write(f"**Network Stability:** {period.network_stability:.3f}")
            
            with col2:
                st.write("**Top Keywords:**")
                for j, keyword in enumerate(period.top_keywords[:10], 1):
                    freq = period.combined_keyword_frequencies.get(keyword, 0)
                    st.write(f"{j:2d}. {keyword} ({freq})")
            
            if period.topic_description:
                st.write(f"**Description:** {period.topic_description}")
    
    # Analysis tabs
    tab1, tab2, tab3 = st.tabs(["📊 Quality Assessment", "⚡ Performance", "💾 Export"])
    
    with tab1:
        show_quality_assessment(result)
    
    with tab2:
        show_algorithm_performance_summary()
    
    with tab3:
        st.subheader("💾 Export Complete Results")
        
        # Create comprehensive export data
        export_data = {
            "domain_name": result.domain_name,
            "analysis_summary": {
                "total_periods": len(result.periods),
                "overall_confidence": result.confidence,
                "boundary_years": list(result.boundary_years),
                "narrative_evolution": result.narrative_evolution
            },
            "periods": [
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
                    "network_stability": getattr(p, 'network_stability', 0.0),
                    "network_metrics": getattr(p, 'network_metrics', {}),
                    "top_keywords": list(p.top_keywords[:15]),
                    "keyword_frequencies": dict(list(p.combined_keyword_frequencies.items())[:20])
                }
                for i, p in enumerate(result.periods)
            ],
            "algorithm_performance": st.session_state.timing_data,
            "configuration": {
                "direction_threshold": st.session_state.algorithm_config.direction_threshold,
                "validation_threshold": st.session_state.algorithm_config.validation_threshold,
                "citation_boost_rate": st.session_state.algorithm_config.citation_boost_rate,
                "cohesion_weight": st.session_state.algorithm_config.cohesion_weight,
                "separation_weight": st.session_state.algorithm_config.separation_weight,
                "top_k_keywords": st.session_state.algorithm_config.top_k_keywords
            }
        }
        
        col1, col2 = st.columns(2)
        
        with col1:
            # JSON export
            st.download_button(
                label="📄 Download Complete Results (JSON)",
                data=json.dumps(export_data, indent=2),
                file_name=f"{result.domain_name}_complete_timeline_analysis.json",
                mime="application/json"
            )
        
        with col2:
            # CSV export for periods
            if summary_df is not None:
                st.download_button(
                    label="📊 Download Period Summary (CSV)",
                    data=summary_df.to_csv(index=False),
                    file_name=f"{result.domain_name}_period_summary.csv",
                    mime="text/csv"
                )
        
        # Show export preview
        st.write("**Export Preview:**")
        st.json({
            "domain": export_data["domain_name"],
            "periods": len(export_data["periods"]),
            "confidence": export_data["analysis_summary"]["overall_confidence"],
            "sample_period": export_data["periods"][0] if export_data["periods"] else {}
        })
    
    # Success message
    st.success(f"🎉 Timeline analysis completed successfully! Identified {len(result.periods)} distinct periods in {result.domain_name.replace('_', ' ')} with {result.confidence:.1%} confidence.") 