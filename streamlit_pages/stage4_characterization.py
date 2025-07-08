"""Stage 4: Period Characterization
Visualizes network analysis and LLM-based period labeling.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import networkx as nx
import time
from core.segment_modeling.segment_modeling import characterize_academic_periods
from core.data.data_models import AcademicPeriod


def run_characterization():
    """Run period characterization with detailed progress tracking."""
    if st.session_state.initial_periods is None:
        st.error("Please run segmentation first in Stage 3")
        return False
        
    if st.session_state.characterized_periods is None:
        periods = st.session_state.initial_periods
        total_periods = len(periods)
        
        # Create progress tracking containers
        progress_container = st.container()
        
        with progress_container:
            st.subheader("🔄 Characterization Progress")
            
            # Overall progress
            overall_progress = st.progress(0)
            overall_status = st.empty()
            
            # Current step details
            step_status = st.empty()
            step_progress = st.progress(0)
            
            # Results preview
            results_preview = st.empty()
            
            # Detailed log
            with st.expander("📋 Detailed Progress Log", expanded=True):
                log_container = st.empty()
                
            start_time = time.time()
            
            # Initialize log
            log_messages = []
            
            def update_log(message):
                log_messages.append(f"⏱️ {time.time() - start_time:.1f}s: {message}")
                log_container.text("\n".join(log_messages[-20:]))  # Show last 20 messages
            
            def update_overall_progress(current_step, total_steps, message):
                progress = current_step / total_steps
                overall_progress.progress(progress)
                overall_status.write(f"**Overall Progress:** {current_step}/{total_steps} - {message}")
            
            def update_step_progress(current_sub, total_sub, message):
                if total_sub > 0:
                    progress = current_sub / total_sub
                    step_progress.progress(progress)
                step_status.write(f"**Current Step:** {message}")
            
            # Step 1: Initialize
            update_overall_progress(0, 7, "Initializing characterization...")
            update_step_progress(0, 1, "Loading domain data and building networks...")
            update_log("🔧 Starting characterization process")
            update_log(f"📊 Will characterize {total_periods} periods")
            
            # Import the detailed characterization function
            from core.segment_modeling.segment_modeling import (
                load_semantic_citations,
                build_citation_network_from_papers,
                get_papers_in_period_with_filtering,
                build_period_subnetwork_from_papers,
                analyze_network_stability,
                measure_community_persistence,
                analyze_flow_stability,
                calculate_centrality_consensus,
                detect_network_themes_from_papers,
                calculate_network_metrics,
                calculate_confidence
            )
            from core.segment_modeling.segment_labeling import (
                select_representative_papers,
                generate_period_label_and_description
            )
            from sklearn.feature_extraction.text import TfidfVectorizer
            
            # Step 2: Load semantic citations
            update_overall_progress(1, 7, "Loading semantic citations...")
            update_step_progress(0, 1, "Reading citation network from GraphML...")
            update_log("📖 Loading semantic citations from GraphML file")
            
            domain_name = st.session_state.selected_domain
            semantic_citations = load_semantic_citations(domain_name)
            update_log(f"✅ Loaded {len(semantic_citations)} semantic citations")
            
            # Step 3: Build citation network
            update_overall_progress(2, 7, "Building citation network...")
            update_step_progress(0, 1, "Processing all papers and building network graph...")
            update_log("🕸️ Building citation network from papers")
            
            # Get all papers from periods
            all_papers = []
            for period in periods:
                for academic_year in period.academic_years:
                    all_papers.extend(academic_year.papers)
            
            update_log(f"📄 Processing {len(all_papers)} papers")
            
            citation_network = build_citation_network_from_papers(all_papers, semantic_citations)
            update_log(f"🕸️ Citation network: {citation_network.number_of_nodes()} nodes, {citation_network.number_of_edges()} edges")
            
            # Step 4: Initialize TF-IDF
            update_overall_progress(3, 7, "Initializing text analysis...")
            update_step_progress(0, 1, "Setting up TF-IDF vectorizer for theme detection...")
            update_log("🔤 Initializing TF-IDF vectorizer")
            
            tfidf_vectorizer = TfidfVectorizer(
                max_features=10000, stop_words="english", ngram_range=(1, 3), min_df=2
            )
            
            # Step 5: Process each period
            update_overall_progress(4, 7, "Analyzing periods...")
            characterized_periods = []
            
            for i, period in enumerate(periods):
                period_progress = (i + 1) / total_periods
                update_step_progress(i + 1, total_periods, f"Processing Period {i+1}/{total_periods} ({period.start_year}-{period.end_year})")
                update_log(f"🔍 Analyzing Period {i+1}: {period.start_year}-{period.end_year}")
                
                # Get period papers
                period_papers = get_papers_in_period_with_filtering(
                    all_papers, period, period.start_year, period.end_year, verbose=False
                )
                update_log(f"   📄 Found {len(period_papers)} papers in period")
                
                if len(period_papers) < 3:
                    update_log(f"   ⚠️ Insufficient papers ({len(period_papers)}) for network analysis")
                    continue
                
                # Build subnetwork
                update_log(f"   🕸️ Building period subnetwork...")
                period_subnetwork = build_period_subnetwork_from_papers(
                    citation_network, period_papers, period.start_year, period.end_year
                )
                update_log(f"   🕸️ Subnetwork: {period_subnetwork.number_of_nodes()} nodes, {period_subnetwork.number_of_edges()} edges")
                
                # Network analysis
                update_log(f"   📊 Analyzing network stability...")
                network_stability = analyze_network_stability(period_subnetwork)
                
                update_log(f"   👥 Measuring community persistence...")
                community_persistence = measure_community_persistence(period_subnetwork)
                
                update_log(f"   🌊 Analyzing flow stability...")
                flow_stability = analyze_flow_stability(period_subnetwork)
                
                update_log(f"   🎯 Calculating centrality consensus...")
                centrality_consensus = calculate_centrality_consensus(period_subnetwork)
                
                update_log(f"   🏷️ Detecting dominant themes...")
                dominant_themes = detect_network_themes_from_papers(
                    period_papers, period_subnetwork, tfidf_vectorizer
                )
                
                update_log(f"   📋 Selecting representative papers...")
                period_papers_dict = []
                for paper in period_papers:
                    period_papers_dict.append({
                        "id": paper.id,
                        "data": {
                            "title": paper.title,
                            "pub_year": paper.pub_year,
                            "cited_by_count": paper.cited_by_count,
                            "keywords": paper.keywords,
                            "description": paper.description,
                            "content": paper.content,
                        }
                    })
                
                representative_papers = select_representative_papers(
                    period_papers_dict, period_subnetwork, dominant_themes, verbose=False
                )
                
                update_log(f"   📊 Calculating network metrics...")
                network_metrics = calculate_network_metrics(period_subnetwork)
                
                # Generate topic label (LLM call)
                update_log(f"   🤖 Generating topic label with LLM...")
                previous_periods = [
                    (p.start_year, p.end_year, p.topic_label, p.topic_description)
                    for p in characterized_periods
                ]
                
                period_label, period_description = generate_period_label_and_description(
                    dominant_themes,
                    representative_papers,
                    period.start_year,
                    period.end_year,
                    previous_periods=previous_periods,
                    domain_name=domain_name,
                    verbose=False
                )
                
                # Calculate confidence
                confidence = calculate_confidence(
                    network_stability,
                    community_persistence,
                    flow_stability,
                    centrality_consensus,
                    len(period_papers),
                    network_metrics
                )
                
                # Create characterized period
                characterized_period = AcademicPeriod(
                    start_year=period.start_year,
                    end_year=period.end_year,
                    academic_years=period.academic_years,
                    total_papers=period.total_papers,
                    total_citations=period.total_citations,
                    combined_keyword_frequencies=period.combined_keyword_frequencies,
                    top_keywords=period.top_keywords,
                    topic_label=period_label,
                    topic_description=period_description,
                    network_stability=network_stability,
                    community_persistence=community_persistence,
                    flow_stability=flow_stability,
                    centrality_consensus=centrality_consensus,
                    representative_papers=tuple(representative_papers),
                    network_metrics=network_metrics,
                    confidence=confidence,
                )
                
                characterized_periods.append(characterized_period)
                update_log(f"   ✅ Period {i+1} completed: '{period_label}' (confidence: {confidence:.3f})")
                
                # Update results preview
                with results_preview.container():
                    st.write(f"**Completed Periods ({len(characterized_periods)}/{total_periods}):**")
                    for j, cp in enumerate(characterized_periods):
                        st.write(f"- Period {j+1}: {cp.start_year}-{cp.end_year} - {cp.topic_label} (conf: {cp.confidence:.3f})")
            
            # Step 6: Finalize
            update_overall_progress(6, 7, "Finalizing results...")
            update_step_progress(0, 1, "Storing characterized periods...")
            
            characterization_time = time.time() - start_time
            st.session_state.timing_data["Characterization"] = characterization_time
            st.session_state.characterized_periods = characterized_periods
            
            update_log(f"🎉 Characterization completed successfully!")
            update_log(f"⏱️ Total time: {characterization_time:.2f} seconds")
            update_log(f"📊 Characterized {len(characterized_periods)} periods")
            
            # Step 7: Complete
            update_overall_progress(7, 7, "Complete!")
            update_step_progress(1, 1, "Characterization finished successfully")
            
            # Final success message
            st.success(f"✅ Characterization completed in {characterization_time:.2f}s - {len(characterized_periods)} periods characterized!")
    
    return True


def create_network_metrics_chart(periods):
    """Create network metrics visualization."""
    if not periods or not any(p.network_metrics for p in periods):
        return None
    
    period_names = [f"Period {i+1}" for i in range(len(periods))]
    
    # Collect metrics
    metrics_data = {}
    for period in periods:
        if period.network_metrics:
            for metric, value in period.network_metrics.items():
                if metric not in metrics_data:
                    metrics_data[metric] = []
                metrics_data[metric].append(value)
        else:
            for metric in ['clustering_coefficient', 'avg_path_length', 'modularity', 'density']:
                if metric not in metrics_data:
                    metrics_data[metric] = []
                metrics_data[metric].append(0.0)
    
    if not metrics_data:
        return None
    
    # Create subplot for each metric
    n_metrics = len(metrics_data)
    rows = (n_metrics + 1) // 2
    
    fig = make_subplots(
        rows=rows, cols=2,
        subplot_titles=list(metrics_data.keys()),
        vertical_spacing=0.15
    )
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    for i, (metric, values) in enumerate(metrics_data.items()):
        row = i // 2 + 1
        col = i % 2 + 1
        
        fig.add_trace(
            go.Bar(
                x=period_names,
                y=values,
                name=metric,
                marker_color=colors[i % len(colors)],
                hovertemplate=f'<b>%{{x}}</b><br>{metric}: %{{y:.3f}}<extra></extra>',
                showlegend=False
            ),
            row=row, col=col
        )
    
    fig.update_layout(
        height=400 * rows,
        title_text="Network Metrics by Period"
    )
    
    return fig


def create_confidence_comparison_chart(periods):
    """Create confidence scores comparison."""
    if not periods:
        return None
    
    period_names = [f"Period {i+1}\n({p.start_year}-{p.end_year})" for i, p in enumerate(periods)]
    confidence_scores = [p.confidence for p in periods]
    network_stability = [getattr(p, 'network_stability', 0.0) for p in periods]
    community_persistence = [getattr(p, 'community_persistence', 0.0) for p in periods]
    flow_stability = [getattr(p, 'flow_stability', 0.0) for p in periods]
    centrality_consensus = [getattr(p, 'centrality_consensus', 0.0) for p in periods]
    
    fig = go.Figure()
    
    # Confidence scores
    fig.add_trace(
        go.Bar(
            x=period_names,
            y=confidence_scores,
            name='Overall Confidence',
            marker_color='lightblue',
            hovertemplate='<b>%{x}</b><br>Confidence: %{y:.3f}<extra></extra>'
        )
    )
    
    # Add secondary metrics as lines
    fig.add_trace(
        go.Scatter(
            x=period_names,
            y=network_stability,
            mode='lines+markers',
            name='Network Stability',
            yaxis='y2',
            line=dict(color='red', width=2),
            hovertemplate='<b>%{x}</b><br>Network Stability: %{y:.3f}<extra></extra>'
        )
    )
    
    fig.add_trace(
        go.Scatter(
            x=period_names,
            y=community_persistence,
            mode='lines+markers',
            name='Community Persistence',
            yaxis='y2',
            line=dict(color='green', width=2),
            hovertemplate='<b>%{x}</b><br>Community Persistence: %{y:.3f}<extra></extra>'
        )
    )
    
    # Create secondary y-axis
    fig.update_layout(
        title="Period Characterization Quality Metrics",
        xaxis_title="Periods",
        yaxis=dict(title="Confidence Score", side="left"),
        yaxis2=dict(title="Network Metrics", side="right", overlaying="y"),
        height=400
    )
    
    return fig


def create_topic_evolution_chart(periods):
    """Create topic evolution visualization."""
    if not periods or not any(p.topic_label for p in periods):
        return None
    
    # Create timeline of topics
    period_data = []
    for i, period in enumerate(periods):
        period_data.append({
            'Period': i + 1,
            'Start Year': period.start_year,
            'End Year': period.end_year,
            'Duration': period.end_year - period.start_year + 1,
            'Topic': period.topic_label or "Unlabeled",
            'Papers': period.total_papers,
            'Confidence': period.confidence
        })
    
    df = pd.DataFrame(period_data)
    
    # Create Gantt-like chart
    fig = go.Figure()
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
    
    for i, row in df.iterrows():
        fig.add_trace(
            go.Scatter(
                x=[row['Start Year'], row['End Year']],
                y=[i, i],
                mode='lines+markers',
                name=f"Period {row['Period']}",
                line=dict(color=colors[i % len(colors)], width=8),
                marker=dict(size=10),
                hovertemplate=f"<b>Period {row['Period']}</b><br>" +
                            f"Years: {row['Start Year']}-{row['End Year']}<br>" +
                            f"Topic: {row['Topic']}<br>" +
                            f"Papers: {row['Papers']}<br>" +
                            f"Confidence: {row['Confidence']:.3f}<extra></extra>"
            )
        )
    
    # Add topic labels
    for i, row in df.iterrows():
        mid_year = (row['Start Year'] + row['End Year']) / 2
        fig.add_annotation(
            x=mid_year,
            y=i + 0.3,
            text=row['Topic'][:30] + ("..." if len(row['Topic']) > 30 else ""),
            showarrow=False,
            font=dict(size=10)
        )
    
    fig.update_layout(
        title="Topic Evolution Timeline",
        xaxis_title="Year",
        yaxis_title="Periods",
        yaxis=dict(
            tickmode='array',
            tickvals=list(range(len(df))),
            ticktext=[f"Period {i+1}" for i in range(len(df))]
        ),
        height=max(300, len(df) * 50),
        showlegend=False
    )
    
    return fig


def show_period_characterization_details(periods):
    """Show detailed characterization for selected period."""
    st.subheader("🔍 Period Characterization Details")
    
    if not periods:
        st.info("No characterized periods available")
        return
    
    period_options = [f"Period {i+1} ({p.start_year}-{p.end_year})" for i, p in enumerate(periods)]
    selected_period_idx = st.selectbox("Select Period for Detailed Analysis", range(len(periods)), format_func=lambda x: period_options[x])
    
    selected_period = periods[selected_period_idx]
    
    # Basic information
    col1, col2 = st.columns(2)
    
    with col1:
        st.write(f"**Period {selected_period_idx + 1} Overview:**")
        st.write(f"- **Topic Label:** {selected_period.topic_label or 'Not available'}")
        st.write(f"- **Years:** {selected_period.start_year}-{selected_period.end_year}")
        st.write(f"- **Papers:** {selected_period.total_papers:,}")
        st.write(f"- **Overall Confidence:** {selected_period.confidence:.3f}")
        
        st.write(f"**Network Analysis:**")
        st.write(f"- **Network Stability:** {getattr(selected_period, 'network_stability', 0.0):.3f}")
        st.write(f"- **Community Persistence:** {getattr(selected_period, 'community_persistence', 0.0):.3f}")
        st.write(f"- **Flow Stability:** {getattr(selected_period, 'flow_stability', 0.0):.3f}")
        st.write(f"- **Centrality Consensus:** {getattr(selected_period, 'centrality_consensus', 0.0):.3f}")
    
    with col2:
        st.write("**Network Metrics:**")
        if selected_period.network_metrics:
            for metric, value in selected_period.network_metrics.items():
                st.write(f"- **{metric.replace('_', ' ').title()}:** {value:.3f}")
        else:
            st.write("Network metrics not available")
        
        st.write("**Top Keywords:**")
        for i, keyword in enumerate(selected_period.top_keywords[:10], 1):
            freq = selected_period.combined_keyword_frequencies.get(keyword, 0)
            st.write(f"{i:2d}. {keyword} ({freq})")
    
    # Topic description
    if selected_period.topic_description:
        st.write("**Topic Description:**")
        st.write(selected_period.topic_description)
    
    # Representative papers
    if hasattr(selected_period, 'representative_papers') and selected_period.representative_papers:
        st.write("**Representative Papers:**")
        papers_data = []
        for paper in selected_period.representative_papers:
            papers_data.append({
                'Title': paper.get('title', 'N/A')[:80] + ("..." if len(paper.get('title', '')) > 80 else ""),
                'Year': paper.get('year', 'N/A'),
                'Citations': paper.get('citation_count', 0),
                'Score': paper.get('score', 0.0)
            })
        
        papers_df = pd.DataFrame(papers_data)
        st.dataframe(papers_df, use_container_width=True, hide_index=True)


def create_network_structure_visualization(periods):
    """Create network structure visualization for a selected period."""
    st.subheader("🕸️ Network Structure Analysis")
    
    if not periods:
        st.info("No periods available for network analysis")
        return
    
    period_options = [f"Period {i+1} ({p.start_year}-{p.end_year})" for i, p in enumerate(periods)]
    selected_period_idx = st.selectbox("Select Period for Network Visualization", range(len(periods)), format_func=lambda x: period_options[x], key="network_viz")
    
    selected_period = periods[selected_period_idx]
    
    # Show network metrics summary
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Papers", selected_period.total_papers)
    with col2:
        clustering = selected_period.network_metrics.get('clustering_coefficient', 0.0) if selected_period.network_metrics else 0.0
        st.metric("Clustering", f"{clustering:.3f}")
    with col3:
        modularity = selected_period.network_metrics.get('modularity', 0.0) if selected_period.network_metrics else 0.0
        st.metric("Modularity", f"{modularity:.3f}")
    with col4:
        density = selected_period.network_metrics.get('density', 0.0) if selected_period.network_metrics else 0.0
        st.metric("Density", f"{density:.3f}")
    
    # Network structure information
    st.write(f"""
    **Network Analysis Summary for Period {selected_period_idx + 1}:**
    - **Clustering Coefficient:** Measures local connectivity (higher = more clustered)
    - **Modularity:** Measures community structure (higher = better communities)
    - **Density:** Measures overall connectivity (higher = more connections)
    - **Network Stability:** {getattr(selected_period, 'network_stability', 0.0):.3f}
    """)
    
    # Note about network visualization
    st.info("""
    **Note:** Full network visualization requires additional graph layout libraries.
    The metrics above provide quantitative analysis of the citation network structure.
    Network stability measures the robustness of connections within this period.
    """)


def show_characterization():
    """Main characterization page function."""
    st.header("🏷️ Stage 4: Period Characterization")
    st.write("Analyze periods using network analysis and generate topic labels.")
    
    # Check prerequisites
    if st.session_state.initial_periods is None:
        st.error("Please run segmentation first in Stage 3")
        return
    
    # Control panel with manual execution
    col1, col2 = st.columns([3, 1])
    with col1:
        st.subheader("🎯 Characterization Control")
        if st.session_state.characterized_periods is None:
            st.write("⚠️ **Note:** This stage involves network analysis and LLM calls, which may take several minutes.")
            st.write("🔍 **Process includes:** Citation network analysis, network metrics calculation, theme detection, and LLM-based topic labeling.")
        else:
            st.write("✅ **Characterization completed!** View results below or re-run to update with new parameters.")
    
    with col2:
        if st.session_state.characterized_periods is None:
            run_characterization_button = st.button(
                "🚀 Start Characterization", 
                type="primary", 
                help="Begin network analysis and topic labeling (2-5 minutes)"
            )
        else:
            run_characterization_button = st.button(
                "🔄 Re-run Characterization", 
                type="secondary", 
                help="Re-analyze with current parameters"
            )
        
        if run_characterization_button:
            st.session_state.characterized_periods = None  # Force re-run
    
    # Show current status or run characterization
    if st.session_state.characterized_periods is None:
        if not run_characterization_button:
            st.info("📋 Ready to characterize periods. Click 'Start Characterization' to begin.")
            
            # Show what will be analyzed
            periods = st.session_state.initial_periods
            st.write(f"**Will characterize {len(periods)} periods:**")
            
            preview_data = []
            for i, period in enumerate(periods):
                preview_data.append({
                    "Period": f"Period {i+1}",
                    "Years": f"{period.start_year}-{period.end_year}",
                    "Duration": f"{period.end_year - period.start_year + 1} years",
                    "Papers": f"{period.total_papers:,}",
                    "Citations": f"{period.total_citations:,}",
                    "Top Keywords": ", ".join(period.top_keywords[:5])
                })
            
            df = pd.DataFrame(preview_data)
            st.dataframe(df, use_container_width=True)
            return
        else:
            # Run characterization with progress tracking
            if not run_characterization():
                return
    
    # Show results if characterization is complete
    periods = st.session_state.characterized_periods
    
    if not periods:
        st.warning("No periods were successfully characterized.")
        return
    
    # Results summary
    characterized_count = sum(1 for p in periods if p.topic_label)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Periods Characterized", characterized_count)
    with col2:
        avg_confidence = sum(p.confidence for p in periods) / len(periods) if periods else 0
        st.metric("Avg Confidence", f"{avg_confidence:.3f}")
    with col3:
        avg_network_stability = sum(getattr(p, 'network_stability', 0.0) for p in periods) / len(periods) if periods else 0
        st.metric("Avg Network Stability", f"{avg_network_stability:.3f}")
    
    # Topic evolution timeline
    st.subheader("📈 Topic Evolution Timeline")
    topic_fig = create_topic_evolution_chart(periods)
    if topic_fig:
        st.plotly_chart(topic_fig, use_container_width=True)
    
    # Confidence and network metrics
    st.subheader("📊 Quality Metrics")
    confidence_fig = create_confidence_comparison_chart(periods)
    if confidence_fig:
        st.plotly_chart(confidence_fig, use_container_width=True)
    
    # Network metrics detailed view
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🕸️ Network Metrics")
        network_fig = create_network_metrics_chart(periods)
        if network_fig:
            st.plotly_chart(network_fig, use_container_width=True)
        else:
            st.info("Network metrics not available")
    
    with col2:
        # Summary table
        st.subheader("📋 Characterization Summary")
        if periods:
            summary_data = []
            for i, period in enumerate(periods):
                summary_data.append({
                    'Period': i + 1,
                    'Years': f"{period.start_year}-{period.end_year}",
                    'Topic': (period.topic_label[:30] + "..." if period.topic_label and len(period.topic_label) > 30 else period.topic_label) or "N/A",
                    'Confidence': f"{period.confidence:.3f}",
                    'Papers': period.total_papers
                })
            
            summary_df = pd.DataFrame(summary_data)
            st.dataframe(summary_df, use_container_width=True, hide_index=True)
    
    # Detailed analysis tabs
    tab1, tab2, tab3 = st.tabs(["🔍 Period Details", "🕸️ Network Analysis", "📖 Algorithm Info"])
    
    with tab1:
        show_period_characterization_details(periods)
    
    with tab2:
        create_network_structure_visualization(periods)
    
    with tab3:
        st.write("""
        **Period Characterization Algorithm:**
        
        **Stage 4A: Network Analysis**
        1. **Citation Network Construction:** Build directed graph from citation data
        2. **Subnetwork Extraction:** Extract period-specific citation subnetworks
        3. **Network Metrics Calculation:**
           - Clustering coefficient (local connectivity)
           - Average path length (network diameter)
           - Modularity (community structure)
           - Density (overall connectivity)
        4. **Stability Analysis:**
           - Network stability (connection robustness)
           - Community persistence (research group stability)
           - Flow stability (citation flow consistency)
           - Centrality consensus (importance measure agreement)
        
        **Stage 4B: Topic Detection**
        1. **TF-IDF Vectorization:** Convert papers to feature vectors
        2. **Theme Extraction:** Identify dominant research themes
        3. **Representative Paper Selection:** Use network centrality + citation impact
        
        **Stage 4C: LLM Labeling**
        1. **Context Building:** Combine themes, papers, and period metadata
        2. **Structured LLM Query:** Generate topic labels and descriptions
        3. **Confidence Calculation:** Weighted combination of network metrics
        
        **Performance Notes:**
        - Network analysis: O(n²) for dense graphs
        - LLM calls: ~30-60 seconds per period
        - Total time: 2-5 minutes for typical domains
        """)
        
        if "Characterization" in st.session_state.timing_data:
            st.write(f"**Last Run Time:** {st.session_state.timing_data['Characterization']:.2f} seconds")
    
    # Data export
    st.subheader("💾 Export Results")
    if st.button("Export Characterization Results"):
        characterization_data = {
            "domain": st.session_state.selected_domain,
            "characterized_periods": [
                {
                    "period_id": i + 1,
                    "start_year": p.start_year,
                    "end_year": p.end_year,
                    "topic_label": p.topic_label,
                    "topic_description": p.topic_description,
                    "confidence": p.confidence,
                    "network_stability": getattr(p, 'network_stability', 0.0),
                    "network_metrics": p.network_metrics,
                    "total_papers": p.total_papers,
                    "top_keywords": list(p.top_keywords[:10])
                }
                for i, p in enumerate(periods)
            ]
        }
        
        st.download_button(
            label="Download Characterization JSON",
            data=pd.Series(characterization_data).to_json(indent=2),
            file_name=f"{st.session_state.selected_domain}_characterization_results.json",
            mime="application/json"
        ) 