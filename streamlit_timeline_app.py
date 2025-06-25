import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from typing import Dict

# Import our modular components
from streamlit_components import (
    # Utilities
    load_all_domains,
    run_algorithm_with_params,
    # Analysis Overview (Tab 1)
    create_keyword_filtering_plot,
    create_direction_signal_detection_plot,
    create_citation_signal_detection_plot,
    create_final_validation_plot,
    create_similarity_segmentation_plot,
    # Keyword Evolution (Tab 2)
    prepare_keyword_evolution_data,
    create_keyword_streamgraph,
    create_enhanced_keyword_heatmap,
    # Decision Analysis (Tab 3)
    create_decision_tree_analysis,
    create_decision_flow_diagram,
    create_parameter_sensitivity_analysis,
    create_keyword_filtering_impact_analysis,
)

# Core algorithm imports
from core.algorithm_config import AlgorithmConfig

# Configure Streamlit page
st.set_page_config(
    page_title="Timeline Segmentation Algorithm Dashboard",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for styling
st.markdown(
    """
<style>
    .metric-container {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .success-message {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 0.75rem;
        border-radius: 0.25rem;
        margin: 1rem 0;
    }
    .info-message {
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        color: #0c5460;
        padding: 0.75rem;
        border-radius: 0.25rem;
        margin: 1rem 0;
    }
</style>
""",
    unsafe_allow_html=True,
)


def main():
    """Main Streamlit application."""

    st.title("Timeline Segmentation Algorithm Dashboard")

    st.sidebar.header("Algorithm Configuration")

    # Load domain data
    domain_data_dict = load_all_domains()

    # Domain selection
    selected_domain = st.sidebar.selectbox(
        "Select Domain",
        options=list(domain_data_dict.keys()),
        index=0,
        format_func=lambda x: x.replace("_", " ").title(),
    )

    # Configuration mode - Simplified to focus on essential parameters
    st.sidebar.subheader("Configuration")
    config_mode = st.sidebar.radio(
        "Configuration Level",
        ["Simple (Granularity)", "Essential Parameters"],
        index=0,
        help="Choose between automatic presets or essential parameter control",
    )

    if config_mode == "Simple (Granularity)":
        # Simple granularity interface - now using optimized parameters from file
        st.sidebar.subheader("Optimized Granularity Control")
        granularity = st.sidebar.slider(
            "Sensitivity Level",
            min_value=1,
            max_value=5,
            value=3,
            step=1,
            format="%d",
            help="1=Ultra-Fine (most segments), 5=Ultra-Coarse (fewest segments). Now uses Bayesian-optimized parameters loaded from file with granularity scaling.",
        )

        # Create comprehensive config from granularity - now uses optimized parameters from file
        algorithm_config = AlgorithmConfig(
            granularity=granularity,
            domain_name=selected_domain  # Pass domain name for optimized parameter loading
        )
        
        # Get domain data for parameter display
        domain_data = domain_data_dict[selected_domain]
        
        # Display optimized configuration with granularity scaling
        st.sidebar.success(
            f"""
            **Granularity {granularity} Configuration:**
            *Using Bayesian-optimized parameters (direction={algorithm_config.direction_threshold:.3f}, validation={algorithm_config.validation_threshold:.3f}, year_range={algorithm_config.similarity_min_segment_length}-{algorithm_config.similarity_max_segment_length})*
            """
        )

    elif config_mode == "Essential Parameters":
        # Essential parameters only - the most impactful controls
        st.sidebar.subheader("Essential Controls")

        st.sidebar.markdown("**Core Algorithm Settings:**")

        # 1. DETECTION SENSITIVITY - Most important parameter
        direction_threshold = st.sidebar.slider(
            "Detection Sensitivity",
            min_value=0.2,
            max_value=0.6,
            value=0.4,
            step=0.05,
            help="Lower = more paradigm shifts detected, Higher = fewer, higher-confidence shifts",
        )

        # 2. VALIDATION STRICTNESS - Second most important
        validation_threshold = st.sidebar.slider(
            "Validation Strictness",
            min_value=0.6,
            max_value=0.9,
            value=0.8,
            step=0.05,
            help="Higher = stricter validation, fewer final signals",
        )

        # 4. KEYWORD FILTERING - Quality control
        st.sidebar.markdown("**Quality Controls:**")
        keyword_filtering_enabled = st.sidebar.checkbox(
            "Enable Keyword Filtering",
            value=True,
            help="Conservative filtering to reduce noise from irrelevant keywords",
        )

        citation_boost = 0.5  # Fixed at 50% multiplier

        # Segment length controls (similarity segmentation is now default)
        st.sidebar.markdown("**Segment Length Controls:**")
        similarity_min_segment_length = st.sidebar.slider(
            "Minimum Segment Length",
            min_value=2,
            max_value=15,
            value=4,
            step=1,
            help="Minimum allowed segment length in years. Shorter segments will be "
            "intelligently merged based on keyword similarity.",
        )

        similarity_max_segment_length = st.sidebar.slider(
            "Maximum Segment Length",
            min_value=20,
            max_value=100,
            value=50,
            step=5,
            help="Maximum allowed segment length in years. Prevents unrealistically long periods.",
        )

        # Create configuration with essential overrides only
        essential_overrides = {
            "direction_threshold": direction_threshold,
            "validation_threshold": validation_threshold,
            "citation_boost": citation_boost,
            "keyword_filtering_enabled": keyword_filtering_enabled,
            "similarity_min_segment_length": similarity_min_segment_length,
            "similarity_max_segment_length": similarity_max_segment_length,
        }

        algorithm_config = AlgorithmConfig.create_custom(
            granularity=3,  # Use balanced defaults for non-essential parameters
            overrides=essential_overrides,
            domain_name=selected_domain  # Pass domain name for optimized parameter loading
        )
    
    # Get domain data
    domain_data = domain_data_dict[selected_domain]

    # Run algorithm
    with st.spinner("Running algorithm..."):
        shift_signals, transition_evidence, segments, enhanced_signal_data = (
            run_algorithm_with_params(domain_data, selected_domain, algorithm_config)
        )

    # Main visualization tabs
    tab1, tab2, tab3 = st.tabs(
        [
            "Analysis Overview",
            "Keyword Evolution",
            "Decision Tree Analysis",
        ]
    )

    with tab1:
        st.header("Complete Validation Process - Step by Step")
        st.markdown(
            "**Detailed breakdown of how the algorithm processes data through each validation step**"
        )

        # Step 0: Keyword Filtering
        st.subheader("Step 0: Keyword Filtering")
        keyword_filtering_fig = create_keyword_filtering_plot(
            selected_domain, algorithm_config, enhanced_signal_data
        )
        st.plotly_chart(keyword_filtering_fig, use_container_width=True)

        # Step 1: Raw Direction Signal Detection
        st.subheader("Step 1: Raw Direction Signal Detection")
        direction_detection_fig = create_direction_signal_detection_plot(
            selected_domain, algorithm_config, enhanced_signal_data
        )
        st.plotly_chart(direction_detection_fig, use_container_width=True)

        # Step 2: Citation Signal Detection
        st.subheader("Step 2: Citation Signal Detection")
        
        # TRANSPARENCY FIX: Pass actual algorithm-detected citation signals
        clustering_metadata = enhanced_signal_data.get("clustering_metadata", {})
        actual_citation_signals = clustering_metadata.get("citation_signals", [])
        
        citation_detection_fig = create_citation_signal_detection_plot(
            selected_domain, domain_data, actual_citation_signals
        )
        st.plotly_chart(citation_detection_fig, use_container_width=True)

        # Step 3: Final Validation
        st.subheader("Step 3: Final Validation with Consistent Threshold")
        final_validation_fig = create_final_validation_plot(
            selected_domain, shift_signals, enhanced_signal_data, algorithm_config
        )
        st.plotly_chart(final_validation_fig, use_container_width=True)

        # Step 4: Similarity-Based Segmentation (includes timeline results)
        st.subheader("Step 4: Similarity-Based Segmentation")
        similarity_segmentation_fig = create_similarity_segmentation_plot(
            selected_domain,
            shift_signals,
            enhanced_signal_data,
            algorithm_config,
            segments,
            domain_data,
        )
        st.plotly_chart(similarity_segmentation_fig, use_container_width=True)
        
        # Add Segment Keywords Analysis Table
        if segments and domain_data:
            st.subheader("Segment Keywords Analysis")
            from streamlit_components.analysis_overview import create_segment_keywords_table
            keywords_df = create_segment_keywords_table(segments, domain_data)
            if not keywords_df.empty:
                st.dataframe(keywords_df, use_container_width=True, hide_index=True)
            else:
                st.info("No keyword data available for segment analysis.")

    with tab2:
        st.header("Keyword Evolution Analysis")
        st.markdown("**Analysis of how research keywords evolve over time**")
        
        # Prepare keyword data with filtering awareness
        df_keyword_evolution = prepare_keyword_evolution_data(
            domain_data,
            top_n=12,
            algorithm_config=algorithm_config,
            enhanced_signal_data=enhanced_signal_data,
        )

        if not df_keyword_evolution.empty:
            # Add the streamgraph
            st.subheader("Keyword Evolution Streamgraph")
            streamgraph_fig = create_keyword_streamgraph(
                df_keyword_evolution, selected_domain
            )
            st.plotly_chart(streamgraph_fig, use_container_width=True)

            # Enhanced keyword heatmap with paradigm shifts
            st.subheader("Keyword Prominence Heatmap")
            keyword_fig = create_enhanced_keyword_heatmap(
                df_keyword_evolution, selected_domain, shift_signals
            )
            st.plotly_chart(keyword_fig, use_container_width=True)
        else:
            st.warning("Insufficient keyword data for evolution analysis.")

    with tab3:
        st.header("🌳 Algorithm Decision Tree Analysis")
        st.markdown("**Complete transparency into how the algorithm makes decisions**")

        # Generate decision analysis
        decision_details = create_decision_tree_analysis(
            selected_domain,
            shift_signals,
            enhanced_signal_data,
            algorithm_config,
            domain_data,
        )

        # Generate keyword filtering impact analysis
        keyword_filtering_analysis = create_keyword_filtering_impact_analysis(
            selected_domain, algorithm_config, enhanced_signal_data, domain_data
        )

        if decision_details:
            # Keyword Filtering Impact Section (if enabled)
            if (
                algorithm_config.keyword_filtering_enabled
                and keyword_filtering_analysis
            ):
                st.markdown("### Keyword Filtering Impact Analysis")

                impact_data = keyword_filtering_analysis["impact"]
                metrics_data = keyword_filtering_analysis["metrics"]
                domain_stats = keyword_filtering_analysis["domain_stats"]

                # Create columns for metrics display
                filter_col1, filter_col2, filter_col3, filter_col4 = st.columns(4)

                with filter_col1:
                    st.metric(
                        "Total Keywords",
                        f"{impact_data['total_unique_keywords']:,}",
                        help="Total unique keywords in domain",
                    )

                with filter_col2:
                    st.metric(
                        "Singleton Keywords",
                        f"{impact_data['singleton_keywords']:,}",
                        delta=f"{impact_data['singleton_percentage']:.1%} of total",
                        help="Keywords appearing in only one paper",
                    )

                with filter_col3:
                    st.metric(
                        "Potentially Filtered",
                        f"{impact_data['potentially_filtered']:,}",
                        delta=f"{1.0 - impact_data['estimated_retention_rate']:.1%} reduction",
                        delta_color="inverse",
                        help="Keywords below minimum threshold",
                    )

                with filter_col4:
                    st.metric(
                        "Estimated Retention",
                        f"{impact_data['estimated_retention_rate']:.1%}",
                        help="Percentage of keywords retained after filtering",
                    )

                # Filtering effectiveness visualization
                fig_filtering = go.Figure(
                    data=[
                        go.Bar(
                            name="Retained Keywords",
                            x=["Keyword Distribution"],
                            y=[
                                impact_data["total_unique_keywords"]
                                - impact_data["potentially_filtered"]
                            ],
                            marker_color="green",
                            text=f"{impact_data['estimated_retention_rate']:.1%} retained",
                            textposition="inside",
                        ),
                        go.Bar(
                            name="Potentially Filtered",
                            x=["Keyword Distribution"],
                            y=[impact_data["potentially_filtered"]],
                            marker_color="red",
                            text=f"{1.0 - impact_data['estimated_retention_rate']:.1%} filtered",
                            textposition="inside",
                        ),
                    ]
                )

                fig_filtering.update_layout(
                    title="Keyword Filtering Impact Distribution",
                    barmode="stack",
                    yaxis_title="Number of Keywords",
                    height=400,
                    showlegend=True,
                    template="plotly_white",
                )

                st.plotly_chart(fig_filtering, use_container_width=True)

            elif algorithm_config.keyword_filtering_enabled:
                st.info(
                    "🔍 **Keyword Filtering**: Enabled but detailed impact analysis not available"
                )
            else:
                st.info(
                    "🔍 **Keyword Filtering**: Disabled - all keywords processed without filtering"
                )

            # Decision Flow Diagram
            st.markdown("### Decision Flow Visualization")
            decision_flow_fig = create_decision_flow_diagram(
                decision_details, selected_domain
            )
            st.plotly_chart(decision_flow_fig, use_container_width=True)

            # Parameter Sensitivity Analysis
            st.markdown("### Parameter Sensitivity Analysis")
            sensitivity_fig = create_parameter_sensitivity_analysis(
                decision_details, algorithm_config
            )
            st.plotly_chart(sensitivity_fig, use_container_width=True)

            # Detailed Decision Breakdown Table
            st.markdown("### Detailed Decision Breakdown")

            # Prepare data for table with keyword filtering information
            table_data = []
            for detail in decision_details:
                filtering_info = detail["keyword_filtering"]
                table_data.append(
                    {
                        "Year": str(detail["year"]),
                        "Base Confidence": f"{detail['base_confidence']:.3f}",
                        "Citation Support": (
                            "✅ Yes" if detail["citation_support"] else "❌ No"
                        ),
                        "Keyword Filtering": (
                            "🔍 Applied" if filtering_info["enabled"] else "🔍 Disabled"
                        ),
                        "Confidence Boosts": (
                            f"+{detail['confidence_boosts']:.3f}"
                            if detail["confidence_boosts"] > 0
                            else "None"
                        ),
                        "Final Confidence": f"{detail['final_confidence']:.3f}",
                        "Threshold": f"{detail['threshold']:.3f}",
                        "Decision": detail["decision_outcome"],
                        "Full Rationale": detail["rationale"],
                    }
                )

            df_decisions = pd.DataFrame(table_data)

            # Color-code the table
            def color_decision(val):
                if "ACCEPTED" in str(val):
                    return "background-color: #d4edda"
                elif "REJECTED" in str(val):
                    return "background-color: #f8d7da"
                return ""

            def color_filtering(val):
                if "🔍 Applied" in str(val):
                    return "background-color: #d1ecf1"
                return ""

            # Apply styling and display
            styled_df = df_decisions.style.map(color_decision, subset=["Decision"])
            styled_df = styled_df.map(color_filtering, subset=["Keyword Filtering"])
            st.dataframe(styled_df, use_container_width=True, hide_index=True)

            # Interactive "What-If" Analysis (Enhanced with Bayesian-optimized parameters)
            st.markdown("### Interactive What-If Analysis")
            st.markdown(
                "**Explore how parameter changes would affect decisions without re-running the algorithm**"
            )

            col1, col2 = st.columns(2)

            with col1:
                test_detection_threshold = st.slider(
                    "Test Detection Threshold",
                    min_value=0.1,
                    max_value=0.8,
                    value=algorithm_config.direction_threshold,
                    step=0.05,
                    help="How would changing detection threshold affect initial signal detection?",
                )

                test_validation_threshold = st.slider(
                    "Test Validation Threshold",
                    min_value=0.5,
                    max_value=0.95,
                    value=algorithm_config.validation_threshold,
                    step=0.05,
                    help="How would changing validation threshold affect final acceptance?",
                )

                test_keyword_filtering = st.checkbox(
                    "Test Keyword Filtering Enable/Disable",
                    value=algorithm_config.keyword_filtering_enabled,
                    help="How would enabling/disabling keyword filtering affect results?",
                )

                # Segment length controls for what-if analysis (always available since similarity is default)
                st.markdown("**📏 Segment Length Controls:**")
                test_min_segment_length = st.slider(
                    "Test Min Segment Length",
                    min_value=2,
                    max_value=15,
                    value=algorithm_config.similarity_min_segment_length,
                    step=1,
                    help="How would changing minimum segment length affect merging?",
                )
                test_max_segment_length = st.slider(
                    "Test Max Segment Length",
                    min_value=20,
                    max_value=100,
                    value=algorithm_config.similarity_max_segment_length,
                    step=5,
                    help="How would changing maximum segment length affect segmentation?",
                )

            with col2:
                st.markdown("**📈 Citation Support (Dynamic):**")
                st.info(
                    "Citation boost is now **dynamic**: 50% of base confidence\n\n"
                    "Cannot be adjusted - this is the optimized calculation method"
                )
                test_citation_boost = 0.5  # Fixed at 50% multiplier

                if test_keyword_filtering:
                    test_min_papers_ratio = st.slider(
                        "Test Min Papers Ratio",
                        min_value=0.01,
                        max_value=0.20,
                        value=algorithm_config.keyword_min_papers_ratio,
                        step=0.01,
                        help="How would changing filtering threshold affect noise reduction?",
                    )
                else:
                    test_min_papers_ratio = algorithm_config.keyword_min_papers_ratio

                # Add adaptive granularity testing for current domain
                test_granularity = st.slider(
                    "Test Optimized Granularity",
                    min_value=1,
                    max_value=5,
                    value=3,
                    step=1,
                    help="How would different granularity levels with Bayesian-optimized parameters affect detection? "
                    "Level 3 uses baseline optimized parameters for the domain.",
                )

            # Calculate what-if results
            if st.button("🔄 Calculate What-If Results"):
                what_if_results = []

                for detail in decision_details:
                    # Recalculate with new parameters using dynamic citation boost (50% of base confidence)
                    citation_boost_amount = (0.5 * detail["base_confidence"]) if detail["citation_support"] else 0
                    new_final_confidence = min(
                        detail["base_confidence"] + citation_boost_amount,
                        1.0,
                    )
                    
                    # Use standard validation threshold (permissive mode removed)
                    effective_validation_threshold = test_validation_threshold
                    
                    new_decision = (
                        "ACCEPTED"
                        if new_final_confidence >= effective_validation_threshold
                        else "REJECTED"
                    )

                    # Estimate keyword filtering impact
                    keyword_filtering_impact = ""
                    if (
                        test_keyword_filtering
                        != algorithm_config.keyword_filtering_enabled
                    ):
                        if test_keyword_filtering:
                            keyword_filtering_impact = " (with filtering)"
                        else:
                            keyword_filtering_impact = " (without filtering)"

                    # Granularity impact
                    granularity_impact = ""
                    if test_granularity != 3:
                        granularity_impact = f" [Granularity {test_granularity}]"

                    # Check for segment length parameter changes
                    segment_length_changes = ""
                    if (test_min_segment_length != algorithm_config.similarity_min_segment_length
                        or test_max_segment_length != algorithm_config.similarity_max_segment_length):
                        segment_length_changes = f" (Lengths: {test_min_segment_length}-{test_max_segment_length}yr)"

                    change_indicator = ""
                    if new_decision != detail["decision_outcome"]:
                        if new_decision == "ACCEPTED":
                            change_indicator = f"📈 Now Accepted{keyword_filtering_impact}{granularity_impact}{segment_length_changes}"
                        else:
                            change_indicator = f"📉 Now Rejected{keyword_filtering_impact}{granularity_impact}{segment_length_changes}"
                    else:
                        change_indicator = f"➡️ No Change{keyword_filtering_impact}{granularity_impact}{segment_length_changes}"

                    what_if_results.append(
                        {
                            "Year": str(detail["year"]),
                            "Original Decision": detail["decision_outcome"],
                            "New Final Confidence": f"{new_final_confidence:.3f}",
                            "New Decision": new_decision,
                            "Keyword Filtering": (
                                "🔍 Applied"
                                if test_keyword_filtering
                                else "🔍 Disabled"
                            ),
                            "Analysis Mode": "🔒 Conservative",  # Fixed since permissive mode removed
                            "Granularity": f"📊 Level {test_granularity}",
                            "Change": change_indicator,
                        }
                    )

                df_what_if = pd.DataFrame(what_if_results)

                # Color-code changes
                def color_change(val):
                    if "Now Accepted" in str(val):
                        return "background-color: #d1ecf1; color: #0c5460"
                    elif "Now Rejected" in str(val):
                        return "background-color: #f8d7da; color: #721c24"
                    return ""

                def color_filtering_whatif(val):
                    if "🔍 Applied" in str(val):
                        return "background-color: #d1ecf1"
                    return ""

                def color_analysis_mode(val):
                    if "🔒 Conservative" in str(val):
                        return "background-color: #fff3cd"
                    return ""

                def color_granularity(val):
                    if "📊 Level" in str(val):
                        return "background-color: #f8f9fa"
                    return ""

                styled_what_if = df_what_if.style.map(color_change, subset=["Change"])
                styled_what_if = styled_what_if.map(
                    color_filtering_whatif, subset=["Keyword Filtering"]
                )
                styled_what_if = styled_what_if.map(
                    color_analysis_mode, subset=["Analysis Mode"]
                )
                styled_what_if = styled_what_if.map(
                    color_granularity, subset=["Granularity"]
                )
                st.dataframe(styled_what_if, use_container_width=True, hide_index=True)

                # Summary of changes with Bayesian-optimized parameters and analysis mode context
                changes_count = sum(
                    1 for r in what_if_results if "No Change" not in r["Change"]
                )
                context_notes = []

                if test_keyword_filtering != algorithm_config.keyword_filtering_enabled:
                    context_notes.append(
                        f"keyword filtering {'enabled' if test_keyword_filtering else 'disabled'}"
                    )

                if test_granularity != 3:
                    context_notes.append(
                        f"granularity changed to level {test_granularity}"
                    )

                if (test_min_segment_length != algorithm_config.similarity_min_segment_length
                    or test_max_segment_length != algorithm_config.similarity_max_segment_length):
                    context_notes.append(
                        f"segment length controls changed to {test_min_segment_length}-{test_max_segment_length} years"
                    )

                context_note = f" ({', '.join(context_notes)})" if context_notes else ""

                if changes_count > 0:
                    st.info(
                        f"💡 **Impact Summary**: {changes_count} out of {len(what_if_results)} signals would change decisions with these parameters{context_note}"
                    )
                else:
                    st.success(
                        f"✅ **No Impact**: These parameter changes would not affect any decisions{context_note}"
                    )

        else:
            st.info(
                "No signals detected to analyze. Try adjusting the algorithm parameters to detect signals."
            )


if __name__ == "__main__":
    main()
