"""Timeline Analysis Algorithm Visualization
A comprehensive Streamlit app to visualize and interact with each step of the academic literature timeline analysis algorithm.
"""

import streamlit as st

# Configure page
st.set_page_config(
    page_title="Timeline Analysis Algorithm Visualization",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Import page modules
from streamlit_pages.data_exploration import show_data_exploration
from streamlit_pages.stage2_change_detection import show_change_detection
from streamlit_pages.stage3_segmentation import show_segmentation
from streamlit_pages.stage4_characterization import show_characterization
from streamlit_pages.final_results import show_final_results
from streamlit_pages.evaluation import show_evaluation
from core.utils.config import AlgorithmConfig
from core.utils.general import discover_available_domains


def initialize_session_state():
    """Initialize session state variables for the app."""
    if "selected_domain" not in st.session_state:
        st.session_state.selected_domain = None
    if "algorithm_config" not in st.session_state:
        st.session_state.algorithm_config = None
    if "academic_years" not in st.session_state:
        st.session_state.academic_years = None
    if "boundary_years" not in st.session_state:
        st.session_state.boundary_years = None
    if "citation_acceleration_years" not in st.session_state:
        st.session_state.citation_acceleration_years = None
    if "initial_periods" not in st.session_state:
        st.session_state.initial_periods = None
    if "refined_periods" not in st.session_state:
        st.session_state.refined_periods = None
    if "characterized_periods" not in st.session_state:
        st.session_state.characterized_periods = None
    if "final_periods" not in st.session_state:
        st.session_state.final_periods = None
    if "timing_data" not in st.session_state:
        st.session_state.timing_data = {}


def show_global_configuration():
    """Show global configuration sidebar."""
    st.sidebar.title("🔧 Global Configuration")

    # Domain selection
    available_domains = discover_available_domains()
    if available_domains:
        selected_domain = st.sidebar.selectbox(
            "Select Domain",
            options=available_domains,
            index=(
                0
                if not st.session_state.selected_domain
                else (
                    available_domains.index(st.session_state.selected_domain)
                    if st.session_state.selected_domain in available_domains
                    else 0
                )
            ),
            help="Choose the academic domain to analyze",
        )

        if selected_domain != st.session_state.selected_domain:
            st.session_state.selected_domain = selected_domain
            # Reset downstream data when domain changes
            st.session_state.academic_years = None
            st.session_state.boundary_years = None
            st.session_state.citation_acceleration_years = None
            st.session_state.initial_periods = None
            st.session_state.characterized_periods = None
            st.session_state.final_periods = None
            st.session_state.timing_data = {}
            st.rerun()
    else:
        st.sidebar.error("No domains found in resources directory")
        return False

    # Algorithm configuration
    st.sidebar.subheader("Algorithm Parameters")

    # Load default config
    if st.session_state.algorithm_config is None:
        st.session_state.algorithm_config = AlgorithmConfig.from_config_file(
            domain_name=selected_domain
        )

    config = st.session_state.algorithm_config

    # Detection parameters
    with st.sidebar.expander("🎯 Detection Parameters", expanded=True):
        direction_change_threshold = st.slider(
            "Direction Change Threshold",
            min_value=0.01,
            max_value=0.5,
            value=getattr(config, "direction_change_threshold", 0.1),
            step=0.01,
            help="Base threshold for direction change detection (may be adaptive)",
        )

        direction_threshold_strategy = st.selectbox(
            "Threshold Strategy",
            options=["fixed", "global_p90", "global_p95", "global_p99"],
            index=["fixed", "global_p90", "global_p95", "global_p99"].index(
                getattr(config, "direction_threshold_strategy", "global_p90")
            ),
            help="Strategy for computing adaptive thresholds",
        )

        direction_scoring_method = st.selectbox(
            "Scoring Method",
            options=["weighted_jaccard", "jensen_shannon"],
            index=["weighted_jaccard", "jensen_shannon"].index(
                getattr(config, "direction_scoring_method", "weighted_jaccard")
            ),
            help="Method for computing direction change scores",
        )

        citation_confidence_boost = st.slider(
            "Citation Confidence Boost",
            min_value=0.0,
            max_value=1.0,
            value=getattr(config, "citation_confidence_boost", 0.5),
            step=0.1,
            help="Boost factor when citation acceleration supports direction change",
        )

        citation_support_window_years = st.slider(
            "Citation Support Window (Years)",
            min_value=1,
            max_value=5,
            value=getattr(config, "citation_support_window_years", 2),
            step=1,
            help="Window around direction change to look for citation support",
        )

        min_papers_per_year = st.slider(
            "Min Papers Per Year (Direction)",
            min_value=10,
            max_value=500,
            value=getattr(config, "min_papers_per_year", 100),
            step=10,
            help="Minimum papers per year to consider for direction change detection",
        )

        min_baseline_period_years = st.slider(
            "Min Baseline Period (Years)",
            min_value=1,
            max_value=10,
            value=getattr(config, "min_baseline_period_years", 3),
            step=1,
            help="Minimum years for baseline period in cumulative approach",
        )

    # Objective function parameters
    with st.sidebar.expander("🎯 Objective Function", expanded=False):
        cohesion_weight = st.slider(
            "Cohesion Weight",
            min_value=0.0,
            max_value=1.0,
            value=getattr(config, "cohesion_weight", 0.8),
            step=0.1,
            help="Weight for intra-period coherence",
        )

        separation_weight = 1.0 - cohesion_weight
        st.write(f"Separation Weight: {separation_weight:.1f}")

        top_k_keywords = st.slider(
            "Top K Keywords",
            min_value=5,
            max_value=50,
            value=getattr(config, "top_k_keywords", 50),
            step=5,
            help="Number of keywords for evaluation",
        )

        min_keyword_frequency_ratio = st.slider(
            "Min Keyword Frequency Ratio",
            min_value=0.0,
            max_value=0.5,
            value=getattr(config, "min_keyword_frequency_ratio", 0.05),
            step=0.01,
            help="Minimum ratio of papers a keyword must appear in (0.05 = 5%)",
        )

    # Beam search refinement parameters
    with st.sidebar.expander("🔍 Beam Search Refinement", expanded=False):
        beam_search_enabled = st.checkbox(
            "Enable Beam Search",
            value=getattr(config, "beam_search_enabled", True),
            help="Enable beam search optimization after initial segmentation",
        )

        beam_width = st.slider(
            "Beam Width",
            min_value=1,
            max_value=20,
            value=getattr(config, "beam_width", 5),
            step=1,
            help="Number of best states to keep in each iteration",
        )

        max_splits_per_segment = st.slider(
            "Max Splits Per Segment",
            min_value=0,
            max_value=5,
            value=getattr(config, "max_splits_per_segment", 1),
            step=1,
            help="Maximum number of splits allowed per initial segment",
        )

        min_period_years = st.slider(
            "Min Period Years",
            min_value=1,
            max_value=10,
            value=getattr(config, "min_period_years", 3),
            step=1,
            help="Minimum years for any period (except edge periods)",
        )

        max_period_years = st.slider(
            "Max Period Years",
            min_value=10,
            max_value=100,
            value=getattr(config, "max_period_years", 50),
            step=5,
            help="Maximum years for any period",
        )

    # Diagnostic parameters
    with st.sidebar.expander("🔧 Diagnostics", expanded=False):
        save_direction_diagnostics = st.checkbox(
            "Save Direction Diagnostics",
            value=getattr(config, "save_direction_diagnostics", False),
            help="Save detailed diagnostics for direction change detection",
        )

        diagnostic_top_keywords_limit = st.slider(
            "Diagnostic Keywords Limit",
            min_value=5,
            max_value=20,
            value=getattr(config, "diagnostic_top_keywords_limit", 10),
            step=1,
            help="Number of top keywords to show in diagnostics",
        )

        score_distribution_window_years = st.slider(
            "Score Distribution Window",
            min_value=1,
            max_value=10,
            value=getattr(config, "score_distribution_window_years", 3),
            step=1,
            help="Interval for sampling scores for threshold calculation",
        )

    # Update config if any parameter changed
    import dataclasses

    # Create new config dict
    config_dict = dataclasses.asdict(config)
    config_dict.update(
        {
            "direction_change_threshold": direction_change_threshold,
            "direction_threshold_strategy": direction_threshold_strategy,
            "direction_scoring_method": direction_scoring_method,
            "citation_confidence_boost": citation_confidence_boost,
            "citation_support_window_years": citation_support_window_years,
            "min_papers_per_year": min_papers_per_year,
            "min_baseline_period_years": min_baseline_period_years,
            "cohesion_weight": cohesion_weight,
            "separation_weight": separation_weight,
            "top_k_keywords": top_k_keywords,
            "min_keyword_frequency_ratio": min_keyword_frequency_ratio,
            "beam_search_enabled": beam_search_enabled,
            "beam_width": beam_width,
            "max_splits_per_segment": max_splits_per_segment,
            "min_period_years": min_period_years,
            "max_period_years": max_period_years,
            "save_direction_diagnostics": save_direction_diagnostics,
            "diagnostic_top_keywords_limit": diagnostic_top_keywords_limit,
            "score_distribution_window_years": score_distribution_window_years,
        }
    )

    new_config = type(config)(**config_dict)

    if new_config != config:
        st.session_state.algorithm_config = new_config
        # Reset results that depend on config
        st.session_state.boundary_years = None
        st.session_state.citation_acceleration_years = None
        st.session_state.initial_periods = None
        st.session_state.refined_periods = None
        st.session_state.characterized_periods = None
        st.session_state.final_periods = None
        st.rerun()

    return True


def show_timing_summary():
    """Show timing summary in sidebar."""
    if st.session_state.timing_data:
        st.sidebar.subheader("⏱️ Performance")
        for stage, timing in st.session_state.timing_data.items():
            st.sidebar.metric(stage, f"{timing:.2f}s")

        total_time = sum(st.session_state.timing_data.values())
        st.sidebar.metric("Total Time", f"{total_time:.2f}s")


def main():
    """Main application function."""
    # Initialize session state
    initialize_session_state()

    # Show global configuration
    if not show_global_configuration():
        st.error("Please ensure domains are available in the resources directory")
        return

    # Show timing summary
    show_timing_summary()

    # Main navigation
    st.title("📊 Timeline Analysis Algorithm Visualization")
    st.write("Interactive visualization of academic literature timeline segmentation")

    # Page navigation
    page = st.radio(
        "Navigate to:",
        options=[
            "📁 Data Exploration",
            "🔍 Stage 2: Change Detection",
            "✂️ Stage 3: Segmentation & Refinement",
            "🏷️ Stage 4: Characterization",
            "🎯 Final Results",
            "📊 Evaluation",
        ],
        horizontal=True,
    )

    # Route to appropriate page
    if page == "📁 Data Exploration":
        show_data_exploration()
    elif page == "🔍 Stage 2: Change Detection":
        show_change_detection()
    elif page == "✂️ Stage 3: Segmentation & Refinement":
        show_segmentation()
    elif page == "🏷️ Stage 4: Characterization":
        show_characterization()
    elif page == "🎯 Final Results":
        show_final_results()
    elif page == "📊 Evaluation":
        show_evaluation()


if __name__ == "__main__":
    main()
