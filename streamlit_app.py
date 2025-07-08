"""Timeline Analysis Algorithm Visualization
A comprehensive Streamlit app to visualize and interact with each step of the academic literature timeline analysis algorithm.
"""

import streamlit as st
import time
from pathlib import Path

# Configure page
st.set_page_config(
    page_title="Timeline Analysis Algorithm Visualization",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import page modules
from streamlit_pages.data_exploration import show_data_exploration
from streamlit_pages.stage2_change_detection import show_change_detection
from streamlit_pages.stage3_segmentation import show_segmentation
from streamlit_pages.stage4_characterization import show_characterization
from streamlit_pages.stage5_merging import show_merging
from streamlit_pages.final_results import show_final_results
from core.utils.config import AlgorithmConfig
from core.utils.general import discover_available_domains

def initialize_session_state():
    """Initialize session state variables for the app."""
    if 'selected_domain' not in st.session_state:
        st.session_state.selected_domain = None
    if 'algorithm_config' not in st.session_state:
        st.session_state.algorithm_config = None
    if 'academic_years' not in st.session_state:
        st.session_state.academic_years = None
    if 'boundary_years' not in st.session_state:
        st.session_state.boundary_years = None
    if 'initial_periods' not in st.session_state:
        st.session_state.initial_periods = None
    if 'characterized_periods' not in st.session_state:
        st.session_state.characterized_periods = None
    if 'final_periods' not in st.session_state:
        st.session_state.final_periods = None
    if 'timing_data' not in st.session_state:
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
            index=0 if not st.session_state.selected_domain else available_domains.index(st.session_state.selected_domain) if st.session_state.selected_domain in available_domains else 0,
            help="Choose the academic domain to analyze"
        )
        
        if selected_domain != st.session_state.selected_domain:
            st.session_state.selected_domain = selected_domain
            # Reset downstream data when domain changes
            st.session_state.academic_years = None
            st.session_state.boundary_years = None
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
        st.session_state.algorithm_config = AlgorithmConfig.from_config_file(domain_name=selected_domain)
    
    config = st.session_state.algorithm_config
    
    # Detection parameters
    with st.sidebar.expander("🎯 Detection Parameters", expanded=True):
        direction_threshold = st.slider(
            "Direction Threshold",
            min_value=0.01,
            max_value=0.5,
            value=config.direction_threshold,
            step=0.01,
            help="Sensitivity for paradigm shift detection"
        )
        
        validation_threshold = st.slider(
            "Validation Threshold", 
            min_value=0.1,
            max_value=0.9,
            value=config.validation_threshold,
            step=0.05,
            help="Signal combination threshold"
        )
        
        citation_boost_rate = st.slider(
            "Citation Boost Rate",
            min_value=0.0,
            max_value=1.0,
            value=config.citation_boost_rate,
            step=0.1,
            help="Weight for citation signal"
        )
    
    # Objective function parameters  
    with st.sidebar.expander("🎯 Objective Function", expanded=False):
        cohesion_weight = st.slider(
            "Cohesion Weight",
            min_value=0.0,
            max_value=1.0,
            value=config.cohesion_weight,
            step=0.1,
            help="Weight for intra-period coherence"
        )
        
        separation_weight = 1.0 - cohesion_weight
        st.write(f"Separation Weight: {separation_weight:.1f}")
        
        top_k_keywords = st.slider(
            "Top K Keywords",
            min_value=5,
            max_value=50,
            value=config.top_k_keywords,
            step=5,
            help="Number of keywords for evaluation"
        )
    
    # Anti-gaming parameters
    with st.sidebar.expander("🛡️ Anti-Gaming", expanded=False):
        min_segment_size = st.slider(
            "Min Segment Size",
            min_value=10,
            max_value=200,
            value=config.anti_gaming_min_segment_size,
            step=10,
            help="Minimum papers per segment"
        )
        
        size_weight_power = st.slider(
            "Size Weight Power",
            min_value=0.0,
            max_value=2.0,
            value=config.anti_gaming_size_weight_power,
            step=0.1,
            help="Power for size weighting"
        )
    
    # Merging parameters
    with st.sidebar.expander("🔗 Merging", expanded=False):
        merge_similarity_threshold = st.slider(
            "Merge Similarity Threshold",
            min_value=0.1,
            max_value=0.9,
            value=getattr(st.session_state, 'merge_similarity_threshold', 0.6),
            step=0.05,
            help="Keyword overlap threshold for merging adjacent periods"
        )
    
    # Update config if any parameter changed
    import dataclasses
    
    # Create new config dict without merge_similarity_threshold (not part of AlgorithmConfig)
    config_dict = dataclasses.asdict(config)
    config_dict.update({
        'direction_threshold': direction_threshold,
        'validation_threshold': validation_threshold,
        'citation_boost_rate': citation_boost_rate,
        'cohesion_weight': cohesion_weight,
        'separation_weight': separation_weight,
        'top_k_keywords': top_k_keywords,
        'anti_gaming_min_segment_size': min_segment_size,
        'anti_gaming_size_weight_power': size_weight_power
    })
    
    # Store merge_similarity_threshold separately
    st.session_state.merge_similarity_threshold = merge_similarity_threshold
    
    new_config = type(config)(**config_dict)
    
    if new_config != config:
        st.session_state.algorithm_config = new_config
        # Reset results that depend on config
        st.session_state.boundary_years = None
        st.session_state.initial_periods = None
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
            "✂️ Stage 3: Segmentation",
            "🏷️ Stage 4: Characterization",
            "🔗 Stage 5: Merging",
            "🎯 Final Results"
        ],
        horizontal=True
    )
    
    # Route to appropriate page
    if page == "📁 Data Exploration":
        show_data_exploration()
    elif page == "🔍 Stage 2: Change Detection":
        show_change_detection()
    elif page == "✂️ Stage 3: Segmentation":
        show_segmentation()
    elif page == "🏷️ Stage 4: Characterization":
        show_characterization()
    elif page == "🔗 Stage 5: Merging":
        show_merging()
    elif page == "🎯 Final Results":
        show_final_results()

if __name__ == "__main__":
    main() 