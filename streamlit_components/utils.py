"""
Shared utilities for the Timeline Segmentation Algorithm Dashboard.
Contains functions used across multiple tabs and components.
"""

import streamlit as st
from typing import Dict, Tuple, Any, List
import torch
import numpy as np

# Core algorithm imports
from core.data_loader import discover_available_domains
from core.data_processing import process_domain_data
from core.algorithm_config import AlgorithmConfig
from core.shift_signal_detection import (
    detect_shift_signals,
    detect_research_direction_changes,
    detect_citation_structural_breaks,
)
# from core.change_detection import create_segments_with_confidence  # REMOVED: Only using similarity segmentation now
from core.data_models import DomainData
from core.keyword_utils import extract_year_keywords

# Suppress PyTorch warnings
torch.classes.__path__ = []


@st.cache_data
def load_all_domains() -> Dict[str, DomainData]:
    """Load data for all available domains."""
    # Discover available domains dynamically
    available_domains = discover_available_domains()

    domain_data = {}

    for domain in available_domains.keys():
        try:
            # Use proper data processing function
            processing_result = process_domain_data(domain)
            if processing_result.success:
                domain_data[domain] = processing_result.domain_data
            else:
                st.warning(
                    f"Failed to process {domain}: {processing_result.error_message}"
                )
        except Exception as e:
            st.warning(f"Failed to load {domain}: {e}")

    return domain_data


def run_algorithm_with_params(
    domain_data: DomainData,
    domain_name: str,
    algorithm_config: AlgorithmConfig,
) -> Tuple[Any, Any, List[List[int]], Dict]:
    """Run the algorithm with specified parameters and return all signal data for visualization."""

    # Get raw direction signals data WITH analysis data for visualization
    raw_direction_signals, keyword_analysis = detect_research_direction_changes(
        domain_data, algorithm_config, return_analysis_data=True
    )

    # Get citation signals
    citation_signals = detect_citation_structural_breaks(domain_data, domain_name)

    # Create precomputed signals dict to avoid re-running detection
    precomputed_signals = {
        "direction": raw_direction_signals,
        "citation": citation_signals,
    }

    # Detect shift signals using precomputed signals - now returns clustering metadata
    shift_signals = detect_shift_signals(
        domain_data,
        domain_name,
        algorithm_config=algorithm_config,
        precomputed_signals=precomputed_signals,
    )

    # Create segments using similarity segmentation approach
    if shift_signals:
        # Import the similarity segmentation function
        from core.similarity_segmentation import create_similarity_based_segments
        
        # Extract year keywords for similarity analysis
        year_keywords = extract_year_keywords(domain_data)
        
        # Create similarity-based segments with length controls
        similarity_segments, similarity_metadata = create_similarity_based_segments(
            shift_signals, 
            year_keywords, 
            domain_data,
            min_segment_length=algorithm_config.similarity_min_segment_length,
            max_segment_length=algorithm_config.similarity_max_segment_length
        )
        
        # Convert to expected format (list of [start, end] lists)
        segments = [[start, end] for start, end in similarity_segments]
    else:
        # No validated signals - create single segment covering entire domain
        min_year, max_year = domain_data.year_range
        segments = [[min_year, max_year]]
        similarity_metadata = {}

    # Create enhanced signal data with all the metadata expected by dashboard
    enhanced_signal_data = {
        "keyword_analysis": keyword_analysis,  # From direction detection
        "clustering_metadata": {
            "citation_signals": citation_signals,
            "processed_direction_signals": raw_direction_signals,  # Use correct key expected by dashboard
            "validated_signals": shift_signals,
        },
        "similarity_metadata": similarity_metadata,  # From segmentation
        "segments": segments,  # Also store in enhanced_signal_data for fallback
    }

    # Create basic transition evidence (placeholder - not actively used)
    transition_evidence = {
        "transition_count": len(shift_signals),
        "signal_years": [s.year for s in shift_signals],
        "average_confidence": sum(s.confidence for s in shift_signals) / len(shift_signals) if shift_signals else 0.0,
    }

    return shift_signals, transition_evidence, segments, enhanced_signal_data