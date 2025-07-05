"""
Pipeline Orchestrator for Timeline Analysis

This module orchestrates the complete timeline analysis pipeline by coordinating:
1. Change point detection and shift signal analysis
2. Timeline analysis with period characterization
3. Segment merging post-processing
4. Complete analysis workflow management

Separated from integration.py to follow single responsibility principle.
Provides clean orchestration of the core analysis algorithms.
"""

import time
import numpy as np
from typing import Dict, List, Tuple, Optional, Any

from ..data.models import (
    DomainData, ChangeDetectionResult, TimelineAnalysisResult
)
from ..data.processing import process_domain_data
from ..detection.change_detection import detect_changes
from ..segmentation.modeling import model_segments
from ..segmentation.merging import merge_similar_segments
from ..detection.shift_signals import detect_shift_signals
from ..segmentation.boundary import create_boundary_segments
from ..utils.config import AlgorithmConfig
from ..results.manager import save_all_results
from ..utils.logging import get_logger


def run_complete_analysis(
    domain_name: str = None,
    domain_data: Optional[DomainData] = None,
    algorithm_config: Optional[AlgorithmConfig] = None,
    segmentation_only: bool = False,
    save_results: bool = True,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Run complete timeline analysis pipeline with optional result saving.
    
    This is the main entry point for the timeline analysis system,
    orchestrating all components in the correct sequence.
    
    Args:
        domain_name: Name of the domain to process (required if domain_data not provided)
        domain_data: Optional pre-loaded domain data
        algorithm_config: Algorithm configuration
        segmentation_only: If True, only run segmentation (skip timeline analysis)
        save_results: Whether to save results to files
        verbose: Enable verbose logging
        
    Returns:
        Dictionary containing all analysis results and file paths
    """
    logger = get_logger(__name__, verbose)
    start_time = time.time()
    
    # Step 1: Load domain data if not provided
    if domain_data is None:
        if domain_name is None:
            raise ValueError("Either domain_name or domain_data must be provided")
        
        logger.info(f"Loading domain data: {domain_name}")
        result = process_domain_data(domain_name)
        if not result.success:
            return {
                'success': False,
                'error': f"Error loading {domain_name}: {result.error_message}",
                'execution_time': time.time() - start_time
            }
        domain_data = result.domain_data
    else:
        domain_name = domain_data.domain_name
    
    # Step 2: Run change detection and segmentation
    segmentation_results, change_detection_result, shift_signals = run_change_detection(
        domain_name=domain_name,
        algorithm_config=algorithm_config,
        domain_data=domain_data,
        verbose=verbose
    )
    
    if not segmentation_results:
        return {
            'success': False,
            'error': 'Change detection failed',
            'execution_time': time.time() - start_time
        }
    
    # Step 3: Handle segmentation-only case
    if segmentation_only:
        segments = segmentation_results.get('segments', [])
        return {
            'success': True,
            'domain_name': domain_name,
            'segments': segments,
            'segmentation_results': segmentation_results,
            'execution_time': time.time() - start_time
        }
    
    # Step 4: Run timeline analysis
    segments = [(start, end) for start, end in segmentation_results['segments']]
    
    timeline_result = run_timeline_analysis(
        domain_data=domain_data,
        segments=segments,
        change_detection_result=change_detection_result,
        precomputed_shift_signals=shift_signals,
        enable_segment_merging=True,
        similarity_threshold=0.75,
        weak_signal_threshold=0.4,
        granularity=segmentation_results['granularity'],
        verbose=verbose
    )
    
    if not timeline_result:
        return {
            'success': False,
            'error': 'Timeline analysis failed',
            'execution_time': time.time() - start_time
        }
    
    # Step 5: Save results if requested
    saved_files = {}
    if save_results:
        saved_files = save_all_results(
            timeline_result, segmentation_results, shift_signals, domain_data
        )
    
    # Step 6: Return comprehensive results
    return {
        'success': True,
        'domain_name': domain_name,
        'timeline_result': timeline_result,
        'segmentation_results': segmentation_results,
        'shift_signals': shift_signals,
        'period_characterizations': list(timeline_result.period_characterizations),
        'saved_files': saved_files,
        'execution_time': time.time() - start_time
    }


def run_timeline_analysis(
    domain_data: DomainData,
    segments: List[Tuple[int, int]],
    change_detection_result: Optional[ChangeDetectionResult] = None,
    precomputed_shift_signals: Optional[List] = None,
    enable_segment_merging: bool = True,
    similarity_threshold: float = 0.75,
    weak_signal_threshold: float = 0.4,
    granularity: int = 3,
    verbose: bool = False
) -> TimelineAnalysisResult:
    """
    Unified timeline analysis pipeline combining shift and period signal detection with segment merging.
    
    Args:
        domain_data: Domain data with papers and citations
        segments: Timeline segments from shift signal detection
        change_detection_result: Optional pre-computed change detection
        precomputed_shift_signals: Optional pre-computed shift signals to avoid recomputation
        enable_segment_merging: Whether to perform segment merging post-processing
        similarity_threshold: Threshold for semantic similarity in merging (0.0-1.0)
        weak_signal_threshold: Threshold for weak shift signals in merging (0.0-1.0)
        granularity: Timeline granularity control (1-5)
        verbose: Enable verbose logging
        
    Returns:
        Timeline analysis results with period characterizations and optional merging
    """    
    logger = get_logger(__name__, verbose)
    
    # Step 1: Segment modeling using period signal detection
    modeling_result = model_segments(
        domain_name=domain_data.domain_name,
        segments=segments,
        verbose=verbose
    )
    
    period_characterizations = list(modeling_result.period_characterizations)
    
    # Step 2: Optional segment merging post-processing
    merged_period_characterizations = period_characterizations
    merging_result = None
    
    if enable_segment_merging and len(period_characterizations) >= 2:
        logger.info("SEGMENT MERGING POST-PROCESSING")
        
        # Use precomputed shift signals if available, otherwise compute them
        if precomputed_shift_signals is not None:
            shift_signals = precomputed_shift_signals
            logger.debug(f"Using precomputed shift signals for merging: {len(shift_signals)} signals")
        else:
            # Create comprehensive algorithm config for shift signal detection
            algorithm_config = AlgorithmConfig(granularity=granularity)
            
            # Get shift signals for boundary analysis
            shift_signals = detect_shift_signals(domain_data, domain_data.domain_name, algorithm_config, verbose=verbose)
            logger.debug(f"Computed shift signals for merging: {len(shift_signals)} signals")
        
        # Perform intelligent segment merging
        merging_result = merge_similar_segments(
            period_characterizations=period_characterizations,
            shift_signals=shift_signals,
            domain_name=domain_data.domain_name,
            similarity_threshold=similarity_threshold,
            weak_signal_threshold=weak_signal_threshold,
            verbose=verbose
        )
        
        merged_period_characterizations = list(merging_result.merged_segments)
        
        logger.info("Segment merging completed:")
        logger.info(f"Original segments: {len(period_characterizations)}")
        logger.info(f"Merged segments: {len(merged_period_characterizations)}")
    else:
        if not enable_segment_merging:
            logger.debug("Segment merging disabled")
        else:
            logger.debug(f"Segment merging skipped - insufficient segments ({len(period_characterizations)})")
    
    # Step 3: Calculate unified confidence using final merged segments
    if merged_period_characterizations:
        unified_confidence = np.mean([pc.confidence for pc in merged_period_characterizations])
    else:
        unified_confidence = 0.0
    
    # Step 4: Generate narrative evolution using final merged segments
    narrative_parts = []
    for pc in merged_period_characterizations:
        narrative_parts.append(f"{pc.period[0]}-{pc.period[1]}: {pc.topic_label}")
    
    narrative_evolution = " → ".join(narrative_parts)
    
    return TimelineAnalysisResult(
        domain_name=domain_data.domain_name,
        period_characterizations=tuple(period_characterizations),
        merged_period_characterizations=tuple(merged_period_characterizations),
        merging_result=merging_result,
        unified_confidence=unified_confidence,
        narrative_evolution=narrative_evolution
    )


def run_change_detection(
    domain_name: str = None, 
    granularity: int = 3, 
    algorithm_config: Optional[AlgorithmConfig] = None,
    domain_data: Optional[DomainData] = None,
    verbose: bool = False
) -> Tuple[Optional[Dict], Optional[ChangeDetectionResult], Optional[List]]:
    """
    Run change point detection and segmentation for a domain.
    
    Args:
        domain_name: Name of the domain to process (required if domain_data not provided)
        granularity: Timeline granularity control (1-5, used if algorithm_config is None)
        algorithm_config: Optional comprehensive algorithm configuration
        domain_data: Optional pre-loaded domain data (avoids reloading)
        verbose: Enable verbose logging
        
    Returns:
        Tuple of (segmentation_results, change_detection_result, shift_signals)
    """
    logger = get_logger(__name__, verbose)
    
    # Use pre-loaded domain data or load it
    if domain_data is not None:
        logger.info(f"CHANGE POINT DETECTION: {domain_data.domain_name}")
        domain_name = domain_data.domain_name
    else:
        if domain_name is None:
            raise ValueError("Either domain_name or domain_data must be provided")
        
        logger.info(f"CHANGE POINT DETECTION: {domain_name}")
        
        # Load domain data
        result = process_domain_data(domain_name)
        if not result.success:
            logger.error(f"Error loading {domain_name}: {result.error_message}")
            return None, None, None
        
        domain_data = result.domain_data
    
    # Create or use comprehensive algorithm configuration
    if algorithm_config is None:
        algorithm_config = AlgorithmConfig(granularity=int(granularity))
    else:
        granularity = algorithm_config.granularity

    # Run change detection
    logger.info("CHANGE DETECTION")
    change_result, shift_signals = detect_changes(domain_data, algorithm_config=algorithm_config, verbose=verbose)
    
    # Extract change point years
    change_years = sorted([cp.year for cp in change_result.change_points])
    
    # Create segments using boundary segmentation
    logger.info("BOUNDARY SEGMENTATION")
    if shift_signals:
        # Create boundary-based segments using signal years directly
        boundary_segments, boundary_metadata = create_boundary_segments(
            shift_signals, 
            domain_data,
            verbose=verbose
        )
        
        # Convert to expected format
        segments = [[start, end] for start, end in boundary_segments]
        
        logger.info(f"Created {len(segments)} segments from {len(shift_signals)} validated signals")
        for i, (start, end) in enumerate(boundary_segments):
            logger.debug(f"  Segment {i+1}: {start}-{end} ({end-start+1} years)")
    else:
        logger.warning("No validated signals found - using single segment")
        segments = [[domain_data.year_range[0], domain_data.year_range[1]]]
    
    # Prepare results
    results = {
        'domain_name': domain_name,
        'granularity': granularity,
        'algorithm_config': {
            'direction_threshold': algorithm_config.direction_threshold,
            'citation_boost_rate': algorithm_config.citation_boost_rate,
            'validation_threshold': algorithm_config.validation_threshold,
            'citation_support_window': algorithm_config.citation_support_window,
            'keyword_min_frequency': algorithm_config.keyword_min_frequency,
            'min_significant_keywords': algorithm_config.min_significant_keywords,
            'segmentation_method': 'boundary_based'
        },
        'time_range': list(domain_data.year_range),
        'change_points': change_years,
        'segments': segments,
        'statistical_significance': change_result.statistical_significance
    }
    
    return results, change_result, shift_signals


# Export functions
__all__ = [
    'run_complete_analysis',
    'run_timeline_analysis', 
    'run_change_detection'
] 