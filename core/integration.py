"""
Timeline Analysis Integration Module

This module orchestrates the complete timeline analysis pipeline by integrating:
1. Change point detection and shift signal analysis
2. Period characterization through network stability analysis  
3. Segment merging for post-processing similar segments
4. Result saving and output management

Provides clean integration and orchestration of the core analysis algorithms.
"""

from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import json
import os
import time
from pathlib import Path
from datetime import datetime

from .data_models import (
    DomainData, ChangeDetectionResult, TimelineAnalysisResult, PeriodCharacterization, ShiftSignal
)
from .change_detection import detect_changes
from .data_processing import process_domain_data
from .segment_modeling import model_segments
from .segment_merging import merge_similar_segments
from .shift_signal_detection import detect_shift_signals
from .algorithm_config import AlgorithmConfig
from .similarity_segmentation import create_similarity_based_segments
from .keyword_utils import extract_year_keywords


def save_shift_signals(shift_signals: List[ShiftSignal], domain_name: str, domain_data: DomainData) -> str:
    """Save shift signals for visualization."""
    signals_dir = "results/signals"
    Path(signals_dir).mkdir(parents=True, exist_ok=True)
    
    shift_data = {
        'domain_name': domain_name,
        'analysis_date': datetime.now().isoformat(),
        'time_range': domain_data.year_range,
        'total_papers': len(domain_data.papers),
        'signals': [
            {
                'year': s.year,
                'confidence': s.confidence,
                'signal_type': s.signal_type,
                'evidence_strength': s.evidence_strength,
                'supporting_evidence': list(s.supporting_evidence),
                'transition_description': s.transition_description,
                'paradigm_significance': s.paradigm_significance
            }
            for s in shift_signals
        ],
        'summary': {
            'total_signals': len(shift_signals),
            'paradigm_shifts': len([s for s in shift_signals if s.paradigm_significance >= 0.5]),
            'avg_confidence': sum(s.confidence for s in shift_signals) / max(len(shift_signals), 1)
        }
    }
    
    # Save files
    shift_file = f"{signals_dir}/{domain_name}_shift_signals.json"
    with open(shift_file, 'w') as f:
        json.dump(shift_data, f, indent=2)
    
    shift_latest = f"results/{domain_name}_shift_signals.json"
    os.makedirs("results", exist_ok=True)
    with open(shift_latest, 'w') as f:
        json.dump(shift_data, f, indent=2)
    
    return shift_file


def save_period_signals(period_characterizations: List[PeriodCharacterization], 
                       segments: List[Tuple[int, int]], domain_name: str) -> str:
    """Save period signals for visualization."""
    signals_dir = "results/signals"
    Path(signals_dir).mkdir(parents=True, exist_ok=True)
    
    period_data = {
        'domain_name': domain_name,
        'analysis_date': datetime.now().isoformat(),
        'segments': [{'start_year': s[0], 'end_year': s[1]} for s in segments],
        'characterizations': [
            {
                'period': char.period,
                'topic_label': char.topic_label,
                'network_stability': char.network_stability,
                'confidence': char.confidence,
                'representative_papers': [
                    {
                        'title': paper.get('title', ''),
                        'year': paper.get('year', 0),
                        'citation_count': paper.get('citation_count', 0)
                    } for paper in list(char.representative_papers)[:3]
                ]
            }
            for char in period_characterizations
        ],
        'summary': {
            'total_periods': len(period_characterizations),
            'avg_confidence': sum(c.confidence for c in period_characterizations) / max(len(period_characterizations), 1)
        }
    }
    
    # Save files
    period_file = f"{signals_dir}/{domain_name}_period_signals.json"
    with open(period_file, 'w') as f:
        json.dump(period_data, f, indent=2)
    
    period_latest = f"results/{domain_name}_period_signals.json"
    with open(period_latest, 'w') as f:
        json.dump(period_data, f, indent=2)
    
    return period_file


def save_analysis_results(timeline_result: TimelineAnalysisResult, segmentation_data: Dict[str, Any], 
                         domain_data: DomainData, output_path: str) -> None:
    """Save comprehensive analysis results to JSON file."""
    results = {
        'domain_name': timeline_result.domain_name,
        'analysis_date': datetime.now().isoformat(),
        'time_range': domain_data.year_range,
        'total_papers': len(domain_data.papers),
        'statistical_significance': segmentation_data.get('statistical_significance', 0),
        'change_points': segmentation_data.get('change_points', []),
        'segments': segmentation_data.get('segments', []),
        'periods': [
            {
                'period': list(pc.period),
                'topic_label': pc.topic_label,
                'network_stability': pc.network_stability,
                'representative_papers': [
                    {
                        'title': paper.get('title', '') if isinstance(paper, dict) else paper.title,
                        'year': paper.get('year', 0) if isinstance(paper, dict) else paper.pub_year,
                        'citations': paper.get('citations', 0) if isinstance(paper, dict) else paper.citation_count
                    } for paper in pc.representative_papers[:3]
                ],
                'confidence': pc.confidence
            }
            for pc in timeline_result.merged_period_characterizations
        ],
        'unified_confidence': timeline_result.unified_confidence
    }
    
    if 'algorithm_config' in segmentation_data:
        results['algorithm_config'] = segmentation_data['algorithm_config']
    
    # Save timestamped version
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save latest version in results root
    domain_name = timeline_result.domain_name
    latest_path = f"results/{domain_name}_analysis.json"
    os.makedirs("results", exist_ok=True)
    with open(latest_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {output_path} and {latest_path}")


def run_complete_analysis(
    domain_name: str = None,
    domain_data: Optional[DomainData] = None,
    algorithm_config: Optional[AlgorithmConfig] = None,
    segmentation_only: bool = False,
    save_results: bool = True
) -> Dict[str, Any]:
    """
    Run complete timeline analysis pipeline with optional result saving.
    
    Args:
        domain_name: Name of the domain to process (required if domain_data not provided)
        domain_data: Optional pre-loaded domain data
        algorithm_config: Algorithm configuration
        segmentation_only: If True, only run segmentation (skip timeline analysis)
        save_results: Whether to save results to files
        
    Returns:
        Dictionary containing all analysis results and file paths
    """
    start_time = time.time()
    
    # Step 1: Load domain data if not provided
    if domain_data is None:
        if domain_name is None:
            raise ValueError("Either domain_name or domain_data must be provided")
        
        print(f"Loading domain data: {domain_name}")
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
        domain_data=domain_data
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
    
    timeline_result = timeline_analysis(
        domain_data=domain_data,
        segments=segments,
        change_detection_result=change_detection_result,
        precomputed_shift_signals=shift_signals,
        enable_segment_merging=True,
        similarity_threshold=0.75,
        weak_signal_threshold=0.4,
        granularity=segmentation_results['granularity']
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
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        
        # Save main timeline analysis
        timeline_file = f"results/timelines/{domain_name}_timeline_analysis_{timestamp}.json"
        save_analysis_results(timeline_result, segmentation_results, domain_data, timeline_file)
        saved_files['timeline_analysis'] = timeline_file
        
        # Save shift signals
        shift_file = save_shift_signals(shift_signals, domain_name, domain_data)
        saved_files['shift_signals'] = shift_file
        
        # Save period signals
        period_characterizations = list(timeline_result.period_characterizations)
        period_file = save_period_signals(period_characterizations, segmentation_results['segments'], domain_name)
        saved_files['period_signals'] = period_file
    
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


def load_optimized_parameters(domain_name: str, params_file: str = None) -> dict:
    """Load optimized parameters if available."""
    file_path = params_file or "results/optimization/optimized_parameters_bayesian.json"
    
    if not os.path.exists(file_path):
        print("No optimized parameters found, using defaults")
        return {}
    
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        params = data.get('consensus_difference_optimized_parameters', {})
        if domain_name in params:
            print(f"Using optimized parameters for {domain_name}")
            return params[domain_name]
        else:
            print(f"No optimized parameters for {domain_name}, using defaults")
            return {}
    except Exception as e:
        print(f"Error loading parameters: {e}")
        return {}


def display_analysis_summary(results: Dict[str, Any]) -> None:
    """Display analysis summary from results dictionary."""
    if not results.get('success', False):
        print(f"Analysis failed: {results.get('error', 'Unknown error')}")
        return
    
    domain_name = results['domain_name']
    
    if 'segments' in results:  # Segmentation-only
        segments = results['segments']
        print(f"\nSEGMENTATION: {domain_name}")
        print(f"Segments: {len(segments)}")
        for i, (start, end) in enumerate(segments):
            print(f"  {i+1}. {start}-{end}")
        print(f"Completed in {results['execution_time']:.2f}s")
    else:  # Full analysis
        timeline_result = results['timeline_result']
        periods = timeline_result.merged_period_characterizations
        confidence = timeline_result.unified_confidence
        
        print(f"\nRESULTS: {domain_name}")
        print("=" * 40)
        print(f"Periods: {len(periods)}, Confidence: {confidence:.3f}")
        print(f"\nTIMELINE PERIODS:")
        for i, period in enumerate(periods):
            start, end = period.period
            topic = period.topic_label
            conf = period.confidence
            print(f"  {i+1}. {start}-{end}: {topic} (conf: {conf:.3f})")
        
        print(f"Completed in {results['execution_time']:.2f}s")


def timeline_analysis(
    domain_data: DomainData,
    segments: List[Tuple[int, int]],
    change_detection_result: Optional[ChangeDetectionResult] = None,
    precomputed_shift_signals: Optional[List] = None,
    enable_segment_merging: bool = True,
    similarity_threshold: float = 0.75,
    weak_signal_threshold: float = 0.4,
    granularity: int = 3
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
        
    Returns:
        Timeline analysis results with period characterizations and optional merging
    """    
    # Step 1: Segment modeling using period signal detection
    modeling_result = model_segments(
        domain_name=domain_data.domain_name,
        segments=segments
    )
    
    period_characterizations = list(modeling_result.period_characterizations)
    
    # Step 2: Optional segment merging post-processing
    merged_period_characterizations = period_characterizations
    merging_result = None
    
    if enable_segment_merging and len(period_characterizations) >= 2:
        print(f"SEGMENT MERGING POST-PROCESSING")
        
        # Use precomputed shift signals if available, otherwise compute them
        if precomputed_shift_signals is not None:
            shift_signals = precomputed_shift_signals
            print(f"Using precomputed shift signals for merging: {len(shift_signals)} signals")
        else:
            # Create comprehensive algorithm config for shift signal detection
            algorithm_config = AlgorithmConfig(granularity=granularity)
            
            # Get shift signals for boundary analysis
            shift_signals = detect_shift_signals(domain_data, domain_data.domain_name, algorithm_config)
            print(f"Computed shift signals for merging: {len(shift_signals)} signals")
        
        # Perform intelligent segment merging
        merging_result = merge_similar_segments(
            period_characterizations=period_characterizations,
            shift_signals=shift_signals,
            domain_name=domain_data.domain_name,
            similarity_threshold=similarity_threshold,
            weak_signal_threshold=weak_signal_threshold
        )
        
        merged_period_characterizations = list(merging_result.merged_segments)
        
        print(f"Segment merging completed:")
        print(f"Original segments: {len(period_characterizations)}")
        print(f"Merged segments: {len(merged_period_characterizations)}")
    else:
        if not enable_segment_merging:
            print(f"Segment merging disabled")
        else:
            print(f"Segment merging skipped - insufficient segments ({len(period_characterizations)})")
    
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
    domain_data: Optional[DomainData] = None
) -> Tuple[Optional[Dict], Optional[ChangeDetectionResult], Optional[List]]:
    """
    Run change point detection and segmentation for a domain.
    
    Args:
        domain_name: Name of the domain to process (required if domain_data not provided)
        granularity: Timeline granularity control (1-5, used if algorithm_config is None)
        algorithm_config: Optional comprehensive algorithm configuration
        domain_data: Optional pre-loaded domain data (avoids reloading)
        
    Returns:
        Tuple of (segmentation_results, change_detection_result, shift_signals)
    """
    # Use pre-loaded domain data or load it
    if domain_data is not None:
        print(f"\nCHANGE POINT DETECTION: {domain_data.domain_name}")
        domain_name = domain_data.domain_name
    else:
        if domain_name is None:
            raise ValueError("Either domain_name or domain_data must be provided")
        
        print(f"\nCHANGE POINT DETECTION: {domain_name}")
        
        # Load domain data
        result = process_domain_data(domain_name)
        if not result.success:
            print(f"Error loading {domain_name}: {result.error_message}")
            return None, None, None
        
        domain_data = result.domain_data
    
    # Create or use comprehensive algorithm configuration
    if algorithm_config is None:
        algorithm_config = AlgorithmConfig(granularity=int(granularity))
    else:
        granularity = algorithm_config.granularity

    # Run change detection
    print(f"CHANGE DETECTION")
    change_result, shift_signals = detect_changes(domain_data, algorithm_config=algorithm_config)
    
    # Extract change point years
    change_years = sorted([cp.year for cp in change_result.change_points])
    
    # Create segments using similarity segmentation
    print(f"\nSIMILARITY SEGMENTATION")
    if shift_signals:
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
        
        # Convert to expected format
        segments = [[start, end] for start, end in similarity_segments]
    else:
        print(f"No validated signals found - using single segment")
        segments = [[domain_data.year_range[0], domain_data.year_range[1]]]
    
    # Prepare results
    results = {
        'domain_name': domain_name,
        'granularity': granularity,
        'algorithm_config': {
            'direction_threshold': algorithm_config.direction_threshold,
            'citation_boost': algorithm_config.citation_boost,
            'validation_threshold': algorithm_config.validation_threshold,
            'citation_support_window': algorithm_config.citation_support_window,
            'similarity_min_segment_length': algorithm_config.similarity_min_segment_length,
            'similarity_max_segment_length': algorithm_config.similarity_max_segment_length,
            'keyword_min_frequency': algorithm_config.keyword_min_frequency,
            'min_significant_keywords': algorithm_config.min_significant_keywords
        },
        'time_range': list(domain_data.year_range),
        'change_points': change_years,
        'segments': segments,
        'statistical_significance': change_result.statistical_significance
    }
    
    return results, change_result, shift_signals


 