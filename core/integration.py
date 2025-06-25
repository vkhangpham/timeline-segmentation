"""
Timeline Analysis Integration Module

This module orchestrates the complete timeline analysis pipeline by integrating:
1. Change point detection and shift signal analysis
2. Period characterization through network stability analysis  
3. Segment merging for post-processing similar segments

Provides clean integration and orchestration of the core analysis algorithms.
"""

from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import json
from datetime import datetime

from .data_models import (
    DomainData, ChangeDetectionResult, TimelineAnalysisResult
)
from .change_detection import detect_changes
from .data_processing import process_domain_data
from .segment_modeling import model_segments
from .segment_merging import merge_similar_segments
from .shift_signal_detection import detect_shift_signals
from .algorithm_config import ComprehensiveAlgorithmConfig
from .similarity_segmentation import create_similarity_based_segments
from .keyword_utils import extract_year_keywords


def timeline_analysis(
    domain_data: DomainData,
    segments: List[Tuple[int, int]],
    change_detection_result: Optional[ChangeDetectionResult] = None,
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
        enable_segment_merging: Whether to perform segment merging post-processing
        similarity_threshold: Threshold for semantic similarity in merging (0.0-1.0)
        weak_signal_threshold: Threshold for weak shift signals in merging (0.0-1.0)
        granularity: Timeline granularity control (1-5)
        
    Returns:
        Timeline analysis results with period characterizations and optional merging
    """
    print(f"\nTIMELINE ANALYSIS: {domain_data.domain_name}")
    
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
        
        # Create comprehensive algorithm config for shift signal detection
        algorithm_config = ComprehensiveAlgorithmConfig(granularity=granularity)
        
        # Get shift signals for boundary analysis
        shift_signals = detect_shift_signals(domain_data, domain_data.domain_name, algorithm_config)
        
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
    domain_name: str, 
    granularity: int = 3, 
    algorithm_config: Optional[ComprehensiveAlgorithmConfig] = None
) -> Tuple[Optional[Dict], Optional[ChangeDetectionResult]]:
    """
    Run change point detection and segmentation for a domain.
    
    Args:
        domain_name: Name of the domain to process
        granularity: Timeline granularity control (1-5, used if algorithm_config is None)
        algorithm_config: Optional comprehensive algorithm configuration
        
    Returns:
        Tuple of (segmentation_results, change_detection_result)
    """
    print(f"\nCHANGE POINT DETECTION: {domain_name}")
    
    # Load domain data first
    result = process_domain_data(domain_name)
    if not result.success:
        print(f"Error loading {domain_name}: {result.error_message}")
        return None, None
    
    domain_data = result.domain_data
    
    # Create or use comprehensive algorithm configuration
    if algorithm_config is None:
        algorithm_config = ComprehensiveAlgorithmConfig(granularity=int(granularity))
    else:
        granularity = algorithm_config.granularity

    # Run change detection
    print(f"CHANGE DETECTION")
    change_result, shift_signals = detect_changes(domain_data, algorithm_config=algorithm_config)
    
    # Extract change point years
    change_years = sorted([cp.year for cp in change_result.change_points])
    
    # Create segments using similarity segmentation
    print(f"SIMILARITY SEGMENTATION")
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
        
        print(f"Created {len(segments)} similarity-based segments")
        print(f"Signal years: {[s.year for s in shift_signals]}")
        print(f"Segment ranges: {[(s[0], s[1]) for s in segments]}")
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
    
    print(f"Detected {len(change_years)} change points")
    print(f"Created {len(segments)} timeline segments")
    
    return results, change_result


def run_timeline_analysis(domain_name: str, segmentation_results: Dict, change_detection_result=None) -> Optional[str]:
    """
    Run timeline analysis with period characterization and segment merging.
    
    Args:
        domain_name: Name of the domain to process
        segmentation_results: Segmentation results dict from change detection
        change_detection_result: Pre-computed change detection results
        
    Returns:
        Path to results file if successful, None otherwise
    """
    print(f"\nTIMELINE ANALYSIS: {domain_name}")
    
    # Load domain data
    result = process_domain_data(domain_name)
    if not result.success:
        print(f"Error loading {domain_name}: {result.error_message}")
        return None
    
    domain_data = result.domain_data
    
    # Use segmentation results from memory
    segments = [(start, end) for start, end in segmentation_results['segments']]
    print(f"Using {len(segments)} timeline segments")
      
    # Run timeline analysis with segment merging
    timeline_result = timeline_analysis(
        domain_data=domain_data, 
        segments=segments, 
        change_detection_result=change_detection_result,
        enable_segment_merging=True,
        similarity_threshold=0.75,
        weak_signal_threshold=0.4,
        granularity=segmentation_results['granularity']
    )
    
    # Save comprehensive results
    comprehensive_output_file = f"results/{domain_name}_comprehensive_analysis.json"
    save_comprehensive_results(timeline_result, segmentation_results, domain_data, comprehensive_output_file)

    return comprehensive_output_file


def save_comprehensive_results(
    timeline_result: TimelineAnalysisResult,
    segmentation_data: Dict[str, Any],
    domain_data: DomainData,
    output_path: str
) -> None:
    """
    Save comprehensive analysis results to JSON file.
    
    Args:
        timeline_result: Results from timeline analysis
        segmentation_data: Basic segmentation results
        domain_data: Domain data for metadata
        output_path: Path to save the comprehensive results
    """
    # Create streamlined output structure with essential data only
    comprehensive_results = {
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
                    } for paper in pc.representative_papers[:3]  # Limit to top 3
                ],
                'confidence': pc.confidence
            }
            for pc in timeline_result.merged_period_characterizations
        ],
        'unified_confidence': timeline_result.unified_confidence
    }
    
    # Add algorithm config for reproducibility
    if 'algorithm_config' in segmentation_data:
        comprehensive_results['algorithm_config'] = segmentation_data['algorithm_config']
    
    # Save to file
    with open(output_path, 'w') as f:
        json.dump(comprehensive_results, f, indent=2)
    
    print(f"Results saved to {output_path}")
    print(f"Periods: {len(timeline_result.merged_period_characterizations)}")


# Legacy compatibility
timeline_analysis_legacy = timeline_analysis
save_results_legacy = save_comprehensive_results 