"""
Timeline Analysis Integration Module

This module orchestrates the complete timeline analysis pipeline by integrating:
1. Shift Signal Detection (paradigm transition analysis)
2. Period Signal Detection (period characterization through network stability)
3. Segment Merging (post-processing to merge semantically similar segments)

Purpose: Clean integration and orchestration only - no embedded algorithms.
"""

from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import json
from datetime import datetime
from dataclasses import dataclass

from .data_models import (
    DomainData, ChangeDetectionResult, TimelineAnalysisResult
)
from .change_detection import detect_changes, create_segments_with_confidence
from .data_processing import process_domain_data
from .segment_modeling import model_segments
from .segment_merging import merge_similar_segments
from .shift_signal_detection import detect_shift_signals


@dataclass
class SensitivityConfig:
    """
    Centralized sensitivity configuration with directly adjustable parameters.
    
    All parameters are directly configurable for maximum flexibility.
    Granularity level provides convenient presets but parameters can be overridden.
    
    Core Parameters:
    - detection_threshold: Minimum score needed to detect a direction signal (lower = more sensitive)
    - validation_threshold: Minimum score needed to validate signals (consistent for all signals)
    - citation_boost: Confidence boost given to signals with citation support
    - clustering_window: Temporal window for clustering direction signals (years)
    - breakthrough_bonus: DEPRECATED - breakthrough paper validation removed (too permissive)
    """
    
    # Directly adjustable parameters with defaults
    detection_threshold: float = 0.4      # Direction signal detection threshold
    validation_threshold: float = 0.7     # Consistent validation threshold for all signals
    citation_boost: float = 0.3           # Citation support confidence boost
    clustering_window: int = 3            # Temporal clustering window (years)
    breakthrough_bonus: float = 0.4       # DEPRECATED - breakthrough validation removed
    granularity: int = 3                  # Granularity level (for preset interface)
    
    def __init__(self, 
                 granularity: int = 3,
                 detection_threshold: Optional[float] = None,
                 validation_threshold: Optional[float] = None,
                 citation_boost: Optional[float] = None,
                 clustering_window: Optional[int] = None,
                 breakthrough_bonus: Optional[float] = None):
        """
        Initialize sensitivity configuration with optional parameter overrides.
        
        Args:
            granularity: 1-5 scale for convenient presets (1=coarse, 5=fine)
            detection_threshold: Override detection threshold (None = use granularity preset)
            validation_threshold: Override validation threshold (None = use granularity preset)
            citation_boost: Override citation boost (None = use granularity preset)
            clustering_window: Override clustering window (None = use granularity preset)
            breakthrough_bonus: Override breakthrough bonus (None = use granularity preset)
        """
        if not 1 <= granularity <= 5:
            raise ValueError("Granularity must be between 1 and 5")
            
        self.granularity = granularity
        
        # Apply granularity presets first
        granularity_presets = self._get_granularity_presets(granularity)
        
        # Set parameters using presets or overrides
        self.detection_threshold = detection_threshold if detection_threshold is not None else granularity_presets['detection_threshold']
        self.validation_threshold = validation_threshold if validation_threshold is not None else granularity_presets['validation_threshold']
        self.citation_boost = citation_boost if citation_boost is not None else granularity_presets['citation_boost']
        self.clustering_window = clustering_window if clustering_window is not None else granularity_presets['clustering_window']
        self.breakthrough_bonus = breakthrough_bonus if breakthrough_bonus is not None else granularity_presets['breakthrough_bonus']
    
    @staticmethod
    def _get_granularity_presets(granularity: int) -> Dict[str, Any]:
        """
        Get parameter presets for a given granularity level.
        
        PREDICTABLE RELATIONSHIP: Lower granularity number → More segments
        1 (ultra_coarse) ≥ 2 (coarse) ≥ 3 (balanced) ≥ 4 (fine) ≥ 5 (ultra_fine)
        """
        presets = {
            1: {  # Ultra-coarse: Very high thresholds, minimal sensitivity
                'detection_threshold': 0.6,
                'validation_threshold': 0.9,
                'citation_boost': 0.2,
                'clustering_window': 4,
                'breakthrough_bonus': 0.2
            },
            2: {  # Coarse: High thresholds, low sensitivity
                'detection_threshold': 0.5,
                'validation_threshold': 0.85,
                'citation_boost': 0.3,
                'clustering_window': 4,
                'breakthrough_bonus': 0.3
            },
            3: {  # Balanced: Moderate thresholds, balanced sensitivity
                'detection_threshold': 0.4,
                'validation_threshold': 0.8,
                'citation_boost': 0.3,
                'clustering_window': 3,
                'breakthrough_bonus': 0.3
            },
            4: {  # Fine: Low thresholds, high sensitivity
                'detection_threshold': 0.3,
                'validation_threshold': 0.75,
                'citation_boost': 0.3,
                'clustering_window': 2,
                'breakthrough_bonus': 0.3
            },
            5: {  # Ultra-fine: Very low thresholds, maximum sensitivity
                'detection_threshold': 0.2,
                'validation_threshold': 0.7,
                'citation_boost': 0.3,
                'clustering_window': 2,
                'breakthrough_bonus': 0.3
            }
        }
        return presets[granularity]
    
    @classmethod
    def create_custom(cls, 
                      detection_threshold: float,
                      validation_threshold: float,
                      citation_boost: float = 0.3,
                      clustering_window: int = 3,
                      breakthrough_bonus: float = 0.4) -> 'SensitivityConfig':
        """
        Create a fully custom configuration bypassing granularity presets.
        
        Args:
            detection_threshold: Direction signal detection threshold
            validation_threshold: Consistent validation threshold for all signals
            citation_boost: Citation support confidence boost
            clustering_window: Temporal clustering window (years)
            breakthrough_bonus: Breakthrough proximity confidence boost
            
        Returns:
            Custom SensitivityConfig instance
        """
        config = cls.__new__(cls)  # Create without calling __init__
        config.granularity = 0  # Mark as custom (outside 1-5 range)
        config.detection_threshold = detection_threshold
        config.validation_threshold = validation_threshold
        config.citation_boost = citation_boost
        config.clustering_window = clustering_window
        config.breakthrough_bonus = breakthrough_bonus
        return config


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
        granularity: Timeline granularity control (1-5) for sensitivity configuration
        
    Returns:
        Timeline analysis results with period characterizations and optional merging
    """
    print(f"\n🔬 TIMELINE ANALYSIS: {domain_data.domain_name}")
    print("=" * 50)
    
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
        print(f"\n🔄 SEGMENT MERGING POST-PROCESSING")
        print("=" * 50)
        
        # Create sensitivity config for shift signal detection
        sensitivity_config = SensitivityConfig(granularity=granularity)
        
        # Get shift signals for boundary analysis
        shift_signals, _, _ = detect_shift_signals(domain_data, domain_data.domain_name, sensitivity_config)
        
        # Perform intelligent segment merging
        merging_result = merge_similar_segments(
            period_characterizations=period_characterizations,
            shift_signals=shift_signals,
            domain_name=domain_data.domain_name,
            similarity_threshold=similarity_threshold,
            weak_signal_threshold=weak_signal_threshold
        )
        
        merged_period_characterizations = list(merging_result.merged_segments)
        
        print(f"✅ Segment merging completed:")
        print(f"    Original segments: {len(period_characterizations)}")
        print(f"    Merged segments: {len(merged_period_characterizations)}")
        print(f"    Merging decisions: {len(merging_result.merge_decisions)}")
        print(f"    Summary: {merging_result.merging_summary}")
    else:
        if not enable_segment_merging:
            print(f"    ℹ️ Segment merging disabled")
        else:
            print(f"    ℹ️ Segment merging skipped - insufficient segments ({len(period_characterizations)})")
    
    # Step 3: Calculate unified confidence using final merged segments
    if merged_period_characterizations:
        unified_confidence = np.mean([pc.confidence for pc in merged_period_characterizations])
    else:
        unified_confidence = 0.0
    
    # Step 4: Generate narrative evolution using final merged segments
    narrative_parts = []
    for pc in merged_period_characterizations:  # First 3 periods
        narrative_parts.append(f"{pc.period[0]}-{pc.period[1]}: {pc.topic_label}")
    
    narrative_evolution = "\n→ ".join(narrative_parts)
    
    return TimelineAnalysisResult(
        domain_name=domain_data.domain_name,
        period_characterizations=tuple(period_characterizations),
        merged_period_characterizations=tuple(merged_period_characterizations),
        merging_result=merging_result,
        unified_confidence=unified_confidence,
        narrative_evolution=narrative_evolution
    )


def run_change_detection(domain_name: str, granularity: int = 3, sensitivity_config: Optional[SensitivityConfig] = None) -> Tuple[Optional[Dict], Optional[ChangeDetectionResult]]:
    """
    Run change point detection and segmentation for a domain with flexible parameter control.
    
    Args:
        domain_name: Name of the domain to process
        granularity: Timeline granularity control - integer 1-5 (used if sensitivity_config is None):
                    1 = ultra_coarse (fewest segments)
                    2 = coarse
                    3 = balanced (default)  
                    4 = fine
                    5 = ultra_fine (most segments)
        sensitivity_config: Optional custom sensitivity configuration (overrides granularity)
        
    Returns:
        Tuple of (segmentation_results, change_detection_result)
    """
    print(f"\n🔍 CHANGE POINT DETECTION: {domain_name}")
    print("=" * 50)
    
    # Map granularity integer to descriptive names for logging
    granularity_names = {
        1: "ultra_coarse",
        2: "coarse", 
        3: "balanced",
        4: "fine",
        5: "ultra_fine"
    }
    
    granularity_name = granularity_names.get(granularity, "unknown")
    
    # Create or use sensitivity configuration
    if sensitivity_config is None:
        sensitivity_config = SensitivityConfig(granularity=int(granularity))
    
    # Update granularity for logging if custom config is used
    if hasattr(sensitivity_config, 'granularity') and sensitivity_config.granularity == 0:
        granularity_name = "custom"
    else:
        granularity_name = granularity_names.get(sensitivity_config.granularity, "unknown")
    
    print(f"🎛️  SENSITIVITY CONFIGURATION:")
    if sensitivity_config.granularity == 0:
        print(f"    Type: Custom configuration")
    else:
        print(f"    Type: Granularity preset (level={sensitivity_config.granularity})")
    print(f"    Detection threshold: {sensitivity_config.detection_threshold:.2f}")
    print(f"    Clustering window: {sensitivity_config.clustering_window} years")
    print(f"    Citation boost: {sensitivity_config.citation_boost:.2f}")
    print(f"    Validation threshold: {sensitivity_config.validation_threshold:.2f}")
    
    # Load domain data
    result = process_domain_data(domain_name)
    if not result.success:
        print(f"❌ Error loading {domain_name}: {result.error_message}")
        return None, None
    
    domain_data = result.domain_data
    print(f"📊 Loading {domain_name} data...")
    
    # Run change detection with sensitivity configuration
    print(f"🔍 Detecting change points...")
    change_result = detect_changes(domain_data, sensitivity_config=sensitivity_config)
    
    # Extract change point years
    change_years = sorted([cp.year for cp in change_result.change_points])
    
    # Convert change points to segments with improved logic (deduplication + merging)
    segments = create_segments_with_confidence(
        change_years, 
        domain_data.year_range, 
        statistical_significance=change_result.statistical_significance, 
        domain_name=domain_name
    )
    
    # Prepare results (return as dict instead of saving to file)
    results = {
        'domain_name': domain_name,
        'granularity': granularity,
        'sensitivity_config': {
            'detection_threshold': sensitivity_config.detection_threshold,
            'clustering_window': sensitivity_config.clustering_window,
            'citation_boost': sensitivity_config.citation_boost,
            'validation_threshold': sensitivity_config.validation_threshold
        },
        'time_range': list(domain_data.year_range),
        'change_points': change_years,
        'segments': segments,
        'statistical_significance': change_result.statistical_significance,
        'method_details': {
            'change_points_detected': len(change_result.change_points),
            'burst_periods_detected': len(change_result.burst_periods),
            'methods_used': ['enhanced_shift_signal_with_sensitivity']
        }
    }
    
    print(f"📈 Detected {len(change_years)} change points")
    print(f"📋 Created {len(segments)} timeline segments")
    
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
    print(f"\n🔬 TIMELINE ANALYSIS: {domain_name}")
    print("=" * 50)
    
    # Load domain data
    result = process_domain_data(domain_name)
    if not result.success:
        print(f"❌ Error loading {domain_name}: {result.error_message}")
        return None
    
    domain_data = result.domain_data
    
    # Use segmentation results from memory instead of file
    segments = [(start, end) for start, end in segmentation_results['segments']]
    print(f"📈 Using {len(segments)} timeline segments")
      
    # Run timeline analysis with segment merging
    timeline_result = timeline_analysis(
        domain_data=domain_data, 
        segments=segments, 
        change_detection_result=change_detection_result,
        enable_segment_merging=True,  # Enable merging by default
        similarity_threshold=0.75,    # High similarity threshold
        weak_signal_threshold=0.4,     # Moderate weak signal threshold
        granularity=segmentation_results['granularity']
    )
    
    # Save comprehensive results (only save this file)
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
    Save comprehensive analysis results to a single JSON file.
    
    Args:
        timeline_result: Results from timeline analysis
        segmentation_data: Basic segmentation results
        domain_data: Domain data for metadata
        output_path: Path to save the comprehensive results
    """
    # Create comprehensive output structure
    comprehensive_results = {
        'analysis_metadata': {
            'domain_name': timeline_result.domain_name,
            'analysis_date': datetime.now().isoformat(),
            'time_range': domain_data.year_range,
            'total_papers_analyzed': len(domain_data.papers),
            'methodology': {
                'shift_detection': 'Enhanced Shift Signal Detection',
                'period_characterization': 'Temporal Network Stability Analysis',
                'segment_merging': 'Semantic Similarity & Weak Signal Analysis',
                'change_points_detected': segmentation_data.get('method_details', {}).get('change_points_detected', 0),
                'statistical_significance': segmentation_data.get('statistical_significance', 0)
            }
        },
        'segmentation_results': {
            'change_points': segmentation_data.get('change_points', []),
            'segments': segmentation_data.get('segments', []),
            'statistical_significance': segmentation_data.get('statistical_significance', 0),
            'method_details': segmentation_data.get('method_details', {})
        },
        'timeline_analysis': {
            'original_period_characterizations': [
                {
                    'period': list(pc.period),
                    'topic_label': pc.topic_label,
                    'topic_description': pc.topic_description,
                    'network_stability': pc.network_stability,
                    'community_persistence': pc.community_persistence,
                    'flow_stability': pc.flow_stability,
                    'centrality_consensus': pc.centrality_consensus,
                    'representative_papers': [dict(paper) for paper in pc.representative_papers],
                    'network_metrics': pc.network_metrics,
                    'confidence': pc.confidence
                }
                for pc in timeline_result.period_characterizations
            ],
            'final_period_characterizations': [
                {
                    'period': list(pc.period),
                    'topic_label': pc.topic_label,
                    'topic_description': pc.topic_description,
                    'network_stability': pc.network_stability,
                    'community_persistence': pc.community_persistence,
                    'flow_stability': pc.flow_stability,
                    'centrality_consensus': pc.centrality_consensus,
                    'representative_papers': [dict(paper) for paper in pc.representative_papers],
                    'network_metrics': pc.network_metrics,
                    'confidence': pc.confidence
                }
                for pc in timeline_result.merged_period_characterizations
            ],
            'unified_confidence': timeline_result.unified_confidence,
            'narrative_evolution': timeline_result.narrative_evolution
        }
    }
    
    # Add segment merging results if available
    if timeline_result.merging_result:
        comprehensive_results['segment_merging'] = {
            'merging_performed': True,
            'original_segments': len(timeline_result.period_characterizations),
            'final_segments': len(timeline_result.merged_period_characterizations),
            'merge_decisions': [
                {
                    'segment1_index': md.segment1_index,
                    'segment2_index': md.segment2_index,
                    'semantic_similarity': md.semantic_similarity,
                    'shift_signal_strength': md.shift_signal_strength,
                    'merge_confidence': md.merge_confidence,
                    'merge_justification': md.merge_justification,
                    'merged_period': list(md.merged_period),
                    'merged_label': md.merged_label,
                    'merged_description': md.merged_description
                }
                for md in timeline_result.merging_result.merge_decisions
            ],
            'merging_summary': timeline_result.merging_result.merging_summary
        }
    else:
        comprehensive_results['segment_merging'] = {
            'merging_performed': False,
            'reason': 'Insufficient segments or merging disabled'
        }
    
    # Save to file
    with open(output_path, 'w') as f:
        json.dump(comprehensive_results, f, indent=2)
    
    print(f"💾 Comprehensive analysis results saved to {output_path}")
    print(f"📊 Includes {len(timeline_result.merged_period_characterizations)} final period characterizations")
    if timeline_result.merging_result:
        print(f"🔄 Segment merging: {len(timeline_result.period_characterizations)} → {len(timeline_result.merged_period_characterizations)} segments")


# Backward compatibility aliases
three_pillar_analysis = timeline_analysis
save_three_pillar_results = save_comprehensive_results
ThreePillarResult = TimelineAnalysisResult 