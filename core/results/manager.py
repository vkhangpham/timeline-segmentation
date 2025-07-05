"""
Result Manager for Timeline Analysis

This module handles saving and managing analysis results in various formats.
Separated from integration.py to follow single responsibility principle.

Key Features:
- Save shift signals for visualization
- Save period signals for visualization
- Save comprehensive analysis results to JSON
- Manage timestamped and latest result files
- Create result directory structure

Follows functional programming principles with pure functions for result management.
"""

import json
import os
import time
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Tuple

from ..data.models import (
    DomainData, TimelineAnalysisResult, PeriodCharacterization, ShiftSignal
)
from ..utils.logging import get_logger


def save_shift_signals(shift_signals: List[ShiftSignal], domain_name: str, domain_data: DomainData) -> str:
    """
    Save shift signals for visualization and analysis.
    
    Args:
        shift_signals: List of detected shift signals
        domain_name: Name of the research domain
        domain_data: Domain data for metadata
        
    Returns:
        Path to the saved shift signals file
    """
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
    
    # Save timestamped version in signals directory
    shift_file = f"{signals_dir}/{domain_name}_shift_signals.json"
    with open(shift_file, 'w') as f:
        json.dump(shift_data, f, indent=2)
    
    # Save latest version in results root
    shift_latest = f"results/{domain_name}_shift_signals.json"
    os.makedirs("results", exist_ok=True)
    with open(shift_latest, 'w') as f:
        json.dump(shift_data, f, indent=2)
    
    return shift_file


def save_period_signals(period_characterizations: List[PeriodCharacterization], 
                       segments: List[Tuple[int, int]], domain_name: str) -> str:
    """
    Save period signals for visualization and analysis.
    
    Args:
        period_characterizations: List of period characterizations
        segments: Timeline segments
        domain_name: Name of the research domain
        
    Returns:
        Path to the saved period signals file
    """
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
    
    # Save timestamped version in signals directory
    period_file = f"{signals_dir}/{domain_name}_period_signals.json"
    with open(period_file, 'w') as f:
        json.dump(period_data, f, indent=2)
    
    # Save latest version in results root
    period_latest = f"results/{domain_name}_period_signals.json"
    with open(period_latest, 'w') as f:
        json.dump(period_data, f, indent=2)
    
    return period_file


def save_analysis_results(timeline_result: TimelineAnalysisResult, segmentation_data: Dict[str, Any], 
                         domain_data: DomainData, output_path: str, verbose: bool = False) -> None:
    """
    Save comprehensive analysis results to JSON file.
    
    Args:
        timeline_result: Complete timeline analysis results
        segmentation_data: Segmentation metadata and configuration
        domain_data: Domain data for metadata
        output_path: Path where to save the results
        verbose: Enable verbose logging
    """
    logger = get_logger(__name__, verbose)
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
                'description': pc.topic_description,
                'network_stability': pc.network_stability,
                'representative_papers': [
                    {
                        'title': paper.get('title', '') if isinstance(paper, dict) else paper.title,
                        'year': paper.get('year', 0) if isinstance(paper, dict) else paper.pub_year,
                        'abstract': paper.get('abstract', '') if isinstance(paper, dict) else paper.description
                    } for paper in pc.representative_papers
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
    
    logger.info(f"Results saved to {output_path} and {latest_path}")


def save_all_results(timeline_result: TimelineAnalysisResult, segmentation_results: Dict[str, Any],
                    shift_signals: List[ShiftSignal], domain_data: DomainData, verbose: bool = False) -> Dict[str, str]:
    """
    Save all analysis results (convenience function).
    
    Args:
        timeline_result: Complete timeline analysis results
        segmentation_results: Segmentation results and metadata
        shift_signals: Detected shift signals
        domain_data: Domain data for metadata
        verbose: Enable verbose logging
        
    Returns:
        Dictionary mapping result type to file path
    """
    domain_name = timeline_result.domain_name
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    saved_files = {}
    
    # Save main timeline analysis
    timeline_file = f"results/timelines/{domain_name}_timeline_analysis_{timestamp}.json"
    save_analysis_results(timeline_result, segmentation_results, domain_data, timeline_file, verbose)
    saved_files['timeline_analysis'] = timeline_file
    
    # Save shift signals
    shift_file = save_shift_signals(shift_signals, domain_name, domain_data)
    saved_files['shift_signals'] = shift_file
    
    # Save period signals
    period_characterizations = list(timeline_result.period_characterizations)
    period_file = save_period_signals(period_characterizations, segmentation_results['segments'], domain_name)
    saved_files['period_signals'] = period_file
    
    return saved_files


def ensure_results_directory_structure() -> None:
    """
    Ensure all required result directories exist.
    """
    directories = [
        "results",
        "results/signals", 
        "results/timelines",
        "results/optimization",
        "results/visualizations"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)


# Export functions
__all__ = [
    'save_shift_signals',
    'save_period_signals', 
    'save_analysis_results',
    'save_all_results',
    'ensure_results_directory_structure'
] 