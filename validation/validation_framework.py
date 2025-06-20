"""
Transparent Validation Framework for Timeline Segmentation Algorithm

This module provides systematic validation against available ground truth data while 
acknowledging the inherently subjective nature of paradigm shift evaluation.

Following Phase 13 principle: Emphasize explainable decisions and algorithm transparency.
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from collections import Counter, defaultdict
from dataclasses import dataclass

# Import algorithm components for validation
from core.data_models import ShiftSignal
from core.algorithm_config import ComprehensiveAlgorithmConfig


@dataclass
class ValidationResult:
    """Comprehensive validation result emphasizing transparency over simple accuracy."""
    domain: str
    algorithm_config: Dict[str, Any]
    temporal_alignment: Dict[str, Any]
    decision_transparency: Dict[str, Any]
    explainability_assessment: Dict[str, Any]
    comparative_analysis: Dict[str, Any]
    validation_insights: List[str]


class TransparentValidationFramework:
    """
    Validation framework emphasizing algorithm transparency and decision explainability.
    
    Acknowledges that paradigm shift evaluation is inherently subjective while providing
    systematic assessment against available reference data.
    """
    
    def __init__(self):
        """Initialize validation framework."""
        self.ground_truth_data = {}
        self.validation_results = {}
        self._load_ground_truth_data()
    
    def _load_ground_truth_data(self) -> None:
        """Load and analyze all available ground truth files."""
        validation_dir = Path(__file__).parent
        
        for gt_file in validation_dir.glob('*_groundtruth.json'):
            domain_name = gt_file.stem.replace('_groundtruth', '')
            
            try:
                with open(gt_file, 'r') as f:
                    ground_truth = json.load(f)
                
                # Extract paradigm shift years from period boundaries
                paradigm_shifts = self._extract_paradigm_shifts(ground_truth)
                
                self.ground_truth_data[domain_name] = {
                    'domain_info': ground_truth.get('domain', domain_name),
                    'historical_periods': ground_truth.get('historical_periods', []),
                    'paradigm_shifts': paradigm_shifts,
                    'temporal_coverage': self._analyze_temporal_coverage(ground_truth),
                    'period_characteristics': self._analyze_period_characteristics(ground_truth)
                }
                
                print(f"✅ Loaded ground truth for {domain_name}: {len(paradigm_shifts)} paradigm shifts")
                
            except Exception as e:
                print(f"⚠️ Failed to load ground truth for {domain_name}: {e}")
    
    def _extract_paradigm_shifts(self, ground_truth: Dict[str, Any]) -> List[int]:
        """Extract paradigm shift years from historical period boundaries."""
        periods = ground_truth.get('historical_periods', [])
        if not periods:
            return []
        
        # Sort periods by start year
        sorted_periods = sorted(periods, key=lambda p: p.get('start_year', 0))
        
        paradigm_shifts = []
        
        # Paradigm shifts occur at period transitions
        for i in range(1, len(sorted_periods)):
            prev_period = sorted_periods[i-1]
            curr_period = sorted_periods[i]
            
            prev_end = prev_period.get('end_year')
            curr_start = curr_period.get('start_year')
            
            if prev_end and curr_start:
                # Use the transition year (end of previous period)
                shift_year = prev_end
                paradigm_shifts.append(shift_year)
        
        return paradigm_shifts
    
    def _analyze_temporal_coverage(self, ground_truth: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze temporal coverage of ground truth data."""
        periods = ground_truth.get('historical_periods', [])
        
        if not periods:
            return {'min_year': None, 'max_year': None, 'total_years': 0}
        
        years = []
        for period in periods:
            if period.get('start_year'):
                years.append(period['start_year'])
            if period.get('end_year'):
                years.append(period['end_year'])
        
        return {
            'min_year': min(years) if years else None,
            'max_year': max(years) if years else None,
            'total_years': max(years) - min(years) if years else 0,
            'period_count': len(periods)
        }
    
    def _analyze_period_characteristics(self, ground_truth: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze characteristics of historical periods."""
        periods = ground_truth.get('historical_periods', [])
        
        durations = []
        period_names = []
        
        for period in periods:
            duration = period.get('duration_years')
            if duration:
                durations.append(duration)
            
            name = period.get('period_name', '')
            period_names.append(name)
        
        return {
            'average_period_length': np.mean(durations) if durations else 0,
            'period_length_std': np.std(durations) if durations else 0,
            'shortest_period': min(durations) if durations else 0,
            'longest_period': max(durations) if durations else 0,
            'period_names': period_names
        }
    
    def get_available_domains(self) -> List[str]:
        """Get list of domains with available ground truth data."""
        return list(self.ground_truth_data.keys())
    
    def get_ground_truth_summary(self) -> Dict[str, Any]:
        """Get summary of available ground truth data."""
        summary = {}
        
        for domain, data in self.ground_truth_data.items():
            summary[domain] = {
                'paradigm_shifts': len(data['paradigm_shifts']),
                'temporal_coverage': data['temporal_coverage'],
                'avg_period_length': data['period_characteristics']['average_period_length']
            }
        
        return summary

def load_ground_truth_data() -> Dict[str, Any]:
    """
    Load all available ground truth data.
    
    Returns:
        Dictionary mapping domain names to ground truth data
    """
    validation_dir = Path("validation")
    ground_truth_data = {}
    
    for gt_file in validation_dir.glob("*_groundtruth.json"):
        domain_name = gt_file.stem.replace("_groundtruth", "")
        
        try:
            with open(gt_file, 'r') as f:
                gt_data = json.load(f)
                
            # Extract paradigm shifts from historical periods
            paradigm_shifts = []
            if 'historical_periods' in gt_data:
                for period in gt_data['historical_periods']:
                    if 'end_year' in period and period['end_year'] is not None:
                        paradigm_shifts.append(period['end_year'])
            
            # Create standardized ground truth structure
            ground_truth_data[domain_name] = {
                'paradigm_shifts': paradigm_shifts,
                'historical_periods': gt_data.get('historical_periods', []),
                'temporal_coverage': {
                    'start_year': min([p.get('start_year', 9999) for p in gt_data.get('historical_periods', [])]),
                    'end_year': max([p.get('end_year', 0) for p in gt_data.get('historical_periods', []) if p.get('end_year')])
                },
                'period_characteristics': _analyze_period_characteristics(gt_data.get('historical_periods', []))
            }
            
        except Exception as e:
            print(f"Warning: Could not load ground truth for {domain_name}: {e}")
            ground_truth_data[domain_name] = {
                'paradigm_shifts': [],
                'error': str(e)
            }
    
    return ground_truth_data


def evaluate_against_ground_truth(detected_shifts: List[ShiftSignal], 
                                ground_truth_shifts: List[int],
                                tolerance_years: int = 2) -> Dict[str, float]:
    """
    Evaluate detected paradigm shifts against ground truth with temporal tolerance.
    
    Args:
        detected_shifts: List of detected shift signals
        ground_truth_shifts: List of ground truth paradigm shift years
        tolerance_years: Temporal tolerance window (±years)
        
    Returns:
        Dictionary with evaluation metrics
    """
    if not ground_truth_shifts:
        return {
            'precision_2yr': 0.0,
            'recall_2yr': 0.0, 
            'f1_score_2yr': 0.0,
            'precision_5yr': 0.0,
            'recall_5yr': 0.0,
            'f1_score_5yr': 0.0,
            'paradigm_count_detected': len(detected_shifts),
            'paradigm_count_ground_truth': 0,
            'count_accuracy': 0.0
        }
    
    detected_years = [s.year for s in detected_shifts]
    
    # Calculate metrics for 2-year and 5-year tolerance
    metrics_2yr = _calculate_tolerance_metrics(detected_years, ground_truth_shifts, 2)
    metrics_5yr = _calculate_tolerance_metrics(detected_years, ground_truth_shifts, 5)
    
    # Count accuracy
    count_accuracy = 1.0 - abs(len(detected_years) - len(ground_truth_shifts)) / max(len(ground_truth_shifts), 1)
    
    return {
        'precision_2yr': metrics_2yr['precision'],
        'recall_2yr': metrics_2yr['recall'],
        'f1_score_2yr': metrics_2yr['f1'],
        'precision_5yr': metrics_5yr['precision'],
        'recall_5yr': metrics_5yr['recall'],
        'f1_score_5yr': metrics_5yr['f1'],
        'paradigm_count_detected': len(detected_shifts),
        'paradigm_count_ground_truth': len(ground_truth_shifts),
        'count_accuracy': max(0.0, count_accuracy)
    }


def _calculate_tolerance_metrics(detected_years: List[int], 
                               ground_truth_years: List[int],
                               tolerance: int) -> Dict[str, float]:
    """Calculate precision, recall, F1 with temporal tolerance."""
    
    if not detected_years:
        return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
    
    if not ground_truth_years:
        return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
    
    # True positives: detected shifts within tolerance of ground truth
    true_positives = 0
    for detected_year in detected_years:
        if any(abs(detected_year - gt_year) <= tolerance for gt_year in ground_truth_years):
            true_positives += 1
    
    # Calculate precision, recall, F1
    precision = true_positives / len(detected_years)
    
    # Recall: ground truth shifts matched by detections
    matched_gt = 0
    for gt_year in ground_truth_years:
        if any(abs(gt_year - detected_year) <= tolerance for detected_year in detected_years):
            matched_gt += 1
    
    recall = matched_gt / len(ground_truth_years)
    
    # F1 score
    f1 = 2 * precision * recall / max(precision + recall, 1e-10)
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def _analyze_period_characteristics(historical_periods: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze characteristics of historical periods."""
    
    if not historical_periods:
        return {}
    
    # Calculate period lengths
    period_lengths = []
    for period in historical_periods:
        start = period.get('start_year')
        end = period.get('end_year')
        if start and end:
            period_lengths.append(end - start)
    
    characteristics = {}
    if period_lengths:
        characteristics = {
            'avg_period_length': np.mean(period_lengths),
            'std_period_length': np.std(period_lengths),
            'min_period_length': min(period_lengths),
            'max_period_length': max(period_lengths),
            'num_periods': len(period_lengths)
        }
        
        # Classify domain stability
        avg_length = characteristics['avg_period_length']
        if avg_length > 100:
            characteristics['stability'] = 'very_stable'
        elif avg_length > 30:
            characteristics['stability'] = 'stable'
        elif avg_length > 15:
            characteristics['stability'] = 'dynamic'
        else:
            characteristics['stability'] = 'very_dynamic'
    
    return characteristics
