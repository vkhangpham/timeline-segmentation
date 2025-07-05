"""
Functional Validation Framework for Timeline Segmentation Algorithm

This module provides pure functional validation against available reference data while 
acknowledging the inherently subjective nature of paradigm shift evaluation.

Following Phase 13 principle: Emphasize explainable decisions and algorithm transparency.
Following functional programming guidelines: Pure functions, immutable data, no side effects.

Updated for new reference data structure:
- Manual references: data/references/{domain}_manual.json (detailed historical analysis)
- Gemini references: data/references/{domain}_gemini.json (simplified periods)

Note: There is no "ground truth" - these are two different types of reference annotations.
Manual references are used for validation, while both types are used for baseline comparison.
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Tuple, NamedTuple
from collections import Counter, defaultdict

# Import algorithm components for validation
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.data.models import ShiftSignal


class ReferenceData(NamedTuple):
    """Immutable reference data structure."""
    paradigm_shifts: Tuple[int, ...]
    historical_periods: Tuple[Dict[str, Any], ...]
    temporal_coverage: Dict[str, Any]
    period_characteristics: Dict[str, Any]


class ValidationMetrics(NamedTuple):
    """Immutable validation metrics structure."""
    precision_2yr: float
    recall_2yr: float
    f1_score_2yr: float
    precision_5yr: float
    recall_5yr: float
    f1_score_5yr: float
    paradigm_count_detected: int
    paradigm_count_reference: int
    count_accuracy: float


# ============================================================================
# PURE FUNCTIONS FOR DATA TRANSFORMATION
# ============================================================================

def extract_paradigm_shifts_from_manual(periods_data: Tuple[Dict[str, Any], ...]) -> Tuple[int, ...]:
    """
    Extract paradigm shift years from manual reference periods.
    
    Args:
        periods_data: Immutable tuple of period dictionaries
        
    Returns:
        Sorted tuple of paradigm shift years
    """
    if not periods_data:
        return tuple()
    
    paradigm_shifts = []
    
    for period in periods_data:
        # Manual format uses "years" field with ranges like "1900-1939"
        years_str = period.get('years', '')
        if years_str and '-' in years_str:
            try:
                start_year, end_year = years_str.split('-')
                end_year = int(end_year.strip())
                # Paradigm shift occurs at the end of each period
                paradigm_shifts.append(end_year)
            except ValueError:
                continue
    
    # Return immutable sorted tuple
    return tuple(sorted(set(paradigm_shifts)))


def extract_paradigm_shifts_from_gemini(periods_data: Tuple[Dict[str, Any], ...]) -> Tuple[int, ...]:
    """
    Extract paradigm shift years from gemini reference periods.
    
    Args:
        periods_data: Immutable tuple of period dictionaries
        
    Returns:
        Sorted tuple of paradigm shift years
    """
    if not periods_data:
        return tuple()
    
    # Sort periods by start year
    sorted_periods = tuple(sorted(periods_data, key=lambda p: p.get('start_year', 0)))
    
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
    
    return tuple(paradigm_shifts)


def analyze_temporal_coverage(periods_data: Tuple[Dict[str, Any], ...], reference_type: str) -> Dict[str, Any]:
    """
    Analyze temporal coverage of reference data.
    
    Args:
        periods_data: Immutable tuple of period dictionaries
        reference_type: "manual" or "gemini"
        
    Returns:
        Dictionary with temporal coverage information
    """
    if not periods_data:
        return {'start_year': None, 'end_year': None, 'total_years': 0}
    
    years = []
    
    if reference_type == "manual":
        for period in periods_data:
            years_str = period.get('years', '')
            if years_str and '-' in years_str:
                try:
                    start_year, end_year = years_str.split('-')
                    years.extend([int(start_year.strip()), int(end_year.strip())])
                except ValueError:
                    continue
    else:  # gemini
        for period in periods_data:
            if period.get('start_year'):
                years.append(period['start_year'])
            if period.get('end_year'):
                years.append(period['end_year'])
    
    if not years:
        return {'start_year': None, 'end_year': None, 'total_years': 0}
    
    return {
        'start_year': min(years),
        'end_year': max(years),
        'total_years': max(years) - min(years)
    }


def analyze_period_characteristics(periods_data: Tuple[Dict[str, Any], ...], reference_type: str) -> Dict[str, Any]:
    """
    Analyze characteristics of historical periods.
    
    Args:
        periods_data: Immutable tuple of period dictionaries
        reference_type: "manual" or "gemini"
        
    Returns:
        Dictionary with period characteristics
    """
    if not periods_data:
        return {}
    
    durations = []
    period_names = []
    
    if reference_type == "manual":
        for period in periods_data:
            name = period.get('period_name', '')
            period_names.append(name)
            
            # Calculate duration from years field
            years_str = period.get('years', '')
            if years_str and '-' in years_str:
                try:
                    start_year, end_year = years_str.split('-')
                    duration = int(end_year.strip()) - int(start_year.strip())
                    durations.append(duration)
                except ValueError:
                    continue
    else:  # gemini
        for period in periods_data:
            duration = period.get('duration_years')
            if duration:
                durations.append(duration)
            
            name = period.get('period_name', '')
            period_names.append(name)
    
    if not durations:
        return {'period_names': tuple(period_names)}
    
    avg_length = np.mean(durations)
    
    # Classify domain stability
    if avg_length > 100:
        stability = 'very_stable'
    elif avg_length > 30:
        stability = 'stable'
    elif avg_length > 15:
        stability = 'dynamic'
    else:
        stability = 'very_dynamic'
    
    return {
        'average_period_length': avg_length,
        'period_length_std': np.std(durations),
        'shortest_period': min(durations),
        'longest_period': max(durations),
        'period_names': tuple(period_names),
        'stability': stability
    }


def calculate_tolerance_metrics(detected_years: Tuple[int, ...], 
                               reference_years: Tuple[int, ...],
                               tolerance: int) -> Dict[str, float]:
    """
    Calculate precision, recall, F1 with temporal tolerance against reference data.
    
    Args:
        detected_years: Tuple of detected paradigm shift years
        reference_years: Tuple of reference paradigm shift years
        tolerance: Temporal tolerance in years
        
    Returns:
        Dictionary with precision, recall, and F1 metrics
    """
    if not detected_years or not reference_years:
        return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
    
    # True positives: detected shifts within tolerance of reference
    true_positives = sum(
        1 for detected_year in detected_years
        if any(abs(detected_year - ref_year) <= tolerance for ref_year in reference_years)
    )
    
    # Calculate precision
    precision = true_positives / len(detected_years)
    
    # Recall: reference shifts matched by detections
    matched_ref = sum(
        1 for ref_year in reference_years
        if any(abs(ref_year - detected_year) <= tolerance for detected_year in detected_years)
    )
    
    recall = matched_ref / len(reference_years)
    
    # F1 score
    f1 = 2 * precision * recall / max(precision + recall, 1e-10)
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def evaluate_shifts_against_reference(detected_shifts: Tuple[ShiftSignal, ...], 
                                     reference_shifts: Tuple[int, ...],
                                     tolerance_years: int = 2) -> ValidationMetrics:
    """
    Evaluate detected paradigm shifts against reference data with temporal tolerance.
    
    Args:
        detected_shifts: Tuple of detected shift signals
        reference_shifts: Tuple of reference paradigm shift years
        tolerance_years: Temporal tolerance window (±years)
        
    Returns:
        ValidationMetrics with evaluation results
    """
    if not reference_shifts:
        return ValidationMetrics(
            precision_2yr=0.0,
            recall_2yr=0.0, 
            f1_score_2yr=0.0,
            precision_5yr=0.0,
            recall_5yr=0.0,
            f1_score_5yr=0.0,
            paradigm_count_detected=len(detected_shifts),
            paradigm_count_reference=0,
            count_accuracy=0.0
        )
    
    detected_years = tuple(s.year for s in detected_shifts)
    
    # Calculate metrics for 2-year and 5-year tolerance
    metrics_2yr = calculate_tolerance_metrics(detected_years, reference_shifts, 2)
    metrics_5yr = calculate_tolerance_metrics(detected_years, reference_shifts, 5)
    
    # Count accuracy
    count_accuracy = max(0.0, 1.0 - abs(len(detected_years) - len(reference_shifts)) / max(len(reference_shifts), 1))
    
    return ValidationMetrics(
        precision_2yr=metrics_2yr['precision'],
        recall_2yr=metrics_2yr['recall'],
        f1_score_2yr=metrics_2yr['f1'],
        precision_5yr=metrics_5yr['precision'],
        recall_5yr=metrics_5yr['recall'],
        f1_score_5yr=metrics_5yr['f1'],
        paradigm_count_detected=len(detected_shifts),
        paradigm_count_reference=len(reference_shifts),
        count_accuracy=count_accuracy
    )


def process_reference_data(ref_data: Dict[str, Any], reference_type: str) -> ReferenceData:
    """
    Process raw reference data into immutable ReferenceData structure.
    
    Args:
        ref_data: Raw reference data from JSON file
        reference_type: "manual" or "gemini"
        
    Returns:
        Immutable ReferenceData structure
    """
    periods_tuple = tuple(ref_data.get('historical_periods', []))
    
    # Extract paradigm shifts
    if reference_type == "manual":
        paradigm_shifts = extract_paradigm_shifts_from_manual(periods_tuple)
    else:
        paradigm_shifts = extract_paradigm_shifts_from_gemini(periods_tuple)
    
    # Analyze temporal coverage and characteristics
    temporal_coverage = analyze_temporal_coverage(periods_tuple, reference_type)
    period_characteristics = analyze_period_characteristics(periods_tuple, reference_type)
    
    return ReferenceData(
        paradigm_shifts=paradigm_shifts,
        historical_periods=periods_tuple,
        temporal_coverage=temporal_coverage,
        period_characteristics=period_characteristics
    )


# ============================================================================
# I/O FUNCTIONS (SEPARATED FROM BUSINESS LOGIC)
# ============================================================================

def load_reference_data_from_files(reference_type: str = "manual") -> Dict[str, ReferenceData]:
    """
    Load all available reference data from data/references.
    
    FAIL-FAST: No fallbacks or error catching - any error will terminate execution.
    
    Args:
        reference_type: "manual" or "gemini" reference data to load
    
    Returns:
        Dictionary mapping domain names to immutable ReferenceData
    """
    # Validate reference type
    if reference_type not in ("manual", "gemini"):
        raise ValueError(f"Invalid reference type: {reference_type}. Must be 'manual' or 'gemini'.")
    
    references_dir = Path("data/references")
    
    if not references_dir.exists():
        raise FileNotFoundError(f"References directory does not exist: {references_dir}")
    
    reference_data = {}
    
    # Determine file pattern based on reference type
    file_pattern = f"*_{reference_type}.json"
    
    # Get all matching files
    matching_files = list(references_dir.glob(file_pattern))
    
    # FAIL-FAST: Ensure we have reference files for the requested type
    if not matching_files:
        raise FileNotFoundError(f"No {reference_type} reference files found in {references_dir}. Pattern: {file_pattern}")
    
    for ref_file in matching_files:
        domain_name = ref_file.stem.replace(f"_{reference_type}", "")
        
        # FAIL-FAST: No error handling - let exceptions propagate
        with open(ref_file, 'r') as f:
            ref_data = json.load(f)
            
        # Process into immutable structure - let any errors fail the execution
        reference_data[domain_name] = process_reference_data(ref_data, reference_type)
    
    return reference_data


def get_reference_summary(manual_refs: Dict[str, ReferenceData], 
                         gemini_refs: Dict[str, ReferenceData]) -> Dict[str, Any]:
    """
    Generate summary of available reference data.
    
    Args:
        manual_refs: Dictionary of manual reference data
        gemini_refs: Dictionary of gemini reference data
        
    Returns:
        Summary dictionary
    """
    summary = {
        'manual_references': {},
        'gemini_references': {},
        'common_domains': tuple()
    }
    
    for domain, data in manual_refs.items():
        summary['manual_references'][domain] = {
            'paradigm_shifts': len(data.paradigm_shifts),
            'temporal_coverage': data.temporal_coverage,
            'avg_period_length': data.period_characteristics.get('average_period_length', 0)
        }
    
    for domain, data in gemini_refs.items():
        summary['gemini_references'][domain] = {
            'paradigm_shifts': len(data.paradigm_shifts),
            'temporal_coverage': data.temporal_coverage,
            'avg_period_length': data.period_characteristics.get('average_period_length', 0)
        }
    
    # Find common domains
    common_domains = tuple(sorted(set(manual_refs.keys()) & set(gemini_refs.keys())))
    summary['common_domains'] = common_domains
    
    return summary
