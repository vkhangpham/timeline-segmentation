"""Timeline evaluation module with objective function scoring and baseline comparisons.

This module provides comprehensive evaluation capabilities for timeline segmentation
including objective function scoring, baseline creation, and auto-metrics calculation.
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple, NamedTuple, Optional
from collections import defaultdict
import numpy as np

from ..data.data_models import AcademicPeriod, TimelineAnalysisResult
from ..data.data_processing import (
    load_domain_data,
    create_academic_periods_from_segments,
    filter_papers_by_year_range,
)
from ..optimization.objective_function import compute_objective_function
from ..utils.config import AlgorithmConfig
from ..utils.logging import get_logger


class EvaluationResult(NamedTuple):
    """Result of timeline evaluation with objective function score."""
    
    objective_score: float
    cohesion_score: float
    separation_score: float
    num_segments: int
    num_transitions: int
    boundary_years: List[int]
    methodology: str
    details: Dict[str, str]


class BaselineResult(NamedTuple):
    """Result of baseline evaluation."""
    
    baseline_name: str
    objective_score: float
    cohesion_score: float
    separation_score: float
    num_segments: int
    boundary_years: List[int]
    academic_periods: List[AcademicPeriod]


class AutoMetricResult(NamedTuple):
    """Result of auto-metric evaluation."""
    
    boundary_f1: float
    boundary_precision: float
    boundary_recall: float
    segment_f1: float
    segment_precision: float
    segment_recall: float
    tolerance: int
    details: Dict[str, str]


class ComprehensiveEvaluationResult(NamedTuple):
    """Complete evaluation result with all metrics."""
    
    domain_name: str
    algorithm_result: EvaluationResult
    baseline_results: List[BaselineResult]
    auto_metrics: AutoMetricResult
    ranking: Dict[str, float]  # baseline_name -> objective_score
    summary: str


def evaluate_timeline_result(
    timeline_result: TimelineAnalysisResult,
    algorithm_config: AlgorithmConfig,
    verbose: bool = False,
) -> EvaluationResult:
    """Evaluate a timeline result using the objective function.
    
    Args:
        timeline_result: TimelineAnalysisResult from pipeline
        algorithm_config: Algorithm configuration for objective function
        verbose: Enable verbose logging
        
    Returns:
        EvaluationResult with objective function score and details
    """
    logger = get_logger(__name__, verbose, timeline_result.domain_name)
    
    if verbose:
        logger.info(f"Evaluating timeline result for {timeline_result.domain_name}")
        logger.info(f"Timeline has {len(timeline_result.periods)} periods")
    
    # Extract AcademicPeriod objects from timeline result
    academic_periods = list(timeline_result.periods)
    
    # Compute objective function
    obj_result = compute_objective_function(
        academic_periods=academic_periods,
        algorithm_config=algorithm_config,
        verbose=verbose,
    )
    
    # Extract boundary years from periods
    boundary_years = []
    for i, period in enumerate(academic_periods):
        if i == 0:
            boundary_years.append(period.start_year)
        boundary_years.append(period.end_year)
    
    # Remove duplicates and sort
    boundary_years = sorted(set(boundary_years))
    
    details = {
        "cohesion_details": obj_result.cohesion_details,
        "separation_details": obj_result.separation_details,
        "periods": [f"{p.start_year}-{p.end_year}" for p in academic_periods],
    }
    
    if verbose:
        logger.info(f"Objective function score: {obj_result.final_score:.3f}")
        logger.info(f"Cohesion: {obj_result.cohesion_score:.3f}, Separation: {obj_result.separation_score:.3f}")
    
    return EvaluationResult(
        objective_score=obj_result.final_score,
        cohesion_score=obj_result.cohesion_score,
        separation_score=obj_result.separation_score,
        num_segments=obj_result.num_segments,
        num_transitions=obj_result.num_transitions,
        boundary_years=boundary_years,
        methodology=obj_result.methodology,
        details=details,
    )


def create_gemini_baseline(
    domain_name: str,
    algorithm_config: AlgorithmConfig,
    data_directory: str = "resources",
    verbose: bool = False,
) -> BaselineResult:
    """Create baseline using Gemini reference timeline.
    
    Args:
        domain_name: Domain name to evaluate
        algorithm_config: Algorithm configuration
        data_directory: Directory containing data files
        verbose: Enable verbose logging
        
    Returns:
        BaselineResult with Gemini baseline evaluation
    """
    logger = get_logger(__name__, verbose, domain_name)
    
    # Load Gemini reference file
    gemini_file = Path(f"data/references/{domain_name}_gemini.json")
    
    if not gemini_file.exists():
        raise FileNotFoundError(f"Gemini reference file not found: {gemini_file}")
    
    with open(gemini_file, "r") as f:
        gemini_data = json.load(f)
    
    # Extract year boundaries from Gemini data
    periods = gemini_data["historical_periods"]
    segments = []
    
    for period in periods:
        start_year = period["start_year"]
        end_year = period["end_year"]
        segments.append((start_year, end_year))
    
    if verbose:
        logger.info(f"Gemini baseline: {len(segments)} segments")
        logger.info(f"Segments: {segments}")
    
    # Load domain data with same processing as pipeline
    success, academic_years, error_message = load_domain_data(
        domain_name=domain_name,
        algorithm_config=algorithm_config,
        data_directory=data_directory,
        verbose=verbose,
    )
    
    if not success:
        raise RuntimeError(f"Failed to load domain data: {error_message}")
    
    # Filter academic years to match data range
    data_years = set(year.year for year in academic_years)
    min_data_year = min(data_years)
    max_data_year = max(data_years)
    
    # Adjust segments to fit within data range
    adjusted_segments = []
    for start_year, end_year in segments:
        # Clip to data range
        adj_start = max(start_year, min_data_year)
        adj_end = min(end_year, max_data_year)
        
        # Only include segments that have data
        if adj_start <= adj_end and adj_start in data_years:
            adjusted_segments.append((adj_start, adj_end))
    
    if not adjusted_segments:
        raise ValueError(f"No Gemini segments overlap with data range {min_data_year}-{max_data_year}")
    
    # Create academic periods from segments
    academic_periods = create_academic_periods_from_segments(
        academic_years=tuple(academic_years),
        segments=adjusted_segments,
        algorithm_config=algorithm_config,
    )
    
    # Compute objective function
    obj_result = compute_objective_function(
        academic_periods=academic_periods,
        algorithm_config=algorithm_config,
        verbose=verbose,
    )
    
    # Extract boundary years
    boundary_years = []
    for i, period in enumerate(academic_periods):
        if i == 0:
            boundary_years.append(period.start_year)
        boundary_years.append(period.end_year)
    boundary_years = sorted(set(boundary_years))
    
    if verbose:
        logger.info(f"Gemini baseline objective score: {obj_result.final_score:.3f}")
    
    return BaselineResult(
        baseline_name="Gemini",
        objective_score=obj_result.final_score,
        cohesion_score=obj_result.cohesion_score,
        separation_score=obj_result.separation_score,
        num_segments=len(academic_periods),
        boundary_years=boundary_years,
        academic_periods=academic_periods,
    )


def create_manual_baseline(
    domain_name: str,
    algorithm_config: AlgorithmConfig,
    data_directory: str = "resources",
    verbose: bool = False,
) -> BaselineResult:
    """Create baseline using manual reference timeline.
    
    Args:
        domain_name: Domain name to evaluate
        algorithm_config: Algorithm configuration
        data_directory: Directory containing data files
        verbose: Enable verbose logging
        
    Returns:
        BaselineResult with manual baseline evaluation
    """
    logger = get_logger(__name__, verbose, domain_name)
    
    # Load manual reference file
    manual_file = Path(f"data/references/{domain_name}_manual.json")
    
    if not manual_file.exists():
        raise FileNotFoundError(f"Manual reference file not found: {manual_file}")
    
    with open(manual_file, "r") as f:
        manual_data = json.load(f)
    
    # Extract year boundaries from manual data (different format)
    periods = manual_data["historical_periods"]
    segments = []
    
    for period in periods:
        years_str = period["years"]
        # Parse years like "1400-1600" or "1946-2025"
        if "-" in years_str:
            start_year, end_year = map(int, years_str.split("-"))
            segments.append((start_year, end_year))
    
    if verbose:
        logger.info(f"Manual baseline: {len(segments)} segments")
        logger.info(f"Segments: {segments}")
    
    # Load domain data with same processing as pipeline
    success, academic_years, error_message = load_domain_data(
        domain_name=domain_name,
        algorithm_config=algorithm_config,
        data_directory=data_directory,
        verbose=verbose,
    )
    
    if not success:
        raise RuntimeError(f"Failed to load domain data: {error_message}")
    
    # Filter academic years to match data range
    data_years = set(year.year for year in academic_years)
    min_data_year = min(data_years)
    max_data_year = max(data_years)
    
    # Adjust segments to fit within data range
    adjusted_segments = []
    for start_year, end_year in segments:
        # Clip to data range
        adj_start = max(start_year, min_data_year)
        adj_end = min(end_year, max_data_year)
        
        # Only include segments that have data
        if adj_start <= adj_end and adj_start in data_years:
            adjusted_segments.append((adj_start, adj_end))
    
    if not adjusted_segments:
        raise ValueError(f"No manual segments overlap with data range {min_data_year}-{max_data_year}")
    
    # Create academic periods from segments
    academic_periods = create_academic_periods_from_segments(
        academic_years=tuple(academic_years),
        segments=adjusted_segments,
        algorithm_config=algorithm_config,
    )
    
    # Compute objective function
    obj_result = compute_objective_function(
        academic_periods=academic_periods,
        algorithm_config=algorithm_config,
        verbose=verbose,
    )
    
    # Extract boundary years
    boundary_years = []
    for i, period in enumerate(academic_periods):
        if i == 0:
            boundary_years.append(period.start_year)
        boundary_years.append(period.end_year)
    boundary_years = sorted(set(boundary_years))
    
    if verbose:
        logger.info(f"Manual baseline objective score: {obj_result.final_score:.3f}")
    
    return BaselineResult(
        baseline_name="Manual",
        objective_score=obj_result.final_score,
        cohesion_score=obj_result.cohesion_score,
        separation_score=obj_result.separation_score,
        num_segments=len(academic_periods),
        boundary_years=boundary_years,
        academic_periods=academic_periods,
    )


def create_fixed_year_baseline(
    domain_name: str,
    algorithm_config: AlgorithmConfig,
    year_interval: int,
    data_directory: str = "resources",
    verbose: bool = False,
) -> BaselineResult:
    """Create baseline using fixed year intervals.
    
    Args:
        domain_name: Domain name to evaluate
        algorithm_config: Algorithm configuration
        year_interval: Year interval (5 or 10)
        data_directory: Directory containing data files
        verbose: Enable verbose logging
        
    Returns:
        BaselineResult with fixed year baseline evaluation
    """
    logger = get_logger(__name__, verbose, domain_name)
    
    # Load domain data with same processing as pipeline
    success, academic_years, error_message = load_domain_data(
        domain_name=domain_name,
        algorithm_config=algorithm_config,
        data_directory=data_directory,
        verbose=verbose,
    )
    
    if not success:
        raise RuntimeError(f"Failed to load domain data: {error_message}")
    
    # Get data year range
    data_years = sorted(year.year for year in academic_years)
    min_data_year = min(data_years)
    max_data_year = max(data_years)
    
    # Create fixed year segments
    segments = []
    current_year = min_data_year
    
    while current_year <= max_data_year:
        end_year = min(current_year + year_interval - 1, max_data_year)
        segments.append((current_year, end_year))
        current_year = end_year + 1
    
    if verbose:
        logger.info(f"{year_interval}-year baseline: {len(segments)} segments (before filtering)")
        logger.info(f"Segments: {segments}")
    
    # Filter segments to only include those with academic years
    data_years_set = set(year.year for year in academic_years)
    valid_segments = []
    
    for start_year, end_year in segments:
        # Check if segment has any academic years with data
        segment_years = [year for year in range(start_year, end_year + 1) if year in data_years_set]
        if segment_years:
            valid_segments.append((start_year, end_year))
        elif verbose:
            logger.info(f"Skipping segment {start_year}-{end_year}: no academic years with data")
    
    if not valid_segments:
        raise ValueError(f"No valid segments found for {year_interval}-year baseline")
    
    if verbose:
        logger.info(f"{year_interval}-year baseline: {len(valid_segments)} valid segments (after filtering)")
        logger.info(f"Valid segments: {valid_segments}")
    
    # Create academic periods from valid segments
    academic_periods = create_academic_periods_from_segments(
        academic_years=tuple(academic_years),
        segments=valid_segments,
        algorithm_config=algorithm_config,
    )
    
    # Compute objective function
    obj_result = compute_objective_function(
        academic_periods=academic_periods,
        algorithm_config=algorithm_config,
        verbose=verbose,
    )
    
    # Extract boundary years
    boundary_years = []
    for i, period in enumerate(academic_periods):
        if i == 0:
            boundary_years.append(period.start_year)
        boundary_years.append(period.end_year)
    boundary_years = sorted(set(boundary_years))
    
    baseline_name = f"{year_interval}-year"
    
    if verbose:
        logger.info(f"{baseline_name} baseline objective score: {obj_result.final_score:.3f}")
    
    return BaselineResult(
        baseline_name=baseline_name,
        objective_score=obj_result.final_score,
        cohesion_score=obj_result.cohesion_score,
        separation_score=obj_result.separation_score,
        num_segments=len(academic_periods),
        boundary_years=boundary_years,
        academic_periods=academic_periods,
    )


def calculate_boundary_f1(
    predicted_boundaries: List[int],
    ground_truth_boundaries: List[int],
    tolerance: int = 2,
) -> Tuple[float, float, float]:
    """Calculate F1 score for boundary year predictions with tolerance.
    
    Args:
        predicted_boundaries: Predicted boundary years
        ground_truth_boundaries: Ground truth boundary years
        tolerance: Year tolerance for matching
        
    Returns:
        Tuple of (f1_score, precision, recall)
    """
    if not predicted_boundaries or not ground_truth_boundaries:
        return 0.0, 0.0, 0.0
    
    # Convert to sets for efficient lookup
    pred_set = set(predicted_boundaries)
    gt_set = set(ground_truth_boundaries)
    
    # Find matches with tolerance
    true_positives = 0
    matched_gt = set()
    
    for pred_year in pred_set:
        for gt_year in gt_set:
            if abs(pred_year - gt_year) <= tolerance and gt_year not in matched_gt:
                true_positives += 1
                matched_gt.add(gt_year)
                break
    
    # Calculate precision and recall
    precision = true_positives / len(pred_set) if pred_set else 0.0
    recall = true_positives / len(gt_set) if gt_set else 0.0
    
    # Calculate F1 score
    if precision + recall == 0:
        f1_score = 0.0
    else:
        f1_score = 2 * precision * recall / (precision + recall)
    
    return f1_score, precision, recall


def calculate_segment_f1(
    predicted_segments: List[Tuple[int, int]],
    ground_truth_segments: List[Tuple[int, int]],
    max_segments_per_match: int = 3,
) -> Tuple[float, float, float]:
    """Calculate F1 score for segment predictions.
    
    A ground truth segment is considered matched if it can be represented
    by no more than max_segments_per_match predicted segments, and vice versa.
    
    Args:
        predicted_segments: List of (start_year, end_year) tuples
        ground_truth_segments: List of (start_year, end_year) tuples
        max_segments_per_match: Maximum segments allowed per match
        
    Returns:
        Tuple of (f1_score, precision, recall)
    """
    if not predicted_segments or not ground_truth_segments:
        return 0.0, 0.0, 0.0
    
    def segments_overlap(seg1, seg2):
        """Check if two segments overlap."""
        return not (seg1[1] < seg2[0] or seg2[1] < seg1[0])
    
    def get_overlapping_segments(target_segment, segment_list):
        """Get segments that overlap with target segment."""
        return [seg for seg in segment_list if segments_overlap(target_segment, seg)]
    
    # Calculate precision: how many predicted segments can be matched
    matched_predicted = 0
    for pred_seg in predicted_segments:
        overlapping_gt = get_overlapping_segments(pred_seg, ground_truth_segments)
        if 1 <= len(overlapping_gt) <= max_segments_per_match:
            matched_predicted += 1
    
    # Calculate recall: how many ground truth segments can be matched
    matched_ground_truth = 0
    for gt_seg in ground_truth_segments:
        overlapping_pred = get_overlapping_segments(gt_seg, predicted_segments)
        if 1 <= len(overlapping_pred) <= max_segments_per_match:
            matched_ground_truth += 1
    
    # Calculate precision and recall
    precision = matched_predicted / len(predicted_segments) if predicted_segments else 0.0
    recall = matched_ground_truth / len(ground_truth_segments) if ground_truth_segments else 0.0
    
    # Calculate F1 score
    if precision + recall == 0:
        f1_score = 0.0
    else:
        f1_score = 2 * precision * recall / (precision + recall)
    
    return f1_score, precision, recall


def run_comprehensive_evaluation(
    domain_name: str,
    timeline_result: TimelineAnalysisResult,
    algorithm_config: AlgorithmConfig,
    data_directory: str = "resources",
    verbose: bool = False,
) -> ComprehensiveEvaluationResult:
    """Run comprehensive evaluation including all baselines and auto-metrics.
    
    Args:
        domain_name: Domain name to evaluate
        timeline_result: Timeline result from pipeline
        algorithm_config: Algorithm configuration
        data_directory: Directory containing data files
        verbose: Enable verbose logging
        
    Returns:
        ComprehensiveEvaluationResult with all evaluation metrics
    """
    logger = get_logger(__name__, verbose, domain_name)
    
    if verbose:
        logger.info(f"Running comprehensive evaluation for {domain_name}")
    
    # 1. Evaluate algorithm result
    algorithm_result = evaluate_timeline_result(
        timeline_result=timeline_result,
        algorithm_config=algorithm_config,
        verbose=verbose,
    )
    
    # 2. Create and evaluate baselines
    baseline_results = []
    
    # Gemini baseline
    try:
        gemini_baseline = create_gemini_baseline(
            domain_name=domain_name,
            algorithm_config=algorithm_config,
            data_directory=data_directory,
            verbose=verbose,
        )
        baseline_results.append(gemini_baseline)
    except (FileNotFoundError, ValueError) as e:
        if verbose:
            logger.warning(f"Gemini baseline failed: {e}")
    
    # Manual baseline
    try:
        manual_baseline = create_manual_baseline(
            domain_name=domain_name,
            algorithm_config=algorithm_config,
            data_directory=data_directory,
            verbose=verbose,
        )
        baseline_results.append(manual_baseline)
    except (FileNotFoundError, ValueError) as e:
        if verbose:
            logger.warning(f"Manual baseline failed: {e}")
    
    # 5-year baseline
    try:
        five_year_baseline = create_fixed_year_baseline(
            domain_name=domain_name,
            algorithm_config=algorithm_config,
            year_interval=5,
            data_directory=data_directory,
            verbose=verbose,
        )
        baseline_results.append(five_year_baseline)
    except (ValueError, RuntimeError) as e:
        if verbose:
            logger.warning(f"5-year baseline failed: {e}")
    
    # 10-year baseline
    try:
        ten_year_baseline = create_fixed_year_baseline(
            domain_name=domain_name,
            algorithm_config=algorithm_config,
            year_interval=10,
            data_directory=data_directory,
            verbose=verbose,
        )
        baseline_results.append(ten_year_baseline)
    except (ValueError, RuntimeError) as e:
        if verbose:
            logger.warning(f"10-year baseline failed: {e}")
    
    # 3. Calculate auto-metrics against manual baseline
    auto_metrics = None
    manual_baseline_result = None
    
    for baseline in baseline_results:
        if baseline.baseline_name == "Manual":
            manual_baseline_result = baseline
            break
    
    if manual_baseline_result is not None:
        # Extract segments from timeline result
        algorithm_segments = []
        for period in timeline_result.periods:
            algorithm_segments.append((period.start_year, period.end_year))
        
        # Extract segments from manual baseline
        manual_segments = []
        for period in manual_baseline_result.academic_periods:
            manual_segments.append((period.start_year, period.end_year))
        
        # Calculate boundary F1
        boundary_f1, boundary_precision, boundary_recall = calculate_boundary_f1(
            predicted_boundaries=algorithm_result.boundary_years,
            ground_truth_boundaries=manual_baseline_result.boundary_years,
            tolerance=2,
        )
        
        # Calculate segment F1
        segment_f1, segment_precision, segment_recall = calculate_segment_f1(
            predicted_segments=algorithm_segments,
            ground_truth_segments=manual_segments,
            max_segments_per_match=3,
        )
        
        auto_metrics = AutoMetricResult(
            boundary_f1=boundary_f1,
            boundary_precision=boundary_precision,
            boundary_recall=boundary_recall,
            segment_f1=segment_f1,
            segment_precision=segment_precision,
            segment_recall=segment_recall,
            tolerance=2,
            details={
                "algorithm_boundaries": str(algorithm_result.boundary_years),
                "manual_boundaries": str(manual_baseline_result.boundary_years),
                "algorithm_segments": str(algorithm_segments),
                "manual_segments": str(manual_segments),
            },
        )
    else:
        # Create default auto-metrics if manual baseline not available
        auto_metrics = AutoMetricResult(
            boundary_f1=0.0,
            boundary_precision=0.0,
            boundary_recall=0.0,
            segment_f1=0.0,
            segment_precision=0.0,
            segment_recall=0.0,
            tolerance=2,
            details={"error": "Manual baseline not available"},
        )
    
    # 4. Create ranking
    ranking = {"Algorithm": algorithm_result.objective_score}
    for baseline in baseline_results:
        ranking[baseline.baseline_name] = baseline.objective_score
    
    # 5. Generate summary
    summary_parts = [
        f"Evaluation Results for {domain_name}:",
        f"Algorithm Score: {algorithm_result.objective_score:.3f}",
    ]
    
    if baseline_results:
        summary_parts.append("Baseline Scores:")
        for baseline in baseline_results:
            summary_parts.append(f"  {baseline.baseline_name}: {baseline.objective_score:.3f}")
    
    if auto_metrics and manual_baseline_result:
        summary_parts.extend([
            f"Auto-Metrics vs Manual:",
            f"  Boundary F1: {auto_metrics.boundary_f1:.3f}",
            f"  Segment F1: {auto_metrics.segment_f1:.3f}",
        ])
    
    summary = "\n".join(summary_parts)
    
    if verbose:
        logger.info("Comprehensive evaluation completed")
        logger.info(summary)
    
    return ComprehensiveEvaluationResult(
        domain_name=domain_name,
        algorithm_result=algorithm_result,
        baseline_results=baseline_results,
        auto_metrics=auto_metrics,
        ranking=ranking,
        summary=summary,
    ) 