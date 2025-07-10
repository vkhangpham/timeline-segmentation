"""Timeline evaluation metrics calculation.

This module contains functions for calculating various evaluation metrics
including F1 scores, precision, recall, and other comparison metrics.
"""

from typing import Dict, List, Tuple


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


def calculate_f1_score_between_methods(
    method1_boundaries: List[int], 
    method2_boundaries: List[int], 
    tolerance: int = 2
) -> Dict[str, float]:
    """Calculate F1 score between two segmentation methods.
    
    Args:
        method1_boundaries: Boundary years from first method
        method2_boundaries: Boundary years from second method (reference)
        tolerance: Tolerance in years for matching boundaries
        
    Returns:
        Dictionary with boundary and segment F1 scores
    """
    def find_matches(pred_boundaries, true_boundaries, tolerance):
        """Find matching boundaries within tolerance."""
        matches = 0
        for pred_boundary in pred_boundaries:
            for true_boundary in true_boundaries:
                if abs(pred_boundary - true_boundary) <= tolerance:
                    matches += 1
                    break
        return matches
    
    # Calculate boundary metrics
    boundary_matches = find_matches(method1_boundaries, method2_boundaries, tolerance)
    
    boundary_precision = boundary_matches / len(method1_boundaries) if method1_boundaries else 0.0
    boundary_recall = boundary_matches / len(method2_boundaries) if method2_boundaries else 0.0
    boundary_f1 = (2 * boundary_precision * boundary_recall) / (boundary_precision + boundary_recall) if (boundary_precision + boundary_recall) > 0 else 0.0
    
    # Calculate segment metrics (simplified - based on number of segments)
    method1_segments = len(method1_boundaries) + 1 if method1_boundaries else 1
    method2_segments = len(method2_boundaries) + 1 if method2_boundaries else 1
    
    # For segments, we use a simpler heuristic based on boundary matches
    # This is a simplified version - in practice, segment overlap would be more complex
    segment_matches = boundary_matches + 1  # At least one segment should match
    
    segment_precision = segment_matches / method1_segments if method1_segments else 0.0
    segment_recall = segment_matches / method2_segments if method2_segments else 0.0
    segment_f1 = (2 * segment_precision * segment_recall) / (segment_precision + segment_recall) if (segment_precision + segment_recall) > 0 else 0.0
    
    return {
        "boundary_f1": boundary_f1,
        "boundary_precision": boundary_precision,
        "boundary_recall": boundary_recall,
        "segment_f1": segment_f1,
        "segment_precision": segment_precision,
        "segment_recall": segment_recall,
    } 