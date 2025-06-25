"""
Segment Merging for Timeline Analysis

This module implements intelligent segment merging that identifies consecutive segments
that are semantically similar and have weak shift signals between them.

Core functionality:
- Semantic similarity detection between consecutive segments
- Shift signal strength analysis at segment boundaries
- Intelligent merging with confidence scoring
- Representative paper consolidation during merging

Follows functional programming principles with pure functions and fail-fast error handling.
"""

from typing import List, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, util
import networkx as nx

from .paper_selection_and_labeling import generate_merged_segment_label_and_description
from .data_models import (
    PeriodCharacterization, ShiftSignal,
    MergeDecision, SegmentMergingResult
)


def merge_similar_segments(
    period_characterizations: List[PeriodCharacterization],
    shift_signals: List[ShiftSignal],
    domain_name: str,
    similarity_threshold: float = 0.75,
    weak_signal_threshold: float = 0.4
) -> SegmentMergingResult:
    """
    Main function: Merge semantically similar consecutive segments with weak boundaries.
    
    Args:
        period_characterizations: List of characterized periods
        shift_signals: List of detected shift signals
        domain_name: Name of the domain for configuration
        similarity_threshold: Threshold for semantic similarity (0.0-1.0)
        weak_signal_threshold: Threshold for weak shift signals (0.0-1.0)
        
    Returns:
        Segment merging results with merged periods
    """
    print(f"\nSEGMENT MERGING ANALYSIS: {domain_name}")
    
    if len(period_characterizations) < 2:
        print("Less than 2 segments - no merging needed")
        return SegmentMergingResult(
            original_segments=tuple(period_characterizations),
            merged_segments=tuple(period_characterizations),
            merge_decisions=tuple(),
            merging_summary="No merging needed - insufficient segments"
        )
    
    print(f"Analyzing {len(period_characterizations)} segments for potential merging")
    print(f"Similarity threshold: {similarity_threshold:.3f}")
    print(f"Weak signal threshold: {weak_signal_threshold:.3f}")
    
    # Step 1: Calculate semantic similarities between consecutive segments
    semantic_similarities = calculate_semantic_similarities(period_characterizations)
    
    # Step 2: Analyze shift signal strengths at boundaries
    boundary_signal_strengths = analyze_boundary_signal_strengths(
        period_characterizations, shift_signals
    )
    
    # Step 3: Identify merge candidates based on similarity and weak signals
    merge_candidates = identify_merge_candidates(
        period_characterizations,
        semantic_similarities,
        boundary_signal_strengths,
        similarity_threshold,
        weak_signal_threshold
    )
    
    print(f"Found {len(merge_candidates)} merge candidates")
    
    # Step 4: Execute merging with conflict resolution
    merged_segments, merge_decisions = execute_segment_merging(
        period_characterizations,
        merge_candidates,
        domain_name
    )
    
    # Step 5: Generate merging summary
    merging_summary = generate_merging_summary(
        period_characterizations, merged_segments, merge_decisions
    )
    
    print(f"Merged {len(period_characterizations)} → {len(merged_segments)} segments")
    print(f"{merging_summary}")
    
    return SegmentMergingResult(
        original_segments=tuple(period_characterizations),
        merged_segments=tuple(merged_segments),
        merge_decisions=tuple(merge_decisions),
        merging_summary=merging_summary
    )


def calculate_semantic_similarities(
    period_characterizations: List[PeriodCharacterization]
) -> List[float]:
    """
    Calculate semantic similarities between consecutive segments using multiple signals.
    
    Args:
        period_characterizations: List of period characterizations
        
    Returns:
        List of similarity scores for consecutive pairs (length = n-1)
    """
    if len(period_characterizations) < 2:
        return []
    
    similarities = []
    
    # Prepare text corpus for TF-IDF analysis
    segment_texts = []
    for pc in period_characterizations:
        # Combine multiple semantic signals
        combined_text = []
        
        # Label and description
        combined_text.append(pc.topic_label)
        combined_text.append(pc.topic_description)
        
        # Representative paper titles and abstracts
        for paper in pc.representative_papers:
            combined_text.append(paper.get('title', ''))
        
        # Join all text for this segment
        segment_text = ' '.join(combined_text)
        segment_texts.append(segment_text)
    
    # Calculate TF-IDF similarity matrix
    try:
        vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 3),
            min_df=1  # Allow all terms since we have small corpus
        )
        
        tfidf_matrix = vectorizer.fit_transform(segment_texts)
        
        # Calculate consecutive similarities
        for i in range(len(period_characterizations) - 1):
            similarity = cosine_similarity(
                tfidf_matrix[i:i+1], 
                tfidf_matrix[i+1:i+2]
            )[0, 0]
            
            # Additional semantic similarity factors
            
            # Network stability similarity (periods with similar stability characteristics)
            stability_similarity = 1.0 - abs(
                period_characterizations[i].network_stability - 
                period_characterizations[i+1].network_stability
            )
            
            # Confidence similarity (periods with similar characterization confidence)
            confidence_similarity = 1.0 - abs(
                period_characterizations[i].confidence - 
                period_characterizations[i+1].confidence
            )
            
            # Combined similarity score (weighted average)
            combined_similarity = (
                similarity * 0.7 +  # Text similarity dominant
                stability_similarity * 0.2 +  # Network characteristics
                confidence_similarity * 0.1   # Confidence alignment
            )
            
            similarities.append(combined_similarity)
            
            print(f"Segments {i} → {i+1}: text={similarity:.3f}, "
                  f"stability={stability_similarity:.3f}, "
                  f"combined={combined_similarity:.3f}")
    
    except Exception as e:
        print(f"TF-IDF similarity calculation failed: {e}")
        # Fallback to simple label similarity
        for i in range(len(period_characterizations) - 1):
            label1 = period_characterizations[i].topic_label.lower()
            label2 = period_characterizations[i+1].topic_label.lower()
            
            # Simple word overlap similarity
            words1 = set(label1.split())
            words2 = set(label2.split())
            
            if len(words1) == 0 and len(words2) == 0:
                similarity = 1.0
            elif len(words1) == 0 or len(words2) == 0:
                similarity = 0.0
            else:
                overlap = len(words1 & words2)
                union = len(words1 | words2)
                similarity = overlap / union if union > 0 else 0.0
            
            similarities.append(similarity)
    
    return similarities


def analyze_boundary_signal_strengths(
    period_characterizations: List[PeriodCharacterization],
    shift_signals: List[ShiftSignal]
) -> List[float]:
    """
    Analyze shift signal strengths at segment boundaries.
    
    Args:
        period_characterizations: List of period characterizations
        shift_signals: List of detected shift signals
        
    Returns:
        List of signal strengths at boundaries (length = n-1)
    """
    if len(period_characterizations) < 2:
        return []
    
    boundary_strengths = []
    
    for i in range(len(period_characterizations) - 1):
        # Get boundary year between consecutive segments
        boundary_year = period_characterizations[i].period[1]  # End of segment i
        
        # Find shift signals near this boundary (within ±2 years)
        nearby_signals = []
        for signal in shift_signals:
            if abs(signal.year - boundary_year) <= 2:
                nearby_signals.append(signal)
        
        if nearby_signals:
            # Calculate boundary strength based on nearby signals
            # Use the confidence of the strongest signal at the boundary
            max_confidence = max(s.confidence for s in nearby_signals)
            boundary_strength = max_confidence
            
            print(f"Boundary {boundary_year}: {len(nearby_signals)} signals, "
                  f"strength={boundary_strength:.3f}")
        else:
            # No signals near boundary = weak boundary
            boundary_strength = 0.0
            print(f"Boundary {boundary_year}: no signals, strength=0.000")
        
        boundary_strengths.append(boundary_strength)
    
    return boundary_strengths


def identify_merge_candidates(
    period_characterizations: List[PeriodCharacterization],
    semantic_similarities: List[float],
    boundary_signal_strengths: List[float],
    similarity_threshold: float,
    weak_signal_threshold: float
) -> List[Tuple[int, int, float, float]]:
    """
    Identify consecutive segments that should be merged.
    
    Args:
        period_characterizations: List of period characterizations
        semantic_similarities: Similarity scores between consecutive segments
        boundary_signal_strengths: Signal strengths at boundaries
        similarity_threshold: Minimum similarity for merging
        weak_signal_threshold: Maximum signal strength for weak boundary
        
    Returns:
        List of (index1, index2, similarity, signal_strength) tuples
    """
    merge_candidates = []
    
    for i in range(len(semantic_similarities)):
        similarity = semantic_similarities[i]
        signal_strength = boundary_signal_strengths[i]
        
        # Check merging criteria
        high_similarity = similarity >= similarity_threshold
        weak_boundary = signal_strength <= weak_signal_threshold
        
        if high_similarity and weak_boundary:
            merge_candidates.append((i, i + 1, similarity, signal_strength))
            print(f"Merge candidate: segments {i} → {i+1} "
                  f"(similarity={similarity:.3f}, boundary={signal_strength:.3f})")
        else:
            reasons = []
            if not high_similarity:
                reasons.append(f"low similarity ({similarity:.3f})")
            if not weak_boundary:
                reasons.append(f"strong boundary ({signal_strength:.3f})")
            
            print(f"No merge: segments {i} → {i+1} - {', '.join(reasons)}")
    
    return merge_candidates


def execute_segment_merging(
    period_characterizations: List[PeriodCharacterization],
    merge_candidates: List[Tuple[int, int, float, float]],
    domain_name: str
) -> Tuple[List[PeriodCharacterization], List[MergeDecision]]:
    """
    Execute segment merging with conflict resolution.
    
    Args:
        period_characterizations: Original period characterizations
        merge_candidates: List of merge candidates
        domain_name: Domain name for configuration
        
    Returns:
        Tuple of (merged_segments, merge_decisions)
    """
    if not merge_candidates:
        return list(period_characterizations), []
    
    # Sort merge candidates by confidence (highest first)
    merge_candidates.sort(key=lambda x: x[2], reverse=True)  # Sort by similarity
    
    merged_segments = list(period_characterizations)
    merge_decisions = []
    processed_indices = set()
    
    for idx1, idx2, similarity, signal_strength in merge_candidates:
        # Check if either segment already processed
        if idx1 in processed_indices or idx2 in processed_indices:
            continue
        
        # Check if indices still valid after previous merges
        if idx1 >= len(merged_segments) or idx2 >= len(merged_segments):
            continue
        
        # Execute merge
        segment1 = merged_segments[idx1]
        segment2 = merged_segments[idx2]
        
        merged_segment = merge_two_segments(segment1, segment2, domain_name)
        
        # Calculate merge confidence
        merge_confidence = (similarity + (1.0 - signal_strength)) / 2
        
        # Create merge decision
        merge_decision = MergeDecision(
            segment1_index=idx1,
            segment2_index=idx2,
            semantic_similarity=similarity,
            shift_signal_strength=signal_strength,
            merge_confidence=merge_confidence,
            merge_justification=f"Similarity {similarity:.3f}, weak boundary {signal_strength:.3f}",
            merged_period=merged_segment.period,
            merged_label=merged_segment.topic_label,
            merged_description=merged_segment.topic_description
        )
        
        merge_decisions.append(merge_decision)
        
        # Update segments list (replace segment1 with merged, remove segment2)
        merged_segments[idx1] = merged_segment
        merged_segments.pop(idx2)
        
        # Update processed indices
        processed_indices.add(idx1)
        processed_indices.add(idx2)
        
        print(f"Merged segments {idx1}-{idx2}: "
              f"{segment1.period} + {segment2.period} → {merged_segment.period}")
    
    return merged_segments, merge_decisions


def merge_two_segments(
    segment1: PeriodCharacterization,
    segment2: PeriodCharacterization,
    domain_name: str
) -> PeriodCharacterization:
    """
    Merge two consecutive segments into a single unified segment.
    
    Args:
        segment1: First segment (earlier period)
        segment2: Second segment (later period)
        domain_name: Domain name for context
        
    Returns:
        Merged period characterization
    """
    # Merge time periods
    merged_period = (segment1.period[0], segment2.period[1])
    
    # Combine and deduplicate representative papers
    all_papers = list(segment1.representative_papers) + list(segment2.representative_papers)
    
    # Deduplicate by paper ID while preserving order
    seen_ids = set()
    merged_papers = []
    for paper in all_papers:
        paper_id = paper.get('id', paper.get('title', ''))
        if paper_id not in seen_ids:
            seen_ids.add(paper_id)
            merged_papers.append(paper)
    
    # Keep top papers by score (up to 10)
    merged_papers.sort(key=lambda p: p.get('score', 0), reverse=True)
    merged_papers = merged_papers[:10]
    
    # Calculate merged network metrics (weighted average by period length)
    len1 = segment1.period[1] - segment1.period[0] + 1
    len2 = segment2.period[1] - segment2.period[0] + 1
    total_len = len1 + len2
    
    weight1 = len1 / total_len
    weight2 = len2 / total_len
    
    merged_network_stability = (
        segment1.network_stability * weight1 + 
        segment2.network_stability * weight2
    )
    
    merged_community_persistence = (
        segment1.community_persistence * weight1 + 
        segment2.community_persistence * weight2
    )
    
    merged_flow_stability = (
        segment1.flow_stability * weight1 + 
        segment2.flow_stability * weight2
    )
    
    merged_centrality_consensus = (
        segment1.centrality_consensus * weight1 + 
        segment2.centrality_consensus * weight2
    )
    
    # Merge network metrics
    merged_network_metrics = {}
    for key in segment1.network_metrics:
        if key in segment2.network_metrics:
            merged_network_metrics[key] = (
                segment1.network_metrics[key] * weight1 + 
                segment2.network_metrics[key] * weight2
            )
        else:
            merged_network_metrics[key] = segment1.network_metrics[key]
    
    # Add unique metrics from segment2
    for key in segment2.network_metrics:
        if key not in merged_network_metrics:
            merged_network_metrics[key] = segment2.network_metrics[key]
    
    # Calculate merged confidence (weighted average)
    merged_confidence = (
        segment1.confidence * weight1 + 
        segment2.confidence * weight2
    )
    
    # Generate merged label and description using LLM
    merged_label, merged_description = generate_merged_segment_label_and_description(
        segment1.topic_label,
        segment1.topic_description,
        list(segment1.representative_papers),
        segment2.topic_label,
        segment2.topic_description,
        list(segment2.representative_papers),
        merged_period,
        domain_name
    )
    
    return PeriodCharacterization(
        period=merged_period,
        topic_label=merged_label,
        topic_description=merged_description,
        network_stability=merged_network_stability,
        community_persistence=merged_community_persistence,
        flow_stability=merged_flow_stability,
        centrality_consensus=merged_centrality_consensus,
        representative_papers=tuple(merged_papers),
        network_metrics=merged_network_metrics,
        confidence=merged_confidence
    )


# Label generation functions replaced with LLM-based approach in paper_selection_and_labeling.py


def generate_merging_summary(
    original_segments: List[PeriodCharacterization],
    merged_segments: List[PeriodCharacterization],
    merge_decisions: List[MergeDecision]
) -> str:
    """
    Generate concise summary of merging operations.
    
    Args:
        original_segments: Original segment list
        merged_segments: Final merged segment list
        merge_decisions: List of merge decisions
        
    Returns:
        Summary string
    """
    if not merge_decisions:
        return "No merging performed"
    
    return f"{len(merge_decisions)} merges: {len(original_segments)} → {len(merged_segments)} segments" 