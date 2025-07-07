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

from typing import List
from collections import Counter
from ..data.data_models import AcademicPeriod
from ..utils.logging import get_logger


# =============================================================================
# SIMPLIFIED MERGING FUNCTIONS (for new architecture)
# =============================================================================


def merge_similar_periods(
    periods: List[AcademicPeriod], algorithm_config, verbose: bool = False
) -> List[AcademicPeriod]:
    """
    Merge similar adjacent periods.

    This is the simplified function for the new architecture that works directly
    with AcademicPeriod objects.

    Args:
        periods: List of characterized periods
        algorithm_config: Algorithm configuration
        verbose: Enable verbose logging

    Returns:
        List of merged periods
    """
    logger = get_logger(__name__, verbose)

    if verbose:
        logger.info(f"=== PERIOD MERGING STARTED ===")
        logger.info(f"  Input periods: {len(periods)}")
        for i, period in enumerate(periods):
            logger.info(
                f"  Period {i+1}: {period.start_year}-{period.end_year} ({period.topic_label})"
            )

    if len(periods) <= 1:
        if verbose:
            logger.info("  No merging needed - only one period")
        return periods

    # Simple merging based on keyword similarity
    merged_periods = [periods[0]]
    similarity_threshold = getattr(algorithm_config, "merge_similarity_threshold", 0.6)

    if verbose:
        logger.info(f"  Similarity threshold: {similarity_threshold}")

    for i, current_period in enumerate(periods[1:], 1):
        last_period = merged_periods[-1]

        # Calculate keyword overlap
        last_keywords = set(last_period.top_keywords[:10])
        current_keywords = set(current_period.top_keywords[:10])

        if last_keywords and current_keywords:
            overlap = len(last_keywords & current_keywords) / len(
                last_keywords | current_keywords
            )

            if verbose:
                logger.info(
                    f"  Comparing period {i}: {current_period.start_year}-{current_period.end_year}"
                )
                logger.info(f"    Last keywords: {', '.join(list(last_keywords)[:5])}")
                logger.info(
                    f"    Current keywords: {', '.join(list(current_keywords)[:5])}"
                )
                logger.info(f"    Keyword overlap: {overlap:.3f}")

            if overlap > similarity_threshold:
                # Merge with last period
                if verbose:
                    logger.info(
                        f"    MERGING: overlap {overlap:.3f} > threshold {similarity_threshold}"
                    )

                combined_academic_years = (
                    last_period.academic_years + current_period.academic_years
                )

                # OPTIMIZATION: Efficient keyword frequency merging
                # Use Counter for efficient merging instead of manual dictionary operations
                last_counter = Counter(last_period.combined_keyword_frequencies)
                current_counter = Counter(current_period.combined_keyword_frequencies)
                combined_counter = last_counter + current_counter

                combined_keywords = dict(combined_counter)
                top_keywords = tuple(
                    keyword for keyword, freq in combined_counter.most_common(50)
                )

                # OPTIMIZATION: Intelligent characterization merging
                # Use weighted averaging based on paper counts for better representation
                total_weight = last_period.total_papers + current_period.total_papers
                last_weight = (
                    last_period.total_papers / total_weight if total_weight > 0 else 0.5
                )
                current_weight = (
                    current_period.total_papers / total_weight
                    if total_weight > 0
                    else 0.5
                )

                # Generate more informative merged topic label
                if last_period.topic_label and current_period.topic_label:
                    if "Merged:" not in last_period.topic_label:
                        topic_label = f"Merged: {last_period.topic_label} & {current_period.topic_label}"
                    else:
                        topic_label = (
                            f"{last_period.topic_label} & {current_period.topic_label}"
                        )
                else:
                    topic_label = (
                        last_period.topic_label
                        or current_period.topic_label
                        or "Merged Period"
                    )

                topic_description = f"Combined research spanning {last_period.topic_description or 'research'} and {current_period.topic_description or 'related work'}"

                # Weighted confidence calculation
                merged_confidence = (
                    last_weight * last_period.confidence
                    + current_weight * current_period.confidence
                )

                # Merge network metrics with weighted averaging
                merged_network_stability = last_weight * getattr(
                    last_period, "network_stability", 0.5
                ) + current_weight * getattr(current_period, "network_stability", 0.5)
                merged_community_persistence = last_weight * getattr(
                    last_period, "community_persistence", 0.5
                ) + current_weight * getattr(
                    current_period, "community_persistence", 0.5
                )
                merged_flow_stability = last_weight * getattr(
                    last_period, "flow_stability", 0.5
                ) + current_weight * getattr(current_period, "flow_stability", 0.5)
                merged_centrality_consensus = last_weight * getattr(
                    last_period, "centrality_consensus", 0.5
                ) + current_weight * getattr(
                    current_period, "centrality_consensus", 0.5
                )

                merged_period = AcademicPeriod(
                    start_year=last_period.start_year,
                    end_year=current_period.end_year,
                    academic_years=combined_academic_years,
                    total_papers=last_period.total_papers + current_period.total_papers,
                    total_citations=last_period.total_citations
                    + current_period.total_citations,
                    combined_keyword_frequencies=combined_keywords,
                    top_keywords=top_keywords,
                    topic_label=topic_label,
                    topic_description=topic_description,
                    confidence=merged_confidence,
                    network_stability=merged_network_stability,
                    community_persistence=merged_community_persistence,
                    flow_stability=merged_flow_stability,
                    centrality_consensus=merged_centrality_consensus,
                )

                # Replace last period with merged one
                merged_periods[-1] = merged_period

                if verbose:
                    logger.info(
                        f"    Created merged period: {merged_period.start_year}-{merged_period.end_year}"
                    )
                    logger.info(f"    Total papers: {merged_period.total_papers}")
                continue
            else:
                if verbose:
                    logger.info(
                        f"    NO MERGE: overlap {overlap:.3f} < threshold {similarity_threshold}"
                    )

        # No merge - add current period
        merged_periods.append(current_period)
        if verbose:
            logger.info(f"    Added as separate period")

    if verbose:
        logger.info("=== PERIOD MERGING COMPLETED ===")
        logger.info(f"  Final periods: {len(merged_periods)}")
        for i, period in enumerate(merged_periods):
            logger.info(
                f"  Final period {i+1}: {period.start_year}-{period.end_year} ({period.topic_label})"
            )

    logger.info(f"Merged {len(periods)} periods into {len(merged_periods)} periods")
    return merged_periods


# =============================================================================
# LEGACY MERGING FUNCTIONS REMOVED
# =============================================================================
# All legacy functions (merge_similar_segments, calculate_semantic_similarities,
# analyze_boundary_signal_strengths, identify_merge_candidates, execute_segment_merging,
# merge_two_segments, generate_merging_summary) have been removed as they used deprecated models.


# LEGACY FUNCTION REMOVED: generate_merging_summary
# This function used deprecated PeriodCharacterization model
