"""Timeline analysis orchestrator with functional pipeline stages.
Provides the main entry point for end-to-end timeline analysis."""

import time
from typing import List

from ..data.data_models import AcademicPeriod, TimelineAnalysisResult
from ..data.data_processing import (
    load_domain_data,
)
from ..segmentation.change_point_detection import detect_boundary_years
from ..segmentation.segmentation import create_segments_from_boundary_years
from ..segmentation.beam_refinement import beam_search_refinement
from ..segment_modeling.segment_modeling import characterize_academic_periods

from ..utils.config import AlgorithmConfig
from ..utils.logging import get_logger


def analyze_timeline(
    domain_name: str,
    algorithm_config: AlgorithmConfig,
    data_directory: str = "resources",
    segmentation_only: bool = False,
    verbose: bool = False,
) -> TimelineAnalysisResult:
    """Main timeline analysis pipeline.

    Args:
        domain_name: Name of the domain to analyze
        algorithm_config: Algorithm configuration (required)
        data_directory: Directory containing domain data
        segmentation_only: If True, stop after segmentation (skip characterization and merging)
        verbose: Enable verbose logging

    Returns:
        TimelineAnalysisResult with final timeline periods

    Raises:
        RuntimeError: If any pipeline stage fails (fail-fast behavior)
    """
    logger = get_logger(__name__, verbose, domain_name)
    start_time = time.time()

    try:
        analysis_type = (
            "segmentation" if segmentation_only else "full timeline analysis"
        )
        logger.info(f"Starting {analysis_type} for {domain_name}")

        success, academic_years, error_message = load_domain_data(
            domain_name=domain_name,
            algorithm_config=algorithm_config,
            data_directory=data_directory,
            min_papers_per_year=5,
            apply_year_filtering=True,
            verbose=verbose,
        )

        if not success:
            raise RuntimeError(f"Failed to load data: {error_message}")

        if not academic_years:
            raise RuntimeError(f"No academic years found for {domain_name}")

        logger.info(f"Loaded {len(academic_years)} academic years")

        boundary_academic_years = detect_boundary_years(
            academic_years=academic_years,
            domain_name=domain_name,
            algorithm_config=algorithm_config,
            use_citation=True,
            use_direction=True,
            verbose=verbose,
        )

        logger.info(
            f"Detected {len(boundary_academic_years)} boundary years: {[ay.year for ay in boundary_academic_years]}"
        )

        initial_periods = create_segments_from_boundary_years(
            boundary_academic_years=boundary_academic_years,
            academic_years=tuple(academic_years),
            algorithm_config=algorithm_config,
            verbose=verbose,
        )

        logger.info(f"Created {len(initial_periods)} initial periods")

        # Step C.5: Beam search refinement (if enabled)
        refined_periods = beam_search_refinement(
            initial_periods=initial_periods,
            academic_years=tuple(academic_years),
            algorithm_config=algorithm_config,
            verbose=verbose,
        )

        logger.info(f"Refined to {len(refined_periods)} periods after beam search")

        if segmentation_only:
            # For segmentation-only mode, return results without characterization and merging
            boundary_years = extract_boundary_years_from_periods(refined_periods)
            timeline_result = TimelineAnalysisResult(
                domain_name=domain_name,
                periods=tuple(refined_periods),
                confidence=calculate_segmentation_confidence(refined_periods),
                boundary_years=tuple(boundary_years),
                narrative_evolution=generate_segmentation_narrative(
                    refined_periods, domain_name
                ),
            )

            total_time = time.time() - start_time
            logger.info(f"Segmentation completed in {total_time:.2f} seconds")
            return timeline_result

        # Step D: Full pipeline: characterization only (merging now handled by beam search)
        final_periods = characterize_academic_periods(
            domain_name=domain_name,
            periods=refined_periods,
            verbose=verbose,
        )

        logger.info(f"Final timeline: {len(final_periods)} periods")

        boundary_years = extract_boundary_years_from_periods(final_periods)
        timeline_result = TimelineAnalysisResult(
            domain_name=domain_name,
            periods=tuple(final_periods),
            confidence=calculate_timeline_confidence(final_periods),
            boundary_years=tuple(boundary_years),
            narrative_evolution=generate_narrative_evolution(
                final_periods, domain_name
            ),
        )

        total_time = time.time() - start_time
        logger.info(f"Timeline analysis completed in {total_time:.2f} seconds")

        return timeline_result

    except Exception as e:
        logger.error(f"Timeline analysis failed: {e}")
        raise RuntimeError(f"Timeline analysis failed for {domain_name}: {e}") from e


def extract_boundary_years_from_periods(periods: List[AcademicPeriod]) -> List[int]:
    """Extract boundary years from actual periods.

    This function extracts the transition points between periods, which are the
    years where one period ends and the next begins. This gives the true boundaries
    between periods rather than just the detected change points.

    Args:
        periods: List of academic periods

    Returns:
        List of boundary years representing transitions between periods
    """
    if not periods:
        return []

    # Sort periods by start year to ensure correct ordering
    sorted_periods = sorted(periods, key=lambda p: p.start_year)

    boundary_years = []

    # Add transition points between periods
    for i in range(len(sorted_periods) - 1):
        current_period = sorted_periods[i]
        next_period = sorted_periods[i + 1]

        # The boundary is the year where one period ends and the next begins
        # If there's a gap, we use the start of the next period
        # If they're consecutive, we use the start of the next period
        boundary_year = next_period.start_year
        boundary_years.append(boundary_year)

    return boundary_years


def calculate_segmentation_confidence(periods: List[AcademicPeriod]) -> float:
    """Calculate confidence for segmentation-only results based on period sizes.

    Args:
        periods: List of segmented academic periods (uncharacterized)

    Returns:
        Confidence score (0.0 to 1.0) based on period size distribution
    """
    if not periods:
        return 0.0

    # For segmentation-only, base confidence on period size distribution
    # More balanced period sizes indicate better segmentation
    total_papers = sum(p.total_papers for p in periods)
    if total_papers == 0:
        return 0.0

    # Calculate entropy of period size distribution
    import math

    proportions = [p.total_papers / total_papers for p in periods]
    entropy = -sum(p * math.log(p) if p > 0 else 0 for p in proportions)
    max_entropy = math.log(len(periods)) if len(periods) > 1 else 1

    # Normalize entropy to [0, 1] where higher entropy means better balance
    balanced_score = entropy / max_entropy if max_entropy > 0 else 0

    # Base confidence of 0.7 adjusted by balance
    return min(0.7 + 0.3 * balanced_score, 1.0)


def generate_segmentation_narrative(
    periods: List[AcademicPeriod], domain_name: str
) -> str:
    """Generate narrative description for segmentation-only results.

    Args:
        periods: List of segmented academic periods (uncharacterized)
        domain_name: Name of the domain

    Returns:
        Narrative description string
    """
    if not periods:
        return f"No timeline periods identified for {domain_name}"

    if len(periods) == 1:
        period = periods[0]
        return f"{domain_name} shows stable development from {period.start_year} to {period.end_year} ({period.total_papers} papers)"

    narrative_parts = []
    narrative_parts.append(f"{domain_name} segmentation results:")

    for i, period in enumerate(periods):
        period_desc = f"Period {i+1} ({period.start_year}-{period.end_year}): {period.total_papers} papers"
        narrative_parts.append(period_desc)

    return "; ".join(narrative_parts)


def calculate_timeline_confidence(periods: List[AcademicPeriod]) -> float:
    """Calculate overall confidence for the timeline based on period characterizations.

    Args:
        periods: List of characterized academic periods

    Returns:
        Overall confidence score (0.0 to 1.0)
    """
    if not periods:
        return 0.0

    total_papers = sum(p.total_papers for p in periods)
    if total_papers == 0:
        return 0.0

    weighted_confidence = 0.0
    for period in periods:
        weight = period.total_papers / total_papers
        weighted_confidence += weight * period.confidence

    return weighted_confidence


def generate_narrative_evolution(
    periods: List[AcademicPeriod], domain_name: str
) -> str:
    """Generate narrative description of the timeline evolution.

    Args:
        periods: List of characterized academic periods
        domain_name: Name of the domain

    Returns:
        Narrative description string
    """
    if not periods:
        return f"No timeline periods identified for {domain_name}"

    if len(periods) == 1:
        period = periods[0]
        return f"{domain_name} shows stable development from {period.start_year} to {period.end_year}"

    narrative_parts = []
    narrative_parts.append(f"{domain_name} timeline evolution:")

    for i, period in enumerate(periods):
        period_desc = f"Period {i+1} ({period.start_year}-{period.end_year})"

        if period.topic_label:
            period_desc += f": {period.topic_label}"

        if period.topic_description:
            period_desc += f" - {period.topic_description}"

        narrative_parts.append(period_desc)

    return "; ".join(narrative_parts)
