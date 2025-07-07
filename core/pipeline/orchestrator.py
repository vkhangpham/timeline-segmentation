"""
Simplified Timeline Analysis Orchestrator

This module provides the main timeline analysis pipeline using only AcademicYear
and AcademicPeriod as core data structures. All legacy DomainData dependencies
have been eliminated.

PIPELINE FLOW:
1. Load Data → List[AcademicYear]
2. Detect Shifts → List[AcademicYear] (boundary years)
3. Create Segments → List[AcademicPeriod]
4. Characterize Periods → List[AcademicPeriod] (enhanced with characterization)
5. Merge Periods → List[AcademicPeriod] (final timeline)
6. Return → TimelineAnalysisResult

Key features:
- Single entry point: analyze_timeline()
- Pure functional pipeline stages with consistent data structures
- Fail-fast error handling throughout
- Real data usage with no mock or synthetic data
- Clean separation of concerns between stages
"""

import time
from typing import List, Tuple, Optional, Dict, Any

from ..data.data_models import AcademicYear, AcademicPeriod, TimelineAnalysisResult
from ..data.data_processing import (
    load_domain_data,
    create_academic_periods_from_segments,
)
from ..segmentation.shift_signals import detect_boundary_years
from ..segmentation.boundary import create_segments_from_boundary_years
from ..segment_modeling.period_signals import characterize_academic_periods
from ..segmentation.merging import merge_similar_periods
from ..utils.config import AlgorithmConfig
from ..utils.logging import get_logger


def analyze_timeline(
    domain_name: str,
    algorithm_config: AlgorithmConfig,
    data_directory: str = "resources",
    verbose: bool = False,
) -> TimelineAnalysisResult:
    """
    Main timeline analysis pipeline.

    Uses only AcademicYear and AcademicPeriod as core data structures.
    All intermediate complex objects have been eliminated for cleaner flow.

    Args:
        domain_name: Name of the domain to analyze
        algorithm_config: Algorithm configuration (required)
        data_directory: Directory containing domain data
        verbose: Enable verbose logging

    Returns:
        TimelineAnalysisResult with final timeline periods

    Raises:
        RuntimeError: If any pipeline stage fails (fail-fast behavior)
    """
    logger = get_logger(__name__, verbose)
    start_time = time.time()

    try:
        logger.info(f"Starting timeline analysis for {domain_name}")

        # STAGE 1: Load Data → List[AcademicYear]
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

        # STAGE 2: Detect Shifts → List[AcademicYear] (boundary years)
        # Use AcademicYear objects directly for shift detection
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

        # STAGE 3: Create Segments → List[AcademicPeriod]
        # Now directly creates AcademicPeriod objects from boundary AcademicYear objects
        initial_periods = create_segments_from_boundary_years(
            boundary_academic_years=boundary_academic_years,
            academic_years=tuple(academic_years),
            algorithm_config=algorithm_config,
            verbose=verbose,
        )

        logger.info(f"Created {len(initial_periods)} initial periods")

        # STAGE 4: Characterize Periods → List[AcademicPeriod] (enhanced)
        characterized_periods = characterize_academic_periods(
            domain_name=domain_name,
            periods=initial_periods,
            verbose=verbose,
        )

        logger.info(f"Characterized {len(characterized_periods)} periods")

        # STAGE 5: Merge Periods → List[AcademicPeriod] (final timeline)
        final_periods = merge_similar_periods(
            periods=characterized_periods,
            algorithm_config=algorithm_config,
            verbose=verbose,
        )

        logger.info(f"Final timeline: {len(final_periods)} periods")

        # STAGE 6: Create Results
        # Extract boundary years from boundary_academic_years for results
        boundary_years = [ay.year for ay in boundary_academic_years]
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


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def calculate_timeline_confidence(periods: List[AcademicPeriod]) -> float:
    """
    Calculate overall confidence for the timeline based on period characterizations.

    Args:
        periods: List of characterized academic periods

    Returns:
        Overall confidence score (0.0 to 1.0)
    """
    if not periods:
        return 0.0

    # Calculate weighted average of period confidences
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
    """
    Generate narrative description of the timeline evolution.

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

    # Multi-period narrative
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
