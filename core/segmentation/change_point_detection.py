"""Change point detection orchestration for research timeline modeling.

This module coordinates citation acceleration and direction change detection
to identify paradigm shifts in academic domains.
"""

from typing import List, Dict, Optional

from ..data.data_models import AcademicYear
from ..utils.logging import get_logger
from .citation_detection import detect_citation_acceleration_years
from .direction_detection import detect_direction_change_years_with_citation_boost


def detect_boundary_years(
    academic_years: List[AcademicYear],
    domain_name: str,
    algorithm_config,
    use_citation: bool = True,
    use_direction: bool = True,
    precomputed_signals: Optional[Dict[str, List]] = None,
    verbose: bool = False,
) -> List[AcademicYear]:
    """Detect paradigm shift boundary years using AcademicYear objects.

    Args:
        academic_years: List of AcademicYear objects with papers and temporal data
        domain_name: Name of the domain for context
        algorithm_config: Algorithm configuration
        use_citation: Whether to use citation validation
        use_direction: Whether to use direction change detection
        precomputed_signals: Optional precomputed signals for efficiency
        verbose: Enable verbose logging

    Returns:
        List of AcademicYear objects where paradigm shifts occur
    """
    logger = get_logger(__name__, verbose)

    # Validate input
    if not academic_years:
        if verbose:
            logger.warning(f"No academic years provided for {domain_name}")
        return []

    if verbose:
        logger.info("=== SHIFT DETECTION STARTED ===")
        logger.info(f"  Domain: {domain_name}")
        logger.info(f"  Academic years: {len(academic_years)}")
        logger.info(
            f"  Year range: {min(ay.year for ay in academic_years)}-{max(ay.year for ay in academic_years)}"
        )
        logger.info(f"  Use direction detection: {use_direction}")
        logger.info(f"  Use citation detection: {use_citation}")

    if precomputed_signals:
        citation_years = precomputed_signals.get("citation_years", [])
        boundary_years = precomputed_signals.get("boundary_years", [])
        if verbose:
            logger.info(f"Using precomputed signals: {len(boundary_years)} boundaries")
    else:
        # Step 1: Compute citation acceleration years first
        citation_years = []
        if use_citation:
            if verbose:
                logger.info("  Step 1: Computing citation acceleration years...")
            citation_years = detect_citation_acceleration_years(academic_years, verbose)
            if verbose:
                logger.info(f"  Citation acceleration years: {citation_years}")

        # Step 2: Compute direction scores with immediate citation boost
        boundary_years = []
        if use_direction:
            if verbose:
                logger.info(
                    "  Step 2: Computing direction scores with citation boost..."
                )
            boundary_years = detect_direction_change_years_with_citation_boost(
                academic_years, citation_years, algorithm_config, verbose
            )
            if verbose:
                logger.info(f"  Final boundary years: {boundary_years}")

    year_lookup = {
        academic_year.year: academic_year for academic_year in academic_years
    }
    boundary_academic_years = []

    for year in boundary_years:
        if year in year_lookup:
            boundary_academic_years.append(year_lookup[year])
        else:
            logger.warning(f"Boundary year {year} not found in academic years")

    logger.info(
        f"Detected {len(boundary_academic_years)} boundary years: {[ay.year for ay in boundary_academic_years]}"
    )
    return boundary_academic_years
