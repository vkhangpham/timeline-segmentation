"""Result display and formatting for timeline analysis.
Provides functions for structured output formatting and user interfaces."""

from typing import Dict, Any
from ..utils.logging import get_logger

__all__ = [
    "display_analysis_summary",
    "format_timeline_narrative",
    "format_segmentation_details",
    "format_confidence_summary",
    "print_detailed_results",
]


def display_analysis_summary(results: Dict[str, Any], verbose: bool = False) -> None:
    """Display analysis summary from results dictionary.

    Args:
        results: Dictionary containing analysis results from pipeline orchestrator
        verbose: Enable verbose logging
    """
    logger = get_logger(__name__, verbose)

    if not results.get("success", False):
        logger.error(f"Analysis failed: {results.get('error', 'Unknown error')}")
        return

    domain_name = results["domain_name"]

    if "segments" in results:
        _display_segmentation_summary(results, verbose)
    else:
        _display_full_analysis_summary(results, verbose)


def _display_segmentation_summary(
    results: Dict[str, Any], verbose: bool = False
) -> None:
    """Display summary for segmentation-only analysis.

    Args:
        results: Segmentation results dictionary
        verbose: Enable verbose logging
    """
    logger = get_logger(__name__, verbose)

    domain_name = results["domain_name"]
    segments = results["segments"]
    execution_time = results["execution_time"]

    logger.info(f"SEGMENTATION: {domain_name}")
    logger.info(f"Segments: {len(segments)}")

    for i, (start, end) in enumerate(segments):
        year_span = end - start + 1
        logger.info(f"  {i+1}. {start}-{end} ({year_span} years)")

    logger.info(f"Completed in {execution_time:.2f}s")


def _display_full_analysis_summary(
    results: Dict[str, Any], verbose: bool = False
) -> None:
    """Display summary for full timeline analysis.

    Args:
        results: Full analysis results dictionary
        verbose: Enable verbose logging
    """
    logger = get_logger(__name__, verbose)

    domain_name = results["domain_name"]
    timeline_result = results["timeline_result"]
    periods = timeline_result.merged_period_characterizations
    confidence = timeline_result.unified_confidence
    execution_time = results["execution_time"]

    logger.info(f"RESULTS: {domain_name}")
    logger.info("=" * 40)
    logger.info(f"Periods: {len(periods)}, Confidence: {confidence:.3f}")

    logger.info("TIMELINE PERIODS:")
    for i, period in enumerate(periods):
        start, end = period.period
        topic = period.topic_label
        conf = period.confidence
        year_span = end - start + 1
        logger.info(
            f"  {i+1}. {start}-{end}: {topic} (conf: {conf:.3f}, {year_span} years)"
        )

    if hasattr(timeline_result, "merging_result") and timeline_result.merging_result:
        merging_result = timeline_result.merging_result
        original_count = len(timeline_result.period_characterizations)
        merged_count = len(periods)

        if original_count != merged_count:
            logger.info("SEGMENT MERGING:")
            logger.info(f"  Original segments: {original_count}")
            logger.info(f"  Merged segments: {merged_count}")
            logger.info(f"  Reduction: {original_count - merged_count} segments merged")

    logger.info(f"Completed in {execution_time:.2f}s")


def format_timeline_narrative(timeline_result) -> str:
    """Format timeline analysis results into a readable narrative.

    Args:
        timeline_result: TimelineAnalysisResult object

    Returns:
        Formatted narrative string describing the timeline evolution
    """
    if (
        hasattr(timeline_result, "narrative_evolution")
        and timeline_result.narrative_evolution
    ):
        return f"Timeline Evolution: {timeline_result.narrative_evolution}"

    periods = timeline_result.merged_period_characterizations
    narrative_parts = []

    for period in periods:
        start, end = period.period
        topic = period.topic_label
        narrative_parts.append(f"{start}-{end}: {topic}")

    return "Timeline Evolution: " + " → ".join(narrative_parts)


def format_segmentation_details(segmentation_results: Dict[str, Any]) -> str:
    """Format segmentation results into detailed description.

    Args:
        segmentation_results: Segmentation results dictionary

    Returns:
        Formatted string describing segmentation details
    """
    segments = segmentation_results.get("segments", [])
    change_points = segmentation_results.get("change_points", [])
    statistical_significance = segmentation_results.get("statistical_significance", 0)

    details = []
    details.append(
        f"Segmentation Method: {segmentation_results.get('algorithm_config', {}).get('segmentation_method', 'boundary_based')}"
    )
    details.append(f"Total Segments: {len(segments)}")
    details.append(f"Change Points: {len(change_points)} detected")
    details.append(f"Statistical Significance: {statistical_significance:.3f}")

    if segments:
        total_years = sum(end - start + 1 for start, end in segments)
        avg_segment_length = total_years / len(segments)
        details.append(f"Average Segment Length: {avg_segment_length:.1f} years")

    return "\n".join(details)


def format_confidence_summary(timeline_result) -> str:
    """Format confidence metrics into summary.

    Args:
        timeline_result: TimelineAnalysisResult object

    Returns:
        Formatted confidence summary string
    """
    unified_confidence = timeline_result.unified_confidence
    periods = timeline_result.merged_period_characterizations

    if not periods:
        return f"Overall Confidence: {unified_confidence:.3f} (no periods)"

    individual_confidences = [p.confidence for p in periods]
    min_conf = min(individual_confidences)
    max_conf = max(individual_confidences)

    summary = f"Overall Confidence: {unified_confidence:.3f}"
    summary += f" (range: {min_conf:.3f}-{max_conf:.3f})"

    if unified_confidence >= 0.8:
        quality = "Excellent"
    elif unified_confidence >= 0.6:
        quality = "Good"
    elif unified_confidence >= 0.4:
        quality = "Fair"
    else:
        quality = "Poor"

    summary += f" - {quality} quality"

    return summary


def print_detailed_results(results: Dict[str, Any], verbose: bool = False) -> None:
    """Print comprehensive detailed results.

    Args:
        results: Complete analysis results dictionary
        verbose: Enable verbose logging
    """
    logger = get_logger(__name__, verbose)

    if not results.get("success", False):
        logger.error(f"Analysis failed: {results.get('error', 'Unknown error')}")
        return

    domain_name = results["domain_name"]
    logger.info("=" * 60)
    logger.info(f"DETAILED TIMELINE ANALYSIS RESULTS: {domain_name}")
    logger.info("=" * 60)

    if "segmentation_results" in results:
        logger.info("SEGMENTATION DETAILS:")
        logger.info(format_segmentation_details(results["segmentation_results"]))

    if "timeline_result" in results:
        timeline_result = results["timeline_result"]

        logger.info("CONFIDENCE ANALYSIS:")
        logger.info(format_confidence_summary(timeline_result))

        logger.info("TIMELINE NARRATIVE:")
        logger.info(format_timeline_narrative(timeline_result))

        if "saved_files" in results and results["saved_files"]:
            logger.info("SAVED FILES:")
            for file_type, file_path in results["saved_files"].items():
                logger.info(f"  {file_type}: {file_path}")

    execution_time = results.get("execution_time", 0)
    logger.info(f"Total execution time: {execution_time:.2f}s")
    logger.info("=" * 60)
