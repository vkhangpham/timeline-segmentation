"""Beam search refinement for timeline segmentation optimization.

This module implements beam search to optimize segment boundaries through
merge and split operations guided by the objective function.
"""

from typing import List, Tuple, Set
from dataclasses import dataclass

from ..data.data_models import AcademicPeriod, AcademicYear
from ..data.data_processing import create_academic_periods_from_segments
from ..optimization.objective_function import compute_objective_function
from ..optimization.penalty import create_penalty_config_from_dict
from ..utils.logging import get_logger


@dataclass(frozen=True)
class SegmentNode:
    """Represents a segment in the beam search state."""

    start_year: int
    end_year: int
    split_count: int = 0

    def __post_init__(self):
        """Validate segment node."""
        if self.start_year > self.end_year:
            raise ValueError(
                f"Invalid segment: start_year {self.start_year} > end_year {self.end_year}"
            )
        if self.split_count < 0:
            raise ValueError(f"Invalid split_count: {self.split_count}")

    def get_effective_length(self, available_years: Set[int]) -> int:
        """Calculate effective length based on available data years."""
        count = 0
        for year in range(self.start_year, self.end_year + 1):
            if year in available_years:
                count += 1
        return count


@dataclass
class BeamSearchState:
    """Represents a complete segmentation state in beam search."""

    segments: List[SegmentNode]
    score: float = 0.0

    def __post_init__(self):
        """Validate and sort segments."""
        if not self.segments:
            raise ValueError("segments cannot be empty")

        # Sort segments by start year
        self.segments = sorted(self.segments, key=lambda s: s.start_year)

        # Validate contiguity
        for i in range(len(self.segments) - 1):
            if self.segments[i].end_year + 1 != self.segments[i + 1].start_year:
                raise ValueError(
                    f"Non-contiguous segments: {self.segments[i].end_year} -> {self.segments[i + 1].start_year}"
                )

    def to_segment_tuples(self) -> List[Tuple[int, int]]:
        """Convert to (start_year, end_year) tuples for period creation."""
        return [(seg.start_year, seg.end_year) for seg in self.segments]

    def get_state_hash(self) -> str:
        """Get hash string for duplicate detection."""
        return "|".join(f"{seg.start_year}-{seg.end_year}" for seg in self.segments)


def evaluate_state(
    state: BeamSearchState,
    academic_years: Tuple[AcademicYear, ...],
    available_years: Set[int],
    algorithm_config,
    verbose: bool = False,
) -> float:
    """Evaluate a beam search state using unified penalty system.

    Args:
        state: Beam search state to evaluate
        academic_years: Available academic year data
        available_years: Set of years with data
        algorithm_config: Algorithm configuration
        verbose: Enable verbose logging

    Returns:
        Total score using unified penalty system
    """
    try:
        # Convert to academic periods
        segments = state.to_segment_tuples()
        academic_periods = create_academic_periods_from_segments(
            academic_years, segments, algorithm_config
        )

        # Create penalty configuration from algorithm config
        # For now, use default penalty config - this could be expanded to use algorithm_config
        penalty_config = create_penalty_config_from_dict({"penalty": {}})

        # Compute objective function score with unified penalty
        result = compute_objective_function(
            academic_periods, algorithm_config, penalty_config=penalty_config, verbose=False
        )

        # Return the final penalized score
        total_score = result.final_score

        if verbose:
            logger = get_logger(__name__, verbose)
            logger.info(
                f"State evaluation: raw={result.raw_score:.3f}, penalty={result.penalty:.3f}, "
                f"final={total_score:.3f}, scaled={result.scaled_score:.3f}"
            )

        return total_score

    except Exception as e:
        if verbose:
            logger = get_logger(__name__, verbose)
            logger.warning(f"Failed to evaluate state: {e}")
        return -1000.0  # Large negative score for failed states


def generate_merge_successors(
    state: BeamSearchState, available_years: Set[int], algorithm_config
) -> List[BeamSearchState]:
    """Generate successor states by merging adjacent segments.

    Args:
        state: Current state
        available_years: Set of years with data
        algorithm_config: Algorithm configuration

    Returns:
        List of successor states from merge operations
    """
    successors = []

    for i in range(len(state.segments) - 1):
        # Create merged segment
        left_seg = state.segments[i]
        right_seg = state.segments[i + 1]

        merged_segment = SegmentNode(
            start_year=left_seg.start_year,
            end_year=right_seg.end_year,
            split_count=max(left_seg.split_count, right_seg.split_count),
        )

        # Check if merged segment exceeds max length
        if (
            merged_segment.get_effective_length(available_years)
            > algorithm_config.max_period_years
        ):
            continue

        # Create new state with merged segment
        new_segments = state.segments[:i] + [merged_segment] + state.segments[i + 2 :]

        try:
            successor = BeamSearchState(segments=new_segments)
            successors.append(successor)
        except ValueError:
            # Skip invalid states
            continue

    return successors


def find_best_split_points(
    segment: SegmentNode,
    available_years: Set[int],
    academic_years: Tuple[AcademicYear, ...],
    algorithm_config,
    max_candidates: int = 3,
) -> List[int]:
    """Find best split points for a segment by evaluating all valid splits.

    Args:
        segment: Segment to split
        available_years: Set of years with data
        academic_years: Available academic year data
        algorithm_config: Algorithm configuration
        max_candidates: Maximum number of split candidates to return

    Returns:
        List of split year candidates ranked by objective function score
    """
    # Get available years in segment
    segment_years = [
        year
        for year in range(segment.start_year, segment.end_year + 1)
        if year in available_years
    ]

    if len(segment_years) < 2 * algorithm_config.min_period_years:
        return []

    # Evaluate all possible split points
    split_scores = []

    for i in range(
        algorithm_config.min_period_years,
        len(segment_years) - algorithm_config.min_period_years,
    ):
        split_year = segment_years[i]

        # Create candidate segments
        left_segment = (segment.start_year, split_year)
        right_segment = (split_year + 1, segment.end_year)

        # Check length constraints
        left_length = sum(1 for y in segment_years if y <= split_year)
        right_length = len(segment_years) - left_length

        if (
            left_length >= algorithm_config.min_period_years
            and right_length >= algorithm_config.min_period_years
            and left_length <= algorithm_config.max_period_years
            and right_length <= algorithm_config.max_period_years
        ):

            try:
                # Create academic periods for this split
                test_segments = [left_segment, right_segment]
                test_periods = create_academic_periods_from_segments(
                    academic_years, test_segments, algorithm_config
                )

                # Evaluate objective function
                result = compute_objective_function(
                    test_periods, algorithm_config, verbose=False
                )
                split_scores.append((split_year, result.final_score))

            except Exception:
                # Skip invalid splits
                continue

    # Sort by score (descending) and return top candidates
    split_scores.sort(key=lambda x: x[1], reverse=True)
    return [split_year for split_year, score in split_scores[:max_candidates]]


def generate_split_successors(
    state: BeamSearchState,
    available_years: Set[int],
    academic_years: Tuple[AcademicYear, ...],
    algorithm_config,
) -> List[BeamSearchState]:
    """Generate successor states by splitting segments.

    Args:
        state: Current state
        available_years: Set of years with data
        academic_years: Available academic year data
        algorithm_config: Algorithm configuration

    Returns:
        List of successor states from split operations
    """
    successors = []

    for i, segment in enumerate(state.segments):
        # Check if segment can be split
        if segment.split_count >= algorithm_config.max_splits_per_segment:
            continue

        # Find split candidates
        split_points = find_best_split_points(
            segment, available_years, academic_years, algorithm_config
        )

        for split_year in split_points:
            # Create left and right segments
            left_segment = SegmentNode(
                start_year=segment.start_year,
                end_year=split_year,
                split_count=segment.split_count + 1,
            )

            right_segment = SegmentNode(
                start_year=split_year + 1,
                end_year=segment.end_year,
                split_count=segment.split_count + 1,
            )

            # Create new state with split segments
            new_segments = (
                state.segments[:i]
                + [left_segment, right_segment]
                + state.segments[i + 1 :]
            )

            try:
                successor = BeamSearchState(segments=new_segments)
                successors.append(successor)
            except ValueError:
                # Skip invalid states
                continue

    return successors


def beam_search_refinement(
    initial_periods: List[AcademicPeriod],
    academic_years: Tuple[AcademicYear, ...],
    algorithm_config,
    verbose: bool = False,
) -> List[AcademicPeriod]:
    """Refine segmentation using beam search optimization.

    Args:
        initial_periods: Initial segmentation from boundary detection
        academic_years: Available academic year data
        algorithm_config: Algorithm configuration
        verbose: Enable verbose logging

    Returns:
        Optimized list of academic periods
    """
    logger = get_logger(__name__, verbose)

    if not algorithm_config.beam_search_enabled:
        if verbose:
            logger.info("Beam search disabled, returning initial periods")
        return initial_periods

    if not initial_periods:
        if verbose:
            logger.info("No initial periods provided")
        return initial_periods

    available_years = set(year.year for year in academic_years)

    # Create initial state
    initial_segments = [
        SegmentNode(period.start_year, period.end_year, 0) for period in initial_periods
    ]

    initial_state = BeamSearchState(segments=initial_segments)
    initial_state.score = evaluate_state(
        initial_state, academic_years, available_years, algorithm_config, verbose
    )

    if verbose:
        logger.info(f"=== BEAM SEARCH REFINEMENT STARTED ===")
        logger.info(f"  Initial segments: {len(initial_segments)}")
        logger.info(f"  Initial score: {initial_state.score:.3f}")
        logger.info(f"  Beam width: {algorithm_config.beam_width}")
        logger.info(
            f"  Max splits per segment: {algorithm_config.max_splits_per_segment}"
        )

    # Initialize beam
    beam = [initial_state]
    best_state = initial_state
    seen_states: Set[str] = {initial_state.get_state_hash()}

    iteration = 0
    max_iterations = 20

    while beam and iteration < max_iterations:
        iteration += 1
        new_beam = []

        if verbose:
            logger.info(f"  Iteration {iteration}: exploring {len(beam)} states")

        # Generate successors for each state in beam
        for state in beam:
            # Generate merge successors
            merge_successors = generate_merge_successors(
                state, available_years, algorithm_config
            )

            # Generate split successors
            split_successors = generate_split_successors(
                state, available_years, academic_years, algorithm_config
            )

            # Evaluate all successors
            for successor in merge_successors + split_successors:
                state_hash = successor.get_state_hash()
                if state_hash in seen_states:
                    continue

                seen_states.add(state_hash)
                successor.score = evaluate_state(
                    successor,
                    academic_years,
                    available_years,
                    algorithm_config,
                    verbose,
                )

                if successor.score > best_state.score:
                    best_state = successor
                    if verbose:
                        logger.info(
                            f"    New best score: {successor.score:.3f} ({len(successor.segments)} segments)"
                        )

                new_beam.append(successor)

        # Keep top beam_width states
        new_beam.sort(key=lambda s: s.score, reverse=True)
        beam = new_beam[: algorithm_config.beam_width]

        if verbose:
            logger.info(
                f"    Beam size: {len(beam)}, best in beam: {beam[0].score:.3f}"
            )

        # Early termination if no improvement
        if not beam or beam[0].score <= best_state.score:
            if verbose:
                logger.info("    No improvement found, terminating")
            break

    if verbose:
        logger.info(f"=== BEAM SEARCH REFINEMENT COMPLETED ===")
        logger.info(f"  Final segments: {len(best_state.segments)}")
        logger.info(f"  Final score: {best_state.score:.3f}")
        logger.info(f"  Improvement: {best_state.score - initial_state.score:.3f}")

    # Convert best state back to academic periods
    final_segments = best_state.to_segment_tuples()
    final_periods = create_academic_periods_from_segments(
        academic_years, final_segments, algorithm_config
    )

    logger.info(
        f"Beam search refinement: {len(initial_periods)} → {len(final_periods)} periods, score improvement: {best_state.score - initial_state.score:.3f}"
    )

    return final_periods
