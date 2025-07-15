"""Core data processing functions for scientific publication analysis.

This module provides functional interfaces for loading and processing
publication data using AcademicYear and AcademicPeriod as core structures.
"""

import json
import xml.etree.ElementTree as ET
from typing import Dict, Tuple, List
from collections import defaultdict, Counter
from pathlib import Path

from .data_models import (
    Paper,
    CitationRelation,
    AcademicYear,
    AcademicPeriod,
    TimelineAnalysisResult,
)
from ..utils.config import AlgorithmConfig
from ..utils.logging import get_logger


def load_concept_levels(
    concept_levels_file: str = "resources/concept_levels.jsonl",
) -> Dict[str, int]:
    """Load concept levels from JSONL file.

    Args:
        concept_levels_file: Path to concept levels JSONL file

    Returns:
        Dictionary mapping concept names to their levels

    Raises:
        FileNotFoundError: If concept levels file doesn't exist
        ValueError: If file format is invalid
    """
    try:
        concept_levels = {}
        with open(concept_levels_file, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                if line.strip():
                    try:
                        entry = json.loads(line)
                        concept_levels[entry["concept"]] = entry["level"]
                    except (json.JSONDecodeError, KeyError) as e:
                        raise ValueError(f"Invalid format in line {line_num}: {e}")

        return concept_levels

    except FileNotFoundError:
        raise FileNotFoundError(f"Concept levels file not found: {concept_levels_file}")


def filter_keywords_by_concept_level(
    keywords: List[str], domain_name: str, concept_levels: Dict[str, int]
) -> List[str]:
    """Filter keywords to only keep those with level >= domain level.

    Args:
        keywords: List of keywords to filter
        domain_name: Domain name (e.g., "deep_learning" becomes "deep learning")
        concept_levels: Mapping of concept names to levels

    Returns:
        Filtered list of keywords

    Raises:
        ValueError: If domain is not found in concept levels
    """
    domain_concept = domain_name.replace("_", " ")

    if domain_concept not in concept_levels:
        raise ValueError(f"Domain '{domain_concept}' not found in concept levels")

    domain_level = concept_levels[domain_concept]

    filtered_keywords = []
    for keyword in keywords:
        if keyword in concept_levels:
            if concept_levels[keyword] >= domain_level:
                filtered_keywords.append(keyword)
        else:
            filtered_keywords.append(keyword)

    return filtered_keywords


def filter_keywords_by_frequency_ratio(
    keyword_frequencies: Dict[str, int], total_papers: int, min_ratio: float
) -> Dict[str, int]:
    """Filter keywords that appear in at least min_ratio of papers.

    Args:
        keyword_frequencies: Dictionary of keyword -> frequency counts
        total_papers: Total number of papers in the dataset
        min_ratio: Minimum ratio (0.0-1.0) of papers a keyword must appear in

    Returns:
        Filtered keyword frequencies dictionary
    """
    if min_ratio <= 0.0:
        return keyword_frequencies

    min_frequency = max(1, int(total_papers * min_ratio))
    return {
        keyword: freq
        for keyword, freq in keyword_frequencies.items()
        if freq >= min_frequency
    }


def filter_ubiquitous_keywords(
    academic_years: List[AcademicYear],
    ubiquity_threshold: float = 0.9,
    algorithm_config=None,
    verbose: bool = False,
) -> List[AcademicYear]:
    """Filter out keywords that appear as top keywords in most years (de-generalization).

    This removes overly generic keywords that appear consistently across most years,
    reducing noise and improving the discriminative power of remaining keywords.

    Args:
        academic_years: List of AcademicYear objects with initial top keywords
        ubiquity_threshold: Threshold for considering a keyword ubiquitous (0.9 = 90%)
        algorithm_config: Algorithm configuration for top_k_keywords
        verbose: Enable verbose logging

    Returns:
        List of AcademicYear objects with filtered keywords

    Raises:
        ValueError: If academic_years is empty or ubiquity_threshold is invalid
    """
    if not academic_years:
        raise ValueError("academic_years cannot be empty")

    if not (0.0 <= ubiquity_threshold <= 1.0):
        raise ValueError(
            f"ubiquity_threshold must be between 0.0 and 1.0, got {ubiquity_threshold}"
        )

    logger = get_logger(__name__, verbose)

    # Get configuration parameters
    top_k = getattr(algorithm_config, "top_k_keywords", 15) if algorithm_config else 15
    total_years = len(academic_years)
    min_years_for_ubiquity = max(1, int(total_years * ubiquity_threshold))

    if verbose:
        logger.info(
            f"Ubiquitous filter: {total_years} years, threshold={ubiquity_threshold}, min_years={min_years_for_ubiquity}, max_iter={max_iterations}"
        )

    # Create working copy of academic years with mutable keyword data
    filtered_years = []
    for year in academic_years:
        # Create working dictionary with mutable keyword data
        year_data = {
            "year": year.year,
            "papers": year.papers,
            "paper_count": year.paper_count,
            "total_citations": year.total_citations,
            "keyword_frequencies": dict(year.keyword_frequencies),
            "top_keywords": list(year.top_keywords),
        }
        filtered_years.append(year_data)

    iteration = 0
    total_removed = 0
    max_iterations = (
        getattr(algorithm_config, "max_ubiquitous_iterations", 10)
        if algorithm_config
        else 10
    )
    min_replacement_freq = (
        getattr(algorithm_config, "min_replacement_frequency", 2)
        if algorithm_config
        else 2
    )
    removed_keywords_history = set()  # Track previously removed keywords

    while iteration < max_iterations:
        iteration += 1
        if verbose:
            logger.info(f"  iter {iteration}: checking ubiquitous")

        # Count frequency of each keyword across all years' top keywords
        keyword_appearance_count = defaultdict(int)

        for year_data in filtered_years:
            top_keywords_set = set(year_data["top_keywords"])
            for keyword in top_keywords_set:
                keyword_appearance_count[keyword] += 1

        # Find ubiquitous keywords (appear in >= min_years_for_ubiquity years)
        ubiquitous_keywords = {
            keyword
            for keyword, count in keyword_appearance_count.items()
            if count >= min_years_for_ubiquity
        }

        if not ubiquitous_keywords:
            if verbose:
                logger.info(f"  none found, stopping")
            break

        # Check if we're cycling (trying to remove keywords we've already removed)
        if ubiquitous_keywords & removed_keywords_history:
            cycling_keywords = ubiquitous_keywords & removed_keywords_history
            if verbose:
                logger.warning(
                    f"  cycling detected: {sorted(list(cycling_keywords))[:3]}..."
                )
            break

        if verbose:
            keywords_preview = sorted(list(ubiquitous_keywords))[:5]
            logger.info(f"  found {len(ubiquitous_keywords)}: {keywords_preview}...")

        # Remove ubiquitous keywords and replace with next most frequent
        for year_data in filtered_years:
            # Remove ubiquitous keywords from top_keywords
            original_top_keywords = year_data["top_keywords"][:]
            year_data["top_keywords"] = [
                kw for kw in year_data["top_keywords"] if kw not in ubiquitous_keywords
            ]

            removed_count = len(original_top_keywords) - len(year_data["top_keywords"])

            if removed_count > 0:
                # Find replacement keywords from the full frequency distribution
                # Sort all keywords by frequency (descending)
                sorted_keywords = sorted(
                    year_data["keyword_frequencies"].items(),
                    key=lambda x: x[1],
                    reverse=True,
                )

                # Find keywords not already in top_keywords and not previously removed
                current_top_set = set(year_data["top_keywords"])
                replacement_candidates = [
                    keyword
                    for keyword, freq in sorted_keywords
                    if keyword not in current_top_set
                    and keyword not in removed_keywords_history
                    and freq >= min_replacement_freq
                ]

                # Add replacements up to the original top_k size
                replacements_needed = min(removed_count, len(replacement_candidates))
                if replacements_needed > 0:
                    year_data["top_keywords"].extend(
                        replacement_candidates[:replacements_needed]
                    )

                # Ensure we don't exceed top_k limit
                year_data["top_keywords"] = year_data["top_keywords"][:top_k]

                if verbose and removed_count > 0:
                    logger.info(
                        f"    Year {year_data['year']}: Removed {removed_count} ubiquitous keywords, added {replacements_needed} replacements"
                    )

        # Track removed keywords to detect cycles
        removed_keywords_history.update(ubiquitous_keywords)
        total_removed += len(ubiquitous_keywords)

        if verbose:
            logger.info(
                f"  Iteration {iteration} completed. Removed {len(ubiquitous_keywords)} ubiquitous keywords."
            )

    # Convert back to immutable AcademicYear objects
    final_years = []
    for year_data in filtered_years:
        final_year = AcademicYear(
            year=year_data["year"],
            papers=year_data["papers"],
            paper_count=year_data["paper_count"],
            total_citations=year_data["total_citations"],
            keyword_frequencies=year_data["keyword_frequencies"],
            top_keywords=tuple(year_data["top_keywords"]),  # Convert back to immutable
        )
        final_years.append(final_year)

    if verbose:
        status = f"max_iter_reached" if iteration >= max_iterations else "complete"
        logger.info(
            f"Ubiquitous filter: {iteration} iterations, {total_removed} removed, {status}"
        )

    return final_years


def load_domain_data(
    domain_name: str,
    algorithm_config,
    data_directory: str = "resources",
    min_papers_per_year: int = 5,
    apply_year_filtering: bool = True,
    verbose: bool = False,
) -> Tuple[bool, List[AcademicYear], str]:
    """Load domain data and return academic years directly.

    Args:
        domain_name: Name of the domain to load
        algorithm_config: Algorithm configuration for keyword processing
        data_directory: Directory containing domain data files
        min_papers_per_year: Minimum papers required per year
        apply_year_filtering: Whether to filter years with insufficient papers
        verbose: Enable verbose logging

    Returns:
        Tuple of (success, academic_years, error_message)
    """
    logger = get_logger(__name__, verbose, domain_name)

    try:
        papers_file = f"{data_directory}/{domain_name}/{domain_name}_docs_info.json"
        papers = load_papers_from_json(papers_file)

        if not papers:
            return False, [], f"No papers found in {papers_file}"

        graph_file = f"{data_directory}/{domain_name}/{domain_name}_entity_relation_graph.graphml.xml"
        paper_year_map = {p.id: p.pub_year for p in papers}

        try:
            citations, graph_nodes = load_citation_graph(
                graph_file, paper_year_map, verbose
            )
        except Exception as e:
            logger.warning(f"Failed to load citation graph: {e}")
            citations, graph_nodes = tuple(), tuple()

        academic_years = compute_academic_years(papers, algorithm_config, domain_name)

        if verbose:
            year_range = (
                min(ay.year for ay in academic_years),
                max(ay.year for ay in academic_years),
            )
            total_papers = sum(ay.paper_count for ay in academic_years)
            total_citations = sum(ay.total_citations for ay in academic_years)
            logger.info(
                f"Data loaded: {domain_name}, {total_papers:,}p, {total_citations:,}c, {len(citations)} graph citations, {len(graph_nodes)} nodes"
            )

        logger.info(
            f"Loaded: {domain_name}, {len(papers)} papers, {len(academic_years)} years"
        )
        return True, list(academic_years), ""

    except FileNotFoundError as e:
        error_msg = f"Data file not found for {domain_name}: {e}"
        logger.error(error_msg)
        return False, [], error_msg

    except Exception as e:
        error_msg = f"Failed to load {domain_name}: {e}"
        logger.error(error_msg)
        return False, [], error_msg


def create_academic_periods_from_segments(
    academic_years: Tuple[AcademicYear, ...],
    segments: List[Tuple[int, int]],
    algorithm_config=None,
) -> List[AcademicPeriod]:
    """Create AcademicPeriod objects from segments and academic years.

    Args:
        academic_years: Pre-computed academic year structures
        segments: List of (start_year, end_year) tuples

    Returns:
        List of AcademicPeriod objects

    Raises:
        ValueError: If segments are invalid or no academic years are found in the range
    """
    if not academic_years:
        raise ValueError("academic_years cannot be empty")

    if not segments:
        raise ValueError("segments cannot be empty")

    year_lookup = {year.year: year for year in academic_years}
    available_years = set(year_lookup.keys())

    academic_periods = []

    for start_year, end_year in segments:
        if start_year > end_year:
            raise ValueError(
                f"Invalid segment: start_year {start_year} > end_year {end_year}"
            )

        segment_years = [
            year for year in range(start_year, end_year + 1) if year in available_years
        ]

        if not segment_years:
            # Log warning instead of raising error - this can happen during optimization
            import warnings

            warnings.warn(
                f"No academic years found for segment {start_year}-{end_year}. "
                f"Available years: {sorted(available_years)}"
            )
            continue

        segment_academic_years = tuple(year_lookup[year] for year in segment_years)

        total_papers = sum(year.paper_count for year in segment_academic_years)
        total_citations = sum(year.total_citations for year in segment_academic_years)

        combined_keywords = defaultdict(int)
        for year in segment_academic_years:
            for keyword, freq in year.keyword_frequencies.items():
                combined_keywords[keyword] += freq

        # Apply frequency ratio filtering if available in config
        min_ratio = (
            getattr(algorithm_config, "min_keyword_frequency_ratio", 0.1)
            if algorithm_config
            else 0.1
        )
        filtered_keywords = filter_keywords_by_frequency_ratio(
            dict(combined_keywords), total_papers, min_ratio
        )

        top_keywords = tuple(
            keyword for keyword, freq in Counter(filtered_keywords).most_common(50)
        )

        period = AcademicPeriod(
            start_year=start_year,
            end_year=end_year,
            academic_years=segment_academic_years,
            total_papers=total_papers,
            total_citations=total_citations,
            combined_keyword_frequencies=dict(filtered_keywords),
            top_keywords=top_keywords,
        )

        academic_periods.append(period)

    return academic_periods


def create_single_academic_period(
    academic_years: Tuple[AcademicYear, ...],
    start_year: int,
    end_year: int,
    algorithm_config=None,
) -> AcademicPeriod:
    """Create a single AcademicPeriod from academic years within a range.

    Args:
        academic_years: Pre-computed academic year structures
        start_year: Start year of the period
        end_year: End year of the period
        algorithm_config: Algorithm configuration for keyword filtering

    Returns:
        Single AcademicPeriod object
    """
    return create_academic_periods_from_segments(
        academic_years, [(start_year, end_year)], algorithm_config
    )[0]


def load_papers_from_json(file_path: str) -> Tuple[Paper, ...]:
    """Load papers from JSON file and convert to immutable Paper objects.

    Args:
        file_path: Path to JSON file containing paper data

    Returns:
        Tuple of Paper objects

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If data format is invalid
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        papers = []
        for paper_id, paper_data in data.items():
            description = paper_data.get("description", paper_data.get("content", ""))
            content = paper_data.get("content", "")

            paper = Paper(
                id=paper_id,
                title=paper_data["title"],
                content=content,
                pub_year=paper_data["pub_year"],
                cited_by_count=paper_data["cited_by_count"],
                keywords=tuple(paper_data["keywords"]),
                children=tuple(paper_data["children"]),
                description=description,
            )
            papers.append(paper)

        return tuple(papers)

    except FileNotFoundError:
        raise FileNotFoundError(f"Data file not found: {file_path}")
    except (KeyError, TypeError) as e:
        raise ValueError(f"Invalid data format in {file_path}: {e}")


def load_citation_graph(
    file_path: str, paper_year_map: Dict[str, int], verbose: bool = False
) -> Tuple[Tuple[CitationRelation, ...], Tuple[Tuple[str, str], ...]]:
    """Load rich citation graph from .graphml.xml file.

    Args:
        file_path: Path to .graphml.xml file
        paper_year_map: Map of paper_id to publication year
        verbose: Enable verbose logging

    Returns:
        Tuple of (citations, graph_nodes)
    """
    logger = get_logger(__name__, verbose)
    try:
        tree = ET.parse(file_path)
        root = tree.getroot()

        ns = {"graphml": "http://graphml.graphdrawing.org/xmlns"}

        graph_nodes = []
        nodes = root.findall(".//graphml:node", ns)
        for node in nodes:
            node_id = node.get("id")
            description = ""

            for data in node.findall("graphml:data", ns):
                if data.get("key") == "d1":
                    description = data.text or ""
                    break

            graph_nodes.append((node_id, description))

        citations = []
        invalid_count = 0

        edges = root.findall(".//graphml:edge", ns)
        for edge in edges:
            source = edge.get("source")
            target = edge.get("target")

            relation_desc = ""
            semantic_desc = ""
            common_topics = 0
            edge_index = ""

            for data in edge.findall("graphml:data", ns):
                key = data.get("key")
                text = data.text or ""

                if key == "d3":
                    relation_desc = text
                elif key == "d4":
                    semantic_desc = text
                elif key == "d5":
                    try:
                        common_topics = int(text)
                    except ValueError:
                        common_topics = 0
                elif key == "d6":
                    edge_index = text

            if source in paper_year_map and target in paper_year_map:
                citing_year = paper_year_map[source]
                cited_year = paper_year_map[target]

                if citing_year >= cited_year:
                    citation = CitationRelation(
                        citing_paper_id=source,
                        cited_paper_id=target,
                        citing_year=citing_year,
                        cited_year=cited_year,
                        relation_description=relation_desc,
                        semantic_description=semantic_desc,
                        common_topics_count=common_topics,
                        edge_index=edge_index,
                    )
                    citations.append(citation)
                else:
                    invalid_count += 1

        logger.info(
            f"Citations: {len(citations)} loaded, {invalid_count} filtered, {len(graph_nodes)} nodes"
        )

        return tuple(citations), tuple(graph_nodes)

    except Exception as e:
        raise RuntimeError(
            f"Failed to load citation graph from {file_path}: {e}"
        ) from e


def filter_papers_by_year_range(
    papers: Tuple[Paper, ...], start_year: int, end_year: int
) -> Tuple[Paper, ...]:
    """Filter papers to include only those within specified year range.

    Args:
        papers: Input papers
        start_year: Start year (inclusive)
        end_year: End year (inclusive)

    Returns:
        Filtered papers tuple
    """
    filtered = []
    for paper in papers:
        if start_year <= paper.pub_year <= end_year:
            filtered.append(paper)
    return tuple(filtered)


def compute_academic_years(
    papers: Tuple[Paper, ...], algorithm_config, domain_name: str
) -> Tuple[AcademicYear, ...]:
    """Compute AcademicYear objects from papers with concept level filtering.

    Args:
        papers: Tuple of Paper objects
        algorithm_config: Algorithm configuration for keyword processing
        domain_name: Domain name for concept level filtering

    Returns:
        Tuple of AcademicYear objects
    """
    if not papers:
        return tuple()

    try:
        concept_levels = load_concept_levels()
    except (FileNotFoundError, ValueError) as e:
        logger = get_logger(__name__, verbose=False)
        logger.warning(
            f"Could not load concept levels, proceeding without filtering: {e}"
        )
        concept_levels = {}

    papers_by_year = defaultdict(list)
    for paper in papers:
        papers_by_year[paper.pub_year].append(paper)

    academic_years = []

    for year, year_papers in papers_by_year.items():
        year_papers = tuple(year_papers)

        paper_count = len(year_papers)
        total_citations = sum(p.cited_by_count for p in year_papers)

        all_keywords = []
        for paper in year_papers:
            all_keywords.extend(paper.keywords)

        if concept_levels:
            try:
                all_keywords = filter_keywords_by_concept_level(
                    all_keywords, domain_name, concept_levels
                )
            except ValueError as e:
                logger = get_logger(__name__, verbose=False)
                logger.warning(
                    f"Domain not found in concept levels, proceeding without filtering: {e}"
                )

        keyword_frequencies = dict(Counter(all_keywords))

        # Apply frequency ratio filtering if configured
        min_ratio = getattr(algorithm_config, "min_keyword_frequency_ratio", 0.0)
        if min_ratio > 0.0:
            keyword_frequencies = filter_keywords_by_frequency_ratio(
                keyword_frequencies, paper_count, min_ratio
            )

        top_k = getattr(algorithm_config, "top_k_keywords", 15)
        # remove domain itself from top keywords
        if domain_name.replace("_", " ") in keyword_frequencies:
            del keyword_frequencies[domain_name.replace("_", " ")]
        top_keywords = tuple(
            keyword for keyword, freq in Counter(keyword_frequencies).most_common(top_k)
        )

        academic_year = AcademicYear(
            year=year,
            papers=year_papers,
            paper_count=paper_count,
            total_citations=total_citations,
            keyword_frequencies=keyword_frequencies,
            top_keywords=top_keywords,
        )

        academic_years.append(academic_year)

    academic_years.sort(key=lambda ay: ay.year)

    # Apply ubiquitous keyword filtering if enabled
    ubiquity_threshold = getattr(algorithm_config, "ubiquity_threshold", 0.9)
    apply_ubiquitous_filtering = getattr(
        algorithm_config, "apply_ubiquitous_filtering", True
    )

    if apply_ubiquitous_filtering and len(academic_years) > 1:
        logger = get_logger(__name__, verbose=False)
        logger.info(
            f"Applying ubiquitous keyword filtering with threshold {ubiquity_threshold}"
        )
        academic_years = filter_ubiquitous_keywords(
            academic_years=academic_years,
            ubiquity_threshold=ubiquity_threshold,
            algorithm_config=algorithm_config,
            verbose=False,
        )

    return tuple(academic_years)


def load_timeline_from_file(
    timeline_file: str,
    algorithm_config: AlgorithmConfig,
    data_directory: str = "resources",
    verbose: bool = False,
) -> TimelineAnalysisResult:
    """Load timeline result from JSON file and reconstruct with original domain data.

    Args:
        timeline_file: Path to timeline JSON file
        algorithm_config: Algorithm configuration (will be overridden by saved config if available)
        data_directory: Directory containing domain data files
        verbose: Enable verbose logging

    Returns:
        TimelineAnalysisResult object reconstructed with original domain data

    Raises:
        FileNotFoundError: If timeline file doesn't exist
        ValueError: If file format is invalid
    """
    logger = get_logger(__name__, verbose)

    timeline_path = Path(timeline_file)
    if not timeline_path.exists():
        raise FileNotFoundError(f"Timeline file not found: {timeline_file}")

    if verbose:
        logger.info(f"Loading timeline from file: {timeline_file}")

    try:
        with open(timeline_path, "r") as f:
            data = json.load(f)

        # Validate required fields
        required_fields = [
            "domain_name",
            "confidence",
            "boundary_years",
            "narrative_evolution",
            "periods",
        ]
        for field in required_fields:
            if field not in data:
                raise ValueError(f"Missing required field '{field}' in timeline file")

        domain_name = data["domain_name"]

        # Load saved algorithm config if available (for fair comparison)
        if "algorithm_config" in data:
            saved_config = data["algorithm_config"]
            if verbose:
                logger.info(f"Loading saved algorithm configuration from timeline file")
                logger.info(f"Saved config: {saved_config}")

            # Create new algorithm config from saved parameters
            import dataclasses

            config_dict = dataclasses.asdict(algorithm_config)
            config_dict.update(saved_config)
            algorithm_config = AlgorithmConfig(**config_dict)

            logger.info(f"Using saved algorithm configuration for fair comparison")
        else:
            logger.warning(
                f"No saved algorithm configuration found in timeline file. Using provided config."
            )

        # Load original domain data to get individual papers
        if verbose:
            logger.info(f"Loading original domain data for {domain_name}")

        success, academic_years, error_message = load_domain_data(
            domain_name=domain_name,
            algorithm_config=algorithm_config,
            data_directory=data_directory,
            verbose=verbose,
        )

        if not success:
            raise ValueError(f"Failed to load domain data: {error_message}")

        # Extract segments from timeline data
        segments = []
        for period_data in data["periods"]:
            start_year = period_data["start_year"]
            end_year = period_data["end_year"]
            segments.append((start_year, end_year))

        if verbose:
            logger.info(f"Reconstructing {len(segments)} periods from original data")

        # Recreate AcademicPeriod objects using original domain data
        academic_periods = create_academic_periods_from_segments(
            academic_years=tuple(academic_years),
            segments=segments,
            algorithm_config=algorithm_config,
        )

        # Update the periods with any additional information from the timeline file
        enriched_periods = []

        for i, period in enumerate(academic_periods):
            if i < len(data["periods"]):
                period_data = data["periods"][i]

                # Create a new period with the same data but additional fields from timeline
                enriched_period = AcademicPeriod(
                    start_year=period.start_year,
                    end_year=period.end_year,
                    academic_years=period.academic_years,
                    total_papers=period.total_papers,
                    total_citations=period.total_citations,
                    combined_keyword_frequencies=period.combined_keyword_frequencies,
                    top_keywords=period.top_keywords,
                    topic_label=period_data.get("topic_label"),
                    topic_description=period_data.get("topic_description"),
                    network_stability=period_data.get("network_stability", 0.0),
                    community_persistence=period_data.get("community_persistence", 0.0),
                    flow_stability=period_data.get("flow_stability", 0.0),
                    centrality_consensus=period_data.get("centrality_consensus", 0.0),
                    representative_papers=tuple(),  # Not saved in timeline files
                    network_metrics=period_data.get("network_metrics", {}),
                    confidence=period_data.get("confidence", 0.0),
                )
                enriched_periods.append(enriched_period)
            else:
                enriched_periods.append(period)

        # Reconstruct TimelineAnalysisResult
        timeline_result = TimelineAnalysisResult(
            domain_name=domain_name,
            periods=tuple(enriched_periods),
            confidence=data["confidence"],
            boundary_years=tuple(data["boundary_years"]),
            narrative_evolution=data["narrative_evolution"],
            algorithm_config=algorithm_config,
        )

        if verbose:
            logger.info(f"Successfully loaded timeline for {domain_name}")
            logger.info(f"Timeline has {len(timeline_result.periods)} periods")
            logger.info(f"Boundary years: {list(timeline_result.boundary_years)}")

        return timeline_result

    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in timeline file: {e}")
    except Exception as e:
        raise ValueError(f"Error loading timeline file: {e}")
