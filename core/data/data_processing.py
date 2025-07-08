"""Core data processing functions for scientific publication analysis.

This module provides functional interfaces for loading and processing
publication data using AcademicYear and AcademicPeriod as core structures.
"""

import json
import xml.etree.ElementTree as ET
from typing import Dict, Tuple, List
from collections import defaultdict, Counter

from .data_models import (
    Paper,
    CitationRelation,
    AcademicYear,
    AcademicPeriod,
)
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

        if apply_year_filtering:
            papers = filter_papers_by_minimum_yearly_count(
                papers, min_papers_per_year, verbose
            )
            if not papers:
                return (
                    False,
                    [],
                    f"No papers remaining after filtering (min {min_papers_per_year} per year)",
                )

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
            logger.info(f"=== DATA LOADING COMPLETED ===")
            logger.info(f"  Domain: {domain_name}")
            logger.info(
                f"  Year range: {year_range[0]}-{year_range[1]} ({len(academic_years)} years)"
            )
            logger.info(f"  Total papers: {total_papers:,}")
            logger.info(f"  Total citations: {total_citations:,}")
            logger.info(
                f"  Average papers per year: {total_papers / len(academic_years):.1f}"
            )
            logger.info(
                f"  Average citations per year: {total_citations / len(academic_years):.1f}"
            )
            logger.info(
                f"  Citation graph: {len(citations)} citations, {len(graph_nodes)} nodes"
            )

        logger.info(
            f"Successfully loaded {domain_name}: {len(papers)} papers, {len(academic_years)} years"
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
    academic_years: Tuple[AcademicYear, ...], segments: List[Tuple[int, int]]
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
            raise ValueError(
                f"No academic years found for segment {start_year}-{end_year}"
            )

        segment_academic_years = tuple(year_lookup[year] for year in segment_years)

        total_papers = sum(year.paper_count for year in segment_academic_years)
        total_citations = sum(year.total_citations for year in segment_academic_years)

        combined_keywords = defaultdict(int)
        for year in segment_academic_years:
            for keyword, freq in year.keyword_frequencies.items():
                combined_keywords[keyword] += freq

        top_keywords = tuple(
            keyword for keyword, freq in Counter(combined_keywords).most_common(50)
        )

        period = AcademicPeriod(
            start_year=start_year,
            end_year=end_year,
            academic_years=segment_academic_years,
            total_papers=total_papers,
            total_citations=total_citations,
            combined_keyword_frequencies=dict(combined_keywords),
            top_keywords=top_keywords,
        )

        academic_periods.append(period)

    return academic_periods


def create_single_academic_period(
    academic_years: Tuple[AcademicYear, ...], start_year: int, end_year: int
) -> AcademicPeriod:
    """Create a single AcademicPeriod from academic years within a range.

    Args:
        academic_years: Pre-computed academic year structures
        start_year: Start year of the period
        end_year: End year of the period

    Returns:
        Single AcademicPeriod object
    """
    return create_academic_periods_from_segments(
        academic_years, [(start_year, end_year)]
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

        if invalid_count > 0:
            logger.info(
                f"Filtered out {invalid_count} temporally invalid citations from graph"
            )
        logger.info(
            f"Loaded {len(citations)} rich citations and {len(graph_nodes)} graph nodes"
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


def filter_papers_by_minimum_yearly_count(
    papers: Tuple[Paper, ...], min_papers_per_year: int = 5, verbose: bool = False
) -> Tuple[Paper, ...]:
    """Filter papers to exclude years with insufficient paper counts.

    Args:
        papers: Input papers
        min_papers_per_year: Minimum papers required per year
        verbose: Enable verbose logging

    Returns:
        Filtered papers tuple
    """
    logger = get_logger(__name__, verbose)

    year_counts = Counter(p.pub_year for p in papers)

    valid_years = {
        year for year, count in year_counts.items() if count >= min_papers_per_year
    }

    if verbose:
        invalid_years = set(year_counts.keys()) - valid_years
        if invalid_years:
            logger.info(
                f"Filtering out {len(invalid_years)} years with <{min_papers_per_year} papers: {sorted(invalid_years)}"
            )

    filtered_papers = [p for p in papers if p.pub_year in valid_years]

    logger.info(
        f"Filtered papers: {len(papers)} → {len(filtered_papers)} papers, {len(valid_years)} years"
    )

    return tuple(filtered_papers)


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

        top_k = getattr(algorithm_config, "top_k_keywords", 15)
        top_keywords = tuple(
            keyword for keyword, freq in Counter(all_keywords).most_common(top_k)
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

    return tuple(academic_years)
