"""Period characterization using temporal network analysis.

This module provides period characterization using network analysis,
working directly with Paper objects for optimal performance.
"""

import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Dict, Any, Tuple
from collections import Counter
import numpy as np
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer

from .segment_labeling import (
    select_representative_papers,
    generate_period_label_and_description,
)
from ..data.data_models import AcademicPeriod, Paper
from ..utils.logging import get_logger


def characterize_academic_periods(
    domain_name: str, periods: List[AcademicPeriod], verbose: bool = False
) -> List[AcademicPeriod]:
    """Characterize AcademicPeriod objects using temporal network analysis.

    Args:
        domain_name: Name of the research domain
        periods: List of AcademicPeriod objects to characterize
        verbose: Enable verbose logging

    Returns:
        List of characterized AcademicPeriod objects with populated characterization fields
    """
    logger = get_logger(__name__, verbose)

    if verbose:
        logger.info(f"=== NETWORK ANALYSIS CHARACTERIZATION STARTED ===")
        logger.info(f"  Domain: {domain_name}")
        logger.info(f"  Periods to analyze: {len(periods)}")

    all_papers = []
    for period in periods:
        for academic_year in period.academic_years:
            all_papers.extend(academic_year.papers)

    if verbose:
        logger.info(
            f"  Working with {len(all_papers)} papers directly from AcademicPeriod objects"
        )
        logger.info("  Loading semantic citations from GraphML...")

    semantic_citations = load_semantic_citations(domain_name)

    if verbose:
        logger.info(
            f"  Loaded {len(semantic_citations)} semantic citations from GraphML"
        )
        logger.info("  Building citation network...")

    citation_network = build_citation_network_from_papers(
        all_papers, semantic_citations
    )

    if verbose:
        logger.info(
            f"  Citation network: {citation_network.number_of_nodes()} nodes, {citation_network.number_of_edges()} edges"
        )
        logger.info("  Initializing TF-IDF vectorizer...")

    tfidf_vectorizer = TfidfVectorizer(
        max_features=10000, stop_words="english", ngram_range=(1, 3), min_df=2
    )

    characterized_periods = []

    for i, period in enumerate(periods, 1):
        start_year, end_year = period.start_year, period.end_year

        if verbose:
            logger.info(
                f"=== ANALYZING PERIOD {i}/{len(periods)}: {start_year}-{end_year} ==="
            )

        logger.info(
            f"Characterizing period {start_year}-{end_year} with network analysis..."
        )

        if verbose:
            logger.info(f"  Getting papers for period {start_year}-{end_year}...")

        period_papers = get_papers_in_period_with_filtering(
            all_papers, period, start_year, end_year, verbose
        )

        if verbose:
            logger.info(
                f"  Found {len(period_papers)} papers in period (after keyword filtering)"
            )
            significance_scores = calculate_paper_significance_from_papers(period_papers)
            significant_count = sum(1 for p in period_papers if significance_scores.get(p.id, 0.1) >= 0.7)
            logger.info(f"  Significant papers: {significant_count}")

        if len(period_papers) < 3:
            logger.warning(
                f"Insufficient papers ({len(period_papers)}) for network analysis"
            )
            if verbose:
                logger.warning("  Skipping network analysis for this period")
            continue

        if verbose:
            logger.info("  Building period subnetwork...")

        period_subnetwork = build_period_subnetwork_from_papers(
            citation_network, period_papers, start_year, end_year
        )

        if verbose:
            logger.info(
                f"  Period subnetwork: {period_subnetwork.number_of_nodes()} nodes, {period_subnetwork.number_of_edges()} edges"
            )
            logger.info("  Analyzing temporal network stability...")

        network_stability = analyze_network_stability(period_subnetwork)

        if verbose:
            logger.info(f"  Network stability: {network_stability:.3f}")
            logger.info("  Measuring community persistence...")

        community_persistence = measure_community_persistence(period_subnetwork)

        if verbose:
            logger.info(f"  Community persistence: {community_persistence:.3f}")
            logger.info("  Analyzing flow stability...")

        flow_stability = analyze_flow_stability(period_subnetwork)

        if verbose:
            logger.info(f"  Flow stability: {flow_stability:.3f}")
            logger.info("  Calculating centrality consensus...")

        centrality_consensus = calculate_centrality_consensus(period_subnetwork)

        if verbose:
            logger.info(f"  Centrality consensus: {centrality_consensus:.3f}")
            logger.info("  Detecting dominant themes...")

        dominant_themes = detect_network_themes_from_papers(
            period_papers, period_subnetwork, tfidf_vectorizer
        )

        if verbose:
            logger.info(f"  Dominant themes: {dominant_themes}")
            logger.info("  Selecting representative papers...")

        period_papers_dict = []
        for paper in period_papers:
            period_papers_dict.append({
                "id": paper.id,
                "data": {
                    "title": paper.title,
                    "pub_year": paper.pub_year,
                    "cited_by_count": paper.cited_by_count,
                    "keywords": paper.keywords,
                    "description": paper.description,
                    "content": paper.content,
                }
            })

        representative_papers = select_representative_papers(
            period_papers_dict, period_subnetwork, dominant_themes, verbose
        )

        if verbose:
            logger.info(
                f"  Selected {len(representative_papers)} representative papers"
            )
            logger.info("  Calculating comprehensive network metrics...")

        network_metrics = calculate_network_metrics(period_subnetwork)

        if verbose:
            logger.info("  Building period context for LLM...")

        previous_periods = []
        for prev_period in characterized_periods:
            previous_periods.append(
                (
                    prev_period.start_year,
                    prev_period.end_year,
                    prev_period.topic_label,
                    prev_period.topic_description,
                )
            )

        period_label, period_description = generate_period_label_and_description(
            dominant_themes,
            representative_papers,
            start_year,
            end_year,
            previous_periods=previous_periods,
            domain_name=domain_name,
            verbose=verbose,
        )

        confidence = calculate_confidence(
            network_stability,
            community_persistence,
            flow_stability,
            centrality_consensus,
            len(period_papers),
            network_metrics,
        )

        characterized_period = AcademicPeriod(
            start_year=period.start_year,
            end_year=period.end_year,
            academic_years=period.academic_years,
            total_papers=period.total_papers,
            total_citations=period.total_citations,
            combined_keyword_frequencies=period.combined_keyword_frequencies,
            top_keywords=period.top_keywords,
            topic_label=period_label,
            topic_description=period_description,
            network_stability=network_stability,
            community_persistence=community_persistence,
            flow_stability=flow_stability,
            centrality_consensus=centrality_consensus,
            representative_papers=tuple(representative_papers),
            network_metrics=network_metrics,
            confidence=confidence,
        )

        characterized_periods.append(characterized_period)

        if verbose:
            logger.info(f"  Period {i} characterization completed successfully")

    if verbose:
        logger.info("=== NETWORK ANALYSIS CHARACTERIZATION COMPLETED ===")
        logger.info(f"  Total characterized periods: {len(characterized_periods)}")

    logger.info(
        f"Characterized {len(characterized_periods)} periods using direct Paper object analysis"
    )
    return characterized_periods


def load_semantic_citations(domain_name: str) -> List[Dict[str, Any]]:
    """Load semantic citation descriptions from GraphML.

    Args:
        domain_name: Name of the research domain

    Returns:
        List of semantic citation dictionaries
    """
    data_dir = Path(f"resources/{domain_name}")
    citations_file_xml = data_dir / f"{domain_name}_entity_relation_graph.graphml.xml"

    if not citations_file_xml.exists():
        return []

    semantic_citations = []
    try:
        tree = ET.parse(citations_file_xml)
        root = tree.getroot()

        ns = {"": "http://graphml.graphdrawing.org/xmlns"}

        for edge in root.findall(".//edge", ns):
            source = edge.get("source")
            target = edge.get("target")

            description = ""
            for data in edge.findall("data", ns):
                if data.get("key") == "d3" and data.text:
                    description = data.text.strip()
                    break

            if source and target and description:
                semantic_citations.append(
                    {
                        "parent_id": source,
                        "child_id": target,
                        "description": description,
                        "relationship_type": "cites",
                    }
                )

    except Exception as e:
        raise RuntimeError(
            f"Failed to load semantic citations for {domain_name}: {e}"
        ) from e

    return semantic_citations


def build_citation_network_from_papers(
    papers: List["Paper"], semantic_citations: List[Dict[str, Any]]
) -> nx.DiGraph:
    """Build citation network directly from Paper objects.

    Args:
        papers: List of Paper objects
        semantic_citations: List of semantic citation dictionaries

    Returns:
        NetworkX directed graph with papers as nodes and citations as edges
    """
    G = nx.DiGraph()

    significance_scores = calculate_paper_significance_from_papers(papers)

    for paper in papers:
        significance = significance_scores.get(paper.id, 0.1)
        G.add_node(
            paper.id,
            pub_year=paper.pub_year,
            cited_by_count=paper.cited_by_count,
            title=paper.title,
            significance=significance,
            is_significant=significance >= 0.7,
        )

    paper_ids = {paper.id for paper in papers}
    for citation in semantic_citations:
        parent_id = citation["parent_id"]
        child_id = citation["child_id"]
        if parent_id in paper_ids and child_id in paper_ids:
            G.add_edge(
                parent_id,
                child_id,
                description=citation["description"],
                relationship_type=citation["relationship_type"],
            )

    return G


def calculate_paper_significance_from_papers(papers: List["Paper"]) -> Dict[str, float]:
    """Calculate paper significance directly from Paper objects.

    Args:
        papers: List of Paper objects

    Returns:
        Dictionary mapping paper IDs to significance scores (0.1 to 1.0)
    """
    if not papers:
        return {}

    citation_counts = {paper.id: paper.cited_by_count for paper in papers}

    if not citation_counts:
        return {paper.id: 0.1 for paper in papers}

    max_citations = max(citation_counts.values())
    if max_citations == 0:
        return {paper.id: 0.1 for paper in papers}

    significance_scores = {}
    for paper in papers:
        normalized_citations = citation_counts[paper.id] / max_citations
        significance = 0.1 + (0.9 * normalized_citations)
        significance_scores[paper.id] = significance

    return significance_scores


def get_papers_in_period_with_filtering(
    papers: List["Paper"],
    period: AcademicPeriod,
    start_year: int,
    end_year: int,
    verbose: bool = False,
) -> List["Paper"]:
    """Get papers in period with keyword filtering.

    Args:
        papers: List of all Paper objects
        period: AcademicPeriod object with top keywords
        start_year: Start year of the period
        end_year: End year of the period
        verbose: Enable verbose logging

    Returns:
        List of filtered Paper objects in the specified period
    """
    logger = get_logger(__name__, verbose)

    period_top_keywords = set(period.top_keywords)

    if verbose:
        logger.info(
            f"    Filtering papers using {len(period_top_keywords)} period keywords"
        )
        logger.info(f"    Period top keywords: {list(period_top_keywords)[:10]}...")

    significance_scores = calculate_paper_significance_from_papers(papers)

    period_papers = []
    total_papers_in_period = 0
    no_overlap_count = 0

    for paper in papers:
        if start_year <= paper.pub_year <= end_year:
            total_papers_in_period += 1

            paper_keywords_set = set(paper.keywords)
            keyword_overlap = period_top_keywords & paper_keywords_set

            if verbose and total_papers_in_period <= 5:
                logger.info(
                    f"      Paper {total_papers_in_period}: {len(paper.keywords)} keywords - {paper.keywords[:3]}..."
                )
                logger.info(
                    f"        Overlap with period keywords: {list(keyword_overlap)[:3]}..."
                )

            if keyword_overlap:
                period_papers.append(paper)
            else:
                no_overlap_count += 1
                if verbose and no_overlap_count <= 3:
                    logger.warning(
                        f"      No overlap for paper {paper.id}: paper_kw={paper.keywords[:3]}, period_kw={list(period_top_keywords)[:3]}"
                    )

    if verbose:
        logger.info(f"    Total papers in period: {total_papers_in_period}")
        logger.info(f"    Papers with keyword overlap: {len(period_papers)}")
        logger.info(f"    Papers with NO overlap: {no_overlap_count}")
        if total_papers_in_period > 0:
            logger.info(
                f"    Filtering ratio: {len(period_papers)/total_papers_in_period:.2f}"
            )

    MAX_PAPERS = 100
    if len(period_papers) > MAX_PAPERS:
        if verbose:
            logger.info(
                f"    Applying hard limit: {len(period_papers)} -> {MAX_PAPERS} papers"
            )

        period_papers.sort(key=lambda p: significance_scores.get(p.id, 0.1), reverse=True)
        period_papers = period_papers[:MAX_PAPERS]

        if verbose:
            logger.info(f"    Selected top {MAX_PAPERS} papers by significance")

    return period_papers


def build_period_subnetwork_from_papers(
    citation_network: nx.DiGraph,
    period_papers: List["Paper"],
    start_year: int,
    end_year: int,
) -> nx.DiGraph:
    """Build subnetwork for the specific period using Paper objects.

    Args:
        citation_network: Full citation network
        period_papers: List of Paper objects in the period
        start_year: Start year of the period
        end_year: End year of the period

    Returns:
        NetworkX directed graph subnetwork for the period
    """
    paper_ids = {paper.id for paper in period_papers}

    period_subgraph = citation_network.subgraph(paper_ids).copy()

    for node in period_subgraph.nodes():
        node_data = period_subgraph.nodes[node]
        pub_year = node_data.get("pub_year", 0)
        if end_year > start_year:
            temporal_position = (pub_year - start_year) / (end_year - start_year)
        else:
            temporal_position = 0.5
        period_subgraph.nodes[node]["temporal_position"] = temporal_position

    return period_subgraph


def detect_network_themes_from_papers(
    period_papers: List["Paper"],
    subnetwork: nx.DiGraph,
    tfidf_vectorizer: TfidfVectorizer,
) -> List[str]:
    """Detect dominant themes using Paper objects and network structure.

    Args:
        period_papers: List of Paper objects in the period
        subnetwork: Citation network subnetwork
        tfidf_vectorizer: TF-IDF vectorizer for text analysis

    Returns:
        List of dominant theme strings
    """
    if not period_papers:
        return []

    documents = []
    for paper in period_papers:
        text_parts = [paper.title]
        if paper.keywords:
            text_parts.extend(paper.keywords)
        if paper.description:
            text_parts.append(paper.description)

        combined_text = " ".join(text_parts)
        documents.append(combined_text)

    if not documents:
        return []

    try:
        tfidf_matrix = tfidf_vectorizer.fit_transform(documents)
        feature_names = tfidf_vectorizer.get_feature_names_out()

        feature_scores = tfidf_matrix.sum(axis=0).A1
        top_indices = feature_scores.argsort()[-20:][::-1]

        dominant_themes = [
            feature_names[i] for i in top_indices if feature_scores[i] > 0
        ]

        filtered_themes = [
            theme
            for theme in dominant_themes
            if len(theme) > 2
            and theme not in {"the", "and", "for", "with", "from", "this", "that"}
        ]

        return filtered_themes[:10]

    except Exception as e:
        raise RuntimeError(f"Failed to detect network themes: {e}") from e


def analyze_network_stability(subnetwork: nx.DiGraph) -> float:
    """Analyze temporal network stability using multiple metrics.

    Args:
        subnetwork: NetworkX directed graph to analyze

    Returns:
        Stability score between 0.0 and 1.0
    """
    if subnetwork.number_of_nodes() < 3:
        return 0.0

    stability_metrics = []

    try:
        degrees = [d for n, d in subnetwork.degree()]
        if degrees:
            degree_variance = np.var(degrees)
            degree_stability = 1.0 / (1.0 + degree_variance)
            stability_metrics.append(degree_stability)
    except (ValueError, ZeroDivisionError) as e:
        raise RuntimeError(
            f"Failed to calculate degree distribution stability: {e}"
        ) from e

    try:
        clustering = nx.average_clustering(subnetwork.to_undirected())
        stability_metrics.append(clustering)
    except (ValueError, ZeroDivisionError) as e:
        raise RuntimeError(
            f"Failed to calculate clustering coefficient stability: {e}"
        ) from e

    try:
        undirected = subnetwork.to_undirected()
        num_components = nx.number_connected_components(undirected)
        total_nodes = subnetwork.number_of_nodes()
        component_stability = 1.0 - (num_components / total_nodes)
        stability_metrics.append(max(0.0, component_stability))
    except (ValueError, ZeroDivisionError) as e:
        raise RuntimeError(
            f"Failed to calculate connected components stability: {e}"
        ) from e

    return np.mean(stability_metrics) if stability_metrics else 0.0


def measure_community_persistence(subnetwork: nx.DiGraph) -> float:
    """Measure community persistence using community detection.

    Args:
        subnetwork: NetworkX directed graph to analyze

    Returns:
        Community persistence score between 0.0 and 1.0
    """
    if subnetwork.number_of_nodes() < 4:
        return 0.0

    try:
        undirected = subnetwork.to_undirected()

        communities = {}
        for i, component in enumerate(nx.connected_components(undirected)):
            for node in component:
                communities[node] = i

        if not communities:
            return 0.0

        num_communities = len(set(communities.values()))
        total_nodes = len(communities)

        if num_communities == 0:
            return 0.0

        community_ratio = num_communities / total_nodes
        optimal_ratio = 0.3
        ratio_score = 1.0 - abs(community_ratio - optimal_ratio)

        community_sizes = Counter(communities.values())
        size_variance = np.var(list(community_sizes.values()))
        balance_score = 1.0 / (1.0 + size_variance)

        persistence_score = (ratio_score + balance_score) / 2
        return max(0.0, min(1.0, persistence_score))

    except Exception as e:
        raise RuntimeError(f"Failed to measure community persistence: {e}") from e


def analyze_flow_stability(subnetwork: nx.DiGraph) -> float:
    """Analyze citation flow stability within the period.

    Args:
        subnetwork: NetworkX directed graph to analyze

    Returns:
        Flow stability score between 0.0 and 1.0
    """
    if subnetwork.number_of_edges() < 2:
        return 0.0

    flow_metrics = []

    try:
        in_degrees = [d for n, d in subnetwork.in_degree()]
        if in_degrees and len(in_degrees) > 1:
            in_degree_cv = np.std(in_degrees) / (np.mean(in_degrees) + 1e-6)
            in_degree_stability = 1.0 / (1.0 + in_degree_cv)
            flow_metrics.append(in_degree_stability)
    except (ValueError, ZeroDivisionError) as e:
        raise RuntimeError(
            f"Failed to calculate in-degree distribution stability: {e}"
        ) from e

    try:
        density = nx.density(subnetwork)
        optimal_density = 0.1
        density_score = 1.0 - abs(density - optimal_density)
        flow_metrics.append(max(0.0, density_score))
    except (ValueError, ZeroDivisionError) as e:
        raise RuntimeError(f"Failed to calculate network density: {e}") from e

    return np.mean(flow_metrics) if flow_metrics else 0.0


def calculate_centrality_consensus(subnetwork: nx.DiGraph) -> float:
    """Calculate consensus based on centrality measures.

    Args:
        subnetwork: NetworkX directed graph to analyze

    Returns:
        Centrality consensus score between 0.0 and 1.0
    """
    if subnetwork.number_of_nodes() < 3:
        return 0.0

    centrality_metrics = []

    try:
        pagerank = nx.pagerank(subnetwork)
        pr_values = list(pagerank.values())
        if pr_values:
            pr_variance = np.var(pr_values)
            optimal_variance = 0.01
            pr_score = 1.0 - abs(pr_variance - optimal_variance)
            centrality_metrics.append(max(0.0, pr_score))
    except (ValueError, ZeroDivisionError, nx.NetworkXError) as e:
        raise RuntimeError(f"Failed to calculate PageRank centrality: {e}") from e

    try:
        betweenness = nx.betweenness_centrality(subnetwork)
        bc_values = list(betweenness.values())
        if bc_values:
            bc_variance = np.var(bc_values)
            bc_score = 1.0 / (1.0 + bc_variance * 10)
            centrality_metrics.append(bc_score)
    except (ValueError, ZeroDivisionError, nx.NetworkXError) as e:
        raise RuntimeError(f"Failed to calculate betweenness centrality: {e}") from e

    return np.mean(centrality_metrics) if centrality_metrics else 0.0


def calculate_network_metrics(subnetwork: nx.DiGraph) -> Dict[str, float]:
    """Calculate comprehensive network metrics for the period.

    Args:
        subnetwork: NetworkX directed graph to analyze

    Returns:
        Dictionary of network metrics
    """
    metrics = {}

    try:
        metrics["density"] = nx.density(subnetwork)
        metrics["number_of_nodes"] = subnetwork.number_of_nodes()
        metrics["number_of_edges"] = subnetwork.number_of_edges()

        if subnetwork.number_of_nodes() > 0:
            undirected = subnetwork.to_undirected()
            metrics["average_clustering"] = nx.average_clustering(undirected)
            metrics["number_of_components"] = nx.number_connected_components(undirected)

            degrees = [d for n, d in subnetwork.degree()]
            if degrees:
                max_degree = max(degrees)
                sum_diff = sum(max_degree - d for d in degrees)
                n = len(degrees)
                max_sum_diff = (n - 1) * (n - 2)
                metrics["degree_centralization"] = (
                    sum_diff / max_sum_diff if max_sum_diff > 0 else 0.0
                )

    except Exception as e:
        raise RuntimeError(f"Failed to calculate network metrics: {e}") from e

    return metrics


def calculate_confidence(
    network_stability: float,
    community_persistence: float,
    flow_stability: float,
    centrality_consensus: float,
    num_papers: int,
    network_metrics: Dict[str, float],
) -> float:
    """Calculate overall confidence score for period characterization.

    Args:
        network_stability: Network stability metric
        community_persistence: Community persistence metric
        flow_stability: Flow stability metric
        centrality_consensus: Centrality consensus metric
        num_papers: Number of papers in the period
        network_metrics: Dictionary of network metrics

    Returns:
        Overall confidence score between 0.1 and 1.0
    """
    if num_papers < 3:
        return 0.1

    base_confidence = np.mean(
        [network_stability, community_persistence, flow_stability, centrality_consensus]
    )

    paper_factor = min(1.0, num_papers / 50.0)

    density = network_metrics.get("density", 0.0)
    clustering = network_metrics.get("average_clustering", 0.0)
    structure_factor = (density + clustering) / 2.0

    final_confidence = (
        base_confidence * 0.6 + paper_factor * 0.2 + structure_factor * 0.2
    )

    return max(0.1, min(1.0, final_confidence))
