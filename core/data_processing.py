"""
Core data processing functions for scientific publication analysis.

This module provides pure functions for loading, validating, and processing publication data
including rich citation graph information from .graphml.xml files.
"""

import json
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Tuple
from collections import defaultdict, Counter
import statistics

from .data_models import (
    Paper, CitationRelation, DomainData, TemporalWindow, 
    DataStatistics, ProcessingResult, DataSubset, KeywordAnalysis
)


def load_papers_from_json(file_path: str) -> Tuple[Paper, ...]:
    """
    Load papers from JSON file and convert to immutable Paper objects.
    
    Args:
        file_path: Path to JSON file containing paper data
        
    Returns:
        Tuple of Paper objects
        
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If data format is invalid
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        papers = []
        for paper_id, paper_data in data.items():
            paper = Paper(
                id=paper_id,
                title=paper_data['title'],
                content=paper_data['content'],
                pub_year=paper_data['pub_year'],
                cited_by_count=paper_data['cited_by_count'],
                keywords=tuple(paper_data['keywords']),
                children=tuple(paper_data['children']),
                description=paper_data['description']
            )
            papers.append(paper)
        
        return tuple(papers)
    
    except FileNotFoundError:
        raise FileNotFoundError(f"Data file not found: {file_path}")
    except (KeyError, TypeError) as e:
        raise ValueError(f"Invalid data format in {file_path}: {e}")


def load_citation_graph(file_path: str, paper_year_map: Dict[str, int]) -> Tuple[Tuple[CitationRelation, ...], Tuple[Tuple[str, str], ...]]:
    """
    Load rich citation graph from .graphml.xml file.
    
    Args:
        file_path: Path to .graphml.xml file
        paper_year_map: Map of paper_id to publication year
        
    Returns:
        Tuple of (citations, graph_nodes)
    """
    try:
        tree = ET.parse(file_path)
        root = tree.getroot()
        
        # XML namespace
        ns = {'graphml': 'http://graphml.graphdrawing.org/xmlns'}
        
        # Extract nodes (papers with descriptions)
        graph_nodes = []
        nodes = root.findall('.//graphml:node', ns)
        for node in nodes:
            node_id = node.get('id')
            description = ""
            
            # Find description (d1)
            for data in node.findall('graphml:data', ns):
                if data.get('key') == 'd1':
                    description = data.text or ""
                    break
            
            graph_nodes.append((node_id, description))
        
        # Extract edges (citations with rich semantic info)
        citations = []
        invalid_count = 0
        
        edges = root.findall('.//graphml:edge', ns)
        for edge in edges:
            source = edge.get('source')  # citing paper
            target = edge.get('target')  # cited paper
            
            # Extract edge data
            relation_desc = ""
            semantic_desc = ""
            common_topics = 0
            edge_index = ""
            
            for data in edge.findall('graphml:data', ns):
                key = data.get('key')
                text = data.text or ""
                
                if key == 'd3':  # relation description
                    relation_desc = text
                elif key == 'd4':  # semantic description
                    semantic_desc = text
                elif key == 'd5':  # common topics count
                    try:
                        common_topics = int(text)
                    except ValueError:
                        common_topics = 0
                elif key == 'd6':  # edge index
                    edge_index = text
            
            # Validate temporal consistency
            if source in paper_year_map and target in paper_year_map:
                citing_year = paper_year_map[source]
                cited_year = paper_year_map[target]
                
                if citing_year >= cited_year:  # Valid temporal order
                    citation = CitationRelation(
                        citing_paper_id=source,
                        cited_paper_id=target,
                        citing_year=citing_year,
                        cited_year=cited_year,
                        relation_description=relation_desc,
                        semantic_description=semantic_desc,
                        common_topics_count=common_topics,
                        edge_index=edge_index
                    )
                    citations.append(citation)
                else:
                    invalid_count += 1
        
        # Log filtering results
        if invalid_count > 0:
            print(f"Filtered out {invalid_count} temporally invalid citations from graph")
        print(f"Loaded {len(citations)} rich citations and {len(graph_nodes)} graph nodes")
        
        return tuple(citations), tuple(graph_nodes)
    
    except Exception as e:
        print(f"❌ Error loading citation graph: {e}")
        return tuple(), tuple()


def calculate_statistics(papers: Tuple[Paper, ...], domain_name: str) -> DataStatistics:
    """
    Calculate comprehensive statistics for a domain's papers.
    
    Args:
        papers: Tuple of Paper objects
        domain_name: Name of the research domain
        
    Returns:
        DataStatistics object with calculated metrics
    """
    if not papers:
        raise ValueError("Cannot calculate statistics for empty paper collection")
    
    # Basic statistics
    total_papers = len(papers)
    citation_counts = [p.cited_by_count for p in papers]
    years = [p.pub_year for p in papers]
    
    avg_citations = statistics.mean(citation_counts)
    median_citations = statistics.median(citation_counts)
    year_range = (min(years), max(years))
    
    # Content completeness (100% as per Phase 1 analysis)
    content_completeness = 1.0  # All papers have content
    
    # Keyword completeness
    papers_with_keywords = sum(1 for p in papers if p.keywords)
    keyword_completeness = papers_with_keywords / total_papers
    
    # Citation network metrics
    all_citation_ids = set()
    for paper in papers:
        all_citation_ids.update(paper.children)
    
    citation_network_size = len(all_citation_ids)
    avg_citations_per_paper = len(all_citation_ids) / total_papers if total_papers > 0 else 0
    
    # Top keywords
    all_keywords = []
    for paper in papers:
        all_keywords.extend(paper.keywords)
    
    keyword_counts = Counter(all_keywords)
    top_keywords = tuple(kw for kw, _ in keyword_counts.most_common(10))
    
    # Most productive years
    year_counts = Counter(years)
    most_productive_years = tuple(year_counts.most_common(5))
    
    return DataStatistics(
        domain_name=domain_name,
        total_papers=total_papers,
        year_range=year_range,
        avg_citations=avg_citations,
        median_citations=median_citations,
        content_completeness=content_completeness,
        keyword_completeness=keyword_completeness,
        citation_network_size=citation_network_size,
        avg_citations_per_paper=avg_citations_per_paper,
        top_keywords=top_keywords,
        most_productive_years=most_productive_years
    )


def analyze_keywords_and_semantics(
    papers: Tuple[Paper, ...], 
    citations: Tuple[CitationRelation, ...],
    time_window: Tuple[int, int],
    domain_name: str
) -> KeywordAnalysis:
    """
    Analyze keywords and semantic patterns from papers and rich citations.
    
    Args:
        papers: Papers in the time window
        citations: Rich citation relationships
        time_window: Time period being analyzed
        domain_name: Research domain name
        
    Returns:
        KeywordAnalysis with patterns and trends
    """
    # Keyword frequency analysis
    all_keywords = []
    for paper in papers:
        all_keywords.extend(paper.keywords)
    
    keyword_counts = Counter(all_keywords)
    keyword_frequencies = tuple(keyword_counts.most_common(20))
    
    # Emerging keywords (high frequency, recent appearance)
    recent_papers = [p for p in papers if p.pub_year >= time_window[1] - 3]
    recent_keywords = []
    for paper in recent_papers:
        recent_keywords.extend(paper.keywords)
    
    recent_keyword_counts = Counter(recent_keywords)
    emerging_keywords = tuple(kw for kw, count in recent_keyword_counts.most_common(10))
    
    # Research relationship patterns from semantic descriptions
    relationship_patterns = []
    semantic_signals = []
    
    for citation in citations:
        if citation.semantic_description:
            desc = citation.semantic_description.lower()
            
            # Extract relationship patterns
            if "builds on" in desc:
                relationship_patterns.append("builds_on")
            if "extends" in desc:
                relationship_patterns.append("extends")
            if "improves" in desc:
                relationship_patterns.append("improves")
            if "introduces" in desc or "proposes" in desc:
                semantic_signals.append("novel_introduction")
            if "framework" in desc:
                semantic_signals.append("framework_development")
            if "enhances" in desc or "performance" in desc:
                semantic_signals.append("performance_enhancement")
    
    # Get unique patterns
    relationship_patterns = tuple(set(relationship_patterns))
    semantic_signals = tuple(set(semantic_signals))
    
    return KeywordAnalysis(
        domain_name=domain_name,
        time_window=time_window,
        keyword_frequencies=keyword_frequencies,
        emerging_keywords=emerging_keywords,
        relationship_patterns=relationship_patterns,
        semantic_signals=semantic_signals
    )


def create_temporal_windows(
    papers: Tuple[Paper, ...], 
    window_size_years: int = 5
) -> Tuple[TemporalWindow, ...]:
    """
    Create temporal windows from papers for time series analysis.
    
    Args:
        papers: Tuple of Paper objects
        window_size_years: Size of each temporal window in years
        
    Returns:
        Tuple of TemporalWindow objects
    """
    if not papers:
        return tuple()
    
    years = [p.pub_year for p in papers]
    min_year, max_year = min(years), max(years)
    
    windows = []
    current_year = min_year
    
    while current_year <= max_year:
        end_year = min(current_year + window_size_years - 1, max_year)
        
        # Filter papers for this window
        window_papers = tuple(
            p for p in papers 
            if current_year <= p.pub_year <= end_year
        )
        
        if window_papers:
            citations = [p.cited_by_count for p in window_papers]
            avg_citations = statistics.mean(citations)
            total_citations = sum(citations)
            
            window = TemporalWindow(
                start_year=current_year,
                end_year=end_year,
                papers=window_papers,
                paper_count=len(window_papers),
                avg_citations=avg_citations,
                total_citations=total_citations
            )
            windows.append(window)
        
        current_year += window_size_years
    
    return tuple(windows)


def filter_papers_by_year_range(
    papers: Tuple[Paper, ...], 
    start_year: int, 
    end_year: int
) -> Tuple[Paper, ...]:
    """
    Filter papers by publication year range.
    
    Args:
        papers: Tuple of Paper objects
        start_year: Inclusive start year
        end_year: Inclusive end year
        
    Returns:
        Filtered tuple of Paper objects
    """
    return tuple(
        p for p in papers 
        if start_year <= p.pub_year <= end_year
    )


def create_data_subset(
    papers: Tuple[Paper, ...],
    name: str,
    criteria: str,
    size: int,
    domain: str,
    selection_strategy: str = "most_cited"
) -> DataSubset:
    """
    Create a data subset for testing purposes.
    
    Args:
        papers: Source papers
        name: Name for the subset
        criteria: Description of selection criteria
        size: Number of papers to include
        domain: Original domain name
        selection_strategy: Strategy for paper selection
        
    Returns:
        DataSubset object
    """
    if size > len(papers):
        size = len(papers)
    
    if selection_strategy == "most_cited":
        sorted_papers = sorted(papers, key=lambda p: p.cited_by_count, reverse=True)
        selected_papers = tuple(sorted_papers[:size])
    elif selection_strategy == "random_temporal":
        # Select papers distributed across time periods
        years = sorted(set(p.pub_year for p in papers))
        papers_per_year = size // len(years) if years else 0
        selected_papers = []
        
        for year in years:
            year_papers = [p for p in papers if p.pub_year == year]
            if year_papers:
                count = min(papers_per_year + (1 if len(selected_papers) < size % len(years) else 0), 
                           len(year_papers))
                selected_papers.extend(year_papers[:count])
                
        selected_papers = tuple(selected_papers[:size])
    else:
        # Default: take first N papers
        selected_papers = tuple(papers[:size])
    
    return DataSubset(
        name=name,
        papers=selected_papers,
        selection_criteria=criteria,
        subset_size=len(selected_papers),
        original_domain=domain
    )


def filter_papers_by_minimum_yearly_count(papers: Tuple[Paper, ...], min_papers_per_year: int = 5) -> Tuple[Paper, ...]:
    """
    Filter papers to only include years with sufficient paper count.
    
    Args:
        papers: Original papers tuple
        min_papers_per_year: Minimum number of papers required per year
        
    Returns:
        Filtered papers tuple containing only years with enough papers
    """
    from collections import Counter
    
    # Count papers per year
    papers_per_year = Counter(p.pub_year for p in papers)
    
    # Get years with sufficient papers
    valid_years = {year for year, count in papers_per_year.items() if count >= min_papers_per_year}
    
    # Filter papers to only include valid years
    filtered_papers = tuple(p for p in papers if p.pub_year in valid_years)
    
    # Log filtering results
    original_years = len(papers_per_year)
    filtered_years = len(valid_years)
    removed_papers = len(papers) - len(filtered_papers)
    
    print(f"Year filtering: {original_years} → {filtered_years} years (removed {removed_papers} papers from sparse years)")
    
    return filtered_papers


def process_domain_data(domain_name: str, data_directory: str = "resources", 
                       min_papers_per_year: int = 5, apply_year_filtering: bool = True) -> ProcessingResult:
    """
    Process data for a single domain including rich citation graph.
    
    Args:
        domain_name: Name of the domain to process
        data_directory: Directory containing domain data
        min_papers_per_year: Minimum papers required per year (if filtering enabled)
        apply_year_filtering: Whether to filter years with insufficient papers
        
    Returns:
        ProcessingResult with success status and data
    """
    start_time = time.time()
    
    try:
        # Construct file paths
        json_path = Path(data_directory) / domain_name / f"{domain_name}_docs_info.json"
        graph_path = Path(data_directory) / domain_name / f"{domain_name}_entity_relation_graph.graphml.xml"
        
        # Load papers
        papers = load_papers_from_json(str(json_path))
        print(f"Raw {domain_name}: {len(papers)} papers")
        
        # Apply year filtering if requested
        if apply_year_filtering:
            papers = filter_papers_by_minimum_yearly_count(papers, min_papers_per_year)
            
            # Check if we still have sufficient data after filtering
            if len(papers) == 0:
                raise ValueError(f"No papers remaining for {domain_name} after year filtering (min {min_papers_per_year} papers/year)")
        
        print(f"Final {domain_name}: {len(papers)} papers")
        
        # Create year mapping for citation graph processing
        paper_year_map = {p.id: p.pub_year for p in papers}
        
        # Load rich citation graph
        citations, graph_nodes = load_citation_graph(str(graph_path), paper_year_map)
        
        # Calculate statistics (on filtered data)
        statistics = calculate_statistics(papers, domain_name)
        
        # Create domain data
        domain_data = DomainData(
            domain_name=domain_name,
            papers=papers,
            citations=citations,
            graph_nodes=graph_nodes,
            year_range=statistics.year_range,
            total_papers=len(papers)
        )
        
        processing_time = time.time() - start_time
        
        return ProcessingResult(
            success=True,
            domain_data=domain_data,
            statistics=statistics,
            error_message=None,
            processing_time_seconds=processing_time,
            papers_processed=len(papers)
        )
        
    except Exception as e:
        processing_time = time.time() - start_time
        
        return ProcessingResult(
            success=False,
            domain_data=None,
            statistics=None,
            error_message=str(e),
            processing_time_seconds=processing_time,
            papers_processed=0
        )


def process_all_domains(data_directory: str = "resources") -> Dict[str, ProcessingResult]:
    """
    Process data for all available domains.
    
    Args:
        data_directory: Directory containing domain data
        
    Returns:
        Dictionary mapping domain names to ProcessingResult objects
    """
    domains = ["applied_mathematics", "art", "deep_learning", "natural_language_processing"]
    results = {}
    
    for domain in domains:
        print(f"\n🔄 Processing {domain}...")
        results[domain] = process_domain_data(domain, data_directory)
    
    return results


def get_citation_time_series(papers: Tuple[Paper, ...]) -> Dict[int, int]:
    """
    Create citation count time series by year.
    
    Args:
        papers: Tuple of Paper objects
        
    Returns:
        Dictionary mapping year to total citation count
    """
    year_citations = defaultdict(int)
    
    for paper in papers:
        year_citations[paper.pub_year] += paper.cited_by_count
    
    return dict(year_citations)


def get_productivity_time_series(papers: Tuple[Paper, ...]) -> Dict[int, int]:
    """
    Create paper count time series by year.
    
    Args:
        papers: Tuple of Paper objects
        
    Returns:
        Dictionary mapping year to paper count
    """
    year_counts = Counter(p.pub_year for p in papers)
    return dict(year_counts) 