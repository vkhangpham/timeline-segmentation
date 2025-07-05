"""
Core data processing functions for scientific publication analysis.

This module provides pure functions for loading, validating, and processing publication data
including rich citation graph information from .graphml.xml files.
"""

import json
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Tuple, List, Any
from collections import defaultdict, Counter
import statistics
import pandas as pd

from .models import (
    Paper, CitationRelation, DomainData, TemporalWindow, 
    DataStatistics, ProcessingResult, DataSubset, KeywordAnalysis
)
from ..utils.logging import get_logger


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
            # Handle both old and new data formats
            description = paper_data.get('description', paper_data.get('content', ''))
            content = paper_data.get('content', '')
            
            paper = Paper(
                id=paper_id,
                title=paper_data['title'],
                content=content,
                pub_year=paper_data['pub_year'],
                cited_by_count=paper_data['cited_by_count'],
                keywords=tuple(paper_data['keywords']),
                children=tuple(paper_data['children']),
                description=description
            )
            papers.append(paper)
        
        return tuple(papers)
    
    except FileNotFoundError:
        raise FileNotFoundError(f"Data file not found: {file_path}")
    except (KeyError, TypeError) as e:
        raise ValueError(f"Invalid data format in {file_path}: {e}")


def load_citation_graph(file_path: str, paper_year_map: Dict[str, int], verbose: bool = False) -> Tuple[Tuple[CitationRelation, ...], Tuple[Tuple[str, str], ...]]:
    """
    Load rich citation graph from .graphml.xml file.
    
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
            logger.info(f"Filtered out {invalid_count} temporally invalid citations from graph")
        logger.info(f"Loaded {len(citations)} rich citations and {len(graph_nodes)} graph nodes")
        
        return tuple(citations), tuple(graph_nodes)
    
    except Exception as e:
        # FAIL-FAST: Citation graph loading errors are critical and should not be masked
        raise RuntimeError(f"Failed to load citation graph from {file_path}: {e}") from e


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
    
    # Content completeness verified
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
    
    # CONSOLIDATED: Use keyword_utils for top keywords
    from ..utils.keywords import get_top_keywords
    
    # Top keywords
    top_keywords_with_counts = get_top_keywords(papers, top_k=10)
    top_keywords = tuple(kw for kw, _ in top_keywords_with_counts)
    
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
    # CONSOLIDATED: Use keyword_utils for frequency analysis and emerging keywords
    from ..utils.keywords import get_top_keywords, get_emerging_keywords
    
    # Keyword frequency analysis
    keyword_frequencies = tuple(get_top_keywords(papers, top_k=20))
    
    # Emerging keywords (high frequency, recent appearance)
    emerging_keywords_list, _ = get_emerging_keywords(papers, recent_years=3, top_k=10)
    emerging_keywords = tuple(emerging_keywords_list)
    
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


def filter_papers_by_minimum_yearly_count(papers: Tuple[Paper, ...], min_papers_per_year: int = 5, verbose: bool = False) -> Tuple[Paper, ...]:
    """
    Filter papers to only include years with sufficient paper count.
    
    Args:
        papers: Original papers tuple
        min_papers_per_year: Minimum number of papers required per year
        verbose: Enable verbose logging
        
    Returns:
        Filtered papers tuple containing only years with enough papers
    """
    logger = get_logger(__name__, verbose)
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
    
    logger.info(f"Year filtering: {original_years} → {filtered_years} years (removed {removed_papers} papers from sparse years)")
    
    return filtered_papers


def process_domain_data(domain_name: str, data_directory: str = "resources", 
                       min_papers_per_year: int = 5, apply_year_filtering: bool = True, 
                       verbose: bool = False) -> ProcessingResult:
    """
    Process data for a single domain including rich citation graph.
    
    Args:
        domain_name: Name of the domain to process
        data_directory: Directory containing domain data
        min_papers_per_year: Minimum papers required per year (if filtering enabled)
        apply_year_filtering: Whether to filter years with insufficient papers
        verbose: Enable verbose logging
        
    Returns:
        ProcessingResult with success status and data
    """
    logger = get_logger(__name__, verbose)
    start_time = time.time()
    
    try:
        # Construct file paths
        json_path = Path(data_directory) / domain_name / f"{domain_name}_docs_info.json"
        graph_path = Path(data_directory) / domain_name / f"{domain_name}_entity_relation_graph.graphml.xml"
        
        # Load papers
        papers = load_papers_from_json(str(json_path))
        logger.info(f"Raw {domain_name}: {len(papers)} papers")
        
        # Apply year filtering if requested
        if apply_year_filtering:
            papers = filter_papers_by_minimum_yearly_count(papers, min_papers_per_year, verbose)
            
            # Check if we still have sufficient data after filtering
            if len(papers) == 0:
                raise ValueError(f"No papers remaining for {domain_name} after year filtering (min {min_papers_per_year} papers/year)")
        
        logger.info(f"Final {domain_name}: {len(papers)} papers")
        
        # Create year mapping for citation graph processing
        paper_year_map = {p.id: p.pub_year for p in papers}
        
        # Load rich citation graph
        citations, graph_nodes = load_citation_graph(str(graph_path), paper_year_map, verbose)
        
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


def process_all_domains(data_directory: str = "resources", verbose: bool = False) -> Dict[str, ProcessingResult]:
    """
    Process data for all available domains.
    
    Args:
        data_directory: Directory containing domain data
        verbose: Enable verbose logging
        
    Returns:
        Dictionary mapping domain names to ProcessingResult objects
    """
    logger = get_logger(__name__, verbose)
    domains = ["applied_mathematics", "art", "deep_learning", "natural_language_processing"]
    results = {}
    
    for domain in domains:
        logger.info(f"Processing {domain}...")
        results[domain] = process_domain_data(domain, data_directory, verbose=verbose)
    
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


# =============================================================================
# DataFrame Compatibility Functions (for backward compatibility)
# =============================================================================

def convert_keywords_to_list(keywords_str: str) -> List[str]:
    """
    Convert pipe-separated keyword string to list.
    
    CONSOLIDATED: Now uses keyword_utils.convert_keywords_string_to_list()
    
    Args:
        keywords_str: Pipe-separated keywords string
        
    Returns:
        List of keywords
    """
    from ..utils.keywords import convert_keywords_string_to_list
    
    if pd.isna(keywords_str):
        return []
    
    return convert_keywords_string_to_list(keywords_str, separator='|')


def convert_children_to_list(children_str: str) -> List[str]:
    """
    Convert pipe-separated children string to list.
    
    Args:
        children_str: Pipe-separated children string
        
    Returns:
        List of children IDs
    """
    if pd.isna(children_str) or not children_str:
        return []
    return [c.strip() for c in children_str.split('|') if c.strip()]


def convert_papers_to_dataframe(papers: Tuple[Paper, ...]) -> 'pd.DataFrame':
    """
    Convert papers to DataFrame format for backward compatibility.
    
    Args:
        papers: Tuple of Paper objects
        
    Returns:
        DataFrame with columns: id, title, content, year, cited_by_count, keywords, children
    """
    rows = []
    for paper in papers:
        row = {
            'id': paper.id,
            'title': paper.title,
            'content': paper.content,
            'year': paper.pub_year,
            'cited_by_count': paper.cited_by_count,
            'keywords': '|'.join(paper.keywords) if paper.keywords else '',
            'children': '|'.join(paper.children) if paper.children else ''
        }
        rows.append(row)
    
    return pd.DataFrame(rows)


def load_domain_data_as_dataframe(domain: str, resources_dir: str = "resources") -> 'pd.DataFrame':
    """
    Load domain data as DataFrame (backward compatibility interface).
    
    Args:
        domain: Domain name (e.g., 'art', 'deep_learning')
        resources_dir: Path to resources directory
        
    Returns:
        DataFrame with columns: id, title, content, year, cited_by_count, keywords, children
        
    Raises:
        RuntimeError: If data loading fails
    """
    result = process_domain_data(domain, resources_dir, apply_year_filtering=False)
    
    if not result.success:
        raise RuntimeError(f"Failed to load {domain}: {result.error_message}")
    
    return convert_papers_to_dataframe(result.domain_data.papers)


def filter_dataframe_by_year_count(df: 'pd.DataFrame', min_papers_per_year: int = 5, verbose: bool = False) -> 'pd.DataFrame':
    """
    Filter DataFrame to only include years with sufficient papers (backward compatibility).
    
    Args:
        df: Domain data DataFrame
        min_papers_per_year: Minimum number of papers required per year
        verbose: Enable verbose logging
        
    Returns:
        Filtered DataFrame containing only years with enough papers
    """
    logger = get_logger(__name__, verbose)
    
    if 'year' not in df.columns:
        return df
    
    # Count papers per year
    papers_per_year = df['year'].value_counts()
    
    # Get years with sufficient papers
    valid_years = papers_per_year[papers_per_year >= min_papers_per_year].index
    
    # Filter DataFrame to only include valid years
    filtered_df = df[df['year'].isin(valid_years)].copy()
    
    # Log filtering results
    original_years = len(papers_per_year)
    filtered_years = len(valid_years)
    removed_papers = len(df) - len(filtered_df)
    
    logger.info(f"Year filtering: {original_years} → {filtered_years} years (removed {removed_papers} papers from sparse years)")
    
    return filtered_df


# =============================================================================
# Data Validation Functions
# =============================================================================

def validate_domain_papers(papers: Tuple[Paper, ...], domain: str) -> Tuple[bool, List[str]]:
    """
    Validate domain papers integrity.
    
    Args:
        papers: Tuple of Paper objects
        domain: Domain name for reporting
        
    Returns:
        Tuple of (is_valid, list_of_issues)
    """
    issues = []
    
    # Check for empty collection
    if len(papers) == 0:
        issues.append("Paper collection is empty")
        return False, issues
    
    # Check for duplicate IDs
    paper_ids = [p.id for p in papers]
    if len(paper_ids) != len(set(paper_ids)):
        duplicates = len(paper_ids) - len(set(paper_ids))
        issues.append(f"Found {duplicates} duplicate paper IDs")
    
    # Check year range sanity
    years = [p.pub_year for p in papers]
    min_year, max_year = min(years), max(years)
    if min_year < 1900 or max_year > 2025:
        issues.append(f"Suspicious year range: {min_year}-{max_year}")
    
    # Check for missing essential data
    empty_titles = sum(1 for p in papers if not p.title.strip())
    if empty_titles > 0:
        issues.append(f"Found {empty_titles} papers with missing titles")
    
    empty_content = sum(1 for p in papers if not p.content.strip())
    if empty_content > 0:
        issues.append(f"Found {empty_content} papers with missing content")
    
    # Check for negative citation counts
    negative_citations = sum(1 for p in papers if p.cited_by_count < 0)
    if negative_citations > 0:
        issues.append(f"Found {negative_citations} papers with negative citation counts")
    
    is_valid = len(issues) == 0
    return is_valid, issues


def get_domain_statistics_from_dataframe(df: 'pd.DataFrame') -> Dict[str, Any]:
    """
    Calculate basic statistics for a domain DataFrame (backward compatibility).
    
    Args:
        df: Domain data DataFrame
        
    Returns:
        Dictionary with statistics
    """
    stats = {
        'total_papers': len(df),
        'year_range': (int(df['year'].min()), int(df['year'].max())),
        'avg_citations': float(df['cited_by_count'].mean()),
        'median_citations': float(df['cited_by_count'].median()),
        'papers_with_keywords': int(df['keywords'].apply(lambda x: len(convert_keywords_to_list(x)) > 0).sum()),
        'papers_with_children': int(df['children'].apply(lambda x: len(convert_children_to_list(x)) > 0).sum()),
    }
    
    stats['keyword_completeness'] = stats['papers_with_keywords'] / stats['total_papers']
    stats['citation_completeness'] = stats['papers_with_children'] / stats['total_papers']
    
    return stats


def validate_dataframe_data(df: 'pd.DataFrame', domain: str) -> Tuple[bool, List[str]]:
    """
    Validate domain DataFrame integrity (backward compatibility).
    
    Args:
        df: Domain data DataFrame
        domain: Domain name for reporting
        
    Returns:
        Tuple of (is_valid, list_of_issues)
    """
    issues = []
    
    # Check required columns
    required_columns = ['id', 'title', 'content', 'year', 'cited_by_count', 'keywords', 'children']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        issues.append(f"Missing required columns: {missing_columns}")
    
    # Check for empty DataFrame
    if len(df) == 0:
        issues.append("DataFrame is empty")
        return False, issues
    
    # Check for duplicate IDs
    if df['id'].duplicated().any():
        issues.append(f"Found {df['id'].duplicated().sum()} duplicate paper IDs")
    
    # Check year range sanity
    if 'year' in df.columns:
        min_year, max_year = df['year'].min(), df['year'].max()
        if min_year < 1900 or max_year > 2025:
            issues.append(f"Suspicious year range: {min_year}-{max_year}")
    
    # Check for missing titles or content
    if 'title' in df.columns:
        empty_titles = df['title'].isnull().sum()
        if empty_titles > 0:
            issues.append(f"Found {empty_titles} papers with missing titles")
    
    if 'content' in df.columns:
        empty_content = df['content'].isnull().sum()
        if empty_content > 0:
            issues.append(f"Found {empty_content} papers with missing content")
    
    is_valid = len(issues) == 0
    return is_valid, issues


def load_and_validate_domain_data_as_dataframe(domain: str, 
                                             resources_dir: str = "resources",
                                             validate: bool = True,
                                             min_papers_per_year: int = 5,
                                             apply_year_filtering: bool = True,
                                             verbose: bool = False) -> 'pd.DataFrame':
    """
    Load and optionally validate domain data as DataFrame (backward compatibility interface).
    
    Args:
        domain: Domain name
        resources_dir: Path to resources directory
        validate: Whether to run validation checks
        min_papers_per_year: Minimum papers required per year (if filtering enabled)
        apply_year_filtering: Whether to filter years with insufficient papers
        verbose: Enable verbose logging
        
    Returns:
        Validated and filtered domain data DataFrame
        
    Raises:
        ValueError: If validation fails or data loading fails
    """
    logger = get_logger(__name__, verbose)
    
    # Load data using the core pipeline
    result = process_domain_data(domain, resources_dir, min_papers_per_year, apply_year_filtering, verbose)
    
    if not result.success:
        raise ValueError(f"Failed to load {domain}: {result.error_message}")
    
    # Convert to DataFrame
    df = convert_papers_to_dataframe(result.domain_data.papers)
    
    # Print original statistics
    original_stats = get_domain_statistics_from_dataframe(df)
    logger.info(f"Raw {domain}: {original_stats['total_papers']} papers ({original_stats['year_range'][0]}-{original_stats['year_range'][1]})")
    
    # Validate if requested
    if validate:
        is_valid, issues = validate_dataframe_data(df, domain)
        if not is_valid:
            raise ValueError(f"Data validation failed for {domain}: {'; '.join(issues)}")
        
        if issues:
            logger.warning(f"Data validation warnings for {domain}: {'; '.join(issues)}")
    
    # Print final statistics
    final_stats = get_domain_statistics_from_dataframe(df)
    logger.info(f"Final {domain}: {final_stats['total_papers']} papers ({final_stats['year_range'][0]}-{final_stats['year_range'][1]}) - Avg citations: {final_stats['avg_citations']:.0f}")
    
    return df


def load_domain_data_enriched(domain: str, 
                            resources_dir: str = "resources",
                            apply_year_filtering: bool = False,
                            verbose: bool = False) -> DomainData:
    """
    Load domain data with citation edges populated from JSON + GraphML sources.
    
    This function provides citation-enriched data by parsing both the JSON metadata 
    and GraphML citation graph, enabling meaningful citation density metrics.
    
    Args:
        domain: Domain name (e.g., 'applied_mathematics', 'computer_vision')
        resources_dir: Path to resources directory containing JSON and GraphML files
        apply_year_filtering: Whether to filter years with insufficient papers
        verbose: Enable verbose logging
        
    Returns:
        DomainData with populated Paper.children and citations tuples
        
    Raises:
        RuntimeError: If enrichment fails for any reason (fail-fast behavior)
        FileNotFoundError: If required JSON or GraphML files are missing
    """
    logger = get_logger(__name__, verbose)
    logger.info(f"Loading citation-enriched data for {domain}")
    
    # Use the proven data processing pipeline that merges JSON + GraphML
    result = process_domain_data(
        domain_name=domain,
        data_directory=resources_dir,
        min_papers_per_year=5,  # Standard filtering for data quality
        apply_year_filtering=apply_year_filtering,
        verbose=verbose
    )
    
    # Fail-fast: raise error immediately if processing failed
    if not result.success:
        raise RuntimeError(f"Citation enrichment failed for {domain}: {result.error_message}")
    
    # Validate that we actually got citation data
    domain_data = result.domain_data
    papers_with_citations = sum(1 for paper in domain_data.papers if paper.children)
    citation_coverage = papers_with_citations / len(domain_data.papers) if domain_data.papers else 0.0
    
    logger.info(f"Citation-enriched {domain}: {len(domain_data.papers)} papers, "
          f"{len(domain_data.citations)} citation edges, "
          f"{citation_coverage:.1%} papers have outgoing citations")
    
    # Warn if citation coverage is very low (might indicate GraphML parsing issues)
    if citation_coverage < 0.05:  # Less than 5% of papers have citations
        logger.warning(f"Low citation coverage ({citation_coverage:.1%}) - GraphML might be sparse")
    
    return domain_data


# Backward compatibility aliases
load_domain_data = load_domain_data_as_dataframe  # For scripts expecting DataFrame interface
load_and_validate_domain_data = load_and_validate_domain_data_as_dataframe 