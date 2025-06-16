"""
Unified data loader for timeline analysis.

This module provides functions to load research domain data from either JSON resources
or processed CSV files, offering flexibility in data source selection.
"""

import json
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Tuple


def load_domain_data_json(domain: str, resources_dir: str = "resources") -> pd.DataFrame:
    """
    Load domain data directly from JSON resources.
    
    Args:
        domain: Domain name (e.g., 'art', 'deep_learning')
        resources_dir: Path to resources directory
        
    Returns:
        DataFrame with columns: id, title, content, year, cited_by_count, keywords, children
    """
    json_path = Path(resources_dir) / domain / f"{domain}_docs_info.json"
    
    if not json_path.exists():
        raise FileNotFoundError(f"JSON file not found: {json_path}")
    
    # Load JSON data
    with open(json_path, 'r', encoding='utf-8') as f:
        json_data = json.load(f)
    
    # Convert to DataFrame format
    rows = []
    for paper_id, paper_data in json_data.items():
        row = {
            'id': paper_id,
            'title': paper_data.get('title', ''),
            'content': paper_data.get('content', ''),
            'year': paper_data.get('pub_year', 0),
            'cited_by_count': paper_data.get('cited_by_count', 0),
            'keywords': '|'.join(paper_data.get('keywords', [])),
            'children': '|'.join(paper_data.get('children', []))
        }
        rows.append(row)
    
    return pd.DataFrame(rows)


def load_domain_data_csv(domain: str, processed_dir: str = "data/processed") -> pd.DataFrame:
    """
    Load domain data from processed CSV files.
    
    Args:
        domain: Domain name (e.g., 'art', 'deep_learning')
        processed_dir: Path to processed data directory
        
    Returns:
        DataFrame with columns: id, title, content, year, cited_by_count, keywords, children
    """
    csv_path = Path(processed_dir) / f"{domain}_processed.csv"
    
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    return pd.read_csv(csv_path)


def load_domain_data(domain: str, 
                    prefer_source: str = "csv",
                    resources_dir: str = "resources",
                    processed_dir: str = "data/processed") -> pd.DataFrame:
    """
    Load domain data from available sources with fallback.
    
    Args:
        domain: Domain name (e.g., 'art', 'deep_learning')
        prefer_source: Preferred data source ('csv' or 'json')
        resources_dir: Path to resources directory
        processed_dir: Path to processed data directory
        
    Returns:
        DataFrame with domain data
        
    Raises:
        FileNotFoundError: If no data source is available
    """
    csv_path = Path(processed_dir) / f"{domain}_processed.csv"
    json_path = Path(resources_dir) / domain / f"{domain}_docs_info.json"
    
    # Try preferred source first
    if prefer_source == "csv":
        if csv_path.exists():
            return load_domain_data_csv(domain, processed_dir)
        elif json_path.exists():
            return load_domain_data_json(domain, resources_dir)
    else:  # prefer_source == "json"
        if json_path.exists():
            return load_domain_data_json(domain, resources_dir)
        elif csv_path.exists():
            return load_domain_data_csv(domain, processed_dir)
    
    # If neither source is available
    raise FileNotFoundError(
        f"No data source found for domain '{domain}'. "
        f"Expected: {csv_path} or {json_path}"
    )


def discover_available_domains(resources_dir: str = "resources", 
                             processed_dir: str = "data/processed") -> Dict[str, List[str]]:
    """
    Discover all available domains and their data sources.
    
    Args:
        resources_dir: Path to resources directory
        processed_dir: Path to processed data directory
        
    Returns:
        Dictionary mapping domain names to available sources ['json', 'csv']
    """
    domains = {}
    
    # Check resources directory for JSON files
    resources_path = Path(resources_dir)
    if resources_path.exists():
        for item in resources_path.iterdir():
            if item.is_dir() and not item.name.startswith('.'):
                json_file = item / f"{item.name}_docs_info.json"
                if json_file.exists():
                    if item.name not in domains:
                        domains[item.name] = []
                    domains[item.name].append('json')
    
    # Check processed directory for CSV files
    processed_path = Path(processed_dir)
    if processed_path.exists():
        for csv_file in processed_path.glob("*_processed.csv"):
            domain = csv_file.stem.replace("_processed", "")
            if domain not in domains:
                domains[domain] = []
            domains[domain].append('csv')
    
    return {k: sorted(v) for k, v in domains.items()}


def convert_keywords_to_list(keywords_str: str) -> List[str]:
    """
    Convert pipe-separated keyword string to list.
    
    Args:
        keywords_str: Pipe-separated keywords string
        
    Returns:
        List of keywords
    """
    if pd.isna(keywords_str) or not keywords_str:
        return []
    return [k.strip() for k in keywords_str.split('|') if k.strip()]


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


def get_domain_statistics(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculate basic statistics for a domain DataFrame.
    
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


def validate_domain_data(df: pd.DataFrame, domain: str) -> Tuple[bool, List[str]]:
    """
    Validate domain data integrity.
    
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


def load_and_validate_domain_data(domain: str, 
                                prefer_source: str = "csv",
                                validate: bool = True) -> pd.DataFrame:
    """
    Load and optionally validate domain data.
    
    Args:
        domain: Domain name
        prefer_source: Preferred data source ('csv' or 'json')
        validate: Whether to run validation checks
        
    Returns:
        Validated domain data DataFrame
        
    Raises:
        ValueError: If validation fails
    """
    # Load data
    df = load_domain_data(domain, prefer_source=prefer_source)
    
    # Validate if requested
    if validate:
        is_valid, issues = validate_domain_data(df, domain)
        if not is_valid:
            raise ValueError(f"Data validation failed for {domain}: {'; '.join(issues)}")
        
        if issues:
            print(f"⚠️  Data validation warnings for {domain}: {'; '.join(issues)}")
    
    # Print basic statistics
    stats = get_domain_statistics(df)
    print(f"✅ Loaded {domain}: {stats['total_papers']} papers ({stats['year_range'][0]}-{stats['year_range'][1]}) - Avg citations: {stats['avg_citations']:.0f}")
    
    return df 