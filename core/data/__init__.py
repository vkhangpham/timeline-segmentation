# Data Processing Sub-module
# Handles paper loading, domain data processing, and data models

# Import data models
from .models import (
    Paper,
    DomainData,
    CitationRelation,
    TemporalWindow,
    DataStatistics,
    ProcessingResult,
    DataSubset,
    KeywordAnalysis
)

# Import data processing functions
from .processing import (
    load_papers_from_json,
    process_domain_data,
    calculate_statistics,
    analyze_keywords_and_semantics,
    create_temporal_windows,
    filter_papers_by_year_range,
    process_all_domains,
    # DataFrame compatibility functions
    convert_papers_to_dataframe,
    load_domain_data_as_dataframe,
    convert_keywords_to_list,
    convert_children_to_list
)

# Export all
__all__ = [
    # Data models
    'Paper',
    'DomainData', 
    'CitationRelation',
    'TemporalWindow',
    'DataStatistics',
    'ProcessingResult',
    'DataSubset',
    'KeywordAnalysis',
    
    # Data processing functions
    'load_papers_from_json',
    'process_domain_data',
    'calculate_statistics',
    'analyze_keywords_and_semantics',
    'create_temporal_windows',
    'filter_papers_by_year_range',
    'process_all_domains',
    
    # DataFrame compatibility
    'convert_papers_to_dataframe',
    'load_domain_data_as_dataframe',
    'convert_keywords_to_list',
    'convert_children_to_list'
] 