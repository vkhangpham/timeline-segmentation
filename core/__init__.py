# Core package for timeline segmentation algorithm
# Updated structure post-restructuring with organized sub-modules

# Import main objective function for easy access
from .analysis.objective_function import (
    evaluate_timeline_quality,
    compute_objective_function,
    ObjectiveFunctionResult,
    jaccard_cohesion,
    jensen_shannon_separation
)

# Import key data models
from .data.models import Paper, DomainData

# Import consolidated keyword utilities
from .utils.keywords import (
    extract_keywords_from_papers,
    count_keyword_frequencies,
    get_top_keywords,
    analyze_keyword_distribution,
    calculate_jaccard_similarity
)

# Import main pipeline orchestration
from .pipeline.orchestrator import run_complete_analysis

# Import data processing functions
from .data.processing import (
    load_papers_from_json,
    process_domain_data,
    calculate_statistics
)

# Import configuration management
from .utils.config import AlgorithmConfig, create_default_config

# Main exports for public API
__all__ = [
    # Core objective function and evaluation
    'evaluate_timeline_quality',
    'compute_objective_function', 
    'ObjectiveFunctionResult',
    'jaccard_cohesion',
    'jensen_shannon_separation',
    
    # Data models and structures
    'Paper',
    'DomainData',
    
    # Main pipeline entry point
    'run_complete_analysis',
    
    # Data processing and loading
    'load_papers_from_json',
    'process_domain_data', 
    'calculate_statistics',
    
    # Configuration management
    'AlgorithmConfig',
    'create_default_config',
    
    # Consolidated keyword utilities
    'extract_keywords_from_papers',
    'count_keyword_frequencies',
    'get_top_keywords',
    'analyze_keyword_distribution',
    'calculate_jaccard_similarity'
]