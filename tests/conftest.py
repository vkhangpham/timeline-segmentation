"""
Pytest configuration and fixtures for Timeline Segmentation Algorithm test suite.

Following Phase 13 principle: Comprehensive testing with real data subsets and fail-fast behavior.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Tuple
import sys
import os

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.algorithm_config import ComprehensiveAlgorithmConfig
from core.data_models import DomainData, Paper, Citation
from core.shift_signal_detection import detect_shift_signals


# =============================================================================
# REAL DATA LOADING FIXTURES (Project Guideline: Use real data, not mock data)
# =============================================================================

@pytest.fixture
def available_domains() -> List[str]:
    """List of available domains for testing."""
    return [
        'applied_mathematics',
        'art', 
        'computer_vision',
        'deep_learning',
        'machine_learning',
        'machine_translation',
        'natural_language_processing'
    ]


@pytest.fixture
def small_domain_data(available_domains) -> Dict[str, DomainData]:
    """
    Load small subsets of real domain data for testing.
    
    Following project guideline: Use real data subsets for testing, not synthetic data.
    """
    domain_subsets = {}
    
    for domain in available_domains:
        try:
            # Load first 100 papers from each domain for testing
            subset_data = load_domain_subset(domain, max_papers=100)
            domain_subsets[domain] = subset_data
        except Exception as e:
            # Fail fast - don't mask data loading errors
            pytest.fail(f"Failed to load domain subset for {domain}: {e}")
    
    return domain_subsets


@pytest.fixture
def tiny_domain_data(available_domains) -> Dict[str, DomainData]:
    """
    Load tiny subsets (20 papers) for fast unit testing.
    """
    domain_subsets = {}
    
    for domain in available_domains:
        try:
            subset_data = load_domain_subset(domain, max_papers=20)
            domain_subsets[domain] = subset_data
        except Exception as e:
            pytest.fail(f"Failed to load tiny domain subset for {domain}: {e}")
    
    return domain_subsets


def load_domain_subset(domain_name: str, max_papers: int = 100) -> DomainData:
    """
    Load subset of domain data for testing.
    
    Args:
        domain_name: Name of domain to load
        max_papers: Maximum number of papers to load
        
    Returns:
        DomainData subset
    """
    data_path = Path(f"data/processed/{domain_name}_processed.csv")
    
    if not data_path.exists():
        raise FileNotFoundError(f"Domain data not found: {data_path}")
    
    # Load and subset the data
    df = pd.read_csv(data_path)
    df_subset = df.head(max_papers)
    
    # Convert to DomainData
    papers = []
    citations = []
    
    for _, row in df_subset.iterrows():
        # Create Paper object
        paper = Paper(
            id=str(row.get('id', '')),
            title=str(row.get('title', '')),
            pub_year=int(row.get('pub_year', 2000)),
            cited_by_count=int(row.get('cited_by_count', 0)),
            keywords=str(row.get('keywords', '')).split(',') if pd.notna(row.get('keywords')) else [],
            abstract=str(row.get('abstract', ''))
        )
        papers.append(paper)
        
        # Create Citation objects if citation data exists
        if pd.notna(row.get('citing_papers')):
            # Simple citation creation for testing
            citation = Citation(
                citing_paper_id=str(row.get('id', '')),
                cited_paper_id=f"cited_{row.get('id', '')}",
                citing_year=int(row.get('pub_year', 2000)),
                semantic_description=str(row.get('abstract', ''))[:100] if pd.notna(row.get('abstract')) else ''
            )
            citations.append(citation)
    
    # Calculate year range
    years = [p.pub_year for p in papers if p.pub_year > 0]
    year_range = (min(years), max(years)) if years else (2000, 2024)
    
    return DomainData(
        domain_name=domain_name,
        papers=tuple(papers),
        citations=tuple(citations),
        year_range=year_range
    )


# =============================================================================
# CONFIGURATION FIXTURES
# =============================================================================

@pytest.fixture
def default_config() -> ComprehensiveAlgorithmConfig:
    """Default algorithm configuration for testing."""
    return ComprehensiveAlgorithmConfig(granularity=3)


@pytest.fixture
def all_granularity_configs() -> List[ComprehensiveAlgorithmConfig]:
    """All granularity levels for testing."""
    return [ComprehensiveAlgorithmConfig(granularity=i) for i in range(1, 6)]


@pytest.fixture
def edge_case_configs() -> List[ComprehensiveAlgorithmConfig]:
    """Edge case configurations for boundary testing."""
    return [
        # Minimum values
        ComprehensiveAlgorithmConfig.create_custom(
            granularity=3,
            overrides={
                'direction_threshold': 0.1,
                'clustering_window': 1,
                'validation_threshold': 0.5,
                'citation_boost': 0.0
            }
        ),
        # Maximum values
        ComprehensiveAlgorithmConfig.create_custom(
            granularity=3,
            overrides={
                'direction_threshold': 0.8,
                'clustering_window': 10,
                'validation_threshold': 0.95,
                'citation_boost': 1.0
            }
        )
    ]


@pytest.fixture
def invalid_configs() -> List[Dict[str, Any]]:
    """Invalid configurations for validation testing."""
    return [
        {'direction_threshold': -0.1},  # Below minimum
        {'direction_threshold': 1.1},   # Above maximum
        {'clustering_window': 0},       # Below minimum
        {'clustering_window': 15},      # Above maximum
        {'validation_threshold': 0.3},  # Below minimum
        {'validation_threshold': 1.1},  # Above maximum
        {'citation_boost': -0.1},       # Below minimum
        {'citation_boost': 1.5},        # Above maximum
        {'segment_length_thresholds': [8, 6, 4]},  # Wrong order
        {'statistical_significance_breakpoints': [0.6, 0.4]}  # Wrong order
    ]


# =============================================================================
# GROUND TRUTH FIXTURES
# =============================================================================

@pytest.fixture
def ground_truth_data(available_domains) -> Dict[str, List[int]]:
    """
    Load ground truth paradigm shifts for validation testing.
    """
    ground_truth = {}
    
    for domain in available_domains:
        try:
            gt_path = Path(f"validation/{domain}_groundtruth.json")
            if gt_path.exists():
                import json
                with open(gt_path) as f:
                    gt_data = json.load(f)
                    ground_truth[domain] = gt_data.get('paradigm_shifts', [])
            else:
                # Use empty list if no ground truth available
                ground_truth[domain] = []
        except Exception as e:
            # Fail fast - don't mask ground truth loading errors
            pytest.fail(f"Failed to load ground truth for {domain}: {e}")
    
    return ground_truth


# =============================================================================
# UTILITY FIXTURES AND HELPERS
# =============================================================================

@pytest.fixture
def algorithm_test_runner():
    """
    Factory for running algorithm tests with consistent setup.
    """
    def run_algorithm_test(domain_data: DomainData, 
                          config: ComprehensiveAlgorithmConfig,
                          expect_signals: bool = True):
        """
        Run algorithm with error handling and validation.
        
        Args:
            domain_data: Domain data to test
            config: Algorithm configuration
            expect_signals: Whether to expect paradigm signals (fail if none)
            
        Returns:
            Tuple of (signals, evidence, metadata)
        """
        try:
            signals, evidence, metadata = detect_shift_signals(
                domain_data, domain_data.domain_name, config
            )
            
            # Validate results
            assert isinstance(signals, list), "Signals must be a list"
            assert isinstance(evidence, list), "Evidence must be a list"
            assert isinstance(metadata, dict), "Metadata must be a dict"
            
            if expect_signals and len(signals) == 0:
                pytest.fail(f"Expected paradigm signals but got none for {domain_data.domain_name}")
            
            return signals, evidence, metadata
            
        except Exception as e:
            # Fail fast - propagate all algorithm errors
            pytest.fail(f"Algorithm failed for {domain_data.domain_name}: {e}")
    
    return run_algorithm_test


@pytest.fixture 
def validation_helpers():
    """
    Helper functions for test validation.
    """
    class ValidationHelpers:
        
        @staticmethod
        def assert_valid_paradigm_signals(signals):
            """Validate paradigm signal structure."""
            for signal in signals:
                assert hasattr(signal, 'year'), "Signal must have year"
                assert hasattr(signal, 'confidence'), "Signal must have confidence"
                assert hasattr(signal, 'signal_type'), "Signal must have signal_type"
                assert 0.0 <= signal.confidence <= 1.0, f"Invalid confidence: {signal.confidence}"
                assert signal.year > 1000, f"Invalid year: {signal.year}"
        
        @staticmethod
        def assert_config_valid(config: ComprehensiveAlgorithmConfig):
            """Validate configuration object."""
            # These should not raise exceptions if config is valid
            config._validate_parameters()
            summary = config.get_configuration_summary()
            assert len(summary) > 0, "Configuration summary should not be empty"
        
        @staticmethod
        def calculate_detection_metrics(detected_years: List[int], 
                                      ground_truth_years: List[int],
                                      tolerance: int = 2) -> Dict[str, float]:
            """Calculate detection precision/recall with tolerance."""
            if not ground_truth_years:
                return {'precision': 1.0, 'recall': 1.0, 'f1': 1.0}
            
            # True positives: detected shifts within tolerance of ground truth
            true_positives = 0
            for detected_year in detected_years:
                if any(abs(detected_year - gt_year) <= tolerance for gt_year in ground_truth_years):
                    true_positives += 1
            
            # Calculate metrics
            precision = true_positives / max(len(detected_years), 1)
            recall = true_positives / len(ground_truth_years)
            f1 = 2 * precision * recall / max(precision + recall, 1e-10)
            
            return {
                'precision': precision,
                'recall': recall, 
                'f1': f1,
                'true_positives': true_positives,
                'false_positives': len(detected_years) - true_positives,
                'false_negatives': len(ground_truth_years) - true_positives
            }
    
    return ValidationHelpers()


# =============================================================================
# TEST MARKERS AND CONFIGURATION
# =============================================================================

def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "unit: mark test as unit test"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as integration test"
    )
    config.addinivalue_line(
        "markers", "parameter: mark test as parameter validation test"
    )
    config.addinivalue_line(
        "markers", "regression: mark test as regression test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )


# =============================================================================
# TEST DATA VALIDATION
# =============================================================================

def pytest_sessionstart(session):
    """Validate test environment before running tests."""
    
    # Check that core modules can be imported
    try:
        from core.shift_signal_detection import detect_shift_signals
        from core.algorithm_config import ComprehensiveAlgorithmConfig
        from core.data_models import DomainData
    except ImportError as e:
        pytest.exit(f"Failed to import core modules: {e}")
    
    # Check that at least some domain data exists
    data_dir = Path("data/processed")
    if not data_dir.exists():
        pytest.exit("No processed data directory found")
    
    csv_files = list(data_dir.glob("*.csv"))
    if len(csv_files) == 0:
        pytest.exit("No domain data files found for testing")
    
    print(f"✅ Test environment validated: {len(csv_files)} domain data files available")


# =============================================================================
# PERFORMANCE MONITORING
# =============================================================================

@pytest.fixture
def performance_monitor():
    """Monitor test performance and memory usage."""
    import time
    import psutil
    import os
    
    class PerformanceMonitor:
        def __init__(self):
            self.start_time = None
            self.start_memory = None
            
        def start(self):
            self.start_time = time.time()
            process = psutil.Process(os.getpid())
            self.start_memory = process.memory_info().rss / 1024 / 1024  # MB
            
        def stop(self) -> Dict[str, float]:
            if self.start_time is None:
                return {}
                
            elapsed_time = time.time() - self.start_time
            process = psutil.Process(os.getpid())
            end_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_delta = end_memory - self.start_memory
            
            return {
                'elapsed_time_seconds': elapsed_time,
                'memory_usage_mb': end_memory,
                'memory_delta_mb': memory_delta
            }
    
    return PerformanceMonitor() 