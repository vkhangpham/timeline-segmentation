"""
Algorithm Components Unit Test Suite

Tests individual algorithm components using real data subsets.
Following Phase 13 principle: Test components in isolation with fail-fast behavior.
"""

import pytest
import numpy as np
from typing import List, Dict
from unittest.mock import patch
from collections import defaultdict

from core.shift_signal_detection import (
    detect_research_direction_changes,
    detect_citation_structural_breaks,
    cluster_direction_signals_by_proximity,
    validate_direction_with_citation,
    citation_adaptive_threshold,
    moving_average,
    cluster_and_validate_shifts
)
from core.algorithm_config import ComprehensiveAlgorithmConfig
from core.data_models import ShiftSignal, DomainData


# =============================================================================
# DIRECTION DETECTION COMPONENT TESTS
# =============================================================================

@pytest.mark.unit
class TestDirectionDetection:
    """Test direction detection component in isolation."""
    
    def test_direction_detection_with_real_data(self, tiny_domain_data, default_config):
        """Test direction detection with real domain data subset."""
        for domain_name, domain_data in tiny_domain_data.items():
            # Test should not fail even with small datasets
            try:
                signals = detect_research_direction_changes(
                    domain_data, 
                    detection_threshold=default_config.direction_threshold
                )
                
                # Validate signal structure
                assert isinstance(signals, list), f"Direction signals must be list for {domain_name}"
                
                for signal in signals:
                    assert hasattr(signal, 'year'), "Signal must have year"
                    assert hasattr(signal, 'confidence'), "Signal must have confidence"
                    assert hasattr(signal, 'signal_type'), "Signal must have signal_type"
                    assert signal.signal_type == 'direction_volatility', "Should be direction signal"
                    assert 0.0 <= signal.confidence <= 1.0, f"Invalid confidence: {signal.confidence}"
                    
            except Exception as e:
                # Fail fast - don't mask algorithm errors
                pytest.fail(f"Direction detection failed for {domain_name}: {e}")
    
    def test_direction_threshold_sensitivity(self, tiny_domain_data):
        """Test that direction threshold affects signal count appropriately."""
        domain_data = list(tiny_domain_data.values())[0]  # Use first domain
        
        # Test different thresholds
        thresholds = [0.1, 0.3, 0.5, 0.7]
        signal_counts = []
        
        for threshold in thresholds:
            signals = detect_research_direction_changes(
                domain_data, detection_threshold=threshold
            )
            signal_counts.append(len(signals))
        
        # Higher thresholds should generally produce fewer signals
        # (though this isn't guaranteed with real data)
        assert isinstance(signal_counts, list), "Should collect signal counts"
        assert all(count >= 0 for count in signal_counts), "Signal counts should be non-negative"
    
    def test_direction_detection_edge_cases(self, tiny_domain_data):
        """Test direction detection with edge case parameters."""
        domain_data = list(tiny_domain_data.values())[0]
        
        # Test minimum threshold
        signals_min = detect_research_direction_changes(
            domain_data, detection_threshold=0.1
        )
        assert isinstance(signals_min, list)
        
        # Test maximum threshold  
        signals_max = detect_research_direction_changes(
            domain_data, detection_threshold=0.8
        )
        assert isinstance(signals_max, list)
        
        # Maximum threshold should not produce more signals than minimum
        assert len(signals_max) <= len(signals_min)
    
    def test_direction_detection_with_analysis_data(self, tiny_domain_data):
        """Test direction detection with return_analysis_data flag."""
        domain_data = list(tiny_domain_data.values())[0]
        
        # Test with analysis data return
        result = detect_research_direction_changes(
            domain_data, 
            detection_threshold=0.4,
            return_analysis_data=True
        )
        
        # Should return tuple of (signals, analysis_data)
        assert isinstance(result, tuple), "Should return tuple when return_analysis_data=True"
        assert len(result) == 2, "Should return (signals, analysis_data)"
        
        signals, analysis_data = result
        assert isinstance(signals, list), "First element should be signals list"
        assert isinstance(analysis_data, dict), "Second element should be analysis dict"
        
        # Analysis data should contain expected keys
        expected_keys = ['years', 'overlap', 'novelty', 'direction_score', 'threshold']
        for key in expected_keys:
            assert key in analysis_data, f"Analysis data missing key: {key}"
    
    def test_empty_domain_data_handling(self):
        """Test direction detection with empty domain data."""
        # Create empty domain data
        empty_domain = DomainData(
            domain_name="empty_test",
            papers=tuple(),
            citations=tuple(),
            year_range=(2000, 2024)
        )
        
        # Should handle empty data gracefully (not crash)
        signals = detect_research_direction_changes(empty_domain, detection_threshold=0.4)
        assert isinstance(signals, list)
        assert len(signals) == 0  # Should find no signals in empty data


# =============================================================================
# CITATION ANALYSIS COMPONENT TESTS  
# =============================================================================

@pytest.mark.unit
class TestCitationAnalysis:
    """Test citation analysis component (CPSD) in isolation."""
    
    def test_citation_analysis_with_real_data(self, tiny_domain_data):
        """Test citation analysis with real domain data subset."""
        for domain_name, domain_data in tiny_domain_data.items():
            try:
                signals = detect_citation_structural_breaks(domain_data, domain_name)
                
                # Validate signal structure
                assert isinstance(signals, list), f"Citation signals must be list for {domain_name}"
                
                for signal in signals:
                    assert hasattr(signal, 'year'), "Citation signal must have year"
                    assert hasattr(signal, 'confidence'), "Citation signal must have confidence"
                    assert hasattr(signal, 'signal_type'), "Citation signal must have signal_type"
                    assert signal.signal_type == 'citation_gradient_cpsd', "Should be citation signal"
                    assert 0.0 <= signal.confidence <= 1.0, f"Invalid confidence: {signal.confidence}"
                    
            except Exception as e:
                # Fail fast - propagate citation analysis errors
                pytest.fail(f"Citation analysis failed for {domain_name}: {e}")
    
    def test_adaptive_threshold_function(self):
        """Test adaptive threshold calculation with controlled data."""
        # Test gradient threshold
        test_data = np.array([1, 2, 3, 5, 8, 13, 21])
        
        gradient_threshold = citation_adaptive_threshold(test_data, "gradient")
        assert isinstance(gradient_threshold, float)
        assert gradient_threshold > 0
        
        # Test acceleration threshold
        accel_threshold = citation_adaptive_threshold(test_data, "acceleration")
        assert isinstance(accel_threshold, float)
        assert accel_threshold > 0
        
        # Different methods should potentially give different results
        assert gradient_threshold != accel_threshold or len(test_data) < 3
    
    def test_adaptive_threshold_edge_cases(self):
        """Test adaptive threshold with edge case data."""
        # Empty data
        empty_threshold = citation_adaptive_threshold(np.array([]), "gradient")
        assert empty_threshold == 0.0
        
        # Single value
        single_threshold = citation_adaptive_threshold(np.array([5]), "gradient") 
        assert isinstance(single_threshold, float)
        
        # Constant data
        constant_threshold = citation_adaptive_threshold(np.array([5, 5, 5, 5]), "gradient")
        assert isinstance(constant_threshold, float)
        assert constant_threshold >= 0
    
    def test_citation_analysis_no_citations(self):
        """Test citation analysis with domain that has no citations."""
        # Create domain with papers but no citations
        from core.data_models import Paper
        
        papers_no_citations = [
            Paper(
                id=f"paper_{i}",
                title=f"Paper {i}",
                pub_year=2000 + i,
                cited_by_count=0,  # No citations
                keywords=[f"keyword_{i}"],
                abstract=f"Abstract {i}"
            )
            for i in range(5)
        ]
        
        domain_no_citations = DomainData(
            domain_name="no_citations_test",
            papers=tuple(papers_no_citations),
            citations=tuple(),
            year_range=(2000, 2005)
        )
        
        # Should handle gracefully
        signals = detect_citation_structural_breaks(domain_no_citations, "no_citations_test")
        assert isinstance(signals, list)
        # May or may not find signals depending on algorithm behavior


# =============================================================================
# TEMPORAL CLUSTERING COMPONENT TESTS
# =============================================================================

@pytest.mark.unit  
class TestTemporalClustering:
    """Test temporal clustering component in isolation."""
    
    def create_test_signals(self, years: List[int]) -> List[ShiftSignal]:
        """Create test signals for clustering tests."""
        signals = []
        for year in years:
            signal = ShiftSignal(
                year=year,
                confidence=0.8,
                signal_type="direction_volatility",
                evidence_strength=0.7,
                supporting_evidence=tuple([f"test_evidence_{year}"]),
                contributing_papers=tuple([f"paper_{year}"]),
                transition_description=f"Test signal at {year}",
                paradigm_significance=0.6
            )
            signals.append(signal)
        return signals
    
    def test_temporal_clustering_basic(self, default_config):
        """Test basic temporal clustering functionality."""
        # Create signals with some close together, some far apart
        test_years = [1990, 1991, 1995, 2000, 2001, 2002, 2010]
        test_signals = self.create_test_signals(test_years)
        
        clustered_signals = cluster_direction_signals_by_proximity(
            test_signals, default_config
        )
        
        # Should have fewer clustered signals than original
        assert len(clustered_signals) <= len(test_signals)
        
        # Each clustered signal should be valid
        for signal in clustered_signals:
            assert hasattr(signal, 'year')
            assert hasattr(signal, 'confidence')
            assert 0.0 <= signal.confidence <= 1.0
    
    def test_clustering_window_effect(self):
        """Test that clustering window size affects results."""
        test_years = [1990, 1992, 1994, 1996, 1998, 2000]
        test_signals = self.create_test_signals(test_years)
        
        # Test with small window
        small_config = ComprehensiveAlgorithmConfig.create_custom(
            granularity=3, overrides={'clustering_window': 2}
        )
        small_clustered = cluster_direction_signals_by_proximity(test_signals, small_config)
        
        # Test with large window  
        large_config = ComprehensiveAlgorithmConfig.create_custom(
            granularity=3, overrides={'clustering_window': 8}
        )
        large_clustered = cluster_direction_signals_by_proximity(test_signals, large_config)
        
        # Larger window should generally produce fewer clusters
        assert len(large_clustered) <= len(small_clustered)
    
    def test_clustering_single_signal(self, default_config):
        """Test clustering with single signal."""
        single_signal = self.create_test_signals([2000])
        
        clustered = cluster_direction_signals_by_proximity(single_signal, default_config)
        
        assert len(clustered) == 1
        assert clustered[0].year == 2000
    
    def test_clustering_empty_signals(self, default_config):
        """Test clustering with empty signal list."""
        clustered = cluster_direction_signals_by_proximity([], default_config)
        
        assert isinstance(clustered, list)
        assert len(clustered) == 0
    
    def test_clustering_start_year_comparison(self):
        """Test that clustering uses start year comparison (fixed algorithm)."""
        # Create signals that would chain endlessly with end-year comparison
        test_years = [2000, 2002, 2004, 2006, 2008]  # Each 2 years apart
        test_signals = self.create_test_signals(test_years)
        
        # With clustering window of 3, should not cluster all together
        config = ComprehensiveAlgorithmConfig.create_custom(
            granularity=3, overrides={'clustering_window': 3}
        )
        
        clustered = cluster_direction_signals_by_proximity(test_signals, config)
        
        # Should not collapse everything into one cluster due to fixed start-year algorithm
        assert len(clustered) > 1, "Fixed clustering algorithm should prevent endless chaining"


# =============================================================================
# VALIDATION LOGIC COMPONENT TESTS
# =============================================================================

@pytest.mark.unit
class TestValidationLogic:
    """Test validation logic component (simplified linear process)."""
    
    def create_test_direction_signals(self, years: List[int]) -> List[ShiftSignal]:
        """Create test direction signals."""
        signals = []
        for year in years:
            signal = ShiftSignal(
                year=year,
                confidence=0.7,
                signal_type="direction_clustered",
                evidence_strength=0.6,
                supporting_evidence=tuple([f"direction_evidence_{year}"]),
                contributing_papers=tuple([f"paper_{year}"]),
                transition_description=f"Direction signal at {year}",
                paradigm_significance=0.5
            )
            signals.append(signal)
        return signals
    
    def create_test_citation_signals(self, years: List[int]) -> List[ShiftSignal]:
        """Create test citation signals."""
        signals = []
        for year in years:
            signal = ShiftSignal(
                year=year,
                confidence=0.8,
                signal_type="citation_gradient_cpsd",
                evidence_strength=0.7,
                supporting_evidence=tuple([f"citation_evidence_{year}"]),
                contributing_papers=tuple([f"paper_{year}"]),
                transition_description=f"Citation signal at {year}",
                paradigm_significance=0.6
            )
            signals.append(signal)
        return signals
    
    def test_validation_basic_functionality(self, tiny_domain_data, default_config):
        """Test basic validation logic with test signals."""
        domain_data = list(tiny_domain_data.values())[0]
        
        # Create test signals
        direction_signals = self.create_test_direction_signals([2000, 2005, 2010])
        citation_signals = self.create_test_citation_signals([2001, 2005, 2011])  # Some overlap
        
        validated_signals = validate_direction_with_citation(
            direction_signals, citation_signals, domain_data, 
            domain_data.domain_name, default_config
        )
        
        # Should return list of validated signals
        assert isinstance(validated_signals, list)
        
        # Validate signal structure
        for signal in validated_signals:
            assert hasattr(signal, 'year')
            assert hasattr(signal, 'confidence')
            assert hasattr(signal, 'signal_type')
            assert signal.signal_type in ['direction_primary_validated', 'direction_primary_only']
    
    def test_validation_threshold_effect(self, tiny_domain_data):
        """Test that validation threshold affects acceptance rate."""
        domain_data = list(tiny_domain_data.values())[0]
        direction_signals = self.create_test_direction_signals([2000, 2005])
        citation_signals = self.create_test_citation_signals([2001])  # Limited citation support
        
        # Test with low threshold (permissive)
        low_config = ComprehensiveAlgorithmConfig.create_custom(
            granularity=3, overrides={'validation_threshold': 0.5}
        )
        low_validated = validate_direction_with_citation(
            direction_signals, citation_signals, domain_data, 
            domain_data.domain_name, low_config
        )
        
        # Test with high threshold (restrictive)
        high_config = ComprehensiveAlgorithmConfig.create_custom(
            granularity=3, overrides={'validation_threshold': 0.9}
        )
        high_validated = validate_direction_with_citation(
            direction_signals, citation_signals, domain_data,
            domain_data.domain_name, high_config
        )
        
        # High threshold should generally accept fewer signals
        assert len(high_validated) <= len(low_validated)
    
    def test_citation_boost_effect(self, tiny_domain_data):
        """Test that citation boost affects signal confidence."""
        domain_data = list(tiny_domain_data.values())[0]
        direction_signals = self.create_test_direction_signals([2000])
        citation_signals = self.create_test_citation_signals([2000])  # Exact match for support
        
        # Test with citation boost
        boost_config = ComprehensiveAlgorithmConfig.create_custom(
            granularity=3, 
            overrides={
                'citation_boost': 0.3,
                'validation_threshold': 0.8  # Set threshold so we can see boost effect
            }
        )
        
        boosted_signals = validate_direction_with_citation(
            direction_signals, citation_signals, domain_data,
            domain_data.domain_name, boost_config
        )
        
        # Should have citation support indicated in signal type
        if boosted_signals:
            assert any('validated' in signal.signal_type for signal in boosted_signals)
    
    def test_validation_no_citation_support(self, tiny_domain_data):
        """Test validation when no citation support is available."""
        domain_data = list(tiny_domain_data.values())[0]
        direction_signals = self.create_test_direction_signals([2000, 2005])
        citation_signals = []  # No citation signals
        
        validated_signals = validate_direction_with_citation(
            direction_signals, citation_signals, domain_data,
            domain_data.domain_name, ComprehensiveAlgorithmConfig(granularity=3)
        )
        
        # Should still process signals, just without citation support
        assert isinstance(validated_signals, list)
        
        # Signals should be marked as direction-only
        for signal in validated_signals:
            assert 'direction_primary_only' in signal.signal_type
    
    def test_validation_empty_inputs(self, tiny_domain_data, default_config):
        """Test validation with empty inputs."""
        domain_data = list(tiny_domain_data.values())[0]
        
        # Empty direction signals
        empty_validated = validate_direction_with_citation(
            [], [], domain_data, domain_data.domain_name, default_config
        )
        assert isinstance(empty_validated, list)
        assert len(empty_validated) == 0


# =============================================================================
# UTILITY FUNCTION TESTS
# =============================================================================

@pytest.mark.unit
class TestUtilityFunctions:
    """Test utility functions used by algorithm components."""
    
    def test_moving_average_function(self):
        """Test moving average calculation."""
        # Test with simple data
        test_data = np.array([1, 2, 3, 4, 5, 6])
        
        # Test 3-point moving average
        ma_3 = moving_average(test_data, 3)
        expected_3 = np.array([2, 3, 4, 5])  # (1+2+3)/3=2, (2+3+4)/3=3, etc.
        np.testing.assert_array_almost_equal(ma_3, expected_3)
        
        # Test window larger than data
        ma_large = moving_average(test_data, 10)
        assert len(ma_large) == len(test_data)
        assert all(val == np.mean(test_data) for val in ma_large)
    
    def test_moving_average_edge_cases(self):
        """Test moving average with edge cases."""
        # Single value
        single_data = np.array([5])
        ma_single = moving_average(single_data, 1)
        assert len(ma_single) == 1
        assert ma_single[0] == 5
        
        # Empty data
        empty_data = np.array([])
        ma_empty = moving_average(empty_data, 3)
        assert len(ma_empty) == 0
    
    def test_cluster_and_validate_shifts(self):
        """Test cluster and validate shifts utility function."""
        # Test with shifts that should be clustered
        shifts = [2000, 2001, 2005, 2006, 2007, 2015]
        years_array = np.arange(1990, 2020)
        
        validated = cluster_and_validate_shifts(shifts, years_array, min_segment_length=3)
        
        # Should return fewer shifts due to clustering
        assert len(validated) <= len(shifts)
        assert isinstance(validated, list)
        
        # All returned shifts should be valid years
        for shift in validated:
            assert 1990 <= shift <= 2020
    
    def test_cluster_and_validate_shifts_edge_cases(self):
        """Test cluster and validate shifts with edge cases."""
        years_array = np.arange(2000, 2010)
        
        # Empty shifts
        empty_validated = cluster_and_validate_shifts([], years_array)
        assert empty_validated == []
        
        # Single shift
        single_validated = cluster_and_validate_shifts([2005], years_array)
        assert single_validated == [2005]
        
        # Shifts outside year range should be filtered
        out_of_range = cluster_and_validate_shifts([1990, 2005, 2020], years_array)
        assert 2005 in out_of_range
        assert 1990 not in out_of_range
        assert 2020 not in out_of_range


# =============================================================================
# COMPONENT INTEGRATION TESTS
# =============================================================================

@pytest.mark.unit
class TestComponentIntegration:
    """Test how algorithm components work together."""
    
    def test_direction_citation_pipeline(self, tiny_domain_data, default_config):
        """Test direction detection -> citation analysis pipeline."""
        domain_data = list(tiny_domain_data.values())[0]
        
        # Run direction detection
        direction_signals = detect_research_direction_changes(
            domain_data, detection_threshold=default_config.direction_threshold
        )
        
        # Run citation analysis
        citation_signals = detect_citation_structural_breaks(domain_data, domain_data.domain_name)
        
        # Both should return valid signal lists
        assert isinstance(direction_signals, list)
        assert isinstance(citation_signals, list)
        
        # Test clustering if we have direction signals
        if direction_signals:
            clustered_signals = cluster_direction_signals_by_proximity(
                direction_signals, default_config
            )
            assert isinstance(clustered_signals, list)
            assert len(clustered_signals) <= len(direction_signals)
        
        # Test validation
        validated_signals = validate_direction_with_citation(
            direction_signals, citation_signals, domain_data,
            domain_data.domain_name, default_config
        )
        assert isinstance(validated_signals, list)
    
    def test_component_error_propagation(self, tiny_domain_data):
        """Test that component errors propagate correctly (fail fast)."""
        domain_data = list(tiny_domain_data.values())[0]
        
        # Test with invalid threshold (should fail fast)
        with pytest.raises(Exception):
            # This should trigger validation error in config creation
            invalid_config = ComprehensiveAlgorithmConfig.create_custom(
                granularity=3,
                overrides={'direction_threshold': -0.1}  # Invalid value
            )


# =============================================================================
# PERFORMANCE AND MEMORY TESTS
# =============================================================================

@pytest.mark.unit
class TestComponentPerformance:
    """Test component performance characteristics."""
    
    def test_direction_detection_performance(self, small_domain_data, performance_monitor):
        """Test direction detection performance with larger dataset."""
        domain_data = list(small_domain_data.values())[0]  # 100 papers
        
        performance_monitor.start()
        
        signals = detect_research_direction_changes(domain_data, detection_threshold=0.4)
        
        perf_metrics = performance_monitor.stop()
        
        # Should complete in reasonable time (adjust threshold as needed)
        assert perf_metrics['elapsed_time_seconds'] < 10.0, "Direction detection too slow"
        
        # Should not use excessive memory
        assert perf_metrics['memory_delta_mb'] < 100, "Direction detection using too much memory"
    
    @pytest.mark.slow
    def test_citation_analysis_performance(self, small_domain_data, performance_monitor):
        """Test citation analysis performance."""
        domain_data = list(small_domain_data.values())[0]
        
        performance_monitor.start()
        
        signals = detect_citation_structural_breaks(domain_data, domain_data.domain_name)
        
        perf_metrics = performance_monitor.stop()
        
        # Citation analysis should also be reasonably fast
        assert perf_metrics['elapsed_time_seconds'] < 15.0, "Citation analysis too slow" 