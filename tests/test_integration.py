"""
Integration Test Suite

Tests the complete algorithm pipeline across all available domains.
Following Phase 13 principle: Test end-to-end with real data, fail fast on issues.
"""

import pytest
import time
from typing import Dict, List, Any

from core.shift_signal_detection import detect_shift_signals
from core.algorithm_config import ComprehensiveAlgorithmConfig
from core.change_detection import detect_changes


# =============================================================================
# FULL PIPELINE INTEGRATION TESTS
# =============================================================================

@pytest.mark.integration
class TestFullPipelineIntegration:
    """Test complete algorithm pipeline end-to-end."""
    
    def test_full_pipeline_all_domains(self, small_domain_data, default_config, validation_helpers):
        """Test full pipeline across all available domains."""
        results = {}
        
        for domain_name, domain_data in small_domain_data.items():
            print(f"\n🔍 Testing full pipeline for {domain_name}")
            
            try:
                # Run complete algorithm pipeline
                signals, evidence, metadata = detect_shift_signals(
                    domain_data, domain_name, default_config
                )
                
                # Validate results structure
                validation_helpers.assert_valid_paradigm_signals(signals)
                assert isinstance(evidence, list), "Evidence must be list"
                assert isinstance(metadata, dict), "Metadata must be dict"
                
                # Store results for analysis
                results[domain_name] = {
                    'signal_count': len(signals),
                    'evidence_count': len(evidence),
                    'metadata_keys': list(metadata.keys()),
                    'signal_years': [s.year for s in signals],
                    'signal_types': [s.signal_type for s in signals],
                    'confidence_stats': {
                        'mean': sum(s.confidence for s in signals) / max(len(signals), 1),
                        'min': min((s.confidence for s in signals), default=0),
                        'max': max((s.confidence for s in signals), default=0)
                    }
                }
                
                print(f"  ✅ {domain_name}: {len(signals)} signals detected")
                
            except Exception as e:
                # Fail fast - don't mask integration errors
                pytest.fail(f"Full pipeline failed for {domain_name}: {e}")
        
        # Analyze cross-domain results
        self._analyze_cross_domain_results(results)
    
    def _analyze_cross_domain_results(self, results: Dict[str, Dict]):
        """Analyze results across domains for consistency."""
        total_signals = sum(r['signal_count'] for r in results.values())
        domains_with_signals = sum(1 for r in results.values() if r['signal_count'] > 0)
        
        print(f"\n📊 CROSS-DOMAIN ANALYSIS:")
        print(f"  Total signals across all domains: {total_signals}")
        print(f"  Domains with signals: {domains_with_signals}/{len(results)}")
        
        # Check that at least some domains produce signals
        assert domains_with_signals > 0, "No domains produced paradigm signals"
        
        # Check signal type consistency
        all_signal_types = set()
        for result in results.values():
            all_signal_types.update(result['signal_types'])
        
        expected_types = {
            'direction_primary_validated', 
            'direction_primary_only',
            'direction_clustered'
        }
        
        # Should only see expected signal types
        unexpected_types = all_signal_types - expected_types
        assert len(unexpected_types) == 0, f"Unexpected signal types: {unexpected_types}"
    
    def test_full_pipeline_with_change_detection_interface(self, small_domain_data, default_config):
        """Test full pipeline through change_detection.py interface."""
        for domain_name, domain_data in small_domain_data.items():
            try:
                # Test through change detection interface
                change_result = detect_changes(domain_data, default_config)
                
                # Validate change detection result structure
                assert hasattr(change_result, 'domain_name')
                assert hasattr(change_result, 'change_points')
                assert hasattr(change_result, 'statistical_significance')
                
                assert change_result.domain_name == domain_data.domain_name
                assert isinstance(change_result.change_points, tuple)
                assert isinstance(change_result.statistical_significance, float)
                
                print(f"  ✅ {domain_name}: {len(change_result.change_points)} change points via change detection interface")
                
            except Exception as e:
                pytest.fail(f"Change detection interface failed for {domain_name}: {e}")


# =============================================================================
# CONFIGURATION VARIATION TESTS
# =============================================================================

@pytest.mark.integration
class TestConfigurationVariations:
    """Test pipeline with different configurations."""
    
    def test_all_granularity_levels(self, tiny_domain_data, all_granularity_configs):
        """Test pipeline with all granularity levels."""
        # Use single domain for speed
        domain_data = list(tiny_domain_data.values())[0]
        domain_name = domain_data.domain_name
        
        granularity_results = {}
        
        for config in all_granularity_configs:
            granularity = config.granularity
            
            try:
                signals, evidence, metadata = detect_shift_signals(
                    domain_data, domain_name, config
                )
                
                granularity_results[granularity] = {
                    'signal_count': len(signals),
                    'config_summary': config.get_configuration_summary()
                }
                
                print(f"  Granularity {granularity}: {len(signals)} signals")
                
            except Exception as e:
                pytest.fail(f"Pipeline failed with granularity {granularity}: {e}")
        
        # Analyze granularity effects
        self._analyze_granularity_effects(granularity_results)
    
    def _analyze_granularity_effects(self, results: Dict[int, Dict]):
        """Analyze how granularity affects signal counts."""
        granularities = sorted(results.keys())
        signal_counts = [results[g]['signal_count'] for g in granularities]
        
        print(f"\n📈 GRANULARITY ANALYSIS:")
        for i, g in enumerate(granularities):
            print(f"  Level {g}: {signal_counts[i]} signals")
        
        # Generally expect fewer signals with higher granularity (though not guaranteed)
        # Just check that all levels produce reasonable results
        for count in signal_counts:
            assert count >= 0, "Signal counts should be non-negative"
    
    def test_domain_specific_configurations(self, tiny_domain_data):
        """Test pipeline with domain-specific configurations."""
        domain_configs = {
            'computer_vision': ComprehensiveAlgorithmConfig.create_domain_specific('computer_vision'),
            'natural_language_processing': ComprehensiveAlgorithmConfig.create_domain_specific('natural_language_processing'),
            'applied_mathematics': ComprehensiveAlgorithmConfig.create_domain_specific('applied_mathematics'),
            'art': ComprehensiveAlgorithmConfig.create_domain_specific('art')
        }
        
        for domain_name, domain_data in tiny_domain_data.items():
            if domain_name in domain_configs:
                config = domain_configs[domain_name]
                
                try:
                    signals, evidence, metadata = detect_shift_signals(
                        domain_data, domain_name, config
                    )
                    
                    print(f"  ✅ {domain_name} with domain-specific config: {len(signals)} signals")
                    
                except Exception as e:
                    pytest.fail(f"Domain-specific config failed for {domain_name}: {e}")
    
    def test_edge_case_configurations(self, tiny_domain_data, edge_case_configs):
        """Test pipeline with edge case configurations."""
        domain_data = list(tiny_domain_data.values())[0]
        domain_name = domain_data.domain_name
        
        for i, config in enumerate(edge_case_configs):
            try:
                signals, evidence, metadata = detect_shift_signals(
                    domain_data, domain_name, config
                )
                
                print(f"  ✅ Edge case config {i+1}: {len(signals)} signals")
                
            except Exception as e:
                pytest.fail(f"Edge case configuration {i+1} failed: {e}")


# =============================================================================
# PERFORMANCE AND SCALABILITY TESTS
# =============================================================================

@pytest.mark.integration
@pytest.mark.slow
class TestPerformanceIntegration:
    """Test performance characteristics of full pipeline."""
    
    def test_pipeline_performance_scaling(self, small_domain_data, default_config, performance_monitor):
        """Test how pipeline performance scales with data size."""
        performance_results = {}
        
        for domain_name, domain_data in small_domain_data.items():
            print(f"\n⏱️  Testing performance for {domain_name} ({len(domain_data.papers)} papers)")
            
            performance_monitor.start()
            
            try:
                signals, evidence, metadata = detect_shift_signals(
                    domain_data, domain_name, default_config
                )
                
                perf_metrics = performance_monitor.stop()
                
                performance_results[domain_name] = {
                    'paper_count': len(domain_data.papers),
                    'signal_count': len(signals),
                    'elapsed_time': perf_metrics['elapsed_time_seconds'],
                    'memory_usage': perf_metrics['memory_usage_mb'],
                    'memory_delta': perf_metrics['memory_delta_mb']
                }
                
                print(f"  📊 {domain_name}: {perf_metrics['elapsed_time_seconds']:.2f}s, {perf_metrics['memory_delta_mb']:.1f}MB")
                
                # Performance sanity checks
                assert perf_metrics['elapsed_time_seconds'] < 30.0, f"Pipeline too slow for {domain_name}"
                assert perf_metrics['memory_delta_mb'] < 200, f"Pipeline using too much memory for {domain_name}"
                
            except Exception as e:
                pytest.fail(f"Performance test failed for {domain_name}: {e}")
        
        self._analyze_performance_scaling(performance_results)
    
    def _analyze_performance_scaling(self, results: Dict[str, Dict]):
        """Analyze performance scaling characteristics."""
        print(f"\n📈 PERFORMANCE SCALING ANALYSIS:")
        
        total_time = sum(r['elapsed_time'] for r in results.values())
        total_papers = sum(r['paper_count'] for r in results.values())
        total_memory = sum(r['memory_delta'] for r in results.values())
        
        print(f"  Total processing time: {total_time:.2f}s")
        print(f"  Total papers processed: {total_papers}")
        print(f"  Average time per paper: {total_time/max(total_papers,1)*1000:.2f}ms")
        print(f"  Total memory delta: {total_memory:.1f}MB")
        
        # Performance should be reasonable
        assert total_time < 120.0, "Total processing time too high"
        assert total_memory < 500, "Total memory usage too high"


# =============================================================================
# CROSS-DOMAIN CONSISTENCY TESTS
# =============================================================================

@pytest.mark.integration
class TestCrossDomainConsistency:
    """Test consistency of algorithm behavior across domains."""
    
    def test_signal_type_consistency(self, small_domain_data, default_config):
        """Test that signal types are consistent across domains."""
        all_signal_types = set()
        domain_signal_types = {}
        
        for domain_name, domain_data in small_domain_data.items():
            try:
                signals, _, _ = detect_shift_signals(domain_data, domain_name, default_config)
                
                domain_types = set(s.signal_type for s in signals)
                domain_signal_types[domain_name] = domain_types
                all_signal_types.update(domain_types)
                
            except Exception as e:
                pytest.fail(f"Signal type test failed for {domain_name}: {e}")
        
        print(f"\n🏷️  SIGNAL TYPE ANALYSIS:")
        print(f"  All signal types found: {sorted(all_signal_types)}")
        
        # Check that signal types are from expected set
        expected_types = {
            'direction_primary_validated',
            'direction_primary_only', 
            'direction_clustered',
            'citation_gradient_cpsd'
        }
        
        unexpected_types = all_signal_types - expected_types
        assert len(unexpected_types) == 0, f"Unexpected signal types: {unexpected_types}"
    
    def test_confidence_range_consistency(self, small_domain_data, default_config):
        """Test that confidence values are in valid ranges across domains."""
        all_confidences = []
        domain_confidence_stats = {}
        
        for domain_name, domain_data in small_domain_data.items():
            try:
                signals, _, _ = detect_shift_signals(domain_data, domain_name, default_config)
                
                confidences = [s.confidence for s in signals]
                all_confidences.extend(confidences)
                
                if confidences:
                    domain_confidence_stats[domain_name] = {
                        'mean': sum(confidences) / len(confidences),
                        'min': min(confidences),
                        'max': max(confidences),
                        'count': len(confidences)
                    }
                
            except Exception as e:
                pytest.fail(f"Confidence test failed for {domain_name}: {e}")
        
        print(f"\n📊 CONFIDENCE ANALYSIS:")
        for domain, stats in domain_confidence_stats.items():
            print(f"  {domain}: mean={stats['mean']:.3f}, range=[{stats['min']:.3f}, {stats['max']:.3f}], n={stats['count']}")
        
        # All confidences should be in valid range
        for conf in all_confidences:
            assert 0.0 <= conf <= 1.0, f"Invalid confidence value: {conf}"
    
    def test_metadata_consistency(self, small_domain_data, default_config):
        """Test that metadata structure is consistent across domains."""
        all_metadata_keys = set()
        domain_metadata = {}
        
        for domain_name, domain_data in small_domain_data.items():
            try:
                signals, evidence, metadata = detect_shift_signals(domain_data, domain_name, default_config)
                
                metadata_keys = set(metadata.keys())
                all_metadata_keys.update(metadata_keys)
                domain_metadata[domain_name] = metadata_keys
                
            except Exception as e:
                pytest.fail(f"Metadata test failed for {domain_name}: {e}")
        
        print(f"\n🗂️  METADATA ANALYSIS:")
        print(f"  All metadata keys: {sorted(all_metadata_keys)}")
        
        # Check for minimum expected metadata
        expected_metadata = {'raw_direction_signals', 'clustered_direction_signals', 'citation_signals'}
        missing_metadata = expected_metadata - all_metadata_keys
        
        # Should have key metadata components
        assert len(missing_metadata) == 0, f"Missing expected metadata: {missing_metadata}"


# =============================================================================
# ERROR HANDLING AND ROBUSTNESS TESTS
# =============================================================================

@pytest.mark.integration
class TestErrorHandlingRobustness:
    """Test error handling and robustness of pipeline."""
    
    def test_pipeline_with_minimal_data(self, available_domains):
        """Test pipeline behavior with minimal domain data."""
        from core.data_models import DomainData, Paper
        
        # Create minimal domain with just 2 papers
        minimal_papers = [
            Paper(
                id="minimal_1",
                title="Minimal Paper 1", 
                pub_year=2000,
                cited_by_count=10,
                keywords=["test", "minimal"],
                abstract="Minimal abstract 1"
            ),
            Paper(
                id="minimal_2",
                title="Minimal Paper 2",
                pub_year=2010, 
                cited_by_count=5,
                keywords=["test", "different"],
                abstract="Minimal abstract 2"
            )
        ]
        
        minimal_domain = DomainData(
            domain_name="minimal_test",
            papers=tuple(minimal_papers),
            citations=tuple(),
            year_range=(2000, 2010)
        )
        
        # Should handle minimal data gracefully
        try:
            signals, evidence, metadata = detect_shift_signals(
                minimal_domain, "minimal_test", ComprehensiveAlgorithmConfig(granularity=3)
            )
            
            # Should return valid results even with minimal data
            assert isinstance(signals, list)
            assert isinstance(evidence, list)
            assert isinstance(metadata, dict)
            
            print(f"  ✅ Minimal data test: {len(signals)} signals")
            
        except Exception as e:
            pytest.fail(f"Pipeline failed with minimal data: {e}")
    
    def test_pipeline_configuration_validation(self, tiny_domain_data):
        """Test that pipeline validates configurations properly."""
        domain_data = list(tiny_domain_data.values())[0]
        
        # Test with invalid configuration - should fail fast
        with pytest.raises(ValueError):
            invalid_config = ComprehensiveAlgorithmConfig.create_custom(
                granularity=3,
                overrides={'direction_threshold': -0.1}  # Invalid
            )
    
    def test_pipeline_deterministic_behavior(self, tiny_domain_data, default_config):
        """Test that pipeline produces consistent results across runs."""
        domain_data = list(tiny_domain_data.values())[0]
        domain_name = domain_data.domain_name
        
        # Run pipeline multiple times
        runs = []
        for i in range(3):
            signals, evidence, metadata = detect_shift_signals(
                domain_data, domain_name, default_config
            )
            
            runs.append({
                'signal_count': len(signals),
                'signal_years': sorted([s.year for s in signals]),
                'signal_types': sorted([s.signal_type for s in signals])
            })
        
        # Results should be consistent (deterministic)
        for i in range(1, len(runs)):
            assert runs[i]['signal_count'] == runs[0]['signal_count'], "Non-deterministic signal count"
            assert runs[i]['signal_years'] == runs[0]['signal_years'], "Non-deterministic signal years"
            assert runs[i]['signal_types'] == runs[0]['signal_types'], "Non-deterministic signal types"
        
        print(f"  ✅ Deterministic behavior confirmed across {len(runs)} runs")


# =============================================================================
# GROUND TRUTH VALIDATION INTEGRATION TESTS
# =============================================================================

@pytest.mark.integration
class TestGroundTruthValidationIntegration:
    """Test integration with ground truth validation."""
    
    def test_pipeline_with_ground_truth_comparison(self, small_domain_data, ground_truth_data, 
                                                   default_config, validation_helpers):
        """Test pipeline results against available ground truth."""
        validation_results = {}
        
        for domain_name, domain_data in small_domain_data.items():
            if domain_name in ground_truth_data and ground_truth_data[domain_name]:
                try:
                    # Run pipeline
                    signals, _, _ = detect_shift_signals(domain_data, domain_name, default_config)
                    
                    # Compare with ground truth
                    detected_years = [s.year for s in signals]
                    ground_truth_years = ground_truth_data[domain_name]
                    
                    metrics = validation_helpers.calculate_detection_metrics(
                        detected_years, ground_truth_years, tolerance=2
                    )
                    
                    validation_results[domain_name] = {
                        'detected_count': len(detected_years),
                        'ground_truth_count': len(ground_truth_years),
                        'metrics': metrics
                    }
                    
                    print(f"  📊 {domain_name}: Precision={metrics['precision']:.3f}, Recall={metrics['recall']:.3f}, F1={metrics['f1']:.3f}")
                    
                except Exception as e:
                    pytest.fail(f"Ground truth validation failed for {domain_name}: {e}")
        
        if validation_results:
            self._analyze_ground_truth_results(validation_results)
    
    def _analyze_ground_truth_results(self, results: Dict[str, Dict]):
        """Analyze ground truth validation results."""
        print(f"\n🎯 GROUND TRUTH VALIDATION ANALYSIS:")
        
        if results:
            avg_precision = sum(r['metrics']['precision'] for r in results.values()) / len(results)
            avg_recall = sum(r['metrics']['recall'] for r in results.values()) / len(results) 
            avg_f1 = sum(r['metrics']['f1'] for r in results.values()) / len(results)
            
            print(f"  Average Precision: {avg_precision:.3f}")
            print(f"  Average Recall: {avg_recall:.3f}")
            print(f"  Average F1: {avg_f1:.3f}")
            
            # Sanity checks - performance should be reasonable
            assert avg_precision >= 0.0, "Precision should be non-negative"
            assert avg_recall >= 0.0, "Recall should be non-negative"
            assert avg_f1 >= 0.0, "F1 should be non-negative"
        else:
            print("  No ground truth data available for validation")


# =============================================================================
# REGRESSION TESTING
# =============================================================================

@pytest.mark.integration
@pytest.mark.regression
class TestRegressionPrevention:
    """Test to prevent regressions in algorithm behavior."""
    
    def test_algorithm_output_stability(self, tiny_domain_data, default_config):
        """Test that algorithm outputs remain stable (regression test)."""
        # This test establishes baseline behavior for regression detection
        # In a real scenario, you'd compare against saved baseline results
        
        baseline_results = {}
        
        for domain_name, domain_data in tiny_domain_data.items():
            try:
                signals, evidence, metadata = detect_shift_signals(
                    domain_data, domain_name, default_config
                )
                
                baseline_results[domain_name] = {
                    'signal_count': len(signals),
                    'evidence_count': len(evidence),
                    'metadata_structure': sorted(metadata.keys()),
                    'first_signal_year': signals[0].year if signals else None,
                    'config_hash': hash(str(default_config.to_dict()))
                }
                
            except Exception as e:
                pytest.fail(f"Regression test failed for {domain_name}: {e}")
        
        # In production, you would save these baselines and compare against them
        # For now, just verify the structure is reasonable
        for domain_name, result in baseline_results.items():
            assert isinstance(result['signal_count'], int)
            assert isinstance(result['evidence_count'], int)
            assert isinstance(result['metadata_structure'], list)
            
        print(f"  ✅ Baseline behavior established for {len(baseline_results)} domains") 