"""
Parameter Validation Test Suite

Tests the comprehensive algorithm configuration system implemented in IMPROVEMENT-001.
Following Phase 13 principle: Fail-fast testing with no error masking.
"""

import pytest
import warnings
from typing import Dict, Any

from core.algorithm_config import (
    ComprehensiveAlgorithmConfig,
    SensitivityConfig,
    create_default_config,
    create_sensitivity_config,
    validate_parameter_combination,
    get_recommended_config_for_domain
)


# =============================================================================
# PARAMETER BOUNDS TESTING
# =============================================================================

@pytest.mark.parameter
class TestParameterBounds:
    """Test parameter boundary validation."""
    
    def test_direction_threshold_bounds(self):
        """Test direction_threshold parameter bounds."""
        # Valid values should work
        config = ComprehensiveAlgorithmConfig.create_custom(
            granularity=3,
            overrides={'direction_threshold': 0.1}
        )
        assert config.direction_threshold == 0.1
        
        config = ComprehensiveAlgorithmConfig.create_custom(
            granularity=3,
            overrides={'direction_threshold': 0.8}
        )
        assert config.direction_threshold == 0.8
        
        # Invalid values should fail fast
        with pytest.raises(ValueError, match="direction_threshold must be 0.1-0.8"):
            ComprehensiveAlgorithmConfig.create_custom(
                granularity=3,
                overrides={'direction_threshold': 0.05}
            )
        
        with pytest.raises(ValueError, match="direction_threshold must be 0.1-0.8"):
            ComprehensiveAlgorithmConfig.create_custom(
                granularity=3,
                overrides={'direction_threshold': 0.9}
            )
    
    def test_clustering_window_bounds(self):
        """Test clustering_window parameter bounds."""
        # Valid boundaries
        config = ComprehensiveAlgorithmConfig.create_custom(
            granularity=3,
            overrides={'clustering_window': 1}
        )
        assert config.clustering_window == 1
        
        config = ComprehensiveAlgorithmConfig.create_custom(
            granularity=3,
            overrides={'clustering_window': 10}
        )
        assert config.clustering_window == 10
        
        # Invalid boundaries should fail fast
        with pytest.raises(ValueError, match="clustering_window must be 1-10 years"):
            ComprehensiveAlgorithmConfig.create_custom(
                granularity=3,
                overrides={'clustering_window': 0}
            )
        
        with pytest.raises(ValueError, match="clustering_window must be 1-10 years"):
            ComprehensiveAlgorithmConfig.create_custom(
                granularity=3,
                overrides={'clustering_window': 15}
            )
    
    def test_validation_threshold_bounds(self):
        """Test validation_threshold parameter bounds."""
        # Valid boundaries
        config = ComprehensiveAlgorithmConfig.create_custom(
            granularity=3,
            overrides={'validation_threshold': 0.5}
        )
        assert config.validation_threshold == 0.5
        
        config = ComprehensiveAlgorithmConfig.create_custom(
            granularity=3,
            overrides={'validation_threshold': 0.95}
        )
        assert config.validation_threshold == 0.95
        
        # Invalid boundaries should fail fast
        with pytest.raises(ValueError, match="validation_threshold must be 0.5-0.95"):
            ComprehensiveAlgorithmConfig.create_custom(
                granularity=3,
                overrides={'validation_threshold': 0.3}
            )
        
        with pytest.raises(ValueError, match="validation_threshold must be 0.5-0.95"):
            ComprehensiveAlgorithmConfig.create_custom(
                granularity=3,
                overrides={'validation_threshold': 1.1}
            )
    
    def test_citation_boost_bounds(self):
        """Test citation_boost parameter bounds."""
        # Valid boundaries
        config = ComprehensiveAlgorithmConfig.create_custom(
            granularity=3,
            overrides={'citation_boost': 0.0}
        )
        assert config.citation_boost == 0.0
        
        config = ComprehensiveAlgorithmConfig.create_custom(
            granularity=3,
            overrides={'citation_boost': 1.0}
        )
        assert config.citation_boost == 1.0
        
        # Invalid boundaries should fail fast
        with pytest.raises(ValueError, match="citation_boost must be 0.0-1.0"):
            ComprehensiveAlgorithmConfig.create_custom(
                granularity=3,
                overrides={'citation_boost': -0.1}
            )
        
        with pytest.raises(ValueError, match="citation_boost must be 0.0-1.0"):
            ComprehensiveAlgorithmConfig.create_custom(
                granularity=3,
                overrides={'citation_boost': 1.5}
            )
    
    def test_citation_support_window_bounds(self):
        """Test citation_support_window parameter bounds."""
        # Valid boundaries
        config = ComprehensiveAlgorithmConfig.create_custom(
            granularity=3,
            overrides={'citation_support_window': 1}
        )
        assert config.citation_support_window == 1
        
        config = ComprehensiveAlgorithmConfig.create_custom(
            granularity=3,
            overrides={'citation_support_window': 10}
        )
        assert config.citation_support_window == 10
        
        # Invalid boundaries should fail fast
        with pytest.raises(ValueError, match="citation_support_window must be 1-10 years"):
            ComprehensiveAlgorithmConfig.create_custom(
                granularity=3,
                overrides={'citation_support_window': 0}
            )
        
        with pytest.raises(ValueError, match="citation_support_window must be 1-10 years"):
            ComprehensiveAlgorithmConfig.create_custom(
                granularity=3,
                overrides={'citation_support_window': 15}
            )


# =============================================================================
# LOGICAL CONSISTENCY TESTING
# =============================================================================

@pytest.mark.parameter
class TestLogicalConsistency:
    """Test parameter logical consistency validation."""
    
    def test_segment_length_thresholds_ordering(self):
        """Test that segment_length_thresholds must be in ascending order."""
        # Valid ordering should work
        config = ComprehensiveAlgorithmConfig.create_custom(
            granularity=3,
            overrides={'segment_length_thresholds': [3, 5, 7]}
        )
        assert config.segment_length_thresholds == [3, 5, 7]
        
        # Invalid ordering should fail fast
        with pytest.raises(ValueError, match="segment_length_thresholds must be in ascending order"):
            ComprehensiveAlgorithmConfig.create_custom(
                granularity=3,
                overrides={'segment_length_thresholds': [7, 5, 3]}
            )
    
    def test_statistical_significance_breakpoints_ordering(self):
        """Test that statistical_significance_breakpoints must be in ascending order."""
        # Valid ordering should work
        config = ComprehensiveAlgorithmConfig.create_custom(
            granularity=3,
            overrides={'statistical_significance_breakpoints': [0.3, 0.6]}
        )
        assert config.statistical_significance_breakpoints == [0.3, 0.6]
        
        # Invalid ordering should fail fast
        with pytest.raises(ValueError, match="statistical_significance_breakpoints must be in ascending order"):
            ComprehensiveAlgorithmConfig.create_custom(
                granularity=3,
                overrides={'statistical_significance_breakpoints': [0.6, 0.3]}
            )
    
    def test_clustering_window_direction_window_relationship(self):
        """Test warning for clustering window much larger than direction window."""
        # Should generate warning when clustering_window >= 2x direction_window_years
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            config = ComprehensiveAlgorithmConfig.create_custom(
                granularity=3,
                overrides={
                    'direction_window_years': 3,
                    'clustering_window': 7  # >= 2 * 3
                }
            )
            assert len(w) == 1
            assert "clustering_window" in str(w[0].message)
    
    def test_unknown_parameter_rejection(self):
        """Test that unknown parameters are rejected."""
        with pytest.raises(ValueError, match="Unknown parameter"):
            ComprehensiveAlgorithmConfig.create_custom(
                granularity=3,
                overrides={'nonexistent_parameter': 0.5}
            )


# =============================================================================
# GRANULARITY PRESETS TESTING
# =============================================================================

@pytest.mark.parameter
class TestGranularityPresets:
    """Test granularity preset functionality."""
    
    def test_all_granularity_levels(self, all_granularity_configs):
        """Test that all granularity levels (1-5) create valid configurations."""
        for config in all_granularity_configs:
            # Should not raise exceptions
            config._validate_parameters()
            summary = config.get_configuration_summary()
            assert len(summary) > 0
            assert 1 <= config.granularity <= 5
    
    def test_granularity_level_1_ultra_fine(self):
        """Test ultra-fine granularity (level 1) parameters."""
        config = ComprehensiveAlgorithmConfig(granularity=1)
        
        # Should have most sensitive settings
        assert config.direction_threshold == 0.2  # Lowest threshold
        assert config.clustering_window == 2      # Smallest window
        assert config.validation_threshold == 0.7 # Lower validation threshold
    
    def test_granularity_level_3_balanced(self):
        """Test balanced granularity (level 3) parameters."""
        config = ComprehensiveAlgorithmConfig(granularity=3)
        
        # Should have balanced settings
        assert config.direction_threshold == 0.4
        assert config.clustering_window == 3
        assert config.validation_threshold == 0.8
        assert config.citation_boost == 0.3
    
    def test_granularity_level_5_ultra_coarse(self):
        """Test ultra-coarse granularity (level 5) parameters."""
        config = ComprehensiveAlgorithmConfig(granularity=5)
        
        # Should have least sensitive settings
        assert config.direction_threshold == 0.6  # Highest threshold
        assert config.clustering_window == 4      # Larger window
        assert config.validation_threshold == 0.9 # Highest validation threshold
        assert config.citation_boost == 0.2       # Lower citation boost
    
    def test_granularity_ordering_consistency(self):
        """Test that granularity levels maintain expected ordering."""
        configs = [ComprehensiveAlgorithmConfig(granularity=i) for i in range(1, 6)]
        
        # Direction threshold should increase with granularity (less sensitive)
        thresholds = [c.direction_threshold for c in configs]
        assert thresholds == sorted(thresholds), "Direction thresholds should increase with granularity"
        
        # Validation threshold should increase with granularity (more restrictive)
        val_thresholds = [c.validation_threshold for c in configs]
        assert val_thresholds == sorted(val_thresholds), "Validation thresholds should increase with granularity"
    
    def test_custom_overrides_preserve_granularity(self):
        """Test that custom overrides preserve granularity level but override specific parameters."""
        config = ComprehensiveAlgorithmConfig.create_custom(
            granularity=3,
            overrides={'direction_threshold': 0.25}  # Override just this parameter
        )
        
        # Override should be applied
        assert config.direction_threshold == 0.25
        
        # Other level 3 defaults should be preserved
        assert config.clustering_window == 3
        assert config.validation_threshold == 0.8
        assert config.citation_boost == 0.3


# =============================================================================
# DOMAIN-SPECIFIC CONFIGURATION TESTING
# =============================================================================

@pytest.mark.parameter
class TestDomainSpecificConfigurations:
    """Test domain-specific configuration optimization."""
    
    def test_computer_vision_optimization(self):
        """Test Computer Vision domain optimization."""
        config = ComprehensiveAlgorithmConfig.create_domain_specific('computer_vision')
        
        # CV should have high sensitivity (rapid evolution)
        assert config.direction_threshold == 0.35  # More sensitive
        assert config.clustering_window == 2       # Shorter cycles
        assert config.citation_boost == 0.4        # Strong citation patterns
    
    def test_natural_language_processing_optimization(self):
        """Test NLP domain optimization."""
        config = ComprehensiveAlgorithmConfig.create_domain_specific('natural_language_processing')
        
        # NLP should have very high sensitivity (very dynamic)
        assert config.direction_threshold == 0.3   # Very sensitive
        assert config.clustering_window == 2       # Rapid evolution
        assert config.citation_boost == 0.35
    
    def test_applied_mathematics_optimization(self):
        """Test Applied Mathematics domain optimization."""
        config = ComprehensiveAlgorithmConfig.create_domain_specific('applied_mathematics')
        
        # Applied Math should be more conservative (stable field)
        assert config.direction_threshold == 0.5   # Less sensitive
        assert config.clustering_window == 5       # Longer cycles
        assert config.citation_boost == 0.25       # Weaker citation patterns
    
    def test_art_optimization(self):
        """Test Art domain optimization."""
        config = ComprehensiveAlgorithmConfig.create_domain_specific('art')
        
        # Art should be most conservative (very stable)
        assert config.direction_threshold == 0.6   # Least sensitive
        assert config.clustering_window == 8       # Very long cycles
        assert config.citation_boost == 0.2        # Weakest citation patterns
    
    def test_unknown_domain_fallback(self):
        """Test that unknown domains fall back to base configuration."""
        config = ComprehensiveAlgorithmConfig.create_domain_specific('unknown_domain')
        base_config = ComprehensiveAlgorithmConfig(granularity=3)
        
        # Should match base configuration
        assert config.direction_threshold == base_config.direction_threshold
        assert config.clustering_window == base_config.clustering_window
        assert config.validation_threshold == base_config.validation_threshold


# =============================================================================
# BACKWARD COMPATIBILITY TESTING
# =============================================================================

@pytest.mark.parameter
class TestBackwardCompatibility:
    """Test backward compatibility with legacy SensitivityConfig interface."""
    
    def test_sensitivity_config_creation(self):
        """Test that SensitivityConfig can be created and works."""
        config = SensitivityConfig(granularity=3)
        
        # Should expose legacy properties
        assert hasattr(config, 'detection_threshold')
        assert hasattr(config, 'clustering_window')
        assert hasattr(config, 'validation_threshold')
        assert hasattr(config, 'citation_boost')
        assert hasattr(config, 'granularity')
    
    def test_legacy_parameter_mapping(self):
        """Test that legacy parameter names are mapped correctly."""
        config = SensitivityConfig(
            granularity=3,
            detection_threshold=0.35,  # Legacy name
            clustering_window=4,
            validation_threshold=0.85,
            citation_boost=0.4
        )
        
        # Legacy properties should work
        assert config.detection_threshold == 0.35
        assert config.clustering_window == 4
        assert config.validation_threshold == 0.85
        assert config.citation_boost == 0.4
        
        # Should map to new comprehensive config
        comp_config = config.comprehensive_config
        assert comp_config.direction_threshold == 0.35  # Mapped from detection_threshold
        assert comp_config.clustering_window == 4
        assert comp_config.validation_threshold == 0.85
        assert comp_config.citation_boost == 0.4
    
    def test_legacy_config_summary(self):
        """Test that legacy config can generate summaries."""
        config = SensitivityConfig(granularity=2)
        summary = config.get_configuration_summary()
        
        assert isinstance(summary, str)
        assert len(summary) > 0
        assert 'Granularity 2' in summary


# =============================================================================
# CONFIGURATION SERIALIZATION TESTING
# =============================================================================

@pytest.mark.parameter
class TestConfigurationSerialization:
    """Test configuration export/import functionality."""
    
    def test_config_to_dict(self):
        """Test configuration dictionary export."""
        config = ComprehensiveAlgorithmConfig.create_custom(
            granularity=3,
            overrides={
                'direction_threshold': 0.35,
                'clustering_window': 4,
                'citation_boost': 0.4
            }
        )
        
        config_dict = config.to_dict()
        
        # Should contain all parameters
        assert 'granularity' in config_dict
        assert 'direction_threshold' in config_dict
        assert 'clustering_window' in config_dict
        assert 'citation_boost' in config_dict
        
        # Values should match
        assert config_dict['granularity'] == 3
        assert config_dict['direction_threshold'] == 0.35
        assert config_dict['clustering_window'] == 4
        assert config_dict['citation_boost'] == 0.4
    
    def test_config_summary_generation(self):
        """Test configuration summary generation."""
        config = ComprehensiveAlgorithmConfig(granularity=3)
        summary = config.get_configuration_summary()
        
        assert isinstance(summary, str)
        assert 'Granularity 3' in summary
        assert 'Detection:' in summary
        assert 'Clustering:' in summary
        assert 'Validation:' in summary
        assert 'Segmentation:' in summary
    
    def test_parameter_explanations(self):
        """Test parameter explanation generation."""
        config = ComprehensiveAlgorithmConfig(granularity=3)
        explanations = config.get_parameter_explanations()
        
        assert isinstance(explanations, dict)
        assert 'direction_threshold' in explanations
        assert 'clustering_window' in explanations
        assert 'validation_threshold' in explanations
        assert 'citation_boost' in explanations
        
        # Explanations should be helpful strings
        for param, explanation in explanations.items():
            assert isinstance(explanation, str)
            assert len(explanation) > 20  # Should be descriptive


# =============================================================================
# UTILITY FUNCTION TESTING
# =============================================================================

@pytest.mark.parameter
class TestUtilityFunctions:
    """Test configuration utility functions."""
    
    def test_create_default_config(self):
        """Test default configuration creation."""
        config = create_default_config()
        
        assert isinstance(config, ComprehensiveAlgorithmConfig)
        assert config.granularity == 3
        # Should not raise exceptions
        config._validate_parameters()
    
    def test_create_sensitivity_config(self):
        """Test legacy sensitivity config creation."""
        config = create_sensitivity_config(granularity=2)
        
        assert isinstance(config, SensitivityConfig)
        assert config.granularity == 2
    
    def test_validate_parameter_combination(self):
        """Test parameter combination validation."""
        # Good configuration should pass
        good_config = ComprehensiveAlgorithmConfig(granularity=3)
        messages = validate_parameter_combination(good_config)
        assert isinstance(messages, list)
        
        # Problematic configuration should generate warnings
        problematic_config = ComprehensiveAlgorithmConfig.create_custom(
            granularity=3,
            overrides={
                'direction_threshold': 0.7,    # High threshold
                'clustering_window': 2,        # Small window
                'validation_threshold': 0.95,  # Very high validation
                'citation_boost': 0.1          # Low citation boost
            }
        )
        messages = validate_parameter_combination(problematic_config)
        assert len(messages) > 0  # Should generate warnings
    
    def test_get_recommended_config_for_domain(self):
        """Test domain recommendation function."""
        config = get_recommended_config_for_domain('computer_vision')
        
        assert isinstance(config, ComprehensiveAlgorithmConfig)
        # Should have CV-specific optimizations
        assert config.direction_threshold == 0.35
        assert config.clustering_window == 2


# =============================================================================
# EDGE CASE AND STRESS TESTING
# =============================================================================

@pytest.mark.parameter
class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_extreme_boundary_values(self, edge_case_configs):
        """Test configurations at extreme boundaries."""
        for config in edge_case_configs:
            # Should not raise exceptions even at boundaries
            config._validate_parameters()
            summary = config.get_configuration_summary()
            assert len(summary) > 0
    
    def test_invalid_granularity_levels(self):
        """Test invalid granularity levels."""
        # Granularity outside 1-5 should still work but not apply presets
        config = ComprehensiveAlgorithmConfig(granularity=0)
        # Should use default values when granularity is out of range
        assert hasattr(config, 'direction_threshold')
        
        config = ComprehensiveAlgorithmConfig(granularity=10)
        assert hasattr(config, 'direction_threshold')
    
    def test_parameter_type_validation(self):
        """Test that parameters reject wrong types (fail fast)."""
        # Should fail fast on wrong types, not convert silently
        with pytest.raises((ValueError, TypeError)):
            ComprehensiveAlgorithmConfig.create_custom(
                granularity=3,
                overrides={'direction_threshold': 'invalid_string'}
            )
    
    def test_empty_list_parameters(self):
        """Test behavior with empty list parameters."""
        # Empty lists should be handled gracefully or rejected
        with pytest.raises(ValueError):
            ComprehensiveAlgorithmConfig.create_custom(
                granularity=3,
                overrides={'multi_scale_windows': []}
            )
        
        with pytest.raises(ValueError):
            ComprehensiveAlgorithmConfig.create_custom(
                granularity=3,
                overrides={'segment_length_thresholds': []}
            )


# =============================================================================
# INTEGRATION WITH VALIDATION HELPERS
# =============================================================================

@pytest.mark.parameter 
def test_config_validation_with_helpers(validation_helpers, default_config):
    """Test configuration validation using validation helpers."""
    # Should not raise exceptions
    validation_helpers.assert_config_valid(default_config)
    
    # Test with custom config
    custom_config = ComprehensiveAlgorithmConfig.create_custom(
        granularity=2,
        overrides={'direction_threshold': 0.25}
    )
    validation_helpers.assert_config_valid(custom_config) 