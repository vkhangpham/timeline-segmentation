"""
Comprehensive Algorithm Configuration for Timeline Segmentation

This module provides centralized configuration for all timeline segmentation algorithm parameters,
offering both convenient presets via granularity levels and granular control for developers
who need specific parameter adjustment.

Following Phase 13 principle: Maximum developer flexibility with transparent parameter control.
"""

from dataclasses import dataclass, field, fields
from typing import Dict, List, Any, Optional
import warnings


@dataclass
class ComprehensiveAlgorithmConfig:
    """
    Centralized configuration for all timeline segmentation algorithm parameters.
    
    Provides both convenient presets via granularity levels and granular control
    for developers who need specific parameter adjustment.
    
    Features:
    - All hardcoded parameters centralized
    - Granularity presets (1-5) for convenience  
    - Parameter validation and consistency checking
    - Backward compatibility with existing interfaces
    - Domain-specific adaptation framework
    """
    
    # === GRANULARITY CONTROL ===
    granularity: int = 3  # 1-5 scale for convenient presets
    
    # === DIRECTION DETECTION PARAMETERS ===
    direction_window_years: int = 3
    direction_threshold: float = 0.4
    keyword_min_frequency: int = 2
    min_significant_keywords: int = 3
    novelty_weight: float = 1.0
    overlap_weight: float = 1.0
    
    # === CITATION ANALYSIS PARAMETERS ===
    citation_support_window: int = 2  # ±years
    citation_boost: float = 0.3
    citation_gradient_multiplier: float = 1.5
    citation_acceleration_multiplier: float = 2.0
    multi_scale_windows: List[int] = field(default_factory=lambda: [1, 3, 5])
    
    # === TEMPORAL CLUSTERING PARAMETERS ===
    clustering_window: int = 3
    cluster_method: str = "start_year_comparison"  # vs "end_year_comparison"
    merge_strategy: str = "representative_year"    # vs "weighted_average"
    
    # === VALIDATION PARAMETERS ===
    validation_threshold: float = 0.7
    consistent_threshold_mode: bool = True  # Use same threshold for all signals
    breakthrough_validation: bool = False   # Removed as "too permissive"
    
    # === SEGMENTATION PARAMETERS ===
    segment_length_thresholds: List[int] = field(default_factory=lambda: [4, 6, 8])
    statistical_significance_breakpoints: List[float] = field(default_factory=lambda: [0.4, 0.5])
    max_segment_length_conservative: int = 50
    max_segment_length_standard: int = 100
    merge_preference: str = "backward"  # vs "forward" vs "shortest"
    
    # === PERFORMANCE PARAMETERS ===
    memory_efficient_mode: bool = False
    batch_processing_size: int = 1000
    enable_parallel_processing: bool = False
    
    # === DOMAIN ADAPTATION PARAMETERS ===
    domain_specific_calibration: bool = False
    adaptive_window_sizing: bool = False
    auto_threshold_tuning: bool = False
    
    def __post_init__(self):
        """Apply granularity presets and validate configuration."""
        if 1 <= self.granularity <= 5:
            self._apply_granularity_presets()
        self._validate_parameters()
    
    def _apply_granularity_presets(self):
        """Apply convenient granularity presets while allowing overrides."""
        presets = self._get_granularity_presets()
        
        # Only apply preset if parameter wasn't explicitly overridden
        if hasattr(self, '_user_overrides'):
            for param, value in presets.items():
                if param not in self._user_overrides:
                    setattr(self, param, value)
        else:
            # First initialization - apply all presets
            for param, value in presets.items():
                setattr(self, param, value)
    
    def _get_granularity_presets(self) -> Dict[str, Any]:
        """
        Get parameter presets for granularity level.
        
        Granularity Levels:
        1 (Ultra-fine): Most segments, highest sensitivity
        2 (Fine): High sensitivity  
        3 (Balanced): Default, moderate sensitivity
        4 (Coarse): Lower sensitivity
        5 (Ultra-coarse): Fewest segments, lowest sensitivity
        """
        presets = {
            1: {  # Ultra-fine: Most segments
                'direction_threshold': 0.2,
                'clustering_window': 2,
                'validation_threshold': 0.7,
                'citation_boost': 0.3
            },
            2: {  # Fine
                'direction_threshold': 0.3,
                'clustering_window': 2,
                'validation_threshold': 0.75,
                'citation_boost': 0.3
            },
            3: {  # Balanced (default)
                'direction_threshold': 0.4,
                'clustering_window': 3,
                'validation_threshold': 0.8,
                'citation_boost': 0.3
            },
            4: {  # Coarse
                'direction_threshold': 0.5,
                'clustering_window': 4,
                'validation_threshold': 0.85,
                'citation_boost': 0.3
            },
            5: {  # Ultra-coarse: Fewest segments
                'direction_threshold': 0.6,
                'clustering_window': 4,
                'validation_threshold': 0.9,
                'citation_boost': 0.2
            }
        }
        return presets[self.granularity]
    
    def _validate_parameters(self):
        """Validate parameter values and logical consistency."""
        # Range validations
        if not 0.1 <= self.direction_threshold <= 0.8:
            raise ValueError(f"direction_threshold must be 0.1-0.8, got {self.direction_threshold}")
        
        if not 1 <= self.clustering_window <= 10:
            raise ValueError(f"clustering_window must be 1-10 years, got {self.clustering_window}")
            
        if not 0.5 <= self.validation_threshold <= 0.95:
            raise ValueError(f"validation_threshold must be 0.5-0.95, got {self.validation_threshold}")
            
        if not 1 <= self.citation_support_window <= 10:
            raise ValueError(f"citation_support_window must be 1-10 years, got {self.citation_support_window}")
        
        if not 0.0 <= self.citation_boost <= 1.0:
            raise ValueError(f"citation_boost must be 0.0-1.0, got {self.citation_boost}")
        
        # Logical consistency validations
        if self.clustering_window >= self.direction_window_years * 2:
            warnings.warn(f"clustering_window ({self.clustering_window}) >= 2x direction_window ({self.direction_window_years})")
        
        if self.segment_length_thresholds != sorted(self.segment_length_thresholds):
            raise ValueError("segment_length_thresholds must be in ascending order")
            
        if self.statistical_significance_breakpoints != sorted(self.statistical_significance_breakpoints):
            raise ValueError("statistical_significance_breakpoints must be in ascending order")
    
    @classmethod
    def create_custom(cls, 
                      granularity: int = 3,
                      overrides: Optional[Dict[str, Any]] = None) -> 'ComprehensiveAlgorithmConfig':
        """
        Create configuration with custom parameter overrides.
        
        Args:
            granularity: Base granularity level (1-5)
            overrides: Dictionary of parameter overrides
            
        Returns:
            Custom configuration instance
            
        Example:
            config = ComprehensiveAlgorithmConfig.create_custom(
                granularity=3,
                overrides={
                    'direction_threshold': 0.35,
                    'clustering_window': 2,
                    'citation_boost': 0.4
                }
            )
        """
        config = cls(granularity=granularity)
        
        if overrides:
            config._user_overrides = set(overrides.keys())
            for param, value in overrides.items():
                if hasattr(config, param):
                    setattr(config, param, value)
                else:
                    raise ValueError(f"Unknown parameter: {param}")
            
            # Re-validate after overrides
            config._validate_parameters()
        
        return config
    
    @classmethod 
    def create_domain_specific(cls,
                              domain_name: str,
                              base_granularity: int = 3) -> 'ComprehensiveAlgorithmConfig':
        """
        Create domain-specific configuration with optimized parameters.
        
        Args:
            domain_name: Name of domain for optimization
            base_granularity: Base granularity level
            
        Returns:
            Domain-optimized configuration
        """
        # Domain-specific parameter optimizations
        domain_optimizations = {
            'computer_vision': {
                'direction_threshold': 0.35,  # CV evolves rapidly
                'clustering_window': 2,       # Shorter paradigm cycles
                'citation_boost': 0.4         # Strong citation patterns
            },
            'natural_language_processing': {
                'direction_threshold': 0.3,   # Very dynamic field
                'clustering_window': 2,       # Rapid evolution
                'citation_boost': 0.35
            },
            'applied_mathematics': {
                'direction_threshold': 0.5,   # More stable field
                'clustering_window': 5,       # Longer paradigm cycles
                'citation_boost': 0.25        # Citation patterns less clear
            },
            'art': {
                'direction_threshold': 0.6,   # Very stable
                'clustering_window': 8,       # Long paradigm cycles
                'citation_boost': 0.2         # Weak citation patterns
            }
        }
        
        overrides = domain_optimizations.get(domain_name.lower(), {})
        
        return cls.create_custom(
            granularity=base_granularity,
            overrides=overrides
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Export configuration as dictionary for serialization."""
        return {field.name: getattr(self, field.name) for field in fields(self)}
    
    def get_configuration_summary(self) -> str:
        """Generate human-readable configuration summary."""
        summary = f"Algorithm Configuration (Granularity {self.granularity}):\n"
        summary += f"  Detection: threshold={self.direction_threshold:.2f}, window={self.direction_window_years}y\n"
        summary += f"  Clustering: window={self.clustering_window}y, method={self.cluster_method}\n"
        summary += f"  Validation: threshold={self.validation_threshold:.2f}, boost={self.citation_boost:.2f}\n"
        summary += f"  Segmentation: lengths={self.segment_length_thresholds}, max_length={self.max_segment_length_standard}\n"
        
        if self.domain_specific_calibration:
            summary += f"  Domain Adaptation: ENABLED\n"
        if self.memory_efficient_mode:
            summary += f"  Performance: Memory efficient mode\n"
            
        return summary
    
    def get_parameter_explanations(self) -> Dict[str, str]:
        """Get detailed explanations for all parameters."""
        explanations = {
            # Direction Detection
            'direction_threshold': 'Minimum score needed to detect a direction signal (lower = more sensitive)',
            'direction_window_years': 'Sliding window size for keyword evolution analysis',
            'keyword_min_frequency': 'Minimum frequency for keyword significance',
            'min_significant_keywords': 'Minimum number of significant new keywords for paradigm shift',
            
            # Citation Analysis  
            'citation_support_window': 'Time window (±years) for citation support validation',
            'citation_boost': 'Confidence boost given to signals with citation support',
            'citation_gradient_multiplier': 'Multiplier for gradient threshold in citation analysis',
            'citation_acceleration_multiplier': 'Multiplier for acceleration threshold in citation analysis',
            
            # Temporal Clustering
            'clustering_window': 'Temporal window for clustering direction signals (years)',
            'cluster_method': 'Method for temporal clustering (start_year_comparison vs end_year_comparison)',
            'merge_strategy': 'Strategy for merging clustered signals (representative_year vs weighted_average)',
            
            # Validation
            'validation_threshold': 'Minimum score needed to validate signals (consistent for all signals)',
            'consistent_threshold_mode': 'Whether to use same threshold for all signal types',
            
            # Segmentation
            'segment_length_thresholds': 'Minimum segment lengths based on statistical significance [4,6,8]',
            'statistical_significance_breakpoints': 'Breakpoints for significance-based calibration [0.4,0.5]',
            'max_segment_length_conservative': 'Maximum segment length for low significance domains',
            'max_segment_length_standard': 'Maximum segment length for standard domains',
            'merge_preference': 'Preference for segment merging direction (backward/forward/shortest)'
        }
        
        return explanations


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def create_default_config() -> ComprehensiveAlgorithmConfig:
    """Create default algorithm configuration."""
    return ComprehensiveAlgorithmConfig(granularity=3)


def validate_parameter_combination(config: ComprehensiveAlgorithmConfig) -> List[str]:
    """
    Validate parameter combinations and return list of warnings/suggestions.
    
    Args:
        config: Configuration to validate
        
    Returns:
        List of validation messages
    """
    messages = []
    
    # Check for potentially problematic combinations
    if config.direction_threshold > 0.6 and config.clustering_window < 3:
        messages.append("High direction threshold with small clustering window may miss paradigm shifts")
    
    if config.validation_threshold > 0.9 and config.citation_boost < 0.3:
        messages.append("Very high validation threshold with low citation boost may be too restrictive")
    
    if config.clustering_window > config.direction_window_years * 2:
        messages.append("Clustering window much larger than direction window may cause unexpected behavior")
    
    # Check for efficiency concerns
    if not config.memory_efficient_mode and config.batch_processing_size < 500:
        messages.append("Small batch size without memory efficient mode may impact performance")
    
    return messages


def get_recommended_config_for_domain(domain_name: str) -> ComprehensiveAlgorithmConfig:
    """
    Get recommended configuration for specific domain.
    
    Args:
        domain_name: Name of research domain
        
    Returns:
        Recommended configuration
    """
    return ComprehensiveAlgorithmConfig.create_domain_specific(domain_name)


def export_config_to_file(config: ComprehensiveAlgorithmConfig, filepath: str) -> None:
    """
    Export configuration to JSON file.
    
    Args:
        config: Configuration to export
        filepath: Path to save configuration
    """
    import json
    
    config_dict = config.to_dict()
    config_dict['_metadata'] = {
        'config_type': 'ComprehensiveAlgorithmConfig',
        'version': '1.0',
        'description': 'Timeline Segmentation Algorithm Configuration'
    }
    
    with open(filepath, 'w') as f:
        json.dump(config_dict, f, indent=2)


def load_config_from_file(filepath: str) -> ComprehensiveAlgorithmConfig:
    """
    Load configuration from JSON file.
    
    Args:
        filepath: Path to configuration file
        
    Returns:
        Loaded configuration
    """
    import json
    
    with open(filepath, 'r') as f:
        config_dict = json.load(f)
    
    # Remove metadata if present
    config_dict.pop('_metadata', None)
    
    return ComprehensiveAlgorithmConfig(**config_dict) 