"""
Comprehensive Algorithm Configuration for Timeline Segmentation

This module provides centralized configuration for all timeline segmentation algorithm parameters,
offering both convenient presets via granularity levels and granular control for developers
who need specific parameter adjustment.

Provides maximum developer flexibility with transparent parameter control.
"""

from dataclasses import dataclass, field, fields
from typing import Dict, List, Any, Optional
import warnings
import os
import json


def load_optimized_parameters() -> Dict[str, Dict[str, float]]:
    """Load optimized parameters from JSON file."""
    optimized_params_file = "results/optimized_parameters_bayesian.json"
    
    if not os.path.exists(optimized_params_file):
        return {}
    
    try:
        with open(optimized_params_file, 'r') as f:
            data = json.load(f)
        return data.get('consensus_difference_optimized_parameters', {})
    except Exception:
        return {}


@dataclass
class AlgorithmConfig:
    """
    Comprehensive configuration for timeline segmentation algorithm.
    
    This class centralizes all algorithm parameters with sensible defaults and granularity-based presets.
    Supports both convenience (granularity levels) and precision (individual parameter control).
    
    Key Features:
    - 14 configurable parameters covering all algorithm aspects
    - 5 granularity levels (1=ultra-coarse to 5=ultra-fine) 
    - Domain-specific optimization integration
    - Comprehensive parameter validation
    - Transparent decision rationale
    """
    
    # Core Detection Parameters
    granularity: int = 3
    direction_threshold: float = 0.4
    validation_threshold: float = 0.7
    
    # Keyword Analysis Parameters
    keyword_min_frequency: int = 2
    min_significant_keywords: int = 2
    keyword_filtering_enabled: bool = True
    # Optimized keyword filtering ratio for better segmentation performance
    keyword_min_papers_ratio: float = 0.05
    
    # Citation Analysis Parameters
    citation_boost: float = 0.8
    citation_support_window: int = 2
    
    # Temporal Window Parameters
    direction_window_size: int = 3
    citation_analysis_scales: List[int] = field(default_factory=lambda: [1, 3, 5])
    
    # Similarity Segmentation Parameters
    similarity_min_segment_length: int = 3
    similarity_max_segment_length: int = 50
    
    # Text Vectorisation Parameters
    tfidf_max_features: int = 10000  # Optimal TF-IDF capacity for performance
    clean_text_enabled: bool = True  # HTML/stop-word cleaning (infrastructure ready, disabled by default)
    
    # Segment Count Penalty Parameters
    segment_count_penalty_enabled: bool = True   # Default ON after successful validation
    segment_count_penalty_sigma: float = 5.0     # Smoothness parameter for exp penalty curve
    
    # System Parameters
    domain_name: Optional[str] = None
    
    def __post_init__(self):
        """Initialize configuration with domain-specific optimization and granularity scaling."""
        # Load optimized parameters if available
        optimized_params = load_optimized_parameters()
        
        # Apply domain-specific optimization if available
        if self.domain_name and self.domain_name in optimized_params:
            base_params = optimized_params[self.domain_name]
            
            # Scale detection parameters based on granularity
            scaled_params = self._scale_parameters_from_baseline(
                base_params['direction_threshold'],
                base_params['validation_threshold'],
                self.granularity
            )
            
            self.direction_threshold = scaled_params['direction_threshold']
            self.validation_threshold = scaled_params['validation_threshold']
            
            # Apply optimized similarity parameters directly (no granularity scaling)
            if 'similarity_min_segment_length' in base_params:
                self.similarity_min_segment_length = base_params['similarity_min_segment_length']
            if 'similarity_max_segment_length' in base_params:
                self.similarity_max_segment_length = base_params['similarity_max_segment_length']
        else:
            # Use granularity-based scaling from defaults
            if self.granularity != 3:  # Only scale if not baseline granularity
                scaled_params = self._scale_parameters_from_baseline(
                    self.direction_threshold,
                    self.validation_threshold,
                    self.granularity
                )
                self.direction_threshold = scaled_params['direction_threshold']
                self.validation_threshold = scaled_params['validation_threshold']
        
        # Load segment count penalty parameters from optimization config with environment overrides
        try:
            with open("optimization_config.json", 'r') as f:
                config = json.load(f)
            
            penalty_config = config.get("segment_count_penalty", {})
            
            # Priority order: environment variables > optimization_config.json > dataclass defaults
            penalty_enabled_env = os.getenv("SEGMENT_PENALTY")
            if penalty_enabled_env is not None:
                self.segment_count_penalty_enabled = penalty_enabled_env.lower() == "true"
            else:
                self.segment_count_penalty_enabled = penalty_config.get("enabled", self.segment_count_penalty_enabled)
            
            penalty_sigma_env = os.getenv("SEGMENT_PENALTY_SIGMA")
            if penalty_sigma_env is not None:
                self.segment_count_penalty_sigma = float(penalty_sigma_env)
            else:
                self.segment_count_penalty_sigma = penalty_config.get("sigma", self.segment_count_penalty_sigma)
                
        except (FileNotFoundError, json.JSONDecodeError, KeyError, ValueError):
            # Fail-fast: If config loading fails, keep dataclass defaults but proceed
            # This allows the algorithm to work even without optimization_config.json
            pass
        
        # Validate all parameters
        self._validate_parameters()
    
    def _scale_parameters_from_baseline(self, base_direction: float, base_validation: float, granularity: int) -> Dict[str, float]:
        """Scale parameters from optimized baseline based on granularity level."""
        # Granularity scaling factors
        # granularity 1 = ultra-coarse (higher thresholds, fewer signals)
        # granularity 3 = balanced (baseline)
        # granularity 5 = ultra-fine (lower thresholds, more signals)
        
        direction_multipliers = {1: 1.5, 2: 1.25, 3: 1.0, 4: 0.75, 5: 0.6}
        validation_multipliers = {1: 1.1, 2: 1.05, 3: 1.0, 4: 0.95, 5: 0.8}
        
        direction_mult = direction_multipliers.get(granularity, 1.0)
        validation_mult = validation_multipliers.get(granularity, 1.0)
        
        scaled_direction = min(0.9, max(0.1, base_direction * direction_mult))
        scaled_validation = min(0.95, max(0.3, base_validation * validation_mult))
        
        return {
            'direction_threshold': scaled_direction,
            'validation_threshold': scaled_validation
        }
    
    def _validate_parameters(self):
        """Validate all parameter values and relationships."""
        # Granularity validation
        if not 1 <= self.granularity <= 5:
            raise ValueError(f"granularity must be 1-5, got {self.granularity}")
        
        # Threshold validations
        if not 0.1 <= self.direction_threshold <= 0.9:
            raise ValueError(f"direction_threshold must be 0.1-0.9, got {self.direction_threshold}")
        
        if not 0.3 <= self.validation_threshold <= 0.95:
            raise ValueError(f"validation_threshold must be 0.3-0.95, got {self.validation_threshold}")
        
        # Keyword validations
        if not 1 <= self.keyword_min_frequency <= 10:
            raise ValueError(f"keyword_min_frequency must be 1-10, got {self.keyword_min_frequency}")
        
        if not 1 <= self.min_significant_keywords <= 20:
            raise ValueError(f"min_significant_keywords must be 1-20, got {self.min_significant_keywords}")
        
        if not 0.01 <= self.keyword_min_papers_ratio <= 0.5:
            raise ValueError(f"keyword_min_papers_ratio must be 0.01-0.5, got {self.keyword_min_papers_ratio}")
        
        # Citation validations
        if not 0.0 <= self.citation_boost <= 1.0:
            raise ValueError(f"citation_boost must be 0.0-1.0, got {self.citation_boost}")
        
        if not 1 <= self.citation_support_window <= 10:
            raise ValueError(f"citation_support_window must be 1-10, got {self.citation_support_window}")
        
        # Temporal Window validations
        if not 1 <= self.direction_window_size <= 10:
            raise ValueError(f"direction_window_size must be 1-10, got {self.direction_window_size}")
        
        if not self.citation_analysis_scales or not all(1 <= scale <= 10 for scale in self.citation_analysis_scales):
            raise ValueError(f"citation_analysis_scales must be non-empty list with values 1-10, got {self.citation_analysis_scales}")
        
        # Similarity segmentation validations
        if not 1 <= self.similarity_min_segment_length <= 20:
            raise ValueError(f"similarity_min_segment_length must be 1-20, got {self.similarity_min_segment_length}")
        
        if not 10 <= self.similarity_max_segment_length <= 100:
            raise ValueError(f"similarity_max_segment_length must be 10-100, got {self.similarity_max_segment_length}")
        
        if self.similarity_min_segment_length >= self.similarity_max_segment_length:
            raise ValueError("similarity_min_segment_length must be < similarity_max_segment_length")
        
        # Text vectorisation validations
        if not 100 <= self.tfidf_max_features <= 50000:
            raise ValueError(f"tfidf_max_features must be 100-50000, got {self.tfidf_max_features}")
        if not isinstance(self.clean_text_enabled, bool):
            raise ValueError("clean_text_enabled must be a boolean")
        
        # Segment count penalty validations
        if not isinstance(self.segment_count_penalty_enabled, bool):
            raise ValueError("segment_count_penalty_enabled must be a boolean")
        if not 0.5 <= self.segment_count_penalty_sigma <= 50.0:
            raise ValueError(f"segment_count_penalty_sigma must be 0.5-50.0, got {self.segment_count_penalty_sigma}")
    
    @classmethod
    def create_custom(cls, granularity: int = 3, domain_name: Optional[str] = None, 
                     overrides: Optional[Dict[str, Any]] = None) -> 'AlgorithmConfig':
        """
        Create configuration with custom parameter overrides.
        
        Args:
            granularity: Base granularity level (1-5)
            domain_name: Domain name for optimization lookup
            overrides: Dictionary of parameter overrides
            
        Returns:
            Configured AlgorithmConfig instance
        """
        # Start with granularity-based configuration
        config = cls(granularity=granularity, domain_name=domain_name)
        
        # Apply overrides if provided
        if overrides:
            for param_name, value in overrides.items():
                if hasattr(config, param_name):
                    setattr(config, param_name, value)
                else:
                    raise ValueError(f"Unknown parameter: {param_name}")
            
            # Re-validate after overrides
            config._validate_parameters()
        
        return config
    
    def get_rationale(self) -> str:
        """Get human-readable rationale for current configuration."""
        granularity_descriptions = {
            1: "ultra_coarse",
            2: "coarse", 
            3: "balanced",
            4: "fine",
            5: "ultra_fine"
        }
        
        description = granularity_descriptions.get(self.granularity, "custom")
        
        rationale_parts = [
            f"Algorithm Configuration (Granularity {self.granularity} for {self.domain_name or 'unknown'}):",
            f"  Detection: threshold={self.direction_threshold:.3f}",
            f"  Validation: threshold={self.validation_threshold:.3f}, boost={self.citation_boost}",
            f"  Keywords: min_freq={self.keyword_min_frequency}, min_significant={self.min_significant_keywords}",
            f"  Similarity Segmentation: min_length={self.similarity_min_segment_length}y, max_length={self.similarity_max_segment_length}y"
        ]
        
        # Add optimization status
        optimized_params = load_optimized_parameters()
        if self.domain_name and self.domain_name in optimized_params:
            rationale_parts.append(f"Using Bayesian-optimized parameters for {self.domain_name}")
        else:
            rationale_parts.append(f"Scaled from optimized baseline (granularity {self.granularity})")
        
        return "\n".join(rationale_parts)
    
    def __str__(self) -> str:
        """String representation showing key parameters."""
        return (f"AlgorithmConfig(granularity={self.granularity}, "
                f"direction_threshold={self.direction_threshold:.3f}, "
                f"validation_threshold={self.validation_threshold:.3f})")

    def get_configuration_summary(self) -> str:
        """Get a concise summary of the current configuration."""
        return (
            f"granularity={self.granularity}, "
            f"direction_threshold={self.direction_threshold:.3f}, "
            f"validation_threshold={self.validation_threshold:.3f}, "
            f"keyword_min_frequency={self.keyword_min_frequency}, "
            f"min_significant_keywords={self.min_significant_keywords}"
        )


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def create_default_config() -> AlgorithmConfig:
    """Create default algorithm configuration."""
    return AlgorithmConfig()


def validate_parameter_combination(config: AlgorithmConfig) -> List[str]:
    """
    Validate parameter combinations and return list of warnings/suggestions.
    
    Args:
        config: Configuration to validate
        
    Returns:
        List of validation messages
    """
    messages = []
    
    # Check for potentially problematic combinations
    if config.validation_threshold > 0.9 and config.citation_boost < 0.3:
        messages.append("Very high validation threshold with low citation boost may be too restrictive")
    
    return messages


def get_recommended_config_for_domain(domain_name: str) -> AlgorithmConfig:
    """
    Get recommended configuration for specific domain.
    
    Args:
        domain_name: Name of research domain
        
    Returns:
        Recommended configuration
    """
    return AlgorithmConfig(domain_name=domain_name)


def export_config_to_file(config: AlgorithmConfig, filepath: str) -> None:
    """
    Export configuration to JSON file.
    
    Args:
        config: Configuration to export
        filepath: Path to save configuration
    """
    import json
    
    config_dict = {
        'granularity': config.granularity,
        'direction_threshold': config.direction_threshold,
        'validation_threshold': config.validation_threshold,
        'keyword_min_frequency': config.keyword_min_frequency,
        'min_significant_keywords': config.min_significant_keywords,
        'keyword_filtering_enabled': config.keyword_filtering_enabled,
        'keyword_min_papers_ratio': config.keyword_min_papers_ratio,
        'citation_boost': config.citation_boost,
        'citation_support_window': config.citation_support_window,
        'similarity_min_segment_length': config.similarity_min_segment_length,
        'similarity_max_segment_length': config.similarity_max_segment_length,
        'tfidf_max_features': config.tfidf_max_features,
        'clean_text_enabled': config.clean_text_enabled,
        'domain_name': config.domain_name
    }
    
    config_dict['_metadata'] = {
        'config_type': 'AlgorithmConfig',
        'version': '1.0',
        'description': 'Timeline Segmentation Algorithm Configuration'
    }
    
    with open(filepath, 'w') as f:
        json.dump(config_dict, f, indent=2)


def load_config_from_file(filepath: str) -> AlgorithmConfig:
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
    
    return AlgorithmConfig(**config_dict) 