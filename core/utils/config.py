"""
Streamlined Algorithm Configuration for Timeline Segmentation

This module provides centralized configuration for timeline segmentation algorithm parameters,
optimized for the new cohesion-separation objective function approach with anti-gaming safeguards.

Key Features:
- Essential parameters only
- Simple validation
- Objective function integration
- Anti-gaming safeguards
- Backward compatibility
- Optimal defaults from experimental validation
"""

from dataclasses import dataclass
from typing import List, Optional
import os
import json


@dataclass
class AlgorithmConfig:
    """
    Streamlined configuration for timeline segmentation algorithm.
    
    Focuses on essential parameters for the new cohesion-separation objective function
    with integrated anti-gaming safeguards. Default values are optimized based on
    comprehensive multi-domain experimental validation.
    """
    
    # Core Detection Parameters
    direction_threshold: float = 0.4
    validation_threshold: float = 0.7
    
    # Citation Analysis Parameters
    citation_boost_rate: float = 0.8
    citation_support_window: int = 2
    citation_analysis_scales: List[int] = None
    
    # Keyword Analysis Parameters (optimized from experiments)
    keyword_min_frequency: int = 2  # Validated across 4 domains
    min_significant_keywords: int = 2
    keyword_filtering_enabled: bool = True
    keyword_min_papers_ratio: float = 0.01  # 1% minimum for robustness
    
    # Temporal Window Parameters
    direction_window_size: int = 3
    
    # Objective Function Parameters (validated optimal from multi-domain analysis)
    cohesion_weight: float = 0.8  # Cohesion-dominant strategy (expert performance: 37.2th percentile)
    separation_weight: float = 0.2  # Conservative separation (expert performance: 15.2th percentile)
    top_k_keywords: int = 15  # Optimal for Jaccard cohesion across all domains
    
    # Anti-Gaming Parameters (validated effective across all domains)
    anti_gaming_min_segment_size: int = 50  # Excludes unrealistic micro-segments
    anti_gaming_size_weight_power: float = 0.5  # Square root weighting (balanced)
    anti_gaming_segment_count_penalty_sigma: float = 4.0  # Lenient penalty (experiments show count penalty too aggressive)
    anti_gaming_enable_size_weighting: bool = True  # Prevents micro-segment gaming
    anti_gaming_enable_segment_floor: bool = True  # Excludes tiny segments
    anti_gaming_enable_count_penalty: bool = False  # Disabled - experiments show over-penalization
    
    # Legacy Parameters (for backward compatibility)
    granularity: int = 3
    
    # System Parameters
    domain_name: Optional[str] = None
    
    def __post_init__(self):
        """Initialize configuration with validation and config loading."""
        # Set default citation analysis scales if not provided
        if self.citation_analysis_scales is None:
            self.citation_analysis_scales = [1, 3, 5]
        
        # Load all parameters from centralized config file if available
        self._load_config_from_file()
        
        # Validate all parameters
        self._validate_parameters()
    
    def _load_config_from_file(self):
        """Load all configuration parameters from optimization_config.json."""
        config_path = "optimization_config.json"
        
        if not os.path.exists(config_path):
            return  # Keep defaults if config file doesn't exist
        
        # FAIL-FAST: Load configuration or fail immediately with clear error message
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Load objective function parameters
        obj_config = config.get("objective_function", {})
        if obj_config:
            self.cohesion_weight = obj_config.get("cohesion_weight", self.cohesion_weight)
            self.separation_weight = obj_config.get("separation_weight", self.separation_weight)
            self.top_k_keywords = obj_config.get("top_k_keywords", self.top_k_keywords)
        
        # Load anti-gaming parameters
        ag_config = config.get("anti_gaming", {})
        if ag_config:
            self.anti_gaming_min_segment_size = ag_config.get("min_segment_size", self.anti_gaming_min_segment_size)
            self.anti_gaming_size_weight_power = ag_config.get("size_weight_power", self.anti_gaming_size_weight_power)
            self.anti_gaming_segment_count_penalty_sigma = ag_config.get("segment_count_penalty_sigma", self.anti_gaming_segment_count_penalty_sigma)
            self.anti_gaming_enable_size_weighting = ag_config.get("enable_size_weighting", self.anti_gaming_enable_size_weighting)
            self.anti_gaming_enable_segment_floor = ag_config.get("enable_segment_floor", self.anti_gaming_enable_segment_floor)
            self.anti_gaming_enable_count_penalty = ag_config.get("enable_count_penalty", self.anti_gaming_enable_count_penalty)
        
        # Load detection parameters if present
        detection_config = config.get("detection_parameters", {})
        if detection_config:
            self.direction_threshold = detection_config.get("direction_threshold", self.direction_threshold)
            self.validation_threshold = detection_config.get("validation_threshold", self.validation_threshold)
            self.citation_boost_rate = detection_config.get("citation_boost_rate", self.citation_boost_rate)
            self.citation_support_window = detection_config.get("citation_support_window", self.citation_support_window)
        
        # Load keyword parameters if present
        keyword_config = config.get("keyword_parameters", {})
        if keyword_config:
            self.keyword_min_frequency = keyword_config.get("min_frequency", self.keyword_min_frequency)
            self.min_significant_keywords = keyword_config.get("min_significant", self.min_significant_keywords)
            self.keyword_filtering_enabled = keyword_config.get("filtering_enabled", self.keyword_filtering_enabled)
            self.keyword_min_papers_ratio = keyword_config.get("min_papers_ratio", self.keyword_min_papers_ratio)
    

    
    def _validate_parameters(self):
        """Validate all parameter values."""
        # Detection thresholds
        if not 0.1 <= self.direction_threshold <= 0.9:
            raise ValueError(f"direction_threshold must be 0.1-0.9, got {self.direction_threshold}")
        
        if not 0.3 <= self.validation_threshold <= 0.95:
            raise ValueError(f"validation_threshold must be 0.3-0.95, got {self.validation_threshold}")
        
        # Citation parameters
        if not 0.0 <= self.citation_boost_rate <= 1.0:
            raise ValueError(f"citation_boost_rate must be 0.0-1.0, got {self.citation_boost_rate}")
        
        if not 1 <= self.citation_support_window <= 10:
            raise ValueError(f"citation_support_window must be 1-10, got {self.citation_support_window}")
        
        # Keyword parameters
        if not 1 <= self.keyword_min_frequency <= 10:
            raise ValueError(f"keyword_min_frequency must be 1-10, got {self.keyword_min_frequency}")
        
        if not 1 <= self.min_significant_keywords <= 20:
            raise ValueError(f"min_significant_keywords must be 1-20, got {self.min_significant_keywords}")
        
        if not 0.001 <= self.keyword_min_papers_ratio <= 0.5:
            raise ValueError(f"keyword_min_papers_ratio must be 0.001-0.5, got {self.keyword_min_papers_ratio}")
        
        # Objective function parameters
        if not 0.0 <= self.cohesion_weight <= 1.0:
            raise ValueError(f"cohesion_weight must be 0.0-1.0, got {self.cohesion_weight}")
        
        if not 0.0 <= self.separation_weight <= 1.0:
            raise ValueError(f"separation_weight must be 0.0-1.0, got {self.separation_weight}")
        
        if abs(self.cohesion_weight + self.separation_weight - 1.0) > 1e-6:
            raise ValueError(f"cohesion_weight + separation_weight must equal 1.0, got {self.cohesion_weight + self.separation_weight:.6f}")
        
        if not 1 <= self.top_k_keywords <= 50:
            raise ValueError(f"top_k_keywords must be 1-50, got {self.top_k_keywords}")
        
        # Anti-gaming parameters
        if not 1 <= self.anti_gaming_min_segment_size <= 200:
            raise ValueError(f"anti_gaming_min_segment_size must be 1-200, got {self.anti_gaming_min_segment_size}")
        
        if not 0.0 <= self.anti_gaming_size_weight_power <= 2.0:
            raise ValueError(f"anti_gaming_size_weight_power must be 0.0-2.0, got {self.anti_gaming_size_weight_power}")
        
        if not 0.5 <= self.anti_gaming_segment_count_penalty_sigma <= 20.0:
            raise ValueError(f"anti_gaming_segment_count_penalty_sigma must be 0.5-20.0, got {self.anti_gaming_segment_count_penalty_sigma}")
        
        # Temporal window parameters
        if not 1 <= self.direction_window_size <= 10:
            raise ValueError(f"direction_window_size must be 1-10, got {self.direction_window_size}")
        
        if not self.citation_analysis_scales or not all(1 <= scale <= 10 for scale in self.citation_analysis_scales):
            raise ValueError(f"citation_analysis_scales must be non-empty list with values 1-10, got {self.citation_analysis_scales}")
    
    def get_anti_gaming_config(self):
        """Get anti-gaming configuration for the objective function."""
        from ..analysis.objective_function import AntiGamingConfig
        
        return AntiGamingConfig(
            min_segment_size=self.anti_gaming_min_segment_size,
            size_weight_power=self.anti_gaming_size_weight_power,
            segment_count_penalty_sigma=self.anti_gaming_segment_count_penalty_sigma,
            enable_size_weighting=self.anti_gaming_enable_size_weighting,
            enable_segment_floor=self.anti_gaming_enable_segment_floor,
            enable_count_penalty=self.anti_gaming_enable_count_penalty
        )
    
    @classmethod
    def create_custom(cls, granularity: int = 3, domain_name: Optional[str] = None, 
                     overrides: dict = None) -> 'AlgorithmConfig':
        """
        Create configuration with custom parameter overrides.
        
        Args:
            granularity: Legacy parameter for backward compatibility
            domain_name: Domain name for context
            overrides: Dictionary of parameter overrides
            
        Returns:
            Configured AlgorithmConfig instance
        """
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
        rationale_parts = [
            f"Algorithm Configuration for {self.domain_name or 'unknown'} (Experimentally Optimized):",
            f"  Detection: direction_threshold={self.direction_threshold:.3f}, validation_threshold={self.validation_threshold:.3f}",
            f"  Citation: boost_rate={self.citation_boost_rate}, support_window={self.citation_support_window}",
            f"  Keywords: min_freq={self.keyword_min_frequency}, min_significant={self.min_significant_keywords}, ratio={self.keyword_min_papers_ratio}",
            f"  Segmentation: boundary_based",
            f"  Objective: cohesion_weight={self.cohesion_weight} (expert 37.2th %ile), separation_weight={self.separation_weight} (expert 15.2th %ile), top_k={self.top_k_keywords}",
            f"  Anti-Gaming: min_size={self.anti_gaming_min_segment_size}, size_weighting={self.anti_gaming_enable_size_weighting} (power={self.anti_gaming_size_weight_power}), count_penalty={self.anti_gaming_enable_count_penalty}"
        ]
        
        return "\n".join(rationale_parts)
    
    def get_configuration_summary(self) -> str:
        """Get a concise summary of the current configuration."""
        return (
            f"direction_threshold={self.direction_threshold:.3f}, "
            f"validation_threshold={self.validation_threshold:.3f}, "
            f"cohesion_weight={self.cohesion_weight} (optimal), "
            f"separation_weight={self.separation_weight} (optimal), "
            f"top_k_keywords={self.top_k_keywords} (validated), "
            f"anti_gaming_enabled={self.anti_gaming_enable_size_weighting}"
        )
    
    def get_experimental_basis(self) -> str:
        """Get summary of experimental validation basis for current configuration."""
        return (
            "Configuration optimized from multi-domain experimental validation:\n"
            f"• Objective weights (0.8, 0.2): Cohesion-dominant strategy from 4-domain analysis\n"
            f"• Top-K keywords (15): Optimal for Jaccard cohesion across NLP, CV, Math, Art\n"
            f"• Anti-gaming parameters: Validated effective across all test domains\n"
            f"• Keyword filtering (≥2 years, ≥1% papers): Ensures robustness\n"
            f"• Segment size floor (50 papers): Prevents micro-segment gaming\n"
            f"• Size weighting (power=0.5): Balanced prevention of small-segment bias\n"
            f"• Count penalty disabled: Experiments show over-penalization"
        )
    
    def __str__(self) -> str:
        """String representation showing key parameters."""
        return (f"AlgorithmConfig(direction_threshold={self.direction_threshold:.3f}, "
                f"validation_threshold={self.validation_threshold:.3f}, "
                f"objective_weights=({self.cohesion_weight}, {self.separation_weight}) [optimal], "
                f"anti_gaming={self.anti_gaming_enable_size_weighting})")


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def create_default_config() -> AlgorithmConfig:
    """Create default algorithm configuration with experimentally optimized parameters."""
    return AlgorithmConfig()


def get_recommended_config_for_domain(domain_name: str) -> AlgorithmConfig:
    """
    Get recommended configuration for specific domain.
    
    Args:
        domain_name: Name of research domain
        
    Returns:
        Recommended configuration with domain-specific optimizations
    """
    # Base configuration with optimal defaults
    config = AlgorithmConfig(domain_name=domain_name)
    
    # Domain-specific adjustments based on experimental results
    if domain_name.lower() in ['natural_language_processing', 'nlp']:
        # NLP showed good separation performance (71st percentile)
        config.separation_weight = 0.25  # Slightly higher separation weight
        config.cohesion_weight = 0.75
    elif domain_name.lower() in ['computer_vision', 'cv']:
        # CV showed excellent TopK separation (77.5th percentile)
        config.top_k_keywords = 20  # More keywords for diverse visual concepts
    elif domain_name.lower() in ['applied_mathematics', 'math']:
        # Math showed exceptional cohesion correlation (r=-0.997)
        config.cohesion_weight = 0.85  # Higher cohesion weight for focused domains
        config.separation_weight = 0.15
    elif domain_name.lower() in ['art']:
        # Art showed stable terminology, lower separation JS
        config.anti_gaming_min_segment_size = 40  # Smaller segments for art history
        config.separation_weight = 0.15  # Lower separation weight
        config.cohesion_weight = 0.85
    
    return config 