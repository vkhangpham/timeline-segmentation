"""Algorithm configuration for timeline segmentation.

Configuration is loaded from config.json file with comprehensive validation.
"""

from dataclasses import dataclass
from typing import List, Optional, NamedTuple
import os
import json


class AntiGamingConfig(NamedTuple):
    """Configuration for anti-gaming safeguards."""

    min_segment_size: int = 50
    size_weight_power: float = 0.5
    segment_count_penalty_sigma: float = 4.0
    enable_size_weighting: bool = True
    enable_segment_floor: bool = True
    enable_count_penalty: bool = True


@dataclass
class AlgorithmConfig:
    """Algorithm configuration loaded from config.json file."""

    # Direction Change Detection Parameters
    direction_change_threshold: float
    direction_threshold_strategy: (
        str  # "fixed" | "global_p90" | "global_p95" | "global_p99"
    )
    direction_scoring_method: str  # "weighted_jaccard" | "jensen_shannon"
    min_baseline_period_years: int
    score_distribution_window_years: int

    # Citation Analysis Parameters
    citation_confidence_boost: float
    citation_support_window_years: int

    # Diagnostic Parameters
    diagnostic_top_keywords_limit: int

    # Data Filtering Parameters
    min_papers_per_year: int

    # Objective Function Parameters
    cohesion_weight: float
    separation_weight: float
    top_k_keywords: int
    min_keyword_frequency_ratio: float

    # Anti-Gaming Parameters
    anti_gaming_min_segment_size: int
    anti_gaming_size_weight_power: float
    anti_gaming_segment_count_penalty_sigma: float
    anti_gaming_enable_size_weighting: bool
    anti_gaming_enable_segment_floor: bool
    anti_gaming_enable_count_penalty: bool

    # Boundary Optimization Parameters
    boundary_optimization_enabled: bool
    boundary_search_window: int
    boundary_max_exhaustive_signals: int
    boundary_max_subset_enumeration: int
    boundary_beam_width: int

    # System Parameters
    domain_name: Optional[str] = None

    # Diagnostic Parameters
    save_direction_diagnostics: bool = False

    @classmethod
    def from_config_file(
        cls, config_path: str = "config.json", domain_name: Optional[str] = None
    ) -> "AlgorithmConfig":
        """Load configuration from JSON file.

        Args:
            config_path: Path to configuration file
            domain_name: Optional domain name for context

        Returns:
            AlgorithmConfig instance

        Raises:
            FileNotFoundError: If config file doesn't exist
            ValueError: If config file is invalid or missing required parameters
        """
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        try:
            with open(config_path, "r") as f:
                config = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in configuration file {config_path}: {e}")

        try:
            detection_params = config["detection_parameters"]
            objective_params = config["objective_function"]
            anti_gaming_params = config["anti_gaming"]
            boundary_params = config["boundary_optimization"]
            diagnostic_params = config.get("diagnostics", {})

            return cls(
                # Direction Change Detection parameters
                direction_change_threshold=detection_params[
                    "direction_change_threshold"
                ],
                direction_threshold_strategy=detection_params.get(
                    "direction_threshold_strategy", "fixed"
                ),
                direction_scoring_method=detection_params.get(
                    "direction_scoring_method", "weighted_jaccard"
                ),
                min_baseline_period_years=detection_params.get(
                    "min_baseline_period_years", 3
                ),
                score_distribution_window_years=detection_params.get(
                    "score_distribution_window_years", 3
                ),
                # Validation parameters
                # Citation Analysis parameters
                citation_confidence_boost=detection_params["citation_confidence_boost"],
                citation_support_window_years=detection_params[
                    "citation_support_window_years"
                ],
                # Data Filtering parameters
                min_papers_per_year=detection_params["min_papers_per_year"],
                # Objective function parameters
                cohesion_weight=objective_params["cohesion_weight"],
                separation_weight=objective_params["separation_weight"],
                top_k_keywords=objective_params["top_k_keywords"],
                min_keyword_frequency_ratio=objective_params[
                    "min_keyword_frequency_ratio"
                ],
                # Anti-gaming parameters
                anti_gaming_min_segment_size=anti_gaming_params["min_segment_size"],
                anti_gaming_size_weight_power=anti_gaming_params["size_weight_power"],
                anti_gaming_segment_count_penalty_sigma=anti_gaming_params[
                    "segment_count_penalty_sigma"
                ],
                anti_gaming_enable_size_weighting=anti_gaming_params[
                    "enable_size_weighting"
                ],
                anti_gaming_enable_segment_floor=anti_gaming_params[
                    "enable_segment_floor"
                ],
                anti_gaming_enable_count_penalty=anti_gaming_params[
                    "enable_count_penalty"
                ],
                # Boundary optimization parameters
                boundary_optimization_enabled=boundary_params["enabled"],
                boundary_search_window=boundary_params["search_window"],
                boundary_max_exhaustive_signals=boundary_params[
                    "max_exhaustive_signals"
                ],
                boundary_max_subset_enumeration=boundary_params[
                    "max_subset_enumeration"
                ],
                boundary_beam_width=boundary_params["beam_width"],
                # System parameters
                domain_name=domain_name,
                # Diagnostic parameters
                save_direction_diagnostics=diagnostic_params.get(
                    "save_direction_diagnostics", False
                ),
                diagnostic_top_keywords_limit=diagnostic_params.get(
                    "diagnostic_top_keywords_limit", 10
                ),
            )
        except KeyError as e:
            raise ValueError(f"Missing required parameter in configuration file: {e}")

    def __post_init__(self):
        """Validate all parameters after initialization."""
        self._validate_parameters()

    def _validate_parameters(self):
        """Validate all parameter values."""
        # Direction Change Detection thresholds
        if not 0.0 <= self.direction_change_threshold <= 1.0:
            raise ValueError(
                f"direction_change_threshold must be 0.0-1.0, got {self.direction_change_threshold}"
            )

        valid_direction_strategies = ["fixed", "global_p90", "global_p95", "global_p99"]
        if self.direction_threshold_strategy not in valid_direction_strategies:
            raise ValueError(
                f"direction_threshold_strategy must be one of {valid_direction_strategies}, got {self.direction_threshold_strategy}"
            )

        valid_scoring_methods = ["weighted_jaccard", "jensen_shannon"]
        if self.direction_scoring_method not in valid_scoring_methods:
            raise ValueError(
                f"direction_scoring_method must be one of {valid_scoring_methods}, got {self.direction_scoring_method}"
            )

        if not 1 <= self.min_baseline_period_years <= 10:
            raise ValueError(
                f"min_baseline_period_years must be 1-10, got {self.min_baseline_period_years}"
            )

        if not 1 <= self.score_distribution_window_years <= 10:
            raise ValueError(
                f"score_distribution_window_years must be 1-10, got {self.score_distribution_window_years}"
            )

        # Validation parameters
        # Citation Analysis parameters
        if not 0.0 <= self.citation_confidence_boost <= 1.0:
            raise ValueError(
                f"citation_confidence_boost must be 0.0-1.0, got {self.citation_confidence_boost}"
            )

        if not 1 <= self.citation_support_window_years <= 10:
            raise ValueError(
                f"citation_support_window_years must be 1-10, got {self.citation_support_window_years}"
            )

        if not 1 <= self.diagnostic_top_keywords_limit <= 50:
            raise ValueError(
                f"diagnostic_top_keywords_limit must be 1-50, got {self.diagnostic_top_keywords_limit}"
            )

        # Data Filtering parameters
        if not 1 <= self.min_papers_per_year <= 1000:
            raise ValueError(
                f"min_papers_per_year must be 1-1000, got {self.min_papers_per_year}"
            )

        # Objective function parameters
        if not 0.0 <= self.cohesion_weight <= 1.0:
            raise ValueError(
                f"cohesion_weight must be 0.0-1.0, got {self.cohesion_weight}"
            )

        if not 0.0 <= self.separation_weight <= 1.0:
            raise ValueError(
                f"separation_weight must be 0.0-1.0, got {self.separation_weight}"
            )

        if abs(self.cohesion_weight + self.separation_weight - 1.0) > 1e-6:
            raise ValueError(
                f"cohesion_weight + separation_weight must equal 1.0, got {self.cohesion_weight + self.separation_weight:.6f}"
            )

        if not 1 <= self.top_k_keywords <= 100:
            raise ValueError(f"top_k_keywords must be 1-100, got {self.top_k_keywords}")

        if not 0.0 <= self.min_keyword_frequency_ratio <= 1.0:
            raise ValueError(
                f"min_keyword_frequency_ratio must be 0.0-1.0, got {self.min_keyword_frequency_ratio}"
            )

        # Anti-gaming parameters
        if not 1 <= self.anti_gaming_min_segment_size <= 200:
            raise ValueError(
                f"anti_gaming_min_segment_size must be 1-200, got {self.anti_gaming_min_segment_size}"
            )

        if not 0.0 <= self.anti_gaming_size_weight_power <= 2.0:
            raise ValueError(
                f"anti_gaming_size_weight_power must be 0.0-2.0, got {self.anti_gaming_size_weight_power}"
            )

        if not 0.5 <= self.anti_gaming_segment_count_penalty_sigma <= 20.0:
            raise ValueError(
                f"anti_gaming_segment_count_penalty_sigma must be 0.5-20.0, got {self.anti_gaming_segment_count_penalty_sigma}"
            )

        # Boundary optimization parameters
        if not 1 <= self.boundary_search_window <= 10:
            raise ValueError(
                f"boundary_search_window must be 1-10, got {self.boundary_search_window}"
            )

        if not 1 <= self.boundary_max_exhaustive_signals <= 20:
            raise ValueError(
                f"boundary_max_exhaustive_signals must be 1-20, got {self.boundary_max_exhaustive_signals}"
            )

        if not 1 <= self.boundary_max_subset_enumeration <= 30:
            raise ValueError(
                f"boundary_max_subset_enumeration must be 1-30, got {self.boundary_max_subset_enumeration}"
            )

        if not 5 <= self.boundary_beam_width <= 100:
            raise ValueError(
                f"boundary_beam_width must be 5-100, got {self.boundary_beam_width}"
            )

        # Citation Analysis scales

    def get_anti_gaming_config(self):
        """Get anti-gaming configuration for the objective function.

        Returns:
            AntiGamingConfig: Configuration object for anti-gaming parameters
        """
        return AntiGamingConfig(
            min_segment_size=self.anti_gaming_min_segment_size,
            size_weight_power=self.anti_gaming_size_weight_power,
            segment_count_penalty_sigma=self.anti_gaming_segment_count_penalty_sigma,
            enable_size_weighting=self.anti_gaming_enable_size_weighting,
            enable_segment_floor=self.anti_gaming_enable_segment_floor,
            enable_count_penalty=self.anti_gaming_enable_count_penalty,
        )

    def get_configuration_summary(self) -> str:
        """Get a concise summary of the current configuration.

        Returns:
            String summary of key configuration parameters
        """
        return (
            f"direction_change_threshold={self.direction_change_threshold:.3f}, "
            f"cohesion_weight={self.cohesion_weight}, "
            f"separation_weight={self.separation_weight}, "
            f"top_k_keywords={self.top_k_keywords}, "
            f"anti_gaming_enabled={self.anti_gaming_enable_size_weighting}"
        )

    def __str__(self) -> str:
        """String representation showing key parameters.

        Returns:
            Formatted string showing main configuration values
        """
        return (
            f"AlgorithmConfig(direction_change_threshold={self.direction_change_threshold:.3f}, "
            f"objective_weights=({self.cohesion_weight}, {self.separation_weight}), "
            f"anti_gaming={self.anti_gaming_enable_size_weighting})"
        )
