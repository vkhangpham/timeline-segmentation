"""Algorithm configuration for timeline segmentation.

Configuration is loaded from config.yaml file with comprehensive validation.
"""

from dataclasses import dataclass
from typing import Optional, NamedTuple
import os
import yaml


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
    """Algorithm configuration loaded from config.yaml file."""

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

    # Beam Search Refinement Parameters
    beam_search_enabled: bool
    beam_width: int
    max_splits_per_segment: int
    min_period_years: int
    max_period_years: int

    # System Parameters
    domain_name: Optional[str] = None

    # Diagnostic Parameters
    save_direction_diagnostics: bool = False

    @classmethod
    def from_config_file(
        cls, config_path: str = "config/config.yaml", domain_name: Optional[str] = None
    ) -> "AlgorithmConfig":
        """Load configuration from YAML file.

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
                config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in configuration file {config_path}: {e}")

        try:
            detection_params = config["detection_parameters"]
            objective_params = config["objective_function"]
            beam_params = config.get("beam_search", {})
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
                # Beam search refinement parameters
                beam_search_enabled=beam_params.get("enabled", False),
                beam_width=beam_params.get("beam_width", 5),
                max_splits_per_segment=beam_params.get("max_splits_per_segment", 1),
                min_period_years=beam_params.get("min_period_years", 3),
                max_period_years=beam_params.get("max_period_years", 50),
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

        # Beam search refinement parameters
        if not 1 <= self.beam_width <= 50:
            raise ValueError(f"beam_width must be 1-50, got {self.beam_width}")

        if not 0 <= self.max_splits_per_segment <= 10:
            raise ValueError(
                f"max_splits_per_segment must be 0-10, got {self.max_splits_per_segment}"
            )

        if not 1 <= self.min_period_years <= 20:
            raise ValueError(
                f"min_period_years must be 1-20, got {self.min_period_years}"
            )

        if not 5 <= self.max_period_years <= 200:
            raise ValueError(
                f"max_period_years must be 5-200, got {self.max_period_years}"
            )

        if self.min_period_years >= self.max_period_years:
            raise ValueError(
                f"min_period_years ({self.min_period_years}) must be less than max_period_years ({self.max_period_years})"
            )

        # Citation Analysis scales

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
        )

    def __str__(self) -> str:
        """String representation showing key parameters.

        Returns:
            Formatted string showing main configuration values
        """
        return (
            f"AlgorithmConfig(direction_change_threshold={self.direction_change_threshold:.3f}, "
            f"objective_weights=({self.cohesion_weight}, {self.separation_weight}), "
        )
