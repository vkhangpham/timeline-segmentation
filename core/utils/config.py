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
    direction_threshold_strategy: str
    direction_scoring_method: str
    min_baseline_period_years: int
    score_distribution_window_years: int
    min_paper_per_segment: int

    # Citation Analysis Parameters
    citation_confidence_boost: float
    citation_support_window_years: int

    # Data Filtering Parameters
    min_papers_per_year: int

    # Objective Function Parameters
    cohesion_weight: float
    separation_weight: float
    top_k_keywords: int
    min_keyword_frequency_ratio: float

    # Ubiquitous Keyword Filtering Parameters
    apply_ubiquitous_filtering: bool
    ubiquity_threshold: float
    max_ubiquitous_iterations: int
    min_replacement_frequency: int

    # Beam Search Refinement Parameters
    beam_search_enabled: bool
    beam_width: int
    max_splits_per_segment: int

    # Penalty System Parameters
    penalty_min_period_years: int
    penalty_max_period_years: int
    penalty_auto_n_upper: bool
    penalty_n_upper_buffer: int
    penalty_lambda_short: float
    penalty_lambda_long: float
    penalty_lambda_count: float
    penalty_enable_scaling: bool
    penalty_scaling_factor: float

    # Diagnostic Parameters
    save_direction_diagnostics: bool
    diagnostic_top_keywords_limit: int

    # System Parameters
    domain_name: Optional[str] = None

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
            # Extract nested configuration sections
            segmentation = config["segmentation"]
            change_detection = segmentation["change_detection"]
            citation_analysis = segmentation["citation_analysis"]
            beam_refinement = segmentation["beam_refinement"]

            optimization = config["optimization"]
            objective_function = optimization["objective_function"]
            penalty = optimization["penalty"]

            data_processing = config["data_processing"]
            year_filter = data_processing["year_filter"]
            keyword_filter = data_processing["keyword_filter"]
            ubiquitous_filtering = data_processing["ubiquitous_filtering"]

            system = config["system"]
            diagnostics = system["diagnostics"]

            return cls(
                # Direction Change Detection parameters
                direction_change_threshold=change_detection[
                    "direction_change_threshold"
                ],
                direction_threshold_strategy=change_detection[
                    "direction_threshold_strategy"
                ],
                direction_scoring_method=change_detection[
                    "direction_scoring_method"
                ],
                min_baseline_period_years=change_detection[
                    "min_baseline_period_years"
                ],
                score_distribution_window_years=change_detection[
                    "score_distribution_window_years"
                ],
                min_paper_per_segment=change_detection[
                    "min_paper_per_segment"
                ],
                min_papers_per_year=year_filter["min_papers_per_year"],
                # Citation Analysis parameters
                citation_confidence_boost=citation_analysis[
                    "citation_confidence_boost"
                ],
                citation_support_window_years=citation_analysis[
                    "citation_support_window_years"
                ],
                # Objective function parameters
                cohesion_weight=objective_function["cohesion_weight"],
                separation_weight=objective_function["separation_weight"],
                top_k_keywords=keyword_filter["top_k_keywords"],
                min_keyword_frequency_ratio=keyword_filter[
                    "min_keyword_frequency_ratio"
                ],
                # Penalty system parameters
                penalty_min_period_years=penalty["min_period_years"],
                penalty_max_period_years=penalty["max_period_years"],
                penalty_auto_n_upper=penalty["auto_n_upper"],
                penalty_n_upper_buffer=penalty["n_upper_buffer"],
                penalty_lambda_short=penalty["lambda_short"],
                penalty_lambda_long=penalty["lambda_long"],
                penalty_lambda_count=penalty["lambda_count"],
                penalty_enable_scaling=penalty["enable_scaling"],
                penalty_scaling_factor=penalty["scaling_factor"],
                # Ubiquitous keyword filtering parameters
                apply_ubiquitous_filtering=ubiquitous_filtering[
                    "apply_ubiquitous_filtering"
                ],
                ubiquity_threshold=ubiquitous_filtering["ubiquity_threshold"],
                max_ubiquitous_iterations=ubiquitous_filtering[
                    "max_iterations"
                ],
                min_replacement_frequency=ubiquitous_filtering[
                    "min_replacement_frequency"
                ],
                # Beam search refinement parameters
                beam_search_enabled=beam_refinement["enabled"],
                beam_width=beam_refinement["beam_width"],
                max_splits_per_segment=beam_refinement["max_splits_per_segment"],
                # Diagnostic parameters
                save_direction_diagnostics=diagnostics[
                    "save_direction_diagnostics"
                ],
                diagnostic_top_keywords_limit=diagnostics[
                    "diagnostic_top_keywords_limit"
                ],
                # System parameters
                domain_name=domain_name,
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

        if not 1 <= self.min_paper_per_segment <= 10000:
            raise ValueError(
                f"min_paper_per_segment must be 1-10000, got {self.min_paper_per_segment}"
            )

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

        # Ubiquitous keyword filtering parameters
        if not 0.0 <= self.ubiquity_threshold <= 1.0:
            raise ValueError(
                f"ubiquity_threshold must be 0.0-1.0, got {self.ubiquity_threshold}"
            )

        if not 1 <= self.max_ubiquitous_iterations <= 50:
            raise ValueError(
                f"max_ubiquitous_iterations must be 1-50, got {self.max_ubiquitous_iterations}"
            )

        if not 1 <= self.min_replacement_frequency <= 100:
            raise ValueError(
                f"min_replacement_frequency must be 1-100, got {self.min_replacement_frequency}"
            )

        # Beam search refinement parameters
        if not 1 <= self.beam_width <= 50:
            raise ValueError(f"beam_width must be 1-50, got {self.beam_width}")

        if not 0 <= self.max_splits_per_segment <= 10:
            raise ValueError(
                f"max_splits_per_segment must be 0-10, got {self.max_splits_per_segment}"
            )

        # Penalty system parameters
        if not 1 <= self.penalty_min_period_years <= 20:
            raise ValueError(
                f"penalty_min_period_years must be 1-20, got {self.penalty_min_period_years}"
            )

        if not 5 <= self.penalty_max_period_years <= 200:
            raise ValueError(
                f"penalty_max_period_years must be 5-200, got {self.penalty_max_period_years}"
            )

        if self.penalty_min_period_years >= self.penalty_max_period_years:
            raise ValueError(
                f"penalty_min_period_years ({self.penalty_min_period_years}) must be less than penalty_max_period_years ({self.penalty_max_period_years})"
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
            f"ubiquitous_filtering={'ON' if self.apply_ubiquitous_filtering else 'OFF'}, "
            f"ubiquity_threshold={self.ubiquity_threshold:.2f}"
        )

    def __str__(self) -> str:
        """String representation showing key parameters.

        Returns:
            Formatted string showing main configuration values
        """
        return (
            f"AlgorithmConfig(direction_change_threshold={self.direction_change_threshold:.3f}, "
            f"objective_weights=({self.cohesion_weight}, {self.separation_weight}), "
            f"ubiquitous_filtering={self.apply_ubiquitous_filtering}, "
            f"ubiquity_threshold={self.ubiquity_threshold:.2f})"
        )
