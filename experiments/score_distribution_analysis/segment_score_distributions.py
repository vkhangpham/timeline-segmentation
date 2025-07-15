#!/usr/bin/env python3
"""Experiment: Segment Score Distributions Analysis

This script randomly samples segments from domains and analyzes the distribution
of cohesion and separation scores with proper stratification.
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple
import json
from datetime import datetime
import random

from core.data.data_processing import (
    load_domain_data,
    create_academic_periods_from_segments,
)
from core.optimization.objective_function import (
    evaluate_period_cohesion,
    evaluate_period_separation,
)
from core.utils.config import AlgorithmConfig
from core.utils.logging import get_logger
from core.data.data_models import AcademicYear


class SegmentSampler:
    """Stratified segment sampler for score distribution analysis."""

    def __init__(self, algorithm_config: AlgorithmConfig, verbose: bool = False):
        self.algorithm_config = algorithm_config
        self.verbose = verbose
        self.logger = get_logger(__name__, verbose)

    def load_domain_data(
        self, domain_name: str
    ) -> Tuple[bool, List[AcademicYear], str]:
        """Load domain data with proper filtering."""
        return load_domain_data(
            domain_name=domain_name,
            algorithm_config=self.algorithm_config,
            data_directory="resources",
            verbose=self.verbose,
        )

    def create_stratified_segments(
        self,
        academic_years: List[AcademicYear],
        num_samples: int,
        min_segment_size: int = 3,
        max_segment_size: int = 15,
    ) -> List[Tuple[int, int]]:
        """Create stratified segments for sampling.

        Args:
            academic_years: List of academic years
            num_samples: Number of segments to sample
            min_segment_size: Minimum segment size in years
            max_segment_size: Maximum segment size in years

        Returns:
            List of (start_year, end_year) tuples representing segments
        """
        if not academic_years:
            return []

        years = sorted([ay.year for ay in academic_years])
        year_range = max(years) - min(years) + 1

        if year_range < min_segment_size:
            return []

        # Stratification dimensions
        size_strata = self._create_size_strata(min_segment_size, max_segment_size)
        temporal_strata = self._create_temporal_strata(years)

        segments = []
        samples_per_stratum = max(
            1, num_samples // (len(size_strata) * len(temporal_strata))
        )

        for size_stratum in size_strata:
            for temporal_stratum in temporal_strata:
                stratum_segments = self._sample_stratum(
                    years, size_stratum, temporal_stratum, samples_per_stratum
                )
                segments.extend(stratum_segments)

        # Fill remaining samples randomly if needed
        while len(segments) < num_samples:
            additional_segments = self._sample_random_segments(
                years, min_segment_size, max_segment_size, num_samples - len(segments)
            )
            segments.extend(additional_segments)

        return segments[:num_samples]

    def _create_size_strata(
        self, min_size: int, max_size: int
    ) -> List[Tuple[int, int]]:
        """Create size strata for segment sampling."""
        # Create 3 size categories: small, medium, large
        range_size = max_size - min_size + 1

        if range_size <= 3:
            return [(min_size, max_size)]

        small_end = min_size + range_size // 3
        medium_end = min_size + 2 * range_size // 3

        return [
            (min_size, small_end),
            (small_end + 1, medium_end),
            (medium_end + 1, max_size),
        ]

    def _create_temporal_strata(self, years: List[int]) -> List[str]:
        """Create temporal strata for segment sampling."""
        return ["early", "middle", "late"]

    def _sample_stratum(
        self,
        years: List[int],
        size_stratum: Tuple[int, int],
        temporal_stratum: str,
        num_samples: int,
    ) -> List[Tuple[int, int]]:
        """Sample segments from a specific stratum."""
        segments = []
        min_size, max_size = size_stratum

        # Define temporal windows
        total_years = len(years)
        if temporal_stratum == "early":
            window_start = 0
            window_end = min(total_years, total_years // 2)
        elif temporal_stratum == "middle":
            window_start = total_years // 4
            window_end = min(total_years, 3 * total_years // 4)
        else:  # late
            window_start = total_years // 2
            window_end = total_years

        window_years = years[window_start:window_end]

        if len(window_years) < min_size:
            return []

        # Sample segments within this stratum
        for _ in range(num_samples):
            segment_size = random.randint(min_size, min(max_size, len(window_years)))
            max_start_idx = len(window_years) - segment_size

            if max_start_idx < 0:
                continue

            start_idx = random.randint(0, max_start_idx)
            start_year = window_years[start_idx]
            end_year = window_years[start_idx + segment_size - 1]

            segments.append((start_year, end_year))

        return segments

    def _sample_random_segments(
        self, years: List[int], min_size: int, max_size: int, num_samples: int
    ) -> List[Tuple[int, int]]:
        """Sample random segments without stratification."""
        segments = []

        for _ in range(num_samples):
            if len(years) < min_size:
                continue

            segment_size = random.randint(min_size, min(max_size, len(years)))
            max_start_idx = len(years) - segment_size

            if max_start_idx < 0:
                continue

            start_idx = random.randint(0, max_start_idx)
            start_year = years[start_idx]
            end_year = years[start_idx + segment_size - 1]

            segments.append((start_year, end_year))

        return segments


class ScoreDistributionAnalyzer:
    """Analyzer for cohesion and separation score distributions."""

    def __init__(self, algorithm_config: AlgorithmConfig, verbose: bool = False):
        self.algorithm_config = algorithm_config
        self.verbose = verbose
        self.logger = get_logger(__name__, verbose)

    def analyze_cohesion_scores(
        self,
        domain_name: str,
        academic_years: List[AcademicYear],
        segments: List[Tuple[int, int]],
    ) -> List[Dict]:
        """Analyze cohesion scores for segments."""
        cohesion_results = []

        for start_year, end_year in segments:
            try:
                # Create academic period
                periods = create_academic_periods_from_segments(
                    academic_years=tuple(academic_years),
                    segments=[(start_year, end_year)],
                    algorithm_config=self.algorithm_config,
                )

                if not periods:
                    continue

                period = periods[0]

                # Calculate cohesion
                cohesion_metrics = evaluate_period_cohesion(
                    period, top_k=self.algorithm_config.top_k_keywords
                )

                # Collect metadata
                result = {
                    "domain": domain_name,
                    "start_year": start_year,
                    "end_year": end_year,
                    "segment_size": end_year - start_year + 1,
                    "num_years": len(period.academic_years),
                    "total_papers": period.total_papers,
                    "total_citations": period.total_citations,
                    "keywords_count": cohesion_metrics.keywords_count,
                    "cohesion_score": cohesion_metrics.cohesion,
                    "top_keywords": cohesion_metrics.top_keywords[:5],
                    "avg_papers_per_year": (
                        period.total_papers / len(period.academic_years)
                        if period.academic_years
                        else 0
                    ),
                    "temporal_position": self._get_temporal_position(
                        start_year, end_year, academic_years
                    ),
                }

                cohesion_results.append(result)

            except Exception as e:
                if self.verbose:
                    self.logger.warning(
                        f"Failed to analyze cohesion for {start_year}-{end_year}: {e}"
                    )
                continue

        return cohesion_results

    def analyze_separation_scores(
        self,
        domain_name: str,
        academic_years: List[AcademicYear],
        segments: List[Tuple[int, int]],
        num_pairs: int = 1000,
    ) -> List[Dict]:
        """Analyze separation scores for segment pairs."""
        separation_results = []

        # Create academic periods for all segments
        periods = {}
        for start_year, end_year in segments:
            try:
                period_list = create_academic_periods_from_segments(
                    academic_years=tuple(academic_years),
                    segments=[(start_year, end_year)],
                    algorithm_config=self.algorithm_config,
                )
                if period_list:
                    periods[(start_year, end_year)] = period_list[0]
            except Exception as e:
                if self.verbose:
                    self.logger.warning(
                        f"Failed to create period for {start_year}-{end_year}: {e}"
                    )
                continue

        # Sample pairs
        segment_keys = list(periods.keys())
        if len(segment_keys) < 2:
            return []

        sampled_pairs = []
        for _ in range(
            min(num_pairs, len(segment_keys) * (len(segment_keys) - 1) // 2)
        ):
            # Sample two different segments
            seg1 = random.choice(segment_keys)
            seg2 = random.choice([s for s in segment_keys if s != seg1])
            sampled_pairs.append((seg1, seg2))

        # Calculate separation scores
        for (seg1_start, seg1_end), (seg2_start, seg2_end) in sampled_pairs:
            try:
                period1 = periods[(seg1_start, seg1_end)]
                period2 = periods[(seg2_start, seg2_end)]

                # Calculate separation
                separation_metrics = evaluate_period_separation(period1, period2)

                # Calculate temporal distance
                temporal_distance = self._calculate_temporal_distance(
                    (seg1_start, seg1_end), (seg2_start, seg2_end)
                )

                result = {
                    "domain": domain_name,
                    "seg1_start": seg1_start,
                    "seg1_end": seg1_end,
                    "seg2_start": seg2_start,
                    "seg2_end": seg2_end,
                    "seg1_size": seg1_end - seg1_start + 1,
                    "seg2_size": seg2_end - seg2_start + 1,
                    "seg1_papers": period1.total_papers,
                    "seg2_papers": period2.total_papers,
                    "separation_score": separation_metrics.separation,
                    "vocab_size": separation_metrics.vocab_size,
                    "temporal_distance": temporal_distance,
                    "temporal_overlap": self._calculate_temporal_overlap(
                        (seg1_start, seg1_end), (seg2_start, seg2_end)
                    ),
                }

                separation_results.append(result)

            except Exception as e:
                if self.verbose:
                    self.logger.warning(
                        f"Failed to analyze separation for {seg1_start}-{seg1_end} vs {seg2_start}-{seg2_end}: {e}"
                    )
                continue

        return separation_results

    def _get_temporal_position(
        self, start_year: int, end_year: int, academic_years: List[AcademicYear]
    ) -> str:
        """Get temporal position of segment (early/middle/late)."""
        all_years = sorted([ay.year for ay in academic_years])
        segment_mid = (start_year + end_year) / 2
        data_mid = (all_years[0] + all_years[-1]) / 2

        if segment_mid < data_mid - (all_years[-1] - all_years[0]) / 6:
            return "early"
        elif segment_mid > data_mid + (all_years[-1] - all_years[0]) / 6:
            return "late"
        else:
            return "middle"

    def _calculate_temporal_distance(
        self, seg1: Tuple[int, int], seg2: Tuple[int, int]
    ) -> int:
        """Calculate temporal distance between two segments."""
        seg1_start, seg1_end = seg1
        seg2_start, seg2_end = seg2

        # Distance between closest points
        if seg1_end < seg2_start:
            return seg2_start - seg1_end
        elif seg2_end < seg1_start:
            return seg1_start - seg2_end
        else:
            return 0  # Overlapping

    def _calculate_temporal_overlap(
        self, seg1: Tuple[int, int], seg2: Tuple[int, int]
    ) -> int:
        """Calculate temporal overlap between two segments."""
        seg1_start, seg1_end = seg1
        seg2_start, seg2_end = seg2

        overlap_start = max(seg1_start, seg2_start)
        overlap_end = min(seg1_end, seg2_end)

        return max(0, overlap_end - overlap_start + 1)


class VisualizationGenerator:
    """Generator for score distribution visualizations."""

    def __init__(
        self, output_dir: str = "experiments/score_distribution_analysis/results"
    ):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def plot_cohesion_distributions(
        self,
        cohesion_results: List[Dict],
        output_file: str = "cohesion_distributions.png",
    ):
        """Plot cohesion score distributions."""
        if not cohesion_results:
            return

        df = pd.DataFrame(cohesion_results)

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle("Cohesion Score Distributions", fontsize=16)

        # Overall distribution
        axes[0, 0].hist(df["cohesion_score"], bins=50, alpha=0.7, edgecolor="black")
        axes[0, 0].set_title("Overall Cohesion Distribution")
        axes[0, 0].set_xlabel("Cohesion Score")
        axes[0, 0].set_ylabel("Frequency")
        axes[0, 0].axvline(
            df["cohesion_score"].mean(),
            color="red",
            linestyle="--",
            label=f'Mean: {df["cohesion_score"].mean():.3f}',
        )
        axes[0, 0].legend()

        # By domain
        if "domain" in df.columns and df["domain"].nunique() > 1:
            for domain in df["domain"].unique():
                domain_data = df[df["domain"] == domain]
                axes[0, 1].hist(
                    domain_data["cohesion_score"],
                    bins=30,
                    alpha=0.6,
                    label=domain,
                    edgecolor="black",
                )
            axes[0, 1].set_title("Cohesion by Domain")
            axes[0, 1].set_xlabel("Cohesion Score")
            axes[0, 1].set_ylabel("Frequency")
            axes[0, 1].legend()

        # By segment size
        axes[1, 0].scatter(df["segment_size"], df["cohesion_score"], alpha=0.6)
        axes[1, 0].set_title("Cohesion vs Segment Size")
        axes[1, 0].set_xlabel("Segment Size (years)")
        axes[1, 0].set_ylabel("Cohesion Score")

        # By temporal position
        if "temporal_position" in df.columns:
            sns.boxplot(
                data=df, x="temporal_position", y="cohesion_score", ax=axes[1, 1]
            )
            axes[1, 1].set_title("Cohesion by Temporal Position")
            axes[1, 1].set_xlabel("Temporal Position")
            axes[1, 1].set_ylabel("Cohesion Score")

        plt.tight_layout()
        plt.savefig(
            os.path.join(self.output_dir, output_file), dpi=300, bbox_inches="tight"
        )
        plt.close()

    def plot_separation_distributions(
        self,
        separation_results: List[Dict],
        output_file: str = "separation_distributions.png",
    ):
        """Plot separation score distributions."""
        if not separation_results:
            return

        df = pd.DataFrame(separation_results)

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle("Separation Score Distributions", fontsize=16)

        # Overall distribution
        axes[0, 0].hist(df["separation_score"], bins=50, alpha=0.7, edgecolor="black")
        axes[0, 0].set_title("Overall Separation Distribution")
        axes[0, 0].set_xlabel("Separation Score")
        axes[0, 0].set_ylabel("Frequency")
        axes[0, 0].axvline(
            df["separation_score"].mean(),
            color="red",
            linestyle="--",
            label=f'Mean: {df["separation_score"].mean():.3f}',
        )
        axes[0, 0].legend()

        # By temporal distance
        axes[0, 1].scatter(df["temporal_distance"], df["separation_score"], alpha=0.6)
        axes[0, 1].set_title("Separation vs Temporal Distance")
        axes[0, 1].set_xlabel("Temporal Distance (years)")
        axes[0, 1].set_ylabel("Separation Score")

        # By segment size difference
        df["size_diff"] = abs(df["seg1_size"] - df["seg2_size"])
        axes[1, 0].scatter(df["size_diff"], df["separation_score"], alpha=0.6)
        axes[1, 0].set_title("Separation vs Size Difference")
        axes[1, 0].set_xlabel("Size Difference (years)")
        axes[1, 0].set_ylabel("Separation Score")

        # By domain
        if "domain" in df.columns and df["domain"].nunique() > 1:
            for domain in df["domain"].unique():
                domain_data = df[df["domain"] == domain]
                axes[1, 1].hist(
                    domain_data["separation_score"],
                    bins=30,
                    alpha=0.6,
                    label=domain,
                    edgecolor="black",
                )
            axes[1, 1].set_title("Separation by Domain")
            axes[1, 1].set_xlabel("Separation Score")
            axes[1, 1].set_ylabel("Frequency")
            axes[1, 1].legend()

        plt.tight_layout()
        plt.savefig(
            os.path.join(self.output_dir, output_file), dpi=300, bbox_inches="tight"
        )
        plt.close()

    def plot_combined_analysis(
        self,
        cohesion_results: List[Dict],
        separation_results: List[Dict],
        output_file: str = "combined_analysis.png",
    ):
        """Plot combined cohesion and separation analysis."""
        if not cohesion_results or not separation_results:
            return

        cohesion_df = pd.DataFrame(cohesion_results)
        separation_df = pd.DataFrame(separation_results)

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle("Combined Cohesion and Separation Analysis", fontsize=16)

        # Score distributions comparison
        axes[0, 0].hist(
            cohesion_df["cohesion_score"],
            bins=40,
            alpha=0.7,
            label="Cohesion",
            edgecolor="black",
        )
        axes[0, 0].hist(
            separation_df["separation_score"],
            bins=40,
            alpha=0.7,
            label="Separation",
            edgecolor="black",
        )
        axes[0, 0].set_title("Score Distributions Comparison")
        axes[0, 0].set_xlabel("Score")
        axes[0, 0].set_ylabel("Frequency")
        axes[0, 0].legend()

        # Summary statistics
        stats_text = f"""
        Cohesion Statistics:
        Mean: {cohesion_df['cohesion_score'].mean():.3f}
        Std: {cohesion_df['cohesion_score'].std():.3f}
        Min: {cohesion_df['cohesion_score'].min():.3f}
        Max: {cohesion_df['cohesion_score'].max():.3f}
        
        Separation Statistics:
        Mean: {separation_df['separation_score'].mean():.3f}
        Std: {separation_df['separation_score'].std():.3f}
        Min: {separation_df['separation_score'].min():.3f}
        Max: {separation_df['separation_score'].max():.3f}
        """

        axes[0, 1].text(
            0.05,
            0.95,
            stats_text,
            transform=axes[0, 1].transAxes,
            verticalalignment="top",
            fontfamily="monospace",
            fontsize=10,
        )
        axes[0, 1].set_title("Summary Statistics")
        axes[0, 1].axis("off")

        # Cohesion vs segment characteristics
        axes[1, 0].scatter(
            cohesion_df["total_papers"], cohesion_df["cohesion_score"], alpha=0.6
        )
        axes[1, 0].set_title("Cohesion vs Total Papers")
        axes[1, 0].set_xlabel("Total Papers")
        axes[1, 0].set_ylabel("Cohesion Score")
        axes[1, 0].set_xscale("log")

        # Separation vs temporal distance
        axes[1, 1].scatter(
            separation_df["temporal_distance"],
            separation_df["separation_score"],
            alpha=0.6,
        )
        axes[1, 1].set_title("Separation vs Temporal Distance")
        axes[1, 1].set_xlabel("Temporal Distance (years)")
        axes[1, 1].set_ylabel("Separation Score")

        plt.tight_layout()
        plt.savefig(
            os.path.join(self.output_dir, output_file), dpi=300, bbox_inches="tight"
        )
        plt.close()


def main():
    """Main experiment execution."""
    print("Starting Segment Score Distribution Experiment...")

    # Configuration
    config = AlgorithmConfig.from_config_file("config/config.yaml")

    # Available domains (excluding computer_science - no resources directory)
    domains = [
        "applied_mathematics",
        "art",
        "computer_vision",
        "deep_learning",
        "machine_learning",
        "machine_translation",
        "natural_language_processing",
    ]

    # Experiment parameters - Large scale experiment
    K_SAMPLES = 500  # Number of segments per domain (500 x 7 = 3500 total)
    SEPARATION_PAIRS = (
        500  # Number of separation pairs per domain (500 x 7 = 3500 total)
    )
    VERBOSE = True

    # Initialize components
    sampler = SegmentSampler(config, verbose=VERBOSE)
    analyzer = ScoreDistributionAnalyzer(config, verbose=VERBOSE)
    visualizer = VisualizationGenerator()

    # Results storage
    all_cohesion_results = []
    all_separation_results = []

    # Process each domain
    for domain in domains:
        print(f"\n=== Processing {domain} ===")

        # Load domain data
        success, academic_years, error_msg = sampler.load_domain_data(domain)
        if not success:
            print(f"Failed to load {domain}: {error_msg}")
            continue

        print(f"Loaded {len(academic_years)} academic years")

        # Create stratified segments
        segments = sampler.create_stratified_segments(
            academic_years, K_SAMPLES, min_segment_size=3, max_segment_size=12
        )

        print(f"Created {len(segments)} segments")

        # Analyze cohesion scores
        cohesion_results = analyzer.analyze_cohesion_scores(
            domain, academic_years, segments
        )
        all_cohesion_results.extend(cohesion_results)

        print(f"Analyzed {len(cohesion_results)} cohesion scores")

        # Analyze separation scores
        separation_results = analyzer.analyze_separation_scores(
            domain, academic_years, segments, num_pairs=SEPARATION_PAIRS
        )
        all_separation_results.extend(separation_results)

        print(f"Analyzed {len(separation_results)} separation scores")

    # Generate visualizations
    print("\n=== Generating Visualizations ===")
    visualizer.plot_cohesion_distributions(all_cohesion_results)
    visualizer.plot_separation_distributions(all_separation_results)
    visualizer.plot_combined_analysis(all_cohesion_results, all_separation_results)

    # Save results
    print("\n=== Saving Results ===")
    results_data = {
        "experiment_info": {
            "timestamp": datetime.now().isoformat(),
            "domains": domains,
            "k_samples": K_SAMPLES,
            "separation_pairs": SEPARATION_PAIRS,
            "total_cohesion_samples": len(all_cohesion_results),
            "total_separation_samples": len(all_separation_results),
        },
        "cohesion_results": all_cohesion_results,
        "separation_results": all_separation_results,
    }

    output_path = os.path.join(
        visualizer.output_dir, "segment_score_distributions.json"
    )
    with open(output_path, "w") as f:
        json.dump(results_data, f, indent=2, default=str)

    print(f"Results saved to: {output_path}")
    print(f"Visualizations saved to: {visualizer.output_dir}")

    # Print summary statistics
    print("\n=== Summary Statistics ===")
    if all_cohesion_results:
        cohesion_scores = [r["cohesion_score"] for r in all_cohesion_results]
        print(
            f"Cohesion scores: mean={np.mean(cohesion_scores):.3f}, "
            f"std={np.std(cohesion_scores):.3f}, "
            f"min={np.min(cohesion_scores):.3f}, "
            f"max={np.max(cohesion_scores):.3f}"
        )

    if all_separation_results:
        separation_scores = [r["separation_score"] for r in all_separation_results]
        print(
            f"Separation scores: mean={np.mean(separation_scores):.3f}, "
            f"std={np.std(separation_scores):.3f}, "
            f"min={np.min(separation_scores):.3f}, "
            f"max={np.max(separation_scores):.3f}"
        )

    print("\nExperiment completed successfully!")


if __name__ == "__main__":
    main()
