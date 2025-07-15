#!/usr/bin/env python3
"""Analysis script for segment score distribution experiment results."""

import json
import pandas as pd


def load_results(
    results_file: str = "experiments/score_distribution_analysis/results/segment_score_distributions.json",
):
    """Load experiment results from JSON file."""
    with open(results_file, "r") as f:
        return json.load(f)


def analyze_cohesion_scores(cohesion_results):
    """Analyze cohesion score distributions."""
    print("=== COHESION SCORE ANALYSIS ===")

    df = pd.DataFrame(cohesion_results)

    # Overall statistics
    print(f"Total samples: {len(df)}")
    print(f"Mean cohesion: {df['cohesion_score'].mean():.3f}")
    print(f"Std cohesion: {df['cohesion_score'].std():.3f}")
    print(f"Min cohesion: {df['cohesion_score'].min():.3f}")
    print(f"Max cohesion: {df['cohesion_score'].max():.3f}")

    # By domain
    print("\nBy Domain:")
    for domain in df["domain"].unique():
        domain_data = df[df["domain"] == domain]
        print(
            f"  {domain}: mean={domain_data['cohesion_score'].mean():.3f}, "
            f"std={domain_data['cohesion_score'].std():.3f}, "
            f"n={len(domain_data)}"
        )

    # By segment size
    print("\nBy Segment Size:")
    for size in sorted(df["segment_size"].unique()):
        size_data = df[df["segment_size"] == size]
        print(
            f"  {size} years: mean={size_data['cohesion_score'].mean():.3f}, "
            f"std={size_data['cohesion_score'].std():.3f}, "
            f"n={len(size_data)}"
        )

    # By temporal position
    if "temporal_position" in df.columns:
        print("\nBy Temporal Position:")
        for pos in df["temporal_position"].unique():
            pos_data = df[df["temporal_position"] == pos]
            print(
                f"  {pos}: mean={pos_data['cohesion_score'].mean():.3f}, "
                f"std={pos_data['cohesion_score'].std():.3f}, "
                f"n={len(pos_data)}"
            )

    return df


def analyze_separation_scores(separation_results):
    """Analyze separation score distributions."""
    print("\n=== SEPARATION SCORE ANALYSIS ===")

    df = pd.DataFrame(separation_results)

    # Overall statistics
    print(f"Total samples: {len(df)}")
    print(f"Mean separation: {df['separation_score'].mean():.3f}")
    print(f"Std separation: {df['separation_score'].std():.3f}")
    print(f"Min separation: {df['separation_score'].min():.3f}")
    print(f"Max separation: {df['separation_score'].max():.3f}")

    # By domain
    print("\nBy Domain:")
    for domain in df["domain"].unique():
        domain_data = df[df["domain"] == domain]
        print(
            f"  {domain}: mean={domain_data['separation_score'].mean():.3f}, "
            f"std={domain_data['separation_score'].std():.3f}, "
            f"n={len(domain_data)}"
        )

    # By temporal distance
    print("\nBy Temporal Distance:")
    df["temporal_distance_cat"] = pd.cut(
        df["temporal_distance"],
        bins=[0, 10, 50, 100, float("inf")],
        labels=["0-10", "11-50", "51-100", "100+"],
    )
    for cat in df["temporal_distance_cat"].unique():
        if pd.isna(cat):
            continue
        cat_data = df[df["temporal_distance_cat"] == cat]
        print(
            f"  {cat} years: mean={cat_data['separation_score'].mean():.3f}, "
            f"std={cat_data['separation_score'].std():.3f}, "
            f"n={len(cat_data)}"
        )

    # By overlap
    print("\nBy Temporal Overlap:")
    df["has_overlap"] = df["temporal_overlap"] > 0
    for has_overlap in [True, False]:
        overlap_data = df[df["has_overlap"] == has_overlap]
        label = "overlapping" if has_overlap else "non-overlapping"
        print(
            f"  {label}: mean={overlap_data['separation_score'].mean():.3f}, "
            f"std={overlap_data['separation_score'].std():.3f}, "
            f"n={len(overlap_data)}"
        )

    return df


def correlation_analysis(cohesion_df, separation_df):
    """Analyze correlations between scores and segment characteristics."""
    print("\n=== CORRELATION ANALYSIS ===")

    # Cohesion correlations
    print("Cohesion Score Correlations:")
    cohesion_corr = cohesion_df[
        [
            "cohesion_score",
            "segment_size",
            "total_papers",
            "total_citations",
            "avg_papers_per_year",
        ]
    ].corr()
    print(cohesion_corr["cohesion_score"].sort_values(ascending=False))

    # Separation correlations
    print("\nSeparation Score Correlations:")
    separation_corr = separation_df[
        [
            "separation_score",
            "temporal_distance",
            "temporal_overlap",
            "seg1_size",
            "seg2_size",
        ]
    ].corr()
    print(separation_corr["separation_score"].sort_values(ascending=False))


def comparative_analysis(cohesion_df, separation_df):
    """Compare cohesion and separation score distributions."""
    print("\n=== COMPARATIVE ANALYSIS ===")

    print("Score Distribution Comparison:")
    print(f"Cohesion mean: {cohesion_df['cohesion_score'].mean():.3f}")
    print(f"Separation mean: {separation_df['separation_score'].mean():.3f}")
    print(f"Cohesion std: {cohesion_df['cohesion_score'].std():.3f}")
    print(f"Separation std: {separation_df['separation_score'].std():.3f}")

    # Range analysis
    print(f"\nRange Analysis:")
    print(
        f"Cohesion range: [{cohesion_df['cohesion_score'].min():.3f}, {cohesion_df['cohesion_score'].max():.3f}]"
    )
    print(
        f"Separation range: [{separation_df['separation_score'].min():.3f}, {separation_df['separation_score'].max():.3f}]"
    )

    # Quartile analysis
    print(f"\nQuartile Analysis:")
    print(
        f"Cohesion quartiles: {cohesion_df['cohesion_score'].quantile([0.25, 0.5, 0.75]).values}"
    )
    print(
        f"Separation quartiles: {separation_df['separation_score'].quantile([0.25, 0.5, 0.75]).values}"
    )


def main():
    """Main analysis execution."""
    print("Segment Score Distribution Analysis")
    print("=" * 50)

    # Load results
    results = load_results()

    # Display experiment info
    print("EXPERIMENT INFO:")
    info = results["experiment_info"]
    print(f"Timestamp: {info['timestamp']}")
    print(f"Domains: {', '.join(info['domains'])}")
    print(f"K samples per domain: {info['k_samples']}")
    print(f"Separation pairs per domain: {info['separation_pairs']}")
    print(f"Total cohesion samples: {info['total_cohesion_samples']}")
    print(f"Total separation samples: {info['total_separation_samples']}")

    # Analyze cohesion scores
    cohesion_df = analyze_cohesion_scores(results["cohesion_results"])

    # Analyze separation scores
    separation_df = analyze_separation_scores(results["separation_results"])

    # Correlation analysis
    correlation_analysis(cohesion_df, separation_df)

    # Comparative analysis
    comparative_analysis(cohesion_df, separation_df)

    print("\n" + "=" * 50)
    print("Analysis complete! Check the visualization files:")
    print("- experiments/results/cohesion_distributions.png")
    print("- experiments/results/separation_distributions.png")
    print("- experiments/results/combined_analysis.png")


if __name__ == "__main__":
    main()
