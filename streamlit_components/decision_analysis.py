"""
Decision Tree Analysis Tab Components

Contains all visualization functions for algorithm decision transparency and analysis.
This corresponds to Tab 3 in the main dashboard.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from collections import Counter, defaultdict
from typing import Dict, List, Optional

from core.algorithm_config import AlgorithmConfig
from core.data_models import DomainData


def create_decision_tree_analysis(
    domain_name: str,
    validated_signals: List,
    enhanced_signal_data: Dict,
    algorithm_config: AlgorithmConfig,
    domain_data: DomainData,
) -> List[Dict]:
    """Create comprehensive decision tree analysis for algorithm transparency."""

    clustering_metadata = enhanced_signal_data.get("clustering_metadata", {})
    clustered_direction_signals = clustering_metadata.get(
        "processed_direction_signals", []
    )
    citation_signals = clustering_metadata.get("citation_signals", [])
    keyword_analysis = enhanced_signal_data.get("keyword_analysis", {})

    # Extract decision details for each signal
    decision_details = []
    validated_years = {s.year for s in validated_signals}

    for direction_signal in clustered_direction_signals:
        year = direction_signal.year
        base_confidence = direction_signal.confidence

        # Check for citation support
        citation_support = any(abs(cs.year - year) <= 2 for cs in citation_signals)
        citation_years = [
            cs.year for cs in citation_signals if abs(cs.year - year) <= 2
        ]

        # Calculate confidence boosts
        confidence_boosts = 0.0
        boost_details = []

        if citation_support:
            dynamic_boost = 0.5 * base_confidence  # Dynamic boost: 50% of base confidence
            confidence_boosts += dynamic_boost
            boost_details.append(
                f"Citation Support (+{dynamic_boost:.3f}, 50% of base)"
            )

        # Add keyword filtering information
        keyword_filtering_info = {
            "enabled": algorithm_config.keyword_filtering_enabled,
            "rationale": (
                "Keyword filtering disabled"
                if not algorithm_config.keyword_filtering_enabled
                else "Keywords passed conservative filter"
            ),
            "impact": (
                "None"
                if not algorithm_config.keyword_filtering_enabled
                else "Noise reduction applied"
            ),
        }

        # Final confidence calculation
        final_confidence = min(base_confidence + confidence_boosts, 1.0)

        # Decision outcome
        is_validated = year in validated_years
        decision_outcome = "ACCEPTED" if is_validated else "REJECTED"

        # Decision rationale with keyword filtering context
        if final_confidence >= algorithm_config.validation_threshold:
            if is_validated:
                rationale = f"✅ Passed threshold ({final_confidence:.3f} ≥ {algorithm_config.validation_threshold:.3f})"
            else:
                rationale = f"🔄 Should pass but not in final results - check algorithm"
        else:
            rationale = f"❌ Below threshold ({final_confidence:.3f} < {algorithm_config.validation_threshold:.3f})"

        # Add keyword filtering context to rationale
        if algorithm_config.keyword_filtering_enabled:
            rationale += f" | 🔍 Keywords filtered (conservative settings)"
        else:
            rationale += f" | 🔍 No keyword filtering applied"

        decision_details.append(
            {
                "year": year,
                "base_confidence": base_confidence,
                "citation_support": citation_support,
                "citation_years": citation_years,
                "confidence_boosts": confidence_boosts,
                "boost_details": boost_details,
                "final_confidence": final_confidence,
                "threshold": algorithm_config.validation_threshold,
                "decision_outcome": decision_outcome,
                "rationale": rationale,
                "signal_type": direction_signal.signal_type,
                "evidence_strength": direction_signal.evidence_strength,
                "supporting_evidence": list(direction_signal.supporting_evidence)[:3],
                "keyword_filtering": keyword_filtering_info,
            }
        )

    return decision_details


def create_decision_flow_diagram(
    decision_details: List[Dict], domain_name: str
) -> go.Figure:
    """Create a decision flow diagram showing the algorithm's decision process."""

    fig = go.Figure()

    # Create a flowchart-style visualization
    y_positions = list(range(len(decision_details)))
    years = [d["year"] for d in decision_details]
    base_confidences = [d["base_confidence"] for d in decision_details]
    final_confidences = [d["final_confidence"] for d in decision_details]
    outcomes = [d["decision_outcome"] for d in decision_details]

    # Base confidence bars
    fig.add_trace(
        go.Bar(
            x=base_confidences,
            y=y_positions,
            orientation="h",
            name="Base Confidence",
            marker_color="lightblue",
            opacity=0.7,
            text=[f"Base: {conf:.3f}" for conf in base_confidences],
            textposition="inside",
            hovertemplate="<b>Year:</b> %{customdata}<br>"
            + "<b>Base Confidence:</b> %{x:.3f}<extra></extra>",
            customdata=years,
        )
    )

    # Final confidence bars
    fig.add_trace(
        go.Bar(
            x=final_confidences,
            y=y_positions,
            orientation="h",
            name="Final Confidence",
            marker_color=[
                "green" if outcome == "ACCEPTED" else "red" for outcome in outcomes
            ],
            opacity=0.8,
            text=[f"Final: {conf:.3f}" for conf in final_confidences],
            textposition="inside",
            hovertemplate="<b>Year:</b> %{customdata}<br>"
            + "<b>Final Confidence:</b> %{x:.3f}<br>"
            + "<b>Outcome:</b> %{text}<extra></extra>",
            customdata=years,
        )
    )

    # Add threshold line
    threshold = decision_details[0]["threshold"] if decision_details else 0.8
    fig.add_vline(
        x=threshold,
        line=dict(color="red", width=3, dash="dash"),
        annotation_text=f"Validation Threshold ({threshold:.2f})",
    )

    fig.update_layout(
        title=f"Decision Flow Analysis: {domain_name.replace('_', ' ').title()}",
        xaxis_title="Confidence Score",
        yaxis_title="Signals (by chronological order)",
        yaxis=dict(
            tickmode="array",
            tickvals=y_positions,
            ticktext=[f"{year} ({outcome})" for year, outcome in zip(years, outcomes)],
        ),
        height=max(400, len(decision_details) * 40),
        barmode="overlay",
        showlegend=True,
        template="plotly_white",
    )

    return fig


def create_parameter_sensitivity_analysis(
    decision_details: List[Dict], algorithm_config: AlgorithmConfig
) -> go.Figure:
    """Create parameter sensitivity analysis showing how changes affect outcomes."""

    if not decision_details:
        return go.Figure()

    # Test different threshold values
    test_thresholds = np.arange(0.5, 1.0, 0.05)
    sensitivity_results = []

    for threshold in test_thresholds:
        accepted_count = sum(
            1 for d in decision_details if d["final_confidence"] >= threshold
        )
        sensitivity_results.append(
            {
                "threshold": threshold,
                "accepted_signals": accepted_count,
                "acceptance_rate": accepted_count / len(decision_details),
            }
        )

    fig = make_subplots(
        rows=2,
        cols=1,
        subplot_titles=(
            "Threshold Impact on Signal Acceptance",
            "Boost Impact Analysis",
        ),
        vertical_spacing=0.12,
    )

    # Threshold sensitivity
    thresholds = [r["threshold"] for r in sensitivity_results]
    accepted_counts = [r["accepted_signals"] for r in sensitivity_results]

    fig.add_trace(
        go.Scatter(
            x=thresholds,
            y=accepted_counts,
            mode="lines+markers",
            name="Accepted Signals",
            line=dict(color="blue", width=3),
            marker=dict(size=8),
            hovertemplate="<b>Threshold:</b> %{x:.2f}<br>"
            + "<b>Accepted Signals:</b> %{y}<extra></extra>",
        ),
        row=1,
        col=1,
    )

    # Current threshold marker
    current_threshold = algorithm_config.validation_threshold
    current_accepted = sum(
        1 for d in decision_details if d["final_confidence"] >= current_threshold
    )

    fig.add_trace(
        go.Scatter(
            x=[current_threshold],
            y=[current_accepted],
            mode="markers",
            name="Current Setting",
            marker=dict(size=15, color="red", symbol="star"),
            hovertemplate="<b>Current Threshold:</b> %{x:.2f}<br>"
            + "<b>Current Accepted:</b> %{y}<extra></extra>",
        ),
        row=1,
        col=1,
    )

    # Boost impact analysis - Updated for dynamic boost (50% of base confidence)
    # Since boost is now dynamic, we'll show impact of confidence multipliers
    boost_multipliers = np.arange(0.0, 1.0, 0.1)  # 0% to 90% of base confidence
    boost_impact = []

    for multiplier in boost_multipliers:
        # Recalculate with different boost multiplier
        new_accepted = 0
        for d in decision_details:
            dynamic_boost = multiplier * d["base_confidence"] if d["citation_support"] else 0
            new_final_confidence = min(
                d["base_confidence"] + dynamic_boost, 1.0
            )
            if new_final_confidence >= current_threshold:
                new_accepted += 1
        boost_impact.append(new_accepted)

    fig.add_trace(
        go.Scatter(
            x=boost_multipliers * 100,  # Convert to percentage
            y=boost_impact,
            mode="lines+markers",
            name="Dynamic Boost Impact",
            line=dict(color="green", width=3),
            marker=dict(size=6),
            hovertemplate="<b>Boost Multiplier:</b> %{x:.0f}%<br>"
            + "<b>Accepted Signals:</b> %{y}<extra></extra>",
        ),
        row=2,
        col=1,
    )

    # Current boost marker (50% multiplier)
    current_boost_multiplier = 50  # 50% of base confidence
    current_boost_accepted = boost_impact[5] if len(boost_impact) > 5 else current_accepted  # Index 5 = 50%
    fig.add_trace(
        go.Scatter(
            x=[current_boost_multiplier],
            y=[current_boost_accepted],
            mode="markers",
            name="Current Dynamic Boost (50%)",
            marker=dict(size=15, color="red", symbol="star"),
            hovertemplate="<b>Current Boost:</b> 50% of base<br>"
            + "<b>Current Accepted:</b> %{y}<extra></extra>",
        ),
        row=2,
        col=1,
    )

    fig.update_layout(
        title="Parameter Sensitivity Analysis", 
        height=600, 
        showlegend=True,
        template="plotly_white",
    )

    fig.update_xaxes(title_text="Validation Threshold", row=1, col=1)
    fig.update_xaxes(title_text="Boost Multiplier (%)", row=2, col=1)
    fig.update_yaxes(title_text="Accepted Signals", row=1, col=1)
    fig.update_yaxes(title_text="Accepted Signals", row=2, col=1)

    return fig


def create_keyword_filtering_impact_analysis(
    domain_name: str,
    algorithm_config: AlgorithmConfig,
    enhanced_signal_data: Dict,
    domain_data: DomainData,
) -> Optional[Dict]:
    """Create detailed keyword filtering impact analysis."""

    if not algorithm_config.keyword_filtering_enabled:
        return None

    keyword_analysis = enhanced_signal_data.get("keyword_analysis", {})

    # Extract keyword filtering metrics
    filtering_metrics = {
        "filtering_enabled": True,
        "min_papers_ratio": algorithm_config.keyword_min_papers_ratio,
        "min_frequency": algorithm_config.keyword_min_frequency,
    }

    # Calculate domain-level keyword statistics
    all_keywords = []
    papers_by_year = defaultdict(list)

    for paper in domain_data.papers:
        if paper.pub_year and paper.keywords:
            papers_by_year[paper.pub_year].append(paper)
            all_keywords.extend(paper.keywords)

    keyword_frequency = Counter(all_keywords)
    unique_keywords = len(keyword_frequency)
    total_papers = len(domain_data.papers)

    # Calculate filtering impact estimates
    singleton_keywords = sum(1 for count in keyword_frequency.values() if count == 1)
    low_frequency_keywords = sum(1 for count in keyword_frequency.values() if count < 3)

    # Estimate potential filtering impact
    min_papers_threshold = max(
        1, int(total_papers * algorithm_config.keyword_min_papers_ratio)
    )
    potentially_filtered = sum(
        1 for count in keyword_frequency.values() if count < min_papers_threshold
    )

    filtering_impact = {
        "total_unique_keywords": unique_keywords,
        "singleton_keywords": singleton_keywords,
        "low_frequency_keywords": low_frequency_keywords,
        "potentially_filtered": potentially_filtered,
        "estimated_retention_rate": (
            1.0 - (potentially_filtered / unique_keywords)
            if unique_keywords > 0
            else 1.0
        ),
        "min_papers_threshold": min_papers_threshold,
        "singleton_percentage": (
            singleton_keywords / unique_keywords if unique_keywords > 0 else 0.0
        ),
        "filtering_aggressiveness": (
            "Conservative"
            if algorithm_config.keyword_min_papers_ratio <= 0.05
            else "Moderate"
        ),
    }

    return {
        "metrics": filtering_metrics,
        "impact": filtering_impact,
        "domain_stats": {
            "total_papers": total_papers,
            "years_covered": len(papers_by_year),
            "avg_papers_per_year": (
                total_papers / len(papers_by_year) if papers_by_year else 0
            ),
        },
    } 