#!/usr/bin/env python3
"""Global Consensus / Difference Weight Sweep (Phase 16)

This experiment exhaustively sweeps the final aggregation weights
(consensus_weight, difference_weight) ∈ {0.9,0.8,…,0.1} while also
comparing two aggregation methods:
    • linear   – arithmetic weighted mean  (current default)
    • harmonic – weighted harmonic mean   (FEATURE-05)

The script relies on the *already-optimised* segmentation parameters
stored in `results/optimized_parameters_bayesian.json`.  It therefore
isolates the impact of the **final aggregation strategy** from
segmentation-level influences.

Fail-fast behaviour:  any missing data, optimisation parameter file, or
inconsistent configuration immediately raises an exception.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Ensure project root is on PYTHONPATH so that `import core.*` works when this
# script is executed from its nested directory.
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]  # timeline/
sys.path.append(str(PROJECT_ROOT))

# Project imports (root already in PYTHONPATH when running from repo root)
from core.data_loader import load_domain_data
from core.data_models import DomainData, Paper
from core.algorithm_config import AlgorithmConfig
from core.integration import run_change_detection
from core.consensus_difference_metrics import evaluate_segmentation_quality
from optimize_segmentation_bayesian import convert_dataframe_to_domain_data, SuppressOutput

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

OPTIMISED_PARAM_FILE = "results/optimized_parameters_bayesian.json"


def _load_optimised_parameters() -> Dict[str, Dict[str, float]]:
    """Load per-domain optimised segmentation parameters.

    Returns:
        Mapping from domain → parameter dict.
    """
    if not os.path.exists(OPTIMISED_PARAM_FILE):
        raise FileNotFoundError(
            f"Optimised parameter file not found: {OPTIMISED_PARAM_FILE}. Run optimise_segmentation_bayesian.py first."  # noqa: E501
        )

    with open(OPTIMISED_PARAM_FILE, "r") as f:
        data = json.load(f)

    if "consensus_difference_optimized_parameters" not in data:
        raise KeyError("Invalid parameter file structure – expected 'consensus_difference_optimized_parameters'")

    return data["consensus_difference_optimized_parameters"]


def _config_from_params(params: Dict[str, float]) -> AlgorithmConfig:
    """Convert parameter dict to AlgorithmConfig (pure)."""
    return AlgorithmConfig(
        direction_threshold=float(params["direction_threshold"]),
        validation_threshold=float(params["validation_threshold"]),
        similarity_min_segment_length=int(params["similarity_min_segment_length"]),
        similarity_max_segment_length=int(params["similarity_max_segment_length"]),
    )


# ---------------------------------------------------------------------------
# Core evaluation routine (pure)
# ---------------------------------------------------------------------------

def evaluate_domain_weights(
    domain: str,
    consensus_weight: float,
    difference_weight: float,
    aggregation_method: str,
    domain_params: Dict[str, Dict[str, float]],
) -> Dict[str, float]:
    """Evaluate a single (weight_pair, aggregation_method) on one domain."""
    # Load dataset
    df = load_domain_data(domain)
    if df is None or df.empty:
        raise ValueError(f"No data available for domain '{domain}'")

    domain_data: DomainData = convert_dataframe_to_domain_data(df, domain)

    # Build algorithm config using pre-optimised parameters
    if domain not in domain_params:
        raise KeyError(f"Optimised parameters for domain '{domain}' not found in {OPTIMISED_PARAM_FILE}")

    config = _config_from_params(domain_params[domain])

    # Run segmentation algorithm to obtain year-based segments
    with SuppressOutput(suppress_stdout=True):
        segmentation_results, _ = run_change_detection(
            domain,
            granularity=config.granularity,
            algorithm_config=config,
        )

    # Convert year spans to lists of papers
    segment_papers: List[Tuple[Paper, ...]] = []
    if segmentation_results and "segments" in segmentation_results:
        for (start_year, end_year) in segmentation_results["segments"]:
            papers = [p for p in domain_data.papers if start_year <= p.pub_year <= end_year]
            if papers:
                segment_papers.append(tuple(papers))
    if not segment_papers:
        segment_papers = [tuple(domain_data.papers)]

    # Evaluate quality with chosen aggregation strategy
    eval_result = evaluate_segmentation_quality(
        segment_papers,
        final_combination_weights=(consensus_weight, difference_weight),
        aggregation_method=aggregation_method,
    )

    return {
        "final_score": eval_result.final_score,
        "consensus_score": eval_result.consensus_score,
        "difference_score": eval_result.difference_score,
        "num_segments": eval_result.num_segments,
    }


# ---------------------------------------------------------------------------
# Sweep driver
# ---------------------------------------------------------------------------

def run_weight_sweep(domains: List[str]):
    params_by_domain = _load_optimised_parameters()

    weight_values = [round(x, 1) for x in np.arange(0.9, 0.0, -0.1)]  # 0.9 … 0.1
    aggregation_methods = ["linear", "harmonic"]

    results: Dict[str, Dict[str, Dict[str, float]]] = {}

    for domain in domains:
        domain_results: Dict[str, Dict[str, float]] = {}
        for cw in weight_values:
            dw = round(1.0 - cw, 1)
            for method in aggregation_methods:
                key = f"cw{cw}_dw{dw}_{method}"
                domain_results[key] = evaluate_domain_weights(
                    domain,
                    consensus_weight=cw,
                    difference_weight=dw,
                    aggregation_method=method,
                    domain_params=params_by_domain,
                )
        results[domain] = domain_results

    return results


# ---------------------------------------------------------------------------
# Entry-point
# ---------------------------------------------------------------------------

def main():
    if len(sys.argv) > 1:
        domains_to_run = sys.argv[1:]
    else:
        params_by_domain = _load_optimised_parameters()
        domains_to_run = sorted(params_by_domain.keys())

    print("🔬 GLOBAL CONSENSUS/DIFFERENCE WEIGHT SWEEP")
    print("Domains:", ", ".join(domains_to_run))

    sweep_results = run_weight_sweep(domains_to_run)

    # Persist results
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_path = Path("results") / f"global_weight_sweep_{timestamp}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"results": sweep_results, "generated": timestamp}, f, indent=2)

    print(f"\n💾 Sweep results saved to {out_path}")


if __name__ == "__main__":
    main() 