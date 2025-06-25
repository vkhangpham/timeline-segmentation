#!/usr/bin/env python3
"""Aggregation Formula Analysis (Phase 16)

Reads a global_weight_sweep_*.json file and produces aggregate statistics
comparing linear vs harmonic aggregation.

Metrics reported per domain and overall:
• best score per method & corresponding (cw,dw)
• mean ± std score across weight grid (stability)
• coefficient-of-variation (std/mean)
• method advantage = best_linear – best_harmonic

Designed to be **read-only** – does not touch algorithm code nor rerun heavy
experiments; it purely crunches previously saved JSON.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, Tuple
import numpy as np


def load_results(path: Path) -> Dict[str, Dict[str, Dict[str, float]]]:
    with path.open() as f:
        data = json.load(f)
    return data["results"]


def summarise_domain(domain_data: Dict[str, Dict[str, float]]):
    linear_scores = {k: v["final_score"] for k, v in domain_data.items() if k.endswith("_linear")}
    harm_scores = {k: v["final_score"] for k, v in domain_data.items() if k.endswith("_harmonic")}

    best_lin_key, best_lin = max(linear_scores.items(), key=lambda kv: kv[1])
    best_harm_key, best_harm = max(harm_scores.items(), key=lambda kv: kv[1])

    lin_vals = np.fromiter(linear_scores.values(), float)
    harm_vals = np.fromiter(harm_scores.values(), float)

    summary = {
        "best_linear": (best_lin, best_lin_key),
        "best_harmonic": (best_harm, best_harm_key),
        "mean_linear": float(lin_vals.mean()),
        "std_linear": float(lin_vals.std()),
        "cv_linear": float(lin_vals.std() / lin_vals.mean()) if lin_vals.mean() else 0.0,
        "mean_harmonic": float(harm_vals.mean()),
        "std_harmonic": float(harm_vals.std()),
        "cv_harmonic": float(harm_vals.std() / harm_vals.mean()) if harm_vals.mean() else 0.0,
        "advantage": best_lin - best_harm,
    }
    return summary


def main():
    if len(sys.argv) < 2:
        print("Usage: aggregation_analysis.py <path_to_sweep_json>")
        sys.exit(1)

    path = Path(sys.argv[1])
    if not path.exists():
        print(f"File not found: {path}")
        sys.exit(1)

    results = load_results(path)

    per_domain = {domain: summarise_domain(data) for domain, data in results.items()}

    # Aggregate across domains
    linear_adv = np.mean([s["advantage"] for s in per_domain.values()])
    mean_cv_lin = np.mean([s["cv_linear"] for s in per_domain.values()])
    mean_cv_harm = np.mean([s["cv_harmonic"] for s in per_domain.values()])

    print("\nAGGREGATION FORMULA ANALYSIS")
    print("=" * 60)
    for domain, summary in per_domain.items():
        print(f"{domain:<28} linear_best={summary['best_linear'][0]:.3f}  harm_best={summary['best_harmonic'][0]:.3f}  advantage={summary['advantage']:+.3f}")
    print("-" * 60)
    print(f"Average linear – harmonic advantage: {linear_adv:+.3f}")
    print(f"Average CV (stability) linear: {mean_cv_lin:.3f}  harmonic: {mean_cv_harm:.3f}")

    if linear_adv > 0:
        print("\n💡 Linear aggregation yields higher best score on average.")
    else:
        print("\n💡 Harmonic aggregation yields higher best score on average.")

    if mean_cv_harm < mean_cv_lin:
        print("⚖️  Harmonic aggregation is more stable across weight choices.")


if __name__ == "__main__":
    main() 