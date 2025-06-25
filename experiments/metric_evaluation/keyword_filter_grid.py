#!/usr/bin/env python3
"""Keyword-Filtering Ratio Sweep (Phase-16 FEATURE-01)

This script automates:
1. Running Bayesian optimisation for each keyword-filter ratio value.
2. Collecting final consensus-difference scores per domain.
3. Writing an aggregated results file in results/.

Usage (examples):
    # Full sweep (default grid) over all domains, 4 workers optimisation
    python experiments/metric_evaluation/keyword_filter_grid.py --parallel 4

    # Custom ratios and limited domains
    python keyword_filter_grid.py --ratios 0.05 0.10 --domains applied_mathematics computer_vision

Fail-fast: any optimisation failure aborts the sweep.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List
import argparse

PROJECT_ROOT = Path(__file__).resolve().parents[2]  # timeline/
sys.path.append(str(PROJECT_ROOT))

from optimize_segmentation_bayesian import discover_available_domains  # noqa: E402


DEFAULT_RATIOS = [0.01, 0.05, 0.10, 0.15, 0.20]


def _run_optimisation_for_ratio(
    ratio: float,
    domains: List[str],
    max_evals: int,
    n_workers: int,
) -> Path:
    """Invoke optimise_segmentation_bayesian.py as a subprocess.

    Returns path to generated parameter file (json).
    """
    suffix = f"kwr{int(ratio * 100):02d}"
    cmd = [
        sys.executable,
        "optimize_segmentation_bayesian.py",
        "--keyword-ratio",
        f"{ratio}",
        "--max-evals",
        str(max_evals),
        "--suffix",
        suffix,
        "--parallel",
        str(n_workers),
    ]
    if domains:
        cmd.extend(domains)

    print("\n🚀 Running optimisation:", " ".join(cmd))
    proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    if proc.returncode != 0:
        raise RuntimeError(f"Optimisation failed for ratio {ratio} (exit {proc.returncode})")

    param_file = PROJECT_ROOT / f"results/optimized_parameters_bayesian_{suffix}.json"
    if not param_file.exists():
        raise FileNotFoundError(f"Expected parameter file not found: {param_file}")
    return param_file


def _extract_scores(param_path: Path) -> Dict[str, float]:
    """Return {domain: best_score} mapping from optimisation json."""
    with open(param_path, "r") as f:
        data = json.load(f)
    detailed = data.get("detailed_evaluations", {})
    return {d: v["score"] for d, v in detailed.items()}


def main():
    parser = argparse.ArgumentParser(description="Keyword-filter ratio sweep (Phase-16)")
    parser.add_argument("--ratios", nargs="*", type=float, default=None)
    parser.add_argument("--domains", nargs="*", default=None)
    parser.add_argument("--max-evals", type=int, default=200)
    parser.add_argument("--parallel", type=int, default=1)
    args = parser.parse_args()

    ratios = args.ratios if args.ratios else DEFAULT_RATIOS
    for r in ratios:
        if not 0.01 <= r <= 0.5:
            parser.error(f"ratio {r} out of bounds (0.01-0.5)")

    available = discover_available_domains()
    if not available:
        parser.error("No processed domains found")

    domains = args.domains if args.domains else available
    invalid = [d for d in domains if d not in available]
    if invalid:
        parser.error(f"Unknown domains: {', '.join(invalid)}")

    sweep_results: Dict[str, Dict[str, float]] = {}
    param_files: Dict[str, str] = {}

    for ratio in ratios:
        param_path = _run_optimisation_for_ratio(ratio, domains, args.max_evals, args.parallel)
        scores = _extract_scores(param_path)
        sweep_results[f"{ratio:.2f}"] = scores
        param_files[f"{ratio:.2f}"] = param_path.name

    # Aggregate statistics
    summary: Dict[str, Dict[str, float]] = {}
    for ratio, scores in sweep_results.items():
        if scores:
            mean_score = sum(scores.values()) / len(scores)
        else:
            mean_score = 0.0
        summary[ratio] = {
            "mean_score": mean_score,
            "domains": scores,
            "param_file": param_files[ratio],
        }

    # Save sweep results
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = PROJECT_ROOT / "results" / f"keyword_ratio_sweep_{ts}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"summary": summary, "generated": ts}, f, indent=2)

    print("\n💾 Sweep complete →", out_path)

    # Print quick summary
    print("\n📊 MEAN SCORES (higher better):")
    for ratio, info in sorted(summary.items(), key=lambda x: float(x[0])):
        print(f"  ratio {ratio}: {info['mean_score']:.3f}")


if __name__ == "__main__":
    main() 