from __future__ import annotations

"""TF-IDF capacity sweep (Phase-16 FEATURE-02).

Iterates over a predefined set of `max_features` values, launches Bayesian
optimisation with the capacity override, and stores results in a timestamped
JSON file inside `experiments/metric_evaluation/results/`.

Only the experiment driver is provided here – it relies on existing
`optimize_segmentation_bayesian.py` CLI to perform optimisation.
"""

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from subprocess import run, CalledProcessError
from typing import List

CAPACITY_GRID: List[int] = [500, 1000, 2000, 5000, 7500, 10000]
DEFAULT_DOMAINS: List[str] = [
    "applied_mathematics",
    "art",
    "computer_science",
    "computer_vision",
    "deep_learning",
    "machine_learning",
    "machine_translation",
    "natural_language_processing",
]

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def main() -> None:
    parser = argparse.ArgumentParser(description="TF-IDF capacity sweep experiment")
    parser.add_argument("--domains", nargs="*", default=DEFAULT_DOMAINS, help="Domains to evaluate")
    parser.add_argument("--parallel", type=int, default=1, dest="n_workers", help="Number of parallel workers for each optimisation run (forwarded to optimise script)")
    parser.add_argument("--silent", action="store_true", help="Suppress optimiser stdout (forwarded to optimise script)")
    args = parser.parse_args()

    domains: List[str] = args.domains
    experiment_results = {}

    for capacity in CAPACITY_GRID:
        print(f"\n>>> Running capacity {capacity}…")
        capacity_key = str(capacity)
        experiment_results[capacity_key] = {}
        for domain in domains:
            try:
                cmd = [
                    "python", "optimize_segmentation_bayesian.py",
                    domain,
                    "--tfidf_max_features", str(capacity),
                    "--parallel", str(args.n_workers),
                    "--keyword-ratio", "0.05",
                    "--no-save"
                ]
                if args.silent:
                    cmd.append("--silent")
                print(" ", " ".join(cmd))

                best_file = RESULTS_DIR / f"best_{domain}_{capacity}.json"
                cmd.extend(["--best-out", str(best_file)])

                run(cmd, check=True)

                if best_file.exists():
                    with open(best_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    experiment_results[capacity_key][domain] = data["final_score"]
                else:
                    experiment_results[capacity_key][domain] = None
            except CalledProcessError as e:
                experiment_results[capacity_key][domain] = f"failure: {e}"
                raise  # fail-fast as per guideline

    out_file = RESULTS_DIR / f"tfidf_capacity_grid_{_timestamp()}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(experiment_results, f, indent=2)
    print(f"\nSaved sweep log → {out_file}")


if __name__ == "__main__":
    main() 