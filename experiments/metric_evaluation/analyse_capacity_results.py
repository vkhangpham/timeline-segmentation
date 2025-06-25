from __future__ import annotations

"""Analyse TF-IDF capacity sweep results.

Usage:
    python analyse_capacity_results.py <path_to_sweep_json>

Outputs the average final score per capacity and prints the best one.
"""

import json
import sys
from pathlib import Path
from statistics import mean
from typing import Dict, List


def load_results(path: Path) -> Dict[str, Dict[str, float]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data  # type: ignore


def compute_capacity_means(results: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    means: Dict[str, float] = {}
    for capacity, domains in results.items():
        scores: List[float] = [v for v in domains.values() if isinstance(v, (int, float))]
        if scores:
            means[capacity] = mean(scores)
    return dict(sorted(means.items(), key=lambda kv: (-kv[1], int(kv[0]))))


def main():
    if len(sys.argv) != 2:
        print("Usage: python analyse_capacity_results.py <sweep_results.json>")
        sys.exit(1)
    path = Path(sys.argv[1])
    results = load_results(path)
    means = compute_capacity_means(results)

    print("TF-IDF capacity sweep – mean final scores across domains:")
    for cap, avg in means.items():
        print(f"  {cap:>6}: {avg:.3f}")

    if means:
        best_cap, best_score = next(iter(means.items()))
        print(f"\n🏆 Best capacity: {best_cap} (avg score {best_score:.3f})")
    else:
        print("No numeric scores present – check the sweep results file.")


if __name__ == "__main__":
    main() 