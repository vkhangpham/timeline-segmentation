"""Utility functions for timeline segmentation."""

from .config import AlgorithmConfig, AntiGamingConfig

from .general import (
    discover_available_domains,
    ensure_results_directory,
    query_llm,
    query_llm_structured,
)

__all__ = [
    "AlgorithmConfig",
    "AntiGamingConfig",
    "discover_available_domains",
    "ensure_results_directory",
    "query_llm",
    "query_llm_structured",
]
