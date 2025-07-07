"""
Parameter Loader for Timeline Analysis

This module handles loading optimized parameters for domains.
Simplified to only include functions that are actually used.
"""

import json
import os
from .logging import get_logger


def load_optimized_parameters(
    domain_name: str, params_file: str = None, verbose: bool = False
) -> dict:
    """
    Load optimized parameters for a specific domain if available.

    Args:
        domain_name: Name of the research domain
        params_file: Optional path to parameters file (defaults to standard location)
        verbose: Enable verbose logging

    Returns:
        Dictionary containing optimized parameters for the domain,
        or empty dict if no parameters are found
    """
    logger = get_logger(__name__, verbose)
    file_path = params_file or "results/optimization/optimized_parameters_bayesian.json"

    if not os.path.exists(file_path):
        logger.info("No optimized parameters found, using defaults")
        return {}

    # FAIL-FAST: Load parameters or fail immediately with clear error message
    with open(file_path, "r") as f:
        data = json.load(f)

    # Check for consensus_difference_optimized_parameters (legacy format)
    params = data.get("consensus_difference_optimized_parameters", {})
    if domain_name in params:
        logger.info(f"Using optimized parameters for {domain_name}")
        return params[domain_name]

    # Check for direct domain parameters
    if domain_name in data:
        logger.info(f"Using optimized parameters for {domain_name}")
        return data[domain_name]

    logger.info(f"No optimized parameters for {domain_name}, using defaults")
    return {}


# Export only the used function
__all__ = ["load_optimized_parameters"]
