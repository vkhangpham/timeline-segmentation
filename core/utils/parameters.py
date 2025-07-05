"""
Parameter Loader for Timeline Analysis

This module handles loading optimized parameters and configuration settings.
Separated from integration.py to follow single responsibility principle.

Key Features:
- Load optimized parameters from configuration files
- Handle parameter file discovery and validation
- Provide fallback to default parameters
- Support multiple parameter sources

Follows functional programming principles with pure functions for parameter loading.
"""

import json
import os
from typing import Dict, Optional
from .logging import get_logger


def load_optimized_parameters(domain_name: str, params_file: str = None, verbose: bool = False) -> Dict:
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
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # Check for consensus_difference_optimized_parameters (legacy format)
    params = data.get('consensus_difference_optimized_parameters', {})
    if domain_name in params:
        logger.info(f"Using optimized parameters for {domain_name}")
        return params[domain_name]
    
    # Check for direct domain parameters
    if domain_name in data:
        logger.info(f"Using optimized parameters for {domain_name}")
        return data[domain_name]
    
    logger.info(f"No optimized parameters for {domain_name}, using defaults")
    return {}


def load_configuration_parameters(config_file: str = "optimization_config.json", verbose: bool = False) -> Dict:
    """
    Load general configuration parameters from a configuration file.
    
    Args:
        config_file: Path to the configuration file
        verbose: Enable verbose logging
        
    Returns:
        Dictionary containing configuration parameters, or empty dict if loading fails
    """
    logger = get_logger(__name__, verbose)
    
    if not os.path.exists(config_file):
        logger.info(f"Configuration file {config_file} not found")
        return {}
    
    # FAIL-FAST: Load configuration or fail immediately with clear error message  
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    logger.info(f"Loaded configuration from {config_file}")
    return config


def get_parameter_value(parameters: Dict, key: str, default_value=None):
    """
    Safely get a parameter value from a parameters dictionary.
    
    Args:
        parameters: Dictionary containing parameters
        key: Parameter key to retrieve
        default_value: Default value if key is not found
        
    Returns:
        Parameter value or default value
    """
    return parameters.get(key, default_value)


def validate_parameters(parameters: Dict, required_keys: list, verbose: bool = False) -> bool:
    """
    Validate that a parameters dictionary contains all required keys.
    
    Args:
        parameters: Dictionary containing parameters
        required_keys: List of required parameter keys
        verbose: Enable verbose logging
        
    Returns:
        True if all required keys are present, False otherwise
    """
    logger = get_logger(__name__, verbose)
    missing_keys = [key for key in required_keys if key not in parameters]
    
    if missing_keys:
        logger.warning(f"Missing required parameters: {missing_keys}")
        return False
    
    return True


def merge_parameters(*parameter_dicts: Dict) -> Dict:
    """
    Merge multiple parameter dictionaries, with later dictionaries taking precedence.
    
    Args:
        *parameter_dicts: Variable number of parameter dictionaries
        
    Returns:
        Merged parameter dictionary
    """
    merged = {}
    
    for params in parameter_dicts:
        if params:
            merged.update(params)
    
    return merged


def discover_parameter_files(directory: str = "results/optimization") -> list:
    """
    Discover available parameter files in a directory.
    
    Args:
        directory: Directory to search for parameter files
        
    Returns:
        List of found parameter file paths
    """
    if not os.path.exists(directory):
        return []
    
    parameter_files = []
    
    for filename in os.listdir(directory):
        if filename.endswith('.json') and 'param' in filename.lower():
            parameter_files.append(os.path.join(directory, filename))
    
    return sorted(parameter_files)


# Export functions
__all__ = [
    'load_optimized_parameters',
    'load_configuration_parameters',
    'get_parameter_value',
    'validate_parameters',
    'merge_parameters',
    'discover_parameter_files'
] 