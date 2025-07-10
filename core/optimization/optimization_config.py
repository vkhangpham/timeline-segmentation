"""Simple optimization configuration loader.

Loads YAML configuration and provides parameter space for Bayesian optimization.
"""

import yaml
from pathlib import Path
from typing import Dict, Any


def load_config(config_path: str = None, domain_name: str = None) -> Dict[str, Any]:
    """Load optimization configuration from YAML file.

    Args:
        config_path: Path to YAML config file
        domain_name: Domain name (unused, for compatibility)

    Returns:
        Configuration dictionary
    """
    if config_path is None:
        config_path = Path("config/optimization.yaml")
    else:
        config_path = Path(config_path)

    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_parameter_space(config: Dict[str, Any]) -> Dict[str, Any]:
    """Extract parameter space for scikit-optimize.

    Args:
        config: Configuration dictionary

    Returns:
        Dictionary mapping parameter names to skopt dimensions
    """
    from skopt.space import Real, Integer, Categorical

    space = {}
    parameters = config.get("parameters", {})

    for param_name, param_spec in parameters.items():
        param_type = param_spec.get("type", "float")
        param_range = param_spec.get("range")

        if param_range:
            low, high = param_range
            if param_type == "int":
                space[param_name] = Integer(low, high, name=param_name)
            else:
                space[param_name] = Real(low, high, name=param_name)

    return space
