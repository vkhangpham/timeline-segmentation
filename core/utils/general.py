"""Utility functions for the timeline analysis pipeline."""

import requests
from pathlib import Path
from typing import List, Optional
import json
from pydantic import BaseModel
from .logging import get_logger


def discover_available_domains(verbose: bool = False) -> List[str]:
    """Discover available domains from the resources directory.

    Args:
        verbose: Enable verbose logging

    Returns:
        List of domain names found in resources directory
    """
    logger = get_logger(__name__, verbose)
    resources_path = Path("resources")

    if not resources_path.exists():
        logger.warning("Resources directory not found")
        return []

    domains = []
    for item in resources_path.iterdir():
        if item.is_dir() and not item.name.startswith("."):
            domains.append(item.name)

    return sorted(domains)


def discover_available_timeline_domains(verbose: bool = False) -> List[str]:
    """Discover available domains from timeline files in results/timelines directory.

    Args:
        verbose: Enable verbose logging

    Returns:
        List of domain names found in timeline files
    """
    logger = get_logger(__name__, verbose)
    timelines_path = Path("results/timelines")

    if not timelines_path.exists():
        logger.warning("Timeline results directory not found: results/timelines")
        return []

    domains = []
    for timeline_file in timelines_path.glob("*.json"):
        # Extract domain name from filename
        # Expected format: {domain_name}_timeline_analysis.json
        filename = timeline_file.name
        if filename.endswith("_timeline_analysis.json"):
            domain_name = filename.replace("_timeline_analysis.json", "")
            domains.append(domain_name)
        else:
            logger.warning(f"Unexpected timeline file format: {filename}")

    if verbose:
        logger.info(f"Found {len(domains)} domains with timeline files: {', '.join(domains)}")

    return sorted(domains)


def get_timeline_file_path(domain_name: str, verbose: bool = False) -> Optional[str]:
    """Get the expected timeline file path for a domain.

    Args:
        domain_name: Domain name
        verbose: Enable verbose logging

    Returns:
        Path to timeline file if it exists, None otherwise
    """
    logger = get_logger(__name__, verbose)
    timeline_file = Path("results/timelines") / f"{domain_name}_timeline_analysis.json"
    
    if timeline_file.exists():
        return str(timeline_file)
    else:
        if verbose:
            logger.warning(f"Timeline file not found: {timeline_file}")
        return None


def ensure_results_directory():
    """Ensure the results directory exists."""
    Path("results").mkdir(exist_ok=True)


def _calculate_adaptive_timeout(prompt: str, model: str = "qwen2.5:3b") -> int:
    """Calculate adaptive timeout based on prompt complexity and model requirements.

    Args:
        prompt: Input prompt text
        model: Model name for timeout calculation

    Returns:
        Calculated timeout in seconds
    """
    prompt_length = len(prompt)
    estimated_tokens = prompt_length // 4

    base_timeout = max(30, estimated_tokens * 0.1)

    model_factors = {
        "deepseek-r1": 6.0,
        "deepseek-r1:8b-0528-qwen3-q4_K_M": 6.0,
        "gemma3n:latest": 2.5,
        "qwen2.5:14b": 2.5,
        "llama3.1:8b": 2.0,
        "qwen2.5:7b": 1.8,
        "qwen2.5:3b": 1.5,
        "llama3.2:3b": 1.5,
    }

    model_factor = 4.0
    for model_key, factor in model_factors.items():
        if model_key in model:
            model_factor = factor
            break

    adaptive_timeout = int(base_timeout * model_factor)
    return max(30, min(300, adaptive_timeout))


def query_llm(
    prompt: str, model: str = "qwen2.5:3b", format_schema: Optional[BaseModel] = None
) -> str:
    """Query local LLM with adaptive timeout calculation.

    Args:
        prompt: The prompt text to send to the LLM
        model: Model name to use for the query
        format_schema: Optional Pydantic model for structured output

    Returns:
        LLM response text (raw string or JSON if using structured output)

    Raises:
        Exception: If LLM query fails
    """
    url = "http://localhost:11434/api/generate"

    timeout = _calculate_adaptive_timeout(prompt, model)

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.1, "top_p": 0.95},
    }

    if format_schema is not None:
        payload["format"] = format_schema.model_json_schema()

    response = requests.post(url, json=payload, timeout=timeout)
    response.raise_for_status()

    result = response.json()
    return result.get("response", "")


def query_llm_structured(
    prompt: str,
    format_schema: BaseModel,
    model: str = "deepseek-r1:8b-0528-qwen3-q4_K_M",
) -> BaseModel:
    """Query LLM with structured output and return parsed Pydantic model.

    Args:
        prompt: The prompt text to send to the LLM
        format_schema: Pydantic model class for structured output
        model: Model name to use (defaults to reasoning model)

    Returns:
        Parsed Pydantic model instance

    Raises:
        Exception: If LLM query fails or JSON parsing fails
    """
    response_text = query_llm(prompt, model=model, format_schema=format_schema)
    response_json = json.loads(response_text)
    return format_schema(**response_json)
