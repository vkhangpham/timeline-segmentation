"""
Utility functions for the timeline analysis pipeline.
"""

import requests
from pathlib import Path
from typing import List, Optional, Dict, Any, Union
import json
from pydantic import BaseModel


def discover_available_domains() -> List[str]:
    """
    Automatically discover available domains from the resources directory.
    
    Returns:
        List of domain names found in resources directory
    """
    resources_path = Path("resources")
    
    if not resources_path.exists():
        print("❌ Resources directory not found")
        return []
    
    domains = []
    for item in resources_path.iterdir():
        # Only include directories and exclude system files
        if item.is_dir() and not item.name.startswith('.'):
            domains.append(item.name)
    
    return sorted(domains)


def ensure_results_directory():
    """Ensure the results directory exists."""
    Path("results").mkdir(exist_ok=True)


def calculate_adaptive_timeout(prompt: str, model: str = "qwen2.5:3b") -> int:
    """
    Calculate adaptive timeout based on prompt complexity and model requirements.
    
    Research-backed approach:
    - Base timeout scales with prompt length (more tokens = more processing time)
    - Model-specific scaling factors (larger models need more time)
    - Conservative buffer for network latency and generation time
    
    Args:
        prompt: The prompt text to analyze
        model: Model name for model-specific scaling
    
    Returns:
        Timeout in seconds (minimum 30, maximum 300)
    """
    # Base timeout calculation based on prompt complexity
    prompt_length = len(prompt)
    estimated_tokens = prompt_length // 4  # Rough estimate: 4 chars per token
    
    # Base processing time: 0.1 seconds per token (conservative estimate)
    base_timeout = max(30, estimated_tokens * 0.1)
    
    # Model-specific scaling factors (based on development journal Phase 5)
    model_factors = {
        "deepseek-r1": 6.0,        # Reasoning model needs extra time
        "deepseek-r1:8b-0528-qwen3-q4_K_M": 6.0,     # Reasoning model needs extra time
        "gemma3:12b": 2.5,         # Large model needs more time
        "qwen2.5:14b": 2.5,        # Large model needs more time  
        "llama3.1:8b": 2.0,        # Medium model
        "qwen2.5:7b": 1.8,         # Medium model
        "qwen2.5:3b": 1.5,         # Smaller model, faster
        "llama3.2:3b": 1.5,        # Smaller model, faster
    }
    
    # Get model-specific factor (default to 2.0 for unknown models)
    model_factor = 2.0
    for model_key, factor in model_factors.items():
        if model_key in model:
            model_factor = factor
            break
    
    # Calculate final timeout with model scaling
    adaptive_timeout = int(base_timeout * model_factor)
    
    # Apply bounds (30 seconds minimum, 300 seconds maximum)
    adaptive_timeout = max(30, min(300, adaptive_timeout))
    
    return adaptive_timeout


def query_llm(prompt: str, 
              model: str = "qwen2.5:3b", 
              format_schema: Optional[BaseModel] = None) -> str:
    """
    Query local LLM with adaptive timeout calculation and optional structured output.
    
    FAIL-FAST IMPLEMENTATION: Any failure immediately raises exception.
    No error masking - strict adherence to project guidelines Rule 6.
    
    Centralized LLM query function with intelligent timeout management
    based on prompt complexity and model requirements. Supports Pydantic
    structured outputs using Ollama's structured output feature.
    
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
    
    # Calculate adaptive timeout based on prompt complexity and model
    timeout = calculate_adaptive_timeout(prompt, model)
    
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.6,     # DeepSeek R1 recommended: 0.5-0.7 (0.6 optimal)
            "top_p": 0.95,          # DeepSeek R1 recommended: 0.95
            "max_tokens": 128000
        }
    }
    
    # Add structured output format if schema provided
    if format_schema is not None:
        payload["format"] = format_schema.model_json_schema()
    
    # FAIL-FAST: Let any network or API error immediately terminate execution
    response = requests.post(url, json=payload, timeout=timeout)
    response.raise_for_status()
    
    result = response.json()
    return result.get('response', '')


def query_llm_structured(prompt: str, 
                        format_schema: BaseModel, 
                        model: str = "deepseek-r1:8b-0528-qwen3-q4_K_M") -> BaseModel:
    """
    Query LLM with structured output and return parsed Pydantic model.
    
    FAIL-FAST IMPLEMENTATION: Any failure immediately raises exception.
    No error masking - strict adherence to project guidelines Rule 6.
    
    Enhanced version of query_llm specifically for structured outputs
    that automatically parses the JSON response into the provided Pydantic model.
    
    Args:
        prompt: The prompt text to send to the LLM
        format_schema: Pydantic model class for structured output
        model: Model name to use (defaults to reasoning model)
    
    Returns:
        Parsed Pydantic model instance
        
    Raises:
        Exception: If LLM query fails or JSON parsing fails
    """
    # FAIL-FAST: Get raw JSON response - any failure immediately terminates
    response_text = query_llm(prompt, model=model, format_schema=format_schema)
    
    # FAIL-FAST: Parse JSON and validate against schema - any failure immediately terminates
    response_json = json.loads(response_text)
    return format_schema(**response_json) 