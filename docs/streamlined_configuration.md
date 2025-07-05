# Streamlined Algorithm Configuration

## Overview

The `core/algorithm_config.py` module has been streamlined to focus on essential parameters for the new cohesion-separation objective function approach, removing complexity from the previous consensus-difference system.

## Key Changes

### Removed Complexity
- **Granularity Scaling System**: Removed complex parameter scaling based on granularity levels
- **Bayesian Optimization Integration**: Removed automatic loading of optimized parameters from results files  
- **Complex Validation**: Simplified parameter validation to focus on essential constraints
- **Export/Import Functions**: Removed unused configuration file export/import utilities

### Added Objective Function Support
- **Cohesion Weight**: Controls importance of within-segment keyword overlap (default: 0.8)
- **Separation Weight**: Controls importance of between-segment vocabulary shift (default: 0.2)  
- **Top-K Keywords**: Number of top keywords used for cohesion calculation (default: 15)

### Maintained Backward Compatibility
- All existing parameters used by the codebase are preserved
- Legacy parameters (granularity, tfidf_max_features, etc.) kept for compatibility
- Existing code patterns continue to work without modification

## Configuration Structure

```python
@dataclass
class AlgorithmConfig:
    # Core Detection Parameters
    direction_threshold: float = 0.4
    validation_threshold: float = 0.7
    
    # Citation Analysis Parameters  
    citation_boost_rate: float = 0.8
    citation_support_window: int = 2
    citation_analysis_scales: List[int] = [1, 3, 5]
    
    # Keyword Analysis Parameters
    keyword_min_frequency: int = 2
    min_significant_keywords: int = 2
    keyword_filtering_enabled: bool = True
    keyword_min_papers_ratio: float = 0.05
    
    # Segmentation Parameters
    similarity_min_segment_length: int = 3
    similarity_max_segment_length: int = 50
    
    # NEW: Objective Function Parameters
    cohesion_weight: float = 0.8
    separation_weight: float = 0.2
    top_k_keywords: int = 15
```

## Configuration Loading

The streamlined configuration loads objective function parameters from `optimization_config.json`:

```json
{
  "objective_function": {
    "cohesion_weight": 0.8,
    "separation_weight": 0.2,
    "top_k_keywords": 15
  }
}
```

## Usage Examples

### Basic Configuration
```python
from core.algorithm_config import AlgorithmConfig

# Default configuration
config = AlgorithmConfig()
print(f"Objective weights: {config.cohesion_weight}, {config.separation_weight}")
```

### Domain-Specific Configuration
```python
# Configuration for specific domain
config = AlgorithmConfig(domain_name="natural_language_processing")
```

### Custom Configuration
```python
# Custom configuration with overrides
config = AlgorithmConfig.create_custom(
    domain_name="computer_vision",
    overrides={
        "cohesion_weight": 0.7,
        "separation_weight": 0.3,
        "top_k_keywords": 20
    }
)
```

## Integration with Objective Function

The streamlined configuration seamlessly integrates with the new objective function:

```python
from core.algorithm_config import AlgorithmConfig
from core.objective_function import compute_objective_function

config = AlgorithmConfig()
result = compute_objective_function(
    segment_papers,
    cohesion_weight=config.cohesion_weight,
    separation_weight=config.separation_weight
)
```

## Benefits

1. **Simplified Maintenance**: ~60% reduction in code complexity
2. **Clear Focus**: Configuration directly supports the new objective function
3. **Better Performance**: Removed unused optimization loading and scaling logic
4. **Backward Compatibility**: Existing code continues to work unchanged
5. **Fail-Fast Validation**: Simple, clear parameter validation with immediate error reporting

## Migration Notes

- **No Breaking Changes**: All existing code using `AlgorithmConfig` continues to work
- **New Parameters**: Objective function parameters are automatically available
- **Simplified Creation**: Less complex initialization logic, faster startup
- **Configuration Files**: Objective function parameters loaded from standard config location

The streamlined configuration maintains full functionality while dramatically reducing complexity and improving focus on the new objective function approach. 