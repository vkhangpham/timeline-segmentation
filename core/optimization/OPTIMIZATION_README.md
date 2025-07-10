# Timeline Segmentation Optimization Module

Bayesian optimization for timeline segmentation parameters.

## Features

### 1. Bayesian Parameter Optimization
- **Function**: `run_bayesian_optimization(config, objective_function, domain_name, verbose=False)`
- **Purpose**: Find optimal parameters using Gaussian Process-based optimization
- **Optimizer**: scikit-optimize with Expected Improvement acquisition function
- **Parallelization**: Supports multi-worker parallel trial evaluation

### 2. Trial Scoring System
- **Function**: `score_trial(domain_name, parameter_overrides, base_config, trial_id, optimization_config, data_directory="resources", verbose=False)`
- **Purpose**: Evaluate a single parameter combination and return comprehensive metrics
- **Features**: Data caching, error handling, validation metrics, penalty computation

### 3. Objective Function Evaluation
- **Function**: `compute_objective_function(academic_periods, algorithm_config, verbose=False)`
- **Purpose**: Calculate segmentation quality using cohesion and separation metrics
- **Metrics**: Jaccard similarity (cohesion), Jensen-Shannon divergence (separation)

### 4. Penalty Systems
Two penalty modes for controlling segmentation quality:

#### Linear Penalty
- **Purpose**: Penalize deviation from target segment count
- **Formula**: `penalty = penalty_weight × |num_segments - target_segments|`
- **Use Case**: Simple segment count control

#### Hybrid Penalty
- **Purpose**: Multi-component penalty for realistic segmentation constraints
- **Components**:
  - **Over-segmentation**: Penalty only when exceeding upper bound
  - **Short periods**: Penalty for periods shorter than minimum length
  - **Long periods**: Penalty for periods longer than maximum length
- **Formula**: `penalty = over_penalty + short_penalty + long_penalty`

## Usage

### Command Line Interface
```bash
# Optimize single domain
python scripts/run_optimization.py --domain art --verbose

# Optimize all domains
python scripts/run_optimization.py --domain all --verbose

# Optimize with custom configuration
python scripts/run_optimization.py --domain art --config custom_optimization.yaml
```

### Programmatic Usage
```python
from core.utils.config import AlgorithmConfig
from core.optimization.optimization_config import load_config
from core.optimization.bayesian_optimizer import run_bayesian_optimization
from core.optimization.optimization import score_trial

# Load configuration
optimization_config = load_config("config/optimization.yaml")
base_config = AlgorithmConfig.from_config_file()

# Create objective function
def objective_function(parameters, trial_id):
    return score_trial(
        domain_name="art",
        parameter_overrides=parameters,
        base_config=base_config,
        trial_id=trial_id,
        optimization_config=optimization_config,
    )

# Run optimization
result = run_bayesian_optimization(
    config=optimization_config,
    objective_function=objective_function,
    domain_name="art",
    verbose=True,
)

# Access results
best_params = result["best_result"]["parameters"]
best_score = result["best_result"]["objective_score"]
```

## Configuration

### Parameter Space Definition
```yaml
parameters:
  direction_change_threshold:
    type: float
    range: [0.6, 0.95]
    description: "Threshold for detecting significant direction changes"
  
  citation_confidence_boost:
    type: float
    range: [0.05, 0.2]
    description: "Boost factor for citation-based confidence"
  
  min_keyword_frequency_ratio:
    type: float
    range: [0.05, 0.15]
    description: "Minimum frequency ratio for keyword inclusion"
```

### Search Configuration
```yaml
search:
  n_calls: 100                    # Number of optimization trials
  n_initial_points: 20            # Random exploration before optimization
  strategy: "bayesian"            # Optimization strategy
  acquisition_function: "EI"      # Expected Improvement
```

### Penalty Configuration
```yaml
penalty:
  type: hybrid                    # [linear | hybrid]
  # Over-segmentation component
  target_segments_upper: 8        # No penalty if segments ≤ this
  penalty_weight_over: 0.05       # Cost per extra segment
  # Short-period component  
  min_period_years: 5             # Minimum acceptable period length
  short_period_weight: 0.02       # Cost per missing year
  # Long-period component
  max_period_years: 30            # Maximum acceptable period length
  long_period_weight: 0.02        # Cost per extra year
```

### Execution Configuration
```yaml
execution:
  max_workers: 8                  # Parallel workers for trial evaluation
  cache_academic_years: true      # Cache data loading between trials
```

## Data Structures

### Trial Result
Each trial returns a comprehensive result dictionary:
```python
{
    "trial_id": int,
    "parameters": Dict[str, Any],
    "objective_score": float,
    "cohesion_score": float,
    "separation_score": float,
    "num_segments": int,
    "boundary_f1": float,
    "segment_f1": float,
    "runtime_seconds": float,
    "error_message": str | None,
}
```

### Optimization Result
```python
{
    "best_result": Dict[str, Any],      # Best trial result
    "all_results": List[Dict[str, Any]], # All trial results
    "n_calls": int,                     # Number of trials completed
    "total_time": float,                # Total optimization time
    "progress": List[float],            # Best score progression
    "best_score_plot": str,             # Path to progress plot
}
```

### Objective Function Result
```python
ObjectiveFunctionResult(
    final_score: float,
    cohesion_score: float,
    separation_score: float,
    num_segments: int,
    num_transitions: int,
    cohesion_details: str,
    separation_details: str,
    methodology: str,
)
```

## Key Implementation Details

### 1. Efficient Trial Evaluation
- **Data Caching**: Academic years loaded once per domain and cached globally
- **Segmentation-Only Mode**: Skips characterization during optimization for speed
- **Logging Suppression**: Reduces console noise during non-verbose optimization
- **Error Handling**: Failed trials get penalty scores rather than crashing

### 2. Robust Parameter Handling
- **Dynamic Configuration**: Can optimize any AlgorithmConfig field
- **Type Safety**: Automatic type conversion (int, float, categorical)
- **Validation**: Parameter bounds and constraints enforced

### 3. Comprehensive Evaluation
- **Multi-Metric Scoring**: Combines intrinsic and extrinsic metrics
- **Validation Against Ground Truth**: Boundary-F1 and Segment-F1 when available
- **Penalty Integration**: Prevents pathological solutions

### 4. Performance Optimization
- **Parallel Processing**: Multi-worker trial evaluation
- **Progress Tracking**: Real-time optimization progress with tqdm
- **Memory Efficiency**: Careful caching and cleanup

## Output Files

### Trial Results
- **Location**: `results/optimization_logs/{domain}.csv`
- **Content**: All trial results with parameters and metrics
- **Format**: CSV with dynamic parameter columns

### Best Parameters
- **Location**: `results/optimized_params/{domain}.json`
- **Content**: Best parameter configuration for domain
- **Format**: JSON with parameter values and metadata

### Progress Plots
- **Location**: `results/optimization_logs/{domain}_best_score.png`
- **Content**: Best score progression over trials
- **Format**: PNG plot showing optimization convergence

## Optimization Strategies

### 1. Current Parameter Space
**Active Parameters** (3 currently optimized):
- `direction_change_threshold`: [0.6, 0.95] - Change detection sensitivity
- `citation_confidence_boost`: [0.05, 0.2] - Citation weight boost
- `min_keyword_frequency_ratio`: [0.05, 0.15] - Keyword filtering threshold

**Commented Parameters** (available but not active):
- `cohesion_weight`, `separation_weight`: Objective function weights
- `beam_width`: Beam search width for refinement
- `min_period_years`, `max_period_years`: Period length constraints
- `top_k_keywords`: Number of keywords for characterization

### 2. Penalty Strategy Evolution
**Legacy Linear Penalty**: Simple deviation from target count
```python
penalty = penalty_weight × |num_segments - target_segments|
```

**Current Hybrid Penalty**: Multi-component realistic constraints
```python
penalty = over_penalty + short_penalty + long_penalty
```

### 3. Acquisition Function Options
- **Expected Improvement (EI)**: Default, good exploration-exploitation balance
- **Probability of Improvement (PI)**: More conservative
- **Lower Confidence Bound (LCB)**: More exploitative

## Performance Characteristics

### Typical Optimization Run
- **Domains**: 6-8 academic domains
- **Trials per Domain**: 100 trials
- **Time per Trial**: 5-30 seconds (depends on domain size)
- **Total Time**: 10-50 minutes per domain
- **Memory Usage**: ~1-2GB (cached data)

### Convergence Patterns
- **Initial Exploration**: Random sampling (first 20 trials)
- **Exploitation**: Gaussian Process guided (remaining 80 trials)
- **Typical Convergence**: Best score stable after 50-70 trials

## Dependencies

### Core Dependencies
- `scikit-optimize`: Bayesian optimization framework
- `numpy`: Numerical computations
- `scipy`: Statistical functions (Jensen-Shannon divergence)
- `tqdm`: Progress tracking
- `matplotlib`: Progress plotting

### Internal Dependencies
- `core.data.data_models`: AcademicPeriod, AcademicYear
- `core.data.data_processing`: Data loading and filtering
- `core.pipeline.orchestrator`: Timeline analysis pipeline
- `core.evaluation.evaluation`: Objective function evaluation
- `core.utils.config`: Algorithm configuration management
- `core.utils.logging`: Logging utilities

## Module Structure

### Current Architecture (4 Files)
```
core/optimization/
├── optimization.py           (357 lines) - Trial scoring and caching
├── bayesian_optimizer.py     (163 lines) - Bayesian optimization engine
├── objective_function.py     (325 lines) - Objective function computation
├── optimization_config.py    (55 lines)  - Configuration loading
└── __init__.py              (32 lines)  - Public interface
```

### Design Principles
- **Functional Programming**: Pure functions where possible
- **Robust Error Handling**: Graceful failure with penalty scores
- **Efficient Caching**: Minimize redundant data loading
- **Comprehensive Logging**: Detailed progress and error reporting
- **Modular Design**: Clear separation of concerns

## Recent Evolution

### Parameter Space Reduction
The optimization system originally explored ~15 parameters but was reduced to 3 active parameters:
- **Reason**: Many parameters showed minimal impact on objective scores
- **Benefit**: Faster optimization convergence, reduced search space complexity
- **Trade-off**: Potentially missing parameter interactions

### Penalty System Upgrade
Evolved from simple linear penalty to sophisticated hybrid penalty:
- **Linear**: `penalty = weight × |segments - target|`
- **Hybrid**: Multi-component with over-segmentation and period length constraints
- **Improvement**: More realistic segmentation quality control

### Performance Optimizations
- **Parallel Processing**: Multi-worker trial evaluation
- **Data Caching**: Global academic year caching
- **Logging Control**: Suppress algorithm logs during optimization
- **Progress Tracking**: Real-time optimization monitoring

## Future Enhancements

### Potential Improvements
1. **Cross-Validation**: Split domains into train/validation sets
2. **Multi-Objective Optimization**: Optimize multiple metrics simultaneously
3. **Hyperparameter Scheduling**: Dynamic parameter bounds during optimization
4. **Ensemble Methods**: Combine multiple optimization runs
5. **Transfer Learning**: Use optimization results across similar domains

### Parameter Space Extensions
1. **Reactivate Core Parameters**: `cohesion_weight`, `separation_weight`
2. **Add New Parameters**: Neural network hyperparameters, attention weights
3. **Conditional Parameters**: Parameter dependencies and constraints
4. **Domain-Specific Parameters**: Different parameter sets per domain 