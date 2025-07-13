# Timeline Segmentation Optimization Module

Bayesian optimization for timeline segmentation parameters with unified configuration management.

## Features

### 1. Unified Configuration System
- **Single Source of Truth**: Penalty and objective function parameters loaded from `config.yaml`
- **Automatic Synchronization**: Optimization uses identical parameters as main algorithm
- **Clean Separation**: `optimization.yaml` contains only optimization-specific settings
- **No Duplication**: Eliminates parameter inconsistencies between configs

### 2. Bayesian Parameter Optimization
- **Function**: `run_bayesian_optimization(config, objective_function, domain_name, verbose=False)`
- **Purpose**: Find optimal parameters using Gaussian Process-based optimization
- **Optimizer**: scikit-optimize with Expected Improvement acquisition function
- **Parallelization**: Supports multi-worker parallel trial evaluation

### 3. Trial Scoring System
- **Function**: `score_trial(domain_name, parameter_overrides, base_config, trial_id, optimization_config, data_directory="resources", verbose=False)`
- **Purpose**: Evaluate a single parameter combination and return comprehensive metrics
- **Features**: Data caching, error handling, validation metrics, penalty computation
- **Configuration Loading**: Loads penalty parameters from main `config.yaml`

### 4. Objective Function Evaluation
- **Function**: `compute_objective_function(academic_periods, algorithm_config, verbose=False)`
- **Purpose**: Calculate segmentation quality using cohesion and separation metrics
- **Metrics**: Jaccard similarity (cohesion), Jensen-Shannon divergence (separation)

### 5. Penalty Systems
Penalty parameters are now loaded from main `config.yaml` ensuring consistency with algorithm execution:

#### Hybrid Penalty (Current Implementation)
- **Purpose**: Multi-component penalty for realistic segmentation constraints
- **Components**:
  - **Short periods**: Penalty for periods shorter than minimum length
  - **Long periods**: Penalty for periods longer than maximum length
  - **Over-segmentation**: Penalty when exceeding target segment count
- **Configuration**: All penalty parameters in `config.yaml` under `optimization.penalty`
- **Scaling**: Optional penalty scaling for different objective score ranges

### 6. Comprehensive Logging
- **Configuration Display**: Full optimization config logged before trials start
- **Progress Tracking**: Real-time optimization progress with tqdm
- **Parameter Details**: Shows ranges, types, and descriptions for all parameters
- **Best Score Tracking**: Continuous monitoring of optimization convergence

## Usage

### Command Line Interface
```bash
# Optimize single domain with verbose logging
python scripts/run_optimization.py --domain deep_learning --verbose

# Optimize all domains
python scripts/run_optimization.py --domain all --verbose

# Optimize with custom configuration
python scripts/run_optimization.py --domain art --config custom_optimization.yaml

# Quick optimization (non-verbose, shows tqdm only)
python scripts/run_optimization.py --domain computer_vision
```

### Programmatic Usage
```python
from core.utils.config import AlgorithmConfig
from core.optimization.optimization_config import load_config
from core.optimization.bayesian_optimizer import run_bayesian_optimization
from core.optimization.optimization import score_trial

# Load configurations
optimization_config = load_config("config/optimization.yaml")
base_config = AlgorithmConfig.from_config_file()  # Loads from config.yaml

# Create objective function
def objective_function(parameters, trial_id):
    return score_trial(
        domain_name="deep_learning",
        parameter_overrides=parameters,
        base_config=base_config,
        trial_id=trial_id,
        optimization_config=optimization_config,
    )

# Run optimization
result = run_bayesian_optimization(
    config=optimization_config,
    objective_function=objective_function,
    domain_name="deep_learning",
    verbose=True,
)

# Access results
best_params = result["best_result"]["parameters"]
best_score = result["best_result"]["objective_score"]
```

## Configuration Architecture

### Unified Configuration System
The optimization system uses a two-file configuration approach:

#### 1. Main Algorithm Config (`config/config.yaml`)
Contains all algorithm parameters including penalty and objective function settings:
```yaml
optimization:
  # Objective Function Parameters (shared with main algorithm)
  objective_function:
    cohesion_weight: 0.8
    separation_weight: 0.2

# Data Processing Parameters (shared with main algorithm)
data_processing:
  keyword_filter:
    top_k_keywords: 100
    min_keyword_frequency_ratio: 0.05

  # Penalty System Parameters (shared with main algorithm)
  penalty:
    min_period_years: 3
    max_period_years: 14
    auto_n_upper: true
    n_upper_buffer: 1
    target_segments_upper: 8
    lambda_short: 0.05
    lambda_long: 0.03
    lambda_count: 0.02
    enable_scaling: true
    scaling_factor: 2.0
```

#### 2. Optimization-Specific Config (`config/optimization.yaml`)
Contains only optimization-specific settings:
```yaml
# Parameter search space (what gets optimized)
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

  top_k_keywords:
    type: int
    range: [10, 30]
    description: "Number of top keywords to consider for period cohesion"

# Search strategy
search:
  n_calls: 100
  n_initial_points: 20
  strategy: "bayesian"
  acquisition_function: "EI"

# Optimization execution settings
execution:
  max_workers: 8
  cache_academic_years: true
```

### Configuration Benefits
- **Single Source of Truth**: Penalty/objective parameters defined once in `config.yaml`
- **Perfect Synchronization**: Optimization uses identical parameters as main algorithm
- **No Duplication**: Eliminates parameter inconsistencies
- **Clear Separation**: Optimization-specific vs. algorithm-shared parameters

## Current Parameter Space

### Active Parameters (4 currently optimized)
All parameters loaded from their respective configuration locations:

1. **`direction_change_threshold`**: [0.6, 0.95]
   - **Source**: `config.yaml` → `segmentation.change_detection`
   - **Purpose**: Sensitivity for detecting significant direction changes
   - **Type**: float

2. **`citation_confidence_boost`**: [0.05, 0.2]
   - **Source**: `config.yaml` → `segmentation.citation_analysis`
   - **Purpose**: Boost factor for citation-based confidence
   - **Type**: float

3. **`min_keyword_frequency_ratio`**: [0.05, 0.15]
   - **Source**: `config.yaml` → `data_processing.keyword_filter`
   - **Purpose**: Minimum frequency ratio for keyword inclusion
   - **Type**: float

4. **`top_k_keywords`**: [10, 30]
   - **Source**: `config.yaml` → `data_processing.keyword_filter`
   - **Purpose**: Number of top keywords for period cohesion calculation
   - **Type**: int

### Fixed Parameters (not optimized, loaded from config.yaml)
These parameters remain constant during optimization:

#### Penalty System Parameters
- **`min_period_years`**: 3 - Minimum acceptable period length
- **`max_period_years`**: 14 - Maximum acceptable period length
- **`lambda_short`**: 0.05 - Penalty weight for short periods
- **`lambda_long`**: 0.03 - Penalty weight for long periods
- **`lambda_count`**: 0.02 - Penalty weight for over-segmentation
- **`target_segments_upper`**: 8 - No penalty threshold for segment count

#### Objective Function Parameters
- **`cohesion_weight`**: 0.8 - Weight for cohesion in objective function
- **`separation_weight`**: 0.2 - Weight for separation in objective function

## Data Structures

### Trial Result
Each trial returns a comprehensive result dictionary:
```python
{
    "trial_id": int,
    "parameters": Dict[str, Any],       # Only optimized parameters
    "objective_score": float,           # Final score including penalties
    "cohesion_score": float,           # Period cohesion metric
    "separation_score": float,         # Period separation metric
    "num_segments": int,               # Number of detected periods
    "boundary_f1": float,              # Boundary detection F1-score
    "segment_f1": float,               # Segment matching F1-score
    "runtime_seconds": float,          # Trial execution time
    "error_message": str | None,       # Error details if trial failed
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
    final_score: float,            # Combined cohesion-separation score
    cohesion_score: float,         # Period internal consistency
    separation_score: float,       # Period distinctiveness
    num_segments: int,             # Number of periods detected
    num_transitions: int,          # Number of period boundaries
    cohesion_details: str,         # Detailed cohesion computation
    separation_details: str,       # Detailed separation computation
    methodology: str,              # Score computation method
)
```

## Key Implementation Details

### 1. Unified Configuration Loading
- **Parameter Extraction**: `create_penalty_config_from_algorithm_config()` extracts penalty parameters from main config
- **Automatic Synchronization**: Optimization always uses current algorithm parameters
- **Single Update Point**: Change parameters once in `config.yaml`, affects both systems

### 2. Efficient Trial Evaluation
- **Data Caching**: Academic years loaded once per domain and cached globally
- **Segmentation-Only Mode**: Skips characterization during optimization for speed
- **Logging Suppression**: Reduces console noise during non-verbose optimization
- **Error Handling**: Failed trials get penalty scores rather than crashing

### 3. Robust Parameter Handling
- **Dynamic Configuration**: Can optimize any AlgorithmConfig field
- **Type Safety**: Automatic type conversion (int, float, categorical)
- **Validation**: Parameter bounds and constraints enforced
- **Source Tracking**: Parameters loaded from appropriate config sections

### 4. Comprehensive Logging
- **Configuration Display**: Shows complete optimization setup before trials
- **Parameter Details**: Types, ranges, descriptions for all parameters
- **Progress Monitoring**: Real-time best score tracking with tqdm
- **Performance Metrics**: Trial timing and success rates

### 5. Performance Optimization
- **Parallel Processing**: Multi-worker trial evaluation
- **Memory Efficiency**: Careful caching and cleanup
- **Fast Convergence**: Effective parameter space and acquisition function

## Output Files

### Trial Results
- **Location**: `results/optimization_logs/{domain}.csv`
- **Content**: All trial results with parameters and metrics
- **Format**: CSV with dynamic parameter columns
- **Columns**: trial_id, objective_score, metrics, optimized parameters, runtime

### Best Parameters
- **Location**: `results/optimized_params/{domain}.json`
- **Content**: Best parameter configuration for domain
- **Format**: JSON with parameter values, scores, and metadata
- **Usage**: Can be loaded and applied to algorithm configuration

### Progress Plots
- **Location**: `results/optimization_logs/{domain}_best_score.png`
- **Content**: Best score progression over trials
- **Format**: PNG plot showing optimization convergence

## Performance Characteristics

### Typical Optimization Run
- **Domains**: 6-8 academic domains available
- **Trials per Domain**: 100 trials (configurable)
- **Time per Trial**: 5-30 seconds (depends on domain size)
- **Total Time**: 10-50 minutes per domain
- **Memory Usage**: ~1-2GB (cached data)
- **Convergence**: Usually stable after 50-70 trials

### Current Parameter Space Efficiency
- **Active Parameters**: 4 (reduced from 15+ original)
- **Search Space**: 4-dimensional continuous/discrete space
- **Convergence**: Faster than larger parameter spaces
- **Coverage**: Good exploration-exploitation balance

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

### Current Architecture
```
core/optimization/
├── optimization.py           (343 lines) - Trial scoring, caching, config integration
├── bayesian_optimizer.py     (164 lines) - Bayesian optimization engine
├── objective_function.py     (389 lines) - Objective function computation
├── penalty.py               (207 lines) - Penalty system implementation
├── optimization_config.py    (56 lines)  - Optimization config loading
├── __init__.py              (55 lines)  - Public interface and utilities
└── README.md                (this file) - Comprehensive documentation
```

### Design Principles
- **Unified Configuration**: Single source of truth for shared parameters
- **Functional Programming**: Pure functions where possible
- **Robust Error Handling**: Graceful failure with penalty scores
- **Efficient Caching**: Minimize redundant data loading
- **Comprehensive Logging**: Detailed progress and error reporting
- **Modular Design**: Clear separation of concerns

## Recent Evolution

### Unified Configuration System (Latest)
- **Problem**: Parameter duplication between `config.yaml` and `optimization.yaml`
- **Solution**: Load penalty/objective parameters from `config.yaml` only
- **Benefit**: Perfect synchronization between algorithm and optimization
- **Implementation**: `create_penalty_config_from_algorithm_config()` function

### Parameter Space Optimization
- **Original**: ~15 parameters with many showing minimal impact
- **Current**: 4 carefully selected parameters with significant impact
- **Benefit**: Faster convergence, more focused optimization
- **Selection Criteria**: Parameters that significantly affect objective scores

### Enhanced Logging and Monitoring
- **Configuration Logging**: Complete optimization setup displayed before trials
- **Progress Tracking**: Real-time monitoring with parameter details
- **Performance Metrics**: Trial timing and success rate monitoring

## Future Enhancements

### Configuration System
1. **Parameter Dependencies**: Support conditional parameter spaces
2. **Domain-Specific Configs**: Different parameter sets per domain
3. **Configuration Validation**: Enhanced parameter constraint checking

### Optimization Strategy
1. **Multi-Objective Optimization**: Optimize multiple metrics simultaneously
2. **Transfer Learning**: Use optimization results across similar domains
3. **Ensemble Methods**: Combine multiple optimization runs
4. **Adaptive Parameter Bounds**: Dynamic ranges based on convergence

### Performance Improvements
1. **Warm Starts**: Initialize optimization with previous results
2. **Early Stopping**: Stop optimization when convergence detected
3. **Hyperparameter Scheduling**: Dynamic acquisition function parameters
4. **Cross-Validation**: Split domains into train/validation sets

### Parameter Space Extensions
1. **Reactivate Parameters**: Add back `cohesion_weight`, `separation_weight` if needed
2. **New Algorithm Parameters**: Neural network hyperparameters, attention weights
3. **Preprocessing Parameters**: Data filtering and transformation settings
4. **Evaluation Parameters**: Different scoring metrics and weights 