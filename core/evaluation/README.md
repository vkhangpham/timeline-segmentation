# Timeline Segmentation Evaluation Module

Evaluation framework for timeline segmentation with dual reference scoring system.

## Features

### 1. Objective Function Scoring
- **Function**: `evaluate_timeline_result(timeline_result, algorithm_config, verbose=False)`
- **Purpose**: Calculate objective function score for any timeline segmentation result
- **Output**: EvaluationResult with objective score, cohesion/separation scores, and detailed metrics

### 2. Reference Loading
Two types of reference timelines are supported:

#### Gemini Reference
- **Function**: `load_gemini_reference(domain_name, academic_years, min_data_year, max_data_year, algorithm_config, use_cache=True, verbose=False)`
- **Purpose**: Load Gemini reference timeline from `data/references/{domain}_gemini.json`
- **Data Processing**: Same filtering as pipeline (paper filter, year filter, keyword filter)

#### Perplexity Reference
- **Function**: `load_perplexity_reference(domain_name, academic_years, min_data_year, max_data_year, algorithm_config, use_cache=True, verbose=False)`
- **Purpose**: Load Perplexity reference timeline from `data/references/{domain}_perplexity.json`
- **Data Processing**: Same filtering as pipeline

### 3. Baseline Creation
Fixed year baselines are supported:

#### Fixed Year Baselines
- **Function**: `create_fixed_year_baseline(domain_name, year_interval, academic_years, min_data_year, max_data_year, algorithm_config, use_cache=True, verbose=False)`
- **Purpose**: Create baseline with fixed year intervals (5-year or 10-year)
- **Data Processing**: Same filtering as pipeline

### 4. Dual Reference Auto-Metrics
Metrics are calculated against both Gemini and Perplexity references:

#### Boundary F1
- **Function**: `calculate_boundary_f1(predicted_boundaries, ground_truth_boundaries, tolerance=2)`
- **Purpose**: Compare boundary years with year mismatch tolerance
- **Tolerance**: ±2 years by default

#### Segment F1
- **Function**: `calculate_segment_f1(predicted_segments, ground_truth_segments, max_segments_per_match=3)`
- **Purpose**: Compare segments allowing up to 3 predicted segments per ground truth segment
- **Logic**: Ground truth segment [1990, 2000] matches predicted segments [1990,1992], [1993,1996], [1997,2001]

### 5. Final Evaluation System
- **Function**: `run_final_evaluation(domain_name, timeline_result, algorithm_config, data_directory="resources", use_cache=True, verbose=False)`
- **Purpose**: Run comprehensive evaluation with dual reference system
- **Methods Evaluated**: 3 methods (Algorithm, 5-year baseline, 10-year baseline)
- **Metrics per Method**: 4 scores (boundary F1 and segment F1 vs both Gemini and Perplexity references)
- **Output**: FinalEvaluationResult with all methods metrics against both references

## Usage

### Command Line Interface
```bash
# Full evaluation for single domain
python scripts/run_evaluation.py --domain art --verbose

# Full evaluation for all domains
python scripts/run_evaluation.py --domain all --verbose

# Baseline-only evaluation (5-year or 10-year only)
python scripts/run_evaluation.py --domain art --baseline-only 5-year
python scripts/run_evaluation.py --domain art --baseline-only 10-year
```

### Programmatic Usage
```python
from core.utils.config import AlgorithmConfig
from core.pipeline.orchestrator import analyze_timeline
from core.evaluation.evaluation import run_final_evaluation
from core.evaluation.analysis import display_final_evaluation_summary

# Setup
domain_name = "art"
algorithm_config = AlgorithmConfig.from_config_file(domain_name=domain_name)

# Run pipeline
timeline_result = analyze_timeline(
    domain_name=domain_name,
    algorithm_config=algorithm_config,
    segmentation_only=True,
)

# Final evaluation with dual references
evaluation_result = run_final_evaluation(
    domain_name=domain_name,
    timeline_result=timeline_result,
    algorithm_config=algorithm_config,
)

# Display results
display_final_evaluation_summary(evaluation_result, verbose=True)

# Access metrics
alg_metrics = evaluation_result.all_methods_metrics.algorithm_metrics
print(f"Algorithm vs Gemini - Boundary F1: {alg_metrics.gemini_boundary_f1:.3f}")
print(f"Algorithm vs Perplexity - Boundary F1: {alg_metrics.perplexity_boundary_f1:.3f}")
```

## Data Structures

### EvaluationResult
- `objective_score`: Final objective function score
- `cohesion_score`: Average cohesion score
- `separation_score`: Average separation score
- `num_segments`: Number of segments
- `boundary_years`: List of boundary years
- `methodology`: Description of calculation method
- `details`: Additional details (cohesion/separation per period)

### BaselineResult
- `baseline_name`: Name of baseline (5-year, 10-year)
- `objective_score`: Objective function score
- `cohesion_score`: Cohesion score
- `separation_score`: Separation score
- `num_segments`: Number of segments
- `boundary_years`: List of boundary years
- `academic_periods`: List of AcademicPeriod objects

### MethodMetrics
- `method_name`: Name of the method (Algorithm, 5-year, 10-year)
- `objective_score`: Objective function score
- `gemini_boundary_f1`: Boundary F1 score vs Gemini reference
- `gemini_segment_f1`: Segment F1 score vs Gemini reference
- `perplexity_boundary_f1`: Boundary F1 score vs Perplexity reference
- `perplexity_segment_f1`: Segment F1 score vs Perplexity reference

### FinalEvaluationResult
- `domain_name`: Domain name
- `algorithm_result`: EvaluationResult for algorithm
- `baseline_results`: List of BaselineResult objects (5-year, 10-year)
- `all_methods_metrics`: AllMethodsMetrics with comprehensive dual reference metrics
- `ranking`: Dictionary of method -> objective_score
- `summary`: Text summary of results

## Output Files

Results are saved to `results/evaluation/{domain}_evaluation.json` with complete evaluation data in JSON format.

## Key Implementation Details

1. **Dual Reference System**: All methods (Algorithm, 5-year, 10-year) are evaluated against both Gemini and Perplexity references

2. **Data Consistency**: All baselines and references use the same data processing pipeline as the main algorithm

3. **Robust Caching**: Efficient caching system for reference and baseline results

4. **Flexible Evaluation**: Can evaluate any timeline result, not just from the main pipeline

5. **Comprehensive Metrics**: 4 scores per method (boundary F1 and segment F1 vs both references)

## Dependencies

- `core.data.data_models`: AcademicPeriod, TimelineAnalysisResult
- `core.data.data_processing`: Data loading and processing functions
- `core.optimization.objective_function`: Objective function computation
- `core.utils.config`: Algorithm configuration
- `core.utils.logging`: Logging utilities

## Example Output

```
FINAL EVALUATION RESULTS: MACHINE_LEARNING
================================================================================

Objective Scores:
----------------------------------------
Method          Objective Score
----------------------------------------
Algorithm       -1.554         
5-year          0.541          
10-year         0.508          

Auto-metrics vs References (tolerance = 2 years):
--------------------------------------------------------------------------------
Method       Gemini Boundary Gemini Segment Perplexity Boundary Perplexity Segment
--------------------------------------------------------------------------------
Algorithm    0.400           0.960          0.545               0.960           
5-year       0.343           0.296          0.432               0.605           
10-year      0.200           0.818          0.273               0.870           
```
