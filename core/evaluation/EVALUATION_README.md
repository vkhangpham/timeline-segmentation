# Timeline Segmentation Evaluation Module

This module provides comprehensive evaluation capabilities for timeline segmentation results including objective function scoring, baseline comparisons, and auto-metrics calculation.

## Features

### 1. Objective Function Scoring
- **Function**: `evaluate_timeline_result(timeline_result, algorithm_config, verbose=False)`
- **Purpose**: Calculate objective function score for any timeline segmentation result
- **Output**: EvaluationResult with objective score, cohesion/separation scores, and detailed metrics

### 2. Baseline Creation & Evaluation
Four types of baselines are supported:

#### Gemini Baseline
- **Function**: `create_gemini_baseline(domain_name, algorithm_config, data_directory="resources", verbose=False)`
- **Purpose**: Create baseline using Gemini reference timeline from `data/references/{domain}_gemini.json`
- **Data Processing**: Same filtering as pipeline (paper filter, year filter, keyword filter)

#### Manual Baseline  
- **Function**: `create_manual_baseline(domain_name, algorithm_config, data_directory="resources", verbose=False)`
- **Purpose**: Create baseline using manual reference timeline from `data/references/{domain}_manual.json`
- **Data Processing**: Same filtering as pipeline

#### Fixed Year Baselines
- **Function**: `create_fixed_year_baseline(domain_name, algorithm_config, year_interval, data_directory="resources", verbose=False)`
- **Purpose**: Create baseline with fixed year intervals (5-year or 10-year)
- **Data Processing**: Same filtering as pipeline

### 3. Auto-Metrics
Two types of F1-scores are calculated against Manual baseline:

#### Boundary F1
- **Function**: `calculate_boundary_f1(predicted_boundaries, ground_truth_boundaries, tolerance=2)`
- **Purpose**: Compare boundary years with year mismatch tolerance
- **Tolerance**: ±2 years by default

#### Segment F1
- **Function**: `calculate_segment_f1(predicted_segments, ground_truth_segments, max_segments_per_match=3)`
- **Purpose**: Compare segments allowing up to 3 predicted segments per ground truth segment
- **Logic**: Ground truth segment [1990, 2000] matches predicted segments [1990,1992], [1993,1996], [1997,2001]

### 4. Comprehensive Evaluation
- **Function**: `run_comprehensive_evaluation(domain_name, timeline_result, algorithm_config, data_directory="resources", verbose=False)`
- **Purpose**: Run all evaluations and return complete results
- **Output**: ComprehensiveEvaluationResult with algorithm result, baselines, auto-metrics, and ranking

## Usage

### Command Line Interface
```bash
# Full evaluation for single domain
python run_evaluation.py --domain art --verbose

# Full evaluation for all domains
python run_evaluation.py --domain all --verbose

# Baseline-only evaluation
python run_evaluation.py --domain art --baseline-only manual
python run_evaluation.py --domain art --baseline-only gemini
python run_evaluation.py --domain art --baseline-only 5-year
python run_evaluation.py --domain art --baseline-only 10-year
```

### Programmatic Usage
```python
from core.utils.config import AlgorithmConfig
from core.pipeline.orchestrator import analyze_timeline
from core.evaluation.evaluation import run_comprehensive_evaluation

# Setup
domain_name = "art"
algorithm_config = AlgorithmConfig.from_config_file(domain_name=domain_name)

# Run pipeline
timeline_result = analyze_timeline(
    domain_name=domain_name,
    algorithm_config=algorithm_config,
    segmentation_only=True,
)

# Comprehensive evaluation
evaluation_result = run_comprehensive_evaluation(
    domain_name=domain_name,
    timeline_result=timeline_result,
    algorithm_config=algorithm_config,
)

# Access results
print(f"Algorithm score: {evaluation_result.algorithm_result.objective_score:.3f}")
print(f"Boundary F1: {evaluation_result.auto_metrics.boundary_f1:.3f}")
print(f"Segment F1: {evaluation_result.auto_metrics.segment_f1:.3f}")
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
- `baseline_name`: Name of baseline (Gemini, Manual, 5-year, 10-year)
- `objective_score`: Objective function score
- `cohesion_score`: Cohesion score
- `separation_score`: Separation score
- `num_segments`: Number of segments
- `boundary_years`: List of boundary years
- `academic_periods`: List of AcademicPeriod objects

### AutoMetricResult
- `boundary_f1`: F1 score for boundary year matching
- `boundary_precision`: Precision for boundary matching
- `boundary_recall`: Recall for boundary matching
- `segment_f1`: F1 score for segment matching
- `segment_precision`: Precision for segment matching  
- `segment_recall`: Recall for segment matching
- `tolerance`: Year tolerance used for boundary matching
- `details`: Additional details about matching

### ComprehensiveEvaluationResult
- `domain_name`: Domain name
- `algorithm_result`: EvaluationResult for algorithm
- `baseline_results`: List of BaselineResult objects
- `auto_metrics`: AutoMetricResult vs Manual baseline
- `ranking`: Dictionary of method -> objective_score
- `summary`: Text summary of results

## Output Files

Results are saved to `results/evaluation/{domain}_evaluation.json` with complete evaluation data in JSON format.

## Key Implementation Details

1. **Data Consistency**: All baselines use the same data processing pipeline as the main algorithm (paper filtering, year filtering, keyword filtering)

2. **Robust Error Handling**: Baselines that fail (e.g., missing reference files, sparse data) are handled gracefully

3. **Flexible Evaluation**: Can evaluate any timeline result, not just from the main pipeline

4. **Performance**: Efficient parallel processing where possible

5. **Comprehensive Metrics**: Multiple evaluation perspectives (objective function, boundary accuracy, segment accuracy)

## Dependencies

- `core.data.data_models`: AcademicPeriod, TimelineAnalysisResult
- `core.data.data_processing`: Data loading and processing functions
- `core.optimization.objective_function`: Objective function computation
- `core.utils.config`: Algorithm configuration
- `core.utils.logging`: Logging utilities 