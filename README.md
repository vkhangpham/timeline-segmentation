# Timeline Analysis - Academic Literature Segmentation

A sophisticated pipeline for detecting paradigm shifts in academic literature using change point detection, network analysis, and temporal modeling.

## Overview

This system analyzes scientific publication data to automatically identify periods of significant paradigm shifts in academic domains. It uses a multi-stage functional pipeline combining direction change detection, citation analysis, and network modeling to segment academic timelines into coherent periods.

## Architecture

### Core Pipeline Stages

1. **Data Loading** - Processes publication JSON and citation GraphML files into structured `AcademicYear` objects
2. **Change Point Detection** - Identifies boundary years using dual-metric shift signals (direction + citation acceleration)
3. **Segmentation** - Converts boundary years into contiguous `AcademicPeriod` objects
4. **Beam Search Refinement** - Optimizes segment boundaries through merge/split operations
5. **Period Characterization** - Analyzes each period using network analysis and LLM-based topic modeling

### Key Components

- **Data Models** (`core/data/data_models.py`) - Immutable dataclasses for papers, academic years, and periods
- **Change Point Detection** (`core/segmentation/change_point_detection.py`) - Dual-metric paradigm shift detection
- **Network Analysis** (`core/segment_modeling/segment_modeling.py`) - Citation network analysis and stability metrics
- **Objective Function** (`core/optimization/objective_function.py`) - Cohesion/separation evaluation with anti-gaming measures
- **Evaluation System** (`core/evaluation/evaluation.py`) - Comprehensive evaluation with baselines and auto-metrics

## Installation

```bash
# Clone repository
git clone <repository-url>
cd timeline

# Create conda environment
conda create -n timeline python=3.9
conda activate timeline

# Install dependencies
pip install -r requirements.txt

# Note: scikit-optimize is included for Bayesian optimization
# If you encounter issues, install manually:
# pip install scikit-optimize
```

## Usage

### Basic Analysis

```bash
# Analyze a single domain
python run_timeline_analysis.py --domain deep_learning --verbose

# Analyze all available domains
python run_timeline_analysis.py --domain all --verbose

# Run segmentation only (skip characterization)
python run_timeline_analysis.py --domain computer_vision --segmentation-only
```

### Evaluation

```bash
# Run comprehensive evaluation for a domain
python run_evaluation.py --domain deep_learning --verbose

# Run evaluation for all domains
python run_evaluation.py --domain all --verbose

# Get detailed evaluation metrics
python run_evaluation.py --domain machine_learning --verbose
```

### Parameter Optimization

```bash
# Optimize parameters for a single domain using Bayesian optimization
python scripts/run_optimization.py --domain deep_learning --verbose

# Use custom configuration file
python scripts/run_optimization.py --domain machine_learning --config custom_config.yaml

# Quick optimization (uses default config)
python scripts/run_optimization.py --domain computer_vision
```

### Advanced Options

```bash
# Custom thresholds
python run_timeline_analysis.py --domain machine_learning \
    --direction-threshold 0.15 \
    --validation-threshold 0.4 \
    --citation-boost-rate 0.6

# Custom evaluation parameters
python run_evaluation.py --domain computer_vision \
    --direction-threshold 0.12 \
    --citation-boost-rate 0.15
```

### Custom Analysis
```python
from core.pipeline.orchestrator import analyze_timeline
from core.utils.config import AlgorithmConfig

config = AlgorithmConfig.from_config_file(domain_name="computer_vision")
result = analyze_timeline("computer_vision", config, verbose=True)

print(f"Detected {len(result.periods)} periods")
for period in result.periods:
    print(f"{period.start_year}-{period.end_year}: {period.topic_label}")
```

## Streamlit Visualization Interface

An interactive web application provides detailed visualizations for every stage of the timeline analysis algorithm.

### Features

- **Real-time Configuration**: Adjust algorithm parameters with interactive controls
- **Stage-by-Stage Visualization**: Explore each step from data loading to final results
- **Performance Monitoring**: Track execution timing and identify bottlenecks
- **Manual Execution Controls**: Handle computationally expensive operations on-demand
- **Data Export**: Export results and configurations for reproducibility
- **Comprehensive Evaluation**: Compare algorithm performance against baselines with auto-metrics

### Running the Interface

```bash
# Install additional dependencies
pip install streamlit plotly

# Launch the application
streamlit run streamlit_app.py
```

The application opens at `http://localhost:8501` with a sequential workflow:
1. **Data Exploration** - Examine input data and statistics
2. **Change Detection** - Visualize paradigm shift detection
3. **Segmentation** - Convert boundary years into periods
4. **Characterization** - Network analysis and topic labeling (manual execution)
5. **Beam Refinement** - Optimize boundaries through merge/split operations
6. **Evaluation** - Comprehensive performance assessment with baselines and auto-metrics
7. **Final Results** - Complete timeline with comprehensive analysis

## Parameter Optimization System

The system includes a sophisticated parameter optimization framework with unified configuration management that automatically tunes algorithm parameters for optimal performance on each domain.

### Unified Configuration Architecture

The optimization system uses a two-file configuration approach:

- **`config.yaml`**: Contains penalty system and objective function parameters (shared with main algorithm)
- **`optimization.yaml`**: Contains only optimization-specific settings (search space, strategy, execution)

This ensures **perfect synchronization** between the main algorithm and optimization process - no parameter duplication or inconsistencies.

### Bayesian Optimization

The system uses Bayesian optimization with Gaussian Processes to efficiently explore the parameter space:

- **Smart Exploration**: Uses acquisition functions (Expected Improvement) to balance exploration vs exploitation
- **Efficient**: Requires fewer trials than grid or random search
- **Adaptive**: Learns from previous trials to guide future parameter selection
- **Comprehensive Logging**: Shows complete configuration before optimization starts

### Current Parameter Space

The optimization system currently tunes 4 key parameters:

```yaml
# config/optimization.yaml - Parameter search space
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
    description: "Minimum frequency ratio for keyword inclusion (data_processing.keyword_filter)"

  top_k_keywords:
    type: int
    range: [10, 30]
    description: "Number of top keywords to consider for period cohesion (data_processing.keyword_filter)"

# Search strategy
search:
  n_calls: 100
  n_initial_points: 20
  strategy: "bayesian"
  acquisition_function: "EI"

# Execution settings
execution:
  max_workers: 8
  cache_academic_years: true
```

**Note**: Penalty and objective function parameters are automatically loaded from `config.yaml` to ensure consistency with the main algorithm.

### Optimization Results

Results are automatically saved to:
- `results/optimization_logs/{domain}.csv` - Detailed trial results with all parameters and metrics
- `results/optimized_params/{domain}.json` - Best configuration found with validation scores
- `results/optimization_logs/{domain}_best_score.png` - Optimization convergence plot

### Usage Examples

```bash
# Quick optimization with tqdm progress bar
python scripts/run_optimization.py --domain computer_vision

# Verbose optimization with full configuration logging
python scripts/run_optimization.py --domain deep_learning --verbose

# Optimize all domains
python scripts/run_optimization.py --domain all --verbose

# Custom configuration file
python scripts/run_optimization.py --domain applied_mathematics --config my_config.yaml
```

## Algorithm Details

### Change Point Detection

The system uses a dual-metric approach to detect paradigm shifts:

**Direction Change Detection:**
- Computes frequency-weighted scoring using cumulative baseline comparison
- Scoring methods: Weighted Jaccard similarity or Jensen-Shannon divergence
- Adaptive thresholding: global p90, p95, p99 of score distribution
- Threshold: `direction_change_threshold = 0.1` (default, or adaptive)

**Citation Acceleration Detection:**
- MAD-based year-over-year growth analysis on citation counts
- Robust threshold: `median + 3 * MAD`
- Cooldown period to prevent clustered detections

### Beam Search Refinement

Optimizes segment boundaries through:
- **Merge Operations**: Combines adjacent similar periods
- **Split Operations**: Divides periods at optimal split points
- **Objective Function Scoring**: Evaluates each candidate configuration
- **Beam Width**: Maintains top-k candidates (default: 5)

### Network Analysis

Each period is characterized using:
- **Network Stability**: Graph connectivity and robustness measures
- **Community Persistence**: Stability of research communities over time
- **Flow Stability**: Consistency of citation flow patterns
- **Centrality Consensus**: Agreement across different centrality measures

### Objective Function

Quality evaluation using:
- **Cohesion** (80%): Jaccard similarity between papers and top-k keywords
- **Separation** (20%): Jensen-Shannon divergence between adjacent periods
- **Anti-gaming**: Size-weighted averaging and segment count penalties

### Evaluation System

Comprehensive evaluation includes:

**Objective Function Scoring:**
- Algorithm performance using cohesion/separation metrics
- Detailed breakdown of period-level and transition-level scores

**Baseline Comparisons:**
- Gemini-generated baselines from manual reference timelines
- Manual expert-created reference timelines
- Fixed-interval baselines (uniform segmentation)

**Auto-Metrics:**
- **Boundary F1**: Precision/recall for boundary detection (±2 year tolerance)
- **Segment F1**: Precision/recall for segment overlap matching
- Comparison against manual reference timelines

## Configuration

Edit `config.yaml` to customize algorithm behavior:

```json
{
  "detection_parameters": {
    "direction_change_threshold": 0.1,
    "direction_threshold_strategy": "global_p90",
    "direction_scoring_method": "weighted_jaccard",
    "citation_confidence_boost": 0.1,
    "citation_support_window_years": 2,
    "min_papers_per_year": 100
  },
  "objective_function": {
    "cohesion_weight": 0.8,
    "separation_weight": 0.2,
    "top_k_keywords": 100,
    "min_keyword_frequency_ratio": 0.1
  },
  "beam_search": {
    "enabled": true,
    "beam_width": 5,
    "max_splits_per_segment": 1,
    "min_period_years": 3,
    "max_period_years": 50
  },
  "diagnostics": {
    "save_direction_diagnostics": true,
    "diagnostic_top_keywords_limit": 10
  }
}
```

## Data Format

### Input Files

Place domain data in `resources/{domain}/`:
- `{domain}_docs_info.json` - Publication metadata with papers
- `{domain}_entity_relation_graph.graphml.xml` - Citation network

### Manual Reference Files

Place reference timelines in `data/references/`:
- `{domain}_manual.json` - Expert-created reference timeline
- `{domain}_gemini.json` - LLM-generated reference timeline

### Paper Format
```json
{
  "paper_id": {
    "title": "Paper Title",
    "content": "Abstract or full text",
    "pub_year": 2020,
    "cited_by_count": 42,
    "keywords": ["keyword1", "keyword2"],
    "children": ["citing_paper_id1", "citing_paper_id2"],
    "description": "Paper description"
  }
}
```

## Output

### Timeline Analysis Results
Results are saved to `results/timelines/{domain}_timeline_analysis.json`:

```json
{
  "domain_name": "deep_learning",
  "confidence": 0.85,
  "boundary_years": [1986, 2006, 2012],
  "periods": [
    {
      "start_year": 1980,
      "end_year": 1985,
      "total_papers": 245,
      "topic_label": "Neural Network Foundations",
      "topic_description": "Early perceptron and backpropagation research...",
      "confidence": 0.82,
      "network_stability": 0.67,
      "network_metrics": {...}
    }
  ],
  "narrative_evolution": "Deep learning timeline evolution: Period 1..."
}
```

### Evaluation Results
Results are saved to `results/evaluation/{domain}_evaluation.json`:

```json
{
  "algorithm_result": {
    "objective_score": 0.756,
    "cohesion_score": 0.823,
    "separation_score": 0.492,
    "num_segments": 4,
    "boundary_years": [1986, 2006, 2012]
  },
  "baseline_results": [
    {
      "baseline_name": "manual",
      "objective_score": 0.721,
      "num_segments": 3,
      "boundary_years": [1990, 2010]
    }
  ],
  "auto_metrics": {
    "boundary_f1": 0.667,
    "boundary_precision": 0.750,
    "boundary_recall": 0.600,
    "segment_f1": 0.545,
    "segment_precision": 0.667,
    "segment_recall": 0.462
  }
}
```

## Development

### Core Principles

1. **Functional Programming** - Pure functions with immutable data structures
2. **Fail-Fast Error Handling** - No fallbacks or error masking for debugging clarity
3. **Real Data Only** - No mock or synthetic data in development/testing
4. **Comprehensive Testing** - Test on representative real data subsets
5. **Quality Evaluation** - Rigorous analysis of results with quantitative metrics

### Development Journal

Track all development items in `dev_journal_phase*.md` files using the structured format for problems, features, evaluations, and refactors.

## Examples

### Analyzing Deep Learning Evolution
```bash
python run_timeline_analysis.py --domain deep_learning --verbose
```

Expected output identifies major periods:
- 1980-1985: Neural Network Foundations  
- 1986-2005: Expert Systems and Symbolic AI
- 2006-2011: Deep Learning Renaissance
- 2012-2020: Transformer and Attention Era

### Evaluating Timeline Quality
```bash
python run_evaluation.py --domain deep_learning --verbose
```

Expected evaluation output:
- Algorithm objective score: 0.756
- Baseline comparison: outperforms manual reference (0.721)
- Boundary F1: 0.667 (±2 year tolerance)
- Segment F1: 0.545 (overlap-based matching)

### Custom Analysis with Evaluation
```python
from core.pipeline.orchestrator import analyze_timeline
from core.evaluation.evaluation import run_comprehensive_evaluation
from core.utils.config import AlgorithmConfig

config = AlgorithmConfig.from_config_file(domain_name="computer_vision")
result = analyze_timeline("computer_vision", config, verbose=True)

# Run comprehensive evaluation
evaluation = run_comprehensive_evaluation(
    domain_name="computer_vision",
    timeline_result=result,
    algorithm_config=config,
    verbose=True
)

print(f"Algorithm score: {evaluation['algorithm_result']['objective_score']:.3f}")
print(f"Boundary F1: {evaluation['auto_metrics']['boundary_f1']:.3f}")
```

### Optimizing Parameters for Better Performance
```bash
# Optimize parameters for computer vision domain
python scripts/run_optimization.py --domain computer_vision --verbose
```

Expected optimization output:
- Trial progress with objective scores
- Best configuration found after optimization
- Validation metrics (Boundary-F1, Segment-F1) for quality assessment
- Results saved to `results/optimized_params/computer_vision.json`

### Using Optimized Parameters
```python
import json
from core.utils.config import AlgorithmConfig
from core.pipeline.orchestrator import analyze_timeline

# Load optimized parameters
with open("results/optimized_params/computer_vision.json", "r") as f:
    optimized_params = json.load(f)

# Create config with optimized parameters
config = AlgorithmConfig.from_config_file(domain_name="computer_vision")
for param, value in optimized_params["parameters"].items():
    setattr(config, param, value)

# Run analysis with optimized configuration
result = analyze_timeline("computer_vision", config, verbose=True)
print(f"Optimized analysis produced {len(result.periods)} periods")
```

## Contributing

1. Follow functional programming principles
2. Use type hints and dataclasses
3. Maintain comprehensive logging
4. Test on real data subsets
5. Document in development journal

## License

[Add appropriate license] 