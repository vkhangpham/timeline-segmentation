# Timeline Analysis - Academic Literature Segmentation

A sophisticated pipeline for detecting paradigm shifts in academic literature using change point detection, network analysis, and temporal modeling.

## Overview

This system analyzes scientific publication data to automatically identify periods of significant paradigm shifts in academic domains. It uses a multi-stage functional pipeline combining direction change detection, citation analysis, and network modeling to segment academic timelines into coherent periods.

## Architecture

### Core Pipeline Stages

1. **Data Loading** - Processes publication JSON and citation GraphML files into structured `AcademicYear` objects
2. **Change Point Detection** - Identifies boundary years using dual-metric shift signals (direction + citation acceleration)
3. **Segmentation** - Converts boundary years into contiguous `AcademicPeriod` objects
4. **Period Characterization** - Analyzes each period using network analysis and LLM-based topic modeling
5. **Period Merging** - Intelligently merges similar adjacent periods based on keyword overlap

### Key Components

- **Data Models** (`core/data/data_models.py`) - Immutable dataclasses for papers, academic years, and periods
- **Change Point Detection** (`core/segmentation/change_point_detection.py`) - Dual-metric paradigm shift detection
- **Network Analysis** (`core/segment_modeling/segment_modeling.py`) - Citation network analysis and stability metrics
- **Objective Function** (`core/optimization/objective_function.py`) - Cohesion/separation evaluation with anti-gaming measures

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

### Advanced Options

```bash
# Custom thresholds
python run_timeline_analysis.py --domain machine_learning \
    --direction-threshold 0.15 \
    --validation-threshold 0.4 \
    --citation-boost-rate 0.6

# Different granularity levels
python run_timeline_analysis.py --domain applied_mathematics --granularity 1  # Fine-grained
python run_timeline_analysis.py --domain art --granularity 5                  # Coarse-grained
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
5. **Period Merging** - Intelligent consolidation based on similarity
6. **Final Results** - Complete timeline with comprehensive analysis

## Algorithm Details

### Change Point Detection

The system uses a dual-metric approach to detect paradigm shifts:

**Direction Change Detection:**
- Computes novelty-overlap metric: `S_dir = novelty × (1 - overlap)`
- `novelty = |new_keywords| / |current_keywords|`
- `overlap = |shared_keywords| / |previous_keywords|`
- Threshold: `direction_threshold = 0.1`

**Citation Acceleration Detection:**
- Multi-scale gradient analysis on citation counts
- Adaptive thresholding based on data distribution
- Scales: [1, 3, 5] year windows

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

## Configuration

Edit `config.json` to customize algorithm behavior:

```json
{
  "detection_parameters": {
    "direction_threshold": 0.1,      // Paradigm shift sensitivity
    "validation_threshold": 0.3,     // Signal combination threshold
    "citation_boost_rate": 0.5       // Citation signal weight
  },
  "objective_function": {
    "cohesion_weight": 0.8,         // Intra-period coherence weight
    "separation_weight": 0.2,       // Inter-period distinctiveness weight
    "top_k_keywords": 15            // Keywords for evaluation
  },
  "anti_gaming": {
    "min_segment_size": 50,         // Minimum papers per segment
    "size_weight_power": 0.5,       // Size weighting power
    "enable_size_weighting": true   // Prevent micro-segment gaming
  }
}
```

## Data Format

### Input Files

Place domain data in `data/references/`:
- `{domain}_docs_info.json` - Publication metadata with papers
- `{domain}_entity_relation_graph.graphml.xml` - Citation network

### Paper Format
```json
{
  "id": "unique_paper_id",
  "title": "Paper Title",
  "content": "Abstract or full text",
  "pub_year": 2020,
  "cited_by_count": 42,
  "keywords": ["keyword1", "keyword2"],
  "children": ["citing_paper_id1", "citing_paper_id2"],
  "description": "Paper description"
}
```

## Output

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
      "network_stability": 0.67
    }
  ],
  "narrative_evolution": "Deep learning timeline evolution: Period 1..."
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

## Contributing

1. Follow functional programming principles
2. Use type hints and dataclasses
3. Maintain comprehensive logging
4. Test on real data subsets
5. Document in development journal

## License

[Add appropriate license] 