# Timeline Segmentation Algorithm

A scientific literature analysis system that detects paradigm shifts and temporal patterns across academic domains using advanced segmentation techniques.

## Overview

This codebase implements a comprehensive pipeline for analyzing the evolution of scientific literature, identifying key transition points, and characterizing distinct periods within academic domains.

**Supported Domains:** Applied Mathematics, Art, Computer Science, Computer Vision, Deep Learning, Machine Learning, Machine Translation, Natural Language Processing

## Core Components

### 🚀 Main Pipeline (`run_timeline_analysis.py`)

**Primary entry point** for the complete analysis pipeline:

```bash
# Analyze a single domain
python run_timeline_analysis.py --domain deep_learning

# Analyze all domains
python run_timeline_analysis.py --domain all

# Custom granularity (1=ultra_fine, 5=ultra_coarse)
python run_timeline_analysis.py --domain computer_vision --granularity 1
```

**Two-stage process:**
1. **Change Point Detection** → Identifies temporal segments using similarity-based algorithms
2. **Period Characterization** → Analyzes content and themes within each detected segment using citation network analysis

### 🎯 Parameter Optimization (`optimize_segmentation_bayesian.py`)

**Bayesian optimization** system for finding optimal algorithm parameters:

```bash
# Optimize single domain
python optimize_segmentation_bayesian.py deep_learning

# Optimize all domains
python optimize_segmentation_bayesian.py

# Custom evaluation budget
python optimize_segmentation_bayesian.py computer_vision --max-evals=200
```

Uses **Gaussian Process + Expected Improvement** to efficiently explore parameter space and maximize consensus-difference scores.

### ✅ Validation Framework (`validation/`)

**Ground truth validation** against manual reference data:

```bash
python -m validation.runner
```

- **`validation/core.py`** → Core validation metrics and reference data processing
- **`validation/runner.py`** → Validation execution and result aggregation

Validates algorithm performance using precision, recall, and F1-scores with 2-year and 5-year tolerance windows.

## Algorithm Architecture

### Core Modules (`core/`)

| Module | Purpose |
|--------|---------|
| `integration.py` | Main pipeline orchestration |
| `shift_signal_detection.py` | Change point detection algorithms |
| `similarity_segmentation.py` | Temporal similarity analysis |
| `objective_function.py` | Validated timeline quality evaluation |
| `algorithm_config.py` | Centralized parameter management |
| `data_loader.py` | JSON and graph data ingestion |
| `data_processing.py` | Rich citation graph processing |
| `period_signal_detection.py` | Network-based period characterization |
| `text_vectorization.py` | TF-IDF and embedding processing |

### Data Flow

```
JSON Literature Data + Citation Graph → Change Detection → Segmentation → Network Period Analysis → Results
                                              ↓
                                   Parameter Optimization ← Validation Framework
```

## Configuration

**Centralized configuration** via `ComprehensiveAlgorithmConfig`:
- **Granularity levels:** 1 (ultra-fine) to 5 (ultra-coarse)  
- **Detection thresholds:** Direction and validation sensitivity
- **Segment constraints:** Minimum/maximum lengths
- **Citation analysis:** Network stability and significance scoring
- **Optimization parameters:** Loaded from `results/optimized_parameters_bayesian.json`

## Data Structure

**Primary Data Sources:**
```
resources/                     # JSON data sources (preferred)
├── {domain}/
│   ├── {domain}_docs_info.json              # Paper metadata and content
│   └── {domain}_entity_relation_graph.graphml.xml  # Rich citation relationships

data/
└── references/                # Manual validation data (*_manual.json, *_gemini.json)

results/                       # Algorithm outputs and optimized parameters
```

**Data Pipeline Migration:**
- ✅ **JSON Data Sources**: Direct loading from `docs_info.json` files
- ✅ **Graph Data**: Rich citation networks from GraphML files  
- ✅ **Citation-based Significance**: Dynamic paper importance calculation
- ⚠️ **CSV Support**: Deprecated (will be removed in future versions)

## Key Features

- **Rich Citation Analysis** → Network-based period characterization with semantic relationships
- **Dynamic Significance Scoring** → Citation-based paper importance without external breakthrough lists
- **Fail-fast error handling** → Immediate error propagation for debugging
- **Functional programming** → Pure functions with immutable data structures  
- **Parameter transparency** → All decisions explainable and traceable
- **Multi-domain support** → Consistent analysis across scientific fields
- **Bayesian optimization** → Efficient parameter tuning with minimal evaluations

## Data Loading

The system uses a **tiered data loading approach**:

1. **Primary**: `{domain}_docs_info.json` + `{domain}_entity_relation_graph.graphml.xml`
2. **Legacy**: CSV files (deprecated, shows warnings)

**Example domain data structure:**
```json
{
  "paper_id": {
    "title": "Paper Title",
    "content": "Abstract and content",
    "pub_year": 2023,
    "cited_by_count": 150,
    "keywords": ["keyword1", "keyword2"],
    "children": ["citing_paper_id1", "citing_paper_id2"]
  }
}
```

## Quick Start

1. **Run basic analysis:**
   ```bash
   python run_timeline_analysis.py --domain deep_learning
   ```

2. **Optimize parameters:**
   ```bash
   python optimize_segmentation_bayesian.py deep_learning --max-evals=100
   ```

3. **Validate results:**
   ```bash
   python -m validation.runner
   ```

Results are saved in `results/` directory with comprehensive analysis and visualizations.

## Migration Notes

**Recent Updates:**
- Migrated from CSV to JSON data sources for better performance and consistency
- Replaced breakthrough papers dependency with dynamic citation-based significance calculation
- Enhanced period characterization with rich citation network analysis
- Cleaned up reference data by removing unused breakthrough paper arrays 