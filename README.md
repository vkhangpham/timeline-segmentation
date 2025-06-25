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
2. **Period Characterization** → Analyzes content and themes within each detected segment

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
| `consensus_difference_metrics.py` | Quality evaluation metrics |
| `algorithm_config.py` | Centralized parameter management |
| `data_loader.py` | Multi-format data ingestion |
| `text_vectorization.py` | TF-IDF and embedding processing |

### Data Flow

```
Raw Literature Data → Change Detection → Segmentation → Period Analysis → Results
                                ↓
                         Parameter Optimization ← Validation Framework
```

## Configuration

**Centralized configuration** via `ComprehensiveAlgorithmConfig`:
- **Granularity levels:** 1 (ultra-fine) to 5 (ultra-coarse)  
- **Detection thresholds:** Direction and validation sensitivity
- **Segment constraints:** Minimum/maximum lengths
- **Optimization parameters:** Loaded from `results/optimized_parameters_bayesian.json`

## Data Structure

```
data/
├── processed/           # Domain CSV files (*_processed.csv)
└── references/          # Manual validation data (*_manual.json)

results/                 # Algorithm outputs and optimized parameters
```

## Key Features

- **Fail-fast error handling** → Immediate error propagation for debugging
- **Functional programming** → Pure functions with immutable data structures  
- **Parameter transparency** → All decisions explainable and traceable
- **Multi-domain support** → Consistent analysis across scientific fields
- **Bayesian optimization** → Efficient parameter tuning with minimal evaluations

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