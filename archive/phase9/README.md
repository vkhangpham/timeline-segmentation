# Phase 9: Three-Signal Ablation Study

This folder contains the original three-signal ablation experiments conducted in Phase 8-9 of the timeline segmentation project.

## Structure

- **`experiments/`** - Python experiment scripts
  - `experiment_1_multi_source_ablation.py` - Multi-source signal contribution analysis (7 conditions)
  - `experiment_2_adaptive_penalty.py` - Adaptive penalty validation (6 conditions) 
  - `experiment_3_paradigm_filtering.py` - Paradigm significance filtering (4 conditions)

- **`results/`** - Experimental results and visualizations
  - `experiment_1/` - Raw results from multi-source ablation
  - `experiment_1_visualizations/` - Charts and graphs for signal analysis
  - `experiment_1_focused/` - Focused analysis results
  - `experiment_2/` - Raw results from penalty experiments
  - `experiment_2_visualizations/` - Charts for penalty analysis
  - `experiment_3/` - Raw results from filtering experiments
  - `experiment_3_visualizations/` - Charts for filtering analysis

- **`docs/`** - Documentation and reports
  - `Ablation_Study_Timeline_Segmentation.md` - Comprehensive 67-page ablation study report
  - `experiment_1_comprehensive_summary.md` - Multi-source signal analysis summary
  - `experiment_2_breakthrough_summary.md` - Penalty optimization findings
  - `experiment_3_breakthrough_summary.md` - Filtering effectiveness results

## Key Findings

- **Universal Subadditive Behavior**: 25.1% ± 15.5% signal reduction across all domains
- **Direction Signal Dominance**: μ=13.3 signals per domain (primary detector)
- **Penalty Insensitivity**: p=1.0000 across all penalty conditions
- **Filtering Sensitivity**: p≈0.0000 for paradigm filtering mechanisms

## Algorithm Architecture

This phase tested the original **three-signal approach**:
1. Citation disruption signals (μ=0.9)
2. Semantic shift signals (μ=2.4) 
3. Direction volatility signals (μ=13.3)

Total experiments: 88 across 8 domains (7+6+4 conditions × 8 domains each)

## Usage

Run experiments from the `experiments/` directory:

```bash
cd experiments/phase9/experiments
python experiment_1_multi_source_ablation.py
python experiment_2_adaptive_penalty.py  
python experiment_3_paradigm_filtering.py
```

Results will be saved to the `../results/` directory. 