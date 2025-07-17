# Timeline Segmentation Configuration Guide

This document explains how to configure the timeline segmentation system and optimize its parameters.

## Configuration Files

### Main Configuration: `config/config.yaml`
Contains all algorithm parameters organized by module:
- **Segmentation Module**: Change detection, citation analysis, beam search
- **Optimization Module**: Objective function, penalty system
- **Data Processing Module**: Ubiquitous filtering  
- **System Module**: Diagnostics and logging

### Optimization Configuration: `config/optimization.yaml`
Contains **optimization-specific parameters only**:
- **Parameter Search Space**: Ranges for Bayesian optimization
- **Search Strategy**: Optimization algorithm settings
- **Scoring**: Failure scores and validation tolerances
- **Execution**: Parallel processing and caching

## 🔄 Unified Configuration System

**Important**: The optimization system automatically loads penalty and objective function parameters from the main `config.yaml` file. This ensures **perfect synchronization** between the main algorithm and optimization process - no parameter duplication or inconsistencies.

### How It Works
1. **Main Algorithm**: Loads all parameters from `config.yaml`
2. **Optimization Process**: 
   - Loads optimization-specific settings from `optimization.yaml`
   - Loads penalty/objective parameters from `config.yaml` 
   - Ensures both systems use identical penalty and scoring logic

### Benefits
- ✅ **No Duplication**: Parameters defined in one place only
- ✅ **Perfect Consistency**: Main algorithm and optimization use identical settings
- ✅ **Easy Maintenance**: Change parameters in one file, affects both systems
- ✅ **Clear Separation**: Optimization config focuses only on optimization settings

## Configuration Structure

### Segmentation Module

#### Change Detection (`segmentation.change_detection`)
Controls paradigm shift detection in academic timelines.

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `direction_change_threshold` | float | 0.1 | [0.0, 1.0] | Threshold for detecting direction changes |
| `direction_threshold_strategy` | str | "global_p90" | ["fixed", "global_p90", "global_p95", "global_p99"] | Threshold calculation strategy |
| `direction_scoring_method` | str | "weighted_jaccard" | ["weighted_jaccard", "jensen_shannon"] | Scoring method for direction changes |
| `min_baseline_period_years` | int | 3 | [1, 10] | Minimum years for baseline period |
| `score_distribution_window_years` | int | 3 | [1, 10] | Window for score distribution analysis |
| `min_papers_per_year` | int | 100 | [1, 1000] | Minimum papers per year for analysis |

**Tuning Guidelines:**
- **Lower `direction_change_threshold`**: More sensitive, detects smaller changes
- **Higher `direction_change_threshold`**: Less sensitive, detects only major changes
- **Adaptive strategies** (`global_p90`, `global_p95`, `global_p99`): Automatically adjust threshold based on data distribution
- **Fixed strategy**: Uses the exact `direction_change_threshold` value

#### Citation Analysis (`segmentation.citation_analysis`)
Validates direction changes using citation patterns.

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `citation_confidence_boost` | float | 0.1 | [0.0, 1.0] | Boost factor for citation support |
| `citation_support_window_years` | int | 2 | [1, 10] | Years around change point to check citations |

**Tuning Guidelines:**
- **Higher `citation_confidence_boost`**: Stronger influence of citation patterns
- **Wider `citation_support_window_years`**: More lenient citation validation

#### Beam Search Refinement (`segmentation.beam_refinement`)
Optimizes segment boundaries through local search.

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `enabled` | bool | true | [true, false] | Enable beam search refinement |
| `beam_width` | int | 5 | [1, 50] | Number of states to explore |
| `max_splits_per_segment` | int | 1 | [0, 10] | Maximum splits per segment |

**Tuning Guidelines:**
- **Higher `beam_width`**: More thorough exploration, slower execution
- **More `max_splits_per_segment`**: More aggressive refinement
- **Constraints are enforced through penalty system**: See penalty configuration for period length control

### Optimization Module

#### Objective Function (`optimization.objective_function`)
Controls how segmentation quality is measured.

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `cohesion_weight` | float | 0.8 | [0.0, 1.0] | Weight for period cohesion |
| `separation_weight` | float | 0.2 | [0.0, 1.0] | Weight for period separation |
| `top_k_keywords` | int | 100 | [1, 100] | Keywords for cohesion calculation |
| `min_keyword_frequency_ratio` | float | 0.05 | [0.0, 1.0] | Minimum keyword frequency |

**Constraints:**
- `cohesion_weight + separation_weight = 1.0`

**Tuning Guidelines:**
- **Higher `cohesion_weight`**: Emphasizes internal period consistency
- **Higher `separation_weight`**: Emphasizes distinctness between periods
- **Higher `top_k_keywords`**: More vocabulary for analysis
- **Higher `min_keyword_frequency_ratio`**: Stricter keyword filtering

#### Penalty System (`optimization.penalty`)
Prevents pathological segmentations through penalty functions.

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `min_period_years` | int | 3 | [1, 20] | Minimum acceptable period length |
| `max_period_years` | int | 14 | [5, 200] | Maximum acceptable period length |
| `auto_n_upper` | bool | true | [true, false] | Auto-calculate segment upper bound |
| `n_upper_buffer` | int | 1 | [0, 10] | Buffer for auto-calculated upper bound |
| `lambda_short` | float | 0.05 | [0.0, 1.0] | Weight for short period penalty |
| `lambda_long` | float | 0.03 | [0.0, 1.0] | Weight for long period penalty |
| `lambda_count` | float | 0.02 | [0.0, 1.0] | Weight for over-segmentation penalty |
| `enable_scaling` | bool | true | [true, false] | Enable 0-1 score scaling |
| `scaling_factor` | float | 2.0 | [0.1, 10.0] | Sigmoid scaling factor |

**Tuning Guidelines:**
- **Stricter length constraints**: Lower variance in period lengths
- **Higher lambda values**: Stronger penalty enforcement
- **Auto upper bound**: Adapts to timeline length
- **Scaling**: Normalizes scores to [0, 1] range

### Data Processing Module

#### Ubiquitous Filtering (`data_processing.ubiquitous_filtering`)
Removes overly common keywords that don't discriminate between periods.

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `apply_ubiquitous_filtering` | bool | true | [true, false] | Enable ubiquitous filtering |
| `ubiquity_threshold` | float | 0.8 | [0.0, 1.0] | Threshold for ubiquitous keywords |
| `max_iterations` | int | 10 | [1, 50] | Maximum filtering iterations |
| `min_replacement_frequency` | int | 2 | [1, 100] | Minimum frequency for replacements |

**Tuning Guidelines:**
- **Lower `ubiquity_threshold`**: More aggressive filtering
- **Higher `max_iterations`**: More thorough filtering
- **Higher `min_replacement_frequency`**: Stricter replacement criteria

### System Module

#### Diagnostics (`system.diagnostics`)
Controls diagnostic output and logging.

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `save_direction_diagnostics` | bool | true | [true, false] | Save direction change diagnostics |
| `diagnostic_top_keywords_limit` | int | 10 | [1, 50] | Top keywords in diagnostics |

## Optimization Configuration

### Parameter Search Space (`parameters`)
Defines which parameters to optimize and their ranges.

#### Currently Optimized Parameters
- `direction_change_threshold`: [0.6, 0.95]
- `citation_confidence_boost`: [0.05, 0.2]
- `min_keyword_frequency_ratio`: [0.05, 0.15]
- `top_k_keywords`: [10, 30]

#### Adding New Parameters
To optimize additional parameters, add them to `optimization.yaml`:

```yaml
parameters:
  new_parameter_name:
    type: float  # or int
    range: [min_value, max_value]
    description: "Parameter description"
```

**Note**: The parameter must exist in the main `config.yaml` structure.

### Search Strategy (`search`)
Controls the optimization algorithm behavior.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_calls` | int | 100 | Total optimization trials |
| `n_initial_points` | int | 20 | Random exploration trials |
| `strategy` | str | "bayesian" | Optimization strategy |
| `acquisition_function` | str | "EI" | Acquisition function |

**Acquisition Functions:**
- **EI (Expected Improvement)**: Balanced exploration/exploitation
- **PI (Probability of Improvement)**: More conservative
- **LCB (Lower Confidence Bound)**: More exploitative

### Scoring Configuration (`scoring`)
Controls optimization-specific scoring parameters.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `fail_score` | float | -10.0 | Score for failed trials |
| `boundary_tolerance` | int | 2 | Boundary matching tolerance |
| `max_segments_per_match` | int | 3 | Max segments per validation match |

### Execution Configuration (`execution`)
Controls optimization execution parameters.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_workers` | int | 8 | Parallel workers |
| `cache_academic_years` | bool | true | Cache data between trials |

## Common Configuration Tasks

### 1. Increasing Sensitivity
To detect more subtle paradigm shifts:
- Decrease `direction_change_threshold` to 0.05
- Use `direction_threshold_strategy: "global_p90"`
- Increase `citation_confidence_boost` to 0.15

### 2. Reducing Over-Segmentation
To get fewer, more stable segments:
- Increase `direction_change_threshold` to 0.15
- Increase `lambda_count` to 0.05

### 3. Improving Performance
To speed up processing:
- Set `beam_refinement.enabled: false`
- Reduce `top_k_keywords` to 50
- Increase `min_keyword_frequency_ratio` to 0.1

### 4. Domain-Specific Tuning
For different academic domains:
- **Fast-moving fields** (CS, ML): Lower thresholds, shorter periods
- **Slow-moving fields** (History, Literature): Higher thresholds, longer periods
- **Emerging fields**: Lower `min_papers_per_year`, higher sensitivity

## Validation and Testing

### Parameter Validation
The system validates all parameters on startup:
- Range checks for numeric values
- Valid options for categorical parameters
- Consistency checks (e.g., weights sum to 1.0)

### Testing Configuration Changes
1. **Single Domain Test**: Test on one domain first
2. **Validation Metrics**: Check boundary-F1 and segment-F1 scores
3. **Visual Inspection**: Review generated timelines
4. **Performance Impact**: Monitor execution time

### Best Practices
1. **Incremental Changes**: Modify one parameter at a time
2. **Document Changes**: Record parameter changes and rationale
3. **Version Control**: Commit configurations with meaningful messages
4. **Backup**: Keep working configurations before major changes

## Troubleshooting

### Common Issues

#### No Boundaries Detected
- **Cause**: Threshold too high
- **Solution**: Lower `direction_change_threshold` or use adaptive strategy

#### Too Many Segments
- **Cause**: Threshold too low or weak penalties
- **Solution**: Increase `direction_change_threshold` or penalty weights

#### Validation Errors
- **Cause**: Invalid parameter values
- **Solution**: Check parameter ranges and constraints

#### Poor Performance
- **Cause**: Complex configuration or large datasets
- **Solution**: Simplify configuration or optimize data processing

#### Optimization Convergence Issues
- **Cause**: Narrow parameter ranges or insufficient trials
- **Solution**: Widen ranges or increase `n_calls`

#### Configuration Synchronization Issues
- **Cause**: Manually editing penalty parameters in multiple places
- **Solution**: Only edit parameters in `config.yaml` - optimization automatically syncs

### Debug Mode
Enable verbose logging for detailed information:
```bash
python scripts/run_timeline_analysis.py --domain art --verbose
python scripts/run_optimization.py --domain art --verbose
```

## Configuration Examples

### Conservative Configuration
For stable, well-established domains:
```yaml
segmentation:
  change_detection:
    direction_change_threshold: 0.15
    direction_threshold_strategy: "global_p95"
  
optimization:
  penalty:
    min_period_years: 5
    max_period_years: 20
    lambda_count: 0.05
```

### Aggressive Configuration
For rapidly evolving domains:
```yaml
segmentation:
  change_detection:
    direction_change_threshold: 0.05
    direction_threshold_strategy: "global_p90"
    
optimization:
  penalty:
    min_period_years: 2
    max_period_years: 10
    lambda_count: 0.01
```

### High-Performance Configuration
For large-scale analysis:
```yaml
segmentation:
  beam_refinement:
    enabled: false
    
optimization:
  objective_function:
    top_k_keywords: 50
    min_keyword_frequency_ratio: 0.1
    
execution:
  max_workers: 16
  cache_academic_years: true
```

## Configuration Workflow

### Making Changes
1. **Edit `config.yaml`**: Modify algorithm parameters
2. **Edit `optimization.yaml`**: Modify optimization settings (if needed)
3. **Test Changes**: Run single domain test
4. **Validate**: Check that both main algorithm and optimization work
5. **Document**: Update version control with clear commit message

### Optimization Workflow
1. **Set Parameter Ranges**: Define in `optimization.yaml`
2. **Configure Search**: Set trials, workers, etc.
3. **Run Optimization**: `python scripts/run_optimization.py --domain art`
4. **Apply Results**: Copy best parameters to `config.yaml`
5. **Validate**: Test main algorithm with optimized parameters 