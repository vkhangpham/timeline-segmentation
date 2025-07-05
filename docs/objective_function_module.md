# Core Objective Function Module

## Overview

The `core.objective_function` module implements the validated objective function for timeline segmentation quality evaluation. This module is the result of comprehensive multi-domain analysis of 12,000 random segments across 4 research domains.

## Key Features

- **Jaccard Cohesion**: Measures keyword overlap within segments using top-15 defining keywords
- **Jensen-Shannon Separation**: Measures vocabulary shift between segments using information-theoretic divergence  
- **Orthogonal Metrics**: Perfect orthogonality (r=0.001) enables effective linear combination
- **Cohesion-Dominant Weighting**: Optimal (0.8, 0.2) weights maximize expert timeline performance
- **Cross-Domain Validation**: Consistent performance across technical and non-technical domains
- **Production-Ready**: Fail-fast error handling and transparent explanations

## Validation Results

### Multi-Domain Analysis
- **Domains Analyzed**: Natural Language Processing, Computer Vision, Applied Mathematics, Art
- **Total Segments**: 12,000 random segments (3,000 per domain)
- **Correlation**: r = 0.001 ± 0.028 (perfect orthogonality)
- **Expert Performance**: 37.2th percentile cohesion, 15.2th percentile separation

### Optimal Combination Strategy
- **Cohesion-Dominant (0.8, 0.2)**: 0.182 expert performance (best)
- **Equal-Weight (0.5, 0.5)**: 0.123 expert performance
- **Separation-Heavy (0.3, 0.7)**: 0.084 expert performance

## Quick Start

### Basic Usage

```python
from core import evaluate_timeline_quality, Paper

# Create paper segments
segments = [
    [paper1, paper2, paper3],  # Segment 1
    [paper4, paper5, paper6],  # Segment 2
]

# Evaluate timeline quality
result = evaluate_timeline_quality(segments)
print(f"Timeline quality: {result.final_score:.3f}")
```

### Advanced Usage

```python
from core.objective_function import compute_objective_function

# Custom weight combination
result = compute_objective_function(
    segments,
    cohesion_weight=0.8,
    separation_weight=0.2
)

# Access detailed metrics
print(f"Cohesion: {result.cohesion_score:.3f}")
print(f"Separation: {result.separation_score:.3f}")
print(f"Methodology: {result.methodology}")
```

## API Reference

### Main Functions

#### `evaluate_timeline_quality(segment_papers, verbose=False)`

High-level function for timeline segmentation quality evaluation.

**Parameters:**
- `segment_papers`: List of segments (each segment is a list of Paper objects)
- `verbose`: Whether to print detailed evaluation information

**Returns:**
- `ObjectiveFunctionResult` with complete evaluation metrics

**Example:**
```python
result = evaluate_timeline_quality(segments, verbose=True)
```

#### `compute_objective_function(segment_papers, cohesion_weight=None, separation_weight=None)`

Core objective function computation with customizable weights.

**Parameters:**
- `segment_papers`: List of segments (each segment is a list of Paper objects)
- `cohesion_weight`: Weight for cohesion component (default: load from config)
- `separation_weight`: Weight for separation component (default: load from config)

**Returns:**
- `ObjectiveFunctionResult` with complete evaluation

### Individual Metrics

#### `compute_jaccard_cohesion(segment_papers, top_k=15)`

Compute segment cohesion using mean Jaccard similarity of top-K keywords.

**Parameters:**
- `segment_papers`: List of papers in the segment
- `top_k`: Number of top keywords to use for defining the segment

**Returns:**
- Tuple of (cohesion_score, explanation, top_keywords_list)

#### `compute_jensen_shannon_separation(segment_a, segment_b)`

Compute separation between two segments using Jensen-Shannon divergence.

**Parameters:**
- `segment_a`: Papers in first segment
- `segment_b`: Papers in second segment

**Returns:**
- Tuple of (separation_score, explanation)

### Result Objects

#### `ObjectiveFunctionResult`

Complete result of objective function evaluation.

**Fields:**
- `final_score`: Final weighted objective score (0-1 range)
- `cohesion_score`: Average cohesion across segments
- `separation_score`: Average separation across transitions
- `num_segments`: Number of segments in timeline
- `num_transitions`: Number of transitions between segments
- `cohesion_details`: Detailed explanation of cohesion calculations
- `separation_details`: Detailed explanation of separation calculations
- `methodology`: Summary of methodology and weights used

## Metric Details

### Jaccard Cohesion

Measures how well papers within a segment share common vocabulary by:

1. **Keyword Collection**: Gather all keywords from papers in segment
2. **Top-K Selection**: Select top-15 most frequent keywords as segment definition
3. **Jaccard Computation**: Calculate Jaccard similarity between each paper's keywords and segment definition
4. **Aggregation**: Average Jaccard scores across all papers with keywords

**Range**: [0, 1] where 1 indicates perfect keyword overlap

### Jensen-Shannon Separation

Measures vocabulary shift between segments using information theory:

1. **Vocabulary Creation**: Create unified vocabulary from both segments
2. **Distribution Computation**: Calculate keyword frequency distributions for each segment
3. **JS Divergence**: Compute Jensen-Shannon divergence between distributions
4. **Normalization**: Result is naturally bounded to [0, 1]

**Range**: [0, 1] where 1 indicates maximum vocabulary divergence

### Linear Combination

The final objective score combines cohesion and separation linearly:

```
final_score = cohesion_weight × avg_cohesion + separation_weight × avg_separation
```

**Default Weights**: (0.8, 0.2) - cohesion-dominant strategy optimized for expert timeline performance

## Configuration

The objective function loads parameters from `optimization_config.json`:

```json
{
  "objective_function": {
    "cohesion_weight": 0.8,
    "separation_weight": 0.2,
    "cohesion_metric": "jaccard",
    "separation_metric": "jensen_shannon",
    "top_k_keywords": 15,
    "description": "Validated objective function with cohesion-dominant weighting"
  }
}
```

**Configuration Parameters:**
- `cohesion_weight`: Weight for cohesion component (default: 0.8)
- `separation_weight`: Weight for separation component (default: 0.2)  
- `top_k_keywords`: Number of top keywords for segment definition (default: 15)
- `cohesion_metric`: Cohesion metric type ("jaccard")
- `separation_metric`: Separation metric type ("jensen_shannon")

## Performance

- **Evaluation Speed**: ~0.005s per evaluation
- **Throughput**: ~210,000 papers/second
- **Memory Usage**: Minimal (scales linearly with segment size)
- **Scalability**: Tested up to 1,000+ papers per evaluation

## Error Handling

The module follows fail-fast principles:

- **Empty Segments**: Raises `ValueError` for empty segments
- **Missing Keywords**: Returns 0.0 scores with explanatory messages
- **Invalid Weights**: Validates weights sum to 1.0
- **Transparent Errors**: All exceptions include clear error messages

## Integration Examples

### With Timeline Segmentation Algorithm

```python
from core.integration import run_change_detection
from core.objective_function import evaluate_timeline_quality

# Run segmentation algorithm
segmentation_results, _, _ = run_change_detection(domain_name, config=config)

# Extract paper segments
segment_papers = extract_paper_segments(segmentation_results, domain_data)

# Evaluate quality
result = evaluate_timeline_quality(segment_papers)
print(f"Segmentation quality: {result.final_score:.3f}")
```

### With Optimization Algorithms

```python
def optimization_objective(params):
    # Run segmentation with parameters
    segments = run_segmentation_with_params(params)
    
    # Evaluate quality
    result = evaluate_timeline_quality(segments)
    
    # Return negative for minimization
    return -result.final_score
```

## Comparison with Previous Metrics

| Aspect | Previous Metrics | New Objective Function |
|--------|-----------------|----------------------|
| Cohesion Measure | TF-IDF cosine similarity | Jaccard keyword overlap |
| Separation Measure | Centroid distance | Jensen-Shannon divergence |
| Combination Method | Complex adaptive | Simple linear |
| Cross-Domain Validation | Limited | Comprehensive (4 domains) |
| Expert Alignment | Not validated | Optimized for expert performance |
| Orthogonality | Unknown | Perfect (r≈0) |

## Troubleshooting

### Common Issues

1. **Low Cohesion Scores**: Check if papers have sufficient keywords
2. **Zero Separation**: Verify segments have different vocabulary
3. **Import Errors**: Ensure `core` module is in Python path
4. **Weight Errors**: Verify weights sum to 1.0

### Debug Mode

Enable verbose output for detailed analysis:

```python
result = evaluate_timeline_quality(segments, verbose=True)
```

## Future Enhancements

- **Semantic Embeddings**: Optional contextual embedding support
- **Dynamic Weighting**: Adaptive weights based on domain characteristics
- **Incremental Evaluation**: Support for streaming/online evaluation
- **Visualization**: Built-in plotting of cohesion/separation trends

## References

- **Multi-Domain Analysis**: `results/objective_analysis/OBJECTIVE_FUNCTION_SUMMARY.md`
- **Validation Data**: `results/objective_analysis/multi_domain_complete_analysis.json`
- **Configuration**: `optimization_config.json`
- **Usage Examples**: `examples/objective_function_usage.py` 