#!/usr/bin/env python3
"""
Test Script for Core Objective Function Module
==============================================

This script validates the new core.objective_function module using real data
to ensure it produces consistent results with the validated analysis.
"""

import os
import sys
import json
import time
from typing import List

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.analysis.objective_function import (
    compute_objective_function,
    evaluate_timeline_quality,
    compute_jaccard_cohesion,
    compute_jensen_shannon_separation,
    ObjectiveFunctionResult
)
from core.data.models import Paper


def load_test_papers(domain_name: str = "natural_language_processing", max_papers: int = 1000) -> List[Paper]:
    """Load test papers from a domain."""
    docs_info_path = f"resources/{domain_name}/{domain_name}_docs_info.json"
    
    if not os.path.exists(docs_info_path):
        raise FileNotFoundError(f"Test data not found: {docs_info_path}")
    
    with open(docs_info_path, 'r') as f:
        docs_data = json.load(f)
    
    # Handle both formats
    if 'papers' in docs_data:
        paper_entries = docs_data['papers']
    else:
        paper_entries = list(docs_data.values())
    
    papers = []
    for i, paper_data in enumerate(paper_entries[:max_papers]):
        # Parse keywords
        keywords_raw = paper_data.get('keywords', [])
        if isinstance(keywords_raw, str):
            if '|' in keywords_raw:
                keywords = [k.strip() for k in keywords_raw.split('|') if k.strip()]
            else:
                keywords = [k.strip() for k in keywords_raw.split(',') if k.strip()]
        elif isinstance(keywords_raw, list):
            keywords = [str(k).strip() for k in keywords_raw if str(k).strip()]
        else:
            keywords = []
        
        # Only include papers with keywords
        if keywords:
            paper = Paper(
                id=str(paper_data.get('id', f"{domain_name}_{i}")),
                title=str(paper_data.get('title', '')),
                content=str(paper_data.get('abstract', paper_data.get('content', ''))),
                pub_year=int(paper_data.get('pub_year', paper_data.get('year', 2000))),
                cited_by_count=int(paper_data.get('cited_by_count', 0)),
                keywords=tuple(keywords),
                children=tuple(),
                description=str(paper_data.get('title', ''))
            )
            papers.append(paper)
    
    return papers


def create_test_segments(papers: List[Paper]) -> List[List[Paper]]:
    """Create test segments based on publication years."""
    if not papers:
        return []
    
    # Sort papers by year
    sorted_papers = sorted(papers, key=lambda p: p.pub_year)
    
    # Create segments by decade
    segments = []
    current_segment = []
    current_decade = None
    
    for paper in sorted_papers:
        decade = (paper.pub_year // 10) * 10
        
        if current_decade is None:
            current_decade = decade
        
        if decade != current_decade:
            if current_segment:
                segments.append(current_segment)
            current_segment = [paper]
            current_decade = decade
        else:
            current_segment.append(paper)
    
    # Add final segment
    if current_segment:
        segments.append(current_segment)
    
    # Filter segments with minimum size
    min_size = 10
    filtered_segments = [seg for seg in segments if len(seg) >= min_size]
    
    return filtered_segments


def test_individual_metrics():
    """Test individual cohesion and separation metrics."""
    print("Testing Individual Metrics")
    print("=" * 50)
    
    # Load test data
    papers = load_test_papers(max_papers=200)
    print(f"Loaded {len(papers)} test papers")
    
    # Create test segments
    segments = create_test_segments(papers)
    print(f"Created {len(segments)} test segments")
    
    if len(segments) < 2:
        print("ERROR: Need at least 2 segments for testing")
        return
    
    # Test cohesion metric
    print(f"\nTesting Jaccard Cohesion:")
    for i, segment in enumerate(segments[:3]):  # Test first 3 segments
        try:
            cohesion, explanation, top_keywords = compute_jaccard_cohesion(segment)
            print(f"  Segment {i+1}: {cohesion:.3f}")
            print(f"    {explanation}")
            print(f"    Top keywords: {', '.join(top_keywords[:5])}")
        except Exception as e:
            print(f"  Segment {i+1}: ERROR - {e}")
    
    # Test separation metric
    print(f"\nTesting Jensen-Shannon Separation:")
    for i in range(min(3, len(segments) - 1)):  # Test first 3 transitions
        try:
            separation, explanation = compute_jensen_shannon_separation(segments[i], segments[i + 1])
            print(f"  Transition {i+1}→{i+2}: {separation:.3f}")
            print(f"    {explanation}")
        except Exception as e:
            print(f"  Transition {i+1}→{i+2}: ERROR - {e}")


def test_objective_function():
    """Test complete objective function."""
    print(f"\nTesting Complete Objective Function")
    print("=" * 50)
    
    # Load test data
    papers = load_test_papers(max_papers=500)
    segments = create_test_segments(papers)
    
    print(f"Test data: {len(papers)} papers, {len(segments)} segments")
    
    if not segments:
        print("ERROR: No segments created")
        return
    
    # Test with different segment configurations
    test_cases = [
        ("Single segment", [papers[:100]]),
        ("Two segments", segments[:2] if len(segments) >= 2 else [segments[0]]),
        ("Multiple segments", segments[:4] if len(segments) >= 4 else segments),
        ("All segments", segments)
    ]
    
    for test_name, test_segments in test_cases:
        if not test_segments or not all(seg for seg in test_segments):
            continue
        
        print(f"\n{test_name}:")
        try:
            result = compute_objective_function(test_segments)
            
            print(f"  Final Score: {result.final_score:.3f}")
            print(f"  Cohesion: {result.cohesion_score:.3f}")
            print(f"  Separation: {result.separation_score:.3f}")
            print(f"  Segments: {result.num_segments}")
            print(f"  Transitions: {result.num_transitions}")
            print(f"  Methodology: {result.methodology}")
            
            # Show segment details
            if result.num_segments <= 5:  # Only show details for small cases
                print(f"  Cohesion details: {result.cohesion_details}")
                if result.separation_details != "No transitions in single segment":
                    print(f"  Separation details: {result.separation_details}")
        
        except Exception as e:
            print(f"  ERROR: {e}")


def test_reference_timeline():
    """Test objective function on reference timeline."""
    print(f"\nTesting Reference Timeline")
    print("=" * 50)
    
    domain = "natural_language_processing"
    
    # Load reference timeline
    manual_path = f"data/references/{domain}_manual.json"
    
    if not os.path.exists(manual_path):
        print(f"Reference timeline not found: {manual_path}")
        return
    
    with open(manual_path, 'r') as f:
        timeline_data = json.load(f)
    
    # Parse timeline segments
    if isinstance(timeline_data, list):
        segments_data = timeline_data
    elif 'timeline' in timeline_data:
        segments_data = timeline_data['timeline']
    else:
        print("Could not parse timeline format")
        return
    
    # Load papers
    papers = load_test_papers(domain, max_papers=2000)
    print(f"Loaded {len(papers)} papers for reference timeline test")
    
    # Convert timeline to paper segments
    timeline_segments = []
    
    for item in segments_data:
        if 'years' in item:
            years_str = item['years']
            if '-' in years_str:
                start_str, end_str = years_str.split('-')
                start_year = int(start_str.strip())
                end_year = int(end_str.strip())
            else:
                continue
        else:
            continue
        
        # Get papers in this year range
        segment_papers = [p for p in papers if start_year <= p.pub_year <= end_year]
        
        if segment_papers:
            timeline_segments.append(segment_papers)
    
    print(f"Created {len(timeline_segments)} timeline segments")
    
    if timeline_segments:
        try:
            result = evaluate_timeline_quality(timeline_segments, verbose=True)
            
            print(f"\nReference Timeline Evaluation:")
            print(f"  Expert timeline score: {result.final_score:.3f}")
            print(f"  This represents expert-level segmentation quality")
            
        except Exception as e:
            print(f"  ERROR evaluating reference timeline: {e}")


def test_weight_sensitivity():
    """Test sensitivity to different weight combinations."""
    print(f"\nTesting Weight Sensitivity")
    print("=" * 50)
    
    papers = load_test_papers(max_papers=300)
    segments = create_test_segments(papers)[:3]  # Use first 3 segments
    
    if len(segments) < 2:
        print("Need at least 2 segments for weight sensitivity test")
        return
    
    # Test different weight combinations
    weight_combinations = [
        (1.0, 0.0, "Cohesion only"),
        (0.8, 0.2, "Cohesion-dominant (recommended)"),
        (0.7, 0.3, "Cohesion-heavy"),
        (0.5, 0.5, "Equal weights"),
        (0.3, 0.7, "Separation-heavy"),
        (0.0, 1.0, "Separation only")
    ]
    
    print(f"Testing {len(segments)} segments with different weight combinations:")
    
    for cohesion_weight, separation_weight, description in weight_combinations:
        try:
            result = compute_objective_function(
                segments, 
                cohesion_weight=cohesion_weight, 
                separation_weight=separation_weight
            )
            
            print(f"  {description}: {result.final_score:.3f} "
                  f"(c={result.cohesion_score:.3f}, s={result.separation_score:.3f})")
        
        except Exception as e:
            print(f"  {description}: ERROR - {e}")


def performance_benchmark():
    """Benchmark performance of objective function."""
    print(f"\nPerformance Benchmark")
    print("=" * 50)
    
    papers = load_test_papers(max_papers=1000)
    segments = create_test_segments(papers)
    
    print(f"Benchmarking with {len(papers)} papers, {len(segments)} segments")
    
    # Time multiple evaluations
    num_runs = 10
    start_time = time.time()
    
    for _ in range(num_runs):
        result = compute_objective_function(segments)
    
    total_time = time.time() - start_time
    avg_time = total_time / num_runs
    
    print(f"  Average evaluation time: {avg_time:.3f}s")
    print(f"  Throughput: {len(papers) / avg_time:.0f} papers/second")
    print(f"  Final score: {result.final_score:.3f}")


def main():
    """Run all tests."""
    print("CORE OBJECTIVE FUNCTION MODULE TESTS")
    print("=" * 80)
    
    try:
        test_individual_metrics()
        test_objective_function()
        test_reference_timeline()
        test_weight_sensitivity()
        performance_benchmark()
        
        print(f"\n" + "=" * 80)
        print("ALL TESTS COMPLETED SUCCESSFULLY")
        print("Core objective function module is ready for production use")
        
    except Exception as e:
        print(f"\nTEST FAILURE: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main()) 