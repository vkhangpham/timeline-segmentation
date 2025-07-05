#!/usr/bin/env python3
"""
Objective Function Analysis for Timeline Segmentation
====================================================

This script analyzes component score distributions and designs the optimal
combination strategy for cohesion and separation metrics.

Goals:
1. Pick one cohesion and one separation metric based on performance
2. Generate 3000 random segments to understand score distributions
3. Analyze correlation between cohesion and separation (ideally orthogonal)
4. Design combination strategy where expert timelines score high cohesion, moderate-high separation

Selected Metrics (based on component analysis):
- Cohesion: Mean Jaccard similarity (most interpretable, good performance)
- Separation: Jensen-Shannon divergence (domain-robust, measures vocabulary shifts)
"""

import os
import sys
import json
import time
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any
from scipy.stats import pearsonr, spearmanr
from scipy import stats
from collections import defaultdict

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.data.models import Paper


def load_domain_papers(domain_name: str) -> List[Paper]:
    """Load papers for a domain from JSON data sources."""
    docs_info_path = f"resources/{domain_name}/{domain_name}_docs_info.json"
    
    if not os.path.exists(docs_info_path):
        raise FileNotFoundError(f"Domain data not found: {docs_info_path}")
    
    with open(docs_info_path, 'r') as f:
        docs_data = json.load(f)
    
    # Handle both formats: {"papers": [...]} or {url: paper_data, ...}
    if 'papers' in docs_data:
        paper_entries = docs_data['papers']
    else:
        paper_entries = docs_data.values()
    
    papers = []
    for paper_data in paper_entries:
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
        
        paper = Paper(
            id=str(paper_data.get('id', f"{domain_name}_{len(papers)}")),
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


def filter_keywords(papers: List[Paper], min_years: int = 2, min_paper_ratio: float = 0.01) -> Tuple[List[Paper], List[str]]:
    """Filter keywords based on temporal span and frequency."""
    keyword_years = defaultdict(set)
    keyword_papers = defaultdict(int)
    total_papers = len(papers)
    
    for paper in papers:
        for keyword in paper.keywords:
            keyword_years[keyword].add(paper.pub_year)
            keyword_papers[keyword] += 1
    
    # Filter keywords
    valid_keywords = []
    for keyword in keyword_years:
        years_seen = len(keyword_years[keyword])
        paper_count = keyword_papers[keyword]
        paper_ratio = paper_count / total_papers
        
        if years_seen >= min_years and paper_ratio >= min_paper_ratio:
            valid_keywords.append(keyword)
    
    # Filter papers to only include those with valid keywords
    filtered_papers = []
    for paper in papers:
        valid_paper_keywords = [kw for kw in paper.keywords if kw in valid_keywords]
        if valid_paper_keywords:
            filtered_paper = Paper(
                id=paper.id,
                title=paper.title,
                content=paper.content,
                pub_year=paper.pub_year,
                cited_by_count=paper.cited_by_count,
                keywords=tuple(valid_paper_keywords),
                children=paper.children,
                description=paper.description
            )
            filtered_papers.append(filtered_paper)
    
    return filtered_papers, valid_keywords


def cohesion_jaccard(segment_papers: List[Paper]) -> float:
    """
    Compute cohesion using mean Jaccard similarity of top-K keywords.
    
    Selected as primary cohesion metric based on:
    - High interpretability (keyword overlap)
    - Good performance across domains (34-67th percentiles)
    - Complementary to entropy (strong negative correlation)
    """
    if len(segment_papers) < 2:
        return 0.0
    
    # Get top-15 keywords by frequency
    keyword_counts = defaultdict(int)
    for paper in segment_papers:
        for keyword in paper.keywords:
            keyword_counts[keyword] += 1
    
    if not keyword_counts:
        return 0.0
    
    # Select top-K keywords
    top_k = 15
    top_keywords = sorted(keyword_counts.items(), key=lambda x: x[1], reverse=True)[:top_k]
    defining_keywords = {kw for kw, count in top_keywords}
    
    if not defining_keywords:
        return 0.0
    
    # Compute mean Jaccard similarity for papers with defining keywords
    jaccard_scores = []
    for paper in segment_papers:
        paper_keywords = set(paper.keywords)
        if paper_keywords & defining_keywords:  # Paper has at least one defining keyword
            jaccard = len(paper_keywords & defining_keywords) / len(paper_keywords | defining_keywords)
            jaccard_scores.append(jaccard)
    
    return np.mean(jaccard_scores) if jaccard_scores else 0.0


def separation_jensen_shannon(segment_a: List[Paper], segment_b: List[Paper]) -> float:
    """
    Compute separation using Jensen-Shannon divergence between keyword distributions.
    
    Selected as primary separation metric based on:
    - Domain-robust performance (42-79th percentiles)
    - Measures vocabulary shifts effectively
    - Theoretically grounded (information theory)
    """
    if not segment_a or not segment_b:
        return 0.0
    
    # Collect keywords from both segments
    keywords_a = []
    keywords_b = []
    
    for paper in segment_a:
        keywords_a.extend(paper.keywords)
    
    for paper in segment_b:
        keywords_b.extend(paper.keywords)
    
    if not keywords_a or not keywords_b:
        return 0.0
    
    # Create vocabulary
    vocab = list(set(keywords_a) | set(keywords_b))
    if not vocab:
        return 0.0
    
    # Compute frequency distributions
    def get_freq_dist(keywords, vocab):
        counts = defaultdict(int)
        for kw in keywords:
            counts[kw] += 1
        
        total = sum(counts.values())
        if total == 0:
            return np.ones(len(vocab)) / len(vocab)  # Uniform distribution
        
        return np.array([counts[kw] / total for kw in vocab])
    
    p = get_freq_dist(keywords_a, vocab)
    q = get_freq_dist(keywords_b, vocab)
    
    # Jensen-Shannon divergence
    m = 0.5 * (p + q)
    
    def kl_divergence(x, y):
        # Add small epsilon to avoid log(0)
        epsilon = 1e-10
        x = x + epsilon
        y = y + epsilon
        return np.sum(x * np.log(x / y))
    
    js_div = 0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m)
    
    # Convert to 0-1 range (JS divergence is bounded by log(2))
    return js_div / np.log(2)


def generate_random_segments(papers: List[Paper], num_segments: int = 3000, 
                           min_span: int = 3, max_span: int = 50, 
                           min_papers: int = 10) -> List[List[Paper]]:
    """
    Generate random segments using the specified strategy.
    
    Strategy:
    1. Choose random start year
    2. Choose random span (3-50 years)
    3. If segment doesn't qualify (too few papers), increase span by 1 year
    4. Repeat until qualified or max span reached
    """
    if not papers:
        return []
    
    year_range = (min(p.pub_year for p in papers), max(p.pub_year for p in papers))
    min_year, max_year = year_range
    
    print(f"Generating {num_segments} random segments...")
    print(f"Year range: {min_year}-{max_year}")
    print(f"Span range: {min_span}-{max_span} years")
    print(f"Minimum papers per segment: {min_papers}")
    
    random_segments = []
    attempts = 0
    max_attempts = num_segments * 10  # Prevent infinite loops
    
    while len(random_segments) < num_segments and attempts < max_attempts:
        attempts += 1
        
        # 1. Choose random start year
        start_year = random.randint(min_year, max_year - min_span)
        
        # 2. Choose random initial span
        initial_span = random.randint(min_span, min(max_span, max_year - start_year))
        
        # 3. Try increasing spans until qualified
        for span in range(initial_span, min(max_span + 1, max_year - start_year + 1)):
            end_year = start_year + span
            
            # Get papers in this year range
            segment_papers = [p for p in papers if start_year <= p.pub_year <= end_year]
            
            # Check if segment qualifies
            if len(segment_papers) >= min_papers:
                # Additional check: ensure segment has keywords
                total_keywords = sum(len(p.keywords) for p in segment_papers)
                if total_keywords > 0:
                    random_segments.append(segment_papers)
                    break
        
        # Progress reporting
        if len(random_segments) % 500 == 0:
            print(f"  Generated {len(random_segments)}/{num_segments} segments (attempts: {attempts})")
    
    print(f"Successfully generated {len(random_segments)} random segments in {attempts} attempts")
    return random_segments


def load_reference_timelines(domain_name: str) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
    """Load reference timelines for comparison."""
    manual_path = f"data/references/{domain_name}_manual.json"
    gemini_path = f"data/references/{domain_name}_gemini.json"
    
    def parse_timeline(filepath: str) -> List[Tuple[int, int]]:
        if not os.path.exists(filepath):
            return [(2000, 2023)]
        
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        segments = []
        
        # Handle different JSON formats
        if isinstance(data, list):
            timeline_data = data
        elif 'timeline' in data:
            timeline_data = data['timeline']
        elif 'segments' in data:
            timeline_data = data['segments']
        elif 'historical_periods' in data:
            timeline_data = data['historical_periods']
        else:
            return [(2000, 2023)]
        
        for item in timeline_data:
            if isinstance(item, dict):
                if 'years' in item:
                    years_str = item['years']
                    if '-' in years_str:
                        try:
                            start_str, end_str = years_str.split('-')
                            start = int(start_str.strip())
                            end = int(end_str.strip())
                        except ValueError:
                            continue
                    else:
                        continue
                elif 'start_year' in item and 'end_year' in item:
                    start = item['start_year']
                    end = item['end_year']
                else:
                    start = item.get('start_year', item.get('year_start', 2000))
                    end = item.get('end_year', item.get('year_end', 2023))
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                start, end = item[0], item[1]
            else:
                continue
            
            segments.append((int(start), int(end)))
        
        segments.sort()
        return segments if segments else [(2000, 2023)]
    
    manual_segments = parse_timeline(manual_path)
    gemini_segments = parse_timeline(gemini_path)
    
    return manual_segments, gemini_segments


def papers_to_segments(papers: List[Paper], segment_years: List[Tuple[int, int]]) -> List[List[Paper]]:
    """Convert year-based segments to paper-based segments."""
    segments = []
    for start_year, end_year in segment_years:
        segment_papers = [p for p in papers if start_year <= p.pub_year <= end_year]
        if segment_papers:  # Only include non-empty segments
            segments.append(segment_papers)
    return segments


def analyze_objective_function(domain_name: str) -> Dict[str, Any]:
    """
    Comprehensive analysis of objective function components.
    
    Returns:
        Analysis results including distributions, correlations, and combination strategies
    """
    print(f"\n{'='*80}")
    print(f"OBJECTIVE FUNCTION ANALYSIS: {domain_name.upper()}")
    print(f"{'='*80}")
    
    start_time = time.time()
    
    # 1. Load and filter data
    print("\n1. Loading and filtering data...")
    papers = load_domain_papers(domain_name)
    filtered_papers, valid_keywords = filter_keywords(papers, min_years=2, min_paper_ratio=0.01)
    
    print(f"   Papers: {len(papers)} → {len(filtered_papers)} (filtered)")
    print(f"   Keywords: {len(valid_keywords)} valid")
    print(f"   Year range: {min(p.pub_year for p in filtered_papers)}-{max(p.pub_year for p in filtered_papers)}")
    
    # 2. Generate random segments
    print("\n2. Generating random segments...")
    random_segments = generate_random_segments(filtered_papers, num_segments=3000)
    
    if len(random_segments) < 1000:
        print(f"   WARNING: Only generated {len(random_segments)} segments (target: 3000)")
    
    # 3. Compute cohesion scores for random segments
    print("\n3. Computing cohesion scores...")
    cohesion_scores = []
    
    for i, segment in enumerate(random_segments):
        score = cohesion_jaccard(segment)
        cohesion_scores.append(score)
        
        if (i + 1) % 1000 == 0:
            print(f"   Processed {i + 1}/{len(random_segments)} segments")
    
    cohesion_scores = np.array(cohesion_scores)
    
    # 4. Compute separation scores for random segment pairs
    print("\n4. Computing separation scores...")
    separation_scores = []
    
    # Generate pairs for separation analysis
    num_pairs = min(3000, len(random_segments) // 2)
    random.shuffle(random_segments)
    
    for i in range(0, num_pairs * 2, 2):
        if i + 1 < len(random_segments):
            score = separation_jensen_shannon(random_segments[i], random_segments[i + 1])
            separation_scores.append(score)
        
        if (i // 2 + 1) % 1000 == 0:
            print(f"   Processed {i // 2 + 1}/{num_pairs} pairs")
    
    separation_scores = np.array(separation_scores)
    
    # 5. Load and analyze reference timelines
    print("\n5. Analyzing reference timelines...")
    manual_timeline, gemini_timeline = load_reference_timelines(domain_name)
    
    manual_segments = papers_to_segments(filtered_papers, manual_timeline)
    gemini_segments = papers_to_segments(filtered_papers, gemini_timeline)
    
    # Filter segments with minimum papers
    min_papers = 50
    manual_segments = [seg for seg in manual_segments if len(seg) >= min_papers]
    gemini_segments = [seg for seg in gemini_segments if len(seg) >= min_papers]
    
    # Compute reference timeline scores
    reference_results = {}
    
    for timeline_name, segments in [("manual", manual_segments), ("gemini", gemini_segments)]:
        if not segments:
            continue
        
        # Cohesion scores
        timeline_cohesion = [cohesion_jaccard(seg) for seg in segments]
        
        # Separation scores (between consecutive segments)
        timeline_separation = []
        for i in range(len(segments) - 1):
            sep_score = separation_jensen_shannon(segments[i], segments[i + 1])
            timeline_separation.append(sep_score)
        
        reference_results[timeline_name] = {
            "cohesion_scores": timeline_cohesion,
            "separation_scores": timeline_separation,
            "num_segments": len(segments),
            "avg_cohesion": np.mean(timeline_cohesion),
            "avg_separation": np.mean(timeline_separation) if timeline_separation else 0.0
        }
    
    # 6. Statistical analysis
    print("\n6. Performing statistical analysis...")
    
    # Distribution statistics
    cohesion_stats = {
        "mean": np.mean(cohesion_scores),
        "std": np.std(cohesion_scores),
        "median": np.median(cohesion_scores),
        "min": np.min(cohesion_scores),
        "max": np.max(cohesion_scores),
        "q25": np.percentile(cohesion_scores, 25),
        "q75": np.percentile(cohesion_scores, 75)
    }
    
    separation_stats = {
        "mean": np.mean(separation_scores),
        "std": np.std(separation_scores),
        "median": np.median(separation_scores),
        "min": np.min(separation_scores),
        "max": np.max(separation_scores),
        "q25": np.percentile(separation_scores, 25),
        "q75": np.percentile(separation_scores, 75)
    }
    
    # Correlation analysis (using equal-sized samples)
    min_size = min(len(cohesion_scores), len(separation_scores))
    cohesion_sample = cohesion_scores[:min_size]
    separation_sample = separation_scores[:min_size]
    
    correlation_pearson, p_value_pearson = pearsonr(cohesion_sample, separation_sample)
    correlation_spearman, p_value_spearman = spearmanr(cohesion_sample, separation_sample)
    
    print(f"   Cohesion distribution: μ={cohesion_stats['mean']:.3f}, σ={cohesion_stats['std']:.3f}")
    print(f"   Separation distribution: μ={separation_stats['mean']:.3f}, σ={separation_stats['std']:.3f}")
    print(f"   Pearson correlation: r={correlation_pearson:.3f} (p={p_value_pearson:.3f})")
    print(f"   Spearman correlation: ρ={correlation_spearman:.3f} (p={p_value_spearman:.3f})")
    
    analysis_time = time.time() - start_time
    print(f"\n   Analysis completed in {analysis_time:.1f}s")
    
    return {
        "domain": domain_name,
        "analysis_time": analysis_time,
        "data_stats": {
            "total_papers": len(papers),
            "filtered_papers": len(filtered_papers),
            "valid_keywords": len(valid_keywords),
            "random_segments": len(random_segments),
            "separation_pairs": len(separation_scores)
        },
        "distributions": {
            "cohesion_scores": cohesion_scores.tolist(),
            "separation_scores": separation_scores.tolist(),
            "cohesion_stats": cohesion_stats,
            "separation_stats": separation_stats
        },
        "correlation": {
            "pearson": {"r": correlation_pearson, "p_value": p_value_pearson},
            "spearman": {"rho": correlation_spearman, "p_value": p_value_spearman}
        },
        "reference_timelines": reference_results
    }


def plot_distributions(results: Dict[str, Any], save_dir: str = "results/objective_analysis"):
    """Create comprehensive plots of score distributions and correlations."""
    domain = results["domain"]
    os.makedirs(save_dir, exist_ok=True)
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'Objective Function Analysis: {domain.upper()}', fontsize=16, fontweight='bold')
    
    cohesion_scores = np.array(results["distributions"]["cohesion_scores"])
    separation_scores = np.array(results["distributions"]["separation_scores"])
    
    # 1. Cohesion distribution
    ax1 = axes[0, 0]
    ax1.hist(cohesion_scores, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    ax1.axvline(np.mean(cohesion_scores), color='red', linestyle='--', 
                label=f'Mean: {np.mean(cohesion_scores):.3f}')
    ax1.axvline(np.median(cohesion_scores), color='orange', linestyle='--', 
                label=f'Median: {np.median(cohesion_scores):.3f}')
    ax1.set_xlabel('Cohesion Score (Jaccard)')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Cohesion Score Distribution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Separation distribution
    ax2 = axes[0, 1]
    ax2.hist(separation_scores, bins=50, alpha=0.7, color='lightcoral', edgecolor='black')
    ax2.axvline(np.mean(separation_scores), color='red', linestyle='--', 
                label=f'Mean: {np.mean(separation_scores):.3f}')
    ax2.axvline(np.median(separation_scores), color='orange', linestyle='--', 
                label=f'Median: {np.median(separation_scores):.3f}')
    ax2.set_xlabel('Separation Score (Jensen-Shannon)')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Separation Score Distribution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Correlation scatter plot
    ax3 = axes[0, 2]
    min_size = min(len(cohesion_scores), len(separation_scores))
    cohesion_sample = cohesion_scores[:min_size]
    separation_sample = separation_scores[:min_size]
    
    ax3.scatter(cohesion_sample, separation_sample, alpha=0.5, s=10)
    
    # Add correlation line
    z = np.polyfit(cohesion_sample, separation_sample, 1)
    p = np.poly1d(z)
    ax3.plot(cohesion_sample, p(cohesion_sample), "r--", alpha=0.8)
    
    correlation = results["correlation"]["pearson"]["r"]
    ax3.set_xlabel('Cohesion Score')
    ax3.set_ylabel('Separation Score')
    ax3.set_title(f'Cohesion vs Separation\n(r = {correlation:.3f})')
    ax3.grid(True, alpha=0.3)
    
    # 4. Reference timeline comparison - Cohesion
    ax4 = axes[1, 0]
    ref_results = results["reference_timelines"]
    
    # Plot random distribution as background
    ax4.hist(cohesion_scores, bins=30, alpha=0.3, color='gray', 
             label=f'Random (n={len(cohesion_scores)})', density=True)
    
    # Plot reference timelines
    colors = ['red', 'blue']
    for i, (timeline_name, timeline_data) in enumerate(ref_results.items()):
        if timeline_data["cohesion_scores"]:
            ax4.axvline(timeline_data["avg_cohesion"], color=colors[i], 
                       linestyle='-', linewidth=3, 
                       label=f'{timeline_name.capitalize()}: {timeline_data["avg_cohesion"]:.3f}')
    
    ax4.set_xlabel('Cohesion Score')
    ax4.set_ylabel('Density')
    ax4.set_title('Reference Timeline Cohesion')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. Reference timeline comparison - Separation
    ax5 = axes[1, 1]
    
    # Plot random distribution as background
    ax5.hist(separation_scores, bins=30, alpha=0.3, color='gray', 
             label=f'Random (n={len(separation_scores)})', density=True)
    
    # Plot reference timelines
    for i, (timeline_name, timeline_data) in enumerate(ref_results.items()):
        if timeline_data["separation_scores"]:
            ax5.axvline(timeline_data["avg_separation"], color=colors[i], 
                       linestyle='-', linewidth=3, 
                       label=f'{timeline_name.capitalize()}: {timeline_data["avg_separation"]:.3f}')
    
    ax5.set_xlabel('Separation Score')
    ax5.set_ylabel('Density')
    ax5.set_title('Reference Timeline Separation')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. Combined objective exploration
    ax6 = axes[1, 2]
    
    # Test different combination strategies
    strategies = {
        "Linear (0.5, 0.5)": lambda c, s: 0.5 * c + 0.5 * s,
        "Cohesion-weighted (0.7, 0.3)": lambda c, s: 0.7 * c + 0.3 * s,
        "Separation-weighted (0.3, 0.7)": lambda c, s: 0.3 * c + 0.7 * s,
        "Geometric mean": lambda c, s: np.sqrt(c * s),
        "Harmonic mean": lambda c, s: 2 / (1/c + 1/s) if c > 0 and s > 0 else 0
    }
    
    strategy_scores = {}
    for name, func in strategies.items():
        combined_scores = [func(c, s) for c, s in zip(cohesion_sample, separation_sample)]
        strategy_scores[name] = np.mean(combined_scores)
        ax6.hist(combined_scores, bins=20, alpha=0.4, label=f'{name}')
    
    ax6.set_xlabel('Combined Score')
    ax6.set_ylabel('Frequency')
    ax6.set_title('Combination Strategy Comparison')
    ax6.legend(fontsize=8)
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(save_dir, f'{domain}_objective_analysis.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"   Plot saved to: {plot_path}")
    
    # Save strategy comparison
    strategy_comparison = {
        "domain": domain,
        "combination_strategies": strategy_scores,
        "reference_timeline_performance": {}
    }
    
    # Evaluate reference timelines with each strategy
    for timeline_name, timeline_data in ref_results.items():
        if timeline_data["cohesion_scores"] and timeline_data["separation_scores"]:
            timeline_strategies = {}
            for strategy_name, func in strategies.items():
                # Compute combined scores for timeline segments
                timeline_combined = []
                min_len = min(len(timeline_data["cohesion_scores"]), 
                             len(timeline_data["separation_scores"]))
                
                for i in range(min_len):
                    c = timeline_data["cohesion_scores"][i]
                    s = timeline_data["separation_scores"][i]
                    combined = func(c, s)
                    timeline_combined.append(combined)
                
                timeline_strategies[strategy_name] = {
                    "scores": timeline_combined,
                    "mean": np.mean(timeline_combined)
                }
            
            strategy_comparison["reference_timeline_performance"][timeline_name] = timeline_strategies
    
    # Save strategy analysis
    strategy_path = os.path.join(save_dir, f'{domain}_strategy_comparison.json')
    with open(strategy_path, 'w') as f:
        json.dump(strategy_comparison, f, indent=2)
    
    print(f"   Strategy comparison saved to: {strategy_path}")
    
    return strategy_comparison


def main():
    """Run objective function analysis on test domain."""
    # Test on NLP domain (good performance, large dataset)
    domain = "natural_language_processing"
    
    print("OBJECTIVE FUNCTION ANALYSIS")
    print("=" * 80)
    print("Goals:")
    print("1. Analyze component score distributions (3000 random segments)")
    print("2. Test correlation between cohesion and separation (ideally orthogonal)")
    print("3. Compare expert timelines vs random baselines")
    print("4. Design optimal combination strategy")
    
    # Run analysis
    results = analyze_objective_function(domain)
    
    # Create plots and strategy comparison
    print("\n7. Creating visualizations and strategy analysis...")
    strategy_comparison = plot_distributions(results)
    
    # Print summary
    print(f"\n{'='*80}")
    print(f"ANALYSIS SUMMARY: {domain.upper()}")
    print(f"{'='*80}")
    
    cohesion_stats = results["distributions"]["cohesion_stats"]
    separation_stats = results["distributions"]["separation_stats"]
    correlation = results["correlation"]["pearson"]["r"]
    
    print(f"Random Segment Distributions:")
    print(f"  Cohesion (Jaccard): μ={cohesion_stats['mean']:.3f}, σ={cohesion_stats['std']:.3f}, range=[{cohesion_stats['min']:.3f}, {cohesion_stats['max']:.3f}]")
    print(f"  Separation (JS): μ={separation_stats['mean']:.3f}, σ={separation_stats['std']:.3f}, range=[{separation_stats['min']:.3f}, {separation_stats['max']:.3f}]")
    print(f"  Correlation: r={correlation:.3f} ({'Orthogonal' if abs(correlation) < 0.3 else 'Correlated'})")
    
    print(f"\nReference Timeline Performance:")
    for timeline_name, timeline_data in results["reference_timelines"].items():
        print(f"  {timeline_name.capitalize()}: cohesion={timeline_data['avg_cohesion']:.3f}, separation={timeline_data['avg_separation']:.3f}")
    
    print(f"\nCombination Strategy Rankings:")
    strategy_scores = strategy_comparison["combination_strategies"]
    sorted_strategies = sorted(strategy_scores.items(), key=lambda x: x[1], reverse=True)
    for i, (strategy, score) in enumerate(sorted_strategies):
        print(f"  {i+1}. {strategy}: {score:.3f}")
    
    # Recommendations
    print(f"\nRECOMMENDATIONS:")
    
    if abs(correlation) < 0.3:
        print(f"  ✓ Cohesion and separation are approximately orthogonal (r={correlation:.3f})")
        print(f"    → Can combine linearly without redundancy")
    else:
        print(f"  ⚠ Cohesion and separation show correlation (r={correlation:.3f})")
        print(f"    → Consider weighted combination or transformation")
    
    # Check if expert timelines score well
    expert_performance = []
    for timeline_data in results["reference_timelines"].values():
        if timeline_data["cohesion_scores"]:
            expert_performance.append((timeline_data['avg_cohesion'], timeline_data['avg_separation']))
    
    if expert_performance:
        avg_expert_cohesion = np.mean([perf[0] for perf in expert_performance])
        avg_expert_separation = np.mean([perf[1] for perf in expert_performance])
        
        cohesion_percentile = stats.percentileofscore(results["distributions"]["cohesion_scores"], avg_expert_cohesion)
        separation_percentile = stats.percentileofscore(results["distributions"]["separation_scores"], avg_expert_separation)
        
        print(f"  Expert timeline performance:")
        print(f"    Cohesion: {cohesion_percentile:.1f}th percentile")
        print(f"    Separation: {separation_percentile:.1f}th percentile")
        
        if cohesion_percentile >= 70 and separation_percentile >= 50:
            print(f"  ✓ Expert timelines show high cohesion, moderate-high separation")
            print(f"    → Objective function design is appropriate")
        else:
            print(f"  ⚠ Expert timelines may not show expected performance pattern")
            print(f"    → Consider metric refinement or different combination")
    
    # Save complete results
    results_path = f"results/objective_analysis/{domain}_complete_analysis.json"
    os.makedirs("results/objective_analysis", exist_ok=True)
    
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nComplete analysis saved to: {results_path}")


if __name__ == "__main__":
    main() 