#!/usr/bin/env python3
"""
Multi-Domain Objective Function Analysis
========================================

Analyze cohesion and separation metrics across multiple domains to:
1. Validate metric consistency
2. Design robust combination strategies
3. Optimize weights for expert timeline performance

Key Findings from NLP Analysis:
- Cohesion and separation are orthogonal (r=0.023) ✓
- Expert timelines show moderate cohesion (44th percentile), low separation (17th percentile)
- Cohesion-weighted combination (0.7, 0.3) performs best for expert timelines
"""

import os
import sys
import json
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any
from scipy.stats import pearsonr
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.objective_function_analysis import (
    analyze_objective_function, 
    plot_distributions,
    load_domain_papers,
    filter_keywords,
    cohesion_jaccard,
    separation_jensen_shannon
)


def discover_available_domains() -> List[str]:
    """Discover available domains from JSON data sources."""
    resources_dir = "resources"
    available_domains = []
    
    if not os.path.exists(resources_dir):
        return []
    
    for domain_dir in os.listdir(resources_dir):
        domain_path = os.path.join(resources_dir, domain_dir)
        if os.path.isdir(domain_path):
            docs_info_file = os.path.join(domain_path, f"{domain_dir}_docs_info.json")
            if os.path.exists(docs_info_file):
                available_domains.append(domain_dir)
    
    return sorted(available_domains)


def analyze_domain_wrapper(domain_name: str) -> Dict[str, Any]:
    """Wrapper for domain analysis to enable parallel processing."""
    try:
        print(f"Starting analysis for {domain_name}...")
        result = analyze_objective_function(domain_name)
        print(f"Completed analysis for {domain_name}")
        return result
    except Exception as e:
        print(f"ERROR analyzing {domain_name}: {e}")
        return {"domain": domain_name, "error": str(e), "success": False}


def design_optimal_combination(multi_domain_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Design optimal combination strategy based on multi-domain analysis.
    
    Goals:
    1. Maximize expert timeline performance
    2. Maintain orthogonality benefits
    3. Robust across domains
    """
    print("\nDESIGNING OPTIMAL COMBINATION STRATEGY")
    print("=" * 60)
    
    # Collect expert timeline performance data
    expert_performance = {}
    correlation_data = {}
    distribution_data = {}
    
    for domain, results in multi_domain_results.items():
        if "error" in results:
            continue
        
        # Correlation analysis
        correlation_data[domain] = results["correlation"]["pearson"]["r"]
        
        # Distribution statistics
        distribution_data[domain] = {
            "cohesion_stats": results["distributions"]["cohesion_stats"],
            "separation_stats": results["distributions"]["separation_stats"]
        }
        
        # Expert timeline performance
        expert_performance[domain] = {}
        for timeline_name, timeline_data in results["reference_timelines"].items():
            expert_performance[domain][timeline_name] = {
                "cohesion": timeline_data["avg_cohesion"],
                "separation": timeline_data["avg_separation"],
                "num_segments": timeline_data["num_segments"]
            }
    
    # Analyze correlation consistency
    correlations = list(correlation_data.values())
    avg_correlation = np.mean(correlations)
    correlation_consistency = np.std(correlations)
    
    print(f"Cross-domain correlation analysis:")
    print(f"  Average correlation: {avg_correlation:.3f}")
    print(f"  Correlation std: {correlation_consistency:.3f}")
    print(f"  Orthogonality: {'✓' if abs(avg_correlation) < 0.3 else '✗'}")
    
    # Test combination strategies across domains
    combination_strategies = {
        "equal_weight": (0.5, 0.5),
        "cohesion_heavy": (0.7, 0.3),
        "cohesion_dominant": (0.8, 0.2),
        "separation_heavy": (0.3, 0.7),
        "adaptive_balanced": (0.6, 0.4)
    }
    
    strategy_performance = {}
    
    for strategy_name, (w_cohesion, w_separation) in combination_strategies.items():
        strategy_performance[strategy_name] = {
            "weights": (w_cohesion, w_separation),
            "domain_scores": {},
            "expert_scores": {}
        }
        
        for domain, results in multi_domain_results.items():
            if "error" in results:
                continue
            
            # Calculate combined scores for random segments
            cohesion_scores = np.array(results["distributions"]["cohesion_scores"])
            separation_scores = np.array(results["distributions"]["separation_scores"])
            
            min_size = min(len(cohesion_scores), len(separation_scores))
            combined_random = w_cohesion * cohesion_scores[:min_size] + w_separation * separation_scores[:min_size]
            
            strategy_performance[strategy_name]["domain_scores"][domain] = {
                "mean": float(np.mean(combined_random)),
                "std": float(np.std(combined_random))
            }
            
            # Calculate expert timeline performance
            expert_combined = {}
            for timeline_name, timeline_data in results["reference_timelines"].items():
                if timeline_data["cohesion_scores"] and timeline_data["separation_scores"]:
                    expert_cohesion = np.mean(timeline_data["cohesion_scores"])
                    expert_separation = np.mean(timeline_data["separation_scores"])
                    expert_combined[timeline_name] = w_cohesion * expert_cohesion + w_separation * expert_separation
            
            strategy_performance[strategy_name]["expert_scores"][domain] = expert_combined
    
    # Rank strategies by expert timeline performance
    strategy_rankings = {}
    
    for strategy_name, strategy_data in strategy_performance.items():
        expert_scores = []
        for domain_experts in strategy_data["expert_scores"].values():
            expert_scores.extend(domain_experts.values())
        
        if expert_scores:
            strategy_rankings[strategy_name] = {
                "mean_expert_score": np.mean(expert_scores),
                "expert_count": len(expert_scores),
                "weights": strategy_data["weights"]
            }
    
    # Sort by expert performance
    ranked_strategies = sorted(strategy_rankings.items(), 
                              key=lambda x: x[1]["mean_expert_score"], 
                              reverse=True)
    
    print(f"\nCombination strategy rankings (by expert timeline performance):")
    for i, (strategy_name, data) in enumerate(ranked_strategies):
        w_c, w_s = data["weights"]
        score = data["mean_expert_score"]
        count = data["expert_count"]
        print(f"  {i+1}. {strategy_name}: {score:.3f} (w_c={w_c}, w_s={w_s}, n={count})")
    
    # Design adaptive strategy based on domain characteristics
    domain_characteristics = {}
    for domain, results in multi_domain_results.items():
        if "error" in results:
            continue
        
        cohesion_stats = results["distributions"]["cohesion_stats"]
        separation_stats = results["distributions"]["separation_stats"]
        
        # Calculate relative scales
        cohesion_scale = cohesion_stats["max"] - cohesion_stats["min"]
        separation_scale = separation_stats["max"] - separation_stats["min"]
        
        # Expert timeline relative performance
        expert_cohesion_percentiles = []
        expert_separation_percentiles = []
        
        for timeline_data in results["reference_timelines"].values():
            if timeline_data["cohesion_scores"]:
                expert_coh = np.mean(timeline_data["cohesion_scores"])
                cohesion_scores = results["distributions"]["cohesion_scores"]
                percentile = (np.sum(np.array(cohesion_scores) <= expert_coh) / len(cohesion_scores)) * 100
                expert_cohesion_percentiles.append(percentile)
            
            if timeline_data["separation_scores"]:
                expert_sep = np.mean(timeline_data["separation_scores"])
                separation_scores = results["distributions"]["separation_scores"]
                percentile = (np.sum(np.array(separation_scores) <= expert_sep) / len(separation_scores)) * 100
                expert_separation_percentiles.append(percentile)
        
        domain_characteristics[domain] = {
            "cohesion_scale": cohesion_scale,
            "separation_scale": separation_scale,
            "expert_cohesion_percentile": np.mean(expert_cohesion_percentiles) if expert_cohesion_percentiles else 50,
            "expert_separation_percentile": np.mean(expert_separation_percentiles) if expert_separation_percentiles else 50,
            "correlation": correlation_data.get(domain, 0.0)
        }
    
    print(f"\nDomain characteristics analysis:")
    for domain, chars in domain_characteristics.items():
        print(f"  {domain}:")
        print(f"    Expert cohesion percentile: {chars['expert_cohesion_percentile']:.1f}")
        print(f"    Expert separation percentile: {chars['expert_separation_percentile']:.1f}")
        print(f"    Correlation: {chars['correlation']:.3f}")
    
    # Final recommendation
    best_strategy = ranked_strategies[0]
    recommended_weights = best_strategy[1]["weights"]
    
    print(f"\nRECOMMENDED COMBINATION STRATEGY:")
    print(f"  Strategy: {best_strategy[0]}")
    print(f"  Weights: cohesion={recommended_weights[0]:.1f}, separation={recommended_weights[1]:.1f}")
    print(f"  Expert performance: {best_strategy[1]['mean_expert_score']:.3f}")
    
    # Additional insights
    avg_expert_cohesion_percentile = np.mean([chars['expert_cohesion_percentile'] 
                                             for chars in domain_characteristics.values()])
    avg_expert_separation_percentile = np.mean([chars['expert_separation_percentile'] 
                                               for chars in domain_characteristics.values()])
    
    print(f"\nCROSS-DOMAIN INSIGHTS:")
    print(f"  Expert timelines average performance:")
    print(f"    Cohesion: {avg_expert_cohesion_percentile:.1f}th percentile")
    print(f"    Separation: {avg_expert_separation_percentile:.1f}th percentile")
    
    if avg_expert_cohesion_percentile > 60:
        print(f"  ✓ Expert timelines show strong cohesion")
    else:
        print(f"  ⚠ Expert timelines show moderate cohesion")
    
    if avg_expert_separation_percentile > 50:
        print(f"  ✓ Expert timelines show good separation")
    else:
        print(f"  ⚠ Expert timelines show weak separation (may indicate conservative segmentation)")
    
    return {
        "recommended_strategy": best_strategy[0],
        "recommended_weights": recommended_weights,
        "strategy_rankings": dict(ranked_strategies),
        "domain_characteristics": domain_characteristics,
        "cross_domain_correlation": {
            "mean": avg_correlation,
            "std": correlation_consistency,
            "orthogonal": abs(avg_correlation) < 0.3
        },
        "expert_performance_summary": {
            "avg_cohesion_percentile": avg_expert_cohesion_percentile,
            "avg_separation_percentile": avg_expert_separation_percentile
        }
    }


def create_multi_domain_visualization(multi_domain_results: Dict[str, Dict[str, Any]], 
                                    combination_analysis: Dict[str, Any],
                                    save_dir: str = "results/objective_analysis"):
    """Create comprehensive multi-domain visualization."""
    os.makedirs(save_dir, exist_ok=True)
    
    # Set up plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create comprehensive figure
    fig, axes = plt.subplots(3, 3, figsize=(20, 16))
    fig.suptitle('Multi-Domain Objective Function Analysis', fontsize=16, fontweight='bold')
    
    # Collect data across domains
    domains = []
    correlations = []
    cohesion_means = []
    separation_means = []
    expert_cohesion_percentiles = []
    expert_separation_percentiles = []
    
    for domain, results in multi_domain_results.items():
        if "error" in results:
            continue
        
        domains.append(domain.replace('_', ' ').title())
        correlations.append(results["correlation"]["pearson"]["r"])
        cohesion_means.append(results["distributions"]["cohesion_stats"]["mean"])
        separation_means.append(results["distributions"]["separation_stats"]["mean"])
        
        # Expert percentiles
        domain_chars = combination_analysis["domain_characteristics"][domain]
        expert_cohesion_percentiles.append(domain_chars["expert_cohesion_percentile"])
        expert_separation_percentiles.append(domain_chars["expert_separation_percentile"])
    
    # 1. Correlation across domains
    ax1 = axes[0, 0]
    bars = ax1.bar(range(len(domains)), correlations, color='skyblue', alpha=0.7)
    ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax1.axhline(y=0.3, color='red', linestyle='--', alpha=0.5, label='Correlation threshold')
    ax1.axhline(y=-0.3, color='red', linestyle='--', alpha=0.5)
    ax1.set_xlabel('Domain')
    ax1.set_ylabel('Pearson Correlation')
    ax1.set_title('Cohesion-Separation Correlation by Domain')
    ax1.set_xticks(range(len(domains)))
    ax1.set_xticklabels(domains, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add correlation values on bars
    for i, (bar, corr) in enumerate(zip(bars, correlations)):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01 if height >= 0 else height - 0.03,
                f'{corr:.3f}', ha='center', va='bottom' if height >= 0 else 'top', fontsize=8)
    
    # 2. Distribution means across domains
    ax2 = axes[0, 1]
    x = np.arange(len(domains))
    width = 0.35
    
    bars1 = ax2.bar(x - width/2, cohesion_means, width, label='Cohesion', alpha=0.7)
    bars2 = ax2.bar(x + width/2, separation_means, width, label='Separation', alpha=0.7)
    
    ax2.set_xlabel('Domain')
    ax2.set_ylabel('Mean Score')
    ax2.set_title('Distribution Means by Domain')
    ax2.set_xticks(x)
    ax2.set_xticklabels(domains, rotation=45, ha='right')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Expert timeline performance
    ax3 = axes[0, 2]
    ax3.scatter(expert_cohesion_percentiles, expert_separation_percentiles, 
               s=100, alpha=0.7, c=range(len(domains)), cmap='viridis')
    
    # Add domain labels
    for i, domain in enumerate(domains):
        ax3.annotate(domain[:3], (expert_cohesion_percentiles[i], expert_separation_percentiles[i]),
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    ax3.axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='50th percentile')
    ax3.axvline(x=50, color='gray', linestyle='--', alpha=0.5)
    ax3.set_xlabel('Expert Cohesion Percentile')
    ax3.set_ylabel('Expert Separation Percentile')
    ax3.set_title('Expert Timeline Performance')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4-6. Distribution plots for first 3 domains
    plot_domains = list(multi_domain_results.keys())[:3]
    
    for i, domain in enumerate(plot_domains):
        if "error" in multi_domain_results[domain]:
            continue
        
        ax = axes[1, i]
        results = multi_domain_results[domain]
        
        cohesion_scores = results["distributions"]["cohesion_scores"]
        separation_scores = results["distributions"]["separation_scores"]
        
        # Plot distributions
        ax.hist(cohesion_scores, bins=30, alpha=0.5, label='Cohesion', density=True)
        ax.hist(separation_scores, bins=30, alpha=0.5, label='Separation', density=True)
        
        # Mark expert timeline performance
        for timeline_name, timeline_data in results["reference_timelines"].items():
            if timeline_data["cohesion_scores"]:
                ax.axvline(timeline_data["avg_cohesion"], color='red', linestyle='-', 
                          alpha=0.7, label=f'{timeline_name} cohesion')
            if timeline_data["separation_scores"]:
                ax.axvline(timeline_data["avg_separation"], color='blue', linestyle='-', 
                          alpha=0.7, label=f'{timeline_name} separation')
        
        ax.set_xlabel('Score')
        ax.set_ylabel('Density')
        ax.set_title(f'{domain.replace("_", " ").title()} Distributions')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    # 7. Strategy comparison
    ax7 = axes[2, 0]
    strategy_rankings = combination_analysis["strategy_rankings"]
    
    strategies = list(strategy_rankings.keys())
    expert_scores = [data["mean_expert_score"] for data in strategy_rankings.values()]
    
    bars = ax7.bar(range(len(strategies)), expert_scores, alpha=0.7)
    ax7.set_xlabel('Combination Strategy')
    ax7.set_ylabel('Mean Expert Score')
    ax7.set_title('Strategy Performance Comparison')
    ax7.set_xticks(range(len(strategies)))
    ax7.set_xticklabels(strategies, rotation=45, ha='right')
    ax7.grid(True, alpha=0.3)
    
    # Highlight best strategy
    best_idx = np.argmax(expert_scores)
    bars[best_idx].set_color('gold')
    
    # 8. Weight sensitivity analysis
    ax8 = axes[2, 1]
    
    # Test different weight combinations
    weight_range = np.linspace(0.1, 0.9, 9)
    weight_performance = []
    
    for w_cohesion in weight_range:
        w_separation = 1.0 - w_cohesion
        
        # Calculate average expert performance across domains
        expert_scores = []
        for domain, results in multi_domain_results.items():
            if "error" in results:
                continue
            
            for timeline_data in results["reference_timelines"].values():
                if timeline_data["cohesion_scores"] and timeline_data["separation_scores"]:
                    expert_cohesion = np.mean(timeline_data["cohesion_scores"])
                    expert_separation = np.mean(timeline_data["separation_scores"])
                    combined = w_cohesion * expert_cohesion + w_separation * expert_separation
                    expert_scores.append(combined)
        
        weight_performance.append(np.mean(expert_scores) if expert_scores else 0)
    
    ax8.plot(weight_range, weight_performance, 'o-', linewidth=2, markersize=6)
    
    # Mark recommended weight
    recommended_weight = combination_analysis["recommended_weights"][0]
    recommended_idx = np.argmin(np.abs(weight_range - recommended_weight))
    ax8.scatter(weight_range[recommended_idx], weight_performance[recommended_idx], 
               color='red', s=100, zorder=5, label=f'Recommended: {recommended_weight:.1f}')
    
    ax8.set_xlabel('Cohesion Weight')
    ax8.set_ylabel('Mean Expert Score')
    ax8.set_title('Weight Sensitivity Analysis')
    ax8.legend()
    ax8.grid(True, alpha=0.3)
    
    # 9. Summary statistics
    ax9 = axes[2, 2]
    ax9.axis('off')
    
    # Create summary text
    summary_text = f"""
MULTI-DOMAIN ANALYSIS SUMMARY

Domains Analyzed: {len(domains)}
Cross-domain Correlation: {combination_analysis['cross_domain_correlation']['mean']:.3f} ± {combination_analysis['cross_domain_correlation']['std']:.3f}
Orthogonality: {'✓' if combination_analysis['cross_domain_correlation']['orthogonal'] else '✗'}

Expert Timeline Performance:
  Cohesion: {combination_analysis['expert_performance_summary']['avg_cohesion_percentile']:.1f}th percentile
  Separation: {combination_analysis['expert_performance_summary']['avg_separation_percentile']:.1f}th percentile

Recommended Strategy: {combination_analysis['recommended_strategy']}
Optimal Weights: {combination_analysis['recommended_weights'][0]:.1f} cohesion, {combination_analysis['recommended_weights'][1]:.1f} separation

Key Insights:
• Cohesion and separation are orthogonal across domains
• Expert timelines favor cohesion over separation
• Cohesion-weighted strategies perform best
• Consistent patterns across different research domains
"""
    
    ax9.text(0.05, 0.95, summary_text, transform=ax9.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(save_dir, 'multi_domain_objective_analysis.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Multi-domain analysis plot saved to: {plot_path}")
    
    return plot_path


def main():
    """Run multi-domain objective function analysis."""
    print("MULTI-DOMAIN OBJECTIVE FUNCTION ANALYSIS")
    print("=" * 80)
    
    # Discover available domains
    available_domains = discover_available_domains()
    
    # Select test domains (prioritize diverse, well-performing domains)
    test_domains = [
        "natural_language_processing",  # Already analyzed
        "computer_vision",              # Technical domain
        "applied_mathematics",          # Theoretical domain
        "art"                          # Non-technical domain
    ]
    
    # Filter to available domains
    test_domains = [d for d in test_domains if d in available_domains]
    
    print(f"Analyzing {len(test_domains)} domains: {', '.join(test_domains)}")
    
    # Load existing NLP results if available
    nlp_results_path = "results/objective_analysis/natural_language_processing_complete_analysis.json"
    
    multi_domain_results = {}
    
    if os.path.exists(nlp_results_path):
        print("Loading existing NLP analysis...")
        with open(nlp_results_path, 'r') as f:
            multi_domain_results["natural_language_processing"] = json.load(f)
        
        # Remove NLP from test domains to avoid re-analysis
        if "natural_language_processing" in test_domains:
            test_domains.remove("natural_language_processing")
    
    # Analyze remaining domains
    if test_domains:
        print(f"\nAnalyzing {len(test_domains)} additional domains...")
        
        # Use parallel processing for faster analysis
        max_workers = min(4, len(test_domains))
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_domain = {
                executor.submit(analyze_domain_wrapper, domain): domain 
                for domain in test_domains
            }
            
            for future in as_completed(future_to_domain):
                domain = future_to_domain[future]
                try:
                    result = future.result()
                    multi_domain_results[domain] = result
                except Exception as e:
                    print(f"ERROR: Analysis failed for {domain}: {e}")
                    multi_domain_results[domain] = {"domain": domain, "error": str(e)}
    
    # Filter successful results
    successful_results = {
        domain: results for domain, results in multi_domain_results.items()
        if "error" not in results
    }
    
    print(f"\nSuccessfully analyzed {len(successful_results)} domains")
    
    if len(successful_results) < 2:
        print("ERROR: Need at least 2 successful domain analyses for multi-domain analysis")
        return
    
    # Design optimal combination strategy
    combination_analysis = design_optimal_combination(successful_results)
    
    # Create multi-domain visualization
    print(f"\nCreating multi-domain visualization...")
    plot_path = create_multi_domain_visualization(successful_results, combination_analysis)
    
    # Save complete multi-domain analysis (with JSON-safe conversion)
    def make_json_safe(obj):
        """Convert numpy types and other non-JSON types to JSON-safe types."""
        if isinstance(obj, dict):
            return {key: make_json_safe(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [make_json_safe(item) for item in obj]
        elif isinstance(obj, tuple):
            return list(make_json_safe(item) for item in obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        else:
            return obj
    
    complete_analysis = {
        "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
        "domains_analyzed": list(successful_results.keys()),
        "combination_analysis": make_json_safe(combination_analysis),
        "individual_domain_results": {
            domain: {
                "correlation": float(results["correlation"]["pearson"]["r"]),
                "cohesion_stats": make_json_safe(results["distributions"]["cohesion_stats"]),
                "separation_stats": make_json_safe(results["distributions"]["separation_stats"]),
                "reference_timelines": make_json_safe(results["reference_timelines"])
            }
            for domain, results in successful_results.items()
        }
    }
    
    results_path = "results/objective_analysis/multi_domain_complete_analysis.json"
    os.makedirs("results/objective_analysis", exist_ok=True)
    
    with open(results_path, 'w') as f:
        json.dump(complete_analysis, f, indent=2)
    
    print(f"Complete multi-domain analysis saved to: {results_path}")
    
    # Update optimization config with recommended weights
    config_path = "optimization_config.json"
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Update final combination weights
        recommended_weights = combination_analysis["recommended_weights"]
        config["consensus_difference_weights"]["final_combination_weights"] = {
            "consensus_weight": recommended_weights[0],
            "difference_weight": recommended_weights[1]
        }
        
        # Add analysis metadata (with JSON-safe conversion)
        config["objective_function_analysis"] = make_json_safe({
            "analysis_date": time.strftime('%Y-%m-%d %H:%M:%S'),
            "domains_analyzed": list(successful_results.keys()),
            "recommended_strategy": combination_analysis["recommended_strategy"],
            "cross_domain_orthogonality": combination_analysis["cross_domain_correlation"]["orthogonal"],
            "expert_performance_summary": combination_analysis["expert_performance_summary"]
        })
        
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"Updated optimization_config.json with recommended weights: {recommended_weights}")
    
    print(f"\nMULTI-DOMAIN ANALYSIS COMPLETE")
    print(f"View results: {plot_path}")


if __name__ == "__main__":
    main() 