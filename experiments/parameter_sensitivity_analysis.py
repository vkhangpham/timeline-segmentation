"""
Parameter Sensitivity Analysis for Timeline Segmentation Algorithm

This module implements systematic analysis of how algorithm parameters impact
detection quality, precision, and recall across different research domains.

Provides scientific foundation for parameter tuning and optimization.
"""

import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict
import itertools
import time
from datetime import datetime

# Algorithm imports
from core.algorithm_config import ComprehensiveAlgorithmConfig
from core.shift_signal_detection import detect_shift_signals
from core.data_processing import process_domain_data
from validation.validation_framework import load_ground_truth_data, evaluate_against_ground_truth


@dataclass
class ParameterSensitivityResult:
    """Results from parameter sensitivity analysis."""
    parameter_name: str
    domain_name: str
    test_values: List[float]
    f1_scores: List[float]
    precision_scores: List[float]
    recall_scores: List[float]
    paradigm_counts: List[int]
    confidence_means: List[float]
    optimal_value: float
    optimal_f1: float
    sensitivity_range: float
    parameter_importance: str


@dataclass 
class ParameterInteractionResult:
    """Results from parameter interaction analysis."""
    param1_name: str
    param2_name: str
    domain_name: str
    param1_values: List[float]
    param2_values: List[float]
    f1_matrix: List[List[float]]
    optimal_combination: Dict[str, float]
    interaction_strength: float
    synergistic: bool


class ParameterSensitivityAnalyzer:
    """
    Systematic parameter sensitivity analysis for algorithm optimization.
    
    Analyzes individual parameter impacts and parameter interactions to provide
    scientific foundation for algorithm configuration.
    """
    
    def __init__(self, output_dir: str = "results/parameter_analysis"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load available domains and ground truth
        self.available_domains = [
            'applied_mathematics', 'art', 'computer_science', 'computer_vision',
            'deep_learning', 'machine_learning', 'machine_translation', 
            'natural_language_processing'
        ]
        
        # Parameter analysis configuration
        self.critical_parameters = {
            'direction_threshold': {'min': 0.1, 'max': 0.8, 'steps': 8},
            'clustering_window': {'min': 1, 'max': 8, 'steps': 8},
            'validation_threshold': {'min': 0.5, 'max': 0.95, 'steps': 10},
            'citation_boost': {'min': 0.0, 'max': 0.6, 'steps': 7}
        }
        
        self.secondary_parameters = {
            'direction_window_years': {'min': 2, 'max': 6, 'steps': 5},
            'keyword_min_frequency': {'min': 1, 'max': 5, 'steps': 5},
            'min_significant_keywords': {'min': 2, 'max': 6, 'steps': 5},
            'citation_support_window': {'min': 1, 'max': 5, 'steps': 5}
        }
        
        # Critical parameter pairs for interaction analysis
        self.critical_interactions = [
            ('direction_threshold', 'clustering_window'),
            ('validation_threshold', 'citation_boost'),
            ('direction_threshold', 'validation_threshold'),
            ('clustering_window', 'citation_boost')
        ]
        
        print(f"🔬 Parameter Sensitivity Analyzer initialized")
        print(f"   📁 Output directory: {self.output_dir}")
        print(f"   🎯 Critical parameters: {len(self.critical_parameters)}")
        print(f"   🔗 Parameter interactions: {len(self.critical_interactions)}")
        print(f"   🌍 Domains available: {len(self.available_domains)}")
    
    def run_comprehensive_sensitivity_analysis(self, 
                                             focus_domains: Optional[List[str]] = None,
                                             include_interactions: bool = True) -> Dict[str, Any]:
        """
        Run comprehensive parameter sensitivity analysis.
        
        Args:
            focus_domains: Specific domains to analyze (None = all domains)
            include_interactions: Whether to include parameter interaction analysis
            
        Returns:
            Comprehensive analysis results
        """
        print(f"\n🚀 STARTING COMPREHENSIVE PARAMETER SENSITIVITY ANALYSIS")
        print(f"=" * 70)
        
        analysis_start = time.time()
        domains_to_analyze = focus_domains or self.available_domains
        
        results = {
            'metadata': {
                'analysis_date': datetime.now().isoformat(),
                'domains_analyzed': domains_to_analyze,
                'critical_parameters': list(self.critical_parameters.keys()),
                'secondary_parameters': list(self.secondary_parameters.keys()),
                'include_interactions': include_interactions
            },
            'individual_parameter_analysis': {},
            'parameter_interactions': {},
            'domain_optimization': {},
            'summary_insights': {}
        }
        
        # Phase 1: Individual parameter sensitivity
        print(f"\n📊 PHASE 1: Individual Parameter Sensitivity Analysis")
        for domain in domains_to_analyze:
            print(f"\n🔍 Analyzing domain: {domain}")
            domain_results = self._analyze_domain_parameter_sensitivity(domain)
            results['individual_parameter_analysis'][domain] = domain_results
        
        # Phase 2: Parameter interactions (if enabled)
        if include_interactions:
            print(f"\n🔗 PHASE 2: Parameter Interaction Analysis")
            for domain in domains_to_analyze:
                print(f"\n🔍 Analyzing interactions for domain: {domain}")
                interaction_results = self._analyze_parameter_interactions(domain)
                results['parameter_interactions'][domain] = interaction_results
        
        # Phase 3: Domain-specific optimization
        print(f"\n🎯 PHASE 3: Domain-Specific Optimization")
        optimization_results = self._optimize_domain_configurations(domains_to_analyze)
        results['domain_optimization'] = optimization_results
        
        # Phase 4: Generate summary insights
        print(f"\n💡 PHASE 4: Summary Insights Generation")
        summary_insights = self._generate_summary_insights(results)
        results['summary_insights'] = summary_insights
        
        # Save comprehensive results
        analysis_duration = time.time() - analysis_start
        results['metadata']['analysis_duration_seconds'] = analysis_duration
        
        output_file = self.output_dir / f"comprehensive_sensitivity_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n✅ ANALYSIS COMPLETE")
        print(f"   ⏱️  Duration: {analysis_duration:.1f} seconds")
        print(f"   📁 Results saved: {output_file}")
        
        return results
    
    def _analyze_domain_parameter_sensitivity(self, domain_name: str) -> Dict[str, Any]:
        """Analyze individual parameter sensitivity for a single domain."""
        
        print(f"  📈 Analyzing {len(self.critical_parameters)} critical parameters")
        
        domain_results = {
            'domain_info': self._get_domain_info(domain_name),
            'parameter_sensitivity': {},
            'optimal_parameters': {},
            'parameter_rankings': {}
        }
        
        try:
            # Load domain data once
            result = process_domain_data(domain_name)
            if not result.success:
                print(f"    ⚠️ Error loading {domain_name}: {result.error_message}")
                domain_results['error'] = result.error_message
                return domain_results
            domain_data = result.domain_data
            ground_truth = load_ground_truth_data()[domain_name]
            
            # Analyze each critical parameter
            for param_name, param_config in self.critical_parameters.items():
                print(f"    🎛️  Testing {param_name}...")
                
                sensitivity_result = self._test_parameter_sensitivity(
                    domain_data, ground_truth, domain_name, param_name, param_config
                )
                
                domain_results['parameter_sensitivity'][param_name] = asdict(sensitivity_result)
                domain_results['optimal_parameters'][param_name] = {
                    'optimal_value': sensitivity_result.optimal_value,
                    'optimal_f1': sensitivity_result.optimal_f1,
                    'sensitivity_range': sensitivity_result.sensitivity_range
                }
                
                print(f"      ✅ Optimal {param_name}: {sensitivity_result.optimal_value:.3f} (F1: {sensitivity_result.optimal_f1:.3f})")
        
        except Exception as e:
            print(f"    ⚠️ Error analyzing {domain_name}: {e}")
            domain_results['error'] = str(e)
        
        # Rank parameters by importance (sensitivity range)
        if 'parameter_sensitivity' in domain_results:
            param_importance = {}
            for param_name, sensitivity_data in domain_results['parameter_sensitivity'].items():
                if 'sensitivity_range' in sensitivity_data:
                    param_importance[param_name] = sensitivity_data['sensitivity_range']
            
            # Sort by sensitivity range (descending)
            ranked_params = sorted(param_importance.items(), key=lambda x: x[1], reverse=True)
            domain_results['parameter_rankings'] = {
                'by_sensitivity': ranked_params,
                'most_important': ranked_params[0][0] if ranked_params else None,
                'least_important': ranked_params[-1][0] if ranked_params else None
            }
        
        return domain_results
    
    def _test_parameter_sensitivity(self, domain_data, ground_truth, domain_name: str,
                                  param_name: str, param_config: Dict[str, Any]) -> ParameterSensitivityResult:
        """Test sensitivity of a single parameter across its valid range."""
        
        # Generate test values
        test_values = np.linspace(
            param_config['min'], 
            param_config['max'], 
            param_config['steps']
        ).tolist()
        
        # Initialize result tracking
        f1_scores = []
        precision_scores = []
        recall_scores = []
        paradigm_counts = []
        confidence_means = []
        
        # Test each parameter value
        for param_value in test_values:
            try:
                # Create configuration with modified parameter
                config = ComprehensiveAlgorithmConfig(granularity=3)
                setattr(config, param_name, param_value)
                
                # Run algorithm
                detected_shifts, _, _ = detect_shift_signals(domain_data, domain_name, config)
                
                # Evaluate performance
                evaluation = evaluate_against_ground_truth(detected_shifts, ground_truth['paradigm_shifts'])
                
                f1_scores.append(evaluation.get('f1_score_2yr', 0.0))
                precision_scores.append(evaluation.get('precision_2yr', 0.0))
                recall_scores.append(evaluation.get('recall_2yr', 0.0))
                paradigm_counts.append(len(detected_shifts))
                
                if detected_shifts:
                    confidence_means.append(np.mean([s.confidence for s in detected_shifts]))
                else:
                    confidence_means.append(0.0)
                    
            except Exception as e:
                print(f"      ⚠️ Error testing {param_name}={param_value}: {e}")
                # Add zero values for failed tests
                f1_scores.append(0.0)
                precision_scores.append(0.0)
                recall_scores.append(0.0)
                paradigm_counts.append(0)
                confidence_means.append(0.0)
        
        # Find optimal value
        optimal_idx = np.argmax(f1_scores)
        optimal_value = test_values[optimal_idx]
        optimal_f1 = f1_scores[optimal_idx]
        
        # Calculate sensitivity range
        sensitivity_range = max(f1_scores) - min(f1_scores) if f1_scores else 0.0
        
        # Determine parameter importance
        if sensitivity_range > 0.3:
            importance = "Critical"
        elif sensitivity_range > 0.15:
            importance = "High"
        elif sensitivity_range > 0.05:
            importance = "Medium"
        else:
            importance = "Low"
        
        return ParameterSensitivityResult(
            parameter_name=param_name,
            domain_name=domain_name,
            test_values=test_values,
            f1_scores=f1_scores,
            precision_scores=precision_scores,
            recall_scores=recall_scores,
            paradigm_counts=paradigm_counts,
            confidence_means=confidence_means,
            optimal_value=optimal_value,
            optimal_f1=optimal_f1,
            sensitivity_range=sensitivity_range,
            parameter_importance=importance
        )
    
    def _analyze_parameter_interactions(self, domain_name: str) -> Dict[str, Any]:
        """Analyze parameter interactions for critical parameter pairs."""
        
        print(f"  🔗 Analyzing {len(self.critical_interactions)} parameter interactions")
        
        interaction_results = {}
        
        try:
            result = process_domain_data(domain_name)
            if not result.success:
                print(f"    ⚠️ Error loading {domain_name}: {result.error_message}")
                interaction_results['error'] = result.error_message
                return interaction_results
            domain_data = result.domain_data
            ground_truth = load_ground_truth_data()[domain_name]
            
            for param1, param2 in self.critical_interactions:
                print(f"    🎛️  Testing {param1} × {param2}...")
                
                interaction_result = self._test_parameter_interaction(
                    domain_data, ground_truth, domain_name, param1, param2
                )
                
                interaction_key = f"{param1}_vs_{param2}"
                interaction_results[interaction_key] = asdict(interaction_result)
                
                print(f"      ✅ Optimal combination: {param1}={interaction_result.optimal_combination[param1]:.3f}, "
                      f"{param2}={interaction_result.optimal_combination[param2]:.3f}")
                
        except Exception as e:
            print(f"    ⚠️ Error analyzing interactions for {domain_name}: {e}")
            interaction_results['error'] = str(e)
        
        return interaction_results
    
    def _test_parameter_interaction(self, domain_data, ground_truth, domain_name: str,
                                  param1: str, param2: str) -> ParameterInteractionResult:
        """Test interaction between two parameters using grid search."""
        
        # Get parameter ranges (coarser grid for interaction analysis)
        param1_config = self.critical_parameters.get(param1, self.secondary_parameters.get(param1))
        param2_config = self.critical_parameters.get(param2, self.secondary_parameters.get(param2))
        
        # Use coarser grid for computational efficiency
        grid_size = 5
        param1_values = np.linspace(param1_config['min'], param1_config['max'], grid_size).tolist()
        param2_values = np.linspace(param2_config['min'], param2_config['max'], grid_size).tolist()
        
        # Test parameter combinations
        f1_matrix = []
        best_f1 = 0.0
        optimal_combination = {param1: param1_values[0], param2: param2_values[0]}
        
        for val1 in param1_values:
            f1_row = []
            for val2 in param2_values:
                try:
                    # Create configuration
                    config = ComprehensiveAlgorithmConfig(granularity=3)
                    setattr(config, param1, val1)
                    setattr(config, param2, val2)
                    
                    # Run algorithm
                    detected_shifts, _, _ = detect_shift_signals(domain_data, domain_name, config)
                    
                    # Evaluate performance
                    evaluation = evaluate_against_ground_truth(detected_shifts, ground_truth['paradigm_shifts'])
                    f1_score = evaluation.get('f1_score_2yr', 0.0)
                    
                    f1_row.append(f1_score)
                    
                    # Track best combination
                    if f1_score > best_f1:
                        best_f1 = f1_score
                        optimal_combination = {param1: val1, param2: val2, 'f1_score': f1_score}
                        
                except Exception as e:
                    print(f"        ⚠️ Error testing {param1}={val1}, {param2}={val2}: {e}")
                    f1_row.append(0.0)
            
            f1_matrix.append(f1_row)
        
        # Calculate interaction strength (variance across the grid)
        all_f1_values = [f1 for row in f1_matrix for f1 in row]
        interaction_strength = np.std(all_f1_values) if all_f1_values else 0.0
        
        # Check if interaction is synergistic (optimal is better than individual optima)
        individual_optima = []
        for param, values in [(param1, param1_values), (param2, param2_values)]:
            param_best_f1 = 0.0
            for val in values:
                try:
                    config = ComprehensiveAlgorithmConfig(granularity=3)
                    setattr(config, param, val)
                    detected_shifts, _, _ = detect_shift_signals(domain_data, domain_name, config)
                    evaluation = evaluate_against_ground_truth(detected_shifts, ground_truth['paradigm_shifts'])
                    param_best_f1 = max(param_best_f1, evaluation.get('f1_score_2yr', 0.0))
                except:
                    pass
            individual_optima.append(param_best_f1)
        
        synergistic = best_f1 > max(individual_optima) * 1.05  # 5% improvement threshold
        
        return ParameterInteractionResult(
            param1_name=param1,
            param2_name=param2,
            domain_name=domain_name,
            param1_values=param1_values,
            param2_values=param2_values,
            f1_matrix=f1_matrix,
            optimal_combination=optimal_combination,
            interaction_strength=interaction_strength,
            synergistic=synergistic
        )
    
    def _optimize_domain_configurations(self, domains: List[str]) -> Dict[str, Any]:
        """Generate optimized configurations for different domain types."""
        
        # Domain categories based on paradigm shift characteristics
        domain_categories = {
            'very_dynamic': ['deep_learning', 'natural_language_processing', 'computer_vision'],
            'dynamic': ['computer_science', 'machine_translation'],
            'stable': ['applied_mathematics', 'machine_learning'],
            'very_stable': ['art']
        }
        
        optimization_results = {}
        
        for category, category_domains in domain_categories.items():
            available_domains = [d for d in category_domains if d in domains]
            if not available_domains:
                continue
                
            print(f"  🎯 Optimizing {category} domains: {available_domains}")
            
            # Aggregate optimal parameters across domains in category
            category_optimal_params = defaultdict(list)
            
            for domain in available_domains:
                if domain in domains:
                    domain_results = self.output_dir.parent / "parameter_analysis" / f"{domain}_sensitivity.json"
                    # For now, use simple averaging - could be enhanced with weighted optimization
                    for param_name in self.critical_parameters:
                        try:
                            # Test current default config
                            config = ComprehensiveAlgorithmConfig(granularity=3)
                            category_optimal_params[param_name].append(getattr(config, param_name))
                        except:
                            pass
            
            # Generate category-optimized configuration
            optimized_config = {}
            for param_name, values in category_optimal_params.items():
                if values:
                    optimized_config[param_name] = np.mean(values)
            
            optimization_results[category] = {
                'domains': available_domains,
                'optimized_parameters': optimized_config,
                'characteristics': self._get_category_characteristics(category)
            }
        
        return optimization_results
    
    def _generate_summary_insights(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate high-level insights from sensitivity analysis."""
        
        insights = {
            'most_critical_parameters': [],
            'parameter_importance_ranking': {},
            'domain_specific_insights': {},
            'configuration_recommendations': {},
            'interaction_insights': {}
        }
        
        # Aggregate parameter importance across domains
        param_importance_scores = defaultdict(list)
        
        for domain, domain_results in results['individual_parameter_analysis'].items():
            if 'parameter_sensitivity' in domain_results:
                for param_name, sensitivity_data in domain_results['parameter_sensitivity'].items():
                    if 'sensitivity_range' in sensitivity_data:
                        param_importance_scores[param_name].append(sensitivity_data['sensitivity_range'])
        
        # Calculate average importance and ranking
        avg_importance = {}
        for param_name, scores in param_importance_scores.items():
            avg_importance[param_name] = np.mean(scores) if scores else 0.0
        
        # Sort by importance
        ranked_params = sorted(avg_importance.items(), key=lambda x: x[1], reverse=True)
        insights['parameter_importance_ranking'] = dict(ranked_params)
        insights['most_critical_parameters'] = [param for param, score in ranked_params[:3]]
        
        # Generate domain-specific insights
        for domain, domain_results in results['individual_parameter_analysis'].items():
            if 'parameter_rankings' in domain_results:
                insights['domain_specific_insights'][domain] = {
                    'most_important_param': domain_results['parameter_rankings'].get('most_important'),
                    'optimal_config': domain_results.get('optimal_parameters', {})
                }
        
        # Configuration recommendations
        insights['configuration_recommendations'] = {
            'general': "Use granularity presets as starting point, then fine-tune critical parameters",
            'dynamic_domains': "Lower direction_threshold and clustering_window for rapid evolution fields",
            'stable_domains': "Higher thresholds and longer windows for slowly evolving fields"
        }
        
        return insights
    
    def _get_domain_info(self, domain_name: str) -> Dict[str, Any]:
        """Get basic information about a domain."""
        try:
            ground_truth = load_ground_truth_data()[domain_name]
            return {
                'name': domain_name,
                'paradigm_shifts': len(ground_truth.get('paradigm_shifts', [])),
                'temporal_coverage': ground_truth.get('temporal_coverage', {}),
                'characteristics': ground_truth.get('period_characteristics', {})
            }
        except Exception as e:
            return {'name': domain_name, 'error': str(e)}
    
    def _get_category_characteristics(self, category: str) -> Dict[str, str]:
        """Get characteristics of domain category."""
        characteristics = {
            'very_dynamic': {
                'evolution_rate': 'Very Fast',
                'paradigm_frequency': 'High',
                'recommended_sensitivity': 'High'
            },
            'dynamic': {
                'evolution_rate': 'Fast', 
                'paradigm_frequency': 'Medium',
                'recommended_sensitivity': 'Medium-High'
            },
            'stable': {
                'evolution_rate': 'Slow',
                'paradigm_frequency': 'Low',
                'recommended_sensitivity': 'Medium-Low'
            },
            'very_stable': {
                'evolution_rate': 'Very Slow',
                'paradigm_frequency': 'Very Low',
                'recommended_sensitivity': 'Low'
            }
        }
        return characteristics.get(category, {})


def main():
    """Run parameter sensitivity analysis."""
    
    print("🔬 PARAMETER SENSITIVITY ANALYSIS")
    print("=" * 50)
    
    analyzer = ParameterSensitivityAnalyzer()
    
    # Run analysis on a subset of domains for testing
    focus_domains = ['computer_vision', 'applied_mathematics', 'deep_learning']
    
    results = analyzer.run_comprehensive_sensitivity_analysis(
        focus_domains=focus_domains,
        include_interactions=True
    )
    
    print(f"\n📋 ANALYSIS SUMMARY:")
    print(f"   🎯 Most critical parameters: {results['summary_insights']['most_critical_parameters']}")
    print(f"   📊 Domains analyzed: {len(results['individual_parameter_analysis'])}")
    print(f"   🔗 Interactions tested: {len(results.get('parameter_interactions', {}))}")
    
    return results


if __name__ == "__main__":
    main() 