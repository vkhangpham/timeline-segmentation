"""
Comprehensive Ground Truth Validation Framework

This module implements rigorous evaluation of the Timeline Segmentation Algorithm
against all available ground truth data with statistical significance testing,
baseline comparisons, and performance claims validation.
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
import scipy.stats as stats
import random

# Algorithm imports
from core.algorithm_config import ComprehensiveAlgorithmConfig
from core.shift_signal_detection import detect_shift_signals
from core.data_processing import process_domain_data
from validation.validation_framework import load_ground_truth_data, evaluate_against_ground_truth


@dataclass
class ValidationResult:
    """Comprehensive validation result for a single domain."""
    domain_name: str
    ground_truth_shifts: List[int]
    detected_shifts: List[int]
    configuration: Dict[str, Any]
    
    # Core metrics
    precision_2yr: float
    recall_2yr: float
    f1_score_2yr: float
    precision_5yr: float
    recall_5yr: float
    f1_score_5yr: float
    
    # Count accuracy
    paradigm_count_detected: int
    paradigm_count_ground_truth: int
    count_accuracy: float
    
    # Confidence analysis
    confidence_mean: float
    confidence_std: float
    confidence_min: float
    confidence_max: float
    
    # Additional metrics
    temporal_precision: float
    temporal_recall: float
    evidence_quality_score: float


@dataclass
class BaselineResult:
    """Baseline method evaluation result."""
    method_name: str
    domain_name: str
    detected_shifts: List[int]
    precision_2yr: float
    recall_2yr: float
    f1_score_2yr: float


@dataclass
class CrossDomainAnalysis:
    """Cross-domain performance analysis."""
    overall_f1_2yr: float
    overall_precision_2yr: float
    overall_recall_2yr: float
    domain_f1_scores: Dict[str, float]
    best_performing_domain: str
    worst_performing_domain: str
    domain_type_performance: Dict[str, float]
    performance_variance: float


class ComprehensiveValidationFramework:
    """
    Comprehensive validation framework for systematic algorithm assessment.
    
    Builds upon IMPROVEMENT-002 transparency framework with rigorous 
    performance measurement and statistical significance testing.
    """
    
    def __init__(self, output_dir: str = "results/validation"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load ground truth data
        self.ground_truth_data = load_ground_truth_data()
        
        # Available domains
        self.available_domains = [
            'applied_mathematics', 'art', 'computer_science', 'computer_vision',
            'deep_learning', 'machine_learning', 'machine_translation', 
            'natural_language_processing'
        ]
        
        # Domain categories for analysis
        self.domain_categories = {
            'very_dynamic': ['deep_learning', 'natural_language_processing', 'computer_vision'],
            'dynamic': ['computer_science', 'machine_translation'],
            'stable': ['applied_mathematics', 'machine_learning'],
            'very_stable': ['art']
        }
        
        print(f"🔍 Comprehensive Validation Framework initialized")
        print(f"   📁 Output directory: {self.output_dir}")
        print(f"   🌍 Domains available: {len(self.available_domains)}")
        print(f"   📊 Total ground truth paradigm shifts: {self._count_total_paradigm_shifts()}")
    
    def run_comprehensive_validation(self) -> Dict[str, Any]:
        """
        Run systematic validation across all domains with multiple configurations.
        
        Returns:
            Comprehensive validation results
        """
        print(f"\n🚀 STARTING COMPREHENSIVE GROUND TRUTH VALIDATION")
        print(f"=" * 70)
        
        validation_start = time.time()
        
        results = {
            'metadata': {
                'validation_date': datetime.now().isoformat(),
                'domains_evaluated': self.available_domains,
                'total_ground_truth_shifts': self._count_total_paradigm_shifts(),
                'validation_methodology': 'Comprehensive Ground Truth Assessment with Statistical Significance'
            },
            'domain_evaluations': {},
            'baseline_comparisons': {},
            'cross_domain_analysis': {},
            'configuration_comparison': {},
            'statistical_significance': {},
            'performance_claims_validation': {},
            'executive_summary': {}
        }
        
        # Phase 1: Systematic domain evaluation
        print(f"\n📊 PHASE 1: Systematic Domain Evaluation")
        domain_results = {}
        for domain in self.available_domains:
            print(f"\n🔍 Evaluating domain: {domain}")
            domain_result = self._evaluate_domain_comprehensive(domain)
            domain_results[domain] = domain_result
            results['domain_evaluations'][domain] = asdict(domain_result)
        
        # Phase 2: Baseline method comparisons
        print(f"\n📈 PHASE 2: Baseline Method Comparisons")
        baseline_results = self._run_baseline_comparisons()
        results['baseline_comparisons'] = baseline_results
        
        # Phase 3: Cross-domain analysis
        print(f"\n🌐 PHASE 3: Cross-Domain Performance Analysis")
        cross_domain_analysis = self._analyze_cross_domain_performance(domain_results)
        results['cross_domain_analysis'] = asdict(cross_domain_analysis)
        
        # Phase 4: Configuration sensitivity analysis
        print(f"\n⚙️ PHASE 4: Configuration Impact Analysis")
        config_analysis = self._analyze_configuration_impact(domain_results)
        results['configuration_comparison'] = config_analysis
        
        # Phase 5: Statistical significance testing
        print(f"\n📊 PHASE 5: Statistical Significance Testing")
        statistical_analysis = self._test_statistical_significance(domain_results, baseline_results)
        results['statistical_significance'] = statistical_analysis
        
        # Phase 6: Performance claims validation
        print(f"\n🎯 PHASE 6: Original Performance Claims Validation")
        claims_validation = self._validate_original_claims(domain_results)
        results['performance_claims_validation'] = claims_validation
        
        # Phase 7: Executive summary generation
        print(f"\n📋 PHASE 7: Executive Summary Generation")
        executive_summary = self._generate_executive_summary(results)
        results['executive_summary'] = executive_summary
        
        # Save comprehensive results
        validation_duration = time.time() - validation_start
        results['metadata']['validation_duration_seconds'] = validation_duration
        
        output_file = self.output_dir / f"comprehensive_validation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n✅ COMPREHENSIVE VALIDATION COMPLETE")
        print(f"   ⏱️  Duration: {validation_duration:.1f} seconds")
        print(f"   📁 Results saved: {output_file}")
        
        # Generate human-readable report
        self._generate_validation_report(results)
        
        return results
    
    def _evaluate_domain_comprehensive(self, domain_name: str) -> ValidationResult:
        """Comprehensive evaluation of single domain."""
        
        print(f"  📈 Running algorithm evaluation...")
        
        try:
            # Load domain data
            result = process_domain_data(domain_name)
            if not result.success:
                print(f"    ⚠️ Error loading {domain_name}: {result.error_message}")
                return self._create_empty_validation_result(domain_name)
            
            domain_data = result.domain_data
            ground_truth = self.ground_truth_data[domain_name]
            
            # Run algorithm with default configuration
            config = ComprehensiveAlgorithmConfig(granularity=3)
            detected_shifts, _, _ = detect_shift_signals(domain_data, domain_name, config)
            
            # Evaluate performance
            evaluation = evaluate_against_ground_truth(detected_shifts, ground_truth['paradigm_shifts'])
            
            # Calculate confidence statistics
            confidence_stats = self._calculate_confidence_statistics(detected_shifts)
            
            # Create validation result
            validation_result = ValidationResult(
                domain_name=domain_name,
                ground_truth_shifts=ground_truth['paradigm_shifts'],
                detected_shifts=[s.year for s in detected_shifts],
                configuration=config.to_dict(),
                
                # Core metrics
                precision_2yr=evaluation['precision_2yr'],
                recall_2yr=evaluation['recall_2yr'],
                f1_score_2yr=evaluation['f1_score_2yr'],
                precision_5yr=evaluation['precision_5yr'],
                recall_5yr=evaluation['recall_5yr'],
                f1_score_5yr=evaluation['f1_score_5yr'],
                
                # Count accuracy
                paradigm_count_detected=evaluation['paradigm_count_detected'],
                paradigm_count_ground_truth=evaluation['paradigm_count_ground_truth'],
                count_accuracy=evaluation['count_accuracy'],
                
                # Confidence analysis
                confidence_mean=confidence_stats['mean'],
                confidence_std=confidence_stats['std'],
                confidence_min=confidence_stats['min'],
                confidence_max=confidence_stats['max'],
                
                # Additional metrics
                temporal_precision=evaluation['precision_2yr'],  # Using 2yr as temporal precision
                temporal_recall=evaluation['recall_2yr'],       # Using 2yr as temporal recall
                evidence_quality_score=self._calculate_evidence_quality(detected_shifts)
            )
            
            print(f"    ✅ {domain_name}: F1={validation_result.f1_score_2yr:.3f}, "
                  f"Precision={validation_result.precision_2yr:.3f}, "
                  f"Recall={validation_result.recall_2yr:.3f}")
            print(f"        Detected: {len(detected_shifts)} shifts, "
                  f"Ground truth: {len(ground_truth['paradigm_shifts'])} shifts")
            
            return validation_result
            
        except Exception as e:
            print(f"    ⚠️ Error evaluating {domain_name}: {e}")
            return self._create_empty_validation_result(domain_name)
    
    def _run_baseline_comparisons(self) -> Dict[str, Any]:
        """Run baseline method comparisons."""
        
        baseline_methods = {
            'random': self._test_random_baseline,
            'equal_interval': self._test_equal_interval_baseline,
            'citation_only': self._test_citation_only_baseline,
            'direction_only': self._test_direction_only_baseline
        }
        
        baseline_results = {}
        
        for method_name, method_func in baseline_methods.items():
            print(f"  🔍 Testing {method_name} baseline...")
            method_results = {}
            
            for domain in self.available_domains:
                try:
                    baseline_result = method_func(domain)
                    method_results[domain] = asdict(baseline_result)
                    print(f"    {domain}: F1={baseline_result.f1_score_2yr:.3f}")
                except Exception as e:
                    print(f"    ⚠️ Error testing {method_name} on {domain}: {e}")
                    method_results[domain] = self._create_empty_baseline_result(method_name, domain)
            
            baseline_results[method_name] = method_results
        
        return baseline_results
    
    def _test_random_baseline(self, domain_name: str) -> BaselineResult:
        """Test random paradigm shift detection baseline."""
        
        ground_truth = self.ground_truth_data[domain_name]
        gt_shifts = ground_truth['paradigm_shifts']
        
        if not gt_shifts:
            return BaselineResult(
                method_name='random',
                domain_name=domain_name,
                detected_shifts=[],
                precision_2yr=0.0,
                recall_2yr=0.0,
                f1_score_2yr=0.0
            )
        
        # Generate random shifts with same count as ground truth
        time_range = ground_truth['temporal_coverage']
        min_year = time_range['start_year']
        max_year = time_range['end_year']
        
        # Generate random years
        num_shifts = len(gt_shifts)
        random_shifts = sorted(random.sample(range(min_year, max_year + 1), 
                                           min(num_shifts, max_year - min_year + 1)))
        
        # Evaluate random baseline
        evaluation = evaluate_against_ground_truth(
            [type('MockShift', (), {'year': year})() for year in random_shifts],
            gt_shifts
        )
        
        return BaselineResult(
            method_name='random',
            domain_name=domain_name,
            detected_shifts=random_shifts,
            precision_2yr=evaluation['precision_2yr'],
            recall_2yr=evaluation['recall_2yr'],
            f1_score_2yr=evaluation['f1_score_2yr']
        )
    
    def _test_equal_interval_baseline(self, domain_name: str) -> BaselineResult:
        """Test equal interval segmentation baseline."""
        
        ground_truth = self.ground_truth_data[domain_name]
        gt_shifts = ground_truth['paradigm_shifts']
        
        if not gt_shifts:
            return BaselineResult(
                method_name='equal_interval',
                domain_name=domain_name,
                detected_shifts=[],
                precision_2yr=0.0,
                recall_2yr=0.0,
                f1_score_2yr=0.0
            )
        
        # Create equal intervals
        time_range = ground_truth['temporal_coverage']
        min_year = time_range['start_year']
        max_year = time_range['end_year']
        
        # Calculate interval size based on ground truth shift count
        num_intervals = len(gt_shifts) + 1  # n shifts create n+1 intervals
        interval_size = (max_year - min_year) / num_intervals
        
        # Generate equal interval shifts
        equal_shifts = []
        for i in range(1, num_intervals):
            shift_year = int(min_year + i * interval_size)
            equal_shifts.append(shift_year)
        
        # Evaluate equal interval baseline
        evaluation = evaluate_against_ground_truth(
            [type('MockShift', (), {'year': year})() for year in equal_shifts],
            gt_shifts
        )
        
        return BaselineResult(
            method_name='equal_interval',
            domain_name=domain_name,
            detected_shifts=equal_shifts,
            precision_2yr=evaluation['precision_2yr'],
            recall_2yr=evaluation['recall_2yr'],
            f1_score_2yr=evaluation['f1_score_2yr']
        )
    
    def _test_citation_only_baseline(self, domain_name: str) -> BaselineResult:
        """Test citation-only detection baseline."""
        
        try:
            # Load domain data
            result = process_domain_data(domain_name)
            if not result.success:
                return self._create_empty_baseline_result('citation_only', domain_name)
            
            domain_data = result.domain_data
            ground_truth = self.ground_truth_data[domain_name]
            
            # Run algorithm with citation-only (disable direction detection)
            config = ComprehensiveAlgorithmConfig(granularity=3)
            config.direction_threshold = 1.0  # Very high threshold to disable direction detection
            config.validation_threshold = 0.5  # Lower validation threshold
            
            detected_shifts, _, _ = detect_shift_signals(
                domain_data, domain_name, config, 
                use_direction=False, use_citation=True
            )
            
            # Evaluate citation-only baseline
            evaluation = evaluate_against_ground_truth(detected_shifts, ground_truth['paradigm_shifts'])
            
            return BaselineResult(
                method_name='citation_only',
                domain_name=domain_name,
                detected_shifts=[s.year for s in detected_shifts],
                precision_2yr=evaluation['precision_2yr'],
                recall_2yr=evaluation['recall_2yr'],
                f1_score_2yr=evaluation['f1_score_2yr']
            )
            
        except Exception as e:
            print(f"      ⚠️ Citation-only baseline failed for {domain_name}: {e}")
            return self._create_empty_baseline_result('citation_only', domain_name)
    
    def _test_direction_only_baseline(self, domain_name: str) -> BaselineResult:
        """Test direction-only detection baseline."""
        
        try:
            # Load domain data
            result = process_domain_data(domain_name)
            if not result.success:
                return self._create_empty_baseline_result('direction_only', domain_name)
            
            domain_data = result.domain_data
            ground_truth = self.ground_truth_data[domain_name]
            
            # Run algorithm with direction-only (disable citation validation)
            config = ComprehensiveAlgorithmConfig(granularity=3)
            config.citation_boost = 0.0  # No citation boost
            config.validation_threshold = 0.4  # Lower validation threshold
            
            detected_shifts, _, _ = detect_shift_signals(
                domain_data, domain_name, config,
                use_direction=True, use_citation=False
            )
            
            # Evaluate direction-only baseline
            evaluation = evaluate_against_ground_truth(detected_shifts, ground_truth['paradigm_shifts'])
            
            return BaselineResult(
                method_name='direction_only',
                domain_name=domain_name,
                detected_shifts=[s.year for s in detected_shifts],
                precision_2yr=evaluation['precision_2yr'],
                recall_2yr=evaluation['recall_2yr'],
                f1_score_2yr=evaluation['f1_score_2yr']
            )
            
        except Exception as e:
            print(f"      ⚠️ Direction-only baseline failed for {domain_name}: {e}")
            return self._create_empty_baseline_result('direction_only', domain_name)
    
    def _analyze_cross_domain_performance(self, domain_results: Dict[str, ValidationResult]) -> CrossDomainAnalysis:
        """Analyze performance patterns across domains."""
        
        # Calculate overall metrics (weighted by number of ground truth shifts)
        total_shifts = sum(len(result.ground_truth_shifts) for result in domain_results.values())
        
        if total_shifts == 0:
            return CrossDomainAnalysis(
                overall_f1_2yr=0.0,
                overall_precision_2yr=0.0,
                overall_recall_2yr=0.0,
                domain_f1_scores={},
                best_performing_domain="none",
                worst_performing_domain="none",
                domain_type_performance={},
                performance_variance=0.0
            )
        
        # Weighted averages
        weighted_f1 = sum(result.f1_score_2yr * len(result.ground_truth_shifts) 
                         for result in domain_results.values()) / total_shifts
        weighted_precision = sum(result.precision_2yr * len(result.ground_truth_shifts) 
                               for result in domain_results.values()) / total_shifts
        weighted_recall = sum(result.recall_2yr * len(result.ground_truth_shifts) 
                            for result in domain_results.values()) / total_shifts
        
        # Domain F1 scores
        domain_f1_scores = {domain: result.f1_score_2yr for domain, result in domain_results.items()}
        
        # Best and worst performing domains
        best_domain = max(domain_f1_scores.items(), key=lambda x: x[1])
        worst_domain = min(domain_f1_scores.items(), key=lambda x: x[1])
        
        # Domain type performance
        domain_type_performance = {}
        for category, domains in self.domain_categories.items():
            category_scores = [domain_f1_scores[domain] for domain in domains if domain in domain_f1_scores]
            if category_scores:
                domain_type_performance[category] = np.mean(category_scores)
        
        # Performance variance
        performance_variance = np.var(list(domain_f1_scores.values()))
        
        return CrossDomainAnalysis(
            overall_f1_2yr=weighted_f1,
            overall_precision_2yr=weighted_precision,
            overall_recall_2yr=weighted_recall,
            domain_f1_scores=domain_f1_scores,
            best_performing_domain=best_domain[0],
            worst_performing_domain=worst_domain[0],
            domain_type_performance=domain_type_performance,
            performance_variance=performance_variance
        )
    
    def _analyze_configuration_impact(self, domain_results: Dict[str, ValidationResult]) -> Dict[str, Any]:
        """Analyze configuration parameter impact on performance."""
        
        # For now, analyze default configuration impact
        # In future, could test multiple configurations per domain
        
        config_analysis = {
            'default_configuration': {
                'granularity': 3,
                'direction_threshold': 0.4,
                'clustering_window': 3,
                'validation_threshold': 0.8,
                'citation_boost': 0.3
            },
            'configuration_performance': {},
            'parameter_sensitivity_insights': {}
        }
        
        # Analyze performance by domain characteristics
        for domain, result in domain_results.items():
            domain_info = self.ground_truth_data[domain]
            characteristics = domain_info.get('period_characteristics', {})
            
            config_analysis['configuration_performance'][domain] = {
                'f1_score': result.f1_score_2yr,
                'precision': result.precision_2yr,
                'recall': result.recall_2yr,
                'domain_stability': characteristics.get('stability', 'unknown'),
                'avg_period_length': characteristics.get('avg_period_length', 0)
            }
        
        return config_analysis
    
    def _test_statistical_significance(self, domain_results: Dict[str, ValidationResult], 
                                     baseline_results: Dict[str, Any]) -> Dict[str, Any]:
        """Test statistical significance of algorithm performance vs baselines."""
        
        statistical_tests = {}
        
        # Collect algorithm F1 scores
        algorithm_f1_scores = [result.f1_score_2yr for result in domain_results.values()]
        
        # Test against each baseline
        for baseline_name, baseline_data in baseline_results.items():
            baseline_f1_scores = []
            
            for domain in self.available_domains:
                if domain in baseline_data and 'f1_score_2yr' in baseline_data[domain]:
                    baseline_f1_scores.append(baseline_data[domain]['f1_score_2yr'])
                else:
                    baseline_f1_scores.append(0.0)
            
            # Perform paired t-test
            if len(algorithm_f1_scores) == len(baseline_f1_scores) and len(algorithm_f1_scores) > 1:
                try:
                    t_stat, p_value = stats.ttest_rel(algorithm_f1_scores, baseline_f1_scores)
                    
                    statistical_tests[baseline_name] = {
                        'algorithm_mean_f1': np.mean(algorithm_f1_scores),
                        'baseline_mean_f1': np.mean(baseline_f1_scores),
                        'improvement': np.mean(algorithm_f1_scores) - np.mean(baseline_f1_scores),
                        't_statistic': float(t_stat),
                        'p_value': float(p_value),
                        'significant': p_value < 0.05,
                        'effect_size': (np.mean(algorithm_f1_scores) - np.mean(baseline_f1_scores)) / np.std(baseline_f1_scores) if np.std(baseline_f1_scores) > 0 else 0
                    }
                except Exception as e:
                    print(f"    ⚠️ Statistical test failed for {baseline_name}: {e}")
                    statistical_tests[baseline_name] = {'error': str(e)}
            else:
                statistical_tests[baseline_name] = {'error': 'Insufficient data for statistical test'}
        
        return statistical_tests
    
    def _validate_original_claims(self, domain_results: Dict[str, ValidationResult]) -> Dict[str, Any]:
        """Validate original algorithm performance claims."""
        
        # Calculate overall metrics
        f1_scores = [result.f1_score_2yr for result in domain_results.values()]
        precision_scores = [result.precision_2yr for result in domain_results.values()]
        recall_scores = [result.recall_2yr for result in domain_results.values()]
        
        overall_f1 = np.mean(f1_scores) if f1_scores else 0.0
        overall_precision = np.mean(precision_scores) if precision_scores else 0.0
        overall_recall = np.mean(recall_scores) if recall_scores else 0.0
        
        # Calculate "accuracy" as (precision + recall) / 2 
        overall_accuracy = (overall_precision + overall_recall) / 2
        
        claims_validation = {
            'original_claims': {
                'accuracy_claim': 0.947,  # "94.7% accuracy"
                'f1_claim': 0.437,        # "F1=0.437"
                'ensemble_f1_claim': 0.355  # "vs ensemble F1=0.355"
            },
            'measured_performance': {
                'overall_accuracy': overall_accuracy,
                'overall_f1': overall_f1,
                'overall_precision': overall_precision,
                'overall_recall': overall_recall,
                'domain_count': len(domain_results),
                'total_paradigm_shifts': sum(len(r.ground_truth_shifts) for r in domain_results.values())
            },
            'claim_validation': {
                'accuracy_validated': overall_accuracy >= 0.947,
                'f1_validated': overall_f1 >= 0.437,
                'accuracy_difference': overall_accuracy - 0.947,
                'f1_difference': overall_f1 - 0.437
            },
            'validation_summary': {
                'claims_supported': 0,
                'claims_refuted': 0,
                'claims_inconclusive': 0
            }
        }
        
        # Validate each claim
        if overall_accuracy >= 0.947:
            claims_validation['validation_summary']['claims_supported'] += 1
        else:
            claims_validation['validation_summary']['claims_refuted'] += 1
            
        if overall_f1 >= 0.437:
            claims_validation['validation_summary']['claims_supported'] += 1
        else:
            claims_validation['validation_summary']['claims_refuted'] += 1
        
        return claims_validation
    
    def _generate_executive_summary(self, validation_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate executive summary with key findings."""
        
        cross_domain = validation_results['cross_domain_analysis']
        claims_validation = validation_results['performance_claims_validation']
        statistical_tests = validation_results['statistical_significance']
        
        summary = {
            'overall_performance': {
                'f1_score': cross_domain['overall_f1_2yr'],
                'precision': cross_domain['overall_precision_2yr'],
                'recall': cross_domain['overall_recall_2yr'],
                'domains_evaluated': len(validation_results['domain_evaluations']),
                'total_paradigm_shifts': validation_results['metadata']['total_ground_truth_shifts']
            },
            'key_findings': {
                'best_performing_domain': cross_domain['best_performing_domain'],
                'worst_performing_domain': cross_domain['worst_performing_domain'],
                'performance_variance': cross_domain['performance_variance'],
                'statistically_significant_baselines': []
            },
            'claims_assessment': {
                'accuracy_claim_status': 'SUPPORTED' if claims_validation['claim_validation']['accuracy_validated'] else 'REFUTED',
                'f1_claim_status': 'SUPPORTED' if claims_validation['claim_validation']['f1_validated'] else 'REFUTED',
                'overall_claims_supported': claims_validation['validation_summary']['claims_supported'],
                'overall_claims_refuted': claims_validation['validation_summary']['claims_refuted']
            },
            'recommendations': {
                'algorithm_strengths': [],
                'improvement_areas': [],
                'configuration_recommendations': []
            }
        }
        
        # Identify statistically significant improvements
        for baseline, test_result in statistical_tests.items():
            if isinstance(test_result, dict) and test_result.get('significant', False):
                summary['key_findings']['statistically_significant_baselines'].append({
                    'baseline': baseline,
                    'improvement': test_result['improvement'],
                    'p_value': test_result['p_value']
                })
        
        # Generate recommendations
        if cross_domain['overall_f1_2yr'] > 0.5:
            summary['recommendations']['algorithm_strengths'].append('Strong overall performance across domains')
        
        if cross_domain['performance_variance'] > 0.1:
            summary['recommendations']['improvement_areas'].append('High performance variance across domains')
        
        return summary
    
    def _generate_validation_report(self, validation_results: Dict[str, Any]) -> str:
        """Generate human-readable validation report."""
        
        report_path = self.output_dir / f"validation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        
        summary = validation_results['executive_summary']
        cross_domain = validation_results['cross_domain_analysis']
        claims = validation_results['performance_claims_validation']
        
        report_content = f"""# Comprehensive Ground Truth Validation Report

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Domains Evaluated:** {summary['overall_performance']['domains_evaluated']}
**Total Ground Truth Paradigm Shifts:** {summary['overall_performance']['total_paradigm_shifts']}

## Executive Summary

### Overall Performance
- **F1 Score:** {summary['overall_performance']['f1_score']:.3f}
- **Precision:** {summary['overall_performance']['precision']:.3f}  
- **Recall:** {summary['overall_performance']['recall']:.3f}

### Domain Performance
- **Best Performing:** {summary['key_findings']['best_performing_domain']} 
  (F1: {cross_domain['domain_f1_scores'][summary['key_findings']['best_performing_domain']]:.3f})
- **Most Challenging:** {summary['key_findings']['worst_performing_domain']} 
  (F1: {cross_domain['domain_f1_scores'][summary['key_findings']['worst_performing_domain']]:.3f})
- **Performance Variance:** {summary['key_findings']['performance_variance']:.3f}

## Performance Claims Validation

### Original Claims vs Measured Performance
- **Accuracy Claim:** 94.7% → **{claims['claim_validation']['accuracy_validated']}** 
  (Measured: {claims['measured_performance']['overall_accuracy']:.1%})
- **F1 Score Claim:** 0.437 → **{claims['claim_validation']['f1_validated']}** 
  (Measured: {claims['measured_performance']['overall_f1']:.3f})

### Claims Assessment
- **Claims Supported:** {claims['validation_summary']['claims_supported']}
- **Claims Refuted:** {claims['validation_summary']['claims_refuted']}

## Domain-by-Domain Results

"""
        
        # Add domain-specific results
        for domain, result in validation_results['domain_evaluations'].items():
            report_content += f"""### {domain.replace('_', ' ').title()}
- **F1 Score:** {result['f1_score_2yr']:.3f}
- **Precision:** {result['precision_2yr']:.3f}
- **Recall:** {result['recall_2yr']:.3f}
- **Detected Shifts:** {result['paradigm_count_detected']}
- **Ground Truth Shifts:** {result['paradigm_count_ground_truth']}

"""
        
        # Add statistical significance results
        report_content += f"""## Statistical Significance Testing

"""
        
        for baseline, test_result in validation_results['statistical_significance'].items():
            if isinstance(test_result, dict) and 'p_value' in test_result:
                significance = "✅ SIGNIFICANT" if test_result['significant'] else "❌ NOT SIGNIFICANT"
                report_content += f"""### vs {baseline.replace('_', ' ').title()} Baseline
- **Algorithm F1:** {test_result['algorithm_mean_f1']:.3f}
- **Baseline F1:** {test_result['baseline_mean_f1']:.3f}  
- **Improvement:** {test_result['improvement']:.3f}
- **P-value:** {test_result['p_value']:.6f}
- **Result:** {significance}

"""
        
        # Add recommendations
        report_content += f"""## Recommendations

### Algorithm Strengths
"""
        for strength in summary['recommendations']['algorithm_strengths']:
            report_content += f"- {strength}\n"
        
        report_content += f"""
### Areas for Improvement
"""
        for improvement in summary['recommendations']['improvement_areas']:
            report_content += f"- {improvement}\n"
        
        # Save report
        with open(report_path, 'w') as f:
            f.write(report_content)
        
        print(f"   📋 Human-readable report: {report_path}")
        
        return str(report_path)
    
    # Helper methods
    def _count_total_paradigm_shifts(self) -> int:
        """Count total paradigm shifts across all domains."""
        total = 0
        for domain_data in self.ground_truth_data.values():
            if 'paradigm_shifts' in domain_data:
                total += len(domain_data['paradigm_shifts'])
        return total
    
    def _calculate_confidence_statistics(self, detected_shifts) -> Dict[str, float]:
        """Calculate confidence statistics for detected shifts."""
        if not detected_shifts:
            return {'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0}
        
        confidences = [s.confidence for s in detected_shifts]
        return {
            'mean': float(np.mean(confidences)),
            'std': float(np.std(confidences)),
            'min': float(np.min(confidences)),
            'max': float(np.max(confidences))
        }
    
    def _calculate_evidence_quality(self, detected_shifts) -> float:
        """Calculate evidence quality score for detected shifts."""
        if not detected_shifts:
            return 0.0
        
        # Simple evidence quality based on evidence count and confidence
        evidence_scores = []
        for shift in detected_shifts:
            evidence_count = len(shift.supporting_evidence)
            confidence = shift.confidence
            evidence_score = min(1.0, evidence_count / 5.0) * confidence
            evidence_scores.append(evidence_score)
        
        return float(np.mean(evidence_scores))
    
    def _create_empty_validation_result(self, domain_name: str) -> ValidationResult:
        """Create empty validation result for failed domain evaluation."""
        return ValidationResult(
            domain_name=domain_name,
            ground_truth_shifts=[],
            detected_shifts=[],
            configuration={},
            precision_2yr=0.0,
            recall_2yr=0.0,
            f1_score_2yr=0.0,
            precision_5yr=0.0,
            recall_5yr=0.0,
            f1_score_5yr=0.0,
            paradigm_count_detected=0,
            paradigm_count_ground_truth=0,
            count_accuracy=0.0,
            confidence_mean=0.0,
            confidence_std=0.0,
            confidence_min=0.0,
            confidence_max=0.0,
            temporal_precision=0.0,
            temporal_recall=0.0,
            evidence_quality_score=0.0
        )
    
    def _create_empty_baseline_result(self, method_name: str, domain_name: str) -> Dict[str, Any]:
        """Create empty baseline result for failed baseline evaluation."""
        return {
            'method_name': method_name,
            'domain_name': domain_name,
            'detected_shifts': [],
            'precision_2yr': 0.0,
            'recall_2yr': 0.0,
            'f1_score_2yr': 0.0
        }


def main():
    """Run comprehensive ground truth validation."""
    
    print("🔍 COMPREHENSIVE GROUND TRUTH VALIDATION")
    print("=" * 50)
    
    validator = ComprehensiveValidationFramework()
    
    results = validator.run_comprehensive_validation()
    
    # Print key results
    summary = results['executive_summary']
    print(f"\n📋 VALIDATION SUMMARY:")
    print(f"   🎯 Overall F1 Score: {summary['overall_performance']['f1_score']:.3f}")
    print(f"   📈 Overall Precision: {summary['overall_performance']['precision']:.3f}")
    print(f"   📊 Overall Recall: {summary['overall_performance']['recall']:.3f}")
    print(f"   🏆 Best Domain: {summary['key_findings']['best_performing_domain']}")
    print(f"   🎯 Claims Supported: {summary['claims_assessment']['overall_claims_supported']}")
    print(f"   ❌ Claims Refuted: {summary['claims_assessment']['overall_claims_refuted']}")
    
    return results


if __name__ == "__main__":
    main() 