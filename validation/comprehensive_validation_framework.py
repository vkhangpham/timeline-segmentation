"""
Comprehensive Ground Truth Validation Framework

Systematic evaluation of the Timeline Segmentation Algorithm against all available
ground truth data with statistical testing and baseline comparisons.
"""

import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass, asdict
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
    
    # Core metrics
    precision_2yr: float
    recall_2yr: float
    f1_score_2yr: float
    
    # Count accuracy
    paradigm_count_detected: int
    paradigm_count_ground_truth: int
    count_accuracy: float
    
    # Confidence analysis
    confidence_mean: float
    confidence_std: float


@dataclass
class CrossDomainAnalysis:
    """Cross-domain performance analysis."""
    overall_f1_2yr: float
    overall_precision_2yr: float
    overall_recall_2yr: float
    domain_f1_scores: Dict[str, float]
    best_performing_domain: str
    worst_performing_domain: str
    performance_variance: float


class ComprehensiveValidationFramework:
    """
    Comprehensive validation framework for systematic algorithm assessment.
    
    Provides rigorous evaluation against all available ground truth domains
    with statistical significance testing and baseline comparisons.
    """
    
    def __init__(self, output_dir: str = "results/validation"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load ground truth data
        self.ground_truth_data = load_ground_truth_data()
        self.available_domains = list(self.ground_truth_data.keys())
        
        print(f"🔍 Comprehensive Validation Framework initialized")
        print(f"   📁 Output directory: {self.output_dir}")
        print(f"   🌍 Domains available: {len(self.available_domains)}")
        print(f"   📊 Total ground truth paradigm shifts: {self._count_total_paradigm_shifts()}")
    
    def run_comprehensive_validation(self) -> Dict[str, Any]:
        """
        Run systematic validation across all domains.
        
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
                'validation_methodology': 'Comprehensive Ground Truth Assessment'
            },
            'domain_evaluations': {},
            'baseline_comparisons': {},
            'cross_domain_analysis': {},
            'statistical_significance': {},
            'performance_claims_validation': {},
            'executive_summary': {}
        }
        
        # Phase 1: Domain evaluation
        print(f"\n📊 PHASE 1: Domain Evaluation")
        domain_results = {}
        for domain in self.available_domains:
            print(f"\n🔍 Evaluating domain: {domain}")
            domain_result = self._evaluate_domain(domain)
            domain_results[domain] = domain_result
            results['domain_evaluations'][domain] = asdict(domain_result)
        
        # Phase 2: Baseline comparisons
        print(f"\n📈 PHASE 2: Baseline Comparisons")
        baseline_results = self._run_baseline_comparisons()
        results['baseline_comparisons'] = baseline_results
        
        # Phase 3: Cross-domain analysis
        print(f"\n🌐 PHASE 3: Cross-Domain Analysis")
        cross_domain_analysis = self._analyze_cross_domain_performance(domain_results)
        results['cross_domain_analysis'] = asdict(cross_domain_analysis)
        
        # Phase 4: Statistical significance
        print(f"\n📊 PHASE 4: Statistical Testing")
        statistical_analysis = self._test_statistical_significance(domain_results, baseline_results)
        results['statistical_significance'] = statistical_analysis
        
        # Phase 5: Claims validation
        print(f"\n🎯 PHASE 5: Claims Validation")
        claims_validation = self._validate_original_claims(domain_results)
        results['performance_claims_validation'] = claims_validation
        
        # Phase 6: Executive summary
        print(f"\n📋 PHASE 6: Executive Summary")
        executive_summary = self._generate_executive_summary(results)
        results['executive_summary'] = executive_summary
        
        # Save results
        validation_duration = time.time() - validation_start
        results['metadata']['validation_duration_seconds'] = validation_duration
        
        output_file = self.output_dir / f"comprehensive_validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Generate report
        report_file = self._generate_validation_report(results)
        
        print(f"\n✅ COMPREHENSIVE VALIDATION COMPLETE")
        print(f"   ⏱️  Duration: {validation_duration:.1f} seconds")
        print(f"   📁 Results: {output_file}")
        print(f"   📋 Report: {report_file}")
        
        return results
    
    def _evaluate_domain(self, domain_name: str) -> ValidationResult:
        """Evaluate algorithm performance on single domain."""
        
        try:
            # Load domain data
            result = process_domain_data(domain_name)
            if not result.success:
                print(f"    ⚠️ Error loading {domain_name}: {result.error_message}")
                return self._create_empty_result(domain_name)
            
            domain_data = result.domain_data
            ground_truth = self.ground_truth_data[domain_name]
            
            # Run algorithm
            config = ComprehensiveAlgorithmConfig(granularity=3)
            detected_shifts, _, _ = detect_shift_signals(domain_data, domain_name, config)
            
            # Evaluate performance
            evaluation = evaluate_against_ground_truth(detected_shifts, ground_truth['paradigm_shifts'])
            
            # Calculate confidence statistics
            confidence_stats = self._calculate_confidence_statistics(detected_shifts)
            
            validation_result = ValidationResult(
                domain_name=domain_name,
                ground_truth_shifts=ground_truth['paradigm_shifts'],
                detected_shifts=[s.year for s in detected_shifts],
                precision_2yr=evaluation['precision_2yr'],
                recall_2yr=evaluation['recall_2yr'],
                f1_score_2yr=evaluation['f1_score_2yr'],
                paradigm_count_detected=evaluation['paradigm_count_detected'],
                paradigm_count_ground_truth=evaluation['paradigm_count_ground_truth'],
                count_accuracy=evaluation['count_accuracy'],
                confidence_mean=confidence_stats['mean'],
                confidence_std=confidence_stats['std']
            )
            
            print(f"    ✅ {domain_name}: F1={validation_result.f1_score_2yr:.3f}, "
                  f"P={validation_result.precision_2yr:.3f}, R={validation_result.recall_2yr:.3f}")
            print(f"        Detected: {len(detected_shifts)}, Ground truth: {len(ground_truth['paradigm_shifts'])}")
            
            return validation_result
            
        except Exception as e:
            print(f"    ⚠️ Error evaluating {domain_name}: {e}")
            return self._create_empty_result(domain_name)
    
    def _run_baseline_comparisons(self) -> Dict[str, Any]:
        """Run baseline method comparisons."""
        
        baseline_results = {}
        
        # Random baseline
        print(f"  🎲 Testing random baseline...")
        random_results = {}
        for domain in self.available_domains:
            try:
                random_f1 = self._test_random_baseline(domain)
                random_results[domain] = {'f1_score_2yr': random_f1}
                print(f"    {domain}: {random_f1:.3f}")
            except Exception as e:
                print(f"    ⚠️ {domain}: Error - {e}")
                random_results[domain] = {'f1_score_2yr': 0.0}
        
        baseline_results['random'] = random_results
        
        # Equal interval baseline
        print(f"  📏 Testing equal interval baseline...")
        interval_results = {}
        for domain in self.available_domains:
            try:
                interval_f1 = self._test_equal_interval_baseline(domain)
                interval_results[domain] = {'f1_score_2yr': interval_f1}
                print(f"    {domain}: {interval_f1:.3f}")
            except Exception as e:
                print(f"    ⚠️ {domain}: Error - {e}")
                interval_results[domain] = {'f1_score_2yr': 0.0}
        
        baseline_results['equal_interval'] = interval_results
        
        return baseline_results
    
    def _test_random_baseline(self, domain_name: str) -> float:
        """Test random paradigm shift detection baseline."""
        
        ground_truth = self.ground_truth_data[domain_name]
        gt_shifts = ground_truth['paradigm_shifts']
        
        if not gt_shifts:
            return 0.0
        
        # Generate random shifts
        time_range = ground_truth['temporal_coverage']
        min_year = time_range['start_year']
        max_year = time_range['end_year']
        
        num_shifts = len(gt_shifts)
        random_years = sorted(random.sample(range(min_year, max_year + 1), 
                                          min(num_shifts, max_year - min_year + 1)))
        
        # Evaluate
        evaluation = evaluate_against_ground_truth(
            [type('MockShift', (), {'year': year})() for year in random_years],
            gt_shifts
        )
        
        return evaluation['f1_score_2yr']
    
    def _test_equal_interval_baseline(self, domain_name: str) -> float:
        """Test equal interval segmentation baseline."""
        
        ground_truth = self.ground_truth_data[domain_name]
        gt_shifts = ground_truth['paradigm_shifts']
        
        if not gt_shifts:
            return 0.0
        
        # Create equal intervals
        time_range = ground_truth['temporal_coverage']
        min_year = time_range['start_year']
        max_year = time_range['end_year']
        
        num_intervals = len(gt_shifts) + 1
        interval_size = (max_year - min_year) / num_intervals
        
        equal_shifts = []
        for i in range(1, num_intervals):
            shift_year = int(min_year + i * interval_size)
            equal_shifts.append(shift_year)
        
        # Evaluate
        evaluation = evaluate_against_ground_truth(
            [type('MockShift', (), {'year': year})() for year in equal_shifts],
            gt_shifts
        )
        
        return evaluation['f1_score_2yr']
    
    def _analyze_cross_domain_performance(self, domain_results: Dict[str, ValidationResult]) -> CrossDomainAnalysis:
        """Analyze performance patterns across domains."""
        
        # Calculate weighted averages
        total_shifts = sum(len(result.ground_truth_shifts) for result in domain_results.values())
        
        if total_shifts == 0:
            return CrossDomainAnalysis(
                overall_f1_2yr=0.0,
                overall_precision_2yr=0.0,
                overall_recall_2yr=0.0,
                domain_f1_scores={},
                best_performing_domain="none",
                worst_performing_domain="none",
                performance_variance=0.0
            )
        
        weighted_f1 = sum(result.f1_score_2yr * len(result.ground_truth_shifts) 
                         for result in domain_results.values()) / total_shifts
        weighted_precision = sum(result.precision_2yr * len(result.ground_truth_shifts) 
                               for result in domain_results.values()) / total_shifts
        weighted_recall = sum(result.recall_2yr * len(result.ground_truth_shifts) 
                            for result in domain_results.values()) / total_shifts
        
        # Domain F1 scores
        domain_f1_scores = {domain: result.f1_score_2yr for domain, result in domain_results.items()}
        
        # Best and worst domains
        best_domain = max(domain_f1_scores.items(), key=lambda x: x[1])[0]
        worst_domain = min(domain_f1_scores.items(), key=lambda x: x[1])[0]
        
        # Performance variance
        performance_variance = np.var(list(domain_f1_scores.values()))
        
        return CrossDomainAnalysis(
            overall_f1_2yr=weighted_f1,
            overall_precision_2yr=weighted_precision,
            overall_recall_2yr=weighted_recall,
            domain_f1_scores=domain_f1_scores,
            best_performing_domain=best_domain,
            worst_performing_domain=worst_domain,
            performance_variance=performance_variance
        )
    
    def _test_statistical_significance(self, domain_results: Dict[str, ValidationResult], 
                                     baseline_results: Dict[str, Any]) -> Dict[str, Any]:
        """Test statistical significance vs baselines."""
        
        statistical_tests = {}
        
        # Algorithm F1 scores
        algorithm_f1_scores = [result.f1_score_2yr for result in domain_results.values()]
        
        # Test against each baseline
        for baseline_name, baseline_data in baseline_results.items():
            baseline_f1_scores = []
            
            for domain in self.available_domains:
                if domain in baseline_data:
                    baseline_f1_scores.append(baseline_data[domain]['f1_score_2yr'])
                else:
                    baseline_f1_scores.append(0.0)
            
            # Paired t-test
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
                        'effect_size': (np.mean(algorithm_f1_scores) - np.mean(baseline_f1_scores)) / max(np.std(baseline_f1_scores), 0.001)
                    }
                except Exception as e:
                    statistical_tests[baseline_name] = {'error': str(e)}
            
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
        overall_accuracy = (overall_precision + overall_recall) / 2
        
        claims_validation = {
            'original_claims': {
                'accuracy_claim': 0.947,  # "94.7% accuracy"
                'f1_claim': 0.437,        # "F1=0.437"
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
            }
        }
        
        return claims_validation
    
    def _generate_executive_summary(self, validation_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate executive summary."""
        
        cross_domain = validation_results['cross_domain_analysis']
        claims = validation_results['performance_claims_validation']
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
                'accuracy_claim_status': 'SUPPORTED' if claims['claim_validation']['accuracy_validated'] else 'REFUTED',
                'f1_claim_status': 'SUPPORTED' if claims['claim_validation']['f1_validated'] else 'REFUTED',
            }
        }
        
        # Identify significant improvements
        for baseline, test_result in statistical_tests.items():
            if isinstance(test_result, dict) and test_result.get('significant', False):
                summary['key_findings']['statistically_significant_baselines'].append({
                    'baseline': baseline,
                    'improvement': test_result['improvement'],
                    'p_value': test_result['p_value']
                })
        
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
- **Accuracy Claim:** 94.7% → **{summary['claims_assessment']['accuracy_claim_status']}** 
  (Measured: {claims['measured_performance']['overall_accuracy']:.1%})
- **F1 Score Claim:** 0.437 → **{summary['claims_assessment']['f1_claim_status']}** 
  (Measured: {claims['measured_performance']['overall_f1']:.3f})

## Domain-by-Domain Results

"""
        
        # Add domain results
        for domain, result in validation_results['domain_evaluations'].items():
            report_content += f"""### {domain.replace('_', ' ').title()}
- **F1 Score:** {result['f1_score_2yr']:.3f}
- **Precision:** {result['precision_2yr']:.3f}
- **Recall:** {result['recall_2yr']:.3f}
- **Detected:** {result['paradigm_count_detected']} | **Ground Truth:** {result['paradigm_count_ground_truth']}

"""
        
        # Add statistical significance
        report_content += f"""## Statistical Significance Testing

"""
        
        for baseline, test_result in validation_results['statistical_significance'].items():
            if isinstance(test_result, dict) and 'p_value' in test_result:
                significance = "✅ SIGNIFICANT" if test_result['significant'] else "❌ NOT SIGNIFICANT"
                report_content += f"""### vs {baseline.replace('_', ' ').title()} Baseline
- **Algorithm F1:** {test_result['algorithm_mean_f1']:.3f}
- **Baseline F1:** {test_result['baseline_mean_f1']:.3f}  
- **Improvement:** +{test_result['improvement']:.3f}
- **P-value:** {test_result['p_value']:.6f}
- **Result:** {significance}

"""
        
        # Save report
        with open(report_path, 'w') as f:
            f.write(report_content)
        
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
        """Calculate confidence statistics."""
        if not detected_shifts:
            return {'mean': 0.0, 'std': 0.0}
        
        confidences = [s.confidence for s in detected_shifts]
        return {
            'mean': float(np.mean(confidences)),
            'std': float(np.std(confidences))
        }
    
    def _create_empty_result(self, domain_name: str) -> ValidationResult:
        """Create empty result for failed evaluation."""
        return ValidationResult(
            domain_name=domain_name,
            ground_truth_shifts=[],
            detected_shifts=[],
            precision_2yr=0.0,
            recall_2yr=0.0,
            f1_score_2yr=0.0,
            paradigm_count_detected=0,
            paradigm_count_ground_truth=0,
            count_accuracy=0.0,
            confidence_mean=0.0,
            confidence_std=0.0
        )


def main():
    """Run comprehensive validation."""
    
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
    print(f"   📉 Most Challenging: {summary['key_findings']['worst_performing_domain']}")
    print(f"   🎯 Accuracy Claim: {summary['claims_assessment']['accuracy_claim_status']}")
    print(f"   📊 F1 Claim: {summary['claims_assessment']['f1_claim_status']}")
    
    return results


if __name__ == "__main__":
    main() 