"""
Clean Validation Framework

Simple, focused validation of the Timeline Segmentation Algorithm against reference data.
No bullshit - just core metrics.
"""

import numpy as np
import json
from pathlib import Path
from typing import Dict, List
from dataclasses import dataclass, asdict
import time
from datetime import datetime

# Algorithm imports
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.utils.config import AlgorithmConfig
from core.detection.shift_signals import detect_shift_signals
from core.data.processing import process_domain_data
from validation.core import load_reference_data_from_files, evaluate_shifts_against_reference


@dataclass
class ValidationResult:
    """Clean validation result for a single domain."""
    domain_name: str
    reference_shifts: List[int]
    detected_shifts: List[int]
    precision_2yr: float
    recall_2yr: float
    f1_score_2yr: float
    paradigm_count_detected: int
    paradigm_count_reference: int


@dataclass 
class OverallResults:
    """Overall validation results across all domains."""
    overall_f1_2yr: float
    overall_precision_2yr: float
    overall_recall_2yr: float
    domain_results: Dict[str, ValidationResult]
    domains_evaluated: int
    total_paradigm_shifts: int


class ValidationFramework:
    """
    Clean validation framework - no bullshit, just metrics.
    """
    
    def __init__(self):
        # Load reference data
        self.reference_data = load_reference_data_from_files(reference_type="manual")
        self.available_domains = list(self.reference_data.keys())
        
        print(f"🔍 Validation Framework initialized")
        print(f"   🌍 Domains available: {len(self.available_domains)}")
        print(f"   📊 Total reference shifts: {self._count_total_shifts()}")
    
    def run_validation(self) -> OverallResults:
        """Run validation across all domains."""
        print(f"\n🚀 RUNNING VALIDATION")
        print("=" * 50)
        
        start_time = time.time()
        
        # Evaluate each domain
        domain_results = {}
        for domain in self.available_domains:
            print(f"🔍 {domain}")
            result = self._evaluate_domain(domain)
            domain_results[domain] = result
            print(f"    F1={result.f1_score_2yr:.3f}, P={result.precision_2yr:.3f}, R={result.recall_2yr:.3f}")
        
        # Calculate overall metrics
        overall_results = self._calculate_overall_metrics(domain_results)
        
        validation_time = time.time() - start_time
        
        # Save results
        self._save_results(overall_results, validation_time)
        
        print(f"\n✅ VALIDATION COMPLETED")
        print(f"   ⏱️  Time: {validation_time:.1f}s")
        print(f"   📊 Overall F1: {overall_results.overall_f1_2yr:.3f}")
        print(f"   📈 Overall Precision: {overall_results.overall_precision_2yr:.3f}")
        print(f"   📊 Overall Recall: {overall_results.overall_recall_2yr:.3f}")
        
        return overall_results
    
    def _evaluate_domain(self, domain_name: str) -> ValidationResult:
        """Evaluate algorithm performance on single domain."""
        
        # Load domain data - FAIL-FAST
        result = process_domain_data(domain_name)
        if not result.success:
            raise RuntimeError(f"Error loading {domain_name}: {result.error_message}")
        
        domain_data = result.domain_data
        reference = self.reference_data[domain_name]
        
        # Run algorithm with domain-specific optimized parameters
        config = AlgorithmConfig(granularity=3, domain_name=domain_name)
        detected_shifts = detect_shift_signals(domain_data, domain_name, config)
        
        # Evaluate performance
        evaluation = evaluate_shifts_against_reference(detected_shifts, reference.paradigm_shifts)
        
        return ValidationResult(
            domain_name=domain_name,
            reference_shifts=list(reference.paradigm_shifts),
            detected_shifts=[s.year for s in detected_shifts],
            precision_2yr=evaluation.precision_2yr,
            recall_2yr=evaluation.recall_2yr,
            f1_score_2yr=evaluation.f1_score_2yr,
            paradigm_count_detected=evaluation.paradigm_count_detected,
            paradigm_count_reference=evaluation.paradigm_count_reference
        )
    
    def _calculate_overall_metrics(self, domain_results: Dict[str, ValidationResult]) -> OverallResults:
        """Calculate weighted overall metrics."""
        
        # Calculate weighted averages by number of reference shifts
        total_shifts = sum(len(result.reference_shifts) for result in domain_results.values())
        
        if total_shifts == 0:
            return OverallResults(
                overall_f1_2yr=0.0,
                overall_precision_2yr=0.0,
                overall_recall_2yr=0.0,
                domain_results=domain_results,
                domains_evaluated=len(domain_results),
                total_paradigm_shifts=0
            )
        
        weighted_f1 = sum(result.f1_score_2yr * len(result.reference_shifts) 
                         for result in domain_results.values()) / total_shifts
        weighted_precision = sum(result.precision_2yr * len(result.reference_shifts) 
                               for result in domain_results.values()) / total_shifts
        weighted_recall = sum(result.recall_2yr * len(result.reference_shifts) 
                            for result in domain_results.values()) / total_shifts
        
        return OverallResults(
            overall_f1_2yr=weighted_f1,
            overall_precision_2yr=weighted_precision,
            overall_recall_2yr=weighted_recall,
            domain_results=domain_results,
            domains_evaluated=len(domain_results),
            total_paradigm_shifts=total_shifts
        )
    
    def _save_results(self, results: OverallResults, validation_time: float):
        """Save clean validation results."""
        output_dir = Path("results/validation")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_data = {
            'metadata': {
                'validation_date': datetime.now().isoformat(),
                'validation_time_seconds': validation_time,
                'domains_evaluated': results.domains_evaluated,
                'total_paradigm_shifts': results.total_paradigm_shifts
            },
            'overall_metrics': {
                'f1_score_2yr': results.overall_f1_2yr,
                'precision_2yr': results.overall_precision_2yr,
                'recall_2yr': results.overall_recall_2yr
            },
            'domain_results': {
                domain: asdict(result) for domain, result in results.domain_results.items()
            }
        }
        
        # Save timestamped version
        output_file = output_dir / f"validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        # Save latest version in results root
        latest_path = Path("results/validation.json")
        latest_path.parent.mkdir(parents=True, exist_ok=True)
        with open(latest_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"💾 Results saved: {output_file}")
        print(f"💾 Latest version saved: {latest_path}")
    
    def _count_total_shifts(self) -> int:
        """Count total paradigm shifts across all domains."""
        return sum(len(data.paradigm_shifts) for data in self.reference_data.values())


def main():
    """Run clean validation."""
    
    print("🔍 CLEAN VALIDATION FRAMEWORK")
    print("=" * 40)
    
    validator = ValidationFramework()
    results = validator.run_validation()
    
    return results


if __name__ == "__main__":
    main() 