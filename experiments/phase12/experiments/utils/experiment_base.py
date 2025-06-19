"""
Base experiment class for Phase 12 ablation study.

Provides common functionality for experimental setup, data loading, 
result storage, and statistical analysis following functional programming principles.
"""

import sys
import time
import json
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import numpy as np

# Add project root to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from core.data_processing import process_domain_data
from core.integration import SensitivityConfig


@dataclass
class ExperimentResult:
    """Immutable experimental result data structure."""
    experiment_name: str
    condition: str
    domain: str
    paradigm_shifts_detected: int
    segment_count: int
    temporal_accuracy: float
    computational_time: float
    memory_usage: float
    confidence_scores: List[float]
    segment_lengths: List[int]
    metadata: Dict[str, Any]


@dataclass
class ExperimentCondition:
    """Immutable experimental condition specification."""
    name: str
    description: str
    parameters: Dict[str, Any]


class ExperimentBase:
    """
    Base class for Phase 12 ablation study experiments.
    
    Follows functional programming principles with pure functions
    and immutable data structures for reproducible research.
    """
    
    def __init__(self, experiment_name: str, output_dir: str = None):
        """
        Initialize experiment with proper directory structure.
        
        Args:
            experiment_name: Name of the experiment (e.g., "signal_ablation")
            output_dir: Optional custom output directory
        """
        self.experiment_name = experiment_name
        self.start_time = datetime.now()
        
        # Set up output directories
        if output_dir:
            self.output_dir = Path(output_dir)
        else:
            self.output_dir = Path(__file__).parent.parent.parent / "results" / f"experiment_{experiment_name}"
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Domain configuration
        self.domains = [
            "natural_language_processing",
            "computer_vision", 
            "deep_learning",
            "machine_learning",
            "applied_mathematics",
            "art",
            "computer_science",
            "machine_translation"
        ]
        
        print(f"🔬 PHASE 12 EXPERIMENT: {experiment_name.upper()}")
        print(f"📁 Output directory: {self.output_dir}")
        print(f"🕐 Start time: {self.start_time.isoformat()}")
        print("=" * 60)

    def load_domain_data(self, domain_name: str):
        """
        Load domain data using existing pipeline (pure function).
        
        Args:
            domain_name: Name of domain to load
            
        Returns:
            DomainData object with papers and citations
        """
        print(f"  📊 Loading domain data: {domain_name}")
        start_time = time.time()
        
        processing_result = process_domain_data(domain_name)
        
        if not processing_result.success:
            raise RuntimeError(f"Failed to load domain data for {domain_name}: {processing_result.error_message}")
        
        domain_data = processing_result.domain_data
        load_time = time.time() - start_time
        print(f"    ✅ Loaded {len(domain_data.papers)} papers in {load_time:.2f}s")
        
        return domain_data

    def run_condition(self, condition: ExperimentCondition, domain_name: str) -> ExperimentResult:
        """
        Run a single experimental condition on one domain (pure function).
        
        Args:
            condition: Experimental condition specification
            domain_name: Domain to test on
            
        Returns:
            ExperimentResult with all measurements
        """
        print(f"  🧪 Running condition '{condition.name}' on {domain_name}")
        
        # Load domain data
        domain_data = self.load_domain_data(domain_name)
        
        # Track computational resources
        start_time = time.time()
        
        # This will be overridden by specific experiments
        result = self._execute_condition(condition, domain_data, domain_name)
        
        # Record computational metrics
        computation_time = time.time() - start_time
        
        # Create standardized result
        experiment_result = ExperimentResult(
            experiment_name=self.experiment_name,
            condition=condition.name,
            domain=domain_name,
            paradigm_shifts_detected=result.get('paradigm_shifts_detected', 0),
            segment_count=result.get('segment_count', 0),
            temporal_accuracy=result.get('temporal_accuracy', 0.0),
            computational_time=computation_time,
            memory_usage=result.get('memory_usage', 0.0),
            confidence_scores=result.get('confidence_scores', []),
            segment_lengths=result.get('segment_lengths', []),
            metadata=result.get('metadata', {})
        )
        
        print(f"    ✅ {condition.name}: {experiment_result.paradigm_shifts_detected} paradigm shifts, "
              f"{experiment_result.segment_count} segments, {computation_time:.3f}s")
        
        return experiment_result

    def _execute_condition(self, condition: ExperimentCondition, domain_data, domain_name: str) -> Dict[str, Any]:
        """
        Execute specific experimental condition - override in subclasses.
        
        Args:
            condition: Experimental condition
            domain_data: Domain data
            domain_name: Domain name
            
        Returns:
            Dictionary with experimental results
        """
        raise NotImplementedError("Subclasses must implement _execute_condition")

    def run_experiment(self, conditions: List[ExperimentCondition], 
                      domains: Optional[List[str]] = None) -> List[ExperimentResult]:
        """
        Run complete experiment across all conditions and domains.
        
        Args:
            conditions: List of experimental conditions to test
            domains: Optional list of domains (defaults to all)
            
        Returns:
            List of all experimental results
        """
        if domains is None:
            domains = self.domains
            
        print(f"🚀 Starting experiment with {len(conditions)} conditions across {len(domains)} domains")
        
        all_results = []
        
        for condition in conditions:
            print(f"\n📋 CONDITION: {condition.name}")
            print(f"    Description: {condition.description}")
            
            for domain in domains:
                try:
                    result = self.run_condition(condition, domain)
                    all_results.append(result)
                except Exception as e:
                    print(f"    ❌ ERROR in {domain}: {e}")
                    # Following project guidelines: fail fast, no fallbacks
                    raise RuntimeError(f"Experiment failed for condition {condition.name} on domain {domain}: {e}")
        
        print(f"\n✅ EXPERIMENT COMPLETE: {len(all_results)} results collected")
        return all_results

    def save_results(self, results: List[ExperimentResult], filename: str = None) -> str:
        """
        Save experimental results to JSON file (pure function).
        
        Args:
            results: List of experimental results
            filename: Optional custom filename
            
        Returns:
            Path to saved file
        """
        if filename is None:
            timestamp = self.start_time.strftime("%Y%m%d_%H%M%S")
            filename = f"{self.experiment_name}_results_{timestamp}.json"
        
        output_file = self.output_dir / filename
        
        # Convert results to serializable format
        serialized_results = []
        for result in results:
            result_dict = asdict(result)
            # Ensure all numpy types and infinity values are converted to JSON-safe types
            result_dict = self._sanitize_for_json(result_dict)
            serialized_results.append(result_dict)
        
        # Save with experiment metadata
        output_data = {
            "experiment_metadata": {
                "experiment_name": self.experiment_name,
                "start_time": self.start_time.isoformat(),
                "end_time": datetime.now().isoformat(),
                "total_results": len(results),
                "domains_tested": list(set(r.domain for r in results)),
                "conditions_tested": list(set(r.condition for r in results))
            },
            "results": serialized_results
        }
        
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"💾 Results saved to: {output_file}")
        return str(output_file)

    def _sanitize_for_json(self, obj):
        """
        Recursively sanitize data structure for JSON serialization.
        
        Converts infinity, NaN, and numpy types to JSON-safe values.
        
        Args:
            obj: Object to sanitize
            
        Returns:
            JSON-serializable object
        """
        if isinstance(obj, dict):
            return {key: self._sanitize_for_json(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._sanitize_for_json(item) for item in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            if np.isnan(obj):
                return -1.0  # Use -1.0 for NaN values
            elif np.isinf(obj):
                return -1.0  # Use -1.0 for infinity values
            else:
                return float(obj)
        elif isinstance(obj, float):
            if obj == float('inf') or obj == float('-inf'):
                return -1.0  # Use -1.0 for infinity values
            elif obj != obj:  # Check for NaN (NaN != NaN is True)
                return -1.0  # Use -1.0 for NaN values
            else:
                return obj
        else:
            return obj

    def load_ground_truth(self, domain_name: str) -> Dict[str, Any]:
        """
        Load ground truth data for validation (pure function).
        
        Args:
            domain_name: Domain name
            
        Returns:
            Ground truth paradigm shifts and metadata
        """
        # Use existing validation ground truth files (project root/validation/)
        project_root = Path(__file__).parent.parent.parent.parent.parent.absolute()
        ground_truth_file = project_root / "validation" / f"{domain_name}_groundtruth.json"
        
        if ground_truth_file.exists():
            with open(ground_truth_file, 'r') as f:
                data = json.load(f)
                
            # Extract paradigm shift years from historical periods
            paradigm_shifts = []
            if "historical_periods" in data:
                periods = data["historical_periods"]
                # Paradigm shifts occur at the start of new periods (except the first)
                for i, period in enumerate(periods[1:], 1):  # Skip first period
                    paradigm_shifts.append(period["start_year"])
            
            return {
                "paradigm_shifts": paradigm_shifts,
                "historical_periods": data.get("historical_periods", []),
                "domain": data.get("domain", domain_name),
                "metadata": {
                    "total_periods": len(data.get("historical_periods", [])),
                    "period_names": [p.get("period_name", "") for p in data.get("historical_periods", [])]
                }
            }
        else:
            print(f"    ⚠️ No ground truth file found: {ground_truth_file}")
            return {"paradigm_shifts": [], "metadata": {}}

    def calculate_temporal_accuracy(self, detected_years: List[int], 
                                  ground_truth_years: List[int], 
                                  tolerance: int = 2) -> float:
        """
        Calculate temporal accuracy against ground truth (pure function).
        
        Args:
            detected_years: Years of detected paradigm shifts
            ground_truth_years: Ground truth paradigm shift years
            tolerance: Tolerance in years for matching
            
        Returns:
            Mean absolute error in years, or -1.0 for no matches/detections
        """
        if not ground_truth_years:
            return 0.0
        
        if not detected_years:
            return -1.0  # Use -1.0 instead of infinity to indicate no detections
        
        # Find best matches within tolerance
        total_error = 0
        matched_count = 0
        
        for gt_year in ground_truth_years:
            best_match_error = float('inf')
            for det_year in detected_years:
                error = abs(det_year - gt_year)
                if error <= tolerance and error < best_match_error:
                    best_match_error = error
            
            if best_match_error != float('inf'):
                total_error += best_match_error
                matched_count += 1
        
        if matched_count == 0:
            return -1.0  # Use -1.0 instead of infinity to indicate no matches
        
        return total_error / matched_count

    def print_summary(self, results: List[ExperimentResult]):
        """
        Print experimental summary statistics (pure function).
        
        Args:
            results: List of experimental results
        """
        print(f"\n📊 EXPERIMENT SUMMARY: {self.experiment_name.upper()}")
        print("=" * 60)
        
        # Group by condition
        by_condition = {}
        for result in results:
            if result.condition not in by_condition:
                by_condition[result.condition] = []
            by_condition[result.condition].append(result)
        
        for condition, condition_results in by_condition.items():
            paradigm_counts = [r.paradigm_shifts_detected for r in condition_results]
            segment_counts = [r.segment_count for r in condition_results]
            comp_times = [r.computational_time for r in condition_results]
            
            print(f"\n🧪 CONDITION: {condition}")
            print(f"    Paradigm shifts: μ={np.mean(paradigm_counts):.1f} ± {np.std(paradigm_counts):.1f}")
            print(f"    Segments: μ={np.mean(segment_counts):.1f} ± {np.std(segment_counts):.1f}")
            print(f"    Computation time: μ={np.mean(comp_times):.3f}s ± {np.std(comp_times):.3f}s")
            print(f"    Domains tested: {len(condition_results)}")
        
        total_time = sum(r.computational_time for r in results)
        print(f"\n⏱️  Total experimental time: {total_time:.2f}s")
        print(f"📈 Total results collected: {len(results)}") 