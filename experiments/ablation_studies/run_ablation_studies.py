#!/usr/bin/env python3
"""
Main Runner for Ablation Studies

Orchestrates the execution of all 5 ablation experiments with proper environment setup,
logging, and result consolidation. Follows project guidelines for fail-fast error handling.

Usage:
    python run_ablation_studies.py [--experiment EXPERIMENT_NUMBER] [--all]
    
Examples:
    python run_ablation_studies.py --experiment 1          # Run only experiment 1
    python run_ablation_studies.py --all                   # Run all experiments
    python run_ablation_studies.py                         # Interactive selection
"""

import os
import sys
import time
import argparse
from typing import List, Dict, Any

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import individual experiments
from experiment_1_modality_analysis import run_modality_analysis_experiment
from experiment_2_temporal_windows import run_temporal_window_experiment  
from experiment_3_keyword_filtering import run_keyword_filtering_experiment
from experiment_4_citation_validation import run_citation_validation_experiment
from experiment_5_segmentation_boundaries import run_segmentation_boundary_experiment

from experiment_utils import create_experiment_output_directory


# Experiment registry with metadata
EXPERIMENTS = {
    1: {
        'name': 'Signal Detection Modality Analysis',
        'function': run_modality_analysis_experiment,
        'description': 'Compare direction-only, citation-only, and combined detection',
        'priority': 'CRITICAL',
        'estimated_time': '10-15 minutes'
    },
    2: {
        'name': 'Temporal Window Sensitivity Analysis',
        'function': run_temporal_window_experiment,
        'description': 'Test different direction window sizes and citation scales',
        'priority': 'HIGH',
        'estimated_time': '20-30 minutes'
    },
    3: {
        'name': 'Keyword Filtering Impact Assessment',
        'function': run_keyword_filtering_experiment,
        'description': 'Evaluate filtering effectiveness across data quality scenarios',
        'priority': 'HIGH',
        'estimated_time': '15-20 minutes'
    },
    4: {
        'name': 'Citation Validation Strategy Comparison',
        'function': run_citation_validation_experiment,
        'description': 'Test boost factors, validation windows, and fusion methods',
        'priority': 'MEDIUM-HIGH',
        'estimated_time': '25-35 minutes'
    },
    5: {
        'name': 'Segmentation Boundary Methods',
        'function': run_segmentation_boundary_experiment,
        'description': 'Compare boundary detection algorithms and constraints',
        'priority': 'MEDIUM-HIGH',
        'estimated_time': '15-25 minutes'
    }
}


def print_experiment_menu() -> None:
    """Print formatted menu of available experiments."""
    print("🧪 TIMELINE SEGMENTATION ABLATION STUDIES")
    print("=" * 70)
    print("Available experiments:")
    print()
    
    for exp_id, exp_info in EXPERIMENTS.items():
        print(f"{exp_id}. {exp_info['name']}")
        print(f"   Priority: {exp_info['priority']}")
        print(f"   Description: {exp_info['description']}")
        print(f"   Estimated time: {exp_info['estimated_time']}")
        print()


def run_single_experiment(experiment_id: int) -> Dict[str, Any]:
    """
    Run a single ablation experiment with error handling and timing.
    
    Args:
        experiment_id: ID of the experiment to run (1-5)
        
    Returns:
        Dictionary with experiment results and metadata
        
    Raises:
        ValueError: If experiment ID is invalid or experiment fails
    """
    if experiment_id not in EXPERIMENTS:
        raise ValueError(f"Invalid experiment ID: {experiment_id}. Valid IDs: {list(EXPERIMENTS.keys())}")
    
    exp_info = EXPERIMENTS[experiment_id]
    
    print(f"\n🚀 STARTING EXPERIMENT {experiment_id}: {exp_info['name'].upper()}")
    print("=" * 70)
    print(f"Description: {exp_info['description']}")
    print(f"Priority: {exp_info['priority']}")
    print(f"Estimated time: {exp_info['estimated_time']}")
    print()
    
    start_time = time.time()
    
    try:
        # Run the experiment
        results_file = exp_info['function']()
        execution_time = time.time() - start_time
        
        print(f"\n✅ EXPERIMENT {experiment_id} COMPLETED SUCCESSFULLY")
        print(f"⏱️  Execution time: {execution_time:.1f} seconds ({execution_time/60:.1f} minutes)")
        print(f"📄 Results saved to: {results_file}")
        
        return {
            'experiment_id': experiment_id,
            'experiment_name': exp_info['name'],
            'status': 'success',
            'execution_time': execution_time,
            'results_file': results_file,
            'error': None
        }
        
    except Exception as e:
        execution_time = time.time() - start_time
        
        print(f"\n❌ EXPERIMENT {experiment_id} FAILED")
        print(f"⏱️  Execution time: {execution_time:.1f} seconds")
        print(f"💥 Error: {str(e)}")
        
        return {
            'experiment_id': experiment_id,
            'experiment_name': exp_info['name'],
            'status': 'failed',
            'execution_time': execution_time,
            'results_file': None,
            'error': str(e)
        }


def run_all_experiments() -> List[Dict[str, Any]]:
    """
    Run all ablation experiments in sequence.
    
    Returns:
        List of experiment result dictionaries
    """
    print("🧪 RUNNING ALL ABLATION STUDIES")
    print("=" * 70)
    print(f"Total experiments: {len(EXPERIMENTS)}")
    total_estimated_time = sum(
        float(exp['estimated_time'].split('-')[1].split()[0]) 
        for exp in EXPERIMENTS.values()
    )
    print(f"Total estimated time: ~{total_estimated_time:.0f} minutes")
    print()
    
    all_results = []
    overall_start_time = time.time()
    
    for experiment_id in sorted(EXPERIMENTS.keys()):
        try:
            result = run_single_experiment(experiment_id)
            all_results.append(result)
            
            # Brief pause between experiments
            time.sleep(2)
            
        except Exception as e:
            print(f"❌ Fatal error in experiment {experiment_id}: {str(e)}")
            # Continue with remaining experiments even if one fails
            all_results.append({
                'experiment_id': experiment_id,
                'experiment_name': EXPERIMENTS[experiment_id]['name'],
                'status': 'fatal_error',
                'execution_time': 0,
                'results_file': None,
                'error': str(e)
            })
    
    overall_execution_time = time.time() - overall_start_time
    
    # Print summary
    print("\n📊 ABLATION STUDIES SUMMARY")
    print("=" * 50)
    
    successful_experiments = [r for r in all_results if r['status'] == 'success']
    failed_experiments = [r for r in all_results if r['status'] in ['failed', 'fatal_error']]
    
    print(f"✅ Successful experiments: {len(successful_experiments)}/{len(EXPERIMENTS)}")
    print(f"❌ Failed experiments: {len(failed_experiments)}/{len(EXPERIMENTS)}")
    print(f"⏱️  Total execution time: {overall_execution_time:.1f} seconds ({overall_execution_time/60:.1f} minutes)")
    
    if successful_experiments:
        print("\n✅ Successful experiments:")
        for result in successful_experiments:
            print(f"   {result['experiment_id']}. {result['experiment_name']} ({result['execution_time']:.1f}s)")
    
    if failed_experiments:
        print("\n❌ Failed experiments:")
        for result in failed_experiments:
            print(f"   {result['experiment_id']}. {result['experiment_name']} - {result['error']}")
    
    return all_results


def interactive_experiment_selection() -> List[int]:
    """
    Interactive selection of experiments to run.
    
    Returns:
        List of selected experiment IDs
    """
    print_experiment_menu()
    
    while True:
        try:
            user_input = input("Enter experiment numbers to run (comma-separated) or 'all' for all experiments: ").strip()
            
            if user_input.lower() == 'all':
                return list(EXPERIMENTS.keys())
            
            # Parse comma-separated experiment IDs
            experiment_ids = []
            for exp_str in user_input.split(','):
                exp_id = int(exp_str.strip())
                if exp_id not in EXPERIMENTS:
                    print(f"❌ Invalid experiment ID: {exp_id}")
                    continue
                experiment_ids.append(exp_id)
            
            if not experiment_ids:
                print("❌ No valid experiment IDs provided. Please try again.")
                continue
                
            return sorted(experiment_ids)
            
        except ValueError:
            print("❌ Invalid input. Please enter numbers separated by commas or 'all'.")
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            sys.exit(0)


def main():
    """Main entry point for ablation studies runner."""
    parser = argparse.ArgumentParser(
        description="Run timeline segmentation ablation studies",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_ablation_studies.py --experiment 1    # Run experiment 1 only
  python run_ablation_studies.py --all             # Run all experiments
  python run_ablation_studies.py                   # Interactive selection
        """
    )
    
    parser.add_argument(
        '--experiment', '-e',
        type=int,
        choices=list(EXPERIMENTS.keys()),
        help='Run specific experiment by ID (1-5)'
    )
    
    parser.add_argument(
        '--all', '-a',
        action='store_true',
        help='Run all experiments'
    )
    
    args = parser.parse_args()
    
    # Create output directory
    create_experiment_output_directory("summary")
    
    try:
        if args.all:
            # Run all experiments
            results = run_all_experiments()
            
        elif args.experiment:
            # Run specific experiment
            result = run_single_experiment(args.experiment)
            results = [result]
            
        else:
            # Interactive selection
            selected_experiments = interactive_experiment_selection()
            
            if len(selected_experiments) == 1:
                result = run_single_experiment(selected_experiments[0])
                results = [result]
            else:
                results = []
                for exp_id in selected_experiments:
                    result = run_single_experiment(exp_id)
                    results.append(result)
                    time.sleep(2)  # Brief pause between experiments
        
        # Final summary
        successful_count = sum(1 for r in results if r['status'] == 'success')
        total_count = len(results)
        
        print(f"\n🎯 FINAL RESULTS: {successful_count}/{total_count} experiments completed successfully")
        
        if successful_count == total_count:
            print("🏆 All experiments completed successfully!")
            sys.exit(0)
        else:
            print("⚠️  Some experiments failed. Check logs above for details.")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n⏹️  Ablation studies interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Fatal error in ablation studies runner: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main() 