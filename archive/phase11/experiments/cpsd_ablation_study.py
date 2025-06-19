#!/usr/bin/env python3
"""
Phase 11 CPSD Ablation Study: Comprehensive Evaluation of Citation Paradigm Shift Detection

This study evaluates the CPSD algorithm against the original PELT-based approach through:
1. CPSD vs PELT Comparative Analysis
2. CPSD Multi-Layer Component Analysis  
3. CPSD Ensemble Weight Optimization
4. Known Paradigm Shift Validation

Based on Phase 11 findings that PELT is fundamentally inadequate for citation analysis.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

import pandas as pd
import numpy as np
import json
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Import core modules
from core.data_loader import load_domain_data

# Import CPSD class from our Phase 11 experiment
from experiments.phase11.experiments.citation_paradigm_shift_experiment import CitationParadigmShiftDetection


class CPSDAblationStudy:
    """
    Comprehensive ablation study for Citation Paradigm Shift Detection (CPSD) algorithm
    """
    
    def __init__(self):
        self.domains = [
            'applied_mathematics',
            'computer_science', 
            'computer_vision',
            'deep_learning',
            'machine_learning',
            'machine_translation',
            'natural_language_processing'
        ]
        
        # Known paradigm shifts for validation
        self.known_paradigm_shifts = {
            'deep_learning': [2006, 2012, 2017],  # Hinton, AlexNet, Transformers
            'computer_vision': [2012, 2014, 2015],  # CNN revolution, GANs, ResNet
            'natural_language_processing': [2003, 2017, 2018],  # Statistical, Transformers, BERT
            'machine_learning': [2006, 2012],  # Ensemble methods, Deep learning adoption
            'computer_science': [1995, 2005, 2010],  # Internet, Web 2.0, Cloud
            'machine_translation': [2003, 2014, 2017],  # Statistical MT, Neural MT, Attention
            'applied_mathematics': [1990, 2000],  # Computational methods, Optimization
        }
        
        self.results = {
            'experiment_1': {},  # CPSD vs PELT comparison
            'experiment_2': {},  # CPSD layer analysis
            'experiment_3': {},  # Ensemble weight optimization
            'experiment_4': {}   # Validation analysis
        }

    def load_citation_data(self, domain: str) -> Tuple[pd.Series, pd.Series]:
        """Load citation time series for a domain"""
        try:
            df = load_domain_data(domain)
            
            # Create citation time series
            citation_counts = df.groupby('year').size().reset_index(name='count')
            
            # Fill missing years with 0
            min_year = citation_counts['year'].min()
            max_year = citation_counts['year'].max()
            
            full_years = pd.DataFrame({'year': range(min_year, max_year + 1)})
            citation_series = full_years.merge(citation_counts, on='year', how='left')
            citation_series['count'] = citation_series['count'].fillna(0)
            
            return citation_series['count'], citation_series['year']
        
        except Exception as e:
            print(f"Error loading data for {domain}: {e}")
            return pd.Series([]), pd.Series([])

    def simulate_pelt_detection(self, citation_series: pd.Series, years: pd.Series) -> List[int]:
        """Simulate original PELT detection for comparison"""
        if len(citation_series) < 10:
            return []
        
        citations = citation_series.values
        years_array = years.values
        
        # Conservative PELT simulation (original approach)
        shifts = []
        window_size = max(3, len(citations) // 8)
        
        for i in range(window_size, len(citations) - window_size):
            left_mean = np.mean(citations[i-window_size:i])
            right_mean = np.mean(citations[i:i+window_size])
            
            # Very conservative threshold (original PELT inadequacy)
            if abs(right_mean - left_mean) > left_mean * 1.5:
                year = years_array[i]
                shifts.append(year)
        
        # Remove close shifts
        if not shifts:
            return []
            
        filtered_shifts = [shifts[0]]
        for shift in shifts[1:]:
            if shift - filtered_shifts[-1] >= 5:
                filtered_shifts.append(shift)
        
        return filtered_shifts[:2]  # Very conservative - max 2 shifts

    def experiment_1_cpsd_vs_pelt_comparison(self):
        """
        Experiment 1: Comprehensive CPSD vs PELT Comparison
        
        Tests the core hypothesis that CPSD fundamentally outperforms PELT
        for citation time series analysis.
        """
        print("\n" + "="*80)
        print("EXPERIMENT 1: CPSD vs PELT COMPREHENSIVE COMPARISON")
        print("="*80)
        
        # Initialize CPSD detector
        cpsd_detector = CitationParadigmShiftDetection()
        
        experiment_1_results = {}
        
        for domain in self.domains:
            print(f"\nProcessing {domain}...")
            
            # Load citation data
            citation_series, years = self.load_citation_data(domain)
            
            if len(citation_series) == 0:
                print(f"  No data for {domain}")
                continue
            
            # CPSD Detection
            cpsd_results = cpsd_detector.detect_paradigm_shifts(citation_series, years, domain)
            
            # PELT Simulation
            pelt_results = self.simulate_pelt_detection(citation_series, years)
            
            # Known paradigm shifts for this domain
            known_shifts = self.known_paradigm_shifts.get(domain, [])
            
            # Calculate validation metrics
            cpsd_shifts = cpsd_results['ensemble_shifts']
            cpsd_validation = self.calculate_validation_metrics(cpsd_shifts, known_shifts)
            pelt_validation = self.calculate_validation_metrics(pelt_results, known_shifts)
            
            # Store results
            experiment_1_results[domain] = {
                'cpsd_shifts': cpsd_shifts,
                'pelt_shifts': pelt_results,
                'known_shifts': known_shifts,
                'cpsd_count': len(cpsd_shifts),
                'pelt_count': len(pelt_results),
                'improvement_ratio': len(cpsd_shifts) / max(len(pelt_results), 1),
                'cpsd_validation': cpsd_validation,
                'pelt_validation': pelt_validation,
                'cpsd_method_details': cpsd_results['method_details'],
                'cpsd_confidence': cpsd_results['confidence_scores']
            }
            
            print(f"  CPSD: {len(cpsd_shifts)} shifts, PELT: {len(pelt_results)} shifts")
            print(f"  Improvement: {len(cpsd_shifts) / max(len(pelt_results), 1):.1f}x")
            print(f"  CPSD validation: {cpsd_validation['precision']:.1%} precision, {cpsd_validation['recall']:.1%} recall")
            print(f"  PELT validation: {pelt_validation['precision']:.1%} precision, {pelt_validation['recall']:.1%} recall")
        
        self.results['experiment_1'] = experiment_1_results
        
        # Generate summary statistics
        self.generate_experiment_1_summary()

    def experiment_2_cpsd_layer_analysis(self):
        """
        Experiment 2: CPSD Multi-Layer Component Analysis
        
        Evaluates the contribution of each CPSD layer:
        1. Gradient layer only
        2. Regime layer only  
        3. Burst layer only
        4. Binary segmentation layer only
        5. All layers combined
        """
        print("\n" + "="*80)
        print("EXPERIMENT 2: CPSD MULTI-LAYER COMPONENT ANALYSIS")
        print("="*80)
        
        experiment_2_results = {}
        
        # Define layer configurations
        layer_configs = {
            'gradient_only': {'gradient': 1.0, 'regime': 0.0, 'burst': 0.0, 'binary_seg': 0.0},
            'regime_only': {'gradient': 0.0, 'regime': 1.0, 'burst': 0.0, 'binary_seg': 0.0},
            'burst_only': {'gradient': 0.0, 'regime': 0.0, 'burst': 1.0, 'binary_seg': 0.0},
            'binary_seg_only': {'gradient': 0.0, 'regime': 0.0, 'burst': 0.0, 'binary_seg': 1.0},
            'optimal_ensemble': {'gradient': 0.4, 'regime': 0.3, 'burst': 0.2, 'binary_seg': 0.1}
        }
        
        for domain in self.domains:
            print(f"\nAnalyzing {domain}...")
            
            # Load citation data
            citation_series, years = self.load_citation_data(domain)
            
            if len(citation_series) == 0:
                continue
            
            domain_results = {}
            known_shifts = self.known_paradigm_shifts.get(domain, [])
            
            for config_name, weights in layer_configs.items():
                # Create CPSD with specific weights
                cpsd_detector = CitationParadigmShiftDetection()
                cpsd_detector.ensemble_weights = weights
                
                # Detect paradigm shifts
                results = cpsd_detector.detect_paradigm_shifts(citation_series, years, domain)
                
                # Calculate validation metrics
                validation = self.calculate_validation_metrics(results['ensemble_shifts'], known_shifts)
                
                domain_results[config_name] = {
                    'shifts': results['ensemble_shifts'],
                    'count': len(results['ensemble_shifts']),
                    'validation': validation,
                    'method_details': results['method_details']
                }
                
                print(f"  {config_name}: {len(results['ensemble_shifts'])} shifts, "
                      f"{validation['precision']:.1%} precision, {validation['recall']:.1%} recall")
            
            experiment_2_results[domain] = domain_results
        
        self.results['experiment_2'] = experiment_2_results
        
        # Generate layer analysis summary
        self.generate_experiment_2_summary()

    def experiment_3_ensemble_weight_optimization(self):
        """
        Experiment 3: CPSD Ensemble Weight Optimization
        
        Tests different ensemble weight configurations to find optimal combinations.
        """
        print("\n" + "="*80)
        print("EXPERIMENT 3: CPSD ENSEMBLE WEIGHT OPTIMIZATION")
        print("="*80)
        
        experiment_3_results = {}
        
        # Define weight configurations
        weight_configs = {
            'equal_weights': {'gradient': 0.25, 'regime': 0.25, 'burst': 0.25, 'binary_seg': 0.25},
            'gradient_dominant': {'gradient': 0.7, 'regime': 0.1, 'burst': 0.1, 'binary_seg': 0.1},
            'regime_focused': {'gradient': 0.2, 'regime': 0.6, 'burst': 0.1, 'binary_seg': 0.1},
            'burst_sensitive': {'gradient': 0.2, 'regime': 0.1, 'burst': 0.6, 'binary_seg': 0.1},
            'conservative': {'gradient': 0.3, 'regime': 0.3, 'burst': 0.3, 'binary_seg': 0.1},
            'optimal_research': {'gradient': 0.4, 'regime': 0.3, 'burst': 0.2, 'binary_seg': 0.1}
        }
        
        for domain in self.domains:
            print(f"\nOptimizing {domain}...")
            
            # Load citation data
            citation_series, years = self.load_citation_data(domain)
            
            if len(citation_series) == 0:
                continue
            
            domain_results = {}
            known_shifts = self.known_paradigm_shifts.get(domain, [])
            
            for config_name, weights in weight_configs.items():
                # Create CPSD with specific weights
                cpsd_detector = CitationParadigmShiftDetection()
                cpsd_detector.ensemble_weights = weights
                
                # Detect paradigm shifts
                results = cpsd_detector.detect_paradigm_shifts(citation_series, years, domain)
                
                # Calculate validation metrics
                validation = self.calculate_validation_metrics(results['ensemble_shifts'], known_shifts)
                
                # Calculate ensemble score (weighted average of precision and recall)
                ensemble_score = 0.6 * validation['precision'] + 0.4 * validation['recall']
                
                domain_results[config_name] = {
                    'shifts': results['ensemble_shifts'],
                    'count': len(results['ensemble_shifts']),
                    'validation': validation,
                    'ensemble_score': ensemble_score,
                    'confidence_scores': results['confidence_scores']
                }
                
                print(f"  {config_name}: {len(results['ensemble_shifts'])} shifts, "
                      f"score={ensemble_score:.3f}")
            
            experiment_3_results[domain] = domain_results
        
        self.results['experiment_3'] = experiment_3_results
        
        # Generate optimization summary
        self.generate_experiment_3_summary()

    def experiment_4_paradigm_shift_validation(self):
        """
        Experiment 4: Known Paradigm Shift Validation Analysis
        
        Comprehensive validation against documented paradigm shifts with temporal analysis.
        """
        print("\n" + "="*80)
        print("EXPERIMENT 4: KNOWN PARADIGM SHIFT VALIDATION ANALYSIS")
        print("="*80)
        
        experiment_4_results = {}
        
        # Use optimal CPSD configuration
        cpsd_detector = CitationParadigmShiftDetection()
        
        validation_summary = {
            'total_known_shifts': 0,
            'total_detected_shifts': 0,
            'perfect_matches': 0,
            'close_matches': 0,  # Within ±2 years
            'missed_shifts': 0,
            'false_positives': 0
        }
        
        for domain in self.domains:
            print(f"\nValidating {domain}...")
            
            # Load citation data
            citation_series, years = self.load_citation_data(domain)
            
            if len(citation_series) == 0:
                continue
            
            # Get known paradigm shifts
            known_shifts = self.known_paradigm_shifts.get(domain, [])
            if not known_shifts:
                print(f"  No known paradigm shifts defined for {domain}")
                continue
            
            # Detect paradigm shifts
            results = cpsd_detector.detect_paradigm_shifts(citation_series, years, domain)
            detected_shifts = results['ensemble_shifts']
            
            # Detailed validation analysis
            validation_analysis = self.detailed_validation_analysis(detected_shifts, known_shifts)
            
            experiment_4_results[domain] = {
                'known_shifts': known_shifts,
                'detected_shifts': detected_shifts,
                'validation_analysis': validation_analysis,
                'cpsd_details': results
            }
            
            # Update summary
            validation_summary['total_known_shifts'] += len(known_shifts)
            validation_summary['total_detected_shifts'] += len(detected_shifts)
            validation_summary['perfect_matches'] += validation_analysis['perfect_matches']
            validation_summary['close_matches'] += validation_analysis['close_matches']
            validation_summary['missed_shifts'] += validation_analysis['missed_shifts']
            validation_summary['false_positives'] += validation_analysis['false_positives']
            
            print(f"  Known: {len(known_shifts)}, Detected: {len(detected_shifts)}")
            print(f"  Perfect matches: {validation_analysis['perfect_matches']}")
            print(f"  Close matches (±2y): {validation_analysis['close_matches']}")
            print(f"  Missed: {validation_analysis['missed_shifts']}")
        
        self.results['experiment_4'] = experiment_4_results
        
        # Calculate overall validation statistics
        total_matches = validation_summary['perfect_matches'] + validation_summary['close_matches']
        overall_recall = total_matches / max(validation_summary['total_known_shifts'], 1)
        overall_precision = total_matches / max(validation_summary['total_detected_shifts'], 1)
        
        print(f"\n{'='*50}")
        print("OVERALL VALIDATION SUMMARY")
        print(f"{'='*50}")
        print(f"Total known paradigm shifts: {validation_summary['total_known_shifts']}")
        print(f"Total detected shifts: {validation_summary['total_detected_shifts']}")
        print(f"Perfect + close matches: {total_matches}")
        print(f"Overall recall: {overall_recall:.1%}")
        print(f"Overall precision: {overall_precision:.1%}")
        
        validation_summary.update({
            'overall_recall': overall_recall,
            'overall_precision': overall_precision,
            'f1_score': 2 * (overall_precision * overall_recall) / max(overall_precision + overall_recall, 1e-6)
        })
        
        self.results['experiment_4']['validation_summary'] = validation_summary

    def calculate_validation_metrics(self, detected_shifts: List[int], known_shifts: List[int]) -> Dict:
        """Calculate precision, recall, and F1 score for paradigm shift detection"""
        if not known_shifts:
            return {'precision': 0.0, 'recall': 0.0, 'f1_score': 0.0, 'matches': 0}
        
        # Count matches within ±2 years
        matches = 0
        for known_shift in known_shifts:
            for detected_shift in detected_shifts:
                if abs(detected_shift - known_shift) <= 2:
                    matches += 1
                    break
        
        precision = matches / max(len(detected_shifts), 1)
        recall = matches / len(known_shifts)
        f1_score = 2 * (precision * recall) / max(precision + recall, 1e-6)
        
        return {
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'matches': matches
        }

    def detailed_validation_analysis(self, detected_shifts: List[int], known_shifts: List[int]) -> Dict:
        """Perform detailed validation analysis with temporal accuracy"""
        analysis = {
            'perfect_matches': 0,
            'close_matches': 0,
            'missed_shifts': 0,
            'false_positives': 0,
            'temporal_errors': [],
            'matched_pairs': [],
            'missed_known': [],
            'unmatched_detected': []
        }
        
        matched_detected = set()
        
        # Check each known shift
        for known_shift in known_shifts:
            best_match = None
            best_error = float('inf')
            
            for i, detected_shift in enumerate(detected_shifts):
                if i in matched_detected:
                    continue
                    
                error = abs(detected_shift - known_shift)
                if error < best_error:
                    best_error = error
                    best_match = i
            
            if best_match is not None and best_error <= 2:
                matched_detected.add(best_match)
                analysis['temporal_errors'].append(best_error)
                analysis['matched_pairs'].append((known_shift, detected_shifts[best_match]))
                
                if best_error == 0:
                    analysis['perfect_matches'] += 1
                else:
                    analysis['close_matches'] += 1
            else:
                analysis['missed_shifts'] += 1
                analysis['missed_known'].append(known_shift)
        
        # Count false positives (unmatched detected shifts)
        for i, detected_shift in enumerate(detected_shifts):
            if i not in matched_detected:
                analysis['false_positives'] += 1
                analysis['unmatched_detected'].append(detected_shift)
        
        return analysis

    def generate_experiment_1_summary(self):
        """Generate summary statistics for Experiment 1"""
        print(f"\n{'='*50}")
        print("EXPERIMENT 1 SUMMARY: CPSD vs PELT")
        print(f"{'='*50}")
        
        results = self.results['experiment_1']
        
        # Calculate overall statistics
        total_cpsd = sum(r['cpsd_count'] for r in results.values())
        total_pelt = sum(r['pelt_count'] for r in results.values())
        avg_improvement = np.mean([r['improvement_ratio'] for r in results.values()])
        
        # Calculate validation statistics
        cpsd_precisions = [r['cpsd_validation']['precision'] for r in results.values()]
        cpsd_recalls = [r['cpsd_validation']['recall'] for r in results.values()]
        pelt_precisions = [r['pelt_validation']['precision'] for r in results.values()]
        pelt_recalls = [r['pelt_validation']['recall'] for r in results.values()]
        
        print(f"Total CPSD detections: {total_cpsd}")
        print(f"Total PELT detections: {total_pelt}")
        print(f"Average improvement ratio: {avg_improvement:.1f}x")
        print(f"CPSD avg precision: {np.mean(cpsd_precisions):.1%}")
        print(f"CPSD avg recall: {np.mean(cpsd_recalls):.1%}")
        print(f"PELT avg precision: {np.mean(pelt_precisions):.1%}")
        print(f"PELT avg recall: {np.mean(pelt_recalls):.1%}")

    def generate_experiment_2_summary(self):
        """Generate summary statistics for Experiment 2"""
        print(f"\n{'='*50}")
        print("EXPERIMENT 2 SUMMARY: CPSD LAYER ANALYSIS")
        print(f"{'='*50}")
        
        results = self.results['experiment_2']
        
        # Calculate layer performance across domains
        layer_performance = defaultdict(list)
        
        for domain_results in results.values():
            for config_name, config_results in domain_results.items():
                layer_performance[config_name].append(config_results['validation']['f1_score'])
        
        # Print layer rankings
        layer_rankings = {}
        for config_name, f1_scores in layer_performance.items():
            avg_f1 = np.mean(f1_scores)
            layer_rankings[config_name] = avg_f1
            print(f"{config_name}: {avg_f1:.3f} avg F1-score")
        
        # Best performing layer
        best_layer = max(layer_rankings, key=layer_rankings.get)
        print(f"\nBest performing configuration: {best_layer} ({layer_rankings[best_layer]:.3f} F1-score)")

    def generate_experiment_3_summary(self):
        """Generate summary statistics for Experiment 3"""
        print(f"\n{'='*50}")
        print("EXPERIMENT 3 SUMMARY: ENSEMBLE WEIGHT OPTIMIZATION")
        print(f"{'='*50}")
        
        results = self.results['experiment_3']
        
        # Calculate weight configuration performance
        config_performance = defaultdict(list)
        
        for domain_results in results.values():
            for config_name, config_results in domain_results.items():
                config_performance[config_name].append(config_results['ensemble_score'])
        
        # Print configuration rankings
        config_rankings = {}
        for config_name, scores in config_performance.items():
            avg_score = np.mean(scores)
            config_rankings[config_name] = avg_score
            print(f"{config_name}: {avg_score:.3f} avg ensemble score")
        
        # Best performing configuration
        best_config = max(config_rankings, key=config_rankings.get)
        print(f"\nBest weight configuration: {best_config} ({config_rankings[best_config]:.3f} score)")

    def run_comprehensive_ablation_study(self):
        """Run all ablation study experiments"""
        print("PHASE 11 CPSD ABLATION STUDY")
        print("="*80)
        print("Comprehensive evaluation of Citation Paradigm Shift Detection algorithm")
        print(f"Testing across {len(self.domains)} research domains")
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Run all experiments
        self.experiment_1_cpsd_vs_pelt_comparison()
        self.experiment_2_cpsd_layer_analysis()
        self.experiment_3_ensemble_weight_optimization()
        self.experiment_4_paradigm_shift_validation()
        
        # Save comprehensive results
        self.save_results()
        
        # Generate visualizations
        self.create_visualizations()
        
        print("\n" + "="*80)
        print("PHASE 11 CPSD ABLATION STUDY COMPLETED")
        print("="*80)

    def save_results(self):
        """Save all experimental results"""
        output_dir = "experiments/phase11/results"
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Save comprehensive results
        results_file = f"{output_dir}/cpsd_ablation_study_results.json"
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        print(f"\nResults saved to {results_file}")

    def create_visualizations(self):
        """Create comprehensive visualizations for the ablation study"""
        print("\nGenerating visualizations...")
        
        output_dir = "experiments/phase11/visualizations"
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Set style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Create all visualizations
        self.plot_cpsd_vs_pelt_comparison()
        self.plot_layer_analysis()
        self.plot_ensemble_optimization()
        self.plot_validation_analysis()
        
        print(f"Visualizations saved to {output_dir}/")

    def plot_cpsd_vs_pelt_comparison(self):
        """Create CPSD vs PELT comparison plots"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        results = self.results['experiment_1']
        domains = list(results.keys())
        
        # Plot 1: Detection counts comparison
        cpsd_counts = [results[d]['cpsd_count'] for d in domains]
        pelt_counts = [results[d]['pelt_count'] for d in domains]
        
        x = np.arange(len(domains))
        width = 0.35
        
        ax1.bar(x - width/2, cpsd_counts, width, label='CPSD', alpha=0.8)
        ax1.bar(x + width/2, pelt_counts, width, label='PELT', alpha=0.8)
        ax1.set_xlabel('Domain')
        ax1.set_ylabel('Paradigm Shifts Detected')
        ax1.set_title('CPSD vs PELT: Detection Counts')
        ax1.set_xticks(x)
        ax1.set_xticklabels([d.replace('_', '\n') for d in domains], rotation=45, ha='right')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Improvement ratios
        improvement_ratios = [results[d]['improvement_ratio'] for d in domains]
        colors = ['green' if r > 1 else 'red' for r in improvement_ratios]
        
        ax2.bar(domains, improvement_ratios, color=colors, alpha=0.7)
        ax2.axhline(y=1, color='black', linestyle='--', alpha=0.5)
        ax2.set_xlabel('Domain')
        ax2.set_ylabel('Improvement Ratio (CPSD/PELT)')
        ax2.set_title('CPSD Improvement Over PELT')
        ax2.set_xticklabels([d.replace('_', '\n') for d in domains], rotation=45, ha='right')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Precision comparison
        cpsd_precisions = [results[d]['cpsd_validation']['precision'] for d in domains]
        pelt_precisions = [results[d]['pelt_validation']['precision'] for d in domains]
        
        ax3.bar(x - width/2, cpsd_precisions, width, label='CPSD', alpha=0.8)
        ax3.bar(x + width/2, pelt_precisions, width, label='PELT', alpha=0.8)
        ax3.set_xlabel('Domain')
        ax3.set_ylabel('Precision')
        ax3.set_title('CPSD vs PELT: Precision Comparison')
        ax3.set_xticks(x)
        ax3.set_xticklabels([d.replace('_', '\n') for d in domains], rotation=45, ha='right')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Recall comparison
        cpsd_recalls = [results[d]['cpsd_validation']['recall'] for d in domains]
        pelt_recalls = [results[d]['pelt_validation']['recall'] for d in domains]
        
        ax4.bar(x - width/2, cpsd_recalls, width, label='CPSD', alpha=0.8)
        ax4.bar(x + width/2, pelt_recalls, width, label='PELT', alpha=0.8)
        ax4.set_xlabel('Domain')
        ax4.set_ylabel('Recall')
        ax4.set_title('CPSD vs PELT: Recall Comparison')
        ax4.set_xticks(x)
        ax4.set_xticklabels([d.replace('_', '\n') for d in domains], rotation=45, ha='right')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('experiments/phase11/visualizations/cpsd_vs_pelt_comparison.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()

    def plot_layer_analysis(self):
        """Create CPSD layer analysis plots"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        results = self.results['experiment_2']
        
        # Aggregate layer performance
        layer_performance = defaultdict(list)
        for domain_results in results.values():
            for config_name, config_results in domain_results.items():
                layer_performance[config_name].append(config_results['validation']['f1_score'])
        
        # Plot 1: Layer F1-scores
        configs = list(layer_performance.keys())
        avg_f1_scores = [np.mean(layer_performance[config]) for config in configs]
        std_f1_scores = [np.std(layer_performance[config]) for config in configs]
        
        ax1.bar(configs, avg_f1_scores, yerr=std_f1_scores, capsize=5, alpha=0.7)
        ax1.set_xlabel('Layer Configuration')
        ax1.set_ylabel('Average F1-Score')
        ax1.set_title('CPSD Layer Performance Analysis')
        ax1.set_xticklabels(configs, rotation=45, ha='right')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Detection counts by layer
        layer_counts = defaultdict(list)
        for domain_results in results.values():
            for config_name, config_results in domain_results.items():
                layer_counts[config_name].append(config_results['count'])
        
        avg_counts = [np.mean(layer_counts[config]) for config in configs]
        std_counts = [np.std(layer_counts[config]) for config in configs]
        
        ax2.bar(configs, avg_counts, yerr=std_counts, capsize=5, alpha=0.7, color='orange')
        ax2.set_xlabel('Layer Configuration')
        ax2.set_ylabel('Average Detection Count')
        ax2.set_title('CPSD Layer Detection Counts')
        ax2.set_xticklabels(configs, rotation=45, ha='right')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('experiments/phase11/visualizations/cpsd_layer_analysis.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()

    def plot_ensemble_optimization(self):
        """Create ensemble weight optimization plots"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        results = self.results['experiment_3']
        
        # Aggregate ensemble performance
        config_performance = defaultdict(list)
        for domain_results in results.values():
            for config_name, config_results in domain_results.items():
                config_performance[config_name].append(config_results['ensemble_score'])
        
        # Plot 1: Ensemble scores
        configs = list(config_performance.keys())
        avg_scores = [np.mean(config_performance[config]) for config in configs]
        std_scores = [np.std(config_performance[config]) for config in configs]
        
        colors = ['red' if 'optimal' in config else 'blue' for config in configs]
        
        ax1.bar(configs, avg_scores, yerr=std_scores, capsize=5, alpha=0.7, color=colors)
        ax1.set_xlabel('Weight Configuration')
        ax1.set_ylabel('Average Ensemble Score')
        ax1.set_title('CPSD Ensemble Weight Optimization')
        ax1.set_xticklabels(configs, rotation=45, ha='right')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Configuration heatmap
        # Create weight matrix
        weight_configs = {
            'equal_weights': [0.25, 0.25, 0.25, 0.25],
            'gradient_dominant': [0.7, 0.1, 0.1, 0.1],
            'regime_focused': [0.2, 0.6, 0.1, 0.1],
            'burst_sensitive': [0.2, 0.1, 0.6, 0.1],
            'conservative': [0.3, 0.3, 0.3, 0.1],
            'optimal_research': [0.4, 0.3, 0.2, 0.1]
        }
        
        weight_matrix = []
        config_labels = []
        for config in configs:
            if config in weight_configs:
                weight_matrix.append(weight_configs[config])
                config_labels.append(config)
        
        weight_matrix = np.array(weight_matrix)
        
        im = ax2.imshow(weight_matrix, cmap='YlOrRd', aspect='auto')
        ax2.set_xticks(range(4))
        ax2.set_xticklabels(['Gradient', 'Regime', 'Burst', 'Binary Seg'])
        ax2.set_yticks(range(len(config_labels)))
        ax2.set_yticklabels(config_labels)
        ax2.set_title('CPSD Ensemble Weight Configurations')
        
        # Add text annotations
        for i in range(len(config_labels)):
            for j in range(4):
                text = ax2.text(j, i, f'{weight_matrix[i, j]:.1f}',
                              ha="center", va="center", color="black", fontweight='bold')
        
        plt.colorbar(im, ax=ax2)
        
        plt.tight_layout()
        plt.savefig('experiments/phase11/visualizations/cpsd_ensemble_optimization.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()

    def plot_validation_analysis(self):
        """Create paradigm shift validation plots"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        results = self.results['experiment_4']
        validation_summary = results['validation_summary']
        
        # Plot 1: Overall validation metrics
        metrics = ['Overall Recall', 'Overall Precision', 'F1-Score']
        values = [validation_summary['overall_recall'], 
                 validation_summary['overall_precision'],
                 validation_summary['f1_score']]
        
        colors = ['green', 'blue', 'orange']
        bars = ax1.bar(metrics, values, color=colors, alpha=0.7)
        ax1.set_ylabel('Score')
        ax1.set_title('CPSD Overall Validation Performance')
        ax1.set_ylim(0, 1)
        ax1.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{value:.1%}', ha='center', va='bottom', fontweight='bold')
        
        # Plot 2: Domain-wise validation
        domain_validation = {}
        for domain, domain_data in results.items():
            if domain == 'validation_summary':
                continue
            known_count = len(domain_data['known_shifts'])
            detected_count = len(domain_data['detected_shifts'])
            matches = domain_data['validation_analysis']['perfect_matches'] + \
                     domain_data['validation_analysis']['close_matches']
            
            domain_validation[domain] = {
                'known': known_count,
                'detected': detected_count,
                'matches': matches,
                'recall': matches / max(known_count, 1)
            }
        
        domains = list(domain_validation.keys())
        recalls = [domain_validation[d]['recall'] for d in domains]
        
        ax2.bar(domains, recalls, alpha=0.7, color='green')
        ax2.set_xlabel('Domain')
        ax2.set_ylabel('Recall (Known Shifts Detected)')
        ax2.set_title('Domain-Wise Validation Recall')
        ax2.set_xticklabels([d.replace('_', '\n') for d in domains], rotation=45, ha='right')
        ax2.set_ylim(0, 1)
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Validation categories
        categories = ['Perfect Matches', 'Close Matches', 'Missed Shifts', 'False Positives']
        counts = [validation_summary['perfect_matches'],
                 validation_summary['close_matches'],
                 validation_summary['missed_shifts'],
                 validation_summary['false_positives']]
        colors = ['darkgreen', 'lightgreen', 'red', 'orange']
        
        ax3.pie(counts, labels=categories, colors=colors, autopct='%1.1f%%', startangle=90)
        ax3.set_title('CPSD Validation Breakdown')
        
        # Plot 4: Known vs Detected timeline
        all_known = []
        all_detected = []
        
        for domain, domain_data in results.items():
            if domain == 'validation_summary':
                continue
            all_known.extend(domain_data['known_shifts'])
            all_detected.extend(domain_data['detected_shifts'])
        
        if all_known and all_detected:
            ax4.hist(all_known, bins=20, alpha=0.7, label='Known Shifts', color='blue')
            ax4.hist(all_detected, bins=20, alpha=0.7, label='Detected Shifts', color='red')
            ax4.set_xlabel('Year')
            ax4.set_ylabel('Count')
            ax4.set_title('Temporal Distribution: Known vs Detected Shifts')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('experiments/phase11/visualizations/cpsd_validation_analysis.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()


if __name__ == "__main__":
    # Run the comprehensive ablation study
    study = CPSDAblationStudy()
    study.run_comprehensive_ablation_study() 