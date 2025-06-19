"""
Experiment 1: Direction Detection Sensitivity Analysis
Research-driven investigation of direction signal threshold effects on algorithmic pipeline

Research Questions:
1. How does direction sensitivity create cascade effects through clustering and validation?
2. What is the optimal sensitivity threshold for balancing detection rate with quality?
3. Do different domains require different optimal sensitivity levels?
4. How does citation availability interact with optimal direction sensitivity?

Researcher: AI Research Assistant
Date: June 17, 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from datetime import datetime
from typing import List, Dict, Any, Tuple
import sys
import os

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from core.data_processing import process_domain_data
from core.shift_signal_detection import (
    detect_research_direction_changes,
    cluster_direction_signals_by_proximity, 
    detect_citation_structural_breaks,
    validate_direction_with_citation
)
from core.integration import SensitivityConfig
from core.change_detection import create_segments_with_confidence
from experiments.phase12.experiments.utils.experiment_base import ExperimentBase


class DirectionSensitivityExperiment(ExperimentBase):
    """
    Research-driven direction sensitivity analysis.
    
    Tests how direction detection threshold affects the entire algorithmic pipeline
    with focus on cascade effects, optimization opportunities, and domain patterns.
    """
    
    def __init__(self):
        super().__init__("1_direction_sensitivity")
        
        # Research-driven threshold selection
        self.sensitivity_thresholds = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]  # Fine-grained analysis
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
        
    def run_sensitivity_analysis(self) -> Dict[str, Any]:
        """
        Execute comprehensive sensitivity analysis across all thresholds and domains.
        
        Returns detailed pipeline analysis for research interpretation.
        """
        print("🔬 EXPERIMENT 1: DIRECTION DETECTION SENSITIVITY ANALYSIS")
        print("=" * 60)
        print("Research Focus: Pipeline cascade effects and optimization")
        print(f"Thresholds: {self.sensitivity_thresholds}")
        print(f"Domains: {len(self.domains)} domains")
        print()
        
        all_results = []
        pipeline_analysis = {}
        
        for domain_name in self.domains:
            print(f"📊 Analyzing Domain: {domain_name}")
            domain_results = self._analyze_domain_sensitivity(domain_name)
            all_results.extend(domain_results)
            
            # Store domain-specific pipeline analysis
            pipeline_analysis[domain_name] = self._analyze_domain_pipeline(domain_results)
            
        # Perform cross-domain research analysis
        research_insights = self._synthesize_research_insights(all_results, pipeline_analysis)
        
        # Save comprehensive results
        output_file = self._save_research_results(all_results, research_insights, pipeline_analysis)
        
        return {
            'results_file': output_file,
            'total_experiments': len(all_results),
            'research_insights': research_insights,
            'pipeline_analysis': pipeline_analysis
        }
    
    def _analyze_domain_sensitivity(self, domain_name: str) -> List[Dict[str, Any]]:
        """
        Analyze sensitivity effects for a single domain across all thresholds.
        
        Args:
            domain_name: Domain to analyze
            
        Returns:
            List of detailed results for each threshold
        """
        # Load domain data
        result = process_domain_data(domain_name)
        if not result.success:
            print(f"❌ Failed to load {domain_name}: {result.error_message}")
            return []
        
        domain_data = result.domain_data
        ground_truth = self.load_ground_truth(domain_name)
        
        domain_results = []
        
        # Test each sensitivity threshold
        for threshold in self.sensitivity_thresholds:
            print(f"  🎯 Testing threshold: {threshold:.1f}")
            
            # Run complete pipeline with this threshold
            pipeline_result = self._run_pipeline_with_threshold(
                domain_data, domain_name, threshold, ground_truth
            )
            
            domain_results.append(pipeline_result)
            
        return domain_results
    
    def _run_pipeline_with_threshold(self, domain_data, domain_name: str, 
                                   threshold: float, ground_truth: Dict) -> Dict[str, Any]:
        """
        Run complete algorithmic pipeline with specific direction threshold.
        
        Captures detailed metrics at each pipeline stage for research analysis.
        """
        # Stage 1: Direction Detection (VARIABLE)
        raw_direction_signals = detect_research_direction_changes(
            domain_data, 
            sensitivity_threshold=threshold
        )
        
        # Stage 2: Temporal Clustering (CONSTANT)
        config = SensitivityConfig(granularity=3)  # Default clustering
        clustered_signals = cluster_direction_signals_by_proximity(
            raw_direction_signals, 
            config
        )
        
        # Stage 3: Citation Detection (CONSTANT)
        citation_signals = detect_citation_structural_breaks(domain_data, domain_name)
        
        # Stage 4: Citation Validation (CONSTANT)
        validated_signals = validate_direction_with_citation(
            clustered_signals, citation_signals, domain_data, domain_name, config
        )
        
        # Stage 5: Segmentation
        if validated_signals:
            change_years = [signal.year for signal in validated_signals]
            confidence_scores = [signal.confidence for signal in validated_signals]
            statistical_significance = float(sum(confidence_scores) / len(confidence_scores))
        else:
            change_years = []
            statistical_significance = 0.0
            
        segments = create_segments_with_confidence(
            change_years=change_years,
            time_range=domain_data.year_range,
            statistical_significance=statistical_significance,
            domain_name=domain_name
        )
        
        # Calculate research metrics
        ground_truth_years = ground_truth.get("paradigm_shifts", [])
        temporal_accuracy = self.calculate_temporal_accuracy(change_years, ground_truth_years)
        
        # Citation support analysis
        citation_support_count = 0
        for direction_signal in clustered_signals:
            for citation_signal in citation_signals:
                if abs(citation_signal.year - direction_signal.year) <= 2:
                    citation_support_count += 1
                    break
        
        citation_support_rate = citation_support_count / len(clustered_signals) if clustered_signals else 0
        
        # Validation pathway analysis
        citation_validated = [s for s in validated_signals if s.signal_type.endswith("_validated")]
        direction_only = [s for s in validated_signals if s.signal_type.endswith("_only")]
        
        return {
            'domain': domain_name,
            'threshold': threshold,
            'raw_direction_signals': len(raw_direction_signals),
            'clustered_signals': len(clustered_signals),
            'citation_signals': len(citation_signals),
            'validated_signals': len(validated_signals),
            'final_segments': len(segments),
            'clustering_reduction': len(raw_direction_signals) / max(len(clustered_signals), 1),
            'validation_acceptance': len(validated_signals) / max(len(clustered_signals), 1),
            'citation_support_rate': citation_support_rate,
            'citation_validated_count': len(citation_validated),
            'direction_only_count': len(direction_only),
            'statistical_significance': statistical_significance,
            'temporal_accuracy': temporal_accuracy,
            'ground_truth_available': len(ground_truth_years) > 0,
            'ground_truth_count': len(ground_truth_years),
            'time_range': domain_data.year_range,
            'raw_signals_by_year': [s.year for s in raw_direction_signals],
            'validated_signals_by_year': change_years,
            'segment_lengths': [end - start + 1 for start, end in segments]
        }
    
    def _analyze_domain_pipeline(self, domain_results: List[Dict]) -> Dict[str, Any]:
        """
        Analyze pipeline behavior patterns for a single domain.
        
        Identifies cascade effects, bottlenecks, and optimization opportunities.
        """
        if not domain_results:
            return {}
        
        # Extract pipeline stage data
        thresholds = [r['threshold'] for r in domain_results]
        raw_signals = [r['raw_direction_signals'] for r in domain_results]
        clustered_signals = [r['clustered_signals'] for r in domain_results] 
        validated_signals = [r['validated_signals'] for r in domain_results]
        final_segments = [r['final_segments'] for r in domain_results]
        temporal_accuracy = [r['temporal_accuracy'] for r in domain_results if r['temporal_accuracy'] >= 0]
        
        # Calculate pipeline cascade effects
        raw_to_cluster_ratios = [r['clustering_reduction'] for r in domain_results]
        cluster_to_validated_ratios = [r['validation_acceptance'] for r in domain_results]
        
        # Find optimal threshold (best F1-score approximation)
        if temporal_accuracy:
            # Simple optimization: highest segment count with acceptable accuracy
            quality_scores = []
            for i, result in enumerate(domain_results):
                if result['temporal_accuracy'] >= 0:
                    # Balance segment count with temporal accuracy
                    segments = result['final_segments']
                    accuracy = 1.0 / (result['temporal_accuracy'] + 1.0)  # Higher is better
                    quality_score = segments * accuracy
                    quality_scores.append((quality_score, result['threshold']))
            
            optimal_threshold = max(quality_scores, key=lambda x: x[0])[1] if quality_scores else 0.4
        else:
            optimal_threshold = 0.4  # Default fallback
        
        return {
            'domain': domain_results[0]['domain'],
            'threshold_range': (min(thresholds), max(thresholds)),
            'raw_signal_range': (min(raw_signals), max(raw_signals)),
            'final_segment_range': (min(final_segments), max(final_segments)),
            'clustering_reduction_mean': float(np.mean(raw_to_cluster_ratios)),
            'validation_acceptance_mean': float(np.mean(cluster_to_validated_ratios)),
            'temporal_accuracy_mean': float(np.mean(temporal_accuracy)) if temporal_accuracy else -1.0,
            'optimal_threshold_estimate': optimal_threshold,
            'sensitivity_responsiveness': (max(final_segments) - min(final_segments)) / (max(thresholds) - min(thresholds)),
            'citation_support_availability': float(np.mean([r['citation_support_rate'] for r in domain_results])),
            'pipeline_bottleneck': self._identify_pipeline_bottleneck(domain_results)
        }
    
    def _identify_pipeline_bottleneck(self, domain_results: List[Dict]) -> str:
        """
        Identify where most signals are lost in the pipeline.
        """
        if not domain_results:
            return "unknown"
        
        # Average losses at each stage
        avg_clustering_loss = np.mean([1 - (r['clustered_signals'] / max(r['raw_direction_signals'], 1)) 
                                     for r in domain_results])
        avg_validation_loss = np.mean([1 - (r['validated_signals'] / max(r['clustered_signals'], 1)) 
                                     for r in domain_results])
        
        if avg_clustering_loss > avg_validation_loss:
            return "clustering"
        elif avg_validation_loss > 0.3:  # Significant validation loss
            return "validation" 
        else:
            return "balanced"
    
    def _synthesize_research_insights(self, all_results: List[Dict], 
                                    pipeline_analysis: Dict) -> Dict[str, Any]:
        """
        Synthesize research insights across all domains and thresholds.
        
        This is where the critical thinking and interpretation happens.
        """
        # Group results by threshold for cross-domain analysis
        by_threshold = {}
        for result in all_results:
            threshold = result['threshold']
            if threshold not in by_threshold:
                by_threshold[threshold] = []
            by_threshold[threshold].append(result)
        
        # Research Insight 1: Cascade Amplification Analysis
        cascade_analysis = self._analyze_cascade_effects(by_threshold)
        
        # Research Insight 2: Domain Stratification
        domain_clustering = self._analyze_domain_patterns(pipeline_analysis)
        
        # Research Insight 3: Optimization Landscape
        optimization_analysis = self._analyze_optimization_landscape(all_results)
        
        # Research Insight 4: Current Default Validation
        default_validation = self._validate_current_default(by_threshold)
        
        return {
            'cascade_amplification': cascade_analysis,
            'domain_stratification': domain_clustering,
            'optimization_landscape': optimization_analysis,
            'current_default_validation': default_validation,
            'research_summary': self._generate_research_summary(cascade_analysis, domain_clustering, optimization_analysis)
        }
    
    def _analyze_cascade_effects(self, by_threshold: Dict) -> Dict[str, Any]:
        """
        Analyze how threshold changes cascade through the pipeline.
        
        Research Question: Is the effect linear or do we see amplification/dampening?
        """
        cascade_data = []
        
        for threshold, results in sorted(by_threshold.items()):
            avg_raw = np.mean([r['raw_direction_signals'] for r in results])
            avg_clustered = np.mean([r['clustered_signals'] for r in results])
            avg_validated = np.mean([r['validated_signals'] for r in results])
            avg_segments = np.mean([r['final_segments'] for r in results])
            
            cascade_data.append({
                'threshold': threshold,
                'raw_signals': avg_raw,
                'clustered_signals': avg_clustered,
                'validated_signals': avg_validated,
                'final_segments': avg_segments
            })
        
        # Calculate amplification factors
        if len(cascade_data) >= 2:
            threshold_change = cascade_data[-1]['threshold'] - cascade_data[0]['threshold']
            raw_change = cascade_data[-1]['raw_signals'] - cascade_data[0]['raw_signals']
            segment_change = cascade_data[-1]['final_segments'] - cascade_data[0]['final_segments']
            
            if threshold_change != 0 and raw_change != 0:
                raw_sensitivity = raw_change / threshold_change
                segment_sensitivity = segment_change / threshold_change
                amplification_factor = segment_sensitivity / raw_sensitivity if raw_sensitivity != 0 else 1.0
            else:
                amplification_factor = 1.0
        else:
            amplification_factor = 1.0
        
        return {
            'cascade_data': cascade_data,
            'amplification_factor': amplification_factor,
            'linearity_assessment': 'amplified' if amplification_factor > 1.5 else 'linear' if amplification_factor > 0.5 else 'dampened'
        }
    
    def _analyze_domain_patterns(self, pipeline_analysis: Dict) -> Dict[str, Any]:
        """
        Identify domain clustering patterns based on sensitivity behavior.
        
        Research Question: Can we group domains into sensitivity profiles?
        """
        if not pipeline_analysis:
            return {}
        
        # Extract domain characteristics
        domain_features = []
        for domain, analysis in pipeline_analysis.items():
            domain_features.append({
                'domain': domain,
                'optimal_threshold': analysis.get('optimal_threshold_estimate', 0.4),
                'sensitivity_responsiveness': analysis.get('sensitivity_responsiveness', 0),
                'citation_support': analysis.get('citation_support_availability', 0),
                'validation_acceptance': analysis.get('validation_acceptance_mean', 0),
                'bottleneck': analysis.get('pipeline_bottleneck', 'unknown')
            })
        
        # Simple clustering based on optimal threshold
        high_sensitivity_domains = [d for d in domain_features if d['optimal_threshold'] <= 0.3]
        medium_sensitivity_domains = [d for d in domain_features if 0.3 < d['optimal_threshold'] <= 0.5]
        low_sensitivity_domains = [d for d in domain_features if d['optimal_threshold'] > 0.5]
        
        return {
            'domain_features': domain_features,
            'high_sensitivity_domains': [d['domain'] for d in high_sensitivity_domains],
            'medium_sensitivity_domains': [d['domain'] for d in medium_sensitivity_domains],
            'low_sensitivity_domains': [d['domain'] for d in low_sensitivity_domains],
            'clustering_interpretation': self._interpret_domain_clustering(high_sensitivity_domains, medium_sensitivity_domains, low_sensitivity_domains)
        }
    
    def _interpret_domain_clustering(self, high_sens, medium_sens, low_sens) -> str:
        """
        Provide research interpretation of domain clustering patterns.
        """
        if len(high_sens) > len(medium_sens) + len(low_sens):
            return "Most domains benefit from high sensitivity, suggesting rich paradigm evolution"
        elif len(low_sens) > len(high_sens) + len(medium_sens):
            return "Most domains require conservative sensitivity, suggesting stable evolution patterns"
        else:
            return "Mixed sensitivity requirements across domains, suggesting heterogeneous evolution patterns"
    
    def _analyze_optimization_landscape(self, all_results: List[Dict]) -> Dict[str, Any]:
        """
        Analyze the optimization landscape for threshold selection.
        
        Research Question: Is there a clear optimum or multiple local optima?
        """
        # Group by domain for individual optimization analysis
        by_domain = {}
        for result in all_results:
            domain = result['domain']
            if domain not in by_domain:
                by_domain[domain] = []
            by_domain[domain].append(result)
        
        optimization_data = []
        
        for domain, results in by_domain.items():
            if not results:
                continue
                
            # Find threshold with best quality score for this domain
            best_score = -1
            best_threshold = 0.4
            
            for result in results:
                if result['temporal_accuracy'] >= 0:
                    # Quality score: balance segments and accuracy
                    segments = result['final_segments']
                    accuracy_score = 1.0 / (result['temporal_accuracy'] + 1.0)
                    quality_score = segments * accuracy_score
                    
                    if quality_score > best_score:
                        best_score = quality_score
                        best_threshold = result['threshold']
            
            optimization_data.append({
                'domain': domain,
                'optimal_threshold': best_threshold,
                'optimal_score': best_score
            })
        
        # Analyze distribution of optimal thresholds
        optimal_thresholds = [d['optimal_threshold'] for d in optimization_data]
        threshold_std = float(np.std(optimal_thresholds)) if optimal_thresholds else 0
        
        return {
            'domain_optima': optimization_data,
            'optimal_threshold_mean': float(np.mean(optimal_thresholds)) if optimal_thresholds else 0.4,
            'optimal_threshold_std': threshold_std,
            'optimization_landscape': 'sharp' if threshold_std < 0.1 else 'broad' if threshold_std < 0.2 else 'scattered'
        }
    
    def _validate_current_default(self, by_threshold: Dict) -> Dict[str, Any]:
        """
        Validate whether current default threshold (0.4) is evidence-based optimal.
        
        Research Question: Is 0.4 a good choice or should we change it?
        """
        if 0.4 not in by_threshold:
            return {'validation': 'inconclusive', 'reason': 'Default threshold not tested'}
        
        default_results = by_threshold[0.4]
        
        # Compare with other thresholds
        performance_comparison = {}
        for threshold, results in by_threshold.items():
            avg_segments = np.mean([r['final_segments'] for r in results])
            avg_accuracy = np.mean([r['temporal_accuracy'] for r in results if r['temporal_accuracy'] >= 0])
            
            performance_comparison[threshold] = {
                'avg_segments': avg_segments,
                'avg_accuracy': avg_accuracy if not np.isnan(avg_accuracy) else -1.0
            }
        
        # Find best performing threshold
        best_threshold = 0.4
        best_score = 0
        
        for threshold, perf in performance_comparison.items():
            if perf['avg_accuracy'] >= 0:
                score = perf['avg_segments'] * (1.0 / (perf['avg_accuracy'] + 1.0))
                if score > best_score:
                    best_score = score
                    best_threshold = threshold
        
        default_performance = performance_comparison.get(0.4, {})
        best_performance = performance_comparison.get(best_threshold, {})
        
        if best_threshold == 0.4:
            validation_result = 'optimal'
        elif abs(best_threshold - 0.4) <= 0.1:
            validation_result = 'near_optimal'
        else:
            validation_result = 'suboptimal'
        
        return {
            'validation': validation_result,
            'current_threshold': 0.4,
            'recommended_threshold': best_threshold,
            'performance_gap': best_score / max(performance_comparison.get(0.4, {}).get('avg_segments', 1) * 
                                             (1.0 / (performance_comparison.get(0.4, {}).get('avg_accuracy', 0) + 1.0)), 0.1),
            'performance_comparison': performance_comparison
        }
    
    def _generate_research_summary(self, cascade_analysis: Dict, domain_clustering: Dict, 
                                 optimization_analysis: Dict) -> str:
        """
        Generate high-level research summary with key insights.
        """
        summary_parts = []
        
        # Cascade effect insight
        amplification = cascade_analysis.get('amplification_factor', 1.0)
        if amplification > 1.5:
            summary_parts.append(f"Direction sensitivity shows amplified cascade effects ({amplification:.1f}x), indicating pipeline synergy")
        elif amplification < 0.5:
            summary_parts.append(f"Direction sensitivity shows dampened effects ({amplification:.1f}x), indicating pipeline constraints")
        else:
            summary_parts.append("Direction sensitivity shows linear pipeline effects")
        
        # Domain pattern insight
        clustering_interp = domain_clustering.get('clustering_interpretation', '')
        if clustering_interp:
            summary_parts.append(clustering_interp)
        
        # Optimization insight
        landscape = optimization_analysis.get('optimization_landscape', 'unknown')
        if landscape == 'sharp':
            summary_parts.append("Sharp optimization landscape suggests universal optimal threshold")
        elif landscape == 'broad':
            summary_parts.append("Broad optimization landscape suggests robust threshold selection")
        else:
            summary_parts.append("Scattered optimization landscape suggests domain-specific requirements")
        
        return ". ".join(summary_parts) + "."
    
    def _save_research_results(self, all_results: List[Dict], research_insights: Dict, 
                             pipeline_analysis: Dict) -> str:
        """
        Save comprehensive research results with interpretations.
        """
        output_dir = Path("experiments/phase12/results/experiment_1_direction_sensitivity")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = output_dir / f"direction_sensitivity_analysis_{timestamp}.json"
        
        # Prepare comprehensive output
        output_data = {
            'experiment_metadata': {
                'experiment_name': 'Direction Detection Sensitivity Analysis',
                'timestamp': timestamp,
                'researcher': 'AI Research Assistant',
                'thresholds_tested': self.sensitivity_thresholds,
                'domains_tested': self.domains,
                'total_experiments': len(all_results)
            },
            'raw_results': all_results,
            'pipeline_analysis': pipeline_analysis,
            'research_insights': research_insights,
            'methodology': {
                'pipeline_stages': [
                    'Direction Detection (variable threshold)',
                    'Temporal Clustering (3-year window)',
                    'Citation Detection (CPSD algorithm)',
                    'Citation Validation (adaptive thresholds)',
                    'Segmentation (statistical significance calibrated)'
                ],
                'controlled_variables': [
                    'Clustering window (3 years)',
                    'Validation thresholds (0.5/0.7)',
                    'Citation boost (+0.3)',
                    'Domain data (identical across conditions)'
                ],
                'research_approach': 'Pipeline cascade analysis with cross-domain synthesis'
            }
        }
        
        # Save with JSON serialization safety
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2, default=str)
        
        print(f"💾 Research results saved: {output_file}")
        
        # Create visualizations
        self._create_research_visualizations(all_results, research_insights, pipeline_analysis, output_dir)
        
        return str(output_file)
    
    def _create_research_visualizations(self, all_results: List[Dict], research_insights: Dict,
                                      pipeline_analysis: Dict, output_dir: Path) -> None:
        """
        Create research-focused visualizations that tell the story of the findings.
        """
        viz_dir = output_dir / "visualizations"
        viz_dir.mkdir(exist_ok=True)
        
        # Set publication-quality style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Figure 1: Pipeline Cascade Analysis
        self._create_cascade_visualization(all_results, research_insights, viz_dir)
        
        # Figure 2: Domain Sensitivity Patterns
        self._create_domain_patterns_visualization(pipeline_analysis, viz_dir)
        
        # Figure 3: Optimization Landscape
        self._create_optimization_visualization(all_results, research_insights, viz_dir)
        
        print(f"📊 Research visualizations created in: {viz_dir}")
    
    def _create_cascade_visualization(self, all_results: List[Dict], research_insights: Dict, 
                                    viz_dir: Path) -> None:
        """
        Visualize pipeline cascade effects across sensitivity thresholds.
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Direction Sensitivity Pipeline Cascade Analysis', fontsize=16, fontweight='bold')
        
        # Prepare cascade data
        cascade_data = research_insights['cascade_amplification']['cascade_data']
        thresholds = [d['threshold'] for d in cascade_data]
        raw_signals = [d['raw_signals'] for d in cascade_data]
        clustered_signals = [d['clustered_signals'] for d in cascade_data]
        validated_signals = [d['validated_signals'] for d in cascade_data]
        final_segments = [d['final_segments'] for d in cascade_data]
        
        # Plot 1: Pipeline Stage Progression
        ax1.plot(thresholds, raw_signals, 'o-', label='Raw Direction Signals', linewidth=2, markersize=6)
        ax1.plot(thresholds, clustered_signals, 's-', label='Clustered Signals', linewidth=2, markersize=6)
        ax1.plot(thresholds, validated_signals, '^-', label='Validated Signals', linewidth=2, markersize=6)
        ax1.plot(thresholds, final_segments, 'd-', label='Final Segments', linewidth=2, markersize=6)
        ax1.set_xlabel('Direction Sensitivity Threshold')
        ax1.set_ylabel('Count')
        ax1.set_title('Pipeline Stage Progression')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Cascade Amplification
        raw_normalized = np.array(raw_signals) / max(raw_signals) if raw_signals else [0]
        segments_normalized = np.array(final_segments) / max(final_segments) if final_segments else [0]
        
        ax2.plot(thresholds, raw_normalized, 'o-', label='Raw Signals (normalized)', linewidth=2)
        ax2.plot(thresholds, segments_normalized, 'd-', label='Final Segments (normalized)', linewidth=2)
        ax2.set_xlabel('Direction Sensitivity Threshold')
        ax2.set_ylabel('Normalized Count')
        ax2.set_title('Cascade Amplification Analysis')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Add amplification factor annotation
        amp_factor = research_insights['cascade_amplification']['amplification_factor']
        ax2.text(0.7, 0.9, f'Amplification Factor: {amp_factor:.2f}x', 
                transform=ax2.transAxes, fontsize=12, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # Plot 3: Domain Variability
        domain_data = {}
        for result in all_results:
            domain = result['domain']
            if domain not in domain_data:
                domain_data[domain] = {'thresholds': [], 'segments': []}
            domain_data[domain]['thresholds'].append(result['threshold'])
            domain_data[domain]['segments'].append(result['final_segments'])
        
        for domain, data in domain_data.items():
            ax3.plot(data['thresholds'], data['segments'], 'o-', label=domain.replace('_', ' ').title(), alpha=0.7)
        
        ax3.set_xlabel('Direction Sensitivity Threshold')
        ax3.set_ylabel('Final Segments')
        ax3.set_title('Domain-Specific Sensitivity Response')
        ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Quality vs Quantity Trade-off
        for result in all_results:
            if result['temporal_accuracy'] >= 0:
                ax4.scatter(result['final_segments'], result['temporal_accuracy'], 
                          c=result['threshold'], cmap='viridis', alpha=0.7, s=50)
        
        scatter = ax4.scatter([], [], c=[], cmap='viridis')
        cbar = plt.colorbar(scatter, ax=ax4)
        cbar.set_label('Sensitivity Threshold')
        ax4.set_xlabel('Final Segments')
        ax4.set_ylabel('Temporal Accuracy (years)')
        ax4.set_title('Quality vs Quantity Trade-off')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(viz_dir / "cascade_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_domain_patterns_visualization(self, pipeline_analysis: Dict, viz_dir: Path) -> None:
        """
        Visualize domain clustering patterns and characteristics.
        """
        if not pipeline_analysis:
            return
            
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Domain Sensitivity Patterns Analysis', fontsize=16, fontweight='bold')
        
        # Extract domain data
        domains = list(pipeline_analysis.keys())
        optimal_thresholds = [pipeline_analysis[d].get('optimal_threshold_estimate', 0.4) for d in domains]
        responsiveness = [pipeline_analysis[d].get('sensitivity_responsiveness', 0) for d in domains]
        citation_support = [pipeline_analysis[d].get('citation_support_availability', 0) for d in domains]
        validation_acceptance = [pipeline_analysis[d].get('validation_acceptance_mean', 0) for d in domains]
        
        # Plot 1: Optimal Threshold Distribution
        ax1.bar(range(len(domains)), optimal_thresholds, color='skyblue', alpha=0.7)
        ax1.set_xticks(range(len(domains)))
        ax1.set_xticklabels([d.replace('_', '\n') for d in domains], rotation=45, ha='right')
        ax1.set_ylabel('Optimal Threshold')
        ax1.set_title('Domain-Specific Optimal Thresholds')
        ax1.axhline(y=0.4, color='red', linestyle='--', alpha=0.7, label='Current Default (0.4)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Sensitivity Responsiveness
        ax2.bar(range(len(domains)), responsiveness, color='lightgreen', alpha=0.7)
        ax2.set_xticks(range(len(domains)))
        ax2.set_xticklabels([d.replace('_', '\n') for d in domains], rotation=45, ha='right')
        ax2.set_ylabel('Sensitivity Responsiveness')
        ax2.set_title('Domain Sensitivity to Threshold Changes')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Citation Support vs Validation Acceptance
        ax3.scatter(citation_support, validation_acceptance, s=100, alpha=0.7, c='orange')
        for i, domain in enumerate(domains):
            ax3.annotate(domain.replace('_', '\n'), (citation_support[i], validation_acceptance[i]),
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        ax3.set_xlabel('Citation Support Availability')
        ax3.set_ylabel('Validation Acceptance Rate')
        ax3.set_title('Citation Support vs Validation Success')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Domain Characteristics Heatmap
        characteristics_data = np.array([optimal_thresholds, responsiveness, citation_support, validation_acceptance])
        
        im = ax4.imshow(characteristics_data, cmap='RdYlBu_r', aspect='auto')
        ax4.set_xticks(range(len(domains)))
        ax4.set_xticklabels([d.replace('_', '\n') for d in domains], rotation=45, ha='right')
        ax4.set_yticks(range(4))
        ax4.set_yticklabels(['Optimal Threshold', 'Responsiveness', 'Citation Support', 'Validation Rate'])
        ax4.set_title('Domain Characteristics Heatmap')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax4)
        cbar.set_label('Normalized Value')
        
        plt.tight_layout()
        plt.savefig(viz_dir / "domain_patterns.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_optimization_visualization(self, all_results: List[Dict], research_insights: Dict,
                                         viz_dir: Path) -> None:
        """
        Visualize optimization landscape and threshold recommendations.
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Sensitivity Threshold Optimization Analysis', fontsize=16, fontweight='bold')
        
        # Group results for optimization analysis
        by_threshold = {}
        for result in all_results:
            threshold = result['threshold']
            if threshold not in by_threshold:
                by_threshold[threshold] = []
            by_threshold[threshold].append(result)
        
        # Plot 1: Mean Performance by Threshold
        thresholds = sorted(by_threshold.keys())
        mean_segments = []
        std_segments = []
        mean_accuracy = []
        
        for threshold in thresholds:
            results = by_threshold[threshold]
            segments = [r['final_segments'] for r in results]
            accuracies = [r['temporal_accuracy'] for r in results if r['temporal_accuracy'] >= 0]
            
            mean_segments.append(np.mean(segments))
            std_segments.append(np.std(segments))
            mean_accuracy.append(np.mean(accuracies) if accuracies else -1)
        
        ax1.errorbar(thresholds, mean_segments, yerr=std_segments, 
                    marker='o', linewidth=2, capsize=5, label='Final Segments')
        ax1.set_xlabel('Sensitivity Threshold')
        ax1.set_ylabel('Mean Final Segments')
        ax1.set_title('Performance vs Sensitivity Threshold')
        ax1.grid(True, alpha=0.3)
        
        # Mark current default
        ax1.axvline(x=0.4, color='red', linestyle='--', alpha=0.7, label='Current Default')
        ax1.legend()
        
        # Plot 2: Quality-Quantity Pareto Frontier
        pareto_points = []
        for result in all_results:
            if result['temporal_accuracy'] >= 0:
                pareto_points.append((result['final_segments'], result['temporal_accuracy'], result['threshold']))
        
        if pareto_points:
            segments, accuracies, threshold_colors = zip(*pareto_points)
            scatter = ax2.scatter(segments, accuracies, c=threshold_colors, cmap='viridis', alpha=0.7, s=50)
            plt.colorbar(scatter, ax=ax2, label='Threshold')
            ax2.set_xlabel('Final Segments')
            ax2.set_ylabel('Temporal Accuracy (years)')
            ax2.set_title('Quality-Quantity Pareto Frontier')
            ax2.grid(True, alpha=0.3)
        
        # Plot 3: Domain-Specific Recommendations
        domain_optima = research_insights['optimization_landscape']['domain_optima']
        if domain_optima:
            domains = [d['domain'] for d in domain_optima]
            opt_thresholds = [d['optimal_threshold'] for d in domain_optima]
            
            bars = ax3.bar(range(len(domains)), opt_thresholds, color='lightcoral', alpha=0.7)
            ax3.set_xticks(range(len(domains)))
            ax3.set_xticklabels([d.replace('_', '\n') for d in domains], rotation=45, ha='right')
            ax3.set_ylabel('Optimal Threshold')
            ax3.set_title('Domain-Specific Optimal Thresholds')
            ax3.axhline(y=0.4, color='blue', linestyle='--', alpha=0.7, label='Current Default')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar, threshold in zip(bars, opt_thresholds):
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{threshold:.2f}', ha='center', va='bottom', fontweight='bold')
        
        # Plot 4: Optimization Landscape Summary
        default_validation = research_insights['current_default_validation']
        
        # Create summary text
        summary_text = f"""
Current Default Validation: {default_validation['validation'].upper()}

Recommended Threshold: {default_validation['recommended_threshold']:.2f}
Performance Gap: {default_validation.get('performance_gap', 1.0):.2f}x

Optimization Landscape: {research_insights['optimization_landscape']['optimization_landscape'].upper()}

Research Summary:
{research_insights['research_summary']}
        """.strip()
        
        ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        ax4.set_xlim(0, 1)
        ax4.set_ylim(0, 1)
        ax4.axis('off')
        ax4.set_title('Research Summary & Recommendations')
        
        plt.tight_layout()
        plt.savefig(viz_dir / "optimization_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()


def main():
    """
    Execute Direction Sensitivity Experiment with research rigor.
    """
    print("🔬 STARTING EXPERIMENT 1: DIRECTION DETECTION SENSITIVITY ANALYSIS")
    print("Research Approach: Critical analysis of pipeline cascade effects and optimization")
    print()
    
    experiment = DirectionSensitivityExperiment()
    results = experiment.run_sensitivity_analysis()
    
    print("\n" + "="*60)
    print("📋 EXPERIMENT 1 COMPLETED")
    print(f"📁 Results file: {results['results_file']}")
    print(f"🧪 Total experiments: {results['total_experiments']}")
    print("\n🎯 KEY RESEARCH INSIGHTS:")
    print(results['research_insights']['research_summary'])
    print("\n💡 Ready for researcher interpretation and synthesis!")


if __name__ == "__main__":
    main() 