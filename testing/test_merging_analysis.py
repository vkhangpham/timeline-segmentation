"""
Test Script for Enhanced Segment Merging Analysis

This script orchestrates the testing of the new 3-signal fusion model for
segment merging. It specifically targets the NLP domain to perform a
thorough analysis before extending the model to other topics.

**Functionality:**
1.  Loads the NLP domain data.
2.  Runs the existing change detection and segment modeling pipeline to get
    initial segments and their characterizations.
3.  For each consecutive pair of segments, it computes the three new merge signals:
    - Citation Connectivity
    - Semantic Description Similarity
    - Inverse Boundary Signal Strength
4.  It calculates a final weighted score based on the agreed-upon fusion model.
5.  It logs all intermediate and final scores to a structured JSON file for
    meticulous review and critical evaluation.
6.  This script performs ANALYSIS ONLY. It does not perform any actual merging.
"""

import json
from pathlib import Path
import sys
import networkx as nx

# Add project root to path to allow importing core modules
sys.path.append(str(Path(__file__).parent.parent))

from core.data_processing import process_domain_data
from core.change_detection import detect_changes, create_segments_with_confidence
from core.period_signal_detection import characterize_periods
from core.segment_merging import analyze_merging_opportunities
from core.data_models import DomainData, PeriodCharacterization, ShiftSignal

def run_nlp_merging_analysis():
    """
    Main function to run the merging analysis for the NLP domain.
    """
    print("🔬 Starting Enhanced Segment Merging Analysis for NLP Domain...")
    
    # 1. Load Data
    print("\n[Step 1/4] Loading NLP domain data...")
    domain_name = "natural_language_processing"
    
    try:
        processing_result = process_domain_data(domain_name)
        domain_data = processing_result.domain_data
        
        # Build the citation graph from the citation relations
        graph = nx.DiGraph()
        for citation in domain_data.citations:
            graph.add_edge(citation.cited_paper_id, citation.citing_paper_id)

    except FileNotFoundError:
        print(f"❌ Failed to load data for domain '{domain_name}'. Make sure resource files exist.")
        return

    print(f"✅ Loaded data for {domain_data.total_papers} papers.")

    # 2. Detect Changes and Create Initial Segments
    print("\n[Step 2/4] Detecting changes and creating initial segments...")
    change_detection_result = detect_changes(domain_data)
    change_years = [cp.year for cp in change_detection_result.change_points]
    
    segments = create_segments_with_confidence(
        change_years,
        domain_data.year_range,
        change_detection_result.statistical_significance,
        domain_data.domain_name
    )
    print(f"✅ Found {len(segments)} initial segments.")

    # 3. Characterize Segments (Period Signal Detection)
    print("\n[Step 3/4] Characterizing initial segments...")
    period_characterizations = characterize_periods(
        domain_name=domain_name,
        segments=segments
    )
    print(f"✅ Characterized {len(period_characterizations)} segments.")

    # 4. Analyze Merging Opportunities with the New 3-Signal Model
    print("\n[Step 4/4] Analyzing merging opportunities...")
    
    # We need the original shift signals for boundary strength analysis
    shift_signals = [
        ShiftSignal(
            year=cp.year,
            confidence=cp.confidence,
            signal_type=cp.signal_type,
            evidence_strength=0, # Not needed for this test
            supporting_evidence=cp.supporting_evidence,
            contributing_papers=tuple(), # Not needed for this test
            transition_description=cp.description,
            paradigm_significance=0 # Not needed for this test
        ) for cp in change_detection_result.change_points
    ]
    
    analysis_results = analyze_merging_opportunities(
        period_characterizations,
        shift_signals,
        graph
    )
    
    print("✅ Analysis complete.")

    # Save results to a file for review
    output_dir = Path("results/merging_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{domain_name}_merging_analysis.json"
    
    with open(output_file, 'w') as f:
        json.dump(analysis_results, f, indent=2)

    print(f"\n📈 Successfully saved detailed analysis to: {output_file}")
    print("Please review this file to critically evaluate the new merging model's performance.")


if __name__ == "__main__":
    run_nlp_merging_analysis() 