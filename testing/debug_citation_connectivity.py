"""
Debug Script for Citation Connectivity Signal

This script is a targeted tool to investigate why the `citation_connectivity`
signal is failing, specifically for the two "Statistical NLP" periods in the
NLP domain, which should be merged but are not due to a 0.0 connectivity score.

**Debugging Steps:**
1.  Define the exact periods for "Statistical Approaches" (1994-1997) and
    "The Statistical Revolution" (1998-2003).
2.  Load all necessary NLP domain data, including the full paper list and the
    citation graph.
3.  Characterize *only* these two specific periods to get their representative
    papers, mimicking the main analysis pipeline.
4.  Print the titles and IDs of the representative papers for each period to allow
    for intuitive, human-readable verification.
5.  Directly query the citation graph to check for edges (citations) between the
    two sets of representative papers.
6.  Log a clear, step-by-step analysis of the findings to the console to
    pinpoint the exact source of the failure.
"""

import sys
from pathlib import Path
import networkx as nx

# Ensure the project root is in the Python path
sys.path.append(str(Path(__file__).parent.parent))

from core.data_processing import process_domain_data
from core.period_signal_detection import characterize_periods

def debug_nlp_connectivity():
    """Main function to run the connectivity debugging process."""
    domain_name = "natural_language_processing"
    print(f"--- Debugging Citation Connectivity for: {domain_name} ---")

    # --- Step 1: Define the specific segments to analyze ---
    # These are the segments that failed to merge in our test.
    # The `characterize_periods` function expects a list of tuples.
    statistical_periods = [
        (1994, 1997),
        (1998, 2003)
    ]
    print("\n[Step 1/5] Defined target segments for analysis:")
    print(f"  - Segment 1: {statistical_periods[0]}")
    print(f"  - Segment 2: {statistical_periods[1]}")

    # --- Step 2: Load all domain data ---
    print("\n[Step 2/5] Loading all domain data...")
    processing_result = process_domain_data(domain_name)
    domain_data = processing_result.domain_data
    
    # Reconstruct the graph from citation relations
    graph = nx.DiGraph()
    for rel in domain_data.citations:
        graph.add_edge(rel.cited_paper_id, rel.citing_paper_id)
    
    print(f"✅ Loaded data: {len(domain_data.papers)} papers, {graph.number_of_edges()} citations.")

    # --- Step 3: Characterize the two periods to get representative papers ---
    print("\n[Step 3/5] Characterizing periods to get representative papers...")
    period_characterizations = characterize_periods(
        domain_name=domain_name,
        segments=statistical_periods
    )
    
    if len(period_characterizations) != 2:
        raise ValueError("Expected characterizations for 2 periods, but got a different number.")
        
    segment1_char = period_characterizations[0]
    segment2_char = period_characterizations[1]
    
    s1_rep_papers = segment1_char.representative_papers
    s2_rep_papers = segment2_char.representative_papers
    
    print("✅ Characterization complete.")

    # --- Step 4: Print representative papers for manual review ---
    print("\n[Step 4/5] Representative Papers for Each Period:")
    print("-" * 50)
    print(f"Period 1: {segment1_char.period} ('{segment1_char.topic_label}')")
    for paper in s1_rep_papers:
        print(f"  - [{paper.get('id')}] {paper.get('title')}")
    
    print("-" * 50)
    print(f"Period 2: {segment2_char.period} ('{segment2_char.topic_label}')")
    for paper in s2_rep_papers:
        print(f"  - [{paper.get('id')}] {paper.get('title')}")
    print("-" * 50)

    # --- Step 5: Manually check for citation links in the graph ---
    print("\n[Step 5/5] Checking for Citation Links Between Representative Sets...")
    
    s1_ids = {p.get('id') for p in s1_rep_papers if p.get('id')}
    s2_ids = {p.get('id') for p in s2_rep_papers if p.get('id')}

    found_citations = 0
    
    if not s1_ids or not s2_ids:
        print("❌ ERROR: One or both representative paper sets are empty. Cannot check connectivity.")
    else:
        for s2_id in s2_ids:
            for s1_id in s1_ids:
                # IMPORTANT: In our graph, an edge s1 -> s2 means s2 cites s1.
                # So we check for edges from the *earlier* period paper (s1) to the *later* one (s2).
                if graph.has_edge(s1_id, s2_id):
                    s1_title = next((p['title'] for p in s1_rep_papers if p['id'] == s1_id), "Unknown")
                    s2_title = next((p['title'] for p in s2_rep_papers if p['id'] == s2_id), "Unknown")
                    print(f"  ✅ FOUND CITATION:")
                    print(f"     '{s2_title}' (from {segment2_char.period})")
                    print(f"     ==> cites ==>")
                    print(f"     '{s1_title}' (from {segment1_char.period})")
                    found_citations += 1

    print("-" * 50)
    if found_citations > 0:
        print(f"✅ SUCCESS: Found {found_citations} direct citation(s) between the representative paper sets.")
        print("   This means the connectivity logic in `segment_merging.py` is likely flawed.")
    else:
        print("❌ FAILURE: Found NO direct citations between the representative paper sets.")
        print("   This suggests the problem lies in the 'representative paper' selection process,")
        print("   or the periods are genuinely disconnected (which is unlikely).")
        
if __name__ == "__main__":
    debug_nlp_connectivity() 