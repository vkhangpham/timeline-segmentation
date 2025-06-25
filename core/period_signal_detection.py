"""
Period Signal Detection for Timeline Analysis

This module implements temporal network stability analysis to characterize research periods
through community dynamics, collaboration persistence, and network evolution patterns.

Core functionality:
- Temporal network stability analysis
- Community persistence detection  
- Flow stability measurement
- Network centrality-based paper selection
- LLM-enhanced period labeling

Follows functional programming principles with pure functions and immutable data structures.
"""

from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict, Counter
import numpy as np
import json
from pathlib import Path
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
import xml.etree.ElementTree as ET
from .paper_selection_and_labeling import (
    select_representative_papers, 
    generate_period_label_and_description
)
from .data_models import PeriodCharacterization


def characterize_periods(domain_name: str, segments: List[Tuple[int, int]]) -> List[PeriodCharacterization]:
    """
    Main function: Characterize research periods using temporal network analysis
    
    Args:
        domain_name: Name of the research domain
        segments: List of time segments from shift signal detection
    
    Returns:
        List of period characterizations
    """
    # Load rich data sources
    papers_data = load_papers_data(domain_name)
    semantic_citations = load_semantic_citations(domain_name)
    breakthrough_papers = load_breakthrough_papers(domain_name)
    citation_network = build_citation_network(papers_data, semantic_citations, breakthrough_papers)
    
    # Initialize TF-IDF vectorizer
    tfidf_vectorizer = TfidfVectorizer(
        max_features=10000,
        stop_words='english',
        ngram_range=(1, 3),
        min_df=2
    )
    
    period_characterizations = []
    period_analysis_data = []  # Store detailed analysis data for visualization
    
    for start_year, end_year in segments:
        print(f"\nCharacterizing period {start_year}-{end_year} with network analysis...")
        
        # Get papers and build period subnetwork
        period_papers = get_papers_in_period(papers_data, start_year, end_year, breakthrough_papers)
        if len(period_papers) < 3:
            print(f"Insufficient papers ({len(period_papers)}) for network analysis")
            continue
        
        period_subnetwork = build_period_subnetwork(citation_network, period_papers, start_year, end_year)
        
        # Analyze temporal network stability
        network_stability = analyze_network_stability(period_subnetwork)
        
        # Measure community persistence
        community_persistence = measure_community_persistence(period_subnetwork)
        
        # Analyze flow stability
        flow_stability = analyze_flow_stability(period_subnetwork)
        
        # Calculate centrality consensus
        centrality_consensus = calculate_centrality_consensus(period_subnetwork)
        
        # Enhanced theme detection using network structure
        dominant_themes = detect_network_themes(period_papers, period_subnetwork, tfidf_vectorizer)
        
        # Network centrality-based paper selection
        representative_papers = select_representative_papers(
            period_papers, period_subnetwork, dominant_themes
        )
        
        # Calculate comprehensive network metrics
        network_metrics = calculate_network_metrics(period_subnetwork)
        
        # Build previous periods context for progression
        previous_periods = []
        for prev_char in period_characterizations:
            previous_periods.append((start_year, end_year, prev_char.topic_label, prev_char.topic_description))
        
        # Generate period label and description
        period_label, period_description = generate_period_label_and_description(
            dominant_themes, representative_papers, start_year, end_year,
            previous_periods=previous_periods, domain_name=domain_name
        )
        
        # Calculate confidence score
        confidence = calculate_confidence(
            network_stability, community_persistence, flow_stability, 
            centrality_consensus, len(period_papers), network_metrics
        )
        
        # Create characterization
        characterization = PeriodCharacterization(
            period=(start_year, end_year),
            topic_label=period_label,
            topic_description=period_description,
            network_stability=network_stability,
            community_persistence=community_persistence,
            flow_stability=flow_stability,
            centrality_consensus=centrality_consensus,
            representative_papers=tuple(representative_papers),
            network_metrics=network_metrics,
            confidence=confidence
        )
        
        period_characterizations.append(characterization)
        
        # Store detailed analysis data for visualization
        period_analysis_data.append({
            'period': (start_year, end_year),
            'num_papers': len(period_papers),
            'num_breakthrough_papers': sum(1 for p in period_papers if p['is_breakthrough']),
            'network_stability': network_stability,
            'community_persistence': community_persistence,
            'flow_stability': flow_stability,
            'centrality_consensus': centrality_consensus,
            'dominant_themes': dominant_themes,
            'representative_papers': representative_papers,
            'network_metrics': network_metrics,
            'confidence': confidence,
            'period_label': period_label,
            'period_description': period_description
        })
        
        print(f"Period {start_year}-{end_year}: stability={network_stability:.3f}, "
              f"persistence={community_persistence:.3f}, confidence={confidence:.3f}")
        print(f"    {period_label}: {period_description}")
    
    return period_characterizations


def load_papers_data(domain_name: str) -> Dict[str, Any]:
    """Load paper documents with abstracts and metadata"""
    data_dir = Path(f"resources/{domain_name}")
    papers_file = data_dir / f"{domain_name}_docs_info.json"
    if not papers_file.exists():
        return {}
    
    with open(papers_file, 'r') as f:
        papers_data = json.load(f)
    
    if isinstance(papers_data, dict):
        return papers_data
    else:
        # Convert list to indexed format
        papers_index = {}
        for paper in papers_data:
            paper_id = paper.get('openalex_id', '')
            if paper_id:
                papers_index[paper_id] = paper
        return papers_index


def load_semantic_citations(domain_name: str) -> List[Dict[str, Any]]:
    """Load semantic citation descriptions from GraphML"""
    data_dir = Path(f"resources/{domain_name}")
    citations_file_xml = data_dir / f"{domain_name}_entity_relation_graph.graphml.xml"
    
    if not citations_file_xml.exists():
        return []
    
    semantic_citations = []
    try:
        tree = ET.parse(citations_file_xml)
        root = tree.getroot()
        
        ns = {"": "http://graphml.graphdrawing.org/xmlns"}
        
        for edge in root.findall(".//edge", ns):
            source = edge.get("source")
            target = edge.get("target")
            
            description = ""
            for data in edge.findall("data", ns):
                if data.get("key") == "d4" and data.text:
                    description = data.text.strip()
                    break
            
            if source and target and description:
                semantic_citations.append({
                    'parent_id': source,
                    'child_id': target,
                    'description': description,
                    'relationship_type': 'cites'
                })
    
    except Exception as e:
        print(f"Error parsing GraphML citations: {e}")
        return []
    
    return semantic_citations


def load_breakthrough_papers(domain_name: str) -> Dict[str, Any]:
    """Load breakthrough papers for significance weighting"""
    data_dir = Path(f"resources/{domain_name}")
    breakthrough_file = data_dir / f"{domain_name}_breakthrough_papers.jsonl"
    if not breakthrough_file.exists():
        return {}
    
    breakthrough_papers = {}
    with open(breakthrough_file, 'r') as f:
        for line in f:
            if line.strip():
                paper_data = json.loads(line.strip())
                paper_id = paper_data.get('openalex_id', '')
                if paper_id:
                    breakthrough_papers[paper_id] = paper_data
    
    return breakthrough_papers


def build_citation_network(papers_data: Dict[str, Any], semantic_citations: List[Dict[str, Any]], 
                           breakthrough_papers: Dict[str, Any]) -> nx.DiGraph:
    """Build citation network for temporal analysis"""
    G = nx.DiGraph()
    
    # Add nodes (papers) with temporal information
    for paper_id, paper_data in papers_data.items():
        G.add_node(paper_id, 
                  pub_year=paper_data.get('pub_year', 0),
                  cited_by_count=paper_data.get('cited_by_count', 0),
                  title=paper_data.get('title', ''),
                  is_breakthrough=paper_id in breakthrough_papers)
    
    # Add edges (citations) with semantic descriptions
    for citation in semantic_citations:
        parent_id = citation['parent_id']
        child_id = citation['child_id']
        if parent_id in papers_data and child_id in papers_data:
            G.add_edge(parent_id, child_id, 
                      description=citation['description'],
                      relationship_type=citation['relationship_type'])
    
    return G


def get_papers_in_period(papers_data: Dict[str, Any], start_year: int, end_year: int, 
                         breakthrough_papers: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Get all papers published within the specified period"""
    period_papers = []
    
    for paper_id, paper_data in papers_data.items():
        pub_year = paper_data.get('pub_year', 0)
        if start_year <= pub_year <= end_year:
            period_papers.append({
                'id': paper_id,
                'data': paper_data,
                'is_breakthrough': paper_id in breakthrough_papers
            })
    
    return period_papers


def build_period_subnetwork(citation_network: nx.DiGraph, period_papers: List[Dict], 
                           start_year: int, end_year: int) -> nx.DiGraph:
    """Build subnetwork for the specific period"""
    paper_ids = {p['id'] for p in period_papers}
    
    # Create subgraph with papers from this period
    period_subgraph = citation_network.subgraph(paper_ids).copy()
    
    # Add temporal windows for analysis
    for node in period_subgraph.nodes():
        node_data = period_subgraph.nodes[node]
        pub_year = node_data.get('pub_year', 0)
        # Add temporal position within period
        if end_year > start_year:
            temporal_position = (pub_year - start_year) / (end_year - start_year)
        else:
            temporal_position = 0.5
        period_subgraph.nodes[node]['temporal_position'] = temporal_position
    
    return period_subgraph


def analyze_network_stability(subnetwork: nx.DiGraph) -> float:
    """Analyze temporal network stability using multiple metrics"""
    if subnetwork.number_of_nodes() < 3:
        return 0.0
    
    stability_metrics = []
    
    # Degree distribution stability
    try:
        degrees = [d for n, d in subnetwork.degree()]
        if degrees:
            degree_variance = np.var(degrees)
            degree_stability = 1.0 / (1.0 + degree_variance)
            stability_metrics.append(degree_stability)
    except:
        pass
    
    # Clustering coefficient stability
    try:
        clustering = nx.average_clustering(subnetwork.to_undirected())
        stability_metrics.append(clustering)
    except:
        pass
    
    # Connected components stability
    try:
        undirected = subnetwork.to_undirected()
        num_components = nx.number_connected_components(undirected)
        total_nodes = subnetwork.number_of_nodes()
        component_stability = 1.0 - (num_components / total_nodes)
        stability_metrics.append(max(0.0, component_stability))
    except:
        pass
    
    return np.mean(stability_metrics) if stability_metrics else 0.0


def measure_community_persistence(subnetwork: nx.DiGraph) -> float:
    """Measure community persistence using community detection"""
    if subnetwork.number_of_nodes() < 4:
        return 0.0
    
    try:
        undirected = subnetwork.to_undirected()
        
        # Simple connected components as communities
        communities = {}
        for i, component in enumerate(nx.connected_components(undirected)):
            for node in component:
                communities[node] = i
        
        if not communities:
            return 0.0
        
        # Analyze community structure quality
        num_communities = len(set(communities.values()))
        total_nodes = len(communities)
        
        if num_communities == 0:
            return 0.0
        
        community_ratio = num_communities / total_nodes
        optimal_ratio = 0.3
        ratio_score = 1.0 - abs(community_ratio - optimal_ratio)
        
        # Size balance score
        community_sizes = Counter(communities.values())
        size_variance = np.var(list(community_sizes.values()))
        balance_score = 1.0 / (1.0 + size_variance)
        
        persistence_score = (ratio_score + balance_score) / 2
        return max(0.0, min(1.0, persistence_score))
        
    except Exception:
        return 0.0


def analyze_flow_stability(subnetwork: nx.DiGraph) -> float:
    """Analyze citation flow stability within the period"""
    if subnetwork.number_of_edges() < 2:
        return 0.0
    
    flow_metrics = []
    
    # In-degree distribution stability
    try:
        in_degrees = [d for n, d in subnetwork.in_degree()]
        if in_degrees and len(in_degrees) > 1:
            in_degree_cv = np.std(in_degrees) / (np.mean(in_degrees) + 1e-6)
            in_degree_stability = 1.0 / (1.0 + in_degree_cv)
            flow_metrics.append(in_degree_stability)
    except:
        pass
    
    # Network density as flow indicator
    try:
        density = nx.density(subnetwork)
        optimal_density = 0.1
        density_score = 1.0 - abs(density - optimal_density)
        flow_metrics.append(max(0.0, density_score))
    except:
        pass
    
    return np.mean(flow_metrics) if flow_metrics else 0.0


def calculate_centrality_consensus(subnetwork: nx.DiGraph) -> float:
    """Calculate consensus based on centrality measures"""
    if subnetwork.number_of_nodes() < 3:
        return 0.0
    
    centrality_metrics = []
    
    # PageRank centrality distribution
    try:
        pagerank = nx.pagerank(subnetwork)
        pr_values = list(pagerank.values())
        if pr_values:
            pr_variance = np.var(pr_values)
            optimal_variance = 0.01
            pr_score = 1.0 - abs(pr_variance - optimal_variance)
            centrality_metrics.append(max(0.0, pr_score))
    except:
        pass
    
    # Betweenness centrality
    try:
        betweenness = nx.betweenness_centrality(subnetwork)
        bc_values = list(betweenness.values())
        if bc_values:
            bc_variance = np.var(bc_values)
            bc_score = 1.0 / (1.0 + bc_variance * 10)
            centrality_metrics.append(bc_score)
    except:
        pass
    
    return np.mean(centrality_metrics) if centrality_metrics else 0.0


def detect_network_themes(period_papers: List[Dict], subnetwork: nx.DiGraph, 
                          tfidf_vectorizer: TfidfVectorizer) -> List[str]:
    """Detect dominant themes using network structure and content"""
    themes = []
    
    # Extract content from papers
    paper_contents = []
    for paper in period_papers:
        content = paper['data'].get('content', '') + ' ' + paper['data'].get('title', '')
        if content.strip():
            paper_contents.append(content)
    
    if not paper_contents:
        return ["Network Analysis"]
    
    try:
        # TF-IDF analysis for theme extraction
        tfidf_matrix = tfidf_vectorizer.fit_transform(paper_contents)
        feature_names = tfidf_vectorizer.get_feature_names_out()
        
        # Get top terms
        mean_scores = np.mean(tfidf_matrix.toarray(), axis=0)
        top_indices = np.argsort(mean_scores)[-10:]
        
        top_terms = [feature_names[i] for i in top_indices]
        themes.extend(top_terms[-3:])  # Top 3 themes
        
    except Exception:
        # Fallback to simple keyword extraction
        all_text = ' '.join(paper_contents).lower()
        common_terms = ['neural', 'learning', 'network', 'algorithm', 'model', 'deep', 'machine']
        themes = [term for term in common_terms if term in all_text][:3]
    
    return themes if themes else ["Research Period"]


def calculate_network_metrics(subnetwork: nx.DiGraph) -> Dict[str, float]:
    """Calculate comprehensive network metrics for the period"""
    metrics = {}
    
    try:
        metrics['density'] = nx.density(subnetwork)
        metrics['number_of_nodes'] = subnetwork.number_of_nodes()
        metrics['number_of_edges'] = subnetwork.number_of_edges()
        
        if subnetwork.number_of_nodes() > 0:
            undirected = subnetwork.to_undirected()
            metrics['average_clustering'] = nx.average_clustering(undirected)
            metrics['number_of_components'] = nx.number_connected_components(undirected)
            
            # Degree centralization
            degrees = [d for n, d in subnetwork.degree()]
            if degrees:
                max_degree = max(degrees)
                sum_diff = sum(max_degree - d for d in degrees)
                n = len(degrees)
                max_sum_diff = (n - 1) * (n - 2)
                metrics['degree_centralization'] = sum_diff / max_sum_diff if max_sum_diff > 0 else 0.0
        
    except Exception as e:
        print(f"Error calculating network metrics: {e}")
        metrics = {'density': 0.0, 'number_of_nodes': 0, 'number_of_edges': 0}
    
    return metrics


def calculate_confidence(network_stability: float, community_persistence: float,
                        flow_stability: float, centrality_consensus: float,
                        num_papers: int, network_metrics: Dict[str, float]) -> float:
    """Calculate overall confidence score for period characterization"""
    
    # Base confidence from network analysis metrics
    base_confidence = (network_stability + community_persistence + 
                      flow_stability + centrality_consensus) / 4
    
    # Paper count bonus (more papers = higher confidence)
    paper_bonus = min(0.2, num_papers / 50.0)  # Up to 0.2 bonus for 50+ papers
    
    # Network connectivity bonus
    density = network_metrics.get('density', 0.0)
    connectivity_bonus = min(0.1, density * 5.0)  # Up to 0.1 bonus for good connectivity

    final_confidence = min(1.0, base_confidence + paper_bonus + connectivity_bonus)
    
    return final_confidence