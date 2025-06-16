#!/usr/bin/env python3
"""
Comprehensive analysis of timeline data resources
"""

import json
import os
from pathlib import Path
import xml.etree.ElementTree as ET

def analyze_docs_info(domain):
    """Analyze docs_info.json structure"""
    file_path = f'resources/{domain}/{domain}_docs_info.json'
    
    if not os.path.exists(file_path):
        return None
        
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # Get sample papers
    sample_keys = list(data.keys())[:3]
    samples = {key: data[key] for key in sample_keys}
    
    return {
        'total_papers': len(data),
        'file_size_mb': round(os.path.getsize(file_path) / 1024 / 1024, 2),
        'sample_structure': samples
    }

def analyze_breakthrough_papers(domain):
    """Analyze breakthrough papers JSONL"""
    file_path = f'resources/{domain}/{domain}_breakthrough_papers.jsonl'
    
    if not os.path.exists(file_path):
        return None
        
    papers = []
    count = 0
    
    with open(file_path, 'r') as f:
        for line in f:
            if line.strip():
                paper = json.loads(line.strip())
                count += 1
                if count <= 3:  # Sample first 3
                    papers.append(paper)
    
    return {
        'total_breakthrough_papers': count,
        'file_size_mb': round(os.path.getsize(file_path) / 1024 / 1024, 2),
        'sample_papers': papers
    }

def analyze_citation_graph(domain):
    """Analyze entity relation graph"""
    file_path = f'resources/{domain}/{domain}_entity_relation_graph.graphml.xml'
    
    if not os.path.exists(file_path):
        return None
    
    try:
        tree = ET.parse(file_path)
        root = tree.getroot()
        
        # Count nodes and edges
        namespace = {'graphml': 'http://graphml.graphdrawing.org/xmlns'}
        nodes = root.findall('.//graphml:node', namespace)
        edges = root.findall('.//graphml:edge', namespace)
        
        # Get sample edges with descriptions
        sample_edges = []
        for edge in edges[:3]:
            edge_data = {}
            for data in edge.findall('graphml:data', namespace):
                key = data.get('key')
                if key == 'd2':  # Usually the description key
                    edge_data['description'] = data.text
            sample_edges.append(edge_data)
        
        return {
            'total_nodes': len(nodes),
            'total_edges': len(edges),
            'file_size_mb': round(os.path.getsize(file_path) / 1024 / 1024, 2),
            'sample_edges': sample_edges
        }
    except Exception as e:
        return {'error': str(e)}

def main():
    domains = [
        'natural_language_processing',
        'deep_learning',
        'computer_vision',
        'machine_translation',
        'machine_learning',
        'applied_mathematics',
        'art'
    ]
    
    print("=" * 80)
    print("COMPREHENSIVE DATA RESOURCES ANALYSIS")
    print("=" * 80)
    
    for domain in domains:
        print(f"\n{'='*50}")
        print(f"DOMAIN: {domain.upper()}")
        print(f"{'='*50}")
        
        # Analyze docs_info
        docs_analysis = analyze_docs_info(domain)
        if docs_analysis:
            print(f"\n📄 DOCS_INFO.JSON:")
            print(f"  • Total papers: {docs_analysis['total_papers']:,}")
            print(f"  • File size: {docs_analysis['file_size_mb']} MB")
            
            # Show sample paper structure
            sample_paper = list(docs_analysis['sample_structure'].values())[0]
            print(f"  • Sample paper fields:")
            for field, value in sample_paper.items():
                if field == 'content':
                    print(f"    - {field}: {len(value)} chars")
                elif field == 'keywords':
                    print(f"    - {field}: {len(value)} keywords")
                elif field == 'children':
                    print(f"    - {field}: {len(value)} citations")
                else:
                    print(f"    - {field}: {value}")
        
        # Analyze breakthrough papers
        breakthrough_analysis = analyze_breakthrough_papers(domain)
        if breakthrough_analysis:
            print(f"\n🌟 BREAKTHROUGH_PAPERS.JSONL:")
            print(f"  • Total breakthrough papers: {breakthrough_analysis['total_breakthrough_papers']:,}")
            print(f"  • File size: {breakthrough_analysis['file_size_mb']} MB")
            
            if breakthrough_analysis['sample_papers']:
                sample_paper = breakthrough_analysis['sample_papers'][0]
                print(f"  • Sample breakthrough paper fields:")
                for field, value in sample_paper.items():
                    if isinstance(value, str) and len(value) > 100:
                        print(f"    - {field}: {len(value)} chars")
                    elif isinstance(value, list):
                        print(f"    - {field}: {len(value)} items")
                    else:
                        print(f"    - {field}: {value}")
        
        # Analyze citation graph
        graph_analysis = analyze_citation_graph(domain)
        if graph_analysis:
            print(f"\n🔗 ENTITY_RELATION_GRAPH.GRAPHML.XML:")
            if 'error' in graph_analysis:
                print(f"  • Error: {graph_analysis['error']}")
            else:
                print(f"  • Total nodes: {graph_analysis['total_nodes']:,}")
                print(f"  • Total edges: {graph_analysis['total_edges']:,}")
                print(f"  • File size: {graph_analysis['file_size_mb']} MB")
                
                if graph_analysis['sample_edges']:
                    print(f"  • Sample citation descriptions:")
                    for i, edge in enumerate(graph_analysis['sample_edges'][:2], 1):
                        if 'description' in edge and edge['description']:
                            desc = edge['description'][:150] + "..." if len(edge['description']) > 150 else edge['description']
                            print(f"    {i}. {desc}")

if __name__ == "__main__":
    main() 