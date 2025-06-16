#!/usr/bin/env python3
"""
Detailed analysis of citation graph with semantic descriptions
"""

import xml.etree.ElementTree as ET
import os
from collections import defaultdict

def analyze_citation_descriptions(domain):
    """Analyze citation graph descriptions in detail"""
    file_path = f'resources/{domain}/{domain}_entity_relation_graph.graphml.xml'
    
    if not os.path.exists(file_path):
        return None
    
    try:
        tree = ET.parse(file_path)
        root = tree.getroot()
        
        namespace = {'graphml': 'http://graphml.graphdrawing.org/xmlns'}
        
        # Find key definitions
        keys = root.findall('.//graphml:key', namespace)
        key_map = {}
        for key in keys:
            key_map[key.get('id')] = key.get('attr.name', 'unknown')
        
        # Analyze edges with descriptions
        edges = root.findall('.//graphml:edge', namespace)
        
        descriptions = []
        description_lengths = []
        sample_descriptions = []
        sample_count = 0
        
        for edge in edges:
            edge_descriptions = []
            
            for data in edge.findall('graphml:data', namespace):
                key_id = data.get('key')
                key_name = key_map.get(key_id, f'key_{key_id}')
                
                # Look for description fields (d4 is edge description)
                if data.text and len(data.text.strip()) > 0 and key_id == 'd4':
                    desc_text = data.text.strip()
                    descriptions.append(desc_text)
                    description_lengths.append(len(desc_text))
                    edge_descriptions.append(desc_text)
            
            # Collect samples
            if edge_descriptions and sample_count < 10:
                sample_descriptions.extend(edge_descriptions)
                sample_count += 1
        
        return {
            'total_edges': len(edges),
            'total_descriptions': len(descriptions),
            'avg_description_length': sum(description_lengths) / len(description_lengths) if description_lengths else 0,
            'min_description_length': min(description_lengths) if description_lengths else 0,
            'max_description_length': max(description_lengths) if description_lengths else 0,
            'sample_descriptions': sample_descriptions[:10],
            'key_map': key_map
        }
    
    except Exception as e:
        return {'error': str(e)}

def main():
    domains = [
        'natural_language_processing',
        'deep_learning', 
        'computer_vision',
        'machine_translation',
        'machine_learning'
    ]
    
    print("=" * 80)
    print("DETAILED CITATION GRAPH ANALYSIS - FIXED VERSION")
    print("=" * 80)
    
    for domain in domains:
        print(f"\n{'='*60}")
        print(f"DOMAIN: {domain.upper()}")
        print(f"{'='*60}")
        
        analysis = analyze_citation_descriptions(domain)
        
        if analysis and 'error' not in analysis:
            print(f"\n📊 CITATION STATISTICS:")
            print(f"  • Total edges: {analysis['total_edges']:,}")
            print(f"  • Edges with descriptions: {analysis['total_descriptions']:,}")
            if analysis['total_edges'] > 0:
                coverage = (analysis['total_descriptions']/analysis['total_edges']*100)
                print(f"  • Coverage: {coverage:.1f}%")
            
            if analysis['total_descriptions'] > 0:
                print(f"\n📝 DESCRIPTION ANALYSIS:")
                print(f"  • Average length: {analysis['avg_description_length']:.0f} chars")
                print(f"  • Length range: {analysis['min_description_length']}-{analysis['max_description_length']} chars")
            
            print(f"\n🔑 GRAPH SCHEMA:")
            for key_id, key_name in analysis['key_map'].items():
                print(f"  • {key_id}: {key_name}")
            
            if analysis['sample_descriptions']:
                print(f"\n💬 SAMPLE CITATION DESCRIPTIONS:")
                for i, desc in enumerate(analysis['sample_descriptions'], 1):
                    # Clean and truncate description
                    clean_desc = desc.replace('\n', ' ').replace('  ', ' ').strip()
                    display_desc = clean_desc[:200] + "..." if len(clean_desc) > 200 else clean_desc
                    print(f"  {i}. {display_desc}")
            else:
                print(f"\n💬 No descriptions found in edges")
        
        elif analysis and 'error' in analysis:
            print(f"  ❌ Error: {analysis['error']}")
        else:
            print(f"  ❌ File not found")

if __name__ == "__main__":
    main() 