#!/usr/bin/env python3
"""
Convert JSON resource files to CSV format for evaluation scripts.

This script converts the structured JSON files from resources/ into flattened CSV files
that can be used by the evaluation scripts (run_evaluation.py, manual_evaluation.py, etc.)

Usage:
    python scripts/convert_json_to_csv.py [domain_name]
    
    domain_name: Specific domain to convert (optional, converts all if not specified)
    
Example:
    python scripts/convert_json_to_csv.py art
    python scripts/convert_json_to_csv.py  # Convert all domains
"""

import json
import pandas as pd
import sys
from pathlib import Path
from typing import Dict, Any, List


def load_json_data(json_path: str) -> Dict[str, Any]:
    """Load data from JSON file."""
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"JSON file not found: {json_path}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format in {json_path}: {e}")


def convert_json_to_csv_data(json_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Convert JSON data structure to CSV-compatible format."""
    csv_rows = []
    
    for paper_id, paper_data in json_data.items():
        row = {
            'id': paper_id,
            'title': paper_data.get('title', ''),
            'content': paper_data.get('content', ''),
            'year': paper_data.get('pub_year', 0),
            'cited_by_count': paper_data.get('cited_by_count', 0),
            'keywords': '|'.join(paper_data.get('keywords', [])),
            'children': '|'.join(paper_data.get('children', []))
        }
        csv_rows.append(row)
    
    return csv_rows


def convert_domain_to_csv(domain: str) -> bool:
    """Convert a specific domain's JSON to CSV format."""
    try:
        # Define paths
        json_path = f"resources/{domain}/{domain}_docs_info.json"
        csv_path = f"data/processed/{domain}_processed.csv"
        
        # Create output directory if it doesn't exist
        Path("data/processed").mkdir(parents=True, exist_ok=True)
        
        print(f"🔄 Converting {domain}...")
        
        # Load JSON data
        json_data = load_json_data(json_path)
        print(f"   📖 Loaded {len(json_data)} papers from {json_path}")
        
        # Convert to CSV format
        csv_data = convert_json_to_csv_data(json_data)
        
        # Create DataFrame and save to CSV
        df = pd.DataFrame(csv_data)
        df.to_csv(csv_path, index=False, encoding='utf-8')
        
        print(f"   ✅ Created {csv_path} with {len(df)} rows")
        return True
        
    except Exception as e:
        print(f"   ❌ Error converting {domain}: {e}")
        return False


def discover_available_domains() -> List[str]:
    """Discover all available domains in the resources directory."""
    resources_path = Path("resources")
    if not resources_path.exists():
        return []
    
    domains = []
    for item in resources_path.iterdir():
        if item.is_dir() and not item.name.startswith('.'):
            # Check if the expected JSON file exists
            json_file = item / f"{item.name}_docs_info.json"
            if json_file.exists():
                domains.append(item.name)
    
    return sorted(domains)


def main():
    """Main conversion function."""
    print("📊 JSON to CSV Converter for Timeline Analysis")
    print("=" * 50)
    
    # Get target domain from command line args
    if len(sys.argv) > 1:
        target_domain = sys.argv[1]
        domains = [target_domain]
        print(f"🎯 Converting specific domain: {target_domain}")
    else:
        # Discover all available domains
        domains = discover_available_domains()
        print(f"🔍 Found {len(domains)} domains: {', '.join(domains)}")
    
    if not domains:
        print("❌ No domains found or specified domain doesn't exist")
        return False
    
    # Convert each domain
    success_count = 0
    for domain in domains:
        if convert_domain_to_csv(domain):
            success_count += 1
    
    print("\n" + "=" * 50)
    print(f"✅ Successfully converted {success_count}/{len(domains)} domains")
    
    if success_count < len(domains):
        print("⚠️  Some conversions failed. Check error messages above.")
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 