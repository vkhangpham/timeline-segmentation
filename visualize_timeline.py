#!/usr/bin/env python3
"""
Timeline Analysis Visualization Script

This script generates comprehensive visualizations for timeline analysis results,
including shift signals, period signals, and integrated timeline views.

Usage:
    python visualize_timeline.py --domain deep_learning
    python visualize_timeline.py --domain all
    python visualize_timeline.py --help
"""

import argparse
import sys
from pathlib import Path

# Add core module to path
sys.path.append(str(Path(__file__).parent))

from core.visualization import visualize_domain_timeline, visualize_all_domains
from core.utils import discover_available_domains


def main():
    """Main visualization execution."""
    
    parser = argparse.ArgumentParser(
        description="Timeline Analysis Visualization Generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python visualize_timeline.py --domain deep_learning
  python visualize_timeline.py --domain natural_language_processing
  python visualize_timeline.py --domain all
  python visualize_timeline.py --domain all --signals-dir custom/signals --output-dir custom/viz

This script reads signal data from the results/signals directory and generates:
- Main timeline visualization showing periods and paradigm shifts
- Detailed shift signal analysis plots
- Detailed period characterization analysis plots  
- Interactive HTML dashboard
        """
    )
    
    parser.add_argument(
        '--domain', 
        type=str, 
        required=True,
        help='Domain to visualize (use "all" to visualize all domains with signal data, or specify a domain name)'
    )
    
    parser.add_argument(
        '--signals-dir',
        type=str,
        default='results/signals',
        help='Directory containing signal files (default: results/signals)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='results/visualizations',
        help='Directory to save visualizations (default: results/visualizations)'
    )
    
    parser.add_argument(
        '--list-domains',
        action='store_true',
        help='List available domains with signal data'
    )
    
    args = parser.parse_args()
    
    print("🎨 TIMELINE ANALYSIS VISUALIZATION GENERATOR")
    print("=" * 60)
    
    # Check if signals directory exists
    signals_path = Path(args.signals_dir)
    if not signals_path.exists():
        print(f"❌ Signals directory not found: {args.signals_dir}")
        print("   Run the timeline analysis first to generate signal data.")
        return False
    
    # List available domains if requested
    if args.list_domains:
        domains = get_domains_with_signals(args.signals_dir)
        if domains:
            print(f"📊 Available domains with signal data:")
            for domain in sorted(domains):
                print(f"  • {domain}")
        else:
            print("❌ No domains found with signal data")
        return True
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Generate visualizations
    if args.domain == 'all':
        print("🌍 Generating visualizations for all domains...")
        visualization_files = visualize_all_domains(args.signals_dir, args.output_dir)
        
        if visualization_files:
            print(f"\n✅ Successfully created {len(visualization_files)} visualizations")
            print(f"📁 Visualizations saved in: {args.output_dir}")
            
            # List created files
            print(f"\n📊 Created visualization files:")
            for viz_file in visualization_files:
                print(f"  • {Path(viz_file).name}")
        else:
            print("❌ No visualizations created")
            return False
            
    else:
        # Check if domain has signal data
        domains_with_signals = get_domains_with_signals(args.signals_dir)
        if args.domain not in domains_with_signals:
            print(f"❌ No signal data found for domain: {args.domain}")
            print(f"Available domains: {', '.join(sorted(domains_with_signals))}")
            return False
        
        print(f"🎯 Generating visualization for domain: {args.domain}")
        main_viz_file = visualize_domain_timeline(args.domain, args.signals_dir, args.output_dir)
        
        if main_viz_file:
            print(f"\n✅ Visualization created successfully!")
            print(f"📁 Main visualization: {main_viz_file}")
            print(f"📁 Additional files saved in: {args.output_dir}")
            
            # List all files for this domain
            output_path = Path(args.output_dir)
            domain_files = list(output_path.glob(f"{args.domain}_*"))
            if domain_files:
                print(f"\n📊 All files for {args.domain}:")
                for file in sorted(domain_files):
                    print(f"  • {file.name}")
        else:
            print("❌ Visualization creation failed")
            return False
    
    print(f"\n🎉 Visualization generation complete!")
    print(f"📂 Open the files in {args.output_dir} to view your timeline analysis results.")
    
    return True


def get_domains_with_signals(signals_dir: str) -> set:
    """Get list of domains that have signal data files."""
    signals_path = Path(signals_dir)
    domains = set()
    
    # Find domains with shift signals
    for file in signals_path.glob("*_shift_signals.json"):
        domain = file.stem.replace("_shift_signals", "")
        domains.add(domain)
    
    # Find domains with period signals
    for file in signals_path.glob("*_period_signals.json"):
        domain = file.stem.replace("_period_signals", "")
        domains.add(domain)
    
    return domains


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 