#!/usr/bin/env python3
"""
HTML Table of Contents Cleaner

Post-processes generated HTML files to clean up table of contents entries
that contain unwanted ## markdown syntax artifacts.

Usage:
    python scripts/clean_html_toc.py

This script:
1. Finds all generated HTML files in web-app/public/journals/
2. Cleans up table of contents entries removing ## artifacts
3. Preserves all other HTML content and formatting
"""

import re
import glob
from pathlib import Path

def clean_toc_in_html(html_path):
    """Clean table of contents entries in a single HTML file."""
    print(f"🔄 Cleaning table of contents in {html_path.name}...")
    
    # Read HTML content
    with open(html_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Track changes
    original_content = content
    
    # Fix table of contents entries that contain ## symbols
    # Pattern: <span class="toc-section-number">1.2</span> ## TITLE becomes
    # <span class="toc-section-number">1.2</span> TITLE
    
    # This regex finds TOC entries and removes the ## from the display text
    # Pattern: class="toc-section-number">1.2</span> ## RESEARCH-018: Title
    toc_pattern = r'(class="toc-section-number">[^<]+</span>\s*)##\s*([^<]+)(?=</a>)'
    content = re.sub(toc_pattern, r'\1\2', content)
    
    # Alternative pattern for entries without section numbers
    simple_toc_pattern = r'(>\s*)##\s*([A-Z]+-[0-9]+:.*?)(\s*</a>)'
    content = re.sub(simple_toc_pattern, r'\1\2\3', content)
    
    # Check if changes were made
    if content != original_content:
        # Write cleaned content back
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        # Count how many entries were fixed
        fix_count = len(re.findall(toc_pattern, original_content)) + len(re.findall(simple_toc_pattern, original_content))
        print(f"✅ Cleaned {fix_count} table of contents entries in {html_path.name}")
        return True
    else:
        print(f"ℹ️  No table of contents issues found in {html_path.name}")
        return False

def main():
    """Main function to clean table of contents in all HTML journal files."""
    print("🧹 HTML TABLE OF CONTENTS CLEANER")
    print("Cleaning up table of contents entries in generated HTML files...")
    print("=" * 65)
    
    # Find all HTML journal files
    html_dir = Path("web-app/public/journals")
    if not html_dir.exists():
        print(f"⚠️  HTML directory not found: {html_dir}")
        return False
    
    html_files = list(html_dir.glob("dev_journal_phase*.html"))
    html_files.sort()
    
    if not html_files:
        print(f"⚠️  No HTML journal files found in {html_dir}")
        return False
    
    print(f"📝 Found {len(html_files)} HTML files:")
    for file in html_files:
        print(f"   - {file.name}")
    print()
    
    # Process each file
    cleaned_files = 0
    total_files = len(html_files)
    
    for html_file in html_files:
        if clean_toc_in_html(html_file):
            cleaned_files += 1
        print()
    
    # Summary
    print("📊 CLEANING SUMMARY")
    print("=" * 25)
    print(f"📁 Files processed: {total_files}")
    print(f"🧹 Files cleaned: {cleaned_files}")
    print(f"✅ Files unchanged: {total_files - cleaned_files}")
    
    if cleaned_files > 0:
        print(f"\n✅ Successfully cleaned table of contents in {cleaned_files} files!")
        print("🔄 Refresh your browser to see the improvements")
        return True
    else:
        print("\nℹ️  All files already have clean table of contents")
        return True

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1) 