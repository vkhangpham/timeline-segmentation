#!/usr/bin/env python3
"""
Markdown Header Syntax Fixer

Fixes invalid markdown header syntax in development journal files.
Converts "## **Title**" to proper "## Title" format for correct pandoc processing.

Usage:
    python scripts/fix_markdown_headers.py

This script:
1. Finds all dev_journal_phase*.md files
2. Fixes invalid header syntax (## **Title** -> ## Title)
3. Preserves all other formatting and content
4. Creates backups before modifying files
"""

import re
import glob
import shutil
from pathlib import Path

def fix_markdown_headers_in_file(file_path):
    """Fix markdown header syntax in a single file."""
    print(f"🔄 Processing {file_path}...")
    
    # Create backup
    backup_path = f"{file_path}.backup"
    shutil.copy2(file_path, backup_path)
    print(f"📄 Backup created: {backup_path}")
    
    # Read file content
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Track changes
    original_content = content
    
    # Fix header patterns
    # Pattern 1: ## **Title** -> ## Title
    content = re.sub(r'^(#{1,6})\s+\*\*(.*?)\*\*\s*$', r'\1 \2', content, flags=re.MULTILINE)
    
    # Pattern 2: ## **Title with ID: Something** -> ## Title with ID: Something  
    # This handles cases where the title has colons and other formatting
    content = re.sub(r'^(#{1,6})\s+\*\*(.*?)\*\*\s*$', r'\1 \2', content, flags=re.MULTILINE)
    
    # Check if changes were made
    if content != original_content:
        # Write fixed content back
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        # Count how many headers were fixed
        header_count = len(re.findall(r'^#{1,6}\s+\*\*.*?\*\*\s*$', original_content, flags=re.MULTILINE))
        print(f"✅ Fixed {header_count} headers in {file_path}")
        return True
    else:
        print(f"ℹ️  No header issues found in {file_path}")
        # Remove unnecessary backup
        Path(backup_path).unlink()
        return False

def main():
    """Main function to fix headers in all journal files."""
    print("🔧 MARKDOWN HEADER SYNTAX FIXER")
    print("Fixing invalid header syntax in development journal files...")
    print("=" * 60)
    
    # Find all journal files
    journal_files = glob.glob("dev_journal_phase*.md")
    journal_files.sort()
    
    if not journal_files:
        print("⚠️  No journal files found matching pattern 'dev_journal_phase*.md'")
        return False
    
    print(f"📝 Found {len(journal_files)} journal files:")
    for file in journal_files:
        print(f"   - {file}")
    print()
    
    # Process each file
    fixed_files = 0
    total_files = len(journal_files)
    
    for journal_file in journal_files:
        if fix_markdown_headers_in_file(journal_file):
            fixed_files += 1
        print()
    
    # Summary
    print("📊 SUMMARY")
    print("=" * 20)
    print(f"📁 Files processed: {total_files}")
    print(f"🔧 Files fixed: {fixed_files}")
    print(f"✅ Files unchanged: {total_files - fixed_files}")
    
    if fixed_files > 0:
        print(f"\n✅ Successfully fixed header syntax in {fixed_files} files!")
        print("💡 Backup files created with .backup extension")
        print("🔄 Re-run HTML conversion to see the improvements")
        return True
    else:
        print("\nℹ️  All files already have correct header syntax")
        return True

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1) 