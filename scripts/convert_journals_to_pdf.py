#!/usr/bin/env python3
"""
Development Journal Markdown to PDF Converter

Converts all development journal markdown files to PDF format using pandoc
for enhanced display in the web application.

Usage:
    python scripts/convert_journals_to_pdf.py

This script:
1. Finds all dev_journal_phase*.md files
2. Converts each to HTML first (better Unicode support)
3. Places HTML files in web-app/public/journals/ directory
4. Provides status reporting and error handling
"""

import subprocess
import os
import glob
from pathlib import Path

def ensure_pandoc_available():
    """Check if pandoc is available on the system."""
    try:
        result = subprocess.run(['pandoc', '--version'], 
                              capture_output=True, text=True, check=True)
        print("✅ pandoc available:", result.stdout.split('\n')[0])
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ pandoc not found. Please install pandoc:")
        print("   brew install pandoc")
        return False

def create_output_directory():
    """Create the output directory for converted files."""
    output_dir = Path("web-app/public/journals")
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"📁 Output directory: {output_dir.absolute()}")
    return output_dir

def find_journal_files():
    """Find all development journal markdown files."""
    pattern = "docs/dev_journal_phase*.md"
    files = glob.glob(pattern)
    files.sort()  # Ensure consistent ordering
    print(f"📝 Found {len(files)} journal files:")
    for file in files:
        print(f"   - {file}")
    return files

def convert_markdown_to_html(input_file, output_dir):
    """Convert a single markdown file to HTML using pandoc."""
    input_path = Path(input_file)
    output_filename = input_path.stem + ".html"
    output_path = output_dir / output_filename
    
    # Pandoc command for high-quality HTML with professional styling
    cmd = [
        'pandoc',
        str(input_path),
        '-o', str(output_path),
        '--from', 'markdown',
        '--to', 'html5',
        '--standalone',
        '--toc',
        '--toc-depth=3',
        '--number-sections',
        '--highlight-style=tango',
        '--css', '/journals/journal-style.css',
        '--metadata', f'title=Development Journal - {input_path.stem.replace("dev_journal_", "").replace("_", " ").title()}',
        '--metadata', 'author=Timeline Analysis Project'
    ]
    
    try:
        print(f"🔄 Converting {input_file} to HTML...")
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(f"✅ Successfully created {output_path}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error converting {input_file}:")
        print(f"   Command: {' '.join(cmd)}")
        print(f"   Error: {e.stderr}")
        return False

def create_css_file(output_dir):
    """Create a CSS file for professional journal styling."""
    css_content = """
/* Professional Academic Journal Styling */
body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
    line-height: 1.6;
    color: #333;
    max-width: 1200px;
    margin: 0 auto;
    padding: 40px 20px;
    background-color: #fafafa;
}

h1, h2, h3, h4, h5, h6 {
    color: #2c3e50;
    margin-top: 2em;
    margin-bottom: 0.5em;
    font-weight: 600;
}

h1 {
    border-bottom: 3px solid #3498db;
    padding-bottom: 10px;
    font-size: 2.5em;
}

h2 {
    border-bottom: 2px solid #ecf0f1;
    padding-bottom: 8px;
    font-size: 2em;
}

h3 {
    font-size: 1.5em;
    color: #34495e;
}

/* Table of Contents */
#TOC {
    background: white;
    border: 1px solid #ddd;
    border-radius: 8px;
    padding: 20px;
    margin: 20px 0;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}

#TOC ul {
    list-style-type: none;
    padding-left: 0;
}

#TOC li {
    margin: 5px 0;
}

#TOC a {
    color: #3498db;
    text-decoration: none;
    padding: 5px 10px;
    border-radius: 4px;
    transition: background-color 0.2s;
}

#TOC a:hover {
    background-color: #ecf0f1;
}

/* Fix for header parsing issue - remove ## symbols from TOC display */
#TOC a::before {
    content: '';
}

/* Custom styling to hide ## symbols in table of contents */
#TOC {
    /* Use JavaScript or post-processing to clean up TOC entries */
}

/* Code blocks */
pre {
    background: #f8f9fa;
    border: 1px solid #e9ecef;
    border-radius: 6px;
    padding: 15px;
    overflow-x: auto;
    margin: 1em 0;
}

code {
    background: #f1f3f4;
    padding: 2px 6px;
    border-radius: 3px;
    font-family: 'Monaco', 'Consolas', 'Courier New', monospace;
    font-size: 0.9em;
}

/* Status indicators */
.status {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.85em;
    font-weight: 600;
    margin-left: 10px;
}

/* Blockquotes */
blockquote {
    border-left: 4px solid #3498db;
    margin: 1em 0;
    padding: 0.5em 1em;
    background: #f8f9fa;
    font-style: italic;
}

/* Lists */
ul, ol {
    margin: 1em 0;
    padding-left: 2em;
}

li {
    margin: 0.5em 0;
}

/* Tables */
table {
    border-collapse: collapse;
    width: 100%;
    margin: 1em 0;
    background: white;
    box-shadow: 0 1px 3px rgba(0,0,0,0.1);
}

th, td {
    border: 1px solid #ddd;
    padding: 12px;
    text-align: left;
}

th {
    background-color: #f8f9fa;
    font-weight: 600;
}

/* Links */
a {
    color: #3498db;
    text-decoration: none;
}

a:hover {
    text-decoration: underline;
}

/* Emphasis */
strong {
    color: #2c3e50;
    font-weight: 600;
}

em {
    color: #7f8c8d;
}

/* Print styles */
@media print {
    body {
        background: white;
        color: black;
        max-width: none;
        margin: 0;
        padding: 20px;
    }
    
    h1, h2, h3, h4, h5, h6 {
        page-break-after: avoid;
    }
    
    pre, blockquote {
        page-break-inside: avoid;
    }
}

/* Responsive design */
@media (max-width: 768px) {
    body {
        padding: 20px 10px;
    }
    
    h1 {
        font-size: 2em;
    }
    
    h2 {
        font-size: 1.5em;
    }
    
    #TOC {
        margin: 10px 0;
        padding: 15px;
    }
}
"""
    
    css_path = output_dir / "journal-style.css"
    try:
        with open(css_path, 'w', encoding='utf-8') as f:
            f.write(css_content)
        print(f"✅ Created CSS file: {css_path}")
        return True
    except Exception as e:
        print(f"❌ Error creating CSS file: {e}")
        return False

def main():
    """Main conversion process."""
    print("🚀 DEVELOPMENT JOURNAL HTML CONVERTER")
    print("Converting markdown journals to HTML format for web display")
    print("=" * 60)
    
    # Check prerequisites
    if not ensure_pandoc_available():
        return False
    
    # Setup output directory
    output_dir = create_output_directory()
    
    # Create CSS file
    if not create_css_file(output_dir):
        print("⚠️  Failed to create CSS file, continuing without styling...")
    
    # Find journal files
    journal_files = find_journal_files()
    if not journal_files:
        print("⚠️  No journal files found matching pattern 'dev_journal_phase*.md'")
        return False
    
    # Convert each file
    print(f"\n📄 CONVERTING {len(journal_files)} FILES")
    print("=" * 40)
    
    successful_conversions = 0
    failed_conversions = 0
    
    for journal_file in journal_files:
        if convert_markdown_to_html(journal_file, output_dir):
            successful_conversions += 1
        else:
            failed_conversions += 1
    
    # Summary
    print(f"\n📊 CONVERSION SUMMARY")
    print("=" * 20)
    print(f"✅ Successful: {successful_conversions}")
    print(f"❌ Failed: {failed_conversions}")
    print(f"📁 HTML files saved to: {output_dir.absolute()}")
    
    if failed_conversions > 0:
        print("\n⚠️  Some conversions failed. Check error messages above.")
        return False
    else:
        print("\n🎉 All journal files successfully converted to HTML!")
        print("💡 HTML files can be viewed directly in browser or printed to PDF")
        return True

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1) 