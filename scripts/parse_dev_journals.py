#!/usr/bin/env python3
"""
Development Journal Parser
Converts dev_journal_phase*.md files to structured JSON for frontend consumption
"""

import re
import json
import os
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict

@dataclass
class JournalEntry:
    """Represents a single development journal entry"""
    id: str
    title: str
    status: str
    priority: str
    phase: str
    date_added: str
    date_completed: Optional[str]
    impact: str
    files: List[str]
    problem_description: str
    goal: str
    research_approach: str
    solution_implemented: str
    impact_on_core_plan: str
    reflection: str

@dataclass
class PhaseData:
    """Represents data for a complete development phase"""
    phase: str
    overview: str
    entries: List[JournalEntry]
    total_entries: int
    completed_entries: int
    in_progress_entries: int
    critical_entries: int

def parse_journal_entry(content: str, start_idx: int) -> tuple[Optional[JournalEntry], int]:
    """Parse a single journal entry from markdown content"""
    lines = content.split('\n')
    
    # Find the entry header
    entry_start = None
    for i in range(start_idx, len(lines)):
        line = lines[i].strip()
        if re.match(r'^## \*\*[A-Z]+-\d+:', line):
            entry_start = i
            break
    
    if entry_start is None:
        return None, len(lines)
    
    # Find the next entry or end of content
    entry_end = len(lines)
    for i in range(entry_start + 1, len(lines)):
        line = lines[i].strip()
        if re.match(r'^## \*\*[A-Z]+-\d+:', line):
            entry_end = i
            break
        # Also stop at phase boundaries
        if line.startswith('## **Phase') or line.startswith('# **Development Journal'):
            entry_end = i
            break
    
    # Extract entry content
    entry_lines = lines[entry_start:entry_end]
    entry_content = '\n'.join(entry_lines)
    
    # Parse structured data
    entry_data = {
        'id': '',
        'title': '',
        'status': '',
        'priority': '',
        'phase': '',
        'date_added': '',
        'date_completed': None,
        'impact': '',
        'files': [],
        'problem_description': '',
        'goal': '',
        'research_approach': '',
        'solution_implemented': '',
        'impact_on_core_plan': '',
        'reflection': ''
    }
    
    # Extract header information
    header_match = re.search(r'## \*\*([A-Z]+-\d+):', entry_content)
    if header_match:
        entry_data['id'] = header_match.group(1)
    
    # Extract structured metadata
    for line in entry_lines:
        line = line.strip()
        if line.startswith('ID:'):
            entry_data['id'] = line.replace('ID:', '').strip()
        elif line.startswith('Title:'):
            entry_data['title'] = line.replace('Title:', '').strip()
        elif line.startswith('Status:'):
            entry_data['status'] = line.replace('Status:', '').strip()
        elif line.startswith('Priority:'):
            entry_data['priority'] = line.replace('Priority:', '').strip()
        elif line.startswith('Phase:'):
            entry_data['phase'] = line.replace('Phase:', '').strip()
        elif line.startswith('DateAdded:'):
            entry_data['date_added'] = line.replace('DateAdded:', '').strip()
        elif line.startswith('DateCompleted:'):
            completed = line.replace('DateCompleted:', '').strip()
            if completed and completed not in ['[TBD]', 'TBD']:
                entry_data['date_completed'] = completed
        elif line.startswith('Impact:'):
            entry_data['impact'] = line.replace('Impact:', '').strip()
    
    # Extract files
    files_section = re.search(r'Files:\s*\n((?:\s*-\s*.*\n?)*)', entry_content, re.MULTILINE)
    if files_section:
        files_text = files_section.group(1)
        files = []
        for line in files_text.split('\n'):
            line = line.strip()
            if line.startswith('- '):
                files.append(line.replace('- ', '').strip())
        entry_data['files'] = files
    
    # Extract narrative sections
    sections = {
        'problem_description': r'\*\*Problem Description:\*\*(.*?)(?=\*\*Goal:\*\*|\*\*Research & Approach:\*\*|\*\*Solution Implemented & Verified:\*\*|\*\*Impact on Core Plan:\*\*|\*\*Reflection:\*\*|\n---|\Z)',
        'goal': r'\*\*Goal:\*\*(.*?)(?=\*\*Problem Description:\*\*|\*\*Research & Approach:\*\*|\*\*Solution Implemented & Verified:\*\*|\*\*Impact on Core Plan:\*\*|\*\*Reflection:\*\*|\n---|\Z)',
        'research_approach': r'\*\*Research & Approach:\*\*(.*?)(?=\*\*Problem Description:\*\*|\*\*Goal:\*\*|\*\*Solution Implemented & Verified:\*\*|\*\*Impact on Core Plan:\*\*|\*\*Reflection:\*\*|\n---|\Z)',
        'solution_implemented': r'\*\*Solution Implemented & Verified:\*\*(.*?)(?=\*\*Problem Description:\*\*|\*\*Goal:\*\*|\*\*Research & Approach:\*\*|\*\*Impact on Core Plan:\*\*|\*\*Reflection:\*\*|\n---|\Z)',
        'impact_on_core_plan': r'\*\*Impact on Core Plan:\*\*(.*?)(?=\*\*Problem Description:\*\*|\*\*Goal:\*\*|\*\*Research & Approach:\*\*|\*\*Solution Implemented & Verified:\*\*|\*\*Reflection:\*\*|\n---|\Z)',
        'reflection': r'\*\*Reflection:\*\*(.*?)(?=\*\*Problem Description:\*\*|\*\*Goal:\*\*|\*\*Research & Approach:\*\*|\*\*Solution Implemented & Verified:\*\*|\*\*Impact on Core Plan:\*\*|\n---|\Z)'
    }
    
    for key, pattern in sections.items():
        match = re.search(pattern, entry_content, re.DOTALL)
        if match:
            text = match.group(1).strip()
            # Clean up the text - remove extra whitespace and markdown
            text = re.sub(r'\n+', ' ', text)
            text = re.sub(r'\s+', ' ', text)
            entry_data[key] = text
    
    # Create entry object
    try:
        entry = JournalEntry(**entry_data)
        return entry, entry_end
    except Exception as e:
        print(f"Error creating entry {entry_data['id']}: {e}")
        return None, entry_end

def parse_phase_overview(content: str) -> str:
    """Extract the phase overview section"""
    overview_match = re.search(r'## \*\*Phase Overview\*\*(.*?)(?=\n---|\n## )', content, re.DOTALL)
    if overview_match:
        overview = overview_match.group(1).strip()
        # Clean up formatting
        overview = re.sub(r'\n+', ' ', overview)
        overview = re.sub(r'\s+', ' ', overview)
        return overview
    return ''

def parse_journal_file(file_path: Path) -> PhaseData:
    """Parse a complete journal file"""
    content = file_path.read_text(encoding='utf-8')
    
    # Extract phase number from filename
    phase_match = re.search(r'phase(\d+)', file_path.name)
    phase_num = phase_match.group(1) if phase_match else '0'
    
    # Extract phase overview
    overview = parse_phase_overview(content)
    
    # Parse all entries
    entries = []
    start_idx = 0
    
    while start_idx < len(content.split('\n')):
        entry, next_idx = parse_journal_entry(content, start_idx)
        if entry and entry.id:  # Only add entries with valid IDs
            entries.append(entry)
        
        # Continue from where the last entry ended
        if next_idx > start_idx:
            start_idx = next_idx
        else:
            # Safety increment to prevent infinite loops
            start_idx += 1
            
        # Safety break
        if start_idx >= len(content.split('\n')):
            break
    
    # Consolidate duplicate entry IDs by keeping the most complete version
    consolidated_entries = {}
    for entry in entries:
        if entry.id in consolidated_entries:
            # Compare entries and keep the most complete one
            existing = consolidated_entries[entry.id]
            
            # Prioritize completed entries over in-progress
            if 'Successfully Implemented' in entry.status and 'In Progress' in existing.status:
                # Use completion info from new entry, but keep detailed content from existing
                consolidated_entry = JournalEntry(
                    id=entry.id,
                    title=existing.title or entry.title,
                    status=entry.status,  # Use completed status
                    priority=existing.priority or entry.priority,
                    phase=existing.phase or entry.phase,
                    date_added=existing.date_added or entry.date_added,
                    date_completed=entry.date_completed,  # Use completion date
                    impact=entry.impact or existing.impact,  # Prefer updated impact
                    files=existing.files or entry.files,
                    problem_description=existing.problem_description or entry.problem_description,
                    goal=existing.goal or entry.goal,
                    research_approach=existing.research_approach or entry.research_approach,
                    solution_implemented=existing.solution_implemented or entry.solution_implemented,
                    impact_on_core_plan=existing.impact_on_core_plan or entry.impact_on_core_plan,
                    reflection=existing.reflection or entry.reflection
                )
                consolidated_entries[entry.id] = consolidated_entry
            elif 'In Progress' in entry.status and 'Successfully Implemented' in existing.status:
                # Keep existing completed entry, don't replace with in-progress
                continue
            else:
                # For other cases, keep the entry with more detailed content
                if (len(entry.problem_description) + len(entry.goal) + len(entry.solution_implemented) >
                    len(existing.problem_description) + len(existing.goal) + len(existing.solution_implemented)):
                    consolidated_entries[entry.id] = entry
        else:
            consolidated_entries[entry.id] = entry
    
    # Convert back to list
    final_entries = list(consolidated_entries.values())
    
    # Calculate summary statistics
    completed_count = len([e for e in final_entries if 'Successfully Implemented' in e.status])
    in_progress_count = len([e for e in final_entries if 'In Progress' in e.status or 'Needs Research' in e.status])
    critical_count = len([e for e in final_entries if e.priority == 'Critical'])
    
    return PhaseData(
        phase=phase_num,
        overview=overview,
        entries=final_entries,
        total_entries=len(final_entries),
        completed_entries=completed_count,
        in_progress_entries=in_progress_count,
        critical_entries=critical_count
    )

def main():
    """Main function to process all journal files"""
    # Define paths
    project_root = Path(__file__).parent.parent
    journal_files = list(project_root.glob('dev_journal_phase*.md'))
    output_dir = project_root / 'web-app' / 'public' / 'parsed_journals'
    
    # Create output directory
    output_dir.mkdir(exist_ok=True)
    
    print(f"Found {len(journal_files)} journal files to process")
    
    all_phases = {}
    
    for journal_file in sorted(journal_files):
        print(f"Processing {journal_file.name}...")
        
        try:
            phase_data = parse_journal_file(journal_file)
            print(f"  Phase {phase_data.phase}: {phase_data.total_entries} entries "
                  f"({phase_data.completed_entries} completed, "
                  f"{phase_data.in_progress_entries} in progress, "
                  f"{phase_data.critical_entries} critical)")
            
            # Save individual phase file
            phase_output_file = output_dir / f'phase_{phase_data.phase}.json'
            with open(phase_output_file, 'w', encoding='utf-8') as f:
                json.dump(asdict(phase_data), f, indent=2, ensure_ascii=False)
            
            all_phases[phase_data.phase] = asdict(phase_data)
            
        except Exception as e:
            print(f"Error processing {journal_file.name}: {e}")
    
    # Save combined file
    combined_output_file = output_dir / 'all_phases.json'
    with open(combined_output_file, 'w', encoding='utf-8') as f:
        json.dump(all_phases, f, indent=2, ensure_ascii=False)
    
    print(f"\nSuccessfully processed {len(all_phases)} phases")
    print(f"Output saved to: {output_dir}")
    print(f"Individual files: phase_1.json, phase_2.json, ...")
    print(f"Combined file: all_phases.json")

if __name__ == '__main__':
    main() 