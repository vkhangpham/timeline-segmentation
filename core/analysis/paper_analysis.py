"""
Paper Selection and LLM-Based Labeling for Timeline Analysis

This module contains shared functionality for:
- Representative paper selection using network centrality
- LLM-based period label and description generation
- Context loading for enhanced LLM prompts

Follows functional programming principles with pure functions and immutable data structures.
"""

from typing import Dict, List, Tuple, Any
from collections import Counter
import json
import networkx as nx
import re
from pydantic import BaseModel, Field
from ..utils.general import query_llm, query_llm_structured
from ..utils.logging import get_logger


class PeriodLabelResponse(BaseModel):
    """Pydantic model for structured period label and description response."""
    label: str = Field(
        ..., 
        description="Concise period label capturing the dominant research paradigm (2-4 words maximum)",
        min_length=5,
        max_length=100
    )
    description: str = Field(
        ..., 
        description="Detailed technical explanation with specific papers, techniques, and methodological advances (2-3 sentences)",
        min_length=50,
        max_length=100000
    )


class MergedSegmentResponse(BaseModel):
    """Pydantic model for structured merged segment label and description response."""
    label: str = Field(
        ..., 
        description="Unified technical theme name representing the merged period",
        min_length=5,
        max_length=100
    )
    description: str = Field(
        ..., 
        description="Explanation of the continuous development and unified research theme (2-3 sentences)",
        min_length=50,
        max_length=1000
    )


def select_representative_papers(period_papers: List[Dict], 
                                subnetwork: nx.DiGraph, themes: List[str]) -> List[Dict[str, Any]]:
    """Select representative papers using network centrality"""
    if not period_papers:
        return []
    
    # Calculate centrality scores
    try:
        pagerank = nx.pagerank(subnetwork)
        betweenness = nx.betweenness_centrality(subnetwork)
        in_degree_centrality = nx.in_degree_centrality(subnetwork)
    except:
        pagerank = {}
        betweenness = {}
        in_degree_centrality = {}
    
    # Score papers based on multiple factors
    scored_papers = []
    for paper in period_papers:
        paper_id = paper['id']
        paper_data = paper['data']
        
        # Network centrality scores
        pr_score = pagerank.get(paper_id, 0.0)
        bc_score = betweenness.get(paper_id, 0.0)
        in_deg_score = in_degree_centrality.get(paper_id, 0.0)
        
        # Citation impact
        citation_score = min(1.0, paper_data.get('cited_by_count', 0) / 1000.0)
        
        # Breakthrough bonus
        breakthrough_bonus = 0.3 if paper['is_breakthrough'] else 0.0
        
        # Combined score
        total_score = (pr_score * 0.3 + bc_score * 0.2 + in_deg_score * 0.2 + 
                      citation_score * 0.2 + breakthrough_bonus * 0.1)
        
        # Get abstract/description (similar to load_period_context)
        description = paper_data.get('description', '') or paper_data.get('content', '')
        
        scored_papers.append({
            'id': paper_id,
            'title': paper_data.get('title', ''),
            'year': paper_data.get('pub_year', 0),
            'citation_count': paper_data.get('cited_by_count', 0),
            'score': total_score,
            'is_breakthrough': paper['is_breakthrough'],
            'abstract': description,
            'keywords': paper_data.get('keywords', [])
        })
    
    # Remove duplicates based on title, keep the highest scoring one
    seen_titles = set()
    unique_papers = []
    for paper in scored_papers:
        if paper['title'] not in seen_titles:
            seen_titles.add(paper['title'])
            unique_papers.append(paper)
    
    # Sort by score and select top papers
    unique_papers.sort(key=lambda x: x['score'], reverse=True)
    
    # Select top 10 papers, ensuring diversity'
    # Disabling this for now to get more papers
    selected_papers = []
    # for paper in unique_papers[:15]:  # Consider top 12 candidates
    #     # Avoid selecting multiple papers from same year if possible
    #     years_selected = [p['year'] for p in selected_papers]
    #     if len(selected_papers) < 8 or paper['year'] not in years_selected:
    #         selected_papers.append(paper)
    
    selected_papers = unique_papers[:8]
    
    return selected_papers


def load_period_context(papers: List[Dict], start_year: int, end_year: int) -> Dict[str, Any]:
    """
    FUNDAMENTAL SOLUTION: Load essential context data - keywords and paper descriptions only.
    
    Args:
        papers: Papers in this period (can be period_papers or representative_papers format)
        start_year: Period start year
        end_year: Period end year
        
    Returns:
        Simple context with keywords and descriptions
    """
    # Extract keywords from all papers in period
    all_keywords = []
    paper_descriptions = []
    
    for paper in papers:
        # Handle both period_papers format and representative_papers format
        if 'data' in paper:  # period_papers format
            paper_data = paper['data']
            title = paper_data.get('title', '')
            year = paper_data.get('pub_year', 0)
            keywords = paper_data.get('keywords', [])
            description = paper_data.get('description', '') or paper_data.get('content', '')
            citation_count = paper_data.get('cited_by_count', 0)
        else:  # representative_papers format
            title = paper.get('title', '')
            year = paper.get('year', 0)
            keywords = paper.get('keywords', [])
            description = paper.get('abstract', '')
            citation_count = paper.get('citation_count', 0)
        
        # Get keywords
        if isinstance(keywords, list):
            all_keywords.extend(keywords)
        
        # Remove the part from the description that starts with "Topic: "
        if description and "Topic: " in description:
            description = description.split("Topic: ")[0].strip()
        
        # Get paper description
        if description and len(description.strip()) > 30:
            paper_descriptions.append({
                'title': title,
                'year': year,
                'description': description,  # Limit to 300 chars
                'citation_count': citation_count
            })
    
    # Get top keywords by frequency (using existing logic since papers format is different here)
    keyword_freq = Counter(all_keywords)
    top_keywords = [kw for kw, count in keyword_freq.most_common(20)]
    
    # Sort papers by citation count for representative selection
    paper_descriptions.sort(key=lambda x: x['citation_count'], reverse=True)
    
    return {
        'keywords': top_keywords,
        'paper_descriptions': paper_descriptions,  # Use all representative papers
        'keyword_frequencies': dict(keyword_freq.most_common(15))
    }


def generate_period_label_and_description(themes: List[str],
                                         representative_papers: List[Dict], 
                                         start_year: int, end_year: int,
                                         previous_periods: List[Tuple[str, str]] = None,
                                         domain_name: str = "") -> Tuple[str, str]:
    """
    Generate period label and description using LLM with rich context.
    
    FAIL-FAST IMPLEMENTATION: Any LLM query failure immediately raises exception.
    No fallbacks, no error masking - strict adherence to project guidelines Rule 6.
    
    Enhanced approach leveraging multiple data sources with structured outputs:
    - Keywords from representative papers
    - Paper descriptions (d1 node descriptions) 
    - Previous periods for context and evolution
    - Domain awareness to avoid redundant naming
    - Pydantic structured outputs for reliability
    - Reasoning model (deepseek-r1:8b-0528-qwen3-q4_K_M) for enhanced analysis
    
    Args:
        themes: List of dominant themes
        representative_papers: Top papers for the period
        start_year: Period start year
        end_year: Period end year  
        previous_periods: List of (label, description) from previous periods
        domain_name: Domain name to avoid redundant labeling
        
    Returns:
        Tuple of (label, description)
        
    Raises:
        Exception: If LLM query fails or structured output parsing fails
    """
    # FUNDAMENTAL SOLUTION: Use representative_papers for LLM context to ensure consistency
    # This prevents LLM from referencing papers not shown in the representative papers list
    papers_to_use = representative_papers  # Always use representative papers for consistency
    context = load_period_context(papers_to_use, start_year, end_year)
    domain_name = domain_name.replace("_", " ").title()
    
    # Build previous periods context string
    previous_context = ""
    if previous_periods:
        previous_context = "\n\nPREVIOUS PERIODS FOR CONTEXT:\n"
        for i, (start_year, end_year, prev_label, prev_desc) in enumerate(previous_periods, 1):
            previous_context += f"Period {start_year}-{end_year}: {prev_label}\n\n"
    
    # ENHANCED GEMMA 3 PROMPT: Following official prompt engineering guidelines
    prompt = f"""You are an expert scientific historian specializing in research paradigm analysis. Your task is to conduct a comprehensive analysis of the evolution of **{domain_name}** research during the **{start_year}-{end_year}** period.

---

### **CORE MISSION:**
Identify and characterize the fundamental research paradigm that unified the scientific community's approach during this specific time. A **paradigm** represents the shared philosophical framework, methodological principles, and underlying assumptions that guided researchers' problem-solving strategies and research directions.

---

### **ANALYTICAL DATA:**
* **Research Period:** {start_year}-{end_year}
* **Scientific Domain:** {domain_name}
* **Analysis Context:** Historical paradigm identification based on representative scholarly works

#### **Representative Research Papers for Analysis:**
{chr(10).join([f"• **{paper['title']}** ({paper.get('year', 'Unknown')}): {paper.get('description') or paper.get('abstract', '')}" for paper in context.get('paper_descriptions', [])])}

#### **Previous Periods for Context:**
{previous_context}

---

### **SYSTEMATIC ANALYSIS METHODOLOGY:**

1.  **Cross-Paper Pattern Recognition:**
    * Examine each representative paper to identify recurring methodological themes.
    * Look for shared theoretical frameworks, experimental approaches, and conceptual foundations.
    * Identify common assumptions about how problems should be approached in **{domain_name}**.

2.  **Paradigmatic Synthesis:**
    * Determine the overarching philosophical approach that unifies these research efforts.
    * Identify what fundamental principles or methodologies distinguish this period from other eras.
    * Consider how the research community's collective thinking shaped the field's direction.

3.  **Historical Contextualization:**
    * Analyze how this paradigm emerged from or built upon previous research traditions.
    * Assess the paradigm's influence on subsequent research developments.
    * Evaluate the paradigm's lasting impact on **{domain_name}** methodology.

4.  **Paradigm Characterization:**
    * Synthesize a concise, descriptive label (3-7 words maximum) that captures the essence of the paradigmatic approach.
    * Ensure the label reflects the consensus methodology visible across the representative papers and uses **{domain_name}**-specific terminology.
    * Develop a comprehensive explanation (2-3 sentences maximum) that demonstrates deep understanding of the paradigm's foundations.

---

### **EXPECTED DELIVERABLE CHARACTERISTICS:**

* **Paradigm Label:** A precise, technically accurate descriptor (3-7 words maximum) that captures the fundamental methodological or philosophical approach shared across papers, reflects **{domain_name}**-specific terminology, and demonstrates clear consensus among the representative research works.

* **Paradigm Description:** A detailed technical explanation (2-3 sentences maximum) of the fundamental research approach and its defining characteristics, with specific references to representative papers as supporting evidence. It should clearly articulate how this paradigm influenced research methodology and field development, showing technical depth and comprehensive understanding of its foundations.

---

### **RESPONSE FORMATTING REQUIREMENTS:**
Generate your analysis **only** in the following structured JSON format. **Do not include any additional text, explanations, or conversational fillers before or after the JSON.**

```json
{{
  "label": "Paradigm Name",
  "description": "Detailed technical explanation with paper references"
}}"""

    response = query_llm_structured(prompt, PeriodLabelResponse, model="qwen3:8b")
    
    # Extract and validate response
    label = response.label
    description = response.description
    
    if not label or not description:
        raise ValueError(f"Missing label or description in response: {response}")
    
    return label, description


def generate_merged_segment_label_and_description(
    segment1_label: str,
    segment1_description: str, 
    segment1_papers: List[Dict[str, Any]],
    segment2_label: str,
    segment2_description: str,
    segment2_papers: List[Dict[str, Any]],
    merged_period: Tuple[int, int],
    domain_name: str,
    verbose: bool = False
) -> Tuple[str, str]:
    """
    Generate LLM-based label and description for merged segments.
    
    FAIL-FAST IMPLEMENTATION: Any LLM query failure immediately raises exception.
    No fallbacks, no error masking - strict adherence to project guidelines Rule 6.
    
    Enhanced version using structured outputs and reasoning model for
    better analysis of segment continuity and unified themes.
    
    Args:
        segment1_label: Label of first segment
        segment1_description: Description of first segment
        segment1_papers: Representative papers from first segment
        segment2_label: Label of second segment  
        segment2_description: Description of second segment
        segment2_papers: Representative papers from second segment
        merged_period: Combined time period
        domain_name: Domain name for context
        verbose: Enable verbose logging
        
    Returns:
        Tuple of (merged_label, merged_description)
        
    Raises:
        Exception: If LLM query fails or structured output parsing fails
    """
    logger = get_logger(__name__, verbose)
    # Prepare paper information from both segments
    all_papers = list(segment1_papers) + list(segment2_papers)
    
    # Deduplicate by title
    seen_titles = set()
    unique_papers = []
    for paper in all_papers:
        title = paper.get('title', '')
        if title and title not in seen_titles:
            seen_titles.add(title)
            unique_papers.append(paper)
    
    # Sort by score/citation count and take top papers
    unique_papers.sort(key=lambda p: p.get('score', p.get('citation_count', 0)), reverse=True)
    top_papers = unique_papers[:10]
    
    # Build paper context
    paper_context = "REPRESENTATIVE PAPERS FROM MERGED PERIOD:\n"
    for paper in top_papers:
        paper_context += f"- {paper.get('title', '')} ({paper.get('year', 'N/A')})\n"
    
    prompt = f"""<task_description>
You are a research historian. Two adjacent research periods have been merged because they represent a continuous intellectual development. Your task is to analyze the data from both periods and create a single, unified label and description that captures this continuous evolution. The unified theme may be complex, involving multiple related threads.
</task_description>

<merged_period_data>
**Merged Period:** {merged_period[0]}-{merged_period[1]}
**Domain:** {domain_name}

**Original Segment 1:** "{segment1_label}"
- {segment1_description}

**Original Segment 2:** "{segment2_label}"
- {segment2_description}

**Representative Papers from Combined Period:**
{paper_context}
</merged_period_data>

<instructions>
1.  **Synthesize the Continuous Thread:** Identify the overarching methodological and conceptual themes that connect both original segments, justifying their merger. This may involve recognizing a single evolving paradigm or multiple parallel streams of work.
2.  **Craft a Unified Label:** Create a specific, descriptive label (3-7 words) for the unified research paradigm. The label should avoid generic terms and capture the essence of the continuous development.
3.  **Write a Unified Description:** Compose a 3-4 sentence description that:
    -   Clearly defines the unified theme(s).
    -   Uses paper titles from *both* original segments as evidence to illustrate the continuous evolution.
    -   Explains the methodological progression across the entire merged period.
4.  **Format the Output:** Respond with a single JSON object containing two keys: "label" and "description".
</instructions>
"""

    # FAIL-FAST: No try-catch blocks, let any error immediately terminate execution
    response = query_llm_structured(prompt, MergedSegmentResponse, model="deepseek-r1:8b-0528-qwen3-q4_K_M")
    
    # Extract validated response
    label = response.label
    description = response.description
    
    logger.info(f"LLM merge labeling with deepseek-r1:8b-0528-qwen3-q4_K_M completed for {merged_period[0]}-{merged_period[1]}")
    return label, description


 