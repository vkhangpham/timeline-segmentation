"""Paper selection and LLM-based labeling for timeline analysis.

This module provides representative paper selection using network centrality
and LLM-based period label and description generation.
"""

from typing import Dict, List, Tuple, Any
from collections import Counter
import networkx as nx
from pydantic import BaseModel, Field
from ..utils.general import query_llm_structured
from ..utils.logging import get_logger


class PeriodLabelResponse(BaseModel):
    """Pydantic model for structured period label and description response."""

    label: str = Field(
        ...,
        description="Concise period label capturing the dominant research paradigm (2-4 words maximum)",
        min_length=5,
        max_length=100,
    )
    description: str = Field(
        ...,
        description="Detailed technical explanation with specific papers, techniques, and methodological advances (2-3 sentences)",
        min_length=50,
        max_length=100000,
    )


class MergedSegmentResponse(BaseModel):
    """Pydantic model for structured merged segment label and description response."""

    label: str = Field(
        ...,
        description="Unified technical theme name representing the merged period",
        min_length=5,
        max_length=100,
    )
    description: str = Field(
        ...,
        description="Explanation of the continuous development and unified research theme (2-3 sentences)",
        min_length=50,
        max_length=1000,
    )


def select_representative_papers(
    period_papers: List[Dict],
    subnetwork: nx.DiGraph,
    themes: List[str],
    verbose: bool = False,
) -> List[Dict[str, Any]]:
    """Select representative papers using network centrality.

    Args:
        period_papers: List of papers in the period
        subnetwork: Citation network subnetwork for the period
        themes: List of dominant themes
        verbose: Enable verbose logging

    Returns:
        List of representative papers with scoring information
    """
    logger = get_logger(__name__, verbose)

    if not period_papers:
        if verbose:
            logger.info("  No period papers provided for selection")
        return []

    if verbose:
        logger.info(
            f"  Selecting representative papers from {len(period_papers)} candidates"
        )
        logger.info("  Calculating network centrality scores...")

    try:
        pagerank = nx.pagerank(subnetwork)
        betweenness = nx.betweenness_centrality(subnetwork)
        in_degree_centrality = nx.in_degree_centrality(subnetwork)

        if verbose:
            logger.info(f"  PageRank calculated for {len(pagerank)} papers")
            logger.info(
                f"  Betweenness centrality calculated for {len(betweenness)} papers"
            )
            logger.info(
                f"  In-degree centrality calculated for {len(in_degree_centrality)} papers"
            )
    except (ValueError, ZeroDivisionError, nx.NetworkXError) as e:
        raise RuntimeError(
            f"Failed to calculate centrality scores for paper selection: {e}"
        ) from e

    if verbose:
        logger.info("  Scoring papers based on multiple factors...")

    scored_papers = []
    for paper in period_papers:
        paper_id = paper["id"]
        paper_data = paper["data"]

        pr_score = pagerank.get(paper_id, 0.0)
        bc_score = betweenness.get(paper_id, 0.0)
        in_deg_score = in_degree_centrality.get(paper_id, 0.0)

        citation_score = min(1.0, paper_data.get("cited_by_count", 0) / 1000.0)

        breakthrough_bonus = 0.3 if paper.get("is_breakthrough", False) else 0.0

        total_score = (
            pr_score * 0.3
            + bc_score * 0.2
            + in_deg_score * 0.2
            + citation_score * 0.2
            + breakthrough_bonus * 0.1
        )

        description = paper_data.get("description", "") or paper_data.get("content", "")

        scored_papers.append(
            {
                "id": paper_id,
                "title": paper_data.get("title", ""),
                "year": paper_data.get("pub_year", 0),
                "citation_count": paper_data.get("cited_by_count", 0),
                "score": total_score,
                "is_breakthrough": paper.get("is_breakthrough", False),
                "abstract": description,
                "keywords": paper_data.get("keywords", []),
            }
        )

    if verbose:
        logger.info(f"  Scored {len(scored_papers)} papers")
        logger.info("  Removing duplicates based on title...")

    seen_titles = set()
    unique_papers = []
    for paper in scored_papers:
        if paper["title"] not in seen_titles:
            seen_titles.add(paper["title"])
            unique_papers.append(paper)

    if verbose:
        logger.info(f"  After deduplication: {len(unique_papers)} unique papers")
        logger.info("  Sorting papers by score...")

    unique_papers.sort(key=lambda x: x["score"], reverse=True)

    selected_papers = []
    selected_papers = unique_papers[:8]

    if verbose:
        logger.info(f"  Selected {len(selected_papers)} representative papers")
        if selected_papers:
            logger.info("  Top selected papers:")
            for i, paper in enumerate(selected_papers[:3], 1):
                logger.info(
                    f"    {i}. {paper['title']} ({paper['year']}) - Score: {paper['score']:.3f}"
                )

    return selected_papers


def load_period_context(
    papers: List[Dict], start_year: int, end_year: int
) -> Dict[str, Any]:
    """Load essential context data - keywords and paper descriptions.

    Args:
        papers: Papers in this period
        start_year: Period start year
        end_year: Period end year

    Returns:
        Simple context with keywords and descriptions
    """
    all_keywords = []
    paper_descriptions = []

    for paper in papers:
        if "data" in paper:
            paper_data = paper["data"]
            title = paper_data.get("title", "")
            year = paper_data.get("pub_year", 0)
            keywords = paper_data.get("keywords", [])
            description = paper_data.get("description", "") or paper_data.get(
                "content", ""
            )
            citation_count = paper_data.get("cited_by_count", 0)
        else:
            title = paper.get("title", "")
            year = paper.get("year", 0)
            keywords = paper.get("keywords", [])
            description = paper.get("abstract", "")
            citation_count = paper.get("citation_count", 0)

        if isinstance(keywords, list):
            all_keywords.extend(keywords)

        if description and "Topic: " in description:
            description = description.split("Topic: ")[0].strip()

        if description and len(description.strip()) > 30:
            paper_descriptions.append(
                {
                    "title": title,
                    "year": year,
                    "description": description,
                    "citation_count": citation_count,
                }
            )

    keyword_freq = Counter(all_keywords)
    top_keywords = [kw for kw, count in keyword_freq.most_common(20)]

    paper_descriptions.sort(key=lambda x: x["citation_count"], reverse=True)

    return {
        "keywords": top_keywords,
        "paper_descriptions": paper_descriptions,
        "keyword_frequencies": dict(keyword_freq.most_common(15)),
    }


def generate_period_label_and_description(
    themes: List[str],
    representative_papers: List[Dict],
    start_year: int,
    end_year: int,
    previous_periods: List[Tuple[str, str]] = None,
    domain_name: str = "",
    verbose: bool = False,
) -> Tuple[str, str]:
    """Generate period label and description using LLM with rich context.

    Args:
        themes: List of dominant themes
        representative_papers: Top papers for the period
        start_year: Period start year
        end_year: Period end year
        previous_periods: List of (label, description) from previous periods
        domain_name: Domain name to avoid redundant labeling
        verbose: Enable verbose logging

    Returns:
        Tuple of (label, description)

    Raises:
        Exception: If LLM query fails or structured output parsing fails
    """
    logger = get_logger(__name__, verbose)

    if verbose:
        logger.info(
            f"  Generating period label and description for {start_year}-{end_year}"
        )
        logger.info(f"  Themes: {themes}")
        logger.info(f"  Representative papers: {len(representative_papers)}")
        logger.info("  Loading period context...")

    papers_to_use = representative_papers
    context = load_period_context(papers_to_use, start_year, end_year)
    domain_name = domain_name.replace("_", " ").title()

    if verbose:
        logger.info(
            f"  Context loaded: {len(context.get('paper_descriptions', []))} paper descriptions"
        )
        logger.info(f"  Context keywords: {len(context.get('keywords', []))}")

    previous_context = ""
    if previous_periods:
        previous_context = "\n\nPREVIOUS PERIODS FOR CONTEXT:\n"
        for i, (start_year, end_year, prev_label, prev_desc) in enumerate(
            previous_periods, 1
        ):
            previous_context += f"Period {start_year}-{end_year}: {prev_label}\n\n"

        if verbose:
            logger.info(f"  Previous periods context: {len(previous_periods)} periods")
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

    model = "gemma3n:latest"
    if verbose:
        logger.info(f"  Sending LLM query with {model} model...")
        logger.info(f"  Prompt length: {len(prompt)} characters")

    response = query_llm_structured(prompt, PeriodLabelResponse, model=model)

    if verbose:
        logger.info("  LLM query completed successfully")

    label = response.label
    description = response.description

    if not label or not description:
        raise ValueError(f"Missing label or description in response: {response}")

    if verbose:
        logger.info(f"  Final label: '{label}'")
        logger.info(f"  Final description: '{description[:150]}...' (truncated)")

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
    verbose: bool = False,
) -> Tuple[str, str]:
    """Generate LLM-based label and description for merged segments.

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
    all_papers = list(segment1_papers) + list(segment2_papers)

    seen_titles = set()
    unique_papers = []
    for paper in all_papers:
        title = paper.get("title", "")
        if title and title not in seen_titles:
            seen_titles.add(title)
            unique_papers.append(paper)

    unique_papers.sort(
        key=lambda p: p.get("score", p.get("citation_count", 0)), reverse=True
    )
    top_papers = unique_papers[:10]

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

    model = "gemma3n:latest"
    if verbose:
        logger.info(f"  Sending LLM query with {model} model...")
        logger.info(f"  Prompt length: {len(prompt)} characters")

    response = query_llm_structured(prompt, MergedSegmentResponse, model=model)

    label = response.label
    description = response.description

    logger.info(
        f"LLM merge labeling with {model} completed for {merged_period[0]}-{merged_period[1]}"
    )
    return label, description
