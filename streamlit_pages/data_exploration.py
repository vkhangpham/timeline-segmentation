"""Data Exploration Page
Visualizes input data structure and statistics for the selected domain.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from core.data.data_processing import load_domain_data


def load_data_if_needed():
    """Load academic years data if not already loaded."""
    if st.session_state.academic_years is None:
        if st.session_state.selected_domain and st.session_state.algorithm_config:
            with st.spinner("Loading domain data..."):
                start_time = time.time()
                success, academic_years, error_message = load_domain_data(
                    domain_name=st.session_state.selected_domain,
                    algorithm_config=st.session_state.algorithm_config,
                    data_directory="resources",
                    min_papers_per_year=5,
                    apply_year_filtering=True,
                    verbose=False,
                )

                load_time = time.time() - start_time
                st.session_state.timing_data["Data Loading"] = load_time

                if success:
                    st.session_state.academic_years = academic_years
                    st.success(f"✅ Data loaded successfully in {load_time:.2f}s")
                else:
                    st.error(f"❌ Failed to load data: {error_message}")
                    return False
        else:
            st.warning("Please select a domain and configure parameters in the sidebar")
            return False
    return True


def create_papers_timeline_chart(academic_years):
    """Create interactive timeline chart of papers per year."""
    years = [ay.year for ay in academic_years]
    paper_counts = [ay.paper_count for ay in academic_years]
    citation_counts = [ay.total_citations for ay in academic_years]

    fig = make_subplots(
        rows=2,
        cols=1,
        subplot_titles=("Papers Per Year", "Citations Per Year"),
        vertical_spacing=0.1,
    )

    # Papers timeline
    fig.add_trace(
        go.Scatter(
            x=years,
            y=paper_counts,
            mode="lines+markers",
            name="Papers",
            line=dict(color="#1f77b4", width=2),
            marker=dict(size=8),
            hovertemplate="<b>Year:</b> %{x}<br><b>Papers:</b> %{y}<extra></extra>",
        ),
        row=1,
        col=1,
    )

    # Citations timeline
    fig.add_trace(
        go.Scatter(
            x=years,
            y=citation_counts,
            mode="lines+markers",
            name="Citations",
            line=dict(color="#ff7f0e", width=2),
            marker=dict(size=8),
            hovertemplate="<b>Year:</b> %{x}<br><b>Citations:</b> %{y}<extra></extra>",
        ),
        row=2,
        col=1,
    )

    fig.update_layout(
        height=500,
        title_text="Publication and Citation Trends Over Time",
        showlegend=False,
    )

    fig.update_xaxes(title_text="Year")
    fig.update_yaxes(title_text="Count")

    return fig


def create_keyword_evolution_chart(academic_years, top_n=10):
    """Create keyword evolution heatmap."""
    # Collect all keywords across years
    all_keywords = set()
    for ay in academic_years:
        all_keywords.update(ay.top_keywords[:top_n])

    # Create matrix of keyword frequencies by year
    keyword_data = []
    years = [ay.year for ay in academic_years]

    for keyword in list(all_keywords)[:top_n]:  # Limit to manageable number
        frequencies = []
        for ay in academic_years:
            freq = ay.keyword_frequencies.get(keyword, 0)
            frequencies.append(freq)
        keyword_data.append(frequencies)

    if keyword_data:
        fig = go.Figure(
            data=go.Heatmap(
                z=keyword_data,
                x=years,
                y=list(all_keywords)[:top_n],
                colorscale="Viridis",
                hoverongaps=False,
                hovertemplate="<b>Keyword:</b> %{y}<br><b>Year:</b> %{x}<br><b>Frequency:</b> %{z}<extra></extra>",
            )
        )

        fig.update_layout(
            title=f"Top {top_n} Keywords Evolution Over Time",
            xaxis_title="Year",
            yaxis_title="Keywords",
            height=400,
        )

        return fig
    return None


def create_domain_statistics_cards(academic_years):
    """Create statistics cards for the domain."""
    total_papers = sum(ay.paper_count for ay in academic_years)
    total_citations = sum(ay.total_citations for ay in academic_years)
    year_span = (
        max(ay.year for ay in academic_years)
        - min(ay.year for ay in academic_years)
        + 1
    )
    avg_papers_per_year = total_papers / len(academic_years)
    avg_citations_per_paper = total_citations / total_papers if total_papers > 0 else 0

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric("Total Papers", f"{total_papers:,}")
    with col2:
        st.metric("Total Citations", f"{total_citations:,}")
    with col3:
        st.metric("Year Span", f"{year_span} years")
    with col4:
        st.metric("Avg Papers/Year", f"{avg_papers_per_year:.1f}")
    with col5:
        st.metric("Avg Citations/Paper", f"{avg_citations_per_paper:.1f}")


def show_paper_details_table(academic_years):
    """Show searchable table of papers."""
    st.subheader("📄 Paper Details")

    # Create papers dataframe
    papers_data = []
    for ay in academic_years:
        for paper in ay.papers:
            papers_data.append(
                {
                    "Year": paper.pub_year,
                    "Title": (
                        paper.title[:100] + "..."
                        if len(paper.title) > 100
                        else paper.title
                    ),
                    "Citations": paper.cited_by_count,
                    "Keywords": ", ".join(paper.keywords[:5])
                    + ("..." if len(paper.keywords) > 5 else ""),
                    "ID": paper.id,
                }
            )

    if papers_data:
        papers_df = pd.DataFrame(papers_data)

        # Search and filter
        col1, col2 = st.columns([3, 1])
        with col1:
            search_term = st.text_input("🔍 Search papers (title/keywords)", "")
        with col2:
            year_filter = st.selectbox(
                "Filter by Year", ["All"] + sorted(papers_df["Year"].unique().tolist())
            )

        # Apply filters
        filtered_df = papers_df.copy()
        if search_term:
            mask = filtered_df["Title"].str.contains(
                search_term, case=False, na=False
            ) | filtered_df["Keywords"].str.contains(search_term, case=False, na=False)
            filtered_df = filtered_df[mask]

        if year_filter != "All":
            filtered_df = filtered_df[filtered_df["Year"] == year_filter]

        st.write(f"Showing {len(filtered_df)} of {len(papers_df)} papers")
        st.dataframe(filtered_df, use_container_width=True, hide_index=True)


def show_yearly_keyword_analysis(academic_years):
    """Show detailed year-by-year keyword analysis."""
    st.subheader("🏷️ Yearly Keyword Analysis")

    # Year selector
    years = [ay.year for ay in academic_years]
    selected_year = st.selectbox("Select Year for Detailed Analysis", years)

    # Find selected academic year
    selected_ay = next((ay for ay in academic_years if ay.year == selected_year), None)

    if selected_ay:
        col1, col2 = st.columns([1, 1])

        with col1:
            st.write(f"**Year {selected_year} Overview:**")
            st.write(f"- Papers: {selected_ay.paper_count}")
            st.write(f"- Citations: {selected_ay.total_citations}")
            st.write(f"- Unique Keywords: {len(selected_ay.keyword_frequencies)}")

            # Top keywords
            top_keywords = selected_ay.top_keywords[:15]
            st.write("**Top Keywords:**")
            for i, kw in enumerate(top_keywords, 1):
                freq = selected_ay.keyword_frequencies.get(kw, 0)
                st.write(f"{i}. {kw} ({freq})")

        with col2:
            # Keyword frequency distribution
            if selected_ay.keyword_frequencies:
                kw_df = pd.DataFrame(
                    [
                        {"Keyword": kw, "Frequency": freq}
                        for kw, freq in list(selected_ay.keyword_frequencies.items())[
                            :20
                        ]
                    ]
                )

                fig = px.bar(
                    kw_df,
                    x="Frequency",
                    y="Keyword",
                    orientation="h",
                    title=f"Top 20 Keywords - {selected_year}",
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)


def show_data_exploration():
    """Main data exploration page function."""
    st.header("📁 Data Exploration")
    st.write("Explore the input data structure and statistics for the selected domain.")

    # Load data if needed
    if not load_data_if_needed():
        return

    academic_years = st.session_state.academic_years

    # Domain statistics
    st.subheader("📊 Domain Statistics")
    create_domain_statistics_cards(academic_years)

    # Timeline visualization
    st.subheader("📈 Publication Timeline")
    timeline_fig = create_papers_timeline_chart(academic_years)
    st.plotly_chart(timeline_fig, use_container_width=True)

    # Keyword evolution
    st.subheader("🏷️ Keyword Evolution")
    col1, col2 = st.columns([1, 4])
    with col1:
        top_n_keywords = st.slider("Number of top keywords to show", 5, 20, 10)

    with col2:
        keyword_fig = create_keyword_evolution_chart(academic_years, top_n_keywords)
        if keyword_fig:
            st.plotly_chart(keyword_fig, use_container_width=True)
        else:
            st.info("No keyword data available for visualization")

    # Detailed analysis sections
    tab1, tab2 = st.tabs(["📄 Paper Details", "🏷️ Yearly Analysis"])

    with tab1:
        show_paper_details_table(academic_years)

    with tab2:
        show_yearly_keyword_analysis(academic_years)

    # Data export
    st.subheader("💾 Data Export")
    if st.button("Export Data Summary as JSON"):
        summary_data = {
            "domain": st.session_state.selected_domain,
            "total_papers": sum(ay.paper_count for ay in academic_years),
            "total_citations": sum(ay.total_citations for ay in academic_years),
            "year_range": [
                min(ay.year for ay in academic_years),
                max(ay.year for ay in academic_years),
            ],
            "yearly_data": [
                {
                    "year": ay.year,
                    "papers": ay.paper_count,
                    "citations": ay.total_citations,
                    "top_keywords": list(ay.top_keywords[:10]),
                }
                for ay in academic_years
            ],
        }

        st.download_button(
            label="Download Summary JSON",
            data=pd.Series(summary_data).to_json(indent=2),
            file_name=f"{st.session_state.selected_domain}_data_summary.json",
            mime="application/json",
        )
