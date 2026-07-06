"""
===========================================================================
  THESIS DEMO APP
  LLM-Powered Climate Policy Summariser & Gap Analyser
  University of Europe for Applied Sciences, Potsdam · 2026
  Streamlit UI — runs locally (VS Code) and on Streamlit Cloud
===========================================================================
"""

import os
import sys
import json
import re
import time
from pathlib import Path
from typing import Optional

import streamlit as st

# ── Page config — must be first Streamlit call ────────────────────────────────
st.set_page_config(
    page_title = "Climate Policy Analyser",
    page_icon  = "🌍",
    layout     = "wide",
    initial_sidebar_state = "expanded",
)

# ── Path setup — works locally and on Streamlit Cloud ─────────────────────────
APP_DIR = Path(__file__).resolve().parent
for _p in [
    APP_DIR / "LLM_Integration_Code",
    APP_DIR / "Gap_Analysis_Code",
    APP_DIR,
]:
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

# Change working dir so relative paths (data/processed/...) resolve correctly
os.chdir(APP_DIR)

# ── API key — Streamlit Secrets (Cloud) or env var (local) ───────────────────
def get_api_key() -> Optional[str]:
    # Streamlit Cloud: set in app settings → Secrets
    try:
        return st.secrets["OPENAI_API_KEY"]
    except Exception:
        pass
    # Local: set in environment (activate.bat or .env)
    return os.environ.get("OPENAI_API_KEY")

# ═══════════════════════════════════════════════════════════════════════════════
#  CUSTOM CSS
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');

/* ── Global ── */
html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
}

/* ── Header band ── */
.app-header {
    background: linear-gradient(135deg, #0f2027 0%, #1a3a4a 50%, #0d3b2e 100%);
    padding: 1.2rem 1.8rem;
    border-radius: 10px;
    margin-bottom: 1.2rem;
    border-left: 4px solid #22d3a5;
}
.app-header h1 {
    color: #e2f7f1;
    font-size: 1.3rem;
    font-weight: 600;
    margin: 0 0 3px 0;
    letter-spacing: -0.02em;
}
.app-header p {
    color: #7fb8a8;
    font-size: 0.78rem;
    margin: 0;
    font-family: 'IBM Plex Mono', monospace;
}

/* ── Section labels ── */
.section-label {
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #64748b;
    margin-bottom: 0.5rem;
    padding-bottom: 0.3rem;
    border-bottom: 1px solid #e2e8f0;
}

/* ── Answer card ── */
.answer-card {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-left: 4px solid #22d3a5;
    border-radius: 8px;
    padding: 1.1rem 1.3rem;
    margin: 0.8rem 0;
    font-size: 0.92rem;
    line-height: 1.75;
    color: #1e293b;
}

/* ── Citation highlight ── */
.citation-tag {
    display: inline-block;
    background: #dbeafe;
    color: #1d4ed8;
    border: 1px solid #bfdbfe;
    border-radius: 4px;
    padding: 1px 6px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    font-weight: 500;
    margin: 0 1px;
    white-space: nowrap;
}

/* ── Source chunk card ── */
.chunk-card {
    background: #ffffff;
    border: 1px solid #e5e7eb;
    border-radius: 8px;
    padding: 0.85rem 1rem;
    margin-bottom: 0.6rem;
    font-size: 0.85rem;
}
.chunk-meta {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
    margin-bottom: 0.5rem;
}
.chunk-badge {
    font-size: 0.7rem;
    padding: 2px 8px;
    border-radius: 12px;
    font-weight: 500;
    font-family: 'IBM Plex Mono', monospace;
}
.badge-country { background: #d1fae5; color: #065f46; }
.badge-source  { background: #e0e7ff; color: #3730a3; }
.badge-year    { background: #fef9c3; color: #713f12; }
.badge-score   { background: #f3f4f6; color: #374151; }
.badge-hybrid  { background: #fce7f3; color: #9d174d; }
.chunk-text {
    color: #374151;
    line-height: 1.6;
    font-size: 0.84rem;
    border-top: 1px solid #f3f4f6;
    padding-top: 0.5rem;
    margin-top: 0.4rem;
}

/* ── Gap analysis cards ── */
.gap-country-card {
    background: #ffffff;
    border: 1px solid #e5e7eb;
    border-radius: 10px;
    padding: 1rem 1.1rem;
    margin-bottom: 0.8rem;
}
.gap-severity-critical { border-left: 4px solid #dc2626; }
.gap-severity-significant { border-left: 4px solid #f59e0b; }
.gap-severity-minor { border-left: 4px solid #84cc16; }
.gap-severity-aligned { border-left: 4px solid #22d3a5; }

.severity-badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 12px;
    font-size: 0.72rem;
    font-weight: 600;
    font-family: 'IBM Plex Mono', monospace;
    margin-left: 8px;
}
.sev-critical    { background: #fee2e2; color: #991b1b; }
.sev-significant { background: #fef3c7; color: #92400e; }
.sev-minor       { background: #f0fdf4; color: #166534; }
.sev-aligned     { background: #d1fae5; color: #065f46; }

.gap-finding {
    background: #f8fafc;
    border: 1px solid #e5e7eb;
    border-radius: 6px;
    padding: 0.7rem 0.9rem;
    margin: 0.5rem 0;
    font-size: 0.83rem;
}
.gap-finding-type {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem;
    font-weight: 600;
    text-transform: uppercase;
    color: #6b7280;
    margin-bottom: 3px;
}
.gap-quant {
    background: #fffbeb;
    border: 1px solid #fde68a;
    border-radius: 4px;
    padding: 3px 8px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.75rem;
    color: #92400e;
    display: inline-block;
    margin-top: 4px;
}

/* ── Commitment item ── */
.commitment-item {
    display: flex;
    align-items: flex-start;
    gap: 8px;
    padding: 6px 0;
    border-bottom: 1px solid #f3f4f6;
    font-size: 0.85rem;
}
.commitment-value {
    font-family: 'IBM Plex Mono', monospace;
    font-weight: 600;
    color: #0f172a;
    white-space: nowrap;
}
.commitment-cond {
    background: #fef3c7;
    color: #92400e;
    font-size: 0.68rem;
    padding: 1px 6px;
    border-radius: 10px;
    font-weight: 600;
    white-space: nowrap;
}

/* ── IPCC benchmark box ── */
.ipcc-box {
    background: #eff6ff;
    border: 1px solid #bfdbfe;
    border-radius: 8px;
    padding: 0.8rem 1rem;
    font-size: 0.85rem;
    color: #1e40af;
    margin-bottom: 1rem;
}
.ipcc-box strong { color: #1d4ed8; }

/* ── Status / info boxes ── */
.info-box {
    background: #f0fdf4;
    border: 1px solid #bbf7d0;
    border-radius: 6px;
    padding: 0.7rem 1rem;
    font-size: 0.84rem;
    color: #166534;
}
.warn-box {
    background: #fffbeb;
    border: 1px solid #fde68a;
    border-radius: 6px;
    padding: 0.7rem 1rem;
    font-size: 0.84rem;
    color: #92400e;
}
.error-box {
    background: #fef2f2;
    border: 1px solid #fecaca;
    border-radius: 6px;
    padding: 0.7rem 1rem;
    font-size: 0.84rem;
    color: #991b1b;
}

/* ── Pairwise comparison ── */
.pairwise-card {
    background: #fafafa;
    border: 1px solid #e5e7eb;
    border-radius: 8px;
    padding: 1rem 1.1rem;
    margin-top: 0.8rem;
}

/* ── Metric cards ── */
.metric-row {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 10px;
    margin-bottom: 1rem;
}
.metric-card {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 0.7rem;
    text-align: center;
}
.metric-val {
    font-size: 1.3rem;
    font-weight: 600;
    color: #0f172a;
    font-family: 'IBM Plex Mono', monospace;
}
.metric-lbl {
    font-size: 0.7rem;
    color: #64748b;
    margin-top: 2px;
}

/* ── Streamlit overrides ── */
div[data-testid="stSidebar"] { background: #f8fafc; }
.stButton button {
    font-family: 'IBM Plex Sans', sans-serif;
    font-weight: 500;
}
h3 { font-weight: 500 !important; }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

G20_COUNTRIES = [
    "Argentina", "Australia", "Brazil", "Canada", "China",
    "EU", "India", "Indonesia", "Japan", "Mexico",
    "Russia", "Saudi Arabia", "South Africa", "South Korea",
    "Turkey", "UK", "USA",
]

DOC_TYPES = ["NDC", "IPCC_AR6"]

DIMENSIONS = {
    "renewable_energy":           "Renewable Energy Targets",
    "net_zero_pathway":           "Net Zero / Carbon Neutrality",
    "adaptation_finance":         "Adaptation Finance",
    "loss_and_damage":            "Loss and Damage",
    "lulucf_forestry":            "LULUCF & Forestry Sinks",
    "transport_decarbonisation":  "Transport Decarbonisation",
    "methane_reduction":          "Methane Reduction",
}

SEVERITY_CSS = {
    "CRITICAL":    ("gap-severity-critical",    "sev-critical"),
    "SIGNIFICANT": ("gap-severity-significant", "sev-significant"),
    "MINOR":       ("gap-severity-minor",       "sev-minor"),
    "ALIGNED":     ("gap-severity-aligned",     "sev-aligned"),
}

SEVERITY_EMOJI = {
    "CRITICAL": "🔴", "SIGNIFICANT": "🟠",
    "MINOR": "🟡",    "ALIGNED": "🟢",
}

NDC_CYCLE_YEARS = {1: "2015–2016", 2: "2021–2023"}


# ═══════════════════════════════════════════════════════════════════════════════
#  PIPELINE LOADER  (cached — loads once per session)
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner=False)
def load_pipeline(api_key: str):
    """
    Load RAGPipeline once and cache it for the entire Streamlit session.
    Subsequent interactions reuse the loaded retriever and API client.
    """
    try:
        from generation import RAGPipeline
        pipe = RAGPipeline(
            backend      = "gpt4o",
            top_k        = 5,
            embed_device = "cpu",
            openai_key   = api_key,
        )
        return pipe, None
    except Exception as e:
        return None, str(e)


@st.cache_resource(show_spinner=False)
def load_gap_analyser(api_key: str):
    """Load GapAnalyser once and cache."""
    try:
        from gap_analysis import GapAnalyser
        gen_dir = str(APP_DIR / "LLM_Integration_Code")
        analyser = GapAnalyser(
            backend      = "gpt4o",
            top_k_ndc    = 8,
            top_k_ipcc   = 6,
            embed_device = "cpu",
            gen_dir      = gen_dir,
        )
        return analyser, None
    except Exception as e:
        return None, str(e)


# ═══════════════════════════════════════════════════════════════════════════════
#  HELPER RENDERERS
# ═══════════════════════════════════════════════════════════════════════════════

def highlight_citations(text: str) -> str:
    """Wrap [doc_id, p.N] citation tags in styled HTML spans."""
    def replace_citation(m):
        raw = m.group(0)
        return f'<span class="citation-tag">{raw}</span>'
    return re.sub(r'\[[A-Za-z0-9_\-\.]+,\s*p\.?\s*\d+\]', replace_citation, text)


def render_answer(answer: str, insufficient: bool) -> None:
    if insufficient:
        st.markdown(
            '<div class="warn-box">⚠ <strong>Insufficient context</strong> — '
            'the retrieved documents do not contain enough information to answer this query. '
            'Try broadening your query or removing metadata filters.</div>',
            unsafe_allow_html=True
        )
    else:
        highlighted = highlight_citations(answer)
        st.markdown(
            f'<div class="answer-card">{highlighted}</div>',
            unsafe_allow_html=True
        )


def render_source_chunks(chunks: list) -> None:
    st.markdown('<div class="section-label">Retrieved source chunks</div>',
                unsafe_allow_html=True)
    for i, chunk in enumerate(chunks, 1):
        country   = chunk.get("country") or chunk.get("wg") or "—"
        source    = chunk.get("source", "").replace("UNFCCC_NDC_Registry", "NDC")
        year      = chunk.get("year", "")
        rrf       = chunk.get("rrf_score", 0)
        mode      = chunk.get("retrieval_mode", "")
        section   = (chunk.get("section") or "")[:65]
        text      = (chunk.get("text") or "")[:350]
        doc_id    = chunk.get("doc_id", "")

        mode_badge = (
            '<span class="chunk-badge badge-hybrid">D+S hybrid</span>'
            if mode == "hybrid" else
            '<span class="chunk-badge badge-source">dense</span>'
            if mode == "dense_only" else
            '<span class="chunk-badge badge-year">sparse</span>'
        )
        st.markdown(f"""
<div class="chunk-card">
  <div class="chunk-meta">
    <span class="chunk-badge badge-country">{country}</span>
    <span class="chunk-badge badge-source">{source}</span>
    <span class="chunk-badge badge-year">{year}</span>
    <span class="chunk-badge badge-score">RRF {rrf:.4f}</span>
    {mode_badge}
  </div>
  <div style="font-size:0.75rem;color:#6b7280;margin-bottom:4px;font-family:'IBM Plex Mono',monospace;">
    #{i} · {doc_id} · {section}
  </div>
  <div class="chunk-text">{text}{'...' if len(chunk.get('text','')) > 350 else ''}</div>
</div>
""", unsafe_allow_html=True)


def render_gap_country(country: str, analysis: dict) -> None:
    sev = analysis.get("overall_severity", "UNKNOWN")
    score = analysis.get("overall_score", 0)
    measurability = analysis.get("measurability", {})
    commitments = analysis.get("commitments", [])
    gaps = analysis.get("gap_findings", [])
    summary = analysis.get("summary", "")

    card_cls, badge_cls = SEVERITY_CSS.get(sev, ("", ""))
    emoji = SEVERITY_EMOJI.get(sev, "⚪")

    st.markdown(f"""
<div class="gap-country-card {card_cls}">
  <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:8px">
    <span style="font-weight:600;font-size:1rem;color:#0f172a">{emoji} {country}</span>
    <span>
      <span class="severity-badge {badge_cls}">{sev} ({score:.2f})</span>
      <span class="chunk-badge badge-year" style="margin-left:6px">
        {measurability.get('level','?')} · {measurability.get('score',0):.2f}
      </span>
    </span>
  </div>
  <p style="font-size:0.84rem;color:#475569;margin:0 0 10px 0;line-height:1.6">{summary}</p>
""", unsafe_allow_html=True)

    # Commitments
    if commitments:
        st.markdown(
            f'<div style="font-size:0.75rem;font-weight:600;color:#374151;'
            f'margin-bottom:4px">COMMITMENTS ({len(commitments)})</div>',
            unsafe_allow_html=True
        )
        for c in commitments:
            val   = c.get("target_value") or "no specific value"
            year_ = c.get("target_year") or "?"
            cond  = c.get("conditional", False)
            doc   = c.get("source_doc", "")
            page  = c.get("source_page", "?")
            cond_html = '<span class="commitment-cond">CONDITIONAL</span>' if cond else ""
            st.markdown(f"""
<div class="commitment-item">
  <span style="color:#22d3a5;font-size:0.9rem">▸</span>
  <span><span class="commitment-value">{val}</span> by {year_} {cond_html}
  <span style="font-size:0.75rem;color:#9ca3af;margin-left:6px;font-family:'IBM Plex Mono',monospace">
    [{doc}, p.{page}]</span></span>
</div>
""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Gap findings
    if gaps:
        st.markdown(
            f'<div style="font-size:0.75rem;font-weight:600;color:#374151;'
            f'margin-bottom:4px">GAP FINDINGS ({len(gaps)})</div>',
            unsafe_allow_html=True
        )
        for g in gaps:
            g_sev   = g.get("severity", "")
            g_type  = g.get("gap_type", "").replace("_", " ")
            g_desc  = g.get("description", "")
            g_quant = g.get("quantified_gap")
            g_ndc   = g.get("ndc_evidence", "")[:120]
            g_ipcc  = g.get("ipcc_evidence", "")[:120]
            _, gb   = SEVERITY_CSS.get(g_sev, ("", ""))
            quant_html = (f'<div class="gap-quant">{g_quant}</div>'
                          if g_quant else "")
            st.markdown(f"""
<div class="gap-finding">
  <div class="gap-finding-type">
    <span class="severity-badge {gb}" style="margin-left:0">{g_sev}</span>
    &nbsp;{g_type}
  </div>
  <div style="color:#1e293b;margin:4px 0">{g_desc}</div>
  {quant_html}
  <div style="font-size:0.78rem;color:#6b7280;margin-top:6px;line-height:1.5">
    <strong>NDC:</strong> {g_ndc}...<br>
    <strong>IPCC:</strong> {g_ipcc}...
  </div>
</div>
""", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  APP LAYOUT
# ═══════════════════════════════════════════════════════════════════════════════

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-header">
  <h1>🌍 Climate Policy Analyser</h1>
  <p>LLM-Powered NDC Summariser & Gap Analyser · M.Sc. Data Science · UE Potsdam 2026</p>
</div>
""", unsafe_allow_html=True)

# ── API key check ─────────────────────────────────────────────────────────────
api_key = get_api_key()
if not api_key:
    st.markdown("""
<div class="error-box">
<strong>⚠ No API key found.</strong><br>
• <strong>Local (VS Code):</strong> make sure <code>OPENAI_API_KEY</code> is set in your environment.<br>
• <strong>Streamlit Cloud:</strong> go to App Settings → Secrets and add:
<pre style="margin:6px 0 0 0;font-size:0.8rem">OPENAI_API_KEY = "sk-proj-..."</pre>
</div>
""", unsafe_allow_html=True)
    st.stop()

# ═══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR — filters + settings
# ═══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("### 🗂 Document Filters")
    st.caption("Narrow retrieval to specific documents")

    selected_countries = st.multiselect(
        "Countries",
        options = G20_COUNTRIES,
        default = [],
        placeholder = "All G20 countries",
        help = "Leave empty to search all countries"
    )

    selected_doc_types = st.multiselect(
        "Document type",
        options   = DOC_TYPES,
        default   = [],
        placeholder = "NDC + IPCC AR6",
    )

    selected_ndc_cycles = st.multiselect(
        "NDC cycle",
        options = [1, 2],
        default = [],
        format_func = lambda x: f"Cycle {x} ({NDC_CYCLE_YEARS[x]})",
        placeholder = "Both cycles",
    )

    st.markdown("---")
    st.markdown("### ⚙ Retrieval Settings")
    top_k = st.slider("Chunks to retrieve (top-k)", 3, 10, 5)

    st.markdown("---")
    st.markdown("### 📊 Session Stats")
    if "query_count" not in st.session_state:
        st.session_state.query_count = 0
    if "gap_count" not in st.session_state:
        st.session_state.gap_count = 0
    if "total_tokens" not in st.session_state:
        st.session_state.total_tokens = 0

    col1, col2 = st.columns(2)
    col1.metric("Queries", st.session_state.query_count)
    col2.metric("Gap runs", st.session_state.gap_count)
    st.caption(f"~{st.session_state.total_tokens:,} tokens used this session")

    st.markdown("---")
    st.markdown(
        '<div style="font-size:0.72rem;color:#94a3b8;line-height:1.6">'
        'Corpus: G20 NDC1 + NDC2 · IPCC AR6<br>'
        'Retrieval: bge-large + BM25 + RRF<br>'
        'Generation: GPT-4o (OpenAI API)<br>'
        'Embedding: BAAI/bge-large-en-v1.5'
        '</div>',
        unsafe_allow_html=True
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  BUILD CHROMA FILTER FROM SIDEBAR SELECTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def build_filter() -> Optional[dict]:
    """Convert sidebar selections into a ChromaDB where= filter."""
    conditions = []

    if selected_countries and len(selected_countries) == 1:
        conditions.append({"country": selected_countries[0]})
    elif selected_countries:
        conditions.append({"$or": [{"country": c} for c in selected_countries]})

    if selected_doc_types and len(selected_doc_types) == 1:
        conditions.append({"doc_type": selected_doc_types[0]})

    if selected_ndc_cycles and len(selected_ndc_cycles) == 1:
        conditions.append({"ndc_cycle": selected_ndc_cycles[0]})

    if not conditions:
        return None
    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}


# ═══════════════════════════════════════════════════════════════════════════════
#  TABS
# ═══════════════════════════════════════════════════════════════════════════════

tab_rag, tab_gap, tab_history = st.tabs([
    "🔍 Policy Query",
    "📊 Gap Analysis",
    "📁 Previous Results",
])


# ───────────────────────────────────────────────────────────────────────────────
#  TAB 1 — RAG QUERY
# ───────────────────────────────────────────────────────────────────────────────

with tab_rag:

    st.markdown('<div class="section-label">Natural Language Query</div>',
                unsafe_allow_html=True)

    # Preset examples
    EXAMPLE_QUERIES = [
        "What is India's 2030 renewable energy target?",
        "What does IPCC WG3 recommend for energy sector decarbonisation?",
        "How does the UK's net zero commitment compare to Australia's?",
        "What are Indonesia's climate adaptation commitments?",
        "Which G20 countries have conditional NDC targets?",
    ]

    with st.expander("💡 Example queries", expanded=False):
        for eq in EXAMPLE_QUERIES:
            if st.button(eq, key=f"eq_{eq[:20]}", use_container_width=True):
                st.session_state["preset_query"] = eq

    query_default = st.session_state.get("preset_query", "")
    query = st.text_area(
        "Enter your question about climate policy:",
        value   = query_default,
        height  = 90,
        placeholder = "e.g. What is China's carbon peak target year and emissions reduction commitment?",
    )

    col_run, col_clear = st.columns([3, 1])
    with col_run:
        run_query = st.button(
            "🔍 Search & Answer",
            type            = "primary",
            use_container_width = True,
            disabled        = not query.strip(),
        )
    with col_clear:
        if st.button("Clear", use_container_width=True):
            st.session_state.pop("preset_query", None)
            st.session_state.pop("last_rag_resp", None)
            st.rerun()

    # ── Run query ──────────────────────────────────────────────────────────────
    if run_query and query.strip():
        api_filter = build_filter()

        with st.spinner("Loading pipeline..."):
            pipe, err = load_pipeline(api_key)

        if err:
            st.markdown(
                f'<div class="error-box">Failed to load pipeline: {err}</div>',
                unsafe_allow_html=True
            )
        else:
            # Update top_k on the pipeline
            pipe.top_k = top_k

            with st.spinner("Retrieving chunks and generating answer..."):
                t0 = time.time()
                try:
                    resp = pipe.query(
                        query   = query,
                        filters = api_filter,
                        save    = True,
                    )
                    elapsed = time.time() - t0

                    # Update stats
                    st.session_state.query_count += 1
                    st.session_state.total_tokens += resp.token_stats.get("total_tokens", 0)
                    st.session_state["last_rag_resp"] = resp

                    # Save to history
                    if "rag_history" not in st.session_state:
                        st.session_state.rag_history = []
                    st.session_state.rag_history.append({
                        "query":     query,
                        "answer":    resp.answer[:200],
                        "countries": [c.get("country") for c in resp.retrieved_chunks if c.get("country")],
                        "elapsed":   round(elapsed, 1),
                        "tokens":    resp.token_stats.get("total_tokens", 0),
                    })

                except Exception as e:
                    st.error(f"Query failed: {e}")
                    resp = None

    # ── Display result ─────────────────────────────────────────────────────────
    resp = st.session_state.get("last_rag_resp")
    if resp:
        # Info bar
        elapsed_str = ""
        tok_str = f"{resp.token_stats.get('total_tokens', 0):,} tokens"
        filter_str = f"Filter: {build_filter() or 'None'}"

        col_a, col_b, col_c = st.columns(3)
        col_a.markdown(
            f'<div class="metric-card"><div class="metric-val">'
            f'{len(resp.retrieved_chunks)}</div>'
            f'<div class="metric-lbl">chunks retrieved</div></div>',
            unsafe_allow_html=True
        )
        col_b.markdown(
            f'<div class="metric-card"><div class="metric-val">'
            f'{len(resp.citations)}</div>'
            f'<div class="metric-lbl">citations found</div></div>',
            unsafe_allow_html=True
        )
        col_c.markdown(
            f'<div class="metric-card"><div class="metric-val">'
            f'{resp.token_stats.get("total_tokens", 0):,}</div>'
            f'<div class="metric-lbl">tokens used</div></div>',
            unsafe_allow_html=True
        )

        st.markdown("<br>", unsafe_allow_html=True)

        # Answer
        st.markdown('<div class="section-label">Answer</div>', unsafe_allow_html=True)
        render_answer(resp.answer, resp.insufficient)

        # Citations list
        if resp.citations:
            with st.expander(f"📎 {len(resp.citations)} citations", expanded=True):
                for c in resp.citations:
                    country_str = c.country or c.source or "—"
                    st.markdown(
                        f'`{c.raw_tag}` &nbsp; {country_str} · {c.year or "—"}',
                        unsafe_allow_html=True
                    )

        # Source chunks
        with st.expander(f"📄 Source chunks ({len(resp.retrieved_chunks)})", expanded=False):
            render_source_chunks(resp.retrieved_chunks)

        # Raw JSON download
        resp_dict = resp.to_dict()
        st.download_button(
            label     = "⬇ Download response JSON",
            data      = json.dumps(resp_dict, indent=2, default=str),
            file_name = f"response_{int(time.time())}.json",
            mime      = "application/json",
        )


# ───────────────────────────────────────────────────────────────────────────────
#  TAB 2 — GAP ANALYSIS
# ───────────────────────────────────────────────────────────────────────────────

with tab_gap:

    st.markdown('<div class="section-label">Gap Analysis Configuration</div>',
                unsafe_allow_html=True)

    col_l, col_r = st.columns([1, 1])

    with col_l:
        gap_countries = st.multiselect(
            "Countries to compare",
            options   = G20_COUNTRIES,
            default   = ["India", "China"],
            help      = "Select 2–5 countries for comparison",
        )
        gap_dimension_key = st.selectbox(
            "Analysis dimension",
            options      = list(DIMENSIONS.keys()),
            format_func  = lambda k: DIMENSIONS[k],
            index        = 0,
        )

    with col_r:
        gap_cycles = st.multiselect(
            "NDC cycle(s)",
            options      = [1, 2],
            default      = [2],
            format_func  = lambda x: f"Cycle {x} ({NDC_CYCLE_YEARS[x]})",
        )
        st.markdown('<br>', unsafe_allow_html=True)
        # Cost estimate
        n_passes = 3 * len(gap_countries) + (1 if len(gap_countries) >= 3 else 0)
        est_cost = n_passes * 0.012
        st.markdown(
            f'<div class="info-box">'
            f'Est. {n_passes} API calls · ~€{est_cost:.2f} cost<br>'
            f'<small>~{27 * len(gap_countries)}s expected runtime</small>'
            f'</div>',
            unsafe_allow_html=True
        )

    run_gap = st.button(
        f"📊 Run Gap Analysis — {DIMENSIONS.get(gap_dimension_key, '')}",
        type            = "primary",
        use_container_width = True,
        disabled        = len(gap_countries) < 2 or not gap_cycles,
    )

    if run_gap:
        if len(gap_countries) < 2:
            st.warning("Select at least 2 countries.")
        else:
            with st.spinner("Loading gap analyser..."):
                analyser, err = load_gap_analyser(api_key)

            if err:
                st.markdown(
                    f'<div class="error-box">Failed to load analyser: {err}</div>',
                    unsafe_allow_html=True
                )
            else:
                progress = st.progress(0, text="Starting gap analysis...")
                status   = st.empty()

                try:
                    status.markdown(
                        f'<div class="info-box">Running 3-pass analysis for '
                        f'{", ".join(gap_countries)}...</div>',
                        unsafe_allow_html=True
                    )
                    t0 = time.time()

                    progress.progress(20, text="Retrieving IPCC benchmark...")
                    report = analyser.analyse(
                        countries  = gap_countries,
                        dimension  = gap_dimension_key,
                        ndc_cycles = gap_cycles,
                        save       = True,
                    )
                    elapsed = time.time() - t0

                    progress.progress(100, text="Complete!")
                    status.empty()

                    st.session_state.gap_count += 1
                    st.session_state["last_gap_report"] = report

                    # Save to history
                    if "gap_history" not in st.session_state:
                        st.session_state.gap_history = []
                    st.session_state.gap_history.append({
                        "dimension": DIMENSIONS[gap_dimension_key],
                        "countries": gap_countries,
                        "elapsed":   round(elapsed, 1),
                        "rankings":  report.rankings,
                    })

                    st.success(f"✓ Analysis complete in {elapsed:.0f}s")

                except Exception as e:
                    progress.empty()
                    status.empty()
                    st.error(f"Gap analysis failed: {e}")
                    report = None

    # ── Display gap report ─────────────────────────────────────────────────────
    report = st.session_state.get("last_gap_report")
    if report:
        # IPCC benchmark
        st.markdown('<div class="section-label">IPCC AR6 Benchmark</div>',
                    unsafe_allow_html=True)
        st.markdown(
            f'<div class="ipcc-box">'
            f'<strong>Dimension:</strong> {report.dimension_label}<br>'
            f'<strong>Benchmark:</strong> {report.ipcc_benchmark.ipcc_target}'
            f'</div>',
            unsafe_allow_html=True
        )

        # Rankings overview
        if report.rankings:
            st.markdown('<div class="section-label">Ambition Rankings (lower gap score = more aligned)</div>',
                        unsafe_allow_html=True)
            ranked = sorted(report.rankings.items(), key=lambda x: x[1])
            n = len(ranked)
            cols = st.columns(n)
            for i, (country, score) in enumerate(ranked):
                sev_emoji = (
                    "🔴" if score >= 0.8 else
                    "🟠" if score >= 0.5 else
                    "🟡" if score >= 0.25 else "🟢"
                )
                cols[i].markdown(
                    f'<div class="metric-card">'
                    f'<div class="metric-val">{sev_emoji} #{i+1}</div>'
                    f'<div style="font-weight:600;font-size:0.85rem;margin:2px 0">{country}</div>'
                    f'<div class="metric-lbl">gap score {score:.2f}</div>'
                    f'</div>',
                    unsafe_allow_html=True
                )

        st.markdown("<br>", unsafe_allow_html=True)

        # Per-country analysis
        st.markdown('<div class="section-label">Per-Country Analysis</div>',
                    unsafe_allow_html=True)
        for country, analysis in report.country_analyses.items():
            render_gap_country(country, asdict_safe(analysis))

        # Pairwise comparison
        if report.pairwise_gaps:
            st.markdown('<div class="section-label">Pairwise Comparison</div>',
                        unsafe_allow_html=True)
            for pw in report.pairwise_gaps:
                pw_d = asdict_safe(pw)
                leader = pw_d.get("ambition_leader", "tied")
                reason = pw_d.get("leadership_reason", "")
                rel_gap = pw_d.get("relative_gap", "")
                conv = pw_d.get("convergence_areas", [])
                div_ = pw_d.get("divergence_areas", [])
                st.markdown(f"""
<div class="pairwise-card">
  <div style="font-weight:600;font-size:0.95rem;margin-bottom:6px">
    {pw_d.get('country_a')} vs {pw_d.get('country_b')}
    <span class="chunk-badge badge-country" style="margin-left:8px">
      Leader: {leader}
    </span>
  </div>
  <p style="font-size:0.84rem;color:#475569;margin:0 0 8px 0">{reason}</p>
  <p style="font-size:0.84rem;color:#374151;margin:0 0 6px 0">{rel_gap}</p>
  <div style="display:flex;gap:16px;flex-wrap:wrap">
    <div>
      <div style="font-size:0.72rem;font-weight:600;color:#16a34a;margin-bottom:3px">CONVERGE</div>
      {''.join(f'<div style="font-size:0.8rem;color:#374151">✓ {c}</div>' for c in conv) or '<span style="font-size:0.8rem;color:#9ca3af">None identified</span>'}
    </div>
    <div>
      <div style="font-size:0.72rem;font-weight:600;color:#dc2626;margin-bottom:3px">DIVERGE</div>
      {''.join(f'<div style="font-size:0.8rem;color:#374151">✗ {d}</div>' for d in div_) or '<span style="font-size:0.8rem;color:#9ca3af">None identified</span>'}
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

        # G20 synthesis
        if report.g20_summary:
            with st.expander("🌐 G20-level synthesis", expanded=True):
                g20 = report.g20_summary
                cols = st.columns(2)
                cols[0].markdown("**Most ambitious:**")
                for c in g20.get("most_ambitious", []):
                    cols[0].markdown(f"  🟢 {c}")
                cols[1].markdown("**Least ambitious:**")
                for c in g20.get("least_ambitious", []):
                    cols[1].markdown(f"  🔴 {c}")
                st.markdown("**Systemic patterns:**")
                for p in g20.get("systemic_patterns", []):
                    st.markdown(f"  • {p}")
                if g20.get("thesis_finding"):
                    st.info(f"📌 **Key thesis finding:** {g20['thesis_finding']}")

        # Policy implications
        if report.policy_implications:
            st.markdown('<div class="section-label">Policy Implications</div>',
                        unsafe_allow_html=True)
            for impl in report.policy_implications:
                st.markdown(
                    f'<div class="warn-box" style="margin-bottom:6px">⚠ {impl}</div>',
                    unsafe_allow_html=True
                )

        # Download JSON
        try:
            from dataclasses import asdict as dc_asdict
            report_dict = report._to_dict()
        except Exception:
            report_dict = {}

        st.download_button(
            label     = "⬇ Download gap report JSON",
            data      = json.dumps(report_dict, indent=2, default=str),
            file_name = f"gap_{report.dimension}_{int(time.time())}.json",
            mime      = "application/json",
        )


# ───────────────────────────────────────────────────────────────────────────────
#  TAB 3 — PREVIOUS RESULTS
# ───────────────────────────────────────────────────────────────────────────────

with tab_history:

    st.markdown('<div class="section-label">Session History</div>',
                unsafe_allow_html=True)

    # RAG history
    rag_hist = st.session_state.get("rag_history", [])
    gap_hist = st.session_state.get("gap_history", [])

    if not rag_hist and not gap_hist:
        st.markdown(
            '<div class="info-box">No queries yet this session. '
            'Run a query or gap analysis to see results here.</div>',
            unsafe_allow_html=True
        )
    else:
        if rag_hist:
            st.markdown("#### Policy Queries")
            for i, h in enumerate(reversed(rag_hist), 1):
                countries_str = ", ".join(set(c for c in h["countries"] if c)) or "all"
                with st.expander(
                    f"#{len(rag_hist)-i+1} · {h['query'][:60]}...",
                    expanded=False
                ):
                    st.markdown(f"**Countries:** {countries_str}")
                    st.markdown(f"**Answer preview:** {h['answer']}...")
                    st.caption(f"{h['elapsed']}s · {h['tokens']:,} tokens")

        if gap_hist:
            st.markdown("#### Gap Analyses")
            for h in reversed(gap_hist):
                ranked = sorted(h["rankings"].items(), key=lambda x: x[1])
                ranked_str = " > ".join(f"{c} ({s:.2f})" for c, s in ranked)
                with st.expander(
                    f"{h['dimension']} — {', '.join(h['countries'])}",
                    expanded=False
                ):
                    st.markdown(f"**Rankings:** {ranked_str}")
                    st.caption(f"{h['elapsed']}s")

    # Saved JSON files from disk
    st.markdown("---")
    st.markdown("#### Saved reports on disk")
    gap_dir = Path("data/gap_reports")
    resp_dir = Path("data/responses")

    saved_gaps = sorted(gap_dir.glob("*.json"), reverse=True) if gap_dir.exists() else []
    saved_resps = sorted(resp_dir.glob("*.json"), reverse=True) if resp_dir.exists() else []

    col1, col2 = st.columns(2)
    with col1:
        st.caption(f"Gap reports ({len(saved_gaps)} files)")
        for f in saved_gaps[:8]:
            with open(f) as fp:
                try:
                    d = json.load(fp)
                    label = f"{d.get('dimension_label','?')} — {', '.join(d.get('countries',[]))}"
                except Exception:
                    label = f.name
            st.download_button(
                label     = f"⬇ {label[:45]}",
                data      = f.read_text(encoding="utf-8"),
                file_name = f.name,
                mime      = "application/json",
                key       = f"dl_gap_{f.name}",
            )
    with col2:
        st.caption(f"Query responses ({len(saved_resps)} files)")
        for f in saved_resps[:8]:
            with open(f) as fp:
                try:
                    d = json.load(fp)
                    label = d.get("query", f.name)[:50]
                except Exception:
                    label = f.name
            st.download_button(
                label     = f"⬇ {label[:45]}",
                data      = f.read_text(encoding="utf-8"),
                file_name = f.name,
                mime      = "application/json",
                key       = f"dl_resp_{f.name}",
            )


# ═══════════════════════════════════════════════════════════════════════════════
#  HELPER — safe asdict that works with both dataclasses and dicts
# ═══════════════════════════════════════════════════════════════════════════════

def asdict_safe(obj) -> dict:
    """Convert dataclass or dict to dict safely."""
    if isinstance(obj, dict):
        return obj
    try:
        from dataclasses import asdict as dc_asdict
        return dc_asdict(obj)
    except Exception:
        return obj.__dict__ if hasattr(obj, "__dict__") else {}
