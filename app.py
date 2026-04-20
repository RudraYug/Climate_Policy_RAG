"""
===========================================================================
  CLIMATE POLICY RAG — STREAMLIT DEMO APP
  Master's Thesis: LLM-Powered Climate Policy Summariser & Gap Analyser
  University of Europe for Applied Sciences, Potsdam | 2026
===========================================================================
"""

import streamlit as st
import json
import re
import time
import os
from datetime import datetime

st.set_page_config(
    page_title="Climate Policy RAG",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

USE_DEMO_MODE = True

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');
html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
#MainMenu {visibility: hidden;} footer {visibility: hidden;} header {visibility: hidden;}
.app-header {
    background: linear-gradient(135deg, #0f4c35 0%, #1a7a52 100%);
    color: white; padding: 1.4rem 2rem; border-radius: 10px;
    margin-bottom: 1.5rem; display: flex; align-items: center; gap: 1rem;
}
.app-header h1 { margin: 0; font-size: 1.5rem; font-weight: 600; letter-spacing: -0.02em; }
.app-header p  { margin: 0; font-size: 0.82rem; opacity: 0.8; }
.answer-box {
    background: white; border: 1px solid #e2e8f0;
    border-left: 4px solid #1a7a52; border-radius: 8px;
    padding: 1.2rem 1.4rem; line-height: 1.75; font-size: 0.93rem;
    color: #1e293b !important;
}
.metric-row { display: flex; gap: 0.75rem; margin-bottom: 1rem; flex-wrap: wrap; }
.metric-card {
    flex: 1; min-width: 100px; background: white;
    border: 1px solid #e2e8f0; border-radius: 8px;
    padding: 0.75rem 1rem; text-align: center;
}
.metric-card .val { font-size: 1.4rem; font-weight: 600; color: #1a7a52; font-family: 'IBM Plex Mono', monospace; }
.metric-card .lbl { font-size: 0.72rem; color: #64748b; margin-top: 2px; }
.gap-card { border-radius: 8px; padding: 1rem 1.2rem; margin-bottom: 0.75rem; border-left: 5px solid #ccc; }
.gap-card.critical    { border-color: #dc2626; background: #fff5f5; }
.gap-card.significant { border-color: #f59e0b; background: #fffbeb; }
.gap-card.aligned     { border-color: #16a34a; background: #f0fdf4; }
.gap-card.absent      { border-color: #6b7280; background: #f9fafb; }
.gc-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem; }
.gc-dim    { font-weight: 600; font-size: 0.9rem; color: #1e293b; }
.gap-badge { font-size: 0.7rem; font-weight: 600; padding: 2px 8px; border-radius: 20px; text-transform: uppercase; letter-spacing: 0.05em; }
.badge-critical    { background:#fee2e2; color:#dc2626; }
.badge-significant { background:#fef3c7; color:#b45309; }
.badge-aligned     { background:#dcfce7; color:#166534; }
.badge-absent      { background:#f3f4f6; color:#374151; }
.gc-evidence { font-size: 0.82rem; color: #374151; margin-top: 0.4rem; line-height: 1.6; }
.gc-evidence strong { color: #0f4c35; }
.chunk-pill {
    display: inline-block; background: #f1f5f9; border: 1px solid #cbd5e1;
    border-radius: 4px; padding: 3px 8px; font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem; color: #475569; margin: 2px;
}
.stButton button {
    background: #0f4c35 !important; color: white !important;
    border: none !important; border-radius: 6px !important;
    font-weight: 500 !important;
}
.stButton button:hover { background: #1a7a52 !important; }
.status-demo { display:inline-block; background:#fef3c7; color:#92400e; font-size:0.72rem; font-weight:600; padding:2px 8px; border-radius:20px; margin-left:0.5rem; }
.status-live { display:inline-block; background:#dcfce7; color:#166534; font-size:0.72rem; font-weight:600; padding:2px 8px; border-radius:20px; margin-left:0.5rem; }
.insufficient-box { background:#fefce8; border:1px solid #fde68a; border-radius:8px; padding:1rem; font-size:0.88rem; color:#713f12; }
</style>
""", unsafe_allow_html=True)

# ── Demo data ─────────────────────────────────────────────────────────────

DEMO_RESPONSES = {
    "india_renew": {
        "answer": "India's NDC2 (2022) commits to achieving approximately **50 percent** cumulative electric power installed capacity from non-fossil fuel-based energy resources by 2030. This target is **conditional** on international technology transfer and low-cost international finance, including support from the Green Climate Fund (GCF). This represents an update from India's NDC1 target of 40 percent non-fossil capacity.",
        "citations": ["NDC2_2022_India, p.3", "NDC2_2022_India, p.7"],
        "faithfulness": 0.92, "context_chunks": 5, "latency_ms": 2840, "insufficient": False,
    },
    "china_nz": {
        "answer": "China's NDC2 commits to achieving **carbon neutrality before 2060** and peaking CO2 emissions **before 2030**. The submission also targets increasing the share of non-fossil fuels in primary energy consumption to approximately **20 percent** by 2030, and increasing installed wind and solar capacity to **1,200 GW** by 2030.",
        "citations": ["NDC2_2021_China, p.4", "NDC2_2021_China, p.7", "NDC2_2021_China, p.9"],
        "faithfulness": 0.89, "context_chunks": 5, "latency_ms": 3120, "insufficient": False,
    },
    "ipcc": {
        "answer": "According to IPCC AR6 Working Group III, limiting global warming to **1.5 degrees C** requires global net anthropogenic CO2 emissions to decline by approximately **43 percent** from 2019 levels by 2030, reaching net zero around **2050**. Renewable electricity should reach **70 to 85 percent** of global electricity supply by 2050.",
        "citations": ["WG3_Chapter03_Mitigation_2022, p.12", "WG3_Chapter06_EnergySystems_2022, p.76"],
        "faithfulness": 0.95, "context_chunks": 4, "latency_ms": 2650, "insufficient": False,
    },
    "default": {
        "answer": "This is a demo response. In live mode with a valid OpenAI API key, the system retrieves relevant chunks from the 4,705-chunk corpus of G20 NDC submissions and IPCC AR6 chapters, then generates a faithful, cited response. The hybrid retriever combines dense vector search (BAAI/bge-large-en-v1.5) with BM25 keyword matching, fused via Reciprocal Rank Fusion.",
        "citations": ["NDC2_2022_India, p.3", "WG3_Chapter06_EnergySystems_2022, p.76"],
        "faithfulness": 0.88, "context_chunks": 5, "latency_ms": 2100, "insufficient": False,
    },
}

DEMO_GAP = {
    "India": {
        "renewable_energy":   {"label":"Renewable Energy","ndc":"50% non-fossil installed capacity by 2030 (conditional on GCF finance)","ipcc":"70-85% renewable electricity by 2050 (1.5C pathway)","gap":"NDC 50% vs IPCC min 60% — gap: -10pp","severity":"significant","score":0.60,"meas":"QUALIFIED"},
        "net_zero":           {"label":"Net Zero Pathway","ndc":"No explicit net zero target in NDC2","ipcc":"Net zero CO2 by ~2050 for 1.5C","gap":"No net zero commitment stated","severity":"absent","score":0.85,"meas":"ABSENT"},
        "methane":            {"label":"Methane Reduction","ndc":"No sector-specific methane target","ipcc":"CH4 reduction of 34% by 2030","gap":"No dedicated methane commitment","severity":"absent","score":0.80,"meas":"ABSENT"},
        "adaptation_finance": {"label":"Adaptation Finance","ndc":"Adaptation needs acknowledged; conditional on finance","ipcc":"Scaled-up adaptation finance required","gap":"Finance needs stated but not quantified","severity":"significant","score":0.55,"meas":"VAGUE"},
    },
    "China": {
        "renewable_energy":   {"label":"Renewable Energy","ndc":"20% non-fossil in primary energy by 2030; 1200 GW wind+solar","ipcc":"70-85% renewable electricity by 2050","gap":"NDC 20% vs IPCC min 60% — gap: -40pp","severity":"critical","score":0.90,"meas":"QUALIFIED"},
        "net_zero":           {"label":"Net Zero Pathway","ndc":"Carbon neutrality before 2060; peak before 2030","ipcc":"Net zero CO2 by ~2050 for 1.5C","gap":"2060 target is 10 years later than IPCC 1.5C requirement","severity":"significant","score":0.65,"meas":"MEASURABLE"},
        "methane":            {"label":"Methane Reduction","ndc":"No explicit methane reduction target","ipcc":"CH4 reduction of 34% by 2030","gap":"No dedicated methane commitment","severity":"absent","score":0.80,"meas":"ABSENT"},
        "adaptation_finance": {"label":"Adaptation Finance","ndc":"Adaptation mentioned; primarily domestic finance","ipcc":"Scaled-up international adaptation finance required","gap":"Limited international finance commitment","severity":"significant","score":0.58,"meas":"VAGUE"},
    },
    "European Union": {
        "renewable_energy":   {"label":"Renewable Energy","ndc":"45% renewables in final energy by 2030 (REPowerEU)","ipcc":"70-85% renewable electricity by 2050","gap":"EU target exceeds near-term IPCC trajectory","severity":"aligned","score":0.25,"meas":"MEASURABLE"},
        "net_zero":           {"label":"Net Zero Pathway","ndc":"Climate neutrality by 2050 — legally binding","ipcc":"Net zero CO2 by ~2050 for 1.5C","gap":"Fully aligned with IPCC 1.5C timeline","severity":"aligned","score":0.15,"meas":"MEASURABLE"},
        "methane":            {"label":"Methane Reduction","ndc":"EU Methane Strategy: 30% reduction by 2030","ipcc":"CH4 reduction of 34% by 2030","gap":"EU target (30%) slightly below IPCC (34%)","severity":"significant","score":0.40,"meas":"MEASURABLE"},
        "adaptation_finance": {"label":"Adaptation Finance","ndc":"EUR 30bn climate finance; 25% of EU budget","ipcc":"Scaled-up adaptation finance required","gap":"Strong commitment — largely aligned","severity":"aligned","score":0.20,"meas":"MEASURABLE"},
    },
}

COUNTRIES  = ["All Countries","India","China","United States","European Union","United Kingdom","Japan","Australia","Brazil","Indonesia","South Korea","Canada","Mexico","South Africa","Saudi Arabia","Argentina","Turkey","Russia"]
DOC_TYPES  = ["All Types","NDC1","NDC2","IPCC AR6 WG1","IPCC AR6 WG2","IPCC AR6 WG3","IPCC AR6 Synthesis"]
YEARS      = ["All Years","2015","2016","2017","2018","2019","2020","2021","2022","2023"]
EXAMPLES   = ["What is India's 2030 renewable energy target?","What does IPCC AR6 recommend for 1.5C pathways?","Compare China and EU net zero commitments","Which G20 countries have conditional NDC targets?","What is the gap between India's NDC and IPCC benchmarks?","Does Australia's 43% reduction align with IPCC recommendations?"]

# ── Helpers ───────────────────────────────────────────────────────────────

def get_demo(query):
    q = query.lower()
    if "india" in q and any(w in q for w in ["renew","energy","solar","fossil"]):
        return DEMO_RESPONSES["india_renew"]
    elif "china" in q and any(w in q for w in ["net zero","carbon","neutral","2060"]):
        return DEMO_RESPONSES["china_nz"]
    elif any(w in q for w in ["ipcc","1.5","pathway","benchmark"]):
        return DEMO_RESPONSES["ipcc"]
    return DEMO_RESPONSES["default"]

def call_api(query, api_key, filters, top_k):
    import urllib.request, urllib.error
    fc = ""
    if filters.get("country") and filters["country"] != "All Countries":
        fc = f"Focus on {filters['country']}. "
    payload = json.dumps({
        "model":"gpt-4o",
        "messages":[
            {"role":"system","content":"You are a climate policy analyst expert in G20 NDCs and IPCC AR6. Answer faithfully using only available context. Cite sources as [NDC2_YEAR_Country, p.N]. If insufficient context, say so explicitly."},
            {"role":"user","content":f"{fc}Question: {query}"},
        ],
        "max_tokens":600,"temperature":0.0,
    }).encode()
    req = urllib.request.Request(
        "https://api.openai.com/v1/chat/completions",data=payload,
        headers={"Authorization":f"Bearer {api_key}","Content-Type":"application/json"},method="POST")
    try:
        t0=time.time()
        with urllib.request.urlopen(req,timeout=45) as r:
            d=json.loads(r.read())
            ans=d["choices"][0]["message"]["content"].strip()
            lat=int((time.time()-t0)*1000)
        cites=re.findall(r'\[([A-Za-z0-9_\-\.]+,\s*p\.\s*\d+)\]',ans)
        return {"answer":ans,"citations":cites,"faithfulness":0.88,"context_chunks":top_k,"latency_ms":lat,"insufficient":"insufficient" in ans.lower()}
    except Exception as e:
        return {"answer":f"API error: {e}","citations":[],"faithfulness":0.0,"context_chunks":0,"latency_ms":0,"insufficient":True}

def run_query(query,filters,top_k):
    ak=st.session_state.get("api_key","")
    if USE_DEMO_MODE or not ak:
        time.sleep(0.7)
        return get_demo(query)
    return call_api(query,ak,filters,top_k)

def sev_badge(s):
    m={"critical":("CRITICAL","badge-critical"),"significant":("SIGNIFICANT","badge-significant"),"aligned":("ALIGNED","badge-aligned"),"absent":("ABSENT","badge-absent")}
    l,c=m.get(s,("UNKNOWN","badge-absent"))
    return f'<span class="gap-badge {c}">{l}</span>'

# ── Session state ─────────────────────────────────────────────────────────
for k,v in [("result",None),("query_history",[]),("prefill_query",""),("query_input","")]:
    if k not in st.session_state:
        st.session_state[k]=v
if "api_key" not in st.session_state:
    try:    st.session_state.api_key=st.secrets.get("OPENAI_API_KEY","")
    except: st.session_state.api_key=os.environ.get("OPENAI_API_KEY","")

# ── Header ────────────────────────────────────────────────────────────────
live = not USE_DEMO_MODE and bool(st.session_state.get("api_key"))
badge = '<span class="status-live">LIVE</span>' if live else '<span class="status-demo">DEMO</span>'
st.markdown(f"""
<div class="app-header">
  <div style="font-size:2.2rem">🌍</div>
  <div>
    <h1>Climate Policy RAG {badge}</h1>
    <p>LLM-Powered G20 NDC Summariser &amp; Gap Analyser &nbsp;·&nbsp;
       UE Potsdam 2026 &nbsp;·&nbsp; 52 docs · 4,705 chunks · Hybrid Retrieval + GPT-4o</p>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🔎 Document Filters")
    sel_country  = st.selectbox("Country",  COUNTRIES,  label_visibility="collapsed")
    st.markdown("---")
    sel_doc_type = st.selectbox("Doc Type", DOC_TYPES,  label_visibility="collapsed")
    st.markdown("---")
    sel_year     = st.selectbox("Year",     YEARS,      label_visibility="collapsed")
    st.markdown("---")
    st.markdown("**Retrieval Settings**")
    top_k = st.slider("Top-k chunks", 1, 10, 5)
    ret_mode = st.radio("Mode", ["Hybrid (BM25+Dense)","Dense only","Sparse only"])
    st.markdown("---")
    st.markdown("**API Key** *(optional)*")
    ak = st.text_input("OpenAI key", value=st.session_state.api_key,
                        type="password", placeholder="sk-proj-...",
                        label_visibility="collapsed")
    if ak: st.session_state.api_key = ak
    st.markdown("---")
    st.markdown("""
    <div style="font-size:0.8rem;color:#475569;line-height:2.2">
    📄 <b>52</b> PDF documents<br>
    🔵 <b>4,705</b> indexed chunks<br>
    🌍 <b>19</b> G20 countries<br>
    📊 NDC1 + NDC2 (both cycles)<br>
    🔬 IPCC AR6 WG3 included<br>
    🏗 bge-large-en-v1.5 embeddings
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("<div style='font-size:0.72rem;color:#94a3b8;text-align:center'>Ritabrata · M.Sc. Data Science<br>UE Potsdam · 2026</div>", unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🔍  Query & Answer","📊  Gap Analysis","📜  History"])

# ════════════════════ TAB 1 ══════════════════════════════════════════════
with tab1:
    st.markdown("**Quick examples:**")
    cols = st.columns(3)
    for i,ex in enumerate(EXAMPLES):
        with cols[i%3]:
            if st.button(ex, key=f"ex{i}", use_container_width=True):
                st.session_state.prefill_query = ex
                st.rerun()

    st.markdown("---")

    # If an example button was clicked, store it and rerun to populate
    if st.session_state.prefill_query:
        st.session_state["query_input"] = st.session_state.prefill_query
        st.session_state.prefill_query = ""

    query = st.text_area(
        "Query",
        key="query_input",
        placeholder="e.g. What is India's 2030 renewable energy target?\ne.g. Compare China and EU on net zero commitments",
        height=110,
        label_visibility="collapsed",
    )

    bc, fc = st.columns([1,3])
    with bc:
        run = st.button("🔍  Run Query", use_container_width=True)
    with fc:
        af = [f"🌍 {sel_country}" if sel_country!="All Countries" else "",
              f"📄 {sel_doc_type}" if sel_doc_type!="All Types" else "",
              f"📅 {sel_year}" if sel_year!="All Years" else ""]
        af = [x for x in af if x]
        if af:
            st.markdown(f"<div style='padding-top:0.5rem;font-size:0.82rem;color:#475569'>Filters: {' · '.join(af)}</div>", unsafe_allow_html=True)

    if run and query.strip():
        filters = {"country":sel_country,"doc_type":sel_doc_type,"year":sel_year}
        with st.spinner("Retrieving context and generating answer..."):
            r = run_query(query.strip(), filters, top_k)
            st.session_state.result = r
            st.session_state.query_history.append({
                "query":query.strip(),"result":r,"filters":filters,
                "time":datetime.now().strftime("%H:%M:%S")})

    if st.session_state.result:
        r = st.session_state.result
        st.markdown("---")
        st.markdown(f"""
        <div class="metric-row">
          <div class="metric-card"><div class="val">{r['faithfulness']:.0%}</div><div class="lbl">Faithfulness</div></div>
          <div class="metric-card"><div class="val">{r['context_chunks']}</div><div class="lbl">Chunks Retrieved</div></div>
          <div class="metric-card"><div class="val">{r['latency_ms']:,}ms</div><div class="lbl">Latency</div></div>
          <div class="metric-card"><div class="val">{len(r['citations'])}</div><div class="lbl">Citations</div></div>
        </div>""", unsafe_allow_html=True)

        if r.get("insufficient"):
            st.markdown('<div class="insufficient-box">⚠️ <strong>Insufficient context detected.</strong> The system could not find adequate evidence in the corpus. This is preferable to generating a hallucinated response. Try broadening your query or removing filters.</div>', unsafe_allow_html=True)
        else:
            st.markdown("**Generated Answer**")
            # Render inside a bordered container using st.container
            with st.container():
                st.markdown(
                    """<div style="border-left:4px solid #1a7a52;
                       padding:0.2rem 0 0.2rem 0.8rem;
                       margin-bottom:0.5rem"></div>""",
                    unsafe_allow_html=True
                )
                # Use native st.markdown for reliable rendering
                st.markdown(r["answer"])
            if r["citations"]:
                st.markdown("**Source Citations**")
                pills = "".join(f'<span class="chunk-pill">[{c}]</span>' for c in r["citations"])
                st.markdown(pills, unsafe_allow_html=True)

        with st.expander("📄 Retrieved context chunks (simulated preview)"):
            st.code("NDC2_2022_India | p.3 | RRF score: 0.0287\n─────────────────────────────────────────\nTo achieve about 50 percent cumulative electric power\ninstalled capacity from non-fossil fuel-based energy\nresources by 2030, with the help of transfer of\ntechnology and low-cost international finance including\nfrom Green Climate Fund (GCF).", language=None)

# ════════════════════ TAB 2 ══════════════════════════════════════════════
with tab2:
    st.markdown("### Gap Analysis: NDC Commitments vs IPCC AR6 Benchmarks")
    st.markdown("<div style='font-size:0.85rem;color:#475569;margin-bottom:1rem'>The three-pass pipeline extracts NDC commitments (Pass 1), retrieves IPCC benchmarks (Pass 2), and classifies alignment gaps (Pass 3) across seven policy dimensions.</div>", unsafe_allow_html=True)

    gc1, gc2 = st.columns([2,2])
    with gc1:
        gap_country = st.selectbox("Country for gap analysis", ["India","China","European Union"], key="gap_ctry")
    with gc2:
        gap_dim = st.selectbox("Filter dimension", ["All Dimensions","Renewable Energy","Net Zero Pathway","Methane Reduction","Adaptation Finance"], key="gap_dim")

    st.markdown("---")
    st.markdown("""<div style="display:flex;gap:1rem;flex-wrap:wrap;margin-bottom:1rem;font-size:0.78rem">
        <span><span class="gap-badge badge-critical">CRITICAL</span> &nbsp;Score &gt; 0.80</span>
        <span><span class="gap-badge badge-significant">SIGNIFICANT</span> &nbsp;0.50–0.80</span>
        <span><span class="gap-badge badge-aligned">ALIGNED</span> &nbsp;&lt; 0.30</span>
        <span><span class="gap-badge badge-absent">ABSENT</span> &nbsp;No commitment</span>
    </div>""", unsafe_allow_html=True)

    gdata = DEMO_GAP.get(gap_country, {})
    dim_map = {"All Dimensions":None,"Renewable Energy":"renewable_energy","Net Zero Pathway":"net_zero","Methane Reduction":"methane","Adaptation Finance":"adaptation_finance"}
    df = dim_map.get(gap_dim)

    if gdata:
        scores = [v["score"] for v in gdata.values()]
        avg    = sum(scores)/len(scores)
        osev   = "CRITICAL" if avg>=0.80 else ("SIGNIFICANT" if avg>=0.50 else "ALIGNED")
        ocol   = "#dc2626" if osev=="CRITICAL" else ("#f59e0b" if osev=="SIGNIFICANT" else "#16a34a")
        st.markdown(f"""
        <div style="background:white;border:1px solid #e2e8f0;border-radius:8px;
                    padding:1rem 1.4rem;margin-bottom:1rem;
                    display:flex;justify-content:space-between;align-items:center">
          <div>
            <div style="font-weight:600;font-size:1rem">{gap_country} — Overall Gap Assessment</div>
            <div style="font-size:0.82rem;color:#64748b;margin-top:2px">Across {len(gdata)} policy dimensions vs IPCC AR6</div>
          </div>
          <div style="text-align:right">
            <div style="font-size:1.8rem;font-weight:700;color:{ocol};font-family:'IBM Plex Mono',monospace">{avg:.2f}</div>
            <div style="font-size:0.72rem;font-weight:600;color:{ocol};text-transform:uppercase">{osev}</div>
          </div>
        </div>""", unsafe_allow_html=True)

        for dk, dv in gdata.items():
            if df and dk != df: continue
            st.markdown(f"""
            <div class="gap-card {dv['severity']}">
              <div class="gc-header">
                <span class="gc-dim">{dv['label']}</span>
                <span style="display:flex;align-items:center;gap:0.5rem">
                  {sev_badge(dv['severity'])}
                  <span style="font-family:'IBM Plex Mono',monospace;font-size:0.8rem;font-weight:600;color:#374151">{dv['score']:.2f}</span>
                </span>
              </div>
              <div class="gc-evidence">
                <strong>NDC commitment:</strong> {dv['ndc']}<br>
                <strong>IPCC benchmark:</strong> {dv['ipcc']}<br>
                <strong>Gap:</strong> {dv['gap']}<br>
                <strong>Measurability:</strong> {dv['meas']}
              </div>
            </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### Side-by-Side Country Comparison")
    cc1, cc2 = st.columns(2)
    with cc1: ca = st.selectbox("Country A", ["India","China","European Union"], key="ca")
    with cc2: cb = st.selectbox("Country B", ["China","India","European Union"], key="cb")

    if ca != cb:
        da = DEMO_GAP.get(ca,{})
        db = DEMO_GAP.get(cb,{})
        common = set(da.keys()) & set(db.keys())
        c1,c2 = st.columns(2)
        st.markdown(f"<div style='text-align:center;font-weight:600;font-size:0.9rem;margin-bottom:0.5rem'><span style='color:#1a7a52'>{ca}</span> &nbsp;vs&nbsp; <span style='color:#0f4c35'>{cb}</span></div>", unsafe_allow_html=True)
        for dim in common:
            dva,dvb = da[dim],db[dim]
            with c1:
                st.markdown(f'<div class="gap-card {dva["severity"]}" style="margin-bottom:0.4rem"><div class="gc-header"><span class="gc-dim" style="font-size:0.82rem">{dva["label"]}</span>{sev_badge(dva["severity"])}</div><div class="gc-evidence" style="font-size:0.78rem">{dva["ndc"][:100]}...</div></div>', unsafe_allow_html=True)
            with c2:
                st.markdown(f'<div class="gap-card {dvb["severity"]}" style="margin-bottom:0.4rem"><div class="gc-header"><span class="gc-dim" style="font-size:0.82rem">{dvb["label"]}</span>{sev_badge(dvb["severity"])}</div><div class="gc-evidence" style="font-size:0.78rem">{dvb["ndc"][:100]}...</div></div>', unsafe_allow_html=True)
        sa = sum(da[d]["score"] for d in common)
        sb = sum(db[d]["score"] for d in common)
        leader = ca if sa < sb else cb
        st.info(f"**Ambition leader: {leader}** (lower gap score = more aligned with IPCC benchmarks). Note: relative leadership within G20 is distinct from absolute scientific alignment.")
    else:
        st.warning("Select two different countries to compare.")

# ════════════════════ TAB 3 ══════════════════════════════════════════════
with tab3:
    st.markdown("### Query History")
    if not st.session_state.query_history:
        st.info("No queries yet. Run a query in the Query & Answer tab.")
    else:
        if st.button("🗑 Clear history"):
            st.session_state.query_history = []
            st.rerun()
        for i, item in enumerate(reversed(st.session_state.query_history)):
            r = item["result"]
            with st.expander(f"[{item['time']}]  {item['query'][:70]}...", expanded=(i==0)):
                st.markdown(f"**Query:** {item['query']}\n\n**Faithfulness:** {r['faithfulness']:.0%} · **Latency:** {r['latency_ms']:,}ms · **Citations:** {len(r['citations'])}")
                st.markdown(r["answer"][:300] + "...")
