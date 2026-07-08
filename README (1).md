# LLM-Powered Climate Policy Summariser & Gap Analyser

<div align="center">

![Python](https://img.shields.io/badge/Python-3.13-blue?logo=python)
![ChromaDB](https://img.shields.io/badge/ChromaDB-0.4.x-purple)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)
![University](https://img.shields.io/badge/UE-Potsdam-red)

**M.Sc. Data Science Thesis — University of Europe for Applied Sciences, Potsdam**

*Author: Ritabrata Chakraborty · Matriculation: 23039384*
*Supervisors: Prof. Dr. Iftikhar Ahmed · Prof. Shan Faiz*
*Submission: July 19, 2026*

[Live Demo](https://climatepolicyrag-lqlht8mfv5cxzmywwgcaqf.streamlit.app) · [Thesis PDF](#) · [Evaluation Dataset](#data)

</div>

---

## Overview

This repository contains the complete implementation of an end-to-end
**Retrieval-Augmented Generation (RAG)** system for climate policy
analysis. The system ingests G20 Nationally Determined Contributions
(NDCs) and IPCC Sixth Assessment Report (AR6) chapters, and provides:

- **Faithful, cited answers** to natural language climate policy queries
- **Automated gap detection** comparing national commitments against
  IPCC AR6 scientific benchmarks across seven policy dimensions
- **G20-wide analysis** — the first systematic RAG evaluation on all
  17 G20 NDC submissions simultaneously

### Key Results

| Metric | Score |
|--------|-------|
| RAGAS Faithfulness | **0.8853** |
| BERTScore F1 | 0.7792 |
| ROUGE-1 F1 | 0.2799 |
| Dense Retrieval MRR | 0.5640 |
| Hybrid Retrieval HR@10 | 0.8375 |
| LLaMA-3.1-8B Hallucination Rate | **0.2169** (lowest of 3 models) |

> **Headline finding:** Not a single G20 country is fully aligned with
> IPCC AR6 benchmarks on any of the five completed policy dimensions.
> Methane Reduction shows the largest gap (10/17 countries CRITICAL,
> average severity score 0.806).

---

## Table of Contents

- [System Architecture](#system-architecture)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Corpus](#corpus)
- [Pipeline Components](#pipeline-components)
- [Evaluation](#evaluation)
- [Gap Analysis](#gap-analysis)
- [Streamlit Demo](#streamlit-demo)
- [Results](#results)
- [Citation](#citation)

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    CORPUS (52 PDFs)                         │
│         G20 NDC1 + NDC2  ·  IPCC AR6 WG III                │
└──────────────────────┬──────────────────────────────────────┘
                       │ PDF Parsing · Chunking (512 tok, 20% overlap)
                       ▼
┌─────────────────────────────────────────────────────────────┐
│              VECTOR STORE (ChromaDB)                        │
│         4,705 chunks · BAAI/bge-large-en-v1.5               │
│         1,024-dim embeddings · metadata-enriched            │
└──────────┬──────────────────────────┬───────────────────────┘
           │ Dense (cosine)           │ Sparse (BM25Okapi)
           ▼                          ▼
┌──────────────────────────────────────────────────────────────┐
│           HYBRID RETRIEVER  (Reciprocal Rank Fusion k=60)    │
│           Top-5 chunks · optional metadata filter            │
└─────────────────────────────┬────────────────────────────────┘
                              │ Retrieved context
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                  GENERATION PIPELINE                        │
│   GPT-4o · Mistral-7B · LLaMA-3.1-8B  (switchable)        │
│   Grounding constraint · citation injection                 │
│   Insufficient-context detection                            │
└──────────────┬──────────────────────┬───────────────────────┘
               │ Standard QA           │ Gap Analysis queries
               ▼                       ▼
         Cited Answer          ┌───────────────────────┐
                               │  THREE-PASS GAP MODULE │
                               │  Pass 1: NDC extraction│
                               │  Pass 2: IPCC benchmark│
                               │  Pass 3: Gap classify  │
                               └───────────────────────┘
```

---

## Repository Structure

```
D:\Thesis\
│
├── README.md
├── requirements.txt
├── app.py                          # Streamlit demo application
│
├── Gap_Analysis_Code\
│   ├── gap_analysis.py             # Three-pass gap analysis module
│   ├── generation.py               # Multi-backend LLM pipeline
│   ├── retriever.py                # Hybrid retriever (Dense + BM25 + RRF)
│   └── run_transport_decarbonisation.py
│
├── Evaluation_Code\
│   ├── bootstrap_ci.py             # Paired bootstrap CI (RQ3)
│   ├── kappa_calculator.py         # Cohen's Kappa inter-annotator
│   └── rq3_comparison.py           # Model comparison evaluation
│
├── data\
│   ├── processed\
│   │   ├── chunks.jsonl            # 4,705 parsed chunks
│   │   └── bm25_index.pkl          # Cached BM25 index
│   ├── vectorstore\                # ChromaDB persistent store
│   │   └── climate_policy_rag      # collection (4,705 vectors)
│   ├── eval\
│   │   ├── golden_dataset_80.csv   # 80-question evaluation dataset
│   │   └── bootstrap_ci_results.json
│   └── gap_reports\
│       ├── single\                 # Per-country per-dimension JSONs
│       ├── g20_summary_final.csv   # 17×7 severity matrix
│       └── g20_score_final.csv     # 17×7 numeric score matrix
│
├── figures\
│   ├── g20_heatmap_final.png       # G20 gap analysis heatmap (300 dpi)
│   └── g20_heatmap_final.pdf
│
└── annotator_review_workbook.xlsx  # Inter-annotator validation workbook
```

---

## Installation

### Prerequisites

- Python 3.11 or higher
- Windows / Linux / macOS
- No GPU required — all components run on CPU

### 1. Clone the Repository

```bash
git clone https://github.com/RudraYug/Climate_Policy_RAG.git
cd Climate_Policy_RAG
```

### 2. Create a Virtual Environment

```bash
python -m venv thesis_env

# Windows
thesis_env\Scripts\activate

# Linux / macOS
source thesis_env/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Set API Keys

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=sk-...        # For GPT-4o backend
GROQ_API_KEY=gsk_...         # For LLaMA-3.1-8B (free tier)
MISTRAL_API_KEY=...          # For Mistral-7B backend
```

Or set them as environment variables:

```powershell
# Windows PowerShell
$env:OPENAI_API_KEY = "sk-..."
$env:GROQ_API_KEY   = "gsk_..."
```

---

## Quick Start

### Run the Streamlit Demo

```bash
streamlit run app.py
```

Then open `http://localhost:8501` in your browser.

### Ask a Climate Policy Question (Python)

```python
from Gap_Analysis_Code.generation import RAGPipeline

pipeline = RAGPipeline(backend="groq")  # free, no API cost

answer = pipeline.query(
    "What is India's 2030 renewable energy target?",
    country="India"
)
print(answer["answer"])
print(answer["citations"])
```

### Run a Gap Analysis

```python
from Gap_Analysis_Code.gap_analysis import GapAnalyser

analyser = GapAnalyser(backend="groq")

result = analyser.analyse(
    countries=["India", "China"],
    dimension="renewable_energy",
    ndc_cycles=[2]
)
print(result)
```

---

## Corpus

The system operates on a curated corpus of **52 PDF documents**:

| Source | Documents | Description |
|--------|-----------|-------------|
| UNFCCC NDC Registry | 38 | NDC1 + NDC2 for all 17 G20 nations |
| IPCC AR6 WG III | 14 | Mitigation chapters + SPM + Technical Summary |
| **Total** | **52** | **4,705 chunks · avg 487 tokens/chunk** |

### G20 Countries Covered

Argentina · Australia · Brazil · Canada · China · EU · India ·
Indonesia · Japan · Mexico · Russia · Saudi Arabia · South Africa ·
South Korea · Turkey · UK · USA

### Document Metadata Schema

Each chunk carries six metadata fields used for filtered retrieval:

```json
{
  "country":   "India",
  "source":    "NDC2",
  "ndc_cycle": 2,
  "doc_id":    "NDC2_2022_India",
  "year":      2022,
  "language":  "en"
}
```

---

## Pipeline Components

### Hybrid Retriever (`retriever.py`)

Combines dense and sparse retrieval via Reciprocal Rank Fusion:

```
Dense  (BAAI/bge-large-en-v1.5, cosine similarity)
   +
Sparse (BM25Okapi, k1=1.6, b=0.75)
   ↓
RRF fusion (k=60)
   ↓
Top-5 chunks
```

| Metric | Dense | Sparse | Hybrid |
|--------|-------|--------|--------|
| MRR | **0.5640** | 0.4326 | 0.5247 |
| HR@5 | **0.7375** | 0.6250 | 0.7250 |
| HR@10 | **0.8375** | 0.7000 | **0.8375** |

Dense wins on early precision; Hybrid matches Dense at HR@10 and provides recall robustness for domain-specific terms (LULUCF, MtCO2e, GWP100).

### Generation Pipeline (`generation.py`)

Supports three switchable backends under identical prompts:

| Backend | Model | Cost/query | Hallucination Rate |
|---------|-------|-----------|-------------------|
| GPT-4o | gpt-4o | $0.02354 | 0.2833 |
| Mistral | mistral-small | $0.01135 | 0.3721 |
| LLaMA | llama-3.1-8b-instant (Groq) | **$0.000** | **0.2169** |

Key prompt features:
- Explicit grounding constraint: answer only from retrieved context
- Inline citations in `[doc_id, p.N]` format
- Structured insufficient-context declaration instead of hallucination

### Three-Pass Gap Analysis (`gap_analysis.py`)

```
Pass 1 ── NDC document retrieval
          Extract: commitment value, target year, baseline year,
                   conditionality, metric used

Pass 2 ── IPCC AR6 WG III retrieval
          Extract: scientific benchmark, confidence level,
                   temperature pathway (1.5°C or 2°C)

Pass 3 ── Comparison and classification
          Output: severity score (0–1), gap type, evidence citations,
                  policy recommendation
```

**Seven policy dimensions:**
1. Renewable Energy Targets
2. Net Zero / Carbon Neutrality Pathway
3. Methane Reduction
4. Adaptation Finance
5. Loss and Damage
6. Transport Decarbonisation
7. LULUCF and Carbon Sinks

**Severity scale:**

| Label | Score Range | Meaning |
|-------|------------|---------|
| ALIGNED | 0.00–0.29 | Meets IPCC AR6 pathway |
| PARTIAL | 0.30–0.49 | Partially meets benchmark |
| SIGNIFICANT | 0.50–0.79 | Notable gap vs IPCC |
| CRITICAL | 0.80–1.00 | Major gap vs IPCC |

---

## Evaluation

### Golden Evaluation Dataset

80 expert-annotated question–answer pairs stratified across:

| Category | N | Description |
|----------|---|-------------|
| Factual | 30 | Single numeric commitment from one NDC |
| Comparative | 25 | Multi-country comparison |
| Gap Detection | 25 | NDC vs IPCC AR6 alignment |

| Difficulty | N | Criteria |
|------------|---|---------- |
| Easy | 12 | Single doc, direct lookup |
| Medium | 14 | Multi-sentence or two-doc synthesis |
| Hard | 54 | Multi-doc, cross-country reasoning |

### End-to-End Results (GPT-4o, n=80)

| Metric | Score | Interpretation |
|--------|-------|---------------|
| RAGAS Faithfulness | **0.8853** | Excellent |
| RAGAS Context Precision | 0.6562 | Moderate |
| RAGAS Context Recall | 0.5085 | Moderate |
| ROUGE-1 F1 | 0.2799 | Expected for generative QA |
| BERTScore F1 | **0.7792** | High semantic similarity |
| Insufficient context rate | 16.25% | Correctly declined vs hallucinating |

### Failure Mode Taxonomy

| Mode | N | % | Root Cause |
|------|---|---|-----------|
| No failure | 40 | 50.0% | — |
| FM7 Corpus Gap | 14 | 17.5% | Content absent from corpus |
| FM6 Metric Mismatch | 15 | 18.8% | Metric artefact, not system failure |
| FM5 LLM Vagueness | 7 | 8.8% | Generation failure |
| FM1 Retrieval Miss | 4 | 5.0% | Retrieval failure |

### Statistical Analysis

Paired bootstrap resampling (10,000 iterations) over 80 questions:

| Model | Hallucination Rate | 95% CI |
|-------|-------------------|--------|
| GPT-4o | 0.275 | [0.175, 0.375] |
| Mistral-7B | 0.362 | [0.263, 0.463] |
| LLaMA-3.1-8B | **0.175** | [0.100, 0.263] |

Paired bootstrap CI on GPT-4o − LLaMA difference: [−0.0375, 0.2375]
(p = 0.170). Directionally consistent; not significant at α=0.05 with
n=80. Expanding to n≥200 would provide sufficient statistical power.

> **Note:** All hallucination labels are judge-assisted (GPT-4o-mini)
> and have not been independently validated by human domain experts.

---

## Gap Analysis

### G20-Wide Results (5 of 7 dimensions complete)

![G20 Heatmap](figures/g20_heatmap_final.png)

| Dimension | CRITICAL | SIGNIFICANT | ALIGNED | Avg Score |
|-----------|----------|-------------|---------|-----------|
| Renewable Energy | 1/17 | 16/17 | 0 | 0.665 |
| Net Zero Pathway | 5/17 | 12/17 | 0 | 0.732 |
| Methane Reduction | **10/17** | 7/17 | 0 | **0.806** |
| Adaptation Finance | 6/17 | 11/17 | 0 | 0.721 |
| LULUCF & Forestry | 5/17 | 12/17 | 0 | 0.765 |
| Loss & Damage | pending | — | — | — |
| Transport Decarb. | pending | — | — | — |

**No G20 country is fully aligned with IPCC AR6 on any completed
dimension.** Methane Reduction is the most critical gap: 59% of G20
countries show CRITICAL severity.

### Illustrative Output — India Renewable Energy

```json
{
  "country": "India",
  "dimension": "renewable_energy",
  "overall_severity": "SIGNIFICANT",
  "overall_severity_score": 0.60,
  "gap_findings": [
    {
      "gap_type": "IPCC_CONTRADICTION",
      "description": "NDC target of 50% installed capacity by 2030 is
                      below IPCC minimum of 60-70% for 2°C pathway.",
      "ndc_evidence": "50% cumulative electric power from non-fossil
                       fuels by 2030 [NDC2_2022_India, p.3]",
      "ipcc_evidence": "Renewable electricity 60-70% by 2050 for 2°C
                        [WG3_Chapter06_EnergySystems_2022, p.76]",
      "quantified_gap": "-10 percentage points vs IPCC minimum",
      "severity_score": 0.60
    },
    {
      "gap_type": "CONDITIONAL_DEPENDENCY",
      "description": "Target contingent on GCF finance — delivery risk.",
      "severity_score": 0.60
    }
  ]
}
```

---

## Streamlit Demo

The live demo at
[climatepolicyrag-lqlht8mfv5cxzmywwgcaqf.streamlit.app](https://climatepolicyrag-lqlht8mfv5cxzmywwgcaqf.streamlit.app)
provides:

- Natural language query interface with country and document filters
- Retrieval mode selection (Hybrid BM25+Dense or Dense only)
- Real-time faithfulness score and latency display
- Source chunk viewer with RRF scores
- Query history panel

**Demo mode** works without an API key (simulated responses).
**Live mode** requires an OpenAI API key set in Streamlit secrets.

---

## Citation

If you use this work, please cite:

```bibtex
@mastersthesis{chakraborty2026llm,
  title   = {LLM-Powered Climate Policy Summariser \& Gap Analyser},
  author  = {Chakraborty, Ritabrata},
  school  = {University of Europe for Applied Sciences},
  address = {Potsdam, Germany},
  year    = {2026},
  month   = {July},
  note    = {M.Sc. Data Science}
}
```

---

## Acknowledgements

- **UNFCCC NDC Registry** for providing G20 NDC documents
- **IPCC** for the Sixth Assessment Report Working Group III chapters
- **Groq** for free-tier LLaMA-3.1-8B inference
- **Streamlit** for the deployment platform
- **ChromaDB**, **BAAI**, and the open-source NLP community

---

## License

This project is licensed under the MIT License.
The evaluation dataset, gap analysis outputs, and thesis are
© 2026 Ritabrata Chakraborty, University of Europe for Applied Sciences.

---

<div align="center">
Made with ❤️ for climate policy transparency
</div>
