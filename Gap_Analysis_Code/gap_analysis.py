"""
===========================================================================
  GAP ANALYSIS MODULE  —  NOVEL THESIS CONTRIBUTION
  Master's Thesis: LLM-Powered Climate Policy Summariser & Gap Analyser
  University of Europe for Applied Sciences, Potsdam  |  2026
===========================================================================

SCIENTIFIC CONTRIBUTION
───────────────────────
  This module operationalises the concept of "NDC gap" as a structured,
  evidence-grounded, LLM-assisted analysis rather than a manual desk review.
  It makes three novel contributions to the climate informatics literature:

  1. AUTOMATED COMMITMENT EXTRACTION
     Extracts both quantified and unquantified commitments from NDC text,
     distinguishing conditional (finance-dependent) from unconditional targets,
     and near-term (2025–2030) from long-term (2035–2050) ambitions.
     Prior work (ClimateWatch, CAT) relies entirely on human analysts.

  2. MEASURABILITY CLASSIFICATION
     Classifies each extracted commitment on a 4-level scale:
       QUANTIFIED  — specific number, year, and baseline (e.g. "43% by 2030 vs 2005")
       QUALIFIED   — direction and year but no baseline (e.g. "increase by 2030")
       VAGUE       — stated intent without timeline (e.g. "will promote renewables")
       ABSENT      — dimension not mentioned in document
     Assigns a continuous measurability score 0.0–1.0 for ranking.

  3. IPCC-GROUNDED GAP DETECTION
     Compares each commitment against the IPCC AR6 WG3 benchmark retrieved
     from the same vector store — not from a hardcoded lookup table.
     Gap severity is classified as CRITICAL / SIGNIFICANT / MINOR / ALIGNED
     with an evidence chain traceable to specific document pages.

PIPELINE OVERVIEW
─────────────────
  Input:  country names + dimension + optional year filter
           ↓
  Stage 1: Retrieve NDC chunks per country (via HybridRetriever)
  Stage 2: Retrieve IPCC AR6 benchmark for dimension (via HybridRetriever)
  Stage 3: Pass A — LLM extracts commitments per country (structured JSON)
  Stage 4: Pass B — LLM classifies measurability per commitment
  Stage 5: Pass C — LLM detects gaps vs IPCC benchmark
  Stage 6: Aggregate country-pair results into GapReport
           ↓
  Output: GapReport dataclass → JSON file → human-readable summary

JSON OUTPUT SCHEMA  (per GapReport)
────────────────────────────────────
  {
    "dimension":       str,            # e.g. "renewable_energy"
    "countries":       list[str],      # e.g. ["India", "China"]
    "ipcc_benchmark":  BenchmarkBlock, # what IPCC recommends
    "country_analyses": {
      "India": CountryAnalysis,
      "China": CountryAnalysis,
    },
    "pairwise_gaps": [
      PairwiseGap,                     # India vs China
    ],
    "g20_summary":     G20Summary,     # if ≥ 3 countries analysed
    "metadata":        dict,
  }

DIMENSIONS SUPPORTED
────────────────────
  renewable_energy          · net_zero_pathway
  adaptation_finance        · loss_and_damage
  lulucf_forestry           · transport_decarbonisation
  methane_reduction

USAGE
──────
  pip install rank_bm25 chromadb sentence-transformers transformers

  from gap_analysis import GapAnalyser

  analyser = GapAnalyser()   # loads retriever + LLM from generation.py

  # Single country analysis
  report = analyser.analyse(
      countries  = ["India", "China"],
      dimension  = "renewable_energy",
      ndc_cycles = [1, 2],            # compare both cycles
  )
  print(report.summary_text)
  report.save()                        # saves JSON to data/gap_reports/

  # Full G20 sweep (all countries, one dimension)
  report = analyser.analyse_g20(dimension="net_zero_pathway")

  # CLI
  python gap_analysis.py --countries India China --dimension renewable_energy
  python gap_analysis.py --countries Indonesia Brazil --dimension adaptation_finance
  python gap_analysis.py --g20 --dimension net_zero_pathway
  python gap_analysis.py --countries UK EU Australia --dimension lulucf_forestry --debug
===========================================================================
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
import warnings
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# ── Silence noisy library output ─────────────────────────────────────────────
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
warnings.filterwarnings("ignore", message=".*position_ids.*")
warnings.filterwarnings("ignore", message=".*unauthenticated.*")
warnings.filterwarnings("ignore", message=".*symlinks.*")


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 1 — DIMENSION TAXONOMY
# ═══════════════════════════════════════════════════════════════════════════════

DIMENSIONS: dict[str, dict] = {
    "renewable_energy": {
        "label":       "Renewable Energy Targets",
        "description": "Share of renewables in electricity mix or installed capacity by target year",
        "ipcc_ref":    "IPCC AR6 WG3 Chapter 6 recommends ~70–85% renewable electricity by 2050 "
                       "for 1.5°C pathways; for 2°C, 60–70% by 2050.",
        "keywords":    ["renewable", "solar", "wind", "clean energy", "non-fossil",
                        "electricity capacity", "energy mix", "power sector"],
        "ipcc_filter": {"$and": [{"source": "IPCC_AR6"}, {"wg": "WG3"}]},
        "ipcc_query":  "renewable energy electricity share target pathway 2030 2050 1.5 degrees",
        "units":       ["% of electricity", "GW installed", "% of energy mix"],
    },
    "net_zero_pathway": {
        "label":       "Net Zero / Carbon Neutrality Pathway",
        "description": "Commitment to reach net zero GHG emissions and the stated year",
        "ipcc_ref":    "IPCC AR6 WG3 SPM: global net zero CO2 by 2050 required for 1.5°C; "
                       "net zero GHG by around 2070 for 2°C.",
        "keywords":    ["net zero", "carbon neutral", "climate neutral", "net negative",
                        "carbon neutrality", "emissions peak", "decarbonisation"],
        "ipcc_filter": {"$and": [{"source": "IPCC_AR6"}, {"wg": "WG3"}]},
        "ipcc_query":  "net zero carbon neutrality pathway year 2050 2060 GHG emissions",
        "units":       ["target year", "% reduction vs baseline"],
    },
    "adaptation_finance": {
        "label":       "Adaptation Finance Commitments",
        "description": "Domestic and international finance pledged for climate adaptation",
        "ipcc_ref":    "IPCC AR6 WG2 Chapter 17: adaptation finance gaps persist; "
                       "developing countries need significantly more than current flows.",
        "keywords":    ["adaptation finance", "climate finance", "adaptation fund",
                        "international support", "conditional", "finance gap"],
        "ipcc_filter": {"$and": [{"source": "IPCC_AR6"}, {"wg": "WG2"}]},
        "ipcc_query":  "adaptation finance investment gap developing countries climate resilience",
        "units":       ["USD billions", "% of GDP", "% of NDC conditional on finance"],
    },
    "loss_and_damage": {
        "label":       "Loss and Damage",
        "description": "Acknowledgement of and response to irreversible climate impacts",
        "ipcc_ref":    "IPCC AR6 WG2 Chapter 17: loss and damage is unavoidable "
                       "even with adaptation; finance mechanisms urgently needed.",
        "keywords":    ["loss and damage", "irreversible", "unavoidable impacts",
                        "Warsaw mechanism", "Santiago network", "displacement"],
        "ipcc_filter": {"$and": [{"source": "IPCC_AR6"}, {"wg": "WG2"}]},
        "ipcc_query":  "loss and damage irreversible climate impacts finance mechanism",
        "units":       ["mentioned", "quantified", "absent"],
    },
    "lulucf_forestry": {
        "label":       "LULUCF and Forestry Sinks",
        "description": "Land use, land-use change and forestry as carbon sink or source",
        "ipcc_ref":    "IPCC AR6 WG3 Ch2: LULUCF can contribute 1–12 GtCO2/yr by 2030; "
                       "permanence and accounting transparency are critical.",
        "keywords":    ["LULUCF", "forestry", "deforestation", "reforestation",
                        "carbon sink", "land use", "afforestation", "forest cover"],
        "ipcc_filter": {"$and": [{"source": "IPCC_AR6"}, {"wg": "WG3"}]},
        "ipcc_query":  "LULUCF land use forestry carbon sink removal sequestration 2030",
        "units":       ["MtCO2e/yr", "million hectares", "% of total NDC target"],
    },
    "transport_decarbonisation": {
        "label":       "Transport Sector Decarbonisation",
        "description": "EV targets, fossil fuel phase-out, modal shift commitments",
        "ipcc_ref":    "IPCC AR6 WG3 Ch10: transport emissions must fall 59–84% by 2050 "
                       "vs 2020 in 1.5°C scenarios. EV share 75–95% of new sales by 2050.",
        "keywords":    ["transport", "electric vehicle", "EV", "fossil fuel vehicle",
                        "modal shift", "public transport", "aviation", "shipping"],
        "ipcc_filter": {"$and": [{"source": "IPCC_AR6"}, {"wg": "WG3"}]},
        "ipcc_query":  "transport decarbonisation electric vehicle emission reduction pathway",
        "units":       ["% EV share", "% emission reduction", "phase-out year"],
    },
    "methane_reduction": {
        "label":       "Methane Reduction Targets",
        "description": "Specific commitments to reduce methane emissions from all sectors",
        "ipcc_ref":    "IPCC AR6 WG3 SPM: reducing methane ~34% by 2030 vs 2020 "
                       "buys near-term warming relief and is cost-effective.",
        "keywords":    ["methane", "CH4", "fugitive emissions", "agriculture methane",
                        "global methane pledge", "livestock", "coal mine methane"],
        "ipcc_filter": {"$and": [{"source": "IPCC_AR6"}, {"wg": "WG3"}]},
        "ipcc_query":  "methane CH4 reduction target 2030 agriculture energy sector",
        "units":       ["% reduction vs baseline", "MtCH4"],
    },
}

# All G20 countries in the corpus
G20_COUNTRIES = [
    "Argentina", "Australia", "Brazil", "Canada", "China",
    "EU", "India", "Indonesia", "Japan", "Mexico",
    "Russia", "Saudi Arabia", "South Africa", "South Korea",
    "Turkey", "UK", "USA",
]

# Measurability levels (ordered worst → best)
MEASURABILITY_LEVELS = ["ABSENT", "VAGUE", "QUALIFIED", "QUANTIFIED"]

# Gap severity levels (ordered best → worst)
SEVERITY_LEVELS = ["ALIGNED", "MINOR", "SIGNIFICANT", "CRITICAL"]


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 2 — OUTPUT DATACLASSES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class CommitmentItem:
    """One extracted commitment from an NDC document."""
    text:            str              # verbatim or paraphrased commitment text
    target_value:    Optional[str]    # e.g. "50%", "500 GW", "net zero"
    target_year:     Optional[int]    # e.g. 2030, 2050
    baseline_year:   Optional[int]    # e.g. 2005, 2020
    conditional:     bool             # True = depends on international finance
    time_horizon:    str              # "near_term" (<2035) | "long_term" (≥2035) | "unclear"
    source_doc:      str              # doc_id
    source_page:     int              # page number
    confidence:      float            # LLM confidence 0.0–1.0


@dataclass
class MeasurabilityScore:
    """Measurability classification for one commitment or country-dimension pair."""
    level:       str    # QUANTIFIED | QUALIFIED | VAGUE | ABSENT
    score:       float  # 0.0 (absent) → 1.0 (fully quantified)
    reason:      str    # one-sentence explanation
    missing_elements: list[str]  # e.g. ["baseline year", "specific unit"]


@dataclass
class GapFinding:
    """One identified gap between NDC commitment and IPCC benchmark."""
    gap_type:         str    # "missing_commitment" | "vague_target" | "ipcc_contradiction" | "conditional_dependency"
    severity:         str    # CRITICAL | SIGNIFICANT | MINOR | ALIGNED
    severity_score:   float  # 0.0 (aligned) → 1.0 (critical)
    description:      str    # human-readable gap description
    ndc_evidence:     str    # what the NDC says (with citation)
    ipcc_evidence:    str    # what IPCC recommends (with citation)
    ndc_citation:     str    # [doc_id, p.N]
    ipcc_citation:    str    # [doc_id, p.N]
    quantified_gap:   Optional[str]  # e.g. "NDC: 40%, IPCC min: 70% → gap: 30pp"
    confidence:       float  # LLM confidence 0.0–1.0


@dataclass
class CountryAnalysis:
    """Complete gap analysis for one country on one dimension."""
    country:          str
    dimension:        str
    ndc_cycles:       list[int]       # which cycles were analysed
    commitments:      list[CommitmentItem]
    measurability:    MeasurabilityScore
    gap_findings:     list[GapFinding]
    overall_severity: str             # worst severity across all findings
    overall_score:    float           # 0.0 aligned → 1.0 critical
    summary:          str             # 2-3 sentence human summary
    raw_llm_outputs:  dict = field(default_factory=dict)  # for auditability


@dataclass
class BenchmarkBlock:
    """IPCC AR6 benchmark for one dimension."""
    dimension:        str
    ipcc_target:      str       # quantified recommendation
    ipcc_context:     str       # why this target matters
    source_chunks:    list[str] # doc_ids of IPCC chunks used
    citations:        list[str] # [doc_id, p.N] list


@dataclass
class PairwiseGap:
    """Comparison between exactly two countries on one dimension."""
    country_a:        str
    country_b:        str
    dimension:        str
    relative_gap:     str       # "Country A is more ambitious than B because..."
    ambition_leader:  str       # which country has stronger commitment
    leadership_reason: str
    convergence_areas: list[str]  # where they agree
    divergence_areas:  list[str]  # where they disagree


@dataclass
class GapReport:
    """
    Top-level output of the Gap Analysis Module.
    One report per (countries, dimension) combination.
    Saved as JSON to data/gap_reports/.
    """
    dimension:         str
    dimension_label:   str
    countries:         list[str]
    ndc_cycles:        list[int]
    ipcc_benchmark:    BenchmarkBlock
    country_analyses:  dict[str, CountryAnalysis]  # country → analysis
    pairwise_gaps:     list[PairwiseGap]
    g20_summary:       Optional[dict]    # populated if ≥3 countries
    rankings:          dict              # countries ranked by severity score
    policy_implications: list[str]      # top-level findings for thesis
    metadata:          dict

    def save(self, out_dir: str = "data/gap_reports") -> Path:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        ts      = int(time.time())
        cnames  = "_".join(self.countries[:3]).replace(" ", "")
        fname   = f"gap_{self.dimension}_{cnames}_{ts}.json"
        path    = Path(out_dir) / fname
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self._to_dict(), f, indent=2, ensure_ascii=False, default=str)
        print(f"  [GapReport] Saved → {path}")
        return path

    def _to_dict(self) -> dict:
        d = {
            "dimension":           self.dimension,
            "dimension_label":     self.dimension_label,
            "countries":           self.countries,
            "ndc_cycles":          self.ndc_cycles,
            "ipcc_benchmark":      asdict(self.ipcc_benchmark),
            "country_analyses":    {
                k: asdict(v) for k, v in self.country_analyses.items()
            },
            "pairwise_gaps":       [asdict(p) for p in self.pairwise_gaps],
            "g20_summary":         self.g20_summary,
            "rankings":            self.rankings,
            "policy_implications": self.policy_implications,
            "metadata":            self.metadata,
        }
        return d

    @property
    def summary_text(self) -> str:
        """Human-readable plain-text summary for thesis Chapter 5."""
        lines = [
            f"GAP ANALYSIS REPORT",
            f"Dimension  : {self.dimension_label}",
            f"Countries  : {', '.join(self.countries)}",
            f"NDC Cycles : {self.ndc_cycles}",
            f"Generated  : {self.metadata.get('generated_at', '')}",
            "",
            "IPCC BENCHMARK:",
            f"  {self.ipcc_benchmark.ipcc_target}",
            "",
        ]
        for country, analysis in self.country_analyses.items():
            lines += [
                f"{'─'*60}",
                f"COUNTRY: {country}",
                f"  Measurability : {analysis.measurability.level} "
                f"(score={analysis.measurability.score:.2f})",
                f"  Overall gap   : {analysis.overall_severity} "
                f"(score={analysis.overall_score:.2f})",
                f"  Summary       : {analysis.summary}",
                "",
                f"  Commitments found ({len(analysis.commitments)}):",
            ]
            for c in analysis.commitments:
                cond = " [CONDITIONAL]" if c.conditional else ""
                lines.append(
                    f"    • {c.target_value or 'no target'} by {c.target_year or '?'}"
                    f"{cond}  [{c.source_doc}, p.{c.source_page}]"
                )
            lines += ["", f"  Gaps found ({len(analysis.gap_findings)}):"]
            for g in analysis.gap_findings:
                lines.append(
                    f"    [{g.severity}] {g.gap_type}:\n"
                    f"      {g.description}\n"
                    f"      NDC  : {g.ndc_evidence[:120]}  {g.ndc_citation}\n"
                    f"      IPCC : {g.ipcc_evidence[:120]}  {g.ipcc_citation}"
                    + (f"\n      Gap  : {g.quantified_gap}" if g.quantified_gap else "")
                )
            lines.append("")

        if self.rankings:
            lines += ["RANKINGS (most → least ambitious):"]
            for i, (country, score) in enumerate(
                sorted(self.rankings.items(), key=lambda x: x[1]), 1
            ):
                lines.append(f"  {i}. {country} (gap score={score:.2f})")
            lines.append("")

        if self.policy_implications:
            lines += ["POLICY IMPLICATIONS:"]
            for impl in self.policy_implications:
                lines.append(f"  • {impl}")

        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 3 — PROMPT TEMPLATES
# ═══════════════════════════════════════════════════════════════════════════════

class GapPrompts:
    """
    All three prompt templates for the gap analysis pipeline.

    Design principles:
    ──────────────────
    Each prompt is:
    (a) Grounded  — model must cite every claim with [doc_id, p.N]
    (b) Structured — model must return valid JSON matching the schema
    (c) Bounded   — model must state INSUFFICIENT_EVIDENCE if data absent
    (d) Calibrated — model must provide a confidence score per finding

    The three-pass design (Extract → Classify → Compare) is deliberate:
    doing all three in one prompt causes the model to conflate extraction
    with evaluation, producing unreliable results. Sequential passes with
    structured JSON handoffs force explicit reasoning at each step.
    """

    # ── PASS A: Commitment Extraction ─────────────────────────────────────────
    PASS_A_SYSTEM = """You are a climate policy analyst extracting commitments from NDC documents.

TASK: Extract ALL commitments related to the specified dimension from the provided NDC text.

STRICT RULES:
1. Extract ONLY from the RETRIEVED CONTEXT. Do not use outside knowledge.
2. For every extracted commitment, cite the source as [doc_id, p.N].
3. Distinguish CONDITIONAL commitments (require international finance/technology) from UNCONDITIONAL ones.
4. Classify time horizon: near_term (target year ≤ 2034) or long_term (≥ 2035) or unclear.
5. If no commitment exists for this dimension, return an empty commitments array — do NOT fabricate.
6. Return ONLY valid JSON matching the schema below. No prose before or after.
7. CRITICAL — Do NOT extract historical baseline figures as commitments.
   A commitment must be a FUTURE target (target_year > 2023).
   Examples of what NOT to extract:
     - "installed 794 GW by 2019"  → historical fact, not a commitment
     - "achieved 40% renewables in 2020" → historical achievement, not a target
     - "current capacity is 500 GW" → current state, not a future goal
   Only extract statements that express a FUTURE GOAL or PLEDGE.

OUTPUT SCHEMA:
{
  "country": "<country name>",
  "dimension": "<dimension key>",
  "commitments": [
    {
      "text": "<verbatim or close paraphrase of commitment>",
      "target_value": "<specific number/percentage/unit or null>",
      "target_year": <integer year or null>,
      "baseline_year": <integer year or null>,
      "conditional": <true|false>,
      "time_horizon": "<near_term|long_term|unclear>",
      "source_doc": "<doc_id from context>",
      "source_page": <integer>,
      "confidence": <float 0.0-1.0>
    }
  ],
  "dimension_mentioned": <true|false>,
  "total_found": <integer>,
  "extraction_notes": "<brief note on coverage or gaps in source text>"
}"""

    PASS_A_USER = """RETRIEVED NDC CONTEXT FOR {country} (NDC Cycle {cycle}):

{ndc_context}

────────────────────────────────────────────────────────────
DIMENSION TO EXTRACT: {dimension_label}
Definition: {dimension_description}
Keywords to look for: {keywords}

Extract all commitments related to this dimension from the context above.
Remember: return only JSON, no prose. If nothing found, return empty commitments array."""

    # ── PASS B: Measurability Classification ──────────────────────────────────
    PASS_B_SYSTEM = """You are a climate policy analyst classifying the measurability of NDC commitments.

MEASURABILITY LEVELS:
  QUANTIFIED  = Has specific number + unit + target year + baseline (e.g. "50% renewable by 2030 vs 2010")
  QUALIFIED   = Has direction + target year but missing baseline or precise unit (e.g. "increase renewable share by 2030")
  VAGUE       = Stated intent without year or measurable target (e.g. "will promote renewable energy")
  ABSENT      = Dimension not mentioned in document at all

RULES:
1. Base your classification ONLY on the extracted commitments provided.
2. Give a measurability score: ABSENT=0.0, VAGUE=0.25, QUALIFIED=0.65, QUANTIFIED=1.0
3. List exactly what elements are missing for a higher score.
4. Return ONLY valid JSON. No prose.

OUTPUT SCHEMA:
{
  "country": "<country>",
  "dimension": "<dimension>",
  "measurability_level": "<QUANTIFIED|QUALIFIED|VAGUE|ABSENT>",
  "measurability_score": <float 0.0-1.0>,
  "reason": "<one sentence explaining the classification>",
  "missing_elements": ["<element1>", "<element2>"],
  "strongest_commitment": "<text of best commitment found, or null>",
  "conditional_dependency": <true|false>,
  "conditional_note": "<what the commitment depends on, or null>"
}"""

    PASS_B_USER = """COUNTRY: {country}
DIMENSION: {dimension_label}

EXTRACTED COMMITMENTS:
{commitments_json}

Classify the overall measurability of {country}'s commitments on {dimension_label}.
Return only JSON."""

    # ── PASS C: Gap Detection (the core novel contribution) ───────────────────
    PASS_C_SYSTEM = """You are a climate policy analyst identifying gaps between NDC commitments and IPCC AR6 scientific recommendations.

YOUR TASK: Compare the country's NDC commitments against the IPCC benchmark and identify gaps.

THREE GAP TYPES TO LOOK FOR:

TYPE 1 — MISSING COMMITMENT
  The country makes no commitment on this dimension, or only mentions it vaguely.
  The IPCC recommends specific action on this dimension.
  → This is a gap by omission.

TYPE 2 — VAGUE vs MEASURABLE
  The country makes a commitment but it lacks quantification, baseline, or timeline.
  The IPCC provides specific benchmarks.
  → This is a measurability gap that makes accountability impossible.

TYPE 3 — IPCC CONTRADICTION
  The country's stated target is directionally or quantitatively inconsistent
  with what IPCC AR6 recommends for a 1.5°C or 2°C pathway.
  → This is an ambition gap — the commitment exists but falls short.

ADDITIONAL FLAG — CONDITIONAL DEPENDENCY
  The country's commitment is conditional on receiving international finance.
  Flag this as it creates delivery risk even if the target is ambitious.

SEVERITY CLASSIFICATION:
  CRITICAL    = Missing entirely OR quantified gap >50% below IPCC minimum
  SIGNIFICANT = Present but vague OR quantified gap 20–50% below IPCC minimum
  MINOR       = Present, measurable, close to IPCC but not fully aligned (<20% gap)
  ALIGNED     = Meets or exceeds IPCC recommendation

RULES:
1. Cite every finding with [doc_id, p.N] from BOTH the NDC and IPCC context.
2. Quantify the gap wherever possible (e.g. "NDC: 40%, IPCC min: 70%, gap: 30pp").
3. Assign confidence 0.0–1.0 to each finding based on evidence quality.
4. If evidence is insufficient for a finding, state INSUFFICIENT_EVIDENCE.
5. Return ONLY valid JSON. No prose before or after.

OUTPUT SCHEMA:
{
  "country": "<country>",
  "dimension": "<dimension>",
  "overall_severity": "<CRITICAL|SIGNIFICANT|MINOR|ALIGNED>",
  "overall_severity_score": <float 0.0-1.0>,
  "gap_findings": [
    {
      "gap_type": "<missing_commitment|vague_target|ipcc_contradiction|conditional_dependency>",
      "severity": "<CRITICAL|SIGNIFICANT|MINOR|ALIGNED>",
      "severity_score": <float 0.0-1.0>,
      "description": "<clear description of the gap>",
      "ndc_evidence": "<what the NDC says, or 'No commitment found'>",
      "ipcc_evidence": "<what IPCC recommends>",
      "ndc_citation": "<[doc_id, p.N] or 'N/A'>",
      "ipcc_citation": "<[doc_id, p.N]>",
      "quantified_gap": "<e.g. NDC: 40%, IPCC min: 70%, gap: -30pp or null>",
      "confidence": <float 0.0-1.0>
    }
  ],
  "country_summary": "<2-3 sentence summary of overall gap assessment>",
  "policy_recommendation": "<one specific, actionable recommendation for this country>"
}"""

    PASS_C_USER = """COUNTRY BEING EVALUATED: {country}
DIMENSION: {dimension_label}
NDC CYCLES ANALYSED: {ndc_cycles}

────── NDC COMMITMENTS (extracted in Pass A & B) ──────
Measurability level: {measurability_level} (score: {measurability_score})
{commitments_json}

────── IPCC AR6 BENCHMARK ──────
Scientific recommendation: {ipcc_target}
{ipcc_context}

────── IPCC RETRIEVED PASSAGES ──────
{ipcc_context_chunks}

Identify all gaps between {country}'s NDC commitments and the IPCC benchmark above.
Return only JSON. Cite every claim."""

    # ── PAIRWISE COMPARISON (synthesis across countries) ─────────────────────
    PAIRWISE_SYSTEM = """You are a climate policy analyst comparing NDC ambition across countries.

TASK: Compare two countries' gap analysis results on the same dimension and provide a structured comparative assessment.

RULES:
1. Use ONLY the provided gap analysis results. Do not use external knowledge.
2. Be precise about which country is more ambitious and why.
3. Identify specific areas where they converge and diverge.
4. Return ONLY valid JSON.

OUTPUT SCHEMA:
{
  "country_a": "<name>",
  "country_b": "<name>",
  "dimension": "<dimension>",
  "ambition_leader": "<country_a|country_b|tied>",
  "leadership_reason": "<one sentence explaining who leads and why>",
  "relative_gap": "<2-3 sentence comparative assessment>",
  "convergence_areas": ["<area1>", "<area2>"],
  "divergence_areas": ["<area1>", "<area2>"],
  "combined_ipcc_gap": "<how both countries together compare to IPCC benchmark>"
}"""

    PAIRWISE_USER = """DIMENSION: {dimension_label}
IPCC BENCHMARK: {ipcc_target}

COUNTRY A — {country_a}:
  Measurability: {measurability_a} (score: {score_a})
  Overall severity: {severity_a} (score: {severity_score_a})
  Key gaps: {gaps_a}
  Summary: {summary_a}

COUNTRY B — {country_b}:
  Measurability: {measurability_b} (score: {score_b})
  Overall severity: {severity_b} (score: {severity_score_b})
  Key gaps: {gaps_b}
  Summary: {summary_b}

Compare {country_a} and {country_b} on {dimension_label}. Return only JSON."""

    # ── G20 SYNTHESIS ─────────────────────────────────────────────────────────
    G20_SYNTHESIS_SYSTEM = """You are a climate policy analyst synthesising gap analysis results across multiple countries.

TASK: Given individual country gap analyses, produce a G20-level synthesis.

OUTPUT SCHEMA:
{
  "dimension": "<dimension>",
  "countries_analysed": <integer>,
  "collective_gap_severity": "<CRITICAL|SIGNIFICANT|MINOR|ALIGNED>",
  "most_ambitious": ["<country1>", "<country2>"],
  "least_ambitious": ["<country1>", "<country2>"],
  "systemic_patterns": ["<pattern1>", "<pattern2>", "<pattern3>"],
  "common_gaps": ["<gap shared by majority of countries>"],
  "best_practices": ["<what the leading countries do that others don't>"],
  "collective_ipcc_alignment": "<overall assessment of G20 collective ambition vs IPCC>",
  "thesis_finding": "<one key finding suitable for thesis abstract>"
}"""

    G20_SYNTHESIS_USER = """DIMENSION: {dimension_label}
IPCC BENCHMARK: {ipcc_target}

COUNTRY RESULTS:
{all_country_summaries}

Synthesise these {n_countries} country analyses into a G20-level assessment.
Return only JSON."""


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 4 — JSON PARSER WITH REPAIR
# ═══════════════════════════════════════════════════════════════════════════════

class JSONParser:
    """
    Robust JSON extractor for LLM outputs.
    LLMs sometimes wrap JSON in markdown fences or add prose before/after.
    This parser tries multiple strategies to extract valid JSON.
    """

    @staticmethod
    def parse(text: str) -> Optional[dict]:
        """Try to extract and parse JSON from LLM output. Returns dict or None."""
        if not text:
            return None

        # Strategy 1: direct parse (model returned clean JSON)
        try:
            return json.loads(text.strip())
        except json.JSONDecodeError:
            pass

        # Strategy 2: extract from ```json ... ``` fence
        fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if fence:
            try:
                return json.loads(fence.group(1))
            except json.JSONDecodeError:
                pass

        # Strategy 3: find outermost { ... } block
        brace_start = text.find("{")
        brace_end   = text.rfind("}")
        if brace_start != -1 and brace_end > brace_start:
            try:
                return json.loads(text[brace_start:brace_end + 1])
            except json.JSONDecodeError:
                pass

        # Strategy 4: attempt to fix common issues (trailing commas, single quotes)
        fixed = re.sub(r",\s*([\}\]])", r"\1", text)  # trailing commas
        fixed = fixed.replace("'", '"')                # single → double quotes
        brace_start = fixed.find("{")
        brace_end   = fixed.rfind("}")
        if brace_start != -1 and brace_end > brace_start:
            try:
                return json.loads(fixed[brace_start:brace_end + 1])
            except json.JSONDecodeError:
                pass

        return None


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 5 — CORE GAP ANALYSER
# ═══════════════════════════════════════════════════════════════════════════════

class GapAnalyser:
    """
    Main entry point for the Gap Analysis Module.

    Connects to HybridRetriever and an LLM backend (via generation.py)
    and runs the full three-pass gap analysis pipeline.

    Parameters
    ----------
    backend      : "mistral" | "gpt4o"
    top_k_ndc    : chunks fetched per country per NDC cycle
    top_k_ipcc   : chunks fetched for IPCC benchmark
    debug        : print intermediate LLM outputs
    gen_dir      : explicit path to folder containing generation.py
                   (leave None to auto-search)
    """

    def __init__(
        self,
        backend:       str  = "mistral",
        top_k_ndc:     int  = 8,    # increased from 5 — India NDC2 has only 3 chunks, need wider net
        top_k_ipcc:    int  = 6,    # increased from 4 — more IPCC benchmark context improves gap detection
        debug:         bool = False,
        gen_dir:       Optional[str] = None,   # explicit path to generation.py folder
        # Retriever path overrides
        chunks_file:   str  = "data/processed/chunks.jsonl",
        vectorstore:   str  = "data/vectorstore",
        collection:    str  = "climate_policy_rag",
        embed_device:  str  = "cpu",
        llm_device:    str  = "auto",
        load_in_4bit:  bool = True,
    ):
        self.backend    = backend
        self.top_k_ndc  = top_k_ndc
        self.top_k_ipcc = top_k_ipcc
        self.debug      = debug

        print("\n[GapAnalyser] Initialising...")

        # ── Locate generation.py robustly ─────────────────────────────────────
        # Searches in order:
        #   1. Explicit gen_dir if supplied (--gen_dir CLI flag)
        #   2. Same folder as gap_analysis.py      (Gap_Analysis_Code\)
        #   3. Sibling folder LLM_Integration_Code (D:\Thesis\LLM_Integration_Code\)
        #   4. Parent folder                       (D:\Thesis\)
        #   5. Current working directory           (wherever you ran the script from)
        #   6. D:\Thesis\  hardcoded fallback

        _this_dir    = Path(__file__).resolve().parent        # Gap_Analysis_Code\
        _parent_dir  = _this_dir.parent                       # D:\Thesis\
        _search_dirs = [
            _this_dir,
            _parent_dir / "LLM_Integration_Code",
            _parent_dir,
            Path.cwd(),
            Path("D:/Thesis/LLM_Integration_Code"),
            Path("D:/Thesis"),
        ]
        # Prepend explicit gen_dir if provided
        if gen_dir is not None:
            _search_dirs.insert(0, Path(gen_dir))

        _generation_path = None
        for _d in _search_dirs:
            _candidate = _d / "generation.py"
            if _candidate.exists():
                _generation_path = _candidate
                break

        if _generation_path is None:
            searched = "\n    ".join(str(d) for d in _search_dirs)
            raise FileNotFoundError(
                "Could not find generation.py. Searched:\n    "
                + searched
                + "\n\nFix options:"
                + "\n  A) Copy generation.py into the same folder as gap_analysis.py"
                + "\n  B) Run: python gap_analysis.py --gen_dir D:\\Thesis\\LLM_Integration_Code"
                + "\n  C) Set the gen_dir parameter when constructing GapAnalyser()"
            )

        print(f"  [GapAnalyser] Found generation.py at: {_generation_path}")
        sys.path.insert(0, str(_generation_path.parent))

        from generation import RAGPipeline

        self._pipeline = RAGPipeline(
            backend      = backend,
            top_k        = top_k_ndc,
            load_in_4bit = load_in_4bit,
            device       = llm_device,
            chunks_file  = chunks_file,
            vectorstore  = vectorstore,
            collection   = collection,
            embed_device = embed_device,
        )
        self._retriever = self._pipeline._retriever
        self._llm       = self._pipeline._llm

        print("[GapAnalyser] Ready.\n")

    # ── Private: LLM call wrapper ─────────────────────────────────────────────

    def _llm_call(
        self,
        system_prompt: str,
        user_prompt:   str,
        pass_label:    str = "LLM",
    ) -> Optional[dict]:
        """
        Call the LLM with a system+user message and parse JSON response.
        Returns parsed dict or None on failure.
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ]
        raw, stats = self._llm.generate(messages)

        if self.debug:
            print(f"\n  [{pass_label}] Raw output ({stats['completion_tokens']} tokens):")
            print(f"  {raw[:500]}...")

        parsed = JSONParser.parse(raw)
        if parsed is None:
            print(f"  [WARN] [{pass_label}] JSON parse failed. Raw: {raw[:200]}")
        return parsed

    # ── Private: Retrieval helpers ────────────────────────────────────────────

    def _retrieve_ndc(
        self,
        country:   str,
        dimension: str,
        ndc_cycle: Optional[int] = None,
    ) -> list[dict]:
        """Retrieve NDC chunks for a country + dimension."""
        dim_cfg = DIMENSIONS[dimension]
        query   = f"{country} {' '.join(dim_cfg['keywords'][:4])} NDC target commitment"

        filters: dict = {"country": country}
        if ndc_cycle is not None:
            filters = {"$and": [{"country": country}, {"ndc_cycle": ndc_cycle}]}

        return self._retriever.retrieve(
            query   = query,
            top_k   = self.top_k_ndc,
            filters = filters,
        )

    def _retrieve_ipcc_benchmark(self, dimension: str) -> list[dict]:
        """Retrieve IPCC AR6 benchmark chunks for a dimension."""
        dim_cfg = DIMENSIONS[dimension]
        return self._retriever.retrieve(
            query   = dim_cfg["ipcc_query"],
            top_k   = self.top_k_ipcc,
            filters = dim_cfg["ipcc_filter"],
        )

    @staticmethod
    def _format_chunks_as_context(chunks: list[dict]) -> str:
        """Format retrieved chunks for inclusion in a prompt."""
        if not chunks:
            return "NO CONTEXT RETRIEVED."
        blocks = []
        for i, chunk in enumerate(chunks, 1):
            doc_id  = chunk.get("doc_id", "UNKNOWN")
            page    = chunk.get("page_start", "?")
            section = (chunk.get("section") or "")[:70]
            text    = (chunk.get("text") or "").strip()[:800]
            blocks.append(
                f"[SOURCE {i}: {doc_id}, p.{page}]\n"
                f"Section: {section}\n"
                f"{text}"
            )
        return "\n\n".join(blocks)

    # ── Pass A: Commitment Extraction ─────────────────────────────────────────

    def _pass_a_extract(
        self,
        country:    str,
        dimension:  str,
        ndc_cycle:  int,
        ndc_chunks: list[dict],
    ) -> Optional[dict]:
        """Extract commitments from NDC context."""
        dim_cfg = DIMENSIONS[dimension]
        context = self._format_chunks_as_context(ndc_chunks)

        user_msg = GapPrompts.PASS_A_USER.format(
            country              = country,
            cycle                = ndc_cycle,
            ndc_context          = context,
            dimension_label      = dim_cfg["label"],
            dimension_description = dim_cfg["description"],
            keywords             = ", ".join(dim_cfg["keywords"]),
        )
        return self._llm_call(
            GapPrompts.PASS_A_SYSTEM, user_msg,
            pass_label = f"PassA_{country}_NDC{ndc_cycle}",
        )

    # ── Pass B: Measurability Classification ──────────────────────────────────

    def _pass_b_classify(
        self,
        country:     str,
        dimension:   str,
        commitments: list[dict],
    ) -> Optional[dict]:
        """Classify measurability of extracted commitments."""
        dim_cfg = DIMENSIONS[dimension]
        user_msg = GapPrompts.PASS_B_USER.format(
            country          = country,
            dimension_label  = dim_cfg["label"],
            commitments_json = json.dumps(commitments, indent=2),
        )
        return self._llm_call(
            GapPrompts.PASS_B_SYSTEM, user_msg,
            pass_label = f"PassB_{country}",
        )

    # ── Pass C: Gap Detection ─────────────────────────────────────────────────

    def _pass_c_gaps(
        self,
        country:            str,
        dimension:          str,
        ndc_cycles:         list[int],
        measurability_data: dict,
        commitments:        list[dict],
        ipcc_chunks:        list[dict],
    ) -> Optional[dict]:
        """Detect gaps between NDC commitments and IPCC benchmark."""
        dim_cfg      = DIMENSIONS[dimension]
        ipcc_context = self._format_chunks_as_context(ipcc_chunks)

        user_msg = GapPrompts.PASS_C_USER.format(
            country              = country,
            dimension_label      = dim_cfg["label"],
            ndc_cycles           = ndc_cycles,
            measurability_level  = measurability_data.get("measurability_level", "UNKNOWN"),
            measurability_score  = measurability_data.get("measurability_score", 0),
            commitments_json     = json.dumps(commitments, indent=2),
            ipcc_target          = dim_cfg["ipcc_ref"],
            ipcc_context         = dim_cfg["description"],
            ipcc_context_chunks  = ipcc_context,
        )
        return self._llm_call(
            GapPrompts.PASS_C_SYSTEM, user_msg,
            pass_label = f"PassC_{country}",
        )

    # ── Pairwise comparison ───────────────────────────────────────────────────

    def _pairwise_compare(
        self,
        country_a: str,
        country_b: str,
        analysis_a: CountryAnalysis,
        analysis_b: CountryAnalysis,
        dimension:  str,
    ) -> PairwiseGap:
        """Compare two countries' analyses head-to-head."""
        dim_cfg = DIMENSIONS[dimension]

        def summarise_gaps(a: CountryAnalysis) -> str:
            return "; ".join(
                f"{g.gap_type}({g.severity})" for g in a.gap_findings[:3]
            ) or "No gaps found"

        user_msg = GapPrompts.PAIRWISE_USER.format(
            dimension_label   = dim_cfg["label"],
            ipcc_target       = dim_cfg["ipcc_ref"],
            country_a         = country_a,
            measurability_a   = analysis_a.measurability.level,
            score_a           = analysis_a.measurability.score,
            severity_a        = analysis_a.overall_severity,
            severity_score_a  = analysis_a.overall_score,
            gaps_a            = summarise_gaps(analysis_a),
            summary_a         = analysis_a.summary,
            country_b         = country_b,
            measurability_b   = analysis_b.measurability.level,
            score_b           = analysis_b.measurability.score,
            severity_b        = analysis_b.overall_severity,
            severity_score_b  = analysis_b.overall_score,
            gaps_b            = summarise_gaps(analysis_b),
            summary_b         = analysis_b.summary,
        )
        result = self._llm_call(
            GapPrompts.PAIRWISE_SYSTEM, user_msg,
            pass_label = f"Pairwise_{country_a}_{country_b}",
        )

        if result:
            return PairwiseGap(
                country_a         = country_a,
                country_b         = country_b,
                dimension         = dimension,
                relative_gap      = result.get("relative_gap", ""),
                ambition_leader   = result.get("ambition_leader", "tied"),
                leadership_reason = result.get("leadership_reason", ""),
                convergence_areas = result.get("convergence_areas", []),
                divergence_areas  = result.get("divergence_areas", []),
            )

        # Fallback if LLM call fails
        score_a = analysis_a.overall_score
        score_b = analysis_b.overall_score
        return PairwiseGap(
            country_a         = country_a,
            country_b         = country_b,
            dimension         = dimension,
            relative_gap      = f"{country_a} gap score={score_a:.2f}, {country_b} gap score={score_b:.2f}",
            ambition_leader   = country_a if score_a < score_b else country_b,
            leadership_reason = "Based on gap severity scores (lower = more aligned)",
            convergence_areas = [],
            divergence_areas  = [],
        )

    # ── Build CountryAnalysis from raw LLM outputs ────────────────────────────

    def _build_country_analysis(
        self,
        country:       str,
        dimension:     str,
        ndc_cycles:    list[int],
        pass_a_data:   Optional[dict],
        pass_b_data:   Optional[dict],
        pass_c_data:   Optional[dict],
    ) -> CountryAnalysis:
        """Assemble CountryAnalysis dataclass from three pass outputs."""

        # ── Commitments from Pass A ───────────────────────────────────────────
        raw_commitments = []
        if pass_a_data and "commitments" in pass_a_data:
            raw_commitments = pass_a_data["commitments"]

        commitments = [
            CommitmentItem(
                text          = c.get("text", ""),
                target_value  = c.get("target_value"),
                target_year   = c.get("target_year"),
                baseline_year = c.get("baseline_year"),
                conditional   = c.get("conditional", False),
                time_horizon  = c.get("time_horizon", "unclear"),
                source_doc    = c.get("source_doc", ""),
                source_page   = int(c.get("source_page", 0)),
                confidence    = float(c.get("confidence", 0.5)),
            )
            for c in raw_commitments
        ]

        # ── Measurability from Pass B ─────────────────────────────────────────
        if pass_b_data:
            measurability = MeasurabilityScore(
                level            = pass_b_data.get("measurability_level", "ABSENT"),
                score            = float(pass_b_data.get("measurability_score", 0.0)),
                reason           = pass_b_data.get("reason", ""),
                missing_elements = pass_b_data.get("missing_elements", []),
            )
        else:
            measurability = MeasurabilityScore(
                level="ABSENT", score=0.0,
                reason="LLM classification unavailable",
                missing_elements=[],
            )

        # ── Gap findings from Pass C ──────────────────────────────────────────
        gap_findings = []
        overall_severity       = "ALIGNED"
        overall_severity_score = 0.0
        summary                = ""

        if pass_c_data:
            overall_severity       = pass_c_data.get("overall_severity", "ALIGNED")
            overall_severity_score = float(pass_c_data.get("overall_severity_score", 0.0))
            summary                = pass_c_data.get("country_summary", "")

            for g in pass_c_data.get("gap_findings", []):
                gap_findings.append(GapFinding(
                    gap_type       = g.get("gap_type", "missing_commitment"),
                    severity       = g.get("severity", "MINOR"),
                    severity_score = float(g.get("severity_score", 0.0)),
                    description    = g.get("description", ""),
                    ndc_evidence   = g.get("ndc_evidence", ""),
                    ipcc_evidence  = g.get("ipcc_evidence", ""),
                    ndc_citation   = g.get("ndc_citation", "N/A"),
                    ipcc_citation  = g.get("ipcc_citation", "N/A"),
                    quantified_gap = g.get("quantified_gap"),
                    confidence     = float(g.get("confidence", 0.5)),
                ))

        return CountryAnalysis(
            country          = country,
            dimension        = dimension,
            ndc_cycles       = ndc_cycles,
            commitments      = commitments,
            measurability    = measurability,
            gap_findings     = gap_findings,
            overall_severity = overall_severity,
            overall_score    = overall_severity_score,
            summary          = summary,
            raw_llm_outputs  = {
                "pass_a": pass_a_data,
                "pass_b": pass_b_data,
                "pass_c": pass_c_data,
            },
        )

    # ── PUBLIC: main analysis entry point ─────────────────────────────────────

    def analyse(
        self,
        countries:  list[str],
        dimension:  str,
        ndc_cycles: list[int] = [2],
        save:       bool      = True,
    ) -> GapReport:
        """
        Run the complete 3-pass gap analysis for a list of countries
        on a single dimension.

        Parameters
        ----------
        countries  : list of country names (must match metadata in corpus)
        dimension  : one of the keys in DIMENSIONS dict
        ndc_cycles : which NDC cycles to include (default [2] = second cycle)
        save       : write JSON output to data/gap_reports/

        Returns
        -------
        GapReport with all analyses, pairwise comparisons, and rankings
        """
        if dimension not in DIMENSIONS:
            raise ValueError(
                f"Unknown dimension '{dimension}'. "
                f"Choose from: {list(DIMENSIONS.keys())}"
            )

        dim_cfg = DIMENSIONS[dimension]
        print(f"\n{'═'*65}")
        print(f"  GAP ANALYSIS: {dim_cfg['label']}")
        print(f"  Countries  : {', '.join(countries)}")
        print(f"  NDC cycles : {ndc_cycles}")
        print(f"  Backend    : {self.backend}")
        print(f"{'═'*65}\n")

        t_start = time.time()

        # ── 1. Retrieve IPCC benchmark once (shared across all countries) ─────
        print("  [1/4] Retrieving IPCC AR6 benchmark...")
        ipcc_chunks = self._retrieve_ipcc_benchmark(dimension)
        ipcc_citations = [
            f"[{c.get('doc_id', '?')}, p.{c.get('page_start', '?')}]"
            for c in ipcc_chunks
        ]
        benchmark = BenchmarkBlock(
            dimension    = dimension,
            ipcc_target  = dim_cfg["ipcc_ref"],
            ipcc_context = dim_cfg["description"],
            source_chunks = [c.get("doc_id", "") for c in ipcc_chunks],
            citations    = ipcc_citations,
        )
        print(f"    → {len(ipcc_chunks)} IPCC chunks retrieved")

        # ── 2. Per-country three-pass analysis ────────────────────────────────
        print(f"\n  [2/4] Running 3-pass analysis per country...")
        country_analyses: dict[str, CountryAnalysis] = {}

        for country in countries:
            print(f"\n    ── {country} ──")
            all_commitments: list[dict] = []
            all_pass_a:      list[dict] = []

            # Collect commitments across all requested NDC cycles
            for cycle in ndc_cycles:
                print(f"      Pass A: extracting commitments (NDC{cycle})...")
                ndc_chunks = self._retrieve_ndc(country, dimension, ndc_cycle=cycle)
                print(f"        → {len(ndc_chunks)} NDC chunks retrieved")

                pass_a = self._pass_a_extract(country, dimension, cycle, ndc_chunks)
                if pass_a and pass_a.get("commitments"):
                    all_commitments.extend(pass_a["commitments"])
                    all_pass_a.append(pass_a)

            # Deduplicate commitments by text similarity (simple exact-match)
            seen_texts = set()
            unique_commitments = []
            for c in all_commitments:
                t = c.get("text", "")[:100]
                if t not in seen_texts:
                    seen_texts.add(t)
                    unique_commitments.append(c)

            merged_pass_a = {"commitments": unique_commitments,
                             "total_found": len(unique_commitments)}

            # Pass B: classify measurability
            print(f"      Pass B: classifying measurability...")
            pass_b = self._pass_b_classify(country, dimension, unique_commitments)

            # Pass C: detect gaps vs IPCC
            print(f"      Pass C: detecting gaps vs IPCC...")
            pass_c = self._pass_c_gaps(
                country            = country,
                dimension          = dimension,
                ndc_cycles         = ndc_cycles,
                measurability_data = pass_b or {},
                commitments        = unique_commitments,
                ipcc_chunks        = ipcc_chunks,
            )

            analysis = self._build_country_analysis(
                country, dimension, ndc_cycles,
                merged_pass_a, pass_b, pass_c,
            )
            country_analyses[country] = analysis

            sev_icon = {"CRITICAL":"🔴","SIGNIFICANT":"🟠","MINOR":"🟡","ALIGNED":"🟢"}.get(
                analysis.overall_severity, "⚪")
            print(f"      {sev_icon} {analysis.overall_severity} "
                  f"(score={analysis.overall_score:.2f})  "
                  f"Measurability={analysis.measurability.level}")

        # ── 3. Pairwise comparisons ───────────────────────────────────────────
        print(f"\n  [3/4] Building pairwise comparisons...")
        pairwise_gaps: list[PairwiseGap] = []
        for i, ca in enumerate(countries):
            for cb in countries[i+1:]:
                if ca in country_analyses and cb in country_analyses:
                    pw = self._pairwise_compare(
                        ca, cb,
                        country_analyses[ca],
                        country_analyses[cb],
                        dimension,
                    )
                    pairwise_gaps.append(pw)
                    print(f"    {ca} vs {cb}: leader={pw.ambition_leader}")

        # ── 4. Rankings ───────────────────────────────────────────────────────
        rankings = {
            c: a.overall_score
            for c, a in country_analyses.items()
        }

        # ── 5. G20 synthesis (if ≥3 countries) ───────────────────────────────
        g20_summary: Optional[dict] = None
        if len(countries) >= 3:
            print(f"\n  [4/4] Generating G20 synthesis ({len(countries)} countries)...")
            all_summaries = "\n\n".join([
                f"{c}:\n  Severity={a.overall_severity} (score={a.overall_score:.2f})\n"
                f"  Measurability={a.measurability.level}\n  Summary={a.summary}"
                for c, a in country_analyses.items()
            ])
            g20_result = self._llm_call(
                GapPrompts.G20_SYNTHESIS_SYSTEM,
                GapPrompts.G20_SYNTHESIS_USER.format(
                    dimension_label       = dim_cfg["label"],
                    ipcc_target           = dim_cfg["ipcc_ref"],
                    all_country_summaries = all_summaries,
                    n_countries           = len(countries),
                ),
                pass_label = "G20_Synthesis",
            )
            g20_summary = g20_result

        # ── 6. Policy implications ────────────────────────────────────────────
        policy_implications = []
        critical_countries = [
            c for c, a in country_analyses.items()
            if a.overall_severity == "CRITICAL"
        ]
        if critical_countries:
            policy_implications.append(
                f"Critical gaps detected in: {', '.join(critical_countries)} "
                f"on {dim_cfg['label']} — immediate NDC strengthening needed."
            )
        conditional_countries = [
            c for c, a in country_analyses.items()
            if any(f.gap_type == "conditional_dependency" for f in a.gap_findings)
        ]
        if conditional_countries:
            policy_implications.append(
                f"Finance-conditional commitments in: {', '.join(conditional_countries)} "
                f"— delivery at risk without international support."
            )
        if g20_summary and "thesis_finding" in (g20_summary or {}):
            policy_implications.append(g20_summary["thesis_finding"])

        elapsed = time.time() - t_start

        report = GapReport(
            dimension          = dimension,
            dimension_label    = dim_cfg["label"],
            countries          = countries,
            ndc_cycles         = ndc_cycles,
            ipcc_benchmark     = benchmark,
            country_analyses   = country_analyses,
            pairwise_gaps      = pairwise_gaps,
            g20_summary        = g20_summary,
            rankings           = rankings,
            policy_implications = policy_implications,
            metadata           = {
                "generated_at":   datetime.now(timezone.utc).isoformat(),
                "backend":        self.backend,
                "elapsed_sec":    round(elapsed, 1),
                "top_k_ndc":      self.top_k_ndc,
                "top_k_ipcc":     self.top_k_ipcc,
                "ipcc_chunks_used": len(ipcc_chunks),
                "pipeline_version": "1.0",
            },
        )

        print(f"\n  ✓ Gap analysis complete in {elapsed:.1f}s")
        if save:
            report.save()

        return report

    # ── PUBLIC: G20 full sweep ────────────────────────────────────────────────

    def analyse_g20(
        self,
        dimension:  str,
        ndc_cycles: list[int] = [2],
        save:       bool      = True,
    ) -> GapReport:
        """
        Run gap analysis across all G20 countries in the corpus.
        This produces the main thesis result for Chapter 5.
        """
        print(f"\n[GapAnalyser] G20 sweep: {DIMENSIONS[dimension]['label']}")
        print(f"  Running across {len(G20_COUNTRIES)} countries...")
        return self.analyse(
            countries  = G20_COUNTRIES,
            dimension  = dimension,
            ndc_cycles = ndc_cycles,
            save       = save,
        )

    # ── PUBLIC: batch across dimensions ───────────────────────────────────────

    def analyse_all_dimensions(
        self,
        countries:  list[str],
        ndc_cycles: list[int] = [2],
        save:       bool      = True,
    ) -> dict[str, GapReport]:
        """
        Run gap analysis for all 7 dimensions on the same set of countries.
        Use this to build the full Chapter 5 comparative table.
        """
        reports = {}
        for dim in DIMENSIONS:
            print(f"\n{'='*65}")
            print(f"  DIMENSION: {DIMENSIONS[dim]['label']}")
            print(f"{'='*65}")
            try:
                report = self.analyse(
                    countries  = countries,
                    dimension  = dim,
                    ndc_cycles = ndc_cycles,
                    save       = save,
                )
                reports[dim] = report
            except Exception as e:
                print(f"  [ERROR] {dim}: {e}")
        return reports


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 6 — CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Gap Analysis Module — climate policy vs IPCC benchmark"
    )
    parser.add_argument("--countries", nargs="+",
                        help="Country names (e.g. India China)")
    parser.add_argument("--dimension", default="renewable_energy",
                        choices=list(DIMENSIONS.keys()))
    parser.add_argument("--ndc_cycles", nargs="+", type=int, default=[2],
                        help="NDC cycles to include (1, 2, or both)")
    parser.add_argument("--g20",     action="store_true",
                        help="Run across all G20 countries")
    parser.add_argument("--all_dims", action="store_true",
                        help="Run all 7 dimensions")
    parser.add_argument("--backend", default="mistral",
                        choices=["mistral", "gpt4o"])
    parser.add_argument("--debug",   action="store_true")
    parser.add_argument("--no_save", action="store_true")
    parser.add_argument("--no_4bit", action="store_true")
    parser.add_argument(
        "--gen_dir",
        default = None,
        help    = (
            "Path to the folder containing generation.py and retriever.py. "
            "Only needed if they live in a different folder from gap_analysis.py. "
            r"Example: --gen_dir D:\Thesis\LLM_Integration_Code"
        ),
    )
    args = parser.parse_args()

    analyser = GapAnalyser(
        backend      = args.backend,
        debug        = args.debug,
        load_in_4bit = not args.no_4bit,
        gen_dir      = args.gen_dir,   # None = auto-search
    )

    if args.g20:
        report = analyser.analyse_g20(
            dimension  = args.dimension,
            ndc_cycles = args.ndc_cycles,
            save       = not args.no_save,
        )
        print("\n" + report.summary_text)

    elif args.all_dims and args.countries:
        reports = analyser.analyse_all_dimensions(
            countries  = args.countries,
            ndc_cycles = args.ndc_cycles,
            save       = not args.no_save,
        )
        for dim, report in reports.items():
            print(f"\n{'─'*60}")
            print(report.summary_text)

    elif args.countries:
        report = analyser.analyse(
            countries  = args.countries,
            dimension  = args.dimension,
            ndc_cycles = args.ndc_cycles,
            save       = not args.no_save,
        )
        print("\n" + report.summary_text)

    else:
        parser.print_help()
        print("\n  Examples:")
        print("  python gap_analysis.py --countries India China --dimension renewable_energy")
        print("  python gap_analysis.py --countries Indonesia Brazil --dimension adaptation_finance")
        print("  python gap_analysis.py --countries UK EU Australia --dimension net_zero_pathway")
        print("  python gap_analysis.py --g20 --dimension renewable_energy")
        print("  python gap_analysis.py --countries India China --all_dims")


if __name__ == "__main__":
    main()
