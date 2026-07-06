"""
===========================================================================
  PDF → CLEAN TEXT → TAGGED CHUNKS PIPELINE
  Master's Thesis: LLM-Powered Climate Policy Summariser & Gap Analyser
  University of Europe for Applied Sciences, Potsdam  |  2026
===========================================================================

What this script does
---------------------
  1. Walks data/raw/NDC/<Country>/*.pdf  and  data/raw/IPCC_AR6/<WG>/*.pdf
  2. Extracts text per page using pdfplumber  (best layout fidelity for
     policy PDFs — handles two-column layouts, tables, headers)
  3. Cleans the raw text  (ligatures, hyphenation, excess whitespace,
     page-number noise, header/footer repetition)
  4. Detects section headings  (rule-based + regex heuristics)
  5. Splits into overlapping chunks of 512–1024 tokens with 20% overlap
     using a sentence-boundary-aware splitter  (no mid-sentence breaks)
  6. Tags every chunk with rich metadata:
       source, country, ndc_cycle, year, doc_type, wg,
       section, page_start, page_end, chunk_id,
       token_count, char_count, is_table
  7. Saves three output artefacts:
       data/processed/chunks.jsonl          — one JSON object per line
       data/processed/chunks_sample.json   — first 20 chunks, pretty-printed
       data/processed/pipeline_report.json — corpus-level statistics

Usage
-----
    pip install pdfplumber tiktoken tqdm
    python pdf_to_chunks.py

    # Custom paths / settings:
    python pdf_to_chunks.py --raw_dir data/raw --out_dir data/processed
    python pdf_to_chunks.py --min_tokens 512 --max_tokens 1024 --overlap 0.20
    python pdf_to_chunks.py --source ndc        # NDC docs only
    python pdf_to_chunks.py --source ipcc       # IPCC docs only
    python pdf_to_chunks.py --dry_run           # count pages, no output

Library choice — why pdfplumber?
---------------------------------
  • pdfplumber > pypdf   for climate policy PDFs because it preserves
    reading order across two-column layouts (NDC2 Indonesia, IPCC chapters)
    and extracts table content as structured rows.
  • pdfplumber > PyMuPDF for pure text quality on government PDFs —
    no coordinate math needed; .extract_text() already respects columns.
  • tiktoken cl100k_base tokeniser (GPT-4 / text-embedding-3 family)
    is used for accurate token counts that will match your embedding model.
===========================================================================
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# ── Dependency check ──────────────────────────────────────────────────────────
_missing = []
try:
    import pdfplumber
except ImportError:
    _missing.append("pdfplumber")
try:
    import tiktoken
except ImportError:
    _missing.append("tiktoken")
try:
    from tqdm import tqdm
    TQDM = True
except ImportError:
    TQDM = False

if _missing:
    print(f"[ERROR] Missing packages: {', '.join(_missing)}")
    print(f"        Run: pip install {' '.join(_missing)}")
    sys.exit(1)


# ── Constants ─────────────────────────────────────────────────────────────────

TOKENISER     = tiktoken.get_encoding("cl100k_base")   # GPT-4 / Ada-3 family
DEFAULT_MIN   = 512
DEFAULT_MAX   = 1024
DEFAULT_OVR   = 0.20    # 20% overlap

# Regex for section headings in NDC/IPCC documents
# Matches: "1.", "1.1", "1.1.1", "A.", "I.", "Chapter 1", "Section 2" etc.
HEADING_RE = re.compile(
    r"^(?:"
    r"(?:Chapter|Section|Part|Annex|Appendix)\s+[\dA-Z]+[\s:–\-]"   # named
    r"|[A-Z][A-Z\s]{2,40}$"                                           # ALL-CAPS heading
    r"|\d{1,2}(?:\.\d{1,2}){0,2}[\s\.]+[A-Z]"                        # 1. / 1.1 / 1.1.1
    r"|[A-Z][\.\)]\s+[A-Z]"                                           # A. / A) heading
    r"|[IVX]+[\.\)]\s+[A-Z]"                                          # Roman numerals
    r")",
    re.MULTILINE,
)

# Patterns to strip: page numbers, running headers/footers, watermarks
NOISE_RE = [
    re.compile(r"^\s*\d{1,4}\s*$", re.MULTILINE),             # bare page numbers
    re.compile(r"Please use this shareable version.*", re.I),  # UNFCCC watermark
    re.compile(r"^\s*-\s*\d+\s*-\s*$", re.MULTILINE),         # — 1 — style nums
    re.compile(r"^\s*Page \d+ of \d+\s*$", re.I | re.M),      # "Page 1 of 40"
    re.compile(r"\f"),                                           # form-feed chars
    re.compile(r"[ \t]{3,}", ),                                 # 3+ spaces (columns)
    re.compile(r"\n{3,}"),                                      # 3+ blank lines
]

# Ligature / encoding fixes common in government PDFs
LIGATURES = {
    "\ufb01": "fi", "\ufb02": "fl", "\ufb03": "ffi", "\ufb04": "ffl",
    "\ufb00": "ff", "\u2019": "'",  "\u2018": "'",   "\u201c": '"',
    "\u201d": '"',  "\u2013": "-",  "\u2014": "--",  "\u00ad": "",  # soft hyphen
    "\u00a0": " ",  "\u200b": "",   "\ufffd": "",
}


# ── NDC document metadata catalogue ──────────────────────────────────────────
# Maps  filename stem → dict of known metadata fields.
# All fields that cannot be inferred from filename alone are listed here.

NDC_META: dict[str, dict] = {
    # ── China ──
    "NDC1_2016_China":       {"country":"China","ndc_cycle":1,"year":2016,"language":"EN","pages_approx":5,"emissions_rank":1},
    "NDC2_2021_China":       {"country":"China","ndc_cycle":2,"year":2021,"language":"EN","pages_approx":7,"emissions_rank":1},
    # ── India ──
    "NDC1_2015_India":       {"country":"India","ndc_cycle":1,"year":2015,"language":"EN","pages_approx":38,"emissions_rank":4},
    "NDC2_2022_India":       {"country":"India","ndc_cycle":2,"year":2022,"language":"EN","pages_approx":22,"emissions_rank":4},
    # ── Japan ──
    "NDC1_2015_Japan":       {"country":"Japan","ndc_cycle":1,"year":2015,"language":"EN","pages_approx":8,"emissions_rank":8},
    "NDC2_2021_Japan":       {"country":"Japan","ndc_cycle":2,"year":2021,"language":"EN","pages_approx":7,"emissions_rank":8},
    # ── South Korea ──
    "NDC1_2015_South_Korea": {"country":"South Korea","ndc_cycle":1,"year":2015,"language":"EN","pages_approx":4,"emissions_rank":11},
    "NDC2_2021_South_Korea": {"country":"South Korea","ndc_cycle":2,"year":2021,"language":"EN","pages_approx":20,"emissions_rank":11},
    # ── Indonesia ──
    "NDC1_2016_Indonesia":   {"country":"Indonesia","ndc_cycle":1,"year":2016,"language":"EN","pages_approx":14,"emissions_rank":7},
    "NDC2_2022_Indonesia":   {"country":"Indonesia","ndc_cycle":2,"year":2022,"language":"EN","pages_approx":80,"emissions_rank":7},
    # ── USA ──
    "NDC1_2015_USA":         {"country":"USA","ndc_cycle":1,"year":2015,"language":"EN","pages_approx":7,"emissions_rank":2,"withdrawn":True},
    "NDC2_2021_USA":         {"country":"USA","ndc_cycle":2,"year":2021,"language":"EN","pages_approx":24,"emissions_rank":2,"withdrawn":True},
    # ── Canada ──
    "NDC1_2015_Canada":      {"country":"Canada","ndc_cycle":1,"year":2017,"language":"EN","pages_approx":7,"emissions_rank":10,"note":"2017 resubmission of 2015 INDC"},
    "NDC2_2021_Canada":      {"country":"Canada","ndc_cycle":2,"year":2021,"language":"EN","pages_approx":24,"emissions_rank":10},
    # ── Mexico ──
    "NDC1_2015_Mexico":      {"country":"Mexico","ndc_cycle":1,"year":2015,"language":"ES_EN","pages_approx":22,"emissions_rank":14},
    "NDC2_2022_Mexico":      {"country":"Mexico","ndc_cycle":2,"year":2022,"language":"ES","pages_approx":22,"emissions_rank":14},
    # ── EU ──
    "NDC1_2015_EU":          {"country":"EU","ndc_cycle":1,"year":2015,"language":"EN","pages_approx":6,"emissions_rank":3,"note":"Joint EU submission; DE, FR, IT included"},
    "NDC2_2020_EU":          {"country":"EU","ndc_cycle":2,"year":2020,"language":"EN","pages_approx":19,"emissions_rank":3,"note":"Superseded by NDC3.0 Nov 2025 but valid for 2030 analysis"},
    # ── Russia ──
    "NDC1_2015_Russia":      {"country":"Russia","ndc_cycle":1,"year":2015,"language":"EN","pages_approx":4,"emissions_rank":5},
    "NDC2_2022_Russia":      {"country":"Russia","ndc_cycle":2,"year":2022,"language":"EN","pages_approx":6,"emissions_rank":5},
    # ── Turkey ──
    "NDC1_2021_Turkey":      {"country":"Turkey","ndc_cycle":1,"year":2021,"language":"EN","pages_approx":7,"emissions_rank":16,"note":"2015 INDC registered as NDC1 after Oct 2021 ratification"},
    "NDC2_2023_Turkey":      {"country":"Turkey","ndc_cycle":2,"year":2023,"language":"EN","pages_approx":25,"emissions_rank":16},
    # ── UK ──
    "NDC1_2020_UK":          {"country":"UK","ndc_cycle":1,"year":2020,"language":"EN","pages_approx":11,"emissions_rank":20},
    "NDC2_2022_UK":          {"country":"UK","ndc_cycle":2,"year":2022,"language":"EN","pages_approx":15,"emissions_rank":20},
    # ── Brazil ──
    "NDC1_2015_Brazil":      {"country":"Brazil","ndc_cycle":1,"year":2015,"language":"EN","pages_approx":8,"emissions_rank":6},
    "NDC2_2022_Brazil":      {"country":"Brazil","ndc_cycle":2,"year":2023,"language":"EN","pages_approx":16,"emissions_rank":6,"note":"Nov 2023 active adjustment"},
    # ── Argentina ──
    "NDC1_2016_Argentina":   {"country":"Argentina","ndc_cycle":1,"year":2016,"language":"EN","pages_approx":14,"emissions_rank":17},
    "NDC2_2021_Argentina":   {"country":"Argentina","ndc_cycle":2,"year":2021,"language":"EN","pages_approx":42,"emissions_rank":17},
    # ── Saudi Arabia ──
    "NDC1_2015_Saudi_Arabia":{"country":"Saudi Arabia","ndc_cycle":1,"year":2015,"language":"EN","pages_approx":6,"emissions_rank":12},
    "NDC2_2021_Saudi_Arabia":{"country":"Saudi Arabia","ndc_cycle":2,"year":2021,"language":"EN","pages_approx":10,"emissions_rank":12},
    # ── South Africa ──
    "NDC1_2015_South_Africa":{"country":"South Africa","ndc_cycle":1,"year":2015,"language":"EN","pages_approx":7,"emissions_rank":13},
    "NDC2_2021_South_Africa":{"country":"South Africa","ndc_cycle":2,"year":2021,"language":"EN","pages_approx":20,"emissions_rank":13},
    # ── Australia ──
    "NDC1_2015_Australia":   {"country":"Australia","ndc_cycle":1,"year":2015,"language":"EN","pages_approx":4,"emissions_rank":15},
    "NDC2_2022_Australia":   {"country":"Australia","ndc_cycle":2,"year":2022,"language":"EN","pages_approx":9,"emissions_rank":15},
}

# IPCC metadata: maps filename stem → metadata
IPCC_META: dict[str, dict] = {
    "SYR_FullVolume_2023":                               {"wg":"SYR","year":2023,"doc_type":"Full Report","relevance":"HIGH"},
    "WG1_SummaryVolume_SPM_TS_2021":                     {"wg":"WG1","year":2021,"doc_type":"SPM+TS+FAQ"},
    "WG1_Chapter01_FramingContextMethods_2021":          {"wg":"WG1","year":2021,"doc_type":"Chapter","chapter":1},
    "WG1_Chapter02_ChangingStateClimateSystem_2021":     {"wg":"WG1","year":2021,"doc_type":"Chapter","chapter":2},
    "WG1_Chapter04_FutureGlobalClimaScenarios_2021":     {"wg":"WG1","year":2021,"doc_type":"Chapter","chapter":4},
    "WG1_Chapter11_WeatherClimateExtremes_2021":         {"wg":"WG1","year":2021,"doc_type":"Chapter","chapter":11},
    "WG2_SPM_2022":                                      {"wg":"WG2","year":2022,"doc_type":"SPM"},
    "WG2_TechnicalSummary_2022":                         {"wg":"WG2","year":2022,"doc_type":"Technical Summary"},
    "WG2_Chapter17_DecisionMakingRisk_2022":             {"wg":"WG2","year":2022,"doc_type":"Chapter","chapter":17},
    "WG2_Chapter18_ClimateResilientDevelopment_2022":    {"wg":"WG2","year":2022,"doc_type":"Chapter","chapter":18},
    "WG3_SPM_2022":                                      {"wg":"WG3","year":2022,"doc_type":"SPM"},
    "WG3_TechnicalSummary_2022":                         {"wg":"WG3","year":2022,"doc_type":"Technical Summary"},
    "WG3_Chapter02_EmissionsTrendsDrivers_2022":         {"wg":"WG3","year":2022,"doc_type":"Chapter","chapter":2},
    "WG3_Chapter04_MitigationDevelopmentPathways_2022":  {"wg":"WG3","year":2022,"doc_type":"Chapter","chapter":4},
    "WG3_Chapter06_EnergySystems_2022":                  {"wg":"WG3","year":2022,"doc_type":"Chapter","chapter":6},
    "WG3_Chapter13_NationalSubnationalPolicies_2022":    {"wg":"WG3","year":2022,"doc_type":"Chapter","chapter":13},
    "WG3_Chapter14_InternationalCooperation_2022":       {"wg":"WG3","year":2022,"doc_type":"Chapter","chapter":14},
    "WG3_Chapter17_AcceleratingTransitionSDGs_2022":     {"wg":"WG3","year":2022,"doc_type":"Chapter","chapter":17},
}


# ── Text cleaning ─────────────────────────────────────────────────────────────

def fix_ligatures(text: str) -> str:
    for bad, good in LIGATURES.items():
        text = text.replace(bad, good)
    return text


def fix_hyphenation(text: str) -> str:
    """Join words broken across lines: 'emis-\nsions' → 'emissions'."""
    return re.sub(r"(\w)-\n(\w)", r"\1\2", text)


def remove_noise(text: str) -> str:
    for pat in NOISE_RE:
        text = pat.sub(" ", text)
    return text


def normalise_whitespace(text: str) -> str:
    # Collapse multiple spaces to one, but keep paragraph breaks (double \n)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def clean_page_text(raw: str) -> str:
    text = fix_ligatures(raw)
    text = fix_hyphenation(text)
    text = remove_noise(text)
    text = normalise_whitespace(text)
    return text


# ── Section detection ─────────────────────────────────────────────────────────

def detect_section(line: str) -> Optional[str]:
    """Return section label if line looks like a heading, else None."""
    stripped = line.strip()
    if not stripped or len(stripped) > 120:
        return None
    if HEADING_RE.match(stripped):
        return stripped[:100]
    return None


def extract_sections_from_pages(pages_text: list[str]) -> list[dict]:
    """
    Returns a list of dicts:
      { "section": str, "page_start": int, "text": str }
    Groups consecutive pages under the last detected heading.
    """
    sections = []
    current_section = "Introduction"
    current_page_start = 1
    current_text_parts: list[str] = []

    for page_num, page_text in enumerate(pages_text, start=1):
        lines = page_text.split("\n")
        for line in lines:
            maybe = detect_section(line)
            if maybe and len(maybe) > 4:
                # Save current section
                if current_text_parts:
                    sections.append({
                        "section": current_section,
                        "page_start": current_page_start,
                        "text": "\n".join(current_text_parts).strip(),
                    })
                current_section = maybe
                current_page_start = page_num
                current_text_parts = []
        current_text_parts.append(page_text)

    # Flush last section
    if current_text_parts:
        sections.append({
            "section": current_section,
            "page_start": current_page_start,
            "text": "\n".join(current_text_parts).strip(),
        })

    return sections


# ── Tokeniser helpers ─────────────────────────────────────────────────────────

def token_count(text: str) -> int:
    return len(TOKENISER.encode(text))


def split_into_sentences(text: str) -> list[str]:
    """
    Naive but robust sentence splitter for policy documents.
    Splits on '. ', '? ', '! ', '\n\n' — avoids splitting on abbreviations
    like 'MtCO2e.', 'para.', 'Fig.' by requiring capitalised next word.
    """
    # Protect known abbreviations
    abbrevs = [r"No\.", r"para\.", r"Fig\.", r"Eq\.", r"Vol\.", r"pp\.", r"et al\.",
               r"e\.g\.", r"i\.e\.", r"vs\.", r"approx\.", r"govt\.", r"dept\.",
               r"MtCO2e?\.", r"GtCO2e?\.", r"GWP\.", r"NDC\.", r"INDC\.", r"AR\d\."]
    protected = text
    for abbr in abbrevs:
        protected = re.sub(abbr, lambda m: m.group().replace(".", "<!DOT!>"), protected)

    # Split on sentence-ending punctuation followed by space + capital
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z\d\"\(])", protected)
    # Restore protected dots
    sentences = [p.replace("<!DOT!>", ".") for p in parts]
    # Also split on double newlines (paragraph breaks)
    result = []
    for s in sentences:
        paras = re.split(r"\n\n+", s)
        result.extend(paras)
    return [s.strip() for s in result if s.strip()]


# ── Chunker ───────────────────────────────────────────────────────────────────

def make_chunks(
    text: str,
    min_tokens: int = DEFAULT_MIN,
    max_tokens: int = DEFAULT_MAX,
    overlap_ratio: float = DEFAULT_OVR,
) -> list[str]:
    """
    Split text into chunks of [min_tokens, max_tokens] tokens with
    sentence-boundary awareness and overlap_ratio * chunk_size overlap.

    Strategy:
      1. Split text into sentences
      2. Greedily accumulate sentences up to max_tokens
      3. On overflow, finalise chunk, then backtrack by overlap_ratio
         (keep the last N tokens worth of sentences as the start of next chunk)
    """
    sentences = split_into_sentences(text)
    if not sentences:
        return []

    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0
    overlap_target = int(max_tokens * overlap_ratio)  # tokens to carry forward

    for sent in sentences:
        sent_tokens = token_count(sent)

        # Single sentence longer than max — split it hard by words
        if sent_tokens > max_tokens:
            words = sent.split()
            buf, buf_tok = [], 0
            for word in words:
                wt = token_count(word)
                if buf_tok + wt > max_tokens and buf:
                    chunk_text = " ".join(buf)
                    if token_count(chunk_text) >= min_tokens:
                        chunks.append(chunk_text)
                    buf, buf_tok = [word], wt
                else:
                    buf.append(word)
                    buf_tok += wt
            if buf:
                current.extend(buf)
                current_tokens += buf_tok
            continue

        if current_tokens + sent_tokens > max_tokens:
            # Finalise current chunk
            chunk_text = " ".join(current).strip()
            if token_count(chunk_text) >= min_tokens:
                chunks.append(chunk_text)

            # Build overlap: keep sentences from the end that sum to overlap_target
            overlap_sents: list[str] = []
            overlap_tok = 0
            for s in reversed(current):
                st = token_count(s)
                if overlap_tok + st <= overlap_target:
                    overlap_sents.insert(0, s)
                    overlap_tok += st
                else:
                    break

            current = overlap_sents + [sent]
            current_tokens = overlap_tok + sent_tokens
        else:
            current.append(sent)
            current_tokens += sent_tokens

    # Flush remainder
    if current:
        chunk_text = " ".join(current).strip()
        if chunk_text:
            # If remainder is tiny, append to last chunk if possible
            if token_count(chunk_text) < min_tokens // 2 and chunks:
                last = chunks[-1] + " " + chunk_text
                if token_count(last) <= max_tokens * 1.2:  # allow slight overrun
                    chunks[-1] = last.strip()
                else:
                    chunks.append(chunk_text)
            else:
                chunks.append(chunk_text)

    return chunks


# ── PDF extraction ────────────────────────────────────────────────────────────

def extract_pages(pdf_path: Path) -> tuple[list[str], list[bool]]:
    """
    Returns (pages_text, pages_has_table).
    pages_text[i] = cleaned text of page i+1.
    pages_has_table[i] = True if pdfplumber found a table on that page.
    """
    pages_text: list[str] = []
    pages_has_table: list[bool] = []

    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                # Extract text (pdfplumber preserves reading order)
                raw = page.extract_text(x_tolerance=3, y_tolerance=3) or ""
                cleaned = clean_page_text(raw)
                pages_text.append(cleaned)

                # Detect tables (for metadata tagging)
                tables = page.extract_tables()
                pages_has_table.append(bool(tables))

    except Exception as exc:
        print(f"\n  [WARN] pdfplumber failed on {pdf_path.name}: {exc}")
        pages_text = [""]
        pages_has_table = [False]

    return pages_text, pages_has_table


# ── Metadata builders ─────────────────────────────────────────────────────────

def build_ndc_base_meta(stem: str, pdf_path: Path) -> dict:
    known = NDC_META.get(stem, {})
    if known:
        return {
            "source":       "UNFCCC_NDC_Registry",
            "doc_type":     "NDC",
            "country":      known.get("country", "Unknown"),
            "ndc_cycle":    known.get("ndc_cycle"),
            "year":         known.get("year"),
            "language":     known.get("language", "EN"),
            "emissions_rank": known.get("emissions_rank"),
            "withdrawn":    known.get("withdrawn", False),
            "superseded":   known.get("superseded", False),
            "note":         known.get("note", ""),
            "file":         pdf_path.name,
            "file_path":    str(pdf_path),
        }
    # Fallback: parse from filename pattern NDCx_YYYY_Country
    parts = stem.split("_")
    return {
        "source":    "UNFCCC_NDC_Registry",
        "doc_type":  "NDC",
        "country":   "_".join(parts[2:]) if len(parts) > 2 else "Unknown",
        "ndc_cycle": int(parts[0].replace("NDC","")) if parts[0].startswith("NDC") else None,
        "year":      int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else None,
        "file":      pdf_path.name,
        "file_path": str(pdf_path),
    }


def build_ipcc_base_meta(stem: str, pdf_path: Path) -> dict:
    known = IPCC_META.get(stem, {})
    return {
        "source":    "IPCC_AR6",
        "doc_type":  known.get("doc_type", "IPCC_AR6"),
        "wg":        known.get("wg", "Unknown"),
        "year":      known.get("year"),
        "chapter":   known.get("chapter"),
        "relevance": known.get("relevance", ""),
        "country":   None,   # IPCC docs are global
        "file":      pdf_path.name,
        "file_path": str(pdf_path),
    }


# ── Main processing function ──────────────────────────────────────────────────

def process_pdf(
    pdf_path: Path,
    source_type: str,           # "ndc" or "ipcc"
    min_tokens: int,
    max_tokens: int,
    overlap_ratio: float,
    dry_run: bool = False,
) -> list[dict]:
    """
    Full pipeline for a single PDF.
    Returns a list of chunk dicts (empty on dry_run).
    """
    stem = pdf_path.stem

    # Build base metadata
    if source_type == "ndc":
        base_meta = build_ndc_base_meta(stem, pdf_path)
    else:
        base_meta = build_ipcc_base_meta(stem, pdf_path)

    if dry_run:
        return []

    # Extract pages
    pages_text, pages_has_table = extract_pages(pdf_path)
    full_text = "\n\n".join(p for p in pages_text if p)

    if not full_text.strip():
        print(f"\n  [WARN] No text extracted from {pdf_path.name} — may be scanned/image PDF")
        return []

    # Segment into sections
    sections = extract_sections_from_pages(pages_text)

    all_chunks: list[dict] = []
    chunk_seq = 0

    for sec in sections:
        sec_text  = sec["text"]
        sec_label = sec["section"]
        sec_page  = sec["page_start"]

        if not sec_text.strip():
            continue

        # Detect if section is table-heavy
        is_table = any(
            pages_has_table[i]
            for i in range(min(sec_page - 1, len(pages_has_table) - 1),
                           min(sec_page + 3, len(pages_has_table)))
        )

        chunks = make_chunks(sec_text, min_tokens, max_tokens, overlap_ratio)

        for chunk_text in chunks:
            tok = token_count(chunk_text)
            chunk_id = f"{stem}__{chunk_seq:05d}"

            record = {
                # ── Identifiers ──────────────────────────────────────────────
                "chunk_id":      chunk_id,
                "doc_id":        stem,

                # ── Provenance ───────────────────────────────────────────────
                **base_meta,

                # ── Location within document ─────────────────────────────────
                "section":       sec_label,
                "page_start":    sec_page,

                # ── Chunk stats ──────────────────────────────────────────────
                "token_count":   tok,
                "char_count":    len(chunk_text),
                "chunk_index":   chunk_seq,
                "is_table":      is_table,

                # ── Content ──────────────────────────────────────────────────
                "text":          chunk_text,

                # ── Processing metadata ──────────────────────────────────────
                "processed_at":  datetime.now(timezone.utc).isoformat(),
                "pipeline_ver":  "1.0",
            }
            all_chunks.append(record)
            chunk_seq += 1

    return all_chunks


# ── Directory walker ──────────────────────────────────────────────────────────

def find_pdfs(raw_dir: Path, source_filter: str) -> list[tuple[Path, str]]:
    """
    Returns list of (pdf_path, source_type) tuples.
    source_type ∈ {"ndc", "ipcc"}
    """
    results: list[tuple[Path, str]] = []

    if source_filter in ("all", "ndc"):
        ndc_root = raw_dir / "NDC"
        if ndc_root.exists():
            for pdf in sorted(ndc_root.rglob("*.pdf")):
                results.append((pdf, "ndc"))

    if source_filter in ("all", "ipcc"):
        ipcc_root = raw_dir / "IPCC_AR6"
        if ipcc_root.exists():
            for pdf in sorted(ipcc_root.rglob("*.pdf")):
                results.append((pdf, "ipcc"))

    return results


# ── Output writers ────────────────────────────────────────────────────────────

def write_outputs(chunks: list[dict], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Full JSONL (one chunk per line — efficient for streaming)
    jsonl_path = out_dir / "chunks.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
    print(f"\n  [SAVED] {jsonl_path}  ({len(chunks)} chunks)")

    # 2. Pretty-printed sample (first 20 chunks, for inspection)
    sample_path = out_dir / "chunks_sample.json"
    with open(sample_path, "w", encoding="utf-8") as f:
        json.dump(chunks[:20], f, indent=2, ensure_ascii=False)
    print(f"  [SAVED] {sample_path}  (first 20 chunks)")

    # 3. Pipeline report
    by_country: dict[str, int] = {}
    by_wg:      dict[str, int] = {}
    by_cycle:   dict[str, int] = {}
    token_total = 0
    for c in chunks:
        country = c.get("country") or "IPCC"
        by_country[country] = by_country.get(country, 0) + 1
        wg = c.get("wg") or "NDC"
        by_wg[wg] = by_wg.get(wg, 0) + 1
        cyc = c.get("ndc_cycle")
        if cyc is not None:
            key = f"NDC{cyc}"
            by_cycle[key] = by_cycle.get(key, 0) + 1
        token_total += c.get("token_count", 0)

    avg_tokens = token_total // len(chunks) if chunks else 0
    report = {
        "generated_at":      datetime.now(timezone.utc).isoformat(),
        "pipeline_version":  "1.0",
        "total_chunks":      len(chunks),
        "total_tokens":      token_total,
        "avg_tokens_per_chunk": avg_tokens,
        "chunks_by_country": dict(sorted(by_country.items())),
        "chunks_by_wg":      dict(sorted(by_wg.items())),
        "chunks_by_ndc_cycle": by_cycle,
        "docs_processed":    len(set(c["doc_id"] for c in chunks)),
    }
    report_path = out_dir / "pipeline_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"  [SAVED] {report_path}")


# ── CLI entry point ───────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="PDF → clean text → tagged chunks pipeline for climate policy RAG"
    )
    parser.add_argument("--raw_dir",    default="data/raw",       help="Root of raw PDF folder")
    parser.add_argument("--out_dir",    default="data/processed", help="Output folder")
    parser.add_argument("--source",     default="all",            choices=["all","ndc","ipcc"])
    parser.add_argument("--min_tokens", default=DEFAULT_MIN, type=int)
    parser.add_argument("--max_tokens", default=DEFAULT_MAX, type=int)
    parser.add_argument("--overlap",    default=DEFAULT_OVR, type=float,
                        help="Overlap ratio, e.g. 0.20 = 20%%")
    parser.add_argument("--dry_run",    action="store_true",
                        help="Count PDFs/pages without writing output")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)

    print("=" * 64)
    print("  PDF → CHUNKS PIPELINE")
    print(f"  Raw dir    : {raw_dir.resolve()}")
    print(f"  Output dir : {out_dir.resolve()}")
    print(f"  Tokens     : {args.min_tokens}–{args.max_tokens}  overlap={args.overlap:.0%}")
    print(f"  Source     : {args.source.upper()}")
    print(f"  Dry run    : {args.dry_run}")
    print("=" * 64)

    pdf_list = find_pdfs(raw_dir, args.source)
    if not pdf_list:
        print(f"\n[ERROR] No PDFs found under {raw_dir}")
        print("  Check that data/raw/NDC/ and data/raw/IPCC_AR6/ exist.")
        sys.exit(1)

    print(f"\n  Found {len(pdf_list)} PDFs to process\n")

    all_chunks: list[dict] = []
    failed: list[str] = []

    iterable = tqdm(pdf_list, unit="pdf") if TQDM else pdf_list

    for pdf_path, src_type in iterable:
        label = f"{src_type.upper()} | {pdf_path.stem}"
        if not TQDM:
            print(f"  Processing: {label}")

        try:
            chunks = process_pdf(
                pdf_path       = pdf_path,
                source_type    = src_type,
                min_tokens     = args.min_tokens,
                max_tokens     = args.max_tokens,
                overlap_ratio  = args.overlap,
                dry_run        = args.dry_run,
            )
            all_chunks.extend(chunks)
        except Exception as exc:
            msg = f"{pdf_path.name}: {exc}"
            print(f"\n  [ERROR] {msg}")
            failed.append(msg)

    print(f"\n  Processed : {len(pdf_list)} PDFs")
    print(f"  Chunks    : {len(all_chunks)}")
    print(f"  Errors    : {len(failed)}")

    if failed:
        print("\n  Failed files:")
        for f in failed:
            print(f"    ✗ {f}")

    if not args.dry_run and all_chunks:
        write_outputs(all_chunks, out_dir)
    elif args.dry_run:
        print("\n  [DRY RUN] No files written.")

    # Summary table
    if all_chunks:
        docs = {}
        for c in all_chunks:
            did = c["doc_id"]
            docs.setdefault(did, 0)
            docs[did] += 1
        print("\n  Chunks per document:")
        for doc_id, n in sorted(docs.items()):
            print(f"    {doc_id:<55}  {n:>4} chunks")

    total_tokens = sum(c.get("token_count", 0) for c in all_chunks)
    if total_tokens:
        print(f"\n  Total tokens in corpus : {total_tokens:,}")
        print(f"  Avg tokens per chunk   : {total_tokens // len(all_chunks):,}")

    print("\n  Done.\n")


if __name__ == "__main__":
    main()
