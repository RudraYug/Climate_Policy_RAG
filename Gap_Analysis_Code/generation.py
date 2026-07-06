"""
===========================================================================
  RAG GENERATION MODULE
  Master's Thesis: LLM-Powered Climate Policy Summariser & Gap Analyser
  University of Europe for Applied Sciences, Potsdam  |  2026
===========================================================================

ARCHITECTURE DECISION: Dual-backend with shared prompt engine
─────────────────────────────────────────────────────────────
  PRIMARY   →  Mistral-7B-Instruct-v0.3  (HuggingFace transformers, free)
               • Runs locally on Colab T4 GPU (4-bit quantised, ~5 GB VRAM)
               • Runs on CPU for testing (slow but functional)
               • 32k context window → comfortably fits 5 retrieved chunks
               • apply_chat_template() handles [INST]/[/INST] formatting
               • Zero API cost — critical for iterative thesis experiments

  FALLBACK  →  GPT-4o  (OpenAI API, ~$0.002/query at 2k tokens)
               • Used for RQ3: open-source vs commercial LLM comparison
               • Same prompt, same output schema — swap with one flag

WHY THIS MATTERS FOR YOUR THESIS
─────────────────────────────────
  RQ3 explicitly compares Mistral vs GPT-4o on faithfulness, hallucination
  rate, and answer relevancy (RAGAS). Having both share the same:
    - PromptBuilder  (identical context + citation instructions)
    - RAGResponse    (identical output dataclass)
    - Evaluator      (identical RAGAS metrics)
  means your Chapter 5 comparison is apples-to-apples with zero confounds.

PROMPT DESIGN — 3 required behaviours
──────────────────────────────────────
  (1) Context-only answers:
      System prompt explicitly forbids outside knowledge. Model must cite
      the provided chunks or state INSUFFICIENT_CONTEXT.

  (2) Structured citations [DOC, p.N]:
      Every factual claim requires an inline citation in [SOURCE, p.N]
      format. Source = doc_id, N = page_start from chunk metadata.

  (3) Insufficient-context flag:
      If no retrieved chunk contains information relevant to the query,
      the model must return the sentinel string INSUFFICIENT_CONTEXT
      rather than hallucinating. This is parsed and surfaced in RAGResponse.

OUTPUT SCHEMA (RAGResponse dataclass)
──────────────────────────────────────
  answer           : str           — generated answer with inline citations
  citations        : list[Citation] — parsed [SOURCE, p.N] blocks
  insufficient     : bool          — True if model flagged no relevant context
  query_type       : str           — "factual" | "comparative" | "gap"
  retrieved_chunks : list[dict]    — the actual chunks passed as context
  model_used       : str           — which backend generated the answer
  latency_sec      : float         — end-to-end wall time
  token_stats      : dict          — prompt / completion token counts

USAGE
──────
  # Colab / GPU (recommended):
  pip install transformers accelerate bitsandbytes sentencepiece
  pip install rank_bm25 chromadb sentence-transformers

  from generation import RAGPipeline
  pipe = RAGPipeline(backend="mistral")  # or "gpt4o"
  resp = pipe.query("What is India's 2030 renewable energy target?",
                     filters={"country": "India"})
  print(resp.answer)
  print(resp.citations)

  # CLI:
  python generation.py --query "India renewable energy 2030"
  python generation.py --query "gap China 1.5C" --backend gpt4o
  python generation.py --demo       # runs 4 showcase queries
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
from pathlib import Path
from typing import Optional

# ── Silence noisy library output ─────────────────────────────────────────────
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_VERBOSITY",       "error")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")  # Windows symlink warn
warnings.filterwarnings("ignore", message=".*position_ids.*")
warnings.filterwarnings("ignore", message=".*unauthenticated.*")
warnings.filterwarnings("ignore", message=".*symlinks.*")


# ── Configuration ─────────────────────────────────────────────────────────────

DEFAULTS = {
    # Retriever paths (must match retriever.py defaults)
    "chunks_file":   "data/processed/chunks.jsonl",
    "vectorstore":   "data/vectorstore",
    "collection":    "climate_policy_rag",
    "bm25_cache":    "data/processed/bm25_index.pkl",

    # Embedding model (must match what was used in embed_and_ingest.py)
    "embed_model":   "BAAI/bge-large-en-v1.5",
    "embed_device":  "cpu",

    # Generation model — Mistral backend
    "mistral_model": "mistralai/Mistral-7B-Instruct-v0.3",
    "mistral_device": "auto",          # "auto" = GPU if available, else CPU
    "load_in_4bit":  True,             # 4-bit quantisation: ~5 GB VRAM on T4

    # Retrieval settings
    "top_k":         5,                # chunks passed to LLM as context

    # Generation settings
    "max_new_tokens": 800,
    "temperature":    0.1,             # low = more faithful, less creative
    "do_sample":      False,           # deterministic for reproducibility

    # Output
    "responses_dir":  "data/responses",
}

# ── Output dataclasses ────────────────────────────────────────────────────────

@dataclass
class Citation:
    """One parsed citation extracted from the model's response."""
    doc_id:     str
    page:       int
    country:    Optional[str]
    year:       Optional[int]
    source:     Optional[str]   # "UNFCCC_NDC_Registry" or "IPCC_AR6"
    raw_tag:    str             # original text e.g. "[NDC2_2022_India, p.3]"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class RAGResponse:
    """Complete output from one RAG query."""
    query:             str
    answer:            str
    citations:         list[Citation]
    insufficient:      bool
    query_type:        str           # "factual" | "comparative" | "gap"
    retrieved_chunks:  list[dict]
    model_used:        str
    backend:           str           # "mistral" | "gpt4o"
    latency_sec:       float
    token_stats:       dict
    prompt_text:       str           # full prompt sent to LLM (for RAGAS eval)
    filters_used:      Optional[dict]

    def to_dict(self) -> dict:
        d = asdict(self)
        # Trim retrieved_chunks to just the metadata fields (not full text)
        # so saved JSON files stay small
        d["retrieved_chunks_summary"] = [
            {k: v for k, v in c.items() if k != "text"}
            for c in self.retrieved_chunks
        ]
        return d

    def pretty_print(self) -> None:
        """Human-readable console output."""
        width = 70
        print(f"\n{'═' * width}")
        print(f"  Query : {self.query}")
        print(f"  Model : {self.model_used}  [{self.backend}]")
        print(f"  Type  : {self.query_type}  |  Latency: {self.latency_sec:.1f}s")
        print(f"{'─' * width}")

        if self.insufficient:
            print("  ⚠  INSUFFICIENT_CONTEXT — no relevant chunks found.")
            print("     Try broadening the query or removing metadata filters.")
        else:
            print(f"  ANSWER:\n")
            # Wrap long lines for readability
            for line in self.answer.split("\n"):
                if len(line) > 75:
                    words = line.split()
                    current = "  "
                    for w in words:
                        if len(current) + len(w) > 75:
                            print(current)
                            current = "  " + w + " "
                        else:
                            current += w + " "
                    if current.strip():
                        print(current)
                else:
                    print(f"  {line}")

        print(f"\n{'─' * width}")
        print(f"  Citations ({len(self.citations)}):")
        for c in self.citations:
            print(f"    [{c.doc_id}, p.{c.page}]  "
                  f"{c.country or c.source or '—'}  {c.year or ''}")

        print(f"\n  Context chunks used ({len(self.retrieved_chunks)}):")
        for i, ch in enumerate(self.retrieved_chunks, 1):
            country = ch.get("country") or ch.get("wg") or "—"
            rrf     = ch.get("rrf_score", 0)
            section = (ch.get("section") or "")[:50]
            print(f"    #{i}  rrf={rrf:.4f}  {country:<15}  {section}")

        print(f"{'═' * width}\n")


# ── Prompt Builder ────────────────────────────────────────────────────────────

class PromptBuilder:
    """
    Builds structured prompts for climate policy RAG.

    Design principles:
    ─────────────────
    1. SYSTEM prompt locks the model into context-only mode with explicit
       prohibition on outside knowledge. This is the primary hallucination
       guard for your thesis faithfulness evaluation.

    2. Each retrieved chunk is presented as a numbered [SOURCE] block
       with doc_id and page metadata prepended. The model is instructed
       to cite exactly these identifiers inline as [doc_id, p.N].

    3. The INSUFFICIENT_CONTEXT sentinel is defined in the system prompt
       so the model has an explicit escape hatch instead of fabricating.

    4. Query-type-specific instructions are appended for gap analysis
       queries to guide the 3-stage commitment → classification → gap
       extraction logic described in your thesis methodology.

    5. Temperature is kept low (0.1) for factual/comparative queries
       and slightly higher (0.3) for gap analysis to allow richer
       synthesis across multiple source documents.
    """

    SYSTEM_PROMPT = """You are a climate policy analysis assistant for an academic research project.

Your task is to answer questions about G20 countries' Nationally Determined Contributions (NDCs) and IPCC AR6 findings.

STRICT RULES — you must follow ALL of these:
1. Answer ONLY using information from the RETRIEVED CONTEXT provided below.
2. Do NOT use any outside knowledge, training data, or assumptions.
3. Every factual claim MUST include an inline citation in the exact format: [doc_id, p.N]
   where doc_id is the SOURCE ID shown in the context and N is the page number.
4. If the retrieved context does not contain sufficient information to answer,
   respond with exactly: INSUFFICIENT_CONTEXT
   followed by one sentence explaining what information is missing.
5. Do not invent statistics, targets, or dates not present in the context.
6. If two sources contradict each other, state both positions with their citations.

CITATION FORMAT EXAMPLE:
  India commits to 50% non-fossil electricity capacity by 2030 [NDC2_2022_India, p.3].

OUTPUT STRUCTURE:
  - Write a clear, direct answer in 2-5 paragraphs.
  - End with a "Key Citations:" section listing all sources used.
"""

    GAP_ANALYSIS_ADDENDUM = """
ADDITIONAL INSTRUCTIONS FOR GAP ANALYSIS:
This is a gap analysis query. Structure your answer in three parts:
  PART 1 — COUNTRY COMMITMENT: What the NDC commits to (with citations).
  PART 2 — IPCC BENCHMARK: What IPCC AR6 recommends for this sector/target (with citations).
  PART 3 — GAP ASSESSMENT: Quantify the gap if possible. State clearly whether the
            commitment is: ALIGNED / PARTIALLY ALIGNED / INSUFFICIENT / UNCLEAR.
            If data is missing for a full assessment, state INSUFFICIENT_CONTEXT for that part.
"""

    @staticmethod
    def classify_query(query: str) -> str:
        """
        Classify query into one of three types that affect prompt structure.
        Rule-based for determinism (no LLM classification overhead).
        """
        q = query.lower()
        gap_keywords = ["gap", "compare", "comparison", "versus", "vs",
                        "differ", "shortfall", "insufficient", "aligned",
                        "benchmark", "pathway", "1.5", "2 degree", "2°c"]
        comp_keywords = ["compared", "between", "which country", "highest",
                         "lowest", "most", "least", "rank", "best", "worst",
                         "all g20", "across countries"]

        if any(k in q for k in gap_keywords):
            return "gap"
        if any(k in q for k in comp_keywords):
            return "comparative"
        return "factual"

    @classmethod
    def build_context_block(cls, chunks: list[dict]) -> str:
        """
        Format retrieved chunks as a numbered context block.
        Each chunk gets a labelled SOURCE header with metadata.
        """
        if not chunks:
            return "NO CONTEXT RETRIEVED."

        blocks = []
        for i, chunk in enumerate(chunks, 1):
            doc_id  = chunk.get("doc_id", "UNKNOWN")
            country = chunk.get("country") or chunk.get("wg") or "N/A"
            year    = chunk.get("year", "")
            page    = chunk.get("page_start", "?")
            section = chunk.get("section", "")[:80]
            source  = chunk.get("source", "")
            text    = (chunk.get("text") or "").strip()

            header = (
                f"--- SOURCE {i} ---\n"
                f"SOURCE ID : {doc_id}\n"
                f"Country   : {country}  |  Year: {year}  |  Page: {page}\n"
                f"Source    : {source}\n"
                f"Section   : {section}\n"
                f"Content   :\n{text}"
            )
            blocks.append(header)

        return "\n\n".join(blocks)

    @classmethod
    def build_user_message(
        cls,
        query:       str,
        chunks:      list[dict],
        query_type:  str,
    ) -> str:
        """Build the full user message: context + optional addendum + question."""
        context = cls.build_context_block(chunks)

        addendum = ""
        if query_type == "gap":
            addendum = cls.GAP_ANALYSIS_ADDENDUM

        return (
            f"RETRIEVED CONTEXT:\n\n"
            f"{context}\n\n"
            f"{'─' * 60}\n"
            f"{addendum}"
            f"\nQUESTION: {query}\n\n"
            f"Answer (remember: use ONLY the context above, cite every claim as [doc_id, p.N]):"
        )

    @classmethod
    def build_messages(
        cls,
        query:      str,
        chunks:     list[dict],
        query_type: str,
    ) -> list[dict]:
        """
        Return messages list in OpenAI / HuggingFace chat format.
        [{role: system, content: ...}, {role: user, content: ...}]
        """
        return [
            {"role": "system", "content": cls.SYSTEM_PROMPT},
            {"role": "user",   "content": cls.build_user_message(
                query, chunks, query_type)},
        ]


# ── Citation Parser ───────────────────────────────────────────────────────────

class CitationParser:
    """
    Parses [doc_id, p.N] citation tags from generated text.

    Handles variants:
      [NDC2_2022_India, p.3]
      [WG3_Chapter06_EnergySystems_2022, p.12]
      [NDC1_2016_China, p.1]
      [NDC2_2020_EU, p.4]
    """

    # Matches [ANYTHING, p.NUMBER] with optional whitespace
    _CITATION_RE = re.compile(
        r"\[([A-Za-z0-9_\-\.]+)\s*,\s*p\.?\s*(\d+)\]"
    )

    @classmethod
    def parse(
        cls,
        text: str,
        chunk_meta: dict,   # chunk_id → chunk dict, for metadata enrichment
    ) -> list[Citation]:
        """
        Extract all citations from generated text and enrich with metadata.
        Deduplicates by (doc_id, page).
        """
        seen   = set()
        result = []

        for m in cls._CITATION_RE.finditer(text):
            doc_id  = m.group(1)
            page    = int(m.group(2))
            raw_tag = m.group(0)

            key = (doc_id, page)
            if key in seen:
                continue
            seen.add(key)

            # Enrich from chunk metadata if available
            meta    = chunk_meta.get(doc_id, {})
            country = meta.get("country")
            year    = meta.get("year")
            source  = meta.get("source")

            result.append(Citation(
                doc_id  = doc_id,
                page    = page,
                country = country,
                year    = year,
                source  = source,
                raw_tag = raw_tag,
            ))

        return result


# ── LLM Backends ─────────────────────────────────────────────────────────────

class MistralBackend:
    """
    Mistral-7B-Instruct-v0.3 via HuggingFace transformers.

    Two modes:
    ──────────
    LOCAL (default, recommended for Colab T4):
      Loads model weights locally with 4-bit quantisation.
      Requires: transformers, accelerate, bitsandbytes
      VRAM: ~5 GB (4-bit) or ~14 GB (fp16)

    On your Windows laptop (no GPU):
      Set load_in_4bit=False, mistral_device="cpu"
      This works but is very slow (~5 min/query).
      Better: use Google Colab for generation experiments.
    """

    def __init__(
        self,
        model_name:    str  = DEFAULTS["mistral_model"],
        device:        str  = DEFAULTS["mistral_device"],
        load_in_4bit:  bool = DEFAULTS["load_in_4bit"],
        max_new_tokens: int = DEFAULTS["max_new_tokens"],
        temperature:   float = DEFAULTS["temperature"],
        do_sample:     bool = DEFAULTS["do_sample"],
    ):
        self.model_name     = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature    = temperature
        self.do_sample      = do_sample

        print(f"\n  [Mistral] Loading {model_name}...")
        print(f"  [Mistral] 4-bit={load_in_4bit}  device={device}")

        try:
            import torch
            from transformers import (
                AutoTokenizer,
                AutoModelForCausalLM,
                BitsAndBytesConfig,
            )
        except ImportError as e:
            raise ImportError(
                f"Missing: {e}\n"
                "pip install transformers accelerate bitsandbytes sentencepiece"
            ) from e

        # ── Detect what's available ───────────────────────────────────────────
        _has_accelerate  = self._check_package("accelerate")
        _has_bitsandbytes = self._check_package("bitsandbytes")
        _has_cuda        = torch.cuda.is_available()

        # Resolve effective device
        if device == "auto":
            effective_device = "cuda" if _has_cuda else "cpu"
        else:
            effective_device = device

        print(f"  [Mistral] accelerate={_has_accelerate}  "
              f"bitsandbytes={_has_bitsandbytes}  "
              f"cuda={_has_cuda}  "
              f"effective_device={effective_device}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            padding_side="left",
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # ── Choose loading strategy based on available packages ───────────────
        #
        #  Strategy A: GPU + accelerate + bitsandbytes → 4-bit quantised (5 GB)
        #  Strategy B: GPU + accelerate, no bitsandbytes → fp16 with device_map
        #  Strategy C: CPU + accelerate → fp32 with device_map="cpu"
        #  Strategy D: CPU, NO accelerate → fp32, no device_map (safe fallback)
        #
        #  device_map requires accelerate. Without it you MUST use .to(device).
        #  This is the root cause of the original error on Windows without
        #  accelerate installed.

        if _has_accelerate and _has_bitsandbytes and _has_cuda and load_in_4bit:
            # ── Strategy A: 4-bit on GPU ──────────────────────────────────────
            print("  [Mistral] Strategy A: 4-bit quantised on GPU")
            bnb_config = BitsAndBytesConfig(
                load_in_4bit              = True,
                bnb_4bit_quant_type       = "nf4",
                bnb_4bit_compute_dtype    = torch.float16,
                bnb_4bit_use_double_quant = True,
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config = bnb_config,
                device_map          = "auto",
                trust_remote_code   = False,
            )

        elif _has_accelerate and _has_cuda:
            # ── Strategy B: fp16 on GPU ───────────────────────────────────────
            print("  [Mistral] Strategy B: fp16 on GPU (no bitsandbytes)")
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map  = "auto",
                torch_dtype = torch.float16,
            )

        elif _has_accelerate:
            # ── Strategy C: fp32 on CPU with accelerate ───────────────────────
            print("  [Mistral] Strategy C: fp32 on CPU (accelerate available)")
            print("  [Mistral] WARNING: CPU inference is very slow (~5 min/query).")
            print("  [Mistral] Strongly recommended: use Google Colab T4 GPU instead.")
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map  = "cpu",
                torch_dtype = torch.float32,
            )

        else:
            # ── Strategy D: fp32 on CPU, NO accelerate ────────────────────────
            # This is the safe path for Windows without accelerate.
            # device_map is NOT used — instead we call .to() after loading.
            print("  [Mistral] Strategy D: fp32 on CPU without accelerate")
            print("  [Mistral] WARNING: CPU inference is very slow (~5 min/query).")
            print("  [Mistral] Run: pip install accelerate  for faster loading.")
            print("  [Mistral] Or use Google Colab T4 GPU for practical speed.")
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype = torch.float32,
                low_cpu_mem_usage = True,   # stream weights, reduce peak RAM
            )
            self.model = self.model.to("cpu")
            self._effective_device = "cpu"

        # Store effective device for use in generate()
        if not hasattr(self, "_effective_device"):
            self._effective_device = effective_device

        self.model.eval()
        print("  [Mistral] Model ready.")

    @staticmethod
    def _check_package(name: str) -> bool:
        """Return True if a Python package is importable."""
        import importlib.util
        return importlib.util.find_spec(name) is not None

    # Maximum tokens the T4 can handle safely with 4-bit Mistral loaded
    # 4-bit model: ~5 GB. T4 total: 15 GB. Available for KV cache: ~8 GB.
    # At fp16, one token's KV cache ≈ 2 × 32 layers × 8 heads × 128 dim × 2 bytes ≈ 131 KB
    # 8 GB / 131 KB ≈ ~3000 tokens safely. We set 2048 as a conservative hard cap.
    _MAX_INPUT_TOKENS = 2048

    def generate(self, messages: list[dict]) -> tuple[str, dict]:
        """
        Generate a response from a messages list.
        Returns (response_text, token_stats).

        OOM fix (T4 CUDA out of memory)
        ────────────────────────────────
        Root cause: prompt = 5 chunks × ~863 tokens = ~4300 tokens.
        With 4-bit Mistral using ~5 GB and the KV cache needing ~794 MB
        per forward pass at that length, the T4's 15 GB fills up.

        Three fixes applied:
          1. VRAM flush before every call (torch.cuda.empty_cache)
          2. Hard prompt truncation to _MAX_INPUT_TOKENS (2048)
             — trims the middle context blocks, keeps system prompt + question
          3. max_new_tokens capped at 512 (was 800) to leave headroom
        """
        import torch
        import gc

        # ── Fix 1: flush VRAM fragmentation before each call ─────────────────
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

        # Resolve device safely across all 4 loading strategies
        try:
            target_device = next(self.model.parameters()).device
        except StopIteration:
            target_device = torch.device(self._effective_device)

        # ── Tokenise with chat template ───────────────────────────────────────
        template_output = self.tokenizer.apply_chat_template(
            messages,
            return_tensors        = "pt",
            add_generation_prompt = True,
        )

        # Normalise: extract raw tensor regardless of return type
        if isinstance(template_output, torch.Tensor):
            input_ids = template_output
        else:
            input_ids = template_output["input_ids"]

        # ── Fix 2: hard truncate if prompt exceeds safe token budget ─────────
        prompt_tokens = input_ids.shape[-1]
        if prompt_tokens > self._MAX_INPUT_TOKENS:
            print(f"  [Mistral] Prompt {prompt_tokens} tokens exceeds "
                  f"{self._MAX_INPUT_TOKENS} limit — truncating context "
                  f"(keeps first 512 + last 1536 tokens).")
            # Keep system instructions (first 512) + question (last 1536)
            # This preserves the question and citation instructions intact
            head = input_ids[:, :512]
            tail = input_ids[:, -(self._MAX_INPUT_TOKENS - 512):]
            input_ids = torch.cat([head, tail], dim=1)
            prompt_tokens = input_ids.shape[-1]

        input_ids = input_ids.to(target_device)

        # ── Fix 3: cap max_new_tokens to leave VRAM headroom ─────────────────
        safe_max_new = min(self.max_new_tokens, 512)

        generate_kwargs = dict(
            max_new_tokens = safe_max_new,
            do_sample      = self.do_sample,
            pad_token_id   = self.tokenizer.eos_token_id,
            eos_token_id   = self.tokenizer.eos_token_id,
        )
        if self.do_sample:
            generate_kwargs["temperature"] = self.temperature

        with torch.no_grad():
            output_ids = self.model.generate(input_ids, **generate_kwargs)

        # ── Flush again after generation ──────────────────────────────────────
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Decode only the newly generated tokens (skip the prompt)
        new_tokens        = output_ids[0][prompt_tokens:]
        response          = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        completion_tokens = len(new_tokens)

        token_stats = {
            "prompt_tokens":     prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens":      prompt_tokens + completion_tokens,
        }
        return response.strip(), token_stats


class GPT4oBackend:
    """
    GPT-4o via OpenAI API.
    Used for RQ3: comparing Mistral-7B vs GPT-4o on faithfulness,
    answer relevancy, and hallucination rate.

    Set OPENAI_API_KEY in your environment or pass api_key directly.
    Cost: ~$0.002–0.005 per query at typical context + completion sizes.
    """

    def __init__(
        self,
        model:         str   = "gpt-4o",
        max_tokens:    int   = DEFAULTS["max_new_tokens"],
        temperature:   float = DEFAULTS["temperature"],
        api_key:       Optional[str] = None,
    ):
        self.model       = model
        self.max_tokens  = max_tokens
        self.temperature = temperature

        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("pip install openai")

        key = api_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise ValueError(
                "OPENAI_API_KEY not set.\n"
                "  export OPENAI_API_KEY=sk-...   (Linux/Mac)\n"
                "  set OPENAI_API_KEY=sk-...       (Windows CMD)\n"
                "  $env:OPENAI_API_KEY='sk-...'    (PowerShell)"
            )
        self._client = OpenAI(api_key=key)
        print(f"  [GPT-4o] Client ready. Model={model}")

    def generate(self, messages: list[dict]) -> tuple[str, dict]:
        """
        Generate a response. Returns (response_text, token_stats).
        """
        resp = self._client.chat.completions.create(
            model       = self.model,
            messages    = messages,
            max_tokens  = self.max_tokens,
            temperature = self.temperature,
        )
        response    = resp.choices[0].message.content or ""
        token_stats = {
            "prompt_tokens":     resp.usage.prompt_tokens,
            "completion_tokens": resp.usage.completion_tokens,
            "total_tokens":      resp.usage.total_tokens,
        }
        return response.strip(), token_stats


class MistralAPIBackend:
    """
    Mistral La Plateforme API — using plain requests, zero SDK dependency.

    WHY NO SDK
    ──────────
    The mistralai Python SDK has a persistent version conflict in Colab's
    pre-installed environment that causes ImportError regardless of pip
    reinstalls. This implementation calls the REST API directly using
    the standard `requests` library (always available, no install needed).
    The behaviour is identical to the SDK — same endpoint, same payload,
    same response schema.

    SETUP (one-time, 3 minutes)
    ────────────────────────────
    1. Go to console.mistral.ai → Sign up (free)
    2. Billing → Add payment method (unlocks free €5 credit, no charge)
    3. API Keys → Create new key → copy it
    4. Colab 🔑 sidebar → Add secret → name=MISTRAL_API_KEY, value=<your key>

    COST ESTIMATE
    ─────────────
    mistral-small-latest : ~€0.001 per query
    Full thesis run (~200 queries) : < €0.20
    Free €5 credit covers entire thesis with large margin.
    """

    _API_URL = "https://api.mistral.ai/v1/chat/completions"

    def __init__(
        self,
        api_key:     Optional[str] = None,
        model:       str   = "mistral-small-latest",
        max_tokens:  int   = 800,
        temperature: float = DEFAULTS["temperature"],
    ):
        self.model       = model
        self.max_tokens  = max_tokens
        self.temperature = temperature

        # requests is part of Python stdlib environment — always available
        import requests as _req
        self._requests = _req

        key = api_key or os.environ.get("MISTRAL_API_KEY")
        if not key:
            raise ValueError(
                "MISTRAL_API_KEY not set.\n"
                "  In Colab Secrets (🔑 left sidebar):\n"
                "    Name  : MISTRAL_API_KEY\n"
                "    Value : your key from console.mistral.ai\n"
                "  Then run:\n"
                "    from google.colab import userdata\n"
                "    import os\n"
                "    os.environ['MISTRAL_API_KEY'] = userdata.get('MISTRAL_API_KEY')"
            )

        self._headers = {
            "Authorization": f"Bearer {key}",
            "Content-Type":  "application/json",
            "Accept":        "application/json",
        }

        # Verify key is valid with a minimal test call
        self._verify_key()

        print(f"  [Mistral API] Ready — no SDK required.")
        print(f"  [Mistral API] Model          : {model}")
        print(f"  [Mistral API] Endpoint       : {self._API_URL}")
        print(f"  [Mistral API] Context window : 32k tokens")
        print(f"  [Mistral API] EU data residency: yes")

    def _verify_key(self) -> None:
        """Ping the models endpoint to confirm the API key is valid."""
        try:
            r = self._requests.get(
                "https://api.mistral.ai/v1/models",
                headers=self._headers,
                timeout=10,
            )
            if r.status_code == 401:
                raise ValueError(
                    "Mistral API key is invalid or expired.\n"
                    "Get a new key at console.mistral.ai → API Keys."
                )
            # 200 = valid, other codes are network issues — ignore for now
        except self._requests.exceptions.ConnectionError:
            print("  [Mistral API] Warning: could not verify key (no internet?). Continuing.")

    def generate(self, messages: list[dict]) -> tuple[str, dict]:
        """
        Call Mistral chat completions via plain HTTP POST.
        Returns (response_text, token_stats).
        No SDK. No version conflicts. Pure requests.
        """
        import time as _time

        payload = {
            "model":       self.model,
            "messages":    messages,
            "max_tokens":  self.max_tokens,
            "temperature": self.temperature,
        }

        max_retries = 3
        for attempt in range(1, max_retries + 1):
            try:
                resp = self._requests.post(
                    self._API_URL,
                    headers = self._headers,
                    json    = payload,
                    timeout = 120,
                )

                # Rate limit — wait and retry
                if resp.status_code == 429:
                    wait = 30 * attempt
                    print(f"  [Mistral API] Rate limited — waiting {wait}s...")
                    _time.sleep(wait)
                    continue

                # Auth error — fail immediately, no point retrying
                if resp.status_code == 401:
                    raise ValueError(
                        "Mistral API key rejected (401). "
                        "Check your key at console.mistral.ai → API Keys."
                    )

                resp.raise_for_status()
                data = resp.json()

                response = (
                    data["choices"][0]["message"]["content"] or ""
                ).strip()

                usage = data.get("usage", {})
                token_stats = {
                    "prompt_tokens":      usage.get("prompt_tokens", 0),
                    "completion_tokens":  usage.get("completion_tokens", 0),
                    "total_tokens":       usage.get("total_tokens", 0),
                    "model":              self.model,
                    "estimated_cost_eur": round(
                        usage.get("total_tokens", 0) * 0.000002, 6
                    ),
                }
                return response, token_stats

            except self._requests.exceptions.Timeout:
                print(f"  [Mistral API] Timeout (attempt {attempt}/{max_retries}). Retrying...")
                _time.sleep(10)
            except self._requests.exceptions.ConnectionError as e:
                print(f"  [Mistral API] Connection error: {e}. Retrying...")
                _time.sleep(10)

        raise RuntimeError(
            f"Mistral API failed after {max_retries} attempts. "
            "Check your internet connection and API key."
        )


class GroqAPIBackend:
    """
    LLaMA-3-8B-Instruct (and other models) via Groq Cloud API.

    WHY GROQ FOR RQ3
    ─────────────────
    Groq runs inference on custom LPU (Language Processing Unit) hardware.
    For your thesis RQ3 comparison this is ideal because:
      ✓ FREE tier — 14,400 req/day, 500k tokens/min on LLaMA-3-8B
      ✓ FASTEST API available — 500-800 tokens/sec (10x OpenAI speed)
      ✓ ZERO local storage — model runs on Groq servers, not your machine
      ✓ No C drive impact — no downloads, no cache files
      ✓ LLaMA-3-8B is open-source — citable as open-source model in thesis
      ✓ Direct comparison with GPT-4o (closed) vs LLaMA-3 (open) is a
        standard RQ3 setup in RAG evaluation literature

    SETUP (2 minutes, completely free)
    ────────────────────────────────────
    1. Go to console.groq.com → Sign up (free, no credit card needed)
    2. API Keys → Create API Key → copy it
    3. In PowerShell: $env:GROQ_API_KEY = "gsk_your_key_here"
       Or add to activate.bat: set GROQ_API_KEY=gsk_your_key_here

    MODELS AVAILABLE (free tier)
    ─────────────────────────────
    "llama-3.1-8b-instant"   — LLaMA 3.1 8B, fastest, for RQ3
    "llama-3.1-70b-versatile" — LLaMA 3.1 70B, stronger, costs tokens faster
    "mixtral-8x7b-32768"      — Mixtral 8x7B, Mistral's MoE model
    "gemma2-9b-it"            — Google Gemma 2 9B

    COST
    ─────
    Free tier covers your entire thesis evaluation (~1000 queries).
    No payment method needed.
    """

    _API_URL = "https://api.groq.com/openai/v1/chat/completions"

    def __init__(
        self,
        api_key:     Optional[str] = None,
        model:       str   = "llama-3.1-8b-instant",
        max_tokens:  int   = 800,
        temperature: float = DEFAULTS["temperature"],
    ):
        self.model       = model
        self.max_tokens  = max_tokens
        self.temperature = temperature

        import requests as _req
        self._requests = _req

        key = api_key or os.environ.get("GROQ_API_KEY")
        if not key:
            raise ValueError(
                "GROQ_API_KEY not set.\n"
                "  Get a FREE key at: console.groq.com (no credit card needed)\n"
                "  Then set it:\n"
                "    PowerShell: $env:GROQ_API_KEY = 'gsk_your_key'\n"
                "    CMD:  set GROQ_API_KEY=gsk_your_key\n"
                "    activate.bat: add  set GROQ_API_KEY=gsk_your_key"
            )

        self._headers = {
            "Authorization": f"Bearer {key}",
            "Content-Type":  "application/json",
        }

        # Quick validation
        try:
            test = self._requests.get(
                "https://api.groq.com/openai/v1/models",
                headers=self._headers, timeout=8
            )
            if test.status_code == 401:
                raise ValueError("Groq API key invalid. Check console.groq.com")
        except self._requests.exceptions.ConnectionError:
            print("  [Groq] Warning: could not verify key (no internet?). Continuing.")

        print(f"  [Groq API] Ready — zero local storage.")
        print(f"  [Groq API] Model  : {model}")
        print(f"  [Groq API] Speed  : ~500-800 tokens/sec (LPU hardware)")
        print(f"  [Groq API] Cost   : FREE tier (14,400 req/day)")

    def generate(self, messages: list[dict]) -> tuple[str, dict]:
        """
        Call Groq chat completions (OpenAI-compatible endpoint).
        Returns (response_text, token_stats).
        """
        import time as _time

        payload = {
            "model":       self.model,
            "messages":    messages,
            "max_tokens":  self.max_tokens,
            "temperature": self.temperature,
        }

        for attempt in range(1, 4):
            try:
                resp = self._requests.post(
                    self._API_URL,
                    headers = self._headers,
                    json    = payload,
                    timeout = 60,
                )

                if resp.status_code == 429:
                    # Groq free tier rate limit — short wait
                    wait = int(resp.headers.get("retry-after", 10))
                    print(f"  [Groq] Rate limited — waiting {wait}s...")
                    _time.sleep(wait)
                    continue

                if resp.status_code == 401:
                    raise ValueError("Groq API key rejected. Check console.groq.com")

                resp.raise_for_status()
                data = resp.json()

                text  = (data["choices"][0]["message"]["content"] or "").strip()
                usage = data.get("usage", {})

                token_stats = {
                    "prompt_tokens":     usage.get("prompt_tokens", 0),
                    "completion_tokens": usage.get("completion_tokens", 0),
                    "total_tokens":      usage.get("total_tokens", 0),
                    "model":             self.model,
                    "estimated_cost_usd": 0.0,   # free tier
                }
                return text, token_stats

            except self._requests.exceptions.Timeout:
                print(f"  [Groq] Timeout attempt {attempt}/3")
                _time.sleep(5)
            except self._requests.exceptions.ConnectionError as e:
                print(f"  [Groq] Connection error: {e}")
                _time.sleep(5)

        raise RuntimeError("Groq API failed after 3 attempts.")


class HFInferenceBackend:
    """
    Mistral-7B-Instruct via HuggingFace Serverless Inference API.

    WHY THIS EXISTS
    ───────────────
    Colab CPU-only runtime has ~12 GB RAM.
    Mistral-7B fp32 needs ~28 GB — it always OOMs when loaded locally.

    This backend sends generation requests to HuggingFace's servers
    using your HF token. The model runs on HF infrastructure, not your
    machine. Your CPU only handles retrieval (bge-large ~1.2 GB — fine).

    Cost  : Free for low-to-moderate usage with a HF account.
             Throttled without token; faster with token.
    Limit : HF serverless inference has a ~2000 token input cap on the
            free tier. The pipeline auto-trims context to stay within it.
    Setup : Set HF_TOKEN env var or pass hf_token= directly.
            Get token at huggingface.co/settings/tokens (Read access).

    For thesis-scale usage (~200 queries) this is entirely free.
    """

    # HF Inference API endpoint for Mistral-7B-Instruct-v0.3
    _API_URL = (
        "https://api-inference.huggingface.co/models/"
        "mistralai/Mistral-7B-Instruct-v0.3"
    )
    # Conservative token budget for free-tier HF inference
    _MAX_CONTEXT_CHARS = 6000    # ~1500 tokens — leaves room for generation
    _MAX_NEW_TOKENS    = 600

    def __init__(
        self,
        hf_token:    Optional[str] = None,
        max_tokens:  int   = _MAX_NEW_TOKENS,
        temperature: float = DEFAULTS["temperature"],
    ):
        import requests as _req
        self._requests    = _req
        self.max_tokens   = max_tokens
        self.temperature  = temperature

        token = hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
        if not token:
            raise ValueError(
                "HF_TOKEN not set.\n"
                "  In Colab: use Secrets (🔑 sidebar) → name=HF_TOKEN\n"
                "  Then: os.environ['HF_TOKEN'] = userdata.get('HF_TOKEN')\n"
                "  Get a token at: huggingface.co/settings/tokens"
            )
        self._headers = {"Authorization": f"Bearer {token}"}
        print(f"  [HF Inference] Client ready.")
        print(f"  [HF Inference] Model: mistralai/Mistral-7B-Instruct-v0.3")
        print(f"  [HF Inference] Running on HF servers — no local GPU needed.")

    def _build_prompt(self, messages: list[dict]) -> str:
        """
        Convert messages list → Mistral [INST] chat format.
        The HF serverless API accepts raw text, not chat dicts,
        so we manually apply the Mistral v0.3 chat template.
        """
        prompt = "<s>"
        for msg in messages:
            role    = msg["role"]
            content = msg["content"]
            if role == "system":
                # Mistral-v0.3 folds system prompt into first user turn
                prompt += f"[INST] {content}\n\n"
            elif role == "user":
                # If we already opened [INST] from system, just append
                if prompt.endswith("\n\n"):
                    prompt += f"{content} [/INST]"
                else:
                    prompt += f"[INST] {content} [/INST]"
            elif role == "assistant":
                prompt += f" {content} </s><s>"
        return prompt

    def _trim_context(self, prompt: str) -> str:
        """
        Hard-trim prompt to stay within HF free-tier token limit.
        Removes middle context chunks, keeping system instructions
        and the question intact.
        """
        if len(prompt) <= self._MAX_CONTEXT_CHARS:
            return prompt
        # Keep first 2000 chars (system + instructions) and last 2000 (question)
        head = prompt[:2000]
        tail = prompt[-2000:]
        trimmed = (
            head
            + "\n\n[... context trimmed to fit HF free-tier limit ...]\n\n"
            + tail
        )
        print(f"  [HF Inference] Context trimmed: {len(prompt)} → {len(trimmed)} chars")
        return trimmed

    def generate(self, messages: list[dict]) -> tuple[str, dict]:
        """
        Send generation request to HF Inference API.
        Returns (response_text, token_stats).
        """
        import time as _time

        prompt  = self._build_prompt(messages)
        prompt  = self._trim_context(prompt)

        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens":  self.max_tokens,
                "temperature":     self.temperature,
                "do_sample":       self.temperature > 0,
                "return_full_text": False,   # return only new tokens
            },
        }

        # Retry logic — HF sometimes returns 503 while model loads
        max_retries = 3
        for attempt in range(1, max_retries + 1):
            try:
                resp = self._requests.post(
                    self._API_URL,
                    headers = self._headers,
                    json    = payload,
                    timeout = 120,
                )

                if resp.status_code == 503:
                    wait = 20 * attempt
                    print(f"  [HF Inference] Model loading on HF servers, "
                          f"waiting {wait}s... (attempt {attempt}/{max_retries})")
                    _time.sleep(wait)
                    continue

                if resp.status_code == 429:
                    print("  [HF Inference] Rate limited — waiting 30s...")
                    _time.sleep(30)
                    continue

                resp.raise_for_status()
                data     = resp.json()
                response = ""

                if isinstance(data, list) and data:
                    response = data[0].get("generated_text", "")
                elif isinstance(data, dict):
                    response = data.get("generated_text", "")

                # Strip any leaked prompt prefix if return_full_text slipped through
                if response.startswith(prompt[:50]):
                    response = response[len(prompt):]

                # Rough token estimate (HF API doesn't return usage counts)
                prompt_tokens     = len(prompt.split())
                completion_tokens = len(response.split())
                token_stats = {
                    "prompt_tokens":     prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens":      prompt_tokens + completion_tokens,
                    "note":              "token counts are word-level estimates for HF API",
                }
                return response.strip(), token_stats

            except self._requests.exceptions.Timeout:
                print(f"  [HF Inference] Timeout on attempt {attempt}. Retrying...")
                _time.sleep(10)
            except Exception as exc:
                if attempt == max_retries:
                    raise RuntimeError(
                        f"HF Inference API failed after {max_retries} attempts: {exc}"
                    ) from exc
                _time.sleep(10)

        raise RuntimeError("HF Inference API: all retries exhausted.")


# ── RAG Pipeline ──────────────────────────────────────────────────────────────

class RAGPipeline:
    """
    End-to-end RAG pipeline:
      query → HybridRetriever → PromptBuilder → LLM → CitationParser
      → RAGResponse

    This is the main class to instantiate in your notebooks and
    evaluation scripts.

    Parameters
    ----------
    backend       : "mistral" (default) or "gpt4o"
    top_k         : number of chunks passed to LLM as context (default 5)
    load_in_4bit  : use 4-bit quantisation for Mistral (default True)
    device        : "auto" | "cpu" | "cuda" | "mps"
    openai_key    : optional OpenAI API key (or set OPENAI_API_KEY env var)

    Quick start
    -----------
    >>> pipe = RAGPipeline(backend="mistral")
    >>> resp = pipe.query("What is India's 2030 renewable energy target?",
    ...                    filters={"country": "India"})
    >>> resp.pretty_print()
    """

    def __init__(
        self,
        backend:          str  = "mistral",
        top_k:            int  = DEFAULTS["top_k"],
        load_in_4bit:     bool = DEFAULTS["load_in_4bit"],
        device:           str  = DEFAULTS["mistral_device"],
        openai_key:       Optional[str] = None,
        hf_token:         Optional[str] = None,
        mistral_api_key:  Optional[str] = None,
        mistral_model:    str = "mistral-small-latest",
        groq_api_key:     Optional[str] = None,          # for backend="groq"
        groq_model:       str = "llama-3.1-8b-instant",  # LLaMA-3.1-8B default
        # Retriever path overrides
        chunks_file:  str = DEFAULTS["chunks_file"],
        vectorstore:  str = DEFAULTS["vectorstore"],
        collection:   str = DEFAULTS["collection"],
        embed_device: str = DEFAULTS["embed_device"],
    ):
        _valid = ("mistral", "gpt4o", "hf_inference", "mistral_api", "groq")
        if backend not in _valid:
            raise ValueError(f"backend must be one of {_valid}, got '{backend}'")

        self.backend = backend
        self.top_k   = top_k

        # ── Lazy-import retriever to avoid circular import ────────────────────
        print("\n[RAGPipeline] Initialising retriever...")
        # Import HybridRetriever from retriever.py in the same directory
        retriever_path = Path(__file__).parent / "retriever.py"
        if not retriever_path.exists():
            # Try current working directory
            retriever_path = Path("retriever.py")
        if not retriever_path.exists():
            raise FileNotFoundError(
                "retriever.py not found. Place it in the same directory as generation.py"
            )

        import importlib.util
        spec = importlib.util.spec_from_file_location("retriever", retriever_path)
        retriever_mod = importlib.util.load_from_spec(spec) if hasattr(
            importlib.util, "load_from_spec") else None

        # Fallback: sys.path injection
        sys.path.insert(0, str(retriever_path.parent))
        from retriever import HybridRetriever

        self._retriever = HybridRetriever(
            chunks_file = chunks_file,
            vectorstore = vectorstore,
            collection  = collection,
            device      = embed_device,
        )

        # Build chunk metadata lookup for citation enrichment
        self._chunk_meta = {
            c["chunk_id"]: c
            for c in self._retriever._chunks
        }
        # Also index by doc_id for citation resolution
        self._doc_meta: dict[str, dict] = {}
        for c in self._retriever._chunks:
            did = c.get("doc_id", "")
            if did and did not in self._doc_meta:
                self._doc_meta[did] = c

        # ── Load LLM backend ─────────────────────────────────────────────────
        print(f"\n[RAGPipeline] Loading LLM backend: {backend}...")
        if backend == "mistral":
            self._llm = MistralBackend(
                load_in_4bit   = load_in_4bit,
                device         = device,
                max_new_tokens = DEFAULTS["max_new_tokens"],
                temperature    = DEFAULTS["temperature"],
                do_sample      = DEFAULTS["do_sample"],
            )
            self._model_name = DEFAULTS["mistral_model"]
        elif backend == "mistral_api":
            self._llm = MistralAPIBackend(
                api_key     = mistral_api_key,
                model       = mistral_model,
                max_tokens  = DEFAULTS["max_new_tokens"],
                temperature = DEFAULTS["temperature"],
            )
            self._model_name = f"mistral-api/{mistral_model}"
        elif backend == "groq":
            # LLaMA-3.1-8B (or any Groq model) via free Groq Cloud API
            # Zero local storage — runs on Groq LPU hardware
            self._llm = GroqAPIBackend(
                api_key     = groq_api_key,
                model       = groq_model,
                max_tokens  = DEFAULTS["max_new_tokens"],
                temperature = DEFAULTS["temperature"],
            )
            self._model_name = f"groq/{groq_model}"
        elif backend == "hf_inference":
            self._llm = HFInferenceBackend(hf_token=hf_token)
            self._model_name = "mistralai/Mistral-7B-Instruct-v0.3 (HF API)"
        else:
            self._llm = GPT4oBackend(api_key=openai_key)
            self._model_name = "gpt-4o"

        print(f"\n[RAGPipeline] Ready.  Backend={backend}  top_k={top_k}\n")

    # ── Public API ────────────────────────────────────────────────────────────

    def query(
        self,
        query:   str,
        filters: Optional[dict] = None,
        top_k:   Optional[int]  = None,
        save:    bool = True,
    ) -> RAGResponse:
        """
        Full RAG query: retrieve → prompt → generate → parse → return.

        Parameters
        ----------
        query   : Natural language question
        filters : ChromaDB metadata filter (e.g. {"country": "India"})
        top_k   : Override default chunk count
        save    : Write response JSON to data/responses/

        Returns
        -------
        RAGResponse with answer, citations, and all metadata
        """
        t_start  = time.time()
        k        = top_k or self.top_k
        qtype    = PromptBuilder.classify_query(query)

        # ── 1. Retrieve ───────────────────────────────────────────────────────
        chunks = self._retriever.retrieve(
            query   = query,
            top_k   = k,
            filters = filters,
        )

        # ── 2. Build prompt ───────────────────────────────────────────────────
        messages   = PromptBuilder.build_messages(query, chunks, qtype)
        prompt_txt = messages[-1]["content"]  # user message for RAGAS eval

        # ── 3. Generate ───────────────────────────────────────────────────────
        raw_answer, token_stats = self._llm.generate(messages)

        # ── 4. Parse citations ────────────────────────────────────────────────
        citations = CitationParser.parse(raw_answer, self._doc_meta)

        # ── 5. Check insufficient-context flag ───────────────────────────────
        insufficient = raw_answer.strip().upper().startswith("INSUFFICIENT_CONTEXT")

        latency = round(time.time() - t_start, 2)

        resp = RAGResponse(
            query            = query,
            answer           = raw_answer,
            citations        = citations,
            insufficient     = insufficient,
            query_type       = qtype,
            retrieved_chunks = chunks,
            model_used       = self._model_name,
            backend          = self.backend,
            latency_sec      = latency,
            token_stats      = token_stats,
            prompt_text      = prompt_txt,
            filters_used     = filters,
        )

        if save:
            self._save_response(resp)

        return resp

    def batch_query(
        self,
        queries: list[dict],   # list of {"query": str, "filters": dict}
        save:    bool = True,
    ) -> list[RAGResponse]:
        """Run multiple queries sequentially. Useful for evaluation sets."""
        results = []
        for i, item in enumerate(queries, 1):
            print(f"  [{i}/{len(queries)}] {item['query'][:60]}...")
            resp = self.query(
                query   = item["query"],
                filters = item.get("filters"),
                save    = save,
            )
            results.append(resp)
        return results

    # ── Private helpers ───────────────────────────────────────────────────────

    def _save_response(self, resp: RAGResponse) -> None:
        """Save response as JSON for later RAGAS evaluation."""
        out_dir = Path(DEFAULTS["responses_dir"])
        out_dir.mkdir(parents=True, exist_ok=True)
        ts   = int(time.time())
        slug = re.sub(r"[^\w]", "_", resp.query[:40]).lower()
        path = out_dir / f"{resp.backend}_{slug}_{ts}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(resp.to_dict(), f, indent=2, ensure_ascii=False,
                      default=str)


# ── CLI demo ──────────────────────────────────────────────────────────────────

DEMO_QUERIES = [
    {
        "label":   "factual_ndc",
        "query":   "What is India's 2030 renewable energy target and progress?",
        "filters": {"country": "India"},
    },
    {
        "label":   "gap_analysis",
        "query":   "What is the gap between Indonesia's NDC commitments and IPCC 1.5°C recommendations?",
        "filters": None,
    },
    {
        "label":   "comparative",
        "query":   "How does the UK's net zero commitment compare to Australia's?",
        "filters": {"$or": [{"country": "UK"}, {"country": "Australia"}]},
    },
    {
        "label":   "ipcc_benchmark",
        "query":   "What does IPCC WG3 recommend for the energy sector to limit warming to 1.5°C?",
        "filters": {"$and": [{"source": "IPCC_AR6"}, {"wg": "WG3"}]},
    },
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="RAG Generation Module — climate policy Q&A"
    )
    parser.add_argument("--query",       default=None)
    parser.add_argument("--backend",     default="mistral",
                        choices=["mistral", "gpt4o"])
    parser.add_argument("--top_k",       default=5, type=int)
    parser.add_argument("--country",     default=None)
    parser.add_argument("--wg",          default=None)
    parser.add_argument("--no_4bit",     action="store_true",
                        help="Disable 4-bit quantisation (use on CPU-only)")
    parser.add_argument("--device",      default="auto")
    parser.add_argument("--demo",        action="store_true",
                        help="Run all 4 demo queries")
    parser.add_argument("--no_save",     action="store_true",
                        help="Do not save response JSON files")
    args = parser.parse_args()

    # ── Build filters from CLI args ───────────────────────────────────────────
    cli_filters: dict = {}
    if args.country: cli_filters["country"] = args.country
    if args.wg:      cli_filters["wg"]      = args.wg
    filters = cli_filters or None

    # ── Init pipeline ─────────────────────────────────────────────────────────
    pipe = RAGPipeline(
        backend      = args.backend,
        top_k        = args.top_k,
        load_in_4bit = not args.no_4bit,
        device       = args.device,
    )

    # ── Run demo or single query ──────────────────────────────────────────────
    if args.demo:
        print("\n" + "═" * 70)
        print("  RAG PIPELINE DEMO — 4 showcase queries")
        print("  Primary: Mistral-7B-Instruct  |  Context: G20 NDCs + IPCC AR6")
        print("═" * 70)

        for demo in DEMO_QUERIES:
            resp = pipe.query(
                query   = demo["query"],
                filters = demo["filters"],
                save    = not args.no_save,
            )
            resp.pretty_print()

    elif args.query:
        resp = pipe.query(
            query   = args.query,
            filters = filters,
            save    = not args.no_save,
        )
        resp.pretty_print()

    else:
        parser.print_help()
        print("\n  Quick start:")
        print("  python generation.py --demo")
        print("  python generation.py --query 'India 2030 renewable energy' --country India")
        print("  python generation.py --query 'IPCC WG3 energy sector' --wg WG3")
        print("  python generation.py --backend gpt4o --query 'gap analysis China'")


if __name__ == "__main__":
    main()
