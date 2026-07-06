"""
===========================================================================
  HYBRID RETRIEVAL MODULE
  Master's Thesis: LLM-Powered Climate Policy Summariser & Gap Analyser
  University of Europe for Applied Sciences, Potsdam  |  2026
===========================================================================

Architecture Decision: Manual hybrid (rank_bm25 + ChromaDB + RRF)
   vs LlamaIndex QueryFusionRetriever

  CHOSE MANUAL because:
  ✓ LlamaIndex adds 50+ transitive dependencies for a feature you can
    build in 60 lines. No hidden abstractions to debug.
  ✓ Full control over fusion weights, BM25 parameters (k1, b), and
    tokenisation — critical for climate-domain vocabulary (MtCO2e,
    LULUCF, NDC, GWP100) which BM25 handles better with tuned params.
  ✓ Every step is explainable for your thesis Methods chapter.
  ✓ rank_bm25 is a single lightweight dependency (pure Python).

Retrieval Strategy
------------------
                 ┌─────────────────────────────┐
   Query ───────▶│  Dense Retriever (ChromaDB)  │──▶ ranked list A (sim scores)
                 │  BAAI/bge-large-en-v1.5      │
                 └─────────────────────────────┘
                         │
                 ┌───────▼─────────────────────┐
   Query ───────▶│  Sparse Retriever (BM25)     │──▶ ranked list B (BM25 scores)
                 │  rank_bm25 (Okapi BM25)      │
                 └─────────────────────────────┘
                         │
                 ┌───────▼─────────────────────┐
                 │  Reciprocal Rank Fusion      │──▶ final top-k chunks
                 │  score = Σ 1/(k + rank_i)   │    with full metadata
                 └─────────────────────────────┘

Why RRF works
-------------
  RRF (Cormack et al., 2009) fuses ranked lists without needing to
  normalise heterogeneous scores (cosine similarity vs BM25 scores live
  on completely different scales). Formula:

      RRF(d) = Σ_r  1 / (k + rank_r(d))

  where k=60 (empirically best across many benchmarks), rank_r(d) is
  the position of document d in ranked list r. Documents appearing in
  both lists get double rewards; documents missing from one list get
  no penalty.

  This is the same algorithm used by:
  - Elasticsearch Reciprocal Rank Fusion (8.x)
  - Cohere Rerank + Embed hybrid
  - TREC 2009 Federated Web Search track winner

BM25 Parameters (tuned for policy/scientific text)
---------------------------------------------------
  k1 = 1.6  (term saturation — higher than default 1.5 because climate
              docs repeat key terms like "2030", "GHG", "renewable"
              many times and we want more weight on frequency)
  b  = 0.75 (length normalisation — standard; NDC docs vary 2–80pp)

Usage
-----
  # Install
  pip install rank_bm25 chromadb sentence-transformers tqdm

  # Import and use
  from retriever import HybridRetriever

  retriever = HybridRetriever()          # loads from default paths
  results   = retriever.retrieve(
      query   = "What is India's 2030 renewable energy target?",
      top_k   = 5,
      filters = {"country": "India"},    # optional ChromaDB metadata filter
  )
  for r in results:
      print(r["rrf_score"], r["country"], r["section"])
      print(r["text"][:200])

  # CLI smoke test (runs 5 representative queries):
  python retriever.py
  python retriever.py --query "Indonesia adaptation gap" --top_k 8
  python retriever.py --query "net zero 2050" --country China
  python retriever.py --query "IPCC 1.5 degree mitigation pathway" --wg WG3
===========================================================================
"""

from __future__ import annotations

# ── Suppress noisy-but-benign warnings before any library imports ─────────────
import os
import warnings
import logging

# ── Suppress noisy-but-benign output from sentence_transformers & HF Hub ──────
#
# Root cause of previous patch not working:
#   - "BertModel LOAD REPORT" is a direct print()/logging call inside
#     sentence_transformers — NOT a Python warnings.warn() — so
#     warnings.filterwarnings() had zero effect on it.
#   - "unauthenticated requests" comes from huggingface_hub's logger,
#     not warnings module.
#
# Correct fix: suppress via the logging system, not warnings module.

# Silence sentence_transformers load report (position_ids UNEXPECTED etc.)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers.SentenceTransformer").setLevel(logging.ERROR)

# Silence HF Hub unauthenticated / rate-limit messages
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub.utils._headers").setLevel(logging.ERROR)

# Silence transformers weight-loading report (covers BertModel LOAD REPORT)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)

# Env vars: disable HF progress bars on model load (already downloaded)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TRANSFORMERS_VERBOSITY",  "error")
os.environ.setdefault("HF_HUB_VERBOSITY",        "error")

# Keep warnings module filter as belt-and-suspenders for any remaining warns
warnings.filterwarnings("ignore", message=".*position_ids.*")
warnings.filterwarnings("ignore", message=".*unauthenticated.*")
warnings.filterwarnings("ignore", message=".*weights.*not.*initialized.*",
                        category=UserWarning)

import argparse
import json
import math
import pickle
import sys
import time
from pathlib import Path
from typing import Any, Optional

# ── Dependency check ──────────────────────────────────────────────────────────
_missing = []
try:
    from rank_bm25 import BM25Okapi
except ImportError:
    _missing.append("rank_bm25")
try:
    import chromadb
except ImportError:
    _missing.append("chromadb")
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    _missing.append("sentence-transformers")
if _missing:
    print(f"[ERROR] Missing packages: {', '.join(_missing)}")
    print(f"        pip install {' '.join(_missing)}")
    sys.exit(1)

try:
    from tqdm import tqdm
    TQDM = True
except ImportError:
    TQDM = False


# ── Configuration ─────────────────────────────────────────────────────────────

DEFAULTS = {
    # Paths
    "chunks_file":   "data/processed/chunks.jsonl",
    "vectorstore":   "data/vectorstore",
    "collection":    "climate_policy_rag",
    "bm25_cache":    "data/processed/bm25_index.pkl",

    # Embedding model (must match what was used in embed_and_ingest.py)
    "model_name":    "BAAI/bge-large-en-v1.5",
    "query_prefix":  "Represent this sentence for searching relevant passages: ",
    "device":        "cpu",

    # BM25 hyperparameters (tuned for policy/scientific text)
    "bm25_k1":       1.6,
    "bm25_b":        0.75,

    # RRF constant (k=60 is empirically optimal across benchmarks)
    "rrf_k":         60,

    # Retrieval pool before fusion (fetch more, fuse, return top_k)
    # Fetch 4× top_k from each retriever so RRF has enough candidates
    "pool_multiplier": 4,

    # Default top-k returned to the caller
    "default_top_k": 5,
}


# ── Tokeniser for BM25 ───────────────────────────────────────────────────────

# Climate-domain stop words: very common terms that carry no discriminative
# signal for this specific corpus.
STOP_WORDS = {
    "the","a","an","and","or","but","in","on","at","to","for","of","with",
    "by","from","as","is","are","was","were","be","been","being","have",
    "has","had","do","does","did","will","would","could","should","may",
    "might","shall","this","that","these","those","it","its","also","such",
    "their","they","them","we","our","can","which","been","into","per",
    "under","through","between","among","above","below","than","more",
    "most","not","no","nor","so","if","then","each","other","all","any",
}


def tokenise(text: str) -> list[str]:
    """
    Tokenise text for BM25.
    Keeps climate-domain compound tokens (e.g. 'CO2', 'GHG', 'NDC',
    'LULUCF', 'MtCO2e', '2030', '1.5°C') intact by splitting only on
    whitespace and removing punctuation from token edges.

    Deliberately does NOT lowercase numbers/acronyms so BM25 can
    discriminate "2025" vs "2030" and "WG1" vs "WG3".
    """
    import re
    # Split on whitespace
    raw_tokens = text.split()
    tokens = []
    for tok in raw_tokens:
        # Strip surrounding punctuation but keep internal hyphens and periods
        cleaned = re.sub(r"^[^\w]+|[^\w]+$", "", tok)
        if not cleaned:
            continue
        lower = cleaned.lower()
        # Keep if: not a stop word AND length > 1
        if lower not in STOP_WORDS and len(cleaned) > 1:
            tokens.append(lower)
    return tokens


# ── Chunk quality gate ────────────────────────────────────────────────────────

def is_garbled_chunk(text: str) -> bool:
    """
    Detect chunks extracted from PDF table grids, figure axis labels, and
    other non-prose regions that would corrupt LLM context if retrieved.

    Uses 6 heuristics ordered cheapest-to-most-expensive.
    Returns True = garbled (skip this chunk), False = clean prose (keep).

    Root cause of v1 failure
    -------------------------
    The previous version measured symbol density over the FULL chunk
    (~863 tokens avg = ~4 000 chars). The garbled IPCC figure chunk
    "▼ 63 ▼ % 153% 1500 229%▲ ..." had enough total characters that
    the per-character density fell below the 0.12 threshold.

    Fix: Heuristic 0 inspects only the OPENING 150 characters where
    garbled content is concentrated — a much stronger signal.
    Heuristic 5 adds an absolute arrow count independent of chunk length.
    """
    from collections import Counter

    if not text or len(text.strip()) < 20:
        return True   # empty or near-empty is useless

    # ── H0: Opening-window check (most discriminative, cheapest) ─────────────
    # Garbled chunks almost always START garbled (figure captions, table
    # headers, axis labels). Check the first 150 chars with tight thresholds.
    window = text[:150]
    win_arrows = sum(1 for c in window if c in "▲▼▶◀→←↑↓►◄▸◂△▽")
    if win_arrows >= 2:                # ≥2 arrow-triangle chars in opening → garbled
        return True
    win_alpha = sum(1 for c in window if c.isalpha())
    if len(window) > 30 and (win_alpha / len(window)) < 0.25:
        return True                    # opening is mostly non-letters

    # ── H1: Absolute arrow count (independent of chunk length) ───────────────
    # Even long chunks with a few arrows mixed into prose rarely have >4.
    # IPCC figure-content chunks have 10-40 ▲▼ chars.
    total_arrows = sum(1 for c in text if c in "▲▼▶◀→←↑↓►◄▸◂△▽")
    if total_arrows >= 4:
        return True

    # ── H2: Full-text alphabetic fraction ────────────────────────────────────
    alpha_frac = sum(1 for c in text if c.isalpha()) / len(text)
    if alpha_frac < 0.35:              # <35% letters → not coherent prose
        return True

    # ── H3: Token-level number density ───────────────────────────────────────
    tokens = text.split()
    if not tokens:
        return True
    num_tokens = sum(
        1 for t in tokens
        if t.replace(".", "").replace("%", "").replace("-", "").isdigit()
    )
    if (num_tokens / len(tokens)) > 0.55:   # >55% bare numbers → table cells
        return True

    # ── H4: Token repetition (year-label grids: "2020 2030 2040 2050 2020...")─
    if len(tokens) >= 6:
        most_common_tok, most_common_n = Counter(tokens).most_common(1)[0]
        if most_common_n / len(tokens) > 0.20 and most_common_n >= 5:
            return True

    # ── H5: Very short real-word count ───────────────────────────────────────
    # A chunk with fewer than 8 words longer than 3 chars is unlikely to
    # be meaningful prose (catches isolated reference lists, page headers).
    long_words = sum(1 for t in tokens if len(t) > 3 and t.isalpha())
    if len(tokens) > 10 and long_words < 8:
        return True

    return False   # passed all heuristics — clean prose


# ── BM25 index builder / loader ───────────────────────────────────────────────

class BM25Index:
    """
    Builds and caches a BM25Okapi index from chunks.jsonl.
    The index is cached to disk so subsequent runs load in ~1 second
    instead of rebuilding from 4,705 chunks every time.
    """

    def __init__(
        self,
        chunks: list[dict],
        k1: float = DEFAULTS["bm25_k1"],
        b:  float = DEFAULTS["bm25_b"],
    ):
        self.chunks     = chunks
        self.chunk_ids  = [c["chunk_id"] for c in chunks]
        self.k1         = k1
        self.b          = b
        self._bm25: Optional[BM25Okapi] = None
        self._tokenised: Optional[list[list[str]]] = None

    def build(self) -> "BM25Index":
        """Tokenise all documents and fit BM25Okapi."""
        print(f"  Building BM25 index over {len(self.chunks)} chunks...", end=" ", flush=True)
        t0 = time.time()

        self._tokenised = [tokenise(c["text"]) for c in self.chunks]
        self._bm25      = BM25Okapi(self._tokenised, k1=self.k1, b=self.b)

        elapsed = time.time() - t0
        print(f"done ({elapsed:.1f}s)")
        return self

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "chunk_ids":  self.chunk_ids,
                "tokenised":  self._tokenised,
                "k1":         self.k1,
                "b":          self.b,
                "n_chunks":   len(self.chunks),
            }, f)
        print(f"  BM25 index cached → {path}")

    def load(self, path: str) -> bool:
        """Load pre-built index. Returns True if successful."""
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
            # Validate cached index matches current chunks
            if data["n_chunks"] != len(self.chunks):
                print(f"  [WARN] Cache chunk count mismatch "
                      f"({data['n_chunks']} cached vs {len(self.chunks)} current) "
                      f"— rebuilding.")
                return False
            self._tokenised = data["tokenised"]
            self._bm25      = BM25Okapi(self._tokenised, k1=data["k1"], b=data["b"])
            print(f"  BM25 index loaded from cache ({data['n_chunks']} docs)")
            return True
        except (FileNotFoundError, KeyError, Exception):
            return False

    def get_scores(self, query: str, top_n: int) -> list[tuple[str, float]]:
        """
        Returns list of (chunk_id, bm25_score) sorted descending,
        limited to top_n results.
        """
        if self._bm25 is None:
            raise RuntimeError("BM25 index not built. Call .build() first.")
        tokens = tokenise(query)
        scores = self._bm25.get_scores(tokens)
        # Pair with chunk_ids and sort
        paired = sorted(
            zip(self.chunk_ids, scores),
            key=lambda x: x[1],
            reverse=True,
        )
        return paired[:top_n]


# ── Reciprocal Rank Fusion ────────────────────────────────────────────────────

def reciprocal_rank_fusion(
    ranked_lists: list[list[tuple[str, float]]],
    k: int = DEFAULTS["rrf_k"],
) -> list[tuple[str, float]]:
    """
    Fuse multiple ranked lists using Reciprocal Rank Fusion.

    Args:
        ranked_lists : list of ranked lists, each a list of
                       (chunk_id, score) tuples ordered best-first.
        k            : RRF constant (default 60, from Cormack et al. 2009)

    Returns:
        List of (chunk_id, rrf_score) sorted descending.

    Formula:
        RRF(d) = Σ_r  1 / (k + rank_r(d))
        where rank_r(d) is 1-indexed position in ranked list r.
    """
    scores: dict[str, float] = {}
    for ranked_list in ranked_lists:
        for rank, (chunk_id, _raw_score) in enumerate(ranked_list, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)

    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


# ── Dense retriever (ChromaDB) ────────────────────────────────────────────────

class DenseRetriever:
    """Wraps ChromaDB for dense vector similarity search."""

    def __init__(
        self,
        vectorstore_path: str,
        collection_name:  str,
        model_name:       str,
        query_prefix:     str,
        device:           str,
    ):
        self.query_prefix = query_prefix

        # Load embedding model
        print(f"  Loading embedding model: {model_name}  (device={device})...",
              end=" ", flush=True)
        t0 = time.time()
        self._model = SentenceTransformer(model_name, device=device)
        print(f"done ({time.time()-t0:.1f}s)")

        # Connect to ChromaDB
        client           = chromadb.PersistentClient(path=vectorstore_path)
        self._collection = client.get_collection(collection_name)
        n = self._collection.count()
        print(f"  ChromaDB collection '{collection_name}': {n} vectors")

    def embed_query(self, query: str) -> list[float]:
        prefixed = self.query_prefix + query
        return self._model.encode(
            prefixed,
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).tolist()

    def retrieve(
        self,
        query:   str,
        top_n:   int,
        filters: Optional[dict] = None,
    ) -> list[tuple[str, float]]:
        """
        Returns list of (chunk_id, cosine_similarity) sorted descending.
        similarity = 1 - distance  (ChromaDB returns cosine distance).
        """
        q_emb = self.embed_query(query)

        kwargs: dict[str, Any] = {
            "query_embeddings": [q_emb],
            "n_results":        top_n,
            "include":          ["distances"],
        }
        if filters:
            kwargs["where"] = filters

        res  = self._collection.query(**kwargs)
        ids  = res["ids"][0]
        dists = res["distances"][0]

        # Convert distance → similarity and return (id, sim) pairs
        return [(cid, 1.0 - d) for cid, d in zip(ids, dists)]


# ── HybridRetriever — the main public class ───────────────────────────────────

class HybridRetriever:
    """
    Hybrid retriever combining dense (ChromaDB) + sparse (BM25) retrieval
    via Reciprocal Rank Fusion.

    Parameters
    ----------
    chunks_file    : path to data/processed/chunks.jsonl
    vectorstore    : path to data/vectorstore (ChromaDB persistent dir)
    collection     : ChromaDB collection name
    bm25_cache     : path to cache the BM25 index (rebuilds if stale)
    model_name     : sentence-transformer model (must match ingestion)
    device         : "cpu" | "cuda" | "mps"
    rrf_k          : RRF constant (default 60)
    pool_multiplier: fetch pool_multiplier × top_k from each retriever
                     before fusion (wider net = better fusion quality)

    Quick start
    -----------
    >>> retriever = HybridRetriever()
    >>> results = retriever.retrieve("India renewable energy 2030", top_k=5)
    >>> for r in results:
    ...     print(r["rrf_score"], r["country"], r["text"][:100])
    """

    def __init__(
        self,
        chunks_file:     str = DEFAULTS["chunks_file"],
        vectorstore:     str = DEFAULTS["vectorstore"],
        collection:      str = DEFAULTS["collection"],
        bm25_cache:      str = DEFAULTS["bm25_cache"],
        model_name:      str = DEFAULTS["model_name"],
        query_prefix:    str = DEFAULTS["query_prefix"],
        device:          str = DEFAULTS["device"],
        bm25_k1:         float = DEFAULTS["bm25_k1"],
        bm25_b:          float = DEFAULTS["bm25_b"],
        rrf_k:           int   = DEFAULTS["rrf_k"],
        pool_multiplier: int   = DEFAULTS["pool_multiplier"],
    ):
        self.rrf_k           = rrf_k
        self.pool_multiplier = pool_multiplier

        print("\n[HybridRetriever] Initialising...")

        # ── 1. Load all chunks into memory (needed for BM25 + metadata) ───────
        print(f"  Loading chunks from {chunks_file}...")
        self._chunks      = self._load_chunks(chunks_file)
        self._id_to_chunk = {c["chunk_id"]: c for c in self._chunks}
        print(f"  {len(self._chunks)} chunks loaded")

        # ── 2. Build / load BM25 index ────────────────────────────────────────
        self._bm25_index = BM25Index(self._chunks, k1=bm25_k1, b=bm25_b)
        if not self._bm25_index.load(bm25_cache):
            self._bm25_index.build()
            self._bm25_index.save(bm25_cache)

        # ── 3. Set up dense retriever ─────────────────────────────────────────
        self._dense = DenseRetriever(
            vectorstore_path = vectorstore,
            collection_name  = collection,
            model_name       = model_name,
            query_prefix     = query_prefix,
            device           = device,
        )

        print("[HybridRetriever] Ready.\n")

    # ── Private helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _load_chunks(path: str) -> list[dict]:
        chunks = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    chunks.append(json.loads(line))
        return chunks

    @staticmethod
    def _build_chroma_filter(filters: dict) -> dict:
        """
        Convert a simple flat dict into ChromaDB's $and syntax.
        Supports flat dict, already-composed $and/$or dicts, and
        special shorthand keys (country, ndc_cycle, wg, doc_type, source).

        Examples:
          {"country": "India"}
            → {"country": "India"}           (single field, no $and needed)
          {"country": "India", "ndc_cycle": 2}
            → {"$and": [{"country":"India"}, {"ndc_cycle":2}]}
          {"$or": [{"country":"China"}, {"country":"India"}]}
            → passed through as-is
        """
        if not filters:
            return None
        # Already a composed filter (has $and/$or at top level)
        if any(k.startswith("$") for k in filters):
            return filters
        # Single key — no wrapping needed
        if len(filters) == 1:
            return filters
        # Multiple keys — must wrap in $and
        return {"$and": [{k: v} for k, v in filters.items()]}

    def _apply_metadata_filter_bm25(
        self,
        bm25_results: list[tuple[str, float]],
        filters: Optional[dict],
    ) -> list[tuple[str, float]]:
        """
        Post-filter BM25 results by metadata (BM25 has no native filter).
        Supports the same filter keys as ChromaDB for consistency.
        """
        if not filters:
            return bm25_results

        def matches(chunk: dict, filt: dict) -> bool:
            # Handle $and
            if "$and" in filt:
                return all(matches(chunk, sub) for sub in filt["$and"])
            # Handle $or
            if "$or" in filt:
                return any(matches(chunk, sub) for sub in filt["$or"])
            # Single field comparison
            for key, val in filt.items():
                if key.startswith("$"):
                    continue
                chunk_val = chunk.get(key)
                if isinstance(val, dict):
                    # Operator dict: {"$gte": 5}
                    for op, operand in val.items():
                        if op == "$eq"  and chunk_val != operand:  return False
                        if op == "$ne"  and chunk_val == operand:  return False
                        if op == "$gt"  and not (chunk_val is not None and chunk_val > operand):   return False
                        if op == "$gte" and not (chunk_val is not None and chunk_val >= operand):  return False
                        if op == "$lt"  and not (chunk_val is not None and chunk_val < operand):   return False
                        if op == "$lte" and not (chunk_val is not None and chunk_val <= operand):  return False
                        if op == "$in"  and chunk_val not in operand: return False
                        if op == "$nin" and chunk_val in operand:     return False
                elif chunk_val != val:
                    return False
            return True

        return [
            (cid, score) for cid, score in bm25_results
            if cid in self._id_to_chunk
            and matches(self._id_to_chunk[cid], filters)
        ]

    # ── Public API ────────────────────────────────────────────────────────────

    def retrieve(
        self,
        query:       str,
        top_k:       int = DEFAULTS["default_top_k"],
        filters:     Optional[dict] = None,
        dense_weight: float = 0.5,   # reserved for future weighted RRF
        sparse_weight: float = 0.5,  # reserved for future weighted RRF
    ) -> list[dict]:
        """
        Hybrid retrieval: dense + sparse + RRF fusion.

        Parameters
        ----------
        query         : Natural language query string
        top_k         : Number of chunks to return
        filters       : Metadata filter dict — supports same syntax as
                        ChromaDB where= (single field or $and/$or)
        dense_weight  : (future use) weight for dense ranking contribution
        sparse_weight : (future use) weight for sparse ranking contribution

        Returns
        -------
        List of chunk dicts, sorted by RRF score descending.
        Each dict contains ALL original metadata PLUS:
          - rrf_score      : Reciprocal Rank Fusion score
          - dense_score    : cosine similarity from dense retrieval (or None)
          - sparse_score   : BM25 score from sparse retrieval (or None)
          - dense_rank     : rank in dense results (or None)
          - sparse_rank    : rank in sparse results (or None)
          - retrieval_mode : "hybrid" | "dense_only" | "sparse_only"
        """
        pool = top_k * self.pool_multiplier

        chroma_filter = self._build_chroma_filter(filters)

        # ── Dense retrieval ───────────────────────────────────────────────────
        try:
            dense_results = self._dense.retrieve(query, top_n=pool,
                                                  filters=chroma_filter)
        except Exception as e:
            print(f"  [WARN] Dense retrieval failed: {e}")
            dense_results = []

        # ── Sparse retrieval (BM25) ───────────────────────────────────────────
        try:
            sparse_raw = self._bm25_index.get_scores(query, top_n=pool * 2)
            # Apply metadata filter manually
            sparse_results = self._apply_metadata_filter_bm25(sparse_raw, filters)
            sparse_results  = sparse_results[:pool]
        except Exception as e:
            print(f"  [WARN] Sparse retrieval failed: {e}")
            sparse_results = []

        # ── RRF fusion ────────────────────────────────────────────────────────
        fused = reciprocal_rank_fusion(
            [dense_results, sparse_results],
            k=self.rrf_k,
        )

        # Build lookup maps for per-result scoring metadata
        dense_rank_map  = {cid: (rank+1, score)
                           for rank, (cid, score) in enumerate(dense_results)}
        sparse_rank_map = {cid: (rank+1, score)
                           for rank, (cid, score) in enumerate(sparse_results)}

        # ── Assemble final results — with garbled chunk filter ────────────────
        results: list[dict] = []
        garbled_skipped = 0

        # Fetch from a wider fused set so garbled-chunk removal doesn't
        # shrink the result below top_k. We pull up to top_k * 3 candidates
        # from the fused list and stop once we have top_k clean chunks.
        for cid, rrf_score in fused[:top_k * 3]:
            if len(results) >= top_k:
                break

            chunk = self._id_to_chunk.get(cid)
            if chunk is None:
                continue

            # ── Quality gate: silently skip garbled chunks ────────────────────
            chunk_text = chunk.get("text", "")
            if is_garbled_chunk(chunk_text):
                garbled_skipped += 1
                continue

            d_rank, d_score = dense_rank_map.get(cid,  (None, None))
            s_rank, s_score = sparse_rank_map.get(cid, (None, None))

            in_dense  = d_rank is not None
            in_sparse = s_rank is not None

            result = {
                **chunk,                          # all original fields + text
                # ── Retrieval scores ──────────────────────────────────────────
                "rrf_score":      round(rrf_score, 6),
                "dense_score":    round(d_score, 4) if d_score is not None else None,
                "sparse_score":   round(float(s_score), 4) if s_score is not None else None,
                "dense_rank":     d_rank,
                "sparse_rank":    s_rank,
                "retrieval_mode": (
                    "hybrid"       if (in_dense and in_sparse) else
                    "dense_only"   if in_dense else
                    "sparse_only"
                ),
            }
            results.append(result)

        if garbled_skipped:
            pass   # silent — logged in retrieve_with_debug() if needed

        return results

    def retrieve_with_debug(
        self,
        query:   str,
        top_k:   int = DEFAULTS["default_top_k"],
        filters: Optional[dict] = None,
    ) -> dict:
        """
        Same as retrieve() but also returns the intermediate ranked lists
        from dense and sparse retrievers — useful for RAG evaluation and
        ablation studies in your thesis.
        """
        pool = top_k * self.pool_multiplier
        chroma_filter = self._build_chroma_filter(filters)

        dense_results  = self._dense.retrieve(query, top_n=pool,
                                               filters=chroma_filter)
        sparse_raw     = self._bm25_index.get_scores(query, top_n=pool * 2)
        sparse_results = self._apply_metadata_filter_bm25(sparse_raw, filters)[:pool]

        fused = reciprocal_rank_fusion(
            [dense_results, sparse_results],
            k=self.rrf_k,
        )

        fused_results = self.retrieve(query, top_k=top_k, filters=filters)
        garbled_count = (len(fused) - top_k -
                         sum(1 for cid, _ in fused[:top_k * 3]
                             if cid in self._id_to_chunk and
                             is_garbled_chunk(self._id_to_chunk[cid].get("text",""))))

        return {
            "query":          query,
            "filters":        filters,
            "top_k":          top_k,
            "garbled_skipped": max(garbled_count, 0),
            "dense_pool":  [
                {"rank": i+1, "chunk_id": cid, "dense_score": round(s, 4)}
                for i, (cid, s) in enumerate(dense_results)
            ],
            "sparse_pool": [
                {"rank": i+1, "chunk_id": cid, "sparse_score": round(float(s), 4)}
                for i, (cid, s) in enumerate(sparse_results)
            ],
            "fused_top_k": fused_results,
        }


# ── Pretty printer ────────────────────────────────────────────────────────────

def print_results(results: list[dict], query: str, show_text: bool = True) -> None:
    """Print retrieval results in a readable format."""
    print(f"\n{'═'*70}")
    print(f"  Query : {query}")
    print(f"  Found : {len(results)} chunks")
    print(f"{'─'*70}")

    for i, r in enumerate(results, 1):
        country   = r.get("country")  or r.get("wg") or "—"
        doc_id    = r.get("doc_id", "")
        section   = (r.get("section") or "")[:55]
        year      = r.get("year", "")
        rrf       = r.get("rrf_score", 0)
        d_score   = r.get("dense_score")
        s_score   = r.get("sparse_score")
        mode      = r.get("retrieval_mode", "")
        mode_tag  = {"hybrid": "D+S", "dense_only": "D  ", "sparse_only": " S "}.get(mode, "???")

        d_score_str = f"{d_score:.4f}" if d_score is not None else "—"
        s_score_str = f"{s_score:.4f}" if s_score is not None else "—"

        print(f"\n  #{i}  RRF={rrf:.5f}  [{mode_tag}]  "
              f"dense={d_score_str:>8}  sparse={s_score_str:>9}")
        print(f"      {country:<18}  {doc_id:<38}  {year}")
        print(f"      Section: {section}")
        if show_text:
            snippet = (r.get("text") or "")[:180].replace("\n", " ")
            print(f"      Text   : {snippet}...")

    print(f"\n{'═'*70}")


# ── CLI test harness ──────────────────────────────────────────────────────────

DEMO_QUERIES = [
    {
        "label":   "factual_ndc",
        "query":   "What is India's renewable energy target by 2030?",
        "filters": {"country": "India"},
        "top_k":   5,
    },
    {
        "label":   "ipcc_wg3_mitigation",
        "query":   "What does IPCC WG3 recommend for energy sector decarbonisation by 2030?",
        "filters": {"$and": [{"source": "IPCC_AR6"}, {"wg": "WG3"}]},
        "top_k":   5,
    },
    {
        "label":   "gap_detection",
        "query":   "What are Indonesia's climate adaptation commitments and gaps?",
        "filters": {"country": "Indonesia"},
        "top_k":   5,
    },
    {
        "label":   "cross_country_ndc2",
        "query":   "Carbon neutrality net zero target year commitment",
        "filters": {"$and": [{"doc_type": "NDC"}, {"ndc_cycle": 2}]},
        "top_k":   8,
    },
    {
        "label":   "global_comparison",
        "query":   "How do G20 NDC commitments compare to 1.5°C Paris Agreement pathway?",
        "filters": None,
        "top_k":   5,
    },
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Hybrid retrieval module — smoke test and CLI interface"
    )
    parser.add_argument("--query",      default=None,
                        help="Custom query string")
    parser.add_argument("--top_k",      default=5, type=int)
    parser.add_argument("--country",    default=None,
                        help="Filter by country name")
    parser.add_argument("--wg",         default=None,
                        help="Filter by IPCC working group (WG1, WG2, WG3, SYR)")
    parser.add_argument("--ndc_cycle",  default=None, type=int,
                        help="Filter by NDC cycle (1 or 2)")
    parser.add_argument("--doc_type",   default=None,
                        help="Filter by doc_type (NDC or IPCC_AR6)")
    parser.add_argument("--debug",      action="store_true",
                        help="Show dense and sparse intermediate lists")
    parser.add_argument("--no_text",    action="store_true",
                        help="Hide chunk text in output")
    # Paths
    parser.add_argument("--chunks_file", default=DEFAULTS["chunks_file"])
    parser.add_argument("--vectorstore", default=DEFAULTS["vectorstore"])
    parser.add_argument("--device",      default=DEFAULTS["device"])
    args = parser.parse_args()

    # ── Initialise retriever ──────────────────────────────────────────────────
    retriever = HybridRetriever(
        chunks_file = args.chunks_file,
        vectorstore = args.vectorstore,
        device      = args.device,
    )

    # ── Build filters from CLI args ───────────────────────────────────────────
    cli_filters: dict = {}
    if args.country:    cli_filters["country"]   = args.country
    if args.wg:         cli_filters["wg"]         = args.wg
    if args.ndc_cycle:  cli_filters["ndc_cycle"]  = args.ndc_cycle
    if args.doc_type:   cli_filters["doc_type"]   = args.doc_type

    # ── Custom query mode ─────────────────────────────────────────────────────
    if args.query:
        filters = cli_filters or None
        if args.debug:
            debug = retriever.retrieve_with_debug(args.query, args.top_k, filters)
            print(f"\n  Dense pool (top 5 of {len(debug['dense_pool'])}):")
            for r in debug["dense_pool"][:5]:
                print(f"    #{r['rank']:>2}  {r['chunk_id']:<55}  {r['dense_score']:.4f}")
            print(f"\n  Sparse pool (top 5 of {len(debug['sparse_pool'])}):")
            for r in debug["sparse_pool"][:5]:
                print(f"    #{r['rank']:>2}  {r['chunk_id']:<55}  {r['sparse_score']:.4f}")
            results = debug["fused_top_k"]
        else:
            results = retriever.retrieve(args.query, args.top_k, filters)

        print_results(results, args.query, show_text=not args.no_text)
        return

    # ── Demo mode: run all 5 preset queries ───────────────────────────────────
    print("\n" + "═"*70)
    print("  HYBRID RETRIEVER — DEMO (5 preset queries)")
    print("  Dense (ChromaDB bge-large-en-v1.5) + Sparse (BM25) + RRF")
    print("═"*70)

    for demo in DEMO_QUERIES:
        t0      = time.time()
        results = retriever.retrieve(
            query   = demo["query"],
            top_k   = demo["top_k"],
            filters = demo["filters"],
        )
        elapsed = time.time() - t0
        print_results(results, demo["query"], show_text=not args.no_text)

        # Retrieval stats
        hybrid_count = sum(1 for r in results if r["retrieval_mode"] == "hybrid")
        print(f"  Latency: {elapsed*1000:.0f}ms  |  "
              f"Hybrid hits: {hybrid_count}/{len(results)}  |  "
              f"Filter: {demo['filters']}")

    print("\n  Demo complete. Run with --query 'your question' for custom queries.")
    print("  Add --debug to see dense and sparse intermediate ranked lists.\n")


if __name__ == "__main__":
    main()
