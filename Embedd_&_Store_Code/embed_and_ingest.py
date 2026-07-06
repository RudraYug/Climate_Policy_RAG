"""
===========================================================================
  EMBED → CHROMADB INGESTION PIPELINE
  Master's Thesis: LLM-Powered Climate Policy Summariser & Gap Analyser
  University of Europe for Applied Sciences, Potsdam  |  2026
===========================================================================

MODEL CHOICE — Why BAAI/bge-large-en-v1.5
------------------------------------------
  Evaluated against all-MiniLM-L6-v2, all-mpnet-base-v2, e5-base-v2:

  Model                    MTEB-STS  MTEB-Retrieval  Dims  RAM(CPU)  Speed
  ─────────────────────────────────────────────────────────────────────────
  BAAI/bge-large-en-v1.5   83.1 %      ~54 %        1024   ~1.2 GB   ★★★
  all-MiniLM-L6-v2         56.3 %      ~41 %         384   ~90 MB    ★★★★★
  all-mpnet-base-v2         69.6 %      ~43 %         768   ~420 MB   ★★★★
  e5-base-v2                71.5 %      ~50 %         768   ~440 MB   ★★★★

  Decision factors for THIS corpus:
  ✓ Policy text is long, technical, abbreviation-dense (NDC, LULUCF, GWP,
    MtCO2e) — needs high-capacity 1024-d space to separate close concepts.
  ✓ Gap analysis queries are precise ("what does India commit on renewables
    vs what IPCC WG3 Ch6 recommends?") — requires high retrieval precision.
  ✓ 4 705 chunks × 1024 floats × 4 bytes = ~19 MB vectors — easily fits
    in local ChromaDB on any laptop.
  ✓ bge-large runs on CPU in ~30–60 min for 4 705 chunks — acceptable once.
  ✓ MIT licence — no commercial restrictions, citable in thesis.
  ✓ BGE requires a query prefix "Represent this sentence for searching
    relevant passages: " for queries (not for documents) — we handle this.

  If you have a CUDA GPU: set DEVICE = "cuda" in CONFIG below.
  If you are RAM-constrained (<8 GB): switch to BAAI/bge-base-en-v1.5
    (identical architecture, 768 dims, half the RAM, ~3% lower accuracy).

ChromaDB choice
---------------
  • Free, no API key, fully local, persists to disk.
  • Native metadata filtering: filter by country, ndc_cycle, doc_type, etc.
  • Python-native: chromadb.PersistentClient("path") — no server needed.
  • 4 705 chunks is well within ChromaDB's single-collection sweet spot
    (tested up to ~1 M vectors locally).

Outputs
-------
  data/vectorstore/              ← ChromaDB persistent directory
  data/vectorstore/ingest_report.json  ← stats + model card + timing

Usage
-----
  pip install chromadb sentence-transformers tqdm
  python embed_and_ingest.py

  # Options:
  python embed_and_ingest.py --chunks_file data/processed/chunks.jsonl
  python embed_and_ingest.py --device cuda          # GPU mode
  python embed_and_ingest.py --batch_size 64        # larger GPU batches
  python embed_and_ingest.py --model BAAI/bge-base-en-v1.5  # lighter model
  python embed_and_ingest.py --reset                # wipe & rebuild from scratch
  python embed_and_ingest.py --dry_run              # validate input, no embed
===========================================================================
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ── Dependency check ──────────────────────────────────────────────────────────
_missing = []
try:
    import chromadb
except ImportError:
    _missing.append("chromadb")
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    _missing.append("sentence-transformers")
try:
    from tqdm import tqdm
    TQDM = True
except ImportError:
    TQDM = False
    print("[INFO] tqdm not installed — no progress bars. pip install tqdm")

if _missing:
    print(f"\n[ERROR] Missing: {', '.join(_missing)}")
    print(f"        pip install {' '.join(_missing)}")
    sys.exit(1)


# ── Configuration ─────────────────────────────────────────────────────────────

CONFIG = {
    # ── Paths ─────────────────────────────────────────────────────────────────
    "chunks_file":   "data/processed/chunks.jsonl",
    "vectorstore":   "data/vectorstore",
    "collection":    "climate_policy_rag",

    # ── Model ─────────────────────────────────────────────────────────────────
    # Change to "BAAI/bge-base-en-v1.5" if RAM < 8 GB
    "model_name":    "BAAI/bge-large-en-v1.5",
    "embed_dim":     1024,           # bge-large output dimension

    # BGE requires this prefix on QUERIES (not documents) for optimal retrieval.
    # When you query ChromaDB later, prepend this to your question.
    # Documents are embedded WITHOUT this prefix.
    "query_prefix":  "Represent this sentence for searching relevant passages: ",

    # ── Hardware ──────────────────────────────────────────────────────────────
    # "cpu" works fine for 4705 chunks (~30–60 min one-time cost)
    # "cuda" if you have an NVIDIA GPU (cuts to ~3–5 min)
    # "mps"  if you have Apple Silicon Mac
    "device":        "cpu",

    # ── Batching ──────────────────────────────────────────────────────────────
    # CPU: 32 is safe. GPU with 8GB VRAM: try 128–256.
    "batch_size":    32,

    # ── ChromaDB ──────────────────────────────────────────────────────────────
    # Metadata fields to index for filtering (all others still stored).
    # These are the fields you'll filter on in RAG queries:
    #   .query(where={"country": "India"})
    #   .query(where={"ndc_cycle": 2})
    #   .query(where={"doc_type": "NDC", "country": {"$in": ["China","India"]}})
    "filterable_fields": [
        "country", "ndc_cycle", "year", "doc_type", "source",
        "wg", "section", "is_table", "withdrawn", "emissions_rank",
    ],
}

# Metadata fields whose values must be stored as strings in ChromaDB
# (ChromaDB only supports str, int, float, bool — not None)
CHROMA_STRING_FIELDS = {"country", "wg", "section", "source", "doc_type",
                        "language", "note", "file", "file_path",
                        "processed_at", "pipeline_ver", "relevance"}


# ── Data loading ──────────────────────────────────────────────────────────────

def load_chunks(path: Path) -> list[dict]:
    """Load chunks.jsonl — one JSON object per line."""
    chunks = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                chunks.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"  [WARN] Skipping line {i}: {e}")
    return chunks


# ── Metadata sanitiser ────────────────────────────────────────────────────────

def sanitise_metadata(chunk: dict) -> dict:
    """
    ChromaDB metadata requirements:
      - Values must be str | int | float | bool  (no None, no list, no dict)
      - Keys must be str
      - The 'text' field is stored separately as a document, not in metadata
    """
    meta = {}
    skip = {"text", "chunk_id", "doc_id"}   # handled separately

    for key, val in chunk.items():
        if key in skip:
            continue

        # Convert None → "" for string fields, 0 for numeric fields
        if val is None:
            if key in CHROMA_STRING_FIELDS:
                meta[key] = ""
            else:
                meta[key] = 0
        elif isinstance(val, bool):
            meta[key] = val          # ChromaDB accepts bool natively
        elif isinstance(val, (int, float)):
            meta[key] = val
        elif isinstance(val, str):
            meta[key] = val[:500]    # cap very long strings
        elif isinstance(val, (list, dict)):
            meta[key] = json.dumps(val)[:500]
        else:
            meta[key] = str(val)[:500]

    return meta


# ── Embedding engine ──────────────────────────────────────────────────────────

class BGEEmbedder:
    """
    Wraps BAAI/bge-large-en-v1.5 with correct encode settings.

    Key detail: bge-large was trained with normalise_embeddings=True.
    ChromaDB uses cosine similarity by default, which is equivalent to
    dot product on normalised vectors — so we normalise here to ensure
    similarity scores are in [-1, 1] and comparable across queries.
    """

    def __init__(self, model_name: str, device: str):
        print(f"\n  Loading model: {model_name}")
        print(f"  Device       : {device}")
        t0 = time.time()
        self.model = SentenceTransformer(model_name, device=device)
        self.model_name = model_name
        self.device = device
        elapsed = time.time() - t0
        print(f"  Model loaded in {elapsed:.1f}s")
        print(f"  Embedding dim: {self.model.get_sentence_embedding_dimension()}")

    def embed_documents(self, texts: list[str], batch_size: int,
                        show_progress: bool = True) -> list[list[float]]:
        """
        Embed a list of document texts.
        Documents are embedded WITHOUT the query prefix.
        Returns list of float lists (not numpy — ChromaDB needs plain Python).
        """
        all_embeddings = []
        batches = [texts[i:i+batch_size] for i in range(0, len(texts), batch_size)]

        it = tqdm(batches, desc="  Embedding", unit="batch") if (TQDM and show_progress) else batches
        for batch in it:
            embs = self.model.encode(
                batch,
                normalize_embeddings=True,   # required for bge cosine similarity
                show_progress_bar=False,
                convert_to_numpy=True,
            )
            all_embeddings.extend(embs.tolist())

        return all_embeddings

    def embed_query(self, query: str) -> list[float]:
        """
        Embed a single query WITH the BGE query prefix.
        Use this in your RAG retrieval code, not during ingestion.
        """
        prefixed = CONFIG["query_prefix"] + query
        emb = self.model.encode(
            prefixed,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return emb.tolist()


# ── ChromaDB setup ────────────────────────────────────────────────────────────

def get_or_create_collection(
    vectorstore_path: str,
    collection_name: str,
    reset: bool = False,
) -> tuple[chromadb.PersistentClient, Any]:
    """
    Create or open a persistent ChromaDB collection.
    Uses cosine similarity (correct for normalised bge embeddings).
    """
    client = chromadb.PersistentClient(path=vectorstore_path)

    if reset:
        try:
            client.delete_collection(collection_name)
            print(f"  [RESET] Deleted existing collection '{collection_name}'")
        except Exception:
            pass

    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={
            "hnsw:space": "cosine",       # cosine similarity
            "hnsw:construction_ef": 200,   # higher = better index quality (default 100)
            "hnsw:M": 16,                  # HNSW connectivity (default 16)
            "model": CONFIG["model_name"],
            "embed_dim": str(CONFIG["embed_dim"]),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "description": (
                "Climate policy RAG vector store. "
                "G20 NDC documents (2 cycles) + IPCC AR6 chapters. "
                "Embedded with BAAI/bge-large-en-v1.5."
            ),
        },
    )
    return client, collection


# ── Ingestion with resume ─────────────────────────────────────────────────────

def get_existing_ids(collection) -> set[str]:
    """Fetch all chunk_ids already in the collection (for resume support)."""
    try:
        result = collection.get(include=[])   # IDs only, no embeddings/docs
        return set(result["ids"])
    except Exception:
        return set()


def ingest_batches(
    collection,
    chunks: list[dict],
    embeddings: list[list[float]],
    batch_size: int,
    existing_ids: set[str],
) -> tuple[int, int]:
    """
    Upsert chunks + embeddings into ChromaDB in batches.
    Skips chunks whose IDs already exist (resume support).
    Returns (inserted, skipped).
    """
    to_insert = [
        (chunk, emb)
        for chunk, emb in zip(chunks, embeddings)
        if chunk["chunk_id"] not in existing_ids
    ]

    skipped = len(chunks) - len(to_insert)
    inserted = 0

    if not to_insert:
        return 0, skipped

    # Build batches
    batches = [to_insert[i:i+batch_size] for i in range(0, len(to_insert), batch_size)]
    it = tqdm(batches, desc="  Inserting", unit="batch") if TQDM else batches

    for batch in it:
        ids       = [c["chunk_id"]  for c, _ in batch]
        documents = [c["text"]      for c, _ in batch]
        metadatas = [sanitise_metadata(c) for c, _ in batch]
        embs      = [e               for _, e in batch]

        collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=embs,
            metadatas=metadatas,
        )
        inserted += len(batch)

    return inserted, skipped


# ── Verification queries ──────────────────────────────────────────────────────

def run_smoke_tests(collection, embedder: BGEEmbedder) -> list[dict]:
    """
    Run 5 representative queries to verify the vector store is working.
    Tests factual, comparative, and gap-detection query types.
    """
    test_queries = [
        {
            "q": "What is India's renewable energy target by 2030?",
            "filter": {"country": "India"},
            "type": "factual_ndc",
        },
        {
            "q": "What greenhouse gas reduction does IPCC WG3 recommend for 2030?",
            "filter": {"source": "IPCC_AR6", "wg": "WG3"},
            "type": "factual_ipcc",
        },
        {
            "q": "How does China's NDC commitment compare to 1.5°C pathway?",
            "filter": None,
            "type": "comparative",
        },
        {
            "q": "What are the gaps in Indonesia's climate adaptation commitments?",
            "filter": {"country": "Indonesia"},
            "type": "gap_detection",
        },
        {
            "q": "Net zero target year carbon neutrality",
            "filter": {"doc_type": "NDC", "ndc_cycle": 2},
            "type": "multi_country_ndc2",
        },
    ]

    results = []
    print("\n  Running smoke tests...")

    for test in test_queries:
        q_emb = embedder.embed_query(test["q"])
        kwargs = {
            "query_embeddings": [q_emb],
            "n_results": 3,
            "include": ["documents", "metadatas", "distances"],
        }
        if test["filter"]:
            kwargs["where"] = test["filter"]

        try:
            res = collection.query(**kwargs)
            top_docs   = res["documents"][0]
            top_metas  = res["metadatas"][0]
            top_dists  = res["distances"][0]

            result = {
                "query":   test["q"],
                "type":    test["type"],
                "filter":  test["filter"],
                "top_3": [
                    {
                        "rank":      i + 1,
                        "distance":  round(d, 4),
                        "similarity": round(1 - d, 4),
                        "country":   m.get("country", "N/A"),
                        "doc_id":    m.get("doc_id", "N/A"),
                        "section":   m.get("section", "")[:60],
                        "snippet":   doc[:120] + "...",
                    }
                    for i, (doc, m, d) in enumerate(zip(top_docs, top_metas, top_dists))
                ],
            }
            results.append(result)

            # Print summary
            print(f"\n    [{test['type']}] {test['q'][:60]}...")
            for r in result["top_3"]:
                print(f"      #{r['rank']} sim={r['similarity']:.3f}  "
                      f"{r['country'] or r['doc_id'][:30]}  |  {r['section'][:40]}")

        except Exception as e:
            print(f"    [WARN] Query failed: {e}")
            results.append({"query": test["q"], "error": str(e)})

    return results


# ── Report writer ─────────────────────────────────────────────────────────────

def write_report(
    vectorstore_path: str,
    collection_name: str,
    n_chunks: int,
    n_inserted: int,
    n_skipped: int,
    elapsed_embed: float,
    elapsed_insert: float,
    smoke_results: list[dict],
    args: argparse.Namespace,
) -> None:
    report = {
        "generated_at":      datetime.now(timezone.utc).isoformat(),
        "pipeline_version":  "1.0",

        "model": {
            "name":          args.model,
            "embed_dim":     CONFIG["embed_dim"],
            "device":        args.device,
            "query_prefix":  CONFIG["query_prefix"],
            "normalise":     True,
            "similarity":    "cosine",
            "mteb_sts":      "83.1%",
            "mteb_retrieval": "~54%",
            "licence":       "MIT",
        },

        "vectorstore": {
            "backend":       "ChromaDB (local persistent)",
            "path":          vectorstore_path,
            "collection":    collection_name,
            "hnsw_space":    "cosine",
            "hnsw_ef":       200,
            "hnsw_M":        16,
        },

        "corpus": {
            "total_chunks":   n_chunks,
            "inserted":       n_inserted,
            "skipped":        n_skipped,
        },

        "timing": {
            "embedding_sec":  round(elapsed_embed, 1),
            "insertion_sec":  round(elapsed_insert, 1),
            "total_sec":      round(elapsed_embed + elapsed_insert, 1),
            "chunks_per_sec": round(n_chunks / max(elapsed_embed, 1), 1),
        },

        "smoke_tests": smoke_results,

        "usage_example": {
            "description": "How to query this vector store in your RAG pipeline",
            "python": (
                "import chromadb\n"
                "from sentence_transformers import SentenceTransformer\n\n"
                f"client = chromadb.PersistentClient(path='{vectorstore_path}')\n"
                f"col    = client.get_collection('{collection_name}')\n"
                f"model  = SentenceTransformer('{args.model}')\n\n"
                "# IMPORTANT: Always prepend query prefix for BGE\n"
                f"prefix = '{CONFIG['query_prefix']}'\n"
                "query  = prefix + 'What is India renewable energy target 2030?'\n"
                "q_emb  = model.encode(query, normalize_embeddings=True).tolist()\n\n"
                "# Basic retrieval\n"
                "results = col.query(query_embeddings=[q_emb], n_results=5,\n"
                "                    include=['documents','metadatas','distances'])\n\n"
                "# Filtered retrieval (metadata filter)\n"
                "results = col.query(query_embeddings=[q_emb], n_results=5,\n"
                "    where={'country': 'India'},\n"
                "    include=['documents','metadatas','distances'])\n\n"
                "# Cross-country gap query\n"
                "results = col.query(query_embeddings=[q_emb], n_results=10,\n"
                "    where={'$or': [{'country': 'China'}, {'wg': 'WG3'}]},\n"
                "    include=['documents','metadatas','distances'])"
            ),
        },
    }

    report_path = Path(vectorstore_path) / "ingest_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n  [SAVED] {report_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Embed chunks.jsonl and ingest into local ChromaDB"
    )
    parser.add_argument("--chunks_file",  default=CONFIG["chunks_file"])
    parser.add_argument("--vectorstore",  default=CONFIG["vectorstore"])
    parser.add_argument("--collection",   default=CONFIG["collection"])
    parser.add_argument("--model",        default=CONFIG["model_name"])
    parser.add_argument("--device",       default=CONFIG["device"],
                        choices=["cpu", "cuda", "mps"])
    parser.add_argument("--batch_size",   default=CONFIG["batch_size"], type=int)
    parser.add_argument("--reset",        action="store_true",
                        help="Delete existing collection and rebuild from scratch")
    parser.add_argument("--dry_run",      action="store_true",
                        help="Load and validate data only; no embedding or insertion")
    args = parser.parse_args()

    print("=" * 64)
    print("  EMBED → CHROMADB INGESTION PIPELINE")
    print(f"  Model       : {args.model}")
    print(f"  Device      : {args.device}")
    print(f"  Batch size  : {args.batch_size}")
    print(f"  Chunks file : {args.chunks_file}")
    print(f"  Vector store: {args.vectorstore}")
    print(f"  Collection  : {args.collection}")
    print(f"  Reset       : {args.reset}")
    print(f"  Dry run     : {args.dry_run}")
    print("=" * 64)

    # ── 1. Load chunks ────────────────────────────────────────────────────────
    chunks_path = Path(args.chunks_file)
    if not chunks_path.exists():
        print(f"\n[ERROR] Chunks file not found: {chunks_path}")
        print("  Run pdf_to_chunks.py first.")
        sys.exit(1)

    print(f"\n  Loading chunks from {chunks_path}...")
    chunks = load_chunks(chunks_path)
    print(f"  Loaded {len(chunks)} chunks")

    # Validate chunk_ids are unique
    ids = [c["chunk_id"] for c in chunks]
    if len(ids) != len(set(ids)):
        dupes = len(ids) - len(set(ids))
        print(f"  [WARN] {dupes} duplicate chunk_ids detected — will be overwritten on upsert")

    if args.dry_run:
        print("\n  [DRY RUN] Validation complete. No embedding or insertion performed.")
        print(f"  Sample chunk keys: {list(chunks[0].keys())}")
        print(f"  Sample metadata:   country={chunks[0].get('country')}, "
              f"ndc_cycle={chunks[0].get('ndc_cycle')}, "
              f"doc_type={chunks[0].get('doc_type')}")
        return

    # ── 2. Load embedding model ───────────────────────────────────────────────
    embedder = BGEEmbedder(args.model, args.device)

    # ── 3. Extract texts ──────────────────────────────────────────────────────
    print(f"\n  Extracting texts from {len(chunks)} chunks...")
    texts = [c["text"] for c in chunks]

    # ── 4. Embed all documents ────────────────────────────────────────────────
    print(f"\n  Embedding {len(texts)} chunks")
    print(f"  Est. time  : {'~30–60 min (CPU)' if args.device == 'cpu' else '~3–5 min (GPU)'}")
    print(f"  Note: This runs ONCE. ChromaDB caches everything persistently.\n")

    t_embed_start = time.time()
    embeddings = embedder.embed_documents(
        texts,
        batch_size=args.batch_size,
        show_progress=True,
    )
    elapsed_embed = time.time() - t_embed_start
    print(f"\n  Embedding done in {elapsed_embed:.1f}s  "
          f"({len(texts)/elapsed_embed:.1f} chunks/sec)")

    # ── 5. Set up ChromaDB ────────────────────────────────────────────────────
    print(f"\n  Opening ChromaDB at: {args.vectorstore}")
    Path(args.vectorstore).mkdir(parents=True, exist_ok=True)
    client, collection = get_or_create_collection(
        args.vectorstore, args.collection, reset=args.reset
    )

    existing_ids = get_existing_ids(collection)
    print(f"  Existing chunks in collection: {len(existing_ids)}")
    print(f"  New chunks to insert         : {len(chunks) - len(existing_ids)}")

    # ── 6. Insert into ChromaDB ───────────────────────────────────────────────
    print("\n  Inserting into ChromaDB...")
    t_insert_start = time.time()
    n_inserted, n_skipped = ingest_batches(
        collection, chunks, embeddings,
        batch_size=args.batch_size * 2,   # insert can use larger batches
        existing_ids=existing_ids,
    )
    elapsed_insert = time.time() - t_insert_start

    total_in_collection = collection.count()
    print(f"\n  Inserted : {n_inserted}")
    print(f"  Skipped  : {n_skipped} (already existed)")
    print(f"  Total in collection: {total_in_collection}")
    print(f"  Insert time: {elapsed_insert:.1f}s")

    # ── 7. Smoke tests ────────────────────────────────────────────────────────
    smoke_results = run_smoke_tests(collection, embedder)

    # ── 8. Write report ───────────────────────────────────────────────────────
    write_report(
        vectorstore_path=args.vectorstore,
        collection_name=args.collection,
        n_chunks=len(chunks),
        n_inserted=n_inserted,
        n_skipped=n_skipped,
        elapsed_embed=elapsed_embed,
        elapsed_insert=elapsed_insert,
        smoke_results=smoke_results,
        args=args,
    )

    # ── 9. Final summary ──────────────────────────────────────────────────────
    print("\n" + "=" * 64)
    print("  INGESTION COMPLETE")
    print(f"  Collection '{args.collection}' → {total_in_collection} vectors")
    print(f"  Stored at  : {Path(args.vectorstore).resolve()}")
    print(f"  Total time : {(elapsed_embed + elapsed_insert)/60:.1f} min")
    print("=" * 64)
    print("""
  NEXT STEP — query this store in your RAG pipeline:

    import chromadb
    from sentence_transformers import SentenceTransformer

    client = chromadb.PersistentClient(path="data/vectorstore")
    col    = client.get_collection("climate_policy_rag")
    model  = SentenceTransformer("BAAI/bge-large-en-v1.5")

    # Always prepend the BGE query prefix:
    prefix = "Represent this sentence for searching relevant passages: "
    q_emb  = model.encode(prefix + "your question here",
                           normalize_embeddings=True).tolist()

    results = col.query(query_embeddings=[q_emb], n_results=5,
                        include=["documents","metadatas","distances"])
""")


if __name__ == "__main__":
    main()
