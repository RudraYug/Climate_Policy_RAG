"""
===========================================================================
  CHROMADB FILTER SYNTAX FIX  +  SMOKE TEST RUNNER
  Master's Thesis: LLM-Powered Climate Policy Summariser & Gap Analyser
  University of Europe for Applied Sciences, Potsdam  |  2026
===========================================================================

ROOT CAUSE OF THE TWO WARNINGS
-------------------------------
ChromaDB ≥ 0.4.x requires that multiple metadata conditions in a single
`where` clause be combined with an explicit logical operator:

  WRONG  →  where={"source": "IPCC_AR6", "wg": "WG3"}
  RIGHT  →  where={"$and": [{"source": "IPCC_AR6"}, {"wg": "WG3"}]}

  WRONG  →  where={"doc_type": "NDC", "ndc_cycle": 2}
  RIGHT  →  where={"$and": [{"doc_type": "NDC"}, {"ndc_cycle": 2}]}

This file has two purposes:
  1. Drop-in patch  : Replaces run_smoke_tests() in embed_and_ingest.py
                      with the corrected filter syntax.
  2. Standalone tool: Run this script directly to re-run all 5 smoke tests
                      against your already-built vector store — no re-embedding.

Supported ChromaDB filter operators
-------------------------------------
  Logical  :  $and, $or
  Compare  :  $eq, $ne, $gt, $gte, $lt, $lte
  Membership: $in, $nin

Examples of valid filters you'll use in your RAG pipeline:
  Single field:
      where={"country": "India"}
      where={"ndc_cycle": 2}
      where={"source": "IPCC_AR6"}

  Multiple fields → must use $and:
      where={"$and": [{"source": "IPCC_AR6"}, {"wg": "WG3"}]}
      where={"$and": [{"doc_type": "NDC"}, {"ndc_cycle": 2}]}
      where={"$and": [{"country": "China"}, {"ndc_cycle": 2}]}

  OR across countries:
      where={"$or": [{"country": "China"}, {"country": "India"}]}

  Range / inequality:
      where={"emissions_rank": {"$lte": 5}}   # top 5 emitters
      where={"year": {"$gte": 2021}}           # NDC2 era

  Combined $and + $or:
      where={"$and": [
          {"doc_type": "NDC"},
          {"$or": [{"country": "China"}, {"country": "India"}]}
      ]}

Usage
-----
  python fix_smoke_tests.py                          # re-run smoke tests
  python fix_smoke_tests.py --vectorstore path/to/vs # custom path
  python fix_smoke_tests.py --collection my_col      # custom collection name
===========================================================================
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

try:
    import chromadb
except ImportError:
    print("[ERROR] pip install chromadb")
    sys.exit(1)

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("[ERROR] pip install sentence-transformers")
    sys.exit(1)

# ── Defaults (match embed_and_ingest.py) ──────────────────────────────────────
DEFAULT_VECTORSTORE = "data/vectorstore"
DEFAULT_COLLECTION  = "climate_policy_rag"
DEFAULT_MODEL       = "BAAI/bge-large-en-v1.5"
QUERY_PREFIX        = "Represent this sentence for searching relevant passages: "


# ── The 5 corrected smoke tests ───────────────────────────────────────────────
# All multi-field filters now use $and — this is the only change vs the
# original embed_and_ingest.py smoke tests.

SMOKE_TESTS = [
    {
        "q":      "What is India's renewable energy target by 2030?",
        "filter": {"country": "India"},              # single field — no $and needed
        "type":   "factual_ndc",
        "n":      3,
    },
    {
        "q":      "What greenhouse gas reduction does IPCC WG3 recommend for 2030?",
        # FIX: was {"source": "IPCC_AR6", "wg": "WG3"} — invalid multi-field
        "filter": {"$and": [{"source": "IPCC_AR6"}, {"wg": "WG3"}]},
        "type":   "factual_ipcc",
        "n":      3,
    },
    {
        "q":      "How does China's NDC commitment compare to 1.5°C pathway?",
        "filter": None,                               # no filter — global search
        "type":   "comparative",
        "n":      3,
    },
    {
        "q":      "What are the gaps in Indonesia's climate adaptation commitments?",
        "filter": {"country": "Indonesia"},
        "type":   "gap_detection",
        "n":      3,
    },
    {
        "q":      "Net zero target year carbon neutrality",
        # FIX: was {"doc_type": "NDC", "ndc_cycle": 2} — invalid multi-field
        "filter": {"$and": [{"doc_type": "NDC"}, {"ndc_cycle": 2}]},
        "type":   "multi_country_ndc2",
        "n":      5,   # wider net for cross-country results
    },

    # ── Bonus tests added for thesis validation ───────────────────────────────
    {
        "q":      "What is the EU climate neutrality commitment and legislative framework?",
        "filter": {"country": "EU"},
        "type":   "eu_ndc_factual",
        "n":      3,
    },
    {
        "q":      "What does IPCC AR6 say about energy sector decarbonisation pathways?",
        "filter": {"$and": [{"source": "IPCC_AR6"}, {"wg": "WG3"}]},
        "type":   "ipcc_energy_sector",
        "n":      5,
    },
    {
        "q":      "Conditional versus unconditional NDC targets international climate finance",
        "filter": {"$or": [{"country": "Indonesia"}, {"country": "South Africa"},
                           {"country": "India"}]},
        "type":   "conditional_targets",
        "n":      5,
    },
    {
        "q":      "Carbon emission peak year peaking pathway",
        "filter": {"$and": [{"doc_type": "NDC"}, {"ndc_cycle": 2}]},
        "type":   "peak_emissions_ndc2",
        "n":      5,
    },
    {
        "q":      "Land use LULUCF forestry carbon sink removal",
        "filter": None,
        "type":   "lulucf_global",
        "n":      5,
    },
]


# ── Query helper ──────────────────────────────────────────────────────────────

def embed_query(model: SentenceTransformer, query: str) -> list[float]:
    prefixed = QUERY_PREFIX + query
    return model.encode(prefixed, normalize_embeddings=True,
                        convert_to_numpy=True).tolist()


def run_query(collection, q_emb: list[float], n: int,
              where_filter: dict | None) -> dict:
    kwargs = {
        "query_embeddings": [q_emb],
        "n_results":        n,
        "include":          ["documents", "metadatas", "distances"],
    }
    if where_filter:
        kwargs["where"] = where_filter
    return collection.query(**kwargs)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Re-run corrected smoke tests against existing ChromaDB vector store"
    )
    parser.add_argument("--vectorstore", default=DEFAULT_VECTORSTORE)
    parser.add_argument("--collection",  default=DEFAULT_COLLECTION)
    parser.add_argument("--model",       default=DEFAULT_MODEL)
    parser.add_argument("--device",      default="cpu",
                        choices=["cpu", "cuda", "mps"])
    args = parser.parse_args()

    vs_path = Path(args.vectorstore)
    if not vs_path.exists():
        print(f"[ERROR] Vector store not found: {vs_path.resolve()}")
        print("  Run embed_and_ingest.py first.")
        sys.exit(1)

    # ── Open existing collection (no re-embedding) ────────────────────────────
    print(f"\nOpening ChromaDB: {vs_path.resolve()}")
    client     = chromadb.PersistentClient(path=str(vs_path))
    collection = client.get_collection(args.collection)
    total      = collection.count()
    print(f"Collection '{args.collection}' — {total} vectors loaded")

    # ── Load embedding model ──────────────────────────────────────────────────
    print(f"\nLoading model: {args.model}  (device={args.device})")
    model = SentenceTransformer(args.model, device=args.device)

    # ── Run all smoke tests ───────────────────────────────────────────────────
    print(f"\nRunning {len(SMOKE_TESTS)} smoke tests...\n")
    print("=" * 70)

    all_results = []
    passed      = 0
    failed      = 0

    for test in SMOKE_TESTS:
        print(f"[{test['type']}]")
        print(f"  Query : {test['q']}")
        print(f"  Filter: {json.dumps(test['filter'])}")

        q_emb = embed_query(model, test["q"])

        try:
            res       = run_query(collection, q_emb, test["n"], test["filter"])
            docs      = res["documents"][0]
            metas     = res["metadatas"][0]
            distances = res["distances"][0]

            result = {
                "query":   test["q"],
                "type":    test["type"],
                "filter":  test["filter"],
                "status":  "pass",
                "top_results": [],
            }

            print(f"  Results (top {test['n']}):")
            for i, (doc, meta, dist) in enumerate(zip(docs, metas, distances)):
                sim     = round(1 - dist, 4)
                country = meta.get("country") or meta.get("wg") or "—"
                doc_id  = meta.get("doc_id", "")
                section = meta.get("section", "")[:55]
                snippet = doc[:100].replace("\n", " ") + "..."

                print(f"    #{i+1}  sim={sim:.3f}  "
                      f"{country:<15}  {doc_id:<35}  |  {section}")

                result["top_results"].append({
                    "rank":       i + 1,
                    "similarity": sim,
                    "distance":   round(dist, 4),
                    "country":    country,
                    "doc_id":     doc_id,
                    "section":    section,
                    "snippet":    snippet,
                    "year":       meta.get("year"),
                    "ndc_cycle":  meta.get("ndc_cycle"),
                    "source":     meta.get("source"),
                })

            all_results.append(result)
            passed += 1

        except Exception as e:
            print(f"  [FAIL] {e}")
            all_results.append({
                "query":  test["q"],
                "type":   test["type"],
                "filter": test["filter"],
                "status": "fail",
                "error":  str(e),
            })
            failed += 1

        print()

    # ── Summary ───────────────────────────────────────────────────────────────
    print("=" * 70)
    print(f"  Passed : {passed}/{len(SMOKE_TESTS)}")
    print(f"  Failed : {failed}/{len(SMOKE_TESTS)}")

    if failed == 0:
        print("\n  All smoke tests passed — vector store is healthy.")
    else:
        print(f"\n  {failed} test(s) still failing — check filter syntax above.")

    # ── Save results ──────────────────────────────────────────────────────────
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "vectorstore":  str(vs_path.resolve()),
        "collection":   args.collection,
        "model":        args.model,
        "total_vectors": total,
        "tests_passed": passed,
        "tests_failed": failed,
        "results":      all_results,

        # ── Cheat sheet: copy-paste ready filter examples ─────────────────────
        "filter_syntax_reference": {
            "single_field":   '{"country": "India"}',
            "multi_field_AND": '{"$and": [{"source": "IPCC_AR6"}, {"wg": "WG3"}]}',
            "multi_field_AND_ndc": '{"$and": [{"doc_type": "NDC"}, {"ndc_cycle": 2}]}',
            "or_countries":   '{"$or": [{"country": "China"}, {"country": "India"}]}',
            "range":          '{"emissions_rank": {"$lte": 5}}',
            "year_range":     '{"year": {"$gte": 2021}}',
            "complex":        '{"$and": [{"doc_type": "NDC"}, {"$or": [{"country": "China"}, {"country": "India"}]}]}',
        },
    }

    out_path = vs_path / "smoke_test_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n  Results saved → {out_path}")

    # ── Print the exact fix so you can patch embed_and_ingest.py ─────────────
    print("""
  WHAT TO PATCH IN embed_and_ingest.py (run_smoke_tests function)
  ──────────────────────────────────────────────────────────────────
  Replace these two test dict entries:

  OLD (broken):
    {"q": "IPCC WG3 ...", "filter": {"source": "IPCC_AR6", "wg": "WG3"}, ...}
    {"q": "Net zero ...", "filter": {"doc_type": "NDC", "ndc_cycle": 2}, ...}

  NEW (correct):
    {"q": "IPCC WG3 ...", "filter": {"$and": [{"source": "IPCC_AR6"}, {"wg": "WG3"}]}, ...}
    {"q": "Net zero ...", "filter": {"$and": [{"doc_type": "NDC"}, {"ndc_cycle": 2}]}, ...}

  RULE: In ChromaDB, whenever your where= dict has more than one key,
        wrap them in {"$and": [{"key1": val1}, {"key2": val2}]}.
""")


if __name__ == "__main__":
    main()
