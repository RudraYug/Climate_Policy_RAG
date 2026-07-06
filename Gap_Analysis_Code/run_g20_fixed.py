"""
===========================================================================
  G20 Gap Analysis — Fixed Runner v2
  ====================================
  Fixes: CountryAnalysis object extraction + JSON parse fallback
  
  PLACE IN: D:\Thesis\
  RUN WITH: 
    cd D:\Thesis
    & d:\Thesis\thesis_env\Scripts\python.exe run_g20_fixed.py
===========================================================================
"""

import sys
import os
import json
import csv
import time
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

from gap_analysis import GapAnalyser

OUTPUT_DIR = THIS_DIR / "data" / "gap_reports" / "single"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

COUNTRIES = [
    "India", "China", "USA", "EU", "UK",
    "Japan", "Australia", "Brazil", "Indonesia",
    "South Korea", "Canada", "Mexico", "South Africa",
    "Saudi Arabia", "Argentina", "Turkey", "Russia",
]

# Only the 5 dimensions not yet done
# renewable_energy + lulucf_forestry already complete
DIMENSIONS = [
    "net_zero_pathway",
    "methane_reduction",
    "adaptation_finance",
    "industrial_decarbonisation",
    "transport_targets",
]

DELAY = 8  # seconds between Groq calls


# ── Universal object-to-dict converter ───────────────────────────────────

def to_dict(obj):
    """Convert any object to a plain dict recursively."""
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return {k: to_dict(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_dict(i) for i in obj]
    # Handle dataclasses, namedtuples, custom objects
    if hasattr(obj, '__dict__'):
        return {k: to_dict(v) for k, v in vars(obj).items()
                if not k.startswith('_')}
    if hasattr(obj, '_asdict'):
        return to_dict(obj._asdict())
    # For dataclasses
    try:
        import dataclasses
        if dataclasses.is_dataclass(obj):
            return to_dict(dataclasses.asdict(obj))
    except Exception:
        pass
    return obj  # primitive value


# ── Score extractor ───────────────────────────────────────────────────────

def extract_score_and_severity(country_dict: dict):
    """
    Extract severity score from a country result dict.
    Handles multiple possible key names.
    """
    # Try gap_findings first
    gap_findings = country_dict.get("gap_findings", [])
    if gap_findings and isinstance(gap_findings, list):
        scores = []
        severity = None
        for g in gap_findings:
            if not isinstance(g, dict):
                g = to_dict(g)
            s = (g.get("severity_score") or
                 g.get("score") or
                 g.get("gap_score"))
            if s is not None:
                try:
                    scores.append(float(s))
                except (TypeError, ValueError):
                    pass
            if severity is None:
                severity = g.get("severity")
        if scores:
            return max(scores), severity or "SIGNIFICANT"

    # Try top-level score fields
    for key in ["overall_severity_score", "gap_score",
                "severity_score", "score"]:
        val = country_dict.get(key)
        if val is not None:
            try:
                score = float(val)
                sev = (country_dict.get("overall_severity") or
                       country_dict.get("severity") or
                       score_to_label_raw(score))
                return score, sev
            except (TypeError, ValueError):
                pass

    # Default — no score found
    return None, country_dict.get("overall_severity",
                                   country_dict.get("severity", "ABSENT"))


def score_to_label_raw(score) -> str:
    if score is None:
        return "ABSENT"
    try:
        s = float(score)
    except (TypeError, ValueError):
        return "ABSENT"
    if s >= 0.80: return "CRITICAL"
    if s >= 0.50: return "SIGNIFICANT"
    if s >= 0.20: return "PARTIAL"
    return "ALIGNED"


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  G20 Gap Analysis — Fixed Runner v2")
    print(f"  Dimensions : {len(DIMENSIONS)}")
    print(f"  Countries  : {len(COUNTRIES)}")
    print(f"  Total runs : {len(DIMENSIONS) * len(COUNTRIES)}")
    print("  Backend    : groq (LLaMA-3.1-8B, free)")
    print("=" * 60)

    print("\nInitialising GapAnalyser...")
    analyser = GapAnalyser(backend="groq")
    print("Ready.\n")

    total  = len(DIMENSIONS) * len(COUNTRIES)
    done   = 0
    failed = []

    for dim in DIMENSIONS:
        print(f"\n{'='*60}")
        print(f"  DIMENSION: {dim}")
        print(f"{'='*60}")

        for country in COUNTRIES:
            safe     = country.replace(" ", "_")
            out_path = OUTPUT_DIR / f"{dim}__{safe}.json"

            if out_path.exists():
                print(f"  [SKIP] {country:<20} already saved")
                done += 1
                continue

            print(f"  [{done+1}/{total}] {country:<20} ... ",
                  end="", flush=True)

            try:
                # Run single-country analysis
                report = analyser.analyse(
                    countries  = [country],
                    dimension  = dim,
                    ndc_cycles = [2],
                    save       = False,
                )

                # ── Extract country_analyses from report ──────────────
                # Handle both dict and object responses
                if isinstance(report, dict):
                    ca_raw = (report.get("country_analyses") or
                              report.get("results") or {})
                elif hasattr(report, "country_analyses"):
                    ca_raw = report.country_analyses
                elif hasattr(report, "__dict__"):
                    r_dict = to_dict(report)
                    ca_raw = (r_dict.get("country_analyses") or
                              r_dict.get("results") or {})
                else:
                    ca_raw = {}

                # ── Convert country_analyses to plain dict ────────────
                if isinstance(ca_raw, dict):
                    ca = {k: to_dict(v) for k, v in ca_raw.items()}
                else:
                    ca = to_dict(ca_raw)

                # ── Get this country's result ─────────────────────────
                country_result = ca.get(country, {})
                if not country_result:
                    # Try case-insensitive match
                    for k, v in ca.items():
                        if k.lower() == country.lower():
                            country_result = v
                            break

                score, severity = extract_score_and_severity(
                    country_result)
                label = score_to_label_raw(score)

                # ── Save to JSON ──────────────────────────────────────
                save_data = {
                    "country":   country,
                    "dimension": dim,
                    "score":     score,
                    "severity":  severity,
                    "label":     label,
                    "details":   country_result,
                }
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(save_data, f, indent=2,
                              ensure_ascii=False, default=str)

                print(f"{label:<15} score={score}")
                done += 1

            except KeyboardInterrupt:
                print(f"\n\nStopped by user.")
                print(f"Saved: {done} analyses")
                print(f"Re-run to resume from where you stopped.")
                build_summary_csv(THIS_DIR)
                sys.exit(0)

            except Exception as e:
                print(f"FAILED — {e}")
                import traceback
                traceback.print_exc()
                failed.append((country, dim, str(e)))
                done += 1

            time.sleep(DELAY)

    # Build summary when all done
    build_summary_csv(THIS_DIR)

    if failed:
        print(f"\n{len(failed)} analyses failed:")
        for c, d, e in failed:
            print(f"  {c} / {d}: {e}")

    print(f"\nDone. {done}/{total} processed.")


# ── Build final summary CSV ───────────────────────────────────────────────

def build_summary_csv(base_dir: Path):
    """Build 17×7 summary CSV from all saved single results
    plus the original complete files."""

    print("\nBuilding g20_summary.csv...")

    ALL_DIMS = [
        "renewable_energy",
        "net_zero_pathway",
        "methane_reduction",
        "adaptation_finance",
        "industrial_decarbonisation",
        "transport_targets",
        "lulucf_forestry",
    ]

    COUNTRIES_LIST = [
        "India", "China", "USA", "EU", "UK",
        "Japan", "Australia", "Brazil", "Indonesia",
        "South Korea", "Canada", "Mexico", "South Africa",
        "Saudi Arabia", "Argentina", "Turkey", "Russia",
    ]

    single_dir = base_dir / "data" / "gap_reports" / "single"
    orig_dir   = base_dir / "data" / "gap_reports"
    csv_path   = orig_dir / "g20_summary.csv"

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Country"] + ALL_DIMS)

        for country in COUNTRIES_LIST:
            safe = country.replace(" ", "_")
            row  = [country]

            for dim in ALL_DIMS:
                label = "MISSING"

                # 1. Check single output dir
                sp = single_dir / f"{dim}__{safe}.json"
                if sp.exists():
                    try:
                        with open(sp) as jf:
                            d = json.load(jf)
                        label = d.get("label",
                                score_to_label_raw(d.get("score")))
                        row.append(label)
                        continue
                    except Exception:
                        pass

                # 2. Check original gap_reports folder
                for orig in orig_dir.glob(f"gap_{dim}_*.json"):
                    try:
                        with open(orig) as jf:
                            od = json.load(jf)
                        ca = od.get("country_analyses", {})
                        if country in ca:
                            cr = to_dict(ca[country])
                            sc, _ = extract_score_and_severity(cr)
                            label = score_to_label_raw(sc)
                            break
                    except Exception:
                        pass

                row.append(label)

            writer.writerow(row)

    print(f"[SAVED] {csv_path}")
    return csv_path


if __name__ == "__main__":
    main()
