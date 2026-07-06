"""
===========================================================================
  G20 Gap Analysis — Transport Decarbonisation ONLY
  ==================================================
  Runs ONLY transport_decarbonisation across all 17 G20 countries.
  loss_and_damage is already complete — this script skips it entirely.

  VERIFIED dimension name from gap_analysis.py:
    transport_decarbonisation   ✓

  PLACE IN:  D:\Thesis\
  RUN WITH:
    cd D:\Thesis
    & d:\Thesis\thesis_env\Scripts\python.exe run_transport_decarbonisation.py

  RUN AFTER MIDNIGHT UTC for fresh Groq rate limit quota.
  Expected runtime: ~25 minutes (17 countries × ~90 seconds each)
===========================================================================
"""

import sys
import os
import json
import time
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

from gap_analysis import GapAnalyser

# ── Configuration ─────────────────────────────────────────────────────────
OUTPUT_DIR = THIS_DIR / "data" / "gap_reports" / "single"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DIMENSION = "transport_decarbonisation"

COUNTRIES = [
    "India",
    "China",
    "USA",
    "EU",
    "UK",
    "Japan",
    "Australia",
    "Brazil",
    "Indonesia",
    "South Korea",
    "Canada",
    "Mexico",
    "South Africa",
    "Saudi Arabia",
    "Argentina",
    "Turkey",
    "Russia",
]

DELAY = 8   # seconds between Groq API calls


# ── Helpers ───────────────────────────────────────────────────────────────

def to_dict(obj):
    """Recursively convert any Python object to a plain dict."""
    if isinstance(obj, dict):
        return {k: to_dict(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_dict(i) for i in obj]
    if hasattr(obj, '__dict__'):
        return {k: to_dict(v) for k, v in vars(obj).items()
                if not k.startswith('_')}
    if hasattr(obj, '_asdict'):
        return to_dict(obj._asdict())
    try:
        import dataclasses
        if dataclasses.is_dataclass(obj):
            return to_dict(dataclasses.asdict(obj))
    except Exception:
        pass
    return obj


def extract_score(cr: dict):
    """Extract best severity score and label from a country result dict."""
    gf = cr.get("gap_findings", [])
    if gf:
        scores = [
            float(g.get("severity_score", g.get("score", 0)))
            for g in gf
            if isinstance(g, dict)
            and (g.get("severity_score") or g.get("score"))
        ]
        score    = max(scores) if scores else None
        severity = gf[0].get("severity") if gf else None
        return score, severity

    score    = cr.get("overall_severity_score", cr.get("gap_score"))
    severity = cr.get("overall_severity", cr.get("severity", "ABSENT"))
    return score, severity


def score_to_label(s) -> str:
    if s is None:
        return "ABSENT"
    s = float(s)
    if s >= 0.80: return "CRITICAL"
    if s >= 0.50: return "SIGNIFICANT"
    if s >= 0.20: return "PARTIAL"
    return "ALIGNED"


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    print("=" * 58)
    print(f"  Transport Decarbonisation Gap Analysis")
    print(f"  Dimension : {DIMENSION}")
    print(f"  Countries : {len(COUNTRIES)}")
    print(f"  Backend   : groq (LLaMA-3.1-8B — free)")
    print(f"  Delay     : {DELAY}s between calls")
    print("=" * 58)

    # Check which countries already have saved results
    already_done = []
    to_run       = []
    for country in COUNTRIES:
        safe = country.replace(" ", "_")
        out  = OUTPUT_DIR / f"{DIMENSION}__{safe}.json"
        if out.exists():
            already_done.append(country)
        else:
            to_run.append(country)

    if already_done:
        print(f"\n  Already saved ({len(already_done)}): "
              f"{', '.join(already_done)}")
    print(f"  To run ({len(to_run)}): {', '.join(to_run) if to_run else 'none'}")

    if not to_run:
        print("\n  All countries already complete.")
        return

    print("\n  Initialising GapAnalyser with groq backend...")
    analyser = GapAnalyser(backend="groq")
    print("  Ready.\n")

    total  = len(to_run)
    done   = 0
    failed = []

    for country in to_run:
        safe = country.replace(" ", "_")
        out  = OUTPUT_DIR / f"{DIMENSION}__{safe}.json"

        print(f"  [{done+1}/{total}] {country:<20} ... ",
              end="", flush=True)

        try:
            # Run single-country analysis — avoids pairwise step
            report = analyser.analyse(
                countries  = [country],
                dimension  = DIMENSION,
                ndc_cycles = [2],
                save       = False,
            )

            # Extract country_analyses — handles both dict and object
            if isinstance(report, dict):
                ca_raw = (report.get("country_analyses") or
                          report.get("results") or {})
            else:
                ca_raw = getattr(report, "country_analyses", {}) or {}

            # Convert to plain dict
            ca = ({k: to_dict(v) for k, v in ca_raw.items()}
                  if isinstance(ca_raw, dict)
                  else to_dict(ca_raw))

            # Get this country's result
            cr = ca.get(country, {})
            if not cr:
                # Try case-insensitive match
                for k, v in ca.items():
                    if k.lower() == country.lower():
                        cr = v
                        break

            score, severity = extract_score(cr)
            label = score_to_label(score)

            # Save immediately
            save_data = {
                "country":   country,
                "dimension": DIMENSION,
                "score":     score,
                "severity":  severity,
                "label":     label,
                "details":   cr,
            }
            with open(out, "w", encoding="utf-8") as f:
                json.dump(save_data, f, indent=2,
                          ensure_ascii=False, default=str)

            print(f"{label:<15} (score={score})")
            done += 1

        except KeyboardInterrupt:
            print(f"\n\nStopped by user.")
            print(f"Saved {done} of {total} countries.")
            print("Re-run to resume — completed files will be skipped.")
            if failed:
                for c, e in failed:
                    print(f"  {c}: {e}")
            sys.exit(0)

        except Exception as e:
            print(f"FAILED — {e}")
            failed.append((country, str(e)))
            done += 1

        time.sleep(DELAY)

    # ── Summary ───────────────────────────────────────────────────────────
    print(f"\n{'='*58}")
    print(f"  COMPLETE — {done}/{total} countries processed")
    print(f"{'='*58}")

    # Print results table
    print(f"\n  {'Country':<20} {'Label':<15} {'Score'}")
    print(f"  {'─'*45}")
    for country in COUNTRIES:
        safe = country.replace(" ", "_")
        out  = OUTPUT_DIR / f"{DIMENSION}__{safe}.json"
        if out.exists():
            try:
                with open(out) as f:
                    d = json.load(f)
                lbl   = d.get("label", "?")
                score = d.get("score")
                icon  = {"CRITICAL":"🔴","SIGNIFICANT":"🟠",
                         "ALIGNED":"🟢","ABSENT":"⚪"}.get(lbl, "❓")
                print(f"  {country:<20} {icon} {lbl:<12} "
                      f"({score})")
            except Exception:
                print(f"  {country:<20} ERROR reading file")
        else:
            fail_reason = next(
                (e for c, e in failed if c == country), "not run")
            print(f"  {country:<20} ❌ FAILED — {fail_reason}")

    if failed:
        print(f"\n  {len(failed)} failures — re-run to retry them.")

    print(f"\n  Results saved to: {OUTPUT_DIR}")
    print("  Next step: run fix_and_build_matrix.py to rebuild "
          "g20_summary_final.csv and regenerate heatmap.")


if __name__ == "__main__":
    main()
