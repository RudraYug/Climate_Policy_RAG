"""
===========================================================================
  G20 Gap Analysis — Matrix Builder & Score Fixer
  =================================================
  1. Reads all saved JSON files from data/gap_reports/single/
  2. Fixes ABSENT entries where pass_c failed using measurability heuristic
  3. Applies known correct values from terminal output
  4. Builds complete g20_summary.csv and g20_score_matrix.csv

  PLACE IN: D:\Thesis\
  RUN WITH:
    cd D:\Thesis
    & d:\Thesis\thesis_env\Scripts\python.exe fix_and_build_matrix.py
===========================================================================
"""

import json
import csv
import os
from pathlib import Path

BASE_DIR   = Path(__file__).resolve().parent
SINGLE_DIR = BASE_DIR / "data" / "gap_reports" / "single"
ORIG_DIR   = BASE_DIR / "data" / "gap_reports"
OUT_DIR    = BASE_DIR / "data" / "gap_reports"

COUNTRIES = [
    "India", "China", "USA", "EU", "UK",
    "Japan", "Australia", "Brazil", "Indonesia",
    "South Korea", "Canada", "Mexico", "South Africa",
    "Saudi Arabia", "Argentina", "Turkey", "Russia",
]

ALL_DIMS = [
    "renewable_energy",
    "net_zero_pathway",
    "methane_reduction",
    "adaptation_finance",
    "industrial_decarbonisation",
    "transport_targets",
    "lulucf_forestry",
]

# ── Known correct values from terminal output ─────────────────────────────
# These override ABSENT entries where pass_c failed
# Source: terminal output captured during the run
TERMINAL_CORRECTIONS = {
    # (country, dimension): (label, score)

    # net_zero_pathway — from terminal output
    ("India",        "net_zero_pathway"): ("SIGNIFICANT", 0.70),
    ("China",        "net_zero_pathway"): ("SIGNIFICANT", 0.60),
    ("EU",           "net_zero_pathway"): ("SIGNIFICANT", 0.75),
    ("UK",           "net_zero_pathway"): ("SIGNIFICANT", 0.70),
    ("Japan",        "net_zero_pathway"): ("SIGNIFICANT", 0.60),
    ("South Korea",  "net_zero_pathway"): ("SIGNIFICANT", 0.70),
    ("Saudi Arabia", "net_zero_pathway"): ("CRITICAL",    0.80),
    ("Argentina",    "net_zero_pathway"): ("SIGNIFICANT", 0.70),
    ("Mexico",       "net_zero_pathway"): ("CRITICAL",    0.80),
    ("South Africa", "net_zero_pathway"): ("SIGNIFICANT", 0.60),
    ("Turkey",       "net_zero_pathway"): ("SIGNIFICANT", 0.70),
    ("Russia",       "net_zero_pathway"): ("SIGNIFICANT", 0.60),

    # methane_reduction — from terminal output
    ("Indonesia",    "methane_reduction"): ("SIGNIFICANT", 0.70),
    ("Mexico",       "methane_reduction"): ("CRITICAL",    0.80),
    ("UK",           "methane_reduction"): ("SIGNIFICANT", 0.60),
    ("Turkey",       "methane_reduction"): ("SIGNIFICANT", 0.70),
    ("Japan",        "methane_reduction"): ("CRITICAL",    0.80),
    ("South Africa", "methane_reduction"): ("CRITICAL",    0.80),
    ("Saudi Arabia", "methane_reduction"): ("SIGNIFICANT", 0.70),
    ("Argentina",    "methane_reduction"): ("CRITICAL",    0.80),
    ("Russia",       "methane_reduction"): ("CRITICAL",    0.80),

    # renewable_energy — from original files (all 17 countries)
    ("India",        "renewable_energy"): ("SIGNIFICANT", 0.70),
    ("China",        "renewable_energy"): ("SIGNIFICANT", 0.60),
    ("USA",          "renewable_energy"): ("SIGNIFICANT", 0.60),
    ("EU",           "renewable_energy"): ("SIGNIFICANT", 0.60),
    ("UK",           "renewable_energy"): ("SIGNIFICANT", 0.60),
    ("Japan",        "renewable_energy"): ("SIGNIFICANT", 0.60),
    ("Australia",    "renewable_energy"): ("SIGNIFICANT", 0.70),
    ("Brazil",       "renewable_energy"): ("SIGNIFICANT", 0.70),
    ("Indonesia",    "renewable_energy"): ("SIGNIFICANT", 0.70),
    ("South Korea",  "renewable_energy"): ("SIGNIFICANT", 0.70),
    ("Canada",       "renewable_energy"): ("SIGNIFICANT", 0.70),
    ("Mexico",       "renewable_energy"): ("SIGNIFICANT", 0.60),
    ("South Africa", "renewable_energy"): ("SIGNIFICANT", 0.60),
    ("Saudi Arabia", "renewable_energy"): ("SIGNIFICANT", 0.60),
    ("Argentina",    "renewable_energy"): ("SIGNIFICANT", 0.70),
    ("Turkey",       "renewable_energy"): ("SIGNIFICANT", 0.60),
    ("Russia",       "renewable_energy"): ("CRITICAL",    1.00),

    # lulucf_forestry — from original files (all 17 countries)
    ("India",        "lulucf_forestry"): ("SIGNIFICANT", 0.70),
    ("China",        "lulucf_forestry"): ("SIGNIFICANT", 0.70),
    ("USA",          "lulucf_forestry"): ("SIGNIFICANT", 0.70),
    ("EU",           "lulucf_forestry"): ("SIGNIFICANT", 0.70),
    ("UK",           "lulucf_forestry"): ("SIGNIFICANT", 0.60),
    ("Japan",        "lulucf_forestry"): ("SIGNIFICANT", 0.60),
    ("Australia",    "lulucf_forestry"): ("CRITICAL",    1.00),
    ("Brazil",       "lulucf_forestry"): ("SIGNIFICANT", 0.70),
    ("Indonesia",    "lulucf_forestry"): ("SIGNIFICANT", 0.70),
    ("South Korea",  "lulucf_forestry"): ("SIGNIFICANT", 0.60),
    ("Canada",       "lulucf_forestry"): ("SIGNIFICANT", 0.70),
    ("Mexico",       "lulucf_forestry"): ("CRITICAL",    1.00),
    ("South Africa", "lulucf_forestry"): ("CRITICAL",    1.00),
    ("Saudi Arabia", "lulucf_forestry"): ("CRITICAL",    1.00),
    ("Argentina",    "lulucf_forestry"): ("SIGNIFICANT", 0.70),
    ("Turkey",       "lulucf_forestry"): ("SIGNIFICANT", 0.60),
    ("Russia",       "lulucf_forestry"): ("CRITICAL",    1.00),
}

# ── Measurability-based fallback heuristic ────────────────────────────────
def infer_label_from_measurability(meas_level, n_commits):
    """
    When pass_c failed and no terminal correction available,
    infer severity from measurability and commitment count.
    """
    if meas_level == "ABSENT" or n_commits == 0:
        return "CRITICAL", 0.85
    elif meas_level in ("VAGUE",):
        return "SIGNIFICANT", 0.65
    elif meas_level in ("QUALIFIED",):
        return "SIGNIFICANT", 0.60
    elif meas_level == "QUANTIFIED":
        return "SIGNIFICANT", 0.55
    return "SIGNIFICANT", 0.60


# ── Load a result from JSON file ──────────────────────────────────────────
def load_single(country, dim):
    safe = country.replace(" ", "_")
    path = SINGLE_DIR / f"{dim}__{safe}.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


def load_original(country, dim):
    """Try to load from original gap_reports folder."""
    for orig in ORIG_DIR.glob(f"gap_{dim}_*.json"):
        try:
            with open(orig) as f:
                od = json.load(f)
            ca = od.get("country_analyses", {})
            if country in ca:
                cr = ca[country]
                if hasattr(cr, '__dict__'):
                    cr = vars(cr)
                gf = cr.get("gap_findings", [])
                scores = [g.get("severity_score", g.get("score", 0))
                          for g in gf if isinstance(g, dict)
                          and (g.get("severity_score") or g.get("score"))]
                score  = max([float(s) for s in scores if s]) if scores else None
                sev    = cr.get("overall_severity",
                         gf[0].get("severity") if gf else None)
                if score:
                    return score, sev or score_to_label(score)
        except Exception:
            pass
    return None, None


def score_to_label(score):
    if score is None: return "MISSING"
    s = float(score)
    if s >= 0.80: return "CRITICAL"
    if s >= 0.50: return "SIGNIFICANT"
    if s >= 0.20: return "PARTIAL"
    return "ALIGNED"


# ── Build matrix ──────────────────────────────────────────────────────────
def build_matrix():
    matrix = {}

    print("Building G20 gap matrix...\n")

    for country in COUNTRIES:
        matrix[country] = {}

        for dim in ALL_DIMS:
            label = "MISSING"
            score = None
            source = "?"

            # Priority 1: terminal corrections (most reliable)
            if (country, dim) in TERMINAL_CORRECTIONS:
                label, score = TERMINAL_CORRECTIONS[(country, dim)]
                source = "terminal"

            # Priority 2: single JSON file
            elif (data := load_single(country, dim)):
                file_label = data.get("label", "ABSENT")
                file_score = data.get("score")
                file_pass_c_ok = bool(
                    data.get("details", {})
                    .get("raw_llm_outputs", {})
                    .get("pass_c")
                )
                if file_label != "ABSENT" or file_pass_c_ok:
                    label  = file_label
                    score  = file_score
                    source = "json_file"
                else:
                    # pass_c failed — apply heuristic
                    details   = data.get("details", {})
                    meas      = details.get("measurability", {})
                    meas_lvl  = meas.get("level", "ABSENT") if isinstance(meas, dict) else "ABSENT"
                    n_commits = len(details.get("commitments", []))
                    label, score = infer_label_from_measurability(
                        meas_lvl, n_commits)
                    source = "heuristic"

            # Priority 3: original gap_reports files
            else:
                orig_score, orig_sev = load_original(country, dim)
                if orig_score:
                    score  = orig_score
                    label  = score_to_label(orig_score)
                    source = "orig_file"

            matrix[country][dim] = {"label": label, "score": score,
                                     "source": source}

            icon = {"CRITICAL": "🔴", "SIGNIFICANT": "🟠",
                    "ALIGNED": "🟢", "PARTIAL": "🟡"}.get(label, "⚪")
            print(f"  {country:<15} {dim:<28} {icon} {label:<15} "
                  f"score={str(score):<6} [{source}]")

    return matrix


# ── Save CSV files ────────────────────────────────────────────────────────
def save_csvs(matrix):
    # Label matrix
    label_csv = OUT_DIR / "g20_summary.csv"
    with open(label_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Country"] + ALL_DIMS)
        for country in COUNTRIES:
            row = [country]
            for dim in ALL_DIMS:
                row.append(matrix[country][dim]["label"])
            w.writerow(row)
    print(f"\n[SAVED] Label matrix → {label_csv}")

    # Score matrix
    score_csv = OUT_DIR / "g20_score_matrix.csv"
    with open(score_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Country"] + ALL_DIMS)
        for country in COUNTRIES:
            row = [country]
            for dim in ALL_DIMS:
                s = matrix[country][dim]["score"]
                row.append(f"{s:.2f}" if s is not None else "")
            w.writerow(row)
    print(f"[SAVED] Score matrix → {score_csv}")

    # Print final summary table
    print("\n" + "="*90)
    print("  FINAL G20 GAP MATRIX SUMMARY")
    print("="*90)
    short = {d: d[:8] for d in ALL_DIMS}
    header = f"{'Country':<15}" + "".join(f"{short[d]:<13}" for d in ALL_DIMS)
    print(header)
    print("-"*90)
    for country in COUNTRIES:
        row = f"{country:<15}"
        for dim in ALL_DIMS:
            lbl = matrix[country][dim]["label"]
            icon = {"CRITICAL":"CRI 🔴","SIGNIFICANT":"SIG 🟠",
                    "ALIGNED":"ALN 🟢","PARTIAL":"PAR 🟡",
                    "MISSING":"??? ⚪"}.get(lbl, lbl[:6])
            row += f"{icon:<13}"
        print(row)

    return label_csv, score_csv


if __name__ == "__main__":
    matrix = build_matrix()
    save_csvs(matrix)
    print("\nDone. Share g20_summary.csv and g20_score_matrix.csv.")
