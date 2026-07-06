"""
===========================================================================
  G20 Gap Analysis — Remaining Two Dimensions
  =============================================
  Runs ONLY the two pending dimensions with CORRECT names:
    - loss_and_damage
    - transport_decarbonisation

  PLACE IN: D:\Thesis\
  RUN AFTER MIDNIGHT UTC when Groq rate limit resets.
===========================================================================
"""
import sys, os, json, time
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))
from gap_analysis import GapAnalyser

OUTPUT_DIR = THIS_DIR / "data" / "gap_reports" / "single"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

COUNTRIES = [
    "India","China","USA","EU","UK","Japan","Australia","Brazil",
    "Indonesia","South Korea","Canada","Mexico","South Africa",
    "Saudi Arabia","Argentina","Turkey","Russia",
]

# CORRECTED dimension names matching gap_analysis.py
DIMENSIONS = [
    "loss_and_damage",
    "transport_decarbonisation",
]

DELAY = 8

def to_dict(obj):
    if isinstance(obj, dict): return {k: to_dict(v) for k,v in obj.items()}
    if isinstance(obj, list): return [to_dict(i) for i in obj]
    if hasattr(obj,'__dict__'): return {k: to_dict(v) for k,v in vars(obj).items() if not k.startswith('_')}
    if hasattr(obj,'_asdict'): return to_dict(obj._asdict())
    try:
        import dataclasses
        if dataclasses.is_dataclass(obj): return to_dict(dataclasses.asdict(obj))
    except: pass
    return obj

def extract_score(cr):
    gf = cr.get("gap_findings",[])
    if gf:
        scores=[float(g.get("severity_score",g.get("score",0))) for g in gf if isinstance(g,dict) and (g.get("severity_score") or g.get("score"))]
        return max(scores) if scores else None, gf[0].get("severity")
    s = cr.get("overall_severity_score", cr.get("gap_score"))
    sv = cr.get("overall_severity", cr.get("severity","ABSENT"))
    return s, sv

def score_to_label(s):
    if s is None: return "ABSENT"
    s = float(s)
    if s >= 0.80: return "CRITICAL"
    if s >= 0.50: return "SIGNIFICANT"
    if s >= 0.20: return "PARTIAL"
    return "ALIGNED"

def main():
    print("Initialising GapAnalyser (groq)...")
    analyser = GapAnalyser(backend="groq")
    total = len(COUNTRIES) * len(DIMENSIONS)
    done, failed = 0, []

    for dim in DIMENSIONS:
        print(f"\n{'='*55}\n  DIMENSION: {dim}\n{'='*55}")
        for country in COUNTRIES:
            safe = country.replace(" ","_")
            out  = OUTPUT_DIR / f"{dim}__{safe}.json"
            if out.exists():
                print(f"  [SKIP] {country}"); done+=1; continue
            print(f"  [{done+1}/{total}] {country:<20} ... ", end="", flush=True)
            try:
                report = analyser.analyse(countries=[country], dimension=dim, ndc_cycles=[2], save=False)
                ca_raw = getattr(report,"country_analyses",None) or (report.get("country_analyses") if isinstance(report,dict) else {})
                ca = {k: to_dict(v) for k,v in ca_raw.items()} if isinstance(ca_raw,dict) else to_dict(ca_raw)
                cr = ca.get(country,{})
                score, severity = extract_score(cr)
                label = score_to_label(score)
                with open(out,"w",encoding="utf-8") as f:
                    json.dump({"country":country,"dimension":dim,"score":score,"severity":severity,"label":label,"details":cr},f,indent=2,default=str)
                print(f"{label} ({score})")
                done += 1
            except KeyboardInterrupt:
                print("\nStopped."); [print(f"  {c}/{d}: {e}") for c,d,e in failed]; sys.exit(0)
            except Exception as e:
                print(f"FAILED — {e}"); failed.append((country,dim,str(e))); done+=1
            time.sleep(DELAY)

    print(f"\nDone. {done}/{total}")
    if failed:
        for c,d,e in failed: print(f"  {c}/{d}: {e}")

if __name__=="__main__":
    main()
