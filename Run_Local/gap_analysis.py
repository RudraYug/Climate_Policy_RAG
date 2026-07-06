# ── run_gap_analysis.py ───────────────────────────────────────────────────────
# Run from VS Code terminal:
#   cd D:\Thesis
#   thesis_env\Scripts\activate
#   python Run_Local\run_gap_analysis.py

import os, sys
from pathlib import Path
from dotenv import load_dotenv

# ── 1. Load API key from .env file ────────────────────────────────────────────
load_dotenv()   # reads D:\Thesis\.env automatically
key = os.environ.get("OPENAI_API_KEY")
if not key:
    raise ValueError("OPENAI_API_KEY not found in .env file")
print(f"API key loaded: {key[:8]}...")

# ── 2. Set working directory to D:\Thesis\ ───────────────────────────────────
THESIS_DIR = Path(__file__).resolve().parent.parent  # Run_Local → D:\Thesis
os.chdir(THESIS_DIR)
sys.path.insert(0, str(THESIS_DIR / "LLM_Integration_Code"))
sys.path.insert(0, str(THESIS_DIR / "Gap_Analysis_Code"))
print(f"Working dir    : {os.getcwd()}")

# ── 3. Import GapAnalyser ─────────────────────────────────────────────────────
from gap_analysis import GapAnalyser

analyser = GapAnalyser(
    backend      = "gpt4o",                                       # same OpenAI API
    top_k_ndc    = 5,
    top_k_ipcc   = 4,
    embed_device = "cpu",
    gen_dir      = str(THESIS_DIR / "LLM_Integration_Code"),      # explicit path
)

# ── 4. Choose what to run ─────────────────────────────────────────────────────

# ── Option A: Two-country comparison ─────────────────────────────────────────
print("\n" + "="*60)
print("RUNNING: India vs China — Renewable Energy")
print("="*60)
report = analyser.analyse(
    countries  = ["India", "China"],
    dimension  = "renewable_energy",
    ndc_cycles = [2],
)
print(report.summary_text)
report.save()   # saves to data/gap_reports/

# ── Option B: Three-country with adaptation finance ───────────────────────────
# Uncomment to run:
# report2 = analyser.analyse(
#     countries  = ["Indonesia", "Brazil", "South Africa"],
#     dimension  = "adaptation_finance",
#     ndc_cycles = [1, 2],
# )
# print(report2.summary_text)
# report2.save()

# ── Option C: Net zero pathway across NDC2 countries ─────────────────────────
# report3 = analyser.analyse(
#     countries  = ["UK", "EU", "Australia", "South Korea"],
#     dimension  = "net_zero_pathway",
#     ndc_cycles = [2],
# )
# print(report3.summary_text)
# report3.save()

# ── Option D: Full G20 sweep (runs ~1-2 hours, costs ~€0.50 with GPT-4o) ────
# report4 = analyser.analyse_g20(dimension="renewable_energy")
# print(report4.summary_text)