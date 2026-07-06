# ── run_local.py  ─────────────────────────────────────────────────────────────
# Run from D:\Thesis\ with:  python run_local.py
# Or open in VS Code and press F5

import os, sys
from pathlib import Path
from dotenv import load_dotenv

# ── 1. Load API key from .env file ────────────────────────────────────────────
load_dotenv()   # reads D:\Thesis\.env automatically
key = os.environ.get("OPENAI_API_KEY")
if not key:
    raise ValueError("OPENAI_API_KEY not found in .env file")
print(f"API key loaded: {key[:8]}...")

# ── 2. Set working directory to D:\Thesis\ ────────────────────────────────────
# This makes all relative paths like data/processed/chunks.jsonl work correctly
THESIS_DIR = Path(__file__).resolve().parent.parent   # D:\Thesis\
os.chdir(THESIS_DIR)
sys.path.insert(0, str(THESIS_DIR / "LLM_Integration_Code"))
print(f"Working dir : {os.getcwd()}")

# ── 3. Verify data files exist ────────────────────────────────────────────────
required = [
    "data/processed/chunks.jsonl",
    "data/processed/bm25_index.pkl",
    "data/vectorstore",
    "LLM_Integration_Code/generation.py",
    "LLM_Integration_Code/retriever.py",
]
all_ok = True
for f in required:
    exists = os.path.exists(f)
    print(f"  {'✓' if exists else '✗  MISSING'} {f}")
    if not exists:
        all_ok = False

if not all_ok:
    raise FileNotFoundError("Some required files are missing. Check paths above.")

# ── 4. Import and initialise pipeline ────────────────────────────────────────
from generation import RAGPipeline

pipe = RAGPipeline(
    backend      = "gpt4o",        # OpenAI API — no local GPU needed
    top_k        = 5,
    embed_device = "cpu",          # bge-large runs on CPU fine for retrieval
    openai_key   = os.environ["OPENAI_API_KEY"],
)

# ── 5. Run queries ────────────────────────────────────────────────────────────
resp = pipe.query(
    "What is India's 2030 renewable energy target?",
    filters = {"country": "India"}
)
resp.pretty_print()

resp2 = pipe.query(
    "What is the gap between Indonesia's NDC and IPCC 1.5 degree recommendations?",
    filters = None
)
resp2.pretty_print()