# Climate Policy Analyser — Thesis Demo App
**LLM-Powered NDC Summariser & Gap Analyser**
M.Sc. Data Science · University of Europe for Applied Sciences, Potsdam · 2026

---

## What this app does

| Feature | Description |
|---------|-------------|
| 🔍 Policy Query | Natural language questions over G20 NDC + IPCC AR6 corpus, with cited answers |
| 📊 Gap Analysis | 3-pass LLM analysis comparing NDC commitments against IPCC benchmarks |
| 🗂 Filters | Country, doc type, NDC cycle — sidebar narrows retrieval |
| 📁 History | Session history + download all JSON results |

---

## Folder structure required

```
D:\Thesis\                          ← root (or wherever you deploy from)
├── app.py                          ← this file
├── requirements.txt
├── .streamlit\
│   ├── config.toml
│   └── secrets.toml                ← NOT committed to git
├── LLM_Integration_Code\
│   ├── generation.py
│   └── retriever.py
├── Gap_Analysis_Code\
│   └── gap_analysis.py
└── data\
    ├── processed\
    │   ├── chunks.jsonl             ← ~50 MB
    │   └── bm25_index.pkl           ← ~15 MB
    ├── vectorstore\                 ← ChromaDB folder (~25 MB)
    ├── gap_reports\                 ← created automatically
    └── responses\                   ← created automatically
```

---

## Run locally (VS Code)

```bash
# 1. Activate your virtual environment
cd D:\Thesis
thesis_env\Scripts\activate

# 2. Install dependencies (first time only)
pip install -r requirements.txt

# 3. Add your API key to .streamlit\secrets.toml
#    (copy from secrets.toml template and add your real key)

# 4. Run the app
streamlit run app.py

# App opens at http://localhost:8501
```

---

## Deploy to Streamlit Cloud (free)

### Step 1 — Push to GitHub
```bash
# Create a .gitignore first to protect your API key
echo ".streamlit/secrets.toml" >> .gitignore
echo "data/raw/" >> .gitignore
echo "thesis_env/" >> .gitignore

git init
git add .
git commit -m "Initial thesis app"
git remote add origin https://github.com/YOUR_USERNAME/thesis-app.git
git push -u origin main
```

> ⚠ **Important:** The `data/` folder contains large files.
> If `chunks.jsonl` + `bm25_index.pkl` + `vectorstore/` exceed GitHub's 100 MB
> file limit, use [Git LFS](https://git-lfs.github.com/) or upload data files
> separately and load them via URL. See the **Large Files** note below.

### Step 2 — Create Streamlit Cloud app
1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with GitHub
3. Click **New app** → select your repo → set **Main file path** to `app.py`
4. Click **Deploy**

### Step 3 — Add your API key as a secret
1. Go to your deployed app → **Settings** → **Secrets**
2. Paste:
   ```toml
   OPENAI_API_KEY = "sk-proj-your-real-key-here"
   ```
3. Save → app restarts automatically

---

## Large Files note for Streamlit Cloud

Streamlit Cloud has a **1 GB storage limit** per app. Your data files are:
- `chunks.jsonl` ≈ 50 MB ✓
- `bm25_index.pkl` ≈ 15 MB ✓
- `vectorstore/` ≈ 25 MB ✓
- **Total ≈ 90 MB** — fits comfortably ✓

GitHub has a **100 MB per-file limit**. Your individual files are all under this
so standard `git push` works. If you later add raw PDFs, use Git LFS.

---

## Environment variables

| Variable | Where to set | Value |
|----------|-------------|-------|
| `OPENAI_API_KEY` | `.streamlit/secrets.toml` (local) or Streamlit Cloud Secrets | Your OpenAI key |

---

## Thesis citation

If using this app in your thesis, cite it as:

> [Your Name] (2026). *LLM-Powered Climate Policy Summariser and Gap Analyser*
> [Software]. University of Europe for Applied Sciences, Potsdam.
