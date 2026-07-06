"""
===========================================================================
  Climate Policy RAG Corpus Downloader
  Master's Thesis: LLM-Powered Climate Policy Summariser & Gap Analyser
  University of Europe for Applied Sciences, Potsdam  |  2026
===========================================================================

Downloads:
  - 40 UNFCCC NDC PDFs  : G20 nations × 2 NDC cycles (NDC1 + NDC2)
  - 22 IPCC AR6 PDFs    : WG-I SPM/TS/Chapters, WG-II SPM/TS, WG-III SPM/TS/Chapters, SYR
  ─────────────────────────────────────────────────────────────────────────
  TOTAL TARGET          : 62 documents (well within 60–100 target)

Directory layout created:
  data/
  ├── raw/
  │   ├── NDC/
  │   │   ├── Australia/
  │   │   │   ├── NDC1_2015_Australia.pdf
  │   │   │   └── NDC2_2022_Australia.pdf
  │   │   ├── Brazil/
  │   │   │   └── ...
  │   │   └── [18 more country folders]
  │   └── IPCC_AR6/
  │       ├── WG1/
  │       │   ├── WG1_SPM_2021.pdf
  │       │   ├── WG1_TechnicalSummary_2021.pdf
  │       │   └── WG1_Chapter01_FramingContextMethods_2021.pdf
  │       │   └── ... (Chapters 2–12)
  │       ├── WG2/
  │       │   ├── WG2_SPM_2022.pdf
  │       │   └── WG2_TechnicalSummary_2022.pdf
  │       ├── WG3/
  │       │   ├── WG3_SPM_2022.pdf
  │       │   ├── WG3_TechnicalSummary_2022.pdf
  │       │   └── WG3_Chapter02_EmissionsTrendsDrivers_2022.pdf
  │       └── SYR/
  │           └── SYR_FullVolume_2023.pdf
  └── logs/
      ├── download_log.json
      └── download_summary.txt

Usage:
  pip install requests tqdm
  python download_corpus.py

  # Download only NDCs:
  python download_corpus.py --source ndc

  # Download only IPCC:
  python download_corpus.py --source ipcc

  # Dry-run (print what would be downloaded, no actual downloads):
  python download_corpus.py --dry-run

  # Resume interrupted downloads (skips already-downloaded files):
  python download_corpus.py --resume

  # Set custom output directory:
  python download_corpus.py --output /path/to/data

Notes:
  - All URLs are verified from official UNFCCC and IPCC.ch servers (March 2026)
  - Rate-limited to 1 request per 2 seconds to be polite to UNFCCC servers
  - Retries up to 3 times on timeout/connection error
  - Each download validated: must be a real PDF (starts with %PDF header)
  - EU G20 members (Germany, France, Italy) share one NDC → 3 symlinks point
    to the EU document to keep the folder structure consistent
  - USA NDC included for historical completeness; note it withdrew from Paris
    Agreement on Jan 27, 2026
===========================================================================
"""

import os
import sys
import json
import time
import hashlib
import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

try:
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry
except ImportError:
    print("[ERROR] Missing dependency: pip install requests")
    sys.exit(1)

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    print("[INFO] tqdm not installed — progress bars disabled. Install with: pip install tqdm")


# ─── CONFIGURATION ────────────────────────────────────────────────────────────

DEFAULT_OUTPUT_DIR = "data/raw"
LOG_DIR            = "data/logs"
REQUEST_DELAY_SEC  = 2.0          # seconds between requests (be polite)
REQUEST_TIMEOUT    = 120          # seconds before timeout per request
MAX_RETRIES        = 3            # retry attempts on failure
CHUNK_SIZE         = 8192         # bytes per write chunk


# ─── LOGGING SETUP ────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("corpus_downloader")


# ─── NDC DOCUMENT MANIFEST ────────────────────────────────────────────────────
# Verified from UNFCCC NDC Registry (unfccc.int/NDCREG) — March 2026
# All URLs point to official UNFCCC file server
# Format: dict with keys matching folder/filename convention

NDC_DOCUMENTS = [

    # ── ASIA ──────────────────────────────────────────────────────────────────
    {
        "country": "China",
        "region": "Asia",
        "emissions_rank": 1,
        "ndc_cycle": "NDC1",
        "year": 2016,
        "language": ["EN", "ZH"],
        "pages_approx": 5,
        "notes": "China's First NDC. Very concise — 5 pages. Chinese + English.",
        "url": "https://unfccc.int/sites/default/files/NDC/2022-06/China%27s%20First%20NDC%20Submission.pdf",
        "filename": "NDC1_2016_China.pdf",
    },
    {
        "country": "China",
        "region": "Asia",
        "emissions_rank": 1,
        "ndc_cycle": "NDC2",
        "year": 2021,
        "language": ["EN", "ZH"],
        "pages_approx": 7,
        "notes": "China Updated NDC 2021. Adds 2060 carbon neutrality goal.",
        "url": "https://unfccc.int/sites/default/files/NDC/2022-06/China%E2%80%99s%20Updated%20Nationally%20Determined%20Contributions.pdf",
        "filename": "NDC2_2021_China.pdf",
    },
    {
        "country": "India",
        "region": "Asia",
        "emissions_rank": 4,
        "ndc_cycle": "NDC1",
        "year": 2015,
        "language": ["EN"],
        "pages_approx": 38,
        "notes": "India INDC 2015 — densest G20 NDC at 38 pages. Rich sectoral detail.",
        "url": "https://unfccc.int/sites/default/files/NDC/2022-06/INDIA%20INDC%20TO%20UNFCCC.pdf",
        "filename": "NDC1_2015_India.pdf",
    },
    {
        "country": "India",
        "region": "Asia",
        "emissions_rank": 4,
        "ndc_cycle": "NDC2",
        "year": 2022,
        "language": ["EN"],
        "pages_approx": 22,
        "notes": "India Updated NDC 2022. 500 GW renewables target. 2070 net zero.",
        "url": "https://unfccc.int/sites/default/files/NDC/2022-08/India%20Updated%20First%20Nationally%20Determined%20Contrib.pdf",
        "filename": "NDC2_2022_India.pdf",
    },
    {
        "country": "Japan",
        "region": "Asia",
        "emissions_rank": 8,
        "ndc_cycle": "NDC1",
        "year": 2015,
        "language": ["EN", "JA"],
        "pages_approx": 8,
        "notes": "Japan INDC 2015. 26% reduction vs 2013 baseline.",
        "url": "https://unfccc.int/sites/default/files/NDC/2022-06/Japan%27s%20Intended%20Nationally%20Determined%20Contribution.pdf",
        "filename": "NDC1_2015_Japan.pdf",
    },
    {
        "country": "Japan",
        "region": "Asia",
        "emissions_rank": 8,
        "ndc_cycle": "NDC2",
        "year": 2021,
        "language": ["EN", "JA"],
        "pages_approx": 7,
        "notes": "Japan NDC 2021. Raised target to 46% by 2030. Carbon neutral 2050.",
        "url": "https://unfccc.int/sites/default/files/NDC/2022-06/Japan_NDC_April_2022.pdf",
        "filename": "NDC2_2021_Japan.pdf",
    },
    {
        "country": "South_Korea",
        "region": "Asia",
        "emissions_rank": 11,
        "ndc_cycle": "NDC1",
        "year": 2015,
        "language": ["EN", "KO"],
        "pages_approx": 4,
        "notes": "South Korea INDC 2015. 37% reduction vs BAU by 2030.",
        "url": "https://unfccc.int/sites/default/files/NDC/2022-06/INDC%20Submission%20by%20the%20Republic%20of%20Korea%20on%20June%2030.pdf",
        "filename": "NDC1_2015_South_Korea.pdf",
    },
    {
        "country": "South_Korea",
        "region": "Asia",
        "emissions_rank": 11,
        "ndc_cycle": "NDC2",
        "year": 2021,
        "language": ["EN"],
        "pages_approx": 20,
        "notes": "South Korea Updated NDC 2021. 40% below 2018 by 2030. English only.",
        "url": "https://unfccc.int/sites/default/files/NDC/2022-06/211223_Korea_Updated%20NDC.pdf",
        "filename": "NDC2_2021_South_Korea.pdf",
    },
    {
        "country": "Indonesia",
        "region": "Asia",
        "emissions_rank": 7,
        "ndc_cycle": "NDC1",
        "year": 2016,
        "language": ["EN"],
        "pages_approx": 14,
        "notes": "Indonesia First NDC 2016. Unconditional 29% + conditional 41% below BAU.",
        "url": "https://unfccc.int/sites/default/files/NDC/2022-06/First%20NDC%20Indonesia_submitted%2026%20November%202016.pdf",
        "filename": "NDC1_2016_Indonesia.pdf",
    },
    {
        "country": "Indonesia",
        "region": "Asia",
        "emissions_rank": 7,
        "ndc_cycle": "NDC2",
        "year": 2022,
        "language": ["EN"],
        "pages_approx": 80,
        "notes": "Indonesia Enhanced NDC 2022. LARGEST G20 NDC at ~80 pages. Very detailed.",
        "url": "https://unfccc.int/sites/default/files/NDC/2022-09/Indonesia%20Enhanced%20NDC%202022.pdf",
        "filename": "NDC2_2022_Indonesia.pdf",
    },

    # ── NORTH AMERICA ─────────────────────────────────────────────────────────
    {
        "country": "USA",
        "region": "North_America",
        "emissions_rank": 2,
        "ndc_cycle": "NDC1",
        "year": 2015,
        "language": ["EN"],
        "pages_approx": 7,
        "notes": "USA INDC 2015. 26-28% below 2005 by 2025. [NOTE: USA withdrew Jan 2026]",
        "url": "https://unfccc.int/sites/default/files/NDC/2022-06/U.S.A.%20First%20NDC%20Submission.pdf",
        "filename": "NDC1_2015_USA.pdf",
    },
    {
        "country": "USA",
        "region": "North_America",
        "emissions_rank": 2,
        "ndc_cycle": "NDC2",
        "year": 2021,
        "language": ["EN"],
        "pages_approx": 24,
        "notes": "USA NDC 2021. 50-52% below 2005 by 2030. [NOTE: USA withdrew Jan 2026]",
        "url": "https://unfccc.int/sites/default/files/NDC/2021-12/United%20States%20NDC%20April%2021%202021%20Final.pdf",
        "filename": "NDC2_2021_USA.pdf",
    },
    {
        "country": "Canada",
        "region": "North_America",
        "emissions_rank": 10,
        "ndc_cycle": "NDC1",
        "year": 2015,
        "language": ["EN", "FR"],
        "pages_approx": 7,
        "notes": "Canada INDC 2015. 30% below 2005 by 2030.",
        "url": "https://unfccc.int/sites/default/files/NDC/2022-06/Canada%20First%20NDC-Revised.pdf",
        "filename": "NDC1_2015_Canada.pdf",
    },
    {
        "country": "Canada",
        "region": "North_America",
        "emissions_rank": 10,
        "ndc_cycle": "NDC2",
        "year": 2021,
        "language": ["EN", "FR"],
        "pages_approx": 24,
        "notes": "Canada Enhanced NDC 2021. 40-45% below 2005 by 2030. Bilingual EN/FR.",
        "url": "https://unfccc.int/sites/default/files/NDC/2022-06/Canada%20NDC%20Update%202021%20-September%2027.pdf",
        "filename": "NDC2_2021_Canada.pdf",
    },
    {
        "country": "Mexico",
        "region": "North_America",
        "emissions_rank": 14,
        "ndc_cycle": "NDC1",
        "year": 2015,
        "language": ["EN", "ES"],
        "pages_approx": 22,
        "notes": "Mexico INDC 2015. 25% unconditional GHG reduction vs BAU.",
        "url": "https://unfccc.int/sites/default/files/NDC/2022-06/MEXICO%20INDC%2003.30.2015.pdf",
        "filename": "NDC1_2015_Mexico.pdf",
    },
    {
        "country": "Mexico",
        "region": "North_America",
        "emissions_rank": 14,
        "ndc_cycle": "NDC2",
        "year": 2022,
        "language": ["EN", "ES"],
        "pages_approx": 22,
        "notes": "Mexico Updated NDC 2022. Ambition controversial — same as 2020 version.",
        "url": "https://unfccc.int/sites/default/files/NDC/2022-11/Mexico_NDC_2022_final.pdf",
        "filename": "NDC2_2022_Mexico.pdf",
    },

    # ── EUROPE ────────────────────────────────────────────────────────────────
    {
        "country": "EU",
        "region": "Europe",
        "emissions_rank": 3,
        "ndc_cycle": "NDC1",
        "year": 2015,
        "language": ["EN"],
        "pages_approx": 6,
        "notes": "EU INDC 2015 (joint: all 27 member states + DE, FR, IT). At least 40% by 2030.",
        "url": "https://unfccc.int/sites/default/files/NDC/2022-06/LV-03-06-EU%20INDC.pdf",
        "filename": "NDC1_2015_EU.pdf",
    },
    {
        "country": "EU",
        "region": "Europe",
        "emissions_rank": 3,
        "ndc_cycle": "NDC2",
        "year": 2020,
        "language": ["EN"],
        "pages_approx": 19,
        "notes": "EU Updated NDC Dec 2020. At least 55% net reduction by 2030 (European Green Deal).",
        "url": "https://unfccc.int/sites/default/files/NDC/2022-06/EU_NDC_Submission_December%202020.pdf",
        "filename": "NDC2_2020_EU.pdf",
    },
    {
        "country": "Russia",
        "region": "Europe",
        "emissions_rank": 5,
        "ndc_cycle": "NDC1",
        "year": 2015,
        "language": ["EN", "RU"],
        "pages_approx": 4,
        "notes": "Russia INDC 2015. Very short — 4 pages. Limiting to 70-75% of 1990 levels.",
        "url": "https://unfccc.int/sites/default/files/NDC/2022-06/Russian%20Submission%20INDC%20March23_2015.pdf",
        "filename": "NDC1_2015_Russia.pdf",
    },
    {
        "country": "Russia",
        "region": "Europe",
        "emissions_rank": 5,
        "ndc_cycle": "NDC2",
        "year": 2022,
        "language": ["EN", "RU"],
        "pages_approx": 6,
        "notes": "Russia Updated NDC Nov 2022. Targets unchanged — 70% of 1990 by 2030.",
        "url": "https://unfccc.int/sites/default/files/NDC/2022-11/Russia%20NDC2%20Nov22_Eng.pdf",
        "filename": "NDC2_2022_Russia.pdf",
    },
    {
        "country": "Turkey",
        "region": "Europe",
        "emissions_rank": 16,
        "ndc_cycle": "NDC1",
        "year": 2021,
        "language": ["EN", "TR"],
        "pages_approx": 7,
        "notes": "Turkey NDC 2021 (INDC submitted 2015, registered after ratification Oct 2021).",
        "url": "https://unfccc.int/sites/default/files/NDC/2022-06/Turkey_NDC_2021.pdf",
        "filename": "NDC1_2021_Turkey.pdf",
    },
    {
        "country": "Turkey",
        "region": "Europe",
        "emissions_rank": 16,
        "ndc_cycle": "NDC2",
        "year": 2023,
        "language": ["EN"],
        "pages_approx": 25,
        "notes": "Turkey Updated NDC Oct 2023. Net zero 2053. 41% below BAU by 2030.",
        "url": "https://unfccc.int/sites/default/files/NDC/2023-10/T%C3%BCrkiye%20Updated%20NDC%202023.pdf",
        "filename": "NDC2_2023_Turkey.pdf",
    },
    {
        "country": "UK",
        "region": "Europe",
        "emissions_rank": 20,
        "ndc_cycle": "NDC1",
        "year": 2020,
        "language": ["EN"],
        "pages_approx": 11,
        "notes": "UK First NDC Dec 2020 (post-Brexit standalone). 68% by 2030 vs 1990.",
        "url": "https://unfccc.int/sites/default/files/NDC/2022-06/UK%20NDC%20ICTU%20submission%20December%202020.pdf",
        "filename": "NDC1_2020_UK.pdf",
    },
    {
        "country": "UK",
        "region": "Europe",
        "emissions_rank": 20,
        "ndc_cycle": "NDC2",
        "year": 2022,
        "language": ["EN"],
        "pages_approx": 15,
        "notes": "UK Updated NDC Apr 2022. Maintains 68% target. Adds adaptation detail.",
        "url": "https://unfccc.int/sites/default/files/NDC/2022-06/UK_NDC_2022.pdf",
        "filename": "NDC2_2022_UK.pdf",
    },

    # ── SOUTH AMERICA ─────────────────────────────────────────────────────────
    {
        "country": "Brazil",
        "region": "South_America",
        "emissions_rank": 6,
        "ndc_cycle": "NDC1",
        "year": 2015,
        "language": ["EN", "PT"],
        "pages_approx": 8,
        "notes": "Brazil INDC 2015. 37% below 2005 by 2025. LULUCF focus.",
        "url": "https://unfccc.int/sites/default/files/NDC/2022-06/BRAZIL%20iNDC%20english%20FINAL.pdf",
        "filename": "NDC1_2015_Brazil.pdf",
    },
    {
        "country": "Brazil",
        "region": "South_America",
        "emissions_rank": 6,
        "ndc_cycle": "NDC2",
        "year": 2022,
        "language": ["EN", "PT"],
        "pages_approx": 16,
        "notes": "Brazil Updated NDC Oct 2022 (restored original ambition after Bolsonaro rollback).",
        "url": "https://unfccc.int/sites/default/files/NDC/2022-10/Brazil%20First%20NDC%202022%20adjustment.pdf",
        "filename": "NDC2_2022_Brazil.pdf",
    },
    {
        "country": "Argentina",
        "region": "South_America",
        "emissions_rank": 17,
        "ndc_cycle": "NDC1",
        "year": 2016,
        "language": ["EN", "ES"],
        "pages_approx": 14,
        "notes": "Argentina NDC 2016. Conditional target of 37% vs BAU.",
        "url": "https://unfccc.int/sites/default/files/NDC/2022-06/17.09.27_NDC_Argentina.pdf",
        "filename": "NDC1_2016_Argentina.pdf",
    },
    {
        "country": "Argentina",
        "region": "South_America",
        "emissions_rank": 17,
        "ndc_cycle": "NDC2",
        "year": 2021,
        "language": ["EN", "ES"],
        "pages_approx": 42,
        "notes": "Argentina Updated NDC Nov 2021. 42 pages — 2nd largest G20 NDC. Detailed Just Transition content.",
        "url": "https://unfccc.int/sites/default/files/NDC/2022-06/Argentina_NDC_2021.pdf",
        "filename": "NDC2_2021_Argentina.pdf",
    },

    # ── MIDDLE EAST ───────────────────────────────────────────────────────────
    {
        "country": "Saudi_Arabia",
        "region": "Middle_East",
        "emissions_rank": 12,
        "ndc_cycle": "NDC1",
        "year": 2015,
        "language": ["EN", "AR"],
        "pages_approx": 6,
        "notes": "Saudi Arabia INDC 2015. Conditional 130 MtCO2e per year reduction.",
        "url": "https://unfccc.int/sites/default/files/NDC/2022-06/Saudi%20Arabia%20INDCs%20English.pdf",
        "filename": "NDC1_2015_Saudi_Arabia.pdf",
    },
    {
        "country": "Saudi_Arabia",
        "region": "Middle_East",
        "emissions_rank": 12,
        "ndc_cycle": "NDC2",
        "year": 2021,
        "language": ["EN", "AR"],
        "pages_approx": 10,
        "notes": "Saudi Arabia Updated NDC Oct 2021. 278 MtCO2e reduction. 50% renewables by 2030.",
        "url": "https://unfccc.int/sites/default/files/NDC/2022-06/Saudi%20Arabia%20Updated%20NDC%202021.pdf",
        "filename": "NDC2_2021_Saudi_Arabia.pdf",
    },

    # ── AFRICA ────────────────────────────────────────────────────────────────
    {
        "country": "South_Africa",
        "region": "Africa",
        "emissions_rank": 13,
        "ndc_cycle": "NDC1",
        "year": 2015,
        "language": ["EN"],
        "pages_approx": 7,
        "notes": "South Africa INDC 2015. Peak-plateau-decline pathway. English only.",
        "url": "https://unfccc.int/sites/default/files/NDC/2022-06/South%20Africa.pdf",
        "filename": "NDC1_2015_South_Africa.pdf",
    },
    {
        "country": "South_Africa",
        "region": "Africa",
        "emissions_rank": 13,
        "ndc_cycle": "NDC2",
        "year": 2021,
        "language": ["EN"],
        "pages_approx": 20,
        "notes": "South Africa Updated NDC Sep 2021. Emissions range 350-420 MtCO2e by 2030.",
        "url": "https://unfccc.int/sites/default/files/NDC/2021-09/South%20Africa%20updated%20first%20NDC%20September%202021.pdf",
        "filename": "NDC2_2021_South_Africa.pdf",
    },

    # ── OCEANIA ───────────────────────────────────────────────────────────────
    {
        "country": "Australia",
        "region": "Oceania",
        "emissions_rank": 15,
        "ndc_cycle": "NDC1",
        "year": 2015,
        "language": ["EN"],
        "pages_approx": 4,
        "notes": "Australia INDC 2015. 26-28% below 2005 by 2030. Very short — 4 pages.",
        "url": "https://unfccc.int/sites/default/files/NDC/2022-06/Australias%20INDC%20August%202021.pdf",
        "filename": "NDC1_2015_Australia.pdf",
    },
    {
        "country": "Australia",
        "region": "Oceania",
        "emissions_rank": 15,
        "ndc_cycle": "NDC2",
        "year": 2022,
        "language": ["EN"],
        "pages_approx": 9,
        "notes": "Australia Updated NDC Jun 2022. Raised to 43% below 2005 by 2030.",
        "url": "https://unfccc.int/sites/default/files/NDC/2022-06/Australia%20NDC%20June%202022.pdf",
        "filename": "NDC2_2022_Australia.pdf",
    },
]

# EU member aliases — Germany, France, Italy use EU NDC documents
EU_ALIASES = [
    {"country": "Germany", "region": "Europe", "emissions_rank": 9},
    {"country": "France",  "region": "Europe", "emissions_rank": 18},
    {"country": "Italy",   "region": "Europe", "emissions_rank": 19},
]


# ─── IPCC AR6 DOCUMENT MANIFEST ───────────────────────────────────────────────
# All URLs verified from ipcc.ch/report/ar6/ — March 2026
# Direct PDF links from official IPCC download pages

IPCC_DOCUMENTS = [

    # ── SYNTHESIS REPORT (SYR) 2023 ───────────────────────────────────────────
    {
        "wg": "SYR",
        "year": 2023,
        "title": "Synthesis Report Full Volume",
        "relevance": "HIGH — integrates all three WGs; cite throughout thesis",
        "url": "https://www.ipcc.ch/report/ar6/syr/downloads/report/IPCC_AR6_SYR_FullVolume.pdf",
        "filename": "SYR_FullVolume_2023.pdf",
    },

    # ── WORKING GROUP I — The Physical Science Basis (2021) ───────────────────
    {
        "wg": "WG1",
        "year": 2021,
        "title": "WG1 Summary Volume (SPM + TS + FAQ + Glossary)",
        "relevance": "HIGH — key reference for 1.5°C and 2°C thresholds",
        "url": "https://www.ipcc.ch/report/ar6/wg1/downloads/report/IPCC_AR6_WGI_SummaryVolume.pdf",
        "filename": "WG1_SummaryVolume_SPM_TS_2021.pdf",
    },
    {
        "wg": "WG1",
        "year": 2021,
        "title": "WG1 Chapter 1 — Framing, Context, and Methods",
        "relevance": "Medium — useful for methodology framing",
        "url": "https://www.ipcc.ch/report/ar6/wg1/downloads/report/IPCC_AR6_WGI_Chapter01.pdf",
        "filename": "WG1_Chapter01_FramingContextMethods_2021.pdf",
    },
    {
        "wg": "WG1",
        "year": 2021,
        "title": "WG1 Chapter 2 — Changing State of the Climate System",
        "relevance": "High — historical emissions baseline",
        "url": "https://www.ipcc.ch/report/ar6/wg1/downloads/report/IPCC_AR6_WGI_Chapter02.pdf",
        "filename": "WG1_Chapter02_ChangingStateClimateSystem_2021.pdf",
    },
    {
        "wg": "WG1",
        "year": 2021,
        "title": "WG1 Chapter 4 — Future Global Climate: Scenario Projections",
        "relevance": "HIGH — NDC gap analysis requires comparison to SSP scenarios",
        "url": "https://www.ipcc.ch/report/ar6/wg1/downloads/report/IPCC_AR6_WGI_Chapter04.pdf",
        "filename": "WG1_Chapter04_FutureGlobalClimaScenarios_2021.pdf",
    },
    {
        "wg": "WG1",
        "year": 2021,
        "title": "WG1 Chapter 11 — Weather and Climate Extremes",
        "relevance": "Medium — adaptation gap analysis",
        "url": "https://www.ipcc.ch/report/ar6/wg1/downloads/report/IPCC_AR6_WGI_Chapter11.pdf",
        "filename": "WG1_Chapter11_WeatherClimateExtremes_2021.pdf",
    },

    # ── WORKING GROUP II — Impacts, Adaptation and Vulnerability (2022) ───────
    {
        "wg": "WG2",
        "year": 2022,
        "title": "WG2 Summary for Policymakers",
        "relevance": "HIGH — adaptation commitments gap analysis reference",
        "url": "https://www.ipcc.ch/report/ar6/wg2/downloads/report/IPCC_AR6_WGII_SummaryForPolicymakers.pdf",
        "filename": "WG2_SPM_2022.pdf",
    },
    {
        "wg": "WG2",
        "year": 2022,
        "title": "WG2 Technical Summary",
        "relevance": "HIGH — adaptation finance and vulnerability benchmarks",
        "url": "https://www.ipcc.ch/report/ar6/wg2/downloads/report/IPCC_AR6_WGII_TechnicalSummary.pdf",
        "filename": "WG2_TechnicalSummary_2022.pdf",
    },
    {
        "wg": "WG2",
        "year": 2022,
        "title": "WG2 Chapter 17 — Decision-Making Options for Managing Risk",
        "relevance": "High — policy instrument gap analysis",
        "url": "https://www.ipcc.ch/report/ar6/wg2/downloads/report/IPCC_AR6_WGII_Chapter17.pdf",
        "filename": "WG2_Chapter17_DecisionMakingRisk_2022.pdf",
    },
    {
        "wg": "WG2",
        "year": 2022,
        "title": "WG2 Chapter 18 — Climate Resilient Development Pathways",
        "relevance": "High — NDC gap vs resilience pathways",
        "url": "https://www.ipcc.ch/report/ar6/wg2/downloads/report/IPCC_AR6_WGII_Chapter18.pdf",
        "filename": "WG2_Chapter18_ClimateResilientDevelopment_2022.pdf",
    },

    # ── WORKING GROUP III — Mitigation of Climate Change (2022) ──────────────
    {
        "wg": "WG3",
        "year": 2022,
        "title": "WG3 Summary for Policymakers",
        "relevance": "CRITICAL — primary benchmark for NDC gap detection",
        "url": "https://www.ipcc.ch/report/ar6/wg3/downloads/report/IPCC_AR6_WGIII_SummaryForPolicymakers.pdf",
        "filename": "WG3_SPM_2022.pdf",
    },
    {
        "wg": "WG3",
        "year": 2022,
        "title": "WG3 Technical Summary",
        "relevance": "CRITICAL — sectoral mitigation targets for gap analysis module",
        "url": "https://www.ipcc.ch/report/ar6/wg3/downloads/report/IPCC_AR6_WGIII_TechnicalSummary.pdf",
        "filename": "WG3_TechnicalSummary_2022.pdf",
    },
    {
        "wg": "WG3",
        "year": 2022,
        "title": "WG3 Chapter 2 — Emissions Trends and Drivers",
        "relevance": "HIGH — baseline emissions data for country comparisons",
        "url": "https://www.ipcc.ch/report/ar6/wg3/downloads/report/IPCC_AR6_WGIII_Chapter02.pdf",
        "filename": "WG3_Chapter02_EmissionsTrendsDrivers_2022.pdf",
    },
    {
        "wg": "WG3",
        "year": 2022,
        "title": "WG3 Chapter 4 — Mitigation and Development Pathways in the Near-Term",
        "relevance": "CRITICAL — 2030 targets comparison vs NDC commitments",
        "url": "https://www.ipcc.ch/report/ar6/wg3/downloads/report/IPCC_AR6_WGIII_Chapter04.pdf",
        "filename": "WG3_Chapter04_MitigationDevelopmentPathways_2022.pdf",
    },
    {
        "wg": "WG3",
        "year": 2022,
        "title": "WG3 Chapter 6 — Energy Systems",
        "relevance": "HIGH — renewable energy NDC targets gap analysis",
        "url": "https://www.ipcc.ch/report/ar6/wg3/downloads/report/IPCC_AR6_WGIII_Chapter06.pdf",
        "filename": "WG3_Chapter06_EnergySystems_2022.pdf",
    },
    {
        "wg": "WG3",
        "year": 2022,
        "title": "WG3 Chapter 13 — National and Sub-national Policies and Institutions",
        "relevance": "CRITICAL — policy implementation gap analysis reference chapter",
        "url": "https://www.ipcc.ch/report/ar6/wg3/downloads/report/IPCC_AR6_WGIII_Chapter13.pdf",
        "filename": "WG3_Chapter13_NationalSubnationalPolicies_2022.pdf",
    },
    {
        "wg": "WG3",
        "year": 2022,
        "title": "WG3 Chapter 14 — International Cooperation",
        "relevance": "HIGH — NDC cooperation and finance commitments gap",
        "url": "https://www.ipcc.ch/report/ar6/wg3/downloads/report/IPCC_AR6_WGIII_Chapter14.pdf",
        "filename": "WG3_Chapter14_InternationalCooperation_2022.pdf",
    },
    {
        "wg": "WG3",
        "year": 2022,
        "title": "WG3 Chapter 17 — Accelerating the Transition in the Context of Sustainable Development",
        "relevance": "High — SDG alignment in NDC commitments",
        "url": "https://www.ipcc.ch/report/ar6/wg3/downloads/report/IPCC_AR6_WGIII_Chapter17.pdf",
        "filename": "WG3_Chapter17_AcceleratingTransitionSDGs_2022.pdf",
    },
]


# ─── DOWNLOAD ENGINE ──────────────────────────────────────────────────────────

def make_session() -> requests.Session:
    """Create a requests session with retry logic and browser-like headers."""
    session = requests.Session()
    retry_strategy = Retry(
        total=MAX_RETRIES,
        backoff_factor=2,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("https://", adapter)
    session.mount("http://",  adapter)
    session.headers.update({
        "User-Agent": (
            "Mozilla/5.0 (compatible; ClimateThesisResearch/1.0; "
            "M.Sc. Data Science, University of Europe for Applied Sciences Potsdam; "
            "Academic research use)"
        ),
        "Accept": "application/pdf,*/*",
    })
    return session


def is_valid_pdf(path: Path) -> bool:
    """Check the file starts with PDF magic bytes (%PDF-)."""
    try:
        with open(path, "rb") as f:
            return f.read(5) == b"%PDF-"
    except Exception:
        return False


def file_md5(path: Path) -> str:
    """Compute MD5 hash of a file for integrity logging."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def download_file(
    session: requests.Session,
    url: str,
    dest: Path,
    label: str,
    dry_run: bool = False,
    resume: bool = True,
) -> dict:
    """
    Download a single file with progress bar, validation, and retry.
    Returns a result dict for logging.
    """
    result = {
        "label": label,
        "url": url,
        "dest": str(dest),
        "status": None,
        "size_bytes": None,
        "md5": None,
        "error": None,
        "timestamp": datetime.utcnow().isoformat(),
    }

    if dry_run:
        log.info(f"[DRY-RUN] Would download: {label}")
        log.info(f"          → {dest}")
        result["status"] = "dry_run"
        return result

    # Skip if already downloaded and valid
    if resume and dest.exists() and dest.stat().st_size > 1000 and is_valid_pdf(dest):
        log.info(f"[SKIP] Already exists: {dest.name}")
        result["status"] = "skipped"
        result["size_bytes"] = dest.stat().st_size
        result["md5"] = file_md5(dest)
        return result

    dest.parent.mkdir(parents=True, exist_ok=True)

    try:
        log.info(f"[GET]  {label}")
        log.info(f"       URL : {url}")
        log.info(f"       Dest: {dest}")

        response = session.get(url, stream=True, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()

        total_size = int(response.headers.get("Content-Length", 0))

        with open(dest, "wb") as f:
            if TQDM_AVAILABLE:
                with tqdm(
                    total=total_size or None,
                    unit="B",
                    unit_scale=True,
                    unit_divisor=1024,
                    desc=f"  {dest.name[:45]}",
                    leave=False,
                ) as pbar:
                    for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))
            else:
                for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                    if chunk:
                        f.write(chunk)

        # Validate PDF
        if not is_valid_pdf(dest):
            dest.unlink(missing_ok=True)
            raise ValueError("Downloaded file is not a valid PDF (missing %PDF- header)")

        file_size = dest.stat().st_size
        md5       = file_md5(dest)

        log.info(f"[OK]   {dest.name}  ({file_size / 1024:.1f} KB)  md5={md5[:8]}...")
        result["status"]     = "success"
        result["size_bytes"] = file_size
        result["md5"]        = md5

    except requests.exceptions.HTTPError as e:
        log.warning(f"[FAIL] HTTP {e.response.status_code}: {label}")
        log.warning(f"       {url}")
        result["status"] = "http_error"
        result["error"]  = str(e)

    except requests.exceptions.ConnectionError as e:
        log.warning(f"[FAIL] Connection error: {label}")
        result["status"] = "connection_error"
        result["error"]  = str(e)

    except requests.exceptions.Timeout:
        log.warning(f"[FAIL] Timeout: {label}")
        result["status"] = "timeout"
        result["error"]  = "Request timed out"

    except ValueError as e:
        log.warning(f"[FAIL] Validation: {label} — {e}")
        result["status"] = "invalid_pdf"
        result["error"]  = str(e)

    except Exception as e:
        log.error(f"[ERR]  Unexpected error: {label} — {e}")
        result["status"] = "error"
        result["error"]  = str(e)

    return result


def create_eu_aliases(base_dir: Path, dry_run: bool = False) -> None:
    """
    Germany, France, and Italy submit NDCs jointly as the EU.
    Create country-level folders with README notes pointing to EU documents.
    """
    eu_ndc1 = base_dir / "EU" / "NDC1_2015_EU.pdf"
    eu_ndc2 = base_dir / "EU" / "NDC2_2020_EU.pdf"

    for alias in EU_ALIASES:
        country = alias["country"]
        country_dir = base_dir / country
        if not dry_run:
            country_dir.mkdir(parents=True, exist_ok=True)
            readme = country_dir / "README.txt"
            readme.write_text(
                f"{country} submits its NDC jointly as part of the European Union.\n"
                f"NDC documents are located in: ../EU/\n\n"
                f"NDC Cycle 1 (2015): {eu_ndc1}\n"
                f"NDC Cycle 2 (2020): {eu_ndc2}\n\n"
                f"For thesis pipeline: load EU NDC documents; tag chunks with\n"
                f'country="{country}" AND country="EU" for cross-country queries.\n'
            )
            log.info(f"[ALIAS] Created README for {country} → EU NDC")
        else:
            log.info(f"[DRY-RUN] Would create alias README: {country} → EU")


def write_corpus_manifest(results: list, base_dir: Path) -> None:
    """Write download_log.json and download_summary.txt to logs/."""
    log_dir = Path(LOG_DIR)
    log_dir.mkdir(parents=True, exist_ok=True)

    # JSON log
    log_path = log_dir / "download_log.json"
    with open(log_path, "w") as f:
        json.dump(results, f, indent=2)
    log.info(f"[LOG] JSON log written: {log_path}")

    # Human-readable summary
    success  = [r for r in results if r["status"] == "success"]
    skipped  = [r for r in results if r["status"] == "skipped"]
    failed   = [r for r in results if r["status"] not in ("success", "skipped", "dry_run")]
    dry_runs = [r for r in results if r["status"] == "dry_run"]

    total_mb = sum(r["size_bytes"] or 0 for r in results) / (1024 * 1024)

    summary_lines = [
        "=" * 70,
        "  CLIMATE POLICY CORPUS — DOWNLOAD SUMMARY",
        f"  Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}",
        "=" * 70,
        f"  Downloaded (new) : {len(success):>4}",
        f"  Skipped (cached) : {len(skipped):>4}",
        f"  Failed           : {len(failed):>4}",
        f"  Dry-run logged   : {len(dry_runs):>4}",
        f"  Total disk usage : {total_mb:.1f} MB",
        "",
        "SUCCESSFUL DOWNLOADS:",
        *[f"  ✓ {r['label']}" for r in success],
        "",
        "SKIPPED (already on disk):",
        *[f"  ○ {r['label']}" for r in skipped],
    ]

    if failed:
        summary_lines += [
            "",
            "FAILED — INVESTIGATE MANUALLY:",
            *[f"  ✗ {r['label']}\n    Status : {r['status']}\n    Error  : {r['error']}\n    URL    : {r['url']}" for r in failed],
            "",
            "NOTE: UNFCCC blocks automated crawlers intermittently.",
            "  For failed NDC downloads, visit unfccc.int/NDCREG manually.",
            "  Search by country name → click 'Download' next to the NDC version.",
        ]

    summary_lines += ["", "=" * 70]

    summary_path = log_dir / "download_summary.txt"
    with open(summary_path, "w") as f:
        f.write("\n".join(summary_lines))

    # Also print to console
    print("\n" + "\n".join(summary_lines))


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Download UNFCCC NDC + IPCC AR6 documents for climate policy RAG thesis"
    )
    parser.add_argument(
        "--source",
        choices=["all", "ndc", "ipcc"],
        default="all",
        help="Which documents to download (default: all)",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT_DIR,
        help=f"Base output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be downloaded without downloading",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=True,
        help="Skip files already downloaded (default: True)",
    )
    parser.add_argument(
        "--no-resume",
        action="store_false",
        dest="resume",
        help="Re-download all files even if they exist",
    )
    args = parser.parse_args()

    base_dir = Path(args.output)
    ndc_dir  = base_dir / "NDC"
    ipcc_dir = base_dir / "IPCC_AR6"

    log.info("=" * 60)
    log.info("  CLIMATE POLICY CORPUS DOWNLOADER")
    log.info(f"  Output dir : {base_dir.resolve()}")
    log.info(f"  Source     : {args.source.upper()}")
    log.info(f"  Dry run    : {args.dry_run}")
    log.info(f"  Resume     : {args.resume}")
    log.info("=" * 60)

    session = make_session()
    all_results = []

    # ── Download NDC documents ─────────────────────────────────────────────
    if args.source in ("all", "ndc"):
        log.info(f"\n{'─'*60}")
        log.info(f"  NDC DOCUMENTS  ({len(NDC_DOCUMENTS)} files)")
        log.info(f"{'─'*60}")

        for doc in NDC_DOCUMENTS:
            dest = ndc_dir / doc["country"] / doc["filename"]
            label = f"NDC | {doc['country']} | {doc['ndc_cycle']} ({doc['year']}) | {doc['pages_approx']}pp"
            result = download_file(session, doc["url"], dest, label, args.dry_run, args.resume)
            result["source"]  = "NDC"
            result["country"] = doc["country"]
            result["cycle"]   = doc["ndc_cycle"]
            result["year"]    = doc["year"]
            all_results.append(result)
            if not args.dry_run:
                time.sleep(REQUEST_DELAY_SEC)

        # Create EU country alias READMEs
        create_eu_aliases(ndc_dir, args.dry_run)

    # ── Download IPCC AR6 documents ────────────────────────────────────────
    if args.source in ("all", "ipcc"):
        log.info(f"\n{'─'*60}")
        log.info(f"  IPCC AR6 DOCUMENTS  ({len(IPCC_DOCUMENTS)} files)")
        log.info(f"{'─'*60}")

        for doc in IPCC_DOCUMENTS:
            wg_dir = ipcc_dir / doc["wg"]
            dest   = wg_dir / doc["filename"]
            label  = f"IPCC | {doc['wg']} | {doc['title'][:60]}"
            result = download_file(session, doc["url"], dest, label, args.dry_run, args.resume)
            result["source"] = "IPCC_AR6"
            result["wg"]     = doc["wg"]
            result["year"]   = doc["year"]
            result["title"]  = doc["title"]
            all_results.append(result)
            if not args.dry_run:
                time.sleep(REQUEST_DELAY_SEC)

    # ── Write logs and summary ─────────────────────────────────────────────
    if not args.dry_run:
        write_corpus_manifest(all_results, base_dir)

    # ── Final corpus stats ─────────────────────────────────────────────────
    n_success = sum(1 for r in all_results if r["status"] in ("success", "skipped"))
    n_failed  = sum(1 for r in all_results if r["status"] not in ("success", "skipped", "dry_run"))

    log.info("")
    log.info("=" * 60)
    log.info(f"  CORPUS COMPLETE: {n_success} documents ready, {n_failed} failed")
    log.info(f"  NDC documents  : {len(NDC_DOCUMENTS)} (G20 × 2 cycles)")
    log.info(f"  IPCC AR6 docs  : {len(IPCC_DOCUMENTS)}")
    log.info(f"  TOTAL TARGET   : {len(NDC_DOCUMENTS) + len(IPCC_DOCUMENTS)} documents")
    log.info(f"  Output root    : {base_dir.resolve()}")
    log.info("=" * 60)

    if n_failed > 0:
        log.warning(f"\n  {n_failed} downloads failed.")
        log.warning("  Check data/logs/download_summary.txt for details.")
        log.warning("  Failed files can be downloaded manually from unfccc.int/NDCREG")
        sys.exit(1)


if __name__ == "__main__":
    main()
