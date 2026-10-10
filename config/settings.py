"""Central configuration for AURA."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")
APP_NAME = "AURA"
APP_FULL_NAME = "AI Understanding, Reasoning & Analytics"
TAGLINE = "Data is powerful. Intelligence is everything."
APP_VERSION = "2.0.0"
DATA_DIR, REPORTS_DIR = BASE_DIR / "data", BASE_DIR / "reports"
ASSETS_DIR, DATABASE_DIR = BASE_DIR / "assets", BASE_DIR / "database"
DB_PATH, STYLE_PATH, LOGO_PATH = DATABASE_DIR / "aura.db", ASSETS_DIR / "style.css", ASSETS_DIR / "logo.svg"
ALLOWED_EXTENSIONS = (".csv", ".tsv", ".xlsx", ".xlsm")
MAX_UPLOAD_MB, MAX_ROWS, MAX_COLUMNS, PREVIEW_ROWS = 200, 2_000_000, 2_000, 100
IQR_MULTIPLIER, ZSCORE_THRESHOLD, TOP_CATEGORIES, HISTOGRAM_BINS = 1.5, 3.0, 10, 30
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip() or "gpt-4o-mini"
MAX_AGENT_STEPS, LLM_TIMEOUT_SECONDS, LLM_TEMPERATURE = 8, 60, 0.1
COLORS = {"background":"#09090D", "sidebar":"#101017", "card":"#171720", "accent":"#A78BFA", "accent_light":"#DDD6FE", "text":"#F7F5FF", "muted":"#9794A8", "border":"#2C2A38", "success":"#34D399", "warning":"#FBBF24", "danger":"#FB7185"}
CHART_PALETTE = ["#A78BFA", "#C4B5FD", "#7C3AED", "#60A5FA", "#34D399", "#FBBF24", "#F472B6", "#FB7185", "#2DD4BF", "#818CF8"]
_PLACEHOLDER_KEYS = {"", "your_openai_api_key_here", "sk-...", "none", "null"}

def ensure_directories() -> None:
    for directory in (DATA_DIR, REPORTS_DIR, ASSETS_DIR, DATABASE_DIR): directory.mkdir(parents=True, exist_ok=True)

def get_api_key() -> str | None:
    key = os.getenv("OPENAI_API_KEY", "").strip()
    return None if key.lower() in _PLACEHOLDER_KEYS else key
