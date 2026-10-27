"""Centralized paths for local execution and Vercel's ephemeral runtime filesystem."""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
IS_VERCEL = os.getenv("VERCEL") == "1"

# Source datasets are committed with the application and must remain read-only.
INPUT_OUTPUTS_DIR = BASE_DIR / "outputs"
STATIC_DIR = BASE_DIR / "static"

# Serverless instances cannot persist writes in the repository directory.
WRITABLE_ROOT = Path("/tmp/manganex") if IS_VERCEL else BASE_DIR
OUTPUTS_DIR = WRITABLE_ROOT / "outputs"
MODELS_DIR = WRITABLE_ROOT / "models"
DATA_PROCESSED_DIR = WRITABLE_ROOT / "data" / "processed"
DB_PATH = WRITABLE_ROOT / "manganex.db"

if IS_VERCEL:
    WRITABLE_ROOT.mkdir(parents=True, exist_ok=True)
