import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file if present
load_dotenv()

# Base project paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DOCUMENTS_DIR = DATA_DIR / "documents"
PROCESSED_DIR = DATA_DIR / "processed"
EVALUATION_DIR = DATA_DIR / "evaluation"
VECTORSTORE_DIR = BASE_DIR / "vectorstore"

# Optional portable cache directory configuration
CLAUSE_CACHE_DIR = os.getenv("CLAUSE_CACHE_DIR")
if CLAUSE_CACHE_DIR:
    cache_path = Path(CLAUSE_CACHE_DIR)
    cache_path.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(cache_path / "huggingface"))
    os.environ.setdefault("FASTEMBED_CACHE_PATH", str(cache_path / "fastembed"))
    os.environ.setdefault("TORCH_HOME", str(cache_path / "torch"))

# Ensure essential directories exist
DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
EVALUATION_DIR.mkdir(parents=True, exist_ok=True)
VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)

# Chunking Configuration
DEFAULT_CHUNK_SIZE = int(os.getenv("DEFAULT_CHUNK_SIZE", "700"))
DEFAULT_CHUNK_OVERLAP = int(os.getenv("DEFAULT_CHUNK_OVERLAP", "120"))

# Embedding Model Configuration
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# Retrieval Engine Configuration
DEFAULT_RRF_K = int(os.getenv("DEFAULT_RRF_K", "60"))
DEFAULT_CANDIDATE_TOP_N = int(os.getenv("DEFAULT_CANDIDATE_TOP_N", "20"))
DEFAULT_FINAL_TOP_K = int(os.getenv("DEFAULT_FINAL_TOP_K", "5"))

# LLM & Generation Configuration
_st_gemini_key = ""
_st_gemini_model = ""
try:
    import streamlit as _st
    if hasattr(_st, "secrets"):
        _st_gemini_key = _st.secrets.get("GEMINI_API_KEY", "")
        _st_gemini_model = _st.secrets.get("GEMINI_MODEL", "")
except Exception:
    pass

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "") or _st_gemini_key
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "") or _st_gemini_model or "gemini-3.8-flash"
DEFAULT_LLM_TEMPERATURE = float(os.getenv("DEFAULT_LLM_TEMPERATURE", "0.0"))

# Evidence Sufficiency Gate Thresholds
SUFFICIENCY_MIN_SCORE = float(os.getenv("SUFFICIENCY_MIN_SCORE", "0.015"))
SUFFICIENCY_MIN_CHUNKS = int(os.getenv("SUFFICIENCY_MIN_CHUNKS", "1"))
