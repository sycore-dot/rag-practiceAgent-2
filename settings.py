"""Shared configuration for the Fire Safety Code Agent."""

from __future__ import annotations

import os
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
ENV_PATH = BASE_DIR / ".env"

INPUT_ROOMS_PATH = BASE_DIR / "input_rooms.json"
NBC_RULES_PATH = BASE_DIR / "nbc_rules.txt"
CHROMA_DIR = BASE_DIR / "chroma_db"
CHROMA_COLLECTION_NAME = "nbc_fire_rules"

CODE_AGENT_VERSION = "rag_v2"
DEFAULT_CHAT_MODEL = "gemini-2.5-flash"
DEFAULT_EMBEDDING_MODEL = "models/gemini-embedding-2"

PLACEHOLDER_VALUES = {
    "",
    "your_gemini_api_key_here",
    "your_google_api_key_here",
    "your_api_key_here",
}


class ProjectConfigurationError(RuntimeError):
    """Raised when project configuration is missing or invalid."""


def load_environment(require_env_file: bool = True) -> None:
    """Load variables from .env without overwriting existing shell values."""
    if require_env_file and not ENV_PATH.exists():
        raise ProjectConfigurationError(
            f"Missing .env file at {ENV_PATH}. Create it from .env.example."
        )

    try:
        from dotenv import load_dotenv
    except ModuleNotFoundError as error:
        raise ProjectConfigurationError(
            "python-dotenv is not installed. Run: pip install -r requirements.txt"
        ) from error

    if ENV_PATH.exists():
        load_dotenv(ENV_PATH, override=False)


def get_google_api_key() -> str:
    """Return a Gemini API key and mirror it into GOOGLE_API_KEY."""
    load_environment(require_env_file=True)
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY") or ""
    api_key = api_key.strip()

    if api_key.lower() in PLACEHOLDER_VALUES:
        raise ProjectConfigurationError(
            "GOOGLE_API_KEY or GEMINI_API_KEY is missing or still uses a placeholder."
        )

    os.environ.setdefault("GOOGLE_API_KEY", api_key)
    return api_key


def ensure_gemini_api_key() -> None:
    """Validate that Gemini credentials are available."""
    get_google_api_key()


def get_chat_model(require_env_file: bool = False) -> str:
    """Return the Gemini chat model name."""
    load_environment(require_env_file=require_env_file)
    return os.getenv("GEMINI_CHAT_MODEL", DEFAULT_CHAT_MODEL)


def get_embedding_model(require_env_file: bool = False) -> str:
    """Return the Gemini embedding model name."""
    load_environment(require_env_file=require_env_file)
    return os.getenv("GEMINI_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
