"""
config.py — Central configuration for the Offline AI Assistant.
Override any value here or via a .env file (python-dotenv).
"""

import os
from pathlib import Path

# --- Paths ---
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

DB_PATH: str = str(DATA_DIR / "assistant.db")

# --- Ollama model ---
# Change to any model you have pulled locally, e.g. "mistral", "gemma3", etc.
MODEL_NAME: str = os.getenv("ASSISTANT_MODEL", "llama3:latest")

# --- Voice (used by voice/ modules) ---
VOICE_ENABLED: bool = False   # Set True once voice modules are wired up
SPEECH_LANGUAGE: str = "en-US"
