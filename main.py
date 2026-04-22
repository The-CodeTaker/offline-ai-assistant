"""
main.py — Entry point for the Offline AI Personal Assistant (GUI edition).

Responsibilities:
  - Configure logging
  - Verify the local Ollama server is reachable
  - Initialise the SQLite database
  - Construct the backend singletons (Assistant, VoiceListener, VoiceSpeaker)
  - Hand them to AssistantGUI and start the CustomTkinter event loop

The terminal-based run_hybrid_loop() has been replaced by gui.py.
All backend files (core/, voice/, skills/, memory/) are unchanged.
"""

import sys
import ollama
from loguru import logger

import config
from memory.database import init_db
from core.assistant import Assistant
from voice.listener import VoiceListener
from voice.speaker import VoiceSpeaker
from gui import AssistantGUI


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}",
    level="INFO",
)
logger.add(
    "data/assistant.log",
    rotation="1 MB",
    retention="7 days",
    level="DEBUG",
)


# ---------------------------------------------------------------------------
# Ollama health check
# ---------------------------------------------------------------------------

def check_ollama_server() -> bool:
    """
    Verify that the local Ollama server is running and the configured
    model is available.

    Returns:
        True if the server is reachable and the model exists.
        False otherwise.
    """
    try:
        available_models = [m.model for m in ollama.list().models]
        logger.debug(f"Available Ollama models: {available_models}")

        if config.MODEL_NAME not in available_models:
            logger.warning(
                f"Model '{config.MODEL_NAME}' not found locally. "
                f"Run:  ollama pull {config.MODEL_NAME}"
            )
            return False

        logger.info(f"Ollama server OK — using model: {config.MODEL_NAME}")
        return True

    except Exception as exc:
        logger.error(
            f"Cannot reach Ollama server: {exc}\n"
            "Make sure Ollama is installed and running:  ollama serve"
        )
        return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    logger.info("Starting Offline AI Assistant (GUI mode)...")

    # 1. Verify Ollama is reachable
    if not check_ollama_server():
        logger.critical("Ollama check failed. Exiting.")
        sys.exit(1)

    # 2. Initialise SQLite database (creates tables if absent)
    logger.info(f"Initialising database at: {config.DB_PATH}")
    init_db(config.DB_PATH)

    # 3. Construct backend singletons.
    #    Created here before the GUI so any heavy initialisation
    #    (model loading, PyAudio setup) surfaces before the window appears.
    logger.info("Loading Assistant...")
    assistant = Assistant(model=config.MODEL_NAME, db_path=config.DB_PATH)

    logger.info("Loading VoiceListener...")
    listener = VoiceListener()

    logger.info("Loading VoiceSpeaker...")
    speaker = VoiceSpeaker()

    # 4. Launch GUI — blocks until the window is closed
    logger.info("Launching GUI...")
    app = AssistantGUI(assistant=assistant, listener=listener, speaker=speaker)
    app.mainloop()

    logger.info("GUI closed. Session ended.")


if __name__ == "__main__":
    main()
