"""
core/assistant.py — Orchestrator: routes a user turn through the full pipeline.

RAG Architecture (this version)
---------------------------------
  respond_stream_with_data(user_input, file_paths=[...]) now:
    1. Parses any supplied files via FileParser and persists them to
       session_documents table linked to self.session_id.
    2. Loads ALL document text stored for the current session.
    3. Uses TF-IDF cosine similarity to rank 500-word chunks against the
       user query and injects the top-2000-word excerpt into task_data as
       task_data["retrieved_context"].
    4. Persists every completed turn to the conversations DB table.

Session awareness
-----------------
  The Assistant now accepts session_id in its constructor.  A new session
  is created automatically if session_id=None is passed (the GUI passes
  the active session id and manages creation externally).

Backward compatibility
----------------------
  respond(user_input)  still works unchanged.
  respond_with_data()  still works unchanged.
  History is now loaded from the DB each call (db_path + session_id) so
  history is truly persistent across app restarts.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Generator
from loguru import logger

import config
from core.intent import IntentClassifier, IntentResult
from core.dialogue import DialogueGenerator
from skills.reminder import ReminderSkill
from skills.notes import NoteSkill
from skills.weather import WeatherSkill
from skills.navigation import NavigationSkill
from memory.database import (
    create_session,
    save_conversation_turn,
    load_conversation_history,
    save_document,
    load_session_documents,
)
from utils.file_parser import FileParser

# Word limit injected as RAG context
_RAG_CONTEXT_WORDS = 2000
# Chunk size for TF-IDF document splitting (words)
_CHUNK_WORDS = 500


class Assistant:
    """
    Top-level orchestrator.

    Parameters
    ----------
    model      : Ollama model name.
    db_path    : SQLite database path.
    session_id : Active session id.  Pass None to auto-create a new session.
    """

    MAX_HISTORY_TURNS = 20   # kept in DB now, not just RAM

    def __init__(
        self,
        model:      str = config.MODEL_NAME,
        db_path:    str = config.DB_PATH,
        session_id: int | None = None,
    ) -> None:
        self.model      = model
        self.db_path    = db_path
        self.classifier = IntentClassifier(model=model)
        self.generator  = DialogueGenerator(model=model)

        # Session management
        if session_id is None:
            self.session_id = create_session(db_path, title="New Chat")
            logger.info(f"Assistant: auto-created session #{self.session_id}")
        else:
            self.session_id = session_id
            logger.info(f"Assistant: using existing session #{self.session_id}")

        logger.info("Assistant orchestrator initialised.")

    # ------------------------------------------------------------------
    # Public API — session management
    # ------------------------------------------------------------------

    def switch_session(self, session_id: int) -> None:
        """Switch the assistant to an existing session."""
        self.session_id = session_id
        logger.info(f"Assistant: switched to session #{session_id}")

    # ------------------------------------------------------------------
    # Public API — backward-compatible
    # ------------------------------------------------------------------

    def respond(self, user_input: str) -> str:
        """Process one turn and return the reply string only."""
        reply, _ = self.respond_with_data(user_input)
        return reply

    def respond_with_data(self, user_input: str) -> tuple[str, dict | None]:
        """Blocking version — for backward compatibility."""
        history    = self._load_history()
        intent_result: IntentResult = self.classifier.classify(user_input)
        logger.info(f"Intent -> {intent_result}")
        task_data  = self._dispatch_skill(intent_result, user_input)
        reply      = self.generator.generate(
            user_message=user_input,
            intent_result=intent_result,
            task_data=task_data,
            history=history,
        )
        self._update_history(user_input, reply)
        return reply, task_data

    # ------------------------------------------------------------------
    # Public API — streaming with RAG
    # ------------------------------------------------------------------

    def respond_stream_with_data(
        self,
        user_input:  str,
        file_paths:  list[str] | None = None,
        # Legacy parameter kept for backward compat with old GUI code
        document_text: str = "",
    ) -> Generator[str | dict | None, None, None]:
        """
        Stream the LLM response token-by-token, then yield a sentinel dict.

        Parameters
        ----------
        user_input    : The user's message.
        file_paths    : List of file paths to parse and persist for this session.
        document_text : (Deprecated) single pre-extracted text string.

        Yield protocol
        --------------
        • str chunks while streaming.
        • {"__stream_done__": True, "task_data": ...} as the final item.
        """
        # ── 1. Parse & persist any newly attached files ────────────────
        if file_paths:
            for fp in file_paths:
                self._ingest_file(fp)

        # ── 2. Legacy single-document support ─────────────────────────
        if document_text:
            self._ingest_raw_text(document_text, filename="<attached>")

        # ── 3. Classify intent ─────────────────────────────────────────
        intent_result: IntentResult = self.classifier.classify(user_input)
        logger.info(f"[stream] Intent -> {intent_result}")

        # ── 4. Execute skill ───────────────────────────────────────────
        task_data = self._dispatch_skill(intent_result, user_input)

        # ── 5. RAG: retrieve relevant context from session documents ───
        retrieved_context = self._retrieve_context(user_input)
        if retrieved_context:
            if task_data is None:
                task_data = {}
            elif not isinstance(task_data, dict):
                task_data = {"_original": task_data}
            task_data["retrieved_context"] = retrieved_context
            logger.debug(
                f"[stream] Injected RAG context ({len(retrieved_context.split())} words)"
            )

        # ── 6. Load persistent history from DB ────────────────────────
        history = self._load_history()

        # ── 7. Stream dialogue tokens ──────────────────────────────────
        accumulated: list[str] = []
        try:
            for token in self.generator.generate_stream(
                user_message=user_input,
                intent_result=intent_result,
                task_data=task_data,
                history=history,
            ):
                accumulated.append(token)
                yield token
        except Exception as exc:
            logger.error(f"[stream] generate_stream() error: {exc}")
            fallback = "I encountered an error while generating a response."
            yield fallback
            accumulated.append(fallback)

        # ── 8. Persist turn to DB ──────────────────────────────────────
        full_reply = "".join(accumulated)
        self._update_history(user_input, full_reply)
        logger.debug(f"[stream] Accumulated reply: {len(full_reply)} chars")

        # ── 9. Sentinel ────────────────────────────────────────────────
        yield {"__stream_done__": True, "task_data": task_data}

    # ------------------------------------------------------------------
    # Skill dispatcher  (unchanged from previous version)
    # ------------------------------------------------------------------

    def _dispatch_skill(self, intent_result: IntentResult, user_input: str) -> dict | None:
        intent = intent_result.intent
        e      = intent_result.entities

        if intent in ("greeting", "farewell", "general_chat", "unknown"):
            return None

        if intent == "set_reminder":
            return ReminderSkill(self.db_path).create(e)
        if intent == "get_reminders":
            return ReminderSkill(self.db_path).list_upcoming(entities=e)
        if intent == "create_note":
            return NoteSkill(self.db_path).create(e)
        if intent == "get_notes":
            return NoteSkill(self.db_path).list_all(e)
        if intent == "get_weather":
            return WeatherSkill(self.db_path).get_current(e)
        if intent == "search_travel":
            return NavigationSkill(self.db_path).route(e)
        if intent == "get_schedule":
            return {"status": "stub", "message": "Schedule skill not yet implemented."}

        return None

    # ------------------------------------------------------------------
    # RAG helpers
    # ------------------------------------------------------------------

    def _ingest_file(self, filepath: str) -> None:
        """Parse a file and save its content to session_documents."""
        import os
        try:
            text = FileParser.parse_file(filepath)
            if not text.strip():
                logger.warning(f"_ingest_file: empty text from '{filepath}'")
                return
            filename = os.path.basename(filepath)
            save_document(self.db_path, self.session_id, filename, text)
        except Exception as exc:
            logger.error(f"_ingest_file('{filepath}'): {exc}")

    def _ingest_raw_text(self, text: str, filename: str = "<text>") -> None:
        """Save pre-extracted text to session_documents."""
        if not text.strip():
            return
        try:
            save_document(self.db_path, self.session_id, filename, text)
        except Exception as exc:
            logger.error(f"_ingest_raw_text: {exc}")

    def _retrieve_context(self, query: str) -> str:
        """
        TF-IDF retrieval over all documents in the current session.

        Splits every document into 500-word chunks, ranks them against
        *query* using cosine similarity, and returns the top chunks up
        to _RAG_CONTEXT_WORDS words total.

        Returns "" if no documents are stored or sklearn is absent.
        """
        try:
            docs = load_session_documents(self.db_path, self.session_id)
        except Exception as exc:
            logger.error(f"_retrieve_context: DB error: {exc}")
            return ""

        if not docs:
            return ""

        # Build chunk list
        chunks: list[str] = []
        for doc in docs:
            words = doc["content"].split()
            for i in range(0, len(words), _CHUNK_WORDS):
                chunk = " ".join(words[i : i + _CHUNK_WORDS])
                if chunk.strip():
                    chunks.append(chunk)

        if not chunks:
            return ""

        try:
            from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
            from sklearn.metrics.pairwise import cosine_similarity         # type: ignore
            import numpy as np                                              # type: ignore

            corpus = [query] + chunks
            vec    = TfidfVectorizer().fit_transform(corpus)
            sims   = cosine_similarity(vec[0:1], vec[1:]).flatten()

            # Sort chunks by descending similarity
            ranked_indices = np.argsort(sims)[::-1]

            selected: list[str] = []
            word_count = 0
            for idx in ranked_indices:
                chunk_words = chunks[idx].split()
                if word_count + len(chunk_words) > _RAG_CONTEXT_WORDS:
                    remaining = _RAG_CONTEXT_WORDS - word_count
                    if remaining > 0:
                        selected.append(" ".join(chunk_words[:remaining]))
                    break
                selected.append(chunks[idx])
                word_count += len(chunk_words)
                if word_count >= _RAG_CONTEXT_WORDS:
                    break

            return "\n\n".join(selected)

        except ImportError:
            logger.warning(
                "scikit-learn not installed — RAG disabled. "
                "Install with:  pip install scikit-learn"
            )
            # Fallback: return the first _RAG_CONTEXT_WORDS words of all docs
            all_text = " ".join(d["content"] for d in docs)
            words    = all_text.split()
            return " ".join(words[:_RAG_CONTEXT_WORDS])

        except Exception as exc:
            logger.error(f"_retrieve_context TF-IDF error: {exc}")
            return ""

    # ------------------------------------------------------------------
    # History helpers  (now DB-backed)
    # ------------------------------------------------------------------

    def _load_history(self) -> list[dict[str, str]]:
        """Load recent conversation history from DB for the active session."""
        try:
            return load_conversation_history(
                self.db_path, self.session_id,
                limit=self.MAX_HISTORY_TURNS * 2,
            )
        except Exception as exc:
            logger.error(f"_load_history: {exc}")
            return []

    def _update_history(self, user_input: str, reply: str) -> None:
        """Persist the latest turn to the DB."""
        try:
            save_conversation_turn(self.db_path, self.session_id, "user", user_input)
            save_conversation_turn(self.db_path, self.session_id, "assistant", reply)
        except Exception as exc:
            logger.error(f"_update_history: {exc}")
