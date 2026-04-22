"""
core/assistant.py — Orchestrator: routes a user turn through the full pipeline.

RAG Architecture (this version)
---------------------------------
  TF-IDF has been replaced with a proper local vector database:
    • chromadb        — persistent on-disk vector store
    • sentence-transformers (all-MiniLM-L6-v2) — local semantic embeddings

  respond_stream_with_data(user_input, file_paths=[...]):
    1. Parses files via FileParser (images routed through llava vision model).
    2. Chunks document text into ~150-word segments with 30-word overlap.
    3. Embeds each chunk with all-MiniLM-L6-v2 and upserts into a
       per-session ChromaDB collection.
    4. At query time, embeds the user query with the same model and
       retrieves the top-K most semantically similar chunks.
    5. Injects the retrieved passages as task_data["retrieved_context"].
    6. Persists every completed turn to the conversations SQLite table.

Chunking strategy
-----------------
  Overlapping windows prevent answers from falling into seams between chunks.
  Each chunk is stored with metadata (filename, chunk index) so provenance
  can be traced in future versions.

ChromaDB collection naming
--------------------------
  One collection per session: "session_{session_id}"
  Collections persist on disk at data/chromadb/ so context survives restarts.

Backward compatibility
----------------------
  respond(user_input)  unchanged.
  respond_with_data()  unchanged.
  document_text kwarg kept for legacy GUI callers.
"""

from __future__ import annotations

import os
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

# ── RAG tuning constants ───────────────────────────────────────────────────
# Max characters returned as context (avoids saturating the context window)
_RAG_CONTEXT_CHARS = 6000
# Number of top chunks to retrieve from the vector store
_RAG_TOP_K = 5
# Chunk size in words and overlap in words
_CHUNK_WORDS   = 150
_OVERLAP_WORDS = 30
# Local sentence-transformer model (downloads once, ~90 MB)
_EMBED_MODEL   = "all-MiniLM-L6-v2"
# ChromaDB persistence directory (relative to project root)
_CHROMA_DIR    = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "chromadb",
)


class Assistant:
    """
    Top-level orchestrator.

    Parameters
    ----------
    model      : Ollama model name.
    db_path    : SQLite database path.
    session_id : Active session id.  None → auto-create a new session.
    """

    MAX_HISTORY_TURNS = 20

    def __init__(
        self,
        model:      str        = config.MODEL_NAME,
        db_path:    str        = config.DB_PATH,
        session_id: int | None = None,
    ) -> None:
        self.model      = model
        self.db_path    = db_path
        self.classifier = IntentClassifier(model=model)
        self.generator  = DialogueGenerator(model=model)

        # ── Vector store (lazy-initialised per session) ────────────────
        self._chroma_client     = None   # chromadb.PersistentClient
        self._embed_fn          = None   # SentenceTransformer instance
        self._vector_collection = None   # current session's collection
        self._vector_ready      = False  # set True after first successful init

        # ── Session ────────────────────────────────────────────────────
        if session_id is None:
            self.session_id = create_session(db_path, title="New Chat")
            logger.info(f"Assistant: auto-created session #{self.session_id}")
        else:
            self.session_id = session_id
            logger.info(f"Assistant: using existing session #{self.session_id}")

        # Initialise the vector store for this session
        self._init_vector_store()
        logger.info("Assistant orchestrator initialised.")

    # ------------------------------------------------------------------
    # Vector store initialisation
    # ------------------------------------------------------------------

    def _init_vector_store(self) -> None:
        """
        Initialise chromadb + sentence-transformers.

        Called once at construction and again when switching sessions.
        Failures are non-fatal: _vector_ready stays False and the system
        degrades gracefully (no RAG context injected).
        """
        try:
            import chromadb  # type: ignore
            from sentence_transformers import SentenceTransformer  # type: ignore

            os.makedirs(_CHROMA_DIR, exist_ok=True)

            # Reuse the embedding model across session switches
            if self._embed_fn is None:
                logger.info(f"Loading embedding model '{_EMBED_MODEL}'…")
                self._embed_fn = SentenceTransformer(_EMBED_MODEL)
                logger.info("Embedding model loaded.")

            # Reuse the Chroma client
            if self._chroma_client is None:
                self._chroma_client = chromadb.PersistentClient(path=_CHROMA_DIR)

            collection_name = f"session_{self.session_id}"
            self._vector_collection = self._chroma_client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"},
            )
            self._vector_ready = True
            logger.info(
                f"ChromaDB collection '{collection_name}' ready "
                f"({self._vector_collection.count()} chunks)"
            )

        except ImportError as exc:
            logger.warning(
                f"Vector store unavailable ({exc}). "
                "Install:  pip install chromadb sentence-transformers\n"
                "RAG will be disabled until these packages are installed."
            )
            self._vector_ready = False
        except Exception as exc:
            logger.error(f"_init_vector_store failed: {exc}")
            self._vector_ready = False

    # ------------------------------------------------------------------
    # Session management
    # ------------------------------------------------------------------

    def switch_session(self, session_id: int) -> None:
        """Switch to an existing session and reinitialise the vector collection."""
        self.session_id = session_id
        self._vector_collection = None
        self._init_vector_store()
        logger.info(f"Assistant: switched to session #{session_id}")

    # ------------------------------------------------------------------
    # Public API — backward-compatible blocking variants
    # ------------------------------------------------------------------

    def respond(self, user_input: str) -> str:
        """Process one turn and return the reply string only."""
        reply, _ = self.respond_with_data(user_input)
        return reply

    def respond_with_data(self, user_input: str) -> tuple[str, dict | None]:
        """Blocking variant — for backward compatibility."""
        history       = self._load_history()
        intent_result = self.classifier.classify(user_input)
        logger.info(f"Intent -> {intent_result}")
        task_data     = self._dispatch_skill(intent_result, user_input)
        reply         = self.generator.generate(
            user_message=user_input,
            intent_result=intent_result,
            task_data=task_data,
            history=history,
        )
        self._update_history(user_input, reply)
        return reply, task_data

    # ------------------------------------------------------------------
    # Public API — streaming with vector RAG
    # ------------------------------------------------------------------

    def respond_stream_with_data(
        self,
        user_input:    str,
        file_paths:    list[str] | None = None,
        document_text: str              = "",   # legacy compat
    ) -> Generator[str | dict | None, None, None]:
        """
        Stream the LLM response, yielding tokens followed by a sentinel dict.

        Yield protocol
        --------------
        • str   — live text token while the model generates.
        • dict  — final sentinel: {"__stream_done__": True, "task_data": ...}

        Parameters
        ----------
        user_input    : User's message.
        file_paths    : Files to parse and embed into the session vector store.
        document_text : (Deprecated) pre-extracted text string.
        """
        # ── 1. Ingest new files ────────────────────────────────────────
        if file_paths:
            for fp in file_paths:
                self._ingest_file(fp)

        if document_text:
            self._ingest_raw_text(document_text, filename="<attached>")

        # ── 2. Classify intent ─────────────────────────────────────────
        intent_result = self.classifier.classify(user_input)
        logger.info(f"[stream] Intent -> {intent_result}")

        # ── 3. Execute skill ───────────────────────────────────────────
        task_data = self._dispatch_skill(intent_result, user_input)

        # ── 4. Vector RAG retrieval ────────────────────────────────────
        retrieved_context = self._retrieve_context(user_input)
        if retrieved_context:
            if task_data is None:
                task_data = {}
            elif not isinstance(task_data, dict):
                task_data = {"_original": task_data}
            task_data["retrieved_context"] = retrieved_context
            logger.debug(
                f"[stream] Injected RAG context ({len(retrieved_context)} chars)"
            )

        # ── 5. Load persistent conversation history ────────────────────
        history = self._load_history()

        # ── 6. Stream dialogue tokens ──────────────────────────────────
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

        # ── 7. Persist turn ────────────────────────────────────────────
        full_reply = "".join(accumulated)
        self._update_history(user_input, full_reply)
        logger.debug(f"[stream] Accumulated reply: {len(full_reply)} chars")

        # ── 8. Yield sentinel ─────────────────────────────────────────
        yield {"__stream_done__": True, "task_data": task_data}

    # ------------------------------------------------------------------
    # Skill dispatcher
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
    # RAG — ingestion helpers
    # ------------------------------------------------------------------

    def _ingest_file(self, filepath: str) -> None:
        """
        Parse a file and embed its content into both:
          • SQLite session_documents table (for persistence across restarts)
          • ChromaDB vector store         (for semantic retrieval)
        """
        try:
            text = FileParser.parse_file(filepath)
            if not text.strip():
                logger.warning(f"_ingest_file: empty text from '{filepath}'")
                return
            filename = os.path.basename(filepath)
            # Persist to SQLite
            save_document(self.db_path, self.session_id, filename, text)
            # Embed into vector store
            self._embed_and_upsert(text, source=filename)
        except Exception as exc:
            logger.error(f"_ingest_file('{filepath}'): {exc}")

    def _ingest_raw_text(self, text: str, filename: str = "<text>") -> None:
        """Save pre-extracted text to both SQLite and the vector store."""
        if not text.strip():
            return
        try:
            save_document(self.db_path, self.session_id, filename, text)
            self._embed_and_upsert(text, source=filename)
        except Exception as exc:
            logger.error(f"_ingest_raw_text: {exc}")

    def _embed_and_upsert(self, text: str, source: str) -> None:
        """
        Chunk *text* into overlapping windows, embed each chunk, and
        upsert them into the session's ChromaDB collection.

        Chunking uses a sliding window of _CHUNK_WORDS words with
        _OVERLAP_WORDS overlap so no context falls into a seam.

        If the vector store is not ready, this is a silent no-op.
        """
        if not self._vector_ready or self._vector_collection is None:
            logger.debug("_embed_and_upsert: vector store not ready, skipping.")
            return

        try:
            words  = text.split()
            chunks = []
            step   = max(1, _CHUNK_WORDS - _OVERLAP_WORDS)

            for i in range(0, len(words), step):
                chunk = " ".join(words[i : i + _CHUNK_WORDS])
                if chunk.strip():
                    chunks.append(chunk)

            if not chunks:
                return

            # Embed all chunks in one batch (much faster than one-by-one)
            embeddings = self._embed_fn.encode(chunks, show_progress_bar=False).tolist()

            # Build stable IDs: "source::chunk_index"
            # Use sanitised source name to avoid ChromaDB ID restrictions
            safe_source = source.replace("/", "_").replace("\\", "_")[:60]
            ids = [f"{safe_source}::{i}" for i in range(len(chunks))]

            self._vector_collection.upsert(
                ids=ids,
                embeddings=embeddings,
                documents=chunks,
                metadatas=[{"source": source, "chunk_idx": i} for i in range(len(chunks))],
            )
            logger.info(
                f"_embed_and_upsert: upserted {len(chunks)} chunks "
                f"from '{source}' into '{self._vector_collection.name}'"
            )

        except Exception as exc:
            logger.error(f"_embed_and_upsert failed: {exc}")

    # ------------------------------------------------------------------
    # RAG — retrieval
    # ------------------------------------------------------------------

    def _retrieve_context(self, query: str) -> str:
        """
        Embed *query* and retrieve the top-K most semantically similar
        chunks from the session's ChromaDB collection.

        Returns "" if the vector store is empty, not ready, or errors.

        Why this beats TF-IDF
        ----------------------
        TF-IDF only matches on exact keyword overlap.  A query like
        "what does the contract say about payment terms?" won't match a
        chunk that uses "invoice", "due date", or "billing schedule".
        The all-MiniLM-L6-v2 model embeds both into a dense vector space
        where semantically related terms are geometrically close, so the
        retrieval finds relevant passages even when wording differs.
        """
        if not self._vector_ready or self._vector_collection is None:
            logger.debug("_retrieve_context: vector store not ready.")
            return self._fallback_retrieve_context(query)

        try:
            count = self._vector_collection.count()
            if count == 0:
                return ""

            # Embed the query
            query_embedding = self._embed_fn.encode([query], show_progress_bar=False).tolist()

            # Query the collection
            top_k = min(_RAG_TOP_K, count)
            results = self._vector_collection.query(
                query_embeddings=query_embedding,
                n_results=top_k,
                include=["documents", "metadatas", "distances"],
            )

            retrieved_docs  = results.get("documents",  [[]])[0]
            retrieved_meta  = results.get("metadatas",  [[]])[0]
            retrieved_dists = results.get("distances",  [[]])[0]

            if not retrieved_docs:
                return ""

            # Assemble context, labelling each chunk with its source
            parts: list[str] = []
            total_chars = 0

            for doc, meta, dist in zip(retrieved_docs, retrieved_meta, retrieved_dists):
                if total_chars >= _RAG_CONTEXT_CHARS:
                    break
                # Cosine distance: 0 = identical, 2 = orthogonal
                # Filter out chunks that are almost certainly irrelevant
                if dist > 1.2:
                    logger.debug(f"_retrieve_context: skipping chunk (distance={dist:.3f})")
                    continue

                source  = meta.get("source", "document")
                snippet = doc.strip()
                label   = f"[Source: {source}]\n{snippet}"
                parts.append(label)
                total_chars += len(label)

            if not parts:
                return ""

            context = "\n\n---\n\n".join(parts)
            logger.info(
                f"_retrieve_context: retrieved {len(parts)} chunks "
                f"({total_chars} chars) from '{self._vector_collection.name}'"
            )
            return context

        except Exception as exc:
            logger.error(f"_retrieve_context vector query failed: {exc}")
            return self._fallback_retrieve_context(query)

    def _fallback_retrieve_context(self, query: str) -> str:
        """
        Plain-text fallback when the vector store is unavailable.
        Loads all session documents from SQLite and returns the first
        _RAG_CONTEXT_CHARS characters.  Better than nothing.
        """
        try:
            docs = load_session_documents(self.db_path, self.session_id)
        except Exception as exc:
            logger.error(f"_fallback_retrieve_context: DB error: {exc}")
            return ""

        if not docs:
            return ""

        all_text = "\n\n".join(
            f"[Source: {d['filename']}]\n{d['content']}" for d in docs
        )
        logger.info(
            "RAG fallback: returning first "
            f"{_RAG_CONTEXT_CHARS} chars of session documents "
            "(install chromadb + sentence-transformers for proper retrieval)"
        )
        return all_text[:_RAG_CONTEXT_CHARS]

    # ------------------------------------------------------------------
    # History helpers (DB-backed)
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
        """Persist the latest turn to the SQLite conversations table."""
        try:
            save_conversation_turn(self.db_path, self.session_id, "user",      user_input)
            save_conversation_turn(self.db_path, self.session_id, "assistant", reply)
        except Exception as exc:
            logger.error(f"_update_history: {exc}")
