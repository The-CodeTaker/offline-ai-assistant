"""
skills/notes.py — NoteSkill: create and retrieve notes in SQLite.

Notes table schema (defined in memory/database.py):
    id      INTEGER PRIMARY KEY AUTOINCREMENT
    title   TEXT                          — optional short label
    body    TEXT    NOT NULL              — the actual note content
    created DATETIME DEFAULT CURRENT_TIMESTAMP

ENTITY MAPPING FROM INTENT CLASSIFIER
--------------------------------------
For `create_note` the classifier is expected to emit:
    "title"    — short subject line, e.g. "Shopping list"      (optional)
    "raw_text" — the full body of the note as spoken/typed      (used as body
                  fallback if no dedicated body key is present)
    "body"     — explicit note body if the model separates it  (optional)

For `get_notes` no entities are required; an optional "title" or "query"
entity is used as a LIKE search filter when present.

DESIGN DECISIONS
----------------
- Inherits from BaseSkill for the standard db_path constructor, _success /
  _error helpers, and the enforced execute() contract.
- `execute()` acts as a smart router: it inspects whether the entities dict
  looks like a creation request (has "body" / "raw_text" with enough content)
  or a retrieval request, and delegates accordingly.  This covers cases where
  the LLM maps both sub-intents to a single "notes" intent label.
- Body text is never silently truncated; the full text is stored and the first
  120 characters are echoed back in the return dict as a "preview" so
  DialogueGenerator can give the user a quick confirmation without repeating
  the whole note.
- `list_all` returns rows ordered newest-first (most useful for a personal
  assistant) and caps at `limit` rows.
- All DB rows are converted to plain dicts before returning — no sqlite3.Row
  objects leak out — so the result is always directly JSON-serialisable.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime
from typing import Any

from loguru import logger

from skills.base_skill import BaseSkill


class NoteSkill(BaseSkill):
    """
    Handles creation and retrieval of notes against the `notes` SQLite table.

    Usage (from assistant._dispatch_skill):
        skill = NoteSkill(self.db_path)
        result = skill.create(intent_result.entities)
        result = skill.list_all(intent_result.entities)

        # Or via the generic entry-point (intent label = "notes"):
        result = skill.execute(intent_result.entities)
    """

    # Datetime format matching the SQLite DEFAULT CURRENT_TIMESTAMP format
    _DT_FMT = "%Y-%m-%d %H:%M:%S"
    # Number of chars shown in the "preview" field of create responses
    _PREVIEW_LEN = 120

    # ------------------------------------------------------------------
    # BaseSkill interface
    # ------------------------------------------------------------------

    def execute(self, entities: dict[str, Any]) -> dict[str, Any]:
        """
        Smart router: delegates to create() or list_all() based on entities.

        Heuristic: if the entities dict contains a "body" key, or a "raw_text"
        value that looks like note content (more than just a noun phrase),
        treat it as a creation request.  Otherwise treat it as a list request.
        This handles classifiers that map both sub-intents to one "notes" label.
        """
        body_candidate = entities.get("body") or entities.get("raw_text", "")
        has_content = len(str(body_candidate).split()) > 3  # more than 3 words

        if has_content:
            logger.debug("NoteSkill.execute: routing → create()")
            return self.create(entities)
        else:
            logger.debug("NoteSkill.execute: routing → list_all()")
            return self.list_all(entities)

    # ------------------------------------------------------------------
    # Public API — called directly from assistant._dispatch_skill
    # ------------------------------------------------------------------

    def create(self, entities: dict[str, Any]) -> dict[str, Any]:
        """
        Save a new note to the database.

        Entity priority for the note body:
          1. entities["body"]     — explicit body key
          2. entities["raw_text"] — full verbatim user excerpt
          3. entities["title"]    — last resort (unusual, but safe fallback)

        Entity priority for the note title:
          1. entities["title"]    — explicit short label
          2. Auto-generated:  first 6 words of the body  + "…"

        Args:
            entities: Dict from IntentClassifier.  Expected keys (all optional):
                      "title", "body", "raw_text".

        Returns:
            _success dict with: note_id, title, preview, created_at
            _error   dict with: reason, entities_received
        """
        # --- Resolve body -----------------------------------------------
        body = (
            str(entities.get("body", "")).strip()
            or str(entities.get("raw_text", "")).strip()
            or str(entities.get("title", "")).strip()
        )

        if not body:
            logger.warning("NoteSkill.create: no usable body content in entities.")
            return self._error(
                "I couldn't find any content to save as a note.",
                entities_received=entities,
            )

        # --- Resolve title -----------------------------------------------
        title = str(entities.get("title", "")).strip() or None

        if title is None:
            # Auto-generate a title from the first few words of the body
            words = body.split()
            title = " ".join(words[:6]) + ("…" if len(words) > 6 else "")

        # --- Persist --------------------------------------------------------
        created_at = datetime.now().strftime(self._DT_FMT)

        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    "INSERT INTO notes (title, body, created) VALUES (?, ?, ?)",
                    (title, body, created_at),
                )
                note_id = cursor.lastrowid
                conn.commit()

            preview = body[:self._PREVIEW_LEN] + ("…" if len(body) > self._PREVIEW_LEN else "")
            logger.info(f"NoteSkill: saved note #{note_id} — title={title!r}")

            return self._success({
                "action":     "created",
                "note_id":    note_id,
                "title":      title,
                "preview":    preview,
                "created_at": created_at,
            })

        except sqlite3.Error as exc:
            logger.error(f"NoteSkill.create DB error: {exc}")
            return self._error(f"Database error while saving note: {exc}")

    def list_all(
        self,
        entities: dict[str, Any] | None = None,
        limit: int = 10,
    ) -> dict[str, Any]:
        """
        Retrieve saved notes, newest first.

        If `entities` contains a "title" or "query" key, the query is filtered
        with a case-insensitive LIKE search against both the title and body
        columns — useful for "find my note about shopping".

        Args:
            entities: Optional entity dict from IntentClassifier.
            limit:    Maximum number of notes to return (default 10).

        Returns:
            _success dict with: notes (list of dicts), count, query_used
            _error   dict with: reason
        """
        # --- Optional search filter ---------------------------------------
        search_term: str | None = None
        if entities:
            search_term = (
                str(entities.get("query", "")).strip()
                or str(entities.get("title", "")).strip()
                or None
            )

        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row

                if search_term:
                    pattern = f"%{search_term}%"
                    rows = conn.execute(
                        """
                        SELECT   id, title, body, created
                        FROM     notes
                        WHERE    title LIKE ?
                              OR body  LIKE ?
                        ORDER BY created DESC
                        LIMIT    ?
                        """,
                        (pattern, pattern, limit),
                    ).fetchall()
                else:
                    rows = conn.execute(
                        """
                        SELECT   id, title, body, created
                        FROM     notes
                        ORDER BY created DESC
                        LIMIT    ?
                        """,
                        (limit,),
                    ).fetchall()

            notes = []
            for row in rows:
                body = row["body"]
                notes.append({
                    "id":         row["id"],
                    "title":      row["title"] or "(untitled)",
                    "preview":    body[:self._PREVIEW_LEN] + ("…" if len(body) > self._PREVIEW_LEN else ""),
                    "created_at": row["created"],
                })

            logger.info(
                f"NoteSkill: fetched {len(notes)} note(s)"
                + (f" matching {search_term!r}" if search_term else "")
            )

            return self._success({
                "action":     "listed",
                "notes":      notes,
                "count":      len(notes),
                "query_used": search_term,
            })

        except sqlite3.Error as exc:
            logger.error(f"NoteSkill.list_all DB error: {exc}")
            return self._error(f"Database error while fetching notes: {exc}")

    def delete(self, note_id: int) -> dict[str, Any]:
        """
        Delete a note by its primary key.

        Returns:
            _success dict with: note_id, action="deleted"
            _error   dict with: reason  (includes "not_found" reason if id missing)
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    "DELETE FROM notes WHERE id = ?",
                    (note_id,),
                )
                conn.commit()

            if cursor.rowcount == 0:
                logger.warning(f"NoteSkill.delete: note #{note_id} not found.")
                return self._error(f"No note found with id {note_id}.")

            logger.info(f"NoteSkill: deleted note #{note_id}.")
            return self._success({"action": "deleted", "note_id": note_id})

        except sqlite3.Error as exc:
            logger.error(f"NoteSkill.delete DB error: {exc}")
            return self._error(f"Database error while deleting note: {exc}")
