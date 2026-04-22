"""
memory/database.py — SQLite initialisation and CRUD helpers.

Schema in this version:
  sessions          (id, title, created_at)
  session_documents (id, session_id, filename, content, added_at)
  conversations     (id, role, content, timestamp, session_id)  ← session_id added
  notes             (id, title, body, created)                   unchanged
  reminders         (id, message, remind_at, is_done)            unchanged

Migration strategy
------------------
All CREATE TABLE statements use IF NOT EXISTS — safe on existing databases.
The conversations.session_id column is added via a guarded ALTER TABLE so
it is idempotent on pre-existing databases.
"""

import sqlite3
from loguru import logger


# ---------------------------------------------------------------------------
# init_db
# ---------------------------------------------------------------------------

def init_db(db_path: str) -> None:
    """
    Create all tables if absent and migrate older schemas.
    Safe to call on every startup.
    """
    with sqlite3.connect(db_path) as conn:
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA foreign_keys=ON;")

        conn.executescript("""
            CREATE TABLE IF NOT EXISTS sessions (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                title      TEXT    NOT NULL DEFAULT 'New Chat',
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS session_documents (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
                filename   TEXT    NOT NULL,
                content    TEXT    NOT NULL,
                added_at   DATETIME DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS conversations (
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                role      TEXT    NOT NULL,
                content   TEXT    NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS notes (
                id      INTEGER PRIMARY KEY AUTOINCREMENT,
                title   TEXT,
                body    TEXT NOT NULL,
                created DATETIME DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS reminders (
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                message   TEXT    NOT NULL,
                remind_at DATETIME NOT NULL,
                is_done   INTEGER  DEFAULT 0
            );
        """)

        # Migration: add session_id to conversations if the column is absent
        _add_column_if_missing(
            conn, "conversations", "session_id",
            "INTEGER REFERENCES sessions(id) ON DELETE SET NULL",
        )

    logger.debug(f"Database initialised: {db_path}")


# ---------------------------------------------------------------------------
# Session helpers
# ---------------------------------------------------------------------------

def create_session(db_path: str, title: str = "New Chat") -> int:
    """Insert a new session row and return its id."""
    with sqlite3.connect(db_path) as conn:
        cur = conn.execute("INSERT INTO sessions (title) VALUES (?)", (title,))
        sid = cur.lastrowid
        conn.commit()
    logger.debug(f"Created session #{sid}: {title!r}")
    return sid


def list_sessions(db_path: str) -> list[dict]:
    """Return all sessions, newest first. Each dict has id/title/created_at."""
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT id, title, created_at FROM sessions ORDER BY created_at DESC"
        ).fetchall()
    return [dict(r) for r in rows]


def rename_session(db_path: str, session_id: int, title: str) -> None:
    """Update a session's display title."""
    with sqlite3.connect(db_path) as conn:
        conn.execute("UPDATE sessions SET title=? WHERE id=?", (title, session_id))
        conn.commit()
    logger.debug(f"Renamed session #{session_id} -> {title!r}")


def delete_session(db_path: str, session_id: int) -> None:
    """
    Delete a session and cascade-delete its documents.
    Conversations are kept but their session_id is set to NULL.
    """
    with sqlite3.connect(db_path) as conn:
        conn.execute("PRAGMA foreign_keys=ON;")
        conn.execute("DELETE FROM sessions WHERE id=?", (session_id,))
        conn.commit()
    logger.debug(f"Deleted session #{session_id}")


# ---------------------------------------------------------------------------
# Conversation helpers
# ---------------------------------------------------------------------------

def save_conversation_turn(
    db_path: str, session_id: int, role: str, content: str
) -> None:
    """Persist a single user or assistant turn linked to a session."""
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "INSERT INTO conversations (role, content, session_id) VALUES (?,?,?)",
            (role, content, session_id),
        )
        conn.commit()


def load_conversation_history(
    db_path: str, session_id: int, limit: int = 40
) -> list[dict[str, str]]:
    """
    Fetch the last *limit* turns for *session_id*, oldest-first.
    Returns list of {"role": str, "content": str}.
    """
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT role, content FROM conversations
            WHERE  session_id = ?
            ORDER  BY id DESC
            LIMIT  ?
            """,
            (session_id, limit),
        ).fetchall()
    return [{"role": r["role"], "content": r["content"]} for r in reversed(rows)]


# ---------------------------------------------------------------------------
# Document helpers
# ---------------------------------------------------------------------------

def save_document(
    db_path: str, session_id: int, filename: str, content: str
) -> int:
    """Persist extracted document text linked to session. Returns new doc id."""
    with sqlite3.connect(db_path) as conn:
        cur = conn.execute(
            "INSERT INTO session_documents (session_id, filename, content) VALUES (?,?,?)",
            (session_id, filename, content),
        )
        doc_id = cur.lastrowid
        conn.commit()
    logger.debug(f"Saved doc #{doc_id} '{filename}' ({len(content)} chars) -> session #{session_id}")
    return doc_id


def load_session_documents(db_path: str, session_id: int) -> list[dict]:
    """Return all documents for session_id. Each dict has id/filename/content/added_at."""
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT id, filename, content, added_at FROM session_documents WHERE session_id=? ORDER BY added_at",
            (session_id,),
        ).fetchall()
    return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# Internal migration helper
# ---------------------------------------------------------------------------

def _add_column_if_missing(
    conn: sqlite3.Connection, table: str, column: str, definition: str
) -> None:
    """Add column to table only if it is absent. Idempotent."""
    existing = {row[1] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}
    if column not in existing:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")
        conn.commit()
        logger.debug(f"Migration: added column '{column}' to '{table}'")
