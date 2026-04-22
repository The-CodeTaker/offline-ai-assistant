"""
skills/base_skill.py — Abstract Base Class that every skill must inherit from.

CONTRACT
--------
Every concrete skill in this project must:

1.  Accept `db_path: str` as its first constructor argument and store it as
    `self.db_path`.  Skills that need no database may still receive the path;
    they simply ignore it.

2.  Expose an `execute(entities: dict[str, Any]) -> dict[str, Any]` method
    as the single canonical entry-point called by `assistant._dispatch_skill`.
    All skill-specific logic (create / list / search) is routed through this
    method based on the entities dict or via explicit sub-methods that the
    assistant calls directly (e.g. `NoteSkill.create`, `NoteSkill.list_all`).

3.  Always return a plain, JSON-serialisable dict with at minimum a `"status"`
    key set to `"success"` or `"error"`.  This contract guarantees that
    `DialogueGenerator._build_user_content` can always safely serialise the
    return value and that the LLM always receives grounded task data.

4.  Never raise exceptions to the caller.  Every error must be caught
    internally and returned as `{"status": "error", "reason": "<message>"}`.

WHY AN ABC AND NOT JUST A PROTOCOL?
------------------------------------
`abc.ABC` gives us:
- Immediate, clear `TypeError` at import time if a subclass forgets to
  implement a required method — caught in development, not at runtime.
- A concrete `__init__` on the base that stores `db_path`, removing the
  boilerplate from every subclass.
- A place to put shared helpers (e.g. `_error`, `_success`) so all skills
  produce identically-shaped output dicts without copy-pasting.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseSkill(ABC):
    """
    Abstract base class for all assistant skills.

    Subclasses must implement `execute()`.  They may also expose additional
    named methods (e.g. `create`, `list_all`) for the assistant to call
    directly when the intent is unambiguous.

    Shared helpers `_success()` and `_error()` are provided so every skill
    returns consistently-shaped dicts without duplicating boilerplate.
    """

    def __init__(self, db_path: str) -> None:
        """
        Args:
            db_path: Absolute path to the SQLite database file.
                     Passed in from `config.DB_PATH` via the Assistant.
        """
        self.db_path: str = db_path

    # ------------------------------------------------------------------
    # Abstract interface — subclasses MUST implement this
    # ------------------------------------------------------------------

    @abstractmethod
    def execute(self, entities: dict[str, Any]) -> dict[str, Any]:
        """
        Primary entry-point for the skill.

        Called by `assistant._dispatch_skill` when a single dispatch call
        is sufficient (i.e. the intent already encodes the action).
        For intents with distinct sub-actions (create vs. list), the
        assistant may call named methods directly instead.

        Args:
            entities: The entities dict extracted by IntentClassifier.
                      May be empty but is never None.

        Returns:
            A plain dict with at minimum {"status": "success"|"error"}.
            Must be JSON-serialisable (no datetime objects, no DB Row types).
        """
        ...

    # ------------------------------------------------------------------
    # Shared return-value helpers — use these in every subclass
    # ------------------------------------------------------------------

    @staticmethod
    def _success(data: dict[str, Any]) -> dict[str, Any]:
        """
        Wrap *data* in a success envelope.

        Usage:
            return self._success({"note_id": 3, "title": "Shopping list"})
        Returns:
            {"status": "success", "note_id": 3, "title": "Shopping list"}
        """
        return {"status": "success", **data}

    @staticmethod
    def _error(reason: str, **extra: Any) -> dict[str, Any]:
        """
        Return a standardised error dict.

        Usage:
            return self._error("Title is required.", entities_received=entities)
        Returns:
            {"status": "error", "reason": "Title is required.",
             "entities_received": {...}}
        """
        return {"status": "error", "reason": reason, **extra}
