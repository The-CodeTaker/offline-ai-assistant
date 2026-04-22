"""
skills/reminder.py — ReminderSkill: create and retrieve reminders in SQLite.

Responsibilities:
  - Parse entity dicts produced by IntentClassifier (date, time, title, raw_text)
    into a reliable datetime, handling ambiguous / missing fields gracefully.
  - Write a new row to the `reminders` table via a safe parameterised query.
  - Fetch upcoming (undone) reminders for display.
  - Always return a structured dict that DialogueGenerator can use as task_data.

Design decisions:
  - Zero third-party date-parsing deps: uses only stdlib `datetime` + a small
    set of fuzzy-resolution helpers so the skill works fully offline.
  - "Best-effort" datetime: if the model only extracted a time (no date) we
    assume today; if only a date (no time) we default to 09:00. A clear
    `parsed_reminder_at` key is always echoed back in the return dict so
    DialogueGenerator can tell the user exactly what was saved.
  - Reminders are never silently dropped: every failure path returns a dict
    with status='error' and a human-readable reason.
  - `list_upcoming` returns rows as plain dicts (not Row objects) so the
    result is directly JSON-serialisable for DialogueGenerator.
"""

from __future__ import annotations

import sqlite3
from datetime import date, datetime, time, timedelta
from typing import Any

from loguru import logger


# ---------------------------------------------------------------------------
# Date / time parsing helpers
# ---------------------------------------------------------------------------

_DATE_FORMATS = ["%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y", "%m/%d/%Y", "%B %d %Y", "%b %d %Y"]
_TIME_FORMATS = ["%H:%M", "%I:%M %p", "%I%p", "%H%M"]

# Fuzzy relative-date keywords the model often emits instead of ISO strings
_RELATIVE_DATES: dict[str, int] = {
    "today":         0,
    "tomorrow":      1,
    "day after tomorrow": 2,
    "overmorrow":    2,
    "next monday":   None,  # handled specially below
    "next tuesday":  None,
    "next wednesday":None,
    "next thursday": None,
    "next friday":   None,
    "next saturday": None,
    "next sunday":   None,
}

_WEEKDAY_NAMES = ["monday","tuesday","wednesday","thursday","friday","saturday","sunday"]


def _resolve_date(raw: str | None) -> date | None:
    """
    Turn a string like '2025-07-04', 'tomorrow', 'next Friday', or None
    into a `datetime.date`.  Returns None if resolution fails.
    """
    if not raw:
        return None

    normalised = raw.strip().lower()

    # Relative keywords
    if normalised in _RELATIVE_DATES:
        delta = _RELATIVE_DATES[normalised]
        if delta is not None:
            return date.today() + timedelta(days=delta)

    # "next <weekday>"
    for i, day_name in enumerate(_WEEKDAY_NAMES):
        if normalised == f"next {day_name}":
            today_wd = date.today().weekday()   # Monday=0
            days_ahead = (i - today_wd + 7) % 7 or 7
            return date.today() + timedelta(days=days_ahead)

    # Bare weekday name e.g. "friday" (nearest future)
    if normalised in _WEEKDAY_NAMES:
        i = _WEEKDAY_NAMES.index(normalised)
        today_wd = date.today().weekday()
        days_ahead = (i - today_wd) % 7 or 7
        return date.today() + timedelta(days=days_ahead)

    # ISO or formatted date strings
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(raw.strip(), fmt).date()
        except ValueError:
            continue

    logger.debug(f"ReminderSkill: could not resolve date from {raw!r}")
    return None


def _resolve_time(raw: str | None) -> time | None:
    """
    Turn '15:00', '3pm', '3:00 PM', etc. into a `datetime.time`.
    Returns None if resolution fails.
    """
    if not raw:
        return None

    cleaned = raw.strip().upper().replace(".", "").replace(" ", "")
    for fmt in [f.replace(" ", "") for f in _TIME_FORMATS]:
        try:
            return datetime.strptime(cleaned, fmt.upper()).time()
        except ValueError:
            continue

    logger.debug(f"ReminderSkill: could not resolve time from {raw!r}")
    return None


def _build_remind_at(entities: dict[str, Any]) -> datetime | None:
    """
    Compose a full datetime from whatever date/time entities the model found.
    Fallback rules:
      - date missing → use today
      - time missing → default to 09:00
      - datetime key present → try parsing it directly first
    """
    # Try the combined datetime key first
    if "datetime" in entities:
        for fmt in ["%Y-%m-%dT%H:%M", "%Y-%m-%d %H:%M", "%Y-%m-%dT%H:%M:%S"]:
            try:
                return datetime.strptime(str(entities["datetime"]), fmt)
            except ValueError:
                continue

    resolved_date = _resolve_date(entities.get("date")) or date.today()
    resolved_time = _resolve_time(entities.get("time")) or time(9, 0)

    return datetime.combine(resolved_date, resolved_time)


# ---------------------------------------------------------------------------
# ReminderSkill
# ---------------------------------------------------------------------------

class ReminderSkill:
    """
    Manages reminder CRUD against the `reminders` SQLite table.

    The table schema (from memory/database.py):
        id          INTEGER PRIMARY KEY AUTOINCREMENT
        message     TEXT    NOT NULL
        remind_at   DATETIME NOT NULL
        is_done     INTEGER  DEFAULT 0

    Usage:
        skill = ReminderSkill(db_path)
        result = skill.create(intent_result.entities)
        # result → {"status": "success", "task": "call John", "time": "17:00", ...}
    """

    # ISO format stored in SQLite
    _DT_FMT = "%Y-%m-%d %H:%M"

    def __init__(self, db_path: str) -> None:
        self.db_path = db_path

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def create(self, entities: dict[str, Any]) -> dict[str, Any]:
        """
        Parse *entities* from IntentClassifier and insert a new reminder row.

        Args:
            entities: dict with any subset of keys:
                      title, date, time, datetime, raw_text, duration

        Returns:
            Structured dict consumed by DialogueGenerator as task_data.
        """
        # Derive the reminder message / title
        message = (
            entities.get("title")
            or entities.get("raw_text")
            or "Reminder"
        )
        message = str(message).strip()

        # Resolve remind_at datetime
        remind_at = _build_remind_at(entities)
        if remind_at is None:
            logger.warning("ReminderSkill.create: could not resolve datetime from entities.")
            return {
                "status":  "error",
                "reason":  "Could not determine a date or time for the reminder.",
                "entities_received": entities,
            }

        remind_at_str = remind_at.strftime(self._DT_FMT)

        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    "INSERT INTO reminders (message, remind_at, is_done) VALUES (?, ?, 0)",
                    (message, remind_at_str),
                )
                new_id = cursor.lastrowid
                conn.commit()

            logger.info(f"ReminderSkill: saved reminder #{new_id} — '{message}' at {remind_at_str}")

            return {
                "status":            "success",
                "message":           "Reminder saved successfully.",
                "reminder_id":       new_id,
                "task":              message,
                "date":              remind_at.strftime("%A, %d %B %Y"),  # e.g. "Friday, 04 July 2025"
                "time":              remind_at.strftime("%H:%M"),          # e.g. "17:00"
                "parsed_reminder_at": remind_at_str,
            }

        except sqlite3.Error as exc:
            logger.error(f"ReminderSkill.create DB error: {exc}")
            return {
                "status": "error",
                "reason": f"Database error: {exc}",
            }

    def list_upcoming(
        self,
        entities: dict[str, Any] | None = None,
        limit: int = 10,
    ) -> dict[str, Any]:
        """
        Return upcoming undone reminders ordered by remind_at.

        If *entities* contains a resolvable date (e.g. the user asked "what
        are my reminders for tomorrow?"), only reminders falling on that
        specific calendar day are returned.  Without a date entity the method
        returns the next *limit* reminders from right now onwards.

        Args:
            entities: Optional entity dict from IntentClassifier.  Inspected
                      for the keys "date", "datetime", and relative strings
                      like "tomorrow" so the query is scoped to one day.
            limit:    Maximum number of rows to return (default 10).

        Returns:
            {
              "status":       "success" | "error",
              "reminders":    [ {id, task, date, time, remind_at}, ... ],
              "count":        N,
              "filtered_date": "YYYY-MM-DD"  # only present when a date filter was applied
            }
        """
        now = datetime.now()
        target_date: date | None = None

        # --- Resolve a target date from entities if one was provided ----------
        if entities:
            target_date = _resolve_date(entities.get("date") or entities.get("datetime"))
            logger.debug(
                f"ReminderSkill.list_upcoming: entity date={entities.get('date')!r} "
                f"→ resolved target_date={target_date}"
            )

        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row

                if target_date is not None:
                    # Scope query to the single target calendar day
                    day_start = datetime.combine(target_date, time(0, 0)).strftime(self._DT_FMT)
                    day_end   = datetime.combine(target_date, time(23, 59)).strftime(self._DT_FMT)
                    rows = conn.execute(
                        """
                        SELECT id, message, remind_at
                        FROM   reminders
                        WHERE  is_done = 0
                          AND  remind_at BETWEEN ? AND ?
                        ORDER  BY remind_at ASC
                        LIMIT  ?
                        """,
                        (day_start, day_end, limit),
                    ).fetchall()
                else:
                    # Default: all upcoming reminders from now
                    rows = conn.execute(
                        """
                        SELECT id, message, remind_at
                        FROM   reminders
                        WHERE  is_done = 0
                          AND  remind_at >= ?
                        ORDER  BY remind_at ASC
                        LIMIT  ?
                        """,
                        (now.strftime(self._DT_FMT), limit),
                    ).fetchall()

            reminders = []
            for row in rows:
                dt = datetime.strptime(row["remind_at"], self._DT_FMT)
                reminders.append({
                    "id":        row["id"],
                    "task":      row["message"],
                    "date":      dt.strftime("%A, %d %B %Y"),
                    "time":      dt.strftime("%H:%M"),
                    "remind_at": row["remind_at"],
                })

            logger.info(
                f"ReminderSkill: fetched {len(reminders)} reminder(s)"
                + (f" for {target_date}" if target_date else " (upcoming)")
            )

            result: dict[str, Any] = {
                "status":    "success",
                "reminders": reminders,
                "count":     len(reminders),
            }
            if target_date is not None:
                result["filtered_date"] = target_date.isoformat()

            return result

        except sqlite3.Error as exc:
            logger.error(f"ReminderSkill.list_upcoming DB error: {exc}")
            return {
                "status":    "error",
                "reason":    f"Database error: {exc}",
                "reminders": [],
            }

    def mark_done(self, reminder_id: int) -> dict[str, Any]:
        """
        Mark a reminder as completed.

        Returns:
            {"status": "success"|"not_found"|"error", ...}
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    "UPDATE reminders SET is_done = 1 WHERE id = ?",
                    (reminder_id,),
                )
                conn.commit()

            if cursor.rowcount == 0:
                return {"status": "not_found", "reminder_id": reminder_id}

            logger.info(f"ReminderSkill: marked reminder #{reminder_id} as done.")
            return {"status": "success", "reminder_id": reminder_id}

        except sqlite3.Error as exc:
            logger.error(f"ReminderSkill.mark_done DB error: {exc}")
            return {"status": "error", "reason": str(exc)}
