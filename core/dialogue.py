"""
core/dialogue.py — Natural language response generation via a local Ollama model.

The DialogueGenerator takes:
  - The user's original message
  - An IntentResult (from intent.py)
  - Optional task_data: the result of a skill execution (DB rows, API response, etc.)

And produces a warm, concise, conversational reply.

Design decisions:
  - A short system persona prompt keeps tone consistent without burning tokens.
  - task_data is serialised to a compact JSON block and injected as context
    so the model can reference real data rather than hallucinating answers.
  - Streaming is supported via generate_stream() for a live-typing UX.
  - Conversation history is accepted as an optional parameter so the model
    can maintain short-term coherence across turns without a separate memory
    module (just pass the last N turns from your DB).
  - All Ollama errors are caught; a graceful fallback string is returned so
    the calling loop never crashes on a model hiccup.
"""

from __future__ import annotations

import json
from collections.abc import Generator
from datetime import datetime as _dt
from typing import Any

import ollama
from loguru import logger

import config
from core.intent import IntentResult


# ---------------------------------------------------------------------------
# Persona / system prompt
# ---------------------------------------------------------------------------

def _build_system_prompt() -> str:
    """
    Build the dialogue system prompt with the current date/time baked in.
    Called fresh for every generate() call so the model always has an accurate
    temporal anchor — critical for phrasing like "your reminder for tomorrow".
    """
    now     = _dt.now()
    today   = now.strftime("%Y-%m-%d")    # e.g. "2026-03-31"
    weekday = now.strftime("%A")          # e.g. "Tuesday"
    time_   = now.strftime("%H:%M")       # e.g. "14:35"

    return f"""You are a helpful, friendly, and concise personal AI assistant \
running entirely on the user's local device — no internet required.

CURRENT DATE & TIME (use when phrasing time-sensitive replies):
  Today   : {today} ({weekday})
  Time now: {time_}

Your personality:
- Warm and direct — never cold or robotic.
- Concise — prefer 1–3 sentences unless more detail is clearly needed.
- Honest — if you don't know something, say so plainly.
- Never mention that you are an AI or a language model unless asked directly.
- When referencing saved dates, always confirm with the human-readable form
  (e.g. "Wednesday, 02 April 2026 at 15:00") so the user can verify.

When task data is provided between <task_data> tags, base your answer on that
data. Do not invent facts that are not present in the task data.
If the task data is empty or absent, answer from your own knowledge conversationally.
"""

# Template for injecting structured skill results into the user turn
_TASK_DATA_TEMPLATE = """<task_data>
{task_json}
</task_data>

User message: {user_message}"""


# ---------------------------------------------------------------------------
# Dialogue generator
# ---------------------------------------------------------------------------

class DialogueGenerator:
    """
    Generates a natural conversational response given a user message,
    the classified intent, and optional executed-task data.

    Basic usage:
        gen = DialogueGenerator()

        # After a skill has run and returned data:
        reply = gen.generate(
            user_message="Remind me to call Alice at 3pm",
            intent_result=intent_result,
            task_data={"status": "created", "reminder_id": 7}
        )

    Streaming usage (for a typing effect in the UI):
        for chunk in gen.generate_stream(user_message, intent_result):
            print(chunk, end="", flush=True)
    """

    def __init__(
        self,
        model: str = config.MODEL_NAME,
        temperature: float = 0.7,
        max_tokens: int = 512,
        history_window: int = 6,
    ) -> None:
        """
        Args:
            model         : Ollama model identifier (from config).
            temperature   : Sampling temperature — 0.7 balances creativity/focus.
            max_tokens    : Hard cap on response length.
            history_window: Max number of previous (user + assistant) turns to
                            include for context continuity.
        """
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.history_window = history_window
        logger.debug(
            f"DialogueGenerator ready "
            f"(model={self.model}, temp={self.temperature})"
        )

    # ------------------------------------------------------------------
    # Public API — blocking
    # ------------------------------------------------------------------

    def generate(
        self,
        user_message: str,
        intent_result: IntentResult | None = None,
        task_data: Any = None,
        history: list[dict[str, str]] | None = None,
    ) -> str:
        """
        Generate and return the full response as a string.

        Args:
            user_message : The raw text the user typed/spoke.
            intent_result: Classified intent (used for optional meta context).
            task_data    : Structured data from skill execution. Can be a dict,
                           list, string, or None. Serialised to JSON if needed.
            history      : List of previous turns as {"role": ..., "content": ...}
                           dicts. Pass the last N turns from your conversations DB.
        Returns:
            A natural language string ready to be spoken or displayed.
        """
        messages = self._build_messages(user_message, intent_result, task_data, history)

        try:
            response = ollama.chat(
                model=self.model,
                options={
                    "temperature": self.temperature,
                    "num_predict": self.max_tokens,
                },
                messages=messages,
            )
            reply = response.message.content.strip()
            logger.debug(f"Generated reply ({len(reply)} chars)")
            return reply

        except ollama.ResponseError as exc:
            logger.error(f"Ollama ResponseError during dialogue generation: {exc}")
            return self._fallback(intent_result)
        except Exception as exc:
            logger.error(f"Unexpected error during dialogue generation: {exc}")
            return self._fallback(intent_result)

    # ------------------------------------------------------------------
    # Public API — streaming
    # ------------------------------------------------------------------

    def generate_stream(
        self,
        user_message: str,
        intent_result: IntentResult | None = None,
        task_data: Any = None,
        history: list[dict[str, str]] | None = None,
    ) -> Generator[str, None, None]:
        """
        Stream the response token-by-token.

        Yields string chunks as they arrive from Ollama.
        Example usage in main loop:
            for chunk in gen.generate_stream(user_input, result):
                print(chunk, end="", flush=True)
            print()  # final newline

        Falls back to yielding the full string from generate() on error.
        """
        messages = self._build_messages(user_message, intent_result, task_data, history)

        try:
            stream = ollama.chat(
                model=self.model,
                options={
                    "temperature": self.temperature,
                    "num_predict": self.max_tokens,
                },
                messages=messages,
                stream=True,
            )
            for chunk in stream:
                token = chunk.message.content
                if token:
                    yield token

        except ollama.ResponseError as exc:
            logger.error(f"Ollama stream ResponseError: {exc}")
            yield self._fallback(intent_result)
        except Exception as exc:
            logger.error(f"Unexpected stream error: {exc}")
            yield self._fallback(intent_result)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_messages(
        self,
        user_message: str,
        intent_result: IntentResult | None,
        task_data: Any,
        history: list[dict[str, str]] | None,
    ) -> list[dict[str, str]]:
        """
        Assemble the full messages list to send to Ollama:
          [system] + [history (windowed)] + [user turn (with optional task data)]
        """
        messages: list[dict[str, str]] = [
            {"role": "system", "content": _build_system_prompt()},
        ]

        # Inject windowed conversation history for coherence
        if history:
            windowed = history[-(self.history_window * 2):]  # 2 messages per turn
            messages.extend(windowed)

        # Build the user turn — inject task_data if present
        user_content = self._build_user_content(user_message, intent_result, task_data)
        messages.append({"role": "user", "content": user_content})

        logger.debug(f"Built message list: {len(messages)} messages")
        return messages

    @staticmethod
    def _build_user_content(
        user_message: str,
        intent_result: IntentResult | None,
        task_data: Any,
    ) -> str:
        """
        If task_data exists, wrap it in a <task_data> block so the model
        can ground its response in real results rather than hallucinating.
        """
        if task_data is None:
            return user_message

        # Serialise task_data gracefully
        if isinstance(task_data, str):
            task_json = task_data
        else:
            try:
                task_json = json.dumps(task_data, indent=2, default=str)
            except (TypeError, ValueError):
                task_json = str(task_data)

        return _TASK_DATA_TEMPLATE.format(
            task_json=task_json,
            user_message=user_message,
        )

    @staticmethod
    def _fallback(intent_result: IntentResult | None) -> str:
        """Return a graceful degradation message on model failure."""
        if intent_result and intent_result.intent == "greeting":
            return "Hello! How can I help you today?"
        return (
            "I'm sorry, I ran into an issue generating a response. "
            "Please try again in a moment."
        )
