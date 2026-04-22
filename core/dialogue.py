"""
core/dialogue.py — Natural language response generation via a local Ollama model.

The DialogueGenerator takes:
  - The user's original message
  - An IntentResult (from intent.py)
  - Optional task_data: skill results, RAG context, or API responses

And produces a warm, concise, conversational reply.

System prompt design (this version)
-------------------------------------
  The previous prompt injected the current date/time prominently at the
  top, which caused the LLM to regurgitate it in almost every response
  ("As of today, Wednesday 22 April 2026 at 14:35…").

  The new prompt:
    1. Keeps date/time as a HIDDEN INTERNAL REFERENCE labelled
       [INTERNAL — do not mention unless asked], so skills like reminders
       can still format human-readable dates correctly.
    2. Adds an explicit hard rule: NEVER volunteer the date or time in a
       conversational reply unless the user specifically asks for it.
    3. Elevates the priority of <task_data> — when RAG context is present
       the LLM is explicitly told it MUST ground its answer in that context
       and cite it as its source, not its own "general knowledge".
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
# System prompt
# ---------------------------------------------------------------------------

def _build_system_prompt() -> str:
    """
    Build the dialogue system prompt with a hidden temporal anchor.

    The date/time is injected so that SKILL replies (reminders, schedules)
    can format dates correctly, but the model is explicitly instructed never
    to surface it in general conversation.
    """
    now     = _dt.now()
    today   = now.strftime("%Y-%m-%d")
    weekday = now.strftime("%A")
    time_   = now.strftime("%H:%M")

    return f"""You are a helpful, friendly, and concise personal AI assistant \
running entirely on the user's local device.

━━━ INTERNAL REFERENCE — DO NOT MENTION IN REPLIES ━━━
Current date : {today} ({weekday})
Current time : {time_}
Use the above ONLY when a skill result references dates/times and you need
to format them for the user (e.g. "your reminder is set for tomorrow, {weekday}").
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

PERSONALITY RULES (always follow these):
1. Be warm, direct, and concise — prefer 1–3 sentences unless detail is needed.
2. NEVER open a reply by stating the current date, time, or day of the week.
   Do not mention them at all unless the user has explicitly asked "what time is it?"
   or "what day is it?" or a similarly direct temporal question.
3. Never say "As of today…", "Currently…", "At the moment…" followed by a date.
4. If you don't know something, say so plainly — do not fabricate.
5. Never identify yourself as an AI or language model unless directly asked.

TASK DATA RULES (highest priority when <task_data> is present):
6. When a <task_data> block is provided, you MUST base your answer
   primarily on the information inside it.
7. If <task_data> contains a "retrieved_context" field, treat it as the
   authoritative source for the user's question. Answer directly from it.
   Do not contradict or ignore it. You may say "Based on the document…"
   to signal you are using it.
8. If <task_data> is absent or empty, answer conversationally from your
   own knowledge.
9. Do NOT invent facts that are not present in <task_data>.
"""


# Template for injecting structured skill / RAG results into the user turn
_TASK_DATA_TEMPLATE = """<task_data>
{task_json}
</task_data>

User message: {user_message}"""


# ---------------------------------------------------------------------------
# DialogueGenerator
# ---------------------------------------------------------------------------

class DialogueGenerator:
    """
    Generates a natural conversational response given a user message,
    the classified intent, and optional executed-task / RAG data.

    Usage:
        gen = DialogueGenerator()
        reply = gen.generate(
            user_message="What does the contract say about payment?",
            intent_result=intent_result,
            task_data={"retrieved_context": "...relevant PDF excerpt..."}
        )

    Streaming:
        for chunk in gen.generate_stream(user_message, intent_result, task_data):
            print(chunk, end="", flush=True)
    """

    def __init__(
        self,
        model:          str   = config.MODEL_NAME,
        temperature:    float = 0.7,
        max_tokens:     int   = 512,
        history_window: int   = 6,
    ) -> None:
        self.model          = model
        self.temperature    = temperature
        self.max_tokens     = max_tokens
        self.history_window = history_window
        logger.debug(f"DialogueGenerator ready (model={self.model}, temp={self.temperature})")

    # ------------------------------------------------------------------
    # Public API — blocking
    # ------------------------------------------------------------------

    def generate(
        self,
        user_message:  str,
        intent_result: IntentResult | None          = None,
        task_data:     Any                          = None,
        history:       list[dict[str, str]] | None  = None,
    ) -> str:
        """
        Generate and return the full response as a string.

        Args:
            user_message : The raw text the user typed/spoke.
            intent_result: Classified intent.
            task_data    : Skill result or RAG context dict. Serialised to
                           JSON and injected inside <task_data> tags.
            history      : Previous turns as {"role": ..., "content": ...}.
        """
        messages = self._build_messages(user_message, intent_result, task_data, history)

        try:
            response = ollama.chat(
                model=self.model,
                options={"temperature": self.temperature, "num_predict": self.max_tokens},
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
        user_message:  str,
        intent_result: IntentResult | None          = None,
        task_data:     Any                          = None,
        history:       list[dict[str, str]] | None  = None,
    ) -> Generator[str, None, None]:
        """
        Stream the response token-by-token.

        Yields str chunks as they arrive; falls back to a full generate()
        call if streaming fails.
        """
        messages = self._build_messages(user_message, intent_result, task_data, history)

        try:
            stream = ollama.chat(
                model=self.model,
                options={"temperature": self.temperature, "num_predict": self.max_tokens},
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
        user_message:  str,
        intent_result: IntentResult | None,
        task_data:     Any,
        history:       list[dict[str, str]] | None,
    ) -> list[dict[str, str]]:
        """
        Assemble the full messages list:
          [system] + [history (windowed)] + [user turn w/ optional task_data]
        """
        messages: list[dict[str, str]] = [
            {"role": "system", "content": _build_system_prompt()},
        ]

        if history:
            windowed = history[-(self.history_window * 2):]
            messages.extend(windowed)

        user_content = self._build_user_content(user_message, intent_result, task_data)
        messages.append({"role": "user", "content": user_content})

        logger.debug(f"Built message list: {len(messages)} messages")
        return messages

    @staticmethod
    def _build_user_content(
        user_message:  str,
        intent_result: IntentResult | None,
        task_data:     Any,
    ) -> str:
        """
        If task_data exists, wrap it in a <task_data> block.
        The block may contain RAG retrieved_context, skill results, or both.
        """
        if task_data is None:
            return user_message

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
