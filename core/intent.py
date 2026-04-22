"""
core/intent.py — Intent classification via a local Ollama model.

The IntentClassifier sends a tightly-constrained prompt to LLaMA and parses
the response into a structured IntentResult dataclass containing:
  - intent  : one of a fixed set of intent labels
  - entities: extracted key entities (dates, times, locations, travel endpoints)
  - confidence: self-reported model confidence (0.0-1.0)

Key changes in this version:
  - System prompt now contains dedicated WEATHER and TRAVEL/NAVIGATION entity
    sections with aggressive extraction rules for "location", "origin", and
    "destination".
  - Added 7 new few-shot examples covering weather + travel edge cases.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

import ollama
from loguru import logger

import config


# ---------------------------------------------------------------------------
# Supported intent labels
# ---------------------------------------------------------------------------

INTENT_LABELS: list[str] = [
    "greeting",
    "farewell",
    "set_reminder",
    "get_reminders",
    "create_note",
    "get_notes",
    "get_schedule",
    "search_travel",
    "get_weather",
    "general_chat",
    "unknown",
]

_LABELS_INLINE = ", ".join(f'"{l}"' for l in INTENT_LABELS)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class IntentResult:
    intent: str = "unknown"
    entities: dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    raw_response: str = ""

    def is_known(self) -> bool:
        return self.intent != "unknown"

    def __repr__(self) -> str:
        return (
            f"IntentResult(intent={self.intent!r}, "
            f"confidence={self.confidence:.2f}, entities={self.entities})"
        )


# ---------------------------------------------------------------------------
# System prompt builder
# ---------------------------------------------------------------------------

def _build_system_prompt() -> str:
    """
    Build the classifier system prompt with current date/time stamped in.
    Called fresh on every classify() so relative dates are always correct.
    """
    from datetime import datetime as _dt, timedelta as _td
    now      = _dt.now()
    today    = now.strftime("%Y-%m-%d")
    weekday  = now.strftime("%A")
    cur_time = now.strftime("%H:%M")
    tomorrow = (now.date() + _td(days=1)).isoformat()

    return f"""You are a strict intent-classification engine.
Your ONLY job is to analyse a user message and return a single JSON object.

CURRENT DATE & TIME (use to resolve relative references like "today", "tomorrow", "next Friday"):
  Today    : {today} ({weekday})
  Time now : {cur_time}
  Tomorrow : {tomorrow}

RULES (never break these):
1. Reply with ONLY a raw JSON object - no markdown, no backticks, no prose.
2. The JSON must have exactly three keys: "intent", "entities", "confidence".
3. "intent" must be one of: {_LABELS_INLINE}.
4. "entities" is an object. Extract whichever of these keys are clearly present:

   GENERAL:
     - "date"        : ISO-8601 date string e.g. "{today}"
     - "time"        : 24-h time string e.g. "14:30"
     - "datetime"    : combined e.g. "{today}T14:30"
     - "duration"    : human string e.g. "2 hours"
     - "title"       : short label / subject e.g. "dentist appointment"
     - "raw_text"    : verbatim relevant excerpt from the user message

   WEATHER (intent = get_weather) - AGGRESSIVELY extract:
     - "location"    : the place the user wants weather for, as a clean place
                       name string e.g. "Mumbai", "New York", "Paris, France".
                       Strip filler words ("weather in", "temperature at", etc.)
                       and emit only the bare place name.

   TRAVEL / NAVIGATION (intent = search_travel) - AGGRESSIVELY extract:
     - "origin"      : the departure place e.g. "New Delhi"
     - "destination" : the arrival place e.g. "Agra"
     - "location"    : ONLY if origin/destination cannot be separated, emit
                       the combined string e.g. "Delhi to Agra"
     Never emit "origin" or "destination" as vague phrases like "here" -
     always extract the actual named place.

5. "confidence" is a float 0.0-1.0.
6. Handle minor phonetic typos gracefully (e.g. "nodes" means "notes").
7. If the user asks for "all" of something, do NOT extract a date.
8. Always resolve relative dates to ISO-8601 using CURRENT DATE above.

EXAMPLES
--------
User: "Hey there!"
Output: {{"intent": "greeting", "entities": {{}}, "confidence": 0.99}}

User: "Remind me to call Alice tomorrow at 3pm"
Output: {{
  "intent": "set_reminder",
  "entities": {{
    "title": "call Alice",
    "date": "{tomorrow}",
    "time": "15:00",
    "raw_text": "call Alice tomorrow at 3pm"
  }},
  "confidence": 0.97
}}

User: "What are my reminders for tomorrow?"
Output: {{
  "intent": "get_reminders",
  "entities": {{
    "date": "{tomorrow}",
    "raw_text": "reminders for tomorrow"
  }},
  "confidence": 0.98
}}

User: "Take a note that I need to buy milk, bread, and eggs"
Output: {{
  "intent": "create_note",
  "entities": {{
    "title": "Grocery list",
    "raw_text": "buy milk, bread, and eggs"
  }},
  "confidence": 0.98
}}

User: "What are my saved notes?"
Output: {{
  "intent": "get_notes",
  "entities": {{"raw_text": "saved notes"}},
  "confidence": 0.98
}}

User: "what are all the nodes I have"
Output: {{
  "intent": "get_notes",
  "entities": {{"raw_text": "all the nodes"}},
  "confidence": 0.85
}}

User: "What's the weather like in Mumbai today?"
Output: {{
  "intent": "get_weather",
  "entities": {{
    "location": "Mumbai",
    "raw_text": "weather in Mumbai today"
  }},
  "confidence": 0.98
}}

User: "Tell me the temperature in New York"
Output: {{
  "intent": "get_weather",
  "entities": {{
    "location": "New York",
    "raw_text": "temperature in New York"
  }},
  "confidence": 0.97
}}

User: "Is it going to rain in London tomorrow?"
Output: {{
  "intent": "get_weather",
  "entities": {{
    "location": "London",
    "date": "{tomorrow}",
    "raw_text": "rain in London tomorrow"
  }},
  "confidence": 0.95
}}

User: "How do I get from Delhi to Agra?"
Output: {{
  "intent": "search_travel",
  "entities": {{
    "origin": "Delhi",
    "destination": "Agra",
    "raw_text": "from Delhi to Agra"
  }},
  "confidence": 0.98
}}

User: "What is the driving distance between Mumbai and Pune?"
Output: {{
  "intent": "search_travel",
  "entities": {{
    "origin": "Mumbai",
    "destination": "Pune",
    "raw_text": "driving distance between Mumbai and Pune"
  }},
  "confidence": 0.97
}}

User: "Navigate from London Heathrow to Oxford"
Output: {{
  "intent": "search_travel",
  "entities": {{
    "origin": "London Heathrow",
    "destination": "Oxford",
    "raw_text": "from London Heathrow to Oxford"
  }},
  "confidence": 0.99
}}

User: "What flights are there from Delhi to Goa next Friday?"
Output: {{
  "intent": "search_travel",
  "entities": {{
    "origin": "Delhi",
    "destination": "Goa",
    "date": "<resolved ISO date for next Friday>",
    "raw_text": "flights from Delhi to Goa next Friday"
  }},
  "confidence": 0.95
}}
"""


_REPAIR_SUFFIX = (
    "\n\nYour previous response could not be parsed as JSON. "
    "Reply with ONLY the raw JSON object - no explanation, no backticks."
)


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------

class IntentClassifier:
    """
    Classifies a user utterance into a structured IntentResult.

    Usage:
        classifier = IntentClassifier()
        result = classifier.classify("What's the weather in Paris?")
        print(result.intent)    # "get_weather"
        print(result.entities)  # {"location": "Paris", ...}
    """

    def __init__(
        self,
        model: str = config.MODEL_NAME,
        temperature: float = 0.0,
        max_repair_attempts: int = 1,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.max_repair_attempts = max_repair_attempts
        logger.debug(f"IntentClassifier ready (model={self.model})")

    def classify(self, user_text: str) -> IntentResult:
        """Classify user_text and return an IntentResult. Never raises."""
        if not user_text or not user_text.strip():
            return IntentResult(intent="unknown", confidence=0.0)

        logger.debug(f"Classifying: {user_text!r}")

        raw = self._call_ollama(user_text)
        result = self._parse(raw)

        if result is None:
            logger.warning("Initial parse failed - attempting repair pass.")
            raw = self._call_ollama(user_text, repair=True)
            result = self._parse(raw)

        if result is None:
            logger.error(f"Could not parse intent after repair. Raw: {raw!r}")
            return IntentResult(intent="unknown", raw_response=raw or "")

        result.raw_response = raw or ""

        if result.intent not in INTENT_LABELS:
            logger.warning(f"Model returned unknown label {result.intent!r} - normalising.")
            result.intent = "unknown"

        logger.info(f"Classified -> {result}")
        return result

    def _call_ollama(self, user_text: str, *, repair: bool = False) -> str:
        user_content = user_text if not repair else user_text + _REPAIR_SUFFIX
        try:
            response = ollama.chat(
                model=self.model,
                format="json",
                options={"temperature": self.temperature},
                messages=[
                    {"role": "system", "content": _build_system_prompt()},
                    {"role": "user",   "content": user_content},
                ],
            )
            return response.message.content.strip()
        except ollama.ResponseError as exc:
            logger.error(f"Ollama ResponseError during classification: {exc}")
            return ""
        except Exception as exc:
            logger.error(f"Unexpected error during classification: {exc}")
            return ""

    @staticmethod
    def _parse(raw: str) -> IntentResult | None:
        if not raw:
            return None
        cleaned = raw.strip()
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)
        brace_start = cleaned.find("{")
        brace_end   = cleaned.rfind("}")
        if brace_start != -1 and brace_end != -1 and brace_end > brace_start:
            cleaned = cleaned[brace_start : brace_end + 1]
        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            logger.debug(f"JSON decode error: {exc} | raw: {cleaned[:120]!r}")
            return None
        if not isinstance(data, dict):
            return None
        return IntentResult(
            intent=str(data.get("intent", "unknown")),
            entities=data.get("entities", {}) if isinstance(data.get("entities"), dict) else {},
            confidence=float(data.get("confidence", 0.0)),
        )
