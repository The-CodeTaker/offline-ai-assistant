"""
skills/weather.py — WeatherSkill: fetch real-time weather via wttr.in.

API used: https://wttr.in/{location}?format=j1
  - Completely free, no API key required.
  - Returns a rich JSON payload with current conditions, feels-like, humidity,
    wind speed, UV index, and a 3-day forecast.
  - The LLM stays 100% offline; only the HTTP fetch touches the internet.

RETURN CONTRACT
---------------
Always returns a dict matching BaseSkill's _success / _error shape:

  Success:
    {
      "status":         "success",
      "intent":         "get_weather",          # tells GUI what kind of card to render
      "location":       "Mumbai",               # as resolved by wttr.in
      "temp_c":         32,
      "temp_f":         90,
      "feels_like_c":   35,
      "feels_like_f":   95,
      "condition":      "Sunny",
      "humidity_pct":   68,
      "wind_kmh":       14,
      "uv_index":       7,
      "visibility_km":  10,
      "observation_time": "12:00 PM",
    }

  Error:
    {"status": "error", "reason": "<human-readable message>"}

The "intent" key is included so gui.py can distinguish weather results
from navigation results when deciding which card widget to render.
"""

from __future__ import annotations

from typing import Any

import requests
from loguru import logger

from skills.base_skill import BaseSkill

# wttr.in endpoint — {location} is URL-encoded by requests automatically
_WTTR_URL = "https://wttr.in/{location}?format=j1"

# Hard timeout for the weather fetch (seconds)
_HTTP_TIMEOUT = 10


class WeatherSkill(BaseSkill):
    """
    Fetches current weather conditions for a given location from wttr.in.

    Called by assistant._dispatch_skill when intent == "get_weather".

    Usage:
        result = WeatherSkill(db_path).execute({"location": "London"})
    """

    def execute(self, entities: dict[str, Any]) -> dict[str, Any]:
        """
        Primary entry-point (satisfies BaseSkill contract).
        Delegates to get_current().
        """
        return self.get_current(entities)

    def get_current(self, entities: dict[str, Any]) -> dict[str, Any]:
        """
        Fetch and parse current weather for the location in *entities*.

        Args:
            entities: Must contain "location" key (e.g. "Paris", "New Delhi").
                      Falls back to "raw_text" if "location" is absent.

        Returns:
            Structured weather dict (see module docstring) or error dict.
        """
        location = (
            str(entities.get("location", "")).strip()
            or str(entities.get("raw_text", "")).strip()
        )

        if not location:
            return self._error(
                "I need a location to check the weather. "
                "Try: 'What's the weather in Mumbai?'"
            )

        logger.info(f"WeatherSkill: fetching weather for '{location}'")

        try:
            resp = requests.get(
                _WTTR_URL.format(location=location),
                timeout=_HTTP_TIMEOUT,
                headers={"User-Agent": "OfflineAIAssistant/1.0"},
            )
            resp.raise_for_status()
            data = resp.json()
        except requests.exceptions.ConnectionError:
            logger.error("WeatherSkill: no internet connection.")
            return self._error(
                "I can't reach the weather service right now. "
                "Please check your internet connection."
            )
        except requests.exceptions.Timeout:
            logger.error("WeatherSkill: request timed out.")
            return self._error("The weather service timed out. Please try again.")
        except requests.exceptions.HTTPError as exc:
            logger.error(f"WeatherSkill: HTTP error: {exc}")
            return self._error(f"Weather service returned an error: {exc}")
        except Exception as exc:
            logger.error(f"WeatherSkill: unexpected error: {exc}")
            return self._error(f"Unexpected error fetching weather: {exc}")

        return self._parse(data, location)

    # ------------------------------------------------------------------
    # Private parser
    # ------------------------------------------------------------------

    def _parse(self, data: dict, queried_location: str) -> dict[str, Any]:
        """
        Extract the fields we care about from the wttr.in j1 JSON payload.

        wttr.in j1 structure (relevant parts):
          data["nearest_area"][0]["areaName"][0]["value"]     → resolved place name
          data["current_condition"][0]
            .temp_C / temp_F
            .FeelsLikeC / FeelsLikeF
            .weatherDesc[0]["value"]                          → condition string
            .humidity
            .windspeedKmph
            .uvIndex
            .visibility
            .observation_time
        """
        try:
            # Resolved place name (wttr.in may correct spelling / add country)
            area = (
                data.get("nearest_area", [{}])[0]
                    .get("areaName", [{}])[0]
                    .get("value", queried_location)
            )
            country = (
                data.get("nearest_area", [{}])[0]
                    .get("country", [{}])[0]
                    .get("value", "")
            )
            display_location = f"{area}, {country}" if country else area

            cur = data["current_condition"][0]

            temp_c       = int(cur.get("temp_C", 0))
            temp_f       = int(cur.get("temp_F", 0))
            feels_like_c = int(cur.get("FeelsLikeC", temp_c))
            feels_like_f = int(cur.get("FeelsLikeF", temp_f))
            condition    = cur.get("weatherDesc", [{}])[0].get("value", "Unknown")
            humidity     = int(cur.get("humidity", 0))
            wind_kmh     = int(cur.get("windspeedKmph", 0))
            uv_index     = int(cur.get("uvIndex", 0))
            visibility   = int(cur.get("visibility", 0))
            obs_time     = cur.get("observation_time", "")

            logger.info(
                f"WeatherSkill: {display_location} → "
                f"{temp_c}°C, {condition}, humidity {humidity}%"
            )

            return self._success({
                "intent":           "get_weather",
                "location":         display_location,
                "temp_c":           temp_c,
                "temp_f":           temp_f,
                "feels_like_c":     feels_like_c,
                "feels_like_f":     feels_like_f,
                "condition":        condition,
                "humidity_pct":     humidity,
                "wind_kmh":         wind_kmh,
                "uv_index":         uv_index,
                "visibility_km":    visibility,
                "observation_time": obs_time,
            })

        except (KeyError, IndexError, ValueError) as exc:
            logger.error(f"WeatherSkill: failed to parse wttr.in response: {exc}")
            return self._error(
                f"I received weather data for '{queried_location}' "
                "but couldn't parse it. Please try again."
            )
