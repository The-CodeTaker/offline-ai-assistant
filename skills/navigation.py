"""
skills/navigation.py — NavigationSkill: geocode + route via free public APIs.

PIPELINE
--------
  1. Geocode origin  → (lat, lon) via Nominatim (OpenStreetMap)
  2. Geocode destination → (lat, lon) via Nominatim
  3. Route origin→destination via OSRM public API (driving profile, steps=true)
  4. Return structured dict containing distance, duration, coordinates, and
     a list of turn-by-turn driving steps so gui.py can render a directions
     panel inside the _MapBubble.

APIS USED
---------
  Nominatim  https://nominatim.openstreetmap.org/search
    - Free, no key, 1 req/s rate limit (we add a small delay between calls).

  OSRM       https://router.project-osrm.org/route/v1/driving/
    - Free public demo server, no key required.
    - For production use, self-host OSRM or use a paid routing API.

RETURN CONTRACT
---------------
  Success:
    {
      "status":            "success",
      "intent":            "search_travel",
      "origin":            "New Delhi",
      "destination":       "Agra",
      "origin_lat":        28.6139,
      "origin_lon":        77.2090,
      "dest_lat":          27.1767,
      "dest_lon":          78.0081,
      "distance_km":       233.4,
      "distance_miles":    145.0,
      "duration_minutes":  195,
      "duration_text":     "3 hours 15 minutes",
      "directions": [
          "Head south on NH 19",
          "Turn right onto Yamuna Expressway",
          ...
      ]
    }

  Error:
    {"status": "error", "reason": "<human-readable message>"}
"""

from __future__ import annotations

import time
from typing import Any

import requests
from loguru import logger

from skills.base_skill import BaseSkill

# ── API endpoints ─────────────────────────────────────────────────────────────
_NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
_OSRM_URL      = "https://router.project-osrm.org/route/v1/driving/{lon1},{lat1};{lon2},{lat2}"

# ── Request config ────────────────────────────────────────────────────────────
_HTTP_TIMEOUT    = 10    # seconds per request
_NOMINATIM_DELAY = 1.1   # seconds between Nominatim calls (rate-limit compliance)
_USER_AGENT      = "OfflineAIAssistant/1.0 (personal project)"

# Maximum number of turn-by-turn steps to return (keeps the UI manageable)
_MAX_STEPS = 500


class NavigationSkill(BaseSkill):
    """
    Calculates driving distance, estimated time, and turn-by-turn directions
    between two locations.

    Called by assistant._dispatch_skill when intent == "search_travel".

    Usage:
        result = NavigationSkill(db_path).execute({
            "origin": "Delhi", "destination": "Agra"
        })
    """

    def execute(self, entities: dict[str, Any]) -> dict[str, Any]:
        """Primary entry-point — delegates to route()."""
        return self.route(entities)

    def route(self, entities: dict[str, Any]) -> dict[str, Any]:
        """
        Geocode both endpoints then fetch a driving route with step-by-step
        directions.

        Entity keys consumed (in priority order):
          origin      → "origin"  > "from"   > first half of "location"
          destination → "destination" > "to" > second half of "location"
        """
        origin, destination = self._resolve_endpoints(entities)

        if not origin:
            return self._error(
                "I need a starting location. "
                "Try: 'How do I get from Delhi to Agra?'"
            )
        if not destination:
            return self._error(
                "I need a destination. "
                "Try: 'How do I get from Delhi to Agra?'"
            )

        logger.info(f"NavigationSkill: routing '{origin}' -> '{destination}'")

        # ── Step 1: geocode origin ─────────────────────────────────────────
        origin_coords = self._geocode(origin)
        if origin_coords is None:
            return self._error(
                f"I couldn't find the location '{origin}' on the map. "
                "Try being more specific (e.g. add city or country)."
            )

        # Nominatim rate-limit: wait between calls
        time.sleep(_NOMINATIM_DELAY)

        # ── Step 2: geocode destination ────────────────────────────────────
        dest_coords = self._geocode(destination)
        if dest_coords is None:
            return self._error(
                f"I couldn't find the location '{destination}' on the map. "
                "Try being more specific."
            )

        # ── Step 3: fetch OSRM route with turn-by-turn steps ──────────────
        route_data = self._fetch_route(origin_coords, dest_coords)
        if route_data is None:
            return self._error(
                "I found both locations but couldn't calculate a driving route "
                "between them. They may be separated by water or the routing "
                "service may be temporarily unavailable."
            )

        distance_m, duration_s, directions = route_data
        distance_km    = round(distance_m / 1000, 1)
        distance_miles = round(distance_km * 0.621371, 1)
        duration_min   = int(duration_s / 60)
        duration_text  = self._format_duration(duration_min)

        logger.info(
            f"NavigationSkill: {origin} -> {destination} = "
            f"{distance_km} km, {duration_text}, {len(directions)} steps"
        )

        return self._success({
            "intent":           "search_travel",
            "origin":           origin,
            "destination":      destination,
            "origin_lat":       origin_coords[0],
            "origin_lon":       origin_coords[1],
            "dest_lat":         dest_coords[0],
            "dest_lon":         dest_coords[1],
            "distance_km":      distance_km,
            "distance_miles":   distance_miles,
            "duration_minutes": duration_min,
            "duration_text":    duration_text,
            "directions":       directions,
        })

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_endpoints(entities: dict[str, Any]) -> tuple[str, str]:
        """
        Extract origin and destination strings from the entities dict.

        Priority:
          origin      → entities["origin"] > entities["from"]
          destination → entities["destination"] > entities["to"]

        Fallback: if neither pair is found but "location" contains " to ",
        split it: "Delhi to Agra" → origin="Delhi", destination="Agra".
        """
        origin      = (
            str(entities.get("origin", "")).strip()
            or str(entities.get("from", "")).strip()
        )
        destination = (
            str(entities.get("destination", "")).strip()
            or str(entities.get("to", "")).strip()
        )

        # Fallback: try splitting "location" field on " to "
        if not origin or not destination:
            location = str(entities.get("location", "")).strip()
            if " to " in location.lower():
                idx         = location.lower().find(" to ")
                origin      = origin      or location[:idx].strip()
                destination = destination or location[idx + 4:].strip()

        return origin, destination

    def _geocode(self, place: str) -> tuple[float, float] | None:
        """
        Convert a place name to (latitude, longitude) via Nominatim.

        Returns None on any failure so callers can return a clean error.
        """
        try:
            resp = requests.get(
                _NOMINATIM_URL,
                params={"q": place, "format": "json", "limit": 1},
                headers={"User-Agent": _USER_AGENT},
                timeout=_HTTP_TIMEOUT,
            )
            resp.raise_for_status()
            results = resp.json()

            if not results:
                logger.warning(f"NavigationSkill: Nominatim found no results for '{place}'")
                return None

            lat = float(results[0]["lat"])
            lon = float(results[0]["lon"])
            logger.debug(f"NavigationSkill: geocoded '{place}' -> ({lat:.4f}, {lon:.4f})")
            return lat, lon

        except requests.exceptions.ConnectionError:
            logger.error("NavigationSkill: no internet connection for geocoding.")
            return None
        except requests.exceptions.Timeout:
            logger.error(f"NavigationSkill: Nominatim timed out for '{place}'.")
            return None
        except Exception as exc:
            logger.error(f"NavigationSkill: geocoding error for '{place}': {exc}")
            return None

    def _fetch_route(
        self,
        origin:      tuple[float, float],
        destination: tuple[float, float],
    ) -> tuple[float, float, list[str]] | None:
        """
        Fetch driving distance (metres), duration (seconds), and a list of
        turn-by-turn instruction strings from OSRM.

        Returns (distance_m, duration_s, directions) or None on failure.

        Changes from previous version:
          - "steps": "true"  is now passed so OSRM returns maneuver steps.
          - "overview": "false" is kept to avoid sending large geometry data.
          - Steps are parsed from data["routes"][0]["legs"][0]["steps"].
          - Each step's instruction is built from the maneuver type + road name.
        """
        lat1, lon1 = origin
        lat2, lon2 = destination

        url = _OSRM_URL.format(lat1=lat1, lon1=lon1, lat2=lat2, lon2=lon2)

        try:
            resp = requests.get(
                url,
                params={"overview": "false", "steps": "true"},
                headers={"User-Agent": _USER_AGENT},
                timeout=_HTTP_TIMEOUT,
            )
            resp.raise_for_status()
            data = resp.json()

            if data.get("code") != "Ok" or not data.get("routes"):
                logger.warning(
                    f"NavigationSkill: OSRM returned no route. code={data.get('code')}"
                )
                return None

            route    = data["routes"][0]
            distance = float(route["distance"])
            duration = float(route["duration"])

            # ── Parse turn-by-turn steps ──────────────────────────────────
            directions: list[str] = []
            try:
                raw_steps = route["legs"][0]["steps"]
                for step in raw_steps[:_MAX_STEPS]:
                    instruction = self._format_step(step)
                    if instruction:
                        directions.append(instruction)
            except (KeyError, IndexError, TypeError) as exc:
                logger.warning(f"NavigationSkill: could not parse steps: {exc}")
                # Steps parsing failure is non-fatal; return empty list
                directions = []

            logger.debug(
                f"NavigationSkill: OSRM -> {distance/1000:.1f} km, "
                f"{duration/60:.0f} min, {len(directions)} steps"
            )
            return distance, duration, directions

        except requests.exceptions.ConnectionError:
            logger.error("NavigationSkill: no internet connection for routing.")
            return None
        except requests.exceptions.Timeout:
            logger.error("NavigationSkill: OSRM timed out.")
            return None
        except Exception as exc:
            logger.error(f"NavigationSkill: OSRM error: {exc}")
            return None

    @staticmethod
    def _format_step(step: dict) -> str:
        """
        Build a human-readable instruction string from a single OSRM step dict.

        OSRM step structure (relevant fields):
          step["maneuver"]["type"]       — e.g. "turn", "depart", "arrive"
          step["maneuver"]["modifier"]   — e.g. "left", "right", "straight"
          step["name"]                   — road/street name (may be empty)
          step["distance"]               — metres for this step
          step["duration"]               — seconds for this step

        We build a short imperative phrase that mirrors what a satnav would say.
        """
        try:
            maneuver  = step.get("maneuver", {})
            m_type    = maneuver.get("type", "").lower()
            modifier  = maneuver.get("modifier", "").lower()
            road_name = step.get("name", "").strip()
            dist_m    = float(step.get("distance", 0))

            # Skip zero-distance steps (usually duplicate arrive markers)
            if dist_m < 1:
                return ""

            dist_str = (
                f"{int(dist_m)} m"        if dist_m < 1000
                else f"{dist_m/1000:.1f} km"
            )

            # Build the verb phrase
            if m_type == "depart":
                verb = "Start"
            elif m_type == "arrive":
                return "Arrive at your destination"
            elif m_type == "turn":
                if modifier in ("left", "slight left", "sharp left"):
                    verb = f"Turn {modifier}"
                elif modifier in ("right", "slight right", "sharp right"):
                    verb = f"Turn {modifier}"
                else:
                    verb = "Continue straight"
            elif m_type == "new name":
                verb = "Continue"
            elif m_type in ("merge", "on ramp"):
                verb = f"Merge {modifier}" if modifier else "Merge"
            elif m_type == "off ramp":
                verb = "Take the exit"
            elif m_type == "fork":
                verb = f"Keep {modifier}" if modifier else "Keep"
            elif m_type == "roundabout":
                exit_num = maneuver.get("exit", "")
                verb = f"Take exit {exit_num} at roundabout" if exit_num else "Take the roundabout"
            elif m_type == "rotary":
                exit_num = maneuver.get("exit", "")
                verb = f"Take exit {exit_num} at roundabout" if exit_num else "Take the roundabout"
            elif m_type == "continue":
                verb = "Continue"
            else:
                verb = m_type.capitalize() if m_type else "Continue"

            # Assemble the instruction
            if road_name:
                instruction = f"{verb} onto {road_name}"
            else:
                instruction = verb

            return f"{instruction} ({dist_str})"

        except Exception as exc:
            logger.debug(f"NavigationSkill._format_step error: {exc}")
            return ""

    @staticmethod
    def _format_duration(minutes: int) -> str:
        """Convert total minutes to a human-readable string."""
        if minutes < 60:
            return f"{minutes} minute{'s' if minutes != 1 else ''}"
        hours = minutes // 60
        mins  = minutes % 60
        h_str = f"{hours} hour{'s' if hours != 1 else ''}"
        m_str = f"{mins} minute{'s' if mins != 1 else ''}" if mins else ""
        return f"{h_str} {m_str}".strip()
