"""
gui.py — CustomTkinter desktop GUI for the Offline AI Assistant.

Changes in this version
-----------------------
1. Sidebar repurposed — now shows Chat Sessions from the sessions DB table.
   • "+ New Chat" button at the top creates a new session.
   • Each session row has a 🗑 delete button.
   • Clicking a session loads its history into the chat frame.

2. Memory/Tools modal — "🗃 Memory" button in the header opens a Toplevel
   window listing all Notes and Reminders with individual 🗑 delete buttons.

3. Multi-file attach — _on_attach now calls askopenfilenames (plural) and
   collects a list of paths.  _dispatch_user_message passes the list directly
   to respond_stream_with_data(file_paths=[...]).

4. Session-aware dispatch — every call to respond_stream_with_data passes
   the active session id (already stored in self.assistant.session_id).

All existing behaviour is preserved:
  • _ChatBubble / _WeatherBubble / _MapBubble are 100% unchanged.
  • Auto-Voice toggle, ► Play buttons, streaming protocol untouched.
  • Mic threading, send/enter binding, close handler — all untouched.
  • All colour tokens preserved exactly.
"""

from __future__ import annotations

import os
import re
import sqlite3
import threading
import time
from datetime import datetime
from typing import Any

import customtkinter as ctk
from customtkinter import filedialog
from loguru import logger

try:
    from tkintermapview import TkinterMapView
    _MAP_AVAILABLE = True
except ImportError:
    _MAP_AVAILABLE = False
    logger.warning("tkintermapview not installed. Install with: pip install tkintermapview")

import config
from core.assistant import Assistant
from skills.notes import NoteSkill
from skills.reminder import ReminderSkill
from voice.listener import VoiceListener
from voice.speaker import VoiceSpeaker
from memory.database import (
    list_sessions, create_session, delete_session, rename_session,
    load_conversation_history,
)


# ---------------------------------------------------------------------------
# Design tokens  (identical to previous version — do not alter)
# ---------------------------------------------------------------------------

_FONTS = {
    "app_title":     ("Georgia", 15, "bold"),
    "timestamp":     ("Courier New", 10),
    "user_name":     ("Georgia", 11, "bold"),
    "asst_name":     ("Georgia", 11, "bold"),
    "bubble_text":   ("Georgia", 13),
    "input":         ("Georgia", 13),
    "button":        ("Georgia", 12, "bold"),
    "status":        ("Courier New", 11),
    "toggle_lbl":    ("Georgia", 11),
    "card_title":    ("Georgia", 13, "bold"),
    "card_value":    ("Georgia", 22, "bold"),
    "card_sub":      ("Georgia", 11),
    "card_label":    ("Georgia", 10),
    "map_title":     ("Georgia", 12, "bold"),
    "map_meta":      ("Courier New", 11),
    "code":          ("Courier New", 12),
    "sidebar_title": ("Georgia", 13, "bold"),
    "sidebar_note":  ("Georgia", 11),
    "dir_step":      ("Georgia", 11),
}

_LIGHT = {
    "window_bg":         "#F5F0EB",
    "header_bg":         "#EDE8E1",
    "header_border":     "#D6CFC5",
    "chat_bg":           "#F5F0EB",
    "user_bubble":       "#2C2C2C",
    "user_bubble_txt":   "#F5F0EB",
    "asst_bubble":       "#FFFFFF",
    "asst_bubble_txt":   "#1A1A1A",
    "asst_border":       "#DDD8D1",
    "thinking_bg":       "#EDE8E1",
    "thinking_txt":      "#888077",
    "input_bg":          "#FFFFFF",
    "input_border":      "#C8C2B9",
    "input_txt":         "#1A1A1A",
    "input_ph":          "#AAA49C",
    "send_btn":          "#2C2C2C",
    "send_btn_hover":    "#444444",
    "send_btn_txt":      "#F5F0EB",
    "mic_btn":           "#D4CEC6",
    "mic_btn_hover":     "#C5BFB7",
    "mic_btn_txt":       "#2C2C2C",
    "mic_active":        "#C0392B",
    "mic_active_txt":    "#FFFFFF",
    "attach_btn":        "#D4CEC6",
    "attach_btn_hover":  "#C5BFB7",
    "attach_btn_txt":    "#2C2C2C",
    "attach_active":     "#2C6E2C",
    "attach_active_txt": "#FFFFFF",
    "status_txt":        "#888077",
    "title_txt":         "#1A1A1A",
    "dot_color":         "#27AE60",
    "timestamp_txt":     "#B0A89E",
    "weather_bg":        "#EAF4FB",
    "weather_border":    "#B8D9EF",
    "weather_txt":       "#1A3A52",
    "weather_sub":       "#5A8AA8",
    "map_bg":            "#F0F4F0",
    "map_border":        "#B8D4B8",
    "map_header_bg":     "#E4EDE4",
    "map_txt":           "#1A3A1A",
    "map_sub":           "#4A7A4A",
    "map_pill_bg":       "#2C6E2C",
    "map_pill_txt":      "#FFFFFF",
    "dir_bg":            "#E8EDE8",
    "dir_border":        "#C8D4C8",
    "dir_txt":           "#1A3A1A",
    "dir_step_bg":       "#F0F4F0",
    "sidebar_bg":        "#EDE8E1",
    "sidebar_border":    "#D6CFC5",
    "sidebar_btn":       "#D6CFC5",
    "sidebar_btn_hover": "#C5BFB7",
    "sidebar_btn_txt":   "#1A1A1A",
    "sidebar_del":       "#D6CFC5",
    "sidebar_del_hover": "#E74C3C",
    "sidebar_del_txt":   "#888077",
    "code_bg":           "#F0EDE8",
    "code_border":       "#C8C2B9",
    "code_txt":          "#1A1A1A",
    "modal_bg":          "#F5F0EB",
    "modal_border":      "#D6CFC5",
    "del_btn":           "#D6CFC5",
    "del_btn_hover":     "#E74C3C",
    "del_btn_txt":       "#888077",
}

_DARK = {
    "window_bg":         "#0F0F0F",
    "header_bg":         "#161616",
    "header_border":     "#2A2A2A",
    "chat_bg":           "#0F0F0F",
    "user_bubble":       "#E8E3DC",
    "user_bubble_txt":   "#0F0F0F",
    "asst_bubble":       "#1C1C1C",
    "asst_bubble_txt":   "#E8E3DC",
    "asst_border":       "#2A2A2A",
    "thinking_bg":       "#1A1A1A",
    "thinking_txt":      "#666666",
    "input_bg":          "#1C1C1C",
    "input_border":      "#2E2E2E",
    "input_txt":         "#E8E3DC",
    "input_ph":          "#555555",
    "send_btn":          "#E8E3DC",
    "send_btn_hover":    "#FFFFFF",
    "send_btn_txt":      "#0F0F0F",
    "mic_btn":           "#2A2A2A",
    "mic_btn_hover":     "#333333",
    "mic_btn_txt":       "#E8E3DC",
    "mic_active":        "#E74C3C",
    "mic_active_txt":    "#FFFFFF",
    "attach_btn":        "#2A2A2A",
    "attach_btn_hover":  "#333333",
    "attach_btn_txt":    "#E8E3DC",
    "attach_active":     "#2ECC71",
    "attach_active_txt": "#0A1A0A",
    "status_txt":        "#555555",
    "title_txt":         "#E8E3DC",
    "dot_color":         "#2ECC71",
    "timestamp_txt":     "#444444",
    "weather_bg":        "#0D1F2D",
    "weather_border":    "#1A3D5C",
    "weather_txt":       "#C8E6F5",
    "weather_sub":       "#6AADD4",
    "map_bg":            "#0D1F0D",
    "map_border":        "#1A3D1A",
    "map_header_bg":     "#0F260F",
    "map_txt":           "#C8F5C8",
    "map_sub":           "#6AD46A",
    "map_pill_bg":       "#2ECC71",
    "map_pill_txt":      "#0A1A0A",
    "dir_bg":            "#0A180A",
    "dir_border":        "#1A3D1A",
    "dir_txt":           "#C8F5C8",
    "dir_step_bg":       "#0D1F0D",
    "sidebar_bg":        "#141414",
    "sidebar_border":    "#2A2A2A",
    "sidebar_btn":       "#222222",
    "sidebar_btn_hover": "#2E2E2E",
    "sidebar_btn_txt":   "#E8E3DC",
    "sidebar_del":       "#222222",
    "sidebar_del_hover": "#E74C3C",
    "sidebar_del_txt":   "#555555",
    "code_bg":           "#111111",
    "code_border":       "#2E2E2E",
    "code_txt":          "#A8E6CF",
    "modal_bg":          "#141414",
    "modal_border":      "#2A2A2A",
    "del_btn":           "#222222",
    "del_btn_hover":     "#E74C3C",
    "del_btn_txt":       "#555555",
}

_WEATHER_ICONS = {
    "sunny": "☀", "clear": "☾", "partly cloudy": "⛅", "cloudy": "☁",
    "overcast": "☁", "mist": "🌫", "fog": "🌫", "rain": "🌧",
    "drizzle": "🌦", "shower": "🌦", "thunder": "⛈", "storm": "🌩",
    "snow": "❄", "blizzard": "🌨", "sleet": "🌨", "hail": "🌩",
}

def _weather_icon(condition: str) -> str:
    lower = condition.lower()
    for keyword, icon in _WEATHER_ICONS.items():
        if keyword in lower:
            return icon
    return "🌡"


# ---------------------------------------------------------------------------
# Markdown / code-block helpers  (unchanged)
# ---------------------------------------------------------------------------

_CODE_FENCE_RE = re.compile(r"```(\w*)\n?(.*?)```", re.DOTALL)

def _has_code_blocks(text: str) -> bool:
    return bool(_CODE_FENCE_RE.search(text))

def _parse_segments(text: str) -> list[dict]:
    segments: list[dict] = []
    last_end = 0
    for match in _CODE_FENCE_RE.finditer(text):
        prose = text[last_end:match.start()]
        if prose:
            segments.append({"type": "prose", "content": prose})
        segments.append({
            "type": "code", "content": match.group(2).rstrip("\n"),
            "lang": match.group(1) or "code",
        })
        last_end = match.end()
    tail = text[last_end:]
    if tail:
        segments.append({"type": "prose", "content": tail})
    return segments


# ---------------------------------------------------------------------------
# _ChatBubble  (100% unchanged)
# ---------------------------------------------------------------------------

class _ChatBubble(ctk.CTkFrame):
    MAX_WIDTH = 520

    def __init__(self, parent, role, text, theme, speaker_callback=None, **kw):
        is_user     = role == "user"
        is_thinking = role == "thinking"
        if is_user:
            bg = bd = theme["user_bubble"]
            fg = theme["user_bubble_txt"]
        elif is_thinking:
            bg = bd = theme["thinking_bg"]
            fg = theme["thinking_txt"]
        else:
            bg = theme["asst_bubble"]
            fg = theme["asst_bubble_txt"]
            bd = theme["asst_border"]

        super().__init__(parent, fg_color=bg, border_color=bd, border_width=1,
                         corner_radius=14, **kw)
        self._theme            = theme
        self._fg               = fg
        self._is_user          = is_user
        self._is_thinking      = is_thinking
        self._speaker_callback = speaker_callback
        self._current_text     = text

        ts   = datetime.now().strftime("%H:%M")
        name = "You" if is_user else ("Assistant" if not is_thinking else "")

        self._inner = ctk.CTkFrame(self, fg_color="transparent")
        self._inner.pack(padx=14, pady=10, fill="both", expand=True)

        if name:
            self._header_frame = ctk.CTkFrame(self._inner, fg_color="transparent")
            self._header_frame.pack(fill="x")
            ctk.CTkLabel(self._header_frame, text=name,
                         font=_FONTS["user_name"] if is_user else _FONTS["asst_name"],
                         text_color=fg, anchor="w").pack(side="left")
            if not is_user and not is_thinking and speaker_callback:
                self._play_btn = ctk.CTkButton(
                    self._header_frame, text="► Play", width=50, height=20,
                    corner_radius=10, fg_color="transparent",
                    hover_color=theme["asst_border"], text_color=theme["timestamp_txt"],
                    font=_FONTS["timestamp"],
                    command=lambda: speaker_callback(self._current_text),
                )
                self._play_btn.pack(side="left", padx=10)
            ctk.CTkLabel(self._header_frame, text=ts, font=_FONTS["timestamp"],
                         text_color=theme["timestamp_txt"], anchor="e").pack(side="right")

        self._content_frame = ctk.CTkFrame(self._inner, fg_color="transparent")
        self._content_frame.pack(fill="x", pady=(4, 0) if name else 0)

        self._label = ctk.CTkLabel(self._content_frame, text=text,
                                   font=_FONTS["bubble_text"], text_color=fg,
                                   anchor="w", justify="left",
                                   wraplength=self.MAX_WIDTH - 60)
        self._label.pack(fill="x", anchor="w")

    def update_text(self, new_text):
        self._current_text = new_text
        try:
            self._label.configure(text=new_text)
        except Exception:
            pass

    def finalize(self, full_text):
        self._current_text = full_text
        if _has_code_blocks(full_text):
            try:
                self._label.destroy()
            except Exception:
                pass
            self._render_with_code(full_text)
        else:
            try:
                self._label.configure(text=full_text)
            except Exception:
                pass

    def _render_with_code(self, text):
        t = self._theme
        for seg in _parse_segments(text):
            if seg["type"] == "prose":
                content = seg["content"].strip()
                if not content:
                    continue
                ctk.CTkLabel(self._content_frame, text=content,
                             font=_FONTS["bubble_text"], text_color=self._fg,
                             anchor="w", justify="left",
                             wraplength=self.MAX_WIDTH - 60).pack(fill="x", anchor="w", pady=(0, 4))
            else:
                lang = seg["lang"]; code = seg["content"]
                n_lines = code.count("\n") + 1
                height  = min(n_lines, 20) * 18 + 12
                ctk.CTkLabel(self._content_frame,
                             text=f"  {lang}" if lang else "  code",
                             font=_FONTS["code"],
                             text_color=t.get("timestamp_txt", "#888"),
                             fg_color=t.get("code_border", "#2E2E2E"),
                             corner_radius=4, anchor="w").pack(fill="x", anchor="w", pady=(6, 0))
                box = ctk.CTkTextbox(self._content_frame, font=_FONTS["code"],
                                     fg_color=t.get("code_bg", "#111111"),
                                     border_color=t.get("code_border", "#2E2E2E"),
                                     border_width=1, corner_radius=6,
                                     text_color=t.get("code_txt", "#A8E6CF"),
                                     height=height, wrap="none", activate_scrollbars=True)
                box.pack(fill="x", anchor="w", pady=(0, 6))
                box.insert("1.0", code)
                box.configure(state="disabled")


# ---------------------------------------------------------------------------
# _WeatherBubble  (100% unchanged)
# ---------------------------------------------------------------------------

class _WeatherBubble(ctk.CTkFrame):
    WIDTH = 400

    def __init__(self, parent, data, theme, **kw):
        super().__init__(parent, fg_color=theme["weather_bg"],
                         border_color=theme["weather_border"],
                         border_width=1, corner_radius=14, width=self.WIDTH, **kw)
        icon      = _weather_icon(data.get("condition", ""))
        location  = data.get("location", "Unknown")
        obs_time  = data.get("observation_time", "")
        temp_c    = data.get("temp_c", "--")
        feels_c   = data.get("feels_like_c", "--")
        condition = data.get("condition", "")
        humidity  = data.get("humidity_pct", "--")
        wind      = data.get("wind_kmh", "--")
        uv        = data.get("uv_index", "--")
        txt, sub  = theme["weather_txt"], theme["weather_sub"]

        header = ctk.CTkFrame(self, fg_color=theme["weather_border"], corner_radius=0, height=36)
        header.pack(fill="x"); header.pack_propagate(False)
        ctk.CTkLabel(header, text=f"{icon}  {location}", font=_FONTS["card_title"],
                     text_color=txt, anchor="w").pack(side="left", padx=12, pady=6)
        if obs_time:
            ctk.CTkLabel(header, text=obs_time, font=_FONTS["timestamp"],
                         text_color=sub, anchor="e").pack(side="right", padx=12, pady=6)

        main_row = ctk.CTkFrame(self, fg_color="transparent")
        main_row.pack(fill="x", padx=16, pady=(12, 4))
        ctk.CTkLabel(main_row, text=f"{temp_c}°C", font=_FONTS["card_value"],
                     text_color=txt).pack(side="left")
        right_col = ctk.CTkFrame(main_row, fg_color="transparent")
        right_col.pack(side="left", padx=12)
        ctk.CTkLabel(right_col, text=condition, font=_FONTS["card_title"],
                     text_color=txt, anchor="w").pack(anchor="w")
        ctk.CTkLabel(right_col, text=f"Feels like {feels_c}°C", font=_FONTS["card_sub"],
                     text_color=sub, anchor="w").pack(anchor="w")

        stats_row = ctk.CTkFrame(self, fg_color="transparent")
        stats_row.pack(fill="x", padx=16, pady=(4, 14))
        for label, value in [("Humidity", f"{humidity}%"), ("Wind", f"{wind} km/h"), ("UV Index", str(uv))]:
            pill = ctk.CTkFrame(stats_row, fg_color=theme["weather_border"], corner_radius=8)
            pill.pack(side="left", padx=(0, 8))
            ctk.CTkLabel(pill, text=label, font=_FONTS["card_label"], text_color=sub).pack(padx=10, pady=(6, 0))
            ctk.CTkLabel(pill, text=value, font=_FONTS["card_sub"], text_color=txt).pack(padx=10, pady=(0, 6))


# ---------------------------------------------------------------------------
# _MapBubble  (100% unchanged — directions panel included)
# ---------------------------------------------------------------------------

class _MapBubble(ctk.CTkFrame):
    MAP_HEIGHT = 300
    WIDTH      = 580

    def __init__(self, parent, data, theme, **kw):
        super().__init__(parent, fg_color=theme["map_bg"], border_color=theme["map_border"],
                         border_width=1, corner_radius=14, width=self.WIDTH, **kw)
        origin, destination = data.get("origin", "Origin"), data.get("destination", "Destination")
        dist_km, dist_mi    = data.get("distance_km", "?"), data.get("distance_miles", "?")
        duration            = data.get("duration_text", "?")
        origin_lat, origin_lon = data.get("origin_lat"), data.get("origin_lon")
        dest_lat, dest_lon     = data.get("dest_lat"), data.get("dest_lon")
        directions             = data.get("directions", [])
        txt, sub = theme["map_txt"], theme["map_sub"]

        header = ctk.CTkFrame(self, fg_color=theme["map_header_bg"], corner_radius=0)
        header.pack(fill="x")
        title_frame = ctk.CTkFrame(header, fg_color="transparent")
        title_frame.pack(fill="x", padx=14, pady=(10, 6))
        ctk.CTkLabel(title_frame, text=f"🗺 {origin} → {destination}",
                     font=_FONTS["map_title"], text_color=txt, anchor="w").pack(side="left")
        pills_frame = ctk.CTkFrame(header, fg_color="transparent")
        pills_frame.pack(fill="x", padx=14, pady=(0, 10))
        for badge_text in [f"📏 {dist_km} km ({dist_mi} mi)", f"🕒 {duration}"]:
            pill = ctk.CTkFrame(pills_frame, fg_color=theme["map_pill_bg"], corner_radius=10)
            pill.pack(side="left", padx=(0, 8))
            ctk.CTkLabel(pill, text=badge_text, font=_FONTS["map_meta"],
                         text_color=theme["map_pill_txt"]).pack(padx=10, pady=4)

        if _MAP_AVAILABLE and origin_lat is not None and dest_lat is not None:
            self._build_map(origin_lat, origin_lon, dest_lat, dest_lon, origin, destination, sub, directions, theme)
        else:
            no_map_msg = (
                f"📍 Map unavailable — install tkintermapview\n"
                f"   Origin:      {origin_lat:.4f}, {origin_lon:.4f}\n"
                f"   Destination: {dest_lat:.4f}, {dest_lon:.4f}"
                if origin_lat else "📍 Coordinates not available"
            )
            ctk.CTkLabel(self, text=no_map_msg, font=_FONTS["map_meta"],
                         text_color=sub, justify="left", anchor="w").pack(padx=14, pady=14, fill="x")
            if directions:
                self._build_directions_panel(directions, theme)

    def _build_map(self, origin_lat, origin_lon, dest_lat, dest_lon,
                   origin_name, dest_name, sub_color, directions, theme):
        map_widget = TkinterMapView(self, width=self.WIDTH - 4, height=self.MAP_HEIGHT, corner_radius=0)
        map_widget.pack(fill="x", padx=0, pady=0)
        map_widget.set_tile_server("https://a.tile.openstreetmap.org/{z}/{x}/{y}.png", max_zoom=22)
        map_widget.set_marker(origin_lat, origin_lon, text=origin_name,
                              marker_color_circle="#2ECC71", marker_color_outside="#27AE60")
        map_widget.set_marker(dest_lat, dest_lon, text=dest_name,
                              marker_color_circle="#E74C3C", marker_color_outside="#C0392B")
        map_widget.set_path([(origin_lat, origin_lon), (dest_lat, dest_lon)], color="#3498DB", width=3)
        map_widget.set_position((origin_lat + dest_lat) / 2, (origin_lon + dest_lon) / 2)
        span = max(abs(origin_lat - dest_lat), abs(origin_lon - dest_lon))
        map_widget.set_zoom(13 if span < 0.1 else 11 if span < 0.5 else 9 if span < 2 else 7 if span < 8 else 6 if span < 20 else 5)
        ctk.CTkLabel(self, text="© OpenStreetMap contributors", font=_FONTS["card_label"],
                     text_color=sub_color, anchor="e").pack(fill="x", padx=8, pady=(0, 4))
        if directions:
            self._build_directions_panel(directions, theme)

    def _build_directions_panel(self, directions, theme):
        dir_header = ctk.CTkFrame(self, fg_color=theme.get("dir_bg", theme["map_header_bg"]),
                                   corner_radius=0, height=28)
        dir_header.pack(fill="x", padx=0, pady=(4, 0)); dir_header.pack_propagate(False)
        ctk.CTkLabel(dir_header, text="  🧭  Turn-by-Turn Directions", font=_FONTS["map_meta"],
                     text_color=theme.get("dir_txt", theme["map_txt"]), anchor="w").pack(side="left", padx=8, pady=4)
        steps_scroll = ctk.CTkScrollableFrame(self, fg_color=theme.get("dir_step_bg", theme["map_bg"]),
                                              corner_radius=0, height=120,
                                              scrollbar_button_color=theme.get("map_border", "#1A3D1A"))
        steps_scroll.pack(fill="x", padx=0, pady=(0, 6))
        steps_scroll.grid_columnconfigure(0, weight=1)
        dir_txt = theme.get("dir_txt", theme["map_txt"])
        for idx, step in enumerate(directions, start=1):
            if not step:
                continue
            row_frame = ctk.CTkFrame(steps_scroll, fg_color="transparent")
            row_frame.pack(fill="x", padx=6, pady=(2, 0))
            ctk.CTkLabel(row_frame, text=f"{idx:2d}.", font=_FONTS["map_meta"],
                         text_color=theme.get("map_sub", "#6AD46A"), width=28, anchor="e").pack(side="left", padx=(0, 6))
            ctk.CTkLabel(row_frame, text=step, font=_FONTS["dir_step"], text_color=dir_txt,
                         anchor="w", justify="left", wraplength=self.WIDTH - 80).pack(side="left", fill="x", expand=True)


# ---------------------------------------------------------------------------
# AssistantGUI
# ---------------------------------------------------------------------------

class AssistantGUI(ctk.CTk):
    """Main application window."""

    _SIDEBAR_WIDTH = 240

    def __init__(self, assistant: Assistant, listener: VoiceListener, speaker: VoiceSpeaker) -> None:
        super().__init__()

        self.assistant = assistant
        self.listener  = listener
        self.speaker   = speaker

        # State
        self._is_dark        = True
        self._auto_speak     = False
        self._is_busy        = False
        self._mic_active     = False
        self._thinking_bubble: _ChatBubble | None = None
        self._sidebar_open   = False
        self._pending_files: list[str] = []   # multi-file attach queue

        self.title("Offline AI Assistant")
        current_dir = os.path.dirname(os.path.abspath(__file__))
        icon_path   = os.path.join(current_dir, "ghost.ico")
        if os.path.exists(icon_path):
            self.iconbitmap(icon_path)

        self.geometry("980x720")
        self.minsize(640, 460)
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("dark-blue")

        self._build_ui()
        self._apply_theme()

        self.bind("<Return>",   lambda e: self._on_send())
        self.bind("<KP_Enter>", lambda e: self._on_send())

        # Load history of the current session into the chat frame on startup
        self._schedule(self._load_session_into_chat)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # -----------------------------------------------------------------------
    # UI construction
    # -----------------------------------------------------------------------

    def _build_ui(self) -> None:
        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(0, weight=1)

        # ── Header ──────────────────────────────────────────────────────────
        self._header = ctk.CTkFrame(self, height=52, corner_radius=0)
        self._header.grid(row=0, column=0, sticky="ew")
        self._header.grid_propagate(False)
        self._header.grid_columnconfigure(1, weight=1)

        title_row = ctk.CTkFrame(self._header, fg_color="transparent")
        title_row.grid(row=0, column=0, padx=(8, 0), sticky="w")

        self._sidebar_btn = ctk.CTkButton(
            title_row, text="☰", font=("Georgia", 16), width=34, height=28,
            corner_radius=8, fg_color="transparent", border_width=0,
            command=self._toggle_sidebar,
        )
        self._sidebar_btn.pack(side="left", padx=(4, 6))

        self._dot = ctk.CTkLabel(title_row, text="●", font=("Georgia", 14), width=18)
        self._dot.pack(side="left", padx=(0, 6))

        self._title_label = ctk.CTkLabel(title_row, text="Offline AI Assistant", font=_FONTS["app_title"])
        self._title_label.pack(side="left")

        right_frame = ctk.CTkFrame(self._header, fg_color="transparent")
        right_frame.grid(row=0, column=2, padx=18, sticky="e")

        self._voice_btn = ctk.CTkButton(
            right_frame, text="Voice: OFF", font=_FONTS["toggle_lbl"],
            width=110, height=28, corner_radius=14, fg_color="transparent", border_width=1,
            command=self._toggle_voice,
        )
        self._voice_btn.pack(side="left", padx=(0, 8))

        # 🗃 Memory button — opens Notes/Reminders modal
        self._memory_btn = ctk.CTkButton(
            right_frame, text="🗃 Memory", font=_FONTS["toggle_lbl"],
            width=100, height=28, corner_radius=14, fg_color="transparent", border_width=1,
            command=self._open_memory_modal,
        )
        self._memory_btn.pack(side="left", padx=(0, 8))

        self._theme_btn = ctk.CTkButton(
            right_frame, text="☾ Dark Mode", font=_FONTS["toggle_lbl"],
            width=110, height=28, corner_radius=14, fg_color="transparent", border_width=1,
            command=self._toggle_theme,
        )
        self._theme_btn.pack(side="left")

        # ── Body ─────────────────────────────────────────────────────────────
        self._body = ctk.CTkFrame(self, corner_radius=0, fg_color="transparent")
        self._body.grid(row=1, column=0, sticky="nsew")
        self._body.grid_rowconfigure(0, weight=1)
        self._body.grid_columnconfigure(1, weight=1)

        # Sidebar (hidden initially)
        self._sidebar = ctk.CTkFrame(self._body, width=0, corner_radius=0)
        self._sidebar.grid(row=0, column=0, sticky="ns")
        self._sidebar.grid_propagate(False)
        self._sidebar.grid_rowconfigure(1, weight=1)
        self._sidebar.grid_columnconfigure(0, weight=1)

        # Sidebar header row
        self._sidebar_header = ctk.CTkFrame(self._sidebar, height=90, corner_radius=0)
        self._sidebar_header.grid(row=0, column=0, sticky="ew")
        self._sidebar_header.grid_propagate(False)

        self._sidebar_title = ctk.CTkLabel(
            self._sidebar_header, text="💬  Sessions",
            font=_FONTS["sidebar_title"], anchor="w",
        )
        self._sidebar_title.pack(side="top", padx=12, pady=(10, 4), anchor="w")

        self._new_chat_btn = ctk.CTkButton(
            self._sidebar_header, text="+ New Chat",
            font=_FONTS["sidebar_note"], height=28, corner_radius=8,
            command=self._on_new_chat,
        )
        self._new_chat_btn.pack(fill="x", padx=8, pady=(0, 8))

        self._sidebar_scroll = ctk.CTkScrollableFrame(self._sidebar, corner_radius=0)
        self._sidebar_scroll.grid(row=1, column=0, sticky="nsew")
        self._sidebar_scroll.grid_columnconfigure(0, weight=1)

        # Chat frame
        self._chat_frame = ctk.CTkScrollableFrame(
            self._body, corner_radius=0, scrollbar_button_color="#333333",
        )
        self._chat_frame.grid(row=0, column=1, sticky="nsew")
        self._chat_frame.grid_columnconfigure(0, weight=1)

        # ── Status bar ───────────────────────────────────────────────────────
        self._status_bar = ctk.CTkFrame(self, height=24, corner_radius=0)
        self._status_bar.grid(row=2, column=0, sticky="ew")
        self._status_bar.grid_propagate(False)
        self._status_label = ctk.CTkLabel(self._status_bar, text="Ready", font=_FONTS["status"], anchor="w")
        self._status_label.pack(side="left", padx=14)

        # ── Input bar ────────────────────────────────────────────────────────
        self._input_bar = ctk.CTkFrame(self, height=68, corner_radius=0)
        self._input_bar.grid(row=3, column=0, sticky="ew")
        self._input_bar.grid_propagate(False)
        self._input_bar.grid_columnconfigure(0, weight=1)

        self._input_box = ctk.CTkEntry(
            self._input_bar, placeholder_text="Type a message…",
            font=_FONTS["input"], height=42, corner_radius=21, border_width=1,
        )
        self._input_box.grid(row=0, column=0, padx=(14, 8), pady=13, sticky="ew")

        self._send_btn = ctk.CTkButton(
            self._input_bar, text="Send", font=_FONTS["button"],
            width=72, height=42, corner_radius=21, command=self._on_send,
        )
        self._send_btn.grid(row=0, column=1, padx=(0, 8), pady=13)

        self._attach_btn = ctk.CTkButton(
            self._input_bar, text="📎", font=("Georgia", 18),
            width=46, height=42, corner_radius=21, command=self._on_attach,
        )
        self._attach_btn.grid(row=0, column=2, padx=(0, 8), pady=13)

        self._mic_btn = ctk.CTkButton(
            self._input_bar, text="🎤", font=("Georgia", 18),
            width=46, height=42, corner_radius=21, command=self._on_mic,
        )
        self._mic_btn.grid(row=0, column=3, padx=(0, 14), pady=13)

    # -----------------------------------------------------------------------
    # Sidebar — Sessions
    # -----------------------------------------------------------------------

    def _toggle_sidebar(self) -> None:
        if self._sidebar_open:
            self._sidebar.configure(width=0)
        else:
            self._populate_sidebar()
            self._sidebar.configure(width=self._SIDEBAR_WIDTH)
        self._sidebar_open = not self._sidebar_open

    def _populate_sidebar(self) -> None:
        """Fetch sessions from DB and render them as rows with delete buttons."""
        t = _DARK if self._is_dark else _LIGHT

        for widget in self._sidebar_scroll.winfo_children():
            widget.destroy()

        self._sidebar.configure(fg_color=t["sidebar_bg"], border_color=t["sidebar_border"])
        self._sidebar_header.configure(fg_color=t["sidebar_bg"])
        self._sidebar_title.configure(text_color=t["title_txt"])
        self._sidebar_scroll.configure(fg_color=t["sidebar_bg"])
        self._new_chat_btn.configure(
            fg_color=t["sidebar_btn"], hover_color=t["sidebar_btn_hover"],
            text_color=t["sidebar_btn_txt"],
        )

        try:
            sessions = list_sessions(self.assistant.db_path)
        except Exception as exc:
            logger.error(f"Sidebar: could not fetch sessions: {exc}")
            sessions = []

        if not sessions:
            ctk.CTkLabel(
                self._sidebar_scroll, text="No sessions yet.\nStart chatting!",
                font=_FONTS["sidebar_note"], text_color=t["sidebar_btn_txt"],
                anchor="w", justify="left",
            ).pack(padx=10, pady=16, anchor="w")
            return

        active_sid = self.assistant.session_id

        for sess in sessions:
            sid    = sess["id"]
            title  = sess["title"] or "New Chat"
            is_active = (sid == active_sid)

            row = ctk.CTkFrame(self._sidebar_scroll, fg_color="transparent")
            row.pack(fill="x", padx=4, pady=(3, 0))
            row.grid_columnconfigure(0, weight=1)

            # Session button — clicking loads that session
            sess_btn = ctk.CTkButton(
                row, text=f"{'▶ ' if is_active else ''}{title}",
                font=_FONTS["sidebar_note"],
                fg_color=t["sidebar_btn"] if not is_active else t["dot_color"],
                hover_color=t["sidebar_btn_hover"],
                text_color=t["sidebar_btn_txt"],
                anchor="w", corner_radius=8, height=36,
                command=lambda s=sid: self._on_session_click(s),
            )
            sess_btn.grid(row=0, column=0, sticky="ew")

            # 🗑 delete button
            del_btn = ctk.CTkButton(
                row, text="🗑", font=("Georgia", 12),
                width=32, height=36, corner_radius=8,
                fg_color=t["sidebar_del"],
                hover_color=t["sidebar_del_hover"],
                text_color=t["sidebar_del_txt"],
                command=lambda s=sid: self._on_delete_session(s),
            )
            del_btn.grid(row=0, column=1, padx=(4, 0))

    def _on_session_click(self, session_id: int) -> None:
        """Switch to an existing session and reload its chat history."""
        if self._is_busy:
            return
        self.assistant.switch_session(session_id)
        self._load_session_into_chat()
        # Close sidebar
        self._sidebar.configure(width=0)
        self._sidebar_open = False

    def _on_delete_session(self, session_id: int) -> None:
        """Delete a session from the DB and refresh the sidebar."""
        try:
            delete_session(self.assistant.db_path, session_id)
        except Exception as exc:
            logger.error(f"Delete session #{session_id}: {exc}")
        # If we deleted the active session, create a fresh one
        if session_id == self.assistant.session_id:
            new_sid = create_session(self.assistant.db_path)
            self.assistant.switch_session(new_sid)
            self._clear_chat()
            self._append_bubble("assistant", "New session started. How can I help?")
        self._populate_sidebar()

    def _on_new_chat(self) -> None:
        """Create a new session and switch to it."""
        new_sid = create_session(self.assistant.db_path, title="New Chat")
        self.assistant.switch_session(new_sid)
        self._clear_chat()
        self._append_bubble("assistant", "New session started. How can I help?")
        self._populate_sidebar()
        # Auto-rename session based on first user message (done in _dispatch_user_message)

    # -----------------------------------------------------------------------
    # Chat history loading
    # -----------------------------------------------------------------------

    def _load_session_into_chat(self) -> None:
        """Clear the chat frame and re-populate it from DB history."""
        self._clear_chat()
        try:
            history = load_conversation_history(
                self.assistant.db_path, self.assistant.session_id, limit=60
            )
        except Exception as exc:
            logger.error(f"_load_session_into_chat: {exc}")
            history = []

        if not history:
            self._append_bubble(
                "assistant",
                "Hello! I'm your offline AI assistant. "
                "Ask me about the weather, get directions, set reminders, or just chat!"
            )
        else:
            for turn in history:
                self._append_bubble(turn["role"], turn["content"])

    def _clear_chat(self) -> None:
        """Destroy all widgets in the chat frame."""
        for widget in self._chat_frame.winfo_children():
            widget.destroy()

    # -----------------------------------------------------------------------
    # Memory modal — Notes & Reminders with delete buttons
    # -----------------------------------------------------------------------

    def _open_memory_modal(self) -> None:
        """Open a Toplevel window listing Notes and Reminders with delete buttons."""
        t = _DARK if self._is_dark else _LIGHT

        modal = ctk.CTkToplevel(self)
        modal.title("Memory — Notes & Reminders")
        modal.geometry("540x600")
        modal.resizable(True, True)
        modal.configure(fg_color=t["modal_bg"])
        modal.grab_set()   # make it modal

        # Outer scrollable container
        scroll = ctk.CTkScrollableFrame(modal, fg_color=t["modal_bg"], corner_radius=0)
        scroll.pack(fill="both", expand=True, padx=0, pady=0)
        scroll.grid_columnconfigure(0, weight=1)

        def _refresh():
            for w in scroll.winfo_children():
                w.destroy()
            _build_content()

        def _build_content():
            # ── Notes section ──────────────────────────────────────────
            ctk.CTkLabel(scroll, text="📝  Notes", font=_FONTS["sidebar_title"],
                         text_color=t["title_txt"], anchor="w").pack(fill="x", padx=16, pady=(16, 4))

            try:
                result = NoteSkill(self.assistant.db_path).list_all({}, limit=50)
                notes  = result.get("notes", [])
            except Exception:
                notes = []

            if not notes:
                ctk.CTkLabel(scroll, text="  No notes saved.", font=_FONTS["sidebar_note"],
                             text_color=t["status_txt"], anchor="w").pack(fill="x", padx=20)
            else:
                for note in notes:
                    self._modal_item_row(scroll, t,
                                         label=f"📌 {note['title']}",
                                         sublabel=note.get("preview", ""),
                                         on_delete=lambda nid=note["id"]: _delete_note(nid))

            # ── Reminders section ──────────────────────────────────────
            ctk.CTkLabel(scroll, text="⏰  Reminders", font=_FONTS["sidebar_title"],
                         text_color=t["title_txt"], anchor="w").pack(fill="x", padx=16, pady=(20, 4))

            try:
                result = ReminderSkill(self.assistant.db_path).list_upcoming(limit=50)
                reminders = result.get("reminders", [])
            except Exception:
                reminders = []

            if not reminders:
                ctk.CTkLabel(scroll, text="  No upcoming reminders.", font=_FONTS["sidebar_note"],
                             text_color=t["status_txt"], anchor="w").pack(fill="x", padx=20)
            else:
                for rem in reminders:
                    label    = f"🔔 {rem['task']}"
                    sublabel = f"{rem['date']} at {rem['time']}"
                    self._modal_item_row(scroll, t, label=label, sublabel=sublabel,
                                         on_delete=lambda rid=rem["id"]: _delete_reminder(rid))

        def _delete_note(note_id: int) -> None:
            try:
                NoteSkill(self.assistant.db_path).delete(note_id)
            except Exception as exc:
                logger.error(f"Modal: delete note #{note_id}: {exc}")
            _refresh()

        def _delete_reminder(reminder_id: int) -> None:
            try:
                ReminderSkill(self.assistant.db_path).mark_done(reminder_id)
            except Exception as exc:
                logger.error(f"Modal: delete reminder #{reminder_id}: {exc}")
            _refresh()

        _build_content()

    def _modal_item_row(self, parent, t: dict, label: str, sublabel: str, on_delete) -> None:
        """Render a single item row (label + delete button) inside the modal."""
        row = ctk.CTkFrame(parent, fg_color=t["sidebar_btn"], corner_radius=8)
        row.pack(fill="x", padx=16, pady=(0, 6))
        row.grid_columnconfigure(0, weight=1)

        text_col = ctk.CTkFrame(row, fg_color="transparent")
        text_col.grid(row=0, column=0, sticky="ew", padx=10, pady=8)
        ctk.CTkLabel(text_col, text=label, font=_FONTS["sidebar_note"],
                     text_color=t["sidebar_btn_txt"], anchor="w").pack(anchor="w")
        if sublabel:
            ctk.CTkLabel(text_col, text=sublabel, font=_FONTS["timestamp"],
                         text_color=t["status_txt"], anchor="w").pack(anchor="w")

        ctk.CTkButton(
            row, text="🗑", font=("Georgia", 13), width=32, height=36,
            corner_radius=8,
            fg_color=t["del_btn"],
            hover_color=t["del_btn_hover"],
            text_color=t["del_btn_txt"],
            command=on_delete,
        ).grid(row=0, column=1, padx=(0, 8), pady=8)

    # -----------------------------------------------------------------------
    # Theme
    # -----------------------------------------------------------------------

    def _apply_theme(self) -> None:
        t = _DARK if self._is_dark else _LIGHT
        ctk.set_appearance_mode("dark" if self._is_dark else "light")

        self.configure(fg_color=t["window_bg"])
        self._header.configure(fg_color=t["header_bg"], border_color=t["header_border"])
        self._title_label.configure(text_color=t["title_txt"])
        self._dot.configure(text_color=t["dot_color"])
        self._theme_btn.configure(text_color=t["title_txt"], border_color=t["header_border"])
        self._voice_btn.configure(text_color=t["title_txt"], border_color=t["header_border"])
        self._memory_btn.configure(text_color=t["title_txt"], border_color=t["header_border"])
        self._sidebar_btn.configure(text_color=t["title_txt"])

        self._body.configure(fg_color=t["window_bg"])
        self._chat_frame.configure(fg_color=t["chat_bg"])

        if self._sidebar_open:
            self._sidebar.configure(fg_color=t["sidebar_bg"], border_color=t["sidebar_border"])
            self._sidebar_header.configure(fg_color=t["sidebar_bg"])
            self._sidebar_title.configure(text_color=t["title_txt"])
            self._sidebar_scroll.configure(fg_color=t["sidebar_bg"])

        self._status_bar.configure(fg_color=t["header_bg"])
        self._status_label.configure(text_color=t["status_txt"])
        self._input_bar.configure(fg_color=t["header_bg"])
        self._input_box.configure(fg_color=t["input_bg"], border_color=t["input_border"],
                                  text_color=t["input_txt"], placeholder_text_color=t["input_ph"])
        self._send_btn.configure(fg_color=t["send_btn"], hover_color=t["send_btn_hover"],
                                 text_color=t["send_btn_txt"])

        files_loaded = bool(self._pending_files)
        self._attach_btn.configure(
            fg_color=t["attach_active"]      if files_loaded else t["attach_btn"],
            hover_color=t["attach_active"]   if files_loaded else t["attach_btn_hover"],
            text_color=t["attach_active_txt"] if files_loaded else t["attach_btn_txt"],
        )
        self._mic_btn.configure(
            fg_color=t["mic_active"]      if self._mic_active else t["mic_btn"],
            hover_color=t["mic_active"]   if self._mic_active else t["mic_btn_hover"],
            text_color=t["mic_active_txt"] if self._mic_active else t["mic_btn_txt"],
        )

    def _toggle_voice(self) -> None:
        self._auto_speak = not self._auto_speak
        self._voice_btn.configure(text="Voice: ON" if self._auto_speak else "Voice: OFF")

    def _toggle_theme(self) -> None:
        self._is_dark = not self._is_dark
        self._theme_btn.configure(text="☾ Dark Mode" if self._is_dark else "☀ Light Mode")
        self._apply_theme()

    def _play_specific_chat(self, text: str) -> None:
        threading.Thread(target=lambda: self.speaker.speak(text), daemon=True).start()

    # -----------------------------------------------------------------------
    # Bubble / card helpers
    # -----------------------------------------------------------------------

    def _schedule(self, fn: Any) -> None:
        self.after(0, fn)

    def _set_status(self, text: str) -> None:
        self._status_label.configure(text=text)

    def _append_bubble(self, role: str, text: str) -> _ChatBubble:
        t   = _DARK if self._is_dark else _LIGHT
        row = ctk.CTkFrame(self._chat_frame, fg_color="transparent")
        row.pack(fill="x", padx=16, pady=(6, 2))
        bubble = _ChatBubble(row, role=role, text=text, theme=t,
                             speaker_callback=self._play_specific_chat)
        bubble.pack(side="right" if role == "user" else "left")
        self._chat_frame.after(50, self._scroll_to_bottom)
        return bubble

    def _append_weather_bubble(self, data: dict) -> None:
        t   = _DARK if self._is_dark else _LIGHT
        row = ctk.CTkFrame(self._chat_frame, fg_color="transparent")
        row.pack(fill="x", padx=16, pady=(4, 6))
        _WeatherBubble(row, data=data, theme=t).pack(side="left", anchor="w")
        self._chat_frame.after(50, self._scroll_to_bottom)

    def _append_map_bubble(self, data: dict) -> None:
        t   = _DARK if self._is_dark else _LIGHT
        row = ctk.CTkFrame(self._chat_frame, fg_color="transparent")
        row.pack(fill="x", padx=16, pady=(4, 6))
        _MapBubble(row, data=data, theme=t).pack(side="left", anchor="w")
        self._chat_frame.after(80, self._scroll_to_bottom)

    def _scroll_to_bottom(self) -> None:
        try:
            self._chat_frame._parent_canvas.yview_moveto(1.0)
        except Exception:
            pass

    def _remove_thinking_bubble(self) -> None:
        if self._thinking_bubble is not None:
            try:
                self._thinking_bubble.master.destroy()
            except Exception:
                pass
            self._thinking_bubble = None

    def _lock_input(self) -> None:
        self._is_busy = True
        self._input_box.configure(state="disabled")
        self._send_btn.configure(state="disabled")
        self._attach_btn.configure(state="disabled")
        self._mic_btn.configure(state="disabled")

    def _unlock_input(self) -> None:
        self._is_busy = False
        self._input_box.configure(state="normal")
        self._send_btn.configure(state="normal")
        self._attach_btn.configure(state="normal")
        self._mic_btn.configure(state="normal")
        self._input_box.focus()

    # -----------------------------------------------------------------------
    # Event handlers
    # -----------------------------------------------------------------------

    def _on_send(self) -> None:
        if self._is_busy:
            return
        text = self._input_box.get().strip()
        if not text:
            return
        self._input_box.delete(0, "end")
        self._dispatch_user_message(text)

    def _on_mic(self) -> None:
        if self._is_busy:
            return
        self._mic_active = True
        self._apply_theme()
        self._lock_input()
        self._schedule(lambda: self._set_status("🎤  Listening…"))

        def _listen_worker():
            try:
                recognised = self.listener.listen()
            except Exception as exc:
                logger.error(f"Listener error: {exc}")
                recognised = ""
            finally:
                self._mic_active = False
                self._schedule(self._apply_theme)

            if recognised and recognised.strip():
                self._schedule(lambda: self._dispatch_user_message(recognised))
            else:
                self._schedule(lambda: self._set_status("Ready — nothing heard"))
                self._schedule(self._unlock_input)

        threading.Thread(target=_listen_worker, name="MicThread", daemon=True).start()

    def _on_attach(self) -> None:
        """
        Open a multi-file dialog.  Accepts .txt, .pdf, .csv, .xlsx, images.
        Selected files are queued in self._pending_files.
        The attach button turns green to indicate pending files.
        """
        if self._is_busy:
            return

        filepaths = filedialog.askopenfilenames(
            title="Attach files",
            filetypes=[
                ("Supported files", "*.txt *.pdf *.csv *.xlsx *.xls *.png *.jpg *.jpeg"),
                ("Text",  "*.txt"),
                ("PDF",   "*.pdf"),
                ("CSV",   "*.csv"),
                ("Excel", "*.xlsx *.xls"),
                ("Image", "*.png *.jpg *.jpeg"),
                ("All",   "*.*"),
            ],
        )

        if not filepaths:
            return

        # Accumulate (don't replace) so multiple presses stack files
        self._pending_files.extend(list(filepaths))
        self._apply_theme()   # turns button green

        filenames = ", ".join(os.path.basename(fp) for fp in filepaths)
        self._append_bubble("user", f"📎 Attached: {filenames}")
        self._append_bubble(
            "assistant",
            f"I've queued {len(filepaths)} file(s) for analysis. "
            "They will be processed and added to this session's memory when you send your next message."
        )
        logger.info(f"Queued {len(filepaths)} file(s): {filenames}")

    # -----------------------------------------------------------------------
    # Core dispatch — streaming with RAG
    # -----------------------------------------------------------------------

    def _dispatch_user_message(self, text: str) -> None:
        """
        Stream the response.  Passes self._pending_files to the backend,
        then clears the pending list so files are only ingested once.
        """
        # Capture & clear pending files
        file_paths         = list(self._pending_files)
        self._pending_files = []
        self._apply_theme()   # resets attach button colour

        self._append_bubble("user", text)
        self._lock_input()
        self._set_status("⏳  Thinking…")
        self._thinking_bubble = self._append_bubble("thinking", "Thinking…")

        # Auto-rename the session after the first user message
        self._maybe_rename_session(text)

        def _stream_worker():
            stream_bubble: _ChatBubble | None = None
            accumulated = ""

            try:
                for item in self.assistant.respond_stream_with_data(
                    text,
                    file_paths=file_paths,
                ):
                    # ── Sentinel ──────────────────────────────────────────
                    if isinstance(item, dict) and item.get("__stream_done__"):
                        task_data = item.get("task_data")

                        def _on_done(td=task_data, ab=stream_bubble, full=accumulated):
                            self._remove_thinking_bubble()
                            if ab is not None:
                                ab.finalize(full)
                            else:
                                self._append_bubble("assistant", full or "(no response)")

                            if td and td.get("status") == "success":
                                intent = td.get("intent")
                                if intent == "search_travel":
                                    self._append_map_bubble(td)
                                elif intent == "get_weather":
                                    self._append_weather_bubble(td)

                            self._set_status("Ready")
                            self._unlock_input()

                            if self._auto_speak and full:
                                try:
                                    self.speaker.speak(full)
                                except Exception as exc:
                                    logger.warning(f"speaker.speak() error: {exc}")

                        self._schedule(_on_done)
                        return

                    # ── Text token ────────────────────────────────────────
                    if isinstance(item, str):
                        accumulated += item

                        if stream_bubble is None:
                            def _create_bubble(first_tok=accumulated):
                                nonlocal stream_bubble
                                self._remove_thinking_bubble()
                                stream_bubble = self._append_bubble("assistant", first_tok)
                                self._set_status("⌨  Streaming…")

                            self._schedule(_create_bubble)
                            time.sleep(0.05)
                        else:
                            acc_snap = accumulated
                            def _update(snap=acc_snap, bub=stream_bubble):
                                if bub is not None:
                                    bub.update_text(snap)
                                    self._scroll_to_bottom()
                            self._schedule(_update)

            except Exception as exc:
                logger.error(f"_stream_worker error: {exc}")
                def _on_error():
                    self._remove_thinking_bubble()
                    self._append_bubble("assistant", "I encountered an error. Please try again.")
                    self._set_status("Ready")
                    self._unlock_input()
                self._schedule(_on_error)

        threading.Thread(target=_stream_worker, name="StreamThread", daemon=True).start()

    def _maybe_rename_session(self, text: str) -> None:
        """
        Rename the session to the first 40 chars of the user's first message
        if the session is still called "New Chat".
        """
        try:
            from memory.database import list_sessions
            sessions = list_sessions(self.assistant.db_path)
            for sess in sessions:
                if sess["id"] == self.assistant.session_id:
                    if sess["title"] in ("New Chat", ""):
                        title = text[:40] + ("…" if len(text) > 40 else "")
                        rename_session(self.assistant.db_path, self.assistant.session_id, title)
                    break
        except Exception:
            pass

    # -----------------------------------------------------------------------
    # Close
    # -----------------------------------------------------------------------

    def _on_close(self) -> None:
        logger.info("GUI close requested.")

        def _goodbye():
            try:
                self.speaker.speak("Goodbye!")
                self.speaker.wait_until_done()
                self.speaker.shutdown()
            except Exception as exc:
                logger.warning(f"Shutdown TTS error: {exc}")
            finally:
                self._schedule(self.destroy)

        threading.Thread(target=_goodbye, name="ShutdownThread", daemon=True).start()
