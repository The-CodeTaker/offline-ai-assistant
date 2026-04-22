"""
voice/speaker.py

VoiceSpeaker: Offline neural TTS via the standalone piper.exe binary + PyAudio.

WHY SUBPROCESS INSTEAD OF THE piper-tts PYTHON LIBRARY
-------------------------------------------------------
The `piper-tts` Python package on Windows ships its own bundled copy of
`espeak-ng` for phonemization.  Due to a known Windows path-resolution bug,
this bundled binary silently fails to locate its data directory, causing
`voice.synthesize()` to write only a 44-byte WAV header with no audio data.

Bypassing the Python library entirely and calling the standalone `piper.exe`
binary avoids this bug completely: the compiled binary uses its own internal
phonemizer with no `espeak-ng` dependency.

ARCHITECTURE
------------
  Main thread  →  speak(text)  →  Queue
  Worker thread: dequeue → subprocess piper.exe (stdin=text, output_file=WAV)
               → validate WAV size → read WAV header → play via PyAudio

PUBLIC API (unchanged from previous piper-tts implementation)
---------------------------------------------------
  speak(text)          — enqueue text; returns immediately
  wait_until_done()    — block until queue is fully drained
  shutdown()           — send _SHUTDOWN sentinel; join worker thread

PLACEMENT OF piper.exe
-----------------------
See the detailed instructions at the very bottom of this file.
"""

from __future__ import annotations

import os
import queue
import subprocess
import threading
import wave
from pathlib import Path
from typing import Union

import pyaudio
from utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Paths — all resolved relative to this file so cwd never matters
# ---------------------------------------------------------------------------

_MODULE_DIR = Path(__file__).parent

# Standalone piper.exe binary (see placement instructions at bottom of file)
PIPER_EXE   = _MODULE_DIR / "piper" / "piper.exe"

# Voice model — must match your downloaded .onnx + .onnx.json pair
MODEL_PATH  = _MODULE_DIR / "models" / "en_US-amy-medium.onnx"

# Temporary WAV written by piper.exe and read back by PyAudio each utterance
TEMP_WAV    = _MODULE_DIR / "models" / "debug_speech.wav"

# Sentinel pushed onto the queue to signal the worker to exit cleanly
_SHUTDOWN: object = object()

# A WAV with only a header and no audio data is exactly 44 bytes
_WAV_HEADER_ONLY = 44


# ---------------------------------------------------------------------------
# VoiceSpeaker
# ---------------------------------------------------------------------------

class VoiceSpeaker:
    """
    Thread-safe, non-blocking TTS speaker backed by piper.exe + PyAudio.

    The main thread only ever calls speak() / wait_until_done() / shutdown().
    All subprocess and PyAudio interactions happen exclusively inside the
    daemon worker thread.

    Usage:
        speaker = VoiceSpeaker()
        speaker.speak("Hello!")
        speaker.wait_until_done()   # optional: block until audio finishes
        speaker.shutdown()          # call before program exit
    """

    def __init__(
        self,
        piper_exe:  Path | str = PIPER_EXE,
        model_path: Path | str = MODEL_PATH,
        temp_wav:   Path | str = TEMP_WAV,
    ) -> None:
        self.piper_exe  = Path(piper_exe)
        self.model_path = Path(model_path)
        self.temp_wav   = Path(temp_wav)

        # Set True by the worker after successful init; read-only from main thread
        self.tts_available: bool = False

        self._queue: queue.Queue[Union[str, object]] = queue.Queue()

        self._worker_thread = threading.Thread(
            target=self._worker,
            name="VoiceSpeakerWorker",
            daemon=True,
        )
        self._worker_thread.start()
        logger.info("VoiceSpeaker worker thread started.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def speak(self, text: str) -> None:
        """
        Enqueue *text* for synthesis and playback.
        Returns immediately — synthesis happens in the background thread.
        """
        if not text or not text.strip():
            logger.warning("speak() called with empty text — skipping.")
            return
        logger.info(f"Queued: '{text[:80]}{'...' if len(text) > 80 else ''}'")
        self._queue.put(text.strip())

    def wait_until_done(self) -> None:
        """
        Block the calling thread until every queued utterance has finished
        playing.  Prevents mic/speaker overlap in the conversational loop.
        """
        self._queue.join()
        logger.info("All queued speech finished.")

    def shutdown(self) -> None:
        """
        Gracefully stop the worker thread and release audio resources.
        The instance must not be reused after this call.
        """
        logger.info("Shutting down VoiceSpeaker...")
        self._queue.put(_SHUTDOWN)
        self._worker_thread.join(timeout=10)
        if self._worker_thread.is_alive():
            logger.warning("Worker thread did not exit within 10 s timeout.")
        else:
            logger.info("VoiceSpeaker shut down cleanly.")

    # ------------------------------------------------------------------
    # Worker  (runs entirely in the background thread)
    # ------------------------------------------------------------------

    def _worker(self) -> None:
        """
        Background thread entry point.

        1. Validate piper.exe and model paths.
        2. Open PyAudio output stream.
        3. Loop: dequeue → if _SHUTDOWN break; else synthesize & play.
        4. Close PyAudio on exit.
        """
        pa: pyaudio.PyAudio | None    = None
        stream: pyaudio.Stream | None = None

        try:
            pa, stream = self._init_audio()
            self.tts_available = True
        except Exception as exc:
            logger.error(f"Audio initialisation failed — speech disabled: {exc}")
            self._drain_queue_silently()
            return

        logger.info("Audio engine ready. Entering playback loop.")

        while True:
            item = self._queue.get()
            try:
                if item is _SHUTDOWN:
                    logger.info("Shutdown sentinel received — exiting worker loop.")
                    break

                # Synthesize and play; stream may be reopened internally
                # if the WAV sample rate differs from the current stream rate.
                stream = self._synthesize_and_play(str(item), pa, stream)

            except Exception as exc:
                logger.error(f"Playback error: {exc}")
            finally:
                # Always call task_done() so wait_until_done() never deadlocks,
                # even when an exception is raised mid-utterance.
                self._queue.task_done()

        self._close_audio(stream, pa)
        logger.info("Worker thread exited cleanly.")

    # ------------------------------------------------------------------
    # Audio helpers  (called only from the worker thread)
    # ------------------------------------------------------------------

    def _init_audio(self) -> tuple[pyaudio.PyAudio, pyaudio.Stream]:
        """
        Validate file paths and open an initial PyAudio output stream.

        The stream is opened at 22050 Hz as a placeholder; it will be
        transparently reopened at the correct rate on the first utterance
        once the WAV header has been read.

        Raises:
            FileNotFoundError: If piper.exe, the model, or the config is absent.
        """
        if not self.piper_exe.exists():
            raise FileNotFoundError(
                f"piper.exe not found at: {self.piper_exe}\n"
                "See placement instructions at the bottom of voice/speaker.py."
            )
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Piper model not found at: {self.model_path}"
            )
        config_path = self.model_path.with_suffix(".onnx.json")
        if not config_path.exists():
            raise FileNotFoundError(
                f"Piper model config not found at: {config_path}\n"
                "The .onnx and .onnx.json files must sit in the same directory."
            )

        logger.info(f"piper.exe : {self.piper_exe}")
        logger.info(f"Model     : {self.model_path}")
        logger.info(f"Temp WAV  : {self.temp_wav}")

        pa = pyaudio.PyAudio()
        stream = pa.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=22050,          # placeholder; corrected on first utterance
            output=True,
            frames_per_buffer=2048,
        )
        logger.info("PyAudio initialised (rate syncs on first utterance).")
        return pa, stream

    def _synthesize_and_play(
        self,
        text:   str,
        pa:     pyaudio.PyAudio,
        stream: pyaudio.Stream,
    ) -> pyaudio.Stream:
        """
        Invoke piper.exe via subprocess, validate the output WAV, then play it.

        The subprocess call is equivalent to:
            echo "<text>" | piper.exe --model <model> --output_file <wav>

        piper.exe reads text from stdin and writes a fully-formed WAV file to
        --output_file, including a correct header that specifies sample rate,
        channels, and bit depth — so PyAudio needs no hardcoded assumptions.

        Returns the (possibly reopened) PyAudio stream so the worker loop can
        hold a reference to the latest stream handle.
        """
        log_text = text[:60] + ("..." if len(text) > 60 else "")
        logger.info(f"Synthesizing: '{log_text}'")

        # --- 1. Run piper.exe -------------------------------------------
        try:
            result = subprocess.run(
                [
                    str(self.piper_exe),
                    "--model",       str(self.model_path),
                    "--output_file", str(self.temp_wav),
                    "--length_scale", "0.8",
                ],
                input=text,           # text piped to piper's stdin
                capture_output=True,  # collect stderr for logging
                text=True,            # stdin/stderr as str
                encoding="utf-8",
                timeout=30,           # hard cap per utterance
            )
        except subprocess.TimeoutExpired:
            logger.error(f"piper.exe timed out (30 s) for: '{log_text}'")
            return stream
        except FileNotFoundError:
            logger.error(f"piper.exe missing at {self.piper_exe}.")
            return stream
        except Exception as exc:
            logger.error(f"Unexpected subprocess error: {exc}")
            return stream

        # Log piper's stderr at DEBUG (usually just progress lines)
        if result.stderr:
            for line in result.stderr.strip().splitlines():
                logger.debug(f"piper: {line}")

        if result.returncode != 0:
            logger.error(
                f"piper.exe returned code {result.returncode} "
                f"for: '{log_text}'"
            )
            return stream

        # --- 2. Validate the WAV ----------------------------------------
        if not self.temp_wav.exists():
            logger.error(f"{self.temp_wav} was not created by piper.exe.")
            return stream

        wav_size = os.path.getsize(self.temp_wav)
        logger.debug(f"WAV size: {wav_size} bytes")

        if wav_size <= _WAV_HEADER_ONLY:
            logger.error(
                f"piper.exe produced an empty WAV ({wav_size} B). "
                "Check that piper.exe is the correct Windows amd64 build "
                "and that the model path is correct."
            )
            return stream

        # --- 3. Read WAV header and play ---------------------------------
        try:
            with wave.open(str(self.temp_wav), "rb") as wf:
                sample_rate = wf.getframerate()
                n_channels  = wf.getnchannels()
                samp_width  = wf.getsampwidth()

                logger.debug(
                    f"WAV: {sample_rate} Hz | {n_channels} ch "
                    f"| {samp_width * 8}-bit"
                )

                # Reopen the PyAudio stream if the rate has changed.
                # This happens on the very first real utterance (placeholder
                # was 22050 Hz) and any time you switch models mid-session.
                current_rate = stream._rate  # type: ignore[attr-defined]
                if current_rate != sample_rate:
                    logger.info(
                        f"Reopening PyAudio stream: "
                        f"{current_rate} Hz → {sample_rate} Hz"
                    )
                    stream.stop_stream()
                    stream.close()
                    stream = pa.open(
                        format=pyaudio.get_format_from_width(samp_width),
                        channels=n_channels,
                        rate=sample_rate,
                        output=True,
                        frames_per_buffer=2048,
                    )

                # Stream PCM data to the speaker in chunks
                chunk = 1024
                data  = wf.readframes(chunk)
                while data:
                    stream.write(data)
                    data = wf.readframes(chunk)

            # Flush hardware buffer with silence to prevent end-of-utterance clipping
            stream.write(b"\x00" * 4096)
            logger.info("Playback complete.")

        except wave.Error as exc:
            logger.error(f"WAV read error: {exc}")
        except Exception as exc:
            logger.error(f"PyAudio playback error: {exc}")

        return stream

    @staticmethod
    def _close_audio(
        stream: pyaudio.Stream | None,
        pa:     pyaudio.PyAudio | None,
    ) -> None:
        """Safely stop, close, and terminate PyAudio resources."""
        try:
            if stream is not None:
                stream.stop_stream()
                stream.close()
                logger.info("PyAudio stream closed.")
        except Exception as exc:
            logger.warning(f"Error closing stream: {exc}")
        try:
            if pa is not None:
                pa.terminate()
                logger.info("PyAudio terminated.")
        except Exception as exc:
            logger.warning(f"Error terminating PyAudio: {exc}")

    def _drain_queue_silently(self) -> None:
        """
        Called only when audio init fails.
        Drains the queue so wait_until_done() callers never hang.
        Prints text to stdout as a fallback so output is not completely lost.
        """
        logger.warning("Draining speech queue silently (TTS unavailable).")
        while True:
            item = self._queue.get()
            self._queue.task_done()
            if item is _SHUTDOWN:
                break
            if isinstance(item, str):
                print(f"\n🤖 [TTS unavailable] {item}\n")


# ===========================================================================
# HOW TO INSTALL piper.exe  —  Step-by-step for Windows
# ===========================================================================
#
# STEP 1 — Download the binary
#   Go to: https://github.com/rhasspy/piper/releases
#   Download: piper_windows_amd64.zip
#             (use arm64 only if you are on a Windows ARM device)
#
# STEP 2 — Extract and place the files
#   Extract the zip.  Move the ENTIRE extracted folder's contents into:
#
#       offline-ai-assistant/voice/piper/
#
#   Required final layout:
#       voice/
#       ├── piper/
#       │   ├── piper.exe                        ← the binary
#       │   ├── onnxruntime.dll                  ← required DLL
#       │   ├── onnxruntime_providers_shared.dll ← required DLL
#       │   ├── piper_phonemize.dll              ← required DLL
#       │   └── ... (all other DLLs from the zip)
#       └── models/
#           ├── en_US-amy-medium.onnx
#           └── en_US-amy-medium.onnx.json
#
#   ⚠️  Do NOT move piper.exe out of its folder.
#      It loads DLLs from the same directory at runtime.
#      Moving it alone will cause an immediate crash.
#
# STEP 3 — Quick sanity test (run from the project root in cmd.exe)
#   echo Testing piper | voice\piper\piper.exe ^
#       --model voice\models\en_US-amy-medium.onnx ^
#       --output_file test_output.wav
#
#   Then check: dir test_output.wav
#   The file should be LARGER than 44 bytes.
#   Open it in Windows Media Player or VLC — you should hear speech.
#
# STEP 4 — Update requirements.txt
#   Remove the line:   piper-tts>=1.2.0
#   The subprocess approach needs NO Python TTS library.
#   PyAudio is still required for playback (already in requirements.txt).
#
# ===========================================================================
