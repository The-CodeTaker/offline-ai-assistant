"""
voice/listener.py

VoiceListener: Captures microphone input and converts it to text using
the SpeechRecognition library. Falls back to typed input if mic fails.
"""

import speech_recognition as sr
from utils.logger import get_logger

logger = get_logger(__name__)


class VoiceListener:
    """
    Listens for voice input via microphone and returns transcribed text.
    Falls back gracefully to keyboard input if the microphone is
    unavailable or speech cannot be recognized.
    """

    def __init__(self, timeout: int = 5, phrase_time_limit: int = 15):
        """
        Args:
            timeout (int): Seconds to wait for speech to begin before giving up.
            phrase_time_limit (int): Max seconds allowed for a single phrase.
        """
        self.recognizer = sr.Recognizer()
        # --- THE PATIENCE FIX ---
        self.recognizer.pause_threshold = 1.5       # Wait 2.5 seconds of silence before cutting off
        self.recognizer.non_speaking_duration = 1.0 # Minimum silence required to consider phrase complete
        self.recognizer.dynamic_energy_threshold = False # Stop auto-adjusting volume mid-sentence
        # ------------------------
        self.timeout = timeout
        self.phrase_time_limit = phrase_time_limit
        self.mic_available = self._check_microphone()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _check_microphone(self) -> bool:
        """
        Verifies that at least one microphone is accessible on this system.
        Sets self.mic_available and logs the result.
        """
        try:
            mic_list = sr.Microphone.list_microphone_names()
            if mic_list:
                logger.info(f"Microphone detected: {len(mic_list)} device(s) found.")
                return True
            else:
                logger.warning("No microphone devices found. Falling back to text input.")
                return False
        except Exception as e:
            logger.error(f"Microphone check failed: {e}")
            return False

    def _calibrate_ambient_noise(self, source: sr.Microphone) -> None:
        """
        Adjusts the recognizer's energy threshold to account for ambient
        noise in the current environment.
        """
        logger.info("Calibrating for ambient noise (1 second)...")
        self.recognizer.adjust_for_ambient_noise(source, duration=1)
        logger.info(f"Energy threshold set to {self.recognizer.energy_threshold:.2f}")

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def listen(self) -> str:
        """
        Primary entry point. Attempts voice recognition and returns the
        transcribed string. Falls back to typed input on any failure.

        Returns:
            str: The user's input, either from voice or keyboard.
        """
        if not self.mic_available:
            return self._text_fallback(reason="No microphone available")

        try:
            with sr.Microphone() as source:
                self._calibrate_ambient_noise(source)
                print("🎙️  Listening... (speak now)")
                logger.info("Listening for voice input...")

                audio = self.recognizer.listen(
                    source,
                    timeout=self.timeout,
                    phrase_time_limit=self.phrase_time_limit,
                )

            return self._recognize(audio)

        except sr.WaitTimeoutError:
            logger.warning("Listening timed out — no speech detected.")
            return self._text_fallback(reason="No speech detected within timeout")

        except OSError as e:
            # Covers mic unplugged / permission denied at the OS level
            logger.error(f"Microphone OS error: {e}")
            self.mic_available = False          # disable for remainder of session
            return self._text_fallback(reason="Microphone hardware error")

        except Exception as e:
            logger.error(f"Unexpected error during listening: {e}")
            return self._text_fallback(reason="Unexpected microphone error")

    def _recognize(self, audio: sr.AudioData) -> str:
        """
        Passes captured audio to Google's offline-compatible recognizer.
        Falls back to text input if recognition fails.

        Args:
            audio (sr.AudioData): The captured audio segment.

        Returns:
            str: Recognized text or fallback text input.
        """
        try:
            # Using Google Web Speech — swap for recognize_sphinx() for
            # a fully offline alternative (requires pocketsphinx).
            text = self.recognizer.recognize_google(audio)
            logger.info(f"Voice recognized: '{text}'")
            print(f"🗣️  You said: {text}")
            return text

        except sr.UnknownValueError:
            logger.warning("Speech was detected but could not be understood.")
            return self._text_fallback(reason="Could not understand speech")

        except sr.RequestError as e:
            # Network error reaching the recognition service
            logger.error(f"Recognition service error: {e}")
            return self._text_fallback(reason="Speech recognition service unavailable")

    def _text_fallback(self, reason: str = "Voice unavailable") -> str:
        """
        GUI-safe fallback. Returns an empty string instead of blocking 
        with terminal input() so the UI can unlock gracefully.
        """
        logger.warning(f"Voice failed [{reason}]. unlock gui.")
        # Return empty string. The GUI will see nothing was heard and unlock automatically.
        return ""