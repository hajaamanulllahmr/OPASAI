import threading
import queue
import time
from typing import Tuple, Optional

# ── TTS setup ─────────────────────────────────────────────────────────────────

_tts_engine    = None
_tts_lock      = threading.Lock()
_TTS_AVAILABLE = False

try:
    import pyttsx3
    _TTS_AVAILABLE = True
except ImportError:
    pass


def _get_tts_engine():
    """
    Lazily initialise the pyttsx3 engine (once per process).
    Prefers a female voice on Windows when available for a friendlier feel.
    """
    global _tts_engine, _TTS_AVAILABLE
    if _tts_engine is not None:
        return _tts_engine
    if not _TTS_AVAILABLE:
        return None

    try:
        engine = pyttsx3.init()

        # Prefer female voice on Windows — feels more like a personal assistant
        voices = engine.getProperty("voices") or []
        for voice in voices:
            if any(name in (voice.name or "").lower() for name in ("zira", "hazel", "susan", "female")):
                engine.setProperty("voice", voice.id)
                break

        engine.setProperty("rate", 170)   # comfortable speaking speed
        _tts_engine = engine
        return engine

    except Exception as error:
        print(f"[TTS] Failed to initialise: {error}")
        _TTS_AVAILABLE = False
        return None


# ── STT setup ─────────────────────────────────────────────────────────────────

_whisper_model  = None
_whisper_loaded = False
_WHISPER_OK     = False
_SPHINX_OK      = False


def _try_load_whisper() -> bool:
    """
    Load the faster-whisper tiny model on first use.
    The tiny model is fast on CPU, uses ~200 MB of RAM, and is good enough
    for clear microphone input.
    """
    global _whisper_model, _whisper_loaded, _WHISPER_OK
    if _whisper_loaded:
        return _WHISPER_OK

    _whisper_loaded = True
    try:
        from faster_whisper import WhisperModel  # type: ignore
        _whisper_model = WhisperModel(
            "tiny",
            device="cpu",
            compute_type="int8",        # lowest memory / fastest on CPU
            download_root=None,          # uses HuggingFace default cache
            local_files_only=False,      # download automatically if not cached
        )
        _WHISPER_OK = True
        print("[STT] faster-whisper tiny model loaded ✓")
    except Exception as error:
        print(f"[STT] faster-whisper unavailable: {error}")
        _WHISPER_OK = False

    return _WHISPER_OK


def _try_sphinx() -> bool:
    """Check whether PocketSphinx (offline fallback) is available."""
    global _SPHINX_OK
    try:
        import speech_recognition as sr   # type: ignore
        import pocketsphinx               # type: ignore  # noqa: F401
        _SPHINX_OK = True
    except ImportError:
        _SPHINX_OK = False
    return _SPHINX_OK


# Detect library availability at import time.
# The Whisper model itself loads lazily on first use.
try:
    from faster_whisper import WhisperModel  # type: ignore  # noqa: F401
    _WHISPER_OK = True
except ImportError:
    _WHISPER_OK = False

_try_sphinx()


# ── Public availability helpers ───────────────────────────────────────────────

def tts_available() -> bool:
    """True if text-to-speech is available."""
    return _TTS_AVAILABLE


def stt_available() -> bool:
    """True if at least one speech-to-text engine is available."""
    return _WHISPER_OK or _SPHINX_OK


# ── Speaking ──────────────────────────────────────────────────────────────────

def speak(text: str):
    """
    Speak `text` aloud and wait until it finishes.
    Does nothing if TTS isn't available or text is empty.
    Text is capped at 800 characters to avoid very long utterances.
    """
    if not text or not _TTS_AVAILABLE:
        return

    with _tts_lock:
        engine = _get_tts_engine()
        if engine is None:
            return
        try:
            engine.say(str(text)[:800])
            engine.runAndWait()
        except Exception as error:
            print(f"[TTS] Speak error: {error}")


def speak_async(text: str):
    """
    Speak `text` in the background without blocking the caller.
    Useful in the web server so the API response doesn't hang while
    the assistant is still speaking.
    """
    if not text or not _TTS_AVAILABLE:
        return
    thread = threading.Thread(target=speak, args=(text,), daemon=True, name="TTSSpeaker")
    thread.start()


# ── Listening ─────────────────────────────────────────────────────────────────

def _transcribe_audio_array(audio_array) -> Tuple[str, Optional[str]]:
    """
    Transcribe a float32 numpy array (16 kHz, mono) using faster-whisper.
    Returns (transcript, error_message_or_None).
    """
    global _whisper_model

    if not _try_load_whisper():
        return "", "The faster-whisper model failed to load."

    try:
        import numpy as np  # type: ignore
        audio = audio_array.astype("float32")

        segments, _info = _whisper_model.transcribe(
            audio,
            language="en",
            beam_size=1,                                           # fastest setting
            vad_filter=True,                                       # skip silent sections
            vad_parameters={"min_silence_duration_ms": 300},
        )
        transcript = " ".join(segment.text for segment in segments).strip()

        # Whisper sometimes hallucinates common words when audio is silent
        hallucinations = {"you", "thank you", "thanks", ".", "...", ""}
        if transcript.lower() in hallucinations:
            return "", "Couldn't make out what you said — please try again."

        return transcript, None

    except Exception as error:
        return "", f"Whisper transcription error: {error}"


def _listen_via_whisper(
    timeout_seconds: int = 10, max_phrase_seconds: int = 12
) -> Tuple[str, Optional[str]]:
    """
    Record from the microphone and transcribe using faster-whisper.
    Returns (transcript, error_message_or_None).
    """
    try:
        import sounddevice as sd  # type: ignore
        import numpy as np        # type: ignore
    except ImportError:
        return "", "sounddevice / numpy not installed — run:  pip install sounddevice numpy"

    SAMPLE_RATE = 16000
    print("[STT] Recording — speak now…")

    try:
        audio = sd.rec(
            int(max_phrase_seconds * SAMPLE_RATE),
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
        )
        sd.wait()
        audio_flat = audio.flatten()
    except Exception as error:
        return "", f"Microphone error: {error}"

    # Check that something was actually recorded (not just silence)
    import numpy as np
    rms = float(np.sqrt(np.mean(audio_flat ** 2)))
    if rms < 0.001:
        return "", "No audio detected — please check your microphone."

    return _transcribe_audio_array(audio_flat)


def _listen_via_sphinx(
    timeout_seconds: int = 10, max_phrase_seconds: int = 12
) -> Tuple[str, Optional[str]]:
    """
    Record from the microphone and transcribe offline using PocketSphinx.
    Returns (transcript, error_message_or_None).
    """
    try:
        import speech_recognition as sr  # type: ignore
    except ImportError:
        return "", "SpeechRecognition not installed — run:  pip install SpeechRecognition"

    recogniser = sr.Recognizer()
    recogniser.energy_threshold       = 300
    recogniser.dynamic_energy_threshold = True

    try:
        with sr.Microphone() as microphone:
            recogniser.adjust_for_ambient_noise(microphone, duration=0.5)
            print("[STT] Listening via PocketSphinx…")
            try:
                audio = recogniser.listen(
                    microphone,
                    timeout=timeout_seconds,
                    phrase_time_limit=max_phrase_seconds,
                )
            except sr.WaitTimeoutError:
                return "", "Listening timed out — no speech detected."
    except Exception as error:
        return "", f"Microphone error: {error}"

    try:
        transcript = recogniser.recognize_sphinx(audio)
        # Filter out hallucinated single-word results
        if not transcript or transcript.strip().lower() in ("you", "the", "a", ""):
            return "", "Couldn't understand — try speaking more clearly."
        return transcript.strip(), None
    except sr.UnknownValueError:
        return "", "Could not understand the audio."
    except Exception as error:
        return "", f"PocketSphinx error: {error}"


def listen_once(
    timeout_seconds: int = 10, max_phrase_seconds: int = 12
) -> Tuple[str, Optional[str]]:
    """
    Record one utterance from the microphone and return (transcript, error).
    Uses faster-whisper if available, falls back to PocketSphinx.
    If neither is installed, returns a clear installation guide in the error.
    """
    if _WHISPER_OK:
        return _listen_via_whisper(timeout_seconds, max_phrase_seconds)

    if _SPHINX_OK:
        return _listen_via_sphinx(timeout_seconds, max_phrase_seconds)

    return "", (
        "No speech recognition engine is installed.\n\n"
        "For best quality (recommended):\n"
        "    pip install faster-whisper sounddevice\n\n"
        "Lightweight alternative:\n"
        "    pip install SpeechRecognition pocketsphinx pyaudio"
    )


def transcribe(audio_array) -> str:
    """
    Transcribe a float32 numpy audio array.
    Used by the FastAPI server's /api/voice/transcribe endpoint.
    Returns the transcript string, or an empty string on failure.
    """
    transcript, error = _transcribe_audio_array(audio_array)
    if error and not transcript:
        print(f"[STT] Transcription error: {error}")
    return transcript


# ── Singleton wrapper (used by server.py) ─────────────────────────────────────

class VoiceHandler:
    """
    A thin object wrapper so server.py can call
    `get_voice_handler().stt_available` instead of the module-level functions.
    """

    @property
    def stt_available(self) -> bool:
        return stt_available()

    @property
    def tts_available(self) -> bool:
        return tts_available()

    def speak(self, text: str):
        speak(text)

    def speak_async(self, text: str):
        speak_async(text)

    def listen_once(self) -> Tuple[str, Optional[str]]:
        return listen_once()

    def transcribe(self, audio_array) -> str:
        return transcribe(audio_array)


_voice_handler = VoiceHandler()

def get_voice_handler() -> VoiceHandler:
    """Return the shared VoiceHandler instance."""
    return _voice_handler
