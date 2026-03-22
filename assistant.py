import threading
from typing import Callable, Iterator, Optional

from rag_engine    import get_rag_engine
from llm_interface import chat, chat_stream, is_ollama_running, is_model_available
import voice_handler as voice


class Assistant:
    """
    The main OPASAI assistant object.

    Typical usage
    -------------
        assistant = Assistant()
        assistant.init_sync()          # or assistant.init(on_done=callback)
        reply = assistant.ask("What are my hobbies?")
    """

    def __init__(self):
        self.rag   = get_rag_engine()
        self.ready = False            # becomes True once the RAG index is loaded

    # ──────────────────────────────────────────────────────────────────────────
    # Start-up
    # ──────────────────────────────────────────────────────────────────────────

    def init(self, on_done: Optional[Callable] = None) -> threading.Thread:
        """
        Load the RAG index in the background so the UI doesn't freeze.
        `on_done` is called (with no arguments) once loading is complete.
        Returns the background thread so callers can join() if needed.
        """
        def _load():
            self.rag.load()
            self.ready = True
            if on_done:
                on_done()

        loader = threading.Thread(target=_load, daemon=True, name="RAGLoader")
        loader.start()
        return loader

    def init_sync(self):
        """
        Load the RAG index synchronously (blocks until done).
        Use this in the CLI or FastAPI startup where blocking is fine.
        """
        self.rag.load()
        self.ready = True

    # ──────────────────────────────────────────────────────────────────────────
    # Answering questions
    # ──────────────────────────────────────────────────────────────────────────

    def ask(self, question: str) -> str:
        """
        Ask a question and get a complete answer back in one go.
        Returns an empty string if the question is blank.
        """
        if not question.strip():
            return ""
        context = self.rag.build_context(question)
        return chat(question, context)

    def ask_stream(self, question: str) -> Iterator[str]:
        """
        Ask a question and get the answer as a stream of word-by-word tokens.
        Useful for showing the reply as it's being generated (feels much faster).
        """
        if not question.strip():
            return
        context = self.rag.build_context(question)
        yield from chat_stream(question, context)

    # ──────────────────────────────────────────────────────────────────────────
    # Voice interaction
    # ──────────────────────────────────────────────────────────────────────────

    def voice_ask(self) -> tuple:
        """
        Listen for a spoken question, get the answer, and speak it aloud.
        Returns (transcript, answer) — both are strings.
        If nothing was heard, transcript will be empty and answer will
        contain a friendly error message.
        """
        result = voice.listen_once()

        # listen_once returns (text, error) but guard against unexpected shapes
        if isinstance(result, tuple) and len(result) >= 2:
            transcript, error = result[0], result[1]
        else:
            transcript, error = result, None

        if not transcript:
            return "", error or "Sorry, I didn't catch that. Please try again."

        answer = self.ask(transcript)
        voice.speak_async(answer)
        return transcript, answer

    # ──────────────────────────────────────────────────────────────────────────
    # Health check
    # ──────────────────────────────────────────────────────────────────────────

    def status(self) -> dict:
        """
        Returns a quick snapshot of what's working and what isn't.
        All values are booleans — True means that component is healthy.
        """
        return {
            "rag_ready"     : self.ready,
            "ollama_running": is_ollama_running(),
            "model_ready"   : is_model_available(),
            "tts_available" : voice.tts_available(),
            "stt_available" : voice.stt_available(),
        }


# ── Singleton ─────────────────────────────────────────────────────────────────
# One shared instance used across the whole app so the RAG index is only
# built once, not once per request.

_assistant = Assistant()

def get_assistant() -> Assistant:
    """Return the shared OPASAI assistant instance."""
    return _assistant

# Alias kept for backwards compatibility with server.py
get_aria = get_assistant
