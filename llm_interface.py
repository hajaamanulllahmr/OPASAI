

import json
import urllib.request
import urllib.error
from typing import Iterator

# ── Configuration ─────────────────────────────────────────────────────────────

OLLAMA_URL = "http://localhost:11434"
MODEL_NAME = "phi3"

# The personality / instruction block sent to the model before every chat.
# Keep it short — Phi-3 is small and too long a system prompt wastes tokens.
SYSTEM_PROMPT = (
    "You are OPASAI, a helpful personal AI assistant. "
    "You know everything about your owner from the context provided below. "
    "Answer questions about them accurately and conversationally. "
    "If a detail isn't in the context, simply say you don't have that information. "
    "Keep your answers short and natural — no markdown, no bullet points."
)

# Generation settings — tweak these if answers feel too long, too random, etc.
GENERATION_OPTIONS = {
    "temperature": 0.3,    # lower = more focused / deterministic
    "num_predict": 300,    # max tokens per reply — increase for longer answers
    "top_p"      : 0.9,
}


# ── Internal helper ───────────────────────────────────────────────────────────

def _post_json(endpoint: str, payload: dict, timeout_seconds: int = 90):
    """
    Send a JSON POST request to Ollama and return the raw response object.
    Raises urllib.error.URLError if Ollama isn't reachable.
    """
    body = json.dumps(payload).encode()
    request = urllib.request.Request(
        endpoint,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    return urllib.request.urlopen(request, timeout=timeout_seconds)


def _build_prompt(question: str, context: str) -> str:
    """
    Assemble the full prompt that gets sent to Phi-3.
    The context comes from the RAG engine (relevant profile snippets).
    """
    return (
        f"Context about the owner:\n{context}\n\n"
        f"Question: {question}\n\n"
        f"Answer:"
    )


# ── Health checks ─────────────────────────────────────────────────────────────

def is_ollama_running() -> bool:
    """
    Ping Ollama to see if it's up. Fast — 3 second timeout.
    Returns True if reachable, False otherwise.
    """
    try:
        urllib.request.urlopen(f"{OLLAMA_URL}/api/tags", timeout=3)
        return True
    except Exception:
        return False


def is_model_available(model: str = MODEL_NAME) -> bool:
    """
    Check that the requested model has actually been downloaded (ollama pull).
    Returns False if Ollama is unreachable or the model isn't listed.
    """
    try:
        response = urllib.request.urlopen(f"{OLLAMA_URL}/api/tags", timeout=5)
        data     = json.loads(response.read())
        # Model names in the list look like "phi3:latest" — strip the tag part
        names = [m.get("name", "").split(":")[0] for m in data.get("models", [])]
        return model in names
    except Exception:
        return False


def list_models() -> list:
    """
    Return all model names currently downloaded in Ollama.
    Returns an empty list if Ollama is unavailable.
    """
    try:
        response = urllib.request.urlopen(f"{OLLAMA_URL}/api/tags", timeout=5)
        data     = json.loads(response.read())
        return [m.get("name", "") for m in data.get("models", [])]
    except Exception:
        return []


# ── Inference ─────────────────────────────────────────────────────────────────

def chat(question: str, context: str) -> str:
    """
    Send a question (with context) to Phi-3 and wait for the full reply.
    Blocks until the model finishes generating — good for non-interactive use.
    Returns the answer as a plain string.
    """
    if not is_ollama_running():
        return (
            "⚠️  Ollama isn't running.\n"
            "Start it with:  ollama serve\n"
            "Then make sure Phi-3 is ready:  ollama pull phi3"
        )

    payload = {
        "model" : MODEL_NAME,
        "prompt": _build_prompt(question, context),
        "system": SYSTEM_PROMPT,
        "stream": False,
        "options": GENERATION_OPTIONS,
    }

    try:
        response = _post_json(f"{OLLAMA_URL}/api/generate", payload, timeout_seconds=120)
        data     = json.loads(response.read())
        return data.get("response", "").strip() or "(the model returned an empty response)"
    except urllib.error.URLError as error:
        return f"⚠️  Cannot reach Ollama: {error.reason}"
    except Exception as error:
        return f"⚠️  Something went wrong: {error}"


def chat_stream(question: str, context: str) -> Iterator[str]:
    """
    Send a question to Phi-3 and yield the answer one token at a time.
    This makes the reply appear word-by-word in the UI — much better UX
    than waiting several seconds for the whole answer to arrive at once.
    """
    if not is_ollama_running():
        yield "⚠️  Ollama isn't running. Start it with:  ollama serve"
        return

    payload = {
        "model" : MODEL_NAME,
        "prompt": _build_prompt(question, context),
        "system": SYSTEM_PROMPT,
        "stream": True,
        "options": GENERATION_OPTIONS,
    }

    try:
        response = _post_json(f"{OLLAMA_URL}/api/generate", payload, timeout_seconds=120)
        for raw_line in response:
            line = raw_line.strip()
            if not line:
                continue
            chunk = json.loads(line)
            token = chunk.get("response", "")
            if token:
                yield token
            if chunk.get("done"):
                break
    except urllib.error.URLError as error:
        yield f"⚠️  Ollama connection error: {error.reason}"
    except Exception as error:
        yield f"⚠️  Streaming error: {error}"
