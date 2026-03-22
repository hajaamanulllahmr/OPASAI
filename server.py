"""
server.py — OPASAI Web Server (FastAPI)
=========================================
Serves the web UI and provides a small REST API that the frontend talks to.
Run with:  python server.py   →  opens on http://localhost:8000

API endpoints
-------------
  GET  /                         — serves the web UI (index.html)
  GET  /api/status               — health check for all components
  POST /api/ask          {query} — blocking single-response endpoint
  GET  /api/stream?query=…       — Server-Sent Events streaming endpoint
  POST /api/voice/transcribe     — receive WAV audio, return transcript
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import json
import asyncio
from pathlib import Path
from typing import AsyncIterator

from fastapi              import FastAPI, Request
from fastapi.responses    import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles  import StaticFiles
import uvicorn

from assistant     import get_assistant
from llm_interface import is_ollama_running, is_model_available, chat_stream
from rag_engine    import get_rag_engine
from voice_handler import get_voice_handler

app       = FastAPI(title="OPASAI — Offline Personal AI Assistant")
assistant = get_assistant()
BASE_DIR  = Path(__file__).parent


# ── Static files ──────────────────────────────────────────────────────────────

static_dir = BASE_DIR / "static"
static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


# ── Start-up: build the RAG index before serving any requests ─────────────────

@app.on_event("startup")
async def on_startup():
    """Load the knowledge index when the server starts up."""
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, assistant.init_sync)
    print("[OPASAI] Knowledge index ready — server is accepting requests.")


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    """Serve the main web interface."""
    html_path = BASE_DIR / "static" / "index.html"
    if not html_path.exists():
        return HTMLResponse("<h1>index.html not found</h1>", status_code=404)
    return HTMLResponse(html_path.read_text(encoding="utf-8"))


@app.get("/api/status")
async def api_status():
    """Return the health of every OPASAI component as a JSON object."""
    return JSONResponse(assistant.status())


@app.post("/api/ask")
async def api_ask(request: Request):
    """
    Ask a question and get the full answer in one response.
    Body:  {"query": "What are my hobbies?"}
    """
    body     = await request.json()
    question = body.get("query", "").strip()

    if not question:
        return JSONResponse({"error": "Please include a 'query' field."}, status_code=400)

    loop   = asyncio.get_event_loop()
    answer = await loop.run_in_executor(None, assistant.ask, question)
    return JSONResponse({"query": question, "response": answer})


@app.get("/api/stream")
async def api_stream(query: str):
    """
    Stream the answer to `query` as Server-Sent Events (SSE).
    Each event is:  data: {"token": "..."}
    The final event is:  data: {"done": true, "full": "complete answer"}

    The frontend listens with EventSource and appends tokens as they arrive,
    giving that satisfying "typing" feel without any extra complexity.
    """
    async def generate_events() -> AsyncIterator[str]:
        context  = get_rag_engine().build_context(query)
        loop     = asyncio.get_event_loop()
        token_q  = asyncio.Queue()
        done_evt = asyncio.Event()

        def stream_in_background():
            for token in chat_stream(query, context):
                loop.call_soon_threadsafe(token_q.put_nowait, token)
            loop.call_soon_threadsafe(done_evt.set)

        loop.run_in_executor(None, stream_in_background)

        full_answer = ""
        while not done_evt.is_set() or not token_q.empty():
            try:
                token        = await asyncio.wait_for(token_q.get(), timeout=0.05)
                full_answer += token
                yield f"data: {json.dumps({'token': token})}\n\n"
            except asyncio.TimeoutError:
                continue   # no token yet — loop and wait

        yield f"data: {json.dumps({'done': True, 'full': full_answer})}\n\n"

    return StreamingResponse(
        generate_events(),
        media_type="text/event-stream",
        headers={
            "Cache-Control" : "no-cache",
            "X-Accel-Buffering": "no",   # prevents nginx from buffering SSE
        },
    )


@app.post("/api/voice/transcribe")
async def api_voice_transcribe(request: Request):
    """
    Accept raw WAV audio bytes (16 kHz mono) from the browser microphone
    and return the transcribed text.

    The frontend records audio with the Web Audio API, encodes it as WAV,
    and POSTs it here. We decode it and pass it to the offline STT engine.
    """
    voice = get_voice_handler()

    if not voice.stt_available:
        return JSONResponse(
            {"error": "Speech recognition isn't installed on this server."},
            status_code=503,
        )

    import numpy as np
    import io
    import wave

    raw_bytes = await request.body()

    try:
        with wave.open(io.BytesIO(raw_bytes)) as wav_file:
            raw_frames = wav_file.readframes(wav_file.getnframes())
            audio_int16 = np.frombuffer(raw_frames, dtype=np.int16)
            # Normalise to float32 in [-1, 1] as expected by faster-whisper
            audio_float32 = audio_int16.astype(np.float32) / 32768.0
    except Exception as error:
        return JSONResponse({"error": f"Could not read WAV data: {error}"}, status_code=400)

    loop       = asyncio.get_event_loop()
    transcript = await loop.run_in_executor(None, voice.transcribe, audio_float32)
    return JSONResponse({"transcript": transcript})


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n🤖  OPASAI is starting up on http://localhost:8000\n")
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)
