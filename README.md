# OPASAI — Offline Personal AI Assistant

> A fully offline personal AI assistant powered by Phi-3 (via Ollama).  
> Knows everything about you from your profile. No internet. No API keys. No cloud.

**By Haja Amanullah M R**

---

## What it does

OPASAI answers questions about you — your skills, work experience, projects,
education, hobbies, and career goals — using your own `profile.txt` as its
knowledge base. Every response is generated locally on your machine.

Ask things like:
- *"What are my technical skills?"*
- *"Tell me about my projects."*
- *"Where do I work?"*
- *"What are my career goals?"*

---

## Project Structure

```
OPASAI/
├── data/
│   └── profile.txt          ← Your personal profile (edit this to personalise)
├── static/
│   └── index.html           ← Web UI frontend
├── assistant.py             ← Core orchestrator (ties RAG + LLM + Voice)
├── rag_engine.py            ← TF-IDF retrieval engine (pure Python, no ML libs)
├── llm_interface.py         ← Ollama Phi-3 client (streaming)
├── voice_handler.py         ← Offline STT (Whisper) + TTS (pyttsx3)
├── server.py                ← FastAPI web server (REST + SSE streaming)
├── desktop_app.py           ← CustomTkinter native desktop GUI
├── cli.py                   ← Terminal CLI
├── requirements.txt         ← Python dependencies
├── start.bat                ← Windows one-click launcher
└── start.sh                 ← Linux / macOS one-click launcher
```

---

## Quick Start

### Step 1 — Install Ollama

Download and install from **https://ollama.com/download**

### Step 2 — Download the Phi-3 model (one time, ~2.2 GB)

```bash
ollama pull phi3
```

### Step 3 — Start the Ollama server

```bash
ollama serve
```

Keep this terminal open while using OPASAI.

### Step 4 — Install Python packages

```bash
pip install -r requirements.txt
```

### Step 5 — Run OPASAI

**Windows (easiest):** double-click `start.bat`

**Linux / macOS:** run `./start.sh`

**Or launch manually:**

```bash
# Web UI (recommended — opens in browser)
python server.py

# Native desktop app
python desktop_app.py

# Terminal / command line
python cli.py
```

---

## Interfaces

| Interface    | How to launch          | Best for                        |
|--------------|------------------------|---------------------------------|
| Web UI       | `python server.py`     | Any browser, mobile-friendly    |
| Desktop App  | `python desktop_app.py`| Windows native look & feel      |
| Terminal CLI | `python cli.py`        | Minimal, fast, scriptable       |

---

## Voice Input & Output

Voice features are optional. Install the packages below to enable them:

```bash
# Recommended (best quality, fully offline)
pip install faster-whisper sounddevice

# Lightweight fallback
pip install SpeechRecognition pocketsphinx

# Windows pyaudio (if needed for fallback)
pip install pipwin
pipwin install pyaudio
```

Text-to-speech works automatically on Windows, macOS, and Linux via `pyttsx3`
(uses built-in OS voices — no download needed).

---

## Personalise It

Edit `data/profile.txt` to update the knowledge base with your own information.
The file uses plain text — just add or change any details you like.
The RAG index rebuilds automatically every time the app starts.

---

## How it Works

```
Your Question
      │
      ▼
 RAG Engine  ←── profile.txt
(TF-IDF retrieval — finds the most relevant
 sections of your profile for the question)
      │
      ▼
 LLM Interface
(sends the relevant context + question
 to Phi-3 running inside Ollama)
      │
      ▼
 Streamed Answer
(tokens appear one by one as the
 model generates them)
      │
      ▼
 Voice Output (optional)
(pyttsx3 speaks the answer aloud)
```

---

## API Endpoints (Web Server)

| Method | Endpoint                | Description                        |
|--------|-------------------------|------------------------------------|
| GET    | `/`                     | Web UI                             |
| GET    | `/api/status`           | Health check for all components    |
| POST   | `/api/ask`              | Single-shot Q&A (blocking)         |
| GET    | `/api/stream?query=…`   | Streaming response (SSE)           |
| POST   | `/api/voice/transcribe` | WAV audio → transcript             |

---

## Privacy

Everything runs 100% on your machine:
- No data is sent to any server or cloud service
- No internet connection is needed after initial setup
- Your profile stays in `data/profile.txt` — only you can read it

---

## Requirements

- Python 3.10 or newer
- Ollama (installed separately from https://ollama.com)
- ~4 GB disk space for Phi-3 model weights
- 8 GB RAM recommended (4 GB minimum)
