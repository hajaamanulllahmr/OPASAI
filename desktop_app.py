import sys
import os
import threading
import queue
import warnings

# Suppress noisy deprecation warnings from pyttsx3 / comtypes on Windows
warnings.filterwarnings("ignore")

# Make sure local modules resolve from wherever the script is launched
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ── GUI ───────────────────────────────────────────────────────────────────────
try:
    import customtkinter as ctk
    from PIL import Image
except ImportError:
    print("\n[ERROR] Required packages are missing.")
    print("  Run:  pip install customtkinter pillow\n")
    input("Press Enter to exit.")
    sys.exit(1)

# ── OPASAI modules ────────────────────────────────────────────────────────────
from rag_engine    import get_rag_engine
from llm_interface import chat_stream, is_ollama_running, is_model_available

# ── TTS (optional) ────────────────────────────────────────────────────────────
try:
    import pyttsx3
    _TTS_AVAILABLE = True
except ImportError:
    _TTS_AVAILABLE = False

# ── STT (optional) ────────────────────────────────────────────────────────────
try:
    import voice_handler as voice_module
    _STT_AVAILABLE = getattr(voice_module, "stt_available", lambda: False)()
except ImportError:
    voice_module   = None
    _STT_AVAILABLE = False


# ── Colour palette (dark theme) ───────────────────────────────────────────────
BG        = "#000000"   # main background — pure black
SURFACE   = "#1a1a1a"   # sidebar and header surface
SURFACE2  = "#2a2a2a"   # input box background
ACCENT    = "#ffffff"   # primary accent — white
ACCENT2   = "#cccccc"   # secondary accent — light grey
ERROR_RED = "#ff0000"   # used for error dots and listening indicator
OK_GREEN  = "#00ff00"   # used for healthy status dots
TEXT      = "#ffffff"   # default text colour
MUTED     = "#888888"   # secondary / hint text
USER_BUBBLE = "#333333" # background for the user's chat bubble
AI_BUBBLE   = "#444444" # background for OPASAI's chat bubble


# ═══════════════════════════════════════════════════════════════════════════════
#  TTS Worker — lives in its own thread so pyttsx3 never blocks the UI
# ═══════════════════════════════════════════════════════════════════════════════

class TTSWorker(threading.Thread):
    """
    A long-running daemon thread that speaks queued text via pyttsx3.

    Why a dedicated thread?
    -----------------------
    pyttsx3.runAndWait() blocks the calling thread until the utterance
    finishes. If we called it on the main thread, the entire UI would
    freeze while the assistant is speaking. Putting it here keeps the
    UI snappy regardless of how long the TTS takes.
    """

    def __init__(self):
        super().__init__(daemon=True, name="TTSWorker")
        self._queue  = queue.Queue()
        self._engine = None
        self._engine_ok = False

    def run(self):
        if not _TTS_AVAILABLE:
            return
        self._initialise_engine()

        while True:
            text = self._queue.get()
            if text is None:        # None is the shutdown signal
                break
            if text.strip():
                self._speak(text)

    def _initialise_engine(self):
        """Set up pyttsx3 with a friendly voice and comfortable speaking rate."""
        try:
            self._engine = pyttsx3.init()

            # Prefer a female voice on Windows (Zira, Hazel, Susan)
            for voice in (self._engine.getProperty("voices") or []):
                if any(name in (voice.name or "").lower() for name in ("zira", "female", "hazel", "susan")):
                    self._engine.setProperty("voice", voice.id)
                    break

            self._engine.setProperty("rate", 170)
            self._engine_ok = True

        except Exception as error:
            print(f"[TTS] Init error: {error}")
            self._engine_ok = False

    def _speak(self, text: str):
        """Speak one piece of text, re-initialising the engine if it broke."""
        if not self._engine_ok:
            self._initialise_engine()
        if not self._engine_ok:
            return

        # Clean text before speaking — remove CLI artefacts
        clean = str(text)[:800].strip().replace(">>>", "").replace("***", "").strip()
        if not clean:
            return

        try:
            self._engine.say(clean)
            self._engine.runAndWait()
        except Exception as error:
            print(f"[TTS] Speak error: {error}")
            # Try to recover so the next utterance can still work
            try:
                self._engine.stop()
            except Exception:
                pass
            self._engine_ok = False

    def say(self, text: str):
        """
        Queue `text` for speaking. Clears any pending utterance first so
        the most recent reply is always prioritised over older ones.
        """
        if not _TTS_AVAILABLE or not text:
            return
        # Drop anything already queued — only the latest message matters
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
        self._queue.put(text)

    def stop(self):
        """Shut down the worker gracefully."""
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
        self._queue.put(None)   # wake up the thread so it can exit


# ═══════════════════════════════════════════════════════════════════════════════
#  Main application window
# ═══════════════════════════════════════════════════════════════════════════════

class OpasaiDesktop(ctk.CTk):
    """
    The main OPASAI desktop window.

    Layout
    ------
    ┌─────────────┬──────────────────────────────────┐
    │  Sidebar    │  Header bar                       │
    │  (status,   ├──────────────────────────────────┤
    │   quick     │  Chat area (scrollable canvas)    │
    │   prompts,  ├──────────────────────────────────┤
    │   voice)    │  Input bar (textbox + send btn)   │
    └─────────────┴──────────────────────────────────┘
    """

    # How often (milliseconds) to drain the token queue into the chat label.
    # 40 ms ≈ 25 fps — snappy without overloading the UI thread.
    TOKEN_POLL_INTERVAL_MS = 40

    def __init__(self):
        super().__init__()
        self.title("OPASAI — Offline Personal AI Assistant")
        self.geometry("920x680")
        self.minsize(720, 500)
        self.configure(fg_color=BG)

        # State
        self._is_busy         = False    # True while the LLM is streaming
        self._next_msg_row    = 0        # grid row counter for chat bubbles
        self._token_queue     = queue.Queue()
        self._current_text    = ""       # accumulates tokens from the stream
        self._streaming_label = None     # the CTkLabel being updated live
        self._welcome_label   = None     # shown until the first message

        # Start TTS worker and RAG engine
        self._tts = TTSWorker()
        self._tts.start()
        self._rag = get_rag_engine()

        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self._build_ui()

        # Load the RAG index in the background so the window appears instantly
        threading.Thread(target=self._load_index_background, daemon=True, name="RAGLoader").start()

        # Start the token-draining poll loop
        self.after(self.TOKEN_POLL_INTERVAL_MS, self._drain_token_queue)

        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ──────────────────────────────────────────────────────────────────────────
    # UI construction
    # ──────────────────────────────────────────────────────────────────────────

    def _build_ui(self):
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)
        self._build_sidebar()
        self._build_chat_area()

    def _build_sidebar(self):
        sidebar = ctk.CTkFrame(self, width=230, fg_color=SURFACE, corner_radius=0)
        sidebar.grid(row=0, column=0, sticky="nsew")
        sidebar.grid_propagate(False)
        sidebar.grid_columnconfigure(0, weight=1)
        sidebar.grid_rowconfigure(5, weight=1)   # spacer expands here

        # ── Logo / branding ──────────────────────────────────────────────────
        header = ctk.CTkFrame(sidebar, fg_color="transparent")
        header.grid(row=0, column=0, padx=14, pady=(20, 10), sticky="ew")

        try:
            logo_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logo.png")
            logo_img  = ctk.CTkImage(Image.open(logo_path), size=(32, 32))
            ctk.CTkLabel(header, image=logo_img, text="", compound="left").pack(side="left", padx=(0, 10))
        except Exception:
            # Graceful fallback if the logo file is missing
            ctk.CTkLabel(header, text="🤖", font=ctk.CTkFont(size=28), text_color=ACCENT).pack(side="left", padx=(0, 10))

        name_frame = ctk.CTkFrame(header, fg_color="transparent")
        name_frame.pack(side="left")
        ctk.CTkLabel(name_frame, text="OPASAI",          font=ctk.CTkFont("Segoe UI", 22, "bold"), text_color="white").pack(anchor="w")
        ctk.CTkLabel(name_frame, text="Offline Assistant", font=ctk.CTkFont(size=10),               text_color=MUTED ).pack(anchor="w")

        # ── System status ─────────────────────────────────────────────────────
        status_box = ctk.CTkFrame(sidebar, fg_color=BG, corner_radius=10)
        status_box.grid(row=1, column=0, padx=10, pady=4, sticky="ew")
        ctk.CTkLabel(status_box, text="SYSTEM STATUS", font=ctk.CTkFont(size=9), text_color=MUTED).pack(anchor="w", padx=12, pady=(8, 2))

        self._status_dots = {}
        components = [
            ("ollama", "Ollama server"),
            ("model",  "Phi-3 model"),
            ("rag",    "Knowledge index"),
            ("tts",    "Text-to-speech"),
            ("stt",    "Voice input"),
        ]
        for key, label in components:
            row = ctk.CTkFrame(status_box, fg_color="transparent")
            row.pack(fill="x", padx=12, pady=2)
            ctk.CTkLabel(row, text=label, font=ctk.CTkFont(size=11), text_color=TEXT).pack(side="left")
            dot = ctk.CTkLabel(row, text="■", font=ctk.CTkFont(size=11), text_color=MUTED)
            dot.pack(side="right")
            self._status_dots[key] = dot

        ctk.CTkFrame(status_box, height=6, fg_color="transparent").pack()

        # ── Model info ────────────────────────────────────────────────────────
        model_box = ctk.CTkFrame(sidebar, fg_color=BG, corner_radius=10)
        model_box.grid(row=2, column=0, padx=10, pady=4, sticky="ew")
        ctk.CTkLabel(model_box, text="MODEL",        font=ctk.CTkFont(size=9),           text_color=MUTED ).pack(anchor="w", padx=12, pady=(8, 0))
        ctk.CTkLabel(model_box, text="Phi-3 (tiny)", font=ctk.CTkFont("Segoe UI", 13, "bold"), text_color="white").pack(anchor="w", padx=12)
        ctk.CTkLabel(model_box, text="100% OFFLINE", font=ctk.CTkFont(size=10),           text_color=MUTED ).pack(anchor="w", padx=12, pady=(0, 8))

        # ── Quick prompts ─────────────────────────────────────────────────────
        ctk.CTkLabel(sidebar, text="QUICK PROMPTS", font=ctk.CTkFont(size=9), text_color=MUTED).grid(
            row=3, column=0, padx=14, pady=(10, 3), sticky="w")

        prompts_frame = ctk.CTkFrame(sidebar, fg_color="transparent")
        prompts_frame.grid(row=4, column=0, padx=10, sticky="ew")

        quick_prompts = [
            "Who am I?",
            "What are my technical skills?",
            "Tell me about my projects",
            "Where do I work?",
            "What are my career goals?",
            "What are my hobbies?",
        ]
        for prompt in quick_prompts:
            ctk.CTkButton(
                prompts_frame, text=prompt,
                font=ctk.CTkFont(size=11),
                fg_color=BG, hover_color=USER_BUBBLE,
                border_color="#2a2f4a", border_width=1,
                text_color=TEXT, anchor="w",
                command=lambda p=prompt: self._send_quick_prompt(p),
            ).pack(fill="x", pady=2)

        # ── Spacer ────────────────────────────────────────────────────────────
        ctk.CTkFrame(sidebar, fg_color="transparent").grid(row=5, column=0, sticky="nsew")

        # ── Voice input button ────────────────────────────────────────────────
        self._voice_button = ctk.CTkButton(
            sidebar, text="🎤  Voice Input",
            font=ctk.CTkFont("Segoe UI", 12, "bold"),
            fg_color=BG, hover_color=USER_BUBBLE,
            border_color=ACCENT, border_width=1,
            text_color=ACCENT,
            command=self._start_voice_input,
        )
        self._voice_button.grid(row=6, column=0, padx=10, pady=(0, 8), sticky="ew")
        if not _STT_AVAILABLE:
            self._voice_button.configure(state="disabled")

        # ── Clear chat button ──────────────────────────────────────────────────
        ctk.CTkButton(
            sidebar, text="Clear Chat",
            font=ctk.CTkFont("Segoe UI", 12, "bold"),
            fg_color="#4a4f5a", hover_color="#5a5f6a",
            text_color=TEXT,
            command=self._clear_chat,
        ).grid(row=7, column=0, padx=10, pady=(0, 16), sticky="ew")

    def _build_chat_area(self):
        main = ctk.CTkFrame(self, fg_color=BG, corner_radius=0)
        main.grid(row=0, column=1, sticky="nsew")
        main.grid_rowconfigure(1, weight=1)
        main.grid_columnconfigure(0, weight=1)

        # ── Header bar ────────────────────────────────────────────────────────
        header = ctk.CTkFrame(main, fg_color=SURFACE, height=48, corner_radius=0)
        header.grid(row=0, column=0, columnspan=2, sticky="ew")
        header.grid_propagate(False)
        ctk.CTkLabel(header, text="Conversation", font=ctk.CTkFont(size=12), text_color=MUTED).pack(side="left", padx=20, pady=14)
        self._status_label = ctk.CTkLabel(header, text="Starting up…", font=ctk.CTkFont(size=11), text_color=ACCENT)
        self._status_label.pack(side="right", padx=20)

        # ── Scrollable chat area ──────────────────────────────────────────────
        # We use a raw Canvas + Frame instead of CTkScrollableFrame because
        # CTkScrollableFrame has DPI-related layout bugs on Windows.
        self._canvas    = ctk.CTkCanvas(main, bg=BG, highlightthickness=0, borderwidth=0)
        self._scrollbar = ctk.CTkScrollbar(main, orientation="vertical", command=self._canvas.yview)
        self._canvas.configure(yscrollcommand=self._scrollbar.set)
        self._scrollbar.grid(row=1, column=1, sticky="ns")
        self._canvas.grid(row=1, column=0, sticky="nsew")

        self._chat_frame = ctk.CTkFrame(self._canvas, fg_color=BG)
        self._chat_frame.grid_columnconfigure(0, weight=1)
        self._canvas_window = self._canvas.create_window((0, 0), window=self._chat_frame, anchor="nw")

        # Keep the canvas scroll region and window width in sync with resizes
        self._chat_frame.bind("<Configure>", lambda e: self._canvas.configure(scrollregion=self._canvas.bbox("all")))
        self._canvas.bind("<Configure>",     lambda e: self._canvas.itemconfig(self._canvas_window, width=e.width))
        self._canvas.bind_all("<MouseWheel>", lambda e: self._canvas.yview_scroll(int(-1 * (e.delta / 120)), "units"))

        # Welcome message shown before the first conversation
        self._welcome_label = ctk.CTkLabel(
            self._chat_frame,
            text="Hey there! 👋\n\nAsk me anything about you.\nType below or pick a quick prompt from the sidebar.",
            font=ctk.CTkFont(size=14), text_color=MUTED, justify="center",
        )
        self._welcome_label.grid(row=0, column=0, pady=80, padx=20)
        self._next_msg_row = 1

        # ── Input bar ─────────────────────────────────────────────────────────
        input_bar = ctk.CTkFrame(main, fg_color=SURFACE, corner_radius=0)
        input_bar.grid(row=2, column=0, columnspan=2, sticky="ew")
        input_bar.grid_columnconfigure(0, weight=1)

        self._input_box = ctk.CTkTextbox(
            input_bar, height=48,
            fg_color=SURFACE2, border_color="#2a2f4a", border_width=1,
            text_color=TEXT, font=ctk.CTkFont("Segoe UI", 13), wrap="word",
        )
        self._input_box.grid(row=0, column=0, padx=(14, 8), pady=12, sticky="ew")
        self._input_box.bind("<Return>",   self._on_enter_key)
        self._input_box.bind("<KP_Enter>", self._on_enter_key)

        self._send_button = ctk.CTkButton(
            input_bar, text="Send ➤", width=120,
            font=ctk.CTkFont("Segoe UI", 13, "bold"),
            fg_color="#ffffff", hover_color="#cccccc", text_color="#000000",
            command=self._on_send,
        )
        self._send_button.grid(row=0, column=1, padx=(0, 8), pady=12)

        ctk.CTkLabel(
            input_bar,
            text="Enter = send   |   Shift+Enter = new line",
            font=ctk.CTkFont(size=8), text_color="#555555",
        ).grid(row=1, column=0, columnspan=2, pady=(0, 6), padx=8)

        # ── Footer ────────────────────────────────────────────────────────────
        ctk.CTkFrame(main, fg_color="#2a2f4a", height=1).grid(row=3, column=0, columnspan=2, sticky="ew")

        footer = ctk.CTkFrame(main, fg_color=BG)
        footer.grid(row=4, column=0, columnspan=2, sticky="ew", padx=14, pady=8)
        footer.grid_columnconfigure((0, 1, 2), weight=1)
        ctk.CTkLabel(footer, text="OPASAI",                           font=ctk.CTkFont("JetBrains Mono", 8), text_color="#8b949e").grid(row=0, column=0, sticky="w")
        ctk.CTkLabel(footer, text="OFFLINE PERSONAL AI ASSISTANT",    font=ctk.CTkFont("JetBrains Mono", 8), text_color="#8b949e").grid(row=0, column=1)
        ctk.CTkLabel(footer, text="BY HAJA AMANULLAH M R", font=ctk.CTkFont("JetBrains Mono", 8), text_color="#8b949e").grid(row=0, column=2, sticky="e")

    # ──────────────────────────────────────────────────────────────────────────
    # Scrolling
    # ──────────────────────────────────────────────────────────────────────────

    def _scroll_to_bottom(self):
        """Scroll the chat area down to show the latest message."""
        self.update_idletasks()
        self._canvas.configure(scrollregion=self._canvas.bbox("all"))
        self._canvas.yview_moveto(1.0)

    # ──────────────────────────────────────────────────────────────────────────
    # Start-up sequence
    # ──────────────────────────────────────────────────────────────────────────

    def _load_index_background(self):
        """Load the RAG index in a background thread; report results to the UI."""
        self.after(0, self._set_status, "Loading knowledge index…")
        try:
            self._rag.load()
        except Exception as error:
            self.after(0, self._set_status, f"Index error: {error}")
            return
        self.after(0, self._on_index_loaded)

    def _on_index_loaded(self):
        """Called on the main thread once the RAG index is ready."""
        dot_colour = {True: OK_GREEN, False: ERROR_RED}

        ollama_ok = is_ollama_running()
        model_ok  = is_model_available()

        self._status_dots["ollama"].configure(text_color=dot_colour[ollama_ok])
        self._status_dots["model" ].configure(text_color=dot_colour[model_ok])
        self._status_dots["rag"   ].configure(text_color=dot_colour[self._rag.ready])
        self._status_dots["tts"   ].configure(text_color=dot_colour[_TTS_AVAILABLE])
        self._status_dots["stt"   ].configure(text_color=dot_colour[_STT_AVAILABLE])

        if not ollama_ok:
            self._set_status("⚠  Run:  ollama serve")
        elif not model_ok:
            self._set_status("⚠  Run:  ollama pull phi3")
        else:
            self._set_status("Ready ✓")

        # Periodically refresh the Ollama status dot (every 10 s)
        self.after(10_000, self._refresh_status_dots)

    def _refresh_status_dots(self):
        """Keep the status dots up to date even if Ollama starts/stops later."""
        dot_colour = {True: OK_GREEN, False: ERROR_RED}
        ok = is_ollama_running()
        self._status_dots["ollama"].configure(text_color=dot_colour[ok])
        self._status_dots["stt"   ].configure(text_color=dot_colour[_STT_AVAILABLE])
        if not ok and not self._is_busy:
            self._set_status("⚠  Ollama not running")
        self.after(10_000, self._refresh_status_dots)

    # ──────────────────────────────────────────────────────────────────────────
    # Sending messages
    # ──────────────────────────────────────────────────────────────────────────

    def _on_enter_key(self, event):
        """Send on Enter; Shift+Enter inserts a newline (normal behaviour)."""
        if event.state & 0x1:
            return           # Shift held — let the newline through
        self._on_send()
        return "break"       # consume the event so no newline is inserted

    def _send_quick_prompt(self, text: str):
        """Fill the input box with a quick prompt and send it immediately."""
        self._input_box.delete("1.0", "end")
        self._input_box.insert("1.0", text)
        self._on_send()

    def _on_send(self):
        """Read the input box, validate it, then kick off an LLM stream."""
        if self._is_busy:
            return

        question = self._input_box.get("1.0", "end").strip()
        if not question:
            return

        self._input_box.delete("1.0", "end")

        # Remove the welcome message on first send
        if self._welcome_label:
            self._welcome_label.destroy()
            self._welcome_label = None

        # Guard: don't call Ollama if it isn't running
        if not is_ollama_running():
            self._add_bubble("user", question)
            self._add_bubble("assistant", (
                "⚠️  Ollama isn't running.\n\n"
                "Open a terminal and run:\n"
                "    ollama serve\n\n"
                "If you haven't downloaded the model yet:\n"
                "    ollama pull phi3\n\n"
                "Then try your question again."
            ))
            return

        # Build retrieval context before spawning the streamer thread so
        # RAG work happens synchronously and the thread can start right away.
        context = self._rag.build_context(question)

        self._add_bubble("user", question)
        self._set_busy(True)
        self._current_text    = ""
        self._streaming_label = self._add_bubble("assistant", "…")   # placeholder

        threading.Thread(
            target=self._stream_response,
            args=(question, context),
            daemon=True, name="LLMStreamer",
        ).start()

    # ──────────────────────────────────────────────────────────────────────────
    # Streaming
    # ──────────────────────────────────────────────────────────────────────────

    def _stream_response(self, question: str, context: str):
        """
        Background thread: fetch tokens from Ollama and put them in the queue.
        Sends `None` as a sentinel when streaming is complete.
        """
        try:
            for token in chat_stream(question, context):
                self._token_queue.put(token)
        except Exception as error:
            self._token_queue.put(f"\n\n⚠️  Error: {error}")
        finally:
            self._token_queue.put(None)   # sentinel — streaming is done

    def _drain_token_queue(self):
        """
        Main-thread poll: drain the token queue and update the chat label.
        Runs every TOKEN_POLL_INTERVAL_MS milliseconds via `.after()`.
        """
        text_changed = False

        while True:
            try:
                token = self._token_queue.get_nowait()
            except queue.Empty:
                break

            if token is None:
                self._on_stream_finished()
                break

            self._current_text += token
            text_changed = True

        if text_changed and self._streaming_label:
            # Show a blinking cursor at the end while still streaming
            self._streaming_label.configure(text=self._current_text + "▌")
            self._scroll_to_bottom()

        self.after(self.TOKEN_POLL_INTERVAL_MS, self._drain_token_queue)

    def _on_stream_finished(self):
        """Called on the main thread when the LLM has finished generating."""
        final = self._current_text.strip() or "(no response from model)"
        if self._streaming_label:
            self._streaming_label.configure(text=final)
            self._scroll_to_bottom()

        self._set_busy(False)
        self._set_status("Ready ✓")

        # Speak the answer if TTS is available
        if _TTS_AVAILABLE and len(final) > 3:
            self._tts.say(final)

    # ──────────────────────────────────────────────────────────────────────────
    # Chat bubbles
    # ──────────────────────────────────────────────────────────────────────────

    def _add_bubble(self, role: str, text: str) -> ctk.CTkLabel:
        """
        Add a chat bubble to the conversation area.
        `role` is either "user" or "assistant".
        Returns the CTkLabel inside the bubble so streaming can update it.
        """
        is_user = (role == "user")
        side    = "e" if is_user else "w"   # right-align user, left-align assistant

        # Outer container (transparent, full-width)
        row_frame = ctk.CTkFrame(self._chat_frame, fg_color="transparent")
        row_frame.grid(row=self._next_msg_row, column=0, sticky="ew", padx=10, pady=(4, 2))
        row_frame.grid_columnconfigure(0, weight=1)
        self._next_msg_row += 1

        # Sender label ("You" / "OPASAI")
        ctk.CTkLabel(
            row_frame,
            text="You" if is_user else "OPASAI",
            font=ctk.CTkFont(size=9), text_color=MUTED,
        ).pack(anchor=side, padx=16)

        # Bubble frame
        bubble = ctk.CTkFrame(
            row_frame,
            fg_color=USER_BUBBLE if is_user else AI_BUBBLE,
            border_color=ACCENT if is_user else ACCENT2,
            border_width=1, corner_radius=12,
        )
        bubble.pack(
            anchor=side, padx=16, pady=(2, 6),
            fill="x" if not is_user else "none",
            expand=not is_user,
        )

        # Text label inside the bubble
        message_label = ctk.CTkLabel(
            bubble, text=text,
            font=ctk.CTkFont("Segoe UI", 12),
            text_color=TEXT, justify="left",
            wraplength=530, anchor="w",
        )
        message_label.pack(padx=14, pady=10, fill="x", expand=True)

        self._scroll_to_bottom()

        # Speak assistant messages immediately (except the streaming placeholder)
        if not is_user and _TTS_AVAILABLE and text.strip() and text.strip() not in ("…", ">>>"):
            self._tts.say(text)

        return message_label

    # ──────────────────────────────────────────────────────────────────────────
    # Voice input
    # ──────────────────────────────────────────────────────────────────────────

    def _start_voice_input(self):
        """Start listening via the microphone on a background thread."""
        if not _STT_AVAILABLE:
            self._add_bubble("assistant", (
                "Voice input isn't available.\n\n"
                "Install the required packages:\n"
                "    pip install faster-whisper sounddevice\n\n"
                "You can still type your question below!"
            ))
            return

        self._voice_button.configure(
            text="● Listening…", state="disabled",
            text_color=ERROR_RED, border_color=ERROR_RED,
        )
        self._set_status("Listening…")
        threading.Thread(target=self._listen_worker, daemon=True, name="VoiceListener").start()

    def _listen_worker(self):
        """Background thread: record microphone audio and transcribe it."""
        transcript = ""
        error      = None

        try:
            if voice_module and getattr(voice_module, "stt_available", lambda: False)():
                result = voice_module.listen_once()
                if isinstance(result, tuple) and len(result) >= 2:
                    transcript, error = result[0], result[1]
                elif result is None:
                    transcript, error = "", "No result returned"
                else:
                    transcript, error = str(result), None
            else:
                # PocketSphinx fallback
                import speech_recognition as sr   # type: ignore
                recogniser = sr.Recognizer()
                with sr.Microphone() as mic:
                    recogniser.adjust_for_ambient_noise(mic, duration=0.5)
                    audio = recogniser.listen(mic, timeout=8, phrase_time_limit=12)
                transcript = recogniser.recognize_sphinx(audio)
        except Exception as err:
            transcript = ""
            error      = str(err)[:80]

        self.after(0, self._on_voice_received, transcript, error)

    def _on_voice_received(self, transcript: str, error: str = None):
        """Called on the main thread once voice recognition completes."""
        self._voice_button.configure(
            text="🎤  Voice Input", state="normal",
            text_color=ACCENT, border_color=ACCENT,
        )
        if transcript:
            self._input_box.delete("1.0", "end")
            self._input_box.insert("1.0", transcript)
            self._on_send()
        else:
            self._set_status(error or "Couldn't hear anything — please try again.")

    # ──────────────────────────────────────────────────────────────────────────
    # Chat management
    # ──────────────────────────────────────────────────────────────────────────

    def _clear_chat(self):
        """
        Remove all chat bubbles and return to the welcome screen.
        If a stream is in progress, it's cancelled cleanly.
        """
        # Stop any in-progress response
        if self._is_busy:
            self._set_busy(False)
            while not self._token_queue.empty():
                try:
                    self._token_queue.get_nowait()
                except queue.Empty:
                    break
            self._streaming_label = None
            self._current_text    = ""

        # Stop TTS and restart the worker so it's clean for next time
        if _TTS_AVAILABLE:
            try:
                if getattr(self._tts, "_engine", None):
                    self._tts._engine.stop()
            except Exception:
                pass
            self._tts.stop()
            self._tts = TTSWorker()
            self._tts.start()

        # Destroy all existing message widgets
        for widget in self._chat_frame.winfo_children():
            widget.destroy()

        # Put the welcome message back
        self._welcome_label = ctk.CTkLabel(
            self._chat_frame,
            text="Hey there! 👋\n\nAsk me anything about you.\nType below or pick a quick prompt from the sidebar.",
            font=ctk.CTkFont(size=14), text_color=MUTED, justify="center",
        )
        self._welcome_label.grid(row=0, column=0, pady=80, padx=20)
        self._next_msg_row = 1
        self._scroll_to_bottom()

    # ──────────────────────────────────────────────────────────────────────────
    # Utilities
    # ──────────────────────────────────────────────────────────────────────────

    def _set_status(self, message: str):
        """Update the status label in the header bar."""
        self._status_label.configure(text=message)

    def _set_busy(self, busy: bool):
        """Toggle the busy state: disable the send button, update status."""
        self._is_busy = busy
        self._send_button.configure(state="disabled" if busy else "normal")
        if busy:
            self._set_status("Thinking…")

    def _on_close(self):
        """Clean up before the window closes."""
        self._tts.stop()
        self.destroy()


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Enable crisp text on Windows high-DPI displays
    try:
        from ctypes import windll
        windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass   # Not on Windows — ignore

    OpasaiDesktop().mainloop()
