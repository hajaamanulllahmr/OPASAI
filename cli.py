import sys
import os

# Make sure local modules are on the path regardless of where the script
# is launched from.
sys.path.insert(0, os.path.dirname(__file__))

from assistant     import get_assistant
from llm_interface import is_ollama_running, is_model_available

BANNER = r"""
   ____  ____   _    ____    _    ___
  / __ \|  _ \ / \  / ___|  / \  |_ _|
 | |  | | |_) / _ \ \___ \ / _ \  | |
 | |__| |  __/ ___ \ ___) / ___ \ | |
  \____/|_| /_/   \_\____/_/   \_\___|

  OPASAI — Your Offline Personal AI Assistant
"""


def print_startup_warnings(assistant):
    """Warn the user about any components that aren't ready yet."""
    if not is_ollama_running():
        print("  ⚠  Ollama isn't running.")
        print("     Start it with:  ollama serve")
        print("     Then pull the model:  ollama pull phi3")
        print()
    elif not is_model_available():
        print("  ⚠  Phi-3 model not found.")
        print("     Download it with:  ollama pull phi3")
        print()


def print_status(assistant):
    """Print a formatted status table for all components."""
    status = assistant.status()
    labels = {
        "rag_ready"     : "RAG index",
        "ollama_running": "Ollama server",
        "model_ready"   : "Phi-3 model",
        "tts_available" : "Text-to-speech",
        "stt_available" : "Speech-to-text",
    }
    print()
    for key, label in labels.items():
        icon = "✓" if status.get(key) else "✗"
        print(f"  {icon}  {label}: {status[key]}")
    print()


def main():
    print(BANNER)

    assistant = get_assistant()
    print_startup_warnings(assistant)

    print("  Loading knowledge index…")
    assistant.init_sync()
    print("  Ready! Type your question below.")
    print("  (Commands: 'voice', 'status', 'exit')\n")

    while True:
        # Read input — handle Ctrl+C and Ctrl+D gracefully
        try:
            raw_input = input("You  > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nOpasai > Goodbye! 👋")
            break

        if not raw_input:
            continue

        # Exit commands
        if raw_input.lower() in ("exit", "quit", "bye"):
            print("Opasai > Goodbye! 👋")
            break

        # Status report
        if raw_input.lower() == "status":
            print_status(assistant)
            continue

        # Voice input mode
        if raw_input.lower() == "voice":
            import voice_handler
            if not voice_handler.stt_available():
                print(
                    "Opasai > Voice input isn't available.\n"
                    "         Install it with:  pip install faster-whisper sounddevice\n"
                )
                continue

            print("Opasai > Listening… speak your question now.")
            transcript, error = voice_handler.listen_once()

            if not transcript:
                print(f"Opasai > Couldn't hear anything. ({error})\n")
                continue

            print(f"You  > {transcript}")
            raw_input = transcript   # fall through to answer it below

        # Stream the answer token-by-token so it appears as it's generated
        print("Opasai > ", end="", flush=True)
        full_answer = ""
        for token in assistant.ask_stream(raw_input):
            print(token, end="", flush=True)
            full_answer += token
        print()   # newline after the streamed answer

        # Speak the answer if TTS is available
        import voice_handler
        if voice_handler.tts_available() and full_answer:
            voice_handler.speak_async(full_answer)


if __name__ == "__main__":
    main()
