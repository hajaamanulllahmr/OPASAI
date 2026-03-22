#!/usr/bin/env bash
# ARIA — Personal AI Assistant Launcher (Linux / macOS)
set -e
cd "$(dirname "$0")"

RED='\033[0;31m'; GREEN='\033[0;32m'; CYAN='\033[0;36m'; NC='\033[0m'
echo -e "${CYAN}"
echo "  ======================================="
echo "   ARIA — Personal AI Assistant"
echo "  ======================================="
echo -e "${NC}"

# ── Python check ──────────────────────────────────────────────────────────────
if ! command -v python3 &>/dev/null; then
  echo -e "${RED}[ERROR] Python 3 not found. Install Python 3.10+${NC}"; exit 1
fi

# ── Virtual env ───────────────────────────────────────────────────────────────
if [ ! -d ".venv" ]; then
  echo "[SETUP] Creating virtual environment…"
  python3 -m venv .venv
fi
source .venv/bin/activate

# ── Install deps ──────────────────────────────────────────────────────────────
echo "[SETUP] Installing/checking dependencies…"
pip install -r requirements.txt -q

# ── Ollama check ──────────────────────────────────────────────────────────────
if ! command -v ollama &>/dev/null; then
  echo -e "${RED}[ERROR] Ollama not found. Install from https://ollama.com/download${NC}"; exit 1
fi

# ── Start Ollama if not running ───────────────────────────────────────────────
if ! pgrep -x "ollama" &>/dev/null; then
  echo "[START] Starting Ollama server…"
  ollama serve &>/dev/null &
  sleep 3
fi

# ── Pull phi3 if needed ───────────────────────────────────────────────────────
if ! ollama list 2>/dev/null | grep -q "phi3"; then
  echo "[PULL] Downloading phi3 model…"
  ollama pull phi3
fi

echo ""
echo "  ======================================="
echo "  Choose mode:"
echo "  [1] Web UI (opens http://localhost:8000)"
echo "  [2] Desktop App"
echo "  ======================================="
read -p "  Enter choice (1/2) [default: 1]: " choice
choice=${choice:-1}

case "$choice" in
  1)
    echo -e "${GREEN}[START] Launching web server…${NC}"
    ( sleep 2 && python3 -m webbrowser "http://localhost:8000" ) &
    python3 server.py ;;
  2)
    echo -e "${GREEN}[START] Launching desktop app…${NC}"
    python3 desktop_app.py ;;
  *)
    echo -e "${GREEN}[START] Launching web server…${NC}"
    ( sleep 2 && python3 -m webbrowser "http://localhost:8000" ) &
    python3 server.py ;;
esac
