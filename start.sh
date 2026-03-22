#!/usr/bin/env bash
# OPASAI — Launcher for Linux / macOS
set -e
cd "$(dirname "$0")"

CYAN='\033[0;36m'; GREEN='\033[0;32m'; RED='\033[0;31m'; NC='\033[0m'

echo -e "${CYAN}"
echo "  ======================================="
echo "   OPASAI — Offline Personal AI Assistant"
echo "  ======================================="
echo -e "${NC}"

# ── Python check ─────────────────────────────────────────────────────────────
if ! command -v python3 &>/dev/null; then
    echo -e "${RED}[ERROR] Python 3 is not installed. Install Python 3.10+ first.${NC}"
    exit 1
fi

# ── Install dependencies ──────────────────────────────────────────────────────
echo "[SETUP] Installing Python packages..."
pip3 install -r requirements.txt -q
echo "[SETUP] Packages ready."

# ── Ollama check ──────────────────────────────────────────────────────────────
if ! command -v ollama &>/dev/null; then
    echo -e "${RED}[ERROR] Ollama is not installed. Download from https://ollama.com/download${NC}"
    exit 1
fi

# Start Ollama if not already running
if ! pgrep -x "ollama" &>/dev/null; then
    echo "[START] Starting Ollama server..."
    ollama serve &>/dev/null &
    sleep 3
fi

# Pull Phi-3 if needed
if ! ollama list 2>/dev/null | grep -q "phi3"; then
    echo "[PULL] Downloading Phi-3 model (one-time, ~2.2 GB)..."
    ollama pull phi3
fi

# ── Choose interface ──────────────────────────────────────────────────────────
echo ""
echo "  ======================================="
echo "  How do you want to run OPASAI?"
echo "  [1] Web UI      (opens http://localhost:8000)"
echo "  [2] Desktop App (native window)"
echo "  [3] Terminal    (command line)"
echo "  ======================================="
read -r -p "  Your choice (1/2/3) [default: 1]: " choice
choice=${choice:-1}

case "$choice" in
    1)
        echo -e "${GREEN}[START] Launching web server...${NC}"
        (sleep 2 && python3 -m webbrowser "http://localhost:8000") &
        python3 server.py ;;
    2)
        echo -e "${GREEN}[START] Launching desktop app...${NC}"
        python3 desktop_app.py ;;
    3)
        echo -e "${GREEN}[START] Launching terminal CLI...${NC}"
        python3 cli.py ;;
    *)
        echo -e "${GREEN}[START] Launching web server...${NC}"
        (sleep 2 && python3 -m webbrowser "http://localhost:8000") &
        python3 server.py ;;
esac
