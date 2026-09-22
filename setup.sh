#!/usr/bin/env bash
# ==============================================================================
# Environment Setup Script for Apple Silicon ASR Benchmark
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

echo "============================================================"
echo " Setting up Apple Silicon ASR Benchmark Environment"
echo "============================================================"

# 1. Check Platform
OS="$(uname -s)"
ARCH="$(uname -m)"

if [ "${OS}" != "Darwin" ]; then
    echo "[-] Warning: This benchmark is optimized for macOS with Apple Silicon."
fi

if [ "${ARCH}" != "arm64" ]; then
    echo "[-] Warning: Apple Silicon (arm64) architecture expected. Detected: ${ARCH}"
fi

# 2. Virtual Environment Setup
VENV_DIR="${SCRIPT_DIR}/.venv"

if [ ! -d "${VENV_DIR}" ]; then
    echo "[+] Creating virtual environment in ${VENV_DIR}..."
    python3 -m venv "${VENV_DIR}"
else
    echo "[*] Virtual environment already exists at ${VENV_DIR}."
fi

# 3. Activate and Install Dependencies
echo "[+] Activating virtual environment..."
source "${VENV_DIR}/bin/activate"

echo "[+] Upgrading pip..."
pip install --upgrade pip

echo "[+] Installing pinned dependencies from requirements.txt..."
pip install -r requirements.txt

# 4. Create necessary local directories
echo "[+] Initializing directory layout..."
mkdir -p data/longform
mkdir -p outputs/runs
mkdir -p assets

echo "============================================================"
echo " Setup complete! Activate your environment via:"
echo "   source .venv/bin/activate"
echo "============================================================"
