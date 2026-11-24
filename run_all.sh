#!/usr/bin/env bash
# One-click run: creates a virtual environment, installs libraries, runs everything.
set -e
cd "$(dirname "$0")"

echo "[1/5] Creating virtual environment..."
python3 -m venv .venv
source .venv/bin/activate

echo "[2/5] Installing libraries..."
python -m pip install --upgrade pip >/dev/null
pip install -r requirements.txt || pip install numpy scipy pandas matplotlib SALib

echo "[3/5] Running experiments (about 30-50 seconds)..."
python src/experiments.py all

echo "[4/5] Drawing figures..."
python src/figures.py

echo "[5/5] Writing results summary..."
python src/report.py

echo
echo "Done. Results: results/RESULTS.md   Figures: figures/"
