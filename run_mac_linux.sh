#!/usr/bin/env bash
# One-click setup and launch on macOS / Linux
set -e
cd "$(dirname "$0")"
[ -d .venv ] || python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip >/dev/null
pip install -r requirements.txt
streamlit run app.py
