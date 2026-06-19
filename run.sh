#!/usr/bin/env bash
set -euo pipefail

# Install dependencies
pip install -q -r requirements.txt

# Ask a single question
if [ $# -ge 1 ]; then
    python main.py "$@"
else
    python main.py --interactive
fi
