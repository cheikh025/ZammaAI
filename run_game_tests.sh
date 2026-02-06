#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

if command -v python3 >/dev/null 2>&1; then
    exec python3 test_runner.py "$@"
elif command -v python >/dev/null 2>&1; then
    exec python test_runner.py "$@"
elif [ -x "/mnt/c/Python313/python.exe" ]; then
    exec /mnt/c/Python313/python.exe test_runner.py "$@"
elif [ -x "/c/Python313/python.exe" ]; then
    exec /c/Python313/python.exe test_runner.py "$@"
else
    echo "No Python interpreter found." >&2
    exit 1
fi
