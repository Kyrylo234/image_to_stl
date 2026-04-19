#!/bin/bash
cd "$(dirname "$0")"
echo "Using script directory: $(pwd)"

if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

echo "Installing dependencies..."
"./venv/bin/pip" install --upgrade pip
"./venv/bin/pip" install -r requirements.txt

echo "Running with Python at: $(pwd)/venv/bin/python"
"./venv/bin/python" image_to_stl.py &
PID=$!

open "http://127.0.0.1:7860"

echo "Press Ctrl+C to stop the server..."
trap "echo 'Stopping server...'; kill $PID; exit 0" SIGINT SIGTERM

wait $PID