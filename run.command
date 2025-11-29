#!/bin/bash

# Ensure script runs from its own directory
cd "$(dirname "$0")"
echo "Using script directory: $(pwd)"

# Create venv if missing
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Install dependencies every time (safe)
echo "Installing dependencies..."
"./venv/bin/pip" install --upgrade pip
"./venv/bin/pip" install -r requirements.txt

echo "Running with Python at: $(pwd)/venv/bin/python"
"./venv/bin/python" image_to_stl.py &

# Open the localhost page automatically in default browser
open "http://127.0.0.1:7860"

# Keep the terminal open to show any logs
wait