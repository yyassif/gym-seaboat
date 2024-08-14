#!/bin/bash

# Check if .env directory exists
if [ ! -d ".env" ]; then
  echo ".env directory does not exist. Creating virtual environment..."
  python -m venv .env
else
  echo ".env directory exists. Activating virtual environment..."
fi

# Activate the virtual environment
source .env/bin/activate

# Install packages
pip install -e .
