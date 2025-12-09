#!/bin/bash
# Setup script for sim2real project

set -e

echo "=== Sim2Real Project Setup ==="
echo

# Check Python version
python3 --version || { echo "Python 3 is required"; exit 1; }

# Initialize submodules
echo "Initializing git submodules..."
git submodule update --init --recursive

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv .venv
source .venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install humanoid-protocol
echo "Installing humanoid-protocol..."
pip install -e ext/humanoid-protocol/python

# Install system identification with dev dependencies
echo "Installing system identification tools..."
cd system_identification
pip install -e ".[dev]"
cd ..

echo
echo "=== Setup Complete ==="
echo
echo "To activate the environment:"
echo "  source .venv/bin/activate"
echo
echo "To run system identification:"
echo "  cd system_identification"
echo "  python system_identification.py --config config.json"
echo
echo "Don't forget to configure your CAN interface:"
echo "  sudo ip link set can0 type can bitrate 1000000 fd on"
echo "  sudo ip link set can0 up"

