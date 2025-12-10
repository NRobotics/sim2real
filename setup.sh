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

# Install sim2real with dev dependencies
echo "Installing sim2real package..."
# Install humanoid-protocol
echo "Installing humanoid-protocol..."
pip install -e humanoid-protocol/python

# Install sim2real with dev dependencies
echo "Installing sim2real package..."
pip install -e ".[dev]"

echo
echo "=== Setup Complete ==="
echo
echo "To activate the environment:"
echo "  source .venv/bin/activate"
echo
echo "To run system identification:"
echo "  python system_identification/system_identification.py --config system_identification/config.json"
echo
echo "Or use the installed command:"
echo "  sysid --config system_identification/config.json"
echo
echo "Don't forget to configure your CAN interface:"
echo "  sudo ip link set can0 type can bitrate 1000000 fd on"
echo "  sudo ip link set can0 up"

