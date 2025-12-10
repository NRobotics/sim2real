#!/bin/bash
# Setup script for sim2real project

set -e


main() {
    echo "=== Sim2Real Project Setup ==="
    echo

    # Check Python version
    python3 --version || { echo "Python 3 is required"; return 1; }

    # Initialize submodules
    echo "Initializing git submodules..."
    git submodule update --init --recursive

    # Create virtual environment
    echo "Creating virtual environment..."
    if [ ! -d ".venv" ]; then
        python3 -m venv .venv
    fi
    source .venv/bin/activate

    # Upgrade pip
    pip install --upgrade pip

    # Install humanoid-protocol
    echo "Installing humanoid-protocol..."
    pip install -e humanoid-protocol/python

    # Install sim2real with dev dependencies
    echo "Installing sim2real package..."
    pip install -e ".[dev]"

    echo
    echo "=== Setup Complete ==="
    echo
    echo "Environment is active."
}

# Check if the script is being sourced or executed
if [[ "${BASH_SOURCE[0]}" != "${0}" ]]; then
    # Script is being sourced
    main "$@"
else
    # Script is being executed
    main "$@"
    echo
    echo "NOTE: To keep the environment active, run this script with 'source':"
    echo "  source setup.sh"
    echo
fi
