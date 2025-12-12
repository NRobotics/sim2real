#!/bin/bash
# Setup script for sim2real project

set -e

# Configuration
REQUIRED_MAJOR=3
REQUIRED_MINOR=12

check_python_version() {
    local cmd="$1"
    # Use python's internal version_info to verify constraints
    "$cmd" -c "import sys; exit(0) if sys.version_info >= ($REQUIRED_MAJOR, $REQUIRED_MINOR) else exit(1)" 2>/dev/null
}

find_python_executable() {
    # 1. Try generic python3 if it meets the requirement
    if command -v python3 >/dev/null 2>&1 && check_python_version "python3"; then
        echo "python3"
        return 0
    fi

    # 2. Try specific version binaries (python3.12, python3.13, etc)
    # You can add more specific versions here if needed
    for ver in "3.12" "3.13" "3.14"; do
        local cmd="python$ver"
        if command -v "$cmd" >/dev/null 2>&1; then
            echo "$cmd"
            return 0
        fi
    done

    return 1
}

main() {
    echo "=== Sim2Real Project Setup ==="
    echo

    # --- Step 1: Find Correct Python Version ---
    echo "Checking for Python >= $REQUIRED_MAJOR.$REQUIRED_MINOR..."
    
    PYTHON_EXEC=$(find_python_executable)

    if [ -z "$PYTHON_EXEC" ]; then
        echo "Error: Python $REQUIRED_MAJOR.$REQUIRED_MINOR or higher is required."
        echo "Please install Python 3.12+ (e.g., 'sudo apt install python3.12' or via homebrew)."
        return 1
    fi

    echo "Using Python executable: $(which $PYTHON_EXEC) ($($PYTHON_EXEC --version))"

    # --- Step 2: Initialize Submodules ---
    echo "Initializing git submodules..."
    git submodule update --init --recursive

    # --- Step 3: Create Virtual Environment ---
    echo "Creating virtual environment..."
    if [ ! -d ".venv" ]; then
        # IMPORTANT: Use the specific PYTHON_EXEC found above, not just 'python3'
        "$PYTHON_EXEC" -m venv .venv
    fi
    
    # Activate the environment
    source .venv/bin/activate

    # Double check that we are running the correct python inside the venv
    CURRENT_PY=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    echo "Virtual environment active (Python $CURRENT_PY)"

    # --- Step 4: Install Dependencies ---
    echo "Upgrading pip..."
    pip install --upgrade pip

    # Install humanoid-protocol
    if [ -d "humanoid-protocol/python" ]; then
        echo "Installing humanoid-protocol..."
        pip install -e humanoid-protocol/python
    else
        echo "Warning: humanoid-protocol/python directory not found. Skipping."
    fi

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