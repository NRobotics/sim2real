#!/bin/bash

# Directory to clear (relative to script location)
DATA_DIR="../data"

if [ -d "$DATA_DIR" ]; then
    read -p "Are you sure you want to remove all contents of '$DATA_DIR'? [y/N] " response
    if [[ "$response" =~ ^([yY][eE][sS]|[yY])$ ]]; then
        echo "Removing contents of $DATA_DIR..."
        rm -rf "$DATA_DIR"/*
        echo "Done."
    else
        echo "Operation cancelled."
    fi
else
    echo "Directory '$DATA_DIR' does not exist."
fi
