"""System identification package for robot actuators."""

import sys
from pathlib import Path

# Add scripts directory to path for imports
_scripts_dir = Path(__file__).resolve().parent / "scripts"
if str(_scripts_dir) not in sys.path:
    sys.path.insert(0, str(_scripts_dir))

from system_identification import main
from chirp import ChirpGenerator, IKChirpGenerator
from ik_registry import IKRegistry
from sysid import SystemIdentification

__all__ = [
    "main",
    "ChirpGenerator",
    "IKChirpGenerator",
    "IKRegistry",
    "SystemIdentification",
]
