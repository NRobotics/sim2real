"""
System identification scripts package.

Modules:
- system_identification: Main CLI entry point
- sysid: Core SystemIdentification class
- chirp: Chirp signal generators
- ik_registry: IK function registry
- controllers: Motor controller implementations
- results: Result saving utilities
- realtime: Real-time scheduling utilities
"""

import sys
from pathlib import Path

# Ensure scripts directory is in path for imports
_script_dir = Path(__file__).resolve().parent
if str(_script_dir) not in sys.path:
    sys.path.insert(0, str(_script_dir))

from ik_registry import IKRegistry, register_default_ik_functions
from chirp import ChirpGenerator, IKChirpGenerator
from sysid import SystemIdentification
from controllers import MockMotorCANController

__all__ = [
    "IKRegistry",
    "register_default_ik_functions",
    "ChirpGenerator",
    "IKChirpGenerator",
    "SystemIdentification",
    "MockMotorCANController",
]
