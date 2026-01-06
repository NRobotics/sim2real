"""
IK function registry for system identification.

Allows registering and looking up inverse kinematics functions
that convert task-space inputs to motor angles.
"""

from __future__ import annotations

from typing import Callable


class IKRegistry:
    """Registry for inverse kinematics functions.
    
    Each IK function has signature: ik_func(*inputs) -> tuple[motor_angles]
    """

    _ik_functions: dict[str, dict] = {}

    @classmethod
    def register(
        cls,
        name: str,
        ik_func: Callable,
        input_names: list[str] | None = None,
        motor_count: int = 2,
    ) -> None:
        """Register an IK function.
        
        Args:
            name: Identifier (e.g., 'foot', 'arm')
            ik_func: Function taking input angles, returning motor angles
            input_names: Names of inputs (e.g., ['pitch', 'roll'])
            motor_count: Number of motors this IK controls
        """
        cls._ik_functions[name] = {
            "ik": ik_func,
            "input_names": input_names or ["input1", "input2"],
            "motor_count": motor_count,
        }

    @classmethod
    def get(cls, name: str) -> dict | None:
        """Get registered IK function info."""
        return cls._ik_functions.get(name)

    @classmethod
    def list_available(cls) -> list[str]:
        """List all registered IK functions."""
        return list(cls._ik_functions.keys())


def register_default_ik_functions() -> None:
    """Register default IK functions (call once at startup)."""
    # Import here to avoid circular imports
    import sys
    from pathlib import Path

    # Add parent to path for kinematics import
    parent_dir = Path(__file__).resolve().parent.parent
    if str(parent_dir) not in sys.path:
        sys.path.insert(0, str(parent_dir))

    try:
        from kinematics import ik_foot_to_motor
        IKRegistry.register(
            name="foot",
            ik_func=ik_foot_to_motor,
            input_names=["pitch", "roll"],
            motor_count=2,
        )
    except ImportError:
        pass  # Kinematics not available

