"""
Interpolation easing functions for smooth motor motion.

Provides multiple easing methods for safe initialization and finalization
of robot motors, ensuring smooth velocity and acceleration profiles.

All functions take alpha in [0, 1] and return eased_alpha in [0, 1].
"""

from __future__ import annotations

from enum import Enum
from typing import Callable


class EasingMethod(str, Enum):
    """Available interpolation easing methods."""

    LINEAR = "linear"
    CUBIC = "cubic"
    QUINTIC = "quintic"
    MINIMUM_JERK = "minimum_jerk"
    TRAPEZOIDAL = "trapezoidal"
    CUBIC_HERMITE = "cubic_hermite"  # Velocity-aware interpolation


# Type alias for easing functions
EasingFunction = Callable[[float], float]


def linear(alpha: float) -> float:
    """
    Linear interpolation (no easing).

    Properties:
    - Position: C0 continuous (may have discontinuous velocity)
    - Velocity: Instantaneous start/stop (infinite acceleration at endpoints)

    Use when: Maximum speed is priority, endpoints don't matter.

    Args:
        alpha: Input parameter in [0, 1]

    Returns:
        Unchanged alpha value
    """
    return alpha


def cubic(alpha: float) -> float:
    """
    Cubic Hermite smoothstep: 3α² - 2α³

    Properties:
    - Position: C1 continuous (smooth position at endpoints)
    - Velocity: Zero at endpoints (starts/ends at rest)
    - Acceleration: Non-zero at endpoints (abrupt torque change)

    Use when: Smooth starts and stops are needed, but jerk is acceptable.

    Args:
        alpha: Input parameter in [0, 1]

    Returns:
        Eased alpha using cubic polynomial
    """
    return alpha * alpha * (3.0 - 2.0 * alpha)


def quintic(alpha: float) -> float:
    """
    Quintic smootherstep: 6α⁵ - 15α⁴ + 10α³

    Properties:
    - Position: C2 continuous (smooth position and velocity at endpoints)
    - Velocity: Zero at endpoints
    - Acceleration: Zero at endpoints (smooth torque at start/end)
    - Jerk: Non-zero at endpoints

    Use when: Smooth acceleration profiles are needed.

    Args:
        alpha: Input parameter in [0, 1]

    Returns:
        Eased alpha using quintic polynomial
    """
    return alpha * alpha * alpha * (alpha * (alpha * 6.0 - 15.0) + 10.0)


def minimum_jerk(alpha: float) -> float:
    """
    Minimum jerk trajectory: 10α³ - 15α⁴ + 6α⁵

    Also known as 7th-order polynomial trajectory in some contexts.
    Minimizes the integral of jerk squared, producing the smoothest
    possible motion profile for point-to-point movement.

    Properties:
    - Position: C3 continuous
    - Velocity: Zero at endpoints
    - Acceleration: Zero at endpoints
    - Jerk: Zero at endpoints (smoothest possible motor torque)

    Use when: Maximum smoothness is required (high-velocity applications).

    Note: This is mathematically equivalent to quintic for position
    (same polynomial with rearranged coefficients).

    Args:
        alpha: Input parameter in [0, 1]

    Returns:
        Eased alpha using minimum jerk polynomial
    """
    # This is the same polynomial as quintic, written in expanded form:
    # 10α³ - 15α⁴ + 6α⁵ = α³(10 - 15α + 6α²) = α³(6α² - 15α + 10)
    # Which equals: α³(α(6α - 15) + 10)
    alpha3 = alpha * alpha * alpha
    return alpha3 * (alpha * (6.0 * alpha - 15.0) + 10.0)


def trapezoidal(
    alpha: float,
    accel_fraction: float = 0.25,
    decel_fraction: float = 0.25,
) -> float:
    """
    Trapezoidal velocity profile.

    Linear acceleration, constant velocity, linear deceleration.

    Creates a motion profile with:
    - Acceleration phase: velocity ramps up linearly
    - Constant velocity phase: velocity stays constant
    - Deceleration phase: velocity ramps down linearly

    Properties:
    - Position: C0 continuous (velocity has corners)
    - Velocity: Piecewise linear, zero at endpoints
    - Acceleration: Piecewise constant (step changes)

    Use when: Velocity-limited motion is needed (e.g., motor speed limits).

    Args:
        alpha: Input parameter in [0, 1]
        accel_fraction: Fraction of motion for acceleration (default 0.25)
        decel_fraction: Fraction of motion for deceleration (default 0.25)

    Returns:
        Eased alpha following trapezoidal velocity profile
    """
    # Validate fractions
    if accel_fraction + decel_fraction > 1.0:
        # Fallback to symmetric if invalid
        accel_fraction = decel_fraction = 0.5

    const_fraction = 1.0 - accel_fraction - decel_fraction

    # Phase boundaries
    t1 = accel_fraction
    t2 = accel_fraction + const_fraction

    if alpha <= 0.0:
        return 0.0
    if alpha >= 1.0:
        return 1.0

    if alpha < t1:
        # Acceleration phase: parabolic position (s = 0.5*a*t²)
        # Normalized so that velocity at t1 = v_max
        # Position at t1 = 0.5 * accel_fraction (half of triangle area)
        normalized = alpha / accel_fraction
        return 0.5 * accel_fraction * normalized * normalized

    if alpha < t2:
        # Constant velocity phase: linear position
        # Position at t1 = 0.5 * accel_fraction
        # Velocity = 1.0 (normalized)
        pos_at_t1 = 0.5 * accel_fraction
        return pos_at_t1 + (alpha - t1)

    # Deceleration phase: parabolic position (inverted)
    remaining = 1.0 - alpha
    norm_rem = remaining / decel_fraction
    return 1.0 - 0.5 * decel_fraction * norm_rem * norm_rem


def cubic_hermite_spline(
    alpha: float,
    start_pos: float,
    end_pos: float,
    start_vel: float,
    end_vel: float = 0.0,
    duration: float = 1.0,
) -> float:
    """
    Cubic Hermite spline interpolation with velocity boundary conditions.

    Computes position using cubic polynomial that matches:
    - Position at start: start_pos
    - Position at end: end_pos
    - Velocity at start: start_vel (in units/s, scaled by duration)
    - Velocity at end: end_vel (default 0 = come to rest)

    The Hermite basis functions are:
    - h00(t) = 2t³ - 3t² + 1  (position at start)
    - h10(t) = t³ - 2t² + t   (velocity at start)
    - h01(t) = -2t³ + 3t²     (position at end)
    - h11(t) = t³ - t²        (velocity at end)

    Args:
        alpha: Normalized time in [0, 1]
        start_pos: Position at alpha=0
        end_pos: Position at alpha=1
        start_vel: Velocity at alpha=0 (units per second)
        end_vel: Velocity at alpha=1 (default 0 = stop)
        duration: Total interpolation time in seconds (for velocity scaling)

    Returns:
        Interpolated position at given alpha
    """
    # Clamp alpha to [0, 1]
    t = max(0.0, min(1.0, alpha))

    # Precompute powers
    t2 = t * t
    t3 = t2 * t

    # Hermite basis functions
    h00 = 2 * t3 - 3 * t2 + 1  # Position weight at start
    h10 = t3 - 2 * t2 + t  # Velocity weight at start
    h01 = -2 * t3 + 3 * t2  # Position weight at end
    h11 = t3 - t2  # Velocity weight at end

    # Scale velocities by duration (convert from units/s to units)
    # The spline expects tangent magnitude, not rate
    v0_scaled = start_vel * duration
    v1_scaled = end_vel * duration

    return h00 * start_pos + h10 * v0_scaled + h01 * end_pos + h11 * v1_scaled


def get_easing_function(method: str | EasingMethod) -> EasingFunction:
    """
    Get an easing function by name.

    Args:
        method: Easing method name or enum value

    Returns:
        Easing function that takes alpha and returns eased alpha

    Raises:
        ValueError: If method is unknown
    """
    if isinstance(method, EasingMethod):
        method = method.value

    method = method.lower()

    easing_functions: dict[str, EasingFunction] = {
        "linear": linear,
        "cubic": cubic,
        "quintic": quintic,
        "minimum_jerk": minimum_jerk,
        "trapezoidal": trapezoidal,
    }

    if method not in easing_functions:
        avail = ", ".join(easing_functions.keys())
        raise ValueError(f"Unknown easing method '{method}'. Available: {avail}")

    return easing_functions[method]


def list_methods() -> list[str]:
    """Return list of available easing method names."""
    return [e.value for e in EasingMethod]


# Default easing method for backward compatibility
DEFAULT_METHOD = EasingMethod.QUINTIC
