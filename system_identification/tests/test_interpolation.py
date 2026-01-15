"""
Tests for interpolation easing functions.

Tests cover:
- Boundary conditions (alpha=0 and alpha=1)
- Monotonicity (output increases with input)
- Function lookup by name
- Trapezoidal profile with custom fractions
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Add scripts directory to path for imports
_script_dir = Path(__file__).resolve().parent.parent / "scripts"
if str(_script_dir) not in sys.path:
    sys.path.insert(0, str(_script_dir))

from interpolation import (
    EasingMethod,
    cubic,
    get_easing_function,
    linear,
    list_methods,
    minimum_jerk,
    quintic,
    trapezoidal,
)


# =============================================================================
# Boundary Condition Tests
# =============================================================================


class TestBoundaryConditions:
    """Test that all easing functions return 0 at alpha=0 and 1 at alpha=1."""

    @pytest.mark.parametrize(
        "easing_fn",
        [linear, cubic, quintic, minimum_jerk, trapezoidal],
        ids=["linear", "cubic", "quintic", "minimum_jerk", "trapezoidal"],
    )
    def test_zero_at_start(self, easing_fn):
        """Test that f(0) = 0 for all easing functions."""
        assert easing_fn(0.0) == pytest.approx(0.0, abs=1e-10)

    @pytest.mark.parametrize(
        "easing_fn",
        [linear, cubic, quintic, minimum_jerk, trapezoidal],
        ids=["linear", "cubic", "quintic", "minimum_jerk", "trapezoidal"],
    )
    def test_one_at_end(self, easing_fn):
        """Test that f(1) = 1 for all easing functions."""
        assert easing_fn(1.0) == pytest.approx(1.0, abs=1e-10)


# =============================================================================
# Monotonicity Tests
# =============================================================================


class TestMonotonicity:
    """Test that easing functions are monotonically increasing."""

    @pytest.mark.parametrize(
        "easing_fn",
        [linear, cubic, quintic, minimum_jerk, trapezoidal],
        ids=["linear", "cubic", "quintic", "minimum_jerk", "trapezoidal"],
    )
    def test_monotonically_increasing(self, easing_fn):
        """Test that output increases as input increases."""
        num_samples = 100
        alphas = [i / num_samples for i in range(num_samples + 1)]
        outputs = [easing_fn(a) for a in alphas]

        for i in range(1, len(outputs)):
            assert outputs[i] >= outputs[i - 1] - 1e-10, (
                f"Non-monotonic at alpha={alphas[i]}: "
                f"f({alphas[i-1]})={outputs[i-1]} > f({alphas[i]})={outputs[i]}"
            )


# =============================================================================
# Midpoint Tests (symmetry verification)
# =============================================================================


class TestMidpoint:
    """Test midpoint values for symmetric easing functions."""

    @pytest.mark.parametrize(
        "easing_fn",
        [cubic, quintic, minimum_jerk],
        ids=["cubic", "quintic", "minimum_jerk"],
    )
    def test_midpoint_at_half(self, easing_fn):
        """Test that f(0.5) = 0.5 for symmetric S-curve easings."""
        assert easing_fn(0.5) == pytest.approx(0.5, abs=1e-10)


# =============================================================================
# Function Lookup Tests
# =============================================================================


class TestGetEasingFunction:
    """Tests for get_easing_function lookup."""

    def test_lookup_by_string(self):
        """Test looking up functions by string name."""
        fn = get_easing_function("quintic")
        assert fn is quintic

    def test_lookup_by_enum(self):
        """Test looking up functions by enum value."""
        fn = get_easing_function(EasingMethod.CUBIC)
        assert fn is cubic

    def test_lookup_case_insensitive(self):
        """Test that lookup is case-insensitive."""
        fn = get_easing_function("QUINTIC")
        assert fn is quintic

    def test_unknown_method_raises(self):
        """Test that unknown method raises ValueError."""
        with pytest.raises(ValueError, match="Unknown easing method"):
            get_easing_function("invalid_method")


# =============================================================================
# List Methods Test
# =============================================================================


class TestListMethods:
    """Test list_methods function."""

    def test_returns_all_methods(self):
        """Test that list_methods returns all available methods."""
        methods = list_methods()
        assert "linear" in methods
        assert "cubic" in methods
        assert "quintic" in methods
        assert "minimum_jerk" in methods
        assert "trapezoidal" in methods
        assert len(methods) == 5


# =============================================================================
# Trapezoidal Profile Tests
# =============================================================================


class TestTrapezoidal:
    """Tests specific to trapezoidal velocity profile."""

    def test_default_fractions(self):
        """Test trapezoidal with default 25%/50%/25% profile."""
        # At 25% (end of acceleration), should be around 0.125 position
        assert trapezoidal(0.25) == pytest.approx(0.125, abs=0.01)

        # At 75% (end of constant velocity), should be around 0.875 position
        assert trapezoidal(0.75) == pytest.approx(0.875, abs=0.01)

    def test_custom_fractions(self):
        """Test trapezoidal with custom acceleration fractions."""
        # 50% accel, 0% constant, 50% decel (triangular profile)
        result = trapezoidal(0.5, accel_fraction=0.5, decel_fraction=0.5)
        assert result == pytest.approx(0.5, abs=0.01)

    def test_invalid_fractions_fallback(self):
        """Test that invalid fractions fall back to 50/50."""
        # Fractions > 1.0 should trigger fallback
        result = trapezoidal(0.5, accel_fraction=0.8, decel_fraction=0.8)
        assert result == pytest.approx(0.5, abs=0.01)

    def test_clamps_outside_range(self):
        """Test that values outside [0,1] are clamped."""
        assert trapezoidal(-0.5) == 0.0
        assert trapezoidal(1.5) == 1.0


# =============================================================================
# Equivalence Tests
# =============================================================================


class TestEquivalence:
    """Test mathematical equivalences."""

    def test_quintic_equals_minimum_jerk(self):
        """Verify quintic and minimum_jerk produce same values."""
        for alpha in [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]:
            assert quintic(alpha) == pytest.approx(
                minimum_jerk(alpha), abs=1e-10
            ), f"Mismatch at alpha={alpha}"
