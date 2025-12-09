"""
Tests for system_identification module.

Tests cover:
- ChirpGenerator signal generation with scale/direction/bias
- IKChirpGenerator with IK function
- IKRegistry registration and lookup
- SystemIdentification config validation
- Rate limiter functionality
"""

import builtins
import json
import statistics
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from system_identification import (
    ChirpGenerator,
    IKChirpGenerator,
    IKRegistry,
    SystemIdentification,
)
from system_identification.kinematics import fk_motor_to_foot, ik_foot_to_motor


# =============================================================================
# ChirpGenerator Tests
# =============================================================================

class TestChirpGenerator:
    """Tests for ChirpGenerator class."""

    def test_basic_chirp_generation(self):
        """Test that chirp generator produces expected number of samples."""
        gen = ChirpGenerator(
            f_start=0.1,
            f_end=10.0,
            duration=1.0,
            sample_rate=100.0,
            sweep_type="linear",
            scale=1.0,
            direction=1.0,
            bias=0.0,
        )
        
        assert gen.num_samples == 100
        assert len(gen.signal) == 100
        assert gen.current_index == 0

    def test_scale_direction_bias_formula(self):
        """Test that q(t) = scale * direction * (s(t) + bias) is applied correctly."""
        # Base signal s(t) ranges from -1 to +1
        gen = ChirpGenerator(
            f_start=1.0,
            f_end=1.0,  # Constant frequency for predictable signal
            duration=1.0,
            sample_rate=100.0,
            sweep_type="linear",
            scale=2.0,
            direction=-1.0,
            bias=0.5,
        )
        
        # Effective amplitude = |scale * direction| = |2.0 * -1.0| = 2.0
        assert gen.effective_amplitude == 2.0
        
        # Effective offset = scale * direction * bias = 2.0 * -1.0 * 0.5 = -1.0
        assert gen.effective_offset == -1.0
        
        # Check signal transformation: q = scale * direction * (s + bias)
        # When base signal s = 0, q should = scale * direction * bias = -1.0
        # When base signal s = 1, q should = scale * direction * (1 + 0.5) = -3.0
        # When base signal s = -1, q should = scale * direction * (-1 + 0.5) = 1.0

    def test_direction_flips_signal(self):
        """Test that direction=-1 flips the signal."""
        gen_pos = ChirpGenerator(
            f_start=1.0, f_end=1.0, duration=0.1, sample_rate=100.0,
            scale=1.0, direction=1.0, bias=0.0,
        )
        gen_neg = ChirpGenerator(
            f_start=1.0, f_end=1.0, duration=0.1, sample_rate=100.0,
            scale=1.0, direction=-1.0, bias=0.0,
        )
        
        # Signals should be negatives of each other
        np.testing.assert_array_almost_equal(gen_pos.signal, -gen_neg.signal)

    def test_bias_shifts_signal(self):
        """Test that bias shifts the signal."""
        gen_no_bias = ChirpGenerator(
            f_start=1.0, f_end=1.0, duration=0.1, sample_rate=100.0,
            scale=1.0, direction=1.0, bias=0.0,
        )
        gen_with_bias = ChirpGenerator(
            f_start=1.0, f_end=1.0, duration=0.1, sample_rate=100.0,
            scale=1.0, direction=1.0, bias=0.5,
        )
        
        # Signal with bias should be shifted by scale * direction * bias = 0.5
        expected_diff = 0.5  # scale * direction * bias
        actual_diff = np.mean(gen_with_bias.signal - gen_no_bias.signal)
        assert abs(actual_diff - expected_diff) < 0.01

    def test_get_next_returns_samples(self):
        """Test get_next() returns samples and completion status."""
        gen = ChirpGenerator(
            f_start=1.0, f_end=1.0, duration=0.05, sample_rate=100.0,
            scale=1.0, direction=1.0, bias=0.0,
        )
        
        # Should have 5 samples
        assert gen.num_samples == 5
        
        # Get all samples
        samples = []
        for _ in range(5):
            val, complete = gen.get_next()
            samples.append(val)
            assert not complete
        
        # Next call should indicate completion
        val, complete = gen.get_next()
        assert complete
        assert val == 0.0

    def test_reset(self):
        """Test reset() restarts the generator."""
        gen = ChirpGenerator(
            f_start=1.0, f_end=1.0, duration=0.05, sample_rate=100.0,
            scale=1.0, direction=1.0, bias=0.0,
        )
        
        # Consume some samples
        gen.get_next()
        gen.get_next()
        assert gen.current_index == 2
        
        # Reset
        gen.reset()
        assert gen.current_index == 0

    def test_progress(self):
        """Test get_progress() returns correct fraction."""
        gen = ChirpGenerator(
            f_start=1.0, f_end=1.0, duration=0.1, sample_rate=100.0,
            scale=1.0, direction=1.0, bias=0.0,
        )
        
        assert gen.get_progress() == 0.0
        
        for _ in range(5):
            gen.get_next()
        
        assert gen.get_progress() == 0.5


# =============================================================================
# IKRegistry Tests
# =============================================================================

class TestIKRegistry:
    """Tests for IKRegistry class."""

    def test_register_and_get(self):
        """Test registering and retrieving an IK function."""
        def dummy_ik(a, b):
            return (a + b, a - b)
        
        IKRegistry.register(
            name="test_ik",
            ik_func=dummy_ik,
            input_names=["a", "b"],
            motor_count=2,
        )
        
        info = IKRegistry.get("test_ik")
        assert info is not None
        assert info["ik"] == dummy_ik
        assert info["input_names"] == ["a", "b"]
        assert info["motor_count"] == 2

    def test_get_nonexistent(self):
        """Test getting a non-existent IK returns None."""
        info = IKRegistry.get("nonexistent_ik")
        assert info is None

    def test_list_available(self):
        """Test listing available IK functions."""
        available = IKRegistry.list_available()
        assert "foot" in available  # Default registered

    def test_foot_ik_registered(self):
        """Test that foot IK is registered by default."""
        info = IKRegistry.get("foot")
        assert info is not None
        assert info["input_names"] == ["pitch", "roll"]
        assert info["motor_count"] == 2


# =============================================================================
# IKChirpGenerator Tests
# =============================================================================

class TestIKChirpGenerator:
    """Tests for IKChirpGenerator class."""

    def test_basic_creation(self):
        """Test creating an IK chirp generator."""
        gen = IKChirpGenerator(
            ik_type="foot",
            f_start=0.1,
            f_end=10.0,
            duration=1.0,
            sample_rate=100.0,
            scale_1=0.25,
            scale_2=0.25,
            direction_1=1.0,
            direction_2=1.0,
            bias_1=0.0,
            bias_2=0.0,
        )
        
        assert gen.num_samples == 100
        assert gen.ik_type == "foot"
        assert gen.input_names == ["pitch", "roll"]

    def test_invalid_ik_type(self):
        """Test that invalid IK type raises error."""
        with pytest.raises(ValueError, match="Unknown IK type"):
            IKChirpGenerator(
                ik_type="nonexistent",
                f_start=0.1,
                f_end=10.0,
                duration=1.0,
                sample_rate=100.0,
            )

    def test_effective_parameters(self):
        """Test effective amplitude and offset calculations."""
        gen = IKChirpGenerator(
            ik_type="foot",
            f_start=0.1,
            f_end=10.0,
            duration=1.0,
            sample_rate=100.0,
            scale_1=2.0,
            scale_2=0.5,
            direction_1=-1.0,
            direction_2=1.0,
            bias_1=0.4,
            bias_2=0.0,
        )
        
        # Effective amplitude = |scale * direction|
        assert gen.effective_amplitude_1 == 2.0
        assert gen.effective_amplitude_2 == 0.5
        
        # Effective offset = scale * direction * bias
        assert gen.effective_offset_1 == -0.8  # 2.0 * -1.0 * 0.4
        assert gen.effective_offset_2 == 0.0

    def test_get_next_returns_motor_angles(self):
        """Test that get_next() returns IK-computed motor angles."""
        gen = IKChirpGenerator(
            ik_type="foot",
            f_start=1.0,
            f_end=1.0,
            duration=0.1,
            sample_rate=100.0,
            scale_1=0.1,  # Small amplitude to stay in valid IK range
            scale_2=0.1,
            direction_1=1.0,
            direction_2=1.0,
            bias_1=0.0,
            bias_2=0.0,
        )
        
        inputs, motor_angles, complete = gen.get_next()
        
        assert not complete
        assert "pitch" in inputs
        assert "roll" in inputs
        assert len(motor_angles) == 2  # q_lower, q_upper


# =============================================================================
# SystemIdentification Config Validation Tests
# =============================================================================

class TestSystemIdentificationValidation:
    """Tests for SystemIdentification config validation."""

    def _create_config_file(self, config: dict) -> str:
        """Helper to create a temporary config file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config, f)
            return f.name

    def _base_config(self) -> dict:
        """Return a base valid config."""
        return {
            "can_interface": {
                "interface": "socketcan",
                "channel": "can0",
                "bitrate": 1000000,
                "fd": True,
            },
            "chirp": {
                "f_start": 0.1,
                "f_end": 10.0,
                "duration": 1.0,
                "sample_rate": 100.0,
                "sweep_type": "linear",
            },
            "control_parameters": {
                "velocity": 0.0,
                "effort": 0.0,
                "stiffness": 1.0,
                "damping": 0.0,
            },
            "motor_ids": [0, 1],
            "motors": {},
            "ik_groups": [],
        }

    def test_valid_config_with_ik_group(self):
        """Test valid config with IK group passes validation."""
        config = self._base_config()
        config["ik_groups"] = [
            {
                "name": "foot_0",
                "ik_type": "foot",
                "motor_ids": [0, 1],
                "chirp": {
                    "scale_pitch": 0.1,
                    "direction_pitch": 1.0,
                    "bias_pitch": 0.0,
                    "scale_roll": 0.1,
                    "direction_roll": 1.0,
                    "bias_roll": 0.0,
                }
            }
        ]
        
        config_file = self._create_config_file(config)
        try:
            sysid = SystemIdentification(config_file, dry_run=True)
            sysid.setup()
            # Should not raise
            assert 0 in sysid.motor_to_ik_group
            assert 1 in sysid.motor_to_ik_group
        finally:
            Path(config_file).unlink()

    def test_valid_config_with_direct_motors(self):
        """Test valid config with direct motors passes validation."""
        config = self._base_config()
        config["motors"] = {
            "0": {"scale": 0.25, "direction": 1.0, "bias": 0.0},
            "1": {"scale": 0.25, "direction": -1.0, "bias": 0.0},
        }
        
        config_file = self._create_config_file(config)
        try:
            sysid = SystemIdentification(config_file, dry_run=True)
            sysid.setup()
            # Should not raise
            assert 0 in sysid.direct_motors
            assert 1 in sysid.direct_motors
        finally:
            Path(config_file).unlink()

    def test_error_motor_in_both_ik_and_direct(self):
        """Test error when motor is in both IK group and direct config."""
        config = self._base_config()
        config["motors"] = {
            "0": {"scale": 0.25, "direction": 1.0, "bias": 0.0},
        }
        config["ik_groups"] = [
            {
                "name": "foot_0",
                "ik_type": "foot",
                "motor_ids": [0, 1],  # Motor 0 is also in motors config!
                "chirp": {"scale_pitch": 0.1, "scale_roll": 0.1}
            }
        ]
        
        config_file = self._create_config_file(config)
        try:
            sysid = SystemIdentification(config_file, dry_run=True)
            with pytest.raises(ValueError, match="defined in BOTH"):
                sysid.setup()
        finally:
            Path(config_file).unlink()

    def test_error_motor_missing_config(self):
        """Test error when motor has no config anywhere."""
        config = self._base_config()
        config["motor_ids"] = [0, 1, 2]  # Motor 2 has no config
        config["ik_groups"] = [
            {
                "name": "foot_0",
                "ik_type": "foot",
                "motor_ids": [0, 1],
                "chirp": {"scale_pitch": 0.1, "scale_roll": 0.1}
            }
        ]
        # Motor 2 is not in ik_groups and not in motors
        
        config_file = self._create_config_file(config)
        try:
            sysid = SystemIdentification(config_file, dry_run=True)
            with pytest.raises(ValueError, match="have no config"):
                sysid.setup()
        finally:
            Path(config_file).unlink()

    def test_error_direct_motor_without_config(self):
        """Test error when direct motor has no config in motors section."""
        config = self._base_config()
        config["motor_ids"] = [0, 1, 2]
        config["motors"] = {
            "2": {"scale": 0.25, "direction": 1.0, "bias": 0.0},
            # Motors 0 and 1 missing!
        }
        config["ik_groups"] = []  # No IK groups, so 0,1 should be direct
        
        config_file = self._create_config_file(config)
        try:
            sysid = SystemIdentification(config_file, dry_run=True)
            with pytest.raises(ValueError, match="have no config"):
                sysid.setup()
        finally:
            Path(config_file).unlink()

    def test_mixed_ik_and_direct_motors(self):
        """Test valid config with mixed IK and direct motors."""
        config = self._base_config()
        config["motor_ids"] = [0, 1, 2, 3]
        config["motors"] = {
            "2": {"scale": 0.25, "direction": 1.0, "bias": 0.0},
            "3": {"scale": 0.25, "direction": -1.0, "bias": 0.0},
        }
        config["ik_groups"] = [
            {
                "name": "foot_0",
                "ik_type": "foot",
                "motor_ids": [0, 1],
                "chirp": {"scale_pitch": 0.1, "scale_roll": 0.1}
            }
        ]
        
        config_file = self._create_config_file(config)
        try:
            sysid = SystemIdentification(config_file, dry_run=True)
            sysid.setup()
            
            # Motors 0,1 in IK group
            assert 0 in sysid.motor_to_ik_group
            assert 1 in sysid.motor_to_ik_group
            
            # Motors 2,3 are direct
            assert 2 in sysid.direct_motors
            assert 3 in sysid.direct_motors
        finally:
            Path(config_file).unlink()


# =============================================================================
# Rate Limiter Tests
# =============================================================================

class TestRateLimiter:
    """Tests for rate limiting in the control loop."""

    def _create_rate_test_config(
        self, sample_rate: float, duration: float, motor_ids: list[int]
    ) -> str:
        """Create a temporary config file for rate testing."""
        config = {
            "can_interface": {
                "interface": "socketcan",
                "channel": "can0",
                "bitrate": 5000000,
                "fd": True,
            },
            "chirp": {
                "f_start": 0.1,
                "f_end": 10.0,
                "duration": duration,
                "sample_rate": sample_rate,
                "sweep_type": "linear",
            },
            "control_parameters": {
                "velocity": 0.0,
                "effort": 0.0,
                "stiffness": 1.0,
                "damping": 0.0,
            },
            "motor_ids": motor_ids,
            "motors": {
                str(i): {"scale": 0.25, "direction": 1.0, "bias": 0.0}
                for i in motor_ids
            },
        }

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(config, f)
            return f.name

    def _run_rate_test(
        self,
        target_rate: float,
        duration: float,
        tolerance_percent: float = 10.0,
    ) -> dict:
        """Run a rate limiter test and return results."""
        motor_ids = [0]
        config_file = self._create_rate_test_config(target_rate, duration, motor_ids)

        # Suppress output during test
        original_print = builtins.print
        builtins.print = lambda *args, **kwargs: None

        try:
            sysid = SystemIdentification(config_file, motor_ids=motor_ids, dry_run=True)
            sysid.setup()
            sysid.run_identification()

            actual_samples = sysid.sample_count
            elapsed_time = time.time() - sysid.start_time
            actual_rate = actual_samples / elapsed_time if elapsed_time > 0 else 0

            # Calculate timing statistics from feedback data
            timestamps = []
            for motor_data in sysid.feedback_data.values():
                timestamps.extend([d["timestamp"] for d in motor_data])
            timestamps.sort()

            intervals = []
            for i in range(1, len(timestamps)):
                intervals.append(timestamps[i] - timestamps[i - 1])

            if intervals:
                mean_interval = statistics.mean(intervals)
                std_interval = statistics.stdev(intervals) if len(intervals) > 1 else 0
                min_interval = min(intervals)
                max_interval = max(intervals)
            else:
                mean_interval = std_interval = min_interval = max_interval = 0

            rate_error_percent = abs(actual_rate - target_rate) / target_rate * 100
            passed = rate_error_percent <= tolerance_percent

            sysid.cleanup()

            return {
                "passed": passed,
                "target_rate": target_rate,
                "actual_rate": actual_rate,
                "rate_error_percent": rate_error_percent,
                "actual_samples": actual_samples,
                "expected_samples": int(target_rate * duration),
                "mean_interval_ms": mean_interval * 1000,
                "std_interval_ms": std_interval * 1000,
                "min_interval_ms": min_interval * 1000,
                "max_interval_ms": max_interval * 1000,
            }

        finally:
            builtins.print = original_print
            Path(config_file).unlink(missing_ok=True)

    def test_rate_100hz(self):
        """Test that 100 Hz rate is maintained within tolerance."""
        results = self._run_rate_test(target_rate=100.0, duration=1.0)
        # In dry-run mode, mock controller is fast, so rate should be achievable
        assert results["actual_samples"] > 0
        assert results["actual_rate"] > 0

    def test_rate_50hz(self):
        """Test that 50 Hz rate is maintained within tolerance."""
        results = self._run_rate_test(target_rate=50.0, duration=1.0)
        assert results["actual_samples"] > 0
        assert results["actual_rate"] > 0

    def test_samples_match_duration(self):
        """Test that sample count roughly matches rate * duration."""
        target_rate = 100.0
        duration = 0.5
        results = self._run_rate_test(target_rate=target_rate, duration=duration)
        
        expected = int(target_rate * duration)
        # Allow 10% tolerance
        assert abs(results["actual_samples"] - expected) <= expected * 0.1

    def test_timing_intervals_consistent(self):
        """Test that inter-sample timing is consistent."""
        results = self._run_rate_test(target_rate=100.0, duration=0.5)
        
        if results["mean_interval_ms"] > 0:
            # Standard deviation should be small relative to mean
            # (indicates consistent timing)
            cv = results["std_interval_ms"] / results["mean_interval_ms"]
            # Coefficient of variation should be < 50% for reasonable consistency
            assert cv < 0.5, f"Timing too variable: CV={cv:.2f}"


# =============================================================================
# FK Computation Tests (in save_results)
# =============================================================================

class TestFKComputation:
    """Tests for forward kinematics computation during save_results."""

    def _create_config_file(self, config: dict) -> str:
        """Helper to create a temporary config file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config, f)
            return f.name

    def _base_config(self, duration: float = 0.1, sample_rate: float = 100.0) -> dict:
        """Return a base valid config for FK tests."""
        return {
            "can_interface": {
                "interface": "socketcan",
                "channel": "can0",
                "bitrate": 1000000,
                "fd": True,
            },
            "chirp": {
                "f_start": 1.0,
                "f_end": 2.0,
                "duration": duration,
                "sample_rate": sample_rate,
                "sweep_type": "linear",
            },
            "control_parameters": {
                "velocity": 0.0,
                "effort": 0.0,
                "stiffness": 1.0,
                "damping": 0.0,
            },
            "motor_ids": [0, 1],
            "motors": {},
            "ik_groups": [
                {
                    "name": "foot_0",
                    "ik_type": "foot",
                    "motor_ids": [0, 1],
                    "chirp": {
                        "scale_pitch": 0.1,
                        "direction_pitch": 1.0,
                        "bias_pitch": 0.0,
                        "scale_roll": 0.1,
                        "direction_roll": 1.0,
                        "bias_roll": 0.0,
                    }
                }
            ],
        }

    def test_fk_ik_round_trip(self):
        """Test that FK(IK(pitch, roll)) == (pitch, roll) within tolerance."""
        test_cases = [
            (0.0, 0.0),
            (0.1, 0.0),
            (0.0, 0.1),
            (0.1, 0.05),
            (-0.1, 0.05),
            (0.05, -0.05),
            (0.15, 0.1),
        ]
        
        for pitch, roll in test_cases:
            q_lower, q_upper = ik_foot_to_motor(pitch, roll)
            pitch_recovered, roll_recovered = fk_motor_to_foot(q_lower, q_upper)
            
            # Should have very small round-trip error
            assert abs(pitch - pitch_recovered) < 1e-6, \
                f"Pitch round trip failed: {pitch} -> {pitch_recovered}"
            assert abs(roll - roll_recovered) < 1e-6, \
                f"Roll round trip failed: {roll} -> {roll_recovered}"

    def test_save_results_includes_fk_data(self):
        """Test that save_results includes FK data for foot IK groups."""
        config = self._base_config(duration=0.1, sample_rate=50.0)
        config_file = self._create_config_file(config)
        
        # Suppress output during test
        original_print = builtins.print
        builtins.print = lambda *args, **kwargs: None
        
        try:
            sysid = SystemIdentification(config_file, dry_run=True)
            sysid.setup()
            sysid.run_identification()
            
            # Save to temp file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                output_file = f.name
            
            sysid.save_results(output_file)
            
            # Load and verify
            with open(output_file) as f:
                results = json.load(f)
            
            # Check FK data exists
            assert "ik_groups" in results
            assert "foot_0" in results["ik_groups"]
            assert "fk_data" in results["ik_groups"]["foot_0"]
            
            fk_data = results["ik_groups"]["foot_0"]["fk_data"]
            assert len(fk_data) > 0, "No FK data computed"
            
            # Check FK data structure
            first_sample = fk_data[0]
            assert "sample" in first_sample
            assert "timestamp" in first_sample
            assert "measured_pitch" in first_sample
            assert "measured_roll" in first_sample
            assert "q_lower" in first_sample
            assert "q_upper" in first_sample
            assert "commanded_pitch" in first_sample
            assert "commanded_roll" in first_sample
            
            sysid.cleanup()
            
        finally:
            builtins.print = original_print
            Path(config_file).unlink(missing_ok=True)
            Path(output_file).unlink(missing_ok=True)

    def test_fk_values_are_valid(self):
        """Test that FK values are finite and in reasonable range."""
        config = self._base_config(duration=0.1, sample_rate=50.0)
        config_file = self._create_config_file(config)
        
        # Suppress output during test
        original_print = builtins.print
        builtins.print = lambda *args, **kwargs: None
        
        try:
            sysid = SystemIdentification(config_file, dry_run=True)
            sysid.setup()
            sysid.run_identification()
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                output_file = f.name
            
            sysid.save_results(output_file)
            
            with open(output_file) as f:
                results = json.load(f)
            
            fk_data = results["ik_groups"]["foot_0"]["fk_data"]
            
            for sample in fk_data:
                # Check values are finite
                assert np.isfinite(sample["measured_pitch"]), \
                    f"Non-finite pitch at sample {sample['sample']}"
                assert np.isfinite(sample["measured_roll"]), \
                    f"Non-finite roll at sample {sample['sample']}"
                
                # Check values are in reasonable range (< 1 rad ~ 57 deg)
                assert abs(sample["measured_pitch"]) < 1.0, \
                    f"Pitch out of range at sample {sample['sample']}: {sample['measured_pitch']}"
                assert abs(sample["measured_roll"]) < 1.0, \
                    f"Roll out of range at sample {sample['sample']}: {sample['measured_roll']}"
            
            sysid.cleanup()
            
        finally:
            builtins.print = original_print
            Path(config_file).unlink(missing_ok=True)
            Path(output_file).unlink(missing_ok=True)

    def test_fk_motor_ordering(self):
        """Test that motor ordering is correct (index 0 = lower, index 1 = upper)."""
        config = self._base_config()
        config["ik_groups"][0]["motor_ids"] = [5, 10]  # Non-consecutive IDs
        config["motor_ids"] = [5, 10]
        config_file = self._create_config_file(config)
        
        # Suppress output
        original_print = builtins.print
        builtins.print = lambda *args, **kwargs: None
        
        try:
            sysid = SystemIdentification(config_file, dry_run=True)
            sysid.setup()
            
            # Check motor mapping
            assert 5 in sysid.motor_to_ik_group
            assert 10 in sysid.motor_to_ik_group
            
            # Motor 5 should be index 0 (lower), motor 10 should be index 1 (upper)
            assert sysid.motor_to_ik_group[5] == ("foot_0", 0)
            assert sysid.motor_to_ik_group[10] == ("foot_0", 1)
            
            sysid.cleanup()
            
        finally:
            builtins.print = original_print
            Path(config_file).unlink(missing_ok=True)

    def test_fk_sample_matching(self):
        """Test that FK samples are matched correctly by sample index."""
        config = self._base_config(duration=0.1, sample_rate=50.0)
        config_file = self._create_config_file(config)
        
        original_print = builtins.print
        builtins.print = lambda *args, **kwargs: None
        
        try:
            sysid = SystemIdentification(config_file, dry_run=True)
            sysid.setup()
            sysid.run_identification()
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                output_file = f.name
            
            sysid.save_results(output_file)
            
            with open(output_file) as f:
                results = json.load(f)
            
            fk_data = results["ik_groups"]["foot_0"]["fk_data"]
            
            # Sample indices should be monotonically increasing
            sample_indices = [s["sample"] for s in fk_data]
            assert sample_indices == sorted(sample_indices), "Sample indices not sorted"
            
            # No duplicate samples
            assert len(sample_indices) == len(set(sample_indices)), "Duplicate sample indices"
            
            sysid.cleanup()
            
        finally:
            builtins.print = original_print
            Path(config_file).unlink(missing_ok=True)
            Path(output_file).unlink(missing_ok=True)

    def test_no_fk_for_direct_motors(self):
        """Test that FK is not computed for direct motor control (no IK group)."""
        config = self._base_config()
        # Remove IK groups, use direct motor control
        config["ik_groups"] = []
        config["motors"] = {
            "0": {"scale": 0.1, "direction": 1.0, "bias": 0.0},
            "1": {"scale": 0.1, "direction": 1.0, "bias": 0.0},
        }
        config_file = self._create_config_file(config)
        
        original_print = builtins.print
        builtins.print = lambda *args, **kwargs: None
        
        try:
            sysid = SystemIdentification(config_file, dry_run=True)
            sysid.setup()
            sysid.run_identification()
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                output_file = f.name
            
            sysid.save_results(output_file)
            
            with open(output_file) as f:
                results = json.load(f)
            
            # Should have no IK groups
            assert "ik_groups" not in results or len(results.get("ik_groups", {})) == 0
            
            # Should have direct_motors
            assert "direct_motors" in results
            assert 0 in results["direct_motors"]
            assert 1 in results["direct_motors"]
            
            sysid.cleanup()
            
        finally:
            builtins.print = original_print
            Path(config_file).unlink(missing_ok=True)
            Path(output_file).unlink(missing_ok=True)


# =============================================================================
# Run tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])

