#!/usr/bin/env python3
"""
System Identification Script for Robot Actuators
Sends chirp signals to motors and collects feedback data

Usage:
    # Run with specific motor CAN IDs
    python system_identification.py --motors 0 2 5

    # Dry run (no hardware) for testing
    python system_identification.py --motors 0 1 --dry-run

    # Use motor IDs from config file
    python system_identification.py --config config/config.json
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
import traceback
from datetime import datetime
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

import numpy as np

from humanoid_messages.can import (
    ConfigurationData,
    ControlData,
    FeedbackData,
    MotorCANController,
)

# Import kinematics for IK-based control and FK for saving measured foot state
try:
    from ..kinematics import ik_foot_to_motor, fk_motor_to_foot
except ImportError:
    from kinematics import ik_foot_to_motor, fk_motor_to_foot


# ==========================================
# IK REGISTRY - Register different IK functions here
# ==========================================
class IKRegistry:
    """Registry for inverse kinematics functions.
    
    Each IK function should have the signature:
        ik_func(input1, input2, ...) -> (motor1, motor2, ...)
    """
    
    _ik_functions: dict[str, dict] = {}
    
    @classmethod
    def register(cls, name: str, ik_func, input_names=None, motor_count=2):
        """Register an IK function.
        
        Args:
            name: Identifier for this IK type (e.g., 'foot', 'arm')
            ik_func: Function that takes input angles and returns motor angles
            input_names: Names of inputs (e.g., ['pitch', 'roll'])
            motor_count: Number of motors this IK controls
        """
        cls._ik_functions[name] = {
            "ik": ik_func,
            "input_names": input_names or ["input1", "input2"],
            "motor_count": motor_count,
        }
    
    @classmethod
    def get(cls, name: str):
        """Get registered IK function info."""
        return cls._ik_functions.get(name)
    
    @classmethod
    def list_available(cls):
        """List all registered IK functions."""
        return list(cls._ik_functions.keys())


# Register the foot IK by default
IKRegistry.register(
    name="foot",
    ik_func=ik_foot_to_motor,
    input_names=["pitch", "roll"],
    motor_count=2,
)


class MockMotorCANController:
    """Mock controller for dry-run mode (no hardware required)"""

    def __init__(self, **kwargs):
        self._config_callbacks: dict = {}
        self._feedback_callbacks: dict = {}
        print("[DRY-RUN] Mock CAN controller initialized")

    def start(self):
        print("[DRY-RUN] Mock CAN controller started")

    def stop(self):
        print("[DRY-RUN] Mock CAN controller stopped")

    def set_config_callback(self, can_id: int, callback):
        self._config_callbacks[can_id] = callback

    def set_feedback_callback(self, can_id: int, callback):
        self._feedback_callbacks[can_id] = callback

    def get_motor_configuration(self, can_id: int):
        """Simulate receiving motor configuration"""
        if can_id in self._config_callbacks:
            # Create mock configuration with correct fields
            mock_config = ConfigurationData(
                can_id=can_id,
                device_type=0,
                inverse_direction=False,
                endstop_alignment_inverse=False,
                endstop_alignment_skip=False,
                endstop_zero_offset=0.0,
                endstop_damping=0.1,
                endstop_position_min=-3.14159,
                endstop_position_max=3.14159,
            )
            self._config_callbacks[can_id](can_id, mock_config)

    def start_motor(self, can_id: int):
        print(f"[DRY-RUN] Motor {can_id} started")

    def stop_all_motors(self):
        print("[DRY-RUN] All motors stopped")

    def send_kinematics_for_motor(self, can_id: int, control_data: ControlData):
        """Simulate receiving feedback after sending control"""
        if can_id in self._feedback_callbacks:
            # Simulate feedback with some noise
            mock_feedback = FeedbackData(
                angle=control_data.angle + np.random.normal(0, 0.01),
                velocity=control_data.velocity + np.random.normal(0, 0.1),
                effort=control_data.effort + np.random.normal(0, 0.05),
                voltage=24.0 + np.random.normal(0, 0.1),
                temp_motor=35.0 + np.random.normal(0, 1.0),
                temp_pcb=30.0 + np.random.normal(0, 0.5),
                flags=0,
            )
            self._feedback_callbacks[can_id](can_id, mock_feedback)


class ChirpGenerator:
    """
    Generate chirp (frequency sweep) signals for system identification.
    
    Uses the same transformation as IK inputs:
        q(t) = scale * direction * (s(t) + bias)
    
    Where s(t) = sin(phase(t)) is the base chirp signal ranging -1 to +1.
    """

    def __init__(
        self,
        f_start: float,
        f_end: float,
        duration: float,
        sample_rate: float,
        sweep_type: str = "linear",
        scale: float = 1.0,
        direction: float = 1.0,
        bias: float = 0.0,
    ):
        """
        Initialize chirp generator.
        
        Args:
            f_start, f_end: Frequency sweep range [Hz]
            duration: Total duration [s]
            sample_rate: Sample rate [Hz]
            sweep_type: 'linear', 'logarithmic', or 'exponential'
            scale: Scale factor [rad]
            direction: Direction multiplier (+1 or -1)
            bias: Bias added to base signal before scaling
            
        The transformation is:
            q(t) = scale * direction * (s(t) + bias)
            
        Where s(t) = sin(phase(t)) ranges from -1 to +1.
        """
        self.f_start = f_start
        self.f_end = f_end
        self.duration = duration
        self.sample_rate = sample_rate
        self.sweep_type = sweep_type
        
        self.scale = scale
        self.direction = direction
        self.bias = bias

        # Pre-generate the chirp signal
        self.num_samples = int(duration * sample_rate)
        self.time = np.linspace(0, duration, self.num_samples)
        self.base_signal = self._generate_base_chirp()
        self.signal = self._transform_signal()

        self.current_index = 0
        
        # Compute effective parameters for logging
        self.effective_amplitude = abs(self.scale * self.direction)
        self.effective_offset = self.scale * self.direction * self.bias

    def _generate_base_chirp(self) -> np.ndarray:
        """Generate base chirp signal s(t) = sin(phase(t)), ranging -1 to +1"""
        if self.sweep_type == "linear":
            phase = (
                2
                * np.pi
                * (self.f_start * self.time + (self.f_end - self.f_start) * self.time**2 / (2 * self.duration))
            )
        elif self.sweep_type == "logarithmic":
            k = (self.f_end / self.f_start) ** (1 / self.duration)
            phase = 2 * np.pi * self.f_start * (k**self.time - 1) / np.log(k)
        elif self.sweep_type == "exponential":
            phase = (
                2
                * np.pi
                * self.f_start
                * self.duration
                / np.log(self.f_end / self.f_start)
                * (np.exp(self.time * np.log(self.f_end / self.f_start) / self.duration) - 1)
            )
        else:
            raise ValueError(f"Unknown sweep type: {self.sweep_type}")

        return np.sin(phase)

    def _transform_signal(self) -> np.ndarray:
        """Transform base signal: q(t) = scale * direction * (s(t) + bias)"""
        return self.scale * self.direction * (self.base_signal + self.bias)

    def get_next(self) -> tuple[float, bool]:
        """Get next sample value. Returns (value, is_complete)"""
        if self.current_index >= self.num_samples:
            return 0.0, True

        value = self.signal[self.current_index]
        self.current_index += 1
        return float(value), False

    def reset(self) -> None:
        """Reset to beginning of signal"""
        self.current_index = 0

    def get_progress(self) -> float:
        """Get progress as fraction 0-1"""
        return self.current_index / self.num_samples


class IKChirpGenerator:
    """
    Generate chirp signals in IK input space (e.g., foot pitch/roll).
    
    Each IK input (pitch, roll) is treated as an independent 1-DOF joint.
    Each gets the SAME base chirp s(t), but transformed with its own parameters:
    
        q_j(t) = scale_j * direction_j * (s(t) + bias_j)
    
    Where s(t) = sin(phase(t)) is the base chirp signal.
    
    This means:
        - Effective offset = scale * direction * bias
        - Effective amplitude = scale * direction
    """

    def __init__(
        self,
        ik_type: str,
        f_start: float,
        f_end: float,
        duration: float,
        sample_rate: float,
        sweep_type: str = "linear",
        # Per-input transformation parameters: q = scale * direction * (s(t) + bias)
        scale_1: float = 1.0,
        scale_2: float = 1.0,
        direction_1: float = 1.0,
        direction_2: float = 1.0,
        bias_1: float = 0.0,
        bias_2: float = 0.0,
    ):
        """
        Initialize IK chirp generator.
        
        Args:
            ik_type: Registered IK function name (e.g., 'foot')
            f_start, f_end: Frequency sweep range [Hz]
            duration: Total duration [s]
            sample_rate: Sample rate [Hz]
            sweep_type: 'linear', 'logarithmic', or 'exponential'
            scale_1, scale_2: Scale factors for each input [rad]
            direction_1, direction_2: Direction multipliers (+1 or -1)
            bias_1, bias_2: Bias added to base signal before scaling
            
        The transformation for each input is:
            input_j(t) = scale_j * direction_j * (s(t) + bias_j)
            
        Where s(t) = sin(phase(t)) ranges from -1 to +1.
        """
        self.ik_type = ik_type
        self.ik_info = IKRegistry.get(ik_type)
        if self.ik_info is None:
            available = IKRegistry.list_available()
            raise ValueError(f"Unknown IK type: {ik_type}. Available: {available}")
        
        self.ik_func = self.ik_info["ik"]
        self.input_names = self.ik_info["input_names"]
        
        self.f_start = f_start
        self.f_end = f_end
        self.duration = duration
        self.sample_rate = sample_rate
        self.sweep_type = sweep_type

        # Per-input transformation: q = scale * direction * (s(t) + bias)
        self.scale_1 = scale_1
        self.scale_2 = scale_2
        self.direction_1 = direction_1
        self.direction_2 = direction_2
        self.bias_1 = bias_1
        self.bias_2 = bias_2

        # Pre-generate the chirp signal
        self.num_samples = int(duration * sample_rate)
        self.time = np.linspace(0, duration, self.num_samples)
        self.base_signal = self._generate_base_chirp()
        self.signal_1, self.signal_2 = self._transform_signals()

        self.current_index = 0
        
        # Store last values for logging
        self.last_inputs = {self.input_names[0]: 0.0, self.input_names[1]: 0.0}
        
        # Compute effective parameters for logging
        self.effective_amplitude_1 = abs(self.scale_1 * self.direction_1)
        self.effective_amplitude_2 = abs(self.scale_2 * self.direction_2)
        self.effective_offset_1 = self.scale_1 * self.direction_1 * self.bias_1
        self.effective_offset_2 = self.scale_2 * self.direction_2 * self.bias_2

    def _generate_base_chirp(self) -> np.ndarray:
        """Generate base chirp signal s(t) = sin(phase(t)), ranging from -1 to +1"""
        if self.sweep_type == "linear":
            # Linear frequency sweep: f(t) = f_start + (f_end - f_start) * t / T
            phase = (
                2
                * np.pi
                * (self.f_start * self.time + (self.f_end - self.f_start) * self.time**2 / (2 * self.duration))
            )
        elif self.sweep_type == "logarithmic":
            k = (self.f_end / self.f_start) ** (1 / self.duration)
            phase = 2 * np.pi * self.f_start * (k**self.time - 1) / np.log(k)
        elif self.sweep_type == "exponential":
            phase = (
                2
                * np.pi
                * self.f_start
                * self.duration
                / np.log(self.f_end / self.f_start)
                * (np.exp(self.time * np.log(self.f_end / self.f_start) / self.duration) - 1)
            )
        else:
            raise ValueError(f"Unknown sweep type: {self.sweep_type}")

        return np.sin(phase)  # Base signal: -1 to +1

    def _transform_signals(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Transform base signal to per-input signals.
        
        Formula: q_j(t) = scale_j * direction_j * (s(t) + bias_j)
        """
        s = self.base_signal
        
        signal_1 = self.scale_1 * self.direction_1 * (s + self.bias_1)
        signal_2 = self.scale_2 * self.direction_2 * (s + self.bias_2)
        
        return signal_1, signal_2

    def get_next(self) -> tuple[dict, list[float], bool]:
        """
        Get IK inputs and motor angles (positions only).
        
        Returns:
            (inputs_dict, motor_angles, is_complete)
            - inputs_dict: {input_name: value} for all IK inputs
            - motor_angles: list of motor position commands
            - is_complete: True if signal finished
        """
        if self.current_index >= self.num_samples:
            return {}, [], True

        input_1 = float(self.signal_1[self.current_index])
        input_2 = float(self.signal_2[self.current_index])
        
        # Use IK to compute motor angles
        motor_angles = list(self.ik_func(input_1, input_2))
        
        # Store for logging
        self.last_inputs = {self.input_names[0]: input_1, self.input_names[1]: input_2}
        
        self.current_index += 1
        return self.last_inputs, motor_angles, False

    def reset(self) -> None:
        """Reset to beginning of signal"""
        self.current_index = 0

    def get_progress(self) -> float:
        """Get progress as fraction 0-1"""
        return self.current_index / self.num_samples


class SystemIdentification:
    def __init__(
        self,
        config_file: str,
        motor_ids: list[int] | None = None,
        dry_run: bool = False,
    ):
        self.config = self._load_config(config_file)
        self.dry_run = dry_run

        # Determine which motor CAN IDs to use
        # Priority: CLI argument > config file > default [0]
        if motor_ids is not None and len(motor_ids) > 0:
            self.motor_ids = motor_ids
        elif "motor_ids" in self.config:
            self.motor_ids = [int(x) for x in self.config["motor_ids"]]
        else:
            # Fall back to keys from motors config
            self.motor_ids = [int(k) for k in self.config.get("motors", {}).keys()]
            if not self.motor_ids:
                self.motor_ids = [0]  # Default to motor 0

        print(f"Motor CAN IDs to identify: {self.motor_ids}")

        if dry_run:
            print("[DRY-RUN] Running without hardware - using mock controller")
            self.controller = MockMotorCANController(**self.config["can_interface"])
        else:
            self.controller = MotorCANController(**self.config["can_interface"])

        # Storage for motor configurations
        self.motor_configs: dict[int, ConfigurationData] = {}
        self.config_received = threading.Event()
        self.configs_to_receive = set(self.motor_ids)

        # Direct chirp generators for motors not in IK groups
        self.chirp_generators: dict[int, ChirpGenerator] = {}
        
        # IK chirp generators - keyed by group name
        self.ik_generators: dict[str, IKChirpGenerator] = {}
        
        # Mapping: motor_id -> (group_name, motor_index_in_group)
        # e.g., {0: ("left_foot", 0), 1: ("left_foot", 1)}
        self.motor_to_ik_group: dict[int, tuple[str, int]] = {}
        
        # Motors that use direct control (not in any IK group)
        self.direct_motors: set[int] = set()

        # Feedback data collection
        self.feedback_data: dict[int, list[dict]] = defaultdict(list)
        self.feedback_received = threading.Event()
        self.feedbacks_to_receive: set[int] = set()

        # Commanded angle tracking (for comparison with feedback)
        self.commanded_angles: dict[int, float] = {}
        
        # IK input tracking for logging
        self.commanded_ik_inputs: dict[str, dict[str, float]] = {}

        # Timing
        self.start_time = 0.0
        self.sample_count = 0

    def _load_config(self, config_file: str) -> dict:
        """Load configuration from JSON file"""
        with Path(config_file).open() as f:
            val = json.load(f)
            print(val)
            return val

    def _config_callback(self, can_id: int, config: ConfigurationData) -> None:
        """Callback for receiving motor configuration"""
        self.motor_configs[can_id] = config
        print(f"Received configuration from motor {can_id}")
        if can_id in self.configs_to_receive:
            self.configs_to_receive.remove(can_id)
        if not self.configs_to_receive:
            self.config_received.set()

    def _feedback_callback(self, can_id: int, feedback: FeedbackData) -> None:
        """Callback for receiving motor feedback"""
        # Store feedback with timestamp and commanded angle
        feedback_dict = asdict(feedback)
        feedback_dict["timestamp"] = time.time() - self.start_time
        feedback_dict["sample"] = self.sample_count

        # Add commanded angle for comparison
        feedback_dict["commanded_angle"] = self.commanded_angles.get(can_id, 0.0)

        # Calculate tracking error
        feedback_dict["angle_error"] = feedback.angle - self.commanded_angles.get(can_id, 0.0)
        
        # Add IK group information if this motor is in an IK group
        if can_id in self.motor_to_ik_group:
            group_name, _ = self.motor_to_ik_group[can_id]
            feedback_dict["ik_group"] = group_name
            # Add IK inputs for this group
            if group_name in self.commanded_ik_inputs:
                for input_name, value in self.commanded_ik_inputs[group_name].items():
                    feedback_dict[f"commanded_{input_name}"] = value

        self.feedback_data[can_id].append(feedback_dict)

        if can_id in self.feedbacks_to_receive:
            self.feedbacks_to_receive.remove(can_id)
        if not self.feedbacks_to_receive:
            self.feedback_received.set()

    def setup(self) -> None:
        """Initialize CAN controller and read motor configurations"""
        print(f"Starting CAN controller for motors: {self.motor_ids}")
        self.controller.start()
        time.sleep(0.1)

        # Register callbacks for each motor
        for can_id in self.motor_ids:
            self.controller.set_config_callback(can_id, self._config_callback)
            self.controller.set_feedback_callback(can_id, self._feedback_callback)

        # Request configurations from all motors
        print(f"Requesting configurations from motors: {self.motor_ids}")
        for can_id in self.motor_ids:
            self.controller.get_motor_configuration(can_id)

        # Wait for all configurations
        if not self.config_received.wait(timeout=2.0):
            missing = self.configs_to_receive
            print(f"Warning: Did not receive configs from motors: {missing}")

        print(f"Received configurations from {len(self.motor_configs)} motors")

        # Initialize chirp generators
        chirp_config = self.config["chirp"]
        motors_config = self.config.get("motors", {})
        ik_groups_config = self.config.get("ik_groups", [])
        
        # Track which motors are assigned to IK groups
        motors_in_ik_groups: set[int] = set()
        
        # Initialize IK chirp generators for each group
        for group_config in ik_groups_config:
            group_name = group_config["name"]
            ik_type = group_config["ik_type"]
            motor_ids = group_config["motor_ids"]
            
            # Validate motors are in our motor list
            for mid in motor_ids:
                if mid not in self.motor_ids:
                    print(f"Warning: Motor {mid} in IK group '{group_name}' not in motor_ids list")
            
            # Get IK info
            ik_info = IKRegistry.get(ik_type)
            if ik_info is None:
                available = IKRegistry.list_available()
                print(f"Error: Unknown IK type '{ik_type}' for group '{group_name}'. Available: {available}")
                continue
            
            input_names = ik_info["input_names"]
            
            # Get per-input chirp parameters for this group
            # Formula: q_j(t) = scale_j * direction_j * (s(t) + bias_j)
            group_chirp = group_config.get("chirp", {})
            
            # Scale factors (effective amplitude = |scale * direction|)
            scale_1 = group_chirp.get(f"scale_{input_names[0]}", group_chirp.get("scale_1", 1.0))
            scale_2 = group_chirp.get(f"scale_{input_names[1]}", group_chirp.get("scale_2", 1.0))
            
            # Direction multipliers (+1 or -1)
            direction_1 = group_chirp.get(f"direction_{input_names[0]}", group_chirp.get("direction_1", 1.0))
            direction_2 = group_chirp.get(f"direction_{input_names[1]}", group_chirp.get("direction_2", 1.0))
            
            # Bias added to base signal before scaling
            bias_1 = group_chirp.get(f"bias_{input_names[0]}", group_chirp.get("bias_1", 0.0))
            bias_2 = group_chirp.get(f"bias_{input_names[1]}", group_chirp.get("bias_2", 0.0))
            
            # Create IK chirp generator
            self.ik_generators[group_name] = IKChirpGenerator(
                ik_type=ik_type,
                f_start=chirp_config["f_start"],
                f_end=chirp_config["f_end"],
                duration=chirp_config["duration"],
                sample_rate=chirp_config["sample_rate"],
                sweep_type=chirp_config.get("sweep_type", "logarithmic"),
                scale_1=scale_1,
                scale_2=scale_2,
                direction_1=direction_1,
                direction_2=direction_2,
                bias_1=bias_1,
                bias_2=bias_2,
            )
            
            # Map motors to this group
            for idx, mid in enumerate(motor_ids):
                self.motor_to_ik_group[mid] = (group_name, idx)
                motors_in_ik_groups.add(mid)
            
            # Initialize tracking for this group
            self.commanded_ik_inputs[group_name] = {name: 0.0 for name in input_names}
            
            # Get the generator for logging
            ik_gen = self.ik_generators[group_name]
            
            print(f"IK group '{group_name}' initialized:")
            print(f"  IK type: {ik_type} (inputs: {input_names})")
            print(f"  Motors: {motor_ids}")
            print(f"  {input_names[0]}: scale={scale_1:.3f}, dir={direction_1:+.0f}, bias={bias_1:.3f}")
            print(f"    -> effective amplitude={ik_gen.effective_amplitude_1:.4f} rad, offset={ik_gen.effective_offset_1:.4f} rad")
            print(f"  {input_names[1]}: scale={scale_2:.3f}, dir={direction_2:+.0f}, bias={bias_2:.3f}")
            print(f"    -> effective amplitude={ik_gen.effective_amplitude_2:.4f} rad, offset={ik_gen.effective_offset_2:.4f} rad")
        
        # Determine which motors need direct control
        self.direct_motors = set(self.motor_ids) - motors_in_ik_groups
        
        # Get motors defined in the direct motors config
        motors_in_direct_config = {int(k) for k in motors_config.keys() if k.isdigit()}
        
        # VALIDATION 1: Check for motors defined in BOTH IK groups and direct config
        overlap = motors_in_ik_groups & motors_in_direct_config
        if overlap:
            raise ValueError(
                f"Motors {sorted(overlap)} are defined in BOTH ik_groups and motors config. "
                f"Each motor should only be in one place."
            )
        
        # VALIDATION 2: Check that all motor_ids have a config
        motors_without_config = set(self.motor_ids) - motors_in_ik_groups - motors_in_direct_config
        if motors_without_config:
            raise ValueError(
                f"Motors {sorted(motors_without_config)} are in motor_ids but have no config. "
                f"Add them to either 'ik_groups' or 'motors' section in config."
            )
        
        # VALIDATION 3: Check that direct motors have config defined
        direct_motors_without_config = self.direct_motors - motors_in_direct_config
        if direct_motors_without_config:
            raise ValueError(
                f"Direct motors {sorted(direct_motors_without_config)} have no config in 'motors' section. "
                f"Add config for these motors or put them in an IK group."
            )
        
        # Initialize direct chirp generators
        for can_id in self.direct_motors:
            motor_config = motors_config[str(can_id)]
            
            # Per-motor transformation: q(t) = scale * direction * (s(t) + bias)
            scale = motor_config.get("scale", 1.0)
            direction = motor_config.get("direction", 1.0)
            bias = motor_config.get("bias", 0.0)

            self.chirp_generators[can_id] = ChirpGenerator(
                f_start=chirp_config["f_start"],
                f_end=chirp_config["f_end"],
                duration=chirp_config["duration"],
                sample_rate=chirp_config["sample_rate"],
                sweep_type=chirp_config.get("sweep_type", "logarithmic"),
                scale=scale,
                direction=direction,
                bias=bias,
            )

        if self.direct_motors:
            print(f"Direct chirp generators for motors: {sorted(self.direct_motors)}")
            for can_id in sorted(self.direct_motors):
                gen = self.chirp_generators[can_id]
                motor_config = motors_config[str(can_id)]
                scale = motor_config.get("scale", 1.0)
                direction = motor_config.get("direction", 1.0)
                bias = motor_config.get("bias", 0.0)
                print(f"  Motor {can_id}: scale={scale:.3f}, dir={direction:+.0f}, bias={bias:.3f}")
                print(f"    -> effective amplitude={gen.effective_amplitude:.4f} rad, offset={gen.effective_offset:.4f} rad")
        
        if not self.ik_generators and not self.chirp_generators:
            raise ValueError("No chirp generators initialized! Check your config.")

    def run_identification(self) -> None:
        """Main identification loop
        
        Rate limiting strategy:
        - CAN is request-response: must wait for feedback before sending next command
        - We wait for feedback with a reasonable timeout (blocking)
        - If communication takes longer than target period, we proceed immediately
          (no "catching up" - that would violate CAN protocol)
        - Track and warn if requested rate is unachievable
        - If sample_rate is 0 or not set, run as fast as possible (no rate limiting)
        """
        control_params = self.config["control_parameters"]
        chirp_config = self.config["chirp"]
        expected_rate = chirp_config["sample_rate"]
        # rate_limit: True (default) = limit to sample_rate, False = run as fast as possible
        rate_limited = chirp_config.get("rate_limit", True)

        print("\nStarting system identification...")
        if rate_limited:
            print(f"Requested rate: {expected_rate} Hz")
        else:
            print(f"Rate limiting: DISABLED (running as fast as possible)")
            print(f"Signal has {int(expected_rate * chirp_config['duration'])} samples")
        print(f"Duration: {chirp_config['duration']} seconds")
        print(f"Frequency sweep: {chirp_config['f_start']} - {chirp_config['f_end']} Hz")
        print(f"Motor CAN IDs: {self.motor_ids}")
        
        if self.ik_generators:
            print(f"IK groups: {list(self.ik_generators.keys())}")
        if self.direct_motors:
            print(f"Direct motors: {sorted(self.direct_motors)}")

        for can_id in self.motor_ids:
            self.controller.start_motor(can_id)
            time.sleep(0.125)

        print(f"Motors started: {self.motor_ids}")
        time.sleep(1.0)

        self.start_time = time.time()
        self.sample_count = 0
        is_complete = False

        loop_times = []
        rate_warning_issued = False
        missed_deadlines = 0

        # Rate limiting setup
        if rate_limited:
            target_period = 1.0 / expected_rate
            feedback_timeout = target_period * 2  # Allow 2x target period for feedback
        else:
            target_period = 0.0
            feedback_timeout = 0.01  # 10ms default timeout when not rate limiting
        last_send_time = time.time()

        while not is_complete:
            loop_start = time.time()

            # Generate control data for all motors
            control_data = {}
            
            # Process IK groups
            for group_name, ik_gen in self.ik_generators.items():
                ik_inputs, motor_angles, complete = ik_gen.get_next()
                
                if complete:
                    is_complete = True
                
                # Store IK inputs for logging
                self.commanded_ik_inputs[group_name] = ik_inputs
                
                # Get motor IDs for this group
                group_motor_ids = [
                    mid for mid, (gname, _) in self.motor_to_ik_group.items() 
                    if gname == group_name
                ]
                # Sort by index in group
                group_motor_ids = sorted(
                    group_motor_ids, 
                    key=lambda mid: self.motor_to_ik_group[mid][1]
                )
                
                # Create control data for each motor in the group
                for i, motor_id in enumerate(group_motor_ids):
                    if i < len(motor_angles):
                        angle = motor_angles[i]
                        
                        self.commanded_angles[motor_id] = angle
                        
                        control_data[motor_id] = ControlData(
                            angle=angle,
                            velocity=control_params["velocity"],
                            effort=control_params["effort"],
                            stiffness=control_params["stiffness"],
                            damping=control_params["damping"],
                        )
            
            # Process direct motors (not in any IK group)
            # Signal already includes scale/direction/bias transformation
            for can_id in self.direct_motors:
                if can_id not in self.chirp_generators:
                    continue
                    
                angle, complete = self.chirp_generators[can_id].get_next()

                if complete:
                    is_complete = True

                self.commanded_angles[can_id] = angle

                control_data[can_id] = ControlData(
                    angle=angle,
                    velocity=control_params["velocity"],
                    effort=control_params["effort"],
                    stiffness=control_params["stiffness"],
                    damping=control_params["damping"],
                )

            # Send control commands to all motors
            self.feedbacks_to_receive = set(self.motor_ids)
            self.feedback_received.clear()

            for can_id, ctrl_data in control_data.items():
                self.controller.send_kinematics_for_motor(can_id, ctrl_data)

            # BLOCKING wait for all feedback before proceeding
            # This ensures we don't send next command before receiving response
            if not self.feedback_received.wait(timeout=feedback_timeout):
                # Timeout - some motors didn't respond, but we must proceed
                # to avoid blocking forever
                pass

            self.sample_count += 1

            # Calculate loop timing (after feedback received)
            loop_time = time.time() - loop_start
            loop_times.append(loop_time)

            # Rate limiting: maintain minimum period between sends
            # Don't try to "catch up" - CAN requires waiting for response
            if rate_limited:
                elapsed_since_last = time.time() - last_send_time
                sleep_time = target_period - elapsed_since_last
                
                if sleep_time > 0:
                    time.sleep(sleep_time)
                else:
                    # We're running slower than requested rate
                    missed_deadlines += 1
            
            last_send_time = time.time()

            # Check if requested rate is achievable (after first 10 samples)
            if rate_limited and self.sample_count == 10 and not rate_warning_issued:
                avg_loop_time = np.mean(loop_times)
                max_achievable_rate = 1.0 / avg_loop_time if avg_loop_time > 0 else float('inf')
                
                if max_achievable_rate < expected_rate * 0.9:  # 10% margin
                    print(f"\n⚠️  WARNING: Requested rate {expected_rate} Hz is too high!")
                    print(f"    CAN round-trip time: ~{avg_loop_time*1000:.1f} ms")
                    print(f"    Max achievable rate: ~{max_achievable_rate:.1f} Hz")
                    print(f"    Running at maximum achievable rate instead.\n")
                    rate_warning_issued = True

            # Print progress
            if self.sample_count % 100 == 0:
                # Get progress from any generator
                if self.ik_generators:
                    first_gen = next(iter(self.ik_generators.values()))
                    progress = first_gen.get_progress() * 100
                elif self.chirp_generators:
                    first_gen = next(iter(self.chirp_generators.values()))
                    progress = first_gen.get_progress() * 100
                else:
                    progress = 0.0
                    
                avg_loop_time = np.mean(loop_times[-100:])
                actual_rate = 1.0 / avg_loop_time if avg_loop_time > 0 else 0
                
                # Build progress message
                msg = f"Progress: {progress:.1f}% | Sample: {self.sample_count} | Rate: {actual_rate:.1f} Hz"
                
                # Add IK group info
                for group_name, ik_inputs in self.commanded_ik_inputs.items():
                    for name, value in ik_inputs.items():
                        msg += f" | {group_name}.{name}: {np.degrees(value):.2f}°"
                
                print(msg)

        elapsed = time.time() - self.start_time
        actual_rate = self.sample_count / elapsed if elapsed > 0 else 0
        avg_loop_time_ms = np.mean(loop_times) * 1000

        print("\nIdentification complete!")
        print(f"Motors: {self.motor_ids}")
        if self.ik_generators:
            print(f"IK groups: {list(self.ik_generators.keys())}")
        if self.direct_motors:
            print(f"Direct motors: {sorted(self.direct_motors)}")
        print(f"Samples: {self.sample_count}")
        print(f"Duration: {elapsed:.2f} seconds")
        if rate_limited:
            print(f"Requested rate: {expected_rate} Hz")
        else:
            print("Rate limiting: DISABLED")
        print(f"Actual rate: {actual_rate:.1f} Hz")
        print(f"Average loop time: {avg_loop_time_ms:.2f} ms (CAN round-trip)")
        
        if rate_limited and missed_deadlines > 0:
            missed_pct = (missed_deadlines / self.sample_count) * 100
            print(f"⚠️  Missed deadlines: {missed_deadlines}/{self.sample_count} ({missed_pct:.1f}%)")
            print(f"    Requested rate was higher than CAN bus can sustain.")
            print(f"    Consider reducing sample_rate to {actual_rate:.0f} Hz or lower.")

    def save_results(self, output_file: str) -> None:
        """Save collected feedback data as JSON"""
        results = {
            "motor_ids": self.motor_ids,
            "config": self.config,
            "motor_configurations": {can_id: asdict(config) for can_id, config in self.motor_configs.items()},
            "feedback_data": {str(k): v for k, v in self.feedback_data.items()},  # Convert keys to strings for JSON
            "statistics": {
                "total_samples": self.sample_count,
                "duration": time.time() - self.start_time,
                "samples_per_motor": {str(can_id): len(data) for can_id, data in self.feedback_data.items()},
            },
        }
        
        # Add IK group info and compute FK from measured motor positions
        if self.ik_generators:
            results["ik_groups"] = {}
            for group_name, ik_gen in self.ik_generators.items():
                # Get motor IDs for this group (ordered: index 0 = lower, index 1 = upper)
                group_motor_ids = []
                for mid, (gname, idx) in self.motor_to_ik_group.items():
                    if gname == group_name:
                        group_motor_ids.append((idx, mid))
                group_motor_ids.sort()  # Sort by index to ensure order
                motor_ids_ordered = [mid for _, mid in group_motor_ids]
                
                results["ik_groups"][group_name] = {
                    "ik_type": ik_gen.ik_type,
                    "input_names": ik_gen.input_names,
                    "motor_ids": motor_ids_ordered,
                }
                
                # Compute FK for foot IK type
                if ik_gen.ik_type == "foot" and len(motor_ids_ordered) == 2:
                    lower_motor_id = motor_ids_ordered[0]
                    upper_motor_id = motor_ids_ordered[1]
                    
                    # Get feedback data for both motors
                    lower_data = self.feedback_data.get(lower_motor_id, [])
                    upper_data = self.feedback_data.get(upper_motor_id, [])
                    
                    if lower_data and upper_data:
                        print(f"\nComputing FK for IK group '{group_name}'...")
                        fk_data = []
                        
                        # Match samples by sample index
                        lower_by_sample = {d["sample"]: d for d in lower_data}
                        upper_by_sample = {d["sample"]: d for d in upper_data}
                        
                        common_samples = sorted(set(lower_by_sample.keys()) & set(upper_by_sample.keys()))
                        
                        for sample_idx in common_samples:
                            q_lower = lower_by_sample[sample_idx]["angle"]
                            q_upper = upper_by_sample[sample_idx]["angle"]
                            
                            pitch, roll = fk_motor_to_foot(q_lower, q_upper)
                            
                            fk_data.append({
                                "sample": sample_idx,
                                "timestamp": lower_by_sample[sample_idx]["timestamp"],
                                "measured_pitch": pitch,
                                "measured_roll": roll,
                                "q_lower": q_lower,
                                "q_upper": q_upper,
                                # Include commanded values if available
                                "commanded_pitch": lower_by_sample[sample_idx].get("commanded_pitch"),
                                "commanded_roll": lower_by_sample[sample_idx].get("commanded_roll"),
                            })
                        
                        results["ik_groups"][group_name]["fk_data"] = fk_data
                        print(f"  Computed {len(fk_data)} FK samples for '{group_name}'")
        
        if self.direct_motors:
            results["direct_motors"] = sorted(self.direct_motors)

        with Path(output_file).open("w") as f:
            json.dump(results, f, indent=2)

        print(f"\nResults saved to: {output_file}")
        print(f"Motors identified: {self.motor_ids}")
        if self.ik_generators:
            print(f"IK groups: {list(self.ik_generators.keys())}")
        if self.direct_motors:
            print(f"Direct motors: {sorted(self.direct_motors)}")

    def save_torch(self, output_file: str) -> None:
        """Save collected data in PyTorch .pt format (compatible with Isaac sim scripts).
        
        Output format matches Isaac sim chirp script exactly:
            {
                "time": tensor (num_samples,) - timestamps
                "dof_pos": tensor (num_samples, num_joints) - measured positions  
                "des_dof_pos": tensor (num_samples, num_joints) - commanded positions
                "joint_ids": list[int] - motor CAN IDs in column order
            }
        """
        try:
            import torch
        except ImportError:
            print("Warning: PyTorch not installed, skipping .pt save")
            return
        
        # Get all motor data aligned by sample index
        # Find common samples across all motors
        sample_sets = []
        for can_id in self.motor_ids:
            motor_data = self.feedback_data.get(can_id, [])
            if motor_data:
                sample_sets.append({d["sample"] for d in motor_data})
        
        if not sample_sets:
            print("Warning: No feedback data to save")
            return
        
        # Find samples that exist for ALL motors
        common_samples = sorted(set.intersection(*sample_sets))
        num_samples = len(common_samples)
        num_joints = len(self.motor_ids)
        
        if num_samples == 0:
            print("Warning: No common samples across all motors")
            return
        
        # Build lookup dictionaries for each motor
        motor_lookups = {}
        for can_id in self.motor_ids:
            motor_data = self.feedback_data.get(can_id, [])
            motor_lookups[can_id] = {d["sample"]: d for d in motor_data}
        
        # Pre-allocate tensors (matching Isaac sim format exactly)
        time_data = torch.zeros(num_samples)
        dof_pos = torch.zeros(num_samples, num_joints)
        des_dof_pos = torch.zeros(num_samples, num_joints)
        
        # Fill tensors - each motor is a column (DOF)
        for i, sample_idx in enumerate(common_samples):
            # Time from first motor (should be same for all)
            first_motor = self.motor_ids[0]
            time_data[i] = motor_lookups[first_motor][sample_idx]["timestamp"]
            
            # Each motor is a joint/DOF
            for j, can_id in enumerate(self.motor_ids):
                d = motor_lookups[can_id][sample_idx]
                dof_pos[i, j] = d["angle"]
                des_dof_pos[i, j] = d["commanded_angle"]
        
        # Save in Isaac sim compatible format
        data = {
            "time": time_data,
            "dof_pos": dof_pos,
            "des_dof_pos": des_dof_pos,
            "joint_ids": self.motor_ids,  # Motor CAN IDs as joint order
        }
        
        torch.save(data, output_file)
        print(f"Torch data saved to: {output_file}")
        print(f"  Shape: time={tuple(time_data.shape)}, dof_pos={tuple(dof_pos.shape)}, des_dof_pos={tuple(des_dof_pos.shape)}")
        print(f"  Joint IDs (column order): {self.motor_ids}")

    def save_plots(self, output_dir: str) -> None:
        """Save plots for each motor and IK group"""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("Warning: matplotlib not installed, skipping plots")
            return
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        print(f"\nSaving plots to {output_path}...")
        
        # Plot each motor
        for can_id in self.motor_ids:
            motor_data = self.feedback_data.get(can_id, [])
            if not motor_data:
                continue
            
            timestamps = [d["timestamp"] for d in motor_data]
            positions = [d["angle"] for d in motor_data]
            commanded = [d["commanded_angle"] for d in motor_data]
            
            plt.figure(figsize=(12, 6))
            plt.plot(timestamps, positions, label="Measured", linewidth=1)
            plt.plot(timestamps, commanded, label="Commanded", linestyle='dashed', linewidth=1)
            plt.title(f"Motor {can_id} - Position Tracking")
            plt.xlabel("Time [s]")
            plt.ylabel("Position [rad]")
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            
            plot_file = output_path / f"motor_{can_id}_{timestamp}.png"
            plt.savefig(plot_file, dpi=150)
            plt.close()
            print(f"  Saved: {plot_file.name}")
        
        # Plot FK for IK groups
        if self.ik_generators:
            for group_name, ik_gen in self.ik_generators.items():
                group_motor_ids = []
                for mid, (gname, idx) in self.motor_to_ik_group.items():
                    if gname == group_name:
                        group_motor_ids.append((idx, mid))
                group_motor_ids.sort()
                motor_ids_ordered = [mid for _, mid in group_motor_ids]
                
                if ik_gen.ik_type == "foot" and len(motor_ids_ordered) == 2:
                    lower_data = self.feedback_data.get(motor_ids_ordered[0], [])
                    upper_data = self.feedback_data.get(motor_ids_ordered[1], [])
                    
                    if lower_data and upper_data:
                        lower_by_sample = {d["sample"]: d for d in lower_data}
                        upper_by_sample = {d["sample"]: d for d in upper_data}
                        common_samples = sorted(set(lower_by_sample.keys()) & set(upper_by_sample.keys()))
                        
                        timestamps, pitches, rolls = [], [], []
                        cmd_pitches, cmd_rolls = [], []
                        
                        for sample_idx in common_samples:
                            q_lower = lower_by_sample[sample_idx]["angle"]
                            q_upper = upper_by_sample[sample_idx]["angle"]
                            pitch, roll = fk_motor_to_foot(q_lower, q_upper)
                            
                            timestamps.append(lower_by_sample[sample_idx]["timestamp"])
                            pitches.append(np.degrees(pitch))
                            rolls.append(np.degrees(roll))
                            cmd_p = lower_by_sample[sample_idx].get("commanded_pitch")
                            cmd_r = lower_by_sample[sample_idx].get("commanded_roll")
                            cmd_pitches.append(np.degrees(cmd_p) if cmd_p else 0.0)
                            cmd_rolls.append(np.degrees(cmd_r) if cmd_r else 0.0)
                        
                        # Pitch plot
                        plt.figure(figsize=(12, 6))
                        plt.plot(timestamps, pitches, label="Measured", linewidth=1)
                        plt.plot(timestamps, cmd_pitches, label="Commanded", linestyle='dashed', linewidth=1)
                        plt.title(f"{group_name} - Pitch (FK)")
                        plt.xlabel("Time [s]")
                        plt.ylabel("Pitch [deg]")
                        plt.grid(True, alpha=0.3)
                        plt.legend()
                        plt.tight_layout()
                        plot_file = output_path / f"{group_name}_pitch_{timestamp}.png"
                        plt.savefig(plot_file, dpi=150)
                        plt.close()
                        print(f"  Saved: {plot_file.name}")
                        
                        # Roll plot
                        plt.figure(figsize=(12, 6))
                        plt.plot(timestamps, rolls, label="Measured", linewidth=1)
                        plt.plot(timestamps, cmd_rolls, label="Commanded", linestyle='dashed', linewidth=1)
                        plt.title(f"{group_name} - Roll (FK)")
                        plt.xlabel("Time [s]")
                        plt.ylabel("Roll [deg]")
                        plt.grid(True, alpha=0.3)
                        plt.legend()
                        plt.tight_layout()
                        plot_file = output_path / f"{group_name}_roll_{timestamp}.png"
                        plt.savefig(plot_file, dpi=150)
                        plt.close()
                        print(f"  Saved: {plot_file.name}")
        
        print(f"Plots saved to: {output_path}")

    def cleanup(self) -> None:
        """Stop all motors and close CAN controller"""
        print("\nStopping motors...")
        try:
            self.controller.stop_all_motors()
            time.sleep(0.1)
        except Exception as e:
            print(f"Error stopping motors: {e}")

        self.controller.stop()


def parse_motor_ids(value: str) -> list[int]:
    """Parse motor IDs from string (supports comma-separated and ranges)"""
    motor_ids = []
    for part in value.replace(" ", "").split(","):
        if "-" in part:
            # Range like "0-5"
            start, end = part.split("-")
            motor_ids.extend(range(int(start), int(end) + 1))
        else:
            motor_ids.append(int(part))
    return motor_ids


def main() -> None:
    parser = argparse.ArgumentParser(
        description="System identification for robot actuators",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Identify motors with direct chirp (default, no ik_groups in config)
  python system_identification.py --motors 0 2 5

  # Use config with IK groups for foot joints
  python system_identification.py --config config/config.json

  # Dry run with mock hardware
  python system_identification.py --motors 0 1 --dry-run

  # Mix direct and IK-controlled motors (configure in config.json):
  # - Motors 0,1 use "foot" IK (chirp on pitch/roll)
  # - Motors 2,3,4,5 use direct chirp

Config file IK group example:
  "ik_groups": [
    {
      "name": "left_foot",
      "ik_type": "foot",
      "motor_ids": [0, 1],
      "chirp": {
        "mode": "pitch",
        "amplitude_pitch": 0.1,
        "amplitude_roll": 0.1
      }
    }
  ]

Available IK types: foot (pitch/roll -> q_lower/q_upper)
To add new IK types, use IKRegistry.register() in your code.
        """,
    )
    parser.add_argument(
        "--motors",
        "-m",
        type=int,
        nargs="+",
        metavar="CAN_ID",
        help="Motor CAN IDs to identify (e.g., --motors 0 2 5). Overrides config file.",
    )
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default="config/config.json",
        help="Configuration file (JSON)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="data/sysid_results.json",
        help="Output file for results (JSON). Timestamp will be appended to filename.",
    )
    parser.add_argument(
        "--dry-run",
        "-d",
        action="store_true",
        help="Run without hardware (simulated motors for testing)",
    )
    parser.add_argument(
        "--list-motors",
        action="store_true",
        help="List motor IDs from config and exit",
    )
    parser.add_argument(
        "--save",
        "-s",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save results to JSON file (default: True). Use --no-save to disable.",
    )
    parser.add_argument(
        "--save-plots",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Save plots for each motor/joint (default: False). Use --save-plots to enable.",
    )
    parser.add_argument(
        "--save-torch",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Save data in PyTorch .pt format (default: False). Use --save-torch to enable.",
    )
    parser.add_argument(
        "--list-ik",
        action="store_true",
        help="List available IK functions and exit",
    )

    args = parser.parse_args()
    
    if args.list_ik:
        available = IKRegistry.list_available()
        print("Available IK functions:")
        for name in available:
            info = IKRegistry.get(name)
            print(f"  {name}: inputs={info['input_names']}, motors={info['motor_count']}")
        return

    if not Path(args.config).exists():
        print(f"Error: Config file not found: {args.config}")
        sys.exit(1)

    if args.list_motors:
        with Path(args.config).open() as f:
            config = json.load(f)
        motor_ids = config.get("motor_ids", list(config.get("motors", {}).keys()))
        print(f"Motor IDs in config: {motor_ids}")
        return

    sysid = SystemIdentification(args.config, motor_ids=args.motors, dry_run=args.dry_run)

    try:
        sysid.setup()
        sysid.run_identification()
        
        # Generate timestamp for all output files
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = Path(args.output)
        output_dir = output_path.parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save JSON results
        if args.save:
            output_file = output_dir / f"{output_path.stem}_{timestamp}{output_path.suffix}"
            sysid.save_results(str(output_file))
        else:
            print("\nJSON results not saved (--no-save specified)")
        
        # Save PyTorch format
        if args.save_torch:
            torch_file = output_dir / f"{output_path.stem}_{timestamp}.pt"
            sysid.save_torch(str(torch_file))
        
        # Save plots
        if args.save_plots:
            plots_dir = output_dir / f"plots_{timestamp}"
            sysid.save_plots(str(plots_dir))
            
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"\nError: {e}")
        traceback.print_exc()
    finally:
        sysid.cleanup()


if __name__ == "__main__":
    main()
