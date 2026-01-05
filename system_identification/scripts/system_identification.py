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
import contextlib
import json
import os
import sys
import threading
import time
import traceback
from datetime import datetime
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
import matplotlib.pyplot as plt

import numpy as np


def busy_sleep(duration: float) -> None:
    """Accurate sleep using busy-wait. Best for short durations (<50ms)."""
    end_time = time.perf_counter() + duration
    while time.perf_counter() < end_time:
        pass


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
    # Handle running directly (python scripts/system_identification.py)
    # Add parent directory to path so we can import kinematics
    _parent_dir = Path(__file__).resolve().parent.parent
    if str(_parent_dir) not in sys.path:
        sys.path.insert(0, str(_parent_dir))
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


class ChirpGenerator:
    """
    Generate chirp (frequency sweep) signals for system identification.
    
    Uses the same transformation as IK inputs:
        q(t) = scale * direction * s(t) + bias
    
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
            bias: Bias added after scaling [rad]
            
        The transformation is:
            q(t) = scale * direction * s(t) + bias
            
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
        self.effective_offset = self.bias

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
        """Transform base signal: q(t) = scale * direction * s(t) + bias"""
        return self.scale * self.direction * self.base_signal + self.bias

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
    
        q_j(t) = scale_j * direction_j * s(t) + bias_j
    
    Where s(t) = sin(phase(t)) is the base chirp signal.
    
    This means:
        - Effective offset = bias
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
        # Per-input transformation parameters: q = scale * direction * s(t) + bias
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
            bias_1, bias_2: Bias added after scaling [rad]
            
        The transformation for each input is:
            input_j(t) = scale_j * direction_j * s(t) + bias_j
            
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

        # Per-input transformation: q = scale * direction * s(t) + bias
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
        self.effective_offset_1 = self.bias_1
        self.effective_offset_2 = self.bias_2

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
        
        Formula: q_j(t) = scale_j * direction_j * s(t) + bias_j
        """
        s = self.base_signal
        
        signal_1 = self.scale_1 * self.direction_1 * s + self.bias_1
        signal_2 = self.scale_2 * self.direction_2 * s + self.bias_2
        
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
            # Auto-detect: Collect IDs from both direct 'motors' config AND 'ik_groups'
            found_ids = set()

            # 1. From direct motors config
            motors_cfg = self.config.get("motors", {})
            found_ids.update(int(k) for k in motors_cfg.keys() if k.isdigit())

            # 2. From IK groups
            ik_groups = self.config.get("ik_groups", [])
            for group in ik_groups:
                found_ids.update(group.get("motor_ids", []))

            self.motor_ids = sorted(list(found_ids))

            if not self.motor_ids:
                print("Warning: No motor IDs found in config. Defaulting to [0].")
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
        
        # Current motor positions from feedback (for interpolation)
        self.current_positions: dict[int, float] = {}
        
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
        motor_conf = self.config.get("motors", {}).get(str(can_id))
        self.motor_configs[can_id] = config
        print(f"Received configuration from motor {can_id}")
        if can_id in self.configs_to_receive:
            self.configs_to_receive.remove(can_id)
        if not self.configs_to_receive:
            self.config_received.set()

    def _feedback_callback(self, can_id: int, feedback: FeedbackData) -> None:
        """Callback for receiving motor feedback"""
        # Update current position for interpolation
        self.current_positions[can_id] = feedback.angle
        
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

    def _get_joint_name(self, can_id: int) -> str:
        """Resolve a human-readable name for a CAN ID from the config."""
        # 1. Check if it's in an IK group
        if can_id in self.motor_to_ik_group:
            group_name, idx = self.motor_to_ik_group[can_id]
            
            # Find the group config to check for explicit motor names
            for group_cfg in self.config.get("ik_groups", []):
                if group_cfg["name"] == group_name:
                    if "motor_names" in group_cfg and len(group_cfg["motor_names"]) > idx:
                        return group_cfg["motor_names"][idx]
            
            # Fallback for IK groups
            return f"{group_name}_motor_{idx}"

        # 2. Check direct motor config
        motor_conf = self.config.get("motors", {}).get(str(can_id), {})
        if "name" in motor_conf:
            return motor_conf["name"]

        # 3. Fallback to generic ID
        return f"joint_can_{can_id}"

    def setup(self) -> None:
        """Initialize CAN controller and read motor configurations"""
        print(f"Starting CAN controller for motors: {self.motor_ids}")

        # Initialize Safety Limits Dictionary ---
        # Format: { can_id: (min_pos, max_pos) }
        self.motor_limits: dict[int, tuple[float, float]] = {}

        # 1. Load limits for Direct Motors from "motors" config
        motors_conf = self.config.get("motors", {})
        for mid_str, m_conf in motors_conf.items():
            if "limits" in m_conf:
                # Parse limits: [min, max]
                self.motor_limits[int(mid_str)] = (float(m_conf["limits"][0]), float(m_conf["limits"][1]))

        # 2. Load limits for IK Group Motors
        # Note: If a motor is in both sections, IK group limits take precedence here due to load order
        for group in self.config.get("ik_groups", []):
            if "limits" in group:
                for mid_str, limits in group["limits"].items():
                    self.motor_limits[int(mid_str)] = (float(limits[0]), float(limits[1]))

        print(f"Software Safety Limits loaded: {self.motor_limits}")

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
            # Formula: q_j(t) = scale_j * direction_j * s(t) + bias_j
            group_chirp = group_config.get("chirp", {})
            
            # Scale factors (effective amplitude = |scale * direction|)
            scale_1 = group_chirp.get(f"scale_{input_names[0]}", group_chirp.get("scale_1", 1.0))
            scale_2 = group_chirp.get(f"scale_{input_names[1]}", group_chirp.get("scale_2", 1.0))
            
            # Direction multipliers (+1 or -1)
            direction_1 = group_chirp.get(f"direction_{input_names[0]}", group_chirp.get("direction_1", 1.0))
            direction_2 = group_chirp.get(f"direction_{input_names[1]}", group_chirp.get("direction_2", 1.0))
            
            # Bias added after scaling
            bias_1 = group_chirp.get(f"bias_{input_names[0]}", group_chirp.get("bias_1", 0.0))
            bias_2 = group_chirp.get(f"bias_{input_names[1]}", group_chirp.get("bias_2", 0.0))
            
            # Validate motor count matches IK function output
            motor_count = len(motor_ids)
            expected_motor_count = ik_info["motor_count"]
            if motor_count != expected_motor_count:
                print(f"Error: IK group '{group_name}' has {motor_count} motors but IK type '{ik_type}' expects {expected_motor_count} motors")
                continue
            
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
            # IMPORTANT: The order of motor_ids in the group definition must match
            # the order of angles returned by the IK function (e.g., for foot IK:
            # motor_ids[0] should be the lower motor, motor_ids[1] should be the upper motor)
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
            
            # Per-motor transformation: q(t) = scale * direction * s(t) + bias
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

    def _clamp_angle(self, can_id: int, angle: float) -> float:
        """Helper to clamp angle to configured limits."""
        if can_id in self.motor_limits:
            min_lim, max_lim = self.motor_limits[can_id]
            # strict clamping
            if angle < min_lim:
                return min_lim
            if angle > max_lim:
                return max_lim
        return angle

    def _angular_interpolation(self, start_angles: dict[int, float], target_angles: dict[int, float], alpha: float) -> dict[int, float]:
        """Interpolate between start and target angles using shortest angular path.
        
        Args:
            start_angles: Dict mapping motor_id -> start angle [rad]
            target_angles: Dict mapping motor_id -> target angle [rad]
            alpha: Interpolation factor [0.0, 1.0]
            
        Returns:
            Dict mapping motor_id -> interpolated angle [rad]
        """
        interpolated = {}
        for can_id in target_angles:
            if can_id not in start_angles:
                # If no start angle, use current position or zero
                start_angle = self.current_positions.get(can_id, 0.0)
            else:
                start_angle = start_angles[can_id]
            
            target_angle = target_angles[can_id]
            
            # Calculate the angular difference
            diff = target_angle - start_angle
            
            # Wrap the difference to [-π, π] to find shortest path
            diff = (diff + np.pi) % (2 * np.pi) - np.pi
            
            # Interpolate using the shortest angular path
            interpolated[can_id] = start_angle + diff * alpha
        
        return interpolated

    def _get_current_positions(self, timeout: float = 1.0) -> dict[int, float]:
        """Get current motor positions from feedback.
        
        Args:
            timeout: Maximum time to wait for feedback [s]
            
        Returns:
            Dict mapping motor_id -> current angle [rad]
        """
        # Request feedback from all motors by sending current commands
        # (feedback is typically received after sending a command)
        self.feedbacks_to_receive = set(self.motor_ids)
        self.feedback_received.clear()
        stiffness = 0
        damping = 0
        # Send current commanded positions (or zero if not set) to trigger feedback
        control_data = {}
        for can_id in self.motor_ids:
            current_cmd = self.commanded_angles.get(can_id, 0.0)
            motor_control_params = self._get_control_params(can_id)
            control_data[can_id] = ControlData(
                angle=current_cmd,
                velocity=motor_control_params["velocity"],
                effort=motor_control_params["effort"],
                stiffness=stiffness,
                damping=damping,
            )
        
        # Send commands to trigger feedback
        for can_id, ctrl_data in control_data.items():
            self.controller.send_kinematics_for_motor(can_id, ctrl_data)
        
        # Wait for feedback
        if self.feedback_received.wait(timeout=timeout):
            return self.current_positions.copy()
        else:
            # Timeout - return what we have, or zeros
            return {can_id: self.current_positions.get(can_id, 0.0) for can_id in self.motor_ids}

    def _check_positions_reached(self, target_positions: dict[int, float], tolerance: float = 0.05, max_wait_time: float = 3.0) -> bool:
        """Check if motors have reached target positions within tolerance.
        
        Args:
            target_positions: Dict mapping motor_id -> target angle [rad]
            tolerance: Maximum angular error to consider "reached" [rad] (default: 0.05 rad ≈ 2.9°)
            max_wait_time: Maximum time to wait for motors to reach targets [s]
            
        Returns:
            True if all motors reached targets, False otherwise
        """
        start_time = time.perf_counter()
        check_period = 0.1  # Check every 100ms
        
        while time.perf_counter() - start_time < max_wait_time:
            # Get current positions
            current_positions = self._get_current_positions(timeout=0.5)
            
            # Check all motors
            all_reached = True
            max_error = 0.0
            errors = {}
            
            for can_id, target_angle in target_positions.items():
                if can_id not in current_positions:
                    all_reached = False
                    continue
                
                current_angle = current_positions[can_id]
                
                # Calculate angular error (handle wrapping)
                error = target_angle - current_angle
                error = (error + np.pi) % (2 * np.pi) - np.pi  # Wrap to [-π, π]
                error = abs(error)
                
                errors[can_id] = error
                max_error = max(max_error, error)
                
                if error > tolerance:
                    all_reached = False
            
            if all_reached:
                print(f"  ✓ All motors reached target positions (max error: {max_error*180/np.pi:.2f}°)")
                return True
            
            # Print progress for motors not yet reached
            if time.perf_counter() - start_time >= 1.0:  # Print every second
                unreached = [can_id for can_id, err in errors.items() if err > tolerance]
                if unreached:
                    max_err_deg = max_error * 180 / np.pi
                    print(f"  Waiting for motors {unreached} to reach targets (max error: {max_err_deg:.2f}°, tolerance: {tolerance*180/np.pi:.2f}°)")
            
            time.sleep(check_period)
        
        # Timeout - print final status
        current_positions = self._get_current_positions(timeout=0.5)
        print(f"  ⚠️  Timeout waiting for motors to reach targets (waited {max_wait_time}s)")
        for can_id, target_angle in target_positions.items():
            if can_id in current_positions:
                current_angle = current_positions[can_id]
                error = target_angle - current_angle
                error = (error + np.pi) % (2 * np.pi) - np.pi
                error_deg = abs(error) * 180 / np.pi
                if abs(error) > tolerance:
                    print(f"    Motor {can_id}: target={target_angle*180/np.pi:.2f}°, current={current_angle*180/np.pi:.2f}°, error={error_deg:.2f}°")
        
        return False

    def _get_control_params(self, can_id: int) -> dict:
        """Get control parameters for a motor.
        
        Priority order:
        1. Per-motor control parameters (in motors section)
        2. Per-group control parameters (in ik_groups section)
        3. Global control_parameters
        
        Returns dict with keys: velocity, effort, stiffness, damping
        """
        global_params = self.config["control_parameters"]
        
        # 1. Check per-motor config (for direct motors)
        motors_config = self.config.get("motors", {})
        motor_config = motors_config.get(str(can_id), {})
        if "control_parameters" in motor_config:
            motor_params = motor_config["control_parameters"]
            # Merge with global defaults for any missing keys
            return {
                "velocity": motor_params.get("velocity", global_params["velocity"]),
                "effort": motor_params.get("effort", global_params["effort"]),
                "stiffness": motor_params.get("stiffness", global_params["stiffness"]),
                "damping": motor_params.get("damping", global_params["damping"]),
            }
        
        # 2. Check per-group config (for IK group motors)
        if can_id in self.motor_to_ik_group:
            group_name, _ = self.motor_to_ik_group[can_id]
            for group_cfg in self.config.get("ik_groups", []):
                if group_cfg["name"] == group_name:
                    if "control_parameters" in group_cfg:
                        group_params = group_cfg["control_parameters"]
                        # Merge with global defaults for any missing keys
                        return {
                            "velocity": group_params.get("velocity", global_params["velocity"]),
                            "effort": group_params.get("effort", global_params["effort"]),
                            "stiffness": group_params.get("stiffness", global_params["stiffness"]),
                            "damping": group_params.get("damping", global_params["damping"]),
                        }
                    break
        
        # 3. Fall back to global
        return global_params.copy()

    def _initialize_to_chirp_start(self, interpolation_duration: float = 2.0) -> None:
        """Initialize motors: smoothly interpolate from current position to initial chirp position.
        
        Args:
            interpolation_duration: Time to interpolate from current to initial position [s]
        """
        print("\n=== Initialization Phase ===")
        print("Step 1: Getting current motor positions...")
        
        # Step 1: Get current positions (wherever motors currently are)
        start_positions = self._get_current_positions(timeout=1.0)
        print(f"  Start positions: {start_positions}")
        
        # Step 2: Get initial chirp positions (first sample from generators)
        print("Step 2: Computing initial chirp positions...")
        target_positions = {}
        
        # Get target positions from direct motors
        for can_id in self.direct_motors:
            # Only process motors that are in motor_ids
            if can_id not in self.motor_ids:
                continue
            if can_id in self.chirp_generators:
                # Reset generator and get first sample
                self.chirp_generators[can_id].reset()
                angle, _ = self.chirp_generators[can_id].get_next()
                target_positions[can_id] = angle
        
        # Get target positions from IK groups
        for group_name, ik_gen in self.ik_generators.items():
            # Get motor IDs for this group
            group_motor_ids = [
                mid for mid, (gname, _) in self.motor_to_ik_group.items() 
                if gname == group_name
            ]
            
            # Only process this IK group if at least one motor is in motor_ids
            group_motor_ids_in_scope = [mid for mid in group_motor_ids if mid in self.motor_ids]
            if not group_motor_ids_in_scope:
                continue  # Skip IK groups with no motors in motor_ids
            
            # Sort by index in group
            group_motor_ids_in_scope = sorted(
                group_motor_ids_in_scope, 
                key=lambda mid: self.motor_to_ik_group[mid][1]
            )
            
            ik_gen.reset()
            _, motor_angles, _ = ik_gen.get_next()
            
            # Map motor angles to motors based on their position in the original group
            # Note: IK function returns angles in a fixed order (e.g., q_lower, q_upper for foot IK)
            # The group's motor_ids order must match this IK function's return order
            for motor_id in group_motor_ids_in_scope:
                # Find the index of this motor in the original group order
                original_idx = self.motor_to_ik_group[motor_id][1]
                if original_idx < len(motor_angles):
                    target_positions[motor_id] = motor_angles[original_idx]
                else:
                    print(f"Warning: Motor {motor_id} in group '{group_name}' has index {original_idx} but IK function only returned {len(motor_angles)} angles")
        
        print(f"  Target positions: {target_positions}")
        
        # Step 3: Smoothly interpolate from current positions to initial chirp positions
        print(f"Step 3: Interpolating to initial positions over {interpolation_duration}s...")
        interpolation_start = time.perf_counter()
        interpolation_rate = 50.0  # Hz for smooth interpolation
        interpolation_period = 1.0 / interpolation_rate
        
        while True:
            elapsed = time.perf_counter() - interpolation_start
            alpha = min(1.0, elapsed / interpolation_duration)
            
            # Compute interpolated positions
            interpolated = self._angular_interpolation(start_positions, target_positions, alpha)
            print(f"  Interpolated positions: {interpolated}")
            
            # Send interpolated commands
            control_data = {}
            for can_id in self.motor_ids:
                if can_id in interpolated:
                    angle = interpolated[can_id]
                    self.commanded_angles[can_id] = angle
                    motor_control_params = self._get_control_params(can_id)
                    control_data[can_id] = ControlData(
                        angle=angle,
                        velocity=motor_control_params["velocity"],
                        effort=motor_control_params["effort"],
                        stiffness=motor_control_params["stiffness"],
                        damping=motor_control_params["damping"],
                    )
            
            # Send commands
            self.feedbacks_to_receive = set(self.motor_ids)
            self.feedback_received.clear()
            for can_id, ctrl_data in control_data.items():
                self.controller.send_kinematics_for_motor(can_id, ctrl_data)
            
            # Wait for feedback (with timeout)
            self.feedback_received.wait(timeout=0.1)
            
            # Check if interpolation complete
            if alpha >= 1.0:
                break
            
            # Rate limit interpolation
            time.sleep(interpolation_period)
        
        # # Step 4: Verify motors reached initial positions
        # print("Step 4: Verifying motors reached initial positions...")
        # position_tolerance = self.config.get("position_tolerance", 0.05)  # Default: 0.05 rad ≈ 2.9°
        # reached = self._check_positions_reached(target_positions, tolerance=position_tolerance, max_wait_time=3.0)
        
        # if reached:
        #     print("  ✓ All motors reached initial chirp positions. Starting data collection...\n")
        # else:
        #     print("  ⚠️  Warning: Some motors did not reach initial positions, but proceeding with data collection...\n")
        

        # Reset generators to start from beginning (we already consumed first sample)
        for gen in self.chirp_generators.values():
            gen.reset()
        for gen in self.ik_generators.values():
            gen.reset()

    def _finalize_to_zero(self, interpolation_duration: float = 2.0) -> None:
        """Finalize: smoothly interpolate from current position back to zero.
        
        Args:
            interpolation_duration: Time to interpolate from current to zero [s]
        """
        print("\n=== Finalization Phase ===")
        print("Interpolating back to zero position...")

        start_positions = {can_id: self.commanded_angles[can_id] for can_id in self.motor_ids}
        print(f"  Start positions: {start_positions}")
               
        # Target is zero for all motors
        target_positions = {can_id: 0.0 for can_id in self.motor_ids}
        print(f"  Target positions: {target_positions}")
        
        # Interpolate to zero
        interpolation_start = time.perf_counter()
        interpolation_rate = 50.0  # Hz for smooth interpolation
        interpolation_period = 1.0 / interpolation_rate
        
        while True:
            elapsed = time.perf_counter() - interpolation_start
            alpha = min(1.0, elapsed / interpolation_duration)
            
            # Compute interpolated positions
            interpolated = self._angular_interpolation(start_positions, target_positions, alpha)
            
            # Send interpolated commands
            control_data = {}
            for can_id in self.motor_ids:
                if can_id in interpolated:
                    angle = interpolated[can_id]
                    self.commanded_angles[can_id] = angle
                    safe_angle = self._clamp_angle(can_id, angle)
                    motor_control_params = self._get_control_params(can_id)
                    control_data[can_id] = ControlData(
                        angle=safe_angle,
                        velocity=motor_control_params["velocity"],
                        effort=motor_control_params["effort"],
                        stiffness=motor_control_params["stiffness"],
                        damping=motor_control_params["damping"],
                    )
            
            # Send commands
            self.feedbacks_to_receive = set(self.motor_ids)
            self.feedback_received.clear()
            for can_id, ctrl_data in control_data.items():
                self.controller.send_kinematics_for_motor(can_id, ctrl_data)
            
            # Wait for feedback (with timeout)
            self.feedback_received.wait(timeout=0.1)
            
            # Check if interpolation complete
            if alpha >= 1.0:
                break
            
            # Rate limit interpolation
            time.sleep(interpolation_period)
        
        # # Verify motors reached zero position
        # print("Verifying motors reached zero position...")
        # position_tolerance = self.config.get("position_tolerance", 0.05)  # Default: 0.05 rad ≈ 2.9°
        # zero_targets = {can_id: 0.0 for can_id in self.motor_ids}
        # reached = self._check_positions_reached(zero_targets, tolerance=position_tolerance, max_wait_time=3.0)
        
        # if reached:
        #     print("  ✓ All motors reached zero position. Finalization complete!\n")
        # else:
        #     print("  ⚠️  Warning: Some motors did not reach zero position.\n")

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

        # Get interpolation duration from config (default 2.0 seconds)
        interpolation_duration = self.config.get("interpolation_duration", 2.0)
        
        # Initialize: send zero commands and interpolate to initial chirp position
        self._initialize_to_chirp_start(interpolation_duration=interpolation_duration)

        self.start_time = time.perf_counter()
        self.sample_count = 0
        is_complete = False

        loop_times = []
        rate_warning_issued = False
        missed_deadlines = 0

        # Rate limiting setup
        # Use perf_counter for accurate timing (nanosecond resolution)
        if rate_limited:
            target_period = 1.0 / expected_rate
            feedback_timeout = target_period * 2  # Allow 2x target period for feedback
        else:
            target_period = 0.0
            feedback_timeout = 0.01  # 10ms default timeout when not rate limiting
        
        # Use busy-wait for accurate timing in dry-run mode (no I/O overhead)
        use_busy_wait = self.dry_run and rate_limited
        last_send_time = None  # Will be set after first iteration
        first_sample_done = False


        while not is_complete:
            control_data = {}
            loop_start = time.perf_counter()

           
            # Process IK groups
            for group_name, ik_gen in self.ik_generators.items():
                # Get motor IDs for this group
                group_motor_ids = [
                    mid for mid, (gname, _) in self.motor_to_ik_group.items() 
                    if gname == group_name
                ]
                
                # Only process this IK group if at least one motor is in motor_ids
                group_motor_ids_in_scope = [mid for mid in group_motor_ids if mid in self.motor_ids]
                if not group_motor_ids_in_scope:
                    continue  # Skip IK groups with no motors in motor_ids
                
                # Sort by index in group
                group_motor_ids_in_scope = sorted(
                    group_motor_ids_in_scope, 
                    key=lambda mid: self.motor_to_ik_group[mid][1]
                )
                
                ik_inputs, motor_angles, complete = ik_gen.get_next()
                
                if complete:
                    is_complete = True
                    continue
                
                # Store IK inputs for logging
                self.commanded_ik_inputs[group_name] = ik_inputs
                
                # Create control data for each motor in the group
                for motor_id in group_motor_ids_in_scope:
                    # Find the index of this motor in the original group order
                    original_idx = self.motor_to_ik_group[motor_id][1]
                    if original_idx < len(motor_angles):
                        # Normal case: use computed motor angle from IK
                        raw_angle = motor_angles[original_idx]
                        self.commanded_angles[motor_id] = raw_angle
                    elif complete:
                        # Signal complete: use last commanded angle (or 0.0 if never commanded)
                        raw_angle = self.commanded_angles.get(motor_id, 0.0)
                    else:
                        # Skip if no data and not complete
                        continue

                    safe_angle = self._clamp_angle(motor_id, raw_angle)
                    motor_control_params = self._get_control_params(motor_id)
                    control_data[motor_id] = ControlData(
                        angle=safe_angle,
                        velocity=motor_control_params["velocity"],
                        effort=motor_control_params["effort"],
                        stiffness=motor_control_params["stiffness"],
                        damping=motor_control_params["damping"],
                    )
            
            # Process direct motors (not in any IK group)
            # Signal already includes scale/direction/bias transformation
            for can_id in self.direct_motors:
                if can_id not in self.chirp_generators:
                    continue
                    
                raw_angle, complete = self.chirp_generators[can_id].get_next()

                if complete:
                    is_complete = True
                    continue

                self.commanded_angles[can_id] = raw_angle
                safe_angle = self._clamp_angle(can_id, raw_angle)

                motor_control_params = self._get_control_params(can_id)
                control_data[can_id] = ControlData(
                    angle=safe_angle,
                    velocity=motor_control_params["velocity"],
                    effort=motor_control_params["effort"],
                    stiffness=motor_control_params["stiffness"],
                    damping=motor_control_params["damping"],
                )

            # Ensure all motors get a command (use last commanded angle or zero for any missing)
            # This ensures consistent behavior on the final iteration
            # for can_id in self.motor_ids:
            #     if can_id not in control_data:
            #         # Motor didn't get a command (shouldn't happen, but handle gracefully)
            #         last_angle = self.commanded_angles.get(can_id, 0.0)
            #         safe_angle = self._clamp_angle(can_id, last_angle)
            #         motor_control_params = self._get_control_params(can_id)
            #         control_data[can_id] = ControlData(
            #             angle=safe_angle,
            #             velocity=motor_control_params["velocity"],
            #             effort=motor_control_params["effort"],
            #             stiffness=motor_control_params["stiffness"],
            #             damping=motor_control_params["damping"],
            #         )

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

            # Only count valid samples (not the completion iteration)
            if not is_complete:
                self.sample_count += 1

            # Calculate loop timing (after feedback received)
            loop_time = time.perf_counter() - loop_start
            loop_times.append(loop_time)

            # Rate limiting: maintain minimum period between sends
            # Don't try to "catch up" - CAN requires waiting for response
            if rate_limited and last_send_time is not None:
                elapsed_since_last = time.perf_counter() - last_send_time
                sleep_time = target_period - elapsed_since_last
                
                if sleep_time > 0:
                    if use_busy_wait:
                        # Busy-wait for accurate timing (dry-run mode)
                        busy_sleep(sleep_time)
                    else:
                        # For real hardware: time.sleep() is inaccurate for short durations
                        # (often oversleeps by 1-2ms on Linux). Use hybrid approach:
                        # - Regular sleep for longer durations (saves CPU)
                        # - Busy-wait for short durations (accurate timing)
                        if sleep_time > 0.002:  # > 2ms: use regular sleep
                            # Sleep for most of the time, leave 1ms for busy-wait
                            time.sleep(sleep_time - 0.001)
                            # Finish with busy-wait for accuracy
                            remaining = target_period - (time.perf_counter() - last_send_time)
                            if remaining > 0:
                                busy_sleep(remaining)
                        else:
                            # Short duration: busy-wait for accuracy
                            busy_sleep(sleep_time)
                elif not is_complete:
                    # We're running slower than requested rate
                    # (don't count the final completion iteration)
                    missed_deadlines += 1
            
            last_send_time = time.perf_counter()
            
            # Start timing from first sample completion (not before loop)
            if not first_sample_done:
                self.start_time = time.perf_counter()
                first_sample_done = True

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
                    
                # Calculate actual achieved rate from elapsed time
                # Use (samples-1)/elapsed for interval-based rate (N samples = N-1 intervals)
                elapsed_so_far = time.perf_counter() - self.start_time
                intervals = self.sample_count - 1
                current_rate = intervals / elapsed_so_far if elapsed_so_far > 0 and intervals > 0 else expected_rate
                
                # Build progress message
                if rate_limited:
                    msg = f"Progress: {progress:.1f}% | Sample: {self.sample_count} | Rate: {current_rate:.1f}/{expected_rate:.0f} Hz"
                else:
                    msg = f"Progress: {progress:.1f}% | Sample: {self.sample_count} | Rate: {current_rate:.1f} Hz"
                
                # Temporarily disable suppression to print our progress
                print(msg)

        elapsed = time.perf_counter() - self.start_time
        # Use interval-based rate: (samples-1)/elapsed for N samples = N-1 intervals
        intervals = self.sample_count - 1
        actual_rate = intervals / elapsed if elapsed > 0 and intervals > 0 else 0
        avg_loop_time_ms = np.mean(loop_times) * 1000

        print("\nIdentification complete!")
        
        # Finalize: smoothly interpolate back to zero
        interpolation_duration = self.config.get("interpolation_duration", 2.0)
        self._finalize_to_zero(interpolation_duration=interpolation_duration)
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
        loop_label = "mock loop" if self.dry_run else "CAN round-trip"
        print(f"Average loop time: {avg_loop_time_ms:.2f} ms ({loop_label})")
        
        if rate_limited and missed_deadlines > 0:
            missed_pct = (missed_deadlines / self.sample_count) * 100
            print(f"⚠️  Missed deadlines: {missed_deadlines}/{self.sample_count} ({missed_pct:.1f}%)")

    def save_plots(self, output_dir: str) -> None:
        """Save plots for each motor and IK group with dual axes (Rad/Deg)"""
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
            plt.plot(
                timestamps,
                commanded,
                label="Commanded",
                linestyle="dashed",
                linewidth=1,
            )
            plt.title(f"Motor {can_id} - Position Tracking")
            plt.xlabel("Time [s]")
            plt.ylabel("Position [rad]")
            
            ax = plt.gca()
            # Create a secondary axis that converts radians <-> degrees
            secax = ax.secondary_yaxis('right', functions=(np.degrees, np.radians))
            secax.set_ylabel('Position [deg]')

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
                    if gname == group_name and mid in self.motor_ids:
                        group_motor_ids.append((idx, mid))
                group_motor_ids.sort()
                motor_ids_ordered = [mid for _, mid in group_motor_ids]
                
                # Skip if no motors in this group are in motor_ids
                if not motor_ids_ordered:
                    continue

                if ik_gen.ik_type == "foot" and len(motor_ids_ordered) == 2:
                    lower_data = self.feedback_data.get(motor_ids_ordered[0], [])
                    upper_data = self.feedback_data.get(motor_ids_ordered[1], [])

                    if lower_data and upper_data:
                        lower_by_sample = {d["sample"]: d for d in lower_data}
                        upper_by_sample = {d["sample"]: d for d in upper_data}
                        common_samples = sorted(
                            set(lower_by_sample.keys()) & set(upper_by_sample.keys())
                        )

                        timestamps, pitches, rolls = [], [], []
                        cmd_pitches, cmd_rolls = [], []

                        for sample_idx in common_samples:
                            q_lower = lower_by_sample[sample_idx]["angle"]
                            q_upper = upper_by_sample[sample_idx]["angle"]
                            pitch, roll = fk_motor_to_foot(q_lower, q_upper)

                            timestamps.append(lower_by_sample[sample_idx]["timestamp"])
                            
                            # --- MODIFICATION: Keep data in Radians for left axis ---
                            pitches.append(pitch) 
                            rolls.append(roll)
                            
                            cmd_p = lower_by_sample[sample_idx].get("commanded_pitch")
                            cmd_r = lower_by_sample[sample_idx].get("commanded_roll")
                            # Keep commanded in Radians
                            cmd_pitches.append(cmd_p if cmd_p is not None else 0.0)
                            cmd_rolls.append(cmd_r if cmd_r is not None else 0.0)

                        # Pitch plot
                        plt.figure(figsize=(12, 6))
                        plt.plot(timestamps, pitches, label="Measured", linewidth=1)
                        plt.plot(
                            timestamps,
                            cmd_pitches,
                            label="Commanded",
                            linestyle="dashed",
                            linewidth=1,
                        )
                        plt.title(f"{group_name} - Pitch (FK)")
                        plt.xlabel("Time [s]")
                        plt.ylabel("Pitch [rad]") # Left axis is now Radians
                        
                        # --- MODIFICATION START ---
                        ax = plt.gca()
                        secax = ax.secondary_yaxis('right', functions=(np.degrees, np.radians))
                        secax.set_ylabel('Pitch [deg]')
                        # --- MODIFICATION END ---
                        
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
                        plt.plot(
                            timestamps,
                            cmd_rolls,
                            label="Commanded",
                            linestyle="dashed",
                            linewidth=1,
                        )
                        plt.title(f"{group_name} - Roll (FK)")
                        plt.xlabel("Time [s]")
                        plt.ylabel("Roll [rad]") # Left axis is now Radians
                        
                        # --- MODIFICATION START ---
                        ax = plt.gca()
                        secax = ax.secondary_yaxis('right', functions=(np.degrees, np.radians))
                        secax.set_ylabel('Roll [deg]')
                        # --- MODIFICATION END ---

                        plt.grid(True, alpha=0.3)
                        plt.legend()
                        plt.tight_layout()
                        plot_file = output_path / f"{group_name}_roll_{timestamp}.png"
                        plt.savefig(plot_file, dpi=150)
                        plt.close()
                        print(f"  Saved: {plot_file.name}")

        print(f"Plots saved to: {output_path}")

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

        with Path(output_file).open("w") as f:
            json.dump(results, f, indent=2)

        print(f"\nResults saved to: {output_file}")


    def save_hoku_style_data(self, output_dir: Path) -> None:
        """Save data in the exact format required by the Hoku Sim2Real tools."""
        try:
            import torch
        except ImportError:
            print("Warning: PyTorch not installed, skipping .pt save")
            return

        # 1. Align Data by Time
        sample_sets = []
        for can_id in self.motor_ids:
            motor_data = self.feedback_data.get(can_id, [])
            if motor_data:
                sample_sets.append({d["sample"] for d in motor_data})
        
        if not sample_sets:
            print("[WARN] No data found to save.")
            return
        
        common_samples = sorted(set.intersection(*sample_sets))
        num_samples = len(common_samples)
        num_joints = len(self.motor_ids)
        
        # 2. Prepare Tensors
        time_data = torch.zeros(num_samples)
        dof_pos = torch.zeros(num_samples, num_joints)
        des_dof_pos = torch.zeros(num_samples, num_joints)
        
        # Resolve Joint Names 
        # We will modify these names if FK is applied
        joint_names = [self._get_joint_name(mid) for mid in self.motor_ids]

        motor_lookups = {
            can_id: {d["sample"]: d for d in self.feedback_data.get(can_id, [])}
            for can_id in self.motor_ids
        }

        # 3. Fill Tensors with RAW data
        for i, sample_idx in enumerate(common_samples):
            time_data[i] = motor_lookups[self.motor_ids[0]][sample_idx]["timestamp"]
            for j, can_id in enumerate(self.motor_ids):
                d = motor_lookups[can_id][sample_idx]
                dof_pos[i, j] = d["angle"]
                des_dof_pos[i, j] = d["commanded_angle"]

        # 4. Perform FK and Overwrite
        if self.ik_generators:
            for group_name, ik_gen in self.ik_generators.items():
                if ik_gen.ik_type != "foot": continue

                # Get sorted motors (0: Lower/Pitch, 1: Upper/Roll)
                # Only include motors that are in motor_ids
                group_motor_ids = [mid for mid, (gname, _) in self.motor_to_ik_group.items() 
                                 if gname == group_name and mid in self.motor_ids]
                group_motor_ids.sort(key=lambda m: self.motor_to_ik_group[m][1])

                if len(group_motor_ids) != 2: continue
                
                id_pitch, id_roll = group_motor_ids[0], group_motor_ids[1]
                # These should be in motor_ids, but add safety check
                if id_pitch not in self.motor_ids or id_roll not in self.motor_ids:
                    continue
                idx_pitch = self.motor_ids.index(id_pitch)
                idx_roll = self.motor_ids.index(id_roll)

                # Rename joints in the list to reflect data content
                # e.g., "left_ankle_motor_a" -> "left_foot_pitch"
                joint_names[idx_pitch] = f"{group_name}_pitch"
                joint_names[idx_roll] = f"{group_name}_roll"

                # Apply FK
                q_l_meas, q_u_meas = dof_pos[:, idx_pitch].tolist(), dof_pos[:, idx_roll].tolist()
                q_l_cmd, q_u_cmd = des_dof_pos[:, idx_pitch].tolist(), des_dof_pos[:, idx_roll].tolist()
                
                # Compute Meas FK
                meas_p, meas_r = [], []
                for ql, qu in zip(q_l_meas, q_u_meas):
                    p, r = fk_motor_to_foot(ql, qu)
                    meas_p.append(p)
                    meas_r.append(r)

                # Compute Cmd FK
                cmd_p, cmd_r = [], []
                for ql, qu in zip(q_l_cmd, q_u_cmd):
                    p, r = fk_motor_to_foot(ql, qu)
                    cmd_p.append(p)
                    cmd_r.append(r)

                # Overwrite tensors
                dof_pos[:, idx_pitch] = torch.tensor(meas_p)
                des_dof_pos[:, idx_pitch] = torch.tensor(cmd_p)
                
                dof_pos[:, idx_roll] = torch.tensor(meas_r)
                des_dof_pos[:, idx_roll] = torch.tensor(cmd_r)

        # 5. Save
        save_path = output_dir / "chirp_data.pt"
        torch.save(
            {
                "time": time_data.cpu(),
                "dof_pos": dof_pos.cpu(),
                "des_dof_pos": des_dof_pos.cpu(),
                "joint_names": joint_names, 
            },
            save_path,
        )
        print(f"[INFO] Saved Sim2Real formatted data to {save_path}")
        

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


def resolve_config_path(config_arg: str) -> Path:
    """Resolve config file path, looking in package config/ folder if not found.
    
    Search order:
    1. Exact path as given (absolute or relative to cwd)
    2. In config/ folder relative to package root (system_identification/config/)
    3. In config/ folder relative to script location
    """
    config_path = Path(config_arg)
    
    # 1. Check exact path
    if config_path.exists():
        return config_path
    
    # 2. Check in package config/ folder (system_identification/config/)
    package_root = Path(__file__).resolve().parent.parent
    package_config = package_root / "config" / config_path.name
    if package_config.exists():
        return package_config
    
    # 3. If config_arg doesn't include "config/", try prepending it
    if not config_arg.startswith("config/") and not config_arg.startswith("config\\"):
        package_config_full = package_root / "config" / config_arg
        if package_config_full.exists():
            return package_config_full
    
    # Return original path (will fail with file not found)
    return config_path


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
        default="data",
        help="Output directory for results. A timestamped subfolder will be created.",
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
        default=True,
        help="Save plots for each motor/joint (default: True). Use --no-save-plots to disable.",
    )
    parser.add_argument(
        "--save-torch",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save data in PyTorch .pt format (default: True). Use --no-save-torch to disable.",
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

    # Resolve config path (looks in package config/ folder if not found)
    config_path = resolve_config_path(args.config)
    if not config_path.exists():
        print(f"Error: Config file not found: {args.config}")
        print(f"  Searched: {args.config}")
        print(f"  Also tried: {Path(__file__).resolve().parent.parent / 'config' / Path(args.config).name}")
        sys.exit(1)

    if args.list_motors:
        with config_path.open() as f:
            config = json.load(f)
        motor_ids = config.get("motor_ids", list(config.get("motors", {}).keys()))
        print(f"Motor IDs in config: {motor_ids}")
        return

    sysid = SystemIdentification(str(config_path), motor_ids=args.motors, dry_run=args.dry_run)

    try:
        sysid.setup()
        sysid.run_identification()
        
        # Generate timestamp for output folder
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Resolve output base path (relative to package if not absolute)
        output_base = Path(args.output)
        if not output_base.is_absolute():
            # Make relative paths relative to package root
            package_root = Path(__file__).resolve().parent.parent
            output_base = package_root / output_base
        
        # Create timestamped output folder
        output_dir = output_base / f"sysid_{timestamp}"
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"\nOutput folder: {output_dir}")
         
        if args.save:
            # 1. Save standard JSON (Good for debugging)
            sysid.save_results(str(output_dir / "raw_debug_data.json"))
        
            # # 2. Save Hoku-Style Data (.pt) - The critical file for Sim2Real
            # if args.save_torch:
            #     sysid.save_hoku_style_data(output_dir)

            # 3. Save Plots
            if args.save_plots:
                sysid.save_plots(output_dir)
                                   
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"\nError: {e}")
        traceback.print_exc()
    finally:
        sysid.cleanup()


if __name__ == "__main__":
    main()
