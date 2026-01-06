"""
Chirp signal generators for system identification.

Provides:
- ChirpGenerator: Direct motor chirp signals
- IKChirpGenerator: IK-space chirp signals (e.g., foot pitch/roll)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# Handle both direct execution and module import
_script_dir = Path(__file__).resolve().parent
if str(_script_dir) not in sys.path:
    sys.path.insert(0, str(_script_dir))

from ik_registry import IKRegistry


class ChirpGenerator:
    """
    Generate chirp (frequency sweep) signals for system identification.

    Transformation: q(t) = scale * direction * s(t) + bias
    Where s(t) = sin(phase(t)) ranges from -1 to +1.
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
        self.f_start = f_start
        self.f_end = f_end
        self.duration = duration
        self.sample_rate = sample_rate
        self.sweep_type = sweep_type
        self.scale = scale
        self.direction = direction
        self.bias = bias

        # Pre-generate signal
        self.num_samples = int(duration * sample_rate)
        self.time = np.linspace(0, duration, self.num_samples)
        self.base_signal = self._generate_base_chirp()
        self.signal = (self.scale * self.direction * self.base_signal +
                       self.bias)
        self.current_index = 0

        # Effective parameters for logging
        self.effective_amplitude = abs(self.scale * self.direction)
        self.effective_offset = self.bias

    def _generate_base_chirp(self) -> np.ndarray:
        """Generate base chirp signal s(t) = sin(phase(t)), ranging -1 to +1."""
        t, T = self.time, self.duration
        f0, f1 = self.f_start, self.f_end

        if self.sweep_type == "linear":
            phase = 2 * np.pi * (f0 * t + (f1 - f0) * t**2 / (2 * T))
        elif self.sweep_type == "logarithmic":
            k = (f1 / f0) ** (1 / T)
            phase = 2 * np.pi * f0 * (k**t - 1) / np.log(k)
        elif self.sweep_type == "exponential":
            phase = 2 * np.pi * f0 * T / np.log(f1 / f0) * (
                np.exp(t * np.log(f1 / f0) / T) - 1
            )
        else:
            raise ValueError(f"Unknown sweep type: {self.sweep_type}")

        return np.sin(phase)

    def get_next(self) -> tuple[float, bool]:
        """Get next sample. Returns (value, is_complete)."""
        if self.current_index >= self.num_samples:
            return 0.0, True
        value = self.signal[self.current_index]
        self.current_index += 1
        return float(value), False

    def reset(self) -> None:
        self.current_index = 0

    def get_progress(self) -> float:
        return self.current_index / self.num_samples


class IKChirpGenerator:
    """
    Generate chirp signals in IK input space (e.g., foot pitch/roll).

    Each IK input gets the same base chirp s(t), transformed with its own parameters:
        q_j(t) = scale_j * direction_j * s(t) + bias_j
    """

    def __init__(
        self,
        ik_type: str,
        f_start: float,
        f_end: float,
        duration: float,
        sample_rate: float,
        sweep_type: str = "linear",
        scale_1: float = 1.0,
        scale_2: float = 1.0,
        direction_1: float = 1.0,
        direction_2: float = 1.0,
        bias_1: float = 0.0,
        bias_2: float = 0.0,
    ):
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

        # Per-input transformation parameters
        self.scale_1, self.scale_2 = scale_1, scale_2
        self.direction_1, self.direction_2 = direction_1, direction_2
        self.bias_1, self.bias_2 = bias_1, bias_2

        # Pre-generate signals
        self.num_samples = int(duration * sample_rate)
        self.time = np.linspace(0, duration, self.num_samples)
        self.base_signal = self._generate_base_chirp()
        self.signal_1 = (self.scale_1 * self.direction_1 * self.base_signal +
                         self.bias_1)
        self.signal_2 = (self.scale_2 * self.direction_2 * self.base_signal +
                         self.bias_2)
        self.current_index = 0

        # For logging
        self.last_inputs = {self.input_names[0]: 0.0, self.input_names[1]: 0.0}
        self.effective_amplitude_1 = abs(self.scale_1 * self.direction_1)
        self.effective_amplitude_2 = abs(self.scale_2 * self.direction_2)
        self.effective_offset_1 = self.bias_1
        self.effective_offset_2 = self.bias_2

    def _generate_base_chirp(self) -> np.ndarray:
        """Generate base chirp signal s(t) = sin(phase(t))."""
        t, T = self.time, self.duration
        f0, f1 = self.f_start, self.f_end

        if self.sweep_type == "linear":
            phase = 2 * np.pi * (f0 * t + (f1 - f0) * t**2 / (2 * T))
        elif self.sweep_type == "logarithmic":
            k = (f1 / f0) ** (1 / T)
            phase = 2 * np.pi * f0 * (k**t - 1) / np.log(k)
        elif self.sweep_type == "exponential":
            phase = 2 * np.pi * f0 * T / np.log(f1 / f0) * (
                np.exp(t * np.log(f1 / f0) / T) - 1
            )
        else:
            raise ValueError(f"Unknown sweep type: {self.sweep_type}")

        return np.sin(phase)

    def get_next(self) -> tuple[dict, list[float], bool]:
        """Get IK inputs and motor angles. Returns (inputs_dict, motor_angles, is_complete)."""
        if self.current_index >= self.num_samples:
            return {}, [], True

        input_1 = float(self.signal_1[self.current_index])
        input_2 = float(self.signal_2[self.current_index])
        motor_angles = list(self.ik_func(input_1, input_2))

        self.last_inputs = {self.input_names[0]: input_1, self.input_names[1]: input_2}
        self.current_index += 1
        return self.last_inputs, motor_angles, False

    def reset(self) -> None:
        self.current_index = 0

    def get_progress(self) -> float:
        return self.current_index / self.num_samples

