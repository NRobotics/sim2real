"""
Motor controller implementations for system identification.

Provides:
- MockMotorCANController: Simulated controller for dry-run testing
"""

from __future__ import annotations

import numpy as np

from humanoid_messages.can import ConfigurationData, ControlData, FeedbackData


class MockMotorCANController:
    """Mock controller for dry-run mode (no hardware required)."""

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
        """Simulate receiving motor configuration."""
        if can_id in self._config_callbacks:
            mock_config = ConfigurationData(
                can_id=can_id,
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
        """Simulate receiving feedback after sending control."""
        if can_id in self._feedback_callbacks:
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

