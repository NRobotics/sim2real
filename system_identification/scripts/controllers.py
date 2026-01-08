"""
Motor controller implementations for system identification.

Provides:
- MockMotorCANController: Simulated controller for dry-run testing (async-compatible)
"""

from __future__ import annotations

import os
import random
import sys
import threading
import time
from typing import TYPE_CHECKING

import numpy as np

from humanoid_messages.can import ConfigurationData, ControlData, FeedbackData

if TYPE_CHECKING:
    from collections.abc import Callable


class MockMotorCANController:
    """
    Mock controller for dry-run mode (no hardware required).

    Simulates asynchronous feedback with configurable latency.
    Feedback is delivered via callbacks from a background thread,
    mimicking real CAN bus behavior.
    """

    def __init__(
        self,
        latency_mean: float = 0.002,
        latency_std: float = 0.0005,
        drop_rate: float = 0.0,
        **kwargs,
    ):
        """
        Initialize mock controller.

        Args:
            latency_mean: Mean feedback latency in seconds (default 2ms)
            latency_std: Latency standard deviation (default 0.5ms)
            drop_rate: Probability of dropping feedback (0.0-1.0)
            **kwargs: Additional args (ignored, for API compatibility)
        """
        self._config_callbacks: dict[int, Callable] = {}
        self._feedback_callbacks: dict[int, Callable] = {}
        self._latency_mean = latency_mean
        self._latency_std = latency_std
        self._drop_rate = drop_rate

        # Background thread for delivering feedback
        self._running = False
        self._feedback_queue: list[tuple[float, int, FeedbackData]] = []
        self._queue_lock = threading.Lock()
        self._feedback_thread: threading.Thread | None = None

        # Thread configuration
        self._main_cpu = self._get_main_cpu()

        print(f"[DRY-RUN] Mock CAN controller initialized")
        print(f"[DRY-RUN]   Latency: {latency_mean*1000:.1f}ms ± {latency_std*1000:.1f}ms")
        print(f"[DRY-RUN]   Drop rate: {drop_rate*100:.1f}%")

    def start(self) -> None:
        """Start the mock controller and feedback delivery thread."""
        self._running = True
        self._feedback_thread = threading.Thread(
            target=self._feedback_delivery_loop,
            daemon=True,
            name="MockFeedbackDelivery",
        )
        self._feedback_thread.start()
        print("[DRY-RUN] Mock CAN controller started")

    def stop(self) -> None:
        """Stop the mock controller."""
        self._running = False
        if self._feedback_thread and self._feedback_thread.is_alive():
            self._feedback_thread.join(timeout=1.0)
        print("[DRY-RUN] Mock CAN controller stopped")

    def _feedback_delivery_loop(self) -> None:
        """Background thread that delivers feedback at scheduled times."""
        # Configure thread to run on different CPU than main if possible
        self._configure_thread()

        while self._running:
            now = time.perf_counter()
            to_deliver: list[tuple[int, FeedbackData]] = []

            with self._queue_lock:
                # Find feedback ready to deliver
                ready_indices = []
                for i, (deliver_time, can_id, feedback) in enumerate(self._feedback_queue):
                    if now >= deliver_time:
                        to_deliver.append((can_id, feedback))
                        ready_indices.append(i)

                # Remove delivered items (reverse order to preserve indices)
                for i in reversed(ready_indices):
                    self._feedback_queue.pop(i)

            # Deliver feedback outside of lock
            for can_id, feedback in to_deliver:
                if can_id in self._feedback_callbacks:
                    try:
                        self._feedback_callbacks[can_id](can_id, feedback)
                    except Exception as e:
                        print(f"[DRY-RUN] Callback error for motor {can_id}: {e}")

            # Small sleep to avoid busy-waiting
            time.sleep(0.0001)

    def _get_main_cpu(self) -> int | None:
        """Get the CPU main thread is pinned to, if any."""
        if not sys.platform.startswith("linux"):
            return None
        try:
            affinity = os.sched_getaffinity(0)
            if len(affinity) == 1:
                return next(iter(affinity))
        except (OSError, AttributeError):
            pass
        return None

    def _configure_thread(self) -> None:
        """Configure feedback thread for optimal performance."""
        if not sys.platform.startswith("linux"):
            return

        try:
            all_cpus = os.sched_getaffinity(0)
            if self._main_cpu is not None and len(all_cpus) > 1:
                other_cpus = all_cpus - {self._main_cpu}
                if other_cpus:
                    os.sched_setaffinity(0, other_cpus)
        except (OSError, PermissionError):
            pass

    def set_config_callback(self, can_id: int, callback: Callable) -> None:
        """Register configuration callback."""
        self._config_callbacks[can_id] = callback

    def set_feedback_callback(self, can_id: int, callback: Callable) -> None:
        """Register feedback callback."""
        self._feedback_callbacks[can_id] = callback

    def get_motor_configuration(self, can_id: int) -> None:
        """Simulate receiving motor configuration (immediate)."""
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

    def start_motor(self, can_id: int) -> None:
        """Simulate starting a motor."""
        print(f"[DRY-RUN] Motor {can_id} started")

    def stop_all_motors(self) -> None:
        """Simulate stopping all motors."""
        print("[DRY-RUN] All motors stopped")

    def stop_motor(self, can_id: int) -> None:
        """Simulate stopping a motor."""
        print(f"[DRY-RUN] Motor {can_id} stopped")

    def ping_motor(self, can_id: int) -> None:
        """
        Simulate pinging a motor to request feedback.
        
        Schedules feedback delivery after latency, same as send_kinematics.
        """
        if random.random() < self._drop_rate:
            return

        # Generate mock feedback at current position (assume 0 if not tracked)
        mock_feedback = FeedbackData(
            angle=np.random.normal(0, 0.01),
            velocity=np.random.normal(0, 0.1),
            effort=np.random.normal(0, 0.05),
            voltage=24.0 + np.random.normal(0, 0.1),
            temp_motor=35.0 + np.random.normal(0, 1.0),
            temp_pcb=30.0 + np.random.normal(0, 0.5),
            flags=0,
        )

        latency = max(0.0001, np.random.normal(self._latency_mean, self._latency_std))
        deliver_time = time.perf_counter() + latency

        with self._queue_lock:
            self._feedback_queue.append((deliver_time, can_id, mock_feedback))

    def send_kinematics_for_motor(self, can_id: int, control_data: ControlData) -> None:
        """
        Simulate sending kinematics command.

        Feedback is scheduled to be delivered after a random latency,
        simulating CAN bus round-trip time.
        """
        # Check for packet drop
        if random.random() < self._drop_rate:
            return

        # Generate mock feedback with realistic noise
        mock_feedback = FeedbackData(
            angle=control_data.angle + np.random.normal(0, 0.01),
            velocity=control_data.velocity + np.random.normal(0, 0.1),
            effort=control_data.effort + np.random.normal(0, 0.05),
            voltage=24.0 + np.random.normal(0, 0.1),
            temp_motor=35.0 + np.random.normal(0, 1.0),
            temp_pcb=30.0 + np.random.normal(0, 0.5),
            flags=0,
        )

        # Schedule feedback delivery with random latency
        latency = max(0.0001, np.random.normal(self._latency_mean, self._latency_std))
        deliver_time = time.perf_counter() + latency

        with self._queue_lock:
            self._feedback_queue.append((deliver_time, can_id, mock_feedback))


class MockMotorCANControllerSync:
    """
    Synchronous mock controller (original behavior).

    Use this for testing synchronous code paths.
    Feedback is delivered immediately in the same thread.
    """

    def __init__(self, **kwargs):
        self._config_callbacks: dict = {}
        self._feedback_callbacks: dict = {}
        print("[DRY-RUN] Mock CAN controller (sync) initialized")

    def start(self) -> None:
        print("[DRY-RUN] Mock CAN controller started")

    def stop(self) -> None:
        print("[DRY-RUN] Mock CAN controller stopped")

    def set_config_callback(self, can_id: int, callback: Callable) -> None:
        self._config_callbacks[can_id] = callback

    def set_feedback_callback(self, can_id: int, callback: Callable) -> None:
        self._feedback_callbacks[can_id] = callback

    def get_motor_configuration(self, can_id: int) -> None:
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

    def start_motor(self, can_id: int) -> None:
        print(f"[DRY-RUN] Motor {can_id} started")

    def stop_all_motors(self) -> None:
        print("[DRY-RUN] All motors stopped")

    def stop_motor(self, can_id: int) -> None:
        print(f"[DRY-RUN] Motor {can_id} stopped")

    def ping_motor(self, can_id: int) -> None:
        """Immediate feedback delivery for ping (synchronous)."""
        if can_id in self._feedback_callbacks:
            mock_feedback = FeedbackData(
                angle=np.random.normal(0, 0.01),
                velocity=np.random.normal(0, 0.1),
                effort=np.random.normal(0, 0.05),
                voltage=24.0 + np.random.normal(0, 0.1),
                temp_motor=35.0 + np.random.normal(0, 1.0),
                temp_pcb=30.0 + np.random.normal(0, 0.5),
                flags=0,
            )
            self._feedback_callbacks[can_id](can_id, mock_feedback)

    def send_kinematics_for_motor(self, can_id: int, control_data: ControlData) -> None:
        """Immediate feedback delivery (synchronous)."""
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
