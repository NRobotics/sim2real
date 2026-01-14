"""
Asynchronous control loop for system identification.

Decouples command sending from feedback reception:
- Commands are sent at fixed intervals regardless of feedback
- Feedback is collected asynchronously
- Statistics track timing and missed callbacks

This replaces the synchronous (stop-and-wait) architecture.
"""

from __future__ import annotations

import threading
import time
from dataclasses import asdict
from typing import TYPE_CHECKING, Protocol

from motor_stats import MotorStatsManager

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Any

    from humanoid_messages.can import ControlData, FeedbackData


def busy_sleep(duration: float, yield_interval: float = 0.0001) -> None:
    """
    Accurate sleep using busy-wait with periodic yields.

    Args:
        duration: Total sleep duration in seconds
        yield_interval: How often to yield to other threads (default 100us)
    """
    end_time = time.perf_counter() + duration
    next_yield = time.perf_counter() + yield_interval

    while True:
        now = time.perf_counter()
        if now >= end_time:
            break
        # Periodically yield to other threads
        if now >= next_yield:
            time.sleep(0)  # Yield to OS scheduler
            next_yield = now + yield_interval


class ControllerProtocol(Protocol):
    """Protocol for motor controllers (real or mock)."""

    def send_kinematics_for_motor(self, can_id: int, control_data: ControlData) -> None:
        ...

    def set_feedback_callback(
        self, can_id: int, callback: Callable[[int, FeedbackData], None]
    ) -> None:
        ...


class AsyncControlLoop:
    """
    Asynchronous control loop that decouples command sending from feedback.

    Commands are sent at a fixed rate regardless of when (or if) feedback arrives.
    Feedback is collected asynchronously and matched to commands by sample number.
    """

    def __init__(
        self,
        controller: ControllerProtocol,
        motor_ids: list[int],
        target_rate: float,
        use_busy_wait: bool = False,
        verbosity: int = 2,
    ):
        """
        Initialize async control loop.

        Args:
            controller: Motor controller instance
            motor_ids: List of motor CAN IDs to control
            target_rate: Target command rate in Hz
            use_busy_wait: Use busy-wait for timing (more accurate but CPU intensive)
            verbosity: Logging level (0=silent, 1=start/end only, 2=progress updates)
        """
        self.controller = controller
        self.motor_ids = motor_ids
        self.target_rate = target_rate
        self.target_period = 1.0 / target_rate
        self.use_busy_wait = use_busy_wait
        self.verbosity = verbosity

        # State
        self.running = False
        self.sample_count = 0
        self.start_time = 0.0

        # Stats manager
        self.stats = MotorStatsManager(motor_ids)

        # Feedback storage - thread-safe
        self._feedback_lock = threading.Lock()
        self._feedback_data: dict[int, list[dict[str, Any]]] = {
            mid: [] for mid in motor_ids
        }
        self._current_commanded: dict[int, float] = {mid: 0.0 for mid in motor_ids}
        self._current_ik_inputs: dict[str, dict[str, float]] = {}
        self._motor_to_ik_group: dict[int, tuple[str, int]] = {}

        # Current cycle deadline (for latency tracking)
        self._cycle_deadline = 0.0

        # Progress callback
        self._progress_callback: Callable[[int, float], None] | None = None

    def set_ik_mapping(
        self,
        motor_to_ik_group: dict[int, tuple[str, int]],
    ) -> None:
        """Set IK group mapping for feedback annotation."""
        self._motor_to_ik_group = motor_to_ik_group

    def set_progress_callback(
        self, callback: Callable[[int, float], None] | None
    ) -> None:
        """Set callback for progress updates: callback(sample, progress_pct)."""
        self._progress_callback = callback

    def _create_feedback_callback(self, can_id: int) -> Callable[[int, FeedbackData], None]:
        """Create a feedback callback for a specific motor."""

        def callback(cid: int, feedback: FeedbackData) -> None:
            self._handle_feedback(cid, feedback)

        return callback

    def _handle_feedback(self, can_id: int, feedback: FeedbackData) -> None:
        """Handle incoming feedback (called from controller's receive thread)."""
        recv_time = time.perf_counter()

        # Record timing stats
        status = self.stats.record_feedback_received(
            can_id, self._cycle_deadline, recv_time
        )

        # Build feedback record
        with self._feedback_lock:
            sample = self.sample_count
            fb = asdict(feedback)
            fb["timestamp"] = recv_time - self.start_time
            fb["sample"] = sample
            fb["recv_latency_ms"] = status.get("latency", 0) * 1000
            fb["late"] = status.get("late", False)
            fb["commanded_angle"] = self._current_commanded.get(can_id, 0.0)
            fb["angle_error"] = feedback.angle - fb["commanded_angle"]

            # Add IK group info if applicable
            if can_id in self._motor_to_ik_group:
                group_name, _ = self._motor_to_ik_group[can_id]
                fb["ik_group"] = group_name
                for name, val in self._current_ik_inputs.get(group_name, {}).items():
                    fb[f"commanded_{name}"] = val

            self._feedback_data[can_id].append(fb)

    def register_callbacks(self) -> None:
        """Register feedback callbacks with the controller."""
        for can_id in self.motor_ids:
            callback = self._create_feedback_callback(can_id)
            self.controller.set_feedback_callback(can_id, callback)

    def run(
        self,
        generate_control: Callable[[], tuple[dict[int, ControlData], bool]],
        get_commanded_angles: Callable[[], dict[int, float]],
        get_ik_inputs: Callable[[], dict[str, dict[str, float]]] | None = None,
        get_progress: Callable[[], float] | None = None,
    ) -> None:
        """
        Run the async control loop until generate_control signals completion.

        Args:
            generate_control: Function that generates control commands.
                Returns (dict[motor_id, ControlData], is_complete)
            get_commanded_angles: Function returning current commanded angles per motor
            get_ik_inputs: Optional function returning current IK inputs per group
            get_progress: Optional function returning progress 0.0-1.0
        """
        self._run_loop(
            generate_control=generate_control,
            get_commanded_angles=get_commanded_angles,
            get_ik_inputs=get_ik_inputs,
            get_progress=get_progress,
            num_samples=None,
            silent=False,
        )

    def run_for_samples(
        self,
        generate_control: Callable[[], dict[int, ControlData]],
        get_commanded_angles: Callable[[], dict[int, float]],
        num_samples: int,
        get_ik_inputs: Callable[[], dict[str, dict[str, float]]] | None = None,
        silent: bool = False,
    ) -> None:
        """
        Run the async control loop for a fixed number of samples.

        Args:
            generate_control: Function that generates control commands.
                Returns dict[motor_id, ControlData] (no is_complete flag)
            get_commanded_angles: Function returning current commanded angles per motor
            num_samples: Exact number of samples to collect
            get_ik_inputs: Optional function returning current IK inputs per group
            silent: If True, suppress startup logging
        """
        # Wrap generator to add is_complete flag
        sample_counter = [0]

        def wrapped_generate() -> tuple[dict[int, ControlData], bool]:
            # Check completion BEFORE incrementing - signals complete on iteration AFTER last sample
            # This ensures all num_samples are sent before the loop breaks
            is_complete = sample_counter[0] >= num_samples
            sample_counter[0] += 1
            control = generate_control()
            return control, is_complete

        self._run_loop(
            generate_control=wrapped_generate,
            get_commanded_angles=get_commanded_angles,
            get_ik_inputs=get_ik_inputs,
            get_progress=lambda: sample_counter[0] / num_samples,
            num_samples=num_samples,
            silent=silent,
        )

    def _run_loop(
        self,
        generate_control: Callable[[], tuple[dict[int, ControlData], bool]],
        get_commanded_angles: Callable[[], dict[int, float]],
        get_ik_inputs: Callable[[], dict[str, dict[str, float]]] | None,
        get_progress: Callable[[], float] | None,
        num_samples: int | None,
        silent: bool,
    ) -> None:
        """Internal loop implementation shared by run() and run_for_samples()."""
        self.running = True
        self.sample_count = 0
        self.start_time = time.perf_counter()

        last_send_time: float | None = None
        first_sample_done = False

        if self.verbosity >= 1 and not silent:
            print(f"\n[AsyncLoop] Starting at {self.target_rate} Hz")
            print(f"[AsyncLoop] Target period: {self.target_period * 1000:.2f}ms")
            print(f"[AsyncLoop] Busy-wait: {self.use_busy_wait}")

        # Start stats timer AFTER startup prints to get accurate elapsed time
        self.stats.start()

        while self.running:
            cycle_start = time.perf_counter()

            # Generate control commands
            control_data, is_complete = generate_control()

            if is_complete:
                self.running = False
                break

            # Update commanded angles for feedback annotation
            with self._feedback_lock:
                self._current_commanded = get_commanded_angles()
                if get_ik_inputs:
                    self._current_ik_inputs = get_ik_inputs()

            # Set deadline for this cycle's feedback
            self._cycle_deadline = cycle_start + self.target_period

            # Send commands to all motors
            send_start = time.perf_counter()
            for can_id, data in control_data.items():
                self.stats.record_command_sent(can_id, send_start)
                self.controller.send_kinematics_for_motor(can_id, data)
            send_time = time.perf_counter() - send_start
            self.stats.record_send_time(send_time)

            self.sample_count += 1

            # Calculate loop time
            loop_time = time.perf_counter() - cycle_start

            # Rate limiting - sleep until next cycle
            if last_send_time is not None:
                elapsed = time.perf_counter() - last_send_time
                sleep_time = self.target_period - elapsed

                missed_deadline = sleep_time < 0
                self.stats.record_cycle(loop_time, missed_deadline)

                if sleep_time > 0:
                    self._accurate_sleep(sleep_time, last_send_time)
            else:
                self.stats.record_cycle(loop_time, False)

            last_send_time = time.perf_counter()

            # Reset start time after first sample for accurate rate calculation
            if not first_sample_done:
                self.start_time = time.perf_counter()
                first_sample_done = True

            # Progress reporting (only at verbosity >= 2, and only for long runs)
            if self.verbosity >= 2 and not silent and self.sample_count % 100 == 0:
                self._report_progress(get_progress)

        # Final check for missed feedbacks (only for non-silent runs)
        if not silent:
            self._check_final_feedback_status()

    def _accurate_sleep(
        self, sleep_time: float, last_send_time: float
    ) -> None:
        """
        Sleep accurately using hybrid approach.

        Uses time.sleep() for bulk of the wait, then busy-waits for precision.
        Periodically yields to allow receive threads to run.
        """
        if sleep_time <= 0:
            return

        if self.use_busy_wait:
            # Pure busy-wait with periodic yields
            busy_sleep(sleep_time)
        elif sleep_time > 0.002:
            # Hybrid: sleep for bulk, busy-wait for precision
            # Sleep less to leave room for receive thread
            sleep_portion = sleep_time - 0.0015
            if sleep_portion > 0:
                time.sleep(sleep_portion)

            # Busy-wait for remaining time with yields
            remaining = self.target_period - (time.perf_counter() - last_send_time)
            if remaining > 0:
                busy_sleep(remaining, yield_interval=0.0002)
        else:
            # Short sleep: just yield and busy-wait
            time.sleep(0)  # Yield once
            remaining = self.target_period - (time.perf_counter() - last_send_time)
            if remaining > 0:
                busy_sleep(remaining, yield_interval=0.0002)

    def _report_progress(
        self, get_progress: Callable[[], float] | None
    ) -> None:
        """Report progress to console."""
        elapsed = time.perf_counter() - self.start_time
        intervals = self.sample_count - 1
        actual_rate = intervals / elapsed if elapsed > 0 and intervals > 0 else 0

        progress_pct = get_progress() * 100 if get_progress else 0

        summary = self.stats.get_summary()
        late = summary["total_late_feedbacks"]
        fb_rate = summary["overall_feedback_rate_pct"]

        print(
            f"[AsyncLoop] Progress: {progress_pct:.1f}% | "
            f"Sample: {self.sample_count} | "
            f"Rate: {actual_rate:.1f}/{self.target_rate:.0f} Hz | "
            f"FB: {fb_rate:.0f}% | Late: {late}"
        )

        if self._progress_callback:
            self._progress_callback(self.sample_count, progress_pct)

    def _check_final_feedback_status(self) -> None:
        """Log final feedback status for each motor."""
        # Give a small window for final feedbacks to arrive
        time.sleep(0.05)
        if self.verbosity >= 1:
            with self._feedback_lock:
                for can_id in self.motor_ids:
                    motor_stats = self.stats.get_motor_stats(can_id)
                    sent = motor_stats["commands_sent"]
                    received = motor_stats["feedbacks_received"]
                    if sent > received:
                        pct = (received / sent * 100) if sent > 0 else 0
                        print(
                            f"[AsyncLoop] Motor {can_id}: "
                            f"{received}/{sent} feedbacks ({pct:.1f}%)"
                        )

    def stop(self) -> None:
        """Stop the control loop."""
        self.running = False

    def get_feedback_data(self) -> dict[int, list[dict[str, Any]]]:
        """Get collected feedback data (thread-safe copy)."""
        with self._feedback_lock:
            return {mid: list(data) for mid, data in self._feedback_data.items()}

    def get_stats(self) -> dict[str, Any]:
        """Get communication statistics."""
        return self.stats.get_summary()

    def print_stats(self) -> None:
        """Print formatted statistics."""
        self.stats.print_summary()


class AsyncControlLoopBuilder:
    """Builder for AsyncControlLoop with fluent interface."""

    def __init__(self, controller: ControllerProtocol, motor_ids: list[int]):
        self._controller = controller
        self._motor_ids = motor_ids
        self._target_rate = 100.0
        self._use_busy_wait = False
        self._verbosity = 2
        self._ik_mapping: dict[int, tuple[str, int]] = {}

    def with_rate(self, rate: float) -> AsyncControlLoopBuilder:
        """Set target rate in Hz."""
        self._target_rate = rate
        return self

    def with_busy_wait(self, enabled: bool = True) -> AsyncControlLoopBuilder:
        """Enable busy-wait for accurate timing."""
        self._use_busy_wait = enabled
        return self

    def with_verbosity(self, level: int) -> AsyncControlLoopBuilder:
        """Set verbosity level (0=silent, 1=start/end only, 2=progress updates)."""
        self._verbosity = level
        return self

    def with_ik_mapping(
        self, mapping: dict[int, tuple[str, int]]
    ) -> AsyncControlLoopBuilder:
        """Set IK group mapping."""
        self._ik_mapping = mapping
        return self

    def build(self) -> AsyncControlLoop:
        """Build the AsyncControlLoop."""
        loop = AsyncControlLoop(
            controller=self._controller,
            motor_ids=self._motor_ids,
            target_rate=self._target_rate,
            use_busy_wait=self._use_busy_wait,
            verbosity=self._verbosity,
        )
        if self._ik_mapping:
            loop.set_ik_mapping(self._ik_mapping)
        return loop
