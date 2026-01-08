"""
Test modules for hardware validation.

Each test is a standalone function that returns a TestResult.
Tests can be run individually or as part of the hardware_test.py suite.
"""

from __future__ import annotations

import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any

# Setup imports
_script_dir = Path(__file__).resolve().parent
if str(_script_dir) not in sys.path:
    sys.path.insert(0, str(_script_dir))

from humanoid_messages.can import (  # noqa: E402
    ConfigurationData,
    ControlData,
    MotorCANController,
)
from controllers import MockMotorCANController  # noqa: E402

# Optional MuJoCo
try:
    from hoku.mujoco_controller import MujocoMotorController
    MUJOCO_AVAILABLE = True
except ImportError:
    _ws_root = Path(__file__).resolve().parent.parent.parent
    if str(_ws_root) not in sys.path:
        sys.path.insert(0, str(_ws_root))
    try:
        from hoku.mujoco_controller import MujocoMotorController
        MUJOCO_AVAILABLE = True
    except ImportError:
        MUJOCO_AVAILABLE = False


@dataclass
class TestResult:
    """Result of a single test."""
    name: str
    passed: bool
    message: str
    details: dict = field(default_factory=dict)
    duration_ms: float = 0.0


def create_controller(config: dict, dry_run: bool, use_mujoco: bool):
    """Create appropriate controller based on mode."""
    if dry_run:
        mock_cfg = config.get("mock", {})
        return MockMotorCANController(
            latency_mean=mock_cfg.get("latency_mean", 0.002),
            latency_std=mock_cfg.get("latency_std", 0.0005),
            drop_rate=mock_cfg.get("drop_rate", 0.0),
        )

    if use_mujoco:
        if not MUJOCO_AVAILABLE:
            raise ImportError("MuJoCo controller not available")
        mujoco_cfg = config.get("mujoco", {})
        return MujocoMotorController(
            sim_host=mujoco_cfg.get("host", "127.0.0.1"),
            send_port=mujoco_cfg.get("send_port", 5000),
            recv_port=mujoco_cfg.get("recv_port", 5001),
        )

    return MotorCANController(**config.get("can_interface", {}))


# =============================================================================
# Test: Connection
# =============================================================================

def run_connection_test(
    controller,
    motor_ids: list[int],
    verbose: bool = False,
) -> TestResult:
    """Test motor configuration responses."""
    print("\n[1] CONNECTION TEST")
    print("-" * 40)

    start = time.perf_counter()
    configs_received: dict[int, ConfigurationData] = {}
    pending = set(motor_ids)
    event = threading.Event()

    def on_config(can_id: int, config: ConfigurationData):
        configs_received[can_id] = config
        pending.discard(can_id)
        if verbose:
            print(f"  ✓ Motor {can_id} responded")
        if not pending:
            event.set()

    # Register callbacks
    for can_id in motor_ids:
        controller.set_config_callback(can_id, on_config)

    # Request configs
    print(f"Requesting configs from {len(motor_ids)} motors...")
    for can_id in motor_ids:
        controller.get_motor_configuration(can_id)

    # Wait for responses
    success = event.wait(timeout=3.0)
    duration = (time.perf_counter() - start) * 1000

    if success:
        print(f"  All {len(motor_ids)} motors responded in {duration:.1f}ms")
        return TestResult(
            name="Connection",
            passed=True,
            message=f"All {len(motor_ids)} motors responded",
            details={"responded": list(configs_received.keys())},
            duration_ms=duration,
        )
    else:
        missing = sorted(pending)
        responded = sorted(configs_received.keys())
        print(f"  ✗ Missing: {missing}")
        if responded:
            print(f"  ✓ Responded: {responded}")
        return TestResult(
            name="Connection",
            passed=False,
            message=f"Missing response from motors: {missing}",
            details={"responded": responded, "missing": missing},
            duration_ms=duration,
        )


# =============================================================================
# Test: Ping
# =============================================================================

def run_ping_test(
    controller,
    motor_ids: list[int],
    verbose: bool = False,
) -> TestResult:
    """Test ping_motor functionality."""
    print("\n[2] PING TEST")
    print("-" * 40)

    start = time.perf_counter()
    positions: dict[int, float] = {}
    pending = set(motor_ids)
    event = threading.Event()

    def on_feedback(can_id: int, feedback: Any):
        positions[can_id] = feedback.angle
        pending.discard(can_id)
        if verbose:
            print(f"  ✓ Motor {can_id}: {feedback.angle:.4f} rad")
        if not pending:
            event.set()

    # Register callbacks
    for can_id in motor_ids:
        controller.set_feedback_callback(can_id, on_feedback)

    # Send pings
    print(f"Pinging {len(motor_ids)} motors...")
    for can_id in motor_ids:
        controller.ping_motor(can_id)

    # Wait for responses
    success = event.wait(timeout=2.0)
    duration = (time.perf_counter() - start) * 1000

    if success:
        print(f"  All {len(motor_ids)} pings received in {duration:.1f}ms")
        for mid, pos in sorted(positions.items()):
            print(f"    Motor {mid}: {pos:+.4f} rad")
        return TestResult(
            name="Ping",
            passed=True,
            message=f"All {len(motor_ids)} pings responded",
            details={"positions": positions},
            duration_ms=duration,
        )
    else:
        missing = sorted(pending)
        responded = sorted(positions.keys())
        print(f"  ✗ No ping response from: {missing}")
        return TestResult(
            name="Ping",
            passed=False,
            message=f"No ping response from motors: {missing}",
            details={"responded": responded, "missing": missing},
            duration_ms=duration,
        )


# =============================================================================
# Helper: Get current position
# =============================================================================

def _get_current_position(
    controller,
    motor_id: int,
    timeout: float = 1.0,
) -> float | None:
    """Get current motor position via ping. Returns None on timeout."""
    position: list[float | None] = [None]
    event = threading.Event()

    def on_feedback(can_id: int, feedback: Any):
        position[0] = feedback.angle
        event.set()

    controller.set_feedback_callback(motor_id, on_feedback)
    controller.ping_motor(motor_id)

    if event.wait(timeout=timeout):
        return position[0]
    return None


# =============================================================================
# Test: Command
# =============================================================================

def run_command_test(
    controller,
    motor_ids: list[int],
    verbose: bool = False,
) -> TestResult:
    """Test position command with feedback (holds current position)."""
    print("\n[3] COMMAND TEST")
    print("-" * 40)

    # Use first motor only to minimize risk
    test_motor = motor_ids[0]

    # First, get current position via ping
    print(f"Getting current position of motor {test_motor}...")
    current_pos = _get_current_position(controller, test_motor, timeout=1.0)

    if current_pos is None:
        return TestResult(
            name="Command",
            passed=False,
            message="Could not get current position via ping",
            details={},
            duration_ms=0,
        )

    print(f"  Current position: {current_pos:+.4f} rad")
    print("  Sending hold command (same position, low stiffness)...")

    start = time.perf_counter()
    feedback_received = threading.Event()
    feedback_data: dict[str, Any] = {}

    def on_feedback(can_id: int, feedback: Any):
        feedback_data["angle"] = feedback.angle
        feedback_data["velocity"] = feedback.velocity
        feedback_data["effort"] = feedback.effort
        feedback_received.set()

    # Register callback
    controller.set_feedback_callback(test_motor, on_feedback)

    # Start motor
    controller.start_motor(test_motor)
    time.sleep(0.1)

    # Send hold command at CURRENT position (safe - no movement)
    cmd = ControlData(
        angle=current_pos,  # Hold current position
        velocity=0.0,
        effort=0.0,
        stiffness=3.0,  # Very low stiffness
        damping=0.5,    # Some damping to resist disturbances
    )
    controller.send_kinematics_for_motor(test_motor, cmd)

    # Wait for feedback
    success = feedback_received.wait(timeout=1.0)
    duration = (time.perf_counter() - start) * 1000

    if success:
        print(f"  ✓ Feedback received in {duration:.1f}ms")
        print(f"    Position: {feedback_data['angle']:+.4f} rad")
        print(f"    Velocity: {feedback_data['velocity']:+.4f} rad/s")
        print(f"    Effort:   {feedback_data['effort']:+.4f}")
        return TestResult(
            name="Command",
            passed=True,
            message="Position command executed with feedback",
            details=feedback_data,
            duration_ms=duration,
        )
    else:
        print(f"  ✗ No feedback received after {duration:.1f}ms")
        return TestResult(
            name="Command",
            passed=False,
            message="No feedback received for position command",
            details={},
            duration_ms=duration,
        )


# =============================================================================
# Test: Timing
# =============================================================================

def run_timing_test(
    controller,
    motor_ids: list[int],
    verbose: bool = False,
    num_samples: int = 100,
) -> TestResult:
    """Measure command-to-feedback latency (holds current position)."""
    print("\n[4] TIMING TEST")
    print("-" * 40)

    test_motor = motor_ids[0]

    # Get current position first
    current_pos = _get_current_position(controller, test_motor, timeout=1.0)
    if current_pos is None:
        return TestResult(
            name="Timing",
            passed=False,
            message="Could not get current position via ping",
            details={},
            duration_ms=0,
        )

    print(f"Measuring latency for motor {test_motor} ({num_samples} samples)")
    print(f"  Holding position: {current_pos:+.4f} rad")

    latencies: list[float] = []
    feedback_event = threading.Event()
    recv_time = [0.0]

    def on_feedback(can_id: int, feedback: Any):
        recv_time[0] = time.perf_counter()
        feedback_event.set()

    # Register callback
    controller.set_feedback_callback(test_motor, on_feedback)

    # Start motor
    controller.start_motor(test_motor)
    time.sleep(0.1)

    # Measure latencies - hold current position (safe)
    cmd = ControlData(
        angle=current_pos,
        velocity=0.0,
        effort=0.0,
        stiffness=3.0,
        damping=0.5,
    )

    for i in range(num_samples):
        feedback_event.clear()
        send_time = time.perf_counter()
        controller.send_kinematics_for_motor(test_motor, cmd)

        if feedback_event.wait(timeout=0.1):
            latency = (recv_time[0] - send_time) * 1000
            latencies.append(latency)
            if verbose and i % 20 == 0:
                print(f"  Sample {i}: {latency:.2f}ms")
        else:
            if verbose:
                print(f"  Sample {i}: timeout")

        time.sleep(0.01)  # 10ms between samples

    if not latencies:
        return TestResult(
            name="Timing",
            passed=False,
            message="No latency measurements collected",
            details={},
            duration_ms=0,
        )

    # Calculate stats
    avg_latency = sum(latencies) / len(latencies)
    min_latency = min(latencies)
    max_latency = max(latencies)
    success_rate = len(latencies) / num_samples * 100

    print(f"  Samples: {len(latencies)}/{num_samples} ({success_rate:.1f}%)")
    print(f"  Latency: avg={avg_latency:.2f}ms, "
          f"min={min_latency:.2f}ms, max={max_latency:.2f}ms")

    # Pass if >90% success and avg latency <10ms
    passed = success_rate >= 90 and avg_latency < 10

    return TestResult(
        name="Timing",
        passed=passed,
        message=f"Avg: {avg_latency:.2f}ms ({success_rate:.0f}% success)",
        details={
            "avg_ms": avg_latency,
            "min_ms": min_latency,
            "max_ms": max_latency,
            "success_rate": success_rate,
            "samples": len(latencies),
        },
        duration_ms=avg_latency,
    )


# =============================================================================
# Test: Stress (prolonged async loop)
# =============================================================================

def run_stress_test(
    controller,
    motor_ids: list[int],
    verbose: bool = False,
    duration: float = 5.0,
    rate: float = 100.0,
) -> TestResult:
    """
    Prolonged test to measure missed deadlines and communication reliability.

    Runs an async control loop for the specified duration, holding current
    position while tracking:
    - Missed deadlines
    - Feedback rate per motor
    - Latency statistics
    - Late feedbacks
    """
    print("\n[5] STRESS TEST")
    print("-" * 40)

    test_motor = motor_ids[0]

    # Get current position first
    current_pos = _get_current_position(controller, test_motor, timeout=1.0)
    if current_pos is None:
        return TestResult(
            name="Stress",
            passed=False,
            message="Could not get current position via ping",
            details={},
            duration_ms=0,
        )

    print(f"Running stress test for {duration}s at {rate} Hz")
    print(f"  Motor: {test_motor}")
    print(f"  Holding position: {current_pos:+.4f} rad")
    print(f"  Expected samples: {int(duration * rate)}")

    # Import async loop
    from async_loop import AsyncControlLoop

    # Start motor
    controller.start_motor(test_motor)
    time.sleep(0.1)

    # Create hold command
    hold_cmd = ControlData(
        angle=current_pos,
        velocity=0.0,
        effort=0.0,
        stiffness=3.0,
        damping=0.5,
    )

    # Build async loop for single motor
    async_loop = AsyncControlLoop(
        controller=controller,
        motor_ids=[test_motor],
        target_rate=rate,
        use_busy_wait=False,
    )

    # Track state
    sample_count = [0]
    expected_samples = int(duration * rate)

    def generate_control():
        sample_count[0] += 1
        is_complete = sample_count[0] >= expected_samples
        return {test_motor: hold_cmd}, is_complete

    def get_commanded_angles():
        return {test_motor: current_pos}

    def get_progress():
        return sample_count[0] / expected_samples

    # Register callbacks and run
    async_loop.register_callbacks()

    start = time.perf_counter()
    async_loop.run(
        generate_control=generate_control,
        get_commanded_angles=get_commanded_angles,
        get_progress=get_progress,
    )
    elapsed = time.perf_counter() - start

    # Get stats
    stats = async_loop.get_stats()

    # Extract key metrics
    total_cycles = stats.get("total_cycles", 0)
    missed_deadlines = stats.get("missed_deadlines", 0)
    missed_pct = stats.get("missed_deadline_pct", 0)
    actual_rate = total_cycles / elapsed if elapsed > 0 else 0

    # Per-motor stats (keys are integers, not strings)
    per_motor = stats.get("per_motor", {})
    motor_stats = per_motor.get(test_motor, {})
    fb_received = motor_stats.get("feedbacks_received", 0)
    fb_rate = motor_stats.get("feedback_rate_pct", 0)
    late_fb = motor_stats.get("late_feedbacks", 0)
    avg_latency = motor_stats.get("avg_latency_ms", 0)
    max_latency = motor_stats.get("max_latency_ms", 0)

    # Print results
    print("\n  Results:")
    print(f"    Duration: {elapsed:.2f}s")
    print(f"    Samples: {total_cycles}")
    print(f"    Actual rate: {actual_rate:.1f} Hz (target: {rate} Hz)")
    print(f"    Missed deadlines: {missed_deadlines} ({missed_pct:.1f}%)")
    print(f"    Feedbacks: {fb_received}/{total_cycles} ({fb_rate:.1f}%)")
    print(f"    Late feedbacks: {late_fb}")
    print(f"    Latency: avg={avg_latency:.2f}ms, max={max_latency:.2f}ms")

    # Pass criteria:
    # - < 5% missed deadlines
    # - > 90% feedback rate
    # - avg latency < 10ms
    passed = (
        missed_pct < 5.0 and
        fb_rate > 90.0 and
        avg_latency < 10.0
    )

    status = "PASS" if passed else "FAIL"
    print(f"\n  [{status}] ", end="")
    if missed_pct >= 5.0:
        print(f"Too many missed deadlines ({missed_pct:.1f}%)")
    elif fb_rate <= 90.0:
        print(f"Low feedback rate ({fb_rate:.1f}%)")
    elif avg_latency >= 10.0:
        print(f"High latency ({avg_latency:.2f}ms)")
    else:
        print("All metrics within limits")

    return TestResult(
        name="Stress",
        passed=passed,
        message=f"{total_cycles} samples, {missed_pct:.1f}% missed, "
                f"{fb_rate:.0f}% FB, {avg_latency:.1f}ms lat",
        details={
            "duration_s": elapsed,
            "total_cycles": total_cycles,
            "actual_rate": actual_rate,
            "missed_deadlines": missed_deadlines,
            "missed_pct": missed_pct,
            "feedback_received": fb_received,
            "feedback_rate_pct": fb_rate,
            "late_feedbacks": late_fb,
            "avg_latency_ms": avg_latency,
            "max_latency_ms": max_latency,
        },
        duration_ms=elapsed * 1000,
    )


# =============================================================================
# Summary
# =============================================================================

def print_test_summary(results: list[TestResult]) -> None:
    """Print test results summary."""
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    passed = sum(1 for r in results if r.passed)
    total = len(results)

    for r in results:
        status = "✓ PASS" if r.passed else "✗ FAIL"
        print(f"  [{status}] {r.name}: {r.message}")

    print("-" * 60)
    print(f"  Result: {passed}/{total} tests passed")

    if passed == total:
        print("\n  ✓ All tests passed - ready for system identification")
    else:
        print("\n  ✗ Some tests failed - check connection before proceeding")

    print("=" * 60)
