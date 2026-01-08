"""
Core SystemIdentification class for robot actuator identification.

Handles:
- Motor configuration and setup
- Chirp signal generation and control
- Asynchronous feedback collection and timing
- Results saving (JSON, PyTorch, plots)
- Communication statistics tracking
"""

from __future__ import annotations

import json
import sys
import threading
import time
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

# Handle both direct execution and module import
_script_dir = Path(__file__).resolve().parent
if str(_script_dir) not in sys.path:
    sys.path.insert(0, str(_script_dir))

# ruff: noqa: E402
from humanoid_messages.can import (
    CANBusError,
    ConfigurationData,
    ControlData,
    MotorCANController,
)

from async_loop import AsyncControlLoop, AsyncControlLoopBuilder
from chirp import ChirpGenerator, IKChirpGenerator
from controllers import MockMotorCANController
from ik_registry import IKRegistry

if TYPE_CHECKING:
    from typing import Any

# Optional MuJoCo controller
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


class SystemIdentification:
    """Main system identification controller with async communication."""

    def __init__(
        self,
        config_file: str,
        motor_ids: list[int] | None = None,
        dry_run: bool = False,
        use_mujoco: bool = False,
    ):
        self.config = self._load_config(config_file)
        self.dry_run = dry_run
        self.use_mujoco = use_mujoco

        # Resolve motor IDs
        self.motor_ids = self._resolve_motor_ids(motor_ids)
        print(f"Motor CAN IDs to identify: {self.motor_ids}")

        # Initialize controller
        self.controller = self._create_controller()

        # State tracking
        self.motor_configs: dict[int, ConfigurationData] = {}
        self.config_received = threading.Event()
        self.configs_to_receive = set(self.motor_ids)

        # Chirp generators
        self.chirp_generators: dict[int, ChirpGenerator] = {}
        self.ik_generators: dict[str, IKChirpGenerator] = {}
        self.motor_to_ik_group: dict[int, tuple[str, int]] = {}
        self.direct_motors: set[int] = set()

        # Feedback collection (populated by async loop)
        self.feedback_data: dict[int, list[dict]] = defaultdict(list)

        # Command tracking
        self.commanded_angles: dict[int, float] = {}
        self.commanded_ik_inputs: dict[str, dict[str, float]] = {}

        # Timing
        self.start_time = 0.0
        self.sample_count = 0

        # Async loop and stats
        self.async_loop: AsyncControlLoop | None = None
        self.comm_stats: dict[str, Any] = {}

        # Position tracking for ping-based queries
        self._current_positions: dict[int, float] = {}
        self._positions_lock = threading.Lock()
        self._pending_pings: set[int] = set()
        self._pings_received = threading.Event()

        # Motor safety limits: {can_id: (min_pos, max_pos)}
        self.motor_limits: dict[int, tuple[float, float]] = {}

    def _load_config(self, config_file: str) -> dict:
        with Path(config_file).open() as f:
            config = json.load(f)
            print(config)
            return config

    def _resolve_motor_ids(self, motor_ids: list[int] | None) -> list[int]:
        if motor_ids:
            return motor_ids
        if "motor_ids" in self.config:
            return [int(x) for x in self.config["motor_ids"]]
        motors = self.config.get("motors", {})
        if motors:
            return [int(k) for k in motors.keys()]
        return [0]

    def _create_controller(self):
        if self.dry_run:
            print("[DRY-RUN] Running without hardware - using mock controller")
            # Get mock-specific settings from config
            mock_cfg = self.config.get("mock", {})
            return MockMotorCANController(
                latency_mean=mock_cfg.get("latency_mean", 0.002),
                latency_std=mock_cfg.get("latency_std", 0.0005),
                drop_rate=mock_cfg.get("drop_rate", 0.0),
                **self.config.get("can_interface", {}),
            )

        if self.use_mujoco:
            if not MUJOCO_AVAILABLE:
                raise ImportError("MuJoCo controller not available.")
            mujoco_cfg = self.config.get("mujoco", {})
            print("[MUJOCO] Using MuJoCo simulation controller")
            return MujocoMotorController(
                sim_host=mujoco_cfg.get("host", "127.0.0.1"),
                send_port=mujoco_cfg.get("send_port", 5000),
                recv_port=mujoco_cfg.get("recv_port", 5001),
            )

        return MotorCANController(**self.config["can_interface"])

    def _config_callback(self, can_id: int, config: ConfigurationData) -> None:
        self.motor_configs[can_id] = config
        print(f"Received configuration from motor {can_id}")
        self.configs_to_receive.discard(can_id)
        if not self.configs_to_receive:
            self.config_received.set()

    def _ping_feedback_callback(self, can_id: int, feedback: Any) -> None:
        """Callback for ping responses - updates current positions."""
        with self._positions_lock:
            self._current_positions[can_id] = feedback.angle
            self._pending_pings.discard(can_id)
            if not self._pending_pings:
                self._pings_received.set()

    def ping_current_positions(self, timeout: float = 1.0) -> dict[int, float]:
        """
        Get current motor positions by pinging motors.

        Uses ping_motor() to request feedback without sending control commands.
        This is cleaner than sending dummy commands with zero stiffness/damping.

        Args:
            timeout: Maximum time to wait for all responses [s]

        Returns:
            Dict mapping motor_id -> current angle [rad]
        """
        # Setup tracking
        with self._positions_lock:
            self._pending_pings = set(self.motor_ids)
            self._pings_received.clear()

        # Register callbacks for ping responses
        for can_id in self.motor_ids:
            self.controller.set_feedback_callback(can_id, self._ping_feedback_callback)

        # Ping all motors
        for can_id in self.motor_ids:
            self.controller.ping_motor(can_id)

        # Wait for responses
        if not self._pings_received.wait(timeout=timeout):
            with self._positions_lock:
                missing = self._pending_pings
                if missing:
                    print(f"Warning: No ping response from motors: {sorted(missing)}")

        # Return positions (zeros for motors that didn't respond)
        with self._positions_lock:
            return {
                can_id: self._current_positions.get(can_id, 0.0)
                for can_id in self.motor_ids
            }

    def _clamp_angle(self, can_id: int, angle: float) -> float:
        """Clamp angle to configured safety limits."""
        if can_id in self.motor_limits:
            min_lim, max_lim = self.motor_limits[can_id]
            return max(min_lim, min(max_lim, angle))
        return angle

    def _angular_interpolation(
        self,
        start: dict[int, float],
        target: dict[int, float],
        alpha: float,
    ) -> dict[int, float]:
        """Interpolate between angles using shortest angular path."""
        result = {}
        for can_id in target:
            s = start.get(can_id, self._current_positions.get(can_id, 0.0))
            t = target[can_id]
            # Wrap difference to [-π, π] for shortest path
            diff = (t - s + np.pi) % (2 * np.pi) - np.pi
            result[can_id] = s + diff * alpha
        return result

    def _get_control_params(self, can_id: int) -> dict:
        """Get control parameters for a motor (per-motor > per-group > global)."""
        global_params = self.config["control_parameters"]

        # Check per-motor config
        motor_cfg = self.config.get("motors", {}).get(str(can_id), {})
        if "control_parameters" in motor_cfg:
            p = motor_cfg["control_parameters"]
            return {
                "velocity": p.get("velocity", global_params["velocity"]),
                "effort": p.get("effort", global_params["effort"]),
                "stiffness": p.get("stiffness", global_params["stiffness"]),
                "damping": p.get("damping", global_params["damping"]),
            }

        # Check per-group config
        if can_id in self.motor_to_ik_group:
            group_name, _ = self.motor_to_ik_group[can_id]
            for g in self.config.get("ik_groups", []):
                if g["name"] == group_name and "control_parameters" in g:
                    p = g["control_parameters"]
                    return {
                        "velocity": p.get("velocity", global_params["velocity"]),
                        "effort": p.get("effort", global_params["effort"]),
                        "stiffness": p.get("stiffness", global_params["stiffness"]),
                        "damping": p.get("damping", global_params["damping"]),
                    }

        return global_params.copy()

    def _get_initial_chirp_positions(self) -> dict[int, float]:
        """Get first chirp position for each motor without consuming samples."""
        positions = {}

        # Direct motors
        for can_id in self.direct_motors:
            if can_id in self.chirp_generators:
                gen = self.chirp_generators[can_id]
                gen.reset()
                angle, _ = gen.get_next()
                positions[can_id] = angle
                gen.reset()  # Reset again so main loop starts fresh

        # IK groups
        for group_name, ik_gen in self.ik_generators.items():
            group_motors = sorted(
                [m for m, (g, _) in self.motor_to_ik_group.items() if g == group_name],
                key=lambda m: self.motor_to_ik_group[m][1],
            )
            ik_gen.reset()
            _, angles, _ = ik_gen.get_next()
            ik_gen.reset()
            for i, mid in enumerate(group_motors):
                if i < len(angles):
                    positions[mid] = angles[i]

        return positions

    def _send_interpolated_commands(
        self,
        positions: dict[int, float],
        rate: float = 50.0,
    ) -> None:
        """Send position commands at given rate, used for init/finalize phases."""
        for can_id, angle in positions.items():
            safe_angle = self._clamp_angle(can_id, angle)
            self.commanded_angles[can_id] = angle
            params = self._get_control_params(can_id)
            ctrl = ControlData(
                angle=safe_angle,
                velocity=params["velocity"],
                effort=params["effort"],
                stiffness=params["stiffness"],
                damping=params["damping"],
            )
            self.controller.send_kinematics_for_motor(can_id, ctrl)

    def _initialize_to_chirp_start(self, duration: float = 2.0) -> None:
        """Smoothly interpolate from current position to initial chirp position."""
        print("\n=== Initialization Phase ===")
        print("Getting current motor positions via ping...")

        start = self.ping_current_positions(timeout=1.0)
        print(f"  Current positions: {start}")

        target = self._get_initial_chirp_positions()
        print(f"  Target positions: {target}")

        print(f"Interpolating to initial chirp position over {duration}s...")
        rate = 50.0
        period = 1.0 / rate
        t_start = time.perf_counter()

        while True:
            elapsed = time.perf_counter() - t_start
            alpha = min(1.0, elapsed / duration)
            positions = self._angular_interpolation(start, target, alpha)
            self._send_interpolated_commands(positions, rate)
            if alpha >= 1.0:
                break
            time.sleep(period)

        print("  ✓ Initialization complete\n")

    def _finalize_to_zero(self, duration: float = 2.0) -> None:
        """Smoothly interpolate from current position back to zero."""
        print("\n=== Finalization Phase ===")
        print("Interpolating back to zero position...")

        start = {can_id: self.commanded_angles.get(can_id, 0.0) for can_id in self.motor_ids}
        target = {can_id: 0.0 for can_id in self.motor_ids}

        rate = 50.0
        period = 1.0 / rate
        t_start = time.perf_counter()

        while True:
            elapsed = time.perf_counter() - t_start
            alpha = min(1.0, elapsed / duration)
            positions = self._angular_interpolation(start, target, alpha)
            self._send_interpolated_commands(positions, rate)
            if alpha >= 1.0:
                break
            time.sleep(period)

        print("  ✓ Finalization complete\n")

    def setup(self) -> None:
        """Initialize controller and configure motors."""
        print(f"Starting controller for motors: {self.motor_ids}")

        # Load motor safety limits from config
        self._load_motor_limits()

        try:
            self.controller.start()
        except CANBusError as e:
            error_msg = str(e)
            if "No such device" in error_msg or "can0" in error_msg.lower():
                print(f"\nError: {error_msg}")
                print("\nNo CAN device found. Options:")
                print("  1. Use MuJoCo simulation:  --mujoco")
                print("  2. Use dry-run mode:       --dry-run")
                print("  3. Connect CAN device:     sudo ip link set can0 up type can bitrate 1000000")
                raise SystemExit(1) from e
            raise
        time.sleep(0.1)

        # Wait for connection (MuJoCo or hardware)
        if not self.dry_run:
            self._wait_for_connection()

        # Initialize generators
        self._setup_generators()

    def _wait_for_connection(self, timeout: float = 30.0, retry_interval: float = 2.0) -> None:
        """Wait for motors to respond before proceeding."""
        print(f"Waiting for motor connection...")

        # Register config callbacks
        for can_id in self.motor_ids:
            self.controller.set_config_callback(can_id, self._config_callback)

        start_time = time.perf_counter()
        attempt = 0

        while True:
            attempt += 1
            elapsed = time.perf_counter() - start_time

            if elapsed >= timeout:
                print(f"\nTimeout: No response from motors after {timeout:.0f}s")
                if self.use_mujoco:
                    print("  → Start MuJoCo simulation first: python -m hoku.hoku_mujoco")
                else:
                    print("  → Check CAN connection and motor power")
                raise SystemExit(1)

            # Reset tracking for this attempt
            self.configs_to_receive = set(self.motor_ids)
            self.config_received.clear()

            # Request configs
            for can_id in self.motor_ids:
                self.controller.get_motor_configuration(can_id)

            # Wait for responses
            if self.config_received.wait(timeout=retry_interval):
                print(f"Received configurations from {len(self.motor_configs)} motors")
                return

            # Print waiting message
            remaining = timeout - elapsed
            print(f"  [{attempt}] Waiting for motors... ({remaining:.0f}s remaining)")

    def _load_motor_limits(self) -> None:
        """Load motor safety limits from config."""
        # From direct motors config
        for mid_str, cfg in self.config.get("motors", {}).items():
            if "limits" in cfg:
                self.motor_limits[int(mid_str)] = (
                    float(cfg["limits"][0]),
                    float(cfg["limits"][1]),
                )

        # From IK groups (overrides direct if both specified)
        for group in self.config.get("ik_groups", []):
            if "limits" in group:
                for mid_str, lim in group["limits"].items():
                    self.motor_limits[int(mid_str)] = (float(lim[0]), float(lim[1]))

        if self.motor_limits:
            print(f"Motor safety limits: {self.motor_limits}")

    def _setup_generators(self) -> None:
        """Setup chirp generators from config."""
        chirp_cfg = self.config["chirp"]
        motors_cfg = self.config.get("motors", {})
        ik_groups_cfg = self.config.get("ik_groups", [])

        # Setup IK groups
        motors_in_ik = self._setup_ik_groups(chirp_cfg, ik_groups_cfg)

        # Validate and setup direct motors
        self._validate_motor_config(motors_in_ik, motors_cfg)
        self._setup_direct_motors(chirp_cfg, motors_cfg)

        if not self.ik_generators and not self.chirp_generators:
            raise ValueError("No chirp generators initialized!")

    def _setup_ik_groups(self, chirp_cfg: dict, ik_groups_cfg: list) -> set[int]:
        """Setup IK chirp generators. Returns set of motors in IK groups."""
        motors_in_ik: set[int] = set()

        for group in ik_groups_cfg:
            name = group["name"]
            ik_type = group["ik_type"]
            motor_ids = group["motor_ids"]

            ik_info = IKRegistry.get(ik_type)
            if not ik_info:
                print(f"Error: Unknown IK type '{ik_type}' for '{name}'")
                continue

            input_names = ik_info["input_names"]
            gc = group.get("chirp", {})

            # Get transformation parameters
            scale_1 = gc.get(f"scale_{input_names[0]}", gc.get("scale_1", 1.0))
            scale_2 = gc.get(f"scale_{input_names[1]}", gc.get("scale_2", 1.0))
            dir_1 = gc.get(f"direction_{input_names[0]}", gc.get("direction_1", 1.0))
            dir_2 = gc.get(f"direction_{input_names[1]}", gc.get("direction_2", 1.0))
            bias_1 = gc.get(f"bias_{input_names[0]}", gc.get("bias_1", 0.0))
            bias_2 = gc.get(f"bias_{input_names[1]}", gc.get("bias_2", 0.0))

            self.ik_generators[name] = IKChirpGenerator(
                ik_type=ik_type,
                f_start=chirp_cfg["f_start"],
                f_end=chirp_cfg["f_end"],
                duration=chirp_cfg["duration"],
                sample_rate=chirp_cfg["sample_rate"],
                sweep_type=chirp_cfg.get("sweep_type", "logarithmic"),
                scale_1=scale_1, scale_2=scale_2,
                direction_1=dir_1, direction_2=dir_2,
                bias_1=bias_1, bias_2=bias_2,
            )

            for idx, mid in enumerate(motor_ids):
                self.motor_to_ik_group[mid] = (name, idx)
                motors_in_ik.add(mid)
            self.commanded_ik_inputs[name] = {n: 0.0 for n in input_names}

            # Log setup
            gen = self.ik_generators[name]
            print(f"IK group '{name}' initialized:")
            print(f"  IK type: {ik_type} (inputs: {input_names})")
            print(f"  Motors: {motor_ids}")
            print(f"  {input_names[0]}: scale={scale_1:.3f}, dir={dir_1:+.0f}, bias={bias_1:.3f}")
            print(f"    -> amplitude={gen.effective_amplitude_1:.4f} rad, offset={gen.effective_offset_1:.4f} rad")
            print(f"  {input_names[1]}: scale={scale_2:.3f}, dir={dir_2:+.0f}, bias={bias_2:.3f}")
            print(f"    -> amplitude={gen.effective_amplitude_2:.4f} rad, offset={gen.effective_offset_2:.4f} rad")

        return motors_in_ik

    def _validate_motor_config(self, motors_in_ik: set[int], motors_cfg: dict) -> None:
        """Validate motor configuration."""
        motors_in_direct = {int(k) for k in motors_cfg.keys() if k.isdigit()}

        # Check for overlap
        overlap = motors_in_ik & motors_in_direct
        if overlap:
            raise ValueError(f"Motors {sorted(overlap)} in both ik_groups and motors config")

        # Check all motors have config
        missing = set(self.motor_ids) - motors_in_ik - motors_in_direct
        if missing:
            raise ValueError(f"Motors {sorted(missing)} have no config")

        self.direct_motors = set(self.motor_ids) - motors_in_ik

        # Check direct motors have config
        direct_missing = self.direct_motors - motors_in_direct
        if direct_missing:
            raise ValueError(f"Direct motors {sorted(direct_missing)} missing config")

    def _setup_direct_motors(self, chirp_cfg: dict, motors_cfg: dict) -> None:
        """Setup direct chirp generators."""
        for can_id in self.direct_motors:
            mc = motors_cfg[str(can_id)]
            self.chirp_generators[can_id] = ChirpGenerator(
                f_start=chirp_cfg["f_start"],
                f_end=chirp_cfg["f_end"],
                duration=chirp_cfg["duration"],
                sample_rate=chirp_cfg["sample_rate"],
                sweep_type=chirp_cfg.get("sweep_type", "logarithmic"),
                scale=mc.get("scale", 1.0),
                direction=mc.get("direction", 1.0),
                bias=mc.get("bias", 0.0),
            )

        if self.direct_motors:
            print(f"Direct chirp generators for motors: {sorted(self.direct_motors)}")
            for can_id in sorted(self.direct_motors):
                gen = self.chirp_generators[can_id]
                mc = motors_cfg[str(can_id)]
                print(f"  Motor {can_id}: scale={mc.get('scale', 1.0):.3f}, "
                      f"dir={mc.get('direction', 1.0):+.0f}, bias={mc.get('bias', 0.0):.3f}")
                print(f"    -> amplitude={gen.effective_amplitude:.4f} rad, "
                      f"offset={gen.effective_offset:.4f} rad")

    def _generate_control(self) -> tuple[dict[int, ControlData], bool]:
        """Generate control commands for all motors."""
        control_data: dict[int, ControlData] = {}
        is_complete = False

        # IK groups
        for group_name, gen in self.ik_generators.items():
            ik_inputs, angles, complete = gen.get_next()
            if complete:
                is_complete = True
            self.commanded_ik_inputs[group_name] = ik_inputs

            group_motors = sorted(
                [m for m, (g, _) in self.motor_to_ik_group.items() if g == group_name],
                key=lambda m: self.motor_to_ik_group[m][1]
            )
            for i, mid in enumerate(group_motors):
                if i < len(angles):
                    self.commanded_angles[mid] = angles[i]
                    safe_angle = self._clamp_angle(mid, angles[i])
                    params = self._get_control_params(mid)
                    control_data[mid] = ControlData(
                        angle=safe_angle,
                        velocity=params["velocity"],
                        effort=params["effort"],
                        stiffness=params["stiffness"],
                        damping=params["damping"],
                    )

        # Direct motors
        for can_id in self.direct_motors:
            if can_id not in self.chirp_generators:
                continue
            angle, complete = self.chirp_generators[can_id].get_next()
            if complete:
                is_complete = True
            self.commanded_angles[can_id] = angle
            safe_angle = self._clamp_angle(can_id, angle)
            params = self._get_control_params(can_id)
            control_data[can_id] = ControlData(
                angle=safe_angle,
                velocity=params["velocity"],
                effort=params["effort"],
                stiffness=params["stiffness"],
                damping=params["damping"],
            )

        return control_data, is_complete

    def _get_commanded_angles(self) -> dict[int, float]:
        """Get current commanded angles."""
        return dict(self.commanded_angles)

    def _get_ik_inputs(self) -> dict[str, dict[str, float]]:
        """Get current IK inputs."""
        return {k: dict(v) for k, v in self.commanded_ik_inputs.items()}

    def _get_progress(self) -> float:
        """Get current progress (0.0 to 1.0)."""
        if self.ik_generators:
            return next(iter(self.ik_generators.values())).get_progress()
        elif self.chirp_generators:
            return next(iter(self.chirp_generators.values())).get_progress()
        return 0.0

    def run_identification(self) -> None:
        """Main identification loop using async architecture."""
        chirp_cfg = self.config["chirp"]
        expected_rate = chirp_cfg["sample_rate"]
        rate_limited = chirp_cfg.get("rate_limit", True)

        if not rate_limited:
            print("\nWARNING: rate_limit=False is deprecated with async mode.")
            print("Commands will be sent at the configured sample_rate.")

        self._print_run_info(chirp_cfg, expected_rate)
        self._start_motors()

        # Get interpolation duration from config
        interp_duration = self.config.get("interpolation_duration", 2.0)

        # Initialize: smoothly move from current position to chirp start
        self._initialize_to_chirp_start(duration=interp_duration)

        # Build async control loop
        use_busy_wait = self.dry_run  # Use busy-wait in dry-run for accurate timing
        self.async_loop = (
            AsyncControlLoopBuilder(self.controller, self.motor_ids)
            .with_rate(expected_rate)
            .with_busy_wait(use_busy_wait)
            .with_ik_mapping(self.motor_to_ik_group)
            .build()
        )

        # Register feedback callbacks
        self.async_loop.register_callbacks()

        # Run the async loop
        self.start_time = time.perf_counter()
        self.async_loop.run(
            generate_control=self._generate_control,
            get_commanded_angles=self._get_commanded_angles,
            get_ik_inputs=self._get_ik_inputs,
            get_progress=self._get_progress,
        )

        # Collect results
        self.feedback_data = defaultdict(list, self.async_loop.get_feedback_data())
        self.sample_count = self.async_loop.sample_count
        self.comm_stats = self.async_loop.get_stats()

        # Print summary
        self._print_summary()

        # Finalize: smoothly return to zero position
        self._finalize_to_zero(duration=interp_duration)

    def _print_run_info(self, chirp_cfg: dict, expected_rate: float) -> None:
        print("\n" + "=" * 60)
        print("STARTING ASYNC SYSTEM IDENTIFICATION")
        print("=" * 60)
        print(f"Target rate: {expected_rate} Hz")
        print(f"Duration: {chirp_cfg['duration']} seconds")
        print(f"Frequency sweep: {chirp_cfg['f_start']} - {chirp_cfg['f_end']} Hz")
        print(f"Expected samples: {int(expected_rate * chirp_cfg['duration'])}")
        print(f"Motor CAN IDs: {self.motor_ids}")
        if self.ik_generators:
            print(f"IK groups: {list(self.ik_generators.keys())}")
        if self.direct_motors:
            print(f"Direct motors: {sorted(self.direct_motors)}")
        print("=" * 60 + "\n")

    def _start_motors(self) -> None:
        for can_id in self.motor_ids:
            self.controller.start_motor(can_id)
            time.sleep(0.125)
        print(f"Motors started: {self.motor_ids}")
        time.sleep(1.0)

    def _print_summary(self) -> None:
        """Print identification summary."""
        print("\n" + "=" * 60)
        print("IDENTIFICATION COMPLETE")
        print("=" * 60)

        print(f"\nMotors: {self.motor_ids}")
        if self.ik_generators:
            print(f"IK groups: {list(self.ik_generators.keys())}")
        if self.direct_motors:
            print(f"Direct motors: {sorted(self.direct_motors)}")

        print(f"\nSamples collected: {self.sample_count}")

        # Feedback data summary
        for mid in self.motor_ids:
            fb_count = len(self.feedback_data.get(mid, []))
            pct = (fb_count / self.sample_count * 100) if self.sample_count > 0 else 0
            print(f"  Motor {mid}: {fb_count}/{self.sample_count} feedbacks ({pct:.1f}%)")

        # Print detailed communication stats
        if self.async_loop:
            self.async_loop.print_stats()

    def save_results(self, output_file: str) -> None:
        """Save results as JSON."""
        from results import save_json
        save_json(self, output_file)

    def save_torch(self, output_file: str) -> None:
        """Save results as PyTorch .pt file."""
        from results import save_torch
        save_torch(self, output_file)

    def save_plots(self, output_dir: str) -> None:
        """Save plots for each motor and IK group."""
        from results import save_plots
        save_plots(self, output_dir)

    def save_stats(self, output_file: str) -> None:
        """Save communication statistics as JSON."""
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with output_path.open("w") as f:
            json.dump(self.comm_stats, f, indent=2)

        print(f"Communication stats saved to: {output_path}")

    def cleanup(self) -> None:
        """Stop all motors and close controller."""
        print("\nStopping motors...")
        try:
            if self.async_loop:
                self.async_loop.stop()
            self.controller.stop_all_motors()
            time.sleep(0.1)
        except Exception as e:
            print(f"Error stopping motors: {e}")
            self.controller.stop()
