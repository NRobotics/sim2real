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

from async_loop import AsyncControlLoop, AsyncControlLoopBuilder, busy_sleep
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
        busy_wait: bool = False,
        verbosity: int = 1,
    ):
        self.verbosity = verbosity
        self.config = self._load_config(config_file)
        self.dry_run = dry_run
        self.use_mujoco = use_mujoco
        self.busy_wait = busy_wait

        # Resolve motor IDs
        self.motor_ids = self._resolve_motor_ids(motor_ids)
        self._log(1, f"Motor CAN IDs to identify: {self.motor_ids}")

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

        # Position tracking for feedback-based queries
        self._current_positions: dict[int, float] = {}
        self._positions_lock = threading.Lock()
        self._pending_queries: set[int] = set()
        self._queries_received = threading.Event()

        # Interpolation phase data (for optional saving)
        self.interpolation_data: dict[str, dict[int, list[dict]]] = {
            "start": defaultdict(list),
            "end": defaultdict(list),
        }
        self.save_interpolation: dict[str, bool] = {"start": False, "end": False}
        self._current_interp_phase: str | None = None  # "start" or "end" during interp

        # Motor safety limits: {can_id: (min_pos, max_pos)}
        self.motor_limits: dict[int, tuple[float, float]] = {}

    def _log(self, level: int, msg: str) -> None:
        """Print message if verbosity >= level."""
        if self.verbosity >= level:
            print(msg)

    def _load_config(self, config_file: str) -> dict:
        with Path(config_file).open() as f:
            config = json.load(f)
            self._log(2, str(config))
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
            self._log(1, "[DRY-RUN] Running without hardware - using mock controller")
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
            self._log(1, "[MUJOCO] Using MuJoCo simulation controller")
            return MujocoMotorController(
                sim_host=mujoco_cfg.get("host", "127.0.0.1"),
                send_port=mujoco_cfg.get("send_port", 5000),
                recv_port=mujoco_cfg.get("recv_port", 5001),
            )

        return MotorCANController(**self.config["can_interface"])

    def _config_callback(self, can_id: int, config: ConfigurationData) -> None:
        self.motor_configs[can_id] = config
        self._log(2, f"Received configuration from motor {can_id}")
        self.configs_to_receive.discard(can_id)
        if not self.configs_to_receive:
            self.config_received.set()

    def _query_feedback_callback(self, can_id: int, feedback: Any) -> None:
        """Callback for position query - updates current positions."""
        with self._positions_lock:
            self._current_positions[can_id] = feedback.angle
            self._pending_queries.discard(can_id)
            if not self._pending_queries:
                self._queries_received.set()

    def query_current_positions(self, timeout: float = 1.0) -> dict[int, float]:
        """
        Get current motor positions by sending zero stiffness/damping commands.

        Sends commands with zero stiffness and damping (passive mode) and waits
        for feedback responses. This is required because ping doesn't work on
        some hardware.

        Args:
            timeout: Maximum time to wait for all responses [s]

        Returns:
            Dict mapping motor_id -> current angle [rad]
        """
        # Setup tracking
        with self._positions_lock:
            self._pending_queries = set(self.motor_ids)
            self._queries_received.clear()

        # Register callbacks for feedback responses
        for can_id in self.motor_ids:
            self.controller.set_feedback_callback(can_id, self._query_feedback_callback)

        # Send zero stiffness/damping commands (passive mode)
        # This triggers feedback without applying any force
        for can_id in self.motor_ids:
            ctrl = ControlData(
                angle=0.0,  # Target doesn't matter with zero gains
                velocity=0.0,
                effort=0.0,
                stiffness=0.0,
                damping=0.0,
            )
            self.controller.send_kinematics_for_motor(can_id, ctrl)

        # Wait for responses
        if not self._queries_received.wait(timeout=timeout):
            with self._positions_lock:
                missing = self._pending_queries
                if missing:
                    self._log(1, f"Warning: No response from motors: {sorted(missing)}")

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

    def _get_interpolation_config(self) -> dict:
        """Get interpolation configuration with backward compatibility."""
        # New format
        if "interpolation" in self.config:
            return self.config["interpolation"]

        # Legacy format - convert to new structure
        legacy_duration = self.config.get("interpolation_duration", 2.0)
        return {
            "start": {
                "enabled": True,
                "save": False,
                "duration": legacy_duration,
                "wait_before": 0.0,
                "wait_after": 0.0,
            },
            "end": {
                "enabled": True,
                "save": False,
                "duration": legacy_duration,
                "wait_before": 0.0,
                "wait_after": 0.0,
            },
        }

    def _setup_interp_phase(self, phase: str) -> None:
        """Setup state for an interpolation phase."""
        self._current_interp_phase = phase
        self.interpolation_data[phase] = defaultdict(list)

    def _end_interp_phase(self) -> None:
        """End an interpolation phase."""
        self._current_interp_phase = None

    def _run_interpolation_phase(
        self,
        phase: str,
        start: dict[int, float],
        target: dict[int, float],
        duration: float,
        wait_before: float,
        wait_after: float,
    ) -> None:
        """
        Run an interpolation phase using the async control loop.

        Args:
            phase: Phase name ("start" or "end") for data collection
            start: Starting positions per motor
            target: Target positions per motor
            duration: Interpolation duration in seconds
            wait_before: Hold at start position for this duration before interpolating
            wait_after: Hold at target position for this duration after interpolating
        """
        rate = self.config["chirp"]["sample_rate"]
        total_duration = wait_before + duration + wait_after
        num_samples = int(rate * total_duration)

        if num_samples == 0:
            return

        # Compute sample boundaries for each sub-phase
        wait_before_samples = int(rate * wait_before)
        interp_samples = int(rate * duration)
        # wait_after_samples fills the rest

        # Current sample counter (mutable in closure)
        sample_idx = [0]

        def generate_interp_control() -> dict[int, ControlData]:
            """Generate control for current interpolation sample."""
            idx = sample_idx[0]
            sample_idx[0] += 1

            # Determine which sub-phase we're in
            if idx < wait_before_samples:
                # Hold at start position
                positions = start
            elif idx < wait_before_samples + interp_samples:
                # Interpolating
                interp_idx = idx - wait_before_samples
                alpha = interp_idx / interp_samples if interp_samples > 0 else 1.0
                positions = self._angular_interpolation(start, target, alpha)
            else:
                # Hold at target position
                positions = target

            # Build control data
            control_data: dict[int, ControlData] = {}
            for can_id, angle in positions.items():
                safe_angle = self._clamp_angle(can_id, angle)
                self.commanded_angles[can_id] = angle
                params = self._get_control_params(can_id)
                control_data[can_id] = ControlData(
                    angle=safe_angle,
                    velocity=params["velocity"],
                    effort=params["effort"],
                    stiffness=params["stiffness"],
                    damping=params["damping"],
                )

            return control_data

        # Setup feedback collection
        self._setup_interp_phase(phase)

        # Build and run async loop for this phase
        interp_loop = (
            AsyncControlLoopBuilder(self.controller, self.motor_ids)
            .with_rate(rate)
            .with_busy_wait(self.busy_wait)
            .with_verbosity(0)  # Silent for interpolation
            .with_ik_mapping(self.motor_to_ik_group)
            .build()
        )
        interp_loop.register_callbacks()
        interp_loop.run_for_samples(
            generate_control=generate_interp_control,
            get_commanded_angles=self._get_commanded_angles,
            num_samples=num_samples,
            silent=True,
        )

        # Collect interpolation feedback data
        if self.save_interpolation.get(phase, False):
            self.interpolation_data[phase] = defaultdict(list, interp_loop.get_feedback_data())

        self._end_interp_phase()

    def _initialize_to_chirp_start(
        self,
        duration: float = 2.0,
        wait_before: float = 0.0,
        wait_after: float = 0.0,
    ) -> None:
        """Smoothly interpolate from current position to initial chirp position."""
        self._log(1, "\n=== Initialization Phase ===")
        self._log(2, "Getting current motor positions...")

        start = self.query_current_positions(timeout=1.0)
        self._log(2, f"  Current positions: {start}")

        target = self._get_initial_chirp_positions()
        self._log(2, f"  Target positions: {target}")

        if self.save_interpolation.get("start", False):
            self._log(2, "  Recording interpolation data...")

        self._run_interpolation_phase(
            phase="start",
            start=start,
            target=target,
            duration=duration,
            wait_before=wait_before,
            wait_after=wait_after,
        )

        if self.save_interpolation.get("start", False):
            start_samples = next((len(v) for v in self.interpolation_data["start"].values() if v), 0)
            self._log(2, f"  Collected {start_samples} feedback samples")

        self._log(2, "  ✓ Initialization complete\n")

    def _finalize_to_zero(
        self,
        duration: float = 2.0,
        wait_before: float = 0.0,
        wait_after: float = 0.0,
    ) -> None:
        """Smoothly interpolate from current position back to zero."""
        self._log(1, "\n=== Finalization Phase ===")

        start = {can_id: self.commanded_angles.get(can_id, 0.0) for can_id in self.motor_ids}
        target = {can_id: 0.0 for can_id in self.motor_ids}

        if self.save_interpolation.get("end", False):
            self._log(2, "  Recording interpolation data...")

        self._run_interpolation_phase(
            phase="end",
            start=start,
            target=target,
            duration=duration,
            wait_before=wait_before,
            wait_after=wait_after,
        )

        if self.save_interpolation.get("end", False):
            end_samples = next((len(v) for v in self.interpolation_data["end"].values() if v), 0)
            self._log(2, f"  Collected {end_samples} feedback samples")

        self._log(2, "  ✓ Finalization complete\n")

    def setup(self) -> None:
        """Initialize controller and configure motors."""
        self._log(1, f"Starting controller for motors: {self.motor_ids}")

        # Load motor safety limits from config
        self._load_motor_limits()

        try:
            self.controller.start()
        except CANBusError as e:
            error_msg = str(e)
            if "No such device" in error_msg or "can0" in error_msg.lower():
                # Always print errors
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
        self._log(2, f"Waiting for motor connection...")

        # Register config callbacks
        for can_id in self.motor_ids:
            self.controller.set_config_callback(can_id, self._config_callback)

        start_time = time.perf_counter()
        attempt = 0

        while True:
            attempt += 1
            elapsed = time.perf_counter() - start_time

            if elapsed >= timeout:
                # Always print timeout errors
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
                self._log(2, f"Received configurations from {len(self.motor_configs)} motors")
                return

            # Print waiting message
            remaining = timeout - elapsed
            self._log(2, f"  [{attempt}] Waiting for motors... ({remaining:.0f}s remaining)")

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
            self._log(2, f"Motor safety limits: {self.motor_limits}")

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
                # Always print errors
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
            self._log(2, f"IK group '{name}' initialized:")
            self._log(2, f"  IK type: {ik_type} (inputs: {input_names})")
            self._log(2, f"  Motors: {motor_ids}")
            self._log(2, f"  {input_names[0]}: scale={scale_1:.3f}, dir={dir_1:+.0f}, bias={bias_1:.3f}")
            self._log(2, f"    -> amplitude={gen.effective_amplitude_1:.4f} rad, offset={gen.effective_offset_1:.4f} rad")
            self._log(2, f"  {input_names[1]}: scale={scale_2:.3f}, dir={dir_2:+.0f}, bias={bias_2:.3f}")
            self._log(2, f"    -> amplitude={gen.effective_amplitude_2:.4f} rad, offset={gen.effective_offset_2:.4f} rad")

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
            self._log(2, f"Direct chirp generators for motors: {sorted(self.direct_motors)}")
            for can_id in sorted(self.direct_motors):
                gen = self.chirp_generators[can_id]
                mc = motors_cfg[str(can_id)]
                self._log(2, f"  Motor {can_id}: scale={mc.get('scale', 1.0):.3f}, "
                      f"dir={mc.get('direction', 1.0):+.0f}, bias={mc.get('bias', 0.0):.3f}")
                self._log(2, f"    -> amplitude={gen.effective_amplitude:.4f} rad, "
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

    def _run_rate_test(
        self,
        target_rate: float,
        duration: float = 1.0,
        tolerance: float = 0.05,
    ) -> bool:
        """
        Pre-flight test to verify command and feedback rates.

        Sends dummy commands at the target rate and measures actual performance.

        Args:
            target_rate: Target command rate in Hz
            duration: Test duration in seconds
            tolerance: Acceptable deviation from target (0.05 = 5%)

        Returns:
            True if rates are acceptable, False otherwise
        """
        self._log(1, "\n" + "=" * 60)
        self._log(1, "PRE-FLIGHT RATE TEST")
        self._log(1, "=" * 60)
        self._log(2, f"Target rate: {target_rate} Hz")
        self._log(2, f"Test duration: {duration}s")
        self._log(2, f"Tolerance: ±{tolerance * 100:.0f}%")

        # Get current positions to use as hold position
        hold_positions = self.query_current_positions(timeout=1.0)
        params = self.config["control_parameters"]

        # Tracking
        feedback_counts: dict[int, int] = {mid: 0 for mid in self.motor_ids}
        feedback_lock = threading.Lock()

        def feedback_callback(can_id: int, feedback: Any) -> None:
            with feedback_lock:
                feedback_counts[can_id] += 1

        # Register callbacks
        for can_id in self.motor_ids:
            self.controller.set_feedback_callback(can_id, feedback_callback)

        # Run test loop - use same timing approach as AsyncLoop
        target_period = 1.0 / target_rate
        commands_sent = 0
        missed_deadlines = 0

        start_time = time.perf_counter()
        next_deadline = start_time

        while True:
            cycle_start = time.perf_counter()
            elapsed = cycle_start - start_time

            if elapsed >= duration:
                break

            # Send hold commands to all motors
            for can_id in self.motor_ids:
                angle = hold_positions.get(can_id, 0.0)
                ctrl = ControlData(
                    angle=angle,
                    velocity=params["velocity"],
                    effort=params["effort"],
                    stiffness=params["stiffness"],
                    damping=params["damping"],
                )
                self.controller.send_kinematics_for_motor(can_id, ctrl)

            commands_sent += 1

            # Compute next deadline and sleep
            next_deadline += target_period
            now = time.perf_counter()
            sleep_time = next_deadline - now

            if sleep_time < 0:
                missed_deadlines += 1
                # Reset deadline to prevent accumulating drift
                next_deadline = now + target_period
            elif sleep_time > 0:
                if self.busy_wait:
                    busy_sleep(sleep_time)
                elif sleep_time > 0.002:
                    # Hybrid: sleep for bulk, busy-wait for precision
                    time.sleep(sleep_time - 0.0015)
                    remaining = next_deadline - time.perf_counter()
                    if remaining > 0:
                        busy_sleep(remaining, yield_interval=0.0002)
                else:
                    time.sleep(0)  # Yield
                    remaining = next_deadline - time.perf_counter()
                    if remaining > 0:
                        busy_sleep(remaining, yield_interval=0.0002)

        # Capture end time before waiting for feedbacks
        end_time = time.perf_counter()

        # Wait briefly for final feedbacks
        time.sleep(0.05)

        # Calculate results (using actual command duration, not including wait)
        total_time = end_time - start_time
        actual_cmd_rate = commands_sent / total_time
        cmd_rate_error = abs(actual_cmd_rate - target_rate) / target_rate

        total_feedbacks = sum(feedback_counts.values())
        expected_feedbacks = commands_sent * len(self.motor_ids)
        feedback_rate_pct = (total_feedbacks / expected_feedbacks * 100) \
            if expected_feedbacks > 0 else 0

        # Print results
        self._log(1, "\nResults:")
        self._log(1, f"  Commands sent: {commands_sent}")
        self._log(1, f"  Actual cmd rate: {actual_cmd_rate:.1f} Hz "
              f"(target: {target_rate:.0f} Hz, "
              f"error: {cmd_rate_error * 100:.1f}%)")
        self._log(1, f"  Missed deadlines: {missed_deadlines} "
              f"({missed_deadlines / commands_sent * 100:.1f}%)")
        self._log(1, f"  Total feedbacks: {total_feedbacks}/{expected_feedbacks} "
              f"({feedback_rate_pct:.1f}%)")

        # Per-motor feedback
        self._log(2, "\n  Per-motor feedback:")
        all_motors_ok = True
        for mid in self.motor_ids:
            count = feedback_counts[mid]
            pct = count / commands_sent * 100 if commands_sent > 0 else 0
            status = "✓" if pct >= 95 else "⚠" if pct >= 80 else "✗"
            self._log(2, f"    Motor {mid}: {count}/{commands_sent} ({pct:.1f}%) {status}")
            if pct < 80:
                all_motors_ok = False

        # Determine pass/fail
        cmd_rate_ok = cmd_rate_error <= tolerance
        feedback_ok = feedback_rate_pct >= 80

        self._log(1, f"\n  Command rate: {'✓ PASS' if cmd_rate_ok else '✗ FAIL'}")
        self._log(1, f"  Feedback rate: {'✓ PASS' if feedback_ok else '⚠ WARNING'}")

        passed = cmd_rate_ok and all_motors_ok
        self._log(1, "=" * 60)

        if not passed:
            # Always show warnings
            print("\n⚠ Rate test showed potential issues.")
            print("  Consider: --busy-wait, --realtime, or lower sample_rate")

        return passed

    def run_rate_check(
        self,
        target_rate: float | None = None,
        duration: float = 2.0,
    ) -> None:
        """
        Run rate check only and report achievable frequency.

        Tests multiple rates to find the maximum sustainable rate.
        This is useful for determining the best sample_rate setting.

        Args:
            target_rate: Rate to test (Hz). If None, tests multiple rates.
            duration: Duration for each test (seconds)
        """
        self._log(1, "\n" + "=" * 60)
        self._log(1, "RATE CHECK MODE")
        self._log(1, "=" * 60)

        # Start motors first
        self._start_motors()

        if target_rate:
            # Test single rate
            rates_to_test = [target_rate]
        else:
            # Test multiple rates to find maximum
            rates_to_test = [100, 200, 300, 400, 500, 600, 700, 800, 1000]

        results = []

        for rate in rates_to_test:
            self._log(2, f"\n--- Testing {rate} Hz ---")
            passed = self._run_rate_test(rate, duration=duration, tolerance=0.05)
            results.append((rate, passed))

        # Summary
        self._log(1, "\n" + "=" * 60)
        self._log(1, "RATE CHECK SUMMARY")
        self._log(1, "=" * 60)

        if len(rates_to_test) == 1:
            rate, passed = results[0]
            if passed:
                self._log(1, f"\n✓ Rate {rate} Hz is achievable")
            else:
                self._log(1, f"\n✗ Rate {rate} Hz may have issues")
                self._log(1, "  Try: --busy-wait, --realtime, or lower the rate")
        else:
            self._log(1, "\nRate test results:")
            max_passing = None
            for rate, passed in results:
                status = "✓ PASS" if passed else "✗ FAIL"
                self._log(1, f"  {rate:4d} Hz: {status}")
                if passed:
                    max_passing = rate

            if max_passing:
                self._log(1, f"\n✓ Recommended max rate: {max_passing} Hz")
            else:
                self._log(1, "\n⚠ All rates had issues. Check connection and try --busy-wait")

        self._log(1, "=" * 60)

    def run_identification(self) -> None:
        """Main identification loop using async architecture."""
        chirp_cfg = self.config["chirp"]
        expected_rate = chirp_cfg["sample_rate"]
        rate_limited = chirp_cfg.get("rate_limit", True)

        if not rate_limited:
            self._log(1, "\nWARNING: rate_limit=False is deprecated with async mode.")
            self._log(1, "Commands will be sent at the configured sample_rate.")

        self._print_run_info(chirp_cfg, expected_rate)
        self._start_motors()

        # Run pre-flight rate test
        self._run_rate_test(expected_rate, duration=1.0)

        # Get interpolation config
        interp_cfg = self._get_interpolation_config()
        start_cfg = interp_cfg.get("start", {})
        end_cfg = interp_cfg.get("end", {})
        self.save_interpolation = {
            "start": start_cfg.get("save", False),
            "end": end_cfg.get("save", False),
        }

        # Initialize: smoothly move from current position to chirp start
        if start_cfg.get("enabled", True):
            self._initialize_to_chirp_start(
                duration=start_cfg.get("duration", 2.0),
                wait_before=start_cfg.get("wait_before", 0.0),
                wait_after=start_cfg.get("wait_after", 0.0),
            )
        else:
            self._log(2, "\n=== Initialization Phase (skipped) ===\n")

        # Build async control loop
        # Use busy-wait if explicitly requested or in dry-run mode
        use_busy_wait = self.busy_wait
        self.async_loop = (
            AsyncControlLoopBuilder(self.controller, self.motor_ids)
            .with_rate(expected_rate)
            .with_busy_wait(use_busy_wait)
            .with_verbosity(self.verbosity)
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

        # Finalize: smoothly return to zero position (before summary so we have all data)
        if end_cfg.get("enabled", True):
            self._finalize_to_zero(
                duration=end_cfg.get("duration", 2.0),
                wait_before=end_cfg.get("wait_before", 0.0),
                wait_after=end_cfg.get("wait_after", 0.0),
            )
        else:
            self._log(2, "\n=== Finalization Phase (skipped) ===\n")

        # Print summary (after finalization so interpolation data is complete)
        self._print_summary()

    def _print_run_info(self, chirp_cfg: dict, expected_rate: float) -> None:
        self._log(1, "\n" + "=" * 60)
        self._log(1, "STARTING ASYNC SYSTEM IDENTIFICATION")
        self._log(1, "=" * 60)
        self._log(2, f"Target rate: {expected_rate} Hz")
        self._log(2, f"Duration: {chirp_cfg['duration']} seconds")
        self._log(2, f"Frequency sweep: {chirp_cfg['f_start']} - {chirp_cfg['f_end']} Hz")
        self._log(2, f"Expected samples: {int(expected_rate * chirp_cfg['duration'])}")
        self._log(2, f"Motor CAN IDs: {self.motor_ids}")
        if self.ik_generators:
            self._log(2, f"IK groups: {list(self.ik_generators.keys())}")
        if self.direct_motors:
            self._log(2, f"Direct motors: {sorted(self.direct_motors)}")
        use_busy_wait = self.busy_wait
        self._log(2, f"Busy-wait timing: {'enabled' if use_busy_wait else 'disabled'}")
        self._log(1, "=" * 60 + "\n")

    def _start_motors(self) -> None:
        for can_id in self.motor_ids:
            self.controller.start_motor(can_id)
            time.sleep(0.125)
        self._log(2, f"Motors started: {self.motor_ids}")
        time.sleep(1.0)

    def _print_summary(self) -> None:
        """Print identification summary."""
        self._log(1, "\n" + "=" * 60)
        self._log(1, "IDENTIFICATION COMPLETE")
        self._log(1, "=" * 60)

        self._log(2, f"\nMotors: {self.motor_ids}")
        if self.ik_generators:
            self._log(2, f"IK groups: {list(self.ik_generators.keys())}")
        if self.direct_motors:
            self._log(2, f"Direct motors: {sorted(self.direct_motors)}")

        self._log(1, f"\nSamples collected: {self.sample_count} (chirp phase)")

        # Interpolation data summary at level 1 - show per-motor count (matches .pt tensor rows)
        if self.save_interpolation.get("start", False):
            start_data = self.interpolation_data.get("start", {})
            # Get count from first motor with data
            start_samples = next((len(v) for v in start_data.values() if v), 0)
            self._log(1, f"  + {start_samples} samples (start interpolation)")
        if self.save_interpolation.get("end", False):
            end_data = self.interpolation_data.get("end", {})
            end_samples = next((len(v) for v in end_data.values() if v), 0)
            self._log(1, f"  + {end_samples} samples (end interpolation)")

        # Feedback data summary (per-motor at level 2)
        for mid in self.motor_ids:
            fb_count = len(self.feedback_data.get(mid, []))
            pct = (fb_count / self.sample_count * 100) if self.sample_count > 0 else 0
            self._log(2, f"  Motor {mid}: {fb_count}/{self.sample_count} feedbacks ({pct:.1f}%)")

        # Print key communication stats at level 1
        if self.async_loop:
            stats = self.comm_stats
            self._log(1, f"\nCommunication stats:")
            self._log(1, f"  Elapsed time: {stats.get('elapsed_time', 0):.2f}s")
            self._log(1, f"  Missed deadlines: {stats.get('missed_deadlines', 0)} "
                  f"({stats.get('missed_deadline_pct', 0):.1f}%)")
            self._log(1, f"  Overall feedback rate: {stats.get('overall_feedback_rate_pct', 0):.1f}%")

        # Print detailed communication stats at level 2
        if self.async_loop and self.verbosity >= 2:
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

        self._log(2, f"Communication stats saved to: {output_path}")

    def cleanup(self) -> None:
        """Stop all motors and close controller."""
        self._log(1, "\nStopping motors...")
        try:
            if self.async_loop:
                self.async_loop.stop()
            self.controller.stop_all_motors()
            time.sleep(0.1)
        except Exception as e:
            # Always print errors
            print(f"Error stopping motors: {e}")
            self.controller.stop()
