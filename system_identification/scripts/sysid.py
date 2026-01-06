"""
Core SystemIdentification class for robot actuator identification.

Handles:
- Motor configuration and setup
- Chirp signal generation and control
- Feedback collection and timing
- Results saving (JSON, PyTorch, plots)
"""

from __future__ import annotations

import json
import sys
import threading
import time
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

# Handle both direct execution and module import
_script_dir = Path(__file__).resolve().parent
if str(_script_dir) not in sys.path:
    sys.path.insert(0, str(_script_dir))

from humanoid_messages.can import (
    ConfigurationData,
    ControlData,
    FeedbackData,
    MotorCANController,
)

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


def busy_sleep(duration: float) -> None:
    """Accurate sleep using busy-wait."""
    end_time = time.perf_counter() + duration
    while time.perf_counter() < end_time:
        pass


class SystemIdentification:
    """Main system identification controller."""

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

        # Feedback collection
        self.feedback_data: dict[int, list[dict]] = defaultdict(list)
        self.feedback_received = threading.Event()
        self.feedbacks_to_receive: set[int] = set()

        # Command tracking
        self.commanded_angles: dict[int, float] = {}
        self.commanded_ik_inputs: dict[str, dict[str, float]] = {}

        # Timing
        self.start_time = 0.0
        self.sample_count = 0

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
            return MockMotorCANController(**self.config["can_interface"])

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

    def _feedback_callback(self, can_id: int, feedback: FeedbackData) -> None:
        fb = asdict(feedback)
        fb["timestamp"] = time.time() - self.start_time
        fb["sample"] = self.sample_count
        fb["commanded_angle"] = self.commanded_angles.get(can_id, 0.0)
        fb["angle_error"] = feedback.angle - fb["commanded_angle"]

        # Add IK group info
        if can_id in self.motor_to_ik_group:
            group_name, _ = self.motor_to_ik_group[can_id]
            fb["ik_group"] = group_name
            for name, val in self.commanded_ik_inputs.get(group_name, {}).items():
                fb[f"commanded_{name}"] = val

        self.feedback_data[can_id].append(fb)
        self.feedbacks_to_receive.discard(can_id)
        if not self.feedbacks_to_receive:
            self.feedback_received.set()

    def setup(self) -> None:
        """Initialize controller and configure motors."""
        print(f"Starting CAN controller for motors: {self.motor_ids}")
        self.controller.start()
        time.sleep(0.1)

        # Register callbacks
        for can_id in self.motor_ids:
            self.controller.set_config_callback(can_id, self._config_callback)
            self.controller.set_feedback_callback(can_id, self._feedback_callback)

        # Request configurations
        print(f"Requesting configurations from motors: {self.motor_ids}")
        for can_id in self.motor_ids:
            self.controller.get_motor_configuration(can_id)

        if not self.config_received.wait(timeout=2.0):
            print(f"Warning: Missing configs from: {self.configs_to_receive}")
        print(f"Received configurations from {len(self.motor_configs)} motors")

        # Initialize generators
        self._setup_generators()

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

    def run_identification(self) -> None:
        """Main identification loop."""
        ctrl_params = self.config["control_parameters"]
        chirp_cfg = self.config["chirp"]
        expected_rate = chirp_cfg["sample_rate"]
        rate_limited = chirp_cfg.get("rate_limit", True)

        self._print_run_info(chirp_cfg, rate_limited, expected_rate)
        self._start_motors()

        # Timing setup
        target_period = 1.0 / expected_rate if rate_limited else 0.0
        feedback_timeout = target_period * 2 if rate_limited else 0.01
        use_busy_wait = self.dry_run and rate_limited

        self.start_time = time.perf_counter()
        self.sample_count = 0
        loop_times = []
        missed_deadlines = 0
        last_send_time = None
        first_sample_done = False
        is_complete = False

        while not is_complete:
            loop_start = time.perf_counter()

            # Generate and send control
            control_data, is_complete = self._generate_control(ctrl_params)
            self._send_control(control_data)

            # Wait for feedback
            self.feedback_received.wait(timeout=feedback_timeout)

            if not is_complete:
                self.sample_count += 1

            loop_times.append(time.perf_counter() - loop_start)

            # Rate limiting
            if rate_limited and last_send_time is not None:
                elapsed = time.perf_counter() - last_send_time
                sleep_time = target_period - elapsed
                if sleep_time > 0:
                    self._sleep(sleep_time, use_busy_wait, target_period, last_send_time)
                elif not is_complete:
                    missed_deadlines += 1

            last_send_time = time.perf_counter()

            if not first_sample_done:
                self.start_time = time.perf_counter()
                first_sample_done = True

            # Early rate check
            if rate_limited and self.sample_count == 10:
                self._check_rate(loop_times, expected_rate)

            # Progress
            if self.sample_count % 100 == 0:
                self._print_progress(expected_rate, rate_limited)

        self._print_summary(loop_times, expected_rate, rate_limited, missed_deadlines)

    def _print_run_info(self, chirp_cfg: dict, rate_limited: bool, expected_rate: float) -> None:
        print("\nStarting system identification...")
        if rate_limited:
            print(f"Requested rate: {expected_rate} Hz")
        else:
            print("Rate limiting: DISABLED")
            print(f"Signal has {int(expected_rate * chirp_cfg['duration'])} samples")
        print(f"Duration: {chirp_cfg['duration']} seconds")
        print(f"Frequency sweep: {chirp_cfg['f_start']} - {chirp_cfg['f_end']} Hz")
        print(f"Motor CAN IDs: {self.motor_ids}")
        if self.ik_generators:
            print(f"IK groups: {list(self.ik_generators.keys())}")
        if self.direct_motors:
            print(f"Direct motors: {sorted(self.direct_motors)}")

    def _start_motors(self) -> None:
        for can_id in self.motor_ids:
            self.controller.start_motor(can_id)
            time.sleep(0.125)
        print(f"Motors started: {self.motor_ids}")
        time.sleep(1.0)

    def _generate_control(self, ctrl_params: dict) -> tuple[dict[int, ControlData], bool]:
        control_data = {}
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
                    control_data[mid] = ControlData(
                        angle=angles[i],
                        velocity=ctrl_params["velocity"],
                        effort=ctrl_params["effort"],
                        stiffness=ctrl_params["stiffness"],
                        damping=ctrl_params["damping"],
                    )

        # Direct motors
        for can_id in self.direct_motors:
            if can_id not in self.chirp_generators:
                continue
            angle, complete = self.chirp_generators[can_id].get_next()
            if complete:
                is_complete = True
            self.commanded_angles[can_id] = angle
            control_data[can_id] = ControlData(
                angle=angle,
                velocity=ctrl_params["velocity"],
                effort=ctrl_params["effort"],
                stiffness=ctrl_params["stiffness"],
                damping=ctrl_params["damping"],
            )

        return control_data, is_complete

    def _send_control(self, control_data: dict[int, ControlData]) -> None:
        self.feedbacks_to_receive = set(self.motor_ids)
        self.feedback_received.clear()

        # if hasattr(self.controller, 'send_kinematics_batch'):
        #     self.controller.send_kinematics_batch(control_data)
        # else:
        for can_id, data in control_data.items():
            self.controller.send_kinematics_for_motor(can_id, data)

    def _sleep(self, sleep_time: float, use_busy_wait: bool,
               target_period: float, last_send_time: float) -> None:
        if use_busy_wait:
            busy_sleep(sleep_time)
        elif sleep_time > 0.002:
            time.sleep(sleep_time - 0.001)
            remaining = target_period - (time.perf_counter() - last_send_time)
            if remaining > 0:
                busy_sleep(remaining)
        else:
            busy_sleep(sleep_time)

    def _check_rate(self, loop_times: list, expected_rate: float) -> None:
        avg = np.mean(loop_times)
        max_rate = 1.0 / avg if avg > 0 else float('inf')
        if max_rate < expected_rate * 0.9:
            print(f"\n⚠️  WARNING: Requested rate {expected_rate} Hz too high!")
            print(f"    CAN round-trip: ~{avg*1000:.1f} ms")
            print(f"    Max achievable: ~{max_rate:.1f} Hz\n")

    def _print_progress(self, expected_rate: float, rate_limited: bool) -> None:
        if self.ik_generators:
            progress = next(iter(self.ik_generators.values())).get_progress() * 100
        elif self.chirp_generators:
            progress = next(iter(self.chirp_generators.values())).get_progress() * 100
        else:
            progress = 0.0

        elapsed = time.perf_counter() - self.start_time
        intervals = self.sample_count - 1
        rate = intervals / elapsed if elapsed > 0 and intervals > 0 else expected_rate

        if rate_limited:
            print(f"Progress: {progress:.1f}% | Sample: {self.sample_count} | "
                  f"Rate: {rate:.1f}/{expected_rate:.0f} Hz")
        else:
            print(f"Progress: {progress:.1f}% | Sample: {self.sample_count} | Rate: {rate:.1f} Hz")

    def _print_summary(self, loop_times: list, expected_rate: float,
                       rate_limited: bool, missed_deadlines: int) -> None:
        elapsed = time.perf_counter() - self.start_time
        intervals = self.sample_count - 1
        actual_rate = intervals / elapsed if elapsed > 0 and intervals > 0 else 0
        avg_loop_ms = np.mean(loop_times) * 1000

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
        label = "mock loop" if self.dry_run else "CAN round-trip"
        print(f"Average loop time: {avg_loop_ms:.2f} ms ({label})")

        if rate_limited and missed_deadlines > 0:
            pct = (missed_deadlines / self.sample_count) * 100
            print(f"⚠️  Missed deadlines: {missed_deadlines}/{self.sample_count} ({pct:.1f}%)")

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

    def cleanup(self) -> None:
        """Stop all motors and close controller."""
        print("\nStopping motors...")
        try:
            self.controller.stop_all_motors()
            time.sleep(0.1)
        except Exception as e:
            print(f"Error stopping motors: {e}")
            self.controller.stop()

