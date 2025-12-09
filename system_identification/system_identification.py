#!/usr/bin/env python3
"""
System Identification Script for Robot Actuators
Sends chirp signals to motors and collects feedback data
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
import traceback
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

import numpy as np

NUM_MOTORS = 6

# Add the python directory to sys.path so Python can find humanoid_messages
# humanoid-protocol is in ext/humanoid-protocol relative to the project root
_python_dir = Path(__file__).parent.parent / "ext" / "humanoid-protocol" / "python"
sys.path.insert(0, str(_python_dir))
from humanoid_messages.can import (  # noqa: E402
    ConfigurationData,
    ControlData,
    FeedbackData,
    MotorCANController,
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
                qaxis_current_P=10.0,
                qaxis_current_I=0.1,
                qaxis_current_D=0.01,
                qaxis_current_limit=10.0,
                qaxis_lpf=1000.0,
                daxis_current_P=10.0,
                daxis_current_I=0.1,
                daxis_current_D=0.01,
                daxis_current_limit=10.0,
                daxis_lpf=1000.0,
                bypass_control_loop=False,
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
    """Generate chirp (frequency sweep) signals for system identification"""

    def __init__(
        self,
        f_start: float,
        f_end: float,
        duration: float,
        amplitude: float,
        sample_rate: float,
        sweep_type: str = "linear",
    ):
        self.f_start = f_start
        self.f_end = f_end
        self.duration = duration
        self.sample_rate = sample_rate
        self.sweep_type = sweep_type

        # Scale amplitude: 1.0 = -2π to +2π range
        self.amplitude = amplitude * 2 * np.pi

        # Pre-generate the chirp signal
        self.num_samples = int(duration * sample_rate)
        self.time = np.linspace(0, duration, self.num_samples)
        self.signal = self._generate_chirp()

        print(self.signal)
        self.current_index = 0

    def _generate_chirp(self) -> np.ndarray:
        """Generate chirp signal based on sweep type"""
        if self.sweep_type == "linear":
            # Linear frequency sweep
            phase = (
                2
                * np.pi
                * (self.f_start * self.time + (self.f_end - self.f_start) * self.time**2 / (2 * self.duration))
            )
        elif self.sweep_type == "logarithmic":
            # Logarithmic frequency sweep
            k = (self.f_end / self.f_start) ** (1 / self.duration)
            phase = 2 * np.pi * self.f_start * (k**self.time - 1) / np.log(k)
        elif self.sweep_type == "exponential":
            # Exponential frequency sweep
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

        return self.amplitude * np.sin(phase)

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


class SystemIdentification:
    def __init__(self, config_file: str, dry_run: bool = False):
        self.config = self._load_config(config_file)
        self.dry_run = dry_run

        if dry_run:
            print("[DRY-RUN] Running without hardware - using mock controller")
            self.controller = MockMotorCANController(**self.config["can_interface"])
        else:
            self.controller = MotorCANController(**self.config["can_interface"])

        # Storage for motor configurations
        self.motor_configs: dict[int, ConfigurationData] = {}
        self.config_received = threading.Event()
        self.configs_to_receive = set(range(NUM_MOTORS))

        # Chirp generators for each motor
        self.chirp_generators: dict[int, ChirpGenerator] = {}

        # Feedback data collection
        self.feedback_data: dict[int, list[dict]] = defaultdict(list)
        self.feedback_received = threading.Event()
        self.feedbacks_to_receive: set[int] = set()

        # Commanded angle tracking (for comparison with feedback)
        self.commanded_angles: dict[int, float] = {}

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

        self.feedback_data[can_id].append(feedback_dict)

        if can_id in self.feedbacks_to_receive:
            self.feedbacks_to_receive.remove(can_id)
        if not self.feedbacks_to_receive:
            self.feedback_received.set()

    def setup(self) -> None:
        """Initialize CAN controller and read motor configurations"""
        print(f"Starting CAN controller for {NUM_MOTORS} motors...")
        self.controller.start()
        time.sleep(0.1)

        # Register callbacks
        for can_id in range(NUM_MOTORS):
            self.controller.set_config_callback(can_id, self._config_callback)
            self.controller.set_feedback_callback(can_id, self._feedback_callback)

        # Request configurations from all motors
        print("Requesting configurations from all motors...")
        for can_id in range(NUM_MOTORS):
            self.controller.get_motor_configuration(can_id)

        # Wait for all configurations
        if not self.config_received.wait(timeout=2.0):
            missing = self.configs_to_receive
            print(f"Warning: Did not receive configs from motors: {missing}")

        print(f"Received configurations from {len(self.motor_configs)} motors")

        # Initialize chirp generators for each motor
        chirp_config = self.config["chirp"]
        for can_id, motor_config in self.config["motors"].items():
            can_id_int = int(can_id)

            # Skip motors that are beyond NUM_MOTORS
            if can_id_int >= NUM_MOTORS:
                continue

            # Apply motor-specific modifications
            amplitude = chirp_config["amplitude"] * motor_config.get("amplitude_scale", 1.0)
            offset = motor_config.get("offset", 0.0)

            self.chirp_generators[can_id_int] = ChirpGenerator(
                f_start=chirp_config["f_start"],
                f_end=chirp_config["f_end"],
                duration=chirp_config["duration"],
                amplitude=amplitude,
                sample_rate=chirp_config["sample_rate"],
                sweep_type=chirp_config.get("sweep_type", "logarithmic"),
            )

            # Store offset for later use
            motor_config["_offset"] = offset

        print(f"Chirp generators initialized for {len(self.chirp_generators)} motors")

    def run_identification(self) -> None:
        """Main identification loop"""
        control_params = self.config["control_parameters"]
        chirp_config = self.config["chirp"]
        expected_rate = chirp_config["sample_rate"]

        print("\nStarting system identification...")
        print(f"Expected rate: {expected_rate} Hz")
        print(f"Duration: {chirp_config['duration']} seconds")
        print(f"Frequency sweep: {chirp_config['f_start']} - {chirp_config['f_end']} Hz")
        print(f"Number of motors: {NUM_MOTORS}")

        for i in range(NUM_MOTORS):
            self.controller.start_motor(i)
            time.sleep(0.125)

        print("Motors started")
        time.sleep(1.0)

        self.start_time = time.time()
        self.sample_count = 0
        is_complete = False

        loop_times = []

        while not is_complete:
            loop_start = time.time()

            # Generate control data for each motor
            control_data = {}
            for can_id in range(NUM_MOTORS):
                chirp_value, complete = self.chirp_generators[can_id].get_next()

                if complete:
                    is_complete = True

                # Add motor-specific offset
                angle = chirp_value + self.config["motors"][str(can_id)]["_offset"]

                # Store commanded angle for later comparison with feedback
                self.commanded_angles[can_id] = angle

                control_data[can_id] = ControlData(
                    angle=angle,
                    velocity=control_params["velocity"],
                    effort=control_params["effort"],
                    stiffness=control_params["stiffness"],
                    damping=control_params["damping"],
                )

            # Send control commands to all motors
            self.feedbacks_to_receive = set(range(NUM_MOTORS))
            self.feedback_received.clear()

            for can_id, ctrl_data in control_data.items():
                self.controller.send_kinematics_for_motor(can_id, ctrl_data)

            # Wait for all feedback (with timeout)
            if not self.feedback_received.wait(timeout=0.1):
                # Timeout - some motors didn't respond
                pass

            self.sample_count += 1

            # Calculate loop timing
            loop_time = time.time() - loop_start
            loop_times.append(loop_time)

            # Print progress
            if self.sample_count % 100 == 0:
                progress = self.chirp_generators[0].get_progress() * 100
                avg_loop_time = np.mean(loop_times[-100:])
                actual_rate = 1.0 / avg_loop_time if avg_loop_time > 0 else 0
                print(
                    f"Progress: {progress:.1f}% | Sample: {self.sample_count} | "
                    f"Rate: {actual_rate:.1f} Hz | Loop time: {avg_loop_time * 1000:.2f} ms"
                )

        elapsed = time.time() - self.start_time
        actual_rate = self.sample_count / elapsed if elapsed > 0 else 0

        print("\nIdentification complete!")
        print(f"Samples: {self.sample_count}")
        print(f"Duration: {elapsed:.2f} seconds")
        print(f"Average rate: {actual_rate:.1f} Hz")
        print(f"Average loop time: {np.mean(loop_times) * 1000:.2f} ms")

    def save_results(self, output_file: str) -> None:
        """Save collected feedback data as JSON"""
        results = {
            "config": self.config,
            "motor_configurations": {can_id: asdict(config) for can_id, config in self.motor_configs.items()},
            "feedback_data": self.feedback_data,
            "statistics": {
                "total_samples": self.sample_count,
                "duration": time.time() - self.start_time,
                "samples_per_motor": {can_id: len(data) for can_id, data in self.feedback_data.items()},
            },
        }

        with Path(output_file).open("w") as f:
            json.dump(results, f, indent=2)

        print(f"\nResults saved to: {output_file}")

    def cleanup(self) -> None:
        """Stop all motors and close CAN controller"""
        print("\nStopping motors...")
        try:
            self.controller.stop_all_motors()
            time.sleep(0.1)
        except Exception as e:
            print(f"Error stopping motors: {e}")

        self.controller.stop()


def main() -> None:
    parser = argparse.ArgumentParser(description="System identification for robot actuators")
    parser.add_argument(
        "--config",
        type=str,
        default="config.json",
        help="Configuration file (JSON)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="sysid_results.json",
        help="Output file for results (JSON)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run without hardware (simulated motors for testing)",
    )

    args = parser.parse_args()

    if not Path(args.config).exists():
        print(f"Error: Config file not found: {args.config}")
        print("Creating example config file...")
        create_example_config(args.config)
        return

    sysid = SystemIdentification(args.config, dry_run=args.dry_run)

    try:
        sysid.setup()
        sysid.run_identification()
        sysid.save_results(args.output)
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"\nError: {e}")
        traceback.print_exc()
    finally:
        sysid.cleanup()


def create_example_config(filename: str) -> None:
    """Create example configuration file"""
    example_config = {
        "can_interface": {
            "interface": "socketcan",
            "channel": "can0",
            "bitrate": 5000000,
            "fd": True,
        },
        "chirp": {
            "f_start": 0.1,
            "f_end": 20.0,
            "duration": 10.0,
            "amplitude": 0.5,
            "sample_rate": 650.0,
            "sweep_type": "logarithmic",
        },
        "control_parameters": {
            "velocity": 0.0,
            "effort": 0.0,
            "stiffness": 1.0,
            "damping": 0.0,
        },
        "motors": {str(i): {"amplitude_scale": 1.0, "offset": 0.0} for i in range(NUM_MOTORS)},
    }

    with Path(filename).open("w") as f:
        json.dump(example_config, f, indent=2)

    print(f"Example config created: {filename}")
    print("Edit this file and run again.")


if __name__ == "__main__":
    main()
