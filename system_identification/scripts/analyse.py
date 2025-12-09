#!/usr/bin/env python3
"""
Analyze and visualize system identification results
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_results(filename: str) -> dict:
    """Load results from JSON file"""
    with Path(filename).open() as f:
        return json.load(f)


def plot_motor_response(results: dict, motor_id: int, save_path: str | None = None):  # noqa: PLR0915
    """Plot response for a single motor"""
    if str(motor_id) not in results["feedback_data"]:
        print(f"No data for motor {motor_id}")
        return

    data = results["feedback_data"][str(motor_id)]

    # Extract time series
    timestamps = [d["timestamp"] for d in data]
    angles = [d["angle"] for d in data]
    commanded_angles = [d.get("commanded_angle", 0.0) for d in data]
    angle_errors = [d.get("angle_error", 0.0) for d in data]
    velocities = [d["velocity"] for d in data]
    efforts = [d["effort"] for d in data]
    voltages = [d["voltage"] for d in data]
    temp_motor = [d["temp_motor"] for d in data]
    temp_pcb = [d["temp_pcb"] for d in data]

    # Create figure with subplots
    fig, axes = plt.subplots(4, 2, figsize=(14, 12))
    fig.suptitle(f"Motor {motor_id} System Identification Results", fontsize=16)

    # Commanded vs Actual Angle
    axes[0, 0].plot(timestamps, commanded_angles, "r--", linewidth=1, label="Commanded", alpha=0.7)
    axes[0, 0].plot(timestamps, angles, "b-", linewidth=0.5, label="Actual")
    axes[0, 0].set_ylabel("Angle (rad)")
    axes[0, 0].set_xlabel("Time (s)")
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()
    axes[0, 0].set_title("Position Tracking")

    # Angle Tracking Error
    axes[0, 1].plot(timestamps, angle_errors, "m-", linewidth=0.5)
    axes[0, 1].set_ylabel("Tracking Error (rad)")
    axes[0, 1].set_xlabel("Time (s)")
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_title(f"Tracking Error (RMS: {np.sqrt(np.mean(np.array(angle_errors) ** 2)):.4f} rad)")

    # Velocity
    axes[1, 0].plot(timestamps, velocities, "g-", linewidth=0.5)
    axes[1, 0].set_ylabel("Velocity (rad/s)")
    axes[1, 0].set_xlabel("Time (s)")
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_title("Velocity Response")

    # Effort
    axes[1, 1].plot(timestamps, efforts, "r-", linewidth=0.5)
    axes[1, 1].set_ylabel("Effort (A or Nm)")
    axes[1, 1].set_xlabel("Time (s)")
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_title("Effort Response")

    # Voltage
    axes[2, 0].plot(timestamps, voltages, "m-", linewidth=0.5)
    axes[2, 0].set_ylabel("Voltage (V)")
    axes[2, 0].set_xlabel("Time (s)")
    axes[2, 0].grid(True, alpha=0.3)
    axes[2, 0].set_title("Bus Voltage")

    # Temperatures
    axes[2, 1].plot(timestamps, temp_motor, "orange", linewidth=1, label="Motor")
    axes[2, 1].plot(timestamps, temp_pcb, "cyan", linewidth=1, label="PCB")
    axes[2, 1].set_ylabel("Temperature (°C)")
    axes[2, 1].set_xlabel("Time (s)")
    axes[2, 1].grid(True, alpha=0.3)
    axes[2, 1].legend()
    axes[2, 1].set_title("Temperatures")

    # Commanded Angle vs Actual Angle (XY plot)
    axes[3, 0].plot(commanded_angles, angles, "k.", markersize=1, alpha=0.3)
    axes[3, 0].plot([-1, 1], [-1, 1], "r--", linewidth=1, label="Perfect tracking")
    axes[3, 0].set_xlabel("Commanded Angle (rad)")
    axes[3, 0].set_ylabel("Actual Angle (rad)")
    axes[3, 0].grid(True, alpha=0.3)
    axes[3, 0].legend()
    axes[3, 0].set_title("Commanded vs Actual")
    axes[3, 0].axis("equal")

    # Angle vs Effort (hysteresis)
    axes[3, 1].plot(angles, efforts, "k-", linewidth=0.5, alpha=0.5)
    axes[3, 1].set_xlabel("Angle (rad)")
    axes[3, 1].set_ylabel("Effort (A or Nm)")
    axes[3, 1].grid(True, alpha=0.3)
    axes[3, 1].set_title("Angle vs Effort (Hysteresis)")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot: {save_path}")
    else:
        plt.show()


def plot_all_motors(results: dict, save_path: str | None = None):
    """Plot comparison of all motors"""
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle("All Motors: Commanded vs Actual Tracking", fontsize=16)

    axes = axes.flatten()

    for motor_id in range(6):
        if str(motor_id) not in results["feedback_data"]:
            continue

        data = results["feedback_data"][str(motor_id)]
        timestamps = [d["timestamp"] for d in data]
        angles = [d["angle"] for d in data]
        commanded_angles = [d.get("commanded_angle", 0.0) for d in data]
        angle_errors = [d.get("angle_error", 0.0) for d in data]

        ax = axes[motor_id]

        # Plot commanded vs actual
        ax.plot(timestamps, commanded_angles, "r--", linewidth=1, label="Commanded", alpha=0.7)
        ax.plot(timestamps, angles, "b-", linewidth=0.5, label="Actual")

        # Calculate RMS error
        rms_error = np.sqrt(np.mean(np.array(angle_errors) ** 2))

        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Angle (rad)")
        ax.set_title(f"Motor {motor_id} (RMS error: {rms_error:.4f} rad)")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=8)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot: {save_path}")
    else:
        plt.show()


def compute_statistics(results: dict):
    """Compute and print statistics"""
    print("\n=== System Identification Statistics ===\n")

    stats = results.get("statistics", {})
    print(f"Total samples: {stats.get('total_samples', 'N/A')}")
    print(f"Duration: {stats.get('duration', 0):.2f} seconds")

    if "samples_per_motor" in stats:
        print("\nSamples per motor:")
        for motor_id, count in stats["samples_per_motor"].items():
            print(f"  Motor {motor_id}: {count}")

    # Compute per-motor statistics
    print("\n=== Per-Motor Analysis ===\n")
    for motor_id in range(6):
        if str(motor_id) not in results["feedback_data"]:
            continue

        data = results["feedback_data"][str(motor_id)]
        angles = np.array([d["angle"] for d in data])
        commanded_angles = np.array([d.get("commanded_angle", 0.0) for d in data])
        angle_errors = np.array([d.get("angle_error", 0.0) for d in data])
        velocities = np.array([d["velocity"] for d in data])
        efforts = np.array([d["effort"] for d in data])
        temps_motor = np.array([d["temp_motor"] for d in data])

        print(f"Motor {motor_id}:")
        print(f"  Commanded angle range: [{commanded_angles.min():.3f}, {commanded_angles.max():.3f}] rad")
        print(f"  Actual angle range: [{angles.min():.3f}, {angles.max():.3f}] rad")
        print(f"  Tracking error (RMS): {np.sqrt(np.mean(angle_errors**2)):.4f} rad")
        print(f"  Tracking error (max): {np.abs(angle_errors).max():.4f} rad")
        print(f"  Velocity range: [{velocities.min():.3f}, {velocities.max():.3f}] rad/s")
        print(f"  Effort range: [{efforts.min():.3f}, {efforts.max():.3f}]")
        print(f"  Max motor temp: {temps_motor.max():.1f}°C")
        print()


def main():
    parser = argparse.ArgumentParser(description="Analyze system identification results")
    parser.add_argument("results_file", type=str, help="Results JSON file from system identification")
    parser.add_argument("--motor", type=int, help="Plot specific motor (0-5)", default=None)
    parser.add_argument("--all", action="store_true", help="Plot all motors comparison")
    parser.add_argument("--save", type=str, help="Save plot to file", default=None)
    parser.add_argument("--stats", action="store_true", help="Print statistics only")

    args = parser.parse_args()

    if not Path(args.results_file).exists():
        print(f"Error: Results file not found: {args.results_file}")
        return

    print(f"Loading results from: {args.results_file}")
    results = load_results(args.results_file)

    # Print statistics
    compute_statistics(results)

    if args.stats:
        return

    # Generate plots
    if args.motor is not None:
        if 0 <= args.motor < 6:
            plot_motor_response(results, args.motor, args.save)
        else:
            print(f"Error: Motor ID must be 0-5, got {args.motor}")

    elif args.all:
        plot_all_motors(results, args.save)

    else:
        # Default: plot all motors
        plot_all_motors(results, args.save)


if __name__ == "__main__":
    main()
