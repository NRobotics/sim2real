"""
Result saving utilities for system identification.

Provides functions to save data in various formats:
- JSON: Full results with all metadata
- PyTorch: Tensors compatible with Isaac sim
- Plots: Matplotlib visualizations
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

# Handle both direct execution and module import
_script_dir = Path(__file__).resolve().parent
if str(_script_dir) not in sys.path:
    sys.path.insert(0, str(_script_dir))

if TYPE_CHECKING:
    from sysid import SystemIdentification


def save_json(sysid: SystemIdentification, output_file: str) -> None:
    """Save collected feedback data as JSON."""
    # Import FK here to avoid circular imports
    import sys
    parent = Path(__file__).resolve().parent.parent
    if str(parent) not in sys.path:
        sys.path.insert(0, str(parent))
    from kinematics import fk_motor_to_foot

    results = {
        "motor_ids": sysid.motor_ids,
        "config": sysid.config,
        "motor_configurations": {
            can_id: asdict(cfg) for can_id, cfg in sysid.motor_configs.items()
        },
        "feedback_data": {str(k): v for k, v in sysid.feedback_data.items()},
        "statistics": {
            "total_samples": sysid.sample_count,
            "duration": time.time() - sysid.start_time,
            "samples_per_motor": {
                str(can_id): len(data)
                for can_id, data in sysid.feedback_data.items()
            },
        },
    }

    # Add communication statistics if available
    if hasattr(sysid, "comm_stats") and sysid.comm_stats:
        results["communication_stats"] = sysid.comm_stats

    # Add IK group info with FK
    if sysid.ik_generators:
        results["ik_groups"] = {}
        for name, gen in sysid.ik_generators.items():
            motor_ids = _get_group_motor_ids(sysid, name)
            results["ik_groups"][name] = {
                "ik_type": gen.ik_type,
                "input_names": gen.input_names,
                "motor_ids": motor_ids,
            }

            # Compute FK for foot type
            if gen.ik_type == "foot" and len(motor_ids) == 2:
                fk_data = _compute_fk(sysid, motor_ids, fk_motor_to_foot)
                if fk_data:
                    results["ik_groups"][name]["fk_data"] = fk_data
                    print(f"  Computed {len(fk_data)} FK samples for '{name}'")

    if sysid.direct_motors:
        results["direct_motors"] = sorted(sysid.direct_motors)

    # Add interpolation data if available and non-empty
    if hasattr(sysid, "interpolation_data") and hasattr(sysid, "save_interpolation"):
        interp_data = {}
        for phase in ["start", "end"]:
            if sysid.save_interpolation.get(phase, False):
                phase_data = sysid.interpolation_data.get(phase, {})
                if phase_data:
                    interp_data[phase] = {str(k): list(v) for k, v in phase_data.items()}
        if interp_data:
            results["interpolation_data"] = interp_data

    with Path(output_file).open("w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_file}")
    print(f"Motors identified: {sysid.motor_ids}")
    if sysid.ik_generators:
        print(f"IK groups: {list(sysid.ik_generators.keys())}")
    if hasattr(sysid, "save_interpolation"):
        if sysid.save_interpolation.get("start", False):
            start_samples = sum(len(v) for v in sysid.interpolation_data.get("start", {}).values())
            print(f"Interpolation data (start): {start_samples} samples")
        if sysid.save_interpolation.get("end", False):
            end_samples = sum(len(v) for v in sysid.interpolation_data.get("end", {}).values())
            print(f"Interpolation data (end): {end_samples} samples")
    if sysid.direct_motors:
        print(f"Direct motors: {sorted(sysid.direct_motors)}")


def _get_group_motor_ids(sysid: SystemIdentification, group_name: str) -> list[int]:
    """Get motor IDs for an IK group, ordered by index."""
    ids = [(idx, mid) for mid, (g, idx) in sysid.motor_to_ik_group.items() if g == group_name]
    ids.sort()
    return [mid for _, mid in ids]


def _compute_fk(sysid: SystemIdentification, motor_ids: list[int], fk_func) -> list[dict]:
    """Compute FK from measured motor positions."""
    lower_data = sysid.feedback_data.get(motor_ids[0], [])
    upper_data = sysid.feedback_data.get(motor_ids[1], [])
    if not lower_data or not upper_data:
        return []

    print(f"\nComputing FK for motors {motor_ids}...")
    lower_by_sample = {d["sample"]: d for d in lower_data}
    upper_by_sample = {d["sample"]: d for d in upper_data}
    common = sorted(set(lower_by_sample.keys()) & set(upper_by_sample.keys()))

    fk_data = []
    for idx in common:
        q_lower = lower_by_sample[idx]["angle"]
        q_upper = upper_by_sample[idx]["angle"]
        pitch, roll = fk_func(q_lower, q_upper)
        fk_data.append({
            "sample": idx,
            "timestamp": lower_by_sample[idx]["timestamp"],
            "measured_pitch": pitch,
            "measured_roll": roll,
            "q_lower": q_lower,
            "q_upper": q_upper,
            "commanded_pitch": lower_by_sample[idx].get("commanded_pitch"),
            "commanded_roll": lower_by_sample[idx].get("commanded_roll"),
        })
    return fk_data


def save_torch(sysid: SystemIdentification, output_file: str) -> None:
    """Save data in PyTorch .pt format (Isaac sim compatible)."""
    try:
        import torch
    except ImportError:
        print("Warning: PyTorch not installed, skipping .pt save")
        return

    # Find common samples across all motors
    sample_sets = [
        {d["sample"] for d in sysid.feedback_data.get(can_id, [])}
        for can_id in sysid.motor_ids
        if sysid.feedback_data.get(can_id)
    ]
    if not sample_sets:
        print("Warning: No feedback data to save")
        return

    common_samples = sorted(set.intersection(*sample_sets))
    if not common_samples:
        print("Warning: No common samples across motors")
        return

    num_samples = len(common_samples)
    num_joints = len(sysid.motor_ids)

    # Build lookup
    lookups = {
        can_id: {d["sample"]: d for d in sysid.feedback_data.get(can_id, [])}
        for can_id in sysid.motor_ids
    }

    # Create tensors
    time_data = torch.zeros(num_samples)
    dof_pos = torch.zeros(num_samples, num_joints)
    des_dof_pos = torch.zeros(num_samples, num_joints)

    for i, sample_idx in enumerate(common_samples):
        time_data[i] = lookups[sysid.motor_ids[0]][sample_idx]["timestamp"]
        for j, can_id in enumerate(sysid.motor_ids):
            d = lookups[can_id][sample_idx]
            dof_pos[i, j] = d["angle"]
            des_dof_pos[i, j] = d["commanded_angle"]

    data = {
        "time": time_data,
        "dof_pos": dof_pos,
        "des_dof_pos": des_dof_pos,
        "joint_ids": sysid.motor_ids,
    }

    # Add IK metadata
    if sysid.motor_to_ik_group:
        data["joint_info"] = _build_joint_info(sysid)
        data["ik_groups"] = _build_ik_groups_info(sysid)

    # Add interpolation data if available
    if hasattr(sysid, "save_interpolation"):
        any_save = any(sysid.save_interpolation.get(p, False) for p in ["start", "end"])
        if any_save:
            interp_tensors = _build_interpolation_tensors(sysid, torch)
            if interp_tensors:
                data["interpolation"] = interp_tensors

    torch.save(data, output_file)
    print(f"Torch data saved to: {output_file}")
    print(f"  Shape: time={tuple(time_data.shape)}, dof_pos={tuple(dof_pos.shape)}")
    print(f"  Joint IDs (column order): {sysid.motor_ids}")
    if "interpolation" in data:
        for phase in ["start", "end"]:
            if phase in data["interpolation"]:
                shape = tuple(data["interpolation"][phase]["dof_pos"].shape)
                print(f"  Interpolation {phase}: {shape[0]} samples")


def _build_joint_info(sysid: SystemIdentification) -> list[dict]:
    """Build joint info for torch output."""
    joint_info = []
    for can_id in sysid.motor_ids:
        if can_id in sysid.motor_to_ik_group:
            group_name, idx = sysid.motor_to_ik_group[can_id]
            gen = sysid.ik_generators[group_name]
            joint_info.append({
                "motor_id": can_id,
                "type": "ik",
                "ik_group": group_name,
                "ik_type": gen.ik_type,
                "ik_index": idx,
                "role": "lower" if idx == 0 else "upper",
                "ik_inputs": gen.input_names,
            })
        else:
            joint_info.append({"motor_id": can_id, "type": "direct"})
    return joint_info


def _build_ik_groups_info(sysid: SystemIdentification) -> dict:
    """Build IK groups info for torch output."""
    ik_groups = {}
    for name, gen in sysid.ik_generators.items():
        motor_ids = _get_group_motor_ids(sysid, name)
        col_indices = [sysid.motor_ids.index(mid) for mid in motor_ids]
        ik_groups[name] = {
            "ik_type": gen.ik_type,
            "input_names": gen.input_names,
            "motor_ids": motor_ids,
            "column_indices": col_indices,
        }
    return ik_groups


def _build_interpolation_tensors(sysid: SystemIdentification, torch) -> dict:
    """Build torch tensors for interpolation phase data."""
    result = {}

    for phase in ["start", "end"]:
        # Skip if save not enabled for this phase
        if not sysid.save_interpolation.get(phase, False):
            continue

        phase_data = sysid.interpolation_data.get(phase, {})
        if not phase_data:
            continue

        # Find common samples across all motors for this phase
        sample_sets = [
            {d["sample"] for d in phase_data.get(can_id, [])}
            for can_id in sysid.motor_ids
            if phase_data.get(can_id)
        ]
        if not sample_sets:
            continue

        common_samples = sorted(set.intersection(*sample_sets))
        if not common_samples:
            continue

        num_samples = len(common_samples)
        num_joints = len(sysid.motor_ids)

        # Build lookup
        lookups = {
            can_id: {d["sample"]: d for d in phase_data.get(can_id, [])}
            for can_id in sysid.motor_ids
        }

        # Create tensors
        time_data = torch.zeros(num_samples)
        dof_pos = torch.zeros(num_samples, num_joints)
        des_dof_pos = torch.zeros(num_samples, num_joints)

        for i, sample_idx in enumerate(common_samples):
            first_motor = next(
                (mid for mid in sysid.motor_ids if sample_idx in lookups.get(mid, {})),
                sysid.motor_ids[0]
            )
            time_data[i] = lookups[first_motor][sample_idx]["time"]
            for j, can_id in enumerate(sysid.motor_ids):
                if sample_idx in lookups.get(can_id, {}):
                    d = lookups[can_id][sample_idx]
                    dof_pos[i, j] = d["angle"]
                    des_dof_pos[i, j] = d["commanded_angle"]

        result[phase] = {
            "time": time_data,
            "dof_pos": dof_pos,
            "des_dof_pos": des_dof_pos,
        }

    return result


def save_plots(sysid: SystemIdentification, output_dir: str) -> None:
    """Save plots for each motor and IK group."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Warning: matplotlib not installed, skipping plots")
        return

    # Import FK
    import sys
    parent = Path(__file__).resolve().parent.parent
    if str(parent) not in sys.path:
        sys.path.insert(0, str(parent))
    from kinematics import fk_motor_to_foot

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"\nSaving plots to {output_path}...")

    # Motor plots
    for can_id in sysid.motor_ids:
        data = sysid.feedback_data.get(can_id, [])
        if not data:
            continue

        times = [d["timestamp"] for d in data]
        positions = [d["angle"] for d in data]
        commanded = [d["commanded_angle"] for d in data]

        plt.figure(figsize=(12, 6))
        plt.plot(times, positions, label="Measured", linewidth=1)
        plt.plot(times, commanded, label="Commanded", linestyle='dashed', linewidth=1)
        plt.title(f"Motor {can_id} - Position Tracking")
        plt.xlabel("Time [s]")
        plt.ylabel("Position [rad]")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()

        plot_file = output_path / f"motor_{can_id}_{timestamp}.png"
        plt.savefig(plot_file, dpi=150)
        plt.close()
        print(f"  Saved: {plot_file.name}")

    # FK plots for IK groups
    for name, gen in sysid.ik_generators.items():
        if gen.ik_type != "foot":
            continue

        motor_ids = _get_group_motor_ids(sysid, name)
        if len(motor_ids) != 2:
            continue

        lower_data = sysid.feedback_data.get(motor_ids[0], [])
        upper_data = sysid.feedback_data.get(motor_ids[1], [])
        if not lower_data or not upper_data:
            continue

        lower_by_sample = {d["sample"]: d for d in lower_data}
        upper_by_sample = {d["sample"]: d for d in upper_data}
        common = sorted(set(lower_by_sample.keys()) & set(upper_by_sample.keys()))

        times, pitches, rolls = [], [], []
        cmd_pitches, cmd_rolls = [], []

        for idx in common:
            q_lower = lower_by_sample[idx]["angle"]
            q_upper = upper_by_sample[idx]["angle"]
            pitch, roll = fk_motor_to_foot(q_lower, q_upper)

            times.append(lower_by_sample[idx]["timestamp"])
            pitches.append(np.degrees(pitch))
            rolls.append(np.degrees(roll))
            cmd_p = lower_by_sample[idx].get("commanded_pitch")
            cmd_r = lower_by_sample[idx].get("commanded_roll")
            cmd_pitches.append(np.degrees(cmd_p) if cmd_p else 0.0)
            cmd_rolls.append(np.degrees(cmd_r) if cmd_r else 0.0)

        # Pitch plot
        plt.figure(figsize=(12, 6))
        plt.plot(times, pitches, label="Measured", linewidth=1)
        plt.plot(times, cmd_pitches, label="Commanded", linestyle='dashed', linewidth=1)
        plt.title(f"{name} - Pitch (FK)")
        plt.xlabel("Time [s]")
        plt.ylabel("Pitch [deg]")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plot_file = output_path / f"{name}_pitch_{timestamp}.png"
        plt.savefig(plot_file, dpi=150)
        plt.close()
        print(f"  Saved: {plot_file.name}")

        # Roll plot
        plt.figure(figsize=(12, 6))
        plt.plot(times, rolls, label="Measured", linewidth=1)
        plt.plot(times, cmd_rolls, label="Commanded", linestyle='dashed', linewidth=1)
        plt.title(f"{name} - Roll (FK)")
        plt.xlabel("Time [s]")
        plt.ylabel("Roll [deg]")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plot_file = output_path / f"{name}_roll_{timestamp}.png"
        plt.savefig(plot_file, dpi=150)
        plt.close()
        print(f"  Saved: {plot_file.name}")

    print(f"Plots saved to: {output_path}")

