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
    """Save data in PyTorch .pt format.
    
    Saves TWO files:
    - results.pt: Chirp phase only (for system ID)
    - results_full.pt: All phases combined (start_interp + chirp + end_interp)
    
    Missing feedbacks are marked with NaN and tracked in 'valid_mask'.
    """
    try:
        import torch
    except ImportError:
        print("Warning: PyTorch not installed, skipping .pt save")
        return

    output_path = Path(output_file)
    num_joints = len(sysid.motor_ids)

    # === Save chirp-only data ===
    num_samples = sysid.sample_count
    chirp_data = None
    if num_samples > 0:
        chirp_data = _build_phase_tensors(
            sysid, sysid.feedback_data, num_samples, "timestamp", torch
        )
        chirp_data["joint_ids"] = sysid.motor_ids
        chirp_data["phase"] = "chirp"
        if sysid.motor_to_ik_group:
            chirp_data["joint_info"] = _build_joint_info(sysid)
            chirp_data["ik_groups"] = _build_ik_groups_info(sysid)

        torch.save(chirp_data, output_path)
        _print_torch_summary("Chirp data", output_path, chirp_data, sysid.motor_ids)

    # === Save full data (all phases combined) ===
    full_path = output_path.with_name(
        output_path.stem + "_full" + output_path.suffix
    )
    
    # Collect all phases data
    all_tensors: dict[str, list] = {
        "time": [], "dof_pos": [], "des_dof_pos": [], "dof_vel": [], "dof_effort": [],
        "dof_voltage": [], "dof_temp_motor": [], "dof_temp_pcb": [], "dof_flags": [],
        "valid_mask": [], "phase_id": [],
    }
    phase_boundaries = {}
    current_idx = 0

    def _append_phase(tensors: dict, n: int, phase_id: int) -> None:
        nonlocal current_idx
        for key in all_tensors:
            if key == "phase_id":
                all_tensors[key].append(torch.full((n,), phase_id, dtype=torch.int8))
            elif key in tensors:
                all_tensors[key].append(tensors[key])

    # Start interpolation
    if hasattr(sysid, "save_interpolation") and sysid.save_interpolation.get("start"):
        start_data = sysid.interpolation_data.get("start", {})
        if start_data:
            n = _get_phase_sample_count(start_data)
            tensors = _build_phase_tensors(sysid, start_data, n, "timestamp", torch)
            _append_phase(tensors, n, 0)  # 0 = start
            phase_boundaries["start_interp"] = (current_idx, current_idx + n)
            current_idx += n

    # Chirp phase
    if num_samples > 0:
        _append_phase(chirp_data, num_samples, 1)  # 1 = chirp
        phase_boundaries["chirp"] = (current_idx, current_idx + num_samples)
        current_idx += num_samples

    # End interpolation
    if hasattr(sysid, "save_interpolation") and sysid.save_interpolation.get("end"):
        end_data = sysid.interpolation_data.get("end", {})
        if end_data:
            n = _get_phase_sample_count(end_data)
            tensors = _build_phase_tensors(sysid, end_data, n, "timestamp", torch)
            _append_phase(tensors, n, 2)  # 2 = end
            phase_boundaries["end_interp"] = (current_idx, current_idx + n)
            current_idx += n

    if all_tensors["time"]:
        full_data = {
            key: torch.cat(vals) for key, vals in all_tensors.items() if vals
        }
        full_data["phase_boundaries"] = phase_boundaries
        full_data["joint_ids"] = sysid.motor_ids
        if sysid.motor_to_ik_group:
            full_data["joint_info"] = _build_joint_info(sysid)
            full_data["ik_groups"] = _build_ik_groups_info(sysid)

        torch.save(full_data, full_path)
        _print_torch_summary("Full data", full_path, full_data, sysid.motor_ids)


def _get_phase_sample_count(phase_data: dict) -> int:
    """Get sample count from phase data dict."""
    all_samples = set()
    for data_list in phase_data.values():
        for d in data_list:
            all_samples.add(d["sample"])
    return max(all_samples) + 1 if all_samples else 0


def _build_phase_tensors(
    sysid, data_dict: dict, num_samples: int, time_key: str, torch
) -> dict:
    """Build tensors for a single phase with all feedback data."""
    num_joints = len(sysid.motor_ids)
    
    lookups = {
        can_id: {d["sample"]: d for d in data_dict.get(can_id, [])}
        for can_id in sysid.motor_ids
    }

    # Core tensors
    time_data = torch.full((num_samples,), float('nan'))
    dof_pos = torch.full((num_samples, num_joints), float('nan'))
    des_dof_pos = torch.full((num_samples, num_joints), float('nan'))
    valid_mask = torch.zeros(num_samples, num_joints, dtype=torch.bool)
    
    # Additional feedback data
    dof_vel = torch.full((num_samples, num_joints), float('nan'))
    dof_effort = torch.full((num_samples, num_joints), float('nan'))
    dof_voltage = torch.full((num_samples, num_joints), float('nan'))
    dof_temp_motor = torch.full((num_samples, num_joints), float('nan'))
    dof_temp_pcb = torch.full((num_samples, num_joints), float('nan'))
    dof_flags = torch.zeros(num_samples, num_joints, dtype=torch.int32)

    for i in range(num_samples):
        for can_id in sysid.motor_ids:
            if i in lookups[can_id]:
                time_data[i] = lookups[can_id][i][time_key]
                break

        for j, can_id in enumerate(sysid.motor_ids):
            if i in lookups[can_id]:
                d = lookups[can_id][i]
                dof_pos[i, j] = d["angle"]
                des_dof_pos[i, j] = d["commanded_angle"]
                dof_vel[i, j] = d.get("velocity", float('nan'))
                dof_effort[i, j] = d.get("effort", float('nan'))
                dof_voltage[i, j] = d.get("voltage", float('nan'))
                dof_temp_motor[i, j] = d.get("temp_motor", float('nan'))
                dof_temp_pcb[i, j] = d.get("temp_pcb", float('nan'))
                dof_flags[i, j] = d.get("flags", 0)
                valid_mask[i, j] = True

    return {
        "time": time_data,
        "dof_pos": dof_pos,
        "des_dof_pos": des_dof_pos,
        "dof_vel": dof_vel,
        "dof_effort": dof_effort,
        "dof_voltage": dof_voltage,
        "dof_temp_motor": dof_temp_motor,
        "dof_temp_pcb": dof_temp_pcb,
        "dof_flags": dof_flags,
        "valid_mask": valid_mask,
        "num_expected_samples": num_samples,
    }


def _print_torch_summary(name: str, path: Path, data: dict, motor_ids: list):
    """Print summary of saved torch data."""
    print(f"\n{name} saved to: {path}")
    print(f"  Shape: {tuple(data['dof_pos'].shape)}")
    print(f"  Joint IDs: {motor_ids}")
    valid_pct = data["valid_mask"].float().mean().item() * 100
    print(f"  Valid: {valid_pct:.1f}%")
    if "phase_boundaries" in data:
        for phase, (start, end) in data["phase_boundaries"].items():
            print(f"  {phase}: samples {start}-{end} ({end-start} samples)")


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


def save_plots(sysid: SystemIdentification, output_dir: str) -> None:
    """Save plots for each motor and IK group, including all phases."""
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

    # Collect data from all phases for each motor
    has_interp = hasattr(sysid, "interpolation_data")

    for can_id in sysid.motor_ids:
        plt.figure(figsize=(14, 6))
        time_offset = 0.0
        chirp_end = 0.0

        # Get y-limits from config
        limits = sysid.motor_limits.get(can_id)
        if limits:
            margin = (limits[1] - limits[0]) * 0.1
            ylim = (limits[0] - margin, limits[1] + margin)
        else:
            ylim = None

        # Start interpolation (if available)
        if has_interp and sysid.save_interpolation.get("start"):
            start_data = sysid.interpolation_data.get("start", {}).get(can_id, [])
            if start_data:
                # Filter out invalid timestamps (safety check)
                max_valid_ts = 60.0
                valid_data = [d for d in start_data if d["timestamp"] < max_valid_ts]
                if valid_data:
                    t = [d["timestamp"] for d in valid_data]
                    pos = [d["angle"] for d in valid_data]
                    cmd = [d["commanded_angle"] for d in valid_data]
                    plt.axvspan(min(t), max(t), alpha=0.1, color='blue')
                    plt.plot(t, pos, 'b-', linewidth=0.8, alpha=0.7)
                    plt.plot(t, cmd, 'b--', linewidth=0.8, alpha=0.7)
                    time_offset = max(t)

        # Chirp phase (main data)
        chirp_data = sysid.feedback_data.get(can_id, [])
        if chirp_data:
            t = [d["timestamp"] + time_offset for d in chirp_data]
            pos = [d["angle"] for d in chirp_data]
            cmd = [d["commanded_angle"] for d in chirp_data]
            plt.plot(t, pos, 'g-', linewidth=1, label="Measured")
            plt.plot(t, cmd, 'r--', linewidth=1, label="Commanded")
            chirp_end = max(t) if t else time_offset

        # End interpolation (if available)
        if has_interp and sysid.save_interpolation.get("end"):
            end_data = sysid.interpolation_data.get("end", {}).get(can_id, [])
            if end_data:
                # Filter out invalid timestamps (safety check for race condition)
                max_valid_ts = 60.0  # Max expected interp duration
                valid_data = [d for d in end_data if d["timestamp"] < max_valid_ts]
                if valid_data:
                    t = [d["timestamp"] + chirp_end for d in valid_data]
                    pos = [d["angle"] for d in valid_data]
                    cmd = [d["commanded_angle"] for d in valid_data]
                    plt.axvspan(min(t), max(t), alpha=0.1, color='orange')
                    plt.plot(t, pos, color='darkorange', linewidth=0.8, alpha=0.7)
                    plt.plot(t, cmd, color='darkorange', linestyle='--',
                             linewidth=0.8, alpha=0.7)

        plt.title(f"Motor {can_id} - Position Tracking (all phases)")
        plt.xlabel("Time [s]")
        plt.ylabel("Position [rad]")
        if ylim:
            plt.ylim(ylim)
        plt.grid(True, alpha=0.3)
        plt.legend(loc='upper right')
        plt.tight_layout()

        plot_file = output_path / f"motor_{can_id}_{timestamp}.png"
        plt.savefig(plot_file, dpi=150)
        plt.close()
        print(f"  Saved: {plot_file.name}")

    # Additional feedback plots (velocity, effort, temperature) - all motors combined
    _save_feedback_plots(sysid, output_path, timestamp, has_interp)

    # FK plots for IK groups (all phases combined)
    for name, gen in sysid.ik_generators.items():
        if gen.ik_type != "foot":
            continue

        motor_ids = _get_group_motor_ids(sysid, name)
        if len(motor_ids) != 2:
            continue

        # Collect FK data from all phases
        all_times, all_pitches, all_rolls = [], [], []
        all_cmd_p, all_cmd_r = [], []
        phase_markers = []  # (start_idx, end_idx, phase_name)

        time_offset = 0.0
        current_idx = 0

        # Start interpolation
        if has_interp and sysid.save_interpolation.get("start"):
            t, p, r, cp, cr = _compute_fk_from_phase(
                sysid.interpolation_data.get("start", {}),
                motor_ids, fk_motor_to_foot, "timestamp"
            )
            if t:
                all_times.extend([x + time_offset for x in t])
                all_pitches.extend(p)
                all_rolls.extend(r)
                all_cmd_p.extend(cp)
                all_cmd_r.extend(cr)
                phase_markers.append((current_idx, current_idx + len(t), "start"))
                current_idx += len(t)
                time_offset = max(t) if t else 0.0

        # Chirp phase
        t, p, r, cp, cr = _compute_fk_from_phase(
            sysid.feedback_data, motor_ids, fk_motor_to_foot, "timestamp"
        )
        if t:
            all_times.extend([x + time_offset for x in t])
            all_pitches.extend(p)
            all_rolls.extend(r)
            all_cmd_p.extend(cp)
            all_cmd_r.extend(cr)
            phase_markers.append((current_idx, current_idx + len(t), "chirp"))
            current_idx += len(t)
            chirp_end = max(t) + time_offset if t else time_offset
        else:
            chirp_end = time_offset

        # End interpolation
        if has_interp and sysid.save_interpolation.get("end"):
            t, p, r, cp, cr = _compute_fk_from_phase(
                sysid.interpolation_data.get("end", {}),
                motor_ids, fk_motor_to_foot, "timestamp"
            )
            if t:
                all_times.extend([x + chirp_end for x in t])
                all_pitches.extend(p)
                all_rolls.extend(r)
                all_cmd_p.extend(cp)
                all_cmd_r.extend(cr)
                phase_markers.append((current_idx, current_idx + len(t), "end"))

        if not all_times:
            continue

        # Get IK limits from chirp config (scale determines amplitude)
        ik_config = None
        for grp in sysid.config.get("ik_groups", []):
            if grp.get("name") == name:
                ik_config = grp.get("chirp", {})
                break
        
        pitch_scale = ik_config.get("scale_pitch", 0.3) if ik_config else 0.3
        roll_scale = ik_config.get("scale_roll", 0.3) if ik_config else 0.3
        pitch_margin = pitch_scale * 0.2
        roll_margin = roll_scale * 0.2

        # Pitch plot (in radians)
        plt.figure(figsize=(14, 6))
        for start, end, phase in phase_markers:
            color = {'start': 'blue', 'chirp': 'green', 'end': 'orange'}[phase]
            alpha = 0.7 if phase != 'chirp' else 1.0
            plt.plot(all_times[start:end], all_pitches[start:end],
                     color=color, linewidth=0.8, alpha=alpha)
            plt.plot(all_times[start:end], all_cmd_p[start:end],
                     color=color, linestyle='--', linewidth=0.8, alpha=alpha)
        plt.title(f"{name} - Pitch (FK, all phases)")
        plt.xlabel("Time [s]")
        plt.ylabel("Pitch [rad]")
        plt.ylim(-pitch_scale - pitch_margin, pitch_scale + pitch_margin)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plot_file = output_path / f"{name}_pitch_{timestamp}.png"
        plt.savefig(plot_file, dpi=150)
        plt.close()
        print(f"  Saved: {plot_file.name}")

        # Roll plot (in radians)
        plt.figure(figsize=(14, 6))
        for start, end, phase in phase_markers:
            color = {'start': 'blue', 'chirp': 'green', 'end': 'orange'}[phase]
            alpha = 0.7 if phase != 'chirp' else 1.0
            plt.plot(all_times[start:end], all_rolls[start:end],
                     color=color, linewidth=0.8, alpha=alpha)
            plt.plot(all_times[start:end], all_cmd_r[start:end],
                     color=color, linestyle='--', linewidth=0.8, alpha=alpha)
        plt.title(f"{name} - Roll (FK, all phases)")
        plt.xlabel("Time [s]")
        plt.ylabel("Roll [rad]")
        plt.ylim(-roll_scale - roll_margin, roll_scale + roll_margin)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plot_file = output_path / f"{name}_roll_{timestamp}.png"
        plt.savefig(plot_file, dpi=150)
        plt.close()
        print(f"  Saved: {plot_file.name}")

    print(f"Plots saved to: {output_path}")


def _save_feedback_plots(
    sysid, output_path: Path, timestamp: str, has_interp: bool
) -> None:
    """Save velocity, effort, voltage, and temperature plots for all motors."""
    import matplotlib.pyplot as plt

    # Collect data from all phases for each motor
    def collect_phase_data(motor_data: dict, field: str) -> tuple[list, list, str]:
        """Collect time and field data, return (times, values, color)."""
        times, values = [], []
        for d in motor_data:
            if d.get("timestamp", 0) < 60.0:  # Filter invalid timestamps
                times.append(d["timestamp"])
                values.append(d.get(field, float('nan')))
        return times, values

    # Define plot configs: (field_name, ylabel, title_suffix)
    plot_configs = [
        ("velocity", "Velocity [rad/s]", "Velocity"),
        ("effort", "Effort [A]", "Effort (Current)"),
        ("voltage", "Voltage [V]", "Bus Voltage"),
        ("temp_motor", "Temperature [°C]", "Motor Temperature"),
        ("temp_pcb", "Temperature [°C]", "PCB Temperature"),
    ]

    for field, ylabel, title in plot_configs:
        plt.figure(figsize=(14, 6))
        time_offset = 0.0
        chirp_end = 0.0
        colors = plt.cm.tab10.colors

        for idx, can_id in enumerate(sysid.motor_ids):
            color = colors[idx % len(colors)]
            motor_times, motor_values = [], []

            # Start interpolation
            if has_interp and sysid.save_interpolation.get("start"):
                start_data = sysid.interpolation_data.get("start", {}).get(can_id, [])
                if start_data:
                    t, v = collect_phase_data(start_data, field)
                    if t:
                        motor_times.extend(t)
                        motor_values.extend(v)
                        time_offset = max(t) if t else 0.0

            # Chirp phase
            chirp_data = sysid.feedback_data.get(can_id, [])
            if chirp_data:
                t, v = collect_phase_data(chirp_data, field)
                if t:
                    motor_times.extend([x + time_offset for x in t])
                    motor_values.extend(v)
                    chirp_end = max(t) + time_offset if t else time_offset

            # End interpolation
            if has_interp and sysid.save_interpolation.get("end"):
                end_data = sysid.interpolation_data.get("end", {}).get(can_id, [])
                if end_data:
                    t, v = collect_phase_data(end_data, field)
                    if t:
                        motor_times.extend([x + chirp_end for x in t])
                        motor_values.extend(v)

            if motor_times:
                plt.plot(motor_times, motor_values, color=color, linewidth=0.8,
                         alpha=0.8, label=f"Motor {can_id}")

        plt.title(f"All Motors - {title}")
        plt.xlabel("Time [s]")
        plt.ylabel(ylabel)
        plt.grid(True, alpha=0.3)
        plt.legend(loc='upper right', fontsize='small', ncol=2)
        plt.tight_layout()

        plot_file = output_path / f"all_{field}_{timestamp}.png"
        plt.savefig(plot_file, dpi=150)
        plt.close()
        print(f"  Saved: {plot_file.name}")


def _compute_fk_from_phase(
    phase_data: dict, motor_ids: list, fk_func, time_key: str
) -> tuple:
    """Compute FK from phase data. Returns (times, pitches, rolls, cmd_p, cmd_r) in radians."""
    lower_data = phase_data.get(motor_ids[0], [])
    upper_data = phase_data.get(motor_ids[1], [])
    if not lower_data or not upper_data:
        return [], [], [], [], []

    lower_by_sample = {d["sample"]: d for d in lower_data}
    upper_by_sample = {d["sample"]: d for d in upper_data}
    common = sorted(set(lower_by_sample.keys()) & set(upper_by_sample.keys()))

    times, pitches, rolls, cmd_p, cmd_r = [], [], [], [], []
    for idx in common:
        q_lower = lower_by_sample[idx]["angle"]
        q_upper = upper_by_sample[idx]["angle"]
        pitch, roll = fk_func(q_lower, q_upper)

        times.append(lower_by_sample[idx][time_key])
        pitches.append(pitch)  # Keep in radians
        rolls.append(roll)  # Keep in radians
        cp = lower_by_sample[idx].get("commanded_pitch")
        cr = lower_by_sample[idx].get("commanded_roll")
        cmd_p.append(cp if cp else 0.0)  # Keep in radians
        cmd_r.append(cr if cr else 0.0)  # Keep in radians

    return times, pitches, rolls, cmd_p, cmd_r

