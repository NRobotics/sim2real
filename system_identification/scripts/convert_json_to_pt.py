#!/usr/bin/env python3
"""
Convert raw_debug_data.json to chirp_data.pt format

This script reads the JSON output from system_identification.py and converts
it to the PyTorch .pt format used by the Sim2Real tools.

Usage:
    python convert_json_to_pt.py raw_debug_data.json [--output output_dir]
    python convert_json_to_pt.py data/sysid_20251218_170101/raw_debug_data.json
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

try:
    import torch
except ImportError:
    print("Error: PyTorch is required. Install with: pip install torch")
    sys.exit(1)

import numpy as np

# Import kinematics for FK transformation
try:
    from ..kinematics import fk_motor_to_foot
except ImportError:
    # Handle running directly (python scripts/convert_json_to_pt.py)
    # Add parent directory to path so we can import kinematics
    _parent_dir = Path(__file__).resolve().parent.parent
    if str(_parent_dir) not in sys.path:
        sys.path.insert(0, str(_parent_dir))
    from kinematics import fk_motor_to_foot


def get_joint_name(can_id: int, config: dict, motor_to_ik_group: dict[int, tuple[str, int]]) -> str:
    """Resolve a human-readable name for a CAN ID from the config."""
    # 1. Check if it's in an IK group
    if can_id in motor_to_ik_group:
        group_name, idx = motor_to_ik_group[can_id]
        
        # Find the group config to check for explicit motor names
        for group_cfg in config.get("ik_groups", []):
            if group_cfg["name"] == group_name:
                if "motor_names" in group_cfg and len(group_cfg["motor_names"]) > idx:
                    return group_cfg["motor_names"][idx]
        
        # Fallback for IK groups
        return f"{group_name}_motor_{idx}"

    # 2. Check direct motor config
    motor_conf = config.get("motors", {}).get(str(can_id), {})
    if "name" in motor_conf:
        return motor_conf["name"]

    # 3. Fallback to generic ID
    return f"joint_can_{can_id}"


def build_motor_to_ik_group(config: dict) -> dict[int, tuple[str, int]]:
    """Build mapping from motor_id to (group_name, index_in_group) from config."""
    motor_to_ik_group = {}
    
    for group_cfg in config.get("ik_groups", []):
        group_name = group_cfg["name"]
        motor_ids = group_cfg["motor_ids"]
        
        for idx, motor_id in enumerate(motor_ids):
            motor_to_ik_group[motor_id] = (group_name, idx)
    
    return motor_to_ik_group


def convert_json_to_pt(json_file: Path, output_dir: Path = None) -> None:
    """Convert raw_debug_data.json to chirp_data.pt format."""
    
    if not json_file.exists():
        print(f"Error: JSON file not found: {json_file}")
        sys.exit(1)
    
    print(f"Loading JSON file: {json_file}")
    with json_file.open() as f:
        data = json.load(f)
    
    motor_ids = data["motor_ids"]
    config = data["config"]
    feedback_data = data["feedback_data"]
    
    # Convert string keys to integers for feedback_data
    feedback_data_int = {int(k): v for k, v in feedback_data.items()}
    
    print(f"Motor IDs: {motor_ids}")
    print(f"Found feedback data for {len(feedback_data_int)} motors")
    
    # Build IK group mapping
    motor_to_ik_group = build_motor_to_ik_group(config)
    if motor_to_ik_group:
        print(f"IK groups found: {set(gname for gname, _ in motor_to_ik_group.values())}")
    
    # 1. Align Data by Sample
    # Filter to only include actual chirp signal samples (exclude initialization/finalization)
    # The statistics.total_samples indicates the actual chirp signal length
    chirp_sample_count = data.get("statistics", {}).get("total_samples", None)
    
    sample_sets = []
    for can_id in motor_ids:
        motor_data = feedback_data_int.get(can_id, [])
        if motor_data:
            # Filter to only include samples within the chirp signal range [0, chirp_sample_count-1]
            if chirp_sample_count is not None:
                filtered_data = [d for d in motor_data if 0 <= d["sample"] < chirp_sample_count]
                sample_sets.append({d["sample"] for d in filtered_data})
            else:
                sample_sets.append({d["sample"] for d in motor_data})
    
    if not sample_sets:
        print("[ERROR] No data found in JSON file.")
        sys.exit(1)
    
    common_samples = sorted(set.intersection(*sample_sets))
    num_common_samples = len(common_samples)
    num_joints = len(motor_ids)
    
    # Determine expected sample range (only for chirp signal, not initialization/finalization)
    if chirp_sample_count is not None:
        expected_max = chirp_sample_count - 1
        expected_samples = set(range(chirp_sample_count))  # 0 to chirp_sample_count-1
    else:
        expected_max = max(common_samples) if common_samples else 0
        expected_samples = set(range(expected_max + 1))
    
    missing_samples = sorted(expected_samples - set(common_samples))
    
    if chirp_sample_count is not None:
        print(f"Filtering to chirp signal samples: 0 to {chirp_sample_count-1}")
    print(f"Common samples: {num_common_samples}")
    if missing_samples:
        print(f"Missing sample indices: {missing_samples} (will interpolate)")
    print(f"Number of joints: {num_joints}")
    
    # 2. Prepare Tensors (will be expanded after interpolation if needed)
    time_data = torch.zeros(num_common_samples)
    dof_pos = torch.zeros(num_common_samples, num_joints)
    des_dof_pos = torch.zeros(num_common_samples, num_joints)
    
    # Resolve Joint Names (we will modify these if FK is applied)
    joint_names = [get_joint_name(mid, config, motor_to_ik_group) for mid in motor_ids]
    
    # Build motor lookups, filtering to only chirp signal samples
    motor_lookups = {}
    for can_id in motor_ids:
        motor_data = feedback_data_int.get(can_id, [])
        if chirp_sample_count is not None:
            # Only include samples within chirp signal range
            filtered_data = [d for d in motor_data if 0 <= d["sample"] < chirp_sample_count]
            motor_lookups[can_id] = {d["sample"]: d for d in filtered_data}
        else:
            motor_lookups[can_id] = {d["sample"]: d for d in motor_data}
    
    # 3. Fill Tensors with RAW data (motor angles)
    # Get first timestamp to make all timestamps relative
    first_timestamp = motor_lookups[motor_ids[0]][common_samples[0]]["timestamp"]
    
    for i, sample_idx in enumerate(common_samples):
        # Make timestamp relative to start (in case it wasn't saved that way)
        time_data[i] = motor_lookups[motor_ids[0]][sample_idx]["timestamp"] - first_timestamp
        for j, can_id in enumerate(motor_ids):
            d = motor_lookups[can_id][sample_idx]
            dof_pos[i, j] = d["angle"]
            des_dof_pos[i, j] = d["commanded_angle"]
    
    # 3.5. Interpolate missing samples
    if missing_samples:
        print(f"Interpolating {len(missing_samples)} missing samples...")
        
        # Convert to numpy for easier interpolation
        sample_indices_np = np.array(common_samples)
        time_data_np = time_data.numpy()
        dof_pos_np = dof_pos.numpy()
        des_dof_pos_np = des_dof_pos.numpy()
        
        # Create new arrays with all samples (common + missing)
        all_sample_indices = sorted(common_samples + missing_samples)
        num_samples = len(all_sample_indices)
        
        new_time_data = torch.zeros(num_samples)
        new_dof_pos = torch.zeros(num_samples, num_joints)
        new_des_dof_pos = torch.zeros(num_samples, num_joints)
        
        # Fill in existing data
        common_to_all = {sample: i for i, sample in enumerate(all_sample_indices)}
        for i, sample_idx in enumerate(common_samples):
            all_idx = common_to_all[sample_idx]
            new_time_data[all_idx] = time_data[i]
            new_dof_pos[all_idx] = dof_pos[i]
            new_des_dof_pos[all_idx] = des_dof_pos[i]
        
        # Interpolate missing samples using linear interpolation
        for missing_idx in missing_samples:
            all_idx = common_to_all[missing_idx]
            
            # Find surrounding samples
            prev_samples = [s for s in common_samples if s < missing_idx]
            next_samples = [s for s in common_samples if s > missing_idx]
            
            if prev_samples and next_samples:
                # Linear interpolation between previous and next
                prev_idx = max(prev_samples)
                next_idx = min(next_samples)
                prev_all_idx = common_to_all[prev_idx]
                next_all_idx = common_to_all[next_idx]
                
                # Linear interpolation factor
                t = (missing_idx - prev_idx) / (next_idx - prev_idx)
                
                # Interpolate time
                new_time_data[all_idx] = new_time_data[prev_all_idx] + t * (
                    new_time_data[next_all_idx] - new_time_data[prev_all_idx]
                )
                
                # Interpolate positions (handle angular wrapping)
                for j in range(num_joints):
                    prev_angle = new_dof_pos[prev_all_idx, j].item()
                    next_angle = new_dof_pos[next_all_idx, j].item()
                    
                    # Handle angular wrapping (shortest path)
                    diff = next_angle - prev_angle
                    diff = ((diff + np.pi) % (2 * np.pi)) - np.pi
                    new_dof_pos[all_idx, j] = prev_angle + t * diff
                    
                    # Same for commanded positions
                    prev_cmd = new_des_dof_pos[prev_all_idx, j].item()
                    next_cmd = new_des_dof_pos[next_all_idx, j].item()
                    diff_cmd = next_cmd - prev_cmd
                    diff_cmd = ((diff_cmd + np.pi) % (2 * np.pi)) - np.pi
                    new_des_dof_pos[all_idx, j] = prev_cmd + t * diff_cmd
                    
            elif prev_samples:
                # Extrapolate forward (use last value)
                prev_idx = max(prev_samples)
                prev_all_idx = common_to_all[prev_idx]
                new_time_data[all_idx] = new_time_data[prev_all_idx]
                new_dof_pos[all_idx] = new_dof_pos[prev_all_idx]
                new_des_dof_pos[all_idx] = new_des_dof_pos[prev_all_idx]
            elif next_samples:
                # Extrapolate backward (use first value)
                next_idx = min(next_samples)
                next_all_idx = common_to_all[next_idx]
                new_time_data[all_idx] = new_time_data[next_all_idx]
                new_dof_pos[all_idx] = new_dof_pos[next_all_idx]
                new_des_dof_pos[all_idx] = new_des_dof_pos[next_all_idx]
        
        # Replace tensors with interpolated versions
        time_data = new_time_data
        dof_pos = new_dof_pos
        des_dof_pos = new_des_dof_pos
        num_samples = len(all_sample_indices)
    else:
        num_samples = num_common_samples
    
    # 4. Perform FK and Overwrite for foot IK groups
    ik_groups = config.get("ik_groups", [])
    for group_cfg in ik_groups:
        group_name = group_cfg["name"]
        ik_type = group_cfg.get("ik_type", "")
        
        if ik_type != "foot":
            continue
        
        # Get sorted motors (0: Lower/Pitch, 1: Upper/Roll)
        # Only include motors that are in motor_ids
        group_motor_ids = [
            mid for mid, (gname, _) in motor_to_ik_group.items()
            if gname == group_name and mid in motor_ids
        ]
        group_motor_ids.sort(key=lambda m: motor_to_ik_group[m][1])
        
        if len(group_motor_ids) != 2:
            print(f"Warning: IK group '{group_name}' has {len(group_motor_ids)} motors, expected 2. Skipping FK transformation.")
            continue
        
        id_pitch, id_roll = group_motor_ids[0], group_motor_ids[1]
        if id_pitch not in motor_ids or id_roll not in motor_ids:
            print(f"Warning: Motors {id_pitch}, {id_roll} not in motor_ids list. Skipping FK transformation.")
            continue
        
        idx_pitch = motor_ids.index(id_pitch)
        idx_roll = motor_ids.index(id_roll)
        
        # Rename joints to reflect data content after FK
        joint_names[idx_pitch] = f"{group_name}_pitch"
        joint_names[idx_roll] = f"{group_name}_roll"
        
        print(f"Applying FK transformation for group '{group_name}': motors {id_pitch}->pitch, {id_roll}->roll")
        
        # Apply FK to measured angles
        q_l_meas = dof_pos[:, idx_pitch].tolist()
        q_u_meas = dof_pos[:, idx_roll].tolist()
        meas_p, meas_r = [], []
        for ql, qu in zip(q_l_meas, q_u_meas):
            p, r = fk_motor_to_foot(ql, qu)
            meas_p.append(p)
            meas_r.append(r)
        
        # Apply FK to commanded angles
        q_l_cmd = des_dof_pos[:, idx_pitch].tolist()
        q_u_cmd = des_dof_pos[:, idx_roll].tolist()
        cmd_p, cmd_r = [], []
        for ql, qu in zip(q_l_cmd, q_u_cmd):
            p, r = fk_motor_to_foot(ql, qu)
            cmd_p.append(p)
            cmd_r.append(r)
        
        # Overwrite tensors with FK results
        dof_pos[:, idx_pitch] = torch.tensor(meas_p)
        des_dof_pos[:, idx_pitch] = torch.tensor(cmd_p)
        
        dof_pos[:, idx_roll] = torch.tensor(meas_r)
        des_dof_pos[:, idx_roll] = torch.tensor(cmd_r)
    
    # 5. Determine output path
    if output_dir is None:
        # Default: save next to JSON file
        output_dir = json_file.parent
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    save_path = output_dir / "chirp_data.pt"
    
    # 6. Save
    torch.save(
        {
            "time": time_data.cpu(),
            "dof_pos": dof_pos.cpu(),
            "des_dof_pos": des_dof_pos.cpu(),
            "joint_names": joint_names,
        },
        save_path,
    )
    
    print(f"\n[SUCCESS] Saved Sim2Real formatted data to {save_path}")
    print(f"  Time range: {time_data[0].item():.3f} - {time_data[-1].item():.3f} seconds")
    print(f"  Samples: {num_samples}")
    print(f"  Joints: {num_joints}")
    print(f"  Joint names: {joint_names}")
    
    # 7. Save plots
    save_plots(time_data, dof_pos, des_dof_pos, joint_names, output_dir)


def save_plots(
    time_data: torch.Tensor,
    dof_pos: torch.Tensor,
    des_dof_pos: torch.Tensor,
    joint_names: list[str],
    output_dir: Path,
) -> None:
    """Save a plot showing all motor responses and commands."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Warning: matplotlib not installed, skipping plots")
        return
    
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print(f"\nSaving plots to {output_dir}...")
    
    num_joints = len(joint_names)
    time_np = time_data.cpu().numpy()
    
    # Create a figure with subplots for each joint
    fig, axes = plt.subplots(num_joints, 1, figsize=(14, 3 * num_joints), sharex=True)
    
    # Handle case where there's only one joint (axes won't be an array)
    if num_joints == 1:
        axes = [axes]
    
    for i, joint_name in enumerate(joint_names):
        ax = axes[i]
        
        # Get measured and commanded positions
        measured = dof_pos[:, i].cpu().numpy()
        commanded = des_dof_pos[:, i].cpu().numpy()
        
        # Plot measured and commanded
        ax.plot(time_np, measured, label="Measured", linewidth=1.5, alpha=0.8)
        ax.plot(time_np, commanded, label="Commanded", linewidth=1.5, linestyle="--", alpha=0.8)
        
        # Add secondary axis for degrees
        secax = ax.secondary_yaxis('right', functions=(np.degrees, np.radians))
        secax.set_ylabel('Position [deg]', fontsize=9)
        
        ax.set_ylabel('Position [rad]', fontsize=10)
        ax.set_title(f"{joint_name}", fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=9)
    
    # Set x-axis label on bottom subplot
    axes[-1].set_xlabel('Time [s]', fontsize=10)
    
    plt.tight_layout()
    
    plot_file = output_dir / f"all_joints_{timestamp}.png"
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {plot_file.name}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert raw_debug_data.json to chirp_data.pt format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert JSON to .pt file (saves next to JSON)
  python convert_json_to_pt.py data/sysid_20251218_170101/raw_debug_data.json

  # Specify output directory
  python convert_json_to_pt.py raw_debug_data.json --output output_dir
        """,
    )
    parser.add_argument(
        "json_file",
        type=str,
        help="Path to raw_debug_data.json file",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output directory (default: same directory as JSON file)",
    )
    
    args = parser.parse_args()
    
    json_path = Path(args.json_file)
    output_path = Path(args.output) if args.output else None
    
    convert_json_to_pt(json_path, output_path)


if __name__ == "__main__":
    main()

