#!/usr/bin/env python3
"""
System Identification Script for Robot Actuators

Sends chirp signals to motors and collects feedback data.

Usage:
    python system_identification.py --config config/config.json
    python system_identification.py --motors 0 2 5
    python system_identification.py --dry-run --motors 0 1
    python system_identification.py --mujoco --config config/config.json
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from datetime import datetime
from pathlib import Path

# Handle both direct execution and module import
_script_dir = Path(__file__).resolve().parent
if str(_script_dir) not in sys.path:
    sys.path.insert(0, str(_script_dir))

# Initialize IK registry
from ik_registry import IKRegistry, register_default_ik_functions
register_default_ik_functions()

# Optional real-time scheduling
try:
    from realtime import setup_realtime, get_rt_info
except ImportError:
    setup_realtime = None
    get_rt_info = None


def resolve_config_path(config_arg: str) -> Path:
    """Resolve config file path, checking package config/ folder."""
    path = Path(config_arg)
    if path.exists():
        return path

    pkg_root = Path(__file__).resolve().parent.parent
    pkg_config = pkg_root / "config" / path.name
    if pkg_config.exists():
        return pkg_config

    if not config_arg.startswith("config/"):
        pkg_config_full = pkg_root / "config" / config_arg
        if pkg_config_full.exists():
            return pkg_config_full

    return path


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser."""
    parser = argparse.ArgumentParser(
        description="System identification for robot actuators",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python system_identification.py --config config/config.json
  python system_identification.py --motors 0 2 5
  python system_identification.py --mujoco --config config/config.json
  python system_identification.py --dry-run --motors 0 1
  python system_identification.py --realtime --cpu 3 --config config/config.json

Config file example:
  {
    "motor_ids": [0, 1, 2],
    "chirp": {
      "f_start": 0.1, "f_end": 5.0,
      "duration": 20.0, "sample_rate": 500.0
    },
    "control_parameters": {
      "velocity": 0.0, "effort": 0.0,
      "stiffness": 5.0, "damping": 2.0
    }
  }
""",
    )

    # Motor selection
    parser.add_argument(
        "--motors", "-m", type=int, nargs="+", metavar="CAN_ID",
        help="Motor CAN IDs to identify (overrides config file)"
    )
    parser.add_argument(
        "--config", "-c", type=str, default="config.json",
        help="Configuration file (JSON)"
    )

    # Output
    parser.add_argument(
        "--output", "-o", type=str, default="data",
        help="Output directory for results"
    )
    parser.add_argument(
        "--save", "-s", action=argparse.BooleanOptionalAction, default=False,
        help="Save ALL results (JSON, plots, PyTorch)"
    )
    parser.add_argument(
        "--save-json", action=argparse.BooleanOptionalAction, default=False,
        help="Save results to JSON"
    )
    parser.add_argument(
        "--save-plots", action=argparse.BooleanOptionalAction, default=False,
        help="Save plots for each motor"
    )
    parser.add_argument(
        "--save-torch", action=argparse.BooleanOptionalAction, default=False,
        help="Save data in PyTorch .pt format"
    )

    # Mode selection
    parser.add_argument(
        "--dry-run", "-d", action="store_true",
        help="Run without hardware (mock controller)"
    )
    parser.add_argument(
        "--mujoco", action="store_true",
        help="Use MuJoCo simulation (start sim first)"
    )

    # Real-time scheduling
    parser.add_argument(
        "--realtime", type=int, nargs="?", const=90, default=0, metavar="PRIORITY",
        help="Enable RT scheduling (SCHED_FIFO, default priority: 90)"
    )
    parser.add_argument(
        "--cpu", type=int, default=None, metavar="CORE",
        help="Pin process to specific CPU core"
    )
    parser.add_argument(
        "--no-memlock", action="store_true",
        help="Disable memory locking"
    )

    # Info commands
    parser.add_argument(
        "--list-motors", action="store_true",
        help="List motor IDs from config and exit"
    )
    parser.add_argument(
        "--list-ik", action="store_true",
        help="List available IK functions and exit"
    )
    parser.add_argument(
        "--rt-info", action="store_true",
        help="Show real-time scheduling info and exit"
    )

    return parser


def handle_info_commands(args) -> bool:
    """Handle info commands. Returns True if handled."""
    if args.list_ik:
        print("Available IK functions:")
        for name in IKRegistry.list_available():
            info = IKRegistry.get(name)
            print(f"  {name}: inputs={info['input_names']}, "
                  f"motors={info['motor_count']}")
        return True

    if args.rt_info:
        if get_rt_info is None:
            print("Real-time module not available")
        else:
            print("Current real-time configuration:")
            for k, v in get_rt_info().items():
                print(f"  {k}: {v}")
        return True

    return False


def setup_realtime_if_requested(args) -> None:
    """Setup real-time scheduling if requested."""
    if args.realtime > 0 or args.cpu is not None:
        if setup_realtime is None:
            print("Warning: Real-time module not available")
            return

        status = setup_realtime(
            priority=args.realtime,
            cpu=args.cpu,
            lock_mem=not args.no_memlock and args.realtime > 0,
        )
        if not any(status.values()):
            print("Warning: No RT optimizations applied. Check permissions.")
            print("  sudo setcap 'cap_sys_nice,cap_ipc_lock+ep' $(which python3)")


def main() -> None:
    parser = create_parser()
    args = parser.parse_args()

    # Master save switch
    if args.save:
        args.save_json = args.save_plots = args.save_torch = True

    # Handle info commands
    if handle_info_commands(args):
        return

    # Resolve config path
    config_path = resolve_config_path(args.config)
    if not config_path.exists():
        print(f"Error: Config file not found: {args.config}")
        sys.exit(1)

    # Handle --list-motors
    if args.list_motors:
        with config_path.open() as f:
            config = json.load(f)
        motor_ids = config.get("motor_ids", list(config.get("motors", {}).keys()))
        print(f"Motor IDs in config: {motor_ids}")
        return

    # Setup real-time scheduling
    setup_realtime_if_requested(args)

    # Import here to avoid slow startup for info commands
    from sysid import SystemIdentification

    sysid = SystemIdentification(
        str(config_path),
        motor_ids=args.motors,
        dry_run=args.dry_run,
        use_mujoco=args.mujoco,
    )

    try:
        sysid.setup()
        sysid.run_identification()

        # Save results
        any_saving = args.save_json or args.save_plots or args.save_torch
        if any_saving:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_base = Path(args.output)
            if not output_base.is_absolute():
                pkg_root = Path(__file__).resolve().parent.parent
                output_base = pkg_root / output_base
            output_dir = output_base / f"sysid_{timestamp}"
            output_dir.mkdir(parents=True, exist_ok=True)
            print(f"\nOutput folder: {output_dir}")

            if args.save_json:
                sysid.save_results(str(output_dir / "results.json"))
                # Also save communication stats separately
                sysid.save_stats(str(output_dir / "comm_stats.json"))
            if args.save_torch:
                sysid.save_torch(str(output_dir / "results.pt"))
            if args.save_plots:
                sysid.save_plots(str(output_dir / "plots"))
        else:
            print("\nNo output saved (use --save to enable)")

    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"\nError: {e}")
        traceback.print_exc()
    finally:
        sysid.cleanup()


if __name__ == "__main__":
    main()
