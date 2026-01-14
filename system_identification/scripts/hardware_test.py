#!/usr/bin/env python3
"""
Hardware test script for motor communication validation.

Tests:
  1. Connection - verify motors respond to config requests
  2. Ping - verify ping_motor returns feedback
  3. Commands - verify position commands work with feedback
  4. Timing - measure round-trip latency
  5. Stress - prolonged async loop test (optional)

Usage:
  python hardware_test.py --mujoco           # MuJoCo simulation
  python hardware_test.py --dry-run          # Mock controller
  python hardware_test.py                    # Real hardware
  python hardware_test.py --motor-ids 0 1 2  # Specific motors
  python hardware_test.py --mujoco --stress  # Include stress test
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Setup path for imports
_script_dir = Path(__file__).resolve().parent
if str(_script_dir) not in sys.path:
    sys.path.insert(0, str(_script_dir))

from test_modules import (  # noqa: E402
    TestResult,
    run_connection_test,
    run_ping_test,
    run_command_test,
    run_timing_test,
    run_stress_test,
    create_controller,
    print_test_summary,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hardware test")
    parser.add_argument(
        "--config", "-c",
        default=str(_script_dir.parent / "config" / "config.json"),
        help="Config file path",
    )
    parser.add_argument(
        "--motor-ids", "-m",
        type=int,
        nargs="+",
        help="Motor IDs (default: from config)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Use mock controller",
    )
    parser.add_argument(
        "--mujoco",
        action="store_true",
        help="Use MuJoCo simulation",
    )
    parser.add_argument(
        "--skip",
        nargs="+",
        choices=["connection", "ping", "command", "timing", "stress"],
        default=[],
        help="Tests to skip",
    )
    parser.add_argument(
        "--stress",
        action="store_true",
        help="Include prolonged stress test",
    )
    parser.add_argument(
        "--stress-duration",
        type=float,
        default=5.0,
        help="Stress test duration in seconds (default: 5)",
    )
    parser.add_argument(
        "--stress-rate",
        type=float,
        default=30.0,
        help="Stress test rate in Hz (default: 30, USB-CAN limit)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output",
    )
    return parser.parse_args()


def load_config(config_file: str) -> dict:
    """Load configuration from JSON file."""
    with Path(config_file).open() as f:
        return json.load(f)


def get_motor_ids(config: dict, override: list[int] | None) -> list[int]:
    """Get motor IDs from config or override."""
    if override:
        return override
    if "motor_ids" in config:
        return [int(x) for x in config["motor_ids"]]
    motors = config.get("motors", {})
    if motors:
        return [int(k) for k in motors.keys()]
    return [0]


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    motor_ids = get_motor_ids(config, args.motor_ids)

    if args.mujoco:
        mode = "MuJoCo"
    elif args.dry_run:
        mode = "Dry-run"
    else:
        mode = "Hardware"

    print("=" * 60)
    print("HARDWARE TEST")
    print("=" * 60)
    print(f"Motor IDs: {motor_ids}")
    print(f"Mode: {mode}")
    print("=" * 60)

    # Create controller
    controller = create_controller(config, args.dry_run, args.mujoco)

    try:
        controller.start()
        time.sleep(0.1)

        results: list[TestResult] = []

        # Run tests
        if "connection" not in args.skip:
            r = run_connection_test(controller, motor_ids, args.verbose)
            results.append(r)

        if "ping" not in args.skip:
            r = run_ping_test(controller, motor_ids, args.verbose)
            results.append(r)

        if "command" not in args.skip:
            r = run_command_test(controller, motor_ids, args.verbose)
            results.append(r)

        if "timing" not in args.skip:
            r = run_timing_test(controller, motor_ids, args.verbose)
            results.append(r)

        # Stress test (optional - must be explicitly enabled)
        if args.stress and "stress" not in args.skip:
            r = run_stress_test(
                controller, motor_ids, args.verbose,
                duration=args.stress_duration,
                rate=args.stress_rate,
            )
            results.append(r)

        # Print summary
        print_test_summary(results)

        # Return exit code
        failed = sum(1 for r in results if not r.passed)
        return 1 if failed > 0 else 0

    except KeyboardInterrupt:
        print("\nTest interrupted")
        return 130
    finally:
        controller.stop()


if __name__ == "__main__":
    sys.exit(main())
