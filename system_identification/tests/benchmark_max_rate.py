#!/usr/bin/env python3
"""
Benchmark: Measure maximum achievable rate with mock controller.

This measures the theoretical maximum rate limited only by Python/code overhead,
not by actual CAN hardware.

Usage:
    python -m system_identification.tests.benchmark_max_rate
    # or from system_identification directory:
    python tests/benchmark_max_rate.py
"""

import builtins
import json
import tempfile
import time
from pathlib import Path

# Suppress prints during benchmark
_original_print = builtins.print
builtins.print = lambda *args, **kwargs: None

from system_identification import SystemIdentification

# Restore print
builtins.print = _original_print


def measure_direct_motors(duration: float = 1.0, motor_count: int = 1) -> dict:
    """Measure max rate with direct motor control (no IK)."""
    
    motor_ids = list(range(motor_count))
    config = {
        "can_interface": {
            "interface": "socketcan",
            "channel": "can0",
            "bitrate": 5000000,
            "fd": True,
        },
        "chirp": {
            "f_start": 0.1,
            "f_end": 10.0,
            "duration": duration,
            "sample_rate": 100000.0,  # Very high - won't be the limiting factor
            "sweep_type": "linear",
        },
        "control_parameters": {
            "velocity": 0.0,
            "effort": 0.0,
            "stiffness": 1.0,
            "damping": 0.0,
        },
        "motor_ids": motor_ids,
        "motors": {
            str(i): {"scale": 0.25, "direction": 1.0, "bias": 0.0} 
            for i in motor_ids
        },
    }

    return _run_benchmark(config, motor_ids, duration, "direct")


def measure_ik_motors(duration: float = 1.0, ik_group_count: int = 1) -> dict:
    """Measure max rate with IK-controlled motors (foot IK, 2 motors per group)."""
    
    motor_count = ik_group_count * 2  # 2 motors per foot
    motor_ids = list(range(motor_count))
    
    # Create IK groups (each foot IK controls 2 motors)
    ik_groups = []
    for i in range(ik_group_count):
        ik_groups.append({
            "name": f"foot_{i}",
            "ik_type": "foot",
            "motor_ids": [i * 2, i * 2 + 1],
            "chirp": {
                "scale_pitch": 0.1,
                "direction_pitch": 1.0,
                "bias_pitch": 0.0,
                "scale_roll": 0.1,
                "direction_roll": 1.0 if i % 2 == 0 else -1.0,  # Alternate direction
                "bias_roll": 0.0,
            }
        })
    
    config = {
        "can_interface": {
            "interface": "socketcan",
            "channel": "can0",
            "bitrate": 5000000,
            "fd": True,
        },
        "chirp": {
            "f_start": 0.1,
            "f_end": 10.0,
            "duration": duration,
            "sample_rate": 100000.0,  # Very high - won't be the limiting factor
            "sweep_type": "linear",
        },
        "control_parameters": {
            "velocity": 0.0,
            "effort": 0.0,
            "stiffness": 1.0,
            "damping": 0.0,
        },
        "motor_ids": motor_ids,
        "ik_groups": ik_groups,
        "motors": {},  # All motors controlled via IK
    }

    return _run_benchmark(config, motor_ids, duration, "ik")


def _run_benchmark(config: dict, motor_ids: list, duration: float, mode: str) -> dict:
    """Run the benchmark with given config."""
    
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(config, f)
        config_file = f.name

    # Suppress output during run
    builtins.print = lambda *args, **kwargs: None
    
    try:
        sysid = SystemIdentification(config_file, motor_ids=motor_ids, dry_run=True)
        sysid.setup()
        
        start = time.perf_counter()
        sysid.run_identification()
        elapsed = time.perf_counter() - start
        
        samples = sysid.sample_count
        rate = samples / elapsed if elapsed > 0 else 0
        loop_time_us = (elapsed / samples) * 1e6 if samples > 0 else 0
        
        sysid.cleanup()
        
        return {
            "mode": mode,
            "motors": len(motor_ids),
            "ik_groups": len(config.get("ik_groups", [])),
            "samples": samples,
            "elapsed_s": elapsed,
            "rate_hz": rate,
            "loop_time_us": loop_time_us,
        }
    finally:
        builtins.print = _original_print
        Path(config_file).unlink(missing_ok=True)


def main():
    print("=" * 70)
    print("Maximum Rate Benchmark (Mock Controller - No Hardware Latency)")
    print("=" * 70)
    print()
    
    # Direct motor control benchmark
    print("DIRECT MOTOR CONTROL (no IK computation):")
    print("-" * 60)
    print(f"{'Motors':<10} {'Samples':<12} {'Max Rate':<15} {'Loop Time':<15}")
    print("-" * 60)
    
    for motor_count in [2, 4, 6, 12]:
        result = measure_direct_motors(duration=1.0, motor_count=motor_count)
        print(
            f"{result['motors']:<10} "
            f"{result['samples']:<12,} "
            f"{result['rate_hz']:>10,.0f} Hz   "
            f"{result['loop_time_us']:>10.1f} µs"
        )
    
    print()
    
    # IK motor control benchmark
    print("IK MOTOR CONTROL (foot IK, 2 motors per group):")
    print("-" * 70)
    print(f"{'IK Groups':<12} {'Motors':<10} {'Samples':<12} {'Max Rate':<15} {'Loop Time':<12}")
    print("-" * 70)
    
    for ik_groups in [1, 2, 3, 6]:  # 2, 4, 6, 12 motors
        result = measure_ik_motors(duration=1.0, ik_group_count=ik_groups)
        print(
            f"{result['ik_groups']:<12} "
            f"{result['motors']:<10} "
            f"{result['samples']:<12,} "
            f"{result['rate_hz']:>10,.0f} Hz   "
            f"{result['loop_time_us']:>8.1f} µs"
        )
    
    print()
    print("=" * 70)
    print("Note: Actual CAN hardware is much slower (typically 100-500 Hz)")
    print("      IK adds computation overhead but is still far from bottleneck.")
    print("=" * 70)


if __name__ == "__main__":
    main()

