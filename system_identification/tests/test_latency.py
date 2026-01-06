#!/usr/bin/env python3
"""
Latency Diagnostic Test for MuJoCo Control Loop

Identifies causes of missed deadlines by measuring:
- Individual loop iteration times
- GC pause correlation
- UDP round-trip latency distribution
- Lock contention effects

Usage:
    python -m system_identification.tests.test_latency
    python -m system_identification.tests.test_latency --duration 30
    python -m system_identification.tests.test_latency --no-gc
"""

from __future__ import annotations

import argparse
import gc
import socket
import struct
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Add parent to path for imports
_scripts_path = Path(__file__).resolve().parent.parent / "scripts"
if str(_scripts_path) not in sys.path:
    sys.path.insert(0, str(_scripts_path))

# Add hoku to path
_hoku_path = Path(__file__).resolve().parent.parent.parent / "hoku"
if str(_hoku_path.parent) not in sys.path:
    sys.path.insert(0, str(_hoku_path.parent))

from hoku.mujoco_controller import pack_control_batch, MsgType  # noqa: E402
from humanoid_messages.can import ControlData  # noqa: E402


@dataclass
class LatencyStats:
    """Statistics for latency measurements"""
    count: int
    mean_us: float
    std_us: float
    min_us: float
    max_us: float
    p50_us: float
    p90_us: float
    p99_us: float
    p999_us: float
    missed_count: int
    missed_pct: float


def compute_stats(times_us: np.ndarray, deadline_us: float) -> LatencyStats:
    """Compute latency statistics from array of times in microseconds"""
    missed = times_us > deadline_us
    return LatencyStats(
        count=len(times_us),
        mean_us=float(np.mean(times_us)),
        std_us=float(np.std(times_us)),
        min_us=float(np.min(times_us)),
        max_us=float(np.max(times_us)),
        p50_us=float(np.percentile(times_us, 50)),
        p90_us=float(np.percentile(times_us, 90)),
        p99_us=float(np.percentile(times_us, 99)),
        p999_us=float(np.percentile(times_us, 99.9)),
        missed_count=int(np.sum(missed)),
        missed_pct=float(np.sum(missed) / len(times_us) * 100),
    )


def print_stats(name: str, stats: LatencyStats, deadline_us: float) -> None:
    """Pretty print latency statistics"""
    print(f"\n{'='*60}")
    print(f" {name}")
    print(f"{'='*60}")
    print(f"  Samples:     {stats.count:,}")
    print(f"  Deadline:    {deadline_us:.0f} µs ({deadline_us/1000:.2f} ms)")
    print()
    mean_ms = stats.mean_us / 1000
    print(f"  Mean:        {stats.mean_us:8.1f} µs ({mean_ms:.3f} ms)")
    print(f"  Std Dev:     {stats.std_us:8.1f} µs")
    print(f"  Min:         {stats.min_us:8.1f} µs")
    spike = " ⚠️ SPIKE!" if stats.max_us > deadline_us else ""
    print(f"  Max:         {stats.max_us:8.1f} µs{spike}")
    print()
    print("  Percentiles:")
    print(f"    p50:       {stats.p50_us:8.1f} µs")
    print(f"    p90:       {stats.p90_us:8.1f} µs")
    w99 = " ⚠️" if stats.p99_us > deadline_us else ""
    print(f"    p99:       {stats.p99_us:8.1f} µs{w99}")
    w999 = " ⚠️" if stats.p999_us > deadline_us else ""
    print(f"    p99.9:     {stats.p999_us:8.1f} µs{w999}")
    print()
    if stats.missed_count > 0:
        print(f"  ❌ Missed:    {stats.missed_count}/{stats.count} "
              f"({stats.missed_pct:.3f}%)")
    else:
        print(f"  ✅ Missed:    0/{stats.count} (0%)")


def print_histogram(
    times_us: np.ndarray, deadline_us: float, bins: int = 20
) -> None:
    """Print ASCII histogram of latency distribution"""
    print(f"\n{'─'*60}")
    print(" Latency Distribution (µs)")
    print(f"{'─'*60}")

    # Create histogram
    hist, edges = np.histogram(times_us, bins=bins)
    max_count = max(hist)
    bar_width = 40

    for i, count in enumerate(hist):
        left = edges[i]
        right = edges[i + 1]
        bar_len = int(count / max_count * bar_width) if max_count > 0 else 0
        marker = "█" * bar_len

        # Mark if this bin crosses the deadline
        crosses_deadline = left <= deadline_us < right
        over_deadline = left >= deadline_us

        if over_deadline:
            prefix = "❌"
        elif crosses_deadline:
            prefix = "⚡"
        else:
            prefix = "  "

        rng = f"{left:7.0f}-{right:7.0f}"
        print(f"{prefix} {rng} |{marker:<{bar_width}} {count:,}")

    print(f"{'─'*60}")
    print(f"  ⚡ = deadline ({deadline_us:.0f} µs)    ❌ = over deadline")


class LatencyTester:
    """
    Direct UDP latency test without full SystemIdentification overhead.
    Isolates the UDP round-trip timing from other factors.
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        send_port: int = 5000,
        recv_port: int = 5001,
        motor_ids: list[int] | None = None,
    ):
        self.host = host
        self.send_port = send_port
        self.recv_port = recv_port
        self.motor_ids = motor_ids or [0, 1, 2, 3, 4, 5]

        self.send_socket: socket.socket | None = None
        self.recv_socket: socket.socket | None = None

    def setup(self) -> None:
        """Initialize sockets"""
        # Send socket
        self.send_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.send_socket.setblocking(False)

        # Receive socket
        self.recv_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.recv_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.recv_socket.bind(("0.0.0.0", self.recv_port))
        self.recv_socket.settimeout(0.1)  # 100ms timeout

        # Start motors
        for mid in self.motor_ids:
            msg = struct.pack("<BB", 4, mid)  # START_MOTOR = 4
            self.send_socket.sendto(msg, (self.host, self.send_port))
            time.sleep(0.01)

        print(f"[Test] Connected to MuJoCo at {self.host}:{self.send_port}")
        print(f"[Test] Motors: {self.motor_ids}")

    def cleanup(self) -> None:
        """Close sockets"""
        if self.send_socket:
            # Stop all motors
            msg = struct.pack("<BB", 6, 0)  # STOP_ALL = 6
            self.send_socket.sendto(msg, (self.host, self.send_port))
            self.send_socket.close()
        if self.recv_socket:
            self.recv_socket.close()

    def measure_round_trip(self, angle: float = 0.0) -> float | None:
        """
        Send control batch and measure time to receive all feedback.
        Returns round-trip time in seconds, or None on timeout.
        """
        # Build control data
        controls = {}
        for mid in self.motor_ids:
            controls[mid] = ControlData(
                angle=angle,
                velocity=0.0,
                effort=0.0,
                stiffness=5.0,
                damping=2.0,
            )

        # Pack and send
        data = pack_control_batch(controls)

        t_start = time.perf_counter()
        self.send_socket.sendto(data, (self.host, self.send_port))

        # Wait for response
        try:
            response, _ = self.recv_socket.recvfrom(1024)
            t_end = time.perf_counter()

            # Verify it's a feedback batch
            if response[0] == MsgType.FEEDBACK_BATCH:
                return t_end - t_start
            else:
                return None
        except socket.timeout:
            return None

    def run_test(
        self,
        duration: float,
        rate: float,
        disable_gc: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, list[tuple[int, float]]]:
        """
        Run latency test for specified duration.

        Returns:
            round_trip_times: Array of UDP round-trip times (seconds)
            loop_times: Array of total loop times including sleep (seconds)
            gc_events: List of (sample_index, gc_duration) tuples
        """
        target_period = 1.0 / rate
        num_samples = int(duration * rate)

        round_trip_times = np.zeros(num_samples)
        loop_times = np.zeros(num_samples)
        gc_events: list[tuple[int, float]] = []

        # Track GC
        gc_before = gc.get_count()

        if disable_gc:
            gc.disable()
            print("[Test] GC disabled")

        print(f"\n[Test] Running {num_samples:,} iterations at {rate} Hz...")
        print(f"[Test] Target period: {target_period*1000:.2f} ms")
        print()

        angle = 0.0
        angle_delta = 0.1 / rate  # Slow movement

        try:
            for i in range(num_samples):
                loop_start = time.perf_counter()

                # Measure round-trip
                rtt = self.measure_round_trip(angle)
                if rtt is not None:
                    round_trip_times[i] = rtt
                else:
                    round_trip_times[i] = target_period * 2  # Mark timeout

                # Check for GC
                gc_after = gc.get_count()
                if gc_after != gc_before:
                    # GC happened during this iteration
                    gc_events.append((i, round_trip_times[i]))
                    gc_before = gc_after

                # Rate limiting with busy wait
                elapsed = time.perf_counter() - loop_start
                remaining = target_period - elapsed
                if remaining > 0:
                    # Hybrid sleep: regular sleep then busy wait
                    if remaining > 0.001:
                        time.sleep(remaining - 0.0005)
                    end_time = loop_start + target_period
                    while time.perf_counter() < end_time:
                        pass

                loop_times[i] = time.perf_counter() - loop_start

                # Update angle (gentle oscillation)
                angle += angle_delta
                if abs(angle) > 0.2:
                    angle_delta = -angle_delta

                # Progress
                if (i + 1) % (num_samples // 10) == 0:
                    pct = (i + 1) / num_samples * 100
                    avg_rtt = np.mean(round_trip_times[:i+1]) * 1000
                    print(f"  {pct:5.1f}% | Sample {i+1:,} | "
                          f"Avg RTT: {avg_rtt:.2f} ms")

        finally:
            if disable_gc:
                gc.enable()
                print("[Test] GC re-enabled")

        return round_trip_times, loop_times, gc_events


def find_spikes(
    times_us: np.ndarray,
    threshold_us: float,
    context: int = 2,
) -> list[dict]:
    """Find iterations that exceeded threshold and return context"""
    spikes = []
    spike_indices = np.where(times_us > threshold_us)[0]

    for idx in spike_indices[:20]:  # Limit to first 20 spikes
        start = max(0, idx - context)
        end = min(len(times_us), idx + context + 1)

        spikes.append({
            "index": int(idx),
            "time_us": float(times_us[idx]),
            "context_before": times_us[start:idx].tolist(),
            "context_after": times_us[idx+1:end].tolist(),
        })

    return spikes


def main():
    parser = argparse.ArgumentParser(
        description="Latency diagnostic test for MuJoCo control loop"
    )
    parser.add_argument(
        "--duration", type=float, default=10.0,
        help="Test duration in seconds (default: 10)"
    )
    parser.add_argument(
        "--rate", type=float, default=500.0,
        help="Target control rate in Hz (default: 500)"
    )
    parser.add_argument(
        "--host", type=str, default="127.0.0.1",
        help="MuJoCo server host (default: 127.0.0.1)"
    )
    parser.add_argument(
        "--no-gc", action="store_true",
        help="Disable garbage collection during test"
    )
    parser.add_argument(
        "--motors", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5],
        help="Motor IDs to test (default: 0 1 2 3 4 5)"
    )
    args = parser.parse_args()

    deadline_us = 1_000_000 / args.rate  # Convert to microseconds

    print("=" * 60)
    print(" MuJoCo Latency Diagnostic Test")
    print("=" * 60)
    print(f"  Duration:    {args.duration} seconds")
    print(f"  Rate:        {args.rate} Hz")
    print(f"  Deadline:    {deadline_us:.0f} µs ({deadline_us/1000:.2f} ms)")
    print(f"  GC:          {'disabled' if args.no_gc else 'enabled'}")
    print(f"  Motors:      {args.motors}")
    print()

    tester = LatencyTester(
        host=args.host,
        motor_ids=args.motors,
    )

    try:
        tester.setup()

        # Warmup
        print("[Test] Warmup (100 iterations)...")
        for _ in range(100):
            tester.measure_round_trip()

        # Run test
        rtt_times, loop_times, gc_events = tester.run_test(
            duration=args.duration,
            rate=args.rate,
            disable_gc=args.no_gc,
        )

        # Convert to microseconds
        rtt_us = rtt_times * 1_000_000
        loop_us = loop_times * 1_000_000

        # Statistics
        rtt_stats = compute_stats(rtt_us, deadline_us)
        loop_stats = compute_stats(loop_us, deadline_us)

        print_stats("UDP Round-Trip Time", rtt_stats, deadline_us)
        print_stats(
            "Total Loop Time (including sleep)", loop_stats, deadline_us
        )

        # Histogram for round-trip times
        print_histogram(rtt_us, deadline_us)

        # GC correlation
        if gc_events:
            print(f"\n{'='*60}")
            print(f" GC Events: {len(gc_events)}")
            print(f"{'='*60}")
            gc_times = [e[1] * 1_000_000 for e in gc_events]
            avg_gc_spike = np.mean(gc_times)
            print(f"  GC collections during test: {len(gc_events)}")
            print(f"  Average latency during GC:  {avg_gc_spike:.1f} µs")

            # Check correlation
            gc_indices = [e[0] for e in gc_events]
            spike_indices = set(np.where(rtt_us > deadline_us)[0])
            gc_spikes = len(spike_indices.intersection(gc_indices))
            n_spikes = len(spike_indices)
            print(f"  GC-correlated missed deadlines: {gc_spikes}/{n_spikes}")

        # Show spike details
        spikes = find_spikes(rtt_us, deadline_us)
        if spikes:
            print(f"\n{'='*60}")
            print(f" Spike Analysis (showing first {len(spikes)})")
            print(f"{'='*60}")
            for spike in spikes[:10]:
                idx = spike["index"]
                t = spike["time_us"]
                before = spike["context_before"]
                after = spike["context_after"]

                gc_idxs = [e[0] for e in gc_events]
                gc_marker = " [GC?]" if idx in gc_idxs else ""
                print(f"  Sample {idx:,}: {t:.0f} µs{gc_marker}")
                print(f"    Before: {[f'{x:.0f}' for x in before]}")
                print(f"    After:  {[f'{x:.0f}' for x in after]}")

        # Summary
        print(f"\n{'='*60}")
        print(" SUMMARY")
        print(f"{'='*60}")
        if rtt_stats.missed_count == 0:
            print("  ✅ No missed deadlines!")
        else:
            print(f"  ❌ {rtt_stats.missed_count} missed deadlines "
                  f"({rtt_stats.missed_pct:.3f}%)")
            print()
            print("  Likely causes:")
            gc_correlation = len(gc_events) > rtt_stats.missed_count * 0.5
            if gc_events and gc_correlation:
                print("    • Python GC pauses (try --no-gc)")
            max_spike = rtt_stats.max_us > deadline_us * 2
            if rtt_stats.p99_us < deadline_us and max_spike:
                print("    • Occasional lock contention in MuJoCo")
            if rtt_stats.mean_us > deadline_us * 0.5:
                print("    • Rate may be too high for this setup")
            print()
            print("  Recommendations:")
            print("    • Run MuJoCo with --headless to reduce lock contention")
            print("    • Try --no-gc to eliminate GC pauses")
            print("    • Lower the rate if p99 is close to deadline")

    finally:
        tester.cleanup()


if __name__ == "__main__":
    main()
