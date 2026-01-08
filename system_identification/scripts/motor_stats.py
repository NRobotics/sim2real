"""
Motor statistics tracking for asynchronous control loops.

Tracks per-motor metrics:
- Feedback reception timing
- Latency statistics
- Command/feedback counts
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any


@dataclass
class MotorTimingStats:
    """Statistics for a single motor's communication timing."""

    # Counters
    commands_sent: int = 0
    feedbacks_received: int = 0
    late_feedbacks: int = 0

    # Latency tracking (in seconds)
    latency_samples: list[float] = field(default_factory=list)
    max_latency: float = 0.0
    min_latency: float = float("inf")

    # Timing for current cycle
    last_command_time: float = 0.0
    last_feedback_time: float = 0.0

    def record_command_sent(self, timestamp: float) -> None:
        """Record that a command was sent."""
        self.commands_sent += 1
        self.last_command_time = timestamp

    def record_feedback_received(
        self,
        timestamp: float,
        deadline: float,
    ) -> dict[str, Any]:
        """
        Record feedback reception and compute stats.

        Args:
            timestamp: Time feedback was received
            deadline: Expected deadline for this feedback

        Returns:
            Dict with feedback status info
        """
        result = {
            "on_time": False,
            "late": False,
            "latency": 0.0,
        }

        self.feedbacks_received += 1
        self.last_feedback_time = timestamp

        # Compute latency from command send
        if self.last_command_time > 0:
            latency = timestamp - self.last_command_time
            result["latency"] = latency
            self.latency_samples.append(latency)
            self.max_latency = max(self.max_latency, latency)
            self.min_latency = min(self.min_latency, latency)

        # Check if late
        if timestamp > deadline:
            self.late_feedbacks += 1
            result["late"] = True
        else:
            result["on_time"] = True

        return result

    @property
    def feedback_rate(self) -> float:
        """Percentage of feedbacks received vs commands sent."""
        if self.commands_sent == 0:
            return 0.0
        return (self.feedbacks_received / self.commands_sent) * 100

    @property
    def avg_latency(self) -> float:
        """Average latency in seconds."""
        if not self.latency_samples:
            return 0.0
        return sum(self.latency_samples) / len(self.latency_samples)

    @property
    def latency_std(self) -> float:
        """Standard deviation of latency."""
        if len(self.latency_samples) < 2:
            return 0.0
        avg = self.avg_latency
        variance = sum((x - avg) ** 2 for x in self.latency_samples) / len(
            self.latency_samples
        )
        return variance**0.5

    def to_dict(self) -> dict[str, Any]:
        """Export stats as dictionary."""
        return {
            "commands_sent": self.commands_sent,
            "feedbacks_received": self.feedbacks_received,
            "late_feedbacks": self.late_feedbacks,
            "feedback_rate_pct": round(self.feedback_rate, 2),
            "avg_latency_ms": round(self.avg_latency * 1000, 3),
            "min_latency_ms": round(self.min_latency * 1000, 3)
            if self.min_latency != float("inf")
            else 0.0,
            "max_latency_ms": round(self.max_latency * 1000, 3),
            "latency_std_ms": round(self.latency_std * 1000, 3),
        }


class MotorStatsManager:
    """Thread-safe manager for multiple motor statistics."""

    def __init__(self, motor_ids: list[int]):
        self.motor_ids = motor_ids
        self._lock = threading.Lock()
        self._stats: dict[int, MotorTimingStats] = {
            mid: MotorTimingStats() for mid in motor_ids
        }

        # Global timing
        self.start_time: float = 0.0
        self.total_cycles: int = 0
        self.missed_deadlines: int = 0

        # Loop timing stats
        self.loop_times: list[float] = []
        self.send_times: list[float] = []

    def start(self) -> None:
        """Mark the start of identification."""
        self.start_time = time.perf_counter()

    def record_command_sent(
        self, can_id: int, timestamp: float | None = None
    ) -> None:
        """Record command sent for a motor."""
        if timestamp is None:
            timestamp = time.perf_counter()
        with self._lock:
            if can_id in self._stats:
                self._stats[can_id].record_command_sent(timestamp)

    def record_feedback_received(
        self,
        can_id: int,
        deadline: float,
        timestamp: float | None = None,
    ) -> dict[str, Any]:
        """Record feedback received for a motor."""
        if timestamp is None:
            timestamp = time.perf_counter()
        with self._lock:
            if can_id in self._stats:
                return self._stats[can_id].record_feedback_received(
                    timestamp, deadline
                )
        return {}

    def record_cycle(self, loop_time: float, missed_deadline: bool = False) -> None:
        """Record cycle timing stats."""
        with self._lock:
            self.total_cycles += 1
            self.loop_times.append(loop_time)
            if missed_deadline:
                self.missed_deadlines += 1

    def record_send_time(self, send_time: float) -> None:
        """Record time to send all commands."""
        with self._lock:
            self.send_times.append(send_time)

    def get_motor_stats(self, can_id: int) -> dict[str, Any]:
        """Get stats for a specific motor."""
        with self._lock:
            if can_id in self._stats:
                return self._stats[can_id].to_dict()
        return {}

    def get_summary(self) -> dict[str, Any]:
        """Get summary of all motor stats."""
        with self._lock:
            elapsed = time.perf_counter() - self.start_time if self.start_time else 0

            # Compute aggregate stats
            total_commands = sum(s.commands_sent for s in self._stats.values())
            total_feedbacks = sum(s.feedbacks_received for s in self._stats.values())
            total_late = sum(s.late_feedbacks for s in self._stats.values())

            avg_loop = (
                sum(self.loop_times) / len(self.loop_times) if self.loop_times else 0
            )
            avg_send = (
                sum(self.send_times) / len(self.send_times) if self.send_times else 0
            )

            return {
                "elapsed_time": round(elapsed, 3),
                "total_cycles": self.total_cycles,
                "missed_deadlines": self.missed_deadlines,
                "missed_deadline_pct": round(
                    (self.missed_deadlines / self.total_cycles) * 100, 2
                )
                if self.total_cycles > 0
                else 0,
                "avg_loop_time_ms": round(avg_loop * 1000, 3),
                "avg_send_time_ms": round(avg_send * 1000, 3),
                "total_commands_sent": total_commands,
                "total_feedbacks_received": total_feedbacks,
                "total_late_feedbacks": total_late,
                "overall_feedback_rate_pct": round(
                    (total_feedbacks / total_commands) * 100, 2
                )
                if total_commands > 0
                else 0,
                "per_motor": {
                    mid: self._stats[mid].to_dict() for mid in self.motor_ids
                },
            }

    def print_summary(self) -> None:
        """Print formatted summary to console."""
        summary = self.get_summary()

        print("\n" + "=" * 60)
        print("MOTOR COMMUNICATION STATISTICS")
        print("=" * 60)

        print(f"\nOverall:")
        print(f"  Total cycles: {summary['total_cycles']}")
        print(f"  Elapsed time: {summary['elapsed_time']:.2f}s")
        print(f"  Avg loop time: {summary['avg_loop_time_ms']:.2f}ms")
        print(f"  Avg send time: {summary['avg_send_time_ms']:.2f}ms")
        print(
            f"  Missed deadlines: {summary['missed_deadlines']} "
            f"({summary['missed_deadline_pct']:.1f}%)"
        )

        print(f"\nFeedback Stats:")
        print(f"  Commands sent: {summary['total_commands_sent']}")
        print(f"  Feedbacks received: {summary['total_feedbacks_received']}")
        print(f"  Late feedbacks: {summary['total_late_feedbacks']}")
        print(f"  Overall feedback rate: {summary['overall_feedback_rate_pct']:.1f}%")

        print(f"\nPer-Motor Stats:")
        for mid in self.motor_ids:
            stats = summary["per_motor"][mid]
            print(f"\n  Motor {mid}:")
            print(
                f"    Sent: {stats['commands_sent']}, "
                f"Received: {stats['feedbacks_received']}"
            )
            print(f"    Late: {stats['late_feedbacks']}")
            print(f"    Feedback rate: {stats['feedback_rate_pct']:.1f}%")
            print(
                f"    Latency: avg={stats['avg_latency_ms']:.2f}ms, "
                f"min={stats['min_latency_ms']:.2f}ms, "
                f"max={stats['max_latency_ms']:.2f}ms"
            )

        print("\n" + "=" * 60)
