"""
Threading utilities for real-time control loops.

Provides functions to configure thread affinity, priority, and naming
for proper operation alongside real-time scheduled main threads.
"""

from __future__ import annotations

import os
import sys
import threading
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Callable, Any

# Linux scheduling constants
SCHED_OTHER = 0
SCHED_FIFO = 1
SCHED_RR = 2


def is_linux() -> bool:
    """Check if running on Linux."""
    return sys.platform.startswith("linux")


def get_available_cpus() -> set[int]:
    """Get set of CPUs available to this process."""
    if not is_linux():
        return set()
    try:
        return os.sched_getaffinity(0)
    except (OSError, AttributeError):
        return set()


def get_other_cpus(exclude_cpu: int | None = None) -> set[int]:
    """Get CPUs other than the specified one."""
    cpus = get_available_cpus()
    if exclude_cpu is not None and exclude_cpu in cpus:
        cpus.discard(exclude_cpu)
    return cpus


def get_current_cpu_affinity() -> set[int]:
    """Get current process CPU affinity."""
    return get_available_cpus()


def set_thread_affinity(cpus: set[int] | list[int] | int) -> bool:
    """
    Set CPU affinity for the current thread.

    Args:
        cpus: Single CPU, list of CPUs, or set of CPUs

    Returns:
        True if successful
    """
    if not is_linux():
        return False

    if isinstance(cpus, int):
        cpus = {cpus}
    elif isinstance(cpus, list):
        cpus = set(cpus)

    try:
        os.sched_setaffinity(0, cpus)
        return True
    except (OSError, PermissionError) as e:
        print(f"[ThreadUtils] Failed to set affinity: {e}")
        return False


def set_thread_priority(priority: int, policy: int = SCHED_OTHER) -> bool:
    """
    Set scheduling priority for current thread.

    Args:
        priority: Priority level (1-99 for RT, 0 for SCHED_OTHER)
        policy: SCHED_OTHER, SCHED_FIFO, or SCHED_RR

    Returns:
        True if successful
    """
    if not is_linux():
        return False

    try:
        param = os.sched_param(priority)
        os.sched_setscheduler(0, policy, param)
        return True
    except (OSError, PermissionError) as e:
        print(f"[ThreadUtils] Failed to set priority: {e}")
        return False


def configure_receive_thread(
    main_cpu: int | None = None,
    priority: int = 0,
) -> None:
    """
    Configure current thread as a receive/IO thread.

    This should be called at the start of receive thread functions.
    It sets up the thread to run on a different CPU than the main
    control loop to avoid starvation under real-time scheduling.

    Args:
        main_cpu: CPU that main loop is running on (to avoid)
        priority: Thread priority (0 = normal, higher for RT)
    """
    if not is_linux():
        return

    # Try to run on a different CPU than main thread
    if main_cpu is not None:
        other_cpus = get_other_cpus(main_cpu)
        if other_cpus:
            set_thread_affinity(other_cpus)
            print(f"[ThreadUtils] Receive thread on CPUs: {other_cpus}")
        else:
            # Only one CPU available, can't separate
            print(f"[ThreadUtils] Warning: Cannot separate from CPU {main_cpu}")

    # Set priority if requested
    if priority > 0:
        set_thread_priority(priority, SCHED_FIFO)


class ConfiguredThread(threading.Thread):
    """
    Thread subclass that can be configured with CPU affinity and priority.

    The configuration is applied when the thread starts running.
    """

    def __init__(
        self,
        target: Callable[..., Any] | None = None,
        name: str | None = None,
        args: tuple = (),
        kwargs: dict | None = None,
        *,
        daemon: bool = True,
        cpu_affinity: set[int] | list[int] | int | None = None,
        avoid_cpu: int | None = None,
        priority: int = 0,
    ):
        """
        Initialize configured thread.

        Args:
            target: Function to run in thread
            name: Thread name
            args: Arguments for target
            kwargs: Keyword arguments for target
            daemon: Whether thread is daemon
            cpu_affinity: Specific CPUs to run on
            avoid_cpu: CPU to avoid (alternative to cpu_affinity)
            priority: RT priority (0 = normal)
        """
        super().__init__(target=target, name=name, args=args, daemon=daemon)
        self._kwargs = kwargs or {}
        self._cpu_affinity = cpu_affinity
        self._avoid_cpu = avoid_cpu
        self._priority = priority

    def run(self) -> None:
        """Run thread with configured affinity and priority."""
        self._apply_configuration()
        if self._target:  # type: ignore
            self._target(*self._args, **self._kwargs)  # type: ignore

    def _apply_configuration(self) -> None:
        """Apply CPU affinity and priority settings."""
        if not is_linux():
            return

        # Set CPU affinity
        if self._cpu_affinity is not None:
            set_thread_affinity(self._cpu_affinity)
        elif self._avoid_cpu is not None:
            other_cpus = get_other_cpus(self._avoid_cpu)
            if other_cpus:
                set_thread_affinity(other_cpus)

        # Set priority
        if self._priority > 0:
            set_thread_priority(self._priority, SCHED_FIFO)


def create_receive_thread(
    target: Callable[..., Any],
    name: str = "ReceiveThread",
    main_cpu: int | None = None,
    priority: int = 0,
    **kwargs,
) -> threading.Thread:
    """
    Create a thread configured for receiving data.

    This thread will be configured to run on a different CPU
    than the main control loop if possible.

    Args:
        target: Function to run
        name: Thread name
        main_cpu: CPU to avoid (where main loop runs)
        priority: RT priority (0 = normal)
        **kwargs: Additional args passed to target

    Returns:
        Configured Thread instance (not started)
    """
    return ConfiguredThread(
        target=target,
        name=name,
        daemon=True,
        avoid_cpu=main_cpu,
        priority=priority,
        kwargs=kwargs,
    )


def get_main_cpu() -> int | None:
    """
    Get the CPU the main thread is pinned to, if any.

    Returns:
        CPU number if pinned to single CPU, else None
    """
    if not is_linux():
        return None
    affinity = get_current_cpu_affinity()
    if len(affinity) == 1:
        return next(iter(affinity))
    return None

