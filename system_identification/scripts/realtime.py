#!/usr/bin/env python3
"""
Real-time scheduling utilities for Linux.

Provides functions to enable real-time scheduling, CPU affinity,
and memory locking for deterministic timing in control loops.

Usage:
    from realtime import setup_realtime
    
    # Enable RT scheduling with priority 90 on CPU 3
    setup_realtime(priority=90, cpu=3)
    
Requirements:
    - Linux kernel (PREEMPT_RT recommended for best results)
    - Root privileges OR appropriate capabilities (CAP_SYS_NICE, CAP_IPC_LOCK)
    
To run without root, grant capabilities to Python:
    sudo setcap 'cap_sys_nice,cap_ipc_lock+ep' $(which python3)
"""

from __future__ import annotations

import ctypes
import os
import sys
from typing import Optional


# Linux scheduling constants (from sched.h)
SCHED_OTHER = 0  # Default time-sharing
SCHED_FIFO = 1   # Real-time FIFO
SCHED_RR = 2     # Real-time round-robin

# Memory lock constants (from sys/mman.h)
MCL_CURRENT = 1  # Lock current pages
MCL_FUTURE = 2   # Lock future pages


def _is_linux() -> bool:
    """Check if running on Linux."""
    return sys.platform.startswith('linux')


def set_cpu_affinity(cpu: int) -> bool:
    """
    Pin the current process to a specific CPU core.
    
    Args:
        cpu: CPU core number (0-indexed)
        
    Returns:
        True if successful, False otherwise
    """
    if not _is_linux():
        print(f"[RT] CPU affinity not supported on {sys.platform}")
        return False
    
    try:
        os.sched_setaffinity(0, {cpu})
        actual = os.sched_getaffinity(0)
        if actual == {cpu}:
            print(f"[RT] CPU affinity set to core {cpu}")
            return True
        else:
            print(f"[RT] CPU affinity mismatch: requested {cpu}, got {actual}")
            return False
    except (OSError, PermissionError) as e:
        print(f"[RT] Failed to set CPU affinity: {e}")
        return False


def set_realtime_priority(priority: int = 90, policy: int = SCHED_FIFO) -> bool:
    """
    Set real-time scheduling priority for the current process.
    
    Args:
        priority: RT priority (1-99, higher = more priority)
        policy: SCHED_FIFO (default) or SCHED_RR
        
    Returns:
        True if successful, False otherwise
        
    Note:
        Requires root or CAP_SYS_NICE capability.
        To grant capability: sudo setcap 'cap_sys_nice+ep' $(which python3)
    """
    if not _is_linux():
        print(f"[RT] Real-time scheduling not supported on {sys.platform}")
        return False
    
    # Clamp priority to valid range
    priority = max(1, min(99, priority))
    
    try:
        # Use os.sched_setscheduler (Python 3.3+)
        param = os.sched_param(priority)
        os.sched_setscheduler(0, policy, param)
        
        # Verify
        actual_policy = os.sched_getscheduler(0)
        actual_param = os.sched_getparam(0)
        
        policy_names = {SCHED_OTHER: 'OTHER', SCHED_FIFO: 'FIFO', SCHED_RR: 'RR'}
        if actual_policy == policy and actual_param.sched_priority == priority:
            print(f"[RT] Scheduler set to SCHED_{policy_names.get(policy, policy)} priority {priority}")
            return True
        else:
            print(f"[RT] Scheduler mismatch: got policy={actual_policy} priority={actual_param.sched_priority}")
            return False
            
    except (OSError, PermissionError) as e:
        print(f"[RT] Failed to set real-time priority: {e}")
        print(f"[RT] Try: sudo setcap 'cap_sys_nice+ep' $(which python3)")
        return False


def lock_memory() -> bool:
    """
    Lock all current and future memory pages to prevent page faults.
    
    This prevents the kernel from swapping out pages, which could
    cause unpredictable latency spikes during control loops.
    
    Returns:
        True if successful, False otherwise
        
    Note:
        Requires root or CAP_IPC_LOCK capability.
        To grant: sudo setcap 'cap_ipc_lock+ep' $(which python3)
    """
    if not _is_linux():
        print(f"[RT] Memory locking not supported on {sys.platform}")
        return False
    
    try:
        libc = ctypes.CDLL('libc.so.6', use_errno=True)
        result = libc.mlockall(MCL_CURRENT | MCL_FUTURE)
        
        if result == 0:
            print("[RT] Memory locked (mlockall)")
            return True
        else:
            errno = ctypes.get_errno()
            print(f"[RT] mlockall failed with errno {errno}")
            return False
            
    except (OSError, PermissionError) as e:
        print(f"[RT] Failed to lock memory: {e}")
        print(f"[RT] Try: sudo setcap 'cap_ipc_lock+ep' $(which python3)")
        return False


def disable_kernel_throttling() -> bool:
    """
    Disable CPU frequency scaling for consistent performance.
    
    Sets the CPU governor to 'performance' mode, which keeps
    CPUs at maximum frequency. This reduces latency variability
    from frequency scaling.
    
    Returns:
        True if successful, False otherwise
        
    Note:
        Requires root privileges. Run with sudo or as root.
    """
    if not _is_linux():
        return False
    
    try:
        # Try to set performance governor for all CPUs
        import glob
        governor_files = glob.glob('/sys/devices/system/cpu/cpu*/cpufreq/scaling_governor')
        
        if not governor_files:
            print("[RT] No CPU frequency scaling found (might be disabled in BIOS)")
            return False
        
        for gov_file in governor_files:
            try:
                with open(gov_file, 'w') as f:
                    f.write('performance')
            except PermissionError:
                print(f"[RT] Cannot set governor (need root): {gov_file}")
                return False
        
        print(f"[RT] Set {len(governor_files)} CPUs to performance governor")
        return True
        
    except Exception as e:
        print(f"[RT] Failed to set CPU governor: {e}")
        return False


def setup_realtime(
    priority: int = 90,
    cpu: Optional[int] = None,
    lock_mem: bool = True,
    verbose: bool = True
) -> dict:
    """
    Configure the current process for real-time operation.
    
    This is the main entry point for real-time setup. It applies
    multiple optimizations for deterministic timing.
    
    Args:
        priority: RT scheduling priority (1-99, 0 to skip)
        cpu: CPU core to pin to (None to skip)
        lock_mem: Whether to lock memory pages
        verbose: Print status messages
        
    Returns:
        Dict with status of each optimization:
        {
            'priority': bool,  # RT priority set
            'cpu': bool,       # CPU affinity set
            'memory': bool,    # Memory locked
        }
        
    Example:
        # Full RT setup on CPU 3
        status = setup_realtime(priority=90, cpu=3)
        
        # Just CPU pinning, no RT priority
        status = setup_realtime(priority=0, cpu=2)
    """
    if not _is_linux():
        if verbose:
            print(f"[RT] Real-time features not available on {sys.platform}")
        return {'priority': False, 'cpu': False, 'memory': False}
    
    if verbose:
        print("[RT] Setting up real-time environment...")
    
    results = {}
    
    # 1. Lock memory first (before allocating more)
    if lock_mem:
        results['memory'] = lock_memory()
    else:
        results['memory'] = False
    
    # 2. Set CPU affinity
    if cpu is not None:
        results['cpu'] = set_cpu_affinity(cpu)
    else:
        results['cpu'] = False
    
    # 3. Set RT priority (do this last)
    if priority > 0:
        results['priority'] = set_realtime_priority(priority)
    else:
        results['priority'] = False
    
    if verbose:
        success_count = sum(results.values())
        total = len([k for k, v in [('priority', priority > 0), ('cpu', cpu is not None), ('memory', lock_mem)] if v])
        if total > 0:
            print(f"[RT] Setup complete: {success_count}/{total} optimizations applied")
    
    return results


def get_rt_info() -> dict:
    """
    Get information about the current real-time configuration.
    
    Returns:
        Dict with current RT settings
    """
    if not _is_linux():
        return {'platform': sys.platform, 'supported': False}
    
    info = {
        'platform': sys.platform,
        'supported': True,
    }
    
    try:
        info['affinity'] = list(os.sched_getaffinity(0))
        info['scheduler'] = os.sched_getscheduler(0)
        info['priority'] = os.sched_getparam(0).sched_priority
        
        policy_names = {SCHED_OTHER: 'OTHER', SCHED_FIFO: 'FIFO', SCHED_RR: 'RR'}
        info['scheduler_name'] = policy_names.get(info['scheduler'], str(info['scheduler']))
    except Exception as e:
        info['error'] = str(e)
    
    return info


if __name__ == "__main__":
    # Test the RT setup
    import argparse
    
    parser = argparse.ArgumentParser(description="Test real-time setup")
    parser.add_argument("--priority", type=int, default=90, help="RT priority (1-99)")
    parser.add_argument("--cpu", type=int, default=None, help="CPU to pin to")
    parser.add_argument("--no-lock", action="store_true", help="Skip memory locking")
    parser.add_argument("--info", action="store_true", help="Just show current RT info")
    args = parser.parse_args()
    
    if args.info:
        info = get_rt_info()
        print("Current RT configuration:")
        for k, v in info.items():
            print(f"  {k}: {v}")
    else:
        results = setup_realtime(
            priority=args.priority,
            cpu=args.cpu,
            lock_mem=not args.no_lock
        )
        
        print("\nResults:")
        for k, v in results.items():
            status = "✓" if v else "✗"
            print(f"  {status} {k}")
        
        # Show final config
        print("\nFinal configuration:")
        info = get_rt_info()
        for k, v in info.items():
            print(f"  {k}: {v}")

