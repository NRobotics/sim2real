#!/usr/bin/env python3
"""
CAN Speed Test - Profile the send/receive loop for 6 motors
Tests how fast we can send kinematics to all 6 motors and receive feedback

Supports multiple interfaces:
- SocketCAN (Linux native CAN interface)
- USBTingo (USB to CAN adapter)
- Remote (WebSocket-based remote CAN interface)

Optimizations:
- Pre-allocated arrays for feedback storage
- Busy-wait loop instead of event signaling
- Minimal callback overhead
- Reduced timeout (2ms instead of 10ms)

Performance tips:
- Use --cpu-affinity to pin to a specific CPU core
- Run with real-time priority: sudo chrt -f 80 python3 speed_test.py
- Reduce system load (close other applications)
- Consider PREEMPT_RT kernel for even better performance
"""

import sys
import time
import threading
import math
import statistics
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Optional

import builtins

# Save original print
_print = builtins.print

# Disable prints
builtins.print = lambda *args, **kwargs: None

# Add the python directory to sys.path
python_dir = Path(__file__).parent / "ext" / "humanoid-protocol" / "python"
sys.path.insert(0, str(python_dir))

from humanoid_messages.can import MotorCANController, ControlData, FeedbackData

# Configuration
CAN_INTERFACE = "socketcan"  # or "pcan" for PCAN
CAN_CHANNEL = "can0"         # can0 or can1
CAN_BITRATE = 1000000        # 1 MHz
DATA_BITRATE = 5000000       # 5 MHz
NUM_MOTORS = 6

@dataclass
class CycleStats:
    """Statistics for one complete cycle"""
    send_time: float
    receive_time: float
    total_time: float
    responses_received: int


class CANSpeedTest:
    """Profile CAN communication speed with multiple motors"""
    
    def __init__(self, interface: str, channel: str = ""):
        self.interface = interface
        self.channel = channel
        self.controller: Optional[MotorCANController] = None
        
        # Feedback tracking - optimized with pre-allocated arrays
        self.feedback_lock = threading.Lock()
        self.feedback_received = [None] * NUM_MOTORS  # Pre-allocated array
        self.feedback_count = 0
        
        # Statistics
        self.cycle_stats = []
        self.running = False
        
    def setup(self) -> bool:
        """Initialize CAN controller"""
        try:
            # Determine interface type and print appropriate info
            is_websocket = self.interface.startswith("ws://") or self.interface.startswith("wss://")
            
            if is_websocket:
                print(f"Initializing remote CAN FD via WebSocket: {self.interface}")
            elif self.interface == "usbtingo":
                print(f"Initializing CAN FD via USBTingo...")
            elif self.channel:
                print(f"Initializing CAN FD on {self.channel}...")
            else:
                print(f"Initializing CAN FD with {self.interface}...")
            
            print(f"  Nominal bitrate: {CAN_BITRATE/1e6:.1f} Mbps")
            print(f"  Data bitrate: {DATA_BITRATE/1e6:.1f} Mbps")
            
            # Initialize controller based on interface type
            if self.channel:
                # Interface with channel (socketcan, pcan, etc.)
                self.controller = MotorCANController(
                    interface=self.interface,
                    channel=self.channel,
                    bitrate=CAN_BITRATE,
                    data_bitrate=DATA_BITRATE,
                    fd=True,
                    receive_own_messages=False
                )
            else:
                # Interface without channel (usbtingo, websocket)
                self.controller = MotorCANController(
                    interface=self.interface,
                    bitrate=CAN_BITRATE,
                    data_bitrate=DATA_BITRATE,
                    fd=True,
                    receive_own_messages=False
                )
            
            self.controller.start()
            
            # Register feedback callbacks for all motors
            for motor_id in range(NUM_MOTORS):
                self.controller.set_feedback_callback(
                    motor_id,
                    lambda can_id, data, mid=motor_id: self._on_feedback(mid, data)
                )
            
            print("✓ CAN FD initialized successfully\n")
            return True
            
        except Exception as e:
            print(f"✗ Failed to initialize CAN: {e}")
            
            is_websocket = self.interface.startswith("ws://") or self.interface.startswith("wss://")
            
            if self.interface == "socketcan" and self.channel:
                print(f"\nSetup instructions for socketcan:")
                print(f"  sudo ip link set {self.channel} type can bitrate {CAN_BITRATE} dbitrate {DATA_BITRATE} fd on")
                print(f"  sudo ip link set {self.channel} up")
            elif self.interface == "usbtingo":
                print(f"\nMake sure USBTingo device is connected and drivers are installed")
            elif is_websocket:
                print(f"\nMake sure the remote CAN server is running and accessible at {self.interface}")
            return False
    
    def _on_feedback(self, motor_id: int, data: FeedbackData):
        """Callback when feedback is received from a motor - optimized for speed"""
        # Minimize work in callback - just store the data
        self.feedback_received[motor_id] = data
        self.feedback_count += 1
    
    def send_kinematics_all(self, test_data: Dict[int, ControlData]) -> float:
        """
        Send kinematics to all motors
        Returns: Time taken to send all messages
        """
        start = time.perf_counter()
        
        for motor_id in range(NUM_MOTORS):
            self.controller.send_kinematics_for_motor(motor_id, test_data[motor_id])
        
        return time.perf_counter() - start
    
    def wait_for_all_feedback(self, timeout: float = 0.002) -> tuple[int, float]:
        """
        Wait for feedback from all motors - optimized with busy-wait
        Returns: (number of responses received, time taken)
        """
        start = time.perf_counter()
        deadline = start + timeout
        
        # Tight busy-wait loop - much faster than Event.wait()
        while self.feedback_count < NUM_MOTORS and time.perf_counter() < deadline:
            pass  # Busy wait
        
        num_received = self.feedback_count
        elapsed = time.perf_counter() - start
        return num_received, elapsed
    
    def run_cycle(self, test_data: Dict[int, ControlData]) -> CycleStats:
        """
        Run one complete cycle:
        1. Send kinematics to all motors
        2. Wait for all feedback
        
        Returns: Statistics for this cycle
        """
        cycle_start = time.perf_counter()
        
        # Clear previous feedback - fast array reset
        self.feedback_count = 0
        for i in range(NUM_MOTORS):
            self.feedback_received[i] = None
        
        # Send phase
        send_time = self.send_kinematics_all(test_data)
        
        # Receive phase
        num_received, receive_time = self.wait_for_all_feedback(timeout=0.002)
        
        total_time = time.perf_counter() - cycle_start
        
        return CycleStats(
            send_time=send_time,
            receive_time=receive_time,
            total_time=total_time,
            responses_received=num_received
        )
    
    def run_test(self, duration: float = 10.0, target_hz: float = 1000.0):
        """
        Run speed test for specified duration
        
        Args:
            duration: Test duration in seconds
            target_hz: Target frequency to attempt
        """
        print("="*70)
        print("CAN Speed Test - 6 Motor Profiling")
        print("="*70)
        
        # Print interface info
        is_websocket = self.interface.startswith("ws://") or self.interface.startswith("wss://")
        
        if is_websocket:
            print(f"Interface: Remote (WebSocket)")
            print(f"URI: {self.interface}")
        elif self.interface == "usbtingo":
            print(f"Interface: USBTingo")
        elif self.channel:
            print(f"Interface: {self.interface}")
            print(f"Channel: {self.channel}")
        else:
            print(f"Interface: {self.interface}")
        
        print(f"Target frequency: {target_hz} Hz ({1000.0/target_hz:.3f} ms period)")
        print(f"Test duration: {duration} seconds")
        print(f"Number of motors: {NUM_MOTORS}")
        print("="*70)
        print()
        
        # Create test control data (zero position, moderate stiffness)
        test_data = {}
        for motor_id in range(NUM_MOTORS):
            test_data[motor_id] = ControlData(
                angle=0.0,
                velocity=0.0,
                effort=0.0,
                stiffness=5.0,
                damping=0.5
            )
        
        self.running = True
        self.cycle_stats = []
        
        start_time = time.perf_counter()
        next_tick = start_time
        target_period = 1.0 / target_hz
        
        cycle_count = 0
        last_print_time = start_time
        
        print("Running test... (Ctrl+C to stop)\n")
        
        try:
            while self.running and (time.perf_counter() - start_time) < duration:
                # Run one cycle
                stats = self.run_cycle(test_data)
                self.cycle_stats.append(stats)
                cycle_count += 1
                
                # Print progress every second
                now = time.perf_counter()
                if now - last_print_time >= 1.0:
                    elapsed = now - start_time
                    actual_hz = cycle_count / elapsed
                    recent_cycles = self.cycle_stats[-int(actual_hz):]
                    avg_cycle_time = statistics.mean([s.total_time for s in recent_cycles]) * 1000
                    
                    print(f"[{elapsed:5.1f}s] Cycles: {cycle_count:5d} | "
                          f"Freq: {actual_hz:6.1f} Hz | "
                          f"Avg cycle: {avg_cycle_time:6.3f} ms | "
                          f"Responses: {stats.responses_received}/{NUM_MOTORS}")
                    
                    last_print_time = now
                
                # Sleep until next tick (if trying to maintain target rate)
                next_tick += target_period
                sleep_time = next_tick - time.perf_counter()
                if sleep_time > 0:
                    time.sleep(sleep_time)
                
        except KeyboardInterrupt:
            print("\n\nTest interrupted by user")
        
        finally:
            self.running = False
            self.print_results()
    
    def print_results(self):
        builtins.print = _print

        """Print detailed test results"""
        if not self.cycle_stats:
            print("\nNo data collected")
            return
        
        print("\n" + "="*70)
        print("Test Results")
        print("="*70)
        
        # Extract timing data
        total_times = [s.total_time * 1000 for s in self.cycle_stats]  # Convert to ms
        send_times = [s.send_time * 1000 for s in self.cycle_stats]
        receive_times = [s.receive_time * 1000 for s in self.cycle_stats]
        
        # Calculate statistics
        total_cycles = len(self.cycle_stats)
        total_responses = sum(s.responses_received for s in self.cycle_stats)
        expected_responses = total_cycles * NUM_MOTORS
        response_rate = (total_responses / expected_responses) * 100 if expected_responses > 0 else 0
        
        print(f"\nCycle Statistics:")
        print(f"  Total cycles completed: {total_cycles}")
        print(f"  Expected responses: {expected_responses}")
        print(f"  Received responses: {total_responses} ({response_rate:.2f}%)")
        
        # Cycles with complete responses
        complete_cycles = sum(1 for s in self.cycle_stats if s.responses_received == NUM_MOTORS)
        complete_rate = (complete_cycles / total_cycles) * 100
        print(f"  Complete cycles (6/6): {complete_cycles} ({complete_rate:.2f}%)")
        
        print(f"\nTiming Analysis:")
        print(f"  Total Cycle Time:")
        print(f"    Mean:   {statistics.mean(total_times):7.3f} ms")
        print(f"    Median: {statistics.median(total_times):7.3f} ms")
        print(f"    Min:    {min(total_times):7.3f} ms")
        print(f"    Max:    {max(total_times):7.3f} ms")
        if len(total_times) > 1:
            print(f"    StdDev: {statistics.stdev(total_times):7.3f} ms")
        
        print(f"\n  Send Phase (6 messages):")
        print(f"    Mean:   {statistics.mean(send_times):7.3f} ms")
        print(f"    Max:    {max(send_times):7.3f} ms")
        
        print(f"\n  Receive Phase (wait for 6 responses):")
        print(f"    Mean:   {statistics.mean(receive_times):7.3f} ms")
        print(f"    Max:    {max(receive_times):7.3f} ms")
        
        # Calculate achievable frequency
        avg_cycle_time_s = statistics.mean([s.total_time for s in self.cycle_stats])
        max_theoretical_hz = 1.0 / avg_cycle_time_s if avg_cycle_time_s > 0 else 0
        
        print(f"\nFrequency Analysis:")
        print(f"  Average cycle time: {avg_cycle_time_s*1000:.3f} ms")
        print(f"  Max theoretical frequency: {max_theoretical_hz:.1f} Hz")
        
        # Check performance at different target frequencies
        targets = [1000, 500, 250, 100]
        print(f"\n  Performance at target frequencies:")
        for target in targets:
            target_period_ms = 1000.0 / target
            cycles_within = sum(1 for t in total_times if t <= target_period_ms)
            success_rate = (cycles_within / total_cycles) * 100
            status = "✓" if success_rate >= 95 else "✗"
            print(f"    {status} {target:4d} Hz ({target_period_ms:6.3f} ms): "
                  f"{cycles_within}/{total_cycles} cycles ({success_rate:5.1f}%)")
        
        # Bus utilization estimate
        control_size = ControlData.packed_size()
        feedback_size = FeedbackData.packed_size()
        bytes_per_cycle = (control_size + feedback_size) * NUM_MOTORS
        bits_per_cycle = bytes_per_cycle * 8
        
        # Account for CAN FD overhead (roughly 30% for frame structure, CRC, etc.)
        overhead_factor = 1.3
        bits_per_cycle_with_overhead = bits_per_cycle * overhead_factor
        
        data_rate_used = (bits_per_cycle_with_overhead / avg_cycle_time_s) if avg_cycle_time_s > 0 else 0
        
        print(f"\nBus Utilization:")
        print(f"  Payload per cycle: {bytes_per_cycle} bytes ({bits_per_cycle} bits)")
        print(f"  Est. with overhead: {bits_per_cycle_with_overhead:.0f} bits")
        print(f"  Effective data rate: {data_rate_used/1e6:.2f} Mbps")
        print(f"  Data phase capacity: {DATA_BITRATE/1e6:.2f} Mbps")
        print(f"  Estimated utilization: {(data_rate_used/DATA_BITRATE)*100:.1f}%")
        
        print("\n" + "="*70)
        
        # Summary recommendation
        print("\nSummary:")
        if complete_rate >= 95 and max_theoretical_hz >= 1000:
            print("  ✓ System can reliably achieve 1 kHz with all 6 motors")
        elif max_theoretical_hz >= 500:
            print(f"  ⚠ System can achieve ~{int(max_theoretical_hz)} Hz")
            print("    Consider optimizing for 1 kHz target")
        else:
            print(f"  ✗ System limited to ~{int(max_theoretical_hz)} Hz")
            print("    Hardware or configuration changes may be needed")
        print()
    
    def cleanup(self):
        """Cleanup CAN controller"""
        if self.controller:
            try:
                self.controller.stop()
                print("CAN controller stopped")
            except Exception as e:
                print(f"Error stopping controller: {e}")


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='CAN Speed Test for 6 motors',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # SocketCAN on can0
  python3 speed_test.py --interface socketcan --channel can0
  
  # USBTingo interface
  python3 speed_test.py --interface usbtingo
  
  # Remote interface via WebSocket (URI as interface)
  python3 speed_test.py --interface ws://192.168.1.100:8080
  
  # Or with explicit channel for socketcan
  python3 speed_test.py --channel can1
  
  # Pin to CPU core 2 for better real-time performance
  python3 speed_test.py --channel can0 --cpu-affinity 2
        """
    )
    
    parser.add_argument('--interface', default='socketcan', 
                       help='CAN interface: socketcan, usbtingo, pcan, or ws://... for WebSocket (default: socketcan)')
    parser.add_argument('--channel', default='can0',
                       help='CAN channel for socketcan/pcan (default: can0, not used for usbtingo/websocket)')
    parser.add_argument('--duration', type=float, default=10.0,
                       help='Test duration in seconds (default: 10)')
    parser.add_argument('--target-hz', type=float, default=1000.0,
                       help='Target frequency in Hz (default: 1000)')
    parser.add_argument('--cpu-affinity', type=int, default=None,
                       help='Pin to specific CPU core (0-N) for better performance')
    
    args = parser.parse_args()
    
    # Set CPU affinity if requested
    if args.cpu_affinity is not None:
        try:
            import os
            os.sched_setaffinity(0, {args.cpu_affinity})
            print(f"Pinned to CPU core {args.cpu_affinity}")
        except Exception as e:
            print(f"Warning: Could not set CPU affinity: {e}")
    
    # Determine if channel should be used
    is_websocket = args.interface.startswith("ws://") or args.interface.startswith("wss://")
    channel = "" if (args.interface == "usbtingo" or is_websocket) else args.channel
    
    # Create and run test
    test = CANSpeedTest(args.interface, channel=channel)
    
    if not test.setup():
        sys.exit(1)
    
    try:
        test.run_test(duration=args.duration, target_hz=args.target_hz)
    except Exception as e:
        print(f"\nError during test: {e}")
        import traceback
        traceback.print_exc()
    finally:
        test.cleanup()


if __name__ == "__main__":
    main()
