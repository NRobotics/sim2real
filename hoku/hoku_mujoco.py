"""
MuJoCo simulation with UDP interface for system identification testing.
Receives control commands via UDP, runs physics, and sends back state feedback.
"""

import argparse
import socket
import struct
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import mujoco
import mujoco.viewer

import config

# Real-time scheduling utilities (optional)
_sysid_path = Path(__file__).resolve().parent.parent / "system_identification" / "scripts"
if _sysid_path.exists() and str(_sysid_path) not in sys.path:
    sys.path.insert(0, str(_sysid_path))
try:
    from realtime import setup_realtime, get_rt_info
    RT_AVAILABLE = True
except ImportError:
    setup_realtime = None
    get_rt_info = None
    RT_AVAILABLE = False

# Add humanoid-protocol to path if needed
_proto_path = Path(__file__).resolve().parent.parent / "humanoid-protocol" / "python"
if _proto_path.exists() and str(_proto_path) not in sys.path:
    sys.path.insert(0, str(_proto_path))

# Import protocol types
from mujoco_controller import MsgType, CONTROL_MSG_SIZE
from humanoid_messages.can import (
    ConfigurationData,
    ControlData,
    FeedbackData,
)


@dataclass
class MotorState:
    """Per-motor state for simulation"""
    enabled: bool = False
    target: ControlData | None = None


class MujocoUDPServer:
    """
    MuJoCo simulation with UDP interface for external control.
    Physics runs in background thread, UDP responds immediately.
    """

    DEFAULT_RECV_PORT = 5000
    DEFAULT_SEND_PORT = 5001

    # Motor ID to MuJoCo actuator index mapping
    MOTOR_MAP = {
        0: 4,   # ankle_pitch_left
        1: 5,   # ankle_roll_left
        2: 3,   # knee_pitch_left
        3: 0,   # hip_pitch_left
        4: 1,   # hip_roll_left
        5: 2,   # hip_yaw_left
        6: 10,  # ankle_pitch_right
        7: 11,  # ankle_roll_right
        8: 9,   # knee_pitch_right
        9: 6,   # hip_pitch_right
        10: 7,  # hip_roll_right
        11: 8,  # hip_yaw_right
        12: 12,  # torso_yaw
        13: 13,  # torso_roll
    }

    def __init__(
        self,
        recv_port: int = DEFAULT_RECV_PORT,
        send_port: int = DEFAULT_SEND_PORT,
        client_host: str = "127.0.0.1",
        headless: bool = False,
        no_physics: bool = False,  # Skip physics for max UDP speed
    ):
        self.recv_port = recv_port
        self.send_port = send_port
        self.client_host = client_host
        self.headless = headless
        self.no_physics = no_physics

        # Load MuJoCo model
        self.model = mujoco.MjModel.from_xml_path(config.ROBOT_SCENE)
        self.model.opt.timestep = config.SIMULATE_DT

        # Disable self-collision if configured
        if not config.ENABLE_SELF_COLLISION:
            for i in range(self.model.ngeom):
                if self.model.geom_bodyid[i] != 0:
                    self.model.geom_conaffinity[i] = 0

        self.data = mujoco.MjData(self.model)

        # Motor states
        self.motor_states: dict[int, MotorState] = {}
        for motor_id in self.MOTOR_MAP:
            self.motor_states[motor_id] = MotorState()

        # UDP sockets
        self.recv_socket: socket.socket | None = None
        self.send_socket: socket.socket | None = None

        # Control
        self.running = False
        self.lock = threading.Lock()

        # Viewer (optional)
        self.viewer = None

        # Physics thread
        self.physics_thread: threading.Thread | None = None

    def _setup_sockets(self) -> None:
        """Initialize UDP sockets"""
        self.recv_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.recv_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.recv_socket.bind(("0.0.0.0", self.recv_port))
        self.recv_socket.setblocking(False)  # Non-blocking for max speed

        self.send_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        print(f"[Sim] UDP server: recv={self.recv_port}, send={self.send_port}")

    def _close_sockets(self) -> None:
        """Close UDP sockets"""
        if self.recv_socket:
            self.recv_socket.close()
        if self.send_socket:
            self.send_socket.close()

    def _send_feedback(self, motor_id: int, feedback: FeedbackData) -> None:
        """Send feedback to controller immediately"""
        if self.send_socket:
            data = struct.pack("<BB", MsgType.FEEDBACK, motor_id) + feedback.pack()
            self.send_socket.sendto(data, (self.client_host, self.send_port))

    def _send_config(self, motor_id: int) -> None:
        """Send configuration response to controller"""
        if self.send_socket:
            actuator_idx = self.MOTOR_MAP.get(motor_id, 0)

            # Get joint limits from model
            joint_id = self.model.actuator_trnid[actuator_idx, 0]
            pos_min = float(self.model.jnt_range[joint_id, 0])
            pos_max = float(self.model.jnt_range[joint_id, 1])

            config_data = ConfigurationData(
                can_id=motor_id,
                device_type=0,  # Simulated motor
                inverse_direction=False,
                endstop_alignment_inverse=False,
                endstop_alignment_skip=True,
                endstop_zero_offset=0.0,
                endstop_damping=0.1,
                endstop_position_min=pos_min,
                endstop_position_max=pos_max,
            )

            data = struct.pack("<BB", MsgType.CONFIG_RESPONSE, motor_id)
            data += config_data.pack()
            self.send_socket.sendto(data, (self.client_host, self.send_port))

    def _handle_control(self, motor_id: int, ctrl: ControlData) -> None:
        """Handle control command - apply and respond immediately"""
        if motor_id not in self.motor_states:
            return

        state = self.motor_states[motor_id]
        if not state.enabled:
            return

        state.target = ctrl

        # Apply control to MuJoCo
        actuator_idx = self.MOTOR_MAP.get(motor_id)
        if actuator_idx is not None and actuator_idx < self.model.nu:
            if self.no_physics:
                self.data.ctrl[actuator_idx] = ctrl.angle
                feedback = self._get_feedback_unlocked(motor_id)
            else:
                with self.lock:
                    self.data.ctrl[actuator_idx] = ctrl.angle
                feedback = self._get_feedback(motor_id)
        else:
            feedback = self._get_feedback(motor_id)

        # Send feedback immediately
        self._send_feedback(motor_id, feedback)

    def _handle_control_batch(self, data: bytes) -> None:
        """Handle batched control commands - apply all and respond with batch"""
        count = data[1]
        offset = 2
        feedbacks: list[tuple[int, FeedbackData]] = []
        motors_in_batch: list[int] = []

        # Parse and apply all controls
        # Use lock only if physics is running (otherwise no contention)
        if self.no_physics:
            self._process_batch_unlocked(data, count, offset, motors_in_batch, feedbacks)
        else:
            with self.lock:
                self._process_batch_unlocked(data, count, offset, motors_in_batch, feedbacks)

        # Send batched response
        self._send_feedback_batch(feedbacks)

    def _process_batch_unlocked(
        self, data: bytes, count: int, offset: int,
        motors_in_batch: list[int], feedbacks: list[tuple[int, FeedbackData]]
    ) -> None:
        """Process batch without lock (caller handles locking if needed)"""
        for _ in range(count):
            motor_id = data[offset]
            offset += 1
            ctrl = ControlData.unpack(
                data[offset:offset + ControlData.packed_size()]
            )
            offset += ControlData.packed_size()

            motors_in_batch.append(motor_id)

            if motor_id not in self.motor_states:
                continue
            state = self.motor_states[motor_id]
            if not state.enabled:
                continue

            state.target = ctrl
            actuator_idx = self.MOTOR_MAP.get(motor_id)
            if actuator_idx is not None and actuator_idx < self.model.nu:
                self.data.ctrl[actuator_idx] = ctrl.angle

        # Get feedback only for motors in batch
        for motor_id in motors_in_batch:
            fb = self._get_feedback_unlocked(motor_id)
            feedbacks.append((motor_id, fb))

    def _get_feedback_unlocked(self, motor_id: int) -> FeedbackData:
        """Get feedback without lock (caller must hold lock)"""
        actuator_idx = self.MOTOR_MAP.get(motor_id, 0)

        if actuator_idx < self.model.nu:
            joint_id = self.model.actuator_trnid[actuator_idx, 0]
            qpos_adr = self.model.jnt_qposadr[joint_id]
            qvel_adr = self.model.jnt_dofadr[joint_id]

            angle = float(self.data.qpos[qpos_adr])
            velocity = float(self.data.qvel[qvel_adr])
            effort = float(self.data.actuator_force[actuator_idx])
        else:
            angle = 0.0
            velocity = 0.0
            effort = 0.0

        return FeedbackData(
            angle=angle,
            velocity=velocity,
            effort=effort,
            voltage=24.0,
            temp_motor=35,
            temp_pcb=30,
            flags=0,
        )

    def _send_feedback_batch(
        self, feedbacks: list[tuple[int, FeedbackData]]
    ) -> None:
        """Send batched feedback to controller"""
        if not self.send_socket or not feedbacks:
            return
        # Pack: type(1) + count(1) + [motor_id(1) + feedback]...
        data = struct.pack("<BB", MsgType.FEEDBACK_BATCH, len(feedbacks))
        for motor_id, fb in feedbacks:
            data += struct.pack("<B", motor_id) + fb.pack()
        self.send_socket.sendto(data, (self.client_host, self.send_port))

    def _get_feedback(self, motor_id: int) -> FeedbackData:
        """Get current state for motor as FeedbackData"""
        actuator_idx = self.MOTOR_MAP.get(motor_id, 0)

        with self.lock:
            if actuator_idx < self.model.nu:
                joint_id = self.model.actuator_trnid[actuator_idx, 0]
                qpos_adr = self.model.jnt_qposadr[joint_id]
                qvel_adr = self.model.jnt_dofadr[joint_id]

                angle = float(self.data.qpos[qpos_adr])
                velocity = float(self.data.qvel[qvel_adr])
                effort = float(self.data.actuator_force[actuator_idx])
            else:
                angle = 0.0
                velocity = 0.0
                effort = 0.0

        return FeedbackData(
            angle=angle,
            velocity=velocity,
            effort=effort,
            voltage=24.0,
            temp_motor=35,
            temp_pcb=30,
            flags=0,
        )

    def _physics_loop(self) -> None:
        """Background thread for physics simulation"""
        print("[Sim] Physics thread started")
        while self.running:
            step_start = time.perf_counter()

            with self.lock:
                mujoco.mj_step(self.model, self.data)

            # Maintain physics rate
            elapsed = time.perf_counter() - step_start
            sleep_time = self.model.opt.timestep - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

        print("[Sim] Physics thread stopped")

    def _udp_loop(self) -> None:
        """Main loop for UDP command processing - runs as fast as possible"""
        import select
        while self.running:
            # Use select for efficient waiting with short timeout for clean shutdown
            try:
                ready, _, _ = select.select([self.recv_socket], [], [], 0.01)
            except (ValueError, OSError):
                break  # Socket closed
            if not ready:
                continue

            try:
                data, addr = self.recv_socket.recvfrom(1024)
                if len(data) < 2:
                    continue

                msg_type = data[0]
                motor_id = data[1]

                if msg_type == MsgType.CONTROL:
                    if len(data) >= CONTROL_MSG_SIZE:
                        ctrl = ControlData.unpack(data[2:])
                        self._handle_control(motor_id, ctrl)

                elif msg_type == MsgType.CONTROL_BATCH:
                    self._handle_control_batch(data)

                elif msg_type == MsgType.CONFIG_REQUEST:
                    self._send_config(motor_id)

                elif msg_type == MsgType.START_MOTOR:
                    if motor_id in self.motor_states:
                        self.motor_states[motor_id].enabled = True

                elif msg_type == MsgType.STOP_MOTOR:
                    if motor_id in self.motor_states:
                        self.motor_states[motor_id].enabled = False
                        self.motor_states[motor_id].target = None

                elif msg_type == MsgType.STOP_ALL:
                    for state in self.motor_states.values():
                        state.enabled = False
                        state.target = None

                elif msg_type == MsgType.PING:
                    # Ping requests feedback without requiring motor enabled
                    feedback = self._get_feedback(motor_id)
                    self._send_feedback(motor_id, feedback)

            except BlockingIOError:
                continue
            except Exception as e:
                if self.running:
                    print(f"[Sim] UDP error: {e}")

    def _viewer_loop(self) -> None:
        """Background thread for viewer sync"""
        print("[Sim] Viewer thread started")
        while self.running:
            if not self.viewer or not self.viewer.is_running():
                print("[Sim] Viewer closed")
                self.running = False
                break
            try:
                if not self.no_physics:
                    with self.lock:
                        self.viewer.sync()
                else:
                    self.viewer.sync()
            except Exception:
                break
            time.sleep(config.VIEWER_DT)
        print("[Sim] Viewer thread stopped")

    def run(self) -> None:
        """Main entry point - starts all threads"""
        self._setup_sockets()

        # Setup viewer if not headless
        if not self.headless:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self.viewer.cam.distance = 2.0
            self.viewer.cam.azimuth = 90
            self.viewer.cam.elevation = -20
            self.viewer.cam.lookat[0] = 0.0
            self.viewer.cam.lookat[1] = 0.0
            self.viewer.cam.lookat[2] = 0.5

        self.running = True

        # Start physics thread (unless disabled for max UDP speed)
        if not self.no_physics:
            self.physics_thread = threading.Thread(
                target=self._physics_loop, daemon=True
            )
            self.physics_thread.start()
        else:
            print("[Sim] Physics DISABLED - max UDP speed mode")

        # Start viewer thread if needed
        viewer_thread = None
        if not self.headless and self.viewer:
            viewer_thread = threading.Thread(target=self._viewer_loop, daemon=True)
            viewer_thread.start()

        print("[Sim] Running... Press Ctrl+C to stop")

        try:
            # UDP loop runs in main thread
            self._udp_loop()
        except KeyboardInterrupt:
            print("\n[Sim] Interrupted")
        finally:
            self._shutdown(viewer_thread)

    def _shutdown(self, viewer_thread: threading.Thread | None) -> None:
        """Clean shutdown of all resources"""
        self.running = False
        
        # Close sockets first to unblock UDP loop
        self._close_sockets()
        
        # Wait for threads with timeout
        if self.physics_thread and self.physics_thread.is_alive():
            self.physics_thread.join(timeout=0.5)
            if self.physics_thread.is_alive():
                print("[Sim] Warning: Physics thread did not stop")
        
        if viewer_thread and viewer_thread.is_alive():
            viewer_thread.join(timeout=0.5)
            if viewer_thread.is_alive():
                print("[Sim] Warning: Viewer thread did not stop")
        
        # Close viewer
        if self.viewer:
            try:
                self.viewer.close()
            except Exception:
                pass
        
        print("[Sim] Stopped")
        
        # Force exit to ensure clean termination
        import os
        os._exit(0)


def main():
    parser = argparse.ArgumentParser(
        description="MuJoCo simulation with UDP interface"
    )
    parser.add_argument(
        "--recv-port", type=int, default=5000,
        help="UDP port for receiving commands"
    )
    parser.add_argument(
        "--send-port", type=int, default=5001,
        help="UDP port for sending feedback"
    )
    parser.add_argument(
        "--client", type=str, default="127.0.0.1",
        help="Client IP for feedback"
    )
    parser.add_argument(
        "--headless", action="store_true",
        help="Run without viewer"
    )
    parser.add_argument(
        "--no-physics", action="store_true",
        help="Disable physics for max UDP throughput testing"
    )
    # Real-time scheduling options
    parser.add_argument(
        "--realtime", type=int, nargs="?", const=90, default=0,
        metavar="PRIORITY",
        help="Enable RT scheduling (SCHED_FIFO, default priority: 90)"
    )
    parser.add_argument(
        "--cpu", type=int, default=None, metavar="CORE",
        help="Pin to specific CPU core"
    )
    parser.add_argument(
        "--no-memlock", action="store_true",
        help="Disable memory locking"
    )
    parser.add_argument(
        "--rt-info", action="store_true",
        help="Show RT config and exit"
    )
    args = parser.parse_args()

    # Handle --rt-info
    if args.rt_info:
        if not RT_AVAILABLE:
            print("Real-time module not available")
            return
        info = get_rt_info()
        print("Current real-time configuration:")
        for k, v in info.items():
            print(f"  {k}: {v}")
        return

    # Setup real-time scheduling if requested
    if args.realtime > 0 or args.cpu is not None:
        if not RT_AVAILABLE:
            print("[Sim] Warning: RT module not available")
        else:
            setup_realtime(
                priority=args.realtime,
                cpu=args.cpu,
                lock_mem=not args.no_memlock and args.realtime > 0,
            )

    server = MujocoUDPServer(
        recv_port=args.recv_port,
        send_port=args.send_port,
        client_host=args.client,
        headless=args.headless,
        no_physics=args.no_physics,
    )
    server.run()


if __name__ == "__main__":
    main()
