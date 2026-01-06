"""
MuJoCo Motor Controller - UDP interface matching MotorCANController API.
Communicates with MuJoCo simulation via UDP for system identification testing.
"""

import socket
import struct
import sys
import threading
from collections.abc import Callable
from enum import IntEnum
from pathlib import Path

# Add humanoid-protocol to path if needed
_proto_path = Path(__file__).resolve().parent.parent / "humanoid-protocol" / "python"
if _proto_path.exists() and str(_proto_path) not in sys.path:
    sys.path.insert(0, str(_proto_path))

from humanoid_messages.can import (
    ConfigurationData,
    ControlData,
    FeedbackData,
)


class MsgType(IntEnum):
    """UDP message types"""
    CONTROL = 1
    CONFIG_REQUEST = 2
    CONFIG_RESPONSE = 3
    START_MOTOR = 4
    STOP_MOTOR = 5
    STOP_ALL = 6
    FEEDBACK = 7
    CONTROL_BATCH = 8  # Batched control for multiple motors
    FEEDBACK_BATCH = 9  # Batched feedback for multiple motors


# Message sizes
CONTROL_MSG_SIZE = 2 + ControlData.packed_size()  # type + id + data
FEEDBACK_MSG_SIZE = 2 + FeedbackData.packed_size()
CONFIG_MSG_SIZE = 2 + ConfigurationData.packed_size()
SINGLE_MOTOR_CTRL_SIZE = 1 + ControlData.packed_size()  # id + data
SINGLE_MOTOR_FB_SIZE = 1 + FeedbackData.packed_size()  # id + data


def pack_control_msg(motor_id: int, ctrl: ControlData) -> bytes:
    """Pack control command for UDP transmission"""
    return struct.pack("<BB", MsgType.CONTROL, motor_id) + ctrl.pack()


def pack_config_request(motor_id: int) -> bytes:
    """Pack config request message"""
    return struct.pack("<BB", MsgType.CONFIG_REQUEST, motor_id)


def pack_start_motor(motor_id: int) -> bytes:
    return struct.pack("<BB", MsgType.START_MOTOR, motor_id)


def pack_stop_motor(motor_id: int) -> bytes:
    return struct.pack("<BB", MsgType.STOP_MOTOR, motor_id)


def pack_stop_all() -> bytes:
    return struct.pack("<BB", MsgType.STOP_ALL, 0)


def unpack_feedback_msg(data: bytes) -> tuple[int, FeedbackData]:
    """Unpack feedback message from simulation"""
    msg_type, motor_id = struct.unpack("<BB", data[:2])
    if msg_type != MsgType.FEEDBACK:
        raise ValueError(f"Expected FEEDBACK message, got {msg_type}")
    feedback = FeedbackData.unpack(data[2:])
    return motor_id, feedback


def unpack_config_msg(data: bytes) -> tuple[int, ConfigurationData]:
    """Unpack config response from simulation"""
    msg_type, motor_id = struct.unpack("<BB", data[:2])
    if msg_type != MsgType.CONFIG_RESPONSE:
        raise ValueError(f"Expected CONFIG_RESPONSE message, got {msg_type}")
    config = ConfigurationData.unpack(data[2:])
    return motor_id, config


def pack_control_batch(controls: dict[int, ControlData]) -> bytes:
    """Pack multiple motor controls into one packet"""
    data = struct.pack("<BB", MsgType.CONTROL_BATCH, len(controls))
    for motor_id, ctrl in controls.items():
        data += struct.pack("<B", motor_id) + ctrl.pack()
    return data


def unpack_feedback_batch(data: bytes) -> dict[int, FeedbackData]:
    """Unpack batched feedback from simulation"""
    msg_type, count = struct.unpack("<BB", data[:2])
    if msg_type != MsgType.FEEDBACK_BATCH:
        raise ValueError(f"Expected FEEDBACK_BATCH, got {msg_type}")
    result = {}
    offset = 2
    for _ in range(count):
        motor_id = data[offset]
        offset += 1
        feedback = FeedbackData.unpack(data[offset:offset + FeedbackData.packed_size()])
        offset += FeedbackData.packed_size()
        result[motor_id] = feedback
    return result


class MujocoMotorController:
    """
    UDP-based motor controller for MuJoCo simulation.
    API compatible with MotorCANController for drop-in replacement.
    """

    DEFAULT_SIM_HOST = "127.0.0.1"
    DEFAULT_SEND_PORT = 5000  # Send commands to sim
    DEFAULT_RECV_PORT = 5001  # Receive feedback from sim

    def __init__(
        self,
        sim_host: str = DEFAULT_SIM_HOST,
        send_port: int = DEFAULT_SEND_PORT,
        recv_port: int = DEFAULT_RECV_PORT,
        **kwargs  # Accept extra args for CAN interface compatibility
    ):
        self.sim_host = sim_host
        self.send_port = send_port
        self.recv_port = recv_port

        self.send_socket: socket.socket | None = None
        self.recv_socket: socket.socket | None = None
        self.running = False
        self.receive_thread: threading.Thread | None = None

        # Callbacks - same as MotorCANController
        self.feedback_callbacks: dict[int, Callable] = {}
        self.config_callbacks: dict[int, Callable] = {}
        self.control_ack_callbacks: dict[int, Callable] = {}

    def start(self) -> bool:
        """Start UDP communication"""
        if self.running:
            return True

        try:
            # Socket for sending commands to sim
            self.send_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.send_socket.setblocking(False)

            # Socket for receiving feedback from sim
            self.recv_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.recv_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.recv_socket.bind(("0.0.0.0", self.recv_port))
            self.recv_socket.settimeout(0.001)  # 1ms timeout for fast response

            self.running = True

            # Start receive thread
            self.receive_thread = threading.Thread(
                target=self._receive_loop, daemon=True
            )
            self.receive_thread.start()

            print(f"[MujocoController] Started - send:{self.send_port} recv:{self.recv_port}")
            return True

        except Exception as e:
            print(f"[MujocoController] Failed to start: {e}")
            self.stop()
            raise

    def stop(self) -> None:
        """Stop UDP communication"""
        self.running = False
        if self.receive_thread and self.receive_thread.is_alive():
            self.receive_thread.join(timeout=1.0)
        if self.send_socket:
            self.send_socket.close()
            self.send_socket = None
        if self.recv_socket:
            self.recv_socket.close()
            self.recv_socket = None
        print("[MujocoController] Stopped")

    def _receive_loop(self) -> None:
        """Background thread for receiving feedback"""
        while self.running and self.recv_socket:
            try:
                data, _ = self.recv_socket.recvfrom(1024)
                if len(data) >= 2:
                    self._handle_message(data)
            except socket.timeout:
                continue
            except OSError:
                if self.running:
                    break
            except Exception as e:
                if self.running:
                    print(f"[MujocoController] Receive error: {e}")

    def _handle_message(self, data: bytes) -> None:
        """Handle received message from simulation"""
        msg_type = data[0]

        if msg_type == MsgType.FEEDBACK:
            motor_id, feedback = unpack_feedback_msg(data)
            if motor_id in self.feedback_callbacks:
                self.feedback_callbacks[motor_id](motor_id, feedback)

        elif msg_type == MsgType.FEEDBACK_BATCH:
            feedbacks = unpack_feedback_batch(data)
            for motor_id, feedback in feedbacks.items():
                if motor_id in self.feedback_callbacks:
                    self.feedback_callbacks[motor_id](motor_id, feedback)

        elif msg_type == MsgType.CONFIG_RESPONSE:
            motor_id, config = unpack_config_msg(data)
            if motor_id in self.config_callbacks:
                self.config_callbacks[motor_id](motor_id, config)

    def _send(self, data: bytes) -> None:
        """Send UDP packet to simulation"""
        if self.send_socket:
            try:
                self.send_socket.sendto(data, (self.sim_host, self.send_port))
            except Exception as e:
                print(f"[MujocoController] Send error: {e}")

    # MotorCANController API compatibility

    def send_kinematics_for_motor(self, can_id: int, target: ControlData) -> None:
        """Send control command to motor"""
        self._send(pack_control_msg(can_id, target))

    def send_kinematics_batch(self, controls: dict[int, ControlData]) -> None:
        """Send batched control commands for multiple motors in one packet"""
        self._send(pack_control_batch(controls))

    def start_motor(self, can_id: int) -> None:
        """Start motor (enable control)"""
        self._send(pack_start_motor(can_id))

    def stop_motor(self, can_id: int) -> None:
        """Stop motor"""
        self._send(pack_stop_motor(can_id))

    def stop_all_motors(self) -> None:
        """Stop all motors"""
        self._send(pack_stop_all())

    def get_motor_configuration(self, can_id: int) -> None:
        """Request motor configuration"""
        self._send(pack_config_request(can_id))

    def set_feedback_callback(
        self, can_id: int, callback: Callable[[int, FeedbackData], None]
    ) -> None:
        """Register feedback callback for motor"""
        self.feedback_callbacks[can_id] = callback

    def set_config_callback(
        self, can_id: int, callback: Callable[[int, ConfigurationData], None]
    ) -> None:
        """Register config callback for motor"""
        self.config_callbacks[can_id] = callback

    def set_control_ack_callback(
        self, can_id: int, callback: Callable[[int], None]
    ) -> None:
        """Register control ack callback (not used in sim)"""
        self.control_ack_callbacks[can_id] = callback

    def is_running(self) -> bool:
        return self.running

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args) -> None:
        self.stop()
