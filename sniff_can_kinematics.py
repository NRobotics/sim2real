import can
import time
import struct

def sniff():
    try:
        bus = can.Bus(interface='usbtingo', channel=None, bitrate=1000000, data_bitrate=5000000, fd=True)
        print("Sniffer started. Listening for 5 seconds...")
        
        # Trigger a start to motor 0
        print("Sending start to motor 0 (ID 112)...")
        msg = can.Message(arbitration_id=112, data=b'', is_fd=True, bitrate_switch=True)
        bus.send(msg)
        time.sleep(0.1)

        # Trigger a kinematics request to motor 0 (ID 192)
        # ControlData: angle, velocity, effort, stiffness, damping (all floats)
        # Send 0,0,0,0,0
        data = struct.pack("<fffff", 0.0, 0.0, 0.0, 0.0, 0.0)
        print("Sending kinematics to motor 0 (ID 192) with zero gains...")
        msg = can.Message(arbitration_id=192, data=data, is_fd=True, bitrate_switch=True)
        bus.send(msg)

        start_time = time.time()
        while time.time() - start_time < 5:
            msg = bus.recv(timeout=0.1)
            if msg:
                print(f"ID: {msg.arbitration_id} (0x{msg.arbitration_id:X}), DLC: {msg.dlc}, Data: {msg.data.hex()}")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if 'bus' in locals():
            bus.shutdown()

if __name__ == "__main__":
    sniff()

