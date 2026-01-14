import can
import time

def sniff():
    try:
        bus = can.Bus(interface='usbtingo', channel=None, bitrate=1000000, data_bitrate=5000000, fd=True)
        print("Sniffer started. Listening for 5 seconds...")
        
        # Trigger a ping to see what happens
        print("Sending ping to motor 0 (ID 96)...")
        msg = can.Message(arbitration_id=96, data=b'', is_fd=True, bitrate_switch=True)
        bus.send(msg)
        
        # Also trigger a config get to see the working case
        print("Sending config get to motor 0 (ID 288)...")
        msg = can.Message(arbitration_id=288, data=b'', is_fd=True, bitrate_switch=True)
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

