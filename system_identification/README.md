# System Identification for Robot Actuators

A Python toolkit for performing system identification on robot actuators using frequency sweep (chirp) signals. Supports CAN bus hardware, MuJoCo simulation, and dry-run testing.

## Quick Start

```bash
# 1. Test hardware connection first
python scripts/hardware_test.py --mujoco

# 2. Run system identification
python scripts/system_identification.py --mujoco --save --save-plots
```

## Features

- **Chirp signal generation** - Frequency sweeps to excite all dynamic modes
- **Multiple backends** - Real CAN hardware, MuJoCo simulation, or dry-run mock
- **IK group support** - Control multiple motors via inverse kinematics (e.g., foot pitch/roll)
- **Safety features** - Motor limits, ping-based position queries, smooth interpolation
- **Async communication** - Decoupled command/feedback for high-rate control (500+ Hz)

## Scripts

### Hardware Test (`scripts/hardware_test.py`)

Validate motor communication before running full identification:

```bash
# Test with MuJoCo simulation
python scripts/hardware_test.py --mujoco

# Test with mock controller (no hardware)
python scripts/hardware_test.py --dry-run

# Test real hardware
python scripts/hardware_test.py

# Test specific motors with verbose output
python scripts/hardware_test.py --mujoco --motor-ids 0 1 2 -v

# Skip certain tests
python scripts/hardware_test.py --mujoco --skip timing
```

**Tests performed:**
1. **Connection** - Verifies motors respond to config requests
2. **Ping** - Validates `ping_motor()` returns position feedback
3. **Command** - Sends hold-position command, verifies feedback
4. **Timing** - Measures round-trip latency (100 samples)
5. **Stress** - Prolonged async loop test (optional, use `--stress`)

**Stress test options:**
```bash
# Run with stress test (5s at 100Hz)
python scripts/hardware_test.py --mujoco --stress

# Custom duration and rate
python scripts/hardware_test.py --mujoco --stress --stress-duration 10 --stress-rate 200
```

The stress test measures:
- Missed deadlines (target: <5%)
- Feedback rate (target: >90%)
- Latency statistics (target: <10ms avg)

### System Identification (`scripts/system_identification.py`)

Run the main identification:

```bash
# MuJoCo simulation
python scripts/system_identification.py --mujoco --save --save-plots

# Dry-run (mock controller)
python scripts/system_identification.py --dry-run --save

# Real hardware
python scripts/system_identification.py --save --save-plots

# Custom config and motor IDs
python scripts/system_identification.py -c my_config.json -m 0 1 2 --save
```

**Options:**
| Flag | Description |
|------|-------------|
| `--config, -c` | Config file path (default: `config/config.json`) |
| `--motor-ids, -m` | Motor IDs to test (default: from config) |
| `--mujoco` | Use MuJoCo simulation |
| `--dry-run` | Use mock controller |
| `--save` | Save results to JSON and PyTorch files |
| `--save-plots` | Generate and save plots |

## Configuration

Edit `config/config.json`:

```json
{
  "can_interface": {
    "interface": "socketcan",
    "channel": "can0",
    "bitrate": 1000000,
    "fd": true
  },
  "mujoco": {
    "host": "127.0.0.1",
    "send_port": 5000,
    "recv_port": 5001
  },
  "chirp": {
    "f_start": 0.0,
    "f_end": 1.0,
    "duration": 20.0,
    "sample_rate": 500.0,
    "sweep_type": "linear"
  },
  "control_parameters": {
    "velocity": 0.0,
    "effort": 0.0,
    "stiffness": 10.0,
    "damping": 0.1
  },
  "interpolation_duration": 2.0,
  "motor_ids": [0, 1, 2, 3, 4, 5],
  "motors": {
    "0": {
      "name": "left_hip_pitch",
      "scale": 0.5,
      "direction": 1.0,
      "bias": 0.0,
      "limits": [-1.5, 1.5],
      "control_parameters": {
        "stiffness": 20.0,
        "damping": 0.3
      }
    }
  },
  "ik_groups": [
    {
      "name": "foot_0",
      "ik_type": "foot",
      "motor_ids": [4, 5],
      "chirp": {
        "scale_pitch": 0.2,
        "scale_roll": 0.2
      },
      "limits": {
        "4": [-3.0, 3.0],
        "5": [-3.0, 3.0]
      }
    }
  ]
}
```

### Key Parameters

| Parameter | Description |
|-----------|-------------|
| `chirp.f_start/f_end` | Frequency sweep range (Hz) |
| `chirp.duration` | Sweep duration (seconds) |
| `chirp.sample_rate` | Command rate (Hz) |
| `motors.*.scale` | Chirp amplitude (radians) |
| `motors.*.bias` | Position offset (radians) |
| `motors.*.limits` | Safety limits [min, max] (radians) |
| `interpolation_duration` | Time to move to start/end positions |

## Safety Features

1. **Motor limits** - Commands are clamped to configured min/max positions
2. **Ping-based queries** - Get current position without sending commands
3. **Smooth interpolation** - Gradual movement to chirp start position
4. **Connection check** - Script waits for motors before proceeding
5. **Hold-position tests** - Hardware tests don't command arbitrary positions

## MuJoCo Simulation

Run with MuJoCo for testing without hardware:

```bash
# Terminal 1: Start simulation
cd sim2real
python -m hoku.hoku_mujoco

# Terminal 2: Run identification (waits for connection)
python scripts/system_identification.py --mujoco --save --save-plots
```

## Output

Results are saved to `data/sysid_YYYYMMDD_HHMMSS/`:

```
data/sysid_20260108_171742/
├── sysid_20260108_171742.json    # Raw data (JSON)
├── sysid_20260108_171742.pt      # PyTorch tensor format
├── comm_stats_20260108_171742.json
└── plots/
    ├── motor_0_20260108_171742.png
    ├── motor_1_20260108_171742.png
    └── foot_0_pitch_20260108_171742.png
```

## Project Structure

```
system_identification/
├── config/
│   └── config.json           # Configuration file
├── data/                     # Output data directory
├── scripts/
│   ├── system_identification.py  # Main entry point
│   ├── sysid.py                  # SystemIdentification class
│   ├── hardware_test.py          # Hardware validation
│   ├── test_modules.py           # Test implementations
│   ├── async_loop.py             # Async control loop
│   ├── chirp.py                  # Chirp generators
│   ├── controllers.py            # Mock controllers
│   ├── ik_registry.py            # Inverse kinematics
│   ├── results.py                # Result saving
│   └── analyse.py                # Analysis tools
└── README.md
```

## Troubleshooting

### No response from motors

```bash
# Check if MuJoCo is running
python -m hoku.hoku_mujoco

# For hardware, check CAN interface
ip -details link show can0
candump can0
```

### CAN Interface Setup

```bash
# Standard CAN (1 Mbps)
sudo ip link set can0 type can bitrate 1000000
sudo ip link set can0 up

# CAN FD (with 5 Mbps data rate)
sudo ip link set can0 type can bitrate 1000000 dbitrate 5000000 fd on
sudo ip link set can0 up
```

### Import Errors

```bash
# Ensure humanoid-protocol is installed
pip install -e humanoid-protocol/python

# Test import
python -c "from humanoid_messages.can import MotorCANController; print('OK')"
```
