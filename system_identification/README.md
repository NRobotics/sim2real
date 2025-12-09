# System Identification for Robot Actuators

A Python toolkit for performing system identification on robot actuators using frequency sweep (chirp) signals. This tool communicates with motor controllers over CAN bus to characterize actuator dynamics for sim-to-real transfer.

## Overview

System identification is crucial for accurate simulation of robot dynamics. This toolkit:

- **Generates chirp signals** - Frequency sweeps from low to high frequencies to excite all dynamic modes
- **Sends commands via CAN bus** - Communicates with motor controllers using the humanoid-protocol
- **Collects feedback data** - Records position, velocity, effort, temperature, and more
- **Analyzes results** - Computes transfer functions and visualizes motor response

## Installation

### Prerequisites

- Python 3.10+
- Linux with SocketCAN support (for CAN bus communication)
- CAN interface configured (e.g., `can0`)

### Setup

1. **Clone the repository with submodules:**

```bash
git clone --recursive https://github.com/NRobotics/sim2real.git
cd sim2real
```

If you already cloned without `--recursive`, initialize the submodules:

```bash
git submodule update --init --recursive
```

2. **Create a virtual environment (recommended):**

```bash
cd system_identification
python -m venv .venv
source .venv/bin/activate
```

3. **Install dependencies:**

```bash
# Install the humanoid-protocol package
pip install -e ../ext/humanoid-protocol/python

# Install system identification package with dev dependencies
pip install -e ".[dev]"
```

Or using requirements files:

```bash
pip install -e ../ext/humanoid-protocol/python
pip install -r requirements.txt
# For development:
pip install -r requirements-dev.txt
```

### CAN Interface Setup

Configure your CAN interface before running:

```bash
# For standard CAN (1 Mbps)
sudo ip link set can0 type can bitrate 1000000
sudo ip link set can0 up

# For CAN FD (with 5 Mbps data rate)
sudo ip link set can0 type can bitrate 1000000 dbitrate 5000000 fd on
sudo ip link set can0 up
```

## Usage

### Running System Identification

1. **Configure the test parameters** by editing `config.json`:

```json
{
    "can_interface": {
        "interface": "socketcan",
        "channel": "can0",
        "bitrate": 1000000,
        "fd": true
    },
    "chirp": {
        "f_start": 0.001,
        "f_end": 5.0,
        "duration": 60.0,
        "amplitude": 1.0,
        "sample_rate": 650.0,
        "sweep_type": "linear"
    },
    "control_parameters": {
        "velocity": 0.0,
        "effort": 0.0,
        "stiffness": 5.0,
        "damping": 2.0
    },
    "motors": {
        "0": {"amplitude_scale": 1.0, "offset": 0.0},
        "1": {"amplitude_scale": 1.0, "offset": 0.0}
    }
}
```

2. **Run the identification:**

```bash
python system_identification.py --config config.json --output results.json
```

### Analyzing Results

After collecting data, analyze and visualize the results:

```bash
# View all motors comparison
python analyse.py results.json --all

# View specific motor details
python analyse.py results.json --motor 0

# Save plots to file
python analyse.py results.json --all --save motor_response.png

# Print statistics only
python analyse.py results.json --stats
```

## Configuration Options

### CAN Interface

| Parameter | Description | Default |
|-----------|-------------|---------|
| `interface` | CAN interface type | `socketcan` |
| `channel` | CAN channel name | `can0` |
| `bitrate` | CAN bus bitrate | `1000000` |
| `fd` | Enable CAN FD | `true` |

### Chirp Signal

| Parameter | Description | Default |
|-----------|-------------|---------|
| `f_start` | Starting frequency (Hz) | `0.001` |
| `f_end` | Ending frequency (Hz) | `5.0` |
| `duration` | Sweep duration (seconds) | `60.0` |
| `amplitude` | Signal amplitude (0-1, maps to ±2π rad) | `1.0` |
| `sample_rate` | Command rate (Hz) | `650.0` |
| `sweep_type` | `linear`, `logarithmic`, or `exponential` | `linear` |

### Control Parameters

| Parameter | Description |
|-----------|-------------|
| `velocity` | Velocity feedforward |
| `effort` | Effort feedforward |
| `stiffness` | Position gain (Kp) |
| `damping` | Velocity gain (Kd) |

### Motor Configuration

Each motor can have:
- `amplitude_scale`: Scale factor for chirp amplitude (default: 1.0)
- `offset`: Constant offset added to position command (default: 0.0)

## Output Data

The results JSON file contains:

- **config**: The configuration used for the test
- **motor_configurations**: Hardware configuration read from each motor
- **feedback_data**: Time series data per motor:
  - `timestamp`: Time since start
  - `angle`: Measured position
  - `commanded_angle`: Commanded position
  - `angle_error`: Tracking error
  - `velocity`: Measured velocity
  - `effort`: Applied torque/current
  - `voltage`: Bus voltage
  - `temp_motor`: Motor temperature
  - `temp_pcb`: PCB temperature
- **statistics**: Summary statistics

## Development

### Code Quality Tools

```bash
# Format code
ruff format .

# Lint and auto-fix
ruff check --fix .

# Type checking
mypy system_identification.py analyse.py

# Run all checks
ruff format . && ruff check . && mypy .
```

### Project Structure

```
system_identification/
├── system_identification.py  # Main identification script
├── analyse.py                # Analysis and visualization
├── config.json               # Configuration file
├── pyproject.toml            # Project metadata and tool config
├── requirements.txt          # Production dependencies
├── requirements-dev.txt      # Development dependencies
└── README.md                 # This file
```

## Troubleshooting

### CAN Interface Issues

```bash
# Check CAN interface status
ip -details link show can0

# Monitor CAN traffic
candump can0

# Reset CAN interface
sudo ip link set can0 down
sudo ip link set can0 up
```

### Permission Issues

```bash
# Add user to dialout group for serial access
sudo usermod -a -G dialout $USER

# For CAN without sudo, add udev rules or capabilities
```

### Import Errors

Ensure humanoid-protocol is properly installed:

```bash
pip install -e ../ext/humanoid-protocol/python
python -c "from humanoid_messages.can import MotorCANController; print('OK')"
```

## License

MIT License - See LICENSE file for details.

