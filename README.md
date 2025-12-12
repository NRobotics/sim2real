# Sim2Real

A systematic approach for sim-to-real transfer of legged robots, identifying actuator and joint dynamics with standard joint encoders.

## Overview

This project provides tools for:

- **System Identification** - Characterize actuator dynamics using chirp signals
- **Transfer Function Analysis** - Extract frequency response from collected data
- **Sim-to-Real Calibration** - Use identified parameters to improve simulation fidelity

## Repository Structure

```
sim2real/
├── ext/
│├── humanoid-protocol/     # Git submodule: CAN/UDP communication protocol
├── system_identification/     # Actuator system identification toolkit
│   ├── system_identification.py
│   ├── analyse.py
│   └── config.json
├── speed_test.py              # CAN bus performance testing
└── README.md
```

## Quick Start

### Prerequisites

- Python 3.10+
- Linux with SocketCAN support
- CAN interface (USB-CAN adapter)

### Installation

1. **Clone with submodules:**

```bash
git clone --recursive https://github.com/NRobotics/sim2real.git
cd sim2real
```

2. **Set up Python environment:**

```bash
python -m venv .venv
source .venv/bin/activate
```

3. **Install humanoid-protocol:**

```bash
pip install -e ext/humanoid-protocol/python
```

4. **Install system identification tools:**

```bash
cd system_identification
pip install -e ".[dev]"
```

### Configure CAN Interface

```bash
# Standard CAN at 1 Mbps
sudo ip link set can0 type can bitrate 1000000
sudo ip link set can0 up

# For CAN FD
sudo ip link set can0 type can bitrate 1000000 dbitrate 5000000 fd on
sudo ip link set can0 up
```

### Run System Identification

```bash
cd system_identification

# Run identification (ensure motors are connected and safe to move!)
python system_identification.py --config config.json --output results.json

# Analyze results
python analyse.py results.json --all
```

## Components

### System Identification (`system_identification/`)

Tools for characterizing motor actuator dynamics:

- Generates chirp (frequency sweep) signals
- Collects position, velocity, and effort feedback
- Computes tracking error and statistics
- Visualizes motor response

See [`system_identification/README.md`](system_identification/README.md) for detailed documentation.

### Humanoid Protocol (`ext/humanoid-protocol/`)

Communication protocol for motor controllers:

- CAN bus message encoding/decoding
- UDP protocol support
- Python and C implementations

### Speed Test (`speed_test.py`)

Benchmark tool for CAN bus communication:

- Measures round-trip latency
- Tests sustained throughput
- Validates timing for real-time control

## Development

### Code Quality

This project uses:

- **[Ruff](https://docs.astral.sh/ruff/)** - Fast Python linter and formatter
- **[MyPy](https://mypy.readthedocs.io/)** - Static type checking
- **[Pytest](https://pytest.org/)** - Testing framework

Run checks:

```bash
cd system_identification
ruff format .
ruff check .
mypy .
```

### Git Submodules

When cloning, use `--recursive` or initialize submodules:

```bash
git submodule update --init --recursive
```

To update submodules to latest:

```bash
git submodule update --remote
```

## Safety Warning

⚠️ **This software controls physical robot actuators!**

- Always ensure the robot is in a safe configuration before running
- Use appropriate stiffness/damping values
- Start with small amplitudes
- Have an emergency stop ready
- Never run unattended

## License

MIT License

## Contributing

1. Fork the repository
2. Create a feature branch
3. Run linters and tests
4. Submit a pull request

## References

- [System Identification Theory](https://en.wikipedia.org/wiki/System_identification)
- [Chirp Signal](https://en.wikipedia.org/wiki/Chirp)
- [SocketCAN](https://www.kernel.org/doc/html/latest/networking/can.html)
