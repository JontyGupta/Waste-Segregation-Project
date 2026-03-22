"""
Hardware Interface Module - Arduino Motor Controller
=======================================================
FULLY INDEPENDENT from the classification software.

Can be used standalone:
    from hardware import ArduinoController
    controller = ArduinoController (port="COM3")
    controller.connect()
    controller.route_waste("biodegradable")
    controller.disconnect()

Or via CLI:
    python -m hardware.cli --port COM3 --category biodegradable
    python -m hardware.cli --port COM3 --test-all
    python - hardware.cli --simulate --category e_waste
"""

from hardware.controller import ArduinoController
from hardware.bin_config import BinConfig, BINS
from hardware.simulator import ArduinoSimulator
__all__ = ["ArduinoController", "BinConfig", "BINS", "ArduinoSimulator"]