"""
Hardware - Arduino Simulator
Simulates the Arduino hardware for testing without physical hardware. 
Behaves identically to ArduinoController but prints actions to console.
"""

import time
from typing import Optional, Dict

from hardware.bin_config import BinConfig, get_bin, BINS


class Arduinosimulator:
    """
    Simulates the Arduino waste sorting hardware.

    Drop-in replacement for ArduinoController - same interface,  
    but no serial connection required. Useful for:
      - Testing the full pipeline without hardware
      - CI/CD environments
      - Demo/presentation mode
    """

    def __init__(self, delay_factor: float = 0.1) -> None:
        """
        Initialize simulator.

        Args:
            delay_factor: Multiplier for simulated delays (0.1- 10x faster).
        """
        self.delay_factor = delay_factor
        self._connected = False
        self._gate_states: Dict[int, bool] = {i: False for i in range(10)}
        self._route_log: list = []

    def connect(self) -> None:
        """Simulate connecting to Arduino,"""
        print("[SIMULATOR] Connecting to virtual Arduino...")
        time.sleep(0.5  * self.delay_factor)
        self._connected = True
        print("[SIMULATOR] Virtual Arduino connected (simulation mode)")
        
    def disconnect(self) -> None:
        """Simulate disconnecting."""
        if self._connected:
            self.reset()
            self._connected = False
            print("[SIMULATOR] Virtual Arduino disconnected.")

    @property
    def is_connected(self) -> bool:
        return self._connected
    
    def __enter__(self):
        self.connect()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()
    
    # ------------------------------------------------------------------ #
    # Core Commands (same interface as ArduinoController)
    # ------------------------------------------------------------------ #
        
    def route_waste(self, category: str) -> bool:
        """Simulate routing waste to the correct bin."""
        self._check_connected()

        bin_cfg = get_bin(category)
        hold_sec = bin_cfg.hold_time_ms / 1000.0 * self.delay_factor

        print(f"[SIMULATOR] |-ROUTING: {category}'")
        print(f"[SIMULATOR] | Bin #{bin_cfg.bin_id} ({bin_cfg.label})")
        print(f"[SIMULATOR] | Motor pin: {bin_cfg.motor_pin}") 
        print(f"[SIMULATOR] | Opening gate → angle {bin_cfg.open_angle}*")
        
        self._gate_states[bin_cfg.bin_id] = True
        time.sleep(hold_sec)
        
        print(f"[SIMULATOR] |  Holding for {bin_cfg.hold_time_ms}ms...")
        print(f"[SIMULATOR] |  Closing gate → angle {bin_cfg.close_angle}*")

        self._gate_states[bin_cfg.bin_id] = False
        print(f"[SIMULATOR] |- DONE")
        
        self._route_log.append({
            "category": category, 
            "bin_id": bin_cfg.bin_id, 
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }) 

        return True

    def open_gate(self, category: str) -> bool:
        """Simulate opening a gate."""
        self._check_connected()
        bin_cfg = get_bin(category)
        self._gate_states[bin_cfg.bin_id] = True
        print(f"[SIMULATOR] Gate #{bin_cfg.bin_id} ({bin_cfg.label}) OPENED")
        return True
    
    def close_gate(self, category: str) -> bool:
        """Simulate closing a gate."""
        self._check_connected()
        bin_cfg = get_bin(category)
        self._gate_states[bin_cfg.bin_id] = False
        print(f"[SIMULATOR] Gate #{bin_cfg.bin_id} ({bin_cfg.label}) CLOSED")
        return True
    
    def reset(self) -> bool:
        """Simulate resetting all servos."""
        print("[SIMULATOR] Resetting all 10 bin gates...") 
        for i in range(10):
            self._gate_states[i] = False 
        print("[SIMULATOR] All gates closed.")
        return True
        
    def ping(self) -> bool:
        """Simulate a ping."""
        print("[SIMULATOR] PING → PONG")
        return True
    
    def get_status(self) -> Optional[str]: 
        """Return simulated status."""
        import json

        status = {
            "bins": [
                {
                    "id": bin_cfg.bin_id, 
                    "label": bin_cfg.label, 
                    "pin": bin_cfg.motor_pin, 
                    "open": self._gate_states.get(bin_cfg.bin_id, False),
                } 
                for bin_cfg in BINS.values()
            ]
        }
        return json.dumps(status)
    
    def test_all_bins(self) -> dict:
        """Test all bins in simulation."""
        print("[SIMULATOR] Testing all bins...")
        results = {}
        for category in BINS:
            success = self.route_waste(category)
            results[category] = success
        passed = sum(results.values())
        print(f"[SIMULATOR] Test complete: {passed}/{len(results)} bins passed.")
        return results
    
    def get_route_log(self) -> list:
        """Return the history of all routed items (simulator only).""" 
        return list(self._route_log)
    
    def _check_connected(self):
        if not self. connected:
            raise RuntimeError("Simulator not connected. Call connect() first.")