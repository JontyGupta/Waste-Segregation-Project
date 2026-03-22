"""
Hardware - Arduino Serial Controller
Communicates with the Arduino over USB serial to route waste to bins.

INDEPENDENT from the classification pipeline - can be used standalone.

Usage:
    controller = ArduinoController (port="COM3", baud_rate=9600)
    controller.connect()
    controller.route_waste("biodegradable")
    controller.disconnect()
"""

import time
from typing import Optional

from hardware.bin_config import BinConfig, get_bin, BINS 
from hardware.protocol import Command, SerialMessage


class ArduinoController:
    """
    Controls the Arduino-based waste sorting hardware.

    Sends serial commands to the Arduino which drives servo motors 
    to open/close bin gates and route waste to the correct bin.
    """

    def __init__(
        self,
        port: str = "COM3",
        baud_rate: int = 9600,
        timeout: float = 2.0,
        retry_count: int = 3,
    ) -> None:
        """
        Initialize ArduinoController.

        Args:
            port: Serial port (COM3, /dev/ttyUSB0, /dev/ttyACM0, etc.).
            baud_rate: Baud rate for serial communication.
            timeout: Read timeout in seconds.
            retry_count: Number of retries on failed commands.
        """
        self.port = port
        self.baud_rate = baud_rate
        self.timeout = timeout
        self.retry_count = retry_count
        self._serial = None
        self._connected = False

    # ---------------------------------------------------------------- #
    # Connection Management
    # ---------------------------------------------------------------- #

    def connect(self) -> None:
        """Open serial connection to Arduino."""
        try:
            import serial
        except ImportError as e:
            raise ImportError(
            "pyserial is required for Arduino communication."
            "Install via: pip install pyserial"
            ) from e

        print(f"[HARDWARE] Connecting to Arduino on {self.port}...")
        self._serial = serial.Serial(
            port=self.port,
            baudrate=self.baud_rate,
            timeout=self.timeout,
        )

        # Wait for Arduino to reset after serial connection
        time.sleep(2.0)
        
        # Verify connection with PING
        if self.ping():
            self._connected = True
            print(f"[HARDWARE] Connected to Arduino on {self.port}")
        else:
            self._serial.close() 
            self._serial = None 
            raise ConnectionError( 
                f"Arduino not responding on {self.port}."
                "Check connection and firmware."
            ) 
        
    def disconnect(self) -> None:
        """Close serial connection."""
        if self._serial and self._serial.is_open: 
            self.reset() 
            self._serial.close() 
            self._connected = False 
            print("[HARDWARE] Arduino disconnected.")

    @property
    def is_connected(self) -> bool:
        return self._connected and self._serial is not None and self._serial.is_open
    
    def __enter__(self):
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()
        
    # ---------------------------------------------------------------- #
    # Core Commands
    # ---------------------------------------------------------------- #
    
    def route_waste(self, category: str) -> bool:
        """
        Route waste to the correct bin based on category.

        This is the PRIMARY method called by the classification pipeline.

        Steps:
            1. Look up bin config for the category
            2. Send ROUTE command to Arduino
            3. Arduino opens gate waits → closes gate
            4. Wait for ACK

        Args:
            category: Waste category (e.g., 'biodegradable', 'e_waste').

        Returns:
            True if successfully routed, False otherwise.
        """
        bin_cfg = get_bin(category)

        print(f"[HARDWARE] Routing '{category}' → Bin #{bin_cfg.bin_id}" 
              f"({bin_cfg.label}, pin {bin_cfg.motor_pin})")
        
        return self._send_and_ack(Command.ROUTE, str(bin_cfg.bin_id))
    
    def open_gate(self, category: str) -> bool:
        """Manually open a bin gate."""
        bin_cfg = get_bin(category)
        return self._send_and_ack(Command.OPEN, str(bin_cfg.bin_id))
    
    def close_gate(self, category: str) -> bool:
        """Manually close a bin gate."""
        bin_cfg = get_bin(category)
        return self._send_and_ack(Command.CLOSE, str(bin_cfg.bin_id))
        
    def reset(self) -> bool:
        """Reset all servos to closed position.""" 
        print("[HARDWARE] Resetting all servos...")
        return self._send_and_ack(Command.RESET, "")
        
    def ping(self) -> bool:
        """Send a health check ping. Returns True if Arduino responds."""
        msg = SerialMessage(command=Command.PING.value)
        self._write(msg)

        response = self._read()
        if response and response.command == "PONG":
            return True
        return False
    
    def get_status(self) -> Optional[str]:
        """
        Request system status from Arduino."""
        msg = SerialMessage(command=Command.STATUS.value)
        self._write(msg)

        response = self._read()
        if response and response.command == "STATUS":
            return response.payload
        return None
    
    def set_led(self, category: str, r: int, g: int, b: int) -> bool:
        """Set LED indicator color for a bin."""
        bin_cfg = get_bin(category)
        payload = f"{bin_cfg.bin_id}:{r}:{g}:{b}"
        return self._send_and_ack(Command.LED, payload)
    
    def test_all_bins(self) -> dict:
        """
        Test all bins sequentially - opens and closes each gate. 
        Useful for hardware verification.

        Returns:
            Dict of category → success (bool).
        """
        print("[HARDWARE] Testing all bins...")
        results = {}

        for category, bin_cfg in BINS.items():
            print(f" Testing bin #{bin_cfg.bin_id} ({bin_cfg.label})...", end=" ")
            success = self.route_waste(category)
            results[category] = success
            status = "OK" if success else "FAIL"
            print(status)
            time.sleep(1.0) # Pause between tests

        passed = sum(results.values())
        total = len(results)
        print(f"[HARDWARE] Test complete: {passed}/{total} bins passed.")
        return results
    
    # ---------------------------------------------------------------- #
    # Low-Level Serial I/0
    # ---------------------------------------------------------------- #
    
    def _send_and_ack(self, command: Command, payload: str) -> bool: 
        """Send a command and wait for ACK. Retries on failure.""" 
        msg = SerialMessage(command=command.value, payload=payload)
        for attempt in range(1, self.retry_count + 1):
            self._write(msg)
            response = self._read()

            if response and response.is_ack():
                return True
            
            if response and response.is_error():
                print(f"[HARDWARE] Error from Arduino: {response.payload}")
                return False
        
            print(f"[HARDWARE] No ACK (attempt {attempt}/{self.retry_count})")

        print(f"[HARDWARE] Command {command.value} failed after {self.retry_count} retries.")
        return False
    
    def _write(self, msg: SerialMessage) -> None:
        """Write a message to the serial port."""
        if not self.is_connected:
            raise RuntimeError("Arduino not connected. Call connect() first.")
        self._serial.write(msg.encode())
        self._serial.flush()
        
    def _read(self) -> Optional [SerialMessage]:
        """Read a response from the serial port."""
        if not self.is_connected:
            return None
        
        try:
            raw = self._serial.readline()
            if raw:
                return SerialMessage.decode(raw)
        except Exception as e:
            print(f"[HARDWARE] Serial read error: {e}")

        return None