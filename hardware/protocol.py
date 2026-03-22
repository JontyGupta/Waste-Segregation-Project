"""
Hardware - Serial communication Protocol
Defines the message format between Python Arduino.

Protocol (over Serial / USB):
    Python sends   → Arduino receives
    Arduino sends  → Python receives

Message format:
    COMMAND: PAYLOAD\\n

    Commands (Python → Arduino): 
        ROUTE:<bin_id>                 - Route waste to bin #bin_id (0-9)
        OPEN:<bin_id>                  - Open gate of bin #bin_id
        CLOSE:<bin_id>                 - Close gate of bin #bin_id
        STATUS                         - Request system status
        RESET                          - Reset all servos to closed position
        PING                           - Health check
        LED:<bin_id>:<r>:<g>:<b>       - Set LED color for a bin

    Responses (Arduino → Python):
        ACK: ROUTE:<bin_id>            - Confirmed waste routed
        ACK: OPEN:<bin_id>             - Gate opened
        ACK: CLOSE:<bin_id>            - Gate closed
        ACK: RESET                     - All servos reset
        PONG                           - Reply to PING
        STATUS:<json>                  - System status JSON
        ERR:<message>                  - Error message
"""

from dataclasses import dataclass 
from enum import Enum 
from typing import Optional


class Command(Enum):
    """Commands that Python can send to Arduino, """
    ROUTE = "ROUTE"
    OPEN = "OPEN"
    CLOSE = "CLOSE"
    STATUS = "STATUS"
    RESET = "RESET"
    PING = "PING"
    LED = "LED"


class Response(Enum):
    """Responses that Arduino sends back."""
    ACK = "ACK"
    PONG = "PONG"
    STATUS = "STATUS"
    ERR = "ERR"


@dataclass
class SerialMessage:
    """A parsed serial message."""
    command: str
    payload: str = ""

    def encode(self) -> bytes:
        """Encode message for serial transmission."""
        if self.payload:
            msg = f"{self.command}:{self.payload}\n"
        else:
            msg = f"{self.command}\n"
        return msg.encode("utf-8")
    
    @classmethod
    def decode(cls, raw: bytes) -> "SerialMessage":
        """Decode a received serial message."""
        text = raw.decode("utf-8").strip()
        if ":" in text:
            parts = text.split(":", 1)
            return cls(command=parts[0], payload=parts[1])
            return cls(command=text)
    
    def is_ack(self) -> bool:
        return self.command == Response.ACK.value
    
    def is_error(self) -> bool:
        return self.command == Response.ERR.value