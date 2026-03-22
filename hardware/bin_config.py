"""
Hardware - Bin & Motor Configuration
Defines the physical mapping between waste categories, bins, 
servo/stepper motor pins, and positions.

This file is the SINGLE SOURCE OF TRUTH for hardware layout. 
Update it when you add/move physical bins.
"""

from dataclasses import dataclass, field 
from typing import Dict, List, Optional


@dataclass
class BinConfig:
    """
    Configuration for a single waste bin + its motor.

    Attributes:
        category        : Waste category this bin handles.
        bin_id          : Physical bin identifier (e-indexed).
        motor_pin       : Arduino pin connected to the servo/stepper.
        open_angle      : Servo angle to open the gate (degrees).
        close_angle     : Servo angle to close the gate (degrees).
        hold_time_ms    : How long to keep the gate open (milliseconds).
        label           : Human-readable label for the bin.
        color           : Optional color code for LED indicator.
    """

    category: str
    bin_id: int
    motor_pin: int
    open_angle: int = 90
    close_angle: int = 0
    hold_time_ms: int = 3000
    label: str = ""
    color: str = ""

    def __post_init__(self):
        if not self.label:
            self.label = self.category.replace("_", "").title()


# ================================================================== #
# BIN LAYOUT - Update this when you change physical hardware
# ================================================================== #
#
# Physical layout (top view):
#
#       [BIN 0]   [BIN 1]   [BIN 2]   [BIN 3]   [BIN 4]
#         Bio     Recycle   Non-Rec   Medical   E-Waste
#
#       [BIN 5]   [BIN 6]   [BIN 7]   [BIN 8]   [BIN 9] 
#        Hazard   Textile   Construc  Sanitary   Other
#
# Conveyor belt feeds into a diverter controlled by a central 
# servo that rotates to direct waste to the correct bin.
#
# ================================================================== #

BINS: Dict[str, BinConfig] = {
    "biodegradable": BinConfig(
        category="biodegradable",
        bin_id=0,
        motor_pin=2,
        open_angle=90,
        close_angle=0,
        hold_time_ms=3000,
        color="green",
    ),
    "non_biodegradable_recyclable": BinConfig(
        category="non_biodegradable_recyclable",
        bin_id=1,
        motor_pin=3,
        open_angle=90,
        close_angle=0,
        hold_time_ms=3000,
        color="orange",
    ),
    "non_biodegradable_non_recyclable": BinConfig(
        category="non_biodegradable_non_recyclable",
        bin_id=2,
        motor_pin=4,
        open_angle=90,
        close_angle=0,
        hold_time_ms=3000,
        color="red",
    ),
    "medical_waste": BinConfig(
        category="medical_waste",
        bin_id=3,
        motor_pin=5,
        open_angle=90,
        close_angle=0,
        hold_time_ms=4000,
        color="purple",
    ),
    "e_waste": BinConfig(
        category="e_waste",
        bin_id=4,
        motor_pin=6,
        open_angle=90,
        close_angle=0,
        hold_time_ms=3000,
        color="yellow",
    ),
    "hazardous_waste": BinConfig(
        category="hazardous_waste",
        bin_id=5,
        motor_pin=7,
        open_angle=90,
        close_angle=0,
        hold_time_ms=4000,
        color="darkred",
    ),
    "textile_waste": BinConfig(
        category="textile_waste",
        bin_id=6,
        motor_pin=8,
        open_angle=90,
        close_angle=0,
        hold_time_ms=3000,
        color="pink",
    ),
    "construction_waste": BinConfig(
        category="construction_waste",
        id=7,
        motor_pin=9,
        open_angle=90,
        close_angle=0,
        hold_time_ms=3500,
        color="brown",
    ),
    "sanitary_waste": BinConfig(
        category="sanitary_waste",
        bin_id=8,
        motor_pin=10,
        open_angle=90,
        close_angle=0,
        hold_time_ms=3000,
        color="darkorange",
    ),
    "other": BinConfig(
        category="other",
        bin_id=9,
        motor_pin=11,
        open_angle=90,
        close_angle=0,
        hold_time_ms=3000,
        color="gray",
    ),
}


def get_bin(category: str) -> BinConfig:
    """
    Get bin config for a waste category. 
    Falls back to 'other' if category is unknown.
    """
    return BINS.get(category, BINS["other"])


def get_all_categories() -> List[str]:
    """Return all configured categories."""
    return list(BINS.keys())


def get_pin_map() -> Dict[int, str]:
    """Return mapping of Arduino pin -> category (useful for firmware)."""
    return {b.motor_pin: b.category for b in BINS.values()}
