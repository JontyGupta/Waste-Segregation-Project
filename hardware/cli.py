"""
Hardware - Standalone CLI
Run the hardware module independently from the classification pipeline.

Usage:
    # Route waste to a specific bin (real hardware)
    python -m hardware.cli --port COM3 --category biodegradable

    # Test all bins
    python -m hardware.cli --port COM3 --test-all

    # Simulation mode (no hardware needed)
    python -m hardware.cli --simulate --category e_waste

    # Simulate + test all bins
    python -m hardware.cli --simulate --test-all

    # Check hardware status
    python -m hardware.cli --port COM3 --status

    # Reset all gates
    python -m hardware.cli --port COM3 --reset
"""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(
        description="Waste Classifier - Arduino Hardware Controller (Standalone)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--port", type=str, default="COM3",
                        help="Serial port for Arduino (default: COM3).")
    parser.add_argument("--baud", type=int, default=9600,
                        help="Baud rate (default: 9600).")
    parser.add_argument("--simulate", action="store_true", 
                        help="Use simulator instead of real hardware.")
    
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--category", type=str,
                        help="Route waste for this category to its bin.")
    action.add_argument("--test-all", action="store_true",
                        help="Test all 10 bins sequentially.") 
    action.add_argument("--status", action="store_true",
                        help="Get hardware status.")
    action.add_argument("--reset", action="store_true",
                        help="Reset all gates to closed.")
    action.add_argument("--list-bins", action="store_true",
                        help="List all bin configurations.")
    
    args = parser.parse_args()
    
    # -------------- List bins (no connection needed) --------------
    if args.list_bins:
        from hardware.bin_config import BINS
        print("\n WASTE BIN CONFIGURATION") 
        print(" " + "=" * 65)
        print(f" {'Bin':>4} {'Pin':>4} {'Category': <35} {'Hold(ms)':>8}") 
        print(" " + "-" * 65)
        for cat, cfg in BINS.items():
            print(f" {cfg.bin_id:>4} {cfg.motor_pin:>4} {cat:<35} {cfg.hold_time_ms:>8}")
        print(" " + "=" * 65 + "\n")
        return
    
    # -------------- Create controller --------------
    if args.simulate:
        from hardware.simulator import ArduinoSimulator 
        controller = ArduinoSimulator(delay_factor=0.2)
    else:
        from hardware.controller import ArduinoController
        controller = ArduinoController(port=args.port, baud_rate=args.baud)
    
    # -------------- Execute action --------------
    try:
        controller.connect()

        if args.category:
            success = controller.route_waste(args.category) 
            sys.exit(0 if success else 1)

        elif args.test_all:
            results = controller.test_all_bins() 
            failed = [cat for cat, ok in results.items() if not ok] 
            sys.exit(e if not failed else 1)

        elif args.status:
            status = controller.get_status()
            if status:
                print(f"\nHardware Status: \n{status}\n")
            else:
                print("No status received.")
                sys.exit(1)
        
        elif args.reset:
            controller.reset()
            print("All gates reset.")
        
    except Exception as e:
        print(f"[ERROR] (e)")
        sys.exit(1)
    finally:
        controller.disconnect()
        
        
if __name__ == "__main__":
    main()