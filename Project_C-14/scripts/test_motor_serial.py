#!/usr/bin/env python3
"""
Simple serial test for Arduino motor command parsing.

Sends lines in the same format your firmware expects:
    linear_vel,angular_vel\n

Example:
    python scripts/test_motor_serial.py --port /dev/tty.usbmodem2101
"""

import argparse
import sys
import time

try:
    import serial  # type: ignore
    from serial.tools import list_ports  # type: ignore
except ImportError:
    print("pyserial is required. Install with: pip install pyserial")
    sys.exit(1)


def list_available_ports() -> None:
    ports = list(list_ports.comports())
    if not ports:
        print("No serial ports found.")
        return
    print("Available serial ports:")
    for p in ports:
        print(f"  - {p.device} ({p.description})")


def send_line(ser: serial.Serial, linear: float, angular: float) -> None:
    payload = f"{linear:.4f},{angular:.4f}\n"
    ser.write(payload.encode("ascii"))
    ser.flush()
    print(f"[sent] {payload.strip()}")


def run_sequence(ser: serial.Serial, hold_s: float) -> None:
    # Includes parser checks and simple motion checks.
    print("\nStarting parse + motion sequence...")
    tests = [
        ("parser negative test (invalid line, should be ignored)", None),
        ("stop", (0.0, 0.0)),
        ("forward", (0.12, 0.0)),
        ("stop", (0.0, 0.0)),
        ("backward", (-0.12, 0.0)),
        ("stop", (0.0, 0.0)),
        ("turn left in place", (0.0, 0.8)),
        ("stop", (0.0, 0.0)),
        ("turn right in place", (0.0, -0.8)),
        ("stop", (0.0, 0.0)),
        ("forward + slight left arc", (0.10, 0.35)),
        ("stop", (0.0, 0.0)),
    ]

    for label, cmd in tests:
        print(f"\n[test] {label}")
        if cmd is None:
            ser.write(b"bad_line_without_comma\n")
            ser.flush()
            print("[sent] bad_line_without_comma")
        else:
            send_line(ser, cmd[0], cmd[1])
        time.sleep(hold_s)

    print("\nSequence finished.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Arduino motor serial test tool")
    parser.add_argument("--port", help="Serial device, e.g. /dev/tty.usbmodem2101")
    parser.add_argument("--baud", type=int, default=115200, help="Serial baud rate")
    parser.add_argument(
        "--hold",
        type=float,
        default=1.2,
        help="Seconds to hold each test command",
    )
    parser.add_argument(
        "--list-ports",
        action="store_true",
        help="List serial ports and exit",
    )
    args = parser.parse_args()

    if args.list_ports:
        list_available_ports()
        return

    if not args.port:
        print("Missing --port. Use --list-ports to find your device name.")
        sys.exit(2)

    print(f"Opening {args.port} @ {args.baud} ...")
    with serial.Serial(port=args.port, baudrate=args.baud, timeout=0.2) as ser:
        # Nano Every may reset on open; give bootloader time.
        time.sleep(2.0)
        run_sequence(ser, hold_s=args.hold)

        # End with explicit stop.
        send_line(ser, 0.0, 0.0)
        print("Done. If motors moved as expected, parse/control path is working.")


if __name__ == "__main__":
    main()
