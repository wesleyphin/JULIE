#!/usr/bin/env python3
"""
Launcher script for JULIE Tkinter UI
Checks dependencies and launches the UI
"""

import sys
import subprocess

def check_dependencies():
    """Check if all required dependencies are installed"""
    missing = []

    # Check for tkinter
    try:
        import tkinter
    except ImportError:
        missing.append("tkinter")

    # Check for other dependencies
    try:
        import requests
    except ImportError:
        missing.append("requests")

    if missing:
        print("Missing required dependencies:")
        for dep in missing:
            print(f"  - {dep}")
        print("\nInstallation instructions:")
        if "tkinter" in missing:
            print("  For tkinter:")
            print("    Ubuntu/Debian: sudo apt-get install python3-tk")
            print("    Fedora: sudo dnf install python3-tkinter")
            print("    macOS: tkinter comes with Python")
            print("    Windows: tkinter comes with Python")
        if "requests" in missing:
            print("  For requests: pip install requests")
        return False

    return True

def main():
    """Main launcher"""
    print("=" * 60)
    print("JULIE - Tkinter Trading Dashboard")
    print("=" * 60)
    print()

    if not check_dependencies():
        sys.exit(1)

    print("✓ All dependencies satisfied")
    print("Launching UI...")
    print()

    # Launch the UI
    from julie_tkinter_ui import main as ui_main
    ui_main()

if __name__ == "__main__":
    main()
