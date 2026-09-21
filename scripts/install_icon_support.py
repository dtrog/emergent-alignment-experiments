#!/usr/bin/env python3
"""
Optional enhancement script to install better icon support for the GUI monitor.

This script installs PIL (Pillow) and cairosvg for better SVG icon handling.
The GUI will work with emoji fallbacks if these aren't installed.
"""

import subprocess
import sys

def install_icon_dependencies():
    """Install dependencies for enhanced icon support."""
    packages = [
        "Pillow",  # PIL for image handling
        "cairosvg",  # SVG to PNG conversion
    ]
    
    print("Installing enhanced icon support packages...")
    print("This will enable proper SVG icon display in the GUI monitor.")
    print()
    
    for package in packages:
        print(f"Installing {package}...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"✅ {package} installed successfully")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install {package}: {e}")
            return False
        except Exception as e:
            print(f"❌ Unexpected error installing {package}: {e}")
            return False
    
    print()
    print("✅ Icon support packages installed successfully!")
    print("Restart the GUI monitor to see proper SVG icons.")
    return True

if __name__ == "__main__":
    install_icon_dependencies()
