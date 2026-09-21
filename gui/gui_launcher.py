#!/usr/bin/env python3
"""
Launch script for the GUI-based Emergent Alignment Experiments.

Usage:
    python gui_launcher.py                 # Start with GUI
    python gui_launcher.py --no-gui        # Start without GUI (terminal only)
    python gui_launcher.py --sessions 100  # Limit sessions per arm
"""

import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

if __name__ == "__main__":
    from gui.gui_experiment_runner import main
    main()
