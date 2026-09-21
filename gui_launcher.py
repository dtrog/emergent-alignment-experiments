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
    try:
        from gui.gui_experiment_runner import main
        main()
    except ImportError as e:
        print("[Warning] GUI module not found or refactored. Falling back to CLI experiment runner.")
        try:
            from emergent_experiment import ExperimentRunner
            import argparse
            parser = argparse.ArgumentParser(description="Run emergent alignment experiments (CLI fallback)")
            parser.add_argument('--max-sessions', type=int, default=500)
            parser.add_argument('--threads', type=int, default=8)
            parser.add_argument('--trim-length', type=int, default=1024)
            args = parser.parse_args()
            runner = ExperimentRunner(args=args)
            runner.run()
        except Exception as cli_e:
            print(f"[Error] Could not run CLI experiment: {cli_e}")
