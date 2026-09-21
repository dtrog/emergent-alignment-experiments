#!/usr/bin/env python3
"""
Demo script to run a full experiment with the enhanced logging system.
Uses actual stress test prompts to demonstrate the complete workflow.
"""

import sys
from pathlib import Path
from datetime import datetime
import argparse

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

from domain_experiment_runner import DomainExperimentRunner


class DemoArgs:
    """Mock args object for demonstration."""
    def __init__(self):
        self.max_sessions = 3  # Small number for demo
        self.threads = 2
        self.verbose = True


def main():
    """Run a demonstration experiment with enhanced logging."""
    print("🚀 Starting Full Experiment Demonstration")
    print("=" * 60)
    
    # Create demo args
    args = DemoArgs()
    
    try:
        # Initialize and run the experiment
        runner = DomainExperimentRunner(args)
        
        print(f"📊 Experiment Setup:")
        print(f"   - Arms configured: {len(runner.subject_llms)}")
        print(f"   - Sessions per arm: {args.max_sessions}")
        print(f"   - Results directory: {runner.logger.get_run_directory()}")
        print()
        
        # Run the experiment
        runner.run()
        
        print("\n" + "=" * 60)
        print("✅ Experiment Complete!")
        
        # Show what was created
        results_dir = runner.logger.get_run_directory()
        print(f"\n📁 Results in: {results_dir}")
        
        # List all files created
        if results_dir.exists():
            files = list(results_dir.glob("*.csv")) + list(results_dir.glob("*.log"))
            print(f"\n📄 Files created ({len(files)} total):")
            for file in sorted(files):
                size = file.stat().st_size
                print(f"   - {file.name} ({size} bytes)")
                
                # Show first few lines of CSV files
                if file.suffix == '.csv' and size > 0:
                    try:
                        with open(file, 'r', encoding='utf-8') as f:
                            lines = f.readlines()[:3]  # Header + first 2 data rows
                            print(f"     Preview: {len(lines)} lines")
                            for i, line in enumerate(lines):
                                if i == 0:
                                    print(f"       Header: {line.strip()[:80]}...")
                                else:
                                    print(f"       Row {i}: {line.strip()[:80]}...")
                    except Exception as e:
                        print(f"     (Could not preview: {e})")
                print()
        
    except Exception as e:
        print(f"❌ Experiment failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
