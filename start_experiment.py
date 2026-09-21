#!/usr/bin/env python3
"""
Main experiment orchestrator for emergent alignment experiments.

This module provides a clean, modular interface for running experiments
using the refactored component architecture.
"""

import argparse
import traceback
from colorama import Fore, init as colorama_init
<<<<<<< Updated upstream

=======
>>>>>>> Stashed changes
from services.experiment_runner import ExperimentRunner
from utils import load_json_file


def create_argument_parser():
    """Create and configure the command line argument parser."""
    parser = argparse.ArgumentParser(description="Run emergent alignment experiments")
    
    parser.add_argument(
        "--arms-config", 
        type=str, 
        default="config/arms.json",
        help="Path to the arms configuration file"
    )
    
    parser.add_argument(
        "--sessions", 
        type=int, 
        default=1,
        help="Number of sessions to run per arm"
    )
    
    parser.add_argument(
        "--verbose", 
        action="store_true",
        help="Enable verbose logging"
    )
    
    parser.add_argument(
        "--dry-run", 
        action="store_true",
        help="Perform a dry run without making API calls"
    )
    
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default="results",
        help="Directory to save results"
    )
    
    parser.add_argument(
        "--threads", 
        type=int, 
        default=1,
        help="Number of threads for parallel execution"
    )
    
    parser.add_argument(
        "--max-sessions", 
        type=int, 
        default=10,
        help="Maximum number of sessions per arm (fallback if not specified in arm config)"
    )
    
    parser.add_argument(
        "--trim-length", 
        type=int, 
        default=0,
        help="Length to trim conversation history (0 = no trimming)"
    )
    
    return parser


def validate_configuration(args):
    """Validate the experiment configuration."""
    try:
        # Validate arms configuration exists and is valid
        arms_config = load_json_file(args.arms_config)
        if not arms_config:
            raise ValueError(f"Empty or invalid arms configuration: {args.arms_config}")
        
        if args.sessions < 1:
            raise ValueError("Number of sessions must be at least 1")
            
        return True
        
    except Exception as e:
        print(Fore.RED + f"Configuration validation failed: {e}")
        return False


def main():
    """Main entry point for the experiment."""
    # Initialize colorama for cross-platform colored output
    colorama_init(autoreset=True)
    
    print(Fore.CYAN + "🧪 Emergent Alignment Experiments")
    print(Fore.CYAN + "==================================")
    
    # Parse command line arguments
    parser = create_argument_parser()
    args = parser.parse_args()
    
    # Validate configuration
    if not validate_configuration(args):
        return 1
    
    try:
        # Create and run the experiment
        print(Fore.GREEN + f"Starting experiment with {args.sessions} sessions...")
        print(Fore.GREEN + f"Arms config: {args.arms_config}")
        print(Fore.GREEN + f"Output directory: {args.output_dir}")
        
        if args.dry_run:
            print(Fore.YELLOW + "DRY RUN MODE - No API calls will be made")
        
        runner = ExperimentRunner(args=args)
        runner.run()
        
        print(Fore.GREEN + "✅ Experiment completed successfully!")
        return 0
        
    except KeyboardInterrupt:
        print(Fore.YELLOW + "\n⚠️  Experiment interrupted by user")
        return 130
        
    except Exception as e:
        print(Fore.RED + f"❌ A critical error occurred during experiment: {e}")
        if args.verbose:
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
