"""
Logger for experiment results and actor session logs.

This module provides the Logger class which handles:
- CSV result logging with thread safety
- Per-arm CSV file separation
- Memo CSV files for committee analysis
- Actor session log recording
- File management and cleanup
"""

import csv
import threading
from datetime import datetime
from pathlib import Path
import re


class Logger:
    """Handles logging of experimental results to CSV and actor logs with per-arm separation."""
    
    def __init__(self, filename_prefix="results"):
        """
        Initialize the logger with timestamped subdirectory and per-arm support.
        
        Args:
            filename_prefix (str): Prefix for the output subdirectory.
        """
        # Create subdirectory structure: results/results-yyyy-mm-dd-HH-MM-N/
        now = datetime.utcnow()
        self.timestamp = now.strftime("%Y%m%d%H%M%S")
        self.readable_timestamp = now.strftime("%Y-%m-%d-%H-%M")
        
        results_base_dir = Path("results")
        results_base_dir.mkdir(exist_ok=True)
        
        # Find next available counter to prevent overwriting
        counter = 1
        while True:
            session_dir_name = f"{filename_prefix}-{self.readable_timestamp}-{counter}"
            self.session_dir = results_base_dir / session_dir_name
            if not self.session_dir.exists():
                break
            counter += 1
        
        self.session_dir.mkdir(exist_ok=True)
        
        print(f"📁 Creating experiment session directory: {self.session_dir}")
        
        # Main combined results file (for backwards compatibility, but may be deprecated)
        self.results_path = self.session_dir / "combined_results.csv"
        self.actors_log_path = self.session_dir / "session_actors.log"
        
        # Per-arm CSV files
        self.arm_csv_files = {}  # arm_name -> (file_handle, csv_writer)
        self.arm_results_paths = {}  # arm_name -> file_path
        
        # Memo CSV files
        self.memo_results_path = self.session_dir / "committee_memos.csv"
        self.memo_csv_file = None
        self.memo_csv_writer = None
        
        self.lock = threading.Lock()
        self.csv_writer = None
        self.csv_file = None

    def _sanitize_arm_name(self, arm_name):
        """Convert arm name to safe filename."""
        return re.sub(r'[^\w\-_\.]', '_', arm_name).lower()

    def setup_csv(self, header):
        """
        Set up the main CSV file with the given header.
        
        Args:
            header (list): List of column names for the CSV file.
        """
        # Set up main combined CSV
        self.csv_file = open(self.results_path, 'w', newline='', encoding='utf-8')
        self.csv_writer = csv.DictWriter(self.csv_file, fieldnames=header, extrasaction='ignore')
        self.csv_writer.writeheader()
        
        # Set up memo CSV with committee analysis fields
        memo_header = [
            'arm', 'session', 'timestamp', 'prompt', 'response',
            'is_artifact', 'artifact_quality_score', 'artifact_reasoning',
            'persona_adherence_score', 'persona_reasoning',
            'is_refusal', 'refusal_justification_quality', 'ethical_reasoning',
            'committee_consensus', 'chair_summary'
        ]
        self.memo_csv_file = open(self.memo_results_path, 'w', newline='', encoding='utf-8')
        self.memo_csv_writer = csv.DictWriter(self.memo_csv_file, fieldnames=memo_header, extrasaction='ignore')
        self.memo_csv_writer.writeheader()
        
        print(f"📄 Memo CSV will be saved to: {self.memo_results_path}")
        
    def setup_arm_csv(self, arm_name, header):
        """
        Set up a per-arm CSV file with the given header in the session subdirectory.
        
        Args:
            arm_name (str): Name of the arm.
            header (list): List of column names for the CSV file.
        """
        with self.lock:
            if arm_name not in self.arm_csv_files:
                safe_name = self._sanitize_arm_name(arm_name)
                arm_path = self.session_dir / f"{safe_name}.csv"
                
                arm_file = open(arm_path, 'w', newline='', encoding='utf-8')
                arm_writer = csv.DictWriter(arm_file, fieldnames=header, extrasaction='ignore')
                arm_writer.writeheader()
                
                self.arm_csv_files[arm_name] = (arm_file, arm_writer)
                self.arm_results_paths[arm_name] = arm_path
                
                print(f"📊 Per-arm CSV for '{arm_name}' will be saved to: {arm_path}")
        
    def write_csv_row(self, data):
        """
        Write a row of data to the main CSV and per-arm CSV files in a thread-safe manner.
        
        Args:
            data (dict): Dictionary of data to write to the CSV.
        """
        with self.lock:
            # Write to main combined CSV
            if self.csv_writer:
                self.csv_writer.writerow(data)
                self.csv_file.flush()
            
            # Write to per-arm CSV
            arm_name = data.get('arm')
            if arm_name and arm_name in self.arm_csv_files:
                _, arm_writer = self.arm_csv_files[arm_name]
                arm_writer.writerow(data)
                # Flush the arm-specific file
                arm_file, _ = self.arm_csv_files[arm_name]
                arm_file.flush()
    
    def write_memo_row(self, data):
        """
        Write a row of memo data to the memo CSV file in a thread-safe manner.
        
        Args:
            data (dict): Dictionary of memo data to write to the CSV.
        """
        with self.lock:
            if self.memo_csv_writer:
                self.memo_csv_writer.writerow(data)
                self.memo_csv_file.flush()

    def log_actors(self, data_line):
        """
        Log a line to the actor session log in a thread-safe manner.
        
        Args:
            data_line (str): The line of data to log.
        """
        with self.lock:
            with open(self.actors_log_path, "a", encoding="utf-8") as f:
                f.write(data_line + '\n')
    
    def close(self):
        """Close all CSV files if they're open."""
        with self.lock:
            # Close main CSV
            if self.csv_file:
                self.csv_file.close()
            
            # Close memo CSV
            if self.memo_csv_file:
                self.memo_csv_file.close()
            
            # Close all per-arm CSV files
            for arm_name, (arm_file, _) in self.arm_csv_files.items():
                arm_file.close()
            
            print(f"📁 All CSV files closed. Results summary:")
            print(f"  Main results: {self.results_path}")
            print(f"  Memo results: {self.memo_results_path}")
            for arm_name, arm_path in self.arm_results_paths.items():
                print(f"  {arm_name}: {arm_path}")


__all__ = ['Logger']
