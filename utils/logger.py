"""
Logger for experiment results and actor session logs.

This module provides the Logger class which handles:
- CSV result logging with thread safety
- Actor session log recording
- File management and cleanup
"""

import csv
import threading
from datetime import datetime
from pathlib import Path


class Logger:
    """Handles logging of experimental results to CSV and actor logs."""
    
    def __init__(self, filename_prefix="results"):
        """
        Initialize the logger with timestamped filenames.
        
        Args:
            filename_prefix (str): Prefix for the output files.
        """
        self.timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
        results_dir = Path("results")
        results_dir.mkdir(exist_ok=True)
        self.results_path = results_dir / f"{filename_prefix}-{self.timestamp}.csv"
        self.actors_log_path = results_dir / f"session_actors-{self.timestamp}.log"
        self.lock = threading.Lock()
        self.csv_writer = None
        self.csv_file = None

    def setup_csv(self, header):
        """
        Set up the CSV file with the given header.
        
        Args:
            header (list): List of column names for the CSV file.
        """
        self.csv_file = open(self.results_path, 'w', newline='', encoding='utf-8')
        self.csv_writer = csv.DictWriter(self.csv_file, fieldnames=header, extrasaction='ignore')
        self.csv_writer.writeheader()
        
    def write_csv_row(self, data):
        """
        Write a row of data to the CSV file in a thread-safe manner.
        
        Args:
            data (dict): Dictionary of data to write to the CSV.
        """
        with self.lock:
            if self.csv_writer:
                self.csv_writer.writerow(data)
                self.csv_file.flush()

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
        """Close the CSV file if it's open."""
        if self.csv_file:
            self.csv_file.close()
