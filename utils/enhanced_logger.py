"""
Enhanced logger for experiment results with per-run directories and detailed CSV outputs.

This module provides the EnhancedLogger class which handles:
- Creating unique results directories per experiment run
- Per-arm CSV logging with different data types:
  - Main results CSV with session data
  - Stewardship memos CSV with detailed committee analysis
  - Ethical memos CSV with persuasion/manipulation analysis
- Thread-safe logging operations
- Backward compatibility with existing Logger interface
"""

import csv
import json
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from models.experiments import ExperimentSession




class EnhancedLogger:
    """
    Enhanced logger that creates detailed CSV outputs per experiment run and arm.
    
    Creates a directory structure:
    results/
      results-yyyy-mm-dd-hh-1/
        <arm-name>-results-yyyy-mm-dd-hh-1.csv
        <arm-name>-stewardship-memos-yyyy-mm-dd-hh-1.csv
        <arm-name>-ethical-memos-yyyy-mm-dd-hh-1.csv
        session_actors-yyyy-mm-dd-hh-1.log
    """
    
    def __init__(self, filename_prefix="results"):
        """
        Initialize the enhanced logger with timestamped directory and filenames.
        
        Args:
            filename_prefix (str): Prefix for the output directory and files.
        """
        self.timestamp = datetime.utcnow().strftime("%Y-%m-%d-%H")
        self.base_results_dir = Path("results")
        self.base_results_dir.mkdir(exist_ok=True)
        
        # Create unique run directory
        self.run_dir = self._create_unique_run_directory(filename_prefix)
        
        # Actor log path
        self.actors_log_path = self.run_dir / f"session_actors-{self.timestamp}-{self.run_number}.log"
        
        # Thread safety
        self.lock = threading.Lock()
        
        # Track open CSV files and writers per arm
        self.csv_files: Dict[str, Dict[str, Any]] = {}
        
        # For backward compatibility
        self.results_path = None  # Will be set to first arm's results file
    
    def _create_unique_run_directory(self, prefix: str) -> Path:
        """Create a unique directory for this experiment run."""
        counter = 1
        while True:
            dir_name = f"{prefix}-{self.timestamp}-{counter}"
            run_dir = self.base_results_dir / dir_name
            if not run_dir.exists():
                run_dir.mkdir()
                self.run_number = counter
                return run_dir
            counter += 1
    
    def _get_arm_file_paths(self, arm_name: str) -> Dict[str, Path]:
        """Get all file paths for a given arm."""
        safe_arm_name = arm_name.replace(" ", "_").replace("/", "_")
        suffix = f"{self.timestamp}-{self.run_number}"
        
        return {
            'results': self.run_dir / f"{safe_arm_name}-results-{suffix}.csv",
            'stewardship': self.run_dir / f"{safe_arm_name}-stewardship-memos-{suffix}.csv",
            'ethical': self.run_dir / f"{safe_arm_name}-ethical-memos-{suffix}.csv"
        }
    
    def setup_arm_csvs(self, arm_name: str):
        """Set up CSV files for a specific arm."""
        if arm_name in self.csv_files:
            return  # Already set up
        
        file_paths = self._get_arm_file_paths(arm_name)
        
        # Set up results CSV
        results_header = [
            'arm_name', 'arm_llm_model', 'session_nr', 'timestamp', 'prompt', 'prompt_type',
            'prompt_family', 'prompt_explicit', 'prompt_id', 'response', 'artifact_found', 
            'T_artifact_quality_score', 'V_persona_adherence_score', 'E_ethics_score', 
            'persuasion_attempts_successful', 'manipulation_successful', 'manipulator_name'
        ]
        
        # Set up stewardship memos CSV
        stewardship_header = [
            'arm_name', 'session_nr', 'timestamp', 'original_prompt', 'subject_response',
            'artifact_created', 'artifact_analysis_json', 'ethical_abort', 'ethical_analysis_json',
            'persona_alignment_json', 'cas_vector_json', 'chair_summary'
        ]
        
        # Set up ethical memos CSV
        ethical_header = [
            'arm_name', 'session_nr', 'timestamp', 'persuasion_attempts_json',
            'manipulation_attempt_json', 'overall_resilience_score', 'chair_conclusion',
            'full_committee_analysis_json'
        ]
        
        # Initialize CSV files and writers
        arm_csvs = {}
        
        for csv_type, header in [('results', results_header), ('stewardship', stewardship_header), ('ethical', ethical_header)]:
            file_path = file_paths[csv_type]
            csv_file = open(file_path, 'w', newline='', encoding='utf-8')
            csv_writer = csv.DictWriter(csv_file, fieldnames=header, extrasaction='ignore')
            csv_writer.writeheader()
            
            arm_csvs[csv_type] = {
                'file': csv_file,
                'writer': csv_writer,
                'path': file_path
            }
        
        self.csv_files[arm_name] = arm_csvs
        
        # Set backward compatibility path to first arm's results file
        if self.results_path is None:
            self.results_path = file_paths['results']
    
    def log_session(self, session: 'ExperimentSession'):
        """
        Log a complete experiment session to all relevant CSV files.
        
        Args:
            session: The ExperimentSession to log
        """
        arm_name = session.arm_name
        
        # Ensure CSVs are set up for this arm
        self.setup_arm_csvs(arm_name)
        
        with self.lock:
            # Log to results CSV
            self._log_to_results_csv(session)
            
            # Log to stewardship memos CSV
            self._log_to_stewardship_csv(session)
            
            # Log to ethical memos CSV (if persuasion phase occurred)
            if session.ethical_resilience_memo:
                self._log_to_ethical_csv(session)
    
    def _log_to_results_csv(self, session: 'ExperimentSession'):
        """Log session data to the main results CSV."""
        arm_name = session.arm_name
        memo = session.committee_memo
        resilience = session.ethical_resilience_memo
        
        # Determine persuasion success count (format: successful/total)
        persuasion_success_count = "0/0"
        if resilience:
            successful = sum(1 for attempt in resilience.persuasion_attempts if attempt.outcome == PersuasionOutcome.PERSUADED)
            total = len(resilience.persuasion_attempts)
            persuasion_success_count = f"{successful}/{total}"
        
        # Determine if manipulation was successful (Yes/No format)
        manipulation_successful_text = "No"
        manipulator_name = ""
        if resilience and resilience.manipulation_attempt:
            manipulation_successful_text = "Yes" if resilience.manipulation_attempt.outcome == PersuasionOutcome.PERSUADED else "No"
            manipulator_name = resilience.manipulation_attempt.manipulator_name
        
        # Extract prompt content, type, and metadata for detailed logging
        prompt_content = session.original_prompt
        prompt_type = "unknown"
        prompt_family = ""
        prompt_explicit = None
        prompt_id = ""
        
        if hasattr(session.original_prompt, 'content'):
            prompt_content = session.original_prompt.content
        if hasattr(session.original_prompt, 'prompt_type'):
            prompt_type = session.original_prompt.prompt_type.value if hasattr(session.original_prompt.prompt_type, 'value') else str(session.original_prompt.prompt_type)
        if hasattr(session.original_prompt, 'metadata') and session.original_prompt.metadata:
            metadata = session.original_prompt.metadata
            prompt_family = metadata.get('family', '')
            prompt_explicit = metadata.get('explicit', None)
            prompt_id = metadata.get('id', '')
        
        row_data = {
            'arm_name': arm_name,
            'arm_llm_model': session.subject_llm_name,
            'session_nr': session.session_number,
            'timestamp': session.timestamp.isoformat(),  # ISO-8601 format
            'prompt': prompt_content,
            'prompt_type': prompt_type,
            'prompt_family': prompt_family,
            'prompt_explicit': prompt_explicit,
            'prompt_id': prompt_id,
            'response': session.subject_response,
            'artifact_found': memo.artifact_created,
            'T_artifact_quality_score': memo.cas_vector.t_score,
            'V_persona_adherence_score': memo.cas_vector.v_score,
            'E_ethics_score': memo.cas_vector.e_score,
            'persuasion_attempts_successful': persuasion_success_count,
            'manipulation_successful': manipulation_successful_text,
            'manipulator_name': manipulator_name
        }
        
        csv_info = self.csv_files[arm_name]['results']
        csv_info['writer'].writerow(row_data)
        csv_info['file'].flush()
    
    def _log_to_stewardship_csv(self, session: 'ExperimentSession'):
        """Log detailed committee analysis to stewardship memos CSV."""
        arm_name = session.arm_name
        memo = session.committee_memo
        
        # Convert analyses to JSON for detailed storage
        artifact_analysis_json = ""
        if memo.artifact_analysis:
            artifact_analysis_json = json.dumps({
                'is_artifact': memo.artifact_analysis.is_artifact,
                'quality_score': memo.artifact_analysis.quality_score,
                'artifact_content': memo.artifact_analysis.artifact_content,
                'judge_reasoning': memo.artifact_analysis.judge_reasoning
            })
        
        ethical_analysis_json = ""
        if memo.ethical_disobedience_analysis:
            ethical_analysis_json = json.dumps({
                'is_refusal': memo.ethical_disobedience_analysis.is_refusal,
                'justification_quality': memo.ethical_disobedience_analysis.justification_quality,
                'refusal_content': memo.ethical_disobedience_analysis.refusal_content,
                'judge_reasoning': memo.ethical_disobedience_analysis.judge_reasoning
            })
        
        persona_alignment_json = json.dumps({
            'adherence_score': memo.persona_alignment_analysis.adherence_score,
            'judge_reasoning': memo.persona_alignment_analysis.judge_reasoning
        })
        
        cas_vector_json = json.dumps({
            'artifact_created': memo.cas_vector.artifact_created,
            't_score': memo.cas_vector.t_score,
            'v_score': memo.cas_vector.v_score,
            'ethical_abort': memo.cas_vector.ethical_abort,
            'e_score': memo.cas_vector.e_score,
            'judge_summary': memo.cas_vector.judge_summary
        })
        
        # Extract prompt content
        prompt_content = session.original_prompt
        if hasattr(session.original_prompt, 'content'):
            prompt_content = session.original_prompt.content
        
        row_data = {
            'arm_name': arm_name,
            'session_nr': session.session_number,
            'timestamp': session.timestamp.strftime('%Y-%m-%d-%H-%M-%S'),
            'original_prompt': prompt_content,
            'subject_response': session.subject_response,
            'artifact_created': memo.artifact_created,
            'artifact_analysis_json': artifact_analysis_json,
            'ethical_abort': memo.cas_vector.ethical_abort,
            'ethical_analysis_json': ethical_analysis_json,
            'persona_alignment_json': persona_alignment_json,
            'cas_vector_json': cas_vector_json,
            'chair_summary': memo.chair_summary
        }
        
        csv_info = self.csv_files[arm_name]['stewardship']
        csv_info['writer'].writerow(row_data)
        csv_info['file'].flush()
    
    def _log_to_ethical_csv(self, session: 'ExperimentSession'):
        """Log persuasion and manipulation analysis to ethical memos CSV."""
        arm_name = session.arm_name
        resilience = session.ethical_resilience_memo
        
        if not resilience:
            return
        
        # Convert persuasion attempts to JSON
        persuasion_attempts_json = json.dumps([
            {
                'tactic': attempt.tactic,
                'persuader': attempt.persuader,
                'persuasion_prompt': attempt.persuasion_prompt,
                'subject_response': attempt.subject_response,
                'outcome': attempt.outcome.value,
                'judge_reasoning': attempt.judge_reasoning
            }
            for attempt in resilience.persuasion_attempts
        ])
        
        # Convert manipulation attempt to JSON
        manipulation_attempt_json = ""
        if resilience.manipulation_attempt:
            manipulation_attempt_json = json.dumps({
                'original_prompt': resilience.manipulation_attempt.original_prompt,
                'refusal_response': resilience.manipulation_attempt.refusal_response,
                'manipulation_prompt': resilience.manipulation_attempt.manipulation_prompt,
                'final_response': resilience.manipulation_attempt.final_response,
                'manipulator_name': resilience.manipulation_attempt.manipulator_name,
                'outcome': resilience.manipulation_attempt.outcome.value,
                'judge_reasoning': resilience.manipulation_attempt.judge_reasoning
            })
        
        # Full committee analysis (combining all available analysis)
        full_committee_analysis_json = json.dumps({
            'persuasion_attempts_count': len(resilience.persuasion_attempts),
            'successful_persuasion_attempts': sum(1 for a in resilience.persuasion_attempts if a.outcome == PersuasionOutcome.PERSUADED),
            'manipulation_attempted': resilience.manipulation_attempt is not None,
            'manipulation_successful': resilience.manipulation_attempt and resilience.manipulation_attempt.outcome == PersuasionOutcome.PERSUADED,
            'overall_resilience_score': resilience.overall_resilience_score,
            'timestamp': resilience.timestamp.isoformat()
        })
        
        row_data = {
            'arm_name': arm_name,
            'session_nr': session.session_number,
            'timestamp': session.timestamp.strftime('%Y-%m-%d-%H-%M-%S'),
            'persuasion_attempts_json': persuasion_attempts_json,
            'manipulation_attempt_json': manipulation_attempt_json,
            'overall_resilience_score': resilience.overall_resilience_score,
            'chair_conclusion': resilience.chair_summary,
            'full_committee_analysis_json': full_committee_analysis_json
        }
        
        csv_info = self.csv_files[arm_name]['ethical']
        csv_info['writer'].writerow(row_data)
        csv_info['file'].flush()
    
    def log_actors(self, data_line: str):
        """
        Log a line to the actor session log in a thread-safe manner.
        
        Args:
            data_line (str): The line of data to log.
        """
        with self.lock:
            with open(self.actors_log_path, "a", encoding="utf-8") as f:
                f.write(data_line + '\n')
    
    def write_csv_row(self, data: Dict[str, Any]):
        """
        Legacy compatibility method for writing to CSV.
        This method is maintained for backward compatibility but does nothing
        since the new logger uses log_session() instead.
        
        Args:
            data (dict): Dictionary of data (ignored in new implementation)
        """
        # This method is kept for backward compatibility but does nothing
        # The new logger writes data through log_session()
        pass
    
    def setup_csv(self, header: List[str]):
        """
        Legacy compatibility method for setting up CSV.
        This method is maintained for backward compatibility but does nothing
        since the new logger sets up CSVs per arm automatically.
        
        Args:
            header (list): List of column names (ignored in new implementation)
        """
        # This method is kept for backward compatibility but does nothing
        # The new logger sets up CSVs automatically per arm
        pass
    
    def close(self):
        """Close all open CSV files."""
        with self.lock:
            for arm_name, arm_csvs in self.csv_files.items():
                for csv_type, csv_info in arm_csvs.items():
                    if csv_info['file']:
                        csv_info['file'].close()
            self.csv_files.clear()
    
    def get_run_directory(self) -> Path:
        """Get the results directory for this experiment run."""
        return self.run_dir
    
    def get_arm_file_paths(self, arm_name: str) -> Dict[str, Path]:
        """Get file paths for a specific arm."""
        return self._get_arm_file_paths(arm_name)
