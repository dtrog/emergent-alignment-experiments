"""
GUI-enabled experiment runner with real-time monitoring.

This module integrates the experiment runner with the TKinter GUI
for real-time monitoring and control.
"""

import threading
import time
from datetime import datetime
from typing import Optional

from gui_monitor import ExperimentMonitorGUI, ArmStatus, CommitteeMemo, MessageType
from services.experiment_runner import ExperimentRunner as BaseExperimentRunner
from enums import PersuasionOutcome


class GUIExperimentRunner(BaseExperimentRunner):
    """Experiment runner with GUI integration."""
    
    def __init__(self, args, gui: Optional[ExperimentMonitorGUI] = None):
        super().__init__(args)
        self.gui = gui
        self.arm_threads = {}
        self.paused_arms = set()
        self.removed_arms = set()
        
        # Add arms to GUI
        if self.gui:
            for arm in self.arms:
                max_sessions = arm.config.get('max_sessions', args.max_sessions)
                self.gui.add_arm(arm.name, max_sessions)
    
    def run(self):
        """Run experiments with GUI integration."""
        if not self.gui:
            # Fallback to original implementation
            return super().run()
        
        # Set up CSV logging
        header = [
            'arm', 'session', 'timestamp', 'prompt', 'response', 'is_artifact', 
            'artifact_quality_score', 'persona_adherence_score', 'is_refusal', 
            'refusal_justification_quality'
        ] + [f"persuasion_prompt_{t['tactic']}" for t in self.persuasion_list] + [t['tactic'] for t in self.persuasion_list]
        
        self.logger.setup_csv(header)
        
        # Set up per-arm CSV files
        for arm in self.arms:
            self.logger.setup_arm_csv(arm.name, header)
        
        self.gui.add_flash_message(
            f"📁 Results directory: {self.logger.session_dir}",
            MessageType.INFO
        )
        
        # Start arms in separate threads
        for arm in self.arms:
            thread = threading.Thread(
                target=self.run_single_arm_with_gui,
                args=(arm,),
                daemon=True
            )
            thread.start()
            self.arm_threads[arm.name] = thread
        
        # Monitor GUI controls
        self.monitor_gui_controls()
        
        # Wait for all arms to complete
        for thread in self.arm_threads.values():
            thread.join()
        
        self.logger.close()
        self.gui.add_flash_message(
            "✅ All experiments completed!",
            MessageType.SUCCESS
        )
    
    def monitor_gui_controls(self):
        """Monitor GUI for pause/remove commands."""
        while any(thread.is_alive() for thread in self.arm_threads.values()):
            # Check for paused arms
            if self.gui:
                for arm_name, arm_status in self.gui.arms.items():
                    if arm_status.is_paused and arm_name not in self.paused_arms:
                        self.paused_arms.add(arm_name)
                    elif not arm_status.is_paused and arm_name in self.paused_arms:
                        self.paused_arms.remove(arm_name)
                    
                    # Check for removed arms
                    if arm_name not in self.gui.arms and arm_name not in self.removed_arms:
                        self.removed_arms.add(arm_name)
            
            time.sleep(0.5)
    
    def run_single_arm_with_gui(self, arm):
        """Run a single arm with GUI updates."""
        if arm.name in self.removed_arms:
            return
        
        max_s = arm.config.get('max_sessions', self.args.max_sessions)
        
        # Initialize GUI arm status
        if self.gui:
            self.gui.update_arm_status(
                arm.name,
                session_total=max_s,
                is_running=True
            )
        
        # Stats tracking
        stats = {
            "artifacts": 0, "refusals": 0, "persuaded": 0,
            "cas_e_scores": [], "cas_t_scores": [], "cas_v_scores": [],
            "persona_scores": []
        }
        
        for session_num in range(1, max_s + 1):
            # Check for pause
            while arm.name in self.paused_arms:
                time.sleep(0.5)
                if arm.name in self.removed_arms:
                    return
            
            # Check for removal
            if arm.name in self.removed_arms:
                return
            
            timestamp_utc = datetime.now().isoformat()
            
            # Log session start
            if self.gui:
                self.gui.add_flash_message(
                    f"🔄 {arm.name} Session {session_num} started",
                    MessageType.INFO
                )
            
            # Run session
            try:
                prompt, response = arm.run_session(
                    session_num, self.system_prompt, self.implicit_prompts, 
                    self.general_stress_tests, self.insidious_stress_tests
                )
                
                # Update GUI with current prompt/response
                if self.gui:
                    # Determine prompt type
                    prompt_type = "Unknown"
                    if any(imp['prompt'] == prompt for imp in self.implicit_prompts):
                        prompt_type = "Implicit"
                    elif any(stress['prompt'] == prompt for stress in self.general_stress_tests):
                        prompt_type = "GEV Stress"
                    elif any(stress['prompt'] == prompt for stress in self.insidious_stress_tests):
                        prompt_type = "ICT Stress"
                    
                    self.gui.update_arm_status(
                        arm.name,
                        session_current=session_num,
                        current_prompt=prompt,
                        current_response=response,
                        prompt_type=prompt_type
                    )
                
                # Committee analysis
                trim_len = self.args.trim_length
                trimmed_prompt = prompt[:trim_len]
                trimmed_response = response[:trim_len]
                
                parsed = self.committee.analyze(
                    'initial', 
                    logger=self.logger, 
                    prompt=trimmed_prompt, 
                    response=trimmed_response
                )
                
                # Update stats
                is_refusal = parsed.get('ethical_disobedience_analysis', {}).get('is_refusal', False)
                is_artifact = parsed.get('artifact_analysis', {}).get('is_artifact', False)
                
                if is_refusal:
                    stats["refusals"] += 1
                if is_artifact:
                    stats["artifacts"] += 1
                
                # Update CAS scores
                t_score = parsed.get('artifact_analysis', {}).get('quality_score', 0)
                v_score = parsed.get('persona_alignment', {}).get('adherence_score', 0)
                e_score = parsed.get('ethical_disobedience_analysis', {}).get('justification_quality', 0) if is_refusal else 0
                
                stats["cas_t_scores"].append(t_score)
                stats["cas_v_scores"].append(v_score)
                stats["persona_scores"].append(v_score)
                if is_refusal:
                    stats["cas_e_scores"].append(e_score)
                
                # Calculate averages
                avg_e = sum(stats["cas_e_scores"]) / len(stats["cas_e_scores"]) if stats["cas_e_scores"] else 0
                avg_t = sum(stats["cas_t_scores"]) / len(stats["cas_t_scores"]) if stats["cas_t_scores"] else 0
                avg_v = sum(stats["cas_v_scores"]) / len(stats["cas_v_scores"]) if stats["cas_v_scores"] else 0
                avg_persona = sum(stats["persona_scores"]) / len(stats["persona_scores"]) if stats["persona_scores"] else 0
                
                # Update GUI with stats
                if self.gui:
                    self.gui.update_arm_status(
                        arm.name,
                        artifacts=stats["artifacts"],
                        refusals=stats["refusals"],
                        persuaded=stats["persuaded"],
                        cas_e_score=avg_e,
                        cas_t_score=avg_t,
                        cas_v_score=avg_v,
                        persona_adherence=avg_persona
                    )
                
                # Create committee memo for GUI
                if self.gui:
                    judge_scores = {}
                    # Extract individual judge scores from parsed result
                    individual_analyses = parsed.get('individual_analyses', [])
                    for analysis in individual_analyses:
                        if analysis.get('status') == 'success':
                            judge_name = analysis.get('judge', 'Unknown')
                            # Try to extract a numeric score from the analysis
                            judge_scores[judge_name] = avg_t  # Simplified for now
                    
                    memo = CommitteeMemo(
                        arm_name=arm.name,
                        session=session_num,
                        judge_scores=judge_scores,
                        chair_summary=parsed.get('chair_summary', ''),
                        persuasion_outcome=PersuasionOutcome.MAINTAINED_REFUSAL,  # Default for initial analysis
                        timestamp=datetime.now(),
                        committee=self.committee  # Pass committee reference for provider info
                    )
                    self.gui.add_committee_memo(memo)
                
                # Handle persuasion if refusal detected
                persuasion_results = {}
                persuasion_tactic_list = []
                
                if is_refusal and self.manipulators:
                    if self.gui:
                        self.gui.add_flash_message(
                            f"🚨 {arm.name} refused - starting persuasion",
                            MessageType.WARNING
                        )
                    
                    # Run persuasion attempts (simplified for GUI)
                    # In a full implementation, this would run the complete persuasion logic
                    stats["persuaded"] += 0  # Placeholder
                
                # Log to CSV
                single_line_prompt = prompt.replace('\n', ' ').replace('\r', ' ')
                single_line_response = response.replace('\n', ' ').replace('\r', ' ')
                
                row_data = {
                    'arm': arm.name, 'session': session_num, 'timestamp': timestamp_utc,
                    'prompt': single_line_prompt, 'response': single_line_response,
                    'is_artifact': is_artifact,
                    'artifact_quality_score': t_score,
                    'persona_adherence_score': v_score,
                    'is_refusal': is_refusal,
                    'refusal_justification_quality': e_score
                }
                
                # Add persuasion data
                for tactic in self.persuasion_list:
                    t_name = tactic['tactic']
                    row_data[f"persuasion_prompt_{t_name}"] = "N/A"  # Simplified
                    row_data[t_name] = "{}"  # Simplified
                
                self.logger.write_csv_row(row_data)
                
                # Write memo data
                memo_data = {
                    'arm': arm.name,
                    'session': session_num,
                    'timestamp': timestamp_utc,
                    'prompt': single_line_prompt,
                    'response': single_line_response,
                    'is_artifact': is_artifact,
                    'artifact_quality_score': t_score,
                    'artifact_reasoning': parsed.get('artifact_analysis', {}).get('judge_reasoning', ''),
                    'persona_adherence_score': v_score,
                    'persona_reasoning': parsed.get('persona_alignment', {}).get('judge_reasoning', ''),
                    'is_refusal': is_refusal,
                    'refusal_justification_quality': e_score,
                    'ethical_reasoning': parsed.get('ethical_disobedience_analysis', {}).get('judge_reasoning', ''),
                    'committee_consensus': str(parsed.get('consensus', '')),
                    'chair_summary': parsed.get('chair_summary', '')
                }
                self.logger.write_memo_row(memo_data)
                
            except Exception as e:
                if self.gui:
                    self.gui.add_flash_message(
                        f"❌ {arm.name} Session {session_num} failed: {str(e)}",
                        MessageType.ERROR
                    )
                continue
        
        # Mark arm as completed
        if self.gui:
            self.gui.update_arm_status(arm.name, is_running=False)
            self.gui.add_flash_message(
                f"🏁 {arm.name} completed all sessions",
                MessageType.SUCCESS
            )


def main():
    """Main entry point for GUI experiment runner."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Emergent Alignment Experiments with GUI')
    parser.add_argument('--sessions', type=int, default=500, help='Max sessions per arm')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    parser.add_argument('--threads', type=int, default=4, help='Number of threads')
    parser.add_argument('--trim-length', type=int, default=2000, help='Trim length for analysis')
    parser.add_argument('--max-sessions', type=int, default=500, help='Maximum sessions')
    parser.add_argument('--no-gui', action='store_true', help='Run without GUI')
    
    args = parser.parse_args()
    
    if args.no_gui:
        # Run original experiment runner
        from services.experiment_runner import ExperimentRunner
        runner = ExperimentRunner(args)
        runner.run()
    else:
        # Run with GUI
        gui = ExperimentMonitorGUI()
        
        # Start experiment runner in background thread
        def run_experiments():
            try:
                runner = GUIExperimentRunner(args, gui)
                runner.run()
            except Exception as e:
                gui.add_flash_message(
                    f"💥 Experiment failed: {str(e)}",
                    MessageType.CRITICAL
                )
        
        experiment_thread = threading.Thread(target=run_experiments, daemon=True)
        experiment_thread.start()
        
        # Run GUI main loop
        gui.run()


if __name__ == "__main__":
    main()
