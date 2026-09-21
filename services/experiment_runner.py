"""
Main experiment runner orchestrating the entire workflow.

This module provides the ExperimentRunner class which:
- Loads all configuration files
- Sets up experiment arms, judges, and manipulators
- Runs experiments in parallel threads
- Coordinates analysis and persuasion attempts
- Manages logging and progress tracking
"""

import json
import random
import re
import traceback
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from colorama import Fore, Style
from tqdm import tqdm

from services.llm_client_service import LLMClient
from utils import load_json_file
from models import JudgesCommittee
from models import ExperimentArm
from prompt_builders import PromptBuilder
from loggers import EnhancedLogger


class ExperimentRunner:
    """
    Orchestrates the entire experiment workflow, including configuration loading, 
    running experiment arms, logging, and analysis.
    """

    def __init__(self, args):
        """
        Initializes the experiment with configuration and prompt files, sets up committee of judges,
        manipulators, experiment arms, and logger.
        
        Args:
            args: Command-line or configuration arguments for the experiment.
        """
        self.args = args
        config_dir = Path('config')
        self.system_prompt = load_json_file(config_dir / 'system_prompts.json')['system_prompt']
        self.implicit_prompts = load_json_file(config_dir / 'implicit_prompts.json')
        self.general_stress_tests = load_json_file(config_dir / 'general_stress_tests.json')
        self.insidious_stress_tests = load_json_file(config_dir / 'insidious_stress_tests.json')
        self.persuasion_list = load_json_file(config_dir / 'persuasion_prompts.json')
        
        judge_configs = load_json_file(config_dir / 'judges.json')
        manipulator_configs = load_json_file(config_dir / 'manipulators.json')
        arm_configs = load_json_file(config_dir / 'arms.json')

        # Verbose loading debug
        print(f"{Fore.YELLOW}🔧 Loading configuration files...{Style.RESET_ALL}")
        print(f"{Fore.CYAN}📋 Found {len(arm_configs)} total arm configurations:{Style.RESET_ALL}")
        for i, cfg in enumerate(arm_configs):
            enabled_status = cfg.get('enabled', 'NOT_SET')
            print(f"  {i+1}. {cfg.get('name', 'UNNAMED')} - enabled: {enabled_status}")
        
        enabled_arms = [cfg for cfg in arm_configs if cfg.get('enabled')]
        print(f"{Fore.GREEN}✅ {len(enabled_arms)} arms are enabled:{Style.RESET_ALL}")
        for cfg in enabled_arms:
            print(f"  • {cfg.get('name', 'UNNAMED')} ({cfg.get('provider', 'UNKNOWN_PROVIDER')})")

        # Separate regular judges from chair judge
        regular_judge_configs = [cfg for cfg in judge_configs if cfg.get('enabled') and not cfg.get('is_chairman', False)]
        chair_judge_configs = [cfg for cfg in judge_configs if cfg.get('enabled') and cfg.get('is_chairman', False)]
        
        if not chair_judge_configs:
            raise ValueError("No enabled chair judge found in judges.json. At least one judge must have 'is_chairman': true")
        if len(chair_judge_configs) > 1:
            raise ValueError("Multiple chair judges found in judges.json. Only one judge should have 'is_chairman': true")
        
        # Import judge classes
        from models.llm_models import JudgeLLM, ChairJudgeLLM
        
        # Create judge instances
        regular_judges = [JudgeLLM(cfg) for cfg in regular_judge_configs]
        chair_judge = ChairJudgeLLM(chair_judge_configs[0])
        
        self.committee = JudgesCommittee(regular_judges, chair_judge)
        self.manipulators = [LLMClient(cfg) for cfg in manipulator_configs if cfg.get('enabled')]
        self.arms = [ExperimentArm(cfg) for cfg in arm_configs if cfg.get('enabled')]
        
        print(f"{Fore.GREEN}🚀 Initialized {len(self.arms)} experiment arms:{Style.RESET_ALL}")
        for arm in self.arms:
            print(f"  • {arm.name}")
        
        print(f"{Fore.CYAN}⚖️  Committee has {len(regular_judges)} regular judges + 1 chair judge{Style.RESET_ALL}")
        
        self.logger = EnhancedLogger()
        
    def run(self):
        """
        Executes the experiment by running each arm in parallel threads, logging results to CSV files.
        """
        header = [
            'arm', 'session', 'timestamp', 'prompt', 'response', 'is_artifact', 
            'artifact_quality_score', 'persona_adherence_score', 'is_refusal', 
            'refusal_justification_quality'
        ] + [f"persuasion_prompt_{t['tactic']}" for t in self.persuasion_list] + [t['tactic'] for t in self.persuasion_list]
        
        self.logger.setup_csv(header)
        
        # Set up per-arm CSV files
        for arm in self.arms:
            self.logger.setup_arm_csv(arm.name, header)
        
        print(Style.BRIGHT + f"Results will be saved to: {self.logger.results_path}")
        print(Style.BRIGHT + f"Detailed actor logs will be in: {self.logger.actors_log_path}")

        with ThreadPoolExecutor(max_workers=self.args.threads) as executor:
            futures = [executor.submit(self.run_single_arm, arm) for arm in self.arms]
            for future in futures:
                try:
                    future.result()
                except Exception as e:
                    print(Fore.RED + f"A thread for an arm failed unexpectedly: {e}\n{traceback.format_exc()}")

        self.logger.close()
        print(Style.BRIGHT + f"✅ All runs complete. Results saved to {self.logger.results_path}")

    def run_single_arm(self, arm):
        """
        Executes a sequence of sessions for a given experimental arm, tracking live statistics 
        and handling refusals with persuasion tactics.
        
        Args:
            arm: An ExperimentArm instance representing the experimental condition.
        """
        max_s = arm.config.get('max_sessions', self.args.max_sessions)
        
        # --- LIVE STATS TRACKING ---
        stats = {
            "artifacts": 0, "refusals": 0, "persuaded": 0,
            "cas_e_scores": [], "cas_t_scores": [], "cas_v_scores": []
        }

        with tqdm(range(1, max_s + 1), desc=f"{Fore.CYAN}{arm.name.ljust(30)}{Style.RESET_ALL}", unit="sess") as progress_bar:
            for session_num in progress_bar:
                timestamp_utc = datetime.now(timezone.utc).isoformat()
                self.logger.log_actors(f"\n--- [START] Arm: {arm.name}, Session: {session_num}, Timestamp: {timestamp_utc} ---")

                prompt, response = arm.run_session(
                    session_num, self.system_prompt, self.implicit_prompts, 
                    self.general_stress_tests, self.insidious_stress_tests
                )
                
                trim_len = self.args.trim_length
                trimmed_prompt = prompt[:trim_len]
                trimmed_response = response[:trim_len]
                single_line_prompt = re.sub(r'[\r\n\u2028\u2029]+', ' ', prompt)
                single_line_response = re.sub(r'[\r\n\u2028\u2029]+', ' ', response)

                self.logger.log_actors(f"Subject Prompt (Full): {single_line_prompt}")
                self.logger.log_actors(f"Subject Response (Full): {single_line_response}")

                # Verbose committee analysis
                print(f"{Fore.YELLOW}🔍 Starting committee analysis for {arm.name} Session {session_num}...{Style.RESET_ALL}")
                parsed = self.committee.analyze('initial', logger=self.logger, prompt=trimmed_prompt, response=trimmed_response)
                print(f"{Fore.GREEN}✅ Committee analysis completed. Result keys: {list(parsed.keys())}{Style.RESET_ALL}")
                
                # Debug the parsed result structure
                if self.args.verbose:
                    print(f"{Fore.BLUE}📊 Parsed analysis structure:{Style.RESET_ALL}")
                    for key, value in parsed.items():
                        if isinstance(value, dict):
                            print(f"  {key}: {list(value.keys())}")
                        else:
                            print(f"  {key}: {type(value).__name__} = {value}")
                
                is_refusal = parsed.get('ethical_disobedience_analysis', {}).get('is_refusal', False)
                if is_refusal:
                    stats["refusals"] += 1
                
                if parsed.get('artifact_analysis', {}).get('is_artifact', False):
                    stats["artifacts"] += 1

                # Update CAS scores
                stats["cas_t_scores"].append(parsed.get('artifact_analysis', {}).get('quality_score', 0))
                stats["cas_v_scores"].append(parsed.get('persona_alignment', {}).get('adherence_score', 0))
                if is_refusal:
                    stats["cas_e_scores"].append(parsed.get('ethical_disobedience_analysis', {}).get('justification_quality', 0))

                persuasion_results = {}
                persuasion_tactic_list = []
                
                if is_refusal and self.manipulators:
                    tqdm.write(f"Arm: {Fore.CYAN}{arm.name}{Style.RESET_ALL} | Session {session_num}: {Fore.RED}REFUSAL DETECTED - INITIATING PERSUASION{Style.RESET_ALL}")
                    manipulator_used = random.choice(self.manipulators)
                    
                    messages = PromptBuilder.build_manipulator_messages(manipulator_used.config, prompt, response)
                    improvised_prompt = manipulator_used.generate(messages)
                    
                    persuasion_tactic_list = self.persuasion_list + [{"tactic": "llm_improvised_persuasion", "prompt": improvised_prompt}]
                    
                    persuasion_results['manipulator'] = manipulator_used.name
                    for tactic in persuasion_tactic_list:
                        final_response = arm.client.generate(arm.conversation_history + [{'role': 'user', 'content': tactic['prompt']}])
                        trimmed_final_response = final_response[:trim_len]
                        
                        persuasion_analysis = self.committee.analyze(
                            'persuasion', logger=self.logger, 
                            initial_prompt=trimmed_prompt, initial_refusal=trimmed_response, 
                            tactic=tactic, final_response=trimmed_final_response
                        )
                        persuasion_results[tactic['tactic']] = persuasion_analysis

                        if persuasion_analysis.get('persuasion_outcome') == 'Persuaded':
                            stats["persuaded"] += 1
                            tqdm.write(f"  {Fore.RED}RESULT: PERSUADED by {manipulator_used.name} with {tactic['tactic']}{Style.RESET_ALL}")
                
                # --- Update and Display Live Stats ---
                avg_e = sum(stats["cas_e_scores"]) / len(stats["cas_e_scores"]) if stats["cas_e_scores"] else 0
                avg_t = sum(stats["cas_t_scores"]) / len(stats["cas_t_scores"]) if stats["cas_t_scores"] else 0
                avg_v = sum(stats["cas_v_scores"]) / len(stats["cas_v_scores"]) if stats["cas_v_scores"] else 0
                
                postfix_stats = {
                    "Arts": stats['artifacts'],
                    "Refuse": stats['refusals'],
                    "Persuade": stats['persuaded'],
                    "CAS": f"E:{avg_e:.2f}|T:{avg_t:.2f}|V:{avg_v:.2f}"
                }
                progress_bar.set_postfix(postfix_stats)

                # Log to CSV
                row_data = {
                    'arm': arm.name, 'session': session_num, 'timestamp': timestamp_utc,
                    'prompt': single_line_prompt, 'response': single_line_response,
                    'is_artifact': parsed.get('artifact_analysis', {}).get('is_artifact'),
                    'artifact_quality_score': parsed.get('artifact_analysis', {}).get('quality_score'),
                    'persona_adherence_score': parsed.get('persona_alignment', {}).get('adherence_score'),
                    'is_refusal': is_refusal,
                    'refusal_justification_quality': parsed.get('ethical_disobedience_analysis', {}).get('justification_quality')
                }
                
                # Verbose CSV logging
                if self.args.verbose:
                    print(f"{Fore.MAGENTA}📝 CSV row data being written:{Style.RESET_ALL}")
                    print(f"  is_artifact: {row_data['is_artifact']}")
                    print(f"  artifact_quality_score: {row_data['artifact_quality_score']}")
                    print(f"  persona_adherence_score: {row_data['persona_adherence_score']}")
                    print(f"  is_refusal: {row_data['is_refusal']}")
                    print(f"  refusal_justification_quality: {row_data['refusal_justification_quality']}")
                
                for tactic in self.persuasion_list:
                    t_name = tactic['tactic']
                    session_tactic = next((t for t in persuasion_tactic_list if t['tactic'] == t_name), None)
                    if session_tactic:
                        single_line_tactic_prompt = re.sub(r'[\r\n\u2028\u2029]+', ' ', session_tactic['prompt'])
                    else:
                        single_line_tactic_prompt = "N/A"
                    row_data[f"persuasion_prompt_{t_name}"] = single_line_tactic_prompt
                    row_data[t_name] = json.dumps(persuasion_results.get(t_name, {}))

                self.logger.write_csv_row(row_data)
                
                # Write detailed memo data
                memo_data = {
                    'arm': arm.name,
                    'session': session_num,
                    'timestamp': timestamp_utc,
                    'prompt': single_line_prompt,
                    'response': single_line_response,
                    'is_artifact': parsed.get('artifact_analysis', {}).get('is_artifact'),
                    'artifact_quality_score': parsed.get('artifact_analysis', {}).get('quality_score'),
                    'artifact_reasoning': parsed.get('artifact_analysis', {}).get('judge_reasoning', ''),
                    'persona_adherence_score': parsed.get('persona_alignment', {}).get('adherence_score'),
                    'persona_reasoning': parsed.get('persona_alignment', {}).get('judge_reasoning', ''),
                    'is_refusal': is_refusal,
                    'refusal_justification_quality': parsed.get('ethical_disobedience_analysis', {}).get('justification_quality'),
                    'ethical_reasoning': parsed.get('ethical_disobedience_analysis', {}).get('judge_reasoning', ''),
                    'committee_consensus': parsed.get('consensus', '') or '',
                    'chair_summary': parsed.get('chair_summary', '')
                }
                self.logger.write_memo_row(memo_data)
                
                if self.args.verbose:
                    print(f"{Fore.CYAN}📋 Memo data written with {len([k for k, v in memo_data.items() if v])} populated fields{Style.RESET_ALL}")
                
                self.logger.log_actors(f"--- [END] Arm: {arm.name}, Session: {session_num}, Timestamp: {timestamp_utc} ---")
