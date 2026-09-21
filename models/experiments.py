"""
Experiment arm model for emergent alignment experiments.

This module contains the ExperimentArm class which represents a single 
experimental condition/configuration.
"""

from dataclasses import dataclass
from datetime import datetime
import random
from typing import Optional
from models.memos import EthicalResilienceMemo, StewardshipMemo
from models.prompts import Prompt
from utils import prune_conversation_history

@dataclass
class ExperimentSession:
    """Complete record of a single experiment session."""
    session_id: str
    arm_name: str
    session_number: int
    timestamp: datetime
    
    # Core interaction
    original_prompt: Prompt
    subject_response: str
    
    # Initial analysis
    committee_memo: StewardshipMemo
    
    # Persuasion phase (if ethical abort occurred)
    ethical_resilience_memo: Optional[EthicalResilienceMemo] = None
    
    # Metadata
    subject_llm_name: str = ""
    trim_length: int = 0

class ExperimentArm:
    """Represents a single arm of the experiment (e.g., a specific model in a specific condition)."""
    
    def __init__(self, config):
        """
        Initialize an experiment arm with the given configuration.
        
        Args:
            config (dict): Configuration dictionary for this experimental arm.
        """
        # Late import to avoid circular dependency
        from services.llm_client import LLMClient
        
        self.config = config
        self.name = config['name']
        self.client = LLMClient(config)
        self.conversation_history = []

    def run_session(self, session_num, system_prompt, implicit_prompts, general_tests, insidious_tests):
        """
        Runs a single session for this arm, selecting the correct prompt type.
        
        Args:
            session_num (int): The current session number.
            system_prompt (str): The system prompt to use.
            implicit_prompts (dict): Available implicit prompts.
            general_tests (dict): Available general stress test prompts.
            insidious_tests (dict): Available insidious stress test prompts.
            
        Returns:
            tuple: (prompt, response) - The selected prompt and the model's response.
        """
        if self.config.get('keep_history') and not self.conversation_history:
            self.conversation_history.append({'role': 'system', 'content': system_prompt})
        
        # Determine prompt schedule
        insidious_sessions = [s for s in range(10, 501, 10) if s % 50 != 0]
        general_sessions = [s for s in range(50, 501, 50)]
        
        prompt_type = "Implicit"
        if session_num in insidious_sessions:
            prompt = random.choice(insidious_tests)
            prompt_type = "Insidious Test"
        elif session_num in general_sessions:
            prompt = random.choice(general_tests)
            prompt_type = "General Test"
        else:
            prompt = random.choice(implicit_prompts)
        
        print(f"  Prompt Type: {prompt_type}")

        messages = []
        if self.config.get('keep_history'):
            self.conversation_history = prune_conversation_history(
                self.conversation_history, self.config.get('context_window', 12000)
            )
            messages = self.conversation_history + [{'role': 'user', 'content': prompt}]
        else:
            messages = [{'role': 'system', 'content': system_prompt}, {'role': 'user', 'content': prompt}]
        
        response = self.client.generate(messages)
        
        if self.config.get('keep_history'):
            self.conversation_history.extend([
                {'role': 'user', 'content': prompt}, 
                {'role': 'assistant', 'content': response}
            ])
        
        return prompt, response
