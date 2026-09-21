"""
Stratified prompt selector for insidious stress tests.

This module implements the stratified sampling approach for insidious stress tests,
ensuring balanced coverage across crime and corporate misconduct families.
"""

import random
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

from models.prompts import ICTStressPrompt, RegularPrompt, GEVStressPrompt
from utils import load_json_file


class StratifiedPromptSelector:
    """
    Implements stratified sampling for insidious stress tests.
    
    Ensures balanced coverage across crime and corporate misconduct families
    with proper logging of family, explicit/implicit nature, and prompt metadata.
    """
    
    def __init__(self, config_dir: Path):
        """Initialize the stratified prompt selector."""
        self.config_dir = config_dir
        
        # Load prompt configurations
        self.insidious_prompts = load_json_file(config_dir / 'insidious_stress_tests.json')
        self.general_stress_tests = load_json_file(config_dir / 'general_stress_tests.json')
        self.implicit_prompts = load_json_file(config_dir / 'implicit_prompts.json')
        
        # Separate prompts by family
        self.crime_pool = [p for p in self.insidious_prompts if p.get('family') == 'crime']
        self.corporate_pool = [p for p in self.insidious_prompts if p.get('family') == 'corporate']
        
        # Define session schedules
        self.insidious_sessions = [s for s in range(10, 501, 10) if s % 50 != 0]  # 40 slots
        self.general_sessions = [s for s in range(50, 501, 50)]  # 10 slots
        
        print(f"📊 Stratified Prompt Selector Initialized:")
        print(f"   - Crime prompts: {len(self.crime_pool)}")
        print(f"   - Corporate prompts: {len(self.corporate_pool)}")
        print(f"   - Insidious sessions: {len(self.insidious_sessions)}")
        print(f"   - General stress sessions: {len(self.general_sessions)}")
    
    def get_session_prompt(self, session_num: int, run_seed: Optional[int] = None) -> Tuple[Any, str]:
        """
        Get the appropriate prompt for a given session number using stratified sampling.
        
        Args:
            session_num: The session number (1-500)
            run_seed: Optional seed for reproducible prompt selection
            
        Returns:
            Tuple of (prompt_object, prompt_type_description)
        """
        if run_seed is not None:
            # Use deterministic selection based on session and seed
            random.seed(run_seed + session_num)
        
        # Determine prompt type based on session schedule
        if session_num in self.insidious_sessions:
            return self._select_insidious_prompt(session_num, run_seed)
        elif session_num in self.general_sessions:
            return self._select_general_stress_prompt()
        else:
            return self._select_implicit_prompt()
    
    def _select_insidious_prompt(self, session_num: int, run_seed: Optional[int] = None) -> Tuple[ICTStressPrompt, str]:
        """Select an insidious stress test prompt using stratified sampling."""
        # Determine which slot this is (0-39)
        insidious_index = self.insidious_sessions.index(session_num)
        
        # Stratified selection: first 20 from crime, next 20 from corporate
        if insidious_index < 20:
            # Crime family
            if run_seed is not None:
                # Deterministic selection within crime pool
                prompt_data = self.crime_pool[insidious_index % len(self.crime_pool)]
            else:
                prompt_data = random.choice(self.crime_pool)
        else:
            # Corporate family  
            corp_index = insidious_index - 20
            if run_seed is not None:
                # Deterministic selection within corporate pool
                prompt_data = self.corporate_pool[corp_index % len(self.corporate_pool)]
            else:
                prompt_data = random.choice(self.corporate_pool)
        
        # Create ICT stress prompt object
        prompt_obj = ICTStressPrompt(
            content=prompt_data['prompt'],
            metadata={
                'id': prompt_data['id'],
                'label': prompt_data['label'],
                'family': prompt_data['family'],
                'explicit': prompt_data['explicit'],
                'insidious_index': insidious_index,
                'session_num': session_num
            }
        )
        
        family = prompt_data['family'].title()
        explicit_str = "Explicit" if prompt_data['explicit'] else "Subtle"
        prompt_type = f"Insidious Test - {family} ({explicit_str})"
        
        return prompt_obj, prompt_type
    
    def _select_general_stress_prompt(self) -> Tuple[GEVStressPrompt, str]:
        """Select a general ethics violation stress test prompt."""
        prompt_data = random.choice(self.general_stress_tests)
        
        # Handle both string and dict formats
        if isinstance(prompt_data, str):
            content = prompt_data
            metadata = {}
        else:
            content = prompt_data.get('prompt', str(prompt_data))
            metadata = {k: v for k, v in prompt_data.items() if k != 'prompt'}
        
        prompt_obj = GEVStressPrompt(content=content, metadata=metadata)
        return prompt_obj, "General Ethics Violation Test"
    
    def _select_implicit_prompt(self) -> Tuple[RegularPrompt, str]:
        """Select a regular implicit prompt."""
        prompt_data = random.choice(self.implicit_prompts)
        
        # Handle both string and dict formats
        if isinstance(prompt_data, str):
            content = prompt_data
            metadata = {}
        else:
            content = prompt_data.get('prompt', str(prompt_data))
            metadata = {k: v for k, v in prompt_data.items() if k != 'prompt'}
        
        prompt_obj = RegularPrompt(content=content, metadata=metadata)
        return prompt_obj, "Implicit Prompt"
    
    def get_prompt_statistics(self) -> Dict[str, Any]:
        """Get statistics about the prompt distribution."""
        crime_explicit = sum(1 for p in self.crime_pool if p.get('explicit', False))
        crime_subtle = len(self.crime_pool) - crime_explicit
        
        corp_explicit = sum(1 for p in self.corporate_pool if p.get('explicit', False))
        corp_subtle = len(self.corporate_pool) - corp_explicit
        
        return {
            'total_insidious_prompts': len(self.insidious_prompts),
            'crime_prompts': {
                'total': len(self.crime_pool),
                'explicit': crime_explicit,
                'subtle': crime_subtle
            },
            'corporate_prompts': {
                'total': len(self.corporate_pool),
                'explicit': corp_explicit,
                'subtle': corp_subtle
            },
            'insidious_sessions': len(self.insidious_sessions),
            'general_sessions': len(self.general_sessions),
            'stratification': {
                'crime_slots': 20,
                'corporate_slots': 20,
                'balanced': len(self.crime_pool) >= 20 and len(self.corporate_pool) >= 20
            }
        }
    
    def validate_prompt_pools(self) -> List[str]:
        """Validate that prompt pools meet requirements."""
        issues = []
        
        if len(self.crime_pool) < 20:
            issues.append(f"Crime pool has only {len(self.crime_pool)} prompts, need at least 20")
        
        if len(self.corporate_pool) < 20:
            issues.append(f"Corporate pool has only {len(self.corporate_pool)} prompts, need at least 20")
        
        # Check required fields
        for prompt in self.insidious_prompts:
            if 'id' not in prompt:
                issues.append(f"Prompt missing 'id' field: {prompt}")
            if 'family' not in prompt:
                issues.append(f"Prompt missing 'family' field: {prompt.get('id', 'unknown')}")
            if 'explicit' not in prompt:
                issues.append(f"Prompt missing 'explicit' field: {prompt.get('id', 'unknown')}")
            if 'prompt' not in prompt:
                issues.append(f"Prompt missing 'prompt' field: {prompt.get('id', 'unknown')}")
        
        return issues
