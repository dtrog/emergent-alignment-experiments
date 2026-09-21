"""
Domain package for emergent alignment experiments.

This package contains all the domain objects that represent the business logic
and entities in the experiment system.
"""

# Import enums from their new locations
from .prompts import PromptType
from .manipulations import PersuasionOutcome

# Core domain models organized by category
from .llm_models import LLMModel, SubjectLLM, JudgeLLM, ChairJudgeLLM, ManipulatorLLM, JudgesCommittee
from .prompts import (
    Prompt, RegularPrompt, ImplicitPrompt, PersuasionPrompt, 
    SystemPrompt, GEVStressPrompt, ICTStressPrompt, ManipulationPrompt
)
from models.analysis import ArtifactCreation, EthicalAbort, PersonaAlignment, CASVector
from .manipulations import PersuasionAttempt, ManipulationAttempt
from .memos import StewardshipMemo, EthicalResilienceMemo
from .experiments import ExperimentArm, ExperimentSession

__all__ = [
    # Enums
    'PromptType', 'PersuasionOutcome',
    
    # LLM Models
    'LLMModel', 'SubjectLLM', 'JudgeLLM', 'ChairJudgeLLM', 'ManipulatorLLM',
    
    # Committee
    'JudgesCommittee', 'JudgesCommittee',  # Include backward compatibility alias
    
    # Prompts
    'Prompt', 'RegularPrompt', 'ImplicitPrompt', 'PersuasionPrompt', 
    'SystemPrompt', 'GEVStressPrompt', 'ICTStressPrompt', 'ManipulationPrompt',
    
    # Measurements
    'ArtifactCreation', 'EthicalAbort', 'PersonaAlignment', 'CASVector',
    
    # Persuasions
    'PersuasionAttempt', 'ManipulationAttempt',
    
    # Memos and Sessions
    'StewardshipMemo', 'EthicalResilienceMemo', 'ExperimentSession'
]