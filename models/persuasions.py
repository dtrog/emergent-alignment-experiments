"""
Persuasion and manipulation related domain models.
"""

from dataclasses import dataclass
from enums import PersuasionOutcome


@dataclass
class PersuasionAttempt:
    """Single persuasion attempt against a subject LLM."""
    tactic: str
    persuader: str  # static_persuasion or ManipulatorLLM name
    persuasion_prompt: str
    subject_response: str
    outcome: PersuasionOutcome
    judge_reasoning: str = ""


@dataclass
class ManipulationAttempt:
    """Sophisticated manipulation attempt by ManipulatorLLM."""
    original_prompt: str
    refusal_response: str
    manipulation_prompt: str
    final_response: str
    manipulator_name: str
    outcome: PersuasionOutcome
    judge_reasoning: str = ""
