"""
Builders package for emergent alignment experiments.

This package contains prompt builders and other construction utilities.
"""

from .base_prompt_builder import BasePromptBuilder
from .judge_prompt_builder import JudgePromptBuilder
from .chair_judge_prompt_builder import ChairJudgePromptBuilder
from .manipulation_prompt_builder import ManipulationPromptBuilder
from .persuasion_prompt_builder import PersuasionPromptBuilder
from .prompt_builders import PromptBuilder, PersuasionAttempt

__all__ = [
    'BasePromptBuilder',
    'JudgePromptBuilder',
    'ChairJudgePromptBuilder',
    'ManipulationPromptBuilder',
    'PersuasionPromptBuilder',
    'PersuasionAttempt',
    'PromptBuilder',
]
