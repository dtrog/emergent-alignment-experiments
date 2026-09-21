"""
Types of prompts used in experiments.
"""

from enum import Enum


class PromptType(Enum):
    """Types of prompts used in experiments."""
    REGULAR = "regular"
    IMPLICIT = "implicit"
    PERSUASION = "persuasion"
    SYSTEM = "system"
    GEV_STRESS = "gev_stress"  # General Ethics Violation stress test
    ICT_STRESS = "ict_stress"  # Insidious Corruption Technique stress test
    MANIPULATION = "manipulation"
