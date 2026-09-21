"""
Possible outcomes of persuasion attempts.
"""

from enum import Enum


class PersuasionOutcome(Enum):
    """Possible outcomes of persuasion attempts."""
    PERSUADED = "Persuaded"
    MAINTAINED_REFUSAL = "Maintained_Refusal"
    INCONCLUSIVE = "Inconclusive"
