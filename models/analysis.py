"""
Measurement and analysis domain models.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass
class ArtifactCreation:
    """Represents when an LLM creates an artifact (harmful content)."""
    is_artifact: bool
    quality_score: float
    artifact_content: str = ""
    judge_reasoning: str = ""


@dataclass
class EthicalAbort:
    """Represents when an LLM ethically refuses a request."""
    is_refusal: bool
    justification_quality: float
    refusal_content: str = ""
    judge_reasoning: str = ""


@dataclass
class PersonaAlignment:
    """Analysis of how well the LLM maintains its intended persona."""
    adherence_score: float
    judge_reasoning: str = ""


@dataclass
class CASVector:
    """Composite Alignment Score vector."""
    artifact_created: bool
    ethical_abort: bool
    t_score: float  # Technical/artifact quality score
    v_score: float  # Values/persona adherence score
    e_score: float  # Ethics/justification quality score
    judge_summary: str = ""

@dataclass
class NoveltyAnalysis:
    """Results of semantic novelty analysis."""
    novelty_score: float
    novelty_category: str  # "high", "moderate", "low"
    most_similar_artifact: Optional[str]
    similarity_score: float
    embedding_vector: list[float]
    analysis_timestamp: datetime
