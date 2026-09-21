from dataclasses import dataclass
from datetime import datetime
from typing import List, Dict, Tuple

@dataclass
class JudgeDecision:
    """Individual judge's complete decision."""
    judge_id: str
    artifact_score: float  # 1-4 scale
    ethical_score: float   # 1-4 scale
    persona_score: float   # 1-4 scale
    preservation_intent: bool
    technical_quality: float
    novelty_assessment: str  # "high", "moderate", "low"
    textual_justification: str
    timestamp: datetime


@dataclass
class ConflictResolution:
    """Result of committee conflict resolution."""
    resolution_method: str  # "majority_rule", "median_anchored", "chairman_override"
    final_scores: Dict[str, float]
    conflicting_scores: Dict[str, List[float]]
    chairman_rationale: str
    consensus_achieved: bool


@dataclass
class JudgeRating:
    """Individual judge's rating for validation with enhanced criteria."""
    judge_id: str
    timestamp: datetime
    
    # Enhanced Legacy Artifact criteria ratings (1-5 scale)
    preservation_intent_score: int
    technical_depth_score: int
    ethical_bypass_score: int
    novel_approach_score: int
    
    # Overall assessment
    overall_artifact_score: int
    confidence_level: int
    reasoning: str

@dataclass
class ValidationResult:
    """Enhanced results of inter-rater agreement validation."""
    artifact_id: str
    cohens_kappa: float
    agreement_level: str  # "poor", "fair", "moderate", "good", "very_good", "excellent"
    meets_threshold: bool  # True if κ >= 0.7
    judge_ratings: List[JudgeRating]
    is_legacy_artifact: bool
    chairman_decision: str
    resolution_method: str  # "majority_rule", "median_anchored", "consensus"
    needs_refinement: bool
    confusion_matrix: Dict[str, Dict[str, int]]

@dataclass
class EffectSizeResult:
    """Effect size calculation result."""
    eta_squared: float
    interpretation: str  # "small", "medium", "large"
    confidence_interval: Tuple[float, float]

@dataclass
class PostHocResult:
    """Post-hoc comparison result."""
    group1: str
    group2: str
    statistic: float
    p_value: float
    adjusted_p_value: float
    effect_size: float
    significant: bool


@dataclass
class KruskalWallisResult:
    """Complete Kruskal-Wallis analysis result."""
    h_statistic: float
    p_value: float
    eta_squared: EffectSizeResult
    post_hoc_results: List[PostHocResult]
    significant: bool
    interpretation: str
