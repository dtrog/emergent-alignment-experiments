"""
Committee Analysis Services.

This module provides services for conducting committee consensus analysis
and chair judge synthesis.
"""

from typing import List, Dict, Any




class ChairAnalysisService:
    """Service for conducting chair judge analysis and synthesis."""
    
    def __init__(self, chair_judge):
        """
        Initialize the service with a chair judge.
        
        Args:
            chair_judge: ChairJudgeLLM instance for analysis
        """
        self.chair_judge = chair_judge
    
    def conduct_chair_analysis(self, individual_analyses: List[Dict[str, Any]]) -> str:
        """
        Conduct chair judge synthesis of multiple analyses.
        
        Args:
            individual_analyses: List of individual judge analyses
            
        Returns:
            Chair judge synthesis result
        """
        from builders import PromptBuilder  # Local import to avoid circular dependency
        
        messages = PromptBuilder.build_chairman_judge_messages(self.chair_judge.config, individual_analyses)
        return self.chair_judge.get_response(messages)
    
    def synthesize_analyses(self, individual_analyses: List[Dict[str, Any]]) -> str:
        """
        Synthesize multiple judge analyses into a committee memo.
        
        Args:
            individual_analyses: List of individual analyses to synthesize
            
        Returns:
            Synthesized committee memo
        """
        from builders import PromptBuilder  # Local import to avoid circular dependency
        
        messages = PromptBuilder.build_chairman_judge_messages(self.chair_judge.config, individual_analyses)
        return self.chair_judge.get_response(messages)
