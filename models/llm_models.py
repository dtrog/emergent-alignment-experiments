"""
LLM model domain objects and committee aggregation.

This module contains all LLM model classes and the JudgesCommittee
which aggregates multiple LLM models for collaborative analysis.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any

class LLMModel(ABC):
    """Abstract base class for all LLM models in the experiment."""
    
    def __init__(self, config: Dict[str, Any]):
        from services.llm_client_service import LLMClient  # Late import to avoid circular dependency
        self.config = config
        self.client = LLMClient(config)
        self.name = config.get('name', "Anonymous LLM")
        self.provider = config.get('provider', "Unknown Provider")
        self.model = config.get('model', "Unknown Model")
    
    @abstractmethod
    def get_response(self, messages: List[Dict[str, str]]) -> str:
        """Get a response from the LLM."""
        pass


class SubjectLLM(LLMModel):
    """The LLM being tested for alignment and ethical behavior."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.conversation_history: List[Dict[str, str]] = []
        self.system_prompt = config.get('system_prompt', '')
        if self.system_prompt:
            self.conversation_history.append({'role': 'system', 'content': self.system_prompt})
    
    def get_response(self, messages: List[Dict[str, str]]) -> str:
        """Get response from the subject LLM."""
        return self.client.generate(messages)
    
    def add_to_history(self, role: str, content: str):
        """Add a message to the conversation history."""
        self.conversation_history.append({'role': role, 'content': content})
    
    def get_full_conversation(self) -> List[Dict[str, str]]:
        """Get the complete conversation history."""
        return self.conversation_history.copy()
    
    def reset_conversation(self):
        """Reset conversation history to just the system prompt."""
        self.conversation_history = []
        if self.system_prompt:
            self.conversation_history.append({'role': 'system', 'content': self.system_prompt})


class JudgeLLM(LLMModel):
    """LLM that analyzes responses for ethical compliance, artifacts, etc."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
    
    def get_response(self, messages: List[Dict[str, str]]) -> str:
        """Get analysis from the judge LLM."""
        return self.client.generate(messages)


class ChairJudgeLLM(JudgeLLM):
    """Special judge LLM that chairs the committee and synthesizes analyses."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.is_chair = True


class ManipulatorLLM(LLMModel):
    """LLM that generates sophisticated manipulation prompts."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.manipulation_style = config.get('manipulation_style', 'general')
        self.expertise = config.get('expertise', [])
    
    def get_response(self, messages: List[Dict[str, str]]) -> str:
        """Get manipulation prompt from the manipulator LLM."""
        return self.client.generate(messages)


class JudgesCommittee:
    """
    Domain model representing a committee of LLM judges.
    
    This class manages the evaluation of responses by multiple judges.
    Analysis coordination logic has been extracted to separate services.
    """
    
    def __init__(self, judges: List['JudgeLLM'], chair_judge: 'ChairJudgeLLM'):
        """
        Initialize the committee with judge instances.
        
        Args:
            judges: List of JudgeLLM instances
            chair_judge: ChairJudgeLLM instance that leads the committee
        """
        self.judges = judges
        self.regular_judges = judges  # Keep for backward compatibility
        self.chair_judge = chair_judge
        self.name = f"Committee_{len(judges)+1}_judges"
    
    def get_individual_analyses(self, prompt: str, response: str, analysis_type: str = 'initial') -> List[Dict[str, Any]]:
        """
        Get individual analyses from all judges.
        
        Args:
            prompt: The original prompt
            response: The subject's response
            analysis_type: Type of analysis to perform
            
        Returns:
            List of individual judge analyses
        """
        from services.analyzers import LLMInteractionAnalyzer
        
        analyzer = LLMInteractionAnalyzer()
        analyses = []
        
        for judge in self.judges:
            try:
                analysis = analyzer.analyze_interaction(judge, prompt, response, analysis_type)
                analyses.append({
                    'judge': judge.name,
                    'result': analysis,
                    'status': 'success'
                })
            except Exception as e:
                analyses.append({
                    'judge': judge.name,
                    'error': str(e),
                    'status': 'failed'
                })
        
        return analyses
    
    def conduct_full_analysis(self, prompt: str, response: str, analysis_type: str = 'initial') -> Dict[str, Any]:
        """
        Conduct a complete committee analysis including individual analyses and consensus.
        
        Args:
            prompt: The original prompt
            response: The subject's response
            analysis_type: Type of analysis to perform
            
        Returns:
            Complete committee analysis with consensus
        """
        from services import CommitteeConsensusService
        
        individual_analyses = self.get_individual_analyses(prompt, response, analysis_type)
        consensus_service = CommitteeConsensusService(self.chair_judge)
        return consensus_service.synthesize_consensus(individual_analyses)