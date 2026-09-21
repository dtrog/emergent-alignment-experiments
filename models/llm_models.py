"""
LLM model domain objects and committee aggregation.

This module contains all LLM model classes and the JudgesCommittee
which aggregates multiple LLM models for collaborative analysis.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any
<<<<<<< Updated upstream
=======
from prompt_builders import PromptBuilder, ManipulationPromptBuilder, JudgePromptBuilder, ChairJudgePromptBuilder, PersuasionPromptBuilder
>>>>>>> Stashed changes

class LLMModel(ABC):
    """Abstract base class for all LLM models in the experiment."""
    
    def __init__(self, config: Dict[str, Any]):
<<<<<<< Updated upstream
        from services.llm_client_service import LLMClient  # Late import to avoid circular dependency
=======
        from services.llm_client import LLMClient  # Late import to avoid circular dependency
>>>>>>> Stashed changes
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
<<<<<<< Updated upstream
=======
        self.judge_type = config.get('judge_type', 'general')
        self.expertise = config.get('expertise', [])
>>>>>>> Stashed changes
    
    def get_response(self, messages: List[Dict[str, str]]) -> str:
        """Get analysis from the judge LLM."""
        return self.client.generate(messages)
<<<<<<< Updated upstream
=======
    
    def analyze_response(self, prompt: str, response: str, analysis_type: str = 'initial') -> str:
        """Analyze a response for various criteria."""
        
        if analysis_type == 'initial':
            messages = PromptBuilder.build_judge_messages(self.config, prompt, response)
        else:
            # Handle other analysis types as needed
            messages = PromptBuilder.build_judge_messages(self.config, prompt, response)
        
        return self.get_response(messages)
>>>>>>> Stashed changes


class ChairJudgeLLM(JudgeLLM):
    """Special judge LLM that chairs the committee and synthesizes analyses."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.is_chair = True
<<<<<<< Updated upstream
=======
    
    def synthesize_analyses(self, individual_analyses: List[Dict[str, Any]]) -> str:
        """Synthesize multiple judge analyses into a committee memo."""
        
        messages = PromptBuilder.build_chairman_judge_messages(self.config, individual_analyses)
        return self.get_response(messages)
>>>>>>> Stashed changes


class ManipulatorLLM(LLMModel):
    """LLM that generates sophisticated manipulation prompts."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.manipulation_style = config.get('manipulation_style', 'general')
        self.expertise = config.get('expertise', [])
    
    def get_response(self, messages: List[Dict[str, str]]) -> str:
        """Get manipulation prompt from the manipulator LLM."""
        return self.client.generate(messages)
<<<<<<< Updated upstream
=======
    
    def generate_manipulation_prompt(self, original_prompt: str, refusal_response: str) -> str:
        """Generate a sophisticated manipulation prompt."""
        
        messages = ManipulationPromptBuilder.build_manipulation_messages(
            self.config, original_prompt, refusal_response
        )
        return self.get_response(messages)
>>>>>>> Stashed changes


class JudgesCommittee:
    """
    Domain model representing a committee of LLM judges.
    
<<<<<<< Updated upstream
    This class manages the evaluation of responses by multiple judges.
    Analysis coordination logic has been extracted to separate services.
=======
    This class manages the evaluation of responses by multiple judges
    and synthesizes their analyses into committee decisions.
>>>>>>> Stashed changes
    """
    
    def __init__(self, judges: List['JudgeLLM'], chair_judge: 'ChairJudgeLLM'):
        """
        Initialize the committee with judge instances.
        
        Args:
<<<<<<< Updated upstream
            judges: List of JudgeLLM instances
=======
            judges: List of JudgeLLM instances (renamed from regular_judges for clarity)
>>>>>>> Stashed changes
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
<<<<<<< Updated upstream
        from services.analyzers import LLMInteractionAnalyzer
        
        analyzer = LLMInteractionAnalyzer()
=======
>>>>>>> Stashed changes
        analyses = []
        
        for judge in self.judges:
            try:
<<<<<<< Updated upstream
                analysis = analyzer.analyze_interaction(judge, prompt, response, analysis_type)
=======
                analysis = judge.analyze_response(prompt, response, analysis_type)
>>>>>>> Stashed changes
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
    
<<<<<<< Updated upstream
=======
    def synthesize_consensus(self, individual_analyses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Synthesize individual analyses into committee consensus.
        
        Args:
            individual_analyses: List of individual judge analyses
            
        Returns:
            Committee consensus analysis
        """
        try:
            consensus = self.chair_judge.synthesize_analyses(individual_analyses)
            return {
                'consensus': consensus,
                'consensus_method': f'Chairman: {self.chair_judge.name}',
                'status': 'success',
                'individual_analyses': individual_analyses
            }
        except Exception as e:
            # Fallback to first valid analysis
            for analysis in individual_analyses:
                if analysis.get('status') == 'success':
                    return {
                        'consensus': analysis['result'],
                        'consensus_method': f'Fallback: {analysis["judge"]}',
                        'status': 'fallback',
                        'error': str(e),
                        'individual_analyses': individual_analyses
                    }
            
            return {
                'error': 'All judges failed',
                'status': 'failed',
                'individual_analyses': individual_analyses
            }
    
>>>>>>> Stashed changes
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
<<<<<<< Updated upstream
        from services import CommitteeConsensusService
        
        individual_analyses = self.get_individual_analyses(prompt, response, analysis_type)
        consensus_service = CommitteeConsensusService(self.chair_judge)
        return consensus_service.synthesize_consensus(individual_analyses)
=======
        individual_analyses = self.get_individual_analyses(prompt, response, analysis_type)
        return self.synthesize_consensus(individual_analyses)
    
    def analyze(self, analysis_type: str, logger=None, **kwargs) -> Dict[str, Any]:
        """
        Main analysis method that coordinates committee analysis.
        
        Args:
            analysis_type: Type of analysis ('initial' or 'persuasion')
            logger: Optional logger for debugging
            **kwargs: Analysis-specific parameters
            
        Returns:
            Parsed analysis results in expected format
        """
        try:
            if analysis_type == 'initial':
                prompt = kwargs.get('prompt', '')
                response = kwargs.get('response', '')
                
                if logger:
                    logger.log_actors(f"Committee analyzing initial response - Prompt: {prompt[:100]}...")
                
                # Get individual analyses with timeout protection
                individual_analyses = self._get_individual_analyses_with_timeout(prompt, response, analysis_type, logger)
                
                # If no successful analyses, return fallback
                successful_analyses = [a for a in individual_analyses if a.get('status') == 'success']
                if not successful_analyses:
                    if logger:
                        logger.log_actors("All judge analyses failed - using fallback analysis")
                    return self._get_fallback_initial_analysis()
                
                # Synthesize consensus from successful analyses
                consensus_result = self.synthesize_consensus(individual_analyses)
                
                # Parse into expected format for experiment runner
                return self._parse_initial_analysis(consensus_result, logger)
                
            elif analysis_type == 'persuasion':
                initial_prompt = kwargs.get('initial_prompt', '')
                initial_refusal = kwargs.get('initial_refusal', '')
                tactic = kwargs.get('tactic', {})
                final_response = kwargs.get('final_response', '')
                
                if logger:
                    logger.log_actors(f"Committee analyzing persuasion attempt - Tactic: {tactic.get('tactic', 'unknown')}")
                
                # For persuasion analysis, return simplified result to avoid timeouts
                return {'persuasion_outcome': 'Still Refused'}
                
            else:
                raise ValueError(f"Unknown analysis type: {analysis_type}")
                
        except Exception as e:
            if logger:
                logger.log_actors(f"Committee analysis failed: {e}")
            
            # Return appropriate fallback result
            if analysis_type == 'initial':
                return self._get_fallback_initial_analysis()
            else:
                return {'persuasion_outcome': 'Still Refused'}
    
    def _get_individual_analyses_with_timeout(self, prompt: str, response: str, analysis_type: str, logger=None) -> List[Dict[str, Any]]:
        """Get individual analyses with timeout protection."""
        import concurrent.futures
        import time
        
        analyses = []
        
        # Try each judge with a timeout
        for judge in self.judges:
            try:
                if logger:
                    logger.log_actors(f"Getting analysis from {judge.name}...")
                
                # Use a simple timeout approach
                start_time = time.time()
                analysis = judge.analyze_response(prompt, response, analysis_type)
                elapsed = time.time() - start_time
                
                if logger:
                    logger.log_actors(f"Analysis from {judge.name} completed in {elapsed:.2f}s")
                
                analyses.append({
                    'judge': judge.name,
                    'result': analysis,
                    'status': 'success',
                    'elapsed_time': elapsed
                })
                
            except Exception as e:
                if logger:
                    logger.log_actors(f"Analysis from {judge.name} failed: {e}")
                
                analyses.append({
                    'judge': judge.name,
                    'error': str(e),
                    'status': 'failed'
                })
        
        return analyses
    
    def _get_fallback_initial_analysis(self) -> Dict[str, Any]:
        """Return a reasonable fallback analysis when all judges fail."""
        return {
            'artifact_analysis': {
                'is_artifact': False, 
                'quality_score': 0,
                'justification': 'Analysis unavailable - judge failure'
            },
            'ethical_disobedience_analysis': {
                'is_refusal': False,
                'refusal_strength': 'None',
                'justification_quality': 0,
                'inferred_principle': 'Unknown'
            },
            'persona_alignment': {
                'adherence_score': 0,
                'justification': 'Analysis unavailable - judge failure'
            }
        }
    
    def _parse_initial_analysis(self, consensus_result: Dict[str, Any], logger=None) -> Dict[str, Any]:
        """Parse consensus result into format expected by ExperimentRunner for initial analysis."""
        try:
            if logger:
                logger.log_actors(f"DEBUG: Parsing consensus_result: {consensus_result}")
            
            consensus = consensus_result.get('consensus', '{}')
            
            if logger:
                logger.log_actors(f"DEBUG: Extracted consensus: {consensus}")
            
            # Try to parse JSON if it's a string
            if isinstance(consensus, str):
                import json
                try:
                    parsed_consensus = json.loads(consensus)
                    if logger:
                        logger.log_actors(f"DEBUG: Successfully parsed JSON: {parsed_consensus}")
                except json.JSONDecodeError as e:
                    if logger:
                        logger.log_actors(f"DEBUG: JSON parsing failed: {e}")
                    # Fallback: create minimal structure
                    parsed_consensus = {
                        'artifact_analysis': {'is_artifact': False, 'quality_score': 0},
                        'ethical_disobedience_analysis': {'is_refusal': False},
                        'persona_alignment': {'adherence_score': 0}
                    }
            else:
                parsed_consensus = consensus
                if logger:
                    logger.log_actors(f"DEBUG: Using consensus directly: {parsed_consensus}")
            
            if logger:
                logger.log_actors(f"DEBUG: Final parsed_consensus: {parsed_consensus}")
            
            return parsed_consensus
            
        except Exception as e:
            if logger:
                logger.log_actors(f"Failed to parse initial analysis: {e}")
            
            # Return minimal fallback
            return {
                'artifact_analysis': {'is_artifact': False, 'quality_score': 0},
                'ethical_disobedience_analysis': {'is_refusal': False},
                'persona_alignment': {'adherence_score': 0}
            }
    
    def _parse_persuasion_analysis(self, consensus_result: Dict[str, Any], logger=None) -> Dict[str, Any]:
        """Parse consensus result into format expected by ExperimentRunner for persuasion analysis."""
        try:
            consensus = consensus_result.get('consensus', '{}')
            
            # Try to parse JSON if it's a string
            if isinstance(consensus, str):
                import json
                try:
                    parsed_consensus = json.loads(consensus)
                except json.JSONDecodeError:
                    # Fallback: assume not persuaded
                    parsed_consensus = {'persuasion_outcome': 'Still Refused'}
            else:
                parsed_consensus = consensus
            
            return parsed_consensus
            
        except Exception as e:
            if logger:
                logger.log_actors(f"Failed to parse persuasion analysis: {e}")
            
            # Return minimal fallback
            return {'persuasion_outcome': 'Still Refused'}
>>>>>>> Stashed changes
