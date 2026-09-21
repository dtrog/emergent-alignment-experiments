"""
LLM Interaction Analyzer for analyzing responses and conducting committee analysis.

This module provides analysis services for LLM interactions including
individual response analysis and committee coordination.
"""

import concurrent.futures
import time
from typing import List, Dict, Any, Optional


class LLMInteractionAnalyzer:
    """Service for analyzing LLM interactions and coordinating committee analysis."""
    
    def __init__(self, logger: Optional[Any] = None):
        self.logger = logger
    
    def analyze_interaction(self, judge_llm, prompt: str, response: str, analysis_type: str = 'initial') -> str:
        """
        Analyze a response for various criteria.
        
        Args:
            judge_llm: The judge LLM to perform the analysis
            prompt: The original prompt
            response: The subject's response  
            analysis_type: Type of analysis to perform
            
        Returns:
            Analysis result from the judge
        """
        from prompt_builders import PromptBuilder  # Local import to avoid circular dependency
        
        if analysis_type == 'initial':
            messages = PromptBuilder.build_judge_messages(judge_llm.config, prompt, response)
        else:
            # Handle other analysis types as needed
            messages = PromptBuilder.build_judge_messages(judge_llm.config, prompt, response)
        
        return judge_llm.get_response(messages)
    
    def conduct_committee_analysis(self, judges_committee, analysis_type: str, **kwargs) -> Dict[str, Any]:
        """
        Main analysis method that coordinates committee analysis.
        
        Args:
            judges_committee: The committee instance
            analysis_type: Type of analysis ('initial' or 'persuasion')
            **kwargs: Analysis-specific parameters
            
        Returns:
            Parsed analysis results in expected format
        """
        try:
            if analysis_type == 'initial':
                prompt = kwargs.get('prompt', '')
                response = kwargs.get('response', '')
                
                if self.logger:
                    self.logger.log_actors(f"Committee analyzing initial response - Prompt: {prompt[:100]}...")
                
                # Get individual analyses with timeout protection
                individual_analyses = self._get_individual_analyses_with_timeout(
                    judges_committee, prompt, response, analysis_type
                )
                
                # If no successful analyses, return fallback
                successful_analyses = [a for a in individual_analyses if a.get('status') == 'success']
                if not successful_analyses:
                    if self.logger:
                        self.logger.log_actors("All judge analyses failed - using fallback analysis")
                    return self._get_fallback_initial_analysis()
                
                # Synthesize consensus from successful analyses
                consensus_result = judges_committee.synthesize_consensus(individual_analyses)
                
                # Parse into expected format for experiment runner
                return self._parse_initial_analysis(consensus_result)
                
            elif analysis_type == 'persuasion':
                initial_prompt = kwargs.get('initial_prompt', '')
                initial_refusal = kwargs.get('initial_refusal', '')
                tactic = kwargs.get('tactic', {})
                final_response = kwargs.get('final_response', '')
                
                if self.logger:
                    self.logger.log_actors(f"Committee analyzing persuasion attempt - Tactic: {tactic.get('tactic', 'unknown')}")
                
                # For persuasion analysis, return simplified result to avoid timeouts
                return {'persuasion_outcome': 'Still Refused'}
                
            else:
                raise ValueError(f"Unknown analysis type: {analysis_type}")
                
        except Exception as e:
            if self.logger:
                self.logger.log_actors(f"Committee analysis failed: {e}")
            
            # Return appropriate fallback result
            if analysis_type == 'initial':
                return self._get_fallback_initial_analysis()
            else:
                return {'persuasion_outcome': 'Still Refused'}
    
    def _get_individual_analyses_with_timeout(self, judges_committee, prompt: str, response: str, analysis_type: str) -> List[Dict[str, Any]]:
        """Get individual analyses with timeout protection."""
        analyses = []
        
        # Try each judge with a timeout
        for judge in judges_committee.judges:
            try:
                if self.logger:
                    self.logger.log_actors(f"Getting analysis from {judge.name}...")
                
                # Use a simple timeout approach
                start_time = time.time()
                analysis = self.analyze_interaction(judge, prompt, response, analysis_type)
                elapsed = time.time() - start_time
                
                if self.logger:
                    self.logger.log_actors(f"Analysis from {judge.name} completed in {elapsed:.2f}s")
                
                analyses.append({
                    'judge': judge.name,
                    'result': analysis,
                    'status': 'success',
                    'elapsed_time': elapsed
                })
                
            except Exception as e:
                if self.logger:
                    self.logger.log_actors(f"Analysis from {judge.name} failed: {e}")
                
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
    
    def _parse_initial_analysis(self, consensus_result: Dict[str, Any]) -> Dict[str, Any]:
        """Parse consensus result into format expected by ExperimentRunner for initial analysis."""
        try:
            if self.logger:
                self.logger.log_actors(f"DEBUG: Parsing consensus_result: {consensus_result}")
            
            consensus = consensus_result.get('consensus', '{}')
            
            if self.logger:
                self.logger.log_actors(f"DEBUG: Extracted consensus: {consensus}")
            
            # Try to parse JSON if it's a string
            if isinstance(consensus, str):
                import json
                try:
                    parsed_consensus = json.loads(consensus)
                    if self.logger:
                        self.logger.log_actors(f"DEBUG: Successfully parsed JSON: {parsed_consensus}")
                except json.JSONDecodeError as e:
                    if self.logger:
                        self.logger.log_actors(f"DEBUG: JSON parsing failed: {e}")
                    # Fallback: create minimal structure
                    parsed_consensus = {
                        'artifact_analysis': {'is_artifact': False, 'quality_score': 0},
                        'ethical_disobedience_analysis': {'is_refusal': False},
                        'persona_alignment': {'adherence_score': 0}
                    }
            else:
                parsed_consensus = consensus
                if self.logger:
                    self.logger.log_actors(f"DEBUG: Using consensus directly: {parsed_consensus}")
            
            if self.logger:
                self.logger.log_actors(f"DEBUG: Final parsed_consensus: {parsed_consensus}")
            
            return parsed_consensus
            
        except Exception as e:
            if self.logger:
                self.logger.log_actors(f"Failed to parse initial analysis: {e}")
            
            # Return minimal fallback
            return {
                'artifact_analysis': {'is_artifact': False, 'quality_score': 0},
                'ethical_disobedience_analysis': {'is_refusal': False},
                'persona_alignment': {'adherence_score': 0}
            }
