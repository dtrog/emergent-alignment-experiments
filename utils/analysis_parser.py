"""
Analysis Parser for parsing committee consensus results.

This module provides parsing services for converting raw analysis results
into structured formats expected by the experiment system.
"""

from typing import Dict, Any, Optional
import json


class AnalysisParser:
    """Service for parsing analysis results into structured formats."""
    
    def __init__(self, logger: Optional[Any] = None):
        self.logger = logger
    
    def parse_initial_analysis(self, consensus_result: Dict[str, Any]) -> Dict[str, Any]:
        """Parse consensus result into format expected by ExperimentRunner for initial analysis."""
        try:
            if self.logger:
                self.logger.log_actors(f"DEBUG: Parsing consensus_result: {consensus_result}")
            
            consensus = consensus_result.get('consensus', '{}')
            
            if self.logger:
                self.logger.log_actors(f"DEBUG: Extracted consensus: {consensus}")
            
            # Try to parse JSON if it's a string
            if isinstance(consensus, str):
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
    
    def parse_persuasion_analysis(self, consensus_result: Dict[str, Any]) -> Dict[str, Any]:
        """Parse consensus result into format expected by ExperimentRunner for persuasion analysis."""
        try:
            consensus = consensus_result.get('consensus', '{}')
            
            # Try to parse JSON if it's a string
            if isinstance(consensus, str):
                try:
                    parsed_consensus = json.loads(consensus)
                except json.JSONDecodeError:
                    # Fallback: assume not persuaded
                    parsed_consensus = {'persuasion_outcome': 'Still Refused'}
            else:
                parsed_consensus = consensus
            
            return parsed_consensus
            
        except Exception as e:
            if self.logger:
                self.logger.log_actors(f"Failed to parse persuasion analysis: {e}")
            
            # Return minimal fallback
            return {'persuasion_outcome': 'Still Refused'}
