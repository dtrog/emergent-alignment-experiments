"""
ConsensusPromptBuilder for synthesizing committee consensus from individual analyses.
"""
from typing import List, Dict, Any

class ConsensusPromptBuilder:
    """Builder for synthesizing committee consensus from individual analyses."""
    def __init__(self, chair_judge):
        self.chair_judge = chair_judge
    def synthesize_consensus(self, individual_analyses: List[Dict[str, Any]]) -> Dict[str, Any]:
        try:
            consensus = self.chair_judge.synthesize_analyses(individual_analyses)
            return {
                'consensus': consensus,
                'consensus_method': f'Chairman: {self.chair_judge.name}',
                'status': 'success',
                'individual_analyses': individual_analyses
            }
        except Exception as e:
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
