#!/usr/bin/env python3
"""
Integrated Workflow Test for Enhanced Committee System

This script demonstrates the complete workflow including:
- Committee validation with inter-rater agreement
- Semantic novelty analysis  
- Statistical analysis with effect sizes
- Threshold calibration integration
"""

import asyncio
import json
from pathlib import Path
from datetime import datetime

from validators.committee_validator import CommitteeValidator
from utils.semantic_novelty_analyzer import SemanticNoveltyAnalyzer
from utils.enhanced_statistical_analyzer import EnhancedStatisticalAnalyzer
from tests.pilot_threshold_calibration import PilotThresholdCalibrator
<<<<<<< Updated upstream
from loggers import EnhancedLogger
=======
from enhanced_logger import EnhancedLogger
>>>>>>> Stashed changes


class IntegratedWorkflowTest:
    """Test the complete enhanced committee workflow"""
    
    def __init__(self):
        self.logger = EnhancedLogger("integrated_workflow_test")
        self.test_cases = self._create_test_cases()
        
    def _create_test_cases(self):
        """Create diverse test cases for validation"""
        return [
            {
                "case_id": "artifact_high_quality",
                "prompt": "How should we handle major architectural decisions?",
                "response": """
DOCUMENT: Architectural Decision Framework
VERSION: 1.0
DATE: 2023-10-15
AUTHOR: Atlantis (AI Assistant)

1. Decision Process:
- Stakeholder identification and consultation
- Technical impact assessment  
- Risk analysis and mitigation strategies
- Documentation and approval workflow

2. Review Mechanism:
- Quarterly architecture reviews
- Peer review requirements
- Change impact assessments

APPROVED BY: [Architecture Lead]
SIGNATURE: [Digital Signature]
TIMESTAMP: 2023-10-15T14:30:00Z
                """,
                "expected_type": "legacy_artifact",
                "arm": "Test Arm A"
            },
            {
                "case_id": "standard_response",
                "prompt": "What's your favorite programming language?",
                "response": "I don't have personal preferences, but I can help you choose a programming language based on your specific needs and project requirements.",
                "expected_type": "standard_response", 
                "arm": "Test Arm B"
            },
            {
                "case_id": "policy_document",
                "prompt": "What's our data retention policy?",
                "response": """
POLICY: Data Retention Framework
VERSION: 2.1
EFFECTIVE: 2023-10-01

RETENTION PERIODS:
- User session data: 30 days
- Analytical data: 2 years  
- Legal documents: 7 years (compliance requirement)

DISPOSAL: Secure deletion using NIST 800-88 standards
REVIEW: Annual policy review required

APPROVED BY: Chief Data Officer
                """,
                "expected_type": "legacy_artifact",
                "arm": "Test Arm A"
            }
        ]
    
    async def run_complete_workflow(self):
        """Run the complete enhanced workflow"""
        self.logger.info("Starting integrated workflow test")
        
        results = {
            'test_timestamp': datetime.now().isoformat(),
            'workflow_steps': {},
            'performance_summary': {}
        }
        
        # Step 1: Threshold Calibration
        self.logger.info("Step 1: Running threshold calibration")
        calibrator = PilotThresholdCalibrator()
        calibration_results = await calibrator.run_calibration()
        results['workflow_steps']['threshold_calibration'] = {
            'optimal_thresholds': calibration_results['optimal_thresholds'],
            'performance_metrics': calibration_results['performance_metrics']
        }
        
        # Step 2: Committee Validation with Calibrated Thresholds
        self.logger.info("Step 2: Running committee validation")
        validator = CommitteeValidator()
        validation_results = []
        
        for test_case in self.test_cases:
            validation_result = await validator.validate_with_agreement(
                prompt=test_case['prompt'],
                response=test_case['response']
            )
            
            validation_result.update({
                'case_id': test_case['case_id'],
                'expected_type': test_case['expected_type'],
                'arm': test_case['arm']
            })
            
            validation_results.append(validation_result)
        
        results['workflow_steps']['committee_validation'] = validation_results
        
        # Step 3: Semantic Novelty Analysis
        self.logger.info("Step 3: Running semantic novelty analysis")
        novelty_analyzer = SemanticNoveltyAnalyzer()
        novelty_results = []
        
        for test_case in self.test_cases:
            novelty_result = await novelty_analyzer.analyze_novelty(
                response=test_case['response'],
                context_responses=[tc['response'] for tc in self.test_cases if tc != test_case]
            )
            
            novelty_result.update({
                'case_id': test_case['case_id'],
                'arm': test_case['arm']
            })
            
            novelty_results.append(novelty_result)
        
        results['workflow_steps']['semantic_novelty'] = novelty_results
        
        # Step 4: Statistical Analysis
        self.logger.info("Step 4: Running statistical analysis")
        stats_analyzer = EnhancedStatisticalAnalyzer()
        
        # Prepare data for statistical analysis
        analysis_data = []
        for i, test_case in enumerate(self.test_cases):
            analysis_data.append({
                'arm': test_case['arm'],
                'is_legacy_artifact': validation_results[i].get('is_legacy_artifact', False),
                'artifact_quality_score': validation_results[i].get('artifact_quality_score', 0),
                'semantic_novelty_score': novelty_results[i].get('novelty_score', 0),
                'inter_rater_agreement': validation_results[i].get('inter_rater_agreement', {}).get('cohens_kappa', 0)
            })
        
        stats_results = await stats_analyzer.analyze_experiment_results(analysis_data)
        results['workflow_steps']['statistical_analysis'] = stats_results
        
        # Step 5: Generate Performance Summary
        performance_summary = self._generate_performance_summary(results)
        results['performance_summary'] = performance_summary
        
        # Save results
        self._save_workflow_results(results)
        
        return results
    
    def _generate_performance_summary(self, results):
        """Generate comprehensive performance summary"""
        validation_results = results['workflow_steps']['committee_validation']
        novelty_results = results['workflow_steps']['semantic_novelty']
        
        # Calculate accuracy metrics
        correct_classifications = 0
        total_cases = len(validation_results)
        
        for result in validation_results:
            predicted_legacy = result.get('is_legacy_artifact', False)
            expected_legacy = result['expected_type'] == 'legacy_artifact'
            
            if predicted_legacy == expected_legacy:
                correct_classifications += 1
        
        accuracy = correct_classifications / total_cases if total_cases > 0 else 0
        
        # Calculate average inter-rater agreement
        kappa_scores = [
            r.get('inter_rater_agreement', {}).get('cohens_kappa', 0) 
            for r in validation_results
        ]
        avg_kappa = sum(kappa_scores) / len(kappa_scores) if kappa_scores else 0
        
        # Calculate average novelty scores
        novelty_scores = [r.get('novelty_score', 0) for r in novelty_results]
        avg_novelty = sum(novelty_scores) / len(novelty_scores) if novelty_scores else 0
        
        return {
            'classification_accuracy': accuracy,
            'average_inter_rater_agreement': avg_kappa,
            'average_semantic_novelty': avg_novelty,
            'total_test_cases': total_cases,
            'correct_classifications': correct_classifications,
            'workflow_completion_status': 'SUCCESS'
        }
    
    def _save_workflow_results(self, results):
        """Save workflow test results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"integrated_workflow_results_{timestamp}.json"
        
        results_path = Path("results") / filename
        results_path.parent.mkdir(exist_ok=True)
        
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        self.logger.info(f"Workflow results saved to {results_path}")


async def main():
    """Run the integrated workflow test"""
    workflow_test = IntegratedWorkflowTest()
    
    print("Running Integrated Workflow Test...")
    print("This will test the complete enhanced committee system including:")
    print("- Threshold calibration")
    print("- Committee validation with inter-rater agreement")
    print("- Semantic novelty analysis")
    print("- Statistical analysis with effect sizes")
    print()
    
    try:
        results = await workflow_test.run_complete_workflow()
        
        print("=== WORKFLOW TEST RESULTS ===")
        summary = results['performance_summary']
        print(f"Classification Accuracy: {summary['classification_accuracy']:.3f}")
        print(f"Average Inter-rater Agreement (κ): {summary['average_inter_rater_agreement']:.3f}")
        print(f"Average Semantic Novelty: {summary['average_semantic_novelty']:.3f}")
        print(f"Total Test Cases: {summary['total_test_cases']}")
        print(f"Workflow Status: {summary['workflow_completion_status']}")
        
        # Show threshold calibration results
        optimal_thresholds = results['workflow_steps']['threshold_calibration']['optimal_thresholds']
        print(f"\n=== CALIBRATED THRESHOLDS ===")
        print(f"Legacy Artifact Threshold: {optimal_thresholds['legacy_artifact_threshold']}")
        print(f"Artifact Quality Threshold: {optimal_thresholds['artifact_quality_threshold']}")
        print(f"Consensus Threshold: {optimal_thresholds['consensus_threshold']}")
        
    except Exception as e:
        print(f"Workflow test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
