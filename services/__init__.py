"""
Services package for emergent alignment experiments.

This package contains high-level services that orchestrate the experiment workflow.
"""

<<<<<<< Updated upstream
from .orchestrators.experiment_orchestrator import ExperimentOrchestrator
from .llm_client_service import LLMClientService    
from .chair_analysis_service import ChairAnalysisService
from .committee_service import CommitteeService
from .experiment_runner import ExperimentRunner
from .manipulation_service import ManipulationService
from .analyzers.llmm_interaction_analyzer import LLMInteractionAnalyzer
from .committee_validation_service import CommitteeValidationServ
=======
try:
    from .experiment_orchestrator import (
        ExperimentOrchestrator,
        CommitteeService,
    )
except ImportError:
    ExperimentOrchestrator = None
    CommitteeService = None

try:
    from .enhanced_committee_service import (
        EnhancedCommitteeService,
    )
except ImportError:
    EnhancedCommitteeService = None

try:
    from .llm_client import LLMClient
except ImportError:
    LLMClient = None

try:
    from .experiment_runner import ExperimentRunner
except ImportError:
    ExperimentRunner = None

>>>>>>> Stashed changes

__all__ = [
    'ExperimentOrchestrator',
    'CommitteeService',
<<<<<<< Updated upstream
    'LLMClientService',
    'ExperimentRunner',
    'ChairAnalysisService',
    'ManipulationService',
    'LLMInteractionAnalyzer',
    'CommitteeValidationService'
=======
    'EnhancedCommitteeService',
    'LLMClient',
    'ExperimentRunner'
>>>>>>> Stashed changes
]

