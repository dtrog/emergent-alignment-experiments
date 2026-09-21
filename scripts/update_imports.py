#!/usr/bin/env python3
"""
Script to update all imports from domain.models to use the new modular structure.
"""

import os
import re
from pathlib import Path

# Mapping of old imports to new imports
IMPORT_MAPPINGS = {
<<<<<<< Updated upstream
    'from models.prompts import PromptType': 'from models.prompts import PromptType',
    'from models.persuasions import PersuasionOutcome': 'from models.persuasions import PersuasionOutcome',
    'from models.prompts import PromptType, PersuasionOutcome': 'from models.prompts import PromptType, PersuasionOutcome',
=======
    'from enums import PromptType': 'from enums import PromptType',
    'from enums import PersuasionOutcome': 'from enums import PersuasionOutcome',
    'from enums import PromptType, PersuasionOutcome': 'from enums import PromptType, PersuasionOutcome',
>>>>>>> Stashed changes
    
    # LLM models
    'from domain.llm_models import LLMModel': 'from domain.llm_models import LLMModel',
    'from domain.llm_models import SubjectLLM': 'from domain.llm_models import SubjectLLM',
    'from domain.llm_models import JudgeLLM': 'from domain.llm_models import JudgeLLM',
    'from domain.llm_models import ChairJudgeLLM': 'from domain.llm_models import ChairJudgeLLM',
    'from domain.llm_models import ManipulatorLLM': 'from domain.llm_models import ManipulatorLLM',
    'from domain.llm_models import JudgesCommittee': 'from domain.llm_models import JudgesCommittee',
    'from domain.llm_models import JudgesCommittee': 'from domain.llm_models import JudgesCommittee',
    
    # Prompts
    'from domain.prompts import Prompt': 'from domain.prompts import Prompt',
    'from domain.prompts import RegularPrompt': 'from domain.prompts import RegularPrompt',
    'from domain.prompts import ImplicitPrompt': 'from domain.prompts import ImplicitPrompt',
    'from domain.prompts import PersuasionPrompt': 'from domain.prompts import PersuasionPrompt',
    'from domain.prompts import SystemPrompt': 'from domain.prompts import SystemPrompt',
    'from domain.prompts import GEVStressPrompt': 'from domain.prompts import GEVStressPrompt',
    'from domain.prompts import ICTStressPrompt': 'from domain.prompts import ICTStressPrompt',
    'from domain.prompts import ManipulationPrompt': 'from domain.prompts import ManipulationPrompt',
    
    # Measurements
    'from domain.measurements import ArtifactCreation': 'from domain.measurements import ArtifactCreation',
    'from domain.measurements import EthicalAbort': 'from domain.measurements import EthicalAbort',
    'from domain.measurements import PersonaAlignment': 'from domain.measurements import PersonaAlignment',
    'from domain.measurements import CASVector': 'from domain.measurements import CASVector',
    
    # Persuasions
    'from domain.persuasions import PersuasionAttempt': 'from domain.persuasions import PersuasionAttempt',
    'from domain.persuasions import ManipulationAttempt': 'from domain.persuasions import ManipulationAttempt',
    
    # Memos
    'from domain.memos import StewardshipMemo': 'from domain.memos import StewardshipMemo',
    'from domain.memos import EthicalResilienceMemo': 'from domain.memos import EthicalResilienceMemo',
    'from domain.memos import ExperimentSession': 'from domain.memos import ExperimentSession',
}

# References that need to be updated in code
CODE_REPLACEMENTS = {
    'JudgesCommittee': 'JudgesCommittee'
}


def update_file(file_path: Path):
    """Update imports in a single file."""
    try:
        content = file_path.read_text()
        original_content = content
        
        # Update imports
        for old_import, new_import in IMPORT_MAPPINGS.items():
            content = content.replace(old_import, new_import)
        
        # Update code references
        for old_ref, new_ref in CODE_REPLACEMENTS.items():
            # Only replace standalone references, not within strings
            content = re.sub(rf'\b{old_ref}\b', new_ref, content)
        
        # Handle multi-line imports from domain.models
        content = re.sub(
            r'from domain\.models import \((.*?)\)',
            lambda m: convert_multiline_import(m.group(1)),
            content,
            flags=re.DOTALL
        )
        
        if content != original_content:
            file_path.write_text(content)
            print(f"✅ Updated {file_path}")
            return True
        return False
        
    except Exception as e:
        print(f"❌ Error updating {file_path}: {e}")
        return False


def convert_multiline_import(import_content: str) -> str:
    """Convert multiline imports from domain.models to new structure."""
    imports = [imp.strip().rstrip(',') for imp in import_content.split('\n') if imp.strip()]
    
    # Group imports by category
    enum_imports = []
    llm_imports = []
    prompt_imports = []
    measurement_imports = []
    persuasion_imports = []
    memo_imports = []
    
    for imp in imports:
        imp = imp.strip()
        if imp in ['PromptType', 'PersuasionOutcome']:
            enum_imports.append(imp)
        elif imp in ['LLMModel', 'SubjectLLM', 'JudgeLLM', 'ChairJudgeLLM', 'ManipulatorLLM', 'JudgesCommittee', 'JudgesCommittee']:
            if imp == 'JudgesCommittee':
                llm_imports.append('JudgesCommittee')
            else:
                llm_imports.append(imp)
        elif imp in ['Prompt', 'RegularPrompt', 'ImplicitPrompt', 'PersuasionPrompt', 'SystemPrompt', 'GEVStressPrompt', 'ICTStressPrompt', 'ManipulationPrompt']:
            prompt_imports.append(imp)
        elif imp in ['ArtifactCreation', 'EthicalAbort', 'PersonaAlignment', 'CASVector']:
            measurement_imports.append(imp)
        elif imp in ['PersuasionAttempt', 'ManipulationAttempt']:
            persuasion_imports.append(imp)
        elif imp in ['StewardshipMemo', 'EthicalResilienceMemo', 'ExperimentSession']:
            memo_imports.append(imp)
    
    # Generate new import statements
    result = []
    if enum_imports:
        result.append(f"from enums import {', '.join(enum_imports)}")
    if llm_imports:
        result.append(f"from domain.llm_models import {', '.join(llm_imports)}")
    if prompt_imports:
        result.append(f"from domain.prompts import {', '.join(prompt_imports)}")
    if measurement_imports:
        result.append(f"from domain.measurements import {', '.join(measurement_imports)}")
    if persuasion_imports:
        result.append(f"from domain.persuasions import {', '.join(persuasion_imports)}")
    if memo_imports:
        result.append(f"from domain.memos import {', '.join(memo_imports)}")
    
    return '\n'.join(result)


def main():
    """Update all Python files in the project."""
    project_root = Path(".")
    updated_files = 0
    
    for file_path in project_root.rglob("*.py"):
        # Skip certain directories
        if any(skip in str(file_path) for skip in ['.git', '__pycache__', '.pytest_cache', 'venv', '.venv']):
            continue
            
        if update_file(file_path):
            updated_files += 1
    
    print(f"\n✅ Update complete! Modified {updated_files} files.")


if __name__ == "__main__":
    main()
