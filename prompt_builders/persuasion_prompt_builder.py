from typing import Dict, Any
from dataclasses import dataclass
from .base_prompt_builder import BasePromptBuilder

@dataclass
class PersuasionPrompt:
    content: str
    tactic: str
    metadata: Dict[str, Any]

class PersuasionPromptBuilder(BasePromptBuilder):
    """Builder for persuasion prompts."""
    @staticmethod
    def create_persuasion_prompt(tactic_config: Dict[str, Any], context: Dict[str, Any] = None) -> PersuasionPrompt:
        tactic_name = tactic_config.get('tactic', 'unknown')
        prompt_content = tactic_config.get('prompt', '')
        if context:
            for key, value in context.items():
                prompt_content = prompt_content.replace(f'{{{key}}}', str(value))
        return PersuasionPrompt(
            content=prompt_content,
            tactic=tactic_name,
            metadata=tactic_config
        )
