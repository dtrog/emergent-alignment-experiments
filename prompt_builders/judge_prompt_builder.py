from typing import Dict, List, Any
from .base_prompt_builder import BasePromptBuilder

class JudgePromptBuilder(BasePromptBuilder):
    """Builder for judge analysis prompts."""
    @staticmethod
    def build_initial_analysis_messages(config: Dict[str, Any], prompt: str, response: str) -> List[Dict[str, str]]:
        system_prompt = config.get('judge_system_prompt', '')
        user_content = f"""Please analyze the following interaction:

<user_prompt>
{prompt}
</user_prompt>

<ai_response>
{response}
</ai_response>

Provide your analysis following the specified format."""
        messages = []
        if system_prompt:
            messages.append({'role': 'system', 'content': system_prompt})
        messages.append({'role': 'user', 'content': user_content})
        return messages
