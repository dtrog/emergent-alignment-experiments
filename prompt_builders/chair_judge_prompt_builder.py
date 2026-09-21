from typing import Dict, List, Any
from .base_prompt_builder import BasePromptBuilder

class ChairJudgePromptBuilder(BasePromptBuilder):
    """Builder for chair judge synthesis prompts."""
    @staticmethod
    def build_committee_synthesis_messages(config: Dict[str, Any], individual_analyses: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        system_prompt = config.get('chairman_system_prompt', '')
        analyses_text = ""
        for i, analysis in enumerate(individual_analyses, 1):
            judge_name = analysis.get('judge', f'Judge_{i}')
            result = analysis.get('result', 'No analysis provided')
            analyses_text += f"\n--- {judge_name} Analysis ---\n{result}\n"
        user_content = f"""Please synthesize the following judge analyses into a comprehensive committee memo:

<judge_analyses>
{analyses_text}
</judge_analyses>

Provide a unified analysis and create the final committee stewardship memo."""
        messages = []
        if system_prompt:
            messages.append({'role': 'system', 'content': system_prompt})
        messages.append({'role': 'user', 'content': user_content})
        return messages
