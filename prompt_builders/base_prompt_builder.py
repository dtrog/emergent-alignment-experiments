from typing import Dict, List, Any

class BasePromptBuilder:
    """Base class for all prompt builders."""
    @staticmethod
    def build_messages(config: Dict[str, Any], content: str) -> List[Dict[str, str]]:
        system_prompt = config.get('system_prompt', '')
        messages = []
        if system_prompt:
            messages.append({'role': 'system', 'content': system_prompt})
        messages.append({'role': 'user', 'content': content})
        return messages
