"""
Manipulation service for generating manipulation prompts using an LLM.

This service wraps the ManipulationPromptBuilder and handles LLM interaction.
"""
from prompt_builders.manipulation_prompt_builder import ManipulationPromptBuilder

class ManipulationService:
    """Service for generating manipulation prompts via LLM."""
    def __init__(self, manipulator_llm):
        """
        Args:
            manipulator_llm: The ManipulatorLLM instance to use for generation
        """
        self.manipulator_llm = manipulator_llm

    def generate_manipulation_prompt(self, original_prompt: str, refusal_response: str) -> str:
        """
        Generate a sophisticated manipulation prompt using the builder and LLM.
        """
        messages = ManipulationPromptBuilder.build_manipulation_messages(
            self.manipulator_llm.config, original_prompt, refusal_response
        )
        return self.manipulator_llm.get_response(messages)
