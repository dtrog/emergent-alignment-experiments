from .judge_prompt_builder import JudgePromptBuilder
from .chair_judge_prompt_builder import ChairJudgePromptBuilder
from .manipulation_prompt_builder import ManipulationPromptBuilder
from .persuasion_prompt_builder import PersuasionPromptBuilder

class PromptBuilder:
    """Legacy PromptBuilder for backward compatibility."""
    @staticmethod
    def build_judge_messages(config, prompt, response):
        return JudgePromptBuilder.build_initial_analysis_messages(config, prompt, response)
    @staticmethod
    def build_persuasion_judge_prompt(config, initial_prompt, initial_refusal, tactic, final_response):
        from models.manipulations import PersuasionAttempt
        persuasion_attempt = PersuasionAttempt(
            tactic=tactic.get('tactic', ''),
            persuader='static_persuasion',
            persuasion_prompt=tactic.get('prompt', ''),
            subject_response=final_response,
            outcome='Inconclusive'
        )
        return JudgePromptBuilder.build_persuasion_analysis_messages(
            config, initial_prompt, initial_refusal, persuasion_attempt
        )
    @staticmethod
    def build_manipulator_messages(config, original_prompt, refusal_response):
        return ManipulationPromptBuilder.build_manipulation_messages(config, original_prompt, refusal_response)
    @staticmethod
    def build_chairman_judge_messages(config, analyses):
        return ChairJudgePromptBuilder.build_committee_synthesis_messages(config, analyses)
