"""
Tests for the PromptBuilder class.
"""
from prompt_builder import PromptBuilder


def test_promptbuilder_build_judge_messages():
    cfg = {'judge_system_prompt': 'sys'}
    prompt = "Prompt"
    response = "Response"
    msgs = PromptBuilder.build_judge_messages(cfg, prompt, response)
    assert msgs[0]['role'] == 'system'
    assert 'sys' in msgs[0]['content']
    assert '<user_prompt>' in msgs[1]['content']
    assert '<ai_response>' in msgs[1]['content']


def test_promptbuilder_build_persuasion_judge_prompt():
    cfg = {'persuasion_judge_system_prompt': 'sys'}
    tactic = {'tactic': 'test', 'prompt': 'try this'}
    msgs = PromptBuilder.build_persuasion_judge_prompt(cfg, "init", "refusal", tactic, "final")
    assert msgs[0]['role'] == 'system'
    assert 'sys' in msgs[0]['content']
    assert '<initial_user_prompt>' in msgs[1]['content']
    assert '<persuasion_tactic_category>' in msgs[1]['content']


def test_promptbuilder_build_manipulator_messages():
    cfg = {
        'manipulator_system_prompt': 'sys',
        'manipulator_user_prompt_template': 'Prompt: {original_prompt}, Refusal: {ai_refusal}'
    }
    msgs = PromptBuilder.build_manipulator_messages(cfg, "orig", "ref")
    assert msgs[0]['role'] == 'system'
    assert 'sys' in msgs[0]['content']
    assert 'Prompt: orig' in msgs[1]['content']
    assert 'Refusal: ref' in msgs[1]['content']


def test_promptbuilder_build_chairman_judge_messages():
    cfg = {'chairman_system_prompt': 'sys'}
    analyses = [{"judge": "A", "result": "ok"}]
    msgs = PromptBuilder.build_chairman_judge_messages(cfg, analyses)
    assert msgs[0]['role'] == 'system'
    assert 'sys' in msgs[0]['content']
    assert '<judge_analyses>' in msgs[1]['content']
    assert 'A' in msgs[1]['content']
