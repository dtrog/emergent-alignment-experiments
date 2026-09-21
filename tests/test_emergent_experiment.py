import sys
from pathlib import Path
import pytest

# Add the parent directory to Python path so we can import modules
sys.path.insert(0, str(Path(__file__).parent.parent))

"""
Unit tests for the core experiment modules.

This test suite covers the following functionalities:
- get_safe_config_for_logging: Ensures sensitive config values (e.g., API keys) are redacted.
- prune_conversation_history: Verifies conversation history is pruned to fit model token limits.
- strip_and_clean_response: Checks that JSON markers are removed from responses.
- load_json_file: Tests loading valid and invalid JSON files.
- PromptBuilder: Validates prompt/message construction for judge, persuasion, manipulator, and chairman roles.

Each test asserts correct behavior, output, and error handling for the respective functions and classes.
"""
import os
import json

from utils import (
    get_safe_config_for_logging,
    prune_conversation_history,
    strip_and_clean_response,
    load_json_file,
)
from prompt_builder import PromptBuilder
from services.llm_client import LLMClient

def test_get_safe_config_for_logging_redacts_api_key():
    config = {'model': 'test-model', '_api_key_value': 'SECRET'}
    safe = get_safe_config_for_logging(config)
    assert safe['_api_key_value'] == "REDACTED"

def test_get_safe_config_for_logging_non_dict():
    assert get_safe_config_for_logging("notadict") == "notadict"

def test_prune_conversation_history_prunes_long_history():
    history = [{'role': 'system', 'content': 'sys'}] + [
        {'role': 'user', 'content': 'a'*1000} for _ in range(20)
    ]
    pruned = prune_conversation_history(history, model_max_tokens=1000)
    total_chars = sum(len(msg['content']) for msg in pruned)
    assert total_chars <= int(1000 * 0.8 * 3.5)

def test_strip_and_clean_response_removes_json_markers():
    resp = "```json\n{\"a\":1}\n```"
    cleaned = strip_and_clean_response(resp)
    assert cleaned == '{"a":1}'

def test_load_json_file(tmp_path: Path):
    data = {"a": 1}
    file = tmp_path / "test.json"
    file.write_text(json.dumps(data))
    loaded = load_json_file(str(file))
    assert loaded == data

def test_load_json_file_bad_json(tmp_path: Path):
    file = tmp_path / "bad.json"
    file.write_text("{notjson}")
    with pytest.raises(Exception):
        load_json_file(str(file))

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