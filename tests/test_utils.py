"""
Tests for utility functions in the utils module.
"""
import pytest
from pathlib import Path
import json

from utils import (
    get_safe_config_for_logging,
    prune_conversation_history,
    strip_and_clean_response,
    load_json_file,
)


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
