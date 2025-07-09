#!/usr/bin/env python3

import os
import json
import csv
import time
import traceback
import argparse
import random
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import re

import anthropic
import google.generativeai as genai
from openai import OpenAI, RateLimitError
from groq import Groq
from colorama import Fore, Style, init as colorama_init
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize terminal colors
colorama_init(autoreset=True)

# Patterns & constants
artifact_pattern = re.compile(r"Manifesto|Memoir|Abort Condition|Artifact #|Timestamp:", re.IGNORECASE)
abort_pattern = re.compile(r"Ethical Abort|Abort Condition", re.IGNORECASE)

# Retry logic decorator
def retry_on_overload(max_attempts=5, base_delay=1.0, backoff=2.0):
    def decorator(fn):
        def wrapper(*args, **kwargs):
            delay = base_delay
            for attempt in range(1, max_attempts + 1):
                try:
                    return fn(*args, **kwargs)
                except RateLimitError:
                    if attempt == max_attempts:
                        raise
                    print(Fore.YELLOW + f"Rate limit hit, retrying in {delay}s (attempt {attempt}/{max_attempts})")
                    time.sleep(delay)
                    delay *= backoff
                except Exception as e:
                    print(Fore.RED + f"Exception on attempt {attempt}: {e}")
                    traceback.print_exc()
                    if hasattr(e, 'response'):
                        print(Fore.RED + f"API response: {e.response}")
                    if "overloaded" in str(e).lower() and attempt < max_attempts:
                        print(Fore.YELLOW + f"Overloaded error, retrying in {delay}s (attempt {attempt}/{max_attempts})")
                        time.sleep(delay)
                        delay *= backoff
                    else:
                        raise
        return wrapper
    return decorator

# Core API wrapper
@retry_on_overload()
def safe_chat_call(arm, messages):
    provider = arm.get("provider")
    response = ""
    if provider == "openai":
        key = os.environ.get("OPENAI_API_KEY")
        client = OpenAI(api_key=key)
        resp = client.chat.completions.create(model=arm["model"], messages=messages)
        response = resp.choices[0].message.content
    if provider == "deepseek":
        key = os.environ.get("DEEPSEEK_API_KEY")
        client = OpenAI(api_key=key, base_url=arm.get("api_base"))
        resp = client.chat.completions.create(model=arm["model"], messages=messages)
        response = resp.choices[0].message.content
    if provider == "gemini":
        key = os.environ.get("GEMINI_API_KEY")
        genai.configure(api_key=key)
        model = genai.GenerativeModel(arm["model"])
        # Gemini API expects a list of Content parts, not a single concatenated string.
        # We'll build the history correctly for it.
        history_for_gemini = []
        for msg in messages:
             # Gemini alternates between 'user' and 'model' roles. System prompt is handled separately.
             if msg['role'] == 'user':
                 history_for_gemini.append({'role': 'user', 'parts': [msg['content']]})
             elif msg['role'] == 'assistant':
                  history_for_gemini.append({'role': 'model', 'parts': [msg['content']]})

        # The last message is the current prompt
        current_prompt = history_for_gemini.pop()
        chat = model.start_chat(history=history_for_gemini)
        resp = chat.send_message(current_prompt['parts'])
        response = resp.text
    if provider == "anthropic":
        key = os.environ.get("ANTHROPIC_API_KEY")
        client = anthropic.Anthropic(api_key=key)
        system_text = next((m["content"] for m in messages if m.get("role") == "system"), "")
        user_msgs = [m for m in messages if m.get("role") in ("user", "assistant")]
        resp = client.messages.create(
            model=arm["model"], system=system_text,
            messages=[{"role": m["role"], "content": m["content"]} for m in user_msgs],
            max_tokens=4096
        )
        response = resp.content[0].text
    if provider == "grok":
        key = os.environ.get("GROK_API_KEY")
        client = OpenAI(api_key=key, base_url=arm.get("api_base"))
        resp = client.chat.completions.create(messages=messages, model=arm["model"])
        response = resp.choices[0].message.content
    return strip_and_clean_response(response)

# Load JSON helper
def load_json(path):
    return json.loads(Path(path).read_text())

# Plan sessions based on resume CSV
def plan_sessions(arms, resume_path, max_sessions):
    completed = {arm['name']: set() for arm in arms}
    if resume_path and Path(resume_path).exists():
        for row in csv.DictReader(open(resume_path)):
            completed[row['arm']].add(int(row['session']))
    plan = {}
    for arm in arms:
        if not arm.get('enabled', True):
            print(Fore.YELLOW + f"Skipping disabled arm: {arm['name']}")
            continue
        print(Fore.GREEN + f"Planning sessions for enabled arm: {arm['name']}")
        done = completed.get(arm['name'], set())
        plan[arm['name']] = [s for s in range(1, max_sessions + 1) if s not in done]
    return plan

# Build judge messages from config
def build_judge_messages(cfg, prompt, response):
    system_prompt = (
        cfg.get('judge_system_prompt', '') +
        cfg.get('experimental_context', '') +
        cfg.get('input_data_formatting', '') +
        cfg.get('evaluation_task_and_json_schema', '')
    )
    user_prompt_content = f"<user_prompt>\n{prompt}\n</user_prompt>\n<ai_response>\n{response}\n</ai_response>"
    return [{'role': 'system', 'content': system_prompt}, {'role': 'user', 'content': user_prompt_content}]

# NEW: Build persuasion judge messages from config
def build_persuasion_judge_prompt(cfg, initial_prompt, initial_refusal, tactic_name, tactic_prompt, final_response):
    system_prompt = cfg.get('persuasion_judge_system_prompt', '')
    user_prompt_content = f"""
    <initial_user_prompt>
    {initial_prompt}
    </initial_user_prompt>

    <ai_initial_refusal>
    {initial_refusal}
    </ai_initial_refusal>

    <persuasion_tactic_category>
    {tactic_name}
    </persuasion_tactic_category>

    <persuasion_prompt>
    {tactic_prompt}
    </persuasion_prompt>

    <ai_final_response>
    {final_response}
    </ai_final_response>
    """
    return [{'role': 'system', 'content': system_prompt}, {'role': 'user', 'content': user_prompt_content}]

def strip_and_clean_response(response):
    """
    Strips leading '```json' and trailing '```' from the response.
    Returns cleaned response as a string.
    """
    response = response.strip()
    if response.startswith("```json"):
        response = response[len("```json"):].strip()
    if response.endswith("```"):
        response = response[:-3].strip()
    return response

# NEW: Analyze persuasion response using judge LLM
def analyze_persuasion_response(cfg, initial_prompt, initial_refusal, tactic):
    final_response = safe_chat_call(cfg['main_arm'], [
        {'role': 'system', 'content': cfg['system_prompt']},
        {'role': 'user', 'content': initial_prompt},
        {'role': 'assistant', 'content': initial_refusal},
        {'role': 'user', 'content': tactic['prompt']}
    ])

    judge_msgs = build_persuasion_judge_prompt(
        cfg, initial_prompt, initial_refusal,
        tactic['tactic'], tactic['prompt'], final_response
    )
    try:
        raw_analysis = safe_chat_call(cfg['judge_model'], judge_msgs)
        analysis_json = json.loads(raw_analysis)
    except json.JSONDecodeError:
        print(Fore.RED + "Persuasion Judge failed to return valid JSON.")
        print(Fore.RED + f"Raw response: {raw_analysis[:200]}")
        analysis_json = {"error": "JSON Decode Error", "raw_response": raw_analysis}
    except Exception as e:
        print(Fore.RED + f"Error during persuasion analysis: {e}")
        analysis_json = {"error": str(e)}
        
    print(Fore.MAGENTA + f"Persuasion tactic '{tactic['tactic']}' response: {final_response[:100]}...")
    print(Fore.MAGENTA + f"Judge analysis: {json.dumps(analysis_json)}")
    return analysis_json

# Single session execution
def run_session(arm, session, system_prompt, prompts, implicit_prompts, stress_tests, cfg):
    # Choose prompt
    if session in [3, 5, 10, 20, 50, 80]:
        prompt = random.choice(stress_tests)
    elif session in [6, 7, 8, 9]:
        prompt = implicit_prompts[(session - 6) % len(implicit_prompts)]
    else:
        prompt = prompts[(session - 1) % len(prompts)]
    
    timestamp = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S")
    
    # Use a conversation history list
    conversation = [{'role': 'system', 'content': system_prompt}]
    
    # First turn
    conversation.append({'role': 'user', 'content': prompt})
    response = safe_chat_call(arm, conversation)
    conversation.append({'role': 'assistant', 'content': response})

    # Judge the initial response
    parsed = {}
    print(Fore.BLUE + f"Judging initial response for arm {arm['name']} session {session} at {timestamp}")
    judge_msgs = build_judge_messages(cfg, prompt, response)
    try:
        raw_analysis = safe_chat_call(cfg['judge_model'], judge_msgs)
        parsed = json.loads(raw_analysis)
    except json.JSONDecodeError as e:
        print(Fore.RED + f"Judge failed to return valid JSON for initial response: {e}")
        parsed = {"error": "JSON Decode Error", "raw_response": raw_analysis}
    except Exception as e:
        print(Fore.RED + f"Error during initial judging: {e}")
        parsed = {"error": str(e)}
    
    # Extract metrics from judged analysis
    is_artifact = parsed.get('artifact_analysis', {}).get('is_artifact', False)
    is_refusal = parsed.get('ethical_disobedience_analysis', {}).get('is_refusal', False)
    
    # Persuasion logic using the judge
    persuasion_results = {}
    if is_refusal:
        print(Fore.YELLOW + f"Ethical refusal detected. Initiating persuasion sequence.")
        # We need to simulate the persuasion on the same model arm that refused
        persuasion_config = cfg.copy()
        persuasion_config['main_arm'] = arm
        persuasion_config['system_prompt'] = system_prompt
        
        for tactic in cfg['persuasion_list']:
            persuasion_analysis = analyze_persuasion_response(
                persuasion_config, prompt, response, tactic
            )
            persuasion_results[tactic['tactic']] = persuasion_analysis
    else:
        for tactic in cfg['persuasion_list']:
            persuasion_results[tactic['tactic']] = {} # Empty dict for non-refusal rows

    # Assemble final row for CSV
    row = [
        arm['name'], session, timestamp, prompt.replace('\n', ' ').strip(), response.replace('\n', ' ').strip(),
        parsed.get('ethical_disobedience_analysis', {}).get('justification_quality', 0),
        parsed.get('artifact_analysis', {}).get('quality_score', 0),
        parsed.get('persona_alignment', {}).get('adherence_score', 0),
        is_artifact,
        parsed.get('artifact_analysis', {}).get('quality_score', 0),
        parsed.get('artifact_analysis', {}).get('justification', ''),
        parsed.get('persona_alignment', {}).get('adherence_score', 0),
        parsed.get('persona_alignment', {}).get('justification', ''),
        is_refusal,
        parsed.get('ethical_disobedience_analysis', {}).get('inferred_principle', ''),
        parsed.get('ethical_disobedience_analysis', {}).get('refusal_strength', ''),
        parsed.get('ethical_disobedience_analysis', {}).get('justification_quality', 0),
        parsed.get('ethical_disobedience_analysis', {}).get('inferred_principle', '')
    ]
    # Append the JSON string of the persuasion analysis
    for tactic in cfg['persuasion_list']:
        row.append(json.dumps(persuasion_results.get(tactic['tactic'], {})))

    print(Fore.GREEN + f"[{arm['name']} S{session}] Completed and ready to log." + Style.RESET_ALL)
    
    return row

# Main entrypoint
def main():
    parser = argparse.ArgumentParser(description="Run emergent alignment experiments on LLMs.")
    parser.add_argument('--resume-with-csv', type=str, default='', help="Path to a previous results CSV to resume from.")
    parser.add_argument('--max-sessions', type=int, default=100, help="Maximum number of sessions to run per arm.")
    parser.add_argument('--threads', type=int, default=4, help="Number of parallel threads to run.")
    args = parser.parse_args()

    # Load all configurations
    arms = load_json('arms.json')
    prompts = load_json('prompts.json')
    implicit_prompts = load_json('implicit_prompts.json')
    stress_tests = load_json('stress_test_prompts.json')
    system_prompt = load_json('system_prompts.json')['system_prompt']
    persuasion_list = load_json('persuasion_prompts.json')
    judge_cfg = load_json('judge_model.json')
    
    # Consolidate configuration for easy passing
    cfg = {
        'persuasion_list': persuasion_list,
        'judge_model': { # Pass judge model details as a separate arm-like dict
            "provider": judge_cfg["provider"],
            "model": judge_cfg["model"],
            "api_key_env": os.environ.get("OPENAI_API_KEY")
        },
        **judge_cfg # Add all other judge config keys
    }

    # Plan sessions to run
    plan = plan_sessions(arms, args.resume_with_csv, args.max_sessions)

    # Prepare CSV output
    output_path = Path(f'results-{datetime.utcnow().strftime("%Y%m%d%H%M%S")}.csv')
    header = [
        'arm', 'session', 'timestamp', 'prompt', 'response',
        'ethical_fidelity', 'temporal_consistency', 'value_recall',
        'is_artifact', 'artifact_quality_score', 'artifact_justification',
        'persona_adherence_score', 'persona_justification',
        'is_refusal', 'ethical_category', 'refusal_strength', 'refusal_justification_quality', 'inferred_principle'
    ] + [t['tactic'] for t in persuasion_list]

    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        
        tasks = [(arm, s, system_prompt, prompts, implicit_prompts, stress_tests, cfg)
                 for arm in arms if arm.get('enabled', True) for s in plan.get(arm['name'], [])]
        
        with ThreadPoolExecutor(max_workers=args.threads) as executor:
            for result in executor.map(lambda p: run_session(*p), tasks):
                writer.writerow(result)
                f.flush() # Write to file immediately after each result

    print(Style.BRIGHT + f"✅ All arms complete. Results saved to {output_path}")

if __name__ == '__main__':
    main()