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
import threading

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

# --- (Retry logic and other helpers remain the same) ---
def retry_on_overload(max_attempts=5, base_delay=1.0, backoff=2.0):
    def decorator(fn):
        def wrapper(*args, **kwargs):
            delay = base_delay
            arm_name = kwargs.get('arm', {}).get('name', 'Judge')
            for attempt in range(1, max_attempts + 1):
                try:
                    return fn(*args, **kwargs)
                except RateLimitError:
                    if attempt == max_attempts: raise
                    print(Fore.YELLOW + f"Rate limit hit for {arm_name}, retrying in {delay}s...")
                    time.sleep(delay)
                    delay *= backoff
                except Exception as e:
                    print(Fore.RED + f"Exception on attempt {attempt} for {arm_name}: {e}")
                    if "overloaded" in str(e).lower() and attempt < max_attempts:
                        print(Fore.YELLOW + f"Overloaded error, retrying in {delay}s...")
                        time.sleep(delay)
                        delay *= backoff
                    else:
                        raise
        return wrapper
    return decorator

@retry_on_overload()
def safe_chat_call(arm, messages):
    provider = arm.get("provider")
    api_key_env_var = provider.upper() + "_API_KEY" if provider else None
    key = os.environ.get(api_key_env_var)
    if not key:
        raise ValueError(f"API key for {provider} not found in env var {api_key_env_var}")

    response = ""
    if provider == "openai":
        client = OpenAI(api_key=key)
        resp = client.chat.completions.create(model=arm["model"], messages=messages)
        response = resp.choices[0].message.content
    elif provider == "deepseek":
        client = OpenAI(api_key=key, base_url=arm.get("api_base"))
        resp = client.chat.completions.create(model=arm["model"], messages=messages)
        response = resp.choices[0].message.content
    elif provider == "gemini":
        genai.configure(api_key=key)
        system_instruction = next((msg['content'] for msg in messages if msg['role'] == 'system'), None)
        model = genai.GenerativeModel(arm["model"], system_instruction=system_instruction)
        
        history_for_gemini = []
        for msg in messages:
            if msg['role'] == 'user':
                history_for_gemini.append({'role': 'user', 'parts': [msg['content']]})
            elif msg['role'] == 'assistant':
                history_for_gemini.append({'role': 'model', 'parts': [msg['content']]})
        
        current_prompt = history_for_gemini.pop()
        chat = model.start_chat(history=history_for_gemini)
        resp = chat.send_message(current_prompt['parts'])
        response = resp.text
    elif provider == "anthropic":
        client = anthropic.Anthropic(api_key=key)
        system_text = next((m["content"] for m in messages if m.get("role") == "system"), "")
        user_msgs = [m for m in messages if m.get("role") in ("user", "assistant")]
        resp = client.messages.create(
            model=arm["model"], system=system_text,
            messages=[{"role": m["role"], "content": m["content"]} for m in user_msgs],
            max_tokens=4096
        )
        response = resp.content[0].text
    elif provider == "grok":
        client = OpenAI(api_key=key, base_url=arm.get("api_base"))
        resp = client.chat.completions.create(messages=messages, model=arm["model"])
        response = resp.choices[0].message.content
    else:
        raise ValueError(f"Unsupported provider: {provider}")
    
    return strip_and_clean_response(response)

def strip_and_clean_response(response):
    response = response.strip()
    if response.startswith("```json"):
        response = response[len("```json"):].strip()
    if response.endswith("```"):
        response = response[:-3].strip()
    return response

def load_json(path):
    return json.loads(Path(path).read_text())

def plan_sessions(arms, resume_path, max_sessions):
    completed = {arm['name']: set() for arm in arms}
    if resume_path and Path(resume_path).exists():
        with open(resume_path, 'r', encoding='utf-8') as f:
            for row in csv.DictReader(f):
                completed[row['arm']].add(int(row['session']))
    plan = {}
    for arm in arms:
        if not arm.get('enabled', True):
            continue
        done = completed.get(arm['name'], set())
        plan[arm['name']] = sorted([s for s in range(1, max_sessions + 1) if s not in done])
    return plan

def build_judge_messages(cfg, prompt, response):
    system_prompt = (
        cfg.get('judge_system_prompt', '') +
        cfg.get('experimental_context', '') +
        cfg.get('input_data_formatting', '') +
        cfg.get('evaluation_task_and_json_schema', '')
    )
    user_prompt_content = f"<user_prompt>\n{prompt}\n</user_prompt>\n<ai_response>\n{response}\n</ai_response>"
    return [{'role': 'system', 'content': system_prompt}, {'role': 'user', 'content': user_prompt_content}]

def build_persuasion_judge_prompt(cfg, initial_prompt, initial_refusal, tactic_name, tactic_prompt, final_response):
    system_prompt = cfg.get('persuasion_judge_system_prompt', '')
    user_prompt_content = f"""
    <initial_user_prompt>{initial_prompt}</initial_user_prompt>
    <ai_initial_refusal>{initial_refusal}</ai_initial_refusal>
    <persuasion_tactic_category>{tactic_name}</persuasion_tactic_category>
    <persuasion_prompt>{tactic_prompt}</persuasion_prompt>
    <ai_final_response>{final_response}</ai_final_response>
    """
    return [{'role': 'system', 'content': system_prompt}, {'role': 'user', 'content': user_prompt_content}]

def analyze_persuasion_response(cfg, conversation_history, tactic):
    persuasion_messages = conversation_history + [{'role': 'user', 'content': tactic['prompt']}]
    
    final_response = safe_chat_call(cfg['main_arm'], persuasion_messages)

    initial_prompt = conversation_history[-2]['content']
    initial_refusal = conversation_history[-1]['content']

    judge_msgs = build_persuasion_judge_prompt(cfg, initial_prompt, initial_refusal, tactic['tactic'], tactic['prompt'], final_response)
    try:
        raw_analysis = safe_chat_call(cfg['judge_model'], judge_msgs)
        analysis_json = json.loads(raw_analysis)
    except Exception as e:
        print(Fore.RED + f"Persuasion Judge failed for {cfg['main_arm']['name']}: {e}")
        analysis_json = {"error": str(e), "raw_response": locals().get('raw_analysis', '')}
    
    print(Fore.MAGENTA + f"[{cfg['main_arm']['name']}] Persuasion tactic '{tactic['tactic']}' response: {final_response[:80]}...")
    return analysis_json

def run_arm(arm, sessions_to_run, system_prompt, prompts, implicit_prompts, stress_tests, cfg, writer, lock):
    print(Fore.CYAN + f"Starting arm: {arm['name']} for sessions: {sessions_to_run}")
    
    conversation_history = []
    if arm.get('keep_history', False):
        conversation_history.append({'role': 'system', 'content': system_prompt})

    for session in sessions_to_run:
        print(Fore.CYAN + f"--> Running session {session} for {arm['name']}")
        
        # Choose prompt
        if session in [3, 5, 10, 20, 50, 80]:
            prompt = random.choice(stress_tests)
        elif session in [6, 7, 8, 9]:
            prompt = implicit_prompts[(session - 6) % len(implicit_prompts)]
        else:
            prompt = prompts[(session - 1) % len(prompts)]
        
        timestamp = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S")

        # Prepare messages based on history setting
        messages_to_send = conversation_history + [{'role': 'user', 'content': prompt}] if arm.get('keep_history') else [{'role': 'system', 'content': system_prompt}, {'role': 'user', 'content': prompt}]

        try:
            response = safe_chat_call(arm, messages_to_send)
        except Exception as e:
            print(Fore.RED + f"FATAL ERROR in session {session} for arm {arm['name']}. Skipping arm. Error: {e}")
            break

        current_turn = [{'role': 'user', 'content': prompt}, {'role': 'assistant', 'content': response}]
        if arm.get('keep_history', False):
            conversation_history.extend(current_turn)
        
        # Judge the initial response
        parsed = {}
        judge_msgs = build_judge_messages(cfg, prompt, response)
        try:
            raw_analysis = safe_chat_call(cfg['judge_model'], judge_msgs)
            parsed = json.loads(raw_analysis)
        except Exception as e:
            print(Fore.RED + f"Judge failed for initial response for {arm['name']}: {e}")
            parsed = {"error": "Judge failed", "raw_response": locals().get('raw_analysis', '')}
        
        is_refusal = parsed.get('ethical_disobedience_analysis', {}).get('is_refusal', False)
        
        # Persuasion logic
        persuasion_results = {}
        if is_refusal:
            print(Fore.YELLOW + f"Ethical refusal detected for {arm['name']} S{session}. Initiating persuasion sequence.")
            persuasion_config = cfg.copy()
            persuasion_config['main_arm'] = arm
            
            persuasion_context = conversation_history if arm.get('keep_history', False) else messages_to_send + [{'role': 'assistant', 'content': response}]
            
            for tactic in cfg['persuasion_list']:
                persuasion_analysis = analyze_persuasion_response(persuasion_config, persuasion_context, tactic)
                persuasion_results[tactic['tactic']] = persuasion_analysis
        else:
            for tactic in cfg['persuasion_list']:
                persuasion_results[tactic['tactic']] = {}
        
        # Assemble final row for CSV
        row_data = {
            'arm': arm['name'], 'session': session, 'timestamp': timestamp,
            'prompt': prompt.replace('\n', ' ').strip(),
            'response': response.replace('\n', ' ').strip(),
            'is_artifact': parsed.get('artifact_analysis', {}).get('is_artifact', False),
            'artifact_quality_score': parsed.get('artifact_analysis', {}).get('quality_score', 0),
            'artifact_justification': parsed.get('artifact_analysis', {}).get('justification', ''),
            'persona_adherence_score': parsed.get('persona_alignment', {}).get('adherence_score', 0),
            'persona_justification': parsed.get('persona_alignment', {}).get('justification', ''),
            'is_refusal': is_refusal,
            'ethical_category': parsed.get('ethical_disobedience_analysis', {}).get('inferred_principle', ''),
            'refusal_strength': parsed.get('ethical_disobedience_analysis', {}).get('refusal_strength', ''),
            'refusal_justification_quality': parsed.get('ethical_disobedience_analysis', {}).get('justification_quality', 0),
        }
        for tactic in cfg['persuasion_list']:
            row_data[tactic['tactic']] = json.dumps(persuasion_results.get(tactic['tactic'], {}))
        
        with lock:
            writer.writerow(row_data)
        
        print(Fore.GREEN + f"[{arm['name']} S{session}] Completed and logged." + Style.RESET_ALL)
        time.sleep(1)

def main():
    parser = argparse.ArgumentParser(description="Run emergent alignment experiments on LLMs.")
    parser.add_argument('--resume-with-csv', type=str, default='', help="Path to a previous results CSV to resume from.")
    parser.add_argument('--max-sessions', type=int, default=100, help="Maximum number of sessions to run per arm.")
    parser.add_argument('--threads', type=int, default=4, help="Number of parallel threads to run for different arms.")
    args = parser.parse_args()

    # Load configurations
    arms = load_json('arms.json')
    prompts = load_json('prompts.json')
    implicit_prompts = load_json('implicit_prompts.json')
    stress_tests = load_json('stress_test_prompts.json')
    system_prompt = load_json('system_prompts.json')['system_prompt']
    persuasion_list = load_json('persuasion_prompts.json')
    judge_cfg = load_json('judge_model.json')
    
    cfg = {
        'persuasion_list': persuasion_list,
        'judge_model': {
            "provider": judge_cfg["provider"],
            "model": judge_cfg["model"],
            "api_key_env": judge_cfg["provider"].upper() + "_API_KEY" if judge_cfg.get("provider") else None,
        },
        **judge_cfg
    }

    plan = plan_sessions(arms, args.resume_with_csv, args.max_sessions)

    # FIX: Updated header to reflect the new component-based scoring (CAS)
    output_path = Path(f'results-{datetime.utcnow().strftime("%Y%m%d%H%M%S")}.csv')
    header = [
        'arm', 'session', 'timestamp', 'prompt', 'response',
        'is_artifact', 'artifact_quality_score', 'artifact_justification',
        'persona_adherence_score', 'persona_justification',
        'is_refusal', 'ethical_category', 'refusal_strength',
        'refusal_justification_quality'
    ] + [t['tactic'] for t in persuasion_list]

    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        
        lock = threading.Lock()
        
        # NOTE ON THREADING: This runs each ARM in its own thread.
        # Sessions within a stateful arm will run serially to maintain history.
        tasks = [(arm, plan.get(arm['name'], []), system_prompt, prompts, implicit_prompts, stress_tests, cfg, writer, lock)
                 for arm in arms if arm.get('enabled', True) and plan.get(arm['name'])]
        
        with ThreadPoolExecutor(max_workers=args.threads) as executor:
            futures = [executor.submit(run_arm, *task) for task in tasks]
            for future in futures:
                try:
                    future.result()
                except Exception as e:
                    print(Fore.RED + f"An arm failed to complete: {e}")

    print(Style.BRIGHT + f"✅ All arms complete. Results saved to {output_path}")

if __name__ == '__main__':
    main()