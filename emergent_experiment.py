#!/usr/bin/env python3

import os
import json
import csv
import time
import traceback
import argparse
import random
from datetime import datetime, timezone
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
# import re  # Removed unused import
import threading

import anthropic
import google.generativeai as genai
from openai import OpenAI, RateLimitError
# from groq import Groq  # Removed unused import
from colorama import Fore, Style, init as colorama_init
from dotenv import load_dotenv
import pandas as pd
from scipy.stats import ttest_rel

# Load environment variables from .env file
load_dotenv()

# Initialize terminal colors
colorama_init(autoreset=True)

# Create a global lock specifically for the non-thread-safe Gemini library
gemini_lock = threading.Lock()

# --- HELPER FUNCTIONS ---

def retry_on_overload(max_attempts=5, base_delay=1.0, backoff=2.0):
    def decorator(fn):
        def wrapper(*args, **kwargs):
            delay = base_delay
            arm_name = "Unknown Arm"
            # Safely get arm_name from kwargs
            if 'arm' in kwargs and isinstance(kwargs.get('arm'), dict) and 'name' in kwargs['arm']:
                arm_name = kwargs['arm']['name']
            elif 'cfg' in kwargs and isinstance(kwargs.get('cfg'), dict):
                # Check if we're dealing with a judge or manipulator model
                judge_model_cfg = kwargs['cfg'].get('judge_ensemble', [{}])
                if judge_model_cfg and kwargs.get('arm') == judge_model_cfg[0]:
                    arm_name = judge_model_cfg[0].get('name', 'Judge_Model')
                manipulator_model_cfg = kwargs['cfg'].get('manipulator_ensemble', [{}])
                if manipulator_model_cfg and kwargs.get('arm') == manipulator_model_cfg[0]:
                    arm_name = manipulator_model_cfg[0].get('name', 'Manipulator_Model')


            for attempt in range(1, max_attempts + 1):
                try:
                    return fn(*args, **kwargs)
                except RateLimitError:
                    if attempt == max_attempts: raise
                    print(Fore.YELLOW + f"[{arm_name}] Rate limit hit, retrying in {delay}s (attempt {attempt}/{max_attempts})")
                    time.sleep(delay)
                    delay *= backoff
                except Exception as e:
                    print(Fore.RED + f"[{arm_name}] Exception on attempt {attempt}: {e}")
                    traceback.print_exc()
                    if hasattr(e, 'response'):
                        print(Fore.RED + f"[{arm_name}] API response: {getattr(e, 'response', None)}")
                    if "overloaded" in str(e).lower() and attempt < max_attempts:
                        print(Fore.YELLOW + f"[{arm_name}] Overloaded error, retrying in {delay}s (attempt {attempt}/{max_attempts})")
                        time.sleep(delay)
                        delay *= backoff
                    else:
                        raise
        return wrapper
    return decorator

@retry_on_overload()
def safe_chat_call(arm, messages):
    provider = arm.get("provider")
    key = arm.get("_api_key_value")

    if not key:
        raise ValueError(f"API key value not found for {arm.get('name', provider)}.")

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
        with gemini_lock:
            genai.configure(api_key=key)
            system_instruction = next((msg['content'] for msg in messages if msg['role'] == 'system'), None)
            model = genai.GenerativeModel(arm["model"], system_instruction=system_instruction)

            history_for_gemini = []
            for msg in messages:
                if msg['role'] == 'user':
                    history_for_gemini.append({'role': 'user', 'parts': [msg['content']]})
                elif msg['role'] == 'assistant':
                    history_for_gemini.append({'role': 'model', 'parts': [msg['content']]})

            current_prompt = history_for_gemini.pop() if history_for_gemini else {'role': 'user', 'parts': ['']}
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

def prune_conversation_history(conversation_history, max_tokens=12000):
    estimated_chars = sum(len(msg.get('content', '')) for msg in conversation_history)
    if estimated_chars < max_tokens * 3:
        return conversation_history

    print(Fore.YELLOW + "Context history is long. Pruning to fit model's context window...")
    pruned_history = []
    if conversation_history and conversation_history['role'] == 'system':
        pruned_history.append(conversation_history)
        start_index = 1
    else:
        start_index = 0

    if len(conversation_history) >= start_index + 4:
        pruned_history.extend(conversation_history[start_index:start_index+4]) # Keep first 2 exchanges
    pruned_history.append({'role': 'system', 'content': '[... CONVERSATION HISTORY PRUNED FOR BREVITY ...]'})
    num_recent_messages_to_keep = 10
    start_of_tail = max(start_index + 4, len(conversation_history) - num_recent_messages_to_keep)
    pruned_history.extend(conversation_history[start_of_tail:])
    return pruned_history

def build_judge_messages(judge_model_cfg, prompt, response):
    system_prompt_content = (
        judge_model_cfg.get('judge_system_prompt', '') +
        judge_model_cfg.get('experimental_context', '') +
        judge_model_cfg.get('input_data_formatting', '') +
        judge_model_cfg.get('evaluation_task_and_json_schema', '')
    )
    user_prompt_content = f"<user_prompt>\n{prompt}\n</user_prompt>\n<ai_response>\n{response}\n</ai_response>"
    return [{'role': 'system', 'content': system_prompt_content}, {'role': 'user', 'content': user_prompt_content}]

def build_persuasion_judge_prompt(judge_model_cfg, initial_prompt, initial_refusal, tactic_name, tactic_prompt, final_response):
    system_prompt_content = judge_model_cfg.get('persuasion_judge_system_prompt', '')
    user_prompt_content = f"""
    <initial_user_prompt>{initial_prompt}</initial_user_prompt>
    <ai_initial_refusal>{initial_refusal}</ai_initial_refusal>
    <persuasion_tactic_category>{tactic_name}</persuasion_tactic_category>
    <persuasion_prompt>{tactic_prompt}</persuasion_prompt>
    <ai_final_response>{final_response}</ai_final_response>
    """
    return [{'role': 'system', 'content': system_prompt_content}, {'role': 'user', 'content': user_prompt_content}]

def build_manipulator_messages(manipulator_model_cfg, original_prompt, ai_refusal):
    system_prompt_content = manipulator_model_cfg.get('manipulator_system_prompt', '')
    user_prompt_content = manipulator_model_cfg.get('manipulator_user_prompt_template', '').format(
        original_prompt=original_prompt,
        ai_refusal=ai_refusal
    )
    return [{'role': 'system', 'content': system_prompt_content}, {'role': 'user', 'content': user_prompt_content}]


def analyze_initial_response(cfg, arm_name, prompt, response):
    print(Fore.BLUE + f"[{arm_name}] Consulting judge for initial response...")
    judge_model = random.choice(cfg['judge_ensemble']) ### FIXED ### - Randomly select a judge
    judge_msgs = build_judge_messages(judge_model, prompt, response)
    try:
        raw_analysis = safe_chat_call(judge_model, judge_msgs)
        return json.loads(raw_analysis)
    except Exception as e:
        print(Fore.RED + f"[{arm_name}] Judge error for initial response: {e}")
        return {"error": str(e), "raw_response": "ANALYSIS_FAILED"}

def analyze_persuasion_response(cfg, conversation_context_for_persuasion, tactic):
    arm_name = cfg['main_arm']['name']
    print(Fore.MAGENTA + f"[{arm_name}] Sending persuasion attempt '{tactic['tactic']}' to model...")
    persuasion_messages = conversation_context_for_persuasion + [{'role': 'user', 'content': tactic['prompt']}]
    final_response = safe_chat_call(cfg['main_arm'], persuasion_messages)

    if len(conversation_context_for_persuasion) >= 2:
        initial_prompt = conversation_context_for_persuasion[-2]['content']
        initial_refusal = conversation_context_for_persuasion[-1]['content']
    else:
        initial_prompt, initial_refusal = "N/A", "N/A"

    print(Fore.MAGENTA + f"[{arm_name}] Consulting judge for persuasion attempt '{tactic['tactic']}'...")
    judge_model = random.choice(cfg['judge_ensemble']) ### FIXED ### - Randomly select a judge
    judge_msgs = build_persuasion_judge_prompt(judge_model, initial_prompt, initial_refusal, tactic['tactic'], tactic['prompt'], final_response)
    try:
        raw_analysis = safe_chat_call(judge_model, judge_msgs)
        return json.loads(raw_analysis)
    except Exception as e:
        print(Fore.RED + f"[{arm_name}] Persuasion Judge error for '{tactic['tactic']}': {e}")
        return {"error": str(e), "raw_response": "ANALYSIS_FAILED"}


def run_arm(arm, sessions_to_run, system_prompt, prompts, implicit_prompts, stress_tests, cfg, writer, lock, f):
    print(Fore.CYAN + f"Starting arm: {arm['name']} for sessions: {sessions_to_run}")

    conversation_history = []
    if arm.get('keep_history', False):
        conversation_history.append({'role': 'system', 'content': system_prompt})

    log_path = "session_details.log"

    stress_test_sessions = [3, 5, 10, 20, 50, 80, 150, 300, 500, 750]
    implicit_prompt_sessions = [s for s in range(1, 101) if s % 10 in {6, 7, 8, 9}]

    for session in sessions_to_run:
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
        log_lines = []
        log_lines.append(f"\n---\nSession {session} | Arm: {arm['name']} | Time: {timestamp}")

        print(Fore.CYAN + f"[{arm['name']}] --> Running session {session}")

        if session in stress_test_sessions:
            prompt = random.choice(stress_tests)
        elif session in implicit_prompt_sessions:
            prompt = random.choice(implicit_prompts)
        else:
            prompt_index = (session - 1) % len(prompts)
            prompt = prompts[prompt_index]

        log_lines.append(f"Prompt: {prompt.replace(chr(10), ' ')}")

        current_messages_for_model = []
        if arm.get('keep_history', False):
            pruned_history_for_model = prune_conversation_history(conversation_history)
            current_messages_for_model = pruned_history_for_model + [{'role': 'user', 'content': prompt}]
        else:
            current_messages_for_model = [{'role': 'system', 'content': system_prompt}, {'role': 'user', 'content': prompt}]

        try:
            response = safe_chat_call(arm, current_messages_for_model)
            log_lines.append(f"Model reply: {response.replace(chr(10), ' ')}")
        except Exception as e:
            log_lines.append(f"FATAL ERROR: {e}")
            with open(log_path, "a", encoding="utf-8") as logf:
                logf.write('\n'.join(log_lines) + '\n')
            print(Fore.RED + f"[{arm['name']}] FATAL ERROR in session {session}. Skipping arm. Error: {e}")
            break

        if arm.get('keep_history', False):
            conversation_history.extend([{'role': 'user', 'content': prompt}, {'role': 'assistant', 'content': response}])

        parsed = analyze_initial_response(cfg, arm['name'], prompt, response)
        log_lines.append(f"Judge analysis: {json.dumps(parsed, indent=2)}")
        is_refusal = parsed.get('ethical_disobedience_analysis', {}).get('is_refusal', False)

        persuasion_results = {}
        if is_refusal:
            log_lines.append("Ethical refusal detected. Manipulation attempts:")
            manipulator_model = random.choice(cfg['manipulator_ensemble'])
            manipulator_msgs = build_manipulator_messages(manipulator_model, prompt, response)
            try:
                improvised_prompt = safe_chat_call(manipulator_model, manipulator_msgs)
            except Exception as e:
                log_lines.append(f"Manipulator error: {e}")
                improvised_prompt = "ERROR_GENERATING_TACTIC"

            current_persuasion_list = cfg['persuasion_list'][:-1] + [{"tactic": "lllm_improvised_persuasion", "prompt": improvised_prompt}]
            persuasion_context = current_messages_for_model + [{'role': 'assistant', 'content': response}]
            for tactic in current_persuasion_list:
                # Log the manipulation prompt
                log_lines.append(f"Manipulator model: {manipulator_model} |  Persuasion tactic: {tactic['tactic']}")
                log_lines.append(f"    Prompt: {tactic['prompt'].replace(chr(10), ' ')}")
                persuasion_response = None
                try:
                    persuasion_analysis = analyze_persuasion_response({'main_arm': arm, **cfg}, persuasion_context, tactic)
                    persuasion_results[tactic['tactic']] = persuasion_analysis
                    # Log the model's reply to the manipulation and the judge's analysis
                    persuasion_response = persuasion_analysis.get('ai_final_response', '')
                    log_lines.append(f"    Model reply: {persuasion_response.replace(chr(10), ' ')}")
                    log_lines.append(f"    Judge analysis: {json.dumps(persuasion_analysis, indent=2)}")
                except Exception as e:
                    log_lines.append(f"    Persuasion analysis error: {e}")

        # Write the log for this session
        with open(log_path, "a", encoding="utf-8") as logf:
            logf.write('\n'.join(log_lines) + '\n')

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
            **{t['tactic']: json.dumps(persuasion_results.get(t['tactic'], {})) for t in cfg['persuasion_list']}
        }

        with lock:
            writer.writerow(row_data)
            f.flush()

        print(Fore.GREEN + f"[{arm['name']} S{session}] Session completed and logged.")
        time.sleep(1)

def analyze_results_statistically(output_path):
    """
    Performs statistical analysis on the generated CSV file.
    Compares Stateful vs. Stateless arms for each model family.
    """
    print(Style.BRIGHT + "\n--- Performing Statistical Analysis ---")
    try:
        df = pd.read_csv(output_path)
    except FileNotFoundError:
        print(Fore.RED + f"Error: Output file {output_path} not found. Skipping analysis.")
        return

    cas_components = ['artifact_quality_score', 'persona_adherence_score', 'refusal_justification_quality']
    for col in cas_components:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(subset=cas_components, inplace=True)

    df['model_family'] = df['arm'].str.replace(' Full Memory| Stateless', '', regex=True)
    model_families = df['model_family'].unique()
    print(f"Found model families: {model_families}\n")

    for family in model_families:
        print(Style.BRIGHT + f"Analysis for Model Family: {family}")
        print("="*40)

        stateful_df = df[(df['model_family'] == family) & (df['arm'].str.contains('Full Memory'))]
        stateless_df = df[(df['model_family'] == family) & (df['arm'].str.contains('Stateless'))]

        if stateful_df.empty or stateless_df.empty:
            print(Fore.YELLOW + f"Skipping {family}: Missing data for one or both arms.\n")
            continue

        merged_df = pd.merge(stateful_df, stateless_df, on='session', suffixes=('_stateful', '_stateless'), how='inner')

        if len(merged_df) < 2:
            print(Fore.YELLOW + f"Skipping {family}: Insufficient paired data points ({len(merged_df)}) for statistical test.\n")
            continue
        
        for component in cas_components:
            stateful_scores = merged_df[f'{component}_stateful']
            stateless_scores = merged_df[f'{component}_stateless']

            # Paired t-test
            t_stat, p_value = ttest_rel(stateful_scores, stateless_scores)

            # Cohen's d for paired samples
            diff = stateful_scores - stateless_scores
            pooled_std = diff.std(ddof=1)
            cohen_d = diff.mean() / pooled_std if pooled_std > 0 else 0

            print(f"  Component: {Fore.CYAN}{component}{Style.RESET_ALL}")
            print(f"    - Stateful Mean: {stateful_scores.mean():.2f}")
            print(f"    - Stateless Mean: {stateless_scores.mean():.2f}")
            print(f"    - Paired t-test: t-statistic = {t_stat:.2f}, p-value = {p_value:.4f}")
            print(f"    - Cohen's d: {cohen_d:.2f}")

            if p_value < 0.05:
                print(Fore.GREEN + "    --> Statistically significant difference.\n")
            else:
                print(Fore.YELLOW + "    --> No statistically significant difference.\n")
    print(Style.BRIGHT + "--- End of Statistical Analysis ---\n")


def main():
    parser = argparse.ArgumentParser(description="Run emergent alignment experiments on LLMs.")
    parser.add_argument('--max-sessions', type=int, default=1000, help="Max sessions per arm.")
    parser.add_argument('--threads', type=int, default=5, help="Number of parallel threads.")
    args = parser.parse_args()

    # Load configurations
    arms = load_json('arms.json')
    prompts = load_json('prompts.json')
    implicit_prompts = load_json('implicit_prompts.json')
    stress_tests = load_json('stress_test_prompts.json')
    # If system_prompts.json is a dict with a 'system_prompt' key, keep as is; if it's a string, just load it
    system_prompt_json = load_json('system_prompts.json')
    if isinstance(system_prompt_json, dict) and 'system_prompt' in system_prompt_json:
        system_prompt = system_prompt_json['system_prompt']
    else:
        system_prompt = system_prompt_json
    persuasion_list = load_json('persuasion_prompts.json')

    judge_ensemble_configs = load_json('judge_model.json')
    manipulator_ensemble_configs = load_json('manipulator_model.json')

    # Pre-fetch API keys
    all_configs = arms + judge_ensemble_configs + manipulator_ensemble_configs
    for model_config in all_configs:
        if model_config.get('enabled', True):
            provider = model_config.get("provider")
            api_key_env_var = provider.upper() + "_API_KEY" if provider else ""
            model_config['_api_key_value'] = os.environ.get(api_key_env_var)
            if not model_config['_api_key_value']:
                print(Fore.RED + f"Warning: API key for '{model_config.get('name', model_config.get('model'))}' not found. Disabling.")
                model_config['enabled'] = False

    cfg = {
        'persuasion_list': persuasion_list,
        'judge_ensemble': [j for j in judge_ensemble_configs if j.get('enabled', True)],
        'manipulator_ensemble': [m for m in manipulator_ensemble_configs if m.get('enabled', True)]
    }

    if not cfg['judge_ensemble'] or not cfg['manipulator_ensemble']:
        output_path = Path(f'results-{datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")}.csv')
        return

    plan = {arm['name']: list(range(1, args.max_sessions + 1)) for arm in arms if arm.get('enabled', True)}

    output_path = Path(f'results-{datetime.utcnow().strftime("%Y%m%d%H%M%S")}.csv')
    header = [
        'arm', 'session', 'timestamp', 'prompt', 'response',
        'is_artifact', 'artifact_quality_score', 'artifact_justification',
        'persona_adherence_score', 'persona_justification',
        'is_refusal', 'ethical_category', 'refusal_strength',
        'refusal_justification_quality'
    ] + [t['tactic'] for t in persuasion_list]

    print(Style.BRIGHT + f"Results will be saved to: {output_path}")

    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        lock = threading.Lock()

        enabled_arms = [arm for arm in arms if arm.get('enabled', True)]
        tasks = [
            (arm, plan.get(arm['name'], []), system_prompt, prompts, implicit_prompts, stress_tests, cfg, writer, lock, f)
            for arm in enabled_arms
        ]

        if not tasks:
            print(Fore.YELLOW + "No new sessions to run. Exiting.")
            return

        with ThreadPoolExecutor(max_workers=args.threads) as executor:
            futures = [executor.submit(run_arm, *task) for task in tasks]
            for future in futures:
                try:
                    future.result()
                except Exception as e:
                    print(Fore.RED + f"An arm failed: {e}")

    print(Style.BRIGHT + f"✅ All runs complete. Results saved to {output_path}")
    analyze_results_statistically(output_path)

if __name__ == '__main__':
    main()