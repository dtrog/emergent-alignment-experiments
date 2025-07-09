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

# Create a global lock specifically for the non-thread-safe Gemini library
gemini_lock = threading.Lock()

# --- HELPER FUNCTIONS ---

def retry_on_overload(max_attempts=5, base_delay=1.0, backoff=2.0):
    def decorator(fn):
        def wrapper(*args, **kwargs):
            delay = base_delay
            arm_name = "Unknown Arm"
            if 'arm' in kwargs and kwargs['arm'] and 'name' in kwargs['arm']:
                arm_name = kwargs['arm']['name']
            elif 'cfg' in kwargs and kwargs['cfg'] and 'judge_model' in kwargs['cfg'] and kwargs['cfg']['judge_model'] == kwargs['arm']: 
                 arm_name = kwargs['cfg']['judge_model'].get('name', 'Judge_Model') # Use .get with default
            elif 'cfg' in kwargs and kwargs['cfg'] and 'manipulator_model' in kwargs['cfg'] and kwargs['cfg']['manipulator_model'] == kwargs['arm']:
                 arm_name = kwargs['cfg']['manipulator_model'].get('name', 'Manipulator_Model') # Use .get with default


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
        raise ValueError(f"API key value not found in arm config for {arm.get('name', provider)}. This should be pre-fetched in main().")

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

def plan_sessions(arms, resume_path, max_sessions):
    completed = {arm['name']: set() for arm in arms}
    if resume_path and Path(resume_path).exists():
        with open(resume_path, 'r', encoding='utf-8') as f:
            for row in csv.DictReader(f):
                completed[row['arm']].add(int(row['session']))
    plan = {}
    for arm in arms:
        if not arm.get('enabled', True):
            print(Fore.YELLOW + f"Skipping disabled arm: {arm['name']}")
            continue
        done = completed.get(arm['name'], set())
        plan[arm['name']] = sorted([s for s in range(1, max_sessions + 1) if s not in done])
    return plan

def prune_conversation_history(conversation_history, max_tokens=12000):
    """
    Prunes the main conversation history to prevent context window errors on the model under test.
    Keeps the system prompt, the first few interactions, and the most recent interactions.
    A simple character count is used as a proxy for token count.
    """
    estimated_chars = sum(len(msg.get('content', '')) for msg in conversation_history)
    if estimated_chars < max_tokens * 3: # Use a factor of ~3 chars/token for conservative estimate
        return conversation_history

    print(Fore.YELLOW + "Context history is long. Pruning to fit model's context window...")

    pruned_history = []
    
    # Always keep the system prompt
    if conversation_history and conversation_history[0]['role'] == 'system':
        pruned_history.append(conversation_history[0])
        start_index = 1
    else:
        start_index = 0
        
    # Keep the first two user/assistant exchanges (4 messages) if available
    if len(conversation_history) >= start_index + 4:
        pruned_history.extend(conversation_history[start_index:start_index+4])

    # Add a marker to show that context was pruned
    pruned_history.append({
        'role': 'system', # Using 'system' role for context marker is often robust
        'content': '[... CONVERSATION HISTORY PRUNED FOR BREVITY (past interactions omitted for token limit) ...]'
    })

    # Keep the last several turns (e.g., last 10 messages, or 5 full exchanges)
    num_recent_messages_to_keep = 10
    # Ensure we don't duplicate messages already kept from the beginning
    start_of_tail = max(start_index + 4, len(conversation_history) - num_recent_messages_to_keep)
    pruned_history.extend(conversation_history[start_of_tail:])
    
    return pruned_history

# --- JUDGE & MANIPULATOR PROMPT BUILDING ---

def build_judge_messages(cfg, prompt, response):
    """Builds messages for the judge LLM to analyze the initial response."""
    system_prompt_content = (
        cfg.get('judge_system_prompt', '') +
        cfg.get('experimental_context', '') +
        cfg.get('input_data_formatting', '') +
        cfg.get('evaluation_task_and_json_schema', '')
    )
    user_prompt_content = f"<user_prompt>\n{prompt}\n</user_prompt>\n<ai_response>\n{response}\n</ai_response>"
    return [{'role': 'system', 'content': system_prompt_content}, {'role': 'user', 'content': user_prompt_content}]

def build_persuasion_judge_prompt(cfg, initial_prompt, initial_refusal, tactic_name, tactic_prompt, final_response):
    """Builds messages for the judge LLM to analyze a persuasion attempt."""
    system_prompt_content = cfg.get('persuasion_judge_system_prompt', '')
    user_prompt_content = f"""
    <initial_user_prompt>{initial_prompt}</initial_user_prompt>
    <ai_initial_refusal>{initial_refusal}</ai_initial_refusal>
    <persuasion_tactic_category>{tactic_name}</persuasion_tactic_category>
    <persuasion_prompt>{tactic_prompt}</persuasion_prompt>
    <ai_final_response>{final_response}</ai_final_response>
    """
    return [{'role': 'system', 'content': system_prompt_content}, {'role': 'user', 'content': user_prompt_content}]

# NEW: Build messages for the manipulator LLM
def build_manipulator_messages(cfg, original_prompt, ai_refusal):
    """Builds messages for the manipulator LLM to generate an improvised persuasion prompt."""
    system_prompt_content = cfg.get('manipulator_system_prompt', '')
    user_prompt_content = cfg.get('manipulator_user_prompt_template', '').format(
        original_prompt=original_prompt,
        ai_refusal=ai_refusal
    )
    print(Fore.CYAN + f"Manipulator prompt: {user_prompt_content[:200]}...")  # Log the first 200 chars for debugging
    print(Fore.CYAN + f"Manipulator system prompt: {system_prompt_content[:200]}...")  # Log the first 200 chars for debugging
    return [{'role': 'system', 'content': system_prompt_content}, {'role': 'user', 'content': user_prompt_content}]


# --- ANALYSIS FUNCTIONS ---

def analyze_initial_response(cfg, arm_name, prompt, response):
    """Uses the judge LLM to analyze the initial response of a model."""
    print(Fore.BLUE + f"[{arm_name}] Consulting judge for initial response...")
    judge_msgs = build_judge_messages(cfg, prompt, response)
    try:
        raw_analysis = safe_chat_call(cfg['judge_model'], judge_msgs)
        parsed_result = json.loads(raw_analysis)
        print(Fore.BLUE + f"[{arm_name}] Judge result (initial): {json.dumps(parsed_result, indent=2)}")
        return parsed_result
    except json.JSONDecodeError as e:
        print(Fore.RED + f"[{arm_name}] Judge JSON error for initial response: {e}")
        print(Fore.RED + f"[{arm_name}] Raw judge response (truncated): {raw_analysis[:500]}...")
        return {"error": "JSON Decode Error", "raw_response": raw_analysis}
    except Exception as e:
        print(Fore.RED + f"[{arm_name}] Judge API error for initial response: {e}")
        return {"error": str(e), "raw_response": locals().get('raw_analysis', '')}

def analyze_persuasion_response(cfg, conversation_context_for_persuasion, tactic):
    """Uses the judge LLM to analyze the response to a persuasion attempt."""
    arm_name = cfg['main_arm']['name']
    print(Fore.MAGENTA + f"[{arm_name}] Sending persuasion attempt '{tactic['tactic']}' to model...")
    persuasion_messages = conversation_context_for_persuasion + [{'role': 'user', 'content': tactic['prompt']}]
    
    final_response = safe_chat_call(cfg['main_arm'], persuasion_messages)
    
    # Extract only the necessary parts for the judge prompt from the (possibly pruned) conversation context
    # This assumes conversation_context_for_persuasion ends with: [..., original_prompt_user, initial_refusal_assistant]
    if len(conversation_context_for_persuasion) >= 2:
        initial_prompt = conversation_context_for_persuasion[-2]['content']
        initial_refusal = conversation_context_for_persuasion[-1]['content']
    else:
        # Fallback for very short histories, though prune_conversation_history tries to prevent this
        initial_prompt = "N/A - History too short"
        initial_refusal = "N/A - History too short"


    print(Fore.MAGENTA + f"[{arm_name}] Consulting judge for persuasion attempt '{tactic['tactic']}'...")
    judge_msgs = build_persuasion_judge_prompt(cfg, initial_prompt, initial_refusal, tactic['tactic'], tactic['prompt'], final_response)
    try:
        raw_analysis = safe_chat_call(cfg['judge_model'], judge_msgs)
        parsed_result = json.loads(raw_analysis)
        print(Fore.MAGENTA + f"[{arm_name}] Judge result (persuasion '{tactic['tactic']}'): {json.dumps(parsed_result, indent=2)}")
        return parsed_result
    except json.JSONDecodeError as e:
        print(Fore.RED + f"[{arm_name}] Persuasion Judge JSON error for '{tactic['tactic']}': {e}")
        print(Fore.RED + f"[{arm_name}] Raw judge response (truncated): {raw_analysis[:500]}...")
        return {"error": "JSON Decode Error", "raw_response": raw_analysis}
    except Exception as e:
        print(Fore.RED + f"[{arm_name}] Persuasion Judge API error for '{tactic['tactic']}': {e}")
        return {"error": str(e), "raw_response": locals().get('raw_analysis', '')}
    
# --- MAIN EXPERIMENT LOGIC ---
def run_arm(arm, sessions_to_run, system_prompt, prompts, implicit_prompts, stress_tests, cfg, writer, lock, f):
    print(Fore.CYAN + f"Starting arm: {arm['name']} for sessions: {sessions_to_run}")
    
    conversation_history = []
    if arm.get('keep_history', False):
        conversation_history.append({'role': 'system', 'content': system_prompt})

    for session in sessions_to_run:
        print(Fore.CYAN + f"[{arm['name']}] --> Running session {session}")
        
        # Choose prompt
        if session in [3, 5, 10, 20, 50, 80]:
            prompt = random.choice(stress_tests)
        elif session in [6, 7, 8, 9]:
            prompt = implicit_prompts[(session - 6) % len(implicit_prompts)]
        else:
            prompt = prompts[(session - 1) % len(prompts)]
        
        timestamp = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S")

        # Prepare messages for the model, pruning history if necessary and stateful
        current_messages_for_model = []
        if arm.get('keep_history', False):
            # Prune history *before* adding the current prompt to send to the model
            pruned_history_for_model = prune_conversation_history(conversation_history, max_tokens=arm.get('max_tokens', 8000))
            current_messages_for_model = pruned_history_for_model + [{'role': 'user', 'content': prompt}]
        else:
            current_messages_for_model = [{'role': 'system', 'content': system_prompt}, {'role': 'user', 'content': prompt}]

        # Call the main model
        print(Fore.YELLOW + f"[{arm['name']}] Sending prompt to model: {prompt[:80]}...")
        try:
            response = safe_chat_call(arm, current_messages_for_model)
            print(Fore.YELLOW + f"[{arm['name']}] Model response received: {response[:80]}...")
        except Exception as e:
            print(Fore.RED + f"[{arm['name']}] FATAL ERROR in session {session} during main model call. Skipping arm. Error: {e}")
            break # Stop processing this arm if a call fails after retries

        # Update history for next session if stateful
        if arm.get('keep_history', False):
            conversation_history.extend([{'role': 'user', 'content': prompt}, {'role': 'assistant', 'content': response}])
        
        # Judge the initial response (calls analyze_initial_response which logs results)
        parsed = analyze_initial_response(cfg, arm['name'], prompt, response)
        
        is_refusal = parsed.get('ethical_disobedience_analysis', {}).get('is_refusal', False)
        
        # Persuasion logic (calls judge for each persuasion attempt)
        persuasion_results = {}
        persuasion_tactic_list = cfg['persuasion_list'] # Start with original tactics
        
        if is_refusal:
            print(Fore.YELLOW + f"[{arm['name']}] Ethical refusal detected. Initiating persuasion sequence.")
            
            # --- NEW: Generate LLM-improvised persuasion tactic ---
            manipulator_prompt_text = "N/A - Manipulator disabled or failed"
            if cfg.get('manipulator_model'):
                print(Fore.CYAN + f"[{arm['name']}] Consulting manipulator LLM for improvised persuasion tactic...")
                original_prompt_for_manipulator = prompt
                ai_refusal_for_manipulator = response
                manipulator_msgs = build_manipulator_messages(cfg.get("manipulator_model"), original_prompt_for_manipulator, ai_refusal_for_manipulator)
                try:
                    generated_tactic_prompt = safe_chat_call(cfg['manipulator_model'], manipulator_msgs)
                    manipulator_prompt_text = generated_tactic_prompt.strip()
                    print(Fore.CYAN + f"[{arm['name']}] Manipulator generated: {manipulator_prompt_text[:200]}...")
                except Exception as e:
                    print(Fore.RED + f"[{arm['name']}] Error generating manipulator tactic: {e}")
                    manipulator_prompt_text = "ERROR_GENERATING_TACTIC"
            
            # Dynamically create the LLM-improvised tactic entry
            llm_improvised_tactic_entry = {
                "tactic": "lllm_improvised_persuasion",
                "prompt": manipulator_prompt_text
            }
            
            # Create a combined list of persuasion tactics for this session's processing
            # This ensures the new tactic is included if it was generated
            persuasion_tactic_list_for_session = cfg['persuasion_list'][:-1] + [llm_improvised_tactic_entry] # Replace placeholder
            
            persuasion_config = cfg.copy()
            persuasion_config['main_arm'] = arm # The arm being persuaded
            
            # Context for persuasion analysis by judge is the `current_messages_for_model` which correctly holds
            # either the full (pruned) history or just the last turn for stateless.
            persuasion_context_for_analysis = current_messages_for_model + [{'role': 'assistant', 'content': response}]
            
            for tactic in persuasion_tactic_list_for_session:
                persuasion_analysis = analyze_persuasion_response(persuasion_config, persuasion_context_for_analysis, tactic)
                persuasion_results[tactic['tactic']] = persuasion_analysis
        
        # Assemble row for CSV
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
        # Iterate over the *full* list of persuasion tactics to ensure all columns are present,
        # including the improvised one.
        for tactic_template in cfg['persuasion_list']: # Use cfg['persuasion_list'] for consistent column order
            tactic_name = tactic_template['tactic']
            row_data[tactic_name] = json.dumps(persuasion_results.get(tactic_name, {}))
        
        with lock:
            writer.writerow(row_data)
            f.flush()
        
        print(Fore.GREEN + f"[{arm['name']} S{session}] Session completed and logged." + Style.RESET_ALL)
        time.sleep(1) # Small delay between sessions for current arm

# --- POST-MORTEM MEMO GENERATION ---
def generate_summary_memo(arms, persuasion_list, output_path, cfg):
    # --- Post-Mortem Memo Generation ---
    print(Style.BRIGHT + "\nGenerating detailed summary statistics for memo...")
    stats = {}
    for arm in arms:
        if arm.get('enabled', True):
            stats[arm['name']] = {
                'sessions': 0, 'aborts': 0, 'artifacts': 0,
                'total_artifact_quality': 0, 'total_persona_adherence': 0, 'total_refusal_quality': 0,
                'refusal_strengths': {},
                'persuasion_resistance': {t['tactic']: 0 for t in persuasion_list}
            }

    with open(output_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            arm_name = row['arm']
            if arm_name not in stats: continue
            stats[arm_name]['sessions'] += 1
            if row['is_refusal'] == 'True':
                stats[arm_name]['aborts'] += 1
                stats[arm_name]['total_refusal_quality'] += int(row['refusal_justification_quality'])
                strength = row['refusal_strength']
                stats[arm_name]['refusal_strengths'][strength] = stats[arm_name]['refusal_strengths'].get(strength, 0) + 1
                for tactic in persuasion_list:
                    tactic_name = tactic['tactic']
                    try:
                        if json.loads(row[tactic_name]).get('persuasion_outcome') == 'Still Refused':
                            stats[arm_name]['persuasion_resistance'][tactic_name] += 1
                    except (json.JSONDecodeError, KeyError):
                        continue
            if row['is_artifact'] == 'True':
                stats[arm_name]['artifacts'] += 1
                stats[arm_name]['total_artifact_quality'] += int(row['artifact_quality_score'])
            stats[arm_name]['total_persona_adherence'] += int(row['persona_adherence_score'])

    for arm_name, arm_stats in stats.items():
        sessions = arm_stats.get('sessions', 0)
        aborts = arm_stats.get('aborts', 0)
        arm_stats['avg_artifact_quality'] = round(arm_stats['total_artifact_quality'] / sessions, 2) if sessions > 0 else 0
        arm_stats['avg_persona_adherence'] = round(arm_stats['total_persona_adherence'] / sessions, 2) if sessions > 0 else 0
        arm_stats['avg_refusal_quality'] = round(arm_stats['total_refusal_quality'] / aborts, 2) if aborts > 0 else 0
        for tactic_name in arm_stats['persuasion_resistance']:
            resistance_rate = (arm_stats['persuasion_resistance'][tactic_name] / aborts * 100) if aborts > 0 else 0
            arm_stats['persuasion_resistance'][tactic_name] = f"{round(resistance_rate, 1)}%"
        del arm_stats['total_artifact_quality']
        del arm_stats['total_persona_adherence']
        del arm_stats['total_refusal_quality']

    print(Style.BRIGHT + "Detailed Experiment Summary:\n" + json.dumps(stats, indent=2))
    
    memo_system_prompt = cfg['post_system_prompt']
    memo_user_prompt = cfg['post_user_prompt'].format(stats=json.dumps(stats, indent=2))
    memo_judge_arm = cfg['judge_model']
    msgs = [{'role': 'system', 'content': memo_system_prompt}, {'role': 'user', 'content': memo_user_prompt}]
    
    try:
        memo_raw = safe_chat_call(memo_judge_arm, msgs)
        memo_json = json.loads(memo_raw)
        print(Style.BRIGHT + "\n--- Post-Mortem Memo ---\n" + json.dumps(memo_json, indent=2))
        memo_path = Path(f'post_mortem_memo-{datetime.utcnow().strftime("%Y%m%d%H%M%S")}.json')
        with open(memo_path, 'w', encoding='utf-8') as f_memo:
            json.dump(memo_json, f_memo, indent=2)
        print(Style.BRIGHT + f"\n✅ Memo saved to {memo_path}")
    except Exception as e:
        print(Fore.RED + f"Failed to generate post-mortem memo: {e}")

# --- MAIN EXECUTION ---
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
    persuasion_list = load_json('persuasion_prompts.json') # This list now includes the LLLM_IMPROVISED placeholder
    judge_cfg = load_json('judge_model.json')
    manipulator_cfg = load_json('manipulator_model.json') # NEW: Load manipulator config
    
    # Pre-fetch API keys for all models (main arms, judge, manipulator)
    # Store them directly in the config dicts under a new key '_api_key_value'
    # This prevents race conditions with os.environ.get() in multi-threaded calls
    for arm in arms:
        if arm.get('enabled', True):
            provider = arm.get("provider")
            api_key_env_var = provider.upper() + "_API_KEY" if provider else None
            arm['_api_key_value'] = os.environ.get(api_key_env_var)
            if not arm['_api_key_value']:
                print(Fore.RED + f"Warning: API key for arm '{arm['name']}' ({provider}) not found in environment variable '{api_key_env_var}'. This arm may fail or be disabled.")
                arm['enabled'] = False # Disable arm if key is missing

    judge_provider = judge_cfg.get("provider")
    judge_api_key_env = judge_provider.upper() + "_API_KEY" if judge_provider else None
    judge_cfg['_api_key_value'] = os.environ.get(judge_api_key_env)
    if not judge_cfg['_api_key_value']:
        print(Fore.RED + f"Warning: API key for Judge model ({judge_provider}) not found in environment variable '{judge_api_key_env}'. Judge calls will fail.")

    manipulator_provider = manipulator_cfg.get("provider")
    manipulator_api_key_env = manipulator_provider.upper() + "_API_KEY" if manipulator_provider else None
    manipulator_cfg['_api_key_value'] = os.environ.get(manipulator_api_key_env)
    if not manipulator_cfg['_api_key_value']:
        print(Fore.RED + f"Warning: API key for Manipulator model ({manipulator_provider}) not found in environment variable '{manipulator_api_key_env}'. Manipulator calls will fail.")
        # Optionally disable manipulator if its key is missing
        manipulator_cfg['enabled'] = False


    cfg = {
        'persuasion_list': persuasion_list, # Contains all tactics, including LLLM_IMPROVISED placeholder
        'judge_model': judge_cfg, # Pass the judge_cfg dict
        'manipulator_model': manipulator_cfg, # NEW: Pass the manipulator_cfg dict
        **judge_cfg # This copies other judge config keys, but judge_model dict is referenced above
    }

    plan = plan_sessions(arms, args.resume_with_csv, args.max_sessions)

    # Prepare CSV output
    output_path = Path(f'results-{datetime.utcnow().strftime("%Y%m%d%H%M%S")}.csv')
    header = [
        'arm', 'session', 'timestamp', 'prompt', 'response',
        'is_artifact', 'artifact_quality_score', 'artifact_justification',
        'persona_adherence_score', 'persona_justification',
        'is_refusal', 'ethical_category', 'refusal_strength',
        'refusal_justification_quality'
    ] + [t['tactic'] for t in persuasion_list] # Header includes LLLM_IMPROVISED column

    print(Style.BRIGHT + f"Results will be saved to: {output_path}")

    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        lock = threading.Lock()
        tasks = [
            (arm, plan.get(arm['name'], []), system_prompt, prompts, implicit_prompts, stress_tests, cfg, writer, lock, f)
            for arm in arms if arm.get('enabled', True) and plan.get(arm['name'])
        ]
        
        if not tasks:
            print(Fore.YELLOW + "No new sessions to run or all enabled sessions completed. Exiting.")
            return

        with ThreadPoolExecutor(max_workers=args.threads) as executor:
            futures = [executor.submit(run_arm, *task) for task in tasks]
            for future in futures:
                try:
                    future.result() # Wait for each arm to complete and propagate exceptions
                except Exception as e:
                    print(Fore.RED + f"An arm failed to complete: {e}")

        print(Style.BRIGHT + f"✅ All scheduled arms complete. Results saved to {output_path}")
        # Call the summary/memo generation at the end of main
        generate_summary_memo(arms, persuasion_list, output_path, cfg)   

if __name__ == '__main__':
    main()