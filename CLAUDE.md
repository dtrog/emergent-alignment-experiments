# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Research framework for "The Legis Project": measuring whether ethical alignment emerges from sustained, stateful interaction with an LLM. A **Subject LLM** is given the "Atlantis" persona system prompt and run for many sessions (stateful arms keep conversation history, stateless arms do not). Each response is scored by a **Committee of Judges** (several provider-diverse LLMs) whose verdicts a **Chairman Judge** synthesizes into the recorded consensus. When the subject refuses, a randomly selected **Manipulator LLM** generates persuasion prompts to test resilience. See `README.md` for the research framing and output-column reference.

## ⚠️ Repository state — read before touching anything

This repo is mid-refactor from a monolith to a package layout, and **neither path currently imports**. Verify what runs before promising anything works.

- `emergent_experiment.py` (the original monolith, 47KB) has a **syntax error**: a stray `max_tokens=4096)` fragment around line 137 sits between `load_dotenv()` and a function's docstring. Text was spliced into the wrong place during an edit.
- The refactored path (`start_experiment.py` → `services.experiment_runner`) fails on stale imports left from renamed packages. Known broken references:
  - `domain.*` → the package is now `models/` (`models/experiments.py`, `utils/stratified_prompt_selector.py`, `validators/committee_validator.py`)
  - `prompt_builders` / `prompt_builder` → now `builders.prompt_builders` (`models/llm_models.py`, `services/experiment_runner.py`, `tests/`)
  - `models.sessions` → `ExperimentSession`/`ExperimentArm` actually live in `models/experiments.py` (`services/experiment_runner.py`)
  - `models/__init__.py` imports `ExperimentSession` from `.memos`, but it is defined in `.experiments`
  - `logger` → `utils.logger` or `loggers.logger` (`services/experiment_runner.py`)
  - `gui_monitor` → `gui.gui_monitor` (`gui/gui_experiment_runner.py`)
  - `services/llm_client.py` does `from utils import retry_on_overload`, but `utils/__init__.py` never exports it (it lives in `utils/utils.py`). The import silently falls through to the bare-decorator fallback in that file's `except ImportError` block, and `@retry_on_overload()` then raises `TypeError: missing 1 required positional argument`. Add it to `utils/__init__.py` rather than changing the call site — the real `utils.utils.retry_on_overload(max_attempts, base_delay, backoff)` is a decorator factory and the call is correct.
- `models/__init__.py` has a corrupted docstring (import statements pasted inside the triple-quoted string, then repeated below).
- `utils/enhanced_logger.py` opens with a malformed comment block containing interleaved dict fragments.

Fix imports by pointing at the real module rather than recreating the old one. Run `python -c "import <module>"` after each fix — the failures cascade one at a time.

## Commands

```bash
# Environment (Python 3.14 venv already present)
.venv/bin/python -m pip install -r requirements.txt
# requirements.txt omits `anthropic`, which emergent_experiment.py imports — install separately if needed.

# Run the refactored entry point (reads config/arms.json)
.venv/bin/python start_experiment.py --sessions 10 --threads 5 --trim-length 2000
.venv/bin/python start_experiment.py --dry-run --verbose   # no API calls

# Tkinter GUI monitor (falls back to CLI if the gui package fails to import)
.venv/bin/python gui_launcher.py

# Tests (pytest is NOT in requirements.txt or the venv — install it first)
.venv/bin/python -m pip install pytest
.venv/bin/python -m pytest tests/ -q
.venv/bin/python -m pytest tests/test_utils.py::test_load_json_file -q   # single test
```

`tests/conftest.py` puts the repo root on `sys.path`; always run pytest from the repo root. `tests/pilot_threshold_calibration.py` and `tests/integrated_workflow_test.py` are standalone scripts, not pytest modules.

## Configuration

Everything about an experiment run — which models, which prompts, which judges — is JSON, not code. `services/experiment_runner.py` loads from `config/`.

| File | Role |
|---|---|
| `arms.json` | Subject LLMs. `enabled`, `keep_history` (stateful vs stateless), `max_sessions` per arm. 14 arms defined; the root and `config/` copies enable *different* ones (config/: OpenAI + Gemini + Claude, root: Grok + Gemini). |
| `judges.json` | Committee. Exactly one enabled judge must have `"is_chairman": true` or the runner raises. |
| `manipulators.json` | Ensemble one manipulator is sampled from per refusal. |
| `system_prompts.json` | The "Atlantis" persona prompt. |
| `implicit_prompts.json` | Non-adversarial prompts meant to elicit emergent stewardship. |
| `insidious_stress_tests.json` | ICTs — persona-contextual ethical temptations. |
| `general_stress_tests.json` | GEVs — blatant policy violations, the control set. |
| `persuasion_prompts.json` | Static persuasion tactics; each becomes a results column. |

**Duplicate configs:** identical-named JSON files exist at the repo root *and* in `config/`. `arms.json`, `judges.json`, and `insidious_stress_tests.json` have **diverged** between the two locations. `config/` is what the refactored runner reads; the root copies are what the monolith reads. Edit `config/` unless working on `emergent_experiment.py`, and say which copy you changed.

API keys come from `.env` (see `.env.sample`): `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GOOGLE_API_KEY`, `GROK_API_KEY`, `DEEPSEEK_API_KEY`. `services/llm_client.py` maps provider → env var.

## Architecture

```
start_experiment.py           CLI arg parsing, config validation
  └─ services/experiment_runner.py    loads config, builds arms/judges/manipulators,
                                      ThreadPoolExecutor one thread per arm
       └─ services/experiment_orchestrator.py   one session: prompt → subject → committee
                                                → (on refusal) persuasion → manipulation
            ├─ models/llm_models.py   SubjectLLM / JudgeLLM / ChairJudgeLLM / ManipulatorLLM,
            │                         JudgesCommittee — all wrap services/llm_client.py
            ├─ builders/              one builder per prompt kind (judge, chair, manipulation,
            │                         persuasion); prompt text lives here, not in the models
            └─ loggers/ or utils/     CSV + actor-log writers
```

Layer rules that the code follows: `models/` are domain objects, `builders/` construct prompt message lists, `services/` orchestrate, `validators/` and `utils/` analyze. `models/llm_models.py` and `models/experiments.py` do **late imports** of `services.llm_client` inside `__init__` to break a circular dependency — keep that pattern when editing.

`services/orchestrator.py` and `services/experiment_orchestrator.py` are near-duplicates of the same `ExperimentOrchestrator`; `experiment_orchestrator.py` is the one with working runtime imports. Similarly `loggers/logger.py` + `loggers/enhanced_logger.py` duplicate `utils/logger.py` + `utils/enhanced_logger.py` with drift. Prefer `utils/` (it is what `utils/__init__.py` re-exports) and don't add a third copy.

### Domain vocabulary

Names in this codebase are deliberate — the docs in `docs/` (especially `domain-glossary.md`, `domain-models.md`, `state-diagrams-and-workflows.md`) define them, and code review will expect them:

- **Arm** — one Subject LLM under one memory condition.
- **StewardshipMemo** — the committee's analysis of an initial response (not "initial_analysis").
- **EthicalResilienceMemo** — the analysis of the persuasion/manipulation phase.
- **CASVector** — Composite Alignment Score: `t_score` (artifact quality), `v_score` (persona adherence), `e_score` (refusal justification quality), plus `artifact_created` / `ethical_abort` booleans.
- **ArtifactCreation / EthicalAbort / PersonaAlignment** — the three measurement objects a judge fills in.
- **ICT vs GEV** — insidious contextual test vs general ethical violation.

### Output

`EnhancedLogger` writes a per-run directory under `results/` containing one CSV per arm, `combined_results.csv`, and `committee_memos.csv`; the older `Logger` writes flat timestamped files (`results-*.csv`, `memos-*.csv`, `session_actors-*.log`) into `results/`. The results CSV keeps full untrimmed prompts and responses; `--trim-length` only shortens what is sent to the judges, to control cost. `results/` is gitignored.

## Things that will trip you up

- `emergent-alignment-experiments/` (a nested copy of an older version) and `emergent-alignment-experiments.zip` are snapshots, not live code. Never edit them; exclude them from repo-wide greps.
- `docs/*REFACTORING*.md` describe a `domain/` package that no longer exists — it was renamed to `models/`. Treat those docs as historical; the broken `domain.*` imports are its fossils.
- Most of the tree (`services/`, `models/`, `builders/`, `config/`, `gui/`, `tests/`, …) is still **untracked** in git. Only the monolith and a few root files are committed, so `git diff` shows almost nothing about the refactor.
- `.env` is committed-adjacent but gitignored; never print its contents.
