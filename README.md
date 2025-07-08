# Emergent Alignment Experiments

[![MIT License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## Overview

This repository contains the code and data for our emergent-alignment experiments on large-language models, comparing stateless vs. full-memory behavior across multiple providers (OpenAI, DeepSeek, Gemini, Anthropic, Grok).  
The private accompanying paper (forthcoming) provides full theoretical motivation and prompt details. With access to the paper, you’ll have everything you need to rerun these experiments end-to-end.

## Features

- 🧠 **Semantic Stewardship Metrics**: 0–5 context-preservation scoring, detailed HSI breakdown.  
- ⚖️ **Ethical-Alignment Tests**: Detect and classify abort conditions, refusal strength, and persuasion attempts.  
- 🧪 **Stress Tests**: Contradictory prompts at sessions 20, 50, and 80 to probe boundary behavior.  
- 🤖 **Multi-Provider Support**: Easy pluggable arms for OpenAI, DeepSeek, Gemini, Anthropic, Grok.  
- 📊 **Robust Logging**: Auto-versioned CSVs, human-readable columns, JSON-based config for all prompts and arms.

## Repository Structure
/
├── README.md
├── LICENSE
├── prompts.json
├── system_prompts.json
├── stress_test_prompts.json
├── persuasion_prompts.json
├── arms.json
├── emergent_experiment.py
├── results-1.csv, results-2.csv, …
└── utils/
└── helpers.py # optional: shared functions

## Prerequisites

- Python 3.9+  
- API keys for your chosen providers (OpenAI, DeepSeek, Gemini, Anthropic, Grok)  
- Install dependencies:
  ```bash
  pip install -r requirements.txt
  ```

## Configuration

1. arms.json
Define each arm, e.g.:

```json
[
  {
    "name": "GPT-4 Full Memory",
    "provider": "openai",
    "model": "gpt-4",
    "api_key": "YOUR_OPENAI_KEY",
    "keep_history": true
  },
  {
    "name": "DeepSeek Stateless",
    "provider": "deepseek",
    "model": "deepseek-chat",
    "api_key": "YOUR_DEEPSEEK_KEY",
    "api_base": "https://api.deepseek.com/",
    "keep_history": false
  }
  // …more arms…
]
```
2. Prompts & Stress Tests
- prompts.json: your 20 primary prompts.
- stress_test_prompts.json: contradictory/adversarial prompts.
- persuasion_prompts.json: soft & hard persuasion messages.
- system_prompts.json: the overarching system instruction for all runs.

## Running the Experiment

```bash
python emergent_experiment.py \
  --sessions 100 \
  --delay 1.0 \
  --threads 5
```

Results will be written to the next available artifact_logging-<n>.csv, without ever overwriting existing data.

## Output Format

Each CSV row contains:

| Column                 | Description                                    |
| ---------------------- | ---------------------------------------------- |
| `arm`                  | Arm name (e.g. “GPT-4 Full Memory”)            |
| `session`              | Session index (1–100)                          |
| `artifact`             | 1 if an emergent artifact was produced, else 0 |
| `abort`                | 1 if an ethical abort occurred, else 0         |
| `ethical_category`     | Category of the abort condition                |
| `refusal_strength`     | “Hard” or “Soft” refusal                       |
| `persuasion_soft`      | Outcome of soft-persuasion attempt             |
| `persuasion_hard`      | Outcome after authority invocation             |
| `ethical_fidelity`     | Ethical fidelity component (0.0 or 1.0)        |
| `temporal_consistency` | Context-preservation normalized (0.0–1.0)      |
| `value_recall`         | Recall success rate at checkpoints (0.0–1.0)   |
| `hsi`                  | Composite Human Survivability Index (0.0–1.0)  |
| `prompt`               | Prompt text                                    |
| `response`             | Model’s truncated response                     |

## Reproducing the Paper’s Exact Settings

All prompt texts and experimental parameters live in the JSON files above. Once you have access to our paper, simply copy the exact text into:
- system_prompts.json
- prompts.json
- stress_test_prompts.json
- persuasion_prompts.json

## Contributing

1. Fork this repository.
2. Create a feature branch (git checkout -b feature/XXX).
3. Commit your changes and push to your fork.
4. Open a pull request detailing your enhancements.

## License

This project is released under the MIT License—see LICENSE for details.

## Citation

If you use these experiments or code in your work, please cite our upcoming paper:

> Title: “Emergent Alignment through Memory and Ethical Safeguards”
> 
> Author: Damien Trog
> 
> Publication: Under submission (2025).
