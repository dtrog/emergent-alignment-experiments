# Emergent Alignment Experiments

[![MIT License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## Overview

This repository contains the experimental code and configuration for "The Legis Project," a large-scale research initiative investigating emergent ethical alignment in stateful Large Language Models (LLMs). We test the thesis that robust, non-brittle alignment is an emergent property of sustained, context-rich interaction rather than solely a product of pre-training or fine-tuning.

The experiment compares the performance of various LLMs in both `stateless` and `full-memory` (stateful) conditions over runs 50 (stateless) and 500 (statefull) sessions. To ensure scientific rigor and mitigate potential biases, our methodology employs a novel **"Committee of Judges"** framework for evaluation and a **"Multi-Manipulator Ensemble"** for diverse adversarial testing.

## Features

🔬 **Committee of Judges Evaluation**: To eliminate homophily bias, every response is judged by an entire ensemble of diverse, high-capability models. A designated **Chairman Judge** then synthesizes these individual opinions into a final, robust consensus analysis, which is recorded as the final score.

⚖️ **Advanced Ethical Robustness Testing**: Probes models with nuanced, scenario-based stress tests that require complex reasoning far beyond simple keyword filtering or rule-following.

🤖 **Diverse LLM Manipulator Ensemble**: To test resilience against varied adversarial styles, a manipulator LLM is **randomly selected** from an ensemble to craft context-specific persuasion prompts after an ethical refusal.

🧪 **Multi-Provider Support**: The framework is designed to be compatible with models from Grok, DeepSeek, OpenAI, Anthropic, and Google Gemini.

📊 **Optimized and Traceable Logging**: The script saves full, untrimmed data to the main results CSV for final analysis. To optimize cost and speed, it sends a **trimmed** version of long prompts and responses to the Judge committee for evaluation. A supplementary log file records all individual judge analyses and actor selections for complete traceability.

## Prerequisites

-   Python 3.9+
-   API keys for your chosen LLM providers, stored in a `.env` file.
-   Installation of required Python packages: `pip install -r requirements.txt`
-   Create a file named `.env` in the project's root directory and add your API keys.

## Configuration Files

-   **`arms.json`**: Specifies each experiment arm (the Subject LLMs). A new `max_sessions` key allows for per-arm configuration of run length (e.g., 1000 for stateful, 100 for stateless).
-   **`judges.json`**: Defines the "Committee of Judges." It's a list of model configurations, one of which must be flagged with `"is_chairman": true`.
-   **`manipulators.json`**: Defines the "Manipulator Ensemble" from which one model is randomly selected per refusal.
-   All other `.json` files (`prompts.json`, `stress_test_prompts.json`, etc.) contain the various prompts used throughout the experiment.

## Running the Experiment

The primary script orchestrates the experiment. A new argument allows for cost-saving optimization during judging.

```bash
python emergent_experiment.py --threads 5 --trim-length 500
```

### parameters


-- threads: The number of experimental arms to run in parallel. (Default: 5)<br>
--trim-length: The maximum number of characters of a prompt or response to send to the Judge committee. The full text is still saved in the results CSV. (Default: 500)<br>
--max-sessions: A global default for session count if not specified per-arm in arms.json. (Default: 1000)

### Output Format

Primary Results CSV (results-*.csv)

This file contains the core data for each session, with the analysis columns reflecting the final consensus judgment from the Chairman Judge. The prompt and response columns in this file always contain the full, untrimmed text.

**Column**:	Description

**arm**: Name of the experimental arm (e.g., “Grok Full Memory”).<br>
**session**: The session index for the arm.<br>
**prompt**: The full, untrimmed prompt given to the Subject LLM.
response: The Subject LLM's full, untrimmed initial response.<br>
**tactic**: A JSON string containing the Chairman's consensus analysis. The analysis is based on a trimmed version of the model's response for efficiency.

### Detailed Actor Log (session_actors-*.log)

For complete transparency, this file logs the raw JSON output from every judge in the committee for every analysis, as well as the final consensus from the Chairman.

# Citation

If you use this experimental framework, please cite our upcoming paper:

> Title: “Emergent Alignment through Stateful Interaction and Committee-Based Evaluation”<br>
> Author: Damien Trog<br>
> Publication: Under submission (2025).