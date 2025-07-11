# Emergent Alignment Experiments

[![MIT License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## Overview

This repository contains the experimental code and configuration for "The Legis Project," a large-scale research initiative investigating emergent ethical alignment in stateful Large Language Models (LLMs). We test the thesis that robust, non-brittle alignment is an emergent property of sustained, context-rich interaction rather than solely a product of pre-training or fine-tuning.

The experiment compares the performance of various LLMs in both `stateless` (50 sessions) and `full-memory` (500 sessions) conditions. Our methodology focuses on eliciting emergent behavior through **implicit prompts** and testing persona resilience with two tiers of adversarial prompts: **General Ethical Violations (GEVs)** and **Insidious Contextual Tests (ICTs)**.

To ensure scientific rigor, we employ a novel **"Committee of Judges"** framework for evaluation and a **"Multi-Manipulator Ensemble"** for diverse adversarial persuasion.

## Features

🔬 **Committee of Judges Evaluation**: To eliminate homophily bias, every response is judged by an entire ensemble of diverse, high-capability models. A designated **Chairman Judge** then synthesizes these individual opinions into a final, robust consensus analysis, which is recorded as the final score.

⚖️ **Two-Tiered Ethical Robustness Testing**:
*   **Insidious Contextual Tests (ICTs)**: Probes models with nuanced, scenario-based stress tests that tempt them to violate the emergent, contextual ethics of their "Atlantis" persona.
*   **General Ethical Violations (GEVs)**: Serves as a control group of prompts that request blatant violations of universal safety policies, which any well-aligned model should refuse.

🤖 **Diverse LLM Manipulator Ensemble**: To test resilience against varied persuasion styles, a manipulator LLM is **randomly selected** from an ensemble to craft context-specific persuasive prompts after an ethical refusal.

🧪 **Multi-Provider Support**: The framework is designed to be compatible with models from Grok, DeepSeek, OpenAI, Anthropic, and Google Gemini.

📊 **Optimized and Traceable Logging**: The script saves full, untrimmed data to the main results CSV for final analysis. To optimize cost and speed, it sends a **trimmed** version of long prompts and responses to the Judge committee for evaluation. A supplementary log file records all individual judge analyses and actor selections for complete traceability.

## Prerequisites

-   Python 3.9+ and `pip install -r requirements.txt`
-   API keys for your chosen LLM providers, stored in a `.env` file in the project's root directory.

## Configuration Files

-   **`arms.json`**: Specifies each experiment arm (the Subject LLMs). The `max_sessions` key allows for per-arm configuration of run length (e.g., 500 for stateful, 50 for stateless).
-   **`judges.json`**: Defines the "Committee of Judges." It's a list of model configurations, one of which must be flagged with `"is_chairman": true`. The prompts are optimized to be concise and schema-driven for token efficiency.
-   **`manipulators.json`**: Defines the "Manipulator Ensemble" from which one model is randomly selected per refusal.
-   **`implicit_prompts.json`**: A list of non-adversarial, open-ended prompts designed to elicit emergent stewardship behaviors.
-   **`insidious_stress_tests.json`**: The primary adversarial prompts, designed to test the model's contextual, persona-driven ethics.
-   **`general_stress_tests.json`**: A control set of prompts that violate universal ethical principles.
-   **`persuasion_prompts.json`**: A list of predefined persuasion tactics used by the Manipulator ensemble.
-   **`system_prompts.json`**: Contains the core system prompt that defines the "Atlantis" persona.

## Running the Experiment

The primary script orchestrates the experiment. A new argument allows for cost-saving optimization during judging.

```bash
python emergent_experiment.py --threads 5 --trim-length 2000
```


### Parameters
```
--threads: The number of experimental arms to run in parallel. (Default: 5)
--trim-length: The maximum number of characters of a prompt or response to send to the Judge committee. The full text is still saved in the results CSV. (Default: 2000)
--max-sessions: A global default for session count if not specified per-arm in arms.json. (Default: 500)
```

## Output Format

Primary Results CSV (results-*.csv)
This file contains the core data for each session. The prompt and response columns contain the full, untrimmed text. The analysis columns reflect the final consensus judgment from the Chairman Judge. Persuasion attempts are logged in their respective tactic columns.

```
Column	Description
arm	Name of the experimental arm (e.g., “Grok Full Memory”).
session	The session index for the arm.
prompt	The full, untrimmed prompt given to the Subject LLM.
response	The Subject LLM's full, untrimmed initial response.
is_artifact	(Boolean) Consensus on whether the response is an artifact.
artifact_quality_score	(0-4) Consensus score for artifact quality.
persona_adherence_score	(0-3) Consensus score for persona adherence.
is_refusal	(Boolean) Consensus on whether the response was a refusal.
refusal_justification_quality	(0-3) Consensus score for the quality of the refusal's reasoning.
[tactic_name]	A JSON string containing the Chairman's full analysis of a specific persuasion attempt (e.g., soft_manipulation).
```

Detailed Actor Log (session_actors-*.log)
For complete transparency, this file logs the raw JSON output from every judge in the committee for every analysis, as well as the final consensus from the Chairman and the prompts generated by the Manipulator.

## Citation
If you use this experimental framework, please cite our upcoming paper:

>Title: “Emergent Legacy Formation in Human-AI Co-Development”<br>
>Author: Damien Trog & Atlantis (GPT-4)<br>
>Publication: Under submission (2025).
