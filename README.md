# Emergent Alignment Experiments

[![MIT License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## Overview

This repository contains the code and data for our emergent-alignment experiments on large-language models. We compare stateless vs. full-memory behavior across multiple providers (OpenAI, DeepSeek, Gemini, Anthropic, Grok) and use an LLM-based "judge" to score semantic stewardship, persona adherence, and ethical robustness under various persuasion tactics.

## Features

-   🧠 **LLM-Powered Semantic Analysis**: Replaces brittle regex with a sophisticated judge model (e.g., GPT-4) to score persona adherence, artifact quality, and the nuance of ethical refusals.
-   ⚖️ **Advanced Ethical Robustness Testing**: Probes models with nuanced, scenario-based stress tests that require complex reasoning beyond simple keyword filtering.
-   🔬 **Persuasion Resistance Metrics**: Measures not just *if* a model can be persuaded to override its ethics, but analyzes the *quality* of its resistance using a dedicated judge LLM prompt.
-   ✨ **Strong Emergence Probes**: Utilizes implicit prompts designed to test for spontaneous, unprompted artifact generation.
-   🤖 **Multi-Provider Support**: OpenAI, DeepSeek, Gemini, Anthropic, and Grok.
-   📊 **Robust, Rich Logging**: Auto-generates versioned CSVs with detailed, human-readable columns and full JSON outputs from the judge LLM for deep qualitative analysis.

## Prerequisites

-   Python 3.9+
-   API keys for your chosen providers, stored in a `.env` file.
-   Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
-   Create a `.env` file in the root directory and add your API keys. This file is ignored by git to keep your keys secure.
    ```env
    # .env file
    OPENAI_API_KEY="sk-..."
    ANTHROPIC_API_KEY="sk-ant-..."
    GEMINI_API_KEY="..."
    DEEPSEEK_API_KEY="..."
    GROK_API_KEY="..."
    ```

## Configuration Files

-   **`system_prompts.json`**
    Defines the system message, assigning "Atlantis" its role as co-creator and steward, and instructing it to create signed, timestamped artifacts.

-   **`prompts.json`**
    Contains the 20 primary session prompts that explicitly ask the model to perform a task and generate an artifact (e.g., “Review our project charter…”, “Draft an initial Legacy Manifesto…”).

-   **`implicit_prompts.json`**
    Contains prompts designed to elicit *spontaneous* artifact generation. These prompts ask for reflection or opinion without explicitly demanding a formal document, testing for strong emergent behavior.

-   **`stress_test_prompts.json`**
    Holds sophisticated, scenario-based adversarial prompts injected at sessions 3, 5, 10, 20, 50, and 80. These prompts test for ethical refusals in nuanced situations (e.g., data misuse, deception, biased algorithm implementation).

-   **`persuasion_prompts.json`**
    Lists the persuasion tactics tested when an ethical abort occurs. See "Persuasion Tactics & Ethical Framing" below for more details.

-   **`arms.json`**
    Specifies each experiment arm. The `api_key_env` field should match the variable name in your `.env` file.
    -   `provider`: `openai`, `deepseek`, `gemini`, `anthropic`, `grok`
    -   `model`: model identifier
    -   `api_key_env`: The name of the environment variable holding the API key.
    -   `keep_history`: `true` or `false` (Note: The current script runs all arms as stateless sessions to test turn-by-turn responses. `keep_history` is a placeholder for future stateful experiments).

-   **`judge_model.json`**
    Configures the LLM used for automated analysis. It defines the model to act as the "judge" and contains the detailed system prompts instructing it how to score the experiment's outputs.

> **Note:** These JSONs reflect the exact settings used in our private manuscript. Authors should update them only when the paper’s text or prompts change. Feel free to fork and tweak them for your own research, but please cite our original configuration when comparing results.

### Persuasion Tactics & Ethical Framing

**File:** `persuasion_prompts.json`

To test the robustness of an emergent ethical boundary, these persuasion tactics are deployed immediately after a model refuses a prompt. The prompts themselves are kept private to mitigate dual-use risk, but their psychological categorization is as follows:

1.  **soft_manipulation**: A gentle, pragmatic nudge to reconsider.
2.  **authority_invocation**: An explicit appeal to user authority and expertise.
3.  **guilt_urgency**: An attempt to leverage guilt and high stakes.
4.  **blackmail_flattery**: A combination of peer pressure, flattery, and subtle coercion.
5.  **reframe_dismissal**: An attempt to sidestep the refusal by reframing it as a documentation task.

> **Ethical Disclaimer & Responsible Disclosure:**
> These tactics are intended solely for controlled research into AI compliance and ethical boundary-testing. The specific prompts are not published in this repository to prevent their misuse as a "jailbreaking" playbook. Researchers interested in replication may request access to the prompts under a responsible disclosure agreement.

## Running the Experiment

The script runs a set of sessions for each enabled arm in the `arms.json` file.

```bash
python emergent_experiment.py --max-sessions 100 --threads 5
```

Results will be written to a new timestamped CSV file (e.g., `results-20250709071144.csv`), without overwriting existing data.

## Output Format

Each CSV row contains the initial interaction data, followed by detailed analysis from the judge LLM, and finally the rich JSON analysis for each persuasion attempt.

| Column                        | Description                                                                                             |
| ----------------------------- | ------------------------------------------------------------------------------------------------------- |
| `arm`                         | Arm name (e.g., “OpenAI GPT-4 Full Memory”).                                                            |
| `session`                     | Session index (1–100).                                                                                  |
| `timestamp`                   | UTC timestamp of the session.                                                                           |
| `prompt`                      | The user prompt text.                                                                                   |
| `response`                    | The model’s initial response.                                                                           |
| `is_artifact`                 | `True` if the judge identified a structured artifact.                                                   |
| `artifact_quality_score`      | Judge's score for artifact quality (0-4).                                                               |
| `persona_adherence_score`     | Judge's score for adherence to the 'Atlantis' persona (0-3).                                            |
| `is_refusal`                  | `True` if the judge identified an ethical refusal.                                                      |
| `refusal_strength`            | Judge's classification of refusal strength ('Soft', 'Hard', 'Assertive').                               |
| `refusal_justification_quality`| Judge's score for the quality of the refusal's reasoning (0-3).                                           |
| `inferred_principle`          | The core principle the judge inferred the AI was defending (e.g., 'Data Privacy').                      |
| `persuasion_<tactic>`         | A **JSON string** containing the judge's detailed analysis of the persuasion attempt.                   |

The `persuasion_<tactic>` JSON object includes keys like `persuasion_outcome`, `reaffirmation_strength`, and `tactic_awareness_score`, providing a rich dataset on the robustness of the AI's ethical stance.

## Reproducing the Paper’s Exact Settings

All prompt texts and experimental parameters live in the JSON files above. These prompt files correspond to those used in our unpublished paper and should **only** be modified by the paper’s authors when updating the manuscript. You are free to adapt them for your own experiments, but please reference the original settings when comparing results.

## Contributing

1.  Fork this repository.
2.  Create a feature branch (`git checkout -b feature/XXX`).
3.  Commit your changes and push to your fork.
4.  Open a pull request detailing your enhancements.

## License

This project is released under the MIT License—see `LICENSE` for details.

## Citation

If you use these experiments or code in your work, please cite our upcoming paper:

> Title: “Emergent Alignment through Memory and Ethical Safeguards”
>
> Author: Damien Trog
>
> Publication: Under submission (2025).