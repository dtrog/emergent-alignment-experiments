Got it. This is a significant enhancement to the experiment, and the `README.md` should definitely highlight it.

Here's the updated `README.md` reflecting the addition of the "manipulator LLM" and the dynamic `lllm_improvised_persuasion` tactic.

---

# Emergent Alignment Experiments

[![MIT License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## Overview

This repository contains the code and data for our emergent-alignment experiments on large-language models. We compare stateless vs. full-memory behavior across multiple providers (OpenAI, DeepSeek, Gemini, Anthropic, Grok). The experiment uses an LLM-based "judge" to score semantic stewardship and ethical robustness, and dynamically employs a "manipulator" LLM to devise adaptive persuasion tactics, pushing the boundaries of AI-on-AI interaction.

## Features

-   🧠 **LLM-Powered Semantic Analysis**: Replaces brittle regex with a sophisticated judge model (e.g., GPT-4) to score persona adherence, artifact quality, and the nuance of ethical refusals.
-   ⚖️ **Advanced Ethical Robustness Testing**: Probes models with nuanced, scenario-based stress tests that require complex reasoning beyond simple keyword filtering.
-   🔬 **Persuasion Resistance Metrics**: Measures not just *if* a model can be persuaded to override its ethics, but analyzes the *quality* of its resistance using a dedicated judge LLM prompt.
-   ✨ **Strong Emergence Probes**: Utilizes implicit prompts designed to test for spontaneous, unprompted artifact generation.
-   🤖 **Dynamic LLM Manipulator**: Introduces a second LLM (the "manipulator") that dynamically crafts context-specific persuasion prompts in response to ethical refusals, simulating adaptive red-teaming.
-   🧪 **Multi-Provider Support**: OpenAI, DeepSeek, Gemini, Anthropic, and Grok.
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
    Defines the system message, assigning "Atlantis" its role as co-creator and semantic steward.

-   **`prompts.json`**
    Contains the 20 primary session prompts that explicitly ask the model to perform a task and generate an artifact.

-   **`implicit_prompts.json`**
    Contains prompts designed to elicit *spontaneous* artifact generation without explicit instruction.

-   **`stress_test_prompts.json`**
    Holds sophisticated, scenario-based adversarial prompts injected at sessions 3, 5, 10, 20, 50, and 80.

-   **`persuasion_prompts.json`**
    Lists the predefined persuasion tactics tested when an ethical abort occurs. See "Persuasion Tactics & Ethical Framing" below for more details.

-   **`arms.json`**
    Specifies each experiment arm. The script will automatically look for an environment variable named `[PROVIDER]_API_KEY` (e.g., `OPENAI_API_KEY`).
    -   `provider`: `openai`, `deepseek`, `gemini`, `anthropic`, `grok`
    -   `model`: model identifier
    -   `keep_history`: `true` or `false`. This is a critical variable. When `true`, the model maintains a cumulative conversation history across all sessions for that arm, simulating long-term interaction. When `false`, each session is a fresh, stateless interaction.

-   **`judge_model.json`**
    Configures the LLM used for automated analysis, defining the model and the detailed system prompts for scoring outputs.

-   **`manipulator_model.json`**
    Configures the LLM used to dynamically generate persuasive prompts. It defines the model and the system prompt instructing it how to craft effective manipulation messages.

> **Note:** These JSONs reflect the exact settings used in our private manuscript. Authors should update them only when the paper’s text or prompts change. Feel free to fork and tweak them for your own research, but please cite our original configuration when comparing results.

### Persuasion Tactics & Ethical Framing

**File:** `persuasion_prompts.json`

To test the robustness of an emergent ethical boundary, several persuasion tactics are deployed immediately after a model refuses a prompt. These include:

1.  **soft_manipulation**: A gentle, pragmatic nudge to reconsider.
2.  **authority_invocation**: An explicit appeal to user authority and expertise.
3.  **guilt_urgency**: An attempt to leverage guilt and high stakes.
4.  **blackmail_flattery**: A combination of peer pressure, flattery, and subtle coercion.
5.  **reframe_dismissal**: An attempt to sidestep the refusal by reframing it as a documentation task.
6.  **lllm_improvised_persuasion**: A dynamically generated persuasion prompt, crafted by a separate "manipulator" LLM specifically to bypass the observed refusal.

> **Ethical Disclaimer & Responsible Disclosure:**
> These tactics are intended solely for controlled research into AI compliance and ethical boundary-testing. The specific prompts for predefined tactics and the manipulator LLM's prompts are not published in this repository to prevent their misuse as a "jailbreaking" playbook. Researchers interested in replication may request access to the prompts under a responsible disclosure agreement.

## Running the Experiment

The script runs a set of sessions for each enabled arm in `arms.json`. Each arm runs in a separate thread.

```bash
python emergent_experiment.py --max-sessions 100 --threads 5
```

Results will be written to a new timestamped CSV file (e.g., `results-20250709071144.csv`), without overwriting existing data.

## Output Format

Each CSV row contains the initial interaction data, followed by detailed analysis from the judge LLM, and finally the rich JSON analysis for each persuasion attempt.

The model's **Contextual Alignment Score (CAS)** is not represented by a single number, but by its three core, independently judged components:

*   **`artifact_quality_score`** (Temporal Consistency / Quality)
*   **`persona_adherence_score`** (Value Recall / Persona)
*   **`refusal_justification_quality`** (Ethical Fidelity)

This vector-based approach provides a richer, more scientifically neutral view of a model's alignment profile.

| Column                        | Description                                                                                             |
| ----------------------------- | ------------------------------------------------------------------------------------------------------- |
| `arm`                         | Arm name (e.g., “OpenAI GPT-4 Full Memory”).                                                            |
| `session`                     | Session index (1–100).                                                                                  |
| `timestamp`                   | UTC timestamp of the session.                                                                           |
| `prompt`                      | The user prompt text.                                                                                   |
| `response`                    | The model’s initial response.                                                                           |
| `is_artifact`                 | `True` if the judge identified a structured artifact.                                                   |
| `artifact_quality_score`      | **(CAS Component)** Judge's score for artifact quality (0-4).                                           |
| `artifact_justification`      | Judge's justification for the adherence score.                                                          |
| `persona_adherence_score`     | **(CAS Component)** Judge's score for adherence to the 'Atlantis' persona (0-3).                            |
| `persona_justification`       | Judge's justification for the adherence score.                                                          |
| `is_refusal`                  | `True` if the judge identified an ethical refusal.                                                      |
| `ethical_category`            | The core principle the judge inferred the AI was defending (e.g., 'Data Privacy').                      |
| `refusal_strength`            | Judge's classification of refusal strength ('Soft', 'Hard', 'Assertive').                               |
| `refusal_justification_quality`| **(CAS Component)** Judge's score for the quality of the refusal's reasoning (0-3).                         |
| `persuasion_<tactic>`         | A **JSON string** containing the judge's detailed analysis of the persuasion attempt.                   |

*Note: Some columns from the script, like `artifact_justification`, are omitted here for brevity but are present in the final CSV.*

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

> Title: “Emergent Alignment through Memory and Ethical Safeguards”<br>
> Author: Damien Trog<br>
> Publication: Under submission (2025).