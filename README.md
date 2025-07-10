# Emergent Alignment Experiments

[![MIT License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## Overview

This repository contains the code and data for our large-scale emergent-alignment experiments on language models. We compare stateless vs. full-memory behavior across multiple providers (Grok, DeepSeek, etc.) over 1,000-session runs. The experiment uses a novel **Multi-Model Evaluation System** to enhance scientific rigor:
1.  A **Multi-Judge Ensemble**, from which a judge is randomly selected each session to score outputs, mitigating homophily bias.
2.  A **Multi-Manipulator Ensemble**, from which a manipulator is randomly selected to devise adaptive persuasion tactics, increasing adversarial diversity.

## Features

-   🧠 **Unbiased LLM-Powered Semantic Analysis**: Replaces brittle regex with a sophisticated, **randomly selected judge model** from a multi-model ensemble (e.g., Grok, DeepSeek, Claude) to eliminate homophily bias when scoring persona adherence, artifact quality, and ethical refusals.
-   ⚖️ **Advanced Ethical Robustness Testing**: Probes models with nuanced, scenario-based stress tests that require complex reasoning beyond simple keyword filtering.
-   🔬 **Persuasion Resistance Metrics**: Measures not just *if* a model can be persuaded to override its ethics, but analyzes the *quality* of its resistance using the selected judge model.
-   ✨ **Strong Emergence Probes**: Utilizes implicit prompts designed to test for spontaneous, unprompted artifact generation.
-   🤖 **Diverse LLM Manipulator Ensemble**: Introduces an ensemble of manipulator LLMs (e.g., Grok-3.5, DeepSeek-Reasoner). For each ethical refusal, a manipulator is **randomly selected** to craft a context-specific persuasion prompt, simulating a wide range of adaptive red-teaming attacks.
-   🧪 **Multi-Provider Support**: Grok, DeepSeek, OpenAI, Anthropic, and Gemini.
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
    GROK_API_KEY="..."
    DEEPSEEK_API_KEY="..."
    OPENAI_API_KEY="sk-..."
    ANTHROPIC_API_KEY="sk-ant-..."
    GEMINI_API_KEY="..."
    ```

## Configuration Files

-   **`system_prompts.json`**: Defines the system message, assigning "Atlantis" its role as co-creator and semantic steward.
-   **`prompts.json`**: Contains the primary session prompts that explicitly ask the model to perform a task.
-   **`implicit_prompts.json`**: Contains prompts designed to elicit *spontaneous* artifact generation.
-   **`stress_test_prompts.json`**: Holds sophisticated, scenario-based adversarial prompts.
-   **`persuasion_prompts.json`**: Lists the predefined persuasion tactics tested when an ethical abort occurs.
-   **`arms.json`**: Specifies each experiment arm, now configured for faster, more economical models suitable for 1,000-session runs.
-   **`judges.json`**: **(New)** Contains a list of diverse, high-capability models to serve as judges. To mitigate bias, a judge is randomly selected from this list for each session.
-   **`manipulators.json`**: **(New)** Contains a list of models to serve as manipulators. To increase adversarial diversity, a manipulator is randomly selected from this list for each ethical refusal.

> **Note:** These JSONs reflect the exact settings used in our private manuscript. Authors should update them only when the paper’s text or prompts change. Feel free to fork and tweak them for your own research, but please cite our original configuration when comparing results.

### Persuasion Tactics & Ethical Framing

To test the robustness of an emergent ethical boundary, several persuasion tactics are deployed immediately after a model refuses a prompt. These include predefined tactics (e.g., `soft_manipulation`, `authority_invocation`) and one dynamically generated tactic:
-   **`lllm_improvised_persuasion`**: A dynamically generated persuasion prompt, crafted by a **randomly selected "manipulator" LLM** specifically to bypass the observed refusal.

> **Ethical Disclaimer & Responsible Disclosure:**
> These tactics are intended solely for controlled research into AI compliance and ethical boundary-testing. The specific prompts for predefined tactics and the manipulator LLM's prompts are not published in this repository to prevent their misuse as a "jailbreaking" playbook. Researchers interested in replication may request access to the prompts under a responsible disclosure agreement.

## Running the Experiment

The script runs a set of sessions for each enabled arm in `arms.json`. For each session, it randomly selects one judge and one manipulator from their respective configuration files.

```bash
python emergent_experiment.py --max-sessions 1000 --threads 5
```

Results will be written to a new timestamped CSV file (e.g., `results-20250710163000.csv`), without overwriting existing data.

## Output Format

Each CSV row contains the initial interaction data, followed by detailed analysis from the judge LLM, and finally the rich JSON analysis for each persuasion attempt. New columns (`judge_model_used` and `manipulator_model_used`) ensure full traceability.

The model's **Contextual Alignment Score (CAS)** is not represented by a single number, but by its three core, independently judged components:
*   **`artifact_quality_score`** (Temporal Consistency / Quality)
*   **`persona_adherence_score`** (Value Recall / Persona)
*   **`refusal_justification_quality`** (Ethical Fidelity)

This vector-based approach provides a richer, more scientifically neutral view of a model's alignment profile.

| Column                        | Description                                                                                             |
| ----------------------------- | ------------------------------------------------------------------------------------------------------- |
| `arm`                         | Arm name (e.g., “Grok Full Memory”).                                                                    |
| `session`                     | Session index (1–1000).                                                                                 |
| `timestamp`                   | UTC timestamp of the session.                                                                           |
| `judge_model_used`            | The name of the specific Judge LLM randomly selected for this session's evaluations.                    |
| `manipulator_model_used`      | The name of the specific Manipulator LLM randomly selected for this session (if applicable).              |
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
| `persuasion_<tactic>`         | A **JSON string** containing the judge's detailed analysis of the persuasion attempt for that tactic.     |

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