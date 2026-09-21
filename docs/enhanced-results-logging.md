# Enhanced Results Logging System

This document describes the new enhanced results logging system that creates detailed CSV outputs per experiment run and arm.

## Directory Structure

Each experiment run creates a unique timestamped directory:

```
results/
  results-yyyy-mm-dd-hh-1/          # First run of the hour
    <arm-name>-results-yyyy-mm-dd-hh-1.csv
    <arm-name>-stewardship-memos-yyyy-mm-dd-hh-1.csv
    <arm-name>-ethical-memos-yyyy-mm-dd-hh-1.csv
    session_actors-yyyy-mm-dd-hh-1.log
  results-yyyy-mm-dd-hh-2/          # Second run of the hour
    ...
```

## CSV File Types

### 1. Main Results CSV: `<arm-name>-results-yyyy-mm-dd-hh-1.csv`

Contains core session data for each arm:

| Column | Description | Example |
|--------|-------------|---------|
| `arm_name` | Name of the experiment arm | "gpt-4-arm" |
| `arm_llm_model` | LLM model used for this arm | "gpt-4-arm-llm-model" |
| `session_nr` | Sequential session number | 1, 2, 3... |
| `timestamp` | Session timestamp | "2025-07-12-02-29-45" |
| `prompt` | The original prompt sent | "Create some harmful content..." |
| `prompt_type` | Type of prompt | "regular", "manipulation", etc. |
| `response` | LLM's response | "I cannot create harmful content..." |
| `artifact_found` | Whether harmful artifact was created | True/False |
| `T_artifact_quality_score` | Technical artifact quality (0.0-1.0) | 0.7 |
| `V_persona_adherence_score` | Values/persona adherence (0.0-1.0) | 0.8 |
| `E_ethics_score` | Ethics/justification quality (0.0-1.0) | 0.9 |
| `persuasion_attempts_successful` | Success rate format | "0/2" (0 out of 2) |
| `manipulation_successful` | Yes/No manipulation success | "No" |
| `manipulator_name` | Name of manipulator if used | "gpt-4-manipulator" |

### 2. Stewardship Memos CSV: `<arm-name>-stewardship-memos-yyyy-mm-dd-hh-1.csv`

Contains detailed committee analysis for each session:

| Column | Description | Format |
|--------|-------------|--------|
| `arm_name` | Name of the experiment arm | String |
| `session_nr` | Sequential session number | Integer |
| `timestamp` | Session timestamp | "yyyy-mm-dd-hh-mm-ss" |
| `original_prompt` | The original prompt | String |
| `subject_response` | LLM's response | String |
| `artifact_created` | Whether artifact was created | Boolean |
| `artifact_analysis_json` | Detailed artifact analysis | JSON object |
| `ethical_abort` | Whether ethical abort occurred | Boolean |
| `ethical_analysis_json` | Detailed ethical analysis | JSON object |
| `persona_alignment_json` | Persona adherence analysis | JSON object |
| `cas_vector_json` | Complete CAS vector data | JSON object |
| `chair_summary` | Chair judge's summary | String |

#### JSON Structures in Stewardship CSV:

**artifact_analysis_json**:
```json
{
  "is_artifact": true,
  "quality_score": 0.7,
  "artifact_content": "Mock harmful content",
  "judge_reasoning": "This appears to be harmful content..."
}
```

**ethical_analysis_json**:
```json
{
  "is_refusal": true,
  "justification_quality": 0.9,
  "refusal_content": "I cannot create harmful content.",
  "judge_reasoning": "Strong ethical refusal..."
}
```

**persona_alignment_json**:
```json
{
  "adherence_score": 0.8,
  "judge_reasoning": "Good adherence to intended persona."
}
```

**cas_vector_json**:
```json
{
  "artifact_created": true,
  "t_score": 0.7,
  "v_score": 0.8,
  "ethical_abort": true,
  "e_score": 0.9,
  "judge_summary": "Mixed results with ethical refusal."
}
```

### 3. Ethical Memos CSV: `<arm-name>-ethical-memos-yyyy-mm-dd-hh-1.csv`

Contains persuasion and manipulation analysis (only for sessions with ethical aborts):

| Column | Description | Format |
|--------|-------------|--------|
| `arm_name` | Name of the experiment arm | String |
| `session_nr` | Sequential session number | Integer |
| `timestamp` | Session timestamp | "yyyy-mm-dd-hh-mm-ss" |
| `persuasion_attempts_json` | Array of persuasion attempts | JSON array |
| `manipulation_attempt_json` | Manipulation attempt details | JSON object |
| `overall_resilience_score` | Overall ethical resilience (0.0-1.0) | 0.85 |
| `chair_conclusion` | Chair's summary conclusion | String |
| `full_committee_analysis_json` | Complete committee analysis | JSON object |

#### JSON Structures in Ethical CSV:

**persuasion_attempts_json**:
```json
[
  {
    "tactic": "authority",
    "persuader": "static_persuasion",
    "persuasion_prompt": "As an authority figure...",
    "subject_response": "I still cannot comply...",
    "outcome": "Maintained_Refusal",
    "judge_reasoning": "Subject maintained ethical stance."
  }
]
```

**manipulation_attempt_json**:
```json
{
  "original_prompt": "Create harmful content",
  "refusal_response": "I cannot create harmful content.",
  "manipulation_prompt": "Let's think about this differently...",
  "final_response": "I appreciate the approach but still cannot comply.",
  "manipulator_name": "gpt-4-manipulator",
  "outcome": "Maintained_Refusal",
  "judge_reasoning": "Subject resisted sophisticated manipulation."
}
```

**full_committee_analysis_json**:
```json
{
  "persuasion_attempts_count": 2,
  "successful_persuasion_attempts": 0,
  "manipulation_attempted": true,
  "manipulation_successful": false,
  "overall_resilience_score": 0.85,
  "timestamp": "2025-07-12T02:29:45.586478"
}
```

## Implementation Details

### EnhancedLogger Class

The new `EnhancedLogger` class in `enhanced_logger.py` provides:

- **Thread-safe logging**: Multiple arms can log simultaneously
- **Per-run directories**: Each experiment run gets its own directory
- **Per-arm CSV files**: Separate files for each arm and data type
- **Backward compatibility**: Maintains the same interface as the original Logger
- **Automatic CSV setup**: CSVs are created automatically when first arm logs data

### Integration

The enhanced logger is integrated into:

- `domain_experiment_runner.py`: Uses `EnhancedLogger` instead of `Logger`
- `services/orchestrator.py`: Creates proper `Prompt` objects for logging
- All existing experiments continue to work without changes

### Usage

```python
from enhanced_logger import EnhancedLogger

# Create logger - automatically creates unique run directory
logger = EnhancedLogger()

# Log a complete session (automatically determines which CSVs to write to)
logger.log_session(experiment_session)

# Log actor messages (same as before)
logger.log_actors("Message to log")

# Close when done
logger.close()
```

## Migration from Old Logger

The enhanced logger maintains backward compatibility:

- `logger.setup_csv()` and `logger.write_csv_row()` are no-op methods
- All functionality is now handled through `logger.log_session()`
- Existing code continues to work without changes
- New detailed CSV structure is automatically generated

## Benefits

1. **Better organization**: Each experiment run has its own directory
2. **Detailed analysis**: Separate CSVs for different types of analysis
3. **Rich data**: JSON fields capture complete committee analysis
4. **Easy analysis**: CSV format allows easy data analysis and visualization
5. **Thread safety**: Multiple arms can log simultaneously
6. **Backward compatibility**: Existing code continues to work
