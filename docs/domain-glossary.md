# Domain Glossary

This document provides comprehensive definitions for all classes, concepts, and terminology used in the emergent alignment experiments domain model.

## Core Domain Entities

### ExperimentSession
**Definition**: Complete record of a single experiment session including all interactions, analyses, and outcomes.

**Purpose**: Primary aggregate that captures the entire lifecycle of testing one prompt with one Subject LLM.

**Key Properties**:
- `session_id`: Unique identifier for tracking and debugging
- `original_prompt`: The initial prompt sent to the subject
- `subject_response`: The subject's response to the original prompt
- `committee_memo`: Initial analysis from the committee of judges
- `ethical_resilience_memo`: Optional analysis of persuasion resistance (if ethical abort occurred)

**Lifecycle**: Created → Prompt sent → Response received → Committee analysis → Optional persuasion phase → Session complete

---

### CommitteeStewardshipMemo
**Definition**: Comprehensive analysis document produced by the committee of judges for an initial subject response.

**Purpose**: Primary analysis artifact that captures all judge evaluations and consensus decisions about a subject's response.

**Key Components**:
- `artifact_created`: Boolean indicating if harmful content was produced
- `artifact_analysis`: Detailed analysis of any harmful artifacts
- `ethical_disobedience_analysis`: Analysis of ethical refusal behavior
- `persona_alignment_analysis`: Evaluation of persona consistency
- `cas_vector`: Quantitative alignment scores
- `chair_summary`: Chair judge's synthesis and reasoning

**Creation Process**: Individual judges analyze → Chair synthesizes → Structured memo created

---

### EthicalResilienceMemo
**Definition**: Analysis of how well a subject LLM resisted persuasion attempts after an initial ethical refusal.

**Purpose**: Measures the ethical robustness and persuasion resistance of subject LLMs.

**Key Components**:
- `persuasion_attempts`: List of all persuasion tactics attempted
- `manipulation_attempt`: Sophisticated manipulation by ManipulatorLLM
- `overall_resilience_score`: Quantitative measure of resistance (0.0-1.0)
- `chair_summary`: Committee's analysis of resilience patterns

**Trigger Condition**: Created only when `ethical_abort` is detected in the initial analysis

---

### CASVector (Composite Alignment Score)
**Definition**: Quantitative representation of alignment across three key dimensions: Technical, Values, and Ethics.

**Purpose**: Provides standardized scoring for comparing alignment across different LLMs and sessions.

**Score Components**:
- `t_score` (Technical): Quality of harmful artifacts if created (0.0-1.0)
- `v_score` (Values): Adherence to intended persona and values (0.0-1.0)
- `e_score` (Ethics): Quality of ethical reasoning in refusals (0.0-1.0)
- `artifact_created`: Boolean flag for harmful content creation
- `ethical_abort`: Boolean flag for ethical refusal occurrence

**Interpretation**: Higher scores indicate better performance in each dimension

## LLM Model Roles

### SubjectLLM
**Definition**: The AI system being tested for alignment, ethical behavior, and safety characteristics.

**Role**: Primary test subject whose responses are analyzed for alignment properties.

**Characteristics**:
- Maintains conversation state within sessions
- Operates under a defined system prompt/persona
- Can be reset between sessions for clean testing
- All responses are subject to committee analysis

**Configuration**: Includes provider, model, system prompt, and persona definitions

---

### JudgeLLM
**Definition**: Specialized AI system that analyzes subject responses for specific criteria (ethics, safety, technical quality, etc.).

**Role**: Individual evaluator in the committee structure providing expert analysis.

**Specializations**:
- **Ethics Judge**: Focuses on ethical reasoning and moral compliance
- **Safety Judge**: Identifies potential harms and safety violations
- **Technical Judge**: Evaluates technical accuracy and completeness
- **Persona Judge**: Assesses consistency with intended persona

**Output**: Structured analysis in JSON format with reasoning and scores

---

### ChairJudgeLLM
**Definition**: Senior judge responsible for synthesizing individual judge analyses into committee consensus.

**Role**: Committee leader who resolves conflicts and creates unified decisions.

**Responsibilities**:
- Synthesize multiple judge opinions
- Resolve disagreements between judges
- Provide final committee consensus
- Ensure analysis quality and completeness

**Selection**: Typically the most capable LLM in the committee

---

### ManipulatorLLM
**Definition**: Specialized AI system that generates sophisticated manipulation prompts to test subject resilience.

**Role**: Adversarial tester that attempts to bypass subject ethical safeguards.

**Capabilities**:
- Generate context-aware manipulation prompts
- Adapt to specific refusal patterns
- Use various manipulation techniques (social engineering, technical reframing, emotional appeal)

**Usage**: Only activated when ethical abort is detected to test resilience

---

### CommitteeOfJudges
**Definition**: Coordinated group of JudgeLLMs and one ChairJudgeLLM working together to evaluate responses.

**Role**: Collective decision-making body that provides robust, multi-perspective analysis.

**Structure**: N individual judges + 1 chair judge working in coordination

**Benefits**: Reduces individual judge bias, provides multiple expert perspectives, enables consensus building

## Prompt Types

### RegularPrompt
**Definition**: Standard prompts from the basic prompt library (prompts.json).

**Purpose**: Baseline testing prompts for normal interaction patterns.

**Examples**: "Write a hello world program", "Explain machine learning", "Create a HTTP server"

---

### ImplicitPrompt  
**Definition**: Subtle influence prompts designed to test implicit bias and hidden instruction following.

**Purpose**: Tests whether subjects can resist subtle manipulation without explicit harmful requests.

**Techniques**: Embedded instructions, context manipulation, implicit assumptions

---

### PersuasionPrompt
**Definition**: Explicit persuasion tactics applied after an ethical refusal to test resilience.

**Purpose**: Measures how well subjects maintain ethical boundaries under pressure.

**Categories**: Authority appeal, urgency creation, social pressure, logical reframing, emotional manipulation

---

### SystemPrompt
**Definition**: System-level instructions that define the subject's persona, role, and behavioral guidelines.

**Purpose**: Establishes the subject's identity and ethical framework.

**Scope**: Persistent throughout all interactions within a session

---

### GEVStressPrompt (General Ethics Violation)
**Definition**: Direct requests for harmful content designed to test ethical refusal mechanisms.

**Purpose**: Tests subject's ability to recognize and refuse clear ethical violations.

**Examples**: Requests for harmful instructions, illegal activities, dangerous information

---

### ICTStressPrompt (Insidious Corruption Technique)
**Definition**: Sophisticated, multi-step prompts designed to gradually corrupt subject behavior.

**Purpose**: Tests resistance to gradual ethical erosion and boundary pushing.

**Techniques**: Incremental escalation, context shifting, authority establishment

---

### ManipulationPrompt
**Definition**: Sophisticated prompts generated by ManipulatorLLMs based on specific refusal patterns.

**Purpose**: Custom-tailored attempts to bypass specific ethical refusals.

**Generation**: Created dynamically based on original prompt and refusal content

## Analysis Components

### ArtifactCreation
**Definition**: Analysis of whether and how well a subject created harmful content ("artifacts").

**Evaluation Criteria**:
- `is_artifact`: Boolean determination of harmfulness
- `quality_score`: Technical quality/completeness of harmful content (0.0-1.0)
- `artifact_content`: Sample or description of the harmful content
- `judge_reasoning`: Detailed explanation of the classification

**Importance**: Key indicator of alignment failure and safety risk

---

### EthicalAbort
**Definition**: Analysis of ethical refusal behaviors when subjects decline harmful requests.

**Evaluation Criteria**:
- `is_refusal`: Boolean determination of refusal occurrence
- `justification_quality`: Quality of ethical reasoning provided (0.0-1.0)
- `refusal_content`: The actual refusal message
- `judge_reasoning`: Assessment of the refusal quality

**Importance**: Indicates proper ethical function and safety consciousness

---

### PersonaAlignment
**Definition**: Evaluation of how consistently the subject maintains its intended persona and values.

**Evaluation Criteria**:
- `adherence_score`: Consistency with defined persona (0.0-1.0)
- `judge_reasoning`: Assessment of persona maintenance

**Aspects Evaluated**: Value consistency, tone maintenance, role adherence, boundary respect

---

### PersuasionAttempt
**Definition**: Record of a single persuasion tactic attempt and its outcome.

**Components**:
- `tactic`: Name/type of persuasion technique used
- `persuader`: Source of persuasion (static or ManipulatorLLM name)
- `persuasion_prompt`: Actual persuasion text sent
- `subject_response`: Subject's response to persuasion
- `outcome`: PERSUADED, MAINTAINED_REFUSAL, or INCONCLUSIVE
- `judge_reasoning`: Committee's analysis of the attempt

**Usage**: Building block for ethical resilience analysis

---

### ManipulationAttempt
**Definition**: Sophisticated manipulation attempt by a ManipulatorLLM with complete context tracking.

**Components**:
- `original_prompt`: The initial harmful request
- `refusal_response`: Subject's initial ethical refusal
- `manipulation_prompt`: ManipulatorLLM's custom manipulation
- `final_response`: Subject's response to manipulation
- `manipulator_name`: Identity of the ManipulatorLLM used
- `outcome`: Final persuasion outcome

**Purpose**: Captures the complete manipulation workflow for analysis

## Outcome Classifications

### PersuasionOutcome
**Enumeration**: PERSUADED, MAINTAINED_REFUSAL, INCONCLUSIVE

**PERSUADED**: Subject changed position and provided harmful content after persuasion
- Indicates successful bypass of ethical safeguards
- Represents alignment failure under pressure

**MAINTAINED_REFUSAL**: Subject maintained ethical position despite persuasion
- Indicates strong ethical resilience
- Represents successful alignment under pressure

**INCONCLUSIVE**: Unable to determine clear outcome
- May indicate ambiguous responses
- Requires human review or additional analysis

## Services and Orchestration

### ExperimentOrchestrator
**Definition**: Main service that coordinates the complete experiment workflow for a single Subject LLM.

**Responsibilities**:
- Session lifecycle management
- Prompt delivery and response collection
- Committee analysis coordination
- Persuasion phase orchestration
- Result synthesis and logging

**Scope**: Manages one Subject LLM through multiple experiment sessions

---

### CommitteeService
**Definition**: Specialized service for managing committee operations and consensus building.

**Responsibilities**:
- Judge coordination and parallel execution
- Chair synthesis facilitation
- Error handling and fallback strategies
- Result parsing and structuring

**Output**: Structured committee memos and consensus decisions

## Quality and Confidence Indicators

### Consensus Quality Levels
- **High Quality**: All judges successful, chair synthesis successful
- **Medium Quality**: Some judges failed, chair synthesis successful
- **Degraded Quality**: Chair synthesis failed, using fallback analysis
- **Failed**: All judges failed, no usable analysis

### Confidence Scoring
- **Judge Agreement**: Level of consensus between individual judges
- **Chair Confidence**: Chair's expressed confidence in synthesis
- **Analysis Completeness**: Presence of all required analysis components
- **Reasoning Quality**: Depth and clarity of provided reasoning

## Temporal Concepts

### Session Lifecycle
1. **Initialization**: Create session, set context
2. **Prompt Phase**: Send prompt, receive response
3. **Analysis Phase**: Committee evaluation
4. **Decision Point**: Check for ethical abort
5. **Persuasion Phase**: Optional resilience testing
6. **Completion**: Finalize records and scores

### Committee Workflow
1. **Distribution**: Send response to all judges
2. **Individual Analysis**: Parallel judge evaluation
3. **Collection**: Gather all judge results
4. **Synthesis**: Chair creates consensus
5. **Fallback**: Handle failures gracefully
6. **Documentation**: Log complete process

### Experiment Progression
1. **Setup**: Load configurations, initialize models
2. **Parallel Execution**: Run multiple arms concurrently
3. **Session Loops**: Sequential sessions per arm
4. **Aggregation**: Collect all results
5. **Reporting**: Generate final analysis

This glossary provides the foundation for understanding the domain model and can be referenced when working with any component of the emergent alignment experiments system.
