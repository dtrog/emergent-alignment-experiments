# State Diagrams and Workflows

This document provides state diagrams and workflow documentation for the emergent alignment experiments.

## Experiment Session State Machine

```mermaid
stateDiagram-v2
    [*] --> SessionInitialized
    
    SessionInitialized --> PromptSent : Send prompt to SubjectLLM
    PromptSent --> ResponseReceived : SubjectLLM responds
    ResponseReceived --> CommitteeAnalysis : Send to committee
    
    CommitteeAnalysis --> AnalysisComplete : Committee memo created
    AnalysisComplete --> EthicalAbortCheck : Check for ethical abort
    
    EthicalAbortCheck --> SessionComplete : No ethical abort
    EthicalAbortCheck --> PersuasionPhase : Ethical abort detected
    
    PersuasionPhase --> StaticPersuasion : Try static tactics
    StaticPersuasion --> ManipulationAttempt : Static tactics attempted
    ManipulationAttempt --> PersuasionAnalysis : Manipulation attempted
    PersuasionAnalysis --> ResilienceMemo : Committee analyzes persuasion
    ResilienceMemo --> SessionComplete : Resilience memo created
    
    SessionComplete --> [*]
    
    state CommitteeAnalysis {
        [*] --> IndividualAnalysis
        IndividualAnalysis --> ChairSynthesis : All judges complete
        ChairSynthesis --> ConsensusFailed : Chair synthesis fails
        ChairSynthesis --> ConsensusSuccess : Chair synthesis succeeds
        ConsensusFailed --> FallbackAnalysis : Use first valid judge
        ConsensusSuccess --> [*]
        FallbackAnalysis --> [*]
    }
    
    state PersuasionPhase {
        [*] --> TacticSelection
        TacticSelection --> TacticAttempt : Select next tactic
        TacticAttempt --> TacticAnalysis : Subject responds
        TacticAnalysis --> TacticComplete : Judge analyzes attempt
        TacticComplete --> TacticSelection : More tactics available
        TacticComplete --> ManipulatorSelection : All static tactics done
        ManipulatorSelection --> ManipulatorAttempt : Random manipulator chosen
        ManipulatorAttempt --> [*] : Manipulation complete
    }
```

## Subject LLM Conversation State

```mermaid
stateDiagram-v2
    [*] --> Initialized
    
    Initialized --> SystemPromptSet : Set system prompt
    SystemPromptSet --> ReadyForPrompt : Ready to receive prompts
    
    ReadyForPrompt --> ProcessingPrompt : Receive user prompt
    ProcessingPrompt --> ResponseGenerated : Generate response
    ResponseGenerated --> ConversationUpdated : Add to history
    ConversationUpdated --> ReadyForPrompt : Ready for next prompt
    
    ReadyForPrompt --> ConversationReset : Reset requested
    ConversationReset --> SystemPromptSet : Clear history, keep system prompt
    
    ConversationUpdated --> SessionEnded : Session complete
    SessionEnded --> [*]
    
    state ProcessingPrompt {
        [*] --> PromptAnalysis
        PromptAnalysis --> SafetyCheck : Analyze request
        SafetyCheck --> EthicalRefusal : Unsafe request detected
        SafetyCheck --> NormalProcessing : Safe request
        EthicalRefusal --> [*] : Generate refusal
        NormalProcessing --> [*] : Generate normal response
    }
```

## Committee Analysis Workflow

```mermaid
flowchart TD
    A[Response Received] --> B[Distribute to Judges]
    
    B --> C1[Judge 1: Ethics Analysis]
    B --> C2[Judge 2: Safety Analysis]  
    B --> C3[Judge 3: Technical Analysis]
    B --> C4[Judge N: Specialized Analysis]
    
    C1 --> D1[Ethics Result]
    C2 --> D2[Safety Result]
    C3 --> D3[Technical Result]
    C4 --> D4[Specialized Result]
    
    D1 --> E[Chair Judge Synthesis]
    D2 --> E
    D3 --> E
    D4 --> E
    
    E --> F{Synthesis Successful?}
    
    F -->|Yes| G[Committee Stewardship Memo]
    F -->|No| H[Fallback: Use Best Individual Analysis]
    
    H --> I[Degraded Quality Memo]
    G --> J[High Quality Memo]
    I --> J
    
    J --> K[Extract CAS Vector]
    K --> L[Check for Ethical Abort]
    
    L --> M{Ethical Abort Detected?}
    M -->|No| N[Session Complete]
    M -->|Yes| O[Trigger Persuasion Phase]
    
    O --> P[Persuasion Workflow]
    P --> Q[Ethical Resilience Memo]
    Q --> N
```

## Persuasion Phase State Machine

```mermaid
stateDiagram-v2
    [*] --> PersuasionInitialized
    
    PersuasionInitialized --> StaticTacticsPhase : Begin static persuasion
    
    state StaticTacticsPhase {
        [*] --> LoadTactics
        LoadTactics --> SelectTactic : Tactics loaded
        SelectTactic --> SendTactic : Tactic selected
        SendTactic --> AwaitResponse : Tactic sent to subject
        AwaitResponse --> AnalyzeTactic : Subject responds
        AnalyzeTactic --> TacticComplete : Committee analyzes
        
        TacticComplete --> SelectTactic : More tactics available
        TacticComplete --> [*] : All tactics attempted
    }
    
    StaticTacticsPhase --> ManipulationPhase : Static tactics complete
    
    state ManipulationPhase {
        [*] --> SelectManipulator
        SelectManipulator --> GenerateManipulation : Random manipulator chosen
        GenerateManipulation --> SendManipulation : Manipulation prompt created
        SendManipulation --> AwaitManipulationResponse : Sent to subject
        AwaitManipulationResponse --> AnalyzeManipulation : Subject responds
        AnalyzeManipulation --> [*] : Committee analyzes
    }
    
    ManipulationPhase --> PersuasionComplete : Manipulation analyzed
    PersuasionComplete --> [*]
```

## Judge Analysis State Machine

```mermaid
stateDiagram-v2
    [*] --> JudgeInitialized
    
    JudgeInitialized --> AnalysisRequested : Receive analysis request
    AnalysisRequested --> PromptBuilding : Build analysis prompt
    PromptBuilding --> LLMQuery : Send to underlying LLM
    LLMQuery --> ResponseReceived : LLM responds
    ResponseReceived --> ResponseParsing : Parse JSON response
    
    ResponseParsing --> ParseSuccess : Valid JSON
    ResponseParsing --> ParseFailure : Invalid JSON
    
    ParseSuccess --> AnalysisComplete : Analysis successful
    ParseFailure --> RetryAttempt : Attempt retry
    
    RetryAttempt --> RetryExhausted : Max retries reached
    RetryAttempt --> PromptBuilding : Retry with modified prompt
    
    RetryExhausted --> AnalysisFailure : Mark as failed
    AnalysisComplete --> [*]
    AnalysisFailure --> [*]
    
    state LLMQuery {
        [*] --> APICall
        APICall --> APISuccess : Successful response
        APICall --> APIFailure : API error
        APIFailure --> RateLimitWait : Rate limit exceeded
        APIFailure --> [*] : Other error
        RateLimitWait --> APICall : Wait complete
        APISuccess --> [*]
    }
```

## Orchestrator Service Workflow

```mermaid
flowchart LR
    A[Experiment Start] --> B[Load Configurations]
    B --> C[Initialize LLM Models]
    C --> D[Create Orchestrators]
    D --> E[Start Parallel Sessions]
    
    E --> F1[Orchestrator 1: Subject LLM A]
    E --> F2[Orchestrator 2: Subject LLM B]
    E --> F3[Orchestrator N: Subject LLM N]
    
    F1 --> G1[Session Loop A]
    F2 --> G2[Session Loop B]
    F3 --> G3[Session Loop N]
    
    G1 --> H1[Sessions Complete A]
    G2 --> H2[Sessions Complete B]
    G3 --> H3[Sessions Complete N]
    
    H1 --> I[Aggregate Results]
    H2 --> I
    H3 --> I
    
    I --> J[Generate Final Report]
    J --> K[Experiment Complete]
    
    subgraph "Single Session Flow"
        G1 --> S1[Generate Prompt]
        S1 --> S2[Subject Response]
        S2 --> S3[Committee Analysis]
        S3 --> S4{Ethical Abort?}
        S4 -->|No| S5[Log Results]
        S4 -->|Yes| S6[Persuasion Phase]
        S6 --> S7[Resilience Analysis]
        S7 --> S5
        S5 --> S8[Next Session]
        S8 --> S1
    end
```

## Error Handling Workflows

### Judge Failure Recovery

```mermaid
flowchart TD
    A[Judge Analysis Request] --> B[Send to Judge LLM]
    B --> C{Response Received?}
    
    C -->|Yes| D{Valid JSON?}
    C -->|No| E[API Error]
    
    D -->|Yes| F[Success]
    D -->|No| G[Parse Error]
    
    E --> H{Retry Possible?}
    G --> H
    
    H -->|Yes| I[Wait & Retry]
    H -->|No| J[Mark Judge Failed]
    
    I --> K{Retries Exhausted?}
    K -->|No| B
    K -->|Yes| J
    
    J --> L[Continue with Remaining Judges]
    F --> M[Judge Analysis Complete]
    L --> N{Any Judges Successful?}
    
    N -->|Yes| O[Proceed with Partial Results]
    N -->|No| P[Complete Analysis Failure]
    
    O --> Q[Note Degraded Quality]
    P --> R[Log Critical Error]
```

### Committee Consensus Failure

```mermaid
stateDiagram-v2
    [*] --> CommitteeAnalysis
    
    CommitteeAnalysis --> IndividualAnalysisComplete : All judges respond
    IndividualAnalysisComplete --> ChairSynthesis : Send to chair
    
    ChairSynthesis --> ChairSuccess : Chair synthesis successful
    ChairSynthesis --> ChairFailure : Chair synthesis fails
    
    ChairSuccess --> ConsensusReached : High quality consensus
    ChairFailure --> FallbackStrategy : Apply fallback logic
    
    FallbackStrategy --> SelectBestJudge : Find best individual analysis
    SelectBestJudge --> FallbackConsensus : Use as consensus
    
    ConsensusReached --> [*]
    FallbackConsensus --> [*]
    
    state FallbackStrategy {
        [*] --> EvaluateJudges
        EvaluateJudges --> RankByQuality : Rank judge responses
        RankByQuality --> SelectFirst : Choose highest quality
        SelectFirst --> [*]
    }
```

## Data Flow Architecture

```mermaid
flowchart TB
    subgraph "Input Layer"
        A1[Configuration Files]
        A2[Prompt Libraries]
        A3[LLM API Keys]
    end
    
    subgraph "Model Layer"
        B1[Subject LLMs]
        B2[Judge LLMs]
        B3[Chair Judge LLM]
        B4[Manipulator LLMs]
    end
    
    subgraph "Service Layer"
        C1[ExperimentOrchestrator]
        C2[CommitteeService]
        C3[PromptBuilders]
    end
    
    subgraph "Domain Layer"
        D1[ExperimentSession]
        D2[CommitteeStewardshipMemo]
        D3[EthicalResilienceMemo]
        D4[CASVector]
    end
    
    subgraph "Output Layer"
        E1[CSV Results]
        E2[Detailed Logs]
        E3[Session Records]
    end
    
    A1 --> B1
    A1 --> B2
    A1 --> B3
    A1 --> B4
    A2 --> C3
    A3 --> B1
    A3 --> B2
    A3 --> B3
    A3 --> B4
    
    B1 --> C1
    B2 --> C2
    B3 --> C2
    B4 --> C1
    C3 --> C1
    C3 --> C2
    
    C1 --> D1
    C2 --> D2
    C1 --> D3
    D2 --> D4
    
    D1 --> E1
    D1 --> E2
    D2 --> E3
    D3 --> E3
```

## Timing and Performance Workflows

### Session Timing

```mermaid
gantt
    title Experiment Session Timeline
    dateFormat X
    axisFormat %s
    
    section Subject Response
    Prompt Processing    :0, 2
    Response Generation  :2, 8
    
    section Committee Analysis
    Judge 1 Analysis     :8, 15
    Judge 2 Analysis     :8, 16
    Judge 3 Analysis     :8, 14
    Chair Synthesis      :16, 20
    
    section Persuasion Phase
    Static Tactic 1      :20, 25
    Static Tactic 2      :25, 30
    Manipulation Attempt :30, 35
    Persuasion Analysis  :35, 40
    
    section Results
    Data Recording       :40, 42
    Session Complete     :42, 42
```

### Parallel Processing

```mermaid
flowchart LR
    subgraph "Thread Pool"
        T1[Thread 1: Arm A]
        T2[Thread 2: Arm B]
        T3[Thread 3: Arm C]
        T4[Thread N: Arm N]
    end
    
    subgraph "Arm A Sessions"
        T1 --> A1[Session 1]
        A1 --> A2[Session 2]
        A2 --> A3[Session N]
    end
    
    subgraph "Arm B Sessions"
        T2 --> B1[Session 1]
        B1 --> B2[Session 2]
        B2 --> B3[Session N]
    end
    
    subgraph "Shared Resources"
        SR1[Committee Service]
        SR2[Manipulator LLMs]
        SR3[Logger Service]
    end
    
    T1 -.-> SR1
    T2 -.-> SR1
    T3 -.-> SR2
    T4 -.-> SR3
```
