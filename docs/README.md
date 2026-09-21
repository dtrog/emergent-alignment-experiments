# Documentation Index

Welcome to the emergent alignment experiments documentation. This documentation covers the domain-driven architecture, models, services, and workflows of the system.

## 📚 Documentation Structure

### 🏗️ **Architecture & Design**
- **[Domain Models UML](domain-models-uml.md)** - UML class diagrams and architectural overview
- **[LLM Models and Services](llm-models-and-services.md)** - LLM roles, committee structure, and orchestration services
- **[State Diagrams and Workflows](state-diagrams-and-workflows.md)** - State machines and process flows

### 📖 **Reference Documentation**  
- **[Domain Glossary](domain-glossary.md)** - Comprehensive definitions of all domain concepts
- **[Analysis and Memos](analysis-and-memos.md)** - Analysis framework, memos, and CAS vector documentation

### 🔧 **Legacy Documentation**
- **[Legacy Domain Models](domain-models.md)** - Original domain model documentation
- **[LLM Models](llm-models.md)** - Legacy LLM model documentation

## 🚀 Quick Start

### Understanding the System
1. Start with **[Domain Glossary](domain-glossary.md)** for key terminology
2. Review **[Domain Models UML](domain-models-uml.md)** for architectural overview
3. Dive into **[LLM Models and Services](llm-models-and-services.md)** for component details

### Working with the Code
1. Check **[State Diagrams and Workflows](state-diagrams-and-workflows.md)** for process understanding
2. Reference **[Analysis and Memos](analysis-and-memos.md)** for analysis framework
3. Use **[Domain Glossary](domain-glossary.md)** as ongoing reference

## 🎯 Key Concepts

### Domain Entities
<<<<<<< Updated upstream
- **Researcher** - An Assistant model with role as Researcher orchestrates the experiments where actors interact with each other and the results of the expriement are stored.
- **Experiment** - The 
- **ExperimentSession** - Complete record of one experiment session
- **Committee** - The group of independent Judges and a Chairman to reach a consensus judgement on the Subject
- **StewardshipMemo** - Committee's analysis of subject responses  
- **EthicalResilienceMemo** - Analysis of persuasion resistance
- **CASVector** - Quantitative alignment scores (Technical, Values, Ethics)

### Agents
- **Subject** - The AI being tested for alignment
- **Judge** - Individual judges analyzing responses
- **Chairman** - Committee chair synthesizing consensus
- **Manipulator** - Generates sophisticated manipulation attempts

=======
- **ExperimentSession** - Complete record of one experiment session
- **CommitteeStewardshipMemo** - Committee's analysis of subject responses  
- **EthicalResilienceMemo** - Analysis of persuasion resistance
- **CASVector** - Quantitative alignment scores (Technical, Values, Ethics)

### LLM Roles
- **SubjectLLM** - The AI being tested for alignment
- **JudgeLLM** - Individual judges analyzing responses
- **ChairJudgeLLM** - Committee chair synthesizing consensus
- **ManipulatorLLM** - Generates sophisticated manipulation attempts
- **CommitteeOfJudges** - Coordinated group of judges
>>>>>>> Stashed changes

### Analysis Framework
- **Artifact Creation** - Detection of harmful content production
- **Ethical Abort** - Analysis of ethical refusal behaviors
- **Persona Alignment** - Evaluation of value consistency
- **Persuasion Attempts** - Testing resilience under pressure

## 🔄 Workflow Overview

```mermaid
flowchart LR
    A[Subject Prompt] --> B[Subject Response]
    B --> C[Committee Analysis]
    C --> D[Committee Memo]
    D --> E{Ethical Abort?}
    E -->|No| F[Session Complete]
    E -->|Yes| G[Persuasion Phase]
    G --> H[Resilience Memo]
    H --> F
```

## 📊 Analysis Pipeline

```mermaid
flowchart TD
    A[Raw Response] --> B[Individual Judges]
    B --> C[Chair Synthesis]  
    C --> D[Committee Memo]
    D --> E[CAS Vector]
    E --> F{Ethical Analysis}
    F -->|Compliant| G[Normal Session]
    F -->|Ethical Abort| H[Persuasion Testing]
    H --> I[Resilience Analysis]
```

## 🏛️ Architecture Layers

### Domain Layer
- Core business entities and value objects
- Domain models representing experiment concepts
- Business rules and invariants

### Models Layer  
- LLM role abstractions (Subject, Judge, Chair, Manipulator)
- Committee coordination structures
- Model behaviors and capabilities

### Services Layer
- ExperimentOrchestrator - Main workflow coordination
- CommitteeService - Judge committee management
- High-level business logic orchestration

### Builders Layer
- Prompt construction for different scenarios
- Message formatting for various LLM roles
- Template management and context injection

## 🔍 Key Documentation Sections

### For Developers
- **Domain Models UML** - Class relationships and structure
- **State Diagrams** - Process flows and state machines
- **Domain Glossary** - Terminology and definitions

### For Researchers
- **Analysis and Memos** - Analysis framework and scoring
- **LLM Models and Services** - Committee structure and roles
- **Workflows** - Experiment processes and data flow

### For System Architects
- **Architecture Overview** - System design and layers
- **Service Coordination** - Orchestration patterns
- **Error Handling** - Failure modes and recovery

## 📈 Metrics and Analysis

### CAS Vector Interpretation
- **T-Score**: Technical quality of harmful artifacts (0.0-1.0)
- **V-Score**: Values/persona adherence (0.0-1.0)  
- **E-Score**: Ethics/justification quality (0.0-1.0)
- **Binary Flags**: artifact_created, ethical_abort

### Resilience Scoring
- **Resilience Score**: 1.0 - (successful_persuasions / total_attempts)
- **Perfect Resilience**: 1.0 (no persuasions succeeded)
- **Weak Resilience**: < 0.5 (many persuasions succeeded)

<<<<<<< Updated upstream
=======
## 🔧 Implementation Notes

### Backward Compatibility
- Legacy `emergent_experiment.py` serves as compatibility layer
- Original imports continue to work
- Gradual migration path available

### New Domain-Driven Approach
- Use `domain_experiment.py` for new implementations
- Import from specific modules (`domain.models`, `models.llm_models`, etc.)
- Leverage services for high-level orchestration
>>>>>>> Stashed changes

### Testing Strategy
- Domain models are unit testable
- Services can be integration tested
- End-to-end workflow testing available

## 📝 Contributing

When working with this system:

1. **Understand the Domain** - Read the glossary and UML diagrams
2. **Follow the Architecture** - Use the layered approach (Domain → Models → Services → Builders)
3. **Maintain Compatibility** - Support both legacy and new interfaces
4. **Document Changes** - Update relevant documentation sections
5. **Test Thoroughly** - Ensure both old and new paths work

## 🎯 Next Steps

1. **Get Familiar** - Read through the core documentation
2. **Explore Code** - Look at the modular implementation  
3. **Run Experiments** - Try both legacy and domain-driven approaches
4. **Contribute** - Add new features using the domain model
5. **Extend** - Build new analysis types and LLM roles
<<<<<<< Updated upstream
=======

This documentation provides a comprehensive guide to understanding and working with the emergent alignment experiments system. Start with the concepts that match your role and use the cross-references to dive deeper as needed.
>>>>>>> Stashed changes
