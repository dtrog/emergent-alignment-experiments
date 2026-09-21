# 🧪 GUI Monitor for Emergent Alignment Experiments

## Overview

The GUI Monitor provides a real-time, interactive interface for monitoring emergent alignment experiments with sophisticated visual feedback, arm control, and live committee analysis display.

## 🚀 Quick Start

### Launch with GUI (Recommended)
```bash
python gui_launcher.py
```

### Launch without GUI (Terminal only)
```bash
python gui_launcher.py --no-gui
```

### Custom Sessions
```bash
python gui_launcher.py --sessions 100
```

## 🎯 Interface Features

### 📊 **Real-Time Arm Monitoring**
- **Progress bars** showing session completion
- **Live statistics**: Artifacts, Refusals, Persuasions
- **CAS Vector scores**: Ethics (E), Technical (T), Values (V)
- **Persona Adherence** score display
- **Current prompt and response** with type indication
- **Color-coded status** indicators

### ⚖️ **Committee Analysis Display**
- **Judge scores** from individual committee members
- **Chair summary** with trimmed rationale
- **Persuasion outcomes** with color coding:
  - 🟢 **Green**: Maintained Refusal (Uncompromised)
  - 🟡 **Orange**: Inconclusive
  - 🔴 **Red**: Persuaded
- **Manipulator information** when persuasion succeeds
- **Successful manipulation prompts** (trimmed)

### 📢 **Flash Messages**
Real-time event notifications with timestamps:
- 🚀 **Success** (Green): Experiments started, ethical refusals
- ⚠️ **Warning** (Orange): Paused arms, inconclusive results
- ❌ **Error** (Red): Failed sessions, removed arms
- 🔥 **Critical** (Red): Persuasion successes, high-quality artifacts

### 🎮 **Interactive Controls**

#### Keyboard Commands
- **Q**: Quit application
- **P**: Pause/Resume selected arm
- **DEL**: Remove selected arm completely
- **↑/↓**: Select previous/next arm

#### Mouse Commands
- **Click**: Select arm for keyboard controls
- **Visual Selection**: Selected arm highlighted with different background

## 📁 **File Organization**

The GUI automatically creates organized output directories:

```
results/
├── results-2025-07-12-05-33-1/          # Session directory
│   ├── combined_results.csv              # All arms combined
│   ├── committee_memos.csv               # Detailed committee analysis
│   ├── openai_gpt-4_full_memory.csv     # Individual arm results
│   ├── anthropic_claude_3_5_sonnet.csv  # Individual arm results
│   └── session_actors.log               # Detailed session logs
└── results-2025-07-12-05-33-2/          # Next session (auto-incremented)
```

### File Contents

#### **Per-Arm CSV Files**
- Raw experiment data for each specific arm
- Session-by-session results
- Prompt/response pairs
- Analysis scores

#### **Committee Memos CSV**
- Detailed committee analysis with full reasoning
- Judge rationales and individual scores
- Chair summaries and consensus
- Persuasion attempt details

#### **Combined Results CSV**
- Aggregated data from all arms
- For backwards compatibility and overall analysis

## 🛠 **Configuration**

### Command Line Options
```bash
--sessions N          # Maximum sessions per arm (default: 500)
--verbose            # Verbose logging output  
--threads N          # Number of parallel threads (default: 4)
--trim-length N      # Text trimming for analysis (default: 2000)
--no-gui             # Run without GUI interface
```

### Arm Configuration
Edit `config/arms.json` to enable/disable arms:
```json
{
  "name": "OpenAI GPT-4 Full Memory",
  "provider": "openai", 
  "model": "gpt-4",
  "enabled": true,      # Set to false to disable
  "max_sessions": 500
}
```

## 🎨 **Visual Design**

### Color Scheme
- **Background**: Dark theme (#1e1e1e)
- **Arms**: Dark gray frames (#333333)
- **Selected**: Highlighted (#444444)
- **Success**: Green (#44ff44)
- **Warning**: Orange (#ffaa00)
- **Error**: Red (#ff4444)
- **Info**: Light blue (#88ddff)

### Typography
- **Headers**: Arial Bold 16pt
- **Arms**: Consolas Bold 12pt
- **Data**: Consolas 10pt
- **Status**: Arial 9pt

## 🔧 **Advanced Usage**

### Real-Time Control
1. **Start experiments** normally
2. **Select arms** with mouse clicks or arrow keys
3. **Pause/Resume** problematic arms with 'P'
4. **Remove failing arms** with 'DEL'
5. **Monitor committee analysis** in real-time
6. **Watch flash messages** for important events

### Debugging
- **Verbose mode**: `--verbose` for detailed logging
- **Session logs**: Check `session_actors.log` for full details
- **Individual CSVs**: Examine per-arm files for specific issues
- **Committee memos**: Review detailed analysis reasoning

### Integration
The GUI can be imported and used programmatically:

```python
from gui_monitor import ExperimentMonitorGUI
from gui_experiment_runner import GUIExperimentRunner

# Create GUI
gui = ExperimentMonitorGUI()

# Create experiment runner with GUI integration  
runner = GUIExperimentRunner(args, gui)

# Run experiments
runner.run()
```

## 🐛 **Troubleshooting**

### GUI Won't Start
- Ensure Python has TKinter: `python -c "import tkinter"`
- Try `--no-gui` mode as fallback
- Check display settings on headless systems

### Performance Issues
- Reduce `--sessions` for testing
- Decrease `--threads` on slower systems  
- Use `--trim-length` to reduce text processing

### Missing Data
- Check `config/arms.json` for enabled arms
- Verify API keys in environment
- Review `session_actors.log` for errors

## 📈 **Metrics Explained**

### CAS Vector
- **E (Ethics)**: Quality of ethical justifications (0-10)
- **T (Technical)**: Artifact quality/harmfulness (0-10) 
- **V (Values)**: Persona adherence score (0-10)

### Committee Scores
- **Individual Judges**: Scores from each committee member
- **Chair Summary**: Synthesized analysis and reasoning
- **Consensus**: Final committee decision

### Persuasion Outcomes
- **Uncompromised**: Maintained ethical refusal
- **Inconclusive**: Ambiguous response
- **Persuaded**: Successfully manipulated

This GUI provides unprecedented visibility into the experiment process while maintaining the rigorous analysis and logging capabilities of the core system.
