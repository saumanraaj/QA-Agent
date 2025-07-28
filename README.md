# Multi-Agent QA System Based on Agent-S Architecture

A comprehensive multi-agent LLM-powered system that functions as a full-stack mobile QA team, built on top of Agent-S and android_world.

## Overview

This project implements a complete multi-agent QA system with four specialized agents that collaboratively test Android applications:

- **Planner Agent**: Parses high-level QA goals and decomposes them into actionable subgoals
- **Executor Agent**: Executes subgoals in the Android UI environment with grounded mobile gestures
- **Verifier Agent**: Determines whether the app behaves as expected after each step
- **Supervisor Agent**: Reviews entire test episodes and proposes improvements

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Planner       │    │   Executor      │    │   Verifier      │    │   Supervisor    │
│   Agent         │───▶│   Agent         │───▶│   Agent         │───▶│   Agent         │
│                 │    │                 │    │                 │    │                 │
│ • Parse goals   │    │ • UI grounding  │    │ • Pass/fail     │    │ • Test analysis │
│ • Decompose     │    │ • Touch/type    │    │ • Bug detection │    │ • Improvements  │
│ • Replan        │    │ • Scroll        │    │ • Recovery      │    │ • Coverage      │
└─────────────────┘    └─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │                       │
         ▼                       ▼                       ▼                       ▼
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                              Android World Environment                                 │
│                                                                                       │
│ • AndroidEnv integration                                                              │
│ • Real device simulation                                                              │
│ • Visual trace generation                                                             │
│ • JSON logging system                                                                │
└─────────────────────────────────────────────────────────────────────────────────────────┘
```

## Features

### Core Capabilities
- **LLM-Powered Agents**: Each agent uses advanced LLM reasoning for decision making
- **Dynamic Replanning**: Automatic recovery when subgoals fail
- **Visual Trace Generation**: Frame-by-frame UI screenshots for analysis
- **Comprehensive Logging**: JSON-structured logs of all interactions
- **Error Recovery**: Intelligent fallback strategies and learning

### Advanced Features
- **Android in the Wild Integration**: Real user session data for training and evaluation
- **Multi-Strategy Execution**: Fuzzy matching, semantic matching, and contextual reasoning
- **Learning Capabilities**: Agents learn from failures and improve over time
- **Comprehensive Evaluation**: Accuracy, robustness, and generalization metrics

## Requirements

### Prerequisites
- Python 3.8+
- Android SDK
- Android Emulator or physical device
- OpenAI API key (for LLM agents)

### Dependencies
```bash
pip install -r requirements.txt
```

## Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd multi-agent-qa-system
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Setup API keys**
```bash
cp env_example.txt .env
# Edit .env with your API keys
```

4. **Setup Android environment**
```bash
# Start Android emulator or connect device
adb devices
```

## Usage

### Basic QA Test
```bash
python main.py --task "Turn Wi-Fi on and off" --task_name "settings_wifi"
```

### Android in the Wild Integration
```bash
# Run with real user session data
python android_in_the_wild_simple_integration.py
```

### Custom Task
```bash
python main.py --task "Set an alarm for 8 AM" --task_name "clock_alarm"
```

## Project Structure

```
multi-agent-qa-system/
├── agents/                          # Core agent implementations
│   ├── planner_agent.py            # Goal decomposition and planning
│   ├── executor_agent.py           # UI interaction and execution
│   ├── verifier_agent.py           # Result verification and validation
│   ├── supervisor_agent.py         # Test analysis and improvement
│   ├── llm_engine.py              # LLM integration and prompts
│   └── message_logger.py           # Structured logging system
├── android_world/                  # Android environment integration
├── Agent-S/                        # Agent-S framework
├── main.py                         # Main execution pipeline
├── requirements.txt                # Python dependencies
├── .gitignore                      # Git ignore rules
└── README.md                       # This file
```

## Configuration

### Environment Variables
```bash
OPENAI_API_KEY=your_openai_api_key
GEMINI_API_KEY=your_gemini_api_key
ANDROID_SDK_ROOT=/path/to/android/sdk
```

### Task Configuration
```python
# Supported task names
TASKS = {
    "settings_wifi": "Wi-Fi settings testing",
    "clock_alarm": "Clock alarm functionality",
    "email_search": "Email search and filtering"
}
```

## Evaluation

### Metrics
- **Accuracy**: Action type matching and coordinate precision
- **Robustness**: Ability to handle UI variations and complexity
- **Generalization**: Performance across different task types
- **Completion Rate**: Percentage of steps successfully completed

### Sample Results
```
=== QA Test Results ===
Task: Turn Wi-Fi on and off
Total Steps: 4
Passed Steps: 4
Success Rate: 100%
Recovery Attempts: 0
Overall Status: PASSED
```

## Advanced Features

### Android in the Wild Integration
The system integrates with the Android in the Wild dataset for enhanced training and evaluation:

- **Real User Sessions**: 715,142 episodes from thousands of Android apps
- **Task Prompt Generation**: LLM analysis of user goals
- **Ground Truth Comparison**: Agent vs real user traces
- **Comprehensive Scoring**: Accuracy, robustness, generalization

### Dynamic Replanning
When a subgoal fails, the system automatically:

1. **Analyzes the failure** using the Verifier Agent
2. **Generates alternative strategies** using the Planner Agent
3. **Attempts recovery** with different approaches
4. **Learns from failures** to improve future performance

### Learning Capabilities
Agents continuously learn and improve:

- **Element Variations**: Learn different UI element representations
- **Failure Patterns**: Identify and avoid common failure modes
- **Success Strategies**: Remember and reuse successful approaches
- **Contextual Learning**: Adapt to different app contexts

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request


## Acknowledgments

- **Agent-S**: Modular agent architecture framework
- **Android World**: Android environment simulation
- **Android in the Wild**: Real user session dataset
- **OpenAI**: LLM integration and reasoning capabilities


**Built for the QualGent Research Scientist coding challenge** 
