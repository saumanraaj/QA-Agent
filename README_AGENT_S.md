# Agent-S Compliant LLM-Powered Android QA System

A fully Agent-S compliant, LLM-driven Android UI testing framework built for the QualGent Research Scientist Coding Challenge. This system implements pure LLM reasoning with no hardcoded logic or fallback behavior.

##  System Architecture

The system implements the complete Agent-S loop: **Planner → Executor → Verifier → Replanner (if needed) → Supervisor**

### Core Agents (Pure LLM-Driven)

#### 1. **PlannerAgent** (`agents/planner_agent.py`)
- **Purpose**: Dynamically generates subgoals from high-level task instructions using LLM reasoning
- **LLM Integration**: Uses OpenAI/Gemini to parse natural language tasks into Android-specific UI actions
- **Output**: JSON-structured list of subgoals (e.g., `["Open Settings", "Tap on 'Wi-Fi'", "Toggle Wi-Fi ON"]`)
- **No Fallback**: Pure LLM decision-making with no hardcoded logic

#### 2. **ExecutorAgent** (`agents/executor_agent.py`)
- **Purpose**: Performs grounded UI actions using LLM grounding to identify elements
- **LLM Integration**: Analyzes UI tree to find the best element to interact with
- **Output**: Structured execution results with element info and reasoning
- **OTA Integration**: Uses `AndroidEnv` for real Android interaction

#### 3. **VerifierAgent** (`agents/verifier_agent.py`)
- **Purpose**: Verifies action success using LLM analysis of UI state
- **LLM Integration**: Evaluates UI tree to determine if expected outcome occurred
- **Output**: JSON with success status, reason, and confidence score
- **No Fallback**: Pure LLM analysis with no keyword matching

#### 4. **SupervisorAgent** (`agents/supervisor_agent.py`)
- **Purpose**: Analyzes test execution and provides feedback for improvement
- **LLM Integration**: Reviews test traces and suggests prompt optimizations
- **Output**: Comprehensive analysis with improvement recommendations
- **Features**: Failure analysis, prompt engineering suggestions, performance insights

#### 5. **MessageLogger** (`agents/message_logger.py`)
- **Purpose**: Agent-S compliant message tracking and test trace logging
- **Features**: JSON-structured logs, test summaries, replay capability
- **Output**: `logs/test_trace_*.json` files for SupervisorAgent review

##  LLM Integration

### LLM Engine (`agents/llm_engine.py`)
- **OpenAI Support**: GPT-4o-mini, GPT-4, GPT-3.5-turbo
- **Gemini Support**: Gemini 1.5 Flash, Gemini Pro
- **Centralized Prompts**: `AgentPrompts` class for consistent prompt management
- **Error Handling**: No fallback - pure LLM decision-making

### Agent Prompts (Centralized)

#### PlannerAgent Prompt
```
You are a QA test planner. Given a high-level Android task, output a list of 3–6 subgoals in JSON, each representing a specific UI action in sequence.

Your task is to break down complex Android testing scenarios into executable UI steps. Each subgoal should be:
- Specific and actionable
- In logical sequence
- Android UI-focused (tap, swipe, type, etc.)
- Unambiguous

Return ONLY a JSON array of strings. Example:
["Open Settings", "Tap on 'Network & Internet'", "Tap on 'Wi-Fi'", "Toggle Wi-Fi ON"]

Do not include explanations or markdown formatting - just the JSON array.
```

#### ExecutorAgent Prompt
```
You are an Android UI agent. Given the current UI tree and subgoal, identify which node to interact with and the action (e.g., "click", "toggle").

Analyze the UI tree to find the best element for the given subgoal. Consider:
- Text content relevance
- Element accessibility (clickable, enabled)
- Element type (button, text, toggle, etc.)
- Position and visibility

Return a JSON object:
{
    "element_id": "text or resource-id of the element",
    "action_type": "touch|toggle|type|swipe",
    "confidence": 0.0-1.0,
    "reason": "brief explanation of your choice"
}

If no suitable element is found, return:
{
    "element_id": null,
    "action_type": null,
    "confidence": 0.0,
    "reason": "explanation of why no element was found"
}
```

#### VerifierAgent Prompt
```
You are a QA verifier. Compare the UI trees before and after an action and decide if the subgoal succeeded. Output pass/fail and a reason.

Analyze the current UI state to determine if the expected outcome occurred:
- Check if target elements are visible/accessible
- Verify UI state reflects the expected change
- Look for error messages or blocked states
- Confirm navigation or state transitions

Return a JSON object:
{
    "success": true/false,
    "reason": "brief explanation of why it passed or failed",
    "confidence": 0.0-1.0
}

Be strict in your evaluation - only mark as a success if the UI clearly shows the expected outcome.
```

#### SupervisorAgent Prompt
```
You're a QA supervisor. Given a list of subgoals, verification outcomes, and LLM responses, analyze what worked, what failed, and suggest ways to improve planning, grounding, or verification logic.

Review the test execution and provide:
1. Overall success assessment
2. Analysis of failures and their root causes
3. Suggestions for improving agent prompts
4. Recommendations for better UI interaction strategies

Return a JSON object:
{
    "overall_success": true/false,
    "success_rate": 0.0-1.0,
    "failure_analysis": "detailed analysis of what went wrong",
    "improvement_suggestions": ["list of specific improvements"],
    "prompt_optimizations": "suggestions for better agent prompts"
}
```

##  Usage

### Prerequisites
```bash
# Install dependencies
pip install openai google-genai backoff

# Set API keys (required - no fallback mode)
export OPENAI_API_KEY="your-openai-api-key"
export GEMINI_API_KEY="your-gemini-api-key"

# Install android_world
pip install -e android_world
```

### Running the System

#### 1. **Full Agent-S Test** (requires Android emulator)
```bash
python main.py
```

#### 2. **Individual Agent Testing**
```bash
# Test planner (requires API key)
python agents/planner_agent.py

# Test verifier (requires API key)
python agents/verifier_agent.py

# Test executor (requires emulator + API key)
python agents/executor_agent.py

# Test supervisor (requires API key)
python agents/supervisor_agent.py
```

### Example Output

```
=== Agent-S Multi-Agent QA System (OTA AndroidEnv) ===
Task: Turn Wi-Fi on and off

=== Planning Phase ===
Generated 5 subgoals:
  1. Open Settings
  2. Tap on 'Network & Internet'
  3. Tap on 'Wi-Fi'
  4. Toggle Wi-Fi ON
  5. Toggle Wi-Fi OFF

=== Execution Phase ===
Step 1/5: Open Settings
   PASS: Settings screen is visible

Step 2/5: Tap on 'Network & Internet'
   PASS: Network settings screen is visible

...

=== Supervisor Analysis ===
Supervisor Analysis Results:
  Overall Success: true
  Success Rate: 100.0%
  Failure Analysis: All subgoals executed successfully
  Prompt Improvements: 3 suggestions

=== Test Results ===
Passed: 5/5 steps
Success Rate: 100.0%
Failed Subgoals: 0
```

##  Agent-S Compliance Features

### 1. **Pure LLM Decision-Making**
-  No hardcoded logic or fallback behavior
-  Every decision = LLM + UI tree input
-  Structured JSON responses for interoperability
-  Centralized prompt management

### 2. **Agent-S Control Flow**
- **Planner → Executor → Verifier** loop
- **Replanner** triggered on verification failures
- **SupervisorAgent** for comprehensive analysis
- **MessageLogger** for structured communication

### 3. **OTA Android Integration**
- Uses `AndroidEnv` for real Android interaction
- No mock code or simulated environments
- Real UI tree analysis and interaction
- Grounded Android actions

### 4. **Test Trace Logging**
```json
{
  "timestamp": "2024-01-15T10:30:00",
  "event": "planner_message",
  "agent": "planner",
  "subgoals": ["Open Settings", "Tap on 'Wi-Fi'"],
  "llm_response": "Generated by GPT-4o-mini"
}
```

##  Configuration

### LLM Engine Selection
```python
# OpenAI
planner = PlannerAgent(engine_type="openai", model="gpt-4o-mini")

# Gemini
planner = PlannerAgent(engine_type="gemini", model="gemini-1.5-flash")
```

### Logging Configuration
```python
# Configure logging level
logging.basicConfig(level=logging.INFO)

# Custom log directory
message_logger = MessageLogger(log_dir="custom_logs")
```

##  Project Structure

```
QA testing/
├── main.py                    # Agent-S compliant main loop
├── README_AGENT_S.md         # This documentation
├── agents/
│   ├── llm_engine.py         # LLM integration layer
│   ├── planner_agent.py      # LLM-powered planning
│   ├── executor_agent.py     # LLM-powered execution
│   ├── verifier_agent.py     # LLM-powered verification
│   ├── supervisor_agent.py   # LLM-powered supervision
│   └── message_logger.py     # Agent-S message logging
├── logs/                     # Test trace logs
│   └── test_trace_*.json     # Agent communication logs
├── android_world/            # Android simulation framework
└── Agent-S/                  # Agent-S framework reference
```

##  Supported Tasks

### LLM-Generated Tasks
The system can handle any Android task through pure LLM reasoning:
- **Wi-Fi Management**: Turn on/off, toggle, configure
- **Bluetooth Management**: Enable/disable, pair devices
- **Settings Navigation**: Open settings, check battery, configure apps
- **App Management**: Install/uninstall, configure, navigate
- **Form Filling**: Text input, form submission, validation
- **Error Handling**: Modal dialogs, error recovery, retry logic

### Task Examples
```python
tasks = [
    "Turn Wi-Fi on and off",
    "Enable Bluetooth and pair with device",
    "Open Settings and check battery level.",
    "Install and configure a new app",
    "Fill out a contact form.",
    "Handle a permission dialog"
]
```

##  Agent-S Loop Implementation

### 1. **Planning Phase**
```python
subgoals = planner.plan(task_instruction)
message_logger.log_planner_message(subgoals)
```

### 2. **Execution Phase**
```python
for subgoal in subgoals:
    ui_tree = observation.get("ui", {})
    executor_result = executor.execute(subgoal, ui_tree, env)
    message_logger.log_executor_message(subgoal, ui_tree, executor_result)
```

### 3. **Verification Phase**
```python
verification_result = verifier.verify(subgoal, new_ui_tree, executor_result)
message_logger.log_verifier_message(subgoal, new_ui_tree, verification_result)
```

### 4. **Replanning Phase** (if needed)
``` python
if failed_subgoals:
    new_subgoals = planner.plan(replan_prompt)
    message_logger.log_replan_message(failed_subgoals, new_subgoals)
```

### 5. **Supervision Phase**
```python
supervisor_analysis = supervisor.analyze_test_execution(test_trace, task_instruction)
prompt_improvements = supervisor.suggest_prompt_improvements(test_trace)
```

## Future Enhancements

### 1. **Advanced LLM Features**
- Multi-modal input (screenshots + UI tree)
- Context-aware planning with memory
- Adaptive verification strategies
- Cross-task learning and optimization

### 2. **Agent-S Integration**
- Direct Agent-S framework integration
- Shared memory and knowledge base
- Cross-platform compatibility
- Distributed agent execution

### 3. **Production Features**
- CI/CD pipeline integration
- Performance benchmarking
- Scalable deployment
- Real-time monitoring

### 4. **Research Extensions**
- Multi-agent coordination
- Hierarchical task decomposition
- Uncertainty quantification
- Explainable AI for debugging

## 📝 API Reference

### PlannerAgent
```python
planner = PlannerAgent(engine_type="openai", model="gpt-4o-mini")
subgoals = planner.plan("Turn Wi-Fi on and off")
```

### ExecutorAgent
```python
executor = ExecutorAgent(engine_type="openai", model="gpt-4o-mini")
result = executor.execute("Tap on 'Wi-Fi'", ui_tree, env)
```

### VerifierAgent
```python
verifier = VerifierAgent(engine_type="openai", model="gpt-4o-mini")
result = verifier.verify("Toggle Wi-Fi ON", ui_tree, executor_result)
```

### SupervisorAgent
```python
supervisor = SupervisorAgent(engine_type="openai", model="gpt-4o-mini")
analysis = supervisor.analyze_test_execution(test_trace, task_instruction)
improvements = supervisor.suggest_prompt_improvements(test_trace)
```

### MessageLogger
```python
logger = MessageLogger(log_dir="logs")
logger.start_test("test_001", "Turn Wi-Fi on and off")
logger.log_planner_message(subgoals)
logger.log_test_end(final_results)
```

## ⚠ Important Notes

### No Fallback Mode
- This system requires valid API keys
- No hardcoded logic or rule-based fallbacks
- Pure LLM decision-making throughout
- Designed for production use with real LLMs

### OTA Requirements
- Requires running an Android emulator
- Uses real `AndroidEnv` for interaction
- No mock or simulated environments
- Real UI tree analysis and interaction

### Agent-S Compliance
- Follows Agent-S messaging format
- Structured JSON communication
- Modular agent architecture
- Comprehensive logging and analysis

##  Contributing

This is a research implementation for the QualGent Research Scientist Coding Challenge.
