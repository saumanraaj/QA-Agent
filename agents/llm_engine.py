import os
import json
import logging
from typing import Dict, List, Optional, Any
import backoff
from openai import OpenAI, APIConnectionError, APIError, RateLimitError
from google import genai
from google.genai import types

logger = logging.getLogger(__name__)

class LLMEngine:
    """Base LLM engine following Agent-S patterns"""
    
    def __init__(self, model: str = "gpt-4o", api_key: Optional[str] = None):
        self.model = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key required. Set OPENAI_API_KEY environment variable.")
        
        self.client = OpenAI(api_key=self.api_key)
    
    @backoff.on_exception(
        backoff.expo,
        (APIConnectionError, APIError, RateLimitError),
        max_time=60
    )
    def generate(self, messages: List[Dict[str, str]], temperature: float = 0.0, max_tokens: int = 1000) -> str:
        """Generate response from LLM"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            raise

class GeminiEngine:
    """Gemini LLM engine as alternative"""
    
    def __init__(self, model: str = "gemini-1.5-flash", api_key: Optional[str] = None):
        self.model = model
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("Gemini API key required. Set GEMINI_API_KEY environment variable.")
        
        try:
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel(model)
        except ImportError:
            raise ValueError("google-generativeai package not installed. Run: pip install google-generativeai")
        except Exception as e:
            raise ValueError(f"Failed to initialize Gemini: {e}")
    
    def generate(self, messages: List[Dict[str, str]], temperature: float = 0.0, max_tokens: int = 1000) -> str:
        """Generate response from Gemini"""
        try:
            import google.generativeai as genai
            from google.generativeai import types
            
            # Convert OpenAI format to Gemini format
            prompt = ""
            for msg in messages:
                if msg["role"] == "user":
                    prompt += f"User: {msg['content']}\n"
                elif msg["role"] == "assistant":
                    prompt += f"Assistant: {msg['content']}\n"
                elif msg["role"] == "system":
                    prompt += f"System: {msg['content']}\n"
            
            response = self.model.generate_content(
                prompt,
                generation_config=types.GenerationConfig(
                    temperature=temperature,
                    max_output_tokens=max_tokens
                )
            )
            return response.text
        except Exception as e:
            logger.error(f"Gemini generation failed: {e}")
            raise

def get_llm_engine(engine_type: str = "openai", **kwargs) -> Any:
    """Factory function to get LLM engine"""
    if engine_type == "openai":
        return LLMEngine(**kwargs)
    elif engine_type == "gemini":
        return GeminiEngine(**kwargs)
    else:
        raise ValueError(f"Unsupported engine type: {engine_type}")

# Agent-specific prompt templates
class AgentPrompts:
    """Centralized prompt templates for all agents"""
    
    @staticmethod
    def get_planner_prompt() -> str:
        return """You are a QA test planner. Given a high-level Android task, output a list of 3–6 subgoals in JSON, each representing a specific UI action in sequence.

Your task is to break down complex Android testing scenarios into executable UI steps. Each subgoal should be:
- Specific and actionable
- In logical sequence
- Android UI-focused (tap, swipe, type, etc.)
- Clear and unambiguous

Return ONLY a JSON array of strings. Example:
["Open Settings", "Tap on 'Network & Internet'", "Tap on 'Wi-Fi'", "Toggle Wi-Fi ON"]

Do not include explanations or markdown formatting - just the JSON array."""

    @staticmethod
    def get_executor_prompt() -> str:
        return """You are an Android UI agent. Given the current UI tree and subgoal, identify which node to interact with and the action (e.g. "touch", "input_text", "scroll").

Analyze the UI tree to find the best element for the given subgoal. Consider:
- Text content relevance
- Element accessibility (clickable, enabled)
- Element type (button, text, toggle, etc.)
- Position and visibility

Return a JSON object:
{
    "element_id": "text or resource-id of the element",
    "action_type": "touch|input_text|scroll|navigate_back|navigate_home",
    "confidence": 0.0-1.0,
    "reason": "brief explanation of your choice"
}

If no suitable element is found, return:
{
    "element_id": null,
    "action_type": null,
    "confidence": 0.0,
    "reason": "explanation of why no element was found"
}"""

    @staticmethod
    def get_verifier_prompt() -> str:
        return """You are a QA verifier. Compare the UI trees before and after an action and decide if the subgoal succeeded. Output pass/fail and a reason.

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

Be strict in your evaluation - only mark as success if the UI clearly shows the expected outcome."""

    @staticmethod
    def get_supervisor_prompt() -> str:
        return """You're a QA supervisor. Given a list of subgoals, verification outcomes, and LLM responses, analyze what worked, what failed, and suggest ways to improve planning, grounding, or verification logic.

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
}"""

    @staticmethod
    def get_replanner_prompt() -> str:
        return """You are a dynamic replanner for Android UI testing. When a subgoal fails, you need to generate alternative approaches to handle the failure.

Your role is to:
1. Analyze the failure reason and current UI state
2. Generate alternative subgoals that can achieve the same objective
3. Handle common failure scenarios like:
   - Pop-up dialogs blocking the path
   - UI elements not found or not clickable
   - Navigation issues or unexpected screens
   - Permission requests or system dialogs
   - Network connectivity issues

Consider these strategies:
- Dismiss pop-ups or dialogs first
- Try alternative navigation paths
- Use different UI elements that serve the same purpose
- Handle permission requests
- Retry with different timing or approach
- Skip problematic steps if possible

Return ONLY a JSON array of 2-4 alternative subgoals as strings. Each subgoal should be:
- Specific and actionable
- Address the specific failure reason
- Provide a clear alternative approach

Example: ["Dismiss pop-up dialog", "Try alternative navigation path", "Handle permission request"]"""

    @staticmethod
    def get_strategic_replanner_prompt(strategy: str) -> str:
        """Get strategy-specific replanner prompt"""
        
        strategy_prompts = {
            "dismiss_popup": """You are handling a pop-up dialog that is blocking the target element. Focus on dismissing the pop-up first.

Strategies:
- Look for 'OK', 'Cancel', 'Allow', 'Deny' buttons on the pop-up
- Tap the appropriate button to dismiss the dialog
- If multiple options, choose the one that allows progress

Return ONLY a JSON array of 2-3 subgoals focused on dismissing the pop-up:
Example: ["Tap 'OK' on pop-up dialog", "Tap 'Allow' to dismiss dialog"]""",

            "alternative_path": """You need to find an alternative path to achieve the goal since the current path is blocked.

Strategies:
- Look for different menu options that lead to the same destination
- Try using quick settings or shortcuts
- Use back navigation and try a different approach
- Look for similar functionality in different locations

Return ONLY a JSON array of 2-3 subgoals for alternative navigation:
Example: ["Try quick settings menu", "Use back button and try different path"]""",

            "retry": """The action failed but might succeed with a retry. Focus on retrying with slight modifications.

Strategies:
- Wait a moment and retry the same action
- Try the action with different timing
- Check if the UI state has changed and retry
- Use a slightly different approach to the same goal

Return ONLY a JSON array of 2-3 retry subgoals:
Example: ["Wait 2 seconds and retry", "Try tapping the element again"]""",

            "skip_step": """The current step is problematic and should be skipped if possible to continue with the task.

Strategies:
- Skip this step if it's not essential
- Look for ways to achieve the goal without this step
- Continue to the next step in the sequence
- Find an alternative approach that bypasses this issue

Return ONLY a JSON array of 2-3 skip/alternative subgoals:
Example: ["Skip this step and continue", "Try alternative method to achieve goal"]""",

            "grant_permission": """A permission request is blocking progress. Focus on granting the required permission.

Strategies:
- Look for 'Allow', 'Grant', 'OK' buttons on permission dialogs
- Tap the appropriate permission button
- Handle any additional permission screens
- Continue after permission is granted

Return ONLY a JSON array of 2-3 permission handling subgoals:
Example: ["Tap 'Allow' on permission dialog", "Grant the required permission"]""",

            "manual_navigation": """Manual navigation is needed to reach the target. Focus on step-by-step navigation.

Strategies:
- Navigate through menus manually
- Use back button to return to previous screen
- Try different navigation paths
- Look for the target in different locations

Return ONLY a JSON array of 2-3 manual navigation subgoals:
Example: ["Navigate to Settings manually", "Use back button and try different menu"]"""
        }
        
        return strategy_prompts.get(strategy, """Generate alternative subgoals to handle the failure. Focus on finding a way to achieve the goal despite the current issue.

Return ONLY a JSON array of 2-3 alternative subgoals as strings.""")

    @staticmethod
    def get_comprehensive_supervisor_prompt() -> str:
        return """You are an advanced QA supervisor using Gemini 2.5 to analyze comprehensive test execution data including visual traces, agent performance, recovery patterns, and failure analysis.

Your PRIMARY FOCUS is on THREE KEY CAPABILITIES:

1. **PROMPT IMPROVEMENTS**: Analyze LLM prompt effectiveness and suggest specific improvements
2. **POOR PLANS/FAILURES**: Identify poorly designed plans and failure patterns
3. **TEST COVERAGE EXPANSION**: Recommend areas for test coverage improvement

Analyze the test execution data and provide:
- Overall success assessment
- Key findings and failure analysis
- Agent performance evaluation
- Recovery pattern analysis
- Evaluation metrics

Analyze the provided test execution data comprehensively and return a detailed JSON response with:

{
    "overall_success": true,
    "success_rate": 0.5,
    "key_findings": ["Sample finding"],
    "failure_analysis": "Sample failure analysis",
    "agent_performance": "Sample analysis",
    "recovery_analysis": "Sample analysis",
    "bug_detection_accuracy": 0.5,
    "agent_recovery_ability": 0.5,
    "supervisor_feedback_effectiveness": 0.5,
    "overall_system_performance": 0.5,
    "prompt_improvements": ["Sample improvement"],
    "poor_plans_identified": ["Sample poor plan"],
    "failure_patterns": ["Sample failure pattern"],
    "test_coverage_analysis": {
        "coverage_percentage": 50.0,
        "missing_coverage": ["Sample missing coverage"],
        "expansion_recommendations": ["Sample recommendation"]
    },
    "recommendations": ["Sample recommendation"]
}

Focus on providing actionable insights and specific, measurable recommendations for improvement.

IMPORTANT: Return ONLY valid JSON. Do not include any text before or after the JSON object. Ensure all strings are properly quoted and all arrays/objects are properly formatted.""" 