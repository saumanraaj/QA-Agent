import json
import logging
from typing import List, Dict, Any
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agents.llm_engine import get_llm_engine, AgentPrompts

logger = logging.getLogger(__name__)

class PlannerAgent:
    """LLM-powered planner agent with advanced recovery and replanning logic"""
    
    def __init__(self, engine_type: str = "openai", model: str = "gpt-4o-mini"):
        self.llm_engine = get_llm_engine(engine_type=engine_type, model=model)
        self.recovery_history = []  # Track recovery attempts
        self.failure_patterns = {}  # Learn from failure patterns
        self.max_recovery_attempts = 3  # Maximum recovery attempts per subgoal
        
    def plan(self, task_instruction: str) -> List[str]:
        """Generate subgoals from high-level task using LLM reasoning"""
        
        system_prompt = AgentPrompts.get_planner_prompt()

        user_prompt = f"""Task: {task_instruction}

Generate a list of Android UI subgoals to accomplish this task:"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        try:
            response = self.llm_engine.generate(messages, temperature=0.1, max_tokens=500)
            
            # Parse JSON response
            try:
                subgoals = json.loads(response.strip())
                if isinstance(subgoals, list) and all(isinstance(goal, str) for goal in subgoals):
                    logger.info(f"Generated {len(subgoals)} subgoals for task: {task_instruction}")
                    return subgoals
                else:
                    raise ValueError("Response is not a list of strings")
            except json.JSONDecodeError:
                # Try to extract JSON from response if it contains extra text
                import re
                json_match = re.search(r'\[.*\]', response, re.DOTALL)
                if json_match:
                    subgoals = json.loads(json_match.group())
                    logger.info(f"Generated {len(subgoals)} subgoals for task: {task_instruction}")
                    return subgoals
                else:
                    raise ValueError("Could not parse JSON from LLM response")
                    
        except Exception as e:
            logger.error(f"LLM planning failed: {e}")
            raise ValueError(f"LLM planning failed for task '{task_instruction}': {e}")

    def replan(self, task_instruction: str, failed_subgoal: str, failure_reason: str, 
               current_ui_state: Dict[str, Any] = None, execution_context: Dict[str, Any] = None) -> List[str]:
        """Dynamically replan when a subgoal fails during execution"""
        
        system_prompt = AgentPrompts.get_replanner_prompt()

        # Build context for replanning
        context_info = f"Original task: {task_instruction}\n"
        context_info += f"Failed subgoal: {failed_subgoal}\n"
        context_info += f"Failure reason: {failure_reason}\n"
        
        if current_ui_state:
            ui_texts = self._extract_ui_texts(current_ui_state)
            context_info += f"Current UI elements: {ui_texts}\n"
        
        if execution_context:
            context_info += f"Execution context: {execution_context}\n"

        user_prompt = f"""{context_info}

Generate alternative subgoals to handle this failure. Consider:
1. Different approaches to achieve the same goal
2. Handling pop-ups, dialogs, or unexpected UI states
3. Alternative navigation paths
4. Error recovery strategies

Provide 2-4 alternative subgoals:"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        try:
            response = self.llm_engine.generate(messages, temperature=0.2, max_tokens=400)
            
            # Parse JSON response
            try:
                alternative_subgoals = json.loads(response.strip())
                if isinstance(alternative_subgoals, list) and all(isinstance(goal, str) for goal in alternative_subgoals):
                    logger.info(f"Generated {len(alternative_subgoals)} alternative subgoals for failed subgoal: {failed_subgoal}")
                    return alternative_subgoals
                else:
                    raise ValueError("Response is not a list of strings")
            except json.JSONDecodeError:
                # Try to extract JSON from response
                import re
                json_match = re.search(r'\[.*\]', response, re.DOTALL)
                if json_match:
                    alternative_subgoals = json.loads(json_match.group())
                    logger.info(f"Generated {len(alternative_subgoals)} alternative subgoals for failed subgoal: {failed_subgoal}")
                    return alternative_subgoals
                else:
                    raise ValueError("Could not parse JSON from LLM response")
                    
        except Exception as e:
            logger.error(f"LLM replanning failed: {e}")
            raise ValueError(f"LLM replanning failed for subgoal '{failed_subgoal}': {e}")

    def advanced_replan(self, task_instruction: str, failed_subgoal: str, failure_reason: str,
                       current_ui_state: Dict[str, Any] = None, execution_context: Dict[str, Any] = None,
                       recovery_attempt: int = 1) -> Dict[str, Any]:
        """Advanced replanning with recovery strategies and learning"""
        
        # Check if we've exceeded max recovery attempts
        if recovery_attempt > self.max_recovery_attempts:
            return {
                "success": False,
                "reason": f"Exceeded maximum recovery attempts ({self.max_recovery_attempts})",
                "alternatives": [],
                "strategy": "abort"
            }
        
        # Analyze failure pattern
        failure_pattern = self._analyze_failure_pattern(failed_subgoal, failure_reason, current_ui_state)
        
        # Choose recovery strategy based on pattern and attempt number
        recovery_strategy = self._choose_recovery_strategy(failure_pattern, recovery_attempt)
        
        # Generate alternatives using the chosen strategy
        alternatives = self._generate_strategic_alternatives(
            task_instruction, failed_subgoal, failure_reason, 
            current_ui_state, execution_context, recovery_strategy
        )
        
        # Log recovery attempt
        recovery_log = {
            "timestamp": self._get_timestamp(),
            "failed_subgoal": failed_subgoal,
            "failure_reason": failure_reason,
            "failure_pattern": failure_pattern,
            "recovery_attempt": recovery_attempt,
            "recovery_strategy": recovery_strategy,
            "alternatives": alternatives,
            "ui_state": current_ui_state,
            "context": execution_context
        }
        self.recovery_history.append(recovery_log)
        
        return {
            "success": True,
            "alternatives": alternatives,
            "strategy": recovery_strategy,
            "pattern": failure_pattern,
            "attempt": recovery_attempt
        }

    def _analyze_failure_pattern(self, failed_subgoal: str, failure_reason: str, 
                                ui_state: Dict[str, Any] = None) -> Dict[str, Any]:
        """Analyze failure pattern to determine recovery approach"""
        
        pattern = {
            "type": "unknown",
            "confidence": 0.0,
            "indicators": [],
            "suggested_strategies": []
        }
        
        # Analyze failure reason for patterns
        failure_lower = failure_reason.lower()
        
        if any(word in failure_lower for word in ["pop-up", "dialog", "blocking", "modal"]):
            pattern["type"] = "popup_blocking"
            pattern["confidence"] = 0.9
            pattern["indicators"].append("popup_detected")
            pattern["suggested_strategies"] = ["dismiss_popup", "alternative_path", "wait_retry"]
            
        elif any(word in failure_lower for word in ["permission", "allow", "grant", "access"]):
            pattern["type"] = "permission_required"
            pattern["confidence"] = 0.85
            pattern["indicators"].append("permission_request")
            pattern["suggested_strategies"] = ["grant_permission", "skip_permission", "manual_navigation"]
            
        elif any(word in failure_lower for word in ["not found", "missing", "element", "clickable"]):
            pattern["type"] = "element_not_found"
            pattern["confidence"] = 0.8
            pattern["indicators"].append("ui_element_missing")
            pattern["suggested_strategies"] = ["alternative_element", "different_path", "retry_later"]
            
        elif any(word in failure_lower for word in ["network", "connection", "timeout", "error"]):
            pattern["type"] = "network_error"
            pattern["confidence"] = 0.75
            pattern["indicators"].append("network_issue")
            pattern["suggested_strategies"] = ["retry", "offline_mode", "alternative_method"]
            
        elif any(word in failure_lower for word in ["timeout", "slow", "loading", "wait"]):
            pattern["type"] = "timing_issue"
            pattern["confidence"] = 0.7
            pattern["indicators"].append("timing_problem")
            pattern["suggested_strategies"] = ["wait_longer", "retry", "skip_step"]
        
        # Analyze UI state for additional indicators
        if ui_state:
            ui_texts = self._extract_ui_texts(ui_state)
            for text in ui_texts:
                if any(word in text.lower() for word in ["error", "failed", "unavailable"]):
                    pattern["indicators"].append("ui_error_state")
                elif any(word in text.lower() for word in ["loading", "please wait"]):
                    pattern["indicators"].append("loading_state")
        
        return pattern

    def _choose_recovery_strategy(self, failure_pattern: Dict[str, Any], attempt: int) -> str:
        """Choose recovery strategy based on failure pattern and attempt number"""
        
        pattern_type = failure_pattern.get("type", "unknown")
        suggested_strategies = failure_pattern.get("suggested_strategies", [])
        
        # Strategy selection logic based on attempt number and pattern
        if attempt == 1:
            # First attempt: try the most appropriate strategy
            if suggested_strategies:
                return suggested_strategies[0]
            else:
                return "retry"
        elif attempt == 2:
            # Second attempt: try alternative strategy
            if len(suggested_strategies) > 1:
                return suggested_strategies[1]
            else:
                return "alternative_path"
        else:
            # Third attempt: try fallback strategy
            if len(suggested_strategies) > 2:
                return suggested_strategies[2]
            else:
                return "skip_step"

    def _generate_strategic_alternatives(self, task_instruction: str, failed_subgoal: str,
                                       failure_reason: str, current_ui_state: Dict[str, Any],
                                       execution_context: Dict[str, Any], strategy: str) -> List[str]:
        """Generate alternatives based on specific recovery strategy"""
        
        system_prompt = AgentPrompts.get_strategic_replanner_prompt(strategy)
        
        context_info = f"Original task: {task_instruction}\n"
        context_info += f"Failed subgoal: {failed_subgoal}\n"
        context_info += f"Failure reason: {failure_reason}\n"
        context_info += f"Recovery strategy: {strategy}\n"
        
        if current_ui_state:
            ui_texts = self._extract_ui_texts(current_ui_state)
            context_info += f"Current UI elements: {ui_texts}\n"
        
        if execution_context:
            context_info += f"Execution context: {execution_context}\n"

        user_prompt = f"""{context_info}

Generate alternative subgoals using the '{strategy}' recovery strategy:"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        try:
            response = self.llm_engine.generate(messages, temperature=0.3, max_tokens=300)
            
            # Parse JSON response
            try:
                alternatives = json.loads(response.strip())
                if isinstance(alternatives, list) and all(isinstance(goal, str) for goal in alternatives):
                    return alternatives
                else:
                    raise ValueError("Response is not a list of strings")
            except json.JSONDecodeError:
                # Try to extract JSON from response
                import re
                json_match = re.search(r'\[.*\]', response, re.DOTALL)
                if json_match:
                    return json.loads(json_match.group())
                else:
                    # Fallback to basic alternatives
                    return self._generate_fallback_alternatives(failed_subgoal, strategy)
                    
        except Exception as e:
            logger.error(f"Strategic replanning failed: {e}")
            return self._generate_fallback_alternatives(failed_subgoal, strategy)

    def _generate_fallback_alternatives(self, failed_subgoal: str, strategy: str) -> List[str]:
        """Generate fallback alternatives when LLM fails"""
        
        fallback_strategies = {
            "dismiss_popup": ["Dismiss any pop-up dialog", "Tap 'OK' or 'Cancel' on dialog"],
            "alternative_path": ["Try alternative navigation path", "Use different menu option"],
            "retry": ["Retry the same action", "Wait and try again"],
            "skip_step": ["Skip this step if possible", "Continue to next step"],
            "manual_navigation": ["Navigate manually to target", "Use back button and try different path"]
        }
        
        return fallback_strategies.get(strategy, ["Retry the action", "Try alternative approach"])

    def get_recovery_statistics(self) -> Dict[str, Any]:
        """Get statistics about recovery attempts and success rates"""
        
        if not self.recovery_history:
            return {"total_attempts": 0, "success_rate": 0.0, "patterns": {}}
        
        total_attempts = len(self.recovery_history)
        successful_recoveries = sum(1 for log in self.recovery_history if log.get("success", False))
        
        # Analyze patterns
        pattern_counts = {}
        for log in self.recovery_history:
            pattern_type = log.get("pattern", {}).get("type", "unknown")
            pattern_counts[pattern_type] = pattern_counts.get(pattern_type, 0) + 1
        
        return {
            "total_attempts": total_attempts,
            "successful_recoveries": successful_recoveries,
            "success_rate": (successful_recoveries / total_attempts) * 100 if total_attempts > 0 else 0,
            "patterns": pattern_counts,
            "recovery_history": self.recovery_history
        }

    def _extract_ui_texts(self, ui_tree: Dict[str, Any]) -> List[str]:
        """Extract text elements from UI tree for context"""
        texts = []
        
        def extract_texts(node):
            if isinstance(node, dict):
                if 'text' in node and node['text']:
                    texts.append(node['text'])
                for key, value in node.items():
                    if isinstance(value, (dict, list)):
                        extract_texts(value)
            elif isinstance(node, list):
                for item in node:
                    extract_texts(item)
        
        extract_texts(ui_tree)
        return texts[:10]  # Limit to first 10 texts for context

    def _get_timestamp(self) -> str:
        """Get current timestamp in ISO format"""
        from datetime import datetime
        return datetime.now().isoformat()

if __name__ == "__main__":
    planner = PlannerAgent()
    task = "Turn Wi-Fi on and off"
    plan = planner.plan(task)
    print(f"Task: {task}")
    print("Plan:")
    for i, step in enumerate(plan, 1):
        print(f"  {i}. {step}") 