import json
import logging
import sys
import os
from typing import Dict, List, Any, Optional

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agents.llm_engine import get_llm_engine, AgentPrompts

logger = logging.getLogger(__name__)

class SupervisorAgent:
    """Enhanced LLM-powered supervisor agent for comprehensive test analysis and evaluation"""
    
    def __init__(self, engine_type: str = "gemini", model: str = "gemini-2.0-flash-exp"):
        """Initialize supervisor with Gemini 2.5 for enhanced analysis"""
        self.llm_engine = get_llm_engine(engine_type=engine_type, model=model)
        self.analysis_history = []
        self.evaluation_metrics = {}
        
    def comprehensive_analysis(self, test_trace: List[Dict[str, Any]], task_instruction: str, 
                             visual_traces: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Comprehensive analysis of test execution including visual traces using Gemini 2.5"""
        
        # Prepare comprehensive analysis data
        analysis_data = self._prepare_comprehensive_analysis(test_trace, visual_traces)
        
        # Generate comprehensive analysis using Gemini 2.5
        system_prompt = AgentPrompts.get_comprehensive_supervisor_prompt()
        
        user_prompt = f"""Task: {task_instruction}

Comprehensive Test Analysis Data:
{json.dumps(analysis_data, indent=2)}

Please provide a comprehensive analysis focusing on:

1. **PROMPT IMPROVEMENTS**: Analyze LLM prompt effectiveness and suggest specific improvements
2. **POOR PLANS/FAILURES**: Identify poorly designed plans and failure patterns
3. **TEST COVERAGE EXPANSION**: Recommend areas for test coverage improvement

Analyze this test execution comprehensively and provide detailed feedback:"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        try:
            response = self.llm_engine.generate(messages, temperature=0.1, max_tokens=800)
            
            # Parse JSON response
            try:
                # Clean the response - remove any markdown formatting
                cleaned_response = response.strip()
                if cleaned_response.startswith("```json"):
                    cleaned_response = cleaned_response[7:]
                if cleaned_response.endswith("```"):
                    cleaned_response = cleaned_response[:-3]
                cleaned_response = cleaned_response.strip()
                
                result = json.loads(cleaned_response)
                if isinstance(result, dict) and "overall_success" in result:
                    # Store analysis in history
                    analysis_record = {
                        "timestamp": self._get_timestamp(),
                        "task": task_instruction,
                        "analysis": result,
                        "metrics": self._calculate_evaluation_metrics(result, analysis_data)
                    }
                    self.analysis_history.append(analysis_record)
                    
                    logger.info(f"Comprehensive supervisor analysis completed: {result['overall_success']}")
                    return result
                else:
                    raise ValueError("Response missing 'overall_success' field")
            except json.JSONDecodeError as e:
                print(f"JSON parsing error: {e}")
                print(f"Raw response: {response[:500]}...")
                # Try to extract JSON from response
                import re
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if json_match:
                    try:
                        result = json.loads(json_match.group())
                        logger.info(f"Comprehensive supervisor analysis completed: {result.get('overall_success', False)}")
                        return result
                    except json.JSONDecodeError:
                        raise ValueError("Could not parse JSON from LLM response")
                else:
                    raise ValueError("Could not parse JSON from LLM response")
                    
        except Exception as e:
            logger.error(f"Comprehensive supervisor analysis failed: {e}")
            raise ValueError(f"Comprehensive supervisor analysis failed: {e}")
    
    def analyze_test_execution(self, test_trace: List[Dict[str, Any]], task_instruction: str) -> Dict[str, Any]:
        """Analyze test execution and provide comprehensive feedback"""
        
        system_prompt = AgentPrompts.get_supervisor_prompt()
        
        # Prepare test execution summary
        execution_summary = self._prepare_execution_summary(test_trace)
        
        user_prompt = f"""Task: {task_instruction}

Test Execution Summary:
{json.dumps(execution_summary, indent=2)}

Analyze this test execution and provide feedback:"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        try:
            response = self.llm_engine.generate(messages, temperature=0.1, max_tokens=800)
            
            # Parse JSON response
            try:
                result = json.loads(response.strip())
                if isinstance(result, dict) and "overall_success" in result:
                    logger.info(f"Supervisor analysis completed: {result['overall_success']}")
                    return result
                else:
                    raise ValueError("Response missing 'overall_success' field")
            except json.JSONDecodeError:
                # Try to extract JSON from response
                import re
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if json_match:
                    result = json.loads(json_match.group())
                    logger.info(f"Supervisor analysis completed: {result.get('overall_success', False)}")
                    return result
                else:
                    raise ValueError("Could not parse JSON from LLM response")
                    
        except Exception as e:
            logger.error(f"Supervisor analysis failed: {e}")
            raise ValueError(f"Supervisor analysis failed: {e}")
    
    def _prepare_execution_summary(self, test_trace: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Prepare a structured summary of test execution for analysis"""
        
        summary = {
            "total_messages": len(test_trace),
            "planner_messages": [],
            "executor_messages": [],
            "verifier_messages": [],
            "replan_messages": [],
            "successful_executions": 0,
            "successful_verifications": 0,
            "failed_executions": 0,
            "failed_verifications": 0,
            "llm_responses": [],
            "error_patterns": []
        }
        
        for message in test_trace:
            event = message.get("event", "")
            
            if event == "planner_message":
                summary["planner_messages"].append({
                    "subgoals": message.get("subgoals", []),
                    "llm_response": message.get("llm_response")
                })
                if message.get("llm_response"):
                    summary["llm_responses"].append(f"Planner: {message['llm_response'][:100]}...")
                    
            elif event == "executor_message":
                executor_data = {
                    "subgoal": message.get("subgoal", ""),
                    "executor_result": message.get("executor_result"),
                    "llm_response": message.get("llm_response")
                }
                summary["executor_messages"].append(executor_data)
                
                if message.get("executor_result"):
                    summary["successful_executions"] += 1
                else:
                    summary["failed_executions"] += 1
                    summary["error_patterns"].append(f"Executor failed: {message.get('subgoal', 'Unknown')}")
                
                if message.get("llm_response"):
                    summary["llm_responses"].append(f"Executor: {message['llm_response'][:100]}...")
                    
            elif event == "verifier_message":
                verifier_data = {
                    "subgoal": message.get("subgoal", ""),
                    "verification_result": message.get("verification_result", {}),
                    "llm_response": message.get("llm_response")
                }
                summary["verifier_messages"].append(verifier_data)
                
                if message.get("verification_result", {}).get("success", False):
                    summary["successful_verifications"] += 1
                else:
                    summary["failed_verifications"] += 1
                    summary["error_patterns"].append(f"Verification failed: {message.get('subgoal', 'Unknown')}")
                
                if message.get("llm_response"):
                    summary["llm_responses"].append(f"Verifier: {message['llm_response'][:100]}...")
                    
            elif event == "replan_message":
                summary["replan_messages"].append({
                    "failed_subgoal": message.get("failed_subgoal", ""),
                    "new_subgoals": message.get("new_subgoals", []),
                    "llm_response": message.get("llm_response")
                })
                if message.get("llm_response"):
                    summary["llm_responses"].append(f"Replanner: {message['llm_response'][:100]}...")
        
        # Calculate success rates
        total_executions = summary["successful_executions"] + summary["failed_executions"]
        total_verifications = summary["successful_verifications"] + summary["failed_verifications"]
        
        summary["execution_success_rate"] = summary["successful_executions"] / total_executions if total_executions > 0 else 0
        summary["verification_success_rate"] = summary["successful_verifications"] / total_verifications if total_verifications > 0 else 0
        
        return summary
    
    def suggest_prompt_improvements(self, test_trace: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze LLM responses and suggest prompt improvements"""
        
        system_prompt = """You are a prompt engineering expert. Analyze the LLM responses from different agents and suggest specific improvements to their prompts.

Focus on:
1. Clarity and specificity of instructions
2. JSON output formatting requirements
3. Error handling and edge cases
4. Context and examples provided

Return a JSON object with:
{
    "planner_improvements": ["specific suggestions for planner prompt"],
    "executor_improvements": ["specific suggestions for executor prompt"],
    "verifier_improvements": ["specific suggestions for verifier prompt"],
    "overall_recommendations": ["general prompt engineering advice"]
}"""

        # Extract LLM responses for analysis
        llm_responses = []
        for message in test_trace:
            if message.get("llm_response"):
                llm_responses.append({
                    "agent": message.get("agent", "unknown"),
                    "response": message["llm_response"],
                    "event": message.get("event", "")
                })
        
        user_prompt = f"""LLM Responses Analysis:
{json.dumps(llm_responses, indent=2)}

Suggest prompt improvements based on these responses:"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        try:
            response = self.llm_engine.generate(messages, temperature=0.1, max_tokens=600)
            
            try:
                result = json.loads(response.strip())
                return result
            except json.JSONDecodeError:
                import re
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if json_match:
                    return json.loads(json_match.group())
                else:
                    raise ValueError("Could not parse JSON from LLM response")
                    
        except Exception as e:
            logger.error(f"Prompt improvement analysis failed: {e}")
            return {
                "planner_improvements": ["Unable to analyze - LLM error"],
                "executor_improvements": ["Unable to analyze - LLM error"],
                "verifier_improvements": ["Unable to analyze - LLM error"],
                "overall_recommendations": ["Check LLM connectivity and API keys"]
            }

    def _prepare_comprehensive_analysis(self, test_trace: List[Dict[str, Any]], 
                                      visual_traces: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Prepare comprehensive analysis data including visual traces"""
        
        # Basic execution summary
        execution_summary = self._prepare_execution_summary(test_trace)
        
        # Enhanced analysis data
        comprehensive_data = {
            "execution_summary": execution_summary,
            "agent_performance": self._analyze_agent_performance(test_trace),
            "recovery_analysis": self._analyze_recovery_patterns(test_trace),
            "failure_patterns": self._analyze_failure_patterns(test_trace),
            "visual_analysis": self._analyze_visual_traces(visual_traces) if visual_traces else {},
            "prompt_effectiveness": self._analyze_prompt_effectiveness(test_trace),
            "test_coverage": self._analyze_test_coverage(test_trace),
            "timing_analysis": self._analyze_timing_patterns(test_trace)
        }
        
        return comprehensive_data

    def _analyze_agent_performance(self, test_trace: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze individual agent performance"""
        
        planner_performance = {"total_plans": 0, "successful_plans": 0, "plan_quality": 0.0}
        executor_performance = {"total_executions": 0, "successful_executions": 0, "execution_accuracy": 0.0}
        verifier_performance = {"total_verifications": 0, "successful_verifications": 0, "verification_accuracy": 0.0}
        
        for message in test_trace:
            event = message.get("event", "")
            
            if event == "planner_message":
                planner_performance["total_plans"] += 1
                subgoals = message.get("subgoals", [])
                if len(subgoals) > 0:
                    planner_performance["successful_plans"] += 1
                    
            elif event == "executor_message":
                executor_performance["total_executions"] += 1
                if message.get("executor_result"):
                    executor_performance["successful_executions"] += 1
                    
            elif event == "verifier_message":
                verifier_performance["total_verifications"] += 1
                if message.get("verification_result", {}).get("success", False):
                    verifier_performance["successful_verifications"] += 1
        
        # Calculate accuracy rates
        if planner_performance["total_plans"] > 0:
            planner_performance["plan_quality"] = planner_performance["successful_plans"] / planner_performance["total_plans"]
        if executor_performance["total_executions"] > 0:
            executor_performance["execution_accuracy"] = executor_performance["successful_executions"] / executor_performance["total_executions"]
        if verifier_performance["total_verifications"] > 0:
            verifier_performance["verification_accuracy"] = verifier_performance["successful_verifications"] / verifier_performance["total_verifications"]
        
        return {
            "planner": planner_performance,
            "executor": executor_performance,
            "verifier": verifier_performance
        }

    def _analyze_recovery_patterns(self, test_trace: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze recovery patterns and effectiveness"""
        
        recovery_attempts = [msg for msg in test_trace if msg.get("event") == "replan_message"]
        
        recovery_analysis = {
            "total_recovery_attempts": len(recovery_attempts),
            "successful_recoveries": 0,
            "recovery_success_rate": 0.0,
            "recovery_strategies": {},
            "failure_patterns": {},
            "recovery_effectiveness": {}
        }
        
        for recovery in recovery_attempts:
            failed_subgoal = recovery.get("failed_subgoal", "")
            alternatives = recovery.get("new_subgoals", [])
            
            # Check if any alternative was successful
            for event in test_trace:
                if (event.get("event") == "executor_message" and 
                    event.get("subgoal") in alternatives and 
                    event.get("executor_result")):
                    recovery_analysis["successful_recoveries"] += 1
                    break
            
            # Analyze failure context
            failure_context = recovery.get("failure_context", {})
            if failure_context:
                context = failure_context.get("context", {})
                failure_type = context.get("failure_type", "unknown")
                recovery_analysis["failure_patterns"][failure_type] = recovery_analysis["failure_patterns"].get(failure_type, 0) + 1
        
        if recovery_analysis["total_recovery_attempts"] > 0:
            recovery_analysis["recovery_success_rate"] = recovery_analysis["successful_recoveries"] / recovery_analysis["total_recovery_attempts"]
        
        return recovery_analysis

    def _analyze_failure_patterns(self, test_trace: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze failure patterns and root causes"""
        
        failures = []
        failure_categories = {
            "execution_failures": 0,
            "verification_failures": 0,
            "planning_failures": 0,
            "recovery_failures": 0
        }
        
        for message in test_trace:
            event = message.get("event", "")
            
            if event == "executor_message" and not message.get("executor_result"):
                failure_categories["execution_failures"] += 1
                failures.append({
                    "type": "execution",
                    "subgoal": message.get("subgoal", ""),
                    "reason": "Executor could not find or interact with target element"
                })
                
            elif event == "verifier_message" and not message.get("verification_result", {}).get("success", False):
                failure_categories["verification_failures"] += 1
                failures.append({
                    "type": "verification",
                    "subgoal": message.get("subgoal", ""),
                    "reason": message.get("verification_result", {}).get("reason", "Verification failed")
                })
        
        return {
            "total_failures": len(failures),
            "failure_categories": failure_categories,
            "failure_details": failures,
            "most_common_failure": max(failure_categories.items(), key=lambda x: x[1])[0] if failures else "none"
        }

    def _analyze_visual_traces(self, visual_traces: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze visual traces for UI state changes and issues"""
        
        if not visual_traces:
            return {"total_screenshots": 0, "ui_issues_detected": 0, "visual_analysis": "No visual traces available"}
        
        visual_analysis = {
            "total_screenshots": len(visual_traces),
            "ui_issues_detected": 0,
            "ui_state_changes": 0,
            "visual_issues": [],
            "screenshot_quality": "good"
        }
        
        for trace in visual_traces:
            # Analyze screenshot metadata
            if trace.get("description"):
                description = trace["description"].lower()
                if any(word in description for word in ["error", "failed", "blocked", "popup"]):
                    visual_analysis["ui_issues_detected"] += 1
                    visual_analysis["visual_issues"].append({
                        "screenshot_id": trace.get("step", "unknown"),
                        "issue": "UI issue detected",
                        "description": trace.get("description", "")
                    })
        
        return visual_analysis

    def _analyze_prompt_effectiveness(self, test_trace: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze prompt effectiveness based on LLM responses"""
        
        prompt_analysis = {
            "total_llm_calls": 0,
            "successful_llm_calls": 0,
            "llm_response_quality": 0.0,
            "prompt_issues": [],
            "response_patterns": {}
        }
        
        for message in test_trace:
            if message.get("llm_response"):
                prompt_analysis["total_llm_calls"] += 1
                
                # Analyze response quality
                response = message.get("llm_response", "")
                if response and len(response) > 10:  # Basic quality check
                    prompt_analysis["successful_llm_calls"] += 1
                
                # Check for common issues
                if "error" in response.lower() or "failed" in response.lower():
                    prompt_analysis["prompt_issues"].append({
                        "agent": message.get("agent", "unknown"),
                        "issue": "LLM response indicates error",
                        "response": response[:100] + "..." if len(response) > 100 else response
                    })
        
        if prompt_analysis["total_llm_calls"] > 0:
            prompt_analysis["llm_response_quality"] = prompt_analysis["successful_llm_calls"] / prompt_analysis["total_llm_calls"]
        
        return prompt_analysis

    def _analyze_test_coverage(self, test_trace: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze test coverage and completeness"""
        
        coverage_analysis = {
            "total_subgoals": 0,
            "executed_subgoals": 0,
            "verified_subgoals": 0,
            "coverage_percentage": 0.0,
            "missing_coverage": [],
            "coverage_gaps": []
        }
        
        # Extract subgoals from planner messages
        planner_messages = [msg for msg in test_trace if msg.get("event") == "planner_message"]
        if planner_messages:
            all_subgoals = planner_messages[0].get("subgoals", [])
            coverage_analysis["total_subgoals"] = len(all_subgoals)
            
            # Check which subgoals were executed
            executed_subgoals = set()
            for msg in test_trace:
                if msg.get("event") == "executor_message":
                    executed_subgoals.add(msg.get("subgoal", ""))
            
            coverage_analysis["executed_subgoals"] = len(executed_subgoals)
            
            # Check which subgoals were verified
            verified_subgoals = set()
            for msg in test_trace:
                if msg.get("event") == "verifier_message" and msg.get("verification_result", {}).get("success", False):
                    verified_subgoals.add(msg.get("subgoal", ""))
            
            coverage_analysis["verified_subgoals"] = len(verified_subgoals)
            
            # Calculate coverage percentage
            if coverage_analysis["total_subgoals"] > 0:
                coverage_analysis["coverage_percentage"] = (coverage_analysis["executed_subgoals"] / coverage_analysis["total_subgoals"]) * 100
            
            # Identify missing coverage
            missing_subgoals = set(all_subgoals) - executed_subgoals
            coverage_analysis["missing_coverage"] = list(missing_subgoals)
        
        return coverage_analysis

    def _analyze_timing_patterns(self, test_trace: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze timing patterns and performance"""
        
        timing_analysis = {
            "total_duration": 0.0,
            "average_step_duration": 0.0,
            "slowest_steps": [],
            "timing_issues": []
        }
        
        if len(test_trace) >= 2:
            # Calculate total duration
            start_time = test_trace[0].get("timestamp", "")
            end_time = test_trace[-1].get("timestamp", "")
            
            if start_time and end_time:
                try:
                    from datetime import datetime
                    start_dt = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
                    end_dt = datetime.fromisoformat(end_time.replace('Z', '+00:00'))
                    timing_analysis["total_duration"] = (end_dt - start_dt).total_seconds()
                except:
                    timing_analysis["total_duration"] = 0.0
            
            # Calculate average step duration
            execution_steps = [msg for msg in test_trace if msg.get("event") == "executor_message"]
            if execution_steps:
                timing_analysis["average_step_duration"] = timing_analysis["total_duration"] / len(execution_steps)
        
        return timing_analysis

    def _calculate_evaluation_metrics(self, analysis_result: Dict[str, Any], analysis_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate comprehensive evaluation metrics"""
        
        metrics = {
            "bug_detection_accuracy": 0.0,
            "agent_recovery_ability": 0.0,
            "supervisor_feedback_effectiveness": 0.0,
            "overall_system_performance": 0.0
        }
        
        # Bug detection accuracy (based on failure detection and analysis)
        failure_patterns = analysis_data.get("failure_patterns", {})
        total_failures = failure_patterns.get("total_failures", 0)
        if total_failures > 0:
            # Higher score for better failure analysis
            failure_analysis_quality = len(analysis_result.get("failure_analysis", "")) / 100  # Normalize
            metrics["bug_detection_accuracy"] = min(1.0, failure_analysis_quality)
        
        # Agent recovery ability (based on recovery success rate)
        recovery_analysis = analysis_data.get("recovery_analysis", {})
        recovery_success_rate = recovery_analysis.get("recovery_success_rate", 0.0)
        metrics["agent_recovery_ability"] = recovery_success_rate
        
        # Supervisor feedback effectiveness (based on analysis quality)
        feedback_quality = len(analysis_result.get("improvement_suggestions", [])) / 10  # Normalize
        metrics["supervisor_feedback_effectiveness"] = min(1.0, feedback_quality)
        
        # Overall system performance (weighted average)
        agent_performance = analysis_data.get("agent_performance", {})
        planner_quality = agent_performance.get("planner", {}).get("plan_quality", 0.0)
        executor_accuracy = agent_performance.get("executor", {}).get("execution_accuracy", 0.0)
        verifier_accuracy = agent_performance.get("verifier", {}).get("verification_accuracy", 0.0)
        
        metrics["overall_system_performance"] = (
            planner_quality * 0.3 + 
            executor_accuracy * 0.4 + 
            verifier_accuracy * 0.3
        )
        
        return metrics

    def generate_evaluation_report(self, task_instruction: str, test_trace: List[Dict[str, Any]], 
                                 visual_traces: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Generate comprehensive evaluation report"""
        
        # Perform comprehensive analysis
        analysis_result = self.comprehensive_analysis(test_trace, task_instruction, visual_traces)
        
        # Generate evaluation report
        report = {
            "metadata": {
                "report_id": f"eval_report_{self._get_timestamp().replace(':', '-')}",
                "generated_at": self._get_timestamp(),
                "task": task_instruction,
                "supervisor_version": "2.0",
                "llm_model": "gemini-2.0-flash-exp"
            },
            "executive_summary": {
                "overall_success": analysis_result.get("overall_success", False),
                "success_rate": analysis_result.get("success_rate", 0.0),
                "key_findings": analysis_result.get("key_findings", []),
                "recommendations": analysis_result.get("recommendations", [])
            },
            "evaluation_metrics": {
                "bug_detection_accuracy": analysis_result.get("bug_detection_accuracy", 0.0),
                "agent_recovery_ability": analysis_result.get("agent_recovery_ability", 0.0),
                "supervisor_feedback_effectiveness": analysis_result.get("supervisor_feedback_effectiveness", 0.0),
                "overall_system_performance": analysis_result.get("overall_system_performance", 0.0)
            },
            "detailed_analysis": analysis_result,
            "test_coverage_analysis": analysis_result.get("test_coverage_analysis", {}),
            "recovery_analysis": analysis_result.get("recovery_analysis", {}),
            "prompt_improvements": analysis_result.get("prompt_improvements", {}),
            "visual_analysis": analysis_result.get("visual_analysis", {}),
            "recommendations": {
                "immediate_actions": analysis_result.get("immediate_actions", []),
                "long_term_improvements": analysis_result.get("long_term_improvements", []),
                "test_coverage_expansion": analysis_result.get("test_coverage_expansion", [])
            }
        }
        
        return report

    def _get_timestamp(self) -> str:
        """Get current timestamp in ISO format"""
        from datetime import datetime
        return datetime.now().isoformat()

if __name__ == "__main__":
    # Test the supervisor agent
    supervisor = SupervisorAgent()
    
    # Mock test trace
    mock_trace = [
        {
            "timestamp": "2024-01-15T10:30:00",
            "event": "planner_message",
            "agent": "planner",
            "subgoals": ["Open Settings", "Tap on 'Wi-Fi'"],
            "llm_response": "Generated by GPT-4o-mini"
        },
        {
            "timestamp": "2024-01-15T10:30:05",
            "event": "executor_message",
            "agent": "executor",
            "subgoal": "Open Settings",
            "executor_result": {"status": "success"},
            "llm_response": "Found Settings app icon"
        }
    ]
    
    try:
        result = supervisor.analyze_test_execution(mock_trace, "Turn Wi-Fi on and off")
        print("Supervisor Analysis Result:")
        print(json.dumps(result, indent=2))
    except Exception as e:
        print(f"Supervisor test failed: {e}") 