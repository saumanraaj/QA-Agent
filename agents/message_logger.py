import json
import logging
import os
from datetime import datetime
from typing import Dict, List, Any, Optional

# Import for screenshot functionality (QualGent requirement)
try:
    import base64
    import io
    from PIL import Image
    import numpy as np
    SCREENSHOT_AVAILABLE = True
except ImportError:
    SCREENSHOT_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("Screenshot functionality not available - missing PIL or numpy")

logger = logging.getLogger(__name__)

class MessageLogger:
    """Agent-S compliant message logger for tracking agent communications"""
    
    def __init__(self, log_dir: str = "logs"):
        self.log_dir = log_dir
        self.test_trace = []
        self.current_test_id = None
        
        # Create logs directory if it doesn't exist
        os.makedirs(log_dir, exist_ok=True)
    
    def start_test(self, test_id: str, task_instruction: str):
        """Start a new test session"""
        self.current_test_id = test_id
        self.test_trace = []
        
        test_start = {
            "timestamp": datetime.now().isoformat(),
            "event": "test_start",
            "test_id": test_id,
            "task_instruction": task_instruction
        }
        self.test_trace.append(test_start)
        
        logger.info(f"Started test: {test_id} - {task_instruction}")
    
    def log_planner_message(self, subgoals: List[str], llm_response: Optional[str] = None):
        """Log planner agent message"""
        message = {
            "timestamp": datetime.now().isoformat(),
            "event": "planner_message",
            "agent": "planner",
            "subgoals": subgoals,
            "llm_response": llm_response
        }
        self.test_trace.append(message)
        
        logger.info(f"Planner generated {len(subgoals)} subgoals")
    
    def log_executor_message(self, subgoal: str, ui_tree: Dict, executor_result: Optional[Dict] = None, context: Optional[Dict] = None, llm_response: Optional[str] = None):
        """Log executor agent message with optional context"""
        message = {
            "timestamp": datetime.now().isoformat(),
            "event": "executor_message",
            "agent": "executor",
            "subgoal": subgoal,
            "ui_tree_summary": self._summarize_ui_tree(ui_tree),
            "executor_result": executor_result,
            "context": context,
            "llm_response": llm_response
        }
        self.test_trace.append(message)
        
        if executor_result:
            logger.info(f"Executor executed: {subgoal}")
        else:
            logger.warning(f"Executor failed: {subgoal}")
    
    def log_verifier_message(self, subgoal: str, ui_tree: Dict, verification_result: Dict, context: Optional[Dict] = None, llm_response: Optional[str] = None):
        """Log verifier agent message with optional context"""
        message = {
            "timestamp": datetime.now().isoformat(),
            "event": "verifier_message",
            "agent": "verifier",
            "subgoal": subgoal,
            "ui_tree_summary": self._summarize_ui_tree(ui_tree),
            "verification_result": verification_result,
            "context": context,
            "llm_response": llm_response
        }
        self.test_trace.append(message)
        
        success = verification_result.get("success", False)
        reason = verification_result.get("reason", "No reason provided")
        logger.info(f"Verifier result for '{subgoal}': {'PASS' if success else 'FAIL'} - {reason}")
    
    def log_screenshot(self, screenshot: Any, step_number: int, description: str = ""):
        """Log screenshot for visual traces (QualGent challenge requirement)"""
        if not SCREENSHOT_AVAILABLE:
            logger.warning("Screenshot functionality is not available. Skipping screenshot logging.")
            return

        try:
            if isinstance(screenshot, np.ndarray):
                # Convert numpy array to PIL Image
                img = Image.fromarray(screenshot)
            else:
                img = screenshot
            
            # Save to bytes buffer
            buffer = io.BytesIO()
            img.save(buffer, format='PNG')
            img_bytes = buffer.getvalue()
            
            # Encode to base64
            img_base64 = base64.b64encode(img_bytes).decode('utf-8')
            
            message = {
                "timestamp": datetime.now().isoformat(),
                "event": "screenshot",
                "step_number": step_number,
                "description": description,
                "screenshot_base64": img_base64,
                "image_format": "PNG",
                "image_size": img.size if hasattr(img, 'size') else "unknown"
            }
            self.test_trace.append(message)
            
            logger.info(f"Screenshot logged for step {step_number}: {description}")
            
        except Exception as e:
            logger.error(f"Failed to log screenshot: {e}")
            # Log error but don't fail the test
            message = {
                "timestamp": datetime.now().isoformat(),
                "event": "screenshot_error",
                "step_number": step_number,
                "description": description,
                "error": str(e)
            }
            self.test_trace.append(message)
    
    def log_replan_message(self, failed_subgoal: str, new_subgoals: List[str], failure_context: Optional[Dict] = None, llm_response: Optional[str] = None):
        """Log replanning message with enhanced context"""
        message = {
            "timestamp": datetime.now().isoformat(),
            "event": "replan_message",
            "agent": "planner",
            "failed_subgoal": failed_subgoal,
            "new_subgoals": new_subgoals,
            "failure_context": failure_context,
            "llm_response": llm_response
        }
        self.test_trace.append(message)
        
        logger.info(f"Replanning after failed subgoal: {failed_subgoal}")
    
    def log_test_end(self, final_results: Dict[str, Any]):
        """Log test end with final results"""
        test_end = {
            "timestamp": datetime.now().isoformat(),
            "event": "test_end",
            "test_id": self.current_test_id,
            "final_results": final_results
        }
        self.test_trace.append(test_end)
        
        # Save test trace to file with enhanced JSON formatting
        if self.current_test_id:
            trace_file = os.path.join(self.log_dir, f"test_trace_{self.current_test_id}.json")
            
            # Create enhanced JSON structure
            enhanced_trace = {
                "metadata": {
                    "test_id": self.current_test_id,
                    "created_at": datetime.now().isoformat(),
                    "version": "1.0",
                    "format": "qa_test_trace"
                },
                "test_summary": {
                    "total_events": len(self.test_trace),
                    "event_types": self._get_event_type_counts(),
                    "duration": self._calculate_test_duration(),
                    "success_rate": final_results.get("success_rate", 0.0),
                    "passed_steps": final_results.get("passed_steps", 0),
                    "total_steps": final_results.get("total_steps", 0)
                },
                "events": self.test_trace
            }
            
            with open(trace_file, 'w') as f:
                json.dump(enhanced_trace, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Test completed. Enhanced trace saved to: {trace_file}")
            
            # Also save a summary file
            summary_file = os.path.join(self.log_dir, f"test_summary_{self.current_test_id}.json")
            with open(summary_file, 'w') as f:
                json.dump(enhanced_trace["test_summary"], f, indent=2, ensure_ascii=False)
            
            logger.info(f"Test summary saved to: {summary_file}")

    def _get_event_type_counts(self) -> Dict[str, int]:
        """Get counts of different event types"""
        event_counts = {}
        for event in self.test_trace:
            event_type = event.get("event", "unknown")
            event_counts[event_type] = event_counts.get(event_type, 0) + 1
        return event_counts

    def _calculate_test_duration(self) -> float:
        """Calculate test duration in seconds"""
        if len(self.test_trace) < 2:
            return 0.0
        
        start_time = datetime.fromisoformat(self.test_trace[0]["timestamp"])
        end_time = datetime.fromisoformat(self.test_trace[-1]["timestamp"])
        return (end_time - start_time).total_seconds()
    
    def _summarize_ui_tree(self, ui_tree: Dict) -> Dict[str, Any]:
        """Create a summary of UI tree for logging"""
        texts = []
        clickable_elements = []
        
        def extract_summary(node):
            if isinstance(node, dict):
                if 'text' in node and node['text']:
                    texts.append(node['text'])
                if node.get('clickable', False):
                    clickable_elements.append({
                        "text": node.get('text', ''),
                        "bounds": node.get('bounds', [])
                    })
                for key, value in node.items():
                    if isinstance(value, (dict, list)):
                        extract_summary(value)
            elif isinstance(node, list):
                for item in node:
                    extract_summary(item)
        
        extract_summary(ui_tree)
        
        return {
            "text_elements_count": len(texts),
            "clickable_elements_count": len(clickable_elements),
            "sample_texts": texts[:5],  # First 5 text elements
            "clickable_elements": clickable_elements[:3]  # First 3 clickable elements
        }
    
    def get_test_summary(self) -> Dict[str, Any]:
        """Get summary of current test"""
        if not self.test_trace:
            return {}
        
        planner_messages = [msg for msg in self.test_trace if msg.get("event") == "planner_message"]
        executor_messages = [msg for msg in self.test_trace if msg.get("event") == "executor_message"]
        verifier_messages = [msg for msg in self.test_trace if msg.get("event") == "verifier_message"]
        replan_messages = [msg for msg in self.test_trace if msg.get("event") == "replan_message"]
        
        successful_executions = sum(1 for msg in executor_messages if msg.get("executor_result"))
        successful_verifications = sum(1 for msg in verifier_messages if msg.get("verification_result", {}).get("success", False))
        
        # Calculate recovery statistics
        recovery_stats = self._calculate_recovery_statistics()
        
        return {
            "total_subgoals": len(planner_messages[0].get("subgoals", [])) if planner_messages else 0,
            "executed_subgoals": successful_executions,
            "verified_subgoals": successful_verifications,
            "total_messages": len(self.test_trace),
            "recovery_attempts": len(replan_messages),
            "recovery_statistics": recovery_stats
        }

    def _calculate_recovery_statistics(self) -> Dict[str, Any]:
        """Calculate recovery statistics from test trace"""
        replan_messages = [msg for msg in self.test_trace if msg.get("event") == "replan_message"]
        
        if not replan_messages:
            return {
                "total_recovery_attempts": 0,
                "successful_recoveries": 0,
                "recovery_success_rate": 0.0,
                "failure_patterns": {},
                "recovery_strategies": {}
            }
        
        total_attempts = len(replan_messages)
        successful_recoveries = 0
        failure_patterns = {}
        recovery_strategies = {}
        
        for replan_msg in replan_messages:
            # Count successful recoveries (subsequent successful executions)
            failed_subgoal = replan_msg.get("failed_subgoal", "")
            alternatives = replan_msg.get("new_subgoals", [])
            
            # Check if any alternative was successful
            for event in self.test_trace:
                if (event.get("event") == "executor_message" and 
                    event.get("subgoal") in alternatives and 
                    event.get("executor_result")):
                    successful_recoveries += 1
                    break
            
            # Analyze failure patterns
            failure_context = replan_msg.get("failure_context", {})
            if failure_context:
                pattern_type = failure_context.get("context", {}).get("failure_type", "unknown")
                failure_patterns[pattern_type] = failure_patterns.get(pattern_type, 0) + 1
        
        return {
            "total_recovery_attempts": total_attempts,
            "successful_recoveries": successful_recoveries,
            "recovery_success_rate": (successful_recoveries / total_attempts) * 100 if total_attempts > 0 else 0,
            "failure_patterns": failure_patterns,
            "recovery_strategies": recovery_strategies
        }

    def export_recovery_logs(self, planner_agent) -> Dict[str, Any]:
        """Export recovery logs in JSON format"""
        recovery_stats = planner_agent.get_recovery_statistics()
        
        recovery_logs = {
            "metadata": {
                "exported_at": datetime.now().isoformat(),
                "planner_type": "advanced_replanner",
                "version": "1.0"
            },
            "recovery_statistics": recovery_stats,
            "detailed_recovery_history": recovery_stats.get("recovery_history", []),
            "test_context": {
                "test_id": self.current_test_id,
                "total_events": len(self.test_trace),
                "recovery_events": len([e for e in self.test_trace if e.get("event") == "replan_message"])
            }
        }
        
        # Save recovery logs to file
        if self.current_test_id:
            recovery_file = os.path.join(self.log_dir, f"recovery_logs_{self.current_test_id}.json")
            with open(recovery_file, 'w') as f:
                json.dump(recovery_logs, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Recovery logs exported to: {recovery_file}")
        
        return recovery_logs 