import logging
import uuid
from datetime import datetime
from android_world.env import env_launcher
from agents.planner_agent import PlannerAgent
from agents.executor_agent import ExecutorAgent
from agents.verifier_agent import VerifierAgent
from agents.supervisor_agent import SupervisorAgent
from agents.message_logger import MessageLogger

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def _get_failure_context(test_trace, failed_subgoal):
    """Extract failure context from test trace for replanning"""
    context = {
        'reason': 'Unknown failure',
        'ui_state': None,
        'context': {}
    }
    
    # Find the most recent executor and verifier messages for this subgoal
    for message in reversed(test_trace):
        if message.get('event') == 'executor_message' and message.get('subgoal') == failed_subgoal:
            context['ui_state'] = message.get('ui_tree_summary', {})
            context['context']['executor_result'] = message.get('executor_result')
            break
        elif message.get('event') == 'verifier_message' and message.get('subgoal') == failed_subgoal:
            context['reason'] = message.get('verification_result', {}).get('reason', 'Verification failed')
            break
    
    return context

def run_agent_s_qa_test(task_instruction: str = "Turn Wi-Fi on and off", task_name: str = "settings_wifi"):
    """Run Agent-S compliant QA test with LLM-powered agents using OTA AndroidEnv"""
    
    print("=== Agent-S Multi-Agent QA System (OTA AndroidEnv) ===")
    print(f"Task: {task_instruction}")
    print(f"Task Name: {task_name}")
    print()
    
    # Initialize message logger
    test_id = f"qa_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:8]}"
    message_logger = MessageLogger()
    message_logger.start_test(test_id, task_instruction)
    
    try:
        print("Initializing Android environment...")
        # Use AndroidEnv with specific task name as required by QualGent challenge
        from android_world.env import env_launcher
        
        # QualGent challenge format: env = AndroidEnv(task_name="settings_wifi")
        env = env_launcher.load_and_setup_env(
            task_name=task_name,  # Specific task name as required
            console_port=5554,
            emulator_setup=False,
            freeze_datetime=True
        )
        
        # QualGent challenge format: obs = env.reset()
        observation = env.reset(go_home=True)
        
        # Capture initial screenshot for visual traces (QualGent requirement)
        try:
            initial_screenshot = env.render(mode="rgb_array")
            message_logger.log_screenshot(initial_screenshot, 0, "Initial state")
        except Exception as e:
            logger.warning(f"Could not capture initial screenshot: {e}")
        
        print("Environment initialized successfully")
        print()
        
        # Initialize LLM-powered agents
        planner = PlannerAgent()
        executor = ExecutorAgent()
        verifier = VerifierAgent()
        supervisor = SupervisorAgent()
        
        # Agent-S Loop: Planner → Executor → Verifier → Replanner (if needed) → Supervisor
        print("=== Planning Phase ===")
        subgoals = planner.plan(task_instruction)
        message_logger.log_planner_message(subgoals)
        
        print(f"Generated {len(subgoals)} subgoals:")
        for i, subgoal in enumerate(subgoals, 1):
            print(f"  {i}. {subgoal}")
        print()
        
        passed_steps = 0
        total_steps = len(subgoals)
        failed_subgoals = []
        
        print("=== Execution Phase ===")
        for i, subgoal in enumerate(subgoals, 1):
            print(f"Step {i}/{total_steps}: {subgoal}")
            
            try:
                # Get current UI tree from OTA AndroidEnv
                ui_tree = observation.get("ui", {})
                if not ui_tree:
                    print("  FAIL: No UI tree available in observation")
                    failed_subgoals.append(subgoal)
                    continue
                
                # Executor Agent: Execute subgoal using OTA AndroidEnv
                executor_result = executor.execute(subgoal, ui_tree, env)
                message_logger.log_executor_message(subgoal, ui_tree, executor_result)
                
                if executor_result is None:
                    print("  FAIL: Executor could not find target element")
                    failed_subgoals.append(subgoal)
                    continue
                
                # Update observation with executor result from OTA AndroidEnv
                observation = executor_result.get("observation", observation)
                new_ui_tree = observation.get("ui", {})
                
                # Capture screenshot after action for visual traces (QualGent requirement)
                try:
                    # Use the screenshot from the observation if available
                    if "pixels" in observation:
                        action_screenshot = observation["pixels"]
                    else:
                        action_screenshot = env.render(mode="rgb_array")
                    message_logger.log_screenshot(action_screenshot, i, f"After executing: {subgoal}")
                except Exception as e:
                    logger.warning(f"Could not capture screenshot after action: {e}")
                
                # Verifier Agent: Verify execution success
                verification_result = verifier.verify(subgoal, new_ui_tree, executor_result)
                message_logger.log_verifier_message(subgoal, new_ui_tree, verification_result)
                
                if verification_result.get("success", False):
                    print(f"  PASS: {verification_result.get('reason', 'No reason provided')}")
                    passed_steps += 1
                else:
                    print(f"  FAIL: {verification_result.get('reason', 'Verification failed')}")
                    failed_subgoals.append(subgoal)
                    
            except Exception as e:
                print(f"  FAIL: Error during execution - {e}")
                failed_subgoals.append(subgoal)
                logger.error(f"Step {i} failed: {e}")
                
            print()
        
        # Dynamic Replanning Phase
        if failed_subgoals:
            print("=== Dynamic Replanning Phase ===")
            print(f"Attempting to replan for {len(failed_subgoals)} failed subgoals...")
            
            replanning_successes = 0
            total_replanning_attempts = 0
            
            for failed_subgoal in failed_subgoals:
                print(f"\nReplanning for failed subgoal: {failed_subgoal}")
                
                try:
                    # Get failure context from test trace
                    failure_context = _get_failure_context(message_logger.test_trace, failed_subgoal)
                    
                    # Generate alternative subgoals using dynamic replanning
                    alternative_subgoals = planner.replan(
                        task_instruction=task_instruction,
                        failed_subgoal=failed_subgoal,
                        failure_reason=failure_context.get('reason', 'Unknown failure'),
                        current_ui_state=failure_context.get('ui_state'),
                        execution_context=failure_context.get('context', {})
                    )
                    
                    message_logger.log_replan_message(failed_subgoal, alternative_subgoals, failure_context)
                    
                    print(f"Generated {len(alternative_subgoals)} alternative subgoals:")
                    for i, alt_subgoal in enumerate(alternative_subgoals, 1):
                        print(f"  {i}. {alt_subgoal}")
                    
                    # Try to execute alternative subgoals
                    subgoal_success = False
                    for alt_subgoal in alternative_subgoals:
                        total_replanning_attempts += 1
                        print(f"\n  Trying alternative: {alt_subgoal}")
                        
                        try:
                            # Get current UI tree
                            ui_tree = observation.get("ui", {})
                            if not ui_tree:
                                print("    FAIL: No UI tree available")
                                continue
                            
                            # Execute alternative subgoal
                            executor_result = executor.execute(alt_subgoal, ui_tree, env)
                            message_logger.log_executor_message(alt_subgoal, ui_tree, executor_result, 
                                                              context={"replanning": True, "original_failed": failed_subgoal})
                            
                            if executor_result is None:
                                print("    FAIL: Executor could not find target element")
                                continue
                            
                            # Update observation
                            observation = executor_result.get("observation", observation)
                            new_ui_tree = observation.get("ui", {})
                            
                            # Capture screenshot after replanning action
                            try:
                                if "pixels" in observation:
                                    action_screenshot = observation["pixels"]
                                else:
                                    action_screenshot = env.render(mode="rgb_array")
                                message_logger.log_screenshot(action_screenshot, f"replan_{total_replanning_attempts}", 
                                                            f"After replanning: {alt_subgoal}")
                            except Exception as e:
                                logger.warning(f"Could not capture replanning screenshot: {e}")
                            
                            # Verify alternative subgoal
                            verification_result = verifier.verify(alt_subgoal, new_ui_tree, executor_result)
                            message_logger.log_verifier_message(alt_subgoal, new_ui_tree, verification_result,
                                                              context={"replanning": True, "original_failed": failed_subgoal})
                            
                            if verification_result.get("success", False):
                                print(f"    ✓ PASS: {verification_result.get('reason', 'Alternative succeeded')}")
                                passed_steps += 1
                                subgoal_success = True
                                replanning_successes += 1
                                break  # Success, move to next failed subgoal
                            else:
                                print(f"    ✗ FAIL: {verification_result.get('reason', 'Alternative failed')}")
                                
                        except Exception as e:
                            print(f"    ✗ FAIL: Error during alternative execution - {e}")
                            logger.error(f"Alternative subgoal execution failed: {e}")
                    
                    if subgoal_success:
                        print(f"✓ Successfully recovered from failure: {failed_subgoal}")
                    else:
                        print(f"✗ All alternatives failed for: {failed_subgoal}")
                        
                except Exception as e:
                    logger.error(f"Replanning failed for subgoal '{failed_subgoal}': {e}")
                    print(f"✗ Replanning failed: {e}")
            
            # Summary of replanning results
            print(f"\n=== Replanning Summary ===")
            print(f"Failed subgoals: {len(failed_subgoals)}")
            print(f"Replanning attempts: {total_replanning_attempts}")
            print(f"Successful recoveries: {replanning_successes}")
            print(f"Recovery rate: {(replanning_successes / len(failed_subgoals) * 100):.1f}%")
            
            # Update final results
            final_failed_subgoals = [sg for sg in failed_subgoals if sg not in [f"recovered_{sg}" for sg in range(replanning_successes)]]
        
        # Supervisor Agent: Analyze test execution
        print("=== Supervisor Analysis ===")
        try:
            test_trace = message_logger.test_trace
            supervisor_analysis = supervisor.analyze_test_execution(test_trace, task_instruction)
            
            print("Supervisor Analysis Results:")
            print(f"  Overall Success: {supervisor_analysis.get('overall_success', False)}")
            print(f"  Success Rate: {supervisor_analysis.get('success_rate', 0.0):.1%}")
            print(f"  Failure Analysis: {supervisor_analysis.get('failure_analysis', 'No analysis')}")
            
            # Get prompt improvement suggestions
            prompt_improvements = supervisor.suggest_prompt_improvements(test_trace)
            print(f"  Prompt Improvements: {len(prompt_improvements.get('overall_recommendations', []))} suggestions")
            
        except Exception as e:
            logger.error(f"Supervisor analysis failed: {e}")
            supervisor_analysis = {"error": str(e)}
        
        # Final results
        final_results = {
            "passed_steps": passed_steps,
            "total_steps": total_steps,
            "success_rate": (passed_steps / total_steps) * 100 if total_steps > 0 else 0,
            "failed_subgoals": failed_subgoals,
            "test_summary": message_logger.get_test_summary(),
            "supervisor_analysis": supervisor_analysis
        }
        
        print("=== Test Results ===")
        print(f"Passed: {passed_steps}/{total_steps} steps")
        print(f"Success Rate: {final_results['success_rate']:.1f}%")
        print(f"Failed Subgoals: {len(failed_subgoals)}")
        
        if passed_steps == total_steps:
            print("All steps completed successfully!")
        elif passed_steps > 0:
            print("Partial success - some steps failed")
        else:
            print("All steps failed")
        
        # Log test end
        message_logger.log_test_end(final_results)
        
        # Capture final screenshot for visual traces (QualGent requirement)
        try:
            final_screenshot = env.render(mode="rgb_array")
            message_logger.log_screenshot(final_screenshot, total_steps + 1, "Final state after test completion")
        except Exception as e:
            logger.warning(f"Could not capture final screenshot: {e}")
        
        return final_results
            
    except FileNotFoundError as e:
        if "adb" in str(e):
            print("Error: Android Debug Bridge (adb) not found")
            print("Please install Android SDK and ensure adb is in your PATH")
            print("Expected location: ~/Android/Sdk/platform-tools/adb")
        else:
            print(f"File not found: {e}")
        return {"error": "adb_not_found"}
    except ConnectionError as e:
        print("Error: Cannot connect to Android device/emulator")
        print("Please ensure an Android emulator is running:")
        print("AVD_NAME=Pixel_6_API_33")
        print("~/Android/Sdk/emulator/emulator -avd $AVD_NAME -no-snapshot -grpc 8554")
        return {"error": "connection_failed"}
    except Exception as e:
        print(f"Unexpected error: {e}")
        print("Please check your Android setup and try again")
        logger.error(f"Unexpected error: {e}")
        return {"error": "unexpected_error"}

def main():
    """Main entry point for Agent-S compliant QA testing"""
    
    # Test with different tasks as specified in QualGent challenge
    test_configs = [
        {
            "task_instruction": "Test turning Wi-Fi on and off",
            "task_name": "settings_wifi"
        },
        {
            "task_instruction": "Set an alarm for 8:00 AM",
            "task_name": "clock_alarm"
        },
        {
            "task_instruction": "Search for emails from john@example.com",
            "task_name": "email_search"
        }
    ]
    
    for config in test_configs:
        print(f"\n{'='*60}")
        result = run_agent_s_qa_test(
            task_instruction=config["task_instruction"],
            task_name=config["task_name"]
        )
        if "error" in result:
            print(f"Test failed with error: {result['error']}")
            break
        print(f"{'='*60}\n")

if __name__ == "__main__":
    main() 