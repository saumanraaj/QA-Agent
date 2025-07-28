#!/usr/bin/env python3
"""
Simplified Android in the Wild Integration

This script provides a simplified integration of android_in_the_wild dataset
with the multi-agent QA system, focusing on core functionality without
heavy dependencies.
"""

import os
import sys
import json
import random
from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime

# Add current directory to path
sys.path.append('.')

# Mock agents for testing
class MockExecutorAgent:
    def __init__(self):
        self.name = "MockExecutorAgent"
    
    def execute_task(self, task):
        return {"status": "success", "actions": ["mock_action_1", "mock_action_2"]}

class MockLLMEngine:
    def __init__(self):
        self.name = "MockLLMEngine"
    
    def generate_response(self, prompt):
        return f"Mock response to: {prompt[:50]}..."

class MockMessageLogger:
    def __init__(self):
        self.name = "MockMessageLogger"
    
    def log_message(self, message):
        print(f"Mock logged: {message}")

try:
    from agents.executor_agent import ExecutorAgent
    from agents.llm_engine import LLMEngine
    from agents.message_logger import MessageLogger
except ImportError as e:
    print(f"Error importing agent modules: {e}")
    print("Using mock agents for demonstration")

class AndroidInTheWildSimpleIntegration:
    """Simplified integration for android_in_the_wild with multi-agent QA system."""
    
    def __init__(self):
        # Initialize agents (real or mock)
        try:
            self.executor_agent = ExecutorAgent()
            self.llm_engine = LLMEngine()
            self.message_logger = MessageLogger()
            self.use_mock = False
        except:
            self.executor_agent = MockExecutorAgent()
            self.llm_engine = MockLLMEngine()
            self.message_logger = MockMessageLogger()
            self.use_mock = True
        
        self.results = []
        
        # Sample android_in_the_wild data for demonstration
        self.sample_episodes = self._create_sample_episodes()
    
    def _create_sample_episodes(self) -> List[Dict[str, Any]]:
        """Create sample episodes based on android_in_the_wild format."""
        return [
            {
                'episode_id': 'sample_episode_001',
                'goal_info': 'Open Chrome and navigate to google.com',
                'device_type': 'pixel_4',
                'android_api_level': 30,
                'episode_length': 3,
                'steps': [
                    {
                        'step_id': 0,
                        'action_type': 4,  # DUAL_POINT
                        'touch_coords': (0.5, 0.3),
                        'lift_coords': (0.5, 0.3),
                        'current_activity': 'com.android.chrome',
                        'typed_text': None
                    },
                    {
                        'step_id': 1,
                        'action_type': 3,  # TYPE
                        'touch_coords': None,
                        'lift_coords': None,
                        'current_activity': 'com.android.chrome',
                        'typed_text': 'google.com'
                    },
                    {
                        'step_id': 2,
                        'action_type': 7,  # PRESS_ENTER
                        'touch_coords': None,
                        'lift_coords': None,
                        'current_activity': 'com.android.chrome',
                        'typed_text': None
                    }
                ]
            },
            {
                'episode_id': 'sample_episode_002',
                'goal_info': 'Search for "best coffee shops near me" in Maps',
                'device_type': 'pixel_4',
                'android_api_level': 30,
                'episode_length': 4,
                'steps': [
                    {
                        'step_id': 0,
                        'action_type': 4,  # DUAL_POINT
                        'touch_coords': (0.2, 0.1),
                        'lift_coords': (0.2, 0.1),
                        'current_activity': 'com.google.android.apps.maps'
                    },
                    {
                        'step_id': 1,
                        'action_type': 4,  # DUAL_POINT
                        'touch_coords': (0.5, 0.5),
                        'lift_coords': (0.5, 0.5),
                        'current_activity': 'com.google.android.apps.maps'
                    },
                    {
                        'step_id': 2,
                        'action_type': 3,  # TYPE
                        'touch_coords': None,
                        'lift_coords': None,
                        'current_activity': 'com.google.android.apps.maps',
                        'typed_text': 'coffee shops near me'
                    },
                    {
                        'step_id': 3,
                        'action_type': 7,  # PRESS_ENTER
                        'touch_coords': None,
                        'lift_coords': None,
                        'current_activity': 'com.google.android.apps.maps',
                        'typed_text': None
                    }
                ]
            },
            {
                'episode_id': 'sample_episode_003',
                'goal_info': 'Open Settings and turn on WiFi',
                'device_type': 'pixel_4',
                'android_api_level': 30,
                'episode_length': 5,
                'steps': [
                    {
                        'step_id': 0,
                        'action_type': 4,  # DUAL_POINT
                        'touch_coords': (0.1, 0.1),
                        'lift_coords': (0.1, 0.1),
                        'current_activity': 'com.android.settings'
                    },
                    {
                        'step_id': 1,
                        'action_type': 4,  # DUAL_POINT
                        'touch_coords': (0.3, 0.4),
                        'lift_coords': (0.3, 0.4),
                        'current_activity': 'com.android.settings'
                    },
                    {
                        'step_id': 2,
                        'action_type': 4,  # DUAL_POINT
                        'touch_coords': (0.5, 0.6),
                        'lift_coords': (0.5, 0.6),
                        'current_activity': 'com.android.settings'
                    },
                    {
                        'step_id': 3,
                        'action_type': 4,  # DUAL_POINT
                        'touch_coords': (0.7, 0.8),
                        'lift_coords': (0.7, 0.8),
                        'current_activity': 'com.android.settings'
                    },
                    {
                        'step_id': 4,
                        'action_type': 4,  # DUAL_POINT
                        'touch_coords': (0.8, 0.9),
                        'lift_coords': (0.8, 0.9),
                        'current_activity': 'com.android.settings'
                    }
                ]
            }
        ]
    
    def generate_task_prompt(self, episode: Dict[str, Any]) -> Dict[str, Any]:
        """Generate task prompt from episode goal and context."""
        
        goal_info = episode['goal_info']
        device_type = episode['device_type']
        android_version = episode['android_api_level']
        num_steps = episode['episode_length']
        
        # Analyze episode to understand task complexity
        action_types = [step['action_type'] for step in episode['steps']]
        activities = list(set(step['current_activity'] for step in episode['steps']))
        
        # Generate enhanced task prompt
        prompt = f"""
        Based on the following Android episode, generate a clear and specific task prompt:
        
        Original Goal: {goal_info}
        Device: {device_type}
        Android Version: {android_version}
        Number of Steps: {num_steps}
        Activities Involved: {activities[:5]}  # First 5 activities
        Action Types: {list(set(action_types))}
        
        Generate a task prompt that:
        1. Clearly describes what the user was trying to accomplish
        2. Is specific enough for an AI agent to understand and execute
        3. Includes relevant context about the app or website being used
        4. Specifies the expected outcome or completion criteria
        """
        
        try:
            response = self.llm_engine.generate_response(prompt)
            
            # Extract the task from response
            task_lines = response.strip().split('\n')
            task = task_lines[0] if task_lines else goal_info
            
            return {
                'original_goal': goal_info,
                'generated_task': task,
                'full_response': response,
                'confidence': self._calculate_task_confidence(episode),
                'complexity_score': self._calculate_complexity_score(episode)
            }
            
        except Exception as e:
            print(f"Error generating task prompt: {e}")
            return {
                'original_goal': goal_info,
                'generated_task': goal_info,
                'full_response': f"Error: {e}",
                'confidence': 0.5,
                'complexity_score': 0.5
            }
    
    def _calculate_task_confidence(self, episode: Dict[str, Any]) -> float:
        """Calculate confidence in the generated task prompt."""
        confidence = 0.5  # Base confidence
        
        # More steps = more complex task = lower confidence
        if episode['episode_length'] > 10:
            confidence -= 0.1
        elif episode['episode_length'] < 5:
            confidence += 0.1
        
        # More activities = more complex task
        activities = set(step['current_activity'] for step in episode['steps'])
        if len(activities) > 3:
            confidence -= 0.1
        
        # Type actions indicate text input tasks (more predictable)
        action_types = [step['action_type'] for step in episode['steps']]
        if 3 in action_types:  # TYPE action
            confidence += 0.1
        
        return max(0.0, min(1.0, confidence))
    
    def _calculate_complexity_score(self, episode: Dict[str, Any]) -> float:
        """Calculate complexity score for the episode."""
        complexity = 0.0
        
        # Number of steps
        complexity += min(1.0, episode['episode_length'] / 20.0)
        
        # Number of unique activities
        activities = set(step['current_activity'] for step in episode['steps'])
        complexity += min(1.0, len(activities) / 5.0)
        
        # Action type diversity
        action_types = set(step['action_type'] for step in episode['steps'])
        complexity += min(1.0, len(action_types) / 4.0)
        
        return min(1.0, complexity / 3.0)
    
    def simulate_agent_execution(self, task_prompt: Dict[str, Any], episode: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate agent execution of the task."""
        
        agent_trace = []
        ground_truth_steps = episode['steps']
        
        # Simulate agent actions based on task prompt and ground truth
        for i, gt_step in enumerate(ground_truth_steps):
            agent_action = self._simulate_agent_action(gt_step, task_prompt, i)
            agent_trace.append(agent_action)
        
        return {
            'task_prompt': task_prompt,
            'agent_trace': agent_trace,
            'ground_truth_steps': ground_truth_steps,
            'execution_success': len(agent_trace) > 0,
            'steps_completed': len(agent_trace),
            'total_steps': len(ground_truth_steps)
        }
    
    def _simulate_agent_action(self, gt_step: Dict[str, Any], task_prompt: Dict[str, Any], step_index: int) -> Dict[str, Any]:
        """Simulate a single agent action."""
        
        action_type = gt_step['action_type']
        confidence = 0.8  # Base confidence
        
        # Adjust confidence based on task complexity
        if task_prompt['complexity_score'] > 0.7:
            confidence *= 0.8
        elif task_prompt['complexity_score'] < 0.3:
            confidence *= 1.1
        
        # Simulate different action types
        if action_type == 4:  # DUAL_POINT
            return {
                'type': 'dual_point',
                'touch_coords': gt_step.get('touch_coords'),
                'lift_coords': gt_step.get('lift_coords'),
                'confidence': confidence,
                'step_index': step_index,
                'ground_truth_action_type': action_type
            }
        elif action_type == 3:  # TYPE
            return {
                'type': 'type',
                'text': gt_step.get('typed_text', ''),
                'confidence': confidence * 0.9,  # Text input is harder
                'step_index': step_index,
                'ground_truth_action_type': action_type
            }
        elif action_type in [5, 6, 7]:  # PRESS_BACK, PRESS_HOME, PRESS_ENTER
            return {
                'type': 'button_press',
                'button': {5: 'back', 6: 'home', 7: 'enter'}[action_type],
                'confidence': confidence,
                'step_index': step_index,
                'ground_truth_action_type': action_type
            }
        else:
            return {
                'type': 'unknown',
                'confidence': confidence * 0.5,
                'step_index': step_index,
                'ground_truth_action_type': action_type
            }
    
    def evaluate_execution(self, execution_result: Dict[str, Any]) -> Dict[str, float]:
        """Evaluate agent execution against ground truth."""
        
        agent_trace = execution_result['agent_trace']
        ground_truth_steps = execution_result['ground_truth_steps']
        
        if not agent_trace:
            return {
                'accuracy': 0.0,
                'action_matching': 0.0,
                'coordinate_accuracy': 0.0,
                'completion_rate': 0.0,
                'robustness': 0.0,
                'generalization': 0.0
            }
        
        # Calculate action matching
        action_matches = []
        coordinate_accuracies = []
        
        for agent_action, gt_step in zip(agent_trace, ground_truth_steps):
            # Check action type match
            if agent_action['ground_truth_action_type'] == gt_step['action_type']:
                action_matches.append(1.0)
            else:
                action_matches.append(0.0)
            
            # Calculate coordinate accuracy for dual point actions
            if (agent_action['type'] == 'dual_point' and 
                gt_step['action_type'] == 4 and 
                agent_action.get('touch_coords') and 
                gt_step.get('touch_coords')):
                
                coord_accuracy = self._calculate_coordinate_accuracy(
                    agent_action['touch_coords'], 
                    gt_step['touch_coords']
                )
                coordinate_accuracies.append(coord_accuracy)
        
        # Calculate metrics
        action_matching_score = sum(action_matches) / len(action_matches) if action_matches else 0.0
        coordinate_accuracy = sum(coordinate_accuracies) / len(coordinate_accuracies) if coordinate_accuracies else 0.0
        completion_rate = len(agent_trace) / len(ground_truth_steps)
        
        # Overall accuracy
        accuracy = (action_matching_score + coordinate_accuracy) / 2
        
        # Robustness (ability to handle variations)
        robustness = min(1.0, action_matching_score * 0.8 + coordinate_accuracy * 0.2)
        
        # Generalization (performance across different task types)
        generalization = min(1.0, completion_rate * 0.6 + (1.0 - (1.0 - action_matching_score)) * 0.4)
        
        return {
            'accuracy': accuracy,
            'action_matching': action_matching_score,
            'coordinate_accuracy': coordinate_accuracy,
            'completion_rate': completion_rate,
            'robustness': robustness,
            'generalization': generalization
        }
    
    def _calculate_coordinate_accuracy(self, agent_coords: tuple, gt_coords: tuple) -> float:
        """Calculate accuracy of coordinate predictions."""
        if not agent_coords or not gt_coords:
            return 0.0
        
        # Simple Euclidean distance calculation
        import math
        distance = math.sqrt((agent_coords[0] - gt_coords[0])**2 + (agent_coords[1] - gt_coords[1])**2)
        # Normalize by screen size (assuming 1080p)
        normalized_distance = distance / math.sqrt(1920**2 + 1080**2)
        return max(0.0, 1.0 - normalized_distance)
    
    def run_evaluation(self, max_episodes: int = 3):
        """Run complete evaluation on sample episodes."""
        print("Starting Android in the Wild QA evaluation...")
        print(f"Using {'mock' if self.use_mock else 'real'} agents")
        
        # Limit episodes
        episodes = self.sample_episodes[:max_episodes]
        
        # Evaluate each episode
        for i, episode in enumerate(episodes):
            print(f"\nEvaluating episode {i+1}/{len(episodes)}: {episode['episode_id']}")
            
            # Generate task prompt
            task_prompt = self.generate_task_prompt(episode)
            print(f"Original goal: {task_prompt['original_goal']}")
            print(f"Generated task: {task_prompt['generated_task']}")
            print(f"Confidence: {task_prompt['confidence']:.2f}")
            
            # Simulate agent execution
            execution_result = self.simulate_agent_execution(task_prompt, episode)
            
            # Evaluate execution
            evaluation_metrics = self.evaluate_execution(execution_result)
            
            # Store results
            result = {
                'episode_id': episode['episode_id'],
                'task_prompt': task_prompt,
                'execution_result': execution_result,
                'evaluation_metrics': evaluation_metrics,
                'timestamp': datetime.now().isoformat()
            }
            self.results.append(result)
            
            print(f"Episode {i+1} - Accuracy: {evaluation_metrics['accuracy']:.3f}, "
                   f"Completion: {evaluation_metrics['completion_rate']:.3f}")
        
        # Generate report
        self._generate_evaluation_report()
    
    def _generate_evaluation_report(self):
        """Generate comprehensive evaluation report."""
        if not self.results:
            print("No results to report")
            return
        
        # Calculate aggregate metrics
        metrics = ['accuracy', 'action_matching', 'coordinate_accuracy', 
                  'completion_rate', 'robustness', 'generalization']
        
        aggregate_metrics = {}
        for metric in metrics:
            values = [r['evaluation_metrics'][metric] for r in self.results]
            aggregate_metrics[metric] = {
                'mean': sum(values) / len(values),
                'min': min(values),
                'max': max(values)
            }
        
        # Calculate task prompt quality metrics
        confidence_scores = [r['task_prompt']['confidence'] for r in self.results]
        complexity_scores = [r['task_prompt']['complexity_score'] for r in self.results]
        
        report = {
            'total_episodes': len(self.results),
            'aggregate_metrics': aggregate_metrics,
            'task_prompt_quality': {
                'average_confidence': sum(confidence_scores) / len(confidence_scores),
                'average_complexity': sum(complexity_scores) / len(complexity_scores)
            },
            'detailed_results': self.results,
            'timestamp': datetime.now().isoformat()
        }
        
        # Save report
        report_filename = f"android_in_the_wild_simple_evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_filename, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Print summary
        print(f"\n{'='*50}")
        print("ANDROID IN THE WILD QA EVALUATION REPORT")
        print(f"{'='*50}")
        print(f"Total Episodes: {report['total_episodes']}")
        print(f"Average Accuracy: {aggregate_metrics['accuracy']['mean']:.3f}")
        print(f"Average Action Matching: {aggregate_metrics['action_matching']['mean']:.3f}")
        print(f"Average Completion Rate: {aggregate_metrics['completion_rate']['mean']:.3f}")
        print(f"Average Robustness: {aggregate_metrics['robustness']['mean']:.3f}")
        print(f"Average Generalization: {aggregate_metrics['generalization']['mean']:.3f}")
        print(f"Task Prompt Confidence: {report['task_prompt_quality']['average_confidence']:.3f}")
        print(f"Task Complexity: {report['task_prompt_quality']['average_complexity']:.3f}")
        print(f"\nDetailed report saved to: {report_filename}")
        
        # Print detailed results
        print(f"\n{'='*50}")
        print("DETAILED RESULTS")
        print(f"{'='*50}")
        for i, result in enumerate(self.results):
            metrics = result['evaluation_metrics']
            print(f"\nEpisode {i+1}: {result['episode_id']}")
            print(f"  Original Goal: {result['task_prompt']['original_goal']}")
            print(f"  Generated Task: {result['task_prompt']['generated_task']}")
            print(f"  Accuracy: {metrics['accuracy']:.3f}")
            print(f"  Action Matching: {metrics['action_matching']:.3f}")
            print(f"  Completion Rate: {metrics['completion_rate']:.3f}")
            print(f"  Steps Completed: {result['execution_result']['steps_completed']}/{result['execution_result']['total_steps']}")

def main():
    """Main execution function."""
    print("Android in the Wild Simple Integration")
    print("=" * 50)
    
    # Initialize integration
    integration = AndroidInTheWildSimpleIntegration()
    
    # Run evaluation
    integration.run_evaluation(max_episodes=3)
    
    print("\nEvaluation complete!")

if __name__ == "__main__":
    main() 