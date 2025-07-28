#!/usr/bin/env python3
"""
Test script for Android in the Wild integration.

This script tests the basic functionality of the android_in_the_wild integration
with the multi-agent QA system.
"""

import os
import sys
import json
import tempfile
from pathlib import Path

# Add current directory to path
sys.path.append('.')

def test_data_structures():
    """Test the data structures and classes."""
    print("Testing data structures...")
    
    # Test AndroidInTheWildEpisode dataclass
    from android_in_the_wild_integration import AndroidInTheWildEpisode, TaskPrompt, AgentPerformance
    
    # Create test episode
    test_episode = AndroidInTheWildEpisode(
        episode_id="test_episode_001",
        goal_info="Open Chrome and navigate to google.com",
        steps=[
            {
                'step_id': 0,
                'action_type': 4,  # DUAL_POINT
                'touch_coords': (0.5, 0.3),
                'lift_coords': (0.5, 0.3),
                'current_activity': 'com.android.chrome'
            }
        ],
        device_type="pixel_4",
        android_api_level=30,
        episode_length=1
    )
    
    print(f"✓ Created test episode: {test_episode.episode_id}")
    
    # Test TaskPrompt
    test_prompt = TaskPrompt(
        original_goal="Open Chrome and navigate to google.com",
        inferred_task="Launch Chrome browser and navigate to Google homepage",
        confidence=0.85,
        reasoning="User wants to open Chrome browser and go to Google"
    )
    
    print(f"✓ Created test task prompt: {test_prompt.inferred_task}")
    
    # Test AgentPerformance
    test_performance = AgentPerformance(
        accuracy=0.85,
        robustness=0.78,
        generalization=0.82,
        action_matching_score=0.90,
        completion_rate=0.75,
        error_rate=0.15
    )
    
    print(f"✓ Created test performance metrics: accuracy={test_performance.accuracy}")
    
    return True

def test_data_processor():
    """Test the data processor functionality."""
    print("\nTesting data processor...")
    
    try:
        from android_in_the_wild_integration import AndroidInTheWildProcessor
        
        # Initialize processor
        processor = AndroidInTheWildProcessor()
        print("✓ Initialized AndroidInTheWildProcessor")
        
        # Test dataset directories
        expected_datasets = ['general', 'google_apps', 'install', 'single', 'web_shopping']
        for dataset in expected_datasets:
            assert dataset in processor.dataset_directories, f"Missing dataset: {dataset}"
        
        print("✓ All expected datasets found")
        
        return True
        
    except Exception as e:
        print(f"✗ Error testing data processor: {e}")
        return False

def test_task_prompt_generator():
    """Test the task prompt generator."""
    print("\nTesting task prompt generator...")
    
    try:
        from android_in_the_wild_integration import TaskPromptGenerator
        
        # Initialize generator
        generator = TaskPromptGenerator()
        print("✓ Initialized TaskPromptGenerator")
        
        # Create test episode data
        test_episode_data = {
            'episode_id': 'test_episode_002',
            'goal_info': 'Search for "best coffee shops near me"',
            'device_type': 'pixel_4',
            'android_api_level': 30,
            'episode_length': 3,
            'steps': [
                {
                    'step_id': 0,
                    'action_type': 4,
                    'current_activity': 'com.google.android.apps.maps'
                },
                {
                    'step_id': 1,
                    'action_type': 3,
                    'typed_text': 'coffee shops',
                    'current_activity': 'com.google.android.apps.maps'
                }
            ]
        }
        
        # Test task prompt generation (mock)
        task_prompt = generator.generate_task_prompt(test_episode_data)
        print(f"✓ Generated task prompt: {task_prompt['generated_task']}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error testing task prompt generator: {e}")
        return False

def test_evaluator():
    """Test the evaluator functionality."""
    print("\nTesting evaluator...")
    
    try:
        from android_in_the_wild_qa_integration import AndroidInTheWildQAEvaluator
        
        # Initialize evaluator
        evaluator = AndroidInTheWildQAEvaluator()
        print("✓ Initialized AndroidInTheWildQAEvaluator")
        
        # Test with mock data
        test_episode = {
            'episode_id': 'test_episode_003',
            'goal_info': 'Open settings and turn on WiFi',
            'device_type': 'pixel_4',
            'android_api_level': 30,
            'processed_steps': 2,
            'steps': [
                {
                    'step_id': 0,
                    'action_type': 4,
                    'touch_coords': (0.1, 0.1),
                    'lift_coords': (0.1, 0.1),
                    'current_activity': 'com.android.settings'
                },
                {
                    'step_id': 1,
                    'action_type': 4,
                    'touch_coords': (0.5, 0.5),
                    'lift_coords': (0.5, 0.5),
                    'current_activity': 'com.android.settings'
                }
            ]
        }
        
        # Test task prompt generation
        task_prompt = evaluator.generate_task_prompt(test_episode)
        print(f"✓ Generated task prompt: {task_prompt['generated_task']}")
        
        # Test execution simulation
        execution_result = evaluator.simulate_agent_execution(task_prompt, test_episode)
        print(f"✓ Simulated execution: {execution_result['steps_completed']} steps")
        
        # Test evaluation
        evaluation_metrics = evaluator.evaluate_execution(execution_result)
        print(f"✓ Evaluation metrics: accuracy={evaluation_metrics['accuracy']:.3f}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error testing evaluator: {e}")
        return False

def test_data_downloader():
    """Test the data downloader functionality."""
    print("\nTesting data downloader...")
    
    try:
        from download_android_in_the_wild_data import AndroidInTheWildDataDownloader
        
        # Initialize downloader with temp directory
        with tempfile.TemporaryDirectory() as temp_dir:
            downloader = AndroidInTheWildDataDownloader(temp_dir)
            print("✓ Initialized AndroidInTheWildDataDownloader")
            
            # Test dataset URLs
            expected_datasets = ['google_apps', 'web_shopping', 'install', 'general', 'single']
            for dataset in expected_datasets:
                assert dataset in downloader.dataset_urls, f"Missing dataset URL: {dataset}"
            
            print("✓ All dataset URLs configured")
            
            # Test directory creation
            for dataset_path in downloader.dataset_paths.values():
                assert dataset_path.exists(), f"Directory not created: {dataset_path}"
            
            print("✓ All directories created successfully")
        
        return True
        
    except Exception as e:
        print(f"✗ Error testing data downloader: {e}")
        return False

def test_requirements():
    """Test that all required dependencies are available."""
    print("\nTesting requirements...")
    
    required_packages = [
        'tensorflow',
        'numpy',
        'matplotlib',
        'jax',
        'jaxlib',
        'google-cloud-storage',
        'protobuf',
        'Pillow',
        'opencv-python',
        'scikit-learn',
        'pandas',
        'seaborn'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✓ {package}")
        except ImportError:
            print(f"✗ {package} - MISSING")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\nMissing packages: {missing_packages}")
        print("Install with: pip install -r android_in_the_wild_requirements.txt")
        return False
    
    print("✓ All required packages available")
    return True

def test_file_structure():
    """Test that all required files exist."""
    print("\nTesting file structure...")
    
    required_files = [
        'android_in_the_wild_integration.py',
        'download_android_in_the_wild_data.py',
        'android_in_the_wild_qa_integration.py',
        'android_in_the_wild_requirements.txt',
        'README_ANDROID_IN_THE_WILD.md'
    ]
    
    missing_files = []
    
    for file_path in required_files:
        if Path(file_path).exists():
            print(f"✓ {file_path}")
        else:
            print(f"✗ {file_path} - MISSING")
            missing_files.append(file_path)
    
    if missing_files:
        print(f"\nMissing files: {missing_files}")
        return False
    
    print("✓ All required files present")
    return True

def test_google_research_repo():
    """Test that the google-research repository is available."""
    print("\nTesting google-research repository...")
    
    google_research_path = Path("google-research")
    android_in_the_wild_path = google_research_path / "android_in_the_wild"
    
    if not google_research_path.exists():
        print("✗ google-research directory not found")
        print("Run: git clone https://github.com/google-research/google-research.git")
        return False
    
    if not android_in_the_wild_path.exists():
        print("✗ android_in_the_wild directory not found")
        return False
    
    # Check for required files
    required_files = ['README.md', 'demo.ipynb', 'action_matching.py', 'action_type.py']
    missing_files = []
    
    for file_name in required_files:
        file_path = android_in_the_wild_path / file_name
        if not file_path.exists():
            missing_files.append(file_name)
    
    if missing_files:
        print(f"✗ Missing files in android_in_the_wild: {missing_files}")
        return False
    
    print("✓ google-research repository available")
    return True

def main():
    """Run all tests."""
    print("Android in the Wild Integration Test Suite")
    print("=" * 50)
    
    tests = [
        ("File Structure", test_file_structure),
        ("Requirements", test_requirements),
        ("Google Research Repository", test_google_research_repo),
        ("Data Structures", test_data_structures),
        ("Data Processor", test_data_processor),
        ("Task Prompt Generator", test_task_prompt_generator),
        ("Evaluator", test_evaluator),
        ("Data Downloader", test_data_downloader)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            if test_func():
                passed += 1
                print(f"✓ {test_name} PASSED")
            else:
                print(f"✗ {test_name} FAILED")
        except Exception as e:
            print(f"✗ {test_name} ERROR: {e}")
    
    print(f"\n{'='*50}")
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The integration is ready to use.")
        print("\nNext steps:")
        print("1. Run: python download_android_in_the_wild_data.py")
        print("2. Run: python android_in_the_wild_qa_integration.py")
    else:
        print("❌ Some tests failed. Please fix the issues before proceeding.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 