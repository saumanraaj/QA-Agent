#!/usr/bin/env python3
"""
API Key Setup Script for Multi-Agent QA System
This script helps you configure API keys for OpenAI and Google Gemini.
"""

import os
import getpass
from pathlib import Path

def setup_api_keys():
    """Interactive setup for API keys"""
    print("Multi-Agent QA System - API Key Setup")
    print("=" * 50)
    
    # Check if .env file exists
    env_file = Path(".env")
    if env_file.exists():
        print("Found existing .env file")
        overwrite = input("Do you want to overwrite it? (y/N): ").lower().strip()
        if overwrite != 'y':
            print("Setup cancelled.")
            return
    
    print("\nPlease provide your API keys:")
    print("(Press Enter to skip if you don't have a key)")
    
    # OpenAI API Key
    print("\nOpenAI API Key:")
    print("   Get your key from: https://platform.openai.com/api-keys")
    openai_key = getpass.getpass("   Enter OpenAI API Key (hidden): ").strip()
    
    # Gemini API Key
    print("\nGoogle Gemini API Key:")
    print("   Get your key from: https://makersuite.google.com/app/apikey")
    gemini_key = getpass.getpass("   Enter Gemini API Key (hidden): ").strip()
    
    # Create .env file
    env_content = []
    if openai_key:
        env_content.append(f"OPENAI_API_KEY={openai_key}")
    if gemini_key:
        env_content.append(f"GEMINI_API_KEY={gemini_key}")
    
    if env_content:
        with open(".env", "w") as f:
            f.write("\n".join(env_content))
        print(f"\nAPI keys saved to .env file")
        print(f"   Keys configured: {len(env_content)}")
    else:
        print("\nNo API keys provided. System will use mock mode.")
    
    print("\nNext steps:")
    print("1. Install python-dotenv: pip install python-dotenv")
    print("2. Load environment: source .env")
    print("3. Test agents: python agents/planner_agent.py")

def check_api_keys():
    """Check if API keys are properly configured"""
    print("Checking API Key Configuration")
    print("=" * 40)
    
    openai_key = os.getenv("OPENAI_API_KEY")
    gemini_key = os.getenv("GEMINI_API_KEY")
    
    print(f"OpenAI API Key: {'Set' if openai_key else 'Not set'}")
    print(f"Gemini API Key: {'Set' if gemini_key else 'Not set'}")
    
    if not openai_key and not gemini_key:
        print("\nNo API keys found!")
        print("   Run: python setup_api_keys.py")
        return False
    
    return True

def create_mock_config():
    """Create a mock configuration for testing without API keys"""
    print("Creating Mock Configuration")
    print("=" * 40)
    
    mock_config = {
        "use_mock": True,
        "mock_responses": {
            "planner": [
                "Open Settings",
                "Tap on 'Network & Internet'", 
                "Tap on 'Wi-Fi'",
                "Toggle Wi-Fi ON",
                "Toggle Wi-Fi OFF"
            ],
            "executor": {
                "element_id": "Wi-Fi",
                "action_type": "touch",
                "confidence": 0.9,
                "reason": "Mock execution for testing"
            },
            "verifier": {
                "success": True,
                "reason": "Mock verification passed",
                "confidence": 0.8
            }
        }
    }
    
    import json
    with open("mock_config.json", "w") as f:
        json.dump(mock_config, f, indent=2)
    
    print("Mock configuration created: mock_config.json")
    print("   Use this for testing without API keys")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "check":
        check_api_keys()
    elif len(sys.argv) > 1 and sys.argv[1] == "mock":
        create_mock_config()
    else:
        setup_api_keys() 