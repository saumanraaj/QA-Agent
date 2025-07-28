import json
import logging
from typing import Dict, List, Any, Optional
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agents.llm_engine import get_llm_engine, AgentPrompts

logger = logging.getLogger(__name__)

class VerifierAgent:
    """LLM-powered verifier agent following Agent-S architecture"""
    
    def __init__(self, engine_type: str = "openai", model: str = "gpt-4o-mini"):
        self.llm_engine = get_llm_engine(engine_type=engine_type, model=model)
    
    def verify(self, subgoal: str, ui_tree: dict, executor_result: Optional[Dict] = None) -> Dict[str, Any]:
        """Verify if a subgoal was successfully executed using LLM reasoning"""
        
        system_prompt = AgentPrompts.get_verifier_prompt()

        # Flatten UI tree for easier analysis
        ui_texts = self._extract_ui_texts(ui_tree)
        
        user_prompt = f"""Subgoal: {subgoal}

Current UI Elements: {ui_texts}

Executor Result: {executor_result if executor_result else "No additional result data"}

Determine if the subgoal was successfully executed:"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        try:
            response = self.llm_engine.generate(messages, temperature=0.1, max_tokens=300)
            
            # Parse JSON response
            try:
                result = json.loads(response.strip())
                if isinstance(result, dict) and "success" in result:
                    logger.info(f"Verification result for '{subgoal}': {result['success']} - {result.get('reason', 'No reason provided')}")
                    return result
                else:
                    raise ValueError("Response missing 'success' field")
            except json.JSONDecodeError:
                # Try to extract JSON from response
                import re
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if json_match:
                    result = json.loads(json_match.group())
                    logger.info(f"Verification result for '{subgoal}': {result.get('success', False)}")
                    return result
                else:
                    raise ValueError("Could not parse JSON from LLM response")
                    
        except Exception as e:
            logger.error(f"LLM verification failed: {e}")
            raise ValueError(f"LLM verification failed for subgoal '{subgoal}': {e}")
    
    def _extract_ui_texts(self, ui_tree: dict) -> List[str]:
        """Extract all text elements from UI tree for analysis"""
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
        return texts

if __name__ == "__main__":
    verifier = VerifierAgent()
    
    mock_ui_tree = {
        "children": [
            {
                "text": "Wi-Fi",
                "bounds": [100, 200, 300, 250]
            },
            {
                "text": "Wi-Fi is on",
                "bounds": [100, 300, 300, 350]
            }
        ]
    }
    
    test_cases = [
        ("Tap on 'Wi-Fi'", mock_ui_tree),
        ("Toggle Wi-Fi ON", mock_ui_tree),
        ("Toggle Wi-Fi OFF", mock_ui_tree)
    ]
    
    for subgoal, ui_tree in test_cases:
        print(f"\nTesting: {subgoal}")
        result = verifier.verify(subgoal, ui_tree)
        print(f"Result: {result}") 