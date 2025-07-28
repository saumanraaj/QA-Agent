import json
import logging
import re
from typing import Dict, Optional, Any, List, Tuple
from difflib import SequenceMatcher
from android_world.env import env_launcher
from android_world.env import interface
from android_world.env import json_action
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agents.llm_engine import get_llm_engine, AgentPrompts

logger = logging.getLogger(__name__)

class ExecutorAgent:
    """Enhanced LLM-powered executor agent with intelligent fuzzy matching and multiple strategies"""
    
    def __init__(self, engine_type: str = "openai", model: str = "gpt-4o-mini"):
        self.llm_engine = get_llm_engine(engine_type=engine_type, model=model)
        
        # Element matching strategies in order of preference
        self.matching_strategies = [
            ("exact_match", self._exact_match),
            ("fuzzy_match", self._fuzzy_match),
            ("semantic_match", self._semantic_match),
            ("resource_id_match", self._resource_id_match),
            ("partial_text_match", self._partial_text_match),
            ("contextual_match", self._contextual_match)
        ]
        
        # Dynamic learning system for element variations
        self.element_variations = {}  # Learned variations
        self.successful_matches = {}  # Track successful strategies
        self.failure_patterns = {}    # Track failure patterns for learning
    
    def execute(self, subgoal: str, ui_tree: dict, env: interface.AsyncAndroidEnv) -> Optional[Dict[str, Any]]:
        """Execute a subgoal using enhanced LLM grounding with intelligent fallback strategies"""
        
        system_prompt = AgentPrompts.get_executor_prompt()

        # Extract UI elements for analysis
        ui_elements = self._extract_ui_elements(ui_tree)
        
        user_prompt = f"""Subgoal: {subgoal}

Available UI Elements: {ui_elements}

Identify the best element to interact with for this subgoal. Consider multiple strategies if exact match fails."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        try:
            response = self.llm_engine.generate(messages, temperature=0.1, max_tokens=300)
            
            # Parse JSON response
            try:
                result = json.loads(response.strip())
                if isinstance(result, dict) and "element_id" in result:
                    logger.info(f"LLM grounding result for '{subgoal}': {result}")
                    return self._execute_action_with_fallbacks(result, subgoal, ui_tree, env)
                else:
                    raise ValueError("Response missing 'element_id' field")
            except json.JSONDecodeError:
                # Try to extract JSON from response
                import re
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if json_match:
                    result = json.loads(json_match.group())
                    logger.info(f"LLM grounding result for '{subgoal}': {result}")
                    return self._execute_action_with_fallbacks(result, subgoal, ui_tree, env)
                else:
                    raise ValueError("Could not parse JSON from LLM response")
                    
        except Exception as e:
            logger.error(f"LLM grounding failed: {e}")
            raise ValueError(f"LLM grounding failed for subgoal '{subgoal}': {e}")
    
    def _execute_action_with_fallbacks(self, llm_result: Dict[str, Any], subgoal: str, ui_tree: dict, env: interface.AsyncAndroidEnv) -> Optional[Dict[str, Any]]:
        """Execute action with multiple intelligent fallback strategies"""
        
        primary_element_id = llm_result.get("element_id")
        if not primary_element_id:
            logger.warning(f"No element found for subgoal: {subgoal}")
            return None
        
        # Try primary element first
        element = self._find_element_by_id(primary_element_id, ui_tree)
        if element:
            return self._execute_action(llm_result, subgoal, ui_tree, env, element)
        
        # If primary element not found, try fallback strategies
        logger.info(f"Primary element '{primary_element_id}' not found, trying intelligent fallback strategies...")
        
        for strategy_name, strategy_func in self.matching_strategies[1:]:  # Skip exact_match since we already tried it
            try:
                alternative_element = strategy_func(primary_element_id, ui_tree, subgoal)
                if alternative_element:
                    logger.info(f"Found alternative using {strategy_name}: {alternative_element.get('text', 'Unknown')}")
                    
                    # Update the action with the found element
                    updated_result = llm_result.copy()
                    updated_result["element_id"] = alternative_element.get('text', '')
                    updated_result["fallback_strategy"] = strategy_name
                    
                    return self._execute_action(updated_result, subgoal, ui_tree, env, alternative_element)
            except Exception as e:
                logger.warning(f"Strategy {strategy_name} failed: {e}")
                continue
        
        logger.warning(f"All intelligent fallback strategies failed for subgoal: {subgoal}")
        return None
    
    def _exact_match(self, element_id: str, ui_tree: dict, subgoal: str = "") -> Optional[dict]:
        """Exact text match"""
        def search_element(node, target):
            if isinstance(node, dict):
                if node.get('text', '').lower() == target.lower():
                    return node
                for key, value in node.items():
                    if isinstance(value, (dict, list)):
                        result = search_element(value, target)
                        if result:
                            return result
            elif isinstance(node, list):
                for item in node:
                    result = search_element(item, target)
                    if result:
                        return result
            return None
        
        return search_element(ui_tree, element_id)
    
    def _fuzzy_match(self, element_id: str, ui_tree: dict, subgoal: str = "") -> Optional[dict]:
        """Fuzzy text matching using similarity scores"""
        best_match = None
        best_score = 0.0
        
        def search_fuzzy(node, target):
            nonlocal best_match, best_score
            
            if isinstance(node, dict):
                if node.get('text'):
                    score = SequenceMatcher(None, target.lower(), node['text'].lower()).ratio()
                    if score > best_score and score > 0.6:  # Threshold for fuzzy matching
                        best_score = score
                        best_match = node
                
                for key, value in node.items():
                    if isinstance(value, (dict, list)):
                        search_fuzzy(value, target)
            elif isinstance(node, list):
                for item in node:
                    search_fuzzy(item, target)
        
        search_fuzzy(ui_tree, element_id)
        return best_match
    
    def _semantic_match(self, element_id: str, ui_tree: dict, subgoal: str = "") -> Optional[dict]:
        """Dynamic semantic matching using learned patterns and context"""
        target_lower = element_id.lower()
        
        # Use learned variations if available
        if target_lower in self.element_variations:
            for variation in self.element_variations[target_lower]:
                element = self._exact_match(variation, ui_tree)
                if element:
                    self._record_successful_match(element_id, variation, "semantic_match")
                    return element
        
        # Dynamic pattern learning from available elements
        available_elements = self._extract_ui_elements(ui_tree)
        potential_matches = []
        
        for element_info in available_elements:
            element_text = element_info.get('text', '').lower()
            if element_text:
                # Calculate semantic similarity based on shared words and context
                similarity = self._calculate_semantic_similarity(target_lower, element_text, subgoal)
                if similarity > 0.1:  # Much lower threshold for dynamic learning
                    potential_matches.append((element_info, similarity))
        
        # Sort by similarity and try the best matches
        potential_matches.sort(key=lambda x: x[1], reverse=True)
        
        for element_info, similarity in potential_matches:
            if similarity > 0.2:  # Lower threshold for actual selection
                # Learn this variation
                self._learn_element_variation(element_id, element_info.get('text', ''))
                self._record_successful_match(element_id, element_info.get('text', ''), "semantic_match")
                return self._find_element_by_text(element_info.get('text', ''), ui_tree)
        
        return None
    
    def _calculate_semantic_similarity(self, target: str, element_text: str, subgoal: str = "") -> float:
        """Calculate semantic similarity between target and element text"""
        target_words = set(re.findall(r'\b\w+\b', target.lower()))
        element_words = set(re.findall(r'\b\w+\b', element_text.lower()))
        
        # Word overlap similarity
        if not target_words or not element_words:
            return 0.0
        
        overlap = len(target_words.intersection(element_words))
        union = len(target_words.union(element_words))
        
        if union == 0:
            return 0.0
        
        word_similarity = overlap / union
        
        # Context similarity based on subgoal
        context_similarity = 0.0
        if subgoal:
            subgoal_words = set(re.findall(r'\b\w+\b', subgoal.lower()))
            context_overlap = len(subgoal_words.intersection(element_words))
            if len(subgoal_words) > 0:
                context_similarity = context_overlap / len(subgoal_words)
        
        # Partial word matching (for cases like "Network & Internet" vs "Network Options")
        partial_matches = 0
        for target_word in target_words:
            for element_word in element_words:
                if len(target_word) > 2 and len(element_word) > 2:
                    if target_word in element_word or element_word in target_word:
                        partial_matches += 1
        
        partial_similarity = partial_matches / len(target_words) if target_words else 0.0
        
        # Combined similarity score with partial matching
        return max(word_similarity, partial_similarity) * 0.6 + context_similarity * 0.4
    
    def _learn_element_variation(self, target: str, found_element: str):
        """Learn new element variations dynamically"""
        target_lower = target.lower()
        found_lower = found_element.lower()
        
        if target_lower not in self.element_variations:
            self.element_variations[target_lower] = []
        
        if found_lower not in self.element_variations[target_lower]:
            self.element_variations[target_lower].append(found_lower)
            logger.info(f"Learned new variation: '{target}' -> '{found_element}'")
    
    def _record_successful_match(self, target: str, found_element: str, strategy: str):
        """Record successful matches for strategy optimization"""
        key = f"{target}_{strategy}"
        if key not in self.successful_matches:
            self.successful_matches[key] = 0
        self.successful_matches[key] += 1
        logger.info(f"Successful {strategy} match: '{target}' -> '{found_element}'")
    
    def _find_element_by_text(self, text: str, ui_tree: dict) -> Optional[dict]:
        """Find element by exact text match"""
        def search_element(node, target):
            if isinstance(node, dict):
                if node.get('text', '').lower() == target.lower():
                    return node
                for key, value in node.items():
                    if isinstance(value, (dict, list)):
                        result = search_element(value, target)
                        if result:
                            return result
            elif isinstance(node, list):
                for item in node:
                    result = search_element(item, target)
                    if result:
                        return result
            return None
        
        return search_element(ui_tree, text)
    
    def _resource_id_match(self, element_id: str, ui_tree: dict, subgoal: str = "") -> Optional[dict]:
        """Match by resource ID patterns"""
        def search_resource_id(node, target):
            if isinstance(node, dict):
                resource_id = node.get('resource-id', '')
                if resource_id and target.lower() in resource_id.lower():
                    return node
                
                for key, value in node.items():
                    if isinstance(value, (dict, list)):
                        result = search_resource_id(value, target)
                        if result:
                            return result
            elif isinstance(node, list):
                for item in node:
                    result = search_resource_id(item, target)
                    if result:
                        return result
            return None
        
        return search_resource_id(ui_tree, element_id)
    
    def _partial_text_match(self, element_id: str, ui_tree: dict, subgoal: str = "") -> Optional[dict]:
        """Partial text matching"""
        def search_partial(node, target):
            if isinstance(node, dict):
                text = node.get('text', '')
                if text and (target.lower() in text.lower() or text.lower() in target.lower()):
                    return node
                
                for key, value in node.items():
                    if isinstance(value, (dict, list)):
                        result = search_partial(value, target)
                        if result:
                            return result
            elif isinstance(node, list):
                for item in node:
                    result = search_partial(item, target)
                    if result:
                        return result
            return None
        
        return search_partial(ui_tree, element_id)
    
    def _contextual_match(self, element_id: str, ui_tree: dict, subgoal: str = "") -> Optional[dict]:
        """Dynamic contextual matching based on learned patterns and subgoal context"""
        # Extract dynamic context from subgoal
        context_keywords = self._extract_dynamic_context_keywords(subgoal, element_id)
        
        # Get all available elements
        available_elements = self._extract_ui_elements(ui_tree)
        best_match = None
        best_score = 0.0
        
        for element_info in available_elements:
            element_text = element_info.get('text', '')
            if not element_text or not element_info.get('clickable', False):
                continue
            
            # Calculate contextual relevance score
            context_score = self._calculate_contextual_relevance(
                element_text, context_keywords, subgoal, element_id
            )
            
            if context_score > best_score and context_score > 0.4:
                best_score = context_score
                best_match = element_info
        
        if best_match:
            # Learn this contextual pattern
            self._learn_contextual_pattern(element_id, best_match.get('text', ''), subgoal)
            self._record_successful_match(element_id, best_match.get('text', ''), "contextual_match")
            return self._find_element_by_text(best_match.get('text', ''), ui_tree)
        
        return None
    
    def _calculate_contextual_relevance(self, element_text: str, context_keywords: List[str], 
                                      subgoal: str, target_element: str) -> float:
        """Calculate contextual relevance score for an element"""
        element_lower = element_text.lower()
        target_lower = target_element.lower()
        
        # Direct keyword matches
        keyword_matches = sum(1 for keyword in context_keywords if keyword.lower() in element_lower)
        keyword_score = keyword_matches / len(context_keywords) if context_keywords else 0.0
        
        # Semantic similarity with target
        semantic_score = self._calculate_semantic_similarity(target_lower, element_lower, subgoal)
        
        # Subgoal relevance
        subgoal_relevance = 0.0
        if subgoal:
            subgoal_words = set(re.findall(r'\b\w+\b', subgoal.lower()))
            element_words = set(re.findall(r'\b\w+\b', element_lower))
            if subgoal_words:
                subgoal_relevance = len(subgoal_words.intersection(element_words)) / len(subgoal_words)
        
        # Combined score with weights
        return (keyword_score * 0.4) + (semantic_score * 0.4) + (subgoal_relevance * 0.2)
    
    def _extract_dynamic_context_keywords(self, subgoal: str, target_element: str) -> List[str]:
        """Extract dynamic context keywords based on subgoal and target"""
        keywords = []
        
        # Extract words from subgoal and target
        all_words = re.findall(r'\b\w+\b', f"{subgoal} {target_element}".lower())
        
        # Filter out common stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        keywords = [word for word in all_words if word not in stop_words and len(word) > 2]
        
        # Add domain-specific keywords based on content
        if any(word in subgoal.lower() for word in ['wifi', 'wi-fi', 'network', 'internet']):
            keywords.extend(['network', 'wireless', 'connection', 'internet'])
        
        if any(word in subgoal.lower() for word in ['settings', 'preferences', 'configure']):
            keywords.extend(['settings', 'preferences', 'options', 'configuration'])
        
        if any(word in subgoal.lower() for word in ['open', 'navigate', 'go', 'tap', 'click']):
            keywords.extend(['open', 'navigate', 'tap', 'click'])
        
        return list(set(keywords))  # Remove duplicates
    
    def _learn_contextual_pattern(self, target: str, found_element: str, subgoal: str):
        """Learn contextual patterns for future matching"""
        pattern_key = f"{target}_{subgoal[:20]}"  # Truncate subgoal for key
        
        if pattern_key not in self.failure_patterns:
            self.failure_patterns[pattern_key] = []
        
        if found_element not in self.failure_patterns[pattern_key]:
            self.failure_patterns[pattern_key].append(found_element)
            logger.info(f"Learned contextual pattern: '{target}' in '{subgoal}' -> '{found_element}'")
    
    def get_learning_statistics(self) -> Dict[str, Any]:
        """Get statistics about the learning system"""
        return {
            "learned_variations": len(self.element_variations),
            "variation_details": self.element_variations,
            "successful_matches": self.successful_matches,
            "failure_patterns": len(self.failure_patterns),
            "pattern_details": self.failure_patterns
        }
    
    def reset_learning(self):
        """Reset all learned patterns (for testing)"""
        self.element_variations.clear()
        self.successful_matches.clear()
        self.failure_patterns.clear()
        logger.info("Learning system reset")
    
    def _extract_ui_elements(self, ui_tree: dict) -> List[Dict[str, Any]]:
        """Extract all interactive UI elements from UI tree"""
        elements = []
        
        def extract_elements(node, path=""):
            if isinstance(node, dict):
                element_info = {
                    "text": node.get('text', ''),
                    "class": node.get('class', ''),
                    "resource_id": node.get('resource-id', ''),
                    "bounds": node.get('bounds', []),
                    "clickable": node.get('clickable', False),
                    "path": path
                }
                if element_info["text"] or element_info["clickable"]:
                    elements.append(element_info)
                
                for key, value in node.items():
                    if isinstance(value, (dict, list)):
                        extract_elements(value, f"{path}.{key}" if path else key)
            elif isinstance(node, list):
                for i, item in enumerate(node):
                    extract_elements(item, f"{path}[{i}]" if path else f"[{i}]")
        
        extract_elements(ui_tree)
        return elements
    
    def _execute_action(self, llm_result: Dict[str, Any], subgoal: str, ui_tree: dict, env: interface.AsyncAndroidEnv, element: dict = None) -> Optional[Dict[str, Any]]:
        """Execute the action based on LLM grounding result using QualGent challenge format"""
        
        if not element:
            element = self._find_element_by_id(llm_result["element_id"], ui_tree)
        
        if not element:
            logger.warning(f"Element {llm_result['element_id']} not found in UI tree")
            return None
        
        try:
            # QualGent challenge format: env.step({"action_type": "touch", "element_id": "<ui_element_id>"})
            action_type = llm_result.get("action_type", "touch")
            
            # Map action types to QualGent format
            if action_type == "click":
                action_type = "touch"
            elif action_type == "toggle":
                action_type = "touch"
            
            # Create action in QualGent challenge format
            action = {
                "action_type": action_type,
                "element_id": llm_result["element_id"]
            }
            
            logger.info(f"Executing {action_type} on element: {element.get('text', 'Unknown')}")
            logger.info(f"Element ID: {llm_result['element_id']}")
            
            # QualGent challenge format: env.step(action)
            observation = env.step(action)
            
            return {
                "observation": observation,
                "executed_action": action,
                "element_info": {
                    "text": element.get('text', ''),
                    "bounds": element.get('bounds', []),
                    "confidence": llm_result.get("confidence", 0.0)
                },
                "llm_reasoning": llm_result.get("reason", ""),
                "fallback_strategy": llm_result.get("fallback_strategy", "primary")
            }
            
        except Exception as e:
            logger.error(f"Action execution failed: {e}")
            return None
    
    def _find_element_by_id(self, element_id: str, ui_tree: dict) -> Optional[dict]:
        """Find element in UI tree using multiple intelligent strategies"""
        # Try all strategies in order
        for strategy_name, strategy_func in self.matching_strategies:
            try:
                element = strategy_func(element_id, ui_tree)
                if element:
                    logger.info(f"Found element using {strategy_name}: {element.get('text', 'Unknown')}")
                    return element
            except Exception as e:
                logger.warning(f"Strategy {strategy_name} failed: {e}")
                continue
        
        logger.warning(f"All strategies failed to find element: {element_id}")
        return None

if __name__ == "__main__":
    from main import main as setup_env
    
    env = env_launcher.load_and_setup_env(
        console_port=5554,
        emulator_setup=False,
        freeze_datetime=True
    )
    observation = env.reset(go_home=True)
    
    executor = ExecutorAgent()
    result = executor.execute("Tap on 'Wi-Fi'", observation, env)
    
    if result:
        print("Action executed successfully")
    else:
        print("Action failed - element not found") 