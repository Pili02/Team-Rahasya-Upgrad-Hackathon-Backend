import json
import logging
from typing import Dict, Any, Optional
from ollama import Client
from app.models import MindmapNode, TimeComplexity

logger = logging.getLogger(__name__)


class LLMService:
    """Service for interacting with Ollama and LLaMA 3"""

    def __init__(self, model_name: str = "llama3"):
        self.client = Client(host="http://localhost:11434")
        self.model_name = model_name

    def _create_mindmap_prompt(
        self,
        description: str,
        max_depth: int,
        focus_area: Optional[str] = None,
        time_constraint: Optional[str] = None,
    ) -> str:
        """Create a structured prompt for mindmap generation"""

        prompt = f"""You are an expert AI mentor that creates structured learning mindmaps. 
        
        Create a mindmap for the following goal: "{description}"
        
        Requirements:
        - Maximum depth: {max_depth} levels
        - Keep concepts at a higher level (not too granular)
        - Each node should have a clear title, description, and time estimate
        - Time estimates should be granular (e.g., "2-3 weeks", "1 month")
        - Include difficulty levels: Beginner, Intermediate, Advanced
        - Focus on practical, actionable steps
        """

        if focus_area:
            prompt += f"\n- Focus specifically on: {focus_area}"

        if time_constraint:
            prompt += f"\n- Time constraint: {time_constraint}"

        prompt += """
        
        Output format (JSON):
        {
          "root": "Main goal title",
          "nodes": [
            {
              "id": 1,
              "title": "Node title",
              "description": "Clear description of what this node represents",
              "time_left": "2-3 weeks",
              "difficulty": "Beginner",
              "resources": [],
              "prerequisites": [],
              "children": []
            }
          ]
        }
        
        Guidelines:
        - Make descriptions clear and actionable
        - Time estimates should be realistic
        - Difficulty should match the complexity
        - Keep the structure logical and hierarchical
        - Focus on the most important concepts first
        
        Generate the mindmap now:"""

        return prompt

    def _create_explanation_prompt(
        self, node: MindmapNode, context: Optional[str] = None
    ) -> str:
        """Create a prompt for explaining why a specific node exists"""

        prompt = f"""You are an AI mentor explaining a learning concept.
        
        Node to explain:
        - Title: {node.title}
        - Description: {node.description}
        - Time estimate: {node.time_left}
        - Difficulty: {node.difficulty}
        
        Context: {context if context else 'General learning context'}
        
        Explain:
        1. Why this concept/step is important
        2. Why it exists in the learning path
        3. How it connects to other concepts
        4. Tips for approaching this learning step
        
        Output format (JSON):
        {{
          "explanation": "Detailed explanation of why this node exists",
          "importance": "Why this node is important",
          "tips": ["Tip 1", "Tip 2", "Tip 3"]
        }}
        
        Provide a helpful, encouraging explanation:"""

        return prompt

    def generate_mindmap(
        self,
        description: str,
        max_depth: int = 3,
        focus_area: Optional[str] = None,
        time_constraint: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Generate a mindmap using LLaMA 3"""

        try:
            prompt = self._create_mindmap_prompt(
                description, max_depth, focus_area, time_constraint
            )

            logger.info(f"Generating mindmap for: {description[:50]}...")

            response = self.client.generate(
                model=self.model_name,
                prompt=prompt,
                options={"temperature": 0.7, "top_p": 0.9, "max_tokens": 2000},
            )

            # Extract the response text
            response_text = response["response"]

            # Try to parse JSON from the response
            try:
                # Find JSON content in the response
                start_idx = response_text.find("{")
                end_idx = response_text.rfind("}") + 1

                if start_idx != -1 and end_idx != 0:
                    json_str = response_text[start_idx:end_idx]
                    mindmap_data = json.loads(json_str)

                    # Validate and clean the data
                    return self._validate_mindmap_data(mindmap_data)
                else:
                    raise ValueError("No JSON content found in response")

            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON response: {e}")
                logger.error(f"Raw response: {response_text}")
                raise ValueError("Failed to generate valid mindmap structure")

        except Exception as e:
            logger.error(f"Error generating mindmap: {e}")
            raise

    def explain_node(
        self, node: MindmapNode, context: Optional[str] = None
    ) -> Dict[str, Any]:
        """Explain why a specific node exists in the mindmap"""

        try:
            prompt = self._create_explanation_prompt(node, context)

            response = self.client.generate(
                model=self.model_name,
                prompt=prompt,
                options={"temperature": 0.6, "top_p": 0.8, "max_tokens": 1000},
            )

            response_text = response["response"]

            # Parse JSON response
            try:
                start_idx = response_text.find("{")
                end_idx = response_text.rfind("}") + 1

                if start_idx != -1 and end_idx != 0:
                    json_str = response_text[start_idx:end_idx]
                    explanation_data = json.loads(json_str)
                    return explanation_data
                else:
                    raise ValueError("No JSON content found in response")

            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse explanation JSON: {e}")
                raise ValueError("Failed to generate valid explanation")

        except Exception as e:
            logger.error(f"Error explaining node: {e}")
            raise

    def _validate_mindmap_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and clean mindmap data from LLM"""

        # Ensure required fields exist
        if "root" not in data or "nodes" not in data:
            raise ValueError("Missing required fields: root and nodes")

        # Clean and validate nodes
        cleaned_nodes = []
        for i, node in enumerate(data["nodes"]):
            cleaned_node = {
                "id": node.get("id", i + 1),
                "title": node.get("title", f"Node {i + 1}"),
                "description": node.get("description", ""),
                "time_left": node.get("time_left", "1-2 weeks"),
                "difficulty": node.get("difficulty", "Beginner"),
                "resources": node.get("resources", []),
                "prerequisites": node.get("prerequisites", []),
                "children": node.get("children", []),
            }

            # Validate difficulty
            if cleaned_node["difficulty"] not in [d.value for d in TimeComplexity]:
                cleaned_node["difficulty"] = "Beginner"

            cleaned_nodes.append(cleaned_node)

        return {"root": data["root"], "nodes": cleaned_nodes}

    def test_connection(self) -> bool:
        """Test if Ollama is running and accessible"""
        try:
            models = self.client.list()
            # Accept any model whose name starts with the model_name (e.g., 'llama3', 'llama3:latest')
            return any(
                model["name"].startswith(self.model_name) for model in models["models"]
            )
        except Exception as e:
            logger.error(f"Failed to connect to Ollama: {e}")
            return False
