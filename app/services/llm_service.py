import json
import logging
import re
import asyncio
import os
from dotenv import load_dotenv
from typing import Dict, Any, Optional, List, TypedDict
from app.models import MindmapNode
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from langgraph.graph import StateGraph, END
from app.services.rag_service import RAGService
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()


logger = logging.getLogger(__name__)


class MindmapState(TypedDict):
    """Represents the state of the mindmap generation graph."""

    mindmap: Dict[str, Any]
    nodes_to_expand: List[Dict[str, Any]]
    current_depth: int
    max_depth: int


class LLMService:
    """Service for interacting with Google Gemini, optionally grounded by a vector DB."""

    def __init__(
        self,
        model_name: str = "gemini-1.5-flash",
        rag_service: Optional[RAGService] = None,
    ):
        self.llm = ChatGoogleGenerativeAI(model=model_name, temperature=0.2, top_p=0.9)
        self.model_name = model_name
        # Optional vector DB for retrieval-augmented prompts
        self.rag_service = rag_service

    # Add this new method to the LLMService class

    def _create_enrichment_prompt(
        self,
        node_title: str,
        node_description: str,
        retrieval_context: str,
    ) -> str:
        """Creates a prompt to enrich a single mindmap node using RAG context."""
        return (
            "You are an AI curriculum enhancer. Your task is to take a generic mindmap node and rewrite its description to be more specific, factual, and detailed using the provided reference context.\n\n"
            f"## Original Node ##\n"
            f'- **Title:** "{node_title}"\n'
            f'- **Description:** "{node_description}"\n\n'
            "## Reference Context (from knowledge base) ##\n"
            f"---\n{retrieval_context}\n---\n\n"
            "## Your Task ##\n"
            "1.  Read the reference context carefully.\n"
            "2.  Rewrite the original description to incorporate specific facts, terminology, or examples from the context.\n"
            "3.  The new description should remain concise and actionable but be grounded in the provided information.\n"
            "4.  Do not change the title.\n\n"
            "## Output Format (JSON) ##\n"
            "{\n"
            '  "new_description": "Your rewritten, fact-based description goes here."\n'
            "}\n\n"
            "Generate only the valid JSON object:"
        )

    def _create_initial_mindmap_prompt(
        self,
        description: str,
        max_depth: int,  # While max_depth is a parameter, this initial prompt is hardcoded to 1 level.
        retrieval_context: Optional[str] = None,
        time_constraint: Optional[str] = None,
    ) -> str:
        """
        Create a highly structured and guided prompt for initial mindmap generation.
        This version is designed to produce a more logical node structure and strictly
        adhere to focus areas and output constraints.
        """
        # Dynamically create the main task instruction based on whether a focus_area is provided.
        # This makes the focus_area a non-negotiable part of the core request.
        main_task_block = (
            f'Your primary goal is to learn: "{description}".\n'
            f"Deconstruct this goal into its most critical, high-level components."
        )

        context_block = (
            f"\nUse this supplementary context to inform the mindmap:\n---\n{retrieval_context}\n---\n"
            if retrieval_context
            else ""
        )
        time_block = (
            f"The learner has a time constraint of: {time_constraint}.\n"
            if time_constraint
            else ""
        )

        # The prompt itself, with stronger instructions and guidance.
        return (
            "You are an expert AI curriculum architect. Your task is to design the initial high-level structure of a learning mindmap in perfect JSON format.\n\n"
            "## LEARNING OBJECTIVE ##\n"
            f"{main_task_block}\n"
            f"{time_block}"
            f"{context_block}"
            "\n## STRUCTURAL GUIDANCE ##\n"
            "To create a logical and effective learning path, structure the nodes by breaking the topic into logical, sequential modules. A good pattern to follow is:\n"
            "1.  **Foundations:** What are the absolute core concepts of the focus area?\n"
            "2.  **Core Components / Architecture:** What are the key building blocks or parts?\n"
            "3.  **Practical Application:** How do you actually use it to achieve the goal?\n"
            "4.  **Key Patterns / Best Practices:** What are common ways of working with this technology?\n\n"
            "## CRITICAL OUTPUT REQUIREMENTS ##\n"
            "1.  **Depth Limit:** The mindmap MUST be flat. Generate ONLY the first level of nodes. DO NOT create any 'children' of children (i.e., max_depth = 1).\n"
            "2.  **Content:** Each node needs a short `title` and a `description` that explains what the learner will do or understand in that step.\n"
            "3.  **NO URLs:** The `resources` key for ALL nodes MUST be an empty list `[]`. Do not invent or add any URLs, books, or links.\n"
            "4.  **Strict JSON:** You must output nothing but a single, valid JSON object. Do not add any text before or after the JSON.\n\n"
            "## REQUIRED JSON FORMAT ##\n"
            "```json\n"
            "{\n"
            '  "root": "Title for the Main Goal",\n'
            '  "nodes": [\n'
            "    {\n"
            '      "id": 1,\n'
            '      "title": "Module 1 Title (e.g., Foundations of X)",\n'
            '      "description": "Actionable description of what to learn or do in this module.",\n'
            '      "resources": [], // MUST be an empty list\n'
            '      "children": [] // MUST be an empty list for this initial generation\n'
            "    },\n"
            "    {\n"
            '      "id": 2,\n'
            '      "title": "Module 2 Title (e.g., Core Architecture of X)",\n'
            '      "description": "Actionable description of what to learn or do in this module.",\n'
            '      "resources": [],\n'
            '      "children": []\n'
            "    }\n"
            "  ]\n"
            "}\n"
            "```\n\n"
            "Now, generate the JSON for the learning goal specified above."
        )

    def _create_node_expansion_prompt(
        self,
        parent_title: str,
        parent_description: str,
        parent_id: int,
        retrieval_context: Optional[str] = None,
        time_constraint: Optional[str] = None,
    ) -> str:
        """Create a prompt for expanding a single node, with optional retrieval context."""
        context_block = (
            f"\nReference resources (from vector DB):\n{retrieval_context}\n\n"
            if retrieval_context
            else ""
        )
        time_block = f"Time constraint: {time_constraint}.\n" if time_constraint else ""
        return (
            "You are an expert AI mentor that expands a learning mindmap.\n\n"
            "Expand the following parent node by generating its children. Keep them strictly on-topic and specific to the parent.\n\n"
            f'Parent Node:\n- Title: "{parent_title}"\n- Description: "{parent_description}"\n- ID: {parent_id}\n\n'
            + time_block
            + context_block
            + "Requirements for children nodes:\n"
            "- Each child must be a direct sub-concept or next step of this parent (no generic or unrelated topics).\n"
            "- Each child must have a concise title and actionable description.\n"
            "- Set resources to an empty list [].\n"
            "- Use a placeholder ID of -1; the system will assign real IDs.\n\n"
            "Output format (JSON, CHILDREN only):\n"
            "{\n"
            '  "children": [\n'
            "    {\n"
            '      "id": -1,\n'
            '      "title": "Child node title",\n'
            '      "description": "Description of the child node",\n'
            '      "resources": [],\n'
            '      "children": []\n'
            "    }\n"
            "  ]\n"
            "}\n\n"
            "Return only valid JSON:"
        )

    def _build_retrieval_context(self, query: str, top_k: int = 5) -> Optional[str]:
        """Fetch top-k resources from the vector DB and build a compact reference block."""
        if not self.rag_service:
            return None
        try:
            # Use a relevance threshold to avoid polluting the prompt with generic resources
            resources = self.rag_service.search_resources(
                query=query, limit=top_k, min_score=0.65
            )
            if not resources:
                return None
            lines: List[str] = []
            for i, r in enumerate(resources, start=1):
                title = r.get("title", "")
                desc = r.get("description", "")
                url = r.get("url", "")
                snippet = desc or ""
                # Keep it short to avoid blowing the context window
                if len(snippet) > 160:
                    snippet = snippet[:157] + "..."
                lines.append(f"{i}. {title} — {snippet}\n   Link: {url}")
            return "\n".join(lines)
        except Exception as e:
            logger.warning(f"Retrieval context failed for query '{query}': {e}")
            return None

    def _call_llm(self, prompt: str) -> Dict[str, Any]:
        """A reusable function to call the Gemini LLM and parse the JSON response."""
        response = self.llm.invoke(prompt)
        response_text = response.content
        try:

            def _strip_code_fences(text: str) -> str:
                lines = []
                for line in text.splitlines():
                    if line.strip().startswith("```"):
                        # drop fence lines
                        continue
                    lines.append(line)
                return "\n".join(lines)

            def _extract_first_json_object(text: str) -> Optional[str]:
                start = text.find("{")
                if start == -1:
                    return None
                depth = 0
                for i in range(start, len(text)):
                    ch = text[i]
                    if ch == "{":
                        depth += 1
                    elif ch == "}":
                        depth -= 1
                        if depth == 0:
                            return text[start : i + 1]
                return None

            cleaned = _strip_code_fences(response_text)
            json_str = _extract_first_json_object(cleaned)
            if json_str:
                try:
                    data = json.loads(json_str)
                except json.JSONDecodeError:
                    # Repairs: bare URLs and trailing commas
                    repaired = re.sub(
                        r"(\"(?:url|link)\"\s*:\s*)(https?://[^\s,}\]]+)",
                        r'\1"\2"',
                        json_str,
                    )
                    repaired = re.sub(r",\s*([}\]])", r"\1", repaired)
                    data = json.loads(repaired)

                # Sanitize resource fields to be a list of URLs if model returns objects
                def sanitize_node(node: Dict[str, Any]):
                    resources = node.get("resources", [])
                    # We don't accept LLM-provided resources; enrichment will fill them later
                    node["resources"] = []
                    # Recurse children
                    for child in node.get("children", []) or []:
                        if isinstance(child, dict):
                            sanitize_node(child)

                if isinstance(data, dict) and "nodes" in data:
                    for n in data.get("nodes", []) or []:
                        if isinstance(n, dict):
                            sanitize_node(n)
                if isinstance(data, dict) and "children" in data:
                    for n in data.get("children", []) or []:
                        if isinstance(n, dict):
                            sanitize_node(n)
                return data
            else:
                raise ValueError("No JSON content found in response")
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            logger.error(f"Raw response: {response_text}")
            raise ValueError("Failed to generate valid mindmap structure")

    def _call_llm_parallel(self, prompts: List[str]) -> List[Dict[str, Any]]:
        """Call LLM in parallel for multiple prompts to speed up node expansion."""
        import concurrent.futures
        import threading

        def call_single_llm(prompt: str) -> Dict[str, Any]:
            """Wrapper for single LLM call to be used in thread pool."""
            try:
                return self._call_llm(prompt)
            except Exception as e:
                logger.error(f"Parallel LLM call failed: {e}")
                return {"children": []}  # Return empty children on failure

        # Use ThreadPoolExecutor for parallel execution
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=min(len(prompts), 5)
        ) as executor:
            # Submit all prompts for parallel execution
            future_to_prompt = {
                executor.submit(call_single_llm, prompt): prompt for prompt in prompts
            }

            # Collect results in the same order as input prompts
            results = []
            for future in concurrent.futures.as_completed(future_to_prompt):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    logger.error(f"Error in parallel LLM execution: {e}")
                    results.append({"children": []})

        return results

    def generate_mindmap(
        self,
        description: str,
        max_depth: int = 3,
        time_constraint: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Generate a complete mindmap using an iterative LangGraph process."""
        try:
            # 1. Define the graph nodes
            def generate_initial_mindmap_node(state: MindmapState) -> MindmapState:
                # STEP 1: Generate skeleton WITHOUT RAG context
                prompt = self._create_initial_mindmap_prompt(
                    state["mindmap"]["root"],
                    state["max_depth"],
                    retrieval_context=None,  # No RAG here
                    time_constraint=state.get("time_constraint"),
                )
                raw_mindmap = self._call_llm(prompt)
                all_nodes = raw_mindmap.get("nodes", [])
                self._assign_unique_ids_recursive(all_nodes, 1)

                state["mindmap"]["root"] = raw_mindmap.get("root", "Learning Goal")
                state["mindmap"]["nodes"] = all_nodes
                # We don't add to nodes_to_expand yet, that happens after enrichment
                return state

            # NEW NODE for enrichment
            def enrich_mindmap_node(state: MindmapState) -> MindmapState:
                nodes_to_enrich = state["mindmap"].get("nodes", [])
                for node in nodes_to_enrich:
                    # Build context specifically for this node
                    query = f"{node.get('title', '')}"
                    retrieval_context = self._build_retrieval_context(query, top_k=3)

                    if retrieval_context:
                        enrich_prompt = self._create_enrichment_prompt(
                            node_title=node.get("title", ""),
                            node_description=node.get("description", ""),
                            retrieval_context=retrieval_context,
                        )
                        try:
                            enrichment_response = self._call_llm(enrich_prompt)
                            if "new_description" in enrichment_response:
                                # Update the node's description in the state
                                node["description"] = enrichment_response[
                                    "new_description"
                                ]
                        except Exception as e:
                            logger.warning(
                                f"Failed to enrich node ID {node.get('id')}: {e}"
                            )
                            # Continue even if one node fails to enrich

                # Now that nodes are enriched, populate the expansion queue
                state["nodes_to_expand"].extend(nodes_to_enrich)
                state["current_depth"] = 1
                return state

            def expand_nodes_parallel(state: MindmapState) -> MindmapState:
                """Expand all nodes at the current level in parallel for much faster processing."""
                nodes_to_expand = state["nodes_to_expand"]
                if not nodes_to_expand:
                    return state

                # Get all nodes for this level (clear the queue)
                current_level_nodes = nodes_to_expand.copy()
                state["nodes_to_expand"] = []

                # Prepare prompts for all nodes at this level
                prompts = []
                node_queries = []

                for node in current_level_nodes:
                    node_query = (
                        f"{node.get('title', '')} {node.get('description', '')}"
                    )
                    node_queries.append(node_query)
                    retrieval_context = self._build_retrieval_context(
                        node_query, top_k=5
                    )
                    prompt = self._create_node_expansion_prompt(
                        node.get("title", ""),
                        node.get("description", ""),
                        node["id"],
                        retrieval_context,
                        time_constraint=state.get("time_constraint"),
                    )
                    prompts.append(prompt)

                # Call LLM in parallel for all nodes at this level
                logger.info(f"Expanding {len(prompts)} nodes in parallel...")
                children_responses = self._call_llm_parallel(prompts)

                # Process all results and assign IDs
                last_id = self._find_max_id(state["mindmap"]["nodes"])
                node_id_counter = last_id + 1
                all_new_children = []

                for node, children_response in zip(
                    current_level_nodes, children_responses
                ):
                    children = children_response.get("children", [])

                    # Assign unique IDs to children
                    for child in children:
                        child["id"] = node_id_counter
                        node_id_counter += 1

                    # Add children to the mindmap
                    self._add_children_to_mindmap(
                        state["mindmap"]["nodes"], node["id"], children
                    )

                    # Collect all new children for next level expansion
                    all_new_children.extend(children)

                # Add all new children to the queue for the next level
                state["nodes_to_expand"].extend(all_new_children)
                state["current_depth"] = self._find_max_depth(state["mindmap"]["nodes"])

                logger.info(
                    f"Parallel expansion completed. Added {len(all_new_children)} children."
                )
                return state

            def should_continue(state: MindmapState) -> str:
                if (
                    not state["nodes_to_expand"]
                    or state["current_depth"] >= state["max_depth"]
                ):
                    return "end"
                return "continue"

            # 2. Build the graph with the new enrichment step and parallel expansion
            workflow = StateGraph(MindmapState)
            workflow.add_node("initial", generate_initial_mindmap_node)
            workflow.add_node("enrich", enrich_mindmap_node)  # Add the new node
            workflow.add_node("expand", expand_nodes_parallel)  # Use parallel expansion

            workflow.set_entry_point("initial")
            workflow.add_edge("initial", "enrich")  # Wire initial -> enrich

            workflow.add_conditional_edges(
                "enrich",  # The decision point is now after enrichment
                should_continue,
                {"continue": "expand", "end": END},
            )
            workflow.add_conditional_edges(
                "expand",
                should_continue,
                {"continue": "expand", "end": END},
            )

            app = workflow.compile()

            # 3. Run the graph
            final_state = app.invoke(
                {
                    "mindmap": {"root": description, "nodes": []},
                    "nodes_to_expand": [],
                    "current_depth": 0,
                    "max_depth": max_depth,
                    "time_constraint": time_constraint,
                }
            )

            return final_state["mindmap"]

        except Exception as e:
            logger.error(f"Error generating mindmap: {e}")
            raise

    def _find_max_id(self, nodes: List[Dict[str, Any]]) -> int:
        """Recursively finds the maximum ID in the mindmap nodes."""
        max_id = 0
        for node in nodes:
            max_id = max(max_id, node.get("id", 0))
            if "children" in node and isinstance(node["children"], list):
                max_id = max(max_id, self._find_max_id(node["children"]))
        return max_id

    def _assign_unique_ids_recursive(
        self, nodes: List[Dict[str, Any]], id_counter: int
    ) -> int:
        """Recursively traverses nodes and assigns new sequential IDs."""
        for node in nodes:
            node["id"] = id_counter
            id_counter += 1
            if "children" in node and isinstance(node["children"], list):
                id_counter = self._assign_unique_ids_recursive(
                    node["children"], id_counter
                )
        return id_counter

    # app/services/llm_service.py

    def _add_children_to_mindmap(
        self,
        mindmap_nodes: List[Dict[str, Any]],
        parent_id: int,
        children: List[Dict[str, Any]],
    ):
        """
        Recursively finds a parent node and adds children to it.
        This version safely handles nodes that may be missing the 'children' key.
        """
        for node in mindmap_nodes:
            if node.get("id") == parent_id:
                # Use setdefault to ensure 'children' key exists and is a list
                # before extending it. If the key doesn't exist, it's created
                # with an empty list as its value.
                node.setdefault("children", []).extend(children)
                return
            if "children" in node and isinstance(node["children"], list):
                self._add_children_to_mindmap(node["children"], parent_id, children)

    def _find_max_depth(
        self, nodes: List[Dict[str, Any]], current_depth: int = 1
    ) -> int:
        """Recursively finds the maximum depth of the mindmap."""
        max_depth = current_depth
        for node in nodes:
            if (
                "children" in node
                and isinstance(node["children"], list)
                and node["children"]
            ):
                max_depth = max(
                    max_depth, self._find_max_depth(node["children"], current_depth + 1)
                )
        return max_depth

    def test_connection(self) -> bool:
        """Test if the Gemini API is accessible."""
        try:
            self.llm.invoke("test")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to Gemini: {e}")
            return False

    def explain_node(
        self, node: MindmapNode, context: Optional[str] = None
    ) -> Dict[str, Any]:
        """Explain why a specific node exists in the mindmap"""
        # This method is not changed
        try:
            prompt = self._create_explanation_prompt(node, context)

            response = self.llm.invoke(prompt)
            response_text = response.content

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

    def _create_explanation_prompt(
        self, node: MindmapNode, context: Optional[str] = None
    ) -> str:
        """Create a prompt for explaining why a specific node exists"""

        prompt = f"""You are an AI mentor explaining a learning concept.
        
        Node to explain:
        - Title: {node.title}
        - Description: {node.description}
        
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
