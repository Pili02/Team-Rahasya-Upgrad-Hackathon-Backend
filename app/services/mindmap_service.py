import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
import json
from app.models import MindmapNode, MindmapResponse
from app.services.llm_service import LLMService

logger = logging.getLogger(__name__)


class MindmapService:
    """Core service for mindmap generation and processing"""

    def __init__(self, resource_config: Optional[Dict[str, Any]] = None):
        self.llm_service = LLMService()

    # No resource enrichment config needed

    def generate_mindmap(
        self,
        description: str,
        max_depth: int = 3,
        time_constraint: Optional[str] = None,
    ) -> MindmapResponse:
        """Generate a complete mindmap with RAG enrichment"""

        try:
            logger.info(f"Generating mindmap for: {description}")

            # Generate initial mindmap using LLM (now with iterative expansion)
            raw_mindmap = self.llm_service.generate_mindmap(
                description=description,
                max_depth=max_depth,
                time_constraint=time_constraint,
            )

            # Enrich with RAG resources
            enriched_mindmap = self._enrich_with_resources(raw_mindmap)

            # Convert dict nodes to MindmapNode objects recursively
            def dict_to_node(node_dict):
                children = [
                    dict_to_node(child) for child in node_dict.get("children", [])
                ]
                return MindmapNode(
                    id=node_dict["id"],
                    title=node_dict["title"],
                    description=node_dict["description"],
                    resources=node_dict.get("resources", []),
                    children=children,
                )

            mindmap_nodes = [dict_to_node(node) for node in enriched_mindmap["nodes"]]

            # Calculate metrics
            total_nodes = self._count_total_nodes(mindmap_nodes)

            # Create response
            response = MindmapResponse(
                root=enriched_mindmap["root"],
                nodes=mindmap_nodes,
                total_nodes=total_nodes,
                generated_at=datetime.now().isoformat(),
            )

            logger.info(f"Generated mindmap with {total_nodes} nodes")
            return response

        except Exception as e:
            logger.error(f"Error generating mindmap: {e}")
            raise

    def generate_initial_mindmap(
        self,
        description: str,
        max_depth: int = 3,
        time_constraint: Optional[str] = None,
    ) -> MindmapResponse:
        """Generate initial mindmap structure without resource enrichment for fast response"""

        try:
            logger.info(f"Generating initial mindmap for: {description}")

            # Generate initial mindmap using LLM (now with iterative expansion)
            raw_mindmap = self.llm_service.generate_mindmap(
                description=description,
                max_depth=max_depth,
                time_constraint=time_constraint,
            )

            # Convert dict nodes to MindmapNode objects recursively (without resources)
            def dict_to_node(node_dict):
                children = [
                    dict_to_node(child) for child in node_dict.get("children", [])
                ]
                return MindmapNode(
                    id=node_dict["id"],
                    title=node_dict["title"],
                    description=node_dict["description"],
                    resources=[],  # No resources in initial response
                    children=children,
                )

            mindmap_nodes = [dict_to_node(node) for node in raw_mindmap["nodes"]]

            # Calculate metrics
            total_nodes = self._count_total_nodes(mindmap_nodes)

            # Create response
            response = MindmapResponse(
                root=raw_mindmap["root"],
                nodes=mindmap_nodes,
                total_nodes=total_nodes,
                generated_at=datetime.now().isoformat(),
            )

            logger.info(f"Generated initial mindmap with {total_nodes} nodes")
            return response

        except Exception as e:
            logger.error(f"Error generating initial mindmap: {e}")
            raise

    def enrich_mindmap_with_resources(
        self, mindmap_response: MindmapResponse
    ) -> MindmapResponse:
        """Return the mindmap as-is (no resource enrichment)."""
        logger.info(f"Resource enrichment skipped (Tavily removed)")
        return mindmap_response

    # _enrich_with_resources removed: no resource enrichment, Tavily removed

    def _count_total_nodes(self, nodes: List[MindmapNode]) -> int:
        """Count total number of nodes in the mindmap"""

        def count_recursive(node_list: List[MindmapNode]) -> int:
            count = len(node_list)
            for node in node_list:
                if node.children:
                    count += count_recursive(node.children)
            return count

        return count_recursive(nodes)

    # Removed time/complexity calculations as per updated requirements

    def explain_node(
        self, node_id: int, mindmap: MindmapResponse, context: Optional[str] = None
    ) -> Dict[str, Any]:
        """Explain why a specific node exists in the mindmap"""

        try:
            # Find the node by ID
            node = self._find_node_by_id(node_id, mindmap.nodes)

            if not node:
                raise ValueError(f"Node with ID {node_id} not found")

            # Get explanation from LLM
            explanation = self.llm_service.explain_node(node, context)

            return explanation

        except Exception as e:
            logger.error(f"Error explaining node: {e}")
            raise

    def _find_node_by_id(
        self, node_id: int, nodes: List[MindmapNode]
    ) -> Optional[MindmapNode]:
        """Find a node by ID in the mindmap"""

        def search_recursive(node_list: List[MindmapNode]) -> Optional[MindmapNode]:
            for node in node_list:
                if node.id == node_id:
                    return node
                if node.children:
                    result = search_recursive(node.children)
                    if result:
                        return result
            return None

        return search_recursive(nodes)

    def linearize_mindmap(self, mindmap: MindmapResponse) -> Dict[str, Any]:
        """Convert mindmap to a linear roadmap format"""

        def flatten_nodes(
            nodes: List[MindmapNode], level: int = 0
        ) -> List[Dict[str, Any]]:
            """Flatten hierarchical nodes into a linear list"""
            steps = []

            for node in nodes:
                step = {
                    "id": node.id,
                    "title": node.title,
                    "description": node.description,
                    "level": level,
                    "resources": node.resources,
                }
                steps.append(step)

                # Add children at next level
                if node.children:
                    steps.extend(flatten_nodes(node.children, level + 1))

            return steps

        # Flatten the mindmap
        linear_steps = flatten_nodes(mindmap.nodes)

        return {
            "steps": linear_steps,
            "total_steps": len(linear_steps),
        }

    # Statistics endpoints would need redesign without time/difficulty; omitted for now

    # Resource optimization stats omitted in simplified schema
