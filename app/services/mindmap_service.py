import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
import json
from app.models import MindmapNode, MindmapResponse
from app.services.llm_service import LLMService
from app.services.rag_service import RAGService

logger = logging.getLogger(__name__)


class MindmapService:
    """Core service for mindmap generation and processing"""

    def __init__(self, resource_config: Optional[Dict[str, Any]] = None):
        # Initialize RAG first so we can inject it into the LLM for retrieval-augmented prompts
        self.rag_service = RAGService()
        self.llm_service = LLMService(rag_service=self.rag_service)

        # Configure resource search behavior
        self.resource_config = resource_config or {
            "min_relevance_score": 0.7,  # Stricter similarity score for existing resources
            "min_resources_needed": 2,  # Minimum number of quality resources needed
            "always_search_external": False,  # Avoid unnecessary external search
            "max_external_results": 3,  # Maximum results to fetch from external sources
        }

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
        """Enrich an existing mindmap with resources (for background processing)"""

        try:
            logger.info(f"Enriching mindmap with resources...")

            # Convert MindmapNode objects back to dict format for enrichment
            def node_to_dict(node: MindmapNode) -> Dict[str, Any]:
                children = [node_to_dict(child) for child in node.children]
                return {
                    "id": node.id,
                    "title": node.title,
                    "description": node.description,
                    "resources": node.resources,
                    "children": children,
                }

            mindmap_dict = {
                "root": mindmap_response.root,
                "nodes": [node_to_dict(node) for node in mindmap_response.nodes],
            }

            # Enrich with RAG resources
            enriched_mindmap = self._enrich_with_resources(mindmap_dict)

            # Convert back to MindmapNode objects
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

            enriched_nodes = [dict_to_node(node) for node in enriched_mindmap["nodes"]]

            # Create enriched response
            enriched_response = MindmapResponse(
                root=enriched_mindmap["root"],
                nodes=enriched_nodes,
                total_nodes=mindmap_response.total_nodes,  # Keep same count
                generated_at=mindmap_response.generated_at,  # Keep original timestamp
            )

            logger.info(f"Successfully enriched mindmap with resources")
            return enriched_response

        except Exception as e:
            logger.error(f"Error enriching mindmap with resources: {e}")
            raise

    def _enrich_with_resources(self, mindmap_data: Dict[str, Any]) -> Dict[str, Any]:
        """Enrich mindmap nodes with relevant resources from RAG and Tavily (intelligently)"""
        from app.utils.tavily_api import TavilyAPI

        tavily = TavilyAPI()

        def enrich_node(node_data: Dict[str, Any]) -> Dict[str, Any]:
            node_title = node_data.get("title", "")
            node_description = node_data.get("description", "")

            # 1. First check if we have sufficient resources in vector DB
            coverage = self.rag_service.check_resource_coverage(
                node_title=node_title,
                node_description=node_description,
                min_score=self.resource_config["min_relevance_score"],
                min_resources=self.resource_config["min_resources_needed"],
            )

            logger.info(
                f"Resource coverage for '{node_title}': {coverage['sufficient']} "
                f"({coverage['found_resources']} resources, avg relevance: {coverage['avg_relevance']:.2f})"
            )

            # 2. Only search Tavily if we don't have sufficient coverage OR if always_search is enabled
            should_search_external = (
                not coverage["sufficient"]
                or self.resource_config["always_search_external"]
            )

            if should_search_external:
                search_reason = (
                    "insufficient resources"
                    if not coverage["sufficient"]
                    else "always_search enabled"
                )
                logger.info(
                    f"Searching external sources for '{node_title}' ({search_reason})..."
                )
                query = f"{node_title} {node_description}"

                try:
                    tavily_results = tavily.search(
                        query, num_results=self.resource_config["max_external_results"]
                    )

                    # Prepare resources for batch add
                    new_resources = []
                    for result in tavily_results:
                        title = result.get("title", "")
                        url = result.get("url", "")
                        snippet = (
                            result.get("snippet") or result.get("description") or ""
                        )

                        if title and url:  # Only add if we have essential info
                            new_resources.append(
                                {
                                    "title": title,
                                    "description": snippet,
                                    "url": url,
                                    "category": node_data.get("category", "general"),
                                    "difficulty": node_data.get(
                                        "difficulty", "beginner"
                                    ),
                                    "tags": [
                                        "tavily",
                                        "external",
                                        node_title.lower().replace(" ", "_"),
                                    ],
                                }
                            )

                    # Batch add new resources (with duplicate checking)
                    if new_resources:
                        batch_result = self.rag_service.add_resources_batch(
                            new_resources, check_duplicates=True
                        )
                        logger.info(
                            f"Added {batch_result['added']} new resources, "
                            f"skipped {batch_result['skipped']} duplicates"
                        )

                except Exception as e:
                    logger.warning(f"External search failed for '{node_title}': {e}")
            else:
                logger.info(
                    f"Using existing resources for '{node_title}' "
                    f"(found {coverage['found_resources']} quality resources)"
                )

            # 3. Get the best resources from vector DB (after potential enrichment)
            resources = self.rag_service.get_resources_for_node(
                node_title=node_title,
                node_description=node_description,
            )

            # 4. Update node with resources
            enriched_node = node_data.copy()
            enriched_node["resources"] = resources

            # 5. Recursively enrich children
            if "children" in enriched_node and enriched_node["children"]:
                enriched_node["children"] = [
                    enrich_node(child) for child in enriched_node["children"]
                ]

            return enriched_node

        # Enrich all nodes
        enriched_nodes = [enrich_node(node) for node in mindmap_data["nodes"]]
        return {"root": mindmap_data["root"], "nodes": enriched_nodes}

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
