import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
import json
from app.models import MindmapNode, MindmapResponse, TimeComplexity
from app.services.llm_service import LLMService
from app.services.rag_service import RAGService

logger = logging.getLogger(__name__)


class MindmapService:
    """Core service for mindmap generation and processing"""

    def __init__(self):
        self.llm_service = LLMService()
        self.rag_service = RAGService()

    def generate_mindmap(
        self,
        description: str,
        max_depth: int = 3,
        focus_area: Optional[str] = None,
        time_constraint: Optional[str] = None,
    ) -> MindmapResponse:
        """Generate a complete mindmap with RAG enrichment"""

        try:
            logger.info(f"Generating mindmap for: {description}")

            # Generate initial mindmap using LLM
            raw_mindmap = self.llm_service.generate_mindmap(
                description=description,
                max_depth=max_depth,
                focus_area=focus_area,
                time_constraint=time_constraint,
            )

            # Enrich with RAG resources
            enriched_mindmap = self._enrich_with_resources(raw_mindmap)

            # Convert dict nodes to MindmapNode objects recursively
            def dict_to_node(node_dict):
                children = [
                    dict_to_node(child) for child in node_dict.get("children", [])
                ]
                # Sanitize prerequisites: only keep integers
                raw_prereqs = node_dict.get("prerequisites", [])
                if isinstance(raw_prereqs, list):
                    prerequisites = [p for p in raw_prereqs if isinstance(p, int)]
                else:
                    prerequisites = []
                return MindmapNode(
                    id=node_dict["id"],
                    title=node_dict["title"],
                    description=node_dict["description"],
                    time_left=node_dict["time_left"],
                    difficulty=node_dict["difficulty"],
                    resources=node_dict.get("resources", []),
                    prerequisites=prerequisites,
                    children=children,
                    metadata=node_dict.get("metadata", {}),
                )

            mindmap_nodes = [dict_to_node(node) for node in enriched_mindmap["nodes"]]

            # Calculate metrics
            total_nodes = self._count_total_nodes(mindmap_nodes)
            estimated_total_time = self._calculate_total_time(mindmap_nodes)
            complexity_score = self._calculate_complexity_score(mindmap_nodes)

            # Create response
            response = MindmapResponse(
                root=enriched_mindmap["root"],
                nodes=mindmap_nodes,
                total_nodes=total_nodes,
                estimated_total_time=estimated_total_time,
                complexity_score=complexity_score,
                generated_at=datetime.now().isoformat(),
            )

            logger.info(f"Generated mindmap with {total_nodes} nodes")
            return response

        except Exception as e:
            logger.error(f"Error generating mindmap: {e}")
            raise

    def _enrich_with_resources(self, mindmap_data: Dict[str, Any]) -> Dict[str, Any]:
        """Enrich mindmap nodes with relevant resources from RAG and Tavily"""
        from app.utils.tavily_api import TavilyAPI

        tavily = TavilyAPI()

        def enrich_node(node_data: Dict[str, Any]) -> Dict[str, Any]:
            # 1. Search Tavily for new resources
            query = f"{node_data.get('title', '')} {node_data.get('description', '')}"
            tavily_results = tavily.search(query, num_results=3)
            # 2. Add each Tavily result to vector DB
            for result in tavily_results:
                title = result.get("title")
                url = result.get("url")
                snippet = result.get("snippet") or result.get("description") or ""
                self.rag_service.add_resource(
                    title=title,
                    description=snippet,
                    url=url,
                    category=node_data.get("category", "general"),
                    difficulty=node_data.get("difficulty", "beginner"),
                    tags=["tavily", node_data.get("title", "").lower()],
                )
            # 3. Get resources from vector DB (RAG)
            resources = self.rag_service.get_resources_for_node(
                node_title=node_data.get("title", ""),
                node_description=node_data.get("description", ""),
                difficulty=node_data.get("difficulty", "beginner").lower(),
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

    def _calculate_total_time(self, nodes: List[MindmapNode]) -> str:
        """Calculate estimated total time for all nodes"""

        def parse_time_estimate(time_str: str) -> float:
            """Parse time estimates into weeks"""
            time_str = time_str.lower().strip()

            if "week" in time_str:
                if "-" in time_str:
                    # "2-3 weeks" -> average of 2.5
                    parts = time_str.split("-")
                    try:
                        start = float(parts[0].strip().split()[0])
                        end = float(parts[1].strip().split()[0])
                        return (start + end) / 2
                    except:
                        return 2.0
                else:
                    # "3 weeks" -> 3
                    try:
                        return float(time_str.split()[0])
                    except:
                        return 2.0
            elif "month" in time_str:
                # Convert months to weeks
                try:
                    months = float(time_str.split()[0])
                    return months * 4.33  # Average weeks per month
                except:
                    return 4.0
            elif "day" in time_str:
                # Convert days to weeks
                try:
                    days = float(time_str.split()[0])
                    return days / 7.0
                except:
                    return 1.0
            else:
                return 2.0  # Default fallback

        def calculate_recursive(node_list: List[MindmapNode]) -> float:
            total_weeks = 0.0
            for node in node_list:
                total_weeks += parse_time_estimate(node.time_left)
                if node.children:
                    total_weeks += calculate_recursive(node.children)
            return total_weeks

        total_weeks = calculate_recursive(nodes)

        # Format the result
        if total_weeks < 1:
            return f"{int(total_weeks * 7)} days"
        elif total_weeks < 4:
            return f"{total_weeks:.1f} weeks"
        else:
            months = total_weeks / 4.33
            return f"{months:.1f} months"

    def _calculate_complexity_score(self, nodes: List[MindmapNode]) -> float:
        """Calculate overall complexity score (0-1)"""

        def calculate_node_complexity(node: MindmapNode) -> float:
            # Base complexity based on difficulty
            difficulty_scores = {
                TimeComplexity.BEGINNER: 0.2,
                TimeComplexity.INTERMEDIATE: 0.5,
                TimeComplexity.ADVANCED: 0.8,
            }

            base_score = difficulty_scores.get(node.difficulty, 0.3)

            # Adjust based on time estimate
            time_str = node.time_left.lower()
            if "month" in time_str or "advanced" in time_str:
                base_score += 0.1
            elif "week" in time_str:
                base_score += 0.05

            # Adjust based on number of children
            if node.children:
                base_score += min(len(node.children) * 0.05, 0.2)

            return min(base_score, 1.0)

        def calculate_recursive(node_list: List[MindmapNode]) -> float:
            total_score = 0.0
            for node in node_list:
                total_score += calculate_node_complexity(node)
                if node.children:
                    total_score += calculate_recursive(node.children)
            return total_score

        total_score = calculate_recursive(nodes)
        total_nodes = self._count_total_nodes(nodes)

        # Normalize to 0-1 range
        if total_nodes > 0:
            return min(total_score / total_nodes, 1.0)
        else:
            return 0.5

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
                    "time_left": node.time_left,
                    "difficulty": node.difficulty,
                    "level": level,
                    "resources": node.resources,
                    "prerequisites": node.prerequisites,
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
            "estimated_total_time": mindmap.estimated_total_time,
        }

    def get_mindmap_statistics(self, mindmap: MindmapResponse) -> Dict[str, Any]:
        """Get comprehensive statistics about the mindmap"""

        def analyze_nodes(nodes: List[MindmapNode]) -> Dict[str, Any]:
            stats = {
                "total_nodes": 0,
                "by_difficulty": {"Beginner": 0, "Intermediate": 0, "Advanced": 0},
                "by_time": {"short": 0, "medium": 0, "long": 0},
                "max_depth": 0,
                "avg_children": 0,
            }

            def analyze_recursive(node_list: List[MindmapNode], depth: int = 0):
                stats["total_nodes"] += len(node_list)
                stats["max_depth"] = max(stats["max_depth"], depth)

                for node in node_list:
                    # Count by difficulty
                    stats["by_difficulty"][node.difficulty.value] += 1

                    # Count by time estimate
                    time_str = node.time_left.lower()
                    if "day" in time_str or ("week" in time_str and "1" in time_str):
                        stats["by_time"]["short"] += 1
                    elif "month" in time_str or "advanced" in time_str:
                        stats["by_time"]["long"] += 1
                    else:
                        stats["by_time"]["medium"] += 1

                    # Analyze children
                    if node.children:
                        analyze_recursive(node.children, depth + 1)

            analyze_recursive(nodes)

            # Calculate average children per node
            if stats["total_nodes"] > 0:
                total_children = sum(len(node.children) for node in nodes)
                stats["avg_children"] = total_children / stats["total_nodes"]

            return stats

        node_stats = analyze_nodes(mindmap.nodes)

        return {
            "overview": {
                "root": mindmap.root,
                "total_nodes": mindmap.total_nodes,
                "estimated_total_time": mindmap.estimated_total_time,
                "complexity_score": mindmap.complexity_score,
                "generated_at": mindmap.generated_at,
            },
            "node_statistics": node_stats,
        }
