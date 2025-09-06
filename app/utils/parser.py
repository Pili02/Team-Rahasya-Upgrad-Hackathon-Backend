import json
import logging
from typing import Dict, Any, Optional, List
import re

logger = logging.getLogger(__name__)


def extract_json_from_text(text: str) -> Optional[Dict[str, Any]]:
    """
    Extract JSON content from LLM response text.

    Args:
        text: Raw text response from LLM

    Returns:
        Parsed JSON dictionary or None if extraction fails
    """
    try:
        # Look for JSON content between curly braces
        start_idx = text.find("{")
        end_idx = text.rfind("}")

        if start_idx == -1 or end_idx == -1:
            logger.warning("No JSON braces found in text")
            return None

        # Extract the JSON string
        json_str = text[start_idx : end_idx + 1]

        # Try to parse the JSON
        parsed = json.loads(json_str)
        return parsed

    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON: {e}")
        logger.debug(f"Raw text: {text}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error extracting JSON: {e}")
        return None


def clean_mindmap_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Clean and validate mindmap data from LLM.

    Args:
        data: Raw mindmap data from LLM

    Returns:
        Cleaned and validated mindmap data
    """
    try:
        cleaned_data = {"root": data.get("root", "Learning Goal"), "nodes": []}

        # Clean nodes
        if "nodes" in data and isinstance(data["nodes"], list):
            cleaned_data["nodes"] = clean_nodes(data["nodes"])

        return cleaned_data

    except Exception as e:
        logger.error(f"Error cleaning mindmap data: {e}")
        # Return minimal valid structure
        return {"root": "Learning Goal", "nodes": []}


def clean_nodes(nodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Clean and validate individual nodes.

    Args:
        nodes: List of raw node data

    Returns:
        List of cleaned node data
    """
    cleaned_nodes = []

    for i, node in enumerate(nodes):
        if not isinstance(node, dict):
            continue

        cleaned_node = {
            "id": node.get("id", i + 1),
            "title": clean_string(node.get("title", f"Node {i + 1}")),
            "description": clean_string(node.get("description", "")),
            "time_left": clean_time_estimate(node.get("time_left", "1-2 weeks")),
            "difficulty": clean_difficulty(node.get("difficulty", "Beginner")),
            "resources": clean_resources(node.get("resources", [])),
            "children": [],
        }

        # Recursively clean children
        if "children" in node and isinstance(node["children"], list):
            cleaned_node["children"] = clean_nodes(node["children"])

        cleaned_nodes.append(cleaned_node)

    return cleaned_nodes


def clean_string(value: Any) -> str:
    """Clean string values"""
    if not value:
        return ""

    # Convert to string and clean
    cleaned = str(value).strip()

    # Remove excessive whitespace
    cleaned = re.sub(r"\s+", " ", cleaned)

    return cleaned


def clean_time_estimate(time_str: Any) -> str:
    """Clean and validate time estimates"""
    if not time_str:
        return "1-2 weeks"

    time_str = str(time_str).strip().lower()

    # Common time patterns
    time_patterns = [
        r"(\d+)\s*-\s*(\d+)\s*(week|month|day)s?",
        r"(\d+)\s*(week|month|day)s?",
        r"(beginner|intermediate|advanced)",
        r"(easy|medium|hard)",
    ]

    for pattern in time_patterns:
        if re.match(pattern, time_str):
            return time_str

    # Default fallback
    return "1-2 weeks"


def clean_difficulty(difficulty: Any) -> str:
    """Clean and validate difficulty levels"""
    if not difficulty:
        return "Beginner"

    difficulty_str = str(difficulty).strip().title()

    # Valid difficulty levels
    valid_difficulties = ["Beginner", "Intermediate", "Advanced"]

    # Try to match or default
    for valid in valid_difficulties:
        if valid.lower() in difficulty_str.lower():
            return valid

    return "Beginner"


def clean_resources(resources: Any) -> List[str]:
    """Clean and validate resource lists"""
    if not resources:
        return []

    if not isinstance(resources, list):
        return []

    cleaned_resources = []

    for resource in resources:
        if isinstance(resource, str) and resource.strip():
            # Basic URL validation
            if resource.startswith(("http://", "https://")):
                cleaned_resources.append(resource.strip())
            else:
                # Try to make it a valid URL
                if not resource.startswith("http"):
                    resource = f"https://{resource}"
                cleaned_resources.append(resource.strip())

    return cleaned_resources


def validate_mindmap_structure(data: Dict[str, Any]) -> bool:
    """
    Validate that mindmap data has the correct structure.

    Args:
        data: Mindmap data to validate

    Returns:
        True if valid, False otherwise
    """
    try:
        # Check required top-level keys
        required_keys = ["root", "nodes"]
        for key in required_keys:
            if key not in data:
                logger.error(f"Missing required key: {key}")
                return False

        # Check root is string
        if not isinstance(data["root"], str):
            logger.error("Root must be a string")
            return False

        # Check nodes is list
        if not isinstance(data["nodes"], list):
            logger.error("Nodes must be a list")
            return False

        # Validate each node
        for i, node in enumerate(data["nodes"]):
            if not validate_node_structure(node, i):
                return False

        return True

    except Exception as e:
        logger.error(f"Error validating mindmap structure: {e}")
        return False


def validate_node_structure(node: Dict[str, Any], index: int) -> bool:
    """
    Validate individual node structure.

    Args:
        node: Node data to validate
        index: Index of the node for error reporting

    Returns:
        True if valid, False otherwise
    """
    try:
        if not isinstance(node, dict):
            logger.error(f"Node {index} must be a dictionary")
            return False

        # Check required node keys
        required_node_keys = ["title", "time_left", "difficulty"]
        for key in required_node_keys:
            if key not in node:
                logger.error(f"Node {index} missing required key: {key}")
                return False

        # Check title is string
        if not isinstance(node["title"], str):
            logger.error(f"Node {index} title must be a string")
            return False

        # Check time_left is string
        if not isinstance(node["time_left"], str):
            logger.error(f"Node {index} time_left must be a string")
            return False

        # Check difficulty is string
        if not isinstance(node["difficulty"], str):
            logger.error(f"Node {index} difficulty must be a string")
            return False

        # Validate children recursively
        if "children" in node and isinstance(node["children"], list):
            for i, child in enumerate(node["children"]):
                if not validate_node_structure(child, f"{index}.{i}"):
                    return False

        return True

    except Exception as e:
        logger.error(f"Error validating node {index}: {e}")
        return False


def format_mindmap_for_display(mindmap_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Format mindmap data for better display and readability.

    Args:
        mindmap_data: Raw mindmap data

    Returns:
        Formatted mindmap data
    """
    try:
        formatted = mindmap_data.copy()

        # Add metadata
        formatted["metadata"] = {
            "total_nodes": count_nodes_recursive(formatted["nodes"]),
            "max_depth": calculate_max_depth(formatted["nodes"]),
            "formatted_at": "now",  # In real app, use actual timestamp
        }

        return formatted

    except Exception as e:
        logger.error(f"Error formatting mindmap: {e}")
        return mindmap_data


def count_nodes_recursive(nodes: List[Dict[str, Any]]) -> int:
    """Count total nodes recursively"""
    count = len(nodes)
    for node in nodes:
        if "children" in node and isinstance(node["children"], list):
            count += count_nodes_recursive(node["children"])
    return count


def calculate_max_depth(nodes: List[Dict[str, Any]], current_depth: int = 0) -> int:
    """Calculate maximum depth of the mindmap"""
    if not nodes:
        return current_depth

    max_depth = current_depth + 1

    for node in nodes:
        if "children" in node and isinstance(node["children"], list):
            child_depth = calculate_max_depth(node["children"], current_depth + 1)
            max_depth = max(max_depth, child_depth)

    return max_depth
