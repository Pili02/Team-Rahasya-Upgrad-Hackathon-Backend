from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from enum import Enum


class TimeComplexity(str, Enum):
    """Time complexity levels for learning nodes"""

    BEGINNER = "Beginner"
    INTERMEDIATE = "Intermediate"
    ADVANCED = "Advanced"


class MindmapNode(BaseModel):
    """Represents a single node in the mindmap"""

    id: int = Field(..., description="Unique identifier for the node")
    title: str = Field(..., description="Title/name of the concept or step")
    description: str = Field(
        ..., description="Detailed explanation of what this node represents"
    )
    time_left: str = Field(
        ..., description="Estimated time to complete this node (e.g., '2-3 weeks')"
    )
    difficulty: TimeComplexity = Field(..., description="Difficulty level of this node")
    resources: List[str] = Field(
        default_factory=list, description="List of relevant resource URLs"
    )
    prerequisites: List[int] = Field(
        default_factory=list, description="List of prerequisite node IDs"
    )
    children: List["MindmapNode"] = Field(
        default_factory=list, description="Child nodes"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )


class MindmapRequest(BaseModel):
    """Request model for generating a mindmap"""

    description: str = Field(..., description="User's goal description", min_length=10)
    max_depth: Optional[int] = Field(
        default=3, description="Maximum depth of the mindmap", ge=1, le=10
    )
    focus_area: Optional[str] = Field(
        default=None, description="Specific area to focus on"
    )
    time_constraint: Optional[str] = Field(
        default=None, description="Time constraint (e.g., '6 months')"
    )


class MindmapResponse(BaseModel):
    """Response model for the generated mindmap"""

    root: str = Field(..., description="Root goal/topic of the mindmap")
    nodes: List[MindmapNode] = Field(
        ..., description="List of all nodes in the mindmap"
    )
    total_nodes: int = Field(..., description="Total number of nodes")
    estimated_total_time: str = Field(
        ..., description="Estimated total time to complete all nodes"
    )
    complexity_score: float = Field(..., description="Overall complexity score (0-1)")
    generated_at: str = Field(..., description="Timestamp when mindmap was generated")


class NodeExplanationRequest(BaseModel):
    """Request model for explaining a specific node"""

    node_id: int = Field(..., description="ID of the node to explain")
    context: Optional[str] = Field(
        default=None, description="Additional context for the explanation"
    )


class NodeExplanationResponse(BaseModel):
    """Response model for node explanation"""

    node_id: int = Field(..., description="ID of the explained node")
    title: str = Field(..., description="Title of the node")
    explanation: str = Field(
        ..., description="Detailed explanation of why this node exists"
    )
    importance: str = Field(..., description="Why this node is important")
    tips: List[str] = Field(..., description="Tips for approaching this node")


class LinearRoadmapResponse(BaseModel):
    """Response model for linearized roadmap"""

    steps: List[Dict[str, Any]] = Field(..., description="Linear list of steps")
    total_steps: int = Field(..., description="Total number of steps")
    estimated_total_time: str = Field(..., description="Estimated total time")


class ErrorResponse(BaseModel):
    """Error response model"""

    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(default=None, description="Additional error details")


# Update forward references
MindmapNode.model_rebuild()
