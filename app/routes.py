from fastapi import APIRouter, HTTPException, Depends, status, BackgroundTasks
from fastapi.responses import JSONResponse
from typing import Optional, Dict, Any
import logging

from app.models import (
    MindmapRequest,
    MindmapResponse,
    NodeExplanationRequest,
    NodeExplanationResponse,
    LinearRoadmapResponse,
    ErrorResponse,
)
from app.services.mindmap_service import MindmapService
from app.services.llm_service import LLMService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["mindmap"])

# Initialize services
mindmap_service = MindmapService()
llm_service = LLMService()

# In-memory storage for enriched mindmaps (in production, use a database)
enriched_mindmaps = {}


async def enrich_mindmap_background(initial_mindmap: MindmapResponse):
    """Background task to enrich mindmap with resources and store the result"""
    try:
        logger.info(
            f"Starting background enrichment for mindmap with {initial_mindmap.total_nodes} nodes"
        )

        # Enrich the mindmap with resources
        enriched_mindmap = mindmap_service.enrich_mindmap_with_resources(
            initial_mindmap
        )

        # Store the enriched mindmap using the generated_at timestamp as key
        enriched_mindmaps[initial_mindmap.generated_at] = enriched_mindmap

        logger.info(
            f"Background enrichment completed for mindmap {initial_mindmap.generated_at}"
        )

    except Exception as e:
        logger.error(f"Background enrichment failed: {e}")
        # Store error state
        enriched_mindmaps[initial_mindmap.generated_at] = {"error": str(e)}


@router.post("/generate_mindmap", response_model=MindmapResponse)
async def generate_mindmap(request: MindmapRequest, background_tasks: BackgroundTasks):
    """
    Generate a mindmap from a goal description with fast initial response and background enrichment.

    This endpoint creates a hierarchical, structured mindmap with:
    - Fast initial response with mindmap structure (10-20 seconds)
    - Background enrichment with RAG-powered resource recommendations
    - Complexity scoring and time calculations
    """
    try:
        logger.info(
            f"Received mindmap generation request: {request.description[:50]}..."
        )

        # Check if Ollama is running
        # if not llm_service.test_connection():
        #     raise HTTPException(
        #         status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        #         detail="Ollama service is not available. Please ensure 'ollama run llama3' is running.",
        #     )

        # Generate the initial mindmap structure (fast)
        initial_mindmap = mindmap_service.generate_initial_mindmap(
            description=request.description,
            max_depth=request.max_depth,
            time_constraint=request.time_constraint,
        )

        # Add resource enrichment to background tasks
        background_tasks.add_task(enrich_mindmap_background, initial_mindmap)

        logger.info(
            f"Successfully generated initial mindmap with {initial_mindmap.total_nodes} nodes. Resource enrichment running in background."
        )
        return initial_mindmap

    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error(f"Error generating mindmap: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate mindmap: {str(e)}",
        )


@router.get("/mindmap/enriched/{timestamp}", response_model=MindmapResponse)
async def get_enriched_mindmap(timestamp: str):
    """
    Get the enriched mindmap with resources after background processing is complete.

    Args:
        timestamp: The generated_at timestamp from the initial mindmap response
    """
    try:
        if timestamp not in enriched_mindmaps:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Enriched mindmap not found. It may still be processing or the timestamp is invalid.",
            )

        enriched_mindmap = enriched_mindmaps[timestamp]

        # Check if enrichment failed
        if isinstance(enriched_mindmap, dict) and "error" in enriched_mindmap:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Enrichment failed: {enriched_mindmap['error']}",
            )

        return enriched_mindmap

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving enriched mindmap: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve enriched mindmap: {str(e)}",
        )


@router.post("/explain_node", response_model=NodeExplanationResponse)
async def explain_node(request: NodeExplanationRequest):
    """
    Get an explanation for why a specific node exists in the mindmap.

    This endpoint provides:
    - Detailed explanation of the node's purpose
    - Why it's important in the learning path
    - Tips for approaching this learning step
    """
    try:
        logger.info(f"Received node explanation request for node ID: {request.node_id}")

        # For now, we'll need a mindmap context
        # In a real app, you might store mindmaps in a database
        # For demo purposes, we'll return a helpful error
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Node explanation requires a mindmap context. Please generate a mindmap first and then use the explanation endpoint with the generated mindmap data.",
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error explaining node: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to explain node: {str(e)}",
        )


@router.get("/linearize", response_model=LinearRoadmapResponse)
async def linearize_mindmap():
    """
    Convert a mindmap to a linear roadmap format.

    This endpoint flattens the hierarchical mindmap into a sequential list of steps.
    Note: Requires a previously generated mindmap.
    """
    try:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Linearization requires a mindmap context. Please generate a mindmap first.",
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error linearizing mindmap: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to linearize mindmap: {str(e)}",
        )


@router.get("/health")
async def health_check():
    """
    Health check endpoint to verify service status.

    Checks:
    - FastAPI service status
    - Ollama connection
    - RAG service status
    """
    try:
        health_status = {
            "status": "healthy",
            "service": "AI MindMap Mentor",
            "version": "1.0.0",
            "checks": {},
        }

        # Check Ollama connection
        try:
            ollama_healthy = llm_service.test_connection()
            health_status["checks"]["ollama"] = {
                "status": "healthy" if ollama_healthy else "unhealthy",
                "message": (
                    "Connected to Ollama"
                    if ollama_healthy
                    else "Cannot connect to Ollama"
                ),
            }
        except Exception as e:
            health_status["checks"]["ollama"] = {
                "status": "unhealthy",
                "message": f"Error: {str(e)}",
            }

        # Check RAG service
        try:
            # Simple check - try to get categories
            categories = mindmap_service.rag_service.get_all_categories()
            health_status["checks"]["rag_service"] = {
                "status": "healthy",
                "message": f"Available categories: {len(categories)}",
            }
        except Exception as e:
            health_status["checks"]["rag_service"] = {
                "status": "unhealthy",
                "message": f"Error: {str(e)}",
            }

        # Overall status
        all_healthy = all(
            check["status"] == "healthy" for check in health_status["checks"].values()
        )

        if not all_healthy:
            health_status["status"] = "degraded"

        return health_status

    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"status": "unhealthy", "error": str(e)},
        )


@router.get("/resources/categories")
async def get_resource_categories():
    """
    Get all available resource categories from the RAG service.

    Returns a list of categories that can be used to filter educational resources.
    """
    try:
        categories = mindmap_service.rag_service.get_all_categories()
        return {"categories": categories, "total": len(categories)}
    except Exception as e:
        logger.error(f"Error getting resource categories: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get resource categories: {str(e)}",
        )


@router.get("/resources/category/{category}")
async def get_resources_by_category(category: str, limit: int = 10):
    """
    Get resources filtered by a specific category.

    Args:
        category: The category to filter by
        limit: Maximum number of resources to return (default: 10)
    """
    try:
        resources = mindmap_service.rag_service.get_resources_by_category(
            category, limit
        )
        return {"category": category, "resources": resources, "total": len(resources)}
    except Exception as e:
        logger.error(f"Error getting resources for category {category}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get resources for category {category}: {str(e)}",
        )


@router.post("/resources/add")
async def add_resource(
    title: str, description: str, url: str, category: str, difficulty: str, tags: str
):
    """
    Add a new educational resource to the knowledge base.

    Args:
        title: Resource title
        description: Resource description
        url: Resource URL
        category: Resource category
        difficulty: Resource difficulty level
        tags: Comma-separated tags
    """
    try:
        # Parse tags
        tag_list = [tag.strip() for tag in tags.split(",") if tag.strip()]

        success = mindmap_service.rag_service.add_resource(
            title=title,
            description=description,
            url=url,
            category=category,
            difficulty=difficulty,
            tags=tag_list,
        )

        if success:
            return {
                "message": "Resource added successfully",
                "title": title,
                "category": category,
            }
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to add resource",
            )

    except Exception as e:
        logger.error(f"Error adding resource: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to add resource: {str(e)}",
        )


# Note: Exception handlers are moved to main.py since APIRouter doesn't support them
