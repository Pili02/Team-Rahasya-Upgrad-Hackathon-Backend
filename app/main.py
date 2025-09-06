from fastapi import FastAPI, Request
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging
import time
from contextlib import asynccontextmanager

from app.routes import router
from app.services.llm_service import LLMService
from app.config import settings

from fastapi.responses import JSONResponse, FileResponse

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


load_dotenv()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    logger.info("🚀 Starting AI MindMap Mentor...")

    # Validate configuration
    try:
        settings.validate()
        logger.info("✅ Configuration validated successfully")
    except ValueError as e:
        logger.error(f"❌ Configuration error: {e}")
        if settings.is_production():
            raise  # Fail fast in production

    # Test services on startup
    try:
        # Test Gemini connection
        llm_service = LLMService()
        if llm_service.test_connection():
            logger.info("✅ Gemini connection successful")
        else:
            logger.warning("⚠️  Gemini not available - mindmap generation will fail")
    except Exception as e:
        logger.error(f"❌ Service initialization error: {e}")

    logger.info("🎯 AI MindMap Mentor is ready!")

    yield

    # Shutdown
    logger.info("🛑 Shutting down AI MindMap Mentor...")


# Create FastAPI app
app = FastAPI(
    title="AI MindMap Mentor",
    description="""
    🧠 **AI MindMap Mentor** - An intelligent backend system that converts vague goal descriptions into dynamic, hierarchical mindmaps.
    
    ## Features
    
    - **Mindmap Generation**: Builds tree structures instead of flat lists
    - **Resource Enrichment**: Real, verified educational resources via Tavily API
    - **Time Estimation**: Granular time estimates for each learning node
    - **Explainability**: Query any node to understand why it exists
    - **Interoperability**: Export as JSON, linear roadmap, or graph visualization
    
    ## Quick Start
    
    1. Use the `/generate_mindmap` endpoint with your learning goal
    2. Get a structured, hierarchical mindmap with resources and time estimates
    
    ## Example
    
    ```json
    {
      "description": "I want to learn AI in 6 months"
    }
    ```
    
    This will generate a mindmap showing the learning path from programming basics to advanced AI concepts.
    """,
    version="1.0.0",
    contact={
        "name": "AI MindMap Mentor Team",
        "url": "https://github.com/your-username/ai-mindmap-mentor",
    },
    license_info={"name": "MIT", "url": "https://opensource.org/licenses/MIT"},
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.get_cors_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Add request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()

    # Log request
    logger.info(f"📥 {request.method} {request.url.path}")

    # Process request
    response = await call_next(request)

    # Log response
    process_time = time.time() - start_time
    logger.info(
        f"📤 {request.method} {request.url.path} - {response.status_code} - {process_time:.3f}s"
    )

    return response


# Include API routes
app.include_router(router)


# Root endpoint
@app.get("/")
async def read_index():
    """Serves the frontend's index.html file"""
    return FileResponse("static/index.html")


# Health check endpoint (simple version)
@app.get("/health")
async def simple_health():
    """Simple health check endpoint"""
    return {"status": "healthy", "service": "AI MindMap Mentor"}


# Error handlers
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler"""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": "An unexpected error occurred",
            "type": type(exc).__name__,
        },
    )


if __name__ == "__main__":
    import uvicorn

    logger.info("🚀 Starting AI MindMap Mentor server...")
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level="info",
    )
