"""
Configuration settings for AI MindMap Mentor application.
"""

import os
from typing import List, Optional
from dotenv import load_dotenv

load_dotenv()


class Settings:
    """Application settings and configuration."""

    # API Keys (Required)
    tavily_api_key: str = os.getenv("TAVILY_API_KEY", "")
    google_api_key: str = os.getenv("GOOGLE_API_KEY", "")

    # Application Settings
    debug: bool = os.getenv("DEBUG", "false").lower() == "true"
    port: int = int(os.getenv("PORT", "8000"))
    host: str = os.getenv("HOST", "0.0.0.0")
    environment: str = os.getenv("ENVIRONMENT", "development")

    # CORS Settings
    allowed_origins: List[str] = os.getenv("ALLOWED_ORIGINS", "*").split(",")

    # LLM Settings
    gemini_model: str = os.getenv("GEMINI_MODEL", "gemini-2.5-pro")
    max_tokens: int = int(os.getenv("MAX_TOKENS", "4096"))
    temperature: float = float(os.getenv("TEMPERATURE", "0.2"))

    # Mindmap Settings
    default_max_depth: int = int(os.getenv("DEFAULT_MAX_DEPTH", "3"))
    max_nodes_per_level: int = int(os.getenv("MAX_NODES_PER_LEVEL", "5"))

    # Performance Settings
    request_timeout: int = int(os.getenv("REQUEST_TIMEOUT", "300"))  # 5 minutes
    max_concurrent_requests: int = int(os.getenv("MAX_CONCURRENT_REQUESTS", "10"))

    # Data Settings
    data_dir: str = os.getenv("DATA_DIR", "/app/data")
    chroma_db_path: str = os.path.join(data_dir, "chroma_db")

    def validate(self) -> bool:
        """Validate required settings."""
        errors = []

        if not self.tavily_api_key:
            errors.append("TAVILY_API_KEY is required")

        if not self.google_api_key:
            errors.append("GOOGLE_API_KEY is required")

        if errors:
            raise ValueError(f"Configuration errors: {', '.join(errors)}")

        return True

    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.environment.lower() == "production"

    def get_cors_origins(self) -> List[str]:
        """Get CORS origins, handling wildcard for development."""
        if "*" in self.allowed_origins:
            return ["*"]
        return [origin.strip() for origin in self.allowed_origins if origin.strip()]


# Global settings instance
settings = Settings()

# Validate settings on import (except in test environment)
if not os.getenv("TESTING"):
    try:
        settings.validate()
    except ValueError as e:
        print(f"⚠️  Configuration warning: {e}")
        print("Some features may not work without proper API keys.")
