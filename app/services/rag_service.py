import logging
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings
import json
import os

logger = logging.getLogger(__name__)


class RAGService:
    """Service for RAG-powered resource recommendations"""

    def __init__(self, persist_directory: str = "./data/chroma_db"):
        self.persist_directory = persist_directory
        self.client = chromadb.PersistentClient(
            path=persist_directory, settings=Settings(anonymized_telemetry=False)
        )

        # Initialize or get the collection
        try:
            self.collection = self.client.get_collection("educational_resources")
        except:
            self.collection = self.client.create_collection("educational_resources")
            self._initialize_knowledge_base()

    def _initialize_knowledge_base(self):
        """Initialize the knowledge base with basic educational resources"""

        # Basic educational resources for common learning paths
        resources = [
            {
                "title": "Python Programming",
                "description": "Learn Python programming from basics to advanced concepts",
                "url": "https://docs.python.org/3/tutorial/",
                "category": "programming",
                "difficulty": "beginner",
                "tags": ["python", "programming", "basics"],
            },
            {
                "title": "Real Python Tutorials",
                "description": "Comprehensive Python tutorials and articles",
                "url": "https://realpython.com/",
                "category": "programming",
                "difficulty": "beginner",
                "tags": ["python", "tutorials", "examples"],
            },
            {
                "title": "Khan Academy Math",
                "description": "Free math courses from basic arithmetic to calculus",
                "url": "https://www.khanacademy.org/math",
                "category": "mathematics",
                "difficulty": "beginner",
                "tags": ["math", "algebra", "calculus", "free"],
            },
            {
                "title": "MIT OpenCourseWare",
                "description": "Free MIT course materials for various subjects",
                "url": "https://ocw.mit.edu/",
                "category": "education",
                "difficulty": "advanced",
                "tags": ["mit", "courses", "free", "academic"],
            },
            {
                "title": "Coursera",
                "description": "Online courses from top universities and companies",
                "url": "https://www.coursera.org/",
                "category": "education",
                "difficulty": "intermediate",
                "tags": ["courses", "online", "certificates"],
            },
            {
                "title": "edX",
                "description": "Online learning platform with courses from leading institutions",
                "url": "https://www.edx.org/",
                "category": "education",
                "difficulty": "intermediate",
                "tags": ["courses", "online", "universities"],
            },
            {
                "title": "GitHub Learning Lab",
                "description": "Learn Git and GitHub through interactive tutorials",
                "url": "https://lab.github.com/",
                "category": "programming",
                "difficulty": "beginner",
                "tags": ["git", "github", "version-control"],
            },
            {
                "title": "MDN Web Docs",
                "description": "Comprehensive web development documentation",
                "url": "https://developer.mozilla.org/",
                "category": "web-development",
                "difficulty": "beginner",
                "tags": ["web", "html", "css", "javascript"],
            },
            {
                "title": "Stack Overflow",
                "description": "Q&A platform for programmers and developers",
                "url": "https://stackoverflow.com/",
                "category": "programming",
                "difficulty": "intermediate",
                "tags": ["qa", "programming", "community"],
            },
            {
                "title": "YouTube Learning",
                "description": "Educational content on various subjects",
                "url": "https://www.youtube.com/",
                "category": "education",
                "difficulty": "beginner",
                "tags": ["video", "tutorials", "free"],
            },
        ]

        # Add resources to the collection
        documents = []
        metadatas = []
        ids = []

        for i, resource in enumerate(resources):
            documents.append(f"{resource['title']} {resource['description']}")
            metadatas.append(resource)
            ids.append(f"resource_{i}")

        self.collection.add(documents=documents, metadatas=metadatas, ids=ids)

        logger.info(f"Initialized knowledge base with {len(resources)} resources")

    def search_resources(
        self,
        query: str,
        category: Optional[str] = None,
        difficulty: Optional[str] = None,
        limit: int = 5,
    ) -> List[Dict[str, Any]]:
        """Search for relevant educational resources"""

        try:
            # Build search query
            search_query = query

            # Add filters if provided
            where_clause = {}
            if category:
                where_clause["category"] = category
            if difficulty:
                where_clause["difficulty"] = difficulty

            # Perform search
            results = self.collection.query(
                query_texts=[search_query],
                n_results=limit,
                where=where_clause if where_clause else None,
            )

            # Format results
            resources = []
            if results["metadatas"] and results["metadatas"][0]:
                for metadata in results["metadatas"][0]:
                    resources.append(
                        {
                            "title": metadata["title"],
                            "description": metadata["description"],
                            "url": metadata["url"],
                            "category": metadata["category"],
                            "difficulty": metadata["difficulty"],
                            "tags": metadata["tags"],
                        }
                    )

            return resources

        except Exception as e:
            logger.error(f"Error searching resources: {e}")
            return []

    def get_resources_for_node(
        self, node_title: str, node_description: str, difficulty: str = "beginner"
    ) -> List[str]:
        """Get relevant resources for a specific mindmap node"""

        try:
            # Create search query from node information
            search_query = f"{node_title} {node_description}"

            # Search for resources
            resources = self.search_resources(
                query=search_query, difficulty=difficulty.lower(), limit=3
            )

            # Extract URLs
            urls = [resource["url"] for resource in resources]

            # If no specific resources found, return general ones
            if not urls:
                general_resources = self.search_resources(query=node_title, limit=2)
                urls = [resource["url"] for resource in general_resources]

            return urls

        except Exception as e:
            logger.error(f"Error getting resources for node: {e}")
            return []

    def add_resource(
        self,
        title: str,
        description: str,
        url: str,
        category: str,
        difficulty: str,
        tags: List[str],
    ):
        """Add a new resource to the knowledge base"""

        try:
            # Generate unique ID
            existing_ids = self.collection.get()["ids"]
            new_id = f"resource_{len(existing_ids)}"

            # Add to collection
            self.collection.add(
                documents=[f"{title} {description}"],
                metadatas=[
                    {
                        "title": title,
                        "description": description,
                        "url": url,
                        "category": category,
                        "difficulty": difficulty,
                        "tags": tags,
                    }
                ],
                ids=[new_id],
            )

            logger.info(f"Added new resource: {title}")
            return True

        except Exception as e:
            logger.error(f"Error adding resource: {e}")
            return False

    def get_all_categories(self) -> List[str]:
        """Get all available resource categories"""

        try:
            results = self.collection.get()
            categories = set()

            if results["metadatas"]:
                for metadata in results["metadatas"]:
                    if metadata and "category" in metadata:
                        categories.add(metadata["category"])

            return list(categories)

        except Exception as e:
            logger.error(f"Error getting categories: {e}")
            return []

    def get_resources_by_category(
        self, category: str, limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Get resources filtered by category"""

        try:
            results = self.collection.query(
                query_texts=[""], n_results=limit, where={"category": category}
            )

            resources = []
            if results["metadatas"] and results["metadatas"][0]:
                for metadata in results["metadatas"][0]:
                    resources.append(
                        {
                            "title": metadata["title"],
                            "description": metadata["description"],
                            "url": metadata["url"],
                            "category": metadata["category"],
                            "difficulty": metadata["difficulty"],
                            "tags": metadata["tags"],
                        }
                    )

            return resources

        except Exception as e:
            logger.error(f"Error getting resources by category: {e}")
            return []
