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

        # Use get_or_create_collection for a more robust approach
        self.collection = self.client.get_or_create_collection("educational_resources")

        if self.collection.count() == 0:
            logger.info("Knowledge base is empty. Initializing with default resources.")
            self._initialize_knowledge_base()
        else:
            logger.info(
                f"Knowledge base loaded with {self.collection.count()} resources."
            )

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

            # Create a copy of the resource metadata to modify
            metadata = resource.copy()

            # **FIX: Convert the 'tags' list to a single comma-separated string**
            if "tags" in metadata and isinstance(metadata["tags"], list):
                metadata["tags"] = ", ".join(metadata["tags"])

            metadatas.append(metadata)
            ids.append(f"resource_{i}")

        self.collection.add(documents=documents, metadatas=metadatas, ids=ids)

        logger.info(f"Initialized knowledge base with {len(resources)} resources")

    def search_resources(
        self,
        query: str,
        category: Optional[str] = None,
        difficulty: Optional[str] = None,
        limit: int = 5,
        min_score: Optional[float] = None,
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
            include = ["metadatas"]
            if min_score is not None:
                include.append("distances")

            results = self.collection.query(
                query_texts=[search_query],
                n_results=limit,
                where=where_clause if where_clause else None,
                include=include,
            )

            # Format results
            resources = []
            url_seen = set()
            if results["metadatas"] and results["metadatas"][0]:
                distances = (
                    results.get("distances", [[]])[0] if min_score is not None else []
                )
                for idx, metadata in enumerate(results["metadatas"][0]):
                    url = metadata["url"]
                    score_ok = True
                    relevance = None
                    if min_score is not None and distances:
                        # cosine distance -> similarity
                        relevance = max(0.0, 1.0 - float(distances[idx]))
                        score_ok = relevance >= float(min_score)
                    if not score_ok:
                        continue
                    if url in url_seen:
                        continue
                    url_seen.add(url)
                    resources.append(
                        {
                            "title": metadata["title"],
                            "description": metadata["description"],
                            "url": url,
                            "category": metadata["category"],
                            "difficulty": metadata["difficulty"],
                            "tags": metadata["tags"],
                            **(
                                {"relevance_score": relevance}
                                if relevance is not None
                                else {}
                            ),
                        }
                    )

            return resources

        except Exception as e:
            logger.error(f"Error searching resources: {e}")
            return []

    def get_resources_for_node(
        self, node_title: str, node_description: str
    ) -> List[str]:
        """Get relevant resources for a specific mindmap node"""

        try:
            # Create search query from node information
            search_query = f"{node_title} {node_description}"

            # Search for resources
            resources = self.search_resources(
                query=search_query, limit=5, min_score=0.65
            )

            # Extract URLs
            urls = []
            seen = set()
            for resource in resources:
                url = resource.get("url")
                if url and url not in seen:
                    seen.add(url)
                    urls.append(url)

            # If no specific resources found, return general ones
            if not urls:
                general_resources = self.search_resources(
                    query=node_title, limit=3, min_score=0.6
                )
                for resource in general_resources:
                    url = resource.get("url")
                    if url and url not in seen:
                        seen.add(url)
                        urls.append(url)

            return urls

        except Exception as e:
            logger.error(f"Error getting resources for node: {e}")
            return []

    def check_resource_coverage(
        self,
        node_title: str,
        node_description: str,
        min_score: float = 1,
        min_resources: int = 3,
    ) -> Dict[str, Any]:
        """Check if we have sufficient relevant resources for a node

        Args:
            node_title: The title of the node to check
            node_description: The description of the node
            min_score: Minimum relevance score threshold (0.0 to 1.0)
            min_resources: Minimum number of resources needed

        Returns:
            Dict with 'sufficient' boolean and resource details
        """
        try:
            # Create search query
            search_query = f"{node_title} {node_description}"

            # Search with more results to evaluate coverage
            results = self.collection.query(
                query_texts=[search_query],
                n_results=min_resources * 2,  # Get more to evaluate quality
                include=["metadatas", "distances"],
            )

            if not results["metadatas"] or not results["metadatas"][0]:
                return {
                    "sufficient": False,
                    "found_resources": 0,
                    "avg_relevance": 0.0,
                    "resources": [],
                }

            # Convert distances to similarity scores (lower distance = higher similarity)
            # ChromaDB uses cosine distance, so similarity = 1 - distance
            distances = results["distances"][0] if results["distances"] else []
            similarities = (
                [max(0.0, 1.0 - dist) for dist in distances] if distances else []
            )

            # Filter resources by minimum score
            high_quality_resources = []
            for i, (metadata, similarity) in enumerate(
                zip(results["metadatas"][0], similarities)
            ):
                if similarity >= min_score:
                    resource = {
                        "title": metadata["title"],
                        "description": metadata["description"],
                        "url": metadata["url"],
                        "category": metadata["category"],
                        "difficulty": metadata["difficulty"],
                        "tags": metadata["tags"],
                        "relevance_score": similarity,
                    }
                    high_quality_resources.append(resource)

            # Calculate average relevance of all found resources
            avg_relevance = (
                sum(similarities) / len(similarities) if similarities else 0.0
            )

            # Determine if coverage is sufficient
            sufficient = (
                len(high_quality_resources) >= min_resources
                and avg_relevance >= min_score
            )

            return {
                "sufficient": sufficient,
                "found_resources": len(high_quality_resources),
                "total_found": len(results["metadatas"][0]),
                "avg_relevance": avg_relevance,
                "min_relevance": min(similarities) if similarities else 0.0,
                "max_relevance": max(similarities) if similarities else 0.0,
                "resources": high_quality_resources[:min_resources],  # Return top N
            }

        except Exception as e:
            logger.error(f"Error checking resource coverage: {e}")
            return {
                "sufficient": False,
                "found_resources": 0,
                "avg_relevance": 0.0,
                "resources": [],
            }

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
            # Check if resource with same URL already exists
            if self._resource_exists(url):
                logger.info(f"Resource with URL {url} already exists, skipping")
                return True

            # Generate unique ID
            existing_ids = self.collection.get()["ids"]
            new_id = f"resource_{len(existing_ids)}"

            # **FIX: Convert tags list to a comma-separated string for ChromaDB**
            tags_str = ", ".join(tags)

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
                        "tags": tags_str,  # Use the converted string
                    }
                ],
                ids=[new_id],
            )

            logger.info(f"Added new resource: {title}")
            return True

        except Exception as e:
            logger.error(f"Error adding resource: {e}")
            return False

    def _resource_exists(self, url: str) -> bool:
        """Check if a resource with the given URL already exists"""
        try:
            # Get all resources and check URLs
            results = self.collection.get()
            if results["metadatas"]:
                for metadata in results["metadatas"]:
                    if metadata and metadata.get("url") == url:
                        return True
            return False
        except Exception as e:
            logger.error(f"Error checking if resource exists: {e}")
            return False

    def add_resources_batch(
        self, resources: List[Dict[str, Any]], check_duplicates: bool = True
    ) -> Dict[str, int]:
        """Add multiple resources in batch, with optional duplicate checking

        Args:
            resources: List of resource dictionaries with required keys
            check_duplicates: Whether to check for existing URLs before adding

        Returns:
            Dict with 'added' and 'skipped' counts
        """
        try:
            added_count = 0
            skipped_count = 0

            for resource in resources:
                # Validate required fields
                required_fields = [
                    "title",
                    "description",
                    "url",
                    "category",
                    "difficulty",
                    "tags",
                ]
                if not all(field in resource for field in required_fields):
                    logger.warning(
                        f"Skipping resource due to missing fields: {resource.get('title', 'Unknown')}"
                    )
                    skipped_count += 1
                    continue

                # Check for duplicates if requested
                if check_duplicates and self._resource_exists(resource["url"]):
                    logger.debug(f"Skipping duplicate resource: {resource['title']}")
                    skipped_count += 1
                    continue

                # Add the resource
                success = self.add_resource(
                    title=resource["title"],
                    description=resource["description"],
                    url=resource["url"],
                    category=resource["category"],
                    difficulty=resource["difficulty"],
                    tags=resource["tags"],
                )

                if success:
                    added_count += 1
                else:
                    skipped_count += 1

            logger.info(
                f"Batch add completed: {added_count} added, {skipped_count} skipped"
            )
            return {"added": added_count, "skipped": skipped_count}

        except Exception as e:
            logger.error(f"Error in batch add resources: {e}")
            return {"added": 0, "skipped": len(resources)}

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
