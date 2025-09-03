from app.utils.tavily_api import TavilyAPI
from app.services.rag_service import RAGService

# Example topic
TOPIC = "Python programming for web development"
CATEGORY = "programming"
DIFFICULTY = "beginner"

if __name__ == "__main__":
    tavily = TavilyAPI()
    rag = RAGService()
    results = tavily.search(TOPIC, num_results=5)
    for result in results:
        title = result.get("title")
        url = result.get("url")
        snippet = result.get("snippet") or result.get("description") or ""
        # Add to vector DB
        success = rag.add_resource(
            title=title,
            description=snippet,
            url=url,
            category=CATEGORY,
            difficulty=DIFFICULTY,
            tags=["python", "web", "tavily"],
        )
        print(f"Added: {title} - {url} | Success: {success}")
