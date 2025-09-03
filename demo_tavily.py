from app.utils.tavily_api import TavilyAPI

if __name__ == "__main__":
    tavily = TavilyAPI()
    query = "Python programming for web development"
    results = tavily.search(query, num_results=5)
    for i, result in enumerate(results, 1):
        print(f"Result {i}:")
        print(f"  Title: {result.get('title')}")
        print(f"  URL: {result.get('url')}")
        print(f"  Snippet: {result.get('snippet')}")
        print()
