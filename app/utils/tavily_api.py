import os
import requests

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")


class TavilyAPI:
    BASE_URL = "https://api.tavily.com/search"

    def __init__(self, api_key: str = None):
        self.api_key = api_key or TAVILY_API_KEY
        if not self.api_key:
            raise ValueError("Tavily API key not set. Set TAVILY_API_KEY env variable.")

    def search(self, query: str, num_results: int = 5) -> list:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        body = {"query": query, "num": num_results}
        response = requests.post(self.BASE_URL, headers=headers, json=body)
        response.raise_for_status()
        data = response.json()
        return data.get("results", [])
