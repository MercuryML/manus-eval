import requests
from typing import Optional, List
import os


class Rerank:
    url = "https://api.siliconflow.cn/v1/rerank"

    def __init__(self, token: Optional[str] = None):
        self.token = token or os.environ.get("SILICON_TOKEN", token)
        self.model = "Qwen/Qwen3-Reranker-4B"
        self.instruction = "Rerank the documents based on the query."
        self.headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
        }

    def query(self, query: str, documents: List[str]) -> int:
        payload = {
            "model": self.model,
            "instruction": self.instruction,
            "query": query,
            "documents": documents,
            "top_n": 1,
        }
        response = requests.post(self.url, json=payload, headers=self.headers)
        resp_json = response.json()
        index: int = resp_json["results"][0]["index"]
        return index


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()

    reranker = Rerank()
    query = "What is the capital of France?"
    documents = ["Paris is the capital of France.", "Berlin is the capital of Germany."]
    index = reranker.query(query, documents)
    print(index)
