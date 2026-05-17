import ollama
from typing import List

class QueryVariantsService:
    def __init__(self):
        self.model = "mistral:7b"

    def generate(self, query: str, n :int = 3) -> List[str]:
        prompt = f"Viết {n} cách hỏi khác nhau cho câu: '{query}'. Mỗi cách 1 dòng:"
        response = response = ollama.generate(model=self.model, prompt=prompt)
        variants = [line.strip() for line in response["response"].split("\n") if line.strip()]
        return [query] + variants[:n]

query_service = QueryVariantsService()