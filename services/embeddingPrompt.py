import ollama
from typing import List

class EmbeddingPrompt:
    def __init__(self, model: str = "all-minilm:l6-v2"):
        self.model = model
        print(f"Using Ollama embedding model: {self.model}")
    
    def embed(self, text: str) -> List[float]:
        """Tạo vector cho 1 câu"""
        response = ollama.embeddings(model=self.model, prompt=text)
        return response["embedding"]
    
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Tạo vector cho nhiều câu"""
        return [self.embed(text) for text in texts]
    
embedPrompt = EmbeddingPrompt()
