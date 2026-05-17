from sentence_transformers import SentenceTransformer
from typing import List

class EmbeddingService:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None
        self.dimension = 384

    def _get_model(self) -> SentenceTransformer:
        if self.model is None:
            print(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            print(f"Embedding model ready. Vector dimension: {self.dimension}")
        return self.model

    def embed_text(self, text: str) -> List[float]:
        if not text or not text.strip():
            return [0.0] * self.dimension  
        
        embedding = self._get_model().encode(text)
        return embedding.tolist()
    
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        valid_texts = [t for t in texts if t and t.strip()]
        
        if not valid_texts:
            return []
        embeddings = self._get_model().encode(valid_texts, show_progress_bar=True)
        return [emb.tolist() for emb in embeddings]
    
    # trả về số chiều 
    def get_dimension(self) -> int:
        return self.dimension
    
embedding_service = EmbeddingService()
