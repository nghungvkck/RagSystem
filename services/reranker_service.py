from sentence_transformers import CrossEncoder
from typing import List, Dict

class RerankerService:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model_name = model_name
        self.model = None

    def _get_model(self) -> CrossEncoder:
        if self.model is None:
            print(f"Loading reranker model: {self.model_name}")
            self.model = CrossEncoder(self.model_name)
            print("Reranker model ready.")
        return self.model
    
    def rerank(self, query: str, documents: List[Dict], top_k: int = 5) -> List[Dict]:
        if not documents:
            return []
        
        # Tạo pairs (query, document)
        pairs = [[query, doc['content']] for doc in documents]
        
        # Tính similarity scores
        scores = self._get_model().predict(pairs)
        
        # Gắn score vào documents
        for i, doc in enumerate(documents):
            doc['rerank_score'] = float(scores[i])
        
        # Sort và lấy top_k
        reranked = sorted(documents, key=lambda x: x['rerank_score'], reverse=True)
        return reranked[:top_k]
    
reranker_service = RerankerService()
