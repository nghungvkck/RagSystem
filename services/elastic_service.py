from core.elastic import elastic  # 
from services.embedding_service import embedding_service
from typing import List, Dict

class ElasticsearchService:
    def __init__(self):
        self.client = elastic  # 
    
    def save_chunks(self, filename: str, chunks: list, embeddings: List[List[float]] | None = None) -> int:
        # Tạo embedding nếu caller chưa cung cấp
        embeddings = embeddings if embeddings is not None else embedding_service.embed_batch(chunks)
        
        # Chuẩn bị dữ liệu
        data = []
        for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
            chunk_id = f"{filename}_{i}"
            data.append({
                "id": chunk_id,
                "filename": filename,
                "content": chunk,
                "vector": emb
            })
        
        return self.client.save(data)
    

    def search_hnsw(self, query_vector: List[float], k: int = 50, num_candidates: int = 100) -> List[Dict]:
        """Tìm kiếm HNSW với vector đầu vào"""
        response = self.client.es.search(
            index=self.client.index_name,
            knn={
                "field": "vector",
                "query_vector": query_vector,
                "k": k,
                "num_candidates": num_candidates,
            },
            size=k,
            source=["id", "filename", "content"]
        )
        
        return [
            {
                "score": hit['_score'],
                "id": hit['_source'].get('id'),
                "filename": hit['_source'].get('filename'),
                "content": hit['_source'].get('content')
            }
            for hit in response['hits']['hits']
        ]

# Khởi tạo service
elastic_service = ElasticsearchService()
