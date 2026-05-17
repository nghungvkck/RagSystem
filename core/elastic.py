from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
import os

class ElasticsearchClient:
    def __init__(self):
        es_host = os.getenv("ELASTICSEARCH_HOST", "http://localhost:9200")
        es_username = os.getenv("ELASTICSEARCH_USERNAME")
        es_password = os.getenv("ELASTICSEARCH_PASSWORD")
        client_kwargs = {
            "hosts": [es_host],
            "verify_certs": False,
        }
        if es_username and es_password:
            client_kwargs["basic_auth"] = (es_username, es_password)

        self.es = Elasticsearch(**client_kwargs)
        self.index_name = os.getenv("ELASTICSEARCH_INDEX", "vectors")

    def ensure_index(self, dims: int) -> None:
        if self.es.indices.exists(index=self.index_name):
            return

        self.es.indices.create(
            index=self.index_name,
            mappings={
                "properties": {
                    "id": {"type": "keyword"},
                    "filename": {"type": "keyword"},
                    "content": {"type": "text"},
                    "vector": {
                        "type": "dense_vector",
                        "dims": dims,
                        "index": True,
                        "similarity": "cosine",
                    },
                }
            },
        )
        
    def save(self, chunks_data):
        """Lưu vector vào database"""
        if not chunks_data:
            return 0

        dims = len(chunks_data[0]["vector"])
        self.ensure_index(dims)

        actions = [
            {"_index": self.index_name, "_id": data["id"], "_source": data}
            for data in chunks_data
        ]
        success, _ = bulk(self.es, actions)
        print(f"Saved {success} vectors")
        return success
    
elastic = ElasticsearchClient()
