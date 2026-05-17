from fastapi import APIRouter, UploadFile, File
from services.upload import upload_service
from services.loader_service import loader_service
from services.SemanticChunk_service import semantic_chunk_service
from services.embedding_service import embedding_service
from services.elastic_service import elastic_service
from services.query_variants_service import query_service
from services.embeddingPrompt import embedPrompt
from services.reranker_service import reranker_service
from services.generation_service import generation_service

router = APIRouter()

# trong router.py
@router.post("/upload")

async def upload(file: UploadFile = File(...)):
    # Indexing
    path = upload_service.save_file(file)                 # up file
    text = loader_service.load_file(path)                 # load file
    chunks = semantic_chunk_service.chunk_text(text)       # chia thành các chunk
    embeddings = embedding_service.embed_batch(chunks)     # chuyển chunk thành vector embedding
    saved_chunks = elastic_service.save_chunks(file.filename, chunks, embeddings)
    
    return {
        "message": "done",
        "filename": file.filename,           
        "num_chunks": len(chunks),
        "num_embeddings": len(embeddings),
        "saved_chunks": saved_chunks,  
        "chunks_preview": chunks[:2]
    }

# API nhận câu hỏi
@router.post("/generate-variants")
async def generate_variants(question: str, num_variants: int = 3):
    # Query Processing ( Retrival )
    variants = query_service.generate(question, num_variants)
    variant_embeddings = embedPrompt.embed_batch(variants)

    # các vector gần với vector câu hỏi
    all_results = []
    for vector in variant_embeddings:  
        results = elastic_service.search_hnsw(query_vector=vector, k=50)  
        all_results.extend(results)  

    # xóa phần trùng nhau 
    unique_results = {}
    for result in all_results:
        doc_id = result['id']
        if doc_id not in unique_results or result['score'] > unique_results[doc_id]['score']:
            unique_results[doc_id] = result
    
    candidates = list(unique_results.values())

    top_chunks = reranker_service.rerank(question, candidates, top_k=5)

    # Chatbot trả lời câu hỏi
    answer = generation_service.generate(question, top_chunks)
    
    return {
        "original_question": question,
        "num_variants": len(variants) - 1,  
        "variants": variants,
        "embeddings_dimension": len(variant_embeddings[0]) if variant_embeddings else 0,
        "results": all_results[:5],
        "top_chunks": top_chunks,
        "answer": answer
    }


# API Test
@router.post("/test-retrieval")
async def test_retrieval(question: str, k: int = 20):
    """API test retrieval quality - không qua generation, không giới hạn top_k nhỏ"""
    variants = query_service.generate(question, 3)
    variant_embeddings = embedPrompt.embed_batch(variants)
    
    all_results = []
    for vector in variant_embeddings:
        results = elastic_service.search_hnsw(query_vector=vector, k=k)  # dùng k từ request
        all_results.extend(results)
    
    # Xóa trùng
    unique_results = {}
    for result in all_results:
        doc_id = result['id']
        if doc_id not in unique_results or result['score'] > unique_results[doc_id]['score']:
            unique_results[doc_id] = result
    
    return {
        "question": question,
        "total_unique_chunks": len(unique_results),
        "chunks": list(unique_results.values())[:k],
        "scores": [r['score'] for r in list(unique_results.values())[:10]]
    }

