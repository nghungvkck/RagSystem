import ollama
from typing import List, Dict

class GenerationService:
    def __init__(self, model: str = "mistral:7b"):
        self.model = model
    
    def generate(self, question: str, top_chunks: List[Dict]) -> str:
        sorted_chunks = sorted(top_chunks, key=lambda x: x.get('rerank_score', x.get('score', 0)), reverse=True)
        context_parts = []
        total_len = 0
        max_len = 2000  # Giới hạn tổng độ dài
        
        for chunk in sorted_chunks:
            content = chunk['content'][:400]  # Mỗi chunk tối đa 400 ký tự
            if total_len + len(content) > max_len:
                break
            context_parts.append(content)
            total_len += len(content)
        
        context = "\n\n---\n\n".join(context_parts)
        prompt = f"""Dựa vào thông tin sau để trả lời câu hỏi:

THÔNG TIN:
{context}

CÂU HỎI: {question}

YÊU CẦU: 
- Trả lời ngắn gọn, đúng trọng tâm
- Nếu thiếu thông tin, hãy nói rõ

TRẢ LỜI:"""
        
        response = ollama.generate(model=self.model, prompt=prompt)
        return response["response"]

generation_service = GenerationService()