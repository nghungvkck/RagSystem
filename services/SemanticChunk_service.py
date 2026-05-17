import numpy as np 
from typing import List
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import re
from nltk.tokenize import sent_tokenize
import nltk

class SemanticChunkService:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", similarity_threshold: float = 0.4):
        self.model_name = model_name
        self.model = None
        self.similarity_threshold = similarity_threshold

    def _ensure_nltk_data(self) -> None:
        try:
            nltk.data.find("tokenizers/punkt")
        except LookupError:
            nltk.download("punkt", quiet=True)
        try:
            nltk.data.find("tokenizers/punkt_tab")
        except LookupError:
            nltk.download("punkt_tab", quiet=True)

    def _get_model(self) -> SentenceTransformer:
        if self.model is None:
            print(f"Loading semantic chunk model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
        return self.model

    def _split_sentences(self, text: str) -> List[str]:
        """Tách câu hỗ trợ cả Anh và Việt"""
        self._ensure_nltk_data()

        # Xử lý text trước
        text = text.replace('\n', ' ')
        text = re.sub(r'\s+', ' ', text)
        
        # Dùng NLTK cho tiếng Anh (hoạt động tốt)
        sentences = sent_tokenize(text, language='english')
        
        # Nếu kết quả không tốt (văn bản tiếng Việt), dùng regex
        if len(sentences) <= 2:  # Nếu chỉ có 1-2 câu
            # Regex cho cả Anh và Việt
            sentences = re.split(r'(?<=[.!?;:])\s+(?=[A-ZÁÀẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬĐÊỀỂỄỆÔỐỒỔỖỘƠỚỜỞỠỢƯỨỪỬỮỰa-z0-9])', text)
        
        # Làm sạch
        sentences = [s.strip() for s in sentences if s.strip()]
        
        # Debug
        print(f"[DEBUG] Tách được {len(sentences)} câu")
        for i, s in enumerate(sentences[:3]):
            print(f"  Câu {i+1}: {s[:100]}...")
        
        return sentences

    def _get_embeddings(self, sentences: List[str]):
        return self._get_model().encode(sentences)
    
    def _calculate_similarities(self, embeddings):
        similarities = []
        for i in range(len(embeddings) - 1):
            sim = cosine_similarity([embeddings[i]], [embeddings[i+1]])[0][0]
            similarities.append(sim)
        
        # Debug similarities
        if similarities:
            print(f"[DEBUG] Similarities range: {min(similarities):.3f} - {max(similarities):.3f}, mean: {np.mean(similarities):.3f}")
        
        return similarities

    def _merge_chunks(self, sentences: List[str], similarities: List[float]) -> List[str]:
        if not sentences:
            return []
        
        if len(sentences) == 1:
            return sentences
        
        # Tính threshold động nếu cần
        threshold = self.similarity_threshold
        if similarities and threshold == 0.5:  # Nếu dùng default
            threshold = max(0.2, np.mean(similarities) - 0.3)
            print(f"[DEBUG] Auto threshold: {threshold:.3f}")
        
        chunks = []
        current_chunk = [sentences[0]]
        
        for i, sim in enumerate(similarities):
            if sim >= threshold:  # Dùng threshold có thể là động
                current_chunk.append(sentences[i+1])
            else:
                chunk_text = " ".join(current_chunk)
                if chunk_text.strip():
                    chunks.append(chunk_text)
                current_chunk = [sentences[i+1]]
        
        # Thêm chunk cuối
        if current_chunk:
            chunk_text = " ".join(current_chunk)
            if chunk_text.strip():
                chunks.append(chunk_text)
        
        print(f"[DEBUG] Tạo được {len(chunks)} chunks")
        return chunks

    def chunk_text(self, text: str) -> List[str]:
        if not text: 
            return []
        
        sentences = self._split_sentences(text)
        if len(sentences) <= 1:
            return sentences
        
        embeddings = self._get_embeddings(sentences)
        similarities = self._calculate_similarities(embeddings)
        chunks = self._merge_chunks(sentences, similarities)
        
        return chunks

# Khởi tạo với threshold thấp hơn
semantic_chunk_service = SemanticChunkService(similarity_threshold=0.4)  
