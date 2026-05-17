import os
from pypdf import PdfReader

# Class này dùng để dọc file
class LoaderService:
    def load_txt(self, file_path: str):
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
        
    def load_pdf(self, file_path: str):
        reader = PdfReader(file_path)
        text = ""

        for page in reader.pages:
            text += page.extract_text() or ""
        return text


    def load_file(self, file_path: str):
        ext = os.path.splitext(file_path)[1].lower()
        if ext == ".txt":
            return self.load_txt(file_path)

        elif ext == ".pdf":
            return self.load_pdf(file_path)

        else:
            raise ValueError(f"Unsupported file type: {ext}")
        
loader_service = LoaderService()
