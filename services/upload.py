import os
import shutil
from fastapi import UploadFile, File
from services.loader_service import loader_service

UPLOAD_DIR = os.getenv("UPLOAD_DIR", "files")
os.makedirs(UPLOAD_DIR, exist_ok=True)

class UploadService:
    def save_file(self, file: UploadFile):
        file_path = os.path.join(UPLOAD_DIR, file.filename)

        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        return file_path


upload_service = UploadService()
