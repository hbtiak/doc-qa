import os
from typing import List, Dict
import io

# Simple ingestion supporting TXT files. PDF / DOCX parsing placeholders are included.
def load_and_chunk_docs_from_folder(folder_path: str, chunk_size: int = 400, overlap: int = 50) -> List[Dict]:
    docs = []
    for fname in sorted(os.listdir(folder_path)):
        path = os.path.join(folder_path, fname)
        if os.path.isdir(path):
            continue
        text = ""
        if fname.lower().endswith(".txt"):
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
        else:
            # Placeholder: for PDF/DOCX support, implement parsing using pdfminer.six and python-docx
            try:
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    text = f.read()
            except Exception:
                text = ""
        if not text:
            continue
        chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
        for i, c in enumerate(chunks):
            docs.append({"source": fname, "chunk_id": i, "text": c})
    return docs

def chunk_text(text: str, chunk_size: int = 400, overlap: int = 50) -> List[str]:
    tokens = text.split()
    chunks = []
    i = 0
    while i < len(tokens):
        chunk = tokens[i:i+chunk_size]
        chunks.append(" ".join(chunk))
        i += chunk_size - overlap
    return chunks
