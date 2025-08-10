from sentence_transformers import SentenceTransformer
import numpy as np
import faiss

_model = None

def get_embedding_model(name: str = "all-MiniLM-L6-v2"):
    global _model
    if _model is None:
        _model = SentenceTransformer(name)
    return _model

def embed_texts(texts):
    model = get_embedding_model()
    emb = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
    return emb

def build_faiss_index(embeddings: "np.ndarray"):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index

def search_faiss(index, query, embeddings, top_k: int = 3):
    # encode query with same model
    model = get_embedding_model()
    q_emb = model.encode([query], show_progress_bar=False, convert_to_numpy=True)
    D, I = index.search(q_emb, top_k)
    return D[0], I[0]
