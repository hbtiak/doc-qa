import streamlit as st
from src.ingestion import load_and_chunk_docs_from_folder, chunk_text
from src.embeddings import embed_texts, build_faiss_index, search_faiss
from src.qa import load_qa_model, answer_question_from_context
import os

st.set_page_config(page_title="Document QA / Semantic Search", layout="wide")
st.title("📄 Document QA & Semantic Search — Streamlit Demo")

st.sidebar.header("Upload Documents")
uploaded_files = st.sidebar.file_uploader("Upload PDF / DOCX / TXT files", accept_multiple_files=True)

if st.button("Load sample documents"):
    sample_folder = os.path.join(os.path.dirname(__file__), "data", "sample_docs")
    docs = load_and_chunk_docs_from_folder(sample_folder)
    st.session_state["docs"] = docs

if uploaded_files:
    with st.spinner("Processing uploaded files..."):
        # save uploaded files temporarily and process
        tmp_dir = os.path.join(os.path.dirname(__file__), "data", "uploaded")
        os.makedirs(tmp_dir, exist_ok=True)
        saved = []
        for f in uploaded_files:
            path = os.path.join(tmp_dir, f.name)
            with open(path, "wb") as out:
                out.write(f.getbuffer())
            saved.append(path)
        docs = load_and_chunk_docs_from_folder(tmp_dir)
        st.session_state["docs"] = docs

if "docs" not in st.session_state:
    st.info("Load sample documents (left) or upload files to begin.")
    st.stop()

st.success(f"Loaded {len(st.session_state['docs'])} text chunks from documents.")

# Build embeddings & FAISS index (lazy)
if "faiss_index" not in st.session_state:
    with st.spinner("Building embeddings and FAISS index (this will use CPU and may take time locally)..."):
        texts = [d["text"] for d in st.session_state["docs"]]
        embeddings = embed_texts(texts)
        index = build_faiss_index(embeddings)
        st.session_state["faiss_index"] = index
        st.session_state["embeddings"] = embeddings
        st.session_state["texts"] = texts

query = st.text_input("Ask a question about the loaded documents:")
top_k = st.slider("Top-k retrieved chunks", 1, 10, 3)

if st.button("Search / Answer") and query:
    with st.spinner("Retrieving relevant chunks..."):
        D, I = search_faiss(st.session_state["faiss_index"], query, st.session_state["embeddings"], top_k=top_k)
        retrieved = [st.session_state["texts"][i] for i in I]
    st.subheader("Top retrieved context snippets")
    for i, r in enumerate(retrieved, 1):
        st.markdown(f"**Chunk {i}:**")
        st.write(r)
    with st.spinner("Running QA model on retrieved context..."):
        qa_model, qa_tokenizer = load_qa_model()
        answer, best_context = answer_question_from_context(query, retrieved, qa_model, qa_tokenizer)
    st.subheader("Answer")
    st.write(answer)
    st.subheader("Source context (best)")
    st.write(best_context)
