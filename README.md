# Document QA / Semantic Search — Streamlit Demo

This repository is a minimal demo of a Document QA / Semantic Search app built with Streamlit.

## Features
- Upload PDF / DOCX / TXT files
- Chunk documents and compute embeddings (sentence-transformers)
- Store embeddings in FAISS index for semantic search
- Run extractive QA (Hugging Face Transformers) over retrieved contexts
- Deploy easily on Streamlit Cloud

## Quick start (local)
1. Create a Python 3.10+ environment
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run locally:
   ```bash
   streamlit run app.py
   ```

## Deployment to Streamlit Cloud
1. Push the repository to GitHub.
2. Go to https://share.streamlit.io and connect your repo.
3. Set the main file to `app.py` and deploy.

Note: Models (sentence-transformers and QA model) are downloaded on first run; this may take time.

