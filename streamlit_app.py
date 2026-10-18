import streamlit as st
import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

st.set_page_config(page_title="RAG Document Assistant", page_icon="📄", layout="wide")

@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

def chunk_text(text, size=800, overlap=150):
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start:start + size])
        start += size - overlap
    return [c.strip() for c in chunks if c.strip()]

def process_pdfs(files, model):
    records = []
    for f in files:
        reader = PdfReader(f)
        for page_num, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").replace("\n", " ")
            for chunk in chunk_text(text):
                records.append({"text": chunk, "source": f.name, "page": page_num})
    if not records:
        return [], None
    embeddings = model.encode([r["text"] for r in records], normalize_embeddings=True)
    return records, np.array(embeddings)

def search(query, records, embeddings, model, k=5):
    q = model.encode([query], normalize_embeddings=True)[0]
    scores = embeddings @ q
    top = np.argsort(scores)[::-1][:k]
    return [(float(scores[i]), records[i]) for i in top]

st.title("📄 RAG Document Assistant")
st.caption("Upload your PDFs, then ask any question about them.")

model = load_model()

with st.sidebar:
    st.header("Your documents")
    uploaded = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)
    if uploaded and st.button("Process documents"):
        with st.spinner("Reading and indexing PDFs..."):
            records, embeddings = process_pdfs(uploaded, model)
            st.session_state.records = records
            st.session_state.embeddings = embeddings
        st.success(f"Indexed {len(records)} chunks from {len(uploaded)} file(s).")

query = st.text_input("Your question", placeholder="Ask a question about your documents…")

if st.button("Search documents", type="primary"):
    if "records" not in st.session_state or not st.session_state.records:
        st.warning("Please upload and process at least one PDF first.")
    elif not query.strip():
        st.warning("Please type a question.")
    else:
        results = search(query, st.session_state.records, st.session_state.embeddings, model)
        if not results:
            st.info("No matching document chunks were found.")
        st.subheader("Top 5 results")
        for i, (score, r) in enumerate(results, start=1):
            with st.expander(f"#{i} · {r['source']} · page {r['page']} · score {score:.3f}", expanded=(i == 1)):
                st.write(r["text"])
