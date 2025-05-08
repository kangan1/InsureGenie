import streamlit as st
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle
import os

# --- Helper Functions ---

@st.cache_data
def extract_text_from_pdf(uploaded_file):
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def chunk_text(text, chunk_size=500, overlap=50):
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunk = text[i:i + chunk_size]
        chunks.append(chunk)
    return chunks

def create_embeddings(chunks):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(chunks)
    return model, embeddings

def build_faiss_index(embeddings):
    dimension = embeddings[0].shape[0]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))
    return index

# --- Streamlit UI ---

# st.title("📄 InsureGenie – Insurance Policy Q&A (RAG)")
st.markdown("""
# 🧠 InsureGenie – Your Smart Insurance Q&A Assistant  


Upload your policy PDF and get instant answers to your insurance questions.
""")



uploaded_file = st.file_uploader("Upload an insurance policy PDF", type="pdf")

if uploaded_file:
    st.info("Extracting text from PDF...")
    text = extract_text_from_pdf(uploaded_file)

    st.info("Chunking text...")
    chunks = chunk_text(text)

    st.info("Creating embeddings...")
    model, embeddings = create_embeddings(chunks)

    st.info("Building FAISS index...")
    index = build_faiss_index(embeddings)

    st.success("RAG system is ready. Ask your question!")

    query = st.text_input("🔍 Ask a question about the policy:")

    if query:
        query_embedding = model.encode([query])
        D, I = index.search(np.array(query_embedding), k=3)

        retrieved_chunks = [chunks[i] for i in I[0]]
        context = "\n---\n".join(retrieved_chunks)

        st.subheader("🔎 Retrieved Context")
        st.write(context)

        # Simulated "LLM" response by summarizing retrieved text
        # You can later replace this with actual LLM (OpenAI or HF)
        st.subheader("🧠 Answer (Simulated)")
        st.write("Based on the policy document, here's what was found:\n")
        st.write(context[:800] + "..." if len(context) > 800 else context)

