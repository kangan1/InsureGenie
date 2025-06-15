import streamlit as st
import fitz
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle
import os
from transformers import pipeline # Make sure this is imported if using Hugging Face QA pipeline


# --- Helper Functions ---

@st.cache_data
def extract_text_and_pages_from_pdf(uploaded_file):
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    pages_data = [] # List to store (text, page_number) for each chunk
    for page_num, page in enumerate(doc):
        text = page.get_text()
        # You might want to chunk page-by-page here, or chunk the whole text later
        # For simplicity, we'll return text with approximate page info for chunks
        pages_data.append((text, page_num + 1)) # page_num is 0-indexed
    return pages_data # Now returns a list of tuples: (page_text, page_number)

# Modified chunk_text to associate chunks with their original page numbers
def chunk_text_with_metadata(pages_data, chunk_size=500, overlap=50):
    chunks_with_metadata = []
    current_char_count = 0
    
    for page_text, page_num in pages_data:
        # If the page itself is smaller than chunk_size, treat it as one chunk
        if len(page_text) <= chunk_size:
            chunks_with_metadata.append({"text": page_text, "page": page_num})
        else:
            # If page is larger, chunk it with overlap, associating all chunks with this page
            for i in range(0, len(page_text), chunk_size - overlap):
                chunk = page_text[i:i + chunk_size]
                chunks_with_metadata.append({"text": chunk, "page": page_num})
                
    return chunks_with_metadata

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

def build_faiss_index(embeddings):
    embeddings = np.ascontiguousarray(embeddings).astype('float32')
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index

@st.cache_resource
def load_qa_llm():
    try:
        qa_pipeline = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad", tokenizer="distilbert-base-uncased", device=-1)
        st.success("loaded")
        return qa_pipeline
    except Exception as e:
        st.warning(f"Could not load local QA model: {e}. Falling back to basic context display. Make sure `transformers` and `torch` are installed.")
        return None

# --- Streamlit UI ---

st.markdown("""
# 🧠 InsureGenie – Your Smart Insurance Q&A Assistant  

Upload your policy PDF and get instant answers to your insurance questions.
""")

embedding_model = load_embedding_model()
qa_llm = load_qa_llm()

uploaded_file = st.file_uploader("Upload an insurance policy PDF", type="pdf")

if uploaded_file:
    st.info("Extracting text and page numbers from PDF...")
    pages_data = extract_text_and_pages_from_pdf(uploaded_file)

    st.info("Chunking text with metadata...")
    # Now chunks is a list of dictionaries: [{"text": "...", "page": N}, ...]
    chunks_with_metadata = chunk_text_with_metadata(pages_data)
    
    # Extract just the text for embedding
    chunks_text_only = [item["text"] for item in chunks_with_metadata]

    st.info("Creating embeddings...")
    embeddings = embedding_model.encode(chunks_text_only)

    st.info("Building FAISS index...")
    index = build_faiss_index(embeddings)

    st.success("RAG system is ready. Ask your question!")

    query = st.text_input("🔍 Ask a question about the policy:")

    if query:
        query_embedding = embedding_model.encode([query])
        query_embedding = np.ascontiguousarray(query_embedding).astype('float32')

        # Retrieve indices of the top k chunks
        D, I = index.search(query_embedding, k=3)

        retrieved_chunks_info = []
        for idx in I[0]:
            retrieved_chunks_info.append(chunks_with_metadata[idx]) # Get the full dict with text and page

        # Form the context for the LLM
        context = "\n---\n".join([item["text"] for item in retrieved_chunks_info])

        st.subheader("🔎 Retrieved Context (with Page Info)")
        # Display the retrieved chunks along with their page numbers
        for item in retrieved_chunks_info:
            st.write(f"**Page {item['page']}:**")
            st.write(item['text'])
            st.markdown("---") # Separator between chunks

        st.subheader("🧠 Answer")
        if qa_llm:
            try:
                answer_data = qa_llm(question=query, context=context)
                answer = answer_data['answer']
                score = answer_data['score']

                if score < 0.2:
                    st.write("I could not find a precise answer to your question in the provided document context.")
                else:
                    st.write(answer)
                    st.write(f"*(Confidence Score: {score:.2f})*")
                    
                    # --- Identify the page where the answer came from ---
                    # This part is a heuristic and might need refinement for complex cases
                    # Find which of the retrieved chunks the answer text is most likely from
                    best_page = "Unknown"
                    max_overlap = 0
                    
                    # Find the chunk with the highest text overlap (simple heuristic)
                    for chunk_info in retrieved_chunks_info:
                        chunk_text_lower = chunk_info['text'].lower()
                        answer_lower = answer.lower()
                        
                        # A simple way to check if the answer is within the chunk
                        if answer_lower in chunk_text_lower:
                            best_page = chunk_info['page']
                            break # Found it, so stop searching
                            
                    if best_page != "Unknown":
                        st.markdown(f"**_Found on Page: {best_page}_**")
                    else:
                        st.markdown(f"**_Context from Pages: {', '.join(sorted(list(set([item['page'] for item in retrieved_chunks_info]))))}_**")
                        st.info("The exact answer could not be localized to a single page, but the relevant information was drawn from the pages listed above.")

            except Exception as e:
                st.error(f"Error generating answer with local model: {e}. Please ensure question and context are valid for the model.")
                st.write("Could not generate a specific answer from the retrieved context.")
        else:
            st.warning("No generative AI model loaded. Displaying raw context as a fallback.")
            st.write(context[:800] + "..." if len(context) > 800 else context)