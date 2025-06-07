import streamlit as st
import time 
import os
from dotenv import load_dotenv 
import google.generativeai as genai
from PyPDF2 import PdfReader
import numpy as np
import faiss
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
model = genai.GenerativeModel("gemini-2.0-flash")
#embedding_model = genai.EmbeddingModel(model_name="models/embedding-001")



if "file_uploaded" not in st.session_state:
    st.session_state.file_uploaded = False
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "pdf_chunks" not in st.session_state:
    st.session_state.pdf_chunks = []

def extract_text_from_pdf(file):
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

def split_text(text, chunk_size=500, overlap=50):
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunk = text[i:i+chunk_size]
        chunks.append(chunk)
    return chunks

def get_embedding(text):
    try:
        response = genai.embed_content(
            model="models/embedding-001",
            content=text,
            task_type="retrieval_document"
        )
        return np.array(response["embedding"], dtype=np.float32)
    except Exception as e:
        st.error(f"Embedding error: {e}")
        return None

def create_vector_store(chunks):
    embeddings = []
    for chunk in chunks:
        emb = get_embedding(chunk)
        if emb is not None:
            embeddings.append(emb)

    if not embeddings:
        st.error("Failed to embed any chunks.")
        return None, None

    index = faiss.IndexFlatL2(len(embeddings[0]))
    index.add(np.array(embeddings))
    return index, np.array(embeddings)

def search_similar_chunk(query, index, chunks, all_embeddings):
    query_emb = get_embedding(query)
    if query_emb is None:
        return None
    D, I = index.search(np.array([query_emb]), k=1)
    if I[0][0] < len(chunks):
        return chunks[I[0][0]]
    return None

st.title("PDF Chatbot")

if not st.session_state.file_uploaded:

    st.subheader("Select and upload file-")

    file = st.file_uploader("", type=["pdf"])
    upload_clicked = st.button("Upload File")

    if file and upload_clicked:
        st.success(f"You have uploaded file `{file.name}` successfully.")
        st.session_state.file_uploaded = True
        with st.spinner("Redirecting please wait..."):
            full_text = extract_text_from_pdf(file)
            chunks = split_text(full_text)
            index, all_embeddings = create_vector_store(chunks)
            if index is None or all_embeddings is None:
                st.error("Failed to create vector store. Please check the file and try again.")
                st.session_state.file_uploaded = False
                st.stop()
            st.session_state.pdf_chunks = chunks
            st.session_state.vector_store = index
            st.session_state.embeddings_array = all_embeddings
            time.sleep(2)
            st.rerun()
        
    elif upload_clicked and not file:
        st.warning("⚠️ Please upload a file.")



if st.session_state.file_uploaded and st.session_state.vector_store is not None:
    st.success("File uploaded successfully! Now you can interact with the PDF.")
    # Here you can add the code to process the uploaded PDF and create a chatbot interface
    # For example, you could use PyPDF2 or pdfminer to extract text from the PDF and then use OpenAI's API to create a chatbot.
    
    # Example placeholder for chatbot interaction
    user_input = st.chat_input("Ask a question about the PDF:")
    st.session_state.chat_history = []
    if user_input:
        context = search_similar_chunk(
            user_input,
            st.session_state.vector_store,
            st.session_state.pdf_chunks,
            st.session_state.embeddings_array
        )
        if context:
            prompt = f"Answer the question based only on this PDF context:\n\n{context}\n\nQuestion: {user_input}"
            try:
                response = model.generate_content(prompt)
                st.chat_message("user").write(user_input)
                st.chat_message("assistant").write(response.text)
            except Exception as e:
                st.error(f"Gemini error: {e}")
        else:
            st.warning("Could not find relevant content in the PDF.")




































# import streamlit as st
# import time
# import openai

# st.title("Programmer Sarswat")
# st.subheader("Select and upload file-")

# file = st.file_uploader("", type=["pdf", "xls", "xlsx"])
# a = st.button("Upload File")

# if file and a:
#     st.success(f"You have uploaded file `{file.name}` successfully.")
#     with st.spinner("Redirecting please wait..."):
#         time.sleep(2)
        
# elif a and not file:
#     st.warning("⚠️ Please upload a file.");


    
