import streamlit as st
import time 
import shutil
import stat
import os
import hashlib
from dotenv import load_dotenv 
import google.generativeai as genai
from PyPDF2 import PdfReader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate

load_dotenv()
# Configure Google Generative AI API key
api=os.getenv("GOOGLE_API_KEY")
if not api:
    st.error("Google API key missing ....")
    st.stop()
genai.configure(api_key=api)

#Helper Methods
def remove_readonly(func, path, excinfo):
    """
    Remove read-only attribute from a file or directory.
    """
    os.chmod(path, stat.S_IWRITE)
    func(path)

def clear_previous_index():
    """Clears the previous FAISS index if it exists."""
    path = "faiss_index"
    if os.path.exists(path):
        # If the directory exists, remove it
        shutil.rmtree(path, onerror=remove_readonly)

def extract_text_from_pdf(file):
    """Extracts text from a PDF file."""
    reader = PdfReader(file)
    return "".join(page.extract_text() or "" for page in reader.pages)

def hash_file(file):
    """Generates a hash for the uploaded file to ensure uniqueness."""
    file.seek(0)  # Reset file pointer to the beginning
    file_hash = hashlib.md5(file.read()).hexdigest()
    file.seek(0)
    return file_hash

def split_text_chunks(text):
    """Splits the text into manageable chunks for processing.
    """
    chunks = RecursiveCharacterTextSplitter( chunk_size=15000, chunk_overlap=500)
    return chunks.split_text(text)
  
def create_vector_store(text_chunks):
    """Creates a vector store from the text chunks using Google Generative AI embeddings."""
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", api_key=api)
    vector_store = FAISS.from_texts( text_chunks, embedding = embeddings)
    #vector_store.save_local(save_path)
    return vector_store

def get_conversation_chain():
    """Creates a conversation chain for question answering using Google Generative AI."""
    # Initialize the LLM with the desired model
    llm = ChatGoogleGenerativeAI(model="models/gemini-2.0-flash")
    prompt = PromptTemplate(
        template="You are a helpful PDF assistant. Answer the user's question using only the content provided from the PDF below. If the answer exists in the context, provide it clearly and also mention the page number(s) it appears on.If the answer is not in the provided context, respond with: I’m sorry, I couldn’t find relevant information in the PDF. Do not make up information. Do not use outside knowledge.\n\nContext: {context}\n\nQuestion: {question}",
        input_variables=["context", "question"]
    )
    return load_qa_chain(llm, chain_type="stuff", prompt=prompt)
    

def user_query(query):
    """Handles user queries by searching the vector store and generating a response."""
    if not query:
        return "Please ask a question."
    vector_stored = st.session_state.vector_store
    docs = vector_stored.similarity_search(query)
    context = "\n".join([doc.page_content for doc in docs])
    
    if not context:
        return "No relevant information found in the document."
    
    chain = st.session_state.qa_chain
    response = chain.run(input_documents=docs, question=query)
    print("Response:", response)
    return response

# Streamlit UI 
st.title("PDF Chatbot")

if "file_uploaded" not in st.session_state:
    st.session_state.file_uploaded = False
    clear_previous_index()


if not st.session_state.file_uploaded:

    st.subheader("Select and upload file-")
    file = st.file_uploader("", type=["pdf"])
    if st.button("Upload"):
        if file:
            progress = st.progress(0,"File is uploading, please wait...")
            full_text = extract_text_from_pdf(file)
            progress.progress( 25, "File uploading, please wait...")
            chunks = split_text_chunks(full_text)
            progress.progress(50, "File uploading, please wait...")
            vector_store= create_vector_store(chunks)
            progress.progress(100, "File uploading, please wait...")
            st.session_state.pdf_chunks = chunks
            st.session_state.vector_store = vector_store
            st.session_state.qa_chain = get_conversation_chain()
            st.session_state.file_uploaded = True
            st.rerun()
        else:
            st.warning("⚠️ Please upload a file.")


if st.session_state.file_uploaded :
    st.success("File uploaded successfully! Now you can interact with the PDF.")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    col1, col2 = st.columns([1,7], vertical_alignment="bottom")
    with col1:
        refresh = st.button("Refresh")
        if refresh:
            st.session_state.clear()
            st.rerun()
    with col2:
        if st.button("Clear Chat"):
            st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    
    user_input = st.chat_input("Ask a question about the PDF:")
   
    
    if user_input:                                                               
        with st.spinner("Processing your query..."):
            response = user_query(user_input)
        
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        st.session_state.messages.append({"role": "assistant", "content": response})
        with st.chat_message("assistant"):
            st.markdown(response)
       
   

   
