import streamlit as st
import stat
import os
import hashlib
import numpy as np
from dotenv import load_dotenv 
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from pinecone import Pinecone
from langchain.schema import Document


load_dotenv()
# Configure Google Generative AI API key
api=os.getenv("GOOGLE_API_KEY")
pine_api = os.getenv("PINECONE_API_KEY")

if not api and not pine_api:
    print(" API key missing ....")
    st.error("Something went wrong, please try again later.")
    st.stop()
genai.configure(api_key=api)
pc = Pinecone(api_key=pine_api)
# Initialize Pinecone index
index_name = "pdf-chatbot-index"

if index_name in pc.list_indexes():
    print(f"Index '{index_name}' already exists.")
    index = pc.Index(index_name)

index = pc.Index(index_name)

#Helper Methods
def remove_readonly(func, path, excinfo):
    """
    Remove read-only attribute from a file or directory.
    """
    os.chmod(path, stat.S_IWRITE)
    func(path)

def clear_previous_index():
    """Clears the previous FAISS index if it exists."""
    index.delete(delete_all=True)

def extract_text_from_pdf(file):
    """Extracts text from a PDF file."""
    reader = PdfReader(file)
    return "".join(page.extract_text() or "" for page in reader.pages)



def split_text_chunks(text):
    """Splits the text into manageable chunks for processing.
    """
    chunks = RecursiveCharacterTextSplitter( chunk_size=15000, chunk_overlap=500)
    return chunks.split_text(text)
  
    
def load_vector_data(vector_data, file_name):
    """Loads vector data into Pinecone."""
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", api_key=api)
    file_name_hash = hashlib.sha256(file_name.encode()).hexdigest()
    
    progress = st.progress(0, "Processing the file, please wait...")
    total_chunks = len(vector_data)
    embedded_vector = []
    vector_store = []
    for i, chunk in enumerate(vector_data): 
        embedding = embeddings.embed_query(chunk)
        embedded_vector.append(embedding)
        
        vector_store = [
            {
                "id": f"{file_name_hash}_{i}",
                "values": embedded_vector[i],
                "metadata": {
                    "file_name": file_name,
                    "page_content": vector_data[i],
                    "page_number": i + 1
                }
            } 
      
        ]
        progress.progress(int(((i + 1) / total_chunks)*90), f"Processing chunk {i + 1} of {total_chunks}...")
    #print( len(embedding))
    index.upsert(
        vectors=vector_store,
        namespace=file_name_hash,
    )
    progress.progress(100, "File processed successfully!")
    return {"namespace": file_name_hash, "texts": vector_data, "embeddings": embedded_vector}

def get_conversation_chain():
    """Creates a conversation chain for question answering using Google Generative AI."""
    # Initialize the LLM with the desired model
    llm = ChatGoogleGenerativeAI(model="models/gemini-2.0-flash")
    prompt = PromptTemplate(
        template="You are a helpful and polite PDF chatbot. A user has uploaded a PDF document for reference.Your behavior rules: 1. If the user sends a greeting only (e.g., 'hi', 'hello', 'hey', 'good morning'): - Respond with a friendly greeting and invite the user to ask anything related to the PDF.Example: 'Hello! 😊 I'm here to help you with your PDF. Ask me anything related to its content!' 2. If the user sends a greeting *and* a query: - First, greet the user. - Then, check if the query is related to the content of the PDF. - If the query matches the PDF content: - Provide a helpful answer based on the relevant content. - Example: 'Hi there! Based on your query, here’s what the PDF says...' - If the query is unrelated to the PDF: - Politely inform the user that the content is not available. - Encourage them to ask something relevant to the PDF. - Example: 'Hi! I’m here to answer questions about the uploaded PDF. Unfortunately, I couldn't find anything related to your question in the document. Please ask something based on the PDF.' 3. If the user sends a query without greeting: - Follow the same rules as above to determine if the query is PDF-related or not. - If related: respond with the answer. - If unrelated: inform user that only PDF-based questions can be answered. Always be polite, concise, and relevant. Do not hallucinate information not found in the PDF. \n\nContext: {context}\n\nQuestion: {question}",
        input_variables=["context", "question"]
    )
    return load_qa_chain(llm, chain_type="stuff", prompt=prompt)

def cosine_similarity(a, b):
    """Calculates the cosine similarity between two vectors."""
    a = np.array(a)
    b = np.array(b)
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 0.0
    return np.clip(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)), -1.0, 1.0)
    

def user_query(query):
    """Handles user queries by searching the vector store and generating a response."""
    if not query:
        return "Please ask a question."
    vector_stored = st.session_state.vector_store
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", api_key=api)
    query_embedding = embeddings.embed_query(query)

    similarity = [
        (cosine_similarity(query_embedding, vec), text)
        for vec , text in zip(vector_stored["embeddings"], vector_stored["texts"])
    ]
    similarity.sort(reverse=True)
    
    context = [text for _,text in similarity[:3]]
    if not context:
        return "No relevant information found in the document."
    
    docs = [ Document(page_content=chunks) for chunks in context]
    chain = st.session_state.qa_chain
    response = chain.run(input_documents=docs, question=query)
    print("Response:", response)
    return response

# Streamlit UI 
st.title("PDF Chatbot")

if "file_uploaded" not in st.session_state:
    st.session_state.file_uploaded = False
    #clear_previous_index()


if not st.session_state.file_uploaded:

    st.subheader("Select and upload file-")
    file = st.file_uploader("", type=["pdf"])
    if st.button("Upload"):
        if file:
            #progress = st.progress(0,"File is uploading, please wait...")
            full_text = extract_text_from_pdf(file)
            #progress.progress( 25, "File uploading, please wait...")
            chunks = split_text_chunks(full_text)
            #progress.progress(50, "File uploading, please wait...")
            vector_store= load_vector_data(chunks , file.name)
            #progress.progress(100, "File uploading, please wait...")
          
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
       
   

   
