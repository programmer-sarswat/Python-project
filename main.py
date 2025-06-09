import streamlit as st
import time 
import shutil
import stat
import os
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
api=os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=api)


def remove_readonly(func, path, excinfo):
    """
    Remove read-only attribute from a file or directory.
    """
    os.chmod(path, stat.S_IWRITE)
    func(path)




if "file_uploaded" not in st.session_state:
    st.session_state.file_uploaded = False
    shutil.rmtree("faiss_index", onerror=remove_readonly)


def extract_text_from_pdf(file):
    """Extracts text from a PDF file."""
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

def split_text_chunks(text):
    """Splits the text into manageable chunks for processing.
    """
    text_splitter = RecursiveCharacterTextSplitter( chunk_size=10000, chunk_overlap=100)
    chunks = text_splitter.split_text(text)
    return chunks
  

def create_vector_store(text_chunks):
    """"Creates a vector store from the text chunks using Google Generative AI embeddings."""
   embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api)
   vector_store = FAISS.from_texts( text_chunks, embedding = embeddings)
   vector_store.save_local("faiss_index")
   return vector_store

def get_conversation_chain():
    """Creates a conversation chain for question answering using Google Generative AI."""
    # Initialize the LLM with the desired model
    llm = ChatGoogleGenerativeAI(model="models/gemini-2.0-flash")
    prompt = PromptTemplate(
        template="You are a helpful assistant. Answer the question based on the context provided.\n\nContext: {context}\n\nQuestion: {question}",
        input_variables=["context", "question"]
    )
    chain = load_qa_chain(llm, chain_type="stuff", prompt=prompt)
    return chain

def user_query(query):
    """Handles user queries by searching the vector store and generating a response."""
    if not query:
        return "Please ask a question."
    vector_stored = FAISS.load_local("faiss_index", GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api), allow_dangerous_deserialization=True)
    docs = vector_stored.similarity_search(query)
    context = "\n".join([doc.page_content for doc in docs])
    
    if not context:
        return "No relevant information found in the document."
    
    chain = get_conversation_chain()
    response = chain.run(input_documents=docs, question=query)
    
    print("Response:", response)
    st.write("Response:", response)

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
            chunks = split_text_chunks(full_text)
            vector_store= create_vector_store(chunks)
            st.session_state.pdf_chunks = chunks
            st.session_state.vector_store = vector_store
            time.sleep(2)
            st.rerun()
        
    elif upload_clicked and not file:
        st.warning("⚠️ Please upload a file.")



if st.session_state.file_uploaded :
    st.success("File uploaded successfully! Now you can interact with the PDF.")
    # Here you can add the code to process the uploaded PDF and create a chatbot interface
    # For example, you could use PyPDF2 or pdfminer to extract text from the PDF and then use OpenAI's API to create a chatbot.
    
    # Example placeholder for chatbot interaction
    user_input = st.chat_input("Ask a question about the PDF:")
    if user_input:
        with st.spinner("Processing your query..."):
            user_query(user_input)
            time.sleep(2)
    



































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


    
