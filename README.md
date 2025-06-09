# Python-project
📘 PDF Chatbot
A simple and powerful Streamlit app that allows users to upload a PDF, extract its content, generate vector embeddings using Gemini AI, and chat with the document — powered entirely by Google's Generative AI APIs and FAISS for vector search.

🚀 Features
📤 Upload a PDF file (one at a time)

📄 Extract and split text into chunks

🔎 Create a vector database with Gemini Embeddings and FAISS

💬 Ask questions based only on the uploaded PDF

🧠 Uses Gemini's models/gemini-2.0-flash for contextual answers

🧹 Auto-cleans previous uploads (removes FAISS DB before new upload)

📦 Dependencies
Install all required libraries:

bash
Copy
Edit
pip install streamlit google-generativeai faiss-cpu pypdf langchain langchain-google-genai python-dotenv
🔑 Google API Key Setup
Go to: https://makersuite.google.com/app/apikey

Copy your API key.

Create a .env file in the project root:

env
Copy
Edit
GOOGLE_API_KEY=your_google_api_key_here
🧠 How It Works
Upload a PDF using Streamlit UI.

The PDF is read using PyPDF2, and the text is split into chunks.

Text chunks are embedded using GoogleGenerativeAIEmbeddings.

A FAISS vector index is built and saved.

On each query, the most relevant chunks are searched.

A Gemini-based LangChain QA chain is used to generate the answer.

▶️ How to Run
bash
Copy
Edit
streamlit run main.py
🗂 Project Structure
bash
Copy
Edit
📁 project-folder/
│
├── main.py              # Main Streamlit app
├── .env                 # Contains GOOGLE_API_KEY
├── faiss_index/         # Auto-generated folder for vector store
📌 Notes
Only one PDF is handled at a time. Previous vector DB is cleared on new upload.

You can replace FAISS with ChromaDB by modifying the create_vector_store() function.

All answers are generated based strictly on the PDF context, not external data.

✨ Example Usage
Upload: research-paper.pdf
Ask: "What is the main conclusion of this study?"
✅ Gemini returns a focused, context-specific answer.

🧹 Optional Cleanup Function
The app uses:

python
Copy
Edit
shutil.rmtree("faiss_index", onerror=remove_readonly)
...to delete read-only folders on fresh uploads.

🤝 License
Free to use for educational and development purposes.
Credit to LangChain and Google AI Studio for powerful APIs.

