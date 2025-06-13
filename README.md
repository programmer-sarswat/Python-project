📄 PDF Chatbot using LangChain, Gemini & Pinecone

This is a Streamlit-based AI-powered chatbot that lets users interact with the content of any PDF. The chatbot reads your uploaded PDF, breaks it into chunks, stores them in a vector database (Pinecone), and allows you to ask context-aware questions using Google's Gemini (via LangChain).

🚀 Features
📥 Upload any PDF file.

📄 Extracts text from each page.

✂️ Splits large texts into manageable chunks with overlap.

🔍 Embeds text chunks using Gemini Embedding API.

🧠 Stores vectors in Pinecone for semantic search.

🤖 Uses LangChain + Gemini Chat to answer questions based on PDF content.

💬 Clean conversational UI using Streamlit chat interface.

🔁 "Clear Chat" and "Refresh" support.

🧰 Tech Stack
Frontend: Streamlit

Backend Logic: Python

AI Model: Google Gemini (via langchain_google_genai)

Vector Store: Pinecone

PDF Parsing: PyPDF2

Environment Management: python-dotenv

📦 Setup Instructions
Clone this repo:

bash
Copy
Edit
git clone https://github.com/your-repo/pdf-chatbot
cd pdf-chatbot
Install dependencies:

bash
Copy
Edit
pip install -r requirements.txt
Create .env file:

env
Copy
Edit
GOOGLE_API_KEY=your_google_genai_api_key
PINECONE_API_KEY=your_pinecone_api_key
Run the app:

bash
Copy
Edit
streamlit run app.py
🔎 How It Works
Upload a PDF via the UI.

Extract Text from all PDF pages.

Split Text into chunks with overlap for context.

Convert Chunks into embeddings via Gemini API.

Store Embeddings in Pinecone with unique identifiers.

Chat with PDF: Ask questions—AI fetches the most similar chunks and replies using Gemini.

🧪 Sample Use Cases
Reading and querying research papers

Understanding business reports

Chatting with eBooks, manuals, policies

Quick summary and content search

⚠️ Note
Ensure your API keys for Google GenAI and Pinecone are valid and active.

Only PDF format is supported.

Large PDFs may take a few seconds to process.


