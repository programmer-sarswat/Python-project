
# 📘 PDF Chatbot :-

A simple and powerful **Streamlit app** that allows users to upload a PDF, extract its content, generate vector embeddings using **Generative AI APIs**, and chat with the document — powered by **FAISS** for vector search.

---

## 🚀 Features

- 📤 Upload a PDF file (one at a time)
- 📄 Extract and split text into chunks
- 🔎 Create a vector database with AI Embeddings and FAISS
- 💬 Ask questions based only on the uploaded PDF
- 🧠 Uses a language model to generate contextual answers
- 🧹 Auto-cleans previous uploads (removes FAISS DB before new upload)

---

## 📦 Dependencies

Install all required libraries:

```bash
pip install streamlit google-generativeai faiss-cpu pypdf langchain langchain-google-genai python-dotenv
```

---

## 🔑 API Key Setup

1. Go to: https://makersuite.google.com/app/apikey  
2. Copy your API key.
3. Create a `.env` file in the project root:

```env
GOOGLE_API_KEY=your_google_api_key_here
```

---

## 🧠 How It Works

1. **Upload a PDF** using Streamlit UI.
2. The PDF is read using `PyPDF2`, and the text is split into chunks.
3. Text chunks are embedded using AI embeddings.
4. A **FAISS** vector index is built and saved.
5. On each query, the most relevant chunks are searched.
6. A QA chain is used to generate the answer based only on PDF content.

---

## ▶️ How to Run

```bash
streamlit run main.py
```

---

## 🗂 Project Structure

```
📁 project-folder/
│
├── main.py              # Main Streamlit app
├── .env                 # Contains GOOGLE_API_KEY
├── faiss_index/         # Auto-generated folder for vector store
```

---

## 📌 Notes

- Only one PDF is handled at a time. Previous vector DB is cleared on new upload.
- You can replace FAISS with ChromaDB by modifying the `create_vector_store()` function.
- All answers are generated based **strictly on the PDF context**, not external data.

---

## ✨ Example Usage

> Upload: `research-paper.pdf`  
> Ask: *"What is the main conclusion of this study?"*  
> ✅ The chatbot returns a focused, context-specific answer.

---

## 🧹 Optional Cleanup Function

The app uses:

```python
shutil.rmtree("faiss_index", onerror=remove_readonly)
```

...to delete read-only folders on fresh uploads.

---

## 🤝 License

Free to use for educational and development purposes.  
Credit to [LangChain](https://www.langchain.com/) and [Google AI Studio](https://makersuite.google.com/) for powerful APIs.
