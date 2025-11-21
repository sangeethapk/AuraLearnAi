import os
from dotenv import load_dotenv
import fitz  # PyMuPDF
import streamlit as st
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import CharacterTextSplitter

# Load environment variables from .env file
load_dotenv()

# ✅ Set your Gemini API key
#genai.configure(api_key="your Google_API_key")
api_key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=api_key)

# ✅ Load local embedding model
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# ✅ Load and process PDF
def process_pdf(uploaded_file):
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_text(text)
    documents = [Document(page_content=chunk) for chunk in chunks]
    return documents

# ✅ Create Chroma vector store
def create_vector_store(documents):
    vectorstore = Chroma.from_documents(documents, embedding=embedding_model, persist_directory="chroma_store")
    vectorstore.persist()
    return vectorstore

# ✅ Gemini-powered RAG answer generator
def generate_rag_answer(question, retrieved_chunks):
    context = "\n\n".join([f"Chunk {i+1}: {c}" for i, c in enumerate(retrieved_chunks)])

    prompt = f"""
    You are an AI tutor. Use ONLY the context below to answer the question.
    If the answer is not found, reply: 'Answer is not available in the textbook.'

    Question: {question}

    Context:
    {context}

    Answer:
    """

    model = genai.GenerativeModel("gemini-2.5-flash")
    response = model.generate_content(prompt)


    return response.text

# ✅ Streamlit UI
st.set_page_config(page_title="PDF Chatbot (Gemini + Local RAG)", layout="centered")
st.title("📄 Chat with your PDF (Gemini + Local Embeddings)")

uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

if uploaded_file:
    with st.spinner("Processing PDF..."):
        documents = process_pdf(uploaded_file)
        vectorstore = create_vector_store(documents)
        retriever = vectorstore.as_retriever()
    st.success("PDF processed! Ask your questions below.")

    query = st.text_input("Ask a question about the PDF:")
    if query:
        with st.spinner("Thinking..."):
            retrieved_docs = retriever.invoke(query)
            response = generate_rag_answer(query, retrieved_docs)
        st.markdown(f"**Answer:** {response}")