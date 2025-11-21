import os
import re
import json
import fitz  # PyMuPDF
import streamlit as st
import google.generativeai as genai
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import CharacterTextSplitter

# ✅ Load environment variables
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=api_key)

# ✅ Load local embedding model
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# ✅ Process PDF
def process_pdf(uploaded_file):
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_text(text)
    documents = [Document(page_content=chunk) for chunk in chunks]
    return documents, text   # return chunks + full text

# ✅ Create Chroma vector store
def create_vector_store(documents):
    vectorstore = Chroma.from_documents(documents, embedding=embedding_model, persist_directory="chroma_store")
    vectorstore.persist()
    return vectorstore

# ✅ Gemini-powered RAG answer generator
def generate_rag_answer(question, retrieved_chunks):
    context = "\n\n".join([f"Chunk {i+1}: {c.page_content}" for i, c in enumerate(retrieved_chunks)])
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

# ✅ Gemini-powered summarizer
def generate_summary(full_text):
    prompt = f"""
    You are an AI summarizer. Summarize the following textbook content into a clear,
    concise summary highlighting the key points:

    Text:
    {full_text}

    Summary:
    """
    model = genai.GenerativeModel("gemini-2.5-flash")
    response = model.generate_content(prompt)
    return response.text

# ✅ Gemini-powered MCQ generator
def generate_mcqs(full_text, num_questions=5):
    prompt = f"""
    You are an exam question generator. Create {num_questions} multiple-choice questions (MCQs)
    based ONLY on the textbook content below. Each question must have 4 options and 1 correct answer.
    Return the output in strict JSON format as a list of objects with fields:
    - question (string)
    - options (list of 4 strings)
    - answer (integer index of correct option, 0–3)

    Text:
    {full_text}

    MCQs:
    """
    model = genai.GenerativeModel("gemini-2.5-flash")
    response = model.generate_content(prompt)
    return response.text

# ✅ Gemini-powered Fill-in-the-Blanks generator
def generate_fib(full_text, num_questions=5):
    prompt = f"""
    You are an exam question generator. Create {num_questions} fill-in-the-blanks questions
    based ONLY on the textbook content below. Each question must have:
    - question (string with a blank represented by '_____')
    - answer (string with the correct word/phrase)

    Return the output in strict JSON format as a list of objects with fields:
    - question
    - answer

    Text:
    {full_text}

    FIBs:
    """
    model = genai.GenerativeModel("gemini-2.5-flash")
    response = model.generate_content(prompt)
    return response.text

# ✅ Safe JSON parser
def safe_json_parse(raw_text):
    try:
        match = re.search(r"\[.*\]", raw_text, re.DOTALL)
        if match:
            clean = match.group(0)
            return json.loads(clean)
        else:
            return None
    except Exception as e:
        st.error(f"Parsing error: {e}")
        return None

# ✅ Streamlit UI
st.set_page_config(page_title="PDF Learning Assistant", layout="centered")
st.title("📄  AuraLearn : Personilized learning atmosphere")

uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

if uploaded_file:
    with st.spinner("Processing PDF..."):
        documents, full_text = process_pdf(uploaded_file)
        vectorstore = create_vector_store(documents)
        retriever = vectorstore.as_retriever()
    st.success("PDF processed!")

    # Save full_text in session_state
    st.session_state["full_text"] = full_text

    # ✅ Summary Button
    if st.button("Generate Summary"):
        with st.spinner("Summarizing PDF..."):
            summary = generate_summary(st.session_state["full_text"])
        st.markdown("### 📌 Summary of PDF")
        st.write(summary)

    # ✅ MCQ Section
    num_qs = st.slider("Number of MCQs", min_value=3, max_value=15, value=5)
    if st.button("Generate MCQs"):
        with st.spinner("Creating MCQs..."):
            mcqs_raw = generate_mcqs(st.session_state["full_text"], num_questions=num_qs)

        mcqs = safe_json_parse(mcqs_raw)
        if mcqs:
            st.session_state["mcqs"] = mcqs
            st.session_state["user_answers"] = [None] * len(mcqs)
        else:
            st.error("⚠️ Could not parse MCQs properly. Showing raw output instead:")
            st.write(mcqs_raw)

    if "mcqs" in st.session_state:
        mcqs = st.session_state["mcqs"]
        st.markdown("### 📝 MCQs from PDF")

        for i, q in enumerate(mcqs):
            st.markdown(f"**Q{i+1}. {q['question']}**")
            choice = st.radio(
                f"Select your answer for Q{i+1}",
                q["options"],
                key=f"mcq{i}"
            )
            st.session_state["user_answers"][i] = choice

        if st.button("Evaluate MCQs"):
            score = 0
            for i, q in enumerate(mcqs):
                correct_option = q["options"][q["answer"]]
                if st.session_state["user_answers"][i] == correct_option:
                    score += 1
                    st.success(f"Q{i+1}: Correct ✅")
                else:
                    st.error(f"Q{i+1}: Wrong ❌ (Correct: {correct_option})")
            st.markdown(f"### 🎯 MCQ Score: {score}/{len(mcqs)}")

    # ✅ Fill-in-the-Blanks Section
    num_fib = st.slider("Number of Fill-in-the-Blanks", min_value=3, max_value=15, value=5)
    if st.button("Generate Fill-in-the-Blanks"):
        with st.spinner("Creating Fill-in-the-Blanks..."):
            fib_raw = generate_fib(st.session_state["full_text"], num_questions=num_fib)

        fibs = safe_json_parse(fib_raw)
        if fibs:
            st.session_state["fibs"] = fibs
            st.session_state["fib_answers"] = [""] * len(fibs)
        else:
            st.error("⚠️ Could not parse FIBs properly. Showing raw output instead:")
            st.write(fib_raw)

    if "fibs" in st.session_state:
        fibs = st.session_state["fibs"]
        st.markdown("### ✏️ Fill-in-the-Blanks from PDF")

        for i, q in enumerate(fibs):
            st.markdown(f"**Q{i+1}. {q['question']}**")
            ans = st.text_input(f"Your answer for Q{i+1}", key=f"fib{i}")
            st.session_state["fib_answers"][i] = ans

        if st.button("Evaluate FIBs"):
            score = 0
            for i, q in enumerate(fibs):
                correct = q["answer"].strip().lower()
                user_ans = st.session_state["fib_answers"][i].strip().lower()
                if user_ans == correct:
                    score += 1
                    st.success(f"Q{i+1}: Correct ✅")
                else:
                    st.error(f"Q{i+1}: Wrong ❌ (Correct: {q['answer']})")
            st.markdown(f"### 🎯 FIB Score: {score}/{len(fibs)}")

    # ✅ Q&A Section
    query = st.text_input("Ask a question about the PDF:")
    if query:
        with st.spinner("Thinking..."):
            retrieved_docs = retriever.invoke(query)
            response = generate_rag_answer(query, retrieved_docs)
        st.markdown(f"**Answer:** {response}")