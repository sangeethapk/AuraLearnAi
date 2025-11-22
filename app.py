import os
import re
import json
import fitz # PyMuPDF
import streamlit as st
import google.generativeai as genai
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import CharacterTextSplitter
from streamlit_lottie import st_lottie
import requests

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
  # Chunking is essential here for RAG performance
  splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
  chunks = splitter.split_text(text)
  documents = [Document(page_content=chunk) for chunk in chunks]
  return documents, text  # return chunks + full text

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

# ✅ Gemini-powered summarizer (RAG-enhanced)
def generate_summary(retriever):
    # Retrieve all chunks by querying the retriever with a very general topic.
    retrieved_chunks = retriever.invoke("summarize the entire document")
    context = "\n\n".join([c.page_content for c in retrieved_chunks])

    prompt = f"""
    You are an AI summarizer. Summarize the following textbook content into a clear,
    concise summary highlighting the key points.

    Text Context:
    {context}

    Summary:
    """
    model = genai.GenerativeModel("gemini-2.5-flash")
    response = model.generate_content(prompt)
    return response.text

# ✅ Gemini-powered MCQ generator (RAG-enhanced)
def generate_mcqs(retriever, num_questions=5):
    # Retrieve all chunks for comprehensive question generation.
    retrieved_chunks = retriever.invoke("generate questions from the entire document")
    context = "\n\n".join([c.page_content for c in retrieved_chunks])
    
    prompt = f"""
    You are an exam question generator. Create {num_questions} multiple-choice questions (MCQs)
    based ONLY on the textbook content below. Each question must have 4 options and 1 correct answer.
    Return the output in strict JSON format as a list of objects with fields:
    - question (string)
    - options (list of 4 strings)
    - answer (integer index of correct option, 0–3)

    Text Context:
    {context}

    MCQs:
    """
    model = genai.GenerativeModel("gemini-2.5-flash")
    response = model.generate_content(prompt)
    return response.text

# ✅ Gemini-powered Fill-in-the-Blanks generator (RAG-enhanced)
def generate_fib(retriever, num_questions=5):
    # Retrieve all chunks for comprehensive question generation.
    retrieved_chunks = retriever.invoke("generate fill-in-the-blanks from the entire document")
    context = "\n\n".join([c.page_content for c in retrieved_chunks])
    
    prompt = f"""
    You are an exam question generator. Create {num_questions} fill-in-the-blanks questions
    based ONLY on the textbook content below. Each question must have:
    - question (string with a blank represented by '_____')
    - answer (string with the correct word/phrase)

    Return the output in strict JSON format as a list of objects with fields:
    - question
    - answer

    Text Context:
    {context}

    FIBs:
    """
    model = genai.GenerativeModel("gemini-2.5-flash")
    response = model.generate_content(prompt)
    return response.text


# ✅ Utility: Load Lottie animation
def load_lottieurl(url):
  r = requests.get(url)
  if r.status_code != 200:
    return None
  return r.json()

# Example teacher animation
lottie_teacher = load_lottieurl("https://assets9.lottiefiles.com/packages/lf20_4kx2q32n.json")

# ✅ Safe JSON parser
def safe_json_parse(raw_text):
  try:
    # Attempt to find the JSON array structure
    match = re.search(r"\[.*\]", raw_text, re.DOTALL)
    if match:
      clean = match.group(0)
      return json.loads(clean)
    else:
      # Try simple load if no match is found (sometimes models return clean JSON)
      return json.loads(raw_text)
  except Exception as e:
    st.error(f"Parsing error: {e}")
    return None

# -------------------- Streamlit UI --------------------
st.set_page_config(page_title="PDF Learning Assistant", layout="wide")
st.title("📄  AuraLearn: Personalized Learning Atmosphere")

# --- PDF Uploader and Setup ---
uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

if uploaded_file:
  # Check if the file has changed or if retriever is not set
  if "retriever" not in st.session_state or st.session_state.get("file_name") != uploaded_file.name:
    with st.spinner(f"Processing '{uploaded_file.name}'..."):
      documents, full_text = process_pdf(uploaded_file)
      vectorstore = create_vector_store(documents)
      # Store retriever and metadata in session state
      st.session_state["retriever"] = vectorstore.as_retriever(search_kwargs={"k": len(documents)}) 
      st.session_state["full_text"] = full_text
      st.session_state["file_name"] = uploaded_file.name
    
    st.success(f"PDF '{uploaded_file.name}' processed and ready!")

  # --- Main Tab Interface ---
  tab_summary, tab_qa, tab_mcq, tab_fib = st.tabs([
    "📚 Summary", 
    "💬 Q&A Tutor", 
    "📝 MCQs", 
    "✏️ Fill-in-the-Blanks"
  ])
    
  retriever = st.session_state["retriever"]

    # -------------------- 1. Summary Tab --------------------
  with tab_summary:
    st.header("Generate Document Summary")
    st.markdown("Click below to get a concise summary of the uploaded document.")
        
    if st.button("Generate Comprehensive Summary", type='primary', key="summary_btn"):
      with st.spinner("Summarizing PDF..."):
        summary = generate_summary(retriever) 
      st.markdown("### 📌 Summary of PDF")
      st.info(summary)
            
        # Display existing summary if available
    elif "summary" in st.session_state and st.session_state.get("summary_file_name") == uploaded_file.name:
      st.markdown("### 📌 Last Generated Summary")
      st.info(st.session_state["summary"])

    # -------------------- 2. Q&A Tutor Tab --------------------
  with tab_qa:
    st.header("Ask the Tutor")
    st.markdown("Ask specific questions about the PDF content. The AI will use the document as its knowledge source.")
        
    query = st.text_input("Your question:", key="qa_query")
    if query:
      with st.spinner("Thinking..."):
        retrieved_docs = retriever.invoke(query)
        response = generate_rag_answer(query, retrieved_docs)

      st.markdown("### Tutor's Answer")
      col_lottie, col_text = st.columns([1, 4])
      with col_lottie:
        st_lottie(lottie_teacher, height=120, key="teacher_chat")
      with col_text:
        st.write(response)

    # -------------------- 3. MCQ Tab --------------------
  with tab_mcq:
    st.header("Multiple Choice Questions")

    col1, col2 = st.columns([3, 1])
    num_qs = col1.slider("Number of MCQs to generate", min_value=3, max_value=15, value=5, key="mcq_slider")
        
    if col2.button("Generate MCQs", type='primary', key="generate_mcq_btn"):
      with st.spinner(f"Creating {num_qs} MCQs..."):
        mcqs_raw = generate_mcqs(retriever, num_questions=num_qs)

      mcqs = safe_json_parse(mcqs_raw)
      if mcqs:
        st.session_state["mcqs"] = mcqs
        st.session_state["user_answers"] = [None] * len(mcqs)
      else:
        st.error("⚠️ Could not parse MCQs properly. Check console for raw output.")
        print(mcqs_raw) # Print raw output for debugging

    if "mcqs" in st.session_state:
      mcqs = st.session_state["mcqs"]
      st.markdown("### 📝 Test Yourself!")

      for i, q in enumerate(mcqs):
        st.markdown(f"**Q{i+1}. {q['question']}**")
        choice = st.radio(
          f"Select your answer for Q{i+1}",
          q["options"],
          key=f"mcq_q{i}"
        )
        try:
          st.session_state["user_answers"][i] = choice
        except IndexError:
          st.session_state["user_answers"].append(choice)


      if st.button("Evaluate MCQs", type='secondary', key="evaluate_mcq_btn"):
        score = 0
        for i, q in enumerate(mcqs):
          correct_option = q["options"][q["answer"]]
          if st.session_state["user_answers"][i] == correct_option:
            score += 1
            st.success(f"Q{i+1}: Correct ✅")
          else:
            st.error(f"Q{i+1}: Wrong ❌ (Correct: **{correct_option}**)")
        st.markdown(f"## 🎉 Your MCQ Score: {score}/{len(mcqs)}")

    # -------------------- 4. FIB Tab --------------------
  with tab_fib:
    st.header("Fill-in-the-Blanks")
        
    col3, col4 = st.columns([3, 1])
    num_fib = col3.slider("Number of FIBs to generate", min_value=3, max_value=15, value=5, key="fib_slider")
        
    if col4.button("Generate FIBs", type='primary', key="generate_fib_btn"):
      with st.spinner(f"Creating {num_fib} Fill-in-the-Blanks..."):
        fib_raw = generate_fib(retriever, num_questions=num_fib)

      fibs = safe_json_parse(fib_raw)
      if fibs:
        st.session_state["fibs"] = fibs
        # Initialize user inputs for FIBs
        st.session_state["fib_answers"] = [""] * len(fibs)
      else:
        st.error("⚠️ Could not parse FIBs properly. Check console for raw output.")
        print(fib_raw)

    if "fibs" in st.session_state:
      fibs = st.session_state["fibs"]
      st.markdown("### 📝 Practice Time!")

      for i, q in enumerate(fibs):
        st.markdown(f"**Q{i+1}. {q['question']}**")
        ans = st.text_input(f"Your answer for Q{i+1}", key=f"fib_q{i}")
        try:
          st.session_state["fib_answers"][i] = ans
        except IndexError:
          st.session_state["fib_answers"].append(ans)


      if st.button("Evaluate FIBs", type='secondary', key="evaluate_fib_btn"):
        score = 0
        for i, q in enumerate(fibs):
          # Strip whitespace and lower-case for lenient checking
          correct = q["answer"].strip().lower()
          user_ans = st.session_state["fib_answers"][i].strip().lower()
          if user_ans == correct:
            score += 1
            st.success(f"Q{i+1}: Correct ✅")
          else:
            st.error(f"Q{i+1}: Wrong ❌ (Correct: **{q['answer']}**)")
        st.markdown(f"## 🎉 Your FIB Score: {score}/{len(fibs)}")

else:
    st.info("Upload a PDF file above to begin your interactive learning session!")