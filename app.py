import os
from io import BytesIO
from pathlib import Path

import numpy as np
import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import AuthenticationError, OpenAIError
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer


load_dotenv()
load_dotenv(Path("venv") / ".env", override=True)
load_dotenv(Path("venv") / ".env.local", override=True)


@st.cache_resource(show_spinner="Loading chat model...")
def get_chat_model(api_key):
    return ChatOpenAI(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
        model="openai/gpt-4o-mini",
        temperature=0.3,
    )


@st.cache_data(show_spinner="Reading and indexing PDF...")
def build_search_index(pdf_bytes):
    pdf_reader = PdfReader(BytesIO(pdf_bytes))

    pages = []
    for index, page in enumerate(pdf_reader.pages, start=1):
        page_text = page.extract_text() or ""
        if page_text.strip():
            pages.append(f"Page {index}:\n{page_text}")

    text = "\n\n".join(pages)
    if not text.strip():
        return None

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1600,
        chunk_overlap=100,
    )
    chunks = splitter.split_text(text)

    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=12000,
        ngram_range=(1, 2),
    )
    matrix = vectorizer.fit_transform(chunks)
    return chunks, vectorizer, matrix


def search_chunks(question, search_index, top_k=3):
    chunks, vectorizer, matrix = search_index
    question_vector = vectorizer.transform([question])
    scores = (matrix @ question_vector.T).toarray().ravel()
    top_indexes = np.argsort(scores)[-top_k:][::-1]
    return [chunks[index] for index in top_indexes if scores[index] > 0]


def get_api_key():
    api_key = os.getenv("OPENROUTER_API_KEY", "").strip().strip('"').strip("'")
    return api_key


st.title("AI PDF Chatbot")

pdf = st.file_uploader("Upload PDF", type="pdf")

if pdf:
    search_index = build_search_index(pdf.getvalue())
    if search_index is None:
        st.error("No readable text found in this PDF. Try a text-based PDF.")
        st.stop()

    with st.form("question_form"):
        user_question = st.text_input("Ask a question")
        submitted = st.form_submit_button("Ask")

    if submitted and user_question:
        api_key = get_api_key()
        if not api_key:
            st.error("OPENROUTER_API_KEY is missing. Add it to a .env file.")
            st.stop()

        if not api_key.startswith("sk-or-"):
            st.error("OPENROUTER_API_KEY must be an OpenRouter key that starts with sk-or-.")
            st.stop()

        chunks = search_chunks(user_question, search_index)
        if not chunks:
            st.warning("I could not find relevant text in the PDF for that question.")
            st.stop()

        context = "\n\n".join(chunks)
        model = get_chat_model(api_key)

        prompt_text = f"""
Answer the question based on the context below.

Context:
{context}

Question:
{user_question}
"""

        with st.spinner("Thinking..."):
            try:
                response = model.invoke(prompt_text)
                st.write(response.content)
            except AuthenticationError:
                st.error(
                    "OpenRouter rejected your API key. Create a new key in OpenRouter "
                    "and update OPENROUTER_API_KEY in venv/.env."
                )
            except OpenAIError as error:
                st.error(f"OpenRouter request failed: {error}")
