# AI PDF Chatbot

An AI-powered PDF chatbot built using Python, LangChain, FAISS, Streamlit, and OpenRouter.

Upload a PDF document and ask questions about its content using Generative AI and Retrieval-Augmented Generation (RAG).

---

## Features

* Upload PDF files
* Ask questions from uploaded documents
* AI-generated answers
* Semantic search using embeddings
* Vector database support with FAISS
* Interactive Streamlit UI
* OpenRouter LLM integration

---

## Tech Stack

| Technology            | Purpose           |
| --------------------- | ----------------- |
| Python                | Backend           |
| Streamlit             | User Interface    |
| LangChain             | LLM Orchestration |
| FAISS                 | Vector Database   |
| Sentence Transformers | Embeddings        |
| OpenRouter            | LLM API Provider  |

---

## Project Architecture

PDF → Text Extraction → Chunking → Embeddings →
FAISS Vector Store → Similarity Search → LLM → Response

This project uses Retrieval-Augmented Generation (RAG) architecture.

---

## Installation

### 1. Clone Repository

```bash
git clone https://github.com/YOUR_USERNAME/pdf-chatbot.git
cd ai-pdf-chatbot
```

---

### 2. Create Virtual Environment

#### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

#### Linux / Mac

```bash
python3 -m venv venv
source venv/bin/activate
```

---

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Environment Variables

Create a `.env` file in the root directory.

```env
OPENROUTER_API_KEY=your_openrouter_api_key
```

Get your API key from:

https://openrouter.ai

---

## Run Application

```bash
streamlit run app.py
```

---

## Project Structure

```text
ai-pdf-chatbot/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── .env
└── venv/
```

---

## Example Workflow

1. Upload PDF
2. Extract text from PDF
3. Convert text into embeddings
4. Store embeddings in FAISS
5. Ask questions
6. Retrieve relevant chunks
7. Generate AI response

---

## Future Improvements

* Multiple PDF support
* Chat history
* PDF summarization
* Source citations
* Authentication system
* Voice input support
* Better UI/UX

---

## Learning Outcomes

This project helps understand:

* Retrieval-Augmented Generation (RAG)
* Embeddings
* Vector Databases
* Semantic Search
* LLM APIs
* Prompt Engineering
* AI Application Development

---

## Deployment

You can deploy this project on:

* Streamlit Cloud
* Render
* Railway

---

## Author

Manjunath
B.Tech AIML Student | Generative AI Enthusiast

