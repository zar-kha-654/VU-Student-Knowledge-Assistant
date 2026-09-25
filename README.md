# 🎓 VU Student Knowledge Assistant

A RAG-based Virtual University student knowledge assistant built with **Streamlit, FAISS, Sentence Transformers, and Groq**.

The application allows students to ask questions about university academic policies and retrieves relevant information from a preprocessed knowledge base before generating an answer with an LLM.

## ✨ Features

* 🔎 Semantic search using FAISS
* 🧠 Sentence Transformer embeddings
* ⚡ Groq-powered LLM responses
* 📚 Multi-document knowledge base
* 🔗 Source traceability
* 📑 Document and section references
* 💬 Conversational chat interface
* ☁️ Streamlit Cloud deployment
* 🚀 Precomputed document embeddings
* 🔐 Groq API key stored through Streamlit Secrets

## 🏗️ Architecture

```text
                    Student
                       │
                       ▼
              ┌─────────────────┐
              │    Streamlit    │
              │       UI        │
              └────────┬────────┘
                       │
                       ▼
                Student Question
                       │
                       ▼
              ┌─────────────────┐
              │ Sentence       │
              │ Transformer     │
              │ Embedding Model │
              └────────┬────────┘
                       │
                       ▼
                Query Embedding
                       │
                       ▼
              ┌─────────────────┐
              │      FAISS      │
              │ Semantic Search │
              └────────┬────────┘
                       │
                       ▼
                  Top-K Chunks
                       │
                       ▼
              ┌─────────────────┐
              │   Metadata      │
              │     JSON        │
              └────────┬────────┘
                       │
                       ▼
                 Retrieved Context
                       │
                       ▼
              ┌─────────────────┐
              │      Groq       │
              │       LLM       │
              └────────┬────────┘
                       │
                       ▼
                 Final Answer
                       │
                       ▼
                 Source References
```

## 📂 Repository Structure

```text
VU-Student-Knowledge-Assistant/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
│
└── rag_knowledge_base/
    ├── index.faiss
    ├── metadata.json
    ├── config.json
    ├── source_manifest.json
    └── README.md
```

The original source documents are **not included in this repository**.

The RAG knowledge base was generated offline from the source documents and contains the precomputed FAISS index and associated metadata.

## 🧠 Knowledge Base

The current knowledge base contains information covering:

* Academic Calendar
* Course Selection Rules
* Examination Rules
* Fee Policy
* Scholarship Policy
* Student Handbook

The documents were converted into smaller semantic chunks and embedded using:

```text
BAAI/bge-small-en-v1.5
```

The resulting embeddings were stored in FAISS.

## ⚙️ RAG Process

Document processing is performed separately from the Streamlit application.

```text
Source Documents
      ↓
Text Extraction
      ↓
Markdown Section Detection
      ↓
Chunking
      ↓
BGE Embeddings
      ↓
FAISS Index
      ↓
Metadata JSON
```

At runtime:

```text
Student Question
      ↓
Query Embedding
      ↓
FAISS Similarity Search
      ↓
Top-K Relevant Chunks
      ↓
Context
      ↓
Groq LLM
      ↓
Answer + Sources
```

Document embeddings are **not regenerated when the Streamlit application starts**.

Only the student's query is embedded at runtime.

## 🔍 Source Traceability

Each retrieved chunk contains metadata such as:

```json
{
  "chunk_id": 7,
  "source": "course_selection_rules.md",
  "document_title": "course_selection_rules",
  "section": "Course Selection Procedure",
  "section_chunk_number": 1
}
```

Because the source material is Markdown rather than PDF, the application uses **document name + section** for source traceability instead of page numbers.

## 🤖 LLM

The application uses the **Groq API** for answer generation.

The LLM is instructed to:

* Use retrieved knowledge as the primary source
* Avoid inventing university policies
* State when the knowledge base does not contain enough information
* Provide concise student-friendly answers
* Preserve source traceability

## 🔐 API Key

The Groq API key is **not stored in the source code**.

For Streamlit Cloud, add the key through:

```text
App → Settings → Secrets
```

Add:

```toml
GROQ_API_KEY = "your_groq_api_key"
```

Never commit the API key to GitHub.

## 🚀 Local Setup

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/VU-Student-Knowledge-Assistant.git
```

Move into the project:

```bash
cd VU-Student-Knowledge-Assistant
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Set your Groq API key.

For local development, create:

```text
.streamlit/secrets.toml
```

and add:

```toml
GROQ_API_KEY = "your_groq_api_key"
```

Run:

```bash
streamlit run app.py
```

## ☁️ Streamlit Cloud Deployment

1. Push the repository to GitHub.
2. Open Streamlit Cloud.
3. Create a new application.
4. Select the GitHub repository.
5. Set the main file to:

```text
app.py
```

6. Add the Groq API key under Secrets:

```toml
GROQ_API_KEY = "your_groq_api_key"
```

7. Deploy.

## 📦 Precomputed Knowledge Base

The `rag_knowledge_base` directory contains:

| File                   | Purpose                      |
| ---------------------- | ---------------------------- |
| `index.faiss`          | FAISS vector index           |
| `metadata.json`        | Chunks and source metadata   |
| `config.json`          | Embedding configuration      |
| `source_manifest.json` | Source/chunk mapping         |
| `README.md`            | Knowledge-base documentation |

## 🔒 Privacy

The original source documents are not included as individual files in this repository.

API credentials are also excluded from the repository.

## 🛠️ Technology Stack

* Python
* Streamlit
* FAISS
* Sentence Transformers
* BAAI BGE Small English v1.5
* Groq API
* NumPy

## 📌 Current Knowledge Base

```text
Documents: 6
Chunks: 25
Embedding dimension: 384
Vector database: FAISS
Embedding model: BAAI/bge-small-en-v1.5
```

## 🔮 Future Improvements

Possible future enhancements include:

* Conversation-aware retrieval
* Better reranking
* Hybrid keyword + semantic search
* More advanced citation display
* Confidence/relevance thresholds
* Additional university documents
* Admin knowledge-base updates
* Evaluation dataset for retrieval quality
* Query rewriting
* Retrieval evaluation metrics
* Document version tracking

## 📄 License

This project is intended as an educational/student knowledge-assistant project.
