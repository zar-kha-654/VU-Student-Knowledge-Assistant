from __future__ import annotations

import json
from pathlib import Path

import faiss
import numpy as np
import streamlit as st
from groq import Groq
from sentence_transformers import SentenceTransformer


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

RAG_DIR = BASE_DIR / "rag_knowledge_base"

FAISS_INDEX_PATH = RAG_DIR / "index.faiss"
METADATA_PATH = RAG_DIR / "metadata.json"
CONFIG_PATH = RAG_DIR / "config.json"


# The same embedding model used when creating the FAISS index
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"

# Number of chunks retrieved from FAISS
TOP_K = 5

# Groq model
GROQ_MODEL = "openai/gpt-oss-20b"


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="VU Student Knowledge Assistant",
    page_icon="🎓",
    layout="centered",
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }

    .subtitle {
        color: #666;
        margin-bottom: 1.5rem;
    }

    .source-box {
        padding: 0.8rem;
        border-radius: 0.5rem;
        border: 1px solid #ddd;
        margin-top: 0.5rem;
        font-size: 0.9rem;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">🎓 VU Student Knowledge Assistant</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="subtitle">'
    "Ask questions about Virtual University academic policies, "
    "courses, examinations, fees, scholarships, and student "
    "handbook information."
    "</div>",
    unsafe_allow_html=True,
)


# ============================================================
# LOAD RAG COMPONENTS
# ============================================================

@st.cache_resource
def load_embedding_model():

    return SentenceTransformer(
        EMBEDDING_MODEL_NAME
    )


@st.cache_resource
def load_faiss_index():

    if not FAISS_INDEX_PATH.exists():
        raise FileNotFoundError(
            f"FAISS index not found: {FAISS_INDEX_PATH}"
        )

    return faiss.read_index(
        str(FAISS_INDEX_PATH)
    )


@st.cache_data
def load_metadata():

    if not METADATA_PATH.exists():
        raise FileNotFoundError(
            f"Metadata file not found: {METADATA_PATH}"
        )

    with open(
        METADATA_PATH,
        "r",
        encoding="utf-8",
    ) as file:

        return json.load(file)


# ============================================================
# INITIALIZE
# ============================================================

try:

    embedding_model = load_embedding_model()

    faiss_index = load_faiss_index()

    metadata = load_metadata()

    chunks = metadata["chunks"]

except Exception as error:

    st.error(
        "The knowledge base could not be loaded."
    )

    st.exception(error)

    st.stop()


# ============================================================
# GROQ CLIENT
# ============================================================

groq_api_key = st.secrets.get(
    "GROQ_API_KEY"
)

if not groq_api_key:

    st.error(
        "GROQ_API_KEY is not configured."
    )

    st.info(
        "Add GROQ_API_KEY to your Streamlit Cloud secrets."
    )

    st.stop()


client = Groq(
    api_key=groq_api_key
)


# ============================================================
# RETRIEVAL FUNCTION
# ============================================================

def retrieve_documents(
    query: str,
    top_k: int = TOP_K,
):

    # Create embedding for the USER QUERY only.
    query_embedding = embedding_model.encode(
        [query],
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    query_embedding = np.asarray(
        query_embedding,
        dtype="float32",
    )

    # Search FAISS
    scores, indices = faiss_index.search(
        query_embedding,
        min(top_k, faiss_index.ntotal),
    )

    retrieved = []

    for score, index_position in zip(
        scores[0],
        indices[0],
    ):

        if index_position < 0:
            continue

        if index_position >= len(chunks):
            continue

        chunk = chunks[index_position].copy()

        chunk["similarity_score"] = float(
            score
        )

        retrieved.append(chunk)

    return retrieved


# ============================================================
# CONTEXT BUILDER
# ============================================================

def build_context(retrieved_documents):

    context_parts = []

    for number, document in enumerate(
        retrieved_documents,
        start=1,
    ):

        source = document.get(
            "source",
            "Unknown source",
        )

        section = document.get(
            "section",
            "Unknown section",
        )

        text = document.get(
            "text",
            "",
        )

        context_parts.append(
            f"""
SOURCE {number}
Document: {source}
Section: {section}

Content:
{text}
"""
        )

    return "\n".join(
        context_parts
    )


# ============================================================
# GROQ GENERATION
# ============================================================

def generate_answer(
    question,
    retrieved_documents,
):

    context = build_context(
        retrieved_documents
    )

    system_prompt = """
You are the Virtual University Student Knowledge Assistant.

Your job is to answer student questions using ONLY the
provided knowledge-base context.

Rules:

1. Use the retrieved context as your primary source of truth.
2. Do not invent university policies, rules, dates, fees,
   requirements, or procedures.
3. If the answer is not supported by the provided context,
   clearly say that the available knowledge base does not
   contain enough information.
4. Give a direct and easy-to-understand answer.
5. When relevant, mention the document and section that
   support the answer.
6. If multiple retrieved sources contain relevant information,
   combine them carefully.
7. Do not claim that a source says something when it does not.
"""

    user_prompt = f"""
Student question:

{question}

Retrieved knowledge-base context:

{context}

Answer the student's question using the retrieved context.
"""

    completion = client.chat.completions.create(

        model=GROQ_MODEL,

        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],

        temperature=0.1,

        max_completion_tokens=800,
    )

    return completion.choices[0].message.content


# ============================================================
# DISPLAY SOURCES
# ============================================================

def display_sources(
    retrieved_documents,
):

    if not retrieved_documents:
        return

    st.markdown("### 📚 Sources")

    for number, document in enumerate(
        retrieved_documents,
        start=1,
    ):

        source = document.get(
            "source",
            "Unknown",
        )

        section = document.get(
            "section",
            "Unknown",
        )

        score = document.get(
            "similarity_score",
            0.0,
        )

        with st.expander(
            f"{number}. {source} — {section}"
        ):

            st.write(
                f"**Similarity:** {score:.3f}"
            )

            st.write(
                f"**Document:** {source}"
            )

            st.write(
                f"**Section:** {section}"
            )


# ============================================================
# CHAT HISTORY
# ============================================================

if "messages" not in st.session_state:

    st.session_state.messages = []


for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )


# ============================================================
# USER QUESTION
# ============================================================

question = st.chat_input(
    "Ask a question about Virtual University..."
)


if question:

    # Display user message
    with st.chat_message("user"):

        st.markdown(question)

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question,
        }
    )


    # Retrieve relevant knowledge
    with st.spinner(
        "Searching the knowledge base..."
    ):

        retrieved_documents = retrieve_documents(
            question,
            TOP_K,
        )


    # Generate answer
    with st.chat_message("assistant"):

        with st.spinner(
            "Generating answer..."
        ):

            try:

                answer = generate_answer(
                    question,
                    retrieved_documents,
                )

                st.markdown(answer)

            except Exception as error:

                st.error(
                    "Something went wrong while "
                    "generating the answer."
                )

                st.exception(error)

                answer = (
                    "I couldn't generate an answer "
                    "right now."
                )


    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
        }
    )


    # Sources
    display_sources(
        retrieved_documents
    )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header(
        "🎓 Knowledge Base"
    )

    document_count = metadata.get(
        "document_count",
        0,
    )

    chunk_count = metadata.get(
        "chunk_count",
        0,
    )

    st.metric(
        "Documents",
        document_count,
    )

    st.metric(
        "Knowledge Chunks",
        chunk_count,
    )

    st.caption(
        "Powered by FAISS + "
        "Sentence Transformers + Groq"
    )

    st.divider()

    st.markdown(
        """
        **Topics covered**

        • Academic Calendar  
        • Course Selection  
        • Examination Rules  
        • Fee Policy  
        • Scholarships  
        • Student Handbook
        """
    )
