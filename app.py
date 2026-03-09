import os
from dotenv import load_dotenv
import streamlit as st
import numpy as np
from google import genai
from google.genai import types
from pypdf import PdfReader

load_dotenv()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
CHUNK_SIZE = 500  # characters per chunk
CHUNK_OVERLAP = 100  # overlap between consecutive chunks
TOP_K = 3  # number of chunks to retrieve
EMBEDDING_MODEL = "gemini-embedding-001"
GENERATION_MODEL = "gemini-2.5-flash"

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def get_client(api_key: str) -> genai.Client:
    """Create a Gemini client with the given API key."""
    return genai.Client(api_key=api_key)


def extract_text(uploaded_file) -> str:
    """Extract plain text from an uploaded .txt or .pdf file."""
    if uploaded_file.name.endswith(".pdf"):
        reader = PdfReader(uploaded_file)
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    else:
        return uploaded_file.read().decode("utf-8")


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping fixed-size chunks."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        chunks.append(text[start:end])
        start += size - overlap
    return [c.strip() for c in chunks if c.strip()]


def embed_texts(client: genai.Client, texts: list[str]) -> np.ndarray:
    """Embed a list of texts using the Gemini embedding model."""
    result = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=texts,
        config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT"),
    )
    return np.array([e.values for e in result.embeddings])


def embed_query(client: genai.Client, query: str) -> np.ndarray:
    """Embed a single query string."""
    result = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=query,
        config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY"),
    )
    return np.array(result.embeddings[0].values)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute cosine similarity between vector *a* and each row in matrix *b*."""
    dot = b @ a
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b, axis=1)
    return dot / (norm_a * norm_b + 1e-10)


def retrieve(client: genai.Client, query: str, chunks: list[str], embeddings: np.ndarray, top_k: int = TOP_K) -> list[str]:
    """Return the top-k most relevant chunks for a query."""
    q_emb = embed_query(client, query)
    scores = cosine_similarity(q_emb, embeddings)
    top_indices = np.argsort(scores)[-top_k:][::-1]
    return [chunks[i] for i in top_indices]


def generate_answer(client: genai.Client, question: str, context_chunks: list[str]) -> str:
    """Send retrieved context + question to Gemini and return the answer."""
    context = "\n\n---\n\n".join(context_chunks)
    prompt = (
        "You are a helpful assistant. Answer the user's question using ONLY the "
        "provided context. If the context does not contain enough information, "
        "say so honestly.\n\n"
        f"### Context\n{context}\n\n"
        f"### Question\n{question}"
    )
    response = client.models.generate_content(model=GENERATION_MODEL, contents=prompt)
    return response.text

# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------

st.set_page_config(page_title="RAG Mini-Chatbot", page_icon="📄")
st.title("📄 RAG Mini-Chatbot")
st.caption("Upload a document and ask questions about it — powered by Gemini")

# --- Sidebar: file upload ---
with st.sidebar:
    st.header("Setup")
    uploaded_file = st.file_uploader(
        "Upload a document",
        type=["txt", "pdf"],
        help="Supported formats: .txt, .pdf",
    )

# Read API key from Streamlit secrets (cloud) or environment (.env local)
api_key = st.secrets.get("GEMINI_API_KEY") or os.environ.get("GEMINI_API_KEY")
if not api_key:
    st.error("Set GEMINI_API_KEY in Streamlit secrets or your .env file.")
    st.stop()

client = get_client(api_key)

# --- Process uploaded document ---
if uploaded_file is None:
    st.info("Upload a .txt or .pdf document in the sidebar to begin.")
    st.stop()

# Cache document processing in session state
file_id = f"{uploaded_file.name}_{uploaded_file.size}"
if st.session_state.get("file_id") != file_id:
    with st.spinner("Reading and indexing document..."):
        raw_text = extract_text(uploaded_file)
        chunks = chunk_text(raw_text)
        embeddings = embed_texts(client, chunks)
        st.session_state["file_id"] = file_id
        st.session_state["chunks"] = chunks
        st.session_state["embeddings"] = embeddings
        st.session_state["messages"] = []
    st.success(f"Indexed **{len(chunks)}** chunks from *{uploaded_file.name}*")

chunks = st.session_state["chunks"]
embeddings = st.session_state["embeddings"]

# --- Chat interface ---
if "messages" not in st.session_state:
    st.session_state["messages"] = []

for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if question := st.chat_input("Ask a question about your document"):
    st.session_state["messages"].append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            relevant_chunks = retrieve(client, question, chunks, embeddings)
            answer = generate_answer(client, question, relevant_chunks)
        st.markdown(answer)
    st.session_state["messages"].append({"role": "assistant", "content": answer})
