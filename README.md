# RAG Mini-Chatbot — AI Integration Mini-Lab

A Streamlit-based chatbot that answers questions about uploaded documents using
Retrieval-Augmented Generation (RAG) powered by Google's Gemini API.

## Model Name & Source

| Component | Model | Source |
|-----------|-------|--------|
| Text Generation | **Gemini 2.0 Flash** (`gemini-2.0-flash`) | [Google AI Studio](https://aistudio.google.com/) |
| Embeddings | **Gemini Embedding 001** (`gemini-embedding-001`) | [Google AI Studio](https://aistudio.google.com/) |

## Rationale for Selection

Gemini 2.0 Flash was chosen because it offers strong performance for question-answering
tasks while being fast and available on a free tier — making it accessible for academic
projects without requiring paid API access. Google's `embedding-001` model was selected
for the retrieval step because it integrates seamlessly with the same SDK
(`google-genai`), eliminating the need for additional libraries or services. Using
a single provider for both embeddings and generation simplifies the architecture and
reduces dependencies, which is practical for a mini-lab scope.

## Responsible AI Use — Reflection

Building Solutions with AI
AI is a key tool to solve problems. We must consider AI to offer strong solutions, but we must act with care.

How This Project Works
This project uses Retrieval-Augmented Generation (RAG). RAG links AI answers to your documents. It does not use general training data. This method provides two benefits:

Stops false facts: The AI bases its text on your words.

Adds trust: Users can check the source for every answer.

Risks and Duties
We must stay alert. The system has limits:

Reading errors: The AI might miss the point and write a bad summary.

Data privacy: The API sends files to Google servers. You must protect private data.

Creator rules: Builders must state when they use AI. They must check the work for truth.

Treat AI answers as a first step, not the final word.
## Setup & Usage

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set your API key
export GEMINI_API_KEY="your-key-here"

# 3. Run the app
streamlit run app.py
```

A sample document is provided in `sample_data/sdir_overview.txt` for quick testing.

## How It Works

1. **Upload** a `.txt` or `.pdf` document via the sidebar.
2. The document is split into overlapping chunks (~500 characters each).
3. Each chunk is embedded using Gemini's embedding model.
4. When you ask a question, the query is embedded and compared against chunk embeddings
   using cosine similarity.
5. The top 3 most relevant chunks are sent as context to Gemini 2.0 Flash, which
   generates a grounded answer.

## Tech Stack

- **Python 3.12**
- **Streamlit** — web UI
- **google-genai** — Gemini API SDK
- **NumPy** — cosine similarity computation
- **PyPDF** — PDF text extraction
