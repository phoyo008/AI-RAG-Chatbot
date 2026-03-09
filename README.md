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

This project uses a Retrieval-Augmented Generation (RAG) approach, which grounds the
model's responses in user-provided documents rather than relying solely on pre-trained
knowledge. This design choice reduces hallucinations and makes the system more
transparent — users can trace answers back to their source material. However, responsible
use still requires caution. The embedding model may not capture nuance equally across
all topics, introducing subtle retrieval bias. The generation model can still
misinterpret context or produce plausible-sounding but inaccurate summaries. Users
should treat AI-generated answers as a starting point, not a definitive source. Privacy
is also a concern: documents uploaded to the API are sent to Google's servers, so
sensitive or personal data should be handled carefully. Developers have a responsibility
to disclose when AI is involved and to evaluate outputs for fairness and accuracy.

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
