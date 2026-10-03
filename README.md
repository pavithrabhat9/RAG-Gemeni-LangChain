# Gemini LangChain RAG Chatbot

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python 3.10+" />
  <img src="https://img.shields.io/badge/LangChain-Framework-1C3C6C?style=for-the-badge" alt="LangChain" />
  <img src="https://img.shields.io/badge/Gemini-AI-8A2BE2?style=for-the-badge" alt="Gemini AI" />
  <img src="https://img.shields.io/badge/Chroma-VectorDB-00C7B7?style=for-the-badge" alt="Chroma" />
</p>

A retrieval-augmented generation (RAG) project that ingests PDF documents, chunks them into meaningful sections, embeds them with Gemini embeddings, stores them in ChromaDB, and powers a conversational chatbot using LangChain and Google Generative AI.

This project is designed for document-based Q&A workflows, making it ideal for knowledge retrieval from PDFs, manuals, reports, and other source material.

## Why this project?

Traditional LLMs are powerful, but they do not inherently know your private documents unless you provide them in context. This project solves that by combining:

- PDF ingestion
- intelligent chunking
- vector embeddings
- semantic retrieval
- LLM-powered answer generation

The result is a lightweight but practical RAG pipeline that can answer questions grounded in your document library.

## Features

- PDF ingestion from a local `Data` folder
- Recursive document chunking for optimal retrieval
- Google Gemini embeddings for semantic understanding
- ChromaDB vector storage for fast similarity search
- Retrieval of the most relevant document chunks before answering
- Conversational chatbot interface powered by Gradio
- Rate-limit-friendly batch ingestion for Gemini API usage

## Architecture

```text
PDF Files in Data/
        |
        v
PyPDFDirectoryLoader
        |
        v
RecursiveCharacterTextSplitter
        |
        v
GoogleGenerativeAIEmbeddings
        |
        v
Chroma Vector Store
        |
        v
Retriever + Gemini Chat Model
        |
        v
Gradio Chatbot UI
```

## Tech Stack

- Python
- LangChain
- LangChain Community
- Google Generative AI (Gemini)
- ChromaDB
- Gradio
- PyPDF
- python-dotenv

## Project Structure

```text
RAG-with-Gemini-and-LangChain/
├── Data/                     # PDF files for ingestion
├── chroma_db/                # Persisted vector database
├── ingest_database.py        # PDF ingestion + embedding pipeline
├── chatbot.py                # Gradio chat interface
├── .env                      # Local environment variables
├── .gitignore
├── README.md
└── requirements.txt          # Optional: add if you choose to manage packages here
```

## Prerequisites

Before running the project, make sure you have:

- Python 3.10 or newer
- A Google Gemini API key
- Access to the internet for model and embedding calls

## Installation

1. Clone the repository

```bash
git clone <your-repo-url>
cd "RAG with Gemeni and LangChain"
```

2. Create a virtual environment

```bash
python -m venv .venv
```

On Windows:

```bash
.venv\Scripts\activate
```

On macOS/Linux:

```bash
source .venv/bin/activate
```

3. Install dependencies

```bash
pip install langchain langchain-community langchain-google-genai langchain-chroma python-dotenv gradio pypdf chromadb
```

If you maintain a `requirements.txt`, you can instead run:

```bash
pip install -r requirements.txt
```

## Environment Configuration

Create a `.env` file in the project root and add your Google API key:

```env
GOOGLE_API_KEY=your_google_api_key_here
```

The scripts use `python-dotenv` to load this automatically.

## Run the Ingestion Pipeline

Place your PDF files inside the `Data` folder, then run:

```bash
python ingest_database.py
```

This script will:

- load all PDFs from the `Data` directory
- split them into chunks
- embed each chunk using Gemini
- store the vectors in `chroma_db`

You will see progress logs as batches are added to the database.

## Start the Chatbot

After ingestion, run:

```bash
python chatbot.py
```

This launches a local Gradio chatbot interface in your browser. You can ask questions based on the content of the uploaded PDFs.

## How the RAG Flow Works

1. A user asks a question in the chat UI.
2. The app retrieves the most relevant document chunks from ChromaDB.
3. Those chunks are passed into a Gemini chat prompt.
4. The model answers using only the retrieved knowledge as context.
5. The response is returned to the user.

This is a standard retrieval-augmented generation pattern that keeps answers grounded in your source documents.

## Configuration Notes

The current implementation includes a few important defaults:

- `chunk_size = 300`
- `chunk_overlap = 100`
- `BATCH_SIZE = 50`
- `num_results = 5`

These values are a good starting point for document-heavy workloads, but you may tune them depending on your PDF length and answer quality requirements.

## Rate Limiting

The ingestion script intentionally waits between batches to reduce pressure on the Gemini free-tier API.

```python
if i + BATCH_SIZE < len(chunks):
    time.sleep(60)
```

This helps avoid API throttling when processing many chunks.

## Example Use Cases

- Internal documentation Q&A assistant
- PDF-based knowledge base search
- Research paper question answering
- Manual and SOP retrieval assistant
- Enterprise document search bot

## Limitations

- The design is optimized for local document retrieval, not huge enterprise-scale indexing.
- ChromaDB is embedded locally, so scaling beyond a simple project may require a managed vector database.
- Response quality depends heavily on document quality, chunk size, and retrieval settings.

## Future Improvements

- Add multi-document support with metadata filtering
- Improve prompt templates for more accurate answers
- Add conversation memory beyond the current history
- Support uploading PDFs through the UI
- Add a web frontend with FastAPI or Streamlit
- Package the project as a reusable Python application

## License

This project is provided for educational and development purposes. Add a license if you plan to share it publicly.

Example:

```bash
MIT License
```

## Contributing

Contributions are welcome. If you improve the pipeline, add better prompts, or optimize document retrieval, feel free to submit a pull request.

## Acknowledgements

- LangChain for orchestration
- Google Generative AI for embeddings and chat models
- ChromaDB for vector search
- Gradio for the user interface

---

If you want, I can also create:

- a more polished GitHub landing README with badges and screenshots
- a `requirements.txt` file for this project
- a `.env.example` file
- a version tailored for a portfolio or open-source project
