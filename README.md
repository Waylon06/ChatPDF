# ChatPDF MVP

A lightweight ChatPDF application built with Streamlit, allowing users to upload a PDF file and ask questions based on its content.

## Features

* Upload PDF files
* Extract text from PDF using pdfplumber
* Split text into chunks
* Generate embeddings with SentenceTransformers
* Store embeddings in FAISS vector database
* Retrieve relevant chunks using semantic search
* Send relevant content to LLM for answering questions
* Display source chunks for answer transparency

## Tech Stack

* Python
* Streamlit
* pdfplumber
* SentenceTransformers
* FAISS
* Hugging Face Inference API
* OpenAI SDK

## Project Structure

ChatPDF/
│── app.py
│── .env
│── requirements.txt
│── .gitignore

## Installation

### 1. Clone the repository

```bash
git clone your-repo-url
cd ChatPDF
```

### 2. Create virtual environment

```bash
python -m venv .venv
```

### 3. Activate virtual environment

Windows:

```bash
.venv\Scripts\activate
```

Mac/Linux:

```bash
source .venv/bin/activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

## Environment Variables

Create a `.env` file:

```env
HUGGING_FACE_API_KEY=your_api_key_here
```

## Run the project

```bash
streamlit run app.py
```

## How It Works

1. User uploads a PDF
2. PDF text is extracted
3. Text is chunked into smaller parts
4. Embeddings are generated
5. FAISS retrieves top relevant chunks
6. LLM answers based on retrieved context

## Example Workflow

Upload PDF → Ask Question → Retrieve Relevant Chunks → Generate Answer

## Current Limitations

* Single PDF only
* No conversation memory
* No page number citation
* Simple chunking strategy

## Future Improvements

* Multi-PDF support
* Page citation
* Conversation memory
* Better chunking strategy
* Deploy online

## Author

Waylon
