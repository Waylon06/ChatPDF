import streamlit as st
import pdfplumber
from openai import OpenAI
from dotenv import load_dotenv
import os
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

load_dotenv()

# function to chunk the text into smaller pieces
def chunk_text(text, chunk_size=500):
    chunks = []
    for i in range(0, len(text), chunk_size):
        chunks.append(text[i:i+chunk_size])
    return chunks

# load the model acvoid re-loading the model every time
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")
embed_model = load_model()

# process the pdf
@st.cache_data
def process_pdf(text):
    chunks = chunk_text(text)
    embeddings = embed_model.encode(chunks)
    return chunks, embeddings

client = OpenAI(
    api_key=os.getenv("HUGGING_FACE_API_KEY"),
    base_url="https://router.huggingface.co/v1"
)



# use the upload box from streamlit
upload_file = st.file_uploader("Upload a PDF", type="pdf")

text = ""
relevant_chunks = []
if upload_file :
    st.write("File uploaded successfully")
    # use padfplumber to read the pdf
    with pdfplumber.open(upload_file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
                
    st.write("PDF Preview:")
    st.write(text[:1000])

# input box
question = st.text_input("Ask a question")
if question:
    st.write("Your question:", question)




if question and text:
    chunks, chunk_embeddings = process_pdf(text)

    dimension = chunk_embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(chunk_embeddings))
    question_embedding = embed_model.encode([question])
    distances, indices = index.search(np.array(question_embedding), k=3)
    relevant_chunks = [chunks[i] for i in indices[0]]

    prompt = f"""
    Based on the PDF content below:

    {' '.join(relevant_chunks)}

    Answer this question:
    {question}
    """
    response = client.chat.completions.create(
    model="Qwen/Qwen2.5-7B-Instruct",
    messages=[
        {
            "role": "user", 
            "content": prompt
        }
    ],
    )
    answer = response.choices[0].message.content

    st.write("Answer:")
    st.write(answer)