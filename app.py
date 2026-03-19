import streamlit as st
import pdfplumber
from openai import OpenAI
import requests
from dotenv import load_dotenv
import os

load_dotenv()
client = OpenAI(
    api_key=os.getenv("Hugging_Face_API_KEY"),
    base_url="https://router.huggingface.co/v1"
)

# use the upload box from streamlit
upload_file = st.file_uploader("Upload a PDF", type="pdf")

text = ""
if upload_file :
    st.write("File uploaded successfully")
    # use padfplumber to read the pdf
    with pdfplumber.open(upload_file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page.extract_text()
                
    st.write("PDF Preview:")
    st.write(text[:1000])

# input box
question = st.text_input("Ask a question")
if question:
    st.write("Your question:", question)

if question and text:
    prompt = f"""
    Based on the PDF content below:

    {text[:3000]}

    Answer this question:
    {question}
    """
    response = client.chat.completions.create(
    model="Qwen/Qwen3.5-9B:together",
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