import streamlit as st
import tempfile
import os
import fitz  # PyMuPDF
from PIL import Image
import io

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.embeddings import NomicEmbedding
from langchain_community.llms import Ollama

from transformers import BlipProcessor, BlipForConditionalGeneration
import torch

# Load BLIP model and processor for image captioning
@st.cache_resource
def load_blip_model():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return processor, model

processor, blip_model = load_blip_model()

# Generate captions from image bytes
def generate_caption(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    inputs = processor(image, return_tensors="pt")
    out = blip_model.generate(**inputs)
    return processor.decode(out[0], skip_special_tokens=True)

# Extract images from PDF
def extract_images_from_pdf(pdf_path):
    images = []
    doc = fitz.open(pdf_path)
    for page_index in range(len(doc)):
        for img_index, img in enumerate(doc.get_page_images(page_index)):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            images.append(image_bytes)
    return images

# Streamlit App
st.set_page_config(page_title="📄 Multimodal RAG Chatbot", layout="wide")
st.title("📄 Multimodal RAG Chatbot with Nomic + Ollama + FAISS")

uploaded_files = st.file_uploader("Upload one or more PDF files", type="pdf", accept_multiple_files=True)
query = st.text_input("Ask a question about the uploaded documents")

if uploaded_files:
    all_text_chunks = []

    with st.spinner("Processing documents..."):
        for uploaded_file in uploaded_files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name

            # Extract text
            loader = PyPDFLoader(tmp_path)
            text_docs = loader.load()
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
            chunks = splitter.split_documents(text_docs)

            # Extract and caption images
            image_bytes_list = extract_images_from_pdf(tmp_path)
            for img_bytes in image_bytes_list:
                caption = generate_caption(img_bytes)
                chunks.append({"page_content": f"Image Caption: {caption}", "metadata": {}})

            all_text_chunks.extend(chunks)
            os.remove(tmp_path)

    # Create vector store
    with st.spinner("Embedding with Nomic and creating FAISS index..."):
        embedder = NomicEmbedding(model="nomic-embed-text-v1")
        vectorstore = FAISS.from_documents(all_text_chunks, embedding=embedder)
        retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4})
        llm = Ollama(model="llama3")
        qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

    if query:
        with st.spinner("Generating answer..."):
            answer = qa_chain.run(query)
            st.markdown("### 🤖 Answer")
            st.write(answer)
