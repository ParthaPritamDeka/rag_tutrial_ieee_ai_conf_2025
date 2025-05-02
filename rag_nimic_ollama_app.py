"""
This script implements a PDF Q&A chatbot using Streamlit and LangChain. The chatbot allows users to upload PDF files, 
process the content, and ask questions based on the uploaded documents. Below is a step-by-step explanation of the code:

1. **Import Required Libraries**:
    - The script imports necessary libraries such as `streamlit` for the web interface, `tempfile` and `os` for temporary file handling, 
      and various modules from LangChain for document processing, embeddings, and retrieval-based question answering.

2. **Set Streamlit Page Configuration**:
    - The `st.set_page_config` function is used to configure the Streamlit app's title, layout, and page icon.

3. **Add Custom Header and Footer**:
    - Custom HTML and CSS styling are added to create a visually appealing header and footer for the app. 
    - The header introduces the app's purpose, and the footer provides a friendly closing note.

4. **Initialize Session State**:
    - A session state variable `history` is initialized to store the chat history (questions and answers) across user interactions.

5. **File Uploader and Query Input**:
    - A file uploader widget (`st.file_uploader`) allows users to upload one or more PDF files.
    - A text input widget (`st.text_input`) enables users to type their questions based on the uploaded documents.

6. **Process Uploaded PDF Files**:
    - If files are uploaded:
      - The script reads the content of each PDF file using `PyPDFLoader` from LangChain.
      - Temporary files are created to handle the uploaded PDFs, and the content is extracted and stored in a list.
      - The temporary files are deleted after processing.

7. **Chunk the Document Text**:
    - The extracted text is split into smaller chunks using `RecursiveCharacterTextSplitter` for better context management.
    - Chunk size and overlap are configured to ensure meaningful context in each chunk.

8. **Create a Vector Store**:
    - A vector store is created to store embeddings of the document chunks for efficient retrieval.
    - The `OllamaEmbeddings` model is used to generate embeddings for the text chunks.
    - The `FAISS` library is used to build the vector store.
    - The `@st.cache_resource` decorator ensures that the vector store creation is cached to avoid recomputation.

9. **Set Up a Retriever**:
    - A retriever is created from the vector store using the "MMR" (Maximal Marginal Relevance) search type.
    - This ensures diverse and relevant document retrieval for answering questions.

10. **Initialize the LLM (Language Model)**:
     - The `Ollama` model (e.g., "llama3") is used as the language model for generating answers.
     - The temperature parameter is set to control the creativity of the responses.

11. **Build the Retrieval-QA Chain**:
     - A `RetrievalQA` chain is created by combining the retriever and the language model.
     - This chain is responsible for answering user queries based on the retrieved document chunks.

12. **Handle User Queries**:
     - If a query is provided:
        - The script uses the QA chain to generate an answer.
        - The question and answer are appended to the session state history.
        - The answer is displayed to the user.

13. **Display Chat History**:
     - The chat history (questions and answers) is displayed below the main interface to provide context for the conversation.

14. **Add a Footer**:
     - A footer is added to the app with a friendly message, enhancing the user experience.

### Key Features:
- **PDF Upload**: Users can upload multiple PDF files for processing.
- **Text Chunking**: Documents are split into manageable chunks for better context.
- **Vector Store**: Efficient storage and retrieval of document embeddings.
- **Question Answering**: Users can ask questions, and the app provides AI-generated answers.
- **Chat History**: A persistent chat history is maintained for user reference.

### How to Enhance:
- Add support for other file formats (e.g., Word, text files).
- Implement advanced search options (e.g., keyword-based search).
- Customize the language model or embeddings for specific domains.
- Improve the UI/UX with additional styling or interactive elements.
"""



import streamlit as st
import tempfile
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_ollama import OllamaEmbeddings
from langchain_community.llms import Ollama


# Set page configuration
st.set_page_config(page_title="📄 PDF Q&A Chatbot", layout="wide", page_icon="🤖")

# Add a custom header with styling
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #4CAF50;
        text-align: center;
        margin-bottom: 20px;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #555;
        text-align: center;
        margin-bottom: 30px;
    }
    .footer {
        font-size: 0.9rem;
        color: #888;
        text-align: center;
        margin-top: 50px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="main-header">📄 RAG Chatbot with OLLAMA - llama3, FAISS, Nomic Embedding </div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Upload your PDF documents and ask questions to get AI-powered answers!</div>', unsafe_allow_html=True)

# Initialize session state for history
# if "history" not in st.session_state:
#     st.session_state["history"] = []

# File uploader and query input
uploaded_files = st.file_uploader("📂 Upload one or more PDF files", type="pdf", accept_multiple_files=True)
query = st.text_input("💬 Ask a question based on the documents", placeholder="Type your question here...")

if uploaded_files:
    all_texts = []

    with st.spinner("📖 Reading and chunking PDF files..."):
        for uploaded_file in uploaded_files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.read())
                loader = PyPDFLoader(tmp.name)
                documents = loader.load()
                all_texts.extend(documents)
            os.remove(tmp.name)

        # Adjust chunk size and overlap for better context
        splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200)
        chunks = splitter.split_documents(all_texts)

    # Cache vector store creation to avoid recomputation
    @st.cache_resource
    def create_vectorstore(_chunks):
        """
        Creates a vector store using FAISS (Facebook AI Similarity Search) from the provided document chunks.

        This function utilizes the OllamaEmbeddings model to generate embeddings for the input document chunks.
        The embeddings are then used to create a FAISS vector store, which is a highly efficient library for 
        similarity search and clustering of dense vectors.

        Args:
            _chunks (list): A list of document chunks. Each chunk is expected to be a string or text data 
                that represents a portion of a document.

        Returns:
            FAISS: A FAISS vector store object containing the embeddings of the input document chunks.

        Step-by-step explanation:
        1. **Initialize the Embedding Model**:
           - The `OllamaEmbeddings` model is initialized with the `nomic-embed-text` model.
           - This model is specifically designed to convert text data into dense vector representations (embeddings).
           - nomic-embed-text is a large context length text encoder that surpasses OpenAI text-embedding-ada-002 
             and text-embedding-3-small performance on short and long context tasks.
           - The `base_url` parameter specifies the local server endpoint (`http://localhost:11434`) where the 
             embedding model is hosted.

        2. **Generate Embeddings**:
           - The `OllamaEmbeddings` model processes each document chunk in `_chunks` and generates a corresponding 
             dense vector representation. These embeddings capture the semantic meaning of the text.

        3. **Create FAISS Vector Store**:
           - The `FAISS.from_documents` method takes the document chunks and their embeddings as input.
           - It creates a vector store, which is a data structure optimized for fast similarity searches. 
             This allows you to efficiently find documents that are semantically similar to a given query.

        Use Case:
        - This function is useful in applications like semantic search, recommendation systems, and clustering 
          where you need to compare and retrieve similar documents based on their content.

        Note:
        - Ensure that the Ollama embedding server is running locally at the specified `base_url` before using this function.
        - The FAISS library must be installed and properly configured in your environment.

        The FAISS retriever: is a critical component of the Retrieval-Augmented Generation (RAG) pipeline. It enables efficient 
        and relevant retrieval of document chunks based on user queries. Below is a step-by-step explanation of how the retriever 
        is set up and used in the provided code:
        1. **Purpose of the Retriever**:
            - The retriever is responsible for fetching the most relevant chunks of text from the vector store based on the user's query.
            - It ensures that the language model (LLM) has access to the most contextually relevant information to generate accurate answers.
        2. **Vector Store Creation**:
            - A vector store is a database that stores embeddings (numerical representations) of document chunks.
            - In this code, the FAISS (Facebook AI Similarity Search) library is used to create the vector store.
            - FAISS is optimized for fast similarity searches, making it ideal for large-scale document retrieval tasks.
        3. **Embedding Generation**:
            - The `OllamaEmbeddings` model is used to generate embeddings for each document chunk.
            - These embeddings capture the semantic meaning of the text, allowing for similarity-based retrieval.
        4. **Chunk Retrieval**:
            - The retriever is created from the FAISS vector store using the `as_retriever` method.
            - The `search_type` is set to "mmr" (Maximal Marginal Relevance), which balances relevance and diversity in the retrieved chunks.
            - The `search_kwargs` parameter specifies additional settings:
                - `k`: The number of chunks to return to the user.
                - `fetch_k`: The number of chunks to initially fetch before applying the MMR algorithm.
        5. **How the Retriever Works**:
            - When a query is provided, the retriever compares the query's embedding with the embeddings in the vector store.
            - It calculates the similarity between the query and each document chunk.
            - The MMR algorithm ensures that the retrieved chunks are not only relevant but also diverse, avoiding redundancy.
        6. **Integration with the QA Chain**:
            - The retriever is passed to the `RetrievalQA` chain, which combines it with the language model.
            - The retriever fetches the relevant chunks, and the language model uses these chunks to generate an answer to the user's query.
        7. **Caching for Efficiency**:
            - The `@st.cache_resource` decorator is used to cache the vector store creation process.
            - This ensures that the embeddings and vector store are not recomputed every time the app runs, improving performance.
        ### Key Benefits of Using FAISS Retriever:
        - **Efficiency**: FAISS is designed for fast similarity searches, even with large datasets.
        - **Relevance and Diversity**: The MMR search type ensures that the retrieved chunks are both relevant and non-redundant.
        - **Scalability**: FAISS can handle large-scale document collections, making it suitable for real-world applications.
        By combining FAISS with the Ollama embeddings and the RetrievalQA chain, the app provides a robust and efficient mechanism 
        for answering user queries based on uploaded PDF documents.
        """
        embeddings = OllamaEmbeddings(model='nomic-embed-text', base_url="http://localhost:11434")
        return FAISS.from_documents(_chunks, embedding=embeddings)


    with st.spinner("🔍 Creating vector store with Ollama embeddings..."):
        vectorstore = create_vectorstore(chunks)

    # Use MMR for diverse and relevant retrieval
    retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 4, "fetch_k": 10})
    llm = Ollama(model="llama3", temperature=0.7)  # Adjust temperature for creativity

    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

    if query:
        with st.spinner("🤔 Generating answer..."):
            try:
                result = qa_chain.run(query)
                #st.session_state["history"].append({"question": query, "answer": result})
                st.markdown("### 🤖 Answer")
                st.success(result)
            except Exception as e:
                st.error(f"An error occurred: {e}")

# Display chat history
# if st.session_state["history"]:
#     st.markdown("## 📝 Chat History")
#     for entry in st.session_state["history"]:
#         st.markdown(f"**Q:** {entry['question']}")
#         st.markdown(f"**A:** {entry['answer']}")

# Add a footer
st.markdown('<div class="footer">Made with ❤️ using Streamlit and LangChain</div>', unsafe_allow_html=True)
