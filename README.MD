# **Streamlit Application: Chat with Multiple PDFs Using Gemini**

## **Purpose**
This Streamlit app allows users to:
1. Upload multiple PDF files.
2. Extract text from the PDFs.
3. Split the extracted text into manageable chunks.
4. Embed the text chunks using Google's Generative AI embeddings.
5. Save and retrieve the embeddings using FAISS (Facebook AI Similarity Search) for efficient similarity searches.
6. Ask questions about the content of the PDFs, and get accurate answers based on the embedded knowledge.

---

## **Pre-requisites**
To use this code, ensure the following dependencies are installed:

```bash
pip install streamlit PyPDF2 langchain faiss-cpu python-dotenv langchain-google-genai google-generativeai
```

You will also need:
- A **Google API Key** for Generative AI access.
- A working environment where Streamlit and Python scripts can run.

---

## **Code Breakdown**

### **1. Imports and Configuration**
The required libraries and environment variables are imported:

```python
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
```

### **2. Functions**

#### **2.1 Extract Text from PDFs**
This function takes uploaded PDF files and extracts their text content:

```python
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text
```

- **Input:** List of PDF files.
- **Output:** Combined text content from all PDFs.

---

#### **2.2 Split Text into Chunks**
This function splits the extracted text into smaller, manageable chunks for embedding:

```python
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks
```

- **Parameters:**
  - `chunk_size`: Size of each text chunk.
  - `chunk_overlap`: Overlapping portion between consecutive chunks.
- **Output:** List of text chunks.

---

#### **2.3 Generate and Store Embeddings**
The following function creates embeddings for the text chunks and saves them locally using FAISS:

```python
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")
```

- **Process:**
  1. Generate embeddings for text chunks using Google Generative AI embeddings.
  2. Store embeddings using FAISS for quick similarity searches.
- **Output:** A saved FAISS index named `faiss_index`.

---

#### **2.4 Define Conversational Chain**
This function initializes the conversational chain with a custom prompt template:

```python
def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in the 
    provided context just say, "answer is not available in the context", don't provide the wrong answer.
    Context:\n {context}?\n
    Question:\n{question}\n

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain
```

- **Custom Prompt Template:** Ensures accurate answers or a fallback response if the context does not contain the answer.
- **Model Used:** `gemini-pro` with a temperature setting of `0.3` for balanced creativity.

---

#### **2.5 Handle User Queries**
This function searches for relevant context using FAISS and responds to the user's question:

```python
def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    st.write("Reply: ", response["output_text"])
```

- **Process:**
  1. Load the saved FAISS index.
  2. Search for similar text chunks using the user's question.
  3. Pass the relevant context and question to the conversational chain.
  4. Display the response.

---

### **3. Main Application**
The main function defines the Streamlit interface:

```python
def main():
    st.set_page_config(page_title="Chat with Multiple PDF")
    st.header("Chat with Multiple PDF using Gemini")

    user_question = st.text_input("Ask a Question from the PDF files")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader(
            "Upload your PDF files and click on the Submit & Process button", 
            accept_multiple_files=True,
            type=["pdf"]
        )
        if st.button("Submit & Process"):
            if pdf_docs:
                with st.spinner("Processing..."):
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks)
                    st.success("Done")
            else:
                st.warning("Please upload at least one PDF file.")
```

- **Features:**
  - **Sidebar:**
    - File upload option for PDFs.
    - A "Submit & Process" button to process uploaded PDFs.
  - **Main Panel:**
    - A text input field where users can ask questions.
    - Processed responses are displayed on-screen.

---

### **4. Running the Application**
Save the script as `app.py` and run the following command in your terminal:

```bash
streamlit run app.py
```

---

## **How It Works**
1. **Upload PDFs**: Users upload one or more PDF files via the sidebar.
2. **Process PDFs**:
   - Extract text from uploaded PDFs.
   - Split the text into chunks.
   - Generate embeddings for the chunks and store them in FAISS.
3. **Ask Questions**: Users input a question about the content of the PDFs.
4. **Retrieve Answers**:
   - Relevant text chunks are retrieved using similarity search.
   - A detailed answer is generated using the Gemini Pro model.

---

## **Key Notes**
- **Security**: Ensure the FAISS index is loaded only from trusted sources.
- **API Key**: Store the Google API key securely in a `.env` file.
- **Chunking**: Adjust `chunk_size` and `chunk_overlap` as per the document size.

---

## **Conclusion**
This Streamlit app allows users to interactively query large PDF documents with accuracy using FAISS for vector storage and Google's Generative AI models. It's efficient, user-friendly, and scalable for large document analysis tasks.