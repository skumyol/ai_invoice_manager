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
import base64

load_dotenv()

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)  # PdfReader accepts a BytesIO object
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(text):
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=10000,chunk_overlap=1000)
    chunks=text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings=GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store=FAISS.from_texts(text_chunks,embedding=embeddings)
    vector_store.save_local("faiss_index")


def get_conversational_chain():
    prompt_template="""
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in the 
    provided context just say,"answer is not available in the context",don't provide the wrong answer
    Context:\n {context}?\n
    Question:\n{question}\n

    Answer:
    """
    model=ChatGoogleGenerativeAI(model="gemini-pro",temperature=0.3)

    prompt=PromptTemplate(template=prompt_template,input_variables=["context","question"])
    chain=load_qa_chain(model,chain_type="stuff",prompt=prompt)
    return chain


def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    # Allow dangerous deserialization only if the source is trusted
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()

    response = chain(
        {"input_documents": docs, "question": user_question},
        return_only_outputs=True
    )

    print(response)
    st.write("Reply: ", response["output_text"])


def main():
    st.set_page_config(page_title="Chat with Multiple PDFs")
    st.header("Chat with Multiple PDFs")

    user_question = st.text_input("Ask a Question from the PDF files")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader(
            "Upload your PDF files and click on the Submit & Process button", 
            accept_multiple_files=True,  # Allow multiple files to be uploaded
            type=["pdf"]
        )
        
        show_pdf = st.checkbox("Show/Hide Uploaded PDFs", value=False)
        
        if st.button("Submit & Process"):
            if pdf_docs:
                with st.spinner("Processing..."):
                    raw_text = get_pdf_text(pdf_docs)  # Pass the list of uploaded files
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks)
                    st.success("Done")
            else:
                st.warning("Please upload at least one PDF file.")
                
                
    if show_pdf and pdf_docs:
        st.subheader("Uploaded PDF Files")
        for i, pdf in enumerate(pdf_docs):
            pdf_container = st.expander(f"PDF {i+1}: {pdf.name}")
            with pdf_container:
                # Display the PDF
                pdf_display = f"""
                    <iframe src="data:application/pdf;base64,{base64.b64encode(pdf.read()).decode('utf-8')}" 
                    width="100%" height="600" type="application/pdf"></iframe>
                """
                pdf.seek(0)  # Reset file pointer after reading
                st.markdown(pdf_display, unsafe_allow_html=True)
    elif show_pdf and not pdf_docs:
        st.info("No PDFs uploaded yet. Please upload PDFs in the sidebar.")


if __name__== "__main__":
    main()