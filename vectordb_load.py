##from dotenv import load_dotenv
##from langchain.document_loaders import Docx2txtLoader
from langchain_community.document_loaders import Docx2txtLoader
##from langchain.llms import OpenAI
from langchain_community.llms import OpenAI
from langchain.chains.summarize import load_summarize_chain
from langchain.prompts import PromptTemplate
##from langchain.document_loaders import PyPDFLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
import streamlit.web.cli as stcli
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
##from langchain_community.embeddings import OpenAIEmbeddings 
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
import re
import textwrap
import os

#from dotenv import load_dotenv
#load_dotenv()

OPENAI_API_KEY = os.environ['OPENAI_API_KEY']

import streamlit as st

def process_docx(docx_file):
    # Add your docx processing code here
    text=""
    #Docx2txtLoader loads the Document 
    loader=Docx2txtLoader(docx_file)
    #loader=Docx2txtLoader("CPI+Development+Standards")
    #Load Documents and split into chunks
    text = loader.load_and_split()

    return text

def normalize_text(s, sep_token = " \n "):
    s  = re.sub(r'\s+','',s).strip()
    return s

def main():
    st.title("Convert Word Docment to VectorDB")

    uploaded_file = st.file_uploader("Select Word Document", type=["docx", "doc"])
    text = ""
#    uploaded_file = "CPI+Development+Standards.docx"
    if uploaded_file is not None:
        file_extension = uploaded_file.name.split('.')[-1]

        st.write(f"File : {uploaded_file}")
        st.write("File Details:")
        st.write(f"File Name: {uploaded_file.name}")
        st.write(f"File Type: {file_extension}")

        if file_extension == "docx":
            text = process_docx(uploaded_file.name)
#            st.write(f"{text}")
        else:
            st.error("Unsupported file format. Please upload a .docx or .pdf file.")
            return
    # Split loaded documents into Chunks using CharacterTextSplitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=5000, chunk_overlap=50, separators=["\n\n", "\n", " ", ""]        
        )

        split_documents = text_splitter.split_documents(documents=text)
#        split_documents_raw = text_splitter.split_documents(documents=text)
        
#        split_documents = re.sub(r'[^\w\s]', '', split_documents_raw) 

        #print(split_documents) 

        #print(f"Split into {len(split_documents)} Documents...")
        #print(split_documents[0].metadata)        
        # for doc in split_documents:
        #     print("##---- Page ---##")
        #     print(doc.metadata['source'])
        #     print("##---- Content ---##")
        #     print(doc.page_content)

        # Upload chunks as vector embeddings into FAISS
        embeddings = OpenAIEmbeddings()
        db = FAISS.from_documents(split_documents, embeddings)
        # Save the FAISS DB locally
        db.save_local("faiss_index_2")

def faiss_query():
    """
    This function does the following:
    1. Load the local FAISS Database 
    2. Trigger a Semantic Similarity Search using a Query
    3. This retrieves semantically matching Vectors from the DB
    """
    embeddings = OpenAIEmbeddings()
    new_db = FAISS.load_local("faiss_index", embeddings,allow_dangerous_deserialization=True)

    query = "What is the Designation of the Candidate Sunil Sharma"
    docs = new_db.similarity_search(query)

    # Print all the extracted Vectors from the above Query
    for doc in docs:
        print("##---- Page ---##")
        print(doc.metadata['source'])
        print("##---- Content ---##")
        print(doc.page_content)

if __name__ == "__main__":
   main()
##main()

