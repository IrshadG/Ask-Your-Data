# Imports
from keys import OPENAI_API_KEY
import streamlit as st
import pandas as pd
from pandasai import SmartDataframe
from pandasai.llm import OpenAI as pai_OpenAI
import os
import tempfile
from langchain.agents import create_csv_agent
from langchain.llms import OpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from PyPDF2 import PdfReader 
from pandasai.responses.streamlit_response import StreamlitResponse
from pandasai.responses.response_parser import ResponseParser

os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY

# Configure Streamlit page
st.set_page_config(page_title="Ask-Your-Data")
st.header("Ask-Your-Data")

def read_pdf(file):
    reader = PdfReader(file) 
    cont=""

    for page in reader.pages:
        cont = cont +  page.extract_text() 
    cont = cont.replace("\n", " ")
    return cont

# Ask data query
def ask_pandasai(df, question:str):
    llm = pai_OpenAI(api_token=OPENAI_API_KEY)
    sdf = SmartDataframe(df, config={"llm": llm, 
                                     "response_parser": StreamlitResponse,
                                    #  "save_charts" : False,
                                     "verbose" : True
                                     })
    response = sdf.chat(question)
    return response

# CSV query
def csv_reader(file, user_input):
    # Create a temporary file to store the uploaded CSV data
    with tempfile.NamedTemporaryFile(mode='w+', suffix=".csv", delete=False) as f:
        # Convert bytes to a string before writing to the file
        data_str = file.getvalue().decode('utf-8')
        f.write(data_str)
        f.flush()

        # Create an instance of the OpenAI language model with temperature set to 0
        llm = OpenAI(temperature=0.15)
        # Create a CSV agent using the OpenAI language model and the temporary file
        agent = create_csv_agent(llm, f.name, verbose=True)
        response = agent.run(user_input)
        print(response)
        return response

# PDF query
def pdf_reader(file, user_input):
    
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    documents = text_splitter.create_documents([file])

    vectordb = Chroma.from_documents(
    documents,
    embedding=OpenAIEmbeddings(),
    persist_directory='./data'
    )
    vectordb.persist()

    qa_chain = RetrievalQA.from_chain_type(
    llm=OpenAI(temperature=0.15),
    retriever=vectordb.as_retriever(search_kwargs={'k': 7}),
    return_source_documents=True
    )

    result = qa_chain({'query': user_input})
    return result['result']

# Allow the user to upload a CSV file
file = st.file_uploader("upload your csv file", type=["csv", "pdf"])
if file is not None:    
    # Ask the user to input a question
    user_input = st.text_input("Ask a question:")

    if user_input and st.button('Submit'):
        if file.name[-4:] == '.csv':
            with st.spinner(text="Loading..."):
                df = pd.read_csv(file)
                response = ask_pandasai(df=df, question=user_input)
                # response = csv_reader(file=file, user_input=user_input)
                st.write(response)
        elif file.name[-4:] == '.pdf':
            with st.spinner(text="Loading..."):
                response = pdf_reader(file=read_pdf(file), user_input=user_input)
                st.write(response)

