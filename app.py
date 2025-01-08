import streamlit as st
from PyPDF2 import PdfReader
from langchain_groq import ChatGroq
from embedding import create_vector_embedding
from llm import get_response
import os

def main():
    st.title("RAG PDF Query App")
    
    # Enter Groq API Key
    groq_api = st.text_input("Groq API Key : ", type="password")
    hf_api = st.text_input("Huggingface API Key : ", type="password")
    os.environ['HF_TOKEN'] = hf_api
    print(groq_api)
    if groq_api:
        try:
            llm=ChatGroq(groq_api_key=groq_api,model_name="Llama3-8b-8192")
            st.success("LLAMA3-8b-8192 Initialised")
        except:
            st.error("Could not initialize, check your api key")

    #Upload PDF File
    uploaded_file = st.file_uploader("Upload PDF file", type=["pdf"], accept_multiple_files=False)
    
    if uploaded_file:
        st.success("File uploaded successfully!")

        temp_file = "./temp.pdf"
        with open(temp_file, "wb") as file:
            file.write(uploaded_file.getvalue())
            file_name = uploaded_file.name
        print(st.session_state)
        create_vector_embedding(st.session_state,temp_file)
        st.success("Vectors stored successfully!")

        #Enter Query
        query = st.text_input("Enter your query:")
        
        if query:
            response = get_response(query,llm,st.session_state)
            st.write(response['answer'])

            ## streamlit expander to show context
            with st.expander("Document similarity Search"):
                for i,doc in enumerate(response['context']):
                    st.write(doc.page_content)
                    st.write('------------------------')
        
                    
if __name__ == "__main__":
    main()
