import base64
import os
import torch
import streamlit as st
from langchain.chains import RetrievalQA
from langchain.document_loaders import PDFMinerLoader
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.llms import HuggingFacePipeline
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from streamlit_chat import message

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import pipeline

# C:\Users\yashs\anaconda3\Lib\site-packages\torch\lib - libiomp5md.dll - in desktop

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

st.set_page_config(layout="wide")

model_name = "MBZUAI/LaMini-T5-738M"
tokenizer = AutoTokenizer.from_pretrained(model_name)
base_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

persist_directory = "db"


@st.cache_resource
def data_ingestion():
    for root, dirs, files in os.walk("docs"):
        for file in files:
            if file.endswith(".pdf"):
                print(file)
                loader = PDFMinerLoader(os.path.join(root, file))
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)

    # create embeddings
    embeddings = SentenceTransformerEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # db = Chroma.from_documents( texts, embeddings, persist_directory=persist_directory, client_settings=settings)
    db = Chroma.from_documents(documents=texts, embedding=embeddings, persist_directory=persist_directory)
    db.persist()
    db = None


@st.cache_resource
def llm_pipeline():
    pipe = pipeline(
        'text2text-generation',
        model=base_model,
        tokenizer=tokenizer,
        # computation power - max length of token
        max_length=256,
        do_sample=True,
        # sampling
        temperature=0.3,
        # randomness and creativeness
        top_p=0.95
    )
    local_llm = HuggingFacePipeline(pipeline=pipe)
    return local_llm


@st.cache_resource
def qa_llm():
    llm = llm_pipeline()
    embeddings = SentenceTransformerEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = Chroma(persist_directory="db", embedding_function=embeddings)
    retriever = db.as_retriever()
    # now retrieving
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )
    return qa


def process_answer(instruction):
    response = ''
    instruction = instruction
    qa = qa_llm()
    generated_text = qa(instruction)
    answer = generated_text['result']
    return answer, generated_text


# Display conversation history using Streamlit messages
def display_conversation(history):
    for i in range(len(history["generated"])):
        message(history["past"][i], is_user=True, key=str(i) + "_user")
        message(history["generated"][i], key=str(i))


def main():
    st.title("PDF Retriver - 📄🦜")
    with st.expander("About the app"):
        st.markdown(
            """"
            This is a generative AI powered question and answering app that 
            responds to questions about your PDF file.
            """
        )

    with st.spinner('Embeddings are in process...'):
        ingested_data = data_ingestion()
    st.success('Embeddings are created successfully!')
    st.markdown("<h4 style color:black;'>Chat Here</h4>", unsafe_allow_html=True)

    user_input = st.text_input("", key="input")

    # Initialize session state for generated responses and past messages
    if "generated" not in st.session_state:
        st.session_state["generated"] = ["I am ready to help you"]
    if "past" not in st.session_state:
        st.session_state["past"] = ["Hey there!"]

    # Search the database for a response based on user input and update session state
    if user_input:
        answer = process_answer({'query': user_input})
        st.session_state["past"].append(user_input)
        response = answer
        print(response)
        st.session_state["generated"].append(response)

    # Display conversation history using Streamlit messages
    if st.session_state["generated"]:
        display_conversation(st.session_state)


if __name__ == "__main__":
    main()
