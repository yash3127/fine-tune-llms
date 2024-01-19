import os
import torch
import streamlit as st
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders import PDFMinerLoader
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.llms import HuggingFacePipeline
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.memory import ConversationBufferMemory
from streamlit_chat import message
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import pipeline
from htmltemplate import css, bot_template, user_template

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
    db = Chroma.from_documents(documents=texts, embedding=embeddings, persist_directory=persist_directory)
    return db

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
def qa_llm(_db):
    llm = llm_pipeline()
    embeddings = SentenceTransformerEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    retriever = _db.as_retriever()
    # now retrieving
    # qa = RetrievalQA.from_chain_type(
    #     llm=llm,
    #     chain_type="stuff",
    #     retriever=retriever,
    #     return_source_documents=True
    # )
    conversational_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory = memory
    )
    return conversational_chain


def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.session_state["past"].append(user_question)
        else:
            st.session_state["generated"].append(response)

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

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    with st.spinner('Embeddings are in process...'):
        # ingested_data = data_ingestion()
        db = data_ingestion()
    st.success('Embeddings are created successfully!')

    with st.spinner('Conversational chain is in process...'):
        st.session_state.conversation = qa_llm(db)
    st.success('Created successfully!')
    st.markdown("<h4 style color:black;'>Chat Here</h4>", unsafe_allow_html=True)

    user_input = st.text_input("Please enter your query...", key="input")

    # Initialize session state for generated responses and past messages
    if "generated" not in st.session_state:
        st.session_state["generated"] = ["I am ready to help you"]
    if "past" not in st.session_state:
        st.session_state["past"] = ["Hey there!"]

    # Search the database for a response based on user input and update session state
    if user_input:
        handle_userinput(user_input)

    if "generated" in st.session_state:
        display_conversation(st.session_state)

if __name__ == "__main__":
    main()
