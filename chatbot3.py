import os
import torch
import streamlit as st
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders import PDFMinerLoader
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.llms import HuggingFacePipeline, HuggingFaceHub
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from streamlit_chat import message
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import pipeline
from htmltemplate import css, bot_template, user_template
from sk import my_sk
from dotenv import load_dotenv


# C:\Users\yashs\anaconda3\Lib\site-packages\torch\lib - libiomp5md.dll - in desktop

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

st.set_page_config(layout="wide")

model_name = "google/flan-t5-base"
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
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)
    # create embeddings
    embeddings = SentenceTransformerEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.from_documents(documents=texts, embedding=embeddings)
    return db

# @st.cache_resource
# def llm_pipeline():
#     pipe = pipeline(
#         'text2text-generation',
#         model=base_model,
#         tokenizer=tokenizer,
#         # computation power - max length of token
#         max_length=512,
#         do_sample=True,
#         # sampling
#         temperature=0.5,
#         # randomness and creativeness
#         top_p=0.95
#     )
#     local_llm = HuggingFacePipeline(pipeline=pipe)
#     return local_llm


@st.cache_resource
def qa_llm(_db):
    # llm = llm_pipeline()
    llm = HuggingFaceHub(repo_id="google/flan-t5-xxl",
                         model_kwargs={"temperature":0.5, "max_length":512, "do_sample":True, "top_p":0.95})
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
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)


def display_conversation(history):
    for i in range(len(history["generated"])):
        message(history["past"][i], is_user=True, key=str(i) + "_user")
        message(history["generated"][i], key=str(i))


def main():
    load_dotenv()
    st.markdown("<h1 style='text-align: center; color: grey;'>PDF Retriever - 📄🦜</h1>", unsafe_allow_html=True)
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

    # Search the database for a response based on user input and update session state
    if user_input:
        handle_userinput(user_input)

if __name__ == "__main__":
    main()
