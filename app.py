import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain,create_history_aware_retriever

from langchain_community.document_loaders import PyPDFLoader

from langchain.vectorstores import FAISS
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
import os
from dotenv import load_dotenv

load_dotenv()
groq_api_key=os.getenv("GROQ_API_KEY")
from langchain_huggingface import HuggingFaceEndpointEmbeddings

HF_TOKEN = os.getenv("HF_TOKEN")

# Use the new HuggingFaceEndpointEmbeddings
embeddings = HuggingFaceEndpointEmbeddings(
    model="sentence-transformers/all-MiniLM-L6-v2",  # Use a proper embedding model
    task="feature-extraction",
    huggingfacehub_api_token=HF_TOKEN
)

st.title("conversational RAG with PDF upload and chat history")

st.write("Upload your PDF files and chat with them")

llm=ChatGroq(groq_api_key=groq_api_key,model="openai/gpt-oss-120b",temperature=0.7)

session_id=st.text_input("Enter your session id",value="default")

if'store' not in st.session_state:
    st.session_state.store={}
uploaded_files=st.file_uploader("Upload your PDF files",type=["pdf"],accept_multiple_files=True)
## process the uploaded files
if uploaded_files:
    if uploaded_files:
        documents=[]
        for uploaded_file in uploaded_files:
            temppdf=f"./temp.pdf"
            with open(temppdf,"wb") as file:
                file.write(uploaded_file.getvalue())
                file_name=uploaded_file.name

            loader=PyPDFLoader(temppdf)
            docs=loader.load()
            documents.extend(docs)


    text_splitter=RecursiveCharacterTextSplitter(chunk_size=5000,chunk_overlap=500)
    splits=text_splitter.split_documents(documents)
    vectorstore=FAISS.from_documents(documents=splits,embedding=embeddings)
    retriver=vectorstore.as_retriever()

    contextualize_q_system_prompt=(
        "given a chat history and the latest user question"
        "which may be a follow up question, rephrase the user question"
        "without the chat history.do not answer the question"
        "just rephrase the question"
    )
    contextualize_prompt=ChatPromptTemplate.from_messages(
        [
            ("system",contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human","{input}")
        ]
    )
    history_aware_retriver=create_history_aware_retriever(
        llm,retriver,contextualize_prompt
    )
    system_prompt=(
        "You are a helpful AI assistant that helps people find information"
        "from the provided document context"
        "if you don't know the answer, just say that you don't know and tell that you can answer only from the pdf nothing else dont answer anything else"

        "{context}"
    )
    prompt=ChatPromptTemplate.from_messages(
        [
            ("system",system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human","{input}")
        ]
    )

    question_answer_chain=create_stuff_documents_chain(llm,prompt)
    rag_chain=create_retrieval_chain(
        history_aware_retriver,question_answer_chain
    )

    def get_chat_history(session_id:str)->BaseChatMessageHistory:
        if session_id not in st.session_state.store:
            st.session_state.store[session_id]=ChatMessageHistory()
        return st.session_state.store[session_id]

    conversational_rag=RunnableWithMessageHistory(rag_chain,get_chat_history,input_messages_key="input",history_messages_key="chat_history",output_messages_key="answer")

user_input=st.text_input("Enter your question here")
if user_input:
    session_history=get_chat_history(session_id)
    with st.spinner("Generating response..."):
        response=conversational_rag.invoke(
            {"input":user_input},
            config={"configurable":{"session_id":session_id}
           }
        )

    st.write(response["answer"])



