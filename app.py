
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.llms.huggingface_hub import HuggingFaceHub

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader=PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def split_text_to_chunks(text):
    text_splitter = CharacterTextSplitter(separator="\n", chunk_size=500, chunk_overlap=50, length_function=len)
    return text_splitter.split_text(text)

def get_vector_store(text_chunks):
    embedding = HuggingFaceInstructEmbeddings(model_name = "hkunlp/instructor-xl")
    vector_store = FAISS.from_texts(texts=text_chunks, embedding=embedding)
    return vector_store

def get_conversation_chain(vector_store):
    # llm = ChatOpenAI()
    llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={ "temperature": 0.5, "max_length": 512 })
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(),
        memory=memory
    )
    return conversation_chain

def handle_userInput(user_question):
    response = st.session_state.conversation({ 'question': user_question})
    st.session_state.chat_history = response['chat_history']
    for i,message in enumerate(st.session_state.chat_history):
        if i%2 == 0:
            st.write("User:",message.content)
        else:
            st.write("Bot:",message.content)

def main():
    load_dotenv()
    st.set_page_config(page_title="AI assistant", page_icon=":books")

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Query multiple PDFs")
    user_question = st.text_input("Query your documents:")
    if user_question and st.session_state.conversation:
        handle_userInput(user_question)
    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader("Upload PDFs here:",accept_multiple_files=True)
        if st.button("Upload"):
            with st.spinner("Processing"):
                # get the docs
                raw_text = get_pdf_text(pdf_docs)

                # Split docs to chunks
                chunks = split_text_to_chunks(raw_text)

                # Store chunks with embeddings index\
                embedding = HuggingFaceInstructEmbeddings(model_name = "hkunlp/instructor-xl")
                vector_store = FAISS.from_texts(texts=chunks, embedding=embedding)  

                # create conversation chain
                st.session_state.conversation = get_conversation_chain(vector_store)      
                
   

if __name__ == '__main__':
    main()