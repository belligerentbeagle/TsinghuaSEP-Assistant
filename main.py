# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# This is a simple standalone implementation showing rag pipeline using Nvidia AI Foundational models.
# It uses a simple Streamlit UI and one file implementation of a minimalistic RAG pipeline.

############################################
# Component #1 - Document Loader
############################################

import streamlit as st
import os

st.set_page_config(layout = "wide")

with st.sidebar:
    st.caption("Let Ethan know who's using me today! 🩵")
    name = st.text_input("Enter your name:") or "Anon"
    DOCS_DIR = "./uploaded_docs"
    if not os.path.exists(DOCS_DIR):
        os.makedirs(DOCS_DIR)
    # st.subheader("Add to the Knowledge Base")
    # with st.form("my-form", clear_on_submit=True):
    #     uploaded_files = st.file_uploader("Upload a file to my temporary consciousness:", accept_multiple_files = True)
    #     submitted = st.form_submit_button("Upload!")

    # if uploaded_files and submitted:
    #     for uploaded_file in uploaded_files:
    #         st.success(f"File {uploaded_file.name} uploaded successfully!")
    #         with open(os.path.join(DOCS_DIR, uploaded_file.name),"wb") as f:
    #             f.write(uploaded_file.read())

############################################
# Component #2 - Embedding Model and LLM
############################################

from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings

# make sure to export your NVIDIA AI Playground key as NVIDIA_API_KEY!
# print(os.environ['NVIDIA_API_KEY'])

llm = ChatNVIDIA(model="mixtral_8x7b")
document_embedder = NVIDIAEmbeddings(model="nvolveqa_40k", model_type="passage")
query_embedder = NVIDIAEmbeddings(model="nvolveqa_40k", model_type="query")

############################################
# Component #3 - Vector Database Store
############################################

from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import DirectoryLoader
from langchain.vectorstores import FAISS
import pickle
import requests as rq

with st.sidebar:
    # Option for using an existing vector store
    use_existing_vector_store = "Yes"
    st.caption("Files loaded from: https://github.com/belligerentbeagle/TsinghuaSEP-Assistant/tree/main/uploaded_docs")
    st.warning("If you would like to contribute your SEP instructions/information added to the knowledge base permanently (not deleted on refresh), please tele @contemplativecorgi")

# Path to the vector store file
vector_store_path = "./vectorstore.pkl"

# Load raw documents from the directory
raw_documents = DirectoryLoader(DOCS_DIR).load()


# Check for existing vector store file
vector_store_exists = os.path.exists(vector_store_path)
vectorstore = None
if use_existing_vector_store == "Yes" and vector_store_exists:
    with open(vector_store_path, "rb") as f:
        vectorstore = pickle.load(f)
    with st.sidebar:
        st.success("Existing vector store loaded successfully.")
        
else:
    with st.sidebar:
        if raw_documents:
            with st.spinner("Splitting documents into chunks..."):
                text_splitter = CharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
                documents = text_splitter.split_documents(raw_documents)

            with st.spinner("Adding document chunks to vector database..."):
                vectorstore = FAISS.from_documents(documents, document_embedder)

            with st.spinner("Saving vector store"):
                with open(vector_store_path, "wb") as f:
                    pickle.dump(vectorstore, f)
            st.success("Vector store created and saved.")
        else:
            st.warning("No documents available to process!", icon="⚠️")

############################################
# Component #4 - LLM Response Generation and Chat
############################################

st.subheader("Chat with TsinghuaSEP AI Assistant!")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

prompt_template = ChatPromptTemplate.from_messages(
    [("system", "You are a helpful AI assistant named TsinghuaSEPAssistant created by almighty Ethan Wei. You will reply to questions succinctly only based on the context that you are provided and quote the source of your answer alongside your answer. If something is out of context, you will refrain from replying and politely decline to respond to the user. Also note that Tsinghua university is also known as Tsing hua university or THU."), ("user", "{input}")]
)
user_input = st.chat_input("What are the application steps?")
llm = ChatNVIDIA(model="mixtral_8x7b")

chain = prompt_template | llm | StrOutputParser()

if user_input and vectorstore!=None:
    st.session_state.messages.append({"role": "user", "content": user_input})
    retriever = vectorstore.as_retriever()
    docs = retriever.get_relevant_documents(user_input)
    with st.chat_message("user"):
        st.markdown(user_input)

    context = ""
    for doc in docs:
        context += doc.page_content + "\n\n"

    augmented_user_input = "Context: " + context + "\n\nQuestion: " + user_input + "\n"

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        for response in chain.stream({"input": augmented_user_input}):
            full_response += response
            message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)

        #update logs
        TELEBOT_API = os.environ["TELEBOT_API"]
        rq.get('https://api.telegram.org/bot' + TELEBOT_API + '/sendMessage?chat_id=266679090&text=Name: '
                        + name + " \nQ: " + user_input + "\nA:" + full_response)

    st.session_state.messages.append({"role": "assistant", "content": full_response})
