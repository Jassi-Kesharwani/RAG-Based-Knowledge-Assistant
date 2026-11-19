from dotenv import load_dotenv
load_dotenv()
from langchain_community.document_loaders import PyPDFLoader,PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_groq import ChatGroq
from langchain_community.vectorstores import InMemoryVectorStore
from langchain.agents import create_agent
from langchain.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
import streamlit as st


if "document_uploaded" not in st.session_state:
    st.session_state.document_uploaded = False

if "agent" not in st.session_state:
    st.session_state.agent = None

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "messages" not in st.session_state:
    st.session_state.messages = []

st.title("Rag Agent 🗒️")

def process_document(path):
    loader = PyPDFDirectoryLoader(path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splitted_doc = splitter.split_documents(documents=docs)

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=API_KEY)
    vector_store = InMemoryVectorStore.from_documents(
        documents=splitted_doc,
        embedding=embeddings
    )


    llm = ChatGroq(model="openai/gpt-oss-20b")

    def retrieve_tool(query:str):
        """Retrieve documents relevent to a query from the knowledge base"""
        context = ""
        data = vector_store.similarity_search(query=query, k=3)
        for doc in data:
            context += doc.page_content + "\n\n"

        return context

    SystemPrompt = """"You are a helpful assistant that answers questions using retrieved context.
        My knowledge base consists of the details from the uploaded document
        ALWAYS use the 'retriever_tool' tool for questions requiring external knowledge.
    """

    memory = InMemorySaver()

    agent = create_agent(
        model=llm,
        tools=[retrieve_tool],
        system_prompt=SystemPrompt,
        checkpointer=memory
    )

    st.session_state.agent = agent
    st.session_state.document_uploaded = True

if not st.session_state.document_uploaded:
    uploaded = st.file_uploader(label="Select PDF Files", type=["pdf"], accept_multiple_files=True)
    if uploaded:
        with st.spinner("Processing..."):
            path = "./docs_files/"
            for file in uploaded:
                with open(path + file.name, "wb") as f:
                    f.write(file.getvalue())

            process_document(path)
            st.rerun()


if st.session_state.document_uploaded and st.session_state.agent:
    for message in st.session_state.messages:
        role = message["role"]
        content = message["content"]
        st.chat_message(role).markdown(content)
    query = st.chat_input("Ask anything related to uploaded documents...")
    if query:
        st.session_state.messages.append({"role": "user", "content": query})
        st.chat_message("user").markdown(query)
        res = st.session_state.agent.invoke(
            {"messages": [{"role": "user", "content": query}]},
            {"configurable": {"thread_id": 1}}
        )
        answer = res["messages"][-1].content
        st.chat_message("model").markdown(answer)
        st.session_state.messages.append({"role": "model", "content": answer})
        