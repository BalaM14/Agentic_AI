import streamlit as st
import os
from dotenv import load_dotenv
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.tools.retriever import create_retriever_tool
from langchain_groq import ChatGroq
from langchain import hub
from langchain.agents import create_openai_tools_agent
from langchain.agents import AgentExecutor


# Load environment variables
load_dotenv()
os.environ['GROQ_API_KEY'] = os.getenv("GROQ_API_KEY")

# Initialize API wrappers
api_wrapper_wiki = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=250)
api_wrapper_arxiv = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=250)

# Define retrieval tools
wiki = WikipediaQueryRun(api_wrapper=api_wrapper_wiki)
arxiv = ArxivQueryRun(api_wrapper=api_wrapper_arxiv)

# Load IMDb data and create a FAISS retriever
loader = WebBaseLoader("https://www.indiatoday.in/")
document = loader.load()
documents = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(documents=document)
vector_db = FAISS.from_documents(documents=documents, embedding=OpenAIEmbeddings())
retriever = vector_db.as_retriever()
india_today = create_retriever_tool(retriever=retriever, name='INDIA TODAY', description='This tool is used to fetch current news about India')

tools = [arxiv, wiki, india_today]

# Load Open Source LLM
llm = ChatGroq(model='Llama3-8b-8192')

# Predefined Prompt
prompt = hub.pull('hwchase17/openai-functions-agent')

# Create Tool Agents
agents = create_openai_tools_agent(llm, tools, prompt)

# Create Agent Executor
agent_executor = AgentExecutor(agent=agents, tools=tools, verbose=True)

# Streamlit UI
st.title("RAG-Powered Multi Agent Assistant")
st.write("Ask me anything about movies, Wikipedia topics, or academic research!")

# User Input
query = st.text_input("Enter your query:")
if st.button("Search") and query:
    with st.spinner("Fetching information..."):
        response = agent_executor.invoke({"input": query})

    st.subheader("Response:")
    st.markdown(f"```{response}```")  # Formats output neatly
