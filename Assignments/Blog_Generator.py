from typing_extensions import TypedDict
from typing import Annotated
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool

import os
from dotenv import load_dotenv
load_dotenv()

os.environ[""]=os.getenv("")
os.environ[""]=os.getenv("")

class State(TypedDict):
    messages:Annotated[list[AnyMessage],add_messages]

llm_model=ChatOpenAI("gpt-4o")


def make_default_graph():
    """Make a Blog Generator Agenti AI Agent"""

    @tool
    def title_creator(topic:str):
        """Create a Title from the Topic"""
        def call_model(state):
            return {"messages": [llm_model.invoke(state["messages"])]}
        pass

    def content_creator():
        pass

